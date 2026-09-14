import torch
import numpy as np
from env import SimpleTrajectory, action_as_dict, CoalescenceChoice, RecombinationChoice
from time_env import validate_temperature


class RolloutWorker:
    """Rollout orchestration for the simplified ARG environment."""

    def __init__(self, env, verbose=False):
        self.env = env
        self.device = env.device
        self.verbose = verbose

    def _rollout_batch(
        self,
        generator,
        episodes,
        random_spec=None,
        return_states=False,
        collect_flows=False,
        fixed_actions=None,
        ):
        
        if collect_flows and generator.loss_type != "subtb":
            raise ValueError("Flow collection requires loss_type=subtb")
        continuous = self.env.time_policy == "cwr_exponential"
        score_dtype = torch.float64 if continuous else torch.float32
        if continuous:
            validate_temperature(random_spec)
        flows_by_traj = [[] for _ in range(episodes)] if collect_flows else None
        corrections_by_traj = [[] for _ in range(episodes)] if collect_flows else None
        states = [self.env.get_initial_state(track_likelihood=collect_flows) for _ in range(episodes)]
        trajectories = [SimpleTrajectory() for _ in states]
        
        
        log_paths_pf_by_traj = [[] for _ in range(episodes)]
        backward_num_parents_by_traj = [[] for _ in range(episodes)]
        
        if self.verbose:
            print(
                f"Rolling out {episodes} trajectory/trajectories in batch "
                f"({len([idx for idx, state in enumerate(states) if not state.is_done])} active)..."
            )

        unfinished = [idx for idx, state in enumerate(states) if not state.is_done]

        while unfinished:
            active_states = [states[idx] for idx in unfinished]
            forced = None
            if fixed_actions is not None:
                forced = []
                for idx in unfinished:
                    step = len(trajectories[idx])
                    if step >= len(fixed_actions[idx]):
                        raise ValueError("Replay trajectory ends before reaching a terminal state")
                    record = fixed_actions[idx][step]
                    action = CoalescenceChoice.from_action(record) or RecombinationChoice.from_action(record)
                    if action is None:
                        raise ValueError("Invalid replay action")
                    if isinstance(action, RecombinationChoice) and action.breakpoint is None:
                        raise ValueError("Replay recombination is missing its breakpoint")
                    if (continuous and action.delta_t is None) or (not continuous and action.time_action is None):
                        raise ValueError("Replay action is missing its waiting time")
                    forced.append(action)
            
            input_dict = self.env.prepare_state_rollout_inputs(
                active_states,
                random_spec=random_spec,
                event_policy=generator.arg_model.event_policy,
            )

            if collect_flows:
                total_log_pf, log_probs, choosen_actions, state_flows = generator(
                    input_dict, return_flows=True, **({'forced_actions': forced} if forced is not None else {}))
                for batch_idx, traj_idx in enumerate(unfinished):
                    flows_by_traj[traj_idx].append(state_flows[batch_idx])
                    corrections_by_traj[traj_idx].append(generator._last_flow_corrections[batch_idx])
            else:
                total_log_pf, log_probs, choosen_actions = generator(
                    input_dict, **({'forced_actions': forced} if forced is not None else {}))

            for batch_idx, traj_idx in enumerate(unfinished):
                state = states[traj_idx]
                coal_actions, recomb_actions = self.env.enumerate_actions(state)

                action = choosen_actions[batch_idx]
                log_paths_pf_by_traj[traj_idx].append(total_log_pf[batch_idx])
                log_prior = self.env.compute_cwr_event_log_prior(state, (coal_actions, recomb_actions), action)

                next_state = self.env.apply_action(
                    state,
                    action,
                    log_prior=log_prior,
                )
                states[traj_idx] = next_state
                trajectories[traj_idx].update(
                    action,
                    log_prior=log_prior,
                    log_reward=next_state.log_reward,
                )

                backward_num_parents_by_traj[traj_idx].append(
                    generator.count_backward_parents(next_state)
                    )
            unfinished = [idx for idx, state in enumerate(states) if not state.is_done]

        if fixed_actions is not None and any(len(traj) != len(actions) for traj, actions in zip(trajectories, fixed_actions)):
            raise ValueError("Replay trajectory contains actions after its terminal state")

        log_paths_pf = self._pad_log_path_lists(log_paths_pf_by_traj, score_dtype, self.device)

        log_paths_pb = [
            -torch.log(torch.tensor(num_parents, dtype=score_dtype, device=self.device))
            for num_parents in backward_num_parents_by_traj
            ]
        
        log_paths_pb = self._pad_log_path_vectors(log_paths_pb, score_dtype, self.device)

        log_rewards = torch.tensor([state.log_reward for state in states], dtype=score_dtype, device=self.device)
        if continuous and not all(bool(torch.isfinite(x).all()) for x in (
            log_paths_pf, log_paths_pb, log_rewards, log_paths_pf.sum(-1), log_paths_pb.sum(-1)
        )):
            raise ValueError("non-finite continuous rollout scores")


        data = {
            "log_paths_pf": log_paths_pf,
            "log_paths_pb": log_paths_pb,
            "log_rewards": log_rewards,
        }
        if collect_flows:
            # Keep terminal posterior precision; the legacy TB outputs stay float32.
            data["log_rewards"] = torch.tensor([s.log_reward for s in states], dtype=torch.float64, device=self.device)
            data["lengths"] = torch.tensor([len(p) for p in log_paths_pf_by_traj], dtype=torch.long, device=self.device)
            for idx, values in enumerate(flows_by_traj):
                values.append(data["log_rewards"][idx] if values else generator.compute_log_Z().double())
            data["state_flows"] = self._pad_log_path_lists(flows_by_traj, torch.float64, self.device)
            data["flow_corrections"] = self._pad_log_path_lists(corrections_by_traj, torch.float64, self.device)
        if return_states:
            data["states"] = states

        return data, trajectories

    def rollout(
        self,
        generator=None,
        episodes=1,
        random_spec=None,
        return_states=False,
        collect_flows=False,
    ):
        """Run one or more model-guided ARG rollouts."""
        if generator is None:
            raise ValueError("Generator is required for rollout")
        return self._rollout_batch(
            generator=generator,
            episodes=episodes,
            random_spec=random_spec,
            return_states=return_states,
            collect_flows=collect_flows,
        )

    def replay(self, generator, trajectories, collect_flows=True, return_states=False):
        """Rescore fixed complete paths with the current policy and exact rewards.

        Only action records are reused; every PF, PB, and state flow is recomputed.
        The caller controls eval mode/no_grad, just as for ordinary rollout.
        """
        if generator.arg_model.event_policy != 'cwr_residual':
            raise ValueError('Replay currently requires event_policy=cwr_residual')
        actions = [traj.actions if hasattr(traj, 'actions') else traj for traj in trajectories]
        if not actions:
            raise ValueError('Replay requires at least one trajectory')
        return self._rollout_batch(generator, len(actions), collect_flows=collect_flows,
                                   return_states=return_states, fixed_actions=actions)

    def _states_to_padded_tree_features(self, states, device=None):
        lineage_features = [
            self._state_to_lineage_features(state, device=device)
            for state in states
        ]
        max_active = max(features.shape[0] for features in lineage_features)
        batch_size = len(lineage_features)
        _, sequence_length, channels = lineage_features[0].shape
        batch_features = lineage_features[0].new_zeros(
            batch_size,
            max_active,
            sequence_length,
            channels,
        )
        batch_nb_seq = torch.empty(batch_size, dtype=torch.long, device=batch_features.device)

        for batch_idx, features in enumerate(lineage_features):
            active_count = features.shape[0]
            batch_features[batch_idx, :active_count] = features
            batch_nb_seq[batch_idx] = active_count

        return batch_features, batch_nb_seq

    def _pad_log_path_lists(self, log_path_lists, dtype, device):
        vectors = [
            torch.stack(log_paths).to(dtype=dtype, device=device)
            if log_paths
            else torch.empty(0, dtype=dtype, device=device)
            for log_paths in log_path_lists
        ]
        return self._pad_log_path_vectors(vectors, dtype, device)

    def _pad_log_path_vectors(self, vectors, dtype, device):
        max_length = max((vector.numel() for vector in vectors), default=0)
        padded = torch.zeros(len(vectors), max_length, dtype=dtype, device=device)
        for row_idx, vector in enumerate(vectors):
            if vector.numel() > 0:
                padded[row_idx, :vector.numel()] = vector.to(dtype=dtype, device=device)
        return padded

    def _log_path_dtype_device(self, log_path_lists):
        for log_paths in log_path_lists:
            if log_paths:
                return log_paths[0].dtype, log_paths[0].device
        seq_arrays = self.env.seq_arrays
        device = seq_arrays.device if hasattr(seq_arrays, "device") else torch.device("cpu")
        return torch.float32, device

    def _state_to_lineage_features(self, state, device=None):
        lineage_features = []

        for lineage in state.active_lineages:
            if lineage.partials is None:
                raise ValueError(
                    f"Active ARG lineage {lineage.node_id} is missing partials"
                )
            feature = lineage.partials
            if not torch.is_tensor(feature):
                feature = torch.as_tensor(feature, dtype=torch.float32)
            feature = feature.float()
            if device is not None:
                feature = feature.to(device)
            feature = self.env.evolution_model.mask_partials(
                feature,
                lineage.material_segments,
            )
            lineage_features.append(self.env.evolution_model.normalize_partials(feature))

        if not lineage_features:
            raise ValueError("Cannot prepare rollout features for a state with no active lineages.")
        return torch.stack(lineage_features, dim=0)

    def _state_to_tree_features(self, state):
        return self._state_to_lineage_features(state).unsqueeze(0)

    def _material_mask_to_site_mask(self, material_mask, device):
        mask = torch.as_tensor(material_mask, dtype=torch.bool, device=device)
        num_blocks = int(self.env.num_blocks)
        if len(mask) == num_blocks:
            return mask.to(dtype=torch.float32)
        raise ValueError(
            f"material mask must have length {num_blocks}, got {len(mask)}"
        )

    def _trajectory_record(self, step, action, log_prior, state, record_diagnostics):
        record = {
            "step": step,
            "action": action_as_dict(action),
            "log_prior": log_prior,
            "active_lineage_count": len(state.active_lineages),
            "is_done": state.is_done,
            "log_reward": state.log_reward,
        }
        if record_diagnostics:
            record["active_counts"] = self.env.get_active_counts(state).tolist()
        return record

    def _generator_device(self, generator):
        device = getattr(generator, "device", None)
        if device is not None:
            return torch.device(device)
        try:
            return next(generator.parameters()).device
        except (AttributeError, StopIteration):
            return self.env.seq_arrays.device
