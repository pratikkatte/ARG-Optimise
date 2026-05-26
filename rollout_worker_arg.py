import torch
import numpy as np
from env import SimpleTrajectory, Trajectory, action_as_dict


class RolloutWorker:
    """Rollout orchestration for the simplified ARG environment."""

    def __init__(self, env, verbose=False):
        self.env = env
        self.device = env.device
        self.verbose = verbose

    def _rollout_one(
        self,
        generator=None,
        random_spec=None,
        record_diagnostics=False,
        sample_backward=False,
        compute_reward=True,
        generate_full_trajectories=True,
    ):
        if generator is not None:
            data, trajectories = self._rollout_batch(
                generator=generator,
                episodes=1,
                random_spec=random_spec,
                record_diagnostics=record_diagnostics,
                sample_backward=sample_backward,
                compute_reward=compute_reward,
                generate_full_trajectories=generate_full_trajectories,
            )
            return data, trajectories[0]

        state = self.env.get_initial_state()
        if generate_full_trajectories:
            trajectory = Trajectory(state)
        else:
            trajectory = SimpleTrajectory()
        step = 0

        if self.verbose:
            print("Rolling out 1 prior-only trajectory...")

        while not state.is_done:
            step += 1
            sampled = self.env.sample_action_from_prior(state)
            if sampled is None:
                raise RuntimeError("ARG prior rollout reached a non-terminal state with no valid action.")
            action, log_prior = sampled

            next_state = self.env.apply_action(
                state,
                action,
                log_prior,
                compute_reward=compute_reward,
            )
            record = self._trajectory_record(step, action, log_prior, next_state, record_diagnostics)
            if generate_full_trajectories:
                trajectory.update(next_state, action, log_prior, next_state.is_done, record=record)
            else:
                trajectory.update(
                    action,
                    log_prior=log_prior,
                    log_reward=next_state.log_reward,
                    record=record,
                    active_lineages=state.active_lineages,
                )
            state = next_state

        if self.verbose:
            log_reward = state.log_reward
            reward_str = f"{log_reward:.4f}" if log_reward is not None else "None"
            print(
                f"Finished prior-only trajectory: {step} steps, log_reward={reward_str}"
            )

        return state, trajectory

    def _rollout_batch(
        self,
        generator,
        episodes,
        random_spec=None,
        ):
        
        states = [self.env.get_initial_state() for _ in range(episodes)]
        trajectories = [Trajectory(x) for x in states]
        
        
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
            
            input_dict = self.env.prepare_state_rollout_inputs(
                active_states,
                random_spec=random_spec,
            )

            total_log_pf, log_probs, choosen_actions = generator(input_dict)

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
                trajectories[traj_idx].update(next_state, action, log_prior=log_prior, done=next_state.is_done)

                backward_num_parents_by_traj[traj_idx].append(
                    generator.count_backward_parents(next_state)
                    )
            unfinished = [idx for idx, state in enumerate(states) if not state.is_done]

        log_paths_pf = self._pad_log_path_lists(log_paths_pf_by_traj, torch.float32, self.device)

        log_paths_pb = [
            -torch.log(torch.tensor(num_parents, dtype=torch.float32, device=self.device))
            for num_parents in backward_num_parents_by_traj
            ]
        
        log_paths_pb = self._pad_log_path_vectors(log_paths_pb, torch.float32, self.device)

        log_rewards = torch.tensor([state.log_reward for state in states], dtype=torch.float32, device=self.device)


        data = {
            "log_paths_pf": log_paths_pf,
            "log_paths_pb": log_paths_pb,
            "log_rewards": log_rewards,
        }

        return data, trajectories

    def rollout(
        self,
        generator=None,
        episodes=1,
        random_spec=None,
        record_diagnostics=False,
        sample_backward=False,
        num_trajectories=None,
        compute_reward=True,
        generate_full_trajectories=True,
    ):
        """
        Run one or more ARG rollouts.

        Passing a generator lets the model sample over all valid ARG actions.
        Omitting it keeps the prior-only rollout path.
        """
        if num_trajectories is not None:
            episodes = num_trajectories

        if generator is not None:
            return self._rollout_batch(
                generator=generator,
                episodes=episodes,
                random_spec=random_spec,
            )

        states = []
        trajectories = []
        for _ in range(episodes):
            state, trajectory = self._rollout_one(
                record_diagnostics=record_diagnostics,
                compute_reward=compute_reward,
                generate_full_trajectories=generate_full_trajectories,
            )
            states.append(state)
            trajectories.append(trajectory)

        if episodes == 1:
            return states[0], trajectories[0]
        return states, trajectories

    def sample_action_from_prior(self, state):
        return self.env.sample_action_from_prior(state)

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
        sequence_length = int(self.env.sequence_length)
        if len(mask) == sequence_length:
            return mask.to(dtype=torch.float32)

        site_mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)
        for block_idx, has_material in enumerate(mask.tolist()):
            if not has_material:
                continue
            start = int(round(block_idx * sequence_length / self.env.num_blocks))
            end = int(round((block_idx + 1) * sequence_length / self.env.num_blocks))
            site_mask[start:end] = True
        return site_mask.to(dtype=torch.float32)

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
