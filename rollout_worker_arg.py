import torch
import numpy as np
from env import SimpleTrajectory, action_as_dict
from refinement import clone_start_state, move_state_partials_to_device


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
        start_states=None,
        action_filter=None,
        max_steps=None,
        ):
        if max_steps is not None:
            max_steps = int(max_steps)
            if max_steps < 1:
                raise ValueError("max_steps must be at least 1 when provided")
        
        states = self._initial_rollout_states(episodes, start_states)
        trajectories = [SimpleTrajectory() for _ in states]
        trajectory_states = [[state] for state in states]
        
        
        log_paths_pf_by_traj = [[] for _ in range(episodes)]
        backward_num_parents_by_traj = [[] for _ in range(episodes)]
        
        if self.verbose:
            print(
                f"Rolling out {episodes} trajectory/trajectories in batch "
                f"({len([idx for idx, state in enumerate(states) if not state.is_done])} active)..."
            )

        unfinished = self._unfinished_indices(states, trajectory_states, max_steps)

        while unfinished:
            active_states = [states[idx] for idx in unfinished]
            
            input_dict = self.env.prepare_state_rollout_inputs(
                active_states,
                random_spec=random_spec,
                action_filter=action_filter,
            )

            total_log_pf, log_probs, choosen_actions = generator(input_dict)

            for batch_idx, traj_idx in enumerate(unfinished):
                state = states[traj_idx]
                coal_actions, recomb_actions = self.env.enumerate_actions(
                    state,
                    action_filter=action_filter,
                )

                action = choosen_actions[batch_idx]
                if action_filter is not None and hasattr(action_filter, "action_touches_blocks"):
                    if not action_filter.action_touches_blocks(state, action):
                        raise ValueError(
                            "sampled action does not touch the local refinement region"
                        )
                log_paths_pf_by_traj[traj_idx].append(total_log_pf[batch_idx])
                log_prior = self.env.compute_cwr_event_log_prior(state, (coal_actions, recomb_actions), action)

                next_state = self.env.apply_action(
                    state,
                    action,
                    log_prior=log_prior,
                )
                states[traj_idx] = next_state
                trajectory_states[traj_idx].append(next_state)
                trajectories[traj_idx].update(
                    action,
                    log_prior=log_prior,
                    log_reward=next_state.log_reward,
                )

                backward_num_parents_by_traj[traj_idx].append(
                    generator.count_backward_parents(next_state)
                    )
            unfinished = self._unfinished_indices(states, trajectory_states, max_steps)

        log_paths_pf = self._pad_log_path_lists(log_paths_pf_by_traj, torch.float32, self.device)

        log_paths_pb = [
            -torch.log(torch.tensor(num_parents, dtype=torch.float32, device=self.device))
            for num_parents in backward_num_parents_by_traj
            ]
        
        log_paths_pb = self._pad_log_path_vectors(log_paths_pb, torch.float32, self.device)

        terminal_mask = torch.tensor(
            [bool(state.is_done) for state in states],
            dtype=torch.bool,
            device=self.device,
        )
        truncated_mask = torch.tensor(
            [
                (
                    max_steps is not None
                    and not bool(state.is_done)
                    and (len(path) - 1) >= max_steps
                )
                for state, path in zip(states, trajectory_states)
            ],
            dtype=torch.bool,
            device=self.device,
        )
        log_rewards = torch.tensor(
            [
                float(state.log_reward)
                if state.is_done and state.log_reward is not None
                else float("nan")
                for state in states
            ],
            dtype=torch.float32,
            device=self.device,
        )
        trajectory_lengths = torch.tensor(
            [len(path) - 1 for path in trajectory_states],
            dtype=torch.long,
            device=self.device,
        )


        data = {
            "log_paths_pf": log_paths_pf,
            "log_paths_pb": log_paths_pb,
            "log_rewards": log_rewards,
            "trajectory_states": trajectory_states,
            "trajectory_lengths": trajectory_lengths,
            "terminal_mask": terminal_mask,
            "truncated_mask": truncated_mask,
        }
        if return_states:
            data["states"] = states

        return data, trajectories

    def rollout(
        self,
        generator=None,
        episodes=1,
        random_spec=None,
        return_states=False,
        start_states=None,
        action_filter=None,
        max_steps=None,
    ):
        """Run one or more model-guided ARG rollouts."""
        if generator is None:
            raise ValueError("Generator is required for rollout")
        return self._rollout_batch(
            generator=generator,
            episodes=episodes,
            random_spec=random_spec,
            return_states=return_states,
            start_states=start_states,
            action_filter=action_filter,
            max_steps=max_steps,
        )

    def _unfinished_indices(self, states, trajectory_states, max_steps):
        unfinished = []
        for idx, state in enumerate(states):
            if state.is_done:
                continue
            if max_steps is not None and (len(trajectory_states[idx]) - 1) >= max_steps:
                continue
            unfinished.append(idx)
        return unfinished

    def _initial_rollout_states(self, episodes, start_states):
        episodes = int(episodes)
        if episodes < 1:
            raise ValueError("episodes must be at least 1")
        if start_states is None:
            return [self.env.get_initial_state() for _ in range(episodes)]
        if len(start_states) != episodes:
            raise ValueError(
                f"start_states length ({len(start_states)}) must match episodes ({episodes})"
            )
        return [
            move_state_partials_to_device(clone_start_state(state), self.env.device)
            for state in start_states
        ]

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
        seq_arrays = getattr(self.env, "seq_arrays", self.env.block_seq_arrays)
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
            return getattr(self.env, "seq_arrays", self.env.block_seq_arrays).device
