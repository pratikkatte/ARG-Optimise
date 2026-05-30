import torch
import numpy as np
from env import SimpleTrajectory, action_as_dict

import random

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
        base_state=None,
        log_pfs=None,
        backward_num_parents_by_traj=None,
        random_spec=None,
        return_states=False,
        window_start=0,
        window_end=None,
        ):
        return generator({
            "mode": "rollout_batch",
            "episodes": episodes,
            "base_state": base_state,
            "log_pfs": log_pfs,
            "backward_num_parents_by_traj": backward_num_parents_by_traj,
            "random_spec": random_spec,
            "return_states": return_states,
            "window_start": window_start,
            "window_end": window_end,
        })

    def _build_refine_candidate(self, terminal_state, window_start, window_end, generator):
        base_state = self.env.delete_genomic_window(terminal_state, window_start, window_end)
        prefix_states, prefix_actions = self.env.reconstruct_prefix_trajectory(base_state)
        backward_num_parents_by_traj = []

        log_pfs = list(generator({
            "mode": "score_actions",
            "states": prefix_states,
            "actions": prefix_actions,
            "window_start": window_start,
            "window_end": window_end,
        }).unbind(0))

        for t in range(len(prefix_actions)):
            s_next = prefix_states[t + 1] if t + 1 < len(prefix_states) else base_state

            num_parents = self.env.count_backward_parents(s_next)
            backward_num_parents_by_traj.append(num_parents)

        return base_state, log_pfs, backward_num_parents_by_traj

    def rollout(
        self,
        terminal_state=None,
        generator=None,
        episodes=1,
        random_spec=None,
        return_states=False,
        window_start=0,
        window_end=None,
        window_ranges=None,
    ):
        """Run one or more model-guided ARG rollouts."""

        if generator is None:
            raise ValueError("Generator is required for rollout")

        log_pfs = []
        backward_num_parents_by_traj = []

        if terminal_state is not None:
            if window_ranges is None and window_start is not None and window_end is not None:
                window_ranges = [(window_start, window_end)]

            if window_ranges is not None:
                if len(window_ranges) == 0:
                    raise ValueError("window_ranges must contain at least one window.")

                candidate_states = []
                candidate_log_pfs = []
                candidate_backward_num_parents = []
                for start, end in window_ranges:
                    base_state, prefix_log_pfs, prefix_backward_num_parents = (
                        self._build_refine_candidate(terminal_state, start, end, generator)
                    )
                    candidate_states.append(base_state)
                    candidate_log_pfs.append(prefix_log_pfs)
                    candidate_backward_num_parents.append(prefix_backward_num_parents)

                if len(candidate_states) == 1:
                    base_state = candidate_states[0]
                    log_pfs = candidate_log_pfs[0]
                    backward_num_parents_by_traj = candidate_backward_num_parents[0]
                    window_start = window_ranges[0][0]
                    window_end = window_ranges[0][1]
                else:
                    base_state = candidate_states
                    log_pfs = candidate_log_pfs
                    backward_num_parents_by_traj = candidate_backward_num_parents
                    window_start = [r[0] for r in window_ranges]
                    window_end = [r[1] for r in window_ranges]
            else:
                base_state = self.env.get_initial_state()

        else:
            base_state = self.env.get_initial_state()

        return self._rollout_batch(
            generator=generator,
            base_state=base_state,
            log_pfs=log_pfs,
            backward_num_parents_by_traj=backward_num_parents_by_traj,
            episodes=episodes,
            random_spec=random_spec,
            return_states=return_states,
            window_start=window_start,
            window_end=window_end,
        )
        

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
