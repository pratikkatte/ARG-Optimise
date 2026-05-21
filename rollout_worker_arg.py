import math

import numpy as np
import torch


class RolloutWorker:
    """Rollout orchestration for the simplified ARG environment."""

    def __init__(self, env, verbose=False):
        self.env = env
        self.verbose = verbose

    def _rollout_one(
        self,
        generator=None,
        random_spec=None,
        record_diagnostics=False,
        sample_backward=False,
        compute_reward=True,
    ):
        if generator is not None:
            data, trajectories = self._rollout_batch(
                generator=generator,
                episodes=1,
                random_spec=random_spec,
                record_diagnostics=record_diagnostics,
                sample_backward=sample_backward,
                compute_reward=compute_reward,
            )
            return data, trajectories[0]

        state = self.env.get_initial_state()
        trajectory = []
        step = 0

        while not state.is_done:
            step += 1
            sampled = self.env.sample_action_from_prior(state)
            if sampled is None:
                raise RuntimeError("ARG prior rollout reached a non-terminal state with no valid action.")
            action, log_prior = sampled

            state = self.env.apply_action(
                state,
                action,
                log_prior,
                compute_reward=compute_reward,
            )
            trajectory.append(self._trajectory_record(step, action, log_prior, state, record_diagnostics))

            if self.verbose:
                print(
                    "step={step:02d} action={action} log_prior={log_prior:.4f} "
                    "active={active_lineage_count} done={is_done}".format(**trajectory[-1])
                )

        return state, trajectory

    def _rollout_batch(
        self,
        generator,
        episodes,
        random_spec=None,
        record_diagnostics=False,
        sample_backward=False,
        compute_reward=True,
    ):
        states = [self.env.get_initial_state() for _ in range(episodes)]
        trajectories = [[] for _ in range(episodes)]
        log_paths_pf_by_traj = [[] for _ in range(episodes)]
        backward_num_parents_by_traj = [[] for _ in range(episodes)]
        step_counts = [0 for _ in range(episodes)]

        unfinished = [idx for idx, state in enumerate(states) if not state.is_done]
        while unfinished:
            active_states = [states[idx] for idx in unfinished]
            device = self._generator_device(generator)
            batch_nb_seq = torch.tensor(
                [len(state.active_lineages) for state in active_states],
                dtype=torch.long,
                device=device,
            )
            selected_event_types, log_event_probs = self._sample_event_types(
                active_states,
                device=device,
            )
            input_dict = self.env.prepare_state_rollout_inputs(
                active_states,
                input_actions=None,
                random_spec=random_spec,
                batch_nb_seq=batch_nb_seq,
                device=device,
            )
            input_dict["selected_event_types"] = selected_event_types
            input_dict["log_event_probs"] = log_event_probs

            ret = generator(input_dict)
            actions = ret.get("actions", ret.get("arg_actions"))
            if actions is None or len(actions) != len(active_states):
                raise RuntimeError("Generator did not return one ARG action per unfinished batch item.")

            log_paths_pf = ret["log_paths_pf"].reshape(-1)
            if log_paths_pf.numel() != len(active_states):
                raise RuntimeError("Generator forward log probabilities do not match unfinished batch size.")

            for batch_idx, traj_idx in enumerate(unfinished):
                state = states[traj_idx]
                action = dict(actions[batch_idx])
                log_prior = self.env.compute_cwr_event_log_prior(state, action)
                log_paths_pf_by_traj[traj_idx].append(log_paths_pf[batch_idx])

                next_state = self.env.apply_action(
                    state,
                    action,
                    log_prior,
                    compute_reward=compute_reward,
                )
                states[traj_idx] = next_state
                step_counts[traj_idx] += 1
                backward_num_parents_by_traj[traj_idx].append(
                    self._count_inverse_arg_actions(next_state)
                )

                trajectories[traj_idx].append(
                    self._trajectory_record(
                        step_counts[traj_idx],
                        action,
                        log_prior,
                        next_state,
                        record_diagnostics,
                    )
                )

                if self.verbose:
                    print(
                        "traj={traj_idx} step={step:02d} action={action} log_prior={log_prior:.4f} "
                        "active={active_lineage_count} done={is_done}".format(
                            traj_idx=traj_idx,
                            **trajectories[traj_idx][-1],
                        )
                    )

            unfinished = [idx for idx, state in enumerate(states) if not state.is_done]

        for idx, state in enumerate(states):
            if not state.is_done:
                raise RuntimeError(f"ARG rollout ended before batch item {idx} reached a terminal state.")

        dtype, device = self._log_path_dtype_device(log_paths_pf_by_traj)
        log_paths_pf = self._pad_log_path_lists(log_paths_pf_by_traj, dtype, device)

        log_paths_pb_vectors = []
        backward_actions = [[] for _ in states]
        for idx, state in enumerate(states):
            num_parents = backward_num_parents_by_traj[idx]
            if sample_backward:
                backward = generator.sample_backward_from_arg(state)
                backward_actions[idx] = backward["forward_actions"]

            if len(num_parents) != len(log_paths_pf_by_traj[idx]):
                raise ValueError(
                    "Backward parent counts must align with forward log probabilities: "
                    f"{len(num_parents)} != {len(log_paths_pf_by_traj[idx])}"
                )

            if num_parents:
                log_paths_pb_vectors.append(
                    -torch.log(torch.tensor(num_parents, dtype=dtype, device=device))
                )
            else:
                log_paths_pb_vectors.append(torch.empty(0, dtype=dtype, device=device))

        log_paths_pb = self._pad_log_path_vectors(log_paths_pb_vectors, dtype, device)
        log_rewards = torch.tensor(
            [
                float(state.log_reward)
                if state.log_reward is not None
                else float("nan")
                for state in states
            ],
            dtype=dtype,
            device=device,
        )

        return {
            "log_paths_pf": log_paths_pf,
            "log_paths_pb": log_paths_pb,
            "log_rewards": log_rewards,
            "states": states,
            "backward_actions": backward_actions,
            "backward_num_parents": backward_num_parents_by_traj,
        }, trajectories

    def rollout(
        self,
        generator=None,
        episodes=1,
        random_spec=None,
        record_diagnostics=False,
        sample_backward=False,
        num_trajectories=None,
        compute_reward=True,
    ):
        """
        Run one or more ARG rollouts.

        Passing a generator uses prior event-type sampling plus model action
        sampling. Omitting it keeps the prior-only rollout path.
        """
        if num_trajectories is not None:
            episodes = num_trajectories

        if generator is not None:
            return self._rollout_batch(
                generator=generator,
                episodes=episodes,
                random_spec=random_spec,
                record_diagnostics=record_diagnostics,
                sample_backward=sample_backward,
                compute_reward=compute_reward,
            )

        states = []
        trajectories = []
        for _ in range(episodes):
            state, trajectory = self._rollout_one(
                record_diagnostics=record_diagnostics,
                compute_reward=compute_reward,
            )
            states.append(state)
            trajectories.append(trajectory)

        if episodes == 1:
            return states[0], trajectories[0]
        return states, trajectories

    def sample_action_from_prior(self, state):
        return self.env.sample_action_from_prior(state)

    def _sample_event_types(self, states, device=None):
        selected_event_types = []
        log_event_probs = []
        for state in states:
            probs = self.env.compute_event_probabilities(state)
            coal_prob = float(probs.get("coal", 0.0))
            recomb_prob = float(probs.get("recomb", 0.0))
            if coal_prob <= 0.0 and recomb_prob <= 0.0:
                raise RuntimeError("ARG rollout reached a non-terminal state with no valid event type.")

            event_types = ["coal", "recomb"]
            event_probs = [coal_prob, recomb_prob]
            idx = np.random.choice(len(event_types), p=event_probs)
            event_type = event_types[idx]
            event_prob = event_probs[idx]

            selected_event_types.append(event_type)
            log_event_probs.append(math.log(event_prob))
        
        return selected_event_types, torch.tensor(
            log_event_probs,
            dtype=torch.float32,
            device=device,
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
        seq_arrays = self.env.seq_arrays.float()
        if device is not None:
            seq_arrays = seq_arrays.to(device)
        lineage_features = []

        for lineage in state.active_lineages:
            if lineage.sequences_indices:
                feature = seq_arrays[lineage.sequences_indices].mean(dim=0)
            else:
                feature = torch.zeros_like(seq_arrays[0])
            site_mask = self._material_mask_to_site_mask(lineage.material_mask, feature.device)
            lineage_features.append(feature * site_mask[:, None])

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
            "action": action,
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

    def _count_inverse_arg_actions(self, state):
        count = 0
        for lineage in state.active_lineages:
            if lineage.event_type != "coal" or len(lineage.children) != 2:
                continue
            child_i, child_j = lineage.children
            if (
                child_i in state.all_nodes
                and child_j in state.all_nodes
                and lineage.node_id in state.all_nodes[child_i].parents
                and lineage.node_id in state.all_nodes[child_j].parents
            ):
                count += 1

        recomb_by_event = {}
        for active_idx, lineage in enumerate(state.active_lineages):
            if (
                lineage.event_type != "recomb"
                or len(lineage.children) != 1
                or lineage.breakpoint is None
                or lineage.recombination_side not in ("left", "right")
            ):
                continue
            key = (lineage.children[0], lineage.breakpoint)
            recomb_by_event.setdefault(key, {})[lineage.recombination_side] = (
                active_idx,
                lineage.node_id,
            )

        for (child_id, _), sides in recomb_by_event.items():
            if "left" not in sides or "right" not in sides or child_id not in state.all_nodes:
                continue
            _, left_id = sides["left"]
            _, right_id = sides["right"]
            child = state.all_nodes[child_id]
            left_parent = state.all_nodes[left_id]
            right_parent = state.all_nodes[right_id]
            if set(child.parents) != {left_id, right_id}:
                continue
            if left_parent.material_segments.intersection_count(right_parent.material_segments) > 0:
                continue
            if left_parent.material_segments.union(right_parent.material_segments) != child.material_segments:
                continue
            count += 1

        if count <= 0:
            raise ValueError("No valid ARG parent states were found for backward probability.")
        return count
