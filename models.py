import itertools

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ARGModel(nn.Module):
    """One-step ARG action policy.

    The model scores candidate coalescent and recombination actions. When the
    caller provides current ARG states, candidates are read from the environment
    so material-mask constraints are respected.
    """

    EVENT_TO_IDX = {"coal": 0, "recomb": 1}
    SCALAR_FEATURES = 8

    def __init__(self, env, cfg=None):
        super().__init__()
        self.env = env
        embedding_size = 32
        hidden_size = 64
        dropout = 0.0

        input_size = int(env.sequence_length) * 4

        self.seq_embedding = nn.Linear(input_size, embedding_size) ## Understand the size
        self.event_embedding = nn.Embedding(len(self.EVENT_TO_IDX), embedding_size)
        self.scalar_embedding = nn.Linear(self.SCALAR_FEATURES, embedding_size)
        self.action_scorer = nn.Sequential(
            nn.Linear(embedding_size * 6, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def model_params(self):
        return list(self.parameters())

    def forward(self, input_dict):
        batch_input = input_dict["batch_input"].float()
        batch_nb_seq = input_dict["batch_nb_seq"]
        batch_size, max_nb_seq, input_size = batch_input.shape
        device = batch_input.device

        if input_size != self.seq_embedding.in_features:
            raise ValueError(
                "batch_input last dimension must match env.sequence_length * 4 "
                f"({self.seq_embedding.in_features}), got {input_size}"
            )

        lineage_reps = self.seq_embedding(batch_input)
        valid_mask = torch.arange(max_nb_seq, device=device)[None, :] < batch_nb_seq[:, None]
        lineage_reps = lineage_reps * valid_mask.unsqueeze(-1)
        summary_reps = lineage_reps.sum(dim=1) / batch_nb_seq.clamp_min(1).unsqueeze(-1)

        seq_features = self._get_seq_features(input_dict, batch_input)
        states = input_dict.get("states")
        action_options = input_dict.get("action_options")
        selected_event_types = input_dict.get("selected_event_types")
        if selected_event_types is not None and len(selected_event_types) != batch_size:
            raise ValueError("selected_event_types must have one entry per batch item.")
        candidate_actions = [
            self._candidate_actions_for_batch_item(
                input_dict,
                batch_idx,
                states,
                action_options,
                selected_event_types,
            )
            for batch_idx in range(batch_size)
        ]

        if any(len(actions) == 0 for actions in candidate_actions):
            raise ValueError("ARGModel.forward received a batch item with no candidate actions.")

        logits = self._score_candidates(
            candidate_actions,
            lineage_reps,
            summary_reps,
            seq_features,
            states,
            batch_nb_seq,
        )
        mask = torch.isneginf(logits)

        input_actions = input_dict.get("input_actions")
        if input_actions is None:
            action_indices = self.sample({"logits": logits, "candidate_actions": candidate_actions},
                                         input_dict.get("random_spec"))
            actions = [
                dict(candidate_actions[batch_idx][action_indices[batch_idx].item()])
                for batch_idx in range(batch_size)
            ]
        else:
            if len(input_actions) != batch_size:
                raise ValueError("input_actions length must match batch size.")
            action_indices = self._indices_for_input_actions(input_actions, candidate_actions, device)
            actions = [dict(action) for action in input_actions]

        conditional_log_paths_pf = self.compute_log_path_pf({"logits": logits}, action_indices)
        log_event_probs = self._event_log_probs_for_batch(
            input_dict,
            batch_size,
            conditional_log_paths_pf,
        )
        log_paths_pf = log_event_probs + conditional_log_paths_pf
        return {
            "actions": actions,
            "arg_actions": actions,
            "action_indices": action_indices,
            "candidate_actions": candidate_actions,
            "logits": logits,
            "mask": mask,
            "log_paths_pf": log_paths_pf,
            "conditional_log_paths_pf": conditional_log_paths_pf,
            "log_event_probs": log_event_probs,
        }

    def sample(self, ret, random_spec):
        logits = ret["logits"]
        candidate_actions = ret["candidate_actions"]
        if random_spec is None:
            random_spec = {"random_action_prob": 0.0}

        if "random_action_prob" in random_spec:
            action_indices = Categorical(logits=logits).sample()
            random_p = random_spec["random_action_prob"]
            if random_p > 0:
                batch_size = logits.shape[0]
                rand_flag = torch.empty(batch_size, device=logits.device).uniform_(0, 1) <= random_p
                for batch_idx in torch.nonzero(rand_flag, as_tuple=False).reshape(-1).tolist():
                    valid_count = len(candidate_actions[batch_idx])
                    action_indices[batch_idx] = torch.randint(
                        valid_count,
                        size=(),
                        device=logits.device,
                    )
            return action_indices

        temperature = random_spec["T"]
        return Categorical(logits=logits / temperature).sample()

    def compute_log_path_pf(self, ret, action_indices):
        logits = ret["logits"]
        batch_idx = torch.arange(logits.shape[0], device=logits.device)
        log_p = self.logsoftmax(logits)
        return log_p[batch_idx, action_indices]

    def _score_candidates(
        self,
        candidate_actions,
        lineage_reps,
        summary_reps,
        seq_features,
        states,
        batch_nb_seq,
    ):
        batch_size = len(candidate_actions)
        max_candidates = max(len(actions) for actions in candidate_actions)
        logits = lineage_reps.new_full((batch_size, max_candidates), float("-inf"))

        for batch_idx, actions in enumerate(candidate_actions):
            state = states[batch_idx] if states is not None else None
            features = self._batched_action_features(
                actions,
                batch_idx,
                lineage_reps,
                summary_reps,
                seq_features,
                state,
                int(batch_nb_seq[batch_idx].item()),
            )
            logits[batch_idx, :len(actions)] = self.action_scorer(features).squeeze(-1)

        return logits

    def _batched_action_features(
        self,
        actions,
        batch_idx,
        lineage_reps,
        summary_reps,
        seq_features,
        state,
        nb_seq,
    ):
        device = lineage_reps.device
        num_actions = len(actions)
        embedding_size = lineage_reps.shape[-1]

        event_indices = torch.tensor(
            [self.EVENT_TO_IDX[action["event_type"]] for action in actions],
            dtype=torch.long,
            device=device,
        )
        event_rep = self.event_embedding(event_indices)
        scalar_rep = self.scalar_embedding(
            self._batched_scalar_features(actions, state, nb_seq, device)
        )

        primary_rep = lineage_reps.new_zeros(num_actions, embedding_size)
        secondary_rep = lineage_reps.new_zeros(num_actions, embedding_size)
        tertiary_rep = lineage_reps.new_zeros(num_actions, embedding_size)

        coal_rows = [
            (row_idx, action["active_lineage_i"], action["active_lineage_j"])
            for row_idx, action in enumerate(actions)
            if action["event_type"] == "coal"
        ]
        if coal_rows:
            rows, left_indices, right_indices = zip(*coal_rows)
            rows = torch.tensor(rows, dtype=torch.long, device=device)
            left_indices = torch.tensor(left_indices, dtype=torch.long, device=device)
            right_indices = torch.tensor(right_indices, dtype=torch.long, device=device)
            left_rep = lineage_reps[batch_idx, left_indices]
            right_rep = lineage_reps[batch_idx, right_indices]
            primary_rep[rows] = left_rep + right_rep
            secondary_rep[rows] = torch.abs(left_rep - right_rep)
            tertiary_rep[rows] = left_rep * right_rep

        recomb_rows = [
            (row_idx, action["active_lineage_i"], action["breakpoint"])
            for row_idx, action in enumerate(actions)
            if action["event_type"] == "recomb"
        ]
        if recomb_rows:
            rows, lineage_indices, breakpoints = zip(*recomb_rows)
            rows = torch.tensor(rows, dtype=torch.long, device=device)
            lineage_indices = torch.tensor(lineage_indices, dtype=torch.long, device=device)
            breakpoints = torch.tensor(breakpoints, dtype=torch.long, device=device)
            primary_rep[rows] = lineage_reps[batch_idx, lineage_indices]
            left_rep, right_rep = self._batched_split_sequence_reps(
                seq_features[batch_idx],
                lineage_indices,
                breakpoints,
                nb_seq,
            )
            secondary_rep[rows] = left_rep
            tertiary_rep[rows] = right_rep

        summary_for_actions = summary_reps[batch_idx].expand(num_actions, -1)
        return torch.cat(
            [
                primary_rep,
                secondary_rep,
                tertiary_rep,
                summary_for_actions,
                event_rep,
                scalar_rep,
            ],
            dim=-1,
        )

    def _batched_split_sequence_reps(self, batch_seq_features, lineage_indices, breakpoints, nb_seq):
        seq_len = batch_seq_features.shape[1]
        weight = self.seq_embedding.weight.view(self.seq_embedding.out_features, seq_len, 4)
        lineage_features = batch_seq_features[:nb_seq]
        site_contrib = torch.einsum("nlc,elc->nle", lineage_features, weight)
        prefix = torch.cat(
            [
                site_contrib.new_zeros(site_contrib.shape[0], 1, site_contrib.shape[-1]),
                torch.cumsum(site_contrib, dim=1),
            ],
            dim=1,
        )
        site_breakpoints = self._site_breakpoints_tensor(breakpoints, seq_len)
        left_without_bias = prefix[lineage_indices, site_breakpoints]
        total_without_bias = prefix[lineage_indices, -1]
        bias = self.seq_embedding.bias
        return left_without_bias + bias, total_without_bias - left_without_bias + bias

    def _site_breakpoints_tensor(self, breakpoints, seq_len):
        if self.env.num_blocks == seq_len:
            return breakpoints.to(dtype=torch.long)
        scaled = torch.round(
            breakpoints.to(dtype=torch.float32)
            * float(seq_len)
            / float(self.env.num_blocks)
        ).to(dtype=torch.long)
        return scaled.clamp(1, seq_len - 1)

    def _batched_scalar_features(self, actions, state, nb_seq, device):
        denom_seq = float(max(nb_seq - 1, 1))
        denom_blocks = float(max(self.env.num_blocks, 1))
        event_is_recomb = torch.tensor(
            [1.0 if action["event_type"] == "recomb" else 0.0 for action in actions],
            dtype=torch.float32,
            device=device,
        )
        active_i = torch.tensor(
            [action.get("active_lineage_i", 0) for action in actions],
            dtype=torch.long,
            device=device,
        )
        active_j = torch.tensor(
            [action.get("active_lineage_j", 0) for action in actions],
            dtype=torch.long,
            device=device,
        )
        breakpoints = torch.tensor(
            [action.get("breakpoint", 0) for action in actions],
            dtype=torch.long,
            device=device,
        )

        material_fraction = torch.ones_like(event_is_recomb)
        overlap_fraction = 1.0 - event_is_recomb
        left_fraction = torch.ones_like(event_is_recomb)
        right_fraction = torch.ones_like(event_is_recomb)

        if state is not None:
            masks = torch.as_tensor(
                np.stack([lineage.material_mask for lineage in state.active_lineages]),
                dtype=torch.bool,
                device=device,
            )
            material_counts = masks.sum(dim=1).to(dtype=torch.float32)
            coal_rows = torch.nonzero(event_is_recomb == 0, as_tuple=False).reshape(-1)
            if coal_rows.numel() > 0:
                left_masks = masks[active_i[coal_rows]]
                right_masks = masks[active_j[coal_rows]]
                material_fraction[coal_rows] = (left_masks | right_masks).sum(dim=1).float() / denom_blocks
                overlap_fraction[coal_rows] = (left_masks & right_masks).sum(dim=1).float() / denom_blocks
                left_fraction[coal_rows] = material_counts[active_i[coal_rows]] / denom_blocks
                right_fraction[coal_rows] = material_counts[active_j[coal_rows]] / denom_blocks

            recomb_rows = torch.nonzero(event_is_recomb == 1, as_tuple=False).reshape(-1)
            if recomb_rows.numel() > 0:
                prefix_counts = torch.cat(
                    [
                        torch.zeros(masks.shape[0], 1, dtype=torch.long, device=device),
                        torch.cumsum(masks.to(dtype=torch.long), dim=1),
                    ],
                    dim=1,
                )
                recomb_i = active_i[recomb_rows]
                recomb_breakpoints = breakpoints[recomb_rows]
                left_counts = prefix_counts[recomb_i, recomb_breakpoints].to(dtype=torch.float32)
                total_counts = material_counts[recomb_i]
                material_fraction[recomb_rows] = total_counts / denom_blocks
                left_fraction[recomb_rows] = left_counts / denom_blocks
                right_fraction[recomb_rows] = (total_counts - left_counts) / denom_blocks
        else:
            recomb_rows = torch.nonzero(event_is_recomb == 1, as_tuple=False).reshape(-1)
            if recomb_rows.numel() > 0:
                left_fraction[recomb_rows] = breakpoints[recomb_rows].float() / denom_blocks
                right_fraction[recomb_rows] = 1.0 - left_fraction[recomb_rows]
                overlap_fraction[recomb_rows] = 0.0

        return torch.stack(
            [
                event_is_recomb,
                active_i.float() / denom_seq,
                active_j.float() / denom_seq,
                breakpoints.float() / denom_blocks,
                material_fraction,
                overlap_fraction,
                left_fraction,
                right_fraction,
            ],
            dim=-1,
        )

    def _build_action_feature(
        self,
        action,
        batch_idx,
        lineage_reps,
        summary_reps,
        seq_features,
        state,
        nb_seq,
    ):
        event_type = action["event_type"]
        event_idx = self.EVENT_TO_IDX[event_type]
        event_tensor = torch.tensor(event_idx, dtype=torch.long, device=lineage_reps.device)
        event_rep = self.event_embedding(event_tensor)
        scalar_rep = self.scalar_embedding(
            self._scalar_features(action, state, nb_seq, lineage_reps.device)
        )

        if event_type == "coal":
            i = action["active_lineage_i"]
            j = action["active_lineage_j"]
            left_rep = lineage_reps[batch_idx, i]
            right_rep = lineage_reps[batch_idx, j]
            primary_rep = left_rep + right_rep
            secondary_rep = torch.abs(left_rep - right_rep)
            tertiary_rep = left_rep * right_rep
        elif event_type == "recomb":
            i = action["active_lineage_i"]
            breakpoint = action["breakpoint"]
            primary_rep = lineage_reps[batch_idx, i]
            left_input, right_input = self._split_sequence_input(seq_features[batch_idx, i], breakpoint)
            secondary_rep = self.seq_embedding(left_input)
            tertiary_rep = self.seq_embedding(right_input)
        else:
            raise ValueError(f"Unknown ARG action event_type: {event_type}")

        return torch.cat(
            [
                primary_rep,
                secondary_rep,
                tertiary_rep,
                summary_reps[batch_idx],
                event_rep,
                scalar_rep,
            ],
            dim=-1,
        )

    def _split_sequence_input(self, lineage_features, breakpoint):
        seq_len = lineage_features.shape[0]
        site_breakpoint = self._site_breakpoint(breakpoint, seq_len)
        left_features = torch.zeros_like(lineage_features)
        right_features = torch.zeros_like(lineage_features)
        left_features[:site_breakpoint] = lineage_features[:site_breakpoint]
        right_features[site_breakpoint:] = lineage_features[site_breakpoint:]
        return left_features.reshape(-1), right_features.reshape(-1)

    def _site_breakpoint(self, breakpoint, seq_len):
        if self.env.num_blocks == seq_len:
            return int(breakpoint)
        scaled = int(round(float(breakpoint) * float(seq_len) / float(self.env.num_blocks)))
        return max(1, min(seq_len - 1, scaled))

    def _scalar_features(self, action, state, nb_seq, device):
        event_type = action["event_type"]
        denom_seq = float(max(nb_seq - 1, 1))
        denom_blocks = float(max(self.env.num_blocks, 1))
        event_is_recomb = 1.0 if event_type == "recomb" else 0.0
        i = action.get("active_lineage_i", 0)
        j = action.get("active_lineage_j", 0)
        breakpoint = action.get("breakpoint", 0)

        material_fraction = 1.0
        overlap_fraction = 1.0 if event_type == "coal" else 0.0
        left_fraction = 1.0
        right_fraction = 1.0

        if state is not None:
            if event_type == "coal":
                mask_i = state.active_lineages[i].material_mask
                mask_j = state.active_lineages[j].material_mask
                union_mask = mask_i | mask_j
                overlap_mask = mask_i & mask_j
                material_fraction = float(union_mask.sum()) / denom_blocks
                overlap_fraction = float(overlap_mask.sum()) / denom_blocks
                left_fraction = float(mask_i.sum()) / denom_blocks
                right_fraction = float(mask_j.sum()) / denom_blocks
            elif event_type == "recomb":
                mask = state.active_lineages[i].material_mask
                left_mask, right_mask = self.env._split_mask(mask, breakpoint)
                material_fraction = float(mask.sum()) / denom_blocks
                left_fraction = float(left_mask.sum()) / denom_blocks
                right_fraction = float(right_mask.sum()) / denom_blocks
        elif event_type == "recomb":
            material_fraction = 1.0
            overlap_fraction = 0.0
            left_fraction = float(breakpoint) / denom_blocks
            right_fraction = 1.0 - left_fraction

        return torch.tensor(
            [
                event_is_recomb,
                float(i) / denom_seq,
                float(j) / denom_seq,
                float(breakpoint) / denom_blocks,
                material_fraction,
                overlap_fraction,
                left_fraction,
                right_fraction,
            ],
            dtype=torch.float32,
            device=device,
        )

    def _candidate_actions_for_batch_item(
        self,
        input_dict,
        batch_idx,
        states,
        action_options,
        selected_event_types,
    ):
        selected_event_type = self._selected_event_type_for_batch_item(
            selected_event_types,
            batch_idx,
        )
        if states is not None:
            return self._state_candidate_actions(states[batch_idx], selected_event_type)

        options = self._action_options_for_batch_item(action_options, batch_idx, input_dict["batch_input"].shape[0])
        if options is not None:
            return self._filter_actions_by_event_type(
                self._actions_from_options(options),
                selected_event_type,
            )

        nb_seq = int(input_dict["batch_nb_seq"][batch_idx].item())
        return self._dense_candidate_actions(nb_seq, selected_event_type)

    def _state_candidate_actions(self, state, selected_event_type=None):
        self._validate_selected_event_type(selected_event_type)
        coal_actions, recomb_weights, recomb_actions = self.env.enumerate_action_options(state) if state.action_options is None else state.action_options
        rates = state.rates if state.rates is not None else self.env.compute_event_rates(
            state,
            coal_actions,
            recomb_weights,
        )
        actions = []
        if selected_event_type in (None, "coal") and rates["lambda_coal"] > 0:
            actions.extend(dict(action) for action in coal_actions)
        if selected_event_type in (None, "recomb") and rates["lambda_recomb"] > 0:
            actions.extend(dict(action) for action in recomb_actions)
        return actions

    def _dense_candidate_actions(self, nb_seq, selected_event_type=None):
        self._validate_selected_event_type(selected_event_type)
        actions = []
        if selected_event_type in (None, "coal"):
            actions.extend(
                {"event_type": "coal", "active_lineage_i": i, "active_lineage_j": j}
                for i, j in itertools.combinations(range(nb_seq), 2)
            )
        if selected_event_type in (None, "recomb") and getattr(self.env, "rho", 1.0) > 0:
            actions.extend(
                {
                    "event_type": "recomb",
                    "active_lineage_i": i,
                    "breakpoint": breakpoint,
                }
                for i in range(nb_seq)
                for breakpoint in range(1, self.env.num_blocks)
            )
        return actions

    def _selected_event_type_for_batch_item(self, selected_event_types, batch_idx):
        if selected_event_types is None:
            return None
        if torch.is_tensor(selected_event_types):
            idx_to_event = {idx: event for event, idx in self.EVENT_TO_IDX.items()}
            selected_event_type = idx_to_event.get(int(selected_event_types[batch_idx].item()))
        else:
            selected_event_type = selected_event_types[batch_idx]
        self._validate_selected_event_type(selected_event_type)
        return selected_event_type

    def _validate_selected_event_type(self, selected_event_type):
        if selected_event_type is not None and selected_event_type not in self.EVENT_TO_IDX:
            raise ValueError(f"Unknown ARG selected_event_type: {selected_event_type}")

    def _filter_actions_by_event_type(self, actions, selected_event_type):
        self._validate_selected_event_type(selected_event_type)
        if selected_event_type is None:
            return actions
        return [action for action in actions if action.get("event_type") == selected_event_type]

    def _event_log_probs_for_batch(self, input_dict, batch_size, reference):
        log_event_probs = input_dict.get("log_event_probs")
        if log_event_probs is None:
            return reference.new_zeros(batch_size)
        if torch.is_tensor(log_event_probs):
            tensor = log_event_probs.to(dtype=reference.dtype, device=reference.device)
        else:
            tensor = torch.tensor(log_event_probs, dtype=reference.dtype, device=reference.device)
        tensor = tensor.reshape(-1)
        if tensor.numel() != batch_size:
            raise ValueError("log_event_probs must have one entry per batch item.")
        return tensor

    def _action_options_for_batch_item(self, action_options, batch_idx, batch_size):
        if action_options is None:
            return None
        if self._looks_like_action_options_tuple(action_options):
            return action_options
        if self._looks_like_action_dict_list(action_options):
            return action_options
        if isinstance(action_options, (list, tuple)) and len(action_options) == batch_size:
            return action_options[batch_idx]
        return action_options

    def _actions_from_options(self, options):
        if self._looks_like_action_options_tuple(options):
            coal_actions, _, recomb_actions = options
            return [dict(action) for action in coal_actions + recomb_actions]
        if isinstance(options, dict):
            return [dict(options)]
        if isinstance(options, (list, tuple)):
            return [dict(action) for action in options]
        raise ValueError("action_options must contain action dicts or env.enumerate_action_options tuples.")

    def _looks_like_action_options_tuple(self, value):
        return (
            isinstance(value, tuple)
            and len(value) == 3
            and isinstance(value[0], list)
            and isinstance(value[1], list)
            and isinstance(value[2], list)
        )

    def _looks_like_action_dict_list(self, value):
        return (
            isinstance(value, (list, tuple))
            and len(value) > 0
            and all(isinstance(item, dict) for item in value)
        )

    def _indices_for_input_actions(self, input_actions, candidate_actions, device):
        indices = []
        for batch_idx, action in enumerate(input_actions):
            normalized_input = self._normalize_action(action)
            for candidate_idx, candidate_action in enumerate(candidate_actions[batch_idx]):
                if self._normalize_action(candidate_action) == normalized_input:
                    indices.append(candidate_idx)
                    break
            else:
                raise ValueError(f"Forced ARG action is not valid for batch item {batch_idx}: {action}")
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _normalize_action(self, action):
        event_type = action.get("event_type")
        if event_type == "coal":
            i = int(action["active_lineage_i"])
            j = int(action["active_lineage_j"])
            left, right = sorted((i, j))
            return ("coal", left, right)
        if event_type == "recomb":
            return ("recomb", int(action["active_lineage_i"]), int(action["breakpoint"]))
        return (event_type,)

    def _get_seq_features(self, input_dict, batch_input):
        if "batch_seq_features" in input_dict:
            return input_dict["batch_seq_features"].float()
        batch_size, active_lineages, _ = batch_input.shape
        return batch_input.reshape(batch_size, active_lineages, self.env.sequence_length, 4)
