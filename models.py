import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ARGModel(nn.Module):
    """One-step ARG action policy.

    The model scores candidate coalescent and recombination actions. When the
    caller provides current ARG states, candidates are read from the environment
    so material-mask constraints are respected.
    """

    EVENT_TO_IDX = {"coal": 0, "recomb": 1}
    SCALAR_FEATURES = 8
    INTERNAL_ACTION_PREFIX = "_"
    DEFAULT_SEQUENCE_ENCODER_BINS = 1024
    DEFAULT_BREAKPOINT_MIXTURES = 4
    BREAKPOINT_POLICIES = {"learned-bin-mass", "uniform"}

    def __init__(self, env, cfg=None):
        super().__init__()
        self.env = env
        embedding_size = 32
        hidden_size = 64
        dropout = 0.0
        self.learn_times = bool(getattr(env, "learn_times", False))
        self.model_version = "compact-binned-v1"
        self.breakpoint_policy = str(
            self._cfg_get(cfg, "breakpoint_policy", "learned-bin-mass")
        )
        if self.breakpoint_policy not in self.BREAKPOINT_POLICIES:
            raise ValueError(
                "breakpoint_policy must be one of: "
                + ", ".join(sorted(self.BREAKPOINT_POLICIES))
            )
        self.breakpoint_mixtures = int(
            self._cfg_get(cfg, "breakpoint_mixtures", self.DEFAULT_BREAKPOINT_MIXTURES)
        )
        if self.breakpoint_mixtures <= 0:
            raise ValueError("breakpoint_mixtures must be positive")

        requested_bins = int(
            self._cfg_get(cfg, "sequence_encoder_bins", self.DEFAULT_SEQUENCE_ENCODER_BINS)
        )
        if requested_bins <= 0:
            raise ValueError("sequence_encoder_bins must be positive")
        self.sequence_encoder_bins = min(requested_bins, int(env.sequence_length))
        input_size = self.sequence_encoder_bins * 4

        self.register_buffer(
            "binned_seq_arrays",
            self._build_binned_sequence_features(),
            persistent=False,
        )
        self.seq_embedding = nn.Linear(input_size, embedding_size)
        self.event_embedding = nn.Embedding(len(self.EVENT_TO_IDX), embedding_size)
        self.scalar_embedding = nn.Linear(self.SCALAR_FEATURES, embedding_size)
        self.event_type_scorer = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(self.EVENT_TO_IDX)),
        )
        nn.init.zeros_(self.event_type_scorer[-1].weight)
        nn.init.zeros_(self.event_type_scorer[-1].bias)
        self.action_scorer = nn.Sequential(
            nn.Linear(embedding_size * 6, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        if self.breakpoint_policy == "learned-bin-mass":
            self.breakpoint_scorer = nn.Sequential(
                nn.Linear(embedding_size * 6, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, self.breakpoint_mixtures * 3),
            )
        else:
            self.breakpoint_scorer = None
        if self.learn_times:
            self.time_scorer = nn.Sequential(
                nn.Linear(embedding_size * 6, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, env.time_env.bins),
            )
        else:
            self.time_scorer = None
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def _cfg_get(self, cfg, key, default):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _build_binned_sequence_features(self):
        seq_arrays = self.env.seq_arrays.detach().to(dtype=torch.float32)
        if seq_arrays.shape[1] == self.sequence_encoder_bins:
            return seq_arrays.clone()
        pooled = F.adaptive_avg_pool1d(
            seq_arrays.permute(0, 2, 1),
            self.sequence_encoder_bins,
        )
        return pooled.permute(0, 2, 1).contiguous()

    def model_params(self):
        return list(self.parameters())

    def _encode_states(self, states, batch_nb_seq=None):
        device = self.binned_seq_arrays.device
        dtype = self.binned_seq_arrays.dtype
        batch_size = len(states)
        if batch_size == 0:
            raise ValueError("ARGModel.forward requires at least one state")

        active_counts = [len(state.active_lineages) for state in states]
        max_active = max(active_counts)
        lineage_inputs = self.binned_seq_arrays.new_zeros(
            batch_size,
            max_active,
            self.sequence_encoder_bins,
            4,
        )

        for batch_idx, state in enumerate(states):
            for lineage_idx, lineage in enumerate(state.active_lineages):
                if lineage.sequences_indices:
                    feature = self.binned_seq_arrays[lineage.sequences_indices].mean(dim=0)
                else:
                    feature = self.binned_seq_arrays.new_zeros(self.sequence_encoder_bins, 4)
                weights = self._material_segments_to_bin_weights(
                    lineage.material_segments,
                    device=device,
                    dtype=dtype,
                )
                lineage_inputs[batch_idx, lineage_idx] = feature * weights[:, None]

        if batch_nb_seq is None:
            batch_nb_seq = torch.tensor(active_counts, dtype=torch.long, device=device)
        else:
            batch_nb_seq = torch.as_tensor(batch_nb_seq, dtype=torch.long, device=device)
            if batch_nb_seq.shape != (batch_size,):
                raise ValueError("batch_nb_seq must have one entry per state")

        flat_inputs = lineage_inputs.reshape(batch_size, max_active, -1)
        lineage_reps = self.seq_embedding(flat_inputs)
        valid_mask = torch.arange(max_active, device=device)[None, :] < batch_nb_seq[:, None]
        lineage_reps = lineage_reps * valid_mask.unsqueeze(-1)
        summary_reps = lineage_reps.sum(dim=1) / batch_nb_seq.clamp_min(1).unsqueeze(-1)
        return lineage_reps, summary_reps, lineage_inputs, batch_nb_seq

    def _material_segments_to_bin_weights(self, material_segments, device, dtype):
        weights = [0.0 for _ in range(self.sequence_encoder_bins)]
        num_blocks = float(max(int(self.env.num_blocks), 1))
        bin_width = num_blocks / float(self.sequence_encoder_bins)
        if bin_width <= 0:
            return torch.zeros(self.sequence_encoder_bins, dtype=dtype, device=device)

        for segment_start, segment_end in material_segments.segments:
            start = max(float(segment_start), 0.0)
            end = min(float(segment_end), num_blocks)
            if end <= start:
                continue
            first_bin = max(0, int(math.floor(start / bin_width)))
            last_bin = min(self.sequence_encoder_bins - 1, int(math.ceil(end / bin_width)) - 1)
            for bin_idx in range(first_bin, last_bin + 1):
                bin_start = float(bin_idx) * bin_width
                bin_end = bin_start + bin_width
                overlap = min(end, bin_end) - max(start, bin_start)
                if overlap > 0:
                    weights[bin_idx] = min(1.0, weights[bin_idx] + overlap / bin_width)
        return torch.tensor(weights, dtype=dtype, device=device)

    def _encode_dense_inputs(self, input_dict):
        if "batch_seq_features" in input_dict:
            seq_features = input_dict["batch_seq_features"].float()
        else:
            batch_input = input_dict["batch_input"].float()
            batch_size, active_lineages, _ = batch_input.shape
            if batch_input.shape[-1] == self.seq_embedding.in_features:
                seq_features = batch_input.reshape(
                    batch_size,
                    active_lineages,
                    self.sequence_encoder_bins,
                    4,
                )
            else:
                seq_features = batch_input.reshape(
                    batch_size,
                    active_lineages,
                    self.env.sequence_length,
                    4,
                )

        batch_size, active_lineages, seq_len, channels = seq_features.shape
        if channels != 4:
            raise ValueError(f"Expected 4 sequence channels, got {channels}")
        if seq_len != self.sequence_encoder_bins:
            pooled = F.adaptive_avg_pool1d(
                seq_features.reshape(batch_size * active_lineages, seq_len, channels)
                .permute(0, 2, 1),
                self.sequence_encoder_bins,
            )
            seq_features = pooled.permute(0, 2, 1).reshape(
                batch_size,
                active_lineages,
                self.sequence_encoder_bins,
                channels,
            )

        batch_input = seq_features.reshape(batch_size, active_lineages, -1)
        if batch_input.shape[-1] != self.seq_embedding.in_features:
            raise ValueError(
                "Encoded batch_input last dimension must match sequence_encoder_bins * 4 "
                f"({self.seq_embedding.in_features}), got {batch_input.shape[-1]}"
            )

        if "batch_nb_seq" in input_dict:
            batch_nb_seq = torch.as_tensor(
                input_dict["batch_nb_seq"],
                dtype=torch.long,
                device=batch_input.device,
            )
        else:
            batch_nb_seq = torch.full(
                (batch_size,),
                active_lineages,
                dtype=torch.long,
                device=batch_input.device,
            )
        return batch_input, seq_features, batch_nb_seq

    def forward(self, input_dict):
        states = input_dict.get("states")
        if states is not None:
            lineage_reps, summary_reps, seq_features, batch_nb_seq = self._encode_states(
                states,
                input_dict.get("batch_nb_seq"),
            )
            batch_size, max_nb_seq, _ = lineage_reps.shape
            device = lineage_reps.device
        else:
            batch_input, seq_features, batch_nb_seq = self._encode_dense_inputs(input_dict)
            batch_size, max_nb_seq, _ = batch_input.shape
            device = batch_input.device
            lineage_reps = self.seq_embedding(batch_input)
            valid_mask = torch.arange(max_nb_seq, device=device)[None, :] < batch_nb_seq[:, None]
            lineage_reps = lineage_reps * valid_mask.unsqueeze(-1)
            summary_reps = lineage_reps.sum(dim=1) / batch_nb_seq.clamp_min(1).unsqueeze(-1)

        valid_mask = torch.arange(max_nb_seq, device=device)[None, :] < batch_nb_seq[:, None]
        lineage_reps = lineage_reps * valid_mask.unsqueeze(-1)
        action_options = input_dict.get("action_options")
        forced_event_types = self._forced_event_types_for_batch(input_dict, batch_size)
        all_candidate_actions = [
            self._candidate_actions_for_batch_item(
                input_dict,
                batch_idx,
                states,
                action_options,
                None,
                batch_size,
            )
            for batch_idx in range(batch_size)
        ]

        if any(len(actions) == 0 for actions in all_candidate_actions):
            raise ValueError("ARGModel.forward received a batch item with no candidate actions.")

        event_logits = self._event_type_logits(
            summary_reps,
            states,
            all_candidate_actions,
        )
        external_log_event_probs = input_dict.get("log_event_probs")
        if forced_event_types is None:
            event_indices = self.sample_event_types(
                event_logits,
                input_dict.get("random_spec"),
            )
            selected_event_types = [
                self._event_type_from_index(int(event_idx.detach().cpu().item()))
                for event_idx in event_indices
            ]
        else:
            selected_event_types = forced_event_types
            event_indices = torch.tensor(
                [self.EVENT_TO_IDX[event_type] for event_type in selected_event_types],
                dtype=torch.long,
                device=device,
            )

        candidate_actions = [
            self._filter_actions_by_event_type(
                all_candidate_actions[batch_idx],
                selected_event_types[batch_idx],
            )
            for batch_idx in range(batch_size)
        ]
        if any(len(actions) == 0 for actions in candidate_actions):
            raise ValueError("Selected ARG event type has no candidate actions.")

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
            selected_candidates = [
                candidate_actions[batch_idx][action_indices[batch_idx].item()]
                for batch_idx in range(batch_size)
            ]
            actions = [
                self._materialize_candidate_action(
                    selected_candidates[batch_idx],
                    device,
                    strip_internal=False,
                )
                for batch_idx in range(batch_size)
            ]
        else:
            if len(input_actions) != batch_size:
                raise ValueError("input_actions length must match batch size.")
            action_indices = self._indices_for_input_actions(input_actions, candidate_actions, device)
            selected_candidates = [
                candidate_actions[batch_idx][action_indices[batch_idx].item()]
                for batch_idx in range(batch_size)
            ]
            actions = [
                self._merge_candidate_metadata(dict(action), selected_candidates[batch_idx])
                for batch_idx, action in enumerate(input_actions)
            ]

        conditional_log_paths_pf = self.compute_log_path_pf({"logits": logits}, action_indices)
        breakpoint_logits = None
        if self.breakpoint_policy == "learned-bin-mass":
            breakpoint_feature_actions = self._breakpoint_feature_actions(
                actions,
                selected_candidates,
            )
            breakpoint_features = self._selected_action_features(
                breakpoint_feature_actions,
                lineage_reps,
                summary_reps,
                seq_features,
                states,
                batch_nb_seq,
            )
            breakpoint_logits = self.breakpoint_scorer(breakpoint_features)
            if input_actions is None:
                self._sample_learned_breakpoints(
                    actions,
                    breakpoint_logits,
                    input_dict.get("random_spec"),
                )
            log_action_detail_pf = self._learned_breakpoint_log_pf(
                actions,
                breakpoint_logits,
                conditional_log_paths_pf,
            )
        else:
            log_action_detail_pf = self._selected_action_detail_log_pf(
                candidate_actions,
                action_indices,
                conditional_log_paths_pf,
            )
        time_logits = None
        time_actions = None
        log_time_pf = conditional_log_paths_pf.new_zeros(batch_size)
        if self.learn_times:
            selected_action_features = self._selected_action_features(
                actions,
                lineage_reps,
                summary_reps,
                seq_features,
                states,
                batch_nb_seq,
            )
            time_logits = self.time_scorer(selected_action_features)
            if input_actions is None:
                time_actions = self.sample_time(time_logits, input_dict.get("random_spec"))
                for batch_idx, action in enumerate(actions):
                    action["time_action"] = int(time_actions[batch_idx].detach().cpu().item())
            else:
                if any("time_action" not in action for action in actions):
                    raise ValueError("learnable ARG times require input_actions to include time_action")
                time_actions = torch.tensor(
                    [int(action["time_action"]) for action in actions],
                    dtype=torch.long,
                    device=device,
                )
            log_time_pf = self.compute_log_time_pf(time_logits, time_actions)
        actions = [self._strip_internal_action_keys(action) for action in actions]
        if external_log_event_probs is None:
            log_event_probs = self.compute_log_event_pf(event_logits, event_indices)
        else:
            log_event_probs = self._event_log_probs_for_batch(
                input_dict,
                batch_size,
                conditional_log_paths_pf,
            )
        log_paths_pf = (
            log_event_probs
            + conditional_log_paths_pf
            + log_action_detail_pf
            + log_time_pf
        )
        return {
            "actions": actions,
            "arg_actions": actions,
            "action_indices": action_indices,
            "candidate_actions": candidate_actions,
            "all_candidate_actions": all_candidate_actions,
            "logits": logits,
            "mask": mask,
            "event_logits": event_logits,
            "event_indices": event_indices,
            "selected_event_types": selected_event_types,
            "time_logits": time_logits,
            "time_actions": time_actions,
            "log_time_pf": log_time_pf,
            "breakpoint_logits": breakpoint_logits,
            "log_action_detail_pf": log_action_detail_pf,
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

    def sample_event_types(self, event_logits, random_spec):
        if random_spec is None:
            random_spec = {"random_action_prob": 0.0}

        if "random_action_prob" in random_spec:
            event_indices = Categorical(logits=event_logits).sample()
            random_p = random_spec["random_action_prob"]
            if random_p > 0:
                batch_size = event_logits.shape[0]
                rand_flag = torch.empty(batch_size, device=event_logits.device).uniform_(0, 1) <= random_p
                for batch_idx in torch.nonzero(rand_flag, as_tuple=False).reshape(-1).tolist():
                    valid_indices = torch.nonzero(
                        torch.isfinite(event_logits[batch_idx]),
                        as_tuple=False,
                    ).reshape(-1)
                    chosen = torch.randint(
                        valid_indices.numel(),
                        size=(),
                        device=event_logits.device,
                    )
                    event_indices[batch_idx] = valid_indices[chosen]
            return event_indices

        temperature = random_spec["T"]
        return Categorical(logits=event_logits / temperature).sample()

    def compute_log_path_pf(self, ret, action_indices):
        logits = ret["logits"]
        batch_idx = torch.arange(logits.shape[0], device=logits.device)
        log_p = self.logsoftmax(logits)
        return log_p[batch_idx, action_indices]

    def compute_log_event_pf(self, event_logits, event_indices):
        batch_idx = torch.arange(event_logits.shape[0], device=event_logits.device)
        log_p = torch.log_softmax(event_logits, dim=-1)
        return log_p[batch_idx, event_indices]

    def sample_time(self, logits, random_spec):
        if random_spec is None:
            random_spec = {"random_action_prob": 0.0}

        if "random_action_prob" in random_spec:
            time_actions = Categorical(logits=logits).sample()
            random_p = random_spec["random_action_prob"]
            if random_p > 0:
                batch_size, actions_num = logits.shape
                rand_flag = torch.empty(batch_size, device=logits.device).uniform_(0, 1) <= random_p
                rand_num = rand_flag.sum().item()
                if rand_num > 0:
                    time_actions[rand_flag] = torch.randint(
                        actions_num,
                        size=(rand_num,),
                        device=logits.device,
                    )
            return time_actions

        temperature = random_spec["T"]
        return Categorical(logits=logits / temperature).sample()

    def compute_log_time_pf(self, time_logits, time_actions):
        batch_idx = torch.arange(time_logits.shape[0], device=time_logits.device)
        log_p = self.logsoftmax(time_logits)
        return log_p[batch_idx, time_actions]

    def _selected_action_features(
        self,
        actions,
        lineage_reps,
        summary_reps,
        seq_features,
        states,
        batch_nb_seq,
    ):
        features = []
        for batch_idx, action in enumerate(actions):
            state = states[batch_idx] if states is not None else None
            features.append(
                self._build_action_feature(
                    action,
                    batch_idx,
                    lineage_reps,
                    summary_reps,
                    seq_features,
                    state,
                    int(batch_nb_seq[batch_idx].item()),
                )
            )
        return torch.stack(features, dim=0)

    def _breakpoint_feature_actions(self, actions, selected_candidates):
        feature_actions = []
        for action, candidate in zip(actions, selected_candidates):
            feature_action = dict(action)
            if candidate.get("_compact_recomb"):
                feature_action["breakpoint"] = int(candidate["breakpoint"])
            feature_actions.append(feature_action)
        return feature_actions

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

        dense_recomb_rows = [
            (row_idx, action["active_lineage_i"], action["breakpoint"])
            for row_idx, action in enumerate(actions)
            if action["event_type"] == "recomb" and not action.get("_compact_recomb")
        ]
        if dense_recomb_rows:
            rows, lineage_indices, breakpoints = zip(*dense_recomb_rows)
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

        compact_recomb_rows = [
            (row_idx, action["active_lineage_i"], action)
            for row_idx, action in enumerate(actions)
            if action["event_type"] == "recomb" and action.get("_compact_recomb")
        ]
        if compact_recomb_rows:
            rows, lineage_indices, compact_actions = zip(*compact_recomb_rows)
            rows = torch.tensor(rows, dtype=torch.long, device=device)
            lineage_indices = torch.tensor(lineage_indices, dtype=torch.long, device=device)
            lineage_rep = lineage_reps[batch_idx, lineage_indices]
            left_fraction = torch.tensor(
                [self._compact_left_fraction(action) for action in compact_actions],
                dtype=lineage_rep.dtype,
                device=device,
            ).unsqueeze(-1)
            right_fraction = 1.0 - left_fraction
            primary_rep[rows] = lineage_rep
            secondary_rep[rows] = lineage_rep * left_fraction
            tertiary_rep[rows] = lineage_rep * right_fraction

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
        rows = []
        for action in actions:
            event_type = action["event_type"]
            event_is_recomb = 1.0 if event_type == "recomb" else 0.0
            active_i = int(action.get("active_lineage_i", 0))
            active_j = int(action.get("active_lineage_j", 0))
            breakpoint = int(action.get("breakpoint", 0))

            material_fraction = 1.0
            overlap_fraction = 1.0 if event_type == "coal" else 0.0
            left_fraction = 1.0
            right_fraction = 1.0

            if state is not None:
                if event_type == "coal":
                    left_segments = state.active_lineages[active_i].material_segments
                    right_segments = state.active_lineages[active_j].material_segments
                    material_fraction = left_segments.union(right_segments).count / denom_blocks
                    overlap_fraction = (
                        left_segments.intersection_count(right_segments) / denom_blocks
                    )
                    left_fraction = left_segments.count / denom_blocks
                    right_fraction = right_segments.count / denom_blocks
                elif event_type == "recomb":
                    segments = state.active_lineages[active_i].material_segments
                    left_segments, right_segments = segments.split(breakpoint)
                    material_fraction = segments.count / denom_blocks
                    overlap_fraction = 0.0
                    left_fraction = left_segments.count / denom_blocks
                    right_fraction = right_segments.count / denom_blocks
            elif event_type == "recomb":
                material_fraction = 1.0
                overlap_fraction = 0.0
                left_fraction = float(breakpoint) / denom_blocks
                right_fraction = 1.0 - left_fraction

            rows.append(
                [
                    event_is_recomb,
                    float(active_i) / denom_seq,
                    float(active_j) / denom_seq,
                    float(breakpoint) / denom_blocks,
                    material_fraction,
                    overlap_fraction,
                    left_fraction,
                    right_fraction,
                ]
            )

        return torch.tensor(rows, dtype=torch.float32, device=device)

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
            primary_rep = lineage_reps[batch_idx, i]
            if action.get("_compact_recomb"):
                left_fraction = self._compact_left_fraction(action)
                secondary_rep = primary_rep * left_fraction
                tertiary_rep = primary_rep * (1.0 - left_fraction)
            else:
                breakpoint = action["breakpoint"]
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
                segments_i = state.active_lineages[i].material_segments
                segments_j = state.active_lineages[j].material_segments
                material_fraction = segments_i.union(segments_j).count / denom_blocks
                overlap_fraction = segments_i.intersection_count(segments_j) / denom_blocks
                left_fraction = segments_i.count / denom_blocks
                right_fraction = segments_j.count / denom_blocks
            elif event_type == "recomb":
                segments = state.active_lineages[i].material_segments
                left_segments, right_segments = segments.split(breakpoint)
                material_fraction = segments.count / denom_blocks
                overlap_fraction = 0.0
                left_fraction = left_segments.count / denom_blocks
                right_fraction = right_segments.count / denom_blocks
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
        batch_size,
    ):
        selected_event_type = self._selected_event_type_for_batch_item(
            selected_event_types,
            batch_idx,
        )
        if states is not None:
            return self._state_candidate_actions(states[batch_idx], selected_event_type)

        options = self._action_options_for_batch_item(action_options, batch_idx, batch_size)
        if options is not None:
            return self._filter_actions_by_event_type(
                self._actions_from_options(options),
                selected_event_type,
            )

        nb_seq = int(input_dict["batch_nb_seq"][batch_idx].item())
        return self._dense_candidate_actions(nb_seq, selected_event_type)

    def _forced_event_types_for_batch(self, input_dict, batch_size):
        selected_event_types = input_dict.get("selected_event_types")
        if selected_event_types is not None:
            if len(selected_event_types) != batch_size:
                raise ValueError("selected_event_types must have one entry per batch item.")
            return [
                self._normalize_event_type_value(selected_event_types, batch_idx)
                for batch_idx in range(batch_size)
            ]

        input_actions = input_dict.get("input_actions")
        if input_actions is None:
            return None
        if len(input_actions) != batch_size:
            raise ValueError("input_actions length must match batch size.")
        event_types = [action.get("event_type") for action in input_actions]
        for event_type in event_types:
            self._validate_selected_event_type(event_type)
        return event_types

    def _normalize_event_type_value(self, selected_event_types, batch_idx):
        if torch.is_tensor(selected_event_types):
            return self._event_type_from_index(int(selected_event_types[batch_idx].item()))
        event_type = selected_event_types[batch_idx]
        self._validate_selected_event_type(event_type)
        return event_type

    def _event_type_from_index(self, event_idx):
        idx_to_event = {idx: event for event, idx in self.EVENT_TO_IDX.items()}
        event_type = idx_to_event.get(int(event_idx))
        self._validate_selected_event_type(event_type)
        return event_type

    def _event_type_logits(self, summary_reps, states, candidate_actions):
        learned_logits = self.event_type_scorer(summary_reps)
        prior_logits = learned_logits.new_full(learned_logits.shape, float("-inf"))

        for batch_idx, actions in enumerate(candidate_actions):
            available_events = {action["event_type"] for action in actions}
            if states is not None:
                probs = self.env.compute_event_probabilities(states[batch_idx])
                for event_type, event_idx in self.EVENT_TO_IDX.items():
                    probability = float(probs.get(event_type, 0.0))
                    if event_type in available_events and probability > 0.0:
                        prior_logits[batch_idx, event_idx] = math.log(probability)
            else:
                if available_events:
                    log_prob = -math.log(len(available_events))
                    for event_type in available_events:
                        prior_logits[batch_idx, self.EVENT_TO_IDX[event_type]] = log_prob

        return learned_logits + prior_logits

    def _state_candidate_actions(self, state, selected_event_type=None):
        self._validate_selected_event_type(selected_event_type)
        prior_options = self.env.enumerate_prior_options(state)
        rates = prior_options.rates
        actions = []
        if selected_event_type in (None, "coal") and rates["lambda_coal"] > 0:
            actions.extend(dict(action) for action in prior_options.coal_actions)
        if selected_event_type in (None, "recomb") and rates["lambda_recomb"] > 0:
            actions.extend(
                self._compact_recombination_action(choice)
                for choice in prior_options.recomb_choices
                if choice.breakpoint_count > 0
            )
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
            for candidate_idx, candidate_action in enumerate(candidate_actions[batch_idx]):
                if self._candidate_matches_input_action(candidate_action, action):
                    indices.append(candidate_idx)
                    break
            else:
                raise ValueError(f"Forced ARG action is not valid for batch item {batch_idx}: {action}")
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _candidate_matches_input_action(self, candidate_action, input_action):
        if candidate_action.get("_compact_recomb"):
            if input_action.get("event_type") != "recomb":
                return False
            if int(candidate_action["active_lineage_i"]) != int(input_action["active_lineage_i"]):
                return False
            breakpoint = int(input_action["breakpoint"])
            return (
                int(candidate_action["_breakpoint_span_start"])
                <= breakpoint
                <= int(candidate_action["_breakpoint_span_end"])
            )
        return self._normalize_action(candidate_action) == self._normalize_action(input_action)

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

    def _compact_recombination_action(self, choice):
        breakpoint_count = int(choice.breakpoint_count)
        breakpoint_start = int(choice.span_start) + 1
        breakpoint_end = int(choice.span_end)
        representative_breakpoint = breakpoint_start + (breakpoint_count - 1) // 2
        return {
            "event_type": "recomb",
            "active_lineage_i": int(choice.active_lineage_i),
            "breakpoint": int(representative_breakpoint),
            "_compact_recomb": True,
            "_breakpoint_span_start": breakpoint_start,
            "_breakpoint_span_end": breakpoint_end,
            "_breakpoint_count": breakpoint_count,
        }

    def _materialize_candidate_action(self, candidate_action, device, strip_internal=True):
        action = dict(candidate_action)
        if action.get("_compact_recomb") and self.breakpoint_policy == "uniform":
            breakpoint_count = int(action["_breakpoint_count"])
            if breakpoint_count <= 0:
                raise ValueError(f"Compact recombination candidate has no valid breakpoints: {action}")
            offset = int(
                torch.randint(
                    breakpoint_count,
                    size=(),
                    device=device,
                ).detach().cpu().item()
            )
            action["breakpoint"] = int(action["_breakpoint_span_start"]) + offset
        if strip_internal:
            return self._strip_internal_action_keys(action)
        return action

    def _sample_learned_breakpoints(self, actions, breakpoint_logits, random_spec=None):
        mix_logits, locs, scales = self._breakpoint_distribution_params(breakpoint_logits)
        for batch_idx, action in enumerate(actions):
            if not action.get("_compact_recomb"):
                continue
            breakpoint_count = int(action["_breakpoint_count"])
            if breakpoint_count <= 0:
                raise ValueError(f"Compact recombination candidate has no valid breakpoints: {action}")
            component = Categorical(logits=mix_logits[batch_idx]).sample()
            noise = torch.randn((), dtype=locs.dtype, device=locs.device)
            raw_position = locs[batch_idx, component] + scales[batch_idx, component] * noise
            normalized = torch.sigmoid(raw_position)
            offset = int(
                torch.floor(normalized * float(breakpoint_count))
                .clamp(0, breakpoint_count - 1)
                .detach()
                .cpu()
                .item()
            )
            action["breakpoint"] = int(action["_breakpoint_span_start"]) + offset

    def _breakpoint_distribution_params(self, breakpoint_logits):
        params = breakpoint_logits.reshape(-1, self.breakpoint_mixtures, 3)
        mix_logits = params[:, :, 0]
        locs = params[:, :, 1].clamp(-12.0, 12.0)
        scales = F.softplus(params[:, :, 2]).clamp_min(1e-3).clamp_max(12.0)
        return mix_logits, locs, scales

    def _learned_breakpoint_log_pf(self, actions, breakpoint_logits, reference):
        values = []
        for batch_idx, action in enumerate(actions):
            if action.get("_compact_recomb"):
                values.append(
                    self._learned_breakpoint_bin_log_prob(
                        action,
                        breakpoint_logits[batch_idx],
                    )
                )
            else:
                values.append(reference.new_tensor(0.0))
        return torch.stack(values).to(dtype=reference.dtype, device=reference.device)

    def _learned_breakpoint_bin_log_prob(self, action, breakpoint_logits_row):
        breakpoint_count = int(action["_breakpoint_count"])
        breakpoint_start = int(action["_breakpoint_span_start"])
        breakpoint = int(action["breakpoint"])
        offset = breakpoint - breakpoint_start
        if offset < 0 or offset >= breakpoint_count:
            raise ValueError(f"Breakpoint is outside compact candidate span: {action}")

        lower = float(offset) / float(breakpoint_count)
        upper = float(offset + 1) / float(breakpoint_count)
        mix_logits, locs, scales = self._breakpoint_distribution_params(
            breakpoint_logits_row.reshape(1, -1)
        )
        mix_logits = mix_logits[0]
        locs = locs[0]
        scales = scales[0]
        standard_normal = torch.distributions.Normal(
            torch.zeros_like(locs),
            torch.ones_like(scales),
        )
        lower_cdf = self._breakpoint_boundary_cdf(lower, locs, scales, standard_normal)
        upper_cdf = self._breakpoint_boundary_cdf(upper, locs, scales, standard_normal)
        component_mass = (upper_cdf - lower_cdf).clamp_min(0.0)
        mixture_mass = (torch.softmax(mix_logits, dim=-1) * component_mass).sum()
        return torch.log(mixture_mass.clamp_min(1e-12))

    def _breakpoint_boundary_cdf(self, value, locs, scales, standard_normal):
        if value <= 0.0:
            return locs.new_zeros(locs.shape)
        if value >= 1.0:
            return locs.new_ones(locs.shape)
        value_tensor = locs.new_tensor(value).clamp(1e-7, 1.0 - 1e-7)
        boundary = torch.logit(value_tensor)
        return standard_normal.cdf((boundary - locs) / scales)

    def _merge_candidate_metadata(self, action, candidate_action):
        if not candidate_action.get("_compact_recomb"):
            return action
        merged = dict(action)
        for key, value in candidate_action.items():
            if key.startswith(self.INTERNAL_ACTION_PREFIX):
                merged[key] = value
        return merged

    def _strip_internal_action_keys(self, action):
        return {
            key: value
            for key, value in action.items()
            if not str(key).startswith(self.INTERNAL_ACTION_PREFIX)
        }

    def _selected_action_detail_log_pf(self, candidate_actions, action_indices, reference):
        values = []
        for batch_idx, actions in enumerate(candidate_actions):
            candidate = actions[action_indices[batch_idx].item()]
            if candidate.get("_compact_recomb"):
                breakpoint_count = int(candidate["_breakpoint_count"])
                values.append(-math.log(breakpoint_count))
            else:
                values.append(0.0)
        return torch.tensor(values, dtype=reference.dtype, device=reference.device)

    def _compact_left_fraction(self, action):
        breakpoint = float(action.get("breakpoint", action["_breakpoint_span_start"]))
        span_start = float(action.get("_breakpoint_span_start", 1)) - 1.0
        span_end = float(action.get("_breakpoint_span_end", self.env.num_blocks - 1))
        span_width = max(span_end - span_start, 1.0)
        return max(0.0, min(1.0, (breakpoint - span_start) / span_width))
