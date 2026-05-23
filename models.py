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
    BREAKPOINT_POLICIES = {"categorical-span", "learned-bin-mass", "uniform"}

    def __init__(self, env, cfg=None):
        super().__init__()
        self.env = env
        self.device = env.device
        embedding_size = 32
        hidden_size = 64
        dropout = 0.0
        self.model_version = "state-three-head-v1"
        requested_breakpoint_policy = str(
            self._cfg_get(cfg, "breakpoint_policy", "categorical-span")
        )
        if requested_breakpoint_policy not in self.BREAKPOINT_POLICIES:
            raise ValueError(
                "breakpoint_policy must be one of: "
                + ", ".join(sorted(self.BREAKPOINT_POLICIES))
            )
        self.breakpoint_policy = "categorical-span"
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
            "source_seq_arrays",
            self._build_source_sequence_features(),
            persistent=False,
        )
        self.seq_embedding = nn.Linear(input_size, embedding_size)
        self.event_embedding = nn.Embedding(len(self.EVENT_TO_IDX), embedding_size)
        self.scalar_embedding = nn.Linear(self.SCALAR_FEATURES, embedding_size)
        self.action_scorer = nn.Sequential(
            nn.Linear(embedding_size * 6, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.breakpoint_scorer = nn.Sequential(
            nn.Linear(embedding_size * 6, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.time_scorer = nn.Sequential(
            nn.Linear(embedding_size * 6, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, env.time_env.bins),
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def _cfg_get(self, cfg, key, default):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _build_source_sequence_features(self):
        return self.env.seq_arrays.detach().to(dtype=torch.float32).clone()

    def model_params(self):
        return list(self.parameters())

    def _encode_states(self, states):
        device = self.source_seq_arrays.device
        dtype = self.source_seq_arrays.dtype
        batch_size = len(states)
        if batch_size == 0:
            raise ValueError("ARGModel.forward requires at least one state")

        active_counts = [len(state.active_lineages) for state in states]
        batch_active_lineage_counts = torch.tensor(active_counts, dtype=torch.long, device=device)
        max_material_segments = max([max([lineage.material_segments.count for lineage in state.active_lineages]) for state in states])
        sequence_length = self.source_seq_arrays.shape[1]
        lineage_seq_features = self.source_seq_arrays.new_zeros(
            batch_size,
            sequence_length,
            max_material_segments,
            4,
        )

        for batch_idx, state in enumerate(states):
            for lineage_idx, lineage in enumerate(state.active_lineages):
                if lineage.sequences_indices:
                    feature = self.source_seq_arrays[lineage.sequences_indices].mean(dim=0)
                else:
                    feature = self.source_seq_arrays.new_zeros(sequence_length, 4)
                weights = self._material_segments_masking(
                    lineage.material_segments,
                    device=device,
                    dtype=dtype,
                )
                lineage_seq_features[batch_idx, lineage_idx] = feature * weights[:, None]

        return self._encode_lineage_features(lineage_seq_features, batch_active_lineage_counts)

    def _material_segments_masking(self, material_segments, device, dtype):
        sequence_length = int(self.source_seq_arrays.shape[1])
        weights = torch.zeros(sequence_length, dtype=torch.float64)  # or float32 if preferred
        num_blocks = float(max(int(self.env.num_blocks), 1))
        # site_width = float(sequence_length) / num_blocks
   
        for segment_start, segment_end in material_segments.segments:
            start = max(float(segment_start), 0.0)
            end = min(float(segment_end), num_blocks)
            if end <= start:
                raise ValueError("segment end must be greater than start")

            weights[start:end] = 1.0
        return torch.tensor(weights, dtype=dtype, device=device)

    def _material_segments_to_bin_weights(self, material_segments, device, dtype):
        site_weights = self._material_segments_to_site_weights(material_segments, device, dtype)
        if site_weights.numel() == self.sequence_encoder_bins:
            return site_weights
        pooled = F.adaptive_avg_pool1d(
            site_weights.reshape(1, 1, -1),
            self.sequence_encoder_bins,
        )
        return pooled.reshape(-1)

    def _encode_lineage_features(self, lineage_seq_features, batch_active_lineage_counts):
        
        seq_features = lineage_seq_features.to(device=self.device, dtype=torch.float32)
        if seq_features.dim() != 4:
            raise ValueError(
                "sequence features must have shape "
                "(batch, active_lineages, sequence_length_or_bins, channels)"
            )

        batch_size, active_lineages, _, _ = seq_features.shape
        
        # seq_features = self._pool_sequence_features(seq_features, seq_len, channels)

        batch_input = lineage_seq_features.reshape(batch_size, active_lineages, -1)
        if batch_input.shape[-1] != self.seq_embedding.in_features:
            raise ValueError(
                "Encoded batch_input last dimension must match sequence_encoder_bins * 4 "
                f"({self.seq_embedding.in_features}), got {batch_input.shape[-1]}"
            )

        if batch_active_lineage_counts is not None:
            batch_nb_seq = torch.as_tensor(batch_nb_seq, dtype=torch.long, device=device)
            if batch_nb_seq.shape != (batch_size,):
                raise ValueError("batch_nb_seq must have one entry per batch item")
            if torch.any(batch_nb_seq < 0) or torch.any(batch_nb_seq > active_lineages):
                raise ValueError("batch_nb_seq entries must be between 0 and active_lineages")
        else:
            batch_active_lineage_counts = torch.full(
                (batch_size,),
                active_lineages,
                dtype=torch.long,
                device=self.device,
            )

        lineage_reps = self.seq_embedding(batch_input)
        valid_mask = torch.arange(active_lineages, device=device)[None, :] < batch_nb_seq[:, None]
        lineage_reps = lineage_reps * valid_mask.unsqueeze(-1)
        summary_reps = lineage_reps.sum(dim=1) / batch_nb_seq.clamp_min(1).unsqueeze(-1)
        return lineage_reps, summary_reps, seq_features, batch_nb_seq

    def _pool_sequence_features(self, seq_features, seq_len, channels):
        if seq_len == self.sequence_encoder_bins:
            return seq_features.contiguous()

        batch_size, active_lineages = seq_features.shape[:2]
        pooled = F.adaptive_avg_pool1d(
            seq_features.reshape(batch_size * active_lineages, seq_len, channels)
            .permute(0, 2, 1),
            self.sequence_encoder_bins,
        )
        return pooled.permute(0, 2, 1).reshape(
            batch_size,
            active_lineages,
            self.sequence_encoder_bins,
            channels,
        )

    def forward(self, input_dict):
        states = input_dict.get("states")
        if states is not None:
            ## State encoding.
            lineage_reps, summary_reps, seq_features, batch_nb_seq = self._encode_states(
                states,
                input_dict.get("batch_nb_seq"),
            )
        else:
            raise ValueError("States are required for the model to run.")

        batch_size, max_nb_seq, _ = lineage_reps.shape
        valid_mask = torch.arange(max_nb_seq, device=self.device)[None, :] < batch_nb_seq[:, None]
        lineage_reps = lineage_reps * valid_mask.unsqueeze(-1)
        all_candidate_actions = input_dict.get("input_actions")
        if states is None:
            raise ValueError("CWR event-rate sampling requires state-based rollout inputs.")
        # all_candidate_actions = [
        #     self._candidate_actions_for_batch_item(
        #         input_dict,
        #         batch_idx,
        #         states,
        #         action_options,
        #         batch_size,
        #     )
        #     for batch_idx in range(batch_size)
        # ]
        

        if any(len(actions) == 0 for actions in all_candidate_actions):
            raise ValueError("ARGModel.forward received a batch item with no candidate actions.")

        logits = self._score_candidates(
            all_candidate_actions,
            lineage_reps, ## Don't know what it contains.
            summary_reps, ## Don't know what it contains.
            seq_features, ## Don't know what it contains. I have a guess.
            states,
            batch_nb_seq,  ## What's this?
        )
        mask = torch.isneginf(logits)

        choosen_actions = self.sample(logits, all_candidate_actions, input_dict.get("random_spec"))
        # if input_actions is not None:
        #     if len(input_actions) != batch_size:
        #         raise ValueError("input_actions length must match batch size.")
        #     action_indices = self._indices_for_input_actions(input_actions, all_candidate_actions, device)
        #     selected_candidates = [
        #         all_candidate_actions[batch_idx][action_indices[batch_idx].item()]
        #         for batch_idx in range(batch_size)
        #     ]
        #     actions = [
        #         self._merge_candidate_metadata(dict(action), selected_candidates[batch_idx])
        #         for batch_idx, action in enumerate(input_actions)
        #     ]

        log_action_pf = self.compute_log_path_pf({"logits": logits}, action_indices)
        (
            breakpoint_logits,
            breakpoint_actions,
            breakpoint_indices,
            log_breakpoint_pf,
        ) = self._select_breakpoints(
            actions,
            lineage_reps,
            summary_reps,
            seq_features,
            states,
            batch_nb_seq,
            input_actions=input_actions,
            random_spec=input_dict.get("random_spec"),
        )
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
        chosen_event_types = [action["event_type"] for action in actions]
        event_indices = torch.tensor(
            [self.EVENT_TO_IDX[event_type] for event_type in chosen_event_types],
            dtype=torch.long,
            device=device,
        )
        event_log_probs = self._event_log_probs_from_action_logits(all_candidate_actions, logits)
        log_event_pf = logits.new_zeros(batch_size)
        actions = [self._strip_internal_action_keys(action) for action in actions]
        log_paths_pf = log_action_pf + log_breakpoint_pf + log_time_pf
        return {
            "actions": actions,
            "arg_actions": actions,
            "action_indices": action_indices,
            "candidate_actions": all_candidate_actions,
            "all_candidate_actions": all_candidate_actions,
            "logits": logits,
            "mask": mask,
            "event_indices": event_indices,
            "chosen_event_types": chosen_event_types,
            "time_logits": time_logits,
            "time_actions": time_actions,
            "log_time_pf": log_time_pf,
            "breakpoint_logits": breakpoint_logits,
            "breakpoint_actions": breakpoint_actions,
            "breakpoint_indices": breakpoint_indices,
            "log_action_pf": log_action_pf,
            "log_breakpoint_pf": log_breakpoint_pf,
            "log_action_detail_pf": log_breakpoint_pf,
            "log_event_pf": log_event_pf,
            "log_paths_pf": log_paths_pf,
            "conditional_log_paths_pf": log_action_pf,
            "event_log_probs": event_log_probs,
        }

    def sample(self, ret, random_spec=None):
        logits = ret["logits"]
        if random_spec is None:
            return Categorical(logits=logits).sample()

        temperature = random_spec["T"]
        return Categorical(logits=logits / temperature).sample()

    def compute_log_path_pf(self, ret, action_indices):
        logits = ret["logits"]
        batch_idx = torch.arange(logits.shape[0], device=self.device)
        log_p = self.logsoftmax(logits)
        return log_p[batch_idx, action_indices]

    def sample_time(self, logits, random_spec):
        if random_spec is None:
            return Categorical(logits=logits).sample()

        temperature = random_spec["T"]
        return Categorical(logits=logits / temperature).sample()

    def sample_breakpoint(self, logits, random_spec):
        if random_spec is None:
            return Categorical(logits=logits).sample()

        temperature = random_spec["T"]
        return Categorical(logits=logits / temperature).sample()

    def compute_log_time_pf(self, time_logits, time_actions):
        batch_idx = torch.arange(time_logits.shape[0], device=time_logits.device)
        log_p = self.logsoftmax(time_logits)
        return log_p[batch_idx, time_actions]

    def _event_log_probs_from_action_logits(self, candidate_actions, logits):
        event_log_probs = logits.new_full(
            (len(candidate_actions), len(self.EVENT_TO_IDX)),
            float("-inf"),
        )
        normalizers = torch.logsumexp(logits, dim=1)
        for batch_idx, actions in enumerate(candidate_actions):
            for event_type, event_idx in self.EVENT_TO_IDX.items():
                indices = [
                    action_idx
                    for action_idx, action in enumerate(actions)
                    if action.get("event_type") == event_type
                ]
                if indices:
                    event_logits = logits[batch_idx, torch.tensor(indices, device=logits.device)]
                    event_log_probs[batch_idx, event_idx] = (
                        torch.logsumexp(event_logits, dim=0) - normalizers[batch_idx]
                    )
        return event_log_probs

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

    def _select_breakpoints(
        self,
        actions,
        lineage_reps,
        summary_reps,
        seq_features,
        states,
        batch_nb_seq,
        input_actions=None,
        random_spec=None,
    ):
        batch_size = len(actions)
        device = lineage_reps.device
        valid_breakpoint_lists = [
            self._valid_breakpoints(action) if action.get("event_type") == "recomb" else []
            for action in actions
        ]
        max_breakpoints = max((len(values) for values in valid_breakpoint_lists), default=0)
        breakpoint_logits = lineage_reps.new_full((batch_size, max_breakpoints), float("-inf"))
        breakpoint_actions = torch.full(
            (batch_size, max_breakpoints),
            -1,
            dtype=torch.long,
            device=device,
        )
        breakpoint_indices = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        log_breakpoint_pf = lineage_reps.new_zeros(batch_size)

        for batch_idx, valid_breakpoints in enumerate(valid_breakpoint_lists):
            if not valid_breakpoints:
                continue

            state = states[batch_idx] if states is not None else None
            nb_seq = int(batch_nb_seq[batch_idx].item())
            breakpoint_actions[batch_idx, :len(valid_breakpoints)] = torch.tensor(
                valid_breakpoints,
                dtype=torch.long,
                device=device,
            )
            breakpoint_feature_actions = [
                dict(actions[batch_idx], breakpoint=breakpoint)
                for breakpoint in valid_breakpoints
            ]
            breakpoint_features = self._batched_action_features(
                breakpoint_feature_actions,
                batch_idx,
                lineage_reps,
                summary_reps,
                seq_features,
                state,
                nb_seq,
            )
            row_logits = self.breakpoint_scorer(breakpoint_features).squeeze(-1)
            breakpoint_logits[batch_idx, :len(valid_breakpoints)] = row_logits

            if input_actions is None:
                selected_idx = int(
                    self.sample_breakpoint(row_logits, random_spec).detach().cpu().item()
                )
            else:
                forced_breakpoint = int(actions[batch_idx]["breakpoint"])
                try:
                    selected_idx = valid_breakpoints.index(forced_breakpoint)
                except ValueError as exc:
                    raise ValueError(
                        "Forced recombination breakpoint is outside the selected "
                        f"candidate span for batch item {batch_idx}: {actions[batch_idx]}"
                    ) from exc

            actions[batch_idx]["breakpoint"] = int(valid_breakpoints[selected_idx])
            breakpoint_indices[batch_idx] = selected_idx
            log_breakpoint_pf[batch_idx] = F.log_softmax(row_logits, dim=0)[selected_idx]

        return breakpoint_logits, breakpoint_actions, breakpoint_indices, log_breakpoint_pf

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
            (row_idx, action["active_lineage_i"], self._action_feature_breakpoint(action))
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
        rows = []
        for action in actions:
            event_type = action["event_type"]
            event_is_recomb = 1.0 if event_type == "recomb" else 0.0
            active_i = int(action.get("active_lineage_i", 0))
            active_j = int(action.get("active_lineage_j", 0))
            breakpoint = self._action_feature_breakpoint(action)

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
            breakpoint = self._action_feature_breakpoint(action)
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

    def _action_feature_breakpoint(self, action):
        breakpoint = action.get("breakpoint")
        if breakpoint is not None:
            return int(breakpoint)

        valid_breakpoints = self._valid_breakpoints(action)
        if valid_breakpoints:
            return int(valid_breakpoints[(len(valid_breakpoints) - 1) // 2])
        return 0

    def _valid_breakpoints(self, action):
        if action.get("event_type") != "recomb":
            return []

        if "_breakpoint_span_start" in action and "_breakpoint_span_end" in action:
            start = int(action["_breakpoint_span_start"])
            end = int(action["_breakpoint_span_end"])
        elif "span_start" in action and "span_end" in action:
            start = int(action["span_start"]) + 1
            end = int(action["span_end"])
        elif action.get("breakpoint") is not None:
            breakpoint = int(action["breakpoint"])
            return [breakpoint]
        else:
            return []

        if end < start:
            return []
        return list(range(start, end + 1))

    def _scalar_features(self, action, state, nb_seq, device):
        event_type = action["event_type"]
        denom_seq = float(max(nb_seq - 1, 1))
        denom_blocks = float(max(self.env.num_blocks, 1))
        event_is_recomb = 1.0 if event_type == "recomb" else 0.0
        i = action.get("active_lineage_i", 0)
        j = action.get("active_lineage_j", 0)
        breakpoint = self._action_feature_breakpoint(action)

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
        if candidate_action.get("event_type") != input_action.get("event_type"):
            return False
        if candidate_action.get("event_type") == "coal":
            return self._normalize_action(candidate_action) == self._normalize_action(input_action)
        if candidate_action.get("event_type") == "recomb":
            if int(candidate_action["active_lineage_i"]) != int(input_action["active_lineage_i"]):
                return False
            if input_action.get("breakpoint") is None:
                return False
            breakpoint = int(input_action["breakpoint"])
            return breakpoint in self._valid_breakpoints(candidate_action)
        return False

    def _normalize_action(self, action):
        event_type = action.get("event_type")
        if event_type == "coal":
            i = int(action["active_lineage_i"])
            j = int(action["active_lineage_j"])
            left, right = sorted((i, j))
            return ("coal", left, right)
        if event_type == "recomb":
            breakpoint = action.get("breakpoint")
            if breakpoint is None:
                return (
                    "recomb",
                    int(action["active_lineage_i"]),
                    tuple(self._valid_breakpoints(action)),
                )
            return ("recomb", int(action["active_lineage_i"]), int(breakpoint))
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
        if action.get("event_type") == "recomb" and action.get("breakpoint") is None:
            action.pop("breakpoint", None)
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
        merged = dict(action)
        for key, value in candidate_action.items():
            if key.startswith(self.INTERNAL_ACTION_PREFIX) or key not in merged:
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
