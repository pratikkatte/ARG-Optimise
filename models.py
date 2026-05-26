import math

from env import CoalescenceChoice, MaterialSegments, RecombinationChoice
from breakpoint_model import BreakpointSplitPositionCNN
from time_model import TimeModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import replace


class ARGModel(nn.Module):
    """One-step ARG action policy.

    The model scores candidate coalescent and recombination actions. When the
    caller provides current ARG states, candidates are read from the environment
    so material-mask constraints are respected.
    """

    def __init__(
        self,
        env,
        embedding_size=32,
        hidden_size=64,
        dropout=0.0,
        breakpoint_hidden_dim=128,
        breakpoint_dropout=0.1,
    ):
        super().__init__()
        self.env = env
        self.device = env.device
        input_size = int(env.sequence_length) * 4

        self.register_buffer(
            "source_seq_arrays",
            self._build_source_sequence_features(),
            persistent=False,
        )
        self.seq_embedding = nn.Linear(input_size, embedding_size)
        self.action_scorer = nn.Sequential(
            nn.Linear(embedding_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.breakpoint_scorer = BreakpointSplitPositionCNN(
            hidden_dim=breakpoint_hidden_dim,
            dropout=breakpoint_dropout,
        ).to(self.device)

        self.time_scorer = TimeModel(
            embedding_size * 4,
            hidden_size,
            dropout,
            env.time_env.bins,
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def _build_source_sequence_features(self):
        return self.env.seq_arrays.detach().to(dtype=torch.float32).clone()

    def model_params(self):
        return list(self.parameters())

    def _encode_lineage_features(self, lineage_seq_features, batch_active_lineage_counts):
        batch_size, active_lineages, seq_len, channels = lineage_seq_features.shape
        if seq_len != int(self.env.sequence_length) or channels != 4:
            raise ValueError(
                "sequence features must have shape "
                f"(batch, active_lineages, {int(self.env.sequence_length)}, 4), "
                f"got {tuple(lineage_seq_features.shape)}"
            )

        batch_input = lineage_seq_features.reshape(batch_size, active_lineages, -1)
        if batch_input.shape[-1] != self.seq_embedding.in_features:
            raise ValueError(
                "Encoded batch_input last dimension must match sequence_length * 4 "
                f"({self.seq_embedding.in_features}), got {batch_input.shape[-1]}"
            )

        batch_input = batch_input.to(device=self.device, dtype=torch.float32)
        batch_active_lineage_counts = batch_active_lineage_counts.to(device=self.device, dtype=torch.long)

        lineage_reps = self.seq_embedding(batch_input)

        valid_mask = torch.arange(active_lineages, device=self.device)[None, :] < batch_active_lineage_counts[:, None]
        lineage_reps = lineage_reps * valid_mask.unsqueeze(-1)
        summary_reps = lineage_reps.sum(dim=1) / batch_active_lineage_counts.clamp_min(1).unsqueeze(-1)
        return lineage_reps, summary_reps, lineage_seq_features, batch_active_lineage_counts

    def _encode_states(self, states):
        batch_size = len(states)
        if batch_size == 0:
            raise ValueError("ARGModel.forward requires at least one state")

        active_counts = [len(state.active_lineages) for state in states]
        batch_active_lineage_counts = torch.tensor(
            active_counts, dtype=torch.long, device=self.device,
        )

        sequence_length = self.source_seq_arrays.shape[1]
        max_active_lineages = max(active_counts, default=0)
        lineage_seq_features = self.source_seq_arrays.new_zeros(
            batch_size,
            max_active_lineages,
            sequence_length,
            4,
        )

        for batch_idx, state in enumerate(states):
            for lineage_idx, lineage in enumerate(state.active_lineages):
                feature = self._lineage_partials_tensor(lineage)
                weights = self._material_segments_masking(
                    lineage.material_segments,
                    device=self.device,
                    dtype=self.source_seq_arrays.dtype,
                )
                masked_feature = feature * weights[:, None]
                lineage_seq_features[batch_idx, lineage_idx] = (
                    self.env.evolution_model.normalize_partials(masked_feature)
                )

        return self._encode_lineage_features(lineage_seq_features, batch_active_lineage_counts)

    def _lineage_partials_tensor(self, lineage):
        if lineage.partials is None:
            raise ValueError(
                f"Active ARG lineage {lineage.node_id} is missing partials; "
                "state transitions must populate ARGLineage.partials"
            )
        partials = lineage.partials
        if torch.is_tensor(partials):
            partials = partials.to(device=self.device, dtype=torch.float32)
        else:
            partials = torch.as_tensor(partials, device=self.device, dtype=torch.float32)
        expected_shape = (int(self.env.sequence_length), 4)
        if tuple(partials.shape) != expected_shape:
            raise ValueError(
                f"Active ARG lineage {lineage.node_id} partials must have shape "
                f"{expected_shape}, got {tuple(partials.shape)}"
            )
        return partials

    def _material_segments_masking(self, material_segments, device, dtype):
        sequence_length = int(self.source_seq_arrays.shape[1])
        weights = torch.zeros(sequence_length, dtype=dtype, device=device)
        num_blocks = float(max(int(self.env.num_blocks), 1))
        site_width = num_blocks / float(max(sequence_length, 1))

        for segment_start, segment_end in material_segments.segments:
            start = max(float(segment_start), 0.0)
            end = min(float(segment_end), num_blocks)
            if end <= start:
                continue

            first_site = max(int(math.floor(start / site_width)), 0)
            last_site = min(int(math.ceil(end / site_width)), sequence_length)
            for site_idx in range(first_site, last_site):
                site_start = site_idx * site_width
                site_end = site_start + site_width
                overlap = max(0.0, min(end, site_end) - max(start, site_start))
                if overlap > 0.0:
                    weights[site_idx] = torch.clamp(
                        weights[site_idx] + weights.new_tensor(overlap / site_width),
                        max=1.0,
                    )
        return weights

    def sample(self, logits, random_spec=None):
        if random_spec is None:
            return Categorical(logits=logits).sample()

        temperature = random_spec["T"]
        return Categorical(logits=logits / temperature).sample()


    def compute_log_path_pf(self, logits, action_indices):
        batch_idx = torch.arange(logits.shape[0], device=logits.device)
        log_p = self.logsoftmax(logits)
        return log_p[batch_idx, action_indices]

    def _batched_action_features(self, actions, batch_idx, lineage_reps, summary_reps):
        num_actions = len(actions)
        embedding_size = lineage_reps.shape[-1]

        primary_rep = lineage_reps.new_zeros(num_actions, embedding_size)
        secondary_rep = lineage_reps.new_zeros(num_actions, embedding_size)
        tertiary_rep = lineage_reps.new_zeros(num_actions, embedding_size)

        coal_rows = [(row_idx, action.active_lineage_i, action.active_lineage_j) for row_idx, action in enumerate(actions) if isinstance(action, CoalescenceChoice)]
        if coal_rows:
            rows, left_indices, right_indices = zip(*coal_rows)
            rows = torch.tensor(rows, dtype=torch.long, device=self.device)
            left_indices = torch.tensor(left_indices, dtype=torch.long, device=self.device)
            right_indices = torch.tensor(right_indices, dtype=torch.long, device=self.device)
            left_rep = lineage_reps[batch_idx, left_indices]
            right_rep = lineage_reps[batch_idx, right_indices]
            primary_rep[rows] = left_rep + right_rep
            secondary_rep[rows] = torch.abs(left_rep - right_rep)
            tertiary_rep[rows] = left_rep * right_rep

        recomb_rows = [(row_idx, action.active_lineage_i, action.breakpoint) for row_idx, action in enumerate(actions) if isinstance(action, RecombinationChoice)]

        if recomb_rows:
            rows, lineage_indices, _ = zip(*recomb_rows)
            rows = torch.tensor(rows, dtype=torch.long, device=self.device)
            lineage_indices = torch.tensor(lineage_indices, dtype=torch.long, device=self.device)
            primary_rep[rows] = lineage_reps[batch_idx, lineage_indices]

        summary_for_actions = summary_reps[batch_idx].expand(num_actions, -1)
        return torch.cat([
            primary_rep,
            secondary_rep,
            tertiary_rep,
            summary_for_actions,
        ], dim=-1)

    def _score_candidates(
        self,
        candidate_actions,
        lineage_reps,
        summary_reps
        ):
        batch_size = len(candidate_actions)
        max_candidates = max(len(actions) for actions in candidate_actions)
        logits = lineage_reps.new_full((batch_size, max_candidates), float("-inf"))
        feat_dim = self.seq_embedding.out_features * 4
        features = lineage_reps.new_zeros(batch_size, max_candidates, feat_dim)

        candidate_counts = []

        for batch_idx, actions in enumerate(candidate_actions):
            n = len(actions)
            candidate_counts.append(n)
            state_action_features  = self._batched_action_features(
                actions,
                batch_idx,
                lineage_reps,
                summary_reps
            )
            features[batch_idx, :n] = state_action_features
        logits = self.action_scorer(features.reshape(-1, feat_dim)).reshape(batch_size, max_candidates).squeeze(-1)

        counts = torch.tensor(candidate_counts, device=self.device)
        valid = torch.arange(max_candidates, device=self.device).unsqueeze(0) < counts.unsqueeze(1)
        masked_logits = logits.masked_fill(~valid, -1e9)
        return masked_logits, features

    def _select_breakpoints(self, action):
        """
        Select breakpoints for a given action.
        """
        breakpoint = self.env.rng.choice(range(action.span_start + 1, action.span_end + 1))
        action = replace(action, breakpoint=breakpoint)
        return action

    def _valid_breakpoints_for_action(self, action):
        return list(range(int(action.span_start) + 1, int(action.span_end) + 1))

    def _breakpoint_logit_indices(self, breakpoints, device):
        sequence_length = int(self.env.sequence_length)
        num_blocks = int(self.env.num_blocks)
        indices = []
        for breakpoint in breakpoints:
            site_split = int(round(float(breakpoint) * sequence_length / num_blocks))
            site_split = min(max(site_split, 1), sequence_length - 1)
            indices.append(site_split - 1)
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _sample_recombination_breakpoint(self, action, lineage_seq_feature, random_spec=None):
        valid_breakpoints = self._valid_breakpoints_for_action(action)
        if not valid_breakpoints:
            raise ValueError(f"Recombination action has no valid breakpoints: {action}")

        bp_logits = self.breakpoint_scorer(lineage_seq_feature.unsqueeze(0))[0]
        logit_indices = self._breakpoint_logit_indices(valid_breakpoints, bp_logits.device)
        valid_logits = bp_logits[logit_indices]
        if random_spec is not None and "T" in random_spec:
            sample_logits = valid_logits / random_spec["T"]
        else:
            sample_logits = valid_logits
        local_idx = Categorical(logits=sample_logits).sample()
        breakpoint = int(valid_breakpoints[int(local_idx.detach().cpu().item())])
        log_p = F.log_softmax(valid_logits, dim=0)[local_idx]
        return breakpoint, log_p
    
    def _event_log_probs_from_action_logits(self, candidate_actions, logits):
        event_log_probs = logits.new_full(
            (len(candidate_actions), len(self.env.event_types)),
            float("-inf"),
        )
        normalizers = torch.logsumexp(logits, dim=1)
        for batch_idx, actions in enumerate(candidate_actions):
            for event_idx, event_type in enumerate(self.env.event_types):
                indices = [
                    action_idx
                    for action_idx, action in enumerate(actions)
                    if (
                        (event_type == "coal" and isinstance(action, CoalescenceChoice))
                        or (event_type == "recomb" and isinstance(action, RecombinationChoice))
                    )
                ]
                if indices:
                    event_logits = logits[batch_idx, torch.tensor(indices, device=logits.device)]
                    event_log_probs[batch_idx, event_idx] = (
                        torch.logsumexp(event_logits, dim=0) - normalizers[batch_idx]
                    )
        return event_log_probs



    def forward(self, all_actions, lineage_reps, summary_reps, lineage_seq_features, batch_active_lineage_counts, random_spec):
        
        all_candidate_actions = all_actions


        if any(len(actions) == 0 for actions in all_candidate_actions):
            raise ValueError("ARGModel.forward received a batch item with no candidate actions.")

        logits, action_features = self._score_candidates(
            all_candidate_actions,
            lineage_reps,
            summary_reps,
        )
        
        # Vectorize processing instead of multiple for-loops.

        # Compute lengths of actions per batch and build index tensor
        action_lengths = [len(actions) for actions in all_candidate_actions]
        max_len = max(action_lengths)

        # Create mask for valid actions in logits
        mask = torch.zeros_like(logits, dtype=torch.bool)
        for i, n in enumerate(action_lengths):
            mask[i, :n] = True

        # Build valid logits tensor (invalid entries set to very low value)
        logits_masked = logits.masked_fill(~mask, float('-inf'))

        # Sample actions in a vectorized way
        # In case there are -inf rows in invalid entries, Categorical supports this
        sampled_action_indices = self.sample(logits_masked, random_spec)
        # sampled_action_indices shape: (batch,)

        # Convert to standard Python ints and collect for indexing
        selected_action_indices = sampled_action_indices.detach().cpu().tolist()

        # Now, retrieve chosen actions and features in a single loop
        choosen_actions = []
        choosen_action_features = []
        for batch_idx, action_idx in enumerate(selected_action_indices):
            choosen_actions.append(all_candidate_actions[batch_idx][action_idx])
            choosen_action_features.append(action_features[batch_idx, action_idx])

        # Compute log pf for action scorer (policy) selection
        log_action_pf = self.compute_log_path_pf(logits, selected_action_indices)

        return log_action_pf, selected_action_indices, choosen_actions, choosen_action_features
