import math

from env import CoalescenceChoice, MaterialSegments, RecombinationChoice
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import replace

CHARACTERS_MAPS = {
    'DNA_WITH_GAP': {
        'A': [1., 0., 0., 0.],
        'C': [0., 1., 0., 0.],
        'G': [0., 0., 1., 0.],
        'T': [0., 0., 0., 1.],
        '-': [1., 1., 1., 1.],
        'N': [1., 1., 1., 1.],
    }
}

VOCAB_NAME = 'DNA_WITH_GAP'
TOKEN_TO_FEATURES = CHARACTERS_MAPS[VOCAB_NAME]

# Stable index <-> token maps derived from CHARACTERS_MAPS insertion order.
INDEX_TO_TOKEN = {idx: token for idx, token in enumerate(TOKEN_TO_FEATURES.keys())}
TOKEN_TO_INDEX = {token: idx for idx, token in INDEX_TO_TOKEN.items()}
TOKEN_FEATURES = torch.tensor(
    [TOKEN_TO_FEATURES[INDEX_TO_TOKEN[idx]] for idx in range(len(INDEX_TO_TOKEN))],
    dtype=torch.float32,
)
VOCAB_SIZE = len(INDEX_TO_TOKEN)

class ResidualDilatedConvBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError('kernel_size must be odd to preserve length with symmetric padding')
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, L]
        residual = x
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        return residual + x

class BreakpointSplitPositionCNN(nn.Module):
    def __init__(
        self,
        input_dim=4,
        hidden_dim=128,
        dilations=None,
        dropout=0.1,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128] * 2

        self.feature_dim = 4
        self.input_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3)
        self.input_activation = nn.GELU()

        self.blocks = nn.ModuleList(
            ResidualDilatedConvBlock(
                hidden_dim=hidden_dim,
                kernel_size=5,
                dilation=dilation,
                dropout=dropout,
            )
            for dilation in dilations
        )

        self.scoring_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x):
        # x: [B, L, 4]
        if x.ndim != 3:
            raise ValueError(f'expected input shape [B, L, {self.feature_dim}], got {tuple(x.shape)}')
        if x.shape[-1] != self.feature_dim:
            raise ValueError(f'expected final feature dimension {self.feature_dim}, got {x.shape[-1]}')
        x = x.float().transpose(1, 2)  # [B, 4, L]
        x = self.input_conv(x)         # [B, C, L]
        x = self.input_activation(x)

        for block in self.blocks:
            x = block(x)              # [B, C, L]

        scores = self.scoring_head(x).squeeze(1)  # [B, L]

        # Keep scores for valid split gaps only. Logit i corresponds to split k=i+1.
        logits = scores[:, :-1].contiguous()      # [B, L - 1]
        return logits


class ARGModel(nn.Module):
    """One-step ARG action policy.

    The model scores candidate coalescent and recombination actions. When the
    caller provides current ARG states, candidates are read from the environment
    so material-mask constraints are respected.
    """

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.device = env.device
        embedding_size = 32
        hidden_size = 64
        dropout = 0.0
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
        self.breakpoint_scorer = BreakpointSplitPositionCNN().to(self.device)

        self.time_scorer = nn.Sequential(
            nn.Linear(embedding_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, env.time_env.bins),
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def _build_source_sequence_features(self):
        return self.env.seq_arrays.detach().to(dtype=torch.float32).clone()

    def model_params(self):
        return list(self.parameters())

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

    def _encode_lineage_features(self, lineage_seq_features, batch_active_lineage_counts):
        """
        """

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

    def sample(self, logits, random_spec=None):
        if random_spec is None:
            return Categorical(logits=logits).sample()

        temperature = random_spec["T"]
        return Categorical(logits=logits / temperature).sample()


    def compute_log_path_pf(self, logits, action_indices):
        batch_idx = torch.arange(logits.shape[0], device=logits.device)
        log_p = self.logsoftmax(logits)
        return log_p[batch_idx, action_indices]


    def compute_log_time_pf(self, time_logits, time_actions):
        batch_idx = torch.arange(time_logits.shape[0], device=time_logits.device)
        log_p = self.logsoftmax(time_logits)
        return log_p[batch_idx, time_actions]


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
        return logits.masked_fill(~valid, float("-inf")), action_features

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



    def forward(self, input_dict):
        states = input_dict.get("states")
        if states is None:
            raise ValueError("States are required for the model to run.")
            
        random_spec = input_dict.get("random_spec")
        event = input_dict.get("event")
        log_event_pf = [ -math.log(event[idx]["probability"]) for idx, event in event.items()]
        
        lineage_reps, summary_reps, lineage_seq_features, batch_active_lineage_counts = self._encode_states(states)

        
        all_candidate_actions = input_dict.get("input_actions")

        print("batch_active_lineage_counts", batch_active_lineage_counts)


        if any(len(actions) == 0 for actions in all_candidate_actions):
            raise ValueError("ARGModel.forward received a batch item with no candidate actions.")

        logits, action_features = self._score_candidates(
            all_candidate_actions,
            lineage_reps,
            summary_reps,
        )
        
        choosen_action_indices = self.sample(logits, random_spec)
        selected_action_indices = [
            int(action_idx.detach().cpu().item())
            for action_idx in choosen_action_indices
        ]

        

        choosen_actions = [
            all_candidate_actions[batch_idx][action_idx]
            for batch_idx, action_idx in enumerate(selected_action_indices)
        ]

        choosen_action_features = [
            action_features[batch_idx, action_idx]
            for batch_idx, action_idx in enumerate(selected_action_indices)
        ]

        # Compute log pf for action scorer (policy) selection
        log_action_pf = self.compute_log_path_pf(logits, choosen_action_indices)

        # Compute log pf for breakpoints (for recombination actions)
        log_p_breakpoints = []
        for idx, chosen_action in enumerate(choosen_actions):
            if isinstance(chosen_action, RecombinationChoice):
                lineage_idx = int(chosen_action.active_lineage_i)
                lineage_feature = lineage_seq_features[idx, lineage_idx]
                breakpoint, log_p_breakpoint = self._sample_recombination_breakpoint(
                    chosen_action,
                    lineage_feature,
                    random_spec=random_spec,
                )
                choosen_actions[idx] = replace(chosen_action, breakpoint=breakpoint)
                log_p_breakpoints.append(log_p_breakpoint)
           
            else:
                log_p_breakpoints.append(logits.new_zeros(()))

        log_breakpoint_pf = torch.stack(log_p_breakpoints)

        # Score the action features for each chosen action to obtain time logits.
        # selected_action_features = []
        # for batch_idx, action_idx in enumerate(selected_action_indices):
        #     action_features = self._batched_action_features(
        #         [all_candidate_actions[batch_idx][action_idx]],
        #         batch_idx,
        #         lineage_reps,
        #         summary_reps,
        #     )
        #     selected_action_features.append(action_features.squeeze(0))

        selected_action_features = torch.stack(choosen_action_features, dim=0)  # shape: [B, F]
        time_logits = self.time_scorer(selected_action_features)
        time_actions = self.sample(time_logits, random_spec)

        for batch_idx, action in enumerate(choosen_actions):
            time = int(time_actions[batch_idx].detach().cpu().item())
            choosen_actions[batch_idx] = replace(action, time_action=time)

        log_time_pf = self.compute_log_time_pf(time_logits, time_actions)

        total_log_pf = log_event_pf + log_action_pf + log_breakpoint_pf + log_time_pf

        log_probs = torch.exp(total_log_pf)
        
        return total_log_pf, log_probs, choosen_actions
