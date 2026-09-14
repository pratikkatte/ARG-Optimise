from env import CoalescenceChoice, MaterialSegments, RecombinationChoice
from breakpoint_model import BreakpointSplitPositionCNN, SparseMixtureBreakpointPolicy
from time_model import TimeModel, CwrExponentialTimeModel, CwrGammaTimeModel, validate_continuous_time_head
from time_env import validate_time_policy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass, replace
from operator import index
import math


@dataclass(frozen=True)
class PackedLineageFeatures:
    """Real lineage sequences in batch order, with immutable row boundaries."""

    tensor: torch.Tensor
    row_offsets: tuple[int, ...]

    def __post_init__(self):
        offsets = tuple(index(value) for value in self.row_offsets)
        object.__setattr__(self, "row_offsets", offsets)
        if self.tensor.ndim != 3 or self.tensor.shape[-1] != 4:
            raise ValueError("Packed sequence features must have shape (lineages, num_blocks, 4)")
        if (len(offsets) < 2 or offsets[0] != 0
                or offsets[-1] != self.tensor.shape[0]
                or any(end <= start for start, end in zip(offsets, offsets[1:]))):
            raise ValueError("Packed row offsets must delimit nonempty states and cover all lineages")

    def get_lineage(self, batch_index, lineage_index):
        """Return a sequence view; negative indices and padded rows are invalid."""
        batch_index, lineage_index = index(batch_index), index(lineage_index)
        if not 0 <= batch_index < len(self.row_offsets) - 1:
            raise IndexError(f"Batch index {batch_index} out of bounds")
        start, end = self.row_offsets[batch_index:batch_index + 2]
        if not 0 <= lineage_index < end - start:
            raise IndexError(f"Lineage index {lineage_index} out of bounds for batch {batch_index}")
        return self.tensor[start + lineage_index]


class TransformerMLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        attention_dropout=0.0,
        projection_dropout=0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"embedding_size ({dim}) must be divisible by transformer_heads ({num_heads})"
            )
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x, key_padding_mask=None):
        batch_size, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask[:, None, None, :],
                float("-inf"),
            )
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=2.0,
        dropout=0.0,
        attention_dropout=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
            dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = TransformerMLP(
            dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout=dropout,
        )

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        mlp_ratio=2.0,
        dropout=0.0,
        attention_dropout=0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(int(depth))
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x, key_padding_mask=None):
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


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
        transformer_depth=6,
        transformer_heads=4,
        transformer_mlp_ratio=2.0,
        attention_dropout=0.0,
        time_hidden_size=256,
        time_layers=3,
        time_dropout=0.0,
        breakpoint_gap_hidden_size=256,
        breakpoint_gap_layers=3,
        breakpoint_gap_dropout=0.0,
        breakpoint_use_position_features=True,
        breakpoint_policy="cnn",
        breakpoint_mixture_hidden_dim=128,
        breakpoint_mixture_layers=4,
        breakpoint_mixture_components=4,
        event_policy="cwr",
        time_policy=None,
        continuous_time_head='exponential',
        cache_lineage_features=False,
    ):
        super().__init__()
        if event_policy not in {"cwr", "cwr_residual"}:
            raise ValueError(f"Unknown event_policy: {event_policy}")
        self.event_policy = event_policy
        self.time_policy = validate_time_policy(env.time_policy if time_policy is None else time_policy)
        if self.time_policy != env.time_policy:
            raise ValueError("Environment and model time_policy disagree")
        self.continuous_time_head = validate_continuous_time_head(continuous_time_head, self.time_policy)
        self.env = env
        self.device = env.device
        self.cache_lineage_features = bool(cache_lineage_features)
        if self.cache_lineage_features:
            from lineage_features import LineageFeatureCache
            self.lineage_feature_cache = LineageFeatureCache()
        if int(embedding_size) % int(transformer_heads) != 0:
            raise ValueError(
                "embedding_size must be divisible by transformer_heads "
                f"(got embedding_size={embedding_size}, transformer_heads={transformer_heads})"
            )
        input_size = int(env.num_blocks) * 4

        self.seq_embedding = nn.Linear(input_size, embedding_size)
        self.summary_token = nn.Parameter(torch.zeros(1, 1, embedding_size))
        nn.init.trunc_normal_(self.summary_token, std=0.1)
        self.encoder = TransformerEncoder(
            dim=embedding_size,
            depth=transformer_depth,
            num_heads=transformer_heads,
            mlp_ratio=transformer_mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.action_scorer = nn.Sequential(
            nn.Linear(embedding_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        breakpoint_kwargs = dict(
            dropout=breakpoint_dropout,
            action_context_dim=embedding_size * 4,
            gap_hidden_dim=breakpoint_gap_hidden_size,
            gap_layers=breakpoint_gap_layers,
            gap_dropout=breakpoint_gap_dropout,
        )
        if breakpoint_policy == "cnn":
            self.breakpoint_scorer = BreakpointSplitPositionCNN(
                hidden_dim=breakpoint_hidden_dim,
                use_position_features=breakpoint_use_position_features,
                **breakpoint_kwargs,
            ).to(self.device)
        elif breakpoint_policy == "sparse_mixture":
            self.breakpoint_scorer = SparseMixtureBreakpointPolicy(
                source_alignment=env.block_seq_arrays,
                hidden_dim=breakpoint_mixture_hidden_dim,
                layers=breakpoint_mixture_layers,
                components=breakpoint_mixture_components,
                **breakpoint_kwargs,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown breakpoint_policy: {breakpoint_policy}")

        if self.time_policy == "categorical":
            self.time_scorer = TimeModel(
                embedding_size * 4,
                time_hidden_size,
                time_dropout,
                env.time_env.bins,
                layers=time_layers,
            )
        else:
            time_model_type = CwrGammaTimeModel if self.continuous_time_head == 'gamma' else CwrExponentialTimeModel
            self.time_scorer = time_model_type(
                embedding_size * 4 + 4, time_hidden_size, time_dropout, layers=time_layers,
            )
        self.logsoftmax = nn.LogSoftmax(dim=1)
        if self.event_policy == "cwr_residual":
            self.event_head = nn.Sequential(
                nn.Linear(embedding_size + 3, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 2),
            )
            nn.init.zeros_(self.event_head[-1].weight)
            nn.init.zeros_(self.event_head[-1].bias)

    def event_log_probs(self, states, summary_reps, event_actions, prior_probs):
        """Untempered CwR-residual policy, ordered as (coal, recomb).

        Terminal rows have no forward distribution and return two -infinities.
        Take prior logs before casting so small positive priors do not underflow.
        """
        features = summary_reps.new_tensor([
            [math.log1p(s.current_time), math.log1p(len(s.active_lineages)),
             s.total_active_blocks / self.env.num_blocks] for s in states
        ])
        corrections = self.event_head(torch.cat((summary_reps, features), dim=-1))
        prior_logs = corrections.new_tensor([
            [math.log(p) if p > 0 else -math.inf for p in row] for row in prior_probs
        ])
        valid = torch.tensor([
            [bool(actions) and p > 0 and not state.is_done
             for actions, p in zip(candidates, probabilities)]
            for state, candidates, probabilities in zip(states, event_actions, prior_probs)
        ], dtype=torch.bool, device=corrections.device)
        available = valid.any(dim=-1)
        nonterminal = torch.tensor([not s.is_done for s in states], device=corrections.device)
        if (nonterminal & ~available).any():
            raise ValueError("Nonterminal state has no available event under the CwR prior")
        logits = (prior_logs + corrections).masked_fill(~valid, -math.inf)
        # Avoid undefined log-softmax/gradients on terminal rows.
        logits = torch.where(available[:, None], logits, torch.zeros_like(logits))
        return F.log_softmax(logits, dim=-1).masked_fill(~valid, -math.inf)

    def _build_source_sequence_features(self):
        return self.env.block_seq_arrays.detach().to(dtype=torch.float32).clone()

    def model_params(self):
        return list(self.parameters())

    def _encode_lineage_features(self, lineage_seq_features, batch_active_lineage_counts):
        """Project packed real sequences, then pad only the small embeddings."""
        packed = lineage_seq_features.tensor
        offsets = lineage_seq_features.row_offsets
        if tuple(packed.shape[1:]) != (int(self.env.num_blocks), 4):
            raise ValueError(
                f"Packed sequence features must have shape (lineages, {int(self.env.num_blocks)}, 4), "
                f"got {tuple(packed.shape)}"
            )
        batch_size = len(offsets) - 1
        active_lineages = max(end - start for start, end in zip(offsets, offsets[1:]))
        batch_active_lineage_counts = batch_active_lineage_counts.to(device=self.device, dtype=torch.long)
        valid_mask = (
            torch.arange(active_lineages, device=self.device)[None, :]
            < batch_active_lineage_counts[:, None]
        )
        projected = self.seq_embedding(packed.reshape(packed.shape[0], -1).to(
            device=self.device, dtype=torch.float32,
        ))
        # Zero-padded inputs previously projected to the bias. Preserve those
        # values (and their gradient path) for the unchanged transformer.
        lineage_reps = self.seq_embedding.bias.expand(batch_size, active_lineages, -1).clone()
        lineage_reps[valid_mask] = projected
        summary_token = self.summary_token.expand(batch_size, -1, -1)
        transformer_input = torch.cat([summary_token, lineage_reps], dim=1)
        key_padding_mask = F.pad(~valid_mask, (1, 0), value=False)
        encoded = self.encoder(transformer_input, key_padding_mask=key_padding_mask)
        summary_reps = encoded[:, 0]
        lineage_reps = encoded[:, 1:] * valid_mask.unsqueeze(-1)
        return lineage_reps, summary_reps, lineage_seq_features, batch_active_lineage_counts

    def _encode_states(self, states):
        """Return lineage/summary embeddings, packed sequences, and active counts."""
        if not states:
            raise ValueError("ARGModel.forward requires at least one state")
        active_counts = [len(state.active_lineages) for state in states]
        offsets = [0]
        for batch_index, count in enumerate(active_counts):
            if count == 0:
                raise ValueError(f"State {batch_index} has no active lineages")
            offsets.append(offsets[-1] + count)
        batch_active_lineage_counts = torch.tensor(active_counts, dtype=torch.long, device=self.device)
        packed = torch.empty(
            (offsets[-1], int(self.env.num_blocks), 4), device=self.device, dtype=torch.float32,
        )
        for batch_idx, state in enumerate(states):
            for lineage_idx, lineage in enumerate(state.active_lineages):
                if self.cache_lineage_features:
                    feature = self.lineage_feature_cache.get(
                        lineage, self.device, self.env.num_blocks,
                        lambda: self._normalized_lineage_feature(lineage))
                else:
                    feature = self._normalized_lineage_feature(lineage)
                packed[offsets[batch_idx] + lineage_idx] = feature
        return self._encode_lineage_features(
            PackedLineageFeatures(packed, tuple(offsets)), batch_active_lineage_counts,
        )

    def _normalized_lineage_feature(self, lineage):
        feature = self._lineage_partials_tensor(lineage)
        weights = self._material_segments_masking(
            lineage.material_segments, device=self.device, dtype=self.env.block_seq_arrays.dtype)
        return self.env.evolution_model.normalize_partials(feature * weights[:, None])

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
        expected_shape = (int(self.env.num_blocks), 4)
        if tuple(partials.shape) != expected_shape:
            raise ValueError(
                f"Active ARG lineage {lineage.node_id} partials must have shape "
                f"{expected_shape}, got {tuple(partials.shape)}"
            )
        return partials

    def _material_segments_masking(self, material_segments, device, dtype):
        num_blocks = int(self.env.num_blocks)
        weights = torch.zeros(num_blocks, dtype=dtype, device=device)

        for segment_start, segment_end in material_segments.segments:
            start = max(int(segment_start), 0)
            end = min(int(segment_end), num_blocks)
            if end <= start:
                continue
            weights[start:end] = 1.0
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
        # Keep both axes even when every state has exactly one candidate.
        # Squeezing [B, 1] into [B] makes the [B, 1] mask broadcast it to
        # [B, B], mixing other states' scores into the action normalization.
        logits = self.action_scorer(features.reshape(-1, feat_dim)).reshape(batch_size, max_candidates)

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
        num_blocks = int(self.env.num_blocks)
        indices = []
        for breakpoint in breakpoints:
            index = min(max(int(breakpoint), 1), num_blocks - 1) - 1
            indices.append(index)
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _sample_recombination_breakpoint(self, action, lineage_seq_feature, action_context, random_spec=None):
        return self.breakpoint_scorer(
            action,
            lineage_seq_feature,
            int(self.env.sequence_length),
            int(self.env.num_blocks),
            action_context,
            random_spec=random_spec,
        )

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



    def forward(self, all_actions, lineage_reps, summary_reps, lineage_seq_features, batch_active_lineage_counts, random_spec,
                action_indices=None):
        """Score actions; lineage_seq_features is internally PackedLineageFeatures."""
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
        sampled_action_indices = (self.sample(logits_masked, random_spec) if action_indices is None
                                  else torch.as_tensor(action_indices, device=logits.device, dtype=torch.long))
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
        log_action_pf = self.compute_log_path_pf(logits_masked, selected_action_indices)

        return log_action_pf, selected_action_indices, choosen_actions, choosen_action_features
