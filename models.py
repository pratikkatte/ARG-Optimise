try:
    from .env import CoalescenceChoice, MaterialSegments, RecombinationChoice
    from .breakpoint_model import BreakpointSplitPositionCNN, VCFBreakpointScorer
    from .recombination_split_bias import (
        RecombinationSplitBiasScorer,
        normalize_recombination_split_bias_config,
    )
    from .cwr_event_gate import normalize_local_cwr_event_gate_config
    from .time_model import TimeModel
    from .time_env import DEFAULT_TIME_BASIS_COMPONENTS
    from .time_context import (
        TIME_CONTEXT_VERSION,
        build_time_context,
        time_context_dim,
        time_context_feature_names,
    )
except ImportError:  # Support the repository's script-style entry points.
    from env import CoalescenceChoice, MaterialSegments, RecombinationChoice
    from breakpoint_model import BreakpointSplitPositionCNN, VCFBreakpointScorer
    from recombination_split_bias import (
        RecombinationSplitBiasScorer,
        normalize_recombination_split_bias_config,
    )
    from cwr_event_gate import normalize_local_cwr_event_gate_config
    from time_model import TimeModel
    from time_env import DEFAULT_TIME_BASIS_COMPONENTS
    from time_context import (
        TIME_CONTEXT_VERSION,
        build_time_context,
        time_context_dim,
        time_context_feature_names,
    )
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class CandidateScoringResult:
    """Shared candidate-aligned policy data for sampling and replay."""

    base_logits: torch.Tensor
    probability_logits: torch.Tensor
    action_features: torch.Tensor
    valid_mask: torch.Tensor
    split_records: tuple[tuple[object | None, ...], ...]
    diagnostics: tuple[dict, ...]


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

        attn_mask = None
        if key_padding_mask is not None:
            # SDPA boolean masks use True for keys that are allowed to
            # participate in attention, the inverse of key_padding_mask.
            attn_mask = ~key_padding_mask[:, None, None, :]
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )
        x = x.transpose(1, 2).reshape(batch_size, tokens, dim)
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


class VariantTokenEncoder(nn.Module):
    def __init__(self, input_dim, embedding_size, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size),
            nn.GELU(),
        )

    def forward(self, token_features):
        return self.net(token_features)


class SparseLineageEncoder(nn.Module):
    TOKEN_FEATURE_DIM = 9
    LINEAGE_FEATURE_DIM = 5

    def __init__(self, env, embedding_size, dropout=0.0):
        super().__init__()
        self.env = env
        self.embedding_size = int(embedding_size)
        self.token_encoder = VariantTokenEncoder(
            self.TOKEN_FEATURE_DIM,
            self.embedding_size,
            dropout=dropout,
        )
        self.lineage_encoder = nn.Sequential(
            nn.Linear(self.LINEAGE_FEATURE_DIM, self.embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_size, self.embedding_size),
        )
        self.attention_query = nn.Parameter(torch.zeros(self.embedding_size))
        nn.init.trunc_normal_(self.attention_query, std=0.02)
        self.output_norm = nn.LayerNorm(self.embedding_size, eps=1e-6)

    def forward(self, token_features, token_mask, lineage_features):
        token_embeddings = self.token_encoder(token_features)
        scores = torch.matmul(token_embeddings, self.attention_query)
        scores = scores / math.sqrt(float(self.embedding_size))
        scores = scores.masked_fill(~token_mask, float("-inf"))

        empty = ~token_mask.any(dim=1)
        if empty.any():
            scores = scores.masked_fill(empty[:, None], 0.0)

        weights = F.softmax(scores, dim=1)
        weights = weights.masked_fill(~token_mask, 0.0)
        pooled = torch.sum(token_embeddings * weights.unsqueeze(-1), dim=1)
        lineage_embedding = self.lineage_encoder(lineage_features)
        return self.output_norm(pooled + lineage_embedding)


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
        time_basis_components=DEFAULT_TIME_BASIS_COMPONENTS,
        time_context_mode="baseline",
        breakpoint_gap_hidden_size=256,
        breakpoint_gap_layers=3,
        breakpoint_gap_dropout=0.0,
        breakpoint_use_position_features=True,
        local_coalescence_similarity_bias=0.0,
        local_prior_action_logit_bias=0.0,
        local_prior_gate_logit_bias=0.0,
        recombination_split_bias=None,
        local_cwr_event_gate=None,
    ):
        super().__init__()
        self.env = env
        self.device = env.device
        self.input_mode = getattr(env, "input_mode", "dense")
        self.local_mode = bool(getattr(env, "is_local", False))
        self.embedding_size = int(embedding_size)
        self.local_coalescence_similarity_bias = float(
            local_coalescence_similarity_bias
        )
        self.local_prior_action_logit_bias = float(local_prior_action_logit_bias)
        self.local_prior_gate_logit_bias = float(local_prior_gate_logit_bias)
        self.recombination_split_bias_config = (
            normalize_recombination_split_bias_config(
                recombination_split_bias
            )
        )
        self.recombination_split_bias = RecombinationSplitBiasScorer(
            env,
            self.recombination_split_bias_config,
        )
        self.local_cwr_event_gate_config = normalize_local_cwr_event_gate_config(
            local_cwr_event_gate
        )
        if self.local_cwr_event_gate_config["enabled"] and not (
            self.local_mode and self.input_mode == "vcf"
        ):
            raise ValueError(
                "model.local_cwr_event_gate is supported only for local VCF "
                "ARG refinement"
            )
        self.time_context_mode = str(time_context_mode).lower()
        self.time_context_dim = time_context_dim(self.time_context_mode)
        self.time_context_version = TIME_CONTEXT_VERSION
        if int(embedding_size) % int(transformer_heads) != 0:
            raise ValueError(
                "embedding_size must be divisible by transformer_heads "
                f"(got embedding_size={embedding_size}, transformer_heads={transformer_heads})"
            )
        if self.input_mode == "vcf":
            self.seq_embedding = None
            self.sparse_lineage_encoder = SparseLineageEncoder(
                env,
                embedding_size=embedding_size,
                dropout=dropout,
            )
        else:
            input_size = int(env.num_blocks) * 4
            self.seq_embedding = nn.Linear(input_size, embedding_size)
            self.sparse_lineage_encoder = None
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
        self.flow_head = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        if self.local_mode:
            self.local_context_encoder = nn.Sequential(
                nn.Linear(8, embedding_size),
                nn.GELU(),
                nn.Linear(embedding_size, embedding_size),
            )
            self.local_role_embedding = nn.Embedding(4, embedding_size)
            self.local_lineage_time_encoder = nn.Linear(2, embedding_size)
            self.local_transition_gate = nn.Linear(embedding_size, 2)
            if self.local_cwr_event_gate_config["enabled"]:
                # Keep every pre-existing parameter initialization identical in
                # matched enabled/disabled runs.  Constructing an nn.Linear
                # normally advances the global RNG even though this head is
                # immediately zero-initialized.
                with torch.random.fork_rng(devices=[]):
                    self.local_cwr_event_residual_head = nn.Linear(
                        embedding_size,
                        1,
                    )
                nn.init.zeros_(self.local_cwr_event_residual_head.weight)
                nn.init.zeros_(self.local_cwr_event_residual_head.bias)
            else:
                self.local_cwr_event_residual_head = None
        else:
            self.local_context_encoder = None
            self.local_role_embedding = None
            self.local_lineage_time_encoder = None
            self.local_transition_gate = None
            self.local_cwr_event_residual_head = None
        if self.input_mode == "vcf":
            self.breakpoint_scorer = VCFBreakpointScorer(
                env,
                hidden_dim=breakpoint_hidden_dim,
                action_context_dim=embedding_size * 4,
                gap_hidden_dim=breakpoint_gap_hidden_size,
                gap_layers=breakpoint_gap_layers,
                gap_dropout=breakpoint_gap_dropout,
            ).to(self.device)
        else:
            self.breakpoint_scorer = BreakpointSplitPositionCNN(
                hidden_dim=breakpoint_hidden_dim,
                dropout=breakpoint_dropout,
                action_context_dim=embedding_size * 4,
                gap_hidden_dim=breakpoint_gap_hidden_size,
                gap_layers=breakpoint_gap_layers,
                gap_dropout=breakpoint_gap_dropout,
                use_position_features=breakpoint_use_position_features,
            ).to(self.device)

        self.time_scorer = TimeModel(
            embedding_size * 4
            + TimeModel.CONTEXT_FEATURE_DIM
            + self.time_context_dim,
            time_hidden_size,
            time_dropout,
            time_basis_components,
            layers=time_layers,
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    @property
    def time_context_feature_names(self):
        return time_context_feature_names(self.time_context_mode)

    def build_time_context(
        self,
        states,
        selected_actions,
        max_deltas,
        *,
        dtype,
    ):
        """Return post-breakpoint biological time features and diagnostics."""

        if not (
            len(states) == len(selected_actions) == len(max_deltas)
        ):
            raise ValueError(
                "states, selected_actions, and max_deltas must have equal length"
            )
        contexts = [
            build_time_context(
                state,
                action,
                self.env,
                max_delta=max_delta,
                mode=self.time_context_mode,
                device=self.device,
                dtype=dtype,
            )
            for state, action, max_delta in zip(
                states,
                selected_actions,
                max_deltas,
            )
        ]
        return (
            torch.stack([context.features for context in contexts], dim=0),
            [dict(context.diagnostics) for context in contexts],
        )

    def _build_source_sequence_features(self):
        return self.env.block_seq_arrays.detach().to(dtype=torch.float32).clone()

    def model_params(self):
        return list(self.parameters())

    def compute_log_state_flows(self, summary_reps):
        return self.flow_head(summary_reps).squeeze(-1)

    def compute_local_gate_logits(self, summary_reps):
        if self.local_transition_gate is None:
            raise ValueError("local transition gate is unavailable in global mode")
        return self.local_transition_gate(summary_reps)

    def compute_local_cwr_event_residual(self, summary_reps):
        """Return the bounded posterior correction to the CwR log odds."""

        if self.local_cwr_event_residual_head is None:
            raise ValueError("local CwR event gate is disabled")
        raw = self.local_cwr_event_residual_head(summary_reps).squeeze(-1)
        bound = float(
            self.local_cwr_event_gate_config["max_abs_residual"]
        )
        return bound * torch.tanh(raw / bound), raw

    def _local_context_features(self, states):
        rows = []
        sequence_length = max(float(self.env.sequence_length), 1.0)
        for state in states:
            target = state.local_target_interval
            if target is None:
                target = (0.0, sequence_length)
            left, right = (float(target[0]), float(target[1]))
            current_time = max(float(state.current_time), 0.0)
            initial_time = max(float(state.local_initial_time), 0.0)
            cut_time = max(
                float(
                    state.local_cut_time
                    if state.local_cut_time is not None
                    else initial_time
                ),
                0.0,
            )
            remaining_fixed = [
                record
                for record in state.fixed_ancestor_schedule
                if int(record["node_id"]) not in state.all_nodes
            ]
            next_fixed = min(
                (float(record["time"]) for record in remaining_fixed),
                default=current_time,
            )
            target_count = max(
                int(state.target_material.count)
                if state.target_material is not None
                else 0,
                1,
            )
            carried_count = sum(
                lineage.material_segments.intersection(
                    state.target_material
                ).count
                for lineage in state.active_lineages
            ) if state.target_material is not None else 0
            rows.append(
                [
                    left / sequence_length,
                    right / sequence_length,
                    max(right - left, 0.0) / sequence_length,
                    math.log1p(cut_time),
                    math.log1p(max(current_time - initial_time, 0.0)),
                    math.log1p(max(next_fixed - current_time, 0.0)),
                    float(len(remaining_fixed))
                    / max(float(len(state.fixed_ancestor_schedule)), 1.0),
                    float(carried_count) / float(target_count),
                ]
            )
        return torch.as_tensor(
            rows,
            dtype=torch.float32,
            device=self.device,
        )

    def _augment_local_lineages(self, encoded_lineages, lineages, states):
        if not self.local_mode:
            return encoded_lineages
        role_indices = []
        time_features = []
        for lineage, state in zip(lineages, states):
            generated_start = state.generated_node_start
            if str(lineage.event_type) == "fixed_source":
                role_index = 3
            elif (
                generated_start is None
                or int(lineage.node_id) < int(generated_start)
            ):
                role_index = 0
            elif str(lineage.event_type) == "recomb":
                role_index = 2
            else:
                role_index = 1
            role_indices.append(role_index)
            current_time = max(float(state.current_time), 1e-12)
            lineage_time = max(float(lineage.time), 0.0)
            time_features.append(
                [
                    math.log1p(lineage_time),
                    (lineage_time - current_time) / current_time,
                ]
            )
        roles = torch.as_tensor(
            role_indices,
            dtype=torch.long,
            device=self.device,
        )
        times = torch.as_tensor(
            time_features,
            dtype=encoded_lineages.dtype,
            device=self.device,
        )
        return (
            encoded_lineages
            + self.local_role_embedding(roles)
            + self.local_lineage_time_encoder(times)
        )

    def _encode_lineage_features(self, lineage_seq_features, batch_active_lineage_counts):
        batch_size, active_lineages, seq_len, channels = lineage_seq_features.shape
        if seq_len != int(self.env.num_blocks) or channels != 4:
            raise ValueError(
                "sequence features must have shape "
                f"(batch, active_lineages, {int(self.env.num_blocks)}, 4), "
                f"got {tuple(lineage_seq_features.shape)}"
            )

        batch_input = lineage_seq_features.reshape(batch_size, active_lineages, -1)
        if batch_input.shape[-1] != self.seq_embedding.in_features:
            raise ValueError(
                "Encoded batch_input last dimension must match num_blocks * 4 "
                f"({self.seq_embedding.in_features}), got {batch_input.shape[-1]}"
            )

        batch_input = batch_input.to(device=self.device, dtype=torch.float32)
        batch_active_lineage_counts = batch_active_lineage_counts.to(device=self.device, dtype=torch.long)

        valid_mask = (
            torch.arange(active_lineages, device=self.device)[None, :]
            < batch_active_lineage_counts[:, None]
        )
        lineage_reps = self.seq_embedding(batch_input)
        summary_token = self.summary_token.expand(batch_size, -1, -1)
        transformer_input = torch.cat([summary_token, lineage_reps], dim=1)

        key_padding_mask = F.pad(~valid_mask, (1, 0), value=False)
        encoded = self.encoder(transformer_input, key_padding_mask=key_padding_mask)

        summary_reps = encoded[:, 0]
        lineage_reps = encoded[:, 1:]
        lineage_reps = lineage_reps * valid_mask.unsqueeze(-1)
        return lineage_reps, summary_reps, lineage_seq_features, batch_active_lineage_counts

    def _encode_states(
        self,
        states,
        visible_lineage_indices_by_state=None,
    ):
        if self.input_mode == "vcf":
            return self._encode_sparse_states(
                states,
                visible_lineage_indices_by_state=(
                    visible_lineage_indices_by_state
                ),
            )
        if visible_lineage_indices_by_state is not None:
            raise ValueError(
                "visible lineage filtering is only supported in VCF mode"
            )

        batch_size = len(states)
        if batch_size == 0:
            raise ValueError("ARGModel.forward requires at least one state")

        active_counts = [len(state.active_lineages) for state in states]
        batch_active_lineage_counts = torch.tensor(
            active_counts, dtype=torch.long, device=self.device,
        )

        num_blocks = self.env.num_blocks
        max_active_lineages = max(active_counts, default=0)
        lineage_seq_features = self.env.block_seq_arrays.new_zeros(
            batch_size,
            max_active_lineages,
            num_blocks,
            4,
        )

        for batch_idx, state in enumerate(states):
            for lineage_idx, lineage in enumerate(state.active_lineages):
                feature = self._lineage_partials_tensor(lineage)
                weights = self._material_segments_masking(
                    lineage.material_segments,
                    device=self.device,
                    dtype=self.env.block_seq_arrays.dtype,
                )
                masked_feature = feature * weights[:, None]
                lineage_seq_features[batch_idx, lineage_idx] = (
                    self.env.evolution_model.normalize_partials(masked_feature)
                )

        return self._encode_lineage_features(lineage_seq_features, batch_active_lineage_counts)

    def _encode_sparse_states(
        self,
        states,
        visible_lineage_indices_by_state=None,
    ):
        batch_size = len(states)
        if batch_size == 0:
            raise ValueError("ARGModel.forward requires at least one state")

        active_counts = [len(state.active_lineages) for state in states]
        max_active_lineages = max(active_counts, default=0)
        if visible_lineage_indices_by_state is None:
            visible_lineage_indices_by_state = [
                tuple(range(len(state.active_lineages)))
                for state in states
            ]
        if len(visible_lineage_indices_by_state) != batch_size:
            raise ValueError(
                "visible_lineage_indices_by_state must contain one entry "
                "per state"
            )
        lineages = []
        lineage_states = []
        for state, requested_indices in zip(
            states,
            visible_lineage_indices_by_state,
        ):
            normalized = tuple(int(value) for value in requested_indices)
            if len(set(normalized)) != len(normalized):
                raise ValueError("visible lineage indices must be unique")
            for lineage_index in normalized:
                if not 0 <= lineage_index < len(state.active_lineages):
                    raise ValueError(
                        "visible lineage index is outside the active frontier"
                    )
                lineages.append(state.active_lineages[lineage_index])
                lineage_states.append(state)
        if not lineages:
            raise ValueError("ARGModel.forward requires at least one active lineage")

        token_features, token_mask, lineage_features = (
            self._sparse_lineage_batch_features(
                lineages,
                lineage_states=lineage_states,
            )
        )
        encoded_lineages = self.sparse_lineage_encoder(
            token_features,
            token_mask,
            lineage_features,
        )
        encoded_lineages = self._augment_local_lineages(
            encoded_lineages,
            lineages,
            lineage_states,
        )

        lineage_reps = encoded_lineages.new_zeros(
            batch_size,
            max_active_lineages,
            self.embedding_size,
        )
        cursor = 0
        for batch_idx, requested_indices in enumerate(
            visible_lineage_indices_by_state
        ):
            for lineage_index in requested_indices:
                lineage_reps[batch_idx, int(lineage_index)] = (
                    encoded_lineages[cursor]
                )
                cursor += 1

        batch_active_lineage_counts = torch.tensor(
            active_counts,
            dtype=torch.long,
            device=self.device,
        )
        visible_mask = torch.zeros(
            batch_size,
            max_active_lineages,
            dtype=torch.bool,
            device=self.device,
        )
        for batch_idx, requested_indices in enumerate(
            visible_lineage_indices_by_state
        ):
            if requested_indices:
                visible_mask[
                    batch_idx,
                    torch.as_tensor(
                        requested_indices,
                        dtype=torch.long,
                        device=self.device,
                    ),
                ] = True
        summary_token = self.summary_token.expand(batch_size, -1, -1)
        if self.local_mode:
            context_embedding = self.local_context_encoder(
                self._local_context_features(states)
            )
            summary_token = summary_token + context_embedding[:, None, :]
        transformer_input = torch.cat([summary_token, lineage_reps], dim=1)
        key_padding_mask = F.pad(~visible_mask, (1, 0), value=False)
        encoded = self.encoder(transformer_input, key_padding_mask=key_padding_mask)

        summary_reps = encoded[:, 0]
        lineage_reps = encoded[:, 1:] * visible_mask.unsqueeze(-1)
        return lineage_reps, summary_reps, states, batch_active_lineage_counts

    def _env_float_tensor(self, attr, fallback):
        value = getattr(self.env, attr, None)
        if value is None:
            return torch.as_tensor(fallback, device=self.device, dtype=torch.float32)
        return value.to(device=self.device, dtype=torch.float32)

    def _sparse_lineage_batch_intervals(
        self,
        lineages,
        lineage_states=None,
    ):
        if lineage_states is None:
            lineage_states = [None] * len(lineages)
        starts = []
        ends = []
        for lineage, state in zip(lineages, lineage_states):
            if state is not None and state.block_boundaries is not None:
                boundaries = torch.as_tensor(
                    state.block_boundaries,
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                boundaries = self._env_float_tensor(
                    "variant_boundary_tensor",
                    self.env.variant_boundaries,
                )
            max_boundary_idx = max(int(boundaries.numel()) - 1, 0)
            span_start = lineage.material_segments.span_start
            span_end = lineage.material_segments.span_end
            if span_start is None or span_end is None:
                start_index = 0
                end_index = max_boundary_idx
            else:
                start_index = min(
                    max(int(span_start), 0),
                    max_boundary_idx,
                )
                end_index = min(
                    max(int(span_end) + 1, 0),
                    max_boundary_idx,
                )
            starts.append(boundaries[int(start_index)])
            ends.append(boundaries[int(end_index)])
        starts = torch.stack(starts)
        ends = torch.stack(ends)
        widths = (ends - starts).clamp_min(1.0)
        return starts, ends, widths

    def _sparse_lineage_batch_features(
        self,
        lineages,
        lineage_states=None,
    ):
        if not lineages:
            raise ValueError("ARGModel.forward requires at least one active lineage")

        count_values = [
            int(len(lineage.variant_indices)) for lineage in lineages
        ]
        max_tokens = max(max(count_values, default=0), 1)
        num_lineages = len(lineages)
        token_features = torch.zeros(
            num_lineages,
            max_tokens,
            SparseLineageEncoder.TOKEN_FEATURE_DIM,
            dtype=torch.float32,
            device=self.device,
        )
        token_mask = torch.zeros(
            num_lineages,
            max_tokens,
            dtype=torch.bool,
            device=self.device,
        )

        interval_starts, interval_ends, interval_widths = (
            self._sparse_lineage_batch_intervals(
                lineages,
                lineage_states=lineage_states,
            )
        )
        seq_len = max(float(self.env.sequence_length), 1.0)
        count_tensor = torch.tensor(count_values, dtype=torch.float32, device=self.device)
        count_norm = count_tensor / max(float(self.env.num_blocks), 1.0)
        length_norm = interval_widths / seq_len
        density = count_norm / length_norm.clamp_min(1e-6)
        lineage_features = torch.stack(
            [
                interval_starts / seq_len,
                interval_ends / seq_len,
                length_norm,
                count_norm,
                density,
            ],
            dim=1,
        )

        block_rows = []
        partial_rows = []
        lineage_ids = []
        token_positions = []
        for lineage_idx, lineage in enumerate(lineages):
            partials = self._lineage_partials_tensor(lineage)
            count = count_values[lineage_idx]
            if count == 0:
                continue
            block_index = torch.as_tensor(
                lineage.variant_indices,
                dtype=torch.long,
                device=self.device,
            )
            if int(block_index.numel()) != count:
                raise ValueError(
                    f"Active ARG lineage {lineage.node_id} VCF row count "
                    f"({count}) does not match variant-index count "
                    f"({int(block_index.numel())})"
                )
            block_rows.append(block_index)
            partial_rows.append(partials)
            lineage_ids.append(
                torch.full((count,), lineage_idx, dtype=torch.long, device=self.device)
            )
            token_positions.append(torch.arange(count, dtype=torch.long, device=self.device))

        if not block_rows:
            return token_features, token_mask, lineage_features

        all_blocks = torch.cat(block_rows, dim=0)
        all_partials = torch.cat(partial_rows, dim=0)
        all_lineage_ids = torch.cat(lineage_ids, dim=0)
        all_token_positions = torch.cat(token_positions, dim=0)

        positions = self.env.variant_position_tensor.index_select(0, all_blocks).to(dtype=torch.float32)
        if lineage_states is not None:
            coordinate_offsets = torch.tensor(
                [
                    float(
                        getattr(state, "vcf_alignment", {}).get(
                            "vcf_coordinate_offset",
                            0.0,
                        )
                    )
                    if state is not None
                    else 0.0
                    for state in lineage_states
                ],
                dtype=torch.float32,
                device=self.device,
            )
            positions = positions + coordinate_offsets.index_select(
                0,
                all_lineage_ids,
            )
        prev_gaps = self.env.variant_prev_gap_tensor.index_select(0, all_blocks).to(dtype=torch.float32)
        next_gaps = self.env.variant_next_gap_tensor.index_select(0, all_blocks).to(dtype=torch.float32)
        abs_pos = positions / seq_len
        rel_pos = (
            positions - interval_starts.index_select(0, all_lineage_ids)
        ) / interval_widths.index_select(0, all_lineage_ids).clamp_min(1.0)
        carried = torch.ones_like(abs_pos)
        packed_features = torch.cat(
            [
                all_partials,
                abs_pos[:, None],
                rel_pos[:, None],
                (prev_gaps / seq_len)[:, None],
                (next_gaps / seq_len)[:, None],
                carried[:, None],
            ],
            dim=1,
        )

        token_features[all_lineage_ids, all_token_positions] = packed_features
        token_mask[all_lineage_ids, all_token_positions] = True
        return token_features, token_mask, lineage_features

    def _sparse_lineage_features(self, lineage, max_tokens):
        partials = self._lineage_partials_tensor(lineage)
        token_features = partials.new_zeros(max_tokens, SparseLineageEncoder.TOKEN_FEATURE_DIM)
        token_mask = torch.zeros(max_tokens, dtype=torch.bool, device=partials.device)

        block_index = torch.as_tensor(
            lineage.variant_indices,
            dtype=torch.long,
            device=self.device,
        )
        count = int(block_index.numel())
        if count > 0:
            positions = self.env.variant_position_tensor.index_select(0, block_index).to(dtype=torch.float32)
            prev_gaps = self.env.variant_prev_gap_tensor.index_select(0, block_index).to(dtype=torch.float32)
            next_gaps = self.env.variant_next_gap_tensor.index_select(0, block_index).to(dtype=torch.float32)

            seq_len = max(float(self.env.sequence_length), 1.0)
            abs_pos = positions / seq_len
            interval_start, interval_end, interval_width = self._lineage_physical_interval(lineage)
            rel_pos = (positions - interval_start) / max(interval_width, 1.0)
            carried = torch.ones_like(abs_pos)
            row_features = torch.cat(
                [
                    partials,
                    abs_pos[:, None],
                    rel_pos[:, None],
                    (prev_gaps / seq_len)[:, None],
                    (next_gaps / seq_len)[:, None],
                    carried[:, None],
                ],
                dim=1,
            )
            token_features[:count] = row_features
            token_mask[:count] = True

        lineage_features = self._lineage_interval_features(lineage, count, partials.device)
        return token_features, token_mask, lineage_features

    def _lineage_physical_interval(self, lineage):
        if lineage.material_segments.span_start is None or lineage.material_segments.span_end is None:
            start = 0.0
            end = float(self.env.sequence_length)
        else:
            start = self.env._block_to_sequence_coordinate(lineage.material_segments.span_start)
            end = self.env._block_to_sequence_coordinate(lineage.material_segments.span_end + 1)
        width = max(float(end - start), 1.0)
        return float(start), float(end), width

    def _lineage_interval_features(self, lineage, variant_count, device):
        start, end, width = self._lineage_physical_interval(lineage)
        seq_len = max(float(self.env.sequence_length), 1.0)
        count_norm = float(variant_count) / max(float(self.env.num_blocks), 1.0)
        length_norm = float(width) / seq_len
        density = count_norm / max(length_norm, 1e-6)
        return torch.tensor(
            [
                start / seq_len,
                end / seq_len,
                length_norm,
                count_norm,
                density,
            ],
            dtype=torch.float32,
            device=device,
        )

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
        expected_shape = (
            (int(len(lineage.variant_indices)), 4)
            if self.input_mode == "vcf"
            else (int(self.env.num_blocks), 4)
        )
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
        summary_reps,
        state_contexts=None,
        ):
        batch_size = len(candidate_actions)
        max_candidates = max(len(actions) for actions in candidate_actions)
        logits = lineage_reps.new_full((batch_size, max_candidates), float("-inf"))
        feat_dim = self.embedding_size * 4
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
        logits = self.action_scorer(features.reshape(-1, feat_dim)).reshape(
            batch_size,
            max_candidates,
        )
        if (
            self.local_mode
            and self.local_prior_action_logit_bias != 0.0
            and state_contexts is not None
        ):
            logits = logits + self._local_prior_action_bias(
                candidate_actions,
                state_contexts,
                max_candidates,
                dtype=logits.dtype,
                device=logits.device,
            )
        if (
            self.local_mode
            and self.input_mode == "vcf"
            and self.local_coalescence_similarity_bias != 0.0
            and state_contexts is not None
        ):
            logits = logits + self._local_similarity_bias(
                candidate_actions,
                state_contexts,
                max_candidates,
                dtype=logits.dtype,
                device=logits.device,
            )

        counts = torch.tensor(candidate_counts, device=self.device)
        valid = torch.arange(max_candidates, device=self.device).unsqueeze(0) < counts.unsqueeze(1)
        # Invalid actions must have exactly zero probability.  A large finite
        # sentinel can leak mass in low-precision arithmetic and makes support
        # diagnostics ambiguous.
        masked_logits = logits.masked_fill(~valid, float("-inf"))
        return masked_logits, features

    def prepare_action_probability_logits(
        self,
        logits,
        candidate_actions,
        state_contexts,
        random_spec=None,
    ):
        """Apply rollout temperature and the mass-preserving split bias."""

        temperature = (
            1.0
            if random_spec is None or "T" not in random_spec
            else float(random_spec["T"])
        )
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("rollout temperature must be finite and positive")
        has_temperature = random_spec is not None and "T" in random_spec
        probability_logits = logits / temperature if has_temperature else logits
        empty_records = tuple(
            tuple(None for _ in actions) for actions in candidate_actions
        )
        empty_diagnostics = tuple(
            {"recombination_split_bias_enabled": False}
            for _ in candidate_actions
        )
        if not self.recombination_split_bias.enabled:
            return probability_logits, empty_records, empty_diagnostics
        if state_contexts is None or len(state_contexts) != len(candidate_actions):
            raise ValueError(
                "enabled recombination split bias requires one state per action row"
            )

        adjusted = probability_logits.clone()
        records_by_row = []
        diagnostics_by_row = []
        lineage_weight = float(
            self.recombination_split_bias_config["lineage_weight"]
        )
        for row, (actions, state) in enumerate(
            zip(candidate_actions, state_contexts)
        ):
            records = self.recombination_split_bias.score_candidates(
                state,
                actions,
                device=logits.device,
                dtype=logits.dtype,
            )
            records_by_row.append(records)
            recombination_indices = [
                index
                for index, action in enumerate(actions)
                if isinstance(action, RecombinationChoice)
            ]
            valid_before = torch.softmax(
                probability_logits[row, : len(actions)],
                dim=0,
            )
            mass_before = (
                valid_before[recombination_indices].sum()
                if recombination_indices
                else valid_before.new_tensor(0.0)
            )
            atomic_adjustments = logits.new_zeros(len(actions))
            all_breakpoint_scores = []
            if recombination_indices and lineage_weight != 0.0:
                recombination_index_tensor = torch.as_tensor(
                    recombination_indices,
                    dtype=torch.long,
                    device=logits.device,
                )
                lineage_scores = torch.stack(
                    [records[index].lineage_score for index in recombination_indices]
                ).to(device=logits.device, dtype=logits.dtype)
                proposed = probability_logits[
                    row, recombination_index_tensor
                ] + lineage_weight * lineage_scores / temperature
                correction = torch.logsumexp(proposed, dim=0) - torch.logsumexp(
                    probability_logits[row, recombination_index_tensor],
                    dim=0,
                )
                final_recombination = proposed - correction
                adjusted[row, recombination_index_tensor] = final_recombination
                atomic_adjustments[recombination_index_tensor] = (
                    final_recombination
                    - probability_logits[row, recombination_index_tensor]
                )
            for record in records:
                if record is not None:
                    all_breakpoint_scores.append(record.breakpoint_scores)

            valid_after = torch.softmax(adjusted[row, : len(actions)], dim=0)
            mass_after = (
                valid_after[recombination_indices].sum()
                if recombination_indices
                else valid_after.new_tensor(0.0)
            )
            if all_breakpoint_scores:
                flattened = torch.cat(all_breakpoint_scores)
                score_min = float(flattened.min().detach().cpu().item())
                score_mean = float(flattened.mean().detach().cpu().item())
                score_max = float(flattened.max().detach().cpu().item())
            else:
                score_min = score_mean = score_max = 0.0
            diagnostics_by_row.append(
                {
                    "recombination_split_bias_enabled": True,
                    "recombination_mass_before_split_bias": float(
                        mass_before.detach().cpu().item()
                    ),
                    "recombination_mass_after_split_bias": float(
                        mass_after.detach().cpu().item()
                    ),
                    "recombination_split_mass_absolute_error": float(
                        torch.abs(mass_after - mass_before)
                        .detach().cpu().item()
                    ),
                    "recombination_split_score_min": score_min,
                    "recombination_split_score_mean": score_mean,
                    "recombination_split_score_max": score_max,
                    "recombination_split_atomic_adjustments": atomic_adjustments,
                }
            )
        return (
            adjusted,
            tuple(records_by_row),
            tuple(diagnostics_by_row),
        )

    def apply_local_cwr_event_gate(
        self,
        probability_logits,
        candidate_actions,
        summary_reps,
        event_rates,
        *,
        random_spec=None,
    ):
        """Compose CwR event probabilities with conditional action policies."""

        if not self.local_cwr_event_gate_config["enabled"]:
            return probability_logits, tuple({} for _ in candidate_actions)
        if event_rates is None or len(event_rates) != len(candidate_actions):
            raise ValueError(
                "enabled local CwR event gate requires one rate record per row"
            )
        if int(summary_reps.shape[0]) != len(candidate_actions):
            raise ValueError(
                "local CwR event gate summary rows must align with candidates"
            )
        temperature = (
            1.0
            if random_spec is None or "T" not in random_spec
            else float(random_spec["T"])
        )
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("rollout temperature must be finite and positive")

        residuals, raw_residuals = self.compute_local_cwr_event_residual(
            summary_reps
        )
        final_logits = torch.full_like(probability_logits, float("-inf"))
        diagnostics = []
        for row, (actions, rates) in enumerate(
            zip(candidate_actions, event_rates)
        ):
            coal_indices = [
                index
                for index, action in enumerate(actions)
                if isinstance(action, CoalescenceChoice)
            ]
            recombination_indices = [
                index
                for index, action in enumerate(actions)
                if isinstance(action, RecombinationChoice)
            ]
            try:
                lambda_coal = float(rates["lambda_coal"])
                lambda_recomb = float(rates["lambda_recomb"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "local CwR event rates require lambda_coal and lambda_recomb"
                ) from exc
            if any(
                not math.isfinite(value) or value < 0.0
                for value in (lambda_coal, lambda_recomb)
            ):
                raise ValueError("local CwR event rates must be finite and nonnegative")
            if bool(coal_indices) != bool(lambda_coal > 0.0):
                raise ValueError(
                    "local coalescence candidates do not match their CwR rate"
                )
            if bool(recombination_indices) != bool(lambda_recomb > 0.0):
                raise ValueError(
                    "local recombination candidates do not match their CwR rate"
                )
            total_rate = lambda_coal + lambda_recomb
            if total_rate <= 0.0:
                raise ValueError("local generated events require a positive CwR rate")

            event_logits = probability_logits.new_full((2,), float("-inf"))
            if coal_indices:
                event_logits[0] = math.log(lambda_coal)
            if recombination_indices:
                event_logits[1] = (
                    math.log(lambda_recomb) + residuals[row]
                )
            event_log_probabilities = torch.log_softmax(
                event_logits / temperature,
                dim=0,
            )
            for event_index, indices in enumerate(
                (coal_indices, recombination_indices)
            ):
                if not indices:
                    continue
                index_tensor = torch.as_tensor(
                    indices,
                    dtype=torch.long,
                    device=probability_logits.device,
                )
                conditional_log_probabilities = torch.log_softmax(
                    probability_logits[row, index_tensor],
                    dim=0,
                )
                final_logits[row, index_tensor] = (
                    event_log_probabilities[event_index]
                    + conditional_log_probabilities
                )

            diagnostics.append(
                {
                    "local_cwr_event_gate_enabled": True,
                    "local_cwr_lambda_coal": lambda_coal,
                    "local_cwr_lambda_recomb": lambda_recomb,
                    "local_cwr_prior_coalescence_probability": (
                        lambda_coal / total_rate
                    ),
                    "local_cwr_prior_recombination_probability": (
                        lambda_recomb / total_rate
                    ),
                    "local_cwr_event_residual": float(
                        residuals[row].detach().cpu().item()
                    ),
                    "local_cwr_event_raw_residual": float(
                        raw_residuals[row].detach().cpu().item()
                    ),
                    "local_cwr_policy_coalescence_probability": float(
                        torch.exp(event_log_probabilities[0])
                        .detach().cpu().item()
                    ) if coal_indices else 0.0,
                    "local_cwr_policy_recombination_probability": float(
                        torch.exp(event_log_probabilities[1])
                        .detach().cpu().item()
                    ) if recombination_indices else 0.0,
                }
            )
        return final_logits, tuple(diagnostics)

    def score_action_candidates(
        self,
        candidate_actions,
        lineage_reps,
        summary_reps,
        *,
        state_contexts=None,
        event_rates=None,
        random_spec=None,
    ) -> CandidateScoringResult:
        """Return every candidate-aligned value used by the policy."""

        base_logits, action_features = self._score_candidates(
            candidate_actions,
            lineage_reps,
            summary_reps,
            state_contexts=state_contexts,
        )
        candidate_counts = torch.as_tensor(
            [len(actions) for actions in candidate_actions],
            dtype=torch.long,
            device=base_logits.device,
        )
        valid_mask = (
            torch.arange(base_logits.shape[1], device=base_logits.device)
            .unsqueeze(0)
            < candidate_counts.unsqueeze(1)
        )
        probability_logits, split_records, diagnostics = (
            self.prepare_action_probability_logits(
                base_logits,
                candidate_actions,
                state_contexts,
                random_spec=random_spec,
            )
        )
        if self.local_cwr_event_gate_config["enabled"]:
            probability_logits, cwr_diagnostics = self.apply_local_cwr_event_gate(
                probability_logits,
                candidate_actions,
                summary_reps,
                event_rates,
                random_spec=random_spec,
            )
            diagnostics = tuple(
                {**split_row, **cwr_row}
                for split_row, cwr_row in zip(diagnostics, cwr_diagnostics)
            )
        return CandidateScoringResult(
            base_logits=base_logits,
            probability_logits=probability_logits,
            action_features=action_features,
            valid_mask=valid_mask,
            split_records=split_records,
            diagnostics=diagnostics,
        )

    def recombination_breakpoint_logit_bias(
        self,
        split_record,
        valid_breakpoints,
    ):
        """Validate and align the handcrafted bias with breakpoint support."""

        if split_record is None:
            return None
        support = tuple(int(value) for value in valid_breakpoints)
        if split_record.breakpoints != support:
            raise ValueError(
                "recombination split-score breakpoint order does not match "
                "the current breakpoint support"
            )
        bias = split_record.breakpoint_bias(
            self.recombination_split_bias_config["breakpoint_weight"]
        )
        if bias.ndim != 1 or int(bias.numel()) != len(support):
            raise ValueError(
                "recombination split-score bias must align with breakpoint support"
            )
        if not bool(torch.isfinite(bias).all().detach().cpu().item()):
            raise ValueError("recombination split-score bias must be finite")
        return bias

    def _local_similarity_bias(
        self,
        candidate_actions,
        states,
        max_candidates,
        *,
        dtype,
        device,
    ):
        rows = torch.zeros(
            len(candidate_actions),
            int(max_candidates),
            dtype=dtype,
            device=device,
        )
        weight = float(self.local_coalescence_similarity_bias)
        for batch_idx, actions in enumerate(candidate_actions):
            if batch_idx >= len(states):
                break
            state = states[batch_idx]
            if not hasattr(state, "active_lineages"):
                continue
            cache = self._local_lineage_similarity_cache(state, dtype, device)
            for action_idx, action in enumerate(actions):
                if not isinstance(action, CoalescenceChoice):
                    continue
                score = self._coalescence_similarity_score(
                    state,
                    action,
                    cache,
                    dtype,
                    device,
                )
                rows[batch_idx, action_idx] = weight * score
        return rows

    def _local_prior_action_bias(
        self,
        candidate_actions,
        states,
        max_candidates,
        *,
        dtype,
        device,
    ):
        rows = torch.zeros(
            len(candidate_actions),
            int(max_candidates),
            dtype=dtype,
            device=device,
        )
        weight = float(self.local_prior_action_logit_bias)
        epsilon = torch.finfo(dtype).tiny
        for batch_idx, actions in enumerate(candidate_actions):
            if batch_idx >= len(states):
                break
            state = states[batch_idx]
            coal_actions = [
                action for action in actions if isinstance(action, CoalescenceChoice)
            ]
            recomb_actions = [
                action for action in actions if isinstance(action, RecombinationChoice)
            ]
            options = getattr(state, "prior_options", None)
            if options is None and hasattr(self.env, "enumerate_prior_options"):
                options = self.env.enumerate_prior_options(state)
            rates = getattr(options, "rates", None) or {}
            lambda_coal = max(float(rates.get("lambda_coal", len(coal_actions))), 0.0)
            lambda_recomb = max(float(rates.get("lambda_recomb", 0.0)), 0.0)
            total_rate = lambda_coal + lambda_recomb
            if total_rate <= 0.0:
                continue
            coal_weights = [
                self._local_coalescence_overlap_weight(state, action)
                for action in coal_actions
            ]
            coal_total = float(sum(coal_weights))
            if coal_actions and not coal_total > 0.0:
                coal_weights = [1.0] * len(coal_actions)
                coal_total = float(len(coal_actions))
            recomb_weights = [
                self._local_recombination_action_weight(state, action)
                for action in recomb_actions
            ]
            recomb_total = float(sum(recomb_weights))
            if recomb_actions and not recomb_total > 0.0:
                recomb_weights = [1.0] * len(recomb_actions)
                recomb_total = float(len(recomb_actions))
            coal_index = 0
            recomb_index = 0
            for action_idx, action in enumerate(actions):
                probability = 0.0
                if isinstance(action, CoalescenceChoice):
                    if lambda_coal > 0.0 and coal_total > 0.0:
                        probability = (
                            lambda_coal
                            / total_rate
                            * float(coal_weights[coal_index])
                            / coal_total
                        )
                    coal_index += 1
                elif isinstance(action, RecombinationChoice):
                    if lambda_recomb > 0.0 and recomb_total > 0.0:
                        probability = (
                            lambda_recomb
                            / total_rate
                            * float(recomb_weights[recomb_index])
                            / recomb_total
                        )
                    recomb_index += 1
                if probability > 0.0:
                    rows[batch_idx, action_idx] = weight * torch.log(
                        torch.as_tensor(
                            max(float(probability), float(epsilon)),
                            dtype=dtype,
                            device=device,
                        )
                    )
        return rows

    def _local_lineage_similarity_cache(self, state, dtype, device):
        cache = []
        for lineage in state.active_lineages:
            variants = tuple(int(variant) for variant in lineage.variant_indices)
            partials = self._lineage_partials_tensor(lineage).to(
                dtype=dtype,
                device=device,
            )
            cache.append(
                {
                    "variants": variants,
                    "variant_set": set(variants),
                    "positions": {
                        int(variant): index
                        for index, variant in enumerate(variants)
                    },
                    "partials": partials,
                }
            )
        return cache

    def _coalescence_similarity_score(self, state, action, cache, dtype, device):
        left = cache[int(action.active_lineage_i)]
        right = cache[int(action.active_lineage_j)]
        common_variants = sorted(left["variant_set"].intersection(right["variant_set"]))
        if not common_variants:
            return torch.zeros((), dtype=dtype, device=device)
        left_index = torch.as_tensor(
            [left["positions"][variant] for variant in common_variants],
            dtype=torch.long,
            device=device,
        )
        right_index = torch.as_tensor(
            [right["positions"][variant] for variant in common_variants],
            dtype=torch.long,
            device=device,
        )
        compatibility = (
            left["partials"].index_select(0, left_index)
            * right["partials"].index_select(0, right_index)
        ).sum(dim=1)
        mean_compatibility = compatibility.clamp(0.0, 1.0).mean()
        return 2.0 * mean_compatibility - 1.0

    def _local_coalescence_overlap_weight(self, state, action):
        left = state.active_lineages[int(action.active_lineage_i)]
        right = state.active_lineages[int(action.active_lineage_j)]
        overlap = left.material_segments.intersection(right.material_segments)
        target = getattr(state, "target_material", None)
        if target is not None:
            overlap = overlap.intersection(target)
        return max(self._local_material_physical_length(state, overlap), 0.0)

    def _local_recombination_action_weight(self, state, action):
        lineage = state.active_lineages[int(action.active_lineage_i)]
        return max(self._local_material_physical_length(state, lineage.material_segments), 0.0)

    @staticmethod
    def _local_material_physical_length(state, material):
        boundaries = getattr(state, "block_boundaries", None)
        if boundaries is None:
            return float(getattr(material, "count", 0))
        length = 0.0
        for start, end in material.segments:
            length += max(float(boundaries[int(end)]) - float(boundaries[int(start)]), 0.0)
        return float(length)

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



    def forward(
        self,
        all_actions,
        lineage_reps,
        summary_reps,
        lineage_seq_features,
        batch_active_lineage_counts,
        random_spec,
        event_rates=None,
    ):
        
        all_candidate_actions = all_actions


        if any(len(actions) == 0 for actions in all_candidate_actions):
            raise ValueError("ARGModel.forward received a batch item with no candidate actions.")

        state_contexts = (
            lineage_seq_features
            if self.local_mode and isinstance(lineage_seq_features, list)
            else None
        )
        scoring = self.score_action_candidates(
            all_candidate_actions,
            lineage_reps,
            summary_reps,
            state_contexts=state_contexts,
            event_rates=event_rates,
            random_spec=random_spec,
        )
        probability_logits = scoring.probability_logits

        # Sample actions in a vectorized way.  ``probability_logits`` already
        # contains the active rollout temperature and every fixed logit bias.
        sampled_action_indices = Categorical(logits=probability_logits).sample()
        # sampled_action_indices shape: (batch,)

        # Convert to standard Python ints and collect for indexing
        selected_action_indices = sampled_action_indices.detach().cpu().tolist()

        # Now, retrieve chosen actions and features in a single loop
        choosen_actions = []
        choosen_action_features = []
        chosen_split_records = []
        for batch_idx, action_idx in enumerate(selected_action_indices):
            choosen_actions.append(all_candidate_actions[batch_idx][action_idx])
            choosen_action_features.append(
                scoring.action_features[batch_idx, action_idx]
            )
            chosen_split_records.append(
                scoring.split_records[batch_idx][action_idx]
            )

        # Record the probability of the distribution that actually sampled the
        # action.  Previously temperature-controlled rollouts sampled from
        # logits / T but recorded log-softmax(logits), which is not a valid
        # trajectory probability when T != 1.
        self.last_action_probability_logits = probability_logits
        self.last_action_valid_mask = scoring.valid_mask
        self.last_action_split_diagnostics = scoring.diagnostics
        log_action_pf = self.compute_log_path_pf(
            probability_logits,
            selected_action_indices,
        )

        return (
            log_action_pf,
            selected_action_indices,
            choosen_actions,
            choosen_action_features,
            chosen_split_records,
        )
