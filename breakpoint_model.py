import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

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
        action_context_dim=128,
        gap_hidden_dim=256,
        gap_layers=3,
        gap_dropout=0.0,
        use_position_features=True,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128] * 2

        self.feature_dim = 4
        self.hidden_dim = int(hidden_dim)
        self.action_context_dim = int(action_context_dim)
        self.use_position_features = bool(use_position_features)
        self.position_feature_dim = 3 if self.use_position_features else 0
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

        self.gap_scorer = self._build_gap_scorer(
            input_dim=self.hidden_dim + self.action_context_dim + self.position_feature_dim,
            hidden_dim=int(gap_hidden_dim),
            layers=int(gap_layers),
            dropout=float(gap_dropout),
        )

    def _build_gap_scorer(self, input_dim, hidden_dim, layers, dropout):
        if layers < 0:
            raise ValueError(f"gap_layers must be non-negative, got {layers}")
        if layers == 0:
            scorer = nn.Sequential(nn.Linear(input_dim, 1))
        else:
            modules = [
                nn.Linear(input_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            ]
            for _ in range(layers - 1):
                modules.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                ])
            modules.append(nn.Linear(hidden_dim, 1))
            scorer = nn.Sequential(*modules)
        scorer.apply(self._init_mlp_weights)
        return scorer

    def _init_mlp_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def gap_features(self, x):
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

        # Keep features for valid split gaps only. Gap row i corresponds to breakpoint k=i+1.
        return x.transpose(1, 2)[:, :-1].contiguous()

    def breakpoint_scorer(self, x):
        return self.gap_features(x)

    def _breakpoint_logit_indices(self, sequence_length, num_blocks, breakpoints, device):
        indices = []
        for breakpoint in breakpoints:
            index = min(max(int(breakpoint), 1), int(num_blocks) - 1) - 1
            indices.append(index)
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _valid_breakpoints_list(self, valid_breakpoints):
        return list(range(
            int(valid_breakpoints.span_start) + 1,
            int(valid_breakpoints.span_end) + 1,
        )) if hasattr(valid_breakpoints, "span_start") else list(valid_breakpoints)

    def _prepare_action_context(self, action_context, device, dtype):
        if action_context is None:
            raise ValueError("action_context is required for breakpoint scoring")
        if torch.is_tensor(action_context):
            action_context = action_context.to(device=device, dtype=dtype)
        else:
            action_context = torch.as_tensor(action_context, device=device, dtype=dtype)
        if action_context.ndim == 2 and action_context.shape[0] == 1:
            action_context = action_context[0]
        if action_context.ndim != 1:
            raise ValueError(f"expected 1D action_context, got shape {tuple(action_context.shape)}")
        if action_context.shape[0] != self.action_context_dim:
            raise ValueError(
                f"expected action_context dim {self.action_context_dim}, got {action_context.shape[0]}"
            )
        return action_context

    def _position_features(self, valid_breakpoints, num_blocks, device, dtype):
        breakpoints = torch.tensor(valid_breakpoints, dtype=dtype, device=device)
        max_gap_count = max(int(num_blocks) - 1, 1)
        absolute_position = breakpoints / float(max(int(num_blocks), 1))

        min_bp = float(min(valid_breakpoints))
        max_bp = float(max(valid_breakpoints))
        relative_denominator = max(max_bp - min_bp, 1.0)
        relative_position = (breakpoints - min_bp) / relative_denominator

        span_width = torch.full_like(
            absolute_position,
            fill_value=float(len(valid_breakpoints)) / float(max_gap_count),
        )
        return torch.stack([absolute_position, relative_position, span_width], dim=1)

    def valid_breakpoint_logits(
        self,
        valid_breakpoints,
        lineage_seq_feature,
        sequence_length,
        num_blocks,
        action_context,
    ):
        valid_breakpoints = self._valid_breakpoints_list(valid_breakpoints)
        if not valid_breakpoints:
            raise ValueError("Recombination action has no valid breakpoints")

        if lineage_seq_feature.ndim == 2:
            lineage_seq_feature = lineage_seq_feature.unsqueeze(0)
        elif lineage_seq_feature.ndim != 3 or lineage_seq_feature.shape[0] != 1:
            raise ValueError(
                "lineage_seq_feature must have shape [L, 4] or [1, L, 4], "
                f"got {tuple(lineage_seq_feature.shape)}"
            )

        gap_features = self.gap_features(lineage_seq_feature)[0]
        logit_indices = self._breakpoint_logit_indices(
            sequence_length,
            num_blocks,
            valid_breakpoints,
            gap_features.device,
        )
        valid_gap_features = gap_features[logit_indices]
        action_context = self._prepare_action_context(
            action_context,
            valid_gap_features.device,
            valid_gap_features.dtype,
        ).expand(len(valid_breakpoints), -1)
        scorer_inputs = [valid_gap_features, action_context]
        if self.use_position_features:
            scorer_inputs.append(
                self._position_features(
                    valid_breakpoints,
                    num_blocks,
                    valid_gap_features.device,
                    valid_gap_features.dtype,
                )
            )
        scorer_input = torch.cat(scorer_inputs, dim=1)
        return self.gap_scorer(scorer_input).squeeze(-1)

    def forward(
        self,
        valid_breakpoints,
        lineage_seq_feature,
        sequence_length,
        num_blocks,
        action_context,
        random_spec=None,
    ):
        valid_breakpoints = self._valid_breakpoints_list(valid_breakpoints)
        valid_logits = self.valid_breakpoint_logits(
            valid_breakpoints,
            lineage_seq_feature,
            sequence_length,
            num_blocks,
            action_context,
        )
        if random_spec is not None and "T" in random_spec:
            sample_logits = valid_logits / random_spec["T"]
        else:
            sample_logits = valid_logits

        local_idx = Categorical(logits=sample_logits).sample()
        breakpoint = int(valid_breakpoints[int(local_idx.detach().cpu().item())])
        log_p = F.log_softmax(valid_logits, dim=0)[local_idx]
        return breakpoint, log_p


class VCFBreakpointScorer(nn.Module):
    def __init__(
        self,
        env,
        hidden_dim=128,
        action_context_dim=128,
        gap_hidden_dim=256,
        gap_layers=3,
        gap_dropout=0.0,
    ):
        super().__init__()
        self.env = env
        self.action_context_dim = int(action_context_dim)
        self.feature_dim = self.action_context_dim + 11
        self.gap_scorer = self._build_gap_scorer(
            input_dim=self.feature_dim,
            hidden_dim=int(gap_hidden_dim or hidden_dim),
            layers=int(gap_layers),
            dropout=float(gap_dropout),
        )

    def _build_gap_scorer(self, input_dim, hidden_dim, layers, dropout):
        if layers < 0:
            raise ValueError(f"gap_layers must be non-negative, got {layers}")
        if layers == 0:
            scorer = nn.Sequential(nn.Linear(input_dim, 1))
        else:
            modules = [
                nn.Linear(input_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            ]
            for _ in range(layers - 1):
                modules.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                ])
            modules.append(nn.Linear(hidden_dim, 1))
            scorer = nn.Sequential(*modules)
        scorer.apply(self._init_mlp_weights)
        return scorer

    def _init_mlp_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _valid_breakpoints_tensor(self, valid_breakpoints, device):
        if hasattr(valid_breakpoints, "span_start"):
            start = int(valid_breakpoints.span_start)
            end = int(valid_breakpoints.span_end)
            if end <= start:
                return torch.empty(0, dtype=torch.long, device=device)
            return torch.arange(start + 1, end + 1, dtype=torch.long, device=device)
        if torch.is_tensor(valid_breakpoints):
            breakpoints = valid_breakpoints.to(device=device, dtype=torch.long)
        else:
            breakpoints = torch.as_tensor(list(valid_breakpoints), dtype=torch.long, device=device)
        if breakpoints.ndim != 1:
            raise ValueError(f"expected 1D valid_breakpoints, got shape {tuple(breakpoints.shape)}")
        return breakpoints

    def _prepare_action_context(self, action_context, device, dtype):
        if action_context is None:
            raise ValueError("action_context is required for breakpoint scoring")
        if torch.is_tensor(action_context):
            action_context = action_context.to(device=device, dtype=dtype)
        else:
            action_context = torch.as_tensor(action_context, device=device, dtype=dtype)
        if action_context.ndim == 2 and action_context.shape[0] == 1:
            action_context = action_context[0]
        if action_context.ndim != 1:
            raise ValueError(f"expected 1D action_context, got shape {tuple(action_context.shape)}")
        if action_context.shape[0] != self.action_context_dim:
            raise ValueError(
                f"expected action_context dim {self.action_context_dim}, got {action_context.shape[0]}"
            )
        return action_context

    def _lineage_partials_tensor(self, lineage, device, dtype):
        partials = lineage.partials
        if not torch.is_tensor(partials):
            partials = torch.as_tensor(partials, device=device, dtype=dtype)
        else:
            partials = partials.to(device=device, dtype=dtype)
        expected_shape = (int(lineage.material_segments.count), 4)
        if tuple(partials.shape) != expected_shape:
            raise ValueError(
                f"VCF lineage partials must have shape {expected_shape}, got {tuple(partials.shape)}"
            )
        return partials

    def _gather_neighbor_partials(self, partials, lineage_blocks, query_blocks):
        gathered = partials.new_zeros((query_blocks.numel(), partials.shape[-1]))
        if query_blocks.numel() == 0 or lineage_blocks.numel() == 0:
            return gathered

        positions = torch.searchsorted(lineage_blocks, query_blocks)
        safe_positions = positions.clamp(max=lineage_blocks.numel() - 1)
        present = (
            (positions >= 0)
            & (positions < lineage_blocks.numel())
            & (lineage_blocks.index_select(0, safe_positions) == query_blocks)
        )
        if bool(present.any().detach().cpu().item()):
            gathered[present] = partials.index_select(0, safe_positions[present])
        return gathered

    def _env_tensor(self, attr, fallback, device, dtype):
        value = getattr(self.env, attr, None)
        if value is None:
            return torch.as_tensor(fallback, device=device, dtype=dtype)
        return value.to(device=device, dtype=dtype)

    def _candidate_features(self, breakpoint_tensor, lineage, device, dtype, sequence_length=None, num_blocks=None):
        if breakpoint_tensor.ndim != 1:
            raise ValueError(f"expected 1D breakpoint tensor, got shape {tuple(breakpoint_tensor.shape)}")
        breakpoints = breakpoint_tensor.to(device=device, dtype=torch.long)
        seq_len = max(float(self.env.sequence_length if sequence_length is None else sequence_length), 1.0)
        block_count = max(int(self.env.num_blocks if num_blocks is None else num_blocks), 1)

        partials = self._lineage_partials_tensor(lineage, device, dtype)
        lineage_blocks = lineage.block_indices_tensor(device)
        positions = self._env_tensor(
            "variant_position_tensor",
            self.env.variant_positions0,
            device,
            dtype,
        )
        boundaries = self._env_tensor(
            "variant_boundary_tensor",
            self.env.variant_boundaries,
            device,
            dtype,
        )

        max_variant_idx = max(int(positions.numel()) - 1, 0)
        left_variant_idx = (breakpoints - 1).clamp(min=0, max=max_variant_idx)
        right_variant_idx = breakpoints.clamp(min=0, max=max_variant_idx)
        gap = (
            positions.index_select(0, right_variant_idx)
            - positions.index_select(0, left_variant_idx)
        ).clamp_min(1.0)

        max_boundary_idx = max(int(boundaries.numel()) - 1, 0)
        coord = boundaries.index_select(0, breakpoints.clamp(min=0, max=max_boundary_idx))
        span_start = lineage.material_segments.span_start
        span_end = lineage.material_segments.span_end
        if span_start is None or span_end is None:
            interval_start = boundaries.new_tensor(0.0)
            interval_end = boundaries.new_tensor(float(seq_len))
        else:
            start_idx = min(max(int(span_start), 0), max_boundary_idx)
            end_idx = min(max(int(span_end) + 1, 0), max_boundary_idx)
            interval_start = boundaries[start_idx]
            interval_end = boundaries[end_idx]
        interval_width = (interval_end - interval_start).clamp_min(1.0)

        split_norm = breakpoints.to(dtype=dtype) / float(block_count)
        gap_norm = gap / seq_len
        rel_pos = (coord - interval_start) / interval_width
        left = self._gather_neighbor_partials(partials, lineage_blocks, breakpoints - 1)
        right = self._gather_neighbor_partials(partials, lineage_blocks, breakpoints)
        scalar_features = torch.stack([split_norm, gap_norm, rel_pos], dim=1)
        return torch.cat([scalar_features, left, right], dim=1)

    def _valid_breakpoint_logits_from_tensor(
        self,
        breakpoint_tensor,
        lineage,
        sequence_length,
        num_blocks,
        action_context,
    ):
        if breakpoint_tensor.numel() == 0:
            raise ValueError("Recombination action has no valid breakpoints")

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        breakpoints = breakpoint_tensor.to(device=device, dtype=torch.long)
        action_context = self._prepare_action_context(action_context, device, dtype).expand(
            breakpoints.numel(),
            -1,
        )
        candidate_features = self._candidate_features(
            breakpoints,
            lineage,
            device,
            dtype,
            sequence_length=sequence_length,
            num_blocks=num_blocks,
        )
        scorer_input = torch.cat([action_context, candidate_features], dim=1)
        return self.gap_scorer(scorer_input).squeeze(-1)

    def valid_breakpoint_logits(
        self,
        valid_breakpoints,
        lineage,
        sequence_length,
        num_blocks,
        action_context,
    ):
        device = next(self.parameters()).device
        breakpoints = self._valid_breakpoints_tensor(valid_breakpoints, device)
        return self._valid_breakpoint_logits_from_tensor(
            breakpoints,
            lineage,
            sequence_length,
            num_blocks,
            action_context,
        )

    def forward(
        self,
        valid_breakpoints,
        lineage,
        sequence_length,
        num_blocks,
        action_context,
        random_spec=None,
    ):
        device = next(self.parameters()).device
        breakpoints = self._valid_breakpoints_tensor(valid_breakpoints, device)
        valid_logits = self._valid_breakpoint_logits_from_tensor(
            breakpoints,
            lineage,
            sequence_length,
            num_blocks,
            action_context,
        )
        if random_spec is not None and "T" in random_spec:
            sample_logits = valid_logits / random_spec["T"]
        else:
            sample_logits = valid_logits

        local_idx = Categorical(logits=sample_logits).sample()
        breakpoint = int(breakpoints[local_idx].detach().cpu().item())
        log_p = F.log_softmax(valid_logits, dim=0)[local_idx]
        return breakpoint, log_p
