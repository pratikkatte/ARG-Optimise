import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .layers import mlp

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

        self.gap_scorer = mlp(
            self.hidden_dim + self.action_context_dim + self.position_feature_dim,
            int(gap_hidden_dim), 1, layers=int(gap_layers), dropout=float(gap_dropout),
        )

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
        score_args = (
            valid_breakpoints,
            lineage_seq_feature,
            sequence_length,
            num_blocks,
            action_context,
        )
        if torch.is_grad_enabled():
            # Full-resolution sequences otherwise retain every CNN activation
            # for every recombination step until the trajectory loss is built.
            valid_logits = checkpoint(
                self.valid_breakpoint_logits,
                *score_args,
                use_reentrant=False,
            )
        else:
            valid_logits = self.valid_breakpoint_logits(*score_args)
        if random_spec is not None and "T" in random_spec:
            sample_logits = valid_logits / random_spec["T"]
        else:
            sample_logits = valid_logits

        local_idx = Categorical(logits=sample_logits).sample()
        breakpoint = int(valid_breakpoints[int(local_idx.detach().cpu().item())])
        log_p = F.log_softmax(valid_logits, dim=0)[local_idx]
        return breakpoint, log_p

