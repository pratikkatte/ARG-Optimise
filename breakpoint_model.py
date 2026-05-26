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

    def breakpoint_scorer(self, x):
        """
        """
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

    def _breakpoint_logit_indices(self, sequence_length, num_blocks, breakpoints, device):
        indices = []
        for breakpoint in breakpoints:
            index = min(max(int(breakpoint), 1), int(num_blocks) - 1) - 1
            indices.append(index)
        return torch.tensor(indices, dtype=torch.long, device=device)

    def forward(self, valid_breakpoints, lineage_seq_feature, sequence_length, num_blocks, random_spec):
        """
        """
        valid_breakpoints = list(range(
            int(valid_breakpoints.span_start) + 1,
            int(valid_breakpoints.span_end) + 1,
        )) if hasattr(valid_breakpoints, "span_start") else list(valid_breakpoints)
        if not valid_breakpoints:
            raise ValueError("Recombination action has no valid breakpoints")

        bp_logits = self.breakpoint_scorer(lineage_seq_feature.unsqueeze(0))[0]
        logit_indices = self._breakpoint_logit_indices(
            sequence_length,
            num_blocks,
            valid_breakpoints,
            bp_logits.device,
        )
        valid_logits = bp_logits[logit_indices]
        if random_spec is not None and "T" in random_spec:
            sample_logits = valid_logits / random_spec["T"]
        else:
            sample_logits = valid_logits

        local_idx = Categorical(logits=sample_logits).sample()
        breakpoint = int(valid_breakpoints[int(local_idx.detach().cpu().item())])
        log_p = F.log_softmax(valid_logits, dim=0)[local_idx]
        return breakpoint, log_p

        
