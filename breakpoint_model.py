import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import math
import numbers

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
        breakpoint=None,
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

        local_idx = (Categorical(logits=sample_logits).sample() if breakpoint is None
                     else valid_breakpoints.index(breakpoint))
        breakpoint = int(valid_breakpoints[int(local_idx)])
        log_p = F.log_softmax(valid_logits, dim=0)[local_idx]
        return breakpoint, log_p


class SparseMixtureBreakpointPolicy(nn.Module):
    """Sparse block tokens and a mixture of discretized, truncated logistics.

    Gaps remain in environment block coordinates, even for multi-base blocks.
    The alignment-derived index cache is rebuilt from the environment on load;
    it is deliberately not part of the checkpoint state dictionary.
    """

    def __init__(self, source_alignment, hidden_dim=128, layers=4, components=4,
                 dropout=0.1, action_context_dim=128, gap_hidden_dim=64,
                 gap_layers=1, gap_dropout=0.0):
        super().__init__()
        if hidden_dim < 1 or layers < 0 or components < 1 or gap_layers < 0:
            raise ValueError("Invalid sparse mixture architecture dimensions")
        if source_alignment.ndim != 3 or source_alignment.shape[-1] != 4:
            raise ValueError("source_alignment must have shape [samples, blocks, 4]")
        if min(source_alignment.shape[:2]) < 1:
            raise ValueError("source_alignment must contain samples and blocks")
        self.num_blocks = source_alignment.shape[1]
        self.hidden_dim = int(hidden_dim)
        self.components = int(components)
        self.action_context_dim = int(action_context_dim)
        with torch.no_grad():
            variable = (source_alignment != source_alignment[:1]).any(dim=0).any(dim=-1)
            indices = variable.nonzero(as_tuple=True)[0]
        self.register_buffer("informative_indices", indices, persistent=False)
        self.input_projection = nn.Linear(8, hidden_dim)
        self.blocks = nn.ModuleList(
            ResidualDilatedConvBlock(hidden_dim, kernel_size=5, dilation=2 ** i,
                                     dropout=dropout) for i in range(layers)
        )
        modules = []
        width = hidden_dim + action_context_dim + 3
        for _ in range(gap_layers):
            modules.extend([nn.Linear(width, gap_hidden_dim), nn.Dropout(gap_dropout), nn.ReLU()])
            width = gap_hidden_dim
        modules.append(nn.Linear(width, 3 * components))
        self.parameter_head = nn.Sequential(*modules)
        self.parameter_head.apply(lambda module: BreakpointSplitPositionCNN._init_mlp_weights(self, module))
        # Equal weights, evenly spaced centers, scales 0.1 + 0.1 * span.
        # Zero final weights make the initialization independent of the input.
        with torch.no_grad():
            output = self.parameter_head[-1]
            output.weight.zero_()
            fractions = (torch.arange(components, dtype=output.bias.dtype) + 0.5) / components
            output.bias[components:2 * components].copy_(torch.logit(fractions))
            output.bias[2 * components:].fill_(math.log(0.1 / 0.9))

    @staticmethod
    def valid_span(candidates, num_blocks):
        if hasattr(candidates, "span_start"):
            a, z = int(candidates.span_start) + 1, int(candidates.span_end)
        elif isinstance(candidates, range):
            if not candidates:
                raise ValueError("Recombination action has no valid breakpoints")
            if candidates.step != 1:
                raise ValueError("Sparse mixture requires contiguous ascending integer gaps")
            a, z = candidates.start, candidates[-1]
        else:
            values = list(candidates)
            if not values:
                raise ValueError("Recombination action has no valid breakpoints")
            if (any(not isinstance(b, numbers.Integral) for b in values)
                    or any(v != values[0] + i for i, v in enumerate(values))):
                raise ValueError("Sparse mixture requires contiguous ascending integer gaps")
            a, z = int(values[0]), int(values[-1])
        if a > z:
            raise ValueError("Recombination action has no valid breakpoints")
        if a < 1 or z >= num_blocks:
            raise ValueError("Valid gaps must lie between blocks 1 and num_blocks - 1")
        return a, z

    def sparse_tokens(self, lineage_seq_feature, sequence_length, num_blocks):
        x = lineage_seq_feature
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        if tuple(x.shape) != (num_blocks, 4) or num_blocks != self.num_blocks:
            raise ValueError("lineage_seq_feature must match the alignment's [blocks, 4] shape")
        if sequence_length < num_blocks:
            raise ValueError("sequence_length must be at least num_blocks")
        # Masked partials use zero rows for absent material, including at holes.
        coverage = x.detach().ne(0).any(dim=-1)
        changes = (coverage[1:] != coverage[:-1]).nonzero(as_tuple=True)[0]
        indices = torch.cat((self.informative_indices, changes, changes + 1,
                             changes.new_tensor([0, num_blocks - 1]))).unique(sorted=True)
        selected = x[indices].to(dtype=self.input_projection.weight.dtype)
        # Match the environment's rounded physical block boundaries.
        position = (indices.to(torch.float64) * (sequence_length / num_blocks)).round()
        position = (position / sequence_length).to(selected.dtype)
        left = position - torch.cat((position.new_zeros(1), position[:-1]))
        right = torch.cat((position[1:], position.new_ones(1))) - position
        tokens = torch.cat((selected, position[:, None], left[:, None], right[:, None],
                            coverage[indices, None].to(selected.dtype)), dim=-1)
        return indices, tokens

    def distribution_parameters(self, valid_breakpoints, lineage_seq_feature,
                                sequence_length, num_blocks, action_context):
        a, z = self.valid_span(valid_breakpoints, num_blocks)
        _, tokens = self.sparse_tokens(lineage_seq_feature, sequence_length, num_blocks)
        x = F.gelu(self.input_projection(tokens)).transpose(0, 1).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        pooled = x.mean(dim=-1)[0]
        context = BreakpointSplitPositionCNN._prepare_action_context(
            self, action_context, pooled.device, pooled.dtype)
        span = pooled.new_tensor([a / num_blocks, z / num_blocks,
                                  (z - a + 1) / max(num_blocks - 1, 1)])
        raw = self.parameter_head(torch.cat((pooled, context, span)))
        weights, locations, scales = raw.double().chunk(3)
        n = z - a + 1
        return (F.log_softmax(weights, dim=-1), a - 0.5 + n * locations.sigmoid(),
                0.1 + n * scales.sigmoid())

    @staticmethod
    def _log_interval_mass(lower, upper, locations, scales):
        # sigmoid(v)-sigmoid(u) = sigmoid(v)*sigmoid(-u)*(1-exp(u-v)).
        # Compute the width separately to avoid subtracting nearby tail logits.
        u = (lower - locations) / scales
        v = (upper - locations) / scales
        return F.logsigmoid(v) + F.logsigmoid(-u) + torch.log(-torch.expm1(-(upper - lower) / scales))

    @classmethod
    def log_probabilities(cls, gaps, a, z, parameters):
        """Marginal log mass, in float64, for scalar or tensor block gaps."""
        log_weights, locations, scales = (p.double() for p in parameters)
        gaps = torch.as_tensor(gaps, device=locations.device, dtype=torch.float64)
        mass = cls._log_interval_mass(gaps[..., None] - 0.5, gaps[..., None] + 0.5,
                                     locations, scales)
        normalizer = cls._log_interval_mass(a - 0.5, z + 0.5, locations, scales)
        result = torch.logsumexp(log_weights + mass - normalizer, dim=-1)
        if a == z:
            result = result * 0.0
        return result.masked_fill((gaps < a) | (gaps > z) | (gaps != gaps.round()), -torch.inf)

    @classmethod
    @torch.no_grad()
    def sample_gap(cls, a, z, parameters, temperature=1.0):
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Breakpoint temperature must be finite and positive")
        if a == z:
            return a
        weights, locations, scales = (p.detach().double() for p in parameters)
        if temperature == 1.0:
            j = Categorical(logits=weights).sample()
            loc, scale = locations[j], scales[j]
            low = torch.sigmoid((a - 0.5 - loc) / scale)
            high = torch.sigmoid((z + 0.5 - loc) / scale)
            uniform = torch.rand((), dtype=torch.float64, device=loc.device)
            probability = low + uniform * (high - low)
            eps = torch.finfo(torch.float64).eps
            draw = loc + scale * torch.logit(probability.clamp(eps, 1 - eps))
            return int(torch.floor(draw + 0.5).clamp(a, z).item())
        best_score = locations.new_tensor(-torch.inf)
        best_gap = locations.new_tensor(a, dtype=torch.long)
        for start in range(a, z + 1, 1024):
            gaps = torch.arange(start, min(start + 1024, z + 1), device=locations.device)
            logits = cls.log_probabilities(gaps, a, z, (weights, locations, scales))
            # -log(Exp(1)) is standard Gumbel; retain only one winning gap.
            gumbel = -torch.empty_like(logits).exponential_().log()
            # Multiplying all scores by T preserves the argmax and avoids
            # overflowing log P / T at very small positive temperatures.
            scores = (logits + temperature * gumbel if temperature < 1.0
                      else logits / temperature + gumbel)
            score, index = scores.max(dim=0)
            best_gap = torch.where(score > best_score, gaps[index], best_gap)
            best_score = torch.maximum(score, best_score)
        return int(best_gap.item())

    def forward(self, valid_breakpoints, lineage_seq_feature, sequence_length,
                num_blocks, action_context, random_spec=None, breakpoint=None):
        a, z = self.valid_span(valid_breakpoints, num_blocks)
        parameters = self.distribution_parameters(range(a, z + 1), lineage_seq_feature,
                                                   sequence_length, num_blocks, action_context)
        if breakpoint is None:
            breakpoint = self.sample_gap(a, z, parameters, (random_spec or {}).get("T", 1.0))
        elif not isinstance(breakpoint, numbers.Integral) or not a <= breakpoint <= z:
            raise ValueError("Replay breakpoint is outside the action's valid span")
        log_p = self.log_probabilities(breakpoint, a, z, parameters)
        return breakpoint, log_p.to(dtype=self.input_projection.weight.dtype)
