"""Paper-inspired, non-learned recombination split scoring.

The scorer is deliberately local to phased-VCF ARG refinement.  It uses
lineage partials to measure whether the two sides of a candidate split are
better supported by different active ancestral lineages than the unsplit
lineage.  The returned tensors are detached policy features, not parameters.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

try:
    from .env import RecombinationChoice
except ImportError:  # Support script-style entry points.
    from env import RecombinationChoice


RECOMBINATION_SPLIT_SCORE_MODE = "partial_compatibility_v1"
DEFAULT_RECOMBINATION_SPLIT_BIAS_CONFIG = {
    "enabled": False,
    "score_mode": RECOMBINATION_SPLIT_SCORE_MODE,
    "lineage_weight": 0.25,
    "breakpoint_weight": 0.25,
    "aggregation_temperature": 1.0,
    "fragmentation_penalty": 0.10,
}


def normalize_recombination_split_bias_config(config: Any) -> dict[str, Any]:
    """Validate and fill the public split-bias configuration."""

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("model.recombination_split_bias must be a mapping")
    unknown = sorted(set(config) - set(DEFAULT_RECOMBINATION_SPLIT_BIAS_CONFIG))
    if unknown:
        raise ValueError(
            "model.recombination_split_bias contains unknown fields: "
            + ", ".join(unknown)
        )

    normalized = dict(DEFAULT_RECOMBINATION_SPLIT_BIAS_CONFIG)
    normalized.update(dict(config))
    if not isinstance(normalized["enabled"], bool):
        raise ValueError(
            "model.recombination_split_bias.enabled must be a boolean"
        )
    score_mode = str(normalized["score_mode"])
    if score_mode != RECOMBINATION_SPLIT_SCORE_MODE:
        raise ValueError(
            "model.recombination_split_bias.score_mode must be "
            f"{RECOMBINATION_SPLIT_SCORE_MODE!r}"
        )
    normalized["score_mode"] = score_mode

    for field in (
        "lineage_weight",
        "breakpoint_weight",
        "aggregation_temperature",
        "fragmentation_penalty",
    ):
        raw = normalized[field]
        if isinstance(raw, bool):
            raise ValueError(
                f"model.recombination_split_bias.{field} must be a number"
            )
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"model.recombination_split_bias.{field} must be a number"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"model.recombination_split_bias.{field} must be finite"
            )
        normalized[field] = value

    if normalized["lineage_weight"] < 0.0:
        raise ValueError(
            "model.recombination_split_bias.lineage_weight must be nonnegative"
        )
    if normalized["breakpoint_weight"] < 0.0:
        raise ValueError(
            "model.recombination_split_bias.breakpoint_weight must be nonnegative"
        )
    if normalized["aggregation_temperature"] <= 0.0:
        raise ValueError(
            "model.recombination_split_bias.aggregation_temperature must be positive"
        )
    if normalized["fragmentation_penalty"] < 0.0:
        raise ValueError(
            "model.recombination_split_bias.fragmentation_penalty must be nonnegative"
        )
    if (
        normalized["enabled"]
        and normalized["lineage_weight"] == 0.0
        and normalized["breakpoint_weight"] == 0.0
    ):
        raise ValueError(
            "enabled model.recombination_split_bias requires a positive lineage "
            "or breakpoint weight"
        )
    return normalized


@dataclass(frozen=True)
class RecombinationSplitScore:
    """Candidate-aligned split scores for one recombination lineage."""

    breakpoints: tuple[int, ...]
    breakpoint_scores: torch.Tensor
    lineage_score: torch.Tensor

    def breakpoint_bias(self, weight: float) -> torch.Tensor:
        return self.breakpoint_scores * float(weight)

    def selected_score(self, breakpoint: int) -> torch.Tensor:
        try:
            index = self.breakpoints.index(int(breakpoint))
        except ValueError as exc:
            raise ValueError(
                f"breakpoint {breakpoint} is outside split-score support"
            ) from exc
        return self.breakpoint_scores[index]


@dataclass(frozen=True)
class _PairContributions:
    blocks: tuple[int, ...]
    prefix: tuple[float, ...]
    total: float

    def left_sum(self, breakpoint: int) -> float:
        index = bisect.bisect_left(self.blocks, int(breakpoint))
        return float(self.prefix[index])


class RecombinationSplitBiasScorer:
    """Compute partial-compatibility split evidence for local VCF states."""

    def __init__(self, env, config: Mapping[str, Any]):
        self.env = env
        self.config = normalize_recombination_split_bias_config(config)
        if self.config["enabled"] and not (
            bool(getattr(env, "is_local", False))
            and str(getattr(env, "input_mode", "")) == "vcf"
        ):
            raise ValueError(
                "model.recombination_split_bias is supported only for local VCF "
                "ARG refinement"
            )

    @property
    def enabled(self) -> bool:
        return bool(self.config["enabled"])

    def score_candidates(
        self,
        state,
        candidate_actions: Sequence[Any],
        *,
        device,
        dtype,
    ) -> tuple[RecombinationSplitScore | None, ...]:
        if not self.enabled:
            return tuple(None for _ in candidate_actions)

        with torch.no_grad():
            lineage_cache = self._lineage_cache(state)
            pair_cache = self._pair_cache(state, lineage_cache)
            records: list[RecombinationSplitScore | None] = []
            recombination_cache = {}
            for action in candidate_actions:
                if not isinstance(action, RecombinationChoice):
                    records.append(None)
                    continue
                cache_key = (
                    int(action.active_lineage_i),
                    int(action.span_start),
                    int(action.span_end),
                )
                if cache_key not in recombination_cache:
                    recombination_cache[cache_key] = self._score_recombination(
                        state,
                        action,
                        lineage_cache,
                        pair_cache,
                        device=device,
                        dtype=dtype,
                    )
                records.append(recombination_cache[cache_key])
            return tuple(records)

    def _lineage_cache(self, state):
        output = []
        block_map = state.variant_block_indices
        for lineage in state.active_lineages:
            variants = tuple(int(value) for value in lineage.variant_indices)
            blocks = tuple(sorted(int(block_map[value]) for value in variants))
            partials = lineage.partials
            if not torch.is_tensor(partials):
                partials = torch.as_tensor(partials, dtype=torch.float32)
            partials = partials.detach()
            if tuple(partials.shape) != (len(variants), 4):
                raise ValueError(
                    "local VCF lineage partials must have shape "
                    f"({len(variants)}, 4), got {tuple(partials.shape)}"
                )
            output.append(
                {
                    "variants": variants,
                    "blocks": blocks,
                    "positions": {
                        variant: index for index, variant in enumerate(variants)
                    },
                    "partials": partials,
                    "material": lineage.material_segments,
                }
            )
        return tuple(output)

    def _pair_cache(self, state, lineage_cache):
        output: dict[tuple[int, int], _PairContributions] = {}
        for first_index, first in enumerate(lineage_cache):
            first_variants = set(first["variants"])
            for second_index in range(first_index + 1, len(lineage_cache)):
                second = lineage_cache[second_index]
                common = sorted(
                    first_variants.intersection(second["variants"]),
                    key=lambda variant: state.variant_block_indices[int(variant)],
                )
                blocks = [
                    int(state.variant_block_indices[int(variant)])
                    for variant in common
                ]
                if common:
                    first_rows = torch.as_tensor(
                        [first["positions"][int(variant)] for variant in common],
                        dtype=torch.long,
                        device=first["partials"].device,
                    )
                    second_rows = torch.as_tensor(
                        [second["positions"][int(variant)] for variant in common],
                        dtype=torch.long,
                        device=second["partials"].device,
                    )
                    first_partials = first["partials"].index_select(
                        0,
                        first_rows,
                    )
                    second_partials = second["partials"].index_select(
                        0,
                        second_rows,
                    ).to(
                        device=first_partials.device,
                        dtype=first_partials.dtype,
                    )
                    values = (
                        2.0
                        * (first_partials * second_partials)
                        .sum(dim=1)
                        .clamp(0.0, 1.0)
                        - 1.0
                    ).detach().cpu().tolist()
                else:
                    values = []
                prefix = [0.0]
                for value in values:
                    prefix.append(prefix[-1] + float(value))
                contributions = _PairContributions(
                    blocks=tuple(blocks),
                    prefix=tuple(prefix),
                    total=float(prefix[-1]),
                )
                output[(first_index, second_index)] = contributions
                output[(second_index, first_index)] = contributions
        return output

    def _score_recombination(
        self,
        state,
        action,
        lineage_cache,
        pair_cache,
        *,
        device,
        dtype,
    ) -> RecombinationSplitScore:
        lineage_index = int(action.active_lineage_i)
        lineage = state.active_lineages[lineage_index]
        cached = lineage_cache[lineage_index]
        valid_breakpoints = tuple(
            int(value) for value in self.env.valid_breakpoints(state, action)
        )
        if not valid_breakpoints:
            raise ValueError("recombination split scoring requires a valid breakpoint")
        prior_weights = tuple(
            float(value)
            for value in self.env.breakpoint_prior_weights(state, action)
        )
        if len(prior_weights) != len(valid_breakpoints):
            raise ValueError(
                "breakpoint prior weights do not align with valid breakpoints"
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in prior_weights):
            raise ValueError("breakpoint prior weights must be finite and positive")

        variant_blocks = cached["blocks"]
        variant_count = len(variant_blocks)
        whole_best = self._best_compatibility(
            lineage_index,
            cached["material"],
            variant_count,
            None,
            lineage_cache,
            pair_cache,
        )
        scores = []
        for breakpoint in valid_breakpoints:
            left_material, right_material = lineage.material_segments.split(breakpoint)
            left_count = bisect.bisect_left(variant_blocks, int(breakpoint))
            right_count = int(variant_count - left_count)
            left_best = self._best_compatibility(
                lineage_index,
                left_material,
                left_count,
                int(breakpoint),
                lineage_cache,
                pair_cache,
                side="left",
            )
            right_best = self._best_compatibility(
                lineage_index,
                right_material,
                right_count,
                int(breakpoint),
                lineage_cache,
                pair_cache,
                side="right",
            )
            split_compatibility = (
                (
                    float(left_count) * left_best
                    + float(right_count) * right_best
                )
                / float(variant_count)
                if variant_count > 0
                else 0.0
            )
            left_length = self._material_length(state, left_material)
            right_length = self._material_length(state, right_material)
            total_length = left_length + right_length
            imbalance = (
                abs(left_length - right_length) / total_length
                if total_length > 0.0
                else 0.0
            )
            scores.append(
                split_compatibility
                - whole_best
                - float(self.config["fragmentation_penalty"]) * imbalance
            )

        breakpoint_scores = torch.as_tensor(
            scores,
            device=device,
            dtype=dtype,
        ).detach()
        weights = torch.as_tensor(
            prior_weights,
            device=device,
            dtype=dtype,
        )
        weights = weights / weights.sum()
        temperature = float(self.config["aggregation_temperature"])
        lineage_score = temperature * torch.logsumexp(
            torch.log(weights) + breakpoint_scores / temperature,
            dim=0,
        )
        return RecombinationSplitScore(
            breakpoints=valid_breakpoints,
            breakpoint_scores=breakpoint_scores,
            lineage_score=lineage_score.detach(),
        )

    def _best_compatibility(
        self,
        lineage_index,
        material,
        variant_count,
        breakpoint,
        lineage_cache,
        pair_cache,
        side=None,
    ) -> float:
        if variant_count <= 0:
            return 0.0
        best = 0.0
        for partner_index, partner in enumerate(lineage_cache):
            if partner_index == lineage_index:
                continue
            if not material.overlaps(partner["material"]):
                continue
            contributions = pair_cache[(lineage_index, partner_index)]
            if breakpoint is None:
                numerator = contributions.total
            else:
                left_sum = contributions.left_sum(int(breakpoint))
                numerator = (
                    left_sum
                    if side == "left"
                    else contributions.total - left_sum
                )
            best = max(best, float(numerator) / float(variant_count))
        return float(best)

    @staticmethod
    def _material_length(state, material) -> float:
        boundaries = state.block_boundaries
        if boundaries is None:
            return float(material.count)
        return float(
            sum(
                max(float(boundaries[end]) - float(boundaries[start]), 0.0)
                for start, end in material.segments
            )
        )


__all__ = [
    "DEFAULT_RECOMBINATION_SPLIT_BIAS_CONFIG",
    "RECOMBINATION_SPLIT_SCORE_MODE",
    "RecombinationSplitBiasScorer",
    "RecombinationSplitScore",
    "normalize_recombination_split_bias_config",
]
