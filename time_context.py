"""Breakpoint-aware biological context for the continuous ARG time policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

try:
    from .env import CoalescenceChoice, MaterialSegments, RecombinationChoice
except ImportError:  # Script-style entry points.
    from env import CoalescenceChoice, MaterialSegments, RecombinationChoice


TIME_CONTEXT_VERSION = "breakpoint-aware-v1"
TIME_CONTEXT_MODES = (
    "baseline",
    "temporal",
    "breakpoint",
    "full",
    "likelihood",
)

LINEAGE_FEATURE_NAMES = (
    "log1p_lineage_age",
    "log1p_descendant_count",
    "carried_span_fraction",
    "log1p_carried_span_relative",
    "log1p_interval_count",
    "log1p_variant_count",
    "log1p_next_fixed_distance",
)

TEMPORAL_FEATURE_NAMES = (
    "event_is_coalescence",
    "event_is_recombination",
    "log1p_current_time",
    "log1p_minimum_time",
    "log1p_maximum_time_or_zero",
    "has_finite_upper_bound",
    "available_window_squashed",
    "log1p_available_window",
    "log1p_next_fixed_distance",
    "normalized_lower_bound_location",
    "log1p_minimum_child_age",
    "log1p_maximum_child_age",
    "log1p_child_age_range",
    "log1p_2ne_mu",
    "log1p_2ne_recombination_rate",
    "local_window_fraction",
    "log1p_local_variant_count",
    "callable_window_fraction",
) + tuple(f"lineage_sum_{name}" for name in LINEAGE_FEATURE_NAMES) + tuple(
    f"lineage_absdiff_{name}" for name in LINEAGE_FEATURE_NAMES
) + tuple(f"lineage_product_{name}" for name in LINEAGE_FEATURE_NAMES) + (
    "pair_intersection_fraction",
    "pair_union_fraction",
    "pair_overlap_to_union",
    "log1p_joint_variant_count",
    "log1p_shared_derived_count",
    "log1p_discordant_allele_count",
    "log1p_descendant_count_sum",
    "log1p_descendant_count_difference",
    "log1p_child_age_difference",
    "pair_has_zero_overlap",
)

BREAKPOINT_FEATURE_NAMES = (
    "left_carried_span_fraction",
    "right_carried_span_fraction",
    "log1p_left_interval_count",
    "log1p_right_interval_count",
    "log1p_left_variant_count",
    "log1p_right_variant_count",
    "left_material_fraction",
    "right_material_fraction",
    "material_imbalance",
    "nearest_variant_left_distance_fraction",
    "nearest_variant_right_distance_fraction",
    "neighboring_variant_gap_fraction",
    "normalized_breakpoint_position",
    "breakpoint_in_variant_free_gap",
    "log1p_left_next_fixed_distance",
    "log1p_right_next_fixed_distance",
)

LIKELIHOOD_FEATURE_NAMES = (
    "signed_log1p_provisional_likelihood_q10",
    "signed_log1p_provisional_likelihood_q50",
    "signed_log1p_provisional_likelihood_q90",
    "signed_log1p_recent_minus_ancient_likelihood",
    "provisional_likelihood_spread_squashed",
    "provisional_best_quantile",
)


def time_context_feature_names(mode: str) -> tuple[str, ...]:
    mode = _validate_mode(mode)
    if mode == "baseline":
        return ()
    if mode == "temporal":
        return TEMPORAL_FEATURE_NAMES
    if mode == "breakpoint":
        return BREAKPOINT_FEATURE_NAMES
    if mode == "full":
        return TEMPORAL_FEATURE_NAMES + BREAKPOINT_FEATURE_NAMES
    return (
        TEMPORAL_FEATURE_NAMES
        + BREAKPOINT_FEATURE_NAMES
        + LIKELIHOOD_FEATURE_NAMES
    )


def time_context_dim(mode: str) -> int:
    return len(time_context_feature_names(mode))


@dataclass(frozen=True)
class TimeContext:
    features: torch.Tensor
    diagnostics: dict[str, float | bool | str | None]


def build_time_context(
    state,
    selected_action,
    env,
    *,
    max_delta: float | None,
    mode: str = "full",
    sampled_breakpoint: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> TimeContext:
    """Build finite post-breakpoint inputs for one generated event.

    Times are in the environment's existing ``t/(2Ne)`` units.  The returned
    tensor contains no learned values and deliberately leaves the existing
    action representation and CwR rate/horizon features untouched.
    """

    mode = _validate_mode(mode)
    device = torch.device(env.device if device is None else device)
    if mode == "baseline":
        return TimeContext(
            features=torch.empty(0, device=device, dtype=dtype),
            diagnostics=_base_diagnostics(state, selected_action, max_delta),
        )

    if not isinstance(selected_action, (CoalescenceChoice, RecombinationChoice)):
        raise TypeError("time context requires a generated ARG action")
    breakpoint = (
        sampled_breakpoint
        if sampled_breakpoint is not None
        else getattr(selected_action, "breakpoint", None)
    )
    if isinstance(selected_action, RecombinationChoice) and breakpoint is None:
        raise ValueError(
            "recombination time context must be built after breakpoint sampling"
        )

    lineages = _selected_lineages(state, selected_action)
    local_length = _local_window_length(state, env)
    local_variant_count = _local_variant_count(state)
    current_time = _finite_nonnegative(state.current_time)
    child_ages = [_finite_nonnegative(lineage.time) for lineage in lineages]
    minimum_child_age = min(child_ages, default=current_time)
    maximum_child_age = max(child_ages, default=current_time)
    minimum_time = max(current_time, maximum_child_age)
    finite_upper = max_delta is not None
    window = 0.0 if max_delta is None else _finite_nonnegative(max_delta)
    maximum_time = current_time + window if finite_upper else 0.0
    normalized_lower = (
        _safe_ratio(current_time - minimum_time, maximum_time - minimum_time)
        if finite_upper
        else 0.0
    )
    population_size = _finite_nonnegative(getattr(env, "population_size", 0.0))
    mutation_rate = _finite_nonnegative(getattr(env, "mutation_rate", 0.0))
    recombination_rate = _finite_nonnegative(
        getattr(env, "recombination_rate", 0.0)
    )
    sequence_length = max(_finite_nonnegative(env.sequence_length), 1.0)

    if len(lineages) == 1:
        first_lineage = _lineage_features(
            state, lineages[0], local_length
        )
        second_lineage = [0.0] * len(LINEAGE_FEATURE_NAMES)
    else:
        first_lineage = _lineage_features(
            state, lineages[0], local_length
        )
        second_lineage = _lineage_features(
            state, lineages[1], local_length
        )
    symmetric_lineages = (
        [left + right for left, right in zip(first_lineage, second_lineage)]
        + [abs(left - right) for left, right in zip(first_lineage, second_lineage)]
        + [left * right for left, right in zip(first_lineage, second_lineage)]
    )

    temporal_values = [
        float(isinstance(selected_action, CoalescenceChoice)),
        float(isinstance(selected_action, RecombinationChoice)),
        math.log1p(current_time),
        math.log1p(minimum_time),
        math.log1p(maximum_time) if finite_upper else 0.0,
        float(finite_upper),
        _squash(window),
        math.log1p(window),
        math.log1p(window),
        normalized_lower,
        math.log1p(minimum_child_age),
        math.log1p(maximum_child_age),
        math.log1p(maximum_child_age - minimum_child_age),
        math.log1p(2.0 * population_size * mutation_rate),
        math.log1p(2.0 * population_size * recombination_rate),
        min(local_length / sequence_length, 1.0),
        math.log1p(local_variant_count),
        min(local_length / sequence_length, 1.0),
        *symmetric_lineages,
    ]
    pair_values, pair_diagnostics = _pair_features(
        state,
        selected_action,
        lineages,
        env,
        local_length,
        device,
        dtype,
    )
    temporal_tensor = torch.cat(
        [
            torch.as_tensor(
                temporal_values,
                device=device,
                dtype=dtype,
            ),
            pair_values,
        ]
    )

    breakpoint_values, breakpoint_diagnostics = _breakpoint_features(
        state,
        selected_action,
        lineages[0],
        env,
        breakpoint,
        local_length,
    )
    breakpoint_tensor = torch.as_tensor(
        breakpoint_values,
        device=device,
        dtype=dtype,
    )
    if mode == "temporal":
        features = temporal_tensor
    elif mode == "breakpoint":
        features = breakpoint_tensor
    else:
        features = torch.cat([temporal_tensor, breakpoint_tensor])
    likelihood_diagnostics: dict[str, float] = {}
    if mode == "likelihood":
        likelihood_values, likelihood_diagnostics = _provisional_likelihood_features(
            state,
            selected_action,
            env,
            max_delta=max_delta,
        )
        features = torch.cat(
            [
                features,
                torch.as_tensor(likelihood_values, device=device, dtype=dtype),
            ]
        )
    expected = time_context_dim(mode)
    if tuple(features.shape) != (expected,):
        raise RuntimeError(
            f"time context mode {mode!r} produced shape {tuple(features.shape)}, "
            f"expected {(expected,)}"
        )
    if not bool(torch.isfinite(features).all().detach().cpu().item()):
        raise ValueError("time context contains a non-finite feature")

    diagnostics = {
        **_base_diagnostics(state, selected_action, max_delta),
        "time_context_mode": mode,
        "minimum_event_time": minimum_time,
        "maximum_event_time": maximum_time if finite_upper else None,
        "child_age_min": minimum_child_age,
        "child_age_max": maximum_child_age,
        "lineage_variant_count": float(
            sum(
                len(
                    _variants_in_material(
                        state,
                        lineage.variant_indices,
                        _focus_material(state, lineage.material_segments),
                    )
                )
                for lineage in lineages
            )
        ),
        "lineage_interval_count": float(
            sum(
                len(_focus_material(state, lineage.material_segments).segments)
                for lineage in lineages
            )
        ),
        "lineage_carried_span_fraction": _safe_ratio(
            sum(
                _material_length(
                    state,
                    _focus_material(state, lineage.material_segments),
                )
                for lineage in lineages
            ),
            local_length,
        ),
        **pair_diagnostics,
        **breakpoint_diagnostics,
        **likelihood_diagnostics,
    }
    return TimeContext(features=features, diagnostics=diagnostics)


def _provisional_likelihood_features(
    state,
    action,
    env,
    *,
    max_delta: float | None,
) -> tuple[list[float], dict[str, float]]:
    """Score three one-step time probes without changing the live state."""
    options = env.enumerate_prior_options(state)
    rate = float(
        options.rates["lambda_coal"] + options.rates["lambda_recomb"]
    )
    increments: list[float] = []
    quantiles = (0.1, 0.5, 0.9)
    candidates = env.enumerate_actions(state)
    for quantile in quantiles:
        delta = env.time_env.quantile_to_delta(
            quantile,
            rate,
            max_delta=max_delta,
        )
        probe = _replace_time(action, quantile, delta, rate, max_delta)
        try:
            log_prior = env.compute_cwr_event_log_prior(
                state,
                candidates,
                probe,
            )
            next_state = env.apply_action(
                state,
                probe,
                log_prior=log_prior,
            )
            increment = float(
                next_state.accumulated_log_likelihood
                - state.accumulated_log_likelihood
            )
        except (ValueError, RuntimeError, IndexError):
            increment = 0.0
        increments.append(increment if math.isfinite(increment) else 0.0)
    low, middle, high = increments
    spread = max(increments) - min(increments)
    best_index = max(range(len(increments)), key=increments.__getitem__)
    values = [
        _signed_log1p(low),
        _signed_log1p(middle),
        _signed_log1p(high),
        _signed_log1p(low - high),
        _squash(spread),
        quantiles[best_index],
    ]
    return values, {
        "provisional_likelihood_q10": low,
        "provisional_likelihood_q50": middle,
        "provisional_likelihood_q90": high,
        "provisional_likelihood_spread": spread,
        "provisional_best_quantile": quantiles[best_index],
    }


def _replace_time(action, quantile, delta, rate, max_delta):
    from dataclasses import replace

    return replace(
        action,
        time_quantile=float(quantile),
        delta_time=float(delta),
        waiting_rate=float(rate),
        fixed_horizon=None if max_delta is None else float(max_delta),
    )


def _signed_log1p(value: float) -> float:
    value = float(value)
    return math.copysign(math.log1p(abs(value)), value)


def _pair_features(
    state,
    action,
    lineages,
    env,
    local_length: float,
    device,
    dtype,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not isinstance(action, CoalescenceChoice):
        return (
            torch.zeros(10, device=device, dtype=dtype),
            {"pair_overlap_fraction": 0.0, "descendant_count": float(len(lineages[0].sequences_indices))},
        )
    left, right = lineages
    left_material = _focus_material(state, left.material_segments)
    right_material = _focus_material(state, right.material_segments)
    intersection = left_material.intersection(right_material)
    union = left_material.union(right_material)
    intersection_length = _material_length(state, intersection)
    union_length = _material_length(state, union)
    intersection_fraction = _safe_ratio(intersection_length, local_length)
    union_fraction = _safe_ratio(union_length, local_length)
    overlap_ratio = _safe_ratio(intersection_length, union_length)
    left_variants = set(int(value) for value in left.variant_indices)
    right_variants = set(int(value) for value in right.variant_indices)
    joint = sorted(left_variants.intersection(right_variants))
    shared_derived, discordant = _allele_compatibility_counts(
        left,
        right,
        joint,
        env,
        device,
    )
    left_descendants = len(left.sequences_indices)
    right_descendants = len(right.sequences_indices)
    values = torch.stack(
        [
            torch.as_tensor(intersection_fraction, device=device, dtype=dtype),
            torch.as_tensor(union_fraction, device=device, dtype=dtype),
            torch.as_tensor(overlap_ratio, device=device, dtype=dtype),
            torch.as_tensor(math.log1p(len(joint)), device=device, dtype=dtype),
            torch.log1p(shared_derived.to(dtype=dtype)),
            torch.log1p(discordant.to(dtype=dtype)),
            torch.as_tensor(
                math.log1p(left_descendants + right_descendants),
                device=device,
                dtype=dtype,
            ),
            torch.as_tensor(
                math.log1p(abs(left_descendants - right_descendants)),
                device=device,
                dtype=dtype,
            ),
            torch.as_tensor(
                math.log1p(abs(float(left.time) - float(right.time))),
                device=device,
                dtype=dtype,
            ),
            torch.as_tensor(float(intersection_length <= 0.0), device=device, dtype=dtype),
        ]
    )
    return values, {
        "pair_overlap_fraction": overlap_ratio,
        "pair_intersection_span": intersection_length,
        "pair_union_span": union_length,
        "joint_variant_count": float(len(joint)),
        "shared_derived_count": float(shared_derived.detach().cpu().item()),
        "discordant_allele_count": float(discordant.detach().cpu().item()),
        "descendant_count": float(left_descendants + right_descendants),
    }


def _breakpoint_features(
    state,
    action,
    lineage,
    env,
    breakpoint,
    local_length: float,
) -> tuple[list[float], dict[str, float]]:
    if not isinstance(action, RecombinationChoice):
        return [0.0] * len(BREAKPOINT_FEATURE_NAMES), {
            "breakpoint_gap_fraction": 0.0,
        }
    breakpoint = int(breakpoint)
    material = _focus_material(state, lineage.material_segments)
    left, right = material.split(breakpoint)
    left_length = _material_length(state, left)
    right_length = _material_length(state, right)
    total_length = max(left_length + right_length, 0.0)
    left_variants = _variants_in_material(state, lineage.variant_indices, left)
    right_variants = _variants_in_material(state, lineage.variant_indices, right)
    coordinate = _block_coordinate(state, breakpoint)
    target_left, target_right = _local_interval(state, env)
    variant_positions = _variant_positions(env, state)
    carried_positions = [
        variant_positions[index]
        for index in lineage.variant_indices
        if int(index) in variant_positions
    ]
    positions_left = [value for value in carried_positions if value < coordinate]
    positions_right = [value for value in carried_positions if value >= coordinate]
    nearest_left = (
        coordinate - max(positions_left) if positions_left else local_length
    )
    nearest_right = (
        min(positions_right) - coordinate if positions_right else local_length
    )
    gap = min(max(nearest_left + nearest_right, 0.0), local_length)
    left_fixed = _next_fixed_distance(state, left)
    right_fixed = _next_fixed_distance(state, right)
    values = [
        _safe_ratio(left_length, local_length),
        _safe_ratio(right_length, local_length),
        math.log1p(len(left.segments)),
        math.log1p(len(right.segments)),
        math.log1p(len(left_variants)),
        math.log1p(len(right_variants)),
        _safe_ratio(left_length, total_length),
        _safe_ratio(right_length, total_length),
        _safe_ratio(abs(left_length - right_length), total_length),
        _safe_ratio(nearest_left, local_length),
        _safe_ratio(nearest_right, local_length),
        _safe_ratio(gap, local_length),
        _safe_ratio(coordinate - target_left, target_right - target_left),
        float(nearest_left > 0.0 and nearest_right > 0.0),
        math.log1p(left_fixed),
        math.log1p(right_fixed),
    ]
    return values, {
        "breakpoint_gap_fraction": _safe_ratio(gap, local_length),
        "breakpoint_gap_length": gap,
        "breakpoint_position": coordinate,
        "left_carried_span": left_length,
        "right_carried_span": right_length,
        "left_variant_count": float(len(left_variants)),
        "right_variant_count": float(len(right_variants)),
    }


def _lineage_features(state, lineage, local_length: float) -> list[float]:
    material = _focus_material(state, lineage.material_segments)
    span = _material_length(state, material)
    variant_count = len(
        _variants_in_material(state, lineage.variant_indices, material)
    )
    return [
        math.log1p(_finite_nonnegative(lineage.time)),
        math.log1p(len(lineage.sequences_indices)),
        _safe_ratio(span, local_length),
        math.log1p(_safe_ratio(span, local_length)),
        math.log1p(len(material.segments)),
        math.log1p(variant_count),
        math.log1p(_next_fixed_distance(state, material)),
    ]


def _allele_compatibility_counts(left, right, variants, env, device):
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if not variants or left.partials is None or right.partials is None:
        return zero, zero
    left_rows = {int(value): index for index, value in enumerate(left.variant_indices)}
    right_rows = {int(value): index for index, value in enumerate(right.variant_indices)}
    common = [value for value in variants if value in left_rows and value in right_rows]
    if not common:
        return zero, zero
    left_index = torch.as_tensor(
        [left_rows[value] for value in common], device=device, dtype=torch.long
    )
    right_index = torch.as_tensor(
        [right_rows[value] for value in common], device=device, dtype=torch.long
    )
    left_partials = torch.as_tensor(left.partials, device=device).index_select(0, left_index)
    right_partials = torch.as_tensor(right.partials, device=device).index_select(0, right_index)
    left_mode = left_partials.argmax(dim=1)
    right_mode = right_partials.argmax(dim=1)
    ref_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    refs = getattr(getattr(env, "variant_data", None), "refs", ())
    reference = torch.as_tensor(
        [ref_map.get(str(refs[index]).upper(), -1) for index in common],
        device=device,
        dtype=torch.long,
    )
    matching = left_mode == right_mode
    shared_derived = (matching & (left_mode != reference)).sum().to(torch.float32)
    discordant = (~matching).sum().to(torch.float32)
    return shared_derived, discordant


def _selected_lineages(state, action):
    if isinstance(action, CoalescenceChoice):
        indices = (int(action.active_lineage_i), int(action.active_lineage_j))
    else:
        indices = (int(action.active_lineage_i),)
    if any(index < 0 or index >= len(state.active_lineages) for index in indices):
        raise ValueError("selected action references an unavailable lineage")
    return tuple(state.active_lineages[index] for index in indices)


def _focus_material(state, material):
    material = MaterialSegments.from_segments(material)
    if state.target_material is not None:
        return material.intersection(state.target_material)
    return material


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


def _block_coordinate(state, block: int) -> float:
    if state.block_boundaries is None:
        return float(block)
    block = min(max(int(block), 0), len(state.block_boundaries) - 1)
    return float(state.block_boundaries[block])


def _variants_in_material(state, variant_indices, material) -> tuple[int, ...]:
    output = []
    for variant in variant_indices:
        block = state.variant_block_indices.get(int(variant))
        if block is not None and material.covers_interval(int(block), int(block) + 1):
            output.append(int(variant))
    return tuple(output)


def _next_fixed_distance(state, material) -> float:
    distances = []
    for record in state.fixed_ancestor_schedule:
        if int(record["node_id"]) in state.all_nodes:
            continue
        fixed_material = MaterialSegments.from_segments(record.get("segments", ()))
        if not material.overlaps(fixed_material):
            continue
        distances.append(max(float(record["time"]) - float(state.current_time), 0.0))
    return min(distances, default=0.0)


def _local_interval(state, env) -> tuple[float, float]:
    interval = state.local_target_interval
    if interval is None and state.block_boundaries is not None:
        interval = (state.block_boundaries[0], state.block_boundaries[-1])
    if interval is None:
        interval = (0.0, float(env.sequence_length))
    left, right = float(interval[0]), float(interval[1])
    if not right > left:
        return 0.0, max(float(env.sequence_length), 1.0)
    return left, right


def _local_window_length(state, env) -> float:
    left, right = _local_interval(state, env)
    return max(right - left, 1.0)


def _local_variant_count(state) -> int:
    return len(state.target_variant_indices)


def _variant_positions(env, state) -> dict[int, float]:
    offset = float(getattr(state, "vcf_alignment", {}).get("vcf_coordinate_offset", 0.0))
    return {
        int(index): float(position) + offset
        for index, position in enumerate(getattr(env, "variant_positions0", ()))
    }


def _base_diagnostics(state, action, max_delta):
    current = _finite_nonnegative(state.current_time)
    return {
        "event_type": "coal" if isinstance(action, CoalescenceChoice) else "recomb",
        "current_time": current,
        "minimum_event_time": current,
        "maximum_event_time": (
            None if max_delta is None else current + _finite_nonnegative(max_delta)
        ),
        "finite_upper_bound": max_delta is not None,
        "available_time_window": (
            0.0 if max_delta is None else _finite_nonnegative(max_delta)
        ),
    }


def _validate_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in TIME_CONTEXT_MODES:
        raise ValueError(
            f"time_context_mode must be one of {TIME_CONTEXT_MODES}, got {mode!r}"
        )
    return mode


def _finite_nonnegative(value: Any) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return max(value, 0.0)


def _safe_ratio(numerator: float, denominator: float) -> float:
    numerator = _finite_nonnegative(numerator)
    denominator = _finite_nonnegative(denominator)
    if denominator <= 0.0:
        return 0.0
    return min(max(numerator / denominator, 0.0), 1.0)


def _squash(value: float) -> float:
    value = _finite_nonnegative(value)
    return value / (1.0 + value)


__all__ = [
    "BREAKPOINT_FEATURE_NAMES",
    "TEMPORAL_FEATURE_NAMES",
    "TIME_CONTEXT_MODES",
    "TIME_CONTEXT_VERSION",
    "TimeContext",
    "build_time_context",
    "time_context_dim",
    "time_context_feature_names",
]
