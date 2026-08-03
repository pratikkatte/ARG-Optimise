from types import SimpleNamespace

import pytest
import torch

from env import (
    ARGLineage,
    ARGState,
    CoalescenceChoice,
    MaterialSegments,
    RecombinationChoice,
)
from time_context import (
    BREAKPOINT_FEATURE_NAMES,
    TIME_CONTEXT_MODES,
    build_time_context,
    time_context_dim,
    time_context_feature_names,
)


def _lineage(node_id, segments, variants, descendants, age, allele_rows):
    return ARGLineage(
        node_id=node_id,
        children=[],
        parents=[],
        material_segments=MaterialSegments.from_segments(segments),
        num_blocks=5,
        partials=torch.as_tensor(allele_rows, dtype=torch.float32),
        variant_indices=tuple(variants),
        sequences_indices=list(descendants),
        event_type="cut",
        time=float(age),
    )


def _state(*, with_variants=True):
    variants_a = (0, 1, 2, 4) if with_variants else ()
    variants_b = (1, 2, 3, 4) if with_variants else ()
    rows_a = (
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        if with_variants
        else torch.empty((0, 4))
    )
    rows_b = (
        [[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1]]
        if with_variants
        else torch.empty((0, 4))
    )
    first = _lineage(0, ((0, 3), (4, 5)), variants_a, (0, 1), 0.4, rows_a)
    second = _lineage(1, ((1, 5),), variants_b, (2, 3, 4), 0.6, rows_b)
    return ARGState(
        active_lineages=[first, second],
        all_nodes={0: first, 1: second},
        max_node_idx=1,
        current_time=0.75,
        target_material=MaterialSegments(((0, 5),)),
        block_boundaries=(0.0, 20.0, 40.0, 60.0, 80.0, 100.0),
        fixed_ancestor_schedule=[
            {"node_id": 9, "time": 1.25, "segments": ((0, 5),)}
        ],
        target_variant_indices=(0, 1, 2, 3, 4) if with_variants else (),
        variant_block_indices=(
            {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
            if with_variants
            else {}
        ),
        local_target_interval=(0.0, 100.0),
        vcf_alignment={"vcf_coordinate_offset": 0.0},
    )


def _env(*, with_variants=True):
    return SimpleNamespace(
        device=torch.device("cpu"),
        sequence_length=100.0,
        population_size=10_000.0,
        mutation_rate=2e-8,
        recombination_rate=2e-8,
        variant_positions0=(10.0, 30.0, 50.0, 70.0, 90.0)
        if with_variants
        else (),
        variant_data=SimpleNamespace(refs=("A", "A", "A", "A", "A"))
        if with_variants
        else SimpleNamespace(refs=()),
    )


def test_time_context_modes_have_stable_finite_shapes():
    state = _state()
    action = CoalescenceChoice(0, 1)
    for mode in (mode for mode in TIME_CONTEXT_MODES if mode != "likelihood"):
        result = build_time_context(
            state,
            action,
            _env(),
            max_delta=None,
            mode=mode,
        )
        assert result.features.shape == (time_context_dim(mode),)
        assert torch.isfinite(result.features).all()
        assert result.diagnostics["maximum_event_time"] is None
        assert not result.diagnostics["finite_upper_bound"]

    assert time_context_dim("likelihood") == time_context_dim("full") + 6


def test_coalescence_time_context_is_invariant_to_pair_order():
    state = _state()
    forward = build_time_context(
        state,
        CoalescenceChoice(0, 1),
        _env(),
        max_delta=0.5,
        mode="full",
    )
    reverse = build_time_context(
        state,
        CoalescenceChoice(1, 0),
        _env(),
        max_delta=0.5,
        mode="full",
    )
    assert torch.equal(forward.features, reverse.features)


def test_recombination_context_is_breakpoint_dependent_and_reconstructs_sides():
    state = _state()
    env = _env()
    first = build_time_context(
        state,
        RecombinationChoice(0, 4, 0, 4, breakpoint=1),
        env,
        max_delta=0.5,
        mode="breakpoint",
    )
    second = build_time_context(
        state,
        RecombinationChoice(0, 4, 0, 4, breakpoint=4),
        env,
        max_delta=0.5,
        mode="breakpoint",
    )
    assert not torch.equal(first.features, second.features)
    values = dict(zip(BREAKPOINT_FEATURE_NAMES, first.features.tolist()))
    assert values["left_carried_span_fraction"] == pytest.approx(0.2)
    assert values["right_carried_span_fraction"] == pytest.approx(0.6)
    assert values["log1p_left_variant_count"] == pytest.approx(torch.log1p(torch.tensor(1.0)).item())
    assert values["log1p_right_variant_count"] == pytest.approx(torch.log1p(torch.tensor(3.0)).item())


def test_time_context_handles_zero_variants_zero_overlap_and_narrow_bounds():
    state = _state(with_variants=False)
    state.active_lineages[1].material_segments = MaterialSegments(((3, 4),))
    result = build_time_context(
        state,
        CoalescenceChoice(0, 1),
        _env(with_variants=False),
        max_delta=1e-15,
        mode="full",
    )
    assert result.features.shape == (time_context_dim("full"),)
    assert torch.isfinite(result.features).all()
    names = time_context_feature_names("full")
    values = dict(zip(names, result.features.tolist()))
    assert values["log1p_local_variant_count"] == 0.0
    assert values["pair_overlap_to_union"] == 0.0
    assert values["pair_has_zero_overlap"] == 1.0


def test_recombination_context_requires_realized_breakpoint():
    with pytest.raises(ValueError, match="after breakpoint sampling"):
        build_time_context(
            _state(),
            RecombinationChoice(0, 4, 0, 4),
            _env(),
            max_delta=0.5,
            mode="full",
        )
