import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arg.env import (
    ARGLineage,
    ARGState,
    CoalescenceChoice,
    MaterialSegments,
    RecombinationChoice,
)
from arg.models import ARGModel
from arg.recombination_split_bias import (
    RecombinationSplitBiasScorer,
    RecombinationSplitScore,
    normalize_recombination_split_bias_config,
)
from arg.train import load_train_config, validate_train_config


ARG_ROOT = Path(__file__).resolve().parents[1]


class _SplitEnv:
    is_local = True
    input_mode = "vcf"

    def __init__(self, breakpoints=(1,), weights=None):
        self.breakpoints = tuple(int(value) for value in breakpoints)
        self.weights = tuple(
            float(value)
            for value in (
                weights if weights is not None else [1.0] * len(self.breakpoints)
            )
        )

    def valid_breakpoints(self, state, action):
        return self.breakpoints

    def breakpoint_prior_weights(self, state, action):
        return self.weights


def _lineage(node_id, rows):
    return ARGLineage(
        node_id=node_id,
        material_segments=MaterialSegments(((0, len(rows)),)),
        num_blocks=len(rows),
        partials=torch.as_tensor(rows, dtype=torch.float32),
        variant_indices=tuple(range(len(rows))),
    )


def _state(lineages):
    variant_count = len(lineages[0].variant_indices)
    return ARGState(
        active_lineages=list(lineages),
        all_nodes={lineage.node_id: lineage for lineage in lineages},
        max_node_idx=max(lineage.node_id for lineage in lineages),
        block_boundaries=tuple(float(value) for value in range(variant_count + 1)),
        variant_block_indices={index: index for index in range(variant_count)},
        local_breakpoint_weights={index: 1.0 for index in range(1, variant_count)},
        likelihood_scope="target",
    )


def _enabled_config(**updates):
    config = {"enabled": True, "fragmentation_penalty": 0.0}
    config.update(updates)
    return normalize_recombination_split_bias_config(config)


def test_split_score_detects_different_left_and_right_partners():
    target = _lineage(0, [[1, 0, 0, 0], [0, 0, 1, 0]])
    left_partner = _lineage(1, [[1, 0, 0, 0], [0, 1, 0, 0]])
    right_partner = _lineage(2, [[0, 1, 0, 0], [0, 0, 1, 0]])
    state = _state([target, left_partner, right_partner])
    action = RecombinationChoice(0, 2, 0, 1)
    scorer = RecombinationSplitBiasScorer(_SplitEnv(), _enabled_config())

    record = scorer.score_candidates(
        state,
        [action],
        device="cpu",
        dtype=torch.float32,
    )[0]

    assert record.breakpoints == (1,)
    assert record.breakpoint_scores.tolist() == pytest.approx([1.0])
    assert float(record.lineage_score) == pytest.approx(1.0)


def test_split_score_does_not_reward_one_partner_that_fits_both_sides():
    target = _lineage(0, [[1, 0, 0, 0], [0, 0, 1, 0]])
    same_partner = _lineage(1, [[1, 0, 0, 0], [0, 0, 1, 0]])
    state = _state([target, same_partner])
    action = RecombinationChoice(0, 2, 0, 1)
    scorer = RecombinationSplitBiasScorer(_SplitEnv(), _enabled_config())

    record = scorer.score_candidates(
        state,
        [action],
        device="cpu",
        dtype=torch.float32,
    )[0]

    assert record.breakpoint_scores.tolist() == pytest.approx([0.0])
    assert float(record.lineage_score) == pytest.approx(0.0)


def test_prior_weighted_logmeanexp_is_not_breakpoint_count_biased():
    target = _lineage(
        0,
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
    )
    partner = _lineage(
        1,
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
    )
    state = _state([target, partner])
    action = RecombinationChoice(0, 3, 0, 2)
    scorer = RecombinationSplitBiasScorer(
        _SplitEnv((1, 2), (1.0, 9.0)),
        _enabled_config(),
    )

    record = scorer.score_candidates(
        state,
        [action],
        device="cpu",
        dtype=torch.float32,
    )[0]

    assert record.breakpoint_scores.tolist() == pytest.approx([0.0, 0.0])
    assert float(record.lineage_score) == pytest.approx(0.0, abs=1e-7)


class _FixedSplitScorer:
    enabled = True

    def score_candidates(self, state, actions, *, device, dtype):
        return (
            None,
            RecombinationSplitScore(
                breakpoints=(1,),
                breakpoint_scores=torch.tensor([0.5], device=device, dtype=dtype),
                lineage_score=torch.tensor(0.5, device=device, dtype=dtype),
            ),
            RecombinationSplitScore(
                breakpoints=(1,),
                breakpoint_scores=torch.tensor([-0.5], device=device, dtype=dtype),
                lineage_score=torch.tensor(-0.5, device=device, dtype=dtype),
            ),
        )


class _ActionAlignedSplitScorer:
    enabled = True

    def score_candidates(self, state, actions, *, device, dtype):
        record = RecombinationSplitScore(
            breakpoints=(1,),
            breakpoint_scores=torch.tensor([0.5], device=device, dtype=dtype),
            lineage_score=torch.tensor(0.5, device=device, dtype=dtype),
        )
        return tuple(
            record if isinstance(action, RecombinationChoice) else None
            for action in actions
        )


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_atomic_split_bias_preserves_recombination_mass_at_sampling_temperature(
    temperature,
):
    fake_model = SimpleNamespace(
        recombination_split_bias=_FixedSplitScorer(),
        recombination_split_bias_config={"lineage_weight": 0.5},
    )
    actions = [[
        CoalescenceChoice(0, 1),
        RecombinationChoice(0, 2, 0, 1),
        RecombinationChoice(1, 2, 0, 1),
    ]]
    logits = torch.tensor([[0.2, -0.1, 0.4]], dtype=torch.float64)

    adjusted, records, diagnostics = ARGModel.prepare_action_probability_logits(
        fake_model,
        logits,
        actions,
        [object()],
        random_spec={"T": temperature},
    )

    before = torch.softmax(logits / temperature, dim=1)[0, 1:].sum()
    after = torch.softmax(adjusted, dim=1)[0, 1:].sum()
    assert float(after) == pytest.approx(float(before), abs=1e-12)
    assert not torch.allclose(
        torch.softmax(adjusted, dim=1)[0, 1:],
        torch.softmax(logits / temperature, dim=1)[0, 1:],
    )
    assert records[0][1] is not None
    assert diagnostics[0]["recombination_split_mass_absolute_error"] < 1e-12


def test_disabled_atomic_split_bias_is_exact_no_op():
    fake_model = SimpleNamespace(
        recombination_split_bias=SimpleNamespace(enabled=False),
    )
    actions = [[CoalescenceChoice(0, 1), RecombinationChoice(0, 2, 0, 1)]]
    logits = torch.tensor([[0.2, -0.1]], dtype=torch.float64)

    adjusted, records, diagnostics = ARGModel.prepare_action_probability_logits(
        fake_model,
        logits,
        actions,
        None,
        random_spec={"T": 0.7},
    )

    assert torch.equal(adjusted, logits / 0.7)
    assert records == ((None, None),)
    assert diagnostics == (({"recombination_split_bias_enabled": False}),)


@pytest.mark.parametrize(
    "actions",
    [
        [CoalescenceChoice(0, 1)],
        [CoalescenceChoice(0, 1), RecombinationChoice(0, 2, 0, 1)],
    ],
)
def test_atomic_split_bias_handles_zero_or_one_recombination_candidate(actions):
    fake_model = SimpleNamespace(
        recombination_split_bias=_ActionAlignedSplitScorer(),
        recombination_split_bias_config={"lineage_weight": 0.5},
    )
    logits = torch.tensor([[0.2, -0.1]][:1], dtype=torch.float64)
    logits = logits[:, : len(actions)]

    adjusted, _, diagnostics = ARGModel.prepare_action_probability_logits(
        fake_model,
        logits,
        [actions],
        [object()],
        random_spec={"T": 0.7},
    )

    assert torch.allclose(adjusted, logits / 0.7, atol=1e-12, rtol=0.0)
    assert diagnostics[0]["recombination_split_mass_absolute_error"] < 1e-12


def test_breakpoint_bias_rejects_reordered_support():
    fake_model = SimpleNamespace(
        recombination_split_bias_config={"breakpoint_weight": 0.25}
    )
    record = RecombinationSplitScore(
        breakpoints=(1, 2),
        breakpoint_scores=torch.tensor([0.1, 0.2]),
        lineage_score=torch.tensor(0.0),
    )
    with pytest.raises(ValueError, match="breakpoint order"):
        ARGModel.recombination_breakpoint_logit_bias(
            fake_model,
            record,
            (2, 1),
        )


@pytest.mark.parametrize(
    "config, message",
    [
        ({"enabled": "yes"}, "enabled"),
        ({"enabled": True, "lineage_weight": 0, "breakpoint_weight": 0}, "positive"),
        ({"score_mode": "future_v2"}, "score_mode"),
        ({"lineage_weight": -1}, "lineage_weight"),
        ({"breakpoint_weight": float("nan")}, "finite"),
        ({"aggregation_temperature": 0}, "temperature"),
        ({"fragmentation_penalty": -1}, "fragmentation_penalty"),
        ({"unknown": 1}, "unknown fields"),
    ],
)
def test_split_bias_config_validation(config, message):
    with pytest.raises(ValueError, match=message):
        normalize_recombination_split_bias_config(config)


def test_enabled_split_bias_rejects_nonlocal_environment():
    with pytest.raises(ValueError, match="local VCF"):
        RecombinationSplitBiasScorer(
            SimpleNamespace(is_local=False, input_mode="vcf"),
            _enabled_config(),
        )


def test_paired_experiment_configs_differ_only_by_output_and_enablement():
    baseline = load_train_config(
        ARG_ROOT / "config/config_1mb_local_refinement_flow_consistency.yaml"
    )
    enabled = load_train_config(
        ARG_ROOT
        / "config/config_1mb_local_refinement_flow_consistency_split_bias.yaml"
    )
    validate_train_config(baseline)
    validate_train_config(enabled)
    assert not baseline["model"]["recombination_split_bias"]["enabled"]
    assert enabled["model"]["recombination_split_bias"]["enabled"]
    assert baseline["output_path"] != enabled["output_path"]

    normalized_baseline = copy.deepcopy(baseline)
    normalized_enabled = copy.deepcopy(enabled)
    normalized_enabled["output_path"] = normalized_baseline["output_path"]
    normalized_enabled["model"]["recombination_split_bias"]["enabled"] = False
    assert normalized_enabled == normalized_baseline


def test_split_scores_are_finite_without_other_lineages():
    target = _lineage(0, [[1, 0, 0, 0], [0, 0, 1, 0]])
    state = _state([target])
    scorer = RecombinationSplitBiasScorer(
        _SplitEnv(),
        _enabled_config(fragmentation_penalty=0.1),
    )
    record = scorer.score_candidates(
        state,
        [RecombinationChoice(0, 2, 0, 1)],
        device="cpu",
        dtype=torch.float32,
    )[0]
    assert torch.isfinite(record.breakpoint_scores).all()
    assert math.isfinite(float(record.lineage_score))


def test_empty_variant_lineage_retains_finite_fragmentation_penalty():
    target = ARGLineage(
        node_id=0,
        material_segments=MaterialSegments(((0, 2),)),
        num_blocks=2,
        partials=torch.empty((0, 4), dtype=torch.float32),
        variant_indices=(),
    )
    state = ARGState(
        active_lineages=[target],
        all_nodes={0: target},
        max_node_idx=0,
        block_boundaries=(0.0, 1.0, 3.0),
        variant_block_indices={},
        local_breakpoint_weights={1: 1.0},
        likelihood_scope="target",
    )
    scorer = RecombinationSplitBiasScorer(
        _SplitEnv(),
        _enabled_config(fragmentation_penalty=0.1),
    )

    record = scorer.score_candidates(
        state,
        [RecombinationChoice(0, 2, 0, 1)],
        device="cpu",
        dtype=torch.float32,
    )[0]

    assert record.breakpoint_scores.tolist() == pytest.approx([-0.1 / 3.0])
    assert torch.isfinite(record.breakpoint_scores).all()
