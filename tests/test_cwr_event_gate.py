import copy
import math
from pathlib import Path

import pytest
import torch

from arg.cwr_event_gate import normalize_local_cwr_event_gate_config
from arg.env import CoalescenceChoice, RecombinationChoice
from arg.models import ARGModel
from arg.train import load_train_config, validate_train_config


ARG_ROOT = Path(__file__).resolve().parents[1]


class _FixedResidualGate:
    local_cwr_event_gate_config = {
        "enabled": True,
        "max_abs_residual": 2.0,
    }

    def __init__(self, residual=0.0):
        self.residual = torch.tensor(float(residual), requires_grad=True)

    def compute_local_cwr_event_residual(self, summary_reps):
        values = self.residual.expand(summary_reps.shape[0])
        return values, values


def _actions():
    return [[
        CoalescenceChoice(0, 1),
        CoalescenceChoice(0, 2),
        RecombinationChoice(0, 2, 0, 1),
        RecombinationChoice(1, 2, 0, 1),
        RecombinationChoice(2, 2, 0, 1),
    ]]


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_zero_residual_gate_reproduces_tempered_cwr_event_mass(temperature):
    gate = _FixedResidualGate(0.0)
    actions = _actions()
    base_logits = torch.tensor(
        [[1.1, -0.4, 0.3, -0.2, 0.8]],
        dtype=torch.float64,
    )
    final, diagnostics = ARGModel.apply_local_cwr_event_gate(
        gate,
        base_logits / temperature,
        actions,
        torch.zeros((1, 3), dtype=torch.float64),
        [{"lambda_coal": 3.0, "lambda_recomb": 1.0}],
        random_spec={"T": temperature},
    )

    probabilities = torch.softmax(final, dim=1)[0]
    expected_event = torch.softmax(
        torch.log(torch.tensor([3.0, 1.0], dtype=torch.float64)) / temperature,
        dim=0,
    )
    assert float(probabilities[:2].sum().detach()) == pytest.approx(
        float(expected_event[0]), abs=1e-12
    )
    assert float(probabilities[2:].sum().detach()) == pytest.approx(
        float(expected_event[1]), abs=1e-12
    )
    assert torch.allclose(
        probabilities[:2] / probabilities[:2].sum(),
        torch.softmax(base_logits[0, :2] / temperature, dim=0),
    )
    assert torch.allclose(
        probabilities[2:] / probabilities[2:].sum(),
        torch.softmax(base_logits[0, 2:] / temperature, dim=0),
    )
    assert diagnostics[0]["local_cwr_event_residual"] == pytest.approx(0.0)


def test_learned_residual_changes_only_event_odds_and_receives_gradient():
    gate = _FixedResidualGate(0.5)
    actions = _actions()
    base_logits = torch.tensor(
        [[1.1, -0.4, 0.3, -0.2, 0.8]],
        dtype=torch.float64,
    )
    final, _ = ARGModel.apply_local_cwr_event_gate(
        gate,
        base_logits,
        actions,
        torch.zeros((1, 3), dtype=torch.float64),
        [{"lambda_coal": 3.0, "lambda_recomb": 1.0}],
    )
    probabilities = torch.softmax(final, dim=1)[0]
    expected_recombination = math.exp(0.5) / (3.0 + math.exp(0.5))
    assert float(probabilities[2:].sum().detach()) == pytest.approx(
        expected_recombination,
        abs=1e-12,
    )

    (-torch.log_softmax(final, dim=1)[0, 2]).backward()
    assert gate.residual.grad is not None
    assert torch.isfinite(gate.residual.grad)
    assert float(gate.residual.grad) != 0.0


def test_gate_handles_one_supported_event_without_phantom_mass():
    gate = _FixedResidualGate(1.0)
    actions = [[CoalescenceChoice(0, 1)]]
    final, diagnostics = ARGModel.apply_local_cwr_event_gate(
        gate,
        torch.tensor([[0.3]]),
        actions,
        torch.zeros((1, 2)),
        [{"lambda_coal": 1.0, "lambda_recomb": 0.0}],
    )
    assert torch.equal(torch.softmax(final, dim=1), torch.ones((1, 1)))
    assert diagnostics[0]["local_cwr_policy_recombination_probability"] == 0.0


@pytest.mark.parametrize(
    "config, message",
    [
        ({"enabled": "yes"}, "enabled"),
        ({"max_abs_residual": 0}, "positive"),
        ({"max_abs_residual": float("inf")}, "finite"),
        ({"unknown": 1}, "unknown fields"),
    ],
)
def test_cwr_event_gate_config_validation(config, message):
    with pytest.raises(ValueError, match=message):
        normalize_local_cwr_event_gate_config(config)


def test_gate_rejects_candidate_rate_support_mismatch():
    gate = _FixedResidualGate()
    with pytest.raises(ValueError, match="recombination candidates"):
        ARGModel.apply_local_cwr_event_gate(
            gate,
            torch.tensor([[0.0]]),
            [[CoalescenceChoice(0, 1)]],
            torch.zeros((1, 2)),
            [{"lambda_coal": 1.0, "lambda_recomb": 1.0}],
        )


def test_gate_only_experiment_config_is_matched_to_baseline():
    baseline = load_train_config(
        ARG_ROOT / "config/config_1mb_local_refinement_flow_consistency.yaml"
    )
    enabled = load_train_config(
        ARG_ROOT
        / "config/config_1mb_local_refinement_flow_consistency_cwr_gate.yaml"
    )
    validate_train_config(baseline)
    validate_train_config(enabled)
    assert not baseline["model"]["local_cwr_event_gate"]["enabled"]
    assert enabled["model"]["local_cwr_event_gate"]["enabled"]

    normalized_enabled = copy.deepcopy(enabled)
    normalized_enabled["output_path"] = baseline["output_path"]
    normalized_enabled["model"]["local_cwr_event_gate"]["enabled"] = False
    assert normalized_enabled == baseline

    combined = load_train_config(
        ARG_ROOT
        / "config/config_1mb_local_refinement_flow_consistency_cwr_gate_split_bias.yaml"
    )
    validate_train_config(combined)
    assert combined["model"]["local_cwr_event_gate"]["enabled"]
    assert combined["model"]["recombination_split_bias"]["enabled"]
    normalized_combined = copy.deepcopy(combined)
    normalized_combined["output_path"] = baseline["output_path"]
    normalized_combined["model"]["local_cwr_event_gate"]["enabled"] = False
    normalized_combined["model"]["recombination_split_bias"]["enabled"] = False
    assert normalized_combined == baseline


def test_enabled_gate_config_rejects_global_training():
    config = load_train_config()
    config["dataset_path"] = str(
        ARG_ROOT / "validation/vcf/sim_l25kb_0.vcf"
    )
    config["output_path"] = "unused"
    config["training"]["epochs"] = 1
    config["refinement"] = {
        "enabled": False,
        "arg_path": None,
        "requests": [],
    }
    config["model"]["local_cwr_event_gate"]["enabled"] = True
    with pytest.raises(ValueError, match="local VCF"):
        validate_train_config(config)
