import copy
import math
from dataclasses import replace

import pytest
import torch

from env import SimpleARGEnvironment
from flow_evaluation import fixed_bank_signature, merge_fixed_evaluation_banks
from tb_gfn import TBGFlowNetGenerator
from tiny_exact_flow import TinyExactFlow, train_tiny_exact_flow
from train import DEFAULT_CONFIG, validate_train_config


def _generator(**kwargs):
    env = SimpleARGEnvironment(
        sequences=["AA", "AA"],
        bp_per_blocks=1,
        recombination_rate=0.0,
        mutation_rate=1e-8,
        reward_C=0.0,
        seed=1,
        device="cpu",
    )
    return TBGFlowNetGenerator(
        env,
        init_z_sample_count=0,
        device="cpu",
        verbose=False,
        policy_lr=1e-3,
        time_policy_lr=3e-4,
        initialize_z_from_prior=False,
        loss_mode="subtb",
        model_kwargs={
            "embedding_size": 8,
            "hidden_size": 16,
            "transformer_depth": 1,
            "transformer_heads": 2,
            "time_hidden_size": 8,
            "time_layers": 1,
        },
        **kwargs,
    )


def test_terminal_weighting_and_scaling_are_exact():
    flows = [torch.tensor([2.0, 1.0, 0.5])]
    pf = torch.tensor([[0.3, -0.2]])
    pb = torch.tensor([[-0.1, 0.4]])
    lengths = torch.tensor([2])
    residual_internal = 2.0 + 0.3 - 1.0 - (-0.1)
    residual_terminal_one = 1.0 - 0.2 - 0.5 - 0.4
    residual_terminal_multi = 2.0 + 0.1 - 0.5 - 0.3
    weight_sum = 0.9 + 0.9 + 0.9**2

    loss, details = TBGFlowNetGenerator._subtb_loss_from_log_flows(
        flows,
        pf,
        pb,
        lengths,
        subtb_lambda=0.9,
        terminal_mask=torch.tensor([True]),
        terminal_loss_weight=10.0,
        residual_scale=50.0,
        return_details=True,
    )
    internal = 0.9 * (residual_internal / 50.0) ** 2 / weight_sum
    terminal = (
        0.9 * (residual_terminal_one / 50.0) ** 2
        + 0.9**2 * (residual_terminal_multi / 50.0) ** 2
    ) / weight_sum
    assert details["internal_loss"] == pytest.approx(internal)
    assert details["terminal_loss"] == pytest.approx(terminal)
    assert loss == pytest.approx(internal + 10.0 * terminal)
    assert len([row for row in details["records"] if row["terminal"]]) == 2


def test_default_terminal_weight_and_scale_reproduce_legacy_objective():
    flows = [torch.tensor([2.0, 1.0, 0.5])]
    pf = torch.tensor([[0.3, -0.2]])
    pb = torch.tensor([[-0.1, 0.4]])
    lengths = torch.tensor([2])
    legacy = TBGFlowNetGenerator._subtb_loss_from_log_flows(
        flows, pf, pb, lengths, 0.9
    )
    separated = TBGFlowNetGenerator._subtb_loss_from_log_flows(
        flows,
        pf,
        pb,
        lengths,
        0.9,
        terminal_mask=torch.tensor([True]),
        terminal_loss_weight=1.0,
        residual_scale=1.0,
    )
    assert torch.equal(legacy, separated)


def test_time_head_has_separate_lr_clip_and_curriculum():
    generator = _generator(
        breakpoint_policy_lr=2e-4,
        breakpoint_gradient_clip_norm=0.5,
        time_head_gradient_clip_norm=1.0,
        subtb_lambda_initial=0.6,
        subtb_lambda_final=0.9,
        subtb_max_span_schedule=[
            {"until_epoch": 2, "value": 4},
            {"until_epoch": None, "value": 8},
        ],
    )
    assert [group["lr"] for group in generator.opt.param_groups] == pytest.approx(
        [1e-3, 2e-4, 3e-4]
    )
    parameter_ids = {
        name: {id(parameter) for parameter in parameters}
        for name, parameters in generator.model_parameter_groups.items()
    }
    assert parameter_ids["structural"].isdisjoint(parameter_ids["breakpoint"])
    assert parameter_ids["structural"].isdisjoint(parameter_ids["time"])
    assert parameter_ids["breakpoint"].isdisjoint(parameter_ids["time"])
    assert set().union(*parameter_ids.values()) == {
        id(parameter) for parameter in generator.arg_model.parameters()
    }
    first = generator.set_training_epoch(0, total_epochs=3)
    last = generator.set_training_epoch(2, total_epochs=3)
    assert first["subtb_active_lambda"] == pytest.approx(0.6)
    assert first["subtb_active_max_span"] == 4
    assert last["subtb_active_lambda"] == pytest.approx(0.9)
    assert last["subtb_active_max_span"] == 8


def test_three_model_health_metrics_report_gradients_updates_and_learning_rates():
    generator = _generator(
        breakpoint_policy_lr=2e-4,
        breakpoint_gradient_clip_norm=0.5,
        time_head_gradient_clip_norm=1.0,
    )
    selected_parameters = {
        name: parameters[0]
        for name, parameters in generator.model_parameter_groups.items()
    }
    loss = sum(parameter.reshape(-1)[0] for parameter in selected_parameters.values())
    loss.backward()
    generator.loss = loss.detach()

    info = generator.update_model()

    for name in ("structural", "breakpoint", "time"):
        prefix = f"models/{name}"
        assert info[f"{prefix}/gradient_present"] is True
        assert info[f"{prefix}/gradient_finite_rate"] == pytest.approx(1.0)
        assert info[f"{prefix}/grad_norm_before_clip"] > 0.0
        assert info[f"{prefix}/update_norm"] > 0.0
        assert info[f"{prefix}/relative_update_norm"] > 0.0
    assert info["models/structural/lr_used"] == pytest.approx(1e-3)
    assert info["models/breakpoint/lr_used"] == pytest.approx(2e-4)
    assert info["models/time/lr_used"] == pytest.approx(3e-4)


@pytest.mark.parametrize("terminal_weight", [1.0, 10.0])
@pytest.mark.parametrize("residual_scale", [1.0, 50.0, 100.0])
def test_tiny_exact_solution_is_invariant(terminal_weight, residual_scale):
    model = TinyExactFlow().set_exact_solution()
    assert model.loss(terminal_weight, residual_scale).item() < 1e-12
    assert model.terminal_probabilities().tolist() == pytest.approx([0.25, 0.75])


def test_tiny_exact_environment_trains_to_reward_proportional_distribution():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, history = train_tiny_exact_flow(
        device=device,
        steps=800,
        terminal_loss_weight=10.0,
        residual_scale=50.0,
    )
    residuals = model.trajectory_residuals()
    maximum = max(
        float(values.detach().abs().max().cpu().item())
        for values in residuals.values()
    )
    assert maximum < 1e-3
    assert model.terminal_probabilities().detach().cpu().tolist() == pytest.approx(
        [0.25, 0.75], abs=1e-4
    )
    assert history[-1] < history[0]


class _State:
    def __init__(self, value):
        self.value = value

    def structural_identity(self):
        return (self.value,)


def _bank(source, action_type):
    bank = {
        "version": 1,
        "seed": 7,
        "sources": [source],
        "generation": {},
        "trajectories": [
            {
                "source": source,
                "context_id": "ctx",
                "states": [_State(0), _State(1)],
                "actions": [{"event_type": action_type}],
            }
        ],
    }
    bank["signature"] = fixed_bank_signature(bank)
    return bank


def test_fixed_bank_signature_and_merge_are_deterministic():
    baseline = _bank("baseline", "coal")
    similarity = _bank("similarity_1", "recomb")
    assert fixed_bank_signature(baseline) == baseline["signature"]
    first = merge_fixed_evaluation_banks(baseline, similarity)
    second = merge_fixed_evaluation_banks(baseline, similarity)
    assert first["signature"] == second["signature"]
    assert first["sources"] == ["baseline", "similarity_1"]


def test_production_config_rejects_partial_trajectory_training():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["training"]["epochs"] = 1
    config["training"]["trajectory_training_mode"] = "mixed"
    config["dataset_path"] = "dummy.vcf"
    config["output_path"] = "dummy-output"
    with pytest.raises(ValueError, match="trajectory_training_mode must be 'complete'"):
        validate_train_config(config)


@pytest.mark.parametrize("event_type", ["coal", "recomb"])
def test_global_forward_transition_has_exact_inverse_parent(event_type):
    recombination_rate = 0.0 if event_type == "coal" else 1.0
    env = SimpleARGEnvironment(
        sequences=["AAAA", "AAAA", "AAAA"],
        bp_per_blocks=1,
        recombination_rate=recombination_rate,
        mutation_rate=1e-8,
        reward_C=0.0,
        seed=3,
        device="cpu",
    )
    generator = TBGFlowNetGenerator(
        env,
        init_z_sample_count=0,
        device="cpu",
        verbose=False,
        initialize_z_from_prior=False,
        loss_mode="subtb",
        model_kwargs={
            "embedding_size": 8,
            "hidden_size": 16,
            "transformer_depth": 1,
            "transformer_heads": 2,
            "time_hidden_size": 8,
            "time_layers": 1,
        },
    )
    state = env.get_initial_state()
    options = env.enumerate_prior_options(state)
    choices = (
        options.coal_actions
        if event_type == "coal"
        else options.recomb_choices
    )
    action = choices[0]
    if event_type == "recomb":
        action = replace(action, breakpoint=int(action.span_start) + 1)
    total_rate = env._total_event_rate(options.rates)
    delta_time = env.time_env.quantile_to_delta(0.5, total_rate)
    action = replace(action, time_quantile=0.5, delta_time=delta_time)
    combined = (list(options.coal_actions), list(options.recomb_choices))
    log_prior = env.compute_cwr_event_log_prior(
        state,
        combined,
        action,
        rates=options.rates,
    )
    child = env.apply_action(state, action, log_prior=log_prior)

    inverse_actions = generator._enumerate_inverse_arg_actions(child)
    reconstructed = [
        generator._apply_inverse_arg_action(child, inverse)[0]
        for inverse in inverse_actions
    ]
    assert generator.count_backward_parents(child) == len(inverse_actions) == 1
    assert reconstructed[0].structural_identity() == state.structural_identity()
    assert -math.log(len(inverse_actions)) == pytest.approx(0.0)


def test_masked_atomic_policy_is_normalized_and_invalid_actions_are_zero():
    env = SimpleARGEnvironment(
        sequences=["AAAA", "AAAA", "AAAA"],
        bp_per_blocks=1,
        recombination_rate=0.0,
        mutation_rate=1e-8,
        reward_C=0.0,
        seed=5,
        device="cpu",
    )
    generator = TBGFlowNetGenerator(
        env,
        init_z_sample_count=0,
        device="cpu",
        verbose=False,
        initialize_z_from_prior=False,
        loss_mode="subtb",
        model_kwargs={
            "embedding_size": 8,
            "hidden_size": 16,
            "transformer_depth": 1,
            "transformer_heads": 2,
            "time_hidden_size": 8,
            "time_layers": 1,
        },
    )
    state = env.get_initial_state()
    options = env.enumerate_prior_options(state)
    action = replace(options.coal_actions[0], time_quantile=0.5)
    child = env.apply_action(state, action, log_prior=0.0)
    states = [state, child]
    candidates = [
        list(env.enumerate_actions(current)[0])
        for current in states
    ]
    lineage_reps, summary_reps, _features, _counts = generator._encode_states(states)
    logits, _action_features = generator.arg_model._score_candidates(
        candidates,
        lineage_reps,
        summary_reps,
        state_contexts=states,
    )
    probabilities = torch.softmax(logits, dim=1)
    assert torch.allclose(probabilities.sum(1), torch.ones(2))
    assert torch.equal(
        probabilities[1, len(candidates[1]) :],
        torch.zeros_like(probabilities[1, len(candidates[1]) :]),
    )
