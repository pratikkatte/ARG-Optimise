import copy
from argparse import Namespace

import pytest
import torch

from env import SimpleARGEnvironment
from lr_control import LearningRateController
from tb_gfn import TBGFlowNetGenerator
from train import (
    DEFAULT_CONFIG,
    apply_cli_overrides,
    config_to_train_kwargs,
    validate_train_config,
)


def _optimizer(lrs):
    parameters = [torch.nn.Parameter(torch.tensor(float(index + 1))) for index in range(len(lrs))]
    return torch.optim.Adam(
        [
            {"params": [parameter], "lr": lr}
            for parameter, lr in zip(parameters, lrs)
        ]
    )


def test_cosine_schedule_warms_up_and_reaches_minimum_on_last_update():
    optimizer = _optimizer([1e-3, 3e-4])
    controller = LearningRateController(
        optimizer,
        group_names=("structural_policy", "time_policy"),
        total_training_steps=6,
        config={
            "type": "cosine",
            "warmup_steps": 2,
            "warmup_start_ratio": 0.1,
            "min_lr_ratio": 0.2,
        },
    )

    factors = [controller.lr_factor]
    used_lrs = [[group["lr"] for group in optimizer.param_groups]]
    for _ in range(6):
        controller.step_update()
        factors.append(controller.lr_factor)
        used_lrs.append([group["lr"] for group in optimizer.param_groups])

    assert factors == pytest.approx([0.1, 0.55, 1.0, 0.8, 0.4, 0.2, 0.2])
    assert used_lrs[0] == pytest.approx([1e-4, 3e-5])
    assert used_lrs[-2] == pytest.approx([2e-4, 6e-5])
    assert controller.metrics()["lr/time_policy"] == pytest.approx(6e-5)


def test_step_schedule_decays_by_optimizer_update_and_respects_floor():
    optimizer = _optimizer([1.0])
    controller = LearningRateController(
        optimizer,
        group_names=("policy",),
        total_training_steps=8,
        config={
            "type": "step",
            "warmup_steps": 0,
            "step_size": 2,
            "step_gamma": 0.5,
            "min_lr_ratio": 0.2,
        },
    )
    factors = [controller.lr_factor]
    for _ in range(7):
        controller.step_update()
        factors.append(controller.lr_factor)
    assert factors == pytest.approx([1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.2, 0.2])


def test_plateau_prefers_fixed_bank_metric_and_reduces_after_patience():
    optimizer = _optimizer([1e-3])
    controller = LearningRateController(
        optimizer,
        group_names=("policy",),
        total_training_steps=20,
        config={
            "type": "plateau",
            "warmup_steps": 0,
            "plateau_metric": "auto",
            "plateau_patience": 1,
            "plateau_factor": 0.5,
            "plateau_threshold": 0.0,
            "min_lr_ratio": 0.1,
        },
    )

    missing = controller.step_metric({"loss": 1.0})
    assert missing["lr/plateau_metric_available"] is False
    first = controller.step_metric(
        {
            "flow_eval/fixed_bank_subtb_mse": 100.0,
            "eval_local_loss_mean": 1.0,
        }
    )
    assert first["lr/plateau_metric_name"] == "flow_eval/fixed_bank_subtb_mse"
    controller.step_metric({"flow_eval/fixed_bank_subtb_mse": 101.0})
    reduced = controller.step_metric({"flow_eval/fixed_bank_subtb_mse": 102.0})
    assert reduced["lr/plateau_reductions"] == 1
    assert reduced["lr/policy"] == pytest.approx(5e-4)


def test_scheduler_state_round_trip_restores_lr_and_progress():
    first_optimizer = _optimizer([1e-3, 2e-3])
    first = LearningRateController(
        first_optimizer,
        group_names=("policy", "log_z"),
        total_training_steps=10,
        config={"type": "cosine", "warmup_steps": 2},
    )
    for _ in range(4):
        first.step_update()

    second_optimizer = _optimizer([1e-3, 2e-3])
    second = LearningRateController(
        second_optimizer,
        group_names=("policy", "log_z"),
        total_training_steps=10,
        config={"type": "cosine", "warmup_steps": 2},
    )
    second.load_state_dict(first.state_dict())
    assert second.optimizer_steps == 4
    assert second.lr_factor == pytest.approx(first.lr_factor)
    assert [group["lr"] for group in second_optimizer.param_groups] == pytest.approx(
        [group["lr"] for group in first_optimizer.param_groups]
    )


def test_generator_checkpoint_contains_and_restores_scheduler_state(tmp_path):
    env = SimpleARGEnvironment(
        sequences=["AA", "AA"],
        bp_per_blocks=1,
        recombination_rate=0.0,
        mutation_rate=1e-8,
        reward_C=0.0,
        seed=1,
        device="cpu",
    )
    kwargs = {
        "init_z_sample_count": 0,
        "device": "cpu",
        "verbose": False,
        "initialize_z_from_prior": False,
        "loss_mode": "tb",
        "model_kwargs": {
            "embedding_size": 8,
            "hidden_size": 16,
            "transformer_depth": 1,
            "transformer_heads": 2,
            "time_hidden_size": 8,
            "time_layers": 1,
        },
        "lr_scheduler_config": {"type": "cosine", "warmup_steps": 2},
        "total_training_steps": 10,
    }
    first = TBGFlowNetGenerator(env, **kwargs)
    first.scheduler.step_update()
    first.optimizer_steps = first.scheduler.optimizer_steps
    path = tmp_path / "scheduled.pt"
    first.save(path)

    second = TBGFlowNetGenerator(env, **kwargs)
    second.load(path, load_optimizer=True, map_location="cpu")
    assert second.optimizer_steps == 1
    assert second.scheduler.optimizer_steps == 1
    assert second.learning_rate_metrics()["lr/factor"] == pytest.approx(
        first.learning_rate_metrics()["lr/factor"]
    )


def test_generator_update_uses_warmup_lr_and_logs_next_lr():
    env = SimpleARGEnvironment(
        sequences=["AA", "AA"],
        bp_per_blocks=1,
        recombination_rate=0.0,
        mutation_rate=1e-8,
        reward_C=0.0,
        seed=1,
        device="cpu",
    )
    generator = TBGFlowNetGenerator(
        env,
        init_z_sample_count=0,
        device="cpu",
        verbose=False,
        policy_lr=1e-3,
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
        lr_scheduler_config={
            "type": "cosine",
            "warmup_steps": 2,
            "warmup_start_ratio": 0.1,
        },
        total_training_steps=6,
    )
    parameter = generator.non_time_policy_params[0]
    before = parameter.detach().clone()
    loss = parameter.sum()
    loss.backward()
    generator.loss = loss.detach()
    info = generator.update_model()

    assert not torch.equal(parameter.detach(), before)
    assert info["lr/used/structural_policy"] == pytest.approx(1e-4)
    assert info["lr/structural_policy"] == pytest.approx(5.5e-4)
    assert info["lr/used/breakpoint_policy"] == pytest.approx(1e-4)
    assert info["lr/breakpoint_policy"] == pytest.approx(5.5e-4)
    assert info["lr/optimizer_step"] == 1


def test_default_yaml_training_enables_warmup_cosine_and_forwards_config():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["dataset_path"] = "dummy.vcf"
    config["output_path"] = "dummy-output"
    config["training"]["epochs"] = 100
    validate_train_config(config)
    kwargs = config_to_train_kwargs(config)
    assert kwargs["lr_scheduler_config"]["type"] == "cosine"
    assert kwargs["lr_scheduler_config"]["warmup_fraction"] == pytest.approx(0.05)


def test_breakpoint_optimizer_and_diagnostic_config_is_normalized_and_forwarded():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["dataset_path"] = "dummy.vcf"
    config["output_path"] = "dummy-output"
    config["training"].update(
        {
            "epochs": 10,
            "breakpoint_policy_lr": "0.0002",
            "breakpoint_gradient_clip_norm": "0.75",
            "model_diagnostics": True,
            "model_diagnostics_update_norm_every": 3,
        }
    )

    validate_train_config(config)
    kwargs = config_to_train_kwargs(config)

    assert kwargs["breakpoint_policy_lr"] == pytest.approx(2e-4)
    assert kwargs["breakpoint_gradient_clip_norm"] == pytest.approx(0.75)
    assert kwargs["model_diagnostics"] is True
    assert kwargs["model_diagnostics_update_norm_every"] == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("breakpoint_policy_lr", float("nan")),
        ("breakpoint_gradient_clip_norm", float("inf")),
        ("model_diagnostics_update_norm_every", 0),
    ],
)
def test_training_config_rejects_invalid_model_diagnostic_values(field, value):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["dataset_path"] = "dummy.vcf"
    config["output_path"] = "dummy-output"
    config["training"]["epochs"] = 10
    config["training"][field] = value
    with pytest.raises(ValueError, match=field):
        validate_train_config(config)


def test_lr_scheduler_cli_overrides_update_nested_config():
    config = copy.deepcopy(DEFAULT_CONFIG)
    updated = apply_cli_overrides(
        config,
        Namespace(
            lr_scheduler_type="step",
            lr_warmup_steps=7,
            lr_step_size=25,
            lr_step_gamma=0.25,
        ),
    )
    assert updated["training"]["lr_scheduler"]["type"] == "step"
    assert updated["training"]["lr_scheduler"]["warmup_steps"] == 7
    assert updated["training"]["lr_scheduler"]["step_size"] == 25
    assert updated["training"]["lr_scheduler"]["step_gamma"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "override,match",
    [
        ({"type": "unknown"}, "lr_scheduler.type"),
        ({"warmup_fraction": 1.0}, "warmup_fraction"),
        ({"warmup_steps": -1}, "warmup_steps"),
        ({"plateau_factor": 1.0}, "plateau_factor"),
        ({"step_gamma": 0.0}, "step_gamma"),
    ],
)
def test_training_config_rejects_invalid_lr_scheduler_values(override, match):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["dataset_path"] = "dummy.vcf"
    config["output_path"] = "dummy-output"
    config["training"]["epochs"] = 10
    config["training"]["lr_scheduler"].update(override)
    with pytest.raises(ValueError, match=match):
        validate_train_config(config)
