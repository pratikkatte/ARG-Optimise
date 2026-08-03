import pytest
import torch

from env import SimpleARGEnvironment
from rollout_worker_arg import RolloutWorker
from tb_gfn import CHECKPOINT_FORMAT_VERSION, TBGFlowNetGenerator


def tiny_generator(loss_mode="subtb", sequences=None):
    if sequences is None:
        sequences = ["AA", "AA"]
    env = SimpleARGEnvironment(
        sequences=sequences,
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
        log_z_lr=1e-3,
        loss_mode=loss_mode,
        subtb_lambda=0.9,
        model_kwargs={
            "embedding_size": 8,
            "hidden_size": 16,
            "transformer_depth": 1,
            "transformer_heads": 2,
            "time_hidden_size": 8,
            "time_layers": 1,
            "breakpoint_gap_layers": 1,
        },
    )
    return env, generator


def test_subtb_formula_uses_all_weighted_pairs_and_ignores_padding():
    log_flows_by_traj = [torch.tensor([2.0, 1.0, 0.5])]
    log_paths_pf = torch.tensor([[0.3, -0.2, 100.0]])
    log_paths_pb = torch.tensor([[-0.1, 0.4, 100.0]])
    trajectory_lengths = torch.tensor([2])

    actual = TBGFlowNetGenerator._subtb_loss_from_log_flows(
        log_flows_by_traj,
        log_paths_pf,
        log_paths_pb,
        trajectory_lengths,
        subtb_lambda=0.9,
    )

    residual_01 = 2.0 + 0.3 - 1.0 - (-0.1)
    residual_12 = 1.0 - 0.2 - 0.5 - 0.4
    residual_02 = 2.0 + 0.1 - 0.5 - 0.3
    expected = (
        0.9 * residual_01**2
        + 0.9 * residual_12**2
        + 0.9**2 * residual_02**2
    ) / (0.9 + 0.9 + 0.9**2)

    assert torch.allclose(actual, torch.tensor(expected, dtype=actual.dtype))


def test_subtb_max_span_limits_weighted_pairs():
    log_flows_by_traj = [torch.tensor([2.0, 1.0, 0.5, -0.25])]
    log_paths_pf = torch.tensor([[0.3, -0.2, 0.1]])
    log_paths_pb = torch.tensor([[-0.1, 0.4, -0.3]])
    trajectory_lengths = torch.tensor([3])

    actual = TBGFlowNetGenerator._subtb_loss_from_log_flows(
        log_flows_by_traj,
        log_paths_pf,
        log_paths_pb,
        trajectory_lengths,
        subtb_lambda=0.9,
        subtb_max_span=1,
    )

    residual_01 = 2.0 + 0.3 - 1.0 - (-0.1)
    residual_12 = 1.0 - 0.2 - 0.5 - 0.4
    residual_23 = 0.5 + 0.1 - (-0.25) - (-0.3)
    expected = (
        0.9 * residual_01**2
        + 0.9 * residual_12**2
        + 0.9 * residual_23**2
    ) / (0.9 + 0.9 + 0.9)

    assert torch.allclose(actual, torch.tensor(expected, dtype=actual.dtype))


def test_rollout_outputs_state_paths_and_terminal_reward_flow():
    _, generator = tiny_generator(loss_mode="subtb")
    worker = RolloutWorker(generator.env)

    outputs, _ = worker.rollout(generator, episodes=2)

    assert "trajectory_states" in outputs
    assert "trajectory_lengths" in outputs
    assert outputs["trajectory_lengths"].tolist() == [1, 1]
    assert outputs["log_paths_pf"].shape[1] == 1
    assert outputs["log_paths_pb"].shape[1] == 1

    for traj_idx, state_path in enumerate(outputs["trajectory_states"]):
        assert len(state_path) == int(outputs["trajectory_lengths"][traj_idx].item()) + 1
        assert state_path[-1].is_done
        terminal_flow = generator.compute_log_state_flows([state_path[-1]])
        assert torch.allclose(terminal_flow, outputs["log_rewards"][traj_idx : traj_idx + 1])

    loss = generator.compute_subtb_loss_from_rollout_outputs(outputs)
    assert torch.isfinite(loss)


def test_old_checkpoint_without_flow_head_loads():
    _, source = tiny_generator(loss_mode="tb")
    _, target = tiny_generator(loss_mode="tb")
    old_state = {
        key: value
        for key, value in source.state_dict().items()
        if not key.startswith("arg_model.flow_head.")
    }

    target.load({"generator_state_dict": old_state}, load_optimizer=False)

    assert "arg_model.flow_head.0.weight" in target.state_dict()


def test_checkpoint_contains_versioned_inference_and_training_state(tmp_path):
    _, source = tiny_generator(loss_mode="tb")
    checkpoint_path = tmp_path / "checkpoints" / "best.pt"

    source.save(
        checkpoint_path,
        metadata={"model_version": "test", "checkpoint_kind": "best"},
        training_state={"epoch": 3, "best_metric_value": 1.25},
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION
    assert checkpoint["metadata"]["checkpoint_kind"] == "best"
    assert checkpoint["training_state"] == {
        "epoch": 3,
        "best_metric_value": 1.25,
    }
    assert "generator_state_dict" in checkpoint
    assert "opt_state_dict" in checkpoint
    assert not list(checkpoint_path.parent.glob(".*.tmp"))


def test_tb_rejects_capped_nonterminal_rollout_outputs():
    _, generator = tiny_generator(loss_mode="tb", sequences=["AA", "AA", "AA"])
    worker = RolloutWorker(generator.env)

    outputs, _ = worker.rollout(generator, episodes=1, max_steps=1)

    assert outputs["terminal_mask"].tolist() == [False]
    with pytest.raises(ValueError, match="Trajectory balance requires terminal"):
        generator.compute_tb_loss_from_rollout_outputs(outputs)
