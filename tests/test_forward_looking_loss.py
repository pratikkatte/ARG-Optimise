import math

import torch

from env import CoalescenceChoice, RecombinationChoice, SimpleARGEnvironment
from rollout_worker_arg import RolloutWorker
from tb_gfn import TBGFlowNetGenerator


def tiny_generator(loss_mode="fl_subtb", sequences=None, recombination_rate=0.0):
    if sequences is None:
        sequences = ["AA", "AA"]
    env = SimpleARGEnvironment(
        sequences=sequences,
        bp_per_blocks=1,
        recombination_rate=recombination_rate,
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


def test_fl_subtb_formula_uses_known_prefix_plus_residual_flow_and_ignores_padding():
    known_prefixes = torch.tensor([0.0, 0.4, 1.0])
    learned_residual_flows = torch.tensor([2.0, 1.0, 0.0])
    log_flows_by_traj = [known_prefixes + learned_residual_flows]
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

    flows = known_prefixes + learned_residual_flows
    residual_01 = flows[0] + 0.3 - flows[1] - (-0.1)
    residual_12 = flows[1] - 0.2 - flows[2] - 0.4
    residual_02 = flows[0] + 0.1 - flows[2] - 0.3
    expected = (
        0.9 * residual_01**2
        + 0.9 * residual_12**2
        + 0.9**2 * residual_02**2
    ) / (0.9 + 0.9 + 0.9**2)

    assert torch.allclose(actual, expected.to(dtype=actual.dtype))


def test_fl_rollout_sets_terminal_partial_prefix_to_exact_reward():
    _, generator = tiny_generator(loss_mode="fl_subtb")
    worker = RolloutWorker(generator.env)

    outputs, _ = worker.rollout(generator, episodes=2)

    for traj_idx, state_path in enumerate(outputs["trajectory_states"]):
        terminal = state_path[-1]
        assert len(state_path) == int(outputs["trajectory_lengths"][traj_idx].item()) + 1
        assert terminal.is_done
        assert math.isclose(
            float(terminal.partial_log_reward),
            float(terminal.log_reward),
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        prefix_increment_sum = sum(
            float(state_path[idx].partial_log_reward)
            - float(state_path[idx - 1].partial_log_reward)
            for idx in range(1, len(state_path))
        )
        assert math.isclose(
            prefix_increment_sum,
            float(terminal.log_reward),
            rel_tol=1e-6,
            abs_tol=1e-6,
        )

    loss = generator.compute_subtb_loss_from_rollout_outputs(outputs)
    assert torch.isfinite(loss)


def test_fl_subtb_capped_rollout_can_end_at_nonterminal_partial_state():
    _, generator = tiny_generator(
        loss_mode="fl_subtb",
        sequences=["AA", "AA", "AA"],
        recombination_rate=0.0,
    )
    worker = RolloutWorker(generator.env)

    outputs, _ = worker.rollout(generator, episodes=1, max_steps=1)

    assert outputs["trajectory_lengths"].tolist() == [1]
    assert outputs["terminal_mask"].tolist() == [False]
    assert outputs["truncated_mask"].tolist() == [True]
    assert not outputs["trajectory_states"][0][-1].is_done
    assert torch.isnan(outputs["log_rewards"][0])

    loss = generator.compute_subtb_loss_from_rollout_outputs(outputs)
    assert torch.isfinite(loss)


def test_coalescence_adds_finite_partial_likelihood_increment():
    env, _ = tiny_generator(
        loss_mode="fl_subtb",
        sequences=["AA", "AA", "AA"],
        recombination_rate=0.0,
    )
    state = env.get_initial_state()

    next_state = env.apply_coalescence(
        state,
        CoalescenceChoice(
            active_lineage_i=0,
            active_lineage_j=1,
            time_quantile=0.5,
        ),
        log_prior=None,
    )

    assert not next_state.is_done
    assert math.isfinite(float(next_state.partial_log_reward))
    assert next_state.terminal_partial_correction == 0.0


def test_recombination_adds_no_partial_likelihood_increment_without_log_prior():
    env, _ = tiny_generator(
        loss_mode="fl_subtb",
        sequences=["AA", "AA", "AA"],
        recombination_rate=1e-8,
    )
    state = env.get_initial_state()

    next_state = env.apply_recombination(
        state,
        RecombinationChoice(
            active_lineage_i=0,
            material_count=2,
            span_start=0,
            span_end=1,
            breakpoint=1,
            time_quantile=0.5,
        ),
        log_prior=None,
    )

    assert not next_state.is_done
    assert next_state.partial_log_reward == state.partial_log_reward
