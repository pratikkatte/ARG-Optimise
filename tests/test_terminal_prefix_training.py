import copy
from pathlib import Path

import pytest

from refinement.training import (
    TerminalPrefixSampler,
    _merge_rollout_metrics,
    train_local_refinement,
)
from train import DEFAULT_CONFIG, config_to_train_kwargs, validate_train_config


def _paths():
    states = [[f"s{index}" for index in range(7)]]
    actions = [[
        {"event_type": "coal"},
        {"event_type": "recomb"},
        {"event_type": "fixed_attachment"},
        {"event_type": "coal"},
        {"event_type": "coal"},
        {"event_type": "fixed_attachment"},
    ]]
    return states, actions


def test_terminal_prefix_sampler_boundary_segments_include_attachment():
    states, actions = _paths()
    sampled = TerminalPrefixSampler(17).sample(
        states,
        actions,
        count=4,
        max_steps=3,
        boundary_fraction=0.5,
    )

    assert sum(sampled["boundary_targeted"]) == 2
    for trajectory_index, start_step, targeted in zip(
        sampled["source_trajectory_indices"],
        sampled["start_steps"],
        sampled["boundary_targeted"],
    ):
        assert sampled["start_states"].pop(0) == states[trajectory_index][start_step]
        if targeted:
            segment = actions[trajectory_index][start_step : start_step + 3]
            assert any(row["event_type"] == "fixed_attachment" for row in segment)


def test_terminal_prefix_sampler_is_seeded_and_falls_back_to_uniform():
    states = [["s0", "s1", "s2"]]
    actions = [[{"event_type": "coal"}, {"event_type": "recomb"}]]
    first = TerminalPrefixSampler(4).sample(states, actions, 4, 2, 1.0)
    second = TerminalPrefixSampler(4).sample(states, actions, 4, 2, 1.0)
    assert first == second
    assert not any(first["boundary_targeted"])


def test_prefix_training_config_is_normalized_and_forwarded():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["dataset_path"] = "dummy.vcf"
    config["output_path"] = "dummy-output"
    config["training"]["epochs"] = 1
    config["training"]["grad_accum_steps"] = 2
    config["training"]["partial_start_mode"] = "terminal_prefix_mixture"
    config["training"]["partial_boundary_fraction"] = "0.5"

    validate_train_config(config)
    kwargs = config_to_train_kwargs(config)
    assert kwargs["partial_start_mode"] == "terminal_prefix_mixture"
    assert kwargs["partial_boundary_fraction"] == pytest.approx(0.5)


@pytest.mark.parametrize("value", [-0.1, 1.1, "bad"])
def test_prefix_training_config_rejects_invalid_boundary_fraction(value):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["dataset_path"] = "dummy.vcf"
    config["output_path"] = "dummy-output"
    config["training"]["epochs"] = 1
    config["training"]["partial_boundary_fraction"] = value
    with pytest.raises(ValueError, match="partial_boundary_fraction"):
        validate_train_config(config)


def test_rollout_metric_merge_reports_per_trajectory_counts():
    row = {
        "mode": "partial",
        "length_mean": 16.0,
        "terminal_rate": 0.0,
        "truncated_rate": 1.0,
        "active_variant_rows_mean": 1.0,
        "active_variant_rows_max": 1,
        "time_count": 1,
        "time_quantile_sum": 0.5,
        "time_delta_sum": 0.1,
        "time_near_boundary_sum": 0,
        "time_finite_density_sum": 1,
        "time_entropy_sum": 1.0,
        "time_entropy_count": 1,
        "time_effective_components_sum": 2.0,
        "trajectory_count": 4,
        "fixed_attachment_count": 12,
        "coalescence_count": 20,
        "recombination_count": 8,
        "start_step_sum": 40.0,
        "start_count": 4,
        "boundary_targeted_sum": 2,
        "policy_row_count": 4,
        "generated_policy_row_count": 4,
        "coalescence_probability_mass_sum": 3.0,
        "recombination_probability_mass_sum": 1.0,
        "valid_coalescence_actions_sum": 12,
        "valid_recombination_actions_sum": 4,
        "selected_gate_probability_sum": 3.2,
        "selected_atomic_action_probability_sum": 2.0,
        "structural_action_support_size_sum": 16,
        "structural_action_entropy_sum": 4.0,
        "structural_action_normalized_entropy_sum": 3.0,
        "structural_action_max_probability_sum": 2.4,
        "breakpoint_decision_count": 2,
        "breakpoint_support_size_sum": 10,
        "breakpoint_entropy_sum": 2.0,
        "breakpoint_normalized_entropy_sum": 1.4,
        "breakpoint_selected_probability_sum": 0.6,
        "breakpoint_max_probability_sum": 0.8,
    }
    merged = _merge_rollout_metrics([row])
    assert merged["train_partial_fixed_attachment_mean"] == pytest.approx(3.0)
    assert merged["train_partial_coalescence_mean"] == pytest.approx(5.0)
    assert merged["train_partial_recombination_mean"] == pytest.approx(2.0)
    assert merged["train_partial_start_step_mean"] == pytest.approx(10.0)
    assert merged["train_partial_boundary_targeted_rate"] == pytest.approx(0.5)
    assert merged[
        "models/structural/behavior/partial/normalized_entropy_mean"
    ] == pytest.approx(0.75)
    assert merged[
        "models/breakpoint/behavior/partial/decision_count"
    ] == 2
    assert merged[
        "models/breakpoint/behavior/partial/selected_probability_mean"
    ] == pytest.approx(0.3)
    assert merged[
        "models/time/behavior/partial/effective_components_mean"
    ] == pytest.approx(2.0)


def test_production_training_uses_only_complete_terminal_trajectories(tmp_path):
    root = Path(__file__).resolve().parents[1]
    history = train_local_refinement(
        dataset_path=str(root / "validation/vcf/sim_l25kb_0.vcf"),
        output_path=str(tmp_path / "prefix-training"),
        device="cpu",
        local_refinement_arg=str(
            root / "validation/output/tsinfer/l25kb_dated.trees"
        ),
        requests=[{
            "id": "region_000001",
            "genomic_range": [386, 23963],
            "cut_time": 25_000,
        }],
        batch_size=1,
        epochs_num=1,
        seed=23,
        use_wandb=False,
        recombination_rate=0.0,
        eval_episodes=0,
        grad_accum_steps=2,
        partial_segment_max_steps=2,
        partial_start_mode="terminal_prefix_mixture",
        partial_boundary_fraction=0.5,
        terminal_requires_exhausted_fixed_schedule=True,
        embedding_size=16,
        hidden_size=32,
        breakpoint_hidden_dim=16,
        transformer_depth=1,
        transformer_heads=1,
        time_context_mode="full",
        time_policy_lr=0.01,
        verbose=False,
    )

    assert len(history) == 1
    assert history[0]["train_terminal_trajectory_length_mean"] > 0
    assert history[0]["train_terminal_terminal_rate"] == pytest.approx(1.0)
    assert not any(key.startswith("train_partial_") for key in history[0])
