import copy
import os
from pathlib import Path

import pytest
import tskit
import torch

try:
    from env import SimpleARGEnvironment, action_active_lineage_indices
    from refinement import (
        LazyCanonicalStateStore,
        LocalRegionActionFilter,
        build_refinement_context_sets,
        build_refinement_contexts,
        build_refinement_source,
    )
except ImportError as exc:
    pytest.skip(
        "legacy automatic refinement API is not present in the explicit "
        f"local-refinement implementation: {exc}",
        allow_module_level=True,
    )
from rollout_worker_arg import RolloutWorker
from tb_gfn import TBGFlowNetGenerator
from train import (
    DEFAULT_CONFIG,
    terminal_completion_max_steps,
    train_local_refinement,
    validate_train_config,
)
from utils import load_vcf_variants


def write_vcf(tmp_path):
    path = tmp_path / "tiny.vcf"
    path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "##contig=<ID=1,length=100>",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0",
                "1\t10\t.\tA\tC\t.\tPASS\t.\tGT\t0|1",
                "1\t40\t.\tG\tT\t.\tPASS\t.\tGT\t1|0",
                "1\t80\t.\tC\tA\t.\tPASS\t.\tGT\t0|1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def write_tree_sequence(tmp_path):
    tables = tskit.TableCollection(sequence_length=100)
    tables.time_units = "generations"
    sample_flags = tskit.NODE_IS_SAMPLE
    tables.nodes.add_row(flags=sample_flags, time=0.0)
    tables.nodes.add_row(flags=sample_flags, time=0.0)
    parent = tables.nodes.add_row(flags=0, time=10.0)
    tables.edges.add_row(left=0.0, right=100.0, parent=parent, child=0)
    tables.edges.add_row(left=0.0, right=100.0, parent=parent, child=1)
    tables.sort()
    path = tmp_path / "tiny.trees"
    tables.tree_sequence().dump(path)
    return path


def tiny_generator(env):
    return TBGFlowNetGenerator(
        env,
        init_z_sample_count=0,
        device="cpu",
        verbose=False,
        policy_lr=1e-3,
        log_z_lr=1e-3,
        loss_mode="fl_subtb",
        subtb_lambda=0.9,
        initialize_z_from_prior=False,
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


def build_source(tmp_path):
    vcf_path = write_vcf(tmp_path)
    trees_path = write_tree_sequence(tmp_path)
    variant_data = load_vcf_variants(vcf_path)
    env = SimpleARGEnvironment(
        variant_data=variant_data,
        population_size=10000,
        mutation_rate=2e-8,
        recombination_rate=0.0,
        reward_C=0.0,
        seed=1,
        device="cpu",
    )
    source = build_refinement_source(env, trees_path, vcf_path)
    return env, source


def test_refinement_source_replays_tree_sequence_and_builds_partial_context(tmp_path):
    env, source = build_source(tmp_path)

    assert source.canonical_terminal_state.is_done
    assert env.get_active_counts(source.canonical_terminal_state).tolist() == [1, 1, 1]

    contexts, diagnostics = build_refinement_contexts(source, top_k=2)

    assert len(diagnostics) == 3
    assert len(contexts) == 1
    context = contexts[0]
    assert context.region.blocks == (0, 1)
    assert context.partial_state.canonical_step_index < source.canonical_terminal_state.canonical_step_index
    assert all(lineage.partials is not None for lineage in context.partial_state.active_lineages)
    assert set(context.target_blocks).issubset(set(context.effective_blocks))
    visible = context.partial_state.local_visible_lineage_indices
    assert visible
    coal_actions, recomb_actions = env.enumerate_actions(
        context.partial_state,
        action_filter=context.action_filter(),
    )
    candidate_indices = action_active_lineage_indices(
        (coal_actions, recomb_actions),
        active_count=len(context.partial_state.active_lineages),
    )
    assert set(candidate_indices).issubset(set(visible))


def test_terminal_action_filter_prefers_monotone_coalescence(tmp_path):
    env, _source = build_source(tmp_path)
    state = env.get_initial_state()

    regular_filter = LocalRegionActionFilter(blocks=[0, 1, 2])
    terminal_filter = LocalRegionActionFilter(
        blocks=[0, 1, 2],
        terminal_completion=True,
    )
    coal_actions, recomb_actions = env.enumerate_actions(
        state,
        action_filter=regular_filter,
    )
    terminal_coal, terminal_recomb = env.enumerate_actions(
        state,
        action_filter=terminal_filter,
    )

    assert coal_actions
    assert recomb_actions
    assert terminal_coal == coal_actions
    assert terminal_recomb == []


def test_canonical_states_are_lazy_and_materialize_requested_steps(tmp_path):
    env, source = build_source(tmp_path)

    assert isinstance(source.canonical_states, LazyCanonicalStateStore)
    assert not isinstance(source.canonical_states, list)
    assert len(source.canonical_states) == len(source.canonical_action_trace) + 1
    assert source._canonical_terminal_state is None

    initial = source.backtrack_to_step(0)
    middle = source.backtrack_to_step(len(source.canonical_states) // 2)
    terminal = source.canonical_terminal_state

    assert initial.canonical_step_index == 0
    assert middle.canonical_step_index > 0
    assert terminal.is_done
    assert env.get_active_counts(terminal).tolist() == [1, 1, 1]
    assert all(lineage.partials is not None for lineage in middle.active_lineages)

    exported = env.save_to_tree_sequence(terminal)
    assert int(exported.sequence_length) == 100
    assert exported.num_edges > 0


def test_l1mb_vcf_refinement_source_builds_lazy_state_store():
    repo_root = Path(__file__).resolve().parents[1]
    trees_path = repo_root / "validation" / "trees" / "sim_l1mb_0.trees"
    vcf_path = repo_root / "validation" / "vcf" / "sim_l1mb_0.vcf"
    if not trees_path.exists() or not vcf_path.exists():
        pytest.skip("1Mb validation fixtures are not available")

    variant_data = load_vcf_variants(vcf_path)
    env = SimpleARGEnvironment(
        variant_data=variant_data,
        population_size=10000,
        mutation_rate=2e-8,
        recombination_rate=2e-8,
        reward_C=0.0,
        seed=1,
        device="cpu",
    )
    source = build_refinement_source(env, trees_path, vcf_path)

    assert isinstance(source.canonical_states, LazyCanonicalStateStore)
    assert source._canonical_terminal_state is None
    assert len(source.canonical_states) == len(source.canonical_action_trace) + 1
    assert source.num_blocks == variant_data.num_variants


@pytest.mark.skipif(
    os.environ.get("ARG_RUN_L1MB_TERMINAL_SMOKE") != "1",
    reason="set ARG_RUN_L1MB_TERMINAL_SMOKE=1 to run the 1Mb terminal smoke",
)
def test_l1mb_vcf_terminal_materialization_active_counts():
    repo_root = Path(__file__).resolve().parents[1]
    trees_path = repo_root / "validation" / "trees" / "sim_l1mb_0.trees"
    vcf_path = repo_root / "validation" / "vcf" / "sim_l1mb_0.vcf"
    if not trees_path.exists() or not vcf_path.exists():
        pytest.skip("1Mb validation fixtures are not available")

    variant_data = load_vcf_variants(vcf_path)
    env = SimpleARGEnvironment(
        variant_data=variant_data,
        population_size=10000,
        mutation_rate=2e-8,
        recombination_rate=2e-8,
        reward_C=0.0,
        seed=1,
        device="cpu",
    )
    source = build_refinement_source(env, trees_path, vcf_path)
    terminal = source.canonical_terminal_state
    active_counts = env.get_active_counts(terminal)

    assert terminal.is_done
    assert int(active_counts.min()) == 1
    assert int(active_counts.max()) == 1
    assert len(terminal.active_lineages) == variant_data.num_variants


def test_rollout_can_complete_from_refinement_partial_state(tmp_path):
    env, source = build_source(tmp_path)
    contexts, _ = build_refinement_contexts(source, top_k=2)
    context = contexts[0]
    generator = tiny_generator(env)
    worker = RolloutWorker(env)

    with torch.no_grad():
        outputs, trajectories = worker.rollout(
            generator,
            episodes=1,
            start_states=[context.partial_state],
            action_filter=context.action_filter(),
            return_states=True,
        )

    assert outputs["states"][0].is_done
    assert outputs["log_rewards"].shape == (1,)
    assert outputs["trajectory_lengths"][0].item() == len(trajectories[0])


def test_terminal_refinement_contexts_use_offsets_before_strategy_step(tmp_path):
    env, source = build_source(tmp_path)

    segment_contexts, terminal_contexts, diagnostics = build_refinement_context_sets(
        source,
        top_k=2,
        terminal_backtrack_lengths=[1],
    )

    assert len(diagnostics) == 3
    assert len(segment_contexts) == 1
    base = segment_contexts[0]
    by_offset = {context.backtrack_offset: context for context in terminal_contexts}
    assert by_offset[0].backtrack_step == base.backtrack_step
    assert by_offset[0].strategy_backtrack_step == base.backtrack_step
    if base.backtrack_step > 0:
        assert by_offset[1].backtrack_step == base.backtrack_step - 1
    assert all(context.rollout_mode == "terminal" for context in terminal_contexts)

    for context in terminal_contexts:
        assert all(
            lineage.partials is not None
            for lineage in context.partial_state.active_lineages
        )
        coal_actions, recomb_actions = env.enumerate_actions(
            context.partial_state,
            action_filter=context.action_filter(),
        )
        assert coal_actions or recomb_actions
        assert context.partial_state.local_visible_lineage_indices


def test_terminal_completion_guard_caps_contexts_far_from_terminal(tmp_path):
    env, source = build_source(tmp_path)

    _segment_contexts, terminal_contexts, _diagnostics = build_refinement_context_sets(
        source,
        top_k=2,
        terminal_backtrack_lengths=[1],
    )
    context = terminal_contexts[0]

    max_steps, reason = terminal_completion_max_steps(
        context,
        env.num_blocks,
        fallback_max_steps=7,
        max_active_block_excess=999,
        max_effective_blocks=999,
    )
    assert max_steps is None
    assert reason == ""

    max_steps, reason = terminal_completion_max_steps(
        context,
        env.num_blocks,
        fallback_max_steps=7,
        max_active_block_excess=0,
        max_effective_blocks=999,
    )
    assert max_steps == 7
    assert "active_block_excess=" in reason


def test_refinement_region_span_controls_expand_and_cap_blocks(tmp_path):
    _env, source = build_source(tmp_path)

    base = source.select_regions(block_groups=[(1,)])[0]
    expanded = source.select_regions(block_groups=[(1,)], min_bp=80)[0]
    expanded_width = expanded.right_bp - expanded.left_bp

    assert set(base.blocks).issubset(set(expanded.blocks))
    assert expanded_width >= 80 or expanded_width == pytest.approx(
        source.analysis_right - source.analysis_left
    )

    capped = source.select_regions(block_groups=[(0, 1, 2)], max_bp=30)[0]
    capped_width = capped.right_bp - capped.left_bp

    assert len(capped.blocks) < 3
    assert capped_width <= 50 or len(capped.blocks) == 1


def test_train_config_normalizes_new_refinement_rollout_fields():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["dataset_path"] = "dummy.vcf"
    config["output_path"] = "dummy_out"
    config["training"]["epochs"] = 1
    config["training"]["partial_segment_max_steps"] = 3
    config["training"]["subtb_max_span"] = "4"
    config["training"]["subtb_state_flow_batch_size"] = "2"
    config["training"]["rollout_microbatch_size"] = "5"
    config["refinement"]["terminal_backtrack_lengths"] = "1,2,5"
    config["refinement"]["terminal_completion_max_active_block_excess"] = "128"
    config["refinement"]["terminal_completion_max_effective_blocks"] = "256"
    config["refinement"]["bad_region_min_bp"] = "100"
    config["refinement"]["bad_region_max_bp"] = "5000"

    validate_train_config(config)

    assert config["training"]["partial_segment_max_steps"] == 3
    assert config["training"]["subtb_max_span"] == 4
    assert config["training"]["subtb_state_flow_batch_size"] == 2
    assert config["training"]["rollout_microbatch_size"] == 5
    assert config["refinement"]["terminal_backtrack_lengths"] == [1, 2, 5]
    assert config["refinement"]["terminal_completion_max_active_block_excess"] == 128
    assert config["refinement"]["terminal_completion_max_effective_blocks"] == 256
    assert config["refinement"]["bad_region_min_bp"] == 100.0
    assert config["refinement"]["bad_region_max_bp"] == 5000.0


def test_train_config_rejects_removed_fixed_time_bins():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["dataset_path"] = "dummy.vcf"
    config["output_path"] = "dummy_out"
    config["training"]["epochs"] = 1
    config["environment"]["time_delta_bin_width"] = 0.001

    with pytest.raises(ValueError, match="continuous-time v2"):
        validate_train_config(config)


def test_train_config_uses_continuous_time_basis_capacity():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["dataset_path"] = "dummy.vcf"
    config["output_path"] = "dummy_out"
    config["training"]["epochs"] = 1
    config["model"]["time_basis_components"] = 12

    validate_train_config(config)

    assert config["model"]["time_basis_components"] == 12


@pytest.mark.parametrize(
    "partial_segment_max_steps,terminal_backtrack_lengths,match",
    [
        (0, [1], "partial_segment_max_steps"),
        (1, [1], "subtb_max_span"),
        (1, [2], "subtb_state_flow_batch_size"),
        (1, [3], "rollout_microbatch_size"),
        (1, "1,0", "terminal_backtrack_lengths"),
    ],
)
def test_train_config_rejects_invalid_refinement_rollout_fields(
    partial_segment_max_steps,
    terminal_backtrack_lengths,
    match,
):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["dataset_path"] = "dummy.vcf"
    config["output_path"] = "dummy_out"
    config["training"]["epochs"] = 1
    config["training"]["partial_segment_max_steps"] = partial_segment_max_steps
    if match == "subtb_max_span":
        config["training"]["subtb_max_span"] = 0
    if match == "subtb_state_flow_batch_size":
        config["training"]["subtb_state_flow_batch_size"] = 0
    if match == "rollout_microbatch_size":
        config["training"]["rollout_microbatch_size"] = 0
    config["refinement"]["terminal_backtrack_lengths"] = terminal_backtrack_lengths

    with pytest.raises(ValueError, match=match):
        validate_train_config(config)


@pytest.mark.parametrize(
    "field,match",
    [
        (
            "terminal_completion_max_active_block_excess",
            "terminal_completion_max_active_block_excess",
        ),
        (
            "terminal_completion_max_effective_blocks",
            "terminal_completion_max_effective_blocks",
        ),
    ],
)
def test_train_config_rejects_invalid_terminal_completion_guards(field, match):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["dataset_path"] = "dummy.vcf"
    config["output_path"] = "dummy_out"
    config["training"]["epochs"] = 1
    config["refinement"][field] = 0

    with pytest.raises(ValueError, match=match):
        validate_train_config(config)


@pytest.mark.parametrize(
    "min_bp,max_bp,match",
    [
        (0, None, "bad_region_min_bp"),
        (None, 0, "bad_region_max_bp"),
        (100, 50, "bad_region_max_bp"),
    ],
)
def test_train_config_rejects_invalid_bad_region_span_controls(
    min_bp,
    max_bp,
    match,
):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["dataset_path"] = "dummy.vcf"
    config["output_path"] = "dummy_out"
    config["training"]["epochs"] = 1
    config["refinement"]["bad_region_min_bp"] = min_bp
    config["refinement"]["bad_region_max_bp"] = max_bp

    with pytest.raises(ValueError, match=match):
        validate_train_config(config)


def test_local_refinement_training_smoke_alternates_segment_and_terminal(tmp_path, capsys):
    vcf_path = write_vcf(tmp_path)
    trees_path = write_tree_sequence(tmp_path)

    history = train_local_refinement(
        dataset_path=str(vcf_path),
        output_path=str(tmp_path / "run"),
        device="cpu",
        local_refinement_arg=str(trees_path),
        checkpoint=None,
        bad_region_top_k=2,
        terminal_backtrack_lengths=[1],
        bp_per_blocks=1,
        batch_size=2,
        epochs_num=1,
        seed=1,
        init_z_sample_count=0,
        use_wandb=False,
        effective_population_size=10000,
        mutation_rate=2e-8,
        recombination_rate=0.0,
        policy_lr=1e-3,
        log_z_lr=1e-3,
        subtb_max_span=1,
        rollout_microbatch_size=1,
        grad_accum_steps=2,
        eval_episodes=0,
        eval_every=1,
        partial_segment_max_steps=1,
        reward_C=0.0,
        embedding_size=8,
        hidden_size=16,
        transformer_depth=1,
        transformer_heads=2,
        breakpoint_hidden_dim=32,
        breakpoint_dropout=0.0,
        dropout=0.0,
        attention_dropout=0.0,
        verbose=False,
    )

    assert len(history) == 1
    info = history[0]
    assert info["train_segment_batches"] == 2
    assert info["train_terminal_batches"] == 2
    assert "train_segment_truncated_rate" in info
    assert "train_terminal_terminal_rate" in info

    output = capsys.readouterr().out
    assert "Detected local refinement bad regions:" in output
    assert "Local refinement rollout mix:" in output
    assert "training partial->partial" in output
    assert "training partial->terminal" in output
    assert "microbatch=1/2" in output
    assert "region=" in output
