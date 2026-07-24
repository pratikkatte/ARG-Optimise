from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest
import tskit

from arg.new_rl import (
    CoalescenceChoice,
    LocalRefinementRequest,
    LocalSamplingConfig,
    RecombinationChoice,
    SimpleARGEnvironment,
    advance_local_state,
    apply_local_action,
    build_fast_trace_from_full_arg,
    build_synthetic_full_arg,
    enumerate_local_prior_actions,
    export_refined_tree_sequence,
    initialize_local_arg_state,
    local_is_terminal,
    local_state_to_proposal,
    prepare_local_refinement,
    reveal_due_fixed_ancestors,
    sample_local_prior_action,
    sample_local_trajectories,
    splice_local_proposal,
    undo_local_transition,
    validate_local_splice,
)
from arg.new_rl.synthetic_full_arg import NODE_IS_RE_EVENT


ARG_ROOT = Path(__file__).resolve().parents[2]
SOURCE_25KB = ARG_ROOT / "validation/output/tsinfer/l25kb_dated.trees"
SOURCE_1MB = ARG_ROOT / "validation/output/tsinfer/l1mb_dated.trees"


def _simple_chain_tree_sequence(*, with_mutations=False):
    tables = tskit.TableCollection(sequence_length=10.0)
    samples = [
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
        for _ in range(3)
    ]
    younger_parent = tables.nodes.add_row(time=5.0)
    root = tables.nodes.add_row(time=10.0)
    tables.edges.add_row(0.0, 10.0, parent=younger_parent, child=samples[0])
    tables.edges.add_row(0.0, 10.0, parent=younger_parent, child=samples[1])
    tables.edges.add_row(0.0, 10.0, parent=root, child=younger_parent)
    tables.edges.add_row(0.0, 10.0, parent=root, child=samples[2])
    if with_mutations:
        outside = tables.sites.add_row(position=1.0, ancestral_state="A")
        target = tables.sites.add_row(position=4.0, ancestral_state="A")
        tables.mutations.add_row(
            site=outside,
            node=samples[1],
            derived_state="C",
            time=1.0,
        )
        tables.mutations.add_row(
            site=target,
            node=samples[0],
            derived_state="G",
            time=1.0,
        )
    tables.sort()
    return tables.tree_sequence()


def _recombining_noop_tree_sequence():
    """Left ancestry is complete before a later right-hand event."""

    tables = tskit.TableCollection(sequence_length=10.0)
    samples = [
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
        for _ in range(2)
    ]
    left_root = tables.nodes.add_row(time=5.0)
    right_root = tables.nodes.add_row(time=10.0)
    for sample in samples:
        tables.edges.add_row(0.0, 5.0, parent=left_root, child=sample)
        tables.edges.add_row(5.0, 10.0, parent=right_root, child=sample)
    tables.sort()
    return tables.tree_sequence()


def _environment(
    ts,
    *,
    recombination_rate=0.0,
    rho=None,
    time_delta_bin_width=1e-5,
):
    return SimpleARGEnvironment(
        num_sequences=ts.num_samples,
        sequence_length=int(ts.sequence_length),
        num_blocks=int(ts.sequence_length),
        population_size=10_000.0,
        recombination_rate=recombination_rate,
        rho=rho,
        structural_only=True,
        time_delta_bin_width=time_delta_bin_width,
    )


def _simple_prepared(*, with_mutations=False):
    source = _simple_chain_tree_sequence(with_mutations=with_mutations)
    prepared = prepare_local_refinement(
        source,
        LocalRefinementRequest((2.0, 8.0), cut_event_index=0),
    )
    return source, prepared


def _active_block_counts(state):
    output = np.zeros(len(state.block_boundaries) - 1, dtype=np.int32)
    for lineage in state.active_lineages:
        for left, right in lineage.material_segments.segments:
            output[left:right] += 1
    return output


def _apply_coalescence_for_nodes(state, prepared, env, left_id, right_id):
    options = enumerate_local_prior_actions(state, prepared.context, env)
    action = next(
        action
        for action in options.coal_actions
        if {
            state.active_lineages[action.active_lineage_i].node_id,
            state.active_lineages[action.active_lineage_j].node_id,
        }
        == {left_id, right_id}
    )
    return apply_local_action(
        state,
        replace(action, time_action=0, delta_time=1e-6),
        prepared.context,
        env,
        log_prior=0.0,
    )


def _target_genotypes(ts, region=(2.0, 8.0)):
    return tuple(
        (
            float(variant.site.position),
            tuple(
                None
                if genotype == tskit.MISSING_DATA
                else variant.alleles[int(genotype)]
                for genotype in variant.genotypes
            ),
        )
        for variant in ts.variants(
            left=region[0],
            right=region[1],
            isolated_as_missing=False,
        )
    )


def _outside_signature(ts, region):
    left, right = region
    values = []
    for edge in ts.edges():
        if edge.left < left:
            values.append(
                (
                    float(edge.left),
                    min(float(edge.right), left),
                    int(edge.parent),
                    int(edge.child),
                )
            )
        if edge.right > right:
            values.append(
                (
                    max(float(edge.left), right),
                    float(edge.right),
                    int(edge.parent),
                    int(edge.child),
                )
            )
    return tuple(sorted(value for value in values if value[0] < value[1]))


def test_initialize_uses_argstate_target_blocks_and_fixed_schedule():
    source, prepared = _simple_prepared()
    env = _environment(source)
    state = initialize_local_arg_state(prepared, env)

    assert state.current_time == 0.0
    assert state.time_scale == 20_000.0
    assert state.generated_node_start == prepared.synthetic_arg.num_nodes
    assert {lineage.node_id for lineage in state.active_lineages} == {0, 1, 2}
    assert all(
        lineage.material_segments.segments == ((2, 8),)
        for lineage in state.active_lineages
    )
    assert state.target_material.segments == ((2, 8),)
    assert [record["node_id"] for record in state.fixed_ancestor_schedule] == [
        3,
        4,
    ]
    assert [record["time"] for record in state.fixed_ancestor_schedule] == [
        5.0 / 20_000.0,
        10.0 / 20_000.0,
    ]
    assert not local_is_terminal(state, prepared.context)


def test_local_rates_match_environment_formulas():
    source, prepared = _simple_prepared()
    env = _environment(source, rho=4.0)
    state = initialize_local_arg_state(prepared, env)
    state.fixed_ancestor_schedule = []
    options = enumerate_local_prior_actions(state, prepared.context, env)

    assert options.rates["lambda_coal"] == len(options.coal_actions) == 3
    expected_active_length = (
        sum(choice.material_count for choice in options.recomb_choices)
        / env.num_blocks
    )
    assert options.rates["total_active_material_length"] == expected_active_length
    assert options.rates["lambda_recomb"] == pytest.approx(
        env.rho / 2.0 * expected_active_length
    )


def test_sampled_action_log_probability_matches_filtered_cwr_prior():
    source, prepared = _simple_prepared()
    env = _environment(source, rho=4.0)
    state = initialize_local_arg_state(prepared, env)
    state.fixed_ancestor_schedule = []
    options = enumerate_local_prior_actions(state, prepared.context, env)

    action = None
    for seed in range(100):
        action, observed = sample_local_prior_action(
            state,
            prepared.context,
            env,
            np.random.default_rng(seed),
        )
        if action is not None:
            break
    assert action is not None

    total_rate = (
        options.rates["lambda_coal"] + options.rates["lambda_recomb"]
    )
    event_masses = env.time_env.time_action_probabilities(total_rate)
    expected = math.log(event_masses[action.time_action])
    if isinstance(action, CoalescenceChoice):
        expected += math.log(options.rates["lambda_coal"] / total_rate)
        expected -= math.log(len(options.coal_actions))
    else:
        weights = np.asarray(
            [choice.material_count for choice in options.recomb_choices],
            dtype=float,
        )
        selected = next(
            index
            for index, choice in enumerate(options.recomb_choices)
            if choice.active_lineage_i == action.active_lineage_i
        )
        breakpoint_count = len(
            range(
                options.recomb_choices[selected].span_start + 1,
                options.recomb_choices[selected].span_end + 1,
            )
        )
        expected += math.log(options.rates["lambda_recomb"] / total_rate)
        expected += math.log(weights[selected] / weights.sum())
        expected -= math.log(breakpoint_count)
    assert observed == pytest.approx(expected)


def test_bounded_wait_and_fixed_attachment_win_at_equal_time_and_undo():
    source, prepared = _simple_prepared()
    env = _environment(source, rho=2.0)
    state = initialize_local_arg_state(prepared, env)
    options = enumerate_local_prior_actions(state, prepared.context, env)
    total_rate = options.rates["lambda_coal"] + options.rates["lambda_recomb"]
    next_fixed = state.fixed_ancestor_schedule[0]["time"]
    event_masses, survival = env.time_env.bounded_waiting_distribution(
        total_rate,
        next_fixed - state.current_time,
    )
    assert sum(event_masses) + survival == pytest.approx(1.0)

    action = replace(
        options.coal_actions[0],
        time_action=0,
        delta_time=next_fixed - state.current_time,
    )
    with pytest.raises(ValueError, match="cannot skip"):
        apply_local_action(
            state,
            action,
            prepared.context,
            env,
            log_prior=0.0,
        )

    attached = reveal_due_fixed_ancestors(
        state,
        prepared.context,
        next_fixed,
    )
    assert {lineage.node_id for lineage in attached.active_lineages} == {2, 3}
    ancestor = attached.all_nodes[3]
    assert ancestor.event_type == "fixed_source"
    assert set(ancestor.children) == {0, 1}
    assert attached.all_nodes[0].material_segments.count == 0
    assert attached.all_nodes[1].material_segments.count == 0
    assert attached.transition_records[-1]["event_type"] == "fixed_attachment"
    assert {
        (
            edge["parent_node_id"],
            edge["child_node_id"],
            edge["segments"],
        )
        for edge in attached.transition_records[-1]["edge_segments"]
    } == {
        (3, 0, ((2, 8),)),
        (3, 1, ((2, 8),)),
    }
    assert not any(lineage.node_id == 4 for lineage in attached.active_lineages)

    restored = undo_local_transition(attached, prepared.context)
    assert {lineage.node_id for lineage in restored.active_lineages} == {0, 1, 2}
    assert 3 not in restored.all_nodes
    assert restored.current_time == state.current_time
    assert all(
        lineage.material_segments.segments == ((2, 8),)
        for lineage in restored.active_lineages
    )


def test_structural_coalescence_matches_environment_core():
    source, prepared = _simple_prepared()
    env = _environment(source)
    state = initialize_local_arg_state(prepared, env)
    options = enumerate_local_prior_actions(state, prepared.context, env)
    action = replace(
        options.coal_actions[0],
        time_action=0,
        delta_time=1e-6,
    )

    direct = env.apply_action(state.clone(copy_partials=False), action, 0.25)
    local = apply_local_action(
        state.clone(copy_partials=False),
        action,
        prepared.context,
        env,
        log_prior=0.25,
    )
    assert [
        (lineage.node_id, lineage.material_segments)
        for lineage in direct.active_lineages
    ] == [
        (lineage.node_id, lineage.material_segments)
        for lineage in local.active_lineages
    ]
    assert direct.current_time == local.current_time
    assert direct.accumulated_log_prior == local.accumulated_log_prior


def test_sampled_structural_actions_have_exact_backward_transitions():
    source = _simple_chain_tree_sequence()
    prepared = prepare_local_refinement(
        source,
        LocalRefinementRequest((0.0, 10.0), cut_event_index=0),
    )
    env = _environment(source, rho=2.0)
    initial = initialize_local_arg_state(prepared, env)

    coal = replace(
        enumerate_local_prior_actions(
            initial,
            prepared.context,
            env,
        ).coal_actions[0],
        time_action=0,
        delta_time=1e-6,
    )
    after_coal = apply_local_action(
        initial,
        coal,
        prepared.context,
        env,
        log_prior=-0.25,
    )
    restored_coal = undo_local_transition(after_coal, prepared.context)
    assert [
        (lineage.node_id, lineage.material_segments, tuple(lineage.parents))
        for lineage in restored_coal.active_lineages
    ] == [
        (lineage.node_id, lineage.material_segments, tuple(lineage.parents))
        for lineage in initial.active_lineages
    ]
    assert restored_coal.max_node_idx == initial.max_node_idx
    assert restored_coal.current_time == initial.current_time
    assert (
        restored_coal.accumulated_log_prior
        == initial.accumulated_log_prior
    )

    recomb = replace(
        enumerate_local_prior_actions(
            initial,
            prepared.context,
            env,
        ).recomb_choices[0],
        breakpoint=5,
        time_action=0,
        delta_time=1e-6,
    )
    after_recomb = apply_local_action(
        initial,
        recomb,
        prepared.context,
        env,
        log_prior=-0.5,
    )
    restored_recomb = undo_local_transition(after_recomb, prepared.context)
    assert [
        (lineage.node_id, lineage.material_segments, tuple(lineage.parents))
        for lineage in restored_recomb.active_lineages
    ] == [
        (lineage.node_id, lineage.material_segments, tuple(lineage.parents))
        for lineage in initial.active_lineages
    ]
    assert restored_recomb.max_node_idx == initial.max_node_idx
    assert restored_recomb.current_time == initial.current_time
    assert (
        restored_recomb.accumulated_log_prior
        == initial.accumulated_log_prior
    )


def test_recombination_conserves_blocks_and_terminal_is_one_root_per_block():
    source, prepared = _simple_prepared()
    env = _environment(source, rho=2.0)
    state = initialize_local_arg_state(prepared, env)
    before = _active_block_counts(state)
    options = enumerate_local_prior_actions(state, prepared.context, env)
    action = replace(
        options.recomb_choices[0],
        breakpoint=5,
        time_action=0,
        delta_time=1e-6,
    )
    state = apply_local_action(
        state,
        action,
        prepared.context,
        env,
        log_prior=0.0,
    )
    assert np.array_equal(before, _active_block_counts(state))
    recombination_ids = tuple(
        lineage.node_id
        for lineage in state.active_lineages
        if lineage.event_type == "recomb"
    )
    assert len(recombination_ids) == 2

    state = _apply_coalescence_for_nodes(state, prepared, env, 1, 2)
    coalescent = state.max_node_idx
    state = _apply_coalescence_for_nodes(
        state,
        prepared,
        env,
        recombination_ids[0],
        coalescent,
    )
    state = _apply_coalescence_for_nodes(
        state,
        prepared,
        env,
        recombination_ids[1],
        state.max_node_idx,
    )
    assert local_is_terminal(state, prepared.context)
    proposal = local_state_to_proposal(state, prepared)
    assert proposal.root_intervals == ((2.0, 8.0, state.max_node_idx),)
    assert sum(event.kind == "recombination" for event in proposal.events) == 1


def test_seeded_sampler_is_reproducible_and_seed_can_change_topology():
    source = _simple_chain_tree_sequence()
    prepared = prepare_local_refinement(
        source,
        LocalRefinementRequest((0.0, 10.0), cut_event_index=0),
    )
    env = _environment(source)
    config = LocalSamplingConfig(
        seed=9,
        sample_count=1,
        max_generated_events=8,
        max_searched_states=500,
        max_restarts=8,
    )
    first = sample_local_trajectories(prepared, env, config)
    second = sample_local_trajectories(prepared, env, config)
    assert first.is_complete
    assert first.proposals == second.proposals
    assert first.trajectories[0].actions == second.trajectories[0].actions
    assert math.isfinite(first.proposals[0].prior_log_probability)

    digests = {
        sample_local_trajectories(
            prepared,
            env,
            replace(config, seed=seed),
        ).proposals[0].topology_digest
        for seed in (2, 3, 4)
    }
    assert len(digests) > 1


def test_sampling_search_limit_returns_diagnostic():
    source = _simple_chain_tree_sequence()
    prepared = prepare_local_refinement(
        source,
        LocalRefinementRequest((0.0, 10.0), cut_event_index=0),
    )
    env = _environment(source)
    batch = sample_local_trajectories(
        prepared,
        env,
        LocalSamplingConfig(
            seed=1,
            sample_count=1,
            max_generated_events=1,
            max_restarts=1,
        ),
    )
    assert not batch.proposals
    assert {
        diagnostic.code for diagnostic in batch.diagnostics
    } == {
        "generated_event_watchdog_reached",
        "sampling_incomplete",
    }


def test_default_sampling_config_has_no_completion_watchdogs():
    config = LocalSamplingConfig()
    assert config.max_generated_events is None
    assert config.max_searched_states is None
    assert config.max_restarts is None


def test_sample_splice_keeps_local_events_genotypes_and_fixed_exterior(tmp_path):
    source, prepared = _simple_prepared(with_mutations=True)
    env = _environment(source, rho=2.0)
    state = initialize_local_arg_state(prepared, env)
    recomb = replace(
        enumerate_local_prior_actions(
            state,
            prepared.context,
            env,
        ).recomb_choices[0],
        breakpoint=5,
        time_action=0,
        delta_time=1e-6,
    )
    state = apply_local_action(
        state,
        recomb,
        prepared.context,
        env,
        log_prior=0.0,
    )
    recombination_ids = tuple(
        lineage.node_id
        for lineage in state.active_lineages
        if lineage.event_type == "recomb"
    )
    state = _apply_coalescence_for_nodes(state, prepared, env, 1, 2)
    state = _apply_coalescence_for_nodes(
        state,
        prepared,
        env,
        recombination_ids[0],
        state.max_node_idx,
    )
    state = _apply_coalescence_for_nodes(
        state,
        prepared,
        env,
        recombination_ids[1],
        state.max_node_idx,
    )
    proposal = local_state_to_proposal(state, prepared)
    result = splice_local_proposal(prepared, proposal)
    refined = result.refined_tree_sequence

    assert result.is_valid, result.validation.errors
    assert result.validation.counts["target_genotypes_preserved"]
    assert _target_genotypes(source) == _target_genotypes(refined)
    assert _outside_signature(refined, (2.0, 8.0)) == _outside_signature(
        source,
        (2.0, 8.0),
    )
    assert any(
        refined.node(node_id).flags & NODE_IS_RE_EVENT
        for node_id in result.local_node_id_map.values()
    )
    mapped_root = result.local_node_id_map[proposal.root_intervals[0][2]]
    assert refined.at(4.0).parent(mapped_root) == tskit.NULL

    output = tmp_path / "refined.trees"
    assert export_refined_tree_sequence(result, output) == output
    assert tskit.load(str(output)).tables.equals(refined.tables)
    with pytest.raises(FileExistsError):
        export_refined_tree_sequence(result, output)


def test_splice_validation_rejects_dangling_non_sample_ancestry():
    source, prepared = _simple_prepared()
    env = _environment(source)
    batch = sample_local_trajectories(
        prepared,
        env,
        LocalSamplingConfig(seed=1, sample_count=1),
    )
    proposal = batch.proposals[0]
    result = splice_local_proposal(prepared, proposal)
    assert result.is_valid

    tables = result.refined_tree_sequence.dump_tables()
    tree = result.refined_tree_sequence.at(4.0)
    root = int(tree.root)
    dangling = tables.nodes.add_row(
        time=float(result.refined_tree_sequence.node(root).time) / 2.0,
    )
    tables.edges.add_row(
        2.0,
        8.0,
        parent=root,
        child=dangling,
    )
    tables.sort()
    malformed = tables.tree_sequence()
    report = validate_local_splice(
        prepared,
        proposal,
        malformed,
        result.local_node_id_map,
        result.removed_source_synthetic_node_ids,
    )
    assert not report.is_valid
    assert report.counts["dangling_target_node_count"] == 1
    assert any(
        "no sample descendants" in error for error in report.errors
    )


def test_no_authorized_history_is_terminal_noop_and_round_trips():
    source = _recombining_noop_tree_sequence()
    prepared = prepare_local_refinement(
        source,
        LocalRefinementRequest((0.0, 5.0), cut_time=7.0),
    )
    env = _environment(source)
    state = initialize_local_arg_state(prepared, env)
    assert state.is_done
    assert not prepared.context.authorized_edge_intervals

    proposal = local_state_to_proposal(state, prepared)
    assert not proposal.nodes
    result = splice_local_proposal(prepared, proposal)
    assert result.is_valid
    assert result.refined_tree_sequence.tables.equals(
        source.tables,
        ignore_provenance=True,
    )


@pytest.fixture(scope="module")
def prepared_25kb():
    source = tskit.load(str(SOURCE_25KB))
    prepared = prepare_local_refinement(
        source,
        LocalRefinementRequest((386.0, 23_963.0), cut_time=25_000.0),
    )
    env = _environment(source, time_delta_bin_width=0.001)
    return source, prepared, env


def test_25kb_prior_construction_splice_and_resyntheticize(prepared_25kb):
    source, prepared, env = prepared_25kb
    batch = sample_local_trajectories(
        prepared,
        env,
        LocalSamplingConfig(
            seed=1,
            sample_count=1,
            max_generated_events=16,
            max_searched_states=2_000,
            max_restarts=8,
        ),
    )
    assert batch.is_complete, batch.diagnostics
    proposal = batch.proposals[0]
    result = splice_local_proposal(prepared, proposal)

    assert result.is_valid, result.validation.errors
    assert result.validation.counts["exterior_unchanged"]
    assert result.validation.counts["target_genotypes_preserved"]
    assert result.validation.counts["dangling_target_node_count"] == 0
    assert proposal.root_intervals
    conversion = build_synthetic_full_arg(result.refined_tree_sequence)
    trace = build_fast_trace_from_full_arg(
        conversion.tree_sequence,
        require_unique_event_times=True,
        allow_no_recombination=True,
    )
    assert trace.num_steps > 0


@pytest.mark.slow
def test_1mb_local_construction_and_clean_export_smoke(tmp_path):
    source = tskit.load(str(SOURCE_1MB))
    prepared = prepare_local_refinement(
        source,
        LocalRefinementRequest((14_000.0, 190_000.0), cut_time=25_000.0),
    )
    env = SimpleARGEnvironment(
        num_sequences=source.num_samples,
        sequence_length=int(source.sequence_length),
        num_blocks=int(source.sequence_length),
        population_size=10_000.0,
        recombination_rate=2e-8,
        structural_only=True,
    )
    batch = sample_local_trajectories(
        prepared,
        env,
        LocalSamplingConfig(seed=1, sample_count=1),
    )
    assert batch.is_complete, batch.diagnostics
    assert len(batch.proposals[0].events) > 12
    result = splice_local_proposal(prepared, batch.proposals[0])
    assert result.is_valid, result.validation.errors
    assert result.validation.counts["dangling_target_node_count"] == 0
    output = export_refined_tree_sequence(
        result,
        tmp_path / "l1mb_refined.trees",
    )
    assert tskit.load(str(output)).tables.equals(
        result.refined_tree_sequence.tables
    )
