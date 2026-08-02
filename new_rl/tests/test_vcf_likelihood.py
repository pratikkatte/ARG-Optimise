import math
from pathlib import Path

import numpy as np
import pytest
import torch
import tskit

from arg.env import CoalescenceChoice, RecombinationChoice, SimpleARGEnvironment
from arg.models import ARGModel
from arg.new_rl import (
    LocalRefinementRequest,
    LocalSamplingConfig,
    RegionLocalVCFView,
    apply_local_action,
    compute_cut_frontier_vcf_partials,
    compute_tree_sequence_vcf_log_likelihood,
    enumerate_local_prior_actions,
    export_refined_tree_sequence,
    initialize_local_arg_state,
    local_state_to_proposal,
    prepare_local_refinement,
    resolve_vcf_tree_sequence_alignment,
    reveal_due_fixed_ancestors,
    sample_local_trajectories,
    splice_local_proposal,
    undo_local_transition,
)
from arg.utils import load_vcf_variants


ARG_ROOT = Path(__file__).resolve().parents[2]


def _write_vcf(tmp_path, genotype="0|1", *, position=2):
    path = tmp_path / "tiny.vcf"
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1,length=10>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tdiploid\n"
        f"1\t{position}\t.\tA\tC\t.\tPASS\t.\tGT\t{genotype}\n"
        f"1\t7\t.\tA\tC\t.\tPASS\t.\tGT\t{genotype}\n"
    )
    return path


def _two_sample_tree(*, duplicate_alignment_site=False):
    tables = tskit.TableCollection(sequence_length=10.0)
    sample_a = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
    sample_c = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
    root = tables.nodes.add_row(time=1_000.0)
    tables.edges.add_row(0.0, 10.0, root, sample_a)
    tables.edges.add_row(0.0, 10.0, root, sample_c)
    positions = (
        [1.0, 2.0, 6.0, 7.0]
        if duplicate_alignment_site
        else [2.0, 7.0]
    )
    for position in positions:
        site = tables.sites.add_row(position=position, ancestral_state="A")
        tables.mutations.add_row(
            site=site,
            node=sample_c,
            derived_state="C",
        )
    tables.sort()
    return tables.tree_sequence()


def _environment(variant_data, *, reward_c=0.0):
    return SimpleARGEnvironment(
        variant_data=variant_data,
        population_size=10_000.0,
        mutation_rate=2e-8,
        recombination_rate=2e-8,
        reward_C=reward_c,
        seed=1,
    )


def _prepared(ts, region=(1.5, 8.5)):
    return prepare_local_refinement(
        ts,
        LocalRefinementRequest(region, cut_event_index=0),
    )


def _jc69_two_different_tip_probability(branch_length):
    decay = math.exp(-4.0 * branch_length / 3.0)
    same = 0.25 + 0.75 * decay
    different = 0.25 - 0.25 * decay
    return 0.25 * (
        2.0 * same * different
        + 2.0 * different * different
    )


def test_alignment_detects_raw_pos_and_explicit_sample_mapping(tmp_path):
    ts = _two_sample_tree()
    data = load_vcf_variants(_write_vcf(tmp_path))
    alignment = resolve_vcf_tree_sequence_alignment(ts, data)
    assert alignment["vcf_coordinate_offset"] == 1.0
    assert alignment["haplotype_index_by_sample_node"] == {0: 0, 1: 1}

    reversed_data = load_vcf_variants(
        _write_vcf(tmp_path, genotype="1|0")
    )
    with pytest.raises(ValueError, match="not genotype-concordant"):
        resolve_vcf_tree_sequence_alignment(ts, reversed_data)
    explicit = resolve_vcf_tree_sequence_alignment(
        ts,
        reversed_data,
        sample_node_to_haplotype={0: 1, 1: 0},
    )
    assert explicit["haplotype_index_by_sample_node"] == {0: 1, 1: 0}


def test_alignment_rejects_ambiguous_coordinate_convention(tmp_path):
    ts = _two_sample_tree(duplicate_alignment_site=True)
    data = load_vcf_variants(_write_vcf(tmp_path))
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_vcf_tree_sequence_alignment(ts, data)
    alignment = resolve_vcf_tree_sequence_alignment(
        ts,
        data,
        vcf_coordinate_offset=1,
    )
    assert alignment["vcf_coordinate_offset"] == 1.0


def test_iterative_jc69_likelihood_matches_hand_calculation(tmp_path):
    ts = _two_sample_tree()
    data = load_vcf_variants(_write_vcf(tmp_path))
    observed = compute_tree_sequence_vcf_log_likelihood(
        ts,
        data,
        mutation_rate=2e-8,
    )
    per_site = _jc69_two_different_tip_probability(1_000.0 * 2e-8)
    assert observed == pytest.approx(2.0 * math.log(per_site), abs=1e-12)


def test_region_local_vcf_view_partitions_target_and_exterior(tmp_path):
    ts = _two_sample_tree()
    data = load_vcf_variants(_write_vcf(tmp_path))
    alignment = resolve_vcf_tree_sequence_alignment(ts, data)
    endpoint_intervals = {
        0: ((1.5, 8.5),),
        1: ((1.5, 8.5),),
    }

    view = compute_cut_frontier_vcf_partials(
        ts,
        data,
        endpoint_intervals,
        (1.5, 4.0),
        mutation_rate=2e-8,
        alignment=alignment,
    )

    assert isinstance(view, RegionLocalVCFView)
    assert view.target_variant_indices == (0,)
    assert view.outside_variant_indices == (1,)
    assert view.target_variant_count == 1
    assert view.outside_variant_count == 1
    assert view.endpoint_variant_row_count == 2
    assert view.outside_log_likelihood == pytest.approx(
        compute_tree_sequence_vcf_log_likelihood(
            ts,
            data,
            mutation_rate=2e-8,
            alignment=alignment,
            variant_indices=(1,),
        )
    )
    summary = view.to_summary_dict()
    assert summary["likelihood_scope"] == "whole_vcf_chromosome"
    assert summary["cached_exterior_likelihood"] is True
    assert summary["target_variant_count"] == 1
    assert summary["outside_variant_count"] == 1


def test_local_grid_and_cut_partials_separate_structure_from_vcf_rows(
    tmp_path,
):
    ts = _two_sample_tree()
    data = load_vcf_variants(_write_vcf(tmp_path))
    env = _environment(data)
    state = initialize_local_arg_state(_prepared(ts), env)

    assert state.block_boundaries == (1.5, 4.5, 8.5)
    assert state.target_material.segments == ((0, 2),)
    assert state.target_variant_indices == (0, 1)
    assert state.variant_block_indices == {0: 0, 1: 1}
    assert state.local_breakpoint_weights == {1: 5.0}
    view_record = state.transition_records[0]["region_vcf_view"]
    assert view_record["global_variant_count"] == 2
    assert view_record["target_variant_count"] == 2
    assert view_record["outside_variant_count"] == 0
    assert view_record["endpoint_variant_row_count"] == 4
    assert all(
        lineage.material_segments.count == 2
        and lineage.variant_indices == (0, 1)
        and tuple(lineage.partials.shape) == (2, 4)
        for lineage in state.active_lineages
    )


def test_local_coalescence_likelihood_matches_clean_splice(tmp_path):
    ts = _two_sample_tree()
    data = load_vcf_variants(_write_vcf(tmp_path))
    env = _environment(data)
    prepared = _prepared(ts)
    state = initialize_local_arg_state(prepared, env)
    action = enumerate_local_prior_actions(
        state,
        prepared.context,
        env,
    ).coal_actions[0]
    terminal = apply_local_action(
        state,
        CoalescenceChoice(
            action.active_lineage_i,
            action.active_lineage_j,
            time_quantile=0.5,
            delta_time=0.01,
        ),
        prepared.context,
        env,
        log_prior=-1.25,
    )

    assert terminal.is_done
    assert terminal.log_reward == pytest.approx(
        terminal.log_likelihood - 1.25
    )
    assert terminal.partial_log_reward == pytest.approx(
        terminal.log_reward
    )
    result = splice_local_proposal(
        prepared,
        local_state_to_proposal(terminal, prepared),
    )
    assert result.is_valid, result.validation.errors
    direct = compute_tree_sequence_vcf_log_likelihood(
        result.refined_tree_sequence,
        data,
        mutation_rate=env.mutation_rate,
    )
    assert terminal.log_likelihood == pytest.approx(direct, abs=1e-8)


def test_recombination_partitions_vcf_rows_and_undo_is_exact(tmp_path):
    ts = _two_sample_tree()
    data = load_vcf_variants(_write_vcf(tmp_path))
    env = _environment(data)
    prepared = _prepared(ts)
    state = initialize_local_arg_state(prepared, env)
    choice = enumerate_local_prior_actions(
        state,
        prepared.context,
        env,
    ).recomb_choices[0]
    next_state = apply_local_action(
        state,
        RecombinationChoice(
            active_lineage_i=choice.active_lineage_i,
            material_count=choice.material_count,
            span_start=choice.span_start,
            span_end=choice.span_end,
            breakpoint=1,
            time_quantile=0.5,
            delta_time=0.005,
        ),
        prepared.context,
        env,
        log_prior=-0.75,
    )
    left_parent, right_parent = next_state.active_lineages[-2:]
    assert left_parent.variant_indices == (0,)
    assert right_parent.variant_indices == (1,)
    assert next_state.accumulated_log_likelihood == pytest.approx(
        state.accumulated_log_likelihood
    )

    restored = undo_local_transition(next_state, prepared.context)
    assert restored.current_time == state.current_time
    assert restored.accumulated_log_prior == state.accumulated_log_prior
    assert (
        restored.accumulated_log_likelihood
        == state.accumulated_log_likelihood
    )
    assert restored.partial_log_reward == state.partial_log_reward
    for observed, expected in zip(
        restored.active_lineages,
        state.active_lineages,
    ):
        assert observed.variant_indices == expected.variant_indices
        assert torch.equal(observed.partials, expected.partials)


def test_fixed_ancestor_attachment_recovers_source_likelihood(tmp_path):
    ts = _two_sample_tree()
    data = load_vcf_variants(_write_vcf(tmp_path))
    env = _environment(data)
    prepared = _prepared(ts)
    state = initialize_local_arg_state(prepared, env)
    assert state.fixed_ancestor_schedule
    fixed_time = state.fixed_ancestor_schedule[0]["time"]
    terminal = reveal_due_fixed_ancestors(
        state,
        prepared.context,
        fixed_time,
        env=env,
    )
    terminal.is_done = True
    # The public transition wrapper normally performs terminal finalization.
    from arg.new_rl import compute_local_terminal_log_likelihood

    local = compute_local_terminal_log_likelihood(
        terminal,
        prepared.context,
        env,
    )
    direct = compute_tree_sequence_vcf_log_likelihood(
        ts,
        data,
        mutation_rate=env.mutation_rate,
    )
    assert local == pytest.approx(direct, abs=1e-12)
    restored = undo_local_transition(terminal, prepared.context)
    assert (
        restored.accumulated_log_likelihood
        == state.accumulated_log_likelihood
    )
    assert restored.partial_log_reward == state.partial_log_reward
    assert [lineage.node_id for lineage in restored.active_lineages] == [
        lineage.node_id for lineage in state.active_lineages
    ]
    for observed, expected in zip(
        restored.active_lineages,
        state.active_lineages,
    ):
        assert observed.variant_indices == expected.variant_indices
        assert torch.equal(observed.partials, expected.partials)


def test_sparse_state_flow_encoder_accepts_geometry_only_lineages(tmp_path):
    ts = _two_sample_tree()
    data = load_vcf_variants(_write_vcf(tmp_path))
    env = _environment(data)
    state = initialize_local_arg_state(
        _prepared(ts, region=(3.0, 4.0)),
        env,
    )
    assert state.target_variant_indices == ()
    assert all(lineage.variant_indices == () for lineage in state.active_lineages)
    assert state.outside_log_likelihood == pytest.approx(
        compute_tree_sequence_vcf_log_likelihood(
            ts,
            data,
            mutation_rate=env.mutation_rate,
        )
    )
    view_record = state.transition_records[0]["region_vcf_view"]
    assert view_record["target_variant_count"] == 0
    assert view_record["outside_variant_count"] == 2
    assert view_record["endpoint_variant_row_count"] == 0

    model = ARGModel(
        env,
        embedding_size=16,
        hidden_size=32,
        transformer_depth=1,
        transformer_heads=4,
        breakpoint_gap_layers=1,
        time_layers=1,
    )
    model.eval()
    with torch.no_grad():
        lineage_reps, summary_reps, _, _ = model._encode_states([state])
        flows = model.compute_log_state_flows(summary_reps)
    assert torch.isfinite(lineage_reps).all()
    assert torch.isfinite(flows).all()


@pytest.mark.slow
def test_25kb_terminal_likelihood_matches_independent_clean_rescore():
    source = tskit.load(
        str(ARG_ROOT / "validation/output/tsinfer/l25kb_dated.trees")
    )
    data = load_vcf_variants(
        ARG_ROOT / "validation/vcf/sim_l25kb_0.vcf"
    )
    prepared = prepare_local_refinement(
        source,
        LocalRefinementRequest(
            (386.0, 23_963.0),
            cut_time=25_000.0,
        ),
    )
    env = _environment(data)
    config = LocalSamplingConfig(
        sample_count=1,
        seed=1,
        max_generated_events=100,
        max_searched_states=10_000,
        max_restarts=4,
    )
    first = sample_local_trajectories(prepared, env, config)
    second = sample_local_trajectories(prepared, env, config)
    assert first.is_complete, first.diagnostics
    assert second.is_complete, second.diagnostics
    assert (
        first.proposals[0].transition_records
        == second.proposals[0].transition_records
    )
    assert first.proposals[0].log_likelihood == second.proposals[0].log_likelihood
    assert first.proposals[0].log_reward == second.proposals[0].log_reward

    proposal = first.proposals[0]
    assert proposal.log_likelihood == pytest.approx(
        proposal.outside_log_likelihood + proposal.local_log_likelihood
    )
    assert proposal.likelihood_alignment["vcf_coordinate_offset"] == 1.0
    assert all(
        "log_prior_increment" in record
        and "log_likelihood_increment" in record
        for record in proposal.transition_records
        if record["event_type"] != "initialization"
    )
    result = splice_local_proposal(prepared, proposal)
    assert result.is_valid, result.validation.errors
    assert result.provenance_record["parameters"]["reward_C"] == 0.0
    assert result.provenance_record["proposal"]["local_cwr_log_prior"] == (
        proposal.prior_log_probability
    )
    direct = compute_tree_sequence_vcf_log_likelihood(
        result.refined_tree_sequence,
        data,
        mutation_rate=env.mutation_rate,
    )
    assert proposal.log_likelihood == pytest.approx(direct, abs=1e-7)


@pytest.mark.slow
def test_1mb_likelihood_construction_export_and_reload_smoke(tmp_path):
    source = tskit.load(
        str(ARG_ROOT / "validation/output/tsinfer/l1mb_dated.trees")
    )
    data = load_vcf_variants(
        ARG_ROOT / "validation/vcf/sim_l1mb_0.vcf"
    )
    prepared = prepare_local_refinement(
        source,
        LocalRefinementRequest(
            (14_000.0, 190_000.0),
            cut_time=25_000.0,
        ),
    )
    env = _environment(data)
    initial = initialize_local_arg_state(prepared, env)
    batch = sample_local_trajectories(
        prepared,
        env,
        LocalSamplingConfig(sample_count=1, seed=1),
        initial_state=initial,
    )
    assert batch.is_complete, batch.diagnostics
    result = splice_local_proposal(prepared, batch.proposals[0])
    assert result.is_valid, result.validation.errors
    assert result.validation.counts["likelihood_parity"]
    output = export_refined_tree_sequence(
        result,
        tmp_path / "l1mb_vcf_local_refined.trees",
    )
    reloaded = tskit.load(str(output))
    assert reloaded.tables.equals(result.refined_tree_sequence.tables)
