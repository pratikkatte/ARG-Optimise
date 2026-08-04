"""Independent accounting checks for forward-looking local ARG rewards.

These tests intentionally recompute JC69 partials and CwR factors from their
defining equations instead of calling the production likelihood/prior helpers.
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest

from arg import refinement
from arg.env import (
    CoalescenceChoice,
    FixedAttachmentChoice,
    LocalARGEnvironment,
    RecombinationChoice,
    SimpleARGEnvironment,
)
from arg.refinement.vcf_likelihood import (
    resolve_vcf_tree_sequence_alignment,
)
from arg.utils import load_vcf_variants


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ARG = ROOT / "validation/output/tsinfer/l25kb_dated.trees"
VCF = ROOT / "validation/vcf/sim_l25kb_0.vcf"
MUTATION_RATE = 2e-8
POPULATION_SIZE = 10_000.0


@pytest.fixture(scope="module")
def local_audit_data():
    variants = load_vcf_variants(str(VCF))
    return {
        "variants": variants,
        "generated": refinement.prepare_local_refinement(
            SOURCE_ARG,
            refinement.LocalRefinementRequest(
                (386.0, 23_963.0),
                cut_time=25_000.0,
            ),
        ),
        "boundary": refinement.prepare_local_refinement(
            SOURCE_ARG,
            refinement.LocalRefinementRequest(
                (1_000.0, 5_000.0),
                cut_time=100.0,
            ),
        ),
    }


def _local_env(local_audit_data, context_name, *, recombination_rate):
    variants = local_audit_data["variants"]
    base = SimpleARGEnvironment(
        sequence_length=int(variants.sequence_length),
        num_sequences=int(variants.num_haplotypes),
        bp_per_blocks=1,
        variant_data=variants,
        device="cpu",
        recombination_rate=float(recombination_rate),
        population_size=POPULATION_SIZE,
        mutation_rate=MUTATION_RATE,
        reward_C=30_000.0,
    )
    return LocalARGEnvironment(
        base,
        {context_name: local_audit_data[context_name]},
        terminal_requires_exhausted_fixed_schedule=True,
    )


def _jc69_matrix(branch_length):
    decay = math.exp(-4.0 * float(branch_length) / 3.0)
    same = 0.25 + 0.75 * decay
    different = 0.25 - 0.25 * decay
    matrix = np.full((4, 4), different, dtype=np.float64)
    np.fill_diagonal(matrix, same)
    return matrix


def _oracle_source_partial(
    tree_sequence,
    tree,
    variants,
    alignment,
    variant_index,
    node_id,
    memo,
):
    """Brute-force Felsenstein pruning oracle for one node and VCF row."""

    node_id = int(node_id)
    if node_id in memo:
        return memo[node_id]
    node = tree_sequence.node(node_id)
    if node.is_sample():
        haplotype = int(
            alignment["haplotype_index_by_sample_node"][node_id]
        )
        combined = np.asarray(
            variants.haplotype_partials[haplotype, int(variant_index)],
            dtype=np.float64,
        ).copy()
    else:
        combined = np.ones(4, dtype=np.float64)

    log_scale = 0.0
    for child_id in tree.children(node_id):
        child_partial, child_scale = _oracle_source_partial(
            tree_sequence,
            tree,
            variants,
            alignment,
            variant_index,
            int(child_id),
            memo,
        )
        edge_time = float(node.time) - float(
            tree_sequence.node(int(child_id)).time
        )
        assert edge_time > 0.0
        combined *= _jc69_matrix(edge_time * MUTATION_RATE) @ child_partial
        log_scale += float(child_scale)

    row_sum = float(combined.sum())
    assert row_sum > 0.0 and math.isfinite(row_sum)
    result = combined / row_sum, log_scale + math.log(row_sum)
    memo[node_id] = result
    return result


def _oracle_tree_log_likelihood(
    tree_sequence,
    variants,
    alignment,
    variant_indices,
):
    total = 0.0
    coordinates = np.asarray(
        alignment["variant_coordinates"],
        dtype=np.float64,
    )
    for variant_index in variant_indices:
        tree = tree_sequence.at(float(coordinates[int(variant_index)]))
        memo = {}
        for root in tree.roots:
            partial, scale = _oracle_source_partial(
                tree_sequence,
                tree,
                variants,
                alignment,
                variant_index,
                int(root),
                memo,
            )
            total += math.log(float(np.sum(partial * 0.25))) + scale
    return float(total)


def _transition_local_partial(partial, delta_scaled, env):
    branch_length = (
        float(delta_scaled)
        * 2.0
        * float(env.population_size)
        * float(env.mutation_rate)
    )
    transitioned = _jc69_matrix(branch_length) @ np.asarray(
        partial,
        dtype=np.float64,
    )
    return transitioned / transitioned.sum()


def _covers(segments, block):
    return any(int(left) <= int(block) < int(right) for left, right in segments)


def _physical_overlap_weight(left, right, state):
    boundaries = state.block_boundaries
    target_segments = (
        state.target_material.segments
        if state.target_material is not None
        else ((0, len(boundaries) - 1),)
    )
    length = 0.0
    for left_start, left_end in left.material_segments.segments:
        for right_start, right_end in right.material_segments.segments:
            for target_start, target_end in target_segments:
                start = max(left_start, right_start, target_start)
                end = min(left_end, right_end, target_end)
                if start < end:
                    length += float(boundaries[end]) - float(boundaries[start])
    return max(length, 1.0) if length > 0.0 else 0.0


def _breakpoint_weights(lineage, state):
    output = []
    material = lineage.material_segments
    for breakpoint, weight in sorted(state.local_breakpoint_weights.items()):
        if not (
            int(material.span_start) < int(breakpoint) <= int(material.span_end)
        ):
            continue
        left, right = material.split(int(breakpoint))
        if left.count and right.count and float(weight) > 0.0:
            output.append((int(breakpoint), float(weight)))
    return output


def _manual_production_event_log_prior(env, state, action):
    """Reconstruct the production local prior without its scoring helper."""

    options = env.enumerate_prior_options(state)
    total_rate = float(
        options.rates["lambda_coal"] + options.rates["lambda_recomb"]
    )
    wait = math.log(total_rate) - total_rate * float(action.delta_time)
    if isinstance(action, CoalescenceChoice):
        assert any(
            {
                int(choice.active_lineage_i),
                int(choice.active_lineage_j),
            }
            == {
                int(action.active_lineage_i),
                int(action.active_lineage_j),
            }
            for choice in options.coal_actions
        )
        choice = (
            math.log(float(options.rates["lambda_coal"]) / total_rate)
            - math.log(len(options.coal_actions))
        )
    elif isinstance(action, RecombinationChoice):
        lineage_weights = [
            sum(
                weight
                for _breakpoint, weight in _breakpoint_weights(
                    state.active_lineages[candidate.active_lineage_i],
                    state,
                )
            )
            for candidate in options.recomb_choices
        ]
        selected = next(
            index
            for index, candidate in enumerate(options.recomb_choices)
            if int(candidate.active_lineage_i) == int(action.active_lineage_i)
        )
        breakpoint_rows = _breakpoint_weights(
            state.active_lineages[action.active_lineage_i],
            state,
        )
        breakpoint_weight = next(
            weight
            for breakpoint, weight in breakpoint_rows
            if int(breakpoint) == int(action.breakpoint)
        )
        choice = (
            math.log(float(options.rates["lambda_recomb"]) / total_rate)
            + math.log(lineage_weights[selected] / sum(lineage_weights))
            + math.log(breakpoint_weight / sum(weight for _, weight in breakpoint_rows))
        )
    else:  # pragma: no cover - helper is called only for generated events
        raise TypeError(type(action))
    return float(wait + choice)


@pytest.mark.parametrize("context_name", ["generated", "boundary"])
def test_initial_cut_likelihood_matches_independent_jc69_pruning(
    local_audit_data,
    context_name,
):
    env = _local_env(
        local_audit_data,
        context_name,
        recombination_rate=2e-8,
    )
    state = env.get_initial_state(context_name)
    prepared = local_audit_data[context_name]
    variants = local_audit_data["variants"]
    coordinates = np.asarray(
        state.vcf_alignment["variant_coordinates"],
        dtype=np.float64,
    )

    oracle_inside_scale = 0.0
    for lineage in state.active_lineages:
        for row_index, variant_index in enumerate(lineage.variant_indices):
            tree = prepared.source_tree_sequence.at(
                float(coordinates[int(variant_index)])
            )
            partial, scale = _oracle_source_partial(
                prepared.source_tree_sequence,
                tree,
                variants,
                state.vcf_alignment,
                int(variant_index),
                int(lineage.node_id),
                {},
            )
            np.testing.assert_allclose(
                lineage.partials[row_index].detach().cpu().numpy(),
                partial,
                rtol=1e-10,
                atol=1e-12,
            )
            oracle_inside_scale += scale

    outside_indices = sorted(
        set(range(int(variants.num_variants)))
        - set(int(value) for value in state.target_variant_indices)
    )
    oracle_outside = _oracle_tree_log_likelihood(
        prepared.source_tree_sequence,
        variants,
        state.vcf_alignment,
        outside_indices,
    )
    assert state.outside_log_likelihood == pytest.approx(
        oracle_outside,
        rel=1e-10,
        abs=1e-8,
    )
    assert state.partial_log_reward == pytest.approx(
        oracle_inside_scale,
        rel=1e-10,
        abs=1e-8,
    )
    assert state.accumulated_log_likelihood == pytest.approx(
        oracle_outside + oracle_inside_scale,
        rel=1e-10,
        abs=1e-8,
    )
    assert state.transition_records[0]["inside_initial_log_scale"] == pytest.approx(
        oracle_inside_scale,
        rel=1e-10,
        abs=1e-8,
    )


def test_generated_coalescence_prior_and_likelihood_increment(
    local_audit_data,
):
    env = _local_env(
        local_audit_data,
        "generated",
        recombination_rate=2e-8,
    )
    state = env.get_initial_state("generated")
    options = env.enumerate_prior_options(state)
    total_rate = float(
        options.rates["lambda_coal"] + options.rates["lambda_recomb"]
    )
    delta = -math.log1p(-0.37) / total_rate
    action = replace(
        options.coal_actions[0],
        time_quantile=0.37,
        delta_time=delta,
    )
    observed_prior = env.compute_cwr_event_log_prior(
        state,
        env.enumerate_actions(state),
        action,
        rates=options.rates,
    )
    assert observed_prior == pytest.approx(
        _manual_production_event_log_prior(env, state, action),
        rel=1e-12,
        abs=1e-12,
    )

    # Each permitted overlapping pair has rate one.  The total-rate waiting
    # density and uniform pair choice therefore cancel to the specified pair's
    # log density -lambda_total * delta.
    canonical_uniform_pair_prior = -total_rate * delta
    assert observed_prior == pytest.approx(
        canonical_uniform_pair_prior,
        rel=1e-12,
        abs=1e-12,
    )

    children = (
        state.active_lineages[action.active_lineage_i],
        state.active_lineages[action.active_lineage_j],
    )
    next_state = env.apply_action(state, action, log_prior=observed_prior)
    parent = next_state.all_nodes[int(next_state.max_node_idx)]
    expected_rows = []
    expected_increment = 0.0
    parent_time = float(next_state.current_time)
    for variant_index in parent.variant_indices:
        combined = np.ones(4, dtype=np.float64)
        for child in children:
            if int(variant_index) not in child.variant_indices:
                continue
            row = child.variant_indices.index(int(variant_index))
            combined *= _transition_local_partial(
                child.partials[row].detach().cpu().numpy(),
                parent_time - float(child.time),
                env,
            )
        row_sum = float(combined.sum())
        expected_increment += math.log(row_sum)
        expected_rows.append(combined / row_sum)

    np.testing.assert_allclose(
        parent.partials.detach().cpu().numpy(),
        np.asarray(expected_rows),
        rtol=1e-10,
        atol=1e-12,
    )
    likelihood_increment = (
        next_state.accumulated_log_likelihood
        - state.accumulated_log_likelihood
    )
    assert likelihood_increment == pytest.approx(
        expected_increment,
        rel=1e-10,
        abs=1e-8,
    )
    assert next_state.partial_log_reward - state.partial_log_reward == pytest.approx(
        observed_prior + expected_increment,
        rel=1e-10,
        abs=1e-8,
    )
    restored = env.undo_transition(next_state)
    assert restored.partial_log_reward == pytest.approx(state.partial_log_reward)
    assert restored.accumulated_log_prior == pytest.approx(
        state.accumulated_log_prior
    )
    assert restored.accumulated_log_likelihood == pytest.approx(
        state.accumulated_log_likelihood
    )


def test_overlap_cwr_has_every_overlapping_pair_at_unit_rate(
    local_audit_data,
):
    env = _local_env(
        local_audit_data,
        "generated",
        recombination_rate=0.0,
    )
    state = env.get_initial_state("generated")
    options = env.enumerate_prior_options(state)
    observed_pairs = {
        tuple(sorted((choice.active_lineage_i, choice.active_lineage_j)))
        for choice in options.coal_actions
    }
    expected_pairs = {
        (left, right)
        for left in range(len(state.active_lineages))
        for right in range(left + 1, len(state.active_lineages))
        if state.active_lineages[left].material_segments.overlaps(
            state.active_lineages[right].material_segments
        )
    }
    assert observed_pairs == expected_pairs
    expected_pair_count = len(expected_pairs)
    assert len(options.coal_actions) == expected_pair_count
    assert options.rates["lambda_coal"] == float(expected_pair_count)

    # This fixture contains different positive overlap lengths.  All supported
    # pairs nevertheless have the same unit rate.
    overlap_weights = [
        _physical_overlap_weight(
            state.active_lineages[choice.active_lineage_i],
            state.active_lineages[choice.active_lineage_j],
            state,
        )
        for choice in options.coal_actions
    ]
    assert all(weight > 0.0 for weight in overlap_weights)
    assert len(set(overlap_weights)) > 1

    all_pairs = {
        (left, right)
        for left in range(len(state.active_lineages))
        for right in range(left + 1, len(state.active_lineages))
    }
    assert expected_pairs < all_pairs

    total_rate = float(options.rates["lambda_coal"])
    delta = 0.37 / total_rate
    expected_log_density_per_pair = -total_rate * delta
    observed_scores = []
    for candidate in options.coal_actions:
        action = replace(
            candidate,
            time_quantile=1.0 - math.exp(-total_rate * delta),
            delta_time=delta,
        )
        observed_scores.append(
            env.compute_cwr_event_log_prior(
                state,
                env.enumerate_actions(state),
                action,
                rates=options.rates,
            )
        )
    assert observed_scores == pytest.approx(
        [expected_log_density_per_pair] * expected_pair_count,
        rel=1e-12,
        abs=1e-12,
    )


def test_overlap_cwr_excludes_disjoint_common_ancestor_events(
    local_audit_data,
):
    env = _local_env(
        local_audit_data,
        "generated",
        recombination_rate=0.0,
    )
    state = env.get_initial_state("generated")
    options = env.enumerate_prior_options(state)
    disjoint_pair = next(
        (left, right)
        for left in range(len(state.active_lineages))
        for right in range(left + 1, len(state.active_lineages))
        if not state.active_lineages[left].material_segments.overlaps(
            state.active_lineages[right].material_segments
        )
    )
    assert disjoint_pair not in {
        tuple(sorted((choice.active_lineage_i, choice.active_lineage_j)))
        for choice in options.coal_actions
    }
    total_rate = float(options.rates["lambda_coal"])
    delta = 0.23 / total_rate
    action = CoalescenceChoice(
        active_lineage_i=disjoint_pair[0],
        active_lineage_j=disjoint_pair[1],
        time_quantile=1.0 - math.exp(-total_rate * delta),
        delta_time=delta,
    )
    with pytest.raises(ValueError, match="invalid local coalescence"):
        env.compute_cwr_event_log_prior(
            state,
            env.enumerate_actions(state),
            action,
            rates=options.rates,
        )


def test_generated_recombination_prior_and_zero_likelihood_scale(
    local_audit_data,
):
    env = _local_env(
        local_audit_data,
        "generated",
        recombination_rate=2e-8,
    )
    state = env.get_initial_state("generated")
    options = env.enumerate_prior_options(state)
    total_rate = float(
        options.rates["lambda_coal"] + options.rates["lambda_recomb"]
    )
    candidate = options.recomb_choices[0]
    breakpoint, breakpoint_weight = _breakpoint_weights(
        state.active_lineages[candidate.active_lineage_i],
        state,
    )[0]
    delta = -math.log1p(-0.41) / total_rate
    action = replace(
        candidate,
        breakpoint=int(breakpoint),
        time_quantile=0.41,
        delta_time=delta,
    )
    observed_prior = env.compute_cwr_event_log_prior(
        state,
        env.enumerate_actions(state),
        action,
        rates=options.rates,
    )
    assert observed_prior == pytest.approx(
        _manual_production_event_log_prior(env, state, action),
        rel=1e-12,
        abs=1e-12,
    )
    # Marginalizing a continuous breakpoint interval of physical width w gives
    # per-gap rate (rho / (2L)) * w in 2Ne-scaled time.
    expected_gap_prior = (
        -total_rate * delta
        + math.log(float(env.rho) / (2.0 * float(env.sequence_length)))
        + math.log(float(breakpoint_weight))
    )
    assert observed_prior == pytest.approx(
        expected_gap_prior,
        rel=1e-12,
        abs=1e-12,
    )

    child = state.active_lineages[action.active_lineage_i]
    next_state = env.apply_action(state, action, log_prior=observed_prior)
    created = [
        next_state.all_nodes[node_id]
        for node_id in range(state.max_node_idx + 1, next_state.max_node_idx + 1)
    ]
    assert len(created) == 2
    for parent in created:
        expected = []
        for variant_index in parent.variant_indices:
            row = child.variant_indices.index(int(variant_index))
            expected.append(
                _transition_local_partial(
                    child.partials[row].detach().cpu().numpy(),
                    float(parent.time) - float(child.time),
                    env,
                )
            )
        np.testing.assert_allclose(
            parent.partials.detach().cpu().numpy(),
            np.asarray(expected),
            rtol=1e-10,
            atol=1e-12,
        )
    assert next_state.accumulated_log_likelihood == pytest.approx(
        state.accumulated_log_likelihood
    )
    assert next_state.partial_log_reward - state.partial_log_reward == pytest.approx(
        observed_prior,
        rel=1e-12,
        abs=1e-12,
    )


def test_local_generated_event_density_plus_fixed_survival_is_normalized(
    local_audit_data,
):
    env = _local_env(
        local_audit_data,
        "boundary",
        recombination_rate=2e-8,
    )
    state = env.get_initial_state("boundary")
    options = env.enumerate_prior_options(state)
    total_rate = float(
        options.rates["lambda_coal"] + options.rates["lambda_recomb"]
    )
    horizon = float(env.next_fixed_ancestor_time(state)) - float(
        state.current_time
    )
    delta = 0.5 * horizon
    wait_log_density = math.log(total_rate) - total_rate * delta

    conditional_action_mass = 0.0
    for candidate in options.coal_actions:
        action = replace(
            candidate,
            time_quantile=env.delta_to_time_quantile(state, delta),
            delta_time=delta,
        )
        score = env.compute_cwr_event_log_prior(
            state,
            env.enumerate_actions(state),
            action,
            rates=options.rates,
        )
        conditional_action_mass += math.exp(score - wait_log_density)
    for candidate in options.recomb_choices:
        lineage = state.active_lineages[candidate.active_lineage_i]
        for breakpoint, _weight in _breakpoint_weights(lineage, state):
            action = replace(
                candidate,
                breakpoint=int(breakpoint),
                time_quantile=env.delta_to_time_quantile(state, delta),
                delta_time=delta,
            )
            score = env.compute_cwr_event_log_prior(
                state,
                env.enumerate_actions(state),
                action,
                rates=options.rates,
            )
            conditional_action_mass += math.exp(score - wait_log_density)
    assert conditional_action_mass == pytest.approx(1.0, rel=1e-12, abs=1e-12)

    generated_mass = 1.0 - math.exp(-total_rate * horizon)
    survival_mass = math.exp(env.fixed_attachment_log_prior(state))
    assert generated_mass + survival_mass == pytest.approx(
        1.0,
        rel=1e-12,
        abs=1e-12,
    )


def test_fixed_attachment_survival_and_likelihood_increment(
    local_audit_data,
):
    env = _local_env(
        local_audit_data,
        "boundary",
        recombination_rate=2e-8,
    )
    state = env.get_initial_state("boundary")
    options = env.enumerate_prior_options(state)
    total_rate = float(
        options.rates["lambda_coal"] + options.rates["lambda_recomb"]
    )
    fixed_time = float(env.next_fixed_ancestor_time(state))
    horizon = fixed_time - float(state.current_time)
    observed_prior = env.fixed_attachment_log_prior(state)
    assert observed_prior == pytest.approx(
        -total_rate * horizon,
        rel=1e-12,
        abs=1e-12,
    )

    next_state = env.apply_action(
        state,
        FixedAttachmentChoice(fixed_time),
        log_prior=observed_prior,
    )
    record = next_state.transition_records[-1]
    assert record["survival_log_probability"] == pytest.approx(observed_prior)
    assert len(record["node_ids"]) == 1
    parent = next_state.all_nodes[int(record["node_ids"][0])]

    expected_rows = []
    expected_increment = 0.0
    for variant_index in parent.variant_indices:
        block = int(state.variant_block_indices[int(variant_index)])
        combined = np.ones(4, dtype=np.float64)
        for attachment in record["attachments"]:
            if not _covers(attachment["segments"], block):
                continue
            child = state.all_nodes[int(attachment["child_node_id"])]
            row = child.variant_indices.index(int(variant_index))
            combined *= _transition_local_partial(
                child.partials[row].detach().cpu().numpy(),
                fixed_time - float(child.time),
                env,
            )
        row_sum = float(combined.sum())
        expected_increment += math.log(row_sum)
        expected_rows.append(combined / row_sum)

    np.testing.assert_allclose(
        parent.partials.detach().cpu().numpy(),
        np.asarray(expected_rows),
        rtol=1e-10,
        atol=1e-12,
    )
    likelihood_increment = (
        next_state.accumulated_log_likelihood
        - state.accumulated_log_likelihood
    )
    assert likelihood_increment == pytest.approx(
        expected_increment,
        rel=1e-10,
        abs=1e-8,
    )
    assert record["log_likelihood_increment"] == pytest.approx(
        expected_increment,
        rel=1e-10,
        abs=1e-8,
    )
    assert next_state.partial_log_reward - state.partial_log_reward == pytest.approx(
        observed_prior + expected_increment,
        rel=1e-10,
        abs=1e-8,
    )


def test_revealed_fixed_source_uses_the_same_overlap_cwr_event_law(
    local_audit_data,
):
    env = _local_env(
        local_audit_data,
        "boundary",
        recombination_rate=2e-8,
    )
    state = env.get_initial_state("boundary")
    first_fixed_time = float(env.next_fixed_ancestor_time(state))
    state = env.apply_action(
        state,
        FixedAttachmentChoice(first_fixed_time),
        log_prior=env.fixed_attachment_log_prior(state),
    )
    fixed_indices = {
        index
        for index, lineage in enumerate(state.active_lineages)
        if lineage.event_type == "fixed_source"
    }
    assert fixed_indices

    options = env.enumerate_prior_options(state)
    expected_pairs = {
        (left, right)
        for left in range(len(state.active_lineages))
        for right in range(left + 1, len(state.active_lineages))
        if state.active_lineages[left].material_segments.overlaps(
            state.active_lineages[right].material_segments
        )
    }
    expected_pair_count = len(expected_pairs)
    assert len(options.coal_actions) == expected_pair_count
    assert options.rates["lambda_coal"] == float(expected_pair_count)
    assert any(
        choice.active_lineage_i in fixed_indices
        or choice.active_lineage_j in fixed_indices
        for choice in options.coal_actions
    )
    assert any(
        choice.active_lineage_i in fixed_indices
        for choice in options.recomb_choices
    )

    next_fixed_time = float(env.next_fixed_ancestor_time(state))
    horizon = next_fixed_time - float(state.current_time)
    delta = 0.5 * horizon
    total_rate = float(
        options.rates["lambda_coal"] + options.rates["lambda_recomb"]
    )
    candidate = next(
        choice
        for choice in options.coal_actions
        if choice.active_lineage_i in fixed_indices
        or choice.active_lineage_j in fixed_indices
    )
    action = replace(
        candidate,
        time_quantile=env.delta_to_time_quantile(state, delta),
        delta_time=delta,
    )
    observed = env.compute_cwr_event_log_prior(
        state,
        env.enumerate_actions(state),
        action,
        rates=options.rates,
    )
    assert observed == pytest.approx(
        -total_rate * delta,
        rel=1e-12,
        abs=1e-12,
    )


def _complete_without_recombination(env, context_name):
    state = env.get_initial_state(context_name)
    initial_partial = float(state.partial_log_reward)
    increments = []
    event_types = []
    for _step in range(256):
        if state.is_done:
            break
        fixed_time = env.next_fixed_ancestor_time(state)
        if fixed_time is not None:
            action = FixedAttachmentChoice(float(fixed_time))
            log_prior = env.fixed_attachment_log_prior(state)
            expected_prior = -(
                sum(env.enumerate_prior_options(state).rates[key] for key in (
                    "lambda_coal",
                    "lambda_recomb",
                ))
                * (float(fixed_time) - float(state.current_time))
            )
            event_type = "fixed_attachment"
        else:
            options = env.enumerate_prior_options(state)
            assert options.coal_actions
            total_rate = float(options.rates["lambda_coal"])
            delta = -math.log1p(-0.5) / total_rate
            action = replace(
                options.coal_actions[0],
                time_quantile=0.5,
                delta_time=delta,
            )
            log_prior = env.compute_cwr_event_log_prior(
                state,
                env.enumerate_actions(state),
                action,
                rates=options.rates,
            )
            expected_prior = _manual_production_event_log_prior(
                env,
                state,
                action,
            )
            event_type = "coal"
        assert log_prior == pytest.approx(expected_prior, rel=1e-12, abs=1e-12)

        previous = state
        state = env.apply_action(previous, action, log_prior=log_prior)
        likelihood_increment = (
            state.accumulated_log_likelihood
            - previous.accumulated_log_likelihood
        )
        terminal_correction = (
            float(state.terminal_partial_correction) if state.is_done else 0.0
        )
        expected_partial_increment = (
            float(log_prior) + likelihood_increment + terminal_correction
        )
        observed_partial_increment = (
            float(state.partial_log_reward)
            - float(previous.partial_log_reward)
        )
        assert observed_partial_increment == pytest.approx(
            expected_partial_increment,
            rel=1e-10,
            abs=1e-8,
        )
        increments.append(observed_partial_increment)
        event_types.append(event_type)
    else:  # pragma: no cover - fail clearly if termination regresses
        raise AssertionError("local accounting audit did not reach a terminal state")
    assert state.is_done
    return state, initial_partial, increments, event_types


@pytest.mark.parametrize(
    ("context_name", "required_event_type"),
    [("generated", "coal"), ("boundary", "fixed_attachment")],
)
def test_complete_local_path_telescopes_to_independently_rescored_terminal_reward(
    local_audit_data,
    context_name,
    required_event_type,
):
    env = _local_env(
        local_audit_data,
        context_name,
        recombination_rate=0.0,
    )
    state, initial_partial, increments, event_types = (
        _complete_without_recombination(env, context_name)
    )
    assert required_event_type in event_types
    assert initial_partial + sum(increments) == pytest.approx(
        state.log_reward,
        rel=1e-10,
        abs=1e-8,
    )
    assert state.partial_log_reward == state.log_reward

    root_correction = (
        float(state.log_likelihood)
        - float(state.accumulated_log_likelihood)
    )
    assert state.terminal_partial_correction == pytest.approx(
        root_correction,
        rel=1e-10,
        abs=1e-8,
    )
    expected_local_reward = (
        float(state.log_likelihood)
        + float(state.accumulated_log_prior)
        - float(state.outside_log_likelihood)
    )
    assert state.log_reward == pytest.approx(
        expected_local_reward,
        rel=1e-10,
        abs=1e-8,
    )
    assert state.absolute_log_reward == pytest.approx(
        float(env.reward_fn.C)
        + float(state.log_likelihood)
        + float(state.accumulated_log_prior),
        rel=1e-10,
        abs=1e-8,
    )

    proposal = env.state_to_proposal(state)
    splice = refinement.splice_local_proposal(
        local_audit_data[context_name],
        proposal,
    )
    assert splice.is_valid, splice.validation.errors
    refined_alignment = resolve_vcf_tree_sequence_alignment(
        splice.refined_tree_sequence,
        local_audit_data["variants"],
    )
    oracle_terminal_likelihood = _oracle_tree_log_likelihood(
        splice.refined_tree_sequence,
        local_audit_data["variants"],
        refined_alignment,
        range(int(local_audit_data["variants"].num_variants)),
    )
    assert state.log_likelihood == pytest.approx(
        oracle_terminal_likelihood,
        rel=1e-10,
        abs=1e-8,
    )
