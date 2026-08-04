from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import tskit

from arg import new_rl as reference
from arg import refinement as production
from arg.env import (
    FixedAttachmentChoice,
    LocalARGEnvironment,
    RecombinationChoice,
    SimpleARGEnvironment,
    action_as_dict,
)
from arg.rollout_worker_arg import RolloutWorker
from arg.tb_gfn import TBGFlowNetGenerator
from arg.flow_evaluation import (
    evaluate_fixed_bank,
    fixed_bank_signature,
    generate_fixed_evaluation_bank,
)
from arg.train import load_train_config, validate_train_config
from arg.utils import load_vcf_variants
from arg.refinement.training import (
    SeededContextSampler,
    _load_shape_compatible_weights,
)
from arg.refinement import local_construction as production_local
from arg.refinement import local_splice as production_splice
from arg.new_rl import local_splice as reference_splice
from arg.refinement.training import train_local_refinement
from arg.refinement.replay import (
    HybridReplayBuffer,
    reconstruct_and_rescore_entries,
)
from arg.infer import run_inference
from arg.validation.scripts.point_accuracy_common import (
    common_metric_values,
    dataframe_from_tree_sequences,
)
from arg.time_context import build_time_context, time_context_dim


ARG_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ARG = ARG_ROOT / "validation/output/tsinfer/l25kb_dated.trees"
VCF = ARG_ROOT / "validation/vcf/sim_l25kb_0.vcf"


def _synthetic_clock_boundary_tables(*, local_child_time):
    tables = tskit.TableCollection(sequence_length=1.0)
    sample = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
    source_parent = tables.nodes.add_row(time=9.9)
    routing_node = tables.nodes.add_row(time=5.0)
    tables.edges.add_row(0.0, 1.0, parent=source_parent, child=routing_node)
    tables.edges.add_row(0.0, 1.0, parent=routing_node, child=sample)
    tables.sort()
    local_child = tables.nodes.add_row(time=float(local_child_time))
    tables.edges.add_row(0.0, 1.0, parent=source_parent, child=local_child)
    return tables, source_parent, routing_node, local_child


@pytest.mark.parametrize("splice_module", [reference_splice, production_splice])
def test_routing_collapse_switches_to_biological_times_before_validation(
    splice_module,
):
    tables, source_parent, routing_node, local_child = (
        _synthetic_clock_boundary_tables(local_child_time=9.95)
    )
    with pytest.raises(tskit.LibraryError, match="parent.*greater"):
        tables.copy().sort()

    old_to_new = splice_module._collapse_nodes(
        tables,
        (routing_node,),
        source_node_times=np.asarray([0.0, 10.0]),
    )
    collapsed = tables.tree_sequence()
    mapped_parent = int(old_to_new[source_parent])
    mapped_child = int(old_to_new[local_child])

    assert collapsed.node(mapped_parent).time == 10.0
    assert collapsed.node(mapped_child).time == 9.95
    assert np.all(
        collapsed.tables.nodes.time[collapsed.tables.edges.parent]
        > collapsed.tables.nodes.time[collapsed.tables.edges.child]
    )


@pytest.mark.parametrize("splice_module", [reference_splice, production_splice])
def test_routing_collapse_rejects_biologically_invalid_local_edge(
    splice_module,
):
    tables, _source_parent, routing_node, _local_child = (
        _synthetic_clock_boundary_tables(local_child_time=10.0)
    )
    with pytest.raises(
        ValueError,
        match="collapsed local splice violates parent.time > child.time",
    ):
        splice_module._collapse_nodes(
            tables,
            (routing_node,),
            source_node_times=np.asarray([0.0, 10.0]),
        )


def _base_env(*, recombination_rate=0.0):
    variants = load_vcf_variants(str(VCF))
    return SimpleARGEnvironment(
        sequence_length=int(variants.sequence_length),
        num_sequences=int(variants.num_haplotypes),
        bp_per_blocks=1,
        variant_data=variants,
        device="cpu",
        recombination_rate=float(recombination_rate),
        population_size=10_000,
        mutation_rate=2e-8,
        reward_C=30_000,
    )


@pytest.fixture(scope="module")
def prepared_contexts():
    generated_request_ref = reference.LocalRefinementRequest(
        (386.0, 23963.0),
        cut_time=25_000.0,
    )
    generated_request_prod = production.LocalRefinementRequest(
        (386.0, 23963.0),
        cut_time=25_000.0,
    )
    boundary_request = production.LocalRefinementRequest(
        (1000.0, 5000.0),
        cut_time=100.0,
    )
    return {
        "reference": reference.prepare_local_refinement(
            SOURCE_ARG,
            generated_request_ref,
        ),
        "generated": production.prepare_local_refinement(
            SOURCE_ARG,
            generated_request_prod,
        ),
        "boundary": production.prepare_local_refinement(
            SOURCE_ARG,
            boundary_request,
        ),
    }


def _state_signature(state):
    return (
        float(state.current_time),
        tuple(lineage.node_id for lineage in state.active_lineages),
        tuple(
            (
                node_id,
                tuple(sorted(lineage.children)),
                tuple(sorted(lineage.parents)),
                tuple(lineage.material_segments.segments),
                lineage.event_type,
                lineage.breakpoint,
                lineage.recombination_side,
                float(lineage.time),
            )
            for node_id, lineage in sorted(state.all_nodes.items())
        ),
    )


def test_provisional_likelihood_time_context_is_finite(prepared_contexts):
    base_env = _base_env(recombination_rate=2e-8)
    local_env = LocalARGEnvironment(
        base_env,
        {"region_000001": prepared_contexts["generated"]},
    )
    state = local_env.get_initial_state("region_000001")
    coal, recomb = local_env.enumerate_actions(state)
    action = (coal + recomb)[0]
    if hasattr(action, "breakpoint"):
        valid = local_env.valid_breakpoints(state, action)
        action = replace(action, breakpoint=int(valid[0]))
    rollout = local_env.prepare_state_rollout_inputs([state])["rollout"][0]
    result = build_time_context(
        state,
        action,
        local_env,
        max_delta=rollout["max_delta"],
        mode="likelihood",
    )
    assert result.features.shape == (time_context_dim("likelihood"),)
    assert torch.isfinite(result.features).all()
    assert math.isfinite(result.diagnostics["provisional_likelihood_spread"])


def test_recombination_split_bias_integrates_atomic_breakpoint_and_rescore(
    prepared_contexts,
):
    base_env = _base_env(recombination_rate=2e-8)
    local_env = LocalARGEnvironment(
        base_env,
        {"region_000001": prepared_contexts["generated"]},
    )
    state = local_env.get_initial_state("region_000001")
    coal, recomb = local_env.enumerate_actions(state)
    assert recomb
    candidates = [list(coal) + list(recomb)]
    generator = TBGFlowNetGenerator(
        local_env,
        init_z_sample_count=0,
        device="cpu",
        verbose=False,
        initialize_z_from_prior=False,
        loss_mode="fl_subtb",
        model_kwargs={
            "embedding_size": 16,
            "hidden_size": 32,
            "transformer_depth": 1,
            "transformer_heads": 1,
            "breakpoint_hidden_dim": 16,
            "recombination_split_bias": {
                "enabled": True,
                "lineage_weight": 0.25,
                "breakpoint_weight": 0.25,
                "fragmentation_penalty": 0.1,
            },
        },
    )
    lineage_reps, summary_reps, _, _ = generator._encode_states([state])
    base_logits, action_features = generator.arg_model._score_candidates(
        candidates,
        lineage_reps,
        summary_reps,
        state_contexts=[state],
    )
    probability_logits, split_records, split_diagnostics = (
        generator.arg_model.prepare_action_probability_logits(
            base_logits,
            candidates,
            [state],
            random_spec={"T": 0.7},
        )
    )
    recombination_indices = list(range(len(coal), len(coal) + len(recomb)))
    base_probabilities = torch.softmax(base_logits / 0.7, dim=1)
    biased_probabilities = torch.softmax(probability_logits, dim=1)
    assert float(
        biased_probabilities[0, recombination_indices].sum().detach()
    ) == pytest.approx(
        float(base_probabilities[0, recombination_indices].sum().detach()),
        abs=1e-6,
    )
    assert split_diagnostics[0][
        "recombination_split_mass_absolute_error"
    ] < 1e-6

    candidate_index = recombination_indices[0]
    action = recomb[0]
    record = split_records[0][candidate_index]
    valid_breakpoints = local_env.valid_breakpoints(state, action)
    assert record.breakpoints == valid_breakpoints
    feature = action_features[0, candidate_index]
    lineage = state.active_lineages[action.active_lineage_i]
    base_breakpoint_logits = generator.breakpoint_model.valid_breakpoint_logits(
        valid_breakpoints,
        lineage,
        local_env.sequence_length,
        len(state.block_boundaries) - 1,
        feature,
        state=state,
    )
    breakpoint_bias = record.breakpoint_bias(0.25)
    biased_breakpoint_logits = generator.breakpoint_model.valid_breakpoint_logits(
        valid_breakpoints,
        lineage,
        local_env.sequence_length,
        len(state.block_boundaries) - 1,
        feature,
        state=state,
        logit_bias=breakpoint_bias,
    )
    assert torch.allclose(
        biased_breakpoint_logits,
        base_breakpoint_logits + breakpoint_bias,
    )
    with pytest.raises(ValueError, match="align"):
        generator.breakpoint_model.valid_breakpoint_logits(
            valid_breakpoints,
            lineage,
            local_env.sequence_length,
            len(state.block_boundaries) - 1,
            feature,
            state=state,
            logit_bias=torch.zeros(len(valid_breakpoints) + 1),
        )
    with pytest.raises(ValueError, match="finite"):
        generator.breakpoint_model.valid_breakpoint_logits(
            valid_breakpoints,
            lineage,
            local_env.sequence_length,
            len(state.block_boundaries) - 1,
            feature,
            state=state,
            logit_bias=torch.full((len(valid_breakpoints),), float("nan")),
        )

    rollout = local_env.prepare_state_rollout_inputs([state])["rollout"][0]
    delta_time = local_env.time_env.quantile_to_delta(
        0.5,
        rollout["total_rate"],
        max_delta=rollout["max_delta"],
    )
    recorded_action = replace(
        action,
        breakpoint=int(valid_breakpoints[0]),
        time_quantile=0.5,
        delta_time=float(delta_time),
    )
    scored = generator.score_local_transitions(
        [state],
        [recorded_action],
        random_spec={"T": 0.7},
    )
    assert torch.isfinite(scored["total"]).all()
    diagnostics = scored["policy_diagnostics"][0]
    assert diagnostics["recombination_split_bias_enabled"]
    assert diagnostics["recombination_split_mass_absolute_error"] < 1e-6
    assert "recombination_split_selected_lineage_score" in diagnostics
    assert "recombination_split_selected_breakpoint_score" in diagnostics

    with torch.no_grad():
        generator.arg_model.local_transition_gate.weight.zero_()
        generator.arg_model.local_transition_gate.bias.copy_(
            torch.tensor([30.0, -30.0])
        )
    worker = RolloutWorker(local_env)
    outputs, _ = worker.rollout(
        generator,
        episodes=4,
        random_spec={"T": 0.7},
        start_states=[state] * 4,
        max_steps=1,
    )
    rescored = generator.score_local_transitions(
        [path[0] for path in outputs["trajectory_states"]],
        [actions[0] for actions in outputs["trajectory_actions"]],
        random_spec={"T": 0.7},
    )
    for trajectory_index, component_row in enumerate(
        outputs["trajectory_log_components"]
    ):
        assert component_row[0]
        for name in ("gate", "atomic_action", "breakpoint", "time", "total"):
            assert float(rescored[name][trajectory_index].detach()) == pytest.approx(
                component_row[0][name],
                abs=1e-5,
            )
    assert all(
        rows[0]["recombination_split_bias_enabled"]
        for rows in outputs["trajectory_policy_diagnostics"]
    )


def test_disabled_recombination_features_are_bit_exact_in_full_policy(
    prepared_contexts,
):
    local_env = LocalARGEnvironment(
        _base_env(recombination_rate=2e-8),
        {"region_000001": prepared_contexts["generated"]},
    )
    common_kwargs = {
        "embedding_size": 16,
        "hidden_size": 32,
        "transformer_depth": 1,
        "transformer_heads": 1,
        "breakpoint_hidden_dim": 16,
    }
    absent = TBGFlowNetGenerator(
        local_env,
        init_z_sample_count=0,
        device="cpu",
        verbose=False,
        initialize_z_from_prior=False,
        loss_mode="fl_subtb",
        model_kwargs=common_kwargs,
    )
    explicit_disabled = TBGFlowNetGenerator(
        local_env,
        init_z_sample_count=0,
        device="cpu",
        verbose=False,
        initialize_z_from_prior=False,
        loss_mode="fl_subtb",
        model_kwargs={
            **common_kwargs,
            "recombination_split_bias": {"enabled": False},
            "local_cwr_event_gate": {"enabled": False},
        },
    )
    explicit_disabled.load_state_dict(absent.state_dict())
    assert tuple(absent.state_dict()) == tuple(explicit_disabled.state_dict())

    state = local_env.get_initial_state("region_000001")
    input_dict = local_env.prepare_state_rollout_inputs(
        [state],
        random_spec={"T": 0.7},
    )
    torch.manual_seed(2026)
    absent_output = absent(input_dict)
    absent_components = {
        name: value.clone()
        for name, value in absent.last_forward_log_components.items()
    }
    torch.manual_seed(2026)
    disabled_output = explicit_disabled(input_dict)

    assert torch.equal(absent_output[0], disabled_output[0])
    assert torch.equal(absent_output[1], disabled_output[1])
    assert [action_as_dict(value) for value in absent_output[2]] == [
        action_as_dict(value) for value in disabled_output[2]
    ]
    for name in ("gate", "atomic_action", "breakpoint", "time", "total"):
        assert torch.equal(
            absent_components[name],
            explicit_disabled.last_forward_log_components[name],
        )


def test_local_cwr_event_gate_uses_prior_mass_and_rescores_exactly(
    prepared_contexts,
):
    local_env = LocalARGEnvironment(
        _base_env(recombination_rate=2e-8),
        {"region_000001": prepared_contexts["generated"]},
    )
    state = local_env.get_initial_state("region_000001")
    candidates = [sum((list(group) for group in local_env.enumerate_actions(state)), [])]
    options = local_env.enumerate_prior_options(state)
    assert options.rates["lambda_coal"] > 0.0
    assert options.rates["lambda_recomb"] > 0.0
    model_kwargs = {
        "embedding_size": 16,
        "hidden_size": 32,
        "transformer_depth": 1,
        "transformer_heads": 1,
        "breakpoint_hidden_dim": 16,
        "local_cwr_event_gate": {
            "enabled": True,
            "max_abs_residual": 2.0,
        },
    }
    generator = TBGFlowNetGenerator(
        local_env,
        init_z_sample_count=0,
        device="cpu",
        verbose=False,
        initialize_z_from_prior=False,
        loss_mode="fl_subtb",
        model_kwargs=model_kwargs,
    )
    residual_head = generator.arg_model.local_cwr_event_residual_head
    assert torch.count_nonzero(residual_head.weight) == 0
    assert torch.count_nonzero(residual_head.bias) == 0

    lineage_reps, summary_reps, _, _ = generator._encode_states([state])
    base_logits, _ = generator.arg_model._score_candidates(
        candidates,
        lineage_reps,
        summary_reps,
        state_contexts=[state],
    )
    scoring = generator.arg_model.score_action_candidates(
        candidates,
        lineage_reps,
        summary_reps,
        state_contexts=[state],
        event_rates=[options.rates],
    )
    probabilities = torch.softmax(scoring.probability_logits, dim=1)[0]
    coal_indices = [
        index
        for index, action in enumerate(candidates[0])
        if not isinstance(action, RecombinationChoice)
    ]
    recombination_indices = [
        index
        for index, action in enumerate(candidates[0])
        if isinstance(action, RecombinationChoice)
    ]
    expected_recombination_mass = float(options.rates["lambda_recomb"]) / (
        float(options.rates["lambda_coal"])
        + float(options.rates["lambda_recomb"])
    )
    assert float(probabilities[recombination_indices].sum().detach()) == pytest.approx(
        expected_recombination_mass,
        abs=1e-6,
    )
    assert torch.allclose(
        probabilities[coal_indices] / probabilities[coal_indices].sum(),
        torch.softmax(base_logits[0, coal_indices], dim=0),
    )
    assert torch.allclose(
        probabilities[recombination_indices]
        / probabilities[recombination_indices].sum(),
        torch.softmax(base_logits[0, recombination_indices], dim=0),
    )

    loss = -torch.log_softmax(scoring.probability_logits, dim=1)[
        0,
        recombination_indices[0],
    ]
    loss.backward()
    assert residual_head.bias.grad is not None
    assert float(residual_head.bias.grad.abs().sum()) > 0.0

    with torch.no_grad():
        generator.arg_model.local_transition_gate.weight.zero_()
        generator.arg_model.local_transition_gate.bias.copy_(
            torch.tensor([30.0, -30.0])
        )
    worker = RolloutWorker(local_env)
    outputs, _ = worker.rollout(
        generator,
        episodes=4,
        random_spec={"T": 0.7},
        start_states=[state] * 4,
        max_steps=1,
    )
    rescored = generator.score_local_transitions(
        [path[0] for path in outputs["trajectory_states"]],
        [actions[0] for actions in outputs["trajectory_actions"]],
        random_spec={"T": 0.7},
    )
    assert torch.allclose(
        rescored["total"],
        outputs["log_paths_pf"][:, 0],
        rtol=1e-5,
        atol=1e-5,
    )
    assert all(
        rows[0]["local_cwr_event_gate_enabled"]
        for rows in outputs["trajectory_policy_diagnostics"]
    )
    assert all(
        rows[0]["local_cwr_selected_event"] in {"coalescence", "recombination"}
        for rows in outputs["trajectory_policy_diagnostics"]
    )


def test_enabling_local_cwr_gate_does_not_perturb_shared_initial_parameters(
    prepared_contexts,
):
    local_env = LocalARGEnvironment(
        _base_env(recombination_rate=2e-8),
        {"region_000001": prepared_contexts["generated"]},
    )
    common_kwargs = {
        "embedding_size": 16,
        "hidden_size": 32,
        "transformer_depth": 1,
        "transformer_heads": 1,
        "breakpoint_hidden_dim": 16,
    }
    torch.manual_seed(71)
    disabled = TBGFlowNetGenerator(
        local_env,
        init_z_sample_count=0,
        device="cpu",
        verbose=False,
        initialize_z_from_prior=False,
        loss_mode="fl_subtb",
        model_kwargs=common_kwargs,
    )
    torch.manual_seed(71)
    enabled = TBGFlowNetGenerator(
        local_env,
        init_z_sample_count=0,
        device="cpu",
        verbose=False,
        initialize_z_from_prior=False,
        loss_mode="fl_subtb",
        model_kwargs={
            **common_kwargs,
            "local_cwr_event_gate": {"enabled": True},
        },
    )
    disabled_state = disabled.state_dict()
    enabled_state = enabled.state_dict()
    shared_names = set(disabled_state) & set(enabled_state)
    assert shared_names
    assert all(
        torch.equal(disabled_state[name], enabled_state[name])
        for name in shared_names
    )


def test_reference_and_production_semantic_parity(prepared_contexts):
    ref_prepared = prepared_contexts["reference"]
    prod_prepared = prepared_contexts["generated"]
    assert ref_prepared.context.selected_event_indices == (
        prod_prepared.context.selected_event_indices
    )
    assert [
        tuple(vars(interval).values())
        for interval in ref_prepared.context.authorized_edge_intervals
    ] == [
        tuple(vars(interval).values())
        for interval in prod_prepared.context.authorized_edge_intervals
    ]

    base_env = _base_env(recombination_rate=0.0)
    ref_state = reference.initialize_local_arg_state(
        ref_prepared,
        base_env,
    )
    local_env = LocalARGEnvironment(
        base_env,
        {"region_000001": prod_prepared},
    )
    prod_state = local_env.get_initial_state("region_000001")
    assert _state_signature(ref_state) == _state_signature(prod_state)

    while not prod_state.is_done:
        ref_options = reference.enumerate_local_prior_actions(
            ref_state,
            ref_prepared.context,
            base_env,
        )
        prod_options = local_env.enumerate_prior_options(prod_state)
        assert ref_options.rates == pytest.approx(prod_options.rates)
        assert [
            action.as_dict() for action in ref_options.coal_actions
        ] == [
            action.as_dict() for action in prod_options.coal_actions
        ]
        action = replace(
            prod_options.coal_actions[0],
            time_quantile=0.5,
            delta_time=base_env.time_env.quantile_to_delta(
                0.5,
                prod_options.rates["lambda_coal"],
            ),
        )
        prod_log_prior = local_env.compute_cwr_event_log_prior(
            prod_state,
            local_env.enumerate_actions(prod_state),
            action,
        )
        ref_state = reference.apply_local_action(
            ref_state,
            action,
            ref_prepared.context,
            base_env,
            prod_log_prior,
        )
        previous_identity = prod_state.structural_identity()
        prod_state = local_env.apply_action(
            prod_state,
            action,
            log_prior=prod_log_prior,
        )
        assert (
            local_env.undo_transition(prod_state).structural_identity()
            == previous_identity
        )
        assert _state_signature(ref_state) == _state_signature(prod_state)

    assert prod_state.log_likelihood == pytest.approx(
        ref_state.log_likelihood,
        rel=1e-10,
        abs=1e-8,
    )
    assert prod_state.absolute_log_reward == pytest.approx(
        ref_state.log_reward,
        rel=1e-10,
        abs=1e-8,
    )
    assert prod_state.log_reward == pytest.approx(
        ref_state.log_reward
        - base_env.reward_fn.C
        - prod_state.outside_log_likelihood,
        rel=1e-10,
        abs=1e-8,
    )
    terminal_clone = prod_state.clone()
    unrevealed_node_id = max(terminal_clone.all_nodes) + 1
    terminal_clone.fixed_ancestor_schedule = [
        {
            "node_id": unrevealed_node_id,
            "time": float(terminal_clone.current_time) + 1.0,
        }
    ]
    assert local_env.is_terminal(terminal_clone)
    terminal_clone.local_terminal_requires_exhausted_fixed_schedule = True
    assert not local_env.is_terminal(terminal_clone)
    assert terminal_clone.clone().local_terminal_requires_exhausted_fixed_schedule
    terminal_clone.fixed_ancestor_schedule[0]["node_id"] = next(
        iter(terminal_clone.all_nodes)
    )
    assert local_env.is_terminal(terminal_clone)

    ref_proposal = reference.local_state_to_proposal(
        ref_state,
        ref_prepared,
    )
    prod_proposal = local_env.state_to_proposal(prod_state)
    assert prod_proposal.topology_digest == ref_proposal.topology_digest
    ref_splice = reference.splice_local_proposal(
        ref_prepared,
        ref_proposal,
    )
    prod_splice = production.splice_local_proposal(
        prod_prepared,
        prod_proposal,
    )
    assert ref_splice.is_valid
    assert prod_splice.is_valid
    assert ref_splice.refined_tree_sequence.tables.equals(
        prod_splice.refined_tree_sequence.tables,
        ignore_provenance=True,
    )


def test_source_anchored_topology_proposals_splice(prepared_contexts):
    prepared = prepared_contexts["generated"]
    base_env = _base_env(recombination_rate=2e-8)

    copy_proposal = production.build_source_copy_proposal(prepared)
    copy_splice = production.splice_local_proposal(prepared, copy_proposal)
    assert copy_splice.is_valid
    assert copy_splice.validation.counts["exterior_unchanged"]
    assert copy_splice.validation.counts["target_genotypes_preserved"]

    topology_proposal = production.build_vcf_distance_topology_proposal(
        prepared,
        base_env,
        window_bp=1_000.0,
        merge_power=2.0,
        root_time_scale=1.0,
    )
    assert topology_proposal.nodes
    assert topology_proposal.edges
    assert topology_proposal.topology_digest != copy_proposal.topology_digest
    topology_splice = production.splice_local_proposal(
        prepared,
        topology_proposal,
    )
    assert topology_splice.is_valid
    assert topology_splice.validation.counts["exterior_unchanged"]
    assert topology_splice.validation.counts["target_genotypes_preserved"]


def test_boundary_gate_masks_mixed_contexts_and_policy_gradients(
    prepared_contexts,
):
    base_env = _base_env(recombination_rate=2e-8)
    local_env = LocalARGEnvironment(
        base_env,
        {
            "boundary": prepared_contexts["boundary"],
            "generated": prepared_contexts["generated"],
        },
    )
    boundary_state = local_env.get_initial_state("boundary")
    generated_state = local_env.get_initial_state("generated")
    coal_actions, _recomb_actions = local_env.enumerate_actions(generated_state)
    coal_weights = [
        production_local._local_coalescence_weight(
            generated_state.active_lineages[action.active_lineage_i],
            generated_state.active_lineages[action.active_lineage_j],
            generated_state,
        )
        for action in coal_actions
    ]
    assert all(weight > 0.0 for weight in coal_weights)
    if len(set(coal_weights)) > 1:
        low_index = int(np.argmin(coal_weights))
        high_index = int(np.argmax(coal_weights))
        delta_time = local_env.time_quantile_to_delta(generated_state, 0.5)
        low_action = replace(
            coal_actions[low_index],
            time_quantile=0.5,
            delta_time=delta_time,
        )
        high_action = replace(
            coal_actions[high_index],
            time_quantile=0.5,
            delta_time=delta_time,
        )
        low_log_prior = local_env.compute_cwr_event_log_prior(
            generated_state,
            local_env.enumerate_actions(generated_state),
            low_action,
        )
        high_log_prior = local_env.compute_cwr_event_log_prior(
            generated_state,
            local_env.enumerate_actions(generated_state),
            high_action,
        )
        assert high_log_prior - low_log_prior == pytest.approx(
            math.log(coal_weights[high_index] / coal_weights[low_index])
        )
    boundary_decision = local_env.prepare_state_rollout_inputs(
        [boundary_state]
    )["rollout"][0]
    assert boundary_decision["can_generate"]
    assert boundary_decision["can_attach_fixed"]
    generated, survival = base_env.time_env.bounded_waiting_distribution(
        boundary_decision["total_rate"],
        boundary_decision["max_delta"],
    )
    assert generated + survival == pytest.approx(1.0)
    assert boundary_decision["generated_prior_mass"] == pytest.approx(
        generated
    )
    zero_rate_decision = local_env._rollout_time_data(
        boundary_state,
        {"lambda_coal": 0.0, "lambda_recomb": 0.0},
    )
    assert not zero_rate_decision["can_generate"]
    assert zero_rate_decision["can_attach_fixed"]
    assert zero_rate_decision["generated_prior_mass"] == 0.0

    generator = TBGFlowNetGenerator(
        local_env,
        init_z_sample_count=0,
        device="cpu",
        verbose=False,
        initialize_z_from_prior=False,
        loss_mode="fl_subtb",
        model_kwargs={
            "embedding_size": 16,
            "hidden_size": 32,
            "transformer_depth": 1,
            "transformer_heads": 1,
            "breakpoint_hidden_dim": 16,
            "time_context_mode": "full",
            "local_prior_action_logit_bias": 1.0,
            "local_prior_gate_logit_bias": 1.0,
        },
    )
    with torch.no_grad():
        _, summaries, _, _ = generator._encode_states([boundary_state])
        gate_probabilities = torch.softmax(
            generator.arg_model.compute_local_gate_logits(summaries),
            dim=-1,
        )
    assert float(gate_probabilities.sum().item()) == pytest.approx(1.0)
    recombination = local_env.enumerate_actions(boundary_state)[1][0]
    valid_breakpoints = local_env.valid_breakpoints(
        boundary_state,
        recombination,
    )
    breakpoint_logits = generator.breakpoint_model.valid_breakpoint_logits(
        valid_breakpoints,
        boundary_state.active_lineages[
            recombination.active_lineage_i
        ],
        local_env.sequence_length,
        len(boundary_state.block_boundaries) - 1,
        torch.zeros(generator.breakpoint_model.action_context_dim),
        state=boundary_state,
    )
    assert breakpoint_logits.shape == (len(valid_breakpoints),)
    assert torch.isfinite(breakpoint_logits).all()
    sampled_breakpoint, sampled_log_probability = generator.breakpoint_model(
        valid_breakpoints,
        boundary_state.active_lineages[recombination.active_lineage_i],
        local_env.sequence_length,
        len(boundary_state.block_boundaries) - 1,
        torch.zeros(generator.breakpoint_model.action_context_dim),
        state=boundary_state,
    )
    breakpoint_diagnostics = generator.breakpoint_model.last_sample_diagnostics
    assert sampled_breakpoint in valid_breakpoints
    assert torch.isfinite(sampled_log_probability)
    assert breakpoint_diagnostics["breakpoint_support_size"] == len(
        valid_breakpoints
    )
    assert 0.0 <= breakpoint_diagnostics["breakpoint_normalized_entropy"] <= 1.0
    assert 0.0 < breakpoint_diagnostics["breakpoint_selected_probability"] <= 1.0
    assert 0.0 < breakpoint_diagnostics["breakpoint_max_probability"] <= 1.0

    outputs, _ = RolloutWorker(local_env).rollout(
        generator,
        episodes=8,
        start_states=[
            boundary_state,
            generated_state,
            boundary_state,
            generated_state,
            boundary_state,
            generated_state,
            boundary_state,
            generated_state,
        ],
        max_steps=1,
    )
    assert torch.isfinite(outputs["log_paths_pf"]).all()
    assert torch.isfinite(outputs["log_paths_pb"]).all()
    structural_diagnostics = [
        row
        for trajectory in outputs["trajectory_policy_diagnostics"]
        for row in trajectory
        if row.get("selected_gate") == "generated"
    ]
    assert structural_diagnostics
    assert all(
        "structural_action_normalized_entropy" in row
        for row in structural_diagnostics
    )
    components = generator.last_forward_log_components
    assert torch.allclose(
        components["total"],
        components["gate"]
        + components["atomic_action"]
        + components["breakpoint"]
        + components["time"],
    )
    assert {
        path[-1].local_context_id
        for path in outputs["trajectory_states"]
    } == {"boundary", "generated"}
    before = generator.arg_model.local_transition_gate.weight.detach().clone()
    generator.accumulate_loss(outputs)
    assert generator.policy_grad_norm() > 0.0
    assert (
        generator.arg_model.local_transition_gate.weight.grad.norm().item()
        > 0.0
    )
    info = generator.update_model()
    assert math.isfinite(float(info["loss"]))
    assert not torch.equal(
        before,
        generator.arg_model.local_transition_gate.weight.detach(),
    )


def test_fixed_attachment_is_deterministic_and_round_trips(
    prepared_contexts,
):
    local_env = LocalARGEnvironment(
        _base_env(recombination_rate=2e-8),
        {"boundary": prepared_contexts["boundary"]},
    )
    state = local_env.get_initial_state("boundary")
    fixed_time = local_env.next_fixed_ancestor_time(state)
    assert fixed_time is not None
    log_prior = local_env.fixed_attachment_log_prior(state)
    next_state = local_env.apply_action(
        state,
        FixedAttachmentChoice(fixed_time),
        log_prior=log_prior,
    )
    record = next_state.transition_records[-1]
    assert record["event_type"] == "fixed_attachment"
    assert len(record["attachments"]) >= 1
    assert local_env.backward_parent_count(next_state) == 1
    assert local_env.backward_log_probability(next_state) == pytest.approx(0.0)
    assert (
        local_env.undo_transition(next_state).structural_identity()
        == state.structural_identity()
    )
    diagnostic_clone = state.clone()
    diagnostic_clone.transition_records.append({"diagnostic": True})
    assert diagnostic_clone.structural_identity() == state.structural_identity()


def test_replay_reconstructs_and_rescores_with_current_policy(
    prepared_contexts,
):
    local_env = LocalARGEnvironment(
        _base_env(recombination_rate=0.0),
        {"ctx": prepared_contexts["boundary"]},
    )
    generator = TBGFlowNetGenerator(
        local_env,
        init_z_sample_count=0,
        device="cpu",
        verbose=False,
        initialize_z_from_prior=False,
        loss_mode="fl_subtb",
        model_kwargs={
            "embedding_size": 8,
            "hidden_size": 16,
            "transformer_depth": 1,
            "transformer_heads": 1,
            "breakpoint_hidden_dim": 8,
            "time_hidden_size": 8,
            "time_layers": 1,
        },
    )
    fresh, _ = RolloutWorker(local_env).rollout(
        generator,
        episodes=1,
        start_states=[local_env.get_initial_state("ctx")],
        max_steps=512,
    )
    assert fresh["terminal_mask"].all()
    stale_log_pf = fresh["log_paths_pf"].detach().clone()

    buffer = HybridReplayBuffer(
        ["ctx"], capacity_per_context=2, top_fraction=1.0, seed=1
    )
    entry, _ = buffer.add(
        "ctx",
        fresh["trajectory_actions"][0],
        fresh["trajectory_states"][0][-1],
        residual_priority=1.0,
        step=0,
    )
    torch.manual_seed(99)
    with torch.no_grad():
        for parameter in generator.arg_model.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))

    replayed = reconstruct_and_rescore_entries(local_env, generator, [entry])
    length = int(replayed["trajectory_lengths"][0].item())
    assert replayed["terminal_mask"].all()
    assert replayed["trajectory_states"][0][-1].structural_identity() == (
        fresh["trajectory_states"][0][-1].structural_identity()
    )
    assert replayed["log_rewards"][0] == pytest.approx(fresh["log_rewards"][0])
    assert torch.equal(replayed["log_paths_pb"], fresh["log_paths_pb"])
    assert not torch.allclose(
        replayed["log_paths_pf"][0, :length],
        stale_log_pf[0, :length],
    )

    generator.accumulate_loss(replayed)
    assert generator.policy_grad_norm() > 0.0


def test_local_transition_rescoring_and_fixed_bank_are_deterministic(
    prepared_contexts,
):
    local_env = LocalARGEnvironment(
        _base_env(recombination_rate=0.0),
        {
            "boundary": prepared_contexts["boundary"],
            "generated": prepared_contexts["generated"],
        },
    )
    generator = TBGFlowNetGenerator(
        local_env,
        init_z_sample_count=0,
        device="cpu",
        verbose=False,
        initialize_z_from_prior=False,
        loss_mode="fl_subtb",
        model_kwargs={
            "embedding_size": 16,
            "hidden_size": 32,
            "transformer_depth": 1,
            "transformer_heads": 1,
            "breakpoint_hidden_dim": 16,
            "time_hidden_size": 16,
            "time_layers": 1,
            "recombination_split_bias": {"enabled": True},
            "local_cwr_event_gate": {"enabled": True},
        },
    )
    worker = RolloutWorker(local_env)
    initial_states = {
        context_id: local_env.get_initial_state(context_id)
        for context_id in local_env.context_ids
    }
    outputs, _ = worker.rollout(
        generator,
        episodes=2,
        random_spec={"T": 0.7},
        start_states=[initial_states["boundary"], initial_states["generated"]],
        max_steps=1,
    )
    rescored = generator.score_local_transitions(
        [path[0] for path in outputs["trajectory_states"]],
        [actions[0] for actions in outputs["trajectory_actions"]],
        random_spec={"T": 0.7},
    )
    assert torch.allclose(
        rescored["total"],
        outputs["log_paths_pf"][:, 0],
        rtol=1e-5,
        atol=1e-5,
    )
    assert torch.allclose(
        rescored["total"],
        rescored["gate"]
        + rescored["atomic_action"]
        + rescored["breakpoint"]
        + rescored["time"],
    )

    first = generate_fixed_evaluation_bank(
        worker,
        generator,
        initial_states,
        episodes=2,
        seed=41,
        source="baseline",
        max_steps=128,
    )
    second = generate_fixed_evaluation_bank(
        worker,
        generator,
        initial_states,
        episodes=2,
        seed=41,
        source="baseline",
        max_steps=128,
    )
    assert fixed_bank_signature(first) == fixed_bank_signature(second)
    metrics = evaluate_fixed_bank(generator, first)
    generator.active_subtb_lambda = 0.2
    generator.active_subtb_max_span = 1
    metrics_with_different_training_objective = evaluate_fixed_bank(generator, first)
    assert metrics_with_different_training_objective[
        "flow_eval/fixed_bank_subtb_mse"
    ] == pytest.approx(metrics["flow_eval/fixed_bank_subtb_mse"])
    for name in (
        "flow_eval/fixed_bank_tb_mse",
        "flow_eval/fixed_bank_subtb_mse",
        "flow_eval/fixed_bank_one_step_mse",
        "flow_eval/fixed_bank_terminal_mse",
    ):
        assert math.isfinite(metrics[name])
    assert metrics[
        "models/structural/fixed_bank/decision_count"
    ] > 0
    assert math.isfinite(
        metrics["models/structural/fixed_bank/selected_nll_mean"]
    )
    assert metrics["models/time/fixed_bank/decision_count"] > 0
    assert math.isfinite(
        metrics["models/time/fixed_bank/selected_log_density_mean"]
    )
    assert metrics["models/breakpoint/fixed_bank/decision_count"] == 0
    assert metrics["models/recombination_split/fixed_bank/decision_count"] > 0
    assert metrics[
        "models/recombination_split/fixed_bank/mass_absolute_error_max"
    ] <= 1e-6
    assert metrics["models/cwr_event_gate/fixed_bank/decision_count"] > 0
    assert metrics[
        "models/cwr_event_gate/fixed_bank/prior_recombination_probability_mean"
    ] == pytest.approx(0.0)
    assert metrics[
        "models/cwr_event_gate/fixed_bank/policy_recombination_probability_mean"
    ] == pytest.approx(0.0)


def test_explicit_request_validation_and_seeded_context_sampling():
    config = load_train_config()
    config["dataset_path"] = str(VCF)
    config["output_path"] = "unused"
    config["training"]["epochs"] = 1
    config["training"]["loss"] = "fl_subtb"
    config["refinement"].update(
        {
            "enabled": True,
            "arg_path": str(SOURCE_ARG),
            "requests": [
                {
                    "genomic_range": [386, 23963],
                    "cut_time": 25_000,
                },
                {
                    "id": "second",
                    "genomic_range": [1000, 5000],
                    "cut_event_index": 0,
                },
            ],
            "terminal_requires_exhausted_fixed_schedule": True,
        }
    )
    validate_train_config(config)
    assert config["refinement"]["requests"][0]["id"] == "region_000001"
    assert config["refinement"]["terminal_requires_exhausted_fixed_schedule"]

    config["refinement"]["terminal_requires_exhausted_fixed_schedule"] = "yes"
    with pytest.raises(ValueError, match="terminal_requires_exhausted"):
        validate_train_config(config)
    config["refinement"]["terminal_requires_exhausted_fixed_schedule"] = True
    config["model"]["local_prior_action_logit_bias"] = 1.0
    config["model"]["local_prior_gate_logit_bias"] = 1.0
    config["model"]["recombination_split_bias"]["enabled"] = True
    validate_train_config(config)
    assert config["model"]["recombination_split_bias"]["enabled"]

    config["model"]["local_prior_action_logit_bias"] = "strong"
    with pytest.raises(ValueError, match="local_prior_action_logit_bias"):
        validate_train_config(config)
    config["model"]["local_prior_action_logit_bias"] = 1.0

    first = SeededContextSampler(("a", "b", "c"), seed=19)
    second = SeededContextSampler(("a", "b", "c"), seed=19)
    assert first.sample(20) == second.sample(20)

    global_config = load_train_config()
    global_config["dataset_path"] = str(VCF)
    global_config["output_path"] = "unused"
    global_config["training"]["epochs"] = 1
    global_config["model"]["recombination_split_bias"]["enabled"] = True
    with pytest.raises(ValueError, match="local VCF ARG refinement"):
        validate_train_config(global_config)

    config["refinement"]["bad_region_top_k"] = 2
    with pytest.raises(ValueError, match="automatic/backtracked"):
        validate_train_config(config)


def test_global_checkpoint_weights_warm_start_local_modules(
    prepared_contexts,
):
    global_env = _base_env(recombination_rate=0.0)
    global_generator = TBGFlowNetGenerator(
        global_env,
        init_z_sample_count=0,
        device="cpu",
        verbose=False,
        initialize_z_from_prior=False,
        loss_mode="fl_subtb",
        model_kwargs={
            "embedding_size": 16,
            "hidden_size": 32,
            "transformer_depth": 1,
            "transformer_heads": 1,
            "breakpoint_hidden_dim": 16,
        },
    )
    local_env = LocalARGEnvironment(
        _base_env(recombination_rate=0.0),
        {"generated": prepared_contexts["generated"]},
    )
    local_generator = TBGFlowNetGenerator(
        local_env,
        init_z_sample_count=0,
        device="cpu",
        verbose=False,
        initialize_z_from_prior=False,
        loss_mode="fl_subtb",
        model_kwargs={
            "embedding_size": 16,
            "hidden_size": 32,
            "transformer_depth": 1,
            "transformer_heads": 1,
            "breakpoint_hidden_dim": 16,
            "local_cwr_event_gate": {"enabled": True},
        },
    )
    report = _load_shape_compatible_weights(
        local_generator,
        {"generator_state_dict": global_generator.state_dict()},
    )
    assert report["loaded_parameter_count"] > 0
    assert any(
        name.startswith("arg_model.local_")
        for name in report["initialized_parameter_names"]
    )
    assert {
        "arg_model.local_cwr_event_residual_head.weight",
        "arg_model.local_cwr_event_residual_head.bias",
    }.issubset(report["initialized_parameter_names"])
    assert not any(
        "local_cwr_event_residual_head" in name
        for name in global_generator.state_dict()
    )


def test_brief_training_reload_inference_likelihood_and_accuracy(tmp_path):
    training_dir = tmp_path / "training"
    history = train_local_refinement(
        dataset_path=str(VCF),
        output_path=str(training_dir),
        device="cpu",
        local_refinement_arg=str(SOURCE_ARG),
        requests=[
            {
                "id": "region_000001",
                "genomic_range": [386, 23963],
                "cut_time": 25_000,
            }
        ],
        batch_size=1,
        epochs_num=1,
        seed=23,
        use_wandb=False,
        recombination_rate=0.0,
        eval_episodes=0,
        grad_accum_steps=2,
        partial_segment_max_steps=2,
        embedding_size=16,
        hidden_size=32,
        breakpoint_hidden_dim=16,
        transformer_depth=1,
        transformer_heads=1,
        time_context_mode="full",
        time_policy_lr=0.01,
        recombination_split_bias={"enabled": True},
        local_cwr_event_gate={"enabled": True},
        verbose=False,
    )
    assert len(history) == 1
    assert math.isfinite(float(history[0]["loss"]))
    checkpoint = training_dir / "checkpoints/best.pt"
    assert checkpoint.exists()
    last_checkpoint = training_dir / "checkpoints/last.pt"
    assert last_checkpoint.exists()
    checkpoint_data = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint_data["metadata"]["model"]["time_context_mode"] == "full"
    assert checkpoint_data["metadata"]["model"][
        "recombination_split_bias"
    ]["enabled"]
    assert checkpoint_data["metadata"]["model"][
        "local_cwr_event_gate"
    ]["enabled"]
    assert any(
        "local_cwr_event_residual_head" in name
        for name in checkpoint_data["generator_state_dict"]
    )
    assert checkpoint_data["metadata"]["checkpoint_kind"] == "best"
    assert checkpoint_data["metadata"]["selection_metric"] == "loss"
    assert checkpoint_data["checkpoint_format_version"] == 2
    assert checkpoint_data["training_state"]["epoch_number"] == 1
    assert checkpoint_data["metadata"]["time_policy_lr"] == pytest.approx(0.01)
    optimizer_lrs = [
        group["lr"] for group in checkpoint_data["opt_state_dict"]["param_groups"]
    ]
    assert optimizer_lrs == pytest.approx([1e-3, 1e-3, 0.01])

    inference_dir = tmp_path / "inference"
    manifest = run_inference(
        checkpoint=str(checkpoint),
        output_dir=str(inference_dir),
        num_args=1,
        batch_size=1,
        seed=23,
        device="cpu",
    )
    assert manifest["num_outputs"] == 1
    trajectory = manifest["requests"][0]["trajectories"][0]
    assert trajectory["splice_validation"]["is_valid"]
    generated_actions = [
        action
        for action in trajectory["actions"]
        if action.get("time_quantile") is not None
    ]
    assert generated_actions
    assert all(
        math.isfinite(float(action["time_policy_entropy"]))
        and math.isfinite(float(action["time_effective_components"]))
        and action["time_context_diagnostics"]["time_context_mode"] == "full"
        for action in generated_actions
    )
    output_tree = Path(trajectory["output_file"])
    refined = tskit.load(str(output_tree))

    variants = load_vcf_variants(str(VCF))
    recomputed = production.compute_tree_sequence_vcf_log_likelihood(
        refined,
        variants,
        mutation_rate=2e-8,
    )
    assert recomputed == pytest.approx(
        trajectory["whole_vcf_log_likelihood"],
        rel=1e-10,
        abs=1e-8,
    )

    accuracy_args = SimpleNamespace(
        truth_trees=ARG_ROOT / "validation/trees/sim_l25kb_0.trees",
        truth_dir=None,
        truth_prefix=None,
        ne=10_000.0,
        nspl=8,
        skip=2,
        max_pairs=2,
        pair_seed=42,
        verbose=False,
    )
    accuracy_frame = dataframe_from_tree_sequences(
        accuracy_args,
        [refined],
    )
    accuracy = common_metric_values(accuracy_frame, legacy_mse=float("nan"))
    assert accuracy["n_segments"] > 0
    assert math.isfinite(float(accuracy["weighted_mse"]))
