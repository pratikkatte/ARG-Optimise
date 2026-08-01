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
    SimpleARGEnvironment,
)
from arg.rollout_worker_arg import RolloutWorker
from arg.tb_gfn import TBGFlowNetGenerator
from arg.train import load_train_config, validate_train_config
from arg.utils import load_vcf_variants
from arg.refinement.training import (
    SeededContextSampler,
    _load_shape_compatible_weights,
)
from arg.refinement.training import train_local_refinement
from arg.infer import run_inference
from arg.validation.scripts.point_accuracy_common import (
    common_metric_values,
    dataframe_from_tree_sequences,
)


ARG_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ARG = ARG_ROOT / "validation/output/tsinfer/l25kb_dated.trees"
VCF = ARG_ROOT / "validation/vcf/sim_l25kb_0.vcf"


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
    assert prod_state.log_reward == pytest.approx(
        ref_state.log_reward,
        rel=1e-10,
        abs=1e-8,
    )
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
        }
    )
    validate_train_config(config)
    assert config["refinement"]["requests"][0]["id"] == "region_000001"
    first = SeededContextSampler(("a", "b", "c"), seed=19)
    second = SeededContextSampler(("a", "b", "c"), seed=19)
    assert first.sample(20) == second.sample(20)

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
        verbose=False,
    )
    assert len(history) == 1
    assert math.isfinite(float(history[0]["loss"]))
    checkpoint = training_dir / "checkpoints/best.pt"
    assert checkpoint.exists()

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
