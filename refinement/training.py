"""Production training workflow for explicit local ARG refinement requests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import random
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

try:
    from ..env import LocalARGEnvironment, SimpleARGEnvironment
    from ..rollout_worker_arg import RolloutWorker
    from ..tb_gfn import TBGFlowNetGenerator
    from ..flow_evaluation import (
        evaluate_fixed_bank,
        generate_fixed_evaluation_bank,
        load_fixed_evaluation_bank,
        save_fixed_evaluation_bank,
    )
    from ..time_env import DEFAULT_TIME_BASIS_COMPONENTS
    from ..recombination_split_bias import (
        normalize_recombination_split_bias_config,
    )
    from ..cwr_event_gate import normalize_local_cwr_event_gate_config
    from ..utils import (
        VCF_PARSER_VERSION,
        is_vcf_path,
        load_vcf_variants,
        validate_local_refinement_span,
    )
except ImportError:  # Support the repository's script-style entry points.
    from env import LocalARGEnvironment, SimpleARGEnvironment
    from rollout_worker_arg import RolloutWorker
    from tb_gfn import TBGFlowNetGenerator
    from flow_evaluation import (
        evaluate_fixed_bank,
        generate_fixed_evaluation_bank,
        load_fixed_evaluation_bank,
        save_fixed_evaluation_bank,
    )
    from time_env import DEFAULT_TIME_BASIS_COMPONENTS
    from recombination_split_bias import (
        normalize_recombination_split_bias_config,
    )
    from cwr_event_gate import normalize_local_cwr_event_gate_config
    from utils import (
        VCF_PARSER_VERSION,
        is_vcf_path,
        load_vcf_variants,
        validate_local_refinement_span,
    )

from .local_refinement import LocalRefinementRequest, prepare_local_refinement
from .evaluation import compare_scores, replay_source_score, score_terminal_state
from .replay import (
    FractionalQuotaAllocator,
    HybridReplayBuffer,
    max_abs_subtb_residuals,
    merge_rollout_outputs,
    normalize_hybrid_replay_config,
    reconstruct_and_rescore_entries,
)


LOCAL_MODEL_VERSION = "local-arg-gflownet-continuous-time-v2"
CONTEXT_FEATURE_SCHEMA = (
    "target_left_fraction",
    "target_right_fraction",
    "target_width_fraction",
    "log1p_cut_time",
    "log1p_elapsed_time",
    "log1p_next_fixed_delta",
    "fixed_ancestor_fraction_remaining",
    "target_material_multiplicity",
)
LINEAGE_ROLE_SCHEMA = (
    "cut",
    "generated_coalescence",
    "generated_recombination",
    "fixed_source",
)
PARTIAL_START_MODES = ("initial", "terminal_prefix_mixture")


class SeededContextSampler:
    """Uniform reproducible sampler over configured context IDs."""

    def __init__(self, context_ids: Sequence[str], seed: int):
        self.context_ids = tuple(str(value) for value in context_ids)
        if not self.context_ids:
            raise ValueError("context_ids must not be empty")
        self.rng = random.Random(int(seed))

    def sample(self, count: int):
        return [
            self.rng.choice(self.context_ids)
            for _ in range(int(count))
        ]


class TerminalPrefixSampler:
    """Select reproducible on-policy partial starts from terminal paths."""

    def __init__(self, seed: int):
        self.rng = random.Random(int(seed))
        self.boundary_cursor = 0

    def sample(
        self,
        trajectory_states,
        trajectory_actions,
        count: int,
        max_steps: int,
        boundary_fraction: float,
    ):
        sources = []
        boundaries = []
        for trajectory_index, (states, actions) in enumerate(
            zip(trajectory_states, trajectory_actions)
        ):
            for state_index in range(len(actions)):
                sources.append((trajectory_index, state_index))
            for action_index, action in enumerate(actions):
                if action.get("event_type") == "fixed_attachment":
                    boundaries.append((trajectory_index, action_index))
        if not sources:
            raise ValueError("terminal trajectories contain no nonterminal prefixes")

        requested_boundary = 0
        if boundaries and float(boundary_fraction) > 0.0:
            requested_boundary = min(
                int(count),
                max(1, int(round(int(count) * float(boundary_fraction)))),
            )

        selected = []
        for _ in range(requested_boundary):
            trajectory_index, action_index = boundaries[
                self.boundary_cursor % len(boundaries)
            ]
            self.boundary_cursor += 1
            earliest = max(0, action_index - int(max_steps) + 1)
            state_index = self.rng.randint(earliest, action_index)
            selected.append((trajectory_index, state_index, True))
        for _ in range(int(count) - len(selected)):
            trajectory_index, state_index = self.rng.choice(sources)
            selected.append((trajectory_index, state_index, False))
        self.rng.shuffle(selected)

        return {
            "start_states": [
                trajectory_states[trajectory_index][state_index]
                for trajectory_index, state_index, _ in selected
            ],
            "source_trajectory_indices": [row[0] for row in selected],
            "start_steps": [row[1] for row in selected],
            "boundary_targeted": [row[2] for row in selected],
        }


def train_local_refinement(
    dataset_path,
    output_path,
    device,
    local_refinement_arg,
    requests,
    checkpoint=None,
    bp_per_blocks=1,
    batch_size=1,
    epochs_num=10,
    seed=7,
    init_z_sample_count=0,
    use_wandb=True,
    effective_population_size=10000,
    mutation_rate=2e-8,
    recombination_rate=2e-8,
    policy_lr=1e-3,
    breakpoint_policy_lr=None,
    time_policy_lr=None,
    log_z_lr=1e-3,
    lr_scheduler_config=None,
    loss_mode="fl_subtb",
    subtb_lambda=0.9,
    subtb_max_span=None,
    subtb_lambda_initial=None,
    subtb_lambda_final=None,
    subtb_max_span_schedule=None,
    terminal_loss_weight=1.0,
    residual_scale=1.0,
    grad_clip=10.0,
    breakpoint_gradient_clip_norm=None,
    time_head_gradient_clip_norm=None,
    time_head_warmup_epochs=0,
    model_diagnostics=True,
    model_diagnostics_update_norm_every=1,
    grad_accum_steps=1,
    min_terminal_trajectories_per_batch=0,
    flow_debug=False,
    flow_debug_max_records=16,
    probability_checks=False,
    fixed_eval_bank_path=None,
    fixed_eval_bank_episodes=0,
    fixed_eval_bank_seed=100007,
    fixed_eval_bank_source="baseline",
    fixed_eval_bank_create_if_missing=False,
    eval_episodes=8,
    eval_every=10,
    partial_segment_max_steps=16,
    partial_start_mode="initial",
    partial_boundary_fraction=0.5,
    trajectory_training_mode="complete",
    complete_trajectory_max_steps=None,
    hybrid_replay_config=None,
    reward_C=30000,
    embedding_size=32,
    hidden_size=64,
    dropout=0.0,
    breakpoint_hidden_dim=128,
    breakpoint_dropout=0.1,
    transformer_depth=6,
    transformer_heads=4,
    transformer_mlp_ratio=2.0,
    attention_dropout=0.0,
    time_basis_components=DEFAULT_TIME_BASIS_COMPONENTS,
    time_context_mode="baseline",
    local_coalescence_similarity_bias=0.0,
    local_prior_action_logit_bias=0.0,
    local_prior_gate_logit_bias=0.0,
    recombination_split_bias=None,
    local_cwr_event_gate=None,
    verbose=True,
    terminal_requires_exhausted_fixed_schedule=False,
    selection_margin=1e-6,
    min_valid_splice_rate=0.90,
    min_unique_topology_rate=0.25,
):
    """Train one conditional local policy across explicit interval/time contexts."""

    if str(loss_mode).lower() != "fl_subtb":
        raise ValueError("local refinement training requires loss_mode='fl_subtb'")
    if not is_vcf_path(dataset_path):
        raise ValueError("local refinement training currently requires a phased VCF")
    if not requests:
        raise ValueError("at least one explicit local refinement request is required")
    batch_size = int(batch_size)
    grad_accum_steps = int(grad_accum_steps)
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be a positive integer")
    min_terminal_trajectories_per_batch = int(
        min_terminal_trajectories_per_batch
    )
    if min_terminal_trajectories_per_batch < 0:
        raise ValueError("min_terminal_trajectories_per_batch must be nonnegative")
    required_terminal_batches = int(
        math.ceil(min_terminal_trajectories_per_batch / batch_size)
    )
    if required_terminal_batches > grad_accum_steps:
        raise ValueError(
            "min_terminal_trajectories_per_batch exceeds the complete-trajectory "
            "capacity of one optimizer step"
        )
    if int(partial_segment_max_steps) <= 0:
        raise ValueError("partial_segment_max_steps must be positive")
    partial_start_mode = str(partial_start_mode).lower()
    if partial_start_mode not in PARTIAL_START_MODES:
        raise ValueError(
            "partial_start_mode must be one of "
            + ", ".join(repr(value) for value in PARTIAL_START_MODES)
        )
    trajectory_training_mode = str(trajectory_training_mode).lower()
    if trajectory_training_mode != "complete":
        raise ValueError(
            "partial trajectory training is temporarily disabled; "
            "trajectory_training_mode must be 'complete'"
        )
    if complete_trajectory_max_steps is not None:
        complete_trajectory_max_steps = int(complete_trajectory_max_steps)
        if complete_trajectory_max_steps <= 0:
            raise ValueError("complete_trajectory_max_steps must be positive or null")
    hybrid_replay_config = normalize_hybrid_replay_config(hybrid_replay_config)
    partial_boundary_fraction = float(partial_boundary_fraction)
    if not 0.0 <= partial_boundary_fraction <= 1.0:
        raise ValueError("partial_boundary_fraction must be between 0 and 1")

    _seed_everything(int(seed))
    device = torch.device(device)
    variant_data = load_vcf_variants(dataset_path)
    base_env = SimpleARGEnvironment(
        sequence_length=int(variant_data.sequence_length),
        num_sequences=int(variant_data.num_haplotypes),
        bp_per_blocks=int(bp_per_blocks),
        variant_data=variant_data,
        device=device,
        recombination_rate=float(recombination_rate),
        population_size=float(effective_population_size),
        mutation_rate=float(mutation_rate),
        reward_C=float(reward_C),
    )

    request_records, prepared_contexts = _prepare_contexts(
        local_refinement_arg,
        requests,
    )
    local_env = LocalARGEnvironment(
        base_env,
        prepared_contexts,
        terminal_requires_exhausted_fixed_schedule=(
            terminal_requires_exhausted_fixed_schedule
        ),
    )
    initial_states = {
        context_id: local_env.get_initial_state(context_id)
        for context_id in local_env.context_ids
    }

    requested_recombination_split_bias = (
        normalize_recombination_split_bias_config(
            recombination_split_bias
        )
    )
    requested_local_cwr_event_gate = normalize_local_cwr_event_gate_config(
        local_cwr_event_gate
    )

    model_kwargs = {
        "embedding_size": int(embedding_size),
        "hidden_size": int(hidden_size),
        "dropout": float(dropout),
        "breakpoint_hidden_dim": int(breakpoint_hidden_dim),
        "breakpoint_dropout": float(breakpoint_dropout),
        "transformer_depth": int(transformer_depth),
        "transformer_heads": int(transformer_heads),
        "transformer_mlp_ratio": float(transformer_mlp_ratio),
        "attention_dropout": float(attention_dropout),
        "local_coalescence_similarity_bias": float(
            local_coalescence_similarity_bias
        ),
        "local_prior_action_logit_bias": float(
            local_prior_action_logit_bias
        ),
        "local_prior_gate_logit_bias": float(
            local_prior_gate_logit_bias
        ),
        "recombination_split_bias": requested_recombination_split_bias,
        "local_cwr_event_gate": requested_local_cwr_event_gate,
        "time_hidden_size": 256,
        "time_layers": 3,
        "time_dropout": 0.0,
        "time_basis_components": int(time_basis_components),
        "time_context_mode": str(time_context_mode).lower(),
        "breakpoint_gap_hidden_size": 256,
        "breakpoint_gap_layers": 3,
        "breakpoint_gap_dropout": 0.0,
        "breakpoint_use_position_features": True,
    }
    source_checkpoint = None
    checkpoint_data = None
    if checkpoint:
        checkpoint_data = _load_checkpoint(checkpoint, map_location="cpu")
        checkpoint_metadata = dict(checkpoint_data.get("metadata") or {})
        _validate_source_checkpoint(checkpoint_metadata, base_env)
        requested_time_context_mode = str(time_context_mode).lower()
        if checkpoint_metadata.get("model"):
            model_kwargs = dict(checkpoint_metadata["model"])
            # Old checkpoints have no biological context and remain exactly
            # loadable as baseline. A requested richer context deliberately
            # reinitializes only shape-incompatible time-head parameters via
            # the existing shape-compatible warm-start path.
            model_kwargs["time_context_mode"] = requested_time_context_mode
            # This is a non-parametric proposal feature.  The current resolved
            # YAML deliberately controls it when warm-starting any checkpoint.
            model_kwargs["recombination_split_bias"] = (
                requested_recombination_split_bias
            )
            # The current YAML also owns the trainable local event-gate
            # architecture when warm-starting. Shape-compatible loading leaves
            # its zero-initialized residual head new for legacy checkpoints.
            model_kwargs["local_cwr_event_gate"] = requested_local_cwr_event_gate
        source_checkpoint = {
            "path": os.path.abspath(checkpoint),
            "sha256": _file_sha256(checkpoint),
            "model_version": checkpoint_metadata.get("model_version"),
            "training_mode": checkpoint_metadata.get("training_mode", "global"),
            "metadata": _json_safe(checkpoint_metadata),
        }

    generator = TBGFlowNetGenerator(
        local_env,
        init_z_sample_count=0,
        device=device,
        verbose=verbose,
        policy_lr=float(policy_lr),
        breakpoint_policy_lr=(
            None
            if breakpoint_policy_lr is None
            else float(breakpoint_policy_lr)
        ),
        time_policy_lr=(
            None if time_policy_lr is None else float(time_policy_lr)
        ),
        log_z_lr=float(log_z_lr),
        grad_clip=float(grad_clip),
        model_kwargs=model_kwargs,
        initialize_z_from_prior=False,
        loss_mode="fl_subtb",
        subtb_lambda=float(subtb_lambda),
        subtb_max_span=subtb_max_span,
        subtb_lambda_initial=subtb_lambda_initial,
        subtb_lambda_final=subtb_lambda_final,
        subtb_max_span_schedule=subtb_max_span_schedule,
        terminal_loss_weight=terminal_loss_weight,
        residual_scale=residual_scale,
        breakpoint_gradient_clip_norm=breakpoint_gradient_clip_norm,
        time_head_gradient_clip_norm=time_head_gradient_clip_norm,
        time_head_warmup_epochs=time_head_warmup_epochs,
        model_diagnostics=model_diagnostics,
        model_diagnostics_update_norm_every=(
            model_diagnostics_update_norm_every
        ),
        flow_debug=flow_debug,
        flow_debug_max_records=flow_debug_max_records,
        probability_checks=probability_checks,
        lr_scheduler_config=lr_scheduler_config,
        total_training_steps=int(epochs_num),
    )
    warm_start_report = None
    if checkpoint_data is not None:
        warm_start_report = _load_shape_compatible_weights(
            generator,
            checkpoint_data,
        )

    os.makedirs(output_path, exist_ok=True)
    checkpoint_dir = os.path.join(output_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
    last_checkpoint_path = os.path.join(checkpoint_dir, "last.pt")
    for request_record in request_records:
        request_dir = os.path.join(
            output_path,
            "requests",
            request_record["id"],
        )
        os.makedirs(request_dir, exist_ok=True)
        _write_json(
            os.path.join(request_dir, "context.json"),
            request_record,
        )

    metadata_base = _build_metadata(
        local_env=local_env,
        variant_data=variant_data,
        dataset_path=dataset_path,
        source_arg_path=local_refinement_arg,
        request_records=request_records,
        initial_states=initial_states,
        bp_per_blocks=bp_per_blocks,
        source_checkpoint=source_checkpoint,
        warm_start_report=warm_start_report,
        model_kwargs=model_kwargs,
        seed=seed,
        reward_C=reward_C,
        effective_population_size=effective_population_size,
        mutation_rate=mutation_rate,
        recombination_rate=recombination_rate,
        policy_lr=policy_lr,
        time_policy_lr=generator.time_policy_lr,
        log_z_lr=log_z_lr,
        subtb_lambda=subtb_lambda,
        subtb_max_span=subtb_max_span,
        grad_clip=grad_clip,
        grad_accum_steps=grad_accum_steps,
        eval_episodes=eval_episodes,
        eval_every=eval_every,
        init_z_sample_count=init_z_sample_count,
        partial_segment_max_steps=partial_segment_max_steps,
        partial_start_mode=partial_start_mode,
        partial_boundary_fraction=partial_boundary_fraction,
        terminal_requires_exhausted_fixed_schedule=(
            terminal_requires_exhausted_fixed_schedule
        ),
    )
    metadata_base["flow_training"] = _json_safe(
        {
            "terminal_loss_weight": float(terminal_loss_weight),
            "residual_scale": float(residual_scale),
            "subtb_lambda_initial": subtb_lambda_initial,
            "subtb_lambda_final": subtb_lambda_final,
            "subtb_max_span_schedule": list(subtb_max_span_schedule or ()),
            "time_head_gradient_clip_norm": time_head_gradient_clip_norm,
            "breakpoint_gradient_clip_norm": breakpoint_gradient_clip_norm,
            "time_head_warmup_epochs": int(time_head_warmup_epochs),
            "breakpoint_policy_lr": float(generator.breakpoint_policy_lr),
            "time_policy_lr": float(generator.time_policy_lr),
            "parameter_groups": {
                "structural": {
                    "base_lr": float(generator.arg_model_lr),
                    "gradient_clip_norm": float(generator.grad_clip),
                },
                "breakpoint": {
                    "base_lr": float(generator.breakpoint_policy_lr),
                    "gradient_clip_norm": float(
                        generator.grad_clip
                        if generator.breakpoint_gradient_clip_norm is None
                        else generator.breakpoint_gradient_clip_norm
                    ),
                },
                "time": {
                    "base_lr": float(generator.time_policy_lr),
                    "gradient_clip_norm": float(
                        generator.grad_clip
                        if generator.time_head_gradient_clip_norm is None
                        else generator.time_head_gradient_clip_norm
                    ),
                },
            },
            "model_diagnostics": bool(model_diagnostics),
            "model_diagnostics_update_norm_every": int(
                model_diagnostics_update_norm_every
            ),
            "min_terminal_trajectories_per_batch": int(
                min_terminal_trajectories_per_batch
            ),
            "trajectory_training_mode": trajectory_training_mode,
            "complete_trajectory_max_steps": complete_trajectory_max_steps,
            "hybrid_replay": dict(hybrid_replay_config),
            "flow_debug": bool(flow_debug),
            "probability_checks": bool(probability_checks),
            "lr_scheduler": generator.lr_scheduler_metadata(),
        }
    )
    _write_json(
        os.path.join(output_path, "refinement_context_manifest.json"),
        metadata_base,
    )

    worker = RolloutWorker(local_env, verbose=verbose)
    fixed_eval_bank = None
    if fixed_eval_bank_path:
        if os.path.exists(fixed_eval_bank_path):
            fixed_eval_bank = load_fixed_evaluation_bank(fixed_eval_bank_path)
        elif fixed_eval_bank_create_if_missing:
            if int(fixed_eval_bank_episodes) <= 0:
                raise ValueError(
                    "fixed_eval_bank_episodes must be positive when creating a bank"
                )
            fixed_eval_bank = generate_fixed_evaluation_bank(
                worker,
                generator,
                initial_states,
                episodes=int(fixed_eval_bank_episodes),
                seed=int(fixed_eval_bank_seed),
                source=str(fixed_eval_bank_source),
                max_steps=complete_trajectory_max_steps,
            )
            save_fixed_evaluation_bank(fixed_eval_bank, fixed_eval_bank_path)
        else:
            raise FileNotFoundError(
                f"fixed evaluation bank does not exist: {fixed_eval_bank_path}"
            )
    sampler = SeededContextSampler(local_env.context_ids, int(seed))
    history = []
    best_loss = float("inf")
    best_metric_name = None
    best_selection_rank = None
    best_selection_eligible = False
    source_scores = {}
    source_score_diagnostics = {}
    for context_id in local_env.context_ids:
        try:
            source_scores[context_id] = replay_source_score(local_env, context_id)
        except Exception as error:
            diagnostic = f"{type(error).__name__}: {error}"
            source_score_diagnostics[context_id] = diagnostic
            print(
                "source_score_unavailable\t"
                f"context={context_id}\terror={diagnostic}",
                flush=True,
            )
    metadata_base["source_scores"] = {
        "available_context_ids": sorted(source_scores),
        "diagnostics": dict(source_score_diagnostics),
    }
    wandb_run = None
    if use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed but training.wandb is true")
        wandb_run = wandb.init()
        wandb_run.define_metric("lr/optimizer_step")
        wandb_run.define_metric(
            "models/*",
            step_metric="lr/optimizer_step",
        )
        wandb_run.define_metric(
            "lr/*",
            step_metric="lr/optimizer_step",
        )
        wandb.config.update(_json_safe(metadata_base))
        wandb_run.summary["source_score_available_context_count"] = len(
            source_scores
        )
        wandb_run.summary["source_score_unavailable_context_count"] = len(
            source_score_diagnostics
        )

    replay_buffer = None
    replay_quota_allocator = None
    replay_mix_rng = None
    if hybrid_replay_config["enabled"]:
        replay_buffer = HybridReplayBuffer(
            local_env.context_ids,
            capacity_per_context=hybrid_replay_config["capacity_per_context"],
            top_fraction=hybrid_replay_config["priority_top_fraction"],
            seed=int(seed) + 310001,
        )
        replay_quota_allocator = FractionalQuotaAllocator(
            hybrid_replay_config["fractions"]
        )
        replay_mix_rng = random.Random(int(seed) + 310003)

    try:
        for epoch in range(int(epochs_num)):
            generator.set_training_epoch(epoch, total_epochs=int(epochs_num))
            sampled_context_ids = []
            rollout_metrics = []
            terminal_trajectory_count = 0
            replay_requested_counts = {
                source: 0 for source in ("fresh", "residual", "reward", "topology")
            }
            replay_actual_counts = dict(replay_requested_counts)
            replay_fallback_fresh_count = 0
            replay_selected_residual_sum = 0.0
            replay_selected_reward_sum = 0.0
            replay_selected_count = 0
            replay_excluded_entry_ids = set()
            for _accumulation in range(grad_accum_steps):
                rollout_mode = "terminal"
                if replay_buffer is None:
                    context_ids = sampler.sample(batch_size)
                    start_states = [
                        initial_states[context_id]
                        for context_id in context_ids
                    ]
                    sampled_context_ids.extend(context_ids)
                    outputs, _trajectories = worker.rollout(
                        generator,
                        episodes=batch_size,
                        start_states=start_states,
                        max_steps=complete_trajectory_max_steps,
                    )
                    row_sources = ["fresh"] * batch_size
                    row_replay_entries = [None] * batch_size
                else:
                    quota = replay_quota_allocator.allocate(batch_size)
                    requested_sources = [
                        source
                        for source in ("fresh", "residual", "reward", "topology")
                        for _ in range(quota[source])
                    ]
                    replay_mix_rng.shuffle(requested_sources)
                    requested_context_ids = sampler.sample(batch_size)
                    fresh_context_ids = []
                    selected_replay_entries = []
                    selected_replay_sources = []
                    for source, context_id in zip(
                        requested_sources,
                        requested_context_ids,
                    ):
                        replay_requested_counts[source] += 1
                        entry = None
                        if source != "fresh":
                            entry = replay_buffer.sample(
                                source,
                                context_id,
                                excluded_ids=replay_excluded_entry_ids,
                            )
                        if entry is None:
                            fresh_context_ids.append(context_id)
                            replay_actual_counts["fresh"] += 1
                            if source != "fresh":
                                replay_fallback_fresh_count += 1
                            continue
                        selected_replay_entries.append(entry)
                        selected_replay_sources.append(source)
                        replay_actual_counts[source] += 1
                        replay_excluded_entry_ids.add(entry.entry_id)

                    fresh_outputs = None
                    if fresh_context_ids:
                        fresh_outputs, _trajectories = worker.rollout(
                            generator,
                            episodes=len(fresh_context_ids),
                            start_states=[
                                initial_states[context_id]
                                for context_id in fresh_context_ids
                            ],
                            max_steps=complete_trajectory_max_steps,
                        )
                    replay_outputs = None
                    if selected_replay_entries:
                        replay_outputs = reconstruct_and_rescore_entries(
                            local_env,
                            generator,
                            selected_replay_entries,
                        )
                    outputs = merge_rollout_outputs(
                        fresh_outputs,
                        replay_outputs,
                    )
                    row_sources = (
                        ["fresh"] * len(fresh_context_ids)
                        + list(selected_replay_sources)
                    )
                    row_replay_entries = (
                        [None] * len(fresh_context_ids)
                        + list(selected_replay_entries)
                    )
                    sampled_context_ids.extend(
                        list(fresh_context_ids)
                        + [entry.context_id for entry in selected_replay_entries]
                    )
                if not bool(outputs["terminal_mask"].all().detach().cpu().item()):
                    raise RuntimeError(
                        "complete trajectory hit complete_trajectory_max_steps "
                        "before termination; no truncated path was used for training"
                    )
                terminal_trajectory_count += int(
                    outputs["terminal_mask"].detach().sum().cpu().item()
                )
                generator.accumulate_loss(
                    outputs,
                    factor=grad_accum_steps,
                )
                if replay_buffer is not None:
                    priorities = max_abs_subtb_residuals(
                        generator._last_balance_details,
                        len(row_sources),
                    )
                    for row_index, (source, entry) in enumerate(
                        zip(row_sources, row_replay_entries)
                    ):
                        terminal_state = outputs["trajectory_states"][row_index][-1]
                        if source == "fresh":
                            added_entry, _status = replay_buffer.add(
                                terminal_state.local_context_id,
                                outputs["trajectory_actions"][row_index],
                                terminal_state,
                                priorities[row_index],
                                epoch,
                            )
                            if added_entry is not None:
                                replay_excluded_entry_ids.add(added_entry.entry_id)
                        else:
                            replay_selected_residual_sum += priorities[row_index]
                            replay_selected_reward_sum += float(
                                terminal_state.log_reward
                            )
                            replay_selected_count += 1
                            replay_buffer.update_priority(
                                entry,
                                priorities[row_index],
                                terminal_state.log_reward,
                                epoch,
                            )
                rollout_metrics.append(_rollout_metrics(rollout_mode, outputs))

            if terminal_trajectory_count < min_terminal_trajectories_per_batch:
                raise RuntimeError(
                    "optimizer step did not contain the configured minimum number "
                    "of complete terminal trajectories"
                )

            info = dict(generator.update_model())
            info["epoch"] = int(epoch)
            info["sampled_context_ids"] = list(sampled_context_ids)
            info.update(_merge_rollout_metrics(rollout_metrics))
            if replay_buffer is not None:
                effective_count = max(batch_size * grad_accum_steps, 1)
                for source in ("fresh", "residual", "reward", "topology"):
                    requested = int(replay_requested_counts[source])
                    actual = int(replay_actual_counts[source])
                    info[f"replay/requested/{source}_count"] = requested
                    info[f"replay/requested/{source}_fraction"] = (
                        requested / effective_count
                    )
                    info[f"replay/actual/{source}_count"] = actual
                    info[f"replay/actual/{source}_fraction"] = (
                        actual / effective_count
                    )
                info["replay/fallback_fresh_count"] = int(
                    replay_fallback_fresh_count
                )
                info["replay/selected_count"] = int(replay_selected_count)
                info["replay/selected_residual_priority_mean"] = (
                    replay_selected_residual_sum / max(replay_selected_count, 1)
                )
                info["replay/selected_log_reward_mean"] = (
                    replay_selected_reward_sum / max(replay_selected_count, 1)
                )
                info.update(replay_buffer.metrics(epoch))
            if flow_debug and generator.last_transition_decomposition:
                decomposition_path = os.path.join(
                    output_path,
                    "flow_decomposition.jsonl",
                )
                with open(decomposition_path, "a", encoding="utf-8") as handle:
                    for record in generator.last_transition_decomposition:
                        handle.write(
                            json.dumps(
                                _json_safe({"epoch": int(epoch), **record}),
                                sort_keys=True,
                            )
                            + "\n"
                        )
                info["flow/decomposition_record_count"] = len(
                    generator.last_transition_decomposition
                )
            with torch.no_grad():
                start_flows = generator.compute_log_state_flows(
                    list(initial_states.values())
                )
            info["log_f_start_mean"] = float(
                start_flows.mean().detach().cpu().item()
            )

            should_evaluate = int(eval_episodes) > 0 and (
                epoch == 0
                or int(eval_every) <= 1
                or (epoch + 1) % int(eval_every) == 0
            )
            if should_evaluate:
                info.update(
                    evaluate_local_refinement(
                        worker,
                        generator,
                        initial_states,
                        episodes=int(eval_episodes),
                        seed=int(seed) + 100000,
                        complete_trajectory_max_steps=(
                            complete_trajectory_max_steps
                        ),
                        source_scores=source_scores,
                        selection_margin=float(selection_margin),
                    )
                )
                if fixed_eval_bank is not None:
                    info.update(evaluate_fixed_bank(generator, fixed_eval_bank))

            info.update(generator.step_lr_scheduler(info))

            history.append(info)
            if wandb_run is not None:
                wandb.log(_json_safe(info), step=epoch + 1)
            loss = float(info["loss"])
            eligible = bool(
                info.get("eval_comparable_count", 0) > 0
                and info.get("eval_valid_splice_rate", 0.0)
                >= float(min_valid_splice_rate)
                and info.get("eval_unique_topology_rate", 0.0)
                >= float(min_unique_topology_rate)
            )
            if "eval_local_loss_mean" in info:
                if eligible:
                    selection_rank = (
                        float(info.get("eval_posterior_improvement_rate", 0.0)),
                        float(info.get("eval_posterior_delta_median", -math.inf)),
                        float(info.get("eval_unique_topology_rate", 0.0)),
                        -float(info["eval_local_loss_mean"]),
                    )
                else:
                    selection_rank = (
                        float(info.get("eval_valid_splice_rate", 0.0)),
                        float(info.get("eval_unique_topology_rate", 0.0)),
                        float(info.get("eval_posterior_improvement_rate", 0.0)),
                        -float(info["eval_local_loss_mean"]),
                    )
                is_best = (
                    (eligible and not best_selection_eligible)
                    or (eligible == best_selection_eligible and (
                        best_selection_rank is None or selection_rank > best_selection_rank
                    ))
                )
                selection_metric_name = "balanced_local_evaluation"
                selection_value = selection_rank
            else:
                selection_rank = (-loss,)
                is_best = best_selection_rank is None or selection_rank > best_selection_rank
                selection_metric_name = "loss"
                selection_value = selection_rank
            if is_best:
                best_loss = float(info.get("eval_local_loss_mean", loss))
                best_metric_name = selection_metric_name
                best_selection_rank = selection_rank
                best_selection_eligible = eligible

            checkpoint_metadata = {
                **metadata_base,
                "epoch": int(epoch),
                "epoch_number": int(epoch) + 1,
                "best_loss": (
                    None if not math.isfinite(best_loss) else float(best_loss)
                ),
                "selection_metric": best_metric_name,
                "selection_value": (
                    None if best_selection_rank is None else list(best_selection_rank)
                ),
                "selection_eligible": bool(best_selection_eligible),
                "selection_margin": float(selection_margin),
                "min_valid_splice_rate": float(min_valid_splice_rate),
                "min_unique_topology_rate": float(min_unique_topology_rate),
                "source_score_diagnostics": dict(source_score_diagnostics),
                "log_f_start_mean": float(info["log_f_start_mean"]),
            }
            training_state = {
                "epoch": int(epoch),
                "epoch_number": int(epoch) + 1,
                "best_metric": best_metric_name,
                "best_metric_value": checkpoint_metadata["selection_value"],
                "rollout_index": int((epoch + 1) * grad_accum_steps),
                "optimizer_step": int(info["lr/optimizer_step"]),
            }
            if is_best:
                generator.save(
                    best_checkpoint_path,
                    metadata={
                        **checkpoint_metadata,
                        "checkpoint_kind": "best",
                    },
                    training_state=training_state,
                )
                info["best_checkpoint_path"] = best_checkpoint_path
            if epoch == int(epochs_num) - 1:
                generator.save(
                    last_checkpoint_path,
                    metadata={
                        **checkpoint_metadata,
                        "checkpoint_kind": "last",
                    },
                    training_state=training_state,
                )
                info["last_checkpoint_path"] = last_checkpoint_path

            print(
                f"Epoch {epoch + 1} local_fl_subtb_loss={loss:.4f} "
                f"logFstart={info['log_f_start_mean']:.4f}"
            )

        with open(
            os.path.join(output_path, "training_history.pkl"),
            "wb",
        ) as handle:
            pickle.dump(history, handle)
    finally:
        if wandb_run is not None:
            wandb.finish()
    return history


def evaluate_local_refinement(
    worker,
    generator,
    initial_states: Mapping[str, Any],
    episodes,
    seed,
    complete_trajectory_max_steps=None,
    source_scores=None,
    selection_margin=1e-6,
):
    if int(episodes) <= 0:
        return {}
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    env_rng_state = (
        worker.env.rng.getstate()
        if hasattr(worker.env.rng, "getstate")
        else None
    )
    was_training = generator.training
    sampler = random.Random(int(seed))
    rows = []
    source_scores = {} if source_scores is None else dict(source_scores)
    try:
        generator.eval()
        _seed_everything(int(seed))
        if hasattr(worker.env.rng, "seed"):
            worker.env.rng.seed(int(seed))
        context_ids = tuple(initial_states)
        with torch.no_grad():
            for _index in range(int(episodes)):
                mode = "terminal"
                context_id = sampler.choice(context_ids)
                outputs, _ = worker.rollout(
                    generator,
                    episodes=1,
                    start_states=[initial_states[context_id]],
                    max_steps=complete_trajectory_max_steps,
                )
                if not bool(outputs["terminal_mask"].all().detach().cpu().item()):
                    raise RuntimeError(
                        "evaluation trajectory hit complete_trajectory_max_steps "
                        "before termination; partial evaluations are disabled"
                    )
                loss = generator.compute_subtb_loss_from_rollout_outputs(
                    outputs
                )
                row = {
                        "mode": mode,
                        "loss": float(loss.detach().cpu().item()),
                        "length": int(
                            outputs["trajectory_lengths"][0]
                            .detach()
                            .cpu()
                            .item()
                        ),
                        "terminal": bool(
                            outputs["terminal_mask"][0].detach().cpu().item()
                        ),
                        "fixed_attachments": int(
                            outputs["fixed_attachment_counts"][0].item()
                        ),
                        "coalescences": int(outputs["coalescence_counts"][0].item()),
                        "recombinations": int(outputs["recombination_counts"][0].item()),
                        "first_fixed_attachment_step": next(
                            (
                                action_index + 1
                                for action_index, action in enumerate(
                                    outputs["trajectory_actions"][0]
                                )
                                if action.get("event_type") == "fixed_attachment"
                            ),
                            None,
                        ),
                    }
                if mode == "terminal" and row["terminal"]:
                    state = outputs["trajectory_states"][0][-1]
                    try:
                        candidate_score = score_terminal_state(worker.env, state)
                        row["candidate_score"] = candidate_score
                        row["splice_valid"] = bool(candidate_score.splice_valid)
                        source_score = source_scores.get(context_id)
                        if source_score is not None:
                            row["comparison"] = compare_scores(
                                source_score,
                                candidate_score,
                                margin=float(selection_margin),
                            )
                    except Exception as error:
                        row["scoring_error"] = str(error)
                        row["splice_valid"] = False
                rows.append(row)
        result = {
            "eval_local_loss_mean": float(
                np.mean([row["loss"] for row in rows])
            ),
            "eval_trajectory_length_mean": float(
                np.mean([row["length"] for row in rows])
            ),
            "eval_terminal_rate": float(
                np.mean([row["terminal"] for row in rows])
            ),
        }
        terminal_rows = [row for row in rows if row["mode"] == "terminal"]
        valid_rows = [row for row in terminal_rows if row.get("splice_valid")]
        comparable = [row["comparison"] for row in valid_rows if row.get("comparison")]
        digests = {
            row["candidate_score"].topology_digest
            for row in valid_rows if row.get("candidate_score") is not None
        }
        deltas = [float(item.posterior_delta) for item in comparable]
        result.update({
            "eval_valid_splice_rate": (
                float(len(valid_rows) / len(terminal_rows)) if terminal_rows else 0.0
            ),
            "eval_unique_topology_count": int(len(digests)),
            "eval_unique_topology_rate": (
                float(len(digests) / len(valid_rows)) if valid_rows else 0.0
            ),
            "eval_comparable_count": int(len(comparable)),
            "eval_source_score_failure_count": int(
                sum(1 for row in valid_rows if row.get("comparison") is None)
            ),
            "eval_posterior_improvement_rate": (
                float(np.mean([item.improves for item in comparable]))
                if comparable else 0.0
            ),
            "eval_posterior_delta_mean": float(np.mean(deltas)) if deltas else float("nan"),
            "eval_posterior_delta_median": float(np.median(deltas)) if deltas else float("nan"),
            "eval_posterior_delta_min": float(np.min(deltas)) if deltas else float("nan"),
            "eval_posterior_delta_max": float(np.max(deltas)) if deltas else float("nan"),
        })
        for component in ("likelihood_delta", "prior_delta"):
            values = [float(getattr(item, component)) for item in comparable]
            result[f"eval_{component}_mean"] = (
                float(np.mean(values)) if values else float("nan")
            )
        for mode in ("terminal",):
            selected = [row for row in rows if row["mode"] == mode]
            if not selected:
                continue
            result[f"eval_{mode}_trajectory_length_mean"] = float(
                np.mean([row["length"] for row in selected])
            )
            result[f"eval_{mode}_fixed_attachment_mean"] = float(
                np.mean([row["fixed_attachments"] for row in selected])
            )
            result[f"eval_{mode}_coalescence_mean"] = float(
                np.mean([row["coalescences"] for row in selected])
            )
            result[f"eval_{mode}_recombination_mean"] = float(
                np.mean([row["recombinations"] for row in selected])
            )
            first_steps = [
                row["first_fixed_attachment_step"]
                for row in selected
                if row["first_fixed_attachment_step"] is not None
            ]
            if first_steps:
                result[f"eval_{mode}_first_fixed_attachment_step_mean"] = float(
                    np.mean(first_steps)
                )
        return result
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if env_rng_state is not None:
            worker.env.rng.setstate(env_rng_state)
        generator.train(was_training)


def _prepare_contexts(source_arg_path, request_specs):
    records = []
    prepared_contexts = {}
    for index, spec in enumerate(request_specs):
        context_id = str(spec.get("id") or f"region_{index + 1:06d}")
        if (
            context_id in {".", ".."}
            or re.fullmatch(r"[A-Za-z0-9_.-]+", context_id) is None
        ):
            raise ValueError(
                "local refinement request IDs may contain only letters, "
                "numbers, '.', '_' and '-'"
            )
        if context_id in prepared_contexts:
            raise ValueError(
                f"duplicate local refinement request id {context_id!r}"
            )
        request = LocalRefinementRequest(
            genomic_range=tuple(spec["genomic_range"]),
            cut_time=spec.get("cut_time"),
            cut_event_index=spec.get("cut_event_index"),
        )
        validate_local_refinement_span(
            request.genomic_range,
            field_name=f"local refinement request {context_id!r} genomic_range",
        )
        prepared = prepare_local_refinement(source_arg_path, request)
        if not prepared.context.is_valid:
            reasons = "; ".join(
                diagnostic.message
                for diagnostic in prepared.context.rejection_diagnostics
            )
            raise ValueError(
                f"local refinement request {context_id!r} is invalid: {reasons}"
            )
        prepared_contexts[context_id] = prepared
        records.append(
            {
                "id": context_id,
                "genomic_range": [
                    float(value) for value in request.genomic_range
                ],
                "cut_time": (
                    None
                    if request.cut_time is None
                    else float(request.cut_time)
                ),
                "cut_event_index": request.cut_event_index,
                "resolved_cut": {
                    "cut_step": int(prepared.context.resolved_cut.cut_step),
                    "current_time": float(
                        prepared.context.resolved_cut.current_time
                    ),
                    "next_event_index": (
                        prepared.context.resolved_cut.next_event_index
                    ),
                    "next_event_time": (
                        prepared.context.resolved_cut.next_event_time
                    ),
                },
                "complexity": dict(prepared.context.complexity),
            }
        )
    return records, prepared_contexts


def _build_metadata(
    *,
    local_env,
    variant_data,
    dataset_path,
    source_arg_path,
    request_records,
    initial_states,
    bp_per_blocks,
    source_checkpoint,
    warm_start_report,
    model_kwargs,
    seed,
    reward_C,
    effective_population_size,
    mutation_rate,
    recombination_rate,
    policy_lr,
    time_policy_lr,
    log_z_lr,
    subtb_lambda,
    subtb_max_span,
    grad_clip,
    grad_accum_steps,
    eval_episodes,
    eval_every,
    init_z_sample_count,
    partial_segment_max_steps,
    partial_start_mode="initial",
    partial_boundary_fraction=0.5,
    terminal_requires_exhausted_fixed_schedule=False,
):
    return _json_safe(
        {
            "training_mode": "local_refinement",
            "model_version": LOCAL_MODEL_VERSION,
            "input_mode": "vcf",
            "dataset_path": os.path.abspath(dataset_path),
            "num_sequences": int(variant_data.num_haplotypes),
            "sequence_length": int(variant_data.sequence_length),
            "num_blocks": int(variant_data.num_variants),
            "num_variants": int(variant_data.num_variants),
            "bp_per_blocks": int(bp_per_blocks),
            "sample_ids": list(variant_data.sample_ids),
            "haplotype_ids": list(variant_data.haplotype_ids),
            "vcf_parser_version": VCF_PARSER_VERSION,
            "vcf_identity": {
                "path": os.path.abspath(dataset_path),
                "sha256": _file_sha256(dataset_path),
                "parser_version": VCF_PARSER_VERSION,
                "alignment_by_request": {
                    context_id: state.vcf_alignment
                    for context_id, state in initial_states.items()
                },
            },
            "region_vcf_views": {
                context_id: _initial_region_vcf_view_record(state)
                for context_id, state in initial_states.items()
            },
            "source_arg": {
                "path": os.path.abspath(source_arg_path),
                "sha256": _file_sha256(source_arg_path),
            },
            "local_refinement_arg": os.path.abspath(source_arg_path),
            "refinement_requests": list(request_records),
            "refinement_contexts": list(request_records),
            "context_feature_schema": list(CONTEXT_FEATURE_SCHEMA),
            "lineage_role_schema": list(LINEAGE_ROLE_SCHEMA),
            "source_checkpoint": source_checkpoint,
            "warm_start_report": warm_start_report,
            "model": dict(model_kwargs),
            "seed": int(seed),
            "reward_C": float(reward_C),
            "rho": float(local_env.rho),
            "effective_population_size": float(effective_population_size),
            "mutation_rate": float(mutation_rate),
            "recombination_rate": float(recombination_rate),
            "policy_lr": float(policy_lr),
            "time_policy_lr": (
                None if time_policy_lr is None else float(time_policy_lr)
            ),
            "log_z_lr": float(log_z_lr),
            "loss": "fl_subtb",
            "subtb_lambda": float(subtb_lambda),
            "subtb_max_span": (
                None if subtb_max_span is None else int(subtb_max_span)
            ),
            "grad_clip": float(grad_clip),
            "grad_accum_steps": int(grad_accum_steps),
            "eval_episodes": int(eval_episodes),
            "eval_every": int(eval_every),
            "init_z_sample_count": int(init_z_sample_count),
            "partial_segment_max_steps": int(partial_segment_max_steps),
            "partial_start_mode": str(partial_start_mode),
            "partial_boundary_fraction": float(partial_boundary_fraction),
            "terminal_requires_exhausted_fixed_schedule": bool(
                terminal_requires_exhausted_fixed_schedule
            ),
            "time": {
                **dict(local_env.time_metadata),
                "time_basis_components": int(
                    model_kwargs["time_basis_components"]
                ),
            },
            **dict(local_env.time_metadata),
            "time_basis_components": int(
                model_kwargs["time_basis_components"]
            ),
            "prior": {
                "effective_population_size": float(
                    effective_population_size
                ),
                "mutation_rate": float(mutation_rate),
                "recombination_rate": float(recombination_rate),
                **dict(local_env.time_metadata),
            },
        }
    )


def _load_shape_compatible_weights(generator, checkpoint):
    source_state = checkpoint.get("generator_state_dict", checkpoint)
    target_state = generator.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state
        and tuple(value.shape) == tuple(target_state[key].shape)
    }
    result = generator.load_state_dict(compatible, strict=False)
    return {
        "loaded_parameter_count": len(compatible),
        "initialized_parameter_names": sorted(result.missing_keys),
        "skipped_source_parameter_names": sorted(
            key for key in source_state if key not in compatible
        ),
    }


def _validate_source_checkpoint(metadata, env):
    model_version = metadata.get("model_version")
    if model_version == LOCAL_MODEL_VERSION:
        if metadata.get("training_mode") != "local_refinement":
            raise ValueError(
                "local checkpoint warm-start requires training_mode='local_refinement'"
            )
        mismatches = []
        for key, expected in (
            ("num_sequences", env.num_sequences),
            ("sequence_length", env.sequence_length),
            ("num_blocks", env.num_blocks),
        ):
            if key in metadata and int(metadata[key]) != int(expected):
                mismatches.append(
                    f"{key}: checkpoint={metadata[key]} environment={expected}"
                )
        time_mismatches = [
            f"{key}: checkpoint={metadata.get(key)!r} environment={expected!r}"
            for key, expected in env.time_metadata.items()
            if metadata.get(key) != expected
        ]
        if mismatches or time_mismatches:
            details = mismatches + time_mismatches
            raise ValueError(
                "local checkpoint is not compatible with this refinement "
                "environment: " + "; ".join(details)
            )
        return

    if model_version != "cwr-event-continuous-time-v2":
        raise ValueError(
            "local continuous-time v2 training can only warm-start from a "
            "global continuous-time v2 checkpoint or a compatible local "
            "refinement checkpoint"
        )
    time_mismatches = [
        f"{key}: checkpoint={metadata.get(key)!r} environment={expected!r}"
        for key, expected in env.time_metadata.items()
        if metadata.get(key) != expected
    ]
    if time_mismatches:
        raise ValueError(
            "source checkpoint continuous-time metadata does not match the "
            "local environment: " + "; ".join(time_mismatches)
        )
    basis_components = metadata.get("time_basis_components")
    model_basis_components = (metadata.get("model") or {}).get(
        "time_basis_components"
    )
    if (
        basis_components is None
        or model_basis_components is None
        or int(basis_components) != int(model_basis_components)
        or int(basis_components) < 2
    ):
        raise ValueError(
            "source checkpoint has incompatible continuous-time basis metadata"
        )
    mismatches = []
    for key, expected in (
        ("num_sequences", env.num_sequences),
        ("sequence_length", env.sequence_length),
        ("num_blocks", env.num_blocks),
    ):
        if key in metadata and int(metadata[key]) != int(expected):
            mismatches.append(
                f"{key}: checkpoint={metadata[key]} environment={expected}"
            )
    if mismatches:
        raise ValueError(
            "global checkpoint is not compatible with the local VCF "
            "environment: " + "; ".join(mismatches)
        )


def _rollout_metrics(mode, outputs):
    quantiles = outputs["time_quantiles"].detach().cpu()
    deltas = outputs["time_delta_times"].detach().cpu()
    densities = outputs["time_log_densities"].detach().cpu()
    entropies = outputs["time_policy_entropies"].detach().cpu()
    effective_components = outputs["time_effective_components"].detach().cpu()
    active_variant_rows = _active_variant_row_counts(outputs)
    trajectory_count = int(outputs["trajectory_lengths"].numel())
    policy_rows = [
        row
        for trajectory in outputs.get("trajectory_policy_diagnostics", ())
        for row in trajectory
    ]
    generated_policy_rows = [
        row for row in policy_rows if row.get("selected_gate") == "generated"
    ]
    breakpoint_policy_rows = [
        row
        for row in policy_rows
        if int(row.get("breakpoint_support_size", 0)) > 0
    ]
    split_policy_rows = [
        row
        for row in generated_policy_rows
        if bool(row.get("recombination_split_bias_enabled", False))
    ]
    selected_split_rows = [
        row
        for row in split_policy_rows
        if "recombination_split_selected_lineage_score" in row
    ]
    cwr_event_rows = [
        row
        for row in generated_policy_rows
        if bool(row.get("local_cwr_event_gate_enabled", False))
    ]
    first_fixed_steps = [
        next(
            (
                index + 1
                for index, action in enumerate(actions)
                if action.get("event_type") == "fixed_attachment"
            ),
            None,
        )
        for actions in outputs.get("trajectory_actions", ())
    ]
    terminal_rows_by_recombination = {}
    for index, path in enumerate(outputs.get("trajectory_states", ())):
        state = path[-1]
        if not bool(getattr(state, "is_done", False)):
            continue
        recombination_count = int(outputs["recombination_counts"][index].item())
        bucket = terminal_rows_by_recombination.setdefault(
            recombination_count,
            {"count": 0, "reward_sum": 0.0, "prior_sum": 0.0, "likelihood_sum": 0.0},
        )
        bucket["count"] += 1
        bucket["reward_sum"] += float(state.log_reward)
        bucket["prior_sum"] += float(state.accumulated_log_prior)
        bucket["likelihood_sum"] += float(
            getattr(state, "log_likelihood", 0.0) or 0.0
        )
    return {
        "mode": str(mode),
        "length_mean": float(
            outputs["trajectory_lengths"].detach().float().mean().cpu().item()
        ),
        "terminal_rate": float(
            outputs["terminal_mask"].detach().float().mean().cpu().item()
        ),
        "truncated_rate": float(
            outputs["truncated_mask"].detach().float().mean().cpu().item()
        ),
        "time_count": int(quantiles.numel()),
        "time_quantile_sum": float(quantiles.sum().item()),
        "time_delta_sum": float(deltas.sum().item()),
        "time_near_boundary_sum": float(
            (quantiles >= 0.99).sum().item()
        ),
        "time_finite_density_sum": float(
            torch.isfinite(densities).sum().item()
        ),
        "time_entropy_sum": float(
            entropies[torch.isfinite(entropies)].sum().item()
        ),
        "time_entropy_count": int(torch.isfinite(entropies).sum().item()),
        "time_effective_components_sum": float(
            effective_components[
                torch.isfinite(effective_components)
            ].sum().item()
        ),
        "time_effective_components_count": int(
            torch.isfinite(effective_components).sum().item()
        ),
        "trajectory_count": trajectory_count,
        "fixed_attachment_count": int(outputs["fixed_attachment_counts"].sum().item()),
        "coalescence_count": int(outputs["coalescence_counts"].sum().item()),
        "recombination_count": int(outputs["recombination_counts"].sum().item()),
        "start_step_sum": 0.0,
        "start_count": 0,
        "boundary_targeted_sum": 0,
        "first_fixed_step_sum": float(
            sum(value for value in first_fixed_steps if value is not None)
        ),
        "first_fixed_step_count": sum(
            value is not None for value in first_fixed_steps
        ),
        "policy_row_count": len(policy_rows),
        "generated_policy_row_count": len(generated_policy_rows),
        "coalescence_probability_mass_sum": float(
            sum(row.get("coalescence_probability_mass", 0.0) for row in generated_policy_rows)
        ),
        "recombination_probability_mass_sum": float(
            sum(row.get("recombination_probability_mass", 0.0) for row in generated_policy_rows)
        ),
        "valid_coalescence_actions_sum": int(
            sum(row.get("valid_coalescence_actions", 0) for row in generated_policy_rows)
        ),
        "valid_recombination_actions_sum": int(
            sum(row.get("valid_recombination_actions", 0) for row in generated_policy_rows)
        ),
        "selected_gate_probability_sum": float(
            sum(row.get("selected_gate_probability", 0.0) for row in policy_rows)
        ),
        "selected_atomic_action_probability_sum": float(
            sum(
                row.get("selected_atomic_action_probability", 0.0)
                for row in generated_policy_rows
            )
        ),
        "structural_action_support_size_sum": int(
            sum(
                row.get("structural_action_support_size", 0)
                for row in generated_policy_rows
            )
        ),
        "structural_action_entropy_sum": float(
            sum(
                row.get("structural_action_entropy", 0.0)
                for row in generated_policy_rows
            )
        ),
        "structural_action_normalized_entropy_sum": float(
            sum(
                row.get("structural_action_normalized_entropy", 0.0)
                for row in generated_policy_rows
            )
        ),
        "structural_action_max_probability_sum": float(
            sum(
                row.get("structural_action_max_probability", 0.0)
                for row in generated_policy_rows
            )
        ),
        "recombination_split_decision_count": len(split_policy_rows),
        "recombination_split_selected_count": len(selected_split_rows),
        "recombination_split_lineage_score_sum": float(
            sum(
                row.get("recombination_split_selected_lineage_score", 0.0)
                for row in selected_split_rows
            )
        ),
        "recombination_split_breakpoint_score_sum": float(
            sum(
                row.get("recombination_split_selected_breakpoint_score", 0.0)
                for row in selected_split_rows
            )
        ),
        "recombination_split_atomic_adjustment_sum": float(
            sum(
                row.get(
                    "recombination_split_selected_atomic_logit_adjustment",
                    0.0,
                )
                for row in selected_split_rows
            )
        ),
        "recombination_split_mass_error_sum": float(
            sum(
                row.get("recombination_split_mass_absolute_error", 0.0)
                for row in split_policy_rows
            )
        ),
        "recombination_split_mass_error_max": float(
            max(
                (
                    row.get("recombination_split_mass_absolute_error", 0.0)
                    for row in split_policy_rows
                ),
                default=0.0,
            )
        ),
        "recombination_split_score_min": float(
            min(
                (row.get("recombination_split_score_min", 0.0) for row in split_policy_rows),
                default=0.0,
            )
        ),
        "recombination_split_score_mean_sum": float(
            sum(
                row.get("recombination_split_score_mean", 0.0)
                for row in split_policy_rows
            )
        ),
        "recombination_split_score_max": float(
            max(
                (row.get("recombination_split_score_max", 0.0) for row in split_policy_rows),
                default=0.0,
            )
        ),
        "local_cwr_event_decision_count": len(cwr_event_rows),
        "local_cwr_prior_recombination_probability_sum": float(
            sum(
                row.get("local_cwr_prior_recombination_probability", 0.0)
                for row in cwr_event_rows
            )
        ),
        "local_cwr_policy_recombination_probability_sum": float(
            sum(
                row.get("local_cwr_policy_recombination_probability", 0.0)
                for row in cwr_event_rows
            )
        ),
        "local_cwr_residual_sum": float(
            sum(
                row.get("local_cwr_event_residual", 0.0)
                for row in cwr_event_rows
            )
        ),
        "local_cwr_residual_abs_sum": float(
            sum(
                abs(row.get("local_cwr_event_residual", 0.0))
                for row in cwr_event_rows
            )
        ),
        "local_cwr_residual_abs_max": float(
            max(
                (
                    abs(row.get("local_cwr_event_residual", 0.0))
                    for row in cwr_event_rows
                ),
                default=0.0,
            )
        ),
        "local_cwr_selected_recombination_count": int(
            sum(
                row.get("local_cwr_selected_event") == "recombination"
                for row in cwr_event_rows
            )
        ),
        "breakpoint_decision_count": len(breakpoint_policy_rows),
        "breakpoint_support_size_sum": int(
            sum(
                row.get("breakpoint_support_size", 0)
                for row in breakpoint_policy_rows
            )
        ),
        "breakpoint_entropy_sum": float(
            sum(
                row.get("breakpoint_entropy", 0.0)
                for row in breakpoint_policy_rows
            )
        ),
        "breakpoint_normalized_entropy_sum": float(
            sum(
                row.get("breakpoint_normalized_entropy", 0.0)
                for row in breakpoint_policy_rows
            )
        ),
        "breakpoint_selected_probability_sum": float(
            sum(
                row.get("breakpoint_selected_probability", 0.0)
                for row in breakpoint_policy_rows
            )
        ),
        "breakpoint_max_probability_sum": float(
            sum(
                row.get("breakpoint_max_probability", 0.0)
                for row in breakpoint_policy_rows
            )
        ),
        "terminal_by_recombination": terminal_rows_by_recombination,
        "active_variant_rows_mean": float(
            np.mean(active_variant_rows) if active_variant_rows else 0.0
        ),
        "active_variant_rows_max": int(
            max(active_variant_rows) if active_variant_rows else 0
        ),
    }


def _merge_rollout_metrics(rows):
    result = {}
    for mode in ("partial", "terminal"):
        selected = [row for row in rows if row["mode"] == mode]
        if not selected:
            continue
        result[f"train_{mode}_trajectory_length_mean"] = float(
            np.mean([row["length_mean"] for row in selected])
        )
        result[f"train_{mode}_terminal_rate"] = float(
            np.mean([row["terminal_rate"] for row in selected])
        )
        result[f"train_{mode}_truncated_rate"] = float(
            np.mean([row["truncated_rate"] for row in selected])
        )
        result[f"train_{mode}_active_variant_rows_mean"] = float(
            np.mean([row["active_variant_rows_mean"] for row in selected])
        )
        result[f"train_{mode}_active_variant_rows_max"] = int(
            max(row["active_variant_rows_max"] for row in selected)
        )
        time_count = max(
            sum(row["time_count"] for row in selected),
            1,
        )
        result[f"train_{mode}_time_quantile_mean"] = float(
            sum(row["time_quantile_sum"] for row in selected)
            / time_count
        )
        result[f"train_{mode}_time_delta_mean"] = float(
            sum(row["time_delta_sum"] for row in selected)
            / time_count
        )
        result[f"train_{mode}_time_near_boundary_rate"] = float(
            sum(row["time_near_boundary_sum"] for row in selected)
            / time_count
        )
        result[f"train_{mode}_time_finite_density_rate"] = float(
            sum(row["time_finite_density_sum"] for row in selected)
            / time_count
        )
        entropy_count = max(
            sum(row["time_entropy_count"] for row in selected),
            1,
        )
        result[f"train_{mode}_time_policy_entropy_mean"] = float(
            sum(row["time_entropy_sum"] for row in selected)
            / entropy_count
        )
        effective_component_count = max(
            sum(
                row.get(
                    "time_effective_components_count",
                    row["time_entropy_count"],
                )
                for row in selected
            ),
            1,
        )
        result[f"train_{mode}_time_effective_components_mean"] = float(
            sum(row["time_effective_components_sum"] for row in selected)
            / effective_component_count
        )
        time_model_prefix = f"models/time/behavior/{mode}"
        result[f"{time_model_prefix}/sample_count"] = int(
            sum(row["time_count"] for row in selected)
        )
        result[f"{time_model_prefix}/quantile_mean"] = result[
            f"train_{mode}_time_quantile_mean"
        ]
        result[f"{time_model_prefix}/near_boundary_rate"] = result[
            f"train_{mode}_time_near_boundary_rate"
        ]
        result[f"{time_model_prefix}/finite_density_rate"] = result[
            f"train_{mode}_time_finite_density_rate"
        ]
        result[f"{time_model_prefix}/entropy_mean"] = result[
            f"train_{mode}_time_policy_entropy_mean"
        ]
        result[f"{time_model_prefix}/effective_components_mean"] = result[
            f"train_{mode}_time_effective_components_mean"
        ]
        trajectory_count = max(
            sum(row["trajectory_count"] for row in selected),
            1,
        )
        result[f"train_{mode}_fixed_attachment_mean"] = float(
            sum(row["fixed_attachment_count"] for row in selected)
            / trajectory_count
        )
        result[f"train_{mode}_coalescence_mean"] = float(
            sum(row["coalescence_count"] for row in selected)
            / trajectory_count
        )
        result[f"train_{mode}_recombination_mean"] = float(
            sum(row["recombination_count"] for row in selected)
            / trajectory_count
        )
        total_coalescence = sum(row["coalescence_count"] for row in selected)
        total_recombination = sum(row["recombination_count"] for row in selected)
        result[f"train_{mode}_recombination_coalescence_ratio"] = float(
            total_recombination / max(total_coalescence, 1)
        )
        result[
            f"models/structural/behavior/{mode}/recombination_coalescence_ratio"
        ] = result[f"train_{mode}_recombination_coalescence_ratio"]
        generated_policy_count = sum(
            row.get("generated_policy_row_count", 0) for row in selected
        )
        if generated_policy_count:
            result[f"train_{mode}_coalescence_probability_mass_mean"] = float(
                sum(row.get("coalescence_probability_mass_sum", 0.0) for row in selected)
                / generated_policy_count
            )
            result[f"train_{mode}_recombination_probability_mass_mean"] = float(
                sum(row.get("recombination_probability_mass_sum", 0.0) for row in selected)
                / generated_policy_count
            )
            result[f"train_{mode}_valid_coalescence_actions_mean"] = float(
                sum(row.get("valid_coalescence_actions_sum", 0) for row in selected)
                / generated_policy_count
            )
            result[f"train_{mode}_valid_recombination_actions_mean"] = float(
                sum(row.get("valid_recombination_actions_sum", 0) for row in selected)
                / generated_policy_count
            )
            result[f"train_{mode}_selected_atomic_action_probability_mean"] = float(
                sum(
                    row.get("selected_atomic_action_probability_sum", 0.0)
                    for row in selected
                ) / generated_policy_count
            )
            structural_prefix = f"models/structural/behavior/{mode}"
            result[f"{structural_prefix}/decision_count"] = int(
                generated_policy_count
            )
            for metric_name, sum_name in (
                ("support_size_mean", "structural_action_support_size_sum"),
                ("entropy_mean", "structural_action_entropy_sum"),
                (
                    "normalized_entropy_mean",
                    "structural_action_normalized_entropy_sum",
                ),
                ("selected_probability_mean", "selected_atomic_action_probability_sum"),
                ("max_probability_mean", "structural_action_max_probability_sum"),
                ("coalescence_probability_mass_mean", "coalescence_probability_mass_sum"),
                ("recombination_probability_mass_mean", "recombination_probability_mass_sum"),
            ):
                result[f"{structural_prefix}/{metric_name}"] = float(
                    sum(row.get(sum_name, 0.0) for row in selected)
                    / generated_policy_count
                )
        else:
            result[
                f"models/structural/behavior/{mode}/decision_count"
            ] = 0
        split_count = sum(
            row.get("recombination_split_decision_count", 0)
            for row in selected
        )
        split_selected_count = sum(
            row.get("recombination_split_selected_count", 0)
            for row in selected
        )
        split_prefix = f"models/recombination_split/behavior/{mode}"
        result[f"{split_prefix}/decision_count"] = int(split_count)
        if split_count:
            result[f"{split_prefix}/mass_absolute_error_mean"] = float(
                sum(
                    row.get("recombination_split_mass_error_sum", 0.0)
                    for row in selected
                )
                / split_count
            )
            result[f"{split_prefix}/mass_absolute_error_max"] = float(
                max(
                    row.get("recombination_split_mass_error_max", 0.0)
                    for row in selected
                )
            )
            result[f"{split_prefix}/candidate_score_min"] = float(
                min(
                    row.get("recombination_split_score_min", 0.0)
                    for row in selected
                    if row.get("recombination_split_decision_count", 0)
                )
            )
            result[f"{split_prefix}/candidate_score_mean"] = float(
                sum(
                    row.get("recombination_split_score_mean_sum", 0.0)
                    for row in selected
                )
                / split_count
            )
            result[f"{split_prefix}/candidate_score_max"] = float(
                max(
                    row.get("recombination_split_score_max", 0.0)
                    for row in selected
                )
            )
        if split_selected_count:
            for metric_name, sum_name in (
                ("selected_lineage_score_mean", "recombination_split_lineage_score_sum"),
                ("selected_breakpoint_score_mean", "recombination_split_breakpoint_score_sum"),
                ("selected_atomic_logit_adjustment_mean", "recombination_split_atomic_adjustment_sum"),
            ):
                result[f"{split_prefix}/{metric_name}"] = float(
                    sum(row.get(sum_name, 0.0) for row in selected)
                    / split_selected_count
                )
        cwr_count = sum(
            row.get("local_cwr_event_decision_count", 0)
            for row in selected
        )
        cwr_prefix = f"models/cwr_event_gate/behavior/{mode}"
        result[f"{cwr_prefix}/decision_count"] = int(cwr_count)
        if cwr_count:
            for metric_name, sum_name in (
                (
                    "prior_recombination_probability_mean",
                    "local_cwr_prior_recombination_probability_sum",
                ),
                (
                    "policy_recombination_probability_mean",
                    "local_cwr_policy_recombination_probability_sum",
                ),
                ("residual_mean", "local_cwr_residual_sum"),
                ("residual_abs_mean", "local_cwr_residual_abs_sum"),
            ):
                result[f"{cwr_prefix}/{metric_name}"] = float(
                    sum(row.get(sum_name, 0.0) for row in selected)
                    / cwr_count
                )
            result[f"{cwr_prefix}/residual_abs_max"] = float(
                max(
                    row.get("local_cwr_residual_abs_max", 0.0)
                    for row in selected
                )
            )
            result[f"{cwr_prefix}/selected_recombination_rate"] = float(
                sum(
                    row.get("local_cwr_selected_recombination_count", 0)
                    for row in selected
                )
                / cwr_count
            )
        policy_count = sum(row.get("policy_row_count", 0) for row in selected)
        if policy_count:
            result[f"train_{mode}_selected_gate_probability_mean"] = float(
                sum(row.get("selected_gate_probability_sum", 0.0) for row in selected)
                / policy_count
            )
        breakpoint_count = sum(
            row.get("breakpoint_decision_count", 0) for row in selected
        )
        breakpoint_prefix = f"models/breakpoint/behavior/{mode}"
        result[f"{breakpoint_prefix}/decision_count"] = int(
            breakpoint_count
        )
        if breakpoint_count:
            for metric_name, sum_name in (
                ("support_size_mean", "breakpoint_support_size_sum"),
                ("entropy_mean", "breakpoint_entropy_sum"),
                ("normalized_entropy_mean", "breakpoint_normalized_entropy_sum"),
                ("selected_probability_mean", "breakpoint_selected_probability_sum"),
                ("max_probability_mean", "breakpoint_max_probability_sum"),
            ):
                result[f"{breakpoint_prefix}/{metric_name}"] = float(
                    sum(row.get(sum_name, 0.0) for row in selected)
                    / breakpoint_count
                )
        fixed_step_count = sum(row.get("first_fixed_step_count", 0) for row in selected)
        if fixed_step_count:
            result[f"train_{mode}_first_fixed_attachment_step_mean"] = float(
                sum(row.get("first_fixed_step_sum", 0.0) for row in selected)
                / fixed_step_count
            )
        recombination_buckets = {}
        for row in selected:
            for count, values in row.get("terminal_by_recombination", {}).items():
                bucket = recombination_buckets.setdefault(
                    int(count),
                    {"count": 0, "reward_sum": 0.0, "prior_sum": 0.0, "likelihood_sum": 0.0},
                )
                for key in bucket:
                    bucket[key] += values[key]
        for count, values in sorted(recombination_buckets.items()):
            denominator = max(values["count"], 1)
            prefix = f"train_{mode}_recombination_count_{count}"
            result[f"{prefix}_terminal_count"] = int(values["count"])
            result[f"{prefix}_reward_mean"] = float(values["reward_sum"] / denominator)
            result[f"{prefix}_prior_mean"] = float(values["prior_sum"] / denominator)
            result[f"{prefix}_likelihood_mean"] = float(
                values["likelihood_sum"] / denominator
            )
        start_count = sum(row["start_count"] for row in selected)
        if start_count:
            result[f"train_{mode}_start_step_mean"] = float(
                sum(row["start_step_sum"] for row in selected) / start_count
            )
            result[f"train_{mode}_boundary_targeted_rate"] = float(
                sum(row["boundary_targeted_sum"] for row in selected)
                / start_count
            )
    return result


def _initial_region_vcf_view_record(state):
    for record in state.transition_records:
        if record.get("event_type") == "initialization":
            return record.get("region_vcf_view")
    return None


def _active_variant_row_counts(outputs):
    rows = []
    for path in outputs.get("trajectory_states", ()):
        for state in path:
            rows.append(
                sum(
                    len(getattr(lineage, "variant_indices", ()))
                    for lineage in state.active_lineages
                )
            )
    return rows


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_checkpoint(path, map_location=None):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True)


__all__ = [
    "CONTEXT_FEATURE_SCHEMA",
    "LINEAGE_ROLE_SCHEMA",
    "LOCAL_MODEL_VERSION",
    "SeededContextSampler",
    "evaluate_local_refinement",
    "train_local_refinement",
]
