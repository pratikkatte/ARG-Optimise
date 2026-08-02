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
    from ..time_env import DEFAULT_TIME_BASIS_COMPONENTS
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
    from time_env import DEFAULT_TIME_BASIS_COMPONENTS
    from utils import (
        VCF_PARSER_VERSION,
        is_vcf_path,
        load_vcf_variants,
        validate_local_refinement_span,
    )

from .local_refinement import LocalRefinementRequest, prepare_local_refinement


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
    log_z_lr=1e-3,
    loss_mode="fl_subtb",
    subtb_lambda=0.9,
    subtb_max_span=None,
    grad_clip=10.0,
    grad_accum_steps=1,
    eval_episodes=8,
    eval_every=10,
    partial_segment_max_steps=16,
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
    verbose=True,
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
    if int(partial_segment_max_steps) <= 0:
        raise ValueError("partial_segment_max_steps must be positive")

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
    local_env = LocalARGEnvironment(base_env, prepared_contexts)
    initial_states = {
        context_id: local_env.get_initial_state(context_id)
        for context_id in local_env.context_ids
    }

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
        "time_hidden_size": 256,
        "time_layers": 3,
        "time_dropout": 0.0,
        "time_basis_components": int(time_basis_components),
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
        if checkpoint_metadata.get("model"):
            model_kwargs = dict(checkpoint_metadata["model"])
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
        log_z_lr=float(log_z_lr),
        grad_clip=float(grad_clip),
        model_kwargs=model_kwargs,
        initialize_z_from_prior=False,
        loss_mode="fl_subtb",
        subtb_lambda=float(subtb_lambda),
        subtb_max_span=subtb_max_span,
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
        log_z_lr=log_z_lr,
        subtb_lambda=subtb_lambda,
        subtb_max_span=subtb_max_span,
        grad_clip=grad_clip,
        grad_accum_steps=grad_accum_steps,
        eval_episodes=eval_episodes,
        eval_every=eval_every,
        init_z_sample_count=init_z_sample_count,
        partial_segment_max_steps=partial_segment_max_steps,
    )
    _write_json(
        os.path.join(output_path, "refinement_context_manifest.json"),
        metadata_base,
    )

    worker = RolloutWorker(local_env, verbose=verbose)
    sampler = SeededContextSampler(local_env.context_ids, int(seed))
    rollout_index = 0
    history = []
    best_loss = float("inf")
    wandb_run = None
    if use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed but training.wandb is true")
        wandb_run = wandb.init()
        wandb.config.update(_json_safe(metadata_base))

    try:
        for epoch in range(int(epochs_num)):
            sampled_context_ids = []
            rollout_metrics = []
            for _accumulation in range(grad_accum_steps):
                rollout_mode = (
                    "partial" if rollout_index % 2 == 0 else "terminal"
                )
                rollout_index += 1
                context_ids = sampler.sample(batch_size)
                sampled_context_ids.extend(context_ids)
                start_states = [
                    initial_states[context_id]
                    for context_id in context_ids
                ]
                outputs, _trajectories = worker.rollout(
                    generator,
                    episodes=batch_size,
                    start_states=start_states,
                    max_steps=(
                        int(partial_segment_max_steps)
                        if rollout_mode == "partial"
                        else None
                    ),
                )
                generator.accumulate_loss(
                    outputs,
                    factor=grad_accum_steps,
                )
                rollout_metrics.append(
                    _rollout_metrics(rollout_mode, outputs)
                )

            info = dict(generator.update_model())
            info["epoch"] = int(epoch)
            info["sampled_context_ids"] = list(sampled_context_ids)
            info.update(_merge_rollout_metrics(rollout_metrics))
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
                        seed=int(seed) + 100000 + epoch,
                        partial_segment_max_steps=int(
                            partial_segment_max_steps
                        ),
                    )
                )

            history.append(info)
            if wandb_run is not None:
                wandb.log(_json_safe(info), step=epoch + 1)
            loss = float(info["loss"])
            if math.isfinite(loss) and loss < best_loss:
                best_loss = loss
                checkpoint_metadata = {
                    **metadata_base,
                    "epoch": int(epoch),
                    "best_loss": float(best_loss),
                    "log_f_start_mean": float(info["log_f_start_mean"]),
                }
                generator.save(
                    best_checkpoint_path,
                    metadata=checkpoint_metadata,
                )
                info["best_checkpoint_path"] = best_checkpoint_path

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
    partial_segment_max_steps=16,
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
    sampler = random.Random(int(seed))
    rows = []
    try:
        _seed_everything(int(seed))
        if hasattr(worker.env.rng, "seed"):
            worker.env.rng.seed(int(seed))
        context_ids = tuple(initial_states)
        with torch.no_grad():
            for index in range(int(episodes)):
                mode = "partial" if index % 2 == 0 else "terminal"
                context_id = sampler.choice(context_ids)
                outputs, _ = worker.rollout(
                    generator,
                    episodes=1,
                    start_states=[initial_states[context_id]],
                    max_steps=(
                        int(partial_segment_max_steps)
                        if mode == "partial"
                        else None
                    ),
                )
                loss = generator.compute_subtb_loss_from_rollout_outputs(
                    outputs
                )
                rows.append(
                    {
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
                    }
                )
        return {
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
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if env_rng_state is not None:
            worker.env.rng.setstate(env_rng_state)


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
    log_z_lr,
    subtb_lambda,
    subtb_max_span,
    grad_clip,
    grad_accum_steps,
    eval_episodes,
    eval_every,
    init_z_sample_count,
    partial_segment_max_steps,
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
    if metadata.get("model_version") != "cwr-event-continuous-time-v2":
        raise ValueError(
            "local continuous-time v2 training can only warm-start from a "
            "global continuous-time v2 checkpoint"
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
    active_variant_rows = _active_variant_row_counts(outputs)
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
        "fixed_attachment_count": int(outputs["fixed_attachment_count"]),
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
        result[f"train_{mode}_fixed_attachment_mean"] = float(
            np.mean([row["fixed_attachment_count"] for row in selected])
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
            weights_only=False,
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
