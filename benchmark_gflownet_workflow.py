"""Profile the local-refinement GFlowNet workflow end to end.

The benchmark uses the normal training components but runs for a caller-chosen
number of epochs so target-size smoke runs are reproducible on GPU nodes.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import wandb
except ImportError:  # W&B is optional when training.wandb is false.
    wandb = None

from env import LocalARGEnvironment, SimpleARGEnvironment
from refinement.training import (
    SeededContextSampler,
    _build_metadata,
    _merge_rollout_metrics,
    _prepare_contexts,
    _rollout_metrics,
    evaluate_local_refinement,
)
from refinement import replay_source_score
from infer import run_inference
from rollout_worker_arg import RolloutWorker
from tb_gfn import TBGFlowNetGenerator
from flow_evaluation import (
    evaluate_fixed_bank,
    generate_fixed_evaluation_bank,
    load_fixed_evaluation_bank,
    save_fixed_evaluation_bank,
)
from train import (
    config_to_refinement_kwargs,
    config_to_train_kwargs,
    load_train_config,
    refinement_enabled,
    save_resolved_config,
    validate_train_config,
)
from utils import load_vcf_variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/config_1mb.yaml",
        help="Training YAML to benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for resolved config, checkpoint, and report.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Benchmark epochs to run; use 1 for the acceptance smoke.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Override the config device.",
    )
    parser.add_argument("--terminal-loss-weight", type=float)
    parser.add_argument("--residual-scale", type=float)
    parser.add_argument("--subtb-lambda", type=float)
    parser.add_argument("--subtb-max-span", type=int)
    parser.add_argument("--time-policy-lr", type=float)
    parser.add_argument("--time-head-gradient-clip-norm", type=float)
    parser.add_argument("--similarity-bias", type=float)
    parser.add_argument(
        "--report-name",
        default="benchmark_report.json",
        help="JSON report filename inside the output directory.",
    )
    parser.add_argument(
        "--inference-args",
        type=int,
        default=1,
        help="Number of refined ARGs to export from the benchmark checkpoint.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=1,
        help="Batch size to use when exporting refined ARGs.",
    )
    parser.add_argument(
        "--inference-temperatures",
        default="default",
        help=(
            "Comma-separated inference temperatures for checkpoint sweeps. "
            "Use 'default' for untempered sampling, e.g. default,0.25,0.5."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help=(
            "Save an additional checkpoint every N epochs. The final epoch is "
            "always saved when this is set."
        ),
    )
    parser.add_argument(
        "--checkpoint-inference",
        action="store_true",
        help=(
            "Run inference for each scheduled checkpoint instead of only the "
            "final compatibility checkpoint."
        ),
    )
    parser.add_argument(
        "--pad-vcf-to-bytes",
        type=int,
        default=None,
        help=(
            "Write a metadata-padded copy of the configured uncompressed VCF "
            "inside the output directory and use it as the benchmark input."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic torch kernels where supported.",
    )
    parser.add_argument(
        "--wandb",
        dest="wandb",
        action="store_true",
        default=None,
        help="Enable W&B logging, overriding training.wandb in the config.",
    )
    parser.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="Disable W&B logging, overriding training.wandb in the config.",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be a positive integer")
    if args.inference_args < 0:
        raise ValueError("--inference-args must be non-negative")
    if args.inference_batch_size < 1:
        raise ValueError("--inference-batch-size must be positive")
    if args.checkpoint_every is not None and args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive when provided")
    inference_temperature_specs = _parse_inference_temperature_specs(
        args.inference_temperatures
    )

    config_path = Path(args.config).resolve()
    config = load_train_config(str(config_path))
    config["training"]["epochs"] = int(args.epochs)
    if args.wandb is not None:
        config["training"]["wandb"] = bool(args.wandb)
    if args.device is not None:
        config["device"] = args.device
    training_overrides = {
        "terminal_loss_weight": args.terminal_loss_weight,
        "residual_scale": args.residual_scale,
        "subtb_lambda": args.subtb_lambda,
        "subtb_max_span": args.subtb_max_span,
        "time_policy_lr": args.time_policy_lr,
        "time_head_gradient_clip_norm": args.time_head_gradient_clip_norm,
    }
    for name, value in training_overrides.items():
        if value is not None:
            config["training"][name] = value
    if args.similarity_bias is not None:
        config["model"]["local_coalescence_similarity_bias"] = (
            args.similarity_bias
        )
    if args.output_dir is not None:
        config["output_path"] = args.output_dir
    elif config.get("output_path"):
        config["output_path"] = str(Path(config["output_path"]) / "benchmark")

    output_dir = Path(config["output_path"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    padding_record = None
    if args.pad_vcf_to_bytes is not None:
        padded_path = output_dir / (
            f"{Path(config['dataset_path']).stem}.padded_"
            f"{int(args.pad_vcf_to_bytes)}b.vcf"
        )
        padding_record = _write_padded_vcf_copy(
            config["dataset_path"],
            padded_path,
            int(args.pad_vcf_to_bytes),
        )
        config["dataset_path"] = str(padded_path)

    validate_train_config(config)
    if not refinement_enabled(config):
        raise ValueError("benchmark_gflownet_workflow currently expects local refinement to be enabled")
    save_resolved_config(config, str(output_dir))

    train_kwargs = config_to_train_kwargs(config)
    refinement_kwargs = config_to_refinement_kwargs(config)
    seed = int(train_kwargs["seed"])
    _seed_everything(seed, deterministic=bool(args.deterministic))
    device = torch.device(train_kwargs["device"])
    profiler = WorkflowProfiler(device)

    report: dict[str, Any] = {
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "seed": seed,
        "deterministic_requested": bool(args.deterministic),
        "environment": _environment_record(device),
        "git": _git_record(),
        "config": config,
        "benchmark_input_padding": padding_record,
        "phases": [],
    }

    with profiler.phase("vcf_parsing", report):
        variant_data = load_vcf_variants(train_kwargs["dataset_path"])
    report["dataset"] = {
        "path": str(Path(train_kwargs["dataset_path"]).resolve()),
        "size_bytes": int(os.path.getsize(train_kwargs["dataset_path"])),
        "sequence_length": int(variant_data.sequence_length),
        "num_variants": int(variant_data.num_variants),
        "num_haplotypes": int(variant_data.num_haplotypes),
    }

    with profiler.phase("preprocessing", report):
        base_env = SimpleARGEnvironment(
            sequence_length=int(variant_data.sequence_length),
            num_sequences=int(variant_data.num_haplotypes),
            bp_per_blocks=int(train_kwargs["bp_per_blocks"]),
            variant_data=variant_data,
            device=device,
            recombination_rate=float(train_kwargs["recombination_rate"]),
            population_size=float(train_kwargs["effective_population_size"]),
            mutation_rate=float(train_kwargs["mutation_rate"]),
            reward_C=float(train_kwargs["reward_C"]),
            seed=seed,
        )

    with profiler.phase("local_refinement_preparation", report):
        request_records, prepared_contexts = _prepare_contexts(
            refinement_kwargs["local_refinement_arg"],
            refinement_kwargs["requests"],
        )
        local_env = LocalARGEnvironment(
            base_env,
            prepared_contexts,
            terminal_requires_exhausted_fixed_schedule=refinement_kwargs.get(
                "terminal_requires_exhausted_fixed_schedule",
                False,
            ),
        )
        initial_states = {
            context_id: local_env.get_initial_state(context_id)
            for context_id in local_env.context_ids
        }
    report["refinement_requests"] = request_records
    report["region_vcf_views"] = _region_vcf_views_from_states(initial_states)

    wandb_run = _initialize_wandb(
        enabled=bool(train_kwargs["use_wandb"]),
        config=config,
        output_dir=output_dir,
        report=report,
    )

    with profiler.phase("model_initialization", report):
        model_kwargs = _model_kwargs(train_kwargs)
        generator = TBGFlowNetGenerator(
            local_env,
            init_z_sample_count=0,
            device=device,
            verbose=bool(train_kwargs["verbose"]),
            policy_lr=float(train_kwargs["policy_lr"]),
            breakpoint_policy_lr=train_kwargs.get("breakpoint_policy_lr"),
            time_policy_lr=(
                None
                if train_kwargs.get("time_policy_lr") is None
                else float(train_kwargs["time_policy_lr"])
            ),
            log_z_lr=float(train_kwargs["log_z_lr"]),
            grad_clip=float(train_kwargs["grad_clip"]),
            model_kwargs=model_kwargs,
            initialize_z_from_prior=False,
            loss_mode="fl_subtb",
            subtb_lambda=float(train_kwargs["subtb_lambda"]),
            subtb_max_span=train_kwargs["subtb_max_span"],
            subtb_lambda_initial=train_kwargs.get("subtb_lambda_initial"),
            subtb_lambda_final=train_kwargs.get("subtb_lambda_final"),
            subtb_max_span_schedule=train_kwargs.get("subtb_max_span_schedule"),
            terminal_loss_weight=float(train_kwargs["terminal_loss_weight"]),
            residual_scale=float(train_kwargs["residual_scale"]),
            breakpoint_gradient_clip_norm=train_kwargs.get(
                "breakpoint_gradient_clip_norm"
            ),
            time_head_gradient_clip_norm=train_kwargs.get(
                "time_head_gradient_clip_norm"
            ),
            time_head_warmup_epochs=int(train_kwargs["time_head_warmup_epochs"]),
            model_diagnostics=bool(train_kwargs["model_diagnostics"]),
            model_diagnostics_update_norm_every=int(
                train_kwargs["model_diagnostics_update_norm_every"]
            ),
            flow_debug=bool(train_kwargs["flow_debug"]),
            flow_debug_max_records=int(train_kwargs["flow_debug_max_records"]),
            probability_checks=bool(train_kwargs["probability_checks"]),
            lr_scheduler_config=train_kwargs.get("lr_scheduler_config"),
            total_training_steps=int(train_kwargs["epochs_num"]),
        )
    metadata_base = _build_metadata(
        local_env=local_env,
        variant_data=variant_data,
        dataset_path=train_kwargs["dataset_path"],
        source_arg_path=refinement_kwargs["local_refinement_arg"],
        request_records=request_records,
        initial_states=initial_states,
        bp_per_blocks=train_kwargs["bp_per_blocks"],
        source_checkpoint=None,
        warm_start_report=None,
        model_kwargs=model_kwargs,
        seed=seed,
        reward_C=train_kwargs["reward_C"],
        effective_population_size=train_kwargs["effective_population_size"],
        mutation_rate=train_kwargs["mutation_rate"],
        recombination_rate=train_kwargs["recombination_rate"],
        policy_lr=train_kwargs["policy_lr"],
        time_policy_lr=generator.time_policy_lr,
        log_z_lr=train_kwargs["log_z_lr"],
        subtb_lambda=train_kwargs["subtb_lambda"],
        subtb_max_span=train_kwargs["subtb_max_span"],
        grad_clip=train_kwargs["grad_clip"],
        grad_accum_steps=train_kwargs["grad_accum_steps"],
        eval_episodes=train_kwargs["eval_episodes"],
        eval_every=train_kwargs["eval_every"],
        init_z_sample_count=train_kwargs["init_z_sample_count"],
        partial_segment_max_steps=train_kwargs["partial_segment_max_steps"],
        terminal_requires_exhausted_fixed_schedule=refinement_kwargs.get(
            "terminal_requires_exhausted_fixed_schedule",
            False,
        ),
    )
    metadata_base["flow_training"] = _json_safe(
        {
            "terminal_loss_weight": train_kwargs["terminal_loss_weight"],
            "residual_scale": train_kwargs["residual_scale"],
            "subtb_lambda_initial": train_kwargs.get("subtb_lambda_initial"),
            "subtb_lambda_final": train_kwargs.get("subtb_lambda_final"),
            "subtb_max_span_schedule": train_kwargs.get(
                "subtb_max_span_schedule"
            ),
            "time_head_gradient_clip_norm": train_kwargs.get(
                "time_head_gradient_clip_norm"
            ),
            "breakpoint_gradient_clip_norm": train_kwargs.get(
                "breakpoint_gradient_clip_norm"
            ),
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
            "time_head_warmup_epochs": train_kwargs["time_head_warmup_epochs"],
            "model_diagnostics": train_kwargs["model_diagnostics"],
            "model_diagnostics_update_norm_every": train_kwargs[
                "model_diagnostics_update_norm_every"
            ],
            "min_terminal_trajectories_per_batch": train_kwargs[
                "min_terminal_trajectories_per_batch"
            ],
            "trajectory_training_mode": train_kwargs[
                "trajectory_training_mode"
            ],
            "complete_trajectory_max_steps": train_kwargs.get(
                "complete_trajectory_max_steps"
            ),
            "flow_debug": train_kwargs["flow_debug"],
            "probability_checks": train_kwargs["probability_checks"],
            "lr_scheduler": generator.lr_scheduler_metadata(),
        }
    )
    with open(
        output_dir / "refinement_context_manifest.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(_json_safe(metadata_base), handle, indent=2, sort_keys=True)

    worker = RolloutWorker(local_env, verbose=bool(train_kwargs["verbose"]))
    fixed_eval_bank = None
    fixed_eval_bank_path = train_kwargs.get("fixed_eval_bank_path")
    if fixed_eval_bank_path:
        if os.path.exists(fixed_eval_bank_path):
            fixed_eval_bank = load_fixed_evaluation_bank(fixed_eval_bank_path)
        elif bool(train_kwargs.get("fixed_eval_bank_create_if_missing")):
            fixed_eval_bank = generate_fixed_evaluation_bank(
                worker,
                generator,
                initial_states,
                episodes=int(train_kwargs["fixed_eval_bank_episodes"]),
                seed=int(train_kwargs["fixed_eval_bank_seed"]),
                source=str(train_kwargs["fixed_eval_bank_source"]),
                max_steps=train_kwargs.get("complete_trajectory_max_steps"),
            )
            save_fixed_evaluation_bank(fixed_eval_bank, fixed_eval_bank_path)
        else:
            raise FileNotFoundError(
                f"fixed evaluation bank does not exist: {fixed_eval_bank_path}"
            )
    sampler = SeededContextSampler(local_env.context_ids, seed)
    history = []
    rollout_index = 0
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
                file=sys.stderr,
                flush=True,
            )
    source_score_status = {
        "available_context_ids": sorted(source_scores),
        "diagnostics": dict(source_score_diagnostics),
    }
    report["source_scores"] = source_score_status
    metadata_base["source_scores"] = source_score_status
    if wandb_run is not None:
        wandb_run.config.update(
            {"source_scores": _json_safe(source_score_status)},
            allow_val_change=True,
        )
        wandb_run.summary["source_score_available_context_count"] = len(
            source_scores
        )
        wandb_run.summary["source_score_unavailable_context_count"] = len(
            source_score_diagnostics
        )
    checkpoint_path = output_dir / "checkpoints" / "best.pt"
    last_checkpoint_path = output_dir / "checkpoints" / "last.pt"
    checkpoint_records: list[dict[str, Any]] = []

    with profiler.phase("training", report):
        for epoch in range(int(train_kwargs["epochs_num"])):
            generator.set_training_epoch(
                epoch,
                total_epochs=int(train_kwargs["epochs_num"]),
            )
            sampled_context_ids = []
            rollout_metrics = []
            terminal_trajectory_count = 0
            for _ in range(int(train_kwargs["grad_accum_steps"])):
                rollout_mode = "terminal"
                rollout_index += 1
                context_ids = sampler.sample(int(train_kwargs["batch_size"]))
                sampled_context_ids.extend(context_ids)
                outputs, _trajectories = worker.rollout(
                    generator,
                    episodes=int(train_kwargs["batch_size"]),
                    start_states=[
                        initial_states[context_id]
                        for context_id in context_ids
                    ],
                    max_steps=train_kwargs.get("complete_trajectory_max_steps"),
                )
                if not bool(outputs["terminal_mask"].all().detach().cpu().item()):
                    raise RuntimeError(
                        "benchmark trajectory hit complete_trajectory_max_steps "
                        "before termination; partial training is disabled"
                    )
                terminal_trajectory_count += int(
                    outputs["terminal_mask"].detach().sum().cpu().item()
                )
                generator.accumulate_loss(
                    outputs,
                    factor=int(train_kwargs["grad_accum_steps"]),
                )
                rollout_metrics.append(_rollout_metrics(rollout_mode, outputs))
            if terminal_trajectory_count < int(
                train_kwargs["min_terminal_trajectories_per_batch"]
            ):
                raise RuntimeError(
                    "benchmark optimizer step did not contain the configured "
                    "minimum number of terminal trajectories"
                )
            info = dict(generator.update_model())
            info["epoch"] = int(epoch)
            info["sampled_context_ids"] = sampled_context_ids
            info.update(_merge_rollout_metrics(rollout_metrics))
            with torch.no_grad():
                start_flows = generator.compute_log_state_flows(
                    list(initial_states.values())
                )
            info["log_f_start_mean"] = float(start_flows.mean().detach().cpu().item())
            should_evaluate = int(train_kwargs["eval_episodes"]) > 0 and (
                epoch == 0
                or int(train_kwargs["eval_every"]) <= 1
                or (epoch + 1) % int(train_kwargs["eval_every"]) == 0
            )
            if should_evaluate:
                info.update(
                    evaluate_local_refinement(
                        worker,
                        generator,
                        initial_states,
                        episodes=int(train_kwargs["eval_episodes"]),
                        seed=seed + 100000,
                        complete_trajectory_max_steps=train_kwargs.get(
                            "complete_trajectory_max_steps"
                        ),
                        source_scores=source_scores,
                        selection_margin=float(train_kwargs["selection_margin"]),
                    )
                )
                if fixed_eval_bank is not None:
                    info.update(evaluate_fixed_bank(generator, fixed_eval_bank))
            info.update(generator.step_lr_scheduler(info))
            history.append(info)
            if wandb_run is not None:
                wandb.log(_json_safe(info), step=epoch + 1)
            eligible = bool(
                info.get("eval_comparable_count", 0) > 0
                and info.get("eval_valid_splice_rate", 0.0)
                >= float(train_kwargs["min_valid_splice_rate"])
                and info.get("eval_unique_topology_rate", 0.0)
                >= float(train_kwargs["min_unique_topology_rate"])
            )
            if "eval_local_loss_mean" in info:
                selection_rank = (
                    (
                        float(info.get("eval_posterior_improvement_rate", 0.0)),
                        float(info.get("eval_posterior_delta_median", -math.inf)),
                        float(info.get("eval_unique_topology_rate", 0.0)),
                        -float(info["eval_local_loss_mean"]),
                    )
                    if eligible else
                    (
                        float(info.get("eval_valid_splice_rate", 0.0)),
                        float(info.get("eval_unique_topology_rate", 0.0)),
                        float(info.get("eval_posterior_improvement_rate", 0.0)),
                        -float(info["eval_local_loss_mean"]),
                    )
                )
                selection_metric_name = "balanced_local_evaluation"
            else:
                selection_rank = (-float(info["loss"]),)
                selection_metric_name = "loss"
            is_best = (
                (eligible and not best_selection_eligible)
                or (eligible == best_selection_eligible and (
                    best_selection_rank is None or selection_rank > best_selection_rank
                ))
            )
            if is_best:
                best_loss = float(info.get("eval_local_loss_mean", info["loss"]))
                best_metric_name = selection_metric_name
                best_selection_rank = selection_rank
                best_selection_eligible = eligible
                _save_workflow_checkpoint(
                    generator,
                    checkpoint_path,
                    metadata_base=metadata_base,
                    config_path=config_path,
                    epoch=epoch,
                    current_loss=float(info["loss"]),
                    best_loss=best_loss,
                    selection_metric=best_metric_name,
                    log_f_start_mean=info["log_f_start_mean"],
                    checkpoint_kind="best",
                    rollout_index=rollout_index,
                )
                info["best_checkpoint_path"] = str(checkpoint_path)
            epoch_number = int(epoch) + 1
            if _should_save_scheduled_checkpoint(
                epoch_number,
                int(train_kwargs["epochs_num"]),
                args.checkpoint_every,
            ):
                scheduled_path = (
                    output_dir
                    / "checkpoints"
                    / f"epoch_{epoch_number:06d}.pt"
                )
                _save_workflow_checkpoint(
                    generator,
                    scheduled_path,
                    metadata_base=metadata_base,
                    config_path=config_path,
                    epoch=epoch,
                    current_loss=float(info["loss"]),
                    best_loss=best_loss,
                    selection_metric=best_metric_name,
                    log_f_start_mean=info["log_f_start_mean"],
                    checkpoint_kind="scheduled",
                    rollout_index=rollout_index,
                )
                checkpoint_record = {
                    "epoch": int(epoch),
                    "epoch_number": int(epoch_number),
                    "path": str(scheduled_path),
                    "loss": float(info["loss"]),
                    "best_loss_so_far": float(best_loss),
                    "log_f_start_mean": float(info["log_f_start_mean"]),
                }
                checkpoint_records.append(checkpoint_record)
                print(
                    "saved_checkpoint\t"
                    f"epoch={epoch_number}\tpath={scheduled_path}",
                    flush=True,
                )

    with profiler.phase("checkpointing", report):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        _save_workflow_checkpoint(
            generator,
            last_checkpoint_path,
            metadata_base=metadata_base,
            config_path=config_path,
            epoch=int(train_kwargs["epochs_num"]) - 1,
            current_loss=float(history[-1]["loss"]),
            best_loss=best_loss,
            selection_metric=best_metric_name,
            log_f_start_mean=float(history[-1]["log_f_start_mean"]),
            checkpoint_kind="last",
            rollout_index=rollout_index,
        )
        with open(output_dir / "training_history.json", "w", encoding="utf-8") as handle:
            json.dump(_json_safe(history), handle, indent=2, sort_keys=True)

    if args.inference_args > 0:
        if args.checkpoint_inference:
            if not checkpoint_records:
                checkpoint_records = [
                    {
                        "epoch": int(train_kwargs["epochs_num"]) - 1,
                        "epoch_number": int(train_kwargs["epochs_num"]),
                        "path": str(checkpoint_path),
                        "loss": float(history[-1]["loss"]),
                        "best_loss_so_far": float(best_loss),
                        "log_f_start_mean": float(
                            history[-1]["log_f_start_mean"]
                        ),
                    }
                ]
            checkpoint_inference = _run_checkpoint_inference_sweep(
                checkpoint_records=checkpoint_records,
                output_dir=output_dir,
                num_args=int(args.inference_args),
                batch_size=int(args.inference_batch_size),
                base_seed=seed,
                device=device,
                temperature_specs=inference_temperature_specs,
                profiler=profiler,
                report=report,
            )
            report["checkpoint_inference"] = checkpoint_inference
            _write_checkpoint_inference_summaries(
                output_dir,
                checkpoint_inference,
            )
            if wandb_run is not None:
                _log_checkpoint_inference_to_wandb(checkpoint_inference)
        else:
            with profiler.phase("local_refinement_inference", report):
                inference_manifest = run_inference(
                    checkpoint=str(checkpoint_path),
                    output_dir=str(output_dir / "inference"),
                    num_args=int(args.inference_args),
                    batch_size=int(args.inference_batch_size),
                    seed=seed,
                    device=str(device),
                    temperature=None,
                )
            report["inference"] = inference_manifest
            report["inference_summary"] = _summarize_local_inference_manifest(
                inference_manifest
            )

    report["history"] = _json_safe(history)
    report["checkpoint_path"] = str(checkpoint_path)
    report["last_checkpoint_path"] = str(last_checkpoint_path)
    report["scheduled_checkpoints"] = checkpoint_records
    report["summary"] = {
        "total_seconds": float(sum(phase["seconds"] for phase in report["phases"])),
        "cpu_max_rss_gib": float(max(phase["max_rss_gib"] for phase in report["phases"])),
        "cuda_peak_allocated_gib": _max_phase_value(report["phases"], "cuda_peak_allocated_gib"),
        "cuda_peak_reserved_gib": _max_phase_value(report["phases"], "cuda_peak_reserved_gib"),
        "nvidia_smi_peak_used_mib": _max_phase_value(report["phases"], "nvidia_smi_memory_used_mib"),
        "active_variant_rows_peak": _max_history_value(
            history,
            (
                "train_partial_active_variant_rows_max",
                "train_terminal_active_variant_rows_max",
            ),
        ),
    }
    report_path = output_dir / args.report_name
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(report), handle, indent=2, sort_keys=True)

    print(f"wrote_report\t{report_path}", flush=True)
    print(f"total_seconds\t{report['summary']['total_seconds']:.3f}", flush=True)
    print(f"cpu_max_rss_gib\t{report['summary']['cpu_max_rss_gib']:.3f}", flush=True)
    if report["summary"]["cuda_peak_allocated_gib"] is not None:
        print(
            "cuda_peak_allocated_gib\t"
            f"{report['summary']['cuda_peak_allocated_gib']:.3f}",
            flush=True,
        )
        print(
            "cuda_peak_reserved_gib\t"
            f"{report['summary']['cuda_peak_reserved_gib']:.3f}",
            flush=True,
        )
    if wandb_run is not None:
        wandb.log(
            {
                f"workflow/{key}": value
                for key, value in report["summary"].items()
                if value is not None
            }
        )
        wandb.finish()


def _initialize_wandb(
    *,
    enabled: bool,
    config: dict[str, Any],
    output_dir: Path,
    report: dict[str, Any],
):
    if not enabled:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed but training.wandb is true")
    run = wandb.init(
        config=_json_safe(config),
        dir=str(output_dir),
    )
    wandb.define_metric("lr/optimizer_step")
    wandb.define_metric("models/*", step_metric="lr/optimizer_step")
    wandb.define_metric("lr/*", step_metric="lr/optimizer_step")
    wandb.define_metric("checkpoint_inference/*", step_metric="checkpoint_epoch")
    report["wandb"] = {
        "enabled": True,
        "run_id": run.id,
        "run_name": run.name,
        "run_url": run.url,
    }
    return run


def _log_checkpoint_inference_to_wandb(
    runs: list[dict[str, Any]],
) -> None:
    for run in runs:
        epoch_number = int(run["checkpoint_epoch_number"])
        temperature_label = str(run["temperature_label"])
        summaries = run.get("summary") or []
        if not summaries:
            wandb.log(
                {
                    "checkpoint_epoch": epoch_number,
                    "checkpoint_inference/error": int(run.get("error") is not None),
                }
            )
            continue
        for summary in summaries:
            request_id = str(summary["request_id"])
            prefix = (
                f"checkpoint_inference/{temperature_label}/{request_id}"
            )
            metrics = {"checkpoint_epoch": epoch_number}
            for key, value in summary.items():
                if key == "request_id" or value is None or isinstance(value, str):
                    continue
                metrics[f"{prefix}/{key}"] = value
            wandb.log(metrics)


def _parse_inference_temperature_specs(value: str) -> list[dict[str, Any]]:
    specs = []
    seen_labels = set()
    for raw_token in str(value).split(","):
        token = raw_token.strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in {"default", "none", "null"}:
            label = "default"
            temperature = None
        else:
            temperature = float(token)
            if temperature <= 0.0 or not math.isfinite(temperature):
                raise ValueError(
                    "inference temperatures must be positive finite values"
                )
            label = "temp_" + token.replace("-", "m").replace(".", "p")
        if label in seen_labels:
            raise ValueError(f"duplicate inference temperature label {label!r}")
        seen_labels.add(label)
        specs.append({"label": label, "temperature": temperature})
    if not specs:
        raise ValueError("--inference-temperatures did not define any specs")
    return specs


def _should_save_scheduled_checkpoint(
    epoch_number: int,
    total_epochs: int,
    checkpoint_every: int | None,
) -> bool:
    if checkpoint_every is None:
        return False
    return epoch_number % int(checkpoint_every) == 0 or epoch_number == total_epochs


def _save_workflow_checkpoint(
    generator: TBGFlowNetGenerator,
    path: Path,
    *,
    metadata_base: dict[str, Any],
    config_path: Path,
    epoch: int,
    current_loss: float,
    best_loss: float,
    selection_metric: str | None,
    log_f_start_mean: float,
    checkpoint_kind: str,
    rollout_index: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selection_value = (
        None if not math.isfinite(best_loss) else float(best_loss)
    )
    generator.save(
        str(path),
        metadata={
            **metadata_base,
            "config_path": str(config_path),
            "epoch": int(epoch),
            "epoch_number": int(epoch) + 1,
            "loss": float(current_loss),
            "best_loss": selection_value,
            "selection_metric": selection_metric,
            "selection_value": selection_value,
            "log_f_start_mean": float(log_f_start_mean),
            "checkpoint_kind": str(checkpoint_kind),
        },
        training_state={
            "epoch": int(epoch),
            "epoch_number": int(epoch) + 1,
            "best_metric": selection_metric,
            "best_metric_value": selection_value,
            "rollout_index": int(rollout_index),
        },
    )


def _run_checkpoint_inference_sweep(
    *,
    checkpoint_records: list[dict[str, Any]],
    output_dir: Path,
    num_args: int,
    batch_size: int,
    base_seed: int,
    device: torch.device,
    temperature_specs: list[dict[str, Any]],
    profiler: "WorkflowProfiler",
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    runs = []
    for checkpoint_index, checkpoint_record in enumerate(checkpoint_records):
        epoch_number = int(checkpoint_record["epoch_number"])
        epoch_label = f"epoch_{epoch_number:06d}"
        for temperature_index, temperature_spec in enumerate(temperature_specs):
            temperature_label = str(temperature_spec["label"])
            temperature = temperature_spec["temperature"]
            inference_seed = int(base_seed) + 1000 * epoch_number + temperature_index
            inference_dir = (
                output_dir
                / "checkpoint_inference"
                / epoch_label
                / temperature_label
            )
            phase_name = (
                "checkpoint_inference_"
                f"{epoch_label}_{temperature_label}"
            )
            run_record = {
                "checkpoint_index": int(checkpoint_index),
                "checkpoint_epoch": int(checkpoint_record["epoch"]),
                "checkpoint_epoch_number": int(epoch_number),
                "checkpoint_path": str(checkpoint_record["path"]),
                "checkpoint_loss": float(checkpoint_record["loss"]),
                "checkpoint_best_loss_so_far": float(
                    checkpoint_record["best_loss_so_far"]
                ),
                "checkpoint_log_f_start_mean": float(
                    checkpoint_record["log_f_start_mean"]
                ),
                "temperature_label": temperature_label,
                "temperature": temperature,
                "seed": int(inference_seed),
                "output_dir": str(inference_dir),
                "num_args": int(num_args),
                "batch_size": int(batch_size),
                "manifest_path": str(inference_dir / "manifest.json"),
                "error": None,
                "summary": [],
            }
            try:
                with profiler.phase(phase_name, report):
                    manifest = run_inference(
                        checkpoint=str(checkpoint_record["path"]),
                        output_dir=str(inference_dir),
                        num_args=int(num_args),
                        batch_size=int(batch_size),
                        seed=inference_seed,
                        device=str(device),
                        temperature=temperature,
                    )
                run_record["summary"] = _summarize_local_inference_manifest(
                    manifest
                )
                print(
                    "checkpoint_inference\t"
                    f"epoch={epoch_number}\ttemperature={temperature_label}\t"
                    f"outputs={manifest.get('output_count')}",
                    flush=True,
                )
            except Exception as error:
                run_record["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                print(
                    "checkpoint_inference_failed\t"
                    f"epoch={epoch_number}\ttemperature={temperature_label}\t"
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
            runs.append(run_record)
    return runs


def _summarize_local_inference_manifest(
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for request in manifest.get("requests", []):
        trajectories = list(request.get("trajectories", []))
        valid = [
            row for row in trajectories
            if row.get("output_file") is not None
        ]
        terminal = [
            row for row in trajectories
            if bool(row.get("terminal"))
        ]
        topology_digests = [
            str(row["topology_digest"])
            for row in valid
            if row.get("topology_digest") is not None
        ]
        summary = {
            "request_id": str(request.get("id")),
            "sample_count": int(request.get("sample_count", len(trajectories))),
            "trajectory_count": int(len(trajectories)),
            "terminal_count": int(len(terminal)),
            "valid_output_count": int(len(valid)),
            "unique_topology_count": int(len(set(topology_digests))),
            "topology_duplicate_rate": (
                0.0
                if not topology_digests
                else 1.0 - len(set(topology_digests)) / len(topology_digests)
            ),
        }
        for key in (
            "log_reward",
            "whole_vcf_log_likelihood",
            "local_cwr_log_prior",
            "trajectory_length",
            "log_P_F",
            "log_P_B",
        ):
            summary.update(_value_summary(key, trajectories))
        rows.append(summary)
    return rows


def _value_summary(key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            values.append(numeric)
    if not values:
        return {
            f"{key}_count": 0,
            f"{key}_mean": None,
            f"{key}_min": None,
            f"{key}_max": None,
            f"{key}_std": None,
        }
    return {
        f"{key}_count": int(len(values)),
        f"{key}_mean": float(np.mean(values)),
        f"{key}_min": float(np.min(values)),
        f"{key}_max": float(np.max(values)),
        f"{key}_std": float(np.std(values)),
    }


def _write_checkpoint_inference_summaries(
    output_dir: Path,
    runs: list[dict[str, Any]],
) -> None:
    json_path = output_dir / "checkpoint_inference_summary.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(runs), handle, indent=2, sort_keys=True)

    flat_rows = []
    for run in runs:
        base = {
            key: value
            for key, value in run.items()
            if key not in {"summary", "error"}
        }
        error = run.get("error")
        base["error_type"] = None if error is None else error.get("type")
        base["error_message"] = None if error is None else error.get("message")
        summaries = run.get("summary") or [{}]
        for summary in summaries:
            flat_rows.append({**base, **summary})

    tsv_path = output_dir / "checkpoint_inference_summary.tsv"
    fieldnames = sorted({key for row in flat_rows for key in row})
    with open(tsv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(_json_safe(flat_rows))


class WorkflowProfiler:
    def __init__(self, device: torch.device):
        self.device = device

    @contextlib.contextmanager
    def phase(self, name: str, report: dict[str, Any]):
        cuda_enabled = self.device.type == "cuda" and torch.cuda.is_available()
        if cuda_enabled:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start_rss = _current_rss_gib()
        start = time.perf_counter()
        try:
            yield
        finally:
            if cuda_enabled:
                torch.cuda.synchronize(self.device)
            seconds = time.perf_counter() - start
            end_rss = _current_rss_gib()
            row = {
                "name": name,
                "seconds": float(seconds),
                "rss_start_gib": float(start_rss),
                "rss_end_gib": float(end_rss),
                "rss_delta_gib": float(end_rss - start_rss),
                "max_rss_gib": float(_max_rss_gib()),
                "cuda_peak_allocated_gib": None,
                "cuda_peak_reserved_gib": None,
                "cuda_current_allocated_gib": None,
                "cuda_current_reserved_gib": None,
                "nvidia_smi_memory_used_mib": _nvidia_smi_memory_used_mib(),
            }
            if cuda_enabled:
                row.update(
                    {
                        "cuda_peak_allocated_gib": _bytes_to_gib(
                            torch.cuda.max_memory_allocated(self.device)
                        ),
                        "cuda_peak_reserved_gib": _bytes_to_gib(
                            torch.cuda.max_memory_reserved(self.device)
                        ),
                        "cuda_current_allocated_gib": _bytes_to_gib(
                            torch.cuda.memory_allocated(self.device)
                        ),
                        "cuda_current_reserved_gib": _bytes_to_gib(
                            torch.cuda.memory_reserved(self.device)
                        ),
                    }
                )
            report["phases"].append(row)
            print(
                f"{name}\tseconds={seconds:.3f}\t"
                f"rss_end_gib={end_rss:.3f}\tmax_rss_gib={row['max_rss_gib']:.3f}",
                flush=True,
            )


def _model_kwargs(train_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "embedding_size": int(train_kwargs["embedding_size"]),
        "hidden_size": int(train_kwargs["hidden_size"]),
        "dropout": float(train_kwargs["dropout"]),
        "breakpoint_hidden_dim": int(train_kwargs["breakpoint_hidden_dim"]),
        "breakpoint_dropout": float(train_kwargs["breakpoint_dropout"]),
        "transformer_depth": int(train_kwargs["transformer_depth"]),
        "transformer_heads": int(train_kwargs["transformer_heads"]),
        "transformer_mlp_ratio": float(train_kwargs["transformer_mlp_ratio"]),
        "attention_dropout": float(train_kwargs["attention_dropout"]),
        "local_coalescence_similarity_bias": float(
            train_kwargs.get("local_coalescence_similarity_bias", 0.0)
        ),
        "local_prior_action_logit_bias": float(
            train_kwargs.get("local_prior_action_logit_bias", 0.0)
        ),
        "local_prior_gate_logit_bias": float(
            train_kwargs.get("local_prior_gate_logit_bias", 0.0)
        ),
        "recombination_split_bias": dict(
            train_kwargs.get("recombination_split_bias") or {}
        ),
        "local_cwr_event_gate": dict(
            train_kwargs.get("local_cwr_event_gate") or {}
        ),
        "time_hidden_size": 256,
        "time_layers": 3,
        "time_dropout": 0.0,
        "time_basis_components": int(train_kwargs["time_basis_components"]),
        "time_context_mode": str(
            train_kwargs.get("time_context_mode", "baseline")
        ),
        "breakpoint_gap_hidden_size": 256,
        "breakpoint_gap_layers": 3,
        "breakpoint_gap_dropout": 0.0,
        "breakpoint_use_position_features": True,
    }


def _seed_everything(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def _environment_record(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
        "cuda_device_properties": _cuda_properties(device),
    }


def _cuda_properties(device: torch.device) -> dict[str, Any] | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(device)
    return {
        "name": props.name,
        "total_memory_gib": _bytes_to_gib(props.total_memory),
        "major": int(props.major),
        "minor": int(props.minor),
        "multi_processor_count": int(props.multi_processor_count),
    }


def _git_record() -> dict[str, Any]:
    return {
        "commit": _run_text(["git", "rev-parse", "HEAD"]),
        "status_short": _run_text(["git", "status", "--short"]),
    }


def _write_padded_vcf_copy(
    source_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    target_bytes: int,
) -> dict[str, Any]:
    if target_bytes <= 0:
        raise ValueError("--pad-vcf-to-bytes must be a positive integer")
    source = Path(source_path)
    if source.suffix == ".gz":
        raise ValueError("--pad-vcf-to-bytes only supports uncompressed .vcf inputs")
    text = source.read_text(encoding="utf-8")
    source_size = len(text.encode("utf-8"))
    if source_size > target_bytes:
        raise ValueError(
            f"{source} is already {source_size:,} bytes, larger than the "
            f"requested padded size of {target_bytes:,} bytes"
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_size == target_bytes:
        destination.write_text(text, encoding="utf-8")
        return {
            "source_path": str(source.resolve()),
            "padded_path": str(destination.resolve()),
            "source_size_bytes": int(source_size),
            "target_size_bytes": int(target_bytes),
            "padded_size_bytes": int(target_bytes),
        }

    header_offset = text.find("#CHROM")
    if header_offset < 0:
        raise ValueError("VCF header with #CHROM was not found")
    pad_prefix = "##benchmark_padding="
    pad_suffix = "\n"
    padding_bytes = (
        int(target_bytes)
        - int(source_size)
        - len(pad_prefix.encode("utf-8"))
        - len(pad_suffix.encode("utf-8"))
    )
    if padding_bytes < 0:
        raise ValueError(
            "requested padded size is too close to the source VCF size to add "
            "a valid metadata padding line"
        )
    padded = (
        text[:header_offset]
        + pad_prefix
        + ("x" * padding_bytes)
        + pad_suffix
        + text[header_offset:]
    )
    padded_size = len(padded.encode("utf-8"))
    if padded_size != int(target_bytes):
        raise RuntimeError(
            f"internal VCF padding error: wrote {padded_size:,} bytes, "
            f"expected {target_bytes:,}"
        )
    destination.write_text(padded, encoding="utf-8")
    return {
        "source_path": str(source.resolve()),
        "padded_path": str(destination.resolve()),
        "source_size_bytes": int(source_size),
        "target_size_bytes": int(target_bytes),
        "padded_size_bytes": int(padded_size),
    }


def _run_text(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _current_rss_gib() -> float:
    statm = Path("/proc/self/statm")
    if statm.exists():
        pages = int(statm.read_text(encoding="utf-8").split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1024**3
    return _max_rss_gib()


def _max_rss_gib() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / 1024**3
    return rss / 1024**2


def _nvidia_smi_memory_used_mib() -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    values = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line:
            values.append(int(float(line)))
    return max(values) if values else None


def _bytes_to_gib(value: int | float) -> float:
    return float(value) / 1024**3


def _max_phase_value(phases: list[dict[str, Any]], key: str) -> Any:
    values = [
        phase[key]
        for phase in phases
        if phase.get(key) is not None
    ]
    return max(values) if values else None


def _max_history_value(history: list[dict[str, Any]], keys: tuple[str, ...]) -> Any:
    values = []
    for row in history:
        for key in keys:
            value = row.get(key)
            if value is not None:
                values.append(value)
    return max(values) if values else None


def _region_vcf_views_from_states(initial_states: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for context_id, state in initial_states.items():
        for record in state.transition_records:
            if record.get("event_type") == "initialization":
                result[str(context_id)] = record.get("region_vcf_view")
                break
        else:
            result[str(context_id)] = None
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
