"""Construction and training loop for the ARG GFlowNet."""

import math
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from arg_environment import SimpleARGEnvironment
from .evaluation import evaluate_generator
from rollout_worker_arg import RolloutWorker
from gflownet import TBGFlowNetGenerator
from .config import MODEL_VERSION, TrainConfig
from utils import load_sequences

try:
    import wandb
except ImportError:
    wandb = None


_WANDB_EVAL_METRICS = {
    "eval_tb_mse": "eval/tb_mse",
    "eval_residual_mean": "eval/residual_mean",
    "eval_residual_std": "eval/residual_std",
    "eval_trajectory_length_mean": "eval/trajectory_length_mean",
    "eval_coalescence_count_mean": "eval/coalescence_count_mean",
    "eval_recombination_count_mean": "eval/recombination_count_mean",
    "eval_initial_learned_coalescence_prob": "eval/initial_event/learned_coalescence_prob",
    "eval_initial_learned_recombination_prob": "eval/initial_event/learned_recombination_prob",
    "eval_initial_mixed_coalescence_prob": "eval/initial_event/mixed_coalescence_prob",
    "eval_initial_mixed_recombination_prob": "eval/initial_event/mixed_recombination_prob",
    "eval_initial_cwr_coalescence_prob": "eval/initial_event/cwr_coalescence_prob",
    "eval_initial_cwr_recombination_prob": "eval/initial_event/cwr_recombination_prob",
    "eval_importance_ess": "eval/importance_ess",
    "eval_importance_ess_fraction": "eval/importance_ess_fraction",
    "eval_importance_max_weight": "eval/importance_max_weight",
    "eval_importance_log_weight_range": "eval/importance_log_weight_range",
    "eval_residual_rmse": "eval/residual_rmse",
}

CONVERGENCE_VERSION = 1
CONVERGENCE_SEED_OFFSETS = (200_000, 300_000, 400_000)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    rollout_worker,
    generator,
    batch_size=1,
    grad_accum_steps=1,
    epoch=None,
    total_epochs=None,
):
    accumulation_batches = max(int(grad_accum_steps), 1)
    for batch_index in range(accumulation_batches):
        progress_label = None
        if epoch is not None:
            epoch_label = f"Epoch {epoch + 1}"
            if total_epochs is not None:
                epoch_label += f"/{total_epochs}"
            progress_label = (
                f"{epoch_label} | batch {batch_index + 1}/{accumulation_batches}"
            )
        outputs, _ = rollout_worker.rollout(
            generator, episodes=batch_size, progress_label=progress_label,
        )
        generator.accumulate_loss(outputs, factor=grad_accum_steps)
    return generator.update_model()


def train(config):
    """Train from a :class:`TrainConfig` or YAML path."""
    if not isinstance(config, TrainConfig):
        config = TrainConfig.load(config)
    seed_everything(config.runtime.seed)
    sequences = load_sequences(config.data.dataset_path)
    env = _build_environment(config, sequences)
    generator = _build_generator(config, env)
    rollout_worker = RolloutWorker(env, verbose=config.runtime.verbose)

    output_dir = Path(config.data.output_path)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    candidate_path = checkpoint_dir / "best_candidate.pt"
    last_path = checkpoint_dir / "last.pt"
    run = _start_logger(config, env)
    history = []
    convergence_history = []
    best_candidate_rmse = float("inf")
    best_converged_rmse = float("inf")
    candidate_checkpoint_written = False
    converged_checkpoint_written = False
    consecutive_passes = 0
    panel_index = 0
    last_info = None

    try:
        for epoch in range(config.training.epochs):
            info = train_epoch(
                rollout_worker, generator,
                config.training.batch_size, config.training.grad_accum_steps,
                epoch=epoch, total_epochs=config.training.epochs,
            )
            if info is None:
                continue
            info = dict(info, epoch=epoch, log_z=float(generator.compute_log_Z().detach().cpu()))
            if _should_evaluate(config, epoch):
                info.update(evaluate_generator(
                    rollout_worker, generator, config.training.eval_episodes,
                    config.runtime.seed + 100_000 + epoch,
                ))
            stop_training = False
            if _should_convergence_evaluate(config, epoch):
                panel_seed = (
                    config.runtime.seed
                    + CONVERGENCE_SEED_OFFSETS[panel_index % len(CONVERGENCE_SEED_OFFSETS)]
                )
                info.update(evaluate_generator(
                    rollout_worker,
                    generator,
                    config.training.convergence_eval_episodes,
                    panel_seed,
                    metric_prefix="convergence_",
                ))
                status, consecutive_passes = _convergence_status(
                    info, config.training, consecutive_passes,
                )
                status["panel_seed"] = panel_seed
                status["panel_index"] = panel_index
                convergence_history.append({"epoch": epoch, **status})
                info["convergence_passed"] = status["passed"]
                info["convergence_current_panel_passed"] = status["current_panel_passed"]
                info["convergence_consecutive_passes"] = consecutive_passes

                current_rmse = float(info["convergence_residual_rmse"])
                metadata = _metadata(
                    config, env, sequences, epoch, float(info["loss"]), info["log_z"],
                    convergence=status,
                    selection_metric="convergence_residual_rmse",
                    selection_value=current_rmse,
                )
                generator.save(last_path, metadata={**metadata, "checkpoint_kind": "last"})
                if current_rmse < best_candidate_rmse:
                    best_candidate_rmse = current_rmse
                    candidate_checkpoint_written = True
                    generator.save(
                        candidate_path,
                        metadata={**metadata, "checkpoint_kind": "best_candidate"},
                    )
                    info["best_candidate_checkpoint_path"] = str(candidate_path)
                if status["passed"] and current_rmse < best_converged_rmse:
                    best_converged_rmse = current_rmse
                    converged_checkpoint_written = True
                    generator.save(
                        best_path,
                        metadata={**metadata, "checkpoint_kind": "best_converged"},
                    )
                    info["best_checkpoint_path"] = str(best_path)
                panel_index += 1
                _write_convergence_report(
                    output_dir,
                    config,
                    convergence_history,
                    candidate_path if candidate_checkpoint_written else None,
                    best_path if converged_checkpoint_written else None,
                )
                stop_training = bool(
                    status["passed"] and config.training.stop_on_convergence
                )

            history.append(info)
            last_info = info
            if run is not None:
                run.log(_wandb_metrics(info, config), step=epoch + 1)
            if config.runtime.verbose:
                print(f"Epoch {epoch + 1}: loss={float(info['loss']):.4f}, logZ={info['log_z']:.4f}")
            if stop_training:
                break
    finally:
        if run is not None:
            run.finish()

    if last_info is not None and not _info_matches_latest_panel(last_info):
        status = _unevaluated_convergence_status(config.training)
        generator.save(
            last_path,
            metadata={
                **_metadata(
                    config,
                    env,
                    sequences,
                    int(last_info["epoch"]),
                    float(last_info["loss"]),
                    float(last_info["log_z"]),
                    convergence=status,
                ),
                "checkpoint_kind": "last",
            },
        )
    _write_convergence_report(
        output_dir,
        config,
        convergence_history,
        candidate_path if candidate_checkpoint_written else None,
        best_path if converged_checkpoint_written else None,
    )
    with (output_dir / "training_history.pkl").open("wb") as handle:
        pickle.dump(history, handle)
    return history


def _convergence_thresholds(options):
    return {
        "min_ess_fraction": float(options.convergence_min_ess_fraction),
        "max_abs_residual_mean": float(options.convergence_max_abs_residual_mean),
        "max_residual_rmse": float(options.convergence_max_residual_rmse),
        "required_consecutive_passes": int(options.convergence_required_passes),
    }


def _convergence_status(info, options, previous_consecutive_passes):
    thresholds = _convergence_thresholds(options)
    metrics = {
        key.removeprefix("convergence_"): float(value)
        for key, value in info.items()
        if key.startswith("convergence_")
    }
    current_pass = (
        metrics["importance_ess_fraction"] >= thresholds["min_ess_fraction"]
        and abs(metrics["residual_mean"]) <= thresholds["max_abs_residual_mean"]
        and metrics["residual_rmse"] <= thresholds["max_residual_rmse"]
    )
    consecutive = previous_consecutive_passes + 1 if current_pass else 0
    return ({
        "version": CONVERGENCE_VERSION,
        "evaluated": True,
        "passed": consecutive >= thresholds["required_consecutive_passes"],
        "current_panel_passed": current_pass,
        "eval_episodes": int(options.convergence_eval_episodes),
        "consecutive_passes": consecutive,
        "thresholds": thresholds,
        "metrics": metrics,
    }, consecutive)


def _unevaluated_convergence_status(options):
    return {
        "version": CONVERGENCE_VERSION,
        "evaluated": False,
        "passed": False,
        "current_panel_passed": False,
        "eval_episodes": int(options.convergence_eval_episodes),
        "consecutive_passes": 0,
        "thresholds": _convergence_thresholds(options),
        "metrics": {},
    }


def _info_matches_latest_panel(info):
    return "convergence_residual_rmse" in info


def _write_convergence_report(
    output_dir, config, history, candidate_path, best_path,
):
    report = {
        "version": CONVERGENCE_VERSION,
        "thresholds": _convergence_thresholds(config.training),
        "convergence_eval_episodes": int(config.training.convergence_eval_episodes),
        "best_candidate_checkpoint": str(candidate_path) if candidate_path else None,
        "best_converged_checkpoint": str(best_path) if best_path else None,
        "passed": best_path is not None,
        "evaluations": history,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "convergence_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def _build_environment(config, sequences):
    values = config.environment
    return SimpleARGEnvironment(
        sequences=sequences,
        sequence_length=len(sequences[0]),
        num_sequences=len(sequences),
        bp_per_blocks=config.data.bp_per_blocks,
        device=config.device,
        seed=config.runtime.seed,
        recombination_rate=values.recombination_rate,
        population_size=values.effective_population_size,
        mutation_rate=values.mutation_rate,
        reward_offset=values.reward_offset,
        time_bins=values.time_bins,
        time_delta_bin_width=values.time_delta_bin_width,
    )


def _build_generator(config, env):
    return TBGFlowNetGenerator(
        env,
        init_z_sample_count=config.training.init_z_sample_count,
        device=config.device,
        verbose=config.runtime.verbose,
        policy_lr=config.optimizer.policy_lr,
        log_z_lr=config.optimizer.log_z_lr,
        grad_clip=config.optimizer.grad_clip,
        model_kwargs=dict(config.model.__dict__),
    )


def _should_evaluate(config, epoch):
    options = config.training
    return options.eval_episodes > 0 and (
        epoch == 0 or options.eval_every <= 1 or (epoch + 1) % options.eval_every == 0
    )


def _should_convergence_evaluate(config, epoch):
    options = config.training
    return options.convergence_eval_episodes > 0 and (
        epoch == 0
        or options.convergence_eval_every <= 1
        or (epoch + 1) % options.convergence_eval_every == 0
    )


def _start_logger(config, env):
    options = config.logging
    if not options.wandb:
        return None
    if wandb is None:
        raise RuntimeError("logging.wandb is true, but wandb is not installed")
    run = wandb.init(
        project=options.project, entity=options.entity, name=options.run_name,
        config={**config.as_dict(), **env.time_metadata, "model_version": MODEL_VERSION},
    )
    run.define_metric("train/loss", summary="min")
    run.define_metric("train/tb_rmse", summary="min")
    run.define_metric("train/log_z", summary="last")
    run.define_metric("eval/tb_mse", summary="min")
    run.define_metric("eval/tb_rmse", summary="min")
    run.define_metric("convergence/residual_rmse", summary="min")
    run.define_metric("convergence/importance_ess_fraction", summary="max")
    return run


def _wandb_metrics(info, config):
    """Return the small, stable metric set used by the W&B dashboard."""
    loss = float(info["loss"])
    metrics = {
        "train/loss": loss,
        "train/tb_rmse": math.sqrt(max(loss, 0.0)),
        "train/log_z": float(info["log_z"]),
        "optimizer/grad_norm": float(info["grad_norm"]),
        "optimizer/grad_norm_pre": float(info.get("grad_norm_pre", info["grad_norm"])),
        "optimizer/grad_norm_post": float(info.get("grad_norm_post", info["grad_norm"])),
        "optimizer/log_z_grad_norm": float(info.get("log_z_grad_norm", 0.0)),
        "optimizer/policy_lr": float(config.optimizer.policy_lr),
        "optimizer/log_z_lr": float(config.optimizer.log_z_lr),
    }
    for source, destination in _WANDB_EVAL_METRICS.items():
        if source in info:
            metrics[destination] = float(info[source])
    if "eval_tb_mse" in info:
        metrics["eval/tb_rmse"] = math.sqrt(max(float(info["eval_tb_mse"]), 0.0))
    for name in ("encoder", "event", "action", "breakpoint", "time", "other"):
        key = f"grad_norm_{name}"
        if key in info:
            metrics[f"optimizer/grad_norm/{name}"] = float(info[key])
    for key, value in info.items():
        if key.startswith("convergence_") and isinstance(value, (int, float, bool)):
            metrics[f"convergence/{key.removeprefix('convergence_')}"] = float(value)
    return metrics


def _metadata(
    config,
    env,
    sequences,
    epoch,
    best_loss,
    log_z,
    *,
    convergence=None,
    selection_metric=None,
    selection_value=None,
):
    return {
        "epoch": epoch, "best_loss": best_loss, "log_z": log_z,
        "sequences": list(sequences), "num_sequences": len(sequences),
        "sequence_length": env.sequence_length, "num_blocks": env.num_blocks,
        "bp_per_blocks": config.data.bp_per_blocks, "rho": env.rho,
        "time": dict(env.time_metadata), **env.time_metadata,
        "effective_population_size": config.environment.effective_population_size,
        "mutation_rate": config.environment.mutation_rate,
        "recombination_rate": config.environment.recombination_rate,
        "reward_offset": config.environment.reward_offset,
        "policy_lr": config.optimizer.policy_lr, "log_z_lr": config.optimizer.log_z_lr,
        "grad_clip": config.optimizer.grad_clip,
        "grad_accum_steps": config.training.grad_accum_steps,
        "eval_episodes": config.training.eval_episodes, "eval_every": config.training.eval_every,
        "model": dict(config.model.__dict__), "seed": config.runtime.seed,
        "init_z_sample_count": config.training.init_z_sample_count,
        "model_version": MODEL_VERSION, "config": config.as_dict(),
        "convergence": dict(convergence or _unevaluated_convergence_status(config.training)),
        "checkpoint_selection_metric": selection_metric,
        "checkpoint_selection_value": selection_value,
    }
