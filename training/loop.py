"""Construction and training loop for the ARG GFlowNet."""

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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(rollout_worker, generator, batch_size=1, grad_accum_steps=1):
    for _ in range(max(int(grad_accum_steps), 1)):
        outputs, _ = rollout_worker.rollout(generator, episodes=batch_size)
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
    checkpoint_path = output_dir / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    run = _start_logger(config, env)
    history, best_loss = [], float("inf")

    try:
        for epoch in range(config.training.epochs):
            info = train_epoch(
                rollout_worker, generator,
                config.training.batch_size, config.training.grad_accum_steps,
            )
            if info is None:
                continue
            info = dict(info, epoch=epoch, log_z=float(generator.compute_log_Z().detach().cpu()))
            if _should_evaluate(config, epoch):
                info.update(evaluate_generator(
                    rollout_worker, generator, config.training.eval_episodes,
                    config.runtime.seed + 100_000 + epoch,
                ))
            history.append(info)
            if run is not None:
                run.log(info, step=epoch + 1)
            if float(info["loss"]) < best_loss:
                best_loss = float(info["loss"])
                generator.save(checkpoint_path, metadata=_metadata(
                    config, env, sequences, epoch, best_loss, info["log_z"],
                ))
                info["best_checkpoint_path"] = str(checkpoint_path)
            if config.runtime.verbose:
                print(f"Epoch {epoch + 1}: loss={float(info['loss']):.4f}, logZ={info['log_z']:.4f}")
    finally:
        if run is not None:
            run.finish()

    with (output_dir / "training_history.pkl").open("wb") as handle:
        pickle.dump(history, handle)
    return history


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


def _start_logger(config, env):
    options = config.logging
    if not options.wandb:
        return None
    if wandb is None:
        raise RuntimeError("logging.wandb is true, but wandb is not installed")
    return wandb.init(
        project=options.project, entity=options.entity, name=options.run_name,
        config={**config.as_dict(), **env.time_metadata, "model_version": MODEL_VERSION},
    )


def _metadata(config, env, sequences, epoch, best_loss, log_z):
    return {
        "epoch": epoch, "best_loss": best_loss, "log_z": log_z,
        "sequences": list(sequences), "num_sequences": len(sequences),
        "sequence_length": env.sequence_length, "num_blocks": env.num_blocks,
        "bp_per_blocks": config.data.bp_per_blocks, "rho": env.rho,
        "time": dict(env.time_metadata), **env.time_metadata,
        "effective_population_size": config.environment.effective_population_size,
        "mutation_rate": config.environment.mutation_rate,
        "recombination_rate": config.environment.recombination_rate,
        "policy_lr": config.optimizer.policy_lr, "log_z_lr": config.optimizer.log_z_lr,
        "grad_clip": config.optimizer.grad_clip,
        "grad_accum_steps": config.training.grad_accum_steps,
        "eval_episodes": config.training.eval_episodes, "eval_every": config.training.eval_every,
        "model": dict(config.model.__dict__), "seed": config.runtime.seed,
        "init_z_sample_count": config.training.init_z_sample_count,
        "model_version": MODEL_VERSION, "config": config.as_dict(),
    }

