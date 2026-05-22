import argparse
import os
import pickle
import random

import numpy as np
import torch

import wandb

from env import SimpleARGEnvironment
from rollout_worker_arg import RolloutWorker
from tb_gfn import TBGFlowNetGenerator
from time_env import (
    DEFAULT_TIME_BINS,
    DEFAULT_TIME_MODEL,
    DEFAULT_TIME_TAIL_PROBABILITY,
)
from utils import load_sequences


DEFAULT_NE = 10000
DEFAULT_R_PER_BP = 2e-8
DEFAULT_MU_PER_BP = 2e-8
DEFAULT_FIXED_EDGE_LENGTH = 0.02
DEFAULT_INIT_Z_SAMPLE_COUNT = 2
DEFAULT_SEQUENCE_ENCODER_BINS = 1024
DEFAULT_BREAKPOINT_MIXTURES = 4
DEFAULT_LOG_Z_LR = 0.05
DEFAULT_GRAD_ACCUM_STEPS = 4
DEFAULT_EVAL_EPISODES = 8
DEFAULT_EVAL_EVERY = 10
MODEL_VERSION = "learned-breakpoint-v1"

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(epoch_id, rollout_worker, generator, batch_size=1, grad_accum_steps=1):
    grad_accum_steps = max(int(grad_accum_steps), 1)
    for _ in range(grad_accum_steps):
        with torch.no_grad():
            ret, trajectories = rollout_worker.rollout(generator, episodes=batch_size)
        generator.accumulate_streaming_tb_loss(
            rollout_worker,
            ret,
            trajectories,
            factor=grad_accum_steps,
        )
    return generator.update_model()


def evaluate_generator(rollout_worker, generator, episodes, seed):
    episodes = int(episodes)
    if episodes <= 0:
        return {}

    env = rollout_worker.env
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    env_rng_state = env.rng.getstate() if hasattr(env.rng, "getstate") else None

    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(env.rng, "seed"):
            env.rng.seed(seed)

        with torch.no_grad():
            outputs, trajectories = rollout_worker.rollout(generator, episodes=episodes)
            log_pf = outputs["log_paths_pf"].sum(-1)
            log_pb = outputs["log_paths_pb"].sum(-1)
            log_rewards = outputs["log_rewards"]
            residuals = generator.compute_log_Z().detach().to(log_pf) + log_pf - (
                log_rewards + log_pb
            )

        lengths = torch.tensor([len(traj) for traj in trajectories], dtype=torch.float32)
        coal_counts = torch.tensor(
            [
                sum(1 for record in traj if record["action"].get("event_type") == "coal")
                for traj in trajectories
            ],
            dtype=torch.float32,
        )
        recomb_counts = torch.tensor(
            [
                sum(1 for record in traj if record["action"].get("event_type") == "recomb")
                for traj in trajectories
            ],
            dtype=torch.float32,
        )
        return {
            "eval_tb_mse": float(residuals.pow(2).mean().detach().cpu().item()),
            "eval_residual_mean": float(residuals.mean().detach().cpu().item()),
            "eval_residual_std": float(
                residuals.std(unbiased=False).detach().cpu().item()
            ),
            "eval_log_pf_mean": float(log_pf.mean().detach().cpu().item()),
            "eval_log_pb_mean": float(log_pb.mean().detach().cpu().item()),
            "eval_log_reward_mean": float(log_rewards.mean().detach().cpu().item()),
            "eval_trajectory_length_mean": float(lengths.mean().item()),
            "eval_coalescence_count_mean": float(coal_counts.mean().item()),
            "eval_recombination_count_mean": float(recomb_counts.mean().item()),
        }
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if env_rng_state is not None:
            env.rng.setstate(env_rng_state)

def parse_time_increments(value):
    if value is None:
        return None
    increments = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not increments:
        raise ValueError("--time-increments must contain at least one positive value")
    if any(increment <= 0 for increment in increments):
        raise ValueError("--time-increments values must all be positive")
    return increments


def train(
    output_path,
    batch_size=1,
    epochs_num=10,
    dataset_path="/private/groups/corbettlab/pratik/git/ARG-Optimise_single_env/new_validation/fasta/sim_l1mb_0.fa",
    seed=7,
    init_z_sample_count=DEFAULT_INIT_Z_SAMPLE_COUNT,
    init_z_verbose=False,
    device="auto",
    use_wandb=True,
    fixed_edge_length=DEFAULT_FIXED_EDGE_LENGTH,
    learn_times=False,
    time_increments=None,
    time_model=DEFAULT_TIME_MODEL,
    time_bins=DEFAULT_TIME_BINS,
    time_tail_probability=DEFAULT_TIME_TAIL_PROBABILITY,
    effective_population_size=DEFAULT_NE,
    mutation_rate=DEFAULT_MU_PER_BP,
    use_time_prior=True,
    recombination_rate=DEFAULT_R_PER_BP,
    breakpoint_policy="learned-bin-mass",
    breakpoint_mixtures=DEFAULT_BREAKPOINT_MIXTURES,
    reward_mode="posterior",
    log_z_lr=DEFAULT_LOG_Z_LR,
    grad_accum_steps=DEFAULT_GRAD_ACCUM_STEPS,
    eval_episodes=DEFAULT_EVAL_EPISODES,
    eval_every=DEFAULT_EVAL_EVERY,
    num_blocks=None,
    sequence_encoder_bins=DEFAULT_SEQUENCE_ENCODER_BINS,
    smoke_test=False,
):
    seed_everything(seed)
    time_increments = None if time_increments is None else list(time_increments)

    sequences = load_sequences(dataset_path)
    sequence_length = len(sequences[0])
    if num_blocks is None:
        num_blocks = min(sequence_length, 256) if smoke_test else sequence_length
    num_blocks = int(num_blocks)
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if num_blocks > sequence_length:
        raise ValueError("num_blocks must be less than or equal to sequence length")
    if smoke_test:
        sequence_encoder_bins = min(int(sequence_encoder_bins), 256)
    rho = 4 * float(effective_population_size) * float(recombination_rate) * sequence_length

    env = SimpleARGEnvironment(
        sequence_length=sequence_length,
        num_blocks=num_blocks,
        rho=rho,
        num_sequences=len(sequences),
        sequences=sequences,
        fixed_edge_length=fixed_edge_length,
        learn_times=learn_times,
        time_increments=time_increments,
        time_model=time_model,
        time_bins=time_bins,
        time_tail_probability=time_tail_probability,
        effective_population_size=effective_population_size,
        mutation_rate=mutation_rate,
        use_time_prior=use_time_prior,
        reward_mode=reward_mode,
        rng=random.Random(seed),
    )
    generator = TBGFlowNetGenerator(
        env,
        init_z_sample_count=init_z_sample_count,
        cfg={
            "sequence_encoder_bins": sequence_encoder_bins,
            "breakpoint_policy": breakpoint_policy,
            "breakpoint_mixtures": breakpoint_mixtures,
        },
        device=device,
        verbose=init_z_verbose,
        log_z_lr=log_z_lr,
    )
    rollout_worker = RolloutWorker(env)
    print(f"Training on device: {generator.device}")

    os.makedirs(output_path, exist_ok=True)
    checkpoints_path = os.path.join(output_path, "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoints_path, "best.pt")

    history = []
    best_loss = float("inf")
    wandb_run = None
    if use_wandb and wandb is not None:
        wandb_run = wandb.init()
        wandb.config.update({
            "device": str(generator.device),
            "learn_times": bool(learn_times),
            "time_model": env.time_model,
            "time_bins": env.time_bins,
            "time_tail_probability": env.time_tail_probability,
            "time_increments": list(time_increments) if time_increments is not None else None,
            "effective_population_size": float(effective_population_size),
            "mutation_rate": float(mutation_rate),
            "use_time_prior": bool(use_time_prior),
            "recombination_rate": float(recombination_rate),
            "breakpoint_policy": generator.arg_model.breakpoint_policy,
            "breakpoint_mixtures": int(generator.arg_model.breakpoint_mixtures),
            "reward_mode": env.reward_mode,
            "log_z_lr": float(log_z_lr),
            "grad_accum_steps": int(grad_accum_steps),
            "eval_episodes": int(eval_episodes),
            "eval_every": int(eval_every),
            "num_blocks": int(num_blocks),
            "sequence_encoder_bins": int(generator.arg_model.sequence_encoder_bins),
            "model_version": MODEL_VERSION,
        })

    try:
        for epoch in range(epochs_num):
            info = train_epoch(
                epoch,
                rollout_worker,
                generator,
                batch_size=batch_size,
                grad_accum_steps=grad_accum_steps,
            )
            log_z = generator.compute_log_Z().detach().cpu().reshape(-1)[0].item()
            if info is None:
                continue

            info = dict(info)
            info["epoch"] = epoch
            info["log_z"] = log_z
            should_eval = int(eval_episodes) > 0 and (
                epoch == 0
                or int(eval_every) <= 1
                or (epoch + 1) % int(eval_every) == 0
            )
            if should_eval:
                info.update(
                    evaluate_generator(
                        rollout_worker,
                        generator,
                        eval_episodes,
                        seed + 100000 + epoch,
                    )
                )
            history.append(info)
            loss = float(info["loss"])

            if wandb_run is not None:
                wandb.log(info, step=epoch + 1)

            if loss < best_loss:
                best_loss = loss
                metadata = build_checkpoint_metadata(
                    epoch=epoch,
                    best_loss=best_loss,
                    log_z=log_z,
                    sequences=sequences,
                    sequence_length=sequence_length,
                    num_blocks=num_blocks,
                    rho=rho,
                    fixed_edge_length=fixed_edge_length,
                    learn_times=learn_times,
                    time_increments=time_increments,
                    time_model=env.time_model,
                    time_bins=env.time_bins,
                    time_tail_probability=env.time_tail_probability,
                    effective_population_size=effective_population_size,
                    mutation_rate=mutation_rate,
                    use_time_prior=use_time_prior,
                    recombination_rate=recombination_rate,
                    breakpoint_policy=generator.arg_model.breakpoint_policy,
                    breakpoint_mixtures=generator.arg_model.breakpoint_mixtures,
                    reward_mode=env.reward_mode,
                    log_z_lr=log_z_lr,
                    grad_accum_steps=grad_accum_steps,
                    eval_episodes=eval_episodes,
                    eval_every=eval_every,
                    seed=seed,
                    init_z_sample_count=init_z_sample_count,
                    sequence_encoder_bins=generator.arg_model.sequence_encoder_bins,
                    model_version=MODEL_VERSION,
                )
                generator.save(best_checkpoint_path, metadata=metadata)
                info["best_checkpoint_path"] = best_checkpoint_path

            eval_text = ""
            if "eval_tb_mse" in info:
                eval_text = (
                    f" eval_tb_mse={info['eval_tb_mse']:.4f}"
                    f" eval_residual_mean={info['eval_residual_mean']:.4f}"
                )
            print(f"Epoch {epoch + 1} loss={loss:.4f} logZ={log_z:.4f}{eval_text}")

        with open(os.path.join(output_path, "training_history.pkl"), "wb") as handle:
            pickle.dump(history, handle)
    finally:
        if wandb_run is not None:
            wandb.finish()
    return history


def build_checkpoint_metadata(
    epoch,
    best_loss,
    log_z,
    sequences,
    sequence_length,
    num_blocks,
    rho,
    fixed_edge_length,
    learn_times,
    time_increments,
    time_model,
    time_bins,
    time_tail_probability,
    effective_population_size,
    mutation_rate,
    use_time_prior,
    recombination_rate,
    breakpoint_policy,
    breakpoint_mixtures,
    reward_mode,
    log_z_lr,
    grad_accum_steps,
    eval_episodes,
    eval_every,
    seed,
    init_z_sample_count,
    sequence_encoder_bins,
    model_version,
):
    return {
        "epoch": int(epoch),
        "best_loss": float(best_loss),
        "log_z": float(log_z),
        "sequences": list(sequences),
        "num_sequences": len(sequences),
        "sequence_length": int(sequence_length),
        "num_blocks": int(num_blocks),
        "rho": float(rho),
        "fixed_edge_length": float(fixed_edge_length),
        "learn_times": bool(learn_times),
        "time_model": time_model,
        "time_bins": int(time_bins),
        "time_tail_probability": float(time_tail_probability),
        "time_increments": list(time_increments) if time_increments is not None else None,
        "effective_population_size": float(effective_population_size),
        "mutation_rate": float(mutation_rate),
        "use_time_prior": bool(use_time_prior),
        "recombination_rate": float(recombination_rate),
        "breakpoint_policy": str(breakpoint_policy),
        "breakpoint_mixtures": int(breakpoint_mixtures),
        "reward_mode": str(reward_mode),
        "log_z_lr": float(log_z_lr),
        "grad_accum_steps": int(grad_accum_steps),
        "eval_episodes": int(eval_episodes),
        "eval_every": int(eval_every),
        "seed": int(seed),
        "init_z_sample_count": int(init_z_sample_count),
        "sequence_encoder_bins": int(sequence_encoder_bins),
        "model_version": str(model_version),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the simplified ARG GFlowNet demo.")
    parser.add_argument("--output-path", default="l025mb_0")
    parser.add_argument("--dataset-path", default="validation/fasta/sim_l25kb_0.fa")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Discrete ARG block count. Defaults to the full sequence length.",
    )
    parser.add_argument(
        "--sequence-encoder-bins",
        type=int,
        default=DEFAULT_SEQUENCE_ENCODER_BINS,
        help="Number of sequence bins used by the compact policy encoder.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a smallest-memory development configuration.",
    )
    parser.add_argument(
        "--init-z-sample-count",
        type=int,
        default=DEFAULT_INIT_Z_SAMPLE_COUNT,
        help="Number of prior rollouts used to initialize logZ; use 1 for quick smoke tests.",
    )
    parser.add_argument("--verbose-init", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--fixed-edge-length", type=float, default=DEFAULT_FIXED_EDGE_LENGTH)
    parser.add_argument("--learn-times", action="store_true")
    parser.add_argument(
        "--time-increments",
        default=None,
        help=(
            "Legacy comma-separated fixed waiting-time increments used with "
            "--learn-times. Omit to use adaptive exponential time bins."
        ),
    )
    parser.add_argument("--time-model", default=DEFAULT_TIME_MODEL)
    parser.add_argument("--time-bins", type=int, default=DEFAULT_TIME_BINS)
    parser.add_argument(
        "--time-tail-probability",
        type=float,
        default=DEFAULT_TIME_TAIL_PROBABILITY,
    )
    parser.add_argument("--effective-population-size", type=float, default=DEFAULT_NE)
    parser.add_argument("--mutation-rate", type=float, default=DEFAULT_MU_PER_BP)
    parser.add_argument("--recombination-rate", type=float, default=DEFAULT_R_PER_BP)
    parser.add_argument(
        "--breakpoint-policy",
        choices=["learned-bin-mass", "uniform"],
        default="learned-bin-mass",
    )
    parser.add_argument(
        "--breakpoint-mixtures",
        type=int,
        default=DEFAULT_BREAKPOINT_MIXTURES,
    )
    parser.add_argument(
        "--reward-mode",
        choices=["posterior", "likelihood-only", "trajectory-prior"],
        default="posterior",
    )
    parser.add_argument("--log-z-lr", type=float, default=DEFAULT_LOG_Z_LR)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=DEFAULT_GRAD_ACCUM_STEPS,
    )
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--no-time-prior", action="store_true")

    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 1
        args.batch_size = 1
        args.init_z_sample_count = 1
        args.no_wandb = True
        args.sequence_encoder_bins = min(args.sequence_encoder_bins, 256)
        args.grad_accum_steps = 1
        args.eval_episodes = min(args.eval_episodes, 1)

    if args.device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected_device = args.device
    print(f"Selected device: {selected_device}")

    train(
        args.output_path,
        batch_size=args.batch_size,
        epochs_num=args.epochs,
        dataset_path=args.dataset_path,
        seed=args.seed,
        init_z_sample_count=args.init_z_sample_count,
        init_z_verbose=args.verbose_init,
        device=args.device,
        use_wandb=not args.no_wandb,
        fixed_edge_length=args.fixed_edge_length,
        learn_times=args.learn_times,
        time_increments=parse_time_increments(args.time_increments),
        time_model=args.time_model,
        time_bins=args.time_bins,
        time_tail_probability=args.time_tail_probability,
        effective_population_size=args.effective_population_size,
        mutation_rate=args.mutation_rate,
        use_time_prior=not args.no_time_prior,
        recombination_rate=args.recombination_rate,
        breakpoint_policy=args.breakpoint_policy,
        breakpoint_mixtures=args.breakpoint_mixtures,
        reward_mode=args.reward_mode,
        log_z_lr=args.log_z_lr,
        grad_accum_steps=args.grad_accum_steps,
        eval_episodes=args.eval_episodes,
        eval_every=args.eval_every,
        num_blocks=args.num_blocks,
        sequence_encoder_bins=args.sequence_encoder_bins,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
