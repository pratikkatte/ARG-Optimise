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
from time_env import DEFAULT_TIME_INCREMENTS
from utils import load_sequences


DEFAULT_NE = 10000
DEFAULT_R_PER_BP = 2e-8
DEFAULT_FIXED_EDGE_LENGTH = 0.02
DEFAULT_INIT_Z_SAMPLE_COUNT = 16


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(epoch_id, rollout_worker, generator, batch_size=1):
    ret, _ = rollout_worker.rollout(generator, episodes=batch_size)
    generator.accumulate_loss(ret)
    return generator.update_model()


def parse_time_increments(value):
    if value is None:
        return list(DEFAULT_TIME_INCREMENTS)
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
    use_time_prior=True,
):
    seed_everything(seed)
    time_increments = list(DEFAULT_TIME_INCREMENTS) if time_increments is None else list(time_increments)

    sequences = load_sequences(dataset_path)
    sequence_length = len(sequences[0])
    # num_blocks = sequence_length
    num_blocks = 10000
    rho = 4 * DEFAULT_NE * DEFAULT_R_PER_BP * num_blocks

    env = SimpleARGEnvironment(
        sequence_length=sequence_length,
        num_blocks=num_blocks,
        rho=rho,
        num_sequences=len(sequences),
        sequences=sequences,
        fixed_edge_length=fixed_edge_length,
        learn_times=learn_times,
        time_increments=time_increments,
        use_time_prior=use_time_prior,
        rng=random.Random(seed),
    )
    generator = TBGFlowNetGenerator(
        env,
        init_z_sample_count=init_z_sample_count,
        device=device,
        verbose=init_z_verbose,
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
            "time_increments": list(time_increments or []),
            "use_time_prior": bool(use_time_prior),
        })

    try:
        for epoch in range(epochs_num):
            info = train_epoch(epoch, rollout_worker, generator, batch_size=batch_size)
            log_z = generator.compute_log_Z().detach().cpu().reshape(-1)[0].item()
            if info is None:
                continue

            info = dict(info)
            info["epoch"] = epoch
            info["log_z"] = log_z
            history.append(info)
            loss = float(info["loss"])

            if wandb_run is not None:
                wandb.log({"epoch": epoch, "loss": loss, "logZ": log_z}, step=epoch + 1)

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
                    use_time_prior=use_time_prior,
                    seed=seed,
                    init_z_sample_count=init_z_sample_count,
                )
                generator.save(best_checkpoint_path, metadata=metadata)
                info["best_checkpoint_path"] = best_checkpoint_path

            print(f"Epoch {epoch + 1} loss={loss:.4f} logZ={log_z:.4f}")

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
    use_time_prior,
    seed,
    init_z_sample_count,
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
        "time_increments": list(time_increments or []),
        "use_time_prior": bool(use_time_prior),
        "seed": int(seed),
        "init_z_sample_count": int(init_z_sample_count),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the simplified ARG GFlowNet demo.")
    parser.add_argument("--output-path", default="l1mb_0")
    parser.add_argument("--dataset-path", default="/private/groups/corbettlab/pratik/git/ARG-Optimise_single_env/new_validation/fasta/sim_l1mb_0.fa")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
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
        default=",".join(str(x) for x in DEFAULT_TIME_INCREMENTS),
        help="Comma-separated categorical waiting-time increments used with --learn-times.",
    )
    parser.add_argument("--no-time-prior", action="store_true")

    args = parser.parse_args()

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
        use_time_prior=not args.no_time_prior,
    )


if __name__ == "__main__":
    main()
