import argparse
import os
import pickle
import random

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

try:
    from .env import SimpleARGEnvironment
    from .rollout_worker_arg import RolloutWorker
    from .tb_gfn import TBGFlowNetGenerator
    from .utils import load_sequences
except ImportError:
    from env import SimpleARGEnvironment
    from rollout_worker_arg import RolloutWorker
    from tb_gfn import TBGFlowNetGenerator
    from utils import load_sequences


DEFAULT_NE = 10000
DEFAULT_R_PER_BP = 1e-8
DEFAULT_FIXED_EDGE_LENGTH = 0.02
DEFAULT_INIT_Z_SAMPLE_COUNT = 2


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


def train(
    output_path,
    batch_size=1,
    epochs_num=10,
    dataset_path="../dataset/DS1.pickle",
    seed=7,
    init_z_sample_count=DEFAULT_INIT_Z_SAMPLE_COUNT,
    use_wandb=True,
):
    seed_everything(seed)

    sequences = load_sequences(dataset_path)
    sequence_length = len(sequences[0])
    num_blocks = sequence_length
    rho = 4 * DEFAULT_NE * DEFAULT_R_PER_BP * sequence_length

    env = SimpleARGEnvironment(
        sequence_length=sequence_length,
        num_blocks=num_blocks,
        rho=rho,
        num_sequences=len(sequences),
        sequences=sequences,
        fixed_edge_length=DEFAULT_FIXED_EDGE_LENGTH,
        rng=random.Random(seed),
    )
    generator = TBGFlowNetGenerator(env, init_z_sample_count=init_z_sample_count)
    rollout_worker = RolloutWorker(env)

    os.makedirs(output_path, exist_ok=True)
    checkpoints_path = os.path.join(output_path, "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoints_path, "best.pt")

    history = []
    best_loss = float("inf")
    wandb_run = None
    if use_wandb and wandb is not None:
        wandb_run = wandb.init()

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
                    fixed_edge_length=DEFAULT_FIXED_EDGE_LENGTH,
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
        "seed": int(seed),
        "init_z_sample_count": int(init_z_sample_count),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the simplified ARG GFlowNet demo.")
    parser.add_argument("--output-path", default=".")
    parser.add_argument("--dataset-path", default="../dataset/DS1.pickle")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()

    train(
        args.output_path,
        batch_size=args.batch_size,
        epochs_num=args.epochs,
        dataset_path=args.dataset_path,
        seed=args.seed,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
