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

from env import SimpleARGEnvironment, action_as_dict
from rollout_worker_arg import RolloutWorker
from tb_gfn import TBGFlowNetGenerator
from time_env import DEFAULT_TIME_BINS, DEFAULT_TIME_DELTA_BIN_WIDTH
from utils import load_sequences


DEFAULT_NE = 10000
DEFAULT_R_PER_BP = 2e-8
DEFAULT_MU_PER_BP = 2e-8
DEFAULT_INIT_Z_SAMPLE_COUNT = 16
DEFAULT_POLICY_LR = 1e-3
DEFAULT_LOG_Z_LR = 1e-3
DEFAULT_GRAD_CLIP = 10.0
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_EVAL_EPISODES = 8
DEFAULT_EVAL_EVERY = 10
DEFAULT_EMBEDDING_SIZE = 32
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_DROPOUT = 0.0
DEFAULT_BREAKPOINT_HIDDEN_DIM = 128
DEFAULT_BREAKPOINT_DROPOUT = 0.1
MODEL_VERSION = "cwr-event-learned-time-v1"

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(
    epoch_id,
    rollout_worker,
    generator,
    batch_size=1,
    grad_accum_steps=1,
):
    grad_accum_steps = max(int(grad_accum_steps), 1)

    for _ in range(grad_accum_steps):
        ret, trajectories = rollout_worker.rollout(
            generator,
            episodes=batch_size,
        )
        generator.accumulate_loss(
            ret,
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
            initial_state = env.get_initial_state()
            initial_event_probs = env.compute_event_probabilities(initial_state)

        lengths = torch.tensor([len(traj) for traj in trajectories], dtype=torch.float32)
        coal_counts = torch.tensor(
            [
                sum(
                    1
                    for action in traj.actions
                    if action_as_dict(action).get("event_type") == "coal"
                )
                for traj in trajectories
            ],
            dtype=torch.float32,
        )
        recomb_counts = torch.tensor(
            [
                sum(
                    1
                    for action in traj.actions
                    if action_as_dict(action).get("event_type") == "recomb"
                )
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
            "eval_initial_coalescence_prob": float(initial_event_probs.get("coal", 0.0)),
            "eval_initial_recombination_prob": float(initial_event_probs.get("recomb", 0.0)),
        }
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if env_rng_state is not None:
            env.rng.setstate(env_rng_state)

def train(
    dataset_path,
    output_path,
    device,
    bp_per_blocks=1,
    batch_size=1,
    epochs_num=10,
    seed=7,
    init_z_sample_count=DEFAULT_INIT_Z_SAMPLE_COUNT,
    init_z_verbose=False,
    use_wandb=False,
    effective_population_size=DEFAULT_NE,
    mutation_rate=DEFAULT_MU_PER_BP,
    recombination_rate=DEFAULT_R_PER_BP,
    policy_lr=DEFAULT_POLICY_LR,
    log_z_lr=DEFAULT_LOG_Z_LR,
    grad_clip=DEFAULT_GRAD_CLIP,
    grad_accum_steps=DEFAULT_GRAD_ACCUM_STEPS,
    eval_episodes=DEFAULT_EVAL_EPISODES,
    eval_every=DEFAULT_EVAL_EVERY,
    time_bins=DEFAULT_TIME_BINS,
    time_delta_bin_width=DEFAULT_TIME_DELTA_BIN_WIDTH,
    embedding_size=DEFAULT_EMBEDDING_SIZE,
    hidden_size=DEFAULT_HIDDEN_SIZE,
    dropout=DEFAULT_DROPOUT,
    breakpoint_hidden_dim=DEFAULT_BREAKPOINT_HIDDEN_DIM,
    breakpoint_dropout=DEFAULT_BREAKPOINT_DROPOUT,
    
):
    seed_everything(seed)
    device = torch.device(device)

    sequences = load_sequences(dataset_path)
    sequence_length = len(sequences[0])

    env = SimpleARGEnvironment(
        sequence_length=sequence_length,
        num_sequences=len(sequences),
        bp_per_blocks = bp_per_blocks,
        sequences=sequences,
        device=device,
        recombination_rate=recombination_rate,
        population_size=effective_population_size,
        mutation_rate=mutation_rate,
        time_bins=time_bins,
        time_delta_bin_width=time_delta_bin_width,
    )
    model_kwargs = {
        "embedding_size": int(embedding_size),
        "hidden_size": int(hidden_size),
        "dropout": float(dropout),
        "breakpoint_hidden_dim": int(breakpoint_hidden_dim),
        "breakpoint_dropout": float(breakpoint_dropout),
    }

    generator = TBGFlowNetGenerator(
        env,
        init_z_sample_count=init_z_sample_count,
        device=device,
        verbose=init_z_verbose,
        policy_lr=policy_lr,
        log_z_lr=log_z_lr,
        grad_clip=grad_clip,
        model_kwargs=model_kwargs,
    )
    print(f"Generator device: {generator.device}")

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
            **env.time_metadata,
            "effective_population_size": float(effective_population_size),
            "mutation_rate": float(mutation_rate),
            "recombination_rate": float(recombination_rate),
            "policy_lr": float(policy_lr),
            "log_z_lr": float(log_z_lr),
            "grad_clip": float(grad_clip),
            "grad_accum_steps": int(grad_accum_steps),
            "eval_episodes": int(eval_episodes),
            "eval_every": int(eval_every),
            "bp_per_blocks": int(bp_per_blocks),
            **model_kwargs,
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

            ## TODO: Check this implementation of evaluation
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
                    bp_per_blocks=bp_per_blocks,
                    time_metadata=env.time_metadata,
                    rho=env.rho,
                    effective_population_size=effective_population_size,
                    mutation_rate=mutation_rate,
                    recombination_rate=recombination_rate,
                    policy_lr=policy_lr,
                    log_z_lr=log_z_lr,
                    grad_clip=grad_clip,
                    grad_accum_steps=grad_accum_steps,
                    eval_episodes=eval_episodes,
                    eval_every=eval_every,
                    model_kwargs=model_kwargs,
                    seed=seed,
                    init_z_sample_count=init_z_sample_count,
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
    bp_per_blocks,
    time_metadata,
    rho,
    effective_population_size,
    mutation_rate,
    recombination_rate,
    policy_lr,
    log_z_lr,
    grad_clip,
    grad_accum_steps,
    eval_episodes,
    eval_every,
    model_kwargs,
    seed,
    init_z_sample_count,
    model_version,
):
    return {
        "epoch": int(epoch),
        "best_loss": float(best_loss),
        "log_z": float(log_z),
        "sequences": list(sequences),
        "num_sequences": len(sequences),
        "sequence_length": int(sequence_length),
        "bp_per_blocks": int(bp_per_blocks),
        "rho": float(rho),
        "time": dict(time_metadata),
        "effective_population_size": float(effective_population_size),
        "mutation_rate": float(mutation_rate),
        "recombination_rate": float(recombination_rate),
        "policy_lr": float(policy_lr),
        "log_z_lr": float(log_z_lr),
        "grad_clip": float(grad_clip),
        "grad_accum_steps": int(grad_accum_steps),
        "eval_episodes": int(eval_episodes),
        "eval_every": int(eval_every),
        "model": dict(model_kwargs),
        "seed": int(seed),
        "init_z_sample_count": int(init_z_sample_count),
        "model_version": str(model_version),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the simplified ARG GFlowNet demo.")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--dataset-path",required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--bp-per-blocks",
        type=int,
        default=1,
        help="Number of bp per block",
    )
    parser.add_argument("--init-z-sample-count", type=int, default=DEFAULT_INIT_Z_SAMPLE_COUNT)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--effective-population-size", type=float, default=DEFAULT_NE)
    parser.add_argument("--mutation-rate", type=float, default=DEFAULT_MU_PER_BP)
    parser.add_argument("--recombination-rate", type=float, default=DEFAULT_R_PER_BP)
    parser.add_argument("--policy-lr", type=float, default=DEFAULT_POLICY_LR)
    parser.add_argument("--log-z-lr", type=float, default=DEFAULT_LOG_Z_LR)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=DEFAULT_GRAD_ACCUM_STEPS,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--time-bins", type=int, default=DEFAULT_TIME_BINS)
    parser.add_argument("--time-delta-bin-width", type=float, default=DEFAULT_TIME_DELTA_BIN_WIDTH)
    parser.add_argument("--embedding-size", type=int, default=DEFAULT_EMBEDDING_SIZE)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--breakpoint-hidden-dim", type=int, default=DEFAULT_BREAKPOINT_HIDDEN_DIM)
    parser.add_argument("--breakpoint-dropout", type=float, default=DEFAULT_BREAKPOINT_DROPOUT)
    parser.add_argument("--wandb", action="store_true", default=False)
    args = parser.parse_args()

    selected_device = "cuda" if torch.cuda.is_available() else "cpu"
                
    print(f"Selected device: {selected_device}")

    train(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        epochs_num=args.epochs,
        bp_per_blocks=args.bp_per_blocks,
        init_z_sample_count=args.init_z_sample_count,
        init_z_verbose=args.verbose,
        seed=args.seed,
        device=selected_device,
        use_wandb=args.wandb,
        effective_population_size=args.effective_population_size,
        mutation_rate=args.mutation_rate,
        recombination_rate=args.recombination_rate,
        policy_lr=args.policy_lr,
        log_z_lr=args.log_z_lr,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum_steps,
        eval_episodes=args.eval_episodes,
        eval_every=args.eval_every,
        time_bins=args.time_bins,
        time_delta_bin_width=args.time_delta_bin_width,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        breakpoint_hidden_dim=args.breakpoint_hidden_dim,
        breakpoint_dropout=args.breakpoint_dropout,
    )


if __name__ == "__main__":
    main()
