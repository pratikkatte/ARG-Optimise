import argparse
import math
import os
import pickle
import random
import sys

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
DEFAULT_EVAL_EPISODES = 128
DEFAULT_EVAL_EVERY = 10
DEFAULT_EMBEDDING_SIZE = 32
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_DROPOUT = 0.0
DEFAULT_BREAKPOINT_HIDDEN_DIM = 128
DEFAULT_BREAKPOINT_DROPOUT = 0.1
DEFAULT_TRANSFORMER_DEPTH = 6
DEFAULT_TRANSFORMER_HEADS = 4
DEFAULT_TRANSFORMER_MLP_RATIO = 2.0
DEFAULT_ATTENTION_DROPOUT = 0.0
DEFAULT_TIME_HIDDEN_SIZE = 256
DEFAULT_TIME_LAYERS = 3
DEFAULT_TIME_DROPOUT = 0.0
DEFAULT_BREAKPOINT_GAP_HIDDEN_SIZE = 256
DEFAULT_BREAKPOINT_GAP_LAYERS = 3
DEFAULT_BREAKPOINT_GAP_DROPOUT = 0.0
DEFAULT_BREAKPOINT_USE_POSITION_FEATURES = True
MODEL_VERSION = "cwr-event-transformer-block-partials-v3"

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
    module_modes = [(module, module.training) for module in generator.modules()]

    try:
        generator.eval()
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
        for module, training in module_modes:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if env_rng_state is not None:
            env.rng.setstate(env_rng_state)


def save_best_checkpoints(generator, info, metadata, checkpoints_path, best_scores):
    """Keep independent minima; residual mean is best when closest to zero."""
    criteria = (
        ("loss", "best.pt", "best_loss"),
        ("eval_tb_mse", "best_eval_loss.pt", "best_eval_loss"),
        ("eval_residual_mean", "best_residual_mean.pt", "best_abs_residual_mean"),
        ("eval_residual_std", "best_residual_std.pt", "best_residual_std"),
    )
    logged = {}
    for metric, filename, best_key in criteria:
        if metric not in info:
            continue
        value = float(info[metric])
        score = abs(value) if metric == "eval_residual_mean" else value
        if not math.isfinite(score):
            continue
        if score < best_scores.get(metric, float("inf")):
            path = os.path.join(checkpoints_path, filename)
            checkpoint_metadata = {
                **metadata,
                **{key: float(value) for key, value in info.items()
                   if key == "loss" or key.startswith("eval_")},
                "checkpoint_metric": metric,
                "checkpoint_metric_value": value,
                "checkpoint_score": score,
            }
            generator.save(path, metadata=checkpoint_metadata)
            best_scores[metric] = score
            path_key = "best_checkpoint_path" if metric == "loss" else f"{best_key}_checkpoint_path"
            logged[path_key] = path
        logged[best_key] = best_scores[metric]
    return logged


def train(
    dataset_path,
    output_path,
    device,
    bp_per_blocks=1,
    batch_size=1,
    epochs_num=10,
    seed=7,
    init_z_sample_count=DEFAULT_INIT_Z_SAMPLE_COUNT,
    use_wandb=True,
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
    transformer_depth=DEFAULT_TRANSFORMER_DEPTH,
    transformer_heads=DEFAULT_TRANSFORMER_HEADS,
    transformer_mlp_ratio=DEFAULT_TRANSFORMER_MLP_RATIO,
    attention_dropout=DEFAULT_ATTENTION_DROPOUT,
    verbose=True,
    
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
        "transformer_depth": int(transformer_depth),
        "transformer_heads": int(transformer_heads),
        "transformer_mlp_ratio": float(transformer_mlp_ratio),
        "attention_dropout": float(attention_dropout),
        "time_hidden_size": int(DEFAULT_TIME_HIDDEN_SIZE),
        "time_layers": int(DEFAULT_TIME_LAYERS),
        "time_dropout": float(DEFAULT_TIME_DROPOUT),
        "breakpoint_gap_hidden_size": int(DEFAULT_BREAKPOINT_GAP_HIDDEN_SIZE),
        "breakpoint_gap_layers": int(DEFAULT_BREAKPOINT_GAP_LAYERS),
        "breakpoint_gap_dropout": float(DEFAULT_BREAKPOINT_GAP_DROPOUT),
        "breakpoint_use_position_features": bool(DEFAULT_BREAKPOINT_USE_POSITION_FEATURES),
    }

    generator = TBGFlowNetGenerator(
        env,
        init_z_sample_count=init_z_sample_count,
        device=device,
        verbose=verbose,
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

    history = []
    best_scores = {}
    wandb_run = None
    
    print(f"use_wandb: {use_wandb}")
    if use_wandb:
        wandb_run = wandb.init()
        wandb.config.update({
            "device": str(generator.device),
            **env.time_metadata,
            "effective_population_size": float(effective_population_size),
            "mutation_rate": float(mutation_rate),
            "recombination_rate": float(recombination_rate),
            "policy_lr": float(policy_lr),
            "log_z_lr": float(log_z_lr),
            "log_z_initialization": "policy_tb_mean",
            "init_z_sample_count": int(init_z_sample_count),
            "initial_log_z": float(generator.compute_log_Z().detach().cpu().item()),
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

            if should_eval:
                info.update(
                    evaluate_generator(
                        rollout_worker,
                        generator,
                        eval_episodes,
                        seed + 100000 + epoch,
                    )
                )
            loss = float(info["loss"])

            metadata = build_checkpoint_metadata(
                epoch=epoch,
                best_loss=min(best_scores.get("loss", float("inf")), loss),
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
            info.update(save_best_checkpoints(
                generator, info, metadata, checkpoints_path, best_scores,
            ))
            history.append(info)

            if wandb_run is not None:
                wandb.log(info, step=epoch + 1)

            eval_text = ""
            if "eval_tb_mse" in info:
                eval_text = (
                    f" eval_tb_mse={info['eval_tb_mse']:.4f}"
                    f" eval_residual_mean={info['eval_residual_mean']:.4f}"
                    f" eval_residual_std={info['eval_residual_std']:.4f}"
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
        "num_blocks": int(sequence_length // bp_per_blocks),
        "bp_per_blocks": int(bp_per_blocks),
        "rho": float(rho),
        "time": dict(time_metadata),
        **dict(time_metadata),
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
        "log_z_initialization": "policy_tb_mean",
        "model_version": str(model_version),
    }


def parse_train_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Train the simplified ARG GFlowNet demo.")
    parser.add_argument("--config", help="YAML settings file; command-line options override its values")
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
    parser.add_argument(
        "--init-z-sample-count", type=int, default=DEFAULT_INIT_Z_SAMPLE_COUNT,
        help="Initial-policy trajectories used to center the trajectory-balance residual",
    )
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--transformer-depth", type=int, default=DEFAULT_TRANSFORMER_DEPTH)
    parser.add_argument("--transformer-heads", type=int, default=DEFAULT_TRANSFORMER_HEADS)
    parser.add_argument("--transformer-mlp-ratio", type=float, default=DEFAULT_TRANSFORMER_MLP_RATIO)
    parser.add_argument("--attention-dropout", type=float, default=DEFAULT_ATTENTION_DROPOUT)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_path = config_parser.parse_known_args(argv)[0].config
    config_args = []
    if config_path:
        import yaml

        try:
            with open(config_path) as handle:
                settings = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError) as exc:
            parser.error(f"Cannot read config {config_path}: {exc}")
        if not isinstance(settings, dict):
            parser.error("Config must contain a mapping of argument names to values")
        actions = {action.dest: action for action in parser._actions
                   if action.dest not in {"help", "config"}}
        for key, value in settings.items():
            if key not in actions:
                parser.error(f"Unknown config setting: {key}")
            action = actions[key]
            if isinstance(action, argparse.BooleanOptionalAction):
                if not isinstance(value, bool):
                    parser.error(f"Config setting {key} must be true or false")
                config_args.append(action.option_strings[0 if value else 1])
            else:
                if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                    parser.error(f"Config setting {key} must be a scalar value")
                config_args.append(f"{action.option_strings[0]}={value}")
    return parser.parse_args(config_args + argv)


def main():
    args = parse_train_args()

    selected_device = "cuda" if torch.cuda.is_available() else "cpu"
                
    print(f"Selected devicesss: {selected_device}")

    train(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        epochs_num=args.epochs,
        bp_per_blocks=args.bp_per_blocks,
        init_z_sample_count=args.init_z_sample_count,
        verbose=args.verbose,
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
        transformer_depth=args.transformer_depth,
        transformer_heads=args.transformer_heads,
        transformer_mlp_ratio=args.transformer_mlp_ratio,
        attention_dropout=args.attention_dropout,
    )


if __name__ == "__main__":
    main()
