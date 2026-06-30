import argparse
import copy
import os
import pickle
import random

import numpy as np
import torch
import yaml

try:
    import wandb
except ImportError:
    wandb = None

from env import SimpleARGEnvironment, action_as_dict
from rollout_worker_arg import RolloutWorker
from tb_gfn import TBGFlowNetGenerator
from time_env import DEFAULT_TIME_BINS, DEFAULT_TIME_DELTA_BIN_WIDTH
from utils import VCF_PARSER_VERSION, is_vcf_path, load_sequences, load_vcf_variants


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
DEFAULT_LOSS = "tb"
DEFAULT_SUBTB_LAMBDA = 0.9
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
MODEL_VERSION = "cwr-event-sparse-vcf-v1"


DEFAULT_CONFIG = {
    "dataset_path": None,
    "output_path": None,
    "device": "auto",
    "training": {
        "epochs": None,
        "batch_size": 10,
        "seed": 7,
        "init_z_sample_count": DEFAULT_INIT_Z_SAMPLE_COUNT,
        "verbose": False,
        "wandb": True,
        "policy_lr": DEFAULT_POLICY_LR,
        "log_z_lr": DEFAULT_LOG_Z_LR,
        "loss": DEFAULT_LOSS,
        "subtb_lambda": DEFAULT_SUBTB_LAMBDA,
        "grad_clip": DEFAULT_GRAD_CLIP,
        "grad_accum_steps": DEFAULT_GRAD_ACCUM_STEPS,
        "eval_episodes": DEFAULT_EVAL_EPISODES,
        "eval_every": DEFAULT_EVAL_EVERY,
    },
    "environment": {
        "bp_per_blocks": 1,
        "effective_population_size": DEFAULT_NE,
        "mutation_rate": DEFAULT_MU_PER_BP,
        "recombination_rate": DEFAULT_R_PER_BP,
        "time_bins": DEFAULT_TIME_BINS,
        "time_delta_bin_width": DEFAULT_TIME_DELTA_BIN_WIDTH,
    },
    "reward": {
        "constant": 30000,
    },
    "model": {
        "embedding_size": DEFAULT_EMBEDDING_SIZE,
        "hidden_size": DEFAULT_HIDDEN_SIZE,
        "dropout": DEFAULT_DROPOUT,
        "breakpoint_hidden_dim": DEFAULT_BREAKPOINT_HIDDEN_DIM,
        "breakpoint_dropout": DEFAULT_BREAKPOINT_DROPOUT,
        "transformer_depth": DEFAULT_TRANSFORMER_DEPTH,
        "transformer_heads": DEFAULT_TRANSFORMER_HEADS,
        "transformer_mlp_ratio": DEFAULT_TRANSFORMER_MLP_RATIO,
        "attention_dropout": DEFAULT_ATTENTION_DROPOUT,
    },
}


CLI_CONFIG_PATHS = {
    "dataset_path": ("dataset_path",),
    "output_path": ("output_path",),
    "device": ("device",),
    "epochs": ("training", "epochs"),
    "batch_size": ("training", "batch_size"),
    "seed": ("training", "seed"),
    "init_z_sample_count": ("training", "init_z_sample_count"),
    "verbose": ("training", "verbose"),
    "wandb": ("training", "wandb"),
    "policy_lr": ("training", "policy_lr"),
    "log_z_lr": ("training", "log_z_lr"),
    "loss": ("training", "loss"),
    "subtb_lambda": ("training", "subtb_lambda"),
    "grad_clip": ("training", "grad_clip"),
    "grad_accum_steps": ("training", "grad_accum_steps"),
    "eval_episodes": ("training", "eval_episodes"),
    "eval_every": ("training", "eval_every"),
    "bp_per_blocks": ("environment", "bp_per_blocks"),
    "effective_population_size": ("environment", "effective_population_size"),
    "mutation_rate": ("environment", "mutation_rate"),
    "recombination_rate": ("environment", "recombination_rate"),
    "time_bins": ("environment", "time_bins"),
    "time_delta_bin_width": ("environment", "time_delta_bin_width"),
    "reward_constant": ("reward", "constant"),
    "embedding_size": ("model", "embedding_size"),
    "hidden_size": ("model", "hidden_size"),
    "dropout": ("model", "dropout"),
    "breakpoint_hidden_dim": ("model", "breakpoint_hidden_dim"),
    "breakpoint_dropout": ("model", "breakpoint_dropout"),
    "transformer_depth": ("model", "transformer_depth"),
    "transformer_heads": ("model", "transformer_heads"),
    "transformer_mlp_ratio": ("model", "transformer_mlp_ratio"),
    "attention_dropout": ("model", "attention_dropout"),
}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_train_config(config_path=None):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return config
    with open(config_path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("YAML training config must contain a mapping at the top level")
    return _deep_merge(config, loaded)


def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_cli_overrides(config, args):
    merged = copy.deepcopy(config)
    for arg_name, path in CLI_CONFIG_PATHS.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            _set_nested(merged, path, value)
    return merged


def _set_nested(config, path, value):
    cursor = config
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value


def validate_train_config(config):
    missing = []
    if not config.get("dataset_path"):
        missing.append("dataset_path")
    if not config.get("output_path"):
        missing.append("output_path")
    if config.get("training", {}).get("epochs") is None:
        missing.append("training.epochs")
    if missing:
        raise ValueError(
            "Missing required training config value(s): "
            + ", ".join(missing)
            + ". Provide them in YAML or via CLI flags."
        )
    training = config.get("training", {})
    loss = str(training.get("loss", DEFAULT_LOSS)).lower()
    if loss not in {"tb", "subtb", "fl_subtb"}:
        raise ValueError("training.loss must be one of 'tb', 'subtb', or 'fl_subtb'")
    subtb_lambda = float(training.get("subtb_lambda", DEFAULT_SUBTB_LAMBDA))
    if subtb_lambda <= 0.0:
        raise ValueError("training.subtb_lambda must be positive")


def config_to_train_kwargs(config):
    training = config["training"]
    environment = config["environment"]
    reward = config["reward"]
    model = config["model"]
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "dataset_path": config["dataset_path"],
        "output_path": config["output_path"],
        "device": device,
        "bp_per_blocks": environment["bp_per_blocks"],
        "batch_size": training["batch_size"],
        "epochs_num": training["epochs"],
        "seed": training["seed"],
        "init_z_sample_count": training["init_z_sample_count"],
        "use_wandb": training["wandb"],
        "effective_population_size": environment["effective_population_size"],
        "mutation_rate": environment["mutation_rate"],
        "recombination_rate": environment["recombination_rate"],
        "policy_lr": training["policy_lr"],
        "log_z_lr": training["log_z_lr"],
        "loss_mode": str(training["loss"]).lower(),
        "subtb_lambda": training["subtb_lambda"],
        "grad_clip": training["grad_clip"],
        "grad_accum_steps": training["grad_accum_steps"],
        "eval_episodes": training["eval_episodes"],
        "eval_every": training["eval_every"],
        "time_bins": environment["time_bins"],
        "time_delta_bin_width": environment["time_delta_bin_width"],
        "reward_C": reward["constant"],
        "embedding_size": model["embedding_size"],
        "hidden_size": model["hidden_size"],
        "dropout": model["dropout"],
        "breakpoint_hidden_dim": model["breakpoint_hidden_dim"],
        "breakpoint_dropout": model["breakpoint_dropout"],
        "transformer_depth": model["transformer_depth"],
        "transformer_heads": model["transformer_heads"],
        "transformer_mlp_ratio": model["transformer_mlp_ratio"],
        "attention_dropout": model["attention_dropout"],
        "verbose": training["verbose"],
    }


def save_resolved_config(config, output_path):
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

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
            initial_state = env.get_initial_state()
            initial_event_probs = env.compute_event_probabilities(initial_state)
            if generator.loss_mode == "fl_subtb":
                corrections = torch.tensor(
                    [
                        float(getattr(path[-1], "terminal_partial_correction", 0.0))
                        for path in outputs["trajectory_states"]
                    ],
                    dtype=log_pf.dtype,
                    device=log_pf.device,
                )
                loss_metrics = {
                    "eval_fl_subtb_mse": float(
                        generator.compute_subtb_loss_from_rollout_outputs(outputs)
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "eval_log_f0": float(
                        generator.compute_root_log_flow().detach().cpu().item()
                    ),
                    "eval_terminal_partial_correction_mean": float(
                        corrections.mean().detach().cpu().item()
                    ),
                    "eval_terminal_partial_correction_abs_mean": float(
                        corrections.abs().mean().detach().cpu().item()
                    ),
                }
            elif generator.loss_mode == "subtb":
                loss_metrics = {
                    "eval_subtb_mse": float(
                        generator.compute_subtb_loss_from_rollout_outputs(outputs)
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "eval_log_f0": float(
                        generator.compute_root_log_flow().detach().cpu().item()
                    ),
                }
            else:
                residuals = generator.compute_log_Z().detach().to(log_pf) + log_pf - (
                    log_rewards + log_pb
                )
                loss_metrics = {
                    "eval_tb_mse": float(residuals.pow(2).mean().detach().cpu().item()),
                    "eval_residual_mean": float(residuals.mean().detach().cpu().item()),
                    "eval_residual_std": float(
                        residuals.std(unbiased=False).detach().cpu().item()
                    ),
                }

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
            **loss_metrics,
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
    use_wandb=True,
    effective_population_size=DEFAULT_NE,
    mutation_rate=DEFAULT_MU_PER_BP,
    recombination_rate=DEFAULT_R_PER_BP,
    policy_lr=DEFAULT_POLICY_LR,
    log_z_lr=DEFAULT_LOG_Z_LR,
    loss_mode=DEFAULT_LOSS,
    subtb_lambda=DEFAULT_SUBTB_LAMBDA,
    grad_clip=DEFAULT_GRAD_CLIP,
    grad_accum_steps=DEFAULT_GRAD_ACCUM_STEPS,
    eval_episodes=DEFAULT_EVAL_EPISODES,
    eval_every=DEFAULT_EVAL_EVERY,
    time_bins=DEFAULT_TIME_BINS,
    time_delta_bin_width=DEFAULT_TIME_DELTA_BIN_WIDTH,
    reward_C=30000,
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
    loss_mode = str(loss_mode).lower()

    variant_data = None
    if is_vcf_path(dataset_path):
        input_mode = "vcf"
        variant_data = load_vcf_variants(dataset_path)
        sequences = None
        sequence_length = int(variant_data.sequence_length)
        num_sequences = int(variant_data.num_haplotypes)
    else:
        input_mode = "dense"
        sequences = load_sequences(dataset_path)
        sequence_length = len(sequences[0])
        num_sequences = len(sequences)

    env = SimpleARGEnvironment(
        sequence_length=sequence_length,
        num_sequences=num_sequences,
        bp_per_blocks = bp_per_blocks,
        sequences=sequences,
        variant_data=variant_data,
        device=device,
        recombination_rate=recombination_rate,
        population_size=effective_population_size,
        mutation_rate=mutation_rate,
        time_bins=time_bins,
        time_delta_bin_width=time_delta_bin_width,
        reward_C=reward_C,
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
        loss_mode=loss_mode,
        subtb_lambda=subtb_lambda,
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
    
    print(f"use_wandb: {use_wandb}")
    if use_wandb:
        wandb_run = wandb.init()
        wandb.config.update({
            "device": str(generator.device),
            "input_mode": input_mode,
            **env.time_metadata,
            "effective_population_size": float(effective_population_size),
            "mutation_rate": float(mutation_rate),
            "recombination_rate": float(recombination_rate),
            "policy_lr": float(policy_lr),
            "log_z_lr": float(log_z_lr),
            "loss": loss_mode,
            "subtb_lambda": float(subtb_lambda),
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
            if generator.loss_mode in {"subtb", "fl_subtb"}:
                with torch.no_grad():
                    log_z = generator.compute_root_log_flow().detach().cpu().reshape(-1)[0].item()
            else:
                log_z = generator.compute_log_Z().detach().cpu().reshape(-1)[0].item()
            if info is None:
                continue

            info = dict(info)
            info["epoch"] = epoch
            info["log_z"] = log_z
            if generator.loss_mode in {"subtb", "fl_subtb"}:
                info["log_f0"] = log_z
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
                    variant_data=variant_data,
                    dataset_path=dataset_path,
                    input_mode=input_mode,
                    sequence_length=sequence_length,
                    bp_per_blocks=bp_per_blocks,
                    time_metadata=env.time_metadata,
                    reward_C=reward_C,
                    rho=env.rho,
                    effective_population_size=effective_population_size,
                    mutation_rate=mutation_rate,
                    recombination_rate=recombination_rate,
                    policy_lr=policy_lr,
                    log_z_lr=log_z_lr,
                    loss_mode=loss_mode,
                    subtb_lambda=subtb_lambda,
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
            elif "eval_subtb_mse" in info:
                eval_text = f" eval_subtb_mse={info['eval_subtb_mse']:.4f}"
            elif "eval_fl_subtb_mse" in info:
                eval_text = f" eval_fl_subtb_mse={info['eval_fl_subtb_mse']:.4f}"
            log_label = "logF0" if generator.loss_mode in {"subtb", "fl_subtb"} else "logZ"
            print(f"Epoch {epoch + 1} loss={loss:.4f} {log_label}={log_z:.4f}{eval_text}")

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
    variant_data,
    dataset_path,
    input_mode,
    sequence_length,
    bp_per_blocks,
    time_metadata,
    reward_C,
    rho,
    effective_population_size,
    mutation_rate,
    recombination_rate,
    policy_lr,
    log_z_lr,
    loss_mode,
    subtb_lambda,
    grad_clip,
    grad_accum_steps,
    eval_episodes,
    eval_every,
    model_kwargs,
    seed,
    init_z_sample_count,
    model_version,
):
    num_blocks = int(
        variant_data.num_variants
        if input_mode == "vcf"
        else sequence_length // bp_per_blocks
    )
    metadata = {
        "epoch": int(epoch),
        "best_loss": float(best_loss),
        "log_z": float(log_z),
        "input_mode": str(input_mode),
        "dataset_path": os.path.abspath(str(dataset_path)),
        "num_sequences": int(
            variant_data.num_haplotypes
            if input_mode == "vcf"
            else len(sequences)
        ),
        "sequence_length": int(sequence_length),
        "num_blocks": int(num_blocks),
        "bp_per_blocks": int(bp_per_blocks),
        "rho": float(rho),
        "time": dict(time_metadata),
        **dict(time_metadata),
        "reward_C": float(reward_C),
        "effective_population_size": float(effective_population_size),
        "mutation_rate": float(mutation_rate),
        "recombination_rate": float(recombination_rate),
        "policy_lr": float(policy_lr),
        "log_z_lr": float(log_z_lr),
        "loss": str(loss_mode),
        "subtb_lambda": float(subtb_lambda),
        "grad_clip": float(grad_clip),
        "grad_accum_steps": int(grad_accum_steps),
        "eval_episodes": int(eval_episodes),
        "eval_every": int(eval_every),
        "model": dict(model_kwargs),
        "seed": int(seed),
        "init_z_sample_count": int(init_z_sample_count),
        "model_version": str(model_version),
    }
    if input_mode == "vcf":
        metadata.update({
            "num_variants": int(variant_data.num_variants),
            "sample_ids": list(variant_data.sample_ids),
            "haplotype_ids": list(variant_data.haplotype_ids),
            "vcf_parser_version": VCF_PARSER_VERSION,
        })
        return metadata

    metadata["sequences"] = list(sequences)
    return {
        **metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the simplified ARG GFlowNet demo.")
    parser.add_argument("--config", "-c", help="Path to YAML training config.")
    parser.add_argument("--output-path")
    parser.add_argument("--dataset-path")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--bp-per-blocks",
        type=int,
        help="Number of bp per block",
    )
    parser.add_argument("--init-z-sample-count", type=int)
    parser.add_argument("--verbose", action="store_true", default=None)
    parser.add_argument("--effective-population-size", type=float)
    parser.add_argument("--mutation-rate", type=float)
    parser.add_argument("--recombination-rate", type=float)
    parser.add_argument("--policy-lr", type=float)
    parser.add_argument("--log-z-lr", type=float)
    parser.add_argument("--loss", choices=["tb", "subtb", "fl_subtb"])
    parser.add_argument("--subtb-lambda", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--time-bins", type=int)
    parser.add_argument("--time-delta-bin-width", type=float)
    parser.add_argument("--reward-constant", type=float)
    parser.add_argument("--embedding-size", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--breakpoint-hidden-dim", type=int)
    parser.add_argument("--breakpoint-dropout", type=float)
    parser.add_argument("--transformer-depth", type=int)
    parser.add_argument("--transformer-heads", type=int)
    parser.add_argument("--transformer-mlp-ratio", type=float)
    parser.add_argument("--attention-dropout", type=float)
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=None)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    args = parser.parse_args()

    config = load_train_config(args.config)
    config = apply_cli_overrides(config, args)
    validate_train_config(config)
    save_resolved_config(config, config["output_path"])
    train_kwargs = config_to_train_kwargs(config)

    print(f"Selected device: {train_kwargs['device']}")
    train(**train_kwargs)


if __name__ == "__main__":
    main()
