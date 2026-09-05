import argparse
import json
import os

import torch

from arg_environment import SimpleARGEnvironment
from rollout_worker_arg import RolloutWorker
from gflownet import TBGFlowNetGenerator
from gflownet.checkpoint import load_checkpoint
from arg_environment.time import DEFAULT_TIME_BIN_SCHEME
from training.config import (
    DEFAULT_LOG_Z_LR,
    MODEL_VERSION,
    DEFAULT_MU_PER_BP,
    DEFAULT_NE,
    MAX_CONVERGENCE_ABS_RESIDUAL_MEAN,
    MAX_CONVERGENCE_RESIDUAL_RMSE,
    MIN_CONVERGENCE_ESS_FRACTION,
    MIN_CONVERGENCE_EVAL_EPISODES,
    MIN_CONVERGENCE_REQUIRED_PASSES,
)
from training.loop import seed_everything
from training.evaluation import compute_importance_diagnostics
from utils import resolve_device


REQUIRED_METADATA_KEYS = {
    "sequences",
    "num_sequences",
    "sequence_length",
    "num_blocks",
    "rho",
    "time_bin_scheme",
    "time_bins",
    "time_delta_bin_width",
    "seed",
    "init_z_sample_count",
    "model_version",
}


def run_inference(
    checkpoint,
    output_dir="inferred_args",
    num_args=1,
    batch_size=1,
    seed=None,
    device="auto",
    temperature=None,
    verbose=False,
    allow_unconverged=False,
):
    if num_args < 1:
        raise ValueError("num_args must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    checkpoint_data = load_checkpoint(checkpoint, map_location="cpu")
    metadata = checkpoint_data.get("metadata", {})
    validate_metadata(metadata, allow_unconverged=allow_unconverged)

    inference_seed = int(metadata["seed"] if seed is None else seed)
    seed_everything(inference_seed)

    env, generator = build_inference_components(
        metadata, inference_seed, resolve_device(device), verbose,
    )
    generator.load(checkpoint_data, load_optimizer=False)
    generator.eval()

    random_spec = build_random_spec(temperature=temperature)
    rollout_outputs, trajectories = run_batched_rollouts(
        RolloutWorker(env, verbose=verbose),
        generator,
        num_args=num_args,
        batch_size=batch_size,
        random_spec=random_spec,
        verbose=verbose,
    )
    inference_diagnostics = compute_importance_diagnostics(
        [state.log_reward for state in rollout_outputs["states"]],
        rollout_outputs["log_paths_pf"].sum(-1),
        rollout_outputs["log_paths_pb"].sum(-1),
        log_z=generator.compute_log_Z().detach(),
    )

    os.makedirs(output_dir, exist_ok=True)
    manifest = build_manifest(
        checkpoint, metadata, inference_seed, random_spec, output_dir,
        env, rollout_outputs, trajectories,
        allow_unconverged=allow_unconverged,
        inference_diagnostics=inference_diagnostics,
    )
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def build_inference_components(metadata, seed, device, verbose=False):
    env = environment_from_metadata(
        metadata,
        seed=seed,
        device=device,
    )
    generator = TBGFlowNetGenerator(
        env,
        init_z_sample_count=metadata["init_z_sample_count"],
        device=device,
        verbose=verbose,
        log_z_lr=float(metadata.get("log_z_lr", DEFAULT_LOG_Z_LR)),
        model_kwargs=dict(metadata.get("model", {})),
        initialize_z_from_prior=False,
    )
    return env, generator


def validate_metadata(metadata, *, allow_unconverged=False):
    missing = sorted(REQUIRED_METADATA_KEYS - set(metadata))
    if missing:
        raise ValueError(
            "Checkpoint metadata is missing fields required for inference: "
            + ", ".join(missing)
        )
    if metadata["time_bin_scheme"] != DEFAULT_TIME_BIN_SCHEME:
        raise ValueError(
            "This inference path requires fixed-delta time-bin checkpoints "
            f"({DEFAULT_TIME_BIN_SCHEME}), got {metadata['time_bin_scheme']!r}."
        )
    model_version = metadata["model_version"]
    is_legacy_v7 = model_version == "pytorch-transformer-yaml-v7"
    if model_version != MODEL_VERSION and not (allow_unconverged and is_legacy_v7):
        raise ValueError(
            "Checkpoint model_version is incompatible with the current model architecture: "
            f"expected {MODEL_VERSION!r}, got {model_version!r}."
        )
    if not _has_certified_convergence(metadata.get("convergence")):
        if not allow_unconverged:
            raise ValueError(
                "Checkpoint has not passed the convergence/ESS gate. Use "
                "--allow-unconverged only for diagnostic proposal sampling."
            )


def _has_certified_convergence(convergence):
    if not isinstance(convergence, dict):
        return False
    metrics = convergence.get("metrics")
    if not isinstance(metrics, dict):
        return False
    required = ("importance_ess_fraction", "residual_mean", "residual_rmse")
    if any(name not in metrics for name in required):
        return False
    try:
        return (
            convergence.get("version") == 1
            and convergence.get("evaluated") is True
            and convergence.get("passed") is True
            and int(convergence.get("eval_episodes", 0)) >= MIN_CONVERGENCE_EVAL_EPISODES
            and int(convergence.get("consecutive_passes", 0))
            >= MIN_CONVERGENCE_REQUIRED_PASSES
            and float(metrics["importance_ess_fraction"])
            >= MIN_CONVERGENCE_ESS_FRACTION
            and abs(float(metrics["residual_mean"]))
            <= MAX_CONVERGENCE_ABS_RESIDUAL_MEAN
            and float(metrics["residual_rmse"])
            <= MAX_CONVERGENCE_RESIDUAL_RMSE
        )
    except (TypeError, ValueError):
        return False


def environment_from_metadata(metadata, seed, device=None):
    env_kwargs = {
        "num_sequences": int(metadata["num_sequences"]),
        "sequence_length": int(metadata["sequence_length"]),
        "num_blocks": int(metadata["num_blocks"]),
        "rho": float(metadata["rho"]),
        "time_bins": int(metadata["time_bins"]),
        "time_delta_bin_width": float(metadata["time_delta_bin_width"]),
        "population_size": float(
            metadata.get("effective_population_size", DEFAULT_NE)
        ),
        "mutation_rate": float(metadata.get("mutation_rate", DEFAULT_MU_PER_BP)),
        "reward_offset": float(metadata.get(
            "reward_offset",
            30_000.0
            if metadata.get("model_version") == "pytorch-transformer-yaml-v7"
            else 0.0,
        )),
        "sequences": list(metadata["sequences"]),
        "seed": seed,
    }
    if device is not None:
        env_kwargs["device"] = device
    return SimpleARGEnvironment(**env_kwargs)


def run_batched_rollouts(
    rollout_worker,
    generator,
    num_args,
    batch_size,
    random_spec,
    verbose=False,
):
    states = []
    trajectories = []
    log_paths_pf_rows = []
    log_paths_pb_rows = []

    with torch.no_grad():
        for start in range(0, num_args, batch_size):
            chunk_size = min(batch_size, num_args - start)
            end = start + chunk_size
            if verbose:
                print(
                    f"Running ARG rollout chunk {start + 1}-{end} of {num_args} "
                    f"(batch_size={chunk_size})",
                    flush=True,
                )
            chunk_outputs, chunk_trajectories = rollout_worker.rollout(
                generator,
                episodes=chunk_size,
                random_spec=random_spec,
                return_states=True,
            )
            states.extend(chunk_outputs["states"])
            trajectories.extend(chunk_trajectories)
            log_paths_pf_rows.extend(
                row.detach().cpu() for row in chunk_outputs["log_paths_pf"].unbind(0)
            )
            log_paths_pb_rows.extend(
                row.detach().cpu() for row in chunk_outputs["log_paths_pb"].unbind(0)
            )
            del chunk_outputs, chunk_trajectories
            if generator.device.type == "cuda":
                torch.cuda.empty_cache()
            if verbose:
                print(
                    f"Completed ARG rollout chunk {start + 1}-{end} of {num_args}",
                    flush=True,
                )

    rollout_outputs = {
        "states": states,
        "log_paths_pf": _pad_log_path_rows(log_paths_pf_rows),
        "log_paths_pb": _pad_log_path_rows(log_paths_pb_rows),
    }
    return rollout_outputs, trajectories


def _pad_log_path_rows(rows):
    if not rows:
        return torch.empty(0, 0)
    return torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)


def build_random_spec(temperature=None):
    if temperature is not None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        return {"T": float(temperature)}
    return None


def build_manifest(
    checkpoint,
    metadata,
    seed,
    random_spec,
    output_dir,
    env,
    rollout_outputs,
    trajectories,
    *,
    allow_unconverged=False,
    inference_diagnostics=None,
):
    states = rollout_outputs["states"]
    log_paths_pf = rollout_outputs["log_paths_pf"].detach().cpu()
    log_paths_pb = rollout_outputs["log_paths_pb"].detach().cpu()

    records = []
    for idx, state in enumerate(states):
        output_path = os.path.join(output_dir, f"arg_{idx + 1:06d}.trees")
        tree_sequence = env.save_to_tree_sequence(state, output_path=output_path)
        segments = env.get_arg_sequence_segments(state)
        records.append(
            {
                "index": idx,
                "output_file": output_path,
                "log_reward": float(state.log_reward),
                "accumulated_log_prior": float(state.accumulated_log_prior),
                "log_path_pf": float(log_paths_pf[idx].sum().item()),
                "log_path_pb": float(log_paths_pb[idx].sum().item()),
                "trajectory_length": len(trajectories[idx]),
                "breakpoints": segments["breakpoints"],
                "segment_count": segments["num_segments"],
                "num_recombination_events": len(segments["recombination_events"]),
                "num_trees": int(tree_sequence.num_trees),
                "num_edges": int(tree_sequence.num_edges),
            }
        )

    return {
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_epoch": int(metadata["epoch"]) if "epoch" in metadata else None,
        "checkpoint_best_loss": (
            float(metadata["best_loss"]) if "best_loss" in metadata else None
        ),
        "checkpoint_kind": metadata.get("checkpoint_kind"),
        "checkpoint_selection_metric": metadata.get("checkpoint_selection_metric"),
        "checkpoint_selection_value": metadata.get("checkpoint_selection_value"),
        "checkpoint_model_version": metadata.get("model_version"),
        "checkpoint_convergence": metadata.get("convergence"),
        "allow_unconverged": bool(allow_unconverged),
        "inference_diagnostics": dict(inference_diagnostics or {}),
        "seed": int(seed),
        "num_args": len(records),
        "random_spec": random_spec,
        "outputs": records,
    }


def main():
    parser = argparse.ArgumentParser(description="Infer ARGs from a saved ARG GFlowNet checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="inferred_args")
    parser.add_argument(
        "--num-args",
        "--num-particles",
        dest="num_args",
        type=int,
        default=1,
        help="Total number of ARGs/particles to generate (default: 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of ARG rollouts to process simultaneously on the GPU (default: 1).",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--allow-unconverged",
        action="store_true",
        help="Allow diagnostic sampling from a failed, unevaluated, or legacy v7 checkpoint.",
    )
    args = parser.parse_args()

    manifest = run_inference(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        num_args=args.num_args,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        temperature=args.temperature,
        verbose=args.verbose,
        allow_unconverged=args.allow_unconverged,
    )
    print(f"Wrote {manifest['num_args']} ARG tree sequence(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
