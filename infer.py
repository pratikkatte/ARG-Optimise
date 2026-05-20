import argparse
import json
import os
import random

import torch

from env import SimpleARGEnvironment
from rollout_worker_arg import RolloutWorker
from tb_gfn import TBGFlowNetGenerator
from train import seed_everything


REQUIRED_METADATA_KEYS = {
    "sequences",
    "num_sequences",
    "sequence_length",
    "num_blocks",
    "rho",
    "fixed_edge_length",
    "seed",
    "init_z_sample_count",
}


def run_inference(
    checkpoint,
    output_dir="inferred_args",
    num_args=1,
    batch_size=1,
    seed=None,
    device="auto",
    random_action_prob=0.0,
    temperature=None,
    verbose=False,
):
    if num_args < 1:
        raise ValueError("num_args must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if temperature is not None and random_action_prob != 0.0:
        raise ValueError("temperature and random_action_prob cannot both be set")

    checkpoint_data = load_checkpoint(checkpoint, map_location="cpu")
    metadata = checkpoint_data.get("metadata", {})
    validate_metadata(metadata)

    inference_seed = int(metadata["seed"] if seed is None else seed)
    seed_everything(inference_seed)

    env = environment_from_metadata(metadata, seed=inference_seed)
    generator = TBGFlowNetGenerator(
        env,
        init_z_sample_count=metadata["init_z_sample_count"],
        device=device,
        verbose=verbose,
    )
    generator.load(checkpoint, load_optimizer=False, map_location=generator.device)
    generator.eval()

    random_spec = build_random_spec(
        random_action_prob=random_action_prob,
        temperature=temperature,
    )
    rollout_worker = RolloutWorker(env, verbose=verbose)
    rollout_outputs, trajectories = run_batched_rollouts(
        rollout_worker,
        generator,
        num_args=num_args,
        batch_size=batch_size,
        random_spec=random_spec,
        verbose=verbose,
    )

    os.makedirs(output_dir, exist_ok=True)
    manifest = build_manifest(
        checkpoint=checkpoint,
        metadata=metadata,
        seed=inference_seed,
        random_spec=random_spec,
        output_dir=output_dir,
        env=env,
        rollout_outputs=rollout_outputs,
        trajectories=trajectories,
    )

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def load_checkpoint(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def validate_metadata(metadata):
    missing = sorted(REQUIRED_METADATA_KEYS - set(metadata))
    if missing:
        raise ValueError(
            "Checkpoint metadata is missing fields required for inference: "
            + ", ".join(missing)
        )


def environment_from_metadata(metadata, seed):
    return SimpleARGEnvironment(
        num_sequences=int(metadata["num_sequences"]),
        sequence_length=int(metadata["sequence_length"]),
        num_blocks=int(metadata["num_blocks"]),
        rho=float(metadata["rho"]),
        fixed_edge_length=float(metadata["fixed_edge_length"]),
        sequences=list(metadata["sequences"]),
        rng=random.Random(seed),
    )


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
    max_length = max(row.numel() for row in rows)
    dtype = rows[0].dtype
    padded = torch.zeros(len(rows), max_length, dtype=dtype)
    for row_idx, row in enumerate(rows):
        if row.numel() > 0:
            padded[row_idx, : row.numel()] = row
    return padded


def build_random_spec(random_action_prob=0.0, temperature=None):
    if temperature is not None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        return {"T": float(temperature)}
    if random_action_prob < 0.0 or random_action_prob > 1.0:
        raise ValueError("random_action_prob must be between 0 and 1")
    return {"random_action_prob": float(random_action_prob)}


def build_manifest(
    checkpoint,
    metadata,
    seed,
    random_spec,
    output_dir,
    env,
    rollout_outputs,
    trajectories,
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
    parser.add_argument("--random-action-prob", type=float, default=0.0)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    manifest = run_inference(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        num_args=args.num_args,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        random_action_prob=args.random_action_prob,
        temperature=args.temperature,
        verbose=args.verbose,
    )
    print(f"Wrote {manifest['num_args']} ARG tree sequence(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
