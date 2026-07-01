import argparse
import json
import os

import torch

from env import SimpleARGEnvironment
from refinement import (
    build_refinement_contexts,
    build_refinement_source,
    parse_block_groups,
    parse_bp_intervals,
)
from rollout_worker_arg import RolloutWorker
from tb_gfn import TBGFlowNetGenerator
from time_env import DEFAULT_TIME_BIN_SCHEME
from train import (
    DEFAULT_LOSS,
    DEFAULT_LOG_Z_LR,
    DEFAULT_SUBTB_LAMBDA,
    MODEL_VERSION,
    DEFAULT_MU_PER_BP,
    DEFAULT_NE,
    seed_everything,
)
from utils import is_vcf_path, load_vcf_variants


REQUIRED_METADATA_KEYS = {
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

VCF_METADATA_KEYS = {
    "dataset_path",
    "num_variants",
    "sample_ids",
    "haplotype_ids",
}

DENSE_METADATA_KEYS = {"sequences"}


def run_inference(
    checkpoint,
    output_dir="inferred_args",
    num_args=1,
    batch_size=1,
    seed=None,
    device="auto",
    temperature=None,
    verbose=False,
    dataset_path=None,
    refine_arg=None,
    bad_region_top_k=None,
    bad_region_blocks=None,
    bad_region_bp=None,
    refine_strategy="before_last_coalescence",
):
    if num_args < 1:
        raise ValueError("num_args must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    checkpoint_data = load_checkpoint(checkpoint, map_location="cpu")
    metadata = checkpoint_data.get("metadata", {})
    validate_metadata(metadata)

    inference_seed = int(metadata["seed"] if seed is None else seed)
    seed_everything(inference_seed)

    resolved_device = resolve_device(device)
    env = environment_from_metadata(
        metadata,
        seed=inference_seed,
        device=resolved_device,
        dataset_path=dataset_path,
    )
    generator = TBGFlowNetGenerator(
        env,
        init_z_sample_count=metadata["init_z_sample_count"],
        cfg={
            "breakpoint_policy": metadata.get("breakpoint_policy", "continuous-bin"),
            "breakpoint_mixtures": int(metadata.get("breakpoint_mixtures", 4)),
        },
        device=resolved_device,
        verbose=verbose,
        log_z_lr=float(metadata.get("log_z_lr", DEFAULT_LOG_Z_LR)),
        model_kwargs=dict(metadata.get("model", {})),
        initialize_z_from_prior=False,
        loss_mode=str(metadata.get("loss", DEFAULT_LOSS)),
        subtb_lambda=float(metadata.get("subtb_lambda", DEFAULT_SUBTB_LAMBDA)),
    )
    generator.load(checkpoint_data, load_optimizer=False, map_location=generator.device)
    generator.eval()

    random_spec = build_random_spec(temperature=temperature)
    rollout_worker = RolloutWorker(env, verbose=verbose)
    if refine_arg is not None:
        return run_refinement_inference(
            checkpoint=checkpoint,
            metadata=metadata,
            seed=inference_seed,
            random_spec=random_spec,
            output_dir=output_dir,
            env=env,
            source_env=environment_from_metadata(
                metadata,
                seed=inference_seed,
                device=torch.device("cpu"),
                dataset_path=dataset_path,
            ),
            rollout_worker=rollout_worker,
            generator=generator,
            refine_arg=refine_arg,
            dataset_path=dataset_path or metadata.get("dataset_path"),
            num_args=num_args,
            batch_size=batch_size,
            bad_region_top_k=bad_region_top_k,
            bad_region_blocks=bad_region_blocks,
            bad_region_bp=bad_region_bp,
            refine_strategy=refine_strategy,
            verbose=verbose,
        )

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


def resolve_device(device):
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested for inference but is not available.")
    return resolved


def validate_metadata(metadata):
    input_mode = metadata.get("input_mode", "dense")
    required = set(REQUIRED_METADATA_KEYS)
    if input_mode == "vcf":
        required |= VCF_METADATA_KEYS
    else:
        required |= DENSE_METADATA_KEYS
    missing = sorted(required - set(metadata))
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
    if metadata["model_version"] != MODEL_VERSION:
        raise ValueError(
            "Checkpoint model_version is incompatible with this sparse VCF implementation: "
            f"expected {MODEL_VERSION!r}, got {metadata['model_version']!r}."
        )


def environment_from_metadata(metadata, seed, device=None, dataset_path=None):
    input_mode = metadata.get("input_mode", "dense")
    variant_data = None
    sequences = None
    if input_mode == "vcf":
        resolved_dataset_path = dataset_path or metadata["dataset_path"]
        if not is_vcf_path(resolved_dataset_path):
            raise ValueError("VCF checkpoints require a .vcf or .vcf.gz dataset path")
        variant_data = load_vcf_variants(resolved_dataset_path)
        if int(variant_data.num_variants) != int(metadata["num_variants"]):
            raise ValueError("VCF variant count does not match checkpoint metadata")
        if int(variant_data.sequence_length) != int(metadata["sequence_length"]):
            raise ValueError("VCF sequence length does not match checkpoint metadata")
        if list(variant_data.haplotype_ids) != list(metadata["haplotype_ids"]):
            raise ValueError("VCF haplotype IDs do not match checkpoint metadata")
    else:
        sequences = list(metadata["sequences"])

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
        "sequences": sequences,
        "variant_data": variant_data,
        "reward_C": float(metadata.get("reward_C", 30000.0)),
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
    start_state=None,
    action_filter=None,
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
                start_states=(
                    [start_state for _ in range(chunk_size)]
                    if start_state is not None
                    else None
                ),
                action_filter=action_filter,
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


def run_refinement_inference(
    checkpoint,
    metadata,
    seed,
    random_spec,
    output_dir,
    env,
    source_env,
    rollout_worker,
    generator,
    refine_arg,
    dataset_path,
    num_args,
    batch_size,
    bad_region_top_k,
    bad_region_blocks,
    bad_region_bp,
    refine_strategy,
    verbose=False,
):
    if not dataset_path:
        raise ValueError("refinement inference requires a VCF dataset path")
    source = build_refinement_source(
        source_env,
        refine_arg,
        dataset_path,
        population_size=float(metadata.get("effective_population_size", DEFAULT_NE)),
        mutation_rate=float(metadata.get("mutation_rate", DEFAULT_MU_PER_BP)),
    )
    contexts, diagnostic_rows = build_refinement_contexts(
        source,
        top_k=bad_region_top_k,
        block_groups=parse_block_groups(bad_region_blocks),
        bp_intervals=parse_bp_intervals(bad_region_bp),
        strategy=refine_strategy,
    )
    if not contexts:
        raise ValueError("no local refinement contexts were selected")

    os.makedirs(output_dir, exist_ok=True)
    region_records = []
    total_outputs = 0
    for context_idx, context in enumerate(contexts, start=1):
        region_dir = os.path.join(output_dir, f"region_{context_idx:06d}")
        os.makedirs(region_dir, exist_ok=True)
        if verbose:
            print(
                f"Refining region {context_idx}/{len(contexts)} "
                f"blocks={list(context.region.blocks)} "
                f"effective_blocks={list(context.effective_blocks)}",
                flush=True,
            )
        rollout_outputs, trajectories = run_batched_rollouts(
            rollout_worker,
            generator,
            num_args=num_args,
            batch_size=batch_size,
            random_spec=random_spec,
            verbose=verbose,
            start_state=context.partial_state,
            action_filter=context.action_filter(),
        )
        region_manifest = build_manifest(
            checkpoint=checkpoint,
            metadata=metadata,
            seed=seed,
            random_spec=random_spec,
            output_dir=region_dir,
            env=env,
            rollout_outputs=rollout_outputs,
            trajectories=trajectories,
        )
        region_manifest["refinement_context"] = context.to_manifest_record()
        with open(
            os.path.join(region_dir, "manifest.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(region_manifest, handle, indent=2)
        region_records.append(
            {
                **context.to_manifest_record(),
                "output_dir": region_dir,
                "outputs": region_manifest["outputs"],
            }
        )
        total_outputs += len(region_manifest["outputs"])

    manifest = {
        "mode": "local_refinement",
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_epoch": int(metadata["epoch"]) if "epoch" in metadata else None,
        "checkpoint_best_loss": (
            float(metadata["best_loss"]) if "best_loss" in metadata else None
        ),
        "seed": int(seed),
        "source_arg": os.path.abspath(refine_arg),
        "dataset_path": os.path.abspath(dataset_path),
        "num_regions": len(region_records),
        "num_args_per_region": int(num_args),
        "num_outputs": int(total_outputs),
        "random_spec": random_spec,
        "refine_strategy": str(refine_strategy),
        "regions": region_records,
        "diagnostics": diagnostic_rows,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


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
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--dataset-path", help="Override the dataset path stored in checkpoint metadata.")
    parser.add_argument(
        "--refine-arg",
        help="Existing .trees ARG to backtrack and locally refine.",
    )
    parser.add_argument(
        "--bad-region-top-k",
        type=int,
        help="Automatically refine the top K suspicious blocks.",
    )
    parser.add_argument(
        "--bad-region-blocks",
        help="Manual block groups, e.g. '1,2,3;8-10'.",
    )
    parser.add_argument(
        "--bad-region-bp",
        help="Manual BP intervals, e.g. '1000-2500;9000-11000'.",
    )
    parser.add_argument(
        "--refine-strategy",
        default="before_last_coalescence",
        choices=["before_last_touch", "before_first_touch", "before_last_coalescence"],
    )
    parser.add_argument("--verbose", action="store_true")
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
        dataset_path=args.dataset_path,
        refine_arg=args.refine_arg,
        bad_region_top_k=args.bad_region_top_k,
        bad_region_blocks=args.bad_region_blocks,
        bad_region_bp=args.bad_region_bp,
        refine_strategy=args.refine_strategy,
    )
    print(
        f"Wrote {manifest.get('num_outputs', manifest.get('num_args', 0))} "
        f"ARG tree sequence(s) to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
