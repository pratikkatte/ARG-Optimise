import argparse
import hashlib
import json
import os
import re

import torch

try:
    from .env import LocalARGEnvironment, SimpleARGEnvironment
    from .rollout_worker_arg import RolloutWorker
    from .tb_gfn import TBGFlowNetGenerator
    from .time_env import DEFAULT_TIME_BIN_SCHEME
    from .train import (
        DEFAULT_LOSS,
        DEFAULT_LOG_Z_LR,
        DEFAULT_SUBTB_LAMBDA,
        MODEL_VERSION,
        DEFAULT_MU_PER_BP,
        DEFAULT_NE,
        seed_everything,
    )
    from .utils import is_vcf_path, load_vcf_variants
    from .validation.paths import experiment_gfn_dir, validate_experiment_name
except ImportError:  # Support the repository's script-style entry points.
    from env import LocalARGEnvironment, SimpleARGEnvironment
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
    from validation.paths import experiment_gfn_dir, validate_experiment_name


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
    output_dir=None,
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
    refine_strategy=None,
    experiment=None,
):
    if num_args < 1:
        raise ValueError("num_args must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    output_dir = resolve_inference_output_dir(
        output_dir=output_dir, experiment=experiment
    )
    checkpoint_data = load_checkpoint(checkpoint, map_location="cpu")
    metadata = checkpoint_data.get("metadata", {})
    validate_metadata(metadata)

    inference_seed = int(metadata["seed"] if seed is None else seed)
    seed_everything(inference_seed)

    resolved_device = resolve_device(device)
    base_env = environment_from_metadata(
        metadata,
        seed=inference_seed,
        device=resolved_device,
        dataset_path=dataset_path,
    )
    is_local_checkpoint = (
        metadata.get("training_mode") == "local_refinement"
    )
    if is_local_checkpoint:
        resolved_dataset_path = dataset_path or metadata["dataset_path"]
        expected_vcf_sha256 = (
            metadata.get("vcf_identity", {}).get("sha256")
        )
        if (
            expected_vcf_sha256
            and _file_sha256(resolved_dataset_path)
            != expected_vcf_sha256
        ):
            raise ValueError(
                "VCF fingerprint does not match the local checkpoint"
            )
    legacy_refinement_values = {
        "bad_region_top_k": bad_region_top_k,
        "bad_region_blocks": bad_region_blocks,
        "bad_region_bp": bad_region_bp,
        "refine_strategy": refine_strategy,
    }
    configured_legacy = sorted(
        key
        for key, value in legacy_refinement_values.items()
        if value is not None
    )
    if configured_legacy:
        raise ValueError(
            "Automatic bad-region inference options are no longer supported: "
            + ", ".join(configured_legacy)
            + ". Train a local checkpoint with explicit refinement.requests."
        )
    if is_local_checkpoint:
        env = local_environment_from_metadata(
            metadata,
            base_env,
            source_arg_path=refine_arg,
        )
    else:
        env = base_env
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
    if is_local_checkpoint:
        return run_local_refinement_inference(
            checkpoint=checkpoint,
            metadata=metadata,
            seed=inference_seed,
            random_spec=random_spec,
            output_dir=output_dir,
            env=env,
            rollout_worker=rollout_worker,
            generator=generator,
            dataset_path=dataset_path or metadata.get("dataset_path"),
            num_args=num_args,
            batch_size=batch_size,
            experiment=experiment,
            verbose=verbose,
        )
    if refine_arg is not None:
        raise ValueError(
            "--refine-arg requires a checkpoint trained with explicit local "
            "refinement requests; global checkpoints cannot perform learned "
            "local refinement directly"
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
        dataset_path=dataset_path or metadata.get("dataset_path"),
        experiment=experiment,
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


def resolve_inference_output_dir(output_dir=None, experiment=None):
    """Resolve the legacy output path or a named validation experiment path."""
    if output_dir is not None and experiment is not None:
        raise ValueError("use either output_dir or experiment, not both")
    if experiment is not None:
        return str(experiment_gfn_dir(validate_experiment_name(experiment)))
    return "inferred_args" if output_dir is None else str(output_dir)


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
    is_local = metadata.get("training_mode") == "local_refinement"
    expected_model_version = (
        "local-arg-gflownet-fl-subtb-v1" if is_local else MODEL_VERSION
    )
    if metadata["model_version"] != expected_model_version:
        raise ValueError(
            "Checkpoint model_version is incompatible with this sparse VCF implementation: "
            f"expected {expected_model_version!r}, got {metadata['model_version']!r}."
        )
    if is_local:
        missing_local = sorted(
            {"source_arg", "refinement_requests"} - set(metadata)
        )
        if missing_local:
            raise ValueError(
                "Local checkpoint metadata is missing: "
                + ", ".join(missing_local)
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


def local_environment_from_metadata(
    metadata,
    base_env,
    source_arg_path=None,
):
    try:
        from .refinement import LocalRefinementRequest, prepare_local_refinement
    except ImportError:
        from refinement import LocalRefinementRequest, prepare_local_refinement

    source_record = metadata["source_arg"]
    resolved_source = source_arg_path or source_record["path"]
    if not os.path.exists(resolved_source):
        raise FileNotFoundError(
            f"local checkpoint source ARG does not exist: {resolved_source}"
        )
    expected_fingerprint = source_record.get("sha256")
    if (
        expected_fingerprint
        and _file_sha256(resolved_source) != expected_fingerprint
    ):
        raise ValueError(
            "source ARG fingerprint does not match the local checkpoint"
        )
    prepared_contexts = {}
    for index, record in enumerate(metadata["refinement_requests"]):
        context_id = str(
            record.get("id") or f"region_{index + 1:06d}"
        )
        if (
            context_id in {".", ".."}
            or re.fullmatch(r"[A-Za-z0-9_.-]+", context_id) is None
        ):
            raise ValueError(
                f"invalid local checkpoint request id {context_id!r}"
            )
        if context_id in prepared_contexts:
            raise ValueError(
                f"duplicate local checkpoint request id {context_id!r}"
            )
        request = LocalRefinementRequest(
            genomic_range=tuple(record["genomic_range"]),
            cut_time=record.get("cut_time"),
            cut_event_index=record.get("cut_event_index"),
        )
        prepared_contexts[context_id] = prepare_local_refinement(
            resolved_source,
            request,
        )
    return LocalARGEnvironment(base_env, prepared_contexts)


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


def run_local_refinement_inference(
    checkpoint,
    metadata,
    seed,
    random_spec,
    output_dir,
    env,
    rollout_worker,
    generator,
    dataset_path,
    num_args,
    batch_size,
    experiment=None,
    verbose=False,
):
    """Sample, splice, validate, and export every checkpoint request."""

    try:
        from .refinement import (
            export_refined_tree_sequence,
            splice_local_proposal,
        )
    except ImportError:
        from refinement import (
            export_refined_tree_sequence,
            splice_local_proposal,
        )

    os.makedirs(output_dir, exist_ok=True)
    request_records = {
        str(record["id"]): record
        for record in metadata["refinement_requests"]
    }
    requests_manifest = []
    total_valid_outputs = 0
    for context_id in env.context_ids:
        request_dir = os.path.join(output_dir, context_id)
        os.makedirs(request_dir, exist_ok=True)
        initial_state = env.get_initial_state(context_id)
        rollout_outputs, trajectories = run_batched_rollouts(
            rollout_worker,
            generator,
            num_args=int(num_args),
            batch_size=int(batch_size),
            random_spec=random_spec,
            verbose=verbose,
            start_state=initial_state,
        )
        prepared = env.prepared_contexts[context_id]
        trajectory_records = []
        for index, (state, trajectory) in enumerate(
            zip(rollout_outputs["states"], trajectories),
            start=1,
        ):
            log_pf = float(
                rollout_outputs["log_paths_pf"][index - 1].sum().item()
            )
            log_pb = float(
                rollout_outputs["log_paths_pb"][index - 1].sum().item()
            )
            record = {
                "index": int(index),
                "terminal": bool(state.is_done),
                "trajectory_length": len(trajectory),
                "log_P_F": log_pf,
                "log_P_B": log_pb,
                "local_cwr_log_prior": float(
                    state.accumulated_log_prior
                ),
                "whole_vcf_log_likelihood": (
                    None
                    if state.log_likelihood is None
                    else float(state.log_likelihood)
                ),
                "log_reward": (
                    None
                    if state.log_reward is None
                    else float(state.log_reward)
                ),
                "actions": list(trajectory.actions),
                "prior_increments": [
                    None if value is None else float(value)
                    for value in trajectory.log_priors
                ],
                "output_file": None,
                "topology_digest": None,
                "splice_validation": None,
                "failure_diagnostics": [],
            }
            try:
                if not state.is_done:
                    raise ValueError(
                        "learned local rollout did not reach a terminal state"
                    )
                proposal = env.state_to_proposal(state)
                record["topology_digest"] = proposal.topology_digest
                if proposal.diagnostics:
                    record["failure_diagnostics"].extend(
                        {
                            "code": diagnostic.code,
                            "message": diagnostic.message,
                            "step": diagnostic.step,
                        }
                        for diagnostic in proposal.diagnostics
                    )
                splice_result = splice_local_proposal(prepared, proposal)
                validation = splice_result.validation
                record["splice_validation"] = {
                    "is_valid": bool(validation.is_valid),
                    "errors": list(validation.errors),
                    "warnings": list(validation.warnings),
                    "counts": dict(validation.counts),
                }
                if not splice_result.is_valid:
                    record["failure_diagnostics"].extend(
                        {
                            "code": "splice_validation",
                            "message": message,
                        }
                        for message in validation.errors
                    )
                else:
                    output_path = os.path.join(
                        request_dir,
                        f"arg_{index:06d}.trees",
                    )
                    export_refined_tree_sequence(
                        splice_result,
                        output_path,
                        overwrite=False,
                    )
                    record["output_file"] = os.path.abspath(output_path)
                    total_valid_outputs += 1
            except Exception as error:
                record["failure_diagnostics"].append(
                    {
                        "code": type(error).__name__,
                        "message": str(error),
                    }
                )
            trajectory_records.append(record)

        request_manifest = {
            "id": context_id,
            "request": request_records[context_id],
            "output_dir": os.path.abspath(request_dir),
            "sample_count": int(num_args),
            "valid_output_count": sum(
                row["output_file"] is not None
                for row in trajectory_records
            ),
            "trajectories": trajectory_records,
        }
        with open(
            os.path.join(request_dir, "manifest.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(request_manifest, handle, indent=2)
        requests_manifest.append(request_manifest)

    manifest = {
        "mode": "local_refinement",
        "experiment": experiment,
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_epoch": metadata.get("epoch"),
        "checkpoint_best_loss": metadata.get("best_loss"),
        "seed": int(seed),
        "source_arg": dict(metadata["source_arg"]),
        "dataset_path": os.path.abspath(dataset_path),
        "output_dir": os.path.abspath(output_dir),
        "num_requests": len(requests_manifest),
        "num_args_per_request": int(num_args),
        "num_outputs": int(total_valid_outputs),
        "output_count": int(total_valid_outputs),
        "random_spec": random_spec,
        "requests": requests_manifest,
    }
    with open(
        os.path.join(output_dir, "manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def _run_refinement_inference_legacy(
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
    experiment=None,
    verbose=False,
):
    from refinement import (
        build_refinement_contexts,
        build_refinement_source,
        parse_block_groups,
        parse_bp_intervals,
    )

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
            dataset_path=dataset_path,
            experiment=experiment,
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
        "experiment": experiment,
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_epoch": int(metadata["epoch"]) if "epoch" in metadata else None,
        "checkpoint_best_loss": (
            float(metadata["best_loss"]) if "best_loss" in metadata else None
        ),
        "seed": int(seed),
        "source_arg": os.path.abspath(refine_arg),
        "dataset_path": os.path.abspath(dataset_path),
        "output_dir": os.path.abspath(output_dir),
        "num_regions": len(region_records),
        "num_args_per_region": int(num_args),
        "num_outputs": int(total_outputs),
        "output_count": int(total_outputs),
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


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    dataset_path=None,
    experiment=None,
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
        "experiment": experiment,
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_epoch": int(metadata["epoch"]) if "epoch" in metadata else None,
        "checkpoint_best_loss": (
            float(metadata["best_loss"]) if "best_loss" in metadata else None
        ),
        "seed": int(seed),
        "dataset_path": (
            os.path.abspath(dataset_path) if dataset_path is not None else None
        ),
        "output_dir": os.path.abspath(output_dir),
        "num_args": len(records),
        "output_count": len(records),
        "random_spec": random_spec,
        "outputs": records,
    }


def _experiment_arg(value):
    try:
        return validate_experiment_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser():
    parser = argparse.ArgumentParser(description="Infer ARGs from a saved ARG GFlowNet checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output-dir",
        default=None,
        help="Explicit output directory (legacy default: inferred_args).",
    )
    output_group.add_argument(
        "--experiment",
        type=_experiment_arg,
        help="Write samples to validation/output/EXPERIMENT/gfn/.",
    )
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
        help="Optional source-ARG override for a local checkpoint; its fingerprint must match.",
    )
    parser.add_argument(
        "--bad-region-top-k",
        type=int,
        help="Deprecated; use explicit checkpoint requests.",
    )
    parser.add_argument(
        "--bad-region-blocks",
        help="Deprecated; use explicit checkpoint requests.",
    )
    parser.add_argument(
        "--bad-region-bp",
        help="Deprecated; use explicit checkpoint requests.",
    )
    parser.add_argument(
        "--refine-strategy",
        default=None,
        choices=["before_last_touch", "before_first_touch", "before_last_coalescence"],
        help="Deprecated; local checkpoints reconstruct explicit requests.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    resolved_output_dir = resolve_inference_output_dir(
        output_dir=args.output_dir, experiment=args.experiment
    )

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
        experiment=args.experiment,
    )
    print(
        f"Wrote {manifest.get('num_outputs', manifest.get('num_args', 0))} "
        f"ARG tree sequence(s) to {resolved_output_dir}"
    )


if __name__ == "__main__":
    main()
