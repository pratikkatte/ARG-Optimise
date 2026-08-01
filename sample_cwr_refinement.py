"""Model-free local ARG refinement by direct CwR-prior sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

try:
    from .env import SimpleARGEnvironment
    from .refinement import (
        LocalRefinementRequest,
        LocalSamplingConfig,
        export_refined_tree_sequence,
        prepare_local_refinement,
        sample_local_trajectories,
        splice_local_proposal,
    )
    from .utils import is_vcf_path, load_vcf_variants
except ImportError:  # Support ``python sample_cwr_refinement.py``.
    package_parent = str(Path(__file__).resolve().parent.parent)
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)
    from arg.env import SimpleARGEnvironment
    from arg.refinement import (
        LocalRefinementRequest,
        LocalSamplingConfig,
        export_refined_tree_sequence,
        prepare_local_refinement,
        sample_local_trajectories,
        splice_local_proposal,
    )
    from arg.utils import is_vcf_path, load_vcf_variants


DEFAULT_SEED = 7
DEFAULT_EFFECTIVE_POPULATION_SIZE = 10_000.0
DEFAULT_MUTATION_RATE = 2e-8
DEFAULT_RECOMBINATION_RATE = 2e-8
DEFAULT_REWARD_CONSTANT = 0.0
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


def run_cwr_refinement_sampler(
    config_path: str | os.PathLike[str],
    *,
    num_trees: int,
    output_dir: str | os.PathLike[str] | None = None,
    seed: int | None = None,
    overwrite: bool = False,
    max_generated_events: int | None = None,
    max_searched_states: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Sample and export model-free local refinements for every config request.

    ``num_trees`` is interpreted per configured refinement request. Each
    construction step is drawn from the local coalescent-with-recombination
    prior; no checkpoint or learned policy is loaded.
    """

    num_trees = int(num_trees)
    if num_trees < 1:
        raise ValueError("num_trees must be at least 1")
    if max_generated_events is not None and int(max_generated_events) < 1:
        raise ValueError("max_generated_events must be at least 1")
    if max_searched_states is not None and int(max_searched_states) < 1:
        raise ValueError("max_searched_states must be at least 1")

    config_path = Path(config_path).expanduser().resolve()
    config = _load_config(config_path)
    refinement = _require_mapping(config.get("refinement"), "refinement")
    environment = _optional_mapping(config.get("environment"), "environment")
    reward = _optional_mapping(config.get("reward"), "reward")
    training = _optional_mapping(config.get("training"), "training")
    requests = _validate_requests(refinement.get("requests"))

    dataset_value = config.get("dataset_path")
    source_arg_value = refinement.get("arg_path")
    if not dataset_value:
        raise ValueError("config dataset_path is required")
    if not source_arg_value:
        raise ValueError("config refinement.arg_path is required")

    config_base = _resolve_config_base(
        config_path,
        (dataset_value, source_arg_value),
    )
    dataset_path = _resolve_input_path(dataset_value, config_base)
    source_arg_path = _resolve_input_path(source_arg_value, config_base)
    if not is_vcf_path(dataset_path):
        raise ValueError(
            "model-free local refinement requires a .vcf or .vcf.gz dataset"
        )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"VCF dataset does not exist: {dataset_path}")
    if not source_arg_path.is_file():
        raise FileNotFoundError(
            f"source ARG tree sequence does not exist: {source_arg_path}"
        )

    root_seed = int(
        training.get("seed", DEFAULT_SEED)
        if seed is None
        else seed
    )
    resolved_output_dir = _resolve_output_dir(
        output_dir,
        config.get("output_path"),
        config_base,
    )
    request_specs = _normalized_request_specs(requests)
    _preflight_output_paths(
        resolved_output_dir,
        [request["id"] for request in request_specs],
        num_trees,
        overwrite=bool(overwrite),
    )

    variant_data = load_vcf_variants(dataset_path)
    env = SimpleARGEnvironment(
        sequence_length=int(variant_data.sequence_length),
        num_sequences=int(variant_data.num_haplotypes),
        bp_per_blocks=int(environment.get("bp_per_blocks", 1)),
        variant_data=variant_data,
        device="cpu",
        recombination_rate=float(
            environment.get(
                "recombination_rate",
                DEFAULT_RECOMBINATION_RATE,
            )
        ),
        population_size=float(
            environment.get(
                "effective_population_size",
                DEFAULT_EFFECTIVE_POPULATION_SIZE,
            )
        ),
        mutation_rate=float(
            environment.get("mutation_rate", DEFAULT_MUTATION_RATE)
        ),
        reward_C=float(reward.get("constant", DEFAULT_REWARD_CONSTANT)),
        seed=root_seed,
    )

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    request_manifests = []
    for request_index, request_spec in enumerate(request_specs):
        context_id = request_spec["id"]
        request_seed = root_seed + request_index
        if verbose:
            print(
                f"Sampling {num_trees} CwR refinement(s) for "
                f"{context_id} with seed {request_seed}",
                flush=True,
            )
        request = LocalRefinementRequest(
            genomic_range=tuple(request_spec["genomic_range"]),
            cut_time=request_spec.get("cut_time"),
            cut_event_index=request_spec.get("cut_event_index"),
        )
        prepared = prepare_local_refinement(source_arg_path, request)
        if not prepared.context.is_valid:
            reasons = "; ".join(
                item.message
                for item in prepared.context.rejection_diagnostics
            )
            raise ValueError(
                f"local refinement request {context_id!r} is invalid: "
                f"{reasons}"
            )

        batch = sample_local_trajectories(
            prepared,
            env,
            LocalSamplingConfig(
                sample_count=num_trees,
                seed=request_seed,
                max_generated_events=max_generated_events,
                max_searched_states=max_searched_states,
                allow_duplicate_topologies=True,
            ),
        )
        if not batch.is_complete or len(batch.proposals) != num_trees:
            details = "; ".join(
                f"{item.code}: {item.message}"
                for item in batch.diagnostics
            )
            raise RuntimeError(
                f"CwR sampling for {context_id!r} produced "
                f"{len(batch.proposals)} of {num_trees} requested trees"
                + (f": {details}" if details else "")
            )

        splice_results = []
        for sample_index, proposal in enumerate(batch.proposals, start=1):
            splice_result = splice_local_proposal(prepared, proposal)
            if not splice_result.is_valid:
                raise RuntimeError(
                    f"CwR sample {sample_index} for {context_id!r} failed "
                    "splice validation: "
                    + "; ".join(splice_result.validation.errors)
                )
            splice_results.append(splice_result)

        request_dir = resolved_output_dir / context_id
        request_dir.mkdir(parents=True, exist_ok=True)
        sample_records = []
        for sample_index, (proposal, trajectory, splice_result) in enumerate(
            zip(batch.proposals, batch.trajectories, splice_results),
            start=1,
        ):
            output_path = request_dir / f"arg_{sample_index:06d}.trees"
            export_refined_tree_sequence(
                splice_result,
                output_path,
                overwrite=bool(overwrite),
            )
            tree_sequence = splice_result.refined_tree_sequence
            event_counts = Counter(
                event.kind for event in proposal.events
            )
            sample_records.append(
                {
                    "index": sample_index,
                    "output_file": str(output_path.resolve()),
                    "topology_digest": proposal.topology_digest,
                    "trajectory_length": len(trajectory),
                    "local_cwr_log_prior": float(
                        proposal.prior_log_probability
                    ),
                    "whole_vcf_log_likelihood": proposal.log_likelihood,
                    "log_reward": proposal.log_reward,
                    "event_counts": dict(sorted(event_counts.items())),
                    "num_trees": int(tree_sequence.num_trees),
                    "num_nodes": int(tree_sequence.num_nodes),
                    "num_edges": int(tree_sequence.num_edges),
                    "splice_validation": {
                        "is_valid": True,
                        "warnings": list(
                            splice_result.validation.warnings
                        ),
                        "counts": dict(
                            splice_result.validation.counts
                        ),
                    },
                }
            )

        request_manifest = {
            "id": context_id,
            "request": request_spec,
            "seed": request_seed,
            "num_trees": num_trees,
            "transition_count": int(batch.transition_count),
            "restart_count": int(batch.restart_count),
            "output_dir": str(request_dir.resolve()),
            "outputs": sample_records,
        }
        _write_json(
            request_dir / "manifest.json",
            request_manifest,
        )
        request_manifests.append(request_manifest)

    manifest = {
        "mode": "cwr_prior_local_refinement",
        "action_source": "coalescent_with_recombination_prior",
        "uses_model": False,
        "config": str(config_path),
        "dataset_path": str(dataset_path),
        "source_arg": {
            "path": str(source_arg_path),
            "sha256": _file_sha256(source_arg_path),
        },
        "output_dir": str(resolved_output_dir.resolve()),
        "seed": root_seed,
        "num_requests": len(request_manifests),
        "num_trees_per_request": num_trees,
        "output_count": len(request_manifests) * num_trees,
        "environment": {
            "effective_population_size": float(env.population_size),
            "mutation_rate": float(env.mutation_rate),
            "recombination_rate": float(env.recombination_rate),
            "rho": float(env.rho),
        },
        "requests": request_manifests,
    }
    _write_json(resolved_output_dir / "manifest.json", manifest)
    return manifest


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("YAML config must contain a mapping at the top level")
    return config


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"config {field} must be a mapping")
    return value


def _optional_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    return _require_mapping(value, field)


def _validate_requests(value: Any) -> Sequence[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            "config refinement.requests must contain at least one request"
        )
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError("every refinement request must be a mapping")
    return value


def _normalized_request_specs(
    requests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized = []
    seen_ids = set()
    for index, request in enumerate(requests, start=1):
        context_id = str(request.get("id") or f"region_{index:06d}")
        if (
            context_id in {".", ".."}
            or REQUEST_ID_PATTERN.fullmatch(context_id) is None
        ):
            raise ValueError(
                f"invalid local refinement request id {context_id!r}"
            )
        if context_id in seen_ids:
            raise ValueError(
                f"duplicate local refinement request id {context_id!r}"
            )
        seen_ids.add(context_id)
        genomic_range = request.get("genomic_range")
        if (
            not isinstance(genomic_range, (list, tuple))
            or len(genomic_range) != 2
        ):
            raise ValueError(
                f"request {context_id!r} genomic_range must have two values"
            )
        has_cut_time = request.get("cut_time") is not None
        has_cut_event_index = request.get("cut_event_index") is not None
        if has_cut_time == has_cut_event_index:
            raise ValueError(
                f"request {context_id!r} must provide exactly one of "
                "cut_time or cut_event_index"
            )
        normalized.append(
            {
                "id": context_id,
                "genomic_range": [
                    float(genomic_range[0]),
                    float(genomic_range[1]),
                ],
                "cut_time": (
                    float(request["cut_time"])
                    if has_cut_time
                    else None
                ),
                "cut_event_index": (
                    int(request["cut_event_index"])
                    if has_cut_event_index
                    else None
                ),
            }
        )
    return normalized


def _resolve_config_base(
    config_path: Path,
    input_values: Sequence[str | os.PathLike[str]],
) -> Path:
    candidates = [
        Path.cwd().resolve(),
        config_path.parent.resolve(),
        config_path.parent.parent.resolve(),
    ]
    unique_candidates = list(dict.fromkeys(candidates))
    for base in unique_candidates:
        if all(
            Path(value).expanduser().is_absolute()
            or (base / Path(value).expanduser()).exists()
            for value in input_values
        ):
            return base
    return unique_candidates[0]


def _resolve_input_path(
    value: str | os.PathLike[str],
    config_base: Path,
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_base / path
    return path.resolve()


def _resolve_output_dir(
    output_dir: str | os.PathLike[str] | None,
    configured_output_path: Any,
    config_base: Path,
) -> Path:
    if output_dir is not None:
        path = Path(output_dir).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()
    if not configured_output_path:
        raise ValueError(
            "--output-dir is required when config output_path is not set"
        )
    path = Path(str(configured_output_path)).expanduser()
    if not path.is_absolute():
        path = config_base / path
    return (path / "cwr_prior_samples").resolve()


def _preflight_output_paths(
    output_dir: Path,
    context_ids: Sequence[str],
    num_trees: int,
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    expected_paths = [output_dir / "manifest.json"]
    for context_id in context_ids:
        request_dir = output_dir / context_id
        expected_paths.append(request_dir / "manifest.json")
        expected_paths.extend(
            request_dir / f"arg_{index:06d}.trees"
            for index in range(1, num_trees + 1)
        )
    existing = [path for path in expected_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing sampler output: "
            + ", ".join(str(path) for path in existing[:3])
            + (" ..." if len(existing) > 3 else "")
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample local ARG refinements step by step from the CwR prior "
            "without loading a learned model."
        )
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="YAML config containing the VCF, source ARG, and requests.",
    )
    parser.add_argument(
        "--num-trees",
        required=True,
        type=int,
        help="Number of refined .trees files to sample per request.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Output directory. Defaults to "
            "<config output_path>/cwr_prior_samples."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed. Defaults to training.seed in the config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of sampler outputs with matching names.",
    )
    parser.add_argument(
        "--max-generated-events",
        type=int,
        help="Optional per-trajectory construction watchdog.",
    )
    parser.add_argument(
        "--max-searched-states",
        type=int,
        help="Optional total state-transition watchdog per request.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = run_cwr_refinement_sampler(
        args.config,
        num_trees=args.num_trees,
        output_dir=args.output_dir,
        seed=args.seed,
        overwrite=args.overwrite,
        max_generated_events=args.max_generated_events,
        max_searched_states=args.max_searched_states,
        verbose=args.verbose,
    )
    print(
        f"Wrote {manifest['output_count']} refined ARG tree sequence(s) to "
        f"{manifest['output_dir']}"
    )


if __name__ == "__main__":
    main()
