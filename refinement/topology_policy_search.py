"""Truth-free search over source-anchored local topology edit proposals."""

from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from ..env import SimpleARGEnvironment
    from ..utils import load_vcf_variants
except ImportError:  # Support ``python -m refinement.topology_policy_search``.
    from env import SimpleARGEnvironment
    from utils import load_vcf_variants

from .local_construction import (
    LocalARGProposal,
    LocalEdgeRecord,
    LocalNodeRecord,
    _active_endpoints_by_block,
    _cluster_hamming_distance,
    _endpoint_haplotype_indices,
    _local_distance_variant_indices,
    _source_tree_root_time,
    initialize_local_arg_state,
)
from .local_refinement import LocalRefinementRequest, prepare_local_refinement
from .local_splice import export_refined_tree_sequence, splice_local_proposal
from .vcf_likelihood import compute_tree_sequence_vcf_log_likelihood


def _parse_optional_float_list(text: str) -> list[float | None]:
    values: list[float | None] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() in {"none", "null"}:
            values.append(None)
        else:
            values.append(float(item))
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one value")
    return values


def _parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one value")
    return values


def _sample_choice(
    logits: torch.Tensor,
    values: list[Any],
    generator: torch.Generator,
) -> tuple[Any, int, torch.Tensor]:
    probabilities = torch.softmax(logits, dim=0)
    index_tensor = torch.multinomial(
        probabilities,
        num_samples=1,
        replacement=True,
        generator=generator,
    ).reshape(())
    index = int(index_tensor.detach().cpu().item())
    return values[index], index, torch.log(probabilities[index_tensor])


def _softmax_sample_pair(
    clusters: list[dict[str, Any]],
    haplotypes: np.ndarray,
    variant_indices: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    pair_indices: list[tuple[int, int]] = []
    distances: list[float] = []
    for first_index in range(len(clusters)):
        first = clusters[first_index]
        for second_index in range(first_index + 1, len(clusters)):
            second = clusters[second_index]
            pair_indices.append((first_index, second_index))
            distances.append(
                _cluster_hamming_distance(
                    first["samples"],
                    second["samples"],
                    haplotypes,
                    variant_indices,
                )
            )
    if not pair_indices:
        raise ValueError("at least two clusters are required")
    if temperature <= 0.0:
        return min(
            pair_indices,
            key=lambda pair: (
                distances[pair_indices.index(pair)],
                int(clusters[pair[0]]["size"]) + int(clusters[pair[1]]["size"]),
                int(clusters[pair[0]]["node"]),
                int(clusters[pair[1]]["node"]),
            ),
        )
    scaled = -np.asarray(distances, dtype=np.float64) / float(temperature)
    scaled = scaled - float(np.max(scaled))
    probabilities = np.exp(scaled)
    probabilities /= float(probabilities.sum())
    return pair_indices[int(rng.choice(len(pair_indices), p=probabilities))]


def _build_policy_proposal(
    *,
    prepared,
    state,
    coordinates: np.ndarray,
    target_variant_indices: np.ndarray,
    haplotypes: np.ndarray,
    haplotype_by_sample_node: dict[int, int],
    source_node_time: np.ndarray,
    window_bp: float | None,
    merge_power: float,
    root_time_scale: float,
    pair_temperature: float,
    rng: np.random.Generator,
) -> LocalARGProposal:
    target_left, target_right = prepared.context.request.genomic_range
    cut_time = float(prepared.context.resolved_cut.current_time)
    active_by_block = _active_endpoints_by_block(state)
    next_node_id = int(prepared.synthetic_arg.num_nodes) + 2_000_000
    node_records: list[LocalNodeRecord] = []
    edge_records: list[LocalEdgeRecord] = []
    root_intervals: list[tuple[float, float, int]] = []
    cursor = float(target_left)
    for block_index, (left, right) in enumerate(
        zip(state.block_boundaries[:-1], state.block_boundaries[1:])
    ):
        left = max(float(left), float(target_left))
        right = min(float(right), float(target_right))
        if not left < right:
            continue
        if not math.isclose(left, cursor, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError("local block grid does not partition the target")
        endpoints = active_by_block.get(int(block_index), ())
        if not endpoints:
            raise RuntimeError(f"no active endpoint covers block {block_index}")
        midpoint = (left + right) / 2.0
        if len(endpoints) == 1:
            root_intervals.append((left, right, int(endpoints[0])))
            cursor = right
            continue

        variant_indices = _local_distance_variant_indices(
            coordinates,
            target_variant_indices,
            midpoint,
            float(target_left),
            float(target_right),
            window_bp,
        )
        clusters = []
        for node_id in endpoints:
            samples = _endpoint_haplotype_indices(
                prepared,
                int(node_id),
                midpoint,
                haplotype_by_sample_node,
            )
            clusters.append(
                {
                    "node": int(node_id),
                    "samples": samples,
                    "time": float(source_node_time[int(node_id)]),
                    "size": max(len(samples), 1),
                }
            )

        source_root_time = _source_tree_root_time(prepared, midpoint)
        root_time = cut_time + float(root_time_scale) * max(
            source_root_time - cut_time,
            1e-3 * float(len(clusters)),
        )
        total_merges = len(clusters) - 1
        merge_count = 0
        while len(clusters) > 1:
            first, second = _softmax_sample_pair(
                clusters,
                haplotypes,
                variant_indices,
                float(pair_temperature),
                rng,
            )
            if second < first:
                first, second = second, first
            right_cluster = clusters.pop(second)
            left_cluster = clusters.pop(first)
            merge_count += 1
            fraction = (merge_count / total_merges) ** float(merge_power)
            parent_time = cut_time + fraction * (root_time - cut_time)
            parent_time = max(
                float(parent_time),
                float(left_cluster["time"]) + 1e-3,
                float(right_cluster["time"]) + 1e-3,
            )
            parent_node_id = next_node_id
            next_node_id += 1
            node_records.append(
                LocalNodeRecord(
                    node_id=parent_node_id,
                    kind="coalescence",
                    time=float(parent_time),
                    flags=0,
                )
            )
            edge_records.append(
                LocalEdgeRecord(
                    left,
                    right,
                    parent_node_id,
                    int(left_cluster["node"]),
                )
            )
            edge_records.append(
                LocalEdgeRecord(
                    left,
                    right,
                    parent_node_id,
                    int(right_cluster["node"]),
                )
            )
            clusters.append(
                {
                    "node": parent_node_id,
                    "samples": tuple(
                        sorted(
                            set(left_cluster["samples"])
                            | set(right_cluster["samples"])
                        )
                    ),
                    "time": float(parent_time),
                    "size": (
                        int(left_cluster["size"])
                        + int(right_cluster["size"])
                    ),
                }
            )
        root_intervals.append((left, right, int(clusters[0]["node"])))
        cursor = right
    if not math.isclose(cursor, float(target_right), rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("local block grid does not cover the target")
    return LocalARGProposal(
        genomic_range=prepared.context.request.genomic_range,
        cut_time=cut_time,
        nodes=tuple(node_records),
        edges=tuple(edge_records),
        events=(),
        root_intervals=tuple(root_intervals),
        authorized_edge_intervals=tuple(prepared.context.authorized_edge_intervals),
        prior_log_probability=0.0,
        transition_records=(),
        status="terminal",
    )


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda is not available")
    torch_generator = torch.Generator(device=device)
    torch_generator.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    prepared = prepare_local_refinement(
        args.source_arg,
        LocalRefinementRequest(
            (float(args.left), float(args.right)),
            cut_time=float(args.cut_time),
        ),
    )
    variants = load_vcf_variants(args.vcf)
    env = SimpleARGEnvironment(
        variant_data=variants,
        device=device,
        recombination_rate=float(args.recombination_rate),
        population_size=float(args.population_size),
        mutation_rate=float(args.mutation_rate),
        reward_C=0,
    )
    state = initialize_local_arg_state(prepared, env)
    alignment = state.vcf_alignment
    coordinates = np.asarray(alignment["variant_coordinates"], dtype=np.float64)
    target_variant_indices = np.flatnonzero(
        (coordinates >= float(args.left)) & (coordinates < float(args.right))
    )
    haplotypes = np.argmax(
        np.asarray(variants.haplotype_partials, dtype=np.float32),
        axis=2,
    )
    haplotype_by_sample_node = {
        int(node_id): int(haplotype_index)
        for node_id, haplotype_index in alignment[
            "haplotype_index_by_sample_node"
        ].items()
    }
    source_node_time = np.asarray(
        prepared.source_tree_sequence.tables.nodes.time,
        dtype=np.float64,
    )

    choices = {
        "window_bp": args.window_bp,
        "merge_power": args.merge_power,
        "root_time_scale": args.root_time_scale,
        "pair_temperature": args.pair_temperature,
    }
    logits = {
        name: torch.nn.Parameter(torch.zeros(len(values), device=device))
        for name, values in choices.items()
    }
    optimizer = torch.optim.Adam(logits.values(), lr=float(args.lr))
    records: list[dict[str, Any]] = []
    heap: list[tuple[float, int, Any, dict[str, Any]]] = []
    seen_digests: set[str] = set()

    for iteration in range(int(args.iterations)):
        batch_records = []
        for batch_index in range(int(args.batch_size)):
            sampled = {}
            sampled_indices = {}
            log_probs = []
            for name, values in choices.items():
                value, index, log_prob = _sample_choice(
                    logits[name],
                    values,
                    torch_generator,
                )
                sampled[name] = value
                sampled_indices[name] = index
                log_probs.append(log_prob)
            proposal = _build_policy_proposal(
                prepared=prepared,
                state=state,
                coordinates=coordinates,
                target_variant_indices=target_variant_indices,
                haplotypes=haplotypes,
                haplotype_by_sample_node=haplotype_by_sample_node,
                source_node_time=source_node_time,
                window_bp=sampled["window_bp"],
                merge_power=sampled["merge_power"],
                root_time_scale=sampled["root_time_scale"],
                pair_temperature=sampled["pair_temperature"],
                rng=rng,
            )
            splice = splice_local_proposal(prepared, proposal)
            if not splice.is_valid:
                score = float("-inf")
                log_likelihood = float("-inf")
            else:
                log_likelihood = compute_tree_sequence_vcf_log_likelihood(
                    splice.refined_tree_sequence,
                    variants,
                    mutation_rate=float(args.mutation_rate),
                    alignment=alignment,
                    variant_indices=target_variant_indices,
                )
                score = float(log_likelihood)
            record = {
                "iteration": int(iteration),
                "batch_index": int(batch_index),
                "score": float(score),
                "inside_vcf_log_likelihood": float(log_likelihood),
                "choices": dict(sampled),
                "choice_indices": dict(sampled_indices),
                "valid": bool(splice.is_valid),
                "topology_digest": proposal.topology_digest,
                "log_prob": torch.stack(log_probs).sum(),
            }
            batch_records.append(record)
            records.append({k: v for k, v in record.items() if k != "log_prob"})
            if splice.is_valid and proposal.topology_digest not in seen_digests:
                seen_digests.add(proposal.topology_digest)
                heapq.heappush(
                    heap,
                    (
                        float(score),
                        len(records),
                        (proposal, splice),
                        {k: v for k, v in record.items() if k != "log_prob"},
                    ),
                )
                if len(heap) > int(args.keep_top):
                    heapq.heappop(heap)

        finite = [
            record
            for record in batch_records
            if math.isfinite(float(record["score"]))
        ]
        if finite:
            finite.sort(key=lambda item: float(item["score"]), reverse=True)
            elite_count = max(1, int(math.ceil(len(finite) * float(args.elite_fraction))))
            elite = finite[:elite_count]
            loss = -torch.stack([record["log_prob"] for record in elite]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            best = float(elite[0]["score"])
        else:
            best = float("-inf")
        print(
            f"iteration={iteration + 1} best_inside_vcf_loglik={best:.6f}",
            flush=True,
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    region_dir = output_dir / "region_100kb_200kb"
    region_dir.mkdir(parents=True, exist_ok=True)
    selected = sorted(heap, key=lambda item: item[0], reverse=True)
    output_records = []
    for index, (score, _ordinal, (_proposal, splice), record) in enumerate(
        selected,
        start=1,
    ):
        output_path = region_dir / f"arg_{index:06d}.trees"
        export_refined_tree_sequence(splice, output_path, overwrite=True)
        output_records.append(
            {
                **record,
                "rank": int(index),
                "output_file": str(output_path),
                "score": float(score),
            }
        )
    manifest = {
        "mode": "source_anchored_topology_policy_search",
        "source_arg": str(Path(args.source_arg).resolve()),
        "vcf": str(Path(args.vcf).resolve()),
        "output_dir": str(output_dir),
        "device": str(device),
        "seed": int(args.seed),
        "iterations": int(args.iterations),
        "batch_size": int(args.batch_size),
        "keep_top": int(args.keep_top),
        "choices": choices,
        "outputs": output_records,
        "records": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (region_dir / "manifest.json").write_text(
        json.dumps(
            {
                "id": "region_100kb_200kb",
                "output_dir": str(region_dir),
                "sample_count": len(output_records),
                "valid_output_count": len(output_records),
                "outputs": output_records,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search source-anchored local topology edits using VCF likelihood only."
    )
    parser.add_argument("--source-arg", required=True)
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--left", type=float, default=100000.0)
    parser.add_argument("--right", type=float, default=200000.0)
    parser.add_argument("--cut-time", type=float, default=25000.0)
    parser.add_argument("--population-size", type=float, default=10000.0)
    parser.add_argument("--mutation-rate", type=float, default=2e-8)
    parser.add_argument("--recombination-rate", type=float, default=2e-8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--keep-top", type=int, default=16)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument(
        "--window-bp",
        type=_parse_optional_float_list,
        default=_parse_optional_float_list("1000,2500,5000,10000,25000,None"),
    )
    parser.add_argument(
        "--merge-power",
        type=_parse_float_list,
        default=_parse_float_list("1,2,4,8,12"),
    )
    parser.add_argument(
        "--root-time-scale",
        type=_parse_float_list,
        default=_parse_float_list("0.75,1.0"),
    )
    parser.add_argument(
        "--pair-temperature",
        type=_parse_float_list,
        default=_parse_float_list("0,0.005,0.01,0.02,0.05"),
    )
    return parser


def main() -> int:
    manifest = run_search(build_parser().parse_args())
    print(
        f"Wrote {len(manifest['outputs'])} topology candidate(s) to "
        f"{manifest['output_dir']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
