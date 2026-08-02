"""Phased-VCF likelihood helpers for user-anchored local ARG refinement.

The functions in this module deliberately operate on ordinary tskit marginal
trees.  Source synthetic routing nodes are therefore not part of the
likelihood calculation, and biological node times exactly match the clean
tree sequence produced by the splice step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import tskit

try:
    from ..utils import BASES, VCFVariantData
except ImportError:  # Support legacy top-level ``new_rl`` imports.
    from utils import BASES, VCFVariantData


_PROB_FLOOR = 1e-300


@dataclass(frozen=True)
class EndpointVCFPartials:
    """Cached VCF partial rows for one cut-frontier endpoint."""

    variant_indices: tuple[int, ...]
    partials: np.ndarray
    sequences_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        variant_indices = tuple(int(value) for value in self.variant_indices)
        partials = np.asarray(self.partials, dtype=np.float64)
        if partials.ndim != 2 or partials.shape[1] != len(BASES):
            raise ValueError(
                "endpoint VCF partials must have shape "
                f"(rows, {len(BASES)}), got {partials.shape}"
            )
        if partials.shape[0] != len(variant_indices):
            raise ValueError(
                "endpoint VCF partial row count must match variant_indices"
            )
        object.__setattr__(self, "variant_indices", variant_indices)
        object.__setattr__(self, "partials", partials)
        object.__setattr__(
            self,
            "sequences_indices",
            tuple(int(value) for value in self.sequences_indices),
        )

    @property
    def row_count(self) -> int:
        return int(len(self.variant_indices))

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            "variant_indices": tuple(self.variant_indices),
            "partials": self.partials,
            "sequences_indices": tuple(self.sequences_indices),
        }

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "variant_count": int(self.row_count),
            "partial_shape": [int(value) for value in self.partials.shape],
            "sequences_indices": list(self.sequences_indices),
        }


@dataclass(frozen=True)
class RegionLocalVCFView:
    """Exact local view of a whole-chromosome VCF likelihood.

    Target variants remain mutable in local rollout states. Outside variants are
    already scored against the fixed exterior source ARG and cached as a
    constant contribution to the whole-VCF terminal likelihood.
    """

    genomic_range: tuple[float, float]
    endpoints: Mapping[int, EndpointVCFPartials]
    target_variant_indices: tuple[int, ...]
    outside_variant_indices: tuple[int, ...]
    outside_log_likelihood: float
    inside_log_scale: float
    alignment: Mapping[str, Any]
    global_variant_count: int

    def __post_init__(self) -> None:
        left, right = (float(value) for value in self.genomic_range)
        if not math.isfinite(left) or not math.isfinite(right) or not left < right:
            raise ValueError("region-local VCF view requires a finite non-empty range")
        target = tuple(int(value) for value in self.target_variant_indices)
        outside = tuple(int(value) for value in self.outside_variant_indices)
        target_set = set(target)
        outside_set = set(outside)
        if len(target_set) != len(target) or len(outside_set) != len(outside):
            raise ValueError("region-local VCF view variant indices must be unique")
        if target_set.intersection(outside_set):
            raise ValueError("target and outside VCF variants must be disjoint")
        expected_variants = set(range(int(self.global_variant_count)))
        if target_set | outside_set != expected_variants:
            raise ValueError(
                "target plus outside VCF variants must cover the full VCF"
            )
        endpoints = {
            int(node_id): (
                endpoint
                if isinstance(endpoint, EndpointVCFPartials)
                else EndpointVCFPartials(**endpoint)
            )
            for node_id, endpoint in self.endpoints.items()
        }
        for node_id, endpoint in endpoints.items():
            extra = set(endpoint.variant_indices) - target_set
            if extra:
                raise ValueError(
                    "endpoint VCF partials carry non-target variants: "
                    f"endpoint={node_id} variants={sorted(extra)}"
                )
        object.__setattr__(self, "genomic_range", (left, right))
        object.__setattr__(self, "target_variant_indices", target)
        object.__setattr__(self, "outside_variant_indices", outside)
        object.__setattr__(self, "endpoints", endpoints)
        object.__setattr__(
            self,
            "outside_log_likelihood",
            float(self.outside_log_likelihood),
        )
        object.__setattr__(self, "inside_log_scale", float(self.inside_log_scale))
        object.__setattr__(self, "alignment", dict(self.alignment))
        object.__setattr__(
            self,
            "global_variant_count",
            int(self.global_variant_count),
        )

    @property
    def target_variant_count(self) -> int:
        return int(len(self.target_variant_indices))

    @property
    def outside_variant_count(self) -> int:
        return int(len(self.outside_variant_indices))

    @property
    def endpoint_variant_row_count(self) -> int:
        return int(sum(endpoint.row_count for endpoint in self.endpoints.values()))

    def endpoint_for_node(self, node_id: int) -> EndpointVCFPartials:
        return self.endpoints[int(node_id)]

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            "endpoints": {
                int(node_id): endpoint.to_legacy_dict()
                for node_id, endpoint in self.endpoints.items()
            },
            "target_variant_indices": tuple(self.target_variant_indices),
            "outside_variant_indices": tuple(self.outside_variant_indices),
            "outside_log_likelihood": float(self.outside_log_likelihood),
            "inside_log_scale": float(self.inside_log_scale),
            "alignment": dict(self.alignment),
        }

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "likelihood_scope": "whole_vcf_chromosome",
            "genomic_range": [float(value) for value in self.genomic_range],
            "global_variant_count": int(self.global_variant_count),
            "target_variant_count": int(self.target_variant_count),
            "outside_variant_count": int(self.outside_variant_count),
            "cached_exterior_likelihood": True,
            "outside_log_likelihood": float(self.outside_log_likelihood),
            "inside_log_scale": float(self.inside_log_scale),
            "endpoint_count": int(len(self.endpoints)),
            "endpoint_variant_row_count": int(self.endpoint_variant_row_count),
            "endpoint_variant_row_count_by_node": {
                str(node_id): int(endpoint.row_count)
                for node_id, endpoint in sorted(self.endpoints.items())
            },
            "vcf_coordinate_offset": self.alignment.get("vcf_coordinate_offset"),
            "vcf_path": self.alignment.get("vcf_path"),
            "vcf_parser_version": self.alignment.get("parser_version"),
        }


def resolve_vcf_tree_sequence_alignment(
    tree_sequence: tskit.TreeSequence,
    variant_data: VCFVariantData,
    *,
    sample_node_to_haplotype: Mapping[int, int | str] | None = None,
    vcf_coordinate_offset: str | float = "auto",
) -> dict[str, Any]:
    """Resolve sample order and the VCF-to-tskit coordinate convention.

    The repository contains both conventional ``POS - 1`` tree coordinates
    and tsinfer inputs whose sites were installed at raw VCF ``POS``.  Auto
    mode tests both conventions against the source alleles and genotypes and
    accepts only one fully concordant mapping.
    """

    samples = tuple(int(value) for value in tree_sequence.samples())
    if len(samples) != int(variant_data.num_haplotypes):
        raise ValueError(
            "VCF haplotype count does not match the tree-sequence sample "
            f"count: vcf={variant_data.num_haplotypes} "
            f"trees={tree_sequence.num_samples}"
        )
    if not math.isclose(
        float(variant_data.sequence_length),
        float(tree_sequence.sequence_length),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "VCF sequence length does not match the tree sequence: "
            f"vcf={variant_data.sequence_length} "
            f"trees={tree_sequence.sequence_length}"
        )

    haplotype_index_by_sample = _resolve_sample_mapping(
        samples,
        variant_data,
        sample_node_to_haplotype,
    )
    if vcf_coordinate_offset == "auto":
        offsets = (0.0, 1.0)
    else:
        try:
            offsets = (float(vcf_coordinate_offset),)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "vcf_coordinate_offset must be 'auto' or a finite number"
            ) from error
        if not math.isfinite(offsets[0]):
            raise ValueError("vcf_coordinate_offset must be finite")

    variants_by_position = {
        float(variant.site.position): variant
        for variant in tree_sequence.variants(
            samples=samples,
            isolated_as_missing=False,
        )
    }
    matching_offsets = []
    mismatch_diagnostics: dict[float, dict[str, Any]] = {}
    for offset in offsets:
        matched = 0
        missing = 0
        mismatched = 0
        first_mismatch = None
        for variant_index, position0 in enumerate(variant_data.positions0):
            coordinate = float(position0) + float(offset)
            source_variant = variants_by_position.get(coordinate)
            if source_variant is None:
                missing += 1
                continue
            source_alleles = {
                str(value)
                for value in source_variant.alleles
                if value is not None
            }
            ref = str(variant_data.refs[variant_index])
            alt = str(variant_data.alts[variant_index])
            if (
                str(source_variant.site.ancestral_state) != ref
                or ref not in source_alleles
                or alt not in source_alleles
            ):
                mismatched += 1
                if first_mismatch is None:
                    first_mismatch = {
                        "variant_index": int(variant_index),
                        "coordinate": coordinate,
                        "source_alleles": tuple(source_variant.alleles),
                        "source_ancestral_state": (
                            source_variant.site.ancestral_state
                        ),
                        "vcf_ref": ref,
                        "vcf_alt": alt,
                    }
                continue
            expected = _vcf_bases_for_samples(
                variant_data,
                variant_index,
                samples,
                haplotype_index_by_sample,
            )
            observed = tuple(
                None
                if int(genotype) < 0
                else source_variant.alleles[int(genotype)]
                for genotype in source_variant.genotypes
            )
            if observed != expected:
                mismatched += 1
                if first_mismatch is None:
                    first_mismatch = {
                        "variant_index": int(variant_index),
                        "coordinate": coordinate,
                        "observed": observed,
                        "expected": expected,
                    }
            else:
                matched += 1
        diagnostic = {
            "matched_variant_count": int(matched),
            "missing_variant_count": int(missing),
            "mismatched_variant_count": int(mismatched),
            "first_mismatch": first_mismatch,
        }
        mismatch_diagnostics[float(offset)] = diagnostic
        if (
            matched == int(variant_data.num_variants)
            and missing == 0
            and mismatched == 0
        ):
            matching_offsets.append(float(offset))

    if len(matching_offsets) != 1:
        if not matching_offsets:
            raise ValueError(
                "VCF samples/coordinates are not genotype-concordant with "
                "the source tree sequence under the tested offsets: "
                f"{mismatch_diagnostics}"
            )
        raise ValueError(
            "VCF coordinate alignment is ambiguous; pass an explicit "
            f"vcf_coordinate_offset. Matching offsets={matching_offsets}"
        )

    offset = matching_offsets[0]
    coordinates = np.asarray(
        variant_data.positions0,
        dtype=np.float64,
    ) + offset
    if np.any(coordinates < 0.0) or np.any(
        coordinates >= float(tree_sequence.sequence_length)
    ):
        raise ValueError("aligned VCF coordinates fall outside the chromosome")

    return {
        "sample_nodes": samples,
        "haplotype_index_by_sample_node": dict(
            sorted(haplotype_index_by_sample.items())
        ),
        "vcf_coordinate_offset": float(offset),
        "variant_coordinates": coordinates,
        "matched_variant_count": int(variant_data.num_variants),
        "parser_version": str(variant_data.parser_version),
        "vcf_path": str(variant_data.path),
    }


def compute_tree_sequence_vcf_log_likelihood(
    tree_sequence: tskit.TreeSequence,
    variant_data: VCFVariantData,
    *,
    mutation_rate: float,
    sample_node_to_haplotype: Mapping[int, int | str] | None = None,
    vcf_coordinate_offset: str | float = "auto",
    alignment: Mapping[str, Any] | None = None,
    variant_indices: Iterable[int] | None = None,
) -> float:
    """Return the JC69 likelihood of phased VCF observations on ``ts``."""

    if alignment is None:
        alignment = resolve_vcf_tree_sequence_alignment(
            tree_sequence,
            variant_data,
            sample_node_to_haplotype=sample_node_to_haplotype,
            vcf_coordinate_offset=vcf_coordinate_offset,
        )
    if variant_indices is None:
        indices = range(int(variant_data.num_variants))
    else:
        indices = tuple(int(value) for value in variant_indices)

    total = 0.0
    for variant_index in indices:
        result = prune_vcf_variant(
            tree_sequence,
            variant_data,
            variant_index,
            mutation_rate=mutation_rate,
            alignment=alignment,
        )
        total += float(result["log_likelihood"])
    if not math.isfinite(total):
        raise ValueError("tree-sequence VCF log likelihood is non-finite")
    return float(total)


def compute_cut_frontier_vcf_partials(
    tree_sequence: tskit.TreeSequence,
    variant_data: VCFVariantData,
    endpoint_intervals: Mapping[int, tuple[tuple[float, float], ...]],
    genomic_range: tuple[float, float],
    *,
    mutation_rate: float,
    alignment: Mapping[str, Any],
    ) -> RegionLocalVCFView:
    """Prune fixed younger ancestry to the cut endpoints.

    The returned partial rows are normalized independently at every VCF site.
    ``inside_log_scale`` contains the normalization terms below the frontier;
    terminal root integration later completes the inside likelihood.
    """

    left, right = (float(genomic_range[0]), float(genomic_range[1]))
    coordinates = np.asarray(
        alignment["variant_coordinates"],
        dtype=np.float64,
    )
    target_indices = tuple(
        int(value)
        for value in np.flatnonzero(
            (coordinates >= left) & (coordinates < right)
        )
    )
    outside_indices = tuple(
        int(value)
        for value in np.flatnonzero(
            (coordinates < left) | (coordinates >= right)
        )
    )

    rows_by_endpoint: dict[int, list[np.ndarray]] = {
        int(node_id): [] for node_id in endpoint_intervals
    }
    variants_by_endpoint: dict[int, list[int]] = {
        int(node_id): [] for node_id in endpoint_intervals
    }
    samples_by_endpoint: dict[int, set[int]] = {
        int(node_id): set() for node_id in endpoint_intervals
    }
    inside_log_scale = 0.0

    for variant_index in target_indices:
        coordinate = float(coordinates[variant_index])
        result = prune_vcf_variant(
            tree_sequence,
            variant_data,
            variant_index,
            mutation_rate=mutation_rate,
            alignment=alignment,
        )
        tree = result["tree"]
        carrying = tuple(
            sorted(
                int(node_id)
                for node_id, intervals in endpoint_intervals.items()
                if _coordinate_in_intervals(coordinate, intervals)
            )
        )
        if not carrying:
            raise ValueError(
                "no cut endpoint carries target VCF variant "
                f"{variant_index} at {coordinate}"
            )
        _validate_frontier_partition(
            tree,
            carrying,
            tuple(int(value) for value in alignment["sample_nodes"]),
            coordinate,
        )
        partials_by_node = result["partials_by_node"]
        log_scales_by_node = result["log_scales_by_node"]
        sample_index_by_node = {
            int(sample): index
            for index, sample in enumerate(alignment["sample_nodes"])
        }
        for node_id in carrying:
            if node_id not in partials_by_node:
                raise ValueError(
                    f"cut endpoint {node_id} is absent at VCF coordinate "
                    f"{coordinate}"
                )
            rows_by_endpoint[node_id].append(
                np.asarray(partials_by_node[node_id], dtype=np.float64)
            )
            variants_by_endpoint[node_id].append(int(variant_index))
            inside_log_scale += float(log_scales_by_node[node_id])
            samples_by_endpoint[node_id].update(
                int(sample_index_by_node[int(sample)])
                for sample in tree.samples(node_id)
            )

    outside_log_likelihood = compute_tree_sequence_vcf_log_likelihood(
        tree_sequence,
        variant_data,
        mutation_rate=mutation_rate,
        alignment=alignment,
        variant_indices=outside_indices,
    )
    endpoint_records = {}
    for node_id in sorted(endpoint_intervals):
        rows = rows_by_endpoint[int(node_id)]
        endpoint_records[int(node_id)] = EndpointVCFPartials(
            variant_indices=tuple(variants_by_endpoint[int(node_id)]),
            partials=(
                np.stack(rows, axis=0)
                if rows
                else np.empty((0, 4), dtype=np.float64)
            ),
            sequences_indices=tuple(
                sorted(samples_by_endpoint[int(node_id)])
            ),
        )
    return RegionLocalVCFView(
        genomic_range=(left, right),
        endpoints=endpoint_records,
        target_variant_indices=target_indices,
        outside_variant_indices=outside_indices,
        outside_log_likelihood=float(outside_log_likelihood),
        inside_log_scale=float(inside_log_scale),
        alignment=dict(alignment),
        global_variant_count=int(variant_data.num_variants),
    )


def prune_vcf_variant(
    tree_sequence: tskit.TreeSequence,
    variant_data: VCFVariantData,
    variant_index: int,
    *,
    mutation_rate: float,
    alignment: Mapping[str, Any],
) -> dict[str, Any]:
    """Iteratively prune one phased VCF variant on its marginal tree."""

    variant_index = int(variant_index)
    if not 0 <= variant_index < int(variant_data.num_variants):
        raise IndexError(f"VCF variant index is out of bounds: {variant_index}")
    mutation_rate = float(mutation_rate)
    if mutation_rate < 0.0 or not math.isfinite(mutation_rate):
        raise ValueError("mutation_rate must be finite and non-negative")

    coordinate = float(alignment["variant_coordinates"][variant_index])
    tree = tree_sequence.at(coordinate)
    sample_nodes = tuple(int(value) for value in alignment["sample_nodes"])
    haplotype_index_by_sample = alignment[
        "haplotype_index_by_sample_node"
    ]
    leaf_rows = np.asarray(
        variant_data.haplotype_partials[:, variant_index, :],
        dtype=np.float64,
    )
    partials_by_node: dict[int, np.ndarray] = {}
    log_scales_by_node: dict[int, float] = {}

    for node_id in tree.nodes(order="postorder"):
        node_id = int(node_id)
        is_sample = tree_sequence.node(node_id).is_sample()
        if is_sample:
            haplotype_index = int(haplotype_index_by_sample[node_id])
            combined = leaf_rows[haplotype_index].copy()
        else:
            combined = np.ones(4, dtype=np.float64)
        children = tuple(int(value) for value in tree.children(node_id))
        if not children and not is_sample:
            raise ValueError(
                f"non-sample node {node_id} has no descendants at "
                f"coordinate {coordinate}"
            )
        log_scale = 0.0
        parent_time = float(tree_sequence.node(node_id).time)
        for child_id in children:
            child_time = float(tree_sequence.node(child_id).time)
            edge_time = parent_time - child_time
            if not edge_time > 0.0:
                raise ValueError(
                    "tree-sequence node times must increase from child to "
                    f"parent: parent={node_id} child={child_id}"
                )
            transition = _jc69_transition_matrix(
                edge_time * mutation_rate
            )
            combined *= transition @ partials_by_node[child_id]
            log_scale += float(log_scales_by_node[child_id])
        row_sum = float(np.sum(combined))
        if not row_sum > 0.0 or not math.isfinite(row_sum):
            raise ValueError(
                f"VCF partials became invalid at node {node_id}, "
                f"coordinate {coordinate}"
            )
        partials_by_node[node_id] = combined / row_sum
        log_scales_by_node[node_id] = log_scale + math.log(row_sum)

    site_log_likelihood = 0.0
    for root in tree.roots:
        root = int(root)
        root_probability = float(
            np.sum(partials_by_node[root] * 0.25)
        )
        site_log_likelihood += (
            math.log(max(root_probability, _PROB_FLOOR))
            + float(log_scales_by_node[root])
        )
    return {
        "tree": tree,
        "coordinate": coordinate,
        "partials_by_node": partials_by_node,
        "log_scales_by_node": log_scales_by_node,
        "log_likelihood": float(site_log_likelihood),
        "sample_nodes": sample_nodes,
    }


def _resolve_sample_mapping(
    samples: tuple[int, ...],
    variant_data: VCFVariantData,
    mapping: Mapping[int, int | str] | None,
) -> dict[int, int]:
    if mapping is None:
        return {
            int(sample): index for index, sample in enumerate(samples)
        }
    if set(int(value) for value in mapping) != set(samples):
        raise ValueError(
            "sample_node_to_haplotype must contain every tree-sequence "
            "sample node exactly once"
        )
    haplotype_index_by_id = {
        str(value): index
        for index, value in enumerate(variant_data.haplotype_ids)
    }
    output = {}
    used = set()
    for sample in samples:
        value = mapping[int(sample)]
        if isinstance(value, str):
            if value not in haplotype_index_by_id:
                raise ValueError(f"unknown VCF haplotype ID {value!r}")
            index = int(haplotype_index_by_id[value])
        else:
            index = int(value)
        if not 0 <= index < int(variant_data.num_haplotypes):
            raise ValueError(f"VCF haplotype index is out of bounds: {index}")
        if index in used:
            raise ValueError("sample-to-haplotype mapping must be one-to-one")
        used.add(index)
        output[int(sample)] = index
    return output


def _vcf_bases_for_samples(
    variant_data: VCFVariantData,
    variant_index: int,
    samples: tuple[int, ...],
    haplotype_index_by_sample: Mapping[int, int],
) -> tuple[str, ...]:
    rows = np.asarray(
        variant_data.haplotype_partials[:, int(variant_index), :],
        dtype=np.float64,
    )
    return tuple(
        BASES[int(np.argmax(rows[int(haplotype_index_by_sample[sample])]))]
        for sample in samples
    )


def _coordinate_in_intervals(
    coordinate: float,
    intervals: tuple[tuple[float, float], ...],
) -> bool:
    return any(
        float(left) <= coordinate < float(right)
        for left, right in intervals
    )


def _validate_frontier_partition(
    tree: tskit.Tree,
    endpoint_ids: tuple[int, ...],
    sample_nodes: tuple[int, ...],
    coordinate: float,
) -> None:
    covered: set[int] = set()
    for node_id in endpoint_ids:
        descendants = {int(value) for value in tree.samples(int(node_id))}
        overlap = covered.intersection(descendants)
        if overlap:
            raise ValueError(
                "cut endpoints have overlapping descendant samples at "
                f"{coordinate}: endpoints={endpoint_ids} overlap={sorted(overlap)}"
            )
        covered.update(descendants)
    expected = set(int(value) for value in sample_nodes)
    if covered != expected:
        raise ValueError(
            "cut endpoints do not partition all samples at "
            f"{coordinate}: missing={sorted(expected - covered)} "
            f"extra={sorted(covered - expected)}"
        )


def _jc69_transition_matrix(branch_length: float) -> np.ndarray:
    branch_length = float(branch_length)
    decay = math.exp(-4.0 * branch_length / 3.0)
    same = 0.25 + 0.75 * decay
    different = 0.25 - 0.25 * decay
    matrix = np.full((4, 4), different, dtype=np.float64)
    np.fill_diagonal(matrix, same)
    return matrix


__all__ = [
    "EndpointVCFPartials",
    "RegionLocalVCFView",
    "compute_cut_frontier_vcf_partials",
    "compute_tree_sequence_vcf_log_likelihood",
    "prune_vcf_variant",
    "resolve_vcf_tree_sequence_alignment",
]
