"""Splice terminal local proposals and export a clean chromosome tree sequence.

The splice is first performed against the synthetic/full ARG because the
authorization contract refers to its edge IDs.  All routing nodes introduced
by that source conversion are then collapsed across the chromosome.  Nodes
created by the local proposal are retained, including explicit paired
recombination nodes and their sampled event times.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tskit

from .local_construction import (
    LocalARGProposal,
    LocalEventRecord,
)
from .vcf_likelihood import compute_tree_sequence_vcf_log_likelihood
from .local_refinement import (
    PreparedLocalRefinement,
    _canonical_segments,
    _intersect_segments,
    _subtract_segments,
)
from .synthetic_full_arg import NODE_IS_RE_EVENT
try:
    from ..utils import load_vcf_variants
except ImportError:  # Support the repository's legacy top-level new_rl import.
    from utils import load_vcf_variants


Interval = tuple[float, float]
LOCAL_REFINEMENT_PROVENANCE_NAME = "new_rl_local_arg_refinement"


@dataclass(frozen=True)
class LocalValidationReport:
    is_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    counts: dict[str, int | float | bool]


@dataclass(frozen=True)
class LocalSpliceResult:
    """A clean whole-chromosome result plus remapping and validation."""

    refined_tree_sequence: tskit.TreeSequence
    proposal: LocalARGProposal
    validation: LocalValidationReport
    source_node_id_map: tuple[int, ...]
    local_node_id_map: dict[int, int]
    removed_source_synthetic_node_ids: tuple[int, ...]
    provenance_record: dict[str, Any]

    @property
    def is_valid(self) -> bool:
        return bool(self.validation.is_valid)


def splice_local_proposal(
    prepared: PreparedLocalRefinement,
    proposal: LocalARGProposal,
) -> LocalSpliceResult:
    """Splice one terminal proposal and remove source routing nodes."""

    if not proposal.is_valid:
        raise ValueError("only a valid terminal local proposal can be spliced")
    if tuple(proposal.authorized_edge_intervals) != tuple(
        prepared.context.authorized_edge_intervals
    ):
        raise ValueError(
            "proposal authorization does not match the prepared refinement context"
        )

    source_synthetic = prepared.synthetic_arg
    tables = source_synthetic.dump_tables()
    _replace_authorized_edges(tables, proposal)
    temporary_to_spliced = _append_local_proposal(tables, proposal)

    original_num_nodes = int(
        prepared.synthetic_conversion.metadata["original_num_nodes"]
    )
    augmented_num_nodes = int(
        prepared.synthetic_conversion.metadata["augmented_num_nodes"]
    )
    source_synthetic_nodes = tuple(
        node_id
        for node_id in range(original_num_nodes, augmented_num_nodes)
        if int(tables.nodes[node_id].flags) & NODE_IS_RE_EVENT
    )
    old_to_clean = _collapse_nodes(
        tables,
        source_synthetic_nodes,
        source_node_times=np.asarray(
            prepared.source_tree_sequence.tables.nodes.time,
            dtype=np.float64,
        ),
    )
    local_node_id_map = {
        int(temporary_node_id): int(old_to_clean[spliced_node_id])
        for temporary_node_id, spliced_node_id in temporary_to_spliced.items()
    }
    if any(value < 0 for value in local_node_id_map.values()):
        raise RuntimeError("a locally generated node was removed during routing collapse")

    _restore_original_biological_tables(
        tables,
        prepared.source_tree_sequence,
    )
    _restore_source_edge_metadata(
        tables,
        prepared.source_tree_sequence,
    )
    tables.sort()
    if proposal.nodes or proposal.authorized_edge_intervals:
        mutation_remap = _remap_target_mutations(
            tables,
            prepared.source_tree_sequence,
            proposal.genomic_range,
        )
    else:
        tables.mutations.replace_with(
            prepared.source_tree_sequence.tables.mutations
        )
        mutation_remap = {
            "target_site_count": sum(
                float(proposal.genomic_range[0])
                <= float(site.position)
                < float(proposal.genomic_range[1])
                for site in prepared.source_tree_sequence.sites()
            ),
            "target_mutation_count": sum(
                float(proposal.genomic_range[0])
                <= float(
                    prepared.source_tree_sequence.site(
                        mutation.site
                    ).position
                )
                < float(proposal.genomic_range[1])
                for mutation in prepared.source_tree_sequence.mutations()
            ),
            "exterior_mutation_count": sum(
                not (
                    float(proposal.genomic_range[0])
                    <= float(
                        prepared.source_tree_sequence.site(
                            mutation.site
                        ).position
                    )
                    < float(proposal.genomic_range[1])
                )
                for mutation in prepared.source_tree_sequence.mutations()
            ),
            "target_genotypes_preserved": True,
            "no_op_copy": True,
        }

    provenance = _build_provenance_record(
        prepared,
        proposal,
        source_synthetic_nodes,
        local_node_id_map,
        mutation_remap,
    )
    tables.provenances.add_row(record=json.dumps(provenance, sort_keys=True))
    refined = tables.tree_sequence()
    source_node_map = tuple(
        int(old_to_clean[node_id])
        if node_id < old_to_clean.size
        else tskit.NULL
        for node_id in range(int(prepared.source_tree_sequence.num_nodes))
    )
    validation = validate_local_splice(
        prepared,
        proposal,
        refined,
        local_node_id_map,
        source_synthetic_nodes,
    )
    return LocalSpliceResult(
        refined_tree_sequence=refined,
        proposal=proposal,
        validation=validation,
        source_node_id_map=source_node_map,
        local_node_id_map=local_node_id_map,
        removed_source_synthetic_node_ids=source_synthetic_nodes,
        provenance_record=provenance,
    )


def export_refined_tree_sequence(
    result: LocalSpliceResult,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a validated refined tree sequence without accidental overwrite."""

    if not result.is_valid:
        raise ValueError(
            "cannot export an invalid local splice: "
            + "; ".join(result.validation.errors)
        )
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refined output already exists: {path}; pass overwrite=True explicitly"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    result.refined_tree_sequence.dump(str(path))
    # Reloading is part of the export contract rather than only a test helper.
    reloaded = tskit.load(str(path))
    if not reloaded.tables.equals(
        result.refined_tree_sequence.tables,
        ignore_provenance=False,
    ):
        raise RuntimeError("reloaded refined tree sequence differs from the export")
    return path


def _replace_authorized_edges(
    tables: tskit.TableCollection,
    proposal: LocalARGProposal,
) -> None:
    removed_by_edge: dict[int, tuple[Interval, ...]] = {}
    for item in proposal.authorized_edge_intervals:
        removed_by_edge[int(item.edge_id)] = _canonical_segments(
            removed_by_edge.get(int(item.edge_id), ())
            + ((float(item.left), float(item.right)),)
        )

    source_edges = list(tables.edges)
    tables.edges.clear()
    for edge_id, edge in enumerate(source_edges):
        remaining = _subtract_segments(
            ((float(edge.left), float(edge.right)),),
            removed_by_edge.get(edge_id, ()),
        )
        for left, right in remaining:
            tables.edges.add_row(
                left=left,
                right=right,
                parent=int(edge.parent),
                child=int(edge.child),
                metadata=edge.metadata,
            )


def _append_local_proposal(
    tables: tskit.TableCollection,
    proposal: LocalARGProposal,
) -> dict[int, int]:
    node_map: dict[int, int] = {}
    empty_node_metadata = tables.nodes.metadata_schema.empty_value
    for node in sorted(proposal.nodes, key=lambda item: item.node_id):
        output_id = tables.nodes.add_row(
            flags=int(node.flags),
            time=float(node.time),
            population=tskit.NULL,
            individual=tskit.NULL,
            metadata=empty_node_metadata,
        )
        node_map[int(node.node_id)] = int(output_id)

    empty_edge_metadata = tables.edges.metadata_schema.empty_value
    for edge in proposal.edges:
        parent = node_map.get(int(edge.parent_node_id), int(edge.parent_node_id))
        child = node_map.get(int(edge.child_node_id), int(edge.child_node_id))
        tables.edges.add_row(
            left=float(edge.left),
            right=float(edge.right),
            parent=parent,
            child=child,
            metadata=empty_edge_metadata,
        )
    return node_map


def _collapse_nodes(
    tables: tskit.TableCollection,
    node_ids: Iterable[int],
    *,
    source_node_times: np.ndarray | None = None,
) -> np.ndarray:
    """Eliminate routing nodes by interval-aware path composition.

    Synthetic conversion can perturb original node times to make its internal
    event schedule globally unique.  Local proposals, however, are bounded by
    the original biological times that will be restored in the exported tree
    sequence.  Keep the synthetic clock while composing routing paths, then
    restore source times before ``TableCollection.subset`` performs its
    implicit strict sort.
    """

    remove = {int(value) for value in node_ids}
    num_nodes = int(tables.nodes.num_rows)
    if remove and (min(remove) < 0 or max(remove) >= num_nodes):
        raise ValueError("routing node removal set contains an invalid node ID")
    mutation_nodes = set(int(value) for value in tables.mutations.node)
    migration_nodes = set(int(value) for value in tables.migrations.node)
    referenced = remove & (mutation_nodes | migration_nodes)
    if referenced:
        raise ValueError(
            "temporary routing nodes unexpectedly carry mutations or migrations: "
            f"{sorted(referenced)[:10]}"
        )

    if remove:
        edges = [
            (
                float(edge.left),
                float(edge.right),
                int(edge.parent),
                int(edge.child),
            )
            for edge in tables.edges
        ]
        outgoing: dict[int, list[tuple[float, float, int]]] = {}
        for left, right, parent, child in edges:
            outgoing.setdefault(parent, []).append((left, right, child))

        composed: list[tuple[float, float, int, int]] = []

        def descend(
            fixed_parent: int,
            left: float,
            right: float,
            child: int,
            path: frozenset[int],
        ) -> None:
            if child not in remove:
                if fixed_parent != child and left < right:
                    composed.append((left, right, fixed_parent, child))
                return
            if child in path:
                raise ValueError("synthetic routing graph contains a cycle")
            next_path = path | {child}
            for edge_left, edge_right, grandchild in outgoing.get(child, ()):
                overlap_left = max(left, edge_left)
                overlap_right = min(right, edge_right)
                if overlap_left < overlap_right:
                    descend(
                        fixed_parent,
                        overlap_left,
                        overlap_right,
                        grandchild,
                        next_path,
                    )

        for left, right, parent, child in edges:
            if parent in remove:
                continue
            descend(parent, left, right, child, frozenset())

        merged = _merge_edge_intervals(composed)
        tables.edges.clear()
        empty_metadata = tables.edges.metadata_schema.empty_value
        for left, right, parent, child in merged:
            tables.edges.add_row(
                left=left,
                right=right,
                parent=parent,
                child=child,
                metadata=empty_metadata,
            )

    if source_node_times is not None:
        _restore_source_node_times(tables, source_node_times)
        _validate_strict_edge_times(tables)

    if not remove:
        return np.arange(num_nodes, dtype=np.int32)

    keep_nodes = np.asarray(
        [node_id for node_id in range(num_nodes) if node_id not in remove],
        dtype=np.int32,
    )
    old_to_new = np.full(num_nodes, tskit.NULL, dtype=np.int32)
    old_to_new[keep_nodes] = np.arange(keep_nodes.size, dtype=np.int32)
    migration_rows = tuple(tables.migrations)
    # tskit 1.0 refuses TableCollection.subset whenever migrations exist,
    # even when all migration nodes are retained. Preserve them explicitly and
    # restore their node references after the node subset has been applied.
    tables.migrations.clear()
    tables.subset(
        keep_nodes,
        record_provenance=False,
        reorder_populations=False,
        remove_unreferenced=False,
    )
    empty_migration_metadata = tables.migrations.metadata_schema.empty_value
    for migration in migration_rows:
        metadata = migration.metadata
        if metadata is None:
            metadata = empty_migration_metadata
        mapped_node = int(old_to_new[int(migration.node)])
        if mapped_node == tskit.NULL:
            raise ValueError(
                "routing collapse removed a node referenced by a migration"
            )
        tables.migrations.add_row(
            left=float(migration.left),
            right=float(migration.right),
            node=mapped_node,
            source=int(migration.source),
            dest=int(migration.dest),
            time=float(migration.time),
            metadata=metadata,
        )
    return old_to_new


def _restore_source_node_times(
    tables: tskit.TableCollection,
    source_node_times: np.ndarray,
) -> None:
    source_node_times = np.asarray(source_node_times, dtype=np.float64)
    if source_node_times.ndim != 1:
        raise ValueError("source node times must be one-dimensional")
    if source_node_times.size > int(tables.nodes.num_rows):
        raise ValueError("source node times exceed the splice node table")
    node_times = np.asarray(tables.nodes.time, dtype=np.float64).copy()
    node_times[: source_node_times.size] = source_node_times
    nodes = tables.nodes
    nodes.set_columns(
        flags=nodes.flags,
        time=node_times,
        population=nodes.population,
        individual=nodes.individual,
        metadata=nodes.metadata,
        metadata_offset=nodes.metadata_offset,
    )


def _validate_strict_edge_times(tables: tskit.TableCollection) -> None:
    node_times = np.asarray(tables.nodes.time, dtype=np.float64)
    edge_parents = np.asarray(tables.edges.parent, dtype=np.int32)
    edge_children = np.asarray(tables.edges.child, dtype=np.int32)
    invalid = np.flatnonzero(
        node_times[edge_parents] <= node_times[edge_children]
    )
    if invalid.size:
        edge_id = int(invalid[0])
        edge = tables.edges[edge_id]
        raise ValueError(
            "collapsed local splice violates parent.time > child.time: "
            f"edge={edge_id} interval=[{edge.left}, {edge.right}) "
            f"parent={edge.parent} time={node_times[int(edge.parent)]} "
            f"child={edge.child} time={node_times[int(edge.child)]}"
        )


def _merge_edge_intervals(
    edges: Iterable[tuple[float, float, int, int]],
) -> tuple[tuple[float, float, int, int], ...]:
    by_pair: dict[tuple[int, int], list[Interval]] = {}
    for left, right, parent, child in edges:
        if left < right:
            by_pair.setdefault((int(parent), int(child)), []).append(
                (float(left), float(right))
            )
    output = []
    for (parent, child), intervals in sorted(by_pair.items()):
        for left, right in _canonical_segments(tuple(intervals)):
            output.append((left, right, parent, child))
    return tuple(output)


def _restore_original_biological_tables(
    tables: tskit.TableCollection,
    source: tskit.TreeSequence,
) -> None:
    """Restore source biological rows before target mutation remapping."""

    source_tables = source.dump_tables()
    source_node_count = int(source.num_nodes)
    if int(tables.nodes.num_rows) < source_node_count:
        raise RuntimeError("routing collapse removed an original biological node")
    retained_local_nodes = [
        tables.nodes[node_id]
        for node_id in range(source_node_count, int(tables.nodes.num_rows))
    ]

    tables.nodes.replace_with(source_tables.nodes)
    empty_metadata = tables.nodes.metadata_schema.empty_value
    for node in retained_local_nodes:
        metadata = node.metadata
        if metadata is None:
            metadata = empty_metadata
        tables.nodes.add_row(
            flags=int(node.flags),
            time=float(node.time),
            population=int(node.population),
            individual=int(node.individual),
            metadata=metadata,
        )

    tables.sites.replace_with(source_tables.sites)
    tables.mutations.clear()
    tables.migrations.replace_with(source_tables.migrations)
    tables.individuals.replace_with(source_tables.individuals)
    tables.populations.replace_with(source_tables.populations)
    tables.provenances.replace_with(source_tables.provenances)
    tables.time_units = source_tables.time_units
    tables.metadata_schema = source_tables.metadata_schema
    tables.metadata = source_tables.metadata


def _remap_target_mutations(
    tables: tskit.TableCollection,
    source: tskit.TreeSequence,
    region: Interval,
) -> dict[str, int | bool]:
    """Preserve exterior mutations and remap target sites parsimoniously.

    The source sample genotypes are the invariant.  Target mutation rows are
    reconstructed only after the proposed topology and node times are fixed.
    """

    provisional = tables.tree_sequence()
    if not np.array_equal(source.samples(), provisional.samples()):
        raise ValueError(
            "refined topology changed the biological sample node ordering"
        )

    left, right = region
    target_variants = {
        int(variant.site.id): (
            np.asarray(variant.genotypes, dtype=np.int32).copy(),
            tuple(variant.alleles),
        )
        for variant in source.variants(
            samples=source.samples(),
            isolated_as_missing=False,
            left=float(left),
            right=float(right),
            copy=True,
        )
    }

    source_mutations_by_site: dict[int, list[Any]] = {}
    for mutation in source.mutations():
        source_mutations_by_site.setdefault(
            int(mutation.site),
            [],
        ).append(mutation)

    mutation_schema = source.tables.mutations.metadata_schema
    tables.mutations.metadata_schema = mutation_schema
    tables.mutations.clear()
    empty_metadata = mutation_schema.empty_value
    exterior_count = 0
    target_count = 0

    for site in source.sites():
        site_id = int(site.id)
        position = float(site.position)
        if not float(left) <= position < float(right):
            old_to_new: dict[int, int] = {}
            for mutation in source_mutations_by_site.get(site_id, ()):
                parent = tskit.NULL
                if int(mutation.parent) != tskit.NULL:
                    parent = old_to_new[int(mutation.parent)]
                old_to_new[int(mutation.id)] = int(
                    tables.mutations.add_row(
                        site=site_id,
                        node=int(mutation.node),
                        derived_state=mutation.derived_state,
                        parent=parent,
                        metadata=mutation.metadata,
                        time=float(mutation.time),
                    )
                )
                exterior_count += 1
            continue

        variant_data = target_variants.get(site_id)
        if variant_data is None:
            raise ValueError(
                f"target site {site_id} at {position} has no source variant"
            )
        genotypes, alleles = variant_data
        tree = provisional.at(position)
        ancestral_state, mapped = tree.map_mutations(
            genotypes,
            alleles,
            ancestral_state=site.ancestral_state,
        )
        if ancestral_state != site.ancestral_state:
            raise ValueError(
                f"target site {site_id} changed its ancestral state"
            )
        mapped_row_ids: list[int] = []
        for mutation in mapped:
            parent = tskit.NULL
            if int(mutation.parent) != tskit.NULL:
                parent = mapped_row_ids[int(mutation.parent)]
            mapped_row_ids.append(
                int(
                    tables.mutations.add_row(
                        site=site_id,
                        node=int(mutation.node),
                        derived_state=mutation.derived_state,
                        parent=parent,
                        metadata=empty_metadata,
                        time=tskit.UNKNOWN_TIME,
                    )
                )
            )
            target_count += 1

    refined = tables.tree_sequence()
    source_target = _target_genotype_signature(source, region)
    refined_target = _target_genotype_signature(refined, region)
    if source_target != refined_target:
        raise ValueError(
            "parsimonious target mutation mapping did not preserve sample genotypes"
        )
    return {
        "target_site_count": len(target_variants),
        "target_mutation_count": target_count,
        "exterior_mutation_count": exterior_count,
        "target_genotypes_preserved": True,
    }


def _target_genotype_signature(
    ts: tskit.TreeSequence,
    region: Interval,
) -> tuple[tuple[float, tuple[str | None, ...]], ...]:
    """Return allele strings per sample so allele-index ordering is irrelevant."""

    left, right = region
    output = []
    for variant in ts.variants(
        samples=ts.samples(),
        isolated_as_missing=False,
        left=float(left),
        right=float(right),
        copy=True,
    ):
        alleles = tuple(variant.alleles)
        observed = tuple(
            None
            if int(value) == tskit.MISSING_DATA
            else str(alleles[int(value)])
            for value in variant.genotypes
        )
        output.append((float(variant.site.position), observed))
    return tuple(output)


def _restore_source_edge_metadata(
    tables: tskit.TableCollection,
    source: tskit.TreeSequence,
) -> None:
    source_by_pair: dict[tuple[int, int], list[Any]] = {}
    for edge in source.edges():
        source_by_pair.setdefault(
            (int(edge.parent), int(edge.child)),
            [],
        ).append(edge)

    output_edges = list(tables.edges)
    tables.edges.metadata_schema = source.tables.edges.metadata_schema
    tables.edges.clear()
    empty_metadata = tables.edges.metadata_schema.empty_value
    restored_edges = []
    for edge in output_edges:
        metadata = empty_metadata
        for source_edge in source_by_pair.get(
            (int(edge.parent), int(edge.child)),
            (),
        ):
            if (
                float(source_edge.left) <= float(edge.left)
                and float(edge.right) <= float(source_edge.right)
            ):
                metadata = source_edge.metadata
                break
        restored_edges.append(
            (
                int(edge.parent),
                int(edge.child),
                float(edge.left),
                float(edge.right),
                metadata,
            )
        )

    restored_edges.sort(key=lambda record: record[:4])
    merged_edges = []
    for parent, child, left, right, metadata in restored_edges:
        if (
            merged_edges
            and merged_edges[-1][0] == parent
            and merged_edges[-1][1] == child
            and merged_edges[-1][4] == metadata
            and left <= merged_edges[-1][3]
        ):
            previous = merged_edges[-1]
            merged_edges[-1] = (
                parent,
                child,
                previous[2],
                max(previous[3], right),
                metadata,
            )
        else:
            merged_edges.append((parent, child, left, right, metadata))

    for parent, child, left, right, metadata in merged_edges:
        tables.edges.add_row(
            left=left,
            right=right,
            parent=parent,
            child=child,
            metadata=metadata,
        )


def _build_provenance_record(
    prepared: PreparedLocalRefinement,
    proposal: LocalARGProposal,
    removed_nodes: tuple[int, ...],
    local_node_map: dict[int, int],
    mutation_remap: dict[str, int | bool],
) -> dict[str, Any]:
    request = prepared.context.request
    initialization = next(
        (
            record
            for record in proposal.transition_records
            if record.get("event_type") == "initialization"
        ),
        {},
    )
    return {
        "software": {
            "name": LOCAL_REFINEMENT_PROVENANCE_NAME,
            "version": "1",
        },
        "parameters": {
            "genomic_range": [
                float(request.genomic_range[0]),
                float(request.genomic_range[1]),
            ],
            "requested_time": request.cut_time,
            "requested_event_index": request.cut_event_index,
            "resolved_cut_step": int(prepared.context.resolved_cut.cut_step),
            "resolved_cut_time": float(
                prepared.context.resolved_cut.current_time
            ),
            "structural_proposal_only": (
                proposal.likelihood_scope == "none"
            ),
            "likelihood_scope": proposal.likelihood_scope,
            "mutation_model": (
                "JC69"
                if proposal.likelihood_scope != "none"
                else None
            ),
            "mutation_rate": initialization.get("mutation_rate"),
            "population_size": initialization.get("population_size"),
            "reward_C": initialization.get("reward_C"),
            "vcf_path": initialization.get("vcf_path"),
            "vcf_parser_version": initialization.get(
                "vcf_parser_version"
            ),
            "vcf_coordinate_offset": initialization.get(
                "vcf_coordinate_offset"
            ),
            "sample_node_to_haplotype": initialization.get(
                "sample_node_to_haplotype"
            ),
            "time_scale": (
                next(
                    (
                        record.get("time_scale")
                        for record in proposal.transition_records
                        if record.get("time_scale") is not None
                    ),
                    None,
                )
            ),
        },
        "proposal": {
            "generated_node_count": len(proposal.nodes),
            "generated_edge_count": len(proposal.edges),
            "generated_event_count": len(proposal.events),
            "prior_log_probability": float(
                proposal.prior_log_probability
            ),
            "whole_chromosome_vcf_log_likelihood": (
                None
                if proposal.log_likelihood is None
                else float(proposal.log_likelihood)
            ),
            "fixed_outside_vcf_log_likelihood": (
                proposal.outside_log_likelihood
            ),
            "reconstructed_inside_vcf_log_likelihood": (
                proposal.local_log_likelihood
            ),
            "local_cwr_log_prior": float(
                proposal.prior_log_probability
            ),
            "terminal_log_reward": (
                None
                if proposal.log_reward is None
                else float(proposal.log_reward)
            ),
            "likelihood_alignment": dict(
                proposal.likelihood_alignment
            ),
            "topology_digest": proposal.topology_digest,
            "root_intervals": [
                [float(left), float(right), int(node_id)]
                for left, right, node_id in proposal.root_intervals
            ],
            "transition_records": list(proposal.transition_records),
        },
        "mutation_remapping": dict(mutation_remap),
        "conversion": {
            "removed_source_synthetic_node_count": len(removed_nodes),
            "removed_source_synthetic_node_ids": list(removed_nodes),
            "local_node_id_map": {
                str(key): int(value)
                for key, value in sorted(local_node_map.items())
            },
        },
    }


def validate_local_splice(
    prepared: PreparedLocalRefinement,
    proposal: LocalARGProposal,
    refined: tskit.TreeSequence,
    local_node_id_map: dict[int, int],
    removed_source_synthetic_node_ids: tuple[int, ...],
) -> LocalValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    source = prepared.source_tree_sequence
    region = proposal.genomic_range

    try:
        refined.dump_tables().tree_sequence()
    except Exception as error:  # pragma: no cover - tskit supplies exact type
        errors.append(f"tskit integrity validation failed: {error}")

    node_time = np.asarray(refined.tables.nodes.time, dtype=np.float64)
    for edge in refined.edges():
        if not node_time[int(edge.parent)] > node_time[int(edge.child)]:
            errors.append(
                "edge violates parent.time > child.time: "
                f"parent={edge.parent} child={edge.child}"
            )
            break

    if any(
        int(refined.node(node_id).flags) & NODE_IS_RE_EVENT
        for node_id in range(int(source.num_nodes), int(refined.num_nodes))
        if node_id not in set(local_node_id_map.values())
    ):
        warnings.append("an unrecognized post-source recombination node remains")

    _validate_local_events(
        proposal.events,
        refined,
        local_node_id_map,
        errors,
    )
    _validate_local_roots(
        proposal,
        refined,
        local_node_id_map,
        errors,
    )
    dangling_target_node_count, dangling_target_tree_count = (
        _validate_no_dangling_target_ancestry(
            refined,
            region,
            errors,
        )
    )
    collapsed_local_recombination_parity = (
        _validate_collapsed_local_recombination_marginals(
            proposal.events,
            refined,
            local_node_id_map,
            errors,
        )
    )
    _validate_exterior(source, refined, region, errors)
    _validate_biological_tables(source, refined, region, errors)
    _validate_mutations(refined, region, errors)
    direct_log_likelihood = _validate_vcf_likelihood_parity(
        proposal,
        refined,
        errors,
    )

    expected_removed = len(removed_source_synthetic_node_ids)
    remaining_source_synthetic = [
        node_id
        for node_id in range(int(source.num_nodes), int(refined.num_nodes))
        if node_id not in set(local_node_id_map.values())
        and int(refined.node(node_id).flags) & NODE_IS_RE_EVENT
    ]
    if remaining_source_synthetic:
        errors.append(
            "temporary source synthetic nodes remain after collapse: "
            f"{remaining_source_synthetic[:10]}"
        )

    counts: dict[str, int | float | bool] = {
        "source_node_count": int(source.num_nodes),
        "refined_node_count": int(refined.num_nodes),
        "refined_edge_count": int(refined.num_edges),
        "removed_source_synthetic_node_count": int(expected_removed),
        "retained_local_node_count": len(local_node_id_map),
        "sampled_event_count": len(proposal.events),
        "local_root_region_count": len(proposal.root_intervals),
        "dangling_target_node_count": dangling_target_node_count,
        "dangling_target_tree_count": dangling_target_tree_count,
        "target_genotypes_preserved": not any(
            message.startswith("target sample genotypes")
            for message in errors
        ),
        "collapsed_local_recombination_parity": (
            collapsed_local_recombination_parity
        ),
        "exterior_unchanged": not any(
            message.startswith("exterior")
            for message in errors
        ),
        "likelihood_parity": not any(
            message.startswith("whole-chromosome VCF likelihood")
            for message in errors
        ),
    }
    if direct_log_likelihood is not None:
        counts["independent_vcf_log_likelihood"] = float(
            direct_log_likelihood
        )
    return LocalValidationReport(
        is_valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        counts=counts,
    )


def _validate_vcf_likelihood_parity(
    proposal: LocalARGProposal,
    refined: tskit.TreeSequence,
    errors: list[str],
) -> float | None:
    """Independently rescore likelihood-enabled proposals after mutation remap."""

    if proposal.log_likelihood is None:
        return None
    initialization = next(
        (
            record
            for record in proposal.transition_records
            if record.get("event_type") == "initialization"
        ),
        None,
    )
    if initialization is None:
        errors.append(
            "whole-chromosome VCF likelihood cannot be validated without "
            "an initialization record"
        )
        return None
    vcf_path = initialization.get("vcf_path")
    mutation_rate = initialization.get("mutation_rate")
    if not vcf_path or mutation_rate is None:
        errors.append(
            "whole-chromosome VCF likelihood cannot be validated without "
            "the VCF path and mutation rate"
        )
        return None
    try:
        variant_data = load_vcf_variants(vcf_path)
        direct = compute_tree_sequence_vcf_log_likelihood(
            refined,
            variant_data,
            mutation_rate=float(mutation_rate),
            sample_node_to_haplotype=initialization.get(
                "sample_node_to_haplotype"
            ),
            vcf_coordinate_offset=initialization.get(
                "vcf_coordinate_offset",
                "auto",
            ),
        )
    except (OSError, ValueError) as error:
        errors.append(
            "whole-chromosome VCF likelihood validation failed: "
            f"{error}"
        )
        return None
    if not math.isclose(
        float(direct),
        float(proposal.log_likelihood),
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        errors.append(
            "whole-chromosome VCF likelihood differs from the independent "
            "clean-tree rescore: "
            f"incremental={proposal.log_likelihood} direct={direct}"
        )
    return float(direct)


def _validate_no_dangling_target_ancestry(
    refined: tskit.TreeSequence,
    region: Interval,
    errors: list[str],
) -> tuple[int, int]:
    """Reject non-sample branches with no sample descendants in the target."""

    left, right = (float(region[0]), float(region[1]))
    occurrences: list[tuple[int, float, float, int]] = []
    affected_trees: set[int] = set()
    affected_nodes: set[int] = set()
    for tree in refined.trees():
        overlap_left = max(left, float(tree.interval.left))
        overlap_right = min(right, float(tree.interval.right))
        if not overlap_left < overlap_right:
            continue
        for node_id in tree.nodes():
            node_id = int(node_id)
            if refined.node(node_id).is_sample():
                continue
            if int(tree.num_samples(node_id)) != 0:
                continue
            affected_nodes.add(node_id)
            affected_trees.add(int(tree.index))
            if len(occurrences) < 10:
                occurrences.append(
                    (
                        node_id,
                        overlap_left,
                        overlap_right,
                        int(tree.parent(node_id)),
                    )
                )
    if affected_nodes:
        errors.append(
            "target marginal trees contain non-sample ancestry with no sample "
            f"descendants: nodes={sorted(affected_nodes)[:10]} "
            f"tree_count={len(affected_trees)} examples={occurrences}"
        )
    return len(affected_nodes), len(affected_trees)


def _validate_local_roots(
    proposal: LocalARGProposal,
    refined: tskit.TreeSequence,
    local_node_map: dict[int, int],
    errors: list[str],
) -> None:
    """Require one parentless proposed root on every target root interval."""

    expected_left, expected_right = proposal.genomic_range
    cursor = float(expected_left)
    for left, right, node_id in proposal.root_intervals:
        left = float(left)
        right = float(right)
        if not math.isclose(left, cursor, rel_tol=0.0, abs_tol=1e-9):
            errors.append(
                "local root intervals do not exactly partition the target range"
            )
            return
        if not left < right:
            errors.append("local root interval is empty")
            return
        mapped_node = int(local_node_map.get(int(node_id), int(node_id)))
        tree = refined.at((left + right) / 2.0)
        if int(tree.parent(mapped_node)) != tskit.NULL:
            errors.append(
                "local root is not parentless on "
                f"[{left}, {right}): node={mapped_node}"
            )
            return
        if tuple(int(root) for root in tree.roots) != (mapped_node,):
            errors.append(
                "refined marginal tree does not have exactly the proposed "
                f"local root on [{left}, {right})"
            )
            return
        cursor = right
    if not math.isclose(
        cursor,
        float(expected_right),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        errors.append(
            "local root intervals do not exactly partition the target range"
        )


def _validate_local_events(
    events: tuple[LocalEventRecord, ...],
    refined: tskit.TreeSequence,
    local_node_map: dict[int, int],
    errors: list[str],
) -> None:
    edges_by_parent: dict[int, list[Any]] = {}
    for edge in refined.edges():
        edges_by_parent.setdefault(int(edge.parent), []).append(edge)
    for event in events:
        mapped_nodes = tuple(
            local_node_map[node_id]
            for node_id in event.node_ids
            if node_id in local_node_map
        )
        if event.kind == "coalescence":
            if len(mapped_nodes) != 1:
                errors.append(
                    f"sampled coalescence step {event.step} is missing its node"
                )
                continue
            node_id = mapped_nodes[0]
            if not math.isclose(
                float(refined.node(node_id).time),
                float(event.time),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                errors.append(
                    f"sampled coalescence step {event.step} changed time"
                )
            if len({int(edge.child) for edge in edges_by_parent.get(node_id, ())}) < 2:
                errors.append(
                    f"sampled coalescence step {event.step} has fewer than two children"
                )
        elif event.kind == "recombination":
            if len(mapped_nodes) != 2:
                errors.append(
                    f"sampled recombination step {event.step} is missing its node pair"
                )
                continue
            left_node, right_node = mapped_nodes
            left_record = refined.node(left_node)
            right_record = refined.node(right_node)
            if not (
                int(left_record.flags) & NODE_IS_RE_EVENT
                and int(right_record.flags) & NODE_IS_RE_EVENT
            ):
                errors.append(
                    f"sampled recombination step {event.step} lost its event flags"
                )
            if not math.isclose(
                float(left_record.time),
                float(right_record.time),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ) or not math.isclose(
                float(left_record.time),
                float(event.time),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                errors.append(
                    f"sampled recombination step {event.step} changed time"
                )
            left_edges = edges_by_parent.get(left_node, ())
            right_edges = edges_by_parent.get(right_node, ())
            left_children = {int(edge.child) for edge in left_edges}
            right_children = {int(edge.child) for edge in right_edges}
            if (
                len(left_children) != 1
                or len(right_children) != 1
                or left_children != right_children
            ):
                errors.append(
                    f"sampled recombination step {event.step} lacks one common child"
                )
            left_intervals = tuple(
                (float(edge.left), float(edge.right)) for edge in left_edges
            )
            right_intervals = tuple(
                (float(edge.left), float(edge.right)) for edge in right_edges
            )
            if (
                not left_intervals
                or not right_intervals
                or _intersect_segments(left_intervals, right_intervals)
            ):
                errors.append(
                    f"sampled recombination step {event.step} has invalid partitions"
                )
            if event.breakpoint is not None:
                if any(
                    right > float(event.breakpoint)
                    for _left, right in left_intervals
                ) or any(
                    left < float(event.breakpoint)
                    for left, _right in right_intervals
                ):
                    errors.append(
                        f"sampled recombination step {event.step} changed breakpoint"
                    )


def _validate_collapsed_local_recombination_marginals(
    events: tuple[LocalEventRecord, ...],
    refined: tskit.TreeSequence,
    local_node_map: dict[int, int],
    errors: list[str],
) -> bool:
    """Check that local routing-node suppression preserves marginal parents."""

    recombination_nodes = tuple(
        sorted(
            {
                int(local_node_map[node_id])
                for event in events
                if event.kind == "recombination"
                for node_id in event.node_ids
                if node_id in local_node_map
            }
        )
    )
    if not recombination_nodes:
        return True

    tables = refined.dump_tables()
    # This check is topological. Parsimony may legitimately place a target
    # mutation on an explicit local recombination node, so remove biological
    # references from the disposable comparison copy before suppressing nodes.
    tables.mutations.clear()
    tables.migrations.clear()
    try:
        old_to_collapsed = _collapse_nodes(tables, recombination_nodes)
        collapsed = tables.tree_sequence()
    except Exception as error:
        errors.append(
            "collapsing sampled recombination nodes failed: "
            f"{error}"
        )
        return False

    removed = set(recombination_nodes)
    breakpoints = sorted(
        {
            float(value)
            for value in (
                tuple(refined.breakpoints())
                + tuple(collapsed.breakpoints())
            )
        }
    )
    retained_nodes = tuple(
        node_id
        for node_id in range(int(refined.num_nodes))
        if node_id not in removed
    )
    for left, right in zip(breakpoints, breakpoints[1:]):
        position = (left + right) / 2.0
        refined_tree = refined.at(position)
        collapsed_tree = collapsed.at(position)
        for old_node_id in retained_nodes:
            collapsed_node_id = int(old_to_collapsed[old_node_id])
            if collapsed_node_id == tskit.NULL:
                errors.append(
                    "collapsing sampled recombination nodes removed retained "
                    f"node {old_node_id}"
                )
                return False
            expected_parent = int(refined_tree.parent(old_node_id))
            while expected_parent in removed:
                expected_parent = int(refined_tree.parent(expected_parent))
            if expected_parent != tskit.NULL:
                expected_parent = int(old_to_collapsed[expected_parent])
            actual_parent = int(
                collapsed_tree.parent(collapsed_node_id)
            )
            if actual_parent != expected_parent:
                errors.append(
                    "collapsing sampled recombination nodes changed a marginal "
                    "parent relationship on "
                    f"[{left}, {right}): node={old_node_id}"
                )
                return False
    return True


def _validate_exterior(
    source: tskit.TreeSequence,
    refined: tskit.TreeSequence,
    region: Interval,
    errors: list[str],
) -> None:
    source_signature = _outside_edge_signature(source, region)
    refined_signature = _outside_edge_signature(refined, region)
    if source_signature != refined_signature:
        errors.append("exterior edge topology or coverage changed")

    for node_id in range(int(source.num_nodes)):
        source_node = source.node(node_id)
        refined_node = refined.node(node_id)
        if (
            source_node.flags != refined_node.flags
            or source_node.time != refined_node.time
            or source_node.population != refined_node.population
            or source_node.individual != refined_node.individual
            or source_node.metadata != refined_node.metadata
        ):
            errors.append(
                f"exterior biological node {node_id} changed"
            )
            break


def _outside_edge_signature(
    ts: tskit.TreeSequence,
    region: Interval,
) -> tuple[tuple[float, float, int, int], ...]:
    values = []
    for edge in ts.edges():
        outside = _subtract_segments(
            ((float(edge.left), float(edge.right)),),
            (region,),
        )
        for left, right in outside:
            values.append((left, right, int(edge.parent), int(edge.child)))
    return _merge_edge_intervals(values)


def _validate_biological_tables(
    source: tskit.TreeSequence,
    refined: tskit.TreeSequence,
    region: Interval,
    errors: list[str],
) -> None:
    source_tables = source.tables
    refined_tables = refined.tables
    for name in ("sites", "migrations", "individuals", "populations"):
        if not getattr(source_tables, name).equals(getattr(refined_tables, name)):
            errors.append(f"biological {name} table changed")
    if _outside_mutation_signature(
        source,
        region,
    ) != _outside_mutation_signature(refined, region):
        errors.append("exterior mutation semantics changed")
    if _target_genotype_signature(
        source,
        region,
    ) != _target_genotype_signature(refined, region):
        errors.append("target sample genotypes changed")


def _outside_mutation_signature(
    ts: tskit.TreeSequence,
    region: Interval,
) -> tuple[tuple[Any, ...], ...]:
    left, right = region
    output = []
    for site in ts.sites():
        if float(left) <= float(site.position) < float(right):
            continue
        global_to_local = {
            int(mutation.id): index
            for index, mutation in enumerate(site.mutations)
        }
        mutations = []
        for mutation in site.mutations:
            parent = tskit.NULL
            if int(mutation.parent) != tskit.NULL:
                parent = int(global_to_local[int(mutation.parent)])
            mutation_time = float(mutation.time)
            mutations.append(
                (
                    int(mutation.node),
                    mutation.derived_state,
                    parent,
                    None if math.isnan(mutation_time) else mutation_time,
                    mutation.metadata,
                )
            )
        output.append(
            (
                float(site.position),
                site.ancestral_state,
                site.metadata,
                tuple(mutations),
            )
        )
    return tuple(output)


def _validate_mutations(
    refined: tskit.TreeSequence,
    region: Interval,
    errors: list[str],
) -> None:
    left, right = region
    mutations = list(refined.mutations())
    for site in refined.sites():
        position = float(site.position)
        if not left <= position < right:
            continue
        tree = refined.at(position)
        tree_nodes = set(int(node_id) for node_id in tree.nodes())
        for mutation in site.mutations:
            if int(mutation.node) not in tree_nodes:
                errors.append(
                    f"mutation {mutation.id} node {mutation.node} is absent "
                    f"from its marginal tree at {position}"
                )
                continue
            if mutation.parent != tskit.NULL:
                parent_mutation = mutations[int(mutation.parent)]
                if not tree.is_descendant(
                    int(mutation.node),
                    int(parent_mutation.node),
                ):
                    errors.append(
                        f"mutation-parent ancestry is invalid for mutation {mutation.id}"
                    )


__all__ = [
    "LOCAL_REFINEMENT_PROVENANCE_NAME",
    "LocalSpliceResult",
    "LocalValidationReport",
    "export_refined_tree_sequence",
    "splice_local_proposal",
    "validate_local_splice",
]
