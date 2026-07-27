"""User-anchored, interval-aware context selection for local ARG refinement.

This module prepares structural refinement contexts only.  It does not mutate
the input tree sequence, reconstruct a replacement history, or assign a
biological probability to the synthetic routing events used by the trace.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import tskit

from .synthetic_full_arg import SyntheticFullARGResult, build_synthetic_full_arg
from .trace import FastARGTrace, build_fast_trace_from_full_arg


Interval = tuple[float, float]
LineageRole = Literal["mutable_target", "promoted_dependency", "fixed_boundary"]
ContextStatus = Literal["valid", "invalid"]
BoundaryRole = Literal[
    "outside_tether",
    "outside_edge_piece",
    "fixed_event_partner",
]

NEAR_GLOBAL_FRACTION = 0.8


@dataclass(frozen=True)
class LocalRefinementRequest:
    """A half-open genomic range and one younger-side trace cut selector."""

    genomic_range: Interval
    cut_time: float | None = None
    cut_event_index: int | None = None

    def __post_init__(self) -> None:
        if len(self.genomic_range) != 2:
            raise ValueError("genomic_range must contain exactly two coordinates")
        genomic_range = (
            float(self.genomic_range[0]),
            float(self.genomic_range[1]),
        )
        object.__setattr__(self, "genomic_range", genomic_range)

        supplied = int(self.cut_time is not None) + int(
            self.cut_event_index is not None
        )
        if supplied != 1:
            raise ValueError(
                "exactly one of cut_time or cut_event_index must be supplied"
            )
        if self.cut_time is not None:
            cut_time = float(self.cut_time)
            if not math.isfinite(cut_time):
                raise ValueError("cut_time must be finite")
            object.__setattr__(self, "cut_time", cut_time)
        if self.cut_event_index is not None:
            if isinstance(self.cut_event_index, bool) or not isinstance(
                self.cut_event_index, numbers.Integral
            ):
                raise ValueError("cut_event_index must be an integer")
            object.__setattr__(
                self,
                "cut_event_index",
                int(self.cut_event_index),
            )


@dataclass(frozen=True)
class ResolvedTraceCut:
    """The trace state immediately before the selected event."""

    cut_step: int
    current_time: float
    next_event_index: int | None
    next_event_time: float | None
    requested_time: float | None
    requested_event_index: int | None
    time_discrepancy: float | None
    cut_side: str = "before"


@dataclass(frozen=True)
class ContextLineage:
    """One lineage participating in the mutable or fixed context."""

    node_id: int
    role: LineageRole
    active_at_cut: bool
    first_active_step: int
    first_active_time: float
    first_dependency_step: int
    first_dependency_time: float
    source_segments: tuple[Interval, ...]
    mutable_segments: tuple[Interval, ...]
    fixed_segments: tuple[Interval, ...]


@dataclass(frozen=True)
class BoundaryAttachment:
    """An immutable segment-level connection to the exterior ARG."""

    role: BoundaryRole
    event_index: int | None
    event_time: float | None
    node_ids: tuple[int, ...]
    edge_ids: tuple[int, ...]
    intervals: tuple[Interval, ...]


@dataclass(frozen=True)
class AuthorizedEdgeInterval:
    """The portion of one source edge authorized for local replacement."""

    event_index: int
    edge_id: int
    parent_node_id: int
    child_node_id: int
    left: float
    right: float


@dataclass(frozen=True)
class SelectedARGEvent:
    """One older trace event selected by interval-aware dependency propagation."""

    event_index: int
    step_after_event: int
    kind: str
    time: float
    node_ids: tuple[int, ...]
    authorized_edge_ids: tuple[int, ...]
    dependent_child_node_ids: tuple[int, ...]
    promoted_node_ids: tuple[int, ...]
    boundary_edge_ids: tuple[int, ...]
    mode: Literal["internal", "mixed_boundary"]


@dataclass(frozen=True)
class DependencyDiagnostic:
    """A structured reason why a context could not be constructed safely."""

    code: str
    message: str
    event_index: int | None = None
    event_time: float | None = None
    node_ids: tuple[int, ...] = ()
    edge_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class LocalRefinementContext:
    """Typed structural contract for a later boundary-aware local sampler."""

    request: LocalRefinementRequest
    resolved_cut: ResolvedTraceCut
    sequence_length: float
    status: ContextStatus
    cut_active_lineages: tuple[ContextLineage, ...]
    promoted_dependency_lineages: tuple[ContextLineage, ...]
    fixed_boundary_lineages: tuple[ContextLineage, ...]
    boundary_attachments: tuple[BoundaryAttachment, ...]
    selected_events: tuple[SelectedARGEvent, ...]
    authorized_edge_intervals: tuple[AuthorizedEdgeInterval, ...]
    rejection_diagnostics: tuple[DependencyDiagnostic, ...]
    complexity: Mapping[str, int | float | bool]

    @property
    def is_valid(self) -> bool:
        return self.status == "valid"

    @property
    def active_lineages(self) -> tuple[ContextLineage, ...]:
        """Mutable cut lineages followed by older promoted dependencies."""

        return self.cut_active_lineages + self.promoted_dependency_lineages

    @property
    def selected_event_indices(self) -> tuple[int, ...]:
        return tuple(event.event_index for event in self.selected_events)

    @property
    def authorized_edge_ids(self) -> tuple[int, ...]:
        return tuple(
            sorted({interval.edge_id for interval in self.authorized_edge_intervals})
        )


@dataclass(frozen=True)
class PreparedLocalRefinement:
    """Synthetic conversion, shared fast trace, and one requested context."""

    source_tree_sequence: tskit.TreeSequence
    synthetic_conversion: SyntheticFullARGResult
    trace: FastARGTrace
    context: LocalRefinementContext

    @property
    def synthetic_arg(self) -> tskit.TreeSequence:
        return self.synthetic_conversion.tree_sequence


@dataclass
class _LineageAccumulator:
    node_id: int
    role: Literal["mutable_target", "promoted_dependency"]
    active_at_cut: bool
    first_active_step: int
    first_active_time: float
    first_dependency_step: int
    first_dependency_time: float
    source_segments: tuple[Interval, ...]
    mutable_segments: tuple[Interval, ...]
    fixed_segments: tuple[Interval, ...]


def prepare_local_refinement(
    ts_or_path: str | Path | tskit.TreeSequence,
    request: LocalRefinementRequest,
) -> PreparedLocalRefinement:
    """Convert an inferred tree sequence, build its trace, and select a context."""

    source_tree_sequence = (
        ts_or_path
        if isinstance(ts_or_path, tskit.TreeSequence)
        else tskit.load(str(Path(ts_or_path)))
    )
    conversion = build_synthetic_full_arg(
        source_tree_sequence,
        split_rule="balanced",
        ensure_unique_event_times=True,
    )
    trace = build_fast_trace_from_full_arg(
        conversion.tree_sequence,
        require_unique_event_times=True,
        allow_no_recombination=True,
    )
    context = trace_local_dependencies(trace, request)
    return PreparedLocalRefinement(
        source_tree_sequence=source_tree_sequence,
        synthetic_conversion=conversion,
        trace=trace,
        context=context,
    )


def resolve_trace_cut(
    trace: FastARGTrace,
    request: LocalRefinementRequest,
) -> ResolvedTraceCut:
    """Resolve a request to the state before its selected trace event."""

    _validate_genomic_range(request.genomic_range, trace.sequence_length)
    if trace.event_count > 1 and np.any(np.diff(trace.event_time) <= 0.0):
        raise ValueError(
            "local refinement requires finite, strictly increasing event times"
        )
    if trace.event_count and np.any(~np.isfinite(trace.event_time)):
        raise ValueError(
            "local refinement requires finite, strictly increasing event times"
        )

    requested_time = request.cut_time
    requested_event_index = request.cut_event_index
    if requested_time is not None:
        if requested_time < 0.0:
            raise ValueError("cut_time must be at least zero")
        if trace.event_count == 0:
            if requested_time != 0.0:
                raise ValueError("an event-free trace only supports cut_time=0")
            cut_step = 0
        else:
            oldest_time = float(trace.event_time[-1])
            if requested_time > oldest_time:
                raise ValueError(
                    f"cut_time {requested_time} is older than the terminal "
                    f"trace event at {oldest_time}"
                )
            cut_step = int(
                np.searchsorted(trace.event_time, requested_time, side="left")
            )
    else:
        assert requested_event_index is not None
        if not 0 <= requested_event_index < trace.event_count:
            raise ValueError(
                "cut_event_index must be in "
                f"[0, {trace.event_count - 1}], got {requested_event_index}"
            )
        cut_step = int(requested_event_index)

    current_time = (
        0.0 if cut_step == 0 else float(trace.event_time[cut_step - 1])
    )
    next_event_index = cut_step if cut_step < trace.event_count else None
    next_event_time = (
        None
        if next_event_index is None
        else float(trace.event_time[next_event_index])
    )
    discrepancy = (
        None
        if requested_time is None or next_event_time is None
        else float(next_event_time - requested_time)
    )
    return ResolvedTraceCut(
        cut_step=cut_step,
        current_time=current_time,
        next_event_index=next_event_index,
        next_event_time=next_event_time,
        requested_time=requested_time,
        requested_event_index=requested_event_index,
        time_discrepancy=discrepancy,
    )


def trace_local_dependencies(
    trace: FastARGTrace,
    request: LocalRefinementRequest,
) -> LocalRefinementContext:
    """Select target-dependent older history while freezing its exterior.

    All active material intersecting the requested interval seeds the scan.
    Older events are visited once in increasing time.  Events with no overlap
    with dependent material remain exterior; selected mixed events retain their
    outside pieces as explicit immutable boundary attachments.
    """

    resolved = resolve_trace_cut(trace, request)
    region = _validate_genomic_range(
        request.genomic_range,
        trace.sequence_length,
    )
    state = trace.initial_state().advance_to(resolved.cut_step)
    cut_frontier = state.compact_active_frontier()
    cut_material = _frontier_material(cut_frontier)

    accumulators: dict[int, _LineageAccumulator] = {}
    dependent: dict[int, tuple[Interval, ...]] = {}
    attachments: list[BoundaryAttachment] = []
    fixed_segments_by_node: dict[int, tuple[Interval, ...]] = {}
    selected_events: list[SelectedARGEvent] = []
    authorized: list[AuthorizedEdgeInterval] = []
    diagnostics: list[DependencyDiagnostic] = []

    for node_id, source_segments in cut_material.items():
        mutable_segments = _intersect_interval(source_segments, region)
        if not mutable_segments:
            continue
        dependent[node_id] = mutable_segments
        fixed_segments = _subtract_segments(source_segments, mutable_segments)
        _record_mutable_lineage(
            accumulators,
            trace,
            node_id=node_id,
            role="mutable_target",
            active_at_cut=True,
            dependency_step=resolved.cut_step,
            dependency_time=resolved.current_time,
            source_segments=source_segments,
            mutable_segments=mutable_segments,
        )
        if fixed_segments:
            _add_attachment(
                attachments,
                fixed_segments_by_node,
                BoundaryAttachment(
                    role="outside_tether",
                    event_index=None,
                    event_time=None,
                    node_ids=(node_id,),
                    edge_ids=(),
                    intervals=fixed_segments,
                ),
            )

    if not dependent:
        diagnostics.append(
            DependencyDiagnostic(
                code="empty_target_frontier",
                message=(
                    "no active lineage carries material inside the requested "
                    "genomic range at the resolved cut"
                ),
            )
        )
        return _assemble_context(
            trace=trace,
            request=request,
            resolved=resolved,
            status="invalid",
            accumulators=accumulators,
            fixed_segments_by_node=fixed_segments_by_node,
            attachments=attachments,
            selected_events=selected_events,
            authorized=authorized,
            diagnostics=diagnostics,
            cut_frontier_lineage_count=len(cut_frontier),
            cut_frontier_node_ids=tuple(int(value) for value in cut_frontier.node_ids),
            scanned_event_count=0,
        )

    cursor = state
    scanned_event_count = 0
    for event_index in range(resolved.cut_step, trace.num_steps):
        scanned_event_count += 1
        edge_ids = _event_edge_ids(trace, event_index)
        triggered_edge_ids = [
            edge_id
            for edge_id in edge_ids
            if _intersect_interval(
                dependent.get(int(trace.edge_child[edge_id]), ()),
                (
                    float(trace.edge_left[edge_id]),
                    float(trace.edge_right[edge_id]),
                ),
            )
        ]
        if not triggered_edge_ids:
            continue

        try:
            cursor.advance_to(event_index)
        except (ValueError, RuntimeError) as error:
            diagnostics.append(
                _transition_diagnostic(trace, event_index, edge_ids, error)
            )
            break

        before_material = _frontier_material(cursor.compact_active_frontier())
        event = trace.event_at_index(event_index)
        target_by_edge: dict[int, tuple[Interval, ...]] = {}
        participant_children: set[int] = set()
        unavailable: list[int] = []
        for edge_id in edge_ids:
            child_id = int(trace.edge_child[edge_id])
            edge_segment = (
                float(trace.edge_left[edge_id]),
                float(trace.edge_right[edge_id]),
            )
            target_segments = _intersect_interval((edge_segment,), region)
            if not target_segments:
                continue
            available_target = _intersect_segments(
                before_material.get(child_id, ()),
                target_segments,
            )
            if available_target != target_segments:
                unavailable.append(edge_id)
                continue
            target_by_edge[edge_id] = target_segments
            participant_children.add(child_id)

        if unavailable:
            diagnostics.append(
                DependencyDiagnostic(
                    code="unresolved_event_coupling",
                    message=(
                        "a selected event requires target material from a child "
                        "that is not active at the event boundary"
                    ),
                    event_index=event_index,
                    event_time=event.time,
                    node_ids=tuple(
                        sorted(
                            {
                                int(trace.edge_child[edge_id])
                                for edge_id in unavailable
                            }
                        )
                    ),
                    edge_ids=tuple(sorted(unavailable)),
                )
            )
            break

        newly_promoted: set[int] = set()
        for child_id in sorted(participant_children):
            source_segments = before_material[child_id]
            child_target = _intersect_interval(source_segments, region)
            if child_id not in dependent:
                newly_promoted.add(child_id)
                dependent[child_id] = child_target
            else:
                dependent[child_id] = _canonical_segments(
                    dependent[child_id] + child_target
                )
            if child_id not in accumulators:
                newly_promoted.add(child_id)
            _record_mutable_lineage(
                accumulators,
                trace,
                node_id=child_id,
                role="promoted_dependency",
                active_at_cut=False,
                dependency_step=event_index,
                dependency_time=float(cursor.current_time),
                source_segments=source_segments,
                mutable_segments=child_target,
            )

        next_dependent = dict(dependent)
        parent_target: dict[int, tuple[Interval, ...]] = {}
        boundary_edge_ids: set[int] = set()
        for edge_id in edge_ids:
            parent_id = int(trace.edge_parent[edge_id])
            child_id = int(trace.edge_child[edge_id])
            edge_segment = (
                float(trace.edge_left[edge_id]),
                float(trace.edge_right[edge_id]),
            )
            target_segments = target_by_edge.get(edge_id, ())
            if target_segments:
                current = next_dependent.get(child_id, ())
                remaining = _subtract_segments(current, (edge_segment,))
                if remaining:
                    next_dependent[child_id] = remaining
                else:
                    next_dependent.pop(child_id, None)
                parent_target[parent_id] = _canonical_segments(
                    parent_target.get(parent_id, ()) + target_segments
                )
                for left, right in target_segments:
                    authorized.append(
                        AuthorizedEdgeInterval(
                            event_index=event_index,
                            edge_id=edge_id,
                            parent_node_id=parent_id,
                            child_node_id=child_id,
                            left=left,
                            right=right,
                        )
                    )

                fixed_edge_segments = _subtract_segments(
                    (edge_segment,),
                    target_segments,
                )
                if fixed_edge_segments:
                    boundary_edge_ids.add(edge_id)
                    _add_attachment(
                        attachments,
                        fixed_segments_by_node,
                        BoundaryAttachment(
                            role="outside_edge_piece",
                            event_index=event_index,
                            event_time=float(event.time),
                            node_ids=(parent_id, child_id),
                            edge_ids=(edge_id,),
                            intervals=fixed_edge_segments,
                        ),
                    )
            else:
                boundary_edge_ids.add(edge_id)
                _add_attachment(
                    attachments,
                    fixed_segments_by_node,
                    BoundaryAttachment(
                        role="fixed_event_partner",
                        event_index=event_index,
                        event_time=float(event.time),
                        node_ids=(parent_id, child_id),
                        edge_ids=(edge_id,),
                        intervals=(edge_segment,),
                    ),
                )

        try:
            cursor.advance()
        except (ValueError, RuntimeError) as error:
            diagnostics.append(
                _transition_diagnostic(trace, event_index, edge_ids, error)
            )
            break

        after_material = _frontier_material(cursor.compact_active_frontier())
        invalid_parent_edges = []
        for parent_id, target_segments in parent_target.items():
            if trace.node_reveal_step[parent_id] != event_index + 1:
                invalid_parent_edges.extend(
                    edge_id
                    for edge_id in target_by_edge
                    if int(trace.edge_parent[edge_id]) == parent_id
                )
                continue
            parent_source = after_material.get(parent_id, ())
            if _intersect_segments(parent_source, target_segments) != target_segments:
                invalid_parent_edges.extend(
                    edge_id
                    for edge_id in target_by_edge
                    if int(trace.edge_parent[edge_id]) == parent_id
                )
                continue
            next_dependent[parent_id] = _canonical_segments(
                next_dependent.get(parent_id, ()) + target_segments
            )
            if parent_id not in accumulators:
                newly_promoted.add(parent_id)
            _record_mutable_lineage(
                accumulators,
                trace,
                node_id=parent_id,
                role="promoted_dependency",
                active_at_cut=False,
                dependency_step=event_index + 1,
                dependency_time=float(event.time),
                source_segments=parent_source,
                mutable_segments=_intersect_interval(parent_source, region),
            )

        if invalid_parent_edges:
            diagnostics.append(
                DependencyDiagnostic(
                    code="unresolved_event_coupling",
                    message=(
                        "a selected event did not reveal its required target-bearing "
                        "parent lineage at the expected trace step"
                    ),
                    event_index=event_index,
                    event_time=event.time,
                    node_ids=event.node_ids,
                    edge_ids=tuple(sorted(set(invalid_parent_edges))),
                )
            )
            break

        dependent = {
            node_id: segments
            for node_id, segments in next_dependent.items()
            if segments
        }
        selected_edge_ids = tuple(sorted(target_by_edge))
        selected_events.append(
            SelectedARGEvent(
                event_index=event_index,
                step_after_event=event_index + 1,
                kind=event.kind,
                time=float(event.time),
                node_ids=event.node_ids,
                authorized_edge_ids=selected_edge_ids,
                dependent_child_node_ids=tuple(sorted(participant_children)),
                promoted_node_ids=tuple(sorted(newly_promoted)),
                boundary_edge_ids=tuple(sorted(boundary_edge_ids)),
                mode=(
                    "mixed_boundary"
                    if boundary_edge_ids
                    else "internal"
                ),
            )
        )

    status: ContextStatus = "invalid" if diagnostics else "valid"
    if status == "valid":
        try:
            cursor.advance_to(trace.num_steps)
        except (ValueError, RuntimeError) as error:
            diagnostics.append(
                DependencyDiagnostic(
                    code="trace_replay_error",
                    message=str(error),
                )
            )
            status = "invalid"

    return _assemble_context(
        trace=trace,
        request=request,
        resolved=resolved,
        status=status,
        accumulators=accumulators,
        fixed_segments_by_node=fixed_segments_by_node,
        attachments=attachments,
        selected_events=selected_events,
        authorized=authorized,
        diagnostics=diagnostics,
        cut_frontier_lineage_count=len(cut_frontier),
        cut_frontier_node_ids=tuple(int(value) for value in cut_frontier.node_ids),
        scanned_event_count=scanned_event_count,
    )


def _assemble_context(
    *,
    trace: FastARGTrace,
    request: LocalRefinementRequest,
    resolved: ResolvedTraceCut,
    status: ContextStatus,
    accumulators: Mapping[int, _LineageAccumulator],
    fixed_segments_by_node: Mapping[int, tuple[Interval, ...]],
    attachments: list[BoundaryAttachment],
    selected_events: list[SelectedARGEvent],
    authorized: list[AuthorizedEdgeInterval],
    diagnostics: list[DependencyDiagnostic],
    cut_frontier_lineage_count: int,
    cut_frontier_node_ids: tuple[int, ...],
    scanned_event_count: int,
) -> LocalRefinementContext:
    mutable_records = tuple(
        _freeze_accumulator(item)
        for item in sorted(accumulators.values(), key=lambda item: item.node_id)
    )
    cut_lineages = tuple(item for item in mutable_records if item.active_at_cut)
    promoted = tuple(item for item in mutable_records if not item.active_at_cut)
    cut_frontier_node_set = set(cut_frontier_node_ids)
    boundary_records = []
    for node_id, segments in sorted(fixed_segments_by_node.items()):
        if not segments:
            continue
        reveal_step = int(trace.node_reveal_step[node_id])
        attachment_steps = []
        for attachment in attachments:
            if node_id not in attachment.node_ids:
                continue
            if attachment.role == "outside_tether":
                attachment_steps.append(resolved.cut_step)
            elif attachment.event_index is not None:
                attachment_steps.append(
                    max(reveal_step, int(attachment.event_index))
                )
        dependency_step = min(attachment_steps, default=reveal_step)
        if dependency_step == 0:
            dependency_time = 0.0
        elif dependency_step <= trace.num_steps:
            dependency_time = float(trace.event_time[dependency_step - 1])
        else:
            # Preserve a usable diagnostic context for a malformed low-level
            # trace whose reveal index lies outside its event schedule.
            dependency_time = float(trace.node_time[node_id])
        boundary_records.append(
            ContextLineage(
                node_id=int(node_id),
                role="fixed_boundary",
                active_at_cut=node_id in cut_frontier_node_set,
                first_active_step=reveal_step,
                first_active_time=float(trace.node_time[node_id]),
                first_dependency_step=dependency_step,
                first_dependency_time=dependency_time,
                source_segments=segments,
                mutable_segments=(),
                fixed_segments=segments,
            )
        )
    boundary = tuple(boundary_records)

    selected_edge_ids = {item.edge_id for item in authorized}
    mutable_node_ids = {item.node_id for item in mutable_records}
    event_fraction = (
        len(selected_events) / trace.event_count if trace.event_count else 0.0
    )
    edge_fraction = (
        len(selected_edge_ids) / trace.edge_parent.size
        if trace.edge_parent.size
        else 0.0
    )
    node_fraction = (
        len(mutable_node_ids) / trace.node_time.size
        if trace.node_time.size
        else 0.0
    )
    near_global = max(event_fraction, edge_fraction, node_fraction) >= (
        NEAR_GLOBAL_FRACTION
    )
    complexity: dict[str, int | float | bool] = {
        "cut_frontier_lineage_count": int(cut_frontier_lineage_count),
        "cut_target_lineage_count": len(cut_lineages),
        "promoted_dependency_lineage_count": len(promoted),
        "fixed_boundary_lineage_count": len(boundary),
        "boundary_attachment_count": len(attachments),
        "scanned_event_count": int(scanned_event_count),
        "selected_event_count": len(selected_events),
        "selected_edge_count": len(selected_edge_ids),
        "authorized_edge_interval_count": len(authorized),
        "authorized_material_length": float(
            sum(item.right - item.left for item in authorized)
        ),
        "mutable_node_fraction": float(node_fraction),
        "selected_event_fraction": float(event_fraction),
        "selected_edge_fraction": float(edge_fraction),
        "near_global_threshold": float(NEAR_GLOBAL_FRACTION),
        "near_global": bool(near_global),
    }
    return LocalRefinementContext(
        request=request,
        resolved_cut=resolved,
        sequence_length=float(trace.sequence_length),
        status=status,
        cut_active_lineages=cut_lineages,
        promoted_dependency_lineages=promoted,
        fixed_boundary_lineages=boundary,
        boundary_attachments=tuple(attachments),
        selected_events=tuple(selected_events),
        authorized_edge_intervals=tuple(authorized),
        rejection_diagnostics=tuple(diagnostics),
        complexity=complexity,
    )


def _record_mutable_lineage(
    accumulators: dict[int, _LineageAccumulator],
    trace: FastARGTrace,
    *,
    node_id: int,
    role: Literal["mutable_target", "promoted_dependency"],
    active_at_cut: bool,
    dependency_step: int,
    dependency_time: float,
    source_segments: tuple[Interval, ...],
    mutable_segments: tuple[Interval, ...],
) -> None:
    source_segments = _canonical_segments(source_segments)
    mutable_segments = _canonical_segments(mutable_segments)
    fixed_segments = _subtract_segments(source_segments, mutable_segments)
    existing = accumulators.get(int(node_id))
    if existing is None:
        reveal_step = int(trace.node_reveal_step[int(node_id)])
        accumulators[int(node_id)] = _LineageAccumulator(
            node_id=int(node_id),
            role=role,
            active_at_cut=bool(active_at_cut),
            first_active_step=reveal_step,
            first_active_time=float(trace.node_time[int(node_id)]),
            first_dependency_step=int(dependency_step),
            first_dependency_time=float(dependency_time),
            source_segments=source_segments,
            mutable_segments=mutable_segments,
            fixed_segments=fixed_segments,
        )
        return
    existing.source_segments = _canonical_segments(
        existing.source_segments + source_segments
    )
    existing.mutable_segments = _canonical_segments(
        existing.mutable_segments + mutable_segments
    )
    existing.fixed_segments = _subtract_segments(
        existing.source_segments,
        existing.mutable_segments,
    )
    existing.active_at_cut = existing.active_at_cut or bool(active_at_cut)
    if active_at_cut:
        existing.role = "mutable_target"


def _freeze_accumulator(item: _LineageAccumulator) -> ContextLineage:
    return ContextLineage(
        node_id=item.node_id,
        role=item.role,
        active_at_cut=item.active_at_cut,
        first_active_step=item.first_active_step,
        first_active_time=item.first_active_time,
        first_dependency_step=item.first_dependency_step,
        first_dependency_time=item.first_dependency_time,
        source_segments=item.source_segments,
        mutable_segments=item.mutable_segments,
        fixed_segments=item.fixed_segments,
    )


def _add_attachment(
    attachments: list[BoundaryAttachment],
    fixed_segments_by_node: dict[int, tuple[Interval, ...]],
    attachment: BoundaryAttachment,
) -> None:
    if not attachment.intervals:
        return
    attachments.append(attachment)
    for node_id in attachment.node_ids:
        fixed_segments_by_node[int(node_id)] = _canonical_segments(
            fixed_segments_by_node.get(int(node_id), ()) + attachment.intervals
        )


def _transition_diagnostic(
    trace: FastARGTrace,
    event_index: int,
    edge_ids: tuple[int, ...],
    error: Exception,
) -> DependencyDiagnostic:
    event = trace.event_at_index(event_index)
    return DependencyDiagnostic(
        code="unresolved_event_coupling",
        message=str(error),
        event_index=event_index,
        event_time=float(event.time),
        node_ids=event.node_ids,
        edge_ids=edge_ids,
    )


def _frontier_material(frontier: Any) -> dict[int, tuple[Interval, ...]]:
    return {
        int(node_id): _canonical_segments(
            frontier.segments_for_index(lineage_index)
        )
        for lineage_index, node_id in enumerate(frontier.node_ids)
    }


def _event_edge_ids(trace: FastARGTrace, event_index: int) -> tuple[int, ...]:
    start = int(trace.event_edge_start[event_index])
    end = int(trace.event_edge_start[event_index + 1])
    return tuple(int(value) for value in trace.revealed_edge_ids[start:end])


def _validate_genomic_range(
    genomic_range: Interval,
    sequence_length: float,
) -> Interval:
    left, right = float(genomic_range[0]), float(genomic_range[1])
    if not (
        math.isfinite(left)
        and math.isfinite(right)
        and 0.0 <= left < right <= float(sequence_length)
    ):
        raise ValueError(
            "genomic_range must satisfy "
            f"0 <= left < right <= {float(sequence_length)}"
        )
    return left, right


def _canonical_segments(segments: tuple[Interval, ...]) -> tuple[Interval, ...]:
    merged: list[Interval] = []
    for left, right in sorted(
        (float(left), float(right))
        for left, right in segments
        if float(left) < float(right)
    ):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return tuple(merged)


def _intersect_interval(
    segments: tuple[Interval, ...],
    interval: Interval,
) -> tuple[Interval, ...]:
    left, right = interval
    return _canonical_segments(
        tuple(
            (max(segment_left, left), min(segment_right, right))
            for segment_left, segment_right in segments
            if max(segment_left, left) < min(segment_right, right)
        )
    )


def _intersect_segments(
    left_segments: tuple[Interval, ...],
    right_segments: tuple[Interval, ...],
) -> tuple[Interval, ...]:
    output: list[Interval] = []
    left_values = _canonical_segments(left_segments)
    right_values = _canonical_segments(right_segments)
    left_index = 0
    right_index = 0
    while left_index < len(left_values) and right_index < len(right_values):
        left_start, left_end = left_values[left_index]
        right_start, right_end = right_values[right_index]
        overlap_start = max(left_start, right_start)
        overlap_end = min(left_end, right_end)
        if overlap_start < overlap_end:
            output.append((overlap_start, overlap_end))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return tuple(output)


def _subtract_segments(
    source_segments: tuple[Interval, ...],
    removed_segments: tuple[Interval, ...],
) -> tuple[Interval, ...]:
    remaining = list(_canonical_segments(source_segments))
    for remove_left, remove_right in _canonical_segments(removed_segments):
        updated: list[Interval] = []
        for source_left, source_right in remaining:
            overlap_left = max(source_left, remove_left)
            overlap_right = min(source_right, remove_right)
            if overlap_left >= overlap_right:
                updated.append((source_left, source_right))
                continue
            if source_left < overlap_left:
                updated.append((source_left, overlap_left))
            if overlap_right < source_right:
                updated.append((overlap_right, source_right))
        remaining = updated
    return tuple(remaining)


__all__ = [
    "AuthorizedEdgeInterval",
    "BoundaryAttachment",
    "ContextLineage",
    "DependencyDiagnostic",
    "LocalRefinementContext",
    "LocalRefinementRequest",
    "PreparedLocalRefinement",
    "ResolvedTraceCut",
    "SelectedARGEvent",
    "prepare_local_refinement",
    "resolve_trace_cut",
    "trace_local_dependencies",
]
