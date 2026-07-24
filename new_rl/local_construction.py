"""Prior-driven local ARG reconstruction using the main ``ARGState`` model.

The local construction starts at a traced time cut, carries only material in
the requested genomic interval, and proceeds backward under the existing
coalescent-with-recombination prior. Source lineages coupled to immutable
exterior material are attached to their active descendants at their fixed
times. Construction finishes when every target block has exactly one active
root; no original upper ancestry or terminal attachment is required.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Iterable, Literal, Mapping, Union

import numpy as np

try:
    from ..env import (
        ARGLineage,
        ARGState,
        CoalescenceChoice,
        MaterialSegments,
        PriorActionOptions,
        RecombinationChoice,
        SimpleARGEnvironment,
        SimpleTrajectory,
        action_as_dict,
    )
except ImportError:  # Support the repository's legacy top-level new_rl import.
    from env import (
        ARGLineage,
        ARGState,
        CoalescenceChoice,
        MaterialSegments,
        PriorActionOptions,
        RecombinationChoice,
        SimpleARGEnvironment,
        SimpleTrajectory,
        action_as_dict,
    )
from .local_refinement import (
    AuthorizedEdgeInterval,
    LocalRefinementContext,
    PreparedLocalRefinement,
    _canonical_segments,
)
from .synthetic_full_arg import NODE_IS_RE_EVENT


Interval = tuple[float, float]
LocalPriorAction = Union[CoalescenceChoice, RecombinationChoice]


@dataclass(frozen=True)
class ConstructionDiagnostic:
    code: str
    message: str
    step: int | None = None
    lineage_ids: tuple[int, ...] = ()
    node_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class LocalNodeRecord:
    node_id: int
    kind: Literal["coalescence", "recombination"]
    time: float
    flags: int


@dataclass(frozen=True)
class LocalEdgeRecord:
    left: float
    right: float
    parent_node_id: int
    child_node_id: int


@dataclass(frozen=True)
class LocalEventRecord:
    step: int
    kind: Literal["coalescence", "recombination"]
    time: float
    action: dict[str, Any]
    input_lineage_ids: tuple[int, ...]
    output_lineage_ids: tuple[int, ...]
    node_ids: tuple[int, ...]
    edge_indices: tuple[int, ...]
    breakpoint: float | None = None


@dataclass(frozen=True)
class LocalARGProposal:
    genomic_range: Interval
    cut_time: float
    nodes: tuple[LocalNodeRecord, ...]
    edges: tuple[LocalEdgeRecord, ...]
    events: tuple[LocalEventRecord, ...]
    root_intervals: tuple[tuple[float, float, int], ...]
    authorized_edge_intervals: tuple[AuthorizedEdgeInterval, ...]
    prior_log_probability: float
    transition_records: tuple[dict[str, Any], ...]
    status: Literal["terminal", "invalid"]
    diagnostics: tuple[ConstructionDiagnostic, ...] = ()

    @property
    def is_valid(self) -> bool:
        return self.status == "terminal" and not self.diagnostics

    @property
    def topology_digest(self) -> str:
        payload = {
            "nodes": [
                (node.kind, round(node.time, 12), node.flags)
                for node in self.nodes
            ],
            "edges": sorted(
                (
                    round(edge.left, 12),
                    round(edge.right, 12),
                    edge.parent_node_id,
                    edge.child_node_id,
                )
                for edge in self.edges
            ),
            "events": [
                (
                    event.kind,
                    round(event.time, 12),
                    None
                    if event.breakpoint is None
                    else round(event.breakpoint, 12),
                )
                for event in self.events
            ],
            "roots": self.root_intervals,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class LocalSamplingConfig:
    sample_count: int = 1
    seed: int = 1
    max_generated_events: int | None = None
    max_searched_states: int | None = None
    max_restarts: int | None = None
    allow_duplicate_topologies: bool = False

    def __post_init__(self) -> None:
        if int(self.sample_count) <= 0:
            raise ValueError("sample_count must be positive")
        if (
            self.max_generated_events is not None
            and int(self.max_generated_events) < 0
        ):
            raise ValueError("max_generated_events must be non-negative")
        if (
            self.max_searched_states is not None
            and int(self.max_searched_states) <= 0
        ):
            raise ValueError("max_searched_states must be positive")
        if self.max_restarts is not None and int(self.max_restarts) <= 0:
            raise ValueError("max_restarts must be positive")


@dataclass(frozen=True)
class LocalSampleBatch:
    proposals: tuple[LocalARGProposal, ...]
    trajectories: tuple[SimpleTrajectory, ...]
    diagnostics: tuple[ConstructionDiagnostic, ...]
    seed: int
    transition_count: int
    restart_count: int

    @property
    def is_complete(self) -> bool:
        return not self.diagnostics


def initialize_local_arg_state(
    prepared: PreparedLocalRefinement,
    env: SimpleARGEnvironment,
) -> ARGState:
    """Create an ``ARGState`` from target-bearing lineages at the trace cut."""

    if not prepared.context.is_valid:
        reasons = "; ".join(
            item.message for item in prepared.context.rejection_diagnostics
        )
        raise ValueError(f"local refinement context is invalid: {reasons}")
    if not env.structural_only:
        raise ValueError(
            "local prior construction requires structural_only=True"
        )
    if not math.isclose(
        float(env.sequence_length),
        float(prepared.context.sequence_length),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "environment sequence_length must equal the tree-sequence length"
        )

    boundaries = _environment_block_boundaries(env)
    target_material = _intervals_to_material(
        (prepared.context.request.genomic_range,),
        boundaries,
    )
    if target_material.count == 0:
        raise ValueError("the requested genomic range contains no ARG blocks")

    endpoint_material: dict[int, MaterialSegments] = {}
    for lineage in prepared.context.cut_active_lineages:
        collapsed = _collapse_temporary_source_endpoint(
            prepared,
            int(lineage.node_id),
            lineage.mutable_segments,
        )
        for endpoint_node_id, intervals in collapsed.items():
            material = _intervals_to_material(intervals, boundaries)
            if material.count == 0:
                continue
            endpoint_material[endpoint_node_id] = (
                endpoint_material.get(endpoint_node_id, MaterialSegments())
                .union(material)
            )
    if not endpoint_material:
        raise ValueError("no target-bearing cut lineage remains after routing collapse")

    time_scale = 2.0 * float(env.population_size)
    if not time_scale > 0.0:
        raise ValueError("population_size must define a positive 2Ne time scale")

    active_lineages = []
    all_nodes: dict[int, ARGLineage] = {}
    for node_id, material in sorted(endpoint_material.items()):
        source_node = prepared.synthetic_arg.node(node_id)
        lineage = ARGLineage(
            node_id=node_id,
            children=[],
            parents=[],
            material_segments=material,
            num_blocks=env.num_blocks,
            partials=None,
            sequences_indices=[],
            event_type="cut",
            time=float(source_node.time) / time_scale,
        )
        active_lineages.append(lineage)
        all_nodes[node_id] = lineage

    generated_node_start = int(prepared.synthetic_arg.num_nodes)
    current_time = (
        float(prepared.context.resolved_cut.current_time) / time_scale
    )
    schedule = _compile_fixed_ancestor_schedule(
        prepared,
        boundaries,
        time_scale,
        endpoint_material,
    )
    state = ARGState(
        active_lineages=active_lineages,
        all_nodes=all_nodes,
        max_node_idx=generated_node_start - 1,
        accumulated_log_prior=0.0,
        partial_log_reward=0.0,
        is_done=False,
        total_active_blocks=sum(
            lineage.material_segments.count
            for lineage in active_lineages
        ),
        current_time=current_time,
        target_material=target_material,
        block_boundaries=boundaries,
        time_scale=time_scale,
        generated_node_start=generated_node_start,
        transition_records=[
            {
                "event_type": "initialization",
                "time": float(
                    prepared.context.resolved_cut.current_time
                ),
                "scaled_time": current_time,
                "time_scale": time_scale,
                "population_size": float(env.population_size),
                "recombination_rate": float(env.recombination_rate),
                "rho": float(env.rho),
                "num_blocks": int(env.num_blocks),
                "block_mode": "vcf" if env.is_vcf_mode else "uniform",
            }
        ],
        fixed_ancestor_schedule=schedule,
    )
    state = reveal_due_fixed_ancestors(
        state,
        prepared.context,
        current_time,
    )
    state.is_done = local_is_terminal(state, prepared.context)
    return state


def enumerate_local_prior_actions(
    state: ARGState,
    context: LocalRefinementContext,
    env: SimpleARGEnvironment,
) -> PriorActionOptions:
    """Return locally authorized CWR choices and their aggregate rates."""

    _require_local_state(state)
    coal_actions, recomb_actions = env.enumerate_actions(state)

    legal_coal = []
    for action in coal_actions:
        left = state.active_lineages[action.active_lineage_i]
        right = state.active_lineages[action.active_lineage_j]
        if (
            left.event_type == "fixed_source"
            and right.event_type == "fixed_source"
        ):
            continue
        if not left.material_segments.overlaps(right.material_segments):
            continue
        legal_coal.append(action)

    legal_recomb = []
    for action in recomb_actions:
        lineage = state.active_lineages[action.active_lineage_i]
        if lineage.event_type == "fixed_source":
            continue
        if not _has_valid_breakpoint(lineage.material_segments, env):
            continue
        legal_recomb.append(action)

    rates = env.compute_event_rates((legal_coal, legal_recomb))
    options = PriorActionOptions(
        coal_actions=tuple(legal_coal),
        recomb_choices=tuple(legal_recomb),
        rates=rates,
    )
    state.rates = rates
    state.prior_options = options
    return options


def sample_local_prior_action(
    state: ARGState,
    context: LocalRefinementContext,
    env: SimpleARGEnvironment,
    rng: np.random.Generator,
) -> tuple[LocalPriorAction | None, float]:
    """Sample one prior action, or ``None`` for a fixed-ancestor attachment."""

    if state.is_done:
        raise ValueError("cannot sample an action from a terminal local state")
    options = enumerate_local_prior_actions(state, context, env)
    total_rate = float(
        options.rates["lambda_coal"]
        + options.rates["lambda_recomb"]
    )
    next_fixed_time = _next_fixed_ancestor_time(state)

    if total_rate <= 0.0:
        if next_fixed_time is None:
            raise ValueError(
                "nonterminal local state has no legal prior action or fixed reveal"
            )
        return None, 0.0

    max_delta = (
        None
        if next_fixed_time is None
        else max(0.0, float(next_fixed_time - state.current_time))
    )
    if max_delta is None:
        event_masses = tuple(
            env.time_env.time_action_probabilities(total_rate)
        )
        survival_mass = 0.0
    else:
        event_masses, survival_mass = (
            env.time_env.bounded_waiting_distribution(
                total_rate,
                max_delta,
            )
        )

    outcomes = np.asarray(
        event_masses + ((survival_mass,) if max_delta is not None else ()),
        dtype=np.float64,
    )
    outcome = int(rng.choice(outcomes.size, p=outcomes / outcomes.sum()))
    if max_delta is not None and outcome == len(event_masses):
        return None, math.log(survival_mass) if survival_mass > 0.0 else -math.inf

    time_action = outcome
    delta_time = env.time_env.time_action_to_delta(
        time_action,
        total_rate,
        max_delta=max_delta,
    )
    wait_log_probability = math.log(event_masses[time_action])

    lambda_coal = float(options.rates["lambda_coal"])
    lambda_recomb = float(options.rates["lambda_recomb"])
    event_probabilities = np.asarray(
        [lambda_coal, lambda_recomb],
        dtype=np.float64,
    ) / total_rate
    event_type = int(rng.choice(2, p=event_probabilities))

    if event_type == 0:
        if not options.coal_actions:
            raise RuntimeError("coalescence rate is positive without legal pairs")
        action_index = int(rng.integers(len(options.coal_actions)))
        action = replace(
            options.coal_actions[action_index],
            time_action=time_action,
            delta_time=float(delta_time),
        )
        choice_log_probability = (
            math.log(lambda_coal / total_rate)
            - math.log(len(options.coal_actions))
        )
    else:
        if not options.recomb_choices:
            raise RuntimeError(
                "recombination rate is positive without legal lineages"
            )
        weights = np.asarray(
            [
                _local_recombination_weight(
                    state.active_lineages[choice.active_lineage_i],
                    env,
                )
                for choice in options.recomb_choices
            ],
            dtype=np.float64,
        )
        choice_index = int(rng.choice(len(weights), p=weights / weights.sum()))
        choice = options.recomb_choices[choice_index]
        lineage = state.active_lineages[choice.active_lineage_i]
        breakpoint, breakpoint_probability = _sample_breakpoint(
            lineage.material_segments,
            env,
            rng,
        )
        action = replace(
            choice,
            breakpoint=int(breakpoint),
            time_action=time_action,
            delta_time=float(delta_time),
        )
        choice_log_probability = (
            math.log(lambda_recomb / total_rate)
            + math.log(weights[choice_index] / weights.sum())
            + math.log(breakpoint_probability)
        )

    return action, wait_log_probability + choice_log_probability


def apply_local_action(
    state: ARGState,
    action: LocalPriorAction,
    context: LocalRefinementContext,
    env: SimpleARGEnvironment,
    log_prior: float,
) -> ARGState:
    """Apply one caller-selected local CWR action to an ``ARGState``."""

    if state.is_done:
        raise ValueError("cannot apply an action to a terminal local state")
    if action.delta_time is None or not float(action.delta_time) > 0.0:
        raise ValueError("local prior action requires a positive delta_time")

    options = enumerate_local_prior_actions(state, context, env)
    if isinstance(action, CoalescenceChoice):
        legal = any(
            {
                candidate.active_lineage_i,
                candidate.active_lineage_j,
            }
            == {
                action.active_lineage_i,
                action.active_lineage_j,
            }
            for candidate in options.coal_actions
        )
    elif isinstance(action, RecombinationChoice):
        legal = any(
            candidate.active_lineage_i == action.active_lineage_i
            for candidate in options.recomb_choices
        )
        if legal:
            lineage = state.active_lineages[action.active_lineage_i]
            legal = _is_valid_breakpoint(
                lineage.material_segments,
                env,
                int(action.breakpoint),
            )
    else:
        raise TypeError(f"unsupported local ARG action {type(action)!r}")
    if not legal:
        raise ValueError("the requested action is not locally authorized")

    next_fixed_time = _next_fixed_ancestor_time(state)
    event_time = float(state.current_time) + float(action.delta_time)
    if next_fixed_time is not None and not event_time < next_fixed_time:
        raise ValueError(
            "sampled local event cannot skip a scheduled fixed ancestor"
        )

    previous_max_node = int(state.max_node_idx)
    input_lineages: tuple[ARGLineage, ...]
    if isinstance(action, CoalescenceChoice):
        input_lineages = (
            state.active_lineages[action.active_lineage_i],
            state.active_lineages[action.active_lineage_j],
        )
    else:
        input_lineages = (
            state.active_lineages[action.active_lineage_i],
        )
    undo = _transition_undo_record(
        state,
        modified_node_ids=tuple(
            int(lineage.node_id) for lineage in input_lineages
        ),
    )
    next_state = env.apply_action(state, action, log_prior=float(log_prior))
    created_node_ids = tuple(
        range(previous_max_node + 1, int(next_state.max_node_idx) + 1)
    )
    if isinstance(action, CoalescenceChoice):
        kind = "coalescence"
        input_node_ids = (
            int(state.active_lineages[action.active_lineage_i].node_id),
            int(state.active_lineages[action.active_lineage_j].node_id),
        )
        breakpoint = None
        edge_segments = tuple(
            {
                "parent_node_id": int(created_node_ids[0]),
                "child_node_id": int(lineage.node_id),
                "segments": tuple(lineage.material_segments.segments),
            }
            for lineage in input_lineages
        )
    else:
        kind = "recombination"
        input_node_ids = (
            int(state.active_lineages[action.active_lineage_i].node_id),
        )
        breakpoint = _block_coordinate(state, int(action.breakpoint))
        left_material, right_material = input_lineages[
            0
        ].material_segments.split(int(action.breakpoint))
        edge_segments = (
            {
                "parent_node_id": int(created_node_ids[0]),
                "child_node_id": int(input_lineages[0].node_id),
                "segments": tuple(left_material.segments),
            },
            {
                "parent_node_id": int(created_node_ids[1]),
                "child_node_id": int(input_lineages[0].node_id),
                "segments": tuple(right_material.segments),
            },
        )

    record = {
        "event_type": kind,
        "time": float(next_state.current_time * next_state.time_scale),
        "scaled_time": float(next_state.current_time),
        "input_node_ids": input_node_ids,
        "created_node_ids": created_node_ids,
        "time_action": int(action.time_action),
        "delta_time": float(action.delta_time),
        "log_prior_increment": float(log_prior),
        "breakpoint": breakpoint,
        "action": action_as_dict(action),
        "edge_segments": edge_segments,
        "_undo": {
            **undo,
            "created_node_ids": created_node_ids,
        },
    }
    next_state.transition_records.append(record)
    next_state.is_done = local_is_terminal(next_state, context)
    return next_state


def advance_local_state(
    state: ARGState,
    context: LocalRefinementContext,
    env: SimpleARGEnvironment,
    rng: np.random.Generator,
) -> tuple[ARGState, dict[str, Any]]:
    """Sample and apply the next prior event or fixed-ancestor attachment."""

    action, log_prior = sample_local_prior_action(
        state,
        context,
        env,
        rng,
    )
    if action is not None:
        next_state = apply_local_action(
            state,
            action,
            context,
            env,
            log_prior,
        )
        return next_state, dict(next_state.transition_records[-1])

    next_fixed_time = _next_fixed_ancestor_time(state)
    if next_fixed_time is None:
        raise RuntimeError(
            "fixed attachment selected without a scheduled ancestor"
        )
    next_state = state.clone(copy_partials=False)
    next_state.current_time = float(next_fixed_time)
    next_state.accumulated_log_prior += float(log_prior)
    next_state.partial_log_reward += float(log_prior)
    next_state = reveal_due_fixed_ancestors(
        next_state,
        context,
        next_fixed_time,
    )
    attachment_record = dict(next_state.transition_records[-1])
    attachment_record.update(
        {
            "waited_from_time": float(
                state.current_time * state.time_scale
            ),
            "waited_from_scaled_time": float(state.current_time),
            "log_prior_increment": float(log_prior),
            "fixed_event_survival": True,
        }
    )
    undo = dict(attachment_record["_undo"])
    undo.update(
        {
            "previous_current_time": float(state.current_time),
            "previous_accumulated_log_prior": float(
                state.accumulated_log_prior
            ),
            "previous_partial_log_reward": float(state.partial_log_reward),
            "previous_is_done": bool(state.is_done),
        }
    )
    attachment_record["_undo"] = undo
    next_state.transition_records[-1] = attachment_record
    next_state.is_done = local_is_terminal(next_state, context)
    return next_state, attachment_record


def reveal_due_fixed_ancestors(
    state: ARGState,
    context: LocalRefinementContext,
    event_time: float,
) -> ARGState:
    """Attach due source ancestors to their active target descendants.

    A source ancestor is never introduced as an independent active lineage.
    At its exact source time it consumes the material carried by active
    lineages descended from its cut endpoints, records interval-specific
    parent-child edges, and replaces those material pieces in the frontier.
    """

    _require_local_state(state)
    event_time = float(event_time)
    due = [
        record
        for record in state.fixed_ancestor_schedule
        if int(record["node_id"]) not in state.all_nodes
        and float(record["time"]) <= event_time + 1e-15
    ]
    if not due:
        if event_time > state.current_time:
            next_state = state.clone(copy_partials=False)
            next_state.current_time = event_time
            return next_state
        return state

    next_state = state.clone(copy_partials=False)
    next_state.current_time = max(float(next_state.current_time), event_time)
    due = sorted(due, key=lambda item: (item["time"], item["node_id"]))
    endpoint_ids = {
        int(endpoint_id)
        for record in due
        for dependency in record.get("dependencies", ())
        for endpoint_id in dependency["endpoint_node_ids"]
    }
    modified_node_ids: set[int] = set()
    for lineage in next_state.active_lineages:
        descendant_endpoints = _lineage_endpoint_ids(
            int(lineage.node_id),
            next_state.all_nodes,
            endpoint_ids,
        )
        if descendant_endpoints:
            modified_node_ids.add(int(lineage.node_id))
    undo = _transition_undo_record(
        state,
        modified_node_ids=tuple(sorted(modified_node_ids)),
    )
    attached_ancestors = []
    attachment_rows: list[dict[str, Any]] = []
    edge_segments: list[dict[str, Any]] = []
    for record in sorted(due, key=lambda item: (item["time"], item["node_id"])):
        node_id = int(record["node_id"])
        material = MaterialSegments.from_segments(record["segments"])
        attached_by_child: dict[int, MaterialSegments] = {}
        endpoint_memo: dict[int, frozenset[int]] = {}
        all_dependency_endpoints = {
            int(endpoint_id)
            for dependency in record.get("dependencies", ())
            for endpoint_id in dependency["endpoint_node_ids"]
        }
        for lineage in next_state.active_lineages:
            lineage_endpoints = _lineage_endpoint_ids(
                int(lineage.node_id),
                next_state.all_nodes,
                all_dependency_endpoints,
                memo=endpoint_memo,
            )
            if not lineage_endpoints:
                continue
            attached = MaterialSegments()
            for dependency in record.get("dependencies", ()):
                required = {
                    int(value)
                    for value in dependency["endpoint_node_ids"]
                }
                if not lineage_endpoints.intersection(required):
                    continue
                dependency_material = MaterialSegments.from_segments(
                    dependency["segments"]
                )
                attached = attached.union(
                    lineage.material_segments.intersection(
                        dependency_material
                    )
                )
            if attached.count:
                attached_by_child[int(lineage.node_id)] = attached

        attached_coverage = MaterialSegments()
        for attached in attached_by_child.values():
            attached_coverage = attached_coverage.union(attached)
        if attached_coverage.segments != material.segments:
            raise ValueError(
                "fixed ancestor cannot be attached to all required target "
                f"material at time {event_time * next_state.time_scale}: "
                f"ancestor={node_id} required={material.segments} "
                f"available={attached_coverage.segments}"
            )
        if not attached_by_child:
            raise ValueError(
                "fixed ancestor has no active target descendant at its event "
                f"time: ancestor={node_id}"
            )

        children = []
        child_sequences: set[int] = set()
        retained_active: list[ARGLineage] = []
        for lineage in next_state.active_lineages:
            child_id = int(lineage.node_id)
            attached = attached_by_child.get(child_id)
            if attached is None:
                retained_active.append(lineage)
                continue
            if not float(record["time"]) > float(lineage.time):
                raise ValueError(
                    "fixed ancestor time must be older than every attached "
                    f"child: ancestor={node_id} child={child_id}"
                )
            children.append(child_id)
            child_sequences.update(int(value) for value in lineage.sequences_indices)
            if node_id not in lineage.parents:
                lineage.parents.append(node_id)
            remaining = _subtract_material(
                lineage.material_segments,
                attached,
            )
            _set_lineage_material(lineage, remaining)
            next_state.all_nodes[child_id] = lineage
            if remaining.count:
                retained_active.append(lineage)
            edge_segments.append(
                {
                    "parent_node_id": node_id,
                    "child_node_id": child_id,
                    "segments": tuple(attached.segments),
                }
            )
            attachment_rows.append(
                {
                    "ancestor_node_id": node_id,
                    "child_node_id": child_id,
                    "segments": tuple(attached.segments),
                }
            )

        lineage = ARGLineage(
            node_id=node_id,
            children=children,
            parents=[],
            material_segments=material,
            num_blocks=max(
                len(next_state.block_boundaries or ()) - 1,
                material.span_end + 1
                if material.span_end is not None
                else 0,
            ),
            partials=None,
            sequences_indices=sorted(child_sequences),
            event_type="fixed_source",
            time=float(record["time"]),
        )
        next_state.all_nodes[node_id] = lineage
        retained_active.append(lineage)
        next_state.active_lineages = retained_active
        attached_ancestors.append(node_id)
    next_state.total_active_blocks = sum(
        lineage.material_segments.count
        for lineage in next_state.active_lineages
    )
    next_state.rates = None
    next_state.prior_options = None
    next_state.transition_records.append(
        {
            "event_type": "fixed_attachment",
            "time": float(event_time * next_state.time_scale),
            "scaled_time": event_time,
            "node_ids": tuple(attached_ancestors),
            "attachments": tuple(attachment_rows),
            "edge_segments": tuple(edge_segments),
            "log_prior_increment": 0.0,
            "forward_log_probability": 0.0,
            "backward_log_probability": 0.0,
            "_undo": {
                **undo,
                "created_node_ids": tuple(attached_ancestors),
            },
        }
    )
    return next_state


def undo_local_transition(
    state: ARGState,
    context: LocalRefinementContext | None = None,
) -> ARGState:
    """Apply the exact inverse of the most recent local transition.

    Fixed attachments therefore participate in the same acyclic forward and
    backward trajectory as sampled coalescence and recombination actions.
    The exterior ARG is context, not mutable state, and is never changed here.
    """

    _require_local_state(state)
    if not state.transition_records:
        raise ValueError("local state has no transition to undo")
    record = state.transition_records[-1]
    if record.get("event_type") == "initialization":
        raise ValueError("cannot undo the local initialization state")
    undo = record.get("_undo")
    if undo is None:
        raise ValueError("latest local transition has no inverse record")

    previous = state.clone(copy_partials=False)
    for node_id in undo.get("created_node_ids", ()):
        previous.all_nodes.pop(int(node_id), None)
    for snapshot in undo.get("modified_nodes", ()):
        lineage = _lineage_from_snapshot(snapshot)
        previous.all_nodes[int(lineage.node_id)] = lineage
    active_ids = tuple(int(value) for value in undo["previous_active_node_ids"])
    try:
        previous.active_lineages = [
            previous.all_nodes[node_id] for node_id in active_ids
        ]
    except KeyError as error:
        raise RuntimeError(
            f"inverse transition cannot restore active node {error.args[0]}"
        ) from error
    previous.max_node_idx = int(undo["previous_max_node_idx"])
    previous.current_time = float(undo["previous_current_time"])
    previous.accumulated_log_prior = float(
        undo["previous_accumulated_log_prior"]
    )
    previous.partial_log_reward = float(undo["previous_partial_log_reward"])
    previous.total_active_blocks = int(undo["previous_total_active_blocks"])
    previous.is_done = bool(undo["previous_is_done"])
    previous.transition_records.pop()
    previous.rates = None
    previous.prior_options = None
    if context is not None:
        previous.is_done = local_is_terminal(previous, context)
    return previous


def local_is_terminal(
    state: ARGState,
    context: LocalRefinementContext,
) -> bool:
    """Return whether each target block is carried by exactly one root."""

    _require_local_state(state)
    target = state.target_material
    if target is None or target.count == 0:
        return False
    total_target_count = 0
    covered = MaterialSegments()
    for lineage in state.active_lineages:
        local_material = lineage.material_segments.intersection(target)
        total_target_count += local_material.count
        if total_target_count > target.count:
            return False
        covered = covered.union(local_material)
    return (
        total_target_count == target.count
        and covered.segments == target.segments
    )


def local_state_to_proposal(
    state: ARGState,
    prepared: PreparedLocalRefinement,
) -> LocalARGProposal:
    """Convert a terminal prior-built ``ARGState`` into a splice proposal."""

    if not local_is_terminal(state, prepared.context):
        raise ValueError("only a one-root-per-block state can become a proposal")
    _require_local_state(state)
    generated_start = int(state.generated_node_start)
    generated_nodes = [
        node
        for node_id, node in sorted(state.all_nodes.items())
        if int(node_id) >= generated_start
    ]
    node_records = tuple(
        LocalNodeRecord(
            node_id=int(node.node_id),
            kind=(
                "recombination"
                if node.event_type == "recomb"
                else "coalescence"
            ),
            time=float(node.time * state.time_scale),
            flags=(
                NODE_IS_RE_EVENT
                if node.event_type == "recomb"
                else 0
            ),
        )
        for node in generated_nodes
    )

    edge_records: list[LocalEdgeRecord] = []
    edge_indices_by_transition: dict[int, list[int]] = {}
    for transition_index, record in enumerate(state.transition_records):
        for edge in record.get("edge_segments", ()):
            material = MaterialSegments.from_segments(edge["segments"])
            for left, right in _material_to_intervals(state, material):
                edge_indices_by_transition.setdefault(
                    transition_index,
                    [],
                ).append(len(edge_records))
                edge_records.append(
                    LocalEdgeRecord(
                        left,
                        right,
                        int(edge["parent_node_id"]),
                        int(edge["child_node_id"]),
                    )
                )

    event_records = []
    event_step = 0
    for transition_index, record in enumerate(state.transition_records):
        if record.get("event_type") not in {
            "coalescence",
            "recombination",
        }:
            continue
        event_step += 1
        node_ids = tuple(int(value) for value in record["created_node_ids"])
        event_records.append(
            LocalEventRecord(
                step=event_step,
                kind=record["event_type"],
                time=float(record["time"]),
                action=dict(record["action"]),
                input_lineage_ids=tuple(
                    int(value) for value in record["input_node_ids"]
                ),
                output_lineage_ids=node_ids,
                node_ids=node_ids,
                edge_indices=tuple(
                    edge_indices_by_transition.get(transition_index, ())
                ),
                breakpoint=record.get("breakpoint"),
            )
        )

    root_intervals = _root_intervals(state)
    return LocalARGProposal(
        genomic_range=prepared.context.request.genomic_range,
        cut_time=float(
            prepared.context.resolved_cut.current_time
        ),
        nodes=node_records,
        edges=tuple(edge_records),
        events=tuple(event_records),
        root_intervals=root_intervals,
        authorized_edge_intervals=(
            prepared.context.authorized_edge_intervals
        ),
        prior_log_probability=float(state.accumulated_log_prior),
        transition_records=tuple(
            _public_transition_record(record)
            for record in state.transition_records
        ),
        status="terminal",
    )


def sample_local_trajectories(
    prepared: PreparedLocalRefinement,
    env: SimpleARGEnvironment,
    config: LocalSamplingConfig | None = None,
) -> LocalSampleBatch:
    """Sample complete local ARGs from the filtered CWR prior."""

    config = LocalSamplingConfig() if config is None else config
    rng = np.random.default_rng(int(config.seed))
    proposals: list[LocalARGProposal] = []
    trajectories: list[SimpleTrajectory] = []
    diagnostics: list[ConstructionDiagnostic] = []
    digests: set[str] = set()
    total_transitions = 0
    restarts = 0
    stopped = False

    while len(proposals) < int(config.sample_count) and not stopped:
        if (
            config.max_restarts is not None
            and restarts >= int(config.max_restarts)
        ):
            break
        restarts += 1
        try:
            state = initialize_local_arg_state(prepared, env)
        except ValueError as error:
            diagnostics.append(
                ConstructionDiagnostic(
                    "initialization_failed",
                    str(error),
                )
            )
            break
        trajectory = SimpleTrajectory()
        generated_events = 0
        while not state.is_done:
            if (
                config.max_generated_events is not None
                and generated_events >= int(config.max_generated_events)
            ):
                diagnostics.append(
                    ConstructionDiagnostic(
                        "generated_event_watchdog_reached",
                        "construction had not reached one root per target block "
                        f"after {generated_events} generated events",
                        step=total_transitions,
                    )
                )
                stopped = True
                break
            if (
                config.max_searched_states is not None
                and total_transitions >= int(config.max_searched_states)
            ):
                diagnostics.append(
                    ConstructionDiagnostic(
                        "searched_state_watchdog_reached",
                        "construction had not reached one root per target block "
                        f"after {total_transitions} state transitions",
                        step=total_transitions,
                    )
                )
                stopped = True
                break
            try:
                state, record = advance_local_state(
                    state,
                    prepared.context,
                    env,
                    rng,
                )
            except (ValueError, RuntimeError) as error:
                diagnostics.append(
                    ConstructionDiagnostic(
                        "prior_transition_failed",
                        str(error),
                        step=total_transitions,
                    )
                )
                stopped = True
                break
            total_transitions += 1
            if record["event_type"] in {"coalescence", "recombination"}:
                generated_events += 1
            trajectory.update(
                record,
                log_prior=record.get("log_prior_increment"),
                record=record,
                active_lineages=[
                    lineage.node_id for lineage in state.active_lineages
                ],
            )
        if not state.is_done:
            break
        proposal = local_state_to_proposal(state, prepared)
        digest = proposal.topology_digest
        if (
            not config.allow_duplicate_topologies
            and digest in digests
        ):
            continue
        digests.add(digest)
        proposals.append(proposal)
        trajectories.append(trajectory)

    if len(proposals) < int(config.sample_count):
        diagnostics.append(
            ConstructionDiagnostic(
                "sampling_incomplete",
                "generated "
                f"{len(proposals)} of {int(config.sample_count)} requested "
                "terminal prior proposals",
            )
        )
    return LocalSampleBatch(
        tuple(proposals),
        tuple(trajectories),
        tuple(diagnostics),
        int(config.seed),
        int(total_transitions),
        int(restarts),
    )


def _environment_block_boundaries(
    env: SimpleARGEnvironment,
) -> tuple[float, ...]:
    if env.is_vcf_mode:
        values = np.asarray(env.variant_boundaries, dtype=np.float64)
    else:
        values = np.linspace(
            0.0,
            float(env.sequence_length),
            int(env.num_blocks) + 1,
            dtype=np.float64,
        )
    if values.size != int(env.num_blocks) + 1:
        raise ValueError("environment block boundaries do not match num_blocks")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("environment block boundaries must increase")
    return tuple(float(value) for value in values)


def _boundary_index(
    coordinate: float,
    boundaries: tuple[float, ...],
) -> int:
    coordinate = float(coordinate)
    index = int(bisect.bisect_left(boundaries, coordinate))
    candidates = [
        candidate
        for candidate in (index - 1, index)
        if 0 <= candidate < len(boundaries)
    ]
    if not candidates:
        raise ValueError(f"coordinate {coordinate} is outside block boundaries")
    selected = min(
        candidates,
        key=lambda candidate: abs(float(boundaries[candidate]) - coordinate),
    )
    tolerance = max(1e-9, abs(coordinate) * 1e-12)
    if not math.isclose(
        float(boundaries[selected]),
        coordinate,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError(
            f"coordinate {coordinate} is not aligned to an environment block "
            "boundary; use a finer block configuration"
        )
    return int(selected)


def _intervals_to_material(
    intervals: Iterable[Interval],
    boundaries: tuple[float, ...],
) -> MaterialSegments:
    segments = []
    for left, right in _canonical_segments(tuple(intervals)):
        start = _boundary_index(left, boundaries)
        end = _boundary_index(right, boundaries)
        if start < end:
            segments.append((start, end))
    return MaterialSegments(tuple(segments))


def _material_to_intervals(
    state: ARGState,
    material: MaterialSegments,
) -> tuple[Interval, ...]:
    boundaries = state.block_boundaries
    if boundaries is None:
        raise ValueError("local ARG state is missing block boundaries")
    return tuple(
        (float(boundaries[start]), float(boundaries[end]))
        for start, end in material.segments
    )


def _block_coordinate(state: ARGState, block_index: int) -> float:
    boundaries = state.block_boundaries
    if boundaries is None:
        raise ValueError("local ARG state is missing block boundaries")
    return float(boundaries[int(block_index)])


def _subtract_material(
    material: MaterialSegments,
    removed: MaterialSegments,
) -> MaterialSegments:
    output: list[tuple[int, int]] = []
    removal = tuple(removed.segments)
    for start, end in material.segments:
        cursor = int(start)
        for remove_start, remove_end in removal:
            if remove_end <= cursor:
                continue
            if remove_start >= end:
                break
            if cursor < remove_start:
                output.append((cursor, min(int(remove_start), int(end))))
            cursor = max(cursor, int(remove_end))
            if cursor >= end:
                break
        if cursor < end:
            output.append((cursor, int(end)))
    return MaterialSegments(output)


def _set_lineage_material(
    lineage: ARGLineage,
    material: MaterialSegments,
) -> None:
    lineage.material_segments = MaterialSegments.from_segments(material)
    lineage._material_mask = None
    lineage.clear_runtime_caches()


def _lineage_snapshot(lineage: ARGLineage) -> dict[str, Any]:
    return {
        "node_id": int(lineage.node_id),
        "children": tuple(int(value) for value in lineage.children),
        "parents": tuple(int(value) for value in lineage.parents),
        "material_segments": tuple(lineage.material_segments.segments),
        "num_blocks": int(lineage.num_blocks),
        "sequences_indices": tuple(
            int(value) for value in lineage.sequences_indices
        ),
        "event_type": lineage.event_type,
        "breakpoint": lineage.breakpoint,
        "recombination_side": lineage.recombination_side,
        "time": float(lineage.time),
    }


def _lineage_from_snapshot(snapshot: Mapping[str, Any]) -> ARGLineage:
    return ARGLineage(
        node_id=int(snapshot["node_id"]),
        children=tuple(int(value) for value in snapshot["children"]),
        parents=tuple(int(value) for value in snapshot["parents"]),
        material_segments=MaterialSegments.from_segments(
            snapshot["material_segments"]
        ),
        num_blocks=int(snapshot["num_blocks"]),
        partials=None,
        sequences_indices=tuple(
            int(value) for value in snapshot["sequences_indices"]
        ),
        event_type=snapshot.get("event_type"),
        breakpoint=snapshot.get("breakpoint"),
        recombination_side=snapshot.get("recombination_side"),
        time=float(snapshot["time"]),
    )


def _transition_undo_record(
    state: ARGState,
    *,
    modified_node_ids: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "previous_active_node_ids": tuple(
            int(lineage.node_id) for lineage in state.active_lineages
        ),
        "previous_max_node_idx": int(state.max_node_idx),
        "previous_current_time": float(state.current_time),
        "previous_accumulated_log_prior": float(
            state.accumulated_log_prior
        ),
        "previous_partial_log_reward": float(state.partial_log_reward),
        "previous_total_active_blocks": int(
            state.total_active_blocks or 0
        ),
        "previous_is_done": bool(state.is_done),
        "modified_nodes": tuple(
            _lineage_snapshot(state.all_nodes[int(node_id)])
            for node_id in modified_node_ids
        ),
    }


def _public_transition_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key != "_undo"
    }


def _lineage_endpoint_ids(
    node_id: int,
    all_nodes: Mapping[int, ARGLineage],
    endpoint_ids: set[int],
    *,
    memo: dict[int, frozenset[int]] | None = None,
    path: frozenset[int] = frozenset(),
) -> frozenset[int]:
    node_id = int(node_id)
    if memo is None:
        memo = {}
    cached = memo.get(node_id)
    if cached is not None:
        return cached
    if node_id in path:
        raise ValueError("local ARG contains a cycle")
    lineage = all_nodes.get(node_id)
    if lineage is None:
        raise ValueError(f"local ARG lineage {node_id} is missing")
    output = {node_id} if node_id in endpoint_ids else set()
    for child_id in lineage.children:
        output.update(
            _lineage_endpoint_ids(
                int(child_id),
                all_nodes,
                endpoint_ids,
                memo=memo,
                path=path | {node_id},
            )
        )
    result = frozenset(output)
    memo[node_id] = result
    return result


def _collapse_temporary_source_endpoint(
    prepared: PreparedLocalRefinement,
    node_id: int,
    intervals: tuple[Interval, ...],
) -> dict[int, tuple[Interval, ...]]:
    original_num_nodes = int(
        prepared.synthetic_conversion.metadata["original_num_nodes"]
    )
    augmented_num_nodes = int(
        prepared.synthetic_conversion.metadata["augmented_num_nodes"]
    )
    temporary = set(range(original_num_nodes, augmented_num_nodes))
    outgoing: dict[int, list[tuple[float, float, int]]] = {}
    for edge in prepared.synthetic_arg.edges():
        outgoing.setdefault(int(edge.parent), []).append(
            (float(edge.left), float(edge.right), int(edge.child))
        )
    output: dict[int, tuple[Interval, ...]] = {}

    def descend(
        current: int,
        left: float,
        right: float,
        path: frozenset[int],
    ) -> None:
        if current not in temporary:
            output[current] = _canonical_segments(
                output.get(current, ()) + ((left, right),)
            )
            return
        if current in path:
            raise ValueError("temporary source routing graph contains a cycle")
        covered = []
        for edge_left, edge_right, child in outgoing.get(current, ()):
            overlap_left = max(left, edge_left)
            overlap_right = min(right, edge_right)
            if overlap_left < overlap_right:
                covered.append((overlap_left, overlap_right))
                descend(
                    child,
                    overlap_left,
                    overlap_right,
                    path | {current},
                )
        if _canonical_segments(covered) != ((left, right),):
            raise ValueError(
                "temporary source routing endpoint does not cover target material"
            )

    for left, right in intervals:
        descend(int(node_id), float(left), float(right), frozenset())
    return output


def _compile_fixed_ancestor_schedule(
    prepared: PreparedLocalRefinement,
    boundaries: tuple[float, ...],
    time_scale: float,
    endpoint_material: Mapping[int, MaterialSegments],
) -> list[dict[str, Any]]:
    original_num_nodes = int(
        prepared.synthetic_conversion.metadata["original_num_nodes"]
    )
    scheduled: dict[int, dict[str, Any]] = {}
    for lineage in prepared.context.promoted_dependency_lineages:
        node_id = int(lineage.node_id)
        if (
            node_id >= original_num_nodes
            or node_id in endpoint_material
            or not lineage.fixed_segments
            or not lineage.mutable_segments
        ):
            continue
        material = _intervals_to_material(
            lineage.mutable_segments,
            boundaries,
        )
        if material.count == 0:
            continue
        source_time = float(
            prepared.source_tree_sequence.node(node_id).time
        ) / time_scale
        existing = scheduled.get(node_id)
        if existing is None:
            scheduled[node_id] = {
                "node_id": node_id,
                "time": source_time,
                "segments": material.segments,
            }
        else:
            combined = MaterialSegments.from_segments(
                existing["segments"]
            ).union(material)
            existing["segments"] = combined.segments
            existing["time"] = min(float(existing["time"]), source_time)
    for record in scheduled.values():
        record["dependencies"] = _fixed_ancestor_dependencies(
            prepared,
            int(record["node_id"]),
            MaterialSegments.from_segments(record["segments"]),
            boundaries,
            endpoint_material,
        )
    return sorted(
        scheduled.values(),
        key=lambda item: (float(item["time"]), int(item["node_id"])),
    )


def _fixed_ancestor_dependencies(
    prepared: PreparedLocalRefinement,
    ancestor_node_id: int,
    material: MaterialSegments,
    boundaries: tuple[float, ...],
    endpoint_material: Mapping[int, MaterialSegments],
) -> tuple[dict[str, Any], ...]:
    """Map each scheduled block to its cut endpoint descendants."""

    block_dependencies: list[tuple[int, tuple[int, ...]]] = []
    tree_iterator = iter(prepared.synthetic_arg.trees())
    try:
        tree = next(tree_iterator)
    except StopIteration as error:  # pragma: no cover - invalid tskit input
        raise ValueError("synthetic ARG contains no marginal trees") from error

    for block in material.to_block_list():
        coordinate = (
            float(boundaries[int(block)])
            + float(boundaries[int(block) + 1])
        ) / 2.0
        while coordinate >= float(tree.interval.right):
            try:
                tree = next(tree_iterator)
            except StopIteration as error:
                raise ValueError(
                    "fixed-ancestor material lies outside the synthetic ARG"
                ) from error
        endpoint_ids = tuple(
            sorted(
                int(endpoint_id)
                for endpoint_id, endpoint_blocks in endpoint_material.items()
                if endpoint_blocks.covers_interval(block, block + 1)
                and (
                    int(endpoint_id) == int(ancestor_node_id)
                    or tree.is_descendant(
                        int(endpoint_id),
                        int(ancestor_node_id),
                    )
                )
            )
        )
        if not endpoint_ids:
            raise ValueError(
                "fixed ancestor has target material without a dependent cut "
                f"endpoint: ancestor={ancestor_node_id} block={block}"
            )
        block_dependencies.append((int(block), endpoint_ids))

    grouped: list[dict[str, Any]] = []
    for block, endpoint_ids in block_dependencies:
        if (
            grouped
            and grouped[-1]["endpoint_node_ids"] == endpoint_ids
            and grouped[-1]["segments"][-1][1] == block
        ):
            start, _end = grouped[-1]["segments"][-1]
            grouped[-1]["segments"] = ((start, block + 1),)
        else:
            grouped.append(
                {
                    "segments": ((block, block + 1),),
                    "endpoint_node_ids": endpoint_ids,
                }
            )
    return tuple(grouped)


def _next_fixed_ancestor_time(state: ARGState) -> float | None:
    times = [
        float(record["time"])
        for record in state.fixed_ancestor_schedule
        if int(record["node_id"]) not in state.all_nodes
        and float(record["time"]) > float(state.current_time) + 1e-15
    ]
    return min(times) if times else None


def _vcf_breakpoints(
    material: MaterialSegments,
    env: SimpleARGEnvironment,
) -> tuple[tuple[int, float], ...]:
    if material.span_start is None or material.span_end is None:
        return ()
    output = []
    for breakpoint in range(
        int(material.span_start) + 1,
        int(material.span_end) + 1,
    ):
        left, right = material.split(breakpoint)
        if left.count == 0 or right.count == 0:
            continue
        weight = (
            float(env._breakpoint_gap_length(breakpoint))
            if env.is_vcf_mode
            else 1.0
        )
        if weight > 0.0:
            output.append((int(breakpoint), weight))
    return tuple(output)


def _has_valid_breakpoint(
    material: MaterialSegments,
    env: SimpleARGEnvironment,
) -> bool:
    if (
        material.count < 2
        or material.span_start is None
        or material.span_end is None
        or material.span_start >= material.span_end
    ):
        return False
    if not env.is_vcf_mode:
        return True
    return bool(_vcf_breakpoints(material, env))


def _is_valid_breakpoint(
    material: MaterialSegments,
    env: SimpleARGEnvironment,
    breakpoint: int,
) -> bool:
    breakpoint = int(breakpoint)
    if (
        material.span_start is None
        or material.span_end is None
        or not int(material.span_start) < breakpoint <= int(material.span_end)
    ):
        return False
    left, right = material.split(breakpoint)
    if left.count == 0 or right.count == 0:
        return False
    return (
        not env.is_vcf_mode
        or float(env._breakpoint_gap_length(breakpoint)) > 0.0
    )


def _local_recombination_weight(
    lineage: ARGLineage,
    env: SimpleARGEnvironment,
) -> float:
    if env.is_vcf_mode:
        return float(
            sum(weight for _breakpoint, weight in _vcf_breakpoints(
                lineage.material_segments,
                env,
            ))
        )
    return float(lineage.material_segments.count)


def _sample_breakpoint(
    material: MaterialSegments,
    env: SimpleARGEnvironment,
    rng: np.random.Generator,
) -> tuple[int, float]:
    if not _has_valid_breakpoint(material, env):
        raise ValueError("recombination lineage has no valid breakpoint")
    if not env.is_vcf_mode:
        count = int(material.span_end - material.span_start)
        breakpoint = int(
            rng.integers(
                int(material.span_start) + 1,
                int(material.span_end) + 1,
            )
        )
        return breakpoint, 1.0 / float(count)
    values = _vcf_breakpoints(material, env)
    weights = np.asarray([weight for _value, weight in values], dtype=np.float64)
    index = int(rng.choice(len(values), p=weights / weights.sum()))
    return int(values[index][0]), float(weights[index] / weights.sum())


def _root_intervals(
    state: ARGState,
) -> tuple[tuple[float, float, int], ...]:
    target = state.target_material
    if target is None:
        return ()
    if not local_is_terminal(state, None):
        raise ValueError("target material does not have exactly one root per block")
    roots = sorted(
        (
            int(start),
            int(end),
            int(lineage.node_id),
        )
        for lineage in state.active_lineages
        for start, end in lineage.material_segments.intersection(target).segments
    )
    output: list[tuple[float, float, int]] = []
    for start, end, node_id in roots:
        left = _block_coordinate(state, start)
        right = _block_coordinate(state, end)
        if (
            output
            and output[-1][2] == node_id
            and output[-1][1] == left
        ):
            output[-1] = (
                output[-1][0],
                right,
                node_id,
            )
        else:
            output.append((left, right, node_id))
    return tuple(output)


def _require_local_state(state: ARGState) -> None:
    if (
        state.target_material is None
        or state.block_boundaries is None
        or state.generated_node_start is None
        or not state.time_scale > 0.0
    ):
        raise ValueError("ARGState was not initialized for local construction")


__all__ = [
    "ConstructionDiagnostic",
    "LocalARGProposal",
    "LocalEdgeRecord",
    "LocalEventRecord",
    "LocalNodeRecord",
    "LocalSampleBatch",
    "LocalSamplingConfig",
    "advance_local_state",
    "apply_local_action",
    "enumerate_local_prior_actions",
    "initialize_local_arg_state",
    "local_is_terminal",
    "local_state_to_proposal",
    "reveal_due_fixed_ancestors",
    "sample_local_prior_action",
    "sample_local_trajectories",
    "undo_local_transition",
]
