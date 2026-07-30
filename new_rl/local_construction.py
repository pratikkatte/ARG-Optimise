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
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from typing import Any, Iterable, Literal, Mapping, Union

import numpy as np
import torch

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
from .vcf_likelihood import (
    compute_cut_frontier_vcf_partials,
    compute_tree_sequence_vcf_log_likelihood,
    resolve_vcf_tree_sequence_alignment,
)


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
    log_likelihood: float | None = None
    outside_log_likelihood: float | None = None
    local_log_likelihood: float | None = None
    log_reward: float | None = None
    likelihood_scope: Literal["whole_vcf_chromosome", "none"] = "none"
    likelihood_alignment: dict[str, Any] = field(default_factory=dict)
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
    *,
    sample_node_to_haplotype: Mapping[int, int | str] | None = None,
    vcf_coordinate_offset: str | float = "auto",
) -> ARGState:
    """Create an ``ARGState`` from target-bearing lineages at the trace cut."""

    if not prepared.context.is_valid:
        reasons = "; ".join(
            item.message for item in prepared.context.rejection_diagnostics
        )
        raise ValueError(f"local refinement context is invalid: {reasons}")
    if not env.structural_only and not env.is_vcf_mode:
        raise ValueError(
            "likelihood-enabled local construction currently requires a VCF "
            "environment; otherwise pass structural_only=True"
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

    endpoint_intervals: dict[int, tuple[Interval, ...]] = {}
    for lineage in prepared.context.cut_active_lineages:
        collapsed = _collapse_temporary_source_endpoint(
            prepared,
            int(lineage.node_id),
            lineage.mutable_segments,
        )
        for endpoint_node_id, intervals in collapsed.items():
            endpoint_intervals[int(endpoint_node_id)] = _canonical_segments(
                endpoint_intervals.get(int(endpoint_node_id), ())
                + tuple(intervals)
            )
    if not endpoint_intervals:
        raise ValueError("no target-bearing cut lineage remains after routing collapse")

    alignment: dict[str, Any] = {}
    if env.is_vcf_mode:
        alignment = resolve_vcf_tree_sequence_alignment(
            prepared.source_tree_sequence,
            env.variant_data,
            sample_node_to_haplotype=sample_node_to_haplotype,
            vcf_coordinate_offset=vcf_coordinate_offset,
        )
        boundaries = _local_structural_boundaries(
            prepared,
            endpoint_intervals,
            alignment,
        )
    else:
        boundaries = _environment_block_boundaries(env)
    target_material = _intervals_to_material(
        (prepared.context.request.genomic_range,),
        boundaries,
    )
    if target_material.count == 0:
        raise ValueError("the requested genomic range contains no ARG blocks")

    endpoint_material: dict[int, MaterialSegments] = {}
    for endpoint_node_id, intervals in endpoint_intervals.items():
        material = _intervals_to_material(intervals, boundaries)
        if material.count:
            endpoint_material[int(endpoint_node_id)] = material
    if not endpoint_material:
        raise ValueError("no target-bearing cut lineage remains after routing collapse")

    time_scale = 2.0 * float(env.population_size)
    if not time_scale > 0.0:
        raise ValueError("population_size must define a positive 2Ne time scale")

    likelihood_data: dict[str, Any] | None = None
    variant_block_indices: dict[int, int] = {}
    local_breakpoint_weights: dict[int, float] = {}
    if env.is_vcf_mode and not env.structural_only:
        likelihood_data = compute_cut_frontier_vcf_partials(
            prepared.source_tree_sequence,
            env.variant_data,
            endpoint_intervals,
            prepared.context.request.genomic_range,
            mutation_rate=float(env.mutation_rate),
            alignment=alignment,
        )
        for variant_index in likelihood_data["target_variant_indices"]:
            coordinate = float(
                alignment["variant_coordinates"][int(variant_index)]
            )
            variant_block_indices[int(variant_index)] = _coordinate_block_index(
                coordinate,
                boundaries,
            )
        local_breakpoint_weights = _local_vcf_breakpoint_weights(
            boundaries,
            alignment,
            prepared.context.request.genomic_range,
        )

    active_lineages = []
    all_nodes: dict[int, ARGLineage] = {}
    for node_id, material in sorted(endpoint_material.items()):
        source_node = prepared.source_tree_sequence.node(node_id)
        endpoint_likelihood = (
            likelihood_data["endpoints"][int(node_id)]
            if likelihood_data is not None
            else None
        )
        lineage = ARGLineage(
            node_id=node_id,
            children=[],
            parents=[],
            material_segments=material,
            num_blocks=len(boundaries) - 1,
            partials=(
                None
                if endpoint_likelihood is None
                else torch.as_tensor(
                    endpoint_likelihood["partials"],
                    dtype=torch.float64,
                    device=env.device,
                )
            ),
            variant_indices=(
                ()
                if endpoint_likelihood is None
                else endpoint_likelihood["variant_indices"]
            ),
            sequences_indices=(
                []
                if endpoint_likelihood is None
                else endpoint_likelihood["sequences_indices"]
            ),
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
    outside_log_likelihood = (
        0.0
        if likelihood_data is None
        else float(likelihood_data["outside_log_likelihood"])
    )
    accumulated_log_likelihood = (
        0.0
        if likelihood_data is None
        else outside_log_likelihood
        + float(likelihood_data["inside_log_scale"])
    )
    state = ARGState(
        active_lineages=active_lineages,
        all_nodes=all_nodes,
        max_node_idx=generated_node_start - 1,
        log_likelihood=None,
        accumulated_log_prior=0.0,
        accumulated_log_likelihood=accumulated_log_likelihood,
        outside_log_likelihood=outside_log_likelihood,
        partial_log_reward=accumulated_log_likelihood,
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
                "mutation_rate": float(env.mutation_rate),
                "recombination_rate": float(env.recombination_rate),
                "rho": float(env.rho),
                "reward_C": float(env.reward_fn.C),
                "num_blocks": int(env.num_blocks),
                "local_structural_block_count": int(len(boundaries) - 1),
                "block_mode": "vcf_exact_local" if env.is_vcf_mode else "uniform",
                "likelihood_scope": (
                    "whole_vcf_chromosome"
                    if likelihood_data is not None
                    else "none"
                ),
                "outside_log_likelihood": outside_log_likelihood,
                "inside_initial_log_scale": (
                    0.0
                    if likelihood_data is None
                    else float(likelihood_data["inside_log_scale"])
                ),
                "vcf_coordinate_offset": alignment.get(
                    "vcf_coordinate_offset"
                ),
                "vcf_path": alignment.get("vcf_path"),
                "vcf_parser_version": alignment.get("parser_version"),
                "sample_node_to_haplotype": alignment.get(
                    "haplotype_index_by_sample_node"
                ),
            }
        ],
        fixed_ancestor_schedule=schedule,
        target_variant_indices=(
            ()
            if likelihood_data is None
            else tuple(likelihood_data["target_variant_indices"])
        ),
        variant_block_indices=variant_block_indices,
        local_breakpoint_weights=local_breakpoint_weights,
        vcf_alignment=dict(alignment),
        likelihood_scope=(
            "whole_vcf_chromosome"
            if likelihood_data is not None
            else "none"
        ),
    )
    state = reveal_due_fixed_ancestors(
        state,
        prepared.context,
        current_time,
        env=env,
    )
    state.is_done = local_is_terminal(state, prepared.context)
    if state.is_done and likelihood_data is not None:
        _finalize_local_terminal_likelihood(
            state,
            prepared.context,
            env,
        )
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
        if not _has_valid_breakpoint(lineage.material_segments, state, env):
            continue
        legal_recomb.append(action)

    if env.is_vcf_mode and state.target_material is not None:
        total_recomb_weight = sum(
            _local_recombination_weight(
                state.active_lineages[choice.active_lineage_i],
                state,
                env,
            )
            for choice in legal_recomb
        )
        total_active_material_length = (
            float(total_recomb_weight)
            / max(float(env.sequence_length), 1.0)
        )
        rates = {
            "lambda_coal": float(len(legal_coal)),
            "lambda_recomb": (
                float(env.rho) / 2.0 * total_active_material_length
            ),
            "total_active_material_length": total_active_material_length,
        }
    else:
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
    if max_delta is not None and max_delta <= 1e-15:
        return None, -total_rate * max_delta
    if max_delta is None:
        generated_mass = 1.0
        survival_mass = 0.0
    else:
        generated_mass, survival_mass = (
            env.time_env.bounded_waiting_distribution(
                total_rate,
                max_delta,
            )
        )

    generate_event = bool(
        rng.choice(
            2,
            p=np.asarray(
                [generated_mass, survival_mass],
                dtype=np.float64,
            ),
        )
        == 0
    )
    if not generate_event:
        return None, math.log(survival_mass) if survival_mass > 0.0 else -math.inf

    time_quantile = env.time_env.sample_prior_quantile(rng)
    delta_time = env.time_env.quantile_to_delta(
        time_quantile,
        total_rate,
        max_delta=max_delta,
    )
    wait_log_probability = env.time_env.waiting_time_log_density(
        delta_time,
        total_rate,
        max_delta=max_delta,
    )

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
            time_quantile=time_quantile,
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
                    state,
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
            state,
            env,
            rng,
        )
        action = replace(
            choice,
            breakpoint=int(breakpoint),
            time_quantile=time_quantile,
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
    if (
        action.time_quantile is None
        or not 0.0 < float(action.time_quantile) < 1.0
    ):
        raise ValueError(
            "local prior action requires time_quantile inside (0, 1)"
        )

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
                state,
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
        "time_quantile": float(action.time_quantile),
        "delta_time": float(action.delta_time),
        "waiting_rate": float(
            options.rates["lambda_coal"]
            + options.rates["lambda_recomb"]
        ),
        "fixed_horizon": (
            None
            if next_fixed_time is None
            else float(next_fixed_time - state.current_time)
        ),
        "time_log_prior_density": float(
            env.time_env.waiting_time_log_density(
                action.delta_time,
                options.rates["lambda_coal"]
                + options.rates["lambda_recomb"],
                max_delta=(
                    None
                    if next_fixed_time is None
                    else float(next_fixed_time - state.current_time)
                ),
            )
        ),
        "log_prior_increment": float(log_prior),
        "log_likelihood_increment": float(
            next_state.accumulated_log_likelihood
            - state.accumulated_log_likelihood
        ),
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
    if next_state.is_done and next_state.likelihood_scope != "none":
        _finalize_local_terminal_likelihood(next_state, context, env)
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
        env=env,
    )
    attachment_record = dict(next_state.transition_records[-1])
    rates = state.prior_options.rates
    total_rate = float(
        rates["lambda_coal"] + rates["lambda_recomb"]
    )
    fixed_horizon = float(next_fixed_time - state.current_time)
    attachment_record.update(
        {
            "waited_from_time": float(
                state.current_time * state.time_scale
            ),
            "waited_from_scaled_time": float(state.current_time),
            "log_prior_increment": float(log_prior),
            "fixed_event_survival": True,
            "waiting_rate": total_rate,
            "fixed_horizon": fixed_horizon,
            "survival_log_probability": float(log_prior),
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
    if next_state.is_done and next_state.likelihood_scope != "none":
        _finalize_local_terminal_likelihood(next_state, context, env)
    return next_state, attachment_record


def reveal_due_fixed_ancestors(
    state: ARGState,
    context: LocalRefinementContext,
    event_time: float,
    *,
    env: SimpleARGEnvironment | None = None,
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
    likelihood_increment = 0.0
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
        parent_partials = None
        parent_variant_indices: tuple[int, ...] = ()
        if state.likelihood_scope != "none":
            if env is None or env.structural_only or not env.is_vcf_mode:
                raise ValueError(
                    "likelihood-enabled fixed attachment requires its VCF "
                    "environment"
                )
            (
                parent_partials,
                parent_variant_indices,
                parent_increment,
            ) = _fixed_ancestor_partials(
                next_state,
                attached_by_child,
                float(record["time"]),
                env,
            )
            likelihood_increment += float(parent_increment)
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
            if state.likelihood_scope != "none":
                retained_variants = _variant_indices_for_material(
                    lineage.variant_indices,
                    remaining,
                    next_state.variant_block_indices,
                )
                lineage.partials = _select_variant_partial_rows(
                    lineage.partials,
                    lineage.variant_indices,
                    retained_variants,
                )
                lineage.variant_indices = retained_variants
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
            partials=parent_partials,
            variant_indices=parent_variant_indices,
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
    next_state.accumulated_log_likelihood += float(likelihood_increment)
    next_state.partial_log_reward += float(likelihood_increment)
    next_state.transition_records.append(
        {
            "event_type": "fixed_attachment",
            "time": float(event_time * next_state.time_scale),
            "scaled_time": event_time,
            "node_ids": tuple(attached_ancestors),
            "attachments": tuple(attachment_rows),
            "edge_segments": tuple(edge_segments),
            "log_prior_increment": 0.0,
            "log_likelihood_increment": float(likelihood_increment),
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
    previous.accumulated_log_likelihood = float(
        undo.get("previous_accumulated_log_likelihood", 0.0)
    )
    previous.log_likelihood = undo.get("previous_log_likelihood")
    previous.log_reward = undo.get("previous_log_reward")
    previous.terminal_partial_correction = float(
        undo.get("previous_terminal_partial_correction", 0.0)
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


def compute_local_terminal_log_likelihood(
    state: ARGState,
    context: LocalRefinementContext,
    env: SimpleARGEnvironment,
) -> float:
    """Complete the whole-chromosome VCF likelihood at local roots."""

    if not local_is_terminal(state, context):
        raise ValueError(
            "local terminal likelihood requires one root per target block"
        )
    if state.likelihood_scope == "none":
        raise ValueError("local state was initialized without VCF likelihoods")
    if env.structural_only or not env.is_vcf_mode:
        raise ValueError(
            "local terminal likelihood requires a likelihood-enabled VCF "
            "environment"
        )

    carriers: dict[int, list[tuple[ARGLineage, int]]] = {
        int(variant_index): []
        for variant_index in state.target_variant_indices
    }
    for lineage in state.active_lineages:
        row_by_variant = {
            int(variant_index): row_index
            for row_index, variant_index in enumerate(
                lineage.variant_indices
            )
        }
        for variant_index, row_index in row_by_variant.items():
            if variant_index in carriers:
                carriers[variant_index].append((lineage, row_index))

    root_log_probability = 0.0
    for variant_index in state.target_variant_indices:
        rows = carriers[int(variant_index)]
        if len(rows) != 1:
            raise ValueError(
                "terminal target VCF row must be carried by exactly one root: "
                f"variant={variant_index} carrier_count={len(rows)}"
            )
        lineage, row_index = rows[0]
        block_index = int(state.variant_block_indices[int(variant_index)])
        if not lineage.material_segments.covers_interval(
            block_index,
            block_index + 1,
        ):
            raise ValueError(
                "terminal VCF row is carried outside its structural material: "
                f"variant={variant_index} lineage={lineage.node_id}"
            )
        partials = env._require_lineage_partials(lineage)
        probability = torch.sum(partials[int(row_index)] * 0.25)
        value = float(probability.detach().cpu().item())
        if not value > 0.0 or not math.isfinite(value):
            raise ValueError(
                f"terminal root probability is invalid for VCF row "
                f"{variant_index}"
            )
        root_log_probability += math.log(value)

    log_likelihood = (
        float(state.accumulated_log_likelihood)
        + float(root_log_probability)
    )
    if not math.isfinite(log_likelihood):
        raise ValueError("local terminal VCF log likelihood is non-finite")
    return float(log_likelihood)


def compute_local_terminal_log_reward(
    state: ARGState,
    context: LocalRefinementContext,
    env: SimpleARGEnvironment,
) -> float:
    """Return ``C + whole VCF likelihood + local CWR prior``."""

    log_likelihood = compute_local_terminal_log_likelihood(
        state,
        context,
        env,
    )
    return float(
        env.reward_fn(
            log_likelihood,
            float(state.accumulated_log_prior),
        )
    )


def _finalize_local_terminal_likelihood(
    state: ARGState,
    context: LocalRefinementContext,
    env: SimpleARGEnvironment,
) -> None:
    log_likelihood = compute_local_terminal_log_likelihood(
        state,
        context,
        env,
    )
    log_reward = float(
        env.reward_fn(
            log_likelihood,
            float(state.accumulated_log_prior),
        )
    )
    state.log_likelihood = float(log_likelihood)
    state.log_reward = log_reward
    state.terminal_partial_correction = float(
        log_reward - state.partial_log_reward
    )
    state.partial_log_reward = log_reward


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
        log_likelihood=(
            None
            if state.log_likelihood is None
            else float(state.log_likelihood)
        ),
        outside_log_likelihood=(
            None
            if state.likelihood_scope == "none"
            else float(state.outside_log_likelihood)
        ),
        local_log_likelihood=(
            None
            if state.log_likelihood is None
            else float(state.log_likelihood)
            - float(state.outside_log_likelihood)
        ),
        log_reward=(
            None if state.log_reward is None else float(state.log_reward)
        ),
        likelihood_scope=(
            "whole_vcf_chromosome"
            if state.likelihood_scope == "whole_vcf_chromosome"
            else "none"
        ),
        likelihood_alignment=_public_likelihood_alignment(
            state.vcf_alignment
        ),
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
    *,
    initial_state: ARGState | None = None,
    sample_node_to_haplotype: Mapping[int, int | str] | None = None,
    vcf_coordinate_offset: str | float = "auto",
) -> LocalSampleBatch:
    """Sample complete local ARGs from one cached cut-state template."""

    config = LocalSamplingConfig() if config is None else config
    rng = np.random.default_rng(int(config.seed))
    proposals: list[LocalARGProposal] = []
    trajectories: list[SimpleTrajectory] = []
    diagnostics: list[ConstructionDiagnostic] = []
    digests: set[str] = set()
    total_transitions = 0
    restarts = 0
    stopped = False
    try:
        if initial_state is None:
            initial_template = initialize_local_arg_state(
                prepared,
                env,
                sample_node_to_haplotype=sample_node_to_haplotype,
                vcf_coordinate_offset=vcf_coordinate_offset,
            )
        else:
            _require_local_state(initial_state)
            initial_template = initial_state
    except ValueError as error:
        return LocalSampleBatch(
            (),
            (),
            (
                ConstructionDiagnostic(
                    "initialization_failed",
                    str(error),
                ),
            ),
            int(config.seed),
            0,
            0,
        )

    while len(proposals) < int(config.sample_count) and not stopped:
        if (
            config.max_restarts is not None
            and restarts >= int(config.max_restarts)
        ):
            break
        restarts += 1
        state = initial_template.clone(copy_partials=False)
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
                log_reward=state.log_reward,
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


def _local_structural_boundaries(
    prepared: PreparedLocalRefinement,
    endpoint_intervals: Mapping[int, tuple[Interval, ...]],
    alignment: Mapping[str, Any],
) -> tuple[float, ...]:
    """Build an exact compact physical grid for one local problem."""

    left, right = (
        float(prepared.context.request.genomic_range[0]),
        float(prepared.context.request.genomic_range[1]),
    )
    values = {left, right}

    def add_interval(interval_left: float, interval_right: float) -> None:
        interval_left = max(left, float(interval_left))
        interval_right = min(right, float(interval_right))
        if interval_left < interval_right:
            values.add(interval_left)
            values.add(interval_right)

    for intervals in endpoint_intervals.values():
        for interval_left, interval_right in intervals:
            add_interval(interval_left, interval_right)
    for lineage in prepared.context.active_lineages:
        for interval_left, interval_right in (
            lineage.mutable_segments + lineage.fixed_segments
        ):
            add_interval(interval_left, interval_right)
    for item in prepared.context.authorized_edge_intervals:
        add_interval(item.left, item.right)
    for attachment in prepared.context.boundary_attachments:
        for interval_left, interval_right in attachment.intervals:
            add_interval(interval_left, interval_right)
    for breakpoint in prepared.source_tree_sequence.breakpoints():
        breakpoint = float(breakpoint)
        if left < breakpoint < right:
            values.add(breakpoint)

    coordinates = np.asarray(
        alignment["variant_coordinates"],
        dtype=np.float64,
    )
    if coordinates.size > 1:
        for midpoint in (coordinates[:-1] + coordinates[1:]) / 2.0:
            midpoint = float(midpoint)
            if left < midpoint < right:
                values.add(midpoint)

    boundaries = tuple(sorted(values))
    if len(boundaries) < 2 or boundaries[0] != left or boundaries[-1] != right:
        raise RuntimeError("local structural grid does not cover the request")
    if any(
        not float(current) < float(next_value)
        for current, next_value in zip(boundaries, boundaries[1:])
    ):
        raise RuntimeError("local structural grid is not strictly increasing")
    return boundaries


def _coordinate_block_index(
    coordinate: float,
    boundaries: tuple[float, ...],
) -> int:
    coordinate = float(coordinate)
    block = bisect.bisect_right(boundaries, coordinate) - 1
    if not 0 <= block < len(boundaries) - 1:
        raise ValueError(
            f"coordinate {coordinate} is outside the local structural grid"
        )
    if not (
        float(boundaries[block])
        <= coordinate
        < float(boundaries[block + 1])
    ):
        raise ValueError(
            f"coordinate {coordinate} cannot be assigned to a local block"
        )
    return int(block)


def _local_vcf_breakpoint_weights(
    boundaries: tuple[float, ...],
    alignment: Mapping[str, Any],
    genomic_range: Interval,
) -> dict[int, float]:
    """Map eligible VCF gap midpoints to exact local boundary indices."""

    left, right = float(genomic_range[0]), float(genomic_range[1])
    coordinates = np.asarray(
        alignment["variant_coordinates"],
        dtype=np.float64,
    )
    output: dict[int, float] = {}
    if coordinates.size < 2:
        return output
    for left_position, right_position in zip(
        coordinates[:-1],
        coordinates[1:],
    ):
        midpoint = float((left_position + right_position) / 2.0)
        if not left < midpoint < right:
            continue
        boundary_index = _boundary_index(midpoint, boundaries)
        output[int(boundary_index)] = float(
            max(float(right_position - left_position), 1.0)
        )
    return output


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


def _variant_indices_for_material(
    variant_indices: Iterable[int],
    material: MaterialSegments,
    variant_block_indices: Mapping[int, int],
) -> tuple[int, ...]:
    return tuple(
        int(variant_index)
        for variant_index in variant_indices
        if any(
            int(start)
            <= int(variant_block_indices[int(variant_index)])
            < int(end)
            for start, end in material.segments
        )
    )


def _select_variant_partial_rows(
    partials: Any,
    source_variant_indices: Iterable[int],
    target_variant_indices: Iterable[int],
) -> torch.Tensor:
    if partials is None:
        raise ValueError("likelihood-enabled lineage is missing partials")
    tensor = (
        partials
        if torch.is_tensor(partials)
        else torch.as_tensor(partials, dtype=torch.float32)
    )
    source = tuple(int(value) for value in source_variant_indices)
    target = tuple(int(value) for value in target_variant_indices)
    if not target:
        return tensor.new_zeros((0, 4))
    position_by_variant = {
        variant_index: row_index
        for row_index, variant_index in enumerate(source)
    }
    try:
        rows = [position_by_variant[value] for value in target]
    except KeyError as error:
        raise ValueError(
            f"target VCF row {error.args[0]} is absent from the lineage"
        ) from error
    return tensor.index_select(
        0,
        torch.as_tensor(rows, dtype=torch.long, device=tensor.device),
    )


def _fixed_ancestor_partials(
    state: ARGState,
    attached_by_child: Mapping[int, MaterialSegments],
    parent_time: float,
    env: SimpleARGEnvironment,
) -> tuple[torch.Tensor, tuple[int, ...], float]:
    attached_variants_by_child = {}
    parent_variants: set[int] = set()
    for child_id, attached_material in attached_by_child.items():
        child = state.all_nodes[int(child_id)]
        variants = _variant_indices_for_material(
            child.variant_indices,
            attached_material,
            state.variant_block_indices,
        )
        attached_variants_by_child[int(child_id)] = variants
        parent_variants.update(variants)
    parent_variant_indices = tuple(sorted(parent_variants))
    if not parent_variant_indices:
        reference = next(
            (
                state.all_nodes[int(child_id)].partials
                for child_id in attached_by_child
                if state.all_nodes[int(child_id)].partials is not None
            ),
            None,
        )
        dtype = (
            reference.dtype
            if torch.is_tensor(reference)
            else torch.float64
        )
        return (
            torch.empty((0, 4), dtype=dtype, device=env.device),
            (),
            0.0,
        )

    position_by_variant = {
        variant_index: row_index
        for row_index, variant_index in enumerate(parent_variant_indices)
    }
    reference = next(
        state.all_nodes[int(child_id)].partials
        for child_id, variants in attached_variants_by_child.items()
        if variants
    )
    dtype = (
        reference.dtype
        if torch.is_tensor(reference)
        else torch.float64
    )
    combined = torch.ones(
        (len(parent_variant_indices), 4),
        dtype=dtype,
        device=env.device,
    )
    carried = torch.zeros(
        len(parent_variant_indices),
        dtype=torch.bool,
        device=env.device,
    )
    for child_id, variants in attached_variants_by_child.items():
        if not variants:
            continue
        child = state.all_nodes[int(child_id)]
        transitioned = env._transition_lineage_partials(
            child,
            float(parent_time),
        )
        selected = _select_variant_partial_rows(
            transitioned,
            child.variant_indices,
            variants,
        )
        parent_rows = torch.as_tensor(
            [position_by_variant[value] for value in variants],
            dtype=torch.long,
            device=combined.device,
        )
        combined[parent_rows] = combined[parent_rows] * selected
        carried[parent_rows] = True
    if not bool(carried.all().detach().cpu().item()):
        raise ValueError("fixed ancestor has an uncovered target VCF row")
    normalized, log_scale = env.normalize_partials_with_log_scale(
        combined,
        carried,
    )
    return normalized, parent_variant_indices, float(log_scale)


def _lineage_snapshot(lineage: ARGLineage) -> dict[str, Any]:
    if torch.is_tensor(lineage.partials):
        partials = lineage.partials.detach().clone()
    else:
        partials = lineage.partials
    return {
        "node_id": int(lineage.node_id),
        "children": tuple(int(value) for value in lineage.children),
        "parents": tuple(int(value) for value in lineage.parents),
        "material_segments": tuple(lineage.material_segments.segments),
        "num_blocks": int(lineage.num_blocks),
        "partials": partials,
        "variant_indices": tuple(int(value) for value in lineage.variant_indices),
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
        partials=snapshot.get("partials"),
        variant_indices=tuple(snapshot.get("variant_indices", ())),
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
        "previous_accumulated_log_likelihood": float(
            state.accumulated_log_likelihood
        ),
        "previous_log_likelihood": state.log_likelihood,
        "previous_log_reward": state.log_reward,
        "previous_terminal_partial_correction": float(
            state.terminal_partial_correction
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


def _public_likelihood_alignment(
    alignment: Mapping[str, Any],
) -> dict[str, Any]:
    if not alignment:
        return {}
    return {
        "vcf_path": alignment.get("vcf_path"),
        "vcf_parser_version": alignment.get("parser_version"),
        "vcf_coordinate_offset": alignment.get("vcf_coordinate_offset"),
        "matched_variant_count": alignment.get("matched_variant_count"),
        "sample_nodes": tuple(
            int(value) for value in alignment.get("sample_nodes", ())
        ),
        "haplotype_index_by_sample_node": dict(
            alignment.get("haplotype_index_by_sample_node", {})
        ),
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
        and float(record["time"]) >= float(state.current_time)
    ]
    return min(times) if times else None


def _vcf_breakpoints(
    material: MaterialSegments,
    state: ARGState,
    env: SimpleARGEnvironment,
) -> tuple[tuple[int, float], ...]:
    if material.span_start is None or material.span_end is None:
        return ()
    output = []
    candidates = (
        sorted(state.local_breakpoint_weights)
        if state.target_material is not None
        else range(
            int(material.span_start) + 1,
            int(material.span_end) + 1,
        )
    )
    for breakpoint in candidates:
        if not (
            int(material.span_start)
            < int(breakpoint)
            <= int(material.span_end)
        ):
            continue
        left, right = material.split(breakpoint)
        if left.count == 0 or right.count == 0:
            continue
        weight = (
            float(
                state.local_breakpoint_weights.get(
                    int(breakpoint),
                    env._breakpoint_gap_length(breakpoint),
                )
            )
            if env.is_vcf_mode
            else 1.0
        )
        if weight > 0.0:
            output.append((int(breakpoint), weight))
    return tuple(output)


def _has_valid_breakpoint(
    material: MaterialSegments,
    state: ARGState,
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
    return bool(_vcf_breakpoints(material, state, env))


def _is_valid_breakpoint(
    material: MaterialSegments,
    state: ARGState,
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
    if not env.is_vcf_mode:
        return True
    return any(
        int(value) == breakpoint and float(weight) > 0.0
        for value, weight in _vcf_breakpoints(material, state, env)
    )


def _local_recombination_weight(
    lineage: ARGLineage,
    state: ARGState,
    env: SimpleARGEnvironment,
) -> float:
    if env.is_vcf_mode:
        return float(
            sum(
                weight
                for _breakpoint, weight in _vcf_breakpoints(
                    lineage.material_segments,
                    state,
                    env,
                )
            )
        )
    return float(lineage.material_segments.count)


def _sample_breakpoint(
    material: MaterialSegments,
    state: ARGState,
    env: SimpleARGEnvironment,
    rng: np.random.Generator,
) -> tuple[int, float]:
    if not _has_valid_breakpoint(material, state, env):
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
    values = _vcf_breakpoints(material, state, env)
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
    "compute_local_terminal_log_likelihood",
    "compute_local_terminal_log_reward",
    "enumerate_local_prior_actions",
    "initialize_local_arg_state",
    "local_is_terminal",
    "local_state_to_proposal",
    "reveal_due_fixed_ancestors",
    "sample_local_prior_action",
    "sample_local_trajectories",
    "undo_local_transition",
]
