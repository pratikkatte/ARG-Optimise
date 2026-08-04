"""Truth-free scoring and selection for local ARG refinement proposals."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

from .local_construction import (
    _collapse_temporary_source_endpoint,
    local_state_to_proposal,
)
from .local_splice import splice_local_proposal

try:
    from ..env import CoalescenceChoice, FixedAttachmentChoice, RecombinationChoice
except ImportError:
    from env import CoalescenceChoice, FixedAttachmentChoice, RecombinationChoice


DEFAULT_SELECTION_MARGIN = 1e-6


@dataclass(frozen=True)
class LocalARGScore:
    whole_log_likelihood: float
    local_log_likelihood: float
    log_prior: float
    log_posterior: float
    topology_digest: str
    splice_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalARGComparison:
    source: LocalARGScore
    candidate: LocalARGScore
    likelihood_delta: float
    prior_delta: float
    posterior_delta: float
    improves: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "candidate": self.candidate.to_dict(),
            "likelihood_delta": float(self.likelihood_delta),
            "prior_delta": float(self.prior_delta),
            "posterior_delta": float(self.posterior_delta),
            "improves": bool(self.improves),
        }


class SourceScoreUnavailable(RuntimeError):
    """Raised when the source ARG cannot be replayed under the local CWR model."""


def score_terminal_state(local_env, state) -> LocalARGScore:
    if not state.is_done or state.log_reward is None or state.log_likelihood is None:
        raise ValueError("local ARG scoring requires a likelihood-scored terminal state")
    prepared = local_env.context_for_state(state)
    proposal = local_state_to_proposal(state, prepared)
    splice = splice_local_proposal(prepared, proposal)
    return LocalARGScore(
        whole_log_likelihood=float(state.log_likelihood),
        local_log_likelihood=float(state.log_likelihood - state.outside_log_likelihood),
        log_prior=float(state.accumulated_log_prior),
        log_posterior=float(state.log_reward),
        topology_digest=proposal.topology_digest,
        splice_valid=bool(splice.is_valid),
    )


def compare_scores(
    source: LocalARGScore,
    candidate: LocalARGScore,
    *,
    margin: float = DEFAULT_SELECTION_MARGIN,
) -> LocalARGComparison:
    likelihood_delta = candidate.local_log_likelihood - source.local_log_likelihood
    prior_delta = candidate.log_prior - source.log_prior
    posterior_delta = candidate.log_posterior - source.log_posterior
    finite = all(math.isfinite(value) for value in (
        likelihood_delta, prior_delta, posterior_delta
    ))
    return LocalARGComparison(
        source=source,
        candidate=candidate,
        likelihood_delta=float(likelihood_delta),
        prior_delta=float(prior_delta),
        posterior_delta=float(posterior_delta),
        improves=bool(candidate.splice_valid and finite and posterior_delta > float(margin)),
    )


def select_best_candidate_record(records) -> dict[str, Any] | None:
    """Return the improving record with greatest delta and stable index tie-break."""

    improving = [
        row
        for row in records
        if (row.get("comparison") or {}).get("improves")
        and row.get("output_file")
    ]
    if not improving:
        return None
    return max(
        improving,
        key=lambda row: (
            float(row["comparison"]["posterior_delta"]),
            -int(row["index"]),
        ),
    )


def replay_source_score(local_env, context_id: str) -> LocalARGScore:
    """Replay the selected source events using the exact local environment.

    The replay deliberately fails closed.  A source event that is not in the
    policy's support cannot be assigned a made-up prior density.
    """

    try:
        return _replay_source_score(local_env, str(context_id))
    except SourceScoreUnavailable:
        raise
    except Exception as error:
        raise SourceScoreUnavailable(str(error)) from error


def _replay_source_score(local_env, context_id: str) -> LocalARGScore:
    prepared = local_env.prepared_contexts[context_id]
    context = prepared.context
    trace = prepared.trace
    state = local_env.get_initial_state(context_id)
    source_to_live = {int(node_id): int(node_id) for node_id in state.all_nodes}
    for lineage in context.cut_active_lineages:
        collapsed = _collapse_temporary_source_endpoint(
            prepared,
            int(lineage.node_id),
            lineage.mutable_segments,
        )
        if len(collapsed) == 1:
            source_to_live[int(lineage.node_id)] = int(next(iter(collapsed)))

    selected = sorted(context.selected_events, key=lambda row: (row.time, row.event_index))
    for selected_event in selected:
        event_time = float(selected_event.time) / float(state.time_scale)
        while True:
            fixed_time = local_env.next_fixed_ancestor_time(state)
            if fixed_time is None or fixed_time > event_time + 1e-12:
                break
            before = set(state.all_nodes)
            log_prior = local_env.fixed_attachment_log_prior(state)
            state = local_env.apply_action(
                state,
                FixedAttachmentChoice(event_time=float(fixed_time)),
                log_prior=log_prior,
            )
            for node_id in set(state.all_nodes) - before:
                source_to_live[int(node_id)] = int(node_id)
        if state.is_done:
            break
        if event_time <= float(state.current_time) + 1e-14:
            continue

        source_event = trace.event_at_index(selected_event.event_index)
        edge_ids = tuple(int(value) for value in source_event.edge_ids)
        child_ids = tuple(sorted({int(trace.edge_child[e]) for e in edge_ids}))
        parent_ids = tuple(int(value) for value in source_event.node_ids)
        active_by_id = {
            int(lineage.node_id): index
            for index, lineage in enumerate(state.active_lineages)
        }
        delta = event_time - float(state.current_time)

        if source_event.kind == "coalescence":
            live_children = [source_to_live.get(value, value) for value in child_ids]
            live_children = [value for value in live_children if value in active_by_id]
            if len(live_children) != 2:
                raise SourceScoreUnavailable(
                    "source coalescence is outside the continuous binary CWR "
                    f"support: active_children={len(live_children)}"
                )
            action = CoalescenceChoice(
                active_lineage_i=active_by_id[live_children[0]],
                active_lineage_j=active_by_id[live_children[1]],
                delta_time=delta,
                time_quantile=local_env.delta_to_time_quantile(state, delta),
            )
        elif source_event.kind == "recombination":
            if len(child_ids) != 1:
                raise SourceScoreUnavailable("source recombination has no unique child")
            live_child = source_to_live.get(child_ids[0], child_ids[0])
            if live_child not in active_by_id:
                raise SourceScoreUnavailable(
                    "source recombination child is not active: "
                    f"source={child_ids[0]} live={live_child} "
                    f"active={sorted(active_by_id)}"
                )
            base_choice = next(
                (
                    choice for choice in local_env.enumerate_prior_options(state).recomb_choices
                    if choice.active_lineage_i == active_by_id[live_child]
                ),
                None,
            )
            if base_choice is None:
                raise SourceScoreUnavailable("source recombination is outside policy support")
            parent_material = []
            for parent_id in parent_ids:
                intervals = [
                    (float(trace.edge_left[e]), float(trace.edge_right[e]))
                    for e in edge_ids if int(trace.edge_parent[e]) == parent_id
                ]
                parent_material.append(intervals)
            boundaries = tuple(float(value) for value in state.block_boundaries)
            candidates = local_env.valid_breakpoints(state, base_choice)
            source_boundaries = [
                right
                for intervals in parent_material
                for _left, right in intervals
                if boundaries[0] < right < boundaries[-1]
            ]
            breakpoint = (
                min(
                    candidates,
                    key=lambda value: min(
                        abs(boundaries[value] - coordinate)
                        for coordinate in source_boundaries
                    ),
                )
                if candidates and source_boundaries
                else None
            )
            if breakpoint is None:
                raise SourceScoreUnavailable("source breakpoint is outside policy support")
            action = replace(
                base_choice,
                breakpoint=int(breakpoint),
                delta_time=delta,
                time_quantile=local_env.delta_to_time_quantile(state, delta),
            )
        else:
            continue

        before_max = int(state.max_node_idx)
        log_prior = local_env.compute_cwr_event_log_prior(state, action)
        state = local_env.apply_action(state, action, log_prior=log_prior)
        created = tuple(range(before_max + 1, int(state.max_node_idx) + 1))
        if len(created) != len(parent_ids):
            raise SourceScoreUnavailable("source event arity differs from local action")
        if source_event.kind == "recombination":
            ordered_source = sorted(
                parent_ids,
                key=lambda parent_id: min(
                    float(trace.edge_left[e])
                    for e in edge_ids if int(trace.edge_parent[e]) == parent_id
                ),
            )
            ordered_created = sorted(
                created,
                key=lambda node_id: (
                    state.all_nodes[node_id].material_segments.span_start
                    if state.all_nodes[node_id].material_segments.span_start is not None
                    else math.inf
                ),
            )
            source_to_live.update(zip(ordered_source, ordered_created))
        else:
            source_to_live.update(zip(parent_ids, created))

    while not state.is_done:
        fixed_time = local_env.next_fixed_ancestor_time(state)
        if fixed_time is None:
            raise SourceScoreUnavailable("source replay ended before a local terminal state")
        log_prior = local_env.fixed_attachment_log_prior(state)
        state = local_env.apply_action(
            state,
            FixedAttachmentChoice(event_time=float(fixed_time)),
            log_prior=log_prior,
        )
    score = score_terminal_state(local_env, state)
    proposal = local_state_to_proposal(state, prepared)
    replayed = splice_local_proposal(prepared, proposal)
    if not replayed.is_valid:
        raise SourceScoreUnavailable("source replay did not produce a valid splice")
    interval = tuple(float(value) for value in context.request.genomic_range)
    if _marginal_topology_signature(replayed.refined_tree_sequence, interval) != (
        _marginal_topology_signature(prepared.source_tree_sequence, interval)
    ):
        raise SourceScoreUnavailable("source replay topology differs from the input ARG")
    return score


def _marginal_topology_signature(tree_sequence, interval):
    left_bound, right_bound = interval
    rows = []
    for tree in tree_sequence.trees():
        left = max(float(tree.interval.left), left_bound)
        right = min(float(tree.interval.right), right_bound)
        if left >= right:
            continue
        clades = []
        for node_id in tree.nodes():
            samples = tuple(sorted(int(value) for value in tree.samples(node_id)))
            if len(samples) >= 2:
                clades.append((samples, round(float(tree.time(node_id)), 9)))
        rows.append((round(left, 9), round(right, 9), tuple(sorted(clades))))
    return tuple(rows)
