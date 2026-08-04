"""Compact hybrid experience replay for local ARG refinement training."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

try:
    from ..env import (
        CoalescenceChoice,
        FixedAttachmentChoice,
        RecombinationChoice,
    )
except ImportError:  # Support the repository's script-style entry points.
    from env import (
        CoalescenceChoice,
        FixedAttachmentChoice,
        RecombinationChoice,
    )
try:
    from ..hybrid_replay_config import (
        DEFAULT_HYBRID_REPLAY_CONFIG,
        REPLAY_SOURCES,
        normalize_hybrid_replay_config,
    )
except ImportError:  # Support the repository's script-style entry points.
    from hybrid_replay_config import (
        DEFAULT_HYBRID_REPLAY_CONFIG,
        REPLAY_SOURCES,
        normalize_hybrid_replay_config,
    )


class FractionalQuotaAllocator:
    """Allocate integer source counts with bounded long-run rounding error."""

    def __init__(self, fractions: Mapping[str, float]):
        self.fractions = {
            name: float(fractions[name]) for name in REPLAY_SOURCES
        }
        self.carry = {name: 0.0 for name in REPLAY_SOURCES}

    def allocate(self, count: int):
        count = int(count)
        if count <= 0:
            raise ValueError("quota count must be positive")
        raw = {
            name: count * self.fractions[name] + self.carry[name]
            for name in REPLAY_SOURCES
        }
        allocated = {name: math.floor(raw[name]) for name in REPLAY_SOURCES}
        remaining = count - sum(allocated.values())
        ranked = sorted(
            REPLAY_SOURCES,
            key=lambda name: (raw[name] - allocated[name], -REPLAY_SOURCES.index(name)),
            reverse=True,
        )
        for name in ranked[:remaining]:
            allocated[name] += 1
        self.carry = {
            name: raw[name] - allocated[name] for name in REPLAY_SOURCES
        }
        return allocated


def _canonical_action(action):
    event_type = str(action["event_type"])
    if event_type == "coal":
        keys = (
            "event_type",
            "active_lineage_i",
            "active_lineage_j",
            "time_quantile",
            "delta_time",
        )
    elif event_type == "recomb":
        keys = (
            "event_type",
            "active_lineage_i",
            "material_count",
            "span_start",
            "span_end",
            "breakpoint",
            "time_quantile",
            "delta_time",
        )
    elif event_type == "fixed_attachment":
        keys = ("event_type", "event_time")
    else:
        raise ValueError(f"unsupported replay action type {event_type!r}")
    return {key: action[key] for key in keys if key in action}


def trajectory_signature(context_id: str, actions: Sequence[Mapping]):
    payload = {
        "context_id": str(context_id),
        "actions": [_canonical_action(action) for action in actions],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def structural_topology_signature(state):
    """Hash local ARG structure while deliberately excluding continuous times."""

    nodes = []
    for node_id, lineage in sorted(state.all_nodes.items()):
        nodes.append(
            (
                int(node_id),
                str(lineage.event_type),
                tuple(sorted(int(value) for value in lineage.children)),
                tuple(sorted(int(value) for value in lineage.parents)),
                tuple(
                    (int(left), int(right))
                    for left, right in lineage.material_segments.segments
                ),
                None if lineage.breakpoint is None else int(lineage.breakpoint),
                lineage.recombination_side,
            )
        )
    payload = {
        "context_id": str(state.local_context_id),
        "nodes": nodes,
        "roots": tuple(int(lineage.node_id) for lineage in state.active_lineages),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass
class ReplayEntry:
    entry_id: int
    context_id: str
    actions: tuple[dict, ...]
    signature: str
    topology_signature: str
    log_reward: float
    residual_priority: float
    inserted_step: int
    last_scored_step: int


class HybridReplayBuffer:
    """Per-context bounded reservoir with prioritized sampling indexes."""

    def __init__(self, context_ids, capacity_per_context, top_fraction, seed):
        self.context_ids = tuple(str(value) for value in context_ids)
        self.capacity_per_context = int(capacity_per_context)
        self.top_fraction = float(top_fraction)
        self.rng = random.Random(int(seed))
        self._entries = {context_id: [] for context_id in self.context_ids}
        self._by_signature = {context_id: {} for context_id in self.context_ids}
        self._seen = {context_id: 0 for context_id in self.context_ids}
        self._next_entry_id = 0

    def __len__(self):
        return sum(len(values) for values in self._entries.values())

    def context_size(self, context_id):
        return len(self._entries[str(context_id)])

    def add(
        self,
        context_id,
        actions,
        terminal_state,
        residual_priority,
        step,
    ):
        context_id = str(context_id)
        canonical_actions = tuple(_canonical_action(action) for action in actions)
        signature = trajectory_signature(context_id, canonical_actions)
        existing = self._by_signature[context_id].get(signature)
        if existing is not None:
            existing.residual_priority = float(residual_priority)
            existing.log_reward = float(terminal_state.log_reward)
            existing.last_scored_step = int(step)
            return existing, "updated"

        self._seen[context_id] += 1
        entry = ReplayEntry(
            entry_id=self._next_entry_id,
            context_id=context_id,
            actions=canonical_actions,
            signature=signature,
            topology_signature=structural_topology_signature(terminal_state),
            log_reward=float(terminal_state.log_reward),
            residual_priority=float(residual_priority),
            inserted_step=int(step),
            last_scored_step=int(step),
        )
        self._next_entry_id += 1
        entries = self._entries[context_id]
        if len(entries) < self.capacity_per_context:
            entries.append(entry)
            self._by_signature[context_id][signature] = entry
            return entry, "inserted"

        reservoir_index = self.rng.randrange(self._seen[context_id])
        if reservoir_index >= self.capacity_per_context:
            return None, "discarded"
        evicted = entries[reservoir_index]
        del self._by_signature[context_id][evicted.signature]
        entries[reservoir_index] = entry
        self._by_signature[context_id][signature] = entry
        return entry, "replaced"

    def update_priority(self, entry, residual_priority, log_reward, step):
        entry.residual_priority = float(residual_priority)
        entry.log_reward = float(log_reward)
        entry.last_scored_step = int(step)

    def sample(self, source, context_id, excluded_ids=()):
        context_id = str(context_id)
        excluded_ids = set(excluded_ids)
        candidates = [
            entry
            for entry in self._entries[context_id]
            if entry.entry_id not in excluded_ids
        ]
        if not candidates:
            return None
        if source in {"residual", "reward"}:
            field = (
                "residual_priority" if source == "residual" else "log_reward"
            )
            candidates.sort(key=lambda entry: getattr(entry, field), reverse=True)
            tier_size = max(1, math.ceil(self.top_fraction * len(candidates)))
            return self.rng.choice(candidates[:tier_size])
        if source == "topology":
            by_topology = {}
            for entry in candidates:
                by_topology.setdefault(entry.topology_signature, []).append(entry)
            topology = self.rng.choice(sorted(by_topology))
            return self.rng.choice(by_topology[topology])
        raise ValueError(f"unsupported replay source {source!r}")

    def metrics(self, current_step):
        entries = [entry for values in self._entries.values() for entry in values]
        result = {
            "replay/buffer_size": len(entries),
            "replay/buffer_capacity": (
                self.capacity_per_context * len(self.context_ids)
            ),
            "replay/unique_topology_count": len(
                {entry.topology_signature for entry in entries}
            ),
        }
        for context_id in self.context_ids:
            result[f"replay/context/{context_id}/size"] = len(
                self._entries[context_id]
            )
        if entries:
            result.update(
                {
                    "replay/residual_priority_mean": sum(
                        entry.residual_priority for entry in entries
                    ) / len(entries),
                    "replay/log_reward_mean": sum(
                        entry.log_reward for entry in entries
                    ) / len(entries),
                    "replay/priority_age_mean": sum(
                        int(current_step) - entry.last_scored_step
                        for entry in entries
                    ) / len(entries),
                }
            )
        else:
            result.update(
                {
                    "replay/residual_priority_mean": 0.0,
                    "replay/log_reward_mean": 0.0,
                    "replay/priority_age_mean": 0.0,
                }
            )
        return result


def _parse_action(action):
    event_type = action.get("event_type")
    if event_type == "fixed_attachment":
        return FixedAttachmentChoice(event_time=float(action["event_time"]))
    parsed = CoalescenceChoice.from_action(action)
    if parsed is None:
        parsed = RecombinationChoice.from_action(action)
    if parsed is None:
        raise ValueError(f"invalid replay action {action!r}")
    return parsed


def _pad_vectors(vectors, device, dtype=torch.float32):
    if not vectors:
        return torch.empty(0, 0, dtype=dtype, device=device)
    return torch.nn.utils.rnn.pad_sequence(
        [vector.to(device=device, dtype=dtype) for vector in vectors],
        batch_first=True,
        padding_value=0.0,
    )


def reconstruct_and_rescore_entries(env, generator, entries):
    """Rebuild replay paths and score their recorded actions with the current policy."""

    state_paths = []
    action_paths = []
    parent_count_paths = []
    flat_states = []
    flat_actions = []
    with torch.no_grad():
        for entry in entries:
            state = env.get_initial_state(entry.context_id)
            states = [state]
            actions = []
            parent_counts = []
            for action_dict in entry.actions:
                action = _parse_action(action_dict)
                if isinstance(action, FixedAttachmentChoice):
                    log_prior = env.fixed_attachment_log_prior(state)
                else:
                    log_prior = env.compute_cwr_event_log_prior(
                        state,
                        env.enumerate_actions(state),
                        action,
                    )
                flat_states.append(state)
                flat_actions.append(action_dict)
                state = env.apply_action(state, action, log_prior=log_prior)
                states.append(state)
                actions.append(dict(action_dict))
                parent_count = generator.count_backward_parents(state)
                if parent_count <= 0:
                    raise RuntimeError(
                        "replayed forward transition has no backward representation"
                    )
                parent_counts.append(parent_count)
            if not state.is_done or state.log_reward is None:
                raise RuntimeError("replayed trajectory did not reach a terminal state")
            if structural_topology_signature(state) != entry.topology_signature:
                raise RuntimeError("replayed trajectory changed structural topology")
            state_paths.append(states)
            action_paths.append(actions)
            parent_count_paths.append(parent_counts)

    scored = generator.score_local_transitions(flat_states, flat_actions)
    pf_vectors = []
    pb_vectors = []
    component_paths = []
    diagnostic_paths = []
    cursor = 0
    for actions, parent_counts in zip(action_paths, parent_count_paths):
        length = len(actions)
        pf_vectors.append(scored["total"][cursor:cursor + length])
        pb_vectors.append(
            -torch.log(
                torch.as_tensor(
                    parent_counts,
                    dtype=torch.float32,
                    device=generator.device,
                )
            )
        )
        component_paths.append(
            [
                {
                    name: float(scored[name][index].detach().cpu().item())
                    for name in (
                        "gate",
                        "atomic_action",
                        "breakpoint",
                        "time",
                        "total",
                    )
                }
                for index in range(cursor, cursor + length)
            ]
        )
        diagnostic_paths.append(
            [dict(row) for row in scored["policy_diagnostics"][cursor:cursor + length]]
        )
        cursor += length

    device = generator.device
    lengths = torch.as_tensor(
        [len(actions) for actions in action_paths],
        dtype=torch.long,
        device=device,
    )
    sampled_actions = [action for actions in action_paths for action in actions]
    generated_actions = [
        action for action in sampled_actions if action.get("time_quantile") is not None
    ]
    generated_time_scores = [
        scored["time"][index]
        for index, action in enumerate(flat_actions)
        if action.get("time_quantile") is not None
    ]
    event_counts = {
        event_type: torch.as_tensor(
            [
                sum(action.get("event_type") == event_type for action in actions)
                for actions in action_paths
            ],
            dtype=torch.long,
            device=device,
        )
        for event_type in ("coal", "recomb", "fixed_attachment")
    }
    return {
        "log_paths_pf": _pad_vectors(pf_vectors, device),
        "log_paths_pb": _pad_vectors(pb_vectors, device),
        "log_rewards": torch.as_tensor(
            [float(path[-1].log_reward) for path in state_paths],
            dtype=torch.float32,
            device=device,
        ),
        "trajectory_states": state_paths,
        "trajectory_actions": action_paths,
        "trajectory_log_components": component_paths,
        "trajectory_policy_diagnostics": diagnostic_paths,
        "backward_parent_counts": parent_count_paths,
        "trajectory_lengths": lengths,
        "terminal_mask": torch.ones(len(entries), dtype=torch.bool, device=device),
        "truncated_mask": torch.zeros(len(entries), dtype=torch.bool, device=device),
        "time_quantiles": torch.as_tensor(
            [float(action["time_quantile"]) for action in generated_actions],
            dtype=torch.float64,
            device=device,
        ),
        "time_delta_times": torch.as_tensor(
            [float(action["delta_time"]) for action in generated_actions],
            dtype=torch.float64,
            device=device,
        ),
        "time_log_densities": (
            torch.stack(generated_time_scores).to(dtype=torch.float64)
            if generated_time_scores
            else torch.empty(0, dtype=torch.float64, device=device)
        ),
        "time_policy_entropies": torch.full(
            (len(generated_actions),), float("nan"), dtype=torch.float64, device=device
        ),
        "time_effective_components": torch.full(
            (len(generated_actions),), float("nan"), dtype=torch.float64, device=device
        ),
        "time_event_times": torch.as_tensor(
            [
                float(action.get("time_context_diagnostics", {}).get("current_time", 0.0))
                + float(action["delta_time"])
                for action in generated_actions
            ],
            dtype=torch.float64,
            device=device,
        ),
        "time_context_diagnostics": [
            dict(action.get("time_context_diagnostics") or {})
            for action in generated_actions
        ],
        "fixed_attachment_count": sum(
            action.get("event_type") == "fixed_attachment"
            for action in sampled_actions
        ),
        "coalescence_counts": event_counts["coal"],
        "recombination_counts": event_counts["recomb"],
        "fixed_attachment_counts": event_counts["fixed_attachment"],
    }


def merge_rollout_outputs(*outputs):
    """Concatenate fresh and replay rollout dictionaries without detaching graphs."""

    outputs = tuple(value for value in outputs if value is not None)
    if not outputs:
        raise ValueError("at least one rollout output is required")
    device = outputs[0]["log_paths_pf"].device
    pf_vectors = []
    pb_vectors = []
    for value in outputs:
        for row, length in enumerate(value["trajectory_lengths"].detach().cpu().tolist()):
            pf_vectors.append(value["log_paths_pf"][row, :int(length)])
            pb_vectors.append(value["log_paths_pb"][row, :int(length)])
    merged = {
        "log_paths_pf": _pad_vectors(pf_vectors, device),
        "log_paths_pb": _pad_vectors(pb_vectors, device),
    }
    tensor_keys = (
        "log_rewards",
        "trajectory_lengths",
        "terminal_mask",
        "truncated_mask",
        "time_quantiles",
        "time_delta_times",
        "time_log_densities",
        "time_policy_entropies",
        "time_effective_components",
        "time_event_times",
        "coalescence_counts",
        "recombination_counts",
        "fixed_attachment_counts",
    )
    for key in tensor_keys:
        merged[key] = torch.cat([value[key] for value in outputs], dim=0)
    list_keys = (
        "trajectory_states",
        "trajectory_actions",
        "trajectory_log_components",
        "trajectory_policy_diagnostics",
        "backward_parent_counts",
        "time_context_diagnostics",
    )
    for key in list_keys:
        merged[key] = [item for value in outputs for item in value[key]]
    merged["fixed_attachment_count"] = sum(
        int(value["fixed_attachment_count"]) for value in outputs
    )
    return merged


def max_abs_subtb_residuals(balance_details, trajectory_count):
    priorities = [0.0] * int(trajectory_count)
    for record in balance_details.get("records", ()):
        index = int(record["trajectory_index"])
        value = abs(float(record["residual"].detach().cpu().item()))
        priorities[index] = max(priorities[index], value)
    return priorities


__all__ = [
    "DEFAULT_HYBRID_REPLAY_CONFIG",
    "FractionalQuotaAllocator",
    "HybridReplayBuffer",
    "REPLAY_SOURCES",
    "max_abs_subtb_residuals",
    "merge_rollout_outputs",
    "normalize_hybrid_replay_config",
    "reconstruct_and_rescore_entries",
    "structural_topology_signature",
    "trajectory_signature",
]
