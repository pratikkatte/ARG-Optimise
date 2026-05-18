import copy
import math
import numbers
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ARGLineage:
    node_id: int
    children: List[int] = field(default_factory=list)
    parents: List[int] = field(default_factory=list)
    material_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    partials: Optional[Any] = None
    sequences_indices: List[int] = field(default_factory=list)
    event_type: Optional[str] = None
    breakpoint: Optional[int] = None
    recombination_side: Optional[str] = None

    def __post_init__(self):
        self.children = list(self.children)
        self.parents = list(self.parents)
        self.material_mask = np.asarray(self.material_mask, dtype=bool).copy()
        self.sequences_indices = list(self.sequences_indices)

    def clone(self):
        return ARGLineage(
            node_id=self.node_id,
            children=list(self.children),
            parents=list(self.parents),
            material_mask=self.material_mask.copy(),
            partials=copy.deepcopy(self.partials),
            sequences_indices=list(self.sequences_indices),
            event_type=self.event_type,
            breakpoint=self.breakpoint,
            recombination_side=self.recombination_side,
        )


@dataclass
class ARGState:
    active_lineages: List[ARGLineage]
    all_nodes: Dict[int, ARGLineage]
    max_node_idx: int
    log_reward: Optional[float] = None
    accumulated_log_prior: float = 0.0
    is_done: bool = False
    action_options: Tuple[List[Dict[str, Any]], List[Tuple[int, int, List[int]]], List[Dict[str, Any]]] = None
    rates: Optional[Dict[str, float]] = None

    def clone(self):
        all_nodes = {node_id: lineage.clone() for node_id, lineage in self.all_nodes.items()}
        active_lineages = [all_nodes[lineage.node_id] for lineage in self.active_lineages]
        return ARGState(
            active_lineages=active_lineages,
            all_nodes=all_nodes,
            max_node_idx=self.max_node_idx,
            log_reward=self.log_reward,
            accumulated_log_prior=self.accumulated_log_prior,
            is_done=self.is_done,
        )


class SimpleARGEnvironment:
    """
    Minimal discrete coalescent-with-recombination ARG prototype.

    This intentionally avoids eete3, continuous breakpoints, likelihood scoring, and
    full continuous coalescent-with-recombination simulation. The placeholder reward
    is the accumulated log prior of the sampled event trajectory.
    """

    def __init__(
        self,
        num_sequences: Optional[int] = None,
        sequence_length: Optional[int] = None,
        num_blocks: Optional[int] = None,
        rho: float = 1.0,
        fixed_edge_length: float = 0.02,
        sequences: Optional[Sequence[Any]] = None,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
    ):
        self.sequences = list(sequences) if sequences is not None else None
        if self.sequences is not None:
            if num_sequences is None:
                num_sequences = len(self.sequences)
            elif int(num_sequences) != len(self.sequences):
                raise ValueError("num_sequences must equal len(sequences)")
            if sequence_length is None and self.sequences:
                sequence_length = len(self.sequences[0])
            if any(len(sequence) != sequence_length for sequence in self.sequences):
                raise ValueError("all sequences must have length sequence_length")

        if sequence_length is None and num_blocks is not None:
            sequence_length = num_blocks
        if num_blocks is None:
            num_blocks = sequence_length
        if num_sequences is None:
            raise ValueError("num_sequences is required when sequences are not provided")
        if sequence_length is None:
            raise ValueError("sequence_length or num_blocks is required")
        if num_blocks != sequence_length:
            raise ValueError("num_blocks must equal sequence_length")

        self.num_sequences = int(num_sequences)
        self.sequence_length = int(sequence_length)
        self.num_blocks = int(num_blocks)
        self.rho = float(rho)
        self.fixed_edge_length = float(fixed_edge_length)
        self.rng = rng if rng is not None else random.Random(seed)
        self.block_indices = np.arange(self.num_blocks)

    def get_initial_state(self):
        active_lineages = []
        all_nodes = {}
        for node_id in range(self.num_sequences):
            lineage = ARGLineage(
                node_id=node_id,
                children=[],
                parents=[],
                material_mask=np.ones(self.num_blocks, dtype=bool),
                partials=None,
                sequences_indices=[node_id],
            )
            active_lineages.append(lineage)
            all_nodes[node_id] = lineage

        state = ARGState(
            active_lineages=active_lineages,
            all_nodes=all_nodes,
            max_node_idx=self.num_sequences - 1,
            log_reward=None,
            accumulated_log_prior=0.0,
            is_done=False,
        )
        state.is_done = self.is_terminal(state)
        return state

    def get_active_counts(self, state):
        if not state.active_lineages:
            return np.zeros(self.num_blocks, dtype=int)
        masks = [lineage.material_mask.astype(int) for lineage in state.active_lineages]
        return np.sum(masks, axis=0)

    def get_active_segments(self, state):
        """Return block segments carried by each active lineage.

        Segment intervals are half-open block coordinates: ``(start, end)`` means
        blocks ``start`` through ``end - 1``. In a terminal state, these active
        segments show the final partition of sequence material.
        """
        return [
            {
                "node_id": lineage.node_id,
                "segments": self.mask_to_segments(lineage.material_mask),
                "blocks": np.flatnonzero(lineage.material_mask).tolist(),
                "sequences_indices": list(lineage.sequences_indices),
            }
            for lineage in state.active_lineages
        ]

    def get_recombination_breakpoints(self, state):
        """Return recombination split metadata recorded in an ARG state."""
        breakpoints_by_event = {}
        for lineage in sorted(state.all_nodes.values(), key=lambda node: node.node_id):
            if lineage.event_type != "recomb":
                continue
            child_node_id = lineage.children[0] if lineage.children else None
            key = (child_node_id, lineage.breakpoint)
            event = breakpoints_by_event.setdefault(
                key,
                {
                    "child_node_id": child_node_id,
                    "breakpoint": lineage.breakpoint,
                    "parent_node_ids": {},
                    "segments_by_side": {},
                    "blocks_by_side": {},
                    "sequences_indices": list(lineage.sequences_indices),
                },
            )
            side = lineage.recombination_side or f"parent_{lineage.node_id}"
            event["parent_node_ids"][side] = lineage.node_id
            event["segments_by_side"][side] = self.mask_to_segments(lineage.material_mask)
            event["blocks_by_side"][side] = np.flatnonzero(lineage.material_mask).tolist()
        return list(breakpoints_by_event.values())

    def mask_to_segments(self, material_mask):
        """Convert a boolean material mask into contiguous half-open segments."""
        mask = np.asarray(material_mask, dtype=bool)
        segments = []
        start = None
        for block_i, has_material in enumerate(mask):
            if has_material and start is None:
                start = block_i
            elif not has_material and start is not None:
                segments.append((start, block_i))
                start = None
        if start is not None:
            segments.append((start, len(mask)))
        return segments

    def is_terminal(self, state):
        return bool(np.all(self.get_active_counts(state) == 1))

    def _valid_breakpoints_for_lineage(self, state, active_lineage_i):
        valid_breakpoints = []
        for breakpoint in range(1, self.num_blocks):
            action = {"event_type": "recomb", "active_lineage_i": active_lineage_i, "breakpoint": breakpoint}
            if self.is_valid_recombination_action(state, action):
                valid_breakpoints.append(breakpoint)
        return valid_breakpoints

    def is_valid_coalescent_action(self, state, action):
        if action.get("event_type") != "coal":
            return False
        i = action.get("active_lineage_i")
        j = action.get("active_lineage_j")
        if not self._is_active_index(state, i) or not self._is_active_index(state, j):
            return False
        if i == j:
            return False
        mask_i = state.active_lineages[i].material_mask
        mask_j = state.active_lineages[j].material_mask
        return bool(np.any(mask_i & mask_j))

    def enumerate_action_options(self, state):
        coal_actions = []
        recomb_weights = []
        recomb_actions = []
        if self.is_terminal(state):
            state.action_options = (coal_actions, recomb_weights, recomb_actions)
            state.rates = self.compute_event_rates(state)
            return coal_actions, recomb_weights, recomb_actions
        for i, lineage in enumerate(state.active_lineages):
            valid_breakpoints = self._valid_breakpoints_for_lineage(state, i)
            if valid_breakpoints:
                recomb_weights.append((i, int(lineage.material_mask.sum()), valid_breakpoints))
                for breakpoint in valid_breakpoints:
                    recomb_actions.append({"event_type": "recomb", "active_lineage_i": i, "breakpoint": breakpoint})
            for j in range(i + 1, len(state.active_lineages)):
                action = {"event_type": "coal", "active_lineage_i": i, "active_lineage_j": j}
                if self.is_valid_coalescent_action(state, action):
                    coal_actions.append(action)

        state.action_options = (coal_actions, recomb_weights, recomb_actions)
        rates = self.compute_event_rates(state)
        state.rates = rates
        return coal_actions, recomb_weights, recomb_actions

    def enumerate_actions(self, state, separate_coal_and_recomb=False):
        coal_actions, _, recomb_actions = self.enumerate_action_options(state)
        if separate_coal_and_recomb:
            return coal_actions, recomb_actions
        else:
            return coal_actions + recomb_actions

    def enumerate_coalescent_actions(self, state):
        coal_actions, _, _ = self.enumerate_action_options(state)
        return coal_actions

    def enumerate_recombination_actions(self, state):
        _, _, recomb_actions = self.enumerate_action_options(state)
        return recomb_actions

    def is_coalescence_action_valid(self, state, action):
        return self.is_valid_coalescent_action(state, action)

    def is_valid_recombination_action(self, state, action):
        if action.get("event_type") != "recomb":
            return False
        i = action.get("active_lineage_i")
        breakpoint = action.get("breakpoint")
        if not self._is_active_index(state, i):
            return False
        if not isinstance(breakpoint, numbers.Integral) or breakpoint < 1 or breakpoint >= self.num_blocks:
            return False
        left_mask, right_mask = self._split_mask(state.active_lineages[i].material_mask, breakpoint)
        return bool(np.any(left_mask) and np.any(right_mask))


    def apply_coalescence(self, state, action, log_prior=None):
        if not self.is_valid_coalescent_action(state, action):
            raise ValueError(f"Invalid coalescence action: {action}")

        if log_prior is None:
            log_prior = self.compute_cwr_event_log_prior(state, action)
        next_state = state.clone()
        i = action["active_lineage_i"]
        j = action["active_lineage_j"]
        child_i = next_state.active_lineages[i]
        child_j = next_state.active_lineages[j]

        parent_id = next_state.max_node_idx + 1
        parent_mask = child_i.material_mask | child_j.material_mask
        parent = ARGLineage(
            node_id=parent_id,
            children=[child_i.node_id, child_j.node_id],
            parents=[],
            material_mask=parent_mask,
            partials=None,
            sequences_indices=sorted(set(child_i.sequences_indices + child_j.sequences_indices)),
            event_type="coal",
        )

        child_i.parents.append(parent.node_id)
        child_j.parents.append(parent.node_id)
        next_state.all_nodes[parent.node_id] = parent
        next_state.active_lineages = [
            lineage for idx, lineage in enumerate(next_state.active_lineages) if idx not in (i, j)
        ]
        next_state.active_lineages.append(parent)
        next_state.max_node_idx = parent.node_id
        next_state.accumulated_log_prior += log_prior
        next_state.log_reward = next_state.accumulated_log_prior
        next_state.is_done = self.is_terminal(next_state)
        return next_state

    def apply_recombination(self, state, action, log_prior=None):
        if not self.is_valid_recombination_action(state, action):
            raise ValueError(f"Invalid recombination action: {action}")

        if log_prior is None:
            log_prior = self.compute_cwr_event_log_prior(state, action)
        next_state = state.clone()
        next_state.action_options = None
        next_state.rates = None
        i = action["active_lineage_i"]
        breakpoint = action["breakpoint"]
        child = next_state.active_lineages[i]
        original_mask = child.material_mask.copy()
        left_mask, right_mask = self._split_mask(original_mask, breakpoint)

        if not np.array_equal(left_mask | right_mask, original_mask):
            raise ValueError("Recombination masks must cover the original material")
        if np.any(left_mask & right_mask):
            raise ValueError("Recombination masks must be disjoint")

        left_parent_id = next_state.max_node_idx + 1
        right_parent_id = next_state.max_node_idx + 2
        left_parent = ARGLineage(
            node_id=left_parent_id,
            children=[child.node_id],
            parents=[],
            material_mask=left_mask,
            partials=None,
            sequences_indices=list(child.sequences_indices),
            event_type="recomb",
            breakpoint=breakpoint,
            recombination_side="left",
        )
        right_parent = ARGLineage(
            node_id=right_parent_id,
            children=[child.node_id],
            parents=[],
            material_mask=right_mask,
            partials=None,
            sequences_indices=list(child.sequences_indices),
            event_type="recomb",
            breakpoint=breakpoint,
            recombination_side="right",
        )

        child.parents = [left_parent.node_id, right_parent.node_id]
        next_state.all_nodes[left_parent.node_id] = left_parent
        next_state.all_nodes[right_parent.node_id] = right_parent
        next_state.active_lineages = [
            lineage for idx, lineage in enumerate(next_state.active_lineages) if idx != i
        ]
        next_state.active_lineages.extend([left_parent, right_parent])
        next_state.max_node_idx = right_parent.node_id
        next_state.accumulated_log_prior += log_prior
        next_state.log_reward = next_state.accumulated_log_prior
        next_state.is_done = self.is_terminal(next_state)
        return next_state

    def apply_action(self, state, action, log_prior=None):
        if action.get("event_type") == "coal":
            return self.apply_coalescence(state, action, log_prior)
        if action.get("event_type") == "recomb":
            return self.apply_recombination(state, action, log_prior)
        raise ValueError(f"Unknown action event_type: {action}")

    def compute_event_rates(self, state, coal_actions=None, recomb_weights=None):
        if coal_actions is None or recomb_weights is None:
            if state.action_options is None:
                self.enumerate_action_options(state)
            coal_actions, recomb_weights, _ = state.action_options

        lambda_coal = float(len(coal_actions))
        total_active_material_length = self.total_active_material_length(state)
        lambda_recomb = 0.0
        if recomb_weights:
            lambda_recomb = self.rho / 2.0 * total_active_material_length
        return {
            "lambda_coal": lambda_coal,
            "lambda_recomb": lambda_recomb,
            "total_active_material_length": total_active_material_length,
        }

    def compute_event_probabilities(self, state):
        rates = self.compute_event_rates(state) if state.rates is None else state.rates
        denom = rates["lambda_coal"] + rates["lambda_recomb"]
        
        if denom <= 0:
            return {"coal": 0.0, "recomb": 0.0}
        return {
            "coal": rates["lambda_coal"] / denom,
            "recomb": rates["lambda_recomb"] / denom,
        }

    def compute_action_prior_distribution(self, state, event_type=None):
        coal_actions, recomb_weights, recomb_actions = self.enumerate_action_options(state) if state.action_options is None else state.action_options
        rates = state.rates if state.rates is not None else self.compute_event_rates(state, coal_actions, recomb_weights)
        denom = rates["lambda_coal"] + rates["lambda_recomb"]
        if denom <= 0:
            return []
        distribution = []
        if event_type == "coal":
            if coal_actions and rates["lambda_coal"] > 0:
                log_prob = math.log(rates["lambda_coal"] / denom) - math.log(len(coal_actions))
                distribution.extend((dict(action), log_prob) for action in coal_actions)
        else:
            if recomb_actions and rates["lambda_recomb"] > 0:
                total_weight = sum(weight for _, weight, _ in recomb_weights)
                if total_weight > 0:
                    weight_by_lineage = {
                        lineage_i: (weight, valid_breakpoints)
                        for lineage_i, weight, valid_breakpoints in recomb_weights
                    }
                    log_recomb_event = math.log(rates["lambda_recomb"] / denom)
                    for action in recomb_actions:
                        weight, valid_breakpoints = weight_by_lineage[action["active_lineage_i"]]
                        log_prob = (
                            log_recomb_event
                            + math.log(weight / total_weight)
                            - math.log(len(valid_breakpoints))
                        )
                        distribution.append((dict(action), log_prob))

        return distribution

    def compute_cwr_event_log_prior(self, state, action, rates=None, recomb_weights=None):
        if state.action_options is None:
            self.enumerate_action_options(state)
        coal_actions = state.action_options[0]
        if recomb_weights is None:
            recomb_weights = state.action_options[1]
        if rates is None:
            rates = state.rates if state.rates is not None else self.compute_event_rates(state)

        denom = rates["lambda_coal"] + rates["lambda_recomb"]
        if denom <= 0:
            return -math.inf
        if action.get("event_type") == "coal":
            if not self.is_valid_coalescent_action(state, action) or not coal_actions:
                return -math.inf
            action_prob = (rates["lambda_coal"] / denom) / len(coal_actions)
            return math.log(action_prob) if action_prob > 0 else -math.inf
        if action.get("event_type") == "recomb":
            if not self.is_valid_recombination_action(state, action):
                return -math.inf
            total_weight = sum(weight for _, weight, _ in recomb_weights)
            if total_weight <= 0:
                return -math.inf
            for lineage_i, weight, valid_breakpoints in recomb_weights:
                if lineage_i != action.get("active_lineage_i"):
                    continue
                if action.get("breakpoint") not in valid_breakpoints:
                    return -math.inf
                action_prob = (
                    (rates["lambda_recomb"] / denom)
                    * (weight / total_weight)
                    / len(valid_breakpoints)
                )
                return math.log(action_prob) if action_prob > 0 else -math.inf
            return -math.inf
        return -math.inf

    def compute_smcprime_event_log_prior(self, state, action, rates=None, recomb_weights=None):
        return self.compute_cwr_event_log_prior(state, action, rates, recomb_weights)

    def sample_action_from_prior(self, state):
        try:
            from .rollout_worker_arg import RolloutWorker
        except ImportError:
            from rollout_worker_arg import RolloutWorker

        return RolloutWorker(self).sample_action_from_prior(state)

    def rollout(self, max_steps=100, num_trajectories=1):
        try:
            from .rollout_worker_arg import RolloutWorker
        except ImportError:
            from rollout_worker_arg import RolloutWorker

        return RolloutWorker(
            self,
            max_steps=max_steps,
            num_trajectories=num_trajectories,
        ).rollout()

    def total_active_material_length(self, state):
        total_blocks = sum(int(lineage.material_mask.sum()) for lineage in state.active_lineages)
        return float(total_blocks) / float(self.num_blocks)

    def _split_mask(self, material_mask, breakpoint):
        mask = np.asarray(material_mask, dtype=bool)
        left_mask = mask & (self.block_indices < breakpoint)
        right_mask = mask & (self.block_indices >= breakpoint)
        return left_mask, right_mask

    def _is_active_index(self, state, idx):
        return isinstance(idx, numbers.Integral) and 0 <= idx < len(state.active_lineages)
