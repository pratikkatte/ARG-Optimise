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
        )


@dataclass
class ARGState:
    active_lineages: List[ARGLineage]
    all_nodes: Dict[int, ARGLineage]
    max_node_idx: int
    log_reward: Optional[float] = None
    accumulated_log_prior: float = 0.0
    is_done: bool = False

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

    This intentionally avoids ete3, continuous breakpoints, likelihood scoring, and
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
        if num_blocks is not None:
            num_blocks = sequence_length

        self.num_sequences = int(num_sequences)
        self.sequence_length = int(sequence_length)
        self.num_blocks = self.sequence_length
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

    def active_count(self, state):
        return self.get_active_counts(state)

    def is_terminal(self, state):
        return bool(np.all(self.get_active_counts(state) == 1))

    def enumerate_coalescent_actions(self, state):
        actions = []
        for i in range(len(state.active_lineages)):
            for j in range(i + 1, len(state.active_lineages)):
                action = {"event_type": "coal", "lineage_i": i, "lineage_j": j}
                if self.is_valid_coalescent_action(state, action):
                    actions.append(action)
        return actions

    def enumerate_recombination_actions(self, state):
        actions = []
        for i in range(len(state.active_lineages)):
            for breakpoint in range(1, self.num_blocks):
                action = {"event_type": "recomb", "lineage_i": i, "breakpoint": breakpoint}
                if self.is_valid_recombination_action(state, action):
                    actions.append(action)
        return actions

    def _valid_breakpoints_for_lineage(self, state, lineage_i):
        valid_breakpoints = []
        for breakpoint in range(1, self.num_blocks):
            action = {"event_type": "recomb", "lineage_i": lineage_i, "breakpoint": breakpoint}
            if self.is_valid_recombination_action(state, action):
                valid_breakpoints.append(breakpoint)
        return valid_breakpoints

    def _recombinable_lineage_weights(self, state):
        lineage_weights = []
        for idx, lineage in enumerate(state.active_lineages):
            valid_breakpoints = self._valid_breakpoints_for_lineage(state, idx)
            if valid_breakpoints:
                lineage_weights.append((idx, int(lineage.material_mask.sum()), valid_breakpoints))
        return lineage_weights

    def enumerate_actions(self, state, separate_coal_and_recomb=False):
        if separate_coal_and_recomb:
            return self.enumerate_coalescent_actions(state), self.enumerate_recombination_actions(state)
        else:
            return self.enumerate_coalescent_actions(state) + self.enumerate_recombination_actions(state)

    def is_valid_coalescent_action(self, state, action):
        if action.get("event_type") != "coal":
            return False
        i = action.get("lineage_i")
        j = action.get("lineage_j")
        if not self._is_active_index(state, i) or not self._is_active_index(state, j):
            return False
        if i == j:
            return False
        mask_i = state.active_lineages[i].material_mask
        mask_j = state.active_lineages[j].material_mask
        return bool(np.any(mask_i & mask_j))

    def is_coalescence_action_valid(self, state, action):
        return self.is_valid_coalescent_action(state, action)

    def is_valid_recombination_action(self, state, action):
        if action.get("event_type") != "recomb":
            return False
        i = action.get("lineage_i")
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
        i = action["lineage_i"]
        j = action["lineage_j"]
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
        i = action["lineage_i"]
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
        )
        right_parent = ARGLineage(
            node_id=right_parent_id,
            children=[child.node_id],
            parents=[],
            material_mask=right_mask,
            partials=None,
            sequences_indices=list(child.sequences_indices),
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

    def compute_event_rates(self, state, coal_actions=None):
        if coal_actions is None:
            coal_actions = self.enumerate_coalescent_actions(state)
        lambda_coal = float(len(coal_actions))
        total_active_material_length = self.total_active_material_length(state)
        lambda_recomb = self.rho / 2.0 * total_active_material_length
        return {
            "lambda_coal": lambda_coal,
            "lambda_recomb": lambda_recomb,
            "total_active_material_length": total_active_material_length,
        }

    def compute_event_probabilities(self, state):
        rates = self.compute_event_rates(state)
        denom = rates["lambda_coal"] + rates["lambda_recomb"]
        if denom <= 0:
            return {"coal": 0.0, "recomb": 0.0}
        return {
            "coal": rates["lambda_coal"] / denom,
            "recomb": rates["lambda_recomb"] / denom,
        }

    def compute_cwr_event_log_prior(self, state, action, rates=None, recomb_weights=None):
        if rates is None:
            rates = self.compute_event_rates(state)

        denom = rates["lambda_coal"] + rates["lambda_recomb"]
        if denom <= 0:
            return -math.inf
        if action.get("event_type") == "recomb":
            if recomb_weights is None:
                recomb_weights = self._recombinable_lineage_weights(state)

            total_weight = sum(weight for _, weight, _ in recomb_weights)
            if total_weight <= 0:
                return -math.inf
            action_prob = rates["lambda_recomb"] / denom
            return math.log(action_prob)
        return -math.inf

    def compute_smcprime_event_log_prior(self, state, action, rates=None, recomb_weights=None):
        return self.compute_cwr_event_log_prior(state, action, rates, recomb_weights)

    def sample_action_from_prior(self, state):
        """
        """
        coal_actions = self.enumerate_coalescent_actions(state)
        if not coal_actions:
            return None

        recomb_weights = self._recombinable_lineage_weights(state)
        rates = self.compute_event_rates(state, coal_actions)

        lambda_coal = rates["lambda_coal"] if coal_actions else 0.0
        lambda_recomb = rates["lambda_recomb"] if recomb_weights else 0.0
        denom = lambda_coal + lambda_recomb
        if denom <= 0:
            return None

        if lambda_recomb > 0 and self.rng.random() < lambda_recomb / denom:
            total_weight = sum(weight for _, weight, _ in recomb_weights)
            selected = self._weighted_choice(recomb_weights, total_weight)
            lineage_i, _, valid_breakpoints = selected
            return {
                "event_type": "recomb",
                "lineage_i": lineage_i,
                "breakpoint": self.rng.choice(valid_breakpoints),
            }, rates, recomb_weights

        if not coal_actions:
            return None
        return dict(self.rng.choice(coal_actions)), rates, recomb_weights

    def rollout(self, max_steps=100):
        state = self.get_initial_state()
        trajectory = []
        for step in range(max_steps):
            if state.is_done:
                break
            action, rates, recomb_weights = self.sample_action_from_prior(state)
            if action is None:
                break
            log_prior = self.compute_cwr_event_log_prior(state, action, rates, recomb_weights)
            state = self.apply_action(state, action, log_prior)
            trajectory.append(
                {
                    "step": step,
                    "action": action,
                    "log_prior": log_prior,
                    "active_lineage_count": len(state.active_lineages),
                    "active_counts": self.get_active_counts(state).tolist(),
                    "is_done": state.is_done,
                    "log_reward": state.log_reward,
                }
            )
        return state, trajectory

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


    def _weighted_choice(self, weighted_items: List[Tuple[int, int, List[int]]], total_weight: float):
        threshold = self.rng.random() * total_weight
        running = 0.0
        for item in weighted_items:
            running += item[1]
            if threshold <= running:
                return item
        return weighted_items[-1]
