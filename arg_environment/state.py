import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .actions import CoalescenceChoice, PriorActionOptions, RecombinationChoice
from .material import MaterialSegments

class ARGLineage:
    def __init__(
        self,
        node_id: int,
        children: Optional[Sequence[int]] = None,
        parents: Optional[Sequence[int]] = None,
        material_mask: Optional[np.ndarray] = None,
        material_segments: Optional[MaterialSegments] = None,
        num_blocks: Optional[int] = None,
        partials: Optional[Any] = None,
        sequences_indices: Optional[Sequence[int]] = None,
        event_type: Optional[str] = None,
        breakpoint: Optional[int] = None,
        recombination_side: Optional[str] = None,
        time: float = 0.0,
        preview_pair_features: Optional[torch.Tensor] = None,
    ):
        self.node_id = int(node_id)
        self.children = list(children or [])
        self.parents = list(parents or [])
        self.partials = partials
        # Encoding-only channels: nucleotide disagreement (4) and overlap (1).
        # Committed lineages contain likelihood partials and leave this unset.
        self.preview_pair_features = preview_pair_features
        self.sequences_indices = list(sequences_indices or [])
        self.event_type = event_type
        self.breakpoint = breakpoint
        self.recombination_side = recombination_side
        self.time = float(time)
        self._material_mask = None

        if material_segments is None:
            mask = np.asarray([] if material_mask is None else material_mask, dtype=bool)
            self._material_mask = mask.copy()
            self.material_segments = MaterialSegments.from_mask(self._material_mask)
            self.num_blocks = int(mask.size if num_blocks is None else num_blocks)
        else:
            self.material_segments = MaterialSegments.from_segments(material_segments)
            self.num_blocks = int(
                num_blocks
                if num_blocks is not None
                else max((end for _, end in self.material_segments.segments), default=0)
            )
            if material_mask is not None:
                self._material_mask = np.asarray(material_mask, dtype=bool).copy()
                self.num_blocks = int(self._material_mask.size)

    @property
    def material_mask(self):
        if self._material_mask is None:
            self._material_mask = self.material_segments.to_mask(self.num_blocks)
        return self._material_mask

    @material_mask.setter
    def material_mask(self, value):
        if value is None:
            self._material_mask = None
            return
        self._material_mask = np.asarray(value, dtype=bool).copy()
        self.num_blocks = int(self._material_mask.size)
        self.material_segments = MaterialSegments.from_mask(self._material_mask)

    @property
    def material_count(self):
        return self.material_segments.count

    @property
    def material_span(self):
        if self.material_segments.count < 2:
            return None
        return (
            self.material_segments.span_start,
            self.material_segments.span_end,
            self.material_segments.count,
        )

    def clone(self, copy_partials=True, copy_mask=True):
        if not copy_partials:
            partials = self.partials
        elif torch.is_tensor(self.partials):
            partials = self.partials.clone()
        else:
            partials = copy.deepcopy(self.partials)

        clone = ARGLineage(
            node_id=self.node_id,
            children=list(self.children),
            parents=list(self.parents),
            material_segments=self.material_segments,
            num_blocks=self.num_blocks,
            partials=partials,
            sequences_indices=list(self.sequences_indices),
            event_type=self.event_type,
            breakpoint=self.breakpoint,
            recombination_side=self.recombination_side,
            time=float(self.time),
            preview_pair_features=(
                self.preview_pair_features.clone()
                if copy_partials and self.preview_pair_features is not None
                else self.preview_pair_features
            ),
        )
        if copy_mask and self._material_mask is not None:
            clone._material_mask = self._material_mask.copy()
        return clone

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
    prior_options: Optional[PriorActionOptions] = None
    total_active_blocks: Optional[int] = None
    current_time: float = 0.0

    def clone(self, copy_partials=False):
        all_nodes = {
            node_id: lineage.clone(copy_partials=copy_partials)
            for node_id, lineage in self.all_nodes.items()
        }
        active_lineages = [all_nodes[lineage.node_id] for lineage in self.active_lineages]
        return ARGState(
            active_lineages=active_lineages,
            all_nodes=all_nodes,
            max_node_idx=self.max_node_idx,
            log_reward=self.log_reward,
            accumulated_log_prior=self.accumulated_log_prior,
            is_done=self.is_done,
            total_active_blocks=self.total_active_blocks,
            current_time=float(self.current_time),
        )


class SimpleTrajectory:
    """Compact trajectory history used when cloned ARG states are not needed."""

    def __init__(self):
        self.actions = []
        self.log_priors = []
        self.records = []
        self.log_reward = None

    def update(self, action, log_prior=None, log_reward=None, record=None, active_lineages=None):
        self.actions.append(action_as_dict(action))
        self.log_priors.append(log_prior)
        self.log_reward = log_reward
        if record is not None:
            self.records.append(record)

    def __len__(self):
        return len(self.actions)


def action_as_dict(action):
    if isinstance(action, dict):
        return dict(action)
    if isinstance(action, CoalescenceChoice):
        return action.as_dict()
    if isinstance(action, RecombinationChoice):
        result = {
            "event_type": "recomb",
            "active_lineage_i": int(action.active_lineage_i),
            "breakpoint": int(action.breakpoint) if action.breakpoint is not None else None,
            "material_count": int(action.material_count),
            "span_start": int(action.span_start),
            "span_end": int(action.span_end),
        }
        if action.time_action is not None:
            result["time_action"] = int(action.time_action)
        return result
    raise ValueError(f"Unknown ARG action: {action}")

class ARGReward:
    """
    Terminal reward helpers for constructed ARG states.
    """

    def __init__(self, C=30000):
        self.C = C

    def __call__(self, log_likelihood, accumulated_log_prior):
        return self.compute_terminal_posterior_log_reward(log_likelihood, accumulated_log_prior)

    def compute_terminal_posterior_log_reward(self, log_likelihood, accumulated_log_prior):
        return float(self.C + log_likelihood + accumulated_log_prior)

