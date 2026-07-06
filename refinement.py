from __future__ import annotations

import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from env import (
    ARGLineage,
    ARGState,
    CoalescenceChoice,
    MaterialSegments,
    RecombinationChoice,
)
from utils import load_vcf_variants


SAME_TIME_EPS_GENERATIONS = 1.0
TIME_ABS_TOL = 1e-9
TIME_REL_TOL = 1e-12
VCF_INDEX_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}
BACKTRACK_STRATEGIES = {
    "before_last_touch",
    "before_first_touch",
    "before_last_coalescence",
}


@dataclass(frozen=True)
class RefinementRegion:
    index: int
    blocks: Tuple[int, ...]
    left_bp: float
    right_bp: float
    max_bad_region_score: float = 0.0
    sum_bad_region_score: float = 0.0
    variant_positions: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class RefinementContext:
    region: RefinementRegion
    partial_state: ARGState
    touching_events: List[Dict[str, Any]]
    backtrack_step: int
    target_blocks: Tuple[int, ...]
    effective_blocks: Tuple[int, ...]
    rollout_mode: str = "segment"
    backtrack_offset: int = 0
    strategy_backtrack_step: Optional[int] = None

    def action_filter(self):
        return LocalRegionActionFilter(self.effective_blocks)

    def to_manifest_record(self):
        return {
            "region": int(self.region.index),
            "blocks": list(self.region.blocks),
            "left_bp": float(self.region.left_bp),
            "right_bp": float(self.region.right_bp),
            "max_bad_region_score": float(self.region.max_bad_region_score),
            "sum_bad_region_score": float(self.region.sum_bad_region_score),
            "variant_positions": list(self.region.variant_positions),
            "backtrack_step": int(self.backtrack_step),
            "target_blocks": list(self.target_blocks),
            "effective_blocks": list(self.effective_blocks),
            "touching_event_steps": [
                int(event["step"]) for event in self.touching_events
            ],
            "partial_active_lineages": len(self.partial_state.active_lineages),
            "partial_total_active_blocks": int(self.partial_state.total_active_blocks),
            "rollout_mode": str(self.rollout_mode),
            "backtrack_offset": int(self.backtrack_offset),
            "strategy_backtrack_step": (
                int(self.strategy_backtrack_step)
                if self.strategy_backtrack_step is not None
                else None
            ),
        }


class LazyCanonicalStateStore:
    """Sequence-like view over canonical replay states without retaining them all."""

    def __init__(self, source, events):
        self.source = source
        self.events = tuple(events)
        self._cache = OrderedDict()
        self._max_cached_states = 16

    def __len__(self):
        return len(self.events) + 1

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        step_index = self._normalize_index(index)
        cached = self._cache.get(step_index)
        if cached is not None:
            self._cache.move_to_end(step_index)
            return cached
        state = self.source._materialize_canonical_state(step_index)
        self._remember(step_index, state)
        return state

    def __iter__(self):
        for step_index in range(len(self)):
            yield self[step_index]

    def nearest_cached_state(self, step_index):
        candidates = [step for step in self._cache if step <= step_index]
        if not candidates:
            return None, None
        step = max(candidates)
        self._cache.move_to_end(step)
        return step, self._cache[step]

    def _remember(self, step_index, state):
        self._cache[int(step_index)] = state
        self._cache.move_to_end(int(step_index))
        while len(self._cache) > self._max_cached_states:
            oldest_step = next(iter(self._cache))
            if oldest_step == 0 and len(self._cache) > 1:
                self._cache.move_to_end(oldest_step)
                oldest_step = next(iter(self._cache))
            self._cache.pop(oldest_step)

    def _normalize_index(self, index):
        step_index = int(index)
        if step_index < 0:
            step_index += len(self)
        if not 0 <= step_index < len(self):
            raise IndexError(
                f"canonical state index must be in [0, {len(self) - 1}], "
                f"got {index}"
            )
        return step_index


class LocalRegionActionFilter:
    """Restrict local rollouts to actions that touch selected ARG blocks."""

    def __init__(self, blocks: Iterable[int]):
        self.blocks = tuple(sorted({int(block) for block in blocks}))
        self.block_set = set(self.blocks)

    def __call__(self, state, coal_actions, recomb_actions):
        coal = [
            action
            for action in coal_actions
            if self._coal_touches_blocks(state, action)
        ]
        recomb = []
        for action in recomb_actions:
            recomb.extend(self._restricted_recombination_choices(state, action))
        return coal, recomb

    def action_touches_blocks(self, state, action):
        if isinstance(action, CoalescenceChoice):
            return self._coal_touches_blocks(state, action)
        if isinstance(action, RecombinationChoice):
            lineage = state.active_lineages[int(action.active_lineage_i)]
            valid = self._valid_recombination_breakpoints(lineage, action)
            return int(action.breakpoint) in valid
        return False

    def _coal_touches_blocks(self, state, action):
        left = state.active_lineages[int(action.active_lineage_i)].material_segments
        right = state.active_lineages[int(action.active_lineage_j)].material_segments
        overlap = left.intersection(right)
        return any(
            block in self.block_set
            for start, end in overlap.segments
            for block in range(start, end)
        )

    def _restricted_recombination_choices(self, state, action):
        lineage = state.active_lineages[int(action.active_lineage_i)]
        segments = lineage.material_segments.intersection(
            MaterialSegments.from_segments(blocks_to_segments(self.block_set))
        )
        choices = []
        for start, end in segments.segments:
            if end - start < 2:
                continue
            choices.append(
                RecombinationChoice(
                    active_lineage_i=int(action.active_lineage_i),
                    material_count=int(end - start),
                    span_start=int(start),
                    span_end=int(end - 1),
                )
            )
        return choices

    def _valid_recombination_breakpoints(self, lineage, action):
        valid = set()
        for breakpoint in range(int(action.span_start) + 1, int(action.span_end) + 1):
            if breakpoint not in self.block_set and breakpoint - 1 not in self.block_set:
                continue
            left, right = lineage.material_segments.split(breakpoint)
            if left.count > 0 and right.count > 0:
                valid.add(int(breakpoint))
        return valid


class RefinementSource:
    def __init__(
        self,
        env,
        trees_path,
        vcf_path,
        population_size=None,
        mutation_rate=None,
        trim_uncovered_flanks=True,
        same_time_eps_generations=SAME_TIME_EPS_GENERATIONS,
    ):
        try:
            import tskit
        except ImportError as exc:
            raise ImportError(
                "tskit is required for local ARG refinement from .trees files."
            ) from exc

        self.tskit = tskit
        self.env = env
        self.trees_path = str(trees_path)
        self.vcf_path = str(vcf_path)
        self.population_size = float(
            env.population_size if population_size is None else population_size
        )
        self.mutation_rate = float(
            env.mutation_rate if mutation_rate is None else mutation_rate
        )
        self.same_time_eps_generations = float(same_time_eps_generations)
        self.ts = tskit.load(self.trees_path)
        self.edges = list(self.ts.edges())
        if not self.edges:
            raise ValueError(f"{self.trees_path} has no edges")
        if not getattr(env, "is_vcf_mode", False):
            raise ValueError("local ARG refinement currently requires a VCF environment")

        self.vcf_data = load_vcf_variants(self.vcf_path)
        self._validate_inputs()

        self.vcf_positions = np.asarray(self.vcf_data.positions0, dtype=np.float64) + 1.0
        self.vcf_partials = np.asarray(
            self.vcf_data.haplotype_partials,
            dtype=np.float64,
        )
        self.vcf_genotypes = np.argmax(self.vcf_partials, axis=-1)
        self.num_blocks = int(env.num_blocks)
        self.block_boundaries_bp = np.asarray(
            env.variant_boundaries,
            dtype=np.float64,
        ).copy()
        if self.block_boundaries_bp.size > 2:
            self.block_boundaries_bp[1:-1] += 1.0
        self.block_table = self._build_block_table()
        self.block_trees = [self.ts.at(float(position)) for position in self.vcf_positions]
        self.samples = [int(sample) for sample in self.ts.samples()]
        self.sample_set = set(self.samples)
        self.sample_index_by_node = {
            node_id: idx for idx, node_id in enumerate(self.samples)
        }
        if self.samples != list(range(int(env.num_sequences))):
            raise ValueError(
                "local refinement expects tskit sample node IDs to match "
                "environment haplotype indices 0..num_sequences-1"
            )

        if trim_uncovered_flanks:
            self.analysis_left = min(float(edge.left) for edge in self.edges)
            self.analysis_right = max(float(edge.right) for edge in self.edges)
        else:
            self.analysis_left = 0.0
            self.analysis_right = float(self.ts.sequence_length)

        self.edge_segments_by_pair = self._build_edge_segments_by_pair()
        self._validate_edge_segments()
        self.parents_by_node, self.children_by_node = self._build_parent_child_maps()
        self.node_blocks, self.node_descendant_samples = self._build_node_block_maps()
        self.material_nodes = sorted(self.node_blocks)
        self.node_local_time = {
            node_id: float(self.ts.node(node_id).time) / (2.0 * self.population_size)
            for node_id in self.material_nodes
        }
        self.canonical_node_metadata = self._initial_canonical_node_metadata()
        (
            self.canonical_action_trace,
            self.canonical_terminal_root_by_block,
            self.canonical_unary_skips,
            self.canonical_same_time_groups_adjusted,
        ) = self._build_canonical_action_trace()
        self._canonical_terminal_state = None
        self.canonical_states = self.replay_canonical_action_trace(
            self.canonical_action_trace
        )
        self._canonical_events_by_block = None

    @property
    def canonical_terminal_state(self):
        if self._canonical_terminal_state is None:
            terminal = self.canonical_states[-1]
            errors = self.validate_canonical_terminal_state(terminal)
            if errors:
                raise ValueError(
                    "canonical terminal validation failed: " + "; ".join(errors)
                )
            self._canonical_terminal_state = terminal
        return self._canonical_terminal_state

    def _validate_inputs(self):
        if int(self.vcf_data.num_haplotypes) != int(self.ts.num_samples):
            raise ValueError(
                f"VCF haplotype count {self.vcf_data.num_haplotypes} does not match "
                f"tree sequence sample count {self.ts.num_samples}"
            )
        if int(self.vcf_data.sequence_length) != int(self.ts.sequence_length):
            raise ValueError(
                f"VCF sequence length {self.vcf_data.sequence_length} does not match "
                f"tree sequence length {self.ts.sequence_length}"
            )
        if int(self.vcf_data.num_variants) != int(self.env.num_blocks):
            raise ValueError(
                f"VCF variant count {self.vcf_data.num_variants} does not match "
                f"environment blocks {self.env.num_blocks}"
            )
        if list(self.vcf_data.haplotype_ids) != list(getattr(self.env, "haplotype_ids", [])):
            raise ValueError("VCF haplotype IDs do not match the environment")

    def _build_block_table(self):
        return [
            {
                "block": int(idx),
                "variant_position0": int(self.vcf_data.positions0[idx]),
                "variant_position_vcf": float(self.vcf_positions[idx]),
                "left_bp": float(self.block_boundaries_bp[idx]),
                "right_bp": float(self.block_boundaries_bp[idx + 1]),
                "width_bp": float(
                    self.block_boundaries_bp[idx + 1] - self.block_boundaries_bp[idx]
                ),
            }
            for idx in range(int(self.env.num_blocks))
        ]

    def _build_edge_segments_by_pair(self):
        edge_segments_by_pair = defaultdict(list)
        for edge in self.edges:
            covered_blocks = np.flatnonzero(
                (self.vcf_positions >= float(edge.left))
                & (self.vcf_positions < float(edge.right))
            )
            if covered_blocks.size == 0:
                continue
            edge_segments_by_pair[(int(edge.parent), int(edge.child))].extend(
                blocks_to_segments(covered_blocks.tolist())
            )
        return {
            pair: canonical_segments(segments)
            for pair, segments in sorted(edge_segments_by_pair.items())
        }

    def _validate_edge_segments(self):
        direct_pairs_by_block = [set() for _ in self.block_table]
        for (parent, child), segments in self.edge_segments_by_pair.items():
            for left_block, right_block in segments:
                for block in range(left_block, right_block):
                    direct_pairs_by_block[block].add((parent, child))

        errors = []
        for block, tree in enumerate(self.block_trees):
            tree_pairs = set()
            for child in tree.nodes():
                parent = tree.parent(child)
                if parent != self.tskit.NULL:
                    tree_pairs.add((int(parent), int(child)))
            if tree_pairs != direct_pairs_by_block[block]:
                errors.append(block)
        if errors:
            raise ValueError(
                "edge table reconstruction does not match marginal trees for "
                f"block(s): {errors[:10]}"
            )

    def _build_parent_child_maps(self):
        parents_by_node = defaultdict(set)
        children_by_node = defaultdict(set)
        for parent, child in self.edge_segments_by_pair:
            parents_by_node[int(child)].add(int(parent))
            children_by_node[int(parent)].add(int(child))
        return parents_by_node, children_by_node

    def _build_node_block_maps(self):
        node_blocks = defaultdict(set)
        node_descendant_samples = defaultdict(set)
        for block, tree in enumerate(self.block_trees):
            for sample in self.samples:
                node = sample
                while node != self.tskit.NULL:
                    node = int(node)
                    node_blocks[node].add(block)
                    node_descendant_samples[node].add(
                        self.sample_index_by_node[int(sample)]
                    )
                    node = tree.parent(node)
        return node_blocks, node_descendant_samples

    def _initial_canonical_node_metadata(self):
        return {
            sample: canonical_node_record(
                sample,
                synthetic=False,
                original_tskit_node=sample,
                original_time_generations=0.0,
                adjusted_time_generations=0.0,
                event_type="sample",
                source="tskit_sample",
            )
            for sample in self.samples
        }

    def _build_canonical_action_trace(self):
        events, terminal_root_by_block, unary_skips = self._build_canonical_events()
        action_trace, same_time_groups_adjusted = self._assign_adjusted_event_times(events)
        return (
            action_trace,
            terminal_root_by_block,
            unary_skips,
            same_time_groups_adjusted,
        )

    def _build_canonical_events(self):
        synthetic_node_start = max(
            int(self.ts.num_nodes),
            max(self.material_nodes, default=-1) + 1,
        )
        node_counter = {"next": synthetic_node_start}
        events = []
        sample_block_lineage = {}

        for sample in self.samples:
            current_child = int(sample)
            for breakpoint in range(1, self.num_blocks):
                left_parent = allocate_synthetic_node_id(node_counter)
                right_parent = allocate_synthetic_node_id(node_counter)
                left_segments = ((breakpoint - 1, breakpoint),)
                right_segments = ((breakpoint, self.num_blocks),)
                event = {
                    "event_type": "recomb",
                    "source": "synthetic_sample_block_split",
                    "synthetic": True,
                    "child_ids": (current_child,),
                    "parent_ids": (left_parent, right_parent),
                    "breakpoint": int(breakpoint),
                    "material_segments": (left_segments[0], right_segments[0]),
                    "left_parent_id": left_parent,
                    "right_parent_id": right_parent,
                    "original_tskit_node": sample,
                    "original_time_generations": 0.0,
                }
                events.append(event)
                self.canonical_node_metadata[left_parent] = canonical_node_record(
                    left_parent,
                    original_tskit_node=sample,
                    original_time_generations=0.0,
                    event_type="recomb",
                    source="synthetic_sample_block_split_left",
                )
                self.canonical_node_metadata[right_parent] = canonical_node_record(
                    right_parent,
                    original_tskit_node=sample,
                    original_time_generations=0.0,
                    event_type="recomb",
                    source="synthetic_sample_block_split_right",
                )
                sample_block_lineage[(sample, breakpoint - 1)] = left_parent
                current_child = right_parent
            sample_block_lineage[(sample, self.num_blocks - 1)] = current_child

        canonical_for_tree_node = {}
        unary_skips = []

        def build_tree_node(block, tree, node):
            node = int(node)
            key = (int(block), node)
            if key in canonical_for_tree_node:
                return canonical_for_tree_node[key]
            if node in self.sample_set:
                canonical_for_tree_node[key] = sample_block_lineage[(node, block)]
                return canonical_for_tree_node[key]

            child_nodes = sorted(int(child) for child in tree.children(node))
            child_lineages = [
                build_tree_node(block, tree, child) for child in child_nodes
            ]
            if len(child_lineages) == 0:
                raise ValueError(
                    f"internal tree node {node} on block {block} has no children"
                )
            if len(child_lineages) == 1:
                unary_skips.append(
                    {
                        "block": int(block),
                        "tskit_node": node,
                        "child_lineage": child_lineages[0],
                    }
                )
                canonical_for_tree_node[key] = child_lineages[0]
                return child_lineages[0]

            current = child_lineages[0]
            original_time = float(self.ts.node(node).time)
            for child_idx, next_child in enumerate(child_lineages[1:], start=1):
                parent_id = allocate_synthetic_node_id(node_counter)
                event = {
                    "event_type": "coal",
                    "source": "synthetic_binary_coalescence",
                    "synthetic": True,
                    "child_ids": (current, next_child),
                    "parent_ids": (parent_id,),
                    "breakpoint": None,
                    "material_segments": ((int(block), int(block) + 1),),
                    "original_tskit_node": node,
                    "original_time_generations": original_time,
                    "block": int(block),
                    "tree_children": tuple(child_nodes),
                    "binary_child_index": int(child_idx),
                }
                events.append(event)
                self.canonical_node_metadata[parent_id] = canonical_node_record(
                    parent_id,
                    original_tskit_node=node,
                    original_time_generations=original_time,
                    event_type="coal",
                    source="synthetic_binary_coalescence",
                )
                current = parent_id
            canonical_for_tree_node[key] = current
            return current

        terminal_root_by_block = {}
        for block, tree in enumerate(self.block_trees):
            if len(tree.roots) != 1:
                raise ValueError(
                    f"block {block} has {len(tree.roots)} roots; "
                    "local refinement requires one marginal root per block"
                )
            terminal_root_by_block[block] = build_tree_node(
                block,
                tree,
                int(tree.roots[0]),
            )
        return events, terminal_root_by_block, unary_skips

    def _assign_adjusted_event_times(self, events):
        indexed_events = []
        for order, event in enumerate(events):
            cloned = dict(event)
            cloned["_canonical_order"] = int(order)
            indexed_events.append(cloned)

        by_original_time = defaultdict(list)
        for event in indexed_events:
            by_original_time[float(event["original_time_generations"])].append(event)

        original_times = sorted(by_original_time)
        adjusted = []
        same_time_groups_adjusted = 0
        for group_idx, original_time in enumerate(original_times):
            group = sorted(
                by_original_time[original_time],
                key=lambda item: item["_canonical_order"],
            )
            next_time = (
                original_times[group_idx + 1]
                if group_idx + 1 < len(original_times)
                else None
            )
            if next_time is None:
                available = self.same_time_eps_generations * max(len(group) + 1, 1)
            else:
                available = max(float(next_time) - float(original_time), 0.0)
            if original_time == 0.0:
                step = min(
                    self.same_time_eps_generations,
                    available / max(len(group) + 1, 1),
                )
                step = max(step, 1e-12)
                base = 0.0
                offsets = range(1, len(group) + 1)
            else:
                if available > 0:
                    step = min(
                        self.same_time_eps_generations,
                        available / max(len(group) + 1, 1),
                    )
                else:
                    step = 1e-12
                step = max(step, 1e-12)
                base = float(original_time)
                offsets = range(len(group))
            if len(group) > 1:
                same_time_groups_adjusted += 1
            for offset, event in zip(offsets, group):
                event["adjusted_time_generations"] = base + step * offset
                event["adjusted_time_t_over_2Ne"] = (
                    event["adjusted_time_generations"]
                    / (2.0 * self.population_size)
                )
                event["same_time_group_size"] = len(group)
                event["same_time_group_step_generations"] = step
                adjusted.append(event)
                for parent_id in event["parent_ids"]:
                    self.canonical_node_metadata[parent_id][
                        "adjusted_time_generations"
                    ] = event["adjusted_time_generations"]

        adjusted.sort(
            key=lambda item: (
                item["adjusted_time_generations"],
                item["_canonical_order"],
            )
        )
        for step_index, event in enumerate(adjusted, start=1):
            event["step"] = int(step_index)
        return adjusted, same_time_groups_adjusted

    def replay_canonical_action_trace(self, events):
        return LazyCanonicalStateStore(self, events)

    def _materialize_canonical_state(self, step_index):
        step_index = int(step_index)
        if not 0 <= step_index <= len(self.canonical_action_trace):
            raise IndexError(
                f"canonical state index must be in [0, {len(self.canonical_action_trace)}], "
                f"got {step_index}"
            )
        cached_step = None
        cached_state = None
        canonical_states = getattr(self, "canonical_states", None)
        if canonical_states is not None and hasattr(canonical_states, "nearest_cached_state"):
            cached_step, cached_state = canonical_states.nearest_cached_state(step_index)

        if cached_state is None:
            state = self.env.get_initial_state()
            state.canonical_step_index = 0
            start_step = 0
        else:
            state = cached_state.clone(copy_partials=False)
            state.canonical_step_index = int(cached_step)
            start_step = int(cached_step)

        active_by_id = active_index_by_node_id(state)
        for event_idx in range(start_step, step_index):
            event = self.canonical_action_trace[event_idx]
            self._apply_canonical_event_in_place(state, event, active_by_id)
            state.canonical_step_index = int(event["step"])
        return state

    def _apply_canonical_event_in_place(self, state, event, active_by_id):
        if event["event_type"] == "recomb":
            self._apply_canonical_recombination_in_place(state, event, active_by_id)
        elif event["event_type"] == "coal":
            self._apply_canonical_coalescence_in_place(state, event, active_by_id)
        else:
            raise ValueError(f"unknown canonical event type: {event['event_type']}")
        self._finalize_canonical_state_in_place(state, event)
        return state

    def _apply_canonical_recombination_in_place(self, state, event, active_by_id):
        child_id = int(event["child_ids"][0])
        if child_id not in active_by_id:
            raise ValueError(
                f"recombination child {child_id} is not active at step {event['step']}"
            )
        child_idx = active_by_id[child_id]
        child = state.active_lineages[child_idx]
        breakpoint = int(event["breakpoint"])
        left_segments, right_segments = child.material_segments.split(breakpoint)
        left_id, right_id = [int(node_id) for node_id in event["parent_ids"]]
        parent_time = float(event["adjusted_time_t_over_2Ne"])

        left_partials, right_partials = self._canonical_recombined_parent_partials(
            child,
            left_segments,
            right_segments,
            parent_time,
        )
        left_parent = ARGLineage(
            node_id=left_id,
            children=[child.node_id],
            parents=[],
            material_segments=left_segments,
            num_blocks=self.num_blocks,
            partials=left_partials,
            sequences_indices=list(child.sequences_indices),
            event_type="recomb",
            breakpoint=breakpoint,
            recombination_side="left",
            time=parent_time,
        )
        right_parent = ARGLineage(
            node_id=right_id,
            children=[child.node_id],
            parents=[],
            material_segments=right_segments,
            num_blocks=self.num_blocks,
            partials=right_partials,
            sequences_indices=list(child.sequences_indices),
            event_type="recomb",
            breakpoint=breakpoint,
            recombination_side="right",
            time=parent_time,
        )
        child.parents = [left_id, right_id]
        child.partials = None
        child.clear_runtime_caches()

        state.all_nodes[left_id] = left_parent
        state.all_nodes[right_id] = right_parent
        state.active_lineages.pop(child_idx)
        active_by_id.pop(child_id)
        self._refresh_active_indices_from(state, active_by_id, child_idx)
        state.active_lineages.extend([left_parent, right_parent])
        active_by_id[left_id] = len(state.active_lineages) - 2
        active_by_id[right_id] = len(state.active_lineages) - 1
        state.max_node_idx = max(state.max_node_idx, left_id, right_id)

    def _canonical_recombined_parent_partials(
        self,
        child,
        left_segments,
        right_segments,
        parent_time,
    ):
        transitioned = self.env._transition_lineage_partials(child, parent_time)
        if self.env.is_vcf_mode:
            left_selected = self.env._select_compact_partials(
                transitioned,
                child.material_segments,
                left_segments,
            )
            right_selected = self.env._select_compact_partials(
                transitioned,
                child.material_segments,
                right_segments,
            )
            return (
                self.env.evolution_model.normalize_partials(left_selected),
                self.env.evolution_model.normalize_partials(right_selected),
            )

        left_masked = self.env.evolution_model.mask_partials(
            transitioned,
            left_segments,
        )
        right_masked = self.env.evolution_model.mask_partials(
            transitioned,
            right_segments,
        )
        return (
            self.env.evolution_model.normalize_partials(left_masked),
            self.env.evolution_model.normalize_partials(right_masked),
        )

    def _apply_canonical_coalescence_in_place(self, state, event, active_by_id):
        child_ids = tuple(int(node_id) for node_id in event["child_ids"])
        missing = [node_id for node_id in child_ids if node_id not in active_by_id]
        if missing:
            raise ValueError(
                f"coalescence child/children {missing} are not active at step "
                f"{event['step']}"
            )
        child_i = state.active_lineages[active_by_id[child_ids[0]]]
        child_j = state.active_lineages[active_by_id[child_ids[1]]]
        parent_id = int(event["parent_ids"][0])
        parent_segments = child_i.material_segments.union(child_j.material_segments)
        overlap_count = child_i.material_segments.intersection_count(
            child_j.material_segments
        )
        parent_time = float(event["adjusted_time_t_over_2Ne"])
        parent_partials, _partial_log_likelihood_increment = (
            self.env._coalesced_parent_partials(
                child_i,
                child_j,
                parent_segments,
                parent_time,
            )
        )
        parent = ARGLineage(
            node_id=parent_id,
            children=[child_i.node_id, child_j.node_id],
            parents=[],
            material_segments=parent_segments,
            num_blocks=self.num_blocks,
            partials=parent_partials,
            sequences_indices=sorted(
                set(child_i.sequences_indices + child_j.sequences_indices)
            ),
            event_type="coal",
            time=parent_time,
        )
        child_i.parents.append(parent_id)
        child_j.parents.append(parent_id)
        child_i.partials = None
        child_j.partials = None
        child_i.clear_runtime_caches()
        child_j.clear_runtime_caches()

        state.all_nodes[parent_id] = parent
        remove_positions = sorted(
            (active_by_id[child_ids[0]], active_by_id[child_ids[1]]),
            reverse=True,
        )
        refresh_from = min(remove_positions)
        for position in remove_positions:
            removed = state.active_lineages.pop(position)
            active_by_id.pop(int(removed.node_id))
        self._refresh_active_indices_from(state, active_by_id, refresh_from)
        state.active_lineages.append(parent)
        active_by_id[parent_id] = len(state.active_lineages) - 1
        state.max_node_idx = max(state.max_node_idx, parent_id)
        if state.total_active_blocks is not None:
            state.total_active_blocks = int(state.total_active_blocks) - int(
                overlap_count
            )

    def _refresh_active_indices_from(self, state, active_by_id, start_idx):
        for idx in range(max(int(start_idx), 0), len(state.active_lineages)):
            active_by_id[int(state.active_lineages[idx].node_id)] = idx

    def _finalize_canonical_state_in_place(self, state, event):
        state.current_time = float(event["adjusted_time_t_over_2Ne"])
        state.log_reward = None
        state.action_options = None
        state.rates = None
        state.prior_options = None
        state.accumulated_log_prior = 0.0
        state.partial_log_reward = 0.0
        state.terminal_partial_correction = 0.0
        if state.total_active_blocks is None:
            state.total_active_blocks = int(
                sum(lineage.material_count for lineage in state.active_lineages)
            )
        state.is_done = self.env.is_terminal(state)

    def apply_canonical_event(self, state, event):
        if event["event_type"] == "recomb":
            return self._apply_canonical_recombination(state, event)
        if event["event_type"] == "coal":
            return self._apply_canonical_coalescence(state, event)
        raise ValueError(f"unknown canonical event type: {event['event_type']}")

    def _apply_canonical_recombination(self, state, event):
        next_state = state.clone(copy_partials=False)
        active_by_id = active_index_by_node_id(next_state)
        child_id = int(event["child_ids"][0])
        if child_id not in active_by_id:
            raise ValueError(
                f"recombination child {child_id} is not active at step {event['step']}"
            )
        child_idx = active_by_id[child_id]
        child = next_state.active_lineages[child_idx].clone(
            copy_partials=False,
            copy_mask=False,
        )
        breakpoint = int(event["breakpoint"])
        left_segments, right_segments = child.material_segments.split(breakpoint)
        left_id, right_id = [int(node_id) for node_id in event["parent_ids"]]
        parent_time = float(event["adjusted_time_t_over_2Ne"])

        left_partials = self.env._recombined_parent_partials(
            child,
            left_segments,
            parent_time,
        )
        right_partials = self.env._recombined_parent_partials(
            child,
            right_segments,
            parent_time,
        )
        left_parent = ARGLineage(
            node_id=left_id,
            children=[child.node_id],
            parents=[],
            material_segments=left_segments,
            num_blocks=self.num_blocks,
            partials=left_partials,
            sequences_indices=list(child.sequences_indices),
            event_type="recomb",
            breakpoint=breakpoint,
            recombination_side="left",
            time=parent_time,
        )
        right_parent = ARGLineage(
            node_id=right_id,
            children=[child.node_id],
            parents=[],
            material_segments=right_segments,
            num_blocks=self.num_blocks,
            partials=right_partials,
            sequences_indices=list(child.sequences_indices),
            event_type="recomb",
            breakpoint=breakpoint,
            recombination_side="right",
            time=parent_time,
        )
        child.parents = [left_id, right_id]
        child.partials = None
        child.clear_runtime_caches()

        next_state.all_nodes[child.node_id] = child
        next_state.all_nodes[left_id] = left_parent
        next_state.all_nodes[right_id] = right_parent
        next_state.active_lineages = [
            lineage
            for idx, lineage in enumerate(next_state.active_lineages)
            if idx != child_idx
        ]
        next_state.active_lineages.extend([left_parent, right_parent])
        return self._finalize_canonical_state(next_state, event)

    def _apply_canonical_coalescence(self, state, event):
        next_state = state.clone(copy_partials=False)
        active_by_id = active_index_by_node_id(next_state)
        child_ids = tuple(int(node_id) for node_id in event["child_ids"])
        missing = [node_id for node_id in child_ids if node_id not in active_by_id]
        if missing:
            raise ValueError(
                f"coalescence child/children {missing} are not active at step "
                f"{event['step']}"
            )
        child_i = next_state.active_lineages[active_by_id[child_ids[0]]].clone(
            copy_partials=False,
            copy_mask=False,
        )
        child_j = next_state.active_lineages[active_by_id[child_ids[1]]].clone(
            copy_partials=False,
            copy_mask=False,
        )
        parent_id = int(event["parent_ids"][0])
        parent_segments = child_i.material_segments.union(child_j.material_segments)
        parent_time = float(event["adjusted_time_t_over_2Ne"])
        parent_partials, partial_log_likelihood_increment = (
            self.env._coalesced_parent_partials(
                child_i,
                child_j,
                parent_segments,
                parent_time,
            )
        )
        parent = ARGLineage(
            node_id=parent_id,
            children=[child_i.node_id, child_j.node_id],
            parents=[],
            material_segments=parent_segments,
            num_blocks=self.num_blocks,
            partials=parent_partials,
            sequences_indices=sorted(
                set(child_i.sequences_indices + child_j.sequences_indices)
            ),
            event_type="coal",
            time=parent_time,
        )
        child_i.parents.append(parent_id)
        child_j.parents.append(parent_id)
        child_i.partials = None
        child_j.partials = None
        child_i.clear_runtime_caches()
        child_j.clear_runtime_caches()

        next_state.all_nodes[child_i.node_id] = child_i
        next_state.all_nodes[child_j.node_id] = child_j
        next_state.all_nodes[parent_id] = parent
        remove_ids = set(child_ids)
        next_state.active_lineages = [
            lineage
            for lineage in next_state.active_lineages
            if lineage.node_id not in remove_ids
        ]
        next_state.active_lineages.append(parent)
        next_state.partial_log_reward += float(partial_log_likelihood_increment)
        return self._finalize_canonical_state(next_state, event)

    def _finalize_canonical_state(self, state, event):
        state.max_node_idx = max(state.all_nodes) if state.all_nodes else -1
        state.current_time = float(event["adjusted_time_t_over_2Ne"])
        state.log_reward = None
        state.action_options = None
        state.rates = None
        state.prior_options = None
        state.accumulated_log_prior = 0.0
        state.partial_log_reward = 0.0
        state.terminal_partial_correction = 0.0
        state.total_active_blocks = int(
            sum(lineage.material_count for lineage in state.active_lineages)
        )
        state.is_done = self.env.is_terminal(state)
        return state

    def validate_canonical_terminal_state(self, state):
        errors = []
        if not state.is_done:
            errors.append("canonical terminal state is not marked done")
        if int(state.total_active_blocks) != int(self.env.num_blocks):
            errors.append(
                f"total_active_blocks={state.total_active_blocks} != "
                f"env.num_blocks={self.env.num_blocks}"
            )
        active_counts = self.env.get_active_counts(state)
        if active_counts.size != self.env.num_blocks or not np.all(active_counts == 1):
            errors.append(
                "terminal active block counts are not exactly one per block: "
                f"{active_counts.tolist()}"
            )
        for parent_id, parent in state.all_nodes.items():
            for child_id in parent.children:
                if child_id not in state.all_nodes:
                    errors.append(f"node {parent_id} references missing child {child_id}")
                    continue
                child = state.all_nodes[child_id]
                if not float(parent.time) > float(child.time):
                    errors.append(
                        f"edge violates parent > child time: parent={parent_id} "
                        f"child={child_id}"
                    )
        missing_active_partials = [
            lineage.node_id
            for lineage in state.active_lineages
            if lineage.partials is None
        ]
        if missing_active_partials:
            errors.append(f"active lineages missing partials: {missing_active_partials}")
        return errors

    def backtrack_to_step(self, step_index):
        step_index = int(step_index)
        if not 0 <= step_index < len(self.canonical_states):
            raise ValueError(
                f"step_index must be in [0, {len(self.canonical_states) - 1}], "
                f"got {step_index}"
            )
        return self.canonical_states[step_index]

    def canonical_events_touching_blocks(self, blocks):
        blocks = {int(block) for block in blocks}
        if not blocks:
            return []
        events_by_block = self._canonical_event_block_index()
        selected = {}
        for block in sorted(blocks):
            if not 0 <= block < self.num_blocks:
                continue
            for event in events_by_block[block]:
                selected[int(event["step"])] = event
        return [selected[step] for step in sorted(selected)]

    def _canonical_event_block_index(self):
        if self._canonical_events_by_block is not None:
            return self._canonical_events_by_block

        events_by_block = [[] for _ in range(int(self.num_blocks))]
        for event in self.canonical_action_trace:
            for start, end in canonical_event_material_segments(event):
                start = max(int(start), 0)
                end = min(int(end), int(self.num_blocks))
                if end <= start:
                    continue
                for block in range(start, end):
                    events_by_block[block].append(event)
        self._canonical_events_by_block = events_by_block
        return events_by_block

    def backtrack_bad_region(self, blocks, strategy="before_last_coalescence"):
        touching, target_step = self._backtrack_bad_region_plan(blocks, strategy)
        state = self.backtrack_to_step(target_step)
        return state, touching, target_step

    def _backtrack_bad_region_plan(self, blocks, strategy="before_last_coalescence"):
        if strategy not in BACKTRACK_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(BACKTRACK_STRATEGIES)}, got {strategy!r}"
            )
        blocks = tuple(sorted({int(block) for block in blocks}))
        if not blocks:
            raise ValueError("at least one block is required")
        touching = self.canonical_events_touching_blocks(blocks)
        if not touching:
            target_step = 0
        elif strategy == "before_last_touch":
            target_step = max(int(event["step"]) for event in touching) - 1
        elif strategy == "before_first_touch":
            target_step = min(int(event["step"]) for event in touching) - 1
        else:
            coalescence_steps = [
                int(event["step"])
                for event in touching
                if event["event_type"] == "coal"
            ]
            target_step = (
                max(coalescence_steps) - 1
                if coalescence_steps
                else max(int(event["step"]) for event in touching) - 1
            )
        target_step = max(0, int(target_step))
        return touching, target_step

    def variant_indices_for_block(self, block):
        row = self.block_table[int(block)]
        return np.flatnonzero(
            (self.vcf_positions >= float(row["left_bp"]))
            & (self.vcf_positions < float(row["right_bp"]))
        )

    def score_bad_regions(self):
        return score_bad_regions(self)

    def select_regions(self, top_k=None, block_groups=None, bp_intervals=None):
        rows = self.score_bad_regions()
        return select_regions(
            self,
            rows,
            top_k=top_k,
            block_groups=block_groups,
            bp_intervals=bp_intervals,
        )

    def build_refinement_contexts(
        self,
        regions,
        strategy="before_last_coalescence",
    ):
        plans = []
        for output_idx, region in enumerate(regions):
            touching, target_step = self._backtrack_bad_region_plan(
                region.blocks,
                strategy=strategy,
            )
            plans.append(
                {
                    "output_idx": output_idx,
                    "region": region,
                    "touching": touching,
                    "target_step": target_step,
                }
            )

        contexts = [None] * len(plans)
        for plan in sorted(plans, key=lambda item: item["target_step"]):
            state = self.backtrack_to_step(plan["target_step"])
            contexts[plan["output_idx"]] = self._make_refinement_context(
                plan["region"],
                state,
                plan["touching"],
                plan["target_step"],
                rollout_mode="segment",
                backtrack_offset=0,
                strategy_backtrack_step=plan["target_step"],
            )
        return contexts

    def build_terminal_refinement_contexts(
        self,
        regions,
        strategy="before_last_coalescence",
        terminal_backtrack_lengths=None,
    ):
        offsets = [0]
        offsets.extend(int(length) for length in (terminal_backtrack_lengths or ()))
        plans = []
        for region in regions:
            touching, strategy_step = self._backtrack_bad_region_plan(
                region.blocks,
                strategy=strategy,
            )
            seen_steps = set()
            for offset in offsets:
                if offset < 0:
                    raise ValueError("terminal backtrack offsets must be non-negative")
                target_step = max(0, int(strategy_step) - int(offset))
                if target_step in seen_steps:
                    continue
                seen_steps.add(target_step)
                plans.append(
                    {
                        "output_idx": len(plans),
                        "region": region,
                        "touching": touching,
                        "target_step": target_step,
                        "backtrack_offset": offset,
                        "strategy_step": strategy_step,
                    }
                )

        contexts = [None] * len(plans)
        for plan in sorted(plans, key=lambda item: item["target_step"]):
            state = self.backtrack_to_step(plan["target_step"])
            contexts[plan["output_idx"]] = self._make_refinement_context(
                plan["region"],
                state,
                plan["touching"],
                plan["target_step"],
                rollout_mode="terminal",
                backtrack_offset=plan["backtrack_offset"],
                strategy_backtrack_step=plan["strategy_step"],
            )
        return contexts

    def build_refinement_context_sets(
        self,
        regions,
        strategy="before_last_coalescence",
        terminal_backtrack_lengths=None,
    ):
        offsets = [0]
        offsets.extend(int(length) for length in (terminal_backtrack_lengths or ()))
        segment_plans = []
        terminal_plans = []

        for region_idx, region in enumerate(regions):
            touching, strategy_step = self._backtrack_bad_region_plan(
                region.blocks,
                strategy=strategy,
            )
            segment_plans.append(
                {
                    "kind": "segment",
                    "output_idx": region_idx,
                    "region": region,
                    "touching": touching,
                    "target_step": strategy_step,
                    "backtrack_offset": 0,
                    "strategy_step": strategy_step,
                }
            )

            seen_steps = set()
            for offset in offsets:
                if offset < 0:
                    raise ValueError("terminal backtrack offsets must be non-negative")
                target_step = max(0, int(strategy_step) - int(offset))
                if target_step in seen_steps:
                    continue
                seen_steps.add(target_step)
                terminal_plans.append(
                    {
                        "kind": "terminal",
                        "output_idx": len(terminal_plans),
                        "region": region,
                        "touching": touching,
                        "target_step": target_step,
                        "backtrack_offset": offset,
                        "strategy_step": strategy_step,
                    }
                )

        segment_contexts = [None] * len(segment_plans)
        terminal_contexts = [None] * len(terminal_plans)
        all_plans = segment_plans + terminal_plans
        for plan in sorted(all_plans, key=lambda item: item["target_step"]):
            state = self.backtrack_to_step(plan["target_step"])
            context = self._make_refinement_context(
                plan["region"],
                state,
                plan["touching"],
                plan["target_step"],
                rollout_mode=plan["kind"],
                backtrack_offset=plan["backtrack_offset"],
                strategy_backtrack_step=plan["strategy_step"],
            )
            if plan["kind"] == "segment":
                segment_contexts[plan["output_idx"]] = context
            else:
                terminal_contexts[plan["output_idx"]] = context

        return segment_contexts, terminal_contexts

    def _make_refinement_context(
        self,
        region,
        state,
        touching,
        target_step,
        rollout_mode,
        backtrack_offset,
        strategy_backtrack_step,
    ):
        missing_partials = [
            lineage.node_id
            for lineage in state.active_lineages
            if lineage.partials is None
        ]
        if missing_partials:
            raise ValueError(
                f"region {region.index} partial state has active lineages "
                f"without partials: {missing_partials}"
            )
        active_counts = self.env.get_active_counts(state)
        unfinished_blocks = {
            int(block)
            for block, count in enumerate(active_counts.tolist())
            if int(count) != 1
        }
        effective_blocks = tuple(
            sorted(set(region.blocks).union(unfinished_blocks))
        )
        return RefinementContext(
            region=region,
            partial_state=state,
            touching_events=touching,
            backtrack_step=target_step,
            target_blocks=tuple(region.blocks),
            effective_blocks=effective_blocks,
            rollout_mode=str(rollout_mode),
            backtrack_offset=int(backtrack_offset),
            strategy_backtrack_step=(
                int(strategy_backtrack_step)
                if strategy_backtrack_step is not None
                else None
            ),
        )

    def observed_variant_base(self, variant_idx, sample_node):
        hap_idx = self.sample_index_by_node[int(sample_node)]
        allele_idx = int(self.vcf_genotypes[hap_idx, int(variant_idx)])
        return VCF_INDEX_TO_BASE[allele_idx]

    def observed_variant_partial(self, variant_idx, sample_node):
        hap_idx = self.sample_index_by_node[int(sample_node)]
        return self.vcf_partials[hap_idx, int(variant_idx)].copy()


def build_refinement_source(env, trees_path, vcf_path, **kwargs):
    return RefinementSource(env, trees_path, vcf_path, **kwargs)


def estimate_time_delta_bin_width_from_trees(
    trees_path,
    population_size,
    time_bins,
    margin=1.05,
):
    try:
        import tskit
    except ImportError as exc:
        raise ImportError(
            "tskit is required to auto-calibrate time bins from .trees files."
        ) from exc

    time_bins = int(time_bins)
    if time_bins < 2:
        raise ValueError("time_bins must be at least 2 for time-bin calibration")
    population_size = float(population_size)
    if population_size <= 0.0:
        raise ValueError("population_size must be positive for time-bin calibration")
    margin = float(margin)
    if margin <= 0.0:
        raise ValueError("time-bin calibration margin must be positive")

    ts = tskit.load(str(trees_path))
    positive_times_generations = np.asarray(
        sorted({float(node.time) for node in ts.nodes() if float(node.time) > 0.0}),
        dtype=np.float64,
    )
    if positive_times_generations.size == 0:
        raise ValueError(
            f"{trees_path} has no positive node times to calibrate time bins"
        )

    local_times = positive_times_generations / (2.0 * population_size)
    event_times = np.concatenate(([0.0], local_times))
    deltas = np.diff(event_times)
    positive_deltas = deltas[deltas > 0.0]
    if positive_deltas.size == 0:
        raise ValueError(
            f"{trees_path} has no positive event-time deltas to calibrate time bins"
        )

    max_delta = float(np.max(positive_deltas))
    finite_bins = time_bins - 1
    width = max_delta * margin / float(finite_bins)
    tail_start = finite_bins * width
    return {
        "source_time_units": str(ts.time_units),
        "time_bins": int(time_bins),
        "finite_bins": int(finite_bins),
        "margin": float(margin),
        "population_size": float(population_size),
        "positive_event_time_count": int(local_times.size),
        "max_event_time_generations": float(np.max(positive_times_generations)),
        "max_event_time_t_over_2Ne": float(np.max(local_times)),
        "max_delta_t_over_2Ne": max_delta,
        "time_delta_bin_width": float(width),
        "tail_start_t_over_2Ne": float(tail_start),
    }


def score_bad_regions(source):
    pairwise_sample_pairs = [
        (source.samples[i], source.samples[j])
        for i in range(len(source.samples))
        for j in range(i + 1, len(source.samples))
    ]

    def local_tree_clades(tree):
        clades = set()
        full = frozenset(range(len(source.samples)))
        for node in tree.nodes():
            node = int(node)
            clade = frozenset(
                source.sample_index_by_node[int(sample)]
                for sample in tree.samples(node)
            )
            if 1 < len(clade) < len(full):
                clades.add(clade)
        return clades

    def clade_jaccard_distance(left_clades, right_clades):
        union = left_clades | right_clades
        if not union:
            return 0.0
        return 1.0 - (len(left_clades & right_clades) / len(union))

    def fitch_variant_score(tree, variant_idx):
        score = 0
        state_sets = {}
        for node in tree.nodes(order="postorder"):
            node = int(node)
            if tree.is_sample(node):
                state_sets[node] = {
                    source.observed_variant_base(variant_idx, node)
                }
                continue
            child_sets = [state_sets[int(child)] for child in tree.children(node)]
            if not child_sets:
                state_sets[node] = set("ACGT")
                continue
            intersection = set.intersection(*child_sets)
            if intersection:
                state_sets[node] = intersection
            else:
                state_sets[node] = set.union(*child_sets)
                score += 1
        return score

    def block_parsimony_metrics(block):
        variant_indices = source.variant_indices_for_block(block)
        tree = source.block_trees[block]
        parsimony_score = 0
        min_required = 0
        informative_variants = 0
        for variant_idx in variant_indices:
            observed = {
                source.observed_variant_base(variant_idx, sample)
                for sample in source.samples
            }
            if len(observed) <= 1:
                continue
            informative_variants += 1
            parsimony_score += fitch_variant_score(tree, variant_idx)
            min_required += len(observed) - 1
        homoplasy_excess = max(0, parsimony_score - min_required)
        width_kb = max(float(source.block_table[block]["width_bp"]) / 1000.0, 1e-12)
        return {
            "variants": int(len(variant_indices)),
            "informative_variants": int(informative_variants),
            "parsimony_score": int(parsimony_score),
            "homoplasy_excess": int(homoplasy_excess),
            "homoplasy_excess_per_kb": float(homoplasy_excess / width_kb),
        }

    def block_log_likelihood(block):
        variant_indices = source.variant_indices_for_block(block)
        if len(variant_indices) == 0:
            return np.nan, np.nan
        tree = source.block_trees[block]
        local_times = {
            int(node): float(source.ts.node(int(node)).time)
            / (2.0 * source.population_size)
            for node in tree.nodes()
        }

        @lru_cache(None)
        def variant_partial(node, variant_idx):
            node = int(node)
            variant_idx = int(variant_idx)
            if tree.is_sample(node):
                return source.observed_variant_partial(variant_idx, node)
            partial = np.ones(4, dtype=np.float64)
            for child in tree.children(node):
                child = int(child)
                child_partial = variant_partial(child, variant_idx)
                edge_time = local_times[node] - local_times[child]
                transition = jc69_transition(
                    edge_time,
                    source.population_size,
                    source.mutation_rate,
                )
                partial *= child_partial @ transition.T
                partial = normalize_row(partial)
            return normalize_row(partial)

        log_likelihood = 0.0
        for variant_idx in variant_indices:
            root_probs = []
            for root in tree.roots:
                partial = variant_partial(int(root), int(variant_idx))
                root_probs.append(max(float(np.sum(partial * 0.25)), 1e-300))
            variant_prob = float(np.prod(root_probs)) if root_probs else 1e-300
            log_likelihood += math.log(max(variant_prob, 1e-300))
        return log_likelihood, log_likelihood / max(len(variant_indices), 1)

    def block_pairwise_genetic_distances(block):
        variant_indices = source.variant_indices_for_block(block)
        if len(variant_indices) == 0:
            return None
        distances = []
        for sample_a, sample_b in pairwise_sample_pairs:
            idx_a = source.sample_index_by_node[sample_a]
            idx_b = source.sample_index_by_node[sample_b]
            calls_a = source.vcf_genotypes[idx_a, variant_indices]
            calls_b = source.vcf_genotypes[idx_b, variant_indices]
            distances.append(float(np.mean(calls_a != calls_b)))
        return np.asarray(distances, dtype=float)

    def block_pairwise_tree_distances(block):
        tree = source.block_trees[block]
        distances = []
        tmrcas = []
        for sample_a, sample_b in pairwise_sample_pairs:
            mrca = int(tree.mrca(sample_a, sample_b))
            tmrca = (
                float(source.ts.node(mrca).time)
                if mrca != source.tskit.NULL
                else np.nan
            )
            tmrcas.append(tmrca)
            if mrca == source.tskit.NULL:
                distances.append(np.nan)
                continue
            distance = (2.0 * tmrca) - float(
                source.ts.node(sample_a).time + source.ts.node(sample_b).time
            )
            distances.append(distance)
        return np.asarray(distances, dtype=float), np.asarray(tmrcas, dtype=float)

    def genotype_tree_residual(block):
        genetic = block_pairwise_genetic_distances(block)
        if genetic is None:
            return np.nan
        tree_distance, _ = block_pairwise_tree_distances(block)
        finite = np.isfinite(genetic) & np.isfinite(tree_distance)
        if finite.sum() < 3:
            return np.nan
        x = tree_distance[finite]
        y = genetic[finite]
        if np.allclose(x, x[0]):
            predicted = np.full_like(y, np.mean(y))
        else:
            slope, intercept = np.polyfit(x, y, 1)
            predicted = slope * x + intercept
        return float(np.sqrt(np.mean((y - predicted) ** 2)))

    def branch_length_metrics(block):
        tree = source.block_trees[block]
        lengths = []
        for node in tree.nodes():
            node = int(node)
            parent = tree.parent(node)
            if parent == source.tskit.NULL:
                continue
            lengths.append(
                float(source.ts.node(parent).time)
                - float(source.ts.node(node).time)
            )
        if not lengths:
            return {"max_branch_generations": np.nan, "mean_branch_generations": np.nan}
        return {
            "max_branch_generations": float(np.max(lengths)),
            "mean_branch_generations": float(np.mean(lengths)),
        }

    def tmrca_outlier_scores(tmrca_vectors):
        if not tmrca_vectors:
            return np.asarray([], dtype=float)
        matrix = np.vstack(tmrca_vectors)
        median = np.nanmedian(matrix, axis=0)
        scores = []
        for row in matrix:
            finite = np.isfinite(row) & np.isfinite(median)
            if finite.sum() == 0:
                scores.append(0.0)
            else:
                scores.append(float(np.nanmean(np.abs(row[finite] - median[finite]))))
        return np.asarray(scores, dtype=float)

    clades_by_block = [local_tree_clades(tree) for tree in source.block_trees]
    topology_instability = []
    for block in range(source.num_blocks):
        distances = []
        if block > 0:
            distances.append(
                clade_jaccard_distance(
                    clades_by_block[block - 1],
                    clades_by_block[block],
                )
            )
        if block + 1 < source.num_blocks:
            distances.append(
                clade_jaccard_distance(
                    clades_by_block[block],
                    clades_by_block[block + 1],
                )
            )
        topology_instability.append(float(max(distances)) if distances else 0.0)

    raw_rows = []
    tmrca_vectors = []
    for block in range(source.num_blocks):
        variant_indices = source.variant_indices_for_block(block)
        log_likelihood, log_likelihood_per_variant = block_log_likelihood(block)
        parsimony = block_parsimony_metrics(block)
        branch_metrics = branch_length_metrics(block)
        _tree_distance, tmrca_vector = block_pairwise_tree_distances(block)
        tmrca_vectors.append(tmrca_vector)
        touching_events = source.canonical_events_touching_blocks([block])
        width_bp = float(source.block_table[block]["width_bp"])
        internal_left = source.block_table[block]["left_bp"] > source.analysis_left
        internal_right = source.block_table[block]["right_bp"] < source.analysis_right
        breakpoint_density_per_kb = (
            int(internal_left) + int(internal_right)
        ) / max(width_bp / 1000.0, 1e-12)
        canonical_event_density_per_kb = len(touching_events) / max(
            width_bp / 1000.0,
            1e-12,
        )
        residual = genotype_tree_residual(block)
        raw_rows.append(
            {
                "block": int(block),
                "left_bp": float(source.block_table[block]["left_bp"]),
                "right_bp": float(source.block_table[block]["right_bp"]),
                "width_bp": float(width_bp),
                "variant_count": int(len(variant_indices)),
                "variant_positions": source.vcf_positions[variant_indices]
                .astype(int)
                .tolist(),
                "local_log_likelihood": float(log_likelihood),
                "neg_log_likelihood_per_variant": (
                    float(-log_likelihood_per_variant)
                    if np.isfinite(log_likelihood_per_variant)
                    else np.nan
                ),
                "informative_variants": parsimony["informative_variants"],
                "low_support_signal": 1.0 / max(parsimony["informative_variants"], 1),
                "parsimony_score": parsimony["parsimony_score"],
                "homoplasy_excess": parsimony["homoplasy_excess"],
                "homoplasy_excess_per_kb": parsimony["homoplasy_excess_per_kb"],
                "topology_instability": topology_instability[block],
                "breakpoint_density_per_kb": breakpoint_density_per_kb,
                "canonical_event_density_per_kb": canonical_event_density_per_kb,
                "genotype_tree_residual": residual,
                "max_branch_generations": branch_metrics["max_branch_generations"],
                "mean_branch_generations": branch_metrics["mean_branch_generations"],
                "long_branch_homoplasy_score": (
                    parsimony["homoplasy_excess_per_kb"]
                    * branch_metrics["max_branch_generations"]
                    if np.isfinite(parsimony["homoplasy_excess_per_kb"])
                    and np.isfinite(branch_metrics["max_branch_generations"])
                    else np.nan
                ),
                "canonical_touching_steps": [
                    int(event["step"]) for event in touching_events
                ],
            }
        )

    tmrca_outlier = tmrca_outlier_scores(tmrca_vectors)
    for row, score in zip(raw_rows, tmrca_outlier):
        row["tmrca_outlier_score"] = float(score)

    diagnostic_components = {
        "likelihood_z": clipped_positive_z(
            [row["neg_log_likelihood_per_variant"] for row in raw_rows]
        ),
        "homoplasy_z": clipped_positive_z(
            [row["homoplasy_excess_per_kb"] for row in raw_rows]
        ),
        "topology_z": clipped_positive_z(
            [row["topology_instability"] for row in raw_rows]
        ),
        "breakpoint_z": clipped_positive_z(
            [row["breakpoint_density_per_kb"] for row in raw_rows]
        ),
        "event_density_z": clipped_positive_z(
            [row["canonical_event_density_per_kb"] for row in raw_rows]
        ),
        "tmrca_z": clipped_positive_z(
            [row["tmrca_outlier_score"] for row in raw_rows]
        ),
        "residual_z": clipped_positive_z(
            [row["genotype_tree_residual"] for row in raw_rows]
        ),
        "long_branch_homoplasy_z": clipped_positive_z(
            [row["long_branch_homoplasy_score"] for row in raw_rows]
        ),
        "low_support_z": clipped_positive_z(
            [row["low_support_signal"] for row in raw_rows]
        ),
    }
    for idx, row in enumerate(raw_rows):
        for name, values in diagnostic_components.items():
            row[name] = float(values[idx])
        row["bad_region_score"] = float(
            sum(values[idx] for values in diagnostic_components.values())
        )
    return sorted(raw_rows, key=lambda row: row["bad_region_score"], reverse=True)


def select_regions(
    source,
    diagnostic_rows,
    top_k=None,
    block_groups=None,
    bp_intervals=None,
):
    score_by_block = {
        int(row["block"]): float(row["bad_region_score"])
        for row in diagnostic_rows
    }
    selected_groups = []
    if block_groups:
        selected_groups.extend(
            [list(group) for group in normalize_block_groups(block_groups)]
        )
    if bp_intervals:
        selected_groups.extend(
            [blocks_from_bp_interval(source, left, right) for left, right in bp_intervals]
        )
    if top_k is not None and int(top_k) > 0:
        top_blocks = [int(row["block"]) for row in diagnostic_rows[: int(top_k)]]
        selected_groups.extend(merge_contiguous_bad_blocks(source, top_blocks))
    if not selected_groups:
        selected_groups.extend(
            merge_contiguous_bad_blocks(
                source,
                [int(diagnostic_rows[0]["block"])],
            )
        )

    unique_groups = []
    seen = set()
    for group in selected_groups:
        blocks = tuple(sorted({int(block) for block in group}))
        if not blocks:
            continue
        if blocks in seen:
            continue
        seen.add(blocks)
        unique_groups.append(blocks)

    regions = []
    for idx, blocks in enumerate(unique_groups, start=1):
        left_bp = float(source.block_table[blocks[0]]["left_bp"])
        right_bp = float(source.block_table[blocks[-1]]["right_bp"])
        variant_positions = []
        for block in blocks:
            variant_positions.extend(
                source.vcf_positions[source.variant_indices_for_block(block)]
                .astype(int)
                .tolist()
            )
        scores = [score_by_block.get(block, 0.0) for block in blocks]
        regions.append(
            RefinementRegion(
                index=idx,
                blocks=tuple(blocks),
                left_bp=left_bp,
                right_bp=right_bp,
                max_bad_region_score=max(scores) if scores else 0.0,
                sum_bad_region_score=sum(scores),
                variant_positions=tuple(variant_positions),
            )
        )
    return regions


def build_refinement_contexts(
    source,
    top_k=None,
    block_groups=None,
    bp_intervals=None,
    strategy="before_last_coalescence",
):
    rows = source.score_bad_regions()
    regions = select_regions(
        source,
        rows,
        top_k=top_k,
        block_groups=block_groups,
        bp_intervals=bp_intervals,
    )
    return source.build_refinement_contexts(regions, strategy=strategy), rows


def build_refinement_context_sets(
    source,
    top_k=None,
    block_groups=None,
    bp_intervals=None,
    strategy="before_last_coalescence",
    terminal_backtrack_lengths=None,
):
    rows = source.score_bad_regions()
    regions = select_regions(
        source,
        rows,
        top_k=top_k,
        block_groups=block_groups,
        bp_intervals=bp_intervals,
    )
    segment_contexts, terminal_contexts = source.build_refinement_context_sets(
        regions,
        strategy=strategy,
        terminal_backtrack_lengths=terminal_backtrack_lengths,
    )
    return segment_contexts, terminal_contexts, rows


def canonical_segments(segments):
    cleaned = []
    for start, end in sorted((int(s), int(e)) for s, e in segments if int(e) > int(s)):
        if cleaned and start <= cleaned[-1][1]:
            cleaned[-1] = (cleaned[-1][0], max(cleaned[-1][1], end))
        else:
            cleaned.append((start, end))
    return tuple(cleaned)


def blocks_to_segments(blocks):
    blocks = sorted({int(block) for block in blocks})
    if not blocks:
        return tuple()
    segments = []
    start = prev = blocks[0]
    for block in blocks[1:]:
        if block == prev + 1:
            prev = block
        else:
            segments.append((start, prev + 1))
            start = prev = block
    segments.append((start, prev + 1))
    return tuple(segments)


def segments_to_blocks(segments):
    blocks = []
    for start, end in segments:
        blocks.extend(range(int(start), int(end)))
    return blocks


def normalize_row(row):
    row = np.asarray(row, dtype=np.float64)
    total = float(row.sum())
    if total > 0:
        return row / total
    return np.full(4, 0.25, dtype=np.float64)


def jc69_transition(edge_time_local, population_size, mutation_rate):
    branch_length = max(float(edge_time_local), 0.0) * (
        2.0 * float(population_size) * float(mutation_rate)
    )
    decay = math.exp(-4.0 * branch_length / 3.0)
    same = 0.25 + 0.75 * decay
    diff = 0.25 - 0.25 * decay
    matrix = np.full((4, 4), diff, dtype=np.float64)
    np.fill_diagonal(matrix, same)
    return matrix


def canonical_node_record(node_id, **kwargs):
    record = {
        "node_id": int(node_id),
        "synthetic": True,
        "original_tskit_node": None,
        "original_time_generations": None,
        "adjusted_time_generations": None,
        "event_type": None,
        "source": "synthetic",
    }
    record.update(kwargs)
    return record


def allocate_synthetic_node_id(counter):
    node_id = int(counter["next"])
    counter["next"] += 1
    return node_id


def active_index_by_node_id(state):
    return {
        int(lineage.node_id): idx
        for idx, lineage in enumerate(state.active_lineages)
    }


def canonical_event_material_segments(event):
    return tuple(tuple(pair) for pair in event.get("material_segments", ()))


def material_segments_touch_blocks(segments, blocks):
    if not blocks:
        return False
    ordered_blocks = sorted({int(block) for block in blocks})
    for start, end in segments:
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        for block in ordered_blocks:
            if block < start:
                continue
            if block >= end:
                break
            return True
    return False


def finite_robust_z(values):
    arr = np.asarray(values, dtype=float)
    z = np.zeros_like(arr, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return z
    center = float(np.nanmedian(arr[finite]))
    mad = float(np.nanmedian(np.abs(arr[finite] - center)))
    if mad > 0:
        z[finite] = (arr[finite] - center) / (1.4826 * mad)
    else:
        std = float(np.nanstd(arr[finite]))
        if std > 0:
            z[finite] = (arr[finite] - float(np.nanmean(arr[finite]))) / std
    z[~finite] = 0.0
    return z


def clipped_positive_z(values):
    return np.maximum(finite_robust_z(values), 0.0)


def blocks_from_bp_interval(source, left_bp, right_bp):
    left_bp = float(left_bp)
    right_bp = float(right_bp)
    if right_bp <= left_bp:
        raise ValueError("BP interval right endpoint must be greater than left endpoint")
    selected = []
    for row in source.block_table:
        if float(row["right_bp"]) > left_bp and float(row["left_bp"]) < right_bp:
            selected.append(int(row["block"]))
    return selected


def merge_contiguous_bad_blocks(source, blocks):
    ordered = sorted({int(block) for block in blocks})
    if not ordered:
        return []
    groups = [[ordered[0]]]
    for block in ordered[1:]:
        previous = groups[-1][-1]
        intervals_touch = math.isclose(
            float(source.block_table[previous]["right_bp"]),
            float(source.block_table[block]["left_bp"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        if block == previous + 1 and intervals_touch:
            groups[-1].append(block)
        else:
            groups.append([block])
    return groups


def normalize_block_groups(block_groups):
    if isinstance(block_groups, str):
        return parse_block_groups(block_groups)
    return [
        tuple(sorted({int(block) for block in group}))
        for group in block_groups
        if group
    ]


def parse_block_groups(text):
    if text is None or str(text).strip() == "":
        return []
    groups = []
    for group_text in str(text).split(";"):
        group_text = group_text.strip()
        if not group_text:
            continue
        blocks = []
        for token in group_text.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start = int(start_text)
                end = int(end_text)
                if end < start:
                    raise ValueError(f"invalid block range {token!r}")
                blocks.extend(range(start, end + 1))
            else:
                blocks.append(int(token))
        groups.append(tuple(sorted({int(block) for block in blocks})))
    return groups


def parse_bp_intervals(text):
    if text is None or str(text).strip() == "":
        return []
    intervals = []
    for interval_text in str(text).split(";"):
        interval_text = interval_text.strip()
        if not interval_text:
            continue
        if "-" not in interval_text:
            raise ValueError(f"invalid BP interval {interval_text!r}")
        left_text, right_text = interval_text.split("-", 1)
        intervals.append((float(left_text), float(right_text)))
    return intervals


def clone_start_state(state):
    cloned = state.clone(copy_partials=False)
    for attr in ("canonical_step_index",):
        if hasattr(state, attr):
            setattr(cloned, attr, getattr(state, attr))
    return cloned


def move_state_partials_to_device(state, device):
    device = torch.device(device)
    for lineage in state.all_nodes.values():
        if torch.is_tensor(lineage.partials):
            lineage.partials = lineage.partials.to(device=device)
            lineage.clear_runtime_caches()
    return state
