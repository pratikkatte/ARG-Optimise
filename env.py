from typing import Optional, Sequence, Any
import bisect
import random

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from rewards import ARGRewardMixin
from utils import (
    GraphSegment,
    MultiLeafState,
    ThreadChoice,
    ThreadPathState,
    ThreadingConfig,
    build_backbone_segments_from_graph_segments,
    compress_thread_path_to_segments,
    enumerate_thread_choices,
    graph_node_sample_ids,
    graph_parent_of_child,
    graph_segments_to_timed_tree_segments,
    graph_with_site,
    graphs_equivalent,
    thread_leaf_into_graph,
)


def _canonical_choice(choice: ThreadChoice) -> tuple[int, tuple[int, ...], int, bool]:
    return (
        choice.branch_child,
        choice.branch_signature,
        choice.time_idx,
        choice.is_root_branch,
    )


def _copy_graph_segment(segment: GraphSegment) -> GraphSegment:
    return GraphSegment(
        start=int(segment.start),
        end=int(segment.end),
        genomic_interval=(
            tuple(float(x) for x in segment.genomic_interval)
            if segment.genomic_interval is not None
            else None
        ),
        graph=segment.graph.clone(),
    )


class _LazySiteGraphSequence:
    def __init__(self, env: "ARGweaverThreadEnv"):
        self.env = env

    def __len__(self) -> int:
        return self.env.config.sequence_length

    def __getitem__(self, site_index: int) -> Data:
        if site_index < 0:
            site_index += self.env.config.sequence_length
        return self.env._site_graph_for_site(site_index)


class _LazySiteChoicesSequence:
    def __init__(self, env: "ARGweaverThreadEnv"):
        self.env = env

    def __len__(self) -> int:
        return self.env.config.sequence_length

    def __getitem__(self, site_index: int) -> tuple[ThreadChoice, ...]:
        if site_index < 0:
            site_index += self.env.config.sequence_length
        return self.env._choices_for_site(site_index)


class _LazyChoiceToIndexSequence:
    def __init__(self, env: "ARGweaverThreadEnv"):
        self.env = env

    def __len__(self) -> int:
        return self.env.config.sequence_length

    def __getitem__(self, site_index: int) -> dict[ThreadChoice, int]:
        if site_index < 0:
            site_index += self.env.config.sequence_length
        return self.env._choice_to_index_for_site(site_index)


class ARGweaverThreadEnv(ARGRewardMixin):
    def __init__(
        self,
        config: ThreadingConfig,
        focal_leaf: int,
        backbone_segments: Sequence[GraphSegment],
    ):
        self.config = config
        self.focal_leaf = int(focal_leaf)
        self.backbone_segments = tuple(
            sorted((_copy_graph_segment(seg) for seg in backbone_segments), key=lambda seg: seg.start)
        )
        self._segment_starts = tuple(seg.start for seg in self.backbone_segments)
        self._segment_ends = tuple(seg.end for seg in self.backbone_segments)
        self._validate_backbone_coverage()

        self._site_graph_template_cache: dict[int, Data] = {}
        self._choice_cache: dict[int, tuple[ThreadChoice, ...]] = {}
        self._choice_to_index_cache: dict[int, dict[ThreadChoice, int]] = {}
        self.site_graphs = _LazySiteGraphSequence(self)
        self.site_trees = self.site_graphs
        self.site_choices = _LazySiteChoicesSequence(self)
        self.choice_to_index = _LazyChoiceToIndexSequence(self)

    def _validate_backbone_coverage(self) -> None:
        expected = 0
        for segment in self.backbone_segments:
            if segment.start != expected:
                raise ValueError(
                    f"Backbone graph segments do not cover site {expected}; "
                    f"next segment starts at {segment.start}"
                )
            if segment.end <= segment.start:
                raise ValueError(f"Invalid empty graph segment {segment}")
            expected = segment.end
        if expected < self.config.sequence_length:
            raise ValueError(
                f"Backbone graph segments cover {expected} sites, "
                f"expected {self.config.sequence_length}"
            )

    def _segment_index_for_site(self, site_index: int) -> int:
        if not (0 <= site_index < self.config.sequence_length):
            raise IndexError(
                f"site_index {site_index} is out of bounds for sequence length "
                f"{self.config.sequence_length}"
            )
        segment_idx = bisect.bisect_right(self._segment_starts, site_index) - 1
        if segment_idx < 0 or site_index >= self._segment_ends[segment_idx]:
            raise ValueError(f"No backbone graph segment covers site {site_index}")
        return segment_idx

    def _site_graph_template_for_segment(self, segment_idx: int) -> Data:
        cached = self._site_graph_template_cache.get(segment_idx)
        if cached is None:
            cached = graph_with_site(self.backbone_segments[segment_idx].graph, -1)
            self._site_graph_template_cache[segment_idx] = cached
        return cached

    def _site_graph_for_site(self, site_index: int) -> Data:
        template = self._site_graph_template_for_segment(
            self._segment_index_for_site(site_index)
        )
        return graph_with_site(template, site_index)

    def _choices_for_site(self, site_index: int) -> tuple[ThreadChoice, ...]:
        segment_idx = self._segment_index_for_site(site_index)
        cached = self._choice_cache.get(segment_idx)
        if cached is None:
            cached = enumerate_thread_choices(
                self._site_graph_template_for_segment(segment_idx),
                self.config.time_grid,
            )
            self._choice_cache[segment_idx] = cached
        return cached

    def _choice_to_index_for_site(self, site_index: int) -> dict[ThreadChoice, int]:
        segment_idx = self._segment_index_for_site(site_index)
        cached = self._choice_to_index_cache.get(segment_idx)
        if cached is None:
            cached = {
                choice: idx for idx, choice in enumerate(self._choices_for_site(site_index))
            }
            self._choice_to_index_cache[segment_idx] = cached
        return cached

    @property
    def sequence_length(self):
        return self.config.sequence_length

    @property
    def geno(self):
        return self.config.geno

    @property
    def time_grid(self):
        return self.config.time_grid

    @property
    def mutation_rate(self):
        return self.config.mutation_rate

    @property
    def reward_temperature(self):
        return self.config.reward_temperature

    @property
    def recomb_rate(self):
        return self.config.recomb_rate

    def reset(self) -> ThreadPathState:
        return ThreadPathState(site_index=0, choices=tuple(), recomb_count=0)

    def is_terminal(self, st: ThreadPathState) -> bool:
        return st.site_index >= self.config.sequence_length

    def valid_actions(self, st: ThreadPathState) -> list[int]:
        if self.is_terminal(st):
            return []
        return list(range(len(self._choices_for_site(st.site_index))))

    def _site_tree_for_encoding(self, st: ThreadPathState) -> Data:
        if self.is_terminal(st):
            return self._site_graph_for_site(self.config.sequence_length - 1)
        return self._site_graph_for_site(st.site_index)

    def _prev_choice(self, st: ThreadPathState) -> Optional[ThreadChoice]:
        return st.choices[-1] if st.choices else None

    def encode(self, st: ThreadPathState, window_size: int = 5) -> Data:
        graph = self._site_tree_for_encoding(st)
        num_nodes = int(graph.num_nodes)
        num_grid_points = len(self.config.time_grid)
        parent = graph_parent_of_child(graph)
        node_sample_ids = graph_node_sample_ids(graph)

        end_times = torch.zeros(num_nodes, dtype=torch.float32)
        max_time = float(self.config.time_grid[-1])
        for node_id, parent_id in enumerate(parent):
            end_times[node_id] = graph.node_times[parent_id] if parent_id != -1 else max_time

        prev_branch = torch.zeros(num_nodes, dtype=torch.float32)
        prev_time_value = torch.zeros(num_nodes, dtype=torch.float32)
        prev_choice = self._prev_choice(st)
        if prev_choice is not None and 0 <= prev_choice.branch_child < num_nodes:
            prev_branch[prev_choice.branch_child] = 1.0
            prev_time_value[prev_choice.branch_child] = float(prev_choice.time_value)

        curr_site = min(st.site_index, self.config.sequence_length)
        window_end = min(curr_site + window_size, self.config.sequence_length)
        geno_slice = self.config.geno[:, curr_site:window_end].float()
        geno_window = F.pad(geno_slice, (0, window_size - geno_slice.shape[1]), value=-1.0)
        node_geno = torch.full((num_nodes, window_size), -1.0, dtype=torch.float32)
        for node_id, sample_id in enumerate(node_sample_ids):
            if sample_id >= 0:
                node_geno[node_id] = geno_window[sample_id]

        out = graph.clone()
        out.x = torch.cat(
            [
                graph.x.float(),
                torch.arange(num_nodes, dtype=torch.float32).unsqueeze(-1),
                torch.tensor(parent, dtype=torch.float32).unsqueeze(-1),
                end_times.unsqueeze(-1),
                prev_branch.unsqueeze(-1),
                prev_time_value.unsqueeze(-1),
                node_geno,
            ],
            dim=1,
        )
        out.genotype_window = geno_window
        out.valid_action_mask = torch.zeros((num_nodes, num_grid_points), dtype=torch.bool)
        if not self.is_terminal(st):
            for choice in self._choices_for_site(st.site_index):
                out.valid_action_mask[choice.branch_child, choice.time_idx] = True
        return out

    def describe_action(
        self,
        st: ThreadPathState,
        action_idx: int,
        leaf_names: Optional[Sequence[str]] = None,
    ) -> str:
        choice = self._choices_for_site(st.site_index)[action_idx]
        if st.choices:
            tag = "[recomb]" if _canonical_choice(choice) != _canonical_choice(st.choices[-1]) else "[stay]"
        else:
            tag = "[start]"

        if choice.is_root_branch:
            branch_label = "root"
        elif leaf_names:
            branch_label = "(" + ",".join(
                leaf_names[s] if s < len(leaf_names) else str(s)
                for s in choice.branch_signature
            ) + ")"
        else:
            branch_label = str(choice.branch_signature)

        return f"site {st.site_index} -> branch {branch_label} @ t{choice.time_idx}={choice.time_value:.2f} {tag}"

    def step(self, st: ThreadPathState, action_idx: int) -> tuple[ThreadPathState, float, bool]:
        if self.is_terminal(st):
            raise RuntimeError("Cannot step a terminal state")
        if action_idx not in self.valid_actions(st):
            raise ValueError(f"Invalid action index {action_idx} at site {st.site_index}")

        choice = self._choices_for_site(st.site_index)[action_idx]
        prev_choice = st.choices[-1] if st.choices else None
        local_log_reward = self.compute_log_local_reward(prev_choice, choice, st.site_index)
        recomb_count = st.recomb_count
        if st.choices and _canonical_choice(choice) != _canonical_choice(st.choices[-1]):
            recomb_count += 1

        next_state = ThreadPathState(
            site_index=st.site_index + 1,
            choices=st.choices + (choice,),
            recomb_count=recomb_count,
        )
        return next_state, local_log_reward, self.is_terminal(next_state)

    def reconstruct_local_graphs(self, path: Sequence[ThreadChoice]) -> tuple[GraphSegment, ...]:
        segments = compress_thread_path_to_segments(path)
        rendered: list[GraphSegment] = []
        for segment in segments:
            start = int(segment["start"])
            end = int(segment["end"])
            threaded = thread_leaf_into_graph(
                self._site_graph_for_site(start),
                self.focal_leaf,
                segment["choice"],
            )
            rendered.append(
                GraphSegment(
                    start=start,
                    end=end,
                    genomic_interval=self.config.genomic_interval_for_site_span(start, end),
                    graph=threaded,
                )
            )
        return tuple(rendered)

    def reconstruct_local_trees(self, path: Sequence[ThreadChoice]) -> tuple[dict[str, Any], ...]:
        return graph_segments_to_timed_tree_segments(self.reconstruct_local_graphs(path))

    def snapshot_state(self, st: ThreadPathState) -> tuple[dict[str, Any], ...]:
        if not st.choices:
            return tuple()
        return self.reconstruct_local_trees(st.choices)

    def generate_random_trajectory(
        self, seed: int | None = None
    ) -> tuple[ThreadPathState, float, list[int]]:
        if seed is not None:
            random.seed(seed)

        st = self.reset()
        actions = []
        total_reward = 0.0
        while not self.is_terminal(st):
            valid_acts = self.valid_actions(st)
            action = random.choice(valid_acts)
            actions.append(action)
            st, reward, _done = self.step(st, action)
            total_reward += reward

        return st, total_reward, actions


class MultiLeafThreadEnv:
    """Outer environment that cycles through leaves while storing PyG graph segments."""

    def __init__(
        self,
        config: ThreadingConfig,
        all_leaf_ids: Sequence[int],
        reference_graph_segments: Sequence[GraphSegment],
    ):
        self.config = config
        self.all_leaf_ids = tuple(int(lid) for lid in all_leaf_ids)
        self.num_leaves = len(self.all_leaf_ids)
        self.reference_graph_segments = tuple(_copy_graph_segment(seg) for seg in reference_graph_segments)
        self._inner_env: Optional[ARGweaverThreadEnv] = None

    @property
    def sequence_length(self):
        return self.config.sequence_length

    @property
    def geno(self):
        return self.config.geno

    @property
    def time_grid(self):
        return self.config.time_grid

    @property
    def mutation_rate(self):
        return self.config.mutation_rate

    @property
    def reward_temperature(self):
        return self.config.reward_temperature

    @property
    def recomb_rate(self):
        return self.config.recomb_rate

    def _build_inner_env(
        self, graph_segments: Sequence[GraphSegment], focal_leaf: int
    ) -> ARGweaverThreadEnv:
        backbone_segments = build_backbone_segments_from_graph_segments(
            graph_segments,
            focal_leaf,
        )
        return ARGweaverThreadEnv(
            self.config,
            focal_leaf=focal_leaf,
            backbone_segments=backbone_segments,
        )

    def _update_graph_segments(
        self,
        graph_segments: Sequence[GraphSegment],
        focal_leaf: int,
        thread_path: Sequence[ThreadChoice],
        inner_env: ARGweaverThreadEnv,
    ) -> tuple[GraphSegment, ...]:
        compressed = compress_thread_path_to_segments(thread_path)
        boundaries = {0, inner_env.config.sequence_length}
        for segment in graph_segments:
            boundaries.add(segment.start)
            boundaries.add(segment.end)
        for segment in compressed:
            boundaries.add(int(segment["start"]))
            boundaries.add(int(segment["end"]))

        updated: list[GraphSegment] = []
        for start, end in zip(sorted(boundaries)[:-1], sorted(boundaries)[1:]):
            if start >= inner_env.config.sequence_length:
                break
            choice = thread_path[start]
            threaded = thread_leaf_into_graph(
                inner_env.site_graphs[start],
                focal_leaf,
                choice,
            )
            interval = inner_env.config.genomic_interval_for_site_span(start, end)
            if updated and graphs_equivalent(updated[-1].graph, threaded):
                prev = updated[-1]
                prev_left = (
                    prev.genomic_interval[0]
                    if prev.genomic_interval is not None
                    else inner_env.config.genomic_interval_for_site_span(prev.start, start)[0]
                )
                updated[-1] = GraphSegment(
                    start=prev.start,
                    end=end,
                    genomic_interval=(prev_left, interval[1]),
                    graph=prev.graph,
                )
            else:
                updated.append(
                    GraphSegment(
                        start=start,
                        end=end,
                        genomic_interval=interval,
                        graph=threaded,
                    )
                )

        return tuple(updated)

    def reset(self) -> MultiLeafState:
        first_leaf = self.all_leaf_ids[0]
        self._inner_env = self._build_inner_env(
            self.reference_graph_segments,
            first_leaf,
        )
        return MultiLeafState(
            current_graph_segments=self.reference_graph_segments,
            leaves_threaded=(),
            current_focal_leaf=first_leaf,
            inner_state=self._inner_env.reset(),
        )

    def is_terminal(self, st: MultiLeafState) -> bool:
        return len(st.leaves_threaded) == self.num_leaves

    def valid_actions(self, st: MultiLeafState) -> list[int]:
        if self.is_terminal(st):
            return []
        assert self._inner_env is not None and st.inner_state is not None
        return self._inner_env.valid_actions(st.inner_state)

    def step(self, st: MultiLeafState, action_idx: int) -> tuple[MultiLeafState, float, bool]:
        if self.is_terminal(st):
            raise RuntimeError("Cannot step a terminal state")
        assert self._inner_env is not None and st.inner_state is not None

        inner_next, inner_reward, inner_done = self._inner_env.step(
            st.inner_state,
            action_idx,
        )
        if not inner_done:
            return (
                MultiLeafState(
                    current_graph_segments=st.current_graph_segments,
                    leaves_threaded=st.leaves_threaded,
                    current_focal_leaf=st.current_focal_leaf,
                    inner_state=inner_next,
                ),
                inner_reward,
                False,
            )

        new_graph_segments = self._update_graph_segments(
            st.current_graph_segments,
            st.current_focal_leaf,
            inner_next.choices,
            self._inner_env,
        )
        new_leaves_threaded = st.leaves_threaded + (st.current_focal_leaf,)

        if len(new_leaves_threaded) == self.num_leaves:
            return (
                MultiLeafState(
                    current_graph_segments=new_graph_segments,
                    leaves_threaded=new_leaves_threaded,
                    current_focal_leaf=None,
                    inner_state=None,
                ),
                inner_reward,
                True,
            )

        next_leaf = self.all_leaf_ids[len(new_leaves_threaded)]
        self._inner_env = self._build_inner_env(new_graph_segments, next_leaf)
        return (
            MultiLeafState(
                current_graph_segments=new_graph_segments,
                leaves_threaded=new_leaves_threaded,
                current_focal_leaf=next_leaf,
                inner_state=self._inner_env.reset(),
            ),
            inner_reward,
            False,
        )

    def encode(self, st: MultiLeafState, window_size: int = 5) -> Data:
        assert self._inner_env is not None and st.inner_state is not None
        return self._inner_env.encode(st.inner_state, window_size=window_size)

    def describe_action(
        self,
        st: MultiLeafState,
        action_idx: int,
        leaf_names: Optional[Sequence[str]] = None,
    ) -> str:
        assert self._inner_env is not None and st.inner_state is not None
        desc = self._inner_env.describe_action(st.inner_state, action_idx, leaf_names=leaf_names)
        focal = (
            leaf_names[st.current_focal_leaf]
            if leaf_names and st.current_focal_leaf is not None and st.current_focal_leaf < len(leaf_names)
            else st.current_focal_leaf
        )
        return f"[leaf {focal}] {desc}"

    def reconstruct_all_local_graphs(self, st: MultiLeafState) -> tuple[GraphSegment, ...]:
        return st.current_graph_segments

    def reconstruct_all_local_trees(self, st: MultiLeafState) -> tuple[dict[str, Any], ...]:
        return graph_segments_to_timed_tree_segments(st.current_graph_segments)

    def generate_random_trajectory(
        self, seed: int | None = None
    ) -> tuple[MultiLeafState, float, list[int]]:
        if seed is not None:
            random.seed(seed)

        st = self.reset()
        actions = []
        total_reward = 0.0
        while not self.is_terminal(st):
            valid_acts = self.valid_actions(st)
            action = random.choice(valid_acts)
            actions.append(action)
            st, reward, _done = self.step(st, action)
            total_reward += reward

        return st, total_reward, actions


class SingleLeafThreadEnv(MultiLeafThreadEnv):
    """Single-leaf GFlowNet environment backed by PyG graph segments."""

    def __init__(
        self,
        config: ThreadingConfig,
        all_leaf_ids: Sequence[int],
        reference_graph_segments: Sequence[GraphSegment],
    ):
        super().__init__(config, all_leaf_ids, reference_graph_segments)
        self.current_graph_segments = self.reference_graph_segments

    def reset(
        self,
        focal_leaf: Optional[int] = None,
        graph_segments: Optional[Sequence[GraphSegment]] = None,
    ) -> MultiLeafState:
        if focal_leaf is None:
            focal_leaf = self.all_leaf_ids[-1]
        focal_leaf = int(focal_leaf)
        if focal_leaf not in self.all_leaf_ids:
            raise ValueError(
                f"focal_leaf {focal_leaf!r} is not in all_leaf_ids {self.all_leaf_ids!r}"
            )

        if graph_segments is not None:
            self.current_graph_segments = tuple(_copy_graph_segment(seg) for seg in graph_segments)

        self._inner_env = self._build_inner_env(self.current_graph_segments, focal_leaf)
        return MultiLeafState(
            current_graph_segments=self.current_graph_segments,
            leaves_threaded=(),
            current_focal_leaf=focal_leaf,
            inner_state=self._inner_env.reset(),
        )

    def is_terminal(self, st: MultiLeafState) -> bool:
        return len(st.leaves_threaded) == 1

    def step(self, st: MultiLeafState, action_idx: int) -> tuple[MultiLeafState, float, bool]:
        if self.is_terminal(st):
            raise RuntimeError("Cannot step a terminal state")
        assert self._inner_env is not None and st.inner_state is not None
        if st.current_focal_leaf is None:
            raise RuntimeError("Single-leaf episode has no focal leaf")

        inner_next, inner_reward, inner_done = self._inner_env.step(
            st.inner_state,
            action_idx,
        )
        if not inner_done:
            return (
                MultiLeafState(
                    current_graph_segments=st.current_graph_segments,
                    leaves_threaded=st.leaves_threaded,
                    current_focal_leaf=st.current_focal_leaf,
                    inner_state=inner_next,
                ),
                inner_reward,
                False,
            )

        new_graph_segments = self._update_graph_segments(
            st.current_graph_segments,
            st.current_focal_leaf,
            inner_next.choices,
            self._inner_env,
        )
        self.current_graph_segments = new_graph_segments
        return (
            MultiLeafState(
                current_graph_segments=new_graph_segments,
                leaves_threaded=st.leaves_threaded + (st.current_focal_leaf,),
                current_focal_leaf=st.current_focal_leaf,
                inner_state=inner_next,
            ),
            inner_reward,
            True,
        )

    def generate_random_trajectory(
        self,
        focal_leaf: int | None = None,
        seed: int | None = None,
    ) -> tuple[MultiLeafState, float, list[int]]:
        if seed is not None:
            random.seed(seed)

        st = self.reset(focal_leaf=focal_leaf)
        actions = []
        total_reward = 0.0
        while not self.is_terminal(st):
            valid_acts = self.valid_actions(st)
            action = random.choice(valid_acts)
            actions.append(action)
            st, reward, _done = self.step(st, action)
            total_reward += reward

        return st, total_reward, actions
