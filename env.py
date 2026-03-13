from utils import ThreadingConfig, ThreadChoice, ThreadPathState, MultiLeafState, SiteBackboneTree, BackboneSegment
from typing import Optional, Tuple, Sequence, Any, List, Dict
from utils import *
from utils import _canonical_choice
import math
import random
import torch

def _canonical_choice(choice: ThreadChoice) -> tuple[int, int, tuple[int, ...]]:
    return choice.branch_child, choice.branch_signature

def _thread_leaf_into_site_tree_full(
    site_tree: SiteBackboneTree,
    focal_leaf: int,
    choice: ThreadChoice) -> dict:
    edges = [tuple(edge) for edge in site_tree.edge_index.t().tolist()]
    node_times = site_tree.node_times.tolist()
    node_sample_ids = list(site_tree.node_sample_ids)

    focal_node = len(node_times)
    node_times.append(0.0)
    node_sample_ids.append(focal_leaf)
    highlight_edges: list[tuple[int, int]] = []

    if choice.is_root_branch:
        new_root = len(node_times)
        node_times.append(float(choice.time_value))
        node_sample_ids.append(-1)
        edges.append((new_root, site_tree.root))
        edges.append((new_root, focal_node))
        highlight_edges.extend([(new_root, site_tree.root), (new_root, focal_node)])
        root = new_root
    else:
        parent = site_tree.parent_of_child[choice.branch_child]
        if parent == -1:
            raise ValueError("Non-root branch choice cannot point to the root without is_root_branch")

        new_internal = len(node_times)
        node_times.append(float(choice.time_value))
        node_sample_ids.append(-1)
        edges = [(u, v) for (u, v) in edges if not (u == parent and v == choice.branch_child)]
        edges.extend(
            [
                (parent, new_internal),
                (new_internal, choice.branch_child),
                (new_internal, focal_node),
            ]
        )
        highlight_edges.extend(
            [
                (parent, new_internal),
                (new_internal, choice.branch_child),
                (new_internal, focal_node),
            ]
        )
        root = site_tree.root

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return {
        "edge_index": edge_index,
        "num_nodes": len(node_times),
        "root": root,
        "node_times": torch.tensor(node_times, dtype=torch.float32),
        "node_sample_ids": tuple(node_sample_ids),
        "highlight_edges": tuple(highlight_edges),
        "highlight_samples": (focal_leaf,),
        "choice": choice,
    }

class ARGweaverThreadEnv:
    def __init__(
        self,
        config: ThreadingConfig,
        focal_leaf: int,
        backbone_segments,
        reference_full_trees=None,
    ):
        self.config = config
        self.focal_leaf = int(focal_leaf)
        self.backbone_segments = tuple(backbone_segments)
        self.reference_full_trees = tuple(reference_full_trees or ())

        self.site_trees = tuple(expand_backbone_segments(self.backbone_segments, self.config.sequence_length))
        self.site_choices = tuple(enumerate_thread_choices(site_tree, self.config.time_grid) for site_tree in self.site_trees)
        self.choice_to_index = tuple({choice: idx for idx, choice in enumerate(choices)} for choices in self.site_choices)

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
        return list(range(len(self.site_choices[st.site_index])))
        
    def _site_tree_for_encoding(self, st: ThreadPathState) -> SiteBackboneTree:
        if self.is_terminal(st):
            return self.site_trees[-1]
        return self.site_trees[st.site_index]

    def _prev_choice(self, st: ThreadPathState) -> Optional[ThreadChoice]:
        return st.choices[-1] if st.choices else None

    def encode(self):
        """
        """
        raise NotImplementedError("to encode data")

    def describe_action(self, st: ThreadPathState, action_idx: int, leaf_names: Optional[Sequence[str]] = None) -> str:
        choice = self.site_choices[st.site_index][action_idx]
        if st.choices:
            tag = "[recomb]" if _canonical_choice(choice) != _canonical_choice(st.choices[-1]) else "[stay]"
        else:
            tag = "[start]"
            
        if choice.is_root_branch:
            branch_label = "root"
        elif leaf_names:
            branch_label = "(" + ",".join(
                leaf_names[s] if s < len(leaf_names) else str(s) for s in choice.branch_signature
            ) + ")"
        else:
            branch_label = str(choice.branch_signature)
            
        return f"site {choice.site} -> branch {branch_label} @ t{choice.time_idx}={choice.time_value:.2f} {tag}"


    def step(self, st: ThreadPathState, action_idx: int) -> tuple[ThreadPathState, float, bool]:
        if self.is_terminal(st):
            raise RuntimeError("Cannot step a terminal state")
        if action_idx not in self.valid_actions(st):
            raise ValueError(f"Invalid action index {action_idx} at site {st.site_index}")

        choice = self.site_choices[st.site_index][action_idx]
        recomb_count = st.recomb_count
        if st.choices and _canonical_choice(choice) != _canonical_choice(st.choices[-1]):
            recomb_count += 1

        next_state = ThreadPathState(
            site_index=st.site_index + 1,
            choices=st.choices + (choice,),
            recomb_count=recomb_count,
        )
        if self.is_terminal(next_state):
            # reward = math.exp(self.config.reward_temperature)
            reward = 1.0
            return next_state, reward, True
        return next_state, 0.0, False

    def reconstruct_local_trees(self, path: Sequence[ThreadChoice]) -> tuple[dict, ...]:
        segments = compress_thread_path_to_segments(path)
        rendered = []
        for segment in segments:
            site_tree = self.site_trees[segment["start"]]
            threaded = _thread_leaf_into_site_tree_full(site_tree, self.focal_leaf, segment["choice"])
            threaded["start"] = segment["start"]
            threaded["end"] = segment["end"]
            rendered.append(threaded)
        return tuple(rendered)

    def snapshot_state(self, st: ThreadPathState) -> tuple[dict, ...]:
        if not st.choices:
            return tuple()
        return self.reconstruct_local_trees(st.choices)

    def generate_random_trajectory(
        self, seed: int | None = None
    ) -> tuple[ThreadPathState, float, list[int]]:
        """Generates a random trajectory by choosing uniform random valid actions."""
        if seed is not None:
            random.seed(seed)

        st = self.reset()
        actions = []
        while not self.is_terminal(st):
            valid_acts = self.valid_actions(st)
            action = random.choice(valid_acts)
            actions.append(action)
            st, reward, done = self.step(st, action)

        return st, reward, actions

class MultiLeafThreadEnv:
    """Outer environment that cycles through all leaf nodes.

    One episode = threading every leaf exactly once (in fixed order).
    Each leaf sub-episode delegates to an inner ``ARGweaverThreadEnv``.
    After each leaf is fully threaded, the full trees are updated with the
    result and the next leaf's backbone is built from the updated trees.
    """

    def __init__(
        self,
        config: ThreadingConfig,
        all_leaf_ids: Sequence[int],
        reference_full_trees: Sequence[dict],
    ):
        self.config = config
        self.all_leaf_ids = tuple(int(lid) for lid in all_leaf_ids)
        self.num_leaves = len(self.all_leaf_ids)
        self.reference_full_trees = tuple(
            {"sites": tuple(seg["sites"]), "tree": seg["tree"]}
            for seg in reference_full_trees
        )
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
        self, full_trees: Sequence[Dict[str, Any]], focal_leaf: int
    ) -> ARGweaverThreadEnv:
        backbone_segments = build_backbone_segments_from_reference(
            full_trees, focal_leaf
        )
        env = ARGweaverThreadEnv(
            self.config,
            focal_leaf=focal_leaf,
            backbone_segments=backbone_segments,
            reference_full_trees=full_trees,
        )
        return env

    def _update_full_trees(self,
        full_trees: Sequence[Dict[str, Any]],
        focal_leaf: int,
        thread_path: Sequence[ThreadChoice],
        inner_env: "ARGweaverThreadEnv",
    ) -> Tuple[Dict[str, Any], ...]:
        """Reconstruct full timed trees after threading *focal_leaf*."""
        segments = compress_thread_path_to_segments(thread_path)
        
        boundaries = set([0, inner_env.config.sequence_length])
        for t in full_trees:
            boundaries.add(t["sites"][0])
            boundaries.add(t["sites"][1])
        for s in segments:
            boundaries.add(s["start"])
            boundaries.add(s["end"])
            
        boundaries = sorted(list(boundaries))
        
        updated: list[Dict[str, Any]] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if start >= inner_env.config.sequence_length:
                break
            site_tree = inner_env.site_trees[start]
            choice = thread_path[start]
            threaded = _thread_leaf_into_site_tree_full(
                site_tree, focal_leaf, choice
            )
            new_tree = timed_tree_from_graph(
                threaded["edge_index"],
                threaded["num_nodes"],
                threaded["root"],
                threaded["node_times"],
                threaded["node_sample_ids"],
            )
            if updated and updated[-1]["tree"] == new_tree:
                updated[-1]["sites"] = (updated[-1]["sites"][0], end)
            else:
                updated.append({"sites": (start, end), "tree": new_tree})
                
        return tuple(updated)

    def reset(self) -> MultiLeafState:
        first_leaf = self.all_leaf_ids[0]
        self._inner_env = self._build_inner_env(
            self.reference_full_trees, first_leaf
        )
        inner_state = self._inner_env.reset()
        return MultiLeafState(
            current_full_trees=self.reference_full_trees,
            leaves_threaded=(),
            current_focal_leaf=first_leaf,
            inner_state=inner_state
        )

    def is_terminal(self, st: MultiLeafState) -> bool:
        return len(st.leaves_threaded) == self.num_leaves

    def valid_actions(self, st: MultiLeafState) -> list[int]:
        if self.is_terminal(st):
            return []
        assert self._inner_env is not None and st.inner_state is not None
        return self._inner_env.valid_actions(st.inner_state)

    def step(
        self, st: MultiLeafState, action_idx: int
    ) -> tuple[MultiLeafState, float, bool]:
        if self.is_terminal(st):
            raise RuntimeError("Cannot step a terminal state")
        assert self._inner_env is not None and st.inner_state is not None

        ## This does one site action for the current focal leaf.
        inner_next, inner_reward, inner_done = self._inner_env.step(
            st.inner_state, action_idx
        )

        
        if not inner_done:
            return (
                MultiLeafState(
                    current_full_trees=st.current_full_trees,
                    leaves_threaded=st.leaves_threaded,
                    current_focal_leaf=st.current_focal_leaf,
                    inner_state=inner_next,
                ),
                0.0,
                False,
            )

        new_full_trees = self._update_full_trees(
            st.current_full_trees,
            st.current_focal_leaf,
            inner_next.choices,
            self._inner_env,
        )

        new_leaves_threaded = st.leaves_threaded + (st.current_focal_leaf,)

        if len(new_leaves_threaded) == self.num_leaves:
            reward = 1.0
            return (
                MultiLeafState(
                    current_full_trees=new_full_trees,
                    leaves_threaded=new_leaves_threaded,
                    current_focal_leaf=None,
                    inner_state=None
                ),
                reward,
                True,
            )

        next_leaf_idx = len(new_leaves_threaded)
        next_leaf = self.all_leaf_ids[next_leaf_idx]
        self._inner_env = self._build_inner_env(new_full_trees, next_leaf)
        new_inner_state = self._inner_env.reset()

        return (
            MultiLeafState(
                current_full_trees=new_full_trees,
                leaves_threaded=new_leaves_threaded,
                current_focal_leaf=next_leaf,
                inner_state=new_inner_state,
            ),
            0.0,
            False,
        )

    def encode(self):
        """
        """
        raise NotImplementedError("This is outer env. To encode data")
        

    def describe_action(self, st: MultiLeafState, action_idx: int, leaf_names: Optional[Sequence[str]] = None) -> str:
        assert self._inner_env is not None and st.inner_state is not None
        desc = self._inner_env.describe_action(st.inner_state, action_idx, leaf_names=leaf_names)
        focal = leaf_names[st.current_focal_leaf] if leaf_names and st.current_focal_leaf < len(leaf_names) else st.current_focal_leaf
        return f"[leaf {focal}] {desc}"

    def reconstruct_all_local_trees(
        self, st: MultiLeafState
    ) -> Tuple[Dict[str, Any], ...]:
        """Return the full trees from the terminal state."""
        return st.current_full_trees

    def generate_random_trajectory(
        self, seed: int | None = None
    ) -> tuple[MultiLeafState, float, list[int]]:
        """
        Generates a random trajectory wihtout model, threading all leaves by choosing random actions.
        """
        if seed is not None:
            random.seed(seed)

        st = self.reset()
        actions = []
        while not self.is_terminal(st):
            valid_acts = self.valid_actions(st)
            action = random.choice(valid_acts)
            actions.append(action)
            st, reward, done = self.step(st, action)

        return st, reward, actions

