from utils import ThreadingConfig, ThreadChoice, ThreadPathState, MultiLeafState, SiteBackboneTree, BackboneSegment
from typing import Optional, Tuple, Sequence, Any, List, Dict
from utils import *
from utils import _canonical_choice, _children_from_edge_index
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

    def get_action_masks(self, st: ThreadPathState, target_node: Optional[int] = None) -> dict:
        if self.is_terminal(st):
            return {}

        site_tree = self.site_trees[st.site_index]
        if target_node is None:
            node_mask = torch.zeros(site_tree.num_nodes, dtype=torch.bool)
            for node in range(site_tree.num_nodes):
                target_time = float(site_tree.node_times[node])
                is_root = (node == site_tree.root)
                if is_root:
                    parent_time = float('inf')
                else:
                    parent = site_tree.parent_of_child[node]
                    parent_time = float(site_tree.node_times[parent])
                
                # Check if there is any valid time in time_grid
                for t in self.time_grid:
                    if float(t) > target_time and (is_root or float(t) <= parent_time):
                        node_mask[node] = True
                        break
            return {"target_node_mask": node_mask}
            
        time_mask = torch.zeros(len(self.time_grid), dtype=torch.bool)
        target_time = float(site_tree.node_times[target_node])
        is_root = (target_node == site_tree.root)
        
        if not is_root:
            parent = site_tree.parent_of_child[target_node]
            parent_time = float(site_tree.node_times[parent])
        else:
            parent_time = float('inf')
            
        for time_idx, t in enumerate(self.time_grid):
            if float(t) > target_time and (is_root or float(t) <= parent_time):
                time_mask[time_idx] = True
                    
        return {"time_mask": time_mask}
        
    def _site_tree_for_encoding(self, st: ThreadPathState) -> SiteBackboneTree:
        if self.is_terminal(st):
            return self.site_trees[-1]
        return self.site_trees[st.site_index]

    def _prev_choice(self, st: ThreadPathState) -> Optional[ThreadChoice]:
        return st.choices[-1] if st.choices else None

    def encode(self, st: ThreadPathState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Takes a local site tree and observed mutational states, and converts them into 
        tensors formatted for a custom self-attention mechanism.
        
        Args:
            site_tree: The local tree at a specific genomic site.
            site_observation: The mutational states (alleles) for the samples.
            
        Returns:
            X: (N, 4) tensor containing node features.
            D: (N, N) tensor containing path distances between nodes.
            A: (N, N) binary tensor where A[i, j] = 1 if i is an ancestor of j.
        """
        site_tree = self._site_tree_for_encoding(st)
        site_observation = self.geno[:, st.site_index]

        children = _children_from_edge_index(site_tree.edge_index, site_tree.num_nodes)
        
        # Post-Order Traversal & Feature Extraction
        post_order_nodes = []
        def postorder_traverse(node):
            for child in children[node]:
                postorder_traverse(child)
            post_order_nodes.append(node)
            
        postorder_traverse(site_tree.root)
        
        # Fitch algorithm to determine internal node states and mutations
        node_states = {}
        for node in post_order_nodes:
            sample_id = site_tree.node_sample_ids[node]
            if sample_id >= 0:
                # Leaf node: allele state
                val = float(site_observation[sample_id].item())
                node_states[node] = {val}
            else:
                # Internal node
                kids = children[node]
                if len(kids) == 0:
                    node_states[node] = {0.0}
                elif len(kids) == 1:
                    node_states[node] = node_states[kids[0]].copy()
                else:
                    left_states = node_states[kids[0]]
                    right_states = node_states[kids[1]]
                    intersect = left_states.intersection(right_states)
                    if intersect:
                        node_states[node] = intersect
                    else:
                        node_states[node] = left_states.union(right_states)
                        
        resolved_states = {}
        mut_above = {}
        
        def preorder_resolve(node, parent_state):
            if parent_state in node_states[node]:
                state = parent_state
            else:
                state = 0.0 if 0.0 in node_states[node] else list(node_states[node])[0]
                
            resolved_states[node] = state
            if parent_state is not None and state != parent_state:
                mut_above[node] = 1.0
            else:
                mut_above[node] = 0.0
                
            for child in children[node]:
                preorder_resolve(child, state)
                
        # Start pre-order from root
        root_state_choice = 0.0 if 0.0 in node_states[site_tree.root] else list(node_states[site_tree.root])[0]
        preorder_resolve(site_tree.root, root_state_choice)
        mut_above[site_tree.root] = 0.0

        # Build feature matrix X
        X_list = []
        for node in post_order_nodes:
            t_i = float(site_tree.node_times[node].item())
            
            parent = site_tree.parent_of_child[node]
            b_i = 0.0 if parent == -1 else float(site_tree.node_times[parent].item()) - t_i
            
            # Count of leaf descendants below this node
            n_i = float(len(site_tree.descendant_signatures[node]))
            
            # Mutational state: sample allele state for leaves, mutation on branch for internal nodes
            sample_id = site_tree.node_sample_ids[node]
            m_i = resolved_states[node] if sample_id >= 0 else mut_above[node]
                
            X_list.append([t_i, b_i, n_i, m_i])
            
        X = torch.tensor(X_list, dtype=torch.float32)
        
        # Relative Tree Positional Encodings
        N = len(post_order_nodes)
        node_to_idx = {node: idx for idx, node in enumerate(post_order_nodes)}
        
        # Build undirected adjacency list for shortest path distance computing
        adj = {i: [] for i in range(site_tree.num_nodes)}
        for node in range(site_tree.num_nodes):
            for child in children[node]:
                adj[node].append(child)
                adj[child].append(node)
                
        # Calculate D: Path Distance Matrix
        D = torch.zeros((N, N), dtype=torch.float32)
        for i, node in enumerate(post_order_nodes):
            dist = {node: 0}
            queue = [node]
            while queue:
                curr = queue.pop(0)
                d = dist[curr]
                for nxt in adj.get(curr, []):
                    if nxt not in dist:
                        dist[nxt] = d + 1
                        queue.append(nxt)
                        
            for j_node, d in dist.items():
                if j_node in node_to_idx:
                    D[i, node_to_idx[j_node]] = d
                    
        # Calculate A: Ancestry Matrix
        A = torch.zeros((N, N), dtype=torch.float32)
        for j_idx, node in enumerate(post_order_nodes):
            curr = site_tree.parent_of_child[node]
            while curr != -1:
                if curr in node_to_idx:
                    i_idx = node_to_idx[curr]
                    A[i_idx, j_idx] = 1.0
                curr = site_tree.parent_of_child[curr]
                
        return X, D, A

    def describe_action(self, st: ThreadPathState, action: tuple[int, int], leaf_names: Optional[Sequence[str]] = None) -> str:
        target_node, time_idx = action
        site_tree = self.site_trees[st.site_index]
        is_root_branch = (target_node == site_tree.root)
        branch_signature = site_tree.descendant_signatures[target_node]
        time_value = self.time_grid[time_idx]

        if st.choices:
            choice = ThreadChoice(
                site=st.site_index, time_idx=time_idx, time_value=time_value,
                branch_child=target_node, is_root_branch=is_root_branch,
                branch_signature=branch_signature
            )
            tag = "[recomb]" if _canonical_choice(choice) != _canonical_choice(st.choices[-1]) else "[stay]"
        else:
            tag = "[start]"
            
        if is_root_branch:
            branch_label = "root"
        elif leaf_names:
            branch_label = "(" + ",".join(
                leaf_names[s] if s < len(leaf_names) else str(s) for s in branch_signature
            ) + ")"
        else:
            branch_label = str(branch_signature)
            
        return f"site {st.site_index} -> branch {branch_label} @ t{time_idx}={time_value:.2f} {tag}"


    def step(self, st: ThreadPathState, action: tuple[int, int]) -> tuple[ThreadPathState, float, bool]:
        if self.is_terminal(st):
            raise RuntimeError("Cannot step a terminal state")

        target_node, time_idx = action
        site_tree = self.site_trees[st.site_index]
        
        choice = ThreadChoice(
            site=st.site_index,
            time_idx=time_idx,
            time_value=self.time_grid[time_idx],
            branch_child=target_node,
            is_root_branch=(target_node == site_tree.root),
            branch_signature=site_tree.descendant_signatures[target_node]
        )

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
    ) -> tuple[ThreadPathState, float, list[tuple[int, int]]]:
        """Generates a random trajectory by choosing uniform random valid actions."""
        if seed is not None:
            random.seed(seed)

        st = self.reset()
        actions = []
        while not self.is_terminal(st):
            node_mask = self.get_action_masks(st)["target_node_mask"]
            valid_nodes = torch.where(node_mask)[0].tolist()
            target_node = random.choice(valid_nodes)
            
            time_mask = self.get_action_masks(st, target_node)["time_mask"]
            valid_times = torch.where(time_mask)[0].tolist()
            time_idx = random.choice(valid_times)
            
            action = (target_node, time_idx)
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

    def get_action_masks(self, st: MultiLeafState, target_node: Optional[int] = None) -> dict:
        if self.is_terminal(st):
            return {}
        assert self._inner_env is not None and st.inner_state is not None
        return self._inner_env.get_action_masks(st.inner_state, target_node)

    def step(
        self, st: MultiLeafState, action: tuple[int, int]
    ) -> tuple[MultiLeafState, float, bool]:
        if self.is_terminal(st):
            raise RuntimeError("Cannot step a terminal state")
        assert self._inner_env is not None and st.inner_state is not None

        ## This does one site action for the current focal leaf.
        inner_next, inner_reward, inner_done = self._inner_env.step(
            st.inner_state, action
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

    def encode(self, st: ThreadPathState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Delegates encoding to the inner environment logic.
        """
        if self._inner_env is None:
            raise RuntimeError("Inner environment is not initialized")
        return self._inner_env.encode(st.inner_state)
        

    def describe_action(self, st: MultiLeafState, action: tuple[int, int], leaf_names: Optional[Sequence[str]] = None) -> str:
        assert self._inner_env is not None and st.inner_state is not None
        desc = self._inner_env.describe_action(st.inner_state, action, leaf_names=leaf_names)
        focal = leaf_names[st.current_focal_leaf] if leaf_names and st.current_focal_leaf < len(leaf_names) else st.current_focal_leaf
        return f"[leaf {focal}] {desc}"

    def reconstruct_all_local_trees(
        self, st: MultiLeafState
    ) -> Tuple[Dict[str, Any], ...]:
        """Return the full trees from the terminal state."""
        return st.current_full_trees

    def generate_random_trajectory(
        self, seed: int | None = None
    ) -> tuple[MultiLeafState, float, list[tuple[int, int]]]:
        """
        Generates a random trajectory wihtout model, threading all leaves by choosing random actions.
        """
        if seed is not None:
            random.seed(seed)

        st = self.reset()
        actions = []
        while not self.is_terminal(st):
            node_mask = self.get_action_masks(st)["target_node_mask"]
            valid_nodes = torch.where(node_mask)[0].tolist()
            target_node = random.choice(valid_nodes)
            
            time_mask = self.get_action_masks(st, target_node)["time_mask"]
            valid_times = torch.where(time_mask)[0].tolist()
            time_idx = random.choice(valid_times)
            
            action = (target_node, time_idx)
            actions.append(action)
            st, reward, done = self.step(st, action)

        return st, reward, actions

