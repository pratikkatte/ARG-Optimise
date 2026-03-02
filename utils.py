from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Sequence, Any
import torch
import math

TimedTree = Any

@dataclass(frozen=True)
class ThreadChoice:
    site: int
    branch_child: int
    branch_signature: Tuple[int, ...]
    time_idx: int
    time_value: float
    is_root_branch: bool


@dataclass(frozen=True)
class ThreadPathState:
    site_index: int
    choices: Tuple[ThreadChoice, ...]
    recomb_count: int

@dataclass(frozen=True)
class MultiLeafState:
    current_full_trees: Tuple[Dict[str, Any], ...]
    leaves_threaded: Tuple[int, ...]
    current_focal_leaf: Optional[int]
    inner_state: Optional[ThreadPathState]
    accumulated_log_score: float

@dataclass(frozen=True)
class ThreadingConfig:
    geno: torch.Tensor
    time_grid: Tuple[float, ...]
    mutation_rate: float
    recomb_rate: float
    reward_temperature: float
    sequence_length: int

    @classmethod
    def from_raw(cls, geno, time_grid, mutation_rate, recomb_rate, reward_temperature=0.2):
        geno_t = torch.as_tensor(geno, dtype=torch.long)
        return cls(
            geno=geno_t,
            time_grid=tuple(float(t) for t in time_grid),
            mutation_rate=float(mutation_rate),
            recomb_rate=float(recomb_rate),
            reward_temperature=float(reward_temperature),
            sequence_length=int(geno_t.shape[1]),
        )

@dataclass(frozen=True)
class BackboneSegment:
    start: int
    end: int
    edge_index: torch.Tensor
    num_nodes: int
    root: int
    node_times: torch.Tensor
    leaf_ids: Tuple[int, ...]
    node_sample_ids: Tuple[int, ...]


@dataclass(frozen=True)
class SiteBackboneTree:
    site: int
    edge_index: torch.Tensor
    num_nodes: int
    root: int
    node_times: torch.Tensor
    branch_children: Tuple[int, ...]
    branch_signatures: Tuple[Tuple[int, ...], ...]
    parent_of_child: Tuple[int, ...]
    node_sample_ids: Tuple[int, ...]
    descendant_signatures: Tuple[Tuple[int, ...], ...]


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

def _choice_prior_weights(curr_choices: Sequence[ThreadChoice]) -> torch.Tensor:
    if len(curr_choices) == 0:
        raise ValueError("Need at least one current-site choice")
    time_counts: dict[int, int] = {}
    for choice in curr_choices:
        time_counts[choice.time_idx] = time_counts.get(choice.time_idx, 0) + 1

    weights = []
    for choice in curr_choices:
        weights.append(math.exp(-choice.time_value) / time_counts[choice.time_idx])
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    return weight_tensor / weight_tensor.sum()

def timed_tree_from_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
    root: int,
    node_times: torch.Tensor,
    node_sample_ids: Sequence[int],
) -> TimedTree:
    children = _children_from_edge_index(edge_index, num_nodes)

    def build(node_id: int) -> TimedTree:
        sample_id = node_sample_ids[node_id]
        if sample_id >= 0:
            return sample_id
        kids = children[node_id]
        if len(kids) != 2:
            raise ValueError(
                f"Internal node {node_id} has {len(kids)} children, expected 2"
            )
        left = build(kids[0])
        right = build(kids[1])
        return ("n", float(node_times[node_id].item()), left, right)

    return build(root)

def _binary_site_log_likelihood(
    edge_index: torch.Tensor,
    num_nodes: int,
    root: int,
    node_times: torch.Tensor,
    node_sample_ids: Sequence[int],
    site_observation: torch.Tensor,
    theta: float,
) -> float:
    children = _children_from_edge_index(edge_index, num_nodes)
    site_values = site_observation.tolist()

    def postorder(node: int) -> torch.Tensor:
        sample_id = node_sample_ids[node]
        if sample_id >= 0:
            state = int(site_values[sample_id])
            out = torch.zeros(2, dtype=torch.float32)
            out[state] = 1.0
            return out

        node_like = torch.ones(2, dtype=torch.float32)
        for child in children[node]:
            branch_length = float(node_times[node].item() - node_times[child].item())
            exp_term = math.exp(-2.0 * theta * max(branch_length, 1e-6))
            p_same = 0.5 + 0.5 * exp_term
            p_diff = 0.5 - 0.5 * exp_term
            child_like = postorder(child)
            contrib = torch.empty(2, dtype=torch.float32)
            contrib[0] = p_same * child_like[0] + p_diff * child_like[1]
            contrib[1] = p_diff * child_like[0] + p_same * child_like[1]
            node_like = node_like * contrib
        return node_like

    root_like = postorder(root)
    likelihood = 0.5 * float(root_like[0].item()) + 0.5 * float(root_like[1].item())
    return math.log(max(likelihood, 1e-12))

def compress_thread_path_to_segments(
    thread_path: Sequence[ThreadChoice],
) -> tuple[dict, ...]:
    if not thread_path:
        return tuple()

    segments: list[dict] = []
    start = 0
    current = thread_path[0]
    current_key = _canonical_choice(current)

    for idx in range(1, len(thread_path)):
        next_key = _canonical_choice(thread_path[idx])
        if next_key != current_key:
            segments.append({"start": start, "end": idx, "choice": current})
            start = idx
            current = thread_path[idx]
            current_key = next_key
    segments.append({"start": start, "end": len(thread_path), "choice": current})
    return tuple(segments)


def score_thread_path(env: "ARGweaverThreadEnv", thread_path: Sequence[ThreadChoice]) -> float:
    if len(thread_path) != env.sequence_length:
        raise ValueError("Thread path length must equal the environment sequence length")

    first_idx = env.choice_to_index[0][thread_path[0]]
    total = float(env.initial_logps[first_idx].item()) + float(env.emission_logps[0][first_idx].item())
    prev_idx = first_idx
    for site in range(1, env.sequence_length):
        curr_idx = env.choice_to_index[site][thread_path[site]]
        total += float(env.transition_logps[site - 1][prev_idx, curr_idx].item())
        total += float(env.emission_logps[site][curr_idx].item())
        prev_idx = curr_idx
    return total

def initial_log_probs(site_choices: Sequence[ThreadChoice]) -> torch.Tensor:
    if len(site_choices) == 0:
        raise ValueError("initial_log_probs requires at least one valid choice")
    return torch.full((len(site_choices),), -math.log(len(site_choices)), dtype=torch.float32)

def emission_log_probs(
    site_tree: SiteBackboneTree,
    choices: Sequence[ThreadChoice],
    focal_leaf: int,
    site_observation: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    emission_values = []
    for choice in choices:
        threaded = _thread_leaf_into_site_tree_full(site_tree, focal_leaf, choice)
        emission_values.append(
            _binary_site_log_likelihood(
                threaded["edge_index"],
                threaded["num_nodes"],
                threaded["root"],
                threaded["node_times"],
                threaded["node_sample_ids"],
                site_observation,
                theta,
            )
        )
    return torch.tensor(emission_values, dtype=torch.float32)

def _canonical_choice(choice: ThreadChoice) -> tuple[int, int, tuple[int, ...]]:
    return choice.site, choice.branch_child, choice.branch_signature

def transition_log_probs(
    prev_choices: Sequence[ThreadChoice],
    curr_choices: Sequence[ThreadChoice],
    rho: float,
    time_grid: Sequence[float],
    site_distance: int = 1,
) -> torch.Tensor:
    if len(prev_choices) == 0 or len(curr_choices) == 0:
        raise ValueError("Transition matrices need non-empty previous and current choices")

    recomb_mass = 1.0 - math.exp(-rho * site_distance)
    stay_mass = 1.0 - recomb_mass
    curr_prior = _choice_prior_weights(curr_choices)

    matrix = torch.empty((len(prev_choices), len(curr_choices)), dtype=torch.float32)
    for prev_idx, prev_choice in enumerate(prev_choices):
        prev_key = _canonical_choice(prev_choice)
        for curr_idx, curr_choice in enumerate(curr_choices):
            prob = recomb_mass * float(curr_prior[curr_idx].item())
            if prev_key == _canonical_choice(curr_choice):
                prob += stay_mass
            matrix[prev_idx, curr_idx] = math.log(max(prob, 1e-12))
    return matrix

def _parent_from_edge_index(edge_index: torch.Tensor, num_nodes: int) -> tuple[int, ...]:
    parent = [-1] * num_nodes
    if edge_index.numel() > 0:
        for u, v in edge_index.t().tolist():
            parent[v] = u
    return tuple(parent)

def _children_from_edge_index(edge_index: torch.Tensor, num_nodes: int) -> tuple[tuple[int, ...], ...]:
    children: list[list[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() > 0:
        for u, v in edge_index.t().tolist():
            children[u].append(v)
    return tuple(tuple(child) for child in children)


def unthread_leaf_from_timed_tree(timed_tree: TimedTree, focal_leaf: int) -> TimedTree:
    def remove(node: TimedTree) -> Optional[TimedTree]:
        if isinstance(node, int):
            return None if node == focal_leaf else node

        label, time_value, left, right = node
        left_out = remove(left)
        right_out = remove(right)
        if left_out is None and right_out is None:
            return None
        if left_out is None:
            return right_out
        if right_out is None:
            return left_out
        return (label, time_value, left_out, right_out)

    out = remove(timed_tree)
    if out is None:
        raise ValueError("Unthreading removed the entire tree")
    return out

def build_backbone_segments_from_reference(
    reference_full_trees: Sequence[dict],
    focal_leaf: int,
) -> list[BackboneSegment]:
    segments: list[BackboneSegment] = []
    for segment in reference_full_trees:
        start, end = segment["sites"]
        backbone_tree = unthread_leaf_from_timed_tree(segment["tree"], focal_leaf)
        edge_index, num_nodes, root, node_times, node_sample_ids = _timed_tree_to_graph_full(backbone_tree)
        leaf_ids = tuple(sorted(sample_id for sample_id in node_sample_ids if sample_id >= 0))
        segments.append(
            BackboneSegment(
                start=start,
                end=end,
                edge_index=edge_index,
                num_nodes=num_nodes,
                root=root,
                node_times=node_times,
                leaf_ids=leaf_ids,
                node_sample_ids=node_sample_ids,
            )
        )
    return segments


def _timed_tree_to_graph_full(
    timed_tree: TimedTree,
) -> tuple[torch.Tensor, int, int, torch.Tensor, Tuple[int, ...]]:
    node_times: list[float] = []
    node_sample_ids: list[int] = []
    edges: list[tuple[int, int]] = []

    def allocate_node(time_value: float, sample_id: int) -> int:
        node_id = len(node_times)
        node_times.append(float(time_value))
        node_sample_ids.append(sample_id)
        return node_id

    def visit(node: TimedTree) -> int:
        if isinstance(node, int):
            return allocate_node(0.0, node)

        label, time_value, left, right = node
        if label != "n":
            raise ValueError(f"Unexpected timed-tree node label: {label}")

        left_id = visit(left)
        right_id = visit(right)
        node_id = allocate_node(float(time_value), -1)
        edges.append((node_id, left_id))
        edges.append((node_id, right_id))
        return node_id

    root = visit(timed_tree)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return (
        edge_index,
        len(node_times),
        root,
        torch.tensor(node_times, dtype=torch.float32),
        tuple(node_sample_ids),
    )

def expand_backbone_segments(
    backbone_segments: Sequence[BackboneSegment],
    sequence_length: int,
) -> list[SiteBackboneTree]:
    site_trees: list[Optional[SiteBackboneTree]] = [None] * sequence_length
    for segment in backbone_segments:
        parent_of_child = _parent_from_edge_index(segment.edge_index, segment.num_nodes)
        children = _children_from_edge_index(segment.edge_index, segment.num_nodes)

        def descendants(node: int) -> tuple[int, ...]:
            sample_id = segment.node_sample_ids[node]
            if sample_id >= 0:
                return (sample_id,)
            leaf_ids: list[int] = []
            for child in children[node]:
                leaf_ids.extend(descendants(child))
            return tuple(sorted(leaf_ids))

        descendant_signatures = tuple(descendants(node) for node in range(segment.num_nodes))
        branch_children = tuple(node for node in range(segment.num_nodes) if parent_of_child[node] != -1)
        branch_signatures = tuple(descendant_signatures[node] for node in branch_children)

        site_tree = SiteBackboneTree(
            site=-1,
            edge_index=segment.edge_index,
            num_nodes=segment.num_nodes,
            root=segment.root,
            node_times=segment.node_times,
            branch_children=branch_children,
            branch_signatures=branch_signatures,
            parent_of_child=parent_of_child,
            node_sample_ids=segment.node_sample_ids,
            descendant_signatures=descendant_signatures,
        )
        for site in range(segment.start, min(segment.end, sequence_length)):
            site_trees[site] = SiteBackboneTree(
                site=site,
                edge_index=site_tree.edge_index,
                num_nodes=site_tree.num_nodes,
                root=site_tree.root,
                node_times=site_tree.node_times,
                branch_children=site_tree.branch_children,
                branch_signatures=site_tree.branch_signatures,
                parent_of_child=site_tree.parent_of_child,
                node_sample_ids=site_tree.node_sample_ids,
                descendant_signatures=site_tree.descendant_signatures,
            )

    missing = [idx for idx, site_tree in enumerate(site_trees) if site_tree is None]
    if missing:
        raise ValueError(f"Backbone segments do not cover sites: {missing}")
    return [site_tree for site_tree in site_trees if site_tree is not None]

def enumerate_thread_choices(
    site_tree: SiteBackboneTree,
    time_grid: Sequence[float],
) -> tuple[ThreadChoice, ...]:
    choices: list[ThreadChoice] = []
    for branch_child, branch_signature in zip(site_tree.branch_children, site_tree.branch_signatures):
        parent = site_tree.parent_of_child[branch_child]
        child_time = float(site_tree.node_times[branch_child].item())
        parent_time = float(site_tree.node_times[parent].item())
        for time_idx, time_value in enumerate(time_grid):
            if child_time < float(time_value) <= parent_time:
                choices.append(
                    ThreadChoice(
                        site=site_tree.site,
                        branch_child=branch_child,
                        branch_signature=branch_signature,
                        time_idx=time_idx,
                        time_value=float(time_value),
                        is_root_branch=False,
                    )
                )

    root_signature = site_tree.descendant_signatures[site_tree.root]
    root_time = float(site_tree.node_times[site_tree.root].item())
    for time_idx, time_value in enumerate(time_grid):
        if float(time_value) > root_time:
            choices.append(
                ThreadChoice(
                    site=site_tree.site,
                    branch_child=site_tree.root,
                    branch_signature=root_signature,
                    time_idx=time_idx,
                    time_value=float(time_value),
                    is_root_branch=True,
                )
            )
    return tuple(choices)

