from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Sequence, Any
import torch
import math
import json
import numpy as np

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

def get_max_actions(num_backbone_lineages: int, time_grid: Sequence[float]) -> int:
    """
    Calculates the absolute theoretical upper bound of mathematical actions 
    if no lineages ever coalesced until the final time point.
    """
    num_grid_points = sum(1 for t in time_grid if t > 0)
    return num_backbone_lineages * num_grid_points

def get_mathematically_valid_actions(backbone_edges: list[dict], time_grid: Sequence[float]) -> list[tuple]:
    """
    Iterates through the actual backbone_edges.
    Finds every valid intersection where a time point t from the time_grid falls strictly 
    within the edge's lifespan: start_time < t <= end_time.
    Returns a list of all valid (edge_id, t) tuples.
    """
    valid_actions = []
    for edge in backbone_edges:
        edge_id = edge["edge_id"]
        start_time = edge["start_time"]
        end_time = edge["end_time"]
        for t in time_grid:
            if start_time < t <= end_time:
                valid_actions.append((edge_id, t))
    return valid_actions

def get_distinct_topological_outcomes(mathematical_actions: list[tuple], backbone_edges: list[dict]) -> list[tuple]:
    """
    Takes the mathematical actions and collapses them based on topological equivalence.
    Logic: If two or more edges share the exact same end_time (e.g., t = 1.0) AND 
    merge into the exact same target_node_at_end, then regrafting the pruned lineage 
    onto *any* of those edges at t = 1.0 results in the exact same polytomy.
    
    Returns a list of distinct actions (states/choices).
    """
    edge_id_to_target = {e["edge_id"]: e["target_node_at_end"] for e in backbone_edges}
    edge_id_to_end_time = {e["edge_id"]: e["end_time"] for e in backbone_edges}
    
    seen_polytomies = set()
    distinct_actions = []
    
    for edge_id, t in mathematical_actions:
        end_time = edge_id_to_end_time[edge_id]
        target_node = edge_id_to_target[edge_id]
        
        if math.isclose(t, end_time) and target_node is not None:
            polytomy_signature = (t, target_node)
            if polytomy_signature not in seen_polytomies:
                seen_polytomies.add(polytomy_signature)
                distinct_actions.append((edge_id, t))
        else:
            distinct_actions.append((edge_id, t))
            
    return distinct_actions

def enumerate_thread_choices(
    site_tree: SiteBackboneTree,
    time_grid: Sequence[float],
) -> tuple[ThreadChoice, ...]:
    backbone_edges = []
    for branch_child in site_tree.branch_children:
        parent = site_tree.parent_of_child[branch_child]
        backbone_edges.append({
            "edge_id": branch_child,
            "start_time": float(site_tree.node_times[branch_child].item()),
            "end_time": float(site_tree.node_times[parent].item()),
            "target_node_at_end": parent
        })
        
    backbone_edges.append({
        "edge_id": site_tree.root,
        "start_time": float(site_tree.node_times[site_tree.root].item()),
        "end_time": float('inf'),
        "target_node_at_end": None
    })
    
    math_actions = get_mathematically_valid_actions(backbone_edges, list(time_grid))
    distinct_actions = get_distinct_topological_outcomes(math_actions, backbone_edges)
    distinct_set = set(distinct_actions)

    choices: list[ThreadChoice] = []
    for branch_child, branch_signature in zip(site_tree.branch_children, site_tree.branch_signatures):
        parent = site_tree.parent_of_child[branch_child]
        child_time = float(site_tree.node_times[branch_child].item())
        parent_time = float(site_tree.node_times[parent].item())
        for time_idx, time_value in enumerate(time_grid):
            if child_time < float(time_value) <= parent_time:
                if (branch_child, float(time_value)) in distinct_set:
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
            if (site_tree.root, float(time_value)) in distinct_set:
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


def _metadata_to_name(metadata: Any) -> Optional[str]:
    if metadata in (None, b"", ""):
        return None
    if isinstance(metadata, bytes):
        try:
            metadata = metadata.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return metadata if metadata else None
    if isinstance(metadata, dict):
        for key in ("name", "sample_id", "id"):
            value = metadata.get(key)
            if value not in (None, ""):
                return str(value)
    return None


def _sample_leaf_names(ts: Any, sample_nodes: Sequence[int]) -> list[str]:
    leaf_names: list[str] = []
    for sample_node in sample_nodes:
        node = ts.node(sample_node)
        name = None
        if node.individual != -1:
            name = _metadata_to_name(ts.individual(node.individual).metadata)
        if name is None:
            name = _metadata_to_name(node.metadata)
        leaf_names.append(name if name is not None else str(sample_node))
    return leaf_names


def _timed_tree_from_tskit_tree(
    tree: Any,
    sample_id_by_node: Dict[int, int],
) -> TimedTree:
    roots = list(tree.roots)
    if len(roots) != 1:
        raise ValueError(
            f"tskit tree interval {tree.interval} has {len(roots)} roots; expected exactly 1"
        )

    def build(node_id: int) -> TimedTree:
        children = list(tree.children(node_id))
        sample_id = sample_id_by_node.get(node_id)
        if sample_id is not None:
            if children:
                raise ValueError(
                    f"Sample node {node_id} has children in interval {tree.interval}; "
                    "only leaf sample nodes are supported"
                )
            return sample_id
        if len(children) != 2:
            raise ValueError(
                f"Internal node {node_id} has {len(children)} children in interval "
                f"{tree.interval}; expected a binary local tree"
            )
        left, right = children
        return ("n", float(tree.time(node_id)), build(left), build(right))

    return build(roots[0])


def _validate_binary_genotypes(geno: np.ndarray) -> None:
    if geno.ndim != 2:
        raise ValueError(f"Expected a 2D genotype matrix, got shape {geno.shape}")
    bad = np.setdiff1d(np.unique(geno), np.array([0, 1], dtype=geno.dtype))
    if bad.size:
        raise ValueError(
            "Only binary 0/1 tskit genotypes are supported; "
            f"found values {bad.tolist()}"
        )


def _validate_binary_variants(ts: Any) -> None:
    for variant in ts.variants():
        if variant.alleles != ("0", "1"):
            raise ValueError(
                "Only binary 0/1 tskit variants are supported; expected "
                f"allele labels ('0', '1'), but site {variant.site.id} has "
                f"alleles {variant.alleles}"
            )


def load_tskit_threading_inputs(
    trees_path: str,
    time_grid_size: int,
) -> tuple[torch.Tensor, tuple[dict, ...], list[str], list[int], tuple[float, ...]]:
    """Load a tskit ``.trees`` file into the existing threading-env inputs.

    The environment steps over tskit variant sites. Local tree genomic intervals
    are converted to site-index intervals covering the variant positions inside
    each tree interval.
    """
    if time_grid_size < 2:
        raise ValueError("time_grid_size must be at least 2")

    try:
        import tskit
    except ImportError as exc:
        raise ImportError("load_tskit_threading_inputs requires the tskit package") from exc

    ts = tskit.load(trees_path)
    if ts.num_sites == 0:
        raise ValueError("The tree sequence has no variant sites to use as env steps")

    sample_nodes = list(ts.samples())
    if not sample_nodes:
        raise ValueError("The tree sequence has no sample nodes")

    geno_np = ts.genotype_matrix().T
    _validate_binary_genotypes(geno_np)
    _validate_binary_variants(ts)
    geno = torch.as_tensor(geno_np, dtype=torch.long)

    node_times = np.asarray(ts.tables.nodes.time, dtype=float)
    if node_times.size == 0:
        raise ValueError("The tree sequence has no nodes")
    time_grid = tuple(
        float(t)
        for t in np.linspace(
            float(node_times.min()),
            float(node_times.max()),
            time_grid_size,
        )
    )

    sample_id_by_node = {
        node_id: sample_id for sample_id, node_id in enumerate(sample_nodes)
    }
    site_positions = np.asarray([site.position for site in ts.sites()], dtype=float)
    reference_full_trees: list[dict] = []

    for tree in ts.trees():
        left = float(tree.interval.left)
        right = float(tree.interval.right)
        start_site = int(np.searchsorted(site_positions, left, side="left"))
        end_site = int(np.searchsorted(site_positions, right, side="left"))
        if start_site == end_site:
            continue
        timed_tree = _timed_tree_from_tskit_tree(tree, sample_id_by_node)
        if reference_full_trees and reference_full_trees[-1]["tree"] == timed_tree:
            reference_full_trees[-1]["sites"] = (reference_full_trees[-1]["sites"][0], end_site)
        else:
            reference_full_trees.append({"sites": (start_site, end_site), "tree": timed_tree})

    if not reference_full_trees:
        raise ValueError("No local trees overlap variant sites")

    expected_start = 0
    for segment in reference_full_trees:
        start, end = segment["sites"]
        if start != expected_start:
            raise ValueError(
                "Converted tskit trees do not cover all variant-site indices; "
                f"expected segment to start at {expected_start}, got {start}"
            )
        expected_start = end
    if expected_start != ts.num_sites:
        raise ValueError(
            "Converted tskit trees do not cover all variant-site indices; "
            f"covered {expected_start} of {ts.num_sites}"
        )

    leaf_names = _sample_leaf_names(ts, sample_nodes)
    all_leaf_ids = list(range(len(sample_nodes)))
    return geno, tuple(reference_full_trees), leaf_names, all_leaf_ids, time_grid
