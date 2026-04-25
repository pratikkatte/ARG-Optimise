import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Sequence, Tuple
from ete3 import Tree
from torch import Tensor

from utils import SiteBackboneTree, ThreadChoice, _children_from_edge_index


# --- Local GNN (neighbor_index format: (num_nodes, k) with -1 padding) ---


def _gcn_node_degree_mask(edge_index: Tensor, x: Tensor) -> Tensor:
    deg = torch.sum(edge_index >= 0, dim=-1, keepdim=True, dtype=x.dtype) + 1.0
    return torch.clamp(deg, min=1.0)


class GCNConvLocal(nn.Module):
    """Single GCN layer matching vbpi-gnn neighbor-index format."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.transform = nn.Linear(self.in_channels, self.out_channels, bias=bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        node_degree = _gcn_node_degree_mask(edge_index, x)
        h = self.transform(x) / node_degree**0.5
        node_feature_padded = torch.cat((h, h.new_zeros(1, self.out_channels)), dim=0)
        neigh_feature = node_feature_padded[edge_index]
        node_and_neigh = torch.cat((neigh_feature, h.unsqueeze(1)), dim=1)
        return torch.sum(node_and_neigh, dim=1) / node_degree**0.5


class GNNStack(nn.Module):
    """Stack of GNN layers (default GCN) with ELU after each layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        bias: bool = True,
        gnn_type: str = "gcn",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        if gnn_type != "gcn":
            raise NotImplementedError(f"Only gcn is implemented locally; got {gnn_type}")
        self.gconvs = nn.ModuleList()
        self.gconvs.append(GCNConvLocal(self.in_channels, self.out_channels, bias=bias))
        for _ in range(self.num_layers - 1):
            self.gconvs.append(GCNConvLocal(self.out_channels, self.out_channels, bias=bias))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i in range(self.num_layers):
            x = self.gconvs[i](x, edge_index)
            x = F.elu(x)
        return x


def attachment_candidate_node_ids(site_tree: SiteBackboneTree) -> List[int]:
    """
    Nodes on which a leaf can attach, matching :func:`utils.enumerate_thread_choices` branches:
    every backbone branch child, plus the root (for the root-regrafting action set).
    """
    cands = set(site_tree.branch_children)
    cands.add(site_tree.root)
    return sorted(cands)


def _graph_postorder(root: int, children: Tuple[Tuple[int, ...], ...]) -> List[int]:
    order: List[int] = []

    def visit(u: int) -> None:
        for v in children[u]:
            visit(v)
        order.append(u)

    visit(root)
    return order


def _graph_preorder(root: int, children: Tuple[Tuple[int, ...], ...]) -> List[int]:
    order: List[int] = []

    def visit(u: int) -> None:
        order.append(u)
        for v in children[u]:
            visit(v)

    visit(root)
    return order
# from phylomodel import PHY

def namenum(tree, taxon, nodetosplitMap=None):
    taxon2idx = {}
    j = len(taxon)
    if nodetosplitMap:
        idx2split = ['']*(2*j-3)
    for i, name in enumerate(taxon):
        taxon2idx[name] = i
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            # assert type(node.name) is str, "The taxon name should be strings"
            if not isinstance(node.name, str):
                warnings.warn("The taxon names are not strings, please check if they are already integers!")
            else:
                node.name = taxon2idx[node.name]
                if nodetosplitMap:
                    idx2split[node.name] = nodetosplitMap[node]
        else:
            node.name, j = j, j+1
            if nodetosplitMap and not node.is_root():
                idx2split[node.name] = nodetosplitMap[node]
    
    if nodetosplitMap:
        return idx2split

class Policy(nn.Module):
    """
    Graph policy: encodes unrooted/ordered tree node features, runs a GNN, and scores
    possible attachment **branch** nodes (child endpoint of each edge + root).
    """

    def __init__(
        self,
        leaf_names: List[str],
        device: str,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.device = device
        self.leaf_names = leaf_names
        n_leaves = len(leaf_names)
        self.register_buffer("leaf_features", torch.eye(n_leaves, dtype=torch.float32))
        in_dim = n_leaves
        self.gnn = GNNStack(
            in_dim, hidden_dim, num_layers=num_layers, bias=True, gnn_type="gcn"
        )
        self.focal_proj = nn.Sequential(
            nn.Linear(n_leaves, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_feature_dim = 4 * hidden_dim + 2
        self.action_edge_head = nn.Sequential(
            nn.Linear(self.edge_feature_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )
        # self.ntips = ntips
        # self.phylo_model = PHY(data, taxa, pden, subModel, scale=scale, device=self.device)
        self.adversarial_loss = nn.BCEWithLogitsLoss()

    def node_embedding(self, tree):
        device = self.device
        for node in tree.traverse('postorder'):
            if node.is_leaf():
                node.c = 0
                node.d = self.leaf_features[node.name]
            else:
                child_c, child_d = 0., 0.
                for child in node.children:
                    child_c += child.c
                    child_d += child.d
                node.c = 1./(3. - child_c)
                node.d = node.c * child_d
    
        node_features, node_idx_list, edge_index = [], [], []            
        for node in tree.traverse('preorder'):
            neigh_idx_list = []
            if not node.is_root():
                node.d = node.c * node.up.d + node.d
                # parent_idx_list.append(node.up.name)
                neigh_idx_list.append(node.up.name)    
                if not node.is_leaf():
                    neigh_idx_list.extend([child.name for child in node.children])
                else:
                    neigh_idx_list.extend([-1, -1])
            else:
                neigh_idx_list.extend([child.name for child in node.children])
                while len(neigh_idx_list) < 3:
                    neigh_idx_list.append(-1)
                
            edge_index.append(neigh_idx_list)                
            node_features.append(node.d)
            node_idx_list.append(node.name)
            
        branch_idx_map = torch.sort(torch.tensor(node_idx_list, dtype=torch.long, device=device), dim=0, descending=False)[1]
        # parent_idxes = torch.LongTensor(parent_idx_list)
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
        # pdb.set_trace()
        return torch.index_select(torch.stack(node_features), 0, branch_idx_map), edge_index[branch_idx_map]

    def node_embedding_site_backbone(self, site_tree: SiteBackboneTree):
        """Same as ``node_embedding`` (ete3) but for :class:`SiteBackboneTree` (graph format)."""
        device = self.device
        n = site_tree.num_nodes
        L = len(self.leaf_names)
        nfeat = self.leaf_features.shape[0]
        children = _children_from_edge_index(site_tree.edge_index, n)
        parent = site_tree.parent_of_child
        node_sample = site_tree.node_sample_ids
        root = site_tree.root
        for u in range(n):
            nch = len(children[u])
            s = node_sample[u]
            if s >= 0:
                if nch != 0:
                    raise ValueError(f"SiteBackboneTree: leaf {u} must have 0 children, got {nch}")
            else:
                if nch != 2:
                    raise ValueError(
                        f"SiteBackboneTree: internal node {u} must have 2 children, got {nch}"
                    )
        j = L
        namenum_name: List[int] = [0] * n
        for u in _graph_postorder(root, children):
            s = node_sample[u]
            if s >= 0:
                namenum_name[u] = s
            else:
                namenum_name[u] = j
                j += 1
        c = torch.zeros(n, device=device)
        d = torch.zeros(n, nfeat, device=device, dtype=self.leaf_features.dtype)
        lf = self.leaf_features.to(device)
        for u in _graph_postorder(root, children):
            s = node_sample[u]
            if s >= 0:
                c[u] = 0.0
                d[u] = lf[s]
            else:
                chs = children[u]
                child_c = c[chs[0]] + c[chs[1]]
                child_d = d[chs[0]] + d[chs[1]]
                c[u] = 1.0 / (3.0 - child_c)
                d[u] = c[u] * child_d
        for u in _graph_preorder(root, children):
            if u == root:
                continue
            p = parent[u]
            d[u] = c[u] * d[p] + d[u]
        node_features: List[torch.Tensor] = []
        node_idx_list: List[int] = []
        edge_list: List[List[int]] = []
        for u in _graph_preorder(root, children):
            s = node_sample[u]
            if u != root:
                neigh: List[int] = [namenum_name[parent[u]]]
                if s >= 0:
                    neigh.extend((-1, -1))
                else:
                    for v in children[u]:
                        neigh.append(namenum_name[v])
            else:
                neigh = [namenum_name[v] for v in children[u]]
                while len(neigh) < 3:
                    neigh.append(-1)
            edge_list.append(neigh)
            node_features.append(d[u])
            node_idx_list.append(namenum_name[u])
        branch_idx_map = torch.sort(
            torch.tensor(node_idx_list, dtype=torch.long, device=device), dim=0, descending=False
        )[1]
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device)
        return torch.index_select(torch.stack(node_features, dim=0), 0, branch_idx_map), edge_index[
            branch_idx_map
        ]
    
    def _focal_leaf_feature(
        self,
        current_focal_leaf: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        leaf_feature = torch.zeros(
            len(self.leaf_names), device=device, dtype=dtype
        )
        leaf_feature[current_focal_leaf] = 1.0
        return leaf_feature

    def _action_choice_tensors(
        self,
        action_choices: Sequence[ThreadChoice],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        branch_child = torch.tensor(
            [choice.branch_child for choice in action_choices],
            dtype=torch.long,
            device=device,
        )
        time_idx = torch.tensor(
            [choice.time_idx for choice in action_choices],
            dtype=torch.long,
            device=device,
        )
        time_value = torch.tensor(
            [float(choice.time_value) for choice in action_choices],
            dtype=dtype,
            device=device,
        )
        is_root = torch.tensor(
            [1.0 if choice.is_root_branch else 0.0 for choice in action_choices],
            dtype=dtype,
            device=device,
        )
        return branch_child, time_idx, time_value, is_root

    def _normalize_action_times(
        self,
        time_value: Tensor,
        time_grid: Optional[Sequence[float]] = None,
    ) -> Tensor:
        if time_grid is not None and len(time_grid) > 0:
            tg = torch.tensor(time_grid, dtype=time_value.dtype, device=time_value.device)
            t_min, t_max = tg[0], tg[-1]
        else:
            t_min = torch.min(time_value)
            t_max = torch.max(time_value)
        return (time_value - t_min) / (t_max - t_min + 1e-8)

    def _build_action_edge_features(
        self,
        node_embeddings: Tensor,
        leaf_feature: Tensor,
        branch_child: Tensor,
        time_value: Tensor,
        is_root: Tensor,
        time_grid: Optional[Sequence[float]] = None,
    ) -> Tensor:
        h_focal = self.focal_proj(leaf_feature.unsqueeze(0)).squeeze(0)
        h_target = node_embeddings[branch_child]
        h_focal_b = h_focal.unsqueeze(0).expand_as(h_target)
        t_norm = self._normalize_action_times(
            time_value, time_grid=time_grid
        ).unsqueeze(-1)
        return torch.cat(
            [
                h_focal_b,
                h_target,
                (h_focal_b - h_target).abs(),
                h_focal_b * h_target,
                t_norm,
                is_root.unsqueeze(-1),
            ],
            dim=-1,
        )

    def forward(
        self,
        site_tree: SiteBackboneTree,
        current_focal_leaf: int,
        action_choices: Sequence[ThreadChoice],
        time_grid: Optional[Sequence[float]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Run the GNN and score valid attachment actions from edge-style action features.

        Args:
            site_tree: Local backbone tree in graph form.
            current_focal_leaf: Index in ``leaf_names`` of the leaf being threaded.
            action_choices: Valid ``ThreadChoice`` candidates aligned to output rows.
            time_grid: Optional global time grid for action-time normalization.

        Returns:
            - ``action_logits`` — ``(A,)`` logits for ``action_choices``
            - ``action_probs`` — ``(A,)`` softmax probabilities
            - ``edge_features`` — ``(A, 4*hidden_dim+2)`` action edge features
            - ``node_embeddings`` — ``(N, hidden_dim)`` node embeddings from GNN
            - ``leaf_feature`` — ``(num_leaves,)`` one-hot focal leaf indicator
        """

        if not (0 <= int(current_focal_leaf) < len(self.leaf_names)):
            raise ValueError(
                f"current_focal_leaf must be in [0, {len(self.leaf_names)}), got {current_focal_leaf!r}"
            )
        if len(action_choices) == 0:
            raise ValueError("action_choices must be non-empty")

        node_features, edge_index = self.node_embedding_site_backbone(site_tree)
        node_embeddings = self.gnn(node_features, edge_index)
        leaf_feature = self._focal_leaf_feature(
            current_focal_leaf,
            device=node_embeddings.device,
            dtype=node_embeddings.dtype,
        )
        branch_child, _time_idx, time_value, is_root = self._action_choice_tensors(
            action_choices,
            device=node_embeddings.device,
            dtype=node_embeddings.dtype,
        )
        if torch.any(branch_child < 0) or torch.any(branch_child >= node_embeddings.shape[0]):
            raise ValueError("branch_child indices in action_choices are out of range")

        edge_features = self._build_action_edge_features(
            node_embeddings=node_embeddings,
            leaf_feature=leaf_feature,
            branch_child=branch_child,
            time_value=time_value,
            is_root=is_root,
            time_grid=time_grid,
        )
        action_logits = self.action_edge_head(edge_features).squeeze(-1)
        action_probs = F.softmax(action_logits, dim=0)
        return action_logits, action_probs, edge_features, node_embeddings, leaf_feature
