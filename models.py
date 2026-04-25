import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Sequence, Tuple, Union
from torch import Tensor

from utils import (
    SiteBackboneTree,
    ThreadChoice,
    _children_from_edge_index,
    _graph_postorder,
    _graph_preorder,
)

from gnn_model import GNNStack


class Policy(nn.Module):
    """
    Graph policy: encodes unrooted/ordered tree node features, runs a GNN, and scores
    possible attachment **branch** nodes (child endpoint of each edge + root).
    """

    def __init__(
        self,
        leaf_names: List[str],
        device: str,
        time_grid: Sequence[float],
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.device = device
        self.leaf_names = leaf_names
        n_leaves = len(leaf_names)
        self.register_buffer("leaf_features", torch.eye(n_leaves, dtype=torch.float32))
        self.log_Z = nn.Parameter(torch.zeros(()))
        in_dim = n_leaves
        self.time_grid = time_grid
        self.gnn = GNNStack(
            in_dim, hidden_dim, num_layers=num_layers, bias=True, gnn_type="gcn"
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

    def node_embedding_site_backbone(
        self,
        site_tree: SiteBackboneTree,
        current_focal_leaf: Optional[int] = None,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, int]]:
        """Same as ``node_embedding`` (ete3) but for :class:`SiteBackboneTree` (graph format)."""
        device = self.device
        n = site_tree.num_nodes
        nfeat = self.leaf_features.shape[0]
        if current_focal_leaf is not None and not (
            0 <= int(current_focal_leaf) < nfeat
        ):
            raise ValueError(
                f"current_focal_leaf must be in [0, {nfeat}), got {current_focal_leaf!r}"
            )
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
        edge_list: List[List[int]] = []
        for u in range(n):
            s = node_sample[u]
            if u != root:
                neigh: List[int] = [parent[u]]
                if s >= 0:
                    neigh.extend((-1, -1))
                else:
                    for v in children[u]:
                        neigh.append(v)
            else:
                neigh = [v for v in children[u]]
                while len(neigh) < 3:
                    neigh.append(-1)
            edge_list.append(neigh)
            node_features.append(d[u])
        if current_focal_leaf is not None:
            focal_node_id = n
            node_features.append(lf[int(current_focal_leaf)])
            edge_list.append([-1, -1, -1])

        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device)
        stacked_node_features = torch.stack(node_features, dim=0)
        if current_focal_leaf is None:
            return stacked_node_features, edge_index
        return stacked_node_features, edge_index, focal_node_id
    
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
        h_focal: Tensor,
        branch_child: Tensor,
        time_value: Tensor,
        is_root: Tensor,
        time_grid: Optional[Sequence[float]] = None,
    ) -> Tensor:
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
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Run the GNN and score valid attachment actions from edge-style action features.

        Args:
            site_tree: Local backbone tree in graph form.
            current_focal_leaf: Index in ``leaf_names`` of the leaf being threaded.
            action_choices: Valid ``ThreadChoice`` candidates aligned to output rows.

        Returns:
            - ``action_logits`` — ``(A,)`` logits for ``action_choices``
            - ``action_probs`` — ``(A,)`` softmax probabilities
            - ``edge_features`` — ``(A, 4*hidden_dim+2)`` action edge features
            - ``node_embeddings`` — ``(N + 1, hidden_dim)`` node embeddings from GNN,
              including the appended isolated focal leaf as the final row
            - ``leaf_feature`` — ``(num_leaves,)`` one-hot focal leaf indicator
        """

        if not (0 <= int(current_focal_leaf) < len(self.leaf_names)):
            raise ValueError(
                f"current_focal_leaf must be in [0, {len(self.leaf_names)}), got {current_focal_leaf!r}"
            )
        if len(action_choices) == 0:
            raise ValueError("action_choices must be non-empty")

        node_features, edge_index, focal_node_idx = self.node_embedding_site_backbone(
            site_tree,
            current_focal_leaf=current_focal_leaf,
        )
        node_embeddings = self.gnn(node_features, edge_index)
        h_focal = node_embeddings[focal_node_idx]
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
        if torch.any(branch_child < 0) or torch.any(branch_child >= site_tree.num_nodes):
            raise ValueError("branch_child indices in action_choices are out of range")

        edge_features = self._build_action_edge_features(
            node_embeddings=node_embeddings,
            h_focal=h_focal,
            branch_child=branch_child,
            time_value=time_value,
            is_root=is_root,
            time_grid=self.time_grid,
        )

        action_logits = self.action_edge_head(edge_features).squeeze(-1)
        action_probs = F.softmax(action_logits, dim=0)
        return action_logits, action_probs, edge_features, node_embeddings, leaf_feature
