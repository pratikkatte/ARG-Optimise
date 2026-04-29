import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Sequence, Tuple, Union
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from gnn_model import GNNStack
from utils import ThreadChoice


class Policy(nn.Module):
    """PyG graph policy for scoring branch/time threading actions."""

    def __init__(
        self,
        leaf_names: List[str],
        device: Union[str, torch.device],
        time_grid: Sequence[float],
        hidden_dim: int = 64,
        num_layers: int = 2,
        gnn_type: str = "gcn",
    ) -> None:
        super().__init__()
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"Policy requested device {requested_device}, but CUDA is not available."
            )
        self.leaf_names = list(leaf_names)
        self.time_grid = tuple(float(t) for t in time_grid)
        self.input_dim = len(self.leaf_names) + 2
        self.register_buffer("leaf_features", torch.eye(len(self.leaf_names), dtype=torch.float32))
        self.log_Z = nn.Parameter(torch.zeros(()))
        self.gnn = GNNStack(
            self.input_dim,
            hidden_dim,
            num_layers=num_layers,
            bias=True,
            gnn_type=gnn_type,
        )
        self.edge_feature_dim = 4 * hidden_dim + 2
        self.action_edge_head = nn.Sequential(
            nn.Linear(self.edge_feature_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.to(requested_device)

    @property
    def device(self) -> torch.device:
        return self.leaf_features.device

    def _validate_graph_x(self, graph: Data) -> Tensor:
        if graph.x is None:
            raise ValueError("PyG graph is missing x node features")
        x = graph.x.float()
        if x.dim() != 2:
            raise ValueError(f"graph.x must be 2D, got shape {tuple(x.shape)}")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"graph.x feature width {x.shape[1]} does not match policy input_dim "
                f"{self.input_dim}; rebuild graph segments with the same leaf_names."
            )
        return x

    def node_embedding_site_backbone(
        self,
        site_graph: Data,
        current_focal_leaf: Optional[int] = None,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, int]]:
        device = self.device
        x = self._validate_graph_x(site_graph).to(device)
        edge_index = getattr(site_graph, "message_edge_index", None)
        if edge_index is None:
            edge_index = to_undirected(site_graph.edge_index, num_nodes=int(site_graph.num_nodes))
        edge_index = edge_index.to(device=device, dtype=torch.long)

        if current_focal_leaf is None:
            return x, edge_index

        current_focal_leaf = int(current_focal_leaf)
        if not (0 <= current_focal_leaf < len(self.leaf_names)):
            raise ValueError(
                f"current_focal_leaf must be in [0, {len(self.leaf_names)}), got {current_focal_leaf!r}"
            )
        focal_x = x.new_zeros((1, self.input_dim))
        focal_x[0, current_focal_leaf] = 1.0
        focal_x[0, len(self.leaf_names) + 1] = 1.0
        focal_node_id = x.shape[0]
        return torch.cat([x, focal_x], dim=0), edge_index, focal_node_id

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
        site_graph: Data,
        current_focal_leaf: int,
        action_choices: Sequence[ThreadChoice],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if len(action_choices) == 0:
            raise ValueError("action_choices must be non-empty")

        node_features, edge_index, focal_node_idx = self.node_embedding_site_backbone(
            site_graph,
            current_focal_leaf=current_focal_leaf,
        )
        node_embeddings = self.gnn(node_features, edge_index)
        h_focal = node_embeddings[focal_node_idx]
        leaf_feature = self._focal_leaf_feature(
            int(current_focal_leaf),
            device=node_embeddings.device,
            dtype=node_embeddings.dtype,
        )
        branch_child, _time_idx, time_value, is_root = self._action_choice_tensors(
            action_choices,
            device=node_embeddings.device,
            dtype=node_embeddings.dtype,
        )
        if torch.any(branch_child < 0) or torch.any(branch_child >= int(site_graph.num_nodes)):
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
