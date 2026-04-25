"""Local GNN (neighbor_index format: (num_nodes, k) with -1 padding)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
