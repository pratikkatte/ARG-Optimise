"""PyTorch Geometric GNN layers for local ARG graphs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv, GraphConv


class GNNStack(nn.Module):
    """Stack of PyG message-passing layers with ELU activations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        bias: bool = True,
        gnn_type: str = "gcn",
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        conv_cls: type[nn.Module]
        if gnn_type == "gcn":
            conv_cls = GCNConv
        elif gnn_type == "graphconv":
            conv_cls = GraphConv
        else:
            raise NotImplementedError(f"Unsupported PyG layer type: {gnn_type}")

        self.gconvs = nn.ModuleList()
        self.gconvs.append(conv_cls(in_channels, out_channels, bias=bias))
        for _ in range(num_layers - 1):
            self.gconvs.append(conv_cls(out_channels, out_channels, bias=bias))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        for conv in self.gconvs:
            x = F.elu(conv(x, edge_index))
        return x
