"""Small, shared PyTorch building blocks."""

import torch.nn as nn


def mlp(input_dim, hidden_dim, output_dim, *, layers=1, dropout=0.0):
    """Build a ReLU MLP; ``layers=0`` gives a single linear projection."""
    if layers < 0:
        raise ValueError(f"layers must be non-negative, got {layers}")
    if layers == 0:
        return nn.Sequential(nn.Linear(input_dim, output_dim))

    modules = []
    for in_dim in [input_dim] + [hidden_dim] * (layers - 1):
        modules.extend((nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)))
    modules.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*modules)


def transformer_encoder(dim, depth, heads, mlp_ratio=2.0, dropout=0.0,
                        attention_dropout=0.0):
    """Return PyTorch's pre-norm, batch-first Transformer encoder."""
    if dim % heads:
        raise ValueError(f"embedding_size ({dim}) must be divisible by transformer_heads ({heads})")
    layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=heads,
        dim_feedforward=int(dim * mlp_ratio),
        dropout=dropout,
        activation="gelu",
        layer_norm_eps=1e-6,
        batch_first=True,
        norm_first=True,
    )
    # TransformerEncoderLayer otherwise shares one dropout value between the
    # attention probabilities and residual/MLP paths.
    layer.self_attn.dropout = attention_dropout
    return nn.TransformerEncoder(layer, int(depth), norm=nn.LayerNorm(dim, eps=1e-6),
                                 enable_nested_tensor=False)

