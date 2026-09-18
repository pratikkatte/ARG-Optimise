"""Lineage Transformer blocks shared by the infinite-sites encoder."""
from torch import nn
from torch.nn import functional as F

class TransformerMLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        attention_dropout=0.0,
        projection_dropout=0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"embedding_size ({dim}) must be divisible by transformer_heads ({num_heads})"
            )
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x, key_padding_mask=None):
        batch_size, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # SDPA uses True for allowed keys; our padding mask uses True for padding.
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = ~key_padding_mask[:, None, None, :]
        dropout_p = self.attn_drop.p if self.training else 0.0
        if dropout_p == 1.0:
            # Fused CUDA SDPA can return NaNs at p=1; preserve zero gradients too.
            x = v * 0.0
        else:
            # Keep the projection dtype (FP32 in the policy) and auto-select the backend.
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                scale=self.scale,
            )

        x = x.transpose(1, 2).reshape(batch_size, tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=2.0,
        dropout=0.0,
        attention_dropout=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
            dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = TransformerMLP(
            dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout=dropout,
        )

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        mlp_ratio=2.0,
        dropout=0.0,
        attention_dropout=0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(int(depth))
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x, key_padding_mask=None):
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return self.norm(x)
