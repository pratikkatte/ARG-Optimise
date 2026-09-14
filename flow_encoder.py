"""Stable state features for the flow head, independent of policy updates."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class FrozenFlowEncoder(nn.Module):
    """Copy the initial policy encoder without sharing parameters or gradients.

    The flow head is trainable; this feature map is fixed and checkpointed.
    Keeping the map independent avoids changing flow predictions every time
    the action policy changes, and gives the exact gradient of this model.
    """

    def __init__(self, policy):
        super().__init__()
        self.seq_embedding = deepcopy(policy.seq_embedding)
        self.summary_token = nn.Parameter(policy.summary_token.detach().clone())
        self.encoder = deepcopy(policy.encoder)
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        # Dropout would make cached warm-up features disagree with training.
        return super().train(False)

    @torch.no_grad()
    def forward(self, features, counts):
        packed, offsets = features.tensor, features.row_offsets
        batch_size = len(offsets) - 1
        active = max(end - start for start, end in zip(offsets, offsets[1:]))
        device = self.summary_token.device
        valid = torch.arange(active, device=device)[None, :] < counts.to(device)[:, None]
        projected = self.seq_embedding(packed.reshape(packed.shape[0], -1).to(
            device=device, dtype=self.seq_embedding.weight.dtype))
        padded = self.seq_embedding.bias.expand(batch_size, active, -1).clone()
        padded[valid] = projected
        tokens = torch.cat((self.summary_token.expand(batch_size, -1, -1), padded), dim=1)
        return self.encoder(tokens, key_padding_mask=F.pad(~valid, (1, 0), value=False))[:, 0]
