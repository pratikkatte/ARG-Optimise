"""Stable state features for the flow head, independent of policy updates."""
from copy import deepcopy

import torch
from torch import nn
from policy.encoding import encode_packed_lineages


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
        encoded, _ = encode_packed_lineages(
            features, counts, self.seq_embedding, self.summary_token, self.encoder,
        )
        return encoded[:, 0]
