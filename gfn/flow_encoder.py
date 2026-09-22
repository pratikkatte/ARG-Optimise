"""A permanently fixed copy of the initial infinite-sites representation."""
from copy import deepcopy
import torch
from torch import nn


class FrozenFlowEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        # Copy all parameters and buffers without another random initialization.
        self.encoder = deepcopy(encoder)
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        # Parent generator.train() must never change this branch's behavior.
        return super().train(False)

    @torch.no_grad()
    def pool_lineages(self, snps, snp_lengths, intervals, interval_lengths):
        return self.encoder.pool_lineages(snps, snp_lengths, intervals, interval_lengths)

    @torch.no_grad()
    def forward(self, observations, *, pooled_embeddings=None):
        return self.encoder(observations, pooled_embeddings=pooled_embeddings)
