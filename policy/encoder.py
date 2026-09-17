"""One trainable SNP/material encoder used by policy and state flow."""
import torch
from torch import nn
from .observations import LINEAGE_DIM, STATE_DIM
from .transformer import TransformerEncoder


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class InfiniteSitesEncoder(nn.Module):
    def __init__(self, sample_count, embedding_size=64, hidden_size=128,
                 transformer_depth=6, transformer_heads=4, transformer_mlp_ratio=2.0,
                 dropout=0.0, attention_dropout=0.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.snp_encoder = mlp(7+2*sample_count, hidden_size, embedding_size)
        self.material_encoder = mlp(4+sample_count, hidden_size, embedding_size)
        self.lineage_projection = mlp(4*embedding_size+LINEAGE_DIM, hidden_size, embedding_size)
        self.summary_token = nn.Parameter(torch.zeros(1, 1, embedding_size))
        nn.init.normal_(self.summary_token, std=.02)
        self.transformer = TransformerEncoder(embedding_size, transformer_depth, transformer_heads,
                    mlp_ratio=transformer_mlp_ratio, dropout=dropout, attention_dropout=attention_dropout)
        self.state_projection = mlp(embedding_size+STATE_DIM, hidden_size, embedding_size)

    @staticmethod
    def pool(tokens, lengths, weights=None):
        lengths = torch.tensor(lengths, device=tokens.device, dtype=torch.long)
        maximum = torch.segment_reduce(tokens, 'max', lengths=lengths)
        maximum = torch.where((lengths > 0)[:, None], maximum, 0.)
        if weights is None:
            total = torch.segment_reduce(tokens, 'sum', lengths=lengths)
            mean = total/lengths.clamp_min(1)[:, None]
        else:
            total = torch.segment_reduce(tokens*weights[:, None], 'sum', lengths=lengths)
            denominator = torch.segment_reduce(weights, 'sum', lengths=lengths)
            mean = total/denominator[:, None]
        return torch.cat((mean, maximum), dim=-1)

    def forward(self, observations):
        snps = self.pool(self.snp_encoder(observations.snps), observations.snp_lengths)
        material = self.pool(self.material_encoder(observations.intervals), observations.interval_lengths,
                             observations.intervals[:, 2])
        lineages = self.lineage_projection(torch.cat((snps, material, observations.lineage_scalars), -1))
        counts = observations.counts
        padded = lineages.new_zeros(len(counts), max(counts), self.embedding_size)
        valid = torch.zeros(len(counts), max(counts), dtype=torch.bool, device=lineages.device)
        for row, (start, end) in enumerate(zip(observations.state_offsets, observations.state_offsets[1:])):
            padded[row, :end-start] = lineages[start:end]
            valid[row, :end-start] = True
        tokens = torch.cat((self.summary_token.expand(len(counts), -1, -1), padded), 1)
        mask = torch.cat((valid.new_ones(len(counts), 1), valid), 1)
        encoded = self.transformer(tokens, key_padding_mask=~mask)
        summary = self.state_projection(torch.cat((encoded[:, 0], observations.state_scalars), -1))
        return encoded[:, 1:]*valid[:, :, None], summary
