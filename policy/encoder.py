"""SNP/material encoding and branch-local pooling for infinite-sites states."""
from dataclasses import dataclass
import torch
from torch import nn
from .observations import LINEAGE_DIM, STATE_DIM
from .transformer import TransformerEncoder


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


@dataclass(frozen=True)
class _PoolPlan:
    sources: tuple
    keys: tuple
    positions: dict
    missing: tuple


class PooledLineageCache:
    """Autograd-connected pools for one fixed-model rollout/backward graph.

    Retain only the preceding call's lookup. Surviving immutable sources reuse
    their pools; new parents are encoded afresh. The caller must discard this
    cache before another backward graph or parameter update. It is deliberately
    not module/checkpoint state and must not be shared between rollouts or encoders.
    """
    def __init__(self):
        self.clear()

    def clear(self):
        self.positions = {}
        self.sources = ()  # Strong refs prevent source-id reuse.
        self.pooled = None

    @staticmethod
    def _select_rows(values, lengths, rows):
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        indices = [j for i in rows for j in range(offsets[i], offsets[i+1])]
        index = torch.tensor(indices, dtype=torch.long, device=values.device)
        return values.index_select(0, index), tuple(lengths[i] for i in rows)

    def prepare(self, lineages):
        """Find new static rows before observation assembly/device transfer."""
        sources = tuple((node.messages, node.snp_indices) for node in lineages)
        if any(messages is None or indices is None for messages, indices in sources):
            raise ValueError('Active lineage is missing infinite-sites messages')
        # Messages/indices are immutable in the environment. Replacements and
        # different descendant material must miss, including zero-SNP lineages.
        keys = tuple((id(messages), id(indices), node.descendants.segments)
                if not messages.flags.writeable and not indices.flags.writeable else object()
                for node, (messages, indices) in zip(lineages, sources))
        positions = dict(self.positions)
        start = 0 if self.pooled is None else len(self.pooled)
        missing = []
        for i, key in enumerate(keys):
            if key not in positions:
                positions[key] = start + len(missing)
                missing.append(i)
        return _PoolPlan(sources, keys, positions, tuple(missing))

    def get(self, encoder, observations, lineages, *, plan=None):
        plan = self.prepare(lineages) if plan is None else plan
        sources, keys, positions, missing = plan.sources, plan.keys, plan.positions, plan.missing
        if missing:
            snps, snp_lengths = self._select_rows(observations.snps, observations.snp_lengths, missing)
            intervals, interval_lengths = self._select_rows(
                observations.intervals, observations.interval_lengths, missing)
            new = encoder.pool_lineages(snps, snp_lengths, intervals, interval_lengths)
            bank = new if self.pooled is None else torch.cat((self.pooled, new), 0)
        else:
            bank = self.pooled
        index = torch.tensor([positions[key] for key in keys], dtype=torch.long, device=bank.device)
        pooled = bank.index_select(0, index)
        self.positions = {key: i for i, key in enumerate(keys)}
        self.sources, self.pooled = sources, pooled
        return pooled


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

    def pool_lineages(self, snps, snp_lengths, intervals, interval_lengths):
        snps = self.pool(self.snp_encoder(snps), snp_lengths)
        material = self.pool(self.material_encoder(intervals), interval_lengths, intervals[:, 2])
        return torch.cat((snps, material), -1)

    def forward(self, observations, *, pooled_embeddings=None):
        if pooled_embeddings is None:
            pooled_embeddings = self.pool_lineages(observations.snps, observations.snp_lengths,
                                                   observations.intervals, observations.interval_lengths)
        lineages = self.lineage_projection(torch.cat((pooled_embeddings, observations.lineage_scalars), -1))
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
