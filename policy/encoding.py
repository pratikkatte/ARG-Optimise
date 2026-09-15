"""Packed lineage encoding shared by trainable and frozen encoders."""
import torch
from torch.nn import functional as F


def encode_packed_lineages(features, counts, seq_embedding, summary_token, encoder):
    """Project real lineages, pad embeddings, and encode with a summary token."""
    packed, offsets = features.tensor, features.row_offsets
    batch_size = len(offsets) - 1
    active = max(end - start for start, end in zip(offsets, offsets[1:]))
    device = summary_token.device
    valid = torch.arange(active, device=device)[None, :] < counts.to(device)[:, None]
    projected = seq_embedding(packed.reshape(packed.shape[0], -1).to(
        device=device, dtype=seq_embedding.weight.dtype))
    # Preserve the projection bias and its gradient for padded inputs.
    padded = seq_embedding.bias.expand(batch_size, active, -1).clone()
    padded[valid] = projected
    tokens = torch.cat((summary_token.expand(batch_size, -1, -1), padded), dim=1)
    encoded = encoder(tokens, key_padding_mask=F.pad(~valid, (1, 0), value=False))
    return encoded, valid
