import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class StateEncodingMixin:
    def _encode_lineage_features(self, lineage_seq_features, batch_active_lineage_counts,
                                preview_pair_features=None):
        batch_size, active_lineages, seq_len, channels = lineage_seq_features.shape
        if seq_len != int(self.env.num_blocks) or channels != 4:
            raise ValueError(
                "sequence features must have shape "
                f"(batch, active_lineages, {int(self.env.num_blocks)}, 4), "
                f"got {tuple(lineage_seq_features.shape)}"
            )

        batch_input = lineage_seq_features.reshape(batch_size, active_lineages, -1)
        if batch_input.shape[-1] != self.seq_embedding.in_features:
            raise ValueError(
                "Encoded batch_input last dimension must match num_blocks * 4 "
                f"({self.seq_embedding.in_features}), got {batch_input.shape[-1]}"
            )

        batch_input = batch_input.to(device=self.device, dtype=torch.float32)
        batch_active_lineage_counts = batch_active_lineage_counts.to(device=self.device, dtype=torch.long)

        valid_mask = (
            torch.arange(active_lineages, device=self.device)[None, :]
            < batch_active_lineage_counts[:, None]
        )
        lineage_reps = self.seq_embedding(batch_input)
        if preview_pair_features is not None:
            expected_shape = (batch_size, active_lineages, seq_len, 5)
            if tuple(preview_pair_features.shape) != expected_shape:
                raise ValueError(f"preview pair features must have shape {expected_shape}")
            pair_input = preview_pair_features.to(device=self.device, dtype=torch.float32)
            lineage_reps = lineage_reps + self.preview_pair_embedding(
                pair_input.reshape(batch_size, active_lineages, -1)
            )
        summary_token = self.summary_token.expand(batch_size, -1, -1)
        transformer_input = torch.cat([summary_token, lineage_reps], dim=1)

        key_padding_mask = F.pad(~valid_mask, (1, 0), value=False)
        encoded = self.encoder(transformer_input, src_key_padding_mask=key_padding_mask)

        summary_reps = encoded[:, 0]
        lineage_reps = encoded[:, 1:]
        lineage_reps = lineage_reps * valid_mask.unsqueeze(-1)
        return lineage_reps, summary_reps, lineage_seq_features, batch_active_lineage_counts

    def _encode_states(self, states):
        batch_size = len(states)
        if batch_size == 0:
            raise ValueError("ARGModel.forward requires at least one state")

        active_counts = [len(state.active_lineages) for state in states]
        batch_active_lineage_counts = torch.tensor(
            active_counts, dtype=torch.long, device=self.device,
        )

        num_blocks = self.env.num_blocks
        max_active_lineages = max(active_counts, default=0)
        lineage_seq_features = self.env.block_seq_arrays.new_zeros(
            batch_size,
            max_active_lineages,
            num_blocks,
            4,
        )
        preview_pair_features = None

        for batch_idx, state in enumerate(states):
            for lineage_idx, lineage in enumerate(state.active_lineages):
                feature = self._lineage_partials_tensor(lineage)
                weights = self._material_segments_masking(
                    lineage.material_segments,
                    device=self.device,
                    dtype=self.env.block_seq_arrays.dtype,
                )
                masked_feature = feature * weights[:, None]
                lineage_seq_features[batch_idx, lineage_idx] = (
                    self.env.evolution_model.normalize_partials(masked_feature)
                )
                if lineage.preview_pair_features is not None:
                    if preview_pair_features is None:
                        preview_pair_features = lineage_seq_features.new_zeros(
                            batch_size, max_active_lineages, num_blocks, 5,
                        )
                    pair_features = lineage.preview_pair_features.to(lineage_seq_features)
                    if tuple(pair_features.shape) != (num_blocks, 5):
                        raise ValueError(f"Lineage preview pair features must have shape {(num_blocks, 5)}")
                    preview_pair_features[batch_idx, lineage_idx] = pair_features * weights[:, None]

        return self._encode_lineage_features(
            lineage_seq_features, batch_active_lineage_counts, preview_pair_features,
        )

    def _lineage_partials_tensor(self, lineage):
        if lineage.partials is None:
            raise ValueError(
                f"Active ARG lineage {lineage.node_id} is missing partials; "
                "state transitions must populate ARGLineage.partials"
            )
        partials = lineage.partials
        if torch.is_tensor(partials):
            partials = partials.to(device=self.device, dtype=torch.float32)
        else:
            partials = torch.as_tensor(partials, device=self.device, dtype=torch.float32)
        expected_shape = (int(self.env.num_blocks), 4)
        if tuple(partials.shape) != expected_shape:
            raise ValueError(
                f"Active ARG lineage {lineage.node_id} partials must have shape "
                f"{expected_shape}, got {tuple(partials.shape)}"
            )
        return partials

    def _material_segments_masking(self, material_segments, device, dtype):
        num_blocks = int(self.env.num_blocks)
        weights = torch.zeros(num_blocks, dtype=dtype, device=device)

        for segment_start, segment_end in material_segments.segments:
            start = max(int(segment_start), 0)
            end = min(int(segment_end), num_blocks)
            if end <= start:
                continue
            weights[start:end] = 1.0
        return weights

    def sample(self, logits, random_spec=None):
        if random_spec is None:
            return Categorical(logits=logits).sample()

        temperature = random_spec["T"]
        return Categorical(logits=logits / temperature).sample()

