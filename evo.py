import torch
import math
import numpy as np


class EvolutionModelTorch(torch.nn.Module):
    """JC69 likelihood model for constructed ARG states."""

    _PROB_FLOOR = 1e-300
    _NON_FINITE_LOG_LIKELIHOOD = -1e6

    def __init__(self, env):
        super().__init__()
        self.env = env
        self._branch_length_scale = (
            2.0 * float(env.population_size) * float(env.mutation_rate)
        )
        self._site_grid_cache = {}
        self._full_material_weights_cache = {}

    def _site_interval_grid(self, device, dtype):
        key = (device, dtype)
        cached = self._site_grid_cache.get(key)
        if cached is not None:
            return cached

        sequence_length = int(self.env.sequence_length)
        num_blocks = float(max(int(self.env.num_blocks), 1))
        site_width = num_blocks / float(max(sequence_length, 1))
        sites = torch.arange(sequence_length, device=device, dtype=dtype)
        grid = (sites * site_width, sites * site_width + site_width, site_width)
        self._site_grid_cache[key] = grid
        return grid

    def _full_material_site_weights(self, device, dtype):
        key = (device, dtype)
        cached = self._full_material_weights_cache.get(key)
        if cached is not None:
            return cached

        weights = torch.ones(
            int(self.env.num_blocks),
            dtype=dtype,
            device=device,
        )
        self._full_material_weights_cache[key] = weights
        return weights

    def compute_arg_log_likelihood(self, state):
        """Compute the JC69 sequence log likelihood of a terminal ARG.

        Each marginal segment induced by recombination breakpoints is scored
        with Felsenstein pruning. ARG node times store t/(2Ne), which are
        converted to substitutions/site before scoring.
        """

        self._require_terminal(state)

        if self.env.sequences is None and not getattr(self.env, "is_vcf_mode", False):
            return 0.0

        seq_arrays = self._observed_arrays_numpy()
        log_likelihood = 0.0

        for block_start, block_end in self.get_arg_sequence_segments(state)["segments"]:
            site_start, site_end = self._observed_slice_for_blocks(block_start, block_end)
            if site_start >= site_end:
                continue

            root_id = self._segment_root_node_id(state, block_start, block_end)
            root_partials, root_log_scale = self._compute_segment_partials(
                state,
                root_id,
                block_start,
                block_end,
                site_start,
                site_end,
                seq_arrays,
                memo={},
            )
            site_probs = np.sum(root_partials * 0.25, axis=1)
            site_probs = np.maximum(site_probs, self._PROB_FLOOR)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_likelihood += float(np.log(site_probs).sum() + root_log_scale.sum())

        if not math.isfinite(log_likelihood):
            return self._NON_FINITE_LOG_LIKELIHOOD
        return float(log_likelihood)

    def get_arg_sequence_segments(self, state):
        breakpoints = self.env._arg_edge_breakpoints(state)
        recombination_events = self.env._arg_recombination_events(state, breakpoints)

        boundaries = [0, *sorted(breakpoints), int(self.env.num_blocks)]
        segments = [
            (start, end)
            for start, end in zip(boundaries, boundaries[1:])
            if start < end
        ]

        return {
            "breakpoints": boundaries[1:-1],
            "segments": segments,
            "num_segments": len(segments),
            "recombination_events": recombination_events,
        }

    def _require_terminal(self, state):
        if not self.env.is_terminal(state):
            raise ValueError("ARG likelihood and posterior reward require a terminal ARGState")

    def _seq_arrays_numpy(self):
        return self.env.seq_arrays.detach().cpu().numpy().astype(float, copy=False)

    def _observed_arrays_numpy(self):
        if getattr(self.env, "is_vcf_mode", False):
            return self.env.block_seq_arrays.detach().cpu().numpy().astype(float, copy=False)
        return self._seq_arrays_numpy()

    def _jc69_transition_matrix(self, edge_length):
        same_prob = 0.25 + 0.75 * math.exp(-4.0 * float(edge_length) / 3.0)
        diff_prob = 0.25 - 0.25 * math.exp(-4.0 * float(edge_length) / 3.0)
        transition_matrix = np.full((4, 4), diff_prob, dtype=float)
        np.fill_diagonal(transition_matrix, same_prob)
        return transition_matrix

    def _edge_length_between(self, state, parent_id, child_id):
        parent_time = float(state.all_nodes[parent_id].time)
        child_time = float(state.all_nodes[child_id].time)
        edge_length = parent_time - child_time
        if edge_length <= 0:
            raise ValueError(
                f"ARG node times must increase from child to parent: "
                f"parent={parent_id} time={parent_time}, child={child_id} time={child_time}"
            )
        return edge_length

    def _branch_length_for_likelihood(self, edge_time):
        return float(edge_time) * self._branch_length_scale

    def transition_partials(self, partials, edge_time):
        """Propagate lineage partials up one ARG branch with the JC69 model."""
        partials = self._as_partials_tensor(partials)
        transition_matrix = self._jc69_transition_matrix_torch(
            edge_time,
            device=partials.device,
            dtype=partials.dtype,
        )
        return partials @ transition_matrix.T

    def material_site_weights(self, material_segments, device=None, dtype=None):
        """Map block-coordinate material intervals to block-row model weights."""
        if dtype is None:
            dtype = self.env.block_seq_arrays.dtype
        if device is None:
            device = self.env.block_seq_arrays.device

        num_blocks = int(self.env.num_blocks)
        if material_segments.segments == ((0, num_blocks),):
            return self._full_material_site_weights(device, dtype)

        weights = torch.zeros(num_blocks, dtype=dtype, device=device)

        for segment_start, segment_end in material_segments.segments:
            start = max(int(segment_start), 0)
            end = min(int(segment_end), num_blocks)
            if end <= start:
                continue
            weights[start:end] = 1.0

        return weights

    def mask_partials(self, partials, material_segments):
        """Zero out blocks where this lineage carries no ancestral material."""
        partials = self._as_partials_tensor(partials)
        if getattr(self.env, "is_vcf_mode", False) and partials.shape[0] == material_segments.count:
            return partials
        weights = self.material_site_weights(
            material_segments,
            device=partials.device,
            dtype=partials.dtype,
        )
        return partials * weights[:, None]

    def normalize_partials(self, partials):
        """Normalize carried-site partial rows while keeping no-material rows zero."""
        partials = self._as_partials_tensor(partials)
        row_sums = partials.sum(dim=-1, keepdim=True)
        return torch.where(
            row_sums > 0,
            partials / row_sums.clamp_min(1e-12),
            torch.zeros_like(partials),
        )

    def _as_partials_tensor(self, partials):
        if partials is None:
            raise ValueError("ARGLineage.partials is required")
        if torch.is_tensor(partials):
            tensor = partials.to(dtype=torch.float32)
        else:
            tensor = torch.as_tensor(partials, dtype=torch.float32, device=self.env.block_seq_arrays.device)
        if getattr(self.env, "is_vcf_mode", False):
            if tensor.ndim != 2 or tensor.shape[-1] != 4:
                raise ValueError(
                    "VCF ARGLineage.partials must have shape [covered_variants, 4], "
                    f"got {tuple(tensor.shape)}"
                )
        else:
            expected_shape = (int(self.env.num_blocks), 4)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"ARGLineage.partials must have shape {expected_shape}, got {tuple(tensor.shape)}"
                )
        return tensor

    def _jc69_transition_matrix_torch(self, edge_time, device, dtype):
        edge_time = torch.as_tensor(edge_time, dtype=dtype, device=device)
        branch_length = edge_time * self._branch_length_scale
        decay = torch.exp(-4.0 * branch_length / 3.0)
        same_prob = 0.25 + 0.75 * decay
        diff_prob = 0.25 - 0.25 * decay
        eye = torch.eye(4, dtype=dtype, device=device)
        return eye * same_prob + (1.0 - eye) * diff_prob

    def _block_to_site(self, block_index):
        site_fraction = (
            float(block_index) * float(self.env.sequence_length) / float(self.env.num_blocks)
        )
        return int(round(site_fraction))

    def _observed_slice_for_blocks(self, block_start, block_end):
        if getattr(self.env, "is_vcf_mode", False):
            return int(block_start), int(block_end)
        return self._block_to_site(block_start), self._block_to_site(block_end)

    def _segment_root_node_id(self, state, block_start, block_end):
        roots = [
            lineage.node_id
            for lineage in state.active_lineages
            if lineage.material_segments.covers_interval(block_start, block_end)
        ]
        if len(roots) != 1:
            raise ValueError(
                "terminal ARG must have exactly one active root covering each sequence segment"
            )
        return roots[0]

    def _normalize_leaf_partials(self, partials):
        """Normalize leaf partials per site (phylo NORMALIZE_LIKELIHOOD behavior)."""
        normalized = np.full_like(partials, 0.25)
        row_sums = partials.sum(axis=-1, keepdims=True)
        np.divide(partials, row_sums, out=normalized, where=row_sums > 0)
        return normalized

    def _rescale_partials(self, partials, log_scale):
        scale = partials.max(axis=1)
        scale = np.maximum(scale, self._PROB_FLOOR)
        log_scale = log_scale + np.log(scale)
        partials = partials / scale[:, np.newaxis]
        return partials, log_scale

    def _compute_segment_partials(
        self,
        state,
        node_id,
        block_start,
        block_end,
        site_start,
        site_end,
        seq_arrays,
        memo,
    ):
        if node_id in memo:
            return memo[node_id]

        node = state.all_nodes[node_id]
        if node_id < self.env.num_sequences:
            partials = self._normalize_leaf_partials(
                seq_arrays[node_id, site_start:site_end].copy()
            )
            log_scale = np.zeros(site_end - site_start, dtype=float)
            partials, log_scale = self._rescale_partials(partials, log_scale)
            result = (partials, log_scale)
            memo[node_id] = result
            return result

        relevant_children = [
            child_id
            for child_id in node.children
            if self._edge_covers_segment(state, node_id, child_id, block_start, block_end)
        ]

        if not relevant_children:
            raise ValueError(f"ARG node {node_id} has no descendants for the requested segment")

        partials = np.ones((site_end - site_start, seq_arrays.shape[-1]), dtype=float)
        log_scale = np.zeros(site_end - site_start, dtype=float)
        for child_id in relevant_children:
            child_partials, child_log_scale = self._compute_segment_partials(
                state,
                child_id,
                block_start,
                block_end,
                site_start,
                site_end,
                seq_arrays,
                memo,
            )
            edge_time = self._edge_length_between(state, node_id, child_id)
            branch_length = self._branch_length_for_likelihood(edge_time)
            transition_matrix = self._jc69_transition_matrix(branch_length)
            child_partials = np.maximum(child_partials, self._PROB_FLOOR)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                partials *= child_partials @ transition_matrix.T
            log_scale += child_log_scale
            partials, log_scale = self._rescale_partials(partials, log_scale)
        result = (partials, log_scale)
        memo[node_id] = result
        return result

    def _edge_covers_segment(self, state, parent_id, child_id, block_start, block_end):
        parent = state.all_nodes[parent_id]
        child = state.all_nodes[child_id]
        return parent.material_segments.intersection_count(
            child.material_segments,
            interval_start=block_start,
            interval_end=block_end,
        ) > 0
