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

    def compute_arg_log_likelihood(self, state):
        """Compute the JC69 sequence log likelihood of a terminal ARG.

        Each marginal segment induced by recombination breakpoints is scored
        with Felsenstein pruning. ARG node times store t/(2Ne), which are
        converted to substitutions/site before scoring.
        """

        self._require_terminal(state)

        if self.env.sequences is None:
            return 0.0

        seq_arrays = self._seq_arrays_numpy()
        log_likelihood = 0.0

        for block_start, block_end in self.env.get_arg_sequence_segments(state)["segments"]:
            site_start = self._block_to_site(block_start)
            site_end = self._block_to_site(block_end)
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

    def _require_terminal(self, state):
        if not self.env.is_terminal(state):
            raise ValueError("ARG likelihood and posterior reward require a terminal ARGState")

    def _seq_arrays_numpy(self):
        return self.env.seq_arrays.detach().cpu().numpy().astype(float, copy=False)

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

    def _block_to_site(self, block_index):
        site_fraction = (
            float(block_index) * float(self.env.sequence_length) / float(self.env.num_blocks)
        )
        return int(round(site_fraction))

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
