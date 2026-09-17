"""Exact CPU messages; no learned features or finite-sites substitutions."""
import math

import numpy as np

from .states import DescendantSegments


class InfiniteSitesTracker:
    initial_log_likelihood = 0.0

    def __init__(self, env):
        self.env = env

    def indices(self, material):
        pieces = [np.arange(*np.searchsorted(self.env.snp_data.positions, [l, r], side='left'))
                  for l, r in material.segments]
        return np.concatenate(pieces).astype(np.int64) if pieces else np.empty(0, dtype=np.int64)

    @staticmethod
    def cache(node, indices, messages):
        if not np.isfinite(messages).all() or np.any(messages < 0):
            raise FloatingPointError('invalid infinite-sites messages')
        indices.setflags(write=False)
        messages.setflags(write=False)
        node.snp_indices, node.messages = indices, messages

    def initialize_leaf(self, node):
        node.descendants = DescendantSegments(((0, self.env.sequence_length, 1 << node.node_id),))
        indices = self.indices(node.material_segments)
        g = self.env.snp_data.genotypes[node.node_id, indices].astype(np.float64)
        self.cache(node, indices, np.column_stack((1 - g, g, np.zeros(len(g)))))
        node.exposure_increment = 0.0

    def parent(self, node, children):
        if len(children) == 2:
            node.descendants = children[0].descendants.merge(children[1].descendants)
        elif len(children) == 1:
            node.descendants = children[0].descendants.restrict(node.material_segments)
        else:
            raise ValueError('ancestral nodes require one or two children')
        if node.descendants.material != node.material_segments:
            raise ValueError('ancestry and material coverage disagree')
        indices = self.indices(node.material_segments)
        # Identity for an absent child; absence is never inferred from message values.
        combined = np.zeros((len(indices), 3), dtype=np.float64)
        combined[:, 0] = combined[:, 1] = 1.0
        exposures = []
        for child in children:
            dt = node.time - child.time
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError('parent times must strictly exceed child times')
            relevant = child.descendants.restrict(node.material_segments)
            exposures.append(dt * sum(r - l for l, r, bits in relevant.segments
                                      if bits != self.env.all_samples))
            common, parent_rows, child_rows = np.intersect1d(
                indices, child.snp_indices, assume_unique=True, return_indices=True)
            if len(common):
                a, d, m = child.messages[child_rows].T
                propagated = m + dt * d
                old = combined[parent_rows].copy()
                combined[parent_rows, 0] = old[:, 0] * a
                combined[parent_rows, 1] = old[:, 1] * d
                combined[parent_rows, 2] = old[:, 2] * a + old[:, 0] * propagated
        node.exposure_increment = math.fsum(exposures)
        self.cache(node, indices, combined)

    def record(self, state, node):
        state.exposure += node.exposure_increment
        for row, index in enumerate(node.snp_indices):
            if (np.isnan(state.completed_site_lengths[index])
                    and node.descendants.at(self.env.snp_data.positions[index]) == self.env.all_samples):
                state.completed_site_lengths[index] = node.messages[row, 2]

    def potential(self, state):
        if not math.isfinite(state.exposure) or state.exposure < 0:
            raise FloatingPointError('invalid genomic exposure')
        lengths = state.completed_site_lengths[~np.isnan(state.completed_site_lengths)]
        if not np.isfinite(lengths).all() or np.any(lengths < 0):
            raise FloatingPointError('invalid completed SNP branch weights')
        if np.any(lengths == 0) or (len(lengths) and self.env.kappa == 0):
            return -math.inf
        result = -self.env.kappa * state.exposure
        if len(lengths):
            result += len(lengths) * math.log(self.env.kappa) + math.fsum(map(math.log, lengths))
        if not math.isfinite(result):
            raise FloatingPointError('finite likelihood overflowed float64')
        return float(result)

    def restore(self, state):
        """Reconstruct caches from graph nodes/edges, independently of action history."""
        state.exposure = 0.0
        state.completed_site_lengths = np.full(self.env.num_variants, np.nan)
        for node in sorted(state.all_nodes.values(), key=lambda n: (n.time, n.node_id)):
            if not node.children:
                if node.node_id >= self.env.num_sequences or node.time != 0:
                    raise ValueError('invalid sample node')
                self.initialize_leaf(node)
            else:
                self.parent(node, [state.all_nodes[c] for c in node.children])
                self.record(state, node)
        state.exposure = math.fsum(n.exposure_increment for n in state.all_nodes.values())
        state.partial_log_likelihood = self.potential(state)
        active = {n.node_id for n in state.active_lineages}
        for node in state.all_nodes.values():
            if node.node_id not in active:
                node.messages = node.snp_indices = None
