"""Incremental site-resolution pruning potential for intermediate ARG flows.

The policy's normalized block features are unchanged. This separate float64
tracker retains the normalization factors needed by a likelihood-based flow.
At a terminal state its potential equals the full JC69 sequence log likelihood.
"""
import math

import torch


class PartialLikelihoodTracker:
    def __init__(self, env):
        self.env = env
        self.initial_log_likelihood = (
            -env.sequence_length * math.log(4.0) if env.sequences is not None else 0.0
        )

    def mask(self, segments):
        mask = torch.zeros(self.env.sequence_length, dtype=torch.bool, device=self.env.device)
        for left, right in segments.segments:
            start = self.env.evolution_model._block_to_site(left)
            end = self.env.evolution_model._block_to_site(right)
            mask[start:end] = True
        return mask

    def initial_partials(self, node):
        if self.env.sequences is None:
            return None
        partials = self.env.seq_arrays[node.node_id].detach().double()
        sums = partials.sum(-1, keepdim=True)
        partials = torch.where(sums > 0, partials / sums.clamp_min(1e-300), 0.25)
        return partials * self.mask(node.material_segments)[:, None]

    def initialize(self, state):
        state.partial_log_likelihood = self.initial_log_likelihood
        for node in state.active_lineages:
            node.likelihood_partials = self.initial_partials(node)

    def parent(self, node, children):
        if self.env.sequences is None:
            return 0.0
        combined = torch.ones(self.env.sequence_length, 4, dtype=torch.float64, device=self.env.device)
        for child in children:
            if child.likelihood_partials is None:
                raise ValueError("Missing incremental likelihood partials for an active lineage")
            edge_time = float(node.time) - float(child.time)
            if edge_time <= 0:
                raise ValueError("Likelihood branches require increasing node times")
            branch_length = edge_time * self.env.evolution_model._branch_length_scale
            decay = math.exp(-4.0 * branch_length / 3.0)
            transition = combined.new_full((4, 4), 0.25 - 0.25 * decay)
            transition.diagonal().fill_(0.25 + 0.75 * decay)
            transitioned = child.likelihood_partials.clamp_min(1e-300) @ transition.T
            mask = self.mask(child.material_segments)
            combined *= torch.where(mask[:, None], transitioned, 1.0)
        mask = self.mask(node.material_segments)
        norm = combined.sum(-1).clamp_min(1e-300)
        increment = float(norm[mask].log().sum().item())
        node.likelihood_partials = torch.where(mask[:, None], combined / norm[:, None], 0.0)
        node.likelihood_log_increment = increment
        return increment

    def restore(self, state):
        needed = {node.node_id for node in state.active_lineages if node.likelihood_partials is None}
        pending = list(needed)
        while pending:
            for child_id in state.all_nodes[pending.pop()].children:
                if state.all_nodes[child_id].likelihood_partials is None and child_id not in needed:
                    needed.add(child_id)
                    pending.append(child_id)
        for node_id in sorted(needed):
            node = state.all_nodes[node_id]
            if not node.children:
                node.likelihood_partials = self.initial_partials(node)
            else:
                self.parent(node, [state.all_nodes[c] for c in node.children])
        state.partial_log_likelihood = self.initial_log_likelihood + sum(
            node.likelihood_log_increment for node in state.all_nodes.values()
        )
