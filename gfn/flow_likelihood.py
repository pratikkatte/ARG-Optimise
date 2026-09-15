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
            same, different = self.env.evolution_model._jc69_transition_probabilities(branch_length)
            transition = combined.new_full((4, 4), different)
            transition.diagonal().fill_(same)
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

    def parent_values_batch(self, specs):
        """Return (partials, increment) for independent (time, material, children).

        Group by child count to keep the scalar pruning order. All increments
        share one device-to-host transfer instead of synchronizing per parent.
        """
        if self.env.sequences is None:
            return [(None, 0.0) for _ in specs]
        if not specs:
            return []
        results = [None] * len(specs)
        increment_batches, result_indices = [], []
        evo = self.env.evolution_model
        for child_count in sorted({len(children) for _, _, children in specs}):
            indices = [i for i, (_, _, children) in enumerate(specs) if len(children) == child_count]
            group = [specs[i] for i in indices]
            combined = torch.ones(len(group), self.env.sequence_length, 4,
                                  dtype=torch.float64, device=self.env.device)
            for slot in range(child_count):
                children = [children[slot] for _, _, children in group]
                if any(child.likelihood_partials is None for child in children):
                    raise ValueError("Missing incremental likelihood partials for an active lineage")
                times = [float(time) - float(child.time)
                         for (time, _, _), child in zip(group, children)]
                if any(time <= 0 for time in times):
                    raise ValueError("Likelihood branches require increasing node times")
                # Match parent(): float64 transition coefficients computed on the host.
                coefficients = combined.new_tensor([
                    evo._jc69_transition_probabilities(time * evo._branch_length_scale) for time in times])
                transitions = coefficients[:, 1, None, None].expand(-1, 4, 4).clone()
                transitions.diagonal(dim1=1, dim2=2).copy_(coefficients[:, 0, None])
                partials = torch.stack([child.likelihood_partials for child in children])
                transitioned = torch.bmm(partials.clamp_min(1e-300), transitions.transpose(1, 2))
                mask = evo.material_masks_batch([child.material_segments for child in children],
                                                site_resolution=True)
                combined = combined * torch.where(mask[:, :, None], transitioned, 1.0)
            mask = evo.material_masks_batch([material for _, material, _ in group], site_resolution=True)
            norm = combined.sum(-1).clamp_min(1e-300)
            increment_batches.append(torch.where(mask, norm.log(), 0.0).sum(-1))
            partials = torch.where(mask[:, :, None], combined / norm[:, :, None], 0.0)
            for index, value in zip(indices, partials.unbind()):
                results[index] = value
            result_indices.extend(indices)
        increments = torch.cat(increment_batches).detach().cpu().tolist()
        for index, increment in zip(result_indices, increments):
            results[index] = (results[index], increment)
        return results
