from .actions import CoalescenceChoice, RecombinationChoice
from .state import ARGLineage


class TransitionMixin:
    def _finalize_transition_state(self, next_state, log_prior):
        if log_prior is not None:
            next_state.accumulated_log_prior += log_prior
        next_state.is_done = self.is_terminal(next_state)
        if next_state.is_done:
            log_likelihood = self.evolution_model.compute_arg_log_likelihood(next_state)
            next_state.log_reward = self.compute_terminal_log_reward(next_state, log_likelihood)
        else:
            next_state.log_reward = None
        return next_state

    def apply_coalescence(self, state, action, log_prior=None):

        rates = state.rates
        if rates is None:
            rates = self.compute_event_rates(self.enumerate_actions(state))
            state.rates = rates

        next_state = state.clone(copy_partials=False)
        i = action.active_lineage_i
        j = action.active_lineage_j

        child_i = next_state.active_lineages[i].clone(copy_partials=False, copy_mask=False)
        child_j = next_state.active_lineages[j].clone(copy_partials=False, copy_mask=False)

        parent_id = next_state.max_node_idx + 1
        parent_segments = child_i.material_segments.union(child_j.material_segments)
        overlap_count = child_i.material_segments.intersection_count(child_j.material_segments)
        delta_t = self.time_env.time_action_to_delta(action.time_action, self._total_event_rate(rates))
        parent_time = float(state.current_time) + delta_t
        next_state.current_time = parent_time
        parent_partials = self._coalesced_parent_partials(
            child_i,
            child_j,
            parent_segments,
            parent_time,
        )
        parent = ARGLineage(
            node_id=parent_id,
            children=[child_i.node_id, child_j.node_id],
            parents=[],
            material_segments=parent_segments,
            num_blocks=self.num_blocks,
            partials=parent_partials,
            sequences_indices=sorted(set(child_i.sequences_indices + child_j.sequences_indices)),
            event_type="coal",
            time=parent_time,
        )

        child_i.parents.append(parent.node_id)
        child_j.parents.append(parent.node_id)
        child_i.partials = None
        child_j.partials = None
        next_state.active_lineages[i] = child_i
        next_state.active_lineages[j] = child_j
        next_state.all_nodes[child_i.node_id] = child_i
        next_state.all_nodes[child_j.node_id] = child_j
        next_state.all_nodes[parent.node_id] = parent
        next_state.active_lineages = [
            lineage for idx, lineage in enumerate(next_state.active_lineages) if idx not in (i, j)
        ]
        next_state.active_lineages.append(parent)
        next_state.max_node_idx = parent.node_id
        if next_state.total_active_blocks is not None:
            next_state.total_active_blocks = int(next_state.total_active_blocks) - overlap_count
        return self._finalize_transition_state(next_state, log_prior)

    def apply_recombination(self, state, action, log_prior=None):
        rates = state.rates
        if rates is None:
            rates = self.compute_event_rates(self.enumerate_actions(state))
            state.rates = rates

        # if log_prior is None:
        #     log_prior = self.compute_cwr_event_log_prior(state, action, rates=rates)
        next_state = state.clone(copy_partials=False)
        current_lineage_idx = action.active_lineage_i
        breakpoint = action.breakpoint
        child = next_state.active_lineages[current_lineage_idx].clone(copy_partials=False, copy_mask=False)
        left_segments, right_segments = child.material_segments.split(breakpoint)

        left_parent_id = next_state.max_node_idx + 1
        right_parent_id = next_state.max_node_idx + 2
        delta_t = self.time_env.time_action_to_delta(action.time_action, self._total_event_rate(rates))

        event_time = float(state.current_time) + delta_t
        next_state.current_time = event_time
        left_partials = self._recombined_parent_partials(child, left_segments, event_time)
        right_partials = self._recombined_parent_partials(child, right_segments, event_time)
        left_parent = ARGLineage(
            node_id=left_parent_id,
            children=[child.node_id],
            parents=[],
            material_segments=left_segments,
            num_blocks=self.num_blocks,
            partials=left_partials,
            sequences_indices=list(child.sequences_indices),
            event_type="recomb",
            breakpoint=breakpoint,
            recombination_side="left",
            time=event_time,
        )
        right_parent = ARGLineage(
            node_id=right_parent_id,
            children=[child.node_id],
            parents=[],
            material_segments=right_segments,
            num_blocks=self.num_blocks,
            partials=right_partials,
            sequences_indices=list(child.sequences_indices),
            event_type="recomb",
            breakpoint=breakpoint,
            recombination_side="right",
            time=event_time,
        )

        child.parents = [left_parent.node_id, right_parent.node_id]
        child.partials = None
        next_state.all_nodes[child.node_id] = child
        next_state.all_nodes[left_parent.node_id] = left_parent
        next_state.all_nodes[right_parent.node_id] = right_parent
        next_state.active_lineages = [
            lineage for idx, lineage in enumerate(next_state.active_lineages) if idx != current_lineage_idx
        ]
        next_state.active_lineages.extend([left_parent, right_parent])
        next_state.max_node_idx = right_parent.node_id
        return self._finalize_transition_state(next_state, log_prior)

    def preview_action_for_time_model(self, state, action):
        """Apply only an action's topology for time-policy conditioning.

        The event time cannot be used while constructing this preview because it
        is the value that the time policy is about to choose.  Consequently, the
        preview pools coalescing children's evidence and retains disagreement
        and overlap channels, or splits a recombining lineage's evidence. It is
        an encoding-only state and must not be committed to a rollout.
        """
        next_state = state.clone(copy_partials=False)

        if isinstance(action, CoalescenceChoice):
            if not action.is_valid_for(next_state.active_lineages):
                raise ValueError(f"Invalid coalescence action: {action}")

            i = int(action.active_lineage_i)
            j = int(action.active_lineage_j)
            child_i = next_state.active_lineages[i]
            child_j = next_state.active_lineages[j]
            parent_segments = child_i.material_segments.union(
                child_j.material_segments
            )
            evidence, pair_features = self._coalescence_preview_features(child_i, child_j)
            parent = ARGLineage(
                node_id=next_state.max_node_idx + 1,
                material_segments=parent_segments,
                num_blocks=self.num_blocks,
                partials=evidence,
                preview_pair_features=pair_features,
                sequences_indices=sorted(
                    set(child_i.sequences_indices + child_j.sequences_indices)
                ),
                event_type="coal",
                time=state.current_time,
            )
            next_state.active_lineages = [
                lineage
                for idx, lineage in enumerate(next_state.active_lineages)
                if idx not in (i, j)
            ]
            next_state.active_lineages.append(parent)
            return next_state

        if isinstance(action, RecombinationChoice):
            if not action.is_valid_for(next_state.active_lineages):
                raise ValueError(f"Invalid recombination action: {action}")
            if action.breakpoint is None:
                raise ValueError(
                    "A recombination breakpoint must be chosen before the time-model preview"
                )

            lineage_idx = int(action.active_lineage_i)
            child = next_state.active_lineages[lineage_idx]
            left_segments, right_segments = child.material_segments.split(
                action.breakpoint
            )
            left_parent = ARGLineage(
                node_id=next_state.max_node_idx + 1,
                material_segments=left_segments,
                num_blocks=self.num_blocks,
                partials=self._recombined_parent_partials(
                    child,
                    left_segments,
                    parent_time=None,
                ),
                sequences_indices=list(child.sequences_indices),
                event_type="recomb",
                breakpoint=int(action.breakpoint),
                recombination_side="left",
                time=state.current_time,
            )
            right_parent = ARGLineage(
                node_id=next_state.max_node_idx + 2,
                material_segments=right_segments,
                num_blocks=self.num_blocks,
                partials=self._recombined_parent_partials(
                    child,
                    right_segments,
                    parent_time=None,
                ),
                sequences_indices=list(child.sequences_indices),
                event_type="recomb",
                breakpoint=int(action.breakpoint),
                recombination_side="right",
                time=state.current_time,
            )
            next_state.active_lineages = [
                lineage
                for idx, lineage in enumerate(next_state.active_lineages)
                if idx != lineage_idx
            ]
            next_state.active_lineages.extend([left_parent, right_parent])
            return next_state

        raise ValueError(f"Unknown action event_type: {action}")

    def apply_action(self, state, action, log_prior=None):
        
        if isinstance(action, RecombinationChoice):
            return self.apply_recombination(
                state,
                action,
                log_prior
            )
        elif isinstance(action, CoalescenceChoice):
            return self.apply_coalescence(
                state,
                action,
                log_prior
            )
        else:
            raise ValueError(f"Unknown action event_type: {action}")


