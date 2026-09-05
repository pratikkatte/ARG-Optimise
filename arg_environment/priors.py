import math
from dataclasses import replace

import numpy as np

from .actions import CoalescenceChoice, PriorActionOptions, RecombinationChoice


class PriorMixin:
    def compute_coalescence_actions(self, state):
        return list(CoalescenceChoice.enumerate_from_active_lineages(state.active_lineages))

    def compute_recombination_actions(self, state):
        return list(RecombinationChoice.enumerate_from_active_lineages(state.active_lineages))

    def enumerate_prior_options(self, state):
        coal_actions, recomb_actions = self.enumerate_actions(state)
        rates = self.compute_event_rates((coal_actions, recomb_actions))
        state.rates = rates
        prior_options = PriorActionOptions(
            coal_actions=tuple(coal_actions),
            recomb_choices=tuple(recomb_actions),
            rates=rates,
        )
        state.prior_options = prior_options
        return prior_options

    def action_options_from_prior_options(self, prior_options):
        actions = []
        if prior_options.rates["lambda_coal"] > 0:
            actions.extend(choice.as_dict() for choice in prior_options.coal_actions)
        if prior_options.rates["lambda_recomb"] > 0:
            actions.extend(
                {
                    "event_type": "recomb",
                    "active_lineage_i": int(choice.active_lineage_i),
                    "material_count": int(choice.material_count),
                    "span_start": int(choice.span_start),
                    "span_end": int(choice.span_end),
                }
                for choice in prior_options.recomb_choices
                if choice.breakpoint_count > 0
            )
        return actions


    def is_terminal(self, state):
        if state.total_active_blocks is None:
            raise ValueError("total_active_blocks is required for terminal check")
        else:
            result = int(state.total_active_blocks) == self.num_blocks
            # bool(np.all(self.get_active_counts(state) == 1)) ## another way, realtime compute. 
            return result

    def compute_event_rates(self, actions):
        coal_actions, recomb_actions = actions

        lambda_coal = float(len(coal_actions))

        total_blocks = sum(choice.material_count for choice in recomb_actions)
        total_active_material_length = float(total_blocks) / float(self.num_blocks)
        lambda_recomb = self.rho / 2.0 * total_active_material_length
        
        return {
            "lambda_coal": lambda_coal,
            "lambda_recomb": lambda_recomb,
            "total_active_material_length": total_active_material_length,
        }

    def compute_event_probabilities(self, state, actions=None):
        if actions is None:
            actions = self.enumerate_actions(state)
        rates = self.compute_event_rates(actions)
        state.rates = rates
        denom = rates["lambda_coal"] + rates["lambda_recomb"]
        if denom <= 0:
            return {"coal": 0.0, "recomb": 0.0}
        return {
            "coal": rates["lambda_coal"] / denom,
            "recomb": rates["lambda_recomb"] / denom,
        }

    def enumerate_actions(self, state):

        coal_actions = self.compute_coalescence_actions(state)
        recomb_actions = self.compute_recombination_actions(state)

        return coal_actions, recomb_actions


    def _sample_prior_step(self, state):
        """Sample one prior coalescence/recombination action and its log prior."""
        event_types = ["coal", "recomb"]
        combined_actions = self.enumerate_actions(state)
        event_probs = list(self.compute_event_probabilities(state, combined_actions).values())
        chosen_event = event_types[np.random.choice(2, p=event_probs)]

        coal_actions, recomb_actions = combined_actions
        if chosen_event == "coal":
            chosen_action = self.rng.choice(coal_actions)
        else:
            prior_result = self._sample_recombination_prior_action(recomb_actions)
            if prior_result is None:
                raise ValueError("No valid recombination actions to sample")
            action_dict, _, selected = prior_result
            chosen_action = replace(
                selected,
                breakpoint=action_dict["breakpoint"],
            )

        time_action = self.time_env.sample_action_from_prior(
            self._total_event_rate(state.rates), self.rng
        )
        chosen_action = replace(chosen_action, time_action=time_action)
        log_prior = self.compute_cwr_event_log_prior(state, combined_actions, chosen_action)
        return chosen_action, log_prior

    def sample_log_rewards(self, num_trajs, verbose=True):
        """Sample prior rollouts sequentially and return terminal log rewards."""
        log_rewards = []
        for traj_idx in range(num_trajs):
            if verbose:
                print(
                    f"Sampling prior trajectory {traj_idx + 1}/{num_trajs} for log Z init..."
                )
            state = self.get_initial_state()
            while not state.is_done:
                action, log_prior = self._sample_prior_step(state)
                state = self.apply_action(state, action, log_prior=log_prior)
            log_rewards.append(state.log_reward)
        return log_rewards

    def compute_cwr_event_log_prior(self, state, combined_actions, action=None, rates=None):
        if action is None:
            action = combined_actions
            combined_actions = self.enumerate_actions(state)
        coal_actions, recomb_actions = combined_actions

        if rates is None:
            rates = state.rates if state.rates is not None else self.compute_event_rates((coal_actions, recomb_actions))
        state.rates = rates
        
        total_rate = self._total_event_rate(rates)
        recomb_total_weight = sum(choice.material_count for choice in recomb_actions)

        wait_log_prior = self.time_env.time_action_log_probability(action.time_action, total_rate)

        if isinstance(action, CoalescenceChoice) and CoalescenceChoice.is_valid_for(action, state.active_lineages):
            action_log_prior = math.log((rates["lambda_coal"] / total_rate) / len(coal_actions))
            
        elif isinstance(action, RecombinationChoice) and RecombinationChoice.is_valid_for(action, state.active_lineages):
            action_log_prior = math.log((rates["lambda_recomb"] / total_rate) * (action.material_count / recomb_total_weight) / action.breakpoint_count)
        else:
            raise ValueError(f"Invalid action: {action}")

        return action_log_prior + wait_log_prior

    def prepare_state_rollout_inputs(
        self,
        states,
        random_spec=None,
    ):
        if len(states) == 0:
            raise ValueError("states must contain at least one ARGState")

        return {
            "states": states,
            "candidate_actions": [self.enumerate_actions(state) for state in states],
            "random_spec": random_spec,
        }

    def _sample_recombination_prior_action(self, recomb_weights):
        total_weight = sum(self._recomb_weight(item) for item in recomb_weights)
        if total_weight <= 0:
            return None

        target = self.rng.random() * total_weight
        cumulative = 0.0
        selected = recomb_weights[-1]
        for item in recomb_weights:
            cumulative += self._recomb_weight(item)
            if target <= cumulative:
                selected = item
                break

        if isinstance(selected, RecombinationChoice):
            if selected.breakpoint_count <= 0:
                return None
            breakpoint = (
                selected.span_start
                + 1
                + self.rng.randrange(selected.breakpoint_count)
            )
            action = {
                "event_type": "recomb",
                "active_lineage_i": selected.active_lineage_i,
                "breakpoint": breakpoint,
            }
            return action, selected.material_count, selected

        lineage_i, lineage_weight, valid_breakpoints = selected
        if not valid_breakpoints:
            return None
        breakpoint = valid_breakpoints[self.rng.randrange(len(valid_breakpoints))]
        action = {
            "event_type": "recomb",
            "active_lineage_i": lineage_i,
            "breakpoint": breakpoint,
        }
        return action, lineage_weight, valid_breakpoints

    def _recomb_weight(self, recomb_weight):
        if isinstance(recomb_weight, RecombinationChoice):
            return recomb_weight.material_count
        return recomb_weight[1]
