import math
import torch
from typing import Optional
from utils import (
    ThreadPathState, 
    ThreadChoice,
    _thread_leaf_into_site_tree_full,
    _binary_site_log_likelihood,
    _choice_prior_weights
)

class ARGRewardMixin:
    """
    Mixin class that provides reward computation for the ARGweaverThreadEnv GFlowNet.
    """

    @staticmethod
    def _transition_choice_key(choice: ThreadChoice) -> tuple[int, tuple[int, ...], int, bool]:
        """
        Key used by the SMC transition prior to decide whether consecutive
        choices represent the same threading configuration at adjacent sites.
        """
        return (
            choice.branch_child,
            choice.branch_signature,
            choice.time_idx,
            choice.is_root_branch,
        )

    def compute_log_local_reward(
        self,
        prev_choice: Optional[ThreadChoice],
        curr_choice: ThreadChoice,
        site_index: int,
    ) -> float:
        """
        Return the additive per-site log reward for a single threading action.

        This exposes the local transition-plus-emission contribution used by
        the full-path reward, scaled by the reward temperature.
        """
        if self.config.reward_temperature <= 0.0:
            raise ValueError("reward_temperature must be strictly positive.")

        log_trans = self._compute_log_transition(prev_choice, curr_choice, site_index)
        log_emit = self._compute_log_emission(curr_choice, site_index)
        return (log_trans + log_emit) / self.config.reward_temperature
    
    def compute_log_reward(self, st: ThreadPathState) -> float:
        """
        Calculates the unnormalized log-posterior of a completely threaded sequence.
        The joint probability of the threading path, scaled by the reward temperature (beta).
        """
        expected_length = self.config.sequence_length
        if st.site_index != expected_length or len(st.choices) != expected_length:
            raise ValueError(
                "State is not terminal. The sequence must be fully threaded to compute reward."
            )
        if self.config.reward_temperature <= 0.0:
            raise ValueError("reward_temperature must be strictly positive.")

        total_log_reward = 0.0
        prev_choice = None

        # Iterate through the full threaded path and accumulate the joint log-score.
        for i, curr_choice in enumerate(st.choices):
            total_log_reward += self.compute_log_local_reward(prev_choice, curr_choice, i)
            prev_choice = curr_choice

        return total_log_reward

    def _compute_log_transition(self, prev_choice: Optional[ThreadChoice], curr_choice: ThreadChoice, site_index: int) -> float:
        """
        Computes the SMC prior log transition probability.
        """
        if not (0 <= site_index < self.config.sequence_length):
            raise IndexError(f"site_index {site_index} is out of bounds for sequence length {self.config.sequence_length}")

        # Get prior probability for the current choice
        curr_choices_at_site = self.site_choices[site_index]
        curr_prior_weights = _choice_prior_weights(curr_choices_at_site)
        curr_choice_idx = self.choice_to_index[site_index][curr_choice]
        prior_prob = float(curr_prior_weights[curr_choice_idx].item())
        
        if prev_choice is None:
            # site_index == 0: compute the stationary coalescent prior \log P(c_1)
            return math.log(max(prior_prob, 1e-12))
        
        # Sequentially Markovian Coalescent (SMC) prior
        recomb_mass = 1.0 - math.exp(-self.config.recomb_rate)
        stay_mass = 1.0 - recomb_mass
        
        # Placeholder transition model:
        # - recombination: resample from the site-specific stationary prior
        # - no recombination: keep the same canonical branch/time attachment
        # This preserves the intended ARGweaver-style conditional structure while
        # leaving room for a richer transition matrix later.
        prob = recomb_mass * prior_prob
        
        same_branch_and_time = self._transition_choice_key(prev_choice) == self._transition_choice_key(curr_choice)
        
        if same_branch_and_time:
            prob += stay_mass
            
        return math.log(max(prob, 1e-12))

    def _compute_log_emission(self, choice: ThreadChoice, site_index: int) -> float:
        """
        Evaluates the log-likelihood of the observed mutations at the given site, 
        using Felsenstein's pruning algorithm.
        """
        if not (0 <= site_index < self.config.sequence_length):
            raise IndexError(f"site_index {site_index} is out of bounds for sequence length {self.config.sequence_length}")

        # 1. Get the local tree and attach the focal lineage at the chosen branch/time.
        site_tree = self.site_trees[site_index]
        threaded = _thread_leaf_into_site_tree_full(site_tree, self.focal_leaf, choice)
        
        # 2. Read the observed alleles at this site across all leaves.
        site_observation = self.config.geno[:, site_index]
        
        # 3. Evaluate the site likelihood with the binary Felsenstein recursion
        # implemented in utils.py. This is the emission term
        # log P_emit(D_i | T_i union c_i).
        theta = self.config.mutation_rate
        log_likelihood = _binary_site_log_likelihood(
            edge_index=threaded["edge_index"],
            num_nodes=threaded["num_nodes"],
            root=threaded["root"],
            node_times=threaded["node_times"],
            node_sample_ids=threaded["node_sample_ids"],
            site_observation=site_observation,
            theta=theta
        )
        
        return log_likelihood
