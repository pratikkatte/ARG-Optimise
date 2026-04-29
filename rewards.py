import math
import torch
from dataclasses import dataclass
from typing import Optional, Sequence
from utils import (
    ThreadPathState, 
    ThreadChoice,
    _binary_site_log_likelihood,
    _children_from_edge_index,
    _choice_prior_weights,
    graph_node_sample_ids,
    graph_root,
    thread_leaf_into_graph,
)

NUC2VEC = {
    "A": [1.0, 0.0, 0.0, 0.0],
    "G": [0.0, 1.0, 0.0, 0.0],
    "C": [0.0, 0.0, 1.0, 0.0],
    "T": [0.0, 0.0, 0.0, 1.0],
    "-": [1.0, 1.0, 1.0, 1.0],
    "?": [1.0, 1.0, 1.0, 1.0],
    "N": [1.0, 1.0, 1.0, 1.0],
    "R": [1.0, 1.0, 0.0, 0.0],
    "Y": [0.0, 0.0, 1.0, 1.0],
    "S": [0.0, 1.0, 1.0, 0.0],
    "W": [1.0, 0.0, 0.0, 1.0],
    "K": [0.0, 1.0, 0.0, 1.0],
    "M": [1.0, 0.0, 1.0, 0.0],
    "B": [0.0, 1.0, 1.0, 1.0],
    "D": [1.0, 1.0, 0.0, 1.0],
    "H": [1.0, 0.0, 1.0, 1.0],
    "V": [1.0, 1.0, 1.0, 0.0],
    ".": [1.0, 1.0, 1.0, 1.0],
    "U": [0.0, 0.0, 0.0, 1.0],
}


@dataclass(frozen=True)
class LocalRewardBreakdown:
    log_transition: float
    log_emission: float
    log_reward: float

    @property
    def unscaled_log_reward(self) -> float:
        return self.log_transition + self.log_emission


def _jc_decomposition(dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rate_matrix = torch.full((4, 4), 1.0 / 3.0, dtype=dtype)
    rate_matrix.fill_diagonal_(-1.0)
    eigenvalues, eigenvectors = torch.linalg.eigh(rate_matrix)
    return eigenvalues, eigenvectors, eigenvectors.t()


def _jc_transition_matrix(branch_length: float, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    eigenvalues, eigenvectors, eigenvectors_inv = _jc_decomposition(dtype=dtype)
    branch_d = torch.exp(eigenvalues * float(branch_length))
    transition = torch.matmul(eigenvectors * branch_d.unsqueeze(0), eigenvectors_inv)
    return transition.clamp_min(torch.finfo(dtype).tiny)


def _iupac_log_mask(base: str) -> torch.Tensor:
    try:
        allowed = torch.tensor(NUC2VEC[base], dtype=torch.float64)
    except KeyError as exc:
        raise KeyError(base) from exc
    return torch.where(
        allowed > 0.0,
        torch.zeros_like(allowed),
        torch.full_like(allowed, -torch.inf),
    )


def nucleotide_loglikelihood(
    edge_index: torch.Tensor,
    num_nodes: int,
    root: int,
    node_times: torch.Tensor,
    node_sample_ids: Sequence[int],
    fasta_sequences: Sequence[str],
    site_index: int,
    theta: float,
    substitution_model: str = "JC",
) -> float:
    if substitution_model.upper() != "JC":
        raise ValueError(
            "Only the JC substitution model is currently supported for FASTA rewards."
        )

    children = _children_from_edge_index(edge_index, num_nodes)
    log_root_prior = torch.full((4,), math.log(0.25), dtype=torch.float64)
    log_transition_by_child: dict[int, torch.Tensor] = {}
    for parent, child in edge_index.t().tolist():
        branch_time = float(node_times[parent].item() - node_times[child].item())
        branch_length = float(theta) * max(branch_time, 1e-6)
        log_transition_by_child[child] = torch.log(_jc_transition_matrix(branch_length))

    def postorder(node: int) -> torch.Tensor:
        sample_id = node_sample_ids[node]
        if sample_id >= 0:
            base = fasta_sequences[sample_id][site_index].upper()
            try:
                return _iupac_log_mask(base)
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported FASTA symbol {base!r} for sample {sample_id} "
                    f"at site {site_index}"
                ) from exc

        node_log_like = torch.zeros(4, dtype=torch.float64)
        for child in children[node]:
            child_like = postorder(child)
            log_transition = log_transition_by_child[child]
            contrib = torch.logsumexp(log_transition + child_like.unsqueeze(0), dim=1)
            node_log_like = node_log_like + contrib
        return node_log_like

    root_like = postorder(root)
    return float(torch.logsumexp(log_root_prior + root_like, dim=0).item())


def nucleotide_logp_joint(
    edge_index: torch.Tensor,
    num_nodes: int,
    root: int,
    node_times: torch.Tensor,
    node_sample_ids: Sequence[int],
    fasta_sequences: Sequence[str],
    site_index: int,
    theta: float,
    substitution_model: str = "JC",
    scale: float = 0.1,
) -> float:
    if scale <= 0.0:
        raise ValueError("scale must be strictly positive.")

    log_likelihood = nucleotide_loglikelihood(
        edge_index=edge_index,
        num_nodes=num_nodes,
        root=root,
        node_times=node_times,
        node_sample_ids=node_sample_ids,
        fasta_sequences=fasta_sequences,
        site_index=site_index,
        theta=theta,
        substitution_model=substitution_model,
    )
    branch_lengths = []
    for parent, child in edge_index.t().tolist():
        branch_time = float(node_times[parent].item() - node_times[child].item())
        branch_lengths.append(max(branch_time, 1e-6))
    log_prior = -sum(
        branch_length / scale + math.log(scale) - math.log(branch_length)
        for branch_length in branch_lengths
    )
    return log_prior + log_likelihood


def loglikelihood(*args, **kwargs) -> float:
    return nucleotide_loglikelihood(*args, **kwargs)


def logp_joint(*args, **kwargs) -> float:
    return nucleotide_logp_joint(*args, **kwargs)


logp_joing = logp_joint


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

        return self.compute_local_reward_breakdown(
            prev_choice, curr_choice, site_index
        ).log_reward

    def compute_local_reward_breakdown(
        self,
        prev_choice: Optional[ThreadChoice],
        curr_choice: ThreadChoice,
        site_index: int,
    ) -> LocalRewardBreakdown:
        if self.config.reward_temperature <= 0.0:
            raise ValueError("reward_temperature must be strictly positive.")

        log_trans = self._compute_log_transition(prev_choice, curr_choice, site_index)
        log_emit = self._compute_log_emission(curr_choice, site_index)
        log_reward = (log_trans + log_emit) / self.config.reward_temperature
        return LocalRewardBreakdown(
            log_transition=log_trans,
            log_emission=log_emit,
            log_reward=log_reward,
        )
    
    def compute_log_path_reward(self, choices: Sequence[ThreadChoice]) -> float:
        """
        Calculates the unnormalized log-posterior of a complete threading path.

        The Markov state intentionally keeps only the current choice, so complete
        path rewards are computed from the explicit trajectory trace.
        """
        expected_length = self.config.sequence_length
        if len(choices) != expected_length:
            raise ValueError(
                "The sequence must be fully threaded to compute reward; "
                f"got {len(choices)} choices for length {expected_length}."
            )
        if self.config.reward_temperature <= 0.0:
            raise ValueError("reward_temperature must be strictly positive.")

        total_log_reward = 0.0
        prev_choice = None

        for i, curr_choice in enumerate(choices):
            total_log_reward += self.compute_log_local_reward(prev_choice, curr_choice, i)
            prev_choice = curr_choice

        return total_log_reward

    def compute_log_reward(self, st: ThreadPathState) -> float:
        """
        Backward-compatible terminal reward helper using the environment trace.
        """
        expected_length = self.config.sequence_length
        if st.site_index != expected_length:
            raise ValueError(
                "State is not terminal. The sequence must be fully threaded to compute reward."
            )
        return self.compute_log_path_reward(self.episode_choices)

    def _compute_log_transition(self, prev_choice: Optional[ThreadChoice], curr_choice: ThreadChoice, site_index: int) -> float:
        """
        Computes the SMC prior log transition probability.
        """
        if not (0 <= site_index < self.config.sequence_length):
            raise IndexError(f"site_index {site_index} is out of bounds for sequence length {self.config.sequence_length}")

        # Get prior probability for the current choice
        curr_choices_at_site = self._choices_for_site(site_index)
        curr_prior_weights = _choice_prior_weights(curr_choices_at_site)
        curr_choice_idx = self._choice_to_index_for_site(site_index)[curr_choice]
        prior_prob = float(curr_prior_weights[curr_choice_idx].item())
        
        if prev_choice is None:
            # site_index == 0: compute the stationary coalescent prior \log P(c_1)
            return math.log(max(prior_prob, 1e-12))
        
        # Sequentially Markovian Coalescent (SMC) prior
        site_distance = float(self.config.site_distance(site_index))
        recomb_mass = 1.0 - math.exp(-self.config.recomb_rate * site_distance)
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
        site_graph = self._site_graph_for_site(site_index)
        threaded = thread_leaf_into_graph(site_graph, self.focal_leaf, choice)
        
        theta = self.config.mutation_rate
        if self.config.fasta_sequences is not None:
            return nucleotide_loglikelihood(
                edge_index=threaded.edge_index,
                num_nodes=int(threaded.num_nodes),
                root=graph_root(threaded),
                node_times=threaded.node_times,
                node_sample_ids=graph_node_sample_ids(threaded),
                fasta_sequences=self.config.fasta_sequences,
                site_index=site_index,
                theta=theta,
                substitution_model=self.config.substitution_model,
            )

        # 2. Read the observed alleles at this site across all leaves.
        site_observation = self.config.geno[:, site_index]

        # 3. Evaluate the site likelihood with the binary Felsenstein recursion
        # implemented in utils.py. This is the emission term
        # log P_emit(D_i | T_i union c_i).
        log_likelihood = _binary_site_log_likelihood(
            edge_index=threaded.edge_index,
            num_nodes=int(threaded.num_nodes),
            root=graph_root(threaded),
            node_times=threaded.node_times,
            node_sample_ids=graph_node_sample_ids(threaded),
            site_observation=site_observation,
            theta=theta
        )
        
        return log_likelihood
