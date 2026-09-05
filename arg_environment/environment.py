import random
from typing import Any, Optional, Sequence

import numpy as np
import torch

from model.evolution import EvolutionModelTorch
from .time import TimeEnvFixedDelta
from .export import ExportMixin
from .material import MaterialSegments
from .priors import PriorMixin
from .state import ARGLineage, ARGReward, ARGState
from .transitions import TransitionMixin

CHARACTERS_MAPS = {
    "DNA_WITH_GAP": {
        "A": [1., 0., 0., 0.], "C": [0., 1., 0., 0.],
        "G": [0., 0., 1., 0.], "T": [0., 0., 0., 1.],
        "-": [1., 1., 1., 1.], "N": [1., 1., 1., 1.],
    }
}


class SimpleARGEnvironment(ExportMixin, TransitionMixin, PriorMixin):
    """
    Minimal discrete coalescent-with-recombination ARG prototype.

    This intentionally avoids eete3, continuous breakpoints, and full continuous
    coalescent-with-recombination simulation. Terminal states are rewarded by the
    canonical CWR prior plus a learned-time JC69 sequence likelihood.
    """

    def __init__(
        self,
        num_sequences: Optional[int] = None,
        sequence_length: Optional[int] = None,
        num_blocks: Optional[int] = None,
        population_size: float = 10000.0,
        effective_population_size: Optional[float] = None,
        mutation_rate: float = 2e-8,
        recombination_rate: float = 2e-8,
        rho: Optional[float] = None,
        sequences: Optional[Sequence[Any]] = None,
        seed: Optional[int] = 7,
        bp_per_blocks: int = 1,
        device: Optional[torch.device] = 'cpu',
        time_bins: Optional[int] = None,
        time_delta_bin_width: Optional[float] = None,
    ):
        self.sequences = list(sequences) if sequences is not None else None
        self.chars_dict = CHARACTERS_MAPS['DNA_WITH_GAP']
        self.event_types = ["coal", "recomb"]
        self.device = torch.device(device)

        if self.sequences is not None:
            num_sequences = len(self.sequences)
            sequence_length = len(self.sequences[0])
            if any(len(sequence) != sequence_length for sequence in self.sequences):
                raise ValueError("all sequences must have length sequence_length")


        self.num_sequences = int(num_sequences)
        self.sequence_length = int(sequence_length)
        if num_blocks is None:
            self.num_blocks = int(sequence_length // bp_per_blocks)
        else:
            self.num_blocks = int(num_blocks)
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        ## Important parameters
        self.recombination_rate = float(recombination_rate)
        if effective_population_size is not None:
            population_size = effective_population_size
        self.population_size = float(population_size)
        self.mutation_rate = float(mutation_rate) ## where are we using this?

        self.rho = (
            float(rho)
            if rho is not None
            else 4 * self.population_size * self.recombination_rate * self.sequence_length
        )

        ## Time environment
        time_env_kwargs = {}
        if time_bins is not None:
            time_env_kwargs["bins"] = int(time_bins)
        if time_delta_bin_width is not None:
            time_env_kwargs["delta_bin_width"] = float(time_delta_bin_width)
        self.time_env = TimeEnvFixedDelta(**time_env_kwargs)

        self.rng = random.Random(seed)

        ## Sequence arrays
        self.block_indices = np.arange(self.num_blocks)

        seq_arrays = np.array([self.seq2array(seq) for seq in self.sequences], dtype=np.float32)

        block_seq_arrays = np.empty(
            (self.num_sequences, self.num_blocks, seq_arrays.shape[-1]),
            dtype=np.float32,
        )
        for block_idx in range(self.num_blocks):
            site_start = int(round(block_idx * self.sequence_length / self.num_blocks))
            site_end = int(round((block_idx + 1) * self.sequence_length / self.num_blocks))
            if site_end <= site_start:
                raise ValueError(
                    "num_blocks must not create empty block intervals for sequence_length"
                )
            block_seq_arrays[:, block_idx, :] = seq_arrays[:, site_start:site_end, :].mean(axis=1)

        self.seq_arrays = torch.nn.Parameter(
            torch.tensor(seq_arrays, dtype=torch.float32, device=self.device),
            requires_grad=False,
        )
        self.block_seq_arrays = torch.nn.Parameter(
            torch.tensor(block_seq_arrays, dtype=torch.float32, device=self.device),
            requires_grad=False,
        )
        
        ## Evolution model
        self.evolution_model = EvolutionModelTorch(self)

        ## Reward function 
        self.reward_fn = ARGReward()

    @property
    def time_metadata(self):
        return {
            "time_bin_scheme": type(self.time_env).__name__,
            "time_bins": int(self.time_env.bins),
            "time_delta_bin_width": float(self.time_env.delta_bin_width),
        }

    def seq2array(self, seq):
        seq = [self.chars_dict[x] for x in seq]
        data = np.array(seq)
        return data

    def _total_event_rate(self, rates):
        total_rate = float(rates["lambda_coal"] + rates["lambda_recomb"])
        if total_rate <= 0 or total_rate is None:
            raise ValueError("waiting-time rate must be positive")
        return total_rate

    def get_initial_state(self):
        active_lineages = []
        all_nodes = {}
        material_segments = MaterialSegments.full(self.num_blocks)
        material_segments_list = [material_segments] * self.num_sequences
        partials_list = self._initial_lineages_partials_batch(material_segments_list)

        total_time = 0.0
        for node_id in range(self.num_sequences):
            # Here, each lineage starts at time 0.0
            lineage = ARGLineage(
                node_id=node_id,
                children=[],
                parents=[],
                material_segments=material_segments,
                num_blocks=self.num_blocks,
                partials=partials_list[node_id],
                sequences_indices=[node_id],
                time=0.0,
            )
            total_time += lineage.time
            active_lineages.append(lineage)
            all_nodes[node_id] = lineage
     

        state = ARGState(
            active_lineages=active_lineages,
            all_nodes=all_nodes,
            max_node_idx=self.num_sequences - 1,
            log_reward=None,
            accumulated_log_prior=0.0,
            is_done=False,
            total_active_blocks=self.num_sequences * self.num_blocks,
            current_time=0.0,
        )
        state.is_done = self.is_terminal(state)
        if state.is_done:
            log_likelihood = self.evolution_model.compute_arg_log_likelihood(state)
            state.log_reward = self.compute_terminal_log_reward(state, log_likelihood)
        return state

    def _initial_lineage_partials(self, node_id, material_segments):
        partials = self.block_seq_arrays[int(node_id)].detach().clone().float()
        return self.evolution_model.mask_partials(partials, material_segments)

    def _initial_lineages_partials_batch(self, material_segments_list):
        """Initialize tip partials for all sequences in one vectorized pass."""
        num_lineages = len(material_segments_list)
        if num_lineages != self.num_sequences:
            raise ValueError(
                f"Expected {self.num_sequences} material segment sets, got {num_lineages}"
            )

        reference_segments = material_segments_list[0]
        segments_match = all(
            ms.segments == reference_segments.segments for ms in material_segments_list
        )
        if segments_match:
            return [
                self._initial_lineage_partials(node_id, reference_segments)
                for node_id in range(num_lineages)
            ]

        return [
            self._initial_lineage_partials(node_id, material_segments)
            for node_id, material_segments in enumerate(material_segments_list)
        ]

    def _require_lineage_partials(self, lineage):
        if lineage.partials is None:
            raise ValueError(f"ARG lineage {lineage.node_id} is missing partials")
        return self.evolution_model._as_partials_tensor(lineage.partials)

    def _transition_lineage_partials(self, lineage, parent_time):
        edge_time = float(parent_time) - float(lineage.time)
        if edge_time <= 0:
            raise ValueError(
                f"ARG node times must increase from child to parent: "
                f"parent_time={parent_time}, child={lineage.node_id} time={lineage.time}"
            )
        partials = self._require_lineage_partials(lineage)
        return self.evolution_model.transition_partials(partials, edge_time)

    def _parent_partials(self, lineage, parent_time):
        if parent_time is None:
            return self._require_lineage_partials(lineage)
        return self._transition_lineage_partials(lineage, parent_time)

    def _coalesced_parent_partials(self, child_i, child_j, parent_segments, parent_time):
        reference = self._require_lineage_partials(child_i)
        combined = torch.ones_like(reference)
        has_material = torch.zeros(
            reference.shape[0],
            1,
            dtype=torch.bool,
            device=reference.device,
        )

        for child in (child_i, child_j):
            transitioned = self._parent_partials(child, parent_time)
            transitioned = self.evolution_model.normalize_partials(transitioned)
            weights = self.evolution_model.material_site_weights(
                child.material_segments,
                device=transitioned.device,
                dtype=transitioned.dtype,
            )
            child_has_material = weights[:, None] > 0
            child_partials = transitioned * weights[:, None]
            combined = torch.where(child_has_material, combined * child_partials, combined)
            has_material = has_material | child_has_material

        combined = torch.where(has_material, combined, torch.zeros_like(combined))
        combined = self.evolution_model.mask_partials(combined, parent_segments)
        return self.evolution_model.normalize_partials(combined)

    def _recombined_parent_partials(self, child, parent_segments, parent_time):
        transitioned = self._parent_partials(child, parent_time)
        masked = self.evolution_model.mask_partials(transitioned, parent_segments)
        return self.evolution_model.normalize_partials(masked)

    def _coalescence_preview_features(self, child_i, child_j):
        """Pool evidence without assuming a branch time or multiplying partials.

        Shared blocks contain the mean child evidence, with separate absolute
        differences and an overlap indicator. Single-child blocks retain that
        child's evidence; blocks carried by neither child remain zero. These are
        policy features, not parent likelihood partials.
        """
        evidence = []
        masks = []
        for child in (child_i, child_j):
            partials = self.evolution_model.normalize_partials(
                self._require_lineage_partials(child)
            )
            mask = self.evolution_model.material_site_weights(
                child.material_segments, device=partials.device, dtype=partials.dtype,
            )[:, None]
            evidence.append(partials * mask)
            masks.append(mask)
        overlap = masks[0] * masks[1]
        mean_evidence = (evidence[0] + evidence[1]) / (masks[0] + masks[1]).clamp_min(1.0)
        disagreement = (evidence[0] - evidence[1]).abs() * overlap
        return mean_evidence, torch.cat([disagreement, overlap], dim=-1)

    def get_active_counts(self, state):
        if not state.active_lineages:
            return np.zeros(self.num_blocks, dtype=int)
        counts = np.zeros(self.num_blocks, dtype=int)
        for lineage in state.active_lineages:
            for start, end in lineage.material_segments.segments:
                counts[start:end] += 1
        return counts
