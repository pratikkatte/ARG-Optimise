from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from env import ARGweaverThreadEnv, SingleLeafThreadEnv
from loss import TrajectoryBalanceLoss
from models import Policy
from utils import MultiLeafState, ThreadChoice, _choice_prior_weights


def _choice_key(choice: ThreadChoice) -> tuple[int, tuple[int, ...], int, bool]:
    return (
        choice.branch_child,
        choice.branch_signature,
        choice.time_idx,
        choice.is_root_branch,
    )


@dataclass(frozen=True)
class Trajectory:
    """One sampled single-leaf threading trajectory."""

    focal_leaf: int
    actions: Tuple[int, ...]
    choices: Tuple[ThreadChoice, ...]
    states: Tuple[MultiLeafState, ...]
    site_indices: Tuple[int, ...]
    valid_action_counts: Tuple[int, ...]
    local_log_rewards: Tuple[float, ...]
    sum_log_pf: Tensor
    sum_log_pb: Tensor
    log_reward: Tensor
    terminal_state: MultiLeafState

    @property
    def length(self) -> int:
        return len(self.actions)

    @property
    def recomb_count(self) -> int:
        recomb_count = 0
        for prev_choice, curr_choice in zip(self.choices, self.choices[1:]):
            if _choice_key(prev_choice) != _choice_key(curr_choice):
                recomb_count += 1
        return recomb_count

    @property
    def total_local_log_reward(self) -> float:
        return float(sum(self.local_log_rewards))


class DeterministicBackwardPolicy(nn.Module):
    """Backward policy for deterministic, unique-parent trajectories."""

    def forward(
        self,
        steps: int | Sequence[ThreadChoice],
        *,
        reference: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **_: object,
    ) -> Tensor:
        num_steps = steps if isinstance(steps, int) else len(steps)
        if num_steps < 0:
            raise ValueError(f"num_steps must be non-negative, got {num_steps}.")
        if reference is not None:
            return reference.new_zeros(())
        return torch.zeros((), device=device, dtype=dtype or torch.float32)

    def sum_log_prob(
        self,
        steps: int | Sequence[ThreadChoice],
        *,
        reference: Optional[Tensor] = None,
        **kwargs: object,
    ) -> Tensor:
        return self.forward(steps, reference=reference, **kwargs)


class BiologicalBackwardPolicy(nn.Module):
    """Bayes-reverse SMC backward policy based on recombination rates."""

    def forward(
        self,
        choices: Sequence[ThreadChoice],
        *,
        inner_env: ARGweaverThreadEnv,
        reference: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        if reference is not None:
            out_device = reference.device
            out_dtype = reference.dtype
        else:
            out_device = device
            out_dtype = dtype or torch.float32

        total_log_pb = 0.0
        for site_index in range(1, len(choices)):
            total_log_pb += self._step_log_prob(
                inner_env,
                site_index=site_index,
                actual_prev=choices[site_index - 1],
                curr_choice=choices[site_index],
            )
        return torch.tensor(total_log_pb, device=out_device, dtype=out_dtype)

    def sum_log_prob(
        self,
        choices: Sequence[ThreadChoice],
        *,
        inner_env: ARGweaverThreadEnv,
        reference: Optional[Tensor] = None,
    ) -> Tensor:
        return self.forward(choices, inner_env=inner_env, reference=reference)

    def _step_log_prob(
        self,
        inner_env: ARGweaverThreadEnv,
        *,
        site_index: int,
        actual_prev: ThreadChoice,
        curr_choice: ThreadChoice,
    ) -> float:
        prev_choices = inner_env.site_choices[site_index - 1]
        curr_choices = inner_env.site_choices[site_index]
        prev_prior = _choice_prior_weights(prev_choices)
        curr_prior = _choice_prior_weights(curr_choices)

        prev_index = inner_env.choice_to_index[site_index - 1][actual_prev]
        curr_index = inner_env.choice_to_index[site_index][curr_choice]
        curr_prior_prob = float(curr_prior[curr_index].item())

        site_distance = float(inner_env.config.site_distance(site_index))
        recomb_mass = 1.0 - math.exp(-inner_env.config.recomb_rate * site_distance)
        stay_mass = 1.0 - recomb_mass
        curr_key = _choice_key(curr_choice)

        log_weights = []
        for prev_idx, prev_choice in enumerate(prev_choices):
            trans_prob = recomb_mass * curr_prior_prob
            if _choice_key(prev_choice) == curr_key:
                trans_prob += stay_mass
            log_weights.append(
                math.log(max(float(prev_prior[prev_idx].item()), 1e-12))
                + math.log(max(trans_prob, 1e-12))
            )

        weights_t = torch.tensor(log_weights, dtype=torch.float64)
        numerator = weights_t[prev_index]
        denominator = torch.logsumexp(weights_t, dim=0)
        return float((numerator - denominator).item())


class TrajectorySampler:
    """Rolls out trajectories from ``SingleLeafThreadEnv`` with a forward policy."""

    def __init__(
        self,
        env: SingleLeafThreadEnv,
        forward_policy: Policy,
        backward_policy: Optional[nn.Module] = None,
        *,
        reset_to_reference: bool = True,
    ) -> None:
        self.env = env
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy or BiologicalBackwardPolicy()
        self.reset_to_reference = bool(reset_to_reference)

    @property
    def device(self) -> torch.device:
        return self.forward_policy.device

    def _resolve_reset_to_reference(
        self,
        reset_to_reference: Optional[bool],
        carry_forward: bool,
    ) -> bool:
        if carry_forward and reset_to_reference:
            raise ValueError("carry_forward=True conflicts with reset_to_reference=True.")
        if carry_forward:
            return False
        if reset_to_reference is None:
            return self.reset_to_reference
        return bool(reset_to_reference)

    def _reset_env(
        self,
        focal_leaf: Optional[int],
        *,
        reset_to_reference: Optional[bool],
        carry_forward: bool,
    ) -> MultiLeafState:
        stable_reset = self._resolve_reset_to_reference(
            reset_to_reference, carry_forward
        )
        graph_segments = self.env.reference_graph_segments if stable_reset else None
        return self.env.reset(focal_leaf=focal_leaf, graph_segments=graph_segments)

    def sample(
        self,
        *,
        focal_leaf: Optional[int] = None,
        temperature: float = 1.0,
        greedy: bool = False,
        reset_to_reference: Optional[bool] = None,
        carry_forward: bool = False,
    ) -> Trajectory:
        if temperature <= 0.0:
            raise ValueError("temperature must be strictly positive.")

        state = self._reset_env(
            focal_leaf,
            reset_to_reference=reset_to_reference,
            carry_forward=carry_forward,
        )
        if state.current_focal_leaf is None:
            raise RuntimeError("Single-leaf reset did not provide a focal leaf.")

        actions: list[int] = []
        choices: list[ThreadChoice] = []
        states: list[MultiLeafState] = [state]
        site_indices: list[int] = []
        valid_action_counts: list[int] = []
        local_log_rewards: list[float] = []
        forward_log_probs: list[Tensor] = []

        while not self.env.is_terminal(state):
            inner_env = self.env._inner_env
            inner_state = state.inner_state
            if inner_env is None or inner_state is None:
                raise RuntimeError("Single-leaf env is missing its active inner state.")

            valid_actions = self.env.valid_actions(state)
            if not valid_actions:
                raise RuntimeError(
                    f"No valid actions at site {inner_state.site_index}."
                )
            action_choices = tuple(
                inner_env.site_choices[inner_state.site_index][action_idx]
                for action_idx in valid_actions
            )
            site_tree = inner_env._site_tree_for_encoding(inner_state)
            action_logits, _action_probs, *_ = self.forward_policy(
                site_tree,
                int(state.current_focal_leaf),
                action_choices,
            )
            scaled_logits = action_logits / float(temperature)
            log_probs = torch.log_softmax(scaled_logits, dim=0)
            if greedy:
                selected_row = torch.argmax(scaled_logits.detach())
            else:
                dist = torch.distributions.Categorical(logits=scaled_logits)
                selected_row = dist.sample()

            selected_idx = int(selected_row.item())
            selected_action = valid_actions[selected_idx]
            selected_choice = action_choices[selected_idx]
            selected_log_prob = log_probs[selected_row]

            next_state, local_reward, _done = self.env.step(state, selected_action)

            actions.append(selected_action)
            choices.append(selected_choice)
            site_indices.append(int(inner_state.site_index))
            valid_action_counts.append(len(valid_actions))
            local_log_rewards.append(float(local_reward))
            forward_log_probs.append(selected_log_prob)
            state = next_state
            states.append(state)

        if forward_log_probs:
            sum_log_pf = torch.stack(forward_log_probs).sum()
        else:
            sum_log_pf = self.forward_policy.log_Z.new_zeros(())

        inner_env = self.env._inner_env
        if inner_env is None or state.inner_state is None:
            raise RuntimeError("Terminal single-leaf env is missing its inner state.")
        choice_trace = tuple(choices)
        sum_log_pb = self.backward_policy.sum_log_prob(
            choice_trace,
            inner_env=inner_env,
            reference=sum_log_pf,
        )
        log_reward = sum_log_pf.new_tensor(
            inner_env.compute_log_path_reward(choice_trace)
        )

        return Trajectory(
            focal_leaf=int(state.current_focal_leaf),
            actions=tuple(actions),
            choices=tuple(choices),
            states=tuple(states),
            site_indices=tuple(site_indices),
            valid_action_counts=tuple(valid_action_counts),
            local_log_rewards=tuple(local_log_rewards),
            sum_log_pf=sum_log_pf,
            sum_log_pb=sum_log_pb,
            log_reward=log_reward,
            terminal_state=state,
        )

    def sample_batch(
        self,
        batch_size: int,
        *,
        focal_leaf: Optional[int] = None,
        temperature: float = 1.0,
        greedy: bool = False,
        reset_to_reference: Optional[bool] = None,
        carry_forward: bool = False,
    ) -> Tuple[Trajectory, ...]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        return tuple(
            self.sample(
                focal_leaf=focal_leaf,
                temperature=temperature,
                greedy=greedy,
                reset_to_reference=reset_to_reference,
                carry_forward=carry_forward,
            )
            for _ in range(batch_size)
        )


def estimate_log_reward_center(
    sampler: TrajectorySampler,
    *,
    num_trajectories: int,
    focal_leaves: Optional[Sequence[int]] = None,
    temperature: float = 1.0,
    reset_to_reference: Optional[bool] = True,
) -> float:
    """Estimate the constant reward shift that makes initial TB residuals small."""
    if num_trajectories <= 0:
        raise ValueError(f"num_trajectories must be positive, got {num_trajectories}.")
    centers: list[float] = []
    for sample_idx in range(num_trajectories):
        focal_leaf = None
        if focal_leaves:
            focal_leaf = int(focal_leaves[sample_idx % len(focal_leaves)])
        trajectory = sampler.sample(
            focal_leaf=focal_leaf,
            temperature=temperature,
            reset_to_reference=reset_to_reference,
        )
        center = (
            trajectory.log_reward
            - trajectory.sum_log_pf.detach()
            + trajectory.sum_log_pb.detach()
        )
        centers.append(float(center.detach().cpu().item()))
    return float(sum(centers) / len(centers))


class GFlowNetTrainer:
    """Small trainer for single-leaf trajectory-balance updates."""

    def __init__(
        self,
        env: SingleLeafThreadEnv,
        forward_policy: Policy,
        optimizer: torch.optim.Optimizer,
        *,
        backward_policy: Optional[nn.Module] = None,
        loss_fn: Optional[TrajectoryBalanceLoss] = None,
        sampler: Optional[TrajectorySampler] = None,
        reset_to_reference: bool = True,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        self.env = env
        self.forward_policy = forward_policy
        self.optimizer = optimizer
        self.backward_policy = backward_policy or BiologicalBackwardPolicy()
        self.loss_fn = loss_fn or TrajectoryBalanceLoss()
        self.max_grad_norm = max_grad_norm
        self.sampler = sampler or TrajectorySampler(
            env,
            forward_policy,
            self.backward_policy,
            reset_to_reference=reset_to_reference,
        )

    def sample_batch(self, *args, **kwargs) -> Tuple[Trajectory, ...]:
        return self.sampler.sample_batch(*args, **kwargs)

    def train_step(
        self,
        batch_size: int,
        *,
        focal_leaf: Optional[int] = None,
        temperature: float = 1.0,
        greedy: bool = False,
        reset_to_reference: Optional[bool] = None,
        carry_forward: bool = False,
    ) -> dict[str, float | Tuple[Trajectory, ...]]:
        trajectories = self.sample_batch(
            batch_size,
            focal_leaf=focal_leaf,
            temperature=temperature,
            greedy=greedy,
            reset_to_reference=reset_to_reference,
            carry_forward=carry_forward,
        )

        sum_log_pf = torch.stack([traj.sum_log_pf for traj in trajectories])
        sum_log_pb = torch.stack([traj.sum_log_pb for traj in trajectories])
        log_reward = torch.stack([traj.log_reward for traj in trajectories])

        loss = self.loss_fn(
            self.forward_policy.log_Z,
            sum_log_pf,
            log_reward,
            sum_log_pb,
        )

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = None
        if self.max_grad_norm is not None:
            grad_norm_t = torch.nn.utils.clip_grad_norm_(
                self.forward_policy.parameters(),
                max_norm=float(self.max_grad_norm),
            )
            grad_norm = float(grad_norm_t.detach().cpu().item())
        self.optimizer.step()

        with torch.no_grad():
            tb_error = self.loss_fn.tb_error(
                self.forward_policy.log_Z,
                sum_log_pf,
                log_reward,
                sum_log_pb,
            )
            lengths = torch.tensor(
                [traj.length for traj in trajectories], dtype=torch.float32
            )
            recomb_counts = torch.tensor(
                [traj.recomb_count for traj in trajectories], dtype=torch.float32
            )

        metrics = {
            "loss": float(loss.detach().item()),
            "mean_log_reward": float(log_reward.detach().mean().item()),
            "mean_centered_log_reward": float(
                (log_reward.detach() - self.loss_fn.log_reward_center).mean().item()
            ),
            "mean_sum_log_pf": float(sum_log_pf.detach().mean().item()),
            "mean_sum_log_pb": float(sum_log_pb.detach().mean().item()),
            "mean_tb_error": float(tb_error.mean().item()),
            "rms_tb_error": float(tb_error.square().mean().sqrt().item()),
            "mean_trajectory_length": float(lengths.mean().item()),
            "mean_recomb_count": float(recomb_counts.mean().item()),
            "log_Z": float(self.forward_policy.log_Z.detach().item()),
            "log_reward_center": float(self.loss_fn.log_reward_center),
            "trajectories": trajectories,
        }
        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm
        return metrics
