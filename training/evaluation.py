"""Deterministic evaluation helpers for training."""

import random

import numpy as np
import torch

from arg_environment import action_as_dict


def compute_importance_diagnostics(
    log_rewards,
    log_pf,
    log_pb,
    *,
    log_z=None,
):
    """Return stable trajectory-balance and importance-weight diagnostics.

    Normalized importance weights do not depend on logZ.  Calculations use
    float64 because a poorly trained proposal can span hundreds of log units.
    """
    def as_cpu_float64(values):
        return torch.as_tensor(values).detach().to(
            device="cpu", dtype=torch.float64,
        ).reshape(-1)

    rewards = as_cpu_float64(log_rewards)
    forward = as_cpu_float64(log_pf)
    backward = as_cpu_float64(log_pb)
    if not (rewards.numel() == forward.numel() == backward.numel()):
        raise ValueError("log_rewards, log_pf, and log_pb must have equal lengths")
    if rewards.numel() == 0:
        raise ValueError("importance diagnostics require at least one trajectory")
    if not all(torch.isfinite(values).all() for values in (rewards, forward, backward)):
        raise ValueError("importance diagnostics require finite log values")

    log_weights = rewards + backward - forward
    log_normalizer = torch.logsumexp(log_weights, dim=0)
    normalized_weights = torch.exp(log_weights - log_normalizer)
    ess = torch.clamp(
        1.0 / normalized_weights.square().sum(),
        max=float(rewards.numel()),
    )
    metrics = {
        "importance_ess": float(ess),
        "importance_ess_fraction": float(ess / rewards.numel()),
        "importance_max_weight": float(normalized_weights.max()),
        "importance_log_weight_range": float(log_weights.max() - log_weights.min()),
    }
    if log_z is not None:
        residuals = as_cpu_float64(log_z) - log_weights
        metrics.update({
            "tb_mse": float(residuals.square().mean()),
            "residual_mean": float(residuals.mean()),
            "residual_std": float(residuals.std(unbiased=False)),
            "residual_rmse": float(residuals.square().mean().sqrt()),
        })
    return metrics


def evaluate_generator(
    rollout_worker,
    generator,
    episodes,
    seed,
    *,
    metric_prefix="eval_",
):
    if episodes <= 0:
        return {}
    env = rollout_worker.env
    states = _save_rng_state(env)
    try:
        _seed_all(seed, env)
        with torch.no_grad():
            outputs, trajectories = rollout_worker.rollout(generator, episodes=episodes)
            log_pf = outputs["log_paths_pf"].sum(-1)
            log_pb = outputs["log_paths_pb"].sum(-1)
            rewards = [trajectory.log_reward for trajectory in trajectories]
            event_probs = generator.compute_event_probabilities([env.get_initial_state()])

        lengths = torch.tensor([len(trajectory) for trajectory in trajectories]).float()
        counts = {
            event: torch.tensor([
                sum(action_as_dict(action).get("event_type") == event for action in trajectory.actions)
                for trajectory in trajectories
            ]).float()
            for event in ("coal", "recomb")
        }
        metrics = compute_importance_diagnostics(
            rewards,
            log_pf,
            log_pb,
            log_z=generator.compute_log_Z().detach(),
        )
        metrics.update({
            "log_pf_mean": float(log_pf.mean()),
            "log_pb_mean": float(log_pb.mean()),
            "log_reward_mean": float(np.mean(rewards)),
            "trajectory_length_mean": float(lengths.mean()),
            "coalescence_count_mean": float(counts["coal"].mean()),
            "recombination_count_mean": float(counts["recomb"].mean()),
        })
        for source, values in event_probs.items():
            metrics[f"initial_{source}_coalescence_prob"] = float(values[0, 0])
            metrics[f"initial_{source}_recombination_prob"] = float(values[0, 1])
        return {f"{metric_prefix}{name}": value for name, value in metrics.items()}
    finally:
        _restore_rng_state(env, states)


def _seed_all(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env.rng.seed(seed)


def _save_rng_state(env):
    return (
        random.getstate(), np.random.get_state(), torch.random.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        env.rng.getstate(),
    )


def _restore_rng_state(env, states):
    python, numpy, torch_state, cuda, env_state = states
    random.setstate(python)
    np.random.set_state(numpy)
    torch.random.set_rng_state(torch_state)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)
    env.rng.setstate(env_state)
