"""Deterministic evaluation helpers for training."""

import random

import numpy as np
import torch

from arg_environment import action_as_dict


def evaluate_generator(rollout_worker, generator, episodes, seed):
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
            rewards = outputs["log_rewards"]
            residuals = generator.compute_log_Z().detach().to(log_pf) + log_pf - rewards - log_pb
            event_probs = generator.compute_event_probabilities([env.get_initial_state()])

        lengths = torch.tensor([len(trajectory) for trajectory in trajectories]).float()
        counts = {
            event: torch.tensor([
                sum(action_as_dict(action).get("event_type") == event for action in trajectory.actions)
                for trajectory in trajectories
            ]).float()
            for event in ("coal", "recomb")
        }
        metrics = {
            "eval_tb_mse": residuals.square().mean(),
            "eval_residual_mean": residuals.mean(),
            "eval_residual_std": residuals.std(unbiased=False),
            "eval_log_pf_mean": log_pf.mean(),
            "eval_log_pb_mean": log_pb.mean(),
            "eval_log_reward_mean": rewards.mean(),
            "eval_trajectory_length_mean": lengths.mean(),
            "eval_coalescence_count_mean": counts["coal"].mean(),
            "eval_recombination_count_mean": counts["recomb"].mean(),
        }
        for source, values in event_probs.items():
            metrics[f"eval_initial_{source}_coalescence_prob"] = values[0, 0]
            metrics[f"eval_initial_{source}_recombination_prob"] = values[0, 1]
        return {name: float(value.detach().cpu()) for name, value in metrics.items()}
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

