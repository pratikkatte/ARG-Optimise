import math


class RolloutWorker:
    """Rollout orchestration for the simplified ARG environment."""

    def __init__(self, env, max_steps=100, verbose=False):
        self.env = env
        self.max_steps = max_steps
        self.verbose = verbose

    def sample_action_from_prior(self, state):
        distribution = self.env.compute_action_prior_distribution(state)
        if not distribution:
            return None

        threshold = self.env.rng.random()
        running = 0.0
        for action, log_prior in distribution:
            running += 0.0 if log_prior == -math.inf else math.exp(log_prior)
            if threshold <= running:
                return dict(action), log_prior
        action, log_prior = distribution[-1]
        return dict(action), log_prior

    def rollout(self, max_steps=None):
        state = self.env.get_initial_state()
        trajectory = []
        rollout_steps = self.max_steps if max_steps is None else max_steps

        for step in range(rollout_steps):
            if state.is_done:
                break

            sampled = self.sample_action_from_prior(state)
            if sampled is None:
                break

            action, log_prior = sampled
            state = self.env.apply_action(state, action, log_prior)
            trajectory.append(
                {
                    "step": step,
                    "action": action,
                    "log_prior": log_prior,
                    "active_lineage_count": len(state.active_lineages),
                    "active_counts": self.env.get_active_counts(state).tolist(),
                    "is_done": state.is_done,
                    "log_reward": state.log_reward,
                }
            )

            if self.verbose:
                print(
                    "step={step:02d} action={action} log_prior={log_prior:.4f} "
                    "active={active_lineage_count} done={is_done}".format(**trajectory[-1])
                )

        return state, trajectory
