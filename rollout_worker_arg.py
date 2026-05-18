import math
import numbers
import numpy as np


class RolloutWorker:
    """Rollout orchestration for the simplified ARG environment."""

    def __init__(self, env, max_steps=100, verbose=False):
        self.env = env
        self.max_steps = max_steps
        self.verbose = verbose

    def sample_action_from_prior(self, state):
        
        probs = self.env.compute_event_probabilities(state)

        chosen_event = np.random.choice(list(probs.keys()), p=list(probs.values()))

        distribution = self.env.compute_action_prior_distribution(state, chosen_event)
        if not distribution:
            return None
        actions, log_priors = zip(*distribution)
        priors = np.array(log_priors)
        max_log = np.max(priors)
        weights = np.exp(priors - max_log)
        probs = weights / weights.sum()
        chosen_idx = np.random.choice(np.arange(len(actions)), p=probs)
        chosen_action = actions[chosen_idx]
        chosen_log_prior = log_priors[chosen_idx]
        return dict(chosen_action), chosen_log_prior

    def _rollout_one(self, max_steps=None):
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

    def rollout(self, generator, num_trajectories=1, max_steps=100):
        """
        """

        seq_arrays = self.env.seq_arrays

        # initial tree features
        tree_features = seq_arrays.unsqueeze(0)
        tree_features = tree_features.repeat(num_trajectories, 1, 1, 1)
        input_dict = self.env.prepare_rollout_inputs(tree_features, None, None)

        ret = generator(input_dict)

        states = []
        trajectories = []
        for _ in range(num_trajectories):
            state, trajectory = self._rollout_one(max_steps=max_steps)
            states.append(state)
            trajectories.append(trajectory)
        return states, trajectories
