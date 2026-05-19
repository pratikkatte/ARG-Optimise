import numpy as np
import torch


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

    def _rollout_one(self, generator=None, random_spec=None, max_steps=None):
        state = self.env.get_initial_state()
        trajectory = []
        rollout_steps = self.max_steps if max_steps is None else max_steps

        for step in range(rollout_steps):
            if state.is_done:
                break

            if generator is None:
                sampled = self.sample_action_from_prior(state)
                if sampled is None:
                    break
                action, log_prior = sampled
            else:
                tree_features = self._state_to_tree_features(state)
                input_dict = self.env.prepare_rollout_inputs(tree_features, None, random_spec)
                input_dict["states"] = [state]
                ret = generator(input_dict)
                actions = ret.get("actions", ret.get("arg_actions"))
                if not actions:
                    break
                action = dict(actions[0])
                log_prior = self.env.compute_cwr_event_log_prior(state, action)

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

    def rollout(self, generator=None, episodes=1, max_steps=100, random_spec=None):
        """
        """
        states = []
        trajectories = []
        for _ in range(episodes):
            state, trajectory = self._rollout_one(
                generator=generator,
                random_spec=random_spec,
                max_steps=max_steps,
            )
            states.append(state)
            trajectories.append(trajectory)
        return states, trajectories

    def _state_to_tree_features(self, state):
        seq_arrays = self.env.seq_arrays.float()
        lineage_features = []

        for lineage in state.active_lineages:
            if lineage.sequences_indices:
                feature = seq_arrays[lineage.sequences_indices].mean(dim=0)
            else:
                feature = torch.zeros_like(seq_arrays[0])
            site_mask = self._material_mask_to_site_mask(lineage.material_mask, feature.device)
            lineage_features.append(feature * site_mask[:, None])

        return torch.stack(lineage_features, dim=0).unsqueeze(0)

    def _material_mask_to_site_mask(self, material_mask, device):
        mask = torch.as_tensor(material_mask, dtype=torch.bool, device=device)
        sequence_length = int(self.env.sequence_length)
        if len(mask) == sequence_length:
            return mask.to(dtype=torch.float32)

        site_mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)
        for block_idx, has_material in enumerate(mask.tolist()):
            if not has_material:
                continue
            start = int(round(block_idx * sequence_length / self.env.num_blocks))
            end = int(round((block_idx + 1) * sequence_length / self.env.num_blocks))
            site_mask[start:end] = True
        return site_mask.to(dtype=torch.float32)
