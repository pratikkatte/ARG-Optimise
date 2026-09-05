import torch
from torch.nn.utils.rnn import pad_sequence

from arg_environment import SimpleTrajectory


class RolloutWorker:
    """Run batched trajectories in the simplified ARG environment."""

    def __init__(self, env, verbose=False):
        self.env = env
        self.device = env.device
        self.verbose = verbose

    def rollout(
        self,
        generator=None,
        episodes=1,
        random_spec=None,
        return_states=False,
    ):
        if generator is None:
            raise ValueError("Generator is required for rollout")
        return self._rollout_batch(
            generator, episodes, random_spec=random_spec,
            return_states=return_states,
        )

    def _rollout_batch(
        self,
        generator,
        episodes,
        random_spec=None,
        return_states=False,
    ):
        states = [self.env.get_initial_state() for _ in range(episodes)]
        trajectories = [SimpleTrajectory() for _ in states]
        forward_logs = [[] for _ in states]
        backward_counts = [[] for _ in states]

        if self.verbose:
            print(f"Rolling out {episodes} trajectory/trajectories in batch...")

        unfinished = self._unfinished_indices(states)
        while unfinished:
            inputs = self.env.prepare_state_rollout_inputs(
                [states[index] for index in unfinished],
                random_spec=random_spec,
            )
            log_pf, actions = generator(inputs)
            self._advance(
                generator, states, trajectories, unfinished,
                inputs["candidate_actions"], actions, log_pf,
                forward_logs, backward_counts,
            )
            unfinished = self._unfinished_indices(states)

        data = self._rollout_data(states, forward_logs, backward_counts)
        if return_states:
            data["states"] = states
        return data, trajectories

    def _advance(
        self,
        generator,
        states,
        trajectories,
        active_indices,
        candidate_actions,
        actions,
        log_pf,
        forward_logs,
        backward_counts,
    ):
        for batch_index, trajectory_index in enumerate(active_indices):
            state = states[trajectory_index]
            action = actions[batch_index]
            log_prior = self.env.compute_cwr_event_log_prior(
                state, candidate_actions[batch_index], action,
            )
            next_state = self.env.apply_action(state, action, log_prior=log_prior)

            states[trajectory_index] = next_state
            trajectories[trajectory_index].update(
                action, log_prior=log_prior, log_reward=next_state.log_reward,
            )
            forward_logs[trajectory_index].append(log_pf[batch_index])
            backward_counts[trajectory_index].append(
                generator.count_backward_parents(next_state)
            )

    def _rollout_data(self, states, forward_logs, backward_counts):
        forward = self._pad([
            torch.stack(values) if values else torch.empty(0, device=self.device)
            for values in forward_logs
        ])
        backward = self._pad([
            -torch.log(torch.as_tensor(values, dtype=torch.float32, device=self.device))
            for values in backward_counts
        ])
        rewards = torch.tensor(
            [state.log_reward for state in states],
            dtype=torch.float32,
            device=self.device,
        )
        return {
            "log_paths_pf": forward,
            "log_paths_pb": backward,
            "log_rewards": rewards,
        }

    def _pad(self, vectors):
        if not vectors:
            return torch.empty(0, 0, dtype=torch.float32, device=self.device)
        vectors = [value.to(dtype=torch.float32, device=self.device) for value in vectors]
        return pad_sequence(vectors, batch_first=True)

    @staticmethod
    def _unfinished_indices(states):
        return [index for index, state in enumerate(states) if not state.is_done]
