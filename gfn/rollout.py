"""Compatible policy sampling, forced replay, and bounded-memory score gradients."""
import math
import torch
from torch.nn.utils.rnn import pad_sequence
from env.env import SimpleTrajectory
from utils import action_as_dict


class RolloutFailure(RuntimeError):
    def __init__(self, message, trajectories):
        self.histories = [[action_as_dict(a) for a in t.actions] for t in trajectories]
        super().__init__(message+'; all attempted action histories are available in .histories')


class RolloutWorker:
    def __init__(self, env, verbose=False, max_events=10000):
        self.env, self.verbose = env, verbose
        if max_events < 1:
            raise ValueError('max_events must be positive')
        self.max_events = int(max_events)

    def _walk(self, generator, episodes, fixed=None, collect_flows=False):
        states = [self.env.get_initial_state() for _ in range(episodes)]
        paths = [SimpleTrajectory() for _ in states]
        while True:
            rows = [i for i, s in enumerate(states) if not s.is_done]
            if not rows:
                break
            try:
                if any(len(paths[i]) >= self.max_events for i in rows):
                    raise ValueError('ARG event limit exceeded')
                actions = None
                if fixed is not None:
                    if any(len(paths[i]) >= len(fixed[i]) for i in rows):
                        raise ValueError('Replay ends before ancestry completes')
                    actions = [fixed[i][len(paths[i])] for i in rows]
                active = [states[i] for i in rows]
                outputs = generator(active, forced_actions=actions, return_flows=collect_flows)
                for k, (row, action) in enumerate(zip(rows, outputs['actions'])):
                    previous = states[row]
                    state = self.env.apply_action(previous, action)
                    prior = self.env.compute_cwr_event_log_prior(previous, action)
                    if generator.count_backward_parents(state) != 1:
                        raise ValueError('Nonunique chronological predecessor')
                    states[row] = state
                    paths[row].update(action, log_prior=prior, log_reward=state.log_reward,
                                      log_proposal=float(outputs['log_pf'][k].detach()))
                    if state.is_done and not math.isfinite(state.log_reward):
                        if state.log_reward == -math.inf:
                            raise ValueError('Compatible policy reached an exact zero-likelihood ARG')
                        raise FloatingPointError('Numerical failure in terminal reward')
            except (ValueError, FloatingPointError, RuntimeError) as exc:
                raise RolloutFailure(str(exc), paths) from exc
            yield rows, outputs, states, paths
        if fixed is not None and any(len(p) != len(a) for p, a in zip(paths, fixed)):
            raise RolloutFailure('Replay has actions after termination', paths)

    def _run(self, generator, episodes, fixed=None, collect_flows=False, return_states=False):
        if episodes < 1:
            raise ValueError('episodes must be positive')
        pf, flows, factors = ([[] for _ in range(episodes)] for _ in range(3))
        for rows, output, states, paths in self._walk(generator, episodes, fixed, collect_flows):
            for k, row in enumerate(rows):
                pf[row].append(output['log_pf'][k])
                factors[row].append(output['factors'][k])
                if collect_flows:
                    flows[row].append(output['flows'][k])
        rewards = torch.tensor([s.log_reward for s in states], dtype=torch.float64, device=generator.device)
        lengths = torch.tensor([len(p) for p in paths], dtype=torch.long, device=generator.device)
        scores = pad_sequence([torch.stack(p) for p in pf], batch_first=True)
        result = dict(log_paths_pf=scores, log_paths_pb=torch.zeros_like(scores), log_rewards=rewards,
                      lengths=lengths, log_factors=pad_sequence([torch.stack(p) for p in factors], batch_first=True))
        if collect_flows:
            result['state_flows'] = pad_sequence([torch.stack(f+[rewards[i]]) for i, f in enumerate(flows)], batch_first=True)
        if return_states:
            result['states'] = states
        return result, paths

    def rollout(self, generator, episodes=1, random_spec=None, return_states=False, collect_flows=False):
        if random_spec is not None and float(random_spec.get('T', 1.)) != 1.:
            raise ValueError('Phase 2 uses temperature 1; tempered densities are not implemented')
        return self._run(generator, episodes, collect_flows=collect_flows, return_states=return_states)

    def replay(self, generator, trajectories, collect_flows=True, return_states=False):
        actions = [t.actions if hasattr(t, 'actions') else t for t in trajectories]
        return self._run(generator, len(actions), actions, collect_flows, return_states)

    def backward_scores(self, generator, trajectories, pf_weights, flow_weights, chunk_steps=16):
        """Recompute scores in chunks with exact d(loss)/d(score) from a first pass.

        No learned activation survives across chunks. Dropout is absent in this
        architecture, so the score function is identical on both passes.
        """
        if chunk_steps < 1:
            raise ValueError('chunk_steps must be positive')
        actions = [t.actions for t in trajectories]
        steps = [0]*len(actions)
        terms = []
        for rows, output, _, _ in self._walk(generator, len(actions), actions, True):
            for k, row in enumerate(rows):
                step = steps[row]
                terms.append(pf_weights[row, step]*output['log_pf'][k]+
                             flow_weights[row, step]*output['flows'][k])
                steps[row] += 1
            if len(terms) >= chunk_steps:
                torch.stack(terms).sum().backward()
                terms.clear()
        if terms:
            torch.stack(terms).sum().backward()
