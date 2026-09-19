"""Cached action support must preserve physical priors and exact training updates."""
import copy
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from env.actions import CoalescenceChoice, RecombinationChoice
from env.env import SimpleARGEnvironment
from env.states import DescendantSegments, MaterialSegments
from training.checkpoints import restore_rng, rng_state, seed_everything
from training.trainer import TrajectoryMixConfig
from validation.tests.test_fresh_score_reuse import CaptureTrainer, CaptureWorker, make_model
from validation.tests.test_infinite_sites_environment import environment, coal, split, assert_same


def assert_exact(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_exact(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_exact(x, y)
    else:
        assert a == b


def physical_actions(state):
    if state.is_done:
        return [], []
    return (list(CoalescenceChoice.enumerate_from_active_lineages(state.active_lineages)),
            list(RecombinationChoice.enumerate_from_active_lineages(state.active_lineages)))


def scalar_support(env, state):
    coal_actions, recomb = physical_actions(state)
    return ([a for a in coal_actions if not env.incompatible_sites(state, a)],
            recomb if env.recombination_rate > 0 else [])


class UncachedEnvironment(SimpleARGEnvironment):
    """Force fresh physical data and independently walk intervals for each pair."""
    def _get_action_context(self, state):
        state._action_context = None
        return super()._get_action_context(state)

    def enumerate_policy_actions(self, state):
        self._check_state(state)
        return scalar_support(self, state)


@pytest.mark.parametrize('event', ['coal', 'recomb'])
def test_one_physical_construction_for_policy_prior_validation_and_step(event):
    env = environment(length=4)
    state = env.get_initial_state()
    with patch.object(CoalescenceChoice, 'enumerate_from_active_lineages',
                      wraps=CoalescenceChoice.enumerate_from_active_lineages) as coal_build, \
         patch.object(RecombinationChoice, 'enumerate_from_active_lineages',
                      wraps=RecombinationChoice.enumerate_from_active_lineages) as recomb_build:
        policy = env.enumerate_policy_actions(state)
        options = env.enumerate_prior_options(state)
        assert options.rates['lambda_coal'] == 3  # Includes incompatible pairs.
        action = (replace(policy[0][0], delta_t=.2) if event == 'coal' else
                  replace(policy[1][0], breakpoint=2, delta_t=.2))
        prior = env.compute_cwr_event_log_prior(state, policy, action, rates={'lambda_coal': 1})
        state, scored = env.step_owned_state(state, action)
        assert scored == prior
        assert coal_build.call_count == recomb_build.call_count == 1
        env.enumerate_policy_actions(state)
        env.enumerate_prior_options(state)
        assert coal_build.call_count == recomb_build.call_count == 2


def test_public_lists_and_rate_dicts_cannot_poison_context():
    env = environment(length=4)
    state = env.get_initial_state()
    expected_physical = physical_actions(state)
    expected_policy = scalar_support(env, state)
    rates = env.compute_event_rates(expected_physical)
    action = replace(expected_policy[0][0], delta_t=.2)
    prior = env.compute_cwr_event_log_prior(state, action)
    for choices in (env.enumerate_actions(state), env.enumerate_policy_actions(state)):
        choices[0].clear()
        choices[1].reverse()
    env.compute_coalescence_actions(state).clear()
    env.compute_recombination_actions(state).clear()
    options = env.enumerate_prior_options(state)
    options.rates.clear()
    state.rates, state.prior_options = {'lambda_coal': -100}, options
    assert env.enumerate_actions(state) == expected_physical
    assert env.enumerate_policy_actions(state) == expected_policy
    assert env.enumerate_prior_options(state).rates == rates
    assert env.compute_cwr_event_log_prior(state, ([], []), action, rates={}) == prior


def test_reorder_and_replaced_material_or_descendants_invalidate_support():
    env = environment(length=4)
    state = env.get_initial_state()
    env.enumerate_policy_actions(state)
    state.active_lineages.reverse()
    assert env.enumerate_policy_actions(state) == scalar_support(env, state)
    node = state.active_lineages[0]
    node.descendants = DescendantSegments(((0, 4, env.all_samples),))
    assert env.enumerate_policy_actions(state) == scalar_support(env, state)
    old_choice = env.enumerate_actions(state)[1][0]
    node.material_segments = MaterialSegments(((0, 2),))
    assert env.enumerate_actions(state) == physical_actions(state)
    assert env.enumerate_prior_options(state).rates == env.compute_event_rates(physical_actions(state))
    with pytest.raises(ValueError, match='invalid recombination span'):
        env.compute_cwr_event_log_prior(state, replace(old_choice, breakpoint=1, delta_t=.2))
    # Rates are also protected when the environment's physical rate is changed.
    env.rho *= 2
    assert env.enumerate_prior_options(state).rates == env.compute_event_rates(physical_actions(state))
    state.is_done = True
    assert env.enumerate_actions(state) == ([], [])
    assert env.enumerate_policy_actions(state) == ([], [])


def test_branch_isolation_restore_and_terminal_cache_lifetime():
    env = environment(length=4)
    original = env.get_initial_state()
    expected = env.enumerate_policy_actions(original)
    snapshot = original._compatibility
    assert not snapshot.descendants.flags.writeable
    assert not snapshot.support.flags.writeable
    merged = coal(env, original)
    divided = split(env, original, breakpoint=2)
    for branch in (divided, merged):
        assert env.enumerate_policy_actions(branch) == scalar_support(env, branch)
        assert branch._compatibility is not snapshot
        assert env.enumerate_policy_actions(original) == expected
        assert original._compatibility is snapshot
    assert merged._action_context is not original._action_context
    broken = divided.clone()
    for node in broken.all_nodes.values():
        node.descendants = node.messages = node.snp_indices = None
    restored = env.restore_state(broken)
    assert_same(restored, divided)
    assert restored._compatibility is None
    assert env.enumerate_policy_actions(restored) == scalar_support(env, restored)
    terminal = coal(env, merged)
    assert terminal.is_done
    assert terminal._action_context is terminal._compatibility is None
    with pytest.raises(ValueError, match='different'):
        environment([[0], [1], [1]], length=4).enumerate_policy_actions(original)


@pytest.mark.parametrize('event,decoded_rows', [('coal', 1), ('recomb', 2)])
def test_surviving_descendants_are_reused_across_events_and_reordering(event, decoded_rows):
    env = environment(length=4)
    state = env.get_initial_state()
    env.enumerate_policy_actions(state)
    state = coal(env, state) if event == 'coal' else split(env, state, breakpoint=2)
    expected = scalar_support(env, state)
    with patch('env.action_context.np.searchsorted', wraps=np.searchsorted) as decode:
        assert env.enumerate_policy_actions(state) == expected
        assert decode.call_count == decoded_rows
    state.active_lineages.reverse()
    expected = scalar_support(env, state)
    with patch('env.action_context.np.searchsorted', wraps=np.searchsorted) as decode:
        assert env.enumerate_policy_actions(state) == expected
        assert decode.call_count == 0


@pytest.mark.parametrize('head', ['gamma', 'exponential'])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_exact_training_updates_with_replay_and_exploration(head, device):
    torch.set_num_threads(1)
    seed_everything(27)
    initial_model = make_model(head, device)
    initial = copy.deepcopy(initial_model.state_dict())
    initial_rng = rng_state(initial_model.env)
    results = []
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        for cached in (False, True):
            g = make_model(head, device)
            if not cached:
                env = g.env
                g.env = UncachedEnvironment(snp_data=env.snp_data,
                    population_size=env.population_size, mutation_rate=env.mutation_rate,
                    recombination_rate=env.recombination_rate, reward_C=env.reward_fn.C)
            g.load_state_dict(initial)
            worker = CaptureWorker(g.env)
            trainer = CaptureTrainer(g, worker, TrajectoryMixConfig(
                replay_fraction=.25, exploration_fraction=.25,
                replay_min_size=2, replay_capacity=32), chunk_steps=3)
            trainer.scored = []
            restore_rng(g.env, initial_rng)
            updates = []
            for _ in range(2):
                metrics = trainer.train_epoch(batch_size=7, grad_accum_steps=3)
                updates.append(copy.deepcopy(dict(metrics=metrics,
                    gradients={k: p.grad for k, p in g.named_parameters() if p.grad is not None},
                    parameters=g.state_dict(), optimizer=g.opt.state_dict(),
                    scheduler=g.scheduler.state_dict(), trainer=trainer.state_dict(),
                    rng=rng_state(g.env))))
            assert updates[-1]['metrics']['replay'] > 0
            results.append((updates, trainer.scored, [p.actions for p in worker.sampled_paths],
                            [p.actions for p in worker.replayed_paths]))
        assert_exact(results[0], results[1])
    finally:
        torch.use_deterministic_algorithms(deterministic)
