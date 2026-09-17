"""Exact scalar-oracle checks for observation construction and training."""
import copy
import gc
import weakref
from dataclasses import fields
from unittest.mock import patch

import numpy as np
import pytest
import torch

from env.env import SimpleARGEnvironment
from env.states import DescendantSegments
from policy import observations as obs
from training.checkpoints import restore_rng, rng_state, seed_everything
from training.trainer import Trainer, TrajectoryMixConfig
from validation.reference.scalar_observations import pack_states as scalar_pack
from validation.tests.test_infinite_sites_environment import environment, split
from validation.tests.test_infinite_sites_neural import environment as neural_environment
from validation.tests.test_fresh_score_reuse import CaptureWorker, make_model


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


def assert_batch(a, b):
    for field in fields(a.observations):
        assert_exact(getattr(a.observations, field.name), getattr(b.observations, field.name))
    assert a.observations.counts == b.observations.counts
    assert_exact(a.actions, b.actions)
    assert_exact(a.physical_rates, b.physical_rates)
    assert_exact(a.allowed_hazards, b.allowed_hazards)


@pytest.mark.parametrize('n', [3, 63, 64, 65, 70, 129])
@pytest.mark.parametrize('snps', [False, True])
def test_initial_states_and_sample_widths(n, snps):
    env = neural_environment(n=n, snps=snps)
    state = env.get_initial_state()
    cache = obs.RawObservationCache()
    for _ in range(2):
        assert_batch(obs.pack_states(env, [state], cache=cache), scalar_pack(env, [state]))
    assert cache.hits == n


def test_real_trajectory_mixed_batches_and_empty_lineages():
    env = environment(length=4, recombination_rate=.02)
    states = [env.get_initial_state()]
    states.append(split(env, states[0], breakpoint=1, dt=.2))
    assert any(not len(node.snp_indices) for node in states[-1].active_lineages)
    for _ in range(1000):
        if states[-1].is_done:
            break
        states.append(env.apply_action(states[-1], env.sample_compatible_step(states[-1]).action))
    assert states[-1].is_done
    cache = obs.RawObservationCache()
    for start in range(0, len(states), 7):
        batch = states[start:start+7]
        expected = scalar_pack(env, batch)
        assert_batch(obs.pack_states(env, batch), expected)
        assert_batch(obs.pack_states(env, batch, cache=cache), expected)
        assert_batch(obs.pack_states(env, batch, cache=cache), expected)


@pytest.mark.parametrize('n', [3, 64, 65, 129])
def test_fragmented_geometry_messages_and_gaps(n):
    positions = np.array([0., np.nextafter(1., 0.), 1., 2., 3., 4., 5., 7.9])
    g = np.zeros((n, len(positions)), dtype=np.uint8)
    g[-1] = 1
    env = environment(g, positions, length=8)
    state = env.get_initial_state()
    node = state.active_lineages[0]
    node.descendants = DescendantSegments(((0, 1, 1 << (n-1)), (1, 2, env.all_samples),
                                         (3, 5, 3), (7, 8, 1)))
    messages = node.messages.copy()
    messages[:, 2] = [0., np.nextafter(0., 1.), 1e-30, .1, 1., 1e30, 1e100, 1e300]
    messages.setflags(write=False)
    node.messages = messages
    assert_batch(obs.pack_states(env, [state]), scalar_pack(env, [state]))
    rows, _ = obs._static_features(env, node, obs._dataset_features(env.snp_data))
    assert rows[3, 7+n:].sum() == 0  # right boundary / material gap
    assert rows[6, 7+n:].sum() == 0
    assert rows[2, 6] == 1


def test_very_large_integer_coordinates():
    length = 2**54
    position = float(2**53)
    env = environment([[1], [0], [0]], [position], length=length)
    state = env.get_initial_state()
    state.active_lineages[0].descendants = DescendantSegments(((2**53+1, length, 1),))
    assert_batch(obs.pack_states(env, [state]), scalar_pack(env, [state]))


def test_dataset_constants_are_shared_readonly_and_weak():
    env = neural_environment()
    data = env.snp_data
    first = obs._dataset_features(data)
    other = SimpleARGEnvironment(snp_data=data, population_size=20, mutation_rate=.1)
    assert obs._dataset_features(other.snp_data) is first
    for value in (first.geometry, first.observed, first.bit_shifts):
        assert not value.flags.writeable
    before = len(obs._DATASET_FEATURES)
    ref = weakref.ref(data)
    del data, env, other
    gc.collect()
    assert ref() is None
    assert len(obs._DATASET_FEATURES) < before


def test_cache_lifetime_mutation_isolation_and_batch_ownership():
    env = neural_environment(length=4)
    state = env.get_initial_state()
    cache = obs.RawObservationCache(max_bytes=180)
    first = obs.pack_states(env, [state], cache=cache)
    assert cache.bytes <= cache.max_bytes
    assert cache.misses == 3 and len(cache.entries) < 3
    saved = first.observations.snps.clone()
    node = state.active_lineages[0]
    node.messages = node.messages.copy()  # writable sources bypass raw caching
    node.messages[:, 2] = .5
    assert_batch(obs.pack_states(env, [state], cache=cache), scalar_pack(env, [state]))
    assert_exact(first.observations.snps, saved)
    node.messages.setflags(write=False)
    cache = obs.RawObservationCache()
    obs.pack_states(env, [state], cache=cache)
    obs.pack_states(env, [state], cache=cache)
    assert cache.hits == 3
    node.descendants = DescendantSegments(((0, 4, env.all_samples),))
    assert_batch(obs.pack_states(env, [state], cache=cache), scalar_pack(env, [state]))
    node.snp_indices = np.array([], dtype=np.int64)
    node.messages = np.empty((0, 3), dtype=np.float64)
    node.messages.setflags(write=False)
    assert_batch(obs.pack_states(env, [state], cache=cache), scalar_pack(env, [state]))
    other = neural_environment(length=8)
    # Even shared sources with equal shapes must not reuse another dataset's rows.
    other_state = other.get_initial_state()
    other_state.active_lineages[1].messages = state.active_lineages[1].messages
    assert_batch(obs.pack_states(other, [other_state], cache=cache), scalar_pack(other, [other_state]))
    del node, state, other_state
    gc.collect()
    assert cache.bytes == 0 and not cache.entries
    assert_exact(first.observations.snps, saved)


def test_errors_and_scalar_log_domain_are_preserved():
    env = neural_environment()
    for pack in (obs.pack_states, scalar_pack):
        with pytest.raises(ValueError, match='At least one'):
            pack(env, [])
        state = env.get_initial_state()
        state.active_lineages[0].messages = None
        with pytest.raises(ValueError, match='missing'):
            pack(env, [state])
        state = env.get_initial_state()
        state.active_lineages[0].messages = state.active_lineages[0].messages.copy()
        state.active_lineages[0].messages[:, 2] = -1
        with pytest.raises(ValueError, match='math domain'):
            pack(env, [state])
        state.active_lineages[0].messages[:, 2] = np.inf
        with pytest.raises(FloatingPointError, match='Nonfinite neural observation'):
            pack(env, [state])


class ScoredTrainer(Trainer):
    def _score_subset(self, *args, **kwargs):
        outputs, paths = super()._score_subset(*args, **kwargs)
        self.scores.append({k: v.detach().clone() for k, v in outputs.items() if torch.is_tensor(v)})
        return outputs, paths


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('head', ['gamma', 'exponential'])
@pytest.mark.parametrize('replay', [False, True])
def test_exact_training_updates(device, head, replay):
    torch.set_num_threads(1)
    seed_everything(27)
    initial_model = make_model(head, device)
    initial = copy.deepcopy(initial_model.state_dict())
    initial_rng = rng_state(initial_model.env)
    results = []
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        for pack in (scalar_pack, obs.pack_states):
            g = make_model(head, device)
            g.load_state_dict(initial)
            worker = CaptureWorker(g.env)
            trainer = ScoredTrainer(g, worker, TrajectoryMixConfig(
                replay_fraction=.25 if replay else 0., exploration_fraction=.25 if replay else 0.,
                replay_min_size=2, replay_capacity=32), chunk_steps=3)
            trainer.scores = []
            restore_rng(g.env, initial_rng)
            with patch('generator.pack_states', pack):
                updates = []
                for _ in range(2):
                    metrics = trainer.train_epoch(batch_size=7, grad_accum_steps=3)
                    updates.append(copy.deepcopy(dict(metrics=metrics,
                        gradients={k: p.grad for k, p in g.named_parameters() if p.grad is not None},
                        parameters=g.state_dict(), optimizer=g.opt.state_dict(),
                        scheduler=g.scheduler.state_dict(), trainer=trainer.state_dict(),
                        rng=rng_state(g.env))))
            if replay:
                assert updates[-1]['metrics']['replay'] > 0
            results.append((updates, trainer.scores, worker.weights,
                            [p.actions for p in worker.sampled_paths]))
        assert_exact(results[0], results[1])
    finally:
        torch.use_deterministic_algorithms(deterministic)
