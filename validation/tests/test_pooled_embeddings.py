"""Pooling reuse preserves nonzero encoder gradients and backward lifetimes."""
import copy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from env.states import DescendantSegments
from gfn.rollout import RolloutFailure, RolloutWorker
from policy.encoder import PooledLineageCache
from training.checkpoints import restore_rng, rng_state, seed_everything
from training.schedules import PolicyTemperatureConfig
from training.trainer import Trainer, TrajectoryMixConfig
from validation.tests.test_fresh_score_reuse import make_model, assert_nested_close
from validation.tests.test_infinite_sites_neural import environment, model


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA hardware unavailable'))]


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    seed_everything(27)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('snps', [False, True])
def test_pools_context_and_nonzero_gradients(device, snps):
    env = environment(n=4, snps=snps, recombination=.01, length=4)
    g = model(env, device=device)
    initial = env.get_initial_state()
    other_episode = env.get_initial_state()  # Repeated node IDs must not alias.
    choice = env.enumerate_policy_actions(initial)[1][0]
    split = env.apply_action(initial, replace(choice, breakpoint=1, delta_t=.2))
    assert any(len(n.snp_indices) == 0 for n in split.active_lineages)
    permuted = split.clone()
    permuted.active_lineages.reverse()
    batches = [[initial, other_episode], [split, other_episode], [permuted], [permuted, permuted]]
    records, gradients, token_counts = [], [], []
    for enabled in (False, True):
        g.opt.zero_grad(set_to_none=True)
        cache = PooledLineageCache() if enabled else None
        counts = [0, 0]
        def count_snps(module, args):
            counts[0] += len(args[0])
        def count_material(module, args):
            counts[1] += len(args[0])
        hooks = [g.state_encoder.snp_encoder.register_forward_pre_hook(count_snps),
                 g.state_encoder.material_encoder.register_forward_pre_hook(count_material)]
        outputs, terms = [], []
        for states in batches:
            _, lineages, summary = g.encode(states, pooled_cache=cache)
            outputs.append((lineages.detach().clone(), summary.detach().clone()))
            # Unequal downstream derivatives expose incorrectly detached or
            # overwritten contributions from repeated uses of the same pool.
            terms.append((len(terms)+1)*(lineages.sin().mean()+summary.square().mean()))
        sum(terms).backward()
        gradients.append({k: p.grad.clone() for k, p in g.state_encoder.named_parameters() if p.grad is not None})
        records.append(outputs)
        token_counts.append(counts)
        for hook in hooks:
            hook.remove()
        if cache is not None:
            cache.clear()
        assert sum(float(v.abs().sum()) for k, v in gradients[-1].items() if k.startswith('material_encoder')) > 0
        if snps:
            assert sum(float(v.abs().sum()) for k, v in gradients[-1].items() if k.startswith('snp_encoder')) > 0
    assert_nested_close(records[0], records[1])
    assert_nested_close(gradients[0], gradients[1])
    assert token_counts[1][1] < token_counts[0][1]
    if snps:
        assert token_counts[1][0] < token_counts[0][0]


def test_replaced_sources_and_material_invalidate_pools():
    g = model(environment(n=4))
    state = g.env.get_initial_state()
    cache = PooledLineageCache()
    g.encode([state], pooled_cache=cache)
    node = state.active_lineages[0]
    messages = node.messages.copy()
    messages[:, 2] = .7
    messages.setflags(write=False)
    node.messages = messages
    for mutate in ('messages', 'material', 'writable'):
        if mutate == 'material':
            node.descendants = DescendantSegments(((0, 1, 1), (1, 2, 3)))
        if mutate == 'writable':
            node.messages = node.messages.copy()
            node.messages[:, 2] = .9
        actual = g.encode([state], pooled_cache=cache)[1:]
        expected = g.encode([state])[1:]
        assert_nested_close(actual, expected)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('head', ['gamma', 'exponential'])
@pytest.mark.parametrize('accum,extras,tempered', [(1, False, False), (3, True, False), (2, False, True)])
def test_complete_updates_with_nonzero_gradients(device, head, accum, extras, tempered):
    initial = make_model(head, device)
    state = copy.deepcopy(initial.state_dict())
    mix = TrajectoryMixConfig(exploration_fraction=.2 if extras else 0.,
        replay_fraction=.2 if extras else 0., replay_min_size=2, replay_capacity=16, replay_grid_size=3)
    warm = Trainer(initial, RolloutWorker(initial.env), mix)
    if extras:
        with torch.no_grad():
            output, paths = warm.worker.rollout(initial, 3, return_states=True)
        for path, terminal in zip(paths, output['states']):
            warm.buffer.add(initial.env, path, terminal, 'policy', 1)
        assert len(warm.buffer) >= mix.replay_min_size
    buffer = copy.deepcopy(warm.buffer.state_dict()) if extras else None
    rng = rng_state(initial.env)
    results = []
    for enabled in (False, True):
        g = make_model(head, device)
        g.load_state_dict(state)
        worker = RolloutWorker(g.env, cache_pooled_embeddings=enabled)
        trainer = Trainer(g, worker, mix, temperature_config=(
            PolicyTemperatureConfig('linear', 2., 1) if tempered else None))
        if extras:
            trainer.buffer.load_state_dict(copy.deepcopy(buffer))
        restore_rng(g.env, rng)
        updates = []
        for _ in range(2):
            metrics = trainer.train_epoch(batch_size=7, grad_accum_steps=accum)
            assert metrics['encoder_grad_norm'] > 0
            if extras:
                assert metrics['replay'] > 0 and metrics['compatible_proposal'] > 0
            grads = {k: p.grad.clone() for k, p in g.named_parameters() if p.grad is not None}
            for prefix in ('state_encoder.snp_encoder', 'state_encoder.material_encoder'):
                assert sum(float(v.abs().sum()) for k, v in grads.items() if k.startswith(prefix)) > 0
            updates.append((metrics, grads, copy.deepcopy(g.state_dict()), copy.deepcopy(g.opt.state_dict())))
        results.append(updates)
    assert_nested_close(results[0], results[1])


def test_cache_cleared_on_return_and_failure_and_direct_walk_uncached():
    g = make_model()
    caches = []
    def create():
        cache = PooledLineageCache()
        caches.append(cache)
        return cache
    with patch('gfn.rollout.PooledLineageCache', side_effect=create):
        worker = RolloutWorker(g.env)
        outputs, paths = worker.rollout(g, 2, collect_flows=True)
        assert caches[-1].pooled is None and not caches[-1].sources
        g.get_loss_from_rollout_outputs(outputs).backward()
        g.opt.step()
        g.opt.zero_grad(set_to_none=True)
        outputs, _ = worker.replay(g, paths)
        g.get_loss_from_rollout_outputs(outputs).backward()
        with pytest.raises(RolloutFailure):
            RolloutWorker(g.env, max_events=1).rollout(g, 2)
        assert len(caches) == 3
        assert all(c.pooled is None and not c.sources and not c.positions for c in caches)
        for _, output, _, _ in worker._walk(g, 1, collect_flows=True):
            (output['log_pf'].sum()+output['flows'].sum()).backward()
        assert len(caches) == 3
