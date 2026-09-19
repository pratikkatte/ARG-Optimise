"""Exact oracles for CPU shortcuts and cache-aware static observation packing."""
import copy
import math
from types import MethodType
from unittest.mock import patch

import numpy as np
import pytest
import torch

from env.actions import CoalescenceChoice
from env.states import DescendantSegments, MaterialSegments
from policy.encoder import PooledLineageCache
from policy.observations import pack_states
from training.checkpoints import restore_rng, rng_state, seed_everything
from training.trainer import TrajectoryMixConfig
from validation.tests.test_action_context import assert_exact
from validation.tests.test_fresh_score_reuse import CaptureTrainer, CaptureWorker, make_model
from validation.tests.test_infinite_sites_environment import environment


def reference_restrict(descendants, material):
    return DescendantSegments(tuple((max(l, a), min(r, z), bits)
        for l, r, bits in descendants.segments for a, z in material.segments
        if max(l, a) < min(r, z)))


def reference_parent(self, node, children):
    """Pre-optimization message calculation; retain the independent intersection."""
    node.descendants = (children[0].descendants.merge(children[1].descendants) if len(children) == 2
                        else reference_restrict(children[0].descendants, node.material_segments))
    assert node.descendants.material == node.material_segments
    indices = self.indices(node.material_segments)
    combined = np.zeros((len(indices), 3), dtype=np.float64)
    combined[:, 0] = combined[:, 1] = 1.
    exposures = []
    for child in children:
        dt = node.time-child.time
        relevant = reference_restrict(child.descendants, node.material_segments)
        exposures.append(dt*sum(r-l for l,r,b in relevant.segments if b != self.env.all_samples))
        common, parent_rows, child_rows = np.intersect1d(
            indices, child.snp_indices, assume_unique=True, return_indices=True)
        if len(common):
            a,d,m = child.messages[child_rows].T
            propagated = m+dt*d
            old = combined[parent_rows].copy()
            combined[parent_rows,0] = old[:,0]*a
            combined[parent_rows,1] = old[:,1]*d
            combined[parent_rows,2] = old[:,2]*a+old[:,0]*propagated
    node.exposure_increment = math.fsum(exposures)
    self.cache(node, indices, combined)


def full_encode(self, states, *, pooled_cache=None):
    """Original full-packing path, including static rows already in the cache."""
    batch = pack_states(self.env, states, self.device, cache=self._observation_cache)
    pooled = None
    if pooled_cache is not None:
        nodes = [n for state in states for n in state.active_lineages]
        pooled = pooled_cache.get(self.state_encoder, batch.observations, nodes)
    lineage, summary = self.state_encoder(batch.observations, pooled_embeddings=pooled)
    return batch, lineage, summary


def test_interval_restriction_matches_cartesian_oracle():
    rng = np.random.default_rng(52)
    for _ in range(300):
        bounds = np.unique(rng.integers(0, 100, 30))
        desc = DescendantSegments(tuple((int(l),int(r),1 << int(rng.integers(0,130)))
            for l,r in zip(bounds[::2],bounds[1::2])))
        bounds = np.unique(rng.integers(0, 100, 30))
        material = MaterialSegments(tuple(zip(bounds[::2],bounds[1::2])))
        assert desc.restrict(material) == reference_restrict(desc, material)
    assert DescendantSegments().restrict(MaterialSegments.full(8)) == DescendantSegments()


@pytest.mark.parametrize('count', [0,1,2,10,64,65,70])
def test_pair_templates_preserve_order_and_immutability(count):
    expected = tuple(CoalescenceChoice(i,j) for i in range(count) for j in range(i+1,count))
    actual = CoalescenceChoice.enumerate_from_active_lineages([None]*count)
    assert actual == expected
    if count <= 64:
        assert actual is CoalescenceChoice.enumerate_from_active_lineages([None]*count)
    if actual:
        with pytest.raises(AttributeError):
            actual[0].active_lineage_i = 99


def test_terminal_shortcut_still_checks_exact_coverage():
    env = environment(length=4)
    state = env.get_initial_state()
    with patch.object(env, 'get_active_counts', wraps=env.get_active_counts) as coverage:
        assert not env.is_terminal(state)
        coverage.assert_not_called()
        state.active_lineages = state.active_lineages[:2]
        for node in state.active_lineages:
            node.descendants = DescendantSegments(((0,2,env.all_samples),))
            node.material_segments = MaterialSegments(((0,2),))
        # Summed length equals four, but overlap and a hole must not be terminal.
        assert not env.is_terminal(state)
        assert coverage.call_count == 1
        state.active_lineages[1].material_segments = MaterialSegments(((2,4),))
        state.active_lineages[1].descendants = DescendantSegments(((2,4,env.all_samples),))
        assert env.is_terminal(state)


@pytest.mark.parametrize('snps', [False,True])
def test_packing_skips_cached_rows_but_refreshes_dynamic_features(snps):
    torch.set_num_threads(1)
    g = make_model()
    env = g.env if snps else environment(np.empty((3,0)), [], length=4)
    if not snps:
        g.env = env
    cache = PooledLineageCache()
    states = [env.get_initial_state()]
    nodes = states[0].active_lineages
    plan = cache.prepare(nodes)
    first = pack_states(env, states, static_rows=plan.missing)
    pools = cache.get(g.state_encoder, first.observations, nodes, plan=plan)
    states[0].current_time += .2
    plan = cache.prepare(nodes)
    assert not plan.missing
    compact = pack_states(env, states, static_rows=plan.missing)
    full = pack_states(env, states)
    assert compact.observations.snps.shape[0] == compact.observations.intervals.shape[0] == 0
    assert_exact(compact.observations.lineage_scalars, full.observations.lineage_scalars)
    assert_exact(compact.observations.state_scalars, full.observations.state_scalars)
    assert_exact(cache.get(g.state_encoder, compact.observations, nodes, plan=plan), pools)
    assert pools.requires_grad
    # Changed immutable sources and writable sources cannot reuse stale pools.
    nodes[0].messages = nodes[0].messages.copy()
    assert cache.prepare(nodes).missing == (0,)
    cache.clear()
    assert cache.prepare(nodes).missing == tuple(range(len(nodes)))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('head', ['gamma','exponential'])
@pytest.mark.parametrize('accum', [1,3])
def test_exact_training_trajectories_values_gradients_and_updates(device, head, accum):
    torch.set_num_threads(1)
    seed_everything(27)
    initial_model = make_model(head,device)
    initial = copy.deepcopy(initial_model.state_dict())
    initial_rng = rng_state(initial_model.env)
    results = []
    for reference in (True,False):
        g = make_model(head,device)
        g.load_state_dict(initial)
        if reference:
            g.encode = MethodType(full_encode,g)
            g.env.likelihood_tracker.parent = MethodType(reference_parent,g.env.likelihood_tracker)
        worker = CaptureWorker(g.env)
        trainer = CaptureTrainer(g,worker,TrajectoryMixConfig(replay_fraction=.25,
            exploration_fraction=.25,replay_min_size=2,replay_capacity=32))
        trainer.scored = []
        restore_rng(g.env,initial_rng)
        updates = []
        for _ in range(2):
            metrics = trainer.train_epoch(batch_size=7,grad_accum_steps=accum)
            assert metrics['encoder_grad_norm'] > 0
            updates.append(copy.deepcopy(dict(metrics=metrics,
                gradients={n:p.grad for n,p in g.named_parameters()},parameters=g.state_dict(),
                optimizer=g.opt.state_dict(),scheduler=g.scheduler.state_dict(),
                trainer=trainer.state_dict(),rng=rng_state(g.env))))
        assert updates[-1]['metrics']['replay'] > 0
        results.append((updates,trainer.scored,[p.actions for p in worker.sampled_paths],
                        [p.actions for p in worker.replayed_paths]))
    assert_exact(results[0],results[1])
