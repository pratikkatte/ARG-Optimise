"""Fresh-score reuse must preserve the full-rescoring training update."""
import copy
import gc
import weakref

import numpy as np
import pytest
import torch

from generator import GFlowNetGenerator
from gfn.rollout import RolloutWorker
from training.checkpoints import restore_rng, rng_state, seed_everything
from training.schedules import LearningRateConfig, PolicyTemperatureConfig, WarmupCosineScheduler
from training.trainer import Trainer, TrajectoryMixConfig
from validation.tests.test_infinite_sites_neural import environment


class CaptureWorker(RolloutWorker):
    def __init__(self, env):
        super().__init__(env)
        self.sampled_paths, self.replayed_paths, self.backward_paths = [], [], []
        self.weights, self.flow_refs, self.sampling_options = [], [], []

    def rollout(self, *args, **kwargs):
        outputs, paths = super().rollout(*args, **kwargs)
        self.sampling_options.append(kwargs)
        self.sampled_paths.extend(paths)
        if kwargs.get('collect_flows'):
            assert outputs['log_paths_pf'].requires_grad
            assert outputs['state_flows'].requires_grad
        else:
            assert all(not v.requires_grad for v in outputs.values() if torch.is_tensor(v))
        if 'state_flows' in outputs:
            self.flow_refs.append(weakref.ref(outputs['state_flows']))
        return outputs, paths

    def replay(self, generator, paths, **kwargs):
        self.replayed_paths.extend(paths)
        return super().replay(generator, paths, **kwargs)

    def backward_scores(self, generator, paths, pf_weights, flow_weights, chunk_steps=16):
        raise AssertionError('Standard training must not recompute scores for backward')


class CaptureTrainer(Trainer):
    force_rescore = False

    def _score_subset(self, subset, sampled=None):
        outputs, paths = super()._score_subset(subset, None if self.force_rescore else sampled)
        self.scored.append({k: v.detach().clone() for k, v in outputs.items() if torch.is_tensor(v)})
        return outputs, paths


class ReferenceTrainer(CaptureTrainer):
    force_rescore = True


def make_model(head='gamma', device='cpu'):
    g = GFlowNetGenerator(environment(recombination=.01, length=4), device=device,
        initialize_z_from_policy=False, model_kwargs=dict(embedding_size=16, hidden_size=32,
            transformer_depth=1, transformer_heads=2, continuous_time_head=head))
    with torch.no_grad():
        # Exercise gradients into the shared encoder, not just zero-initialized heads.
        for head_module in (g.flow_head, g.arg_model.action_head, g.arg_model.event_head,
                            g.arg_model.breakpoint_head.parameters_head):
            head_module[-1].weight.normal_(0, .03)
        g.arg_model.time_head.output_layer.weight.normal_(0, .03)
    g.scheduler = WarmupCosineScheduler(g.opt, LearningRateConfig('cosine', 10, 0, .1, .1))
    return g


def assert_nested_close(first, second):
    if torch.is_tensor(first):
        torch.testing.assert_close(first, second, atol=2e-5, rtol=2e-5)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_close(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_nested_close(a, b)
    elif isinstance(first, float):
        assert first == pytest.approx(second, abs=2e-5, rel=2e-5)
    else:
        assert first == second


@pytest.mark.parametrize('head', ['gamma', 'exponential', 'gamma_mixture'])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA hardware unavailable'))])
@pytest.mark.parametrize('batch,accum,exploration,replay', [
    (5, 1, 0., 0.), (7, 3, 0., 0.), (7, 3, 0., .25),
    (7, 3, .25, 0.), (7, 3, .25, .25), (8, 2, 0., .25),
])
def test_reuse_matches_full_rescoring(head, device, batch, accum, exploration, replay):
    torch.set_num_threads(1)
    seed_everything(27)
    g = make_model(head, device)
    initial = copy.deepcopy(g.state_dict())
    mix = TrajectoryMixConfig(exploration_fraction=exploration, replay_fraction=replay,
                              replay_min_size=2, replay_capacity=16, replay_grid_size=3)
    warm = Trainer(g, RolloutWorker(g.env), mix)
    if replay:
        with torch.no_grad():
            output, paths = warm.worker.rollout(g, 2, return_states=True)
        for path, state in zip(paths, output['states']):
            warm.buffer.add(g.env, path, state, 'policy', 1)
    buffer_state = copy.deepcopy(warm.buffer.state_dict()) if replay else None
    sampling_rng = rng_state(g.env)
    results = []
    for trainer_type in (ReferenceTrainer, CaptureTrainer):
        model = make_model(head, device)
        model.load_state_dict(initial)
        worker = CaptureWorker(model.env)
        trainer = trainer_type(model, worker, mix, chunk_steps=3)
        trainer.scored = []
        if replay:
            trainer.buffer.load_state_dict(copy.deepcopy(buffer_state))
        restore_rng(model.env, sampling_rng)
        metrics = trainer.train_epoch(batch_size=batch, grad_accum_steps=accum)
        gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
        results.append((model, worker, trainer, metrics, gradients, rng_state(model.env)))

    old, new = results
    assert_nested_close(old[3], new[3])
    assert_nested_close(old[4], new[4])
    assert_nested_close(old[0].state_dict(), new[0].state_dict())
    assert_nested_close(old[0].opt.state_dict(), new[0].opt.state_dict())
    assert old[0].scheduler.state_dict() == new[0].scheduler.state_dict()
    assert_nested_close(old[1].weights, new[1].weights)
    for expected, actual in zip(old[2].scored, new[2].scored):
        assert_nested_close(expected, actual)
        assert torch.equal(expected['lengths'], actual['lengths'])
        assert torch.equal(expected['log_rewards'], actual['log_rewards'])
        rows = torch.arange(len(actual['lengths']), device=device)
        assert torch.equal(actual['state_flows'][rows, actual['lengths']], actual['log_rewards'])
    assert [p.actions for p in old[1].sampled_paths] == [p.actions for p in new[1].sampled_paths]
    assert [p.log_proposals for p in old[1].sampled_paths] == [p.log_proposals for p in new[1].sampled_paths]
    assert [p.actions for p in old[1].backward_paths] == [p.actions for p in new[1].backward_paths]
    assert torch.equal(old[5]['torch'], new[5]['torch'])
    assert_nested_close(old[5], new[5])
    if replay:
        assert old[2].buffer.state_dict() == new[2].buffer.state_dict()
    fresh = new[3]['fresh']
    assert len(old[1].replayed_paths) == batch
    assert len(new[1].replayed_paths) == batch-fresh
    assert not new[1].backward_paths
    assert not {id(p) for p in new[1].sampled_paths} & {id(p) for p in new[1].replayed_paths}
    gc.collect()
    assert all(ref() is None for ref in new[1].flow_refs)


def test_tempered_sampling_falls_back_until_exactly_one():
    torch.set_num_threads(1)
    seed_everything(7)
    g = make_model()
    worker = CaptureWorker(g.env)
    trainer = Trainer(g, worker, TrajectoryMixConfig(replay_fraction=0.),
                      temperature_config=PolicyTemperatureConfig('linear', 2., 1))
    first = trainer.train_epoch(batch_size=3, grad_accum_steps=2)
    assert first['policy_temperature'] == 2.
    assert len(worker.replayed_paths) == 3
    assert all(not c['collect_flows'] and not c['return_states'] for c in worker.sampling_options)
    worker.replayed_paths.clear()
    worker.sampling_options.clear()
    second = trainer.train_epoch(batch_size=3, grad_accum_steps=2)
    assert second['policy_temperature'] == 1.
    assert not worker.replayed_paths
    assert all(c['collect_flows'] and c['return_states'] for c in worker.sampling_options)


@pytest.mark.parametrize('head', ['gamma', 'exponential', 'gamma_mixture'])
def test_flow_collection_preserves_sampling_and_rng(head):
    torch.set_num_threads(1)
    seed_everything(27)
    g = make_model(head)
    worker = RolloutWorker(g.env)
    before = rng_state(g.env)
    with torch.no_grad():
        plain, paths = worker.rollout(g, 5)
        after = rng_state(g.env)
        restore_rng(g.env, before)
        collected, collected_paths = worker.rollout(g, 5, collect_flows=True, return_states=True)
        assert_nested_close(after, rng_state(g.env))
        replayed, _ = worker.replay(g, collected_paths)
    assert [p.actions for p in paths] == [p.actions for p in collected_paths]
    for key in plain:
        torch.testing.assert_close(plain[key], collected[key], atol=0, rtol=0)
    for key in replayed:
        torch.testing.assert_close(replayed[key], collected[key], atol=0, rtol=0)
    assert len(set(collected['lengths'].tolist())) > 1


@pytest.mark.parametrize('long_fresh', [False, True])
def test_mixed_subsets_pad_either_side_without_moving_terminal(long_fresh):
    torch.set_num_threads(1)
    seed_everything(27)
    g = make_model()
    worker = RolloutWorker(g.env)
    trainer = Trainer(g, worker, TrajectoryMixConfig(replay_fraction=0.))
    with torch.no_grad():
        _, paths = worker.rollout(g, 5)
        ordered = sorted(paths, key=len)
        assert len(ordered[0]) < len(ordered[-1])
        subset = [ordered[-1], ordered[0]] if long_fresh else [ordered[0], ordered[-1]]
        saved, _ = worker.replay(g, subset[:1], return_states=True)
        expected, _ = worker.replay(g, subset, return_states=True)
        actual, rescored = trainer._score_subset(subset, saved)
    assert [p.actions for p in rescored] == [p.actions for p in subset]
    for key in expected:
        if torch.is_tensor(expected[key]):
            torch.testing.assert_close(expected[key], actual[key], atol=2e-5, rtol=2e-5)
    for row, length in enumerate(actual['lengths'].tolist()):
        assert actual['state_flows'][row, length] == actual['log_rewards'][row]
        assert torch.count_nonzero(actual['state_flows'][row, length+1:]) == 0
        for key in ('log_paths_pf', 'log_paths_pb', 'log_factors'):
            assert torch.count_nonzero(actual[key][row, length:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable')
def test_cuda_mixed_score_assembly():
    torch.set_num_threads(1)
    seed_everything(27)
    g = make_model(device='cuda')
    worker = RolloutWorker(g.env)
    trainer = Trainer(g, worker, TrajectoryMixConfig(replay_fraction=0.))
    with torch.no_grad():
        saved, fresh = worker.rollout(g, 2, collect_flows=True, return_states=True)
        _, extra = worker.rollout(g, 1)
        expected, _ = worker.replay(g, fresh+extra, return_states=True)
        actual, _ = trainer._score_subset(fresh+extra, saved)
    for key in expected:
        if torch.is_tensor(expected[key]):
            assert actual[key].is_cuda
            torch.testing.assert_close(expected[key], actual[key], atol=2e-5, rtol=2e-5)
