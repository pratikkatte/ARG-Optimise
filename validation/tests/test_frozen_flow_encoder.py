"""Frozen initial representations, independent gradient paths, and resumable training."""
import copy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from env.actions import CoalescenceChoice
from gfn.flow_encoder import FrozenFlowEncoder
from gfn.rollout import RolloutFailure, RolloutWorker
from infer import run_inference
from policy.encoder import PooledLineageCache
from training.checkpoints import (generator_from_checkpoint, load_checkpoint, restore_rng,
                                  rng_state, seed_everything)
from training.configuration import config_notes, parse_train_args, resolve_config
from training.evaluation import evaluate_generator
from training.schedules import LearningRateConfig, WarmupCosineScheduler
from training.trainer import Trainer, TrajectoryMixConfig
from validation.tests.test_fresh_score_reuse import CaptureTrainer, ReferenceTrainer, assert_nested_close
from validation.tests.test_infinite_sites_neural import environment, model


ROOT = Path(__file__).resolve().parents[2]
DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA hardware unavailable'))]


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    seed_everything(27)


def frozen_model(env=None, **kwargs):
    g = model(env or environment(recombination=.01, length=4),
              flow_encoder_mode='frozen_initial', tb_loss_weight=.25, **kwargs)
    # Nonzero output weights ensure gradient tests exercise the full branches.
    with torch.no_grad():
        for head in (g.flow_head, g.arg_model.action_head, g.arg_model.event_head,
                     g.arg_model.breakpoint_head.parameters_head):
            head[-1].weight.normal_(0, .03)
        g.arg_model.time_head.output_layer.weight.normal_(0, .03)
    return g


def snapshot(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def assert_unchanged(module, expected):
    assert module.state_dict().keys() == expected.keys()
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, expected[name], atol=0, rtol=0)


@pytest.mark.parametrize('encoder_lr', [None, .0002])
def test_initial_copy_has_independent_storage_without_consuming_rng(encoder_lr):
    seed_everything(7)
    shared = model(encoder_lr=encoder_lr)
    after_shared = torch.get_rng_state()
    seed_everything(7)
    frozen = model(flow_encoder_mode='frozen_initial', encoder_lr=encoder_lr)
    assert torch.equal(torch.get_rng_state(), after_shared)
    for name, value in shared.state_dict().items():
        torch.testing.assert_close(value, frozen.state_dict()[name], atol=0, rtol=0)
    assert_unchanged(frozen.flow_encoder.encoder, snapshot(frozen.state_encoder))
    for name, value in frozen.state_encoder.state_dict().items():
        assert value.data_ptr() != frozen.flow_encoder.encoder.state_dict()[name].data_ptr()
    trainable = {id(p) for p in frozen.parameters() if p.requires_grad}
    optimized = [id(p) for group in frozen.opt.param_groups for p in group['params']]
    assert len(optimized) == len(set(optimized))
    assert set(optimized) == trainable
    assert all(not p.requires_grad for p in frozen.flow_encoder.parameters())
    assert frozen.flow_head[-1].weight.count_nonzero() == 0
    for mode in (True, False, True):
        frozen.train(mode)
        assert frozen.state_encoder.training == mode
        assert all(not m.training for m in frozen.flow_encoder.modules())


@pytest.mark.parametrize('device', DEVICES)
def test_gradient_isolation_and_fixed_features_across_policy_updates(device):
    g = frozen_model(environment(), device=device)
    states = [g.env.get_initial_state()]
    frozen_weights = snapshot(g.flow_encoder)
    features = g._encode(states, g.flow_encoder)[2].clone()
    assert not features.requires_grad
    policy_weights, head_weights = snapshot(g.state_encoder), snapshot(g.flow_head)
    g.state_flows(states).sum().backward()
    assert all(p.grad is None for p in g.state_encoder.parameters())
    assert all(p.grad is None for p in g.arg_model.parameters())
    assert all(p.grad is None for p in g.flow_encoder.parameters())
    assert sum(p.grad.abs().sum().item() for p in g.flow_head.parameters()) > 0
    g.opt.step()
    assert_unchanged(g.state_encoder, policy_weights)
    assert any(not torch.equal(v, head_weights[k]) for k, v in g.flow_head.state_dict().items())

    g.opt.zero_grad(set_to_none=True)
    g(states, forced_actions=[CoalescenceChoice(1, 2, delta_t=.2)])['log_pf'].sum().backward()
    assert sum(p.grad.abs().sum().item() for p in g.state_encoder.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in g.flow_head.parameters())
    g.opt.step()
    assert any(not torch.equal(v, policy_weights[k]) for k, v in g.state_encoder.state_dict().items())
    for mode in (False, True, False, True):
        g.train(mode)
        torch.testing.assert_close(g._encode(states, g.flow_encoder)[2], features, atol=0, rtol=0)
        assert all(not m.training for m in g.flow_encoder.modules())
    assert_unchanged(g.flow_encoder, frozen_weights)
    assert all(p.grad is None for p in g.flow_encoder.parameters())


def test_source_intermediate_terminal_routing_and_shared_scale_is_inactive():
    g = frozen_model(environment())
    source = g.env.get_initial_state()
    intermediate = g.env.apply_action(source, CoalescenceChoice(1, 2, delta_t=.2))
    states = [source, intermediate]
    batch, _, features = g._encode(states, g.flow_encoder)
    expected_residual = g.flow_head(torch.cat((features, batch.observations.state_scalars), -1)).squeeze(-1).double()
    baseline = torch.tensor([g.env.reward_fn.C+s.accumulated_log_prior+s.partial_log_likelihood
                             for s in states], dtype=torch.float64)
    # The initial centering offset is C and output scale is one.
    expected = baseline+expected_residual
    torch.testing.assert_close(g.state_flows(states), expected, atol=0, rtol=0)
    with patch.object(g.state_encoder, 'forward', side_effect=AssertionError('source used policy encoder')):
        torch.testing.assert_close(g.compute_log_Z(), expected[0], atol=1e-7, rtol=1e-7)
    for scale in (0., .3, 1.):
        g.flow_encoder_grad_scale = scale
        # Even a caller supplying a live policy summary must use frozen features.
        policy_batch, _, policy_summary = g.encode(states)
        torch.testing.assert_close(g.state_flows(states, policy_summary+100, policy_batch.observations),
                                   expected, atol=0, rtol=0)
    output, _ = RolloutWorker(g.env).rollout(g, 3, collect_flows=True, return_states=True)
    for row, terminal in enumerate(output['states']):
        assert output['state_flows'][row, output['lengths'][row]].item() == terminal.log_reward
        assert g.state_flows([terminal]).item() == terminal.log_reward


@pytest.mark.parametrize('snps', [False, True])
def test_branches_pack_missing_observations_against_their_own_cache(snps):
    env = environment(n=4, snps=snps, recombination=.01, length=4)
    g = frozen_model(env)
    with torch.no_grad():
        g.state_encoder.snp_encoder[-1].bias.add_(.5)
        g.state_encoder.material_encoder[-1].bias.add_(.5)
    initial = env.get_initial_state()
    choice = env.enumerate_policy_actions(initial)[1][0]
    split = env.apply_action(initial, replace(choice, breakpoint=1, delta_t=.2))
    permuted = split.clone()
    permuted.active_lineages.reverse()
    # Separate episode/source identities plus changing row order and duplicate rows.
    batches = [[initial], [split, env.get_initial_state()], [permuted], [permuted, permuted]]
    policy_cache, flow_cache = PooledLineageCache(), PooledLineageCache()
    # Deliberately different cache contents: compact policy observations omit
    # raw rows that the empty flow cache still needs.
    g.encode([initial], pooled_cache=policy_cache)
    for states in batches:
        actual = g(states, return_flows=True, pooled_cache=policy_cache, flow_pooled_cache=flow_cache)
        expected = g(states, forced_actions=actual['actions'], return_flows=True)
        assert_nested_close(actual['flows'], expected['flows'])
        assert_nested_close(actual['log_pf'], expected['log_pf'])
        assert policy_cache.pooled.requires_grad
        assert not flow_cache.pooled.requires_grad
        assert policy_cache.pooled.data_ptr() != flow_cache.pooled.data_ptr()
    with pytest.raises(ValueError, match='separate pooled caches'):
        g([initial], pooled_cache=policy_cache, flow_pooled_cache=policy_cache)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('accum', [1, 3])
def test_fresh_replay_and_accumulation_match_uncached_full_rescoring(device, accum):
    g = frozen_model(device=device)
    initial = snapshot(g)
    frozen_weights = snapshot(g.flow_encoder)
    mix = TrajectoryMixConfig(replay_fraction=.25, exploration_fraction=.25,
                              replay_min_size=2, replay_capacity=16, replay_grid_size=3)
    warm = Trainer(g, RolloutWorker(g.env), mix)
    with torch.no_grad():
        output, paths = warm.worker.rollout(g, 3, return_states=True)
    for path, terminal in zip(paths, output['states']):
        warm.buffer.add(g.env, path, terminal, 'policy', 1)
    buffer, random_state = copy.deepcopy(warm.buffer.state_dict()), rng_state(g.env)
    results = []
    for cached, trainer_type in ((False, ReferenceTrainer), (True, CaptureTrainer)):
        candidate = frozen_model(device=device)
        candidate.load_state_dict(initial)
        trainer = trainer_type(candidate, RolloutWorker(candidate.env, cache_pooled_embeddings=cached), mix)
        trainer.scored = []
        trainer.buffer.load_state_dict(copy.deepcopy(buffer))
        restore_rng(candidate.env, random_state)
        updates = []
        for _ in range(2):
            metrics = trainer.train_epoch(batch_size=7, grad_accum_steps=accum)
            assert metrics['fresh'] > 0 and metrics['replay'] > 0 and metrics['compatible_proposal'] > 0
            assert metrics['encoder_grad_norm'] > 0 and metrics['flow_grad_norm'] > 0
            assert_unchanged(candidate.flow_encoder, frozen_weights)
            assert all(p.grad is None for p in candidate.flow_encoder.parameters())
            gradients = {k: p.grad.clone() for k, p in candidate.named_parameters() if p.grad is not None}
            updates.append((metrics, gradients, snapshot(candidate), copy.deepcopy(candidate.opt.state_dict())))
        results.append(updates)
    assert_nested_close(results[0], results[1])


def test_rollout_cache_lifetimes_and_policy_only_skips_frozen_encoder():
    g = frozen_model()
    caches = []
    def create():
        cache = PooledLineageCache()
        caches.append(cache)
        return cache
    worker = RolloutWorker(g.env)
    with patch('gfn.rollout.PooledLineageCache', side_effect=create):
        with patch.object(g.flow_encoder, 'forward', side_effect=AssertionError('policy-only flow encoding')):
            with torch.no_grad():
                worker.rollout(g, 2)
                g.compute_event_probabilities(g.env.get_initial_state())
                g.initialize_flow_center(batch_size=2)
        assert len(caches) == 2  # One policy cache each for sampling and initialization.
        outputs, paths = worker.rollout(g, 2, collect_flows=True)
        assert len(caches) == 4
        g.get_loss_from_rollout_outputs(outputs).backward()
        g.opt.step(); g.opt.zero_grad(set_to_none=True)
        outputs, _ = worker.replay(g, paths)
        g.get_loss_from_rollout_outputs(outputs).backward()
        with pytest.raises(RolloutFailure):
            RolloutWorker(g.env, max_events=1).rollout(g, 2, collect_flows=True)
        assert len(caches) == 8
        assert all(c.pooled is None and not c.positions and not c.sources for c in caches)


def test_exact_resume_restores_original_frozen_weights_and_supports_evaluation_inference(tmp_path):
    g = frozen_model()
    frozen_weights = snapshot(g.flow_encoder)
    g.scheduler = WarmupCosineScheduler(g.opt, LearningRateConfig('cosine', 10, 0, .1, .1))
    mix = TrajectoryMixConfig(replay_fraction=.25, replay_min_size=2, replay_capacity=16, replay_grid_size=3)
    trainer = Trainer(g, RolloutWorker(g.env), mix)
    trainer.train_epoch(batch_size=4, grad_accum_steps=2)
    path = tmp_path/'frozen.pt'
    g.save(path, trainer=trainer)
    checkpoint = load_checkpoint(path)
    assert checkpoint['schema_version'] == 2 and checkpoint['metadata']['flow_head_version'] == 6
    assert checkpoint['metadata']['generator_config']['flow_encoder_mode'] == 'frozen_initial'
    expected_metrics = trainer.train_epoch(batch_size=4, grad_accum_steps=2)
    expected_weights, expected_optimizer = snapshot(g), copy.deepcopy(g.opt.state_dict())
    expected_trainer = copy.deepcopy(trainer.state_dict())
    restored = generator_from_checkpoint(checkpoint, optimizer=True, restore_random=True)
    restored_trainer = Trainer(restored, RolloutWorker(restored.env), mix)
    restored_trainer.load_state_dict(checkpoint['trainer'])
    assert_unchanged(restored.flow_encoder, frozen_weights)
    assert any(not torch.equal(p, restored.flow_encoder.encoder.state_dict()[k])
               for k, p in restored.state_encoder.state_dict().items())
    assert restored_trainer.train_epoch(batch_size=4, grad_accum_steps=2) == expected_metrics
    assert_unchanged(restored, expected_weights)
    assert_nested_close(restored.opt.state_dict(), expected_optimizer)
    assert restored_trainer.state_dict() == expected_trainer
    assert restored.scheduler.state_dict() == g.scheduler.state_dict()
    before_rng = rng_state(restored.env)
    metrics, _ = evaluate_generator(restored, 3, batch_size=2, density=False)
    assert metrics['eval_max_likelihood_error'] < 1e-9
    assert_nested_close(before_rng, rng_state(restored.env))
    assert restored.training and not restored.flow_encoder.training
    assert_unchanged(restored, expected_weights)
    with patch.object(FrozenFlowEncoder, 'forward', side_effect=AssertionError('inference should only sample policy')):
        manifest = run_inference(path, tmp_path/'inference', num_args=3, batch_size=2)
    assert manifest['summary']['num_completed'] == 3
    assert manifest['summary']['max_likelihood_error'] < 1e-9


def test_legacy_shared_checkpoint_and_cross_mode_rejection(tmp_path):
    shared, frozen = model(), frozen_model(environment())
    legacy = shared.save(tmp_path/'shared.pt')
    del legacy['metadata']['generator_config']['flow_encoder_mode']
    restored = generator_from_checkpoint(legacy, optimizer=True)
    assert restored.flow_encoder_mode == 'shared' and restored.flow_encoder is None
    assert_unchanged(restored, snapshot(shared))
    frozen_checkpoint = frozen.save(tmp_path/'frozen.pt')
    for target, checkpoint in ((frozen, legacy), (shared, frozen_checkpoint)):
        with pytest.raises(ValueError, match='cross-mode'):
            target.load(checkpoint)


@pytest.mark.parametrize('location', ['generator_config', 'resolved_config', 'trainer'])
def test_discarded_warmup_checkpoints_are_rejected(tmp_path, location):
    g = model()
    checkpoint = g.save(tmp_path/'original.pt')
    if location == 'trainer':
        checkpoint['trainer'] = {'flow_encoder_gradient': {}}
    else:
        checkpoint['metadata'].setdefault(location, {})['flow_encoder_warmup_steps'] = 500
    bad = tmp_path/'warmup.pt'
    torch.save(checkpoint, bad)
    for load in (lambda: load_checkpoint(bad), lambda: generator_from_checkpoint(checkpoint),
                 lambda: g.load(checkpoint)):
        with pytest.raises(ValueError, match='Discarded flow-gradient warm-up'):
            load()


def test_configuration_and_resume_mode_cannot_change(tmp_path):
    assert resolve_config({})['flow_encoder_mode'] == 'shared'
    c = resolve_config(parse_train_args(['--config', str(ROOT/'config/paper_datasets/blp_r1_batch128/config.yaml')]))
    assert c['flow_encoder_mode'] == 'frozen_initial'
    assert c['batch_size'] == 128 and c['resume_checkpoint'] is None and c['flow_warmup_steps'] == 0
    assert c['subtb_lambda'] == .9 and c['tb_loss_weight'] == .25
    assert c['output_path'] == 'runs/paper_datasets_stable/blp_r1_lambda09_batch128_frozen_initial'
    assert c['wandb_name'] == 'blp_r1_lambda09_batch128_frozen_initial'
    assert 'Inactive' in config_notes(c)['flow_encoder_grad_scale']
    assert resolve_config(parse_train_args(['--flow-encoder-mode', 'shared']))['flow_encoder_mode'] == 'shared'
    for invalid in ({'flow_encoder_mode': 'typo'}, {'flow_encoder_warmup_steps': 500},
                    {'flow_encoder_ramp_steps': 500}):
        with pytest.raises(ValueError):
            resolve_config(invalid)
    with pytest.raises(ValueError, match='flow_encoder_mode'):
        model(flow_encoder_mode='typo')
    from train import train
    for mode, other in (('shared', 'frozen_initial'), ('frozen_initial', 'shared')):
        g = model(flow_encoder_mode=mode)
        path = tmp_path/(mode+'.pt')
        g.save(path, metadata={'resolved_config': resolve_config({'flow_encoder_mode': mode})})
        with pytest.raises(ValueError, match='Cannot change flow_encoder_mode'):
            train(resume_checkpoint=str(path), output_path=str(tmp_path/'run'), flow_encoder_mode=other)
