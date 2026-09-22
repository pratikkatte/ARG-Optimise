"""Batched startup and faster masks must preserve the scientific calculation."""
import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from gfn.rollout import RolloutWorker
from training.checkpoints import seed_everything, generator_from_checkpoint
from training.configuration import parse_train_args, resolve_config
from validation.tests.test_infinite_sites_neural import model, environment
from validation.tests.test_infinite_sites_environment import environment as snp_environment


def reference_actions(env, state):
    coal, recomb = env.enumerate_actions(state)
    return ([a for a in coal if not env.incompatible_sites(state, a)],
            recomb if env.recombination_rate > 0 else [])


@pytest.mark.parametrize('sample_count,snps', [(4, True), (4, False), (67, True)])
def test_fast_mask_matches_full_conflicts(sample_count, snps):
    # Crossing SNP patterns require recombination; a site on the integer link
    # also exercises half-open routing. 67 samples catches fixed-width bitsets.
    genotypes = np.zeros((sample_count, 3 if snps else 0), dtype=np.uint8)
    if snps:
        genotypes[[0, sample_count-1], 0] = 1
        genotypes[[1, sample_count-1], 1] = 1
        genotypes[2, 2] = 1
    env = snp_environment(genotypes, [.5, 1., 3.5] if snps else [], length=4)
    state = env.get_initial_state()
    for _ in range(1000):
        expected = reference_actions(env, state)
        assert env.enumerate_policy_actions(state) == expected
        if state.is_done:
            break
        step = env.sample_compatible_step(state)
        state = env.apply_action(state, step.action)
    else:
        pytest.fail('Compatibility fixture exceeded its event limit')
    reference = env.evaluate_terminal(state)
    assert state.partial_log_likelihood == pytest.approx(reference.log_likelihood, abs=1e-9)


def test_singleton_initialization_preserves_seeded_reference():
    torch.set_num_threads(1)
    seed_everything(7)
    first = model(environment(recombination=.001, length=4))
    second = model(environment(recombination=.001, length=4))
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    seed_everything(27)
    with patch.object(first.env, 'enumerate_policy_actions',
                      side_effect=lambda s: reference_actions(first.env, s)):
        first.initialize_flow_center()
    expected_rng = torch.get_rng_state()
    seed_everything(27)
    second.initialize_flow_center()
    assert torch.equal(expected_rng, torch.get_rng_state())
    torch.testing.assert_close(first.flow_init_offset, second.flow_init_offset, atol=0, rtol=0)
    torch.testing.assert_close(first.flow_output_scale, second.flow_output_scale, atol=0, rtol=0)


def test_initialization_batches_keep_every_draw_and_exact_scores(tmp_path):
    torch.set_num_threads(1)
    seed_everything(7)
    g = model(environment(recombination=.01, length=4), init_z_batch_size=2)
    g.init_z_sample_count = 5
    batches = []
    original = RolloutWorker.rollout

    def capture(worker, generator, episodes=1):
        assert not torch.is_grad_enabled()
        outputs, paths = original(worker, generator, episodes, return_states=True)
        batches.append((outputs, paths))
        return outputs, paths

    with patch.object(RolloutWorker, 'rollout', capture), \
            patch('env.env.evaluate_infinite_sites', side_effect=AssertionError('initialization used reference')):
        g.initialize_flow_center()
    assert [len(paths) for _, paths in batches] == [2, 2, 1]
    targets = []
    for outputs, paths in batches:
        with torch.no_grad():
            rescored, _ = RolloutWorker(g.env).replay(g, paths)
        torch.testing.assert_close(outputs['log_paths_pf'], rescored['log_paths_pf'], atol=1e-10, rtol=0)
        for row, state in enumerate(outputs['states']):
            reference = g.env.evaluate_terminal(state)
            assert state.partial_log_likelihood == pytest.approx(reference.log_likelihood, abs=1e-9)
            targets.append(outputs['log_rewards'][row]-outputs['log_paths_pf'][row].sum())
    targets = torch.stack(targets)
    torch.testing.assert_close(g.flow_init_offset, targets.mean(), atol=0, rtol=0)
    torch.testing.assert_close(g.flow_output_scale, targets.std(unbiased=False).clamp_min(1.), atol=0, rtol=0)
    assert all(p.grad is None for p in g.parameters())
    checkpoint = g.save(tmp_path/'model.pt')
    restored = generator_from_checkpoint(checkpoint)
    assert restored.init_z_batch_size == 2
    torch.testing.assert_close(restored.flow_init_offset, g.flow_init_offset, atol=0, rtol=0)
    # Existing infinite-sites checkpoints predate the optional batching setting.
    del checkpoint['metadata']['generator_config']['init_z_batch_size']
    assert generator_from_checkpoint(checkpoint).init_z_batch_size == 1


def test_initialization_batch_configuration():
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='init_z_batch_size'):
            resolve_config(dict(init_z_batch_size=invalid))
        with pytest.raises(ValueError, match='init_z_batch_size'):
            model(init_z_batch_size=invalid)
    assert resolve_config({})['init_z_batch_size'] == 1
    assert resolve_config(parse_train_args(['--init-z-batch-size', '16']))['init_z_batch_size'] == 16
