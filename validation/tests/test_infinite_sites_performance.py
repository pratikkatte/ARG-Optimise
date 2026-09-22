"""Performance paths must preserve exact support, replay, and gradients."""
from unittest.mock import patch

import numpy as np
import pytest
import torch

from gfn.rollout import RolloutWorker
from training.checkpoints import seed_everything
from training.configuration import resolve_config, parse_train_args
from validation.tests.test_infinite_sites_environment import environment, assert_same
from validation.tests.test_infinite_sites_neural import model


@pytest.mark.parametrize('samples', [4, 64, 70])
def test_vectorized_support_matches_scalar_bitsets(samples):
    genotypes = np.zeros((samples, 3), dtype=np.uint8)
    genotypes[[0, samples-1], 0] = 1
    genotypes[[1, samples-1], 1] = 1
    genotypes[0, 2] = 1
    env = environment(genotypes, [.25, 1.5, 3.75], length=4)
    state = env.get_initial_state()
    for _ in range(25):
        coal, recomb = env.enumerate_actions(state)
        expected = [a for a in coal if not env.incompatible_sites(state, a)]
        assert env.enumerate_policy_actions(state) == (expected, recomb)
        if state.is_done:
            break
        state = env.apply_action(state, env.sample_compatible_step(state).action)


def test_owned_steps_match_nonmutating_steps_and_exact_priors():
    env = environment(length=4, recombination_rate=.5)
    reference, owned = env.get_initial_state(), env.get_initial_state()
    for _ in range(2000):
        step = env.sample_compatible_step(reference)
        before = reference.clone(copy_partials=True)
        advanced = env.apply_action(reference, step.action)
        assert_same(reference, before)
        identity = owned
        owned, prior = env.step_owned_state(owned, step.action)
        assert owned is identity and prior == step.log_prior
        assert_same(owned, advanced)
        assert owned.actions == advanced.actions
        for key, node in owned.all_nodes.items():
            assert node.parents == advanced.all_nodes[key].parents
            assert node.children == advanced.all_nodes[key].children
        reference = advanced
        if reference.is_done:
            break
    assert reference.is_done
    assert owned.log_reward == reference.log_reward


@pytest.mark.parametrize('flow_scale_mode', ['fixed', 'empirical'])
def test_initialization_batches_keep_every_target_including_partial_batch(flow_scale_mode):
    torch.set_num_threads(1)
    g = model(environment(recombination_rate=.01), flow_scale_mode=flow_scale_mode)
    g.init_z_sample_count = 5
    worker = RolloutWorker(g.env)
    seed_everything(19)
    with torch.no_grad():
        targets = []
        for count in (2, 2, 1):
            outputs, _ = worker.rollout(g, count)
            targets.append(outputs['log_rewards']-outputs['log_paths_pf'].sum(-1))
    targets = torch.cat(targets)
    seed_everything(19)
    with patch.object(RolloutWorker, 'rollout', autospec=True,
                      side_effect=RolloutWorker.rollout) as rollout:
        g.initialize_flow_center(batch_size=2)
    assert [call.kwargs['episodes'] for call in rollout.call_args_list] == [2, 2, 1]
    torch.testing.assert_close(g.flow_init_offset, targets.mean(), atol=0, rtol=0)
    expected_scale = targets.std(unbiased=False).clamp_min(1) if flow_scale_mode == 'empirical' else targets.new_tensor(1.)
    torch.testing.assert_close(g.flow_output_scale, expected_scale, atol=0, rtol=0)
    assert resolve_config(parse_train_args(['--init-z-batch-size', '7']))['init_z_batch_size'] == 7
    with pytest.raises(ValueError, match='init_z_batch_size'):
        resolve_config(dict(init_z_batch_size=0))


def test_batched_action_head_matches_rowwise_scores_and_gradients():
    torch.set_num_threads(1)
    g = model(environment(length=4))
    with torch.no_grad():
        g.arg_model.action_head[-1].weight.normal_(0, .1)
        _, paths = RolloutWorker(g.env).rollout(g, 3)
    states = [g.env.get_initial_state() for _ in paths]
    actions = [p.actions[0] for p in paths]
    # Different candidate counts exercise the concatenation/split boundary.
    states[1] = g.env.apply_action(states[1], paths[1].actions[0])
    actions[1] = paths[1].actions[1]
    batched = g(states, forced_actions=actions)['log_pf']
    batched.sum().backward()
    gradients = {name: p.grad.clone() for name, p in g.named_parameters() if p.grad is not None}
    g.opt.zero_grad(set_to_none=True)
    rowwise = torch.cat([g([s], forced_actions=[a])['log_pf'] for s, a in zip(states, actions)])
    rowwise.sum().backward()
    torch.testing.assert_close(batched, rowwise, atol=1e-6, rtol=1e-6)
    for name, p in g.named_parameters():
        if name in gradients:
            torch.testing.assert_close(p.grad, gradients[name], atol=2e-5, rtol=2e-5)
