"""Batched T=1 policy matches the retained scalar scorer and exact densities."""
import copy
from dataclasses import replace
import math
from types import MethodType
from unittest.mock import patch

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from breakpoint_model import SparseMixtureBreakpointPolicy
from env.actions import RecombinationChoice
from gfn.rollout import RolloutWorker
from policy.models import ARGModel, InfiniteSitesBreakpointHead
from policy.time_model import CwrExponentialTimeModel, CwrGammaTimeModel
from training.checkpoints import seed_everything
from validation.tests.test_fresh_score_reuse import make_model
from validation.tests.test_infinite_sites_neural import environment, model

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))]


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    seed_everything(39)


def scalar_forward(self, env, states, batch, lineages, summary, forced_actions=None, temperature=1.):
    # The retained tempered implementation contains the original scalar T=1
    # algorithm, including canonical candidate ordering and scalar mixtures.
    return self._forward_tempered(env, states, batch, lineages, summary, forced_actions, temperature)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('head', ['gamma', 'exponential'])
def test_complete_fixed_path_scores_gradients_and_updates(device, head):
    optimized = make_model(head, device)
    if head == 'gamma':
        with torch.no_grad():
            optimized.arg_model.time_head.shape_layer.weight.normal_(0, .02)
    reference = copy.deepcopy(optimized)
    reference.arg_model.forward = MethodType(scalar_forward, reference.arg_model)
    with torch.no_grad():
        _, paths = RolloutWorker(reference.env).rollout(reference, 7)
    assert any(isinstance(a, RecombinationChoice) for p in paths for a in p.actions)
    results = []
    for g in (reference, optimized):
        output, _ = RolloutWorker(g.env).replay(g, paths)
        loss = g.get_loss_from_rollout_outputs(output)
        loss.backward()
        grads = {k:p.grad.detach().clone() for k,p in g.named_parameters() if p.grad is not None}
        for prefix in ('state_encoder', 'arg_model', 'flow_head'):
            assert sum(float(v.double().square().sum()) for k,v in grads.items() if k.startswith(prefix)) > 0
        torch.nn.utils.clip_grad_norm_(g.parameters(), g.grad_clip)
        g.opt.step()
        results.append((output, grads, {k:p.detach().clone() for k,p in g.named_parameters()}))
    for left, right in zip(results[0], results[1]):
        assert left.keys() == right.keys()
        for key in left:
            torch.testing.assert_close(left[key], right[key], atol=2e-5, rtol=2e-5, msg=key)
    for prefix in ('state_encoder', 'arg_model', 'flow_head'):
        keys = [k for k in results[0][1] if k.startswith(prefix)]
        norm = sum(float(results[0][1][k].double().square().sum()) for k in keys)**.5
        error = sum(float((results[0][1][k].double()-results[1][1][k].double()).square().sum()) for k in keys)**.5
        assert error <= 2e-5*norm


@pytest.mark.parametrize('device', DEVICES)
def test_breakpoint_probabilities_and_sampling(device):
    head = InfiniteSitesBreakpointHead(8, hidden_size=12, components=3).to(device).double()
    with torch.no_grad():
        head.parameters_head[-1].weight.normal_(0, .1)
    context = torch.randn(8, device=device, dtype=torch.float64)
    for a, z in ((1, 1), (1, 9), (4, 9), (20000003, 20000009)):
        length = max(11, z+2)
        choice = RecombinationChoice(0, 10, a-1, z)
        _, _, parameters = head.parameters_for(choice, context, length)
        gaps = torch.arange(a, z+1, device=device)
        expected = SparseMixtureBreakpointPolicy.log_probabilities(gaps, a, z, parameters)
        spans = torch.tensor([[a, z]], device=device).expand(len(gaps), -1)
        _, actual = head.forward_batch(spans, context.expand(len(gaps), -1), length, gaps)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(actual.exp().sum(), actual.new_tensor(1.), atol=1e-12, rtol=1e-12)
        n = 12000
        with torch.no_grad():
            drawn, scores = head.forward_batch(spans[:1].expand(n, -1), context.expand(n, -1), length)
        assert not drawn.requires_grad and torch.isfinite(scores).all()
        counts = torch.bincount(drawn-a, minlength=z-a+1)/n
        probabilities = expected.exp()
        tolerance = 6*(probabilities*(1-probabilities)/n).sqrt()+.002
        assert ((counts-probabilities).abs() < tolerance).all()
        if a == z:
            assert torch.equal(scores, torch.zeros_like(scores))


@pytest.mark.parametrize('device', DEVICES)
def test_batched_categorical_support_and_frequencies(device):
    probabilities = torch.tensor([.1, .3, 0., .6], device=device, dtype=torch.float64)
    samples = ARGModel._sample_logs(probabilities.log().expand(30000, -1))
    frequency = torch.bincount(samples, minlength=4)/len(samples)
    torch.testing.assert_close(frequency.double(), probabilities, atol=.012, rtol=0)
    assert frequency[2] == 0


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('gamma', [False, True])
def test_combined_time_density_gradients_and_shared_parameters(device, gamma):
    cls = CwrGammaTimeModel if gamma else CwrExponentialTimeModel
    head = cls(4, 8, 0.).to(device)
    corrections = torch.tensor([[.3, -.5], [-.2, .7]], device=device, dtype=torch.float64)[:, :2 if gamma else 1]
    corrections.requires_grad_()
    baseline = corrections.new_tensor([2., 7.])
    with patch.object(head, '_distribution_parameters', wraps=head._distribution_parameters) as calculate:
        waits, scores = head.sample_and_log_time_pf(corrections, baseline)
        assert calculate.call_count == 1
    assert not waits.requires_grad
    mean_rate = baseline*corrections[:, 0].exp()
    if gamma:
        shape = corrections[:, 1].exp()
        expected = torch.distributions.Gamma(shape, mean_rate*shape).log_prob(waits.detach())
    else:
        expected = torch.distributions.Exponential(mean_rate).log_prob(waits.detach())
    torch.testing.assert_close(scores, expected, atol=1e-12, rtol=1e-12)
    actual_grad = torch.autograd.grad(scores.sum(), corrections, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(expected.sum(), corrections)[0]
    torch.testing.assert_close(actual_grad, reference_grad, atol=1e-12, rtol=1e-12)
    for value in (0., -1., math.inf, math.nan):
        with pytest.raises(ValueError):
            head.sample_and_log_time_pf(corrections, [value, 1.])
        with pytest.raises(ValueError):
            head.compute_log_time_pf(corrections, [value, 1.], baseline)
    for value in (math.inf, math.nan, 1000., -1000.):
        with pytest.raises(ValueError):
            head.sample_and_log_time_pf(torch.full_like(corrections, value), baseline)


@pytest.mark.parametrize('device', DEVICES)
def test_shape_one_preserves_exponential_sampling_and_detached_waits(device):
    exp = CwrExponentialTimeModel(4, 8, 0.).to(device)
    gamma = CwrGammaTimeModel(4, 8, 0.).to(device)
    corrections = torch.zeros(40, 2, device=device)
    rates = torch.logspace(-3, 3, 40, device=device, dtype=torch.float64)
    torch.manual_seed(909)
    a, score_a = exp.sample_and_log_time_pf(corrections[:, :1], rates)
    torch.manual_seed(909)
    b, score_b = gamma.sample_and_log_time_pf(corrections, rates)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(score_a, score_b, atol=0, rtol=0)
    for head, c in ((exp, corrections[:, :1]), (gamma, corrections)):
        c = c.clone().requires_grad_(); waits = a.clone().requires_grad_()
        grads = torch.autograd.grad(head.compute_log_time_pf(c, waits, rates).sum(), (c, waits), allow_unused=True)
        assert grads[0] is not None and grads[1] is None


@pytest.mark.parametrize('device', DEVICES)
def test_forced_actions_validation_reproducibility_and_tempered_path(device):
    g = make_model(device=device)
    states = [g.env.get_initial_state() for _ in range(8)]
    for temperature in (1., 1.5):
        seed_everything(53); first = g(states, temperature=temperature)
        seed_everything(53); second = g(states, temperature=temperature)
        assert first['actions'] == second['actions']
        forced = g(states, forced_actions=first['actions'], temperature=temperature)
        torch.testing.assert_close(first['factors'], forced['factors'], atol=1e-10, rtol=1e-10)
    with pytest.raises(ValueError, match='match'):
        g(states, forced_actions=first['actions'][:1])
    action = g.env.enumerate_policy_actions(states[0])[1][0]
    with pytest.raises(ValueError, match='Breakpoint'):
        g(states[:1], forced_actions=[replace(action, breakpoint=0, delta_t=.2)])
    with pytest.raises(ValueError, match='candidate'):
        g(states[:1], forced_actions=[replace(action, active_lineage_i=100, breakpoint=1, delta_t=.2)])
    # Different candidates/span widths, including a one-link-only choice.
    env = environment(recombination=.01, length=2)
    singleton = model(env, device=device)
    choice = env.enumerate_policy_actions(env.get_initial_state())[1][0]
    out = singleton([env.get_initial_state()], forced_actions=[replace(choice, breakpoint=1, delta_t=.1)])
    assert out['factors'][0, 2] == 0


@pytest.mark.parametrize('device', DEVICES)
def test_no_per_row_scalar_reads_and_one_breakpoint_network_call(device):
    class ScalarCount(TorchDispatchMode):
        def __init__(self):
            super().__init__(); self.count = 0
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if '_local_scalar_dense' in str(func):
                self.count += 1
            return func(*args, **(kwargs or {}))
    # A recombination-only support exercises the breakpoint network on every row.
    from validation.tests.test_infinite_sites_environment import environment as fixture_env
    env = fixture_env([[1,0], [1,1], [0,1]], [.5, 1.5], length=2)
    g = model(env, device=device)
    counts = []
    for size in (1, 8, 32):
        states = [env.get_initial_state() for _ in range(size)]
        batch, lineages, summary = g.encode(states)
        calls = []
        hook = g.arg_model.breakpoint_head.parameters_head.register_forward_hook(lambda *args: calls.append(1))
        count = ScalarCount()
        with count:
            g.arg_model(env, states, batch, lineages, summary)
        hook.remove()
        assert len(calls) == 1
        counts.append(count.count)
    assert counts[0] == counts[1] == counts[2]
