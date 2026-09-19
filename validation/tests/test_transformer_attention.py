"""SDPA preserves FP32 attention, padding semantics, and first-order gradients."""
import copy
from types import MethodType

import pytest
import torch

from policy.transformer import MultiHeadSelfAttention, TransformerEncoder


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))]


def explicit_attention(self, x, key_padding_mask=None):
    """Independent unfused reference for the original attention equation."""
    batch, tokens, dim = x.shape
    q, k, v = [part.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
               for part in self.qkv(x).chunk(3, dim=-1)]
    weights = (q @ k.transpose(-2, -1)) / self.head_dim ** 0.5
    if key_padding_mask is not None:
        weights = weights.masked_fill(key_padding_mask[:, None, None, :], -torch.inf)
    weights = self.attn_drop(weights.softmax(-1))
    values = (weights @ v).transpose(1, 2).reshape(batch, tokens, dim)
    return self.proj_drop(self.proj(values))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dim,depth', [(32, 0), (64, 6)])
@pytest.mark.parametrize('mask_kind', ['none', 'padded', 'summary_only'])
def test_sdpa_matches_explicit_forward_and_gradients(device, dim, depth, mask_kind):
    torch.set_num_threads(1)
    torch.manual_seed(91)
    model = (TransformerEncoder(dim, depth, 4) if depth
             else MultiHeadSelfAttention(dim, 4)).to(device)
    reference = copy.deepcopy(model)
    for module in reference.modules():
        if isinstance(module, MultiHeadSelfAttention):
            module.forward = MethodType(explicit_attention, module)
    x = torch.randn(3, 13, dim, device=device, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    mask = None
    if mask_kind != 'none':
        mask = torch.ones(3, 13, dtype=torch.bool, device=device)
        # The encoder always retains its summary token, even for empty states.
        mask[:, 0] = False
        if mask_kind == 'padded':
            mask[0] = False
            mask[1, :7] = False
            mask[2, [2, 4, 8]] = False

    actual = model(x, key_padding_mask=mask)
    expected = reference(expected_x, key_padding_mask=mask)
    assert actual.dtype == torch.float32
    assert all(p.dtype == torch.float32 for p in model.parameters())
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    upstream = torch.randn_like(actual)
    (actual * upstream).mean().backward()
    (expected * upstream).mean().backward()
    torch.testing.assert_close(x.grad, expected_x.grad, rtol=2e-5, atol=1e-7)
    for (name, parameter), (_, expected_parameter) in zip(
            model.named_parameters(), reference.named_parameters()):
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, expected_parameter.grad,
                                   rtol=2e-5, atol=1e-7, msg=name)


@pytest.mark.parametrize('device', DEVICES)
def test_padding_cannot_change_valid_outputs_or_receive_gradients(device):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    model = MultiHeadSelfAttention(64, 4).to(device)
    x = torch.randn(2, 11, 64, device=device, requires_grad=True)
    mask = torch.arange(11, device=device)[None, :] >= torch.tensor(
        [5, 1], device=device)[:, None]
    actual = model(x, key_padding_mask=mask)
    changed = x.detach().clone()
    changed[mask] = torch.randn_like(changed[mask]) * 100
    with torch.no_grad():
        other = model(changed, key_padding_mask=mask)
    torch.testing.assert_close(actual[~mask], other[~mask], rtol=1e-5, atol=1e-6)
    actual[~mask].square().sum().backward()
    assert torch.count_nonzero(x.grad[mask]) == 0
    assert torch.count_nonzero(x.grad[~mask]) > 0


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('probability', [0.25, 1.0])
def test_attention_dropout_respects_train_and_eval(device, probability):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    model = MultiHeadSelfAttention(32, 4, attention_dropout=probability).to(device)
    x = torch.randn(2, 7, 32, device=device, requires_grad=True)
    training = model(x)
    if probability == 1.0:
        torch.testing.assert_close(training, model.proj.bias.expand_as(training),
                                   rtol=0, atol=0)
    else:
        with torch.no_grad():
            assert not torch.equal(training, model(x))
    training.sum().backward()
    assert torch.isfinite(x.grad).all()
    if probability == 1.0:
        assert torch.count_nonzero(x.grad) == 0
        assert torch.count_nonzero(model.qkv.weight.grad) == 0
        assert torch.count_nonzero(model.qkv.bias.grad) == 0
        assert torch.count_nonzero(model.proj.weight.grad) == 0
    else:
        assert torch.count_nonzero(x.grad) > 0
    with torch.no_grad():
        model.eval()
        evaluation = model(x)
        torch.testing.assert_close(evaluation, explicit_attention(model, x),
                                   rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(evaluation, model(x), rtol=0, atol=0)
        assert not torch.equal(evaluation, training)
