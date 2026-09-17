"""Float64 all-segment SubTB loss and diagnostics."""
import math
import torch
from gfn.objectives import validate_objective

def _logadd(a, b):
    m = max(a, b)
    return m + math.log1p(math.exp(min(a, b) - m))


def geometric_subtb_loss(log_pf, log_pb, flows, lengths, log_rewards, subtb_lambda=0.9):
    """Average normalized all-segment losses equally across complete trajectories.

    ``flows[b, :lengths[b]+1]`` includes exact source and terminal boundaries.
    For zero actions, flows[b, 0] is the source; log_rewards supplies the terminal.
    Padding is ignored, including nonfinite padding. Arithmetic is float64.
    """
    validate_objective("subtb", subtb_lambda, 1.0)
    if log_pf.ndim != 2 or log_pb.shape != log_pf.shape:
        raise ValueError("forward/backward paths must have matching [B,T] shapes")
    b, t = log_pf.shape
    if b == 0 or flows.shape != (b, t + 1) or lengths.shape != (b,) or log_rewards.shape != (b,):
        raise ValueError("invalid flows, lengths, rewards or empty batch shape")
    if lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("lengths must be integer tensors")
    if any(x.device != log_pf.device for x in (log_pb, flows, lengths, log_rewards)):
        raise ValueError("all inputs must be on the same device")
    if bool(((lengths < 0) | (lengths > t)).any()):
        raise ValueError("lengths outside padded trajectory bounds")
    mask = torch.arange(t, device=lengths.device)[None, :] < lengths[:, None]
    state_mask = torch.arange(t + 1, device=lengths.device)[None, :] <= lengths[:, None]
    pf = torch.where(mask, log_pf.double(), 0.0)
    pb = torch.where(mask, log_pb.double(), 0.0)
    f = torch.where(state_mask, flows.double(), 0.0)
    rewards = log_rewards.double()
    if not all(bool(torch.isfinite(x).all()) for x in (pf, pb, f, rewards)):
        raise ValueError("non-finite active trajectory inputs")
    # Subtract endpoints first, preserving small residuals at large flow offsets.
    delta = torch.where(mask, (f[:, :-1] - f[:, 1:]) + (pf - pb), 0.0)
    mean = f.new_zeros(b)
    variance = f.new_zeros(b)
    result = (f[:, 0] - rewards).square() * (lengths == 0)
    log_suffix_weight = log_total_weight = -math.inf
    log_lambda = math.log(subtb_lambda) if subtb_lambda else -math.inf
    for end in range(t):
        log_extended = log_lambda + log_suffix_weight
        log_suffix_weight = _logadd(0.0, log_extended)
        # Mixture of a new one-step suffix and all extended previous suffixes.
        old_fraction = math.exp(log_extended - log_suffix_weight)
        new_fraction = math.exp(-log_suffix_weight)
        variance = old_fraction * variance + old_fraction * new_fraction * mean.square()
        mean = delta[:, end] + old_fraction * mean
        log_total_new = _logadd(log_total_weight, log_suffix_weight)
        weight = math.exp(log_suffix_weight - log_total_new)
        old_weight = math.exp(log_total_weight - log_total_new)
        result = torch.where(mask[:, end], old_weight * result + weight * (variance + mean.square()), result)
        log_total_weight = log_total_new
    return result.mean()


@torch.no_grad()
def subtb_diagnostics(log_pf, log_pb, flows, lengths, log_rewards, loss, subtb_lambda=0.9):
    """Separate terminal-boundary error from interior error without enumerating segments."""
    b, t = log_pf.shape
    if t == 0:
        return {"subtb_terminal_loss": float(loss), "subtb_interior_loss": 0.0,
                "subtb_terminal_weight": 1.0}
    index = torch.arange(t, device=lengths.device)[None, :]
    active = index < lengths[:, None]
    increments = torch.where(active, log_pf.double() - log_pb.double(), 0.0)
    prefix = torch.cat((increments.new_zeros(b, 1), increments.cumsum(-1)), dim=-1)
    tail = (flows[:, :-1].double() - log_rewards[:, None].double()
            + prefix.gather(1, lengths[:, None]) - prefix[:, :-1])
    tail = torch.where(active, tail, 0.0)
    distance = lengths[:, None] - index
    if subtb_lambda == 0:
        weights = (distance == 1).double() / lengths[:, None].clamp_min(1)
    else:
        log_lambda = math.log(subtb_lambda)
        # Sum over segment lengths, accounting for every possible start.
        log_total = torch.logsumexp(torch.where(
            active, distance.clamp_min(1).double().log() + index.double() * log_lambda, -torch.inf
        ), dim=-1, keepdim=True)
        log_total = torch.where(lengths[:, None] > 0, log_total, 0.0)
        weights = torch.where(active, ((distance - 1).double() * log_lambda - log_total).exp(), 0.0)
    terminal = (weights * tail.square()).sum(-1)
    terminal = torch.where(lengths == 0, (flows[:, 0] - log_rewards).square(), terminal)
    terminal_loss = float(terminal.mean())
    weight = torch.where(lengths == 0, 1.0, weights.sum(-1)).mean()
    return {"subtb_terminal_loss": terminal_loss,
            "subtb_interior_loss": max(0.0, float(loss) - terminal_loss),
            "subtb_terminal_weight": float(weight)}
