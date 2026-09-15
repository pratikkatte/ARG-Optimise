"""Exact geometric SubTB with linear scalar work and autograd storage."""
import math

import torch
from gfn.objectives import validate_objective
from policy.models import PackedLineageFeatures
from gfn.flow_encoder import FrozenFlowEncoder
from gfn.flow_likelihood import PartialLikelihoodTracker


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


class SubTBMixin:
    """SubTB operations on the public generator; owns no separate state."""

    def _initialize_subtb_environment(self):
        self.env.flow_likelihood = PartialLikelihoodTracker(self.env)

    def _initialize_subtb_flow(self):
        # CPU construction under fork_rng leaves policy/sampling RNG unchanged.
        with torch.random.fork_rng(devices=[]):
            self.flow_head = torch.nn.Sequential(
                torch.nn.Linear(self.model_kwargs.get("embedding_size", 32) + (8 if self.flow_head_version >= 2 else 4),
                                self.model_kwargs.get("hidden_size", 64), device="cpu"),
                torch.nn.SiLU(),
                torch.nn.Linear(self.model_kwargs.get("hidden_size", 64), 1, device="cpu"),
            ).to(self.device)
            torch.nn.init.zeros_(self.flow_head[-1].weight)
            torch.nn.init.zeros_(self.flow_head[-1].bias)
        self.register_buffer("flow_init_offset", torch.tensor(
            self.last_log_z_target, dtype=torch.float64, device=self.device))
        if self.flow_head_version >= 2:
            self.register_buffer("flow_output_scale", torch.tensor(
                max(1.0, self.initial_log_z_target_std), dtype=torch.float64, device=self.device))
        if self.flow_head_version >= 3:
            self.flow_encoder = FrozenFlowEncoder(self.arg_model)
        if self.flow_head_version == 4:
            # A fixed initialization buffer, not a view or detached alias
            # of the live normalizer. Subsequent Z updates cannot move it.
            self.register_buffer('flow_baseline_log_z', self.compute_log_Z().detach().clone())
        self.flow_params = list(self.flow_head.parameters())
        self.opt.add_param_group({"params": self.flow_params, "lr": self.flow_lr})
        self.gradient_clipping_params.extend(self.flow_params)

    def compute_source_log_flow(self):
        if not self.neural_source_flow:
            return self._Z
        state = self.env.get_initial_state()
        features = PackedLineageFeatures(
            torch.stack([self.env.evolution_model.normalize_partials(node.partials)
                         for node in state.active_lineages]),
            (0, self.env.num_sequences))
        summary = self.flow_encoder(features, torch.tensor([self.env.num_sequences], device=self.device))
        previous = getattr(self, '_last_flow_corrections', None)
        value = self.state_flows([state], summary)[0]
        self._last_flow_corrections = previous
        return value

    def state_flows(self, states, summary_reps):
        feature_rows = [
            [s.accumulated_log_prior / self.env.sequence_length,
             math.log1p(s.current_time), math.log1p(len(s.active_lineages)),
             s.total_active_blocks / self.env.num_blocks] for s in states
        ]
        if self.flow_head_version >= 2:
            initial_ll = self.env.flow_likelihood.initial_log_likelihood
            scale = self.flow_output_scale.item()
            remaining = []
            partial_likelihoods = []
            for state, row in zip(states, feature_rows):
                partial_ll = state.partial_log_likelihood
                if partial_ll is None and self.neural_source_flow and state.is_done:
                    # The exact terminal boundary also accepts inference states,
                    # where the optional intermediate likelihood tracker is off.
                    partial_ll = state.log_reward - self.env.reward_fn.C - state.accumulated_log_prior
                if partial_ll is None:
                    raise ValueError("Likelihood flow requires tracked partial likelihoods")
                partial_likelihoods.append(partial_ll)
                fraction = (state.total_active_blocks / self.env.num_blocks - 1) / max(self.env.num_sequences - 1, 1)
                ages = [max(0.0, state.current_time - node.time) for node in state.active_lineages]
                mean_age = sum(ages) / len(ages)
                age_std = math.sqrt(sum((age - mean_age)**2 for age in ages) / len(ages))
                remaining.append(fraction)
                row.extend([(partial_ll - initial_ll) / scale, fraction,
                            math.log1p(mean_age), math.log1p(age_std)])
        features = summary_reps.new_tensor(feature_rows)
        residual = self.flow_head(torch.cat((summary_reps, features), dim=-1)).squeeze(-1).double()
        prior = torch.tensor([s.accumulated_log_prior for s in states],
                             dtype=torch.float64, device=self.device)
        predicted = self.flow_init_offset + prior + residual
        if self.flow_head_version >= 2:
            partial_ll = prior.new_tensor(partial_likelihoods)
            source_potential = self.env.reward_fn.C + initial_ll
            baseline_log_z = (self.flow_init_offset if self.neural_source_flow else
                              self.flow_baseline_log_z if self.flow_head_version == 4
                              else self.compute_log_Z())
            predicted = (self.env.reward_fn.C + prior + partial_ll
                         + prior.new_tensor(remaining) * (baseline_log_z - source_potential)
                         + self.flow_output_scale * residual)
        self._last_flow_corrections = residual.detach() * (
            self.flow_output_scale if self.flow_head_version >= 2 else 1.0)
        # Each forward event allocates new node IDs, so this identifies the source in O(1).
        source = torch.tensor([s.max_node_idx == self.env.num_sequences - 1 for s in states], device=self.device)
        if self.neural_source_flow:
            terminal = torch.tensor([s.is_done for s in states], device=self.device)
            reward = prior.new_tensor([s.log_reward if s.is_done else 0. for s in states])
            return torch.where(terminal, reward, predicted)
        return torch.where(source, self.compute_log_Z().double(), predicted)

    def _get_subtb_loss_from_rollout_outputs(self, rollout_outputs):
        return geometric_subtb_loss(
            rollout_outputs["log_paths_pf"], rollout_outputs["log_paths_pb"],
            rollout_outputs["state_flows"], rollout_outputs["lengths"],
            rollout_outputs["log_rewards"], self.subtb_lambda,
        )

    @torch.no_grad()
    def get_subtb_diagnostics(self, outputs, loss=None):
        if loss is None:
            loss = self.get_loss_from_rollout_outputs(outputs)
        metrics = subtb_diagnostics(outputs['log_paths_pf'], outputs['log_paths_pb'],
            outputs['state_flows'], outputs['lengths'], outputs['log_rewards'], loss, self.subtb_lambda)
        if 'flow_corrections' in outputs:
            corrections = outputs['flow_corrections']
            index = torch.arange(corrections.shape[1], device=corrections.device)[None, :]
            values = corrections[(index > 0) & (index < outputs['lengths'][:, None])]
            metrics['flow_correction_mean'] = values.mean().item() if values.numel() else 0.0
            metrics['flow_correction_std'] = values.std(unbiased=False).item() if values.numel() else 0.0
        return metrics


    def _accumulate_subtb_diagnostics(self, rollout_outputs, loss, factor):
        for key, value in self.get_subtb_diagnostics(rollout_outputs, loss).items():
            self.accumulated_diagnostics[key] = self.accumulated_diagnostics.get(key, 0.0) + value / factor

    def _update_subtb_info(self, info):
        info["policy_grad_norm"] = self.policy_grad_norm()
        info["flow_head_grad_norm"] = self._grad_norm(self.flow_params)
        if not self.neural_source_flow:
            info["log_z_grad"] = self.log_z_grad()
        info.update(self.accumulated_diagnostics)
        self.accumulated_diagnostics = {}

    def _clip_subtb_gradients(self):
        # The scaled flow head must not determine the policy clipping factor.
        torch.nn.utils.clip_grad_norm_(self.policy_params, self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.flow_params, self.grad_clip)
