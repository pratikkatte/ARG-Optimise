"""Infinite-sites action heads; the generator owns the shared state encoder."""
from dataclasses import replace
import math
import torch
from torch import nn
from torch.distributions import Categorical
from env.actions import CoalescenceChoice, RecombinationChoice
from env.priors import total_event_rate
from breakpoint_model import SparseMixtureBreakpointPolicy
from .encoder import mlp
from .time_model import CwrGammaTimeModel, CwrExponentialTimeModel


class InfiniteSitesBreakpointHead(nn.Module):
    def __init__(self, context_dim, hidden_size=128, components=4, layers=1,
                 gap_hidden_size=64, gap_layers=0):
        super().__init__()
        self.components = components
        modules, width = [], context_dim+3
        for size, count in ((hidden_size, layers), (gap_hidden_size, gap_layers)):
            for _ in range(count):
                modules.extend([nn.Linear(width, size), nn.SiLU()]); width = size
        modules.append(nn.Linear(width, 3*components))
        self.parameters_head = nn.Sequential(*modules)
        output = self.parameters_head[-1]
        nn.init.zeros_(output.weight); nn.init.zeros_(output.bias)
        with torch.no_grad():
            fractions = (torch.arange(components)+.5)/components
            output.bias[components:2*components].copy_(torch.logit(fractions))
            output.bias[2*components:].fill_(math.log(.1/.9))

    def parameters_for(self, choice, context, length):
        a, z = SparseMixtureBreakpointPolicy.valid_span(choice, length)
        span = context.new_tensor([a/length, z/length, (z-a+1)/max(length-1, 1)])
        weights, centers, scales = self.parameters_head(torch.cat((context, span))).double().chunk(3)
        return a, z, (weights.log_softmax(-1), a-.5+(z-a+1)*centers.sigmoid(),
                      .1+(z-a+1)*scales.sigmoid())

    def forward(self, choice, context, length, breakpoint=None, temperature=1.0):
        a, z, parameters = self.parameters_for(choice, context, length)
        if breakpoint is None:
            breakpoint = SparseMixtureBreakpointPolicy.sample_gap(a, z, parameters, temperature=temperature)
        if not isinstance(breakpoint, int) or not a <= breakpoint <= z:
            raise ValueError('Breakpoint is outside the physical recombination span')
        score = SparseMixtureBreakpointPolicy.log_probabilities(breakpoint, a, z, parameters)
        if temperature != 1.:
            # Normalize the tempered distribution over EVERY physical link.
            normalizer = score.new_tensor(-torch.inf)
            for start in range(a, z+1, 1024):
                gaps = torch.arange(start, min(start+1024, z+1), device=context.device)
                logs = SparseMixtureBreakpointPolicy.log_probabilities(gaps, a, z, parameters)/temperature
                normalizer = torch.logaddexp(normalizer, torch.logsumexp(logs, 0))
            score = score/temperature-normalizer
        return breakpoint, score


class ARGModel(nn.Module):
    event_policy = 'cwr_residual'
    time_policy = 'cwr_exponential'
    continuous_time_head = 'gamma'

    def __init__(self, embedding_size=64, hidden_size=128, breakpoint_mixture_components=4,
                 breakpoint_mixture_hidden_dim=None, breakpoint_mixture_layers=1,
                 breakpoint_gap_hidden_size=64, breakpoint_gap_layers=0,
                 continuous_time_head='gamma', time_hidden_dim=None, time_layers=2):
        super().__init__()
        self.event_head = mlp(embedding_size, hidden_size, 2)
        self.action_head = mlp(4*embedding_size, hidden_size, 1)
        self.breakpoint_head = InfiniteSitesBreakpointHead(4*embedding_size, breakpoint_mixture_hidden_dim or hidden_size,
                    breakpoint_mixture_components, breakpoint_mixture_layers,
                    breakpoint_gap_hidden_size, breakpoint_gap_layers)
        self.continuous_time_head = continuous_time_head
        head = CwrGammaTimeModel if continuous_time_head == 'gamma' else CwrExponentialTimeModel
        self.time_head = head(4*embedding_size+4, time_hidden_dim or hidden_size, 0., layers=time_layers)
        for head in (self.event_head, self.action_head):
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

    def event_log_probs(self, batch, summary, temperature=1.0):
        hazards = summary.new_tensor(batch.allowed_hazards, dtype=torch.float64)
        available = hazards > 0
        if not available.any(-1).all():
            raise ValueError('Cannot sample an event from a terminal or dead-end state')
        return ((hazards.log()+self.event_head(summary).double())/temperature).masked_fill(~available, -torch.inf).log_softmax(-1)

    @staticmethod
    def contexts(choices, lineage, summary):
        indices = torch.tensor([a.active_lineage_i for a in choices], device=lineage.device)
        first = lineage[indices]
        if isinstance(choices[0], CoalescenceChoice):
            second = lineage[torch.tensor([a.active_lineage_j for a in choices], device=lineage.device)]
            return torch.cat((first+second, (first-second).abs(), first*second,
                              summary.expand(len(choices), -1)), -1)
        return torch.cat((first, torch.zeros_like(first), torch.zeros_like(first),
                          summary.expand(len(choices), -1)), -1)

    def forward(self, env, states, batch, lineages, summary, forced_actions=None, temperature=1.0):
        if not math.isfinite(temperature) or temperature < 1:
            raise ValueError('Policy temperature must be finite and >= 1')
        event_logs = self.event_log_probs(batch, summary, temperature)
        event_indices = (Categorical(logits=event_logs).sample().tolist() if forced_actions is None else
                         [int(isinstance(a, RecombinationChoice)) for a in forced_actions])
        choices_by_row = [batch.actions[row][kind] for row, kind in enumerate(event_indices)]
        if any(not choices for choices in choices_by_row):
            raise ValueError('Forced action has no compatible support')
        candidate_contexts = [self.contexts(choices, lineages[row], summary[row])
                              for row, choices in enumerate(choices_by_row)]
        # Score every candidate in one neural call; normalization stays per ARG.
        hidden = self.action_head[:-1](torch.cat(candidate_contexts))
        # The final scalar bias cancels in every candidate softmax. Omit it
        # here to avoid Adam amplifying roundoff in its mathematically zero
        # gradient when the candidate batch size changes. Keep the parameter
        # in the module so existing checkpoint layouts remain compatible.
        residuals = torch.nn.functional.linear(hidden, self.action_head[-1].weight).squeeze(-1).double().split(
            [len(choices) for choices in choices_by_row])
        actions, contexts, factors = [], [], []
        for row, kind in enumerate(event_indices):
            choices, context = choices_by_row[row], candidate_contexts[row]
            baseline = context.new_tensor([a.breakpoint_count if kind else 1 for a in choices], dtype=torch.float64).log()
            logits = baseline+residuals[row]
            logs = (logits/temperature).log_softmax(-1)
            if forced_actions is None:
                selected = int(Categorical(logits=logs).sample())
            else:
                forced = forced_actions[row]
                canonical = replace(forced, delta_t=None, time_action=None,
                                    **({'breakpoint': None} if kind else {}))
                try:
                    selected = choices.index(canonical)
                except ValueError as exc:
                    raise ValueError('Forced action is not a compatible physical candidate') from exc
            action, chosen_context = choices[selected], context[selected]
            breakpoint_log = logs.new_zeros(())
            if kind:
                bp, breakpoint_log = self.breakpoint_head(action, chosen_context, env.sequence_length,
                                       None if forced_actions is None else forced_actions[row].breakpoint, temperature=temperature)
                action = replace(action, breakpoint=bp)
            actions.append(action); contexts.append(chosen_context)
            factors.append(torch.stack((event_logs[row, kind], logs[selected], breakpoint_log)))
        rates = [total_event_rate(r) for r in batch.physical_rates]
        contexts = torch.stack(contexts)
        timing = contexts.new_tensor([[math.log1p(s.current_time), math.log(rate),
                    float(isinstance(a, RecombinationChoice)),
                    a.breakpoint/env.sequence_length if isinstance(a, RecombinationChoice) else 0.]
                    for s, a, rate in zip(states, actions, rates)])
        rates = contexts.new_tensor(rates, dtype=torch.float64)
        corrections = self.time_head(torch.cat((contexts, timing), -1))
        waits = (self.time_head.sample(corrections, rates) if forced_actions is None else
                 rates.new_tensor([a.delta_t for a in forced_actions]))
        time_logs = self.time_head.compute_log_time_pf(corrections, waits, rates)
        actions = [replace(a, delta_t=float(dt)) for a, dt in zip(actions, waits.tolist())]
        factors = torch.cat((torch.stack(factors), time_logs[:, None]), -1)
        if not torch.isfinite(factors).all():
            raise FloatingPointError('Nonfinite policy factor')
        return factors.sum(-1), actions, factors
