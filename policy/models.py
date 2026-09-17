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
from .time_model import CwrGammaTimeModel


class InfiniteSitesBreakpointHead(nn.Module):
    def __init__(self, context_dim, hidden_size=128, components=4):
        super().__init__()
        self.components = components
        self.parameters_head = mlp(context_dim+3, hidden_size, 3*components)
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

    def forward(self, choice, context, length, breakpoint=None):
        a, z, parameters = self.parameters_for(choice, context, length)
        if breakpoint is None:
            breakpoint = SparseMixtureBreakpointPolicy.sample_gap(a, z, parameters)
        if not isinstance(breakpoint, int) or not a <= breakpoint <= z:
            raise ValueError('Breakpoint is outside the physical recombination span')
        return breakpoint, SparseMixtureBreakpointPolicy.log_probabilities(breakpoint, a, z, parameters)


class ARGModel(nn.Module):
    event_policy = 'cwr_residual'
    time_policy = 'cwr_exponential'
    continuous_time_head = 'gamma'

    def __init__(self, embedding_size=64, hidden_size=128, breakpoint_mixture_components=4):
        super().__init__()
        self.event_head = mlp(embedding_size, hidden_size, 2)
        self.action_head = mlp(4*embedding_size, hidden_size, 1)
        self.breakpoint_head = InfiniteSitesBreakpointHead(4*embedding_size, hidden_size,
                                                          breakpoint_mixture_components)
        self.time_head = CwrGammaTimeModel(4*embedding_size+4, hidden_size, 0., layers=2)
        for head in (self.event_head, self.action_head):
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

    def event_log_probs(self, batch, summary):
        hazards = summary.new_tensor(batch.allowed_hazards, dtype=torch.float64)
        available = hazards > 0
        if not available.any(-1).all():
            raise ValueError('Cannot sample an event from a terminal or dead-end state')
        return (hazards.log()+self.event_head(summary).double()).masked_fill(~available, -torch.inf).log_softmax(-1)

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

    def forward(self, env, states, batch, lineages, summary, forced_actions=None):
        event_logs = self.event_log_probs(batch, summary)
        event_indices = (Categorical(logits=event_logs).sample().tolist() if forced_actions is None else
                         [int(isinstance(a, RecombinationChoice)) for a in forced_actions])
        actions, contexts, factors = [], [], []
        for row, kind in enumerate(event_indices):
            choices = batch.actions[row][kind]
            if not choices:
                raise ValueError('Forced action has no compatible support')
            context = self.contexts(choices, lineages[row], summary[row])
            baseline = context.new_tensor([a.breakpoint_count if kind else 1 for a in choices], dtype=torch.float64).log()
            logits = baseline+self.action_head(context).squeeze(-1).double()
            logs = logits.log_softmax(-1)
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
                                       None if forced_actions is None else forced_actions[row].breakpoint)
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
