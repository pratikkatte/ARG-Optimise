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
from .time_model import CwrGammaTimeModel, CwrExponentialTimeModel, CwrGammaMixtureTimeModel


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

    def forward_batch(self, spans, contexts, length, breakpoints=None):
        """T=1 physical-link distributions; all results stay on the device.

        spans is [rows, 2] with inclusive integer endpoints. The scalar API
        above also serves the existing, exactly normalized tempered path.
        """
        if spans.ndim != 2 or spans.shape != (len(contexts), 2) or spans.dtype != torch.long:
            raise ValueError('Breakpoint spans must be an integer [rows, 2] tensor')
        a, z = spans.unbind(-1)
        lower, upper = spans.double().unbind(-1)
        n = z-a+1
        features = torch.stack((lower/length, upper/length,
                                n.double()/max(length-1, 1)), -1).to(contexts.dtype)
        raw = self.parameters_head(torch.cat((contexts, features), -1)).double()
        weights, centers, scales = raw.chunk(3, -1)
        weights = weights.log_softmax(-1)
        locations = lower[:, None]-.5+n[:, None]*centers.sigmoid()
        scales = .1+n[:, None]*scales.sigmoid()
        # Check the whole batch before feeding any invalid value to a sampler.
        valid = ((a >= 1) & (a <= z) & (z < length)).all()
        valid = valid & torch.isfinite(raw).all() & torch.isfinite(locations).all() & torch.isfinite(scales).all()
        if not valid:
            raise ValueError('Invalid breakpoint spans or nonfinite mixture parameters')
        if breakpoints is None:
            with torch.no_grad():
                component = torch.multinomial(weights.detach().exp(), 1)
                loc = locations.detach().gather(1, component).squeeze(-1)
                scale = scales.detach().gather(1, component).squeeze(-1)
                low = torch.sigmoid((lower-.5-loc)/scale)
                high = torch.sigmoid((upper+.5-loc)/scale)
                probability = low+torch.rand_like(low)*(high-low)
                eps = torch.finfo(torch.float64).eps
                draw = loc+scale*torch.logit(probability.clamp(eps, 1-eps))
                breakpoints = torch.minimum(torch.maximum(torch.floor(draw+.5).long(), a), z)
        elif breakpoints.dtype != torch.long or breakpoints.shape != a.shape:
            raise ValueError('Breakpoints must be an integer tensor matching the batch')
        elif not ((breakpoints >= a) & (breakpoints <= z)).all():
            raise ValueError('Breakpoint is outside the physical recombination span')
        mass = SparseMixtureBreakpointPolicy._log_interval_mass
        gaps = breakpoints.double()[:, None]
        logs = torch.logsumexp(weights+mass(gaps-.5, gaps+.5, locations, scales)
                               -mass(lower[:, None]-.5, upper[:, None]+.5, locations, scales), -1)
        return breakpoints, torch.where(a == z, logs*0., logs)


class ARGModel(nn.Module):
    event_policy = 'cwr_residual'
    time_policy = 'cwr_exponential'
    continuous_time_head = 'gamma'

    def __init__(self, embedding_size=64, hidden_size=128, breakpoint_mixture_components=4,
                 breakpoint_mixture_hidden_dim=None, breakpoint_mixture_layers=1,
                 breakpoint_gap_hidden_size=64, breakpoint_gap_layers=0,
                 continuous_time_head='gamma', time_hidden_dim=None, time_layers=2,
                 time_mixture_components=4, time_parameterization='legacy'):
        super().__init__()
        self.event_head = mlp(embedding_size, hidden_size, 2)
        self.action_head = mlp(4*embedding_size, hidden_size, 1)
        self.breakpoint_head = InfiniteSitesBreakpointHead(4*embedding_size, breakpoint_mixture_hidden_dim or hidden_size,
                    breakpoint_mixture_components, breakpoint_mixture_layers,
                    breakpoint_gap_hidden_size, breakpoint_gap_layers)
        self.continuous_time_head = continuous_time_head
        heads = {'gamma': CwrGammaTimeModel, 'exponential': CwrExponentialTimeModel,
                 'gamma_mixture': CwrGammaMixtureTimeModel}
        if continuous_time_head not in heads:
            raise ValueError('Unknown continuous time head')
        extra = {'components': time_mixture_components} if continuous_time_head == 'gamma_mixture' else {}
        self.time_head = heads[continuous_time_head](4*embedding_size+4,
            time_hidden_dim or hidden_size, 0., layers=time_layers,
            parameterization=time_parameterization, **extra)
        for head in (self.event_head, self.action_head):
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

    def event_log_probs(self, batch, summary, temperature=1.0):
        if any(any(not math.isfinite(h) or h < 0 for h in row) or not any(h > 0 for h in row)
               for row in batch.allowed_hazards):
            raise ValueError('Cannot sample an event from a terminal or dead-end state with invalid hazards')
        hazards = summary.new_tensor(batch.allowed_hazards, dtype=torch.float64)
        available = hazards > 0
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
        if temperature != 1.:
            return self._forward_tempered(env, states, batch, lineages, summary, forced_actions, temperature)
        if forced_actions is not None and len(forced_actions) != len(states):
            raise ValueError('Forced actions must match the state batch')
        event_logs = self.event_log_probs(batch, summary)
        kinds = (self._sample_logs(event_logs).tolist() if forced_actions is None else
                 [int(isinstance(a, RecombinationChoice)) for a in forced_actions])
        choices_by_row = [batch.actions[row][kind] for row, kind in enumerate(kinds)]
        if any(not choices for choices in choices_by_row):
            raise ValueError('Forced action has no compatible support')
        selected_cpu = []
        if forced_actions is not None:
            for choices, action, kind in zip(choices_by_row, forced_actions, kinds):
                canonical = replace(action, delta_t=None, time_action=None,
                                    **({'breakpoint': None} if kind else {}))
                try:
                    selected_cpu.append(choices.index(canonical))
                except ValueError as exc:
                    raise ValueError('Forced action is not a compatible physical candidate') from exc
                if kind:
                    a, z = SparseMixtureBreakpointPolicy.valid_span(action, env.sequence_length)
                    if not isinstance(action.breakpoint, int) or not a <= action.breakpoint <= z:
                        raise ValueError('Breakpoint is outside the physical recombination span')

        # Two context batches, with no embedding padding and no GPU work in
        # the CPU metadata loop. Each record stores row/i/j/column/a/z/weight.
        records, groups = [], []
        width = max(map(len, choices_by_row))
        lookup = [[0]*width for _ in states]
        for kind in (0, 1):
            start = len(records)
            for row, choices in enumerate(choices_by_row):
                if kinds[row] != kind:
                    continue
                for column, action in enumerate(choices):
                    a, z = SparseMixtureBreakpointPolicy.valid_span(action, env.sequence_length) if kind else (0, 0)
                    lookup[row][column] = len(records)
                    records.append((row, action.active_lineage_i, action.active_lineage_i if kind else action.active_lineage_j,
                                    column, a, z, action.breakpoint_count if kind else 1))
            groups.append((start, len(records)))
        metadata = torch.tensor(records, device=lineages.device, dtype=torch.long)
        packed = []
        for kind, (start, end) in enumerate(groups):
            if start == end:
                continue
            row, i, j = metadata[start:end, :3].unbind(-1)
            first = lineages[row, i]
            if kind:
                context = torch.cat((first, torch.zeros_like(first), torch.zeros_like(first), summary[row]), -1)
            else:
                second = lineages[row, j]
                context = torch.cat((first+second, (first-second).abs(), first*second, summary[row]), -1)
            packed.append(context)
        contexts = torch.cat(packed)
        hidden = self.action_head[:-1](contexts)
        # The scalar bias cancels in the softmax; preserve checkpoint layout
        # without training that mathematically zero-gradient parameter.
        residuals = torch.nn.functional.linear(hidden, self.action_head[-1].weight).squeeze(-1).double()
        logits = residuals.new_full((len(states), width), -torch.inf)
        logits = logits.index_put((metadata[:, 0], metadata[:, 3]), residuals+metadata[:, 6].double().log())
        logs = logits.log_softmax(-1)
        selected = (self._sample_logs(logs) if forced_actions is None else
                    torch.tensor(selected_cpu, device=lineages.device, dtype=torch.long))
        selected_groups = torch.tensor(lookup, device=lineages.device).gather(1, selected[:, None]).squeeze(-1)
        contexts = contexts[selected_groups]
        selected_spans = metadata[selected_groups, 4:6]
        recomb_rows = [row for row, kind in enumerate(kinds) if kind]
        gaps = torch.zeros(len(states), dtype=torch.long, device=lineages.device)
        breakpoint_logs = logs.new_zeros(len(states))
        if recomb_rows:
            rows = torch.tensor(recomb_rows, device=lineages.device)
            forced_gaps = (None if forced_actions is None else
                           torch.tensor([forced_actions[row].breakpoint for row in recomb_rows], device=lineages.device))
            drawn, scores = self.breakpoint_head.forward_batch(selected_spans[rows], contexts[rows],
                                                              env.sequence_length, forced_gaps)
            gaps = gaps.index_copy(0, rows, drawn)
            breakpoint_logs = breakpoint_logs.index_copy(0, rows, scores)
        rates = [total_event_rate(r) for r in batch.physical_rates]
        timing = contexts.new_tensor([[math.log1p(s.current_time), math.log(rate), float(kind), rate]
                                      for s, rate, kind in zip(states, rates, kinds)], dtype=torch.float64)
        time_features = torch.cat((contexts, timing[:, :3].to(contexts.dtype),
                                   (gaps.double()/env.sequence_length).to(contexts.dtype)[:, None]), -1)
        corrections = self.time_head(time_features)
        if forced_actions is None:
            waits, time_logs = self.time_head.sample_and_log_time_pf(corrections, timing[:, 3])
        else:
            waits = timing.new_tensor([a.delta_t for a in forced_actions])
            time_logs = self.time_head.compute_log_time_pf(corrections, waits, timing[:, 3])
        event_index = torch.tensor(kinds, device=lineages.device)[:, None]
        factors = torch.stack((event_logs.gather(1, event_index).squeeze(-1),
                               logs.gather(1, selected[:, None]).squeeze(-1), breakpoint_logs, time_logs), -1)
        if not torch.isfinite(factors).all():
            raise FloatingPointError('Nonfinite policy factor')
        if forced_actions is None:
            # Keep integers separate from float64 waits (coordinates need not
            # fit in float64's exact integer range). No per-row device reads.
            decisions = torch.stack((selected, gaps), -1).tolist()
            actions = [replace(choices[index], delta_t=dt, **({'breakpoint': bp} if kind else {}))
                       for choices, kind, (index, bp), dt in zip(choices_by_row, kinds, decisions, waits.tolist())]
        else:
            actions = [replace(a, time_action=None, delta_t=float(a.delta_t)) for a in forced_actions]
        return factors.sum(-1), actions, factors

    @staticmethod
    @torch.no_grad()
    def _sample_logs(logs):
        probabilities = logs.detach().exp()
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError('Nonfinite policy distribution')
        return torch.multinomial(probabilities, 1).squeeze(-1)

    def _forward_tempered(self, env, states, batch, lineages, summary, forced_actions, temperature):
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
