import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from env.time_env import validate_temperature


def _validate_parameterization(value):
    if value not in ('legacy', 'bounded_v1'):
        raise ValueError('time_parameterization must be legacy or bounded_v1')
    return value


def _mean_correction(value, parameterization):
    return 3.*torch.tanh(value/3.) if parameterization == 'bounded_v1' else value


def _shape(value, parameterization):
    return 1.+19.*value.sigmoid() if parameterization == 'bounded_v1' else value.exp()


class TimeModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, output_dim, layers=3):
        super().__init__()
        layers = int(layers)
        if layers < 0:
            raise ValueError(f"layers must be non-negative, got {layers}")

        if layers > 0:
            modules = []
            for layer in range(layers):
                modules.extend([
                    nn.Linear(input_dim if layer == 0 else hidden_dim, hidden_dim),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                ])
            self.feature = nn.Sequential(*modules)
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        else:
            self.feature = None
            self.output_layer = nn.Linear(input_dim, output_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def compute_log_time_pf(self, time_logits, time_actions):
        batch_idx = torch.arange(time_logits.shape[0], device=time_logits.device)
        log_p = F.log_softmax(time_logits, dim=1)
        return log_p[batch_idx, time_actions]

    def sample(self, time_logits, random_spec):
        temperature = validate_temperature(random_spec, time_component=True)
        return Categorical(logits=time_logits / temperature).sample()

    def forward(self, action_features):
        if self.feature is not None:
            action_features = self.feature(action_features)
        return self.output_layer(action_features)


class CwrExponentialTimeModel(TimeModel):
    """Residual exponential rate; samples are fixed actions for score gradients."""

    def __init__(self, input_dim, hidden_dim, dropout, layers=3, parameterization='legacy'):
        super().__init__(input_dim, hidden_dim, dropout, 1, layers)
        self.parameterization = _validate_parameterization(parameterization)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    @staticmethod
    def _positive(values, name):
        if not bool((torch.isfinite(values) & (values > 0)).all()):
            raise ValueError(f"continuous {name} must be finite and positive")

    def rates(self, corrections, baseline_rates):
        log_rates, rates, checks = self._rate_values(corrections, baseline_rates)
        self._check_status(checks)
        return log_rates, rates

    @staticmethod
    def _valid_positive(values):
        return (torch.isfinite(values) & (values > 0)).all()

    @staticmethod
    def _check_status(checks, extra=()):
        """One host transfer for all validation results at a sampling boundary."""
        values = torch.stack([value for value, _ in checks] + list(extra)).tolist()
        for valid, (_, message) in zip(values, checks):
            if not valid:
                raise ValueError(message)
        return values[len(checks):]

    def _rate_values(self, corrections, baseline_rates):
        raw = corrections.squeeze(-1).double()
        g = _mean_correction(raw, self.parameterization)
        baseline_rates = torch.as_tensor(baseline_rates, device=g.device, dtype=torch.float64)
        rates = baseline_rates * g.exp()
        checks = [(self._valid_positive(baseline_rates), 'continuous baseline rates must be finite and positive'),
                  (torch.isfinite(raw).all(), 'non-finite continuous rate correction'),
                  (self._valid_positive(rates), 'continuous policy rates must be finite and positive')]
        return baseline_rates.log() + g, rates, checks

    def _distribution_parameters(self, corrections, baseline_rates):
        log_rates, rates, checks = self._rate_values(corrections, baseline_rates)
        return None, log_rates, rates, checks

    def _prepare_distribution(self, corrections, baseline_rates, temperature=None):
        shape, log_rates, rates, checks = self._distribution_parameters(corrections, baseline_rates)
        behavior_rates, exponential = None, shape is None
        if temperature is not None:
            behavior_rates = rates / temperature
            checks.append((self._valid_positive(behavior_rates),
                           'continuous behavior rates must be finite and positive'))
        extra = ((shape == 1).all(),) if shape is not None and temperature is not None else ()
        flags = self._check_status(checks, extra)
        if flags:
            exponential = flags[0]
        return shape, log_rates, rates, behavior_rates, exponential

    @staticmethod
    @torch.no_grad()
    def _draw(shape, behavior_rates, exponential):
        # Parameters have already been checked together. Do not repeat the
        # distribution constructors' synchronous CUDA argument validation.
        # The shape-one flag retains the exponential stream at migration.
        if exponential:
            return torch.distributions.Exponential(behavior_rates.detach(), validate_args=False).sample()
        return torch.distributions.Gamma(shape.detach(), behavior_rates.detach(), validate_args=False).sample()

    @staticmethod
    def _log_density(shape, log_rates, rates, waits):
        if shape is None:
            return log_rates - rates * waits
        return shape*log_rates-torch.lgamma(shape)+(shape-1)*waits.log()-rates*waits

    def _score_parameters(self, shape, log_rates, rates, waits):
        waits = torch.as_tensor(waits, device=rates.device, dtype=torch.float64).detach()
        scores = self._log_density(shape, log_rates, rates, waits)
        self._check_status([(self._valid_positive(waits), 'continuous waits must be finite and positive'),
                            (torch.isfinite(scores).all(), 'non-finite continuous policy log density')])
        return scores

    def sample(self, corrections, baseline_rates, random_spec=None):
        temperature = validate_temperature(random_spec, time_component=True)
        with torch.no_grad():
            shape, _, _, behavior, exponential = self._prepare_distribution(corrections, baseline_rates, temperature)
            waits = self._draw(shape, behavior, exponential)
            self._positive(waits, "sampled waits")
        return waits

    def sample_and_log_time_pf(self, corrections, baseline_rates, random_spec=None):
        """Share differentiable parameters; sampled waits remain fixed actions."""
        temperature = validate_temperature(random_spec, time_component=True)
        shape, log_rates, rates, behavior, exponential = self._prepare_distribution(
            corrections, baseline_rates, temperature)
        waits = self._draw(shape, behavior, exponential)
        return waits, self._score_parameters(shape, log_rates, rates, waits)

    def compute_log_time_pf(self, corrections, waits, baseline_rates):
        shape, log_rates, rates, _, _ = self._prepare_distribution(corrections, baseline_rates)
        return self._score_parameters(shape, log_rates, rates, waits)


def validate_continuous_time_head(head, time_policy):
    if head not in ('exponential', 'gamma', 'gamma_mixture'):
        raise ValueError(f'Unknown continuous_time_head: {head!r}')
    if head in ('gamma', 'gamma_mixture') and time_policy != 'cwr_exponential':
        raise ValueError('The gamma head requires continuous CwR timing')
    return head


class CwrGammaTimeModel(CwrExponentialTimeModel):
    """Learn the mean and shape while retaining the exponential CwR prior.

    Given residual mean-rate g and log shape h, concentration=exp(h) and
    rate=baseline_rate*exp(g+h). Thus mean=exp(-g)/baseline_rate, independently
    of shape. At h=0 this exactly recovers the previous exponential policy.
    """

    def __init__(self, input_dim, hidden_dim, dropout, layers=3, parameterization='legacy'):
        super().__init__(input_dim, hidden_dim, dropout, layers, parameterization)
        self.shape_layer = nn.Linear(hidden_dim if layers > 0 else input_dim, 1)
        nn.init.zeros_(self.shape_layer.weight)
        nn.init.zeros_(self.shape_layer.bias)
        if self.parameterization == 'bounded_v1':
            nn.init.constant_(self.shape_layer.bias, math.log(.1/18.9))

    def forward(self, action_features):
        if self.feature is not None:
            action_features = self.feature(action_features)
        return torch.cat((self.output_layer(action_features), self.shape_layer(action_features)), dim=-1)

    def gamma_parameters(self, corrections, baseline_rates):
        shape, log_rates, rates, _, _ = self._prepare_distribution(corrections, baseline_rates)
        return shape, log_rates, rates

    def _distribution_parameters(self, corrections, baseline_rates):
        if corrections.ndim != 2 or corrections.shape[-1] != 2:
            raise ValueError('Gamma timing requires mean-rate and log-shape corrections')
        log_mean_rates, mean_rates, checks = self._rate_values(corrections[:, :1], baseline_rates)
        raw_shape = corrections[:, 1].double()
        shape = _shape(raw_shape, self.parameterization)
        log_shape = shape.log() if self.parameterization == 'bounded_v1' else raw_shape
        rates = mean_rates*shape
        checks.extend([(torch.isfinite(raw_shape).all(), 'non-finite gamma shape correction'),
                       (self._valid_positive(shape), 'continuous gamma shapes must be finite and positive'),
                       (self._valid_positive(rates), 'continuous gamma rates must be finite and positive')])
        return shape, log_mean_rates+log_shape, rates, checks


class CwrGammaMixtureTimeModel(TimeModel):
    """Conditional Gamma mixture with an exactly evaluated marginal density.

    Component selection is an internal sampling operation, not an ARG action.
    Replay and SubTB therefore score logsumexp over every component. The
    physical Hudson waiting-time prior is unchanged. Distinct initial shapes
    break mixture symmetry while keeping all component means at the prior mean.
    """

    def __init__(self, input_dim, hidden_dim, dropout, layers=3, components=4, parameterization='legacy'):
        if isinstance(components, bool) or not isinstance(components, int) or components < 1:
            raise ValueError('time mixture components must be a positive integer')
        super().__init__(input_dim, hidden_dim, dropout, 3 * components, layers)
        self.components = components
        self.parameterization = _validate_parameterization(parameterization)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        with torch.no_grad():
            initial = torch.arange(1, components + 1, dtype=self.output_layer.bias.dtype)
            if self.parameterization == 'bounded_v1':
                initial = torch.logit((initial.clamp(1.1, 19.)-1.)/19.)
            else:
                initial = initial.log()
            self.output_layer.bias[2 * components:].copy_(initial)

    def mixture_parameters(self, corrections, baseline_rates):
        if corrections.ndim != 2 or corrections.shape[-1] != 3 * self.components:
            raise ValueError('Gamma mixture requires logits, mean-rate and log-shape corrections')
        logits, mean_corrections, log_shapes = corrections.double().chunk(3, dim=-1)
        mean_corrections = _mean_correction(mean_corrections, self.parameterization)
        shapes = _shape(log_shapes, self.parameterization)
        if self.parameterization == 'bounded_v1':
            log_shapes = shapes.log()
        baseline = torch.as_tensor(baseline_rates, device=corrections.device, dtype=torch.float64)
        if baseline.shape != (len(corrections),):
            raise ValueError('one physical baseline rate is required per time distribution')
        CwrExponentialTimeModel._positive(baseline, 'baseline rates')
        log_rates = baseline.log()[:, None] + mean_corrections + log_shapes
        rates = log_rates.exp()
        CwrExponentialTimeModel._check_status([
            (torch.isfinite(corrections).all(), 'non-finite mixture corrections'),
            (CwrExponentialTimeModel._valid_positive(shapes), 'invalid mixture shapes'),
            (CwrExponentialTimeModel._valid_positive(rates), 'invalid mixture rates')])
        return logits.log_softmax(-1), shapes, log_rates, rates

    @staticmethod
    def _score(parameters, waits):
        log_weights, shapes, log_rates, rates = parameters
        waits = torch.as_tensor(waits, device=rates.device, dtype=torch.float64).detach()
        if waits.shape != (len(rates),):
            raise ValueError('one waiting time is required per mixture')
        CwrExponentialTimeModel._positive(waits, 'waits')
        component_scores = CwrExponentialTimeModel._log_density(
            shapes, log_rates, rates, waits[:, None])
        scores = torch.logsumexp(log_weights + component_scores, dim=-1)
        if not bool(torch.isfinite(scores).all()):
            raise ValueError('non-finite mixture time density')
        return scores

    @staticmethod
    @torch.no_grad()
    def _draw(parameters, temperature):
        log_weights, shapes, _, rates = parameters
        selected = torch.distributions.Categorical(logits=log_weights).sample()[:, None]
        shape = shapes.gather(1, selected).squeeze(1)
        rate = rates.gather(1, selected).squeeze(1) / temperature
        waits = torch.distributions.Gamma(shape, rate).sample()
        CwrExponentialTimeModel._positive(waits, 'sampled waits')
        return waits

    def sample(self, corrections, baseline_rates, random_spec=None):
        temperature = validate_temperature(random_spec, time_component=True)
        with torch.no_grad():
            return self._draw(self.mixture_parameters(corrections, baseline_rates), temperature)

    def sample_and_log_time_pf(self, corrections, baseline_rates, random_spec=None):
        temperature = validate_temperature(random_spec, time_component=True)
        parameters = self.mixture_parameters(corrections, baseline_rates)
        waits = self._draw(parameters, temperature)
        return waits, self._score(parameters, waits)

    def compute_log_time_pf(self, corrections, waits, baseline_rates):
        return self._score(self.mixture_parameters(corrections, baseline_rates), waits)
