import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from env.time_env import validate_temperature


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

    def __init__(self, input_dim, hidden_dim, dropout, layers=3):
        super().__init__(input_dim, hidden_dim, dropout, 1, layers)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    @staticmethod
    def _positive(values, name):
        if not bool((torch.isfinite(values) & (values > 0)).all()):
            raise ValueError(f"continuous {name} must be finite and positive")

    def rates(self, corrections, baseline_rates):
        g = corrections.squeeze(-1).double()
        baseline_rates = torch.as_tensor(baseline_rates, device=g.device, dtype=torch.float64)
        self._positive(baseline_rates, "baseline rates")
        if not bool(torch.isfinite(g).all()):
            raise ValueError("non-finite continuous rate correction")
        rates = baseline_rates * g.exp()
        self._positive(rates, "policy rates")
        return baseline_rates.log() + g, rates

    def sample(self, corrections, baseline_rates, random_spec=None):
        temperature = validate_temperature(random_spec, time_component=True)
        with torch.no_grad():
            _, rates = self.rates(corrections, baseline_rates)
            behavior_rates = rates / temperature
            self._positive(behavior_rates, "behavior rates")
            waits = torch.distributions.Exponential(behavior_rates).sample()
            self._positive(waits, "sampled waits")
        return waits

    def compute_log_time_pf(self, corrections, waits, baseline_rates):
        log_rates, rates = self.rates(corrections, baseline_rates)
        waits = torch.as_tensor(waits, device=rates.device, dtype=torch.float64).detach()
        self._positive(waits, "waits")
        scores = log_rates - rates * waits
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("non-finite continuous policy log density")
        return scores


def validate_continuous_time_head(head, time_policy):
    if head not in ('exponential', 'gamma'):
        raise ValueError(f'Unknown continuous_time_head: {head!r}')
    if head == 'gamma' and time_policy != 'cwr_exponential':
        raise ValueError('The gamma head requires continuous CwR timing')
    return head


class CwrGammaTimeModel(CwrExponentialTimeModel):
    """Learn the mean and shape while retaining the exponential CwR prior.

    Given residual mean-rate g and log shape h, concentration=exp(h) and
    rate=baseline_rate*exp(g+h). Thus mean=exp(-g)/baseline_rate, independently
    of shape. At h=0 this exactly recovers the previous exponential policy.
    """

    def __init__(self, input_dim, hidden_dim, dropout, layers=3):
        super().__init__(input_dim, hidden_dim, dropout, layers)
        self.shape_layer = nn.Linear(hidden_dim if layers > 0 else input_dim, 1)
        nn.init.zeros_(self.shape_layer.weight)
        nn.init.zeros_(self.shape_layer.bias)

    def forward(self, action_features):
        if self.feature is not None:
            action_features = self.feature(action_features)
        return torch.cat((self.output_layer(action_features), self.shape_layer(action_features)), dim=-1)

    def gamma_parameters(self, corrections, baseline_rates):
        if corrections.ndim != 2 or corrections.shape[-1] != 2:
            raise ValueError('Gamma timing requires mean-rate and log-shape corrections')
        log_mean_rates, mean_rates = super().rates(corrections[:, :1], baseline_rates)
        log_shape = corrections[:, 1].double()
        shape = log_shape.exp()
        self._positive(shape, 'gamma shapes')
        rates = mean_rates*shape
        self._positive(rates, 'gamma rates')
        return shape, log_mean_rates+log_shape, rates

    def sample(self, corrections, baseline_rates, random_spec=None):
        temperature = validate_temperature(random_spec, time_component=True)
        with torch.no_grad():
            shape, _, rates = self.gamma_parameters(corrections, baseline_rates)
            behavior_rates = rates/temperature
            self._positive(behavior_rates, 'behavior rates')
            # Preserve the exact old sampling stream at checkpoint migration.
            # General gamma exploration scales the mean by T; scores remain
            # those of the untempered policy, as for the exponential head.
            if bool((shape == 1).all()):
                waits = torch.distributions.Exponential(behavior_rates).sample()
            else:
                waits = torch.distributions.Gamma(shape, behavior_rates).sample()
            self._positive(waits, 'sampled waits')
            return waits

    def compute_log_time_pf(self, corrections, waits, baseline_rates):
        shape, log_rates, rates = self.gamma_parameters(corrections, baseline_rates)
        waits = torch.as_tensor(waits, device=rates.device, dtype=torch.float64).detach()
        self._positive(waits, 'waits')
        scores = shape*log_rates-torch.lgamma(shape)+(shape-1)*waits.log()-rates*waits
        if not bool(torch.isfinite(scores).all()):
            raise ValueError('non-finite continuous gamma policy log density')
        return scores
