import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from time_env import validate_temperature


class TimeModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, output_dim, layers=3):
        super().__init__()
        layers = int(layers)
        if layers < 0:
            raise ValueError(f"layers must be non-negative, got {layers}")

        if layers > 0:
            modules = [
                nn.Linear(input_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            ]
            for _ in range(layers - 1):
                modules.extend([
                    nn.Linear(hidden_dim, hidden_dim),
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
        if random_spec is None:
            return Categorical(logits=time_logits).sample()
        temperature = random_spec["T"]
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
        temperature = validate_temperature(random_spec)
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
