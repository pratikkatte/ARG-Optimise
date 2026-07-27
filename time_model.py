import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


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

    def _masked_logits(self, time_logits, action_mask=None):
        if action_mask is None:
            return time_logits
        action_mask = torch.as_tensor(
            action_mask,
            dtype=torch.bool,
            device=time_logits.device,
        )
        if action_mask.shape != time_logits.shape:
            raise ValueError(
                "time action mask must match time logits shape: "
                f"mask={tuple(action_mask.shape)} "
                f"logits={tuple(time_logits.shape)}"
            )
        if not bool(action_mask.any(dim=1).all().detach().cpu().item()):
            raise ValueError("every generated transition needs a valid time bin")
        return time_logits.masked_fill(~action_mask, float("-inf"))

    def compute_log_time_pf(
        self,
        time_logits,
        time_actions,
        action_mask=None,
    ):
        time_logits = self._masked_logits(time_logits, action_mask)
        batch_idx = torch.arange(time_logits.shape[0], device=time_logits.device)
        log_p = F.log_softmax(time_logits, dim=1)
        return log_p[batch_idx, time_actions]

    def sample(self, time_logits, random_spec, action_mask=None):
        time_logits = self._masked_logits(time_logits, action_mask)
        if random_spec is None:
            return Categorical(logits=time_logits).sample()
        temperature = random_spec["T"]
        return Categorical(logits=time_logits / temperature).sample()

    def forward(self, action_features):
        if self.feature is not None:
            action_features = self.feature(action_features)
        return self.output_layer(action_features)
