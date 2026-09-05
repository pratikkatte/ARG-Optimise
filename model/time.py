import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .layers import mlp


class TimeModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, output_dim, layers=3):
        super().__init__()
        self.network = mlp(input_dim, hidden_dim, output_dim, layers=int(layers), dropout=dropout)

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
        return self.network(action_features)

    @property
    def output_layer(self):
        """Compatibility alias for callers inspecting the final projection."""
        return self.network[-1]

