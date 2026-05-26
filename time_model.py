from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class TimeModel(nn.Sequential):
    def __init__(self, input_dim, hidden_dim, dropout, output_dim):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

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
        return super().forward(action_features)
