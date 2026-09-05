import torch
import torch.nn as nn

from .breakpoint import BreakpointSplitPositionCNN
from .state_encoder import StateEncodingMixin
from .action_head import ActionPolicyMixin
from .layers import mlp, transformer_encoder
from .time import TimeModel


class ARGModel(StateEncodingMixin, ActionPolicyMixin, nn.Module):
    """One-step ARG action policy.

    The model chooses an event type with a learned/CwR mixture and then scores
    candidate actions within that event. Candidates come from the environment,
    so material-mask constraints are respected.
    """

    def __init__(
        self,
        env,
        embedding_size=32,
        hidden_size=64,
        dropout=0.0,
        event_hidden_size=64,
        event_dropout=0.0,
        event_prior_weight=0.1,
        breakpoint_hidden_dim=128,
        breakpoint_dropout=0.1,
        transformer_depth=6,
        transformer_heads=4,
        transformer_mlp_ratio=2.0,
        attention_dropout=0.0,
        time_hidden_size=256,
        time_layers=3,
        time_dropout=0.0,
        breakpoint_gap_hidden_size=256,
        breakpoint_gap_layers=3,
        breakpoint_gap_dropout=0.0,
        breakpoint_use_position_features=True,
    ):
        super().__init__()
        self.env = env
        self.device = env.device
        input_size = int(env.num_blocks) * 4

        self.event_prior_weight = float(event_prior_weight)
        if not 0.0 <= self.event_prior_weight <= 1.0:
            raise ValueError(
                "event_prior_weight must be between 0 and 1 inclusive, "
                f"got {event_prior_weight}"
            )

        self.seq_embedding = nn.Linear(input_size, embedding_size)
        self.summary_token = nn.Parameter(torch.zeros(1, 1, embedding_size))
        nn.init.trunc_normal_(self.summary_token, std=0.1)
        self.encoder = transformer_encoder(
            embedding_size, transformer_depth, transformer_heads,
            transformer_mlp_ratio, dropout, attention_dropout,
        )
        self.action_scorer = mlp(embedding_size * 4, hidden_size, 1, dropout=dropout)
        self.event_scorer = mlp(
            embedding_size, event_hidden_size, len(env.event_types), dropout=event_dropout,
        )
        self.breakpoint_scorer = BreakpointSplitPositionCNN(
            hidden_dim=breakpoint_hidden_dim,
            dropout=breakpoint_dropout,
            action_context_dim=embedding_size * 4,
            gap_hidden_dim=breakpoint_gap_hidden_size,
            gap_layers=breakpoint_gap_layers,
            gap_dropout=breakpoint_gap_dropout,
            use_position_features=breakpoint_use_position_features,
        ).to(self.device)

        self.time_scorer = TimeModel(
            embedding_size,
            time_hidden_size,
            time_dropout,
            env.time_env.bins,
            layers=time_layers,
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)
        # Preview-only absolute nucleotide differences and ancestral overlap.
        # No bias: missing pair features contribute exactly zero.
        self.preview_pair_embedding = nn.Linear(int(env.num_blocks) * 5, embedding_size, bias=False)

