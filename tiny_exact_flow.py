"""A fully enumerable two-terminal GFlowNet correctness problem."""

from __future__ import annotations

import math

import torch


class TinyExactFlow(torch.nn.Module):
    """root -> {left, right} -> {terminal_1, terminal_3}.

    Terminal rewards are 1 and 3, every backward transition is deterministic,
    and the exact partition is 4.  The reward-proportional root policy is
    therefore (1/4, 3/4).
    """

    log_rewards = (0.0, math.log(3.0))

    def __init__(self, device="cpu"):
        super().__init__()
        self.root_logits = torch.nn.Parameter(torch.zeros(2, device=device))
        self.log_flows = torch.nn.Parameter(torch.zeros(3, device=device))

    def trajectory_residuals(self):
        log_pf = torch.log_softmax(self.root_logits, dim=0)
        rewards = self.log_flows.new_tensor(self.log_rewards)
        # One-step root->branch, branch->terminal, and full TB residuals.
        internal = self.log_flows[0] + log_pf - self.log_flows[1:]
        terminal = self.log_flows[1:] - rewards
        full = self.log_flows[0] + log_pf - rewards
        return {"internal": internal, "terminal": terminal, "full": full}

    def loss(self, terminal_loss_weight=1.0, residual_scale=1.0):
        residuals = self.trajectory_residuals()
        scale = float(residual_scale)
        return (
            residuals["internal"].div(scale).square().mean()
            + residuals["full"].div(scale).square().mean()
            + float(terminal_loss_weight)
            * residuals["terminal"].div(scale).square().mean()
        )

    def terminal_probabilities(self):
        return torch.softmax(self.root_logits, dim=0)

    def set_exact_solution(self):
        with torch.no_grad():
            self.root_logits.copy_(
                self.root_logits.new_tensor([0.0, math.log(3.0)])
            )
            self.log_flows.copy_(
                self.log_flows.new_tensor(
                    [math.log(4.0), 0.0, math.log(3.0)]
                )
            )
        return self


def train_tiny_exact_flow(
    *,
    device="cpu",
    steps=1500,
    learning_rate=0.03,
    terminal_loss_weight=1.0,
    residual_scale=1.0,
    seed=7,
):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = TinyExactFlow(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    history = []
    for step in range(int(steps)):
        optimizer.zero_grad()
        loss = model.loss(
            terminal_loss_weight=terminal_loss_weight,
            residual_scale=residual_scale,
        )
        loss.backward()
        optimizer.step()
        if step in {0, int(steps) - 1}:
            history.append(float(loss.detach().cpu().item()))
    return model, history

