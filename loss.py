import torch
import torch.nn as nn


class TrajectoryBalanceLoss(nn.Module):
    """
    Trajectory Balance loss for the left-to-right ARGweaver GFlowNet.

    This computes the mean squared residual of
    `log_Z_theta + sum_log_P_F - log_R`, where `log_Z_theta` is the learnable
    log partition function, `sum_log_P_F` is the per-trajectory sum of forward
    log-probabilities, and `log_R` is the per-trajectory terminal log-reward.
    """

    def forward(
        self,
        log_Z: torch.Tensor,
        sum_log_P_F: torch.Tensor,
        log_R: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the batched Trajectory Balance objective.

        Args:
            log_Z: Scalar tensor for `log Z_theta`.
            sum_log_P_F: Tensor of shape `(batch_size,)` for `sum log P_F`.
            log_R: Tensor of shape `(batch_size,)` for terminal `log R`.

        Returns:
            The mean squared TB error across the batch.
        """
        if log_Z.numel() != 1:
            raise ValueError(
                f"log_Z must contain exactly one element, got shape {tuple(log_Z.shape)}."
            )
        if sum_log_P_F.dim() != 1:
            raise ValueError(
                "sum_log_P_F must be a 1D tensor of shape (batch_size,), "
                f"got shape {tuple(sum_log_P_F.shape)}."
            )
        if log_R.dim() != 1:
            raise ValueError(
                f"log_R must be a 1D tensor of shape (batch_size,), got shape {tuple(log_R.shape)}."
            )
        if sum_log_P_F.shape != log_R.shape:
            raise ValueError(
                "sum_log_P_F and log_R must have identical shapes, "
                f"got {tuple(sum_log_P_F.shape)} and {tuple(log_R.shape)}."
            )

        tb_error = log_Z.reshape(()) + sum_log_P_F - log_R
        return tb_error.square().mean()
