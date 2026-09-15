"""Trajectory-balance loss, also used for SubTB reporting."""
import torch


def tb_diagnostics(log_pf, log_pb, log_rewards, log_z):
    """TB residual metrics, evaluated only when TB reporting is requested."""
    residuals = log_z.detach().to(log_pf) + log_pf - (log_rewards + log_pb)
    mse = float(residuals.pow(2).mean().detach().cpu().item())
    return {
        "tb_loss": mse,
        "tb_mse": mse,
        "residual_rmse": float(residuals.pow(2).mean().sqrt().detach().cpu()),
        "residual_mean": float(residuals.mean().detach().cpu().item()),
        "residual_std": float(residuals.std(unbiased=False).detach().cpu().item()),
    }


class TBMixin:
    """TB calculations on the public generator; owns no separate state."""

    def get_tb_loss_from_rollout_outputs(self, rollout_outputs):
        log_paths_pf = rollout_outputs['log_paths_pf']
        log_paths_pb = rollout_outputs['log_paths_pb']
        log_rewards = torch.as_tensor(
            rollout_outputs['log_rewards'],
            dtype=log_paths_pf.dtype,
            device=log_paths_pf.device,
        )

        log_pf = log_paths_pf.sum(-1)
        log_pb = log_paths_pb.sum(-1)

        
        log_z = self.compute_log_Z(None).reshape(-1).to(log_paths_pf)

        forward_value = log_z + log_pf
        backward_value = log_rewards + log_pb

        loss = self.loss_fn(forward_value, backward_value)

        return loss
