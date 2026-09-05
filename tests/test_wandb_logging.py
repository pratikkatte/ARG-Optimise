import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from training.config import TrainConfig
from training.loop import _start_logger, _wandb_metrics


def make_config(wandb_enabled=True):
    return TrainConfig(
        data=SimpleNamespace(dataset_path="input.fa", output_path="output", bp_per_blocks=1),
        runtime=SimpleNamespace(device="cpu", seed=7, verbose=False),
        training=SimpleNamespace(
            epochs=10, batch_size=2, init_z_sample_count=1,
            grad_accum_steps=1, eval_episodes=2, eval_every=5,
        ),
        optimizer=SimpleNamespace(policy_lr=1e-3, log_z_lr=2e-3, grad_clip=10.0),
        environment=SimpleNamespace(),
        model=SimpleNamespace(),
        logging=SimpleNamespace(
            wandb=wandb_enabled, project="arg-optimise", entity=None, run_name=None,
        ),
    )


class WandbLoggingTests(unittest.TestCase):
    def test_metric_payload_is_filtered_and_derives_rmse(self):
        info = {
            "loss": 25.0,
            "log_z": 12.0,
            "grad_norm": 8.0,
            "param_norm": 99.0,
            "epoch": 3,
            "eval_tb_mse": 9.0,
            "eval_residual_mean": -1.0,
            "eval_residual_std": 2.0,
            "eval_trajectory_length_mean": 7.0,
            "eval_coalescence_count_mean": 4.0,
            "eval_recombination_count_mean": 3.0,
            "eval_initial_learned_coalescence_prob": 0.4,
            "eval_initial_learned_recombination_prob": 0.6,
            "eval_initial_mixed_coalescence_prob": 0.5,
            "eval_initial_mixed_recombination_prob": 0.5,
            "eval_initial_cwr_coalescence_prob": 0.7,
            "eval_initial_cwr_recombination_prob": 0.3,
            "eval_log_reward_mean": 100.0,
            "convergence_importance_ess_fraction": 0.5,
            "convergence_residual_rmse": 0.75,
            "grad_norm_time": 6.0,
        }

        metrics = _wandb_metrics(info, make_config())

        self.assertEqual(metrics["train/tb_rmse"], 5.0)
        self.assertEqual(metrics["eval/tb_rmse"], 3.0)
        self.assertEqual(metrics["optimizer/policy_lr"], 1e-3)
        self.assertIn("eval/initial_event/mixed_recombination_prob", metrics)
        self.assertEqual(metrics["convergence/importance_ess_fraction"], 0.5)
        self.assertEqual(metrics["optimizer/grad_norm/time"], 6.0)
        self.assertNotIn("param_norm", metrics)
        self.assertNotIn("epoch", metrics)
        self.assertNotIn("eval_log_reward_mean", metrics)

    def test_non_evaluation_payload_has_no_eval_metrics(self):
        metrics = _wandb_metrics(
            {"loss": 2.0, "log_z": 3.0, "grad_norm": 4.0}, make_config(),
        )

        self.assertTrue(math.isclose(metrics["train/tb_rmse"], math.sqrt(2.0)))
        self.assertFalse(any(name.startswith("eval/") for name in metrics))

    def test_logger_initializes_project_and_defines_summaries(self):
        run = Mock()
        wandb_module = Mock()
        wandb_module.init.return_value = run
        env = SimpleNamespace(time_metadata={"time_bins": 4})

        with patch("training.loop.wandb", wandb_module):
            actual = _start_logger(make_config(), env)

        self.assertIs(actual, run)
        init_kwargs = wandb_module.init.call_args.kwargs
        self.assertEqual(init_kwargs["project"], "arg-optimise")
        self.assertEqual(init_kwargs["config"]["time_bins"], 4)
        run.define_metric.assert_any_call("train/loss", summary="min")
        run.define_metric.assert_any_call("train/log_z", summary="last")
        run.define_metric.assert_any_call("eval/tb_mse", summary="min")

    def test_disabled_logger_does_not_initialize_wandb(self):
        wandb_module = Mock()
        with patch("training.loop.wandb", wandb_module):
            actual = _start_logger(
                make_config(wandb_enabled=False), SimpleNamespace(time_metadata={}),
            )

        self.assertIsNone(actual)
        wandb_module.init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
