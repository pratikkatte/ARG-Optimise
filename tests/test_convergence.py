import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from arg_environment.time import DEFAULT_TIME_BIN_SCHEME
from infer import run_inference, validate_metadata
from gflownet.checkpoint import load_checkpoint
from training.config import MODEL_VERSION
from training.evaluation import compute_importance_diagnostics
from training.loop import _convergence_status, train


def convergence_options(**overrides):
    values = {
        "convergence_min_ess_fraction": 0.25,
        "convergence_max_abs_residual_mean": 1.0,
        "convergence_max_residual_rmse": 2.0,
        "convergence_required_passes": 3,
        "convergence_eval_episodes": 256,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def checkpoint_metadata(**overrides):
    values = {
        "sequences": ["AAAA", "ACAA"],
        "num_sequences": 2,
        "sequence_length": 4,
        "num_blocks": 4,
        "rho": 0.0,
        "time_bin_scheme": DEFAULT_TIME_BIN_SCHEME,
        "time_bins": 4,
        "time_delta_bin_width": 0.001,
        "seed": 7,
        "init_z_sample_count": 1,
        "model_version": MODEL_VERSION,
    }
    values.update(overrides)
    return values


def passed_convergence():
    return {
        "version": 1,
        "evaluated": True,
        "passed": True,
        "eval_episodes": 256,
        "consecutive_passes": 3,
        "metrics": {
            "importance_ess_fraction": 0.5,
            "residual_mean": 0.1,
            "residual_rmse": 0.5,
        },
    }


class ImportanceDiagnosticsTests(unittest.TestCase):
    def test_equal_weights_have_full_ess_and_are_shift_invariant(self):
        expected = compute_importance_diagnostics(
            [10.0, 10.0, 10.0, 10.0], [2.0] * 4, [0.0] * 4, log_z=8.0,
        )
        shifted = compute_importance_diagnostics(
            [1010.0] * 4, [1002.0] * 4, [0.0] * 4, log_z=8.0,
        )

        self.assertAlmostEqual(expected["importance_ess"], 4.0)
        self.assertAlmostEqual(expected["importance_ess_fraction"], 1.0)
        self.assertEqual(expected, shifted)
        self.assertEqual(expected["residual_rmse"], 0.0)

    def test_dominant_weight_is_stable_over_large_log_range(self):
        metrics = compute_importance_diagnostics(
            [1000.0, 0.0, -1000.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        )

        self.assertTrue(math.isclose(metrics["importance_ess"], 1.0))
        self.assertTrue(math.isclose(metrics["importance_max_weight"], 1.0))
        self.assertEqual(metrics["importance_log_weight_range"], 2000.0)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            compute_importance_diagnostics([], [], [])
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_importance_diagnostics([float("nan")], [0.0], [0.0])


class ConvergenceGateTests(unittest.TestCase):
    def test_requires_three_consecutive_passing_panels(self):
        info = {
            "convergence_importance_ess_fraction": 0.5,
            "convergence_residual_mean": 0.1,
            "convergence_residual_rmse": 0.5,
        }
        consecutive = 0
        statuses = []
        for _ in range(3):
            status, consecutive = _convergence_status(
                info, convergence_options(), consecutive,
            )
            statuses.append(status)

        self.assertEqual([status["passed"] for status in statuses], [False, False, True])
        self.assertEqual(consecutive, 3)

    def test_failed_panel_resets_consecutive_count(self):
        info = {
            "convergence_importance_ess_fraction": 0.01,
            "convergence_residual_mean": 0.1,
            "convergence_residual_rmse": 0.5,
        }
        status, consecutive = _convergence_status(info, convergence_options(), 2)
        self.assertFalse(status["current_panel_passed"])
        self.assertEqual(consecutive, 0)

    def test_inference_requires_passed_metadata(self):
        with self.assertRaisesRegex(ValueError, "has not passed"):
            validate_metadata(checkpoint_metadata())

        validate_metadata(checkpoint_metadata(), allow_unconverged=True)
        validate_metadata(checkpoint_metadata(convergence=passed_convergence()))

    def test_v7_requires_diagnostic_override(self):
        metadata = checkpoint_metadata(model_version="pytorch-transformer-yaml-v7")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_metadata(metadata)
        validate_metadata(metadata, allow_unconverged=True)


class CheckpointLifecycleTests(unittest.TestCase):
    def test_only_passing_evaluation_creates_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta = root / "input.fa"
            fasta.write_text(">a\nAAAA\n>b\nACAA\n", encoding="utf-8")
            output = root / "training"
            config = root / "config.yaml"
            config.write_text(
                f"""
data: {{dataset_path: {fasta}, output_path: {output}}}
runtime: {{device: cpu, seed: 5, verbose: false}}
training:
  epochs: 3
  batch_size: 1
  init_z_sample_count: 1
  eval_episodes: 0
  convergence_eval_episodes: 256
  convergence_eval_every: 1
  convergence_min_ess_fraction: 0.25
  convergence_max_abs_residual_mean: 1.0
  convergence_max_residual_rmse: 2.0
  convergence_required_passes: 3
  stop_on_convergence: true
environment: {{recombination_rate: 0.0, time_bins: 4}}
model:
  embedding_size: 8
  hidden_size: 8
  event_hidden_size: 8
  transformer_depth: 1
  transformer_heads: 2
  time_hidden_size: 8
  time_layers: 1
  breakpoint_hidden_dim: 8
  breakpoint_gap_hidden_size: 8
  breakpoint_gap_layers: 1
""",
                encoding="utf-8",
            )

            convergence_metrics = {
                "convergence_importance_ess": 200.0,
                "convergence_importance_ess_fraction": 0.5,
                "convergence_importance_max_weight": 0.01,
                "convergence_importance_log_weight_range": 1.0,
                "convergence_tb_mse": 0.25,
                "convergence_residual_mean": 0.1,
                "convergence_residual_std": 0.4,
                "convergence_residual_rmse": 0.5,
            }
            with patch(
                "training.loop.evaluate_generator",
                return_value=convergence_metrics,
            ):
                history = train(config)
            metadata = load_checkpoint(
                output / "checkpoints" / "best.pt", map_location="cpu",
            )["metadata"]
            manifest = run_inference(
                output / "checkpoints" / "best.pt",
                root / "inference",
                num_args=1,
                device="cpu",
            )

        self.assertEqual(len(history), 3)
        self.assertTrue(metadata["convergence"]["passed"])
        self.assertEqual(metadata["checkpoint_kind"], "best_converged")
        self.assertEqual(
            metadata["checkpoint_selection_metric"], "convergence_residual_rmse",
        )
        self.assertFalse(manifest["allow_unconverged"])
        self.assertIn("importance_ess_fraction", manifest["inference_diagnostics"])


if __name__ == "__main__":
    unittest.main()
