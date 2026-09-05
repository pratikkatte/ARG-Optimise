import tempfile
import unittest
from pathlib import Path

import torch

from model.layers import transformer_encoder
from training.config import TrainConfig


class TrainingConfigTests(unittest.TestCase):
    def load_text(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(text, encoding="utf-8")
            return TrainConfig.load(path)

    def test_minimal_yaml_uses_defaults(self):
        config = self.load_text("""
data:
  dataset_path: input.fa
  output_path: output
""")
        self.assertEqual(config.training.epochs, 10)
        self.assertEqual(config.model.embedding_size, 32)
        self.assertEqual(config.runtime.device, "auto")
        self.assertFalse(config.logging.wandb)
        self.assertEqual(config.training.convergence_eval_episodes, 0)
        self.assertEqual(config.training.convergence_min_ess_fraction, 0.25)
        self.assertEqual(config.environment.reward_offset, 0.0)

    def test_unknown_setting_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown model settings: mystery"):
            self.load_text("""
data: {dataset_path: input.fa, output_path: output}
model: {mystery: 1}
""")

    def test_invalid_head_dimension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            self.load_text("""
data: {dataset_path: input.fa, output_path: output}
model: {embedding_size: 7, transformer_heads: 2}
""")

    def test_invalid_dropout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dropout settings"):
            self.load_text("""
data: {dataset_path: input.fa, output_path: output}
model: {breakpoint_dropout: 1.0}
""")

    def test_negative_rate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rates cannot be negative"):
            self.load_text("""
data: {dataset_path: input.fa, output_path: output}
environment: {recombination_rate: -0.1}
""")

    def test_invalid_convergence_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ess_fraction"):
            self.load_text("""
data: {dataset_path: input.fa, output_path: output}
training: {convergence_min_ess_fraction: 1.1}
""")

    def test_certification_panel_cannot_be_too_small(self):
        with self.assertRaisesRegex(ValueError, "0 or at least 256"):
            self.load_text("""
data: {dataset_path: input.fa, output_path: output}
training: {convergence_eval_episodes: 64}
""")

    def test_cpu_device_resolves(self):
        config = self.load_text("""
data: {dataset_path: input.fa, output_path: output}
runtime: {device: cpu}
""")
        self.assertEqual(config.device, torch.device("cpu"))


class TransformerTests(unittest.TestCase):
    def test_padding_does_not_change_unmasked_tokens(self):
        torch.manual_seed(3)
        encoder = transformer_encoder(8, depth=1, heads=2).eval()
        inputs = torch.randn(1, 3, 8)
        changed = inputs.clone()
        changed[:, 2] = 1000
        mask = torch.tensor([[False, False, True]])
        with torch.no_grad():
            expected = encoder(inputs, src_key_padding_mask=mask)
            actual = encoder(changed, src_key_padding_mask=mask)
        torch.testing.assert_close(actual[:, :2], expected[:, :2])


if __name__ == "__main__":
    unittest.main()
