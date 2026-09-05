from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

import torch

from arg_environment import SimpleARGEnvironment
from gflownet import TBGFlowNetGenerator
from gflownet.checkpoint import load_checkpoint
from infer import run_inference
from model.breakpoint import BreakpointSplitPositionCNN
from rollout_worker_arg import RolloutWorker
from training.loop import train


SMALL_MODEL = {
    "embedding_size": 8,
    "hidden_size": 8,
    "event_hidden_size": 8,
    "transformer_depth": 1,
    "transformer_heads": 2,
    "time_hidden_size": 8,
    "time_layers": 1,
    "breakpoint_hidden_dim": 8,
    "breakpoint_gap_hidden_size": 8,
    "breakpoint_gap_layers": 1,
}


def make_generator(env):
    return TBGFlowNetGenerator(
        env,
        init_z_sample_count=1,
        initialize_z_from_prior=False,
        verbose=False,
        model_kwargs=SMALL_MODEL,
    )


class RolloutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.env = SimpleARGEnvironment(
            sequences=["AAAA", "ACAA", "AAGA"], recombination_rate=0.0,
        )

    def test_rollout_uses_padded_tensors_and_backpropagates(self):
        generator = make_generator(self.env)
        outputs, trajectories = RolloutWorker(self.env).rollout(
            generator, episodes=2, return_states=True,
        )

        self.assertEqual(outputs["log_paths_pf"].shape, (2, 2))
        self.assertEqual(outputs["log_paths_pb"].shape, (2, 2))
        self.assertEqual([len(trajectory) for trajectory in trajectories], [2, 2])
        self.assertTrue(all(state.is_done for state in outputs["states"]))
        generator.get_loss_from_rollout_outputs(outputs).backward()
        self.assertTrue(any(p.grad is not None for p in generator.arg_model.parameters()))

    def test_padding_accepts_unequal_and_empty_rows(self):
        worker = RolloutWorker(self.env)
        padded = worker._pad([torch.tensor([1.0, 2.0]), torch.empty(0)])
        torch.testing.assert_close(padded, torch.tensor([[1.0, 2.0], [0.0, 0.0]]))


class BackwardActionTests(unittest.TestCase):
    def setUp(self):
        self.env = SimpleARGEnvironment(sequences=["AAAA", "ACAA", "AAGA"])
        self.generator = make_generator(self.env)
        self.state = self.env.get_initial_state()

    def test_latest_coalescence_has_one_inverse(self):
        coal, _ = self.env.enumerate_actions(self.state)
        child = self.env.apply_action(
            self.state, replace(coal[0], time_action=0), log_prior=0.0,
        )
        actions = self.generator._enumerate_inverse_arg_actions(child)
        self.assertEqual([action["event_type"] for action in actions], ["coal"])

    def test_paired_recombination_has_one_inverse(self):
        _, recombinations = self.env.enumerate_actions(self.state)
        child = self.env.apply_action(
            self.state,
            replace(recombinations[0], breakpoint=2, time_action=0),
            log_prior=0.0,
        )
        actions = self.generator._enumerate_inverse_arg_actions(child)
        self.assertEqual([action["event_type"] for action in actions], ["recomb"])


class CheckpointTests(unittest.TestCase):
    def test_path_and_mapping_load_restore_parameters_and_metadata(self):
        env = SimpleARGEnvironment(sequences=["AAAA", "ACAA", "AAGA"])
        generator = make_generator(env)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            generator.save(path, metadata={"epoch": 3})
            checkpoint = load_checkpoint(path, map_location="cpu")

            restored = make_generator(
                SimpleARGEnvironment(sequences=["AAAA", "ACAA", "AAGA"])
            )
            metadata = restored.load(checkpoint, load_optimizer=True)
            restored_from_path = make_generator(
                SimpleARGEnvironment(sequences=["AAAA", "ACAA", "AAGA"])
            )
            self.assertEqual(
                restored_from_path.load(path, load_optimizer=False), {"epoch": 3}
            )

        self.assertEqual(metadata, {"epoch": 3})
        torch.testing.assert_close(restored.compute_log_Z(), generator.compute_log_Z())
        torch.testing.assert_close(
            restored_from_path.compute_log_Z(), generator.compute_log_Z()
        )
        self.assertEqual(set(restored.state_dict()), set(generator.state_dict()))

    def test_minimal_training_and_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta = root / "input.fa"
            fasta.write_text(">a\nAAAA\n>b\nACAA\n>c\nAAGA\n", encoding="utf-8")
            output = root / "training"
            config = root / "config.yaml"
            config.write_text(
                f"""
data:
  dataset_path: {fasta}
  output_path: {output}
runtime: {{device: cpu, seed: 5, verbose: false}}
training: {{epochs: 1, batch_size: 1, init_z_sample_count: 1, eval_episodes: 0}}
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

            history = train(config)
            checkpoint = output / "checkpoints" / "best.pt"
            manifest = run_inference(
                checkpoint, root / "inference", num_args=1, device="cpu",
            )

        self.assertEqual(len(history), 1)
        self.assertEqual(manifest["num_args"], 1)
        self.assertEqual(len(manifest["outputs"]), 1)


class BreakpointTests(unittest.TestCase):
    def test_breakpoint_indices_are_tensorized_and_clamped(self):
        model = BreakpointSplitPositionCNN(hidden_dim=8, dilations=[1])
        indices = model._breakpoint_logit_indices(4, [0, 2, 99], "cpu")
        torch.testing.assert_close(indices, torch.tensor([0, 1, 2]))


if __name__ == "__main__":
    unittest.main()
