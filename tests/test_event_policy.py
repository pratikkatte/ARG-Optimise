"""Tests for learned coalescence/recombination event selection."""
import unittest

import torch

from arg_environment import SimpleARGEnvironment
from model import ARGModel
from gflownet import TBGFlowNetGenerator


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


class EventPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.env = SimpleARGEnvironment(sequences=["AAAA", "ACAA", "AAGA"])

    def _model(self, weight=0.1):
        return ARGModel(self.env, event_prior_weight=weight, **SMALL_MODEL)

    def test_probability_level_mixture_is_exact(self):
        model = self._model(weight=0.1)
        for parameter in model.event_scorer.parameters():
            torch.nn.init.zeros_(parameter)
        summary = torch.zeros(1, SMALL_MODEL["embedding_size"])
        valid = torch.tensor([[True, True]])
        prior = torch.tensor([[0.8, 0.2]])

        learned, mixed = model.compute_event_probabilities(summary, valid, prior)

        torch.testing.assert_close(learned, torch.tensor([[0.5, 0.5]]))
        torch.testing.assert_close(mixed, torch.tensor([[0.53, 0.47]]))
        torch.testing.assert_close(mixed.sum(dim=1), torch.ones(1))

    def test_unavailable_event_is_never_assigned_probability(self):
        model = self._model(weight=0.1)
        summary = torch.randn(2, SMALL_MODEL["embedding_size"])
        valid = torch.tensor([[True, False], [False, True]])
        prior = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        learned, mixed = model.compute_event_probabilities(summary, valid, prior)

        torch.testing.assert_close(learned, prior)
        torch.testing.assert_close(mixed, prior)

    def test_temperature_only_changes_learned_component_before_mixing(self):
        model = self._model(weight=0.1)
        for parameter in model.event_scorer.parameters():
            torch.nn.init.zeros_(parameter)
        model.event_scorer[-1].bias.data.copy_(torch.tensor([2.0, 0.0]))
        summary = torch.zeros(1, SMALL_MODEL["embedding_size"])
        valid = torch.tensor([[True, True]])
        prior = torch.tensor([[0.2, 0.8]])

        learned, mixed = model.compute_event_probabilities(
            summary,
            valid,
            prior,
            random_spec={"T": 2.0},
        )

        expected_learned = torch.softmax(torch.tensor([[1.0, 0.0]]), dim=1)
        torch.testing.assert_close(learned, expected_learned)
        torch.testing.assert_close(mixed, 0.9 * expected_learned + 0.1 * prior)

    def test_weight_boundaries_recover_learned_and_prior_policies(self):
        summary = torch.zeros(1, SMALL_MODEL["embedding_size"])
        valid = torch.tensor([[True, True]])
        prior = torch.tensor([[0.9, 0.1]])
        outputs = []
        for weight in (0.0, 1.0):
            model = self._model(weight=weight)
            for parameter in model.event_scorer.parameters():
                torch.nn.init.zeros_(parameter)
            outputs.append(model.compute_event_probabilities(summary, valid, prior))

        learned_only, mixed_at_zero = outputs[0]
        _, mixed_at_one = outputs[1]
        torch.testing.assert_close(mixed_at_zero, learned_only)
        torch.testing.assert_close(mixed_at_one, prior)

    def test_mixed_log_probability_trains_event_head(self):
        model = self._model(weight=0.1)
        summary = torch.randn(1, SMALL_MODEL["embedding_size"])
        valid = torch.tensor([[True, True]])
        prior = torch.tensor([[0.7, 0.3]])

        _, mixed = model.compute_event_probabilities(summary, valid, prior)
        (-torch.log(mixed[0, 0])).backward()

        gradient = model.event_scorer[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_rollout_input_no_longer_preselects_event(self):
        state = self.env.get_initial_state()
        inputs = self.env.prepare_state_rollout_inputs([state])

        self.assertNotIn("event", inputs)
        self.assertNotIn("input_actions", inputs)
        coal, recomb = inputs["candidate_actions"][0]
        self.assertTrue(coal)
        self.assertTrue(recomb)

    def test_generator_reports_learned_cwr_and_mixed_diagnostics(self):
        generator = TBGFlowNetGenerator(
            self.env,
            init_z_sample_count=1,
            initialize_z_from_prior=False,
            verbose=False,
            model_kwargs={**SMALL_MODEL, "event_prior_weight": 0.1},
        )
        probabilities = generator.compute_event_probabilities(
            [self.env.get_initial_state()]
        )

        self.assertEqual(set(probabilities), {"learned", "cwr", "mixed"})
        expected = 0.9 * probabilities["learned"] + 0.1 * probabilities["cwr"]
        torch.testing.assert_close(probabilities["mixed"], expected)

    def test_forward_path_probability_backpropagates_to_event_head(self):
        generator = TBGFlowNetGenerator(
            self.env,
            init_z_sample_count=1,
            initialize_z_from_prior=False,
            verbose=False,
            model_kwargs={**SMALL_MODEL, "event_prior_weight": 0.1},
        )
        inputs = self.env.prepare_state_rollout_inputs([self.env.get_initial_state()])

        log_pf, _ = generator(inputs)
        (-log_pf.sum()).backward()

        gradient = generator.arg_model.event_scorer[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_zero_prior_support_masks_recombination(self):
        env = SimpleARGEnvironment(
            sequences=["AAAA", "ACAA", "AAGA"],
            recombination_rate=0.0,
        )
        generator = TBGFlowNetGenerator(
            env,
            init_z_sample_count=1,
            initialize_z_from_prior=False,
            verbose=False,
            model_kwargs={**SMALL_MODEL, "event_prior_weight": 0.1},
        )

        probabilities = generator.compute_event_probabilities([env.get_initial_state()])

        expected = torch.tensor([[1.0, 0.0]])
        torch.testing.assert_close(probabilities["learned"], expected)
        torch.testing.assert_close(probabilities["mixed"], expected)

    def test_invalid_prior_weight_is_rejected(self):
        for weight in (-0.01, 1.01):
            with self.subTest(weight=weight):
                with self.assertRaisesRegex(ValueError, "event_prior_weight"):
                    self._model(weight=weight)


if __name__ == "__main__":
    unittest.main()
