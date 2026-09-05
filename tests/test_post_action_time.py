"""Check topology-only time conditioning independently of a training run."""
from dataclasses import replace
import unittest

import torch

from arg_environment import CoalescenceChoice, MaterialSegments, SimpleARGEnvironment
from gflownet import TBGFlowNetGenerator


class PostActionTimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.env = SimpleARGEnvironment(sequences=['AAAA', 'ACAA', 'AAGA'])
        self.state = self.env.get_initial_state()

    def test_preview_changes_topology_without_mutating_source(self):
        coal, recomb = self.env.enumerate_actions(self.state)
        original = self.state.clone(copy_partials=True)
        for action, expected_count in [(coal[0], 2), (replace(recomb[0], breakpoint=2), 4)]:
            with self.subTest(action=action):
                preview = self.env.preview_action_for_time_model(self.state, action)
                self.assertEqual(len(preview.active_lineages), expected_count)
                self.assertEqual(preview.current_time, self.state.current_time)
                self.assertEqual(preview.accumulated_log_prior, self.state.accumulated_log_prior)
                self.assertIsNone(preview.log_reward)
                for old, actual in zip(original.active_lineages, self.state.active_lineages):
                    self.assertEqual(old.node_id, actual.node_id)
                    self.assertEqual(old.parents, actual.parents)
                    self.assertEqual(old.material_segments, actual.material_segments)
                    torch.testing.assert_close(old.partials, actual.partials)
                timed = self.env.apply_action(self.state, replace(action, time_action=0), log_prior=0.0)
                self.assertGreater(timed.current_time, self.state.current_time)
                self.assertEqual(
                    [x.material_segments for x in preview.active_lineages],
                    [x.material_segments for x in timed.active_lineages],
                )

    def test_recombination_requires_breakpoint_and_preserves_material(self):
        _, recomb = self.env.enumerate_actions(self.state)
        with self.assertRaisesRegex(ValueError, 'breakpoint'):
            self.env.preview_action_for_time_model(self.state, recomb[0])
        preview = self.env.preview_action_for_time_model(self.state, replace(recomb[0], breakpoint=2))
        left, right = preview.active_lineages[-2:]
        self.assertEqual(left.material_segments.intersection_count(right.material_segments), 0)
        self.assertEqual(left.material_segments.union(right.material_segments), self.state.active_lineages[0].material_segments)
        torch.testing.assert_close(left.partials + right.partials, self.state.active_lineages[0].partials)

    def test_time_receives_preview_summary_and_backpropagates(self):
        generator = TBGFlowNetGenerator(
            self.env, init_z_sample_count=1, initialize_z_from_prior=False, verbose=False,
            model_kwargs={'embedding_size': 8, 'transformer_depth': 1,
                          'transformer_heads': 2, 'time_hidden_size': 16},
        )
        generator.eval()
        coal, recomb = self.env.enumerate_actions(self.state)
        captured = []
        hook = generator.time_model.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0])
        )
        try:
            log_pf, actions = generator({
                'states': [self.state, self.state],
                'candidate_actions': [([coal[0]], []), ([], [recomb[0]])],
                'random_spec': None,
            })
        finally:
            hook.remove()
        previews = [self.env.preview_action_for_time_model(self.state, a) for a in actions]
        _, expected, _, _ = generator._encode_states(previews)
        torch.testing.assert_close(captured[0], expected)
        self.assertTrue(bool(torch.isfinite(log_pf).all()))
        self.assertTrue(all(a.time_action is not None for a in actions))
        self.assertIsNotNone(actions[1].breakpoint)
        # Backpropagate only the time likelihood to prove the second encoder pass is connected.
        time_logits = generator.time_model(captured[0])
        chosen = torch.tensor([a.time_action for a in actions])
        loss = -generator.time_model.compute_log_time_pf(time_logits, chosen).sum()
        loss.backward()
        for parameter in [generator.time_model.output_layer.weight,
                          generator.arg_model.seq_embedding.weight,
                          generator.arg_model.preview_pair_embedding.weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_terminal_coalescence_preview_can_be_encoded(self):
        env = SimpleARGEnvironment(sequences=['AAAA', 'ACAA'])
        state = env.get_initial_state()
        preview = env.preview_action_for_time_model(state, CoalescenceChoice(0, 1))
        self.assertEqual(len(preview.active_lineages), 1)
        parent = preview.active_lineages[0]
        torch.testing.assert_close(parent.partials[1], torch.tensor([0.5, 0.5, 0., 0.]))
        torch.testing.assert_close(parent.preview_pair_features[1], torch.tensor([1., 1., 0., 0., 1.]))
        generator = TBGFlowNetGenerator(
            env, init_z_sample_count=1, initialize_z_from_prior=False, verbose=False,
            model_kwargs={'embedding_size': 8, 'transformer_depth': 1, 'transformer_heads': 2},
        )
        _, summary, _, _ = generator._encode_states([preview])
        self.assertTrue(bool(torch.isfinite(generator.time_model(summary)).all()))

    def test_fragmented_material_masks_evidence_and_disagreement(self):
        left, right = self.state.active_lineages[:2]
        left.material_segments = MaterialSegments(((0, 2),))
        right.material_segments = MaterialSegments(((1, 3),))
        action = CoalescenceChoice(0, 1)
        preview = self.env.preview_action_for_time_model(self.state, action)
        parent = preview.active_lineages[-1]
        torch.testing.assert_close(parent.partials, torch.tensor([
            [1., 0., 0., 0.], [0.5, 0.5, 0., 0.],
            [1., 0., 0., 0.], [0., 0., 0., 0.],
        ]))
        torch.testing.assert_close(parent.preview_pair_features, torch.tensor([
            [0., 0., 0., 0., 0.], [1., 1., 0., 0., 1.],
            [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.],
        ]))
        reverse = self.env.preview_action_for_time_model(self.state, CoalescenceChoice(1, 0))
        torch.testing.assert_close(parent.partials, reverse.active_lineages[-1].partials)
        torch.testing.assert_close(parent.preview_pair_features, reverse.active_lineages[-1].preview_pair_features)

    def test_overlapping_spans_with_disjoint_material_cannot_coalesce(self):
        self.state.active_lineages[0].material_segments = MaterialSegments(((0, 1), (3, 4)))
        self.state.active_lineages[1].material_segments = MaterialSegments(((1, 3),))
        action = CoalescenceChoice(0, 1)
        self.assertFalse(action.is_valid_for(self.state.active_lineages))
        self.assertNotIn(action, CoalescenceChoice.enumerate_from_active_lineages(self.state.active_lineages))
        with self.assertRaisesRegex(ValueError, 'Invalid coalescence'):
            self.env.preview_action_for_time_model(self.state, action)

    def test_disagreement_reaches_time_logits_when_mean_evidence_is_identical(self):
        ambiguous = self.env.get_initial_state().clone(copy_partials=True)
        for child in ambiguous.active_lineages[:2]:
            child.partials[1] = torch.tensor([0.5, 0.5, 0., 0.])
        action = CoalescenceChoice(0, 1)
        conflict_preview = self.env.preview_action_for_time_model(self.state, action)
        ambiguous_preview = self.env.preview_action_for_time_model(ambiguous, action)
        torch.testing.assert_close(conflict_preview.active_lineages[-1].partials,
                                   ambiguous_preview.active_lineages[-1].partials)
        generator = TBGFlowNetGenerator(
            self.env, init_z_sample_count=1, initialize_z_from_prior=False, verbose=False,
            model_kwargs={'embedding_size': 8, 'transformer_depth': 1, 'transformer_heads': 2},
        )
        generator.eval()
        _, summary, _, _ = generator._encode_states([conflict_preview, ambiguous_preview])
        logits = generator.time_model(summary)
        self.assertGreater(float((logits[0] - logits[1]).detach().abs().max()), 0.)

    def test_timed_transition_uses_evolved_likelihood_not_preview_evidence(self):
        left, right = self.state.active_lineages[:2]
        left.time, right.time = 0.2, 0.4
        self.state.current_time = 0.4
        preview = self.env.preview_action_for_time_model(self.state, CoalescenceChoice(0, 1))
        parents = []
        for time_action in (0, 20):
            timed = self.env.apply_action(self.state, CoalescenceChoice(0, 1, time_action), log_prior=0.)
            parent = timed.active_lineages[-1]
            expected = torch.ones_like(parent.partials)
            for child in (left, right):
                evolved = self.env.evolution_model.transition_partials(
                    child.partials, parent.time - child.time,
                )
                expected *= self.env.evolution_model.normalize_partials(evolved)
            expected /= expected.sum(-1, keepdim=True)
            torch.testing.assert_close(parent.partials, expected)
            self.assertIsNone(parent.preview_pair_features)
            self.assertTrue(bool((parent.partials[1] > 0).all()))
            self.assertFalse(torch.allclose(parent.partials, preview.active_lineages[-1].partials))
            parents.append(parent)
        self.assertFalse(torch.allclose(parents[0].partials, parents[1].partials))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
