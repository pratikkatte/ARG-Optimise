"""Direct SubTB updates preserve the former chunked gradient calculation."""
import copy
import math
import weakref
from unittest.mock import patch

import pytest
import torch

from gfn.rollout import RolloutWorker
from training.checkpoints import seed_everything
from training.configuration import config_notes, resolve_config
from training.trainer import Trainer, TrajectoryMixConfig
from validation.tests.test_fresh_score_reuse import make_model, assert_nested_close


@pytest.mark.parametrize('head', ['gamma', 'exponential', 'gamma_mixture'])
@pytest.mark.parametrize('accum', [1, 3])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_direct_update_matches_chunked_reference(head, accum, device):
    torch.set_num_threads(1)
    seed_everything(27)
    reference = make_model(head, device)
    direct = make_model(head, device)
    direct.load_state_dict(reference.state_dict())
    worker = RolloutWorker(reference.env)
    batch = 7
    with torch.no_grad():
        _, paths = worker.rollout(reference, batch)

    # The old method: score without activations, differentiate the full loss
    # with respect to scores, then recompute in chunks for parameter gradients.
    reference.opt.zero_grad(set_to_none=True)
    micro_size = math.ceil(batch / accum)
    reference_loss = 0.
    for start in range(0, batch, micro_size):
        subset = paths[start:start+micro_size]
        with torch.no_grad():
            scores, _ = worker.replay(reference, subset)
        scores['log_paths_pf'].requires_grad_()
        scores['state_flows'].requires_grad_()
        loss = reference.get_loss_from_rollout_outputs(scores) * (len(subset) / batch)
        weights = torch.autograd.grad(loss, (scores['log_paths_pf'], scores['state_flows']))
        worker.backward_scores(reference, subset, *weights, chunk_steps=3)
        reference_loss += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(reference.parameters(), reference.grad_clip, error_if_nonfinite=True)
    reference.opt.step()
    reference.scheduler.step()

    class FixedWorker(RolloutWorker):
        cursor = 0

        def rollout(self, generator, episodes, **kwargs):
            selected = paths[self.cursor:self.cursor+episodes]
            self.cursor += episodes
            return self.replay(generator, selected, collect_flows=kwargs['collect_flows'],
                               return_states=kwargs['return_states'])

    trainer = Trainer(direct, FixedWorker(direct.env), TrajectoryMixConfig(replay_fraction=0.))
    with patch.object(trainer.worker, 'backward_scores', side_effect=AssertionError('Recomputation')):
        metrics = trainer.train_epoch(batch_size=batch, grad_accum_steps=accum)
    assert metrics['loss'] == pytest.approx(reference_loss, rel=2e-6, abs=2e-6)
    assert_nested_close(reference.state_dict(), direct.state_dict())
    assert_nested_close(reference.opt.state_dict(), direct.opt.state_dict())
    assert reference.scheduler.state_dict() == direct.scheduler.state_dict()
    for a, b in zip(reference.parameters(), direct.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-5)


def test_sampling_graph_is_backpropagated_and_released_per_microbatch():
    torch.set_num_threads(1)
    seed_everything(27)
    model = make_model()
    saved_refs = []

    class SavedTensor:
        def __init__(self, tensor):
            # Keep storage without creating a reference cycle back into its graph.
            self.tensor = tensor.detach()

    def pack(tensor):
        saved = SavedTensor(tensor)
        saved_refs.append(weakref.ref(saved))
        return saved

    class StreamingWorker(RolloutWorker):
        def __init__(self, env):
            super().__init__(env)
            self.backward_count = self.rollout_count = 0
            self.score_refs = []

        def rollout(self, *args, **kwargs):
            assert self.backward_count == self.rollout_count
            assert all(ref() is None for ref in self.score_refs)
            assert all(ref() is None for ref in saved_refs)
            outputs, paths = super().rollout(*args, **kwargs)
            self.rollout_count += 1
            def record_backward(gradient):
                self.backward_count += 1
            outputs['log_paths_pf'].register_hook(record_backward)
            self.score_refs.append(weakref.ref(outputs['log_paths_pf']))
            return outputs, paths

    worker = StreamingWorker(model.env)
    trainer = Trainer(model, worker, TrajectoryMixConfig(replay_fraction=0.))
    with patch.object(worker, 'replay', side_effect=AssertionError('Fresh rescoring')), \
         patch.object(worker, 'backward_scores', side_effect=AssertionError('Recomputation')), \
         patch.object(model.opt, 'step', wraps=model.opt.step) as step, \
         torch.autograd.graph.saved_tensors_hooks(pack, lambda saved: saved.tensor):
        trainer.train_epoch(batch_size=7, grad_accum_steps=3)
    assert worker.backward_count == worker.rollout_count == 3
    assert all(ref() is None for ref in worker.score_refs)
    assert saved_refs and all(ref() is None for ref in saved_refs)
    step.assert_called_once()


def test_legacy_chunk_size_is_inactive_on_resume():
    torch.set_num_threads(1)
    seed_everything(27)
    model = make_model()
    original = Trainer(model, RolloutWorker(model.env), chunk_steps=16)
    state = copy.deepcopy(original.state_dict())
    resumed = Trainer(model, RolloutWorker(model.env), chunk_steps=640000)
    resumed.load_state_dict(state)
    assert resumed.completed_updates == original.completed_updates
    assert resumed.buffer.state_dict() == original.buffer.state_dict()
    assert 'Inactive' in config_notes(resolve_config({}))['chunk_steps']
