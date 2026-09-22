"""Real-process evaluation, nonblocking training, queue recovery and failure isolation."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from train import train
from training.async_evaluation import AsyncEvaluator, publish_checkpoint
from training.checkpoints import load_checkpoint
from training.configuration import resolve_config, parse_train_args
from training.trainer import Trainer
from gfn.rollout import RolloutWorker
from validation.tests.test_infinite_sites_configuration import fixture_dataset
from validation.tests.test_infinite_sites_neural import model


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def saved_checkpoint(path, step):
    generator = model()
    trainer = Trainer(generator, RolloutWorker(generator.env))
    trainer.completed_updates = step
    generator.save(path, trainer=trainer, metadata=dict(step=step))
    return path


def evaluation_config(**overrides):
    return resolve_config(dict(eval_async=True, eval_episodes=2, eval_batch_size=2,
                               eval_async_device='cpu', **overrides))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='An allocated CUDA GPU is required'))])
def test_training_finishes_while_evaluator_is_held_and_rng_is_unchanged(tmp_path, device):
    dataset, observed = fixture_dataset(tmp_path)
    output = tmp_path/'async'
    gate = tmp_path/'held_worker.py'
    # Deterministic concurrency witness: the evaluator cannot start until training
    # has completed all updates. A per-evaluation join would fail this test.
    gate.write_text('''import json,os,sys,time
from pathlib import Path
output=Path(sys.argv[1]); deadline=time.monotonic()+30
while not (output/'run_status.json').exists():
    if time.monotonic()>deadline: raise RuntimeError('Training waited for evaluation')
    time.sleep(.02)
status=json.loads((output/'run_status.json').read_text())
assert status['completed_updates']==3 and status['status']=='completed'
assert not list((output/'async_eval/results').glob('*.json'))
(output/'training_completed_before_evaluation.json').write_text(json.dumps(status))
os.execv(sys.executable,[sys.executable,'-u','-m','training.async_evaluation',
    '--output',str(output),'--parent-pid',str(os.getppid())])
''')
    class HeldEvaluator(AsyncEvaluator):
        def __init__(self, output, config):
            super().__init__(output, config, [sys.executable, str(gate), str(output)])

    options = dict(dataset_path=str(dataset), epochs=3, batch_size=2,
        init_z_sample_count=2, replay_fraction=0., verbose=False, device=device,
        model_kwargs=dict(embedding_size=16, hidden_size=32, transformer_depth=1, transformer_heads=2))
    with patch('train.load_snp_dataset', return_value=observed):
        baseline, baseline_trainer = train(output_path=str(tmp_path/'baseline'), **options)
    events, definitions, exits = [], [], []
    run = SimpleNamespace(id='async-test', summary={},
        log=lambda values,step=None:events.append((step,dict(values))),
        define_metric=lambda *args,**kwargs:definitions.append((args,kwargs)),
        finish=lambda exit_code:exits.append(exit_code))
    with patch('train.load_snp_dataset', return_value=observed), \
         patch('train.AsyncEvaluator', HeldEvaluator), \
         patch('train.evaluate_generator', side_effect=AssertionError('Evaluation ran in training process')), \
         patch.dict('sys.modules', wandb=SimpleNamespace(init=lambda **kwargs:run)):
        generator, trainer = train(output_path=str(output), eval_async=True,
            eval_async_device=device, eval_async_shutdown_seconds=90., eval_every=1,
            eval_episodes=2, eval_batch_size=2, checkpoint_every=3,
            terminal_eval_repeats=2, terminal_eval_repeat_every=2, wandb=True, **options)
    assert trainer.completed_updates == baseline_trainer.completed_updates == 3
    assert (output/'training_completed_before_evaluation.json').exists()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, generator.state_dict()[key], rtol=0, atol=0)
    assert ((('eval_*',), {'step_metric':'eval_checkpoint_step'})) in definitions
    assert [v['eval_checkpoint_step'] for _,v in events if 'eval_checkpoint_step' in v] == [1,2,3]
    assert all(step is None for step,_ in events)  # W&B manages its history index.
    assert [v['train_update'] for _,v in events if 'train_update' in v] == [1,2,3]
    assert exits == [0]
    rows = [json.loads(line) for line in (output/'evaluation.jsonl').read_text().splitlines()]
    assert [(r['step'],r['repeat']) for r in rows] == [(1,0),(2,0),(2,1),(3,0)]
    assert all(r['eval_independent_checked']==2 for r in rows)
    assert load_checkpoint(output/'checkpoints/latest.pt')['trainer']['completed_updates']==3
    best = json.loads((output/'checkpoints/best_eval.json').read_text())
    data = load_checkpoint(output/'checkpoints/best_eval.pt')
    assert data['trainer']['completed_updates'] == best['step']
    assert data['metadata']['best_eval_score'] == best['score']
    assert json.loads((output/'async_eval/status.json').read_text())['pending']==0


def test_failed_checkpoint_does_not_block_later_evaluation(tmp_path):
    good = saved_checkpoint(tmp_path/'checkpoint_0002.pt',2)
    worker = AsyncEvaluator(tmp_path/'run',evaluation_config())
    worker.submit(tmp_path/'missing.pt',1,1)
    worker.submit(good,2,1)
    results = list(worker.finish(90))
    assert [(r['step'],r['status']) for r in results] == [(1,'failed'),(2,'completed')]
    status = json.loads((tmp_path/'run/async_eval/status.json').read_text())
    assert status['failed']==1 and status['completed']==1


def test_shutdown_timeout_preserves_queue_and_resume_does_not_duplicate(tmp_path):
    checkpoint = saved_checkpoint(tmp_path/'checkpoint_0001.pt',1)
    output = tmp_path/'run'
    config = evaluation_config()
    held = AsyncEvaluator(output,config,[sys.executable,'-c','import time; time.sleep(120)'])
    held.submit(checkpoint,1,1)
    assert list(held.finish(0)) == []
    assert json.loads((output/'async_eval/status.json').read_text())['pending']==['step_000001.json']
    resumed = AsyncEvaluator(output,config)
    resumed.submit(checkpoint,1,1)
    assert [r['step'] for r in resumed.finish(90)] == [1]
    again = AsyncEvaluator(output,config)
    again.submit(checkpoint,1,1)
    assert list(again.finish(90)) == []
    assert len((output/'evaluation.jsonl').read_text().splitlines())==1


def test_latest_publication_preserves_immutable_checkpoint(tmp_path):
    source = tmp_path/'checkpoint_0050.pt'; source.write_bytes(b'old checkpoint')
    latest = tmp_path/'latest.pt'
    publish_checkpoint(source,latest)
    replacement = tmp_path/'checkpoint_0100.pt'; replacement.write_bytes(b'new checkpoint')
    publish_checkpoint(replacement,latest)
    assert source.read_bytes()==b'old checkpoint' and latest.read_bytes()==b'new checkpoint'


def test_async_off_cadence_wall_stop_resumes_exactly(tmp_path):
    dataset, observed = fixture_dataset(tmp_path)
    options = dict(dataset_path=str(dataset), epochs=3, batch_size=2,
        init_z_sample_count=2, replay_fraction=0., verbose=False,
        model_kwargs=dict(embedding_size=16, hidden_size=32, transformer_depth=1, transformer_heads=2))
    output = tmp_path/'run'
    with patch('train.load_snp_dataset', return_value=observed):
        baseline, _ = train(output_path=str(tmp_path/'baseline'), **options)
        _, stopped = train(output_path=str(output), eval_async=True, eval_async_device='cpu',
            eval_episodes=2, eval_batch_size=2, eval_every=50, checkpoint_every=50,
            eval_async_shutdown_seconds=90, max_wall_seconds=1e-12, **options)
        assert stopped.completed_updates == 1
        assert load_checkpoint(output/'checkpoints/latest.pt')['trainer']['completed_updates']==1
        assert (output/'checkpoints/checkpoint_0001.pt').exists()
        resumed, trainer = train(resume_checkpoint=str(output/'checkpoints/latest.pt'),
                                 max_wall_seconds=0.)
    assert trainer.completed_updates == 3
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    rows = [json.loads(line) for line in (output/'evaluation.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == [1,3]
    assert load_checkpoint(output/'checkpoints/latest.pt')['trainer']['completed_updates']==3


def test_async_cli_and_validation():
    options = parse_train_args(['--eval-async','--eval-async-device','cuda',
                                '--eval-every','50','--eval-episodes','512'])
    assert resolve_config(options)['eval_async_device']=='cuda'
    with pytest.raises(ValueError,match='eval_episodes'):
        resolve_config(dict(eval_async=True,eval_episodes=0))
