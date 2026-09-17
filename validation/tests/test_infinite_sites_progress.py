"""Progress must expose long stages without changing the scientific run."""
import copy
import json
from unittest.mock import patch

import pytest
import torch

from gfn.rollout import RolloutWorker, RolloutFailure
from train import train
from training.checkpoints import seed_everything, rng_state
from training.configuration import resolve_config, parse_train_args
from training.evaluation import evaluate_generator
from training.progress import ProgressReporter
from training.trainer import Trainer, TrajectoryMixConfig
from validation.tests.test_infinite_sites_neural import environment, model
from validation.tests.test_infinite_sites_configuration import fixture_dataset


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_progress_is_throttled_and_flushed_without_advancing_history(tmp_path, capsys):
    clock = [0.]
    reporter = ProgressReporter(tmp_path/'progress.jsonl', clock=lambda:clock[0], every_seconds=15.)
    summary = {}
    reporter.summary = summary
    reporter.begin('flow_initialization', initialized=0, initialization_total=200)
    clock[0] = 5.
    reporter.update(initialized=1, events_max=50)
    assert len(records(reporter.path)) == 1
    clock[0] = 15.
    assert reporter.due()
    reporter.update(initialized=2, events_max=80)
    assert records(reporter.path)[-1]['initialized'] == 2
    assert summary['progress']['elapsed_seconds'] == 15.
    assert '[progress 15.0s] flow_initialization' in capsys.readouterr().out
    reporter.verbose = False
    reporter.begin('train_backward', step=1)
    assert capsys.readouterr().out == ''
    assert records(reporter.path)[-1]['phase'] == 'train_backward'
    assert 'initialized' not in records(reporter.path)[-1]
    with pytest.raises(ValueError, match='progress_every_seconds'):
        resolve_config(dict(progress_every_seconds=0))
    assert resolve_config(parse_train_args(['--progress-every-seconds','2']))['progress_every_seconds'] == 2.


def test_progress_preserves_initialization_sampling_gradients_and_replay(tmp_path):
    torch.set_num_threads(1)
    seed_everything(7)
    first = model(environment(recombination=.001, length=4))
    second = model(environment(recombination=.001, length=4))
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    # Force the event callback to execute on every event, covering the paths
    # which ordinarily execute only every 15 seconds.
    clock = iter(range(100000))
    second.progress_reporter = ProgressReporter(tmp_path/'progress.jsonl', verbose=False,
                                                every_seconds=.1, clock=lambda:float(next(clock)))
    mix = TrajectoryMixConfig(exploration_fraction=.25, replay_fraction=.25, replay_min_size=2)
    results = []
    for generator in (first, second):
        seed_everything(27); generator.env.rng.seed(27)
        generator.initialize_flow_center()
        trainer = Trainer(generator, RolloutWorker(generator.env), mix, seed=27, chunk_steps=1)
        updates = [trainer.train_epoch(batch_size=4, grad_accum_steps=2) for _ in range(2)]
        metrics, details = evaluate_generator(generator, 4, seed=100027)
        results.append((updates, trainer.buffer.state_dict(), metrics, details, rng_state(generator.env)))
    for a,b in zip(results[0][:4], results[1][:4]):
        assert a == b
    for name,value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], atol=0, rtol=0)
    assert torch.equal(results[0][4]['torch'], results[1][4]['torch'])
    assert results[0][4]['environment'] == results[1][4]['environment']
    log = records(second.progress_reporter.path)
    assert log[0]['phase'] == 'flow_initialization' and log[0]['initialized'] == 0
    assert any(row.get('events_max',0)>0 for row in log)
    assert {'train_sampling','train_exploration','train_scoring','train_backward',
            'optimizer_update','replay_selection','replay_admission','evaluation_complete'} <= {r['phase'] for r in log}


def test_failure_keeps_progress_and_action_history(tmp_path):
    torch.set_num_threads(1)
    path,data = fixture_dataset(tmp_path)
    with patch('train.load_snp_dataset', return_value=data), pytest.raises(RolloutFailure):
        train(dataset_path=str(path), output_path=str(tmp_path/'run'), epochs=1, max_events=1,
              init_z_sample_count=2, verbose=False, model_kwargs=dict(embedding_size=16,
              hidden_size=32, transformer_depth=1, transformer_heads=2))
    log = records(tmp_path/'run/progress.jsonl')
    assert log[0]['phase'] == 'ready'
    assert any(row['phase'] == 'flow_initialization' for row in log)
    assert log[-1]['phase'] == 'failed'
    failure = json.loads((tmp_path/'run/failure.json').read_text())
    assert len(failure['histories'][0]) == 1
