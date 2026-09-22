"""Validate frozen long-run bundles and launch arguments without submitting jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml
from validation.scripts.prepare_long_run import prepare, ROOT


def arguments(output, **overrides):
    options = dict(output=str(output), source=str(ROOT),
        config=str(ROOT / 'config/paper_datasets/stable/r1.yaml'), checkpoint=None,
        hours=48, epochs=10000, wandb=False)
    options.update(overrides)
    return argparse.Namespace(**options)


def test_frozen_config_and_batch_launch_branches(tmp_path):
    bundle = prepare(arguments(tmp_path / 'bundle with spaces'))
    config = yaml.safe_load((bundle / 'config.yaml').read_text())
    assert config['max_wall_seconds'] == 47 * 3600
    assert config['eval_every'] == config['checkpoint_every'] == 250
    assert config['terminal_eval_repeat_every'] == 1000
    manifest = json.loads((bundle / 'manifest.json').read_text())
    assert manifest['submitted'] is False
    for name, expected in manifest['source_sha256'].items():
        assert hashlib.sha256((bundle / 'source' / name).read_bytes()).hexdigest() == expected
    script = bundle / 'submit.sbatch'
    subprocess.run(['bash', '-n', str(script)], check=True)
    mock_bin = tmp_path / 'mock_bin'
    mock_bin.mkdir()
    mock = mock_bin / 'srun'
    mock.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
        'from pathlib import Path\n'
        "Path(os.environ['ARG_LAUNCH_TEST_LOG']).write_text(json.dumps(sys.argv[1:]))\n")
    mock.chmod(0o755)
    log = tmp_path / 'arguments.json'
    env = {**os.environ, 'PATH':str(mock_bin) + os.pathsep + os.environ['PATH'],
           'ARG_LAUNCH_TEST_LOG':str(log)}
    subprocess.run(['bash', str(script)], env=env, check=True, capture_output=True)
    first = json.loads(log.read_text())
    assert first == ['--unbuffered', sys.executable, '-u', str(bundle / 'source/train.py'),
                     '--config', str(bundle / 'config.yaml')]
    missing = subprocess.run(['bash', str(script), '--resume'], env=env, capture_output=True)
    assert missing.returncode != 0 and b'No latest.pt' in missing.stderr
    latest = bundle / 'run/checkpoints/latest.pt'
    latest.parent.mkdir(parents=True)
    latest.touch()  # Mock srun only records arguments; it does not load this file.
    subprocess.run(['bash', str(script), '--resume'], env=env, check=True, capture_output=True)
    resumed = json.loads(log.read_text())
    assert resumed[resumed.index('--resume-checkpoint') + 1] == str(latest)
    assert resumed[resumed.index('--dataset-path') + 1] == config['dataset_path']
    assert '--config' not in resumed
    with pytest.raises(ValueError, match='empty output'):
        prepare(arguments(bundle))


def test_checkpoint_bundle_preserves_optimizer_input(tmp_path):
    import torch
    from validation.tests.test_infinite_sites_neural import model
    from gfn.rollout import RolloutWorker
    from training.trainer import Trainer
    torch.set_num_threads(1)
    g = model()
    trainer = Trainer(g, RolloutWorker(g.env))
    checkpoint = tmp_path / 'original.pt'
    g.save(checkpoint, trainer=trainer, metadata=dict(resolved_config={'dataset_path':str(tmp_path)}))
    bundle = prepare(arguments(tmp_path / 'resume_bundle', config=None, checkpoint=str(checkpoint)))
    assert (bundle / 'start.pt').read_bytes() == checkpoint.read_bytes()
    config = yaml.safe_load((bundle / 'config.yaml').read_text())
    assert config['resume_checkpoint'] == str(bundle / 'start.pt')
    assert not set(config).intersection({'model_kwargs','subtb_lambda','tb_loss_weight',
        'policy_lr','encoder_lr','flow_lr','lr_schedule','lr_schedule_steps'})
