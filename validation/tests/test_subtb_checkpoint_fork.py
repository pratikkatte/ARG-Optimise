"""A lambda experiment resumes the same state with only its requested objective change."""
import hashlib
from unittest.mock import patch

import pytest
import torch
from train import train
from training.checkpoints import load_checkpoint
from validation.scripts.fork_subtb_checkpoint import prepare
from validation.tests.test_batch_checkpoint_fork import assert_same
from validation.tests.test_infinite_sites_configuration import fixture_dataset


def test_subtb_fork_retains_state_and_uses_new_objective(tmp_path):
    torch.set_num_threads(1)
    dataset, data = fixture_dataset(tmp_path)
    with patch('train.load_snp_dataset', return_value=data):
        train(dataset_path=str(dataset), output_path=str(tmp_path/'parent'), epochs=1,
              batch_size=2, replay_fraction=0., init_z_sample_count=2, checkpoint_every=1,
              subtb_lambda=1., tb_loss_weight=.25, verbose=False,
              model_kwargs=dict(embedding_size=16, hidden_size=32,
                  transformer_depth=1, transformer_heads=2))
        parent = tmp_path/'parent/checkpoints/checkpoint_0001.pt'
        digest = hashlib.sha256(parent.read_bytes()).hexdigest()
        child = prepare(parent, tmp_path/'fork', .9)
        old, new = load_checkpoint(parent), load_checkpoint(child)
        for key in ('generator_state_dict','opt_state_dict','trainer','rng','scheduler'):
            assert_same(old[key], new[key])
        for key in ('observations','environment','environment_fingerprint','model'):
            assert_same(old['metadata'][key], new['metadata'][key])
        for key in ('resolved_config','generator_config'):
            a, b = old['metadata'][key], new['metadata'][key]
            assert {k for k in a if a[k] != b[k]} == {'subtb_lambda'}
            assert b['subtb_lambda'] == .9 and b['tb_loss_weight'] == .25
        assert hashlib.sha256(parent.read_bytes()).hexdigest() == digest
        g, trainer = train(resume_checkpoint=str(child), output_path=str(tmp_path/'continued'),
                           epochs=2, verbose=False)
        assert g.subtb_lambda == .9 and g.tb_loss_weight == .25
        assert trainer.completed_updates == 2
    continued = load_checkpoint(tmp_path/'continued/checkpoints/checkpoint_0002.pt')
    assert continued['metadata']['training_fork'] == new['metadata']['training_fork']
    with pytest.raises(ValueError, match='empty output'):
        prepare(parent, tmp_path/'fork', .9)
    for value in (1., -1., float('nan'), float('inf'), True):
        with pytest.raises(ValueError):
            prepare(parent, tmp_path/'invalid', value)
