"""Batch experiments preserve the scientific problem and optimizer history."""
import hashlib
import json
from unittest.mock import patch
import pytest
import torch
from train import train
from training.checkpoints import load_checkpoint
from validation.scripts.fork_batch_checkpoint import prepare
from validation.tests.test_infinite_sites_configuration import fixture_dataset


def assert_same(a,b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:
            assert_same(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):
            assert_same(x,y)
    elif hasattr(a,'shape'):
        import numpy as np
        np.testing.assert_array_equal(a,b)
    else:
        assert a==b


def test_batch_fork_retains_state_and_resumes_with_new_batch(tmp_path):
    torch.set_num_threads(1)
    dataset,data=fixture_dataset(tmp_path)
    with patch('train.load_snp_dataset',return_value=data):
        train(dataset_path=str(dataset),output_path=str(tmp_path/'parent'),epochs=1,
              batch_size=2,replay_fraction=0.,init_z_sample_count=2,checkpoint_every=1,
              verbose=False,model_kwargs=dict(embedding_size=16,hidden_size=32,
                  transformer_depth=1,transformer_heads=2))
        parent=tmp_path/'parent/checkpoints/checkpoint_0001.pt'
        digest=hashlib.sha256(parent.read_bytes()).hexdigest()
        child=prepare(parent,tmp_path/'fork',4,2)
        old,new=load_checkpoint(parent),load_checkpoint(child)
        for key in ('generator_state_dict','opt_state_dict','trainer','rng','scheduler'):
            assert_same(old[key],new[key])
        for key in ('observations','environment','environment_fingerprint','generator_config','model'):
            assert_same(old['metadata'][key],new['metadata'][key])
        assert new['metadata']['training_fork']['parent_sha256']==digest
        assert hashlib.sha256(parent.read_bytes()).hexdigest()==digest
        assert new['metadata']['resolved_config']['batch_size']==4
        assert new['metadata']['resolved_config']['grad_accum_steps']==2
        _,trainer=train(resume_checkpoint=str(child),output_path=str(tmp_path/'continued'),
                        epochs=2,verbose=False)
    assert trainer.completed_updates==2
    row=json.loads((tmp_path/'continued/training.jsonl').read_text().splitlines()[0])
    assert row['fresh']+row['compatible_proposal']+row['replay']==4
    assert row['grad_accum_steps']==2
    continued=load_checkpoint(tmp_path/'continued/checkpoints/checkpoint_0002.pt')
    assert continued['metadata']['training_fork']==new['metadata']['training_fork']
    with pytest.raises(ValueError,match='empty output'):
        prepare(parent,tmp_path/'fork',4,2)
    with pytest.raises(ValueError):
        prepare(parent,tmp_path/'invalid',4,5)
