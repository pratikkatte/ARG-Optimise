"""Training failures retain attempted action histories."""
import json
from unittest.mock import patch

import pytest
import torch

from gfn.rollout import RolloutFailure
from train import train
from validation.tests.test_infinite_sites_configuration import fixture_dataset


def test_failure_keeps_action_history(tmp_path):
    torch.set_num_threads(1)
    path,data = fixture_dataset(tmp_path)
    with patch('train.load_snp_dataset', return_value=data), pytest.raises(RolloutFailure):
        train(dataset_path=str(path), output_path=str(tmp_path/'run'), epochs=1, max_events=1,
              init_z_sample_count=2, verbose=False, model_kwargs=dict(embedding_size=16,
              hidden_size=32, transformer_depth=1, transformer_heads=2))
    assert not (tmp_path/'run/progress.jsonl').exists()
    failure = json.loads((tmp_path/'run/failure.json').read_text())
    assert len(failure['histories'][0]) == 1
