"""Published checkpoints load, match their datasets, and reproduce saved draws."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from env.snp_data import load_snp_dataset
from infer import collect_samples
from training.checkpoints import generator_from_checkpoint, load_checkpoint
from utils import action_from_dict

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = {
    'r1': 'r1_checkpoint_5650',
    'r2': 'checkpoint_4650',
    'r4': 'from1750_best_eval_step4300',
}


@pytest.mark.parametrize('dataset', sorted(PUBLISHED))
def test_published_checkpoint(dataset):
    torch.set_num_threads(1)  # small models; oversubscribing threads is far slower
    name = PUBLISHED[dataset]
    checkpoint = ROOT / 'paper/checkpoints/final' / dataset / f'{name}.pt'
    manifest_path = ROOT / 'paper/outputs/argflow' / dataset / name / 'manifest.json'
    if not checkpoint.is_file() or not manifest_path.is_file():
        pytest.skip('published checkpoints/draws are not installed')
    manifest = json.loads(manifest_path.read_text())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == manifest['checkpoint_sha256']

    generator = generator_from_checkpoint(load_checkpoint(checkpoint), 'cpu', optimizer=False)
    observed = load_snp_dataset(ROOT / 'paper/datasets' / dataset / 'rep0')
    np.testing.assert_array_equal(observed.genotypes, generator.env.snp_data.genotypes)
    np.testing.assert_array_equal(observed.positions, generator.env.snp_data.positions)
    assert generator.env.dataset_fingerprint == manifest['environment_fingerprint']

    # Replaying a saved draw through the environment reproduces its likelihood.
    record = manifest['samples'][0]
    state = generator.env.replay([action_from_dict(a) for a in record['actions']])
    assert state.is_done
    assert state.partial_log_likelihood == pytest.approx(record['log_likelihood'], rel=0, abs=1e-8)

    # Fresh policy samples are complete and their likelihoods agree with the independent check.
    records, trees, _ = collect_samples(generator, 2, 2, seed=0)
    for r, ts in zip(records, trees):
        assert r['log_likelihood'] == pytest.approx(r['independent_log_likelihood'], rel=0, abs=1e-8)
        assert ts.num_samples == observed.num_haplotypes
