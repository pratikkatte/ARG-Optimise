"""Check direct output-directory routing without loading posterior samples."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pytest
from paper.scripts.paper_datasets.cli import load_config


def test_directories_replace_archived_sources(tmp_path):
    for dataset in ('r1', 'r2', 'r4'):
        folder = tmp_path / 'argflow' / dataset / 'checkpoint'
        folder.mkdir(parents=True)
        (folder / 'manifest.json').write_text('{}')
    argv = [item for method in ('argflow', 'singer', 'arginfer')
            for item in (f'--{method}-dir', str(tmp_path / method))]
    _, config = load_config(argparse.ArgumentParser(), argv)
    for dataset, settings in config['datasets'].items():
        sources = settings['sources']
        assert [s['expected_samples'] for s in sources] == [1800, 1000, 1800]
        assert sources[0]['directory'] == str(tmp_path / 'argflow' / dataset / 'checkpoint')
        assert sources[1]['input_dir'] == str(tmp_path / 'arginfer' / 'inputs' / dataset)
        assert sources[2]['directory'] == str(tmp_path / 'arginfer' / dataset)
        assert config['figure2']['main_argflows_checkpoints'][dataset] == 'checkpoint'
        assert config['figure3']['main_argflows_checkpoints'][dataset] == 'checkpoint'
    extra = tmp_path / 'argflow' / 'r1' / 'other'
    extra.mkdir()
    (extra / 'manifest.json').write_text('{}')
    with pytest.raises(SystemExit):
        load_config(argparse.ArgumentParser(), argv)


def test_partial_overrides_rejected():
    with pytest.raises(SystemExit):
        load_config(argparse.ArgumentParser(), ['--argflow-dir', 'samples'])


def test_config_only_preserved():
    _, config = load_config(argparse.ArgumentParser(), [])
    assert config['datasets']['r1']['sources'][0]['directory'].startswith('validation/')
