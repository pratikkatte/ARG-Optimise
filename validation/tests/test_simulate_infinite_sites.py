import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import tskit
import yaml

SCRIPT = Path(__file__).resolve().parents[1]/'scripts'/'simulate_infinite_sites.py'
spec = importlib.util.spec_from_file_location('standalone_infinite_sites', SCRIPT)
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)


@pytest.mark.parametrize('mu', [0, 1e-7])
def test_yaml_run_and_complete_genotype_exports(tmp_path, mu):
    config_path = tmp_path/'config.yaml'
    settings = {**sim.DEFAULTS, 'mutation_rate': mu, 'output_dir': 'output'}
    config_path.write_text(yaml.safe_dump(settings))
    sim.main(['--config', str(config_path)])
    rep = tmp_path/'output'/settings['dataset_name']/'rep0'
    name = settings['dataset_name']
    ts = tskit.load(rep/f'{name}.trees')
    full = tskit.load(rep/f'{name}.full.trees')
    assert ts.num_mutations == ts.num_sites
    assert (ts.num_sites == 0) == (mu == 0)
    assert full.num_mutations == 0
    assert ts.num_samples == 8
    assert len(list((rep/'tcoalmap').glob('*.tc'))) == 28
    variants = list(ts.variants())
    assert all(len(v.site.mutations) == 1 and len(v.alleles) == 2 for v in variants)
    assert all(set(v.alleles) <= set('ACGT') for v in variants)
    records = [line.split('\t') for line in (rep/f'{name}.vcf').read_text().splitlines()
               if not line.startswith('#')]
    with (rep/f'{name}.positions.tsv').open() as handle:
        mapping = list(csv.DictReader(handle, delimiter='\t'))
    assert len(records) == len(mapping) == ts.num_sites
    positions = [int(row[1]) for row in records]
    assert positions == sorted(set(positions)) and all(p > 0 for p in positions)
    for variant, record, row in zip(variants, records, mapping):
        assert record[3] == variant.site.ancestral_state
        assert record[4] == variant.alleles[1]
        assert all('|' in gt for gt in record[9:])
        assert [int(g) for gt in record[9:] for g in gt.split('|')] == list(variant.genotypes)
        assert float(row['position_zero_based']) == variant.site.position
        assert int(row['vcf_position_one_based']) == int(record[1])
    metadata = json.loads((rep/'metadata.json').read_text())
    assert metadata['mutation_model'] == 'infinite_sites'
    assert metadata['mutation_seed'] == 43
    assert metadata['simulation']['sim_mutations']['parameters']['discrete_genome'] is False
    assert metadata['simulation']['sim_ancestry']['parameters']['population_size'] == 10000
    assert metadata['summary']['sequence_length_bp'] == 2000
    # Same seeds reproduce site coordinates, mutation carriers, times and labels.
    sim.main(['--config', str(config_path)])
    repeated = tskit.load(rep/f'{name}.trees')
    assert ts.tables.sites == repeated.tables.sites
    assert ts.tables.mutations == repeated.tables.mutations


def test_legacy_vcf_collision_mapping_keeps_every_snp(tmp_path):
    tables = tskit.TableCollection(2)
    for _ in range(2):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    root = tables.nodes.add_row(time=10)
    for sample in range(2):
        tables.edges.add_row(0, 2, root, sample)
    original_positions = [0.01, 0.05, 0.15, 0.9, 1.1]
    for pos in original_positions:
        site = tables.sites.add_row(pos, 'A')
        tables.mutations.add_row(site=site, node=0, derived_state='G', time=1)
    tables.sort()
    ts = tables.tree_sequence()
    vcf, mapping = tmp_path/'test.vcf', tmp_path/'positions.tsv'
    shifted = sim.write_vcf_and_position_map(ts, vcf, mapping, '1')
    records = [line.split('\t') for line in vcf.read_text().splitlines() if not line.startswith('#')]
    assert len(records) == len(original_positions)
    assert [int(row[1]) for row in records] == [1, 2, 3, 4, 5]
    assert '##contig=<ID=1,length=5>' in vcf.read_text()
    assert shifted == 5
    with mapping.open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    assert [float(row['position_zero_based']) for row in rows] == original_positions
    assert np.array_equal(ts.tables.sites.position, original_positions)


def test_replicate_seeds_and_optional_full_ancestry(tmp_path):
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump({
        'num_samples': 2, 'num_replicates': 2, 'sequence_length': 50,
        'mutation_rate': 0, 'record_full_arg': False, 'output_dir': 'data',
    }))
    sim.main(['--config', str(path)])
    for i in range(2):
        rep = tmp_path/'data'/sim.DEFAULTS['dataset_name']/f'rep{i}'
        metadata = json.loads((rep/'metadata.json').read_text())
        assert metadata['ancestry_seed'] == 42+i*100000
        assert metadata['mutation_seed'] == 43+i*100000
        assert not list(rep.glob('*.full.trees'))


@pytest.mark.parametrize('setting,value', [
    ('num_samples', 3), ('mutation_rate', -1), ('seed', 2**32-1),
    ('mutation_rate', 'nan'), ('record_full_arg', 'yes'), ('mutation_rtae', 1e-7),
])
def test_invalid_config(tmp_path, setting, value):
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump({setting: value}))
    with pytest.raises(ValueError):
        sim.load_config(path)
