"""Simulation/export invariants; all reference and output files are temporary."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import tskit
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'merge_msprime.py'
spec = importlib.util.spec_from_file_location('merge_msprime', SCRIPT)
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)


def binary_tree(positions=(0.1, 1.9, 7.99), length=8):
    tables = tskit.TableCollection(length)
    tables.time_units = 'generations'
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    root = tables.nodes.add_row(time=10)
    tables.edges.add_row(0, length, root, 0)
    tables.edges.add_row(0, length, root, 1)
    for i, position in enumerate(positions):
        site = tables.sites.add_row(position, '0')
        tables.mutations.add_row(site=site, node=i % 2, derived_state='1', time=5)
    tables.sort()
    return tables.tree_sequence()


def fasta_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith('>'):
            rows.append('')
        else:
            rows[-1] += line
    return rows


def test_recoding_preserves_mutations_and_matches_vcf_fasta(tmp_path):
    original = binary_tree()
    reference = 'ACGTACGT'
    ts, dropped = sim.to_infinite_sites_nucleotides(original, reference, seed=44)
    assert dropped == 0
    assert not any(sim.vcf_site_mask(ts))
    assert np.array_equal(original.genotype_matrix(), ts.genotype_matrix())
    assert np.array_equal(ts.tables.sites.position, [0, 1, 7])
    assert np.array_equal(original.tables.mutations.node, ts.tables.mutations.node)
    assert np.array_equal(original.tables.mutations.time, ts.tables.mutations.time)
    assert ts.tables.reference_sequence.data == reference
    assert all(len(s.mutations) == 1 for s in ts.sites())
    repeat, _ = sim.to_infinite_sites_nucleotides(original, reference, seed=44)
    assert ts.tables.sites == repeat.tables.sites
    assert ts.tables.mutations == repeat.tables.mutations

    fasta = tmp_path / 'test.fa'
    vcf = tmp_path / 'test.vcf'
    sim.write_haplotype_fasta(ts, fasta, reference_sequence=reference)
    sim.write_vcf(ts, vcf)
    rows = fasta_rows(fasta)
    records = [line.split('\t') for line in vcf.read_text().splitlines() if not line.startswith('#')]
    assert len(records) == 3
    assert [int(row[1]) for row in records] == [1, 2, 8]
    assert '##contig=<ID=1,length=8>' in vcf.read_text()
    for record, variant in zip(records, ts.variants()):
        pos = int(record[1])-1
        assert record[3] == reference[pos] == variant.site.ancestral_state
        assert record[4] != record[3]
        gt = [int(x) for x in record[9].split('|')]
        assert gt == list(variant.genotypes)
        assert [row[pos] for row in rows] == [variant.alleles[g] for g in gt]
    assert all(rows[sample][pos] == reference[pos] for sample in range(2)
               for pos in range(8) if pos not in (0, 1, 7))


def test_collision_is_explicit_and_drop_keeps_original_truth(tmp_path):
    original = binary_tree((0.1, 0.7, 3.2))
    with pytest.raises(ValueError, match='1 mutations collide'):
        sim.to_infinite_sites_nucleotides(original, 'ACGTACGT', seed=1)
    ts, dropped = sim.to_infinite_sites_nucleotides(
        original, 'ACGTACGT', seed=1, collision_policy='drop')
    assert dropped == 1
    assert ts.num_sites == ts.num_mutations == 2
    assert original.num_sites == 3
    assert np.array_equal(ts.genotype_matrix(), original.genotype_matrix()[[0, 2]])
    provenance = json.loads(ts.provenance(ts.num_provenances-1).record)['parameters']
    assert provenance['num_dropped_sites'] == 1
    assert provenance['collision_policy'] == 'drop'
    metadata = sim.write_metadata(
        ts, tmp_path/'metadata.json', dataset_name='fixture', replicate_index=0,
        contig_id='1', reference_fasta=tmp_path/'ref.fa', reference_contig='test',
        reference_start=0, reference_sequence='ACGTACGT', infinite_sites_ts=original,
        export_collision_policy='drop', num_dropped_sites=dropped)
    assert metadata['integer_export']['num_continuous_sites'] == 3
    assert metadata['integer_export']['num_dropped_collision_sites'] == 1
    assert metadata['integer_export']['all_mutations_retained'] is False
    assert metadata['summary']['num_exported_sites'] == 2
    assert [s['continuous_source_site_id'] for s in metadata['sites']] == [0, 2]


def test_recurrent_mutations_and_ambiguous_reference_are_rejected():
    original = binary_tree()
    tables = original.dump_tables()
    tables.mutations.add_row(site=0, node=0, derived_state='0', parent=0, time=1)
    tables.sort()
    with pytest.raises(ValueError, match='exactly one binary'):
        sim.to_infinite_sites_nucleotides(tables.tree_sequence(), 'ACGTACGT', seed=1)
    with pytest.raises(ValueError, match='only A/C/G/T'):
        sim.to_infinite_sites_nucleotides(original, 'NCGTACGT', seed=1)
    with pytest.raises(ValueError, match='match the integer sequence length'):
        sim.to_infinite_sites_nucleotides(original, 'ACGT', seed=1)


def test_fractional_ancestry_breakpoints_cannot_be_floored():
    tables = binary_tree((0.7,)).dump_tables()
    tables.edges.clear()
    other_root = tables.nodes.add_row(time=12)
    for child in (0, 1):
        tables.edges.add_row(0, 0.5, 2, child)
        tables.edges.add_row(0.5, 8, other_root, child)
    tables.sort()
    with pytest.raises(ValueError, match='integer ancestry breakpoints'):
        sim.to_infinite_sites_nucleotides(tables.tree_sequence(), 'ACGTACGT', seed=1)


@pytest.mark.parametrize('mu', [0, 1e-7])
def test_complete_simulation_metadata_truth_and_exports(tmp_path, mu):
    reference = 'ACGT'*500
    ref = tmp_path/'reference.fa'
    ref.write_text('>test\n'+reference+'\n')
    sim.simulate(nrep=1, n=8, dataset_name='fixture', mu=mu, rec=5e-8,
                 Ne=10000, length=2000, seed=42, output_dir=tmp_path,
                 reference_fasta=ref, reference_url=None, reference_contig='test',
                 reference_start=0)
    rep = tmp_path/'fixture'/'rep0'
    ts = tskit.load(rep/'fixture.trees')
    exact = tskit.load(rep/'fixture.infinite_sites.trees')
    full = tskit.load(rep/'fixture.full.trees')
    metadata = json.loads((rep/'metadata.json').read_text())
    assert full.num_mutations == 0
    assert full.num_nodes >= ts.num_nodes
    assert exact.num_mutations == exact.num_sites == ts.num_sites
    assert (ts.num_sites == 0) == (mu == 0)
    assert all(len(site.mutations) == 1 for site in exact.sites())
    assert np.array_equal(exact.genotype_matrix(), ts.genotype_matrix())
    assert metadata['mutation_model']['name'] == 'infinite_sites'
    assert metadata['simulation']['sim_mutations']['parameters']['discrete_genome'] is False
    assert metadata['simulation']['sim_ancestry']['parameters']['record_full_arg'] is True
    assert metadata['simulation']['to_infinite_sites_nucleotides']['parameters']['seed'] == 44
    assert metadata['integer_export']['all_mutations_retained']
    assert metadata['summary']['num_filtered_sites'] == 0
    assert metadata['reference']['vcf_ref_mismatch_positions_zero_based'] == []
    assert metadata['full_ancestry']['record_full_arg'] is True
    assert len(list((rep/'tcoalmap').glob('*.tc'))) == 28
    rows = fasta_rows(rep/'fixture.fa')
    assert len(rows) == 8 and all(len(row) == 2000 for row in rows)
    records = [line for line in (rep/'fixture.vcf').read_text().splitlines()
               if not line.startswith('#')]
    assert len(records) == ts.num_sites
    for variant, record in zip(ts.variants(), records):
        fields = record.split('\t')
        genotypes = [int(g) for sample in fields[9:] for g in sample.split('|')]
        assert genotypes == list(variant.genotypes)
        pos = int(variant.site.position)
        assert [row[pos] for row in rows] == [variant.alleles[g] for g in genotypes]


def test_failed_integer_export_still_saves_exact_truth(tmp_path, monkeypatch):
    exact = binary_tree((0.1, 0.7))
    monkeypatch.setattr(sim.msprime, 'sim_mutations', lambda *args, **kwargs: exact)
    ref = tmp_path/'reference.fa'
    ref.write_text('>test\nACGTACGT\n')
    with pytest.raises(ValueError, match='collide'):
        sim.simulate(nrep=1, n=2, dataset_name='fixture', length=8, output_dir=tmp_path,
                     reference_fasta=ref, reference_url=None, reference_contig='test',
                     reference_start=0)
    rep = tmp_path/'fixture'/'rep0'
    assert tskit.load(rep/'fixture.infinite_sites.trees').num_sites == 2
    assert (rep/'fixture.full.trees').exists()
    assert not (rep/'fixture.vcf').exists()


@pytest.mark.parametrize('setting,value', [('export_collision_policy', 'ignore'), ('seed', 2**32-2)])
def test_config_rejects_bad_policy_and_seed_range(tmp_path, setting, value):
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump({setting: value}))
    with pytest.raises(ValueError):
        sim.load_config(path)
