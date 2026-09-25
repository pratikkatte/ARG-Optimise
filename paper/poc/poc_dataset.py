"""Create the explicitly designed two-base observations and a scored candidate."""
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_poc_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if set(config) != {'output_dir', 'dataset', 'training', 'model', 'evaluation'}:
        raise ValueError('POC config requires output_dir, dataset, training, model, evaluation')
    d = config['dataset']
    if (d['num_samples'] != 2 or d['sequence_length'] != 2 or d['positions'] != [.5, 1.5]
            or d['genotypes'] != [[1, 0], [0, 1]]):
        raise ValueError('This POC bundle is the specified two-haplotype, two-singleton dataset')
    if min(d['population_size'], d['mutation_rate'], d['recombination_rate']) <= 0:
        raise ValueError('POC rates must be positive, including recombination')
    return config


def generate_poc(config):
    from env.snp_data import load_snp_dataset
    from env.env import SimpleARGEnvironment
    from paper.poc.poc_reference import rejection_sample, trajectory

    d = config['dataset']
    directory = ROOT / config['output_dir'] / 'dataset' / 'rep0'
    directory.mkdir(parents=True, exist_ok=True)
    name = d['name']
    parameters = dict(dataset_name=name, num_samples=2, num_replicates=1, sequence_length=2,
                      population_size=d['population_size'], mutation_rate=d['mutation_rate'],
                      recombination_rate=d['recombination_rate'], contig_id='1', seed=d['seed'])
    metadata = dict(schema_version=1, dataset_name=name, replicate_index=0, parameters=parameters,
        mutation_model='infinite_sites', mutation_positions='continuous', mutation_alphabet='ACGT',
        vcf_ref='known simulated ancestral allele; no external reference genome',
        num_haplotypes=2, num_sites=2, num_mutations=2, time_units='generations',
        construction='Fixed synthetic observations specified before training; no simulated truth ARG.',
        sample_nodes_in_haplotype_order=[0, 1],
        summary=dict(sequence_length_bp=2, num_haplotypes=2, num_individuals=1, num_sites=2, num_mutations=2),
        files=dict(vcf=name+'.vcf', position_map=name+'.positions.tsv',
                   reference_candidate='reference_candidate.trees'))
    metadata_path = directory / 'metadata.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError('Existing POC dataset differs; use a new output_dir')
    candidate = directory / 'reference_candidate.trees'
    if metadata_path.exists() and candidate.exists():
        existing = load_snp_dataset(directory)
        np.testing.assert_array_equal(existing.genotypes, d['genotypes'])
        np.testing.assert_array_equal(existing.positions, d['positions'])
        return directory, candidate, d['mutation_rate']
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    (directory / (name+'.vcf')).write_text(
        '##fileformat=VCFv4.2\n##contig=<ID=1,length=2>\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased genotype">\n'
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tspl0\n'
        '1\t1\t0\tA\tC\t.\tPASS\t.\tGT\t1|0\n'
        '1\t2\t1\tA\tC\t.\tPASS\t.\tGT\t0|1\n')
    (directory / (name+'.positions.tsv')).write_text(
        'site_id\tposition_zero_based\tvcf_position_one_based\n0\t0.5\t1\n1\t1.5\t2\n')
    data = load_snp_dataset(directory)
    np.testing.assert_array_equal(data.genotypes, d['genotypes'])
    records, _ = rejection_sample(1, d['seed'], 2, 2*d['population_size']*d['mutation_rate'],
                                  2*d['population_size']*d['recombination_rate'])
    env = SimpleARGEnvironment(snp_data=data, population_size=d['population_size'],
        mutation_rate=d['mutation_rate'], recombination_rate=d['recombination_rate'])
    state = env.replay(trajectory(records[0]).actions)
    env.save_to_tree_sequence(state, candidate)
    (directory / 'observations.json').write_text(json.dumps(d, indent=2)+'\n')
    return directory, candidate, d['mutation_rate']
