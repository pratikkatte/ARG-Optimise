"""Standalone msprime infinite-sites simulation from a YAML configuration.

Outputs .trees, .vcf, a VCF position map, pairwise coalescence times and metadata.
The tree sequence retains exact continuous positions. VCF uses the same legacy
integer coordinate conversion as the ARGsims scripts; no mutations are dropped.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import msprime
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'config.yaml'
DEFAULTS = {
    'dataset_name': 'sim_2kb_infinite_sites_simple',
    'num_replicates': 1, 'num_samples': 8, 'population_size': 10000,
    'mutation_rate': 1e-7, 'recombination_rate': 5e-8, 'sequence_length': 2000,
    'seed': 42, 'contig_id': '1', 'output_dir': '../datasets',
    'record_full_arg': True,
}


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open() as handle:
        supplied = yaml.safe_load(handle)
    if not isinstance(supplied, dict):
        raise ValueError('config must be a YAML mapping')
    unknown = supplied.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError('unknown settings: ' + ', '.join(sorted(map(str, unknown))))
    config = {**DEFAULTS, **supplied}
    for key, minimum in [('num_replicates', 1), ('num_samples', 2),
                         ('sequence_length', 1), ('seed', 1)]:
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    if config['num_samples'] % 2:
        raise ValueError('num_samples counts haplotypes and must be even')
    if config['seed'] + (config['num_replicates']-1)*100000 + 1 >= 2**32:
        raise ValueError('replicate seeds exceed the msprime seed range')
    for key in ('population_size', 'mutation_rate', 'recombination_rate'):
        try:
            value = float(config[key])
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f'{key} must be a finite number') from None
        if (isinstance(config[key], bool) or not math.isfinite(value) or value < 0
                or (key == 'population_size' and value == 0)):
            raise ValueError(f'invalid {key}: {config[key]}')
        config[key] = value
    name = config['dataset_name']
    if (not isinstance(name, str) or not name.strip() or name in ('.', '..')
            or '/' in name or '\\' in name):
        raise ValueError('dataset_name must be a filename stem without directories')
    if type(config['record_full_arg']) is not bool:
        raise ValueError('record_full_arg must be true or false')
    contig = config['contig_id']
    if (isinstance(contig, bool) or not isinstance(contig, (str, int))
            or not str(contig) or any(c.isspace() for c in str(contig))):
        raise ValueError('contig_id must be a nonempty identifier')
    config['contig_id'] = str(contig)
    if not isinstance(config['output_dir'], str) or not config['output_dir'].strip():
        raise ValueError('output_dir must be a nonempty path')
    config['output_dir'] = (path.parent / Path(config['output_dir']).expanduser()).resolve()
    return config


def write_vcf_and_position_map(ts, vcf_path, map_path, contig_id):
    """Keep every SNP; record the actual integer positions written by tskit."""
    kwargs = {'ploidy': 2} if ts.num_individuals == 0 else {}
    with Path(vcf_path).open('w') as handle:
        ts.write_vcf(handle, contig_id=contig_id, position_transform='legacy',
                     individual_names=[f'spl{i}' for i in range(ts.num_samples//2)],
                     **kwargs)
    shifted = 0
    with Path(vcf_path).open() as vcf, Path(map_path).open('w') as mapping:
        writer = csv.writer(mapping, delimiter='\t', lineterminator='\n')
        writer.writerow(['site_id', 'position_zero_based', 'vcf_position_one_based'])
        records = (line.split('\t') for line in vcf if not line.startswith('#'))
        for site in ts.sites():
            position = int(next(records)[1])
            writer.writerow([site.id, repr(site.position), position])
            shifted += position != round(site.position)
    return shifted


def simulate(config):
    name = config['dataset_name']
    for rep in range(config['num_replicates']):
        seed = config['seed'] + rep*100000
        ancestry = msprime.sim_ancestry(
            samples=config['num_samples']//2, ploidy=2,
            population_size=config['population_size'],
            sequence_length=config['sequence_length'],
            recombination_rate=config['recombination_rate'], discrete_genome=True,
            record_full_arg=config['record_full_arg'], random_seed=seed,
        )
        # Continuous mutation positions enforce infinite sites. The nucleotide
        # alphabet supplies A/C/G/T directly, without reference/FASTA conversion.
        ts = msprime.sim_mutations(
            ancestry.simplify(), rate=config['mutation_rate'],
            model=msprime.InfiniteSites(msprime.NUCLEOTIDES),
            discrete_genome=False, keep=False, random_seed=seed+1,
        )
        if any(len(site.mutations) != 1 for site in ts.sites()):
            raise RuntimeError('expected exactly one mutation per site')
        directory = config['output_dir'] / name / f'rep{rep}'
        directory.mkdir(parents=True, exist_ok=True)
        if config['record_full_arg']:
            ancestry.dump(directory / f'{name}.full.trees')
        ts.dump(directory / f'{name}.trees')
        shifted = write_vcf_and_position_map(
            ts, directory/f'{name}.vcf', directory/f'{name}.positions.tsv', config['contig_id'])
        tcdir = directory/'tcoalmap'
        tcdir.mkdir(exist_ok=True)
        samples = list(ts.samples())
        for i, first in enumerate(samples):
            for j in range(i+1, len(samples)):
                with (tcdir/f'{name}_spls{i}-{j}.tc').open('w') as handle:
                    for tree in ts.trees():
                        print(*tree.interval, tree.tmrca(first, samples[j]), sep='\t', file=handle)
        simulation = {}
        for provenance in ts.provenances():
            record = json.loads(provenance.record)
            parameters = dict(record.get('parameters', {}))
            command = parameters.pop('command', None)
            if command in ('sim_ancestry', 'sim_mutations'):
                parameters.pop('tree_sequence', None)
                simulation[command] = {'software': record['software'], 'parameters': parameters}
        metadata = {
            'schema_version': 1, 'dataset_name': name, 'replicate_index': rep,
            'parameters': {key: str(value) if isinstance(value, Path) else value
                           for key, value in config.items()},
            'ancestry_seed': seed, 'mutation_seed': seed+1,
            'mutation_model': 'infinite_sites', 'mutation_alphabet': 'ACGT',
            'mutation_positions': 'continuous', 'num_mutations': ts.num_mutations,
            'num_sites': ts.num_sites, 'num_haplotypes': ts.num_samples,
            'num_trees': ts.num_trees, 'time_units': ts.time_units,
            'vcf_positions': 'legacy: round, then advance ties/zero to positive distinct integers',
            'vcf_positions_advanced_beyond_rounding': shifted,
            'coordinate_note': ('VCF positions are approximate and can cross ancestry breakpoints '
                                'or exceed the simulated length. Use .trees or .positions.tsv '
                                'for exact mutation positions and genomic distances.'),
            'vcf_ref': 'known simulated ancestral allele; no external reference genome',
            'full_ancestry': 'unmutated .full.trees when record_full_arg is true',
            'sample_nodes_in_haplotype_order': [int(node) for node in ts.samples()],
            'software': {'msprime': msprime.__version__},
            'simulation': simulation,
            'summary': {'sequence_length_bp': config['sequence_length'],
                        'num_haplotypes': ts.num_samples, 'num_individuals': ts.num_individuals,
                        'num_sites': ts.num_sites, 'num_mutations': ts.num_mutations},
            'files': {'vcf': f'{name}.vcf', 'ground_truth_trees': f'{name}.trees',
                      'position_map': f'{name}.positions.tsv',
                      'pairwise_coalescence_directory': 'tcoalmap'},
        }
        (directory/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print(f'{directory}: {ts.num_sites} SNPs, {ts.num_trees} trees; '
              f'{shifted} VCF positions advanced beyond rounding')


def main(argv=None, *, default_config=DEFAULT_CONFIG):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=default_config,
                        help=f'YAML settings; defaults to {default_config}')
    args = parser.parse_args(argv)
    try:
        simulate(load_config(args.config))
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
