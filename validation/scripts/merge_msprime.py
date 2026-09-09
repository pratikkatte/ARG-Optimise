## ancestry + JC69 mutations (modern msprime; replaces 1_msprime_sim + 2_msprimefinitesites)

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import urllib.request

import msprime
import numpy as np
import yaml

refdir = '../reference/'
contig_id = '1'
seed = 42

HG38_URL = 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz'
HG38_FASTA = refdir + 'hg38.fa.gz'
HG38_CONTIG = 'chr1'
HG38_START = 10_000_000
_ACGT = frozenset('ACGT')
VALIDATION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = VALIDATION_DIR / 'config/simulate_config.yaml'


def ensure_hg38_reference(reference_fasta=HG38_FASTA, reference_url=HG38_URL):
    reference_fasta = os.fspath(reference_fasta)
    if os.path.exists(reference_fasta) and os.path.getsize(reference_fasta) > 0:
        return reference_fasta
    if not reference_url:
        raise ValueError('reference FASTA is missing or empty: ' + reference_fasta)
    os.makedirs(os.path.dirname(os.path.abspath(reference_fasta)), exist_ok=True)

    tmp_path = reference_fasta + '.tmp'
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print('downloading reference from', reference_url)
    print('writing reference to', reference_fasta)
    with urllib.request.urlopen(reference_url) as response, open(tmp_path, 'wb') as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp_path, reference_fasta)
    return reference_fasta


def read_reference_window(
    length, contig=HG38_CONTIG, start=HG38_START,
    reference_fasta=HG38_FASTA, reference_url=HG38_URL,
):
    length = int(length)
    start = int(start)
    if length < 0:
        raise ValueError('reference window length must be non-negative')
    if start < 0:
        raise ValueError('reference window start must be non-negative')
    if length == 0:
        return ''

    reference_fasta = ensure_hg38_reference(reference_fasta, reference_url)
    end = start + length
    seq_parts = []
    collected = 0
    pos = 0
    in_contig = False
    found_contig = False

    opener = gzip.open if reference_fasta.endswith('.gz') else open
    with opener(reference_fasta, 'rt') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if in_contig:
                    break
                name = line[1:].split()[0]
                in_contig = name == contig
                found_contig = found_contig or in_contig
                pos = 0
                continue
            if not in_contig:
                continue

            line = line.upper()
            next_pos = pos + len(line)
            if next_pos <= start:
                pos = next_pos
                continue
            if pos >= end:
                break

            left = max(start - pos, 0)
            right = min(end - pos, len(line))
            piece = line[left:right]
            seq_parts.append(piece)
            collected += len(piece)
            pos = next_pos
            if collected >= length:
                break

    if not found_contig:
        raise ValueError('contig {} not found in {}'.format(contig, reference_fasta))
    if collected != length:
        raise ValueError(
            'reference window {}:{}-{} is shorter than requested length {}'.format(
                contig, start, end, length
            )
        )
    return ''.join(seq_parts)


def vcf_site_mask(ts):
    mask = [False] * ts.num_sites
    seen_positions = set()
    for variant in ts.variants():
        site = variant.site
        position = float(site.position)
        int_position = int(position)
        bad = (
            position != int_position
            or int_position < 0
            or any(
                len(str(allele).upper()) != 1 or str(allele).upper() not in _ACGT
                for allele in variant.alleles
                if allele is not None
            )
            or len([a for a in variant.alleles if a is not None]) > 2
            or int_position in seen_positions
        )
        if bad:
            mask[site.id] = True
        else:
            seen_positions.add(int_position)
    return mask


def write_vcf(ts, vcf_path, contig_id=contig_id, site_mask=None):
    """Export an existing tree sequence using one-based VCF coordinates."""
    if site_mask is None:
        site_mask = vcf_site_mask(ts)
    vcf_kwargs = {
        'contig_id': contig_id,
        'individual_names': ['spl' + str(s) for s in range(ts.num_samples // 2)],
        'site_mask': np.asarray(site_mask, dtype=bool),
        # tskit also transforms the contig length; keep that endpoint fixed.
        'position_transform': lambda positions: np.minimum(
            np.asarray(positions) + 1, ts.sequence_length
        ),
    }
    if ts.num_individuals == 0:
        vcf_kwargs['ploidy'] = 2
    with open(vcf_path, 'w', encoding='utf-8') as vcffh:
        ts.write_vcf(vcffh, **vcf_kwargs)


def write_haplotype_fasta(ts, fasta_path, site_mask=None, reference_sequence=None):
    """Use tskit's FASTA exporter with the same site selection as the VCF.

    The reference supplies nonvariant bases. Native FASTA headers are n<node ID>,
    in ts.samples() order. Filtering an export copy leaves the truth ARG intact.
    """
    sequence_length = int(ts.sequence_length)
    if sequence_length != float(ts.sequence_length):
        raise ValueError('FASTA export requires integer sequence length')

    if reference_sequence is None:
        reference_sequence = read_reference_window(sequence_length)
    if site_mask is None:
        site_mask = vcf_site_mask(ts)
    export_ts = ts.delete_sites(np.flatnonzero(site_mask), record_provenance=False)
    export_ts.write_fasta(
        fasta_path, reference_sequence=reference_sequence, wrap_width=80
    )


def write_metadata(
    ts, metadata_path, *, dataset_name, replicate_index, contig_id,
    reference_fasta, reference_contig, reference_start, reference_sequence,
    reference_url=None, site_mask=None,
):
    """Describe a replicate, its provenance, and the coordinate/export conventions.

    Breakpoints are boundaries between marginal trees. They do not count all
    historical recombination events, which the default simulation does not record.
    Simulation parameters come from the saved tree sequence's provenance.
    """
    metadata_path = Path(metadata_path)
    if site_mask is None:
        site_mask = vcf_site_mask(ts)
    if len(site_mask) != ts.num_sites:
        raise ValueError('site_mask must have one entry per tree-sequence site')
    if len(reference_sequence) != ts.sequence_length:
        raise ValueError('reference_sequence must span the complete tree sequence')
    provenance = [json.loads(p.record) for p in ts.provenances()]
    simulation = {}
    for record in provenance:
        parameters = dict(record.get('parameters', {}))
        command = parameters.pop('command', None)
        if command in ('sim_ancestry', 'sim_mutations'):
            parameters.pop('tree_sequence', None)
            simulation[command] = {
                'software': record.get('software', {}),
                'parameters': parameters,
            }
    sample_nodes = [int(node) for node in ts.samples()]
    sites = []
    segregating_sites = 0
    exported_segregating_sites = 0
    reference_mismatches = []
    for variant in ts.variants():
        site = variant.site
        position = int(site.position)
        segregating = len(set(variant.genotypes) - {-1}) > 1
        exported = not bool(site_mask[site.id])
        segregating_sites += int(segregating)
        exported_segregating_sites += int(segregating and exported)
        if exported and site.ancestral_state != reference_sequence[position]:
            reference_mismatches.append(position)
        sites.append({
            'site_id': site.id,
            'position_zero_based': float(site.position),
            'vcf_position_one_based': position + 1 if exported else None,
            'ancestral_state': site.ancestral_state,
            'alleles': list(variant.alleles),
            'num_mutations': len(site.mutations),
            'is_segregating': segregating,
            'exported_to_vcf_and_fasta': exported,
        })
    trees = []
    for tree in ts.trees():
        trees.append({
            'index': tree.index,
            'left': float(tree.interval.left),
            'right': float(tree.interval.right),
            'num_sites': tree.num_sites,
            'num_mutations': tree.num_mutations,
            'num_roots': tree.num_roots,
            'root_times_generations': [float(ts.node(root).time) for root in tree.roots],
        })
    breakpoints = [float(position) for position in ts.breakpoints()][1:-1]
    metadata = {
        'schema_version': 1,
        'dataset_name': dataset_name,
        'replicate_index': int(replicate_index),
        'summary': {
            'sequence_length_bp': int(ts.sequence_length),
            'num_haplotypes': ts.num_samples,
            'num_individuals': ts.num_individuals,
            'num_sites': ts.num_sites,
            'num_segregating_sites': segregating_sites,
            'num_mutations': ts.num_mutations,
            'num_exported_sites': ts.num_sites - sum(bool(x) for x in site_mask),
            'num_exported_segregating_sites': exported_segregating_sites,
            'num_filtered_sites': sum(bool(x) for x in site_mask),
            'num_trees': ts.num_trees,
            'num_tree_breakpoints': len(breakpoints),
            'num_nodes': ts.num_nodes,
            'num_edges': ts.num_edges,
            'num_pairwise_coalescence_files': ts.num_samples * (ts.num_samples - 1) // 2,
            'time_units': ts.time_units,
        },
        'simulation': simulation,
        'coordinates': {
            'tree_sites_and_fasta_indices': 'Zero-based, local to the simulated window.',
            'vcf_positions': 'One-based, local to the simulated window: POS = tree site position + 1.',
            'tree_and_coalescence_intervals': 'Zero-based half-open [left, right).',
            'vcf_contig_id': str(contig_id),
        },
        'reference': {
            'fasta_path': str(Path(reference_fasta).resolve()),
            'download_url': reference_url,
            'contig': reference_contig,
            'start_zero_based': int(reference_start),
            'end_exclusive': int(reference_start + ts.sequence_length),
            'usage': 'Background for nonvariant FASTA bases. JC69 site ancestral states are simulated independently of this reference.',
            'vcf_ref_definition': 'Simulated ancestral allele; not necessarily the reference FASTA base.',
            'vcf_ref_mismatch_positions_zero_based': reference_mismatches,
        },
        'samples': [
            {
                'haplotype_index': index,
                'tree_node_id': node,
                'fasta_header': 'n' + str(node),
                'vcf_sample': 'spl' + str(index // 2),
                'vcf_phase_index': index % 2,
                'individual_id': int(ts.node(node).individual),
            }
            for index, node in enumerate(sample_nodes)
        ],
        'tree_breakpoints_zero_based': breakpoints,
        'breakpoint_note': 'Internal boundaries between marginal trees; not a count or complete record of historical recombination events.',
        'sites': sites,
        'trees': trees,
        'files': {
            'vcf': dataset_name + '.vcf',
            'fasta': dataset_name + '.fa',
            'ground_truth_trees': dataset_name + '.trees',
            'pairwise_coalescence_directory': 'tcoalmap',
            'pairwise_coalescence_filename_pattern': dataset_name + '_spls{i}-{j}.tc',
            'file_paths_relative_to': 'The directory containing this metadata.json.',
        },
    }
    with metadata_path.open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return metadata


def simulate(
    nrep,
    n,
    dataset_name,
    mu=2e-8,
    rec=2e-8,
    Ne=10000,
    length=100 * 10**6,
    seed=seed,
    contig_id=contig_id,
    output_dir=VALIDATION_DIR / 'datasets',
    reference_fasta=HG38_FASTA,
    reference_url=HG38_URL,
    reference_contig=HG38_CONTIG,
    reference_start=HG38_START,
):
    """Write each replicate's files under output_dir/dataset_name/rep<i>/."""
    reference_sequence = read_reference_window(
        length, contig=reference_contig, start=reference_start,
        reference_fasta=reference_fasta, reference_url=reference_url,
    )
    dataset_dir = Path(output_dir) / dataset_name
    for i in range(nrep):
        print('rep', i)
        rep_dir = dataset_dir / ('rep' + str(i))
        coalescence_dir = rep_dir / 'tcoalmap'
        coalescence_dir.mkdir(parents=True, exist_ok=True)
        ancestry_seed = seed + i * 100_000
        ancestry_ts = msprime.sim_ancestry(
            samples=n // 2,
            ploidy=2,
            population_size=Ne,
            sequence_length=length,
            recombination_rate=rec,
            discrete_genome=True,
            random_seed=ancestry_seed,
        )
        ts = msprime.sim_mutations(
            ancestry_ts,
            rate=mu,
            model=msprime.JC69(),
            discrete_genome=True,
            keep=False,
            random_seed=ancestry_seed + 1,
        )
        site_mask = vcf_site_mask(ts)
        sample_ids = list(ts.samples())
        vcfpath = rep_dir / (dataset_name + '.vcf')
        write_vcf(ts, vcfpath, contig_id=contig_id, site_mask=site_mask)
        print('writing vcf to', vcfpath)
        tsfile = rep_dir / (dataset_name + '.trees')
        ts.dump(tsfile)
        print('writing trees to', tsfile)
        fastafile = rep_dir / (dataset_name + '.fa')
        write_haplotype_fasta(
            ts, fastafile, site_mask=site_mask, reference_sequence=reference_sequence
        )
        print('writing fasta to', fastafile)
        for s1 in range(0, n - 1):
            for s2 in range(s1 + 1, n):
                tcpath = coalescence_dir / (
                    dataset_name + '_spls' + str(s1) + '-' + str(s2) + '.tc'
                )
                with open(tcpath, 'w', encoding='utf-8') as coaltimefh:
                    for tree in ts.trees():
                        left, right = tree.interval
                        coalescence_time = tree.tmrca(sample_ids[s1], sample_ids[s2])
                        print(left, right, coalescence_time, sep='\t', file=coaltimefh)
        metadata_path = rep_dir / 'metadata.json'
        write_metadata(
            ts, metadata_path, dataset_name=dataset_name, replicate_index=i,
            contig_id=contig_id, reference_fasta=reference_fasta,
            reference_contig=reference_contig, reference_start=reference_start,
            reference_sequence=reference_sequence, reference_url=reference_url,
            site_mask=site_mask,
        )
        print('writing metadata to', metadata_path)


_CONFIG_DEFAULTS = {
    'dataset_name': 'sim_2k_super_easy_human',
    'num_replicates': 1,
    'num_samples': 8,
    'population_size': 10000,
    'mutation_rate': 1.29e-8,
    'recombination_rate': 1.253e-8,
    'sequence_length': 2000,
    'seed': 42,
    'contig_id': '1',
    'output_dir': '../datasets',
    'reference_fasta': '../reference/hg38.fa.gz',
    'reference_url': HG38_URL,
    'reference_contig': HG38_CONTIG,
    'reference_start': HG38_START,
}


def load_config(config_path):
    """Validate simulation settings and resolve paths relative to the YAML file."""
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open(encoding='utf-8') as handle:
        supplied = yaml.safe_load(handle)
    if not isinstance(supplied, dict):
        raise ValueError('simulation config must be a YAML mapping')
    unknown = supplied.keys() - _CONFIG_DEFAULTS.keys()
    if unknown:
        raise ValueError('unknown simulation settings: ' + ', '.join(sorted(map(str, unknown))))
    config = {**_CONFIG_DEFAULTS, **supplied}
    for key, minimum in (
        ('num_replicates', 1), ('num_samples', 2), ('sequence_length', 1),
        ('seed', 1), ('reference_start', 0),
    ):
        value = config[key]
        if type(value) is not int or value < minimum:
            raise ValueError('{} must be an integer >= {}'.format(key, minimum))
    if config['num_samples'] % 2:
        raise ValueError('num_samples must be even (two haplotypes per diploid individual)')
    if config['seed'] + (config['num_replicates'] - 1) * 100_000 + 1 >= 2**32:
        raise ValueError('seed and num_replicates exceed the msprime random seed range')
    for key in ('population_size', 'mutation_rate', 'recombination_rate'):
        try:
            value = float(config[key])
        except (TypeError, ValueError, OverflowError):
            raise ValueError('{} must be a finite number'.format(key)) from None
        minimum_ok = value > 0 if key == 'population_size' else value >= 0
        if isinstance(config[key], bool) or not math.isfinite(value) or not minimum_ok:
            bound = 'positive' if key == 'population_size' else 'non-negative'
            raise ValueError('{} must be finite and {}'.format(key, bound))
        config[key] = value
    name = config['dataset_name']
    if (not isinstance(name, str) or not name.strip()
            or name in ('.', '..') or '/' in name or '\\' in name):
        raise ValueError('dataset_name must be a non-empty filename stem without directories')
    for key in ('contig_id', 'reference_contig'):
        value = config[key]
        if (isinstance(value, bool) or not isinstance(value, (str, int))
                or not str(value) or any(char.isspace() for char in str(value))):
            raise ValueError('{} must be a non-empty contig identifier'.format(key))
        config[key] = str(value)
    if config['reference_url'] is not None and (
        not isinstance(config['reference_url'], str) or not config['reference_url'].strip()
    ):
        raise ValueError('reference_url must be a URL string or null to disable downloads')
    for key in ('output_dir', 'reference_fasta'):
        value = config[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError('{} must be a non-empty path'.format(key))
        config[key] = (config_path.parent / Path(value).expanduser()).resolve()
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Simulate diploid ancestry and JC69 mutations from a YAML configuration.'
    )
    parser.add_argument(
        '--config', type=Path, default=DEFAULT_CONFIG,
        help='YAML settings (default: validation/config/simulate_config.yaml)',
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    print('simulating', config['dataset_name'])
    simulate(
        nrep=config['num_replicates'], n=config['num_samples'],
        dataset_name=config['dataset_name'], output_dir=config['output_dir'],
        mu=config['mutation_rate'], rec=config['recombination_rate'],
        Ne=config['population_size'], length=config['sequence_length'],
        seed=config['seed'], contig_id=config['contig_id'],
        reference_fasta=config['reference_fasta'],
        reference_url=config['reference_url'], reference_contig=config['reference_contig'],
        reference_start=config['reference_start'],
    )


if __name__ == '__main__':
    main()
