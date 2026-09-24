"""Strict, lossless ingestion of explicitly annotated real SNP observations."""
import csv
import gzip
import hashlib
from pathlib import Path

import numpy as np

from .snp_data import SNPData, _integer


def load_real_snp_dataset(directory, metadata):
    directory = Path(directory)
    if (type(metadata.get('schema_version')) is not int or metadata['schema_version'] != 2
            or metadata.get('observation_model') != 'polarized_biallelic_snps'
            or metadata.get('inference_model') != 'infinite_sites'):
        raise ValueError('expected schema-v2 polarized real SNP observations for infinite-sites inference')
    params, files = metadata.get('parameters', {}), metadata.get('files', {})
    origin = _integer(metadata.get('genomic_start_zero_based'), 'genomic_start_zero_based')
    length = _integer(params.get('sequence_length'), 'sequence_length', 1)
    contig = str(params.get('contig_id', ''))
    samples = metadata.get('sample_ids')
    if (not isinstance(samples, list) or not samples or any(not isinstance(s, str) or not s for s in samples)
            or len(set(samples)) != len(samples)):
        raise ValueError('sample_ids must specify unique individual IDs in VCF order')
    paths = {}
    hashes = metadata.get('input_sha256', {})
    for key in ('vcf', 'sites', 'observation_intervals'):
        name = files.get(key)
        if not isinstance(name, str) or not name:
            raise ValueError(f'real-data metadata requires files.{key}')
        path = directory / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get(key):
            raise ValueError(f'real-data input checksum mismatch: {key}')
        paths[key] = path
    with paths['sites'].open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    by_position = {}
    for row in rows:
        pos = _integer(row.get('position_1based'), 'site position', 1)
        if pos in by_position or row.get('chrom') != contig:
            raise ValueError('sites table requires unique positions on the declared contig')
        if _integer(row.get('local_position_0based'), 'local position') != pos - origin - 1:
            raise ValueError('sites table coordinate transform disagrees with metadata')
        if row.get('polarizable') != '1':
            raise ValueError('all real SNPs require an explicit inferred ancestral allele; no silent filtering')
        by_position[pos] = row
    intervals = []
    for line in paths['observation_intervals'].read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) != 3 or fields[0] != contig:
            raise ValueError('observation BED must have three columns on the declared contig')
        intervals.append((_integer(fields[1], 'observation left'), _integer(fields[2], 'observation right')))

    records, seen, names = [], set(), None
    opener = gzip.open if paths['vcf'].suffix == '.gz' else open
    with opener(paths['vcf'], 'rt') as handle:
        for line in handle:
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                if names is not None:
                    raise ValueError('duplicate VCF header')
                names = line.rstrip().split('\t')[9:]
                if names != samples:
                    raise ValueError('VCF sample order differs from metadata')
                continue
            if not line.strip() or line.startswith('#'):
                continue
            f = line.rstrip().split('\t')
            if names is None or len(f) != 9 + len(samples) or f[0] != contig:
                raise ValueError('malformed real VCF record or contig mismatch')
            pos = _integer(f[1], 'VCF position', 1)
            if pos in seen or pos not in by_position:
                raise ValueError('VCF and sites table must match exactly without duplicate positions')
            if not origin < pos <= origin + length:
                raise ValueError('VCF position outside declared genomic window')
            ref, alt = f[3:5]
            row = by_position[pos]
            aa = row['ancestral_allele']
            if (ref not in ('A', 'C', 'G', 'T') or alt not in ('A', 'C', 'G', 'T') or ref == alt
                    or (ref, alt) != (row['REF'], row['ALT']) or aa not in (ref, alt)):
                raise ValueError('real SNP alleles or ancestral annotation disagree')
            info = dict(item.split('=', 1) for item in f[7].split(';') if '=' in item)
            if info.get('AA', aa) != aa:
                raise ValueError('VCF AA disagrees with sites annotation')
            if f[6] not in ('.', 'PASS'):
                raise ValueError('filtered real VCF record; loader never silently drops records')
            formats = f[8].split(':')
            if formats.count('GT') != 1:
                raise ValueError('real VCF requires GT')
            index, calls = formats.index('GT'), []
            for value in f[9:]:
                parts = value.split(':')
                gt = parts[index] if index < len(parts) else ''
                if gt not in ('0|0', '0|1', '1|0', '1|1'):
                    raise ValueError('complete phased diploid 0/1 genotypes required; no imputation')
                # Bijective relabeling; original nucleotide calls are preserved.
                calls.extend(int(x) ^ int(aa == alt) for x in gt.split('|'))
            records.append((pos-origin-1, _integer(row['site_id'], 'site ID'), aa,
                            alt if aa == ref else ref, calls))
            seen.add(pos)
    if names is None or seen != by_position.keys():
        raise ValueError('VCF and annotated sites must contain exactly the same observations')
    data = SNPData(
        genotypes=np.array([r[4] for r in records], dtype=np.uint8).T if records else np.empty((2*len(samples), 0)),
        positions=[r[0] for r in records], sequence_length=length,
        site_ids=tuple(r[1] for r in records), ancestral_states=tuple(r[2] for r in records),
        derived_states=tuple(r[3] for r in records),
        haplotype_ids=tuple(f'{s}_{phase}' for s in samples for phase in range(2)),
        contig_id=contig, observation_intervals=tuple(intervals),
    )
    for key, value in (('num_haplotypes', data.num_haplotypes), ('num_sites', data.num_variants)):
        if _integer(metadata.get(key), key) != value:
            raise ValueError(f'real-data {key} disagrees with observations')
    if _integer(params.get('num_samples'), 'num_samples') != data.num_haplotypes:
        raise ValueError('num_samples must count haplotypes')
    return data
