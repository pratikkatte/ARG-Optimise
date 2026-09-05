## ancestry + JC69 mutations (modern msprime; replaces 1_msprime_sim + 2_msprimefinitesites)
import gzip
import os
import urllib.request

import msprime
import numpy as np

vcfdir = '../vcf/'
tcdir = '../tcoalmsp/'
tsdir = '../trees/'
fastadir = '../fasta/'
refdir = '../reference/'
nrep = 1
contig_id = '1'
seed = 42

HG38_URL = 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz'
HG38_FASTA = refdir + 'hg38.fa.gz'
HG38_CONTIG = 'chr1'
HG38_START = 10_000_000
_ACGT = frozenset('ACGT')
_ALT_BASE = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}


def ensure_hg38_reference():
    os.makedirs(refdir, exist_ok=True)
    if os.path.exists(HG38_FASTA) and os.path.getsize(HG38_FASTA) > 0:
        return HG38_FASTA

    tmp_path = HG38_FASTA + '.tmp'
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print('downloading hg38 reference from', HG38_URL)
    print('writing reference to', HG38_FASTA)
    with urllib.request.urlopen(HG38_URL) as response, open(tmp_path, 'wb') as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp_path, HG38_FASTA)
    return HG38_FASTA


def read_reference_window(length, contig=HG38_CONTIG, start=HG38_START):
    length = int(length)
    start = int(start)
    if length < 0:
        raise ValueError('reference window length must be non-negative')
    if start < 0:
        raise ValueError('reference window start must be non-negative')
    if length == 0:
        return ''

    ensure_hg38_reference()
    end = start + length
    seq_parts = []
    collected = 0
    pos = 0
    in_contig = False
    found_contig = False

    with gzip.open(HG38_FASTA, 'rt') as handle:
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
        raise ValueError('contig {} not found in {}'.format(contig, HG38_FASTA))
    if collected != length:
        raise ValueError(
            'reference window {}:{}-{} is shorter than requested length {}'.format(
                contig, start, end, length
            )
        )
    return ''.join(seq_parts)


def _allele_to_base(allele, reference_base):
    if allele is None:
        return 'N'
    base = str(allele).upper()
    if len(base) == 1 and base in _ACGT:
        return base
    if base == '0':
        return reference_base if reference_base in _ACGT else 'N'
    if base == '1':
        return _ALT_BASE.get(reference_base, 'N')
    return 'N'


def vcf_site_mask(ts):
    mask = [False] * ts.num_sites
    seen_positions = set()
    for variant in ts.variants():
        site = variant.site
        position = float(site.position)
        int_position = int(position)
        bad = (
            position != int_position
            or int_position <= 0
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


def write_haplotype_fasta(ts, fasta_path, site_mask=None):
    sequence_length = int(ts.sequence_length)
    if sequence_length != float(ts.sequence_length):
        raise ValueError('FASTA export requires integer sequence length')

    reference = read_reference_window(sequence_length)
    seqs = [bytearray(reference.encode('ascii')) for _ in range(ts.num_samples)]
    for variant in ts.variants():
        if site_mask is not None and site_mask[variant.site.id]:
            continue
        _apply_variant(seqs, reference, variant)

    with open(fasta_path, 'w', encoding='utf-8') as handle:
        for sample_idx, seq in enumerate(seqs):
            handle.write('>hap{:03d}\n'.format(sample_idx))
            text = seq.decode('ascii')
            for start in range(0, len(text), 80):
                handle.write(text[start : start + 80] + '\n')


def _apply_variant(sequences, reference, variant):
    position = int(variant.site.position)
    if not 0 <= position < len(reference):
        return
    alleles = variant.alleles
    if not alleles:
        for sequence in sequences:
            sequence[position] = ord('N')
        return
    bases = [_allele_to_base(allele, reference[position].upper()) for allele in alleles]
    for sample, allele_index in enumerate(variant.genotypes):
        base = bases[int(allele_index)] if 0 <= allele_index < len(bases) else 'N'
        sequences[sample][position] = ord(base if base in _ACGT else 'N')


def _write_pair_times(ts, sample_ids, output_dir, output_name, sample_count):
    os.makedirs(output_dir, exist_ok=True)
    for first in range(sample_count - 1):
        for second in range(first + 1, sample_count):
            path = output_dir + output_name + f'_spls{first}-{second}.tc'
            with open(path, 'w', encoding='utf-8') as handle:
                for tree in ts.trees():
                    left, right = tree.interval
                    time = tree.tmrca(sample_ids[first], sample_ids[second])
                    print(left, right, time, sep='\t', file=handle)


def simulate(
    nrep,
    pref,
    n,
    mu=2e-8,
    rec=2e-8,
    Ne=10000,
    length=100 * 10**6,
    seed=seed,
    contig_id=contig_id,
    vcfdir=vcfdir,
    tcdir=tcdir,
    tsdir=tsdir,
    fastadir=fastadir,
):
    os.makedirs(vcfdir, exist_ok=True)
    os.makedirs(tsdir, exist_ok=True)
    os.makedirs(fastadir, exist_ok=True)
    for i in range(nrep):
        _simulate_replicate(
            i, pref, n, mu, rec, Ne, length, seed,
            contig_id, vcfdir, tcdir, tsdir, fastadir,
        )


def _simulate_replicate(
    index, pref, n, mu, rec, population_size, length, base_seed,
    contig, vcf_dir, time_dir, tree_dir, fasta_dir,
):
    print('rep', index)
    ancestry_seed = base_seed + index * 100_000
    ancestry = msprime.sim_ancestry(
        samples=n // 2, ploidy=2, population_size=population_size,
        sequence_length=length, recombination_rate=rec, discrete_genome=True,
        random_seed=ancestry_seed,
    )
    ts = msprime.sim_mutations(
        ancestry, rate=mu, model=msprime.JC69(), discrete_genome=True,
        keep=False, random_seed=ancestry_seed + 1,
    )
    mask = vcf_site_mask(ts)
    output_name = f'sim_{pref}{index}'
    vcf_path = vcf_dir + output_name + '.vcf'
    vcf_options = {
        'contig_id': contig,
        'individual_names': [f'spl{sample}' for sample in range(n // 2)],
        'site_mask': np.asarray(mask, dtype=bool),
    }
    if ts.num_individuals == 0:
        vcf_options['ploidy'] = 2
    with open(vcf_path, 'w', encoding='utf-8') as handle:
        ts.write_vcf(handle, **vcf_options)
    tree_path = tree_dir + output_name + '.trees'
    fasta_path = fasta_dir + output_name + '.fa'
    ts.dump(tree_path)
    write_haplotype_fasta(ts, fasta_path, site_mask=mask)
    print('writing vcf to', vcf_path)
    print('writing trees to', tree_path)
    print('writing fasta to', fasta_path)
    _write_pair_times(
        ts, list(ts.samples()), time_dir + f'rep{index}/', output_name, n,
    )


## standard simulation — 1 Mb
# coalescent model
# 4 diploids
# Ne 10000
# recombination 2e-8
# mutation 2e-8 (JC69)
# length 1mb
# replicates 1
# Run from ``main`` below.

## change number of samples
# coalescent model
# 40 diploids           *
# Ne 10000
# recombination 2e-8
# mutation 2e-8
# length 1mb
# replicates 10
# print('change number of samples')
# simulate(20, pref='n200_', n=200, rec=2e-8, Ne=10000, length=5*10**6)
# print('80 samples')
# simulate(nrep, pref='n80_', n=80, rec=2e-8, Ne=10000, length=10**6)
# print('32 samples')
# simulate(nrep, pref='n32_', n=32, rec=2e-8, Ne=10000, length=10**6)
# print('16 samples')
# simulate(nrep, pref='n16_', n=16, rec=2e-8, Ne=10000, length=10**6)
# print('4 samples')
# simulate(nrep, pref='n4_', n=4, rec=2e-8, Ne=10000, length=10**6)
# print('2 samples')
# simulate(nrep, pref='n2_', n=2, rec=2e-8, Ne=10000, length=10**6)

## /10 recombination
# coalescent model
# 4 diploids
# Ne 10000
# recombination 2e-9    *
# mutation 2e-8
# length 1mb
# replicates 1
# print('10x lower recombination')
# simulate(nrep, pref='r2e9_', n=8, rec=2e-9, Ne=10000, length=10**6)
# print('10x higher recombination')
# simulate(nrep, pref='r2e7_', n=8, rec=2e-7, Ne=10000, length=10**6)

## *10 mutation
# coalescent model
# 4 diploids
# Ne 10000
# recombination 2e-8
# mutation 2e-7         *
# length 1mb
# replicates 1
# print('10x mutation')
# simulate(nrep, pref='m2e7_', n=8, mu=2e-7, Ne=10000, length=10**6)
# print('/10 mutation')
# simulate(nrep, pref='m2e9_', n=8, mu=2e-9, Ne=10000, length=10**6)

# mut/rec ratio = 2,4
# print('2x mut/rec ratio')
# simulate(1, pref='m4e8_', n=8, mu=4e-8, Ne=10000, length=10**6)
# print('4x mut/rec ratio')
# simulate(1, pref='m8e8_', n=8, mu=8e-8, Ne=10000, length=10**6)

## shorter input sequences
# coalescent model
# 4 diploids
# Ne 10000
# recombination 2e-8
# mutation 2e-8
# print('5mb sequence length')
# simulate(nrep=1, pref='l5mb_', n=8, rec=2e-8, Ne=10000, length=5*10**6)
# print('250kb sequence length')
# simulate(nrep=1, pref='l250kb_', n=8, rec=2e-8, Ne=10000, length=250*10**3)
def main():
    print('1mb sequence length')
    simulate(nrep=1, pref='l1mb_', n=8, rec=2e-8, Ne=10000, length=10**6)
    print('25kb sequence length')
    simulate(nrep=1, pref='l25kb_', n=8, rec=2e-8, Ne=10000, length=25_000)


if __name__ == '__main__':
    main()
