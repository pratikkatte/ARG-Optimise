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
    output_name=None,
):
    if output_name is not None and nrep != 1:
        raise ValueError('output_name requires exactly one replicate')
    os.makedirs(vcfdir, exist_ok=True)
    os.makedirs(tsdir, exist_ok=True)
    os.makedirs(fastadir, exist_ok=True)
    for i in range(nrep):
        print('rep', i)
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
        outname = output_name or 'sim_' + pref + str(i)
        vcfpath = vcfdir + outname + '.vcf'
        write_vcf(ts, vcfpath, contig_id=contig_id, site_mask=site_mask)
        print('writing vcf to', vcfpath)
        tsfile = tsdir + outname + '.trees'
        ts.dump(tsfile)
        print('writing trees to', tsfile)
        fastafile = fastadir + outname + '.fa'
        write_haplotype_fasta(ts, fastafile, site_mask=site_mask)
        print('writing fasta to', fastafile)
        rep_dir = tcdir + 'rep' + str(i) + '/'
        os.makedirs(rep_dir, exist_ok=True)
        for s1 in range(0, n - 1):
            for s2 in range(s1 + 1, n):
                tcpath = rep_dir + outname + '_spls' + str(s1) + '-' + str(s2) + '.tc'
                with open(tcpath, 'w', encoding='utf-8') as coaltimefh:
                    for tree in ts.trees():
                        left, right = tree.interval
                        coalescence_time = tree.tmrca(sample_ids[s1], sample_ids[s2])
                        print(left, right, coalescence_time, sep='\t', file=coaltimefh)


## standard simulation — 1 Mb
# coalescent model
# 4 diploids
# Ne 10000
# recombination 2e-8
# mutation 2e-8 (JC69)
# length 1mb
# replicates 1
# print('1mb sequence length')
# simulate(nrep=1, pref='l1mb_', n=8, rec=2e-8, Ne=10000, length=10**6)

## 2 kb with human mutation and recombination rates (per bp per generation)
if __name__ == '__main__':
    print('2kb sequence length, human rates')
    simulate(
        nrep=1, pref='2k_super_easy_human_', n=8,
        mu=1.29e-8, rec=1.253e-8, Ne=10000, length=2000, seed=42,
        output_name='sim_2k_super_easy_human',
    )

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
# print('25kb sequence length')
# simulate(nrep=1, pref='l25kb_', n=8, rec=2e-8, Ne=10000, length=25_000)
