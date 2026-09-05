"""Simulate ancestry and JC69 mutations for ARG validation datasets."""

import argparse
from pathlib import Path

import msprime
import numpy as np


VALIDATION_DIR = Path(__file__).resolve().parent.parent
DEFAULT_NREP = 1
DEFAULT_PREFIX = "poc_easy"
DEFAULT_HAPLOTYPES = 8
DEFAULT_LENGTH = 10_000
DEFAULT_POPULATION_SIZE = 10_000
DEFAULT_MUTATION_RATE = 1e-7
DEFAULT_RECOMBINATION_RATE = 1e-8
DEFAULT_SEED = 42
DEFAULT_CONTIG_ID = "1"

_ACGT = frozenset("ACGT")
_ALPHABET = np.asarray(list("ACGT"))
_ALT_BASE = {"A": "T", "T": "A", "C": "G", "G": "C"}


def synthetic_reference(length, seed):
    """Return a reproducible, uniformly sampled A/C/G/T reference sequence."""
    length = int(length)
    if length < 0:
        raise ValueError("reference length must be non-negative")
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(_ALPHABET, size=length))


def _allele_to_base(allele, reference_base):
    if allele is None:
        return "N"
    base = str(allele).upper()
    if len(base) == 1 and base in _ACGT:
        return base
    if base == "0":
        return reference_base if reference_base in _ACGT else "N"
    if base == "1":
        return _ALT_BASE.get(reference_base, "N")
    return "N"


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
                len(str(allele).upper()) != 1
                or str(allele).upper() not in _ACGT
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


def write_haplotype_fasta(ts, fasta_path, site_mask=None, reference_seed=0):
    sequence_length = int(ts.sequence_length)
    if sequence_length != float(ts.sequence_length):
        raise ValueError("FASTA export requires integer sequence length")

    reference = synthetic_reference(sequence_length, reference_seed)
    seqs = [bytearray(reference.encode("ascii")) for _ in range(ts.num_samples)]
    for variant in ts.variants():
        if site_mask is not None and site_mask[variant.site.id]:
            continue
        _apply_variant(seqs, reference, variant)

    fasta_path = Path(fasta_path)
    with fasta_path.open("w", encoding="utf-8") as handle:
        for sample_idx, seq in enumerate(seqs):
            handle.write(">hap{:03d}\n".format(sample_idx))
            text = seq.decode("ascii")
            for start in range(0, len(text), 80):
                handle.write(text[start : start + 80] + "\n")


def _apply_variant(sequences, reference, variant):
    position = int(variant.site.position)
    if not 0 <= position < len(reference):
        return
    alleles = variant.alleles
    if not alleles:
        for sequence in sequences:
            sequence[position] = ord("N")
        return
    bases = [_allele_to_base(allele, reference[position].upper()) for allele in alleles]
    for sample, allele_index in enumerate(variant.genotypes):
        base = bases[int(allele_index)] if 0 <= allele_index < len(bases) else "N"
        sequences[sample][position] = ord(base if base in _ACGT else "N")


def _write_pair_times(ts, sample_ids, output_dir, output_name, sample_count):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for first in range(sample_count - 1):
        for second in range(first + 1, sample_count):
            path = output_dir / f"{output_name}_spls{first}-{second}.tc"
            with path.open("w", encoding="utf-8") as handle:
                for tree in ts.trees():
                    left, right = tree.interval
                    time = tree.tmrca(sample_ids[first], sample_ids[second])
                    print(left, right, time, sep="\t", file=handle)


def _normalise_prefix(prefix):
    prefix = str(prefix).strip().rstrip("_")
    if not prefix:
        raise ValueError("prefix must not be empty")
    if Path(prefix).name != prefix:
        raise ValueError("prefix must be a filename component, not a path")
    return prefix


def _validate_parameters(nrep, n, mu, rec, population_size, length, seed):
    if nrep <= 0:
        raise ValueError("replicate count must be positive")
    if n <= 0 or n % 2:
        raise ValueError("haplotype count must be a positive even number")
    if length <= 0:
        raise ValueError("sequence length must be positive")
    if population_size <= 0:
        raise ValueError("population size must be positive")
    if mu < 0 or rec < 0:
        raise ValueError("mutation and recombination rates cannot be negative")
    final_seed = seed + (nrep - 1) * 100_000 + 1
    if seed <= 0 or final_seed >= 2**32:
        raise ValueError("seed range must fit msprime's positive 32-bit seed limit")


def simulate(
    nrep,
    pref,
    n,
    mu=DEFAULT_MUTATION_RATE,
    rec=DEFAULT_RECOMBINATION_RATE,
    Ne=DEFAULT_POPULATION_SIZE,
    length=DEFAULT_LENGTH,
    seed=DEFAULT_SEED,
    contig_id=DEFAULT_CONTIG_ID,
    vcfdir=VALIDATION_DIR / "vcf",
    tcdir=VALIDATION_DIR / "tcoalmsp",
    tsdir=VALIDATION_DIR / "trees",
    fastadir=VALIDATION_DIR / "fasta",
):
    """Generate one or more independently seeded simulation replicates."""
    _validate_parameters(nrep, n, mu, rec, Ne, length, seed)
    prefix = _normalise_prefix(pref)
    vcfdir = Path(vcfdir)
    tcdir = Path(tcdir)
    tsdir = Path(tsdir)
    fastadir = Path(fastadir)
    for directory in (vcfdir, tsdir, fastadir):
        directory.mkdir(parents=True, exist_ok=True)
    for index in range(nrep):
        _simulate_replicate(
            index,
            prefix,
            n,
            mu,
            rec,
            Ne,
            length,
            seed,
            contig_id,
            vcfdir,
            tcdir,
            tsdir,
            fastadir,
        )


def _simulate_replicate(
    index,
    prefix,
    n,
    mu,
    rec,
    population_size,
    length,
    base_seed,
    contig,
    vcf_dir,
    time_dir,
    tree_dir,
    fasta_dir,
):
    print("rep", index)
    ancestry_seed = base_seed + index * 100_000
    ancestry = msprime.sim_ancestry(
        samples=n // 2,
        ploidy=2,
        population_size=population_size,
        sequence_length=length,
        recombination_rate=rec,
        discrete_genome=True,
        random_seed=ancestry_seed,
    )
    ts = msprime.sim_mutations(
        ancestry,
        rate=mu,
        model=msprime.JC69(),
        discrete_genome=True,
        keep=False,
        random_seed=ancestry_seed + 1,
    )
    mask = vcf_site_mask(ts)
    output_name = f"sim_{prefix}_{index}"
    vcf_path = vcf_dir / f"{output_name}.vcf"
    vcf_options = {
        "contig_id": contig,
        "individual_names": [f"spl{sample}" for sample in range(n // 2)],
        "site_mask": np.asarray(mask, dtype=bool),
    }
    if ts.num_individuals == 0:
        vcf_options["ploidy"] = 2
    with vcf_path.open("w", encoding="utf-8") as handle:
        ts.write_vcf(handle, **vcf_options)

    tree_path = tree_dir / f"{output_name}.trees"
    fasta_path = fasta_dir / f"{output_name}.fa"
    ts.dump(tree_path)
    write_haplotype_fasta(
        ts, fasta_path, site_mask=mask, reference_seed=ancestry_seed + 2
    )
    print("writing vcf to", vcf_path)
    print("writing trees to", tree_path)
    print("writing fasta to", fasta_path)
    _write_pair_times(
        ts,
        list(ts.samples()),
        time_dir / f"rep{index}",
        output_name,
        n,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Simulate an msprime ARG dataset with JC69 mutations."
    )
    parser.add_argument("--replicates", type=int, default=DEFAULT_NREP)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--haplotypes", type=int, default=DEFAULT_HAPLOTYPES)
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument(
        "--population-size", type=float, default=DEFAULT_POPULATION_SIZE
    )
    parser.add_argument("--mutation-rate", type=float, default=DEFAULT_MUTATION_RATE)
    parser.add_argument(
        "--recombination-rate", type=float, default=DEFAULT_RECOMBINATION_RATE
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--contig-id", default=DEFAULT_CONTIG_ID)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=VALIDATION_DIR,
        help="Root containing vcf/, trees/, fasta/, and tcoalmsp/ outputs.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    theta = 4 * args.population_size * args.mutation_rate * args.length
    rho = 4 * args.population_size * args.recombination_rate * args.length
    print(
        "simulation parameters:",
        f"Ne={args.population_size:g}",
        f"n={args.haplotypes}",
        f"L={args.length}",
        f"mu={args.mutation_rate:g}",
        f"r={args.recombination_rate:g}",
        f"theta={theta:g}",
        f"rho={rho:g}",
    )
    try:
        simulate(
            nrep=args.replicates,
            pref=args.prefix,
            n=args.haplotypes,
            mu=args.mutation_rate,
            rec=args.recombination_rate,
            Ne=args.population_size,
            length=args.length,
            seed=args.seed,
            contig_id=args.contig_id,
            vcfdir=args.output_root / "vcf",
            tcdir=args.output_root / "tcoalmsp",
            tsdir=args.output_root / "trees",
            fastadir=args.output_root / "fasta",
        )
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
