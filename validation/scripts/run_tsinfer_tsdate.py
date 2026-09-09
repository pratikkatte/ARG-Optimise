#!/usr/bin/env python3
"""Infer and date a dataset supplied as .trees, aligned FASTA, or .samples.

Defaults match sim_2k_super_easy_0. FASTA records are haplotypes, in sample
order; the most frequent allele is assumed ancestral (alphabetical tie break).
Tree-sequence input contributes genotypes, ancestral alleles, and sample times,
never its ancestral topology or mutation times.
"""

import argparse
from collections import Counter
from contextlib import closing
import math
from pathlib import Path


def read_fasta(path):
    names, sequences = [], []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                names.append(line[1:].strip())
                sequences.append("")
            elif sequences:
                sequences[-1] += line.upper()
            else:
                raise ValueError("FASTA sequence encountered before a header")
    if len(sequences) < 2 or not sequences[0]:
        raise ValueError("FASTA must contain at least two nonempty haplotypes")
    if len({len(seq) for seq in sequences}) != 1:
        raise ValueError("FASTA must be aligned: all sequences need equal length")
    if any(set(seq) - set("ACGTN?-") for seq in sequences):
        raise ValueError("FASTA supports A/C/G/T and N/?/- for missing bases")
    return names, sequences


def load_samples(path, tsinfer, tskit):
    if path.suffix == ".samples":
        return tsinfer.SampleData.load(str(path))
    if path.suffix == ".trees":
        source = tskit.load(str(path))
        with tsinfer.SampleData(sequence_length=source.sequence_length) as samples:
            for node in source.samples():
                samples.add_individual(ploidy=1, time=source.node(node).time)
            for variant in source.variants():
                samples.add_site(
                    variant.site.position,
                    genotypes=variant.genotypes,
                    alleles=variant.alleles,
                    ancestral_allele=variant.alleles.index(variant.site.ancestral_state),
                )
        return samples
    if path.suffix.lower() not in {".fa", ".fasta", ".fna"}:
        raise ValueError("Supported inputs: .trees, .samples, .fa, .fasta, .fna")
    names, sequences = read_fasta(path)
    print("FASTA: assuming the most frequent allele is ancestral; N/?/- are missing.")
    with tsinfer.SampleData(sequence_length=len(sequences[0])) as samples:
        for name in names:
            samples.add_individual(ploidy=1, metadata={"name": name})
        for position, column in enumerate(zip(*sequences)):
            counts = Counter(base for base in column if base in "ACGT")
            if len(counts) < 2:
                continue
            alleles = sorted(counts, key=lambda base: (-counts[base], base))
            lookup = {base: index for index, base in enumerate(alleles)}
            samples.add_site(
                position,
                genotypes=[lookup.get(base, tskit.MISSING_DATA) for base in column],
                alleles=alleles,
                ancestral_allele=0,
            )
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Input dataset path")
    parser.add_argument("--output-dir", type=Path, default=Path("validation/output/tsinfer"))
    parser.add_argument("--mutation-rate", type=float, default=1e-7)
    parser.add_argument("--recombination-rate", type=float, default=1e-8)
    parser.add_argument("--ne", type=float, default=10000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.dataset.is_file():
        parser.error(f"Dataset does not exist: {args.dataset}")
    for name in ("mutation_rate", "recombination_rate", "ne"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    inferred_path = args.output_dir / f"{args.dataset.stem}_inferred.trees"
    dated_path = args.output_dir / f"{args.dataset.stem}_dated.trees"
    for path in (inferred_path, dated_path):
        if path.resolve() == args.dataset.resolve():
            parser.error("Output path must differ from input")
        if path.exists() and not args.overwrite:
            parser.error(f"Output exists: {path}; use --overwrite to replace it")
    try:
        import tskit
        import tsinfer
        import tsdate
    except (ImportError, RuntimeError) as error:
        parser.exit(1, f"Dependency error: {error}\nInstall with: "
                    "python -m pip install tsinfer tsdate 'zarr<3'\n")

    try:
        samples = load_samples(args.dataset, tsinfer, tskit)
    except ValueError as error:
        parser.error(str(error))
    with closing(samples):
        if samples.num_sites == 0:
            parser.error("Dataset has no variant sites to infer")
        print(f"Loaded {samples.num_samples} haplotypes, {samples.num_sites} sites, "
              f"{samples.sequence_length:g} bp", flush=True)
        print(f"mu={args.mutation_rate:g}, r={args.recombination_rate:g}, "
              f"Ne={args.ne:g}", flush=True)
        inferred = tsinfer.infer(
            samples,
            recombination_rate=args.recombination_rate,
            mismatch_ratio=1.0,
            num_threads=args.threads,
            progress_monitor=True,
        ).simplify(keep_unary=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inferred.dump(str(inferred_path))
    print(f"Wrote {inferred_path}", flush=True)
    dated = tsdate.date(
        inferred,
        mutation_rate=args.mutation_rate,
        population_size=args.ne,
        method="inside_outside",
    )
    dated.dump(str(dated_path))
    print(f"Wrote {dated_path} ({dated.num_trees} trees)", flush=True)


if __name__ == "__main__":
    main()
