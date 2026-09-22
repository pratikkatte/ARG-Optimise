#!/usr/bin/env python3
"""Convert this project's simulated, phased VCFs to ARGinfer nucleotide inputs.

Uses REF as the known ancestral allele, as specified in dataset metadata.
Preserves the VCF's integer coordinates to match the SINGER input; does not
read truth trees or silently change the dataset's legacy coordinate convention.
Only the Python standard library is required.
"""
import argparse
import hashlib
import json
from pathlib import Path


def convert(source, name):
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("vcf_ref") != "known simulated ancestral allele; no external reference genome":
        raise ValueError(f"{source}: REF is not documented as the ancestral allele")
    length = metadata["parameters"]["sequence_length"]
    vcf = source / f"{name}.vcf"
    positions, ancestors, columns = [], [], []
    samples = None
    chrom = None
    for line in vcf.read_text().splitlines():
        if line.startswith("#CHROM"):
            samples = line.split("\t")[9:]
        elif line and not line.startswith("#"):
            fields = line.split("\t")
            if samples is None or len(fields) != 9 + len(samples):
                raise ValueError(f"{vcf}: missing header or malformed row")
            if chrom is None:
                chrom = fields[0]
            if fields[0] != chrom:
                raise ValueError(f"{vcf}: multiple contigs")
            pos, ref, alt = int(fields[1]), fields[3], fields[4]
            if not (0 <= pos < length) or (positions and pos <= positions[-1]):
                raise ValueError(f"{vcf}: invalid or duplicate position {pos}")
            if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT" or ref == alt:
                raise ValueError(f"{vcf}: expected biallelic nucleotide SNP at {pos}")
            gt_index = fields[8].split(":").index("GT")
            alleles = []
            for sample in fields[9:]:
                gt = sample.split(":")[gt_index]
                if gt not in {"0|0", "0|1", "1|0", "1|1"}:
                    raise ValueError(f"{vcf}: expected complete phased diploid GT, found {gt}")
                alleles.extend((ref, alt)[int(x)] for x in gt.split("|"))
            if not 0 < sum(a != ref for a in alleles) < len(alleles):
                raise ValueError(f"{vcf}: non-segregating site {pos}")
            positions.append(pos)
            ancestors.append(ref)
            columns.append(alleles)
    if len(columns) < 2:
        raise ValueError("This ARGinfer reader needs at least two SNPs")
    n = len(columns[0])
    if n != metadata["num_haplotypes"] or len(columns) != metadata["num_sites"]:
        raise ValueError(f"{vcf}: VCF dimensions disagree with metadata")
    files = {
        "haplotypes.txt": "".join("\t".join(row) + "\n" for row in zip(*columns)),
        "ancestral.txt": "\n".join(ancestors) + "\n",
        "positions.txt": "\n".join(map(str, positions)) + "\n",
        "samples.tsv": "row\tvcf_sample\tphase\n" + "".join(
            f"{2*i+j}\t{sample}\t{j}\n" for i, sample in enumerate(samples) for j in range(2)
        ),
    }
    files["manifest.json"] = json.dumps({
        "source_vcf": str(vcf.resolve()),
        "source_sha256": hashlib.sha256(vcf.read_bytes()).hexdigest(),
        "num_haplotypes": n, "num_snps": len(columns),
        "sequence_length": length,
        "population_size": metadata["parameters"]["population_size"],
        "mutation_rate": metadata["parameters"]["mutation_rate"],
        "recombination_rate": metadata["parameters"]["recombination_rate"],
        "coordinates": "Unchanged integer VCF positions; project legacy rounding convention",
        "ancestral_alleles": "REF, documented as known simulated ancestral allele",
    }, indent=2) + "\n"
    return files, n, len(columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, required=True, help="paper_datasets directory")
    parser.add_argument("--output", type=Path, required=True, help="New destination directory")
    parser.add_argument("--replicate", default="rep0")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists; use a new directory: {args.output}")
    prepared = {name: convert(args.datasets / name / args.replicate, name) for name in ("r1", "r2", "r4")}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, (files, n, m) in prepared.items():
        dest = args.output / name
        dest.mkdir()
        for filename, contents in files.items():
            (dest / filename).write_text(contents)
        print(f"{name}: {n} haplotypes, {m} SNPs -> {dest}")


if __name__ == "__main__":
    main()
