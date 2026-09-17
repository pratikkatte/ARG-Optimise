"""Score a candidate ancestry against a simulator SNP bundle; print strict JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import tskit

# Permit direct invocation from any working directory without installation.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from env.infinite_sites import evaluate_infinite_sites
from env.snp_data import load_snp_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate-dir", required=True, type=Path,
                        help="Directory containing simulator metadata.json, VCF, and position map")
    parser.add_argument("--trees", required=True, type=Path,
                        help="Explicit candidate ancestry; mutation records are ignored")
    parser.add_argument("--mutation-rate", required=True, type=float,
                        help="Mutations per bp per generation")
    parser.add_argument("--sample-nodes", type=int, nargs="+",
                        help="Candidate sample node IDs in genotype-row order (default: ts.samples())")
    args = parser.parse_args(argv)
    try:
        data = load_snp_dataset(args.replicate_dir)
        ts = tskit.load(args.trees)
        result = evaluate_infinite_sites(ts, data, mutation_rate=args.mutation_rate,
                                        sample_nodes=args.sample_nodes)
        report = {
            "replicate_dir": str(args.replicate_dir),
            "candidate_trees": str(args.trees),
            "num_haplotypes": data.num_haplotypes,
            "num_variants": data.num_variants,
            "sequence_length": data.sequence_length,
            "contig_id": data.contig_id,
            "haplotype_ids": list(data.haplotype_ids),
            "sample_nodes": list(map(int, ts.samples() if args.sample_nodes is None else args.sample_nodes)),
            "mutation_rate_per_bp_per_generation": args.mutation_rate,
            "log_likelihood": None if result.zero_likelihood else result.log_likelihood,
            "zero_likelihood": result.zero_likelihood,
            "exposure_generations_bp": result.exposure,
            "incompatible_site_ids": list(result.incompatible_site_ids),
            "site_ids": list(data.site_ids),
            "compatible_branch_lengths_generations": result.compatible_branch_lengths.tolist(),
            "likelihood_convention": "polarized mutation-pattern density; fixed data-only factors omitted",
        }
        output = json.dumps(report, indent=2, allow_nan=False)
    except (OSError, ValueError, tskit.FileFormatError, tskit.LibraryError) as exc:
        parser.error(str(exc))
    print(output)


if __name__ == "__main__":
    main()
