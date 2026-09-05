#!/usr/bin/env python3
"""ARGsims-style point accuracy for SINGER posterior outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .point_accuracy import (
        add_common_args,
        dataframe_from_tree_sequences,
        load_posterior_tree_samples,
        load_singer_bed_segments,
        plot_limits_from_args,
        run_analysis,
        run_singer_plots,
    )
except ImportError:
    from point_accuracy import (
        add_common_args,
        dataframe_from_tree_sequences,
        load_posterior_tree_samples,
        load_singer_bed_segments,
        plot_limits_from_args,
        run_analysis,
        run_singer_plots,
    )


METHOD_LABEL = "SINGER"


def add_input_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--from-bed",
        action="store_true",
        help="Read ARGsims matched *_posterior.bed files instead of posterior trees.",
    )
    ap.add_argument(
        "-p",
        "--bed-prefix",
        default=None,
        help="Simulation prefix used in ARGsims SINGER/ARGweaver BED names.",
    )
    ap.add_argument(
        "--bed-dir",
        type=Path,
        default=None,
        help="Directory containing matched SINGER BED files.",
    )
    ap.add_argument(
        "--mcspl",
        default=None,
        help="MCMC sample label used in ARGsims BED names, e.g. 200-1200.",
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory of posterior .trees samples.",
    )
    ap.add_argument(
        "--sample-prefix",
        default=None,
        help="Filename prefix before posterior *.trees sample number.",
    )
    ap.add_argument("--burnin-samples", type=int, default=0)
    ap.add_argument("--max-posterior-samples", type=int, default=None)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ARGsims-style point accuracy for SINGER.")
    add_common_args(ap)
    add_input_args(ap)
    args = ap.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    plot_limits_from_args(args)
    if args.from_bed:
        if args.bed_prefix is None:
            raise SystemExit("--bed-prefix is required with --from-bed")
        if args.mcspl is None:
            raise SystemExit("--mcspl is required with --from-bed")
        return
    if args.input_dir is None or args.sample_prefix is None:
        raise SystemExit("--input-dir and --sample-prefix are required unless --from-bed is used")


def dataframe_from_args(args: argparse.Namespace):
    if args.from_bed:
        bed_dir = args.bed_dir or args.output_prefix.expanduser().resolve().parent
        return load_singer_bed_segments(
            bed_dir=bed_dir,
            bed_prefix=args.bed_prefix,
            nspl=args.nspl,
            skip=args.skip,
            mcspl=args.mcspl,
        )
    samples = load_posterior_tree_samples(
        args.input_dir,
        args.sample_prefix,
        burnin_samples=args.burnin_samples,
        max_posterior_samples=args.max_posterior_samples,
    )
    if args.verbose:
        print(f"Loaded {len(samples)} posterior tree sequence(s)", flush=True)
    return dataframe_from_tree_sequences(args, samples)


def run_from_args(args: argparse.Namespace) -> float:
    return run_analysis(
        args,
        dataframe_from_args,
        run_singer_plots,
        method_label=METHOD_LABEL,
        vmax=1e10,
    )


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
