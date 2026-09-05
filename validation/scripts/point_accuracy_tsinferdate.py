#!/usr/bin/env python3
"""ARGsims-style point accuracy for tsinfer+tsdate outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .point_accuracy import (
        add_common_args,
        dataframe_from_tree_sequences,
        load_tree_sequence,
        load_tsinferdate_bed_segments,
        plot_limits_from_args,
        run_analysis,
        run_tsinferdate_plots,
    )
except ImportError:
    from point_accuracy import (
        add_common_args,
        dataframe_from_tree_sequences,
        load_tree_sequence,
        load_tsinferdate_bed_segments,
        plot_limits_from_args,
        run_analysis,
        run_tsinferdate_plots,
    )


METHOD_LABEL = "tsdate"


def add_input_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--from-bed",
        action="store_true",
        help="Read ARGsims matched *_post.bed(.gz) files instead of a tree sequence.",
    )
    ap.add_argument(
        "-p",
        "--bed-prefix",
        default=None,
        help="BED stem before pair index, e.g. sim_l1mb_priorDef_spls.",
    )
    ap.add_argument(
        "--bed-dir",
        type=Path,
        default=None,
        help="Directory containing matched BED files.",
    )
    ap.add_argument(
        "--inferred-trees",
        type=Path,
        default=None,
        help="Single dated tsinfer+tsdate .trees file.",
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="ARGsims-style point accuracy for tsinfer+tsdate."
    )
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
        return
    if args.inferred_trees is None:
        raise SystemExit("--inferred-trees is required unless --from-bed is used")


def dataframe_from_args(args: argparse.Namespace):
    if args.from_bed:
        bed_dir = args.bed_dir or args.output_prefix.expanduser().resolve().parent
        return load_tsinferdate_bed_segments(
            bed_dir=bed_dir,
            bed_prefix=args.bed_prefix,
            nspl=args.nspl,
            skip=args.skip,
        )
    inferred = load_tree_sequence(
        args.inferred_trees, f"tsinfer+tsdate tree sequence {args.inferred_trees}"
    )
    if args.verbose:
        print(f"Loaded inferred TS: {args.inferred_trees}", flush=True)
    return dataframe_from_tree_sequences(args, [inferred])


def run_from_args(args: argparse.Namespace) -> float:
    return run_analysis(
        args,
        dataframe_from_args,
        run_tsinferdate_plots,
        method_label=METHOD_LABEL,
        tag="ts",
        vmax=1e11,
    )


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
