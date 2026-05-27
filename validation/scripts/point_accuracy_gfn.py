#!/usr/bin/env python3
"""ARGsims-style point accuracy for GFN ARG outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .point_accuracy_common import (
        add_common_args,
        dataframe_from_tree_sequences,
        load_posterior_tree_samples,
        load_tree_sequence,
        parse_limits,
        plot_limits_from_args,
        prepare_output_prefix,
        run_tsinferdate_plots,
        write_standard_outputs,
    )
except ImportError:
    from point_accuracy_common import (
        add_common_args,
        dataframe_from_tree_sequences,
        load_posterior_tree_samples,
        load_tree_sequence,
        parse_limits,
        plot_limits_from_args,
        prepare_output_prefix,
        run_tsinferdate_plots,
        write_standard_outputs,
    )


METHOD_LABEL = "GFN"


def add_input_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--inferred-trees",
        type=Path,
        default=None,
        help="Single GFN-exported .trees file.",
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory of GFN-exported .trees samples.",
    )
    ap.add_argument(
        "--sample-prefix",
        default=None,
        help="Filename prefix before GFN *.trees sample number.",
    )
    ap.add_argument("--burnin-samples", type=int, default=0)
    ap.add_argument("--max-posterior-samples", type=int, default=None)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ARGsims-style point accuracy for GFN.")
    add_common_args(ap)
    add_input_args(ap)
    args = ap.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    parse_limits(args.xlim)
    parse_limits(args.ylim)
    parse_limits(args.xlim_log)
    parse_limits(args.ylim_log)
    has_single = args.inferred_trees is not None
    has_samples = args.input_dir is not None or args.sample_prefix is not None
    if has_single and has_samples:
        raise SystemExit("Use either --inferred-trees or --input-dir/--sample-prefix, not both")
    if has_single:
        return
    if args.input_dir is None or args.sample_prefix is None:
        raise SystemExit("Use --inferred-trees or both --input-dir and --sample-prefix")


def dataframe_from_args(args: argparse.Namespace):
    if args.inferred_trees is not None:
        inferred = load_tree_sequence(
            args.inferred_trees, f"GFN tree sequence {args.inferred_trees}"
        )
        if args.verbose:
            print(f"Loaded inferred TS: {args.inferred_trees}", flush=True)
        return dataframe_from_tree_sequences(args, [inferred])
    samples = load_posterior_tree_samples(
        args.input_dir,
        args.sample_prefix,
        burnin_samples=args.burnin_samples,
        max_posterior_samples=args.max_posterior_samples,
    )
    if args.verbose:
        print(f"Loaded {len(samples)} GFN tree sequence sample(s)", flush=True)
    return dataframe_from_tree_sequences(args, samples)


def run_from_args(args: argparse.Namespace) -> float:
    out_prefix = prepare_output_prefix(args.output_prefix)
    xlim, ylim, xlim_log, ylim_log = plot_limits_from_args(args)
    df = dataframe_from_args(args)
    print(f"segments: {len(df)} rows", flush=True)
    legacy_mse = run_tsinferdate_plots(
        df,
        out_prefix,
        xlim=xlim,
        ylim=ylim,
        xlim_log=xlim_log,
        ylim_log=ylim_log,
        label=METHOD_LABEL,
        tag="gfn",
        vmax=1e11,
    )
    write_standard_outputs(
        df,
        out_prefix,
        method_label=METHOD_LABEL,
        legacy_mse=legacy_mse,
        xlim=xlim,
        ylim=ylim,
        xlim_log=xlim_log,
        ylim_log=ylim_log,
    )
    print(f"MSEall = {legacy_mse}", flush=True)
    return legacy_mse


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
