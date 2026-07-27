#!/usr/bin/env python3
"""ARGsims-style point accuracy for SINGER posterior outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .point_accuracy_common import (
        ValidationResult,
        add_common_args,
        clean_artifact_names,
        common_metric_values,
        dataframe_from_tree_sequences,
        load_posterior_tree_samples,
        load_singer_bed_segments,
        parse_limits,
        plot_limits_from_args,
        prepare_output_prefix,
        run_singer_plots,
        write_standard_outputs,
    )
except ImportError:
    from point_accuracy_common import (
        ValidationResult,
        add_common_args,
        clean_artifact_names,
        common_metric_values,
        dataframe_from_tree_sequences,
        load_posterior_tree_samples,
        load_singer_bed_segments,
        parse_limits,
        plot_limits_from_args,
        prepare_output_prefix,
        run_singer_plots,
        write_standard_outputs,
    )


METHOD_LABEL = "SINGER"
METHOD_NAME = "singer"

_CLEAN_ARTIFACTS = {
    "meanest_lin_clean.png": "point_estimate_linear_clean.png",
    "meanest_log_clean.png": "point_estimate_log_clean.png",
    "meanest_lin.png": "point_estimate_linear.png",
    "meanest_log.png": "point_estimate_log.png",
    "MSEall.txt": "mse.txt",
    "MeanMSE_lin.png": "mean_mse_linear.png",
    "meanest_mean_lin.png": "mean_estimate_linear.png",
    "commonMetrics.tsv": "metrics.tsv",
    "timeSummary.tsv": "time_summary.tsv",
}


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
    parse_limits(args.xlim)
    parse_limits(args.ylim)
    parse_limits(args.xlim_log)
    parse_limits(args.ylim_log)
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


def evaluate_from_args(
    args: argparse.Namespace, *, output_dir: Path | None = None
) -> ValidationResult:
    if output_dir is None:
        out_prefix = prepare_output_prefix(args.output_prefix)
    else:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        out_prefix = output_dir / "result"
    xlim, ylim, xlim_log, ylim_log = plot_limits_from_args(args)
    method_label = getattr(args, "method_label", None) or METHOD_LABEL
    df = dataframe_from_args(args)
    print(f"segments: {len(df)} rows", flush=True)
    legacy_mse = run_singer_plots(
        df,
        out_prefix,
        xlim=xlim,
        ylim=ylim,
        xlim_log=xlim_log,
        ylim_log=ylim_log,
        label=method_label,
        vmax=1e10,
    )
    write_standard_outputs(
        df,
        out_prefix,
        method_label=method_label,
        legacy_mse=legacy_mse,
        xlim=xlim,
        ylim=ylim,
        xlim_log=xlim_log,
        ylim_log=ylim_log,
    )
    metrics = common_metric_values(df, legacy_mse)
    artifacts = (
        clean_artifact_names(out_prefix, _CLEAN_ARTIFACTS)
        if output_dir is not None
        else tuple()
    )
    print(f"MSEall = {legacy_mse}", flush=True)
    return ValidationResult(
        method=METHOD_NAME,
        method_label=method_label,
        metrics=metrics,
        legacy_mse=legacy_mse,
        artifacts=artifacts,
    )


def run_from_args(args: argparse.Namespace) -> float:
    return evaluate_from_args(args).legacy_mse


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
