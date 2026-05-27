#!/usr/bin/env python3
"""Compatibility wrapper for method-specific point-accuracy scripts.

New code should call one of:

- point_accuracy_tsinferdate.py
- point_accuracy_singer.py
- point_accuracy_gfn.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .point_accuracy_common import *  # noqa: F401,F403
    from .point_accuracy_common import add_common_args, parse_limits
    from .point_accuracy_gfn import run_from_args as run_gfn_from_args
    from .point_accuracy_singer import run_from_args as run_singer_from_args
    from .point_accuracy_tsinferdate import run_from_args as run_tsinferdate_from_args
except ImportError:
    from point_accuracy_common import *  # noqa: F401,F403
    from point_accuracy_common import add_common_args, parse_limits
    from point_accuracy_gfn import run_from_args as run_gfn_from_args
    from point_accuracy_singer import run_from_args as run_singer_from_args
    from point_accuracy_tsinferdate import run_from_args as run_tsinferdate_from_args


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper for ARGsims-style point accuracy. Prefer "
            "point_accuracy_tsinferdate.py, point_accuracy_singer.py, or "
            "point_accuracy_gfn.py for new runs."
        )
    )
    ap.add_argument(
        "--method",
        choices=("tsinfer", "tsinferdate", "singer", "gfn"),
        default="tsinfer",
        help="Inference method (default: tsinfer).",
    )
    add_common_args(ap)
    ap.add_argument(
        "--from-bed",
        action="store_true",
        help="Read pre-built ARGsims BEDs instead of tree sequences.",
    )
    ap.add_argument(
        "-p",
        "--bed-prefix",
        default=None,
        help="BED file stem before pair index (--from-bed).",
    )
    ap.add_argument(
        "--bed-dir",
        type=Path,
        default=None,
        help="Directory for --from-bed inputs (default: output-prefix parent).",
    )
    ap.add_argument(
        "--mcspl",
        default=None,
        help="MCMC label for SINGER ARGsims-style BED names (--from-bed).",
    )
    ap.add_argument(
        "--inferred-trees",
        type=Path,
        default=None,
        help="[tsinfer/gfn] Single dated/exported .trees file.",
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="[singer/gfn] Directory of posterior .trees samples.",
    )
    ap.add_argument(
        "--sample-prefix",
        default=None,
        help="[singer/gfn] Filename prefix before *.trees.",
    )
    ap.add_argument("--burnin-samples", type=int, default=0)
    ap.add_argument("--max-posterior-samples", type=int, default=None)
    args = ap.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    parse_limits(args.xlim)
    parse_limits(args.ylim)
    parse_limits(args.xlim_log)
    parse_limits(args.ylim_log)
    method = "tsinferdate" if args.method == "tsinfer" else args.method
    if method == "tsinferdate":
        if args.from_bed:
            if args.bed_prefix is None:
                raise SystemExit("--bed-prefix is required with --from-bed")
        elif args.inferred_trees is None:
            raise SystemExit("--inferred-trees is required for method tsinfer")
    elif method == "singer":
        if args.from_bed:
            if args.bed_prefix is None:
                raise SystemExit("--bed-prefix is required with --from-bed")
            if args.mcspl is None:
                raise SystemExit("--mcspl is required with --from-bed and method singer")
        elif args.input_dir is None or args.sample_prefix is None:
            raise SystemExit("--input-dir and --sample-prefix are required for method singer")
    elif method == "gfn":
        has_single = args.inferred_trees is not None
        has_samples = args.input_dir is not None or args.sample_prefix is not None
        if args.from_bed:
            raise SystemExit("--from-bed is not supported for method gfn")
        if has_single and has_samples:
            raise SystemExit("Use either --inferred-trees or --input-dir/--sample-prefix")
        if not has_single and (args.input_dir is None or args.sample_prefix is None):
            raise SystemExit("Use --inferred-trees or both --input-dir and --sample-prefix")


def main() -> None:
    args = parse_args()
    if args.method in ("tsinfer", "tsinferdate"):
        run_tsinferdate_from_args(args)
    elif args.method == "singer":
        run_singer_from_args(args)
    else:
        run_gfn_from_args(args)


if __name__ == "__main__":
    main()
