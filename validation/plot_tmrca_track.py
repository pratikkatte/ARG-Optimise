#!/usr/bin/env python3
"""Plot matched pairwise-TMRCA RMSE along the chromosome for ARG samples."""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .scripts.point_accuracy_common import (
    common_metric_values,
    dataframe_from_tree_sequences,
    load_posterior_tree_samples,
)


def _model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("models must be LABEL=INPUT_DIR")
    label, directory = value.split("=", 1)
    return label, Path(directory)


def _weighted_bin_rmse(frame: pd.DataFrame, left: float, right: float) -> float:
    overlap = np.maximum(
        0.0,
        np.minimum(frame["end"].to_numpy(dtype=float), right)
        - np.maximum(frame["start"].to_numpy(dtype=float), left),
    )
    valid = (
        (overlap > 0)
        & np.isfinite(frame["Simulated"].to_numpy(dtype=float))
        & np.isfinite(frame["PosteriorMean"].to_numpy(dtype=float))
    )
    if not valid.any():
        return float("nan")
    error = (
        frame["PosteriorMean"].to_numpy(dtype=float)[valid]
        - frame["Simulated"].to_numpy(dtype=float)[valid]
    )
    return float(np.sqrt(np.average(error**2, weights=overlap[valid])))


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    metrics: list[dict[str, float | str]] = []
    common_args = Namespace(
        truth_trees=args.truth,
        truth_dir=None,
        truth_prefix=None,
        ne=args.ne,
        nspl=args.haplotypes,
        skip=args.skip,
        max_pairs=args.max_pairs,
        pair_seed=args.pair_seed,
        verbose=False,
    )
    for label, directory in args.model:
        trees = load_posterior_tree_samples(
            directory,
            args.sample_prefix,
            max_posterior_samples=args.posterior_samples,
        )
        frame = dataframe_from_tree_sequences(common_args, trees)
        frames[label] = frame
        full = common_metric_values(frame, float("nan"))
        local = frame[
            (frame["end"] > args.local_start)
            & (frame["start"] < args.local_end)
        ].copy()
        local["start"] = np.maximum(local["start"], args.local_start)
        local["end"] = np.minimum(local["end"], args.local_end)
        local["len"] = local["end"] - local["start"]
        local_values = common_metric_values(local, float("nan"))
        metrics.append(
            {
                "model": label,
                "posterior_samples": len(trees),
                "full_rmse_t_2ne": full["weighted_rmse"],
                "full_rmse_generations": float(full["weighted_rmse"]) * 2 * args.ne,
                "local_rmse_t_2ne": local_values["weighted_rmse"],
                "local_rmse_generations": float(local_values["weighted_rmse"]) * 2 * args.ne,
                "local_log_rmse": local_values["weighted_log_rmse"],
                "local_spearman": local_values["weighted_spearman_correlation"],
                "local_95pct_coverage": local_values["posterior_95pct_coverage"],
            }
        )
    sequence_length = max(float(frame["end"].max()) for frame in frames.values())
    bins = np.arange(0.0, sequence_length + args.bin_width, args.bin_width)
    rows = []
    for label, frame in frames.items():
        for left, right in zip(bins[:-1], bins[1:]):
            rows.append(
                {
                    "model": label,
                    "left": left,
                    "right": min(right, sequence_length),
                    "midpoint": (left + min(right, sequence_length)) / 2,
                    "rmse_t_2ne": _weighted_bin_rmse(frame, left, right),
                }
            )
    track = pd.DataFrame(rows)
    track["rmse_generations"] = track["rmse_t_2ne"] * 2.0 * args.ne
    track.to_csv(args.output_dir / "rmse_track.tsv", sep="\t", index=False)
    pd.DataFrame(metrics).to_csv(
        args.output_dir / "tmrca_summary.tsv", sep="\t", index=False
    )

    fig, ax = plt.subplots(figsize=(9.0, 4.25))
    for label, group in track.groupby("model", sort=False):
        ax.plot(group["midpoint"] / 1000.0, group["rmse_generations"], marker="o", label=label)
    ax.axvspan(args.local_start / 1000.0, args.local_end / 1000.0, color="0.8", alpha=0.35)
    ax.set_xlabel("Genomic position (kb)")
    ax.set_ylabel("Pairwise TMRCA RMSE (generations)")
    ax.set_title("Matched posterior-mean TMRCA error across the 1 Mb ARG")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "full_1mb_rmse_track.png", dpi=200)
    plt.close(fig)

    local_track = track[
        (track["right"] > args.local_start) & (track["left"] < args.local_end)
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.25))
    for label, group in local_track.groupby("model", sort=False):
        ax.plot(group["midpoint"] / 1000.0, group["rmse_generations"], marker="o", label=label)
    ax.set_xlabel("Genomic position (kb)")
    ax.set_ylabel("Pairwise TMRCA RMSE (generations)")
    ax.set_title("Detailed RMSE in the predefined 400–500 kb interval")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "local_400kb_500kb_rmse.png", dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--model", type=_model, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ne", type=float, default=10_000.0)
    parser.add_argument("--haplotypes", type=int, default=8)
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--pair-seed", type=int, default=42)
    parser.add_argument("--sample-prefix", default="arg_")
    parser.add_argument("--posterior-samples", type=int, default=16)
    parser.add_argument("--bin-width", type=float, default=25_000.0)
    parser.add_argument("--local-start", type=float, default=400_000.0)
    parser.add_argument("--local-end", type=float, default=500_000.0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
