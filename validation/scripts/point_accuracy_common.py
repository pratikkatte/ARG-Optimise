#!/usr/bin/env python3
"""Shared point-accuracy helpers for ARGsims-style validation scripts."""

from __future__ import annotations

import argparse
import gzip
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tskit
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext

PAIR_RE = re.compile(r"_spls(\d+)-(\d+)\.tc$")
SAMPLE_RE = re.compile(r"_(\d+)\.trees$")

DEFAULT_XLIM = (0.0, 16.0)
DEFAULT_YLIM = (0.0, 8.0)
DEFAULT_XLIM_LOG = (-4.0, 1.5)
DEFAULT_YLIM_LOG = (-4.0, 1.5)


@dataclass(frozen=True)
class TruthInterval:
    left: float
    right: float
    tcoal_2ne: float


@dataclass(frozen=True)
class PairSegment:
    pair: tuple[int, int]
    left: float
    right: float
    truth: float
    posterior_mean: float
    posterior_median: float
    posterior_low: float = float("nan")
    posterior_high: float = float("nan")

    @property
    def length(self) -> float:
        return self.right - self.left


@dataclass(frozen=True)
class ValidationResult:
    """Structured result returned by reusable method adapters."""

    method: str
    method_label: str
    metrics: dict[str, float | int | str]
    legacy_mse: float
    artifacts: tuple[Path, ...]


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--truth-dir",
        type=Path,
        default=None,
        help="Directory of truth *_splsX-Y.tc tracks.",
    )
    ap.add_argument(
        "--truth-prefix",
        default=None,
        help="Truth filename stem before _spls, e.g. sim_l1mb_0.",
    )
    ap.add_argument(
        "--truth-trees",
        type=Path,
        default=None,
        help="Optional msprime truth .trees; if loadable, replaces .tc tracks.",
    )
    ap.add_argument(
        "--ne",
        type=float,
        default=10000.0,
        help="Effective population size Ne; tree-sequence times are divided by 2*Ne.",
    )
    ap.add_argument("-n", "--nspl", type=int, required=True, help="Number of haplotypes.")
    ap.add_argument(
        "-s",
        "--skip",
        type=int,
        default=1,
        help="Stride on pair indices (default 1).",
    )
    ap.add_argument(
        "-o",
        "--output-prefix",
        type=Path,
        required=True,
        help="Output path prefix; files are written as <prefix><suffix>.",
    )
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--pair-seed", type=int, default=42)
    ap.add_argument("--xlim", default="0,16")
    ap.add_argument("--ylim", default="0,8")
    ap.add_argument("--xlim-log", default="-4,1.5")
    ap.add_argument("--ylim-log", default="-4,1.5")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--region-start", type=float, default=None)
    ap.add_argument("--region-end", type=float, default=None)


def clip_dataframe_to_region(
    df: pd.DataFrame,
    region_start: float | None,
    region_end: float | None,
) -> pd.DataFrame:
    """Clip aligned genomic segments to a half-open validation interval."""
    if region_start is None and region_end is None:
        return df
    if region_start is None or region_end is None or region_end <= region_start:
        raise ValueError("region requires finite bounds satisfying start < end")
    clipped = df[
        (df["end"].to_numpy(dtype=float) > region_start)
        & (df["start"].to_numpy(dtype=float) < region_end)
    ].copy()
    if clipped.empty:
        raise RuntimeError(
            f"No aligned segments overlap validation region [{region_start}, {region_end})"
        )
    clipped["start"] = np.maximum(
        clipped["start"].to_numpy(dtype=float), region_start
    )
    clipped["end"] = np.minimum(
        clipped["end"].to_numpy(dtype=float), region_end
    )
    clipped["len"] = clipped["end"] - clipped["start"]
    return clipped[clipped["len"] > 0].reset_index(drop=True)


def require_truth_args(args: argparse.Namespace) -> None:
    if args.truth_trees is None and args.truth_dir is None:
        raise SystemExit("--truth-dir or --truth-trees is required for tree inputs")
    if args.truth_trees is None and args.truth_prefix is None:
        raise SystemExit("--truth-prefix is required when --truth-trees is not provided")


def parse_pair_from_truth_name(path: Path) -> tuple[int, int]:
    match = PAIR_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse pair from truth filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def parse_limits(value: str) -> tuple[float, float]:
    fields = value.split(",")
    if len(fields) != 2:
        raise SystemExit(f"Expected comma-separated lower,upper limit, got {value!r}")
    lower, upper = map(float, fields)
    if upper <= lower:
        raise SystemExit(f"Plot limit upper must exceed lower, got {value!r}")
    return lower, upper


def plot_limits_from_args(
    args: argparse.Namespace,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    return (
        parse_limits(args.xlim),
        parse_limits(args.ylim),
        parse_limits(args.xlim_log),
        parse_limits(args.ylim_log),
    )


def prepare_output_prefix(output_prefix: Path) -> Path:
    out_prefix = output_prefix.expanduser().resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    return out_prefix


def clean_artifact_names(
    out_prefix: Path, suffix_to_name: dict[str, str]
) -> tuple[Path, ...]:
    """Rename prefix-based legacy artifacts into a clean method directory."""
    artifacts: list[Path] = []
    for suffix, clean_name in suffix_to_name.items():
        source = out_prefix.parent / f"{out_prefix.name}{suffix}"
        if not source.exists():
            continue
        target = out_prefix.parent / clean_name
        source.replace(target)
        artifacts.append(target)
    return tuple(sorted(artifacts))


def load_truth_tracks(
    truth_dir: Path, prefix: str, ne: float
) -> dict[tuple[int, int], list[TruthInterval]]:
    truth_files = sorted(truth_dir.glob(f"{prefix}_spls*-*.tc"))
    if not truth_files:
        raise FileNotFoundError(
            f"No truth files found with pattern {prefix}_spls*-*.tc in {truth_dir}"
        )
    scale = 2.0 * ne
    tracks: dict[tuple[int, int], list[TruthInterval]] = {}
    for truth_file in truth_files:
        pair = parse_pair_from_truth_name(truth_file)
        intervals: list[TruthInterval] = []
        with truth_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                fields = line.strip().split("\t")
                if len(fields) < 3:
                    continue
                left = float(fields[0])
                right = float(fields[1])
                if right <= left:
                    continue
                intervals.append(
                    TruthInterval(
                        left=left,
                        right=right,
                        tcoal_2ne=float(fields[2]) / scale,
                    )
                )
        if not intervals:
            raise ValueError(f"Truth file has no valid intervals: {truth_file}")
        tracks[pair] = intervals
    return tracks


def warn_if_not_generations(ts: tskit.TreeSequence, label: str) -> None:
    if ts.time_units != "generations":
        print(
            f"warning: {label} time_units={ts.time_units!r}; evaluator assumes "
            "raw TMRCA values are generations before dividing by 2*Ne.",
            file=sys.stderr,
            flush=True,
        )


def load_tree_sequence(path: Path, label: str) -> tskit.TreeSequence:
    ts = tskit.load(path)
    warn_if_not_generations(ts, label)
    return ts


def load_truth_tracks_from_trees(
    truth_trees: Path,
    pairs: Iterable[tuple[int, int]],
    ne: float,
) -> dict[tuple[int, int], list[TruthInterval]]:
    ts = load_tree_sequence(truth_trees, f"truth tree sequence {truth_trees}")
    samples = ts.samples()
    n = len(samples)
    scale = 2.0 * ne
    tracks: dict[tuple[int, int], list[TruthInterval]] = {pair: [] for pair in pairs}
    for pair in tracks:
        if pair[0] < 0 or pair[1] < 0 or pair[0] >= n or pair[1] >= n:
            raise ValueError(
                f"Pair {pair} is out of range for truth tree sequence with {n} samples"
            )
    for tree in ts.trees():
        left, right = tree.interval
        if right <= left:
            continue
        for pair, intervals in tracks.items():
            tcoal = tree.tmrca(samples[pair[0]], samples[pair[1]]) / scale
            intervals.append(
                TruthInterval(
                    left=float(left),
                    right=float(right),
                    tcoal_2ne=float(tcoal),
                )
            )
    return tracks


def resolve_truth_trees_path(truth_trees_arg: Path | None) -> Path | None:
    if truth_trees_arg is None:
        return None
    p = truth_trees_arg.expanduser().resolve()
    if not p.is_file():
        return None
    try:
        tskit.load(str(p))
    except Exception as exc:
        raise SystemExit(
            f"--truth-trees {p} exists but could not be loaded as a tree sequence: {exc}"
        ) from exc
    return p


def build_truth_tracks(
    args: argparse.Namespace, pairs: list[tuple[int, int]]
) -> dict[tuple[int, int], list[TruthInterval]]:
    truth_trees = resolve_truth_trees_path(
        args.truth_trees.expanduser() if args.truth_trees else None
    )
    if truth_trees is not None:
        return load_truth_tracks_from_trees(truth_trees, pairs, args.ne)
    if args.truth_trees is not None and args.truth_dir is None:
        raise SystemExit(
            f"--truth-trees {args.truth_trees} is not a loadable tree sequence "
            "and no --truth-dir fallback was provided"
        )
    return load_truth_tracks(args.truth_dir, args.truth_prefix, args.ne)


def select_pairs(
    pairs: Iterable[tuple[int, int]],
    max_pairs: int | None,
    seed: int,
) -> list[tuple[int, int]]:
    ordered = sorted(pairs)
    if max_pairs is None or max_pairs >= len(ordered):
        return ordered
    if max_pairs <= 0:
        raise ValueError("--max-pairs must be positive when provided")
    rng = random.Random(seed)
    return sorted(rng.sample(ordered, max_pairs))


def iter_pairs(nspl: int, skip: int) -> list[tuple[int, int]]:
    return [
        (s1, s2)
        for s1 in range(0, nspl - 1, skip)
        for s2 in range(s1 + 1, nspl, skip)
    ]


def _sample_path_key(path: Path) -> tuple[str, int, str]:
    match = SAMPLE_RE.search(path.name)
    if match is None:
        return (path.name, -1, path.name)
    return (path.name[: match.start(1)], int(match.group(1)), path.name)


def load_posterior_tree_samples(
    input_dir: Path,
    sample_prefix: str,
    *,
    burnin_samples: int = 0,
    max_posterior_samples: int | None = None,
) -> list[tskit.TreeSequence]:
    sample_paths = sorted(
        input_dir.glob(f"{sample_prefix}*.trees"), key=_sample_path_key
    )
    if not sample_paths:
        raise FileNotFoundError(
            f"No {sample_prefix}*.trees files found in {input_dir}"
        )
    if burnin_samples < 0:
        raise ValueError("--burnin-samples cannot be negative")
    if burnin_samples >= len(sample_paths):
        raise ValueError(
            f"--burnin-samples={burnin_samples} drops all {len(sample_paths)} samples"
        )
    sample_paths = sample_paths[burnin_samples:]
    if max_posterior_samples is not None:
        if max_posterior_samples <= 0:
            raise ValueError("--max-posterior-samples must be positive when provided")
        sample_paths = sample_paths[:max_posterior_samples]
    if not sample_paths:
        raise ValueError(
            "No posterior samples left after --burnin-samples / --max-posterior-samples"
        )
    return [
        load_tree_sequence(path, f"posterior tree sequence {path}")
        for path in sample_paths
    ]


def _truth_value_at(
    intervals: list[TruthInterval], position: float, start_idx: int
) -> tuple[float, int]:
    idx = start_idx
    while idx < len(intervals) and position >= intervals[idx].right:
        idx += 1
    if idx >= len(intervals):
        return float("nan"), idx
    if intervals[idx].left <= position < intervals[idx].right:
        return intervals[idx].tcoal_2ne, idx
    return float("nan"), idx


def _posterior_values_at(
    trees: list[tskit.TreeSequence],
    pair: tuple[int, int],
    position: float,
    scale: float,
) -> tuple[float, ...]:
    left_idx, right_idx = pair
    vals: list[float] = []
    for ts in trees:
        samples = ts.samples()
        n = len(samples)
        if left_idx < 0 or right_idx < 0 or left_idx >= n or right_idx >= n:
            raise ValueError(
                f"Pair {pair} is out of range for tree sequence with {n} samples"
            )
        tree = ts.at(position)
        try:
            tmrca = tree.tmrca(samples[left_idx], samples[right_idx]) / scale
        except ValueError:
            tmrca = float("nan")
        vals.append(float(tmrca))
    return tuple(vals)


def combine_pair_segments(
    pair: tuple[int, int],
    truth_intervals: list[TruthInterval],
    trees: list[tskit.TreeSequence],
    ne: float,
) -> list[PairSegment]:
    sequence_length = float(trees[0].sequence_length)
    for ts in trees[1:]:
        if not math.isclose(float(ts.sequence_length), sequence_length):
            raise ValueError("Tree samples have inconsistent sequence lengths")

    truth_intervals = sorted(truth_intervals, key=lambda iv: (iv.left, iv.right))
    breakpoints = {0.0, sequence_length}
    breakpoints.update(interval.left for interval in truth_intervals)
    breakpoints.update(interval.right for interval in truth_intervals)
    for ts in trees:
        breakpoints.update(float(bp) for bp in ts.breakpoints())

    sorted_bp = sorted(bp for bp in breakpoints if 0.0 <= bp <= sequence_length)
    segments: list[PairSegment] = []
    truth_idx = 0
    scale = 2.0 * ne
    for left, right in zip(sorted_bp[:-1], sorted_bp[1:]):
        if right <= left:
            continue
        mid = (left + right) / 2.0
        truth_val, truth_idx = _truth_value_at(truth_intervals, mid, truth_idx)
        if not math.isfinite(truth_val):
            continue
        posterior_vals = _posterior_values_at(trees, pair, mid, scale)
        finite_vals = [x for x in posterior_vals if math.isfinite(x)]
        if not finite_vals:
            continue
        segments.append(
            PairSegment(
                pair=pair,
                left=left,
                right=right,
                truth=truth_val,
                posterior_mean=float(np.mean(finite_vals)),
                posterior_median=float(np.median(finite_vals)),
                posterior_low=float(np.quantile(finite_vals, 0.025)),
                posterior_high=float(np.quantile(finite_vals, 0.975)),
            )
        )
    return segments


def collect_segments_from_trees(
    *,
    truth_tracks: dict[tuple[int, int], list[TruthInterval]],
    pairs: list[tuple[int, int]],
    inferred_trees: list[tskit.TreeSequence],
    ne: float,
    verbose: bool,
) -> list[PairSegment]:
    all_segments: list[PairSegment] = []
    for pi, pair in enumerate(pairs, start=1):
        if pair not in truth_tracks:
            print(f"skip pair {pair}: no truth track", file=sys.stderr)
            continue
        if verbose:
            print(f"pair {pi}/{len(pairs)} {pair}: aligning segments ...", flush=True)
        all_segments.extend(
            combine_pair_segments(pair, truth_tracks[pair], inferred_trees, ne)
        )
    if not all_segments:
        raise RuntimeError("No aligned segments were generated.")
    return all_segments


def dataframe_from_tree_sequences(
    args: argparse.Namespace, inferred_trees: list[tskit.TreeSequence]
) -> pd.DataFrame:
    require_truth_args(args)
    requested_pairs = iter_pairs(args.nspl, args.skip)
    all_truth = build_truth_tracks(args, requested_pairs)
    pairs = select_pairs(all_truth.keys(), args.max_pairs, args.pair_seed)
    valid_requested = set(requested_pairs)
    pairs = [pair for pair in pairs if pair in valid_requested]
    segments = collect_segments_from_trees(
        truth_tracks=all_truth,
        pairs=pairs,
        inferred_trees=inferred_trees,
        ne=args.ne,
        verbose=args.verbose,
    )
    return segments_to_dataframe(segments)


def segments_to_dataframe(segments: list[PairSegment]) -> pd.DataFrame:
    columns = [
        "chr",
        "start",
        "end",
        "Simulated",
        "PosteriorMean",
        "PosteriorMedian",
        "PosteriorLow",
        "PosteriorHigh",
        "len",
    ]
    if not segments:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(
        {
            "chr": "1",
            "start": [int(round(s.left)) for s in segments],
            "end": [int(round(s.right)) for s in segments],
            "Simulated": [s.truth for s in segments],
            "PosteriorMean": [s.posterior_mean for s in segments],
            "PosteriorMedian": [s.posterior_median for s in segments],
            "PosteriorLow": [s.posterior_low for s in segments],
            "PosteriorHigh": [s.posterior_high for s in segments],
            "len": [s.length for s in segments],
        },
        columns=columns,
    )


def load_bed_pair(path: Path, *, has_median: bool) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt"
    with opener(path, mode, encoding="utf-8") as fh:
        if has_median:
            names = [
                "chr",
                "start",
                "end",
                "Simulated",
                "PosteriorMean",
                "PosteriorMedian",
            ]
            usecols = range(6)
        else:
            names = ["chr", "start", "end", "Simulated", "PosteriorMean"]
            usecols = range(5)
        df = pd.read_csv(fh, sep="\t", header=None, usecols=usecols, names=names)
    df = df[df["end"] > df["start"]].copy()
    if "PosteriorMedian" not in df:
        df["PosteriorMedian"] = df["PosteriorMean"]
    df["len"] = (df["end"] - df["start"]).astype(float)
    return df


def load_tsinferdate_bed_segments(
    *, bed_dir: Path, bed_prefix: str, nspl: int, skip: int
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for s1, s2 in iter_pairs(nspl, skip):
        pair = f"{s1}-{s2}"
        path = bed_dir / f"{bed_prefix}{pair}_post.bed.gz"
        if not path.is_file():
            path = bed_dir / f"{bed_prefix}{pair}_post.bed"
        if not path.is_file():
            raise FileNotFoundError(f"Missing tsinferdate BED for pair {pair}: {path}")
        df = load_bed_pair(path, has_median=False)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError("No tsinferdate BED segments loaded.")
    return pd.concat(frames, ignore_index=True)


def load_singer_bed_segments(
    *, bed_dir: Path, bed_prefix: str, nspl: int, skip: int, mcspl: str
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for s1, s2 in iter_pairs(nspl, skip):
        pair = f"{s1}-{s2}"
        path = bed_dir / f"arg-sample_sim_{bed_prefix}_{pair}_Tcpair_spl{mcspl}_posterior.bed"
        if not path.is_file():
            path = bed_dir / f"{bed_prefix}_pair_{pair}_posterior.bed"
        if not path.is_file():
            raise FileNotFoundError(f"Missing SINGER BED for pair {pair}: {path}")
        df = load_bed_pair(path, has_median=True)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError("No SINGER BED segments loaded.")
    return pd.concat(frames, ignore_index=True)


def aggregate_pairs_linear(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            [df["Simulated"].round(1), df["PosteriorMean"].round(1)], as_index=False
        )["len"]
        .sum()
        .rename(columns={"len": "x"})
    )


def aggregate_pairs_log10(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["Simulated"] > 0) & (df["PosteriorMean"] > 0)].copy()
    if d.empty:
        return pd.DataFrame(columns=["Simulated", "PosteriorMean", "x"])
    d["ls"] = np.log10(d["Simulated"].values)
    d["lp"] = np.log10(d["PosteriorMean"].values)
    return (
        d.groupby([d["ls"].round(1), d["lp"].round(1)], as_index=False)["len"]
        .sum()
        .rename(columns={"ls": "Simulated", "lp": "PosteriorMean", "len": "x"})
    )


def heatmap_tables_from_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return aggregate_pairs_linear(df), aggregate_pairs_log10(df)


def mse_by_rounded_bins(toplot: pd.DataFrame) -> pd.DataFrame:
    if toplot.empty:
        return pd.DataFrame(columns=["SimRound", "len", "Mean", "SqErr", "MSE"])
    records: list[dict] = []
    for sim, group in toplot.groupby("Simulated"):
        weights = group["x"].to_numpy(dtype=float)
        post = group["PosteriorMean"].to_numpy(dtype=float)
        sim_values = group["Simulated"].to_numpy(dtype=float)
        length_sum = weights.sum()
        if length_sum <= 0:
            continue
        sq_err = float(((post - sim_values) ** 2 * weights).sum())
        records.append(
            {
                "SimRound": sim,
                "len": length_sum,
                "Mean": float(np.average(post, weights=weights)),
                "SqErr": sq_err,
                "MSE": sq_err / length_sum,
            }
        )
    return pd.DataFrame(records).sort_values("SimRound").reset_index(drop=True)


def mse_by_segments(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["SimRound", "len", "Mean", "MSE", "WtdSqErr"])
    work = df.copy()
    work["SimRound"] = work["Simulated"].round(1)
    work["WtdMean"] = work["PosteriorMean"] * work["len"]
    work["WtdSqErr"] = (work["PosteriorMean"] - work["Simulated"]) ** 2 * work["len"]
    grouped = (
        work.groupby("SimRound", as_index=False)
        .agg(WtdMean=("WtdMean", "sum"), WtdSqErr=("WtdSqErr", "sum"), len=("len", "sum"))
    )
    grouped["Mean"] = grouped["WtdMean"] / grouped["len"]
    grouped["MSE"] = grouped["WtdSqErr"] / grouped["len"]
    return grouped.sort_values("SimRound").reset_index(drop=True)


def raster_heatmap(
    toplot: pd.DataFrame,
    out_path: Path,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    ylabel: str,
    with_colorbar: bool,
    vmax: float,
) -> None:
    if toplot.empty:
        return
    width = 0.1
    nxb = int(round((xlim[1] - xlim[0]) / width)) + 1
    nyb = int(round((ylim[1] - ylim[0]) / width)) + 1
    x0, y0 = xlim[0], ylim[0]
    x_edges = x0 + width * np.arange(nxb)
    y_edges = y0 + width * np.arange(nyb)
    z = np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=float)
    for _, row in toplot.iterrows():
        i = int(round((float(row["Simulated"]) - x0) / width))
        j = int(round((float(row["PosteriorMean"]) - y0) / width))
        if 0 <= i < z.shape[1] and 0 <= j < z.shape[0]:
            z[j, i] += float(row["x"])
    z_display = np.where(z > 0, z, np.nan)
    fig, ax = plt.subplots(
        figsize=(2 if not with_colorbar else 3, 2.5 if with_colorbar else 2.0)
    )
    im = ax.imshow(
        z_display,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1] + width, y_edges[0], y_edges[-1] + width],
        interpolation="nearest",
        cmap="magma",
        norm=LogNorm(vmin=1, vmax=vmax, clip=True),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    tdiag = min(xlim[1] - x0, ylim[1] - y0)
    if tdiag > 0:
        ax.plot(
            (x0, x0 + tdiag),
            (y0, y0 + tdiag),
            color="white",
            alpha=0.45,
            lw=0.6,
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if with_colorbar:
        cbar = fig.colorbar(
            im,
            ax=ax,
            format=LogFormatterMathtext(10, labelOnlyBase=False),
        )
        cbar.set_label("nSites", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fill_grid(
    toplot: pd.DataFrame, xlim: tuple[float, float], ylim: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    width = 0.1
    nxb = int(round((xlim[1] - xlim[0]) / width)) + 1
    nyb = int(round((ylim[1] - ylim[0]) / width)) + 1
    x0, y0 = xlim[0], ylim[0]
    x_edges = x0 + width * np.arange(nxb)
    y_edges = y0 + width * np.arange(nyb)
    z = np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=float)
    for _, row in toplot.iterrows():
        i = int(round((float(row["Simulated"]) - x0) / width))
        j = int(round((float(row["PosteriorMean"]) - y0) / width))
        if 0 <= i < z.shape[1] and 0 <= j < z.shape[0]:
            z[j, i] += float(row["x"])
    z_display = np.where(z > 0, z, np.nan)
    return x_edges, y_edges, z_display, width


def overlay_mean(
    toplot: pd.DataFrame,
    mean_table: pd.DataFrame,
    out_path: Path,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    ylabel: str,
    vmax: float,
) -> None:
    if toplot.empty:
        return
    x_edges, y_edges, z_display, width = fill_grid(toplot, xlim, ylim)
    fig, ax = plt.subplots(figsize=(3, 2.5))
    im = ax.imshow(
        z_display,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1] + width, y_edges[0], y_edges[-1] + width],
        interpolation="nearest",
        cmap="magma",
        norm=LogNorm(vmin=1, vmax=vmax, clip=True),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    x0, y0 = xlim[0], ylim[0]
    tdiag = min(xlim[1] - x0, ylim[1] - y0)
    if tdiag > 0:
        ax.plot(
            (x0, x0 + tdiag),
            (y0, y0 + tdiag),
            color="white",
            alpha=0.75,
            lw=0.5,
            zorder=2,
        )
    if not mean_table.empty:
        ax.plot(
            mean_table["SimRound"],
            mean_table["Mean"],
            color="0.45",
            linewidth=1.0,
            zorder=3,
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.colorbar(im, ax=ax, format=LogFormatterMathtext(10, labelOnlyBase=False))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_mean_mse(mean_table: pd.DataFrame, out_path: Path, *, ylabel: str) -> None:
    if mean_table.empty:
        return
    fig, ax1 = plt.subplots(figsize=(3, 2.5))
    ax1.plot(mean_table["SimRound"], mean_table["Mean"], color="C0")
    ax1.plot(mean_table["SimRound"], mean_table["MSE"] / 20, color="C0", ls="--")
    ax1.set_xlabel("Simulated Tcoal\n(2Ne generations)")
    ax1.set_ylabel(ylabel)
    ax1.set_xlim(*DEFAULT_XLIM)
    ax1.set_ylim(*DEFAULT_YLIM)
    tdiag = min(DEFAULT_XLIM[1], DEFAULT_YLIM[1])
    ax1.plot((0, tdiag), (0, tdiag), color="0.5", lw=0.8, zorder=0)
    ax2 = ax1.twinx()
    y1, y2 = ax1.get_ylim()
    ax2.set_ylim(y1 * 20, y2 * 20)
    ax2.set_ylabel("Mean square error")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_tsinferdate_plots(
    df: pd.DataFrame,
    out_prefix: Path,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlim_log: tuple[float, float],
    ylim_log: tuple[float, float],
    label: str = "tsdate",
    tag: str = "ts",
    vmax: float = 1e11,
) -> float:
    toplotall, toplotall_log = heatmap_tables_from_df(df)

    raster_heatmap(
        toplotall,
        out_prefix.parent / f"{out_prefix.name}_{tag}_pointest_lin_clean.png",
        xlim=xlim,
        ylim=ylim,
        xlabel="",
        ylabel="",
        with_colorbar=False,
        vmax=vmax,
    )
    raster_heatmap(
        toplotall_log,
        out_prefix.parent / f"{out_prefix.name}_{tag}_pointest_log_clean.png",
        xlim=xlim_log,
        ylim=ylim_log,
        xlabel="",
        ylabel="",
        with_colorbar=False,
        vmax=vmax,
    )
    raster_heatmap(
        toplotall,
        out_prefix.parent / f"{out_prefix.name}_{tag}_pointest_lin.png",
        xlim=xlim,
        ylim=ylim,
        xlabel="Simulated coalTime\n(2Ne generations)",
        ylabel=f"{label} coalTime\n(2Ne generations)",
        with_colorbar=True,
        vmax=vmax,
    )
    raster_heatmap(
        toplotall_log,
        out_prefix.parent / f"{out_prefix.name}_{tag}_pointest_log.png",
        xlim=xlim_log,
        ylim=ylim_log,
        xlabel="Simulated coalTime\n(log10 2Ne generations)",
        ylabel=f"{label} coalTime\n(log10 2Ne generations)",
        with_colorbar=True,
        vmax=vmax,
    )

    mean_table = mse_by_rounded_bins(toplotall)
    mean_table_log = mse_by_rounded_bins(toplotall_log)
    mean_table.loc[:, ["SimRound", "len", "Mean", "MSE"]].to_csv(
        out_prefix.parent / f"{out_prefix.name}meanPerSim.txt",
        sep="\t",
        index=False,
        float_format="%.10g",
    )
    mseall = (
        float(mean_table["SqErr"].sum() / mean_table["len"].sum())
        if not mean_table.empty
        else float("nan")
    )
    (out_prefix.parent / f"{out_prefix.name}MSEall.txt").write_text(
        f"{mseall}\n", encoding="utf-8"
    )
    plot_mean_mse(
        mean_table,
        out_prefix.parent / f"{out_prefix.name}meanMSE_lin.png",
        ylabel=f"{label} Tcoal\n(2Ne generations)",
    )
    overlay_mean(
        toplotall,
        mean_table,
        out_prefix.parent / f"{out_prefix.name}_{tag}_meanest_mean_lin.png",
        xlim=xlim,
        ylim=ylim,
        xlabel="Simulated coalTime\n(2Ne generations)",
        ylabel=f"{label} coalTime\n(2Ne generations)",
        vmax=vmax,
    )
    overlay_mean(
        toplotall_log,
        mean_table_log,
        out_prefix.parent / f"{out_prefix.name}meanest_mean_log.png",
        xlim=xlim_log,
        ylim=ylim_log,
        xlabel="Simulated Tcoal\n(log10 2Ne generations)",
        ylabel=f"{label} Tcoal\n(log10 2Ne gen.)",
        vmax=vmax,
    )
    print(f"Wrote {label} plots with prefix {out_prefix}", flush=True)
    return mseall


def run_singer_plots(
    df: pd.DataFrame,
    out_prefix: Path,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlim_log: tuple[float, float],
    ylim_log: tuple[float, float],
    label: str = "SINGER",
    vmax: float = 1e10,
) -> float:
    toplotall, toplotall_log = heatmap_tables_from_df(df)
    ylabel = f"{label} Tcoal\n(2Ne generations)"
    ylabel_log = f"{label} Tcoal\n(log10 2Ne gen.)"

    raster_heatmap(
        toplotall,
        out_prefix.parent / f"{out_prefix.name}meanest_lin_clean.png",
        xlim=xlim,
        ylim=ylim,
        xlabel="",
        ylabel="",
        with_colorbar=False,
        vmax=vmax,
    )
    raster_heatmap(
        toplotall_log,
        out_prefix.parent / f"{out_prefix.name}meanest_log_clean.png",
        xlim=xlim_log,
        ylim=ylim_log,
        xlabel="",
        ylabel="",
        with_colorbar=False,
        vmax=vmax,
    )
    raster_heatmap(
        toplotall,
        out_prefix.parent / f"{out_prefix.name}meanest_lin.png",
        xlim=xlim,
        ylim=ylim,
        xlabel="Simulated Tcoal\n(2Ne generations)",
        ylabel=ylabel,
        with_colorbar=True,
        vmax=vmax,
    )
    raster_heatmap(
        toplotall_log,
        out_prefix.parent / f"{out_prefix.name}meanest_log.png",
        xlim=xlim_log,
        ylim=ylim_log,
        xlabel="Simulated Tcoal\n(log10 2Ne generations)",
        ylabel=ylabel_log,
        with_colorbar=True,
        vmax=vmax,
    )

    mean_table = mse_by_segments(df)
    mseall = (
        float(mean_table["WtdSqErr"].sum() / mean_table["len"].sum())
        if not mean_table.empty
        else float("nan")
    )
    (out_prefix.parent / f"{out_prefix.name}MSEall.txt").write_text(
        f"{mseall}\n", encoding="utf-8"
    )
    plot_mean_mse(
        mean_table,
        out_prefix.parent / f"{out_prefix.name}MeanMSE_lin.png",
        ylabel=ylabel,
    )
    overlay_mean(
        toplotall,
        mean_table,
        out_prefix.parent / f"{out_prefix.name}meanest_mean_lin.png",
        xlim=xlim,
        ylim=ylim,
        xlabel="Simulated Tcoal\n(2Ne generations)",
        ylabel=ylabel,
        vmax=vmax,
    )
    print(f"Wrote {label} plots with prefix {out_prefix}", flush=True)
    return mseall


def _finite_weighted_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (
        np.isfinite(df["Simulated"].to_numpy(dtype=float))
        & np.isfinite(df["PosteriorMean"].to_numpy(dtype=float))
        & np.isfinite(df["len"].to_numpy(dtype=float))
        & (df["len"].to_numpy(dtype=float) > 0)
    )
    return (
        df.loc[mask, "Simulated"].to_numpy(dtype=float),
        df.loc[mask, "PosteriorMean"].to_numpy(dtype=float),
        df.loc[mask, "len"].to_numpy(dtype=float),
    )


def common_metric_values(df: pd.DataFrame, legacy_mse: float) -> dict[str, float | str]:
    sim, post, weights = _finite_weighted_arrays(df)
    total_weight = float(weights.sum()) if len(weights) else 0.0
    if total_weight <= 0:
        return {
            "n_segments": int(len(df)),
            "total_length": 0.0,
            "weighted_mse": float("nan"),
            "weighted_rmse": float("nan"),
            "weighted_log_rmse": float("nan"),
            "weighted_log_correlation": float("nan"),
            "weighted_spearman_correlation": float("nan"),
            "posterior_95pct_coverage": float("nan"),
            "weighted_mae": float("nan"),
            "weighted_bias": float("nan"),
            "legacy_mseall": legacy_mse,
        }
    diff = post - sim
    mse = float(np.sum((diff**2) * weights) / total_weight)
    positive = (sim > 0) & (post > 0)
    if np.any(positive):
        log_weights = weights[positive]
        log_diff = np.log(post[positive]) - np.log(sim[positive])
        log_rmse = math.sqrt(
            float(np.sum((log_diff**2) * log_weights) / log_weights.sum())
        )
    else:
        log_rmse = float("nan")
    if np.any(positive) and np.sum(positive) >= 2:
        log_sim = np.log(sim[positive])
        log_post = np.log(post[positive])
        log_weights = weights[positive]
        log_corr = _weighted_correlation(log_sim, log_post, log_weights)
    else:
        log_corr = float("nan")
    sim_ranks = pd.Series(sim).rank(method="average").to_numpy(dtype=float)
    post_ranks = pd.Series(post).rank(method="average").to_numpy(dtype=float)
    spearman = _weighted_correlation(sim_ranks, post_ranks, weights)
    coverage = float("nan")
    if {"PosteriorLow", "PosteriorHigh"}.issubset(df.columns):
        finite_frame = df.loc[
            np.isfinite(df["Simulated"].to_numpy(dtype=float))
            & np.isfinite(df["PosteriorLow"].to_numpy(dtype=float))
            & np.isfinite(df["PosteriorHigh"].to_numpy(dtype=float))
            & np.isfinite(df["len"].to_numpy(dtype=float))
            & (df["len"].to_numpy(dtype=float) > 0)
        ]
        if not finite_frame.empty:
            covered = (
                (finite_frame["Simulated"] >= finite_frame["PosteriorLow"])
                & (finite_frame["Simulated"] <= finite_frame["PosteriorHigh"])
            ).to_numpy(dtype=float)
            coverage_weights = finite_frame["len"].to_numpy(dtype=float)
            coverage = float(np.average(covered, weights=coverage_weights))
    return {
        "n_segments": int(len(df)),
        "total_length": total_weight,
        "weighted_mse": mse,
        "weighted_rmse": math.sqrt(mse),
        "weighted_log_rmse": log_rmse,
        "weighted_log_correlation": log_corr,
        "weighted_spearman_correlation": spearman,
        "posterior_95pct_coverage": coverage,
        "weighted_mae": float(np.sum(np.abs(diff) * weights) / total_weight),
        "weighted_bias": float(np.sum(diff * weights) / total_weight),
        "legacy_mseall": legacy_mse,
    }


def _weighted_correlation(
    left: np.ndarray, right: np.ndarray, weights: np.ndarray
) -> float:
    total = float(weights.sum())
    if total <= 0 or len(weights) < 2:
        return float("nan")
    left_mean = float(np.sum(left * weights) / total)
    right_mean = float(np.sum(right * weights) / total)
    left_centered = left - left_mean
    right_centered = right - right_mean
    covariance = float(np.sum(weights * left_centered * right_centered) / total)
    left_variance = float(np.sum(weights * left_centered**2) / total)
    right_variance = float(np.sum(weights * right_centered**2) / total)
    denominator = math.sqrt(left_variance * right_variance)
    return covariance / denominator if denominator > 0 else float("nan")


def write_common_metrics(
    df: pd.DataFrame, out_prefix: Path, *, method_label: str, legacy_mse: float
) -> None:
    metrics = common_metric_values(df, legacy_mse)
    rows = [{"metric": "method", "value": method_label}]
    rows.extend({"metric": key, "value": value} for key, value in metrics.items())
    pd.DataFrame(rows).to_csv(
        out_prefix.parent / f"{out_prefix.name}commonMetrics.tsv",
        sep="\t",
        index=False,
    )


def _summary_stats(values: pd.Series) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return {
            "n": 0,
            "min": float("nan"),
            "q01": float("nan"),
            "q05": float("nan"),
            "median": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
            "max": float("nan"),
        }
    quantiles = finite.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "n": int(finite.shape[0]),
        "min": float(finite.min()),
        "q01": float(quantiles.loc[0.01]),
        "q05": float(quantiles.loc[0.05]),
        "median": float(quantiles.loc[0.5]),
        "q95": float(quantiles.loc[0.95]),
        "q99": float(quantiles.loc[0.99]),
        "max": float(finite.max()),
    }


def _count_outside_limits(
    df: pd.DataFrame, xlim: tuple[float, float], ylim: tuple[float, float]
) -> int:
    return int(
        (
            (df["Simulated"] < xlim[0])
            | (df["Simulated"] > xlim[1])
            | (df["PosteriorMean"] < ylim[0])
            | (df["PosteriorMean"] > ylim[1])
        ).sum()
    )


def _count_outside_log_limits(
    df: pd.DataFrame, xlim_log: tuple[float, float], ylim_log: tuple[float, float]
) -> int:
    positive = (df["Simulated"] > 0) & (df["PosteriorMean"] > 0)
    if not positive.any():
        return 0
    log_sim = np.log10(df.loc[positive, "Simulated"].to_numpy(dtype=float))
    log_post = np.log10(df.loc[positive, "PosteriorMean"].to_numpy(dtype=float))
    return int(
        (
            (log_sim < xlim_log[0])
            | (log_sim > xlim_log[1])
            | (log_post < ylim_log[0])
            | (log_post > ylim_log[1])
        ).sum()
    )


def write_time_summary(
    df: pd.DataFrame,
    out_prefix: Path,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlim_log: tuple[float, float],
    ylim_log: tuple[float, float],
    method_label: str,
) -> None:
    rows: list[dict[str, str | float | int]] = [
        {"series": "metadata", "metric": "method", "value": method_label},
        {"series": "metadata", "metric": "units", "value": "t/(2Ne)"},
    ]
    for series, column in (
        ("Simulated", "Simulated"),
        ("PosteriorMean", "PosteriorMean"),
    ):
        stats = _summary_stats(df[column])
        rows.extend(
            {"series": series, "metric": metric, "value": value}
            for metric, value in stats.items()
        )
    rows.extend(
        [
            {
                "series": "plot_limits",
                "metric": "linear_x",
                "value": f"{xlim[0]:.10g},{xlim[1]:.10g}",
            },
            {
                "series": "plot_limits",
                "metric": "linear_y",
                "value": f"{ylim[0]:.10g},{ylim[1]:.10g}",
            },
            {
                "series": "plot_limits",
                "metric": "log10_x",
                "value": f"{xlim_log[0]:.10g},{xlim_log[1]:.10g}",
            },
            {
                "series": "plot_limits",
                "metric": "log10_y",
                "value": f"{ylim_log[0]:.10g},{ylim_log[1]:.10g}",
            },
            {
                "series": "clipping",
                "metric": "linear_points_outside",
                "value": _count_outside_limits(df, xlim, ylim),
            },
            {
                "series": "clipping",
                "metric": "log10_points_outside",
                "value": _count_outside_log_limits(df, xlim_log, ylim_log),
            },
            {"series": "clipping", "metric": "total_rows", "value": len(df)},
        ]
    )
    pd.DataFrame(rows).to_csv(
        out_prefix.parent / f"{out_prefix.name}timeSummary.tsv",
        sep="\t",
        index=False,
    )
    for column in ("Simulated", "PosteriorMean"):
        stats = _summary_stats(df[column])
        print(
            f"{column}: n={stats['n']} min={stats['min']:.10g} "
            f"median={stats['median']:.10g} max={stats['max']:.10g}",
            flush=True,
        )


def write_standard_outputs(
    df: pd.DataFrame,
    out_prefix: Path,
    *,
    method_label: str,
    legacy_mse: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlim_log: tuple[float, float],
    ylim_log: tuple[float, float],
) -> None:
    write_common_metrics(df, out_prefix, method_label=method_label, legacy_mse=legacy_mse)
    write_time_summary(
        df,
        out_prefix,
        xlim=xlim,
        ylim=ylim,
        xlim_log=xlim_log,
        ylim_log=ylim_log,
        method_label=method_label,
    )
