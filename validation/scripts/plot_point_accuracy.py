#!/usr/bin/env python3
"""Point-accuracy evaluation for tsinfer+tsdate vs Singer (ARGsims paper metrics).

Builds aligned per-segment tables from tree sequences + truth tracks, then plots
heatmaps, weighted means, and MSE using the same logic as:

- tsinfer: scripts/tsinferdate/4_plot_pointestimates_tsinfer.R
- singer:  scripts/argweaver/plot_pointestimates.R (ARGweaver posterior-BED workflow)

Segment alignment is ported from ARG-Optimise validation_scripts/validate.py.

Example (tsinfer)::

    python3 plot_point_accuracy.py --method tsinfer \\
      --inferred-trees .../output/tsinfer/l1mb_dated.trees \\
      --truth-dir .../tcoalmsp/rep0 --truth-prefix sim_l1mb_0 \\
      --ne 10000 -n 8 -o .../out_tsinfer/eval

Example (singer)::

    python3 plot_point_accuracy.py --method singer \\
      --input-dir .../singer_trees --sample-prefix sim_l1mb_0_ \\
      --truth-dir .../tcoalmsp/rep0 --truth-prefix sim_l1mb_0 \\
      --ne 10000 -n 8 -o .../out_singer/eval
"""

from __future__ import annotations

import argparse
import gzip
import math
import os
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

    @property
    def length(self) -> float:
        return self.right - self.left


def parse_pair_from_truth_name(path: Path) -> tuple[int, int]:
    match = PAIR_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse pair from truth filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


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
                tcoal = float(fields[2]) / scale
                intervals.append(
                    TruthInterval(left=left, right=right, tcoal_2ne=tcoal)
                )
        if not intervals:
            raise ValueError(f"Truth file has no valid intervals: {truth_file}")
        tracks[pair] = intervals
    return tracks


def load_truth_tracks_from_trees(
    truth_trees: Path,
    pairs: Iterable[tuple[int, int]],
    ne: float,
) -> dict[tuple[int, int], list[TruthInterval]]:
    ts = tskit.load(truth_trees)
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
                    left=float(left), right=float(right), tcoal_2ne=float(tcoal)
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
    return [tskit.load(path) for path in sample_paths]


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
            )
        )
    return segments


def iter_pairs(nspl: int, skip: int) -> list[tuple[int, int]]:
    return [
        (s1, s2)
        for s1 in range(0, nspl - 1, skip)
        for s2 in range(s1 + 1, nspl, skip)
    ]


def segments_to_dataframe(segments: list[PairSegment]) -> pd.DataFrame:
    if not segments:
        return pd.DataFrame(
            columns=["chr", "start", "end", "Simulated", "PosteriorMean", "len"]
        )
    return pd.DataFrame(
        {
            "chr": "1",
            "start": [int(round(s.left)) for s in segments],
            "end": [int(round(s.right)) for s in segments],
            "Simulated": [s.truth for s in segments],
            "PosteriorMean": [s.posterior_mean for s in segments],
            "len": [s.length for s in segments],
        }
    )


def load_bed_pair(path: Path, *, has_median: bool) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt"
    with opener(path, mode, encoding="utf-8") as fh:
        if has_median:
            names = ["chr", "start", "end", "Simulated", "PosteriorMean", "PosteriorMedian"]
            usecols = range(6)
        else:
            names = ["chr", "start", "end", "Simulated", "PosteriorMean"]
            usecols = range(5)
        df = pd.read_csv(fh, sep="\t", header=None, usecols=usecols, names=names)
    df = df[df["end"] > df["start"]].copy()
    df["len"] = (df["end"] - df["start"]).astype(float)
    return df


def _aggregate_pairs_linear(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            [df["Simulated"].round(1), df["PosteriorMean"].round(1)], as_index=False
        )["len"]
        .sum()
        .rename(columns={"len": "x"})
    )


def _aggregate_pairs_log10(df: pd.DataFrame) -> pd.DataFrame:
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


def _mse_by_bin_tsinfer(toplot: pd.DataFrame) -> pd.DataFrame:
    """ddply(toplotall, .(Simulated), ...) from 4_plot_pointestimates_tsinfer.R."""
    if toplot.empty:
        return pd.DataFrame(columns=["SimRound", "len", "Mean", "SqErr", "MSE"])
    rec: list[dict] = []
    for sim, g in toplot.groupby("Simulated"):
        w = g["x"].to_numpy(dtype=float)
        post = g["PosteriorMean"].to_numpy(dtype=float)
        svec = g["Simulated"].to_numpy(dtype=float)
        len_sum = w.sum()
        if len_sum <= 0:
            continue
        mean = float(np.average(post, weights=w))
        sq = float(((post - svec) ** 2 * w).sum())
        rec.append(
            {
                "SimRound": sim,
                "len": len_sum,
                "Mean": mean,
                "SqErr": sq,
                "MSE": sq / len_sum,
            }
        )
    return pd.DataFrame(rec).sort_values("SimRound").reset_index(drop=True)


def _mse_by_segments_singer(df: pd.DataFrame) -> pd.DataFrame:
    """ARGweaver plot_pointestimates.R segment-level group_by(SimRound)."""
    if df.empty:
        return pd.DataFrame(columns=["SimRound", "len", "Mean", "MSE", "WtdSqErr"])
    work = df.copy()
    work["SimRound"] = work["Simulated"].round(1)
    work["WtdMean"] = work["PosteriorMean"] * work["len"]
    work["WtdSqErr"] = (work["PosteriorMean"] - work["Simulated"]) ** 2 * work["len"]
    g = (
        work.groupby("SimRound", as_index=False)
        .agg(WtdMean=("WtdMean", "sum"), WtdSqErr=("WtdSqErr", "sum"), len=("len", "sum"))
    )
    g["Mean"] = g["WtdMean"] / g["len"]
    g["MSE"] = g["WtdSqErr"] / g["len"]
    return g.sort_values("SimRound").reset_index(drop=True)


def _raster_heatmap(
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
    w = 0.1
    nxb = int(round((xlim[1] - xlim[0]) / w)) + 1
    nyb = int(round((ylim[1] - ylim[0]) / w)) + 1
    x0, y0 = xlim[0], ylim[0]
    x_edges = x0 + w * np.arange(nxb)
    y_edges = y0 + w * np.arange(nyb)
    z = np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=float)
    for _, r in toplot.iterrows():
        i = int(round((float(r["Simulated"]) - x0) / w))
        j = int(round((float(r["PosteriorMean"]) - y0) / w))
        if 0 <= i < z.shape[1] and 0 <= j < z.shape[0]:
            z[j, i] += float(r["x"])
    zd = np.where(z > 0, z, np.nan)
    fig, ax = plt.subplots(figsize=(2 if not with_colorbar else 3, 2.5 if with_colorbar else 2.0))
    im = ax.imshow(
        zd,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1] + w, y_edges[0], y_edges[-1] + w],
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


def _fill_grid(
    toplot: pd.DataFrame, xlim: tuple[float, float], ylim: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    w = 0.1
    nxb = int(round((xlim[1] - xlim[0]) / w)) + 1
    nyb = int(round((ylim[1] - ylim[0]) / w)) + 1
    x0, y0 = xlim[0], ylim[0]
    x_edges = x0 + w * np.arange(nxb)
    y_edges = y0 + w * np.arange(nyb)
    z = np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=float)
    for _, r in toplot.iterrows():
        i = int(round((float(r["Simulated"]) - x0) / w))
        j = int(round((float(r["PosteriorMean"]) - y0) / w))
        if 0 <= i < z.shape[1] and 0 <= j < z.shape[0]:
            z[j, i] += float(r["x"])
    zd = np.where(z > 0, z, np.nan)
    return x_edges, y_edges, zd, w


def _overlay_mean(
    toplot: pd.DataFrame,
    m2: pd.DataFrame,
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
    x_edges, y_edges, zd, w = _fill_grid(toplot, xlim, ylim)
    fig, ax = plt.subplots(figsize=(3, 2.5))
    im = ax.imshow(
        zd,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1] + w, y_edges[0], y_edges[-1] + w],
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
            (x0, x0 + tdiag), (y0, y0 + tdiag), color="white", alpha=0.75, lw=0.5, zorder=2
        )
    if not m2.empty:
        ax.plot(m2["SimRound"], m2["Mean"], color="0.45", linewidth=1.0, zorder=3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.colorbar(im, ax=ax, format=LogFormatterMathtext(10, labelOnlyBase=False))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_mean_mse_tsinfer(
    m2: pd.DataFrame, out_path: Path, *, ylabel: str
) -> None:
    if m2.empty:
        return
    fig, ax1 = plt.subplots(figsize=(3, 2.5))
    ax1.plot(m2["SimRound"], m2["Mean"], color="C0")
    ax1.plot(m2["SimRound"], m2["MSE"] / 20, color="C0", ls="--")
    ax1.set_xlabel("Simulated Tcoal\n(2Ne generations)")
    ax1.set_ylabel(ylabel)
    ax1.set_xlim(0, 16)
    ax1.set_ylim(0, 8)
    tdiag = min(8.0, 16.0)
    ax1.plot((0, tdiag), (0, tdiag), color="0.5", lw=0.8, zorder=0)
    ax2 = ax1.twinx()
    y1, y2 = ax1.get_ylim()
    ax2.set_ylim(y1 * 20, y2 * 20)
    ax2.set_ylabel("Mean square error")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_mean_mse_singer(
    means: pd.DataFrame, out_path: Path, *, ylabel: str
) -> None:
    if means.empty:
        return
    fig, ax1 = plt.subplots(figsize=(3, 2.5))
    ax1.plot(means["SimRound"], means["Mean"], color="C0")
    ax1.plot(means["SimRound"], means["MSE"] / 20, color="C0", ls="--")
    ax1.set_xlabel("Simulated Tcoal\n(2Ne generations)")
    ax1.set_ylabel(ylabel)
    ax1.set_xlim(0, 16)
    ax1.set_ylim(0, 8)
    tdiag = min(8.0, 16.0)
    ax1.plot((0, tdiag), (0, tdiag), color="0.5", lw=0.8, zorder=0)
    ax2 = ax1.twinx()
    y1, y2 = ax1.get_ylim()
    ax2.set_ylim(y1 * 20, y2 * 20)
    ax2.set_ylabel("Mean square error")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _collect_segments_from_trees(
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
            print(f"pair {pi}/{len(pairs)} {pair}: aligning segments …", flush=True)
        segs = combine_pair_segments(
            pair, truth_tracks[pair], inferred_trees, ne
        )
        all_segments.extend(segs)
    if not all_segments:
        raise RuntimeError("No aligned segments were generated.")
    return all_segments


def _heatmap_tables_from_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    chunk_lin = _aggregate_pairs_linear(df)
    chunk_log = _aggregate_pairs_log10(df)
    return chunk_lin, chunk_log


def _merge_heatmap_chunks(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    if not chunks:
        return pd.DataFrame(columns=["Simulated", "PosteriorMean", "x"])
    return (
        pd.concat(chunks, ignore_index=True)
        .groupby(["Simulated", "PosteriorMean"], as_index=False)["x"]
        .sum()
    )


def run_plots_tsinfer(
    df: pd.DataFrame,
    out_prefix: Path,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlim_log: tuple[float, float],
    ylim_log: tuple[float, float],
) -> float:
    toplotall, toplotall_log = _heatmap_tables_from_df(df)
    pref = str(out_prefix)
    vmax = 1e11

    _raster_heatmap(
        toplotall,
        out_prefix.parent / f"{out_prefix.name}_ts_pointest_lin_clean.png",
        xlim=xlim,
        ylim=ylim,
        xlabel="",
        ylabel="",
        with_colorbar=False,
        vmax=vmax,
    )
    _raster_heatmap(
        toplotall_log,
        out_prefix.parent / f"{out_prefix.name}_ts_pointest_log_clean.png",
        xlim=xlim_log,
        ylim=ylim_log,
        xlabel="",
        ylabel="",
        with_colorbar=False,
        vmax=vmax,
    )
    _raster_heatmap(
        toplotall,
        out_prefix.parent / f"{out_prefix.name}_ts_pointest_lin.png",
        xlim=xlim,
        ylim=ylim,
        xlabel="Simulated coalTime\n(2Ne generations)",
        ylabel="tsdate coalTime\n(2Ne generations)",
        with_colorbar=True,
        vmax=vmax,
    )
    _raster_heatmap(
        toplotall_log,
        out_prefix.parent / f"{out_prefix.name}_ts_pointest_log.png",
        xlim=xlim_log,
        ylim=ylim_log,
        xlabel="Simulated coalTime\n(log10 2Ne generations)",
        ylabel="tsdate coalTime\n(log10 2Ne generations)",
        with_colorbar=True,
        vmax=vmax,
    )

    toplot2 = _mse_by_bin_tsinfer(toplotall)
    toplot_log2 = _mse_by_bin_tsinfer(toplotall_log)
    toplot2.loc[:, ["SimRound", "len", "Mean", "MSE"]].to_csv(
        out_prefix.parent / f"{out_prefix.name}meanPerSim.txt",
        sep="\t",
        index=False,
        float_format="%.10g",
    )
    mseall = (
        float(toplot2["SqErr"].sum() / toplot2["len"].sum())
        if not toplot2.empty
        else float("nan")
    )
    (out_prefix.parent / f"{out_prefix.name}MSEall.txt").write_text(
        f"{mseall}\n", encoding="utf-8"
    )
    _plot_mean_mse_tsinfer(
        toplot2,
        out_prefix.parent / f"{out_prefix.name}meanMSE_lin.png",
        ylabel="tsdate Tcoal\n(2Ne generations)",
    )
    _overlay_mean(
        toplotall,
        toplot2,
        out_prefix.parent / f"{out_prefix.name}_ts_meanest_mean_lin.png",
        xlim=xlim,
        ylim=ylim,
        xlabel="Simulated coalTime\n(2Ne generations)",
        ylabel="tsdate coalTime\n(2Ne generations)",
        vmax=vmax,
    )
    _overlay_mean(
        toplotall_log,
        toplot_log2,
        out_prefix.parent / f"{out_prefix.name}meanest_mean_log.png",
        xlim=xlim_log,
        ylim=ylim_log,
        xlabel="Simulated Tcoal\n(log10 2Ne generations)",
        ylabel="ARGweaver Tcoal\n(log10 2Ne gen.)",
        vmax=vmax,
    )
    print(f"Wrote tsinfer plots with prefix {pref}", flush=True)
    return mseall


def run_plots_singer(
    df: pd.DataFrame,
    out_prefix: Path,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlim_log: tuple[float, float],
    ylim_log: tuple[float, float],
) -> float:
    chunk_lin = _aggregate_pairs_linear(df)
    chunk_log = _aggregate_pairs_log10(df)
    toplotall = chunk_lin
    toplotall_log = chunk_log
    vmax = 1e10
    ylabel_singer = "SINGER Tcoal\n(2Ne generations)"
    ylabel_singer_log = "SINGER Tcoal\n(log10 2Ne gen.)"

    _raster_heatmap(
        toplotall,
        out_prefix.parent / f"{out_prefix.name}meanest_lin_clean.png",
        xlim=xlim,
        ylim=ylim,
        xlabel="",
        ylabel="",
        with_colorbar=False,
        vmax=vmax,
    )
    _raster_heatmap(
        toplotall_log,
        out_prefix.parent / f"{out_prefix.name}meanest_log_clean.png",
        xlim=xlim_log,
        ylim=ylim_log,
        xlabel="",
        ylabel="",
        with_colorbar=False,
        vmax=vmax,
    )
    _raster_heatmap(
        toplotall,
        out_prefix.parent / f"{out_prefix.name}meanest_lin.png",
        xlim=xlim,
        ylim=ylim,
        xlabel="Simulated Tcoal\n(2Ne generations)",
        ylabel=ylabel_singer,
        with_colorbar=True,
        vmax=vmax,
    )
    _raster_heatmap(
        toplotall_log,
        out_prefix.parent / f"{out_prefix.name}meanest_log.png",
        xlim=xlim_log,
        ylim=ylim_log,
        xlabel="Simulated Tcoal\n(log10 2Ne generations)",
        ylabel=ylabel_singer_log,
        with_colorbar=True,
        vmax=vmax,
    )

    meansall = _mse_by_segments_singer(df)
    mseall = (
        float(meansall["WtdSqErr"].sum() / meansall["len"].sum())
        if not meansall.empty
        else float("nan")
    )
    (out_prefix.parent / f"{out_prefix.name}MSEall.txt").write_text(
        f"{mseall}\n", encoding="utf-8"
    )
    _plot_mean_mse_singer(
        meansall,
        out_prefix.parent / f"{out_prefix.name}MeanMSE_lin.png",
        ylabel=ylabel_singer,
    )
    _overlay_mean(
        toplotall,
        meansall,
        out_prefix.parent / f"{out_prefix.name}meanest_mean_lin.png",
        xlim=xlim,
        ylim=ylim,
        xlabel="Simulated Tcoal\n(2Ne generations)",
        ylabel=ylabel_singer,
        vmax=vmax,
    )
    print(f"Wrote singer plots with prefix {out_prefix}", flush=True)
    return mseall


def load_segments_from_bed(
    *,
    method: str,
    bed_dir: Path,
    bed_pref: str,
    nspl: int,
    skip: int,
    mcspl: str | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for s1, s2 in iter_pairs(nspl, skip):
        pair = f"{s1}-{s2}"
        if method == "tsinfer":
            path = bed_dir / f"{bed_pref}{pair}_post.bed.gz"
            if not path.is_file():
                path = bed_dir / f"{bed_pref}{pair}_post.bed"
            df = load_bed_pair(path, has_median=False)
        else:
            if mcspl is None:
                raise SystemExit("--mcspl required for --from-bed with method singer")
            path = bed_dir / f"arg-sample_sim_{bed_pref}_{pair}_Tcpair_spl{mcspl}_posterior.bed"
            if not path.is_file():
                path = bed_dir / f"{bed_pref}_pair_{pair}_posterior.bed"
            df = load_bed_pair(path, has_median=True)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError("No BED segments loaded.")
    return pd.concat(frames, ignore_index=True)


def build_truth_tracks(
    args: argparse.Namespace, pairs: list[tuple[int, int]]
) -> dict[tuple[int, int], list[TruthInterval]]:
    truth_trees = resolve_truth_trees_path(
        args.truth_trees.expanduser() if args.truth_trees else None
    )
    if truth_trees is not None:
        return load_truth_tracks_from_trees(truth_trees, pairs, args.ne)
    return load_truth_tracks(args.truth_dir, args.truth_prefix, args.ne)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Point-accuracy heatmaps and MSE for tsinfer or Singer vs msprime truth."
    )
    ap.add_argument(
        "--method",
        choices=("tsinfer", "singer"),
        default="tsinfer",
        help="Inference method (default: tsinfer).",
    )
    ap.add_argument(
        "--truth-dir",
        type=Path,
        required=True,
        help="Directory of truth *_splsX-Y.tc tracks.",
    )
    ap.add_argument(
        "--truth-prefix",
        required=True,
        help="Truth filename stem before _spls (e.g. sim_l1mb_0).",
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
        help="Effective population size Ne; times divided by 2*Ne.",
    )
    ap.add_argument("-n", "--nspl", type=int, required=True, help="Number of haplotypes.")
    ap.add_argument(
        "-s", "--skip", type=int, default=1, help="Stride on pair indices (default 1)."
    )
    ap.add_argument(
        "-o",
        "--output-prefix",
        type=Path,
        required=True,
        help="Output path prefix (directory created; files are <prefix><suffix>).",
    )
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--pair-seed", type=int, default=42)
    ap.add_argument(
        "--from-bed",
        action="store_true",
        help="Read pre-built BEDs instead of tree sequences (audit mode).",
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
        help="MCMC label for singer ARGweaver-style BED names (--from-bed).",
    )
    ap.add_argument("--xlim", default="0,16")
    ap.add_argument("--ylim", default="0,8")
    ap.add_argument("--xlim-log", default="-4,1.5")
    ap.add_argument("--ylim-log", default="-4,1.5")
    ap.add_argument("-v", "--verbose", action="store_true")

    ap.add_argument(
        "--inferred-trees",
        type=Path,
        default=None,
        help="[tsinfer] Single dated .trees file.",
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="[singer] Directory of posterior .trees samples.",
    )
    ap.add_argument(
        "--sample-prefix",
        default=None,
        help="[singer] Filename prefix before *.trees.",
    )
    ap.add_argument("--burnin-samples", type=int, default=0)
    ap.add_argument("--max-posterior-samples", type=int, default=None)

    args = ap.parse_args()
    if args.from_bed:
        if args.bed_prefix is None:
            raise SystemExit("--bed-prefix is required with --from-bed")
    elif args.method == "tsinfer":
        if args.inferred_trees is None:
            raise SystemExit("--inferred-trees is required for method tsinfer")
    else:
        if args.input_dir is None or args.sample_prefix is None:
            raise SystemExit("--input-dir and --sample-prefix are required for method singer")

    return args


def main() -> None:
    args = parse_args()
    out_prefix = args.output_prefix.expanduser().resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    xlim = tuple(map(float, args.xlim.split(",")))
    ylim = tuple(map(float, args.ylim.split(",")))
    xlim_log = tuple(map(float, args.xlim_log.split(",")))
    ylim_log = tuple(map(float, args.ylim_log.split(",")))

    if args.from_bed:
        bed_dir = args.bed_dir or out_prefix.parent
        df = load_segments_from_bed(
            method=args.method,
            bed_dir=bed_dir,
            bed_pref=args.bed_prefix,
            nspl=args.nspl,
            skip=args.skip,
            mcspl=args.mcspl,
        )
    else:
        all_truth = build_truth_tracks(args, list(iter_pairs(args.nspl, args.skip)))
        pairs = select_pairs(all_truth.keys(), args.max_pairs, args.pair_seed)
        pairs = [p for p in pairs if p in iter_pairs(args.nspl, args.skip)]
        if args.method == "tsinfer":
            inferred = tskit.load(args.inferred_trees)
            if args.verbose:
                print(f"Loaded inferred TS: {args.inferred_trees}", flush=True)
            segments = _collect_segments_from_trees(
                truth_tracks=all_truth,
                pairs=pairs,
                inferred_trees=[inferred],
                ne=args.ne,
                verbose=args.verbose,
            )
        else:
            samples = load_posterior_tree_samples(
                args.input_dir,
                args.sample_prefix,
                burnin_samples=args.burnin_samples,
                max_posterior_samples=args.max_posterior_samples,
            )
            if args.verbose:
                print(f"Loaded {len(samples)} posterior tree sequence(s)", flush=True)
            segments = _collect_segments_from_trees(
                truth_tracks=all_truth,
                pairs=pairs,
                inferred_trees=samples,
                ne=args.ne,
                verbose=args.verbose,
            )
        df = segments_to_dataframe(segments)

    print(f"segments: {len(df)} rows", flush=True)

    if args.method == "tsinfer":
        mse = run_plots_tsinfer(
            df,
            out_prefix,
            xlim=xlim,
            ylim=ylim,
            xlim_log=xlim_log,
            ylim_log=ylim_log,
        )
    else:
        mse = run_plots_singer(
            df,
            out_prefix,
            xlim=xlim,
            ylim=ylim,
            xlim_log=xlim_log,
            ylim_log=ylim_log,
        )
    print(f"MSEall = {mse}", flush=True)


if __name__ == "__main__":
    main()
