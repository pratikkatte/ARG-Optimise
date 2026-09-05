#!/usr/bin/env python3
"""Shared point-accuracy helpers for ARGsims-style validation scripts."""

from __future__ import annotations

import argparse
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import tskit

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

    @property
    def length(self) -> float:
        return self.right - self.left


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

