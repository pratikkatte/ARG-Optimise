import argparse
import sys
from pathlib import Path
import math

import numpy as np
import pandas as pd
import tskit

from .inputs import (
    PairSegment, TruthInterval, build_truth_tracks, iter_pairs,
    load_tree_sequence, require_truth_args, select_pairs,
)

SAMPLE_RE = __import__("re").compile(r"_(\d+)\.trees$")

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
            "len": [s.length for s in segments],
        },
        columns=columns,
    )

