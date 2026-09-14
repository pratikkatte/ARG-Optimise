#!/usr/bin/env python3
"""Compatibility CLI for shared exact-span TMRCA metrics."""
import argparse
import json
from pathlib import Path
import sys
import tskit

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.posterior_summary import (
    TruthInterval, PairSegment, iter_pairs, _truth_value_at, _posterior_values_at, combine_pair_segments, collect_segments_from_trees, segments_to_dataframe, _finite_weighted_arrays, common_metric_values, COPIED_FROM_SHA256, aligned_pair_times, point_accuracy_metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--truth-trees', type=Path, required=True)
    parser.add_argument('--inferred-trees', type=Path, nargs='+', required=True)
    parser.add_argument('--ne', type=float, default=10000.)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    truth = tskit.load(args.truth_trees)
    posterior = [tskit.load(path) for path in args.inferred_trees]
    metrics, details, frame = point_accuracy_metrics(truth, posterior, args.ne)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(str(args.output_prefix)+'_segments.tsv', sep='\t', index=False)
    Path(str(args.output_prefix)+'_metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
    Path(str(args.output_prefix)+'_details.json').write_text(json.dumps(details, allow_nan=False))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
