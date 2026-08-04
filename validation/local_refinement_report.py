"""Offline source-versus-selected reports for local ARG refinement inference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import tskit

from .scripts.point_accuracy_common import (
    clip_dataframe_to_region,
    common_metric_values,
    dataframe_from_tree_sequences,
)


def build_local_refinement_report(
    manifest_path,
    *,
    truth_trees=None,
    ne=10_000.0,
    max_pairs=None,
    pair_seed=42,
    skip=1,
):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_record = manifest.get("source_arg") or {}
    source_path = source_record.get("path") if isinstance(source_record, dict) else source_record
    rows = []
    for request in manifest.get("requests", ()):
        request_spec = request.get("request") or {}
        left, right = (float(value) for value in request_spec["genomic_range"])
        row = {
            "request_id": request["id"],
            "selected_source": bool(request.get("selected_source", True)),
            **dict(request.get("evaluation") or {}),
        }
        if truth_trees is not None:
            args = SimpleNamespace(
                truth_trees=Path(truth_trees), truth_dir=None, truth_prefix=None,
                ne=float(ne), nspl=8, skip=int(skip), max_pairs=max_pairs,
                pair_seed=int(pair_seed), verbose=False,
            )
            for label, path in (
                ("source", source_path),
                ("selected", request["selected_output_file"]),
            ):
                tree_sequence = tskit.load(str(path))
                frame = dataframe_from_tree_sequences(args, [tree_sequence])
                frame = clip_dataframe_to_region(frame, left, right)
                metrics = common_metric_values(frame, legacy_mse=float("nan"))
                for name, value in metrics.items():
                    row[f"{label}_{name}"] = value
            for name in (
                "weighted_mse", "weighted_rmse", "weighted_log_rmse",
                "weighted_mae", "weighted_bias", "weighted_log_correlation",
                "weighted_spearman_correlation", "posterior_95pct_coverage",
            ):
                source_value = row.get(f"source_{name}")
                selected_value = row.get(f"selected_{name}")
                row[f"selected_minus_source_{name}"] = (
                    float(selected_value) - float(source_value)
                    if source_value is not None and selected_value is not None
                    and math.isfinite(float(source_value))
                    and math.isfinite(float(selected_value))
                    else float("nan")
                )
        rows.append(row)
    return {
        "manifest": str(manifest_path.resolve()),
        "truth_trees": None if truth_trees is None else str(Path(truth_trees).resolve()),
        "requests": rows,
    }


def write_local_refinement_report(report, output_prefix):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    tsv_path = output_prefix.with_suffix(".tsv")
    json_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    pd.DataFrame(report["requests"]).to_csv(tsv_path, sep="\t", index=False)
    return json_path, tsv_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--truth-trees")
    parser.add_argument("--ne", type=float, default=10_000.0)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--pair-seed", type=int, default=42)
    parser.add_argument("--skip", type=int, default=1)
    args = parser.parse_args(argv)
    report = build_local_refinement_report(
        args.manifest, truth_trees=args.truth_trees, ne=args.ne,
        max_pairs=args.max_pairs, pair_seed=args.pair_seed, skip=args.skip,
    )
    write_local_refinement_report(report, args.output_prefix)


if __name__ == "__main__":
    main()
