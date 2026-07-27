"""Run GFN, tsinfer+tsdate, and SINGER validation for one experiment."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import textwrap
import uuid
from argparse import Namespace
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from .configuration import (
    METHOD_NAMES,
    MethodConfig,
    PipelineConfig,
    load_config,
    preflight_config,
)
from .paths import OUTPUT_ROOT, VALIDATION_DIR, experiment_dir
from .scripts.point_accuracy_common import ValidationResult
from .scripts.point_accuracy_gfn import evaluate_from_args as evaluate_gfn
from .scripts.point_accuracy_singer import evaluate_from_args as evaluate_singer
from .scripts.point_accuracy_tsinferdate import (
    evaluate_from_args as evaluate_tsinfer,
)


DEFAULT_CONFIG = VALIDATION_DIR / "config.yaml"
METHOD_RUNNERS: dict[
    str, Callable[..., ValidationResult]
] = {
    "gfn": evaluate_gfn,
    "tsinfer": evaluate_tsinfer,
    "singer": evaluate_singer,
}


class ResultExistsError(RuntimeError):
    """Raised when a completed result would be replaced without permission."""


def _method_args(
    config: PipelineConfig, method: MethodConfig, output_dir: Path
) -> Namespace:
    truth = config.truth
    options = config.validation
    return Namespace(
        truth_dir=truth.tracks_dir,
        truth_prefix=truth.tracks_prefix,
        truth_trees=truth.trees,
        ne=truth.ne,
        nspl=truth.haplotypes,
        skip=options.skip,
        output_prefix=output_dir / "result",
        max_pairs=options.max_pairs,
        pair_seed=options.pair_seed,
        xlim=options.xlim,
        ylim=options.ylim,
        xlim_log=options.xlim_log,
        ylim_log=options.ylim_log,
        verbose=options.verbose,
        from_bed=method.mode == "bed",
        inferred_trees=method.inferred_trees,
        input_dir=method.input_dir,
        sample_prefix=method.sample_prefix,
        bed_dir=method.bed_dir,
        bed_prefix=method.bed_prefix,
        mcspl=method.mcspl,
        burnin_samples=method.burnin_samples,
        max_posterior_samples=method.max_posterior_samples,
        method_label=method.label,
    )


def _summary_rows(results: list[ValidationResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        rows.append(
            {
                "method": result.method,
                "method_label": result.method_label,
                "n_segments": result.metrics["n_segments"],
                "total_length": result.metrics["total_length"],
                "weighted_mse": result.metrics["weighted_mse"],
                "weighted_rmse": result.metrics["weighted_rmse"],
                "weighted_mae": result.metrics["weighted_mae"],
                "weighted_bias": result.metrics["weighted_bias"],
                "legacy_mseall": result.metrics["legacy_mseall"],
            }
        )
    return rows


def _write_comparison_plot(rows: list[dict[str, object]], path: Path) -> None:
    labels = [
        "\n".join(textwrap.wrap(str(row["method_label"]), width=18))
        for row in rows
    ]
    values = [float(row["weighted_rmse"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.75))
    bars = ax.bar(labels, values, color=("C0", "C1", "C2"))
    ax.set_ylabel("Weighted RMSE (t / 2Ne)")
    ax.set_title("ARG point-accuracy comparison")
    finite_values = [value for value in values if value == value]
    if finite_values:
        upper = max(finite_values)
        ax.set_ylim(0, upper * 1.18 if upper > 0 else 1.0)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        if value == value:
            ax.annotate(
                f"{value:.4g}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _path_text(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _method_manifest(method: MethodConfig) -> dict[str, object]:
    return {
        "mode": method.mode,
        "label": method.label,
        "inferred_trees": _path_text(method.inferred_trees),
        "input_dir": _path_text(method.input_dir),
        "sample_prefix": method.sample_prefix,
        "bed_dir": _path_text(method.bed_dir),
        "bed_prefix": method.bed_prefix,
        "mcspl": method.mcspl,
        "burnin_samples": method.burnin_samples,
        "max_posterior_samples": method.max_posterior_samples,
    }


def _write_manifest(
    config: PipelineConfig,
    results: list[ValidationResult],
    staging_dir: Path,
) -> None:
    truth = config.truth
    manifest = {
        "status": "complete",
        "experiment": config.experiment,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config.source),
        "output_directory": str(
            experiment_dir(
                config.experiment, output_root=config.output_root
            )
            / "results"
        ),
        "truth": {
            "trees": _path_text(truth.trees),
            "tracks_dir": _path_text(truth.tracks_dir),
            "tracks_prefix": truth.tracks_prefix,
            "ne": truth.ne,
            "haplotypes": truth.haplotypes,
        },
        "validation": asdict(config.validation),
        "methods": {
            name: _method_manifest(config.methods[name]) for name in METHOD_NAMES
        },
        "results": {
            result.method: {
                "method_label": result.method_label,
                "metrics": result.metrics,
                "artifacts": [
                    f"{result.method}/{artifact.name}"
                    for artifact in result.artifacts
                ],
            }
            for result in results
        },
    }
    (staging_dir / "run_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )


def _publish_results(
    staging_dir: Path, results_dir: Path, *, force: bool
) -> None:
    if results_dir.exists() and not force:
        raise ResultExistsError(
            f"results already exist at {results_dir}; pass --force to replace them"
        )
    if not results_dir.exists():
        staging_dir.replace(results_dir)
        return

    backup = results_dir.parent / f".results-backup-{uuid.uuid4().hex}"
    results_dir.replace(backup)
    try:
        staging_dir.replace(results_dir)
    except Exception:
        backup.replace(results_dir)
        raise
    shutil.rmtree(backup)


def run_validation(
    config: PipelineConfig,
    *,
    force: bool = False,
    method_runners: Mapping[str, Callable[..., ValidationResult]] | None = None,
) -> Path:
    """Run and atomically publish a complete three-method comparison."""
    preflight_config(config)
    experiment_root = experiment_dir(
        config.experiment, output_root=config.output_root
    )
    results_dir = experiment_root / "results"
    if results_dir.exists() and not force:
        raise ResultExistsError(
            f"results already exist at {results_dir}; pass --force to replace them"
        )

    experiment_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".results-", dir=str(experiment_root))
    )
    runners = METHOD_RUNNERS if method_runners is None else method_runners
    try:
        results: list[ValidationResult] = []
        for method_name in METHOD_NAMES:
            print(f"Validating {method_name} ...", flush=True)
            method_dir = staging_dir / method_name
            args = _method_args(
                config, config.methods[method_name], method_dir
            )
            result = runners[method_name](args, output_dir=method_dir)
            if result.method != method_name:
                raise RuntimeError(
                    f"{method_name} evaluator returned result for {result.method}"
                )
            results.append(result)

        rows = _summary_rows(results)
        pd.DataFrame(rows).to_csv(
            staging_dir / "summary.tsv",
            sep="\t",
            index=False,
            float_format="%.10g",
        )
        _write_comparison_plot(rows, staging_dir / "weighted_rmse.png")
        _write_manifest(config, results, staging_dir)
        _publish_results(staging_dir, results_dir, force=force)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    return results_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate GFN, tsinfer+tsdate, and SINGER outputs for one experiment."
        )
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment folder name under validation/output.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Validation YAML (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace this experiment's completed results, but never its GFN samples.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(
            args.config, args.experiment, output_root=OUTPUT_ROOT
        )
        results_dir = run_validation(config, force=args.force)
    except Exception as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        return 2
    print(f"Validation results written to {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
