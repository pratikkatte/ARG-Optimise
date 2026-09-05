import math
from pathlib import Path

import numpy as np
import pandas as pd

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
            "weighted_mae": float("nan"),
            "weighted_bias": float("nan"),
            "legacy_mseall": legacy_mse,
        }
    diff = post - sim
    mse = float(np.sum((diff**2) * weights) / total_weight)
    return {
        "n_segments": int(len(df)),
        "total_length": total_weight,
        "weighted_mse": mse,
        "weighted_rmse": math.sqrt(mse),
        "weighted_mae": float(np.sum(np.abs(diff) * weights) / total_weight),
        "weighted_bias": float(np.sum(diff * weights) / total_weight),
        "legacy_mseall": legacy_mse,
    }


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
    rows = _time_summary_rows(
        df, xlim, ylim, xlim_log, ylim_log, method_label,
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


def _time_summary_rows(df, xlim, ylim, xlim_log, ylim_log, method_label):
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
    rows.extend([
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
    return rows


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
