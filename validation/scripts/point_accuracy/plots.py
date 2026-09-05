from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext

from .inputs import DEFAULT_XLIM, DEFAULT_YLIM
from .tables import heatmap_tables_from_df, mse_by_rounded_bins, mse_by_segments


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
    x_edges, y_edges, z_display, width = fill_grid(toplot, xlim, ylim)
    x0, y0 = xlim[0], ylim[0]
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
    i = np.rint((toplot["Simulated"].to_numpy(dtype=float) - x0) / width).astype(int)
    j = np.rint((toplot["PosteriorMean"].to_numpy(dtype=float) - y0) / width).astype(int)
    valid = (i >= 0) & (i < z.shape[1]) & (j >= 0) & (j < z.shape[0])
    np.add.at(z, (j[valid], i[valid]), toplot["x"].to_numpy(dtype=float)[valid])
    z_display = np.where(z > 0, z, np.nan)
    return x_edges, y_edges, z_display, width


def write_heatmap_pair(
    table,
    clean_path,
    labelled_path,
    *,
    xlim,
    ylim,
    xlabel,
    ylabel,
    vmax,
):
    common = dict(xlim=xlim, ylim=ylim, vmax=vmax)
    raster_heatmap(
        table, clean_path, xlabel="", ylabel="", with_colorbar=False, **common,
    )
    raster_heatmap(
        table, labelled_path, xlabel=xlabel, ylabel=ylabel,
        with_colorbar=True, **common,
    )


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


def write_mse_outputs(
    mean_table,
    out_prefix,
    *,
    error_column,
    plot_name,
    ylabel,
    table_name=None,
):
    if table_name is not None:
        mean_table.loc[:, ["SimRound", "len", "Mean", "MSE"]].to_csv(
            out_prefix.parent / f"{out_prefix.name}{table_name}",
            sep="\t",
            index=False,
            float_format="%.10g",
        )
    mse = (
        float(mean_table[error_column].sum() / mean_table["len"].sum())
        if not mean_table.empty
        else float("nan")
    )
    (out_prefix.parent / f"{out_prefix.name}MSEall.txt").write_text(
        f"{mse}\n", encoding="utf-8"
    )
    plot_mean_mse(
        mean_table, out_prefix.parent / f"{out_prefix.name}{plot_name}",
        ylabel=ylabel,
    )
    return mse


def overlay_tsinferdate_means(
    linear_table,
    log_table,
    linear_means,
    log_means,
    out_prefix,
    *,
    xlim,
    ylim,
    xlim_log,
    ylim_log,
    label,
    tag,
    vmax,
):
    overlay_mean(
        linear_table,
        linear_means,
        out_prefix.parent / f"{out_prefix.name}_{tag}_meanest_mean_lin.png",
        xlim=xlim,
        ylim=ylim,
        xlabel="Simulated coalTime\n(2Ne generations)",
        ylabel=f"{label} coalTime\n(2Ne generations)",
        vmax=vmax,
    )
    overlay_mean(
        log_table,
        log_means,
        out_prefix.parent / f"{out_prefix.name}meanest_mean_log.png",
        xlim=xlim_log,
        ylim=ylim_log,
        xlabel="Simulated Tcoal\n(log10 2Ne generations)",
        ylabel=f"{label} Tcoal\n(log10 2Ne gen.)",
        vmax=vmax,
    )


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
    write_heatmap_pair(
        toplotall,
        out_prefix.parent / f"{out_prefix.name}_{tag}_pointest_lin_clean.png",
        out_prefix.parent / f"{out_prefix.name}_{tag}_pointest_lin.png",
        xlim=xlim,
        ylim=ylim,
        xlabel="Simulated coalTime\n(2Ne generations)",
        ylabel=f"{label} coalTime\n(2Ne generations)",
        vmax=vmax,
    )
    write_heatmap_pair(
        toplotall_log,
        out_prefix.parent / f"{out_prefix.name}_{tag}_pointest_log_clean.png",
        out_prefix.parent / f"{out_prefix.name}_{tag}_pointest_log.png",
        xlim=xlim_log,
        ylim=ylim_log,
        xlabel="Simulated coalTime\n(log10 2Ne generations)",
        ylabel=f"{label} coalTime\n(log10 2Ne generations)",
        vmax=vmax,
    )
    mean_table = mse_by_rounded_bins(toplotall)
    mean_table_log = mse_by_rounded_bins(toplotall_log)
    mseall = write_mse_outputs(
        mean_table, out_prefix,
        error_column="SqErr",
        table_name="meanPerSim.txt",
        plot_name="meanMSE_lin.png",
        ylabel=f"{label} Tcoal\n(2Ne generations)",
    )
    overlay_tsinferdate_means(
        toplotall, toplotall_log, mean_table, mean_table_log, out_prefix,
        xlim=xlim, ylim=ylim, xlim_log=xlim_log, ylim_log=ylim_log,
        label=label, tag=tag, vmax=vmax,
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
    write_heatmap_pair(
        toplotall,
        out_prefix.parent / f"{out_prefix.name}meanest_lin_clean.png",
        out_prefix.parent / f"{out_prefix.name}meanest_lin.png",
        xlim=xlim,
        ylim=ylim,
        xlabel="Simulated Tcoal\n(2Ne generations)",
        ylabel=ylabel,
        vmax=vmax,
    )
    write_heatmap_pair(
        toplotall_log,
        out_prefix.parent / f"{out_prefix.name}meanest_log_clean.png",
        out_prefix.parent / f"{out_prefix.name}meanest_log.png",
        xlim=xlim_log,
        ylim=ylim_log,
        xlabel="Simulated Tcoal\n(log10 2Ne generations)",
        ylabel=ylabel_log,
        vmax=vmax,
    )
    mean_table = mse_by_segments(df)
    mseall = write_mse_outputs(
        mean_table, out_prefix,
        error_column="WtdSqErr",
        plot_name="MeanMSE_lin.png",
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
