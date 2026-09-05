import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from .inputs import iter_pairs

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



