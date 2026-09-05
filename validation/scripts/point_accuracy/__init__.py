"""Reusable point-accuracy loading, metrics, plotting, and reporting."""

from .inputs import add_common_args, load_tree_sequence, plot_limits_from_args
from .plots import run_singer_plots, run_tsinferdate_plots
from .segments import dataframe_from_tree_sequences, load_posterior_tree_samples
from .tables import load_singer_bed_segments, load_tsinferdate_bed_segments
from .workflow import run_analysis

__all__ = [
    "add_common_args",
    "dataframe_from_tree_sequences",
    "load_posterior_tree_samples",
    "load_singer_bed_segments",
    "load_tree_sequence",
    "load_tsinferdate_bed_segments",
    "plot_limits_from_args",
    "run_analysis",
    "run_singer_plots",
    "run_tsinferdate_plots",
]
