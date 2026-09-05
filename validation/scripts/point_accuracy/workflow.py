"""Shared execution flow for method-specific point-accuracy commands."""

from .inputs import plot_limits_from_args, prepare_output_prefix
from .reporting import write_standard_outputs


def run_analysis(args, load_dataframe, plot, *, method_label, **plot_options):
    """Load data, create method plots, and write the standard reports."""
    output_prefix = prepare_output_prefix(args.output_prefix)
    xlim, ylim, xlim_log, ylim_log = plot_limits_from_args(args)
    limits = {
        "xlim": xlim,
        "ylim": ylim,
        "xlim_log": xlim_log,
        "ylim_log": ylim_log,
    }
    dataframe = load_dataframe(args)
    print(f"segments: {len(dataframe)} rows", flush=True)
    mse = plot(
        dataframe, output_prefix, label=method_label,
        **limits, **plot_options,
    )
    write_standard_outputs(
        dataframe, output_prefix, method_label=method_label,
        legacy_mse=mse, **limits,
    )
    print(f"MSEall = {mse}", flush=True)
    return mse
