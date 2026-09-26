# Dataset point-accuracy plots

Run from the ARG-Optimise repository root:

```bash
python paper/validation/script/point_accuracy_gfn.py --help
python paper/validation/script/point_accuracy_singer.py --help
python paper/validation/script/point_accuracy_tsinferdate.py --help
```

These commands generate individual dataset/method truth-versus-estimate TMRCA
plots, including linear and logarithmic heatmaps. They accept paper datasets
and other compatible simulated datasets; moving them does not change their
arguments, computations, output names, or plotting appearance.

`point_accuracy_common.py` contains shared plotting and input helpers.
`plot_point_accuracy.py` dispatches to a method-specific command via `--method`.
The previous plotting entry points in `validation/scripts/` have been removed;
use the paths above or import from `paper.validation.script`.
`validation/scripts/run_gfn.py` invokes the GFN plotter here after inference.

The exact manuscript Figure 2 pipeline remains in
[`paper/scripts/paper_datasets/`](../../scripts/paper_datasets/README.md).
Its binning and rendering differ from these standalone validation plots.
