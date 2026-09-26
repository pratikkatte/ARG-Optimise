# Validation commands

Run commands from the repository root. Standalone commands are retained even
when no other script imports them. Each Python CLI documents its arguments
with `--help`; some workflows require external inference programs or saved data.

## Simulation and scoring

- [Dataset simulator](../../paper/scripts/simulate_infinite_sites.py):
  `python paper/scripts/simulate_infinite_sites.py --config paper/datasets/r1_dataset.yaml`.
  Its guide and tests also live in `paper/scripts/`; the old compatibility
  entry point has been removed. Pass `--config` explicitly because the default
  `paper/config/config.yaml` is absent. See the
  [simulator guide](../../paper/scripts/simulate_infinite_sites.md).
- [Candidate scoring](score_infinite_sites.md), `sample_infinite_sites.py`,
  `validate_infinite_sites_neural.py`, and `validate_rep0_neural_run.py`.
- The small posterior correctness experiment lives in [paper/poc](../../paper/poc/README.md).

## Baseline inference and point accuracy

- [GFN runner](run_gfn.md), `run_singer.sh`, and `run_tsinfer_tsdate.py`.
- SINGER convergence: the shell launchers use `singer_convergence.py`.
- The [point-accuracy plotters](../../paper/validation/script/README.md) live in
  `paper/validation/script/`: `point_accuracy_gfn.py`, `point_accuracy_singer.py`,
  and `point_accuracy_tsinferdate.py` share `point_accuracy_common.py`.
  `plot_point_accuracy.py` also lives there. The former plotting paths here
  have been removed; use the new paths. `point_accuracy_subtb.py` remains a
  separate metrics CLI.
- `prepare_arginfer_inputs.py` prepares ARGinfer inputs;
  `evaluate_arginfer.py` evaluates saved results.

## Manuscript evaluation

The [paper table and figure pipeline](../../paper/scripts/paper_datasets/README.md)
and manuscript comparison scripts now live in `paper/scripts/`.
`evaluate_gfn_manuscript.py` here is a compatibility entry point for saved report scripts.
Shared `evaluate_arginfer.py` and `prepare_arginfer_inputs.py` remain here.
`sample_final_checkpoints.py` also stays here as a supporting sampling utility.

## Training diagnostics and utilities

`prepare_long_run.py`, `evaluate_convergence.py`, and
`compress_evaluation_reports.py` remain available.
The experiment-specific temperature/density-fit suite, convergence-pilot scripts,
calibration diagnostic, and checkpoint-fork scripts have been removed.
Existing reports and historical provenance have not been rewritten.

## Historical scripts

Four older scripts moved to [archive/](archive/README.md): the hard-coded
`ts_infer.py` / `vcf_to_samples.py` pair and the two older TMRCA plotting scripts.
Their former command paths are retired without wrappers. Historical reports
and saved provenance have not been rewritten.
