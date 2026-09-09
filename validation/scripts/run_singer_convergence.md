Run from the repository root:

```bash
bash validation/scripts/run_singer_convergence.sh --dry-run
bash validation/scripts/run_singer_convergence.sh
```

The script uses `phylogfn_orig` at `$HOME/anaconda3/envs/phylogfn_orig`, or the
interpreter specified by `PYTHON_BIN`. It reuses the completed run at
`validation/datasets/human_25kb_super_easy/output/singer_thin500_seed42/rep0`.
It runs seeds **123, 456, and 789** sequentially on the **same rep0 VCF**, with
50,000 burn-in iterations, thinning 500, and 200 retained ARGs per chain.
The SINGER wrapper and converter paths come from the completed reference run.

New chains go into
`validation/datasets/human_25kb_super_easy/output/singer_convergence_rep0/chains/seedSEED/`.
Rerunning this script reuses any completed chains whose metadata, VCF hash,
seed, sampling settings, and posterior indices match. Existing incomplete
directories cause an error instead of being overwritten. Use `--output-root`
to choose a fresh location after a failed run. Seed 42 is always reused, never
rerun or copied. Its VCF and saved settings must match the requested analysis.

The `report/` directory under the output root contains:

- `report.md`: interpretation, recovery counts, thresholds, and limitations.
- `diagnostics.csv`: rank-normalized split R-hat, combined bulk/tail ESS, mean
  Monte Carlo standard error (MCSE), MCSE/posterior SD, and flags per quantity.
- `overview.png` and `all_traces.pdf`: overlaid independent-chain traces.
- `seedSEED_trace.csv`: extracted retained draws, preserving original indices.
- `manifest.json`: interpreter, versions, input/output hashes, tool paths,
  sampling settings, and recovery counts.
- `summary.json`: machine-readable diagnostic outcome.

The screening defaults are R-hat <1.01, bulk and tail ESS >=400 across four
chains, and mean MCSE <=5% of posterior SD. The MCSE threshold is a heuristic;
scientific accuracy requirements may require a different tolerance. Options
`--max-rhat`, `--min-ess`, and `--max-mcse-sd` change the screening thresholds.
Globally constant traces are labeled uninformative. Undefined diagnostics or
chains stuck at different values are flagged. The program exits successfully
when analysis finishes, even if diagnostics have concerns: inspect
`summary.json`'s `verdict` (`diagnostic_concerns` or `monitored_thresholds_met`).
Inference or validation errors stop the program with a nonzero exit code.

These are diagnostics, not a proof of convergence or posterior correctness.
All local TMRCA/root-age traces and global summaries should be inspected for
drift. The script records recovery events; statistical diagnostics cannot
verify that numerical recovery preserves the target distribution. It does
not combine different simulation replicates or silently discard extra burn-in.
It does not run a subsequent longer-chain stability comparison automatically.

Dependencies: numpy, scipy, pandas, matplotlib, tskit, and ArviZ (tested with
0.22.0). The current session has supplementary ArviZ packages under
`/tmp/singer_convergence_arviz`; these are discovered if ArviZ is missing from
the environment. This directory is temporary. For another installation, put
the dependencies in `phylogfn_orig`, or set `SINGER_DIAGNOSTICS_PATH` to a
compatible supplemental package directory. The script never installs packages.

`--dataset`, `--reference-run`, and `--output-root` can select another completed
reference and dataset. Sampling overrides must match the seed-42 reference;
for example, requesting 1,000 samples requires a completed 1,000-sample
reference unless `--run-all-seeds` is selected. In that mode the completed
reference supplies input/tool validation and may have fewer retained samples;
all four full-length chains live in the new output root.
`--dry-run` checks metadata, hashes, and posterior filenames without
running inference, computing diagnostics, or creating files. A normal run also
loads existing trees before starting new chains.

For the longer follow-up on the current 25 kb rep0 analysis:

```bash
bash validation/scripts/run_singer_convergence_long.sh --dry-run
bash validation/scripts/run_singer_convergence_long.sh
```

This launcher requests **1,000 retained ARGs per seed** (42, 123, 456, 789),
with burn-in 50,000 and thinning 500. It runs each chain from the beginning in
`output/singer_convergence_rep0_long/chains/seedSEED/`. It does not modify or
resume the original shorter chains. Each full run requests 1,100 saved states,
retaining indices 100–1099. Reruns reuse already completed full-length chains.

The new `output/singer_convergence_rep0_long/report/` also compares against
`output/singer_convergence_rep0/report/`, after validating the baseline VCF,
seeds, burn-in and thinning. Additional artifacts are:

- `short_vs_long_diagnostics.csv`: old/new R-hat, ESS, and MCSE.
- `short_vs_long_means.csv`: per-chain short-run and long-run means, plus
  first/last-half means for the longer chain.
- `stability.png`: moving means for total branch length and pairwise TMRCA,
  with the previous per-chain means indicated.
- `late_half_diagnostics.csv`: a secondary check using the last half of each
  longer retained chain. The full retained-chain result remains the main result.

Longer reruns share initial seeds with the shorter runs, so their estimates
are correlated; the comparison does not treat these as independent samples.
Longer sampling is an experiment, not a guarantee of convergence or a fix for
SINGER's numerical failures. If any chain fails, the script stops and preserves
the partial output. It does not automatically discard failed chains.
