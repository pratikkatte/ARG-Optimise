# Shared ARG evaluation

Run from the repository root with the environment used for training:

```bash
python eval/eval.py --config config/config_learned_event_500.yaml
# Equivalent package invocation:
python -m eval.eval --config config/config_learned_event_500.yaml
```

An unchanged training YAML supplies `dataset_path`, `output_path`, `loss_type`,
`eval_episodes`, `terminal_eval_repeats`, `terminal_eval_grid_size`, and `seed`.
The runner restores model settings and physical parameters from the checkpoint,
then verifies the dataset against its saved sequences. Training also accepts the
optional `evaluation` mapping; it does not apply its settings to optimization.

Repository-relative YAML paths resolve against the repository root. Config
filenames can be passed as paths or as names found in `config/`. CLI overrides
take precedence over the evaluation mapping. For example:

```bash
python eval/eval.py --config config/config_learned_event_500.yaml \
  --checkpoint runs/trash/sim_500_subtb_hudson_corrected_baseline/best_on_policy.pt \
  --output-dir validation/reports/standalone_eval_sim_500
```

Automatic selection uses `checkpoints/best_eval_subtb_loss.pt` for SubTB and
`checkpoints/best_eval_loss.pt` for TB, then the diagnostic layout's
`best_on_policy.pt`. It never substitutes a training-loss or latest checkpoint
and never searches archived runs automatically. The example
`config/eval_sim_500.yaml` explicitly points at an existing archived run.

## Evaluation settings

```yaml
evaluation:
  checkpoint: best_eval                 # Or a repository-relative/absolute path
  metrics: [density_fit, ess, posterior_summary]
  output_dir: validation/reports/my_eval # Default: <output_path>/evaluation
  num_samples: 256                      # Per repeat; defaults to eval_episodes
  repeats: 3                           # Defaults to terminal_eval_repeats
  batch_size: 32
  seed: 100007                          # Default: training seed + 100000
  device: auto                         # auto, cpu, or cuda
  grid_size: 100
  rank_bins: 20                        # At most M+1 bins for M posterior draws
  bank_per_stratum: 64
  bank_candidates: 768                  # Even, split equally between prior/policy
  # density_bank: validation/reports/another_eval/density_bank.json.gz
  baselines:
    singer:
      trees: validation/my_singer/singer_*.trees
      sample_order: [a, b, c, d]        # Replace with actual FASTA headers
      burnin: 0                        # Number of saved files to discard
      stride: 1
    tsinfer_tsdate:
      trees: validation/my_tsdate/dated.trees
      sample_order: [a, b, c, d]
    relate:
      trees: validation/my_relate/dated.trees
      sample_order: [a, b, c, d]
```

The baseline example is illustrative; paths and sample names must be replaced.
`trees` accepts a glob, a path, or a list of these. Glob results use natural
numeric ordering. Lists preserve their order. Burn-in and stride apply to that
combined order. Inputs must already be converted to `.trees`. `sample_order`
names every sample in the file's `ts.samples()` order; the runner explicitly
maps that order to the FASTA. It requires generation times and complete marginal
trees covering the same sequence length. Baseline methods are never executed.

A single baseline file is a point estimate. Multiple SINGER files are treated
as posterior samples; multiple Relate files default to conditional branch-time
samples. An optional `kind` must match this interpretation (`point`,
`conditional_times`, or `posterior`). A point estimate must have one file. GFN–SINGER distributional
comparison requires SINGER's posterior kind. SINGER's SMC target is recorded as
an approximate reference, not the full Hudson posterior.

## Calculations and artifacts

- `eval/density_fit.py`: regression slope, Pearson correlation, and centered
  slope-one RMSE on a frozen bank. Raw fits use log reward; prior-relative fits
  subtract the log prior and compare against log likelihood. One intercept is
  used across each fit's strata. Undefined slopes/correlations are `null`.
- `eval/ess.py`: float64 log-weight accounting, ESS, ESS/N, largest normalized
  weight, population log-weight SD, and the existing importance evidence estimate.
  Pooled ESS is recalculated from all fresh weights. It is not average batch ESS.
- `eval/posterior_summary.py`: exact span-weighted truth TMRCA error and rooted RF,
  grid-based clade probabilities and TMRCA Wasserstein distance, and posterior
  covariance across ARG draws at genomic lags. Clade RMSE averages squared errors
  over the union of observed clades per position, then over positions. Time
  distances are in `2Ne`, covariance in `(2Ne)^2`. Covariance uses `ddof=1` and
  is unavailable for one draw. The representative ARG minimizes mean grid RF
  to its ensemble; ties select the first draw. Truth never selects the medoid.
- `eval/tmrca_ranks.py`: histogram of the true pairwise TMRCA rank among the
  posterior draws, with exact genomic span weights and equal haplotype-pair
  weights. All `M+1` ranks, from zero through `M`, are included for `M` draws.
- `eval/rank_kl.py`: `D_KL(observed rank histogram || uniform-rank reference)`
  in nats, without pseudocounts. Zero observed mass contributes zero.
- `eval/interval_coverage.py`: empirical equal-tail 50%, 70%, and 90% interval
  coverage and mean width. Endpoints use linear quantiles and are inclusive.
  The intervals are the 25th–75th, 15th–85th, and 5th–95th percentiles.

The three truth-calibration modules run automatically with `posterior_summary`,
including per-repeat and pooled results. They reuse the aligned pairwise times
and the already sampled ARGs. Pooled ranks and quantiles are recalculated from
the concatenated draws; they are not averages of the per-repeat statistics.
Coverage preserves `eval_truth_interval_90_coverage` and adds the corresponding
`50` and `70` keys. Rank KL is `eval_truth_tmrca_rank_kl`; all histogram masses,
quantiles, and tie fractions are saved in `tmrca_calibration` in `results.json`.
The legacy terminal evaluator also imports these modules, retaining its selected
grid or exact-span weighting. `--rank-bins 20` overrides the standalone YAML
setting; legacy training evaluation uses 20 bins.

### SINGER definitions and provenance

These diagnostics implement [SINGER Figure 4b–c and its posterior-sampling
assessment](https://www.nature.com/articles/s41588-025-02317-9).
We inspected [SINGER source at commit
eb8e39b1a15be4a9a4df4fdaab61847bf73515d7](https://github.com/popgenmethods/SINGER/tree/eb8e39b1a15be4a9a4df4fdaab61847bf73515d7).
The repository contains C++ inference and a Python
[pairwise coalescence-time helper](https://github.com/popgenmethods/SINGER/blob/eb8e39b1a15be4a9a4df4fdaab61847bf73515d7/SINGER/SINGER/compute_pairwise_coalescence_times.py),
which uses branch diversity divided by two for contemporary haplotype pairs.
The rank/KL/coverage benchmark code was not found in the inspected source.
Consequently these are independent Python implementations of the paper's
definitions, not translations of an available C++ benchmark. We retain exact
marginal-tree times: ranking window-averaged times would measure a different
quantity. Array functions expect truth `[positions, pairs]`, posterior
`[draws, positions, pairs]`, and optional position span weights, in common time
units; the evaluator supplies times in `2Ne`.

The paper does not specify the exact binning, tie, or quantile interpolation
conventions in the text inspected. Our explicit conventions are:

- Integer ranks count posterior times below truth. If `k` draws equal truth
  exactly, the observation contributes equally to the `k+1` possible tie-broken
  ranks. This deterministic average uses no RNG and introduces no time jitter.
  `tie_cell_fraction` records how much weighted genomic/pair mass has ties;
  extensive ties can mask errors, so examine this alongside the histogram.
- Ranks are grouped into consecutive bins, capped at `M+1` bins. Under uniform
  ranks a bin containing `b` ranks has reference mass `b/(M+1)`, which need not
  equal `1/bin_count`. The histogram and KL use these exact reference masses.
  KL is computed on the displayed bins, after averaging any exact ties.
  Keep draw counts, binning, and tie conventions fixed for checkpoint comparisons;
  coarse bins and small samples limit sensitivity. One bin always yields zero KL.
- Interval endpoints use NumPy's `method='linear'`, preserving the previous
  90% coverage calculation. Mean widths have the same `2Ne` units as TMRCA.
  Empirical quantile coverage has finite-draw error; nominal coverage is a
  reference, not a guarantee for a small sample.

Truth-based calibration is unavailable without simulated truth. In the standalone
report, a baseline point estimate or an ensemble with only one draw has no new
calibration plot or reported rank KL; its legacy descriptive interval coverage
remains available. The low-level array functions accept one draw, but their
outputs still require this distinction when interpreting them.
Relate's multiple branch-time samples are labelled `conditional_times`; their
summaries do not establish calibration of the full ARG posterior.

Ranks are uniform in the simulation-based calibration experiment over repeated
datasets generated from the assumed prior and likelihood, with posterior draws
for each dataset. See the [SBC definition](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html).
On a single dataset these outputs are descriptive diagnostics. Linked positions
and haplotype pairs are dependent, and sampling repeats on one dataset are not
independent simulated datasets. No independent-observation confidence bands or
significance tests are attached. Neither uniform ranks nor small rank KL alone
establishes convergence of the full ARG posterior.

Fresh policy samples are generated at temperature 1, once per repeat, and reused
across metrics. The density bank is a separate evaluation-only sample. Its first
creation uses independent prior and policy draws, stratifies by reward and
structural diversity, and saves all actions. Subsequent checkpoints rescore the
same actions. Use the same bank for comparisons between runs; strata describe
constructed coverage, not posterior probability masses. Frozen-bank ESS is never
reported. Legacy banks require their original adjacent `protocol.json`; replay
also checks their stored rewards and prior/likelihood decomposition.

Reusing an automatic bank requires its original `bank_per_stratum`. To explicitly
accept another existing bank size, set `density_bank`; to construct a different
bank, choose a new output directory. Candidate-budget settings apply only during
initial creation. An insufficient unique-history pool fails with its observed
stratum size and the settings to adjust.

The standalone frozen-bank scorer currently requires `cwr_residual` checkpoints,
matching the existing replay implementation. Legacy `cwr` checkpoints can select
`--metrics ess posterior_summary`. When a sampled history has multiple backward
choices, density fit is labelled a trajectory-balance diagnostic, rather than a
marginal terminal density.

Simulated truth is discovered from the dataset's adjacent `metadata.json`.
Without it, density/ESS and supplied-reference comparisons remain available.
Unconfigured baselines are listed as unavailable. Configured missing/invalid
files, mismatched identities, and nonfinite samples are errors, not dropped data.

Results go into `step_<update>_<checkpoint-hash>_<protocol-hash>/`:

- `protocol.json`, `results.json`, `metrics.csv`, and `report.md`;
- `fixed_scores.json.gz` and separate per-repeat score files and `.trees` draws;
- density scatter plots, ESS by repeat, and posterior covariance curves.
- `tmrca_ranks.csv` (including rank KL), `interval_coverage.csv`,
  `tmrca_ranks.png`, and `interval_coverage.png`, when truth and ensembles are
  available. CSVs include each sampling repeat and pooled results; plots show
  pooled results. These are also linked in `report.md`.

The shared bank sits above checkpoint-specific results. Results include input,
code, and output hashes; complete matching results can be reused. A changed
checkpoint or configuration receives a different result directory. Shared
functions are imported by training, legacy terminal evaluation, and maintained
validation scripts; historical source archives are retained.

## Small saved-checkpoint smoke run

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python eval/eval.py \
  --config config/eval_sim_500.yaml \
  --checkpoint runs/trash/sim_500_subtb_hudson_corrected_baseline/best_on_policy.pt \
  --output-dir validation/reports/standalone_eval_smoke \
  --device cpu --num-samples 8 --repeats 2 --batch-size 4 \
  --bank-per-stratum 4 --bank-candidates 48
```

These smoke sample counts validate execution, not scientific posterior quality.
For the full evaluation, use the configured budgets and independently repeated
datasets/training runs as appropriate.

The completed [CPU smoke report](../validation/reports/standalone_eval_smoke/README.md)
includes saved outputs and comparisons against the pre-refactor implementations.
The [TMRCA calibration smoke report](../validation/reports/tmrca_calibration_smoke/README.md)
adds the new histograms, KL, and three coverage levels, with an independent
verification against saved tree sequences and the current test results.
