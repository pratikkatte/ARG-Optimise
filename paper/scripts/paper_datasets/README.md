# Reproduce the paper table and figures

## Quick start: the three scripts we ran

Run these commands from the repository root, in order. All three scripts use
the adjacent [config.yaml](config.yaml) and the existing `phylogfn_orig` environment.
They evaluate saved samples; they do not rerun training or sampling.

```bash
# 1. Section 5.1 / Table 2: accuracy and uncertainty metrics
conda run --no-capture-output -n phylogfn_orig python \
  paper/scripts/paper_datasets/evaluate.py

# 2. Figure 2: posterior-mean pairwise TMRCA versus simulated truth
conda run --no-capture-output -n phylogfn_orig python \
  paper/scripts/paper_datasets/figure_2.py

# 3. Figure 3: posterior-summary agreement with ARGInfer and SINGER
conda run --no-capture-output -n phylogfn_orig python \
  paper/scripts/paper_datasets/figure_3.py
```

| Script | Main outputs, relative to the repository root |
| --- | --- |
| [evaluate.py](evaluate.py) | `validation/paper_datasets/report/section_5_1_final_checkpoints/section_5_1.csv` and `.md` |
| [figure_2.py](figure_2.py) | `validation/paper_datasets/report/figure_2_final_checkpoints/figure_2.png`, `.pdf`, and `.svg` |
| [figure_3.py](figure_3.py) | `validation/paper_datasets/report/figure_3_final_checkpoints/figure_3.png`, `.pdf`, `.svg`, and `metrics.csv` |

The current configuration uses `fixed_universe` normalization (1,012 clades)
for the table's Brier score and Figure 3's clade RMSE. Both main figures select
r1 checkpoint `1900_491fgs0e`, r2 `6600`, and r4 `1750`. Figure 2 also saves an
alternative using r1 `5650`. Figure 3 annotates pair-and-position-averaged
Wasserstein distance (`local_mean`), with pooled distance saved separately.

## Reproduce the Section 5.1 table

From the repository root, using the existing environment with ARGInfer installed:

```bash
conda run --no-capture-output -n phylogfn_orig python \
  paper/scripts/paper_datasets/evaluate.py
```

Or use `/private/home/pkatte/anaconda3/envs/phylogfn_orig/bin/python` directly.
The script also works from this directory with `python evaluate.py`.
It evaluates saved files on CPU; no training or GPU inference is needed.

The default configuration is the adjacent `config.yaml`. For another config:

```bash
python paper/scripts/paper_datasets/evaluate.py --config /path/to/config.yaml
```

All data/output paths in YAML are relative to the repository root, or may be
absolute. The working directory does not change their interpretation.

## Configuration

- `datasets`: dataset metadata/ground-truth location and sources to evaluate.
  Set `enabled: false` on a dataset or individual source to skip it.
- `sources`: each has `method`, `format` (`argflow`, `singer`, or `arginfer`),
  and `directory`. Add another ARGFlows checkpoint by adding a source entry.
- `metrics`: remove names to run only the desired subset of the five metrics.
- `credible_level`: default 0.90, giving equal-tail 5th/95th-percentile bounds.
- `brier_normalization`: `observed_union` matches the existing paper table;
  `fixed_universe` uses all possible nontrivial rooted clades. Both are saved
  in JSON whenever `clade_brier` is requested. The supplied configuration now
  defaults to `fixed_universe`, using the same denominator for every method.
- `expected_samples`: required retained draw count; sources can override it.
  Changing it does not truncate inputs: unexpected counts fail validation.
- `burnin_samples`: additional saved draws to discard, available for baselines.
  Update a source's expected count if changing burn-in.
- `expected_thin`: ARGInfer saved-iteration spacing (default 1,000).
- `unknown_time_units`: explicit interpretation of SINGER's `unknown` metadata.
  The supplied files use the existing converter's generation convention;
  `generations` changes only the in-memory label, not numeric times or files.
- `chunk_size`: genomic intervals processed at once; changes memory/runtime,
  not metric definitions. The default 64 bounds the draw-by-position arrays.
- `verify_sample_hashes`: compare ARGFlows samples with the generation manifest.
  All evaluated files are hashed and their identities saved regardless.
- `output_dir`: generated CSV, Markdown, JSON and per-pair results destination.
  Reruns recompute the configured metrics and replace those reports.

## Included draws

The supplied configuration evaluates all four final ARGFlows checkpoints,
1,000 draws each; 1,000 SINGER draws per dataset (files 100–1099); and 1,800
ARGInfer draws per dataset (saved iterations 201,000–2,000,000). ARGInfer's
unnumbered `arg.arg` scratch file is excluded. No additional burn-in is applied.
Different checkpoints and datasets are evaluated separately, never pooled.
The table reports the number of draws; equal counts do not imply equal effective
sample sizes for independent policy draws and correlated MCMC draws.

## Exact metric definitions

Every posterior draw has equal weight. No importance reweighting or resampling
is performed. Let `L` be sequence length, `P = n(n-1)/2` the number of unordered
haplotype pairs, `M` the number of draws, and `t*(x,p)` the simulated true pairwise
TMRCA. All time values are divided by `2 Ne`, with Ne read from dataset metadata.

1. **Pairwise TMRCA RMSE:** square root of the genome/pair mean of
   `(mean_draw TMRCA(x,p) - t*(x,p))^2`. The square root is taken after averaging
   squared errors across all pairs and genomic spans; this is neither the mean
   of per-pair RMSEs nor the RMSE of individual posterior draws.
2. **Rooted RF:** genome/draw mean of the symmetric difference between inferred
   and true sets of nontrivial descendant clades. This is raw rooted RF,
   not normalized RF. Singleton clades, the full/root clade and duplicate clades
   from unary nodes are excluded. For 10 fully binary samples, the maximum is 16.
3. **Clade Brier:** for clade probability `q_c(x)` and truth indicator `y_c(x)`,
   integrate the average of `(q_c(x)-y_c(x))^2` over genomic span.
   `observed_union` divides at each position by the number of distinct clades
   in truth or any retained draw from that row. This reproduces the old table,
   but the denominator depends on method/draw count. `fixed_universe` divides
   by `2^n-n-2` (1,012 for n=10), allowing the same denominator across methods.
   JSON also contains the undivided genome-mean sum. These definitions have
   different scales; do not mix them in one column.
4. **Credible-interval coverage:** genome/pair mean of the indicator that truth
   falls inside the equal-tail interval. Endpoints are inclusive. Quantiles use
   NumPy's `linear` interpolation across retained draws. CSV uses a fraction;
   Markdown displays a percentage.
5. **Credible-interval width:** genome/pair mean of upper minus lower quantile,
   in 2Ne generations. Interpret width together with coverage.

All integrals use exact tree-sequence breakpoints and genomic-span weights,
not an evenly spaced position grid. Topology metrics use a sweep of clade-count
changes; time metrics align all truth/sample breakpoints in bounded chunks.
Truth sample order is explicit in dataset metadata. Tree simplification removes
representation-specific unary ancestry above the local sample MRCA.

## Input validation and outputs

The evaluator checks sample counts, sequence lengths, time units, haplotype order,
complete local ancestry, and observed SNP patterns. ARGFlows files and checkpoints
are checked against their generation hashes. Trusted local ARGInfer pickle files
undergo native structural validation, ancestral-material conversion and mutation
descendant checks. Run without Python's `-O` flag. SINGER's site/genotype data are
verified before its explicit generation-unit interpretation is applied.

ARGFlows uses exact continuous simulation SNP positions. The saved baselines
used rounded integer VCF positions; their input identity/mutation checks preserve
those coordinates. All methods are evaluated against the same unmodified
simulated truth. This small input-coordinate difference should remain disclosed.

Default outputs are under
`validation/paper_datasets/report/section_5_1_final_checkpoints/`:

- `section_5_1.csv` and `section_5_1.md`: complete comparison table.
- `<dataset>/<method>/<source>/results.json`: metrics, both Brier definitions,
  exact input hashes, sample selection, units, and runtime.
- `tmrca_by_pair.csv` in each source folder: selected metrics for each pair.
- `config.used.yaml` and `provenance.json`: configuration and software/source
  identities needed to reproduce the calculation.

Only a successful run emits the final table. Coverage is descriptive for these
three simulated datasets; linked genomic positions/pairs are dependent. Baseline
agreement and high coverage alone do not establish posterior convergence or
general calibration. No checkpoint is selected using truth scores.

Run the focused analytic/reference tests with:

```bash
python -m pytest -q paper/scripts/paper_datasets/test_evaluate.py
```

## Reproduce Figure 2

```bash
conda run --no-capture-output -n phylogfn_orig python \
  paper/scripts/paper_datasets/figure_2.py
```

This script uses the same datasets and saved sources as the table. It recomputes
each source's posterior mean from all retained draws and checks the figure's
RMSE against the corresponding table `results.json`, including matching truth
and sample hashes. If the table report is absent, it still computes the figure
but records that no table crosscheck was available. It does not reuse the old
pilot-checkpoint figures. The `--config` argument accepts another YAML file.

The `figure2` section of `config.yaml` controls output location, method order,
ARGFlows checkpoint selection, logarithmic axis limits, number of histogram bins,
color limits, resolution, and whether alternative checkpoint figures are saved.
The main figure uses r1 `checkpoint_1900_491fgs0e`, r2 `checkpoint_6600`, and r4
`checkpoint_1750`, as selected by the user. A second full figure changes only the
r1 ARGFlows panel to `checkpoint_5650`. Dataset row order follows the YAML.

Figure outputs are under `validation/paper_datasets/report/figure_2_final_checkpoints/`:

- `figure_2.png`, `figure_2.pdf`, `figure_2.svg`: main Figure 2.
- `figure_2_r1_checkpoint_5650.*`: alternative r1 checkpoint.
- `caption.txt`: manuscript caption.
- Per-source NPZ files: exact truth/posterior-mean TMRCA pairs, normalized
  span/pair weights, histogram mass, and below-range mass.
- Per-source JSON plus `provenance.json` and `config.used.yaml`: checkpoint
  identities, statistics, file hashes, table crosschecks, and plotting settings.

Both axes use 2Ne generations, with equal logarithmic bins from 0.01 to 10 by
default. Color is the percentage of genomic-span/pair mass in a bin, normalized
to 100% separately for each panel, and displayed on a common logarithmic color
scale. No smoothing is applied. The dashed diagonal shows exact agreement.
Pearson correlation is weighted by genomic span and equal pair weights, on the
original time values, not log times. RMSE uses the same weights as the table.
Inferred values below the display range are shown as downward boundary markers;
they remain in both statistics, and their percentage is printed in the panel.
Values above the configured range cause an error instead of being silently
dropped. All plots retain the manuscript's existing three-column layout.

Figure-specific numerical tests:

```bash
python -m pytest -q paper/scripts/paper_datasets/test_figure_2.py
```

## Reproduce Figure 3

```bash
conda run --no-capture-output -n phylogfn_orig python \
  paper/scripts/paper_datasets/figure_3.py
```

The script accepts `--config /path/to/config.yaml` and uses the `figure3` section.
It reads the saved draws, checks their identity/structure, computes exact
posterior-summary comparisons, and exports PNG/PDF/SVG versions of Figure 3.
The main ARGFlows checkpoints match Figure 2: r1 `1900_491fgs0e`, r2 `6600`,
and r4 `1750`. Each ARGFlows and SINGER ensemble has 1,000 draws; each ARGInfer
ensemble has 1,800. No additional burn-in, inference, or importance weighting
is introduced.

The new figure corrects a difference between the old Figure 3 code and the
manuscript's Wasserstein definition. Both distances are saved:

- **Mean local Wasserstein** (the selected annotation) computes the empirical
  1D Wasserstein distance between methods separately for every unordered
  haplotype pair and every genomic interval, then integrates exact spans and
  averages over pairs. In notation, `sum_p integral W1(P_A(t|x,p), P_B(t|x,p)) dx / (L*P)`.
  It uses the union of sample breakpoints, not a genomic grid. Unequal sample
  counts are handled by exact empirical quantile integration, without
  interpolation or resampling. Times and distances are in 2Ne generations.
- **Pooled Wasserstein** first mixes over genomic positions and haplotype pairs,
  then compares those one-dimensional mixtures. This matches the old code but
  can hide local disagreement: swapping two pairs' distributions can preserve
  the pooled distribution. It is bounded above by the mean local distance.

Set `wasserstein_annotation: local_mean` (default, matching the manuscript and
the user's selection) or `pooled`. Both metrics are always calculated. Panel (a)
shows pooled marginal distributions for visualization, with common reflected
Gaussian smoothing (bandwidth 0.10; histogram width 0.025 in 2Ne units). Metrics
use unsmoothed empirical samples. The axis is linear to 2 and logarithmic above;
density remains per unit linear time, so area measured visually on this axis
is not probability mass. Full empirical tails are retained.

Panel (b) compares posterior clade probabilities across methods. Its RMSE is
`sqrt(sum_c integral (p_ARGFlows(c,x)-p_reference(c,x))^2 dx / (L*C))`.
The default `clade_normalization: fixed_universe` uses `C = 2^n-n-2` (1,012
for 10 haplotypes) in both reference comparisons and every dataset. Clades
absent from both compared methods contribute zero error; clades absent from
all three methods are explicitly included at (0,0) in the density plots.
Thus the plotted weights and RMSE use the same denominator. `shared_union`
reproduces the old figure's denominator: the genomic mean size of the clade
union across all three methods. Both RMSE versions are saved in the metrics.
This is a method-to-method probability RMSE, **not** the truth-based Brier score
from Table 2. No truth TMRCA or true-clade membership enters these metrics;
the shared loaders may read truth to verify input haplotype identity.

`chunk_size` controls temporary working memory for local Wasserstein; it does
not change the metric. Plot bin widths, smoothing bandwidth, clade bins, color
limits, and DPI are configurable separately. These affect rendering only.

Outputs are under `validation/paper_datasets/report/figure_3_final_checkpoints/`:

- `figure_3.png`, `.pdf`, `.svg`, and `caption.txt`.
- `metrics.csv` / `metrics.json`: both Wasserstein definitions, both clade RMSE
  definitions, selected checkpoints and retained draw counts.
- `<dataset>_wasserstein_by_pair.csv`: exact span-averaged distances for all pairs.
- `<dataset>_<method>.npz` / `.json`: empirical TMRCA masses, clade change tracks,
  source hashes, sample selection, and population size.
- Clade NPZs: unbinned probabilities/weights and displayed density-bin masses.
- `config.used.yaml` / `provenance.json`: run configuration, software versions,
  metric source hashes, and every input draw's hash.

These compare marginal summaries with reference samplers; they do not establish
equality with the full ARG posterior or MCMC convergence. The baseline input
coordinate/time-unit conventions described above also apply here.

Run numerical/reference checks with:

```bash
python -m pytest -q paper/scripts/paper_datasets/test_figure_3.py
```

### Evaluate newly generated paper outputs

All three commands accept the method output roots directly:

```bash
python paper/scripts/paper_datasets/evaluate.py \
  --argflow-dir paper/outputs/argflow \
  --singer-dir paper/outputs/SINGER \
  --arginfer-dir paper/outputs/ARGInfer
```

Use the same arguments with `figure_2.py` and `figure_3.py`. Supply all three
roots together; relative paths are resolved against the repository root.
Each root contains `r1`, `r2`, and `r4`. ARGFlow must have exactly one
checkpoint subdirectory containing `manifest.json` per dataset. SINGER uses
`<dataset>/trees/trees_<index>.trees`; ARGInfer uses `<dataset>/arg<iteration>.arg`
and its prepared inputs at `inputs/<dataset>` (also used to validate SINGER).
Existing manifest, hash, genotype, and inventory checks remain enabled.
The paper protocol expects 1,800 ARGFlow, 1,000 SINGER, and 1,800 ARGInfer draws.
Results go to `paper/outputs/evaluation`, `paper/outputs/figure_2`, and
`paper/outputs/figure_3`. Metric and plotting defaults still come from the
bundled config; `--config` remains available for custom settings and the
original config-only interface is unchanged.
