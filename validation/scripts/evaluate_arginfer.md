# Validate saved ARGinfer posterior draws

This evaluator reads trusted local ARGinfer `.arg` pickle files. It does not run
inference, modify the input files, install packages, or require PyTables. Use the
`phylogfn_orig` Conda environment. From the repository root:

```bash
source /private/home/pkatte/anaconda3/etc/profile.d/conda.sh
conda activate phylogfn_orig
```

Run one dataset at a time:

```bash
python validation/scripts/evaluate_arginfer.py \
  --arg-dir validation/datasets/paper_datasets/output/arginfer/r1/job_38078901 \
  --dataset-dir validation/datasets/paper_datasets/r1/rep0 \
  --burnin-samples 0 \
  --output-dir validation/reports/arginfer/r1
```

```bash
python validation/scripts/evaluate_arginfer.py \
  --arg-dir validation/datasets/paper_datasets/output/arginfer/r2/job_38078901 \
  --dataset-dir validation/datasets/paper_datasets/r2/rep0 \
  --burnin-samples 0 \
  --output-dir validation/reports/arginfer/r2
```

```bash
python validation/scripts/evaluate_arginfer.py \
  --arg-dir validation/datasets/paper_datasets/output/arginfer/r4/job_38078901 \
  --dataset-dir validation/datasets/paper_datasets/r4/rep0 \
  --burnin-samples 0 \
  --output-dir validation/reports/arginfer/r4
```

Alternatively replace `python` with
`/private/home/pkatte/anaconda3/envs/phylogfn_orig/bin/python`; activation is then
unnecessary. Output directories must not already exist. A failed run retains
its partial output and `failure.json`; choose another output directory to retry.
Only a complete report has a `SUCCESS` marker.

## Saved-draw selection and integrity

The three paper outputs have 1,800 saved draws each, iterations 201,000 through
2,000,000 at spacing 1,000. The inference launch removed its initial 200,000
iterations; `--burnin-samples 0` discards no additional draws. Files are sorted
by numeric iteration. Unexpected names, duplicate iterations, or an interior
gap in the expected spacing cause failure. The native `arg.arg` scratch snapshot
is ignored because it has no saved iteration and is not a posterior draw. Point
`--arg-dir` at the `job_...` directory containing the numbered files. First/last iterations and draw count
are reported: regular spacing alone cannot establish that a chain ran for its
entire intended duration. `--expected-thin` defaults to 1,000.

The input directory defaults to `<dataset-directory>/../arginfer_inputs` and
can be overridden with `--input-dir`. The evaluator reproduces the expected
input text from the dataset VCF and compares haplotypes, ancestral alleles,
positions, sample identities, and manifest settings/hash. It verifies truth
genotypes in the explicit `sample_nodes_in_haplotype_order` metadata order.

Every ARG undergoes native structural validation, conversion to a tskit tree
sequence, mutation-descendant checks, and a comparison of native ancestral
material length with integrated marginal-tree branch length. Recombination
children's ancestral segments are split at their stored breakpoints. Unary
nodes are suppressed. Invalid draws cause failure; they are never silently
discarded. Do not use Python's `-O` option, since ARGinfer's validator uses
assertions. SHA-256 provenance records input files, every retained ARG, native
ARGinfer source, and the metric implementation.

The recomputed native likelihood is also compared with an independent
infinite-sites calculation from marginal trees: `-mu * integrated_branch_length`
plus the sum of `log(mu * compatible_branch_length)` over observed SNPs.

The existing run logs end in a PyTables dependency error at summary-table
serialization. This evaluator reconstructs likelihood, prior, and other trace
summaries from the saved ARGs. Native log likelihood and log prior use the
input manifest's mutation rate, recombination rate, population size, and length.
These scores retain ARGinfer's normalization conventions; they are not marginal
likelihood/evidence estimates. Move acceptance rates cannot be reconstructed
from these thinned ARG files.

## Truth summaries

All reported times are **generations**. Divide by the reported `2Ne` divisor for
coalescent units; divide time MSE by its square. Truth and posterior are aligned
on the exact union of their marginal-tree breakpoints, weighted by genomic span.
All 45 haplotype pairs have equal weight for these 10-haplotype datasets.
Posterior draws have equal weight. Processing uses position chunks (default
`--chunk-size 64`) instead of a full draws × genome × pairs array.

- Pairwise TMRCA and root time: truth mean, posterior mean/median summaries,
  posterior-mean bias, MAE, MSE, RMSE, and posterior-median MAE. Per-pair results
  are in `tmrca_by_pair.csv`.
- Equal-tail 50%, 70%, 90%, and 95% intervals: empirical truth coverage and mean
  width. Quantiles use NumPy linear interpolation; endpoints are inclusive.
- Truth rank: rank among all retained draws, including ranks 0 and M. Exact
  ties split weight uniformly over their possible ranks. Default 20 bins
  (`--rank-bins`); KL compares observed bin mass to the discrete-uniform rank
  reference, in nats. Bin reference masses can differ when M+1 is not divisible
  by the bin count. Rank distributions for root time and pairwise TMRCA are
  reported separately, using the repository's existing rank implementation.
- Rooted topology: represented by nontrivial descendant clades, ignoring node
  IDs, unary nodes, and times. Expected raw Robinson–Foulds distance to truth,
  true-topology posterior probability, observed topology count, and entropy
  are weighted over the genome. RF is the size of the symmetric clade-set
  difference. `clade_brier` averages squared support errors over the union of
  observed and true clades at each position; `true_clade_support` averages
  support of the true nontrivial clades.
- Topology credible sets: most frequent local topologies accumulated to
  50/70/90/95% empirical posterior mass, including **all frequency ties** at
  the boundary. Report truth membership, set size, and actual mass. Actual mass
  can exceed the requested level. An unobserved true topology is never covered.
- Structure: recombination event counts (ancestral/nonancestral as classified
  by native ARGinfer), unique event breakpoints, marginal-tree counts, ARG node
  counts, and genome-mean total branch length. If `.full.trees` is available,
  the truth event count is half the number of msprime recombination-parent
  nodes. Event counts are distinct from topology-changing breakpoint counts.
  Raw inferred ARG node counts have no truth comparison because representations
  can differ.

The inference input used integer VCF positions, whereas simulated truth retains
its original coordinates. This evaluator preserves those conventions and does
not silently move mutation positions or truth breakpoints. Rank and coverage
are descriptive for each dataset, not a simulation-based calibration experiment:
linked positions and haplotype pairs are dependent observations.

## Single-chain mixing diagnostics

Each dataset is a different target with one chain. They are never pooled as
replicate chains. R-hat is unavailable and left null. No automatic result claims
full posterior convergence.

ArviZ supplies bulk, tail, and mean ESS and Monte Carlo standard error of the
mean. ESS is in effective **saved draws**, not raw MCMC iterations. ESS fraction
is bulk ESS divided by retained draws and is not artificially capped at 1.
Report lag-1 autocorrelation, MCSE/posterior SD, early/late half means, and their
difference divided by the full-trace SD. Constant traces get undefined ESS and
`constant_uninformative` status. Such a trace may be genuinely fixed or stuck.

Monitored quantities include reconstructed log likelihood/prior/posterior,
global tree summaries, all pairwise times and root times at five evenly spaced
genomic midpoint positions, each observed clade's presence at those positions,
and true-topology membership. `--diagnostic-positions` controls this grid; local
ESS is not an exhaustive diagnostic of every base or the entire ARG posterior.
The report also provides diagnostics after discarding the first half of the
retained chain, and global ESS for 25%, 50%, and 100% prefixes.

The default status checks flag bulk or tail ESS below 400 (`--min-ess`) or
MCSE/SD above 0.05. `thresholds_met_single_chain` means only those numerical
checks were met. See [ArviZ ESS documentation](https://python.arviz.org/en/v0.23.4/api/generated/arviz.ess.html).

## Artifacts and tests

`report.md` provides the overview; `summary.json` contains aggregate results.
`diagnostics.csv`, `late_half_diagnostics.csv`, and `ess_evolution.csv` contain
the mixing results. `traces.csv.gz` preserves saved-draw order and original
iteration IDs. `genomic_profile.csv.gz` contains local TMRCA/topology summaries;
`recombination_breakpoints.csv.gz` preserves event positions and multiplicity.
PNG files show traces, autocorrelation, ESS, calibration, and genomic profiles.
`settings.json` and `provenance.json` record reproducibility information.

Synthetic tests exercise recombining-ARG conversion round trips, invalid inputs,
mutation identity, numeric file ordering, missing draws, tied topology credible
sets, chunked versus dense calibration, autocorrelated/constant ESS traces,
and complete CLI report generation without touching the paper-dataset chains:

```bash
python -m pytest -q validation/tests/test_evaluate_arginfer.py
```
