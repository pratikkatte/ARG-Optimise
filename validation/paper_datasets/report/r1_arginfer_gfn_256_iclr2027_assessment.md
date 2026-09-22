# ICLR 2027 manuscript assessment

ARGinfer versus GFlowNet: corrected 256-draw comparison on r1

Prepared 22 September 2026. Scope: statistical accuracy and posterior approximation; runtime and compute cost are excluded.

## Assessment

The corrected comparison is suitable as a pilot experiment or case study, but is not yet sufficient as the main evidence for accurate posterior sampling. Using 256 draws per method is reasonable. The larger concerns are replication, target alignment, and whether the evidence supports the manuscript’s claims.

ICLR 2027 guidance emphasizes scientifically rigorous evidence supporting the claims and the significance of the contribution. It does not require every contribution to achieve state-of-the-art results. This assessment concerns experimental support, not a prediction of acceptance.

[ICLR 2027 Reviewer Guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines)

## Evaluation protocol

Dataset: r1/rep0, 10 haplotypes, 25,000 bp, 45 haplotype pairs. ARGinfer: first 256 retained draws, iterations 201,000–456,000, spacing 1,000; no additional burn-in or thinning. GFN: the same 256 fresh unweighted draws from the best checkpoint at step 5,900 of the 6,000-update run; sampling seed 20260922. The checkpoint was selected by evaluation log-weight spread.

Both methods are compared with the same simulated truth, using exact genomic-span weighting and equal haplotype-pair weighting. All times and interval widths are in generations. TMRCA RMSE is the error of the ensemble mean against truth. Rooted RF is averaged over draws and genomic spans; normalized RF divides by 16.

Representation correction: both ensembles are simplified to genealogies of the sampled haplotypes, removing unary ancestry above the local sample MRCA. This corrects earlier GFN root-time, total-branch-length, and marginal-tree-count diagnostics. Pairwise TMRCA, clade signatures, and importance ESS are unaffected. Thirteen synthetic validation tests passed.

## Corrected results

| Question | Metric | ARGinfer | GFN, unweighted |
| --- | --- | --- | --- |
| Are estimated genealogies accurate? | Pairwise TMRCA RMSE | 8,369.51 | 12,573.11 |
|  | Root TMRCA RMSE | 9,480.71 | 17,503.63 |
|  | Mean rooted RF / normalized RF | 9.9999 / 0.6250 | 10.7598 / 0.6725 |
| Are uncertainty estimates useful? | Pairwise rank KL (nats) | 0.1014 | 0.3004 |
|  | Root-time rank KL (nats) | 0.5391 | 0.7758 |
| Is topology uncertainty represented? | Mean support of true clades | 37.50% | 32.75% |
|  | Fixed-universe clade Brier | 0.00543918 | 0.00544068 |
|  | 90% / 95% topology-set truth coverage | 4.12% / 4.12% | 19.48% / 19.48% |
| Are estimates numerically reliable? | Pairwise TMRCA MCMC ESS (bulk / tail) | 162.36 / 203.79 | Not a target-fit diagnostic |
|  | Root-time MCMC ESS (bulk / tail) | 180.25 / 216.15 | Not a target-fit diagnostic |
|  | R-hat | Unavailable: one chain | Not applicable |
|  | Importance ESS | Not applicable | 1.90 / 256 |

Lower RMSE, RF, and Brier indicate better performance. Rank KL describes deviation from a discrete-uniform rank reference; on one dataset it is not a formal calibration test. Clade Brier uses the same universe of 1,012 nontrivial rooted clades for both methods, avoiding method-dependent denominators.

## Interval coverage and width

Each cell reports empirical truth coverage followed by mean interval width in generations. Intervals are equal-tailed with inclusive endpoints and linear quantile interpolation.

| Nominal | Pairwise ARGinfer | Pairwise GFN | Root ARGinfer | Root GFN |
| --- | --- | --- | --- | --- |
| 50% | 50.71% / 8,284 | 62.58% / 18,741 | 65.44% / 13,275 | 50.33% / 27,418 |
| 70% | 75.75% / 13,179 | 81.22% / 30,396 | 73.34% / 20,959 | 81.60% / 42,100 |
| 90% | 89.29% / 22,894 | 89.97% / 52,805 | 94.82% / 34,745 | 92.98% / 70,919 |
| 95% | 94.52% / 29,193 | 96.35% / 66,803 | 98.99% / 44,184 | 92.98% / 87,955 |

## What the evidence supports

• ARGinfer has lower pairwise and root-time RMSE on these draws. GFN errors are approximately 1.50 times and 1.85 times larger, respectively.

• The methods have similar 90% TMRCA coverage, but GFN intervals are approximately 2.31 times wider for pairwise TMRCA and 2.04 times wider for root time. Similar coverage therefore does not establish equally informative uncertainty.

• Clade Brier scores are nearly equal. GFN has higher empirical exact-topology coverage in this example, but cutoff frequency ties make both methods’ nominal 90% and 95% sets include all observed topologies. Those sets do not account for unobserved topology mass and should not be presented as calibrated topology uncertainty.

• GFN importance ESS is 1.8977 out of 256 (0.741%). One draw carries 70.78% of normalized importance weight. This is a substantial concern for a claim that the learned distribution closely matches the full posterior. It does not imply that every marginal summary must be inaccurate.

## Work needed for a stronger manuscript comparison

1. Replicate across independently simulated datasets within r1, r2, and r4, and across GFN training/sampling seeds. Report paired differences and uncertainty across datasets. Linked genomic positions and haplotype pairs are not independent experimental replicates.

2. Validate the ARGinfer reference using independent chains before selecting the reported 256 draws. A single-chain ESS table cannot establish between-chain convergence.

3. Align SNP coordinates and document compatible target-model assumptions. Currently all 16 r1 SNP coordinates differ: GFN uses exact mutation positions while ARGinfer uses rounded positions, with maximum absolute difference 0.4173 bp. The effect has not been quantified.

4. Keep 256 draws as the main equal-count benchmark and add a sample-count sensitivity analysis, especially for topology coverage and rank histograms. Predefine checkpoint and draw-selection rules.

5. Match the claims to the evidence. Strong posterior-calibration claims need repeated simulations under a suitable calibration design. Report importance ESS separately from MCMC autocorrelation ESS.

[Simulation-based calibration: Talts et al.](https://arxiv.org/abs/1804.06788)

## Suggested manuscript wording

“On this simulated dataset, the GFlowNet produces diverse, mutation-compatible ARGs, but exhibits larger TMRCA errors and broader uncertainty intervals than ARGinfer. Importance-weight concentration indicates remaining posterior approximation error.”

The current experiment does not support the stronger statement that GFN matches ARGinfer’s posterior quality. It can be included transparently as preliminary evidence or a case study, alongside its limitations.

## Reproducibility and artifacts

Environment: phylogfn_orig.

Main comparison script: validation/scripts/compare_r1_arg_summaries_256.py

ARGinfer validation and shared metrics: validation/scripts/evaluate_arginfer.py

GFN diagnostics: validation/scripts/gfn_autocorrelation_ess.py

Frozen GFN sampling code: runs/r1_subtb_tb003_continuation_2026-09-21/source/infer.py

Corrected results: validation/reports/gfn_arginfer_style/r1_best5900/summaries_256_marginal

Files: summary.json, comparison.csv, report.md, paper_results.md, rank_histograms.pdf, interval_coverage_width.pdf, and per-method clade probability/score files.

Tests: validation/tests/test_evaluate_arginfer.py and validation/tests/test_clade_comparison.py (13 passed).
