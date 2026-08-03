# Breakpoint-aware node-time refinement

## Outcome

The breakpoint-aware time head is implemented without changing the ARG action
semantics, reward target, FL-SubTB objective, Bernstein-beta family, or CwR
quantile/Jacobian accounting.  On the fixed seed-7 400--500 kb refinement,
longer training provides a coherent positive timing signal by epoch 30.

The original 2% local-RMSE target is not yet demonstrated.  Under the revised
criterion (evidence that longer training can improve this input ARG), the result
is promising: epoch 30 improves local pairwise-TMRCA RMSE by 1.15% relative to
the matched full-context epoch-10 run, while log-RMSE, rank correlation, and
posterior coverage all improve.

| Run | Posterior samples | Local RMSE (generations) | Local log-RMSE | Spearman | 95% coverage | Full 1 Mb RMSE (generations) |
|---|---:|---:|---:|---:|---:|---:|
| A baseline context, time LR 1e-3, epoch 10 | 15 | 21,575.66 | 0.84880 | 0.72368 | 0.04015 | 16,770.74 |
| E full context, time LR 1e-3, epoch 10 | 16 | 21,579.04 | 0.84814 | 0.72229 | 0.03984 | 16,771.17 |
| E full context, time LR 1e-3, epoch 20 | 16 | 21,601.65 | 0.84507 | 0.71922 | 0.04991 | 16,774.09 |
| E full context, time LR 1e-3, epoch 30 | 13 | **21,330.52** | **0.83306** | **0.75132** | **0.07238** | **16,739.31** |
| E full context, time LR 1e-3, epoch 50 | 7 | 21,986.52 | 0.84307 | 0.74430 | 0.04407 | 16,824.05 |

Epoch 30 versus matched E epoch 10 is -1.1517% local physical RMSE and
-0.1900% whole-chromosome RMSE.  Epoch 30 versus the A epoch-10 baseline is
-1.1362% locally.  These are posterior-sample estimates from one seed, so they
establish potential rather than a final multi-seed effect size.

The requested clean 50-epoch run completed after correcting a floating-point
fixed-boundary edge case.  Epoch 50 is worse than epoch 30: local physical RMSE
is 3.08% higher than epoch 30 and 1.89% higher than epoch 10.  It also exported
only 7 of 16 terminal trajectories.  The other nine reached terminal states but
failed tskit serialization because extremely short successive branches rounded
some parent/child node times to equality.  Thus this configuration begins to
overtrain or lose numerical separation after epoch 30; epoch 30 remains the
best retained checkpoint.

## Implementation

- Structural actions remain atomic: coalescence/recombination selection and
  breakpoint sampling are unchanged externally.
- Time context is computed after the structural choice and, for recombination,
  after the realized breakpoint.
- The full context contains explicit temporal bounds, symmetric lineage/pair
  summaries, mutation/variant summaries, genomic material/exposure summaries,
  breakpoint-conditioned left/right summaries, and local model parameters.
- Baseline, temporal-only, breakpoint-only, and full-context modes are
  configuration-selectable and checkpointed with a versioned schema.
- A separate optional time-policy learning rate permits targeted optimization;
  `1e-3` was stable, while `1e-2` was too aggressive for the full context.
- Diagnostics record time entropy/effective component count, sampled quantile,
  physical delta, boundary distances, context strata, time-head gradient norm,
  and time-containing FL-SubTB residuals.
- Validation supports exact interval clipping and reports physical/log RMSE,
  log correlation, Spearman correlation, and posterior 95% coverage.
- A frozen-structure score-grid benchmark is included for topology-independent
  inspection of the learned conditional time density.

## Ablation interpretation

With the original shared `1e-4` learning rate, A/B/D/E were numerically
identical after 10 epochs and the 16-component time mixture stayed almost
uniform.  This identified optimization, rather than representational capacity,
as the immediate bottleneck.  A dedicated time LR of `1e-3` moved the density
without the collapse seen at `1e-2`.

At ten epochs, the matched A-versus-E comparison was essentially tied
(E was 0.016% worse in physical RMSE but slightly better in log-RMSE).  The
frozen-structure benchmark was mixed across seven event strata.  Therefore the
epoch-30 gain should not be claimed as a causal effect of breakpoint context
alone without a matched A epoch-30 and multiple seeds.  It does show that the
implemented time-learning path can extract additional timing signal from this
ARG with longer optimization, which is the revised success criterion.

## Reproducible artifacts

- Epoch-30 checkpoint:
  `runs/time_context_E_full_timelr1e3_seed7_epoch50/checkpoints/epoch_000030.pt`
- Epoch-30 inference:
  `runs/time_context_E_full_timelr1e3_seed7_epoch50/checkpoint_curve/epoch30/`
- Exact local validation:
  `validation/output/time_context_E_full_timelr1e3_seed7_epoch30/results/summary.tsv`
- Learning-curve/full-track metrics and plots:
  `runs/time_context_results/learning_curve_seed7_through_epoch50/`
- Epoch-50 checkpoint and inference:
  `runs/time_context_E_full_timelr1e3_seed7_epoch50_fixed/`
- Frozen-structure A/E benchmark:
  `runs/frozen_time_A_vs_E_timelr1e3_seed7_fast/`

Verification: `python -m pytest -q` completed with 153 passed and 8 skipped.
The only warning was the environment's existing NumPy/SciPy compatibility
warning; the fallback path completed successfully.
