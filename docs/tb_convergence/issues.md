# Known GFN issues

## Current evidence

The `example2_best_n100` run is not a valid posterior approximation:

- exact length-weighted TMRCA MSE is `0.8985`, versus `0.1469` for SINGER;
- weighted truth/prediction correlation is `-0.066`, versus `0.915` for SINGER;
- normalized importance ESS is effectively `1/100` and one sample receives
  essentially all normalized importance weight;
- the best observed evaluation TB residual mean remains about `+195`, and all
  100 training updates exceeded the gradient clipping threshold;
- the initial mixed recombination probability is `0.848`, versus `0.364` under
  the CwR prior;
- inferred ARGs have a median of 23 marginal trees, versus 10 in the truth;
- rooted marginal-tree clade F1 is about `0.041`, versus `0.702` for SINGER.

These measurements show that unweighted model draws are proposal samples, not
posterior samples. Posterior accuracy comparisons are meaningful only after
the convergence gate passes.

## Issue list

1. **TB non-convergence and importance-weight collapse.** Training-batch loss
   selected the old checkpoint despite large residuals and unusable ESS. This
   phase addresses this issue.
2. **Event/topology collapse.** The policy generates excessive recombination
   and marginal topologies largely unrelated to truth. Deferred until a
   converged TB baseline exists.
3. **Time-bin scale mismatch.** Fixed bins cover only `0..0.063` in `t/(2Ne)`;
   most long waits share one deterministic tail representative. Deferred.
4. **Insufficient time conditioning.** The time head does not receive explicit
   rates, current time, child ages, or event type. Deferred.
5. **Lossy sequence representation.** Fifty-base blocks are represented by
   average nucleotide composition, discarding aligned mutation patterns.
   Deferred.
6. **Breakpoint quality.** Breakpoint overproduction and poor localization are
   more serious than the 50-base coordinate grid itself. Deferred.
7. **Legacy metric inconsistency.** GFN's printed legacy MSE rounds values before
   aggregation, whereas `commonMetrics.tsv` contains the comparable exact
   length-weighted MSE. Deferred.

## Exit condition for this phase

A v8 checkpoint must pass three consecutive 256-rollout panels with ESS/N at
least `0.25`, absolute residual mean at most `1.0`, and residual RMSE at most
`2.0`, then reproduce ESS/N at least `0.25` on an independent held-out
256-rollout inference run.

