# Fix 01: importance-ESS diagnostics

**Status:** Implemented; full POC verification pending.

For each trajectory, the unnormalized log importance weight is

```text
log_w = log_reward + log_path_pb - log_path_pf
```

The implementation normalizes these values with `logsumexp` in float64 and
reports:

- `importance_ess` and `importance_ess_fraction`;
- `importance_max_weight`;
- `importance_log_weight_range`;
- TB residual mean, standard deviation, MSE, and RMSE when logZ is available.

Calculations reject empty, unequal-length, and non-finite inputs. ESS is
invariant to an additive shift of every log weight. Deterministic evaluation
preserves and restores Python, NumPy, Torch, CUDA, and environment RNG state.

Quick metrics use the `eval_` prefix. Checkpoint-quality panels use the
`convergence_` prefix and are written to local history, W&B, checkpoint
metadata, and `convergence_report.json`. Inference manifests contain the same
diagnostics for the generated run.

Automated tests cover equal weights, a single dominant weight across 2,000 log
units, shift invariance, invalid inputs, and consecutive-panel state.

