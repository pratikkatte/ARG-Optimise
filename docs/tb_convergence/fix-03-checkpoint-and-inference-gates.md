# Fix 03: checkpoint and inference gates

**Status:** Implemented; full POC verification pending.

## Checkpoint lifecycle

- `last.pt` records the latest convergence-panel state, or the final
  unevaluated state when convergence evaluation is disabled.
- `best_candidate.pt` records the lowest deterministic convergence-panel TB
  residual RMSE and is diagnostic only unless its metadata later passes.
- `best.pt` is created only by a checkpoint that has passed three consecutive
  panels. Stochastic training-batch loss cannot create it.

Checkpoint metadata contains convergence schema version, thresholds, panel
seed/index, evaluation size, metrics, current-panel result, consecutive passes,
overall pass status, and checkpoint kind. `convergence_report.json` provides the
complete panel history and selected paths.

## Inference behavior

Inference validates convergence before constructing the model or writing an
output directory. A failed, unevaluated, missing-metadata, or v7 checkpoint is
rejected by default. `--allow-unconverged` permits diagnostic proposal sampling
and records the override in the manifest.

V7 checkpoints are never certified as posterior checkpoints. Under the
diagnostic override, their 256-element logZ vector is summed into the v8 scalar
and their missing reward offset is interpreted as the legacy `30000.0`.

Automated tests cover strict rejection before output creation, explicit
override behavior, v7 handling, metadata pass behavior, and manifest fields.

