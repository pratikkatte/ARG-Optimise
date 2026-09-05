# Fix 02: TB training stabilization

**Status:** Implemented; full POC verification pending.

## Changes

- LogZ is one scalar parameter instead of the sum of 256 independently updated
  parameters.
- The terminal reward offset is explicit checkpoint/configuration metadata.
  New training uses `0.0`; v7 diagnostic loading restores the legacy `30000.0`.
- Training records global gradient norm before and after clipping, logZ gradient
  norm, and pre-clipping norms for the encoder, event, action, breakpoint, time,
  and uncategorized parameter groups.
- The stabilized POC run uses policy LR `1e-4`, logZ LR `5e-2`, global clipping
  at `10`, at most 1,000 epochs, and early stopping on convergence.

The reward offset changes only a common log-reward constant and therefore does
not change the target distribution. It reduces large-number cancellation while
the faster scalar logZ update addresses residual mean directly.

## Verification

Unit tests verify scalar checkpoint round trips and legacy vector-logZ
conversion. The POC run must record whether each module remains clipping-bound
and must meet the convergence criteria in `issues.md`; otherwise no posterior
checkpoint is produced.

