# Learning-rate control

YAML-driven training uses optimizer-update-based warmup and cosine decay by
default. This matters because one training `epoch` in this repository is one
optimizer update after all configured gradient-accumulation batches.

```yaml
training:
  epochs: 1000
  policy_lr: 1.0e-4
  breakpoint_policy_lr: 1.0e-4
  time_policy_lr: 3.0e-4
  log_z_lr: 1.0e-3
  lr_scheduler:
    type: cosine
    warmup_steps: null
    warmup_fraction: 0.05
    warmup_start_ratio: 0.1
    min_lr_ratio: 0.1
```

With this example, the structural, breakpoint, and time-policy groups start at
10% of their respective base learning rates, warm up over 50 optimizer updates,
and then cosine-decay to 10% of their base rates. A TB run also schedules the
`log_z` group. The training history and W&B logs contain `lr/used/<group>` for
the rate consumed by the completed update, `lr/<group>` for the next update,
plus `lr/factor` and `lr/optimizer_step`.

Available scheduler types are:

- `constant`: optional warmup followed by the base learning rates.
- `cosine`: optional warmup followed by cosine decay.
- `step`: optional warmup followed by multiplicative reductions every
  `step_size` optimizer updates.
- `plateau`: optional warmup followed by evaluation-metric-driven reductions.

Step reduction uses `step_size`, `step_gamma`, and `min_lr_ratio`:

```yaml
lr_scheduler:
  type: step
  warmup_fraction: 0.05
  step_size: 100
  step_gamma: 0.5
  min_lr_ratio: 0.1
```

Plateau reduction uses evaluation events rather than training batches:

```yaml
lr_scheduler:
  type: plateau
  warmup_fraction: 0.05
  plateau_metric: auto
  plateau_mode: min
  plateau_factor: 0.5
  plateau_patience: 5
  plateau_threshold: 0.001
  plateau_threshold_mode: rel
  plateau_cooldown: 0
  min_lr_ratio: 0.1
```

`plateau_metric: auto` selects the first finite available metric in this order:

1. fixed-bank SubTB MSE;
2. fixed-bank terminal MSE;
3. ordinary FL-SubTB, SubTB, or TB evaluation MSE;
4. local-refinement evaluation loss.

It intentionally does not fall back to stochastic training loss. Set
`plateau_metric: loss` explicitly if update-loss-driven reduction is desired.
Scheduler state is stored in every checkpoint and restored together with the
optimizer when `TBGFlowNetGenerator.load(..., load_optimizer=True)` is used
with the same scheduler configuration and training horizon.

## Three-model W&B analysis

The policy is optimized as three disjoint parameter groups: `structural`,
`breakpoint`, and `time`. Each group can have its own learning rate and clipping
threshold:

```yaml
training:
  policy_lr: 1.0e-4
  breakpoint_policy_lr: 1.0e-4
  time_policy_lr: 3.0e-4
  grad_clip: 10.0
  breakpoint_gradient_clip_norm: 10.0
  time_head_gradient_clip_norm: 1.0
  model_diagnostics: true
  model_diagnostics_update_norm_every: 1
```

W&B groups all model-health series below `models/<model>/`. The important
optimization signals are `gradient_present`, `gradient_finite_rate`,
`gradient_nonfinite_detected`, `gradient_zero_rate`, `parameter_finite_rate`,
`grad_norm_before_clip`, `grad_norm_after_clip`, `gradient_clipped`,
`clip_scale`, `param_norm`, `update_norm`, `relative_update_norm`,
`update_applied`, `update_finite`, `lr_used`, and `lr_next`. Exact update norms
require a parameter snapshot; `model_diagnostics_update_norm_every` can reduce
that cost on large runs while leaving gradient diagnostics enabled every
update.

Behavior series are grouped below `models/<model>/behavior/<rollout_mode>/`:

- Structural: action entropy, normalized entropy, selected and maximum action
  probability, coalescence/recombination probability mass, and realized event
  ratio.
- Breakpoint: decision count, support size, entropy, normalized entropy, and
  selected/maximum breakpoint probability.
- Time: sample count, mixture entropy, effective component count, quantile
  mean, boundary rate, and finite-density rate.

When a fixed evaluation bank is enabled, deterministic rescoring also logs
`models/<model>/fixed_bank/` metrics. These include selected-action NLL for the
structural and breakpoint categoricals, selected log density for time, finite
rates, normalized categorical entropy, and a
`recombination_conditioned_residual_*` series. The latter is a diagnostic for
transitions on which the breakpoint head participates; it is not a claim that
the breakpoint head alone caused the full flow residual.

A missing gradient or negligible relative update identifies a disconnected or
stalled head. A low finite-gradient rate or persistent heavy clipping identifies
instability. Falling normalized entropy with selected probability close to the
maximum identifies policy concentration; judge whether that is useful learning
or premature collapse alongside fixed-bank flow residuals and reward metrics.
