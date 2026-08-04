# Local CwR-anchored event gate

## Purpose

The local refinement policy has two structural levels. It first chooses between
a generated event and the next fixed-source attachment. For a generated event,
the optional CwR gate then chooses coalescence versus recombination before the
existing action scorer chooses a pair or lineage conditional on that event.

This hierarchy prevents the number of legal candidates in one event class from
implicitly determining the total probability assigned to that class.

## Gate

For the current state, the local CwR construction provides

```text
lambda_C = number of legal overlapping coalescence pairs
lambda_R = rho / 2 * normalized active ancestral-material length
```

The learned recombination log odds are

```text
logit P(R | state) = log(lambda_R) - log(lambda_C) + d_theta(state)
```

`d_theta` is a scalar linear head over the transformer summary representation.
Its weights and bias are initialized to zero, so at rollout temperature 1 the
initial gate exactly reproduces the CwR event probabilities. The residual is
bounded smoothly as

```text
d_theta = bound * tanh(raw_residual / bound)
```

At rollout temperature `T`, the CwR event logits and residual are divided by
`T`, matching the temperature semantics of the other policy decisions. Event
classes with zero rate are masked exactly.

Within the selected class, the existing neural candidate logits are normalized
only over that class. The recombination split bias, when enabled, therefore
changes the conditional lineage distribution but not the CwR event probability.

## Configuration

```yaml
model:
  local_cwr_event_gate:
    enabled: false
    max_abs_residual: 2.0
```

The feature is supported only for local VCF refinement and defaults to disabled.
Disabled or absent configurations retain the previous state dictionary and
probability path. Enabled checkpoints store the normalized configuration and
the residual-head parameters. When warm-starting, the current YAML determines
whether the head exists; an old checkpoint initializes a newly enabled head at
zero through shape-compatible loading. Inference restores the checkpoint's
saved configuration.

## Diagnostics

Each generated decision records the two rates, prior and policy recombination
probabilities, bounded and raw residuals, selected event, and selected-event
probability. Training aggregates these under
`models/cwr_event_gate/behavior/{mode}` and fixed-bank evaluation under
`models/cwr_event_gate/fixed_bank`.

## Matched experiment

- Baseline: `config/config_1mb_local_refinement_flow_consistency.yaml`
- Gate only: `config/config_1mb_local_refinement_flow_consistency_cwr_gate.yaml`
- Split only: `config/config_1mb_local_refinement_flow_consistency_split_bias.yaml`
- Gate + split: `config/config_1mb_local_refinement_flow_consistency_cwr_gate_split_bias.yaml`

The files share the dataset, request, seed, optimizer, transition budget, and
fixed bank. Compare event calibration, residual magnitude, recombination count,
trajectory length, terminal reward, terminal/one-step/SubTB residuals, and
runtime before combining both features.
