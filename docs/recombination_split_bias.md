# Paper-inspired recombination split bias

## Purpose

Local backward-time recombination selects one active lineage and a breakpoint,
then splits that lineage into two ancestral fragments. The optional split bias
uses current VCF partials to prefer splits whose left and right fragments have
better support from different active ancestral lineages than the unsplit
lineage. It is a local approximation of the building-block and linkage ideas in
Stadler and Wagner's algebraic theory of recombination spaces; it does not build
or diagonalize the full recombination-space Laplacian.

The feature changes the proposal policy only. It does not change legal actions,
the CWR prior, likelihood, terminal reward, backward policy, or checkpoint
parameter shapes.

## Score

For fragment `X` of lineage `i` and an overlapping active lineage `j`, the
compatibility is

```text
C(X, j) = sum_v [2 clamp(dot(partial_i[v], partial_j[v]), 0, 1) - 1]
          / number_of_variants_carried_by_X
```

The sum covers variants carried by both lineages. Missing shared variants
therefore contribute zero. The best support `B_X` is the maximum of zero and
the compatibility over all other overlapping active lineages.

For breakpoint `b`, with left/right variant counts `n_L`, `n_R` and physical
material lengths `m_L`, `m_R`, the split score is

```text
G(i, b) = (n_L B_L + n_R B_R) / (n_L + n_R)
          - B_i
          - fragmentation_penalty * abs(m_L - m_R) / (m_L + m_R)
```

The lineage score is a prior-weighted log-mean-exp over valid breakpoints. The
weights are the existing VCF gap-length breakpoint prior, so a lineage does not
receive a larger score merely because it has more valid breakpoints.

## Policy integration and invariant

The existing structural action model and breakpoint model remain separate.
The aggregated lineage score biases which lineage is selected, while the
breakpoint-specific score biases the existing breakpoint logits.

After applying rollout temperature, the lineage adjustments are recentered so
the log-sum-exp over all recombination actions is exactly the same as before the
feature. Consequently, conditional on choosing a generated event, the total
probability assigned to recombination is preserved. The bias only redistributes
that mass among recombination lineages. Breakpoint probabilities are conditional
on an already-selected recombination and may change normally.

The invariant applies at any positive rollout temperature. Different selected
lineages and breakpoints can still lead to different future states, so complete
trajectory lengths and total recombination counts must still be monitored.

When the local CwR event gate is enabled, it owns the total coalescence versus
recombination probability. The split bias then acts only on the conditional
recombination-lineage distribution. Its group correction is the difference
between the recombination log-sum-exp after and before adding lineage scores;
subtracting that common value keeps the recombination group's unnormalized mass
unchanged before conditional normalization.

## Configuration

```yaml
model:
  recombination_split_bias:
    enabled: false
    score_mode: partial_compatibility_v1
    lineage_weight: 0.25
    breakpoint_weight: 0.25
    aggregation_temperature: 1.0
    fragmentation_penalty: 0.10
```

The feature is supported only for local phased-VCF refinement. It has no
trainable parameters. The normalized configuration is stored in checkpoint
metadata; old checkpoints default to disabled.

Per-transition diagnostics record candidate and selected scores, the selected
atomic adjustment, recombination mass before and after adjustment, and the
absolute preservation error. Aggregates are logged below
`models/recombination_split/behavior/*` and
`models/recombination_split/fixed_bank/*`.

## Matched A/B experiment

Use these configurations:

- Off: `config/config_1mb_local_refinement_flow_consistency.yaml`
- On: `config/config_1mb_local_refinement_flow_consistency_split_bias.yaml`

Keep seeds and transition budgets matched. Compare raw fixed-bank terminal,
one-step, and SubTB MSE together with terminal reward, recombination count,
trajectory length, sampled transition count, and runtime. A lower scaled
training loss alone is not evidence that the feature improves posterior
sampling or credit assignment.

## Limitations

- The score is a deterministic compatibility heuristic, not a learned action
  value and not the full spectral construction from the paper.
- It uses only variants visible in the current local state and does not estimate
  LD or long-range epistasis separately.
- Immediate recombination probability is preserved, but downstream behavior is
  not guaranteed to remain length matched.
