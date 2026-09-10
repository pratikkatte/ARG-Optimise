Enable the optional continuous time policy with a fresh output directory:

```bash
python train.py --config config.yaml --time-policy cwr_exponential \
  --output-path runs/cwr_exponential_new
```

YAML accepts `time_policy: cwr_exponential`. The default is `categorical`.
Both TB and SubTB, both event policies, and both breakpoint policies support
continuous timing. Inference reads the mode from checkpoint metadata:

```bash
python infer.py --checkpoint runs/cwr_exponential_new/checkpoints/best.pt \
  --output-dir inferred_continuous --num-args 10
```

For the selected event/action/breakpoint, the time network predicts a scalar
correction `g`. It samples a wait from `Exp(lambda0 * exp(g))`, where `lambda0`
is the sum of the environment's pre-action coalescence and recombination rates.
The final layer starts at zero, so initial timing follows the continuous CwR
prior. The selected-action representation is extended with `log1p(current_time)`,
`log(lambda0)`, a recombination indicator, and `breakpoint / num_blocks` (zero
for coalescence). This uses the existing encoder pass and policy optimizer.

The forward score includes the **policy log density**
`log(lambda0) + g - lambda0 * exp(g) * delta_t`.
The biological prior includes the **original prior log density**
`log(lambda0) - lambda0 * delta_t`; the learned correction never enters it.
Event, pair/lineage, breakpoint, and likelihood definitions are unchanged.
Samples are detached when scoring the policy, retaining the score gradient
`1 - lambda_theta * delta_t` through the rate. Densities may exceed one and
their log densities may be positive. Continuous reward densities and old
categorical reward masses are not directly comparable scores.

Waits and rate arithmetic, log densities, and accumulated rollout scores use
float64; neural weights retain their existing dtype. Executable continuous
actions carry `delta_t`, with no `time_action`. Waits are applied directly in
internal units of **2Ne generations**; exported `.trees` node ages remain in
generations. Nonfinite or nonpositive rates/waits, nonfinite log scores, and
event increments that cannot advance the float64 clock raise errors. Waits
are never clipped, floored, binned, or resampled after numerical failures.

Backward transitions reconstruct waits by subtracting consecutive event
timestamps. The newest allocated node IDs identify the last event, including
the two parents created by recombination. The backward probability is over
valid reverse events; reversing stored timing adds no exponential density.
The triangular transformation between waits and chronological event times
has determinant one. TB/SubTB, source log-Z, and terminal reward boundaries
are unchanged.

At inference temperature `T`, timing samples follow `Exp(lambda_theta / T)`.
Reported forward scores always use the untempered policy; when `T != 1` they
are **not the behavior density**. Temperature must be finite and positive.
Other action components retain their existing temperature conventions.

Continuous checkpoints record scheme `cwr_exponential_v1` and internal units
`2Ne`. Bin settings apply only to categorical timing. Missing policy metadata
means categorical for legacy checkpoints. Loading weights or optimizer state
across timing modes is rejected; start a fresh checkpoint in a separate output
directory. There is no automatic conversion.

An exponential distribution still has its mode at zero and standard deviation
equal to its mean. Learning a conditional rate changes its scale but cannot
represent arbitrary waiting-time shapes. Full 25 kb training and any MSE
improvement require separate experiments.
