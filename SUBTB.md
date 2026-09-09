Optional SubTB training
======================

TB remains the default. Start a fresh SubTB run with:

```bash
python train.py --config config.yaml --loss-type subtb \
  --subtb-lambda 0.9 --flow-lr 0.0001 \
  --output-path runs/human_25kb_subtb
```

The YAML settings are `loss_type: tb | subtb`, `subtb_lambda: 0.9`, and
`flow_lr: 0.0001`. CLI options override YAML. Omitting `flow_lr` uses the resolved
`policy_lr`. Lambda must be finite and nonnegative; the flow learning rate must
be finite and positive. Lambda zero uses one-step segments, one weights all
segments equally, and values above one favor longer segments. Lambda 0.9 is a
starting value, not a validated optimum for ARG inference.

The objective uses every subtrajectory, with weight `lambda ** (length - 1)`,
normalized separately within each trajectory before averaging the batch. This
implements geometric [Subtrajectory Balance (Madan et al., 2023)](https://proceedings.mlr.press/v202/madan23a.html).
There is no auxiliary TB mixture loss.

`subtb.py` evaluates the objective without enumerating segments. At each ending
position, a weighted mixture combines the new one-step residual with the
previous suffix sums extended by the current residual. If the previous mean and
central variance are `m` and `v`, and the normalized weight of the extended
suffixes is `a`, the new moments are `delta + a*m` and
`a*v + a*(1-a)*m*m`. The implementation computes both mixture fractions from
log weights rather than subtracting one from the other. A second weighted
running average combines the suffix second moments. This gives O(BT) scalar
work and autograd storage. Flow differences, residuals, and the loss use float64;
the existing policy keeps its precision. Padding does not contribute to the loss
or gradients. A zero-action trajectory uses the source-versus-reward residual.

Only SubTB models have a flow head. A two-layer MLP uses the shared state summary
and scaled accumulated prior, time, active-lineage count, and active-block count.
Its last layer starts at zero. Intermediate log flows equal the fixed initial
policy log-Z estimate plus accumulated log prior plus the head output. Source
flows use trainable log Z; terminal flows use the full posterior log reward.
Rollouts collect each scalar flow once without extra encoder calls or ARG-state
copies. Both intermediate endpoints receive gradients. Head construction
preserves policy and sampling RNG state. Head parameters have a separate Adam
learning rate and share policy gradient clipping; log Z remains separately
optimized and excluded from clipping.

Training reports `loss` for the selected objective, `subtb_loss` and `tb_loss`
when SubTB is enabled, `flow_head_grad_norm`, and trajectory-length median/p95/max.
Evaluation retains the existing TB metrics and adds `eval_subtb_loss` and length
quantiles. Evaluation restores RNG states and individual module modes.

Checkpoint meanings remain:

- `best.pt`: lowest training TB loss, including when training uses SubTB.
- `best_eval_loss.pt`: lowest evaluation TB residual MSE.
- `best_residual_mean.pt`: evaluation residual mean closest to zero.
- `best_residual_std.pt`: lowest evaluation residual standard deviation.
- `best_eval_subtb_loss.pt`: lowest evaluation SubTB loss, for SubTB runs.

Checkpoints store the objective, resolved loss/learning-rate settings, flow-head
version, and fixed initialization offset. Inference reconstructs the saved
architecture and skips flow collection. Missing objective metadata identifies a
legacy TB checkpoint. Loading across objectives is rejected; this version does
not introduce TB-to-SubTB conversion or a training-resume CLI. Changing lambda
while restoring an optimizer is also rejected.

Do not compare raw TB and SubTB objective values as quality measures. Use common
TB residuals, posterior reward, exported ARG accuracy, diversity, and elapsed
time. The bounded validation results are recorded in
[validation/reports/subtb_2026-09-09](validation/reports/subtb_2026-09-09).
