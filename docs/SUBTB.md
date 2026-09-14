# SubTB training

The active diagnostic is a fresh 2 kb joint policy-and-flow run with the
simulation's mutation rate `1e-7`, recombination rate `5e-8`, and Ne `10000`:

```bash
conda run --no-capture-output -n phylogfn_orig python -u \
  validation/reports/subtb_trainability_2026-09-10/benchmark.py \
  --config config_learned_event_2k.yaml --updates 10000 --fixed-episodes 128 \
  --output runs/sim_2kb_subtb_hudson_neural_source
```

Use a fresh output directory. The diagnostic runner requires CUDA by default.
The existing background job is recorded in that directory's `launch.json`;
do not launch another job over it. Its 10,000 joint updates use 128 training
trajectories each, with no resumed parameters and no flow warm-up. The objective
remains making SubTB trainable on `validation/datasets/sim_10k`; the shorter
sequence provides faster diagnostics first. See the
[current experiment report](validation/reports/subtb_trainability_2026-09-10/hudson_neural_source_experiment.md).

## Neural source and intermediate flows

`flow_head_version: 5` has no trainable logZ scalar. The same scalar-output flow
network predicts source and intermediate log flows, while terminal flows equal
the exact posterior log reward. There are only policy and flow optimizer groups.
The separate fixed flow encoder and useful likelihood/state features remain.

Let `Phi(s)` be the tracked partial log likelihood, `C` the reward offset, and
`h(s) = (active_material / num_blocks - 1) / (num_sequences - 1)`. For source and
intermediate states:

```
log F_theta(s) = C + accumulated_log_prior(s) + Phi(s)
              + h(s) * (flow_init_offset - C - Phi(source))
              + flow_output_scale * flow_head(features(s))
```

The offset and scale are fixed checkpointed initialization buffers. They are
computed from 200 unchanged initial-policy trajectories, using the mean and SD
of `log reward + log PB - log PF` (SD floored at one). This centering is not an
exact partition estimate. At the source, the fixed terms reduce to the offset,
and the shared network supplies the learned correction. Subsequent optimizer
updates cannot move either buffer. Both source and intermediate endpoints
receive flow-head gradients. Exact terminal rewards receive no flow gradient.

The head receives the separate encoder's summary, accumulated prior, elapsed
time, lineage/material counts, partial likelihood, remaining material fraction,
and lineage-age mean/SD. `flow_likelihood.py` tracks float64 site-resolution
pruning partials and normalization factors, separately from normalized policy
features. Recombination partitions likelihood contributions; nonoverlapping
common ancestry preserves their sum. At termination, the potential equals the
independently computed JC69 likelihood. Inference skips this extra tracker.

The legacy `compute_log_Z()` reporting API delegates to the neural source flow
for version 5; it does not create a scalar parameter. New logs use
`source_log_flow` / `eval_source_log_flow`. `log_z_lr` and `log_z_grad` do not
apply to version 5.

## Hudson prior and Gamma policy

`arg_prior: hudson` permits common-ancestor events between every unordered
lineage pair, including nonoverlapping material. In units of 2Ne, each pair's
hazard is 1. Recombination uses every integer link between a lineage's leftmost
and rightmost ancestral bases, including trapped gaps. Each one-base link has
hazard `2Ne*r`. Lineage selection is proportional to eligible links; breakpoint
selection is uniform over them. Continuous waits have an exponential prior with
the summed event hazard. Thus an event with hazard `a` has log prior
`log(a) - total_hazard*wait`.

The environment retains resolved material until every position has an MRCA,
corresponding to msprime's `stop_at_local_mrca=False` convention. This retains
extra unary ancestral history; local TMRCA and topology are the validation
targets. The convention is explicit in checkpoints. Independent simulation at
the actual 2 kb parameters agrees with msprime's marginal TMRCA distributions
under both stopping conventions; see the experiment report and JSON results.
Legacy checkpoints missing `arg_prior` retain the old overlap-only prior.

`continuous_time_head: gamma` learns waiting-time mean and shape:

```
shape = exp(h)
rate = prior_total_event_rate * exp(g) * shape
mean_wait = 1 / (prior_total_event_rate * exp(g))
```

The mean is independent of shape. Shape one recovers the exponential policy.
Scores use the exact Gamma density at the fixed sampled wait. The prior remains
exponential. `time_policy: cwr_exponential` selects this continuous-time contract;
`continuous_time_head` selects the learned proposal family separately. The
legacy default remains `exponential`.

## Objective and optimization

Every subtrajectory contributes squared log-balance error with weight
`lambda ** (length - 1)`, normalized within each trajectory, then averaged over
the batch. There is no auxiliary TB loss. This is geometric
[Subtrajectory Balance (Madan et al., 2023)](https://proceedings.mlr.press/v202/madan23a.html).
Lambda zero uses one-step segments; one weights all segments equally; values
above one favor longer segments. The current run uses 2.0. No lambda value is
a universal trainability guarantee.

`subtb.py` uses a stable weighted mean/variance recurrence with O(BT) scalar work
and autograd storage. It computes mixture fractions from log weights and keeps
flow differences, residuals, and losses in float64. Padding contributes neither
loss nor gradients. A zero-action trajectory uses the source/reward boundary.

The configuration uses policy LR `1e-4`, flow LR `1e-3`, and separate gradient
clipping at 10. Both policy and flow train from the first update. Optional
`flow_warmup_steps` remains available for other experiments; it is zero here.
Warm-up alone or passing tests does not complete the trainability goal.

## Evaluation and saved artifacts

Every 50 updates, 256 fresh terminal samples supply fresh-policy SubTB and
sampling-quality evaluation. Initialization and every 250 updates use three
independent repeats. A fixed set of 128 initial-policy paths is evaluation-only;
replay recomputes current policy probabilities and state flows and verifies exact
terminal rewards. Complete actions and a SHA256 fingerprint are retained.

Evaluation and report generation run without gradients and restore Python,
NumPy, environment, CPU/CUDA Torch RNGs and every module's original mode.
Training loss is smoothed over 50 updates. Importance ESS applies only to fresh
policy samples, not the fixed off-policy trajectories.

The new `tmrca_method: point_accuracy` uses an independent source copy of the
calculation called by `validation/scripts/point_accuracy_gfn.py`, in
`validation/scripts/point_accuracy_subtb.py`. It imports neither original file.
TMRCA sample-mean RMSE, spread, and descriptive 5th–95th interval coverage use
exact genomic-span weights over the union of breakpoints, with equal weights
for haplotype pairs. Times are in 2Ne units. Truth/sample identities and units
are verified against simulation metadata and all exported truth-site alleles.

Structural diversity uses canonical rooted clades on a fixed 100-position grid,
ignoring internal IDs, branch times, and unary nodes. It includes topology
frequencies and average pairwise local normalized rooted RF. Truth RF uses exact
genomic spans. Other metrics include reward mean/median/SD/quantiles, sampling
validity, recombination and marginal-tree counts, ESS/N, maximum normalized
importance weight, log-weight SD, full residual mean/SD/RMSE, and terminal and
intermediate SubTB contributions. Empirical interval coverage is descriptive,
not a guarantee of posterior calibration.

The runner retains every evaluated checkpoint plus initial/latest/best-fixed/
best-on-policy files, with optimizer and sampling RNG states. It updates compact
Markdown/CSV tables and PNG/PDF curves under `sampling_report/` after each
evaluation. Protocol version 2 identifies the new exact-span TMRCA calculation;
comparison code rejects mixed protocols. Re-evaluate checkpoints with
`validation/reports/subtb_trainability_2026-09-10/evaluate_checkpoints.py`.
Do not select posterior quality solely from the lowest SubTB checkpoint.

## Learning-rate scheduling

Learning rates remain constant by default. Both `train.py` and the diagnostic
`validation/reports/subtb_trainability_2026-09-10/benchmark.py` accept these
optional settings:

```yaml
lr_schedule: cosine
lr_schedule_steps: 10000
lr_warmup_steps: 500
lr_warmup_start_factor: 0.1
lr_min_factor: 0.1
```

The configured policy and flow learning rates are the peaks. One multiplier
applies to every optimizer group, preserving their relative rates. Starting at
10% of the peaks, the learning rates rise linearly for 500 joint updates, then
decay along a cosine to 10% at update 10,000. Rates stay at that floor if training
continues beyond the schedule. There are no restarts. This uses the cosine form
documented by [PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html),
with a multiplicative floor for the separate policy/flow groups.

The opt-in [500 bp configuration](config_learned_event_500_cosine.yaml) uses policy
peak `1e-4` and flow peak `1e-3`, so both their starting and final rates are
`1e-5` and `1e-4`, respectively. This is an experiment configuration, not a
validated ESS improvement. Existing comparison configurations retain their
constant rates.

Scheduling advances after each joint optimizer update, once per accumulated
batch. Evaluation and the optional flow-only initialization phase do not
advance it. Flow-only initialization uses the starting learning rate. Logged
`policy_lr` and `flow_lr` are the rates used for the completed update; optimizer
and checkpoint rates are prepared for the next update.

Checkpoints retain schedule configuration, peak rates, and completed updates
alongside Adam state. Resume with the same cosine settings; `lr_schedule_steps`
can be explicit so shorter diagnostic runs keep the intended schedule horizon.
When that setting is zero, the diagnostic runner restores the saved horizon
on resume. Conflicting schedule settings or explicit LR overrides during a
scheduled resume are rejected. Introducing a schedule to an older constant-LR
checkpoint starts at its existing joint-update count. Weight-only inference
does not restore a training schedule.

For example, use the same flags with any of the four variant configurations:

```bash
python train.py --config config_learned_event_500_exploration.yaml \
  --output-path runs/sim_500_subtb_hudson_exploration_cosine \
  --lr-schedule cosine --lr-schedule-steps 10000 --lr-warmup-steps 500
```

## Compatibility

The optional `exploration_fraction` and `replay_fraction` settings mix fresh
prior paths and training-only replay within the existing trajectory budget.
They default to zero. Replayed actions are rescored with current PF/PB and
flows; reward and learning rates remain unchanged. The buffer combines a
reservoir with a topology-capped high-reward archive and local-tree-stratified
draws. Full configuration, exclusions, RNG and buffer contents are checkpointed.
See [controlled exploration/replay](validation/reports/subtb_trainability_2026-09-10/controlled_exploration_replay.md).

Action probability version 2 fixes a single-candidate tensor squeeze that
made PF depend on other batch members. The sole legal action now has log
probability zero. New checkpoints identify the corrected scorer. Trained
version-1 optimizer states require explicit action-probability migration;
loading weights alone uses corrected scoring and warns about changed metrics.
Empty update-0 checkpoints remain valid common initializations. The corrected
comparison uses a new pure-policy baseline and two optional exploration runs.

TB remains the constructor/CLI default. Existing 10 kb settings and historical
flow versions 1–4 remain available for reading older experiments; the current
2 kb diagnostic explicitly selects version 5. Inference reconstructs saved
prior, timing, and flow versions automatically. Ordinary loading across priors,
objectives, time-head families, or flow versions fails explicitly. Changing
lambda while restoring an optimizer is also rejected.

Legacy versions use a source-only scalar logZ. Version 4 has an independent
fixed intermediate baseline; versions 2/3 preserve their older live-logZ
baseline for compatibility. Explicit version-3-to-4 flow migration preserves
predictions and Adam moments. Old 256-value Z checkpoints sum into a scalar
with a warning and reset only Z's Adam history. Explicit exponential-to-Gamma
migration applies to legacy scalar-source checkpoints and preserves existing
policy/flow weights and moments. None of these migrations is used for the fresh
version-5 run.

The main `train.py` checkpoint names retain their previous meanings:
`best.pt` minimizes training TB; `best_eval_loss.pt` minimizes evaluation TB;
`best_residual_mean.pt` and `best_residual_std.pt` minimize their residual metrics;
`best_eval_subtb_loss.pt` minimizes evaluation SubTB. The diagnostic runner uses
the explicit fixed/fresh SubTB checkpoint names described above.
