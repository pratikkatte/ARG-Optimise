# GFlowNet flow-consistency audit

## Status and current conclusion

The implementation now has a terminal-only production training and evaluation
mode. Every rollout supplied to an optimizer or ordinary evaluation step must
reach a terminal state. `complete_trajectory_max_steps` is a failure guard: a
trajectory that reaches the cap raises an error and is not used as a partial
sample. The low-level SubTB routine still understands nonterminal paths so its
mathematics and regression tests remain available for a later, deliberate
reintroduction of partial trajectories.

One mathematical bookkeeping bug was confirmed and corrected: when a rollout
temperature `T != 1` was used, atomic actions and recombination breakpoints were
sampled from tempered logits but their recorded forward log-probabilities came
from untempered logits. The recorded `log P_F` now uses the exact distribution
that sampled each component.

The previously suspected fixed-attachment mismatch is not present in the
current factorization. A fixed attachment has a learned forward proposal-gate
probability and deterministic backward probability one. Its biological target
survival mass is included once in the accumulated prior/reward. Proposal and
target probabilities need not be equal; they must not be conflated.

The current A100 one-step experiments do **not** meet the flow-consistency
acceptance criteria. Residual scaling makes the optimized loss numerically
small, but fixed-bank residuals remain almost unchanged after one optimizer
step. This is conditioning, not evidence that consistency is fixed. There is
also no evidence here about posterior accuracy.

## Audited equations

For a generated local transition,

```text
log P_F = log P(generated gate)
        + log P(atomic structural action)
        + log P(recombination breakpoint | recombination), if applicable
        + log q(delta_t | state, action, generated before boundary)
```

For a fixed attachment,

```text
log P_F = log P(fixed-attachment gate)
```

The continuous proposal density is evaluated in physical scaled-time units:

```text
log q(delta_t) = log q_U(u)
               + log(lambda) - lambda * delta_t
               - log(1 - exp(-lambda * H))
```

where the final term is present only when a fixed boundary at horizon `H`
truncates the generated-event interval. This is the conditional-CDF Jacobian.
The fixed-boundary target-prior increment is the complementary survival log
mass `-lambda * H`.

The backward policy is uniform over valid structural parents:

```text
log P_B(parent | child) = -log(number of valid parents)
```

Fixed attachment has exactly one parent, hence `log P_B = 0`. Coalescence and
recombination inverse construction are tested by applying a forward action,
enumerating the child state's inverse actions, and reconstructing the original
structural identity.

State flow is:

```text
log F(s) = learned potential(s)                         for SubTB
log F(s) = partial target score(s) + learned potential(s) for FL-SubTB
log F(x) = log R(x)                                    for terminal x
```

For any subtrajectory `s_i -> ... -> s_j`, the raw residual is

```text
delta_ij = log F(s_i) + sum(log P_F)
         - log F(s_j) - sum(log P_B)
```

Only subtrajectories whose endpoint is the actual completed trajectory endpoint
are classified as terminal. A terminal term is never also placed in the
internal component. Optimization uses `delta_raw / residual_scale`, while all
evaluation and percentile metrics remain in raw log units.

To preserve old behavior exactly at the defaults, internal and terminal
weighted sums use the historical all-subtrajectory denominator:

```text
loss = internal_scaled_loss
     + terminal_loss_weight * terminal_scaled_loss
```

With `terminal_loss_weight=1` and `residual_scale=1`, this is bit-for-bit the
legacy SubTB objective in the regression test.

## Dominant causes and classifications

| Finding | Classification | Evidence |
|---|---|---|
| Tempered samples recorded untempered atomic/breakpoint `log P_F` | Corrected mathematical bug | Sampling/rescoring reconstruction tests at `T=0.7` pass |
| Terminal terms were mixed into a much larger undifferentiated SubTB aggregate | Diagnostic/optimization design problem | Terminal terms now have independent raw MSE/count/percentiles and explicit weight |
| Partial/terminal alternating production batches could starve and obscure the boundary constraint | Correctness-stage sampling problem | Production train, benchmark, and ordinary evaluation are now terminal-only |
| Raw residuals make time-head gradients large in the baseline | Optimization-conditioning problem | Baseline one-step time gradient norm 40.77 before clipping; configurable scaling/LR/clipping added |
| Similarity bias changes proposal mass and generates longer, recombination-heavy paths | Behavior-changing heuristic | Bias 1.0 produces ~59% longer and ~99% more recombinations in matched A100 smoke |
| Fixed-attachment proposal gate vs target survival probability | Not a current bug | Gate is in `P_F`; survival is in target prior once; deterministic `P_B=1` and inverse tests pass |

No terminal reward, prior definition, likelihood, action space, or
Bernstein-beta time family was changed.

## Files and principal functions

| File | Functions or areas changed |
|---|---|
| `models.py` | `ARGModel._score_candidates`, `ARGModel.forward`: exact-zero invalid action probabilities and tempered recorded action probability |
| `breakpoint_model.py` | Both breakpoint samplers: record probability from the sampled tempered logits |
| `tb_gfn.py` | `TBGFlowNetGenerator`: local gate factorization, deterministic transition rescoring, support assertions, separated SubTB objective/metrics, curricula, time optimizer group/clipping/warm-up, flow decomposition |
| `rollout_worker_arg.py` | `RolloutWorker._rollout_batch`: record PF components, policy mass/count diagnostics, backward parent counts, reconstruction/support assertions |
| `refinement/training.py` | `train_local_refinement`, `evaluate_local_refinement`: complete terminal trajectories only, fixed bank, curriculum, W&B/debug records, terminal and behavior checks |
| `benchmark_gflownet_workflow.py` | Complete-only benchmark loop, all flow configuration forwarding, fixed-bank evaluation, ablation CLI overrides |
| `flow_evaluation.py` | Versioned deterministic complete-trajectory bank generation, signatures, merge/coverage validation, policy rescoring and raw fixed-bank metrics |
| `tiny_exact_flow.py` | Fully enumerable two-terminal exact-flow environment and trainer |
| `train.py` | Backward-compatible config defaults, validation, and forwarding; production mode restricted to `complete` |
| `tests/test_flow_consistency.py` | Exact loss/scaling tests, exact enumerable convergence, inverse-parent, action mask, terminal-only config, bank determinism |
| `tests/test_integrated_local_gfn.py` | Local gate/PF decomposition, gradient, fixed attachment inverse, rescoring, fixed-bank integration |
| `tests/test_terminal_prefix_training.py` | Production regression now asserts terminal rate 1 and absence of partial metrics |

## Configuration

Backward-compatible defaults are:

| Field | Default | Meaning |
|---|---:|---|
| `training.trajectory_training_mode` | `complete` | Only accepted production mode for the correctness stage |
| `training.complete_trajectory_max_steps` | `null` | Optional fail-fast cap; never converts a rollout to a training partial |
| `training.min_terminal_trajectories_per_batch` | `0` | Lower-bound assertion; complete-only mode naturally supplies the whole batch |
| `training.terminal_loss_weight` | `1.0` | Multiplier on the separately computed terminal component |
| `training.residual_scale` | `1.0` | Fixed optimization-only divisor |
| `training.time_policy_lr` | `null` | Uses the structural policy LR when unset |
| `training.time_head_gradient_clip_norm` | `null` | Uses the global clip when unset |
| `training.time_head_warmup_epochs` | `0` | Time head is trainable immediately |
| `training.subtb_lambda_initial/final` | `null` | Fixed `subtb_lambda` remains active |
| `training.subtb_max_span_schedule` | `[]` | Fixed `subtb_max_span` remains active |
| `training.flow_debug` | `false` | Do not emit per-transition decompositions |
| `training.probability_checks` | `false` | Disable expensive runtime support assertions |

The recommended first complete-only experiment is
`config/config_1mb_local_refinement_flow_consistency.yaml`. Its key changes are
`lambda=0.6`, span 4, terminal weight 10, residual scale 50, time LR 0.0003,
time clip 1, similarity bias 0.25, and at least four terminal trajectories per
optimizer step.

## Diagnostics and W&B

Training logs raw internal one-step, internal multi-step, terminal one-step,
terminal multi-step, total raw MSE, signed mean, absolute mean, RMSE,
p50/p90/p95/p99, action-type groups, span groups, and length-normalized RMSE.
Scaled internal/terminal optimization components and their weighted total are
logged separately. Policy diagnostics include action-type probability mass,
valid action counts, selected probabilities, recombination/coalescence ratio,
length, first fixed attachment, and reward/prior/likelihood by recombination
count. Time-head gradient norms are logged before and after clipping.

With debug mode enabled, `flow_decomposition.jsonl` contains state IDs, action
type, source/destination flow, partial score and learned potential, every PF/PB
component, totals, reconstruction errors, and the raw one-step residual.

Fixed-bank metrics use the requested `flow_eval/fixed_bank_*` names and include
action/source breakdowns. Evaluation always rescales trajectories under the
current model but uses a fixed evaluator definition (`lambda=0.9`, span 16), so
changing the training curriculum cannot change the metric definition.

## Validation

CPU suite: `188 passed, 8 skipped`. CUDA targeted suite on an NVIDIA
A100-SXM4-80GB: `28 passed`. The exact two-terminal DAG has rewards 1 and 3,
partition 4, and target terminal probabilities `(0.25, 0.75)`. It converges to
maximum absolute residual below `1e-3`; terminal weights 1 and 10 and residual
scales 1, 50, and 100 have the same exact optimum and keep both terminals
reachable.

The current fixed bank is seed/version/signature controlled, has 32 complete
terminal trajectories, 7 coalescence-heavy and 25 recombination-heavy paths,
and fixed attachments in all 32. It currently has one recorded trajectory
source (`similarity_0.25_initial`). The bank merge and required-source
validation are implemented, but a final acceptance bank must add a separately
generated baseline source before making multi-source claims.

### One-epoch A100 diagnostic ablation

All rows use seed 7, identical architecture, 16 complete training trajectories,
32 ordinary complete evaluations, and the same fixed bank. Because lengths
differ, sampled transition counts are included. This is a smoke comparison,
not a three-seed or matched-transition acceptance result.

| Run | Fixed terminal MSE | Fixed one-step MSE | Fixed SubTB MSE | Train terminal MSE | Train internal 1-step MSE | Recombs | Length | Transitions | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline complete-only | 97,113 | 3,007 | 27,945 | 99,949 | 2,086 | 37.25 | 67.94 | 1,087 | 271 |
| Similarity 1.0 | 96,949 | 2,997 | 27,826 | 97,285 | 1,200 | 74.19 | 107.75 | 1,724 | 466 |
| Similarity 0.25 | 97,084 | 3,005 | 27,940 | 98,853 | 1,753 | 44.88 | 76.75 | 1,228 | 295 |
| Similarity 0.25 + terminal weight 10 | 97,016 | 3,005 | 27,941 | 98,853 | 1,753 | 44.88 | 76.75 | 1,228 | 305 |
| Recommended full | 97,003 | 3,005 | 27,940 | 98,853 | 1,753 | 44.88 | 76.75 | 1,228 | 266 |

Similarity 1.0 reduces fixed terminal MSE only about 0.17% while increasing
mean recombinations about 99%, length about 59%, sampled transitions about 59%,
and wall time about 72%. The recommended full row keeps length within 20% of
baseline but recombinations are about 20.5% higher, just outside the requested
behavior bound. Its optimized loss is small because residuals are divided by
50; raw fixed-bank improvements are only about 0.1% after this single step.
At similarity 0.25, terminal weight 10 improves fixed terminal MSE by only
about 0.07% relative to similarity 0.25 alone and does not improve fixed SubTB
after one step. The identical sampled path statistics in those rows are useful:
the small difference is due to gradient weighting, not a different batch.

## What remains before acceptance

1. Run longer matched-transition experiments for at least three seeds. Epoch
   matching is insufficient because complete trajectory lengths differ.
2. Build and version the final bank from both baseline and similarity-trained
   policies, then evaluate every checkpoint on that merged bank.
3. Use transition decompositions on the largest terminal residuals to determine
   whether the remaining ~300-log-unit boundary error is predominantly terminal
   reward scale, state potential, fixed attachment, or a still-unfound support
   error.
4. Require the stated 10x terminal, 2x one-step, and 3x SubTB fixed-bank
   improvements before calling flow inconsistency mitigated.
5. Only after those checks pass should partial trajectories be reintroduced as
   an efficiency feature and tested for non-degradation on the same bank.

Lower TB/SubTB loss alone does not establish reward-proportional posterior
sampling in the ARG environment. At present, posterior accuracy remains
unestablished.
