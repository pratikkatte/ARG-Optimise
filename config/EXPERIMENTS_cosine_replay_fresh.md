# Fresh cosine and replay experiments

## Question and primary comparison

Does combining cosine learning-rate decay with 25% replay improve posterior
diagnostics and reduce posterior-mean pairwise TMRCA RMSE on sim_500?
This is a hypothesis; the config does not guarantee lower RMSE.

Use `config_learned_event_500_cosine_replay_fresh.yaml` as the common config.
All runs start fresh, with the same model, physical parameters, SubTB objective,
10,000 updates, and batch size 128. The historical cosine config loads an
update-500 checkpoint, so it is not a matched fresh-start control.

| Arm | LR schedule | Replay fraction | Purpose |
| --- | --- | --- | --- |
| baseline | constant | 0 | Fresh reference |
| cosine | cosine | 0 | Effect of LR decay |
| replay | constant | 0.25 | Effect of replay |
| cosine_replay | cosine | 0.25 | Combined effect |

Use seeds 7, 17, and 27 for every arm (12 runs). Matching seeds aligns initial
model weights; subsequent sampling paths differ. Evaluate each arm after the
same number of scored histories, and also report generated histories, transitions,
and wall time: replay reduces the fresh-generation budget. There is no LR
warm-up in this first comparison. Test the historical 500-update LR warm-up as
a separate follow-up. Keep sampling temperature at 1; the current trainer rejects
temperature annealing combined with replay.

## Training commands

Run from the repository root. First inspect the seed-7 pilot for all four arms;
then repeat with seeds 17 and 27. Each command below runs a full 10,000 updates.
Use unused output directories for every new run.

```bash
cfg=config/config_learned_event_500_cosine_replay_fresh.yaml
for experiment_seed in 7 17 27; do
  for arm in baseline cosine replay cosine_replay; do
    schedule=constant
    replay_share=0.0
    case "$arm" in cosine|cosine_replay) schedule=cosine ;; esac
    case "$arm" in replay|cosine_replay) replay_share=0.25 ;; esac
    python train.py --config "$cfg" \
      --seed "$experiment_seed" \
      --lr-schedule "$schedule" --replay-fraction "$replay_share" \
      --output-path "runs/sim_500_subtb_hudson_${arm}_fresh_seed${experiment_seed}" || break 2
  done
done
```

The YAML alone launches the combined arm at seed 7. The loop includes that run;
do not launch both into the same directory. Training CLI overrides are not written
back to the YAML: supply explicit checkpoint and report paths during evaluation.

## Fixed checkpoint evaluation

Predeclare updates **250, 500, 1000, 2000, 4000, 6000, 8000, 10000** for each run.
The primary endpoint is update 10000; use the complete curve to assess stability.
Do not select checkpoints by minimum truth RMSE. `best_eval` is an additional
loss-selected diagnostic, not a replacement for the fixed endpoint.

Use 5 repeats of 1024 fresh temperature-1 policy draws per checkpoint. These
repeats measure sampling variability; the three training seeds measure training
variability. Keep budgets and evaluation seeds identical across comparisons.
Larger sample counts reduce Monte Carlo noise, not model bias.

Create the density bank once, using the baseline seed-7 update-250 checkpoint:

```bash
python eval/eval.py \
  --config config/config_learned_event_500_cosine_replay_fresh.yaml \
  --checkpoint runs/sim_500_subtb_hudson_baseline_fresh_seed7/checkpoints/checkpoint_0250.pt \
  --output-dir validation/reports/sim_500_fresh_convergence/baseline_seed7
```

Freeze and reuse its `density_bank.json.gz` for every subsequent checkpoint,
arm, and training seed on this dataset. For example:

```bash
for step in 0250 0500 1000 2000 4000 6000 8000 10000; do
  python eval/eval.py \
    --config config/config_learned_event_500_cosine_replay_fresh.yaml \
    --checkpoint "runs/sim_500_subtb_hudson_cosine_replay_fresh_seed7/checkpoints/checkpoint_${step}.pt" \
    --density-bank validation/reports/sim_500_fresh_convergence/baseline_seed7/density_bank.json.gz \
    --output-dir validation/reports/sim_500_fresh_convergence/cosine_replay_seed7 || break
done
```

Repeat this evaluation loop for the other arm/seed paths. Checkpoint-specific
subdirectories keep their results separate. Keep the original bank and its
provenance files. If bank creation fails for insufficient unique histories,
increase its candidate budget before freezing the shared bank.

## Evidence and summaries

- **Primary accuracy endpoint:** `eval_truth_pair_tmrca_rmse`, exact genomic-span
  weighted error of the posterior mean against simulated pairwise TMRCA, in
  `2Ne` units. At Ne=10000, multiply by 20000 for generations. Report per-seed
  values and paired differences versus each control, then their mean and range.
- **Posterior fit:** frozen-bank raw and prior-relative density slopes approaching
  1, decreasing centered density RMSE, and increasing fresh-sample ESS/N. Check
  largest importance weight as well. Training loss alone is insufficient; where
  backward histories are nonunique, density fits diagnose trajectory balance.
- **TMRCA summaries:** posterior mean, median, standard deviation, 50/70/90%
  interval coverage and widths, bias, and rank histogram/KL. Plot posterior mean
  and 90% bands along genomic position for a preselected pair (e.g. n0/n1),
  alongside truth; report aggregate errors across all pairs. Saved posterior
  draws and summary artifacts support these plots.
- **Stability:** inspect the 6000/8000/10000 checkpoints for drift in TMRCA mean,
  spread, covariance, RMSE, and ESS/N relative to evaluation-repeat variability.
  Report all checkpoints even when RMSE rises; a better posterior fit need not
  monotonically reduce realized error on one simulated dataset.

Treat lower RMSE with collapsed intervals or poor ESS as inconclusive posterior
progress. Report the factorial interaction in endpoint RMSE as
`combined - cosine - replay + baseline` for each seed; a negative value suggests
an additional combined reduction. Three seeds provide a pilot, not a precise
significance estimate.

## Follow-up after the pilot

1. If the combination helps, compare replay fractions 0.125 and 0.25 at the same
   cosine schedule. If the density fit improves but TMRCA remains unstable,
   test a 20,000-update horizon with matched controls and record the extra cost.
2. Test 500 LR warm-up updates against the zero-warm-up combined arm as a
   separate experiment; keep the horizon and all other settings fixed.
3. Lock the selected settings before testing additional independent simulated
   datasets with matching rates (target at least 10 new 500 bp replicates for an
   initial generalization check). Use a separate frozen bank per dataset.
   Linked genomic positions and repeated draws on rep0 are not independent
   simulated datasets. Aggregate calibration across independent datasets before
   making broader posterior-calibration claims.

See [evaluation documentation](../docs/EVALUATION.md) for metric definitions,
artifact layouts, and optional SINGER reference comparisons.
