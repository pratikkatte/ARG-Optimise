# Stable paper-dataset training

These fresh-run templates read Ne, mutation rate, and recombination rate from each observed dataset's metadata. The three datasets correspond to `validation/config/paper_dataset/r_{1,2,4}.yaml`; the observations and scientific target are not changed.

The templates are candidates under validation, not established converged settings. The implementation adds:

- `time_parameterization: bounded_v1`: Gamma shapes between 1 and 20, smooth log mean-rate corrections between −3 and 3, with exactly the same marginal density for sampling and replay. These bounds constrain the proposal, not the Hudson target prior.
- `state_feature_transform: signed_log`: compresses unbounded prior/likelihood neural inputs, leaving the actual reward unchanged.
- Separate `encoder_lr`, `flow_encoder_grad_scale`, and optional `tb_loss_weight` controls. Training logs distinguish total objective, SubTB, and full-trajectory residual loss.
- `best_checkpoint_metric: eval_log_weight_std`: selects `best_eval.pt` by calibration rather than the changing on-policy training loss. Fresh ESS and density slopes remain separate diagnostics.
- `eval_initial: true`: a real untrained baseline at the same evaluation sample count.
- `max_wall_seconds`: graceful interruption after a completed update, writing `checkpoints/latest.pt` and an explicit `run_status.json`. SIGUSR1/SIGTERM request the same graceful behavior; send them with enough lead time for a rollout/evaluation to finish.

Evaluation writes a resumable checkpoint before sampling. A numerical evaluation failure remains a failure, with actions/diagnostics where available and nonzero W&B exit status. The independent likelihood evaluator subtracts internal node times before converting branch durations to generations; it still checks the same tolerance and uses independent tree traversal.

For example:

```bash
python train.py --config config/paper_datasets/stable/r2.yaml
python train.py --resume-checkpoint runs/paper_datasets_stable/r2_seed7/checkpoints/latest.pt \
  --output-path runs/paper_datasets_stable/r2_seed7 --device cuda --epochs 10000
```

Do not change scientific rates, architecture, optimizer settings, or objective while doing an exact resume. Existing checkpoints without the new transform metadata retain the original timing/input semantics. They do **not** silently become bounded models; use a fresh run to adopt the new parameterization.

The first measured pilots use batch 32, two transformer layers, `initial_recombination_bias: -1.5`, one microbatch, and 50 warm-up updates. The bias changes only the starting residual event policy, retains positive probability for every compatible event, and is learned normally afterward. Their frozen source/configs/logs are in `runs/paper_datasets_stable/pilot_20260921_a`. The larger batch-64/depth-6 templates above remain explicit alternatives; pilot evidence will determine the recommended final configuration.

The historical convergence-pilot launcher has been retired. `validation/scripts/evaluate_convergence.py` compares repeated fresh-policy samples and one fixed independent history bank; it never reports an ESS on a selected bank.
