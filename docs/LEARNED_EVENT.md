# Optional learned event policy

Run `python train.py --config config_learned_event.yaml` to use the learned
event selector with the existing sparse-mixture breakpoint policy and SubTB
settings. This configuration writes to
`runs/human_25kb_super_easy_sparse_mixture_subtb_learned_event`.
`--event-policy cwr` overrides the YAML; omitted settings default to `cwr`.

For `event_policy: cwr_residual`, the event distribution is
`softmax(log P_CwR(event | state) + correction)`.
The correction head takes the encoded state summary, `log1p(current_time)`,
`log1p(active_lineage_count)`, and `total_active_blocks / num_blocks`.
It uses `Linear → SiLU → Linear(2)` with `hidden_size` hidden units and a
zero-initialized final layer, so initial probabilities match the CwR prior.
Events with zero prior probability or no valid actions remain masked.

The head trains at `policy_lr` through TB or SubTB. Each rollout step shares
one state encoding between event selection and action scoring. The selected
event log probability contributes once to the forward score. Inference
temperature changes sampling; forward scores use the untempered policy.

Checkpoints store `event_policy` inside model metadata, which inference uses
to reconstruct the architecture. Missing metadata selects the original CwR
architecture. Loading across event-policy modes raises an error.

`eval_initial_coalescence_prob` and `eval_initial_recombination_prob` report
the forward policy. The corresponding `eval_initial_prior_coalescence_prob`
and `eval_initial_prior_recombination_prob` report the CwR prior. Existing
trajectory-length and recombination-count metrics remain available.

Focused verification: `OMP_NUM_THREADS=1 python -m pytest -q tests/test_learned_event.py`.
It covers analytical two-event learning under both objectives, masks,
forward scores, CPU training, checkpoint restoration, and inference with both
breakpoint policies. Full 25 kb training and posterior-convergence experiments
are separate follow-up work.
