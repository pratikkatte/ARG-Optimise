# Infinite-sites neural training (Phase 2)

The policy and state flow now use **one shared trainable encoder**. This supports fresh SubTB training, compatible exploration, topology-diverse replay, self-contained checkpoints, resumed training, inference, and optional truth evaluation. These workflows accept a simulator replicate directory containing VCF, exact position map, and metadata. FASTA inputs and JC69 checkpoints are rejected.

## Start a run

From the repository root:

```bash
python3 train.py --config config/config_infinite_sites_rep0.yaml
```

The supplied config runs 20 updates on `validation/datasets/sim_5k_mr20/rep0`: batch size 2, seed 7, 25% replay after eight entries, and a 10,000-event limit. Population size and per-generation rates come from the replicate metadata unless explicitly overridden. Overrides are recorded in the checkpoint. A fresh run requires an empty output directory.

Outputs include `training.jsonl` and numbered checkpoints. The default is CPU with one PyTorch thread. Environment state always stays on CPU in float64; neural execution may use CUDA. MPS is unsupported because scoring requires float64.

At sampling temperature one, training retains the fresh trajectories' neural activations and uses their policy scores and flows directly in the full SubTB loss. Each microbatch calls `loss.backward()` once, with no score recomputation for gradients. Compatible-proposal and replay-buffer paths are scored once under the current policy. Higher-temperature samples are generated without gradients, then scored under the temperature-one policy. Activations are released after each microbatch; gradients accumulate until one clip and optimizer/scheduler step per update. These graphs are not stored in checkpoints.

Console output shows initialization start and completion, plus one concise `Z init` line per initialization batch with the completed trajectory count, mean events per ARG, and batch elapsed time. With `init_z_batch_size: 1`, this prints after every initialization trajectory. Training then prints one line per epoch with SubTB loss and elapsed time, including evaluation SubTB loss when evaluation runs (averaged across repeats). Detailed metrics remain in JSONL files and W&B. Use `--no-verbose` to suppress training console output.

Resume to a new total update count:

```bash
python3 train.py \
  --resume-checkpoint runs/infinite_sites_rep0_phase2_seed7/checkpoints/checkpoint_0020.pt \
  --output-path runs/infinite_sites_rep0_phase2_seed7 \
  --epochs 40
```

Resume restores observations, model and optimizer weights, scheduler state when present, replay storage, RNG states, and the completed update count. The saved run determines batch size, architecture, rates, replay configuration, and learning-rate schedule. Runtime settings such as the event limit and evaluation cadence may be overridden. The original dataset directory is optional. If supplied, its observations must match. Scientific rates and architecture cannot change on resume.

## Sample and evaluate

```bash
python3 infer.py \
  --checkpoint runs/infinite_sites_rep0_phase2_seed7/checkpoints/checkpoint_0020.pt \
  --output-dir runs/infinite_sites_rep0_phase2_seed7/inference \
  --num-args 16 --batch-size 2 --seed 100007
```

Inference uses the observations embedded in the checkpoint and needs no source dataset or truth files. Every generated ARG is checked with the independent infinite-sites evaluator. Outputs are ancestry-only `.trees` files and a manifest of exact likelihoods, physical priors, forward densities, rewards, action histories, and fresh-policy importance diagnostics. Sampling temperature is one.

```bash
python3 eval/eval.py \
  --checkpoint runs/infinite_sites_rep0_phase2_seed7/checkpoints/checkpoint_0020.pt \
  --output-dir runs/infinite_sites_rep0_phase2_seed7/evaluation \
  --num-samples 16 \
  --dataset-path validation/datasets/sim_5k_mr20/rep0
```

`--dataset-path` is optional in evaluation. Providing it enables truth-based TMRCA and topology summaries using the explicit metadata sample-node mapping. Continuous SNP positions and observed alleles are checked against that mapping. Optimizer updates, policy features, and replay initialization never read truth ancestry. Optional periodic evaluation can read truth after an update. ESS is reported only for fresh untempered policy draws; passing a smoke test does not establish posterior calibration on rep0.

A dead end, event limit, or zero-likelihood policy result fails the run. `failure.json` retains attempted action histories; inference also retains records for earlier completed batches. There is no automatic rejection/resampling or reward floor.


## Full experiment configuration

`config/config_infinite_sites_rep0_cosine_replay.yaml` is the full infinite-sites equivalent of `config/config_learned_event_2k_cosine_replay_fresh.yaml`. The original file is unchanged. The new file targets CUDA, 10,000 optimizer updates, 256 trajectories per update, eight gradient accumulation microbatches, cosine decay, topology-diverse replay, periodic evaluation, and W&B. Its explicit scientific rates match rep0.

```bash
python3 train.py --config config/config_infinite_sites_rep0_cosine_replay.yaml
```

A smaller CUDA check, using the same architecture and scheduler:

```bash
python3 train.py --config config/config_infinite_sites_rep0_cosine_replay.yaml \
  --epochs 20 --batch-size 2 --grad-accum-steps 1 \
  --init-z-sample-count 8 --eval-episodes 8 --eval-batch-size 2 \
  --checkpoint-every 10 --no-wandb --output-path runs/infinite_sites_rep0_cuda_check
```

Use `--device cpu` when CUDA is unavailable. The CUDA-specific test is skipped on machines without CUDA; the reference environment remains CPU float64. The full configuration is an experiment setup, not a demonstrated convergence recipe.

All declared YAML keys also have CLI overrides, except the nested `evaluation` mapping. Boolean overrides include `--no-wandb`, `--no-terminal-eval`, and `--no-verbose`. Unknown keys fail before training. Every run writes `resolved_config.yaml`, `configuration_notes.json`, training logs, and checkpoint metadata containing the effective configuration and rates.

| Controls | Meaning |
| --- | --- |
| `epochs`, `batch_size` | Total optimizer updates and total trajectories per update. |
| `grad_accum_steps` | Split each update into microbatches; backpropagate the full SubTB loss directly for each microbatch. Clip and step once per full update. |
| `chunk_steps` | Inactive legacy setting, accepted for existing configs and checkpoints. |
| `replay_fraction`, `replay_capacity`, `replay_grid_size`, `replay_per_topology`, `replay_min_size` | Topology-diverse replay: reservoir plus topology-capped elite storage. The topology quota applies to the elite half. |
| `exploration_fraction` | Fraction from the compatible diagnostic proposal; actual proposal and physical prior densities remain distinct. |
| `lr_schedule`, `lr_schedule_steps`, `lr_warmup_steps`, `lr_warmup_start_factor`, `lr_min_factor` | Constant or warm-up/cosine schedule, applied to policy/shared and flow-head rates. Scheduler progress and base rates resume exactly. |
| `policy_temperature_schedule`, `policy_temperature_start`, `policy_temperature_anneal_steps` | Constant temperature one, or discrete linear annealing toward one. Annealing currently requires replay and compatible exploration fractions zero. Timing and rewards remain untempered. Higher-temperature samples are rescored at temperature one; fresh temperature-one scores are reused. |
| `embedding_size`, `hidden_size`, `transformer_depth`, `transformer_heads`, `transformer_mlp_ratio` | Shared encoder, pooling projections, lineage Transformer, and head dimensions. |
| `breakpoint_mixture_hidden_dim`, `breakpoint_mixture_layers`, `breakpoint_mixture_components` | Hidden width, hidden MLP-layer count, and logistic-mixture count on the shared action/span representation. No nucleotide CNN is used. |
| `breakpoint_gap_hidden_size`, `breakpoint_gap_layers` | Width and number of additional hidden layers before mixture parameter outputs. |
| `continuous_time_head`, `time_hidden_dim`, `time_layers` | Gamma, Gamma-mixture, or exponential head, hidden width and hidden-layer count. The full physical Hudson rate remains the baseline. |
| `time_mixture_components` | Number of components when `continuous_time_head: gamma_mixture`. Each component learns its mean and shape. Policy scores marginalize component identity with logsumexp; components are not additional ARG actions. Existing Gamma/exponential heads are unchanged. |
| `policy_lr`, `flow_lr`, `subtb_lambda`, `grad_clip` | Shared/policy and flow-head learning rates, all-segment SubTB weighting, and global clipping threshold. Per-component unclipped norms are logged. |
| `init_z_sample_count` | Number of initial-policy trajectories defining fixed flow-centering/scaling buffers. There is no separate scalar Z. |
| `effective_population_size`, `mutation_rate`, `recombination_rate`, `reward_C` | Explicit scientific overrides and reward offset. Omitted rates come from the dataset metadata. |
| `wandb`, `wandb_project`, `wandb_entity`, `wandb_name`, `wandb_mode`, `verbose` | Optional W&B logging and console output. Offline mode is supported. JSONL output is always written. |
| `checkpoint_every`, `resume_checkpoint`, `max_events`, `seed`, `device`, `cpu_threads` | Persistence, update-level resumption, per-history event limit, random seed, neural device, and PyTorch CPU thread count. |

### Legacy settings with explicit restrictions

The older file cannot be used unchanged: its FASTA path, `flow_head_version: 5`, and `breakpoint_dropout: 0.1` need replacement. The new file makes these changes explicitly.

- `dataset_path` must identify an infinite-sites replicate directory. The selected legacy file's 2 kb rates must not be mistaken for the 5 kb rep0 rates.
- `flow_head_version` must be **6**; version 5 identifies the retired flow architecture. `flow_warmup_steps` must be zero because frozen-feature warm-up is incompatible with a shared trainable encoder.
- `dropout`, `attention_dropout`, and `breakpoint_dropout` must be zero. Stochastic masks would make sampled, replayed, and gradient-recomputed policy scores inconsistent. Nonzero values are rejected rather than quietly disabled.
- `time_bins` and `time_delta_bin_width` are accepted legacy fields, explicitly recorded as **inactive** under continuous timing.
- `breakpoint_hidden_dim` is accepted and recorded as **inactive**: it controlled the retired nucleotide CNN. Use the mixture/gap widths for the current head.
- `flow_warmup_episodes` is accepted and recorded as **inactive** while warm-up is disabled.
- Supported scientific/model selections remain `arg_prior: hudson`, `bp_per_blocks: 1`, `event_policy: cwr_residual`, `loss_type: subtb`, `time_policy: cwr_exponential`, and `breakpoint_policy: sparse_mixture`.

### Periodic and standalone evaluation

`eval_episodes`, `eval_every`, and `eval_batch_size` enable fresh temperature-one draws after optimizer updates; evaluation also runs after the final update. `eval_density_slope` adds full-history and prior-relative density fits. `eval_independent_likelihood` controls independent rescoring **only in evaluation**. Ordinary training and replay rewards always use the incremental tracker.

`terminal_eval` enables optional truth summaries. `terminal_eval_grid_size` and `tmrca_method` configure those summaries. At multiples of `terminal_eval_repeat_every`, `terminal_eval_repeats` independent evaluation batches are run, each containing `eval_episodes` trajectories. RNG and module modes are preserved, so evaluation does not alter the next training trajectory. Evaluation samples never enter replay. `best_eval.pt` is selected by the mean fresh-policy SubTB evaluation loss; truth metrics are never used for checkpoint selection.

Training writes `evaluation.jsonl` and individual reports under `evaluation/`. W&B receives one combined metric record per update. No independent evaluator is called inside an optimizer step.

The nested `evaluation` mapping configures the separate posterior-evaluation command:

```bash
python3 eval/eval.py --config config/config_infinite_sites_rep0_cosine_replay.yaml
```

`checkpoint: best_eval` resolves to the run's `checkpoints/best_eval.pt`; an explicit checkpoint can override it. `metrics`, `num_samples`, `repeats`, `batch_size`, `seed`, `device`, `grid_size`, and `rank_bins` control fresh draws and optional truth summaries. `density_fit` also creates a separate compatible-proposal bank using `bank_candidates` and `bank_per_stratum`, excluding histories retained in training replay. All candidates and their proposal provenance are recorded; insufficient bank support is an error. The bank is never added to replay and has no importance ESS. Fresh-sample ESS is reported separately for each repeat.

## Representation and probability model

`policy/observations.py` packs ragged SNP and material tokens. SNP tokens contain `(a,d,log1p(m))`, normalized continuous position, adjacent observed-SNP gaps, a completion flag, and separate binary vectors for observed derived carriers and local descendants. Material tokens contain normalized boundaries and length, completion status, and descendant bits. Integer bitsets are decoded, never converted wholesale to floating point.

Separate two-layer MLPs pool SNPs by mean and maximum, and material intervals by length-weighted mean and maximum. Empty SNP pools are zero with an explicit `has_snps` scalar; their material is still encoded. Time, lineage age, material length, span, interval count, and SNP count accompany the pools. Messages remain defined at their stored node time. Network widths depend on sample count, not the number of SNPs or physical bases; models remain specific to one observed dataset.

The projected lineage embeddings and a summary token enter the lineage Transformer. A shared state representation then feeds separate action and flow heads. The encoder is registered once and receives gradients from both. Its parameters use the policy learning rate; the flow head has its own optimizer group. An optional internal cache retains only immutable numerical input features, with weak references and a bounded byte budget. Age and state statistics are refreshed, and learned embeddings are always recomputed.

The policy factors into event, pair/lineage, breakpoint, and waiting-time terms. Event logits use **allowed** hazard totals plus learned residuals. Coalescence choices are compatible pairs; recombination lineage baselines are their physical link counts. The breakpoint head is a four-component mixture of truncated discretized logistics over the entire integer span, including trapped gaps and SNP-free intervals. The Gamma timing head (or the configurable exponential head) uses the **unmasked physical** Hudson rate as its baseline. All four factors are normalized and accumulated in float64. The environment independently scores the physical prior, with no compatibility adjustment.

Every generated timed history has one chronological predecessor, so its backward log probability is zero. Undoing a recombination removes its paired parent nodes together; predecessor reconstruction uses environment restoration and timed replay.

For unfinished states:

```
log F(s) = C + accumulated_log_prior + Phi(s)
           + w(s) * (B0 - C) + sigma * flow_head(shared_features)
w(s) = (total_carried_length / physical_length - 1) / (sample_count - 1)
```

`Phi` is the Phase 1 closed-edge potential. `B0` is the mean initial-policy `log R - log PF`; `sigma` is the corresponding standard deviation, bounded below by one for scaling the flow output. These are fixed initialization buffers, not exact estimates of the evidence. The source flow is learned with the shared network and has no separate scalar log-Z parameter. The terminal boundary equals the exact log reward. The potential is not a likelihood marginalized over future completions.

Training differentiates the exact all-segment SubTB loss directly through the full trajectory graphs in each microbatch. There is no SubTB window or chunked score recomputation. This removes a neural scoring pass at the cost of retaining activations for all events in that microbatch. Increase `grad_accum_steps` to reduce concurrent trajectories if needed; a single long trajectory must still fit in memory. Legacy `chunk_steps` values have no effect, including on resume. Tests compare direct gradients with the former chunked calculation. The architecture has no dropout. Frozen-feature flow warm-up is rejected.

## Checkpoints and verification

Checkpoints store the observed binary matrix, float64 positions, haplotype order, allele labels, physical length, rates, fingerprint, model/feature/flow versions, weights, optimizer, scheduler, replay, and RNG states. They contain no ground-truth ancestry. Replay schema 2 stores timed actions, exact prior and reward, proposal provenance and density, and topology signatures; current neural scores are always recomputed.

Fast scientific and neural checks:

```bash
python3 -m pytest validation/tests/test_snp_data.py \
  validation/tests/test_infinite_sites.py \
  validation/tests/test_infinite_sites_environment.py \
  validation/tests/test_infinite_sites_neural.py \
  validation/tests/test_infinite_sites_configuration.py -q
```

The fixed-budget statistical acceptance study is separate:

```bash
python3 validation/scripts/validate_infinite_sites_neural.py \
  --output-dir /tmp/infinite_sites_posterior_acceptance
```

It uses seeds 7, 17, and 27; 2,000 updates and 5,000 independently checked draws per fixture. Two-sample examples with zero or one SNP have a Gamma waiting-time posterior. A three-sample singleton example checks independently derived topology probabilities and both waiting-time means. All outcomes, including failed gates, are reported. No seed selection or training-budget extension occurs automatically.

Multi-dataset learning, legacy checkpoint conversion, and a long rep0 convergence study remain outside Phase 2.

To audit the supplied rep0 smoke checkpoint, including all retained training histories, initialization draws, and 16 fresh checkpoint samples:

```bash
python3 validation/scripts/validate_rep0_neural_run.py \
  --checkpoint runs/infinite_sites_rep0_phase2_seed7/checkpoints/checkpoint_0020.pt \
  --output-dir /tmp/infinite_sites_rep0_acceptance \
  --dataset-path validation/datasets/sim_5k_mr20/rep0
```
