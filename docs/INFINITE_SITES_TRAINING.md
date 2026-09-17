# Infinite-sites neural training (Phase 2)

The policy and state flow now use **one shared trainable encoder**. This supports fresh SubTB training, compatible exploration, topology-diverse replay, self-contained checkpoints, resumed training, inference, and optional truth evaluation. These workflows accept a simulator replicate directory containing VCF, exact position map, and metadata. FASTA inputs and JC69 checkpoints are rejected.

## Start a run

From the repository root:

```bash
python3 train.py --config config/config_infinite_sites_rep0.yaml
```

The supplied config runs 20 updates on `validation/datasets/sim_5k_mr20/rep0`: batch size 2, seed 7, 25% replay after eight entries, and a 10,000-event limit. Population size and per-generation rates come from the replicate metadata unless explicitly overridden. Overrides are recorded in the checkpoint. A fresh run requires an empty output directory.

Outputs include `training.jsonl` and numbered checkpoints. The default is CPU with one PyTorch thread. Environment state always stays on CPU in float64; neural execution may use CUDA. MPS is unsupported because scoring requires float64.

Resume to a new total update count:

```bash
python3 train.py \
  --resume-checkpoint runs/infinite_sites_rep0_phase2_seed7/checkpoints/checkpoint_0020.pt \
  --output-path runs/infinite_sites_rep0_phase2_seed7 \
  --epochs 40
```

Resume restores observations, model and optimizer weights, scheduler state when present, replay storage, RNG states, and the completed update count. The saved run determines batch size, event limit, score-chunk size, architecture, rates, and replay configuration. The original dataset directory is optional. If supplied, its observations must match. Scientific rates and architecture cannot change on resume.

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

`--dataset-path` is optional in evaluation. Providing it enables truth-based TMRCA and topology summaries using the explicit metadata sample-node mapping. Continuous SNP positions and observed alleles are checked against that mapping. Training, policy features, and replay initialization never read truth ancestry. ESS is reported only for fresh untempered policy draws; passing a smoke test does not establish posterior calibration on rep0.

A dead end, event limit, or zero-likelihood policy result fails the run. `failure.json` retains attempted action histories; inference also retains records for earlier completed batches. There is no automatic rejection/resampling or reward floor.

## Representation and probability model

`policy/observations.py` packs ragged SNP and material tokens. SNP tokens contain `(a,d,log1p(m))`, normalized continuous position, adjacent observed-SNP gaps, a completion flag, and separate binary vectors for observed derived carriers and local descendants. Material tokens contain normalized boundaries and length, completion status, and descendant bits. Integer bitsets are decoded, never converted wholesale to floating point.

Separate two-layer MLPs pool SNPs by mean and maximum, and material intervals by length-weighted mean and maximum. Empty SNP pools are zero with an explicit `has_snps` scalar; their material is still encoded. Time, lineage age, material length, span, interval count, and SNP count accompany the pools. Messages remain defined at their stored node time. Network widths depend on sample count, not the number of SNPs or physical bases; models remain specific to one observed dataset.

The projected lineage embeddings and a summary token enter the lineage Transformer. A shared state representation then feeds separate action and flow heads. The encoder is registered once and receives gradients from both. Its parameters use the policy learning rate; the flow head has its own optimizer group. An optional internal cache retains only immutable numerical input features, with weak references and a bounded byte budget. Age and state statistics are refreshed, and learned embeddings are always recomputed.

The policy factors into event, pair/lineage, breakpoint, and waiting-time terms. Event logits use **allowed** hazard totals plus learned residuals. Coalescence choices are compatible pairs; recombination lineage baselines are their physical link counts. The breakpoint head is a four-component mixture of truncated discretized logistics over the entire integer span, including trapped gaps and SNP-free intervals. The Gamma timing head uses the **unmasked physical** Hudson rate as its baseline. All four factors are normalized and accumulated in float64. The environment independently scores the physical prior, with no compatibility adjustment.

Every generated timed history has one chronological predecessor, so its backward log probability is zero. Undoing a recombination removes its paired parent nodes together; predecessor reconstruction uses environment restoration and timed replay.

For unfinished states:

```
log F(s) = C + accumulated_log_prior + Phi(s)
           + w(s) * (B0 - C) + sigma * flow_head(shared_features)
w(s) = (total_carried_length / physical_length - 1) / (sample_count - 1)
```

`Phi` is the Phase 1 closed-edge potential. `B0` is the mean initial-policy `log R - log PF`; `sigma` is the corresponding standard deviation, bounded below by one for scaling the flow output. These are fixed initialization buffers, not exact estimates of the evidence. The source flow is learned with the shared network and has no separate scalar log-Z parameter. The terminal boundary equals the exact log reward. The potential is not a likelihood marginalized over future completions.

Training retains the exact all-segment SubTB loss. To bound memory on long histories, it first computes loss derivatives with respect to numerical forward scores and flows, then replays neural scores in short chunks to accumulate their parameter gradients. The chain rule gives the same gradient as a direct full-trajectory graph; tests compare the two. The architecture has no dropout, and parameters stay unchanged between passes. Frozen-feature flow warm-up is rejected.

## Checkpoints and verification

Checkpoints store the observed binary matrix, float64 positions, haplotype order, allele labels, physical length, rates, fingerprint, model/feature/flow versions, weights, optimizer, scheduler, replay, and RNG states. They contain no ground-truth ancestry. Replay schema 2 stores timed actions, exact prior and reward, proposal provenance and density, and topology signatures; current neural scores are always recomputed.

Fast scientific and neural checks:

```bash
python3 -m pytest validation/tests/test_snp_data.py \
  validation/tests/test_infinite_sites.py \
  validation/tests/test_infinite_sites_environment.py \
  validation/tests/test_infinite_sites_neural.py -q
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
