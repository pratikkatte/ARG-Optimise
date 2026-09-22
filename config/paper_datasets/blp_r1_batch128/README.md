# R1 with a permanently frozen initial flow encoder

Run from the repository root with Python 3.11:

```bash
python train.py --config config/paper_datasets/blp_r1_batch128/config.yaml
```

This is a fresh CUDA run on `validation/datasets/paper_datasets/r1/rep0`, using
batch 128, SubTB lambda 0.9, and auxiliary TB weight 0.25. Its output directory is
`runs/paper_datasets_stable/blp_r1_lambda09_batch128_frozen_initial`; the W&B name
is `blp_r1_lambda09_batch128_frozen_initial`. All other R1 scientific, model,
learning-rate, replay, and evaluation settings are retained.

## What learns

`flow_encoder_mode: frozen_initial` deep-copies the complete initial
`InfiniteSitesEncoder` before any optimizer updates, without another random
initialization. The copy includes SNP/material encoders, projections, summary
token, and Transformer. It stays frozen and in evaluation mode forever.

The live policy encoder learns through policy scores. The existing flow head
learns from the frozen summary and current state features. Both branches train
from update one through SubTB + 0.25 * TB, with the existing global gradient
clipping. There is no head-only prefit, ramp, synchronization, or later unfreeze.
Source and intermediate flows use the fixed representation; terminal flows
remain exact rewards. The current infinite-sites prior/likelihood baseline,
flow-head architecture, centering/scaling, and initialization are unchanged.

Each branch has its own pooled-embedding cache for each rollout/microbatch.
Only raw, nonlearned observations are shared. Policy-only sampling does not run
the frozen encoder. Flow scoring costs an additional encoder forward pass but
keeps no backward activations for that branch.

Other configurations default to `flow_encoder_mode: shared`. In that mode,
`flow_encoder_grad_scale` remains a static multiplier on flow-to-encoder
gradients. It is inactive in frozen mode and is omitted from this R1 YAML.
`flow_warmup_steps: 0` explicitly disables the retired head-only prefit.

## Checkpoints and resume

Checkpoints retain schema 2 and flow-head version 6, record the encoder mode,
and save the complete frozen weights. Resuming restores those exact weights;
it never uses a new copy of the learned policy encoder. Missing mode metadata
means legacy shared behavior. Cross-mode loads/resumes and checkpoints from
the discarded gradient warm-up/ramp experiment are rejected. Use a fresh run
when changing architecture.

## Reduced CPU smoke test

This retains the R1 architecture and rates but uses three updates with two
trajectories per update. Use an empty output directory:

```bash
python train.py --config config/paper_datasets/blp_r1_batch128/config.yaml \
  --device cpu --cpu-threads 1 --epochs 3 --batch-size 2 \
  --checkpoint-every 1 --no-wandb --no-eval-async --no-terminal-eval \
  --eval-episodes 0 --output-path /tmp/blp_r1_frozen_initial_smoke
```

Focused regression checks:

```bash
python -m pytest validation/tests/test_frozen_flow_encoder.py \
  validation/tests/test_infinite_sites_neural.py \
  validation/tests/test_pooled_embeddings.py -q
```
