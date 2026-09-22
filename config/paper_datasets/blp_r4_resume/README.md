# r4 batch-64 continuation experiment

The config starts from a prepared fork of r4 checkpoint500. Training changes
only batch size32 to64; lambda2.0, auxiliary TB weight0, learning rates, model,
scientific rates and original observations remain as in the r4 checkpoint.
Optimizer moments, scheduler, replay and random states are retained. The next
update is501. This is a new batch-size experiment; its future trajectory differs
from the batch32 continuation.

Copy these to the same relative locations under `/u/pratikkatte7/ARG-Optimise`:

- `config/paper_datasets/blp_r4_resume/`
- `runs/paper_datasets_stable/blp_r4_batch64_from0500/` (prepared `start.pt`,
  `fork.json`, runtime `source/`, and verification manifest)
- `validation/datasets/paper_datasets/r4/rep0/`

The source checkpoint is preserved at
`runs/paper_datasets_stable/long48h_r4_from_b0500/start.pt`. Do not substitute it
for the new fork: its saved batch size is32 and the trainer rejects changing it
through an ordinary resume override. `fork.json` records the batch change and
source checkpoint hash.

Inside an allocated GPU job, from the destination repository root:

```bash
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
srun /u/pratikkatte7/.conda/envs/args/bin/python -u \
  runs/paper_datasets_stable/blp_r4_batch64_from0500/source/train.py \
  --config config/paper_datasets/blp_r4_resume/config.yaml
```

The r2 launch script hardcodes its own config/checkpoint/dataset/output paths;
use r4 paths when adapting that script. This YAML contains all three destination
paths explicitly. New outputs and `checkpoints/latest.pt` are under:

```text
/u/pratikkatte7/ARG-Optimise/runs/paper_datasets_stable/blp_r4_batch64_from0500/run/
```

For a later allocation, append
`--resume-checkpoint /u/pratikkatte7/ARG-Optimise/runs/paper_datasets_stable/blp_r4_batch64_from0500/run/checkpoints/latest.pt`
to the command. Preserve the existing output directory to recover pending
evaluation requests.

The runtime includes asynchronous GPU evaluation every50 updates, using512
fresh histories and3 repeats every1000 updates, matching the r2 evaluation
schedule. Optional tree-truth summaries are disabled; ESS, density slope and
independent likelihood validation remain enabled. The output protocol differs
from the old r4 pilot; batch size is the only training hyperparameter changed.
The frozen runtime differs from the old r4 source only in the three files that
implement asynchronous evaluation and resume orchestration. No training or remote job is launched by
preparing this config and checkpoint.

Validation passed: exact model/optimizer/trainer/replay/RNG/scheduler and
scientific-metadata equality against the original checkpoint, unchanged parent
file hash, and an actual CPU restore at update500 with the full new config.
The only changed resolved training hyperparameter in the fork is batch size.
No training updates were performed; destination GPU execution remains untested.
