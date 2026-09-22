# BLP r2 continuation from update 700

`train_r2_resume.sbatch.blp` follows the account, partition, Python environment
and repository path in `train.sbatch.blp`. It requests one GPU and runs the r2
batch64 / lambda0.9 / TB0.25 continuation for up to47 training hours. The config
explicitly lists checkpoint-compatible settings in the same flat YAML style as
`config/paper_datasets/sub_lambda_1/r1.yaml`.

Copy these files/directories to the same relative locations under
`/u/pratikkatte7/ARG-Optimise` on the destination cluster:

- `train_r2_resume.sbatch.blp`
- `config/paper_datasets/blp_r2_resume/`
- `runs/paper_datasets_stable/long48h_r2_from_f0700/start.pt`
- `runs/paper_datasets_stable/long48h_r2_from_f0700/source/` (the updated frozen runtime)
- `validation/datasets/paper_datasets/r2/rep0/` (the complete original dataset)

The starting checkpoint SHA-256 is
`9655fc1ca2378abe85c17ab11fd3b9ea6dda797ad01e7d44bb11f32170f41b64`.
The checkpoint and frozen runtime may not be included in an ordinary Git checkout;
copy them explicitly. The destination Python environment needs this runtime's
dependencies, including CUDA PyTorch and W&B. Authenticate W&B on that cluster
before submission if it is not already configured.

From the destination repository root:

```bash
mkdir -p logs
sbatch train_r2_resume.sbatch.blp
```

This starts at update701, restoring model, optimizer, scheduler, replay and RNG.
Checkpoints are written under:

```text
/u/pratikkatte7/ARG-Optimise/runs/paper_datasets_stable/blp_r2_lambda09_batch64_from0700/checkpoints/
```

For another allocation, use the latest saved state:

```bash
sbatch train_r2_resume.sbatch.blp --resume
```

Every50 updates, an immutable full checkpoint is queued for one separate GPU
evaluator. Training continues while evaluation runs on the same GPU. Results
and durable queue/status/logs are in the output directory, and delayed W&B
evaluation metrics use their checkpoint update as the plot axis. Unfinished
requests survive shutdown and are processed on resume. Sharing the GPU may
reduce throughput; concurrent memory/performance has not been tested on this
destination cluster.

The script prints the selected resume path and rejects accidental overwrites.
`ARG_REPO` and `ARG_PYTHON` environment overrides support a different install
location; the script overrides the three YAML filesystem paths accordingly.
No remote transfer or job submission is performed by preparing these files.

Local validation passed: shell syntax; mocked initial and subsequent launches;
missing-checkpoint, overwrite and invalid-argument guards; and an actual CPU
restore of checkpoint700 using the new full YAML and frozen runtime. The restore
accepted all immutable training settings and loaded optimizer/replay/RNG/scheduler
state without taking updates. Destination-cluster scheduling and GPU execution
have not been tested here.
