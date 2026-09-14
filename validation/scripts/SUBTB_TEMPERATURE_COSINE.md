# Fifth 500 bp variant: temperature annealing and cosine learning rates

This is a separate baseline experiment, with no prior-exploration or replay batches.
It starts from `runs/sim_500_subtb_shared_initialization/initial.pt` and retains the
original 128 held-out histories. The four earlier configurations are unchanged.
The reward and all physical/model parameters remain the same: full Hudson histories,
Gamma waiting policy, exact terminal reward, lambda 2, and neural source/intermediate
flows without a trainable logZ. This experiment changes two training controls together;
an improvement cannot be attributed to temperature alone or cosine decay alone.

## Schedules

Let `k` be the number of completed joint optimizer updates before the next batch:

* Discrete sampling temperature: `1 + 0.5 * max(0, 1 - k / 2000)`.
  This applies to event, pair/lineage, and breakpoint choices. Waiting-time temperature
  stays 1. At `k >= 2000`, sampling follows exactly the ordinary code path.
* LR multiplier: `0.1 + 0.9 * (1 + cos(pi * min(k, 10000) / 10000)) / 2`.
  Base policy/flow rates are `1e-4`/`1e-3`; both use this multiplier, with no warm-up
  or restarts. The floor after 10,000 completed updates is `1e-5`/`1e-4`.

Update 1 uses `k=0`. Update 2001 is the first fully untempered training update.
The scheduler advances once per joint optimizer update, independently of gradient
accumulation, evaluation, wall-clock time, and allocation boundaries. Checkpoints
retain schedule configuration/position, Adam moments, and all training RNG states.
Resume rejects different schedules and preserves the original horizons.

Only generation is tempered. SubTB scores ordinary `T=1` forward probabilities on
the generated histories and retains the exact reward. The sampled training distribution
is therefore a behavior policy, not the density used for posterior evaluation.
Tempered training histories must not receive ordinary-policy ESS or evidence estimates.
The legacy inference option `{"T": value}` retains its previous behavior; the new
training option `{"T": value, "time_T": 1}` explicitly overrides waiting-time sampling.

## Launch on another GPU node

Enter your allocated GPU node first; the launcher does not request another allocation.
Use the same shared workspace and `phylogfn_orig` environment:

```bash
cd /private/home/pkatte/aim3/iclr2017/temp/ARG-Optimise
conda activate phylogfn_orig

python validation/scripts/launch_subtb_temperature_cosine.py \
  --allocation-end auto \
  --output runs/sim_500_subtb_hudson_baseline_temperature_cosine \
  --dry-run

OMP_NUM_THREADS=2 python validation/scripts/launch_subtb_temperature_cosine.py \
  --allocation-end auto \
  --output runs/sim_500_subtb_hudson_baseline_temperature_cosine

tail -f runs/sim_500_subtb_hudson_baseline_temperature_cosine/training.log
```

The launcher validates CUDA, the fixed variant configuration, shared initialization,
and output path before creating any run artifacts. It launches a detached process
within the allocation and records its PID, command, node, GPU, deadline, and hashes
in `launch.json`. `auto` obtains Slurm's `EndTime`; an explicit ISO end time with
timezone offset may be supplied instead. It stops five minutes before that end time,
checking between completed updates and evaluation batches. A running batch/update
finishes before stopping. Checkpoints are saved atomically every five updates and
before evaluation; every evaluated checkpoint is retained. SIGTERM/SIGUSR1 request
the same graceful checkpoint stop. Existing output directories are rejected.

Resume a stopped run into a new directory in the next allocation:

```bash
OMP_NUM_THREADS=2 python validation/scripts/launch_subtb_temperature_cosine.py \
  --resume runs/sim_500_subtb_hudson_baseline_temperature_cosine/latest.pt \
  --allocation-end auto \
  --output "runs/sim_500_subtb_hudson_baseline_temperature_cosine_continuation_${SLURM_JOB_ID}"
```

For subsequent resumes, supply the latest checkpoint from the latest continuation.
The total target remains 10,000 updates, not 10,000 additional updates. The first
launcher does not automatically restart the four earlier variants.

## Compare with the original four variants

Routine evaluation is unchanged: 256 fresh `T=1` samples every 50 updates, the original
fixed held-out set, and the existing repeated terminal-evaluation cadence. Temperature
and actual learning rates are logged with training losses and sample counts.

Once the fifth variant has an immutable update-200 checkpoint, run:

```bash
OMP_NUM_THREADS=2 python validation/scripts/evaluate_subtb_density_fit.py extend \
  --base-evaluation validation/reports/subtb_trainability_2026-09-10/sim_500/density_fit \
  --run baseline_temperature_cosine=runs/sim_500_subtb_hudson_baseline_temperature_cosine \
  --steps 200 \
  --device cuda \
  --output validation/reports/subtb_temperature_cosine_2026-09-11/comparison_0200
```

This command obtains the current Slurm deadline automatically; alternatively provide
`--stop-at-unix` with a future safe-stop timestamp. Rerun the same command/output to
resume incomplete evaluation. It copies the original bank byte-for-byte, preserves
strata/provenance, checks replay/held-out overlap, and reuses compatible cached scores
by checkpoint and bank hashes. It never adds fifth-variant samples to the bank.
Every selected comparison update receives five independent repeats of 256 fresh
ordinary-policy samples using the original seed formula `100007 + step + 1000003*repeat`.

For later comparisons, use a new output, choose equal saved `--steps`, and supply
additional `--run LABEL=DIRECTORY` bindings for the existing variant labels
`corrected_baseline`, `exploration`, `exploration_replay`, and `baseline_replay`.
Resume origins are followed from recorded metadata; continuation directory names
are not guessed. Missing or conflicting checkpoints fail rather than mixing budgets.

The report includes raw/prior-relative calibration plots, slope-one residual errors,
repeated and globally pooled ESS/evidence estimates, TMRCA/RF/diversity, and SubTB/flow
diagnostics. `training_schedules.csv` and `.png`/`.pdf` show temperatures and actual LRs.
Bank strata describe constructed coverage, not posterior mass. Additional training
and better calibration are required before claiming posterior convergence.

## Validation

The tests use pytest (a development dependency; the launcher itself does not need it):

```bash
OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_policy_temperature_schedule.py \
  tests/test_learning_rate_schedule.py \
  tests/test_subtb_density_fit.py \
  tests/test_subtb_density_extension.py
```

If pytest is absent from the chosen environment, install that development dependency
before running the test command. Validation logs and the isolated 500 bp CUDA startup
and resume artifacts are in `validation/reports/subtb_temperature_cosine_2026-09-11`.
Those smoke checkpoints are not the fifth experiment and are not posterior comparisons.
