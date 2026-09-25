# Two-locus posterior POC

This folder contains the two-haplotype, two-SNP correctness experiment. It
depends on the shared model, environment, and training code in the repository;
it is not a standalone package.

`config.yaml` was recovered from the parent of commit `6dd6839`. Its scientific
and training settings are unchanged. Outputs retain the configured location
`paper/outputs/poc/`, relative to the repository root.

Run commands from the repository root, using an environment with the repository
dependencies plus SciPy and pytest installed:

```bash
python -m pytest paper/poc/test_poc.py -q
python paper/poc/run_poc.py --reference-only --device cpu
python paper/poc/run_poc.py
```

The full experiment defaults to CUDA, three training seeds, 3,000 updates per
seed, and 20,000 final samples per distribution. Use `--device cpu` to run on
CPU. Use `--resume` to resume saved runs.

After training, audit each seed and produce the supplement summary:

```bash
python paper/poc/audit_poc.py --run paper/outputs/poc/runs/main_gamma_mixture_seed7
python paper/poc/audit_poc.py --run paper/outputs/poc/runs/main_gamma_mixture_seed17
python paper/poc/audit_poc.py --run paper/outputs/poc/runs/main_gamma_mixture_seed27
python paper/poc/summarize_poc.py
```

## Files

| File | Purpose |
| --- | --- |
| `poc_dataset.py` | Fixed observations and a compatible reference candidate ARG |
| `poc_reference.py` | Independent evidence calculation and posterior rejection sampler |
| `poc_metrics.py` | Full-history distribution comparisons and marginal diagnostics |
| `run_poc.py` | Dataset preparation, reference generation, training, and evaluation |
| `config.yaml` | Dataset, model, training, and evaluation settings |
| `test_poc.py` | Reference, density, replay, checkpoint, and metric checks |
| `audit_poc.py` | Saved-artifact and checkpoint-density checks |
| `derive_poc_reference.py` | Independent analytic moments for the default scientific parameters |
| `summarize_poc.py` | Table and figure from audited runs |
| `queue_poc_comparison.py` | Optional control supervisor; requires an existing detached-run process record and psutil |

The summary script retains historical narrative about the original execution
hardware and interrupted runs; review that prose against any new execution
before using its report in a manuscript.

The shared scorer also supports this configuration:

```bash
python validation/scripts/score_infinite_sites.py --generate-poc --config paper/poc/config.yaml
```
