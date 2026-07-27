# ARG inference and validation

This folder provides one consistent comparison of GFN, tsinfer+tsdate, and
SINGER point accuracy. Run the commands below from the repository's `arg/`
directory in an activated Python 3.11+ environment.

## 1. Infer GFN ARG samples

```bash
python infer.py \
  --checkpoint runs/my-model/checkpoints/best.pt \
  --experiment l25kb-trial-01 \
  --num-args 100 \
  --batch-size 10
```

This writes:

```text
validation/output/l25kb-trial-01/gfn/
  arg_000001.trees
  ...
  manifest.json
```

`--output-dir` remains available for older workflows, but it cannot be combined
with `--experiment`.

## 2. Configure the baselines

Edit `validation/config.yaml`. Relative paths are resolved from the directory
containing that YAML. The included configuration already points at the 25 kb
truth and tsinfer+tsdate files.

SINGER inference is external to this repository. Before validating, either:

- put converted posterior tree sequences at
  `validation/output/singer/l25kb/singer_*.trees`; or
- change `methods.singer.input_dir` and `sample_prefix` to the actual files.

The default GFN input is
`validation/output/<experiment>/gfn/arg_*.trees`, so a separate GFN path is not
normally needed.

Each method must use exactly one input mode:

- `inferred_trees`: one dated/exported `.trees` file;
- `input_dir` plus `sample_prefix`: posterior `.trees` samples; or
- `bed_dir` plus `bed_prefix`: precomputed ARGsims BED files. SINGER BED input
  also requires `mcspl`.

GFN does not accept BED input, and tsinfer uses either one dated tree sequence
or BED input.

## 3. Validate all methods

```bash
python -m validation.run --experiment l25kb-trial-01
```

To use a different YAML:

```bash
python -m validation.run \
  --experiment l25kb-trial-01 \
  --config validation/my-validation.yaml
```

The command checks all truth and method inputs before evaluating anything. A
successful run writes:

```text
validation/output/l25kb-trial-01/results/
  gfn/
    metrics.tsv
    time_summary.tsv
    ...
  tsinfer/
  singer/
  summary.tsv
  weighted_rmse.png
  run_manifest.yaml
```

All methods use the same truth, effective population size, sample pairs, and
plot limits. Results are staged and published only when all three evaluations
succeed.

The command refuses to replace an existing `results/` folder. Rerun explicitly
with:

```bash
python -m validation.run --experiment l25kb-trial-01 --force
```

`--force` replaces only validation results; it never removes the GFN samples.
