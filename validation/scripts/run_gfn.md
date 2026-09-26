# GFN inference and validation

From the repository root, run:

```bash
python validation/scripts/run_gfn.py runs/human_2kb_super_easy
```

This selects `checkpoints/best_eval_loss.pt`, generates 100 ARG samples with
`infer.py`, and runs `paper/validation/script/point_accuracy_gfn.py` on all samples
against the dataset's replicate 0 truth. The checkpoint's saved seed is used.
Device selection defaults to automatic CUDA/CPU detection; batch size is 1.

Outputs are saved in:

```text
validation/datasets/human_2kb_super_easy/output/gfn/rep0/best_eval_loss/
  arg_000001.trees ... arg_000100.trees
  manifest.json
  workflow.json
  inference.log
  validation/
    gfn_*.png
    gfn_*.tsv
    gfn_*.txt
    validation.log
```

The dataset defaults to `validation/datasets/<run directory name>`. Use
`--dataset-dir` and `--replicate` when the training run has a different name or
uses another replicate. The script checks that the checkpoint's embedded
sequences match the selected FASTA in the same order. It reads haplotype count
and effective population size from the replicate's `metadata.json`.

```bash
python validation/scripts/run_gfn.py runs/human_2kb_super_easy \
  --num-args 200 --batch-size 8 --device cuda --seed 42 \
  --output-dir validation/datasets/human_2kb_super_easy/output/gfn/rep0/eval_200
```

Use `--dry-run` to check inputs and print commands without generating results.
Use `--checkpoint best.pt` (or an absolute checkpoint path) to select another
checkpoint. Existing nonempty output directories are rejected to avoid mixing
samples from different runs. On failure, logs and any completed outputs remain
available for inspection. Both stages use the Python environment running this
script and require the existing inference and plotting dependencies.
