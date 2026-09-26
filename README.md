# ARGFlow

Amortized posterior sampling of ancestral recombination graphs (ARGs) with
generative flow networks.

## Install

Python 3.11, Linux.

```bash
pip install -r requirements.txt
```

For CPU-only machines, install PyTorch first (see the note in `requirements.txt`).
SINGER 0.1.9-beta is a separate binary download; ARGInfer is installed from PyPI.

## Layout

| Path | Contents |
| --- | --- |
| `env/`, `policy/`, `gfn/`, `training/`, `eval/` | Environment, policy, GFlowNet objective, training, metrics |
| `train.py`, `infer.py` | Training and sampling entry points |
| `paper/datasets/` | Simulated datasets r1, r2, r4 and baseline inputs |
| `paper/config/` | Training configurations for r1, r2, r4 |
| `paper/ablation/` | SubTB lambda ablation configurations |
| `paper/scripts/` | Baseline runners, Table 2 and Figures 2–3 |
| `paper/poc/` | Correctness experiment ([README](paper/poc/README.md)) |
| `paper/validation/` | Appendix E.1 |
| `tests/` | Core correctness tests |

Generated artifacts go to `paper/checkpoints/` and `paper/outputs/`.

## Reproduce the paper results

The published checkpoints and posterior samples are distributed separately
from the code. Place them at:

```
paper/checkpoints/{r1,r2,r4}/checkpoint.pt
paper/outputs/argflow/{r1,r2,r4}/<checkpoint>/     # 1,800 ARGFlow draws + manifest.json
paper/outputs/SINGER/{r1,r2,r4}/trees/            # SINGER draws
paper/outputs/ARGInfer/{r1,r2,r4}/                # ARGInfer draws
```

Evaluation selects the first 1,000 draws per method after burn-in, preserving
the supplied archives. Then, from the repository root (CPU only):

```bash
python paper/scripts/paper_datasets/evaluate.py   # Table 2  -> paper/outputs/evaluation/
python paper/scripts/paper_datasets/figure_2.py   # Figure 2 -> paper/outputs/figure_2/
python paper/scripts/paper_datasets/figure_3.py   # Figure 3 -> paper/outputs/figure_3/
```

Settings and metric definitions: [paper/scripts/paper_datasets/README.md](paper/scripts/paper_datasets/README.md).

## Reproduce

```bash
# 1. Datasets (already included in paper/datasets/)
python paper/scripts/simulate_infinite_sites.py --config paper/datasets/r1_dataset.yaml

# 2. Train (GPU); checkpoints go to paper/checkpoints/<dataset>/
python train.py --config paper/config/r1.yaml

# 3. Sample 1,000 ARGs from a checkpoint -> paper/outputs/argflow/<dataset>/<checkpoint>/
python validation/scripts/sample_final_checkpoints.py \
  --checkpoint paper/checkpoints/r1/checkpoints/best_.pt --dataset r1 \
  --num-args 1000 --seed 20260925

# 4. Baselines (prepared inputs are included under each dataset)
bash paper/scripts/run_arginfer.sh r1
SINGER_DIR=/path/to/singer-0.1.9-beta-linux-x86_64 bash paper/scripts/run_singer.sh r1
```

ARGInfer reads `paper/datasets/<dataset>/arginfer_inputs/` directly. To prepare
these inputs for a newly generated dataset, use:

```bash
python validation/scripts/prepare_arginfer_inputs.py --datasets paper/datasets
```

The converter refuses to overwrite existing input directories. Preserve or
remove the old prepared inputs before regenerating them. Evaluation checks
that they match the dataset VCF.

Repeat for `r2` and `r4`. Training is stochastic, so a new run gives a new
model, not the published checkpoints.

## Tests

```bash
python -m pytest tests paper -q
```
