# ARG-Optimise

This repository trains and evaluates a trajectory-balance GFlowNet for
ancestral recombination graphs. The implementation is split by responsibility:
environment state and transitions live in `arg_environment/`, neural and
evolution models live in `model/`, GFlowNet mechanics live in `gflownet/`, and
training configuration and loops live in `training/`. Sequence/visualization
helpers live in `utils/`, and point-accuracy validation lives in
`validation/scripts/point_accuracy/`.
Exploratory notebooks are collected in `notebook/`.

## Training

Copy or edit `config.yaml`, then run:

```bash
python train.py --config config.yaml
```

The YAML file contains data paths, runtime settings, optimizer values,
environment parameters, all model hyperparameters, and optional Weights &
Biases settings. Unknown fields and invalid values are rejected early. The
checked-in configuration uses `logging.wandb: false` so local runs do not
contact external services by default.

Old checkpoints created with the hand-written transformer are intentionally
incompatible with the PyTorch-native transformer architecture.

## Tests

```bash
python -m pytest -q
```

If using the project Conda environment without pytest:

```bash
conda run -n phylogfn_orig python -m unittest discover -s tests -v
```
