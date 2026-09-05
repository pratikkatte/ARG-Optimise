# ARG-Optimise

This repository trains and evaluates a trajectory-balance GFlowNet for
ancestral recombination graphs. The implementation is split by responsibility:
environment state and transitions live in `arg_environment/`, neural and
evolution models live in `model/`, GFlowNet mechanics live in `gflownet/`, and
training configuration and loops live in `training/`. Sequence/visualization
helpers live in `utils/`, and point-accuracy validation lives in
`validation/scripts/point_accuracy/`.
Exploratory notebooks are collected in `notebook/`.

Trajectory-balance convergence criteria, importance-ESS diagnostics, and the
strict posterior checkpoint workflow are documented in
[`docs/tb_convergence/`](docs/tb_convergence/README.md).

## Training

Copy or edit `config.yaml`, then run:

```bash
python train.py --config config.yaml
```

The YAML file contains data paths, runtime settings, optimizer values,
environment parameters, all model hyperparameters, and optional Weights &
Biases settings. Unknown fields and invalid values are rejected early. The
checked-in configuration logs a compact set of training, evaluation, optimizer,
and trajectory diagnostics to the `arg-optimise` W&B project every optimizer
update, with evaluation performed every five updates. Set `logging.wandb: false`
for fully local runs; the complete metric history is always saved locally.

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
