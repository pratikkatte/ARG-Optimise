# Local ARG refinement experiments

This directory now has one supported workflow:

1. represent the input as a synthetic/full ARG with explicit paired
   recombination nodes and unique event times;
2. build the compact backward-time event trace in `trace.py`;
3. find exact closed ancestral cones with
   `synthetic_arg_exact_closed_cones_25kb.ipynb`;
4. use a cone's genomic interval, cut step, and frontier as the fixed boundary
   for a later local reconstruction method.

## Files

- `synthetic_arg_exact_closed_cones_25kb.ipynb` is the canonical 25 kb
  discovery experiment. It scans every event cut and exposes the discovered
  regions, times, frontier lineages, nodes, edges, and validation diagnostics.
- `trace.py` contains the compact event schedule and reversible active-frontier
  cursor used by the notebook.
- `benchmark_fast_trace.py` measures trace construction and cursor movement on
  larger synthetic/full ARG files.
- `tests/` covers synthetic-ARG conversion, unique event times, event replay,
  reversible frontier movement, and graph materialization.

The current scope is structural closure. Local GFlowNet reconstruction and
global coalescent-prior scoring are intentionally kept out of this directory
until their boundary conditions and state representation are implemented.

## Run the checks

From the workspace root (the directory containing `arg/`):

```bash
conda activate phylogfn_311
python -m pytest -q arg/new_rl/tests
```

The canonical notebook is also configured to use the `phylogfn_311` Jupyter
kernel.

The canonical input used by the notebook is:

```text
arg/validation/output/tsinfer/l25kb_dated_synthetic_full_arg.trees
```
