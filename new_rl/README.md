# Local ARG refinement experiments

This directory now has one supported workflow:

1. represent the input as a synthetic/full ARG with explicit paired
   recombination nodes and unique event times using `synthetic_full_arg.py`;
2. build the compact backward-time event trace in `trace.py`;
3. find exact closed ancestral cones with `exact_closed_cones.py` and
   `synthetic_arg_exact_closed_cones_25kb.ipynb`;
4. use a cone's genomic interval, cut step, and frontier as the fixed boundary
   for a later local reconstruction method.

## Files

- `synthetic_arg_exact_closed_cones_25kb.ipynb` is the canonical 25 kb
  discovery experiment. It scans every event cut and exposes the discovered
  regions, times, frontier lineages, nodes, edges, and validation diagnostics.
- `exact_closed_cones.py` contains the reusable exact all-cut scanner plus the
  conservative normal-tree-sequence candidate generator. The normal stage is
  evaluated against—not substituted for—the synthetic/full-ARG oracle.
- `normal_ts_two_stage_exact_closed_cones.ipynb` compares the 25 kb and 1 Mb
  fixtures. It reports candidate counts and recall for topology adjacency tiers
  1/2/4/8/16/32, then confirms 100% recall using every proper normal-breakpoint
  interval as the exhaustive fallback.
- `normal_ts_edge_closed_regions.py` is the direct normal-tree first stage: at
  each normal time cut, it finds connected older-edge components whose genomic
  support is contiguous and has no overlapping outside older edge. Its default
  reverse-time incremental scanner adds each normal edge once; the original
  rebuild-at-every-cut scanner remains available only for parity checks.
- `normal_ts_direct_edge_closed_regions.ipynb` runs that direct edge-closure
  scan on the 25 kb and 1 Mb normal inputs. It does not construct a
  synthetic/full ARG.
- `synthetic_full_arg.py` contains the core tree-sequence conversion used by
  this workflow: topology completion, unique event times, mutation-time
  sanitization, and provenance. It has no visualization or animation
  dependency.
- `trace.py` contains the compact event schedule and reversible active-frontier
  cursor used by the notebook.
- `benchmark_fast_trace.py` measures trace construction and cursor movement on
  larger synthetic/full ARG files.
- `tests/` covers synthetic-ARG conversion, unique event times, event replay,
  reversible frontier movement, and graph materialization.

The current scope is structural closure. Local GFlowNet reconstruction and
global coalescent-prior scoring are intentionally kept out of this directory
until their boundary conditions and state representation are implemented.

## Convert an input tree sequence

```python
from arg.new_rl import build_synthetic_full_arg

result = build_synthetic_full_arg(input_tree_sequence)
synthetic_arg = result.tree_sequence
conversion_summary = result.metadata
```

The default conversion creates balanced recombination topology and globally
unique event times. Pass `split_rule="left_to_right"` only when that explicit
topology is needed for comparison.

## Run the checks

From the workspace root (the directory containing `arg/`):

```bash
conda activate phylogfn_311
python -m pytest -q arg/new_rl/tests
```

The canonical notebook is also configured to use the `phylogfn_311` Jupyter
kernel.

The two-stage notebook uses the same kernel and deliberately keeps its exact
scan independent of the candidate catalog. This makes it an equivalence oracle
for candidate recall; candidate-directed verification is a later optimization.

The notebook starts from the ordinary inferred tree sequence:

```text
arg/validation/output/tsinfer/l25kb_dated.trees
```

It converts, saves, reloads, and traces this generated artifact:

```text
arg/validation/output/tsinfer/l25kb_dated_new_rl_synthetic_full_arg.trees
```
