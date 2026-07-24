# Local ARG refinement experiments

This directory now has two supported structural workflows. Both begin by:

1. represent the input as a synthetic/full ARG with explicit paired
   recombination nodes and unique event times using `synthetic_full_arg.py`;
2. build the compact backward-time event trace in `trace.py`;

The user-anchored workflow then resolves a supplied genomic interval and time
with `local_refinement.py`, propagates interval-aware dependencies, and returns
the mutable lineages plus an immutable exterior boundary contract.
`local_construction.py` initializes the existing `ARGState` representation and
applies the coalescent-with-recombination prior from `env.py` one event at a
time. `local_splice.py` installs a terminal proposal, removes source-only
synthetic routing nodes, remaps target mutations while preserving sample
genotypes, validates the fixed exterior, and exports a clean whole-chromosome
tree sequence.

The automatic structural-closure workflow instead:

1. proposes direct edge-closed regions from the normal tree sequence;
2. finds one exact older-suffix cut witness for each proposed region with
   `exact_closed_cones.py`;
3. uses a witness's genomic interval, cut step, and frontier as the fixed
   boundary for a later local reconstruction method.

## Files

- `synthetic_arg_exact_closed_cones_25kb.ipynb` is the canonical 25 kb
  discovery experiment. It scans every event cut and exposes the discovered
  regions, times, frontier lineages, nodes, edges, and validation diagnostics.
- `exact_closed_cones.py` contains the reusable exact all-cut scanner, the
  conservative normal-tree-sequence candidate generator, and the incremental
  existential-witness scanner. The latter adds every older event once and
  selects the valid cut with the fewest internal events and frontier nodes.
- `normal_ts_two_stage_exact_closed_cones.ipynb` compares the 25 kb and 1 Mb
  fixtures. It reports candidate counts and recall for topology adjacency tiers
  1/2/4/8/16/32, then confirms 100% recall using every proper normal-breakpoint
  interval as the exhaustive fallback.
- `normal_ts_edge_closed_regions.py` is the direct normal-tree first stage: at
  each normal time cut, it finds connected older-edge components whose genomic
  support is contiguous and has no overlapping outside older edge. Its default
  reverse-time incremental scanner adds each normal edge once; the original
  rebuild-at-every-cut scanner remains available only for parity checks.
- `normal_ts_direct_edge_closed_regions.ipynb` runs the direct edge-closure
  scan on the 25 kb and 1 Mb normal inputs, constructs their deterministic
  synthetic/full ARGs, and reports one exact local-refinement witness for each
  Stage-1 region that has one.
- `synthetic_full_arg.py` contains the core tree-sequence conversion used by
  this workflow: topology completion, unique event times, mutation-time
  sanitization, and provenance. It has no visualization or animation
  dependency.
- `trace.py` contains the compact event schedule and reversible active-frontier
  cursor used by the notebook.
- `local_refinement.py` resolves user-supplied interval/time cuts and builds a
  typed dependency and fixed-boundary context without mutating the input ARG.
- `local_construction.py` provides the `ARGState` initialization, locally
  filtered prior actions, time-gated fixed-ancestor attachments, reversible
  one-event state transitions, terminal detection, and seeded structural
  proposals.
- `local_splice.py` replaces only authorized edge intervals, collapses every
  temporary source routing node across the chromosome, preserves sampled local
  event nodes, validates the hybrid result, and writes a new `.trees` file.
- `user_anchored_local_arg_traceback.ipynb` is the runnable notebook for
  loading a normal `.trees` file, saving its synthetic/full ARG, building the
  trace, sampling a local history for a user interval and event time, and
  exporting a clean refined tree sequence.
- `benchmark_fast_trace.py` measures trace construction and cursor movement on
  larger synthetic/full ARG files.
- `tests/` covers synthetic-ARG conversion, unique event times, event replay,
  reversible frontier movement, and graph materialization.

The current scope is structural construction and export. The randomized
outputs are structural proposals, not posterior samples: biological prior
scoring, likelihood evaluation, GFlowNet training, and posterior correction
remain separate milestones.

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

## Build a user-anchored local refinement context

The user-anchored workflow accepts an ordinary inferred tree sequence, converts
it without overwriting the source file, builds one shared `FastARGTrace`, and
selects the target-dependent older history at a requested half-open genomic
interval and time:

```python
from arg.new_rl import LocalRefinementRequest, prepare_local_refinement

prepared = prepare_local_refinement(
    "inferred_and_dated.trees",
    LocalRefinementRequest(
        genomic_range=(10_000.0, 20_000.0),
        cut_time=25_000.0,
    ),
)

trace = prepared.trace
context = prepared.context
assert context.is_valid

cut_lineages = context.cut_active_lineages
promoted_dependencies = context.promoted_dependency_lineages
fixed_boundary = context.boundary_attachments
```

A numeric time resolves to the state immediately before the first trace event
whose time is greater than or equal to the requested time. Callers that already
hold a trace can use `trace_local_dependencies(trace, request)` directly, and
an exact zero-based trace event can be selected with `cut_event_index`.

Authorization is stored at edge-interval granularity. Target-bearing material
is mutable, while outside pieces on the same lineage or event are returned as
fixed tethers and attachment constraints. Every edge interval, node property,
event, mutation, and metadata record not explicitly authorized by the context
remains exterior. An unresolved mixed dependency returns an invalid context
with structured diagnostics rather than silently dropping a participant.

Inserted source routing nodes and their deconflicted times are deterministic
trace devices, not inferred biological recombination observations. They are
removed during clean export. In contrast, locally sampled paired recombination
nodes and local coalescence nodes are proposal events and remain in the final
tree sequence.

## Construct, sample, splice, and export a local history

Local construction uses the same `ARGState`, `ARGLineage`,
`MaterialSegments`, `CoalescenceChoice`, and `RecombinationChoice` types as the
global environment. Configure the prior explicitly; population size controls
the conversion from tree-sequence times to coalescent units, and `rho` or
`recombination_rate` controls the recombination rate:

```python
from arg.new_rl import (
    LocalSamplingConfig,
    SimpleARGEnvironment,
    export_refined_tree_sequence,
    initialize_local_arg_state,
    sample_local_trajectories,
    splice_local_proposal,
)

env = SimpleARGEnvironment(
    num_sequences=prepared.source_tree_sequence.num_samples,
    sequence_length=int(prepared.source_tree_sequence.sequence_length),
    num_blocks=int(prepared.source_tree_sequence.sequence_length),
    population_size=10_000,
    recombination_rate=2e-8,
    structural_only=True,
)
initial_state = initialize_local_arg_state(prepared, env)

batch = sample_local_trajectories(
    prepared,
    env,
    LocalSamplingConfig(
        sample_count=4,
        seed=23,
    ),
)
if not batch.proposals:
    raise RuntimeError(batch.diagnostics)

proposal = batch.proposals[0]
splice_result = splice_local_proposal(prepared, proposal)
assert splice_result.validation.is_valid

refined_tree_sequence = splice_result.refined_tree_sequence
export_refined_tree_sequence(
    splice_result,
    "inferred_and_dated.local_refined.trees",
)
```

For explicit step-by-step control, call
`enumerate_local_prior_actions(state, context, env)`, choose or sample a
`CoalescenceChoice` or `RecombinationChoice`, and pass it to
`apply_local_action`. `advance_local_state` performs the sampling and
transition together.

At each step, the filtered CWR prior uses the number of legal overlapping
coalescence pairs and `rho / 2` times active target material length.
Coalescence pairs are uniform. Recombining lineages are material-weighted, and
breakpoints are uniform over legal block boundaries (physical-gap-weighted in
VCF mode). Waiting time is sampled in the existing `TimeEnvFixedDelta` bins.
If a cross-window source ancestor is reached first, the local event survival
mass is recorded. Every fixed ancestor tied at that time is then connected to
the active lineages carrying material descended from its cut endpoints. Those
material pieces are replaced by the ancestor in the active frontier before
another local event is sampled. A fixed ancestor is never inserted as an
independent lineage with no descendants. If its required descendant material
is unavailable, the transition is rejected.

`undo_local_transition(state, context)` exactly reverses the most recent
coalescence, recombination, or fixed-ancestor attachment. Fixed attachments
have conditional forward and backward probability one; the probability of
surviving to their boundary time remains part of the forward CWR probability.

Construction stops immediately when every target block is carried by exactly
one active lineage. Adjacent blocks with the same root are reported as one
nonrecombining root interval. These roots remain parentless; no original upper
parent, terminal anchor, or attachment action is required.

By default, `LocalSamplingConfig` has no event, state, or restart cap:
construction continues until every target block has exactly one root.
`max_generated_events`, `max_searched_states`, and `max_restarts` remain
available only as explicit diagnostic watchdogs. If supplied and reached, the
sampler stops with a structured diagnostic; it does not repeatedly discard and
rebuild the same partial history. The stored `prior_log_probability` contains
the waiting-time, event-type, participant, breakpoint, and fixed-event survival
terms. These are structural prior proposals, not posterior samples.

Clean splicing first edits the temporary full ARG at authorized edge-subinterval
granularity. It then collapses only the source synthetic nodes identified by
conversion provenance, restores normal source tables, and leaves locally
generated event nodes intact. Once topology and times are fixed, target
mutation rows are replaced with a parsimonious `Tree.map_mutations` mapping
from the original sample genotypes. Outside mutation semantics remain
unchanged. The validator checks genotype preservation, event representation,
parentless local roots, edge time ordering, mutation ancestry, the absence of
non-sample target branches with zero sample descendants, and exact marginal
parent/coverage equality outside the requested half-open interval.
Export refuses to overwrite an existing file unless `overwrite=True` is
explicitly supplied. A context with no authorized older history produces a
terminal no-op proposal.

## Run the checks

From the workspace root (the directory containing `arg/`):

```bash
conda activate phylogfn_311
python -m pytest -q arg/new_rl/tests
```

The canonical notebook is also configured to use the `phylogfn_311` Jupyter
kernel.

The two-stage equivalence notebook deliberately keeps its exact scan independent
of the candidate catalog so it remains a recall oracle. The direct-region
notebook uses the candidate-directed incremental witness scan and benchmarks it
against that independent all-cut implementation.

The notebook starts from the ordinary inferred tree sequence:

```text
arg/validation/output/tsinfer/l25kb_dated.trees
```

It converts, saves, reloads, and traces this generated artifact:

```text
arg/validation/output/tsinfer/l25kb_dated_new_rl_synthetic_full_arg.trees
```
