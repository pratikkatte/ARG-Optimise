# User-anchored local ARG refinement

This directory contains one supported structural workflow:

1. represent the input as a synthetic/full ARG with explicit paired
   recombination nodes and unique event times using `synthetic_full_arg.py`;
2. build the compact backward-time event trace in `trace.py`;
3. resolve a supplied genomic interval and time with `local_refinement.py`,
   propagate interval-aware dependencies, and return the mutable lineages plus
   an immutable exterior boundary contract;
4. initialize the existing `ARGState` representation with
   `local_construction.py` and apply the coalescent-with-recombination prior
   from `env.py` one event at a time;
5. install a terminal proposal with `local_splice.py`, remove source-only
   synthetic routing nodes, remap target mutations while preserving sample
   genotypes, validate the fixed exterior, and export a clean whole-chromosome
   tree sequence.

## Files

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
- `tests/` covers synthetic conversion, event replay, user-cut dependency
  tracing, prior-driven forward/backward construction, clean splicing,
  mutation preservation, and exterior validation.

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

`user_anchored_local_arg_traceback.ipynb` exposes configurable source,
synthetic-output, and clean refined-output paths. It never overwrites the
source tree sequence.
