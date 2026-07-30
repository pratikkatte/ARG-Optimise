# User-anchored local ARG refinement

This directory contains one supported local-refinement workflow:

1. represent the input as a synthetic/full ARG with explicit paired
   recombination nodes and unique event times using `synthetic_full_arg.py`;
2. build the compact backward-time event trace in `trace.py`;
3. resolve a supplied genomic interval and time with `local_refinement.py`,
   propagate interval-aware dependencies, and return the mutable lineages plus
   an immutable exterior boundary contract;
4. initialize the existing `ARGState` representation with
   `local_construction.py`, prune phased VCF observations to the cut
   frontier, and apply the coalescent-with-recombination prior from `env.py`
   one event at a time;
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
  likelihood-aware one-event state transitions, terminal detection, reward,
  and seeded prior proposals.
- `vcf_likelihood.py` provides strict VCF/tree-sequence alignment and an
  iterative JC69 pruning engine used for cut partials, terminal likelihoods,
  and independent clean-export rescoring.
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

The current sampler draws structural actions from the CWR prior and evaluates
each completed proposal with phased-VCF JC69 likelihood and posterior reward.
The outputs remain prior proposals, not posterior samples: a learned local
forward policy, backward policy, GFlowNet training, and posterior correction
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
global environment. A likelihood-enabled environment takes the phased VCF
directly. Configure the prior and likelihood explicitly: population size
controls conversion from tree-sequence times to coalescent units, `rho` or
`recombination_rate` controls recombination, and `mutation_rate` controls JC69
branch lengths:

```python
from arg.new_rl import (
    LocalSamplingConfig,
    SimpleARGEnvironment,
    export_refined_tree_sequence,
    initialize_local_arg_state,
    sample_local_trajectories,
    splice_local_proposal,
)
from arg.utils import load_vcf_variants

variant_data = load_vcf_variants("observations.vcf")
env = SimpleARGEnvironment(
    variant_data=variant_data,
    population_size=10_000,
    mutation_rate=2e-8,
    recombination_rate=2e-8,
)
initial_state = initialize_local_arg_state(prepared, env)

batch = sample_local_trajectories(
    prepared,
    env,
    LocalSamplingConfig(
        sample_count=4,
        seed=23,
    ),
    initial_state=initial_state,
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

VCF haplotypes match `ts.samples()` positionally by default. Pass
`sample_node_to_haplotype` to `initialize_local_arg_state` when that ordering
differs. Coordinate alignment tests both `POS - 1` and `POS` against source
alleles and genotypes and accepts only one concordant convention; pass
`vcf_coordinate_offset=0` or `1` if both are genuinely concordant.

The local structural grid is the exact union of requested endpoints, source
and authorization boundaries, source tree breakpoints, and VCF gap
boundaries. `MaterialSegments` indexes this grid, while
`ARGLineage.variant_indices` contains only observed VCF rows. Invariant target
blocks therefore still participate in ancestry/root completion without
allocating fake likelihood rows.

For explicit step-by-step control, call
`enumerate_local_prior_actions(state, context, env)`, choose or sample a
`CoalescenceChoice` or `RecombinationChoice`, and pass it to
`apply_local_action`. `advance_local_state` performs the sampling and
transition together.

At each step, the filtered CWR prior uses the number of legal overlapping
coalescence pairs and `rho / 2` times active target material length.
Coalescence pairs are uniform. Recombining lineages are material-weighted, and
breakpoints are uniform over legal block boundaries (physical-gap-weighted in
VCF mode). Waiting time follows the exact constant-\(N_e\) exponential CWR law
in \(2N_e\)-scaled time. A generated event is represented by a continuous
conditional-CDF quantile, so there is no time-bin width or quantization. If a
cross-window source ancestor is reached first, its exact exponential survival
mass is recorded and the ancestor is revealed at its fixed boundary time.
Every fixed ancestor tied at that time is then connected to the active lineages
carrying material descended from its cut endpoints. Those material pieces are
replaced by the ancestor in the active frontier before another local event is
sampled. A fixed ancestor is never inserted as an independent lineage with no
descendants. If its required descendant material is unavailable, the
transition is rejected.

`undo_local_transition(state, context)` exactly reverses the most recent
coalescence, recombination, or fixed-ancestor attachment. Fixed attachments
have conditional forward and backward probability one; the probability of
surviving to their boundary time remains part of the forward CWR probability.

Construction stops immediately when every target block is carried by exactly
one active lineage. Adjacent blocks with the same root are reported as one
nonrecombining root interval. These roots remain parentless; no original upper
parent, terminal anchor, or attachment action is required.

Every likelihood-enabled transition preserves the VCF rows carried by each
lineage and accumulates normalization terms in
`state.accumulated_log_likelihood`. At terminal states, each target VCF row
must have exactly one root carrier. Root partials are integrated against the
JC69 stationary distribution and combined with the fixed likelihood outside
the requested interval:

```text
whole chromosome likelihood = fixed outside likelihood + rebuilt inside likelihood
log reward = reward_C + whole chromosome likelihood + local CWR log prior
```

`partial_log_reward` is the available forward-looking prefix during
construction and equals the exact terminal reward once all target blocks have
one root. Trajectory records keep likelihood and prior increments separate;
they are not labeled as learned `log P_F` or `log P_B`.

By default, `LocalSamplingConfig` has no event, state, or restart cap:
construction continues until every target block has exactly one root.
`max_generated_events`, `max_searched_states`, and `max_restarts` remain
available only as explicit diagnostic watchdogs. If supplied and reached, the
sampler stops with a structured diagnostic; it does not repeatedly discard and
rebuild the same partial history. The stored `prior_log_probability` contains
the waiting-time, event-type, participant, breakpoint, and fixed-event survival
terms. `compute_tree_sequence_vcf_log_likelihood` independently rescores a
normal tree sequence and serves as the clean-export parity oracle.

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
