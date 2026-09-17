# Infinite-sites ARG environment (Phase 1)

The environment now uses polarized, fully observed SNP data and exact single-mutation messages. Neural policy encoding, GFlowNet training, checkpoint inference, and trained-model evaluation are deferred to Phase 2. Their entrypoints fail explicitly; FASTA/JC69 environment inputs and old checkpoints are not supported.

## Use the environment

```python
from env.snp_data import load_snp_dataset
from env.env import SimpleARGEnvironment

data = load_snp_dataset('validation/datasets/sim_5k_mr20/rep0')
env = SimpleARGEnvironment(
    snp_data=data, population_size=100000,
    mutation_rate=2.5e-8, recombination_rate=1.25e-9, seed=7,
)
state, trajectory = env.sample_compatible_trajectory(max_events=10000)
reference = env.evaluate_terminal(state)
assert abs(state.partial_log_likelihood - reference.log_likelihood) < 1e-9
```

`sample_compatible_trajectory` produces **diagnostic proposal samples, not posterior samples**. It may create many more recombinations than a useful learned policy. A dead end or event limit raises an error; there is no silent rejection/resampling loop.

For a command-line run that checks every completed ARG independently:

```bash
python3 validation/scripts/sample_infinite_sites.py \
  --replicate-dir validation/datasets/sim_5k_mr20/rep0 \
  --seed 7 --samples 1 --max-events 10000 \
  --output-dir /tmp/arg-infinite-sites-diagnostic
```

The output directory must be new. It receives ancestry-only `.trees` files and a JSON manifest containing likelihood, prior, proposal density, reward, exposure, and event count. The script reads observations and simulation parameters; it does not open ground-truth ancestry.

## State and numerical conventions

`sequence_length` and `num_blocks` describe physical geometry (5000 one-base blocks for rep0); `num_variants` is independently 95. SNP coordinates retain float64 continuous positions. Recombination actions use integer links, including links in trapped gaps. Sample genotype rows have the fixed haplotype order in `SNPData`.

Each `ARGLineage` has a coverage union (`material_segments`), annotated `descendants` intervals, carried `snp_indices`, and float64 `messages` with columns `(a,d,m)`. Adjacent intervals with different descendant bitsets remain separate. Message rows correspond only to carried SNPs, including ancestral observations. Zero rows are never used to identify missing material. A SNP-free lineage has shape `(0,3)` messages and retains all of its interval geometry.

Messages are defined at the lineage's `time`, not automatically at the current global event time:

- Leaves 0 and 1 initialize to `(1,0,0)` and `(0,1,0)`.
- An edge of duration `t` adds `t*d` to `m`.
- An overlapping merge gives `(a1*a2, d1*d2, m1*a2+a1*m2)` after propagation.
- A position carried by only one child is propagated unchanged through the merge.
- Recombination partitions coverage using `[left,right)`, routing a SNP at the breakpoint to the right parent.

Message arrays are immutable. Inactive-node message caches are released. Clone operations own their topology lists and state-level arrays, and can share immutable messages safely.

Times are in `t/(2Ne)`. `state.exposure` is closed-edge exposure in internal-time × bp; multiply by `2Ne` to compare with the independent evaluator's generation × bp exposure. `state.completed_site_lengths` contains internal-time compatible branch weights, with NaN marking unresolved sites. Zero is a genuine impossible branch weight, not the unresolved sentinel.

The intermediate potential is `-kappa * exposure + sum_completed(log(kappa) + log(branch_weight))`, where `kappa=2Ne*mu`. It starts at zero. At termination all SNPs are complete and this equals the independent infinite-sites likelihood. It is not a completion-marginalized intermediate likelihood.

SNP-free intervals contribute exposure. Branches with the full sample descendant set do not add exposure above the local MRCA, although this material remains present in the Hudson graph and rates until the global stopping condition. Termination requires complete ancestry everywhere, including intervals with no SNPs.

## Actions, prior, and diagnostic proposal

- `enumerate_actions(state)` returns the physical coalescence pairs and recombination lineage spans. Prior rates are always calculated from these actions.
- `enumerate_policy_actions(state)` removes incompatible coalescence pairs and zero-rate recombination choices. It does not remove lineages from the state.
- `apply_action(state, action)` validates before cloning and updating the state, computes the physical prior automatically, and returns a new state. An optional supplied prior must agree with that calculation.
- `IncompatibleActionError.site_ids` identifies the conflicting observed SNPs.
- `apply_actions` provides a scalar-equivalent CPU batch interface.

For a SNP with derived carrier set D, a merged local descendant set B is forbidden exactly when it contains some of D and some non-D samples but not all of D. The check uses local bitsets; a currently zero `m` is not itself a rejection rule. Input validation checks laminar carrier sets within each integer cell that recombination cannot separate, or across the entire genome when the recombination rate is zero.

`sample_compatible_step(state)` returns a `CompatibleStep(action, log_proposal, log_prior)`. Discrete action weights are the physical hazards (one per pair and `2Ne*r` per link), normalized over allowed actions. Waiting times retain the full physical rate Λ, including masked pairs. If H is the sum of allowed action hazards and h is the chosen hazard, the joint proposal density is `(h/H) * Λ*exp(-Λ*dt)`; the joint prior density is `h*exp(-Λ*dt)`. These are deliberately recorded separately.

`sample_prior_step` remains an unconditioned physical proposal and can propose a merge that `apply_action` rejects. It is not used by the compatible diagnostic sampler.

## Restoration and independent checks

`replay(actions)` starts from observations and replays an exact timed prefix. `restore_state(state)` rebuilds descendant/message caches and exposure from stored graph nodes and reconstructs chronological events and prior scores, without requiring cached actions or likelihood values. Original input states are not modified.

`save_to_tree_sequence` exports complete ancestry with node times converted to generations once. `evaluate_terminal` uses the existing independent evaluator against the observed genotypes, ignoring mutation records. The terminal reward remains `C + log_prior + log_likelihood`; genuine negative infinity is preserved, while NaNs and numerical overflow raise errors. No JC69 fallback or reward floor is present.

Focused verification:

```bash
python3 -m pytest validation/tests/test_snp_data.py \
  validation/tests/test_infinite_sites.py \
  validation/tests/test_infinite_sites_environment.py -q
```

Tests cover independent analytical scores, exhaustive small topologies, two-interval reachability, invariant material, local-MRCA stems, prior/mask separation, reconstruction, neural entrypoint guards, the 25-event rep0 full-ancestry replay, and fixed-seed generated candidates. The truth replay is a test fixture only and is not part of diagnostic sampling. These checks establish environment correctness, not posterior calibration or a training speedup.
