# Infinite-sites simulation

Run from the repository root:

```sh
python3 validation/scripts/merge_msprime.py --config validation/config/simulate_config_2k.yaml
```

The presets use dataset names ending in `_infinite_sites`, so they write to new
directories alongside the existing JC69 datasets. The script's default config
is `validation/config/simulate_config.yaml`. Paths in a config are relative to
that config file. Rerunning an identical dataset name overwrites matching files.

The simulator records full diploid Hudson ancestry, simplifies it to marginal
trees, and calls `msprime.sim_mutations` with `BinaryMutationModel`,
`discrete_genome=False`, and `keep=False`. Each mutation has a distinct continuous
position and a single ancestral-to-derived change. This is the
[msprime infinite-sites construction](https://tskit.dev/msprime/docs/stable/mutations.html#discrete-or-continuous).

For replicate `i`, the ancestry seed is `seed + 100000*i`, the mutation seed is
one greater, and the nucleotide-label seed is two greater. The reference window
must contain only A/C/G/T. Its bases define the ancestral nucleotide sequence;
each mutated site receives a uniformly chosen different nucleotide. This
relabeling preserves the original mutation's descendant carriers and time.
It does not introduce a JC69 substitution process.

Each successful replicate contains:

| File | Meaning |
| --- | --- |
| `<name>.full.trees` | Complete ancestry with recombination nodes, before mutations and simplification |
| `<name>.infinite_sites.trees` | Exact mutation realization: simplified ancestry, original fractional positions, binary 0/1 alleles |
| `<name>.trees` | The same marginal ancestry with integer-position A/C/G/T sites for comparison with VCF/FASTA |
| `<name>.vcf` | Phased diploid genotypes, one-based positions; REF is the known ancestral/reference base |
| `<name>.fa` | One complete haplotype per sample node, in tree-sequence sample order |
| `tcoalmap/*.tc` | Pairwise coalescence times over zero-based, half-open genomic intervals |
| `metadata.json` | Model, seeds/provenance, sample correspondence, original site coordinates, export counts, and file descriptions |

## Continuous positions and integer exports

FASTA and the downstream VCF workflows require integer base positions. Export
therefore floors each continuous position. Ancestry breakpoints are already
integers, so this does not move a mutation into a different marginal tree.
The original fractional positions are retained in `.infinite_sites.trees` and
reported for each retained site in metadata.

Two distinct infinite-sites mutations can fall within the same integer base.
They cannot both be represented as distinct loci in an ordinary fixed-length
FASTA. The default `export_collision_policy: error` stops the integer export
if this happens, after saving the full ancestry and exact continuous mutation
data. It does not resimulate, shift mutations between trees, or silently remove
events. An error leaves that replicate's VCF/FASTA/metadata export incomplete;
do not use stale exports from an earlier run with the same dataset name.

An explicit `export_collision_policy: drop` retains the first continuous site
per integer base for the integer exports. The script prints the number removed
and records it in metadata. The original continuous mutation data remain intact.
**This is a lossy approximation:** dropping sites changes the mutation-count
distribution. Use `.infinite_sites.trees` for analyses requiring the exact
infinite-sites realization. A zero-drop export retains all mutation events,
but its positions are still discretized.

## Checks and scope

```sh
python3 -m pytest validation/tests/test_merge_msprime.py -q
```

Checks cover single-mutation sites, nucleotide/genotype consistency, first and
last base coordinates, collisions, zero mutations, full ancestry preservation,
metadata, coalescence maps, and rejection of invalid input.

This change updates simulation and exports. The GFlowNet training likelihood
still uses JC69 and needs its own infinite-sites implementation before a run
can be described as infinite-sites posterior inference.
