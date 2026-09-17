# Independent infinite-sites SNP scorer

From the repository root:

```sh
python3 validation/scripts/score_infinite_sites.py \
  --replicate-dir validation/datasets/sim_5k_mr20/rep0 \
  --trees validation/datasets/sim_5k_mr20/rep0/sim_5k_mr20.full.trees \
  --mutation-rate 2.5e-8
```

The command reads inputs and prints JSON. For this replicate it should report
16 haplotypes, 95 variants, a physical span of 5,000 bp, and log likelihood
approximately **-733.476814024274**. The simplified `.trees` candidate gives the
same score. The mutation rate must be supplied explicitly; it is not taken from
the simulation parameters automatically.

The observation loader reads only the VCF, exact-position map, and metadata.
It never reads ground-truth ancestry. Candidate ancestry is a separate input,
and its site and mutation records are ignored. The CLI needs NumPy and tskit;
it does not import PyTorch, msprime, or the training code.

## Python interfaces

```python
import tskit
from env.snp_data import load_snp_dataset
from env.infinite_sites import evaluate_infinite_sites

data = load_snp_dataset("validation/datasets/sim_5k_mr20/rep0")
candidate = tskit.load("path/to/candidate.trees")
result = evaluate_infinite_sites(candidate, data, mutation_rate=2.5e-8)
print(result.log_likelihood, result.incompatible_site_ids)
```

For an existing completed environment state, export in memory:

```python
candidate = env.save_to_tree_sequence(terminal_state)
result = evaluate_infinite_sites(candidate, data, mutation_rate=2.5e-8)
```

This requires the candidate to have the same samples and physical span as the
observations. The existing exporter converts internal `t/(2Ne)` times to
generations already. **Do not apply a second population-size conversion.**
This milestone does not make SNP data trainable through the existing JC69 path.

`data.genotypes` has shape `[haplotypes, variants]`, dtype `uint8`, and values
0/1. `data.positions` is a `float64` array of exact zero-based positions.
Arrays are read-only. `sequence_length` remains the physical span, even when
the number of variants is zero or many variants lie inside a single base.
Ancestral/derived nucleotide labels, original site IDs, and contig ID are retained.

Haplotype rows follow VCF sample-column order, with the left genotype allele
first: `spl0:0`, `spl0:1`, `spl1:0`, etc. By default they map to
`candidate.samples()` in order. If candidate node numbering differs, supply
`sample_nodes=[...]` (or CLI `--sample-nodes ...`) in **genotype-row order**.
The mapping must cover exactly all candidate sample nodes. Node IDs from the
simulation metadata are not automatically applied to other candidates.

## Model and validation

The scorer computes `-mu * A + sum(log(mu) + log(b_i))`, where `A` is genomic
branch exposure below local MRCAs in generation-bp and `b_i` is the sum of
compatible branch lengths for SNP i in generations. Unary subdivisions add
their durations; stems above local MRCAs contribute nothing. SNP-free intervals
still contribute to exposure. Intervals are half-open `[left, right)`.

The likelihood conditions on known ancestral states and omits fixed
nucleotide-label factors. It is a mutation-pattern density, not a normalized
probability for a rounded VCF position or a posterior reward including a prior.

Only schema-v1 bundles from the standalone simulator are supported: one contig,
continuous SNP positions, known ancestral REF, fully phased diploid calls,
biallelic segregating SNPs, and complete observations across the physical span.
VCF integer positions and contig length may have been shifted during export;
the position map and metadata provide the authoritative physical geometry.

Candidate ancestry must connect all samples across the entire span, use
`time_units="generations"`, and have all samples at time zero. Invalid inputs
raise `ValueError`; the CLI exits with a readable error. A valid but incompatible
candidate returns `-inf` and incompatible site IDs. JSON uses `null` plus
`zero_likelihood: true`, without nonstandard Infinity values. A zero rate with
observed SNPs also has zero likelihood, even if its topology is compatible.

## Tests

```sh
python3 -m pytest validation/tests/test_snp_data.py validation/tests/test_infinite_sites.py -q
```

Synthetic tests are self-contained. The additional `rep0` regressions run when
the local dataset is present and otherwise skip. msprime is used only for
independent test comparisons, never to calculate the production score.
