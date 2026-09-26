# Standalone infinite-sites simulator

Run from the repository root:

```sh
python3 paper/scripts/simulate_infinite_sites.py --config paper/datasets/r1_dataset.yaml
```

The canonical implementation lives in `paper/scripts/simulate_infinite_sites.py`.
The old compatibility entry point in `validation/scripts/` has been removed. Pass `--config` explicitly: the command retains its default of
`paper/config/config.yaml`, which is currently absent; it does not silently
select another dataset.

The simulator needs only msprime, its tskit dependency, and PyYAML.
No reference genome, reference download, or FASTA conversion is involved.

The first [ARGsims script](https://github.com/deboraycb/ARGsims/blob/main/scripts/1_msprime_sim.py)
uses the legacy msprime infinite-sites simulation. The
[second script](https://github.com/deboraycb/ARGsims/blob/main/scripts/2_msprimefinitesites.py)
replaces its mutations using JC69 and modern `sim_mutations`, whose mutation
coordinates default to a discrete genome. The new script combines ancestry and
mutation simulation in one place:

```python
msprime.sim_mutations(
    ancestry.simplify(),
    rate=mutation_rate,
    model=msprime.InfiniteSites(msprime.NUCLEOTIDES),
    discrete_genome=False,
    keep=False,
    random_seed=mutation_seed,
)
```

**The setting `discrete_genome=False` enforces unique mutation positions.**
Changing the mutation-model name alone does not do that in the modern API.
`InfiniteSites(NUCLEOTIDES)` is a compatibility class using the same nucleotide
label probabilities as JC69; this does not allow repeated hits when mutation
positions are continuous. See the [msprime mutation documentation](https://tskit.dev/msprime/docs/stable/mutations.html#discrete-or-continuous).

Edit `paper/datasets/r1_dataset.yaml`, `r2_dataset.yaml`, or `r4_dataset.yaml`
to set rates, sample size, length, replicates, seeds,
and output directory. Samples are haplotypes, with two per diploid individual.
Output paths are relative to the YAML file, and each replicate gets `rep<i>/`.
Rerunning an identical dataset name overwrites matching output files.

Outputs are `.trees`, `.vcf`, `.positions.tsv`, `metadata.json`, and pairwise
coalescence times in `tcoalmap/`. With `record_full_arg: true`, an additional
`.full.trees` file holds the unsimplified, unmutated ancestry.

The `.trees` file preserves the exact floating-point mutation positions and
A/C/G/T alleles. Like the linked scripts, VCF export uses tskit's `legacy`
transform: round coordinates, then advance ties or zero positions to distinct
positive integers. **Every SNP is kept**, and `.positions.tsv` maps each VCF
position back to the exact simulated position. Integer VCF coordinates are an
approximation: adjustment may cross an ancestry breakpoint or, for dense data,
extend beyond the original sequence length. Use the tree sequence or coordinate
map when exact positions and genomic distances are needed for ARG likelihoods.

The generated nucleotide REF is the known simulated ancestral allele; it is not
anchored to an external reference genome. This script changes simulation only;
the existing GFlowNet likelihood remains JC69.

```sh
python3 -m pytest paper/scripts/test_simulate_infinite_sites.py -q
```
