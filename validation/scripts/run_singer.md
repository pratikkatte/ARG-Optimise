Run from the repository root:

```bash
bash validation/scripts/run_singer.sh validation/datasets/human_2kb_super_easy
```

The script reads `metadata.json` either directly inside the input directory or
inside each immediate `rep*/` directory. It uses `files.vcf` (or discovers a unique
VCF beside the metadata), `simulation.sim_ancestry.parameters.population_size`,
`recombination_rate`, and `ploidy`, `simulation.sim_mutations.parameters.rate`,
and `summary.sequence_length_bp`. Both `.vcf` and `.vcf.gz` are supported.
Inputs must contain phased, complete biallelic SNP genotypes in local coordinates.
The script calls `singer_master` with `-vcf input_prefix`, `-m mutation_rate`,
and `-ratio recombination_rate / mutation_rate`. For example, mutation rate
`1e-7` and recombination rate `5e-8` produce `-m 1e-7 -ratio 0.5`.
The staged file is named `input.vcf`; the `-vcf` argument omits `.vcf`.
Haploid inputs receive `-ploidy 1`, verified in the installed 0.1.9 wrapper's
help; diploid inputs use the default.

Defaults are `-polar 0.99`, 10,000 burn-in iterations, 200 retained posterior
samples, thinning of 100, and seed 42. This requests 300 saved ARGs from SINGER,
then converts only indices 100–299. Burn-in is rounded up to a multiple of thinning.
These are generous sampling defaults, not a guarantee of convergence or accuracy.
The VCF REF must represent the ancestral allele for polarization to be meaningful;
the simulated dataset's metadata documents that convention.

```bash
# Preview the inputs, resolved parameters, and commands without writing outputs.
bash validation/scripts/run_singer.sh validation/datasets/human_2kb_super_easy --dry-run

# Increase sampling, saving this run separately.
bash validation/scripts/run_singer.sh validation/datasets/human_2kb_super_easy \
  --burnin 20000 --samples 500 --thin 100 --seed 123 \
  --output-dir validation/datasets/human_2kb_super_easy/output/singer_seed123
```

For the example dataset, results go into
`validation/datasets/human_2kb_super_easy/output/singer/rep0/`:

- `singer_100.trees` through `singer_299.trees`: retained posterior ARGs.
- `raw/`: all raw SINGER samples, including burn-in, and its MCMC statistics log.
- `singer.log`, `conversion.log`: inference and conversion console logs.
- `commands.txt`, `run.json`, `input_metadata.json`: commands, sampling settings,
  and a copy of the input metadata.
- `SUCCESS`: written only after exactly the requested number of posterior files
  exists and every expected index loads with the correct haplotype count and
  sequence length.

Validate a replicate by passing only its output folder:

```bash
python validation/scripts/point_accuracy_singer.py \
  validation/datasets/human_25kb_super_easy/output/singer/rep0
```

`--folder-dir` and `--input-dir` also accept this folder. The validator finds
`singer_*.trees` and ground truth beside the output or in the matching dataset
replicate (here, `human_25kb_super_easy/rep0`). It uses metadata to identify the
truth filename and population size, and reads the haplotype count from the truth.
Reports and plots are saved under `<output-folder>/point_accuracy/singer*`.
Explicit options such as `--truth-trees`, `--ne`, `--nspl`, `--sample-prefix`,
and `--output-prefix` override these defaults. `--burnin-samples` defaults to 0;
burn-in has already been excluded. Raw SINGER
outputs preserve its coordinate convention (`-start 0 -end sequence_length`);
VCF site positions are not shifted during conversion.

The script checks PATH for `singer_master`, then
`$HOME/singer/SINGER/releases/singer-0.1.9-beta-linux-x86_64/singer_master`,
then `$HOME/singer/SINGER/release/singer_master`.
Set `SINGER_BIN=/path/to/singer_master`,
`CONVERT_TO_TSKIT=/path/to/convert_to_tskit`, and `PYTHON_BIN=/path/to/python3`
to choose another installation. By default the converter is beside the wrapper;
both run with the selected Python, which needs numpy and tskit.
The installed wrapper splits its internal command on whitespace, so installation
and output paths must not contain whitespace.
Existing output directories are rejected; use a new `--output-dir` for reruns.

Recovery is handled entirely by `singer_master`; there is no `--max-retries`
option or custom debug loop. Wrapper and conversion failures stop the script.
Inspect the logs if no `SUCCESS` file is present. Prior numerical failures are
recorded in [the investigation report](../reports/singer_investigation_2026-09-08/report.md).

See the [upstream SINGER instructions](https://github.com/popgenmethods/SINGER)
for parameter definitions and convergence diagnostics. Examine traces across
samples and compare independent seeds before treating a long run as converged.
