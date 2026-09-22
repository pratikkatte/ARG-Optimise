# Paper simulation datasets

Each configuration generates one replicate of 25,000 bp with 10 haplotypes
(five diploid individuals), diploid effective population size 10,000, and
mutation rate `1e-8` per base per generation. The ratio is **mutation divided
by recombination**, `mu/r`. This gives three datasets in total, one per ratio.

| Configuration | mu/r | Recombination rate | Base simulation seed |
| --- | ---: | ---: | ---: |
| `r_1.yaml` | 1 | `1e-8` | 10001 |
| `r_2.yaml` | 2 | `5e-9` | 20001 |
| `r_4.yaml` | 4 | `2.5e-9` | 40001 |

Replicate `i` uses ancestry seed `base + 100000*i` and mutation seed
`base + 100000*i + 1`. These seeds are distinct across all three panels and
do not reuse the earlier development schedule starting at 42. Changing the
seeds changes the realized SNP counts; there is no SNP-count cap or filtering.

Increasing this ratio lowers recombination while keeping mutation fixed; it does
not increase the expected number of SNPs. With these constant-size diploid
coalescent settings, the expected segregating-site count across replicates is
`4 * Ne * mu * L * H_(n-1)`, approximately 28.29 for each configuration. Individual
replicates vary because both ancestry and mutation placement are random. The
saved r1/r2/r4 replicates contain 16/45/31 SNPs, respectively; their ordering is
not an error or an expected ordering by ratio. These are three different
observed datasets, not a controlled comparison holding the genealogy fixed.

## Generate data

Run from the repository root in an environment with the simulator dependencies:

```bash
for ratio in 1 2 4; do
  python3 validation/scripts/simulate_infinite_sites.py \
    --config "validation/config/paper_dataset/r_${ratio}.yaml" || break
done
```

Outputs go to `validation/datasets/paper_datasets/r{1,2,4}/rep0/`.
Each replicate includes VCF, exact SNP positions, metadata, mutated simplified
truth trees, and pairwise coalescence times. The additional `.full.trees` file
contains unmutated full ancestry. Rerunning a configuration overwrites its
matching simulation outputs.

## Train on a replicate

`config/config_paper_dataset_cosine_replay.yaml` uses the current human training
hyperparameters and reads population size and both scientific rates from the
selected replicate's metadata. Its default input is `r1/rep0`:

```bash
python3 train.py --config config/config_paper_dataset_cosine_replay.yaml
```

The trainer fits one observed dataset per run. Select another ratio's dataset with
CLI overrides, using a separate output directory:

```bash
python3 train.py --config config/config_paper_dataset_cosine_replay.yaml \
  --dataset-path validation/datasets/paper_datasets/r2/rep0 \
  --output-path runs/paper_datasets/r2/rep0_seed7
```

Use the same pattern for `r4/rep0`, for three training runs in total with one
training seed each. Scientific rate overrides
are omitted from the template so switching datasets also switches the rates.
The template uses CUDA, batch size 4, 10,000 updates, and training seed 7;
simulation seeds and training seeds serve separate purposes. Assess convergence
and posterior accuracy for each run.
