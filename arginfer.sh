#!/bin/bash
#SBATCH --job-name=arginfer
#SBATCH --partition=long
#SBATCH --account=standard
#SBATCH --qos=normal
#SBATCH --array=1,2,4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --export=ALL
#SBATCH --output=logs/arginfer-%A_%a.out
#SBATCH --error=logs/arginfer-%A_%a.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the ARG-Optimise repository root}"

# Initialize Conda for this non-interactive batch shell.
source /private/home/pkatte/anaconda3/etc/profile.d/conda.sh
conda activate phylogfn_orig
set -u

command -v arginfer >/dev/null || {
  echo "arginfer is not on PATH in the phylogfn_orig environment." >&2
  exit 1
}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

case "$SLURM_ARRAY_TASK_ID" in
  1) dataset=r1; rate=1e-8 ;;
  2) dataset=r2; rate=5e-9 ;;
  4) dataset=r4; rate=2.5e-9 ;;
  *) echo "Unsupported dataset index: $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

input="validation/datasets/paper_datasets/arginfer_inputs/$dataset"
out="validation/datasets/paper_datasets/output/arginfer/$dataset/job_${SLURM_ARRAY_JOB_ID}"

for filename in haplotypes.txt ancestral.txt positions.txt; do
  [[ -s "$input/$filename" ]] || {
    echo "Missing or empty input: $input/$filename" >&2
    exit 1
  }
done

# ARGinfer deletes an existing output directory, so refuse to reuse one.
if [[ -e "$out" || -L "$out" || -e "${out}.log" || -L "${out}.log" ]]; then
  echo "Output already exists; refusing to overwrite: $out" >&2
  exit 1
fi
mkdir -p "$(dirname "$out")"
printf 'Dataset=%s recombination_rate=%s output=%s\n' "$dataset" "$rate" "$out"

python3 -m arginfer infer \
  --input_path "$input" \
  --haplotype_name haplotypes.txt \
  --ancAllele_name ancestral.txt \
  --snpPos_name positions.txt \
  --sample_size 10 \
  --seq_length 25000 \
  --Ne 10000 \
  --mutation_rate 1e-8 \
  --recombination_rate "$rate" \
  --iteration 2000001 \
  --burn 200000 \
  --thin 1000 \
  --outpath "$out" \
  2>&1 | tee "${out}.log"
