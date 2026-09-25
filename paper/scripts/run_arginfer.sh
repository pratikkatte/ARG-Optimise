#!/bin/bash
# Run the ARGInfer baseline used in the paper on one dataset (r1, r2, or r4).
#
#   bash paper/scripts/run_arginfer.sh r1
#
# Inputs:  paper/datasets/<dataset>/arginfer_inputs
# Outputs: ${OUTPUT_ROOT:-paper/outputs/ARGInfer}/<dataset>
# On SLURM, submit as an array (1, 2, 4) and the dataset is taken from
# SLURM_ARRAY_TASK_ID. Requires `pip install arginfer` (see README).
set -euo pipefail
cd "$(dirname "$0")/../.."

dataset="${1:-}"
if [[ -z "$dataset" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  dataset="r${SLURM_ARRAY_TASK_ID}"
fi
case "$dataset" in
  r1) rate=1e-8 ;;
  r2) rate=5e-9 ;;
  r4) rate=2.5e-9 ;;
  *) echo "Usage: $0 {r1|r2|r4}" >&2; exit 1 ;;
esac

input="paper/datasets/$dataset/arginfer_inputs"
out="${OUTPUT_ROOT:-paper/outputs/ARGInfer}/$dataset"

for filename in haplotypes.txt ancestral.txt positions.txt; do
  [[ -s "$input/$filename" ]] || { echo "Missing or empty input: $input/$filename" >&2; exit 1; }
done

# ARGInfer deletes an existing output directory, so refuse to reuse one.
if [[ -e "$out" || -L "$out" ]]; then
  echo "Output already exists; refusing to overwrite: $out (set OUTPUT_ROOT)" >&2
  exit 1
fi
mkdir -p "$(dirname "$out")"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'Dataset=%s recombination_rate=%s output=%s\n' "$dataset" "$rate" "$out"

python -m arginfer infer \
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
  2>&1 | tee "$out.log"
