#!/bin/bash
# Run the SINGER baseline used in the paper on one dataset (r1, r2, or r4).
#
#   SINGER_DIR=/path/to/singer-0.1.9-beta-linux-x86_64 bash paper/scripts/run_singer.sh r1
#
# Input:   paper/datasets/<dataset>/rep0/<dataset>.vcf
# Outputs: ${OUTPUT_ROOT:-paper/outputs/SINGER}/<dataset>/{seed7,trees}
# The paper used SINGER 0.1.9-beta, seed 7, 1,100 samples thinned by 500,
# converted to tskit from sample 100 onward.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${SINGER_DIR:?Set SINGER_DIR to the SINGER release directory}"

dataset="${1:-}"
case "$dataset" in
  r1) ratio=1 ;;
  r2) ratio=0.5 ;;
  r4) ratio=0.25 ;;
  *) echo "Usage: $0 {r1|r2|r4}" >&2; exit 1 ;;
esac

out="${OUTPUT_ROOT:-paper/outputs/SINGER}/$dataset"
if [[ -e "$out" ]]; then
  echo "Output already exists; refusing to overwrite: $out (set OUTPUT_ROOT)" >&2
  exit 1
fi
mkdir -p "$out/seed7" "$out/trees"

python "$SINGER_DIR/singer_master" -Ne 10000 -ratio "$ratio" -start 0 -end 25000 \
  -polar 0.99 -n 1100 -thin 500 -m 1e-8 -seed 7 \
  -vcf "paper/datasets/$dataset/rep0/$dataset" \
  -output "$out/seed7/seed7"

python "$SINGER_DIR/convert_to_tskit" \
  -input "$out/seed7/seed7" -output "$out/trees/trees" \
  -start 100 -end 1100 -step 1
