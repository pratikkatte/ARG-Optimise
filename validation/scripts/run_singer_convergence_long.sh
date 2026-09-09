#!/usr/bin/env bash
# Longer rep0 chains, with comparison against the existing four-chain report.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
exec bash "$script_dir/run_singer_convergence.sh" \
    --run-all-seeds --burnin 50000 --thin 500 --samples 1000 \
    --output-root "$repo_root/validation/datasets/human_25kb_super_easy/output/singer_convergence_rep0_long" \
    --baseline-report "$repo_root/validation/datasets/human_25kb_super_easy/output/singer_convergence_rep0/report" \
    "$@"
