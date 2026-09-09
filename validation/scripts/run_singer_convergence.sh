#!/usr/bin/env bash
# Run independent chains and diagnostics with the requested conda environment.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-$HOME/anaconda3/envs/phylogfn_orig/bin/python}
[[ -x "$python_bin" ]] || {
    printf 'Python not found: %s; set PYTHON_BIN to phylogfn_orig/bin/python\n' "$python_bin" >&2
    exit 1
}
exec "$python_bin" -u "$script_dir/singer_convergence.py" "$@"
