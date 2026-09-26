#!/usr/bin/env bash
# Run independent chains and diagnostics with the requested conda environment.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-$(command -v python3 || true)}
[[ -x "$python_bin" ]] || {
    printf 'Python not found: %s; set PYTHON_BIN to your environment's python\n' "$python_bin" >&2
    exit 1
}
exec "$python_bin" -u "$script_dir/singer_convergence.py" "$@"
