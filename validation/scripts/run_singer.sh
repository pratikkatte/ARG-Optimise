#!/usr/bin/env bash
# Run SINGER on a metadata-backed dataset or its rep*/ directories.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_singer.sh DATASET_DIR [options]

Find metadata.json and its VCF in DATASET_DIR or immediate rep*/ directories.
Read Ne, mutation/recombination rates, ploidy and sequence length from metadata.
Save results in DATASET_DIR/output/singer/ (one subdirectory per replicate).

Options:
  --burnin N       Burn-in in MCMC iterations; rounded up to --thin (100)
  --samples N      Number of posterior ARGs to retain after burn-in (200)
  --thin N         MCMC iterations between saved ARGs (100)
  --seed N         SINGER random seed (42)
  --output-dir P   Override the output root
  --dry-run       Validate inputs and print commands without running inference
  -h, --help      Show this help

Polarization is fixed at 0.99. Existing run directories are never overwritten.
Dependencies: SINGER and its convert_to_tskit, Python 3 with numpy and tskit.
Set SINGER_BIN, CONVERT_TO_TSKIT, or PYTHON_BIN to override tool discovery.
SINGER_BIN must name singer_master; its wrapper handles recovery.

Example:
  bash validation/scripts/run_singer.sh validation/datasets/human_2kb_super_easy
EOF
}
die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
burnin=10000 samples=200 thin=100 seed=42 dry_run=0 dataset='' output_root=''
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=1; shift ;;
        --burnin|--samples|--thin|--seed|--output-dir)
            (($# >= 2)) || die "$1 requires a value"
            case "$1" in
                --burnin) burnin=$2 ;; --samples) samples=$2 ;;
                --thin) thin=$2 ;; --seed) seed=$2 ;; --output-dir) output_root=$2 ;;
            esac
            shift 2 ;;
        -*) die "Unknown option: $1" ;;
        *) [[ -z "$dataset" ]] || die "Provide only one dataset directory"
           dataset=$1; shift ;;
    esac
done
[[ -n "$dataset" ]] || { usage >&2; exit 1; }
[[ -d "$dataset" ]] || die "Dataset directory does not exist: $dataset"
for name in burnin samples thin seed; do
    value=${!name}
    [[ "$value" =~ ^[0-9]{1,9}$ ]] || die "$name must be an integer from 0 to 999999999"
    printf -v "$name" '%d' "$((10#$value))"
done
((samples > 0 && thin > 0)) || die "samples and thin must be positive"
burnin_samples=$(((burnin + thin - 1) / thin))
total_samples=$((burnin_samples + samples))
((total_samples <= 2147483647)) || die "Too many samples for SINGER"
python_bin=${PYTHON_BIN:-python3}
command -v "$python_bin" >/dev/null || die "Python not found: $python_bin"

# Prefer PATH, then the installed 0.1.9 release (compatible with tskit 1.x).
singer_master=${SINGER_BIN:-}
if [[ -z "$singer_master" ]]; then
    singer_master=$(command -v singer_master || true)
    if [[ -z "$singer_master" ]]; then
        for candidate in \
            "$HOME/singer/SINGER/releases/singer-0.1.9-beta-linux-x86_64/singer_master" \
            "$HOME/singer/SINGER/release/singer_master"; do
            if [[ -x "$candidate" ]]; then singer_master=$candidate; break; fi
        done
    fi
fi
[[ -n "$singer_master" ]] || die "singer_master not found; set SINGER_BIN=/path/to/singer_master"
singer_master=$(command -v "$singer_master") || die "singer_master executable not found"
singer_master=$("$python_bin" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$singer_master")
[[ -x "$singer_master" && "${singer_master##*/}" == singer_master ]] || die "SINGER_BIN must name singer_master"
[[ ! "$singer_master" =~ [[:space:]] ]] || die "singer_master requires an installation path without whitespace"
converter=${CONVERT_TO_TSKIT:-${singer_master%/*}/convert_to_tskit}
converter=$(command -v "$converter") || die "convert_to_tskit not found; set CONVERT_TO_TSKIT"
if (( ! dry_run )); then
    "$python_bin" -c 'import numpy, tskit' || die "Install numpy and tskit in the selected Python environment"
fi

# Validate every input before starting any costly runs. Tab-separated fields are
# data only: never source/eval metadata or interpolate it into shell code.
dataset_info=$("$python_bin" - "$dataset" "$output_root" <<'PY'
import gzip
import json
import math
from pathlib import Path
import sys

def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a positive scalar; rate maps/demographies need a separate workflow")
    return value

try:
    root = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve() if sys.argv[2] else root / "output" / "singer"
    direct = root / "metadata.json"
    metadata_files = [direct] if direct.is_file() else sorted(root.glob("rep*/metadata.json"))
    if not metadata_files:
        raise ValueError(f"No metadata.json in {root} or its rep*/ directories")
    for metadata in metadata_files:
        data = json.loads(metadata.read_text())
        ancestry = data["simulation"]["sim_ancestry"]["parameters"]
        mutation = data["simulation"]["sim_mutations"]["parameters"]
        ne = positive(ancestry.get("population_size"), "population_size")
        mu = positive(mutation.get("rate"), "mutation rate")
        recombination_rate = positive(ancestry.get("recombination_rate"), "recombination_rate")
        ratio = positive(recombination_rate / mu, "recombination/mutation ratio")
        length = positive(data["summary"]["sequence_length_bp"], "sequence_length_bp")
        if int(length) != length:
            raise ValueError("sequence_length_bp must be an integer")
        ploidy = ancestry.get("ploidy", 2)
        if ploidy not in (1, 2):
            raise ValueError("SINGER requires haploid or diploid VCF genotypes")
        vcf_name = data.get("files", {}).get("vcf")
        if vcf_name:
            vcf = (metadata.parent / vcf_name).resolve()
        else:
            candidates = sorted([*metadata.parent.glob("*.vcf"), *metadata.parent.glob("*.vcf.gz")])
            if len(candidates) != 1:
                raise ValueError(f"Expected exactly one VCF beside {metadata}; found {len(candidates)}")
            vcf = candidates[0].resolve()
        if not vcf.is_file() or not str(vcf).endswith((".vcf", ".vcf.gz")):
            raise ValueError(f"Missing or unsupported VCF: {vcf}")
        # SINGER reads GT first and assumes complete, phased, biallelic data.
        # Fail explicitly instead of silently treating unsupported calls as REF.
        opener = gzip.open if str(vcf).endswith(".gz") else open
        sites, previous, chrom, count = 0, 0, None, None
        with opener(vcf, "rt") as handle:
            for line in handle:
                if line.startswith("#CHROM"):
                    count = len(line.rstrip().split("\t")) - 9
                    continue
                if line.startswith("#"):
                    continue
                fields = line.rstrip().split("\t")
                if count is None or count < 1 or len(fields) != count + 9:
                    raise ValueError(f"Invalid VCF header/record in {vcf}")
                pos = int(fields[1])
                if not previous < pos <= length or (chrom is not None and fields[0] != chrom):
                    raise ValueError("VCF must contain one contig, with unique sorted local positions in [1, sequence_length]")
                previous, chrom = pos, fields[0]
                if len(fields[3]) != 1 or len(fields[4]) != 1 or fields[4] not in "ACGT" or fields[3] not in "ACGT":
                    raise ValueError("VCF must contain biallelic SNPs")
                if fields[8].split(":")[0] != "GT":
                    raise ValueError("SINGER requires GT first in FORMAT")
                alleles = []
                for call in fields[9:]:
                    gt = call.split(":")[0].split("|")
                    if len(gt) != ploidy or any(a not in ("0", "1") for a in gt):
                        raise ValueError("VCF requires complete phased 0/1 genotypes matching metadata ploidy")
                    alleles.extend(gt)
                if len(set(alleles)) != 2:
                    raise ValueError("VCF must contain segregating sites only")
                sites += 1
        if sites < 2:
            raise ValueError("Need at least two segregating SNPs for this SINGER/converter workflow")
        expected = data["summary"].get("num_haplotypes")
        if expected is not None and count * ploidy != expected:
            raise ValueError("VCF sample count does not match metadata num_haplotypes")
        out = output if metadata == direct else output / metadata.parent.name
        if any(c.isspace() for c in str(out)):
            raise ValueError("singer_master requires an output path without whitespace")
        if out.exists():
            raise ValueError(f"Output directory already exists: {out}; use --output-dir for a new run")
        row = [metadata, vcf, ne, mu, ratio, int(length), ploidy, out]
        if any(any(c in str(value) for c in "\t\n\r") for value in row):
            raise ValueError("Paths containing tabs/newlines are unsupported")
        print("\t".join(map(str, row)))
except (OSError, ValueError, KeyError, TypeError) as exc:
    sys.exit(f"Error: {exc}")
PY
)

printf 'Burn-in: %s iterations (%s saved samples); retain: %s ARGs; thin: %s; polar: 0.99\n' \
    "$((burnin_samples * thin))" "$burnin_samples" "$samples" "$thin"
while IFS=$'\t' read -r metadata vcf ne mu ratio length ploidy out; do
    # Stage the VCF to avoid using any existing SINGER .index sidecar. Keep all
    # generated files inside the output directory, including decompressed VCFs.
    input_prefix="$out/input"
    raw_prefix="$out/raw/singer"
    posterior_prefix="$out/singer"
    command=("$python_bin" "$singer_master" -vcf "$input_prefix" -output "$raw_prefix"
        -Ne "$ne" -m "$mu" -ratio "$ratio" -start 0 -end "$length"
        -n "$total_samples" -thin "$thin" -polar 0.99 -seed "$seed")
    # Verified with the installed 0.1.9 singer_master -h; diploid is the default.
    if ((ploidy == 1)); then command+=(-ploidy 1); fi
    convert=("$python_bin" "$converter" -input "$raw_prefix" -output "$posterior_prefix"
        -start "$burnin_samples" -end "$total_samples" -step 1)
    printf '\nDataset: %s\nOutput: %s\n' "${metadata%/*}" "$out"
    printf '%q ' "${command[@]}"; printf '\n'
    printf '%q ' "${convert[@]}"; printf '\n'
    if ((dry_run)); then continue; fi
    mkdir -p -- "${out%/*}"
    mkdir -- "$out"
    mkdir -- "$out/raw"
    if [[ "$vcf" == *.gz ]]; then gzip -dc -- "$vcf" > "$input_prefix.vcf"
    else ln -s -- "$vcf" "$input_prefix.vcf"; fi
    cp -- "$metadata" "$out/input_metadata.json"
    {
        printf '%q ' "${command[@]}"; printf '\n'
        printf '%q ' "${convert[@]}"; printf '\n'
    } > "$out/commands.txt"
    "$python_bin" - "$out/run.json" "$burnin" "$burnin_samples" "$samples" "$thin" "$seed" <<'PY'
import json
from pathlib import Path
import sys
path, burnin, discarded, retained, thin, seed = sys.argv[1:]
Path(path).write_text(json.dumps(dict(
    burnin_iterations_requested=int(burnin),
    burnin_iterations_effective=int(discarded) * int(thin),
    burnin_samples=int(discarded), posterior_samples=int(retained),
    thin=int(thin), seed=int(seed), polar=0.99,
    first_posterior_index=int(discarded),
    end_posterior_index_exclusive=int(discarded) + int(retained),
), indent=2) + "\n")
PY
    # Let singer_master own its recovery and seed handling.
    ulimit -c 0
    "${command[@]}" 2>&1 | tee "$out/singer.log"
    "${convert[@]}" 2>&1 | tee "$out/conversion.log"
    "$python_bin" - "$out" "$burnin_samples" "$total_samples" "$length" <<'PY'
import json
from pathlib import Path
import sys
import tskit
out = Path(sys.argv[1])
expected_samples = json.loads((out / "input_metadata.json").read_text())["summary"]["num_haplotypes"]
expected_count = int(sys.argv[3]) - int(sys.argv[2])
actual_count = len(list(out.glob("singer_*.trees")))
if actual_count != expected_count:
    sys.exit(f"Error: expected {expected_count} posterior trees, found {actual_count}")
for index in range(int(sys.argv[2]), int(sys.argv[3])):
    path = out / f"singer_{index}.trees"
    ts = tskit.load(path)
    if ts.num_samples != expected_samples or ts.sequence_length != int(sys.argv[4]):
        sys.exit(f"Error: unexpected sample count or sequence length in {path}")
(out / "SUCCESS").write_text("All retained posterior tree sequences loaded and validated.\n")
PY
    printf 'Saved %s posterior ARGs: %s_*.trees (burn-in already excluded)\n' "$samples" "$posterior_prefix"
done <<< "$dataset_info"
