#!/usr/bin/env python3
"""Infer a saved GFN checkpoint, then run point accuracy against dataset truth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Training run, e.g. runs/human_2kb_super_easy")
    parser.add_argument("--checkpoint", type=Path, default=Path("best.pt"),
                        help="Filename inside RUN_DIR/checkpoints, or an absolute path")
    parser.add_argument("--dataset-dir", type=Path,
                        help="Default: validation/datasets/<run directory name>")
    parser.add_argument("--replicate", default="rep0", help="Dataset replicate (default: rep0)")
    parser.add_argument("--output-dir", type=Path,
                        help="Override the directory for samples and its validation/ subfolder")
    parser.add_argument("--num-args", "--num-particles", type=positive_int, default=100)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, help="Default: checkpoint seed")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and print commands only")
    args = parser.parse_args(argv)
    if args.temperature is not None and (not math.isfinite(args.temperature) or args.temperature <= 0):
        parser.error("--temperature must be positive and finite")
    if not args.replicate or Path(args.replicate).name != args.replicate or args.replicate in (".", ".."):
        parser.error("--replicate must be a directory name such as rep0")
    return args


def build_workflow(args):
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser()
    if not checkpoint.is_absolute():
        checkpoint = run_dir / "checkpoints" / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint not found: {checkpoint}")
    dataset = (args.dataset_dir.expanduser().resolve() if args.dataset_dir
               else REPO_ROOT / "validation" / "datasets" / run_dir.name)
    replicate = dataset / args.replicate
    metadata = json.loads((replicate / "metadata.json").read_text())
    truth = (replicate / metadata["files"]["ground_truth_trees"]).resolve()
    fasta = (replicate / metadata["files"]["fasta"]).resolve()
    for path in (truth, fasta):
        if not path.is_file():
            raise ValueError(f"Dataset input not found: {path}")
    output = (args.output_dir.expanduser().resolve() if args.output_dir else
              dataset / "output" / "gfn" / args.replicate / checkpoint.stem)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output is not empty: {output}; use --output-dir for a new run")

    # Inference uses the sequences embedded in the checkpoint. Validate their
    # order as well as their content before comparing to this replicate's truth.
    sys.path.insert(0, str(REPO_ROOT))
    from infer import load_checkpoint, validate_metadata
    from utils import load_sequences

    saved = load_checkpoint(str(checkpoint), map_location="cpu")["metadata"]
    validate_metadata(saved)
    if list(saved["sequences"]) != load_sequences(str(fasta)):
        raise ValueError(f"Checkpoint sequences do not match {fasta}; select the training replicate")
    nspl = int(metadata["summary"]["num_haplotypes"])
    length = int(metadata["summary"]["sequence_length_bp"])
    if nspl != int(saved["num_sequences"]) or length != int(saved["sequence_length"]):
        raise ValueError("Dataset dimensions do not match checkpoint metadata")
    ne = float(metadata["simulation"]["sim_ancestry"]["parameters"]["population_size"])
    if not math.isfinite(ne) or ne <= 0:
        raise ValueError("Dataset population_size must be positive and finite")
    seed = int(saved["seed"] if args.seed is None else args.seed)
    inference = [sys.executable, str(REPO_ROOT / "infer.py"),
                 "--checkpoint", str(checkpoint), "--output-dir", str(output),
                 "--num-args", str(args.num_args), "--batch-size", str(args.batch_size),
                 "--device", args.device, "--seed", str(seed)]
    if args.temperature is not None:
        inference.extend(["--temperature", str(args.temperature)])
    validation = [sys.executable, str(REPO_ROOT / "validation/scripts/point_accuracy_gfn.py"),
                  "--truth-trees", str(truth), "--ne", str(ne), "--nspl", str(nspl),
                  "--input-dir", str(output), "--sample-prefix", "arg_",
                  "--output-prefix", str(output / "validation" / "gfn_")]
    if args.verbose:
        inference.append("--verbose")
        validation.append("--verbose")
    return output, {"checkpoint": str(checkpoint), "dataset_dir": str(dataset),
                    "replicate": args.replicate, "truth_trees": str(truth),
                    "seed": seed, "num_args": args.num_args,
                    "inference_command": inference, "validation_command": validation}


def run_logged(command, log_path):
    with log_path.open("w") as log:
        with subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)


def main(argv=None):
    args = parse_args(argv)
    try:
        output, workflow = build_workflow(args)
        for stage in ("inference", "validation"):
            print(f"{stage}: {shlex.join(workflow[stage + '_command'])}", flush=True)
        if args.dry_run:
            return 0
        output.mkdir(parents=True, exist_ok=True)
        (output / "workflow.json").write_text(json.dumps(workflow, indent=2) + "\n")
        run_logged(workflow["inference_command"], output / "inference.log")
        (output / "validation").mkdir()
        run_logged(workflow["validation_command"], output / "validation" / "validation.log")
        print(f"ARG samples: {output}\nValidation: {output / 'validation'}")
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
