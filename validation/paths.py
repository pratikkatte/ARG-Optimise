"""Shared paths and experiment-name validation."""

from __future__ import annotations

import re
from pathlib import Path


VALIDATION_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = VALIDATION_DIR / "output"
EXPERIMENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_experiment_name(name: str) -> str:
    """Return a safe experiment name or raise ``ValueError``."""
    if not isinstance(name, str) or not EXPERIMENT_NAME_RE.fullmatch(name):
        raise ValueError(
            "experiment name must start with a letter or number and contain only "
            "letters, numbers, '.', '_', or '-'"
        )
    if name in {".", ".."}:
        raise ValueError("experiment name cannot be '.' or '..'")
    return name


def experiment_dir(name: str, *, output_root: Path = OUTPUT_ROOT) -> Path:
    return output_root / validate_experiment_name(name)


def experiment_gfn_dir(name: str, *, output_root: Path = OUTPUT_ROOT) -> Path:
    return experiment_dir(name, output_root=output_root) / "gfn"

