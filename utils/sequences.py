"""Sequence file loading helpers."""

import pickle
from pathlib import Path


def read_fasta(path):
    """Return FASTA records as ``{header: sequence}``."""
    records, header, parts = {}, None, []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(parts)
                header, parts = line, []
            elif line:
                parts.append(line)
    if header is not None:
        records[header] = "".join(parts)
    return records


def load_sequences(path):
    """Load aligned sequences from FASTA or a pickled list/dictionary."""
    path = Path(path)
    if path.suffix.lower() in {".fa", ".fasta", ".fas"}:
        sequences = list(read_fasta(path).values())
    else:
        with path.open("rb") as handle:
            data = pickle.load(handle)
        sequences = list(data.values()) if isinstance(data, dict) else list(data)
    if not sequences:
        raise ValueError(f"no sequences found in {path}")
    return [sequence.replace("?", "-") for sequence in sequences]
