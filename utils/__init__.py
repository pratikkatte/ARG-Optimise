"""Public sequence and visualization utilities."""

from .sequences import load_sequences, read_fasta
from .device import resolve_device

__all__ = ["draw_state", "load_sequences", "read_fasta", "resolve_device"]


def __getattr__(name):
    if name == "draw_state":
        from .visualization import draw_state
        return draw_state
    raise AttributeError(name)
