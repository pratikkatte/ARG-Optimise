"""Public sequence and visualization utilities."""

from .sequences import load_sequences, read_fasta

__all__ = ["draw_state", "load_sequences", "read_fasta"]


def __getattr__(name):
    if name == "draw_state":
        from .visualization import draw_state
        return draw_state
    raise AttributeError(name)
