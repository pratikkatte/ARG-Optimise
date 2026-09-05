"""Shared PyTorch device selection."""

import torch


def resolve_device(name="auto"):
    """Resolve ``auto`` and reject unavailable CUDA requests."""
    name = "auto" if name is None else name
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but CUDA is unavailable")
    return device
