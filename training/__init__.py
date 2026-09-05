"""Configuration and runtime helpers for model training."""

from .config import TrainConfig

__all__ = ["TrainConfig", "evaluate_generator", "seed_everything", "train", "train_epoch"]


def __getattr__(name):
    if name == "evaluate_generator":
        from .evaluation import evaluate_generator
        return evaluate_generator
    if name in {"seed_everything", "train", "train_epoch"}:
        from . import loop
        return getattr(loop, name)
    raise AttributeError(name)
