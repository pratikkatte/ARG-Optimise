"""Models used by the ARG policy and likelihood."""

__all__ = ["ARGModel", "BreakpointSplitPositionCNN", "EvolutionModelTorch", "TimeModel"]


def __getattr__(name):
    """Load model classes lazily so environment and policy imports stay acyclic."""
    modules = {
        "ARGModel": ("arg_policy", "ARGModel"),
        "BreakpointSplitPositionCNN": ("breakpoint", "BreakpointSplitPositionCNN"),
        "EvolutionModelTorch": ("evolution", "EvolutionModelTorch"),
        "TimeModel": ("time", "TimeModel"),
    }
    if name not in modules:
        raise AttributeError(name)
    module_name, class_name = modules[name]
    from importlib import import_module
    return getattr(import_module(f"{__name__}.{module_name}"), class_name)
