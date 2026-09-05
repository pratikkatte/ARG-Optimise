"""Compatibility facade for the split point-accuracy package."""

try:
    from .point_accuracy import *  # noqa: F401,F403
except ImportError:
    from point_accuracy import *  # noqa: F401,F403
