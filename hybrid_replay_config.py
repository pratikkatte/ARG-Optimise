"""Configuration contract for local hybrid experience replay."""

from __future__ import annotations

import math
from typing import Mapping


REPLAY_SOURCES = ("fresh", "residual", "reward", "topology")
DEFAULT_HYBRID_REPLAY_CONFIG = {
    "enabled": False,
    "capacity_per_context": 200,
    "fractions": {
        "fresh": 0.50,
        "residual": 0.25,
        "reward": 0.15,
        "topology": 0.10,
    },
    "priority_top_fraction": 0.10,
    "residual_priority": "max_abs_subtb",
    "priority_refresh": "on_use",
}


def normalize_hybrid_replay_config(config=None):
    """Validate and fill defaults for the local hybrid replay configuration."""

    raw = dict(config or {})
    unknown = sorted(set(raw) - set(DEFAULT_HYBRID_REPLAY_CONFIG))
    if unknown:
        raise ValueError(
            "training.hybrid_replay has unknown field(s): " + ", ".join(unknown)
        )
    normalized = dict(DEFAULT_HYBRID_REPLAY_CONFIG)
    normalized.update({key: value for key, value in raw.items() if key != "fractions"})
    if not isinstance(normalized["enabled"], bool):
        raise ValueError("training.hybrid_replay.enabled must be true or false")

    try:
        capacity = int(normalized["capacity_per_context"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            "training.hybrid_replay.capacity_per_context must be a positive integer"
        ) from error
    if capacity <= 0:
        raise ValueError(
            "training.hybrid_replay.capacity_per_context must be a positive integer"
        )
    normalized["capacity_per_context"] = capacity

    fractions = dict(DEFAULT_HYBRID_REPLAY_CONFIG["fractions"])
    supplied_fractions = raw.get("fractions") or {}
    if not isinstance(supplied_fractions, Mapping):
        raise ValueError("training.hybrid_replay.fractions must be a mapping")
    unknown_fractions = sorted(set(supplied_fractions) - set(REPLAY_SOURCES))
    if unknown_fractions:
        raise ValueError(
            "training.hybrid_replay.fractions has unknown source(s): "
            + ", ".join(unknown_fractions)
        )
    fractions.update(supplied_fractions)
    for name in REPLAY_SOURCES:
        try:
            value = float(fractions[name])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"training.hybrid_replay.fractions.{name} must be a number"
            ) from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"training.hybrid_replay.fractions.{name} must be finite and nonnegative"
            )
        fractions[name] = value
    if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("training.hybrid_replay.fractions must sum to 1")
    if normalized["enabled"] and fractions["fresh"] <= 0.0:
        raise ValueError(
            "training.hybrid_replay.fractions.fresh must be positive when replay is enabled"
        )
    normalized["fractions"] = fractions

    try:
        top_fraction = float(normalized["priority_top_fraction"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            "training.hybrid_replay.priority_top_fraction must be a number"
        ) from error
    if not math.isfinite(top_fraction) or not 0.0 < top_fraction <= 1.0:
        raise ValueError(
            "training.hybrid_replay.priority_top_fraction must be in (0, 1]"
        )
    normalized["priority_top_fraction"] = top_fraction

    if str(normalized["residual_priority"]) != "max_abs_subtb":
        raise ValueError(
            "training.hybrid_replay.residual_priority must be 'max_abs_subtb'"
        )
    normalized["residual_priority"] = "max_abs_subtb"
    if str(normalized["priority_refresh"]) != "on_use":
        raise ValueError(
            "training.hybrid_replay.priority_refresh must be 'on_use'"
        )
    normalized["priority_refresh"] = "on_use"
    return normalized


__all__ = [
    "DEFAULT_HYBRID_REPLAY_CONFIG",
    "REPLAY_SOURCES",
    "normalize_hybrid_replay_config",
]
