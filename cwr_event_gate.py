"""Configuration for the local CwR-anchored structural event gate."""

from __future__ import annotations

import math
from typing import Any, Mapping


DEFAULT_LOCAL_CWR_EVENT_GATE_CONFIG = {
    "enabled": False,
    "max_abs_residual": 2.0,
}


def normalize_local_cwr_event_gate_config(config: Any) -> dict[str, Any]:
    """Validate and fill the public local CwR event-gate configuration."""

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("model.local_cwr_event_gate must be a mapping")
    unknown = sorted(set(config) - set(DEFAULT_LOCAL_CWR_EVENT_GATE_CONFIG))
    if unknown:
        raise ValueError(
            "model.local_cwr_event_gate contains unknown fields: "
            + ", ".join(unknown)
        )

    normalized = dict(DEFAULT_LOCAL_CWR_EVENT_GATE_CONFIG)
    normalized.update(dict(config))
    if not isinstance(normalized["enabled"], bool):
        raise ValueError("model.local_cwr_event_gate.enabled must be a boolean")
    raw_bound = normalized["max_abs_residual"]
    if isinstance(raw_bound, bool):
        raise ValueError(
            "model.local_cwr_event_gate.max_abs_residual must be a number"
        )
    try:
        bound = float(raw_bound)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "model.local_cwr_event_gate.max_abs_residual must be a number"
        ) from exc
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError(
            "model.local_cwr_event_gate.max_abs_residual must be finite and positive"
        )
    normalized["max_abs_residual"] = bound
    return normalized


__all__ = [
    "DEFAULT_LOCAL_CWR_EVENT_GATE_CONFIG",
    "normalize_local_cwr_event_gate_config",
]
