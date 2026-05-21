"""Shared artifact parsing helpers for derived diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def as_dict(value: Any) -> dict[str, Any]:
    """Return a dict payload or an empty dict for malformed artifact fields."""
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    """Return a list payload or an empty list for malformed artifact fields."""
    return value if isinstance(value, list) else []


def as_float(value: Any) -> float | None:
    """Return numeric artifact values as floats, preserving bool compatibility."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return None


def as_bool(value: Any) -> bool | None:
    """Return bool artifact values without truthiness guessing."""
    return value if isinstance(value, bool) else None


def delta(left: float | None, right: float | None) -> float | None:
    """Return left minus right when both values are present."""
    if left is None or right is None:
        return None
    return float(left - right)


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Return a guarded numeric ratio."""
    if numerator is None or denominator is None or abs(denominator) <= 1e-12:
        return None
    return float(numerator / denominator)


def load_json_dict(path: Path) -> dict[str, Any]:
    """Load a JSON artifact and enforce the object payload boundary."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object artifact: {path}")
    return payload
