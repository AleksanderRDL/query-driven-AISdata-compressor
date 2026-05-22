"""Shared benchmark report value helpers."""

from __future__ import annotations

from typing import Any

LOW_COMPRESSION_THRESHOLD = 0.05 + 1e-9


def as_float(value: Any) -> float | None:
    """Coerce a metric-like value to float, preserving missing/non-numeric values."""
    if value is None:
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


def audit_ratio_prefix(ratio: float) -> str:
    """Return a stable, CSV-safe field prefix for one audit compression ratio."""
    return f"audit_ratio_{float(ratio):.4f}".replace(".", "p")
