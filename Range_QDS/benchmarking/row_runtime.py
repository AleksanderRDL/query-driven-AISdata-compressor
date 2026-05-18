"""Runtime and training-history row fields for benchmark reports."""

from __future__ import annotations

from typing import Any

from benchmarking.common import as_float


def dominant_runtime_phase_fields(
    timings: dict[str, Any], elapsed_seconds: float
) -> dict[str, Any]:
    """Return the largest parsed child phase and its fraction of wall time."""
    phase_rows = [
        row
        for row in timings.get("phase_timings", [])
        if row.get("name") is not None and as_float(row.get("seconds")) is not None
    ]
    if not phase_rows:
        return {
            "runtime_bottleneck_phase": None,
            "runtime_bottleneck_seconds": None,
            "runtime_bottleneck_fraction": None,
        }
    best = max(phase_rows, key=lambda row: float(row["seconds"]))
    seconds = float(best["seconds"])
    elapsed = as_float(elapsed_seconds)
    return {
        "runtime_bottleneck_phase": str(best["name"]),
        "runtime_bottleneck_seconds": seconds,
        "runtime_bottleneck_fraction": seconds / elapsed if elapsed and elapsed > 0.0 else None,
    }


def mean_history_value(run_json: dict[str, Any] | None, key: str) -> float | None:
    """Return mean numeric training-history value for one key."""
    history = (run_json or {}).get("training_history", [])
    values = [as_float(row.get(key)) for row in history if isinstance(row, dict)]
    numeric = [value for value in values if value is not None]
    return float(sum(numeric) / len(numeric)) if numeric else None


def last_history_value(run_json: dict[str, Any] | None, key: str) -> float | None:
    """Return the last numeric training-history value for one key."""
    history = (run_json or {}).get("training_history", [])
    for row in reversed(history):
        if not isinstance(row, dict):
            continue
        value = as_float(row.get(key))
        if value is not None:
            return value
    return None


def phase_seconds(timings: dict[str, Any], name: str) -> float | None:
    """Extract one phase duration from parsed timings."""
    for row in timings.get("phase_timings", []):
        if row.get("name") == name:
            return float(row["seconds"])
    return None


def phase_seconds_with_prefix(timings: dict[str, Any], prefix: str) -> float | None:
    """Extract the first phase duration whose name starts with a prefix."""
    for row in timings.get("phase_timings", []):
        if str(row.get("name", "")).startswith(prefix):
            return float(row["seconds"])
    return None


def mean_epoch_seconds(timings: dict[str, Any]) -> float | None:
    """Return mean epoch duration from parsed stdout timings."""
    values = [float(row["seconds"]) for row in timings.get("epoch_timings", [])]
    return float(sum(values) / len(values)) if values else None


def collapse_warning_summary(run_json: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize collapse diagnostics without conflating any epoch with the selected checkpoint."""
    if not run_json:
        return {
            "collapse_warning_any": None,
            "collapse_warning_count": None,
            "best_epoch_collapse_warning": None,
            "min_pred_std": None,
            "best_epoch_pred_std": None,
        }
    history = run_json.get("training_history", [])
    collapse_count = sum(1 for row in history if bool(row.get("collapse_warning", False)))
    pred_std_values = [float(row["pred_std"]) for row in history if row.get("pred_std") is not None]
    best_epoch = run_json.get("best_epoch")
    best_row = None
    if best_epoch is not None:
        best_epoch_int = int(best_epoch)
        for idx, row in enumerate(history):
            epoch_one_based = int(row.get("epoch", idx)) + 1
            if epoch_one_based == best_epoch_int:
                best_row = row
                break
    return {
        "collapse_warning_any": bool(collapse_count > 0),
        "collapse_warning_count": int(collapse_count),
        "best_epoch_collapse_warning": (
            bool(best_row.get("collapse_warning", False)) if best_row is not None else None
        ),
        "min_pred_std": min(pred_std_values) if pred_std_values else None,
        "best_epoch_pred_std": (
            float(best_row["pred_std"])
            if best_row is not None and best_row.get("pred_std") is not None
            else None
        ),
    }
