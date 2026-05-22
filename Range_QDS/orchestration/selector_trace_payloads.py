"""Selector trace payload parsing and mask helpers."""

from __future__ import annotations

import math
from typing import Any

import torch


def selector_trace_mask_from_indices(
    *,
    indices: Any,
    point_count: int,
    source_name: str,
) -> torch.Tensor:
    """Build a bool mask from trace-persisted absolute indices."""
    if not isinstance(indices, list):
        raise ValueError(f"{source_name}.indices must be a list")
    mask = torch.zeros((int(point_count),), dtype=torch.bool)
    seen: set[int] = set()
    for raw_idx in indices:
        if isinstance(raw_idx, bool):
            raise ValueError(f"{source_name}.indices must contain integer indices")
        idx = int(raw_idx)
        if idx < 0 or idx >= int(point_count):
            raise ValueError(f"{source_name}.indices index out of bounds: {idx}")
        if idx in seen:
            raise ValueError(f"{source_name}.indices duplicate index: {idx}")
        seen.add(idx)
        mask[idx] = True
    return mask


def source_masks_from_selector_trace(
    selector_trace: dict[str, Any],
    *,
    point_count: int,
) -> dict[str, torch.Tensor]:
    """Return source-specific retained masks from learned-segment trace schema 7."""
    payload_names = {
        "skeleton": "skeleton_retained_mask",
        "learned": "learned_retained_mask",
        "fallback": "fallback_retained_mask",
        "length_repair": "length_repair_retained_mask",
    }
    out: dict[str, torch.Tensor] = {}
    for source, payload_name in payload_names.items():
        payload = selector_trace.get(payload_name)
        if not isinstance(payload, dict) or not bool(payload.get("available", False)):
            continue
        mask = selector_trace_mask_from_indices(
            indices=payload.get("indices"),
            point_count=point_count,
            source_name=payload_name,
        )
        declared_count = payload.get("retained_count")
        if declared_count is not None and int(declared_count) != int(mask.sum().item()):
            raise ValueError(
                f"{payload_name}.retained_count mismatch: "
                f"declared={int(declared_count)} actual={int(mask.sum().item())}"
            )
        out[source] = mask
    return out


def trace_mask_state_from_selector_trace(
    *,
    selector_trace: dict[str, Any] | None,
    point_count: int,
) -> dict[str, torch.Tensor]:
    if selector_trace is None:
        return {}
    mask_payload_names = {
        "final_retained": "retained_mask",
        "pre_repair_retained": "pre_repair_retained_mask",
        "skeleton_retained": "skeleton_retained_mask",
        "learned_retained": "learned_retained_mask",
        "fallback_retained": "fallback_retained_mask",
        "length_repair_retained": "length_repair_retained_mask",
    }
    out: dict[str, torch.Tensor] = {}
    for state_name, payload_name in mask_payload_names.items():
        payload = selector_trace.get(payload_name)
        if not isinstance(payload, dict) or not bool(payload.get("available", False)):
            continue
        try:
            out[state_name] = selector_trace_mask_from_indices(
                indices=payload.get("indices"),
                point_count=point_count,
                source_name=payload_name,
            )
        except ValueError:
            continue
    return out


def selector_segment_context_rows_from_trace(
    *,
    selector_trace: dict[str, Any] | None,
    point_count: int,
) -> list[dict[str, Any]]:
    if selector_trace is None:
        return []
    payload = selector_trace.get("segment_source_attribution")
    if not isinstance(payload, dict) or not bool(payload.get("available", False)):
        return []
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            continue
        try:
            start = int(raw_row["start"])
            end = int(raw_row["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or end > int(point_count) or end <= start:
            continue
        out.append(raw_row)
    return out


def optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def selector_segment_context_for_point(
    *,
    index: int,
    segment_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    point_index = int(index)
    for row in segment_rows:
        start = int(row["start"])
        end = int(row["end"])
        if start <= point_index < end:
            segment_length = max(1, end - start)
            offset = int(point_index - start)
            return {
                "source": "segment_source_attribution",
                "segment_index": optional_int(row.get("segment_index")),
                "allocation_order_index": optional_int(row.get("allocation_order_index")),
                "trajectory_index": optional_int(row.get("trajectory_id")),
                "segment_start": start,
                "segment_end": end,
                "segment_length": int(segment_length),
                "point_offset_in_segment": offset,
                "point_fraction_in_segment": float(offset / max(1, segment_length - 1)),
                "segment_score": optional_float(row.get("segment_score")),
                "segment_score_rank": optional_int(row.get("segment_score_rank")),
                "segment_score_source": str(row.get("segment_score_source", "")),
                "segment_length_support_score": optional_float(
                    row.get("segment_length_support_score")
                ),
                "segment_length_support_rank": optional_int(row.get("segment_length_support_rank")),
                "segment_allocation_weight": optional_float(row.get("segment_allocation_weight")),
                "segment_allocation_weight_rank": optional_int(
                    row.get("segment_allocation_weight_rank")
                ),
                "segment_allocation_count": optional_int(row.get("segment_allocation_count")),
                "retained_count": optional_int(row.get("retained_count")),
                "retained_fraction": optional_float(row.get("retained_fraction")),
                "skeleton_count": optional_int(row.get("skeleton_count")),
                "learned_count": optional_int(row.get("learned_count")),
                "fallback_count": optional_int(row.get("fallback_count")),
                "length_repair_count": optional_int(row.get("length_repair_count")),
                "unattributed_count": optional_int(row.get("unattributed_count")),
            }
    return None
