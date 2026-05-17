"""Coverage and acceptance helpers for generated range query workloads."""

from __future__ import annotations

from typing import Any

import torch

from queries.range_geometry import points_in_range_box
from queries.workload_diagnostics import range_query_diagnostic

RANGE_COVERAGE_CALIBRATION_MODES = ("profile_sampled_query_count", "uncovered_anchor_chasing")


def _normalize_coverage_calibration_mode(mode: str | None, fallback: str) -> str:
    """Return a known target-coverage calibration mode."""
    normalized = str(mode or fallback).strip().lower()
    if normalized not in RANGE_COVERAGE_CALIBRATION_MODES:
        raise ValueError(
            "coverage_calibration_mode must be one of "
            f"{RANGE_COVERAGE_CALIBRATION_MODES}; got {mode!r}."
        )
    return normalized


def point_coverage_mask_for_query(points: torch.Tensor, query: dict[str, Any]) -> torch.Tensor:
    """Return the point-level dataset coverage induced by one query.

    Coverage follows the exact range box that produces query-specific training
    signal.
    """
    mask = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
    if points.numel() == 0:
        return mask

    query_type = str(query["type"]).lower()
    params = query["params"]
    if query_type == "range":
        return points_in_range_box(points, params)

    raise ValueError(f"Only range queries are supported for coverage; got query type: {query_type}")


def query_coverage_mask(points: torch.Tensor, typed_queries: list[dict[str, Any]]) -> torch.Tensor:
    """Return the union of covered points for a workload."""
    covered = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
    for query in typed_queries:
        covered |= point_coverage_mask_for_query(points, query)
    return covered


def _normalize_target_coverage(target_coverage: float | None) -> float | None:
    """Normalize coverage targets supplied as fractions or percentages."""
    if target_coverage is None:
        return None
    target = float(target_coverage)
    if target > 1.0:
        if target <= 100.0:
            target = target / 100.0
        else:
            raise ValueError(
                "target_coverage must be a fraction in (0, 1] or a percent in (0, 100]."
            )
    if target <= 0.0 or target > 1.0:
        raise ValueError("target_coverage must be a fraction in (0, 1] or a percent in (0, 100].")
    return target


def _normalize_coverage_overshoot(range_max_coverage_overshoot: float | None) -> float | None:
    """Normalize coverage overshoot tolerances supplied as fractions or percentages."""
    if range_max_coverage_overshoot is None:
        return None
    tolerance = float(range_max_coverage_overshoot)
    if tolerance > 1.0:
        if tolerance <= 100.0:
            tolerance = tolerance / 100.0
        else:
            raise ValueError(
                "range_max_coverage_overshoot must be a non-negative fraction or percent."
            )
    if tolerance < 0.0:
        raise ValueError("range_max_coverage_overshoot must be non-negative when provided.")
    return tolerance


def _range_acceptance_enabled(
    range_min_point_hits: int | None,
    range_max_point_hit_fraction: float | None,
    range_min_trajectory_hits: int | None,
    range_max_trajectory_hit_fraction: float | None,
    range_max_box_volume_fraction: float | None,
    range_duplicate_iou_threshold: float | None,
) -> bool:
    """Return whether any range acceptance filter is active."""
    return any(
        value is not None
        for value in (
            range_min_point_hits,
            range_max_point_hit_fraction,
            range_min_trajectory_hits,
            range_max_trajectory_hit_fraction,
            range_max_box_volume_fraction,
            range_duplicate_iou_threshold,
        )
    )


def _range_acceptance_state(
    enabled: bool, max_attempts: int | None, requested_queries: int
) -> dict[str, Any]:
    """Create JSON-safe acceptance counters for workload generation."""
    return {
        "enabled": bool(enabled),
        "attempts": 0,
        "accepted": 0,
        "rejected": 0,
        "rejection_reasons": {},
        "exhausted": False,
        "max_attempts": int(max_attempts) if max_attempts is not None else None,
        "minimum_queries": int(requested_queries),
        "requested_queries": int(requested_queries),
    }


def _record_rejection(state: dict[str, Any], reason: str) -> None:
    """Update range acceptance rejection counters."""
    state["rejected"] = int(state.get("rejected", 0)) + 1
    reasons = state.setdefault("rejection_reasons", {})
    reasons[reason] = int(reasons.get(reason, 0)) + 1


def _record_rejection_for_query(state: dict[str, Any], reason: str, query: dict[str, Any]) -> None:
    """Update rejection counters, including profile-family attribution when available."""
    _record_rejection(state, reason)
    metadata = query.get("_metadata") or {}
    anchor_family = metadata.get("anchor_family")
    footprint_family = metadata.get("footprint_family")
    if anchor_family is not None:
        by_anchor = state.setdefault("rejection_reasons_by_anchor_family", {})
        anchor_key = str(anchor_family)
        anchor_reasons = by_anchor.setdefault(anchor_key, {})
        anchor_reasons[reason] = int(anchor_reasons.get(reason, 0)) + 1
    if footprint_family is not None:
        by_footprint = state.setdefault("rejection_reasons_by_footprint_family", {})
        footprint_key = str(footprint_family)
        footprint_reasons = by_footprint.setdefault(footprint_key, {})
        footprint_reasons[reason] = int(footprint_reasons.get(reason, 0)) + 1


def _accept_range_query(
    query: dict[str, Any],
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    accepted_range_queries: list[dict[str, Any]],
    bounds: dict[str, float],
    *,
    range_min_point_hits: int | None,
    range_max_point_hit_fraction: float | None,
    range_min_trajectory_hits: int | None,
    range_max_trajectory_hit_fraction: float | None,
    range_max_box_volume_fraction: float | None,
    range_duplicate_iou_threshold: float | None,
) -> tuple[bool, str]:
    """Validate a generated range query against optional acceptance filters."""
    diagnostic = range_query_diagnostic(
        points,
        boundaries,
        query,
        query_index=len(accepted_range_queries),
        previous_range_queries=accepted_range_queries,
        bounds=bounds,
        max_point_hit_fraction=range_max_point_hit_fraction,
        max_trajectory_hit_fraction=range_max_trajectory_hit_fraction,
        max_box_volume_fraction=range_max_box_volume_fraction,
        duplicate_iou_threshold=range_duplicate_iou_threshold,
    )
    if range_min_point_hits is not None and diagnostic["point_hits"] < int(range_min_point_hits):
        return False, "too_few_point_hits"
    if range_min_trajectory_hits is not None and diagnostic["trajectory_hits"] < int(
        range_min_trajectory_hits
    ):
        return False, "too_few_trajectory_hits"
    if diagnostic["is_too_broad"]:
        return False, "too_broad"
    if range_duplicate_iou_threshold is not None and diagnostic["near_duplicate_of"] is not None:
        return False, "near_duplicate"
    return True, "accepted"
