"""Workload signature helpers for generated range query workloads."""

from __future__ import annotations

from typing import Any

import torch

from queries.workload_diagnostics import compute_range_workload_diagnostics


def _counts_from_metadata(typed_queries: list[dict[str, Any]], key: str) -> dict[str, int]:
    """Count query metadata values."""
    counts: dict[str, int] = {}
    for query in typed_queries:
        metadata = query.get("_metadata") or {}
        value = str(metadata.get(key, "unspecified"))
        counts[value] = int(counts.get(value, 0)) + 1
    return counts


def _range_workload_signature(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    coverage_fraction: float,
    profile_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the v1 workload signature artifact described by the rework guide."""
    diagnostics = compute_range_workload_diagnostics(
        points=points,
        boundaries=boundaries,
        typed_queries=typed_queries,
        duplicate_iou_threshold=0.65,
        coverage_fraction=coverage_fraction,
    )
    summary = diagnostics["summary"]
    query_rows = diagnostics["queries"]
    point_hit_counts = [int(row["point_hits"]) for row in query_rows]
    trajectory_hit_counts = [int(row["trajectory_hits"]) for row in query_rows]
    point_hit_fractions = [float(row["point_hit_fraction"]) for row in query_rows]
    trajectory_hit_fractions = [float(row["trajectory_hit_fraction"]) for row in query_rows]
    spatial_radii = []
    time_spans = []
    for query in typed_queries:
        metadata = query.get("_metadata") or {}
        params = query["params"]
        spatial_radii.append(
            float(metadata.get("spatial_radius_km", metadata.get("range_spatial_km", 0.0)) or 0.0)
        )
        time_spans.append(float(params["t_end"] - params["t_start"]) / 3600.0)

    def quantiles(values: list[float]) -> dict[str, float]:
        if not values:
            return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "p10": float(torch.quantile(tensor, 0.10).item()),
            "p50": float(torch.quantile(tensor, 0.50).item()),
            "p90": float(torch.quantile(tensor, 0.90).item()),
        }

    profile_id = "legacy_generator"
    if profile_metadata is not None:
        profile_id = str(profile_metadata.get("profile_id", profile_id))
    return {
        "profile_id": profile_id,
        "coverage_actual": float(coverage_fraction),
        "query_count": len(typed_queries),
        "total_points": int(points.shape[0]),
        "total_trajectories": len(boundaries),
        "anchor_family_counts": _counts_from_metadata(typed_queries, "anchor_family"),
        "footprint_family_counts": _counts_from_metadata(typed_queries, "footprint_family"),
        "point_hits_per_query": {
            "p10": float(summary["point_hit_count_p10"]),
            "p50": float(summary["point_hit_count_p50"]),
            "p90": float(summary["point_hit_count_p90"]),
        },
        "point_hit_counts_per_query": point_hit_counts,
        "point_hit_fractions_per_query": point_hit_fractions,
        "ship_hits_per_query": {
            "p10": float(summary["trajectory_hit_count_p10"]),
            "p50": float(summary["trajectory_hit_count_p50"]),
            "p90": float(summary["trajectory_hit_count_p90"]),
        },
        "ship_hit_counts_per_query": trajectory_hit_counts,
        "ship_hit_fractions_per_query": trajectory_hit_fractions,
        "trajectory_hits_per_query": {
            "p10": float(summary["trajectory_hit_count_p10"]),
            "p50": float(summary["trajectory_hit_count_p50"]),
            "p90": float(summary["trajectory_hit_count_p90"]),
        },
        "trajectory_hit_counts_per_query": trajectory_hit_counts,
        "trajectory_hit_fractions_per_query": trajectory_hit_fractions,
        "time_span_hours_per_query": quantiles(time_spans),
        "spatial_radius_km_per_query": quantiles(spatial_radii),
        "near_duplicate_rate": float(summary["near_duplicate_query_rate"]),
        "broad_query_rate": float(summary["too_broad_query_rate"]),
        "empty_query_rate": float(summary["empty_query_rate"]),
        "train_eval_signature_distance": None,
    }
