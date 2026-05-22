"""Query workload generation for the AIS-QDS query types. See workloads/README.md for details."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import torch

from data_preparation.trajectory_index import boundaries_from_trajectories
from workloads.generation.anchors import (
    _anchor_weights_for_family,
    _anchor_weights_for_mode,
    _normalize_range_anchor_mode,
    _sample_anchor_point,
)
from workloads.generation.coverage import (
    _accept_range_query,
    _normalize_coverage_calibration_mode,
    _normalize_coverage_overshoot,
    _normalize_target_coverage,
    _range_acceptance_enabled,
    _range_acceptance_state,
    _record_rejection_for_query,
    point_coverage_mask_for_query,
    query_coverage_mask,
)
from workloads.generation.profile_query_plan import _profile_query_plan, _profile_query_settings
from workloads.generation.signatures import _range_workload_signature
from workloads.generation.workload_profiles import (
    LEGACY_GENERATOR_PROFILE,
    max_point_hit_fraction_for_coverage,
    range_workload_profile,
    workload_profile_metadata,
)
from workloads.query_types import (
    normalize_pure_workload_map,
    pad_query_features,
    validated_range_query_params,
)
from workloads.range_geometry import KM_PER_DEG_LAT, MIN_EQUIRECTANGULAR_COS_LAT
from workloads.typed_workload import TypedQueryWorkload

DEFAULT_RANGE_SPATIAL_FRACTION = 0.08
DEFAULT_RANGE_TIME_FRACTION = 0.15
DEFAULT_RANGE_FOOTPRINT_JITTER = 0.5
DEFAULT_RANGE_TIME_DOMAIN_MODE = "dataset"
DEFAULT_RANGE_ANCHOR_MODE = "mixed_density"
RANGE_TIME_DOMAIN_MODES = ("dataset", "anchor_day")
SECONDS_PER_DAY = 24.0 * 3600.0
EPOCH_LIKE_SECONDS = 366.0 * SECONDS_PER_DAY

_BuildCandidateRangeQuery = Callable[[torch.Tensor | None, int | None], dict[str, Any] | None]
_RecordAcceptedRangeQuery = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class _GenerationDiagnosticConfig:
    """Shared fields for query-generation diagnostics."""

    mode: str
    profile_id: str
    query_count_mode: str
    coverage_calibration_mode: str
    requested_queries: int
    max_queries: int
    target_coverage: float | None
    range_time_domain_mode: str
    range_anchor_mode: str
    range_spatial_fraction: float
    range_time_fraction: float
    range_spatial_km: float | None
    range_time_hours: float | None
    range_footprint_jitter: float
    range_max_coverage_overshoot: float | None
    coverage_guard_enabled: bool
    max_allowed_coverage: float | None
    profile_query_plan: dict[str, Any]


def _query_generation_diagnostics(
    config: _GenerationDiagnosticConfig,
    *,
    stop_reason: str,
    target_reached_query_count: int | None = None,
    coverage_at_target_reached: float | None = None,
    final_query_count: int | None = None,
) -> dict[str, Any]:
    """Build common query-generation diagnostics for fixed and coverage modes."""
    diagnostics: dict[str, Any] = {
        "mode": config.mode,
        "workload_profile_id": config.profile_id,
        "query_count_mode": config.query_count_mode,
        "coverage_calibration_mode": config.coverage_calibration_mode,
        "minimum_queries": int(config.requested_queries),
        "requested_queries": int(config.requested_queries),
        "max_queries": int(config.max_queries),
        "target_coverage": config.target_coverage,
        "range_time_domain_mode": config.range_time_domain_mode,
        "range_anchor_mode": config.range_anchor_mode,
        "range_spatial_fraction": float(config.range_spatial_fraction),
        "range_time_fraction": float(config.range_time_fraction),
        "range_spatial_km": config.range_spatial_km,
        "range_time_hours": config.range_time_hours,
        "range_footprint_jitter": float(config.range_footprint_jitter),
        "range_max_coverage_overshoot": config.range_max_coverage_overshoot,
        "coverage_guard_enabled": bool(config.coverage_guard_enabled),
        "max_allowed_coverage": config.max_allowed_coverage,
        "stop_reason": stop_reason,
        "profile_query_plan": {
            "enabled": bool(config.profile_query_plan.get("enabled", False)),
            "requested_queries": int(
                config.profile_query_plan.get("requested_queries", config.requested_queries)
            ),
            "anchor_family_planned_counts": dict(
                config.profile_query_plan.get("anchor_family_planned_counts") or {}
            ),
            "footprint_family_planned_counts": dict(
                config.profile_query_plan.get("footprint_family_planned_counts") or {}
            ),
        },
    }
    if config.mode == "target_coverage":
        diagnostics.update(
            {
                "target_reached_query_count": target_reached_query_count,
                "coverage_at_target_reached": coverage_at_target_reached,
                "extra_queries_after_target_reached": (
                    int((final_query_count or 0) - target_reached_query_count)
                    if target_reached_query_count is not None
                    else None
                ),
            }
        )
    return diagnostics


def _dataset_bounds(points: torch.Tensor) -> dict[str, float]:
    """Compute global point-cloud bounds for query generation. See workloads/README.md for details."""
    return {
        "t_min": float(points[:, 0].min().item()),
        "t_max": float(points[:, 0].max().item()),
        "lat_min": float(points[:, 1].min().item()),
        "lat_max": float(points[:, 1].max().item()),
        "lon_min": float(points[:, 2].min().item()),
        "lon_max": float(points[:, 2].max().item()),
    }


def _jitter_scale(generator: torch.Generator, jitter: float) -> float:
    """Return a random multiplicative scale in [1-jitter, 1+jitter]."""
    amount = float(jitter)
    if amount < 0.0:
        raise ValueError("range_footprint_jitter must be non-negative.")
    if amount <= 0.0:
        return 1.0
    scale = 1.0 + amount * (2.0 * float(torch.rand(1, generator=generator).item()) - 1.0)
    return max(1e-6, scale)


def _normalize_range_time_domain_mode(mode: str) -> str:
    """Normalize range-query time-domain mode names."""
    normalized = str(mode).strip().lower()
    if normalized not in RANGE_TIME_DOMAIN_MODES:
        raise ValueError(
            f"range_time_domain_mode must be one of {RANGE_TIME_DOMAIN_MODES}; got {mode!r}."
        )
    return normalized


def _anchor_day_time_bounds(anchor_time: float, bounds: dict[str, float]) -> tuple[float, float]:
    """Return the 24-hour time domain containing the anchor point.

    AIS tensors usually carry seconds relative to the loaded CSV minimum. If a
    caller passes epoch-like seconds, align to calendar UTC day boundaries;
    otherwise align to 24-hour source-file chunks from the dataset lower bound.
    """
    dataset_min = float(bounds["t_min"])
    dataset_max = float(bounds["t_max"])
    if dataset_max <= dataset_min:
        return dataset_min, dataset_max

    if dataset_min >= EPOCH_LIKE_SECONDS:
        day_start = math.floor(float(anchor_time) / SECONDS_PER_DAY) * SECONDS_PER_DAY
    else:
        day_offset = max(0.0, float(anchor_time) - dataset_min)
        day_start = dataset_min + math.floor(day_offset / SECONDS_PER_DAY) * SECONDS_PER_DAY
    day_end = day_start + SECONDS_PER_DAY
    return max(dataset_min, day_start), min(dataset_max, day_end)


def _query_time_bounds_for_mode(
    anchor_time: float,
    bounds: dict[str, float],
    range_time_domain_mode: str,
) -> tuple[float, float]:
    """Return the allowed temporal clamp bounds for one range query."""
    mode = _normalize_range_time_domain_mode(range_time_domain_mode)
    if mode == "dataset":
        return float(bounds["t_min"]), float(bounds["t_max"])
    return _anchor_day_time_bounds(float(anchor_time), bounds)


def _make_range_query(
    points: torch.Tensor,
    bounds: dict[str, float],
    generator: torch.Generator,
    anchor_mask: torch.Tensor | None = None,
    anchor_weights: torch.Tensor | None = None,
    anchor_weight_probability: float = 1.0,
    range_spatial_fraction: float = DEFAULT_RANGE_SPATIAL_FRACTION,
    range_time_fraction: float = DEFAULT_RANGE_TIME_FRACTION,
    range_spatial_km: float | None = None,
    range_time_hours: float | None = None,
    range_footprint_jitter: float = DEFAULT_RANGE_FOOTPRINT_JITTER,
    range_time_domain_mode: str = DEFAULT_RANGE_TIME_DOMAIN_MODE,
    elongation_allowed: bool = False,
    min_point_hits: int | None = None,
    min_point_hit_fraction: float | None = None,
    max_point_hit_fraction: float | None = None,
    target_point_hit_fraction: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate one range query. See workloads/README.md for details."""
    time_domain_mode = _normalize_range_time_domain_mode(range_time_domain_mode)
    spatial_fraction = float(range_spatial_fraction)
    time_fraction = float(range_time_fraction)
    spatial_km = None if range_spatial_km is None else float(range_spatial_km)
    time_hours = None if range_time_hours is None else float(range_time_hours)
    if (spatial_km is None and spatial_fraction <= 0.0) or (
        time_hours is None and time_fraction <= 0.0
    ):
        raise ValueError("range_spatial_fraction and range_time_fraction must be positive.")
    if spatial_km is not None and spatial_km <= 0.0:
        raise ValueError("range_spatial_km must be positive when provided.")
    if time_hours is not None and time_hours <= 0.0:
        raise ValueError("range_time_hours must be positive when provided.")
    anchor_point = _sample_anchor_point(
        points,
        generator,
        candidate_mask=anchor_mask,
        anchor_weights=anchor_weights,
        anchor_weight_probability=anchor_weight_probability,
    )
    lat_jitter = _jitter_scale(generator, range_footprint_jitter)
    lon_jitter = _jitter_scale(generator, range_footprint_jitter)
    time_jitter = _jitter_scale(generator, range_footprint_jitter)
    if spatial_km is None:
        lat_w = spatial_fraction * (bounds["lat_max"] - bounds["lat_min"]) * lat_jitter
        lon_w = spatial_fraction * (bounds["lon_max"] - bounds["lon_min"]) * lon_jitter
    else:
        lat_w = (spatial_km / KM_PER_DEG_LAT) * lat_jitter
        cos_lat = max(
            MIN_EQUIRECTANGULAR_COS_LAT,
            abs(math.cos(math.radians(float(anchor_point[1].item())))),
        )
        lon_w = (spatial_km / (KM_PER_DEG_LAT * cos_lat)) * lon_jitter
    corridor_axis = "none"
    elongation_factor = 1.0
    cross_axis_factor = 1.0
    if bool(elongation_allowed):
        heading = float(anchor_point[4].item()) if int(anchor_point.numel()) > 4 else 90.0
        heading_rad = math.radians(heading % 180.0)
        north_south_alignment = abs(math.cos(heading_rad))
        east_west_alignment = abs(math.sin(heading_rad))
        elongation_factor = 2.50
        cross_axis_factor = 0.45
        if east_west_alignment >= north_south_alignment:
            lon_w *= elongation_factor
            lat_w *= cross_axis_factor
            corridor_axis = "east_west"
        else:
            lat_w *= elongation_factor
            lon_w *= cross_axis_factor
            corridor_axis = "north_south"
    if time_hours is None:
        t_w = time_fraction * (bounds["t_max"] - bounds["t_min"]) * time_jitter
    else:
        t_w = time_hours * 3600.0 * time_jitter
    anchor_time = float(anchor_point[0].item())
    time_min, time_max = _query_time_bounds_for_mode(anchor_time, bounds, time_domain_mode)
    query_metadata = dict(metadata or {})

    def query_for_widths(lat_width: float, lon_width: float, time_width: float) -> dict[str, Any]:
        return {
            "type": "range",
            "params": {
                "lat_min": float(max(bounds["lat_min"], anchor_point[1].item() - lat_width)),
                "lat_max": float(min(bounds["lat_max"], anchor_point[1].item() + lat_width)),
                "lon_min": float(max(bounds["lon_min"], anchor_point[2].item() - lon_width)),
                "lon_max": float(min(bounds["lon_max"], anchor_point[2].item() + lon_width)),
                "t_start": float(max(time_min, anchor_time - time_width)),
                "t_end": float(min(time_max, anchor_time + time_width)),
            },
        }

    min_hits = max(0, int(min_point_hits or 0))
    total_points = int(points.shape[0])
    if isinstance(min_point_hit_fraction, (int, float)):
        min_hits = max(min_hits, math.ceil(float(min_point_hit_fraction) * total_points))
    max_hits: int | None = None
    if isinstance(max_point_hit_fraction, (int, float)):
        max_hits = max(0, math.floor(float(max_point_hit_fraction) * total_points))
    target_hits: int | None = None
    if isinstance(target_point_hit_fraction, (int, float)):
        target_hits = max(0, round(float(target_point_hit_fraction) * total_points))
        target_hits = max(target_hits, min_hits)
        if max_hits is not None:
            target_hits = min(target_hits, max_hits)
    calibrated_scale = 1.0
    query = query_for_widths(lat_w, lon_w, t_w)
    if min_hits > 0 or max_hits is not None:
        base_lat_w = float(lat_w)
        base_lon_w = float(lon_w)
        base_t_w = float(t_w)
        for _calibration_step in range(4):
            hit_count = int(point_coverage_mask_for_query(points, query).sum().item())
            if min_hits > 0 and hit_count < min_hits:
                scale = min(1.35, max(1.05, math.sqrt(float(min_hits) / max(1.0, hit_count))))
                calibrated_scale = min(2.0, calibrated_scale * scale)
            elif max_hits is not None and hit_count > max_hits:
                scale = max(0.75, math.sqrt(float(max_hits) / max(1.0, hit_count)))
                calibrated_scale = max(0.50, calibrated_scale * scale)
            elif target_hits is not None and target_hits > 0:
                lower_target = max(min_hits, math.floor(float(target_hits) * 0.90))
                upper_target = math.ceil(float(target_hits) * 1.10)
                if max_hits is not None:
                    upper_target = min(upper_target, max_hits)
                if hit_count < lower_target:
                    scale = min(
                        1.25, max(1.02, math.sqrt(float(target_hits) / max(1.0, hit_count)))
                    )
                    calibrated_scale = min(2.0, calibrated_scale * scale)
                elif hit_count > upper_target:
                    scale = max(
                        0.80, min(0.98, math.sqrt(float(target_hits) / max(1.0, hit_count)))
                    )
                    calibrated_scale = max(0.50, calibrated_scale * scale)
                else:
                    break
            else:
                break
            query = query_for_widths(
                base_lat_w * calibrated_scale,
                base_lon_w * calibrated_scale,
                base_t_w * calibrated_scale,
            )
        if abs(calibrated_scale - 1.0) > 1e-9:
            query_metadata["point_hit_band_calibration_scale"] = float(calibrated_scale)
    if bool(elongation_allowed):
        query_metadata.update(
            {
                "corridor_axis": corridor_axis,
                "corridor_elongation_factor": float(elongation_factor),
                "corridor_cross_axis_factor": float(cross_axis_factor),
            }
        )
    if query_metadata:
        query["_metadata"] = query_metadata
    return query


def _finalize_workload(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    generator: torch.Generator,
    generation_diagnostics: dict[str, Any] | None = None,
) -> TypedQueryWorkload:
    """Shuffle, featurize, and attach point-coverage metadata."""
    if typed_queries:
        shuffle_order = torch.randperm(len(typed_queries), generator=generator).tolist()
        typed_queries = [typed_queries[i] for i in shuffle_order]

    features, type_ids = pad_query_features(typed_queries)
    covered = query_coverage_mask(points, typed_queries)
    covered_points = int(covered.sum().item())
    total_points = int(points.shape[0])
    coverage_fraction = float(covered_points / total_points) if total_points > 0 else 0.0
    diagnostics = dict(generation_diagnostics or {})
    query_generation = dict(diagnostics.get("query_generation") or {})
    type_counts: dict[str, int] = {}
    for query in typed_queries:
        query_type = str(query.get("type", "unknown"))
        type_counts[query_type] = int(type_counts.get(query_type, 0)) + 1
    query_generation.update(
        {
            "final_query_count": len(typed_queries),
            "type_counts": type_counts,
            "covered_points": covered_points,
            "total_points": total_points,
            "final_coverage": coverage_fraction,
        }
    )
    diagnostics["query_generation"] = query_generation
    if typed_queries and all(
        str(query.get("type", "")).lower() == "range" for query in typed_queries
    ):
        profile_metadata = diagnostics.get("workload_profile")
        diagnostics["workload_signature"] = _range_workload_signature(
            points=points,
            boundaries=boundaries,
            typed_queries=typed_queries,
            coverage_fraction=coverage_fraction,
            profile_metadata=profile_metadata if isinstance(profile_metadata, dict) else None,
        )
    return TypedQueryWorkload(
        query_features=features,
        typed_queries=typed_queries,
        type_ids=type_ids,
        coverage_fraction=coverage_fraction,
        covered_points=covered_points,
        total_points=total_points,
        generation_diagnostics=diagnostics,
    )


def _generate_target_coverage_queries(
    *,
    points: torch.Tensor,
    build_candidate_range_query: _BuildCandidateRangeQuery,
    record_accepted_range_query: _RecordAcceptedRangeQuery,
    range_acceptance: dict[str, Any],
    coverage_target: float,
    query_limit: int,
    requested_queries: int,
    coverage_mode: str,
    coverage_guard_enabled: bool,
    max_allowed_coverage: float | None,
    max_range_attempts: int | None,
    calibrated_query_count_mode: bool,
) -> tuple[list[dict[str, Any]], str, int | None, float | None]:
    """Generate accepted range queries until coverage or attempt limits stop the loop."""
    generated_queries: list[dict[str, Any]] = []
    covered = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
    target_reached_query_count: int | None = None
    coverage_at_target_reached: float | None = None
    stop_reason = "max_queries_reached"

    while len(generated_queries) < query_limit:
        current_coverage = float(covered.float().mean().item()) if points.shape[0] > 0 else 0.0
        if (
            not calibrated_query_count_mode
            and len(generated_queries) >= requested_queries
            and current_coverage >= coverage_target
        ):
            stop_reason = "target_coverage_reached"
            break
        anchor_mask = (
            (~covered)
            if coverage_mode == "uncovered_anchor_chasing"
            and current_coverage < coverage_target
            else None
        )
        query = build_candidate_range_query(anchor_mask, len(generated_queries))
        if query is None:
            if range_acceptance.get("exhausted"):
                stop_reason = "range_acceptance_exhausted"
                break
            continue
        query_mask = point_coverage_mask_for_query(points, query)
        if coverage_guard_enabled and max_allowed_coverage is not None:
            candidate_coverage = (
                float((covered | query_mask).float().mean().item())
                if points.shape[0] > 0
                else 0.0
            )
            if candidate_coverage > max_allowed_coverage:
                _record_rejection_for_query(range_acceptance, "coverage_overshoot", query)
                if (
                    max_range_attempts is not None
                    and int(range_acceptance.get("attempts", 0)) >= max_range_attempts
                ):
                    range_acceptance["exhausted"] = True
                    stop_reason = "range_coverage_guard_exhausted"
                    break
                continue
        record_accepted_range_query(query)
        generated_queries.append(query)
        covered |= query_mask
        new_coverage = float(covered.float().mean().item()) if points.shape[0] > 0 else 0.0
        if target_reached_query_count is None and new_coverage >= coverage_target:
            target_reached_query_count = len(generated_queries)
            coverage_at_target_reached = float(new_coverage)
            if calibrated_query_count_mode and len(generated_queries) >= requested_queries:
                stop_reason = "target_coverage_reached"
                break

        final_coverage = float(covered.float().mean().item()) if points.shape[0] > 0 else 0.0
        if (
            stop_reason == "max_queries_reached"
            and len(generated_queries) >= requested_queries
            and final_coverage >= coverage_target
        ):
            stop_reason = "target_coverage_reached"
            break

    final_coverage = float(covered.float().mean().item()) if points.shape[0] > 0 else 0.0
    if (
        stop_reason == "max_queries_reached"
        and len(generated_queries) >= requested_queries
        and final_coverage >= coverage_target
    ):
        stop_reason = "target_coverage_reached"
    return (
        generated_queries,
        stop_reason,
        target_reached_query_count,
        coverage_at_target_reached,
    )


def _generate_fixed_count_queries(
    *,
    build_candidate_range_query: _BuildCandidateRangeQuery,
    record_accepted_range_query: _RecordAcceptedRangeQuery,
    range_acceptance: dict[str, Any],
    requested_queries: int,
) -> tuple[list[dict[str, Any]], str]:
    """Generate accepted range queries until the fixed query count is reached."""
    generated_queries: list[dict[str, Any]] = []
    stop_reason = "fixed_count_completed"
    while len(generated_queries) < requested_queries:
        query = build_candidate_range_query(None, len(generated_queries))
        if query is None:
            if range_acceptance.get("exhausted"):
                stop_reason = "range_acceptance_exhausted"
                break
            continue
        record_accepted_range_query(query)
        generated_queries.append(query)
    return generated_queries, stop_reason


def generate_typed_query_workload(
    trajectories: list[torch.Tensor],
    n_queries: int,
    workload_map: dict[str, float],
    seed: int,
    target_coverage: float | None = None,
    max_queries: int | None = None,
    range_spatial_fraction: float = DEFAULT_RANGE_SPATIAL_FRACTION,
    range_time_fraction: float = DEFAULT_RANGE_TIME_FRACTION,
    range_spatial_km: float | None = None,
    range_time_hours: float | None = None,
    range_footprint_jitter: float = DEFAULT_RANGE_FOOTPRINT_JITTER,
    range_time_domain_mode: str = DEFAULT_RANGE_TIME_DOMAIN_MODE,
    range_anchor_mode: str = DEFAULT_RANGE_ANCHOR_MODE,
    range_min_point_hits: int | None = None,
    range_max_point_hit_fraction: float | None = None,
    range_min_trajectory_hits: int | None = None,
    range_max_trajectory_hit_fraction: float | None = None,
    range_max_box_volume_fraction: float | None = None,
    range_duplicate_iou_threshold: float | None = None,
    range_acceptance_max_attempts: int | None = None,
    range_max_coverage_overshoot: float | None = None,
    workload_profile_id: str | None = None,
    coverage_calibration_mode: str | None = None,
) -> TypedQueryWorkload:
    """Generate a range-query workload and padded feature tensor. See workloads/README.md for details."""
    profile = range_workload_profile(workload_profile_id)
    profile_enabled = profile.profile_id != LEGACY_GENERATOR_PROFILE.profile_id
    time_domain_mode = _normalize_range_time_domain_mode(
        profile.time_domain_mode if profile_enabled else range_time_domain_mode
    )
    coverage_mode = _normalize_coverage_calibration_mode(
        coverage_calibration_mode,
        profile.coverage_calibration_mode if profile_enabled else "uncovered_anchor_chasing",
    )
    anchor_mode = _normalize_range_anchor_mode(range_anchor_mode)
    points = torch.cat(trajectories, dim=0)
    bounds = _dataset_bounds(points)
    boundaries = boundaries_from_trajectories(trajectories)

    normalize_pure_workload_map(workload_map)
    generator = torch.Generator().manual_seed(int(seed))
    anchor_weights, anchor_weight_probability = _anchor_weights_for_mode(points, anchor_mode)

    coverage_target = _normalize_target_coverage(
        profile.target_coverage if profile_enabled and target_coverage is None else target_coverage
    )
    coverage_overshoot = _normalize_coverage_overshoot(
        profile.max_coverage_overshoot
        if profile_enabled and range_max_coverage_overshoot is None
        else range_max_coverage_overshoot
    )
    range_max_point_hit_fraction_explicit = range_max_point_hit_fraction is not None
    if profile_enabled:
        if range_min_point_hits is None:
            range_min_point_hits = profile.min_points_per_query
        if range_min_trajectory_hits is None:
            range_min_trajectory_hits = profile.min_trajectories_per_query
        if range_max_point_hit_fraction is None:
            range_max_point_hit_fraction = max_point_hit_fraction_for_coverage(coverage_target)
        if range_duplicate_iou_threshold is None:
            range_duplicate_iou_threshold = profile.max_near_duplicate_hitset_jaccard
        if range_acceptance_max_attempts is None:
            range_acceptance_max_attempts = max(
                1, int(profile.max_attempt_multiplier) * max(1, int(n_queries))
            )
    coverage_guard_enabled = coverage_target is not None and coverage_overshoot is not None
    max_allowed_coverage = (
        min(1.0, float(coverage_target) + float(coverage_overshoot))
        if coverage_guard_enabled and coverage_target is not None and coverage_overshoot is not None
        else None
    )
    query_acceptance_enabled = _range_acceptance_enabled(
        range_min_point_hits,
        None,
        range_max_point_hit_fraction,
        range_min_trajectory_hits,
        range_max_trajectory_hit_fraction,
        range_max_box_volume_fraction,
        range_duplicate_iou_threshold,
    )
    acceptance_enabled = query_acceptance_enabled or coverage_guard_enabled
    requested_for_attempts = max(1, int(n_queries))
    default_max_attempts = 50 * requested_for_attempts if acceptance_enabled else None
    max_range_attempts = (
        int(range_acceptance_max_attempts)
        if range_acceptance_max_attempts is not None
        else default_max_attempts
    )
    if max_range_attempts is not None and max_range_attempts <= 0:
        raise ValueError("range_acceptance_max_attempts must be positive when provided.")
    range_acceptance = _range_acceptance_state(
        acceptance_enabled, max_range_attempts, requested_for_attempts
    )
    accepted_range_queries: list[dict[str, Any]] = []
    profile_query_plan_slots = requested_for_attempts
    if coverage_target is not None and max_queries is not None and int(max_queries) > 0:
        profile_query_plan_slots = max(profile_query_plan_slots, int(max_queries))
    profile_query_plan = _profile_query_plan(
        profile, requested_queries=profile_query_plan_slots, workload_seed=int(seed)
    )
    diagnostic_config = _GenerationDiagnosticConfig(
        mode="target_coverage" if coverage_target is not None else "fixed_count",
        profile_id=profile.profile_id,
        query_count_mode=profile.query_count_mode,
        coverage_calibration_mode=coverage_mode,
        requested_queries=requested_for_attempts,
        max_queries=requested_for_attempts,
        target_coverage=coverage_target,
        range_time_domain_mode=time_domain_mode,
        range_anchor_mode=anchor_mode,
        range_spatial_fraction=float(range_spatial_fraction),
        range_time_fraction=float(range_time_fraction),
        range_spatial_km=None if range_spatial_km is None else float(range_spatial_km),
        range_time_hours=None if range_time_hours is None else float(range_time_hours),
        range_footprint_jitter=float(range_footprint_jitter),
        range_max_coverage_overshoot=coverage_overshoot,
        coverage_guard_enabled=bool(coverage_guard_enabled),
        max_allowed_coverage=max_allowed_coverage,
        profile_query_plan=profile_query_plan,
    )

    def record_accepted_range_query(query: dict[str, Any]) -> None:
        """Record a query as accepted after all filters have passed."""
        if not acceptance_enabled:
            return
        range_acceptance["accepted"] = int(range_acceptance["accepted"]) + 1
        accepted_range_queries.append(
            {
                "type": "range",
                "params": validated_range_query_params(query),
                "query_index": len(accepted_range_queries),
            }
        )

    def build_candidate_range_query(
        anchor_mask: torch.Tensor | None = None,
        query_index: int | None = None,
    ) -> dict[str, Any] | None:
        """Build one query, applying optional range acceptance filters."""
        if acceptance_enabled:
            if (
                max_range_attempts is not None
                and int(range_acceptance["attempts"]) >= max_range_attempts
            ):
                range_acceptance["exhausted"] = True
                return None
            range_acceptance["attempts"] = int(range_acceptance["attempts"]) + 1
        profile_query = (
            _profile_query_settings(
                profile,
                generator,
                query_index=query_index,
                workload_seed=int(seed),
                query_plan=profile_query_plan,
            )
            if profile_enabled
            else {}
        )
        query_anchor_weights = anchor_weights
        query_anchor_probability = anchor_weight_probability
        query_spatial_km = range_spatial_km
        query_time_hours = range_time_hours
        query_min_point_hit_fraction: float | None = None
        query_max_point_hit_fraction = range_max_point_hit_fraction
        query_target_point_hit_fraction: float | None = None
        query_metadata: dict[str, Any] = {}
        if profile_query:
            query_anchor_weights, query_anchor_probability = _anchor_weights_for_family(
                points,
                str(profile_query["anchor_family"]),
            )
            query_spatial_km = float(profile_query["range_spatial_km"])
            query_time_hours = float(profile_query["range_time_hours"])
            profile_min_point_fraction = profile_query.get("min_point_hit_fraction")
            if isinstance(profile_min_point_fraction, (int, float)):
                query_min_point_hit_fraction = float(profile_min_point_fraction)
            profile_max_point_fraction = profile_query.get("max_point_hit_fraction")
            if not range_max_point_hit_fraction_explicit and isinstance(
                profile_max_point_fraction, (int, float)
            ):
                query_max_point_hit_fraction = (
                    min(float(query_max_point_hit_fraction), float(profile_max_point_fraction))
                    if query_max_point_hit_fraction is not None
                    else float(profile_max_point_fraction)
                )
            profile_target_point_fraction = profile_query.get("target_point_hit_fraction")
            if isinstance(profile_target_point_fraction, (int, float)):
                query_target_point_hit_fraction = float(profile_target_point_fraction)
            query_metadata = {
                "workload_profile_id": profile.profile_id,
                "anchor_family": str(profile_query["anchor_family"]),
                "footprint_family": str(profile_query["footprint_family"]),
                "spatial_radius_km": float(query_spatial_km),
                "time_half_window_hours": float(query_time_hours),
                "elongation_allowed": bool(profile_query.get("elongation_allowed", False)),
                "min_point_hit_fraction": query_min_point_hit_fraction,
                "max_point_hit_fraction": query_max_point_hit_fraction,
                "target_point_hit_fraction": query_target_point_hit_fraction,
            }
        query = _make_range_query(
            points,
            bounds,
            generator,
            anchor_mask=anchor_mask,
            anchor_weights=query_anchor_weights,
            anchor_weight_probability=query_anchor_probability,
            range_spatial_fraction=range_spatial_fraction,
            range_time_fraction=range_time_fraction,
            range_spatial_km=query_spatial_km,
            range_time_hours=query_time_hours,
            range_footprint_jitter=range_footprint_jitter,
            range_time_domain_mode=time_domain_mode,
            elongation_allowed=bool(profile_query.get("elongation_allowed", False)),
            min_point_hits=range_min_point_hits,
            min_point_hit_fraction=query_min_point_hit_fraction,
            max_point_hit_fraction=query_max_point_hit_fraction,
            target_point_hit_fraction=query_target_point_hit_fraction,
            metadata=query_metadata,
        )
        if not query_acceptance_enabled:
            return query
        accepted, reason = _accept_range_query(
            query,
            points,
            boundaries,
            accepted_range_queries,
            bounds,
            range_min_point_hits=range_min_point_hits,
            range_min_point_hit_fraction=query_min_point_hit_fraction,
            range_max_point_hit_fraction=query_max_point_hit_fraction,
            range_min_trajectory_hits=range_min_trajectory_hits,
            range_max_trajectory_hit_fraction=range_max_trajectory_hit_fraction,
            range_max_box_volume_fraction=range_max_box_volume_fraction,
            range_duplicate_iou_threshold=range_duplicate_iou_threshold,
        )
        if not accepted:
            _record_rejection_for_query(range_acceptance, reason, query)
            return None
        return query

    if coverage_target is not None:
        requested_queries = requested_for_attempts
        if max_queries is not None and int(max_queries) <= 0:
            raise ValueError("max_queries must be positive when target_coverage is set.")
        query_limit = max(
            requested_queries, int(max_queries) if max_queries is not None else requested_queries
        )
        generated_queries, stop_reason, target_reached_query_count, coverage_at_target_reached = (
            _generate_target_coverage_queries(
                points=points,
                build_candidate_range_query=build_candidate_range_query,
                record_accepted_range_query=record_accepted_range_query,
                range_acceptance=range_acceptance,
                coverage_target=float(coverage_target),
                query_limit=query_limit,
                requested_queries=requested_queries,
                coverage_mode=coverage_mode,
                coverage_guard_enabled=coverage_guard_enabled,
                max_allowed_coverage=max_allowed_coverage,
                max_range_attempts=max_range_attempts,
                calibrated_query_count_mode=profile.query_count_mode == "calibrated_to_coverage",
            )
        )
        query_generation = _query_generation_diagnostics(
            replace(diagnostic_config, max_queries=query_limit),
            stop_reason=stop_reason,
            target_reached_query_count=target_reached_query_count,
            coverage_at_target_reached=coverage_at_target_reached,
            final_query_count=len(generated_queries),
        )
        return _finalize_workload(
            points,
            boundaries,
            generated_queries,
            generator,
            generation_diagnostics={
                "workload_profile": workload_profile_metadata(profile),
                "range_acceptance": range_acceptance,
                "query_generation": query_generation,
            },
        )

    requested_queries = max(0, int(n_queries))
    generated_queries, stop_reason = _generate_fixed_count_queries(
        build_candidate_range_query=build_candidate_range_query,
        record_accepted_range_query=record_accepted_range_query,
        range_acceptance=range_acceptance,
        requested_queries=requested_queries,
    )

    return _finalize_workload(
        points,
        boundaries,
        generated_queries,
        generator,
        generation_diagnostics={
            "range_acceptance": range_acceptance,
            "workload_profile": workload_profile_metadata(profile),
            "query_generation": _query_generation_diagnostics(
                replace(
                    diagnostic_config,
                    requested_queries=requested_queries,
                    max_queries=requested_queries,
                    target_coverage=None,
                    coverage_guard_enabled=False,
                    max_allowed_coverage=None,
                ),
                stop_reason=stop_reason,
            ),
        },
    )
