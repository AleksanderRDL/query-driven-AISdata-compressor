"""Range-query audit scoring helpers."""

from __future__ import annotations

from typing import Any, cast

import torch

from data_preparation.trajectory_index import (
    trajectory_ids_for_points,
    trajectory_ids_from_mask,
)
from scoring.metrics import (
    _cumulative_polyline_length_km,
    _polyline_length_km,
    _trajectory_sed_ped_km,
    f1_score,
)
from scoring.query_cache import (
    RangeQueryAuditSupport,
    RangeSegmentAuditGeometry,
    RangeTrajectoryAuditSupport,
    ScoringQueryCache,
)
from scoring.range_query_metadata_summary import (
    RANGE_QUERY_COMPONENT_KEYS as RANGE_QUERY_COMPONENT_KEYS,
)
from scoring.range_query_metadata_summary import (
    _mean,
    _query_local_query_local_utility_summary,
    _range_query_family_labels,
    _range_query_metadata_component_summary,
)
from scoring.range_usefulness import (
    RANGE_USEFULNESS_SCHEMA_VERSION,
    RANGE_USEFULNESS_WEIGHTS,
    range_usefulness_gap_ablation_scores,
    range_usefulness_score_from_components,
    range_usefulness_weight_summary,
)
from workloads.range_geometry import (
    KM_PER_DEG_LAT,
    points_in_range_box,
    segment_box_bracket_indices,
    segment_pairs_box_crossings,
)


def _range_point_f1(retained_mask: torch.Tensor, range_mask: torch.Tensor) -> float:
    """Compute range F1 over retained point instances inside the query box."""
    return _point_subset_f1(
        retained_mask.to(device=range_mask.device, dtype=torch.bool), range_mask
    )


def _range_query_point_recall(retained_mask: torch.Tensor, range_mask: torch.Tensor) -> float:
    """Return direct retained query-point recall for one range query."""
    full_hits = int(range_mask.sum().item())
    if full_hits <= 0:
        return 1.0
    retained_hits = int(
        (retained_mask.to(device=range_mask.device, dtype=torch.bool) & range_mask).sum().item()
    )
    return float(max(0.0, min(1.0, retained_hits / full_hits)))


def score_range_boundary_preservation(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    retained_mask: torch.Tensor,
    typed_queries: list[dict],
    query_cache: ScoringQueryCache | None = None,
) -> float:
    """Score retained range entry/exit points separately from range point F1."""
    audit = score_range_usefulness(
        points=points,
        boundaries=boundaries,
        retained_mask=retained_mask,
        typed_queries=typed_queries,
        query_cache=query_cache,
    )
    return float(audit.get("range_entry_exit_f1", 0.0))


def _point_subset_f1(retained_mask: torch.Tensor, support_mask: torch.Tensor) -> float:
    full_hits = int(support_mask.sum().item())
    if full_hits <= 0:
        return 1.0
    retained_hits = int((retained_mask & support_mask).sum().item())
    if retained_hits <= 0:
        return 0.0
    recall = float(retained_hits / full_hits)
    return float((2.0 * recall) / (1.0 + recall))


def _point_index_subset_f1(retained_mask: torch.Tensor, support_indices_cpu: torch.Tensor) -> float:
    """Compute support-point F1 from compact support indices."""
    full_hits = int(support_indices_cpu.numel())
    if full_hits <= 0:
        return 1.0
    support_indices = support_indices_cpu.to(device=retained_mask.device, dtype=torch.long)
    retained_hits = int(retained_mask[support_indices].sum().item())
    if retained_hits <= 0:
        return 0.0
    recall = float(retained_hits / full_hits)
    return float((2.0 * recall) / (1.0 + recall))


def _range_boundary_indices_for_trajectories(
    range_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    trajectory_ids: list[int],
) -> torch.Tensor:
    """Return compact in-box entry/exit point indices for hit trajectories."""
    boundary_indices: list[torch.Tensor] = []
    for trajectory_id in trajectory_ids:
        if trajectory_id < 0 or trajectory_id >= len(boundaries):
            continue
        start, end = boundaries[trajectory_id]
        if end <= start:
            continue
        traj_in = range_mask[start:end]
        if not bool(traj_in.any().item()):
            continue
        enters = torch.zeros_like(traj_in)
        enters[1:] = traj_in[1:] & ~traj_in[:-1]
        enters[0] = traj_in[0]
        exits = torch.zeros_like(traj_in)
        exits[:-1] = traj_in[:-1] & ~traj_in[1:]
        exits[-1] = traj_in[-1]
        local_indices = torch.where(enters | exits)[0]
        if local_indices.numel() > 0:
            boundary_indices.append((local_indices + int(start)).detach().cpu())
    if not boundary_indices:
        return torch.empty((0,), dtype=torch.long)
    return torch.cat(boundary_indices).to(dtype=torch.long)


def _range_crossing_bracket_indices_for_trajectories(
    points_cpu: torch.Tensor,
    params: dict[str, float],
    boundaries: list[tuple[int, int]],
    segment_geometry: RangeSegmentAuditGeometry | None = None,
) -> torch.Tensor:
    """Return point pairs bracketing range-box boundary/pass-through crossings."""
    if segment_geometry is not None:
        candidates = (
            (segment_geometry.time_max_cpu >= float(params["t_start"]))
            & (segment_geometry.time_min_cpu <= float(params["t_end"]))
            & (segment_geometry.lat_max_cpu >= float(params["lat_min"]))
            & (segment_geometry.lat_min_cpu <= float(params["lat_max"]))
            & (segment_geometry.lon_max_cpu >= float(params["lon_min"]))
            & (segment_geometry.lon_min_cpu <= float(params["lon_max"]))
        )
        candidate_starts = segment_geometry.start_indices_cpu[candidates]
        if candidate_starts.numel() == 0:
            return torch.empty((0,), dtype=torch.long)
        crossing = segment_pairs_box_crossings(
            points_cpu[candidate_starts],
            points_cpu[candidate_starts + 1],
            params,
        )
        crossing_starts = candidate_starts[crossing]
        if crossing_starts.numel() == 0:
            return torch.empty((0,), dtype=torch.long)
        bracket_indices = torch.empty((crossing_starts.numel() * 2,), dtype=torch.long)
        bracket_indices[0::2] = crossing_starts
        bracket_indices[1::2] = crossing_starts + 1
        return torch.unique(bracket_indices, sorted=True).to(dtype=torch.long)
    return segment_box_bracket_indices(points_cpu, boundaries, params).detach().cpu()


def _range_turn_weights_for_points(points_cpu: torch.Tensor) -> torch.Tensor:
    """Return retained-independent route-change weights for one in-query trajectory slice."""
    count = int(points_cpu.shape[0])
    weights = torch.zeros((count,), dtype=torch.float32)
    if count >= 3:
        coords = points_cpu[:, 1:3].float()
        before = torch.linalg.vector_norm(coords[1:-1] - coords[:-2], dim=1)
        after = torch.linalg.vector_norm(coords[2:] - coords[1:-1], dim=1)
        shortcut = torch.linalg.vector_norm(coords[2:] - coords[:-2], dim=1)
        curvature = torch.clamp(before + after - shortcut, min=0.0)
        weights[1:-1] = curvature
    if points_cpu.shape[1] >= 8:
        weights = torch.maximum(weights, points_cpu[:, 7].float().clamp(min=0.0))
    return weights


def _build_range_query_audit_support(
    points_cpu: torch.Tensor,
    boundaries: list[tuple[int, int]],
    range_mask: torch.Tensor,
    point_trajectory_ids: torch.Tensor,
    params: dict[str, float],
    segment_geometry: RangeSegmentAuditGeometry | None = None,
) -> RangeQueryAuditSupport:
    """Build retained-independent support for one range query."""
    range_mask = range_mask.bool()
    full_ids = tuple(trajectory_ids_from_mask(range_mask, point_trajectory_ids))
    boundary_indices_cpu = _range_boundary_indices_for_trajectories(
        range_mask, boundaries, list(full_ids)
    )
    crossing_bracket_indices_cpu = _range_crossing_bracket_indices_for_trajectories(
        points_cpu,
        params,
        boundaries,
        segment_geometry,
    )
    range_mask_cpu = range_mask.detach().cpu()
    trajectory_support: list[RangeTrajectoryAuditSupport] = []
    for trajectory_id in full_ids:
        if trajectory_id < 0 or trajectory_id >= len(boundaries):
            continue
        start, end = boundaries[trajectory_id]
        if end <= start:
            continue
        in_offsets = torch.where(range_mask_cpu[start:end])[0].cpu()
        if in_offsets.numel() == 0:
            continue

        times = points_cpu[start:end, 0]
        full_span = float((times[in_offsets[-1]] - times[in_offsets[0]]).item())
        full_points = points_cpu[start + in_offsets]
        full_length = _polyline_length_km(full_points[:, 1], full_points[:, 2])
        distance_offsets = _cumulative_polyline_length_km(full_points[:, 1], full_points[:, 2])
        turn_weights = _range_turn_weights_for_points(full_points)
        trajectory_support.append(
            RangeTrajectoryAuditSupport(
                trajectory_id=int(trajectory_id),
                start=int(start),
                end=int(end),
                in_offsets_cpu=in_offsets,
                turn_weights_cpu=turn_weights.cpu(),
                distance_offsets_km_cpu=distance_offsets.cpu(),
                full_time_span=float(full_span),
                full_length_km=float(full_length),
            )
        )

    return RangeQueryAuditSupport(
        range_mask=range_mask,
        boundary_indices_cpu=boundary_indices_cpu,
        crossing_bracket_indices_cpu=crossing_bracket_indices_cpu,
        full_trajectory_ids=full_ids,
        trajectories=tuple(trajectory_support),
    )


def _range_query_audit_support(
    *,
    points: torch.Tensor,
    points_cpu: torch.Tensor,
    boundaries: list[tuple[int, int]],
    query_index: int,
    query: dict,
    point_trajectory_ids: torch.Tensor,
    query_cache: ScoringQueryCache | None,
) -> RangeQueryAuditSupport:
    """Return retained-independent audit support, using caller cache when available."""

    def build_range_mask() -> torch.Tensor:
        return points_in_range_box(points, query["params"])

    def build_support() -> RangeQueryAuditSupport:
        if query_cache is not None:
            range_mask = query_cache.get_support_mask(query_index, build_range_mask)
            segment_geometry = query_cache.get_range_segment_geometry(points_cpu, boundaries)
        else:
            range_mask = build_range_mask()
            segment_geometry = None
        return _build_range_query_audit_support(
            points_cpu=points_cpu,
            boundaries=boundaries,
            range_mask=range_mask,
            point_trajectory_ids=point_trajectory_ids,
            params=query["params"],
            segment_geometry=segment_geometry,
        )

    if query_cache is not None:
        return query_cache.get_range_audit_support(query_index, build_support)
    return build_support()


def _range_gap_coverage_for_offsets(
    in_offsets: torch.Tensor, retained_offsets: torch.Tensor
) -> float:
    """Score whether retained in-query points avoid one large local gap."""
    full_count = int(in_offsets.numel())
    retained_count = int(retained_offsets.numel())
    if full_count <= 0:
        return 1.0
    if full_count == 1:
        return 1.0 if retained_count > 0 else 0.0
    if retained_count <= 0:
        return 0.0

    retained_positions = torch.searchsorted(in_offsets, retained_offsets)
    leading_missing = retained_positions[0]
    trailing_missing = (full_count - 1) - retained_positions[-1]
    max_missing = torch.maximum(leading_missing, trailing_missing)
    if retained_positions.numel() >= 2:
        interior_missing = retained_positions[1:] - retained_positions[:-1] - 1
        if interior_missing.numel() > 0:
            max_missing = torch.maximum(max_missing, interior_missing.max())

    denom = float(max(1, full_count - 2))
    return float(max(0.0, min(1.0, 1.0 - float(max_missing.item()) / denom)))


def _range_gap_span_coverage_for_positions(
    values: torch.Tensor,
    retained_positions: torch.Tensor,
    full_span: float,
) -> float:
    """Score the largest missing run by elapsed time or along-track distance."""
    full_count = int(values.numel())
    retained_count = int(retained_positions.numel())
    if full_count <= 0:
        return 1.0
    if full_count == 1:
        return 1.0 if retained_count > 0 else 0.0
    if retained_count <= 0:
        return 0.0
    if full_span <= 1e-9:
        return 1.0

    retained_positions = retained_positions.to(dtype=torch.long)
    max_gap = values.new_tensor(0.0, dtype=torch.float32)
    first_pos = int(retained_positions[0].item())
    last_pos = int(retained_positions[-1].item())
    if first_pos > 0:
        max_gap = torch.maximum(max_gap, values[first_pos].float() - values[0].float())
    if last_pos < full_count - 1:
        max_gap = torch.maximum(max_gap, values[-1].float() - values[last_pos].float())
    if retained_positions.numel() >= 2:
        left = retained_positions[:-1]
        right = retained_positions[1:]
        has_missing = (right - left) > 1
        if bool(has_missing.any().item()):
            interior_spans = values[right[has_missing]].float() - values[left[has_missing]].float()
            max_gap = torch.maximum(max_gap, interior_spans.max())

    return float(max(0.0, min(1.0, 1.0 - float(max_gap.item()) / float(full_span))))


def _range_ship_coverage_for_offsets(
    in_offsets: torch.Tensor, retained_offsets: torch.Tensor
) -> float:
    """Return per-ship point-subset F1 for one in-query trajectory slice."""
    full_count = int(in_offsets.numel())
    if full_count <= 0:
        return 1.0
    retained_count = int(retained_offsets.numel())
    if retained_count <= 0:
        return 0.0
    recall = float(retained_count / full_count)
    return float((2.0 * recall) / (1.0 + recall))


def _range_turn_coverage_for_mask(
    turn_weights: torch.Tensor, retained_local: torch.Tensor
) -> float:
    """Return weighted point-subset F1 over route-change support."""
    turn_weights = turn_weights.to(dtype=torch.float32).clamp(min=0.0)
    full_mass = float(turn_weights.sum().item())
    if full_mass <= 1e-12:
        return 1.0
    retained_mass = float(turn_weights[retained_local].sum().item())
    if retained_mass <= 0.0:
        return 0.0
    recall = retained_mass / full_mass
    return float((2.0 * recall) / (1.0 + recall))


def _query_local_interpolation_fidelity(
    *,
    points_cpu: torch.Tensor,
    retained_cpu: torch.Tensor,
    support: RangeTrajectoryAuditSupport,
) -> float:
    """Score reconstruction of in-query points from retained full-trajectory anchors."""
    start = int(support.start)
    end = int(support.end)
    if end <= start:
        return 1.0
    in_offsets = support.in_offsets_cpu.long()
    if int(in_offsets.numel()) <= 0:
        return 1.0
    local_points = points_cpu[start:end]
    full_retained = retained_cpu[start:end].bool()
    retained_idx = torch.where(full_retained)[0]
    if int(retained_idx.numel()) < 2:
        return 0.0
    query_retained = full_retained[in_offsets]
    removed_offsets = in_offsets[~query_retained]
    if int(removed_offsets.numel()) == 0:
        return 1.0
    retained_in_query_count = int(query_retained.sum().item())
    local_evidence_factor = min(1.0, float(retained_in_query_count) / 2.0)
    pos = torch.searchsorted(retained_idx, removed_offsets)
    valid = (pos > 0) & (pos < int(retained_idx.numel()))
    if not bool(valid.any().item()):
        return 0.0
    removed_offsets = removed_offsets[valid]
    pos = pos[valid]
    left_idx = retained_idx[pos - 1]
    right_idx = retained_idx[pos]
    times = local_points[:, 0].float()
    lats = local_points[:, 1].float()
    lons = local_points[:, 2].float()
    t_l = times[left_idx]
    t_r = times[right_idx]
    t_p = times[removed_offsets]
    alpha = ((t_p - t_l) / (t_r - t_l).clamp(min=1e-9)).clamp(0.0, 1.0)
    interp_lat = lats[left_idx] + alpha * (lats[right_idx] - lats[left_idx])
    interp_lon = lons[left_idx] + alpha * (lons[right_idx] - lons[left_idx])
    cos_lat = torch.cos(torch.deg2rad(lats[removed_offsets]))
    dx_km = (lons[removed_offsets] - interp_lon) * cos_lat * KM_PER_DEG_LAT
    dy_km = (lats[removed_offsets] - interp_lat) * KM_PER_DEG_LAT
    sed_km = torch.sqrt(dx_km * dx_km + dy_km * dy_km)

    cos_lat_left = torch.cos(torch.deg2rad(lats[left_idx]))
    bx_km = (lons[right_idx] - lons[left_idx]) * cos_lat_left * KM_PER_DEG_LAT
    by_km = (lats[right_idx] - lats[left_idx]) * KM_PER_DEG_LAT
    px_km = (lons[removed_offsets] - lons[left_idx]) * cos_lat_left * KM_PER_DEG_LAT
    py_km = (lats[removed_offsets] - lats[left_idx]) * KM_PER_DEG_LAT
    chord_len = torch.sqrt(bx_km * bx_km + by_km * by_km).clamp(min=1e-9)
    ped_km = torch.abs(bx_km * py_km - by_km * px_km) / chord_len
    avg_error_km = float(((sed_km + ped_km) * 0.5).mean().item())
    avg_segment_km = support.full_length_km / float(max(1, int(in_offsets.numel()) - 1))
    raw_fidelity = 1.0 / (1.0 + avg_error_km / max(avg_segment_km, 1e-6))
    return float(max(0.0, min(1.0, raw_fidelity * local_evidence_factor)))


def _range_trajectory_detail_scores_for_query(
    points_cpu: torch.Tensor,
    retained_cpu: torch.Tensor,
    trajectory_support: tuple[RangeTrajectoryAuditSupport, ...],
) -> tuple[float, float, float, float, float, float, float, float]:
    """Return query-level per-ship coverage, temporal, gap, turn, and route-fidelity scores."""
    ship_coverage_scores: list[float] = []
    temporal_scores: list[float] = []
    gap_scores: list[float] = []
    gap_time_scores: list[float] = []
    gap_distance_scores: list[float] = []
    turn_scores: list[float] = []
    shape_scores: list[float] = []
    interpolation_scores: list[float] = []
    times = points_cpu[:, 0]
    for support in trajectory_support:
        in_offsets = support.in_offsets_cpu
        start = int(support.start)
        retained_local = retained_cpu[start + in_offsets]
        retained_offsets = in_offsets[retained_local]
        if retained_offsets.numel() == 0:
            ship_coverage_scores.append(0.0)
            temporal_scores.append(0.0)
            gap_scores.append(0.0)
            gap_time_scores.append(0.0)
            gap_distance_scores.append(0.0)
            turn_scores.append(0.0)
            shape_scores.append(0.0)
            interpolation_scores.append(
                _query_local_interpolation_fidelity(
                    points_cpu=points_cpu,
                    retained_cpu=retained_cpu,
                    support=support,
                )
            )
            continue

        ship_coverage_scores.append(_range_ship_coverage_for_offsets(in_offsets, retained_offsets))
        retained_positions = torch.searchsorted(in_offsets, retained_offsets)

        if support.full_time_span <= 1e-9:
            temporal_score = 1.0
        elif retained_offsets.numel() < 2:
            temporal_score = 0.0
        else:
            retained_span = float(
                (times[start + retained_offsets[-1]] - times[start + retained_offsets[0]]).item()
            )
            temporal_score = float(max(0.0, min(1.0, retained_span / support.full_time_span)))
        temporal_scores.append(temporal_score)
        gap_scores.append(_range_gap_coverage_for_offsets(in_offsets, retained_offsets))
        gap_time_scores.append(
            _range_gap_span_coverage_for_positions(
                values=times[start + in_offsets].cpu(),
                retained_positions=retained_positions,
                full_span=support.full_time_span,
            )
        )
        gap_distance_scores.append(
            _range_gap_span_coverage_for_positions(
                values=support.distance_offsets_km_cpu,
                retained_positions=retained_positions,
                full_span=support.full_length_km,
            )
        )
        turn_scores.append(_range_turn_coverage_for_mask(support.turn_weights_cpu, retained_local))
        interpolation_scores.append(
            _query_local_interpolation_fidelity(
                points_cpu=points_cpu,
                retained_cpu=retained_cpu,
                support=support,
            )
        )

        if support.full_length_km <= 1e-9:
            shape_scores.append(1.0)
        elif retained_offsets.numel() < 2:
            shape_scores.append(0.0)
        else:
            local_points = points_cpu[start + in_offsets]
            sed_sum, _sed_max, ped_sum, _ped_max, removed_count = _trajectory_sed_ped_km(
                local_points[:, 0],
                local_points[:, 1],
                local_points[:, 2],
                retained_local,
            )
            if removed_count <= 0:
                fidelity = 1.0
            else:
                avg_error_km = (sed_sum + ped_sum) / float(2 * removed_count)
                avg_segment_km = support.full_length_km / float(max(1, int(in_offsets.numel()) - 1))
                fidelity = 1.0 / (1.0 + avg_error_km / max(avg_segment_km, 1e-6))
            shape_scores.append(float(max(0.0, min(1.0, temporal_score * fidelity))))

    return (
        _mean(ship_coverage_scores, default=1.0),
        _mean(temporal_scores, default=1.0),
        _mean(gap_scores, default=1.0),
        _mean(gap_time_scores, default=1.0),
        _mean(gap_distance_scores, default=1.0),
        _mean(turn_scores, default=1.0),
        _mean(shape_scores, default=1.0),
        _mean(interpolation_scores, default=1.0),
    )


def _range_ship_evidence_counts_for_query(
    *,
    retained_cpu: torch.Tensor,
    trajectory_support: tuple[RangeTrajectoryAuditSupport, ...],
) -> dict[str, Any]:
    """Return per-query ship-presence evidence counts for family diagnostics."""
    full_counts: list[int] = []
    retained_counts: list[int] = []
    single_full = 0
    single_retained = 0
    multi_full = 0
    multi_retained = 0
    for support in trajectory_support:
        in_offsets = support.in_offsets_cpu
        full_count = int(in_offsets.numel())
        if full_count <= 0:
            continue
        start = int(support.start)
        retained_local = retained_cpu[start + in_offsets]
        retained_count = int(retained_local.sum().item())
        full_counts.append(full_count)
        retained_counts.append(retained_count)
        if full_count == 1:
            single_full += 1
            if retained_count > 0:
                single_retained += 1
        else:
            multi_full += 1
            if retained_count > 0:
                multi_retained += 1

    full_ship_count = len(full_counts)
    retained_ship_count = sum(1 for count in retained_counts if count > 0)
    missed_ship_count = max(0, full_ship_count - retained_ship_count)
    return {
        "full_trajectory_hit_count": int(full_ship_count),
        "retained_trajectory_hit_count": int(retained_ship_count),
        "missed_trajectory_hit_count": int(missed_ship_count),
        "missed_trajectory_hit_fraction": float(
            missed_ship_count / full_ship_count if full_ship_count > 0 else 0.0
        ),
        "ship_presence_recall": float(
            retained_ship_count / full_ship_count if full_ship_count > 0 else 1.0
        ),
        "single_point_full_trajectory_hit_count": int(single_full),
        "single_point_retained_trajectory_hit_count": int(single_retained),
        "single_point_ship_presence_recall": float(
            single_retained / single_full if single_full > 0 else 1.0
        ),
        "multi_point_full_trajectory_hit_count": int(multi_full),
        "multi_point_retained_trajectory_hit_count": int(multi_retained),
        "multi_point_ship_presence_recall": float(
            multi_retained / multi_full if multi_full > 0 else 1.0
        ),
        "full_query_hit_points_per_hit_trajectory_mean": _mean(
            [float(count) for count in full_counts]
        ),
        "retained_query_hit_points_per_hit_trajectory_mean": _mean(
            [float(count) for count in retained_counts]
        ),
    }


def score_range_usefulness(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    retained_mask: torch.Tensor,
    typed_queries: list[dict],
    query_cache: ScoringQueryCache | None = None,
) -> dict[str, Any]:
    """Audit range simplification with point, ship, crossing, temporal, gap, turn, and shape components.

    `range_point_f1` is the current retained in-box point metric. The combined
    `range_usefulness_score` is an audit score for comparing candidates; it is
    intentionally reported separately instead of replacing final F1 semantics.
    """
    if query_cache is not None:
        query_cache.validate(points, boundaries, typed_queries)

    retained_bool = retained_mask.to(device=points.device, dtype=torch.bool)
    point_trajectory_ids = trajectory_ids_for_points(points.shape[0], boundaries, points.device)
    points_cpu = points.detach().cpu()
    retained_cpu = retained_bool.detach().cpu()

    point_recall_scores: list[float] = []
    point_scores: list[float] = []
    ship_scores: list[float] = []
    ship_coverage_scores: list[float] = []
    entry_exit_scores: list[float] = []
    crossing_scores: list[float] = []
    temporal_scores: list[float] = []
    gap_scores: list[float] = []
    gap_time_scores: list[float] = []
    gap_distance_scores: list[float] = []
    turn_scores: list[float] = []
    shape_scores: list[float] = []
    interpolation_scores: list[float] = []
    query_component_rows: list[dict[str, Any]] = []

    for query_index, query in enumerate(typed_queries):
        if str(query.get("type", "")).lower() != "range":
            continue
        support = _range_query_audit_support(
            points=points,
            points_cpu=points_cpu,
            boundaries=boundaries,
            query_index=query_index,
            query=query,
            point_trajectory_ids=point_trajectory_ids,
            query_cache=query_cache,
        )
        range_mask = support.range_mask.to(device=points.device, dtype=torch.bool)
        retained_in_range = retained_bool & range_mask
        point_recall = _range_query_point_recall(retained_bool, range_mask)
        point_score = _range_point_f1(retained_bool, range_mask)
        point_recall_scores.append(point_recall)
        point_scores.append(point_score)

        retained_ids = trajectory_ids_from_mask(retained_in_range, point_trajectory_ids)
        ship_score = f1_score(set(support.full_trajectory_ids), set(retained_ids))
        ship_scores.append(ship_score)
        ship_evidence_counts = _range_ship_evidence_counts_for_query(
            retained_cpu=retained_cpu,
            trajectory_support=support.trajectories,
        )

        entry_exit_score = _point_index_subset_f1(retained_bool, support.boundary_indices_cpu)
        crossing_score = _point_index_subset_f1(retained_bool, support.crossing_bracket_indices_cpu)
        entry_exit_scores.append(entry_exit_score)
        crossing_scores.append(crossing_score)

        (
            ship_coverage,
            temporal_score,
            gap_score,
            gap_time_score,
            gap_distance_score,
            turn_score,
            shape_score,
            interpolation_score,
        ) = _range_trajectory_detail_scores_for_query(
            points_cpu=points_cpu,
            retained_cpu=retained_cpu,
            trajectory_support=support.trajectories,
        )
        ship_coverage_scores.append(ship_coverage)
        temporal_scores.append(temporal_score)
        gap_scores.append(gap_score)
        gap_time_scores.append(gap_time_score)
        gap_distance_scores.append(gap_distance_score)
        turn_scores.append(turn_score)
        shape_scores.append(shape_score)
        interpolation_scores.append(interpolation_score)
        range_gap_min_coverage = min(float(gap_time_score), float(gap_distance_score))
        row_components = {
            "query_point_recall": float(point_recall),
            "range_point_f1": float(point_score),
            "range_ship_f1": float(ship_score),
            "range_ship_coverage": float(ship_coverage),
            "range_entry_exit_f1": float(entry_exit_score),
            "range_crossing_f1": float(crossing_score),
            "range_temporal_coverage": float(temporal_score),
            "range_gap_coverage": float(gap_score),
            "range_gap_time_coverage": float(gap_time_score),
            "range_gap_distance_coverage": float(gap_distance_score),
            "range_gap_min_coverage": float(range_gap_min_coverage),
            "range_turn_coverage": float(turn_score),
            "range_shape_score": float(shape_score),
            "range_query_local_interpolation_fidelity": float(interpolation_score),
        }
        anchor_family, footprint_family = _range_query_family_labels(cast(dict[str, Any], query))
        query_component_rows.append(
            {
                "query_index": int(query_index),
                "anchor_family": anchor_family,
                "footprint_family": footprint_family,
                "full_point_hit_count": int(range_mask.sum().item()),
                "retained_point_hit_count": int(retained_in_range.sum().item()),
                "full_trajectory_hit_count": len(support.full_trajectory_ids),
                "retained_trajectory_hit_count": len(retained_ids),
                "ship_evidence_counts": ship_evidence_counts,
                "range_components": row_components,
                "range_usefulness_score": float(
                    range_usefulness_score_from_components(row_components)
                ),
                **_query_local_query_local_utility_summary(row_components),
            }
        )

    query_count = len(point_scores)
    query_point_recall = _mean(point_recall_scores)
    range_point_f1 = _mean(point_scores)
    range_ship_f1 = _mean(ship_scores)
    range_ship_coverage = _mean(ship_coverage_scores)
    range_entry_exit_f1 = _mean(entry_exit_scores)
    range_crossing_f1 = _mean(crossing_scores)
    range_temporal_coverage = _mean(temporal_scores)
    range_gap_coverage = _mean(gap_scores)
    range_gap_time_coverage = _mean(gap_time_scores)
    range_gap_distance_coverage = _mean(gap_distance_scores)
    range_turn_coverage = _mean(turn_scores)
    range_shape_score = _mean(shape_scores)
    range_query_local_interpolation_fidelity = _mean(interpolation_scores)
    components = {
        "query_point_recall": query_point_recall,
        "range_point_f1": range_point_f1,
        "range_ship_f1": range_ship_f1,
        "range_ship_coverage": range_ship_coverage,
        "range_entry_exit_f1": range_entry_exit_f1,
        "range_crossing_f1": range_crossing_f1,
        "range_temporal_coverage": range_temporal_coverage,
        "range_gap_coverage": range_gap_coverage,
        "range_gap_time_coverage": range_gap_time_coverage,
        "range_gap_distance_coverage": range_gap_distance_coverage,
        "range_turn_coverage": range_turn_coverage,
        "range_shape_score": range_shape_score,
        "range_query_local_interpolation_fidelity": range_query_local_interpolation_fidelity,
    }
    range_usefulness_score = range_usefulness_score_from_components(components)
    gap_ablation_scores = range_usefulness_gap_ablation_scores(components)
    return {
        "range_usefulness_schema_version": int(RANGE_USEFULNESS_SCHEMA_VERSION),
        "range_query_count": int(query_count),
        "query_point_recall": float(query_point_recall),
        "range_point_f1": float(range_point_f1),
        "range_ship_f1": float(range_ship_f1),
        "range_ship_coverage": float(range_ship_coverage),
        "range_entry_exit_f1": float(range_entry_exit_f1),
        "range_crossing_f1": float(range_crossing_f1),
        "range_temporal_coverage": float(range_temporal_coverage),
        "range_gap_coverage": float(range_gap_coverage),
        "range_gap_time_coverage": float(range_gap_time_coverage),
        "range_gap_distance_coverage": float(range_gap_distance_coverage),
        "range_gap_min_coverage": float(gap_ablation_scores["range_gap_min_coverage"]),
        "range_turn_coverage": float(range_turn_coverage),
        "range_shape_score": float(range_shape_score),
        "range_query_local_interpolation_fidelity": float(range_query_local_interpolation_fidelity),
        "range_usefulness_score": float(range_usefulness_score),
        "range_usefulness_gap_time_score": float(
            gap_ablation_scores["range_usefulness_gap_time_score"]
        ),
        "range_usefulness_gap_distance_score": float(
            gap_ablation_scores["range_usefulness_gap_distance_score"]
        ),
        "range_usefulness_gap_min_score": float(
            gap_ablation_scores["range_usefulness_gap_min_score"]
        ),
        "range_usefulness_gap_ablation_version": int(
            gap_ablation_scores["range_usefulness_gap_ablation_version"]
        ),
        "range_usefulness_weights": dict(RANGE_USEFULNESS_WEIGHTS),
        "range_usefulness_weight_summary": range_usefulness_weight_summary(),
        "range_query_metadata_component_summary": _range_query_metadata_component_summary(
            query_component_rows
        ),
    }
