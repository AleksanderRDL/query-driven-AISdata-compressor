"""Segment-level QueryLocalUtility target helpers."""

from __future__ import annotations

import math

import torch

from workloads.range_geometry import local_equirectangular_distance_km


def _segment_budget_targets(
    point_value: torch.Tensor,
    boundaries: list[tuple[int, int]],
    segment_size: int,
) -> torch.Tensor:
    """Assign each point its segment's normalized query-local value mass."""
    out = torch.zeros_like(point_value.float())
    segment_masses: list[torch.Tensor] = []
    segment_slices: list[tuple[int, int]] = []
    size = max(1, int(segment_size))
    for start, end in boundaries:
        for seg_start in range(int(start), int(end), size):
            seg_end = min(int(end), seg_start + size)
            if seg_end <= seg_start:
                continue
            mass = point_value[seg_start:seg_end].float().clamp(min=0.0).sum()
            segment_masses.append(mass)
            segment_slices.append((seg_start, seg_end))
    if not segment_masses:
        return out
    masses = torch.stack(segment_masses)
    max_mass = masses.max().clamp(min=1e-6)
    normalized = (masses / max_mass).clamp(0.0, 1.0)
    for value, (seg_start, seg_end) in zip(normalized, segment_slices, strict=True):
        out[seg_start:seg_end] = value
    return out


def _segment_pooled_targets(
    point_value: torch.Tensor,
    boundaries: list[tuple[int, int]],
    segment_size: int,
    *,
    pool: str,
) -> torch.Tensor:
    """Assign each point its segment's pooled point value for allocation diagnostics."""
    out = torch.zeros_like(point_value.float())
    segment_values: list[torch.Tensor] = []
    segment_slices: list[tuple[int, int]] = []
    size = max(1, int(segment_size))
    for start, end in boundaries:
        for seg_start in range(int(start), int(end), size):
            seg_end = min(int(end), seg_start + size)
            if seg_end <= seg_start:
                continue
            local = point_value[seg_start:seg_end].float().clamp(min=0.0)
            if int(local.numel()) <= 0:
                continue
            if str(pool) == "max":
                value = local.max()
            elif str(pool) == "top20_mean":
                keep = min(int(local.numel()), max(1, math.ceil(0.20 * int(local.numel()))))
                value = torch.topk(local, k=keep, largest=True).values.mean()
            elif str(pool) == "mean":
                value = local.mean()
            else:
                raise ValueError(f"Unsupported segment pool: {pool!r}")
            segment_values.append(value)
            segment_slices.append((seg_start, seg_end))
    if not segment_values:
        return out
    values = torch.stack(segment_values)
    max_value = values.max().clamp(min=1e-6)
    normalized = (values / max_value).clamp(0.0, 1.0)
    for value, (seg_start, seg_end) in zip(normalized, segment_slices, strict=True):
        out[seg_start:seg_end] = value
    return out


def _ship_query_pair_fractional_segment_targets(
    *,
    query_hit_masks: list[torch.Tensor],
    boundaries: list[tuple[int, int]],
    segment_size: int,
    point_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Return segment scores with one fractional credit per ship-query pair."""
    out = torch.zeros((int(point_count),), dtype=torch.float32, device=device)
    segment_values: list[float] = []
    segment_slices: list[tuple[int, int, int]] = []
    size = max(1, int(segment_size))
    for trajectory_id, (start, end) in enumerate(boundaries):
        for seg_start in range(int(start), int(end), size):
            seg_end = min(int(end), seg_start + size)
            if seg_end <= seg_start:
                continue
            segment_slices.append((int(trajectory_id), int(seg_start), int(seg_end)))
            segment_values.append(0.0)
    if not segment_slices:
        return out

    segments_by_trajectory: dict[int, list[int]] = {}
    for segment_idx, (trajectory_id, _seg_start, _seg_end) in enumerate(segment_slices):
        segments_by_trajectory.setdefault(int(trajectory_id), []).append(int(segment_idx))

    for query_mask in query_hit_masks:
        query_hit = query_mask.to(device=device, dtype=torch.bool)
        for _trajectory_id, segment_indices in segments_by_trajectory.items():
            hit_segment_indices = [
                segment_idx
                for segment_idx in segment_indices
                if bool(
                    query_hit[segment_slices[segment_idx][1] : segment_slices[segment_idx][2]]
                    .any()
                    .item()
                )
            ]
            if not hit_segment_indices:
                continue
            credit = 1.0 / float(len(hit_segment_indices))
            for segment_idx in hit_segment_indices:
                segment_values[segment_idx] += credit

    if max(segment_values, default=0.0) <= 1e-12:
        return out
    max_value = max(segment_values)
    for value, (_trajectory_id, seg_start, seg_end) in zip(
        segment_values,
        segment_slices,
        strict=True,
    ):
        out[int(seg_start) : int(seg_end)] = float(value / max_value)
    return out


def _lat_lon_distance_km(
    points: torch.Tensor, left_idx: torch.Tensor, right_idx: torch.Tensor
) -> torch.Tensor:
    """Return approximate lat/lon distance in km for local index pairs."""
    left = points[left_idx.long()]
    right = points[right_idx.long()]
    lat1 = left[:, 1].float()
    lon1 = left[:, 2].float()
    lat2 = right[:, 1].float()
    lon2 = right[:, 2].float()
    return local_equirectangular_distance_km(lat1, lon1, lat2, lon2)


def _path_length_support_targets(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    segment_size: int,
    highpass_quantile: float = 0.50,
) -> torch.Tensor:
    """Assign each point its segment's normalized query-free path-length support."""
    n_points = int(points.shape[0])
    raw = torch.zeros((n_points,), dtype=torch.float32, device=points.device)
    if n_points <= 0 or int(points.shape[1]) < 3:
        return raw

    for start, end in boundaries:
        start_i = int(start)
        end_i = int(end)
        count = int(end_i - start_i)
        if count < 3:
            continue
        local = points[start_i:end_i]
        mid_idx = torch.arange(1, count - 1, dtype=torch.long, device=points.device)
        prev_idx = mid_idx - 1
        next_idx = mid_idx + 1
        via_mid = _lat_lon_distance_km(local, prev_idx, mid_idx) + _lat_lon_distance_km(
            local, mid_idx, next_idx
        )
        shortcut = _lat_lon_distance_km(local, prev_idx, next_idx)
        raw[start_i + mid_idx] = torch.clamp(via_mid - shortcut, min=0.0)

    out = torch.zeros_like(raw)
    size = max(1, int(segment_size))
    quantile = max(0.0, min(1.0, float(highpass_quantile)))
    for start, end in boundaries:
        segment_masses: list[torch.Tensor] = []
        segment_slices: list[tuple[int, int]] = []
        for seg_start in range(int(start), int(end), size):
            seg_end = min(int(end), seg_start + size)
            if seg_end <= seg_start:
                continue
            segment_masses.append(raw[seg_start:seg_end].float().clamp(min=0.0).sum())
            segment_slices.append((seg_start, seg_end))
        if not segment_masses:
            continue
        masses = torch.stack(segment_masses)
        if float(masses.max().item()) <= 1e-12:
            continue
        if int(masses.numel()) == 1:
            normalized = masses / masses.max().clamp(min=1e-6)
        else:
            threshold = torch.quantile(masses, quantile)
            span = (masses.max() - threshold).clamp(min=1e-6)
            normalized = ((masses - threshold) / span).clamp(0.0, 1.0)
        for value, (seg_start, seg_end) in zip(normalized, segment_slices, strict=True):
            out[seg_start:seg_end] = value
    return out


def query_local_utility_path_length_support_targets(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    *,
    segment_size: int = 32,
    highpass_quantile: float = 0.50,
) -> torch.Tensor:
    """Return the query-free path-length support target used by QueryLocalUtility heads."""
    return _path_length_support_targets(
        points,
        boundaries,
        segment_size=max(1, int(segment_size)),
        highpass_quantile=float(highpass_quantile),
    )
