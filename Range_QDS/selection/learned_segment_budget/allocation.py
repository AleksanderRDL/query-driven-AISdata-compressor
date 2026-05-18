"""Segment construction and learned-slot allocation helpers."""

from __future__ import annotations

import math
from typing import Any

import torch

from selection.learned_segment_budget.constants import SEGMENT_ALLOCATION_WEIGHT_FLOOR


def _total_budget(boundaries: list[tuple[int, int]], compression_ratio: float) -> int:
    """Return the comparable per-trajectory total budget."""
    total = 0
    ratio = min(1.0, max(0.0, float(compression_ratio)))
    for start, end in boundaries:
        count = int(end - start)
        if count <= 0:
            continue
        total += min(count, max(2, math.ceil(ratio * count)))
    return total


def _max_skeleton_fraction(compression_ratio: float) -> float:
    """Return guide-recommended maximum skeleton share."""
    ratio = float(compression_ratio)
    if ratio <= 0.01:
        return 0.50
    if ratio <= 0.02:
        return 0.40
    if ratio <= 0.05:
        return 0.25
    if ratio <= 0.10:
        return 0.20
    return 0.15


def _segment_rows(
    scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    segment_size: int,
    segment_scores: torch.Tensor | None = None,
    points: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    """Return candidate segment rows with predicted value."""
    rows: list[dict[str, Any]] = []
    size = max(1, int(segment_size))
    segment_values = scores if segment_scores is None else segment_scores.to(device=scores.device)
    points_cpu = points.detach().cpu().float() if points is not None else None
    for trajectory_id, (start, end) in enumerate(boundaries):
        for seg_start in range(int(start), int(end), size):
            seg_end = min(int(end), seg_start + size)
            if seg_end <= seg_start:
                continue
            if segment_scores is None:
                local_segment = scores[seg_start:seg_end].float()
                top_count = min(
                    int(local_segment.numel()),
                    max(1, math.ceil(0.20 * int(local_segment.numel()))),
                )
                segment_score = float(torch.topk(local_segment, k=top_count).values.mean().item())
                segment_score_source = "point_score_top20_mean"
            else:
                local_segment = segment_values[seg_start:seg_end].float()
                head_top_count = min(
                    int(local_segment.numel()),
                    max(1, math.ceil(0.20 * int(local_segment.numel()))),
                )
                segment_score = float(
                    torch.topk(local_segment, k=head_top_count).values.mean().item()
                )
                segment_score_source = "segment_budget_head_top20_mean"
            length_support_score = (
                _segment_path_length_support(points_cpu[seg_start:seg_end])
                if points_cpu is not None
                else 0.0
            )
            rows.append(
                {
                    "segment_index": len(rows),
                    "trajectory_id": int(trajectory_id),
                    "start": int(seg_start),
                    "end": int(seg_end),
                    "score": segment_score,
                    "score_source": segment_score_source,
                    "length_support_score": float(length_support_score),
                    "length": int(seg_end - seg_start),
                }
            )
    return rows


def _segment_path_length_support(segment_points: torch.Tensor) -> float:
    """Return query-free segment curvature/excess length support."""
    if int(segment_points.shape[0]) < 3 or int(segment_points.shape[-1]) < 3:
        return 0.0
    lats = segment_points[:, 1].float()
    lons = segment_points[:, 2].float()
    lat_mid = torch.deg2rad((lats[1:] + lats[:-1]) * 0.5)
    dy = (lats[1:] - lats[:-1]) * 111.32
    dx = (lons[1:] - lons[:-1]) * 111.32 * torch.clamp(torch.cos(lat_mid).abs(), min=0.10)
    local_path = torch.sqrt(dx * dx + dy * dy).sum()
    endpoint_lat_mid = torch.deg2rad((lats[-1] + lats[0]) * 0.5)
    endpoint_dy = (lats[-1] - lats[0]) * 111.32
    endpoint_dx = (
        (lons[-1] - lons[0]) * 111.32 * torch.clamp(torch.cos(endpoint_lat_mid).abs(), min=0.10)
    )
    shortcut = torch.sqrt(endpoint_dx * endpoint_dx + endpoint_dy * endpoint_dy)
    return float(torch.clamp(local_path - shortcut, min=0.0).item())


def _segment_score_stats(segment_rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Return compact segment score diagnostics."""
    if not segment_rows:
        return {
            "segment_score_count": 0,
            "segment_score_min": 0.0,
            "segment_score_max": 0.0,
            "segment_score_mean": 0.0,
            "segment_score_std": 0.0,
            "segment_score_span": 0.0,
        }
    values = torch.tensor(
        [
            float(row.get("score", 0.0)) if math.isfinite(float(row.get("score", 0.0))) else 0.0
            for row in segment_rows
        ],
        dtype=torch.float32,
    )
    return {
        "segment_score_count": int(values.numel()),
        "segment_score_min": float(values.min().item()),
        "segment_score_max": float(values.max().item()),
        "segment_score_mean": float(values.mean().item()),
        "segment_score_std": float(values.std(unbiased=False).item()),
        "segment_score_span": float((values.max() - values.min()).item()),
    }


def _normalized_row_values(segment_rows: list[dict[str, Any]], key: str) -> list[float]:
    """Return normalized finite row values for one numeric row key."""
    raw_values = [
        float(row.get(key, 0.0)) if math.isfinite(float(row.get(key, 0.0))) else 0.0
        for row in segment_rows
    ]
    if not raw_values:
        return []
    min_value = min(raw_values)
    max_value = max(raw_values)
    span = max_value - min_value
    if span <= 1e-12:
        return [0.0 for _row in segment_rows]
    return [(value - min_value) / span for value in raw_values]


def _segment_allocation_weights(
    segment_rows: list[dict[str, Any]],
    *,
    segment_length_support_weight: float = 0.0,
    segment_allocation_weight_floor: float = SEGMENT_ALLOCATION_WEIGHT_FLOOR,
) -> list[float]:
    """Return positive row weights; equal scores degrade to uniform allocation."""
    if not segment_rows:
        return []
    score_values = _normalized_row_values(segment_rows, "score")
    score_has_signal = max(score_values, default=0.0) > 1e-12
    support_weight = max(0.0, min(1.0, float(segment_length_support_weight)))
    support_has_signal = False
    if support_weight > 0.0:
        support_values = _normalized_row_values(segment_rows, "length_support_score")
        support_has_signal = max(support_values, default=0.0) > 1e-12
        if support_has_signal and score_has_signal:
            score_values = [
                (1.0 - support_weight) * score + support_weight * support
                for score, support in zip(score_values, support_values, strict=True)
            ]
        elif support_has_signal:
            score_values = [support_weight * support for support in support_values]
    if not score_has_signal and not support_has_signal:
        return [1.0 for _row in segment_rows]
    weight_floor = max(0.0, float(segment_allocation_weight_floor))
    return [weight_floor + value for value in score_values]


def _allocate_segment_budgets(
    *,
    segment_rows: list[dict[str, Any]],
    retained: torch.Tensor,
    remaining: int,
    budget: int,
    boundaries: list[tuple[int, int]],
    max_budget_share_per_trajectory: float,
    fairness_preallocation_enabled: bool = True,
    segment_length_support_weight: float = 0.0,
    segment_allocation_weight_floor: float = SEGMENT_ALLOCATION_WEIGHT_FLOOR,
) -> dict[int, int]:
    """Allocate learned slots with score-weighted diminishing returns."""
    if remaining <= 0 or not segment_rows:
        return {}
    valid_trajectory_count = sum(1 for start, end in boundaries if int(end - start) > 0)
    share_cap = math.ceil(
        float(budget) * max(0.01, min(1.0, float(max_budget_share_per_trajectory)))
    )
    fair_share_cap = math.ceil(float(budget) / float(max(1, valid_trajectory_count)))
    max_per_trajectory = max(1, share_cap, fair_share_cap)
    trajectory_allocations = {
        idx: int(retained[start:end].sum().item()) for idx, (start, end) in enumerate(boundaries)
    }
    segment_allocations: dict[int, int] = {}
    weights = _segment_allocation_weights(
        segment_rows,
        segment_length_support_weight=segment_length_support_weight,
        segment_allocation_weight_floor=segment_allocation_weight_floor,
    )
    for row, weight in zip(segment_rows, weights, strict=True):
        row["allocation_weight"] = float(weight)
    remaining_slots = int(remaining)

    # Trajectories with enough total learned budget should not be reduced to
    # endpoints-only retention. This is query-free sanity structure, so expose
    # it as a switch and as a diagnostic ablation rather than hiding it.
    if fairness_preallocation_enabled and remaining_slots >= max(1, valid_trajectory_count):
        trajectory_best_rows: dict[int, tuple[float, float, int, int]] = {}
        for segment_idx, row in enumerate(segment_rows):
            trajectory_id = int(row["trajectory_id"])
            start = int(row["start"])
            score = float(row["score"])
            allocation_weight = float(weights[segment_idx])
            best = trajectory_best_rows.get(trajectory_id)
            if (
                best is None
                or allocation_weight > best[0]
                or (allocation_weight == best[0] and score > best[1])
                or (allocation_weight == best[0] and score == best[1] and start < best[2])
            ):
                trajectory_best_rows[trajectory_id] = (
                    allocation_weight,
                    score,
                    start,
                    segment_idx,
                )

        for _, _score, _start, segment_idx in sorted(
            trajectory_best_rows.values(),
            key=lambda item: (float(item[0]), float(item[1]), -int(item[2])),
            reverse=True,
        ):
            if remaining_slots <= 0:
                break
            row = segment_rows[segment_idx]
            trajectory_id = int(row["trajectory_id"])
            if trajectory_allocations.get(trajectory_id, 0) >= max_per_trajectory:
                continue
            start = int(row["start"])
            end = int(row["end"])
            capacity = (
                int(row["length"])
                - int(retained[start:end].sum().item())
                - int(segment_allocations.get(segment_idx, 0))
            )
            if capacity <= 0:
                continue
            segment_allocations[segment_idx] = int(segment_allocations.get(segment_idx, 0)) + 1
            trajectory_allocations[trajectory_id] = (
                int(trajectory_allocations.get(trajectory_id, 0)) + 1
            )
            remaining_slots -= 1

    if remaining_slots <= 0:
        return segment_allocations

    while remaining_slots > 0:
        best_idx: int | None = None
        best_key: tuple[float, int, float, int] | None = None
        for segment_idx, row in enumerate(segment_rows):
            trajectory_id = int(row["trajectory_id"])
            if trajectory_allocations.get(trajectory_id, 0) >= max_per_trajectory:
                continue
            current = int(segment_allocations.get(segment_idx, 0))
            start = int(row["start"])
            end = int(row["end"])
            capacity = int(row["length"]) - int(retained[start:end].sum().item()) - current
            if capacity <= 0:
                continue
            weight = max(1e-6, float(weights[segment_idx]))
            priority = math.log(weight) - math.log(float(current + 1))
            key = (priority, -current, float(row["score"]), -start)
            if best_key is None or key > best_key:
                best_key = key
                best_idx = segment_idx
        if best_idx is None:
            break
        row = segment_rows[best_idx]
        trajectory_id = int(row["trajectory_id"])
        segment_allocations[best_idx] = int(segment_allocations.get(best_idx, 0)) + 1
        trajectory_allocations[trajectory_id] = (
            int(trajectory_allocations.get(trajectory_id, 0)) + 1
        )
        remaining_slots -= 1
    return segment_allocations
