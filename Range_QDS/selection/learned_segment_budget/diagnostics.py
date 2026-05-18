"""Query-free selector diagnostics for learned segment-budget traces."""

from __future__ import annotations

import math
from typing import Any

import torch

from selection.learned_segment_budget.allocation import _allocate_segment_budgets
from selection.learned_segment_budget.length_repair import (
    _fill_missing_by_length_gain,
    _select_with_spacing,
)


def _entropy(counts: list[int]) -> tuple[float, float]:
    """Return raw and normalized Shannon entropy for positive counts."""
    positive = [int(count) for count in counts if int(count) > 0]
    total = sum(positive)
    if total <= 0:
        return 0.0, 0.0
    entropy = 0.0
    for count in positive:
        probability = float(count) / float(total)
        entropy -= probability * math.log(probability)
    if len(positive) <= 1:
        return float(entropy), 0.0
    return float(entropy), float(entropy / math.log(float(len(positive))))


def _trajectory_counts(mask: torch.Tensor, boundaries: list[tuple[int, int]]) -> list[int]:
    """Count selected points per trajectory."""
    return [int(mask[int(start) : int(end)].sum().item()) for start, end in boundaries]


def _mask_indices_payload(mask: torch.Tensor | None) -> dict[str, Any]:
    """Return a compact JSON payload for a diagnostic retained mask."""
    if mask is None:
        return {
            "available": False,
            "diagnostic_only": True,
            "query_free": True,
            "reason": "not_run",
            "retained_count": None,
            "indices": [],
        }
    mask_bool = mask.detach().cpu().bool().flatten()
    indices = [int(idx) for idx in torch.where(mask_bool)[0].tolist()]
    return {
        "available": True,
        "diagnostic_only": True,
        "query_free": True,
        "retained_count": len(indices),
        "indices": indices,
    }


def _descending_ranks(values: list[float]) -> list[int]:
    """Return 1-based descending ranks for deterministic diagnostic ordering."""
    order = sorted(range(len(values)), key=lambda idx: (float(values[idx]), -idx), reverse=True)
    ranks = [0 for _value in values]
    for rank, idx in enumerate(order, start=1):
        ranks[int(idx)] = int(rank)
    return ranks


def _segment_source_attribution(
    *,
    segment_rows: list[dict[str, Any]],
    segment_allocations: dict[int, int],
    retained: torch.Tensor,
    skeleton_mask: torch.Tensor,
    learned_mask: torch.Tensor,
    fallback_mask: torch.Tensor,
    length_repair_mask: torch.Tensor,
) -> dict[str, Any]:
    """Return query-free retained-source attribution per selector segment."""
    if not segment_rows:
        return {"available": False, "reason": "segment_rows_missing"}
    point_count = int(retained.numel())
    masks = {
        "retained": retained.detach().cpu().bool(),
        "skeleton": skeleton_mask.detach().cpu().bool(),
        "learned": learned_mask.detach().cpu().bool(),
        "fallback": fallback_mask.detach().cpu().bool(),
        "length_repair": length_repair_mask.detach().cpu().bool(),
    }
    if any(int(mask.numel()) != point_count for mask in masks.values()):
        return {"available": False, "reason": "mask_shape_mismatch"}
    attributed = masks["skeleton"] | masks["learned"] | masks["fallback"] | masks["length_repair"]
    score_values = [
        float(row.get("score", 0.0)) if math.isfinite(float(row.get("score", 0.0))) else 0.0
        for row in segment_rows
    ]
    score_ranks = _descending_ranks(score_values)
    length_support_values = [
        float(row.get("length_support_score", 0.0))
        if math.isfinite(float(row.get("length_support_score", 0.0)))
        else 0.0
        for row in segment_rows
    ]
    length_support_ranks = _descending_ranks(length_support_values)
    allocation_weight_values = [
        float(row.get("allocation_weight", 0.0))
        if math.isfinite(float(row.get("allocation_weight", 0.0)))
        else 0.0
        for row in segment_rows
    ]
    allocation_weight_ranks = _descending_ranks(allocation_weight_values)
    rows: list[dict[str, Any]] = []
    summary_counts = {
        "retained": 0,
        "skeleton": 0,
        "learned": 0,
        "fallback": 0,
        "length_repair": 0,
        "unattributed": 0,
        "allocation": 0,
    }
    segment_presence = {
        "retained": 0,
        "skeleton": 0,
        "learned": 0,
        "fallback": 0,
        "length_repair": 0,
        "allocation": 0,
    }
    for allocation_order_index, row in enumerate(segment_rows):
        start = int(row["start"])
        end = int(row["end"])
        if start < 0 or end > point_count or end <= start:
            continue
        canonical_segment_index = int(row.get("segment_index", allocation_order_index))
        retained_count = int(masks["retained"][start:end].sum().item())
        skeleton_count = int(masks["skeleton"][start:end].sum().item())
        learned_count = int(masks["learned"][start:end].sum().item())
        fallback_count = int(masks["fallback"][start:end].sum().item())
        length_repair_count = int(masks["length_repair"][start:end].sum().item())
        unattributed_count = int(
            (masks["retained"][start:end] & ~attributed[start:end]).sum().item()
        )
        allocation_count = int(segment_allocations.get(allocation_order_index, 0))
        summary_counts["retained"] += retained_count
        summary_counts["skeleton"] += skeleton_count
        summary_counts["learned"] += learned_count
        summary_counts["fallback"] += fallback_count
        summary_counts["length_repair"] += length_repair_count
        summary_counts["unattributed"] += unattributed_count
        summary_counts["allocation"] += allocation_count
        if retained_count > 0:
            segment_presence["retained"] += 1
        if skeleton_count > 0:
            segment_presence["skeleton"] += 1
        if learned_count > 0:
            segment_presence["learned"] += 1
        if fallback_count > 0:
            segment_presence["fallback"] += 1
        if length_repair_count > 0:
            segment_presence["length_repair"] += 1
        if allocation_count > 0:
            segment_presence["allocation"] += 1
        rows.append(
            {
                "segment_index": canonical_segment_index,
                "allocation_order_index": int(allocation_order_index),
                "trajectory_id": int(row["trajectory_id"]),
                "start": start,
                "end": end,
                "length": int(end - start),
                "segment_score": float(score_values[allocation_order_index]),
                "segment_score_rank": int(score_ranks[allocation_order_index]),
                "segment_score_source": str(row.get("score_source", "")),
                "segment_length_support_score": float(
                    length_support_values[allocation_order_index]
                ),
                "segment_length_support_rank": int(length_support_ranks[allocation_order_index]),
                "segment_allocation_weight": float(
                    allocation_weight_values[allocation_order_index]
                ),
                "segment_allocation_weight_rank": int(
                    allocation_weight_ranks[allocation_order_index]
                ),
                "segment_allocation_count": allocation_count,
                "retained_count": retained_count,
                "retained_fraction": float(retained_count / max(1, end - start)),
                "skeleton_count": skeleton_count,
                "learned_count": learned_count,
                "fallback_count": fallback_count,
                "length_repair_count": length_repair_count,
                "unattributed_count": unattributed_count,
            }
        )
    return {
        "available": True,
        "diagnostic_only": True,
        "query_free": True,
        "description": "Per-segment retained-point attribution by selector source before eval-query scoring.",
        "segment_count": len(rows),
        "summary": {
            "retained_count_total": int(summary_counts["retained"]),
            "skeleton_count_total": int(summary_counts["skeleton"]),
            "learned_count_total": int(summary_counts["learned"]),
            "fallback_count_total": int(summary_counts["fallback"]),
            "length_repair_count_total": int(summary_counts["length_repair"]),
            "unattributed_count_total": int(summary_counts["unattributed"]),
            "segment_allocation_count_total": int(summary_counts["allocation"]),
            "segments_with_retained": int(segment_presence["retained"]),
            "segments_with_skeleton": int(segment_presence["skeleton"]),
            "segments_with_learned": int(segment_presence["learned"]),
            "segments_with_fallback": int(segment_presence["fallback"]),
            "segments_with_length_repair": int(segment_presence["length_repair"]),
            "segments_with_allocation": int(segment_presence["allocation"]),
        },
        "rows": rows,
    }


def _safe_finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except TypeError, ValueError:
        return float(default)
    return number if math.isfinite(number) else float(default)


def _pearson_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / float(len(left))
    right_mean = sum(right) / float(len(right))
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    if left_var <= 1e-12 or right_var <= 1e-12:
        return None
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    return float(covariance / math.sqrt(left_var * right_var))


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: float(values[idx]))
    ranks = [0.0 for _value in values]
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        average_rank = float(index + end + 2) / 2.0
        for rank_index in range(index, end + 1):
            ranks[order[rank_index]] = average_rank
        index = end + 1
    return ranks


def _spearman_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return _pearson_correlation(_average_ranks(left), _average_ranks(right))


def _allocation_alignment_group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "segment_count": 0,
            "allocation_count_total": 0,
            "extra_allocation_count_total": 0,
            "average_allocation_count": None,
            "average_extra_allocation_count": None,
            "average_length_support_score": None,
            "average_segment_score": None,
            "average_allocation_weight": None,
        }
    segment_count = len(rows)
    allocation_total = sum(int(row["allocation_count"]) for row in rows)
    extra_total = sum(int(row["extra_allocation_count"]) for row in rows)
    return {
        "segment_count": int(segment_count),
        "allocation_count_total": int(allocation_total),
        "extra_allocation_count_total": int(extra_total),
        "average_allocation_count": float(allocation_total / float(segment_count)),
        "average_extra_allocation_count": float(extra_total / float(segment_count)),
        "average_length_support_score": float(
            sum(float(row["length_support_score"]) for row in rows) / float(segment_count)
        ),
        "average_segment_score": float(
            sum(float(row["segment_score"]) for row in rows) / float(segment_count)
        ),
        "average_allocation_weight": float(
            sum(float(row["allocation_weight"]) for row in rows) / float(segment_count)
        ),
    }


def _histogram(values: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        counts[key] = int(counts.get(key, 0)) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _segment_allocation_alignment_diagnostics(
    *,
    segment_rows: list[dict[str, Any]],
    segment_allocations: dict[int, int],
) -> dict[str, Any]:
    """Report whether learned extra slots follow score or query-free length support."""
    if not segment_rows:
        return {"available": False, "reason": "segment_rows_missing"}

    rows: list[dict[str, Any]] = []
    for allocation_order_index, row in enumerate(segment_rows):
        allocation_count = max(0, int(segment_allocations.get(allocation_order_index, 0)))
        rows.append(
            {
                "allocation_order_index": int(allocation_order_index),
                "trajectory_id": int(row.get("trajectory_id", -1)),
                "allocation_count": int(allocation_count),
                "extra_allocation_count": int(max(0, allocation_count - 1)),
                "length_support_score": _safe_finite_float(row.get("length_support_score", 0.0)),
                "segment_score": _safe_finite_float(row.get("score", 0.0)),
                "allocation_weight": _safe_finite_float(row.get("allocation_weight", 0.0)),
            }
        )

    allocation_counts = [float(row["allocation_count"]) for row in rows]
    extra_counts = [float(row["extra_allocation_count"]) for row in rows]
    length_support_scores = [float(row["length_support_score"]) for row in rows]
    segment_scores = [float(row["segment_score"]) for row in rows]
    allocation_weights = [float(row["allocation_weight"]) for row in rows]
    total_extra = int(sum(int(row["extra_allocation_count"]) for row in rows))
    segment_count = len(rows)

    top_groups: dict[str, Any] = {}
    for fraction in (0.10, 0.20, 0.30):
        top_count = max(1, math.floor(float(segment_count) * float(fraction)))
        top_length = sorted(rows, key=lambda row: float(row["length_support_score"]), reverse=True)[
            :top_count
        ]
        top_score = sorted(rows, key=lambda row: float(row["segment_score"]), reverse=True)[
            :top_count
        ]
        top_weight = sorted(rows, key=lambda row: float(row["allocation_weight"]), reverse=True)[
            :top_count
        ]
        length_ids = {int(row["allocation_order_index"]) for row in top_length}
        score_ids = {int(row["allocation_order_index"]) for row in top_score}
        top_groups[f"top_{round(fraction * 100.0)}_percent"] = {
            "by_length_support": _allocation_alignment_group_stats(top_length),
            "by_segment_score": _allocation_alignment_group_stats(top_score),
            "by_allocation_weight": _allocation_alignment_group_stats(top_weight),
            "length_support_segment_score_overlap_count": len(length_ids & score_ids),
            "length_support_segment_score_overlap_fraction": float(
                len(length_ids & score_ids) / float(top_count)
            ),
        }

    length_deciles: list[dict[str, Any]] = []
    by_length = sorted(rows, key=lambda row: float(row["length_support_score"]))
    for decile_index in range(10):
        start = decile_index * segment_count // 10
        end = (decile_index + 1) * segment_count // 10
        stats = _allocation_alignment_group_stats(by_length[start:end])
        stats["decile"] = int(decile_index + 1)
        length_deciles.append(stats)

    by_trajectory: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_trajectory.setdefault(int(row["trajectory_id"]), []).append(row)
    top3_length_extra_counts: list[int] = []
    top4_length_extra_counts: list[int] = []
    top3_score_overlap_counts: list[int] = []
    for trajectory_rows in by_trajectory.values():
        length_order = sorted(
            trajectory_rows,
            key=lambda row: float(row["length_support_score"]),
            reverse=True,
        )
        score_order = sorted(
            trajectory_rows, key=lambda row: float(row["segment_score"]), reverse=True
        )
        top3_length_ids = {
            int(row["allocation_order_index"]) for row in length_order[: min(3, len(length_order))]
        }
        top4_length_ids = {
            int(row["allocation_order_index"]) for row in length_order[: min(4, len(length_order))]
        }
        top3_score_ids = {
            int(row["allocation_order_index"]) for row in score_order[: min(3, len(score_order))]
        }
        top3_length_extra_counts.append(
            sum(
                int(row["extra_allocation_count"])
                for row in trajectory_rows
                if int(row["allocation_order_index"]) in top3_length_ids
            )
        )
        top4_length_extra_counts.append(
            sum(
                int(row["extra_allocation_count"])
                for row in trajectory_rows
                if int(row["allocation_order_index"]) in top4_length_ids
            )
        )
        top3_score_overlap_counts.append(len(top3_length_ids & top3_score_ids))

    length_allocation_correlation = _pearson_correlation(length_support_scores, allocation_counts)
    score_allocation_correlation = _pearson_correlation(segment_scores, allocation_counts)
    allocation_weight_correlation = _pearson_correlation(allocation_weights, allocation_counts)
    length_extra_correlation = _pearson_correlation(length_support_scores, extra_counts)
    diagnosis = "allocation_alignment_inconclusive"
    if (
        length_allocation_correlation is not None
        and score_allocation_correlation is not None
        and length_allocation_correlation < 0.10
        and score_allocation_correlation >= 0.50
    ):
        diagnosis = "extra_slots_score_dominated_not_length_support_aligned"
    elif length_allocation_correlation is not None and length_allocation_correlation >= 0.25:
        diagnosis = "length_support_materially_influences_allocation"

    return {
        "available": True,
        "diagnostic_only": True,
        "query_free": True,
        "description": (
            "Allocation alignment between query-free length support, learned segment score, "
            "allocation weight, and actual learned slot counts."
        ),
        "segment_count": int(segment_count),
        "allocation_count_total": int(sum(int(row["allocation_count"]) for row in rows)),
        "extra_allocation_count_total": int(total_extra),
        "allocation_count_histogram": _histogram([int(row["allocation_count"]) for row in rows]),
        "extra_allocation_count_histogram": _histogram(
            [int(row["extra_allocation_count"]) for row in rows]
        ),
        "length_support_to_allocation_pearson": length_allocation_correlation,
        "length_support_to_allocation_spearman": _spearman_correlation(
            length_support_scores, allocation_counts
        ),
        "length_support_to_extra_allocation_pearson": length_extra_correlation,
        "segment_score_to_allocation_pearson": score_allocation_correlation,
        "segment_score_to_allocation_spearman": _spearman_correlation(
            segment_scores, allocation_counts
        ),
        "allocation_weight_to_allocation_pearson": allocation_weight_correlation,
        "allocation_weight_to_allocation_spearman": _spearman_correlation(
            allocation_weights, allocation_counts
        ),
        "top_groups": top_groups,
        "length_support_deciles": length_deciles,
        "trajectory_top_length_support_extra_capture": {
            "trajectory_count": len(by_trajectory),
            "average_top3_length_support_extra_count": (
                float(sum(top3_length_extra_counts) / float(len(top3_length_extra_counts)))
                if top3_length_extra_counts
                else None
            ),
            "average_top4_length_support_extra_count": (
                float(sum(top4_length_extra_counts) / float(len(top4_length_extra_counts)))
                if top4_length_extra_counts
                else None
            ),
            "top3_length_support_extra_count_histogram": _histogram(top3_length_extra_counts),
            "top3_length_support_score_overlap_count_histogram": _histogram(
                top3_score_overlap_counts
            ),
        },
        "component_diagnosis": diagnosis,
    }


def _polyline_length_km(points: torch.Tensor) -> float:
    """Return approximate retained polyline length in km for lat/lon points."""
    if int(points.shape[0]) < 2:
        return 0.0
    lats = points[:, 1].float()
    lons = points[:, 2].float()
    lat_rad = torch.deg2rad(lats)
    lon_rad = torch.deg2rad(lons)
    dlat = lat_rad[1:] - lat_rad[:-1]
    dlon = lon_rad[1:] - lon_rad[:-1]
    a = (
        torch.sin(dlat / 2.0) ** 2
        + torch.cos(lat_rad[:-1]) * torch.cos(lat_rad[1:]) * torch.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt(torch.clamp(1.0 - a, min=1e-9)))
    return float((6371.0 * c).sum().item())


def _quantile(values: list[float], q: float) -> float | None:
    """Return a deterministic linear quantile for small diagnostic lists."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, float(q))) * float(len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - float(lower)
    return float((1.0 - fraction) * ordered[lower] + fraction * ordered[upper])


def _retained_index_gap_stats(mask: torch.Tensor) -> tuple[int, float]:
    """Return max and mean retained-index gap for one trajectory mask."""
    indices = torch.where(mask.detach().cpu().bool())[0]
    if int(indices.numel()) < 2:
        return 0, 0.0
    gaps = (indices[1:] - indices[:-1]).float()
    return int(gaps.max().item()), float(gaps.mean().item())


def _selector_geometry_diagnostics(
    *,
    points: torch.Tensor | None,
    retained: torch.Tensor,
    skeleton_mask: torch.Tensor,
    learned_mask: torch.Tensor,
    fallback_mask: torch.Tensor,
    length_repair_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
) -> dict[str, Any]:
    """Return query-free path-length diagnostics for the selector trace."""
    if points is None or int(points.shape[0]) != int(retained.numel()) or int(points.shape[-1]) < 3:
        return {"available": False, "reason": "missing_or_mismatched_points"}

    points_cpu = points.detach().cpu().float()
    retained_cpu = retained.detach().cpu().bool()
    skeleton_cpu = skeleton_mask.detach().cpu().bool()
    learned_cpu = learned_mask.detach().cpu().bool()
    fallback_cpu = fallback_mask.detach().cpu().bool()
    repair_cpu = length_repair_mask.detach().cpu().bool()
    skeleton_plus_learned_cpu = skeleton_cpu | learned_cpu

    total_original_km = 0.0
    total_retained_km = 0.0
    total_skeleton_km = 0.0
    total_skeleton_plus_learned_km = 0.0
    total_fallback_km = 0.0
    trajectory_rows: list[dict[str, float | int]] = []
    preservation_values: list[float] = []
    max_gap_values: list[float] = []
    mean_gap_values: list[float] = []

    for trajectory_id, (start, end) in enumerate(boundaries):
        start_i = int(start)
        end_i = int(end)
        if end_i - start_i < 2:
            continue
        local_points = points_cpu[start_i:end_i]
        original_km = _polyline_length_km(local_points)
        if original_km <= 1e-9:
            continue
        local_retained = retained_cpu[start_i:end_i]
        local_skeleton = skeleton_cpu[start_i:end_i]
        local_skeleton_plus_learned = skeleton_plus_learned_cpu[start_i:end_i]
        local_fallback = fallback_cpu[start_i:end_i]

        retained_km = (
            _polyline_length_km(local_points[local_retained])
            if int(local_retained.sum().item()) >= 2
            else 0.0
        )
        skeleton_km = (
            _polyline_length_km(local_points[local_skeleton])
            if int(local_skeleton.sum().item()) >= 2
            else 0.0
        )
        skeleton_plus_learned_km = (
            _polyline_length_km(local_points[local_skeleton_plus_learned])
            if int(local_skeleton_plus_learned.sum().item()) >= 2
            else 0.0
        )
        fallback_with_skeleton = local_skeleton | local_fallback
        fallback_km = (
            _polyline_length_km(local_points[fallback_with_skeleton])
            if int(fallback_with_skeleton.sum().item()) >= 2
            else 0.0
        )
        max_gap, mean_gap = _retained_index_gap_stats(local_retained)
        preservation = float(max(0.0, min(1.0, retained_km / original_km)))

        total_original_km += original_km
        total_retained_km += retained_km
        total_skeleton_km += skeleton_km
        total_skeleton_plus_learned_km += skeleton_plus_learned_km
        total_fallback_km += fallback_km
        preservation_values.append(preservation)
        max_gap_values.append(float(max_gap))
        mean_gap_values.append(float(mean_gap))
        trajectory_rows.append(
            {
                "trajectory_id": int(trajectory_id),
                "original_length_km": float(original_km),
                "retained_length_km": float(retained_km),
                "retained_length_preservation": preservation,
                "retained_count": int(local_retained.sum().item()),
                "learned_count": int(learned_cpu[start_i:end_i].sum().item()),
                "length_repair_count": int(repair_cpu[start_i:end_i].sum().item()),
                "max_retained_index_gap": int(max_gap),
                "mean_retained_index_gap": float(mean_gap),
            }
        )

    if total_original_km <= 1e-9:
        return {"available": False, "reason": "zero_original_length"}

    worst = sorted(
        trajectory_rows,
        key=lambda row: (
            float(row["retained_length_preservation"]),
            -float(row["original_length_km"]),
        ),
    )[:5]
    retained_preservation = float(total_retained_km / total_original_km)
    skeleton_preservation = float(total_skeleton_km / total_original_km)
    skeleton_plus_learned_preservation = float(total_skeleton_plus_learned_km / total_original_km)
    return {
        "available": True,
        "trajectory_count": len(trajectory_rows),
        "total_original_length_km": float(total_original_km),
        "retained_length_km": float(total_retained_km),
        "retained_length_preservation": retained_preservation,
        "skeleton_length_preservation": skeleton_preservation,
        "skeleton_plus_learned_length_preservation": skeleton_plus_learned_preservation,
        "fallback_with_skeleton_length_preservation": float(total_fallback_km / total_original_km),
        "learned_length_gain_over_skeleton": float(
            skeleton_plus_learned_preservation - skeleton_preservation
        ),
        "fallback_length_gain_over_skeleton": float(
            total_fallback_km / total_original_km - skeleton_preservation
        ),
        "trajectory_length_preservation_min": _quantile(preservation_values, 0.0),
        "trajectory_length_preservation_p10": _quantile(preservation_values, 0.10),
        "trajectory_length_preservation_p50": _quantile(preservation_values, 0.50),
        "trajectory_length_preservation_p90": _quantile(preservation_values, 0.90),
        "trajectory_length_preservation_below_0_8_count": int(
            sum(value < 0.80 for value in preservation_values)
        ),
        "trajectory_length_preservation_below_0_8_fraction": float(
            sum(value < 0.80 for value in preservation_values) / max(1, len(preservation_values))
        ),
        "trajectory_length_preservation_below_0_5_count": int(
            sum(value < 0.50 for value in preservation_values)
        ),
        "trajectory_max_retained_index_gap_p50": _quantile(max_gap_values, 0.50),
        "trajectory_max_retained_index_gap_p90": _quantile(max_gap_values, 0.90),
        "trajectory_max_retained_index_gap_max": _quantile(max_gap_values, 1.0),
        "trajectory_mean_retained_index_gap_p50": _quantile(mean_gap_values, 0.50),
        "worst_trajectories": worst,
    }


def _mask_length_preservation(
    *,
    points: torch.Tensor | None,
    mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
) -> float | None:
    """Return aggregate retained/original path-length ratio for one mask."""
    if points is None or int(points.shape[0]) != int(mask.numel()) or int(points.shape[-1]) < 3:
        return None
    points_cpu = points.detach().cpu().float()
    mask_cpu = mask.detach().cpu().bool()
    original_km = 0.0
    retained_km = 0.0
    for start, end in boundaries:
        start_i = int(start)
        end_i = int(end)
        if end_i - start_i < 2:
            continue
        local_points = points_cpu[start_i:end_i]
        local_original = _polyline_length_km(local_points)
        if local_original <= 1e-9:
            continue
        local_mask = mask_cpu[start_i:end_i]
        local_retained = (
            _polyline_length_km(local_points[local_mask])
            if int(local_mask.sum().item()) >= 2
            else 0.0
        )
        original_km += local_original
        retained_km += local_retained
    if original_km <= 1e-9:
        return None
    return float(retained_km / original_km)


def _length_only_mask_for_segment_allocations(
    *,
    scores: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    segment_rows: list[dict[str, Any]],
    segment_allocations: dict[int, int],
    skeleton_retained: torch.Tensor,
    budget: int,
    min_temporal_spacing_fraction_within_segment: float,
) -> tuple[torch.Tensor, int]:
    retained = skeleton_retained.detach().cpu().bool().clone()
    scores_cpu = scores.detach().cpu().float()
    points_cpu = points.detach().cpu().float()
    for segment_idx, keep_count in segment_allocations.items():
        if int(keep_count) <= 0 or int(segment_idx) < 0 or int(segment_idx) >= len(segment_rows):
            continue
        row = segment_rows[int(segment_idx)]
        start = int(row["start"])
        end = int(row["end"])
        trajectory_id = int(row["trajectory_id"])
        trajectory_start, trajectory_end = boundaries[trajectory_id]
        trajectory_start = int(trajectory_start)
        trajectory_end = int(trajectory_end)
        trajectory_scores = torch.full(
            (max(0, trajectory_end - trajectory_start),),
            -float("inf"),
            dtype=torch.float32,
        )
        segment_local_start = max(0, start - trajectory_start)
        segment_local_end = min(int(trajectory_scores.numel()), end - trajectory_start)
        if segment_local_end <= segment_local_start:
            continue
        trajectory_scores[segment_local_start:segment_local_end] = scores_cpu[start:end]
        existing = torch.where(retained[trajectory_start:trajectory_end])[0]
        min_spacing = math.floor(
            float(end - start) * float(min_temporal_spacing_fraction_within_segment)
        )
        selected = _select_with_spacing(
            trajectory_scores,
            int(keep_count),
            trajectory_id=trajectory_id,
            existing_indices=existing,
            min_spacing=min_spacing,
            local_points=points_cpu[trajectory_start:trajectory_end],
            geometry_gain_weight=1.0,
        )
        retained[trajectory_start + selected] = True

    length_fill_count = _fill_missing_by_length_gain(
        retained=retained,
        points=points_cpu,
        boundaries=boundaries,
        budget=int(budget),
    )
    return retained, int(length_fill_count)


def _allocation_counterfactual_diagnostics(
    *,
    scores: torch.Tensor,
    points: torch.Tensor | None,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
    segment_rows: list[dict[str, Any]],
    segment_allocations: dict[int, int],
    skeleton_retained: torch.Tensor,
    budget: int,
    max_budget_share_per_trajectory: float,
    fairness_preallocation_enabled: bool,
    segment_allocation_weight_floor: float,
    min_temporal_spacing_fraction_within_segment: float,
) -> dict[str, Any]:
    """Test whether query-free length-support allocation can clear length."""
    if points is None or int(points.shape[0]) != int(scores.numel()) or not segment_rows:
        return {"available": False, "reason": "missing_points_or_segment_rows"}
    primary_length = _mask_length_preservation(
        points=points, mask=skeleton_retained, boundaries=boundaries
    )
    if primary_length is None:
        return {"available": False, "reason": "length_preservation_unavailable"}

    remaining = max(0, int(budget) - int(skeleton_retained.sum().item()))
    if remaining <= 0:
        return {"available": False, "reason": "no_learned_allocation_budget"}

    counterfactual_rows = [dict(row) for row in segment_rows]
    length_support_allocations = _allocate_segment_budgets(
        segment_rows=counterfactual_rows,
        retained=skeleton_retained.detach().cpu().bool(),
        remaining=remaining,
        budget=int(budget),
        boundaries=boundaries,
        max_budget_share_per_trajectory=float(max_budget_share_per_trajectory),
        fairness_preallocation_enabled=bool(fairness_preallocation_enabled),
        segment_length_support_weight=1.0,
        segment_allocation_weight_floor=float(segment_allocation_weight_floor),
    )
    counterfactual_retained, length_fill_count = _length_only_mask_for_segment_allocations(
        scores=scores,
        points=points,
        boundaries=boundaries,
        segment_rows=counterfactual_rows,
        segment_allocations=length_support_allocations,
        skeleton_retained=skeleton_retained,
        budget=int(budget),
        min_temporal_spacing_fraction_within_segment=(min_temporal_spacing_fraction_within_segment),
    )
    counterfactual_length = _mask_length_preservation(
        points=points, mask=counterfactual_retained, boundaries=boundaries
    )
    if counterfactual_length is None:
        return {"available": False, "reason": "counterfactual_length_unavailable"}

    current_allocation_total = sum(int(value) for value in segment_allocations.values())
    counterfactual_allocation_total = sum(
        int(value) for value in length_support_allocations.values()
    )
    allocation_overlap = sum(
        min(
            int(segment_allocations.get(segment_idx, 0)),
            int(length_support_allocations.get(segment_idx, 0)),
        )
        for segment_idx in set(segment_allocations) | set(length_support_allocations)
    )
    current_extra_total = sum(max(0, int(value) - 1) for value in segment_allocations.values())
    counterfactual_extra_total = sum(
        max(0, int(value) - 1) for value in length_support_allocations.values()
    )
    extra_overlap = sum(
        min(
            max(0, int(segment_allocations.get(segment_idx, 0)) - 1),
            max(0, int(length_support_allocations.get(segment_idx, 0)) - 1),
        )
        for segment_idx in set(segment_allocations) | set(length_support_allocations)
    )
    gate_target = 0.80
    return {
        "available": True,
        "diagnostic_only": True,
        "query_free": True,
        "description": (
            "Counterfactual learned-slot allocation using query-free geometric "
            "length support with length-only point choice."
        ),
        "compression_ratio": float(compression_ratio),
        "total_budget_count": int(budget),
        "skeleton_retained_count": int(skeleton_retained.sum().item()),
        "current_allocation_count_total": int(current_allocation_total),
        "counterfactual_allocation_count_total": int(counterfactual_allocation_total),
        "current_extra_allocation_count_total": int(current_extra_total),
        "counterfactual_extra_allocation_count_total": int(counterfactual_extra_total),
        "allocation_overlap_count": int(allocation_overlap),
        "allocation_overlap_fraction": float(allocation_overlap / max(1, current_allocation_total)),
        "extra_allocation_overlap_count": int(extra_overlap),
        "extra_allocation_overlap_fraction": float(extra_overlap / max(1, current_extra_total)),
        "counterfactual_length_fill_count": int(length_fill_count),
        "counterfactual_retained_count": int(counterfactual_retained.sum().item()),
        "counterfactual_under_budget_count": max(
            0, int(budget) - int(counterfactual_retained.sum().item())
        ),
        "length_gate_target": float(gate_target),
        "length_support_allocation_counterfactual_preservation": float(counterfactual_length),
        "length_support_allocation_counterfactual_gate_would_pass": bool(
            counterfactual_length >= gate_target
        ),
        "component_diagnosis": (
            "length_support_allocation_counterfactual_can_clear_length"
            if counterfactual_length >= gate_target
            else "length_support_allocation_counterfactual_cannot_clear_length"
        ),
    }


def _allocation_point_selection_diagnostics(
    *,
    scores: torch.Tensor,
    points: torch.Tensor | None,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
    segment_rows: list[dict[str, Any]],
    segment_allocations: dict[int, int],
    skeleton_retained: torch.Tensor,
    primary_retained: torch.Tensor,
    budget: int,
    min_temporal_spacing_fraction_within_segment: float,
) -> dict[str, Any]:
    """Separate segment-allocation length capacity from within-segment point choice."""
    if points is None or int(points.shape[0]) != int(scores.numel()) or not segment_allocations:
        return {"available": False, "reason": "missing_points_or_allocations"}

    primary_length = _mask_length_preservation(
        points=points, mask=primary_retained, boundaries=boundaries
    )
    if primary_length is None:
        return {"available": False, "reason": "length_preservation_unavailable"}

    length_only_retained, length_fill_count = _length_only_mask_for_segment_allocations(
        scores=scores,
        points=points,
        boundaries=boundaries,
        segment_rows=segment_rows,
        segment_allocations=segment_allocations,
        skeleton_retained=skeleton_retained,
        budget=int(budget),
        min_temporal_spacing_fraction_within_segment=(min_temporal_spacing_fraction_within_segment),
    )
    length_only_preservation = _mask_length_preservation(
        points=points,
        mask=length_only_retained,
        boundaries=boundaries,
    )
    if length_only_preservation is None:
        return {"available": False, "reason": "counterfactual_length_unavailable"}
    retained_count = int(length_only_retained.sum().item())
    gate_target = 0.80
    return {
        "available": True,
        "diagnostic_only": True,
        "description": "Same learned segment allocations, but length-only point choice inside those allocations.",
        "compression_ratio": float(compression_ratio),
        "total_budget_count": int(budget),
        "primary_retained_stage": "pre_length_repair",
        "primary_retained_count": int(primary_retained.sum().item()),
        "counterfactual_retained_count": retained_count,
        "counterfactual_under_budget_count": max(0, int(budget) - retained_count),
        "counterfactual_length_fill_count": int(length_fill_count),
        "primary_length_preservation": float(primary_length),
        "same_allocation_length_only_point_selection_preservation": float(length_only_preservation),
        "same_allocation_length_only_delta": float(length_only_preservation - primary_length),
        "length_gate_target": float(gate_target),
        "same_allocation_length_only_gate_would_pass": bool(
            length_only_preservation >= gate_target
        ),
        "component_diagnosis": (
            "point_selection_can_clear_length_with_current_allocation"
            if length_only_preservation >= gate_target
            else "current_segment_allocation_cannot_clear_length_even_with_length_only_point_selection"
        ),
    }
