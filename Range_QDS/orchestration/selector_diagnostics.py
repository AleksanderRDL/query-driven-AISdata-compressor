"""Selector diagnostic helpers used by run orchestration."""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from orchestration.segment_audits import segment_top_mean
from scoring.method_scoring import _endpoint_sanity, score_range_usefulness
from scoring.methods import FrozenMaskMethod
from scoring.metrics import compute_geometric_distortion, compute_length_preservation
from scoring.query_cache import ScoringQueryCache
from scoring.query_useful_v1 import query_useful_v1_from_range_audit
from selection.learned_segment_budget import (
    GEOMETRY_TIE_BREAKER_WEIGHT,
    SEGMENT_ALLOCATION_WEIGHT_FLOOR,
    SEGMENT_LENGTH_SUPPORT_ALLOCATION_WEIGHT,
    SEGMENT_SCORE_POINT_BLEND_WEIGHT,
    blend_segment_support_scores,
    simplify_with_learned_segment_budget_v1,
)


def learned_segment_frozen_method(
    *,
    name: str,
    scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
    segment_scores: torch.Tensor | None = None,
    segment_point_scores: torch.Tensor | None = None,
    path_length_support_scores: torch.Tensor | None = None,
    points: torch.Tensor | None = None,
    learned_segment_geometry_gain_weight: float = GEOMETRY_TIE_BREAKER_WEIGHT,
    learned_segment_allocation_length_support_weight: float = (
        SEGMENT_LENGTH_SUPPORT_ALLOCATION_WEIGHT
    ),
    learned_segment_allocation_weight_floor: float = SEGMENT_ALLOCATION_WEIGHT_FLOOR,
    learned_segment_score_blend_weight: float = SEGMENT_SCORE_POINT_BLEND_WEIGHT,
    learned_segment_fairness_preallocation: bool = True,
    learned_segment_length_repair_fraction: float = 0.0,
    learned_segment_length_repair_score_protection_fraction: float = 0.0,
    learned_segment_length_support_blend_weight: float = 0.0,
) -> FrozenMaskMethod:
    """Freeze a score-based learned-segment diagnostic mask before query scoring."""
    selector_segment_scores = blend_segment_support_scores(
        segment_scores=segment_scores,
        path_length_support_scores=path_length_support_scores,
        path_length_support_weight=float(learned_segment_length_support_blend_weight),
    )
    selector_segment_point_scores = (
        None if segment_point_scores is None else segment_point_scores.detach().cpu().float()
    )
    retained_mask = simplify_with_learned_segment_budget_v1(
        scores.detach().cpu().float(),
        boundaries,
        compression_ratio,
        segment_scores=selector_segment_scores,
        segment_point_scores=selector_segment_point_scores,
        points=None if points is None else points.detach().cpu().float(),
        geometry_gain_weight=float(learned_segment_geometry_gain_weight),
        segment_length_support_weight=float(learned_segment_allocation_length_support_weight),
        segment_allocation_weight_floor=float(learned_segment_allocation_weight_floor),
        segment_score_point_blend_weight=float(learned_segment_score_blend_weight),
        fairness_preallocation_enabled=bool(learned_segment_fairness_preallocation),
        length_repair_fraction=float(learned_segment_length_repair_fraction),
        length_repair_score_protection_fraction=float(
            learned_segment_length_repair_score_protection_fraction
        ),
    )
    return FrozenMaskMethod(name=name, retained_mask=retained_mask.detach().cpu())


def pre_repair_frozen_method_from_trace(
    *,
    name: str,
    selector_trace: dict[str, Any],
    point_count: int,
) -> FrozenMaskMethod:
    """Build a frozen diagnostic method from trace-persisted pre-repair retained indices."""
    payload = selector_trace.get("pre_repair_retained_mask")
    if not isinstance(payload, dict) or not bool(payload.get("available", False)):
        reason = (
            payload.get("reason", "missing_pre_repair_retained_mask")
            if isinstance(payload, dict)
            else "missing_pre_repair_retained_mask"
        )
        raise ValueError(str(reason))
    raw_indices = payload.get("indices")
    if not isinstance(raw_indices, list):
        raise ValueError("pre_repair_retained_mask.indices must be a list")
    retained_mask = torch.zeros((int(point_count),), dtype=torch.bool)
    seen: set[int] = set()
    for raw_idx in raw_indices:
        if isinstance(raw_idx, bool):
            raise ValueError("pre_repair_retained_mask.indices must contain integer indices")
        idx = int(raw_idx)
        if idx < 0 or idx >= int(point_count):
            raise ValueError(f"pre_repair_retained_mask index out of bounds: {idx}")
        if idx in seen:
            raise ValueError(f"pre_repair_retained_mask duplicate index: {idx}")
        seen.add(idx)
        retained_mask[idx] = True
    declared_count = payload.get("retained_count")
    if declared_count is not None and int(declared_count) != int(retained_mask.sum().item()):
        raise ValueError(
            "pre_repair_retained_mask retained_count mismatch: "
            f"declared={int(declared_count)} actual={int(retained_mask.sum().item())}"
        )
    return FrozenMaskMethod(name=name, retained_mask=retained_mask)


def selector_segment_score_source_label(
    *,
    segment_scores: torch.Tensor | None,
    path_length_support_scores: torch.Tensor | None,
    length_support_blend_weight: float,
) -> str:
    """Return an honest selector trace label for segment allocation scores."""
    weight = max(0.0, min(1.0, float(length_support_blend_weight)))
    if path_length_support_scores is not None and weight >= 1.0 - 1e-12:
        return "path_length_support_head_mean"
    if path_length_support_scores is not None and weight > 0.0:
        return "segment_budget_path_length_support_blend_mean"
    if segment_scores is not None:
        return "segment_budget_head_mean"
    return "point_score_top20_mean"


def neutral_segment_scores_for_ablation(segment_scores: torch.Tensor) -> torch.Tensor:
    """Return neutral segment scores for the no-segment-budget-head ablation."""
    return torch.zeros_like(segment_scores.detach().cpu().float())


def segment_score_top_band_for_ablation(
    segment_scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    *,
    segment_size: int = 32,
    top_fraction: float,
) -> torch.Tensor:
    """Return binary segment scores that keep only a top score band authoritative."""
    scores = segment_scores.detach().cpu().float().flatten()
    out = torch.zeros_like(scores)
    segment_rows: list[tuple[float, int, int, int]] = []
    size = max(1, int(segment_size))
    for start, end in boundaries:
        for seg_start in range(int(start), int(end), size):
            seg_end = min(int(end), int(seg_start) + size)
            if seg_end <= seg_start:
                continue
            segment_rows.append(
                (
                    segment_top_mean(scores, seg_start, seg_end),
                    -int(seg_start),
                    int(seg_start),
                    int(seg_end),
                )
            )
    if not segment_rows:
        return out.reshape(segment_scores.detach().cpu().shape)
    fraction = max(0.0, min(1.0, float(top_fraction)))
    keep_count = max(1, min(len(segment_rows), math.ceil(fraction * len(segment_rows))))
    for _score, _neg_start, seg_start, seg_end in sorted(segment_rows, reverse=True)[:keep_count]:
        out[seg_start:seg_end] = 1.0
    return out.reshape(segment_scores.detach().cpu().shape)


def segment_score_quantile_bands_for_ablation(
    segment_scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    *,
    segment_size: int = 32,
    band_count: int,
) -> torch.Tensor:
    """Return segment scores collapsed into coarse rank bands."""
    scores = segment_scores.detach().cpu().float().flatten()
    out = torch.zeros_like(scores)
    segment_rows: list[tuple[float, int, int, int]] = []
    size = max(1, int(segment_size))
    for start, end in boundaries:
        for seg_start in range(int(start), int(end), size):
            seg_end = min(int(end), int(seg_start) + size)
            if seg_end <= seg_start:
                continue
            segment_rows.append(
                (
                    segment_top_mean(scores, seg_start, seg_end),
                    -int(seg_start),
                    int(seg_start),
                    int(seg_end),
                )
            )
    if not segment_rows:
        return out.reshape(segment_scores.detach().cpu().shape)
    bands = max(1, int(band_count))
    ordered = sorted(segment_rows, reverse=True)
    total = len(ordered)
    for rank_index, (_score, _neg_start, seg_start, seg_end) in enumerate(ordered):
        band = (bands - 1) - min(
            bands - 1,
            math.floor(float(rank_index * bands) / float(total)),
        )
        out[seg_start:seg_end] = float(band)
    return out.reshape(segment_scores.detach().cpu().shape)


def _diagnostic_mask_from_indices(
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
        mask = _diagnostic_mask_from_indices(
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


def _score_vector_or_none(values: torch.Tensor | None, point_count: int) -> torch.Tensor | None:
    if values is None:
        return None
    vector = values.detach().cpu().float()
    if vector.ndim == 2 and int(vector.shape[1]) == 1:
        vector = vector[:, 0]
    elif vector.ndim != 1:
        return None
    if int(vector.numel()) != int(point_count):
        return None
    return vector


def _bounded_candidate_indices(
    *,
    mask: torch.Tensor,
    ranking_scores: torch.Tensor | None,
    limit: int,
) -> list[int]:
    indices = torch.where(mask.detach().cpu().bool())[0]
    if int(indices.numel()) <= 0:
        return []
    max_count = max(0, int(limit))
    if max_count <= 0:
        return []
    if int(indices.numel()) <= max_count:
        return [int(idx) for idx in indices.tolist()]
    if ranking_scores is None:
        positions = torch.linspace(0, int(indices.numel()) - 1, steps=max_count).round().long()
        return [int(idx) for idx in indices[positions].unique(sorted=True).tolist()]

    scores = ranking_scores[indices].float()
    high_count = max(1, max_count // 2)
    low_count = max(0, max_count - high_count)
    high_local = torch.topk(scores, k=min(high_count, int(scores.numel())), largest=True).indices
    if low_count > 0:
        low_local = torch.topk(scores, k=min(low_count, int(scores.numel())), largest=False).indices
        selected = torch.cat([indices[high_local], indices[low_local]])
    else:
        selected = indices[high_local]
    return [int(idx) for idx in selected.unique(sorted=True).tolist()]


def _query_useful_v1_score_for_mask(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    retained_mask: torch.Tensor,
    query_cache: ScoringQueryCache | None,
) -> float:
    range_audit = score_range_usefulness(
        points=points,
        boundaries=boundaries,
        retained_mask=retained_mask,
        typed_queries=typed_queries,
        query_cache=query_cache,
    )
    geometric = compute_geometric_distortion(points, boundaries, retained_mask)
    length = compute_length_preservation(points, boundaries, retained_mask)
    endpoint_sanity = _endpoint_sanity(retained_mask.detach().cpu().bool(), boundaries)
    query_useful = query_useful_v1_from_range_audit(
        range_audit,
        length_preservation=length,
        avg_sed_km=float(geometric.get("avg_sed_km", 0.0)),
        endpoint_sanity=endpoint_sanity,
    )
    score = query_useful.get("query_useful_v1_score", 0.0)
    return float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0


def _pearson(left: list[float], right: list[float]) -> float | None:
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
    cursor = 0
    while cursor < len(order):
        end = cursor
        while end + 1 < len(order) and values[order[end + 1]] == values[order[cursor]]:
            end += 1
        average_rank = float(cursor + end + 2) / 2.0
        for rank_index in range(cursor, end + 1):
            ranks[order[rank_index]] = average_rank
        cursor = end + 1
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return _pearson(_average_ranks(left), _average_ranks(right))


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / float(len(values))) if values else None


def _score_alignment_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"candidate_count": len(rows)}
    if not rows:
        return summary
    marginals = [float(row["marginal_query_useful_v1"]) for row in rows]
    summary.update(
        {
            "mean_marginal_query_useful_v1": _mean(marginals),
            "positive_marginal_fraction": float(
                sum(1 for value in marginals if value > 0.0) / float(len(marginals))
            ),
            "max_marginal_query_useful_v1": max(marginals),
            "min_marginal_query_useful_v1": min(marginals),
        }
    )
    for score_key in ("raw_score", "selector_score", "segment_score"):
        valid = [
            (float(row[score_key]), float(row["marginal_query_useful_v1"]))
            for row in rows
            if row.get(score_key) is not None
        ]
        if len(valid) < 2:
            summary[score_key] = {"available": False, "reason": "fewer_than_two_values"}
            continue
        scores = [score for score, _marginal in valid]
        marginal_values = [marginal for _score, marginal in valid]
        order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        bucket_count = max(1, len(order) // 4)
        top = [marginal_values[idx] for idx in order[:bucket_count]]
        bottom = [marginal_values[idx] for idx in order[-bucket_count:]]
        top_mean = _mean(top)
        bottom_mean = _mean(bottom)
        summary[score_key] = {
            "available": True,
            "pearson": _pearson(scores, marginal_values),
            "spearman": _spearman(scores, marginal_values),
            "top_quartile_mean_marginal": top_mean,
            "bottom_quartile_mean_marginal": bottom_mean,
            "top_minus_bottom_marginal": (
                None if top_mean is None or bottom_mean is None else float(top_mean - bottom_mean)
            ),
        }
    return summary


def _group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "")), []).append(row)
    return grouped


def retained_decision_marginal_query_useful_diagnostics(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    primary_retained_mask: torch.Tensor,
    raw_scores: torch.Tensor | None = None,
    selector_scores: torch.Tensor | None = None,
    segment_scores: torch.Tensor | None = None,
    source_masks: dict[str, torch.Tensor] | None = None,
    selector_trace: dict[str, Any] | None = None,
    query_cache: ScoringQueryCache | None = None,
    max_retained_per_source: int = 64,
    max_removed_candidates: int = 128,
) -> dict[str, Any]:
    """Score bounded point-level QueryUsefulV1 marginals after masks are frozen.

    Retained candidates use leave-one-out loss. Removed candidates use add-one
    gain. The diagnostic is bounded and diagnostic-only; it is not a training
    target and must not affect retained-mask construction.
    """
    point_count = int(primary_retained_mask.numel())
    primary_mask = primary_retained_mask.detach().cpu().bool().flatten()
    if int(points.shape[0]) != point_count:
        raise ValueError(
            "primary_retained_mask must match points: "
            f"mask={point_count} points={int(points.shape[0])}"
        )
    if int(primary_mask.sum().item()) <= 0:
        return {
            "available": False,
            "diagnostic_only": True,
            "reason": "empty_primary_retained_mask",
        }

    source_mask_map: dict[str, torch.Tensor] = {}
    if source_masks is not None:
        source_mask_map.update(
            {
                str(name): mask.detach().cpu().bool().flatten()
                for name, mask in source_masks.items()
                if int(mask.numel()) == point_count
            }
        )
    if selector_trace is not None:
        source_mask_map.update(
            source_masks_from_selector_trace(selector_trace, point_count=point_count)
        )
    attributed = torch.zeros_like(primary_mask)
    for mask in source_mask_map.values():
        attributed |= mask & primary_mask
    unattributed = primary_mask & ~attributed
    if bool(unattributed.any().item()):
        source_mask_map["unattributed"] = unattributed
    if not source_mask_map:
        source_mask_map["retained"] = primary_mask

    raw_vector = _score_vector_or_none(raw_scores, point_count)
    selector_vector = _score_vector_or_none(selector_scores, point_count)
    segment_vector = _score_vector_or_none(segment_scores, point_count)
    ranking_vector = selector_vector if selector_vector is not None else raw_vector
    if ranking_vector is None:
        ranking_vector = segment_vector

    started_at = time.perf_counter()
    effective_query_cache = query_cache
    query_cache_created = False
    if effective_query_cache is None:
        effective_query_cache = ScoringQueryCache.for_workload(points, boundaries, typed_queries)
        query_cache_created = True

    primary_score = _query_useful_v1_score_for_mask(
        points=points,
        boundaries=boundaries,
        typed_queries=typed_queries,
        retained_mask=primary_mask,
        query_cache=effective_query_cache,
    )

    rows: list[dict[str, Any]] = []

    def _row(
        *,
        index: int,
        source: str,
        decision: str,
        marginal: float,
    ) -> dict[str, Any]:
        return {
            "point_index": int(index),
            "source": source,
            "decision": decision,
            "marginal_query_useful_v1": float(marginal),
            "raw_score": None if raw_vector is None else float(raw_vector[int(index)].item()),
            "selector_score": (
                None if selector_vector is None else float(selector_vector[int(index)].item())
            ),
            "segment_score": (
                None if segment_vector is None else float(segment_vector[int(index)].item())
            ),
        }

    for source, source_mask in sorted(source_mask_map.items()):
        candidate_mask = source_mask & primary_mask
        for index in _bounded_candidate_indices(
            mask=candidate_mask,
            ranking_scores=ranking_vector,
            limit=max_retained_per_source,
        ):
            candidate = primary_mask.clone()
            candidate[int(index)] = False
            score = _query_useful_v1_score_for_mask(
                points=points,
                boundaries=boundaries,
                typed_queries=typed_queries,
                retained_mask=candidate,
                query_cache=effective_query_cache,
            )
            rows.append(
                _row(
                    index=index,
                    source=source,
                    decision="retained_removal_loss",
                    marginal=float(primary_score - score),
                )
            )

    removed_mask = ~primary_mask
    for index in _bounded_candidate_indices(
        mask=removed_mask,
        ranking_scores=ranking_vector,
        limit=max_removed_candidates,
    ):
        candidate = primary_mask.clone()
        candidate[int(index)] = True
        score = _query_useful_v1_score_for_mask(
            points=points,
            boundaries=boundaries,
            typed_queries=typed_queries,
            retained_mask=candidate,
            query_cache=effective_query_cache,
        )
        rows.append(
            _row(
                index=index,
                source="removed",
                decision="removed_addition_gain",
                marginal=float(score - primary_score),
            )
        )

    by_source = {
        name: _score_alignment_summary(group_rows)
        for name, group_rows in sorted(_group_rows(rows, "source").items())
    }
    by_decision = {
        name: _score_alignment_summary(group_rows)
        for name, group_rows in sorted(_group_rows(rows, "decision").items())
    }
    elapsed_seconds = float(time.perf_counter() - started_at)
    return {
        "available": True,
        "diagnostic_only": True,
        "exact_query_useful_v1_marginals": True,
        "performance_mode": "exact_cached_query_support",
        "elapsed_seconds": elapsed_seconds,
        "query_cache_provided": query_cache is not None,
        "query_cache_created": query_cache_created,
        "query_cache_support_mask_count": int(
            len(effective_query_cache.support_masks) if effective_query_cache is not None else 0
        ),
        "query_cache_range_audit_support_count": int(
            len(effective_query_cache.range_audit_supports)
            if effective_query_cache is not None
            else 0
        ),
        "query_cache_range_segment_geometry_available": bool(
            effective_query_cache is not None
            and effective_query_cache.range_segment_geometry is not None
        ),
        "masks_frozen_before_query_scoring_required": True,
        "description": (
            "Bounded retained-decision marginal QueryUsefulV1 diagnostic. "
            "Retained rows are leave-one-out loss; removed rows are add-one gain."
        ),
        "primary_query_useful_v1": float(primary_score),
        "retained_count": int(primary_mask.sum().item()),
        "point_count": point_count,
        "max_retained_per_source": int(max_retained_per_source),
        "max_removed_candidates": int(max_removed_candidates),
        "score_fields_available": {
            "raw_score": raw_vector is not None,
            "selector_score": selector_vector is not None,
            "segment_score": segment_vector is not None,
        },
        "candidate_count": len(rows),
        "overall": _score_alignment_summary(rows),
        "by_source": by_source,
        "by_decision": by_decision,
        "rows": sorted(
            rows,
            key=lambda row: (
                str(row["decision"]),
                str(row["source"]),
                -float(row["marginal_query_useful_v1"]),
                int(row["point_index"]),
            ),
        ),
    }
