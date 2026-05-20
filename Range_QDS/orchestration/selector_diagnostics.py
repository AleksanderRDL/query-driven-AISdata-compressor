"""Selector diagnostic helpers used by run orchestration."""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from learning.model_features import transform_workload_blind_range_prior_features
from learning.query_prior_fields import QUERY_PRIOR_FIELD_NAMES, sample_query_prior_fields
from learning.targets.query_local_utility import (
    QUERY_LOCAL_UTILITY_HEAD_NAMES,
    query_local_utility_path_length_support_targets,
    query_local_utility_point_score,
)
from orchestration import selector_marginal_alignment, selector_trace_payloads
from orchestration.segment_audits import segment_top_mean
from scoring.method_scoring import _endpoint_sanity, score_range_usefulness
from scoring.methods import FrozenMaskMethod
from scoring.metrics import compute_geometric_distortion, compute_length_preservation
from scoring.query_cache import ScoringQueryCache
from scoring.query_local_utility import query_local_utility_from_range_audit
from selection.learned_segment_budget import (
    GEOMETRY_TIE_BREAKER_WEIGHT,
    SEGMENT_ALLOCATION_WEIGHT_FLOOR,
    SEGMENT_LENGTH_SUPPORT_ALLOCATION_WEIGHT,
    SEGMENT_SCORE_POINT_BLEND_WEIGHT,
    SEGMENT_TRANSFER_CALIBRATION_MODE_NONE,
    blend_segment_support_scores,
    simplify_with_learned_segment_budget,
)


def factorized_score_component_vectors_from_logits(
    head_logits: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Return point-level diagnostic score components from frozen factorized heads."""
    if head_logits is None:
        return {}
    logits = head_logits.detach().cpu().float()
    if logits.ndim != 2 or int(logits.shape[1]) < len(QUERY_LOCAL_UTILITY_HEAD_NAMES):
        return {}
    probabilities = torch.sigmoid(logits[:, : len(QUERY_LOCAL_UTILITY_HEAD_NAMES)]).contiguous()
    out = {
        f"head_probability_{head_name}": probabilities[:, head_idx].contiguous()
        for head_idx, head_name in enumerate(QUERY_LOCAL_UTILITY_HEAD_NAMES)
    }
    out.update(
        {
            f"head_logit_{head_name}": logits[:, head_idx].contiguous()
            for head_idx, head_name in enumerate(QUERY_LOCAL_UTILITY_HEAD_NAMES)
        }
    )
    q_hit = probabilities[:, 0].float().clamp(0.0, 1.0)
    behavior = probabilities[:, 1].float().clamp(0.0, 1.0)
    boundary = probabilities[:, 2].float().clamp(0.0, 1.0)
    replacement = probabilities[:, 3].float().clamp(0.0, 1.0)
    behavior_multiplier = 0.5 + behavior
    replacement_multiplier = 0.75 + 0.25 * replacement
    q_behavior_replacement = q_hit * behavior_multiplier * replacement_multiplier
    boundary_bonus = 0.25 * boundary
    composed_score = query_local_utility_point_score(
        q_hit=q_hit,
        behavior=behavior,
        boundary=boundary,
        replacement=replacement,
    )
    out.update(
        {
            "factorized_behavior_multiplier": behavior_multiplier.contiguous(),
            "factorized_replacement_multiplier": replacement_multiplier.contiguous(),
            "factorized_q_behavior_replacement_term": q_behavior_replacement.contiguous(),
            "factorized_boundary_bonus": boundary_bonus.contiguous(),
            "factorized_composed_score": composed_score.contiguous(),
        }
    )
    return out


def query_prior_component_vectors_for_points(
    points: torch.Tensor,
    query_prior_field: dict[str, Any] | None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return sampled and model-facing query-prior vectors for row diagnostics."""
    if not isinstance(query_prior_field, dict):
        return {}, {}
    sampled = sample_query_prior_fields(points, query_prior_field).detach().cpu().float()
    model_prior = transform_workload_blind_range_prior_features(sampled).detach().cpu().float()
    sampled_vectors = {
        str(name): sampled[:, field_idx].contiguous()
        for field_idx, name in enumerate(QUERY_PRIOR_FIELD_NAMES)
        if field_idx < int(sampled.shape[1])
    }
    model_vectors = {
        str(name): model_prior[:, field_idx].contiguous()
        for field_idx, name in enumerate(QUERY_PRIOR_FIELD_NAMES)
        if field_idx < int(model_prior.shape[1])
    }
    return sampled_vectors, model_vectors


def query_free_retained_removal_teacher_proxy_vectors(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    *,
    segment_size: int = 32,
) -> dict[str, torch.Tensor]:
    """Return query-free proxy teachers for retained-removal marginal diagnostics."""
    point_count = int(points.shape[0])
    if point_count <= 0:
        return {}
    points_cpu = points.detach().cpu().float()
    endpoint_support = torch.zeros((point_count,), dtype=torch.float32)
    for start, end in boundaries:
        if int(end) <= int(start):
            continue
        endpoint_support[int(start)] = 1.0
        endpoint_support[int(end) - 1] = 1.0
    path_support = (
        query_local_utility_path_length_support_targets(
            points_cpu,
            boundaries,
            segment_size=max(1, int(segment_size)),
        )
        .detach()
        .cpu()
        .float()
    )
    endpoint_or_path_support = torch.maximum(endpoint_support, path_support)
    return {
        "query_free_endpoint_support": endpoint_support.contiguous(),
        "query_free_path_length_support_target": path_support.contiguous(),
        "query_free_endpoint_or_path_support": endpoint_or_path_support.contiguous(),
    }


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
    learned_segment_transfer_calibration_mode: str = SEGMENT_TRANSFER_CALIBRATION_MODE_NONE,
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
    retained_mask = simplify_with_learned_segment_budget(
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
        segment_transfer_calibration_mode=str(learned_segment_transfer_calibration_mode),
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
        return "path_length_support_head_top20_mean"
    if path_length_support_scores is not None and weight > 0.0:
        return "segment_budget_path_length_support_blend_top20_mean"
    if segment_scores is not None:
        return "segment_budget_head_top20_mean"
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


def _score_component_vectors(
    score_component_vectors: dict[str, torch.Tensor] | None,
    point_count: int,
) -> dict[str, torch.Tensor]:
    if not score_component_vectors:
        return {}
    out: dict[str, torch.Tensor] = {}
    for name, values in sorted(score_component_vectors.items()):
        vector = _score_vector_or_none(values, point_count)
        if vector is not None:
            out[str(name)] = vector
    return out


def _trajectory_index_for_point(index: int, boundaries: list[tuple[int, int]]) -> int | None:
    point_index = int(index)
    for trajectory_idx, (start, end) in enumerate(boundaries):
        if int(start) <= point_index < int(end):
            return int(trajectory_idx)
    return None


def _prefixed_row_values(
    *,
    vectors: dict[str, torch.Tensor],
    prefix: str,
    index: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, vector in vectors.items():
        if not name.startswith(prefix):
            continue
        out[name.removeprefix(prefix)] = float(vector[int(index)].item())
    return out


def _row_score_component_values(
    *,
    vectors: dict[str, torch.Tensor],
    index: int,
) -> dict[str, float]:
    return {
        name: float(vector[int(index)].item())
        for name, vector in vectors.items()
        if not name.startswith("head_probability_") and not name.startswith("head_logit_")
    }


def _rank_rows_desc(
    rows: list[dict[str, Any]],
    *,
    output_name: str,
    value_getter: Any,
) -> None:
    valid: list[tuple[int, float]] = []
    for row_idx, row in enumerate(rows):
        value = value_getter(row)
        if value is None:
            continue
        valid.append((row_idx, float(value)))
    if not valid:
        return
    ordered = sorted(valid, key=lambda item: (-item[1], int(rows[item[0]]["point_index"])))
    count = len(ordered)
    for rank_idx, (row_idx, _value) in enumerate(ordered, start=1):
        rows[row_idx][f"{output_name}_candidate_rank"] = int(rank_idx)
        rows[row_idx][f"{output_name}_candidate_rank_fraction"] = float(rank_idx / count)


def _top_marginal_miss_bucket_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_counts: dict[str, int] = {}
    for row in rows:
        for bucket in row.get("failure_buckets", []):
            bucket_counts[str(bucket)] = bucket_counts.get(str(bucket), 0) + 1
    top_rows = sorted(
        rows,
        key=lambda row: (-float(row["marginal_query_local_utility"]), int(row["point_index"])),
    )[: min(16, len(rows))]
    return {
        "available": bool(rows),
        "diagnostic_only": True,
        "bucket_policy": {
            "top_rank_fraction_max": 0.25,
            "low_rank_fraction_min": 0.75,
            "prior_present_threshold": 0.05,
            "head_positive_threshold": 0.20,
            "heuristic_only": True,
        },
        "bucket_counts": bucket_counts,
        "top_marginal_row_count": len(top_rows),
        "top_marginal_rows": top_rows,
    }


def _attach_top_marginal_miss_diagnostics(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    _rank_rows_desc(
        rows,
        output_name="marginal_query_local_utility",
        value_getter=lambda row: row.get("marginal_query_local_utility"),
    )
    for score_name in ("raw_score", "selector_score", "segment_score"):
        _rank_rows_desc(
            rows,
            output_name=score_name,
            value_getter=lambda row, key=score_name: row.get(key),
        )

    component_names = sorted(
        {str(name) for row in rows for name in (row.get("score_components") or {}).keys()}
    )
    component_rank_by_row: list[dict[str, int]] = [dict() for _row in rows]
    for component_name in component_names:
        valid = [
            (row_idx, float(row["score_components"][component_name]))
            for row_idx, row in enumerate(rows)
            if row.get("score_components") is not None
            and row["score_components"].get(component_name) is not None
        ]
        ordered = sorted(valid, key=lambda item: (-item[1], int(rows[item[0]]["point_index"])))
        for rank_idx, (row_idx, _value) in enumerate(ordered, start=1):
            component_rank_by_row[row_idx][component_name] = int(rank_idx)

    for row_idx, row in enumerate(rows):
        marginal_rank = row.get("marginal_query_local_utility_candidate_rank")
        if marginal_rank is not None and component_rank_by_row[row_idx]:
            row["query_local_utility_component_candidate_ranks"] = component_rank_by_row[row_idx]
            row["query_local_utility_component_minus_marginal_rank"] = {
                name: int(component_rank) - int(marginal_rank)
                for name, component_rank in component_rank_by_row[row_idx].items()
            }

    query_free_proxy_names = sorted(
        {str(name) for row in rows for name in (row.get("query_free_teacher_proxies") or {}).keys()}
    )
    query_free_proxy_rank_by_row: list[dict[str, int]] = [dict() for _row in rows]
    for proxy_name in query_free_proxy_names:
        valid = [
            (row_idx, float(row["query_free_teacher_proxies"][proxy_name]))
            for row_idx, row in enumerate(rows)
            if row.get("query_free_teacher_proxies") is not None
            and row["query_free_teacher_proxies"].get(proxy_name) is not None
        ]
        ordered = sorted(valid, key=lambda item: (-item[1], int(rows[item[0]]["point_index"])))
        for rank_idx, (row_idx, _value) in enumerate(ordered, start=1):
            query_free_proxy_rank_by_row[row_idx][proxy_name] = int(rank_idx)

    for row_idx, row in enumerate(rows):
        marginal_rank = row.get("marginal_query_local_utility_candidate_rank")
        if marginal_rank is not None and query_free_proxy_rank_by_row[row_idx]:
            row["query_free_teacher_proxy_candidate_ranks"] = query_free_proxy_rank_by_row[row_idx]
            row["query_free_teacher_proxy_minus_marginal_rank"] = {
                name: int(proxy_rank) - int(marginal_rank)
                for name, proxy_rank in query_free_proxy_rank_by_row[row_idx].items()
            }

    for row_idx, row in enumerate(rows):
        buckets: list[str] = []
        marginal_fraction = row.get("marginal_query_local_utility_candidate_rank_fraction")
        raw_fraction = row.get("raw_score_candidate_rank_fraction")
        selector_fraction = row.get("selector_score_candidate_rank_fraction")
        segment_fraction = row.get("segment_score_candidate_rank_fraction")
        top_marginal = marginal_fraction is not None and float(marginal_fraction) <= 0.25
        low_any_score = any(
            value is not None and float(value) >= 0.75
            for value in (raw_fraction, selector_fraction, segment_fraction)
        )
        top_any_score = any(
            value is not None and float(value) <= 0.25
            for value in (raw_fraction, selector_fraction, segment_fraction)
        )
        low_marginal = marginal_fraction is not None and float(marginal_fraction) >= 0.75
        sampled_prior_values = list((row.get("sampled_prior_channels") or {}).values())
        model_prior_values = list((row.get("model_prior_channels") or {}).values())
        head_probability_values = list((row.get("head_probabilities") or {}).values())
        max_sampled_prior = max([float(value) for value in sampled_prior_values], default=0.0)
        max_model_prior = max([float(value) for value in model_prior_values], default=0.0)
        max_head_probability = max(
            [float(value) for value in head_probability_values],
            default=0.0,
        )
        if top_marginal and low_any_score:
            buckets.append("high_marginal_under_ranked_by_scores")
        if top_any_score and low_marginal:
            buckets.append("high_score_low_exact_marginal")
        if sampled_prior_values and max_sampled_prior <= 1e-6:
            buckets.append("prior_missing_or_out_of_support")
        if max_sampled_prior >= 0.05 and max_head_probability < 0.20:
            buckets.append("prior_present_but_head_flat")
        if (
            max_head_probability >= 0.20
            and raw_fraction is not None
            and float(raw_fraction) >= 0.75
        ):
            buckets.append("head_positive_but_final_score_suppresses_it")
        if (
            raw_fraction is not None
            and float(raw_fraction) <= 0.25
            and any(
                value is not None and float(value) >= 0.75
                for value in (selector_fraction, segment_fraction)
            )
        ):
            buckets.append("raw_score_good_but_segment_allocation_loses_it")
        stage_state = row.get("selector_stage_state") or {}
        if (
            row.get("source") in {"skeleton", "length_repair", "fallback"}
            or bool(stage_state.get("length_repair_retained", False))
            or (
                bool(stage_state.get("final_retained", False))
                and not bool(stage_state.get("pre_repair_retained", True))
            )
        ):
            buckets.append("length_repair_or_skeleton_overrides_learned_decision")
        if top_marginal and max_model_prior >= 0.05 and max_head_probability < 0.20:
            buckets.append("model_prior_present_but_head_flat")
        top_query_free_proxy = any(
            int(proxy_rank) <= max(1, math.ceil(0.25 * len(rows)))
            for proxy_rank in query_free_proxy_rank_by_row[row_idx].values()
        )
        if top_marginal and top_query_free_proxy and low_any_score:
            buckets.append("query_free_teacher_proxy_high_but_active_score_under_ranked")
        row["failure_buckets"] = sorted(set(buckets))


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


def _query_local_utility_score_for_mask(
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
    query_local_utility = query_local_utility_from_range_audit(
        range_audit,
        length_preservation=length,
        avg_sed_km=float(geometric.get("avg_sed_km", 0.0)),
        endpoint_sanity=endpoint_sanity,
    )
    score = query_local_utility.get("query_local_utility_score", 0.0)
    return float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0


def retained_decision_marginal_query_local_utility_diagnostics(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    primary_retained_mask: torch.Tensor,
    raw_scores: torch.Tensor | None = None,
    selector_scores: torch.Tensor | None = None,
    segment_scores: torch.Tensor | None = None,
    score_component_vectors: dict[str, torch.Tensor] | None = None,
    query_free_teacher_proxy_vectors: dict[str, torch.Tensor] | None = None,
    sampled_prior_vectors: dict[str, torch.Tensor] | None = None,
    model_prior_vectors: dict[str, torch.Tensor] | None = None,
    source_masks: dict[str, torch.Tensor] | None = None,
    selector_trace: dict[str, Any] | None = None,
    query_cache: ScoringQueryCache | None = None,
    max_retained_per_source: int = 64,
    max_removed_candidates: int = 128,
    teacher_usage_split: str = "unknown",
) -> dict[str, Any]:
    """Score bounded point-level QueryLocalUtility marginals after masks are frozen.

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
            selector_trace_payloads.source_masks_from_selector_trace(
                selector_trace, point_count=point_count
            )
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
    component_vectors = _score_component_vectors(score_component_vectors, point_count)
    query_free_proxy_vectors = _score_component_vectors(
        query_free_teacher_proxy_vectors,
        point_count,
    )
    sampled_prior_component_vectors = _score_component_vectors(sampled_prior_vectors, point_count)
    model_prior_component_vectors = _score_component_vectors(model_prior_vectors, point_count)
    trace_mask_state = selector_trace_payloads.trace_mask_state_from_selector_trace(
        selector_trace=selector_trace, point_count=point_count
    )
    selector_segment_context_rows = (
        selector_trace_payloads.selector_segment_context_rows_from_trace(
            selector_trace=selector_trace,
            point_count=point_count,
        )
    )
    ranking_vector = selector_vector if selector_vector is not None else raw_vector
    if ranking_vector is None:
        ranking_vector = segment_vector

    started_at = time.perf_counter()
    effective_query_cache = query_cache
    query_cache_created = False
    if effective_query_cache is None:
        effective_query_cache = ScoringQueryCache.for_workload(points, boundaries, typed_queries)
        query_cache_created = True

    primary_score = _query_local_utility_score_for_mask(
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
            "marginal_query_local_utility": float(marginal),
            "raw_score": None if raw_vector is None else float(raw_vector[int(index)].item()),
            "selector_score": (
                None if selector_vector is None else float(selector_vector[int(index)].item())
            ),
            "segment_score": (
                None if segment_vector is None else float(segment_vector[int(index)].item())
            ),
            "trajectory_index": _trajectory_index_for_point(int(index), boundaries),
            "selector_stage_state": {
                name: bool(mask[int(index)].item()) for name, mask in trace_mask_state.items()
            },
            "selector_segment_context": selector_trace_payloads.selector_segment_context_for_point(
                index=int(index),
                segment_rows=selector_segment_context_rows,
            ),
            "head_probabilities": _prefixed_row_values(
                vectors=component_vectors,
                prefix="head_probability_",
                index=int(index),
            ),
            "head_logits": _prefixed_row_values(
                vectors=component_vectors,
                prefix="head_logit_",
                index=int(index),
            ),
            "sampled_prior_channels": {
                name: float(vector[int(index)].item())
                for name, vector in sampled_prior_component_vectors.items()
            },
            "model_prior_channels": {
                name: float(vector[int(index)].item())
                for name, vector in model_prior_component_vectors.items()
            },
            "query_local_utility_score_components": _row_score_component_values(
                vectors=component_vectors,
                index=int(index),
            ),
            "query_free_teacher_proxies": {
                name: float(vector[int(index)].item())
                for name, vector in query_free_proxy_vectors.items()
            },
            "score_components": {
                name: float(vector[int(index)].item()) for name, vector in component_vectors.items()
            },
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
            score = _query_local_utility_score_for_mask(
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
        score = _query_local_utility_score_for_mask(
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

    _attach_top_marginal_miss_diagnostics(rows)
    top_marginal_miss_diagnostics = _top_marginal_miss_bucket_summary(rows)
    top_marginal_miss_summary = dict(top_marginal_miss_diagnostics)
    top_marginal_miss_summary.pop("top_marginal_rows", None)
    top_marginal_miss_summary["top_marginal_rows_in_selector_trace_only"] = True

    by_source = {
        name: selector_marginal_alignment.score_alignment_summary(group_rows)
        for name, group_rows in sorted(
            selector_marginal_alignment.group_rows_by_field(rows, "source").items()
        )
    }
    by_decision = {
        name: selector_marginal_alignment.score_alignment_summary(group_rows)
        for name, group_rows in sorted(
            selector_marginal_alignment.group_rows_by_field(rows, "decision").items()
        )
    }
    guard_coupling_summary = (
        selector_marginal_alignment.query_free_teacher_proxy_guard_coupling_summary(rows)
    )
    learned_teacher_summary = (
        selector_marginal_alignment.learned_controllable_marginal_teacher_summary(rows)
    )
    separated_teacher_summary = selector_marginal_alignment.separated_marginal_teacher_targets(
        rows,
        teacher_usage_split=teacher_usage_split,
    )
    elapsed_seconds = float(time.perf_counter() - started_at)
    return {
        "available": True,
        "diagnostic_only": True,
        "exact_query_local_utility_marginals": True,
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
            "Bounded retained-decision marginal QueryLocalUtility diagnostic. "
            "Retained rows are leave-one-out loss; removed rows are add-one gain."
        ),
        "primary_query_local_utility": float(primary_score),
        "retained_count": int(primary_mask.sum().item()),
        "point_count": point_count,
        "max_retained_per_source": int(max_retained_per_source),
        "max_removed_candidates": int(max_removed_candidates),
        "score_fields_available": {
            "raw_score": raw_vector is not None,
            "selector_score": selector_vector is not None,
            "segment_score": segment_vector is not None,
        },
        "score_component_fields_available": {
            name: True for name in sorted(component_vectors.keys())
        },
        "context_fields_available": {
            "sampled_prior_channels": {
                name: True for name in sorted(sampled_prior_component_vectors.keys())
            },
            "model_prior_channels": {
                name: True for name in sorted(model_prior_component_vectors.keys())
            },
            "query_free_teacher_proxies": {
                name: True for name in sorted(query_free_proxy_vectors.keys())
            },
            "selector_stage_state": {name: True for name in sorted(trace_mask_state.keys())},
            "selector_segment_context": bool(selector_segment_context_rows),
            "trajectory_index": True,
        },
        "candidate_count": len(rows),
        "overall": selector_marginal_alignment.score_alignment_summary(rows),
        "by_source": by_source,
        "by_decision": by_decision,
        "query_free_teacher_proxy_guard_coupling_summary": guard_coupling_summary,
        "learned_controllable_marginal_teacher_summary": learned_teacher_summary,
        "separated_marginal_teacher_summary": separated_teacher_summary,
        "top_marginal_miss_summary": top_marginal_miss_summary,
        "top_marginal_miss_diagnostics": top_marginal_miss_diagnostics,
        "rows": sorted(
            rows,
            key=lambda row: (
                str(row["decision"]),
                str(row["source"]),
                -float(row["marginal_query_local_utility"]),
                int(row["point_index"]),
            ),
        ),
    }
