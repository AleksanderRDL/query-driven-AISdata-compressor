"""Selector diagnostic helpers used by run orchestration."""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from learning.model_features import transform_workload_blind_range_v2_prior_features
from learning.query_prior_fields import QUERY_PRIOR_FIELD_NAMES, sample_query_prior_fields
from learning.targets.query_useful_v1 import (
    QUERY_USEFUL_V1_HEAD_NAMES,
    query_useful_v1_path_length_support_targets,
    query_useful_v1_point_score,
)
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
    SEGMENT_TRANSFER_CALIBRATION_MODE_NONE,
    blend_segment_support_scores,
    simplify_with_learned_segment_budget_v1,
)


def factorized_score_component_vectors_from_logits(
    head_logits: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Return point-level diagnostic score components from frozen factorized heads."""
    if head_logits is None:
        return {}
    logits = head_logits.detach().cpu().float()
    if logits.ndim != 2 or int(logits.shape[1]) < len(QUERY_USEFUL_V1_HEAD_NAMES):
        return {}
    probabilities = torch.sigmoid(logits[:, : len(QUERY_USEFUL_V1_HEAD_NAMES)]).contiguous()
    out = {
        f"head_probability_{head_name}": probabilities[:, head_idx].contiguous()
        for head_idx, head_name in enumerate(QUERY_USEFUL_V1_HEAD_NAMES)
    }
    out.update(
        {
            f"head_logit_{head_name}": logits[:, head_idx].contiguous()
            for head_idx, head_name in enumerate(QUERY_USEFUL_V1_HEAD_NAMES)
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
    composed_score = query_useful_v1_point_score(
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
    model_prior = transform_workload_blind_range_v2_prior_features(sampled).detach().cpu().float()
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
    path_support = query_useful_v1_path_length_support_targets(
        points_cpu,
        boundaries,
        segment_size=max(1, int(segment_size)),
    ).detach().cpu().float()
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


def _trace_mask_state(
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
            out[state_name] = _diagnostic_mask_from_indices(
                indices=payload.get("indices"),
                point_count=point_count,
                source_name=payload_name,
            )
        except ValueError:
            continue
    return out


def _selector_segment_context_rows(
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


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _selector_segment_context_for_point(
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
                "segment_index": _optional_int(row.get("segment_index")),
                "allocation_order_index": _optional_int(row.get("allocation_order_index")),
                "trajectory_index": _optional_int(row.get("trajectory_id")),
                "segment_start": start,
                "segment_end": end,
                "segment_length": int(segment_length),
                "point_offset_in_segment": offset,
                "point_fraction_in_segment": float(offset / max(1, segment_length - 1)),
                "segment_score": _optional_float(row.get("segment_score")),
                "segment_score_rank": _optional_int(row.get("segment_score_rank")),
                "segment_score_source": str(row.get("segment_score_source", "")),
                "segment_length_support_score": _optional_float(
                    row.get("segment_length_support_score")
                ),
                "segment_length_support_rank": _optional_int(
                    row.get("segment_length_support_rank")
                ),
                "segment_allocation_weight": _optional_float(
                    row.get("segment_allocation_weight")
                ),
                "segment_allocation_weight_rank": _optional_int(
                    row.get("segment_allocation_weight_rank")
                ),
                "segment_allocation_count": _optional_int(row.get("segment_allocation_count")),
                "retained_count": _optional_int(row.get("retained_count")),
                "retained_fraction": _optional_float(row.get("retained_fraction")),
                "skeleton_count": _optional_int(row.get("skeleton_count")),
                "learned_count": _optional_int(row.get("learned_count")),
                "fallback_count": _optional_int(row.get("fallback_count")),
                "length_repair_count": _optional_int(row.get("length_repair_count")),
                "unattributed_count": _optional_int(row.get("unattributed_count")),
            }
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
        key=lambda row: (-float(row["marginal_query_useful_v1"]), int(row["point_index"])),
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
        output_name="marginal_query_useful_v1",
        value_getter=lambda row: row.get("marginal_query_useful_v1"),
    )
    for score_name in ("raw_score", "selector_score", "segment_score"):
        _rank_rows_desc(
            rows,
            output_name=score_name,
            value_getter=lambda row, key=score_name: row.get(key),
        )

    component_names = sorted(
        {
            str(name)
            for row in rows
            for name in (row.get("score_components") or {}).keys()
        }
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
        marginal_rank = row.get("marginal_query_useful_v1_candidate_rank")
        if marginal_rank is not None and component_rank_by_row[row_idx]:
            row["query_useful_v1_component_candidate_ranks"] = component_rank_by_row[row_idx]
            row["query_useful_v1_component_minus_marginal_rank"] = {
                name: int(component_rank) - int(marginal_rank)
                for name, component_rank in component_rank_by_row[row_idx].items()
            }

    query_free_proxy_names = sorted(
        {
            str(name)
            for row in rows
            for name in (row.get("query_free_teacher_proxies") or {}).keys()
        }
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
        marginal_rank = row.get("marginal_query_useful_v1_candidate_rank")
        if marginal_rank is not None and query_free_proxy_rank_by_row[row_idx]:
            row["query_free_teacher_proxy_candidate_ranks"] = query_free_proxy_rank_by_row[row_idx]
            row["query_free_teacher_proxy_minus_marginal_rank"] = {
                name: int(proxy_rank) - int(marginal_rank)
                for name, proxy_rank in query_free_proxy_rank_by_row[row_idx].items()
            }

    for row_idx, row in enumerate(rows):
        buckets: list[str] = []
        marginal_fraction = row.get("marginal_query_useful_v1_candidate_rank_fraction")
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
        if max_head_probability >= 0.20 and raw_fraction is not None and float(raw_fraction) >= 0.75:
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
            or (bool(stage_state.get("final_retained", False)) and not bool(stage_state.get("pre_repair_retained", True)))
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


def _value_marginal_alignment_summary(
    values: list[float],
    marginal_values: list[float],
) -> dict[str, Any]:
    if len(values) != len(marginal_values) or len(values) < 2:
        return {
            "available": False,
            "reason": "fewer_than_two_values",
            "count": min(len(values), len(marginal_values)),
        }
    value_min = min(values)
    value_max = max(values)
    if value_max - value_min <= 1e-12:
        return {
            "available": False,
            "reason": "no_value_variation",
            "count": len(values),
            "value_min": value_min,
            "value_max": value_max,
        }
    order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)
    bucket_count = max(1, len(order) // 4)
    top = [marginal_values[idx] for idx in order[:bucket_count]]
    bottom = [marginal_values[idx] for idx in order[-bucket_count:]]
    top_mean = _mean(top)
    bottom_mean = _mean(bottom)
    return {
        "available": True,
        "count": len(values),
        "pearson": _pearson(values, marginal_values),
        "spearman": _spearman(values, marginal_values),
        "top_quartile_mean_marginal": top_mean,
        "bottom_quartile_mean_marginal": bottom_mean,
        "top_minus_bottom_marginal": (
            None if top_mean is None or bottom_mean is None else float(top_mean - bottom_mean)
        ),
        "value_min": value_min,
        "value_max": value_max,
    }


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
        summary[score_key] = _value_marginal_alignment_summary(
            [score for score, _marginal in valid],
            [marginal for _score, marginal in valid],
        )
    component_summary = _nested_value_alignment_summary(rows, "score_components")
    if component_summary:
        summary["score_component_alignment"] = component_summary
    query_free_proxy_summary = _nested_value_alignment_summary(rows, "query_free_teacher_proxies")
    if query_free_proxy_summary:
        summary["query_free_teacher_proxy_alignment"] = query_free_proxy_summary
    return summary


def _nested_value_alignment_summary(
    rows: list[dict[str, Any]],
    row_field_name: str,
) -> dict[str, Any]:
    value_names = sorted(
        {
            str(name)
            for row in rows
            for name in (row.get(row_field_name) or {}).keys()
        }
    )
    if not value_names:
        return {}
    nested_summary: dict[str, Any] = {}
    for value_name in value_names:
        valid = [
            (
                float(row[row_field_name][value_name]),
                float(row["marginal_query_useful_v1"]),
            )
            for row in rows
            if row.get(row_field_name) is not None
            and row[row_field_name].get(value_name) is not None
        ]
        nested_summary[value_name] = _value_marginal_alignment_summary(
            [score for score, _marginal in valid],
            [marginal for _score, marginal in valid],
        )
    return nested_summary


def _group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "")), []).append(row)
    return grouped


def _guard_owned_retained_row(row: dict[str, Any]) -> bool:
    if str(row.get("decision")) != "retained_removal_loss":
        return False
    source = str(row.get("source", ""))
    stage_state = row.get("selector_stage_state") or {}
    return (
        source in {"skeleton", "length_repair", "fallback"}
        or bool(stage_state.get("skeleton_retained", False))
        or bool(stage_state.get("length_repair_retained", False))
        or bool(stage_state.get("fallback_retained", False))
    )


def _compact_proxy_subset_alignment_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        summary[score_key] = _value_marginal_alignment_summary(
            [score for score, _marginal in valid],
            [marginal for _score, marginal in valid],
        )
    proxy_summary = _nested_value_alignment_summary(rows, "query_free_teacher_proxies")
    if proxy_summary:
        summary["query_free_teacher_proxy_alignment"] = proxy_summary
    return summary


def _query_free_teacher_proxy_guard_coupling_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {
            "available": False,
            "diagnostic_only": True,
            "reason": "no_retained_marginal_rows",
        }

    retained_rows = [row for row in rows if row.get("decision") == "retained_removal_loss"]
    guard_rows = [row for row in retained_rows if _guard_owned_retained_row(row)]
    learned_controllable_rows = [
        row
        for row in retained_rows
        if str(row.get("source")) == "learned" and not _guard_owned_retained_row(row)
    ]
    non_guard_rows = [row for row in retained_rows if not _guard_owned_retained_row(row)]
    removed_rows = [row for row in rows if row.get("decision") == "removed_addition_gain"]

    subsets = {
        "all_retained_removal": retained_rows,
        "learned_controllable_retained_removal": learned_controllable_rows,
        "non_guard_retained_removal": non_guard_rows,
        "guard_owned_retained_removal": guard_rows,
        "skeleton_retained_removal": [
            row for row in retained_rows if str(row.get("source")) == "skeleton"
        ],
        "length_repair_retained_removal": [
            row for row in retained_rows if str(row.get("source")) == "length_repair"
        ],
        "removed_addition_gain": removed_rows,
    }
    subset_summaries = {
        name: _compact_proxy_subset_alignment_summary(subset_rows)
        for name, subset_rows in subsets.items()
    }

    def endpoint_top_minus(subset_name: str) -> float | None:
        summary = subset_summaries.get(subset_name, {})
        proxy_summary = summary.get("query_free_teacher_proxy_alignment")
        if not isinstance(proxy_summary, dict):
            return None
        endpoint_summary = proxy_summary.get("query_free_endpoint_support")
        if not isinstance(endpoint_summary, dict):
            return None
        value = endpoint_summary.get("top_minus_bottom_marginal")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    all_top_minus = endpoint_top_minus("all_retained_removal")
    learned_top_minus = endpoint_top_minus("learned_controllable_retained_removal")
    guard_top_minus = endpoint_top_minus("guard_owned_retained_removal")
    guard_coupling_suspected = (
        all_top_minus is not None
        and all_top_minus > 0.0
        and (
            learned_top_minus is None
            or learned_top_minus <= 0.0
            or (guard_top_minus is not None and guard_top_minus > learned_top_minus)
        )
    )

    return {
        "available": True,
        "diagnostic_only": True,
        "proxy_family": "query_free_teacher_proxies",
        "primary_proxy": "query_free_endpoint_support",
        "retained_removal_count": len(retained_rows),
        "learned_controllable_retained_removal_count": len(learned_controllable_rows),
        "non_guard_retained_removal_count": len(non_guard_rows),
        "guard_owned_retained_removal_count": len(guard_rows),
        "subsets": subset_summaries,
        "endpoint_proxy_guard_coupling": {
            "available": all_top_minus is not None,
            "rule": (
                "Suspect guard coupling when endpoint proxy alignment is positive "
                "overall but absent, nonpositive, or weaker on learned-controllable "
                "retained-removal rows."
            ),
            "all_retained_top_minus_bottom_marginal": all_top_minus,
            "learned_controllable_top_minus_bottom_marginal": learned_top_minus,
            "guard_owned_top_minus_bottom_marginal": guard_top_minus,
            "guard_coupling_suspected": bool(guard_coupling_suspected),
        },
    }


def _learned_controllable_marginal_teacher_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    retained_rows = [row for row in rows if row.get("decision") == "retained_removal_loss"]
    learned_rows = [
        row
        for row in retained_rows
        if str(row.get("source")) == "learned" and not _guard_owned_retained_row(row)
    ]
    summary = _score_alignment_summary(learned_rows)
    marginals = [float(row["marginal_query_useful_v1"]) for row in learned_rows]
    value_variation = (max(marginals) - min(marginals)) if len(marginals) >= 2 else 0.0
    usable_candidate = len(marginals) >= 2 and value_variation > 1e-12
    summary.update(
        {
            "available": bool(learned_rows),
            "diagnostic_only": True,
            "teacher_signal": "exact_retained_removal_marginal_query_useful_v1",
            "teacher_scope": "learned_controllable_retained_removal",
            "decision": "retained_removal_loss",
            "source": "learned",
            "guard_exclusion_policy": (
                "Excludes skeleton, fallback, length-repair, and any retained row whose "
                "selector stage state marks skeleton/fallback/length-repair ownership."
            ),
            "query_conditioned_teacher_requires_train_or_selection_workload": True,
            "eval_time_feature_allowed": False,
            "retained_removal_count": len(retained_rows),
            "learned_controllable_retained_removal_count": len(learned_rows),
            "teacher_value_variation": float(value_variation),
            "candidate_for_train_side_calibration": bool(usable_candidate),
        }
    )
    if not learned_rows:
        summary["reason"] = "no_learned_controllable_retained_removal_rows"
    elif not usable_candidate:
        summary["reason"] = "insufficient_teacher_value_variation"
    return summary


_TRAIN_OR_CHECKPOINT_TEACHER_USAGE_SPLITS = frozenset({"train", "checkpoint_selection"})


def _separated_teacher_candidate_rejection_reason(
    *,
    teacher_usage_split: str,
    teacher_target_shape_viable: bool,
) -> str:
    if not teacher_target_shape_viable:
        return "insufficient_teacher_target_shape"
    if str(teacher_usage_split).startswith("eval"):
        return "eval_split_query_conditioned_teacher_not_allowed_for_training"
    if str(teacher_usage_split) == "unknown":
        return "unknown_split_query_conditioned_teacher_not_allowed_for_training"
    return "split_not_allowed_for_train_or_checkpoint_teacher"


def _separated_marginal_teacher_targets(
    rows: list[dict[str, Any]],
    *,
    teacher_usage_split: str = "unknown",
) -> dict[str, Any]:
    retained_rows = [row for row in rows if row.get("decision") == "retained_removal_loss"]
    learned_rows = [
        row
        for row in retained_rows
        if str(row.get("source")) == "learned" and not _guard_owned_retained_row(row)
    ]
    contextual_rows = [
        row for row in learned_rows if isinstance(row.get("selector_segment_context"), dict)
    ]
    usage_split = str(teacher_usage_split)
    usage_allowed = usage_split in _TRAIN_OR_CHECKPOINT_TEACHER_USAGE_SPLITS
    summary: dict[str, Any] = {
        "available": False,
        "diagnostic_only": True,
        "teacher_signal": "exact_retained_removal_marginal_query_useful_v1",
        "teacher_scope": "learned_controllable_retained_removal",
        "teacher_shape": "separated_segment_and_within_segment_point_targets",
        "teacher_usage_split": usage_split,
        "teacher_usage_allowed_for_train_or_checkpoint": bool(usage_allowed),
        "teacher_target_shape_viable": False,
        "decision": "retained_removal_loss",
        "source": "learned",
        "eval_time_feature_allowed": False,
        "query_conditioned_teacher_requires_train_or_selection_workload": True,
        "guard_exclusion_policy": (
            "Excludes skeleton, fallback, length-repair, and any retained row whose "
            "selector stage state marks skeleton/fallback/length-repair ownership."
        ),
        "retained_removal_count": len(retained_rows),
        "learned_controllable_retained_removal_count": len(learned_rows),
        "rows_with_selector_segment_context": len(contextual_rows),
        "rows_missing_selector_segment_context": max(0, len(learned_rows) - len(contextual_rows)),
        "candidate_for_train_side_teacher": False,
        "candidate_for_train_side_teacher_reason": (
            _separated_teacher_candidate_rejection_reason(
                teacher_usage_split=usage_split,
                teacher_target_shape_viable=False,
            )
        ),
    }
    if not learned_rows:
        summary["reason"] = "no_learned_controllable_retained_removal_rows"
        return summary
    if not contextual_rows:
        summary["reason"] = "missing_selector_segment_context"
        return summary

    grouped: dict[tuple[int | None, int | None, int | None, int | None], list[dict[str, Any]]] = {}
    for row in contextual_rows:
        context = row.get("selector_segment_context") or {}
        key = (
            _optional_int(context.get("trajectory_index")),
            _optional_int(context.get("segment_index")),
            _optional_int(context.get("segment_start")),
            _optional_int(context.get("segment_end")),
        )
        grouped.setdefault(key, []).append(row)

    max_segment_sum = 0.0
    max_point_positive = 0.0
    for group_rows in grouped.values():
        positives = [
            max(0.0, float(row.get("marginal_query_useful_v1", 0.0))) for row in group_rows
        ]
        max_segment_sum = max(max_segment_sum, sum(positives))
        max_point_positive = max(max_point_positive, max(positives, default=0.0))

    segment_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    for (trajectory_idx, segment_idx, segment_start, segment_end), group_rows in grouped.items():
        context = group_rows[0].get("selector_segment_context") or {}
        positive_marginals = [
            max(0.0, float(row.get("marginal_query_useful_v1", 0.0))) for row in group_rows
        ]
        raw_marginals = [float(row.get("marginal_query_useful_v1", 0.0)) for row in group_rows]
        segment_positive_sum = sum(positive_marginals)
        segment_positive_max = max(positive_marginals, default=0.0)
        local_max = max(segment_positive_max, 1e-12)
        ordered_group = sorted(
            group_rows,
            key=lambda row: (
                -max(0.0, float(row.get("marginal_query_useful_v1", 0.0))),
                int(row.get("point_index", -1)),
            ),
        )
        segment_rows.append(
            {
                "trajectory_index": trajectory_idx,
                "segment_index": segment_idx,
                "segment_start": segment_start,
                "segment_end": segment_end,
                "segment_length": _optional_int(context.get("segment_length")),
                "row_count": len(group_rows),
                "positive_row_count": sum(1 for value in positive_marginals if value > 0.0),
                "raw_segment_positive_marginal_sum": float(segment_positive_sum),
                "raw_segment_max_point_marginal": float(segment_positive_max),
                "raw_segment_mean_point_marginal": _mean(raw_marginals),
                "segment_target": (
                    float(segment_positive_sum / max_segment_sum)
                    if max_segment_sum > 1e-12
                    else 0.0
                ),
                "selector_segment_score_rank": _optional_int(context.get("segment_score_rank")),
                "selector_segment_length_support_rank": _optional_int(
                    context.get("segment_length_support_rank")
                ),
                "selector_segment_allocation_weight_rank": _optional_int(
                    context.get("segment_allocation_weight_rank")
                ),
                "selector_segment_allocation_count": _optional_int(
                    context.get("segment_allocation_count")
                ),
                "selector_segment_learned_count": _optional_int(context.get("learned_count")),
                "top_point_index": _optional_int(ordered_group[0].get("point_index"))
                if ordered_group
                else None,
            }
        )
        for local_rank, row in enumerate(ordered_group, start=1):
            context = row.get("selector_segment_context") or {}
            positive_marginal = max(0.0, float(row.get("marginal_query_useful_v1", 0.0)))
            point_rows.append(
                {
                    "point_index": _optional_int(row.get("point_index")),
                    "trajectory_index": _optional_int(row.get("trajectory_index")),
                    "segment_index": segment_idx,
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "point_offset_in_segment": _optional_int(
                        context.get("point_offset_in_segment")
                    ),
                    "raw_point_marginal": float(row.get("marginal_query_useful_v1", 0.0)),
                    "point_target_within_segment": float(positive_marginal / local_max),
                    "point_target_global": (
                        float(positive_marginal / max_point_positive)
                        if max_point_positive > 1e-12
                        else 0.0
                    ),
                    "intra_segment_teacher_rank": int(local_rank),
                    "selector_score_candidate_rank_fraction": _optional_float(
                        row.get("selector_score_candidate_rank_fraction")
                    ),
                    "segment_score_candidate_rank_fraction": _optional_float(
                        row.get("segment_score_candidate_rank_fraction")
                    ),
                    "selector_segment_score_rank": _optional_int(
                        context.get("segment_score_rank")
                    ),
                    "selector_segment_allocation_count": _optional_int(
                        context.get("segment_allocation_count")
                    ),
                }
            )

    point_values = [float(row["raw_point_marginal"]) for row in point_rows]
    teacher_value_variation = max(point_values) - min(point_values) if point_values else 0.0
    positive_segment_target_count = sum(
        1 for row in segment_rows if float(row["raw_segment_positive_marginal_sum"]) > 0.0
    )
    positive_point_target_count = sum(
        1 for row in point_rows if float(row["raw_point_marginal"]) > 0.0
    )
    teacher_target_shape_viable = (
        len(point_rows) >= 2
        and positive_point_target_count > 0
        and teacher_value_variation > 1e-12
    )
    candidate_for_train_side_teacher = bool(teacher_target_shape_viable and usage_allowed)
    candidate_reason = (
        "candidate_available"
        if candidate_for_train_side_teacher
        else _separated_teacher_candidate_rejection_reason(
            teacher_usage_split=usage_split,
            teacher_target_shape_viable=teacher_target_shape_viable,
        )
    )
    segment_rows = sorted(
        segment_rows,
        key=lambda row: (
            -float(row["raw_segment_positive_marginal_sum"]),
            row["trajectory_index"] if row["trajectory_index"] is not None else -1,
            row["segment_index"] if row["segment_index"] is not None else -1,
        ),
    )
    point_rows = sorted(
        point_rows,
        key=lambda row: (
            -float(row["raw_point_marginal"]),
            row["point_index"] if row["point_index"] is not None else -1,
        ),
    )
    summary.update(
        {
            "available": True,
            "reason": None,
            "teacher_value_variation": float(teacher_value_variation),
            "segment_target_normalization": (
                "segment_positive_marginal_sum_div_global_max_segment_sum"
            ),
            "point_target_normalization": (
                "positive_point_marginal_div_segment_max_and_global_max"
            ),
            "segment_target_count": len(segment_rows),
            "point_target_count": len(point_rows),
            "positive_segment_target_count": positive_segment_target_count,
            "positive_point_target_count": positive_point_target_count,
            "segment_target_rows": segment_rows,
            "point_target_rows": point_rows,
            "teacher_target_shape_viable": bool(teacher_target_shape_viable),
            "candidate_for_train_side_teacher": candidate_for_train_side_teacher,
            "candidate_for_train_side_teacher_reason": candidate_reason,
        }
    )
    return summary


def _teacher_vector_rejection(reason: str, *, teacher_usage_split: str | None) -> dict[str, Any]:
    return {
        "available": False,
        "diagnostic_only": True,
        "reason": str(reason),
        "teacher_usage_split": teacher_usage_split,
        "eval_time_feature_allowed": False,
    }


def _normalized_blend_score_vector(scores: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    values = scores.detach().cpu().float().flatten()
    if int(values.numel()) <= 0:
        return values, {"available": False, "reason": "empty_score_vector"}
    finite = torch.isfinite(values)
    if not bool(finite.any().item()):
        return torch.zeros_like(values), {"available": False, "reason": "no_finite_scores"}
    clean = torch.where(finite, values, torch.zeros_like(values))
    finite_values = clean[finite]
    min_value = float(finite_values.min().item())
    max_value = float(finite_values.max().item())
    span = max_value - min_value
    if span <= 1e-12:
        return torch.zeros_like(clean), {
            "available": False,
            "reason": "no_score_variation",
            "score_min": min_value,
            "score_max": max_value,
        }
    normalized = torch.zeros_like(clean)
    normalized[finite] = ((finite_values - min_value) / span).clamp(0.0, 1.0)
    return normalized, {
        "available": True,
        "score_min": min_value,
        "score_max": max_value,
        "score_span": float(span),
    }


def hybrid_marginal_teacher_selector_score_vectors(
    *,
    primary_point_scores: torch.Tensor,
    primary_segment_scores: torch.Tensor | None,
    primary_segment_score_source_label: str | None = None,
    teacher_point_scores: torch.Tensor,
    teacher_segment_scores: torch.Tensor,
    teacher_weight: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    """Blend dense primary selector scores with sparse exact-marginal teacher scores."""
    point_primary = primary_point_scores.detach().cpu().float().flatten()
    teacher_points = teacher_point_scores.detach().cpu().float().flatten().clamp(0.0, 1.0)
    teacher_segments = teacher_segment_scores.detach().cpu().float().flatten().clamp(0.0, 1.0)
    if point_primary.shape != teacher_points.shape or point_primary.shape != teacher_segments.shape:
        return None, None, {
            "available": False,
            "diagnostic_only": True,
            "reason": "score_shape_mismatch",
            "primary_point_count": int(point_primary.numel()),
            "teacher_point_count": int(teacher_points.numel()),
            "teacher_segment_point_count": int(teacher_segments.numel()),
        }
    if primary_segment_scores is None:
        primary_segment = point_primary
        primary_segment_source = primary_segment_score_source_label or "primary_point_scores"
    else:
        primary_segment = primary_segment_scores.detach().cpu().float().flatten()
        primary_segment_source = (
            primary_segment_score_source_label or "primary_segment_scores"
        )
        if primary_segment.shape != point_primary.shape:
            return None, None, {
                "available": False,
                "diagnostic_only": True,
                "reason": "primary_segment_score_shape_mismatch",
                "primary_point_count": int(point_primary.numel()),
                "primary_segment_point_count": int(primary_segment.numel()),
            }
    weight = max(0.0, min(1.0, float(teacher_weight)))
    normalized_point, point_diag = _normalized_blend_score_vector(point_primary)
    normalized_segment, segment_diag = _normalized_blend_score_vector(primary_segment)
    if not bool(point_diag.get("available", False)):
        return None, None, {
            "available": False,
            "diagnostic_only": True,
            "reason": f"primary_point_scores_{point_diag.get('reason', 'unavailable')}",
            "teacher_weight": weight,
        }
    if not bool(segment_diag.get("available", False)):
        return None, None, {
            "available": False,
            "diagnostic_only": True,
            "reason": f"{primary_segment_source}_{segment_diag.get('reason', 'unavailable')}",
            "teacher_weight": weight,
        }
    hybrid_points = ((1.0 - weight) * normalized_point + weight * teacher_points).clamp(0.0, 1.0)
    hybrid_segments = (
        (1.0 - weight) * normalized_segment + weight * teacher_segments
    ).clamp(0.0, 1.0)
    teacher_positive_points = int((teacher_points > 0.0).sum().item())
    teacher_positive_segment_points = int((teacher_segments > 0.0).sum().item())
    diagnostics = {
        "available": True,
        "diagnostic_only": True,
        "teacher_weight": weight,
        "primary_weight": float(1.0 - weight),
        "primary_segment_score_source": primary_segment_source,
        "point_count": int(point_primary.numel()),
        "teacher_positive_point_score_count": teacher_positive_points,
        "teacher_positive_segment_score_point_count": teacher_positive_segment_points,
        "hybrid_positive_point_score_count": int((hybrid_points > 0.0).sum().item()),
        "hybrid_positive_segment_score_point_count": int((hybrid_segments > 0.0).sum().item()),
        "primary_point_score_diagnostics": point_diag,
        "primary_segment_score_diagnostics": segment_diag,
        "hybrid_point_score_max": float(hybrid_points.max().item())
        if int(hybrid_points.numel()) > 0
        else 0.0,
        "hybrid_segment_score_max": float(hybrid_segments.max().item())
        if int(hybrid_segments.numel()) > 0
        else 0.0,
    }
    return hybrid_segments, hybrid_points, diagnostics


def separated_marginal_teacher_selector_score_vectors(
    summary: dict[str, Any],
    *,
    point_count: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    """Build selector score vectors from a full, train/checkpoint teacher payload.

    This is intentionally stricter than target construction: compact summaries
    and eval summaries must be rejected because they are not valid training or
    checkpoint-selection teacher sources.
    """
    usage_split = str(summary.get("teacher_usage_split", "unknown"))
    if bool(summary.get("eval_time_feature_allowed", False)):
        return (
            None,
            None,
            _teacher_vector_rejection(
                "eval_time_teacher_features_not_allowed",
                teacher_usage_split=usage_split,
            ),
        )
    if usage_split not in _TRAIN_OR_CHECKPOINT_TEACHER_USAGE_SPLITS:
        return (
            None,
            None,
            _teacher_vector_rejection(
                _separated_teacher_candidate_rejection_reason(
                    teacher_usage_split=usage_split,
                    teacher_target_shape_viable=bool(
                        summary.get("teacher_target_shape_viable", False)
                    ),
                ),
                teacher_usage_split=usage_split,
            ),
        )
    if not bool(summary.get("teacher_usage_allowed_for_train_or_checkpoint", False)):
        return (
            None,
            None,
            _teacher_vector_rejection(
                "teacher_usage_split_not_allowed_for_train_or_checkpoint",
                teacher_usage_split=usage_split,
            ),
        )
    if not bool(summary.get("candidate_for_train_side_teacher", False)):
        return (
            None,
            None,
            _teacher_vector_rejection(
                str(
                    summary.get(
                        "candidate_for_train_side_teacher_reason",
                        "not_a_train_side_teacher_candidate",
                    )
                ),
                teacher_usage_split=usage_split,
            ),
        )
    if int(point_count) <= 0:
        return (
            None,
            None,
            _teacher_vector_rejection("empty_point_domain", teacher_usage_split=usage_split),
        )
    segment_target_rows = summary.get("segment_target_rows")
    point_target_rows = summary.get("point_target_rows")
    if not isinstance(segment_target_rows, list) or not isinstance(point_target_rows, list):
        return (
            None,
            None,
            _teacher_vector_rejection(
                "missing_target_rows_full_selector_trace_required",
                teacher_usage_split=usage_split,
            ),
        )

    segment_scores = torch.zeros((int(point_count),), dtype=torch.float32)
    point_scores = torch.zeros((int(point_count),), dtype=torch.float32)
    malformed_segment_rows = 0
    applied_segment_rows = 0
    for row in segment_target_rows:
        if not isinstance(row, dict):
            malformed_segment_rows += 1
            continue
        start = _optional_int(row.get("segment_start"))
        end = _optional_int(row.get("segment_end"))
        if start is None or end is None or end <= start or start < 0 or end > int(point_count):
            malformed_segment_rows += 1
            continue
        target = max(0.0, min(1.0, float(row.get("segment_target", 0.0))))
        segment_scores[start:end] = torch.maximum(
            segment_scores[start:end],
            torch.full((int(end - start),), target, dtype=torch.float32),
        )
        applied_segment_rows += 1

    malformed_point_rows = 0
    applied_point_rows = 0
    for row in point_target_rows:
        if not isinstance(row, dict):
            malformed_point_rows += 1
            continue
        point_idx = _optional_int(row.get("point_index"))
        if point_idx is None or point_idx < 0 or point_idx >= int(point_count):
            malformed_point_rows += 1
            continue
        target_value = row.get("point_target_within_segment")
        if target_value is None:
            target_value = row.get("point_target_global", 0.0)
        target = max(0.0, min(1.0, float(target_value)))
        point_scores[int(point_idx)] = max(float(point_scores[int(point_idx)].item()), target)
        applied_point_rows += 1

    positive_segment_points = int((segment_scores > 0.0).sum().item())
    positive_point_count = int((point_scores > 0.0).sum().item())
    if applied_segment_rows <= 0 or applied_point_rows <= 0:
        return (
            None,
            None,
            _teacher_vector_rejection(
                "no_valid_teacher_target_rows",
                teacher_usage_split=usage_split,
            ),
        )
    if positive_segment_points <= 0 or positive_point_count <= 0:
        return (
            None,
            None,
            _teacher_vector_rejection(
                "teacher_score_vectors_have_no_positive_support",
                teacher_usage_split=usage_split,
            ),
        )

    diagnostics = {
        "available": True,
        "diagnostic_only": True,
        "teacher_usage_split": usage_split,
        "eval_time_feature_allowed": False,
        "source": "separated_marginal_teacher_summary",
        "segment_score_source": "segment_target",
        "point_score_source": "point_target_within_segment",
        "point_count": int(point_count),
        "segment_target_row_count": len(segment_target_rows),
        "point_target_row_count": len(point_target_rows),
        "applied_segment_target_row_count": int(applied_segment_rows),
        "applied_point_target_row_count": int(applied_point_rows),
        "malformed_segment_target_row_count": int(malformed_segment_rows),
        "malformed_point_target_row_count": int(malformed_point_rows),
        "positive_segment_score_point_count": positive_segment_points,
        "positive_point_score_count": positive_point_count,
        "segment_score_max": float(segment_scores.max().item()),
        "point_score_max": float(point_scores.max().item()),
        "requires_full_selector_trace_rows": True,
        "compact_summary_allowed": False,
    }
    return segment_scores, point_scores, diagnostics


def retained_decision_marginal_query_useful_diagnostics(
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
    component_vectors = _score_component_vectors(score_component_vectors, point_count)
    query_free_proxy_vectors = _score_component_vectors(
        query_free_teacher_proxy_vectors,
        point_count,
    )
    sampled_prior_component_vectors = _score_component_vectors(sampled_prior_vectors, point_count)
    model_prior_component_vectors = _score_component_vectors(model_prior_vectors, point_count)
    trace_mask_state = _trace_mask_state(selector_trace=selector_trace, point_count=point_count)
    selector_segment_context_rows = _selector_segment_context_rows(
        selector_trace=selector_trace,
        point_count=point_count,
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
            "trajectory_index": _trajectory_index_for_point(int(index), boundaries),
            "selector_stage_state": {
                name: bool(mask[int(index)].item()) for name, mask in trace_mask_state.items()
            },
            "selector_segment_context": _selector_segment_context_for_point(
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
            "query_useful_v1_score_components": _row_score_component_values(
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

    _attach_top_marginal_miss_diagnostics(rows)
    top_marginal_miss_diagnostics = _top_marginal_miss_bucket_summary(rows)
    top_marginal_miss_summary = dict(top_marginal_miss_diagnostics)
    top_marginal_miss_summary.pop("top_marginal_rows", None)
    top_marginal_miss_summary["top_marginal_rows_in_selector_trace_only"] = True

    by_source = {
        name: _score_alignment_summary(group_rows)
        for name, group_rows in sorted(_group_rows(rows, "source").items())
    }
    by_decision = {
        name: _score_alignment_summary(group_rows)
        for name, group_rows in sorted(_group_rows(rows, "decision").items())
    }
    guard_coupling_summary = _query_free_teacher_proxy_guard_coupling_summary(rows)
    learned_teacher_summary = _learned_controllable_marginal_teacher_summary(rows)
    separated_teacher_summary = _separated_marginal_teacher_targets(
        rows,
        teacher_usage_split=teacher_usage_split,
    )
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
        "overall": _score_alignment_summary(rows),
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
                -float(row["marginal_query_useful_v1"]),
                int(row["point_index"]),
            ),
        ),
    }
