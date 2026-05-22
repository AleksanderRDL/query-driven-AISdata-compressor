"""Selector diagnostic helpers used by run orchestration."""

from __future__ import annotations

import math
from typing import Any

import torch

from learning.model_features import transform_workload_blind_range_prior_features
from learning.query_prior_fields import QUERY_PRIOR_FIELD_NAMES, sample_query_prior_fields
from learning.targets.query_local_utility import (
    QUERY_LOCAL_UTILITY_HEAD_NAMES,
    query_local_utility_path_length_support_targets,
    query_local_utility_point_score,
)
from orchestration.segment_audits import segment_top_mean
from orchestration.selector_marginal_diagnostics import (
    retained_decision_marginal_query_local_utility_diagnostics as retained_decision_marginal_query_local_utility_diagnostics,
)
from scoring.methods import FrozenMaskMethod
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
    query_hit_branch = 0.50 * q_hit
    behavior_branch = 0.45 * behavior
    pre_replacement_score = query_hit_branch + behavior_branch
    replacement_multiplier = 0.75 + 0.25 * replacement
    replacement_modulated_score = pre_replacement_score * replacement_multiplier
    boundary_bonus = 0.05 * boundary
    composed_score = query_local_utility_point_score(
        q_hit=q_hit,
        behavior=behavior,
        boundary=boundary,
        replacement=replacement,
    )
    out.update(
        {
            "factorized_query_hit_branch": query_hit_branch.contiguous(),
            "factorized_behavior_branch": behavior_branch.contiguous(),
            "factorized_pre_replacement_score": pre_replacement_score.contiguous(),
            "factorized_replacement_multiplier": replacement_multiplier.contiguous(),
            "factorized_replacement_modulated_score": replacement_modulated_score.contiguous(),
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
    base_segment_score_source: str = "segment_budget_head_top20_mean",
) -> str:
    """Return an honest selector trace label for segment allocation scores."""
    weight = max(0.0, min(1.0, float(length_support_blend_weight)))
    if path_length_support_scores is not None and weight >= 1.0 - 1e-12:
        return "path_length_support_head_top20_mean"
    base_source = (
        str(base_segment_score_source)
        if segment_scores is not None
        else "point_score_top20_mean"
    )
    if path_length_support_scores is not None and weight > 0.0:
        if base_source == "segment_budget_head_top20_mean":
            return "segment_budget_path_length_support_blend_top20_mean"
        return f"{base_source}_path_length_support_blend_top20_mean"
    return base_source


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
