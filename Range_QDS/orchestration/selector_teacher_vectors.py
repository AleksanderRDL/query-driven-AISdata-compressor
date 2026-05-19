"""Teacher score-vector builders for selector diagnostics."""

from __future__ import annotations

from typing import Any

import torch

from orchestration import selector_marginal_alignment, selector_trace_payloads


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
        return (
            None,
            None,
            {
                "available": False,
                "diagnostic_only": True,
                "reason": "score_shape_mismatch",
                "primary_point_count": int(point_primary.numel()),
                "teacher_point_count": int(teacher_points.numel()),
                "teacher_segment_point_count": int(teacher_segments.numel()),
            },
        )
    if primary_segment_scores is None:
        primary_segment = point_primary
        primary_segment_source = primary_segment_score_source_label or "primary_point_scores"
    else:
        primary_segment = primary_segment_scores.detach().cpu().float().flatten()
        primary_segment_source = primary_segment_score_source_label or "primary_segment_scores"
        if primary_segment.shape != point_primary.shape:
            return (
                None,
                None,
                {
                    "available": False,
                    "diagnostic_only": True,
                    "reason": "primary_segment_score_shape_mismatch",
                    "primary_point_count": int(point_primary.numel()),
                    "primary_segment_point_count": int(primary_segment.numel()),
                },
            )
    weight = max(0.0, min(1.0, float(teacher_weight)))
    normalized_point, point_diag = _normalized_blend_score_vector(point_primary)
    normalized_segment, segment_diag = _normalized_blend_score_vector(primary_segment)
    if not bool(point_diag.get("available", False)):
        return (
            None,
            None,
            {
                "available": False,
                "diagnostic_only": True,
                "reason": f"primary_point_scores_{point_diag.get('reason', 'unavailable')}",
                "teacher_weight": weight,
            },
        )
    if not bool(segment_diag.get("available", False)):
        return (
            None,
            None,
            {
                "available": False,
                "diagnostic_only": True,
                "reason": f"{primary_segment_source}_{segment_diag.get('reason', 'unavailable')}",
                "teacher_weight": weight,
            },
        )
    hybrid_points = ((1.0 - weight) * normalized_point + weight * teacher_points).clamp(0.0, 1.0)
    hybrid_segments = ((1.0 - weight) * normalized_segment + weight * teacher_segments).clamp(
        0.0, 1.0
    )
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
    if usage_split not in selector_marginal_alignment.TRAIN_OR_CHECKPOINT_TEACHER_USAGE_SPLITS:
        return (
            None,
            None,
            _teacher_vector_rejection(
                selector_marginal_alignment.separated_teacher_candidate_rejection_reason(
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
        start = selector_trace_payloads.optional_int(row.get("segment_start"))
        end = selector_trace_payloads.optional_int(row.get("segment_end"))
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
        point_idx = selector_trace_payloads.optional_int(row.get("point_index"))
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
