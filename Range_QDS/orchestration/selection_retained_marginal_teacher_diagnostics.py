"""Retained-marginal teacher diagnostics for selection-causality runs."""

from __future__ import annotations

from typing import Any

import torch

from config.run_config import RunConfig
from learning.outputs import TrainingOutputs
from orchestration.selector_diagnostics import (
    factorized_score_component_vectors_from_logits,
    query_free_retained_removal_teacher_proxy_vectors,
    query_prior_component_vectors_for_points,
    retained_decision_marginal_query_local_utility_diagnostics,
    selector_segment_score_source_label,
)
from scoring.query_cache import ScoringQueryCache
from selection.learned_segment_budget import simplify_with_learned_segment_budget_with_trace


def _compact_selection_retained_marginal_teacher_summary(
    payload: dict[str, Any],
    *,
    source_path: str,
    split_name: str,
) -> dict[str, Any]:
    summary_keys = (
        "available",
        "diagnostic_only",
        "exact_query_local_utility_marginals",
        "performance_mode",
        "primary_query_local_utility",
        "retained_count",
        "point_count",
        "max_retained_per_source",
        "max_removed_candidates",
        "score_fields_available",
        "score_component_fields_available",
        "context_fields_available",
        "candidate_count",
        "overall",
        "by_source",
        "by_decision",
        "query_free_teacher_proxy_guard_coupling_summary",
        "learned_controllable_marginal_teacher_summary",
        "separated_marginal_teacher_summary",
        "top_marginal_miss_summary",
    )
    summary = {key: payload.get(key) for key in summary_keys if key in payload}
    separated_summary = summary.get("separated_marginal_teacher_summary")
    if isinstance(separated_summary, dict):
        compact_separated_summary = dict(separated_summary)
        compact_separated_summary.pop("segment_target_rows", None)
        compact_separated_summary.pop("point_target_rows", None)
        compact_separated_summary["segment_target_rows_in_selector_trace_only"] = True
        compact_separated_summary["point_target_rows_in_selector_trace_only"] = True
        summary["separated_marginal_teacher_summary"] = compact_separated_summary
    top_miss_summary = summary.get("top_marginal_miss_summary")
    if isinstance(top_miss_summary, dict):
        compact_top_miss_summary = dict(top_miss_summary)
        compact_top_miss_summary.pop("top_marginal_rows", None)
        compact_top_miss_summary["top_marginal_rows_in_selector_trace_only"] = True
        summary["top_marginal_miss_summary"] = compact_top_miss_summary
    summary.update(
        {
            "split": split_name,
            "source_path": source_path,
            "rows_in_selector_trace_only": True,
            "query_conditioned_teacher_allowed_for_train_or_checkpoint_diagnostics_only": True,
            "eval_time_feature_allowed": False,
        }
    )
    return summary


def selection_retained_marginal_teacher_diagnostics(
    *,
    trained: TrainingOutputs,
    selection_points: torch.Tensor,
    selection_boundaries: list[tuple[int, int]],
    typed_queries: Any,
    selection_query_cache: ScoringQueryCache | None,
    config: RunConfig,
    split_name: str,
    selector_trace_source_path: str,
    primary_mask: torch.Tensor,
    primary_scores: Any,
    primary_raw_preds: Any,
    primary_head_logits: Any,
    primary_segment_scores: Any,
    primary_path_length_support_scores: Any,
    primary_selector_segment_scores: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    selection_marginal_teacher_summary: dict[str, Any] = {
        "available": False,
        "diagnostic_only": True,
        "split": split_name,
        "reason": "not_run",
    }
    if not isinstance(primary_scores, torch.Tensor):
        return None, selection_marginal_teacher_summary

    try:
        trace_mask, trace = simplify_with_learned_segment_budget_with_trace(
            primary_scores,
            selection_boundaries,
            float(config.model.compression_ratio),
            segment_scores=(
                primary_selector_segment_scores
                if isinstance(primary_selector_segment_scores, torch.Tensor)
                else None
            ),
            segment_point_scores=(
                primary_segment_scores if isinstance(primary_segment_scores, torch.Tensor) else None
            ),
            points=selection_points.detach().cpu().float(),
            geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
            segment_length_support_weight=float(
                config.model.learned_segment_allocation_length_support_weight
            ),
            segment_allocation_weight_floor=float(
                config.model.learned_segment_allocation_weight_floor
            ),
            segment_score_point_blend_weight=float(config.model.learned_segment_score_blend_weight),
            segment_transfer_calibration_mode=str(
                config.model.learned_segment_transfer_calibration_mode
            ),
            fairness_preallocation_enabled=bool(
                config.model.learned_segment_fairness_preallocation
            ),
            length_repair_fraction=float(config.model.learned_segment_length_repair_fraction),
            length_repair_score_protection_fraction=float(
                config.model.learned_segment_length_repair_score_protection_fraction
            ),
            segment_score_source_label=selector_segment_score_source_label(
                segment_scores=(
                    primary_selector_segment_scores
                    if isinstance(primary_selector_segment_scores, torch.Tensor)
                    else None
                ),
                path_length_support_scores=(
                    primary_path_length_support_scores
                    if isinstance(primary_path_length_support_scores, torch.Tensor)
                    else None
                ),
                length_support_blend_weight=float(
                    config.model.learned_segment_length_support_blend_weight
                ),
                base_segment_score_source="segment_budget_head_top20_mean",
            ),
        )
        mask_matches_primary = bool(
            torch.equal(trace_mask.detach().cpu(), primary_mask.detach().cpu())
        )
        trace["retained_mask_matches_primary"] = mask_matches_primary
        if split_name == "checkpoint_selection":
            trace["retained_mask_matches_selection_primary"] = mask_matches_primary
        trace["frozen_primary_retained_count"] = int(primary_mask.sum().item())
        sampled_prior_vectors, model_prior_vectors = query_prior_component_vectors_for_points(
            selection_points.detach().cpu().float(),
            trained.feature_context.get("query_prior_field"),
        )
        teacher_proxy_vectors = query_free_retained_removal_teacher_proxy_vectors(
            selection_points.detach().cpu().float(),
            selection_boundaries,
        )
        marginal_points = (
            selection_points
            if selection_query_cache is not None
            else selection_points.detach().cpu().float()
        )
        trace["retained_decision_marginal_query_local_utility_alignment"] = (
            retained_decision_marginal_query_local_utility_diagnostics(
                points=marginal_points,
                boundaries=selection_boundaries,
                typed_queries=typed_queries,
                primary_retained_mask=trace_mask.detach().cpu().bool(),
                raw_scores=primary_raw_preds,
                selector_scores=primary_scores,
                segment_scores=(
                    primary_segment_scores
                    if isinstance(primary_segment_scores, torch.Tensor)
                    else None
                ),
                score_component_vectors=factorized_score_component_vectors_from_logits(
                    primary_head_logits if isinstance(primary_head_logits, torch.Tensor) else None
                ),
                query_free_teacher_proxy_vectors=teacher_proxy_vectors,
                sampled_prior_vectors=sampled_prior_vectors,
                model_prior_vectors=model_prior_vectors,
                selector_trace=trace,
                query_cache=selection_query_cache,
                max_retained_per_source=32,
                max_removed_candidates=64,
                teacher_usage_split=split_name,
            )
        )
        selection_marginal_teacher_summary = _compact_selection_retained_marginal_teacher_summary(
            trace["retained_decision_marginal_query_local_utility_alignment"],
            source_path=selector_trace_source_path,
            split_name=split_name,
        )
        return trace, selection_marginal_teacher_summary
    except Exception as exc:  # pragma: no cover - diagnostic should not break selection.
        return (
            {
                "available": False,
                "reason": "retained_marginal_teacher_diagnostic_failed",
                "error": str(exc),
            },
            {
                "available": False,
                "diagnostic_only": True,
                "reason": "diagnostic_failed",
                "error": str(exc),
                "source_path": selector_trace_source_path,
                "split": split_name,
            },
        )
