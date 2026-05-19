"""Checkpoint-selection causality diagnostics for query-driven runs."""

from __future__ import annotations

from typing import Any

import torch

from config.run_config import RunConfig
from learning.outputs import TrainingOutputs
from learning.predictability_audit import query_prior_predictability_scores
from learning.query_prior_fields import zero_query_prior_field_like
from orchestration.causality import (
    causality_ablation_diagnostics_payload,
    head_ablation_sensitivity,
    model_prior_feature_sensitivity,
    prior_ablation_sensitivity_from_tensors,
    prior_feature_sample_sensitivity,
    retained_mask_comparison,
    training_outputs_with_query_prior_field,
)
from orchestration.mlqds_method_factory import build_mlqds_method
from orchestration.model_ablations import (
    raw_predictions_without_factorized_head,
    scores_without_factorized_head,
    shuffled_query_prior_field,
)
from orchestration.selector_diagnostics import (
    factorized_score_component_vectors_from_logits,
    hybrid_marginal_teacher_selector_score_vectors,
    learned_segment_frozen_method,
    neutral_segment_scores_for_ablation,
    query_free_retained_removal_teacher_proxy_vectors,
    query_prior_component_vectors_for_points,
    retained_decision_marginal_query_useful_diagnostics,
    selector_segment_score_source_label,
    separated_marginal_teacher_selector_score_vectors,
)
from scoring.method_scoring import score_method
from scoring.methods import FrozenMaskMethod, MLQDSMethod
from scoring.metrics import MethodScore
from scoring.query_cache import ScoringQueryCache
from selection.learned_segment_budget import (
    blend_segment_support_scores,
    simplify_with_learned_segment_budget_v1_with_trace,
)
from selection.selector_types import LEARNED_SEGMENT_BUDGET_SELECTOR_TYPE
from workloads.query_types import single_workload_type


def _compact_selection_retained_marginal_teacher_summary(
    payload: dict[str, Any],
) -> dict[str, Any]:
    source_path = (
        "selector_trace_diagnostics.selection_primary."
        "retained_decision_marginal_query_useful_alignment"
    )
    summary_keys = (
        "available",
        "diagnostic_only",
        "exact_query_useful_v1_marginals",
        "performance_mode",
        "primary_query_useful_v1",
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
            "split": "checkpoint_selection",
            "source_path": source_path,
            "rows_in_selector_trace_only": True,
            "query_conditioned_teacher_allowed_for_train_or_checkpoint_diagnostics_only": True,
            "eval_time_feature_allowed": False,
        }
    )
    return summary


def build_selection_causality_diagnostics(
    *,
    trained: TrainingOutputs,
    selection_points: torch.Tensor | None,
    selection_boundaries: list[tuple[int, int]] | None,
    selection_workload: Any | None,
    eval_workload_map: dict[str, float],
    selection_query_cache: ScoringQueryCache | None,
    config: RunConfig,
    seeds: Any,
) -> dict[str, Any]:
    """Return checkpoint-validation ablation diagnostics without changing selection."""
    if selection_points is None or selection_boundaries is None or selection_workload is None:
        return {"available": False, "reason": "missing_selection_split"}
    if (
        str(getattr(config.model, "selector_type", "")).lower()
        != LEARNED_SEGMENT_BUDGET_SELECTOR_TYPE
    ):
        return {"available": False, "reason": "requires_learned_segment_budget_v1"}

    workload_type = single_workload_type(eval_workload_map)

    def _mlqds_method(
        *,
        name: str,
        trained_outputs: TrainingOutputs,
        workload: Any,
    ) -> MLQDSMethod:
        return build_mlqds_method(
            name=name,
            trained=trained_outputs,
            workload=workload,
            workload_map=eval_workload_map,
            config=config,
            range_geometry_blend=0.0,
        )

    primary_method = _mlqds_method(
        name="MLQDS_selection_primary",
        trained_outputs=trained,
        workload=selection_workload,
    )
    try:
        primary_mask = (
            primary_method.simplify(
                selection_points,
                selection_boundaries,
                float(config.model.compression_ratio),
            )
            .detach()
            .cpu()
        )
    except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
        return {"available": False, "reason": "primary_mask_freeze_failed", "error": str(exc)}

    primary_score = score_method(
        method=FrozenMaskMethod(name="MLQDS", retained_mask=primary_mask),
        points=selection_points,
        boundaries=selection_boundaries,
        typed_queries=selection_workload.typed_queries,
        workload_map=eval_workload_map,
        compression_ratio=config.model.compression_ratio,
        query_cache=selection_query_cache,
    )
    primary_scores = getattr(primary_method, "_score_cache", None)
    primary_raw_preds = getattr(primary_method, "_raw_pred_cache", None)
    primary_head_logits = getattr(primary_method, "_head_logit_cache", None)
    primary_segment_scores = getattr(primary_method, "_segment_score_cache", None)
    primary_path_length_support_scores = getattr(
        primary_method, "_path_length_support_score_cache", None
    )
    primary_selector_segment_scores = getattr(primary_method, "_selector_segment_score_cache", None)
    if isinstance(primary_scores, torch.Tensor):
        primary_scores = primary_scores.detach().cpu().float()
    if isinstance(primary_raw_preds, torch.Tensor):
        primary_raw_preds = primary_raw_preds.detach().cpu().float()
    if isinstance(primary_head_logits, torch.Tensor):
        primary_head_logits = primary_head_logits.detach().cpu().float()
    if isinstance(primary_segment_scores, torch.Tensor):
        primary_segment_scores = primary_segment_scores.detach().cpu().float()
    if isinstance(primary_path_length_support_scores, torch.Tensor):
        primary_path_length_support_scores = (
            primary_path_length_support_scores.detach().cpu().float()
        )
    if isinstance(primary_selector_segment_scores, torch.Tensor):
        primary_selector_segment_scores = primary_selector_segment_scores.detach().cpu().float()

    selection_selector_trace: dict[str, Any] | None = None
    selection_marginal_teacher_summary: dict[str, Any] = {
        "available": False,
        "diagnostic_only": True,
        "split": "checkpoint_selection",
        "reason": "not_run",
    }
    separated_teacher_selector_diagnostic: dict[str, Any] = {
        "available": False,
        "diagnostic_only": True,
        "split": "checkpoint_selection",
        "reason": "not_run",
    }
    separated_teacher_method_name = "MLQDS_checkpoint_selection_marginal_teacher_selector"
    hybrid_teacher_weights = (0.10, 0.25)
    hybrid_teacher_method_names = {
        weight: f"MLQDS_checkpoint_selection_marginal_teacher_primary_blend_w{int(weight * 100):02d}"
        for weight in hybrid_teacher_weights
    }
    separated_teacher_hybrid_selector_diagnostics: dict[str, Any] = {
        "available": False,
        "diagnostic_only": True,
        "split": "checkpoint_selection",
        "reason": "not_run",
        "methods": {},
    }
    if isinstance(primary_scores, torch.Tensor):
        try:
            trace_mask, trace = simplify_with_learned_segment_budget_v1_with_trace(
                primary_scores,
                selection_boundaries,
                float(config.model.compression_ratio),
                segment_scores=(
                    primary_selector_segment_scores
                    if isinstance(primary_selector_segment_scores, torch.Tensor)
                    else None
                ),
                segment_point_scores=(
                    primary_segment_scores
                    if isinstance(primary_segment_scores, torch.Tensor)
                    else None
                ),
                points=selection_points.detach().cpu().float(),
                geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                segment_length_support_weight=float(
                    config.model.learned_segment_allocation_length_support_weight
                ),
                segment_allocation_weight_floor=float(
                    config.model.learned_segment_allocation_weight_floor
                ),
                segment_score_point_blend_weight=float(
                    config.model.learned_segment_score_blend_weight
                ),
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
                    segment_scores=primary_segment_scores
                    if isinstance(primary_segment_scores, torch.Tensor)
                    else None,
                    path_length_support_scores=primary_path_length_support_scores
                    if isinstance(primary_path_length_support_scores, torch.Tensor)
                    else None,
                    length_support_blend_weight=float(
                        config.model.learned_segment_length_support_blend_weight
                    ),
                ),
            )
            trace["retained_mask_matches_selection_primary"] = bool(
                torch.equal(trace_mask.detach().cpu(), primary_mask.detach().cpu())
            )
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
            trace["retained_decision_marginal_query_useful_alignment"] = (
                retained_decision_marginal_query_useful_diagnostics(
                    points=marginal_points,
                    boundaries=selection_boundaries,
                    typed_queries=selection_workload.typed_queries,
                    primary_retained_mask=trace_mask.detach().cpu().bool(),
                    raw_scores=primary_raw_preds,
                    selector_scores=primary_scores,
                    segment_scores=primary_segment_scores
                    if isinstance(primary_segment_scores, torch.Tensor)
                    else None,
                    score_component_vectors=factorized_score_component_vectors_from_logits(
                        primary_head_logits
                        if isinstance(primary_head_logits, torch.Tensor)
                        else None
                    ),
                    query_free_teacher_proxy_vectors=teacher_proxy_vectors,
                    sampled_prior_vectors=sampled_prior_vectors,
                    model_prior_vectors=model_prior_vectors,
                    selector_trace=trace,
                    query_cache=selection_query_cache,
                    max_retained_per_source=32,
                    max_removed_candidates=64,
                    teacher_usage_split="checkpoint_selection",
                )
            )
            selection_marginal_teacher_summary = (
                _compact_selection_retained_marginal_teacher_summary(
                    trace["retained_decision_marginal_query_useful_alignment"]
                )
            )
            selection_selector_trace = trace
        except Exception as exc:  # pragma: no cover - diagnostic should not break selection.
            selection_marginal_teacher_summary = {
                "available": False,
                "diagnostic_only": True,
                "split": "checkpoint_selection",
                "reason": "diagnostic_failed",
                "error": str(exc),
                "source_path": (
                    "selector_trace_diagnostics.selection_primary."
                    "retained_decision_marginal_query_useful_alignment"
                ),
            }
            selection_selector_trace = {
                "available": False,
                "reason": "retained_marginal_teacher_diagnostic_failed",
                "error": str(exc),
            }

    ablation_methods: list[FrozenMaskMethod] = []
    freeze_failures: dict[str, str] = {}
    prior_sensitivity: dict[str, Any] = {}
    head_sensitivity: dict[str, Any] = {}

    geometry_gain_weight = float(config.model.learned_segment_geometry_gain_weight)
    allocation_length_support_weight = float(
        config.model.learned_segment_allocation_length_support_weight
    )
    allocation_weight_floor = float(config.model.learned_segment_allocation_weight_floor)
    repair_score_protection_fraction = float(
        config.model.learned_segment_length_repair_score_protection_fraction
    )
    if selection_selector_trace is not None:
        retained_marginal = selection_selector_trace.get(
            "retained_decision_marginal_query_useful_alignment"
        )
        separated_summary = (
            retained_marginal.get("separated_marginal_teacher_summary")
            if isinstance(retained_marginal, dict)
            else None
        )
        if isinstance(separated_summary, dict):
            try:
                teacher_segment_scores, teacher_point_scores, vector_diagnostics = (
                    separated_marginal_teacher_selector_score_vectors(
                        separated_summary,
                        point_count=int(selection_points.shape[0]),
                    )
                )
                separated_teacher_selector_diagnostic = {
                    **vector_diagnostics,
                    "split": "checkpoint_selection",
                    "method_name": separated_teacher_method_name,
                    "selector_diagnostic_only": True,
                }
                if teacher_segment_scores is not None and teacher_point_scores is not None:
                    teacher_method = learned_segment_frozen_method(
                        name=separated_teacher_method_name,
                        scores=teacher_point_scores,
                        boundaries=selection_boundaries,
                        compression_ratio=float(config.model.compression_ratio),
                        segment_scores=teacher_segment_scores,
                        segment_point_scores=teacher_point_scores,
                        points=selection_points,
                        learned_segment_geometry_gain_weight=0.0,
                        learned_segment_allocation_length_support_weight=0.0,
                        learned_segment_allocation_weight_floor=allocation_weight_floor,
                        learned_segment_score_blend_weight=1.0,
                        learned_segment_transfer_calibration_mode=str(
                            config.model.learned_segment_transfer_calibration_mode
                        ),
                        learned_segment_fairness_preallocation=bool(
                            config.model.learned_segment_fairness_preallocation
                        ),
                        learned_segment_length_repair_fraction=float(
                            config.model.learned_segment_length_repair_fraction
                        ),
                        learned_segment_length_repair_score_protection_fraction=(
                            repair_score_protection_fraction
                        ),
                    )
                    ablation_methods.append(teacher_method)
                    separated_teacher_selector_diagnostic.update(
                        {
                            "frozen_mask_available": True,
                            "retained_count": int(teacher_method.retained_mask.sum().item()),
                            "uses_eval_queries": False,
                            "uses_checkpoint_selection_queries": True,
                            "geometry_tie_breaker_weight": 0.0,
                            "segment_length_support_weight": 0.0,
                            "segment_score_point_blend_weight": 1.0,
                        }
                    )
                    hybrid_method_diagnostics: dict[str, Any] = {}
                    primary_scores_for_hybrid = primary_scores
                    if not isinstance(primary_scores_for_hybrid, torch.Tensor):
                        raise ValueError("missing_primary_scores_for_hybrid_teacher_selector")
                    if isinstance(primary_selector_segment_scores, torch.Tensor):
                        primary_hybrid_segment_scores = primary_selector_segment_scores
                        primary_hybrid_segment_score_source = "primary_selector_segment_scores"
                    elif isinstance(primary_segment_scores, torch.Tensor):
                        primary_hybrid_segment_scores = primary_segment_scores
                        primary_hybrid_segment_score_source = "primary_segment_scores"
                    else:
                        primary_hybrid_segment_scores = None
                        primary_hybrid_segment_score_source = "primary_point_scores"
                    for teacher_weight in hybrid_teacher_weights:
                        hybrid_method_name = hybrid_teacher_method_names[teacher_weight]
                        try:
                            hybrid_segment_scores, hybrid_point_scores, hybrid_diag = (
                                hybrid_marginal_teacher_selector_score_vectors(
                                    primary_point_scores=primary_scores_for_hybrid,
                                    primary_segment_scores=primary_hybrid_segment_scores,
                                    primary_segment_score_source_label=(
                                        primary_hybrid_segment_score_source
                                    ),
                                    teacher_point_scores=teacher_point_scores,
                                    teacher_segment_scores=teacher_segment_scores,
                                    teacher_weight=teacher_weight,
                                )
                            )
                            hybrid_diag = {
                                **hybrid_diag,
                                "split": "checkpoint_selection",
                                "method_name": hybrid_method_name,
                                "selector_diagnostic_only": True,
                                "uses_eval_queries": False,
                                "uses_checkpoint_selection_queries": True,
                            }
                            if hybrid_segment_scores is None or hybrid_point_scores is None:
                                hybrid_method_diagnostics[hybrid_method_name] = hybrid_diag
                                continue
                            hybrid_method = learned_segment_frozen_method(
                                name=hybrid_method_name,
                                scores=hybrid_point_scores,
                                boundaries=selection_boundaries,
                                compression_ratio=float(config.model.compression_ratio),
                                segment_scores=hybrid_segment_scores,
                                segment_point_scores=hybrid_point_scores,
                                points=selection_points,
                                learned_segment_geometry_gain_weight=geometry_gain_weight,
                                learned_segment_allocation_length_support_weight=(
                                    allocation_length_support_weight
                                ),
                                learned_segment_allocation_weight_floor=allocation_weight_floor,
                                learned_segment_score_blend_weight=float(
                                    config.model.learned_segment_score_blend_weight
                                ),
                                learned_segment_transfer_calibration_mode=str(
                                    config.model.learned_segment_transfer_calibration_mode
                                ),
                                learned_segment_fairness_preallocation=bool(
                                    config.model.learned_segment_fairness_preallocation
                                ),
                                learned_segment_length_repair_fraction=float(
                                    config.model.learned_segment_length_repair_fraction
                                ),
                                learned_segment_length_repair_score_protection_fraction=(
                                    repair_score_protection_fraction
                                ),
                            )
                            ablation_methods.append(hybrid_method)
                            hybrid_diag.update(
                                {
                                    "frozen_mask_available": True,
                                    "retained_count": int(
                                        hybrid_method.retained_mask.sum().item()
                                    ),
                                    "geometry_tie_breaker_weight": geometry_gain_weight,
                                    "segment_length_support_weight": (
                                        allocation_length_support_weight
                                    ),
                                    "segment_score_point_blend_weight": float(
                                        config.model.learned_segment_score_blend_weight
                                    ),
                                }
                            )
                            hybrid_method_diagnostics[hybrid_method_name] = hybrid_diag
                        except Exception as exc:  # pragma: no cover
                            freeze_failures[hybrid_method_name] = str(exc)
                            hybrid_method_diagnostics[hybrid_method_name] = {
                                "available": False,
                                "diagnostic_only": True,
                                "split": "checkpoint_selection",
                                "reason": "hybrid_teacher_selector_diagnostic_failed",
                                "error": str(exc),
                                "method_name": hybrid_method_name,
                                "teacher_weight": float(teacher_weight),
                                "uses_eval_queries": False,
                                "uses_checkpoint_selection_queries": True,
                            }
                    hybrid_available = any(
                        bool(diag.get("available", False))
                        for diag in hybrid_method_diagnostics.values()
                        if isinstance(diag, dict)
                    )
                    separated_teacher_hybrid_selector_diagnostics = {
                        "available": bool(hybrid_available),
                        "diagnostic_only": True,
                        "split": "checkpoint_selection",
                        "reason": None
                        if hybrid_available
                        else "no_hybrid_teacher_selector_methods_available",
                        "teacher_weights": [float(weight) for weight in hybrid_teacher_weights],
                        "methods": hybrid_method_diagnostics,
                    }
            except Exception as exc:  # pragma: no cover - diagnostic should not break selection.
                freeze_failures[separated_teacher_method_name] = str(exc)
                separated_teacher_selector_diagnostic = {
                    "available": False,
                    "diagnostic_only": True,
                    "split": "checkpoint_selection",
                    "reason": "teacher_selector_diagnostic_failed",
                    "error": str(exc),
                    "method_name": separated_teacher_method_name,
                }
        else:
            separated_teacher_selector_diagnostic = {
                "available": False,
                "diagnostic_only": True,
                "split": "checkpoint_selection",
                "reason": "missing_full_separated_marginal_teacher_summary",
                "method_name": separated_teacher_method_name,
            }
            separated_teacher_hybrid_selector_diagnostics = {
                "available": False,
                "diagnostic_only": True,
                "split": "checkpoint_selection",
                "reason": "missing_full_separated_marginal_teacher_summary",
                "methods": {},
            }
    if isinstance(primary_scores, torch.Tensor) and geometry_gain_weight > 0.0:
        try:
            selection_segment_scores = (
                primary_selector_segment_scores
                if isinstance(primary_selector_segment_scores, torch.Tensor)
                else None
            )
            ablation_methods.append(
                learned_segment_frozen_method(
                    name="MLQDS_without_geometry_tie_breaker",
                    scores=primary_scores,
                    boundaries=selection_boundaries,
                    compression_ratio=float(config.model.compression_ratio),
                    segment_scores=selection_segment_scores,
                    segment_point_scores=primary_segment_scores,
                    points=selection_points,
                    learned_segment_geometry_gain_weight=0.0,
                    learned_segment_allocation_length_support_weight=(
                        allocation_length_support_weight
                    ),
                    learned_segment_allocation_weight_floor=allocation_weight_floor,
                    learned_segment_score_blend_weight=float(
                        config.model.learned_segment_score_blend_weight
                    ),
                    learned_segment_transfer_calibration_mode=str(
                        config.model.learned_segment_transfer_calibration_mode
                    ),
                    learned_segment_fairness_preallocation=bool(
                        config.model.learned_segment_fairness_preallocation
                    ),
                    learned_segment_length_repair_fraction=float(
                        config.model.learned_segment_length_repair_fraction
                    ),
                    learned_segment_length_repair_score_protection_fraction=(
                        repair_score_protection_fraction
                    ),
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            freeze_failures["MLQDS_without_geometry_tie_breaker"] = str(exc)

    if isinstance(primary_scores, torch.Tensor) and allocation_length_support_weight > 0.0:
        try:
            selection_segment_scores = (
                primary_selector_segment_scores
                if isinstance(primary_selector_segment_scores, torch.Tensor)
                else None
            )
            ablation_methods.append(
                learned_segment_frozen_method(
                    name="MLQDS_without_segment_length_support_allocation",
                    scores=primary_scores,
                    boundaries=selection_boundaries,
                    compression_ratio=float(config.model.compression_ratio),
                    segment_scores=selection_segment_scores,
                    segment_point_scores=primary_segment_scores,
                    points=selection_points,
                    learned_segment_geometry_gain_weight=float(
                        config.model.learned_segment_geometry_gain_weight
                    ),
                    learned_segment_allocation_length_support_weight=0.0,
                    learned_segment_allocation_weight_floor=allocation_weight_floor,
                    learned_segment_score_blend_weight=float(
                        config.model.learned_segment_score_blend_weight
                    ),
                    learned_segment_transfer_calibration_mode=str(
                        config.model.learned_segment_transfer_calibration_mode
                    ),
                    learned_segment_fairness_preallocation=bool(
                        config.model.learned_segment_fairness_preallocation
                    ),
                    learned_segment_length_repair_fraction=float(
                        config.model.learned_segment_length_repair_fraction
                    ),
                    learned_segment_length_repair_score_protection_fraction=(
                        repair_score_protection_fraction
                    ),
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            freeze_failures["MLQDS_without_segment_length_support_allocation"] = str(exc)

    if isinstance(primary_scores, torch.Tensor) and isinstance(
        primary_segment_scores, torch.Tensor
    ):
        try:
            neutral_segment_scores = neutral_segment_scores_for_ablation(primary_segment_scores)
            no_segment_selector_scores = blend_segment_support_scores(
                segment_scores=neutral_segment_scores,
                path_length_support_scores=(
                    primary_path_length_support_scores
                    if isinstance(primary_path_length_support_scores, torch.Tensor)
                    else None
                ),
                path_length_support_weight=float(
                    config.model.learned_segment_length_support_blend_weight
                ),
            )
            no_segment = learned_segment_frozen_method(
                name="MLQDS_without_segment_budget_head",
                scores=primary_scores,
                boundaries=selection_boundaries,
                compression_ratio=float(config.model.compression_ratio),
                segment_scores=no_segment_selector_scores,
                segment_point_scores=neutral_segment_scores,
                points=selection_points,
                learned_segment_geometry_gain_weight=float(
                    config.model.learned_segment_geometry_gain_weight
                ),
                learned_segment_allocation_length_support_weight=allocation_length_support_weight,
                learned_segment_allocation_weight_floor=allocation_weight_floor,
                learned_segment_score_blend_weight=float(
                    config.model.learned_segment_score_blend_weight
                ),
                learned_segment_transfer_calibration_mode=str(
                    config.model.learned_segment_transfer_calibration_mode
                ),
                learned_segment_fairness_preallocation=bool(
                    config.model.learned_segment_fairness_preallocation
                ),
                learned_segment_length_repair_fraction=float(
                    config.model.learned_segment_length_repair_fraction
                ),
                learned_segment_length_repair_score_protection_fraction=(
                    repair_score_protection_fraction
                ),
            )
            ablation_methods.append(no_segment)
            head_sensitivity["MLQDS_without_segment_budget_head"] = {
                **head_ablation_sensitivity(
                    primary_scores=primary_scores,
                    ablation_scores=primary_scores,
                    primary_raw_predictions=primary_raw_preds
                    if isinstance(primary_raw_preds, torch.Tensor)
                    else None,
                    ablation_raw_predictions=primary_raw_preds
                    if isinstance(primary_raw_preds, torch.Tensor)
                    else None,
                    primary_segment_scores=(
                        primary_selector_segment_scores
                        if isinstance(primary_selector_segment_scores, torch.Tensor)
                        else primary_segment_scores
                    ),
                    ablation_segment_scores=no_segment_selector_scores,
                    primary_mask=primary_mask,
                    ablation_mask=no_segment.retained_mask,
                ),
                "disabled_head_name": "segment_budget_target",
                "ablation_mode": "neutral_constant_segment_scores",
            }
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            freeze_failures["MLQDS_without_segment_budget_head"] = str(exc)

    if isinstance(primary_scores, torch.Tensor) and isinstance(
        primary_path_length_support_scores, torch.Tensor
    ):
        try:
            path_length_segment_method = learned_segment_frozen_method(
                name="MLQDS_path_length_support_segment_head_diagnostic",
                scores=primary_scores,
                boundaries=selection_boundaries,
                compression_ratio=float(config.model.compression_ratio),
                segment_scores=primary_path_length_support_scores,
                points=selection_points,
                learned_segment_geometry_gain_weight=float(
                    config.model.learned_segment_geometry_gain_weight
                ),
                learned_segment_allocation_length_support_weight=allocation_length_support_weight,
                learned_segment_allocation_weight_floor=allocation_weight_floor,
                learned_segment_score_blend_weight=float(
                    config.model.learned_segment_score_blend_weight
                ),
                learned_segment_transfer_calibration_mode=str(
                    config.model.learned_segment_transfer_calibration_mode
                ),
                learned_segment_fairness_preallocation=bool(
                    config.model.learned_segment_fairness_preallocation
                ),
                learned_segment_length_repair_fraction=float(
                    config.model.learned_segment_length_repair_fraction
                ),
                learned_segment_length_repair_score_protection_fraction=(
                    repair_score_protection_fraction
                ),
            )
            ablation_methods.append(path_length_segment_method)
            head_sensitivity["MLQDS_path_length_support_segment_head_diagnostic"] = {
                **head_ablation_sensitivity(
                    primary_scores=primary_scores,
                    ablation_scores=primary_scores,
                    primary_raw_predictions=primary_raw_preds
                    if isinstance(primary_raw_preds, torch.Tensor)
                    else None,
                    ablation_raw_predictions=primary_raw_preds
                    if isinstance(primary_raw_preds, torch.Tensor)
                    else None,
                    primary_segment_scores=(
                        primary_selector_segment_scores
                        if isinstance(primary_selector_segment_scores, torch.Tensor)
                        else primary_segment_scores
                        if isinstance(primary_segment_scores, torch.Tensor)
                        else None
                    ),
                    ablation_segment_scores=primary_path_length_support_scores,
                    primary_mask=primary_mask,
                    ablation_mask=path_length_segment_method.retained_mask,
                ),
                "diagnostic_only": True,
                "replacement_head_name": "path_length_support_target",
                "ablation_mode": "path_length_support_as_segment_scores",
            }
            path_length_allocation_method = learned_segment_frozen_method(
                name="MLQDS_path_length_support_allocation_only_diagnostic",
                scores=primary_scores,
                boundaries=selection_boundaries,
                compression_ratio=float(config.model.compression_ratio),
                segment_scores=primary_path_length_support_scores,
                segment_point_scores=primary_segment_scores,
                points=selection_points,
                learned_segment_geometry_gain_weight=float(
                    config.model.learned_segment_geometry_gain_weight
                ),
                learned_segment_allocation_length_support_weight=allocation_length_support_weight,
                learned_segment_allocation_weight_floor=allocation_weight_floor,
                learned_segment_score_blend_weight=float(
                    config.model.learned_segment_score_blend_weight
                ),
                learned_segment_transfer_calibration_mode=str(
                    config.model.learned_segment_transfer_calibration_mode
                ),
                learned_segment_fairness_preallocation=bool(
                    config.model.learned_segment_fairness_preallocation
                ),
                learned_segment_length_repair_fraction=float(
                    config.model.learned_segment_length_repair_fraction
                ),
                learned_segment_length_repair_score_protection_fraction=(
                    repair_score_protection_fraction
                ),
            )
            ablation_methods.append(path_length_allocation_method)
            head_sensitivity["MLQDS_path_length_support_allocation_only_diagnostic"] = {
                **head_ablation_sensitivity(
                    primary_scores=primary_scores,
                    ablation_scores=primary_scores,
                    primary_raw_predictions=primary_raw_preds
                    if isinstance(primary_raw_preds, torch.Tensor)
                    else None,
                    ablation_raw_predictions=primary_raw_preds
                    if isinstance(primary_raw_preds, torch.Tensor)
                    else None,
                    primary_segment_scores=(
                        primary_selector_segment_scores
                        if isinstance(primary_selector_segment_scores, torch.Tensor)
                        else primary_segment_scores
                        if isinstance(primary_segment_scores, torch.Tensor)
                        else None
                    ),
                    ablation_segment_scores=primary_path_length_support_scores,
                    primary_mask=primary_mask,
                    ablation_mask=path_length_allocation_method.retained_mask,
                ),
                "diagnostic_only": True,
                "replacement_head_name": "path_length_support_target",
                "ablation_mode": "path_length_support_allocation_only",
            }
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            freeze_failures["MLQDS_path_length_support_segment_head_diagnostic"] = str(exc)

    if (
        isinstance(primary_scores, torch.Tensor)
        and isinstance(primary_head_logits, torch.Tensor)
        and isinstance(primary_selector_segment_scores, torch.Tensor)
    ):
        try:
            behavior_raw_preds = raw_predictions_without_factorized_head(
                model=trained.model,
                head_logits=primary_head_logits,
                disabled_head_name="conditional_behavior_utility",
            )
            behavior_scores = scores_without_factorized_head(
                model=trained.model,
                head_logits=primary_head_logits,
                disabled_head_name="conditional_behavior_utility",
                boundaries=selection_boundaries,
                workload_type=workload_type,
                score_mode=config.model.mlqds_score_mode,
                score_temperature=float(config.model.mlqds_score_temperature),
                rank_confidence_weight=float(config.model.mlqds_rank_confidence_weight),
            )
            no_behavior = learned_segment_frozen_method(
                name="MLQDS_without_behavior_utility_head",
                scores=behavior_scores,
                boundaries=selection_boundaries,
                compression_ratio=float(config.model.compression_ratio),
                segment_scores=primary_selector_segment_scores,
                segment_point_scores=primary_segment_scores,
                points=selection_points,
                learned_segment_geometry_gain_weight=float(
                    config.model.learned_segment_geometry_gain_weight
                ),
                learned_segment_allocation_length_support_weight=allocation_length_support_weight,
                learned_segment_allocation_weight_floor=allocation_weight_floor,
                learned_segment_score_blend_weight=float(
                    config.model.learned_segment_score_blend_weight
                ),
                learned_segment_transfer_calibration_mode=str(
                    config.model.learned_segment_transfer_calibration_mode
                ),
                learned_segment_fairness_preallocation=bool(
                    config.model.learned_segment_fairness_preallocation
                ),
                learned_segment_length_repair_fraction=float(
                    config.model.learned_segment_length_repair_fraction
                ),
                learned_segment_length_repair_score_protection_fraction=(
                    repair_score_protection_fraction
                ),
            )
            ablation_methods.append(no_behavior)
            head_sensitivity["MLQDS_without_behavior_utility_head"] = {
                **head_ablation_sensitivity(
                    primary_scores=primary_scores,
                    ablation_scores=behavior_scores,
                    primary_raw_predictions=primary_raw_preds
                    if isinstance(primary_raw_preds, torch.Tensor)
                    else None,
                    ablation_raw_predictions=behavior_raw_preds,
                    primary_segment_scores=primary_selector_segment_scores,
                    ablation_segment_scores=primary_selector_segment_scores,
                    primary_mask=primary_mask,
                    ablation_mask=no_behavior.retained_mask,
                ),
                "disabled_head_name": "conditional_behavior_utility",
                "ablation_mode": "neutral_multiplicative_head",
            }
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            freeze_failures["MLQDS_without_behavior_utility_head"] = str(exc)

    query_prior_field = trained.feature_context.get("query_prior_field")
    if isinstance(query_prior_field, dict) and isinstance(primary_scores, torch.Tensor):
        try:
            prior_scores = (
                query_prior_predictability_scores(selection_points, query_prior_field)
                .detach()
                .cpu()
            )
            ablation_methods.append(
                learned_segment_frozen_method(
                    name="MLQDS_prior_field_only_score",
                    scores=prior_scores,
                    boundaries=selection_boundaries,
                    compression_ratio=float(config.model.compression_ratio),
                    points=selection_points,
                    learned_segment_geometry_gain_weight=float(
                        config.model.learned_segment_geometry_gain_weight
                    ),
                    learned_segment_allocation_length_support_weight=(
                        allocation_length_support_weight
                    ),
                    learned_segment_allocation_weight_floor=allocation_weight_floor,
                    learned_segment_score_blend_weight=float(
                        config.model.learned_segment_score_blend_weight
                    ),
                    learned_segment_transfer_calibration_mode=str(
                        config.model.learned_segment_transfer_calibration_mode
                    ),
                    learned_segment_fairness_preallocation=bool(
                        config.model.learned_segment_fairness_preallocation
                    ),
                    learned_segment_length_repair_fraction=float(
                        config.model.learned_segment_length_repair_fraction
                    ),
                    learned_segment_length_repair_score_protection_fraction=(
                        repair_score_protection_fraction
                    ),
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            freeze_failures["MLQDS_prior_field_only_score"] = str(exc)
        prior_ablation_fields = {
            "MLQDS_shuffled_prior_fields": shuffled_query_prior_field(
                query_prior_field,
                seed=int(seeds.eval_query_seed) + 72_003,
            ),
            "MLQDS_without_query_prior_features": zero_query_prior_field_like(query_prior_field),
        }
        for ablation_name, ablation_field in prior_ablation_fields.items():
            try:
                prior_sensitivity_key = (
                    "shuffled_prior_fields"
                    if ablation_name == "MLQDS_shuffled_prior_fields"
                    else "without_query_prior_features"
                )
                prior_feature_sensitivity = prior_feature_sample_sensitivity(
                    points=selection_points,
                    primary_prior_field=query_prior_field,
                    ablation_prior_field=ablation_field,
                )
                model_prior_sensitivity = model_prior_feature_sensitivity(
                    points=selection_points,
                    point_dim=int(getattr(trained.model, "point_dim", selection_points.shape[1])),
                    scaler=trained.scaler,
                    primary_prior_field=query_prior_field,
                    ablation_prior_field=ablation_field,
                    boundaries=selection_boundaries,
                )
                ablation_trained = training_outputs_with_query_prior_field(
                    trained,
                    ablation_field,
                )
                ablation_method = _mlqds_method(
                    name=ablation_name,
                    trained_outputs=ablation_trained,
                    workload=selection_workload,
                )
                ablation_mask = ablation_method.simplify(
                    selection_points,
                    selection_boundaries,
                    float(config.model.compression_ratio),
                )
                ablation_point_scores = getattr(ablation_method, "_score_cache", None)
                ablation_raw_preds = getattr(ablation_method, "_raw_pred_cache", None)
                ablation_head_logits = getattr(ablation_method, "_head_logit_cache", None)
                prior_sensitivity[prior_sensitivity_key] = (
                    prior_ablation_sensitivity_from_tensors(
                        sampled_prior_features=prior_feature_sensitivity,
                        model_prior_features=model_prior_sensitivity,
                        primary_scores=primary_scores,
                        ablation_scores=ablation_point_scores
                        if isinstance(ablation_point_scores, torch.Tensor)
                        else None,
                        primary_raw_predictions=primary_raw_preds
                        if isinstance(primary_raw_preds, torch.Tensor)
                        else None,
                        ablation_raw_predictions=ablation_raw_preds
                        if isinstance(ablation_raw_preds, torch.Tensor)
                        else None,
                        primary_head_logits=primary_head_logits
                        if isinstance(primary_head_logits, torch.Tensor)
                        else None,
                        ablation_head_logits=ablation_head_logits
                        if isinstance(ablation_head_logits, torch.Tensor)
                        else None,
                        primary_mask=primary_mask,
                        ablation_mask=ablation_mask,
                    )
                )
                ablation_methods.append(
                    FrozenMaskMethod(
                        name=ablation_name,
                        retained_mask=ablation_mask.detach().cpu(),
                    )
                )
            except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
                freeze_failures[ablation_name] = str(exc)

    if not ablation_methods:
        payload = {
            "available": False,
            "reason": "no_validation_ablations_frozen",
            "ablation_freeze_failures": freeze_failures,
            "selection_retained_decision_marginal_teacher": selection_marginal_teacher_summary,
            "separated_marginal_teacher_selector_diagnostic": (
                separated_teacher_selector_diagnostic
            ),
            "separated_marginal_teacher_hybrid_selector_diagnostics": (
                separated_teacher_hybrid_selector_diagnostics
            ),
        }
        if selection_selector_trace is not None:
            payload["selection_selector_trace_diagnostics"] = selection_selector_trace
        return payload

    ablation_scores: dict[str, MethodScore] = {}
    mask_diagnostics: dict[str, dict[str, Any]] = {}
    for method in ablation_methods:
        mask_diagnostics[method.name] = retained_mask_comparison(
            primary_mask=primary_mask,
            ablation_mask=method.retained_mask,
            expected_shape=primary_mask.shape,
        )
        ablation_scores[method.name] = score_method(
            method=method,
            points=selection_points,
            boundaries=selection_boundaries,
            typed_queries=selection_workload.typed_queries,
            workload_map=eval_workload_map,
            compression_ratio=config.model.compression_ratio,
            query_cache=selection_query_cache,
        )
    if separated_teacher_method_name in ablation_scores:
        teacher_score = ablation_scores[separated_teacher_method_name]
        separated_teacher_selector_diagnostic.update(
            {
                "query_useful_v1_score": float(teacher_score.query_useful_v1_score),
                "primary_query_useful_v1_delta": float(
                    primary_score.query_useful_v1_score - teacher_score.query_useful_v1_score
                ),
                "teacher_minus_primary_query_useful_v1": float(
                    teacher_score.query_useful_v1_score - primary_score.query_useful_v1_score
                ),
                "mask_diagnostics": mask_diagnostics.get(separated_teacher_method_name),
            }
        )
    hybrid_methods = separated_teacher_hybrid_selector_diagnostics.get("methods")
    if isinstance(hybrid_methods, dict):
        for method_name, method_diag in hybrid_methods.items():
            if not isinstance(method_diag, dict) or method_name not in ablation_scores:
                continue
            hybrid_score = ablation_scores[method_name]
            method_diag.update(
                {
                    "query_useful_v1_score": float(hybrid_score.query_useful_v1_score),
                    "primary_query_useful_v1_delta": float(
                        primary_score.query_useful_v1_score - hybrid_score.query_useful_v1_score
                    ),
                    "teacher_minus_primary_query_useful_v1": float(
                        hybrid_score.query_useful_v1_score - primary_score.query_useful_v1_score
                    ),
                    "mask_diagnostics": mask_diagnostics.get(method_name),
                }
            )

    payload = causality_ablation_diagnostics_payload(
        primary=primary_score,
        ablations=ablation_scores,
        mask_diagnostics=mask_diagnostics,
    )
    for name, tradeoff_diagnostics in payload["tradeoff_diagnostics"].items():
        if name in head_sensitivity:
            head_sensitivity[name]["query_useful_component_tradeoff"] = tradeoff_diagnostics
    payload.update(
        {
            "split": "checkpoint_selection",
            "diagnostic_only": True,
            "query_count": len(selection_workload.typed_queries),
            "ablation_freeze_failures": freeze_failures,
            "prior_sensitivity_diagnostics": prior_sensitivity,
            "head_ablation_sensitivity_diagnostics": head_sensitivity,
            "selection_retained_decision_marginal_teacher": selection_marginal_teacher_summary,
            "separated_marginal_teacher_selector_diagnostic": (
                separated_teacher_selector_diagnostic
            ),
            "separated_marginal_teacher_hybrid_selector_diagnostics": (
                separated_teacher_hybrid_selector_diagnostics
            ),
        }
    )
    if selection_selector_trace is not None:
        payload["selection_selector_trace_diagnostics"] = selection_selector_trace
    return payload
