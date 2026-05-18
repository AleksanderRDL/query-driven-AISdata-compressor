"""Causality and ablation diagnostic helpers for run reporting."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from learning.model_features import (
    WORKLOAD_BLIND_RANGE_V2_MODEL_DISABLED_PRIOR_FIELDS,
    WORKLOAD_BLIND_RANGE_V2_MODEL_PRIOR_TRANSFORM,
    build_query_free_point_features_for_dim,
)
from learning.outputs import TrainingOutputs
from learning.query_prior_fields import (
    QUERY_PRIOR_FIELD_NAMES,
    query_prior_field_metadata,
    sample_query_prior_fields,
)
from learning.targets.query_useful_v1 import QUERY_USEFUL_V1_HEAD_NAMES
from scoring.metrics import MethodScore
from scoring.query_useful_v1 import QUERY_USEFUL_V1_COMPONENT_WEIGHTS

PRIOR_ABLATION_SCORE_OUTPUT_SEMANTICS = "final_selector_score_after_mlqds_score_conversion"
PRIOR_ABLATION_DIAGNOSTIC_CHAIN = (
    "sampled_prior_features",
    "model_prior_features",
    "head_output",
    "raw_prediction",
    "score_output",
    "retained_mask",
)


def build_learned_slot_summary(
    selector_budget_diagnostics: dict[str, Any],
    compression_ratio: float,
    selector_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return learned-slot accounting without pretending budget rows are proof."""
    eval_diagnostics = selector_budget_diagnostics.get("eval")
    if not isinstance(eval_diagnostics, dict):
        return {
            "learned_controlled_retained_slots": None,
            "learned_controlled_retained_slot_fraction": None,
            "learned_slot_accounting_status": "missing_selector_diagnostics",
        }
    rows = eval_diagnostics.get("budget_rows")
    if not isinstance(rows, list) or not rows:
        return {
            "learned_controlled_retained_slots": None,
            "learned_controlled_retained_slot_fraction": None,
            "learned_slot_accounting_status": "missing_budget_rows",
        }
    selected_row = None
    for row in rows:
        if (
            isinstance(row, dict)
            and abs(float(row.get("compression_ratio", -1.0)) - float(compression_ratio)) <= 1e-9
        ):
            selected_row = row
            break
    if selected_row is None:
        selected_row = rows[0] if isinstance(rows[0], dict) else None
    if selected_row is None:
        return {
            "learned_controlled_retained_slots": None,
            "learned_controlled_retained_slot_fraction": None,
            "learned_slot_accounting_status": "invalid_budget_row",
        }
    planned_slots = int(selected_row.get("learned_slot_count", 0))
    planned_fraction = float(selected_row.get("learned_slot_fraction_of_budget", 0.0))
    summary = {
        "learned_controlled_retained_slots": planned_slots,
        "learned_controlled_retained_slot_fraction": planned_fraction,
        "total_retained_slot_budget": int(selected_row.get("total_budget_count", 0)),
        "minimal_skeleton_slot_cap": int(selected_row.get("minimal_skeleton_slot_cap", 0)),
        "no_fixed_85_percent_temporal_scaffold": bool(
            selected_row.get("no_fixed_85_percent_temporal_scaffold", False)
        ),
        "planned_learned_controlled_retained_slots": planned_slots,
        "planned_learned_controlled_retained_slot_fraction": planned_fraction,
        "learned_slot_accounting_status": "budget_level_accounting_only",
        "learned_slot_accounting_note": (
            "Counts planned learned-controlled selector budget. "
            "Per-retained-point skeleton-vs-learned attribution is not yet recorded."
        ),
    }
    if not isinstance(selector_trace, dict) or not selector_trace.get(
        "point_attribution_available"
    ):
        return summary

    actual_slots = int(selector_trace.get("learned_controlled_retained_slots", 0))
    actual_fraction = float(selector_trace.get("learned_controlled_retained_slot_fraction", 0.0))
    summary.update(
        {
            "learned_controlled_retained_slots": actual_slots,
            "learned_controlled_retained_slot_fraction": actual_fraction,
            "actual_learned_controlled_retained_slots": actual_slots,
            "actual_learned_controlled_retained_slot_fraction": actual_fraction,
            "skeleton_retained_count": int(selector_trace.get("skeleton_retained_count", 0)),
            "fallback_retained_count": int(selector_trace.get("fallback_retained_count", 0)),
            "unattributed_retained_count": int(
                selector_trace.get("unattributed_retained_count", 0)
            ),
            "trajectories_with_at_least_one_learned_decision": int(
                selector_trace.get("trajectories_with_at_least_one_learned_decision", 0)
            ),
            "trajectories_with_zero_learned_decisions": int(
                selector_trace.get("trajectories_with_zero_learned_decisions", 0)
            ),
            "segment_budget_entropy": float(selector_trace.get("segment_budget_entropy", 0.0)),
            "segment_budget_entropy_normalized": float(
                selector_trace.get("segment_budget_entropy_normalized", 0.0)
            ),
            "segments_with_learned_budget": int(
                selector_trace.get("segments_with_learned_budget", 0)
            ),
            "learned_slot_accounting_status": "point_attribution_available",
            "learned_slot_accounting_note": (
                "Counts actual retained points attributed to skeleton, learned segment allocation, "
                "or fallback fill after masks were frozen."
            ),
        }
    )
    if "retained_mask_matches_frozen_primary" in selector_trace:
        summary["selector_trace_retained_mask_matches_primary"] = bool(
            selector_trace.get("retained_mask_matches_frozen_primary")
        )
    return summary


def query_useful_delta(
    primary: MethodScore,
    ablations: dict[str, MethodScore],
    name: str,
) -> float | None:
    """Return primary minus ablation QueryUsefulV1 if the ablation exists."""
    ablation = ablations.get(name)
    if ablation is None:
        return None
    return float(primary.query_useful_v1_score - ablation.query_useful_v1_score)


def query_useful_component_delta_summary(
    *,
    primary: MethodScore,
    ablations: dict[str, MethodScore],
    top_k: int = 5,
) -> dict[str, Any]:
    """Return component-level QueryUsefulV1 deltas for causality ablations."""
    primary_components = dict(primary.query_useful_v1_components or {})
    if not primary_components:
        return {}
    summary: dict[str, Any] = {}
    limit = max(0, int(top_k))
    for name, ablation in sorted(ablations.items()):
        ablation_components = dict(ablation.query_useful_v1_components or {})
        if not ablation_components:
            summary[name] = {"available": False, "reason": "missing_ablation_components"}
            continue
        component_names = sorted(set(primary_components) | set(ablation_components))
        component_deltas: dict[str, float] = {}
        weighted_deltas: dict[str, float] = {}
        for component in component_names:
            delta = float(primary_components.get(component, 0.0)) - float(
                ablation_components.get(component, 0.0)
            )
            component_deltas[component] = delta
            weighted_deltas[component] = delta * float(
                QUERY_USEFUL_V1_COMPONENT_WEIGHTS.get(component, 0.0)
            )

        rows = [
            {
                "component": component,
                "component_delta": float(component_deltas[component]),
                "weighted_delta": float(weighted_deltas[component]),
            }
            for component in component_names
        ]
        positive_rows = sorted(rows, key=lambda row: float(row["weighted_delta"]), reverse=True)
        negative_rows = sorted(rows, key=lambda row: float(row["weighted_delta"]))
        query_delta = float(primary.query_useful_v1_score - ablation.query_useful_v1_score)
        weighted_sum = float(sum(weighted_deltas.values()))
        summary[name] = {
            "available": True,
            "query_useful_v1_delta": query_delta,
            "component_deltas": component_deltas,
            "weighted_component_deltas": weighted_deltas,
            "component_weighted_delta_sum": weighted_sum,
            "component_delta_residual": float(query_delta - weighted_sum),
            "top_positive_weighted_component_deltas": [
                row for row in positive_rows if float(row["weighted_delta"]) > 0.0
            ][:limit],
            "top_negative_weighted_component_deltas": [
                row for row in negative_rows if float(row["weighted_delta"]) < 0.0
            ][:limit],
        }
    return summary


def causality_ablation_tradeoff_summary(
    *,
    component_deltas: dict[str, Any],
    mask_diagnostics: dict[str, dict[str, Any]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Connect ablation mask movement to QueryUsefulV1 component movement."""

    def _numeric(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    def _row_list(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        rows: list[dict[str, Any]] = []
        for row in value:
            if isinstance(row, dict):
                rows.append(dict(row))
        return rows[: max(0, int(top_k))]

    summary: dict[str, Any] = {}
    ablation_names = sorted(set(component_deltas) | set(mask_diagnostics))
    for name in ablation_names:
        component_row = component_deltas.get(name)
        mask_row = mask_diagnostics.get(name, {})
        mask_available = (
            bool(mask_row.get("available", False)) if isinstance(mask_row, dict) else False
        )
        if not isinstance(component_row, dict) or not bool(component_row.get("available", False)):
            reason = "missing_component_delta"
            if isinstance(component_row, dict) and component_row.get("reason") is not None:
                reason = str(component_row.get("reason"))
            summary[name] = {
                "available": False,
                "reason": reason,
                "retained_mask_available": mask_available,
            }
            continue

        weighted_raw = component_row.get("weighted_component_deltas", {})
        weighted_values = [
            float(value)
            for value in (weighted_raw.values() if isinstance(weighted_raw, dict) else [])
            if not isinstance(value, bool) and isinstance(value, (int, float))
        ]
        positive_weighted_sum = float(sum(value for value in weighted_values if value > 0.0))
        negative_weighted_sum = float(sum(value for value in weighted_values if value < 0.0))
        absolute_weighted_sum = float(sum(abs(value) for value in weighted_values))

        changed_count = (
            _numeric(mask_row.get("retained_symmetric_difference_count"))
            if isinstance(mask_row, dict)
            else None
        )
        changed_denominator = (
            changed_count if changed_count is not None and changed_count > 0.0 else None
        )
        query_delta = float(component_row.get("query_useful_v1_delta", 0.0))
        top_positive = _row_list(component_row.get("top_positive_weighted_component_deltas"))
        top_negative = _row_list(component_row.get("top_negative_weighted_component_deltas"))
        if changed_denominator is None:
            delta_per_changed_decision = None
            positive_sum_per_changed_decision = None
            negative_sum_per_changed_decision = None
            tradeoff_status = (
                "retained_mask_unchanged"
                if mask_available
                else "component_delta_without_mask_diagnostics"
            )
        else:
            delta_per_changed_decision = float(query_delta / changed_denominator)
            positive_sum_per_changed_decision = float(positive_weighted_sum / changed_denominator)
            negative_sum_per_changed_decision = float(negative_weighted_sum / changed_denominator)
            if query_delta > 0.0:
                tradeoff_status = "mask_change_helped_primary_metric"
            elif query_delta < 0.0:
                tradeoff_status = "mask_change_hurt_primary_metric"
            else:
                tradeoff_status = "mask_change_neutral_primary_metric"

        summary[name] = {
            "available": True,
            "tradeoff_status": tradeoff_status,
            "query_useful_v1_delta": query_delta,
            "component_weighted_delta_sum": component_row.get("component_weighted_delta_sum"),
            "component_delta_residual": component_row.get("component_delta_residual"),
            "positive_weighted_component_delta_sum": positive_weighted_sum,
            "negative_weighted_component_delta_sum": negative_weighted_sum,
            "absolute_weighted_component_delta_sum": absolute_weighted_sum,
            "retained_mask_available": mask_available,
            "retained_mask_changed": mask_row.get("retained_mask_changed")
            if isinstance(mask_row, dict)
            else None,
            "retained_symmetric_difference_count": changed_count,
            "retained_mask_jaccard": mask_row.get("retained_mask_jaccard")
            if isinstance(mask_row, dict)
            else None,
            "retained_mask_hamming_fraction": (
                mask_row.get("retained_mask_hamming_fraction")
                if isinstance(mask_row, dict)
                else None
            ),
            "query_useful_v1_delta_per_changed_retained_decision": delta_per_changed_decision,
            "positive_weighted_component_delta_sum_per_changed_retained_decision": (
                positive_sum_per_changed_decision
            ),
            "negative_weighted_component_delta_sum_per_changed_retained_decision": (
                negative_sum_per_changed_decision
            ),
            "dominant_positive_weighted_component_delta": top_positive[0] if top_positive else None,
            "dominant_negative_weighted_component_delta": top_negative[0] if top_negative else None,
            "top_positive_weighted_component_deltas": top_positive,
            "top_negative_weighted_component_deltas": top_negative,
        }
    return summary


def causality_ablation_diagnostics_payload(
    *,
    primary: MethodScore,
    ablations: dict[str, MethodScore],
    mask_diagnostics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return reusable score, component, and mask diagnostics for ablations."""
    component_deltas = query_useful_component_delta_summary(
        primary=primary,
        ablations=ablations,
    )
    tradeoff_diagnostics = causality_ablation_tradeoff_summary(
        component_deltas=component_deltas,
        mask_diagnostics=mask_diagnostics,
    )
    return {
        "available": True,
        "primary_query_useful_v1_score": float(primary.query_useful_v1_score),
        "ablation_scores": {
            name: float(metrics.query_useful_v1_score)
            for name, metrics in sorted(ablations.items())
        },
        "ablation_query_useful_deltas": {
            name: float(primary.query_useful_v1_score - metrics.query_useful_v1_score)
            for name, metrics in sorted(ablations.items())
        },
        "component_deltas": component_deltas,
        "mask_diagnostics": mask_diagnostics,
        "tradeoff_diagnostics": tradeoff_diagnostics,
    }


LEARNING_CAUSALITY_MIN_MATERIAL_DELTA = 0.005
SHUFFLED_SCORE_DELTA_FRACTION_OF_UNIFORM_GAP_MIN = 0.60


def learning_causality_delta_gate_config(
    *,
    primary: MethodScore,
    uniform: MethodScore | None,
) -> dict[str, Any]:
    """Return material QueryUsefulV1 delta thresholds for learning-causality checks."""
    min_delta = float(LEARNING_CAUSALITY_MIN_MATERIAL_DELTA)
    thresholds = {
        "shuffled_scores_should_lose": min_delta,
        "untrained_model_should_lose": min_delta,
        "shuffled_prior_fields_should_lose": min_delta,
        "without_query_prior_features_should_lose": min_delta,
        "without_behavior_utility_head_should_lose": min_delta,
        "without_segment_budget_head_should_lose": min_delta,
        "prior_field_only_should_not_match_trained": min_delta,
    }
    uniform_gap = None
    if uniform is not None:
        uniform_gap = float(primary.query_useful_v1_score - uniform.query_useful_v1_score)
        if uniform_gap > 0.0:
            thresholds["shuffled_scores_should_lose"] = max(
                min_delta,
                float(SHUFFLED_SCORE_DELTA_FRACTION_OF_UNIFORM_GAP_MIN) * uniform_gap,
            )
    return {
        "min_material_query_useful_delta": min_delta,
        "shuffled_score_delta_fraction_of_uniform_gap_min": float(
            SHUFFLED_SCORE_DELTA_FRACTION_OF_UNIFORM_GAP_MIN
        ),
        "mlqds_uniform_query_useful_gap": uniform_gap,
        "thresholds": thresholds,
    }


def score_ablation_sensitivity(
    *,
    primary_scores: torch.Tensor | None,
    ablation_scores: torch.Tensor | None,
    primary_mask: torch.Tensor | None,
    ablation_mask: torch.Tensor | None,
) -> dict[str, Any]:
    """Return score- and mask-level sensitivity for a frozen ablation."""
    if primary_scores is None or ablation_scores is None:
        return {"available": False, "reason": "missing_scores"}
    primary = primary_scores.detach().cpu().float().flatten()
    ablation = ablation_scores.detach().cpu().float().flatten()
    if int(primary.numel()) == 0 or primary.shape != ablation.shape:
        return {
            "available": False,
            "reason": "score_shape_mismatch",
            "primary_score_count": int(primary.numel()),
            "ablation_score_count": int(ablation.numel()),
        }
    finite = torch.isfinite(primary) & torch.isfinite(ablation)
    if not bool(finite.any().item()):
        return {"available": False, "reason": "no_finite_scores"}
    primary_f = primary[finite]
    ablation_f = ablation[finite]
    delta = primary_f - ablation_f
    primary_std = float(primary_f.std(unbiased=False).item()) if int(primary_f.numel()) > 1 else 0.0
    ablation_std = (
        float(ablation_f.std(unbiased=False).item()) if int(ablation_f.numel()) > 1 else 0.0
    )

    topk_jaccard: float | None = None
    mask_diagnostics = retained_mask_comparison(
        primary_mask=primary_mask,
        ablation_mask=ablation_mask,
        expected_shape=primary.shape,
    )
    if primary_mask is not None and ablation_mask is not None:
        primary_bool = primary_mask.detach().cpu().bool().flatten()
        ablation_bool = ablation_mask.detach().cpu().bool().flatten()
        if primary_bool.shape == ablation_bool.shape == primary.shape:
            retained_count = int(primary_bool.sum().item())
            if retained_count > 0:
                k = min(retained_count, int(primary.numel()))
                primary_top = torch.zeros_like(primary_bool)
                ablation_top = torch.zeros_like(ablation_bool)
                primary_top[torch.topk(primary, k=k, largest=True).indices] = True
                ablation_top[torch.topk(ablation, k=k, largest=True).indices] = True
                top_intersection = int((primary_top & ablation_top).sum().item())
                top_union = int((primary_top | ablation_top).sum().item())
                topk_jaccard = float(top_intersection / max(1, top_union))

    return {
        "available": True,
        "score_count": int(primary.numel()),
        "finite_score_count": int(finite.sum().item()),
        "mean_abs_score_delta": float(delta.abs().mean().item()),
        "max_abs_score_delta": float(delta.abs().max().item()),
        "mean_signed_score_delta": float(delta.mean().item()),
        "primary_score_std": primary_std,
        "ablation_score_std": ablation_std,
        "retained_count": mask_diagnostics.get("primary_retained_count"),
        "retained_mask_changed": mask_diagnostics.get("retained_mask_changed"),
        "retained_mask_jaccard": mask_diagnostics.get("retained_mask_jaccard"),
        "retained_mask_hamming_fraction": mask_diagnostics.get("retained_mask_hamming_fraction"),
        "score_topk_jaccard_at_retained_count": topk_jaccard,
    }


def head_ablation_sensitivity(
    *,
    primary_scores: torch.Tensor | None,
    ablation_scores: torch.Tensor | None,
    primary_mask: torch.Tensor | None,
    ablation_mask: torch.Tensor | None,
    primary_raw_predictions: torch.Tensor | None = None,
    ablation_raw_predictions: torch.Tensor | None = None,
    primary_segment_scores: torch.Tensor | None = None,
    ablation_segment_scores: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Return score, raw-prediction, and segment-score sensitivity for one ablation."""
    diagnostics: dict[str, Any] = {
        "selector_score": score_ablation_sensitivity(
            primary_scores=primary_scores,
            ablation_scores=ablation_scores,
            primary_mask=primary_mask,
            ablation_mask=ablation_mask,
        )
    }
    if primary_raw_predictions is not None or ablation_raw_predictions is not None:
        diagnostics["raw_prediction"] = score_ablation_sensitivity(
            primary_scores=primary_raw_predictions,
            ablation_scores=ablation_raw_predictions,
            primary_mask=primary_mask,
            ablation_mask=ablation_mask,
        )
    if primary_segment_scores is not None or ablation_segment_scores is not None:
        diagnostics["segment_score"] = score_ablation_sensitivity(
            primary_scores=primary_segment_scores,
            ablation_scores=ablation_segment_scores,
            primary_mask=primary_mask,
            ablation_mask=ablation_mask,
        )
    return diagnostics


def _head_logit_matrix(head_logits: torch.Tensor) -> torch.Tensor:
    """Return head logits as [point, head] for sensitivity diagnostics."""
    logits = head_logits.detach().cpu().float()
    if logits.ndim == 3 and int(logits.shape[0]) == 1:
        logits = logits.squeeze(0)
    return logits


def head_output_sensitivity(
    *,
    primary_head_logits: torch.Tensor | None,
    ablation_head_logits: torch.Tensor | None,
) -> dict[str, Any]:
    """Return per-head logit and probability sensitivity for one model ablation."""
    if primary_head_logits is None or ablation_head_logits is None:
        return {"available": False, "reason": "missing_head_logits"}
    primary = _head_logit_matrix(primary_head_logits)
    ablation = _head_logit_matrix(ablation_head_logits)
    if primary.ndim != 2 or ablation.ndim != 2 or primary.shape != ablation.shape:
        return {
            "available": False,
            "reason": "head_logit_shape_mismatch",
            "primary_shape": list(primary.shape),
            "ablation_shape": list(ablation.shape),
        }
    head_names: list[str] = [str(name) for name in QUERY_USEFUL_V1_HEAD_NAMES]
    logit = _feature_matrix_sensitivity(
        primary=primary,
        ablation=ablation,
        feature_names=head_names,
        point_count=int(primary.shape[0]),
    )
    probability = _feature_matrix_sensitivity(
        primary=torch.sigmoid(primary),
        ablation=torch.sigmoid(ablation),
        feature_names=head_names,
        point_count=int(primary.shape[0]),
    )
    per_head: dict[str, dict[str, float | int | bool | None]] = {}
    logit_per_feature = logit.get("per_feature") if isinstance(logit, dict) else {}
    probability_per_feature = (
        probability.get("per_feature") if isinstance(probability, dict) else {}
    )
    for head_name in head_names:
        logit_row = (
            logit_per_feature.get(head_name, {}) if isinstance(logit_per_feature, dict) else {}
        )
        probability_row = (
            probability_per_feature.get(head_name, {})
            if isinstance(probability_per_feature, dict)
            else {}
        )
        per_head[head_name] = {
            "finite_count": logit_row.get("finite_count"),
            "mean_abs_logit_delta": logit_row.get("mean_abs_delta"),
            "max_abs_logit_delta": logit_row.get("max_abs_delta"),
            "mean_abs_probability_delta": probability_row.get("mean_abs_delta"),
            "max_abs_probability_delta": probability_row.get("max_abs_delta"),
            "primary_probability_mean": probability_row.get("primary_mean"),
            "ablation_probability_mean": probability_row.get("ablation_mean"),
        }
    return {
        "available": bool(logit.get("available") and probability.get("available")),
        "point_count": int(primary.shape[0]),
        "head_count": int(primary.shape[1]),
        "head_names": head_names,
        "head_logits_changed": bool(logit.get("sampled_inputs_changed", False)),
        "head_probabilities_changed": bool(probability.get("sampled_inputs_changed", False)),
        "mean_abs_head_logit_delta": logit.get("mean_abs_feature_delta"),
        "max_abs_head_logit_delta": logit.get("max_abs_feature_delta"),
        "mean_abs_head_probability_delta": probability.get("mean_abs_feature_delta"),
        "max_abs_head_probability_delta": probability.get("max_abs_feature_delta"),
        "logit": logit,
        "probability": probability,
        "per_head": per_head,
    }


def prior_ablation_sensitivity_payload(
    *,
    sampled_prior_features: dict[str, Any],
    model_prior_features: dict[str, Any],
    score_output: dict[str, Any],
    retained_mask: dict[str, Any],
    raw_prediction: dict[str, Any],
    head_output: dict[str, Any],
) -> dict[str, Any]:
    """Return one canonical prior-ablation sensitivity chain for run artifacts."""
    named_score_output = dict(score_output)
    named_score_output["semantics"] = PRIOR_ABLATION_SCORE_OUTPUT_SEMANTICS
    return {
        "available": True,
        "diagnostic_chain": list(PRIOR_ABLATION_DIAGNOSTIC_CHAIN),
        "sampled_prior_features": sampled_prior_features,
        "model_prior_features": model_prior_features,
        "score_output": named_score_output,
        "retained_mask": retained_mask,
        "raw_prediction": raw_prediction,
        "head_output": head_output,
    }


def prior_ablation_sensitivity_from_tensors(
    *,
    sampled_prior_features: dict[str, Any],
    model_prior_features: dict[str, Any],
    primary_scores: torch.Tensor | None,
    ablation_scores: torch.Tensor | None,
    primary_raw_predictions: torch.Tensor | None,
    ablation_raw_predictions: torch.Tensor | None,
    primary_head_logits: torch.Tensor | None,
    ablation_head_logits: torch.Tensor | None,
    primary_mask: torch.Tensor | None,
    ablation_mask: torch.Tensor | None,
) -> dict[str, Any]:
    """Return the full prior-ablation sensitivity chain from cached tensors."""
    return prior_ablation_sensitivity_payload(
        sampled_prior_features=sampled_prior_features,
        model_prior_features=model_prior_features,
        score_output=score_ablation_sensitivity(
            primary_scores=primary_scores,
            ablation_scores=ablation_scores,
            primary_mask=primary_mask,
            ablation_mask=ablation_mask,
        ),
        retained_mask=retained_mask_comparison(
            primary_mask=primary_mask,
            ablation_mask=ablation_mask,
            expected_shape=primary_scores.shape if primary_scores is not None else None,
        ),
        raw_prediction=score_ablation_sensitivity(
            primary_scores=primary_raw_predictions,
            ablation_scores=ablation_raw_predictions,
            primary_mask=primary_mask,
            ablation_mask=ablation_mask,
        ),
        head_output=head_output_sensitivity(
            primary_head_logits=primary_head_logits,
            ablation_head_logits=ablation_head_logits,
        ),
    )


def training_outputs_with_query_prior_field(
    trained: TrainingOutputs,
    query_prior_field: dict[str, Any],
) -> TrainingOutputs:
    """Return training outputs with a swapped query-prior field and matching metadata."""
    return replace(
        trained,
        feature_context={
            **trained.feature_context,
            "query_prior_field": query_prior_field,
            "query_prior_field_metadata": query_prior_field_metadata(query_prior_field),
        },
    )


def retained_mask_comparison(
    *,
    primary_mask: torch.Tensor | None,
    ablation_mask: torch.Tensor | None,
    expected_shape: torch.Size | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Return retained-mask overlap diagnostics for a frozen ablation."""
    if primary_mask is None or ablation_mask is None:
        return {"available": False, "reason": "missing_masks"}
    primary_bool = primary_mask.detach().cpu().bool().flatten()
    ablation_bool = ablation_mask.detach().cpu().bool().flatten()
    if expected_shape is not None:
        expected_numel = 1
        for dim in tuple(expected_shape):
            expected_numel *= int(dim)
        if (
            int(primary_bool.numel()) != expected_numel
            or int(ablation_bool.numel()) != expected_numel
        ):
            return {
                "available": False,
                "reason": "mask_shape_mismatch",
                "primary_mask_count": int(primary_bool.numel()),
                "ablation_mask_count": int(ablation_bool.numel()),
                "expected_mask_count": expected_numel,
            }
    if primary_bool.shape != ablation_bool.shape:
        return {
            "available": False,
            "reason": "mask_shape_mismatch",
            "primary_mask_count": int(primary_bool.numel()),
            "ablation_mask_count": int(ablation_bool.numel()),
        }
    intersection = int((primary_bool & ablation_bool).sum().item())
    union = int((primary_bool | ablation_bool).sum().item())
    primary_count = int(primary_bool.sum().item())
    ablation_count = int(ablation_bool.sum().item())
    symmetric_difference = int((primary_bool != ablation_bool).sum().item())
    return {
        "available": True,
        "primary_retained_count": primary_count,
        "ablation_retained_count": ablation_count,
        "retained_intersection_count": intersection,
        "retained_union_count": union,
        "retained_symmetric_difference_count": symmetric_difference,
        "retained_mask_changed": bool(symmetric_difference > 0),
        "retained_mask_jaccard": float(intersection / max(1, union)),
        "retained_mask_hamming_fraction": float(
            symmetric_difference / max(1, int(primary_bool.numel()))
        ),
    }


def prior_feature_sample_sensitivity(
    *,
    points: torch.Tensor,
    primary_prior_field: dict[str, Any] | None,
    ablation_prior_field: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return sampled query-prior feature sensitivity at eval-compression points."""
    if primary_prior_field is None:
        return {"available": False, "reason": "missing_primary_prior_field"}
    primary = sample_query_prior_fields(points, primary_prior_field).detach().cpu().float()
    ablation = sample_query_prior_fields(points, ablation_prior_field).detach().cpu().float()
    return _feature_matrix_sensitivity(
        primary=primary,
        ablation=ablation,
        feature_names=QUERY_PRIOR_FIELD_NAMES,
        point_count=int(points.shape[0]),
        primary_prior_field=primary_prior_field,
        points=points,
    )


def _feature_matrix_sensitivity(
    *,
    primary: torch.Tensor,
    ablation: torch.Tensor,
    feature_names: tuple[str, ...] | list[str],
    point_count: int,
    primary_prior_field: dict[str, Any] | None = None,
    points: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Return feature-delta diagnostics for aligned primary/ablation matrices."""
    if int(primary.numel()) == 0 or primary.shape != ablation.shape:
        return {
            "available": False,
            "reason": "feature_shape_mismatch",
            "primary_shape": list(primary.shape),
            "ablation_shape": list(ablation.shape),
        }
    finite = torch.isfinite(primary) & torch.isfinite(ablation)
    if not bool(finite.any().item()):
        return {"available": False, "reason": "no_finite_sampled_features"}
    delta = primary - ablation
    finite_delta = delta[finite]
    per_feature: dict[str, dict[str, float | int]] = {}
    named_features = list(feature_names)
    for idx in range(int(primary.shape[1])):
        name = named_features[idx] if idx < len(named_features) else f"feature_{idx}"
        primary_col = primary[:, idx]
        ablation_col = ablation[:, idx]
        col_finite = torch.isfinite(primary_col) & torch.isfinite(ablation_col)
        if not bool(col_finite.any().item()):
            per_feature[name] = {"finite_count": 0}
            continue
        primary_f = primary_col[col_finite]
        ablation_f = ablation_col[col_finite]
        col_delta = primary_f - ablation_f
        per_feature[name] = {
            "finite_count": int(col_finite.sum().item()),
            "mean_abs_delta": float(col_delta.abs().mean().item()),
            "max_abs_delta": float(col_delta.abs().max().item()),
            "primary_mean": float(primary_f.mean().item()),
            "ablation_mean": float(ablation_f.mean().item()),
            "primary_std": float(primary_f.std(unbiased=False).item())
            if int(primary_f.numel()) > 1
            else 0.0,
            "ablation_std": float(ablation_f.std(unbiased=False).item())
            if int(ablation_f.numel()) > 1
            else 0.0,
            "primary_nonzero_fraction": float((primary_f.abs() > 1e-12).float().mean().item()),
            "ablation_nonzero_fraction": float((ablation_f.abs() > 1e-12).float().mean().item()),
        }
    primary_flat = primary[torch.isfinite(primary)]
    ablation_flat = ablation[torch.isfinite(ablation)]
    outside_extent_fraction: float | None = None
    extent = primary_prior_field.get("extent") if isinstance(primary_prior_field, dict) else None
    if isinstance(extent, dict) and points is not None and int(points.shape[0]) > 0:
        lat = points[:, 1].detach().cpu().float()
        lon = points[:, 2].detach().cpu().float()
        outside = (
            (lat < float(extent.get("lat_min", -float("inf"))))
            | (lat > float(extent.get("lat_max", float("inf"))))
            | (lon < float(extent.get("lon_min", -float("inf"))))
            | (lon > float(extent.get("lon_max", float("inf"))))
        )
        outside_extent_fraction = float(outside.float().mean().item())
    return {
        "available": True,
        "point_count": int(point_count),
        "feature_count": int(primary.shape[1]),
        "finite_value_count": int(finite.sum().item()),
        "sampled_inputs_changed": bool(float(finite_delta.abs().max().item()) > 1e-9),
        "mean_abs_feature_delta": float(finite_delta.abs().mean().item()),
        "max_abs_feature_delta": float(finite_delta.abs().max().item()),
        "mean_signed_feature_delta": float(finite_delta.mean().item()),
        "primary_feature_mean": float(primary_flat.mean().item())
        if int(primary_flat.numel()) > 0
        else 0.0,
        "ablation_feature_mean": float(ablation_flat.mean().item())
        if int(ablation_flat.numel()) > 0
        else 0.0,
        "primary_feature_std": (
            float(primary_flat.std(unbiased=False).item()) if int(primary_flat.numel()) > 1 else 0.0
        ),
        "ablation_feature_std": (
            float(ablation_flat.std(unbiased=False).item())
            if int(ablation_flat.numel()) > 1
            else 0.0
        ),
        "primary_nonzero_fraction": float((primary.abs() > 1e-12).float().mean().item()),
        "ablation_nonzero_fraction": float((ablation.abs() > 1e-12).float().mean().item()),
        "points_outside_prior_extent_fraction": outside_extent_fraction,
        "per_feature": per_feature,
    }


def model_prior_feature_sensitivity(
    *,
    points: torch.Tensor,
    point_dim: int,
    scaler: Any,
    primary_prior_field: dict[str, Any] | None,
    ablation_prior_field: dict[str, Any] | None,
    boundaries: list[tuple[int, int]] | None = None,
    trajectory_mmsis: list[int] | None = None,
) -> dict[str, Any]:
    """Return prior-feature sensitivity at the actual model-input and scaler levels."""
    if primary_prior_field is None:
        return {"available": False, "reason": "missing_primary_prior_field"}
    prior_dim = len(QUERY_PRIOR_FIELD_NAMES)
    point_dim_int = int(point_dim)
    if point_dim_int < prior_dim:
        return {
            "available": False,
            "reason": "point_dim_smaller_than_prior_dim",
            "point_dim": point_dim_int,
            "prior_feature_count": prior_dim,
        }
    try:
        primary_model_points = build_query_free_point_features_for_dim(
            points,
            point_dim_int,
            boundaries=boundaries,
            trajectory_mmsis=trajectory_mmsis,
            query_prior_field=primary_prior_field,
        )
        ablation_model_points = build_query_free_point_features_for_dim(
            points,
            point_dim_int,
            boundaries=boundaries,
            trajectory_mmsis=trajectory_mmsis,
            query_prior_field=ablation_prior_field,
        )
    except Exception as exc:
        return {
            "available": False,
            "reason": "model_point_feature_build_failed",
            "error": str(exc),
            "point_dim": point_dim_int,
        }
    if primary_model_points.shape != ablation_model_points.shape:
        return {
            "available": False,
            "reason": "model_point_feature_shape_mismatch",
            "primary_shape": list(primary_model_points.shape),
            "ablation_shape": list(ablation_model_points.shape),
        }
    try:
        primary_normalized = scaler.transform_points(primary_model_points)
        ablation_normalized = scaler.transform_points(ablation_model_points)
    except Exception as exc:
        return {
            "available": False,
            "reason": "model_point_feature_scaling_failed",
            "error": str(exc),
            "point_dim": point_dim_int,
        }
    prior_slice = slice(-prior_dim, None)
    model_prior_features = _feature_matrix_sensitivity(
        primary=primary_model_points[:, prior_slice].detach().cpu().float(),
        ablation=ablation_model_points[:, prior_slice].detach().cpu().float(),
        feature_names=QUERY_PRIOR_FIELD_NAMES,
        point_count=int(points.shape[0]),
    )
    normalized_prior_features = _feature_matrix_sensitivity(
        primary=primary_normalized[:, prior_slice].detach().cpu().float(),
        ablation=ablation_normalized[:, prior_slice].detach().cpu().float(),
        feature_names=QUERY_PRIOR_FIELD_NAMES,
        point_count=int(points.shape[0]),
    )
    scaler_min = getattr(scaler, "point_min", None)
    scaler_max = getattr(scaler, "point_max", None)
    scaler_prior_ranges: dict[str, float] = {}
    if isinstance(scaler_min, torch.Tensor) and isinstance(scaler_max, torch.Tensor):
        min_prior = scaler_min.detach().cpu().float()[prior_slice]
        max_prior = scaler_max.detach().cpu().float()[prior_slice]
        ranges = torch.clamp(max_prior - min_prior, min=0.0)
        for idx, name in enumerate(QUERY_PRIOR_FIELD_NAMES):
            if idx < int(ranges.numel()):
                scaler_prior_ranges[name] = float(ranges[idx].item())
    return {
        "available": bool(
            model_prior_features.get("available") and normalized_prior_features.get("available")
        ),
        "point_dim": point_dim_int,
        "prior_feature_count": prior_dim,
        "disabled_prior_fields": list(WORKLOAD_BLIND_RANGE_V2_MODEL_DISABLED_PRIOR_FIELDS),
        "model_prior_feature_transform": WORKLOAD_BLIND_RANGE_V2_MODEL_PRIOR_TRANSFORM,
        "model_input_prior_features": model_prior_features,
        "normalized_model_prior_features": normalized_prior_features,
        "scaler_prior_feature_ranges": scaler_prior_ranges,
    }


def prior_sample_gate_failures(prior_sensitivity_diagnostics: dict[str, Any]) -> list[str]:
    """Return failures showing prior-feature ablations did not exercise useful inputs."""
    shuffled = prior_sensitivity_diagnostics.get("shuffled_prior_fields")
    if not isinstance(shuffled, dict):
        return []
    sampled = shuffled.get("sampled_prior_features")
    if not isinstance(sampled, dict) or not sampled.get("available"):
        return []
    failures: list[str] = []
    primary_nonzero = float(sampled.get("primary_nonzero_fraction") or 0.0)
    if primary_nonzero <= 1e-6:
        failures.append("sampled_query_prior_features_all_zero")
    if not bool(sampled.get("sampled_inputs_changed", False)):
        failures.append("shuffled_prior_fields_did_not_change_sampled_inputs")
    model_prior = shuffled.get("model_prior_features")
    if isinstance(model_prior, dict):
        model_input = model_prior.get("model_input_prior_features")
        if isinstance(model_input, dict) and model_input.get("available"):
            if not bool(model_input.get("sampled_inputs_changed", False)):
                failures.append("shuffled_prior_fields_did_not_change_model_inputs")
        normalized = model_prior.get("normalized_model_prior_features")
        if isinstance(normalized, dict) and normalized.get("available"):
            if not bool(normalized.get("sampled_inputs_changed", False)):
                failures.append("shuffled_prior_fields_did_not_change_normalized_model_inputs")
    outside_fraction = sampled.get("points_outside_prior_extent_fraction")
    if outside_fraction is not None and float(outside_fraction) > 0.50:
        failures.append("eval_points_mostly_outside_query_prior_extent")
    return failures
