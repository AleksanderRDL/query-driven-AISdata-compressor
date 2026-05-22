"""Causality and ablation diagnostic helpers for run reporting."""

from __future__ import annotations

from typing import Any

from orchestration.causality_sensitivity import (
    PRIOR_ABLATION_DIAGNOSTIC_CHAIN,
    PRIOR_ABLATION_SCORE_OUTPUT_SEMANTICS,
    head_ablation_sensitivity,
    head_output_sensitivity,
    marginal_row_delta_path_diagnostics,
    model_prior_feature_sensitivity,
    prior_ablation_sensitivity_from_tensors,
    prior_ablation_sensitivity_payload,
    prior_feature_sample_sensitivity,
    prior_sample_gate_failures,
    retained_mask_comparison,
    score_ablation_sensitivity,
    score_rank_margin_boundary_diagnostics,
    training_outputs_with_query_prior_field,
)
from scoring.metrics import MethodScore
from scoring.query_local_utility import QUERY_LOCAL_UTILITY_COMPONENT_WEIGHTS

__all__ = [
    "PRIOR_ABLATION_DIAGNOSTIC_CHAIN",
    "PRIOR_ABLATION_SCORE_OUTPUT_SEMANTICS",
    "head_ablation_sensitivity",
    "head_output_sensitivity",
    "marginal_row_delta_path_diagnostics",
    "model_prior_feature_sensitivity",
    "prior_ablation_sensitivity_from_tensors",
    "prior_ablation_sensitivity_payload",
    "prior_feature_sample_sensitivity",
    "prior_sample_gate_failures",
    "retained_mask_comparison",
    "score_ablation_sensitivity",
    "score_rank_margin_boundary_diagnostics",
    "training_outputs_with_query_prior_field",
]


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


def query_local_utility_delta(
    primary: MethodScore,
    ablations: dict[str, MethodScore],
    name: str,
) -> float | None:
    """Return primary minus ablation QueryLocalUtility if the ablation exists."""
    ablation = ablations.get(name)
    if ablation is None:
        return None
    return float(primary.query_local_utility_score - ablation.query_local_utility_score)


def query_local_utility_component_delta_summary(
    *,
    primary: MethodScore,
    ablations: dict[str, MethodScore],
    top_k: int = 5,
) -> dict[str, Any]:
    """Return component-level QueryLocalUtility deltas for causality ablations."""
    primary_components = dict(primary.query_local_utility_components or {})
    if not primary_components:
        return {}
    summary: dict[str, Any] = {}
    limit = max(0, int(top_k))
    for name, ablation in sorted(ablations.items()):
        ablation_components = dict(ablation.query_local_utility_components or {})
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
                QUERY_LOCAL_UTILITY_COMPONENT_WEIGHTS.get(component, 0.0)
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
        query_delta = float(primary.query_local_utility_score - ablation.query_local_utility_score)
        weighted_sum = float(sum(weighted_deltas.values()))
        summary[name] = {
            "available": True,
            "query_local_utility_delta": query_delta,
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
    """Connect ablation mask movement to QueryLocalUtility component movement."""

    def _numeric(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    def _row_list(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        rows = [dict(row) for row in value if isinstance(row, dict)]
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
        query_delta = float(component_row.get("query_local_utility_delta", 0.0))
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
            "query_local_utility_delta": query_delta,
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
            "query_local_utility_delta_per_changed_retained_decision": delta_per_changed_decision,
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
    component_deltas = query_local_utility_component_delta_summary(
        primary=primary,
        ablations=ablations,
    )
    tradeoff_diagnostics = causality_ablation_tradeoff_summary(
        component_deltas=component_deltas,
        mask_diagnostics=mask_diagnostics,
    )
    return {
        "available": True,
        "primary_query_local_utility_score": float(primary.query_local_utility_score),
        "ablation_scores": {
            name: float(metrics.query_local_utility_score)
            for name, metrics in sorted(ablations.items())
        },
        "ablation_query_local_utility_deltas": {
            name: float(primary.query_local_utility_score - metrics.query_local_utility_score)
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
    """Return material QueryLocalUtility delta thresholds for learning-causality checks."""
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
        uniform_gap = float(primary.query_local_utility_score - uniform.query_local_utility_score)
        if uniform_gap > 0.0:
            thresholds["shuffled_scores_should_lose"] = max(
                min_delta,
                float(SHUFFLED_SCORE_DELTA_FRACTION_OF_UNIFORM_GAP_MIN) * uniform_gap,
            )
    return {
        "min_material_query_local_utility_delta": min_delta,
        "shuffled_score_delta_fraction_of_uniform_gap_min": float(
            SHUFFLED_SCORE_DELTA_FRACTION_OF_UNIFORM_GAP_MIN
        ),
        "mlqds_uniform_query_local_utility_gap": uniform_gap,
        "thresholds": thresholds,
    }
