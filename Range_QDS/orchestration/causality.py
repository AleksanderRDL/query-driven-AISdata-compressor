"""Causality and ablation diagnostic helpers for run reporting."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import permutations
from typing import Any

import torch

from learning.model_features import (
    WORKLOAD_BLIND_RANGE_MODEL_DISABLED_PRIOR_FIELDS,
    WORKLOAD_BLIND_RANGE_MODEL_PRIOR_TRANSFORM,
    build_query_free_point_features_for_dim,
)
from learning.outputs import TrainingOutputs
from learning.query_prior_fields import (
    QUERY_PRIOR_FIELD_NAMES,
    query_prior_field_metadata,
    sample_query_prior_fields,
)
from learning.targets.query_local_utility import QUERY_LOCAL_UTILITY_HEAD_NAMES
from scoring.metrics import MethodScore
from scoring.query_local_utility import QUERY_LOCAL_UTILITY_COMPONENT_WEIGHTS

PRIOR_ABLATION_SCORE_OUTPUT_SEMANTICS = "final_selector_score_after_mlqds_score_conversion"
PRIOR_ABLATION_DIAGNOSTIC_CHAIN = (
    "sampled_prior_features",
    "model_prior_features",
    "head_output",
    "raw_prediction",
    "score_output",
    "marginal_row_delta_path",
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


def _rankdata(values: list[float]) -> list[float]:
    """Return 1-based average ranks for ascending values."""
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = float((cursor + 1 + end) / 2.0)
        for idx in range(cursor, end):
            ranks[ordered[idx][0]] = average_rank
        cursor = end
    return ranks


def _spearman_from_pairs(pairs: list[tuple[float, float]]) -> float | None:
    finite_pairs = [
        (float(left), float(right))
        for left, right in pairs
        if math.isfinite(float(left)) and math.isfinite(float(right))
    ]
    if len(finite_pairs) < 2:
        return None
    left_ranks = _rankdata([left for left, _right in finite_pairs])
    right_ranks = _rankdata([right for _left, right in finite_pairs])
    left = torch.tensor(left_ranks, dtype=torch.float32)
    right = torch.tensor(right_ranks, dtype=torch.float32)
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denom = left_centered.norm() * right_centered.norm()
    if float(denom.item()) <= 1e-12:
        return None
    return float((left_centered * right_centered).sum().item() / float(denom.item()))


def _topk_boundary_diagnostics(
    *,
    primary: torch.Tensor,
    delta: torch.Tensor,
    retained_count: int,
) -> dict[str, Any]:
    point_count = int(primary.numel())
    k = min(max(0, int(retained_count)), point_count)
    if k <= 0:
        return {"available": False, "reason": "empty_retained_count"}
    if k >= point_count:
        return {"available": False, "reason": "retained_count_covers_all_points"}
    order = torch.argsort(primary, descending=True)
    topk = torch.zeros((point_count,), dtype=torch.bool)
    topk[order[:k]] = True
    kth_score = float(primary[order[k - 1]].item())
    next_score = float(primary[order[k]].item())
    margin = float(kth_score - next_score)
    max_abs_delta = float(delta.abs().max().item()) if point_count > 0 else 0.0
    ratio = None if margin <= 1e-12 else float(max_abs_delta / margin)

    non_top_gap = kth_score - primary
    top_gap = primary - next_score
    non_top_positive_cross = (~topk) & (delta > 0.0) & (delta >= non_top_gap)
    top_negative_cross = topk & (delta < 0.0) & ((-delta) >= top_gap)
    near_boundary = torch.zeros((point_count,), dtype=torch.bool)
    near_width = max(1, min(point_count, k))
    low = max(0, k - near_width)
    high = min(point_count, k + near_width)
    near_boundary[order[low:high]] = True

    return {
        "available": True,
        "retained_count": int(k),
        "point_count": point_count,
        "kth_primary_score": kth_score,
        "next_primary_score": next_score,
        "topk_boundary_margin": margin,
        "max_abs_score_delta": max_abs_delta,
        "max_abs_score_delta_to_topk_boundary_margin": ratio,
        "non_topk_positive_delta_ge_gap_count": int(non_top_positive_cross.sum().item()),
        "topk_negative_delta_ge_gap_count": int(top_negative_cross.sum().item()),
        "near_boundary_count": int(near_boundary.sum().item()),
        "near_boundary_mean_abs_score_delta": float(delta[near_boundary].abs().mean().item())
        if bool(near_boundary.any().item())
        else None,
        "score_delta_crosses_topk_boundary": bool(
            non_top_positive_cross.any().item() or top_negative_cross.any().item()
        ),
    }


def _actual_retained_score_boundary(
    *,
    primary: torch.Tensor,
    primary_mask: torch.Tensor | None,
) -> dict[str, Any]:
    if primary_mask is None:
        return {"available": False, "reason": "missing_primary_mask"}
    mask = primary_mask.detach().cpu().bool().flatten()
    if mask.shape != primary.shape:
        return {"available": False, "reason": "mask_shape_mismatch"}
    retained = primary[mask]
    removed = primary[~mask]
    if int(retained.numel()) <= 0 or int(removed.numel()) <= 0:
        return {"available": False, "reason": "empty_retained_or_removed_partition"}
    min_retained = float(retained.min().item())
    max_removed = float(removed.max().item())
    margin = float(min_retained - max_removed)
    return {
        "available": True,
        "diagnostic_only": True,
        "semantics": "not_a_hard_boundary_for_segmented_selector",
        "retained_min_primary_score": min_retained,
        "removed_max_primary_score": max_removed,
        "actual_retained_score_margin": margin,
        "actual_retained_scores_are_threshold_separable": bool(margin >= 0.0),
    }


def _selector_trace_marginal_rows(selector_trace: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(selector_trace, dict):
        return []
    alignment = selector_trace.get("retained_decision_marginal_query_local_utility_alignment")
    if not isinstance(alignment, dict):
        return []
    rows = alignment.get("rows")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _row_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _row_score_delta_alignment(
    *,
    rows: list[dict[str, Any]],
    primary: torch.Tensor,
    ablation: torch.Tensor,
    max_rows: int,
) -> dict[str, Any]:
    enriched: list[dict[str, Any]] = []
    pairs: list[tuple[float, float]] = []
    for row in rows:
        point_index = row.get("point_index")
        if isinstance(point_index, bool) or not isinstance(point_index, int):
            continue
        idx = int(point_index)
        if idx < 0 or idx >= int(primary.numel()):
            continue
        marginal = _row_float(row, "marginal_query_local_utility")
        if marginal is None:
            continue
        primary_score = float(primary[idx].item())
        ablation_score = float(ablation[idx].item())
        score_delta = float(primary_score - ablation_score)
        pairs.append((score_delta, marginal))
        enriched.append(
            {
                "point_index": idx,
                "decision": row.get("decision"),
                "source": row.get("source"),
                "marginal_query_local_utility": marginal,
                "primary_score": primary_score,
                "ablation_score": ablation_score,
                "score_delta": score_delta,
                "score_delta_helpful_for_marginal": bool(score_delta > 0.0),
                "marginal_rank_fraction": _row_float(
                    row,
                    "marginal_query_local_utility_candidate_rank_fraction",
                ),
                "selector_score_rank_fraction": _row_float(
                    row,
                    "selector_score_candidate_rank_fraction",
                ),
                "segment_score_rank_fraction": _row_float(
                    row,
                    "segment_score_candidate_rank_fraction",
                ),
                "failure_buckets": list(row.get("failure_buckets") or []),
            }
        )
    if not enriched:
        return {"available": False, "reason": "missing_aligned_marginal_rows"}

    ordered = sorted(
        enriched,
        key=lambda row: (
            -float(row["marginal_query_local_utility"]),
            int(row["point_index"]),
        ),
    )
    top_count = max(1, math.ceil(0.25 * len(ordered)))
    bottom_count = max(1, math.ceil(0.25 * len(ordered)))
    top_rows = ordered[:top_count]
    bottom_rows = ordered[-bottom_count:]
    missed_rows = [
        row
        for row in ordered
        if row.get("decision") == "removed_addition_gain"
        and (
            row.get("marginal_rank_fraction") is None
            or float(row.get("marginal_rank_fraction") or 1.0) <= 0.25
        )
    ]
    under_ranked_rows = [
        row
        for row in ordered
        if "high_marginal_under_ranked_by_scores" in set(row.get("failure_buckets") or [])
    ]

    def _mean_delta(local_rows: list[dict[str, Any]]) -> float | None:
        if not local_rows:
            return None
        return float(sum(float(row["score_delta"]) for row in local_rows) / len(local_rows))

    def _positive_fraction(local_rows: list[dict[str, Any]]) -> float | None:
        if not local_rows:
            return None
        return float(
            sum(1 for row in local_rows if float(row["score_delta"]) > 0.0) / len(local_rows)
        )

    top_mean = _mean_delta(top_rows)
    bottom_mean = _mean_delta(bottom_rows)
    missed_mean = _mean_delta(missed_rows)
    under_ranked_mean = _mean_delta(under_ranked_rows)
    spearman = _spearman_from_pairs(pairs)
    if top_mean is not None and top_mean <= 0.0:
        classification = "prior_delta_non_positive_for_top_marginal_rows"
    elif missed_rows and missed_mean is not None and missed_mean <= 0.0:
        classification = "prior_delta_non_positive_for_missed_high_marginal_rows"
    elif under_ranked_rows and under_ranked_mean is not None and under_ranked_mean <= 0.0:
        classification = "prior_delta_non_positive_for_under_ranked_high_marginal_rows"
    elif spearman is not None and spearman < 0.0:
        classification = "prior_delta_globally_wrong_way_for_marginal_rows"
    else:
        classification = "prior_delta_not_obviously_wrong_way_for_marginal_rows"

    return {
        "available": True,
        "diagnostic_only": True,
        "row_count": len(enriched),
        "score_delta_to_marginal_spearman": spearman,
        "top_marginal_row_count": len(top_rows),
        "top_marginal_mean_score_delta": top_mean,
        "top_marginal_positive_score_delta_fraction": _positive_fraction(top_rows),
        "bottom_marginal_mean_score_delta": bottom_mean,
        "top_minus_bottom_mean_score_delta": None
        if top_mean is None or bottom_mean is None
        else float(top_mean - bottom_mean),
        "missed_high_marginal_row_count": len(missed_rows),
        "missed_high_marginal_mean_score_delta": missed_mean,
        "missed_high_marginal_positive_score_delta_fraction": _positive_fraction(missed_rows),
        "under_ranked_high_marginal_row_count": len(under_ranked_rows),
        "under_ranked_high_marginal_mean_score_delta": under_ranked_mean,
        "under_ranked_high_marginal_positive_score_delta_fraction": _positive_fraction(
            under_ranked_rows
        ),
        "classification": classification,
        "top_marginal_rows": ordered[: max(0, int(max_rows))],
    }


def _score_vector_or_none(values: torch.Tensor | None) -> torch.Tensor | None:
    if values is None:
        return None
    vector = values.detach().cpu().float().flatten()
    if int(vector.numel()) == 0:
        return None
    return vector


def _score_pair_for_row(
    *,
    primary: torch.Tensor | None,
    ablation: torch.Tensor | None,
    idx: int,
) -> dict[str, Any] | None:
    if primary is None or ablation is None:
        return None
    if idx < 0 or idx >= int(primary.numel()) or idx >= int(ablation.numel()):
        return None
    primary_value = float(primary[idx].item())
    ablation_value = float(ablation[idx].item())
    if not math.isfinite(primary_value) or not math.isfinite(ablation_value):
        return None
    delta = float(primary_value - ablation_value)
    return {
        "primary": primary_value,
        "ablation": ablation_value,
        "delta": delta,
        "positive_delta": bool(delta > 0.0),
    }


def _head_delta_pairs_for_row(
    *,
    primary_logits: torch.Tensor | None,
    ablation_logits: torch.Tensor | None,
    idx: int,
) -> dict[str, dict[str, Any]]:
    if primary_logits is None or ablation_logits is None:
        return {}
    primary = _head_logit_matrix(primary_logits)
    ablation = _head_logit_matrix(ablation_logits)
    if primary.ndim != 2 or ablation.ndim != 2 or primary.shape != ablation.shape:
        return {}
    if idx < 0 or idx >= int(primary.shape[0]):
        return {}
    head_names = [str(name) for name in QUERY_LOCAL_UTILITY_HEAD_NAMES]
    head_count = min(len(head_names), int(primary.shape[1]))
    out: dict[str, dict[str, Any]] = {}
    for head_idx in range(head_count):
        head_name = head_names[head_idx]
        primary_logit = float(primary[idx, head_idx].item())
        ablation_logit = float(ablation[idx, head_idx].item())
        if not math.isfinite(primary_logit) or not math.isfinite(ablation_logit):
            continue
        primary_prob = float(torch.sigmoid(primary[idx, head_idx]).item())
        ablation_prob = float(torch.sigmoid(ablation[idx, head_idx]).item())
        logit_delta = float(primary_logit - ablation_logit)
        probability_delta = float(primary_prob - ablation_prob)
        out[head_name] = {
            "primary_logit": primary_logit,
            "ablation_logit": ablation_logit,
            "logit_delta": logit_delta,
            "positive_logit_delta": bool(logit_delta > 0.0),
            "primary_probability": primary_prob,
            "ablation_probability": ablation_prob,
            "probability_delta": probability_delta,
            "positive_probability_delta": bool(probability_delta > 0.0),
        }
    return out


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    value_f = float(value)
    if not math.isfinite(value_f):
        return None
    return value_f


def _safe_logit(value: float) -> float:
    clipped = min(max(float(value), 1e-5), 1.0 - 1e-5)
    return float(math.log(clipped / (1.0 - clipped)))


def _factorized_replacement_modulated_score(values: dict[str, float]) -> float:
    query_local = float(
        0.50 * values["query_hit_probability"]
        + 0.45 * values["conditional_behavior_utility"]
    )
    replacement_multiplier = float(0.75 + 0.25 * values["replacement_representative_value"])
    return float(query_local * replacement_multiplier)


def _factorized_replacement_modulated_shapley_deltas(
    *,
    primary: dict[str, float],
    ablation: dict[str, float],
) -> dict[str, float]:
    factor_names = (
        "query_hit_probability",
        "conditional_behavior_utility",
        "replacement_representative_value",
    )
    contributions = {name: 0.0 for name in factor_names}
    permutation_count = 0
    for order in permutations(factor_names):
        current = {name: float(ablation[name]) for name in factor_names}
        before = _factorized_replacement_modulated_score(current)
        for name in order:
            current[name] = float(primary[name])
            after = _factorized_replacement_modulated_score(current)
            contributions[name] += float(after - before)
            before = after
        permutation_count += 1
    if permutation_count <= 0:
        return contributions
    return {
        name: float(value / permutation_count) for name, value in contributions.items()
    }


def _factorized_composition_for_row(
    *,
    head_deltas: dict[str, dict[str, Any]],
    raw_prediction: dict[str, Any] | None,
) -> dict[str, Any] | None:
    required_heads = (
        "query_hit_probability",
        "conditional_behavior_utility",
        "boundary_event_utility",
        "replacement_representative_value",
    )
    primary: dict[str, float] = {}
    ablation: dict[str, float] = {}
    for head_name in required_heads:
        pair = head_deltas.get(head_name)
        if not isinstance(pair, dict):
            return None
        primary_probability = _finite_number(pair.get("primary_probability"))
        ablation_probability = _finite_number(pair.get("ablation_probability"))
        if primary_probability is None or ablation_probability is None:
            return None
        primary[head_name] = min(max(primary_probability, 0.0), 1.0)
        ablation[head_name] = min(max(ablation_probability, 0.0), 1.0)

    modulated_primary = _factorized_replacement_modulated_score(primary)
    modulated_ablation = _factorized_replacement_modulated_score(ablation)
    modulated_delta = float(modulated_primary - modulated_ablation)
    boundary_primary = float(0.05 * primary["boundary_event_utility"])
    boundary_ablation = float(0.05 * ablation["boundary_event_utility"])
    boundary_delta = float(boundary_primary - boundary_ablation)
    unclamped_primary = float(modulated_primary + boundary_primary)
    unclamped_ablation = float(modulated_ablation + boundary_ablation)
    composed_primary = min(max(unclamped_primary, 0.0), 1.0)
    composed_ablation = min(max(unclamped_ablation, 0.0), 1.0)
    composed_delta = float(composed_primary - composed_ablation)
    clamp_delta = float(composed_delta - (modulated_delta + boundary_delta))
    logit_primary = _safe_logit(composed_primary)
    logit_ablation = _safe_logit(composed_ablation)
    logit_delta = float(logit_primary - logit_ablation)

    shapley = _factorized_replacement_modulated_shapley_deltas(
        primary=primary, ablation=ablation
    )
    contribution_deltas = {
        "query_hit_branch_shapley": shapley["query_hit_probability"],
        "behavior_branch_shapley": shapley["conditional_behavior_utility"],
        "replacement_modulation_shapley": shapley["replacement_representative_value"],
        "boundary_bonus": boundary_delta,
        "clamp": clamp_delta,
    }
    finite_contributions = {
        name: value
        for name, value in contribution_deltas.items()
        if math.isfinite(float(value))
    }
    raw_delta = None
    if isinstance(raw_prediction, dict):
        raw_delta = _finite_number(raw_prediction.get("delta"))
    residual = float(raw_delta - logit_delta) if raw_delta is not None else None
    positive_items = {
        name: value for name, value in finite_contributions.items() if value > 0.0
    }
    negative_items = {
        name: value for name, value in finite_contributions.items() if value < 0.0
    }
    dominant_positive = (
        max(positive_items.items(), key=lambda item: float(item[1]))
        if positive_items
        else None
    )
    dominant_negative = (
        min(negative_items.items(), key=lambda item: float(item[1]))
        if negative_items
        else None
    )
    return {
        "available": True,
        "semantics": (
            "Exact factorized QueryLocalUtility point-score decomposition in "
            "probability space; branch Shapley terms sum to the "
            "replacement-modulated query-local delta."
        ),
        "probabilities": {
            head_name: {
                "primary": primary[head_name],
                "ablation": ablation[head_name],
                "delta": float(primary[head_name] - ablation[head_name]),
            }
            for head_name in required_heads
        },
        "replacement_modulated_query_local_term": {
            "primary": modulated_primary,
            "ablation": modulated_ablation,
            "delta": modulated_delta,
        },
        "boundary_bonus": {
            "primary": boundary_primary,
            "ablation": boundary_ablation,
            "delta": boundary_delta,
        },
        "composed_score": {
            "primary": composed_primary,
            "ablation": composed_ablation,
            "delta": composed_delta,
        },
        "composed_logit": {
            "primary": logit_primary,
            "ablation": logit_ablation,
            "delta": logit_delta,
        },
        "raw_prediction_delta_residual": residual,
        "contribution_deltas": contribution_deltas,
        "dominant_positive_contribution": (
            {"name": dominant_positive[0], "delta": float(dominant_positive[1])}
            if dominant_positive is not None
            else None
        ),
        "dominant_negative_contribution": (
            {"name": dominant_negative[0], "delta": float(dominant_negative[1])}
            if dominant_negative is not None
            else None
        ),
    }


def _mean_from_rows(
    rows: list[dict[str, Any]],
    getter: Any,
) -> float | None:
    values: list[float] = []
    for row in rows:
        value = getter(row)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        value_f = float(value)
        if math.isfinite(value_f):
            values.append(value_f)
    if not values:
        return None
    return float(sum(values) / len(values))


def _positive_fraction_from_rows(
    rows: list[dict[str, Any]],
    getter: Any,
) -> float | None:
    values: list[float] = []
    for row in rows:
        value = getter(row)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        value_f = float(value)
        if math.isfinite(value_f):
            values.append(value_f)
    if not values:
        return None
    return float(sum(1 for value in values if value > 0.0) / len(values))


def _stage_delta(row: dict[str, Any], stage: str) -> float | None:
    pair = row.get(stage)
    if not isinstance(pair, dict):
        return None
    value = pair.get("delta")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _group_delta_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    head_names = [str(name) for name in QUERY_LOCAL_UTILITY_HEAD_NAMES]
    summary: dict[str, Any] = {"row_count": len(rows)}
    for stage in ("score_output", "raw_prediction", "segment_score"):
        summary[f"{stage}_mean_delta"] = _mean_from_rows(
            rows, lambda row, stage=stage: _stage_delta(row, stage)
        )
        summary[f"{stage}_positive_delta_fraction"] = _positive_fraction_from_rows(
            rows, lambda row, stage=stage: _stage_delta(row, stage)
        )

    composition_terms = (
        "query_hit_branch_shapley",
        "behavior_branch_shapley",
        "replacement_modulation_shapley",
        "boundary_bonus",
        "clamp",
    )
    summary["factorized_composition_available"] = any(
        isinstance(row.get("factorized_composition"), dict) for row in rows
    )
    summary["factorized_composed_score_mean_delta"] = _mean_from_rows(
        rows,
        lambda row: (
            row.get("factorized_composition", {})
            .get("composed_score", {})
            .get("delta")
            if isinstance(row.get("factorized_composition"), dict)
            else None
        ),
    )
    summary["factorized_composed_logit_mean_delta"] = _mean_from_rows(
        rows,
        lambda row: (
            row.get("factorized_composition", {})
            .get("composed_logit", {})
            .get("delta")
            if isinstance(row.get("factorized_composition"), dict)
            else None
        ),
    )
    summary["factorized_raw_prediction_delta_residual_mean"] = _mean_from_rows(
        rows,
        lambda row: (
            row.get("factorized_composition", {}).get("raw_prediction_delta_residual")
            if isinstance(row.get("factorized_composition"), dict)
            else None
        ),
    )
    contribution_means = {
        term: _mean_from_rows(
            rows,
            lambda row, term=term: (
                row.get("factorized_composition", {})
                .get("contribution_deltas", {})
                .get(term)
                if isinstance(row.get("factorized_composition"), dict)
                else None
            ),
        )
        for term in composition_terms
    }
    contribution_positive = {
        term: _positive_fraction_from_rows(
            rows,
            lambda row, term=term: (
                row.get("factorized_composition", {})
                .get("contribution_deltas", {})
                .get(term)
                if isinstance(row.get("factorized_composition"), dict)
                else None
            ),
        )
        for term in composition_terms
    }
    summary["factorized_contribution_mean_delta"] = contribution_means
    summary["factorized_contribution_positive_delta_fraction"] = contribution_positive
    finite_contribution_means = {
        term: value
        for term, value in contribution_means.items()
        if isinstance(value, int | float) and math.isfinite(float(value))
    }
    if finite_contribution_means:
        negative_term, negative_delta = min(
            finite_contribution_means.items(), key=lambda item: float(item[1])
        )
        positive_term, positive_delta = max(
            finite_contribution_means.items(), key=lambda item: float(item[1])
        )
        summary["factorized_most_negative_mean_contribution"] = {
            "name": negative_term,
            "delta": float(negative_delta),
        }
        summary["factorized_most_positive_mean_contribution"] = {
            "name": positive_term,
            "delta": float(positive_delta),
        }
    else:
        summary["factorized_most_negative_mean_contribution"] = None
        summary["factorized_most_positive_mean_contribution"] = None

    logit_means: dict[str, float | None] = {}
    prob_means: dict[str, float | None] = {}
    logit_positive: dict[str, float | None] = {}
    prob_positive: dict[str, float | None] = {}
    for head_name in head_names:
        logit_means[head_name] = _mean_from_rows(
            rows,
            lambda row, head_name=head_name: (
                row.get("head_deltas", {}).get(head_name, {}).get("logit_delta")
                if isinstance(row.get("head_deltas"), dict)
                else None
            ),
        )
        prob_means[head_name] = _mean_from_rows(
            rows,
            lambda row, head_name=head_name: (
                row.get("head_deltas", {}).get(head_name, {}).get("probability_delta")
                if isinstance(row.get("head_deltas"), dict)
                else None
            ),
        )
        logit_positive[head_name] = _positive_fraction_from_rows(
            rows,
            lambda row, head_name=head_name: (
                row.get("head_deltas", {}).get(head_name, {}).get("logit_delta")
                if isinstance(row.get("head_deltas"), dict)
                else None
            ),
        )
        prob_positive[head_name] = _positive_fraction_from_rows(
            rows,
            lambda row, head_name=head_name: (
                row.get("head_deltas", {}).get(head_name, {}).get("probability_delta")
                if isinstance(row.get("head_deltas"), dict)
                else None
            ),
        )
    summary["head_logit_mean_delta_by_head"] = logit_means
    summary["head_probability_mean_delta_by_head"] = prob_means
    summary["head_logit_positive_delta_fraction_by_head"] = logit_positive
    summary["head_probability_positive_delta_fraction_by_head"] = prob_positive
    finite_logit = {
        head: value
        for head, value in logit_means.items()
        if isinstance(value, int | float) and math.isfinite(float(value))
    }
    finite_prob = {
        head: value
        for head, value in prob_means.items()
        if isinstance(value, int | float) and math.isfinite(float(value))
    }
    if finite_logit:
        head, value = max(finite_logit.items(), key=lambda item: float(item[1]))
        summary["max_head_logit_mean_delta"] = float(value)
        summary["max_head_logit_mean_delta_head"] = head
    else:
        summary["max_head_logit_mean_delta"] = None
        summary["max_head_logit_mean_delta_head"] = None
    if finite_prob:
        head, value = max(finite_prob.items(), key=lambda item: float(item[1]))
        summary["max_head_probability_mean_delta"] = float(value)
        summary["max_head_probability_mean_delta_head"] = head
    else:
        summary["max_head_probability_mean_delta"] = None
        summary["max_head_probability_mean_delta_head"] = None
    return summary


def _classify_row_delta_path(top_summary: dict[str, Any]) -> str:
    score_delta = top_summary.get("score_output_mean_delta")
    raw_delta = top_summary.get("raw_prediction_mean_delta")
    segment_delta = top_summary.get("segment_score_mean_delta")
    max_prob_delta = top_summary.get("max_head_probability_mean_delta")
    max_logit_delta = top_summary.get("max_head_logit_mean_delta")
    if isinstance(score_delta, int | float) and float(score_delta) > 0.0:
        return "score_output_moves_top_marginal_rows"
    if isinstance(raw_delta, int | float) and float(raw_delta) > 0.0:
        return "score_composition_suppresses_positive_raw_delta"
    if isinstance(segment_delta, int | float) and float(segment_delta) > 0.0:
        return "segment_score_delta_not_reflected_in_final_score"
    if isinstance(max_prob_delta, int | float) and float(max_prob_delta) > 0.0:
        return "raw_score_suppresses_positive_head_probability_delta"
    if isinstance(max_logit_delta, int | float) and float(max_logit_delta) > 0.0:
        return "probability_or_raw_score_suppresses_positive_head_logit_delta"
    return "prior_delta_absent_or_non_positive_before_heads_for_top_marginal_rows"


def marginal_row_delta_path_diagnostics(
    *,
    selector_trace: dict[str, Any] | None,
    primary_scores: torch.Tensor | None,
    ablation_scores: torch.Tensor | None,
    primary_raw_predictions: torch.Tensor | None,
    ablation_raw_predictions: torch.Tensor | None,
    primary_segment_scores: torch.Tensor | None,
    ablation_segment_scores: torch.Tensor | None,
    primary_head_logits: torch.Tensor | None,
    ablation_head_logits: torch.Tensor | None,
    primary_mask: torch.Tensor | None,
    ablation_mask: torch.Tensor | None,
    max_rows: int = 16,
) -> dict[str, Any]:
    rows = _selector_trace_marginal_rows(selector_trace)
    if not rows:
        return {"available": False, "reason": "missing_selector_trace_marginal_rows"}
    score_primary = _score_vector_or_none(primary_scores)
    score_ablation = _score_vector_or_none(ablation_scores)
    raw_primary = _score_vector_or_none(primary_raw_predictions)
    raw_ablation = _score_vector_or_none(ablation_raw_predictions)
    segment_primary = _score_vector_or_none(primary_segment_scores)
    segment_ablation = _score_vector_or_none(ablation_segment_scores)
    primary_mask_vec = (
        primary_mask.detach().cpu().bool().flatten() if primary_mask is not None else None
    )
    ablation_mask_vec = (
        ablation_mask.detach().cpu().bool().flatten() if ablation_mask is not None else None
    )

    enriched: list[dict[str, Any]] = []
    for row in rows:
        point_index = row.get("point_index")
        if isinstance(point_index, bool) or not isinstance(point_index, int):
            continue
        idx = int(point_index)
        marginal = _row_float(row, "marginal_query_local_utility")
        if marginal is None:
            continue
        head_deltas = _head_delta_pairs_for_row(
            primary_logits=primary_head_logits,
            ablation_logits=ablation_head_logits,
            idx=idx,
        )
        raw_prediction = _score_pair_for_row(
            primary=raw_primary, ablation=raw_ablation, idx=idx
        )
        out: dict[str, Any] = {
            "point_index": idx,
            "decision": row.get("decision"),
            "source": row.get("source"),
            "marginal_query_local_utility": marginal,
            "marginal_rank_fraction": _row_float(
                row, "marginal_query_local_utility_candidate_rank_fraction"
            ),
            "selector_score_rank_fraction": _row_float(
                row, "selector_score_candidate_rank_fraction"
            ),
            "segment_score_rank_fraction": _row_float(
                row, "segment_score_candidate_rank_fraction"
            ),
            "failure_buckets": list(row.get("failure_buckets") or []),
            "score_output": _score_pair_for_row(
                primary=score_primary, ablation=score_ablation, idx=idx
            ),
            "raw_prediction": raw_prediction,
            "segment_score": _score_pair_for_row(
                primary=segment_primary, ablation=segment_ablation, idx=idx
            ),
            "head_deltas": head_deltas,
        }
        factorized_composition = _factorized_composition_for_row(
            head_deltas=head_deltas,
            raw_prediction=raw_prediction,
        )
        if factorized_composition is not None:
            out["factorized_composition"] = factorized_composition
        if primary_mask_vec is not None and idx < int(primary_mask_vec.numel()):
            out["primary_retained"] = bool(primary_mask_vec[idx].item())
        if ablation_mask_vec is not None and idx < int(ablation_mask_vec.numel()):
            out["ablation_retained"] = bool(ablation_mask_vec[idx].item())
        if "primary_retained" in out and "ablation_retained" in out:
            out["retained_changed"] = bool(out["primary_retained"] != out["ablation_retained"])
        enriched.append(out)
    if not enriched:
        return {"available": False, "reason": "missing_aligned_marginal_rows"}

    ordered = sorted(
        enriched,
        key=lambda row: (
            -float(row["marginal_query_local_utility"]),
            int(row["point_index"]),
        ),
    )
    top_count = max(1, math.ceil(0.25 * len(ordered)))
    bottom_count = max(1, math.ceil(0.25 * len(ordered)))
    top_rows = ordered[:top_count]
    missed_rows = [
        row
        for row in ordered
        if row.get("decision") == "removed_addition_gain"
        and (
            row.get("marginal_rank_fraction") is None
            or float(row.get("marginal_rank_fraction") or 1.0) <= 0.25
        )
    ]
    under_ranked_rows = [
        row
        for row in ordered
        if "high_marginal_under_ranked_by_scores" in set(row.get("failure_buckets") or [])
    ]
    groups = {
        "top_marginal": _group_delta_summary(top_rows),
        "bottom_marginal": _group_delta_summary(ordered[-bottom_count:]),
        "missed_high_marginal": _group_delta_summary(missed_rows),
        "under_ranked_high_marginal": _group_delta_summary(under_ranked_rows),
    }
    return {
        "available": True,
        "diagnostic_only": True,
        "semantics": (
            "primary_minus_ablation deltas on retained-marginal rows; positive deltas "
            "mean the primary prior configuration raises that stage versus the ablation."
        ),
        "classification": _classify_row_delta_path(groups["top_marginal"]),
        "row_count": len(enriched),
        "head_names": [str(name) for name in QUERY_LOCAL_UTILITY_HEAD_NAMES],
        "stage_available": {
            "score_output": score_primary is not None and score_ablation is not None,
            "raw_prediction": raw_primary is not None and raw_ablation is not None,
            "segment_score": segment_primary is not None and segment_ablation is not None,
            "head_output": primary_head_logits is not None and ablation_head_logits is not None,
            "retained_mask": primary_mask_vec is not None and ablation_mask_vec is not None,
        },
        "groups": groups,
        "top_marginal_rows": ordered[: max(0, int(max_rows))],
    }


def score_rank_margin_boundary_diagnostics(
    *,
    primary_scores: torch.Tensor | None,
    ablation_scores: torch.Tensor | None,
    primary_mask: torch.Tensor | None,
    ablation_mask: torch.Tensor | None,
    selector_trace: dict[str, Any] | None = None,
    max_rows: int = 16,
) -> dict[str, Any]:
    """Diagnose whether score deltas are large and directed enough to move masks."""
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
    if not bool(finite.all().item()):
        return {"available": False, "reason": "non_finite_scores"}
    delta = primary - ablation
    mask_diag = retained_mask_comparison(
        primary_mask=primary_mask,
        ablation_mask=ablation_mask,
        expected_shape=primary_scores.shape,
    )
    retained_count = int(mask_diag.get("primary_retained_count") or 0)
    topk_boundary = _topk_boundary_diagnostics(
        primary=primary,
        delta=delta,
        retained_count=retained_count,
    )
    actual_boundary = _actual_retained_score_boundary(
        primary=primary,
        primary_mask=primary_mask,
    )
    rows = _selector_trace_marginal_rows(selector_trace)
    marginal_alignment = _row_score_delta_alignment(
        rows=rows,
        primary=primary,
        ablation=ablation,
        max_rows=max_rows,
    )
    topk_crosses = bool(topk_boundary.get("score_delta_crosses_topk_boundary", False))
    mask_changed = bool(mask_diag.get("retained_mask_changed", False))
    marginal_class = str(marginal_alignment.get("classification", "missing_marginal_rows"))
    ratio = topk_boundary.get("max_abs_score_delta_to_topk_boundary_margin")
    below_margin = (
        isinstance(ratio, int | float)
        and math.isfinite(float(ratio))
        and float(ratio) < 1.0
        and not topk_crosses
    )
    if mask_changed:
        classification = "prior_score_delta_moves_retained_mask"
    elif (
        marginal_class.startswith("prior_delta_wrong_way")
        or marginal_class.startswith("prior_delta_non_positive")
        or marginal_class.startswith("prior_delta_globally_wrong_way")
    ):
        classification = marginal_class
    elif below_margin:
        classification = "prior_score_deltas_below_topk_rank_margin"
    elif not topk_crosses:
        classification = "prior_score_deltas_do_not_cross_score_boundary"
    else:
        classification = "prior_score_boundary_effect_not_material_to_selector_mask"
    return {
        "available": True,
        "diagnostic_only": True,
        "semantics": (
            "primary_minus_ablation score deltas; positive values mean the primary prior "
            "configuration raises the query-free selector score versus the ablation."
        ),
        "classification": classification,
        "score_count": int(primary.numel()),
        "retained_mask": mask_diag,
        "topk_score_boundary": topk_boundary,
        "actual_retained_score_boundary": actual_boundary,
        "marginal_row_score_delta_alignment": marginal_alignment,
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
    head_names: list[str] = [str(name) for name in QUERY_LOCAL_UTILITY_HEAD_NAMES]
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
    score_rank_margin_boundary: dict[str, Any] | None = None,
    marginal_row_delta_path: dict[str, Any] | None = None,
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
        "score_rank_margin_boundary": (
            score_rank_margin_boundary
            if isinstance(score_rank_margin_boundary, dict)
            else {"available": False, "reason": "not_computed"}
        ),
        "marginal_row_delta_path": (
            marginal_row_delta_path
            if isinstance(marginal_row_delta_path, dict)
            else {"available": False, "reason": "not_computed"}
        ),
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
    selector_trace: dict[str, Any] | None = None,
    primary_segment_scores: torch.Tensor | None = None,
    ablation_segment_scores: torch.Tensor | None = None,
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
        score_rank_margin_boundary=score_rank_margin_boundary_diagnostics(
            primary_scores=primary_scores,
            ablation_scores=ablation_scores,
            primary_mask=primary_mask,
            ablation_mask=ablation_mask,
            selector_trace=selector_trace,
        ),
        marginal_row_delta_path=marginal_row_delta_path_diagnostics(
            selector_trace=selector_trace,
            primary_scores=primary_scores,
            ablation_scores=ablation_scores,
            primary_raw_predictions=primary_raw_predictions,
            ablation_raw_predictions=ablation_raw_predictions,
            primary_segment_scores=primary_segment_scores,
            ablation_segment_scores=ablation_segment_scores,
            primary_head_logits=primary_head_logits,
            ablation_head_logits=ablation_head_logits,
            primary_mask=primary_mask,
            ablation_mask=ablation_mask,
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
        "disabled_prior_fields": list(WORKLOAD_BLIND_RANGE_MODEL_DISABLED_PRIOR_FIELDS),
        "model_prior_feature_transform": WORKLOAD_BLIND_RANGE_MODEL_PRIOR_TRANSFORM,
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
