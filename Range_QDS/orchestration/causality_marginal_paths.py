"""Marginal-row and score-boundary causality diagnostics."""

from __future__ import annotations

import math
from itertools import permutations
from typing import Any

import torch

from learning.targets.query_local_utility import QUERY_LOCAL_UTILITY_HEAD_NAMES
from orchestration.causality_score_sensitivity import retained_mask_comparison


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


def _head_logit_matrix(head_logits: torch.Tensor) -> torch.Tensor:
    """Return head logits as [point, head] for sensitivity diagnostics."""
    logits = head_logits.detach().cpu().float()
    if logits.ndim == 3 and int(logits.shape[0]) == 1:
        logits = logits.squeeze(0)
    return logits


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
        0.50 * values["query_hit_probability"] + 0.45 * values["conditional_behavior_utility"]
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
    contributions: dict[str, float] = dict.fromkeys(factor_names, 0.0)
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
    return {name: float(value / permutation_count) for name, value in contributions.items()}


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

    shapley = _factorized_replacement_modulated_shapley_deltas(primary=primary, ablation=ablation)
    contribution_deltas = {
        "query_hit_branch_shapley": shapley["query_hit_probability"],
        "behavior_branch_shapley": shapley["conditional_behavior_utility"],
        "replacement_modulation_shapley": shapley["replacement_representative_value"],
        "boundary_bonus": boundary_delta,
        "clamp": clamp_delta,
    }
    finite_contributions = {
        name: value for name, value in contribution_deltas.items() if math.isfinite(float(value))
    }
    raw_delta = None
    if isinstance(raw_prediction, dict):
        raw_delta = _finite_number(raw_prediction.get("delta"))
    residual = float(raw_delta - logit_delta) if raw_delta is not None else None
    positive_items = {name: value for name, value in finite_contributions.items() if value > 0.0}
    negative_items = {name: value for name, value in finite_contributions.items() if value < 0.0}
    dominant_positive = (
        max(positive_items.items(), key=lambda item: float(item[1])) if positive_items else None
    )
    dominant_negative = (
        min(negative_items.items(), key=lambda item: float(item[1])) if negative_items else None
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
            row.get("factorized_composition", {}).get("composed_score", {}).get("delta")
            if isinstance(row.get("factorized_composition"), dict)
            else None
        ),
    )
    summary["factorized_composed_logit_mean_delta"] = _mean_from_rows(
        rows,
        lambda row: (
            row.get("factorized_composition", {}).get("composed_logit", {}).get("delta")
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
                row.get("factorized_composition", {}).get("contribution_deltas", {}).get(term)
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
                row.get("factorized_composition", {}).get("contribution_deltas", {}).get(term)
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
        raw_prediction = _score_pair_for_row(primary=raw_primary, ablation=raw_ablation, idx=idx)
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
            "segment_score_rank_fraction": _row_float(row, "segment_score_candidate_rank_fraction"),
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
    elif marginal_class.startswith(
        (
            "prior_delta_wrong_way",
            "prior_delta_non_positive",
            "prior_delta_globally_wrong_way",
        )
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
