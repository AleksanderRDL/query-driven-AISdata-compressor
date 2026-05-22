"""Diagnostic helpers for QueryLocalUtility targets."""

from __future__ import annotations

import math
from typing import Any

import torch

from learning.factorized_target_diagnostics import support_fraction_by_threshold
from learning.targets.query_local_utility_core import (
    _normalize_0_1,
    query_local_utility_point_score,
)
from learning.targets.query_local_utility_family import DIAGNOSTIC_TRAINABILITY_FOCUS_FAMILIES
from learning.targets.query_local_utility_segments import (
    _segment_budget_targets,
    _segment_pooled_targets,
    _ship_query_pair_fractional_segment_targets,
)


def _rank_correlation(x: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> float | None:
    """Return a lightweight Spearman-style rank correlation on valid positions."""
    valid = valid.to(dtype=torch.bool)
    if int(valid.sum().item()) < 2:
        return None
    xv = x[valid].float()
    yv = y[valid].float()
    if (
        float(xv.std(unbiased=False).item()) <= 1e-12
        or float(yv.std(unbiased=False).item()) <= 1e-12
    ):
        return None
    x_order = torch.argsort(xv, stable=True)
    y_order = torch.argsort(yv, stable=True)
    x_rank = torch.empty_like(xv)
    y_rank = torch.empty_like(yv)
    rank_values = torch.arange(int(xv.numel()), dtype=torch.float32, device=xv.device)
    x_rank[x_order] = rank_values
    y_rank[y_order] = rank_values
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denom = x_centered.norm() * y_centered.norm()
    if float(denom.item()) <= 1e-12:
        return None
    return float((x_centered * y_centered).sum().item() / float(denom.item()))


def _rank_vector(values: torch.Tensor) -> torch.Tensor | None:
    """Return stable ranks for a 1-D tensor, or None when the tensor is constant."""
    if int(values.numel()) < 2:
        return None
    values = values.float()
    if float(values.std(unbiased=False).item()) <= 1e-12:
        return None
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(int(values.numel()), dtype=torch.float32, device=values.device)
    return ranks


def _residualize_against(values: torch.Tensor, control: torch.Tensor) -> torch.Tensor | None:
    """Return linear residuals of values after regressing out control."""
    control_centered = control.float() - control.float().mean()
    denom = (control_centered * control_centered).sum()
    if float(denom.item()) <= 1e-12:
        return None
    values_f = values.float()
    slope = ((values_f - values_f.mean()) * control_centered).sum() / denom
    fitted = values_f.mean() + slope * control_centered
    return values_f - fitted


def _tensor_correlation(x: torch.Tensor, y: torch.Tensor) -> float | None:
    """Return Pearson correlation between 1-D tensors."""
    if int(x.numel()) < 2 or int(y.numel()) < 2:
        return None
    x_centered = x.float() - x.float().mean()
    y_centered = y.float() - y.float().mean()
    denom = x_centered.norm() * y_centered.norm()
    if float(denom.item()) <= 1e-12:
        return None
    return float((x_centered * y_centered).sum().item() / float(denom.item()))


def _partial_rank_correlation(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    control: torch.Tensor,
    valid: torch.Tensor,
) -> float | None:
    """Return Spearman-style partial correlation of x and y controlling for control."""
    valid = valid.to(dtype=torch.bool)
    if int(valid.sum().item()) < 3:
        return None
    x_rank = _rank_vector(x[valid].float())
    y_rank = _rank_vector(y[valid].float())
    control_rank = _rank_vector(control[valid].float())
    if x_rank is None or y_rank is None or control_rank is None:
        return None
    x_residual = _residualize_against(x_rank, control_rank)
    y_residual = _residualize_against(y_rank, control_rank)
    if x_residual is None or y_residual is None:
        return None
    return _tensor_correlation(x_residual, y_residual)


def _topk_overlap_and_mass_recall(
    *,
    ranker: torch.Tensor,
    reference: torch.Tensor,
    valid: torch.Tensor,
    ratio: float,
) -> dict[str, float]:
    """Return overlap and reference-mass recall when ranking by another target."""
    valid = valid.to(dtype=torch.bool)
    valid_count = int(valid.sum().item())
    if valid_count <= 0:
        return {"overlap": 0.0, "reference_mass_recall": 0.0}
    keep = min(valid_count, max(1, math.ceil(float(ratio) * valid_count)))
    ranker_values = ranker[valid].float()
    reference_values = reference[valid].float().clamp(min=0.0)
    reference_mass = float(reference_values.sum().item())
    if reference_mass <= 1e-12:
        return {"overlap": 0.0, "reference_mass_recall": 0.0}
    selected_by_ranker = torch.topk(ranker_values, k=keep, largest=True).indices
    selected_by_reference = torch.topk(reference_values, k=keep, largest=True).indices
    selected_mask = torch.zeros((valid_count,), dtype=torch.bool, device=ranker.device)
    reference_mask = torch.zeros((valid_count,), dtype=torch.bool, device=ranker.device)
    selected_mask[selected_by_ranker] = True
    reference_mask[selected_by_reference] = True
    ideal_mass = float(reference_values[selected_by_reference].sum().item())
    selected_mass = float(reference_values[selected_by_ranker].sum().item())
    return {
        "overlap": float((selected_mask & reference_mask).sum().item() / keep),
        "reference_mass_recall": float(selected_mass / max(ideal_mass, 1e-12)),
    }


def _conditional_behavior_target_alignment(
    *,
    behavior: torch.Tensor,
    valid: torch.Tensor,
    references: dict[str, torch.Tensor],
    ratio: float = 0.05,
) -> dict[str, Any]:
    """Summarize whether behavior labels rank points like query-useful reference labels."""
    valid = valid.to(dtype=torch.bool)
    out: dict[str, Any] = {
        "valid_point_count": int(valid.sum().item()),
        "topk_ratio": float(ratio),
    }
    for reference_name, reference in references.items():
        safe_name = str(reference_name)
        out[f"spearman_with_{safe_name}"] = _rank_correlation(behavior, reference, valid)
        topk = _topk_overlap_and_mass_recall(
            ranker=behavior,
            reference=reference,
            valid=valid,
            ratio=ratio,
        )
        out[f"topk_overlap_with_{safe_name}"] = topk["overlap"]
        out[f"topk_{safe_name}_mass_recall_ranked_by_behavior"] = topk["reference_mass_recall"]
    return out


def _ship_query_evidence_target_alignment(
    *,
    ship_query_evidence: torch.Tensor,
    valid: torch.Tensor,
    rankers: dict[str, torch.Tensor],
    query_hit_masks: list[torch.Tensor],
    boundaries: list[tuple[int, int]],
    ratio: float = 0.05,
) -> dict[str, Any]:
    """Summarize whether active target rankers recover one-credit-per-ship evidence."""
    valid = valid.to(dtype=torch.bool)
    reference = ship_query_evidence.float().clamp(min=0.0)
    valid_reference = reference[valid]
    out: dict[str, Any] = {
        "available": bool(valid.any().item()),
        "diagnostic_only": True,
        "reference": "ship_query_evidence",
        "valid_point_count": int(valid.sum().item()),
        "reference_positive_point_count": int((valid_reference > 0.0).sum().item()),
        "reference_mass": float(valid_reference.sum().item()),
        "topk_ratio": float(ratio),
        "rankers": {},
    }
    for ranker_name, ranker in rankers.items():
        topk = _topk_overlap_and_mass_recall(
            ranker=ranker,
            reference=reference,
            valid=valid,
            ratio=ratio,
        )
        row: dict[str, Any] = {
            "spearman_with_ship_query_evidence": _rank_correlation(ranker, reference, valid),
            "topk_overlap_with_ship_query_evidence": topk["overlap"],
            "topk_ship_query_evidence_mass_recall": topk["reference_mass_recall"],
        }
        row.update(
            _ship_query_pair_coverage_at_topk(
                ranker=ranker,
                valid=valid,
                query_hit_masks=query_hit_masks,
                boundaries=boundaries,
                ratio=ratio,
            )
        )
        out["rankers"][str(ranker_name)] = row
    return out


def _ship_query_pair_coverage_at_topk(
    *,
    ranker: torch.Tensor,
    valid: torch.Tensor,
    query_hit_masks: list[torch.Tensor],
    boundaries: list[tuple[int, int]],
    ratio: float,
) -> dict[str, Any]:
    """Return trajectory-query coverage when retaining top-ranked hit points."""
    valid = valid.to(dtype=torch.bool)
    valid_count = int(valid.sum().item())
    out: dict[str, Any] = {
        "ship_query_topk_ratio": float(ratio),
        "ship_query_topk_selected_point_count": 0,
        "ship_query_pair_count": 0,
        "ship_query_pair_covered_count": 0,
        "ship_query_pair_coverage_at_topk": 0.0,
    }
    if valid_count <= 0 or not query_hit_masks:
        return out

    keep = min(valid_count, max(1, math.ceil(float(ratio) * valid_count)))
    valid_indices = torch.where(valid)[0]
    top_local = torch.topk(ranker[valid].float(), k=keep, largest=True).indices
    selected = torch.zeros_like(valid, dtype=torch.bool)
    selected[valid_indices[top_local]] = True
    coverage = _ship_query_pair_coverage_for_selected(
        selected=selected,
        query_hit_masks=query_hit_masks,
        boundaries=boundaries,
    )
    out.update(
        {
            "ship_query_topk_selected_point_count": keep,
            "ship_query_pair_count": coverage["ship_query_pair_count"],
            "ship_query_pair_covered_count": coverage["ship_query_pair_covered_count"],
            "ship_query_pair_coverage_at_topk": coverage["ship_query_pair_coverage"],
        }
    )
    return out


def _ship_query_pair_coverage_for_selected(
    *,
    selected: torch.Tensor,
    query_hit_masks: list[torch.Tensor],
    boundaries: list[tuple[int, int]],
) -> dict[str, Any]:
    """Return trajectory-query pair coverage for a selected point mask."""
    selected = selected.to(dtype=torch.bool)

    pair_count = 0
    covered_count = 0
    for query_mask in query_hit_masks:
        query_hit = query_mask.to(device=selected.device, dtype=torch.bool)
        for start, end in boundaries:
            local_hit = query_hit[int(start) : int(end)]
            if not bool(local_hit.any().item()):
                continue
            pair_count += 1
            if bool((selected[int(start) : int(end)] & local_hit).any().item()):
                covered_count += 1
    return {
        "ship_query_pair_count": int(pair_count),
        "ship_query_pair_covered_count": int(covered_count),
        "ship_query_pair_coverage": float(covered_count / max(1, pair_count)),
    }


def _two_stage_segment_point_selection_diagnostics(
    *,
    segment_scores: torch.Tensor,
    point_scores: torch.Tensor,
    reference: torch.Tensor,
    valid: torch.Tensor,
    query_hit_masks: list[torch.Tensor],
    boundaries: list[tuple[int, int]],
    segment_size: int,
    ratio: float,
) -> dict[str, Any]:
    """Approximate segment allocation followed by within-segment point choice."""
    valid = valid.to(dtype=torch.bool)
    valid_count = int(valid.sum().item())
    out: dict[str, Any] = {
        "two_stage_available": valid_count > 0,
        "two_stage_topk_ratio": float(ratio),
        "two_stage_selected_point_count": 0,
        "two_stage_selected_segment_count": 0,
        "two_stage_ship_query_evidence_mass_recall": 0.0,
        "two_stage_ship_query_pair_count": 0,
        "two_stage_ship_query_pair_covered_count": 0,
        "two_stage_ship_query_pair_coverage": 0.0,
    }
    if valid_count <= 0:
        return out

    keep = min(valid_count, max(1, math.ceil(float(ratio) * valid_count)))
    device = valid.device
    segment_rows: list[tuple[float, int, int]] = []
    size = max(1, int(segment_size))
    for start, end in boundaries:
        for seg_start in range(int(start), int(end), size):
            seg_end = min(int(end), seg_start + size)
            if seg_end <= seg_start:
                continue
            local_valid = valid[seg_start:seg_end]
            if not bool(local_valid.any().item()):
                continue
            local_segment_scores = segment_scores[seg_start:seg_end].float()[local_valid]
            top_count = min(
                int(local_segment_scores.numel()),
                max(1, math.ceil(0.20 * int(local_segment_scores.numel()))),
            )
            segment_score = float(
                torch.topk(local_segment_scores, k=top_count, largest=True).values.mean().item()
            )
            segment_rows.append((segment_score, int(seg_start), int(seg_end)))
    if not segment_rows:
        out["two_stage_available"] = False
        out["reason"] = "no_valid_segments"
        return out
    segment_rows.sort(key=lambda row: (float(row[0]), -int(row[1])), reverse=True)

    selected = torch.zeros_like(valid, dtype=torch.bool, device=device)
    selected_segment_indices: set[int] = set()
    made_progress = True
    while int(selected.sum().item()) < keep and made_progress:
        made_progress = False
        for segment_rank, (_score, seg_start, seg_end) in enumerate(segment_rows):
            if int(selected.sum().item()) >= keep:
                break
            local_available = valid[seg_start:seg_end] & ~selected[seg_start:seg_end]
            if not bool(local_available.any().item()):
                continue
            local_indices = torch.where(local_available)[0] + int(seg_start)
            local_scores = point_scores[local_indices].float()
            best_local = local_indices[torch.argmax(local_scores)]
            selected[int(best_local.item())] = True
            selected_segment_indices.add(int(segment_rank))
            made_progress = True

    selected_count = int(selected.sum().item())
    valid_reference = reference[valid].float().clamp(min=0.0)
    ideal_indices = torch.topk(valid_reference, k=keep, largest=True).indices
    ideal_mass = float(valid_reference[ideal_indices].sum().item())
    selected_mass = float(reference[selected].float().clamp(min=0.0).sum().item())
    coverage = _ship_query_pair_coverage_for_selected(
        selected=selected,
        query_hit_masks=query_hit_masks,
        boundaries=boundaries,
    )
    out.update(
        {
            "two_stage_selected_point_count": selected_count,
            "two_stage_selected_segment_count": len(selected_segment_indices),
            "two_stage_ship_query_evidence_mass_recall": float(
                selected_mass / max(ideal_mass, 1e-12)
            ),
            "two_stage_ship_query_pair_count": coverage["ship_query_pair_count"],
            "two_stage_ship_query_pair_covered_count": coverage["ship_query_pair_covered_count"],
            "two_stage_ship_query_pair_coverage": coverage["ship_query_pair_coverage"],
        }
    )
    return out


def _family_conditioned_target_trainability_diagnostics(
    *,
    family_evidence: dict[str, dict[str, dict[str, Any]]],
    rankers: dict[str, torch.Tensor],
    boundaries: list[tuple[int, int]],
    ratio: float = 0.05,
) -> dict[str, Any]:
    """Summarize target-signal quality by workload family."""
    out: dict[str, Any] = {
        "available": bool(family_evidence),
        "diagnostic_only": True,
        "group_by": {},
        "focus_families": {
            group_key: sorted(values)
            for group_key, values in DIAGNOSTIC_TRAINABILITY_FOCUS_FAMILIES.items()
        },
        "interpretation": (
            "Target-side diagnostic only. It identifies whether current heads and "
            "candidate rankers expose ship/point evidence within blocker families."
        ),
    }
    for group_key, family_rows in family_evidence.items():
        group_out: dict[str, Any] = {}
        for family, evidence in family_rows.items():
            ship_evidence = evidence["ship_query_evidence"]
            family_valid = evidence["query_hit_probability"] > 0.0
            alignment = _ship_query_evidence_target_alignment(
                ship_query_evidence=ship_evidence,
                valid=family_valid,
                rankers=rankers,
                query_hit_masks=evidence["query_hit_masks"],
                boundaries=boundaries,
                ratio=ratio,
            )
            ranker_rows = alignment.get("rankers", {})
            target_shapes = {
                name: _target_distribution_summary(ranker, family_valid)
                for name, ranker in rankers.items()
            }
            weak_rankers = []
            ranker_items = ranker_rows.items() if isinstance(ranker_rows, dict) else []
            for name, row in ranker_items:
                if not isinstance(row, dict):
                    continue
                spearman = row.get("spearman_with_ship_query_evidence")
                if spearman is None or float(spearman) < 0.0:
                    weak_rankers.append(str(name))
            focus_family = family in DIAGNOSTIC_TRAINABILITY_FOCUS_FAMILIES.get(
                group_key, frozenset()
            )
            group_out[family] = {
                "available": alignment["available"],
                "focus_family": focus_family,
                "query_count": int(evidence["query_count"]),
                "valid_hit_point_count": int(family_valid.sum().item()),
                "ship_query_evidence_positive_point_count": alignment[
                    "reference_positive_point_count"
                ],
                "ship_query_evidence_mass": alignment["reference_mass"],
                "topk_ratio": float(ratio),
                "ranker_alignment": ranker_rows,
                "target_shapes": target_shapes,
                "weak_ship_evidence_rankers": weak_rankers,
                "target_trainability_status": (
                    "weak_active_target_signal"
                    if focus_family and weak_rankers
                    else "diagnostic_only"
                ),
            }
        out["group_by"][group_key] = group_out
    return out


def _conditional_behavior_candidate_diagnostics(
    *,
    current_behavior: torch.Tensor,
    valid: torch.Tensor,
    replacement: torch.Tensor,
    segment_budget: torch.Tensor,
    references: dict[str, torch.Tensor],
    query_hit_masks: list[torch.Tensor] | None = None,
    boundaries: list[tuple[int, int]] | None = None,
    ratio: float = 0.05,
) -> dict[str, Any]:
    """Return diagnostics for behavior-target candidates without changing labels."""
    valid = valid.to(dtype=torch.bool)
    replacement_signal = _normalize_0_1(replacement, valid)
    segment_signal = _normalize_0_1(segment_budget, valid)
    candidates = {
        "current_local_behavior": current_behavior,
        "replacement_gated_local_behavior": current_behavior * (0.25 + 0.75 * replacement_signal),
        "segment_gated_local_behavior": current_behavior * (0.25 + 0.75 * segment_signal),
        "replacement_support_only_local_behavior": torch.where(
            replacement > 0.0,
            current_behavior,
            torch.zeros_like(current_behavior),
        ),
        "replacement_segment_gated_local_behavior": current_behavior
        * (0.20 + 0.50 * replacement_signal + 0.30 * segment_signal),
    }
    out: dict[str, Any] = {}
    for name, candidate in candidates.items():
        valid_candidate = candidate[valid].float().clamp(min=0.0)
        row = _conditional_behavior_target_alignment(
            behavior=candidate,
            valid=valid,
            references=references,
        )
        row["support_fraction_by_threshold"] = support_fraction_by_threshold(candidate, valid)
        row["target_mass"] = float(valid_candidate.sum().item())
        row["target_mean"] = (
            float(valid_candidate.mean().item()) if int(valid_candidate.numel()) > 0 else 0.0
        )
        row["target_std"] = (
            float(valid_candidate.std(unbiased=False).item())
            if int(valid_candidate.numel()) > 0
            else 0.0
        )
        if query_hit_masks is not None and boundaries is not None:
            row.update(
                _ship_query_pair_coverage_at_topk(
                    ranker=candidate,
                    valid=valid,
                    query_hit_masks=query_hit_masks,
                    boundaries=boundaries,
                    ratio=ratio,
                )
            )
        out[str(name)] = row
    return out


def _conditional_behavior_replacement_partial_alignment(
    *,
    behavior: torch.Tensor,
    replacement: torch.Tensor,
    valid: torch.Tensor,
    references: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Return behavior alignment after controlling for replacement rank."""
    valid = valid.to(dtype=torch.bool)
    out: dict[str, Any] = {
        "available": bool(valid.any().item()),
        "diagnostic_only": True,
        "control": "replacement_representative_value",
        "valid_point_count": int(valid.sum().item()),
        "behavior_replacement_spearman": _rank_correlation(behavior, replacement, valid),
        "references": {},
    }
    for reference_name, reference in references.items():
        out["references"][str(reference_name)] = {
            "behavior_spearman": _rank_correlation(behavior, reference, valid),
            "replacement_spearman": _rank_correlation(replacement, reference, valid),
            "behavior_partial_spearman_controlling_replacement": _partial_rank_correlation(
                x=behavior,
                y=reference,
                control=replacement,
                valid=valid,
            ),
        }
    return out


def _target_distribution_summary(values: torch.Tensor, valid: torch.Tensor) -> dict[str, Any]:
    """Return compact target-shape stats on the selected support."""
    valid = valid.to(dtype=torch.bool)
    valid_values = values[valid].float().clamp(min=0.0)
    return {
        "support_fraction_by_threshold": support_fraction_by_threshold(values, valid),
        "target_mass": float(valid_values.sum().item()),
        "target_mean": (
            float(valid_values.mean().item()) if int(valid_values.numel()) > 0 else 0.0
        ),
        "target_std": (
            float(valid_values.std(unbiased=False).item()) if int(valid_values.numel()) > 0 else 0.0
        ),
    }


def _segment_budget_ship_presence_candidate_alignment(
    *,
    active_segment_budget: torch.Tensor,
    final_score: torch.Tensor,
    q_hit: torch.Tensor,
    ship_query_evidence: torch.Tensor,
    valid: torch.Tensor,
    query_hit_masks: list[torch.Tensor],
    boundaries: list[tuple[int, int]],
    segment_size: int,
    ratio: float = 0.05,
) -> dict[str, Any]:
    """Compare diagnostic segment-budget candidates against ship-query evidence."""
    valid = valid.to(dtype=torch.bool)
    normalized_final = _normalize_0_1(final_score, valid)
    normalized_q_hit = _normalize_0_1(q_hit, valid)
    normalized_ship = _normalize_0_1(ship_query_evidence, valid)
    candidates = {
        "active_segment_budget_target": active_segment_budget,
        "ship_presence_segment_budget_candidate": _segment_budget_targets(
            ship_query_evidence,
            boundaries,
            segment_size,
        ),
        "final_score_ship_presence_blend_segment_budget_candidate": _segment_budget_targets(
            0.50 * normalized_final + 0.50 * normalized_ship,
            boundaries,
            segment_size,
        ),
        "query_hit_ship_presence_blend_segment_budget_candidate": _segment_budget_targets(
            0.50 * normalized_q_hit + 0.50 * normalized_ship,
            boundaries,
            segment_size,
        ),
    }
    alignment = _ship_query_evidence_target_alignment(
        ship_query_evidence=ship_query_evidence,
        valid=valid,
        rankers=candidates,
        query_hit_masks=query_hit_masks,
        boundaries=boundaries,
        ratio=ratio,
    )
    out: dict[str, Any] = {
        "available": alignment["available"],
        "diagnostic_only": True,
        "active_training_target_unchanged": True,
        "candidate_usage": "diagnostic_only_not_training_semantics",
        "segment_size_points": int(segment_size),
        "topk_ratio": float(ratio),
        "reference": "ship_query_evidence",
        "candidates": {},
    }
    ranker_rows = alignment.get("rankers", {})
    for name, candidate in candidates.items():
        row: dict[str, Any] = {}
        if isinstance(ranker_rows, dict) and isinstance(ranker_rows.get(name), dict):
            row.update(ranker_rows[name])
        row.update(_target_distribution_summary(candidate, valid))
        row["spearman_with_final_score"] = _rank_correlation(candidate, final_score, valid)
        row["spearman_with_query_hit_probability"] = _rank_correlation(candidate, q_hit, valid)
        final_topk = _topk_overlap_and_mass_recall(
            ranker=candidate,
            reference=final_score,
            valid=valid,
            ratio=ratio,
        )
        q_hit_topk = _topk_overlap_and_mass_recall(
            ranker=candidate,
            reference=q_hit,
            valid=valid,
            ratio=ratio,
        )
        row["topk_overlap_with_final_score"] = final_topk["overlap"]
        row["topk_final_score_mass_recall"] = final_topk["reference_mass_recall"]
        row["topk_overlap_with_query_hit_probability"] = q_hit_topk["overlap"]
        row["topk_query_hit_probability_mass_recall"] = q_hit_topk["reference_mass_recall"]
        out["candidates"][str(name)] = row
    return out


def _family_local_target_candidate_alignment(
    *,
    family_evidence: dict[str, dict[str, dict[str, Any]]],
    rankers: dict[str, torch.Tensor],
    boundaries: list[tuple[int, int]],
    segment_size: int,
    ratio: float = 0.05,
) -> dict[str, Any]:
    """Compare family-local target candidates against active evidence-gate semantics."""
    required_rankers = {
        "final_score",
        "query_hit_probability",
        "conditional_behavior_utility",
        "boundary_event_utility",
        "replacement_representative_value",
        "segment_budget_target",
    }
    missing = sorted(required_rankers.difference(rankers))
    out: dict[str, Any] = {
        "available": bool(family_evidence) and not missing,
        "diagnostic_only": True,
        "active_training_target_unchanged": False,
        "candidate_usage": (
            "active_query_hit_target_uses_raw_query_hit_ship_evidence_multiplier; "
            "remaining_family_candidates_diagnostic_only"
        ),
        "segment_size_points": int(segment_size),
        "topk_ratio": float(ratio),
        "reference": "family_ship_query_evidence",
        "focus_families": {
            group_key: sorted(values)
            for group_key, values in DIAGNOSTIC_TRAINABILITY_FOCUS_FAMILIES.items()
        },
        "group_by": {},
    }
    if missing:
        out["reason"] = "missing_required_rankers"
        out["missing_rankers"] = missing
        return out

    final_score = rankers["final_score"].float().clamp(0.0, 1.0)
    active_q_hit = rankers["query_hit_probability"].float().clamp(0.0, 1.0)
    behavior = rankers["conditional_behavior_utility"].float().clamp(0.0, 1.0)
    boundary = rankers["boundary_event_utility"].float().clamp(0.0, 1.0)
    replacement = rankers["replacement_representative_value"].float().clamp(0.0, 1.0)
    active_segment_budget = rankers["segment_budget_target"].float().clamp(0.0, 1.0)
    zero = torch.zeros_like(final_score)

    for group_key, family_rows in family_evidence.items():
        group_out: dict[str, Any] = {}
        for family, evidence in family_rows.items():
            family_q_hit = evidence["query_hit_probability"].float().clamp(0.0, 1.0)
            family_ship = evidence["ship_query_evidence"].float().clamp(0.0, 1.0)
            family_valid = family_q_hit > 0.0
            focus_family = family in DIAGNOSTIC_TRAINABILITY_FOCUS_FAMILIES.get(
                group_key, frozenset()
            )
            if not bool(family_valid.any().item()):
                group_out[str(family)] = {
                    "available": False,
                    "focus_family": focus_family,
                    "query_count": int(evidence["query_count"]),
                    "reason": "no_family_hit_points",
                }
                continue

            normalized_family_hit = _normalize_0_1(family_q_hit, family_valid)
            normalized_ship = _normalize_0_1(family_ship, family_valid)
            normalized_boundary = _normalize_0_1(boundary, family_valid)
            normalized_replacement = _normalize_0_1(replacement, family_valid)

            query_hit_ship_blend = torch.where(
                family_valid,
                (0.65 * normalized_family_hit + 0.35 * normalized_ship).clamp(0.0, 1.0),
                zero,
            )
            ship_gated_behavior = torch.where(
                family_valid,
                (behavior * (0.35 + 0.65 * normalized_ship)).clamp(0.0, 1.0),
                zero,
            )
            boundary_replacement_ship_score = torch.where(
                family_valid,
                (
                    0.45 * normalized_ship
                    + 0.30 * normalized_replacement
                    + 0.25 * normalized_boundary
                ).clamp(0.0, 1.0),
                zero,
            )
            composed_score = torch.where(
                family_valid,
                query_local_utility_point_score(
                    q_hit=query_hit_ship_blend,
                    behavior=ship_gated_behavior,
                    boundary=boundary,
                    replacement=replacement,
                ),
                zero,
            )
            segment_budget_sum = _segment_budget_targets(
                composed_score,
                boundaries,
                segment_size,
            )
            query_hit_ship_segment_top20 = _segment_pooled_targets(
                query_hit_ship_blend,
                boundaries,
                segment_size,
                pool="top20_mean",
            )
            query_hit_ship_segment_max = _segment_pooled_targets(
                query_hit_ship_blend,
                boundaries,
                segment_size,
                pool="max",
            )
            composed_segment_top20 = _segment_pooled_targets(
                composed_score,
                boundaries,
                segment_size,
                pool="top20_mean",
            )
            ship_pair_fractional_segment = _ship_query_pair_fractional_segment_targets(
                query_hit_masks=evidence["query_hit_masks"],
                boundaries=boundaries,
                segment_size=segment_size,
                point_count=int(final_score.numel()),
                device=final_score.device,
            )
            candidates = {
                "family_query_hit_ship_blend_candidate": query_hit_ship_blend,
                "family_ship_gated_behavior_candidate": ship_gated_behavior,
                "family_boundary_replacement_ship_score_candidate": (
                    boundary_replacement_ship_score
                ),
                "family_local_composed_score_candidate": composed_score,
                "family_local_segment_budget_candidate": segment_budget_sum,
                "family_query_hit_ship_segment_top20_mean_candidate": (
                    query_hit_ship_segment_top20
                ),
                "family_query_hit_ship_segment_max_candidate": query_hit_ship_segment_max,
                "family_composed_segment_top20_mean_candidate": composed_segment_top20,
                "family_ship_query_pair_fractional_segment_candidate": (
                    ship_pair_fractional_segment
                ),
            }
            baselines = {
                "active_final_score": final_score,
                "active_query_hit_probability": active_q_hit,
                "active_segment_budget_target": active_segment_budget,
            }
            alignment = _ship_query_evidence_target_alignment(
                ship_query_evidence=family_ship,
                valid=family_valid,
                rankers={**baselines, **candidates},
                query_hit_masks=evidence["query_hit_masks"],
                boundaries=boundaries,
                ratio=ratio,
            )
            ranker_rows = alignment.get("rankers", {})

            candidate_rows: dict[str, Any] = {}
            segment_candidate_names = {
                "family_local_segment_budget_candidate",
                "family_query_hit_ship_segment_top20_mean_candidate",
                "family_query_hit_ship_segment_max_candidate",
                "family_composed_segment_top20_mean_candidate",
                "family_ship_query_pair_fractional_segment_candidate",
            }
            for name, candidate in candidates.items():
                row: dict[str, Any] = {}
                if isinstance(ranker_rows, dict) and isinstance(ranker_rows.get(name), dict):
                    row.update(ranker_rows[name])
                row.update(_target_distribution_summary(candidate, family_valid))
                row["spearman_with_active_final_score"] = _rank_correlation(
                    candidate,
                    final_score,
                    family_valid,
                )
                row["spearman_with_active_query_hit_probability"] = _rank_correlation(
                    candidate,
                    active_q_hit,
                    family_valid,
                )
                final_topk = _topk_overlap_and_mass_recall(
                    ranker=candidate,
                    reference=final_score,
                    valid=family_valid,
                    ratio=ratio,
                )
                q_hit_topk = _topk_overlap_and_mass_recall(
                    ranker=candidate,
                    reference=active_q_hit,
                    valid=family_valid,
                    ratio=ratio,
                )
                row["topk_overlap_with_active_final_score"] = final_topk["overlap"]
                row["topk_active_final_score_mass_recall"] = final_topk["reference_mass_recall"]
                row["topk_overlap_with_active_query_hit_probability"] = q_hit_topk["overlap"]
                row["topk_active_query_hit_probability_mass_recall"] = q_hit_topk[
                    "reference_mass_recall"
                ]
                if name in segment_candidate_names:
                    row["two_stage_point_ranker"] = "family_query_hit_ship_blend_candidate"
                    row.update(
                        _two_stage_segment_point_selection_diagnostics(
                            segment_scores=candidate,
                            point_scores=query_hit_ship_blend,
                            reference=family_ship,
                            valid=family_valid,
                            query_hit_masks=evidence["query_hit_masks"],
                            boundaries=boundaries,
                            segment_size=segment_size,
                            ratio=ratio,
                        )
                    )
                candidate_rows[str(name)] = row

            baseline_rows: dict[str, Any] = {}
            for name in baselines:
                if isinstance(ranker_rows, dict) and isinstance(ranker_rows.get(name), dict):
                    baseline_rows[str(name)] = dict(ranker_rows[name])

            candidate_spearmans = [
                float(row["spearman_with_ship_query_evidence"])
                for row in candidate_rows.values()
                if row.get("spearman_with_ship_query_evidence") is not None
            ]
            baseline_spearmans = [
                float(row["spearman_with_ship_query_evidence"])
                for row in baseline_rows.values()
                if row.get("spearman_with_ship_query_evidence") is not None
            ]
            segment_candidate_two_stage_pair_coverages = [
                float(row["two_stage_ship_query_pair_coverage"])
                for name, row in candidate_rows.items()
                if name in segment_candidate_names
                and row.get("two_stage_ship_query_pair_coverage") is not None
            ]
            segment_candidate_two_stage_mass_recalls = [
                float(row["two_stage_ship_query_evidence_mass_recall"])
                for name, row in candidate_rows.items()
                if name in segment_candidate_names
                and row.get("two_stage_ship_query_evidence_mass_recall") is not None
            ]
            best_candidate_spearman = max(candidate_spearmans) if candidate_spearmans else None
            best_baseline_spearman = max(baseline_spearmans) if baseline_spearmans else None
            group_out[str(family)] = {
                "available": alignment["available"],
                "focus_family": focus_family,
                "query_count": int(evidence["query_count"]),
                "valid_hit_point_count": int(family_valid.sum().item()),
                "ship_query_evidence_positive_point_count": alignment[
                    "reference_positive_point_count"
                ],
                "ship_query_evidence_mass": alignment["reference_mass"],
                "active_baseline_alignment": baseline_rows,
                "candidate_alignment": candidate_rows,
                "best_candidate_spearman_with_ship_query_evidence": best_candidate_spearman,
                "best_active_baseline_spearman_with_ship_query_evidence": best_baseline_spearman,
                "best_segment_candidate_two_stage_ship_query_pair_coverage": (
                    max(segment_candidate_two_stage_pair_coverages)
                    if segment_candidate_two_stage_pair_coverages
                    else None
                ),
                "best_segment_candidate_two_stage_ship_query_evidence_mass_recall": (
                    max(segment_candidate_two_stage_mass_recalls)
                    if segment_candidate_two_stage_mass_recalls
                    else None
                ),
                "candidate_signal_status": (
                    "diagnostic_candidate_improves_family_ship_signal"
                    if focus_family
                    and best_candidate_spearman is not None
                    and (
                        best_baseline_spearman is None
                        or best_candidate_spearman > best_baseline_spearman
                    )
                    else "diagnostic_only"
                ),
            }
        out["group_by"][group_key] = group_out
    return out
