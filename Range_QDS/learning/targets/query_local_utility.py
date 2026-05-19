"""Factorized QueryLocalUtility target construction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from learning.factorized_target_diagnostics import (
    factorized_target_diagnostics,
    support_fraction_by_threshold,
)
from learning.targets.query_local_utility_family import (
    DIAGNOSTIC_TRAINABILITY_FOCUS_FAMILIES,
    _range_query_family_evidence,
)
from learning.targets.query_local_utility_segments import (
    _segment_budget_targets,
    _segment_pooled_targets,
    _ship_query_pair_fractional_segment_targets,
    query_local_utility_path_length_support_targets,
)
from workloads.query_types import NUM_QUERY_TYPES, QUERY_TYPE_ID_RANGE
from workloads.range_geometry import points_in_range_box, segment_box_bracket_indices

QUERY_LOCAL_UTILITY_FACTORIZED_TARGET_MODE = "query_local_utility_factorized"
QUERY_LOCAL_UTILITY_SEGMENT_BUDGET_QUERY_SHIP_MAX_POOL_TARGET_MODE = (
    "query_local_utility_factorized_segment_budget_query_ship_max_pool"
)
QUERY_LOCAL_UTILITY_QUERY_SHIP_LOCAL_HEADS_TARGET_MODE = (
    "query_local_utility_factorized_query_ship_local_heads"
)
QUERY_LOCAL_UTILITY_EXPERIMENTAL_TARGET_MODES = frozenset(
    {
        QUERY_LOCAL_UTILITY_SEGMENT_BUDGET_QUERY_SHIP_MAX_POOL_TARGET_MODE,
        QUERY_LOCAL_UTILITY_QUERY_SHIP_LOCAL_HEADS_TARGET_MODE,
    }
)
QUERY_LOCAL_UTILITY_TARGET_MODES = frozenset(
    {
        QUERY_LOCAL_UTILITY_FACTORIZED_TARGET_MODE,
        *QUERY_LOCAL_UTILITY_EXPERIMENTAL_TARGET_MODES,
    }
)
QUERY_LOCAL_UTILITY_HEAD_NAMES = (
    "query_hit_probability",
    "conditional_behavior_utility",
    "boundary_event_utility",
    "replacement_representative_value",
    "segment_budget_target",
    "path_length_support_target",
)
QUERY_LOCAL_UTILITY_FINAL_LABEL_FORMULA = (
    "query_hit_times_behavior_with_conditional_replacement_modulation_plus_boundary"
)


@dataclass
class QueryLocalUtilityTargetBundle:
    """Scalar and factorized training labels for QueryLocalUtility."""

    labels: torch.Tensor
    labelled_mask: torch.Tensor
    head_targets: torch.Tensor
    head_mask: torch.Tensor
    diagnostics: dict[str, Any]


def query_local_utility_point_score(
    *,
    q_hit: torch.Tensor,
    behavior: torch.Tensor,
    boundary: torch.Tensor,
    replacement: torch.Tensor,
) -> torch.Tensor:
    """Return the scalar QueryLocalUtility point score used by labels and v2 logits."""
    q_hit = q_hit.float().clamp(0.0, 1.0)
    behavior = behavior.float().clamp(0.0, 1.0)
    boundary = boundary.float().clamp(0.0, 1.0)
    replacement = replacement.float().clamp(0.0, 1.0)
    return (q_hit * (0.5 + behavior) * (0.75 + 0.25 * replacement) + 0.25 * boundary).clamp(
        0.0, 1.0
    )


def _query_ship_blend_signal(
    *,
    q_hit: torch.Tensor,
    ship_query_evidence: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return a query-local presence utility that preserves ship-level evidence."""
    valid = valid.to(dtype=torch.bool)
    normalized_q_hit = _normalize_0_1(q_hit, valid)
    normalized_ship = _normalize_0_1(ship_query_evidence, valid)
    blended = (0.65 * normalized_q_hit + 0.35 * normalized_ship).clamp(0.0, 1.0)
    return torch.where(valid, blended, torch.zeros_like(blended))


def _query_ship_local_behavior_signal(
    *,
    behavior: torch.Tensor,
    boundary: torch.Tensor,
    replacement: torch.Tensor,
    ship_query_evidence: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return a compact query-local behavior utility for retained-set value."""
    valid = valid.to(dtype=torch.bool)
    normalized_behavior = _normalize_0_1(behavior, valid)
    normalized_boundary = _normalize_0_1(boundary, valid)
    normalized_replacement = _normalize_0_1(replacement, valid)
    normalized_ship = _normalize_0_1(ship_query_evidence, valid)
    utility = (
        0.45 * normalized_ship
        + 0.25 * normalized_behavior
        + 0.20 * normalized_replacement
        + 0.10 * normalized_boundary
    ).clamp(0.0, 1.0)
    return torch.where(valid, utility, torch.zeros_like(utility))


def _normalize_0_1(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Normalize non-negative values to [0, 1] on the selected support."""
    out = values.float().clamp(min=0.0)
    support = mask if mask is not None else torch.ones_like(out, dtype=torch.bool)
    if not bool(support.any().item()):
        return torch.zeros_like(out)
    local = out[support]
    max_value = float(local.max().item()) if int(local.numel()) > 0 else 0.0
    if max_value <= 1e-12:
        return torch.zeros_like(out)
    return (out / max_value).clamp(0.0, 1.0)


def _trajectory_change_weights(
    points: torch.Tensor, boundaries: list[tuple[int, int]]
) -> torch.Tensor:
    """Return sparse query-free behavior-change weights per point."""
    n_points = int(points.shape[0])
    if n_points <= 0:
        return torch.empty((0,), dtype=torch.float32, device=points.device)
    weights = torch.zeros((n_points,), dtype=torch.float32, device=points.device)
    indices = torch.arange(n_points, device=points.device)
    prev_idx = torch.clamp(indices - 1, min=0)
    next_idx = torch.clamp(indices + 1, max=n_points - 1)
    if points.shape[1] > 7:
        weights = torch.maximum(weights, points[:, 7].float().clamp(min=0.0))
    if points.shape[1] > 4:
        prev_heading = torch.abs(points[:, 4].float() - points[prev_idx, 4].float())
        next_heading = torch.abs(points[next_idx, 4].float() - points[:, 4].float())
        prev_heading = torch.minimum(prev_heading, 360.0 - prev_heading).clamp(min=0.0) / 180.0
        next_heading = torch.minimum(next_heading, 360.0 - next_heading).clamp(min=0.0) / 180.0
        weights = torch.maximum(weights, torch.maximum(prev_heading, next_heading))
    if points.shape[1] > 3:
        prev_speed = torch.abs(points[:, 3].float() - points[prev_idx, 3].float())
        next_speed = torch.abs(points[next_idx, 3].float() - points[:, 3].float())
        speed_change = torch.maximum(prev_speed, next_speed)
        weights = torch.maximum(weights, _normalize_0_1(speed_change))
    weights = weights.clamp(0.0, 1.0)
    sparse = torch.zeros_like(weights)
    for start, end in boundaries:
        local = weights[int(start) : int(end)]
        if int(local.numel()) <= 0:
            continue
        max_value = local.max()
        min_value = local.min()
        if float((max_value - min_value).item()) <= 1e-6:
            continue
        threshold = torch.quantile(local, 0.70)
        sparse[int(start) : int(end)] = (
            (local - threshold) / (max_value - threshold).clamp(min=1e-6)
        ).clamp(0.0, 1.0)
    return sparse


def _boundary_indices_for_query(
    range_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
) -> torch.Tensor:
    """Return in-box entry/exit point indices for one query."""
    parts: list[torch.Tensor] = []
    mask_cpu = range_mask.detach().cpu()
    for start, end in boundaries:
        if end <= start:
            continue
        local = mask_cpu[start:end]
        if not bool(local.any().item()):
            continue
        enters = torch.zeros_like(local)
        exits = torch.zeros_like(local)
        enters[1:] = local[1:] & ~local[:-1]
        enters[0] = local[0]
        exits[:-1] = local[:-1] & ~local[1:]
        exits[-1] = local[-1]
        offsets = torch.where(enters | exits)[0]
        if int(offsets.numel()) > 0:
            parts.append(offsets.to(dtype=torch.long) + int(start))
    if not parts:
        return torch.empty((0,), dtype=torch.long, device=range_mask.device)
    return torch.cat(parts).to(device=range_mask.device, dtype=torch.long).unique(sorted=True)


def _query_replacement_support(
    query_value: torch.Tensor,
    boundaries: list[tuple[int, int]],
    keep_fraction: float = 0.50,
) -> torch.Tensor:
    """Return sparse query-local representative support for replacement-value labels."""
    support = torch.zeros_like(query_value, dtype=torch.bool)
    ratio = min(1.0, max(0.0, float(keep_fraction)))
    positive = query_value > 0.0
    for start, end in boundaries:
        cursor = int(start)
        while cursor < int(end):
            while cursor < int(end) and not bool(positive[cursor].item()):
                cursor += 1
            run_start = cursor
            while cursor < int(end) and bool(positive[cursor].item()):
                cursor += 1
            run_end = cursor
            run_len = int(run_end - run_start)
            if run_len <= 0:
                continue
            keep_count = min(run_len, max(1, math.ceil(ratio * run_len)))
            local_values = query_value[run_start:run_end]
            local_idx = torch.topk(local_values, k=keep_count, largest=True).indices
            support[run_start + local_idx] = True
    return support


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
    """Compare simple family-local target candidates without changing labels."""
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
        "active_training_target_unchanged": True,
        "candidate_usage": "diagnostic_only_family_local_not_training_semantics",
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


def _segment_budget_point_value_for_target_mode(
    *,
    target_mode: str,
    final_score: torch.Tensor,
    q_hit: torch.Tensor,
    ship_query_evidence: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the point value used to build the active segment-budget head target."""
    mode = str(target_mode).lower()
    if mode == QUERY_LOCAL_UTILITY_FACTORIZED_TARGET_MODE:
        return final_score, {
            "segment_budget_target_base_source": "query_local_utility_final_score",
            "segment_budget_target_variant": "active_final_score",
            "segment_budget_target_aggregation": "sum",
            "segment_budget_target_experimental": False,
            "final_success_allowed": True,
        }
    if mode == QUERY_LOCAL_UTILITY_SEGMENT_BUDGET_QUERY_SHIP_MAX_POOL_TARGET_MODE:
        point_value = _query_ship_blend_signal(
            q_hit=q_hit,
            ship_query_evidence=ship_query_evidence,
            valid=valid,
        )
        return point_value, {
            "segment_budget_target_base_source": (
                "normalized_query_hit_probability_plus_normalized_ship_query_evidence"
            ),
            "segment_budget_target_variant": "query_ship_blend_max_pool",
            "segment_budget_target_aggregation": "max_pool",
            "segment_budget_target_point_formula": (
                "0.65_normalized_query_hit_probability_plus_0.35_normalized_ship_query_evidence"
            ),
            "segment_budget_target_experimental": True,
            "final_success_allowed": False,
        }
    if mode == QUERY_LOCAL_UTILITY_QUERY_SHIP_LOCAL_HEADS_TARGET_MODE:
        return q_hit.float().clamp(0.0, 1.0), {
            "segment_budget_target_base_source": "query_ship_local_presence_head_target",
            "segment_budget_target_variant": "query_ship_local_heads_max_pool",
            "segment_budget_target_aggregation": "max_pool",
            "segment_budget_target_point_formula": ("query_ship_local_presence_head_target"),
            "segment_budget_target_experimental": True,
            "final_success_allowed": False,
        }
    raise ValueError(
        "Unsupported QueryLocalUtility target mode: "
        f"{target_mode!r}. Expected one of {sorted(QUERY_LOCAL_UTILITY_TARGET_MODES)!r}."
    )


def build_query_local_utility_targets(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    segment_size: int = 32,
    target_mode: str = QUERY_LOCAL_UTILITY_FACTORIZED_TARGET_MODE,
) -> QueryLocalUtilityTargetBundle:
    """Build factorized QueryLocalUtility labels from learning range workloads only."""
    target_mode = str(target_mode).lower()
    if target_mode not in QUERY_LOCAL_UTILITY_TARGET_MODES:
        raise ValueError(
            "target_mode must be one of "
            f"{sorted(QUERY_LOCAL_UTILITY_TARGET_MODES)!r}; got {target_mode!r}."
        )
    n_points = int(points.shape[0])
    device = points.device
    labels = torch.zeros((n_points, NUM_QUERY_TYPES), dtype=torch.float32, device=device)
    labelled_mask = torch.zeros((n_points, NUM_QUERY_TYPES), dtype=torch.bool, device=device)
    head_targets = torch.zeros(
        (n_points, len(QUERY_LOCAL_UTILITY_HEAD_NAMES)), dtype=torch.float32, device=device
    )
    head_mask = torch.zeros_like(head_targets, dtype=torch.bool)
    range_queries = [
        query for query in typed_queries if str(query.get("type", "")).lower() == "range"
    ]
    if n_points <= 0 or not range_queries:
        diagnostics = factorized_target_diagnostics(
            head_targets, head_mask, QUERY_LOCAL_UTILITY_HEAD_NAMES, boundaries
        )
        return QueryLocalUtilityTargetBundle(
            labels, labelled_mask, head_targets, head_mask, diagnostics
        )

    behavior_base = _trajectory_change_weights(points, boundaries)
    query_hit_count = torch.zeros((n_points,), dtype=torch.float32, device=device)
    behavior_mass = torch.zeros_like(query_hit_count)
    boundary_mass = torch.zeros_like(query_hit_count)
    replacement_mass = torch.zeros_like(query_hit_count)
    ship_query_evidence_mass = torch.zeros_like(query_hit_count)
    query_hit_masks: list[torch.Tensor] = []

    points_cpu = points.detach().cpu()
    for query in range_queries:
        mask = points_in_range_box(points, query["params"]).to(device=device, dtype=torch.bool)
        if not bool(mask.any().item()):
            continue
        query_hit_masks.append(mask)
        query_hit_count[mask] += 1.0
        behavior_mass[mask] += behavior_base[mask]
        for start, end in boundaries:
            local_mask = mask[int(start) : int(end)]
            if not bool(local_mask.any().item()):
                continue
            local_hit_count = int(local_mask.sum().item())
            if local_hit_count <= 0:
                continue
            local_indices = torch.where(local_mask)[0] + int(start)
            ship_query_evidence_mass[local_indices] += float(1.0 / local_hit_count)
        boundary_idx = _boundary_indices_for_query(mask, boundaries)
        if int(boundary_idx.numel()) > 0:
            boundary_mass[boundary_idx] += 1.0
        crossing_idx = segment_box_bracket_indices(points_cpu, boundaries, query["params"]).to(
            device=device
        )
        if int(crossing_idx.numel()) > 0:
            boundary_mass[crossing_idx] += 1.0
        query_value = torch.zeros_like(query_hit_count)
        query_value[mask] = 0.50 + 0.35 * behavior_base[mask]
        if int(boundary_idx.numel()) > 0:
            query_value[boundary_idx] += 0.40
        if int(crossing_idx.numel()) > 0:
            query_value[crossing_idx] += 0.30
        replacement_support = _query_replacement_support(query_value, boundaries)
        replacement_mass[replacement_support] += query_value[replacement_support].clamp(min=0.0)

    query_count = float(max(1, len(range_queries)))
    q_hit = (query_hit_count / query_count).clamp(0.0, 1.0)
    behavior = torch.zeros_like(q_hit)
    hit_positive = query_hit_count > 0
    behavior[hit_positive] = (
        behavior_mass[hit_positive] / query_hit_count[hit_positive].clamp(min=1.0)
    ).clamp(0.0, 1.0)
    boundary = (boundary_mass / query_count).clamp(0.0, 1.0).square()
    replacement = torch.zeros_like(q_hit)
    replacement[hit_positive] = (
        replacement_mass[hit_positive] / query_hit_count[hit_positive].clamp(min=1.0)
    ).clamp(0.0, 1.0)
    ship_query_evidence = (ship_query_evidence_mass / query_count).clamp(0.0, 1.0)
    target_q_hit = q_hit
    target_behavior = behavior
    head_variant_diagnostics: dict[str, Any] = {
        "query_hit_target_variant": "active_query_hit_probability",
        "query_hit_target_base_source": "range_query_point_hit_probability",
        "conditional_behavior_target_variant": "active_local_behavior_change",
        "conditional_behavior_target_base_source": "query_hit_conditioned_trajectory_change",
        "final_label_variant": "active_query_local_utility_point_score",
    }
    if target_mode == QUERY_LOCAL_UTILITY_QUERY_SHIP_LOCAL_HEADS_TARGET_MODE:
        target_q_hit = _query_ship_blend_signal(
            q_hit=q_hit,
            ship_query_evidence=ship_query_evidence,
            valid=hit_positive,
        )
        target_behavior = _query_ship_local_behavior_signal(
            behavior=behavior,
            boundary=boundary,
            replacement=replacement,
            ship_query_evidence=ship_query_evidence,
            valid=hit_positive,
        )
        head_variant_diagnostics.update(
            {
                "query_hit_target_variant": "query_ship_local_presence_utility",
                "query_hit_target_base_source": (
                    "0.65_normalized_query_hit_probability_plus_0.35_normalized_ship_query_evidence"
                ),
                "conditional_behavior_target_variant": "query_ship_local_behavior_utility",
                "conditional_behavior_target_base_source": (
                    "0.45_normalized_ship_query_evidence_plus_"
                    "0.25_normalized_behavior_change_plus_"
                    "0.20_normalized_replacement_value_plus_"
                    "0.10_normalized_boundary_event_utility"
                ),
                "final_label_variant": "query_ship_local_heads_composed_score",
            }
        )
    final_score = query_local_utility_point_score(
        q_hit=target_q_hit,
        behavior=target_behavior,
        boundary=boundary,
        replacement=replacement,
    )
    segment_budget_point_value, segment_budget_variant_diagnostics = (
        _segment_budget_point_value_for_target_mode(
            target_mode=target_mode,
            final_score=final_score,
            q_hit=target_q_hit,
            ship_query_evidence=ship_query_evidence,
            valid=hit_positive,
        )
    )
    segment_budget_aggregation = str(
        segment_budget_variant_diagnostics.get("segment_budget_target_aggregation", "sum")
    )
    if segment_budget_aggregation == "sum":
        segment_budget = _segment_budget_targets(
            segment_budget_point_value,
            boundaries,
            segment_size,
        )
    elif segment_budget_aggregation == "max_pool":
        segment_budget = _segment_pooled_targets(
            segment_budget_point_value,
            boundaries,
            segment_size,
            pool="max",
        )
    else:
        raise ValueError(
            "Unsupported QueryLocalUtility segment-budget aggregation: "
            f"{segment_budget_aggregation!r}."
        )
    path_length_support = query_local_utility_path_length_support_targets(
        points,
        boundaries,
        segment_size=segment_size,
    )
    family_evidence = _range_query_family_evidence(
        points=points,
        boundaries=boundaries,
        range_queries=range_queries,
    )
    head_targets[:, 0] = target_q_hit
    head_targets[:, 1] = target_behavior
    head_targets[:, 2] = boundary
    head_targets[:, 3] = replacement
    head_targets[:, 4] = segment_budget
    head_targets[:, 5] = path_length_support
    head_mask[:] = True
    head_mask[:, 1] = hit_positive

    labels[:, QUERY_TYPE_ID_RANGE] = final_score
    labelled_mask[:, QUERY_TYPE_ID_RANGE] = True
    diagnostics = factorized_target_diagnostics(
        head_targets,
        head_mask,
        QUERY_LOCAL_UTILITY_HEAD_NAMES,
        boundaries,
    )
    diagnostics.update(
        {
            "target_family": "QueryLocalUtilityFactorized",
            "target_mode": target_mode,
            "range_query_count": len(range_queries),
            "segment_size_points": int(segment_size),
            "segment_budget_target_training": "point_repeated_plus_segment_level_listwise_loss",
            **head_variant_diagnostics,
            **segment_budget_variant_diagnostics,
            "segment_budget_segment_level_loss_enabled": True,
            "path_length_support_target_training": "query_free_segment_path_length_removal_loss_highpass",
            "path_length_support_target_base_source": "per_point_path_length_removal_loss_segment_highpass_mass",
            "path_length_support_target_query_free": True,
            "path_length_support_target_highpass_quantile": 0.50,
            "behavior_change_highpass_quantile": 0.70,
            "conditional_behavior_utility_training": "masked_to_query_hit_points",
            "replacement_representative_value_normalization": "conditional_on_query_hit",
            "replacement_value_is_true_counterfactual_marginal_gain": False,
            "final_label_formula": QUERY_LOCAL_UTILITY_FINAL_LABEL_FORMULA,
            "final_boundary_bonus_uses_squared_event_probability": True,
            "final_label_positive_fraction": float((final_score > 0.0).float().mean().item()),
            "final_label_support_fraction_by_threshold": support_fraction_by_threshold(final_score),
            "final_label_mass": float(final_score.sum().item()),
            "conditional_behavior_target_alignment": _conditional_behavior_target_alignment(
                behavior=target_behavior,
                valid=hit_positive,
                references={
                    "final_score": final_score,
                    "query_hit_probability": target_q_hit,
                    "ship_query_evidence": ship_query_evidence,
                    "replacement_representative_value": replacement,
                    "segment_budget_target": segment_budget,
                    "path_length_support_target": path_length_support,
                },
            ),
            "conditional_behavior_candidate_alignment": _conditional_behavior_candidate_diagnostics(
                current_behavior=target_behavior,
                valid=hit_positive,
                replacement=replacement,
                segment_budget=segment_budget,
                references={
                    "final_score": final_score,
                    "query_hit_probability": target_q_hit,
                    "ship_query_evidence": ship_query_evidence,
                    "replacement_representative_value": replacement,
                    "segment_budget_target": segment_budget,
                    "path_length_support_target": path_length_support,
                },
                query_hit_masks=query_hit_masks,
                boundaries=boundaries,
            ),
            "ship_query_evidence_target_alignment": _ship_query_evidence_target_alignment(
                ship_query_evidence=ship_query_evidence,
                valid=hit_positive,
                rankers={
                    "final_score": final_score,
                    "query_hit_probability": target_q_hit,
                    "conditional_behavior_utility": target_behavior,
                    "boundary_event_utility": boundary,
                    "replacement_representative_value": replacement,
                    "segment_budget_target": segment_budget,
                    "path_length_support_target": path_length_support,
                },
                query_hit_masks=query_hit_masks,
                boundaries=boundaries,
            ),
            "family_conditioned_target_trainability": (
                _family_conditioned_target_trainability_diagnostics(
                    family_evidence=family_evidence,
                    rankers={
                        "final_score": final_score,
                        "query_hit_probability": target_q_hit,
                        "conditional_behavior_utility": target_behavior,
                        "boundary_event_utility": boundary,
                        "replacement_representative_value": replacement,
                        "segment_budget_target": segment_budget,
                        "path_length_support_target": path_length_support,
                    },
                    boundaries=boundaries,
                )
            ),
            "family_local_target_candidate_alignment": (
                _family_local_target_candidate_alignment(
                    family_evidence=family_evidence,
                    rankers={
                        "final_score": final_score,
                        "query_hit_probability": target_q_hit,
                        "conditional_behavior_utility": target_behavior,
                        "boundary_event_utility": boundary,
                        "replacement_representative_value": replacement,
                        "segment_budget_target": segment_budget,
                        "path_length_support_target": path_length_support,
                    },
                    boundaries=boundaries,
                    segment_size=segment_size,
                )
            ),
            "segment_budget_ship_presence_candidate_alignment": (
                _segment_budget_ship_presence_candidate_alignment(
                    active_segment_budget=segment_budget,
                    final_score=final_score,
                    q_hit=target_q_hit,
                    ship_query_evidence=ship_query_evidence,
                    valid=hit_positive,
                    query_hit_masks=query_hit_masks,
                    boundaries=boundaries,
                    segment_size=segment_size,
                )
            ),
        }
    )
    return QueryLocalUtilityTargetBundle(
        labels, labelled_mask, head_targets, head_mask, diagnostics
    )
