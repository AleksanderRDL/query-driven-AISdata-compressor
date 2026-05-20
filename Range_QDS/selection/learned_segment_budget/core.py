"""learned_segment_budget selector public orchestration."""

from __future__ import annotations

import math
from typing import Any

import torch

from selection.learned_segment_budget.allocation import (
    _allocate_segment_budgets,
    _apply_segment_transfer_calibration,
    _max_skeleton_fraction,
    _segment_rows,
    _segment_score_stats,
    _total_budget,
)
from selection.learned_segment_budget.constants import (
    GEOMETRY_TIE_BREAKER_WEIGHT,
    LEARNED_SEGMENT_BUDGET_SCHEMA_VERSION,
    LEARNED_SEGMENT_BUDGET_TRACE_SCHEMA_VERSION,
    SEGMENT_ALLOCATION_WEIGHT_FLOOR,
    SEGMENT_LENGTH_SUPPORT_ALLOCATION_WEIGHT,
    SEGMENT_SCORE_POINT_BLEND_WEIGHT,
    SEGMENT_TRANSFER_CALIBRATION_MODE_CHOICES,
    SEGMENT_TRANSFER_CALIBRATION_MODE_NONE,
)
from selection.learned_segment_budget.diagnostics import (
    _allocation_counterfactual_diagnostics,
    _allocation_point_selection_diagnostics,
    _segment_source_attribution,
)
from selection.learned_segment_budget.length_repair import (
    _apply_length_repair_swaps,
    _select_with_spacing,
)
from selection.learned_segment_budget.trace import _selector_trace
from selection.retained_mask_selectors import deterministic_topk_with_jitter
from selection.selector_types import LEARNED_SEGMENT_BUDGET_SELECTOR_TYPE

__all__ = [
    "GEOMETRY_TIE_BREAKER_WEIGHT",
    "LEARNED_SEGMENT_BUDGET_SCHEMA_VERSION",
    "LEARNED_SEGMENT_BUDGET_TRACE_SCHEMA_VERSION",
    "SEGMENT_ALLOCATION_WEIGHT_FLOOR",
    "SEGMENT_LENGTH_SUPPORT_ALLOCATION_WEIGHT",
    "SEGMENT_SCORE_POINT_BLEND_WEIGHT",
    "SEGMENT_TRANSFER_CALIBRATION_MODE_CHOICES",
    "SEGMENT_TRANSFER_CALIBRATION_MODE_NONE",
    "blend_segment_support_scores",
    "learned_segment_budget_diagnostics",
    "simplify_with_learned_segment_budget",
    "simplify_with_learned_segment_budget_with_trace",
]


def blend_segment_support_scores(
    *,
    segment_scores: torch.Tensor | None,
    path_length_support_scores: torch.Tensor | None,
    path_length_support_weight: float,
) -> torch.Tensor | None:
    """Blend query segment scores with learned query-free path-length support scores."""
    weight = max(0.0, min(1.0, float(path_length_support_weight)))
    if weight <= 0.0 or path_length_support_scores is None:
        return None if segment_scores is None else segment_scores.detach().cpu().float()
    path_scores = path_length_support_scores.detach().cpu().float()
    if segment_scores is None:
        return path_scores
    segment = segment_scores.detach().cpu().float()
    if segment.shape != path_scores.shape:
        raise ValueError(
            "segment_scores and path_length_support_scores must have matching shape: "
            f"got {tuple(segment.shape)} and {tuple(path_scores.shape)}."
        )
    return (1.0 - weight) * segment + weight * path_scores


def simplify_with_learned_segment_budget_with_trace(
    scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
    *,
    segment_size: int = 32,
    min_temporal_spacing_fraction_within_segment: float = 0.10,
    max_budget_share_per_trajectory: float = 0.20,
    segment_scores: torch.Tensor | None = None,
    segment_point_scores: torch.Tensor | None = None,
    points: torch.Tensor | None = None,
    geometry_gain_weight: float = GEOMETRY_TIE_BREAKER_WEIGHT,
    segment_length_support_weight: float = SEGMENT_LENGTH_SUPPORT_ALLOCATION_WEIGHT,
    segment_allocation_weight_floor: float = SEGMENT_ALLOCATION_WEIGHT_FLOOR,
    segment_score_point_blend_weight: float = SEGMENT_SCORE_POINT_BLEND_WEIGHT,
    segment_transfer_calibration_mode: str = SEGMENT_TRANSFER_CALIBRATION_MODE_NONE,
    fairness_preallocation_enabled: bool = True,
    length_repair_fraction: float = 0.0,
    length_repair_score_protection_fraction: float = 0.0,
    segment_score_source_label: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Retain points and return skeleton/learned/fallback attribution."""
    point_total = int(scores.numel())
    retained = torch.zeros((point_total,), dtype=torch.bool, device=scores.device)
    skeleton_mask = torch.zeros_like(retained)
    learned_mask = torch.zeros_like(retained)
    fallback_mask = torch.zeros_like(retained)
    length_repair_mask = torch.zeros_like(retained)
    length_repair_protected_mask = torch.zeros_like(retained)
    if point_total <= 0:
        trace = _selector_trace(
            retained=retained,
            skeleton_mask=skeleton_mask,
            learned_mask=learned_mask,
            fallback_mask=fallback_mask,
            length_repair_mask=length_repair_mask,
            boundaries=boundaries,
            compression_ratio=compression_ratio,
            budget=0,
            skeleton_cap=0,
            segment_rows=[],
            segment_allocations={},
            segment_count=0,
            segment_score_source="none",
            segment_budget_allocation_method="none",
            fairness_preallocation_enabled=fairness_preallocation_enabled,
            geometry_gain_weight=geometry_gain_weight,
            segment_length_support_weight=segment_length_support_weight,
            segment_allocation_weight_floor=segment_allocation_weight_floor,
            segment_score_point_blend_weight=segment_score_point_blend_weight,
            segment_transfer_calibration_summary={
                "mode": SEGMENT_TRANSFER_CALIBRATION_MODE_NONE,
                "applied": False,
                "reason": "no_points",
                "uses_post_selection_attribution": False,
                "uses_length_support_counter_signal": False,
                "base_segment_length_support_weight": float(segment_length_support_weight),
                "effective_segment_length_support_weight": float(segment_length_support_weight),
            },
            length_repair_fraction=0.0,
            length_repair_score_protection_fraction=0.0,
            length_repair_swap_count=0,
            length_repair_protected_mask=length_repair_protected_mask,
            points=points,
        )
        return retained, trace
    budget = min(point_total, _total_budget(boundaries, compression_ratio))
    skeleton_cap = math.floor(float(budget) * _max_skeleton_fraction(compression_ratio))

    skeleton_count = 0
    skeleton_candidates: list[tuple[float, int, int]] = []
    for trajectory_id, (start, end) in enumerate(boundaries):
        count = int(end - start)
        if count <= 0:
            continue
        local_budget = min(count, max(2, math.ceil(float(compression_ratio) * count)))
        if local_budget >= 2:
            for point_idx in (int(start), int(end - 1)):
                if not retained[point_idx]:
                    retained[point_idx] = True
                    skeleton_mask[point_idx] = True
                    skeleton_count += 1
        else:
            mid = int(start + count // 2)
            candidates = torch.tensor(
                [int(start), mid, int(end - 1)], dtype=torch.long, device=scores.device
            ).unique()
            best = candidates[torch.argmax(scores[candidates].float())]
            skeleton_candidates.append(
                (float(scores[best].item()), trajectory_id, int(best.item()))
            )
    optional_skeleton_slots = max(0, int(skeleton_cap) - int(skeleton_count))
    if optional_skeleton_slots > 0 and skeleton_candidates:
        skeleton_candidates.sort(key=lambda item: (item[0], -item[2]), reverse=True)
        for _score, _trajectory_id, point_idx in skeleton_candidates[:optional_skeleton_slots]:
            if not retained[point_idx]:
                retained[point_idx] = True
                skeleton_mask[point_idx] = True
                skeleton_count += 1
                if skeleton_count >= skeleton_cap:
                    break

    remaining = budget - int(retained.sum().item())
    if remaining <= 0:
        trace = _selector_trace(
            retained=retained,
            skeleton_mask=skeleton_mask,
            learned_mask=learned_mask,
            fallback_mask=fallback_mask,
            length_repair_mask=length_repair_mask,
            boundaries=boundaries,
            compression_ratio=compression_ratio,
            budget=budget,
            skeleton_cap=skeleton_cap,
            segment_rows=[],
            segment_allocations={},
            segment_count=0,
            segment_score_source="none",
            segment_budget_allocation_method="none",
            fairness_preallocation_enabled=fairness_preallocation_enabled,
            geometry_gain_weight=geometry_gain_weight,
            segment_length_support_weight=segment_length_support_weight,
            segment_allocation_weight_floor=segment_allocation_weight_floor,
            segment_score_point_blend_weight=segment_score_point_blend_weight,
            segment_transfer_calibration_summary={
                "mode": str(segment_transfer_calibration_mode).lower(),
                "applied": False,
                "reason": "no_learned_slots",
                "uses_post_selection_attribution": False,
                "uses_length_support_counter_signal": False,
                "base_segment_length_support_weight": float(segment_length_support_weight),
                "effective_segment_length_support_weight": float(segment_length_support_weight),
            },
            length_repair_fraction=0.0,
            length_repair_score_protection_fraction=0.0,
            length_repair_swap_count=0,
            length_repair_protected_mask=length_repair_protected_mask,
            points=points,
        )
        return retained, trace

    if segment_scores is not None and int(segment_scores.numel()) != point_total:
        raise ValueError(
            "segment_scores must match scores length: "
            f"got {int(segment_scores.numel())}, expected {point_total}."
        )
    if segment_point_scores is not None and int(segment_point_scores.numel()) != point_total:
        raise ValueError(
            "segment_point_scores must match scores length: "
            f"got {int(segment_point_scores.numel())}, expected {point_total}."
        )
    point_segment_scores = segment_scores if segment_point_scores is None else segment_point_scores
    segment_rows = _segment_rows(
        scores,
        boundaries,
        segment_size,
        segment_scores=segment_scores,
        points=points,
    )
    segment_rows.sort(key=lambda row: (float(row["score"]), -int(row["start"])), reverse=True)
    segment_score_source = (
        str(segment_score_source_label)
        if segment_score_source_label is not None
        else "segment_budget_head_top20_mean"
        if segment_scores is not None
        else "point_score_top20_mean"
    )
    effective_segment_length_support_weight = (
        float(segment_length_support_weight) if points is not None else 0.0
    )
    segment_transfer_calibration_summary, effective_segment_length_support_weight = (
        _apply_segment_transfer_calibration(
            segment_rows,
            mode=segment_transfer_calibration_mode,
            segment_length_support_weight=effective_segment_length_support_weight,
            segment_allocation_weight_floor=float(segment_allocation_weight_floor),
        )
    )
    if bool(segment_transfer_calibration_summary.get("applied", False)):
        segment_rows.sort(key=lambda row: (float(row["score"]), -int(row["start"])), reverse=True)
        segment_score_source = (
            f"{segment_score_source}+{segment_transfer_calibration_summary['mode']}"
        )
    segment_score_stats = _segment_score_stats(segment_rows)
    segment_allocations = _allocate_segment_budgets(
        segment_rows=segment_rows,
        retained=retained,
        remaining=remaining,
        budget=budget,
        boundaries=boundaries,
        max_budget_share_per_trajectory=max_budget_share_per_trajectory,
        fairness_preallocation_enabled=fairness_preallocation_enabled,
        segment_length_support_weight=effective_segment_length_support_weight,
        segment_allocation_weight_floor=float(segment_allocation_weight_floor),
    )
    skeleton_retained_for_diagnostic = retained.detach().cpu().bool().clone()

    for segment_idx, keep_count in segment_allocations.items():
        row = segment_rows[segment_idx]
        start = int(row["start"])
        end = int(row["end"])
        trajectory_id = int(row["trajectory_id"])
        trajectory_start, trajectory_end = boundaries[trajectory_id]
        trajectory_start = int(trajectory_start)
        trajectory_end = int(trajectory_end)
        trajectory_scores = torch.full(
            (max(0, trajectory_end - trajectory_start),),
            -float("inf"),
            dtype=torch.float32,
            device=scores.device,
        )
        segment_local_start = max(0, start - trajectory_start)
        segment_local_end = min(int(trajectory_scores.numel()), end - trajectory_start)
        if segment_local_end <= segment_local_start:
            continue
        trajectory_scores[segment_local_start:segment_local_end] = scores[start:end].float()
        local_retained = retained[trajectory_start:trajectory_end]
        existing = torch.where(local_retained)[0]
        min_spacing = math.floor(
            float(end - start) * float(min_temporal_spacing_fraction_within_segment)
        )
        segment_aux_scores = None
        segment_score_weight = 0.0
        if point_segment_scores is not None:
            segment_aux_scores = torch.full_like(trajectory_scores, -float("inf"))
            segment_aux_scores[segment_local_start:segment_local_end] = point_segment_scores[
                start:end
            ].to(
                device=trajectory_scores.device,
                dtype=torch.float32,
            )
            segment_aux_local_scores = segment_aux_scores[segment_local_start:segment_local_end]
            segment_score_finite = torch.isfinite(segment_aux_local_scores)
            if bool(segment_score_finite.any().item()):
                segment_score_weight = float(segment_score_point_blend_weight)
        selected = _select_with_spacing(
            trajectory_scores,
            int(keep_count),
            trajectory_id=trajectory_id,
            existing_indices=existing,
            min_spacing=min_spacing,
            local_points=None if points is None else points[trajectory_start:trajectory_end],
            geometry_gain_weight=float(geometry_gain_weight),
            segment_aux_scores=segment_aux_scores,
            segment_score_weight=float(segment_score_weight)
            if point_segment_scores is not None
            else 0.0,
        )
        absolute_selected = trajectory_start + selected
        new_selected = absolute_selected[~retained[absolute_selected]]
        retained[new_selected] = True
        learned_mask[new_selected] = True

    if int(retained.sum().item()) < budget:
        candidate_scores = scores.float().clone()
        candidate_scores[retained] = -float("inf")
        missing = min(
            budget - int(retained.sum().item()), int(torch.isfinite(candidate_scores).sum().item())
        )
        if missing > 0:
            fallback_selected = deterministic_topk_with_jitter(
                candidate_scores, missing, point_total + 31337
            )
            retained[fallback_selected] = True
            fallback_mask[fallback_selected] = True
    pre_repair_retained_for_diagnostic = retained.detach().cpu().bool().clone()
    pre_repair_learned_for_diagnostic = learned_mask.detach().cpu().bool().clone()
    pre_repair_fallback_for_diagnostic = fallback_mask.detach().cpu().bool().clone()
    pre_repair_length_repair_for_diagnostic = torch.zeros_like(pre_repair_retained_for_diagnostic)
    allocation_diagnostics = _allocation_point_selection_diagnostics(
        scores=scores,
        points=points,
        boundaries=boundaries,
        compression_ratio=float(compression_ratio),
        segment_rows=segment_rows,
        segment_allocations=segment_allocations,
        skeleton_retained=skeleton_retained_for_diagnostic,
        primary_retained=pre_repair_retained_for_diagnostic,
        budget=budget,
        min_temporal_spacing_fraction_within_segment=float(
            min_temporal_spacing_fraction_within_segment
        ),
    )
    allocation_counterfactual_diagnostics = _allocation_counterfactual_diagnostics(
        scores=scores,
        points=points,
        boundaries=boundaries,
        compression_ratio=float(compression_ratio),
        segment_rows=segment_rows,
        segment_allocations=segment_allocations,
        skeleton_retained=skeleton_retained_for_diagnostic,
        budget=budget,
        max_budget_share_per_trajectory=float(max_budget_share_per_trajectory),
        fairness_preallocation_enabled=bool(fairness_preallocation_enabled),
        segment_allocation_weight_floor=float(segment_allocation_weight_floor),
        min_temporal_spacing_fraction_within_segment=float(
            min_temporal_spacing_fraction_within_segment
        ),
    )
    pre_repair_source_attribution = _segment_source_attribution(
        segment_rows=segment_rows,
        segment_allocations=segment_allocations,
        retained=pre_repair_retained_for_diagnostic,
        skeleton_mask=skeleton_retained_for_diagnostic,
        learned_mask=pre_repair_learned_for_diagnostic,
        fallback_mask=pre_repair_fallback_for_diagnostic,
        length_repair_mask=pre_repair_length_repair_for_diagnostic,
    )
    length_repair_swap_count = _apply_length_repair_swaps(
        scores=scores,
        points=points,
        boundaries=boundaries,
        retained=retained,
        learned_mask=learned_mask,
        fallback_mask=fallback_mask,
        length_repair_mask=length_repair_mask,
        repair_fraction=float(length_repair_fraction),
        score_protection_fraction=float(length_repair_score_protection_fraction),
        length_repair_protected_mask=length_repair_protected_mask,
    )
    trace = _selector_trace(
        retained=retained,
        skeleton_mask=skeleton_mask,
        learned_mask=learned_mask,
        fallback_mask=fallback_mask,
        length_repair_mask=length_repair_mask,
        boundaries=boundaries,
        compression_ratio=compression_ratio,
        budget=budget,
        skeleton_cap=skeleton_cap,
        segment_rows=segment_rows,
        segment_allocations=segment_allocations,
        segment_count=len(segment_rows),
        segment_score_source=segment_score_source,
        segment_score_stats=segment_score_stats,
        segment_budget_allocation_method="score_weighted_diminishing_priority",
        fairness_preallocation_enabled=fairness_preallocation_enabled,
        geometry_gain_weight=geometry_gain_weight,
        segment_length_support_weight=(
            effective_segment_length_support_weight
        ),
        segment_allocation_weight_floor=float(segment_allocation_weight_floor),
        segment_score_point_blend_weight=segment_score_point_blend_weight,
        segment_transfer_calibration_summary=segment_transfer_calibration_summary,
        length_repair_fraction=float(length_repair_fraction),
        length_repair_score_protection_fraction=float(length_repair_score_protection_fraction),
        length_repair_swap_count=length_repair_swap_count,
        length_repair_protected_mask=length_repair_protected_mask,
        points=points,
        allocation_point_selection_diagnostics=allocation_diagnostics,
        allocation_counterfactual_diagnostics=allocation_counterfactual_diagnostics,
        pre_repair_segment_source_attribution=pre_repair_source_attribution,
        pre_repair_retained_mask=pre_repair_retained_for_diagnostic,
    )
    return retained, trace


def simplify_with_learned_segment_budget(
    scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
    *,
    segment_size: int = 32,
    min_temporal_spacing_fraction_within_segment: float = 0.10,
    max_budget_share_per_trajectory: float = 0.20,
    segment_scores: torch.Tensor | None = None,
    segment_point_scores: torch.Tensor | None = None,
    points: torch.Tensor | None = None,
    geometry_gain_weight: float = GEOMETRY_TIE_BREAKER_WEIGHT,
    segment_length_support_weight: float = SEGMENT_LENGTH_SUPPORT_ALLOCATION_WEIGHT,
    segment_allocation_weight_floor: float = SEGMENT_ALLOCATION_WEIGHT_FLOOR,
    segment_score_point_blend_weight: float = SEGMENT_SCORE_POINT_BLEND_WEIGHT,
    segment_transfer_calibration_mode: str = SEGMENT_TRANSFER_CALIBRATION_MODE_NONE,
    fairness_preallocation_enabled: bool = True,
    length_repair_fraction: float = 0.0,
    length_repair_score_protection_fraction: float = 0.0,
    segment_score_source_label: str | None = None,
) -> torch.Tensor:
    """Retain a minimal skeleton, then allocate remaining budget by learned segment value."""
    retained, _trace = simplify_with_learned_segment_budget_with_trace(
        scores,
        boundaries,
        compression_ratio,
        segment_size=segment_size,
        min_temporal_spacing_fraction_within_segment=min_temporal_spacing_fraction_within_segment,
        max_budget_share_per_trajectory=max_budget_share_per_trajectory,
        segment_scores=segment_scores,
        segment_point_scores=segment_point_scores,
        points=points,
        geometry_gain_weight=geometry_gain_weight,
        segment_length_support_weight=segment_length_support_weight,
        segment_allocation_weight_floor=segment_allocation_weight_floor,
        segment_score_point_blend_weight=segment_score_point_blend_weight,
        segment_transfer_calibration_mode=segment_transfer_calibration_mode,
        fairness_preallocation_enabled=fairness_preallocation_enabled,
        length_repair_fraction=length_repair_fraction,
        length_repair_score_protection_fraction=length_repair_score_protection_fraction,
        segment_score_source_label=segment_score_source_label,
    )
    return retained


def learned_segment_budget_diagnostics(
    boundaries: list[tuple[int, int]],
    compression_ratios: list[float] | tuple[float, ...],
) -> dict[str, Any]:
    """Return selector contribution diagnostics independent of model scores."""
    rows: list[dict[str, Any]] = []
    trajectory_count = sum(1 for start, end in boundaries if int(end - start) > 0)
    for ratio in compression_ratios:
        budget = _total_budget(boundaries, float(ratio))
        skeleton_cap = math.floor(float(budget) * _max_skeleton_fraction(float(ratio)))
        learned_slots = max(0, int(budget) - int(skeleton_cap))
        rows.append(
            {
                "compression_ratio": float(ratio),
                "trajectory_count": int(trajectory_count),
                "total_budget_count": int(budget),
                "minimal_skeleton_slot_cap": int(skeleton_cap),
                "learned_slot_count": int(learned_slots),
                "learned_slot_fraction_of_budget": float(learned_slots / max(1, int(budget))),
                "no_fixed_85_percent_temporal_scaffold": True,
            }
        )
    return {
        "schema_version": int(LEARNED_SEGMENT_BUDGET_SCHEMA_VERSION),
        "selector_type": LEARNED_SEGMENT_BUDGET_SELECTOR_TYPE,
        "budget_rows": rows,
    }
