"""Trace payload construction for learned segment-budget selector runs."""

from __future__ import annotations

from typing import Any

import torch

from selection.learned_segment_budget.constants import (
    GEOMETRY_TIE_BREAKER_WEIGHT,
    LEARNED_SEGMENT_BUDGET_TRACE_SCHEMA_VERSION,
    SEGMENT_ALLOCATION_WEIGHT_FLOOR,
    SEGMENT_SCORE_POINT_BLEND_WEIGHT,
)
from selection.learned_segment_budget.diagnostics import (
    _entropy,
    _mask_indices_payload,
    _segment_allocation_alignment_diagnostics,
    _segment_source_attribution,
    _selector_geometry_diagnostics,
    _trajectory_counts,
)


def _selector_trace(
    *,
    retained: torch.Tensor,
    skeleton_mask: torch.Tensor,
    learned_mask: torch.Tensor,
    fallback_mask: torch.Tensor,
    length_repair_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
    budget: int,
    skeleton_cap: int,
    segment_rows: list[dict[str, Any]] | None,
    segment_allocations: dict[int, int],
    segment_count: int,
    segment_score_source: str,
    segment_score_stats: dict[str, float | int] | None = None,
    segment_budget_allocation_method: str = "none",
    fairness_preallocation_enabled: bool = True,
    geometry_gain_weight: float = GEOMETRY_TIE_BREAKER_WEIGHT,
    segment_length_support_weight: float = 0.0,
    segment_allocation_weight_floor: float = SEGMENT_ALLOCATION_WEIGHT_FLOOR,
    segment_score_point_blend_weight: float = SEGMENT_SCORE_POINT_BLEND_WEIGHT,
    length_repair_fraction: float = 0.0,
    length_repair_score_protection_fraction: float = 0.0,
    length_repair_swap_count: int = 0,
    length_repair_protected_mask: torch.Tensor | None = None,
    points: torch.Tensor | None = None,
    allocation_point_selection_diagnostics: dict[str, Any] | None = None,
    allocation_counterfactual_diagnostics: dict[str, Any] | None = None,
    pre_repair_segment_source_attribution: dict[str, Any] | None = None,
    pre_repair_retained_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Return JSON-serializable attribution for the retained mask."""
    retained_count = int(retained.sum().item())
    skeleton_count = int(skeleton_mask.sum().item())
    learned_count = int(learned_mask.sum().item())
    fallback_count = int(fallback_mask.sum().item())
    length_repair_count = int(length_repair_mask.sum().item())
    protected_count = (
        0
        if length_repair_protected_mask is None
        else int(length_repair_protected_mask.sum().item())
    )
    attributed = skeleton_mask | learned_mask | fallback_mask | length_repair_mask
    unattributed_count = int((retained & ~attributed).sum().item())
    trajectory_learned_counts = _trajectory_counts(learned_mask, boundaries)
    trajectories_with_learned = sum(1 for count in trajectory_learned_counts if int(count) > 0)
    valid_trajectory_count = sum(1 for start, end in boundaries if int(end - start) > 0)
    entropy, entropy_normalized = _entropy(list(segment_allocations.values()))
    return {
        "schema_version": int(LEARNED_SEGMENT_BUDGET_TRACE_SCHEMA_VERSION),
        "selector_type": "learned_segment_budget_v1",
        "compression_ratio": float(compression_ratio),
        "total_point_count": int(retained.numel()),
        "total_budget_count": int(budget),
        "retained_count": retained_count,
        "minimal_skeleton_slot_cap": int(skeleton_cap),
        "skeleton_retained_count": skeleton_count,
        "skeleton_cap_exceeded_for_endpoint_sanity": bool(skeleton_count > int(skeleton_cap)),
        "learned_controlled_retained_slots": learned_count,
        "learned_controlled_retained_slot_fraction": float(learned_count / max(1, int(budget))),
        "learned_fraction_of_retained_count": float(learned_count / max(1, retained_count)),
        "fallback_retained_count": fallback_count,
        "length_repair_retained_count": length_repair_count,
        "length_repair_fraction": float(length_repair_fraction),
        "length_repair_score_protection_fraction": float(length_repair_score_protection_fraction),
        "length_repair_score_protected_count": int(protected_count),
        "length_repair_score_protected_fraction_of_budget": float(
            protected_count / max(1, int(budget))
        ),
        "length_repair_swap_count": int(length_repair_swap_count),
        "unattributed_retained_count": unattributed_count,
        "trajectory_count": int(valid_trajectory_count),
        "trajectories_with_at_least_one_learned_decision": int(trajectories_with_learned),
        "trajectories_with_zero_learned_decisions": int(
            max(0, valid_trajectory_count - trajectories_with_learned)
        ),
        "trajectory_learned_decision_counts": trajectory_learned_counts,
        "trajectory_skeleton_counts": _trajectory_counts(skeleton_mask, boundaries),
        "trajectory_fallback_counts": _trajectory_counts(fallback_mask, boundaries),
        "segments_considered_count": int(segment_count),
        "segments_with_learned_budget": int(
            sum(1 for count in segment_allocations.values() if int(count) > 0)
        ),
        "segment_budget_allocation_count": int(
            sum(int(count) for count in segment_allocations.values())
        ),
        "segment_budget_entropy": entropy,
        "segment_budget_entropy_normalized": entropy_normalized,
        "segment_score_source": str(segment_score_source),
        "segment_budget_allocation_method": str(segment_budget_allocation_method),
        "trajectory_fairness_preallocation_enabled": bool(fairness_preallocation_enabled),
        "geometry_tie_breaker_weight": float(geometry_gain_weight),
        "segment_length_support_weight": float(segment_length_support_weight),
        "segment_allocation_weight_floor": float(segment_allocation_weight_floor),
        "segment_score_point_blend_weight": float(segment_score_point_blend_weight),
        "no_fixed_85_percent_temporal_scaffold": True,
        "point_attribution_available": True,
        "geometry_diagnostics": _selector_geometry_diagnostics(
            points=points,
            retained=retained,
            skeleton_mask=skeleton_mask,
            learned_mask=learned_mask,
            fallback_mask=fallback_mask,
            length_repair_mask=length_repair_mask,
            boundaries=boundaries,
        ),
        "allocation_point_selection_diagnostics": (
            allocation_point_selection_diagnostics
            if allocation_point_selection_diagnostics is not None
            else {"available": False, "reason": "not_run"}
        ),
        "allocation_counterfactual_diagnostics": (
            allocation_counterfactual_diagnostics
            if allocation_counterfactual_diagnostics is not None
            else {"available": False, "reason": "not_run"}
        ),
        "segment_allocation_alignment_diagnostics": _segment_allocation_alignment_diagnostics(
            segment_rows=[] if segment_rows is None else segment_rows,
            segment_allocations=segment_allocations,
        ),
        "segment_source_attribution": _segment_source_attribution(
            segment_rows=[] if segment_rows is None else segment_rows,
            segment_allocations=segment_allocations,
            retained=retained,
            skeleton_mask=skeleton_mask,
            learned_mask=learned_mask,
            fallback_mask=fallback_mask,
            length_repair_mask=length_repair_mask,
        ),
        "pre_repair_segment_source_attribution": (
            pre_repair_segment_source_attribution
            if pre_repair_segment_source_attribution is not None
            else {"available": False, "reason": "not_run"}
        ),
        "pre_repair_retained_mask": _mask_indices_payload(pre_repair_retained_mask),
        **(segment_score_stats or {}),
    }
