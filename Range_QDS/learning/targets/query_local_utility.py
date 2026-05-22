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
from learning.targets.query_local_utility_core import (
    _normalize_0_1,
    query_local_utility_point_score,
)
from learning.targets.query_local_utility_family import _range_query_family_evidence
from learning.targets.query_local_utility_segments import (
    _segment_budget_targets,
    _segment_pooled_targets,
    query_local_utility_path_length_support_targets,
)
from learning.targets.query_local_utility_target_diagnostics import (
    _conditional_behavior_candidate_diagnostics,
    _conditional_behavior_replacement_partial_alignment,
    _conditional_behavior_target_alignment,
    _family_conditioned_target_trainability_diagnostics,
    _family_local_target_candidate_alignment,
    _segment_budget_ship_presence_candidate_alignment,
    _ship_query_evidence_target_alignment,
)
from learning.targets.query_local_utility_target_diagnostics import (
    _rank_correlation as _rank_correlation,
)
from learning.targets.query_local_utility_target_diagnostics import (
    _topk_overlap_and_mass_recall as _topk_overlap_and_mass_recall,
)
from workloads.query_types import NUM_QUERY_TYPES, QUERY_TYPE_ID_RANGE, validated_range_query_params
from workloads.range_geometry import points_in_range_box, segment_box_bracket_indices

QUERY_LOCAL_UTILITY_FACTORIZED_TARGET_MODE = "query_local_utility_factorized"
QUERY_LOCAL_UTILITY_TARGET_MODES = frozenset(
    {
        QUERY_LOCAL_UTILITY_FACTORIZED_TARGET_MODE,
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
    "additive_raw_query_hit_and_behavior_with_conditional_replacement_modulation_plus_boundary"
)
QUERY_LOCAL_UTILITY_REPLACEMENT_KEEP_FRACTION = 0.35
QUERY_LOCAL_UTILITY_QUERY_EVIDENCE_BASE_WEIGHT = 0.65
QUERY_LOCAL_UTILITY_QUERY_EVIDENCE_SHIP_MULTIPLIER_WEIGHT = 0.35


@dataclass
class QueryLocalUtilityTargetBundle:
    """Scalar and factorized training labels for QueryLocalUtility."""

    labels: torch.Tensor
    labelled_mask: torch.Tensor
    head_targets: torch.Tensor
    head_mask: torch.Tensor
    diagnostics: dict[str, Any]


def _query_segment_local_behavior_signal(
    *,
    behavior: torch.Tensor,
    q_hit: torch.Tensor,
    valid: torch.Tensor,
    boundaries: list[tuple[int, int]],
    segment_size: int,
) -> torch.Tensor:
    """Return behavior utility coupled to query-hit and segment support."""
    valid = valid.to(dtype=torch.bool)
    normalized_behavior = _normalize_0_1(behavior, valid)
    segment_behavior = _segment_pooled_targets(
        normalized_behavior * q_hit.float().clamp(0.0, 1.0),
        boundaries,
        segment_size,
        pool="top20_mean",
    )
    segment_query = _segment_pooled_targets(
        q_hit.float().clamp(0.0, 1.0),
        boundaries,
        segment_size,
        pool="top20_mean",
    )
    segment_multiplier = (
        0.45
        + 0.35 * _normalize_0_1(segment_behavior, valid)
        + 0.20 * _normalize_0_1(segment_query, valid)
    ).clamp(0.0, 1.0)
    utility = (normalized_behavior * segment_multiplier).clamp(0.0, 1.0)
    return torch.where(valid, utility, torch.zeros_like(utility))


def _query_evidence_multiplier_candidate(
    *,
    ship_query_evidence: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return a bounded ship-evidence multiplier on supported query-hit points."""
    valid = valid.to(dtype=torch.bool)
    zero = torch.zeros_like(ship_query_evidence, dtype=torch.float32)
    if not bool(valid.any().item()):
        return zero
    normalized_ship = _normalize_0_1(ship_query_evidence, valid)
    return torch.where(valid, normalized_ship, zero)


def _query_evidence_gate_target(
    *,
    q_hit: torch.Tensor,
    ship_query_evidence: torch.Tensor,
    hit_positive: torch.Tensor,
    family_evidence: dict[str, dict[str, dict[str, Any]]],
) -> torch.Tensor:
    """Return a raw-q-hit-scale-preserving query-evidence head target."""
    q_hit = q_hit.float().clamp(0.0, 1.0)
    hit_positive = hit_positive.to(dtype=torch.bool)
    multiplier_candidates = [
        _query_evidence_multiplier_candidate(
            ship_query_evidence=ship_query_evidence,
            valid=hit_positive,
        )
    ]
    for family_rows in family_evidence.values():
        for evidence in family_rows.values():
            family_q_hit = evidence.get("query_hit_probability")
            family_ship = evidence.get("ship_query_evidence")
            if not isinstance(family_q_hit, torch.Tensor):
                continue
            if not isinstance(family_ship, torch.Tensor):
                continue
            if family_q_hit.shape != q_hit.shape or family_ship.shape != q_hit.shape:
                continue
            family_valid = (family_q_hit > 0.0) & hit_positive
            multiplier = _query_evidence_multiplier_candidate(
                ship_query_evidence=family_ship.float().clamp(0.0, 1.0),
                valid=family_valid,
            )
            if bool((multiplier > 0.0).any().item()):
                multiplier_candidates.append(multiplier)
    stacked = torch.stack(multiplier_candidates, dim=0)
    positive = stacked > 0.0
    positive_count = positive.sum(dim=0).clamp(min=1)
    evidence_multiplier = (stacked.sum(dim=0) / positive_count.to(dtype=stacked.dtype)).clamp(
        0.0, 1.0
    )
    query_evidence_gate = (
        QUERY_LOCAL_UTILITY_QUERY_EVIDENCE_BASE_WEIGHT
        + QUERY_LOCAL_UTILITY_QUERY_EVIDENCE_SHIP_MULTIPLIER_WEIGHT * evidence_multiplier
    ).clamp(0.0, 1.0)
    return torch.where(hit_positive, q_hit * query_evidence_gate, torch.zeros_like(q_hit))


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
    keep_fraction: float = QUERY_LOCAL_UTILITY_REPLACEMENT_KEEP_FRACTION,
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


def _segment_budget_point_value_for_target_mode(
    *,
    target_mode: str,
    final_score: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return the point value used to build the active segment-budget head target."""
    mode = str(target_mode).lower()
    if mode == QUERY_LOCAL_UTILITY_FACTORIZED_TARGET_MODE:
        return final_score, {
            "segment_budget_target_base_source": "query_local_utility_final_score",
            "segment_budget_target_variant": "active_final_score",
            "segment_budget_target_aggregation": "top20_mean",
            "segment_budget_target_experimental": False,
            "final_success_allowed": True,
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
        params = validated_range_query_params(query)
        mask = points_in_range_box(points, params).to(device=device, dtype=torch.bool)
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
        crossing_idx = segment_box_bracket_indices(points_cpu, boundaries, params).to(
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
    family_evidence = _range_query_family_evidence(
        points=points,
        boundaries=boundaries,
        range_queries=range_queries,
    )
    target_q_hit = _query_evidence_gate_target(
        q_hit=q_hit,
        ship_query_evidence=ship_query_evidence,
        hit_positive=hit_positive,
        family_evidence=family_evidence,
    )
    target_behavior = _query_segment_local_behavior_signal(
        behavior=behavior,
        q_hit=target_q_hit,
        valid=hit_positive,
        boundaries=boundaries,
        segment_size=segment_size,
    )
    head_variant_diagnostics: dict[str, Any] = {
        "query_hit_target_variant": "raw_query_hit_ship_evidence_multiplier",
        "query_hit_target_base_source": (
            "raw_query_hit_probability_times_0.65_plus_"
            "0.35_positive_mean_normalized_ship_query_evidence"
        ),
        "query_hit_target_semantics": ("raw_query_hit_scale_preserving_ship_evidence_ranker"),
        "query_hit_target_raw_query_hit_base_weight": (
            QUERY_LOCAL_UTILITY_QUERY_EVIDENCE_BASE_WEIGHT
        ),
        "query_hit_target_ship_query_evidence_multiplier_weight": (
            QUERY_LOCAL_UTILITY_QUERY_EVIDENCE_SHIP_MULTIPLIER_WEIGHT
        ),
        "query_hit_target_family_conditioned": True,
        "conditional_behavior_target_variant": "query_segment_local_behavior_utility",
        "conditional_behavior_target_base_source": (
            "normalized_query_hit_conditioned_trajectory_change_times_"
            "0.45_plus_0.35_segment_behavior_support_plus_"
            "0.20_segment_raw_query_hit_evidence_multiplier_support"
        ),
        "final_label_variant": ("additive_raw_query_hit_behavior_query_local_utility_point_score"),
    }
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
        )
    )
    segment_budget_aggregation = str(
        segment_budget_variant_diagnostics.get("segment_budget_target_aggregation", "top20_mean")
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
    elif segment_budget_aggregation == "top20_mean":
        segment_budget = _segment_pooled_targets(
            segment_budget_point_value,
            boundaries,
            segment_size,
            pool="top20_mean",
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
            "replacement_representative_keep_fraction": (
                QUERY_LOCAL_UTILITY_REPLACEMENT_KEEP_FRACTION
            ),
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
            "conditional_behavior_replacement_partial_alignment": (
                _conditional_behavior_replacement_partial_alignment(
                    behavior=target_behavior,
                    replacement=replacement,
                    valid=hit_positive,
                    references={
                        "final_score": final_score,
                        "query_hit_probability": target_q_hit,
                        "ship_query_evidence": ship_query_evidence,
                        "segment_budget_target": segment_budget,
                        "path_length_support_target": path_length_support,
                    },
                )
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
