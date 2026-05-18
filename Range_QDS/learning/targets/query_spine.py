"""Train-query temporal-spine scalar target builders."""

from __future__ import annotations

import math

import torch

from learning.targets.common import (
    _apply_temporal_target_blend,
    _retained_frequency_from_scores,
    _target_budget_ratios,
    _target_budget_weights,
)
from selection.retained_mask_selectors import (
    deterministic_topk_with_jitter,
    evenly_spaced_indices,
)
from workloads.query_types import QUERY_TYPE_ID_RANGE
from workloads.range_geometry import points_in_range_box


def _range_query_spine_scores(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, object]],
    model_config: object,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return query-derived temporal-spine source scores for training only.

    By default, each ``(query, trajectory-hit)`` group contributes equal mass.
    With ``range_query_spine_mass_mode="query"``, each train query contributes
    unit mass split equally across its hit trajectories. In both cases, each
    group's mass is spread over a small set of evenly spaced in-query anchors,
    with high-turn points added as extra shape anchors. This target is query
    aware only during supervision construction; the trained model still receives
    point features only at compression time.
    """
    source_scores = torch.zeros((points.shape[0],), dtype=torch.float32, device=points.device)
    spine_fraction = max(
        0.0, min(1.0, float(getattr(model_config, "range_query_spine_fraction", 0.10) or 0.0))
    )
    if spine_fraction <= 0.0:
        raise ValueError("range_query_spine_fraction must be positive for query_spine_frequency.")
    mass_mode = str(getattr(model_config, "range_query_spine_mass_mode", "hit_group")).lower()
    if mass_mode not in {"hit_group", "query"}:
        raise ValueError("range_query_spine_mass_mode must be 'hit_group' or 'query'.")
    turn_fraction = 0.25
    range_query_count = 0
    hit_group_count = 0
    selected_anchor_count = 0
    selected_turn_anchor_count = 0
    query_with_hits_count = 0
    max_query_hit_group_count = 0

    for query_index, query in enumerate(typed_queries):
        if str(query.get("type", "")).lower() != "range":
            continue
        range_query_count += 1
        params = query.get("params")
        if not isinstance(params, dict):
            continue
        box_support = points_in_range_box(points, params)
        query_groups: list[torch.Tensor] = []
        for trajectory_id, (start, end) in enumerate(boundaries):
            if end <= start:
                continue
            local_offsets = torch.where(box_support[start:end])[0]
            local_count = int(local_offsets.numel())
            if local_count <= 0:
                continue
            hit_group_count += 1
            spine_count = min(local_count, max(1, math.ceil(spine_fraction * local_count)))
            if local_count >= 2:
                spine_count = max(2, spine_count)
            local_spine_offsets = evenly_spaced_indices(local_count, spine_count, points.device)
            selected_offsets = local_offsets[local_spine_offsets]

            if points.shape[1] > 7 and local_count >= 3:
                turn_scores = points[start + local_offsets, 7].float().clamp(min=0.0)
                turn_count = min(local_count, max(1, math.ceil(turn_fraction * spine_count)))
                if bool((turn_scores > 0.0).any().item()):
                    turn_local = deterministic_topk_with_jitter(
                        turn_scores,
                        keep_count=turn_count,
                        trajectory_id=(query_index + 1) * 10007 + trajectory_id,
                    )
                    turn_offsets = local_offsets[turn_local]
                    selected_offsets = torch.unique(
                        torch.cat([selected_offsets, turn_offsets]), sorted=True
                    )
                    selected_turn_anchor_count += int(turn_offsets.numel())

            global_indices = start + selected_offsets
            if global_indices.numel() == 0:
                continue
            query_groups.append(global_indices)
            selected_anchor_count += int(global_indices.numel())

        query_hit_group_count = len(query_groups)
        if query_hit_group_count > 0:
            query_with_hits_count += 1
            max_query_hit_group_count = max(max_query_hit_group_count, query_hit_group_count)
            for global_indices in query_groups:
                group_mass = 1.0
                if mass_mode == "query":
                    group_mass = 1.0 / float(query_hit_group_count)
                source_scores[global_indices] += float(group_mass) / float(global_indices.numel())

    if range_query_count > 0:
        source_scores = source_scores / float(range_query_count)
    positive = source_scores > 0.0
    diagnostics = {
        "query_spine_fraction": float(spine_fraction),
        "query_spine_mass_mode": mass_mode,
        "query_spine_range_query_count": int(range_query_count),
        "query_spine_query_with_hits_count": int(query_with_hits_count),
        "query_spine_hit_group_count": int(hit_group_count),
        "query_spine_hit_groups_per_hit_query_mean": float(
            hit_group_count / max(1, query_with_hits_count)
        ),
        "query_spine_hit_groups_per_hit_query_max": int(max_query_hit_group_count),
        "query_spine_selected_anchor_count": int(selected_anchor_count),
        "query_spine_selected_turn_anchor_count": int(selected_turn_anchor_count),
        "query_spine_source_positive_count": int(positive.sum().item()),
        "query_spine_source_positive_mass": (
            float(source_scores[positive].sum().item()) if bool(positive.any().item()) else 0.0
        ),
    }
    return source_scores, diagnostics


def range_query_spine_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, object]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Convert per-query temporal-spine source scores into retained frequency."""
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")
    if int(points.shape[0]) != int(labels.shape[0]):
        raise ValueError("points and labels must have the same point count.")

    ratios = _target_budget_ratios(model_config)
    source_scores, spine_diagnostics = _range_query_spine_scores(
        points=points,
        boundaries=boundaries,
        typed_queries=typed_queries,
        model_config=model_config,
    )
    if not bool((source_scores > 0.0).any().item()):
        raise ValueError("query_spine_frequency target found no positive source scores.")
    retained_frequency, used = _retained_frequency_from_scores(
        source_scores=source_scores,
        boundaries=boundaries,
        model_config=model_config,
        ratios=ratios,
    )
    if used <= 0:
        raise ValueError("query_spine_frequency target did not use any budget ratios.")
    retained_frequency, temporal_blend_diagnostics = _apply_temporal_target_blend(
        retained_frequency=retained_frequency,
        boundaries=boundaries,
        model_config=model_config,
        ratios=ratios,
    )

    transformed = labels.clone()
    transformed[:, type_idx] = retained_frequency.clamp(0.0, 1.0)
    transformed_mask = labelled_mask.clone()
    transformed_mask[:, type_idx] = True
    positive = transformed[:, type_idx] > 0.0
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "mode": "query_spine_frequency",
        "source": "range_query_temporal_spines",
        "budget_loss_ratios": list(ratios),
        "budget_weights": list(_target_budget_weights(model_config, ratios)),
        "budget_weight_power": float(
            getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0
        ),
        "labelled_point_count": int(transformed.shape[0]),
        "positive_label_count": positive_count,
        "positive_label_fraction": float(positive_count / max(1, int(transformed.shape[0]))),
        "positive_label_mass": float(transformed[positive, type_idx].sum().item())
        if positive_count > 0
        else 0.0,
        "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
        "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
    }
    diagnostics.update(spine_diagnostics)
    diagnostics.update(temporal_blend_diagnostics)
    return transformed, transformed_mask, diagnostics
