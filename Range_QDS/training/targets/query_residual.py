"""Train-query residual-anchor scalar target builders."""

from __future__ import annotations

import math

import torch

from queries.query_types import QUERY_TYPE_ID_RANGE
from queries.range_geometry import points_in_range_box
from simplification.simplify_trajectories import (
    deterministic_topk_with_jitter,
    evenly_spaced_indices,
)
from training.targets.common import (
    _target_budget_ratios,
    _target_budget_weights,
    _temporal_base_mask_for_ratio,
)


def _query_residual_priority_positions(
    *,
    points: torch.Tensor,
    global_indices: torch.Tensor,
    local_base_mask: torch.Tensor,
    query_keep_count: int,
) -> list[int]:
    """Return in-query local positions preferred for learned residual fill."""
    count = int(global_indices.numel())
    if count <= 0:
        return []

    selected: list[int] = []
    selected_set: set[int] = set()

    def add_positions(positions: list[int] | torch.Tensor) -> None:
        raw_positions = (
            positions.detach().cpu().tolist() if isinstance(positions, torch.Tensor) else positions
        )
        for value in raw_positions:
            pos = int(value)
            if pos < 0 or pos >= count or pos in selected_set:
                continue
            if bool(local_base_mask[pos].item()):
                continue
            selected_set.add(pos)
            selected.append(pos)

    # Boundary evidence is disproportionately useful for entry/exit and temporal span.
    add_positions([0, count - 1])

    base_positions = torch.where(local_base_mask)[0]
    if base_positions.numel() > 0:
        anchors = torch.cat(
            [
                torch.tensor([-1], dtype=torch.long, device=global_indices.device),
                base_positions.to(dtype=torch.long),
                torch.tensor([count], dtype=torch.long, device=global_indices.device),
            ]
        )
        gap_left = anchors[:-1]
        gap_right = anchors[1:]
        missing = gap_right - gap_left - 1
        ordered_gaps = torch.argsort(missing, descending=True)
        mids: list[int] = []
        for gap_idx in ordered_gaps.detach().cpu().tolist():
            if int(missing[int(gap_idx)].item()) <= 0:
                continue
            left = int(gap_left[int(gap_idx)].item())
            right = int(gap_right[int(gap_idx)].item())
            mids.append((left + right) // 2)
        add_positions(mids)
    else:
        add_positions([count // 2])

    if points.shape[1] > 7 and count >= 3:
        turn_count = min(count, max(1, math.ceil(0.25 * float(max(1, query_keep_count)))))
        turn_scores = points[global_indices, 7].float().clamp(min=0.0)
        if bool((turn_scores > 0.0).any().item()):
            turn_positions = deterministic_topk_with_jitter(
                turn_scores,
                keep_count=turn_count,
                trajectory_id=count * 7919 + query_keep_count,
            )
            add_positions(turn_positions)

    spaced_count = min(count, max(1, int(query_keep_count)))
    add_positions(evenly_spaced_indices(count, spaced_count, global_indices.device))
    add_positions(torch.arange(count, device=global_indices.device))
    return selected


def _range_query_residual_scores(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, object]],
    model_config: object,
    ratios: tuple[float, ...],
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return budgeted train-query residual-anchor frequencies.

    For each train range query and budget, this simulates the query-blind
    temporal base, then labels only the residual anchors needed to improve
    range continuity, boundary context, turns, and shape inside the query.
    """
    n_points = int(points.shape[0])
    target = torch.zeros((n_points,), dtype=torch.float32, device=points.device)
    temporal_fraction = float(getattr(model_config, "mlqds_temporal_fraction", 0.0))
    multiplier = max(
        0.0, float(getattr(model_config, "range_query_residual_multiplier", 1.0) or 0.0)
    )
    if multiplier <= 0.0:
        raise ValueError(
            "range_query_residual_multiplier must be positive for query_residual_frequency."
        )
    mass_mode = str(getattr(model_config, "range_query_residual_mass_mode", "query")).lower()
    if mass_mode not in {"query", "point"}:
        raise ValueError("range_query_residual_mass_mode must be 'query' or 'point'.")

    range_query_count = 0
    used_budget_count = 0
    total_hit_group_count = 0
    total_selected_anchor_count = 0
    total_selected_residual_count = 0
    total_base_anchor_count = 0
    per_budget: list[dict[str, object]] = []
    budget_weights = _target_budget_weights(model_config, ratios)
    used_weight = 0.0

    range_queries = [
        query
        for query in typed_queries
        if str(query.get("type", "")).lower() == "range" and isinstance(query.get("params"), dict)
    ]
    range_query_count = len(range_queries)
    if range_query_count <= 0:
        return target, {
            "query_residual_range_query_count": 0,
            "query_residual_used_budget_count": 0,
            "query_residual_multiplier": float(multiplier),
            "query_residual_mass_mode": mass_mode,
        }

    for ratio, budget_weight in zip(ratios, budget_weights, strict=False):
        ratio_value = min(1.0, max(0.0, float(ratio)))
        if ratio_value <= 0.0:
            continue
        used_budget_count += 1
        budget_scores = torch.zeros_like(target)
        base_mask = _temporal_base_mask_for_ratio(
            n_points=n_points,
            boundaries=boundaries,
            ratio=ratio_value,
            temporal_fraction=temporal_fraction,
            device=points.device,
        )
        base_anchor_count = int(base_mask.sum().item())
        budget_hit_group_count = 0
        budget_selected_anchor_count = 0
        budget_selected_residual_count = 0

        for query in range_queries:
            params = query["params"]
            if not isinstance(params, dict):
                raise ValueError("Range query params must be a dictionary.")
            box_support = points_in_range_box(points, params)
            query_selected: list[torch.Tensor] = []
            for start, end in boundaries:
                if end <= start:
                    continue
                in_offsets = torch.where(box_support[start:end])[0]
                count = int(in_offsets.numel())
                if count <= 0:
                    continue
                budget_hit_group_count += 1
                local_global = start + in_offsets
                local_base = base_mask[local_global]
                query_keep_count = min(
                    count,
                    max(1, math.ceil(float(multiplier) * ratio_value * float(count))),
                )
                if count >= 2:
                    query_keep_count = max(2, query_keep_count)
                residual_needed = max(0, query_keep_count - int(local_base.sum().item()))
                if residual_needed <= 0:
                    continue
                priority_positions = _query_residual_priority_positions(
                    points=points,
                    global_indices=local_global,
                    local_base_mask=local_base,
                    query_keep_count=query_keep_count,
                )
                selected_positions = priority_positions[:residual_needed]
                if not selected_positions:
                    continue
                selected_global = local_global[
                    torch.tensor(selected_positions, dtype=torch.long, device=points.device)
                ]
                query_selected.append(selected_global)
                budget_selected_anchor_count += int(query_keep_count)
                budget_selected_residual_count += int(selected_global.numel())

            if query_selected:
                selected = torch.unique(torch.cat(query_selected), sorted=False)
                if selected.numel() > 0:
                    mass = 1.0 / float(selected.numel()) if mass_mode == "query" else 1.0
                    budget_scores[selected] += float(mass)

        budget_scores = budget_scores / float(range_query_count)
        target += float(budget_weight) * budget_scores
        used_weight += float(budget_weight)
        total_hit_group_count += budget_hit_group_count
        total_selected_anchor_count += budget_selected_anchor_count
        total_selected_residual_count += budget_selected_residual_count
        total_base_anchor_count += base_anchor_count
        positive = budget_scores > 0.0
        per_budget.append(
            {
                "budget_ratio": float(ratio_value),
                "budget_weight": float(budget_weight),
                "temporal_base_point_count": int(base_anchor_count),
                "hit_group_count": int(budget_hit_group_count),
                "selected_anchor_count": int(budget_selected_anchor_count),
                "selected_residual_count": int(budget_selected_residual_count),
                "positive_label_count": int(positive.sum().item()),
                "positive_label_mass": (
                    float(budget_scores[positive].sum().item())
                    if bool(positive.any().item())
                    else 0.0
                ),
            }
        )

    if used_weight > 1e-12:
        target = target / float(used_weight)
    positive = target > 0.0
    diagnostics = {
        "query_residual_range_query_count": int(range_query_count),
        "query_residual_used_budget_count": int(used_budget_count),
        "query_residual_budget_weights": list(budget_weights),
        "query_residual_budget_weight_power": float(
            getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0
        ),
        "query_residual_multiplier": float(multiplier),
        "query_residual_mass_mode": mass_mode,
        "query_residual_hit_group_count": int(total_hit_group_count),
        "query_residual_temporal_base_point_count": int(total_base_anchor_count),
        "query_residual_selected_anchor_count": int(total_selected_anchor_count),
        "query_residual_selected_residual_count": int(total_selected_residual_count),
        "query_residual_source_positive_count": int(positive.sum().item()),
        "query_residual_source_positive_mass": (
            float(target[positive].sum().item()) if bool(positive.any().item()) else 0.0
        ),
        "query_residual_per_budget": per_budget,
    }
    return target, diagnostics


def range_query_residual_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, object]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build train-query residual-anchor frequency labels for a blind student."""
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")
    if int(points.shape[0]) != int(labels.shape[0]):
        raise ValueError("points and labels must have the same point count.")

    ratios = _target_budget_ratios(model_config)
    target, residual_diagnostics = _range_query_residual_scores(
        points=points,
        boundaries=boundaries,
        typed_queries=typed_queries,
        model_config=model_config,
        ratios=ratios,
    )
    if not bool((target > 0.0).any().item()):
        raise ValueError("query_residual_frequency target found no positive source scores.")

    transformed = labels.clone()
    transformed[:, type_idx] = target.clamp(0.0, 1.0)
    transformed_mask = labelled_mask.clone()
    transformed_mask[:, type_idx] = True
    positive = transformed[:, type_idx] > 0.0
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "mode": "query_residual_frequency",
        "source": "range_query_residual_anchors",
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
    diagnostics.update(residual_diagnostics)
    return transformed, transformed_mask, diagnostics
