"""Query-free structural scalar range target builders."""

from __future__ import annotations

import torch

from queries.query_types import QUERY_TYPE_ID_RANGE
from training.targets.common import (
    _apply_temporal_target_blend,
    _retained_frequency_from_scores,
    _scale01,
    _target_budget_ratios,
    _target_budget_weights,
)
from training.training_losses import _safe_quantile

RANGE_STRUCTURAL_TARGET_WEIGHTS = {
    "uniqueness": 0.40,
    "turn": 0.20,
    "gap": 0.15,
    "globality": 0.15,
    "endpoint": 0.10,
}


def _query_free_structural_scores(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return MLSimp-inspired query-free structural point scores.

    The score is deliberately train/eval-query blind: it uses only the observed
    trajectory geometry and dynamics. It is intended as a training regularizer
    for retained-frequency labels, not as an inference-time geometry blend.
    """
    if points.ndim != 2 or int(points.shape[1]) < 3:
        raise ValueError("points must have shape [n_points, point_dim>=3].")

    structural = torch.zeros((int(points.shape[0]),), dtype=torch.float32, device=points.device)
    component_mass = {
        "uniqueness": 0.0,
        "turn": 0.0,
        "gap": 0.0,
        "globality": 0.0,
        "endpoint": 0.0,
    }
    scored_trajectories = 0
    for start, end in boundaries:
        start_i = int(start)
        end_i = int(end)
        count = end_i - start_i
        if count <= 0:
            continue
        scored_trajectories += 1
        local = points[start_i:end_i].float()
        if count == 1:
            structural[start_i:end_i] = 1.0
            component_mass["endpoint"] += 1.0
            continue

        coords = local[:, 1:3]
        coord_min = coords.min(dim=0).values
        coord_span = (coords.max(dim=0).values - coord_min).clamp(min=1e-6)
        coords_norm = (coords - coord_min) / coord_span

        uniqueness = torch.zeros((count,), dtype=torch.float32, device=points.device)
        if count >= 3:
            midpoint = 0.5 * (coords_norm[:-2] + coords_norm[2:])
            uniqueness[1:-1] = torch.linalg.vector_norm(coords_norm[1:-1] - midpoint, dim=1)
            endpoint_value = (
                float(uniqueness[1:-1].max().item())
                if bool((uniqueness[1:-1] > 0).any().item())
                else 1.0
            )
        else:
            endpoint_value = 1.0
        uniqueness[0] = endpoint_value
        uniqueness[-1] = endpoint_value
        uniqueness = _scale01(uniqueness)

        turn = (
            _scale01(local[:, 7].clamp(min=0.0))
            if int(local.shape[1]) > 7
            else torch.zeros((count,), dtype=torch.float32, device=points.device)
        )

        times = local[:, 0]
        deltas = torch.diff(times).abs() if count >= 2 else times.new_empty((0,))
        prev_gap = torch.cat([times.new_zeros((1,)), deltas])
        next_gap = torch.cat([deltas, times.new_zeros((1,))])
        gap = _scale01(torch.maximum(prev_gap, next_gap))

        centroid = coords_norm.mean(dim=0)
        dist_to_centroid = torch.linalg.vector_norm(coords_norm - centroid.unsqueeze(0), dim=1)
        globality = 1.0 - _scale01(dist_to_centroid)

        endpoint = torch.zeros((count,), dtype=torch.float32, device=points.device)
        endpoint[0] = 1.0
        endpoint[-1] = 1.0
        if int(local.shape[1]) > 6:
            endpoint = torch.maximum(endpoint, local[:, 5].clamp(0.0, 1.0))
            endpoint = torch.maximum(endpoint, local[:, 6].clamp(0.0, 1.0))

        local_structural = (
            RANGE_STRUCTURAL_TARGET_WEIGHTS["uniqueness"] * uniqueness
            + RANGE_STRUCTURAL_TARGET_WEIGHTS["turn"] * turn
            + RANGE_STRUCTURAL_TARGET_WEIGHTS["gap"] * gap
            + RANGE_STRUCTURAL_TARGET_WEIGHTS["globality"] * globality
            + RANGE_STRUCTURAL_TARGET_WEIGHTS["endpoint"] * endpoint
        ).clamp(0.0, 1.0)
        structural[start_i:end_i] = local_structural
        component_mass["uniqueness"] += float(uniqueness.sum().item())
        component_mass["turn"] += float(turn.sum().item())
        component_mass["gap"] += float(gap.sum().item())
        component_mass["globality"] += float(globality.sum().item())
        component_mass["endpoint"] += float(endpoint.sum().item())

    positive = structural > 0.0
    diagnostics: dict[str, object] = {
        "structural_score_trajectory_count": int(scored_trajectories),
        "structural_score_positive_count": int(positive.sum().item()),
        "structural_score_positive_fraction": float(
            int(positive.sum().item()) / max(1, int(structural.numel()))
        ),
        "structural_score_positive_mass": (
            float(structural[positive].sum().item()) if bool(positive.any().item()) else 0.0
        ),
        "structural_score_component_weights": dict(RANGE_STRUCTURAL_TARGET_WEIGHTS),
        "structural_score_component_mass": component_mass,
    }
    if bool(positive.any().item()):
        diagnostics.update(
            {
                "structural_score_p50": float(_safe_quantile(structural[positive], 0.50).item()),
                "structural_score_p95": float(_safe_quantile(structural[positive], 0.95).item()),
            }
        )
    return structural, diagnostics


def range_structural_retained_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Blend train workload usefulness with query-free structural scores.

    This is a training-only target transform. The deployed scorer still sees
    only query-free point features and final eval masks are frozen before
    held-out range queries are scored.
    """
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")
    if int(points.shape[0]) != int(labels.shape[0]):
        raise ValueError("points and labels must have the same point count.")

    ratios = _target_budget_ratios(model_config)
    blend = max(
        0.0, min(1.0, float(getattr(model_config, "range_structural_target_blend", 0.25) or 0.0))
    )
    label_scores = labels[:, type_idx].float().clamp(0.0, 1.0)
    structural_scores, structural_diagnostics = _query_free_structural_scores(points, boundaries)
    source_mode = str(getattr(model_config, "range_structural_target_source_mode", "blend")).lower()
    if source_mode not in {"blend", "boost"}:
        raise ValueError("range_structural_target_source_mode must be 'blend' or 'boost'.")
    if source_mode == "boost":
        source_scores = label_scores * (1.0 + blend * structural_scores)
    else:
        source_scores = ((1.0 - blend) * label_scores + blend * structural_scores).clamp(0.0, 1.0)

    base_frequency, base_used = _retained_frequency_from_scores(
        source_scores=label_scores,
        boundaries=boundaries,
        model_config=model_config,
        ratios=ratios,
    )
    retained_frequency, used = _retained_frequency_from_scores(
        source_scores=source_scores,
        boundaries=boundaries,
        model_config=model_config,
        ratios=ratios,
    )
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
    source_positive = source_scores > 0.0
    base_positive = base_frequency > 0.0
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "mode": "structural_retained_frequency",
        "source": "range_training_labels_plus_query_free_structural_scores",
        "budget_loss_ratios": list(ratios),
        "budget_weights": list(_target_budget_weights(model_config, ratios)),
        "budget_weight_power": float(
            getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0
        ),
        "structural_target_blend": float(blend),
        "structural_target_source_mode": source_mode,
        "labelled_point_count": int(transformed.shape[0]),
        "positive_label_count": positive_count,
        "positive_label_fraction": float(positive_count / max(1, int(transformed.shape[0]))),
        "positive_label_mass": float(transformed[positive, type_idx].sum().item())
        if positive_count > 0
        else 0.0,
        "source_positive_label_count": int(source_positive.sum().item()),
        "source_positive_label_mass": (
            float(source_scores[source_positive].sum().item())
            if bool(source_positive.any().item())
            else 0.0
        ),
        "base_retained_frequency_budget_count": int(base_used),
        "base_retained_frequency_positive_label_count": int(base_positive.sum().item()),
        "base_retained_frequency_positive_label_mass": (
            float(base_frequency[base_positive].sum().item())
            if bool(base_positive.any().item())
            else 0.0
        ),
        "teacher_retained_frequency_budget_count": int(used),
        "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
        "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
    }
    diagnostics.update(structural_diagnostics)
    diagnostics.update(temporal_blend_diagnostics)
    return transformed, transformed_mask, diagnostics
