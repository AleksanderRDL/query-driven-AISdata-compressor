"""Neighborhood-marginal scalar range target builders."""

from __future__ import annotations

import math

import torch

from selection.retained_mask_selectors import (
    deterministic_topk_with_jitter,
    evenly_spaced_indices,
)
from training.targets.common import (
    _apply_temporal_target_blend,
    _target_budget_ratios,
    _target_budget_weights,
)
from workloads.query_types import QUERY_TYPE_ID_RANGE


def _local_window_sum(values: torch.Tensor, radius: int) -> torch.Tensor:
    """Return inclusive fixed-radius window sums for one trajectory vector."""
    count = int(values.numel())
    if count <= 0:
        return values.new_empty((0,), dtype=torch.float32)
    if int(radius) <= 0:
        return values.float()
    positions = torch.arange(count, device=values.device)
    left = torch.clamp(positions - int(radius), min=0)
    right = torch.clamp(positions + int(radius) + 1, max=count)
    prefix = torch.cat([values.new_zeros((1,), dtype=torch.float32), values.float().cumsum(dim=0)])
    return prefix[right] - prefix[left]


def _erase_local_window(values: torch.Tensor, index: int, radius: int) -> None:
    """Mark one selected point's local label neighborhood as covered."""
    count = int(values.numel())
    if count <= 0:
        return
    left = max(0, int(index) - max(0, int(radius)))
    right = min(count, int(index) + max(0, int(radius)) + 1)
    values[left:right] = 0.0


def _marginal_coverage_mask_from_scores(
    source_scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    ratio: float,
    model_config: object,
) -> torch.Tensor:
    """Greedily retain points that cover remaining local label mass.

    Unlike plain top-k retained-frequency labels, selecting a point consumes
    nearby label mass before the next selection. This gives broad, redundant
    workloads a set-aware target without passing queries into the model.
    """
    retained = torch.zeros_like(source_scores, dtype=torch.bool)
    radius_scale = max(
        0.0, float(getattr(model_config, "range_marginal_target_radius_scale", 0.50) or 0.0)
    )
    base_fraction = min(1.0, max(0.0, float(getattr(model_config, "mlqds_temporal_fraction", 0.0))))
    for trajectory_id, (start, end) in enumerate(boundaries):
        point_count = int(end - start)
        if point_count <= 0:
            continue
        total_keep_count = min(point_count, max(2, math.ceil(float(ratio) * point_count)))
        local_scores = source_scores[start:end].float().clamp(min=0.0)
        selected = torch.zeros((point_count,), dtype=torch.bool, device=source_scores.device)
        base_keep_count = 0
        if base_fraction > 0.0:
            base_keep_count = min(
                total_keep_count, max(2, math.ceil(total_keep_count * base_fraction))
            )
        base_indices = evenly_spaced_indices(point_count, base_keep_count, source_scores.device)
        if base_indices.numel() > 0:
            selected[base_indices] = True

        expected_spacing = float(point_count) / float(max(1, total_keep_count))
        radius = max(0, math.ceil(radius_scale * expected_spacing))
        uncovered = local_scores.clone()
        for index in base_indices.detach().cpu().tolist():
            _erase_local_window(uncovered, int(index), radius)

        remaining = total_keep_count - int(selected.sum().item())
        positions = torch.arange(point_count, dtype=torch.float32, device=source_scores.device)
        for step in range(max(0, remaining)):
            available = ~selected
            if not bool(available.any().item()):
                break
            gains = _local_window_sum(uncovered, radius)
            if not bool((gains[available] > 1e-12).any().item()):
                if bool(selected.any().item()):
                    selected_positions = torch.where(selected)[0].float()
                    gains = (
                        torch.abs(positions.unsqueeze(1) - selected_positions.unsqueeze(0))
                        .min(dim=1)
                        .values
                    )
                else:
                    gains = torch.ones_like(gains)
            gains = gains + 1e-3 * local_scores
            gains = gains.masked_fill(~available, float("-inf"))
            next_idx = deterministic_topk_with_jitter(
                gains,
                keep_count=1,
                trajectory_id=trajectory_id * 1009 + step,
            )
            if next_idx.numel() == 0 or not torch.isfinite(gains[next_idx[0]]):
                break
            idx = int(next_idx[0].item())
            selected[idx] = True
            _erase_local_window(uncovered, idx, radius)

        retained[start:end] = selected
    return retained


def range_marginal_coverage_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Convert range labels into neighborhood-marginal retained frequency."""
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")

    ratios = _target_budget_ratios(model_config)
    budget_weights = _target_budget_weights(model_config, ratios)
    source_scores = labels[:, type_idx].float().clamp(min=0.0)
    retained_frequency = torch.zeros_like(source_scores)
    used = 0
    used_weight = 0.0
    for ratio, budget_weight in zip(ratios, budget_weights, strict=False):
        mask = _marginal_coverage_mask_from_scores(
            source_scores=source_scores,
            boundaries=boundaries,
            ratio=float(ratio),
            model_config=model_config,
        )
        retained_frequency += float(budget_weight) * mask.to(dtype=retained_frequency.dtype)
        used += 1
        used_weight += float(budget_weight)
    if used_weight > 1e-12:
        retained_frequency = retained_frequency / float(used_weight)
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
        "mode": "marginal_coverage_frequency",
        "source": "range_training_labels",
        "budget_loss_ratios": list(ratios),
        "budget_weights": list(budget_weights),
        "budget_weight_power": float(
            getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0
        ),
        "labelled_point_count": int(transformed.shape[0]),
        "positive_label_count": positive_count,
        "positive_label_fraction": float(positive_count / max(1, int(transformed.shape[0]))),
        "positive_label_mass": float(transformed[positive, type_idx].sum().item())
        if positive_count > 0
        else 0.0,
        "marginal_target_radius_scale": float(
            getattr(model_config, "range_marginal_target_radius_scale", 0.50) or 0.0
        ),
        "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
        "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
    }
    diagnostics.update(temporal_blend_diagnostics)
    return transformed, transformed_mask, diagnostics
