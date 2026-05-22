"""Aggregation builders for scalar range-training target families."""

from __future__ import annotations

import torch

from learning.targets.common import (
    _target_budget_ratios,
    _target_budget_weights,
    aggregate_range_label_sets,
)
from learning.targets.marginal_coverage import range_marginal_coverage_training_labels
from learning.targets.retained_frequency import (
    range_global_budget_retained_frequency_training_labels,
    range_retained_frequency_training_labels,
)
from learning.targets.structural import range_structural_retained_frequency_training_labels
from workloads.query_types import QUERY_TYPE_ID_RANGE


def aggregate_range_structural_retained_frequency_training_labels(
    label_sets: list[tuple[torch.Tensor, torch.Tensor]],
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Average structural-retained targets over independent train workloads."""
    transformed_sets: list[tuple[torch.Tensor, torch.Tensor]] = []
    per_replicate: list[dict[str, object]] = []
    for labels, labelled_mask in label_sets:
        transformed, transformed_mask, diagnostics = (
            range_structural_retained_frequency_training_labels(
                labels=labels,
                labelled_mask=labelled_mask,
                points=points,
                boundaries=boundaries,
                model_config=model_config,
                type_idx=type_idx,
            )
        )
        transformed_sets.append((transformed, transformed_mask))
        per_replicate.append(diagnostics)

    aggregated, aggregated_mask, diagnostics = aggregate_range_label_sets(
        transformed_sets,
        type_idx=type_idx,
        source="range_structural_retained_frequency_training_label_replicates",
    )
    diagnostics.update(
        {
            "mode": "structural_retained_frequency",
            "budget_loss_ratios": list(_target_budget_ratios(model_config)),
            "budget_weights": list(
                _target_budget_weights(model_config, _target_budget_ratios(model_config))
            ),
            "budget_weight_power": float(
                getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0
            ),
            "structural_target_blend": float(
                max(
                    0.0,
                    min(
                        1.0,
                        float(getattr(model_config, "range_structural_target_blend", 0.25) or 0.0),
                    ),
                )
            ),
            "structural_target_source_mode": str(
                getattr(model_config, "range_structural_target_source_mode", "blend")
            ).lower(),
            "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
            "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
            "per_replicate": per_replicate,
        }
    )
    return aggregated, aggregated_mask, diagnostics


def aggregate_range_retained_frequency_training_labels(
    label_sets: list[tuple[torch.Tensor, torch.Tensor]],
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Average retained-frequency targets over independent train workloads."""
    transformed_sets: list[tuple[torch.Tensor, torch.Tensor]] = []
    per_replicate: list[dict[str, object]] = []
    for labels, labelled_mask in label_sets:
        transformed, transformed_mask, diagnostics = range_retained_frequency_training_labels(
            labels=labels,
            labelled_mask=labelled_mask,
            boundaries=boundaries,
            model_config=model_config,
            type_idx=type_idx,
        )
        transformed_sets.append((transformed, transformed_mask))
        per_replicate.append(diagnostics)

    aggregated, aggregated_mask, diagnostics = aggregate_range_label_sets(
        transformed_sets,
        type_idx=type_idx,
        source="range_retained_frequency_training_label_replicates",
    )
    diagnostics.update(
        {
            "mode": "retained_frequency",
            "budget_loss_ratios": list(_target_budget_ratios(model_config)),
            "budget_weights": list(
                _target_budget_weights(model_config, _target_budget_ratios(model_config))
            ),
            "budget_weight_power": float(
                getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0
            ),
            "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
            "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
            "per_replicate": per_replicate,
        }
    )
    return aggregated, aggregated_mask, diagnostics


def aggregate_range_global_budget_retained_frequency_training_labels(
    label_sets: list[tuple[torch.Tensor, torch.Tensor]],
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Average global-budget retained targets over independent train workloads."""
    transformed_sets: list[tuple[torch.Tensor, torch.Tensor]] = []
    per_replicate: list[dict[str, object]] = []
    for labels, labelled_mask in label_sets:
        transformed, transformed_mask, diagnostics = (
            range_global_budget_retained_frequency_training_labels(
                labels=labels,
                labelled_mask=labelled_mask,
                boundaries=boundaries,
                model_config=model_config,
                type_idx=type_idx,
            )
        )
        transformed_sets.append((transformed, transformed_mask))
        per_replicate.append(diagnostics)

    aggregated, aggregated_mask, diagnostics = aggregate_range_label_sets(
        transformed_sets,
        type_idx=type_idx,
        source="range_global_budget_retained_frequency_training_label_replicates",
    )
    diagnostics.update(
        {
            "mode": "global_budget_retained_frequency",
            "budget_loss_ratios": list(_target_budget_ratios(model_config)),
            "budget_weights": list(
                _target_budget_weights(model_config, _target_budget_ratios(model_config))
            ),
            "budget_weight_power": float(
                getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0
            ),
            "global_budget_min_points_per_trajectory": 2,
            "per_replicate": per_replicate,
        }
    )
    return aggregated, aggregated_mask, diagnostics


def aggregate_range_marginal_coverage_training_labels(
    label_sets: list[tuple[torch.Tensor, torch.Tensor]],
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Average marginal-coverage retained targets over independent train workloads."""
    transformed_sets: list[tuple[torch.Tensor, torch.Tensor]] = []
    per_replicate: list[dict[str, object]] = []
    for labels, labelled_mask in label_sets:
        transformed, transformed_mask, diagnostics = range_marginal_coverage_training_labels(
            labels=labels,
            labelled_mask=labelled_mask,
            boundaries=boundaries,
            model_config=model_config,
            type_idx=type_idx,
        )
        transformed_sets.append((transformed, transformed_mask))
        per_replicate.append(diagnostics)

    aggregated, aggregated_mask, diagnostics = aggregate_range_label_sets(
        transformed_sets,
        type_idx=type_idx,
        source="range_marginal_coverage_training_label_replicates",
    )
    diagnostics.update(
        {
            "mode": "marginal_coverage_frequency",
            "budget_loss_ratios": list(_target_budget_ratios(model_config)),
            "budget_weights": list(
                _target_budget_weights(model_config, _target_budget_ratios(model_config))
            ),
            "budget_weight_power": float(
                getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0
            ),
            "marginal_target_radius_scale": float(
                getattr(model_config, "range_marginal_target_radius_scale", 0.50) or 0.0
            ),
            "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
            "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
            "per_replicate": per_replicate,
        }
    )
    return aggregated, aggregated_mask, diagnostics
