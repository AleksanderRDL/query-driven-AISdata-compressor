"""Aggregation builders for scalar range-training target families."""

from __future__ import annotations

import torch

from learning.targets.common import (
    _apply_temporal_target_blend,
    _retained_frequency_from_scores,
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
from scoring.range_usefulness import RANGE_USEFULNESS_WEIGHTS
from workloads.query_types import QUERY_TYPE_ID_RANGE

RANGE_CONTINUITY_TARGET_WEIGHTS = {
    "range_entry_exit_f1": 0.22,
    "range_crossing_f1": 0.16,
    "range_temporal_coverage": 0.22,
    "range_gap_coverage": 0.22,
    "range_turn_coverage": 0.08,
    "range_shape_score": 0.10,
}


def range_component_retained_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    component_labels: dict[str, torch.Tensor],
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
    component_weights: dict[str, float] | None = None,
    mode: str = "component_retained_frequency",
    source: str = "range_component_training_labels",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build retained-frequency targets independently per RangeUseful component.

    The regular retained-frequency target can be dominated by point-hit and ship
    presence labels. This variant gives temporal, gap, turn, and shape labels a
    direct path into the retained-set target while remaining query-blind at
    inference time.
    """
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")

    ratios = _target_budget_ratios(model_config)
    budget_weights = _target_budget_weights(model_config, ratios)
    weights = component_weights or dict(RANGE_USEFULNESS_WEIGHTS)
    component_target = torch.zeros((labels.shape[0],), dtype=torch.float32, device=labels.device)
    available_weight = 0.0
    per_component: dict[str, dict[str, object]] = {}
    for component_name, component_weight in weights.items():
        component = component_labels.get(component_name)
        if component is None:
            continue
        if component.shape != labels.shape:
            raise ValueError(
                f"component {component_name!r} has shape {tuple(component.shape)}, expected {tuple(labels.shape)}."
            )
        source_scores = component[:, type_idx].float().clamp(min=0.0)
        source_positive = source_scores > 0.0
        source_mass = (
            float(source_scores[source_positive].sum().item())
            if bool(source_positive.any().item())
            else 0.0
        )
        if source_mass <= 1e-12:
            per_component[component_name] = {
                "source_positive_label_count": 0,
                "source_positive_label_mass": 0.0,
                "used": False,
            }
            continue
        retained_frequency, used = _retained_frequency_from_scores(
            source_scores=source_scores,
            boundaries=boundaries,
            model_config=model_config,
            ratios=ratios,
        )
        if used <= 0:
            continue
        weight = float(component_weight)
        component_target += weight * retained_frequency
        available_weight += weight
        target_positive = retained_frequency > 0.0
        per_component[component_name] = {
            "source_positive_label_count": int(source_positive.sum().item()),
            "source_positive_label_mass": source_mass,
            "target_positive_label_count": int(target_positive.sum().item()),
            "target_positive_label_mass": (
                float(retained_frequency[target_positive].sum().item())
                if bool(target_positive.any().item())
                else 0.0
            ),
            "weight": weight,
            "used": True,
        }

    if available_weight <= 1e-12:
        raise ValueError(
            "component_retained_frequency target found no positive component label mass."
        )
    component_target = (component_target / float(available_weight)).clamp(0.0, 1.0)
    component_blend = max(
        0.0, min(1.0, float(getattr(model_config, "range_component_target_blend", 1.0)))
    )
    base_retained_frequency = None
    if component_blend < 1.0:
        base_retained_frequency, _used = _retained_frequency_from_scores(
            source_scores=labels[:, type_idx].float().clamp(min=0.0),
            boundaries=boundaries,
            model_config=model_config,
            ratios=ratios,
        )
        component_target = (
            (1.0 - component_blend) * base_retained_frequency + component_blend * component_target
        ).clamp(0.0, 1.0)
    component_target, temporal_blend_diagnostics = _apply_temporal_target_blend(
        retained_frequency=component_target,
        boundaries=boundaries,
        model_config=model_config,
        ratios=ratios,
    )

    transformed = labels.clone()
    transformed[:, type_idx] = component_target.to(dtype=transformed.dtype)
    transformed_mask = labelled_mask.clone()
    transformed_mask[:, type_idx] = True
    positive = transformed[:, type_idx] > 0.0
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "mode": mode,
        "source": source,
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
        "available_component_weight": float(available_weight),
        "component_target_blend": float(component_blend),
        "base_retained_frequency_positive_label_count": (
            int((base_retained_frequency > 0.0).sum().item())
            if base_retained_frequency is not None
            else None
        ),
        "component_diagnostics": per_component,
        "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
        "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
    }
    diagnostics.update(temporal_blend_diagnostics)
    return transformed, transformed_mask, diagnostics


def range_continuity_retained_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    component_labels: dict[str, torch.Tensor],
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build a retained-frequency target from continuity and boundary components only."""
    return range_component_retained_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        component_labels=component_labels,
        boundaries=boundaries,
        model_config=model_config,
        type_idx=type_idx,
        component_weights=dict(RANGE_CONTINUITY_TARGET_WEIGHTS),
        mode="continuity_retained_frequency",
        source="range_continuity_component_training_labels",
    )


def aggregate_range_component_label_sets(
    label_sets: list[tuple[torch.Tensor, torch.Tensor]],
    component_label_sets: list[dict[str, torch.Tensor] | None],
    type_idx: int = QUERY_TYPE_ID_RANGE,
    aggregation: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, object]]:
    """Aggregate raw range labels and their component-specific label streams."""
    if len(component_label_sets) != len(label_sets):
        raise ValueError("component_label_sets must have the same length as label_sets.")
    if not component_label_sets or any(
        component_labels is None for component_labels in component_label_sets
    ):
        raise ValueError("all component_label_sets entries must be present.")

    aggregated, aggregated_mask, diagnostics = aggregate_range_label_sets(
        label_sets,
        type_idx=type_idx,
        source="range_component_label_replicates",
        aggregation=aggregation,
    )
    aggregated_components: dict[str, torch.Tensor] = {}
    component_diagnostics: dict[str, object] = {}
    for component_name in RANGE_USEFULNESS_WEIGHTS:
        component_sets: list[tuple[torch.Tensor, torch.Tensor]] = []
        for (labels, labelled_mask), component_labels in zip(
            label_sets, component_label_sets, strict=False
        ):
            if component_labels is None:
                raise RuntimeError("all component_label_sets entries must be present.")
            component = component_labels.get(component_name)
            if component is None:
                raise ValueError(f"component_label_sets missing {component_name!r}.")
            if component.shape != labels.shape:
                raise ValueError(
                    f"component {component_name!r} has shape {tuple(component.shape)}, expected {tuple(labels.shape)}."
                )
            component_sets.append((component, labelled_mask))
        component_aggregated, _component_mask, component_diag = aggregate_range_label_sets(
            component_sets,
            type_idx=type_idx,
            source=f"range_component_{component_name}_replicates",
            aggregation=aggregation,
        )
        aggregated_components[component_name] = component_aggregated
        component_diagnostics[component_name] = component_diag

    diagnostics["component_aggregation"] = component_diagnostics
    return aggregated, aggregated_mask, aggregated_components, diagnostics


def aggregate_range_component_retained_frequency_training_labels(
    label_sets: list[tuple[torch.Tensor, torch.Tensor]],
    component_label_sets: list[dict[str, torch.Tensor] | None],
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Average component-retained targets over independent train workloads."""
    if len(component_label_sets) != len(label_sets):
        raise ValueError("component_label_sets must have the same length as label_sets.")
    if not component_label_sets or any(
        component_labels is None for component_labels in component_label_sets
    ):
        raise ValueError("all component_label_sets entries must be present.")

    transformed_sets: list[tuple[torch.Tensor, torch.Tensor]] = []
    per_replicate: list[dict[str, object]] = []
    for (labels, labelled_mask), component_labels in zip(
        label_sets, component_label_sets, strict=False
    ):
        if component_labels is None:
            raise RuntimeError("all component_label_sets entries must be present.")
        transformed, transformed_mask, diagnostics = (
            range_component_retained_frequency_training_labels(
                labels=labels,
                labelled_mask=labelled_mask,
                component_labels=component_labels,
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
        source="range_component_retained_frequency_training_label_replicates",
    )
    diagnostics.update(
        {
            "mode": "component_retained_frequency",
            "budget_loss_ratios": list(_target_budget_ratios(model_config)),
            "budget_weights": list(
                _target_budget_weights(model_config, _target_budget_ratios(model_config))
            ),
            "budget_weight_power": float(
                getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0
            ),
            "component_target_blend": float(
                getattr(model_config, "range_component_target_blend", 1.0)
            ),
            "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
            "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
            "per_replicate": per_replicate,
        }
    )
    return aggregated, aggregated_mask, diagnostics


def aggregate_range_continuity_retained_frequency_training_labels(
    label_sets: list[tuple[torch.Tensor, torch.Tensor]],
    component_label_sets: list[dict[str, torch.Tensor] | None],
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Average continuity-retained targets over independent train workloads."""
    if len(component_label_sets) != len(label_sets):
        raise ValueError("component_label_sets must have the same length as label_sets.")
    if not component_label_sets or any(
        component_labels is None for component_labels in component_label_sets
    ):
        raise ValueError("all component_label_sets entries must be present.")

    transformed_sets: list[tuple[torch.Tensor, torch.Tensor]] = []
    per_replicate: list[dict[str, object]] = []
    for (labels, labelled_mask), component_labels in zip(
        label_sets, component_label_sets, strict=False
    ):
        if component_labels is None:
            raise RuntimeError("all component_label_sets entries must be present.")
        transformed, transformed_mask, diagnostics = (
            range_continuity_retained_frequency_training_labels(
                labels=labels,
                labelled_mask=labelled_mask,
                component_labels=component_labels,
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
        source="range_continuity_retained_frequency_training_label_replicates",
    )
    diagnostics.update(
        {
            "mode": "continuity_retained_frequency",
            "budget_loss_ratios": list(_target_budget_ratios(model_config)),
            "budget_weights": list(
                _target_budget_weights(model_config, _target_budget_ratios(model_config))
            ),
            "budget_weight_power": float(
                getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0
            ),
            "component_target_blend": float(
                getattr(model_config, "range_component_target_blend", 1.0)
            ),
            "continuity_component_weights": dict(RANGE_CONTINUITY_TARGET_WEIGHTS),
            "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
            "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
            "per_replicate": per_replicate,
        }
    )
    return aggregated, aggregated_mask, diagnostics


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
