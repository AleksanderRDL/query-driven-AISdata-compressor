"""Retained-frequency scalar range target builders."""

from __future__ import annotations

import math

import torch

from queries.query_types import QUERY_TYPE_ID_RANGE
from simplification.simplify_trajectories import (
    simplify_with_global_score_budget,
    simplify_with_temporal_score_hybrid,
)
from training.model_features import (
    HISTORICAL_PRIOR_CLOCK_DIM,
    HISTORICAL_PRIOR_DENSITY_DIM,
    build_historical_prior_point_features,
)
from training.targets.common import (
    _apply_temporal_target_blend,
    _numeric_diagnostic,
    _retained_frequency_from_scores,
    _target_budget_ratios,
    _target_budget_weights,
)
from training.training_losses import _safe_quantile


def range_retained_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Convert range label values into oracle retained-set frequency targets.

    This remains workload-blind at inference: only training-workload labels are
    transformed, and the eval compressor still receives point features only.
    """
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")

    ratios = _target_budget_ratios(model_config)
    budget_weights = _target_budget_weights(model_config, ratios)
    source_scores = labels[:, type_idx].float().clamp(min=0.0)
    source_positive = source_scores > 0.0
    retained_frequency = torch.zeros_like(source_scores)
    used = 0
    used_weight = 0.0
    for ratio, budget_weight in zip(ratios, budget_weights, strict=False):
        mask = simplify_with_temporal_score_hybrid(
            source_scores,
            boundaries,
            float(ratio),
            temporal_fraction=float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
            diversity_bonus=float(getattr(model_config, "mlqds_diversity_bonus", 0.0)),
            hybrid_mode=str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
            stratified_center_weight=float(
                getattr(model_config, "mlqds_stratified_center_weight", 0.0)
            ),
            min_learned_swaps=int(getattr(model_config, "mlqds_min_learned_swaps", 0)),
        )
        retained_frequency += float(budget_weight) * (mask & source_positive).to(
            dtype=retained_frequency.dtype
        )
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
        "mode": "retained_frequency",
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
        "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
        "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
    }
    diagnostics.update(temporal_blend_diagnostics)
    return transformed, transformed_mask, diagnostics


def _global_budget_retained_frequency_from_scores(
    source_scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
    ratios: tuple[float, ...],
) -> tuple[torch.Tensor, int]:
    """Return retained frequency from global score allocation across trajectories."""
    source_positive = source_scores > 0.0
    retained_frequency = torch.zeros_like(source_scores, dtype=torch.float32)
    used = 0
    used_weight = 0.0
    for ratio, budget_weight in zip(
        ratios, _target_budget_weights(model_config, ratios), strict=False
    ):
        mask = simplify_with_global_score_budget(
            source_scores,
            boundaries,
            float(ratio),
        )
        retained_frequency += float(budget_weight) * (mask & source_positive).to(
            dtype=retained_frequency.dtype
        )
        used += 1
        used_weight += float(budget_weight)
    if used_weight > 1e-12:
        retained_frequency = retained_frequency / float(used_weight)
    return retained_frequency, used


def range_global_budget_retained_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Convert range labels into global-scarcity retained-frequency targets.

    This is training-only supervision. It uses train-workload labels to ask
    which useful points would win under a database-level budget, then the eval
    compressor still scores points without eval queries.
    """
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")

    ratios = _target_budget_ratios(model_config)
    budget_weights = _target_budget_weights(model_config, ratios)
    source_scores = labels[:, type_idx].float().clamp(min=0.0)
    retained_frequency, used = _global_budget_retained_frequency_from_scores(
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
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "mode": "global_budget_retained_frequency",
        "source": "range_training_labels_global_budget",
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
        "global_budget_frequency_budget_count": int(used),
        "global_budget_min_points_per_trajectory": 2,
    }
    diagnostics.update(temporal_blend_diagnostics)
    return transformed, transformed_mask, diagnostics


def _historical_prior_support_mask(
    targets: torch.Tensor,
    boundaries: list[tuple[int, int]],
    support_ratio: float,
) -> torch.Tensor:
    """Return per-trajectory top-target support for historical-prior teachers."""
    ratio = min(1.0, max(0.0, float(support_ratio)))
    support_mask = torch.zeros((int(targets.shape[0]),), dtype=torch.bool, device=targets.device)
    if ratio >= 1.0:
        support_mask[:] = True
        return support_mask
    if ratio <= 0.0:
        return support_mask

    for start, end in boundaries:
        point_count = int(end - start)
        if point_count <= 0:
            continue
        keep_count = min(point_count, max(1, math.ceil(ratio * point_count)))
        if keep_count >= point_count:
            support_mask[start:end] = True
            continue
        local_targets = targets[start:end].float()
        local_indices = torch.topk(local_targets, k=keep_count, largest=True).indices
        support_mask[start + local_indices] = True
    return support_mask


def _minmax_scale_feature_matrix(features: torch.Tensor) -> torch.Tensor:
    """Scale feature columns with the same min-max convention as FeatureScaler."""
    if features.ndim != 2:
        raise ValueError("features must be a matrix.")
    if int(features.shape[0]) == 0:
        return features.float()
    values = features.float()
    feature_min = values.min(dim=0).values
    feature_max = values.max(dim=0).values
    denom = (feature_max - feature_min).clamp(min=1e-6)
    return (values - feature_min) / denom


def _weight_historical_prior_features(features: torch.Tensor, model_config: object) -> torch.Tensor:
    """Apply historical-prior feature weights used by the KNN diagnostic model."""
    weighted = features.float().clone()
    point_dim = int(weighted.shape[1])
    density_dim = HISTORICAL_PRIOR_DENSITY_DIM if point_dim >= 21 else 0
    clock_dim = HISTORICAL_PRIOR_CLOCK_DIM if point_dim >= 23 else 0
    clock_weight = max(0.0, float(getattr(model_config, "historical_prior_clock_weight", 0.0)))
    density_weight = max(0.0, float(getattr(model_config, "historical_prior_density_weight", 1.0)))
    if clock_dim > 0 and clock_weight != 1.0:
        clock_start = point_dim - density_dim - clock_dim
        clock_end = point_dim - density_dim
        weighted[:, clock_start:clock_end] *= clock_weight
    if density_dim > 0 and density_weight != 1.0:
        weighted[:, -density_dim:] *= density_weight
    return weighted


def _historical_prior_teacher_scores(
    points: torch.Tensor,
    targets: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Score train points with a leave-one-out query-free historical KNN teacher."""
    if points.ndim != 2:
        raise ValueError("points must have shape [n_points, point_dim].")
    if targets.ndim != 1 or int(targets.shape[0]) != int(points.shape[0]):
        raise ValueError("targets must be a vector matching points.")
    if int(points.shape[0]) == 0:
        raise ValueError("historical-prior teacher requires at least one point.")

    features = _minmax_scale_feature_matrix(build_historical_prior_point_features(points))
    weighted_features = _weight_historical_prior_features(features, model_config)
    support_ratio = min(
        1.0, max(0.0, float(getattr(model_config, "historical_prior_support_ratio", 1.0)))
    )
    support_mask = _historical_prior_support_mask(targets, boundaries, support_ratio)
    if not bool(support_mask.any().item()):
        raise ValueError("historical_prior_support_ratio removed every teacher support point.")
    pre_min_support_count = int(support_mask.sum().item())

    min_target = max(0.0, float(getattr(model_config, "historical_prior_min_target", 0.0)))
    if min_target > 0.0:
        support_mask &= targets >= min_target
        if not bool(support_mask.any().item()):
            raise ValueError(
                "historical_prior_min_target removed every teacher support point; "
                f"threshold={min_target:.6f}, max_target={float(targets.max().item()):.6f}."
            )

    support_indices = torch.where(support_mask)[0]
    support_features = weighted_features[support_indices]
    support_targets = targets[support_indices].float().clamp(0.0, 1.0)
    support_count = int(support_features.shape[0])
    k = min(max(1, int(getattr(model_config, "historical_prior_k", 32))), support_count)
    chunk_size = max(1, min(max(1, int(getattr(model_config, "query_chunk_size", 1024))), 1024))
    scores = torch.empty((int(points.shape[0]),), dtype=torch.float32, device=points.device)

    all_indices = torch.arange(
        int(points.shape[0]), dtype=support_indices.dtype, device=points.device
    )
    support_indices = support_indices.to(device=points.device)
    support_features = support_features.to(device=points.device)
    support_targets = support_targets.to(device=points.device)
    weighted_features = weighted_features.to(device=points.device)
    for start in range(0, int(points.shape[0]), chunk_size):
        end = min(int(points.shape[0]), start + chunk_size)
        distances = torch.cdist(weighted_features[start:end], support_features, p=2).square()
        same_point = all_indices[start:end].unsqueeze(1) == support_indices.unsqueeze(0)
        if bool(same_point.any().item()):
            distances = distances.masked_fill(same_point, float("inf"))
        nearest_distances, nearest_idx = torch.topk(distances, k=k, largest=False, dim=1)
        finite = torch.isfinite(nearest_distances)
        weights = torch.where(
            finite, 1.0 / (nearest_distances + 1e-4), torch.zeros_like(nearest_distances)
        )
        denom = weights.sum(dim=1)
        local_scores = (weights * support_targets[nearest_idx]).sum(dim=1) / denom.clamp(min=1e-9)
        fallback = targets[start:end].float().clamp(0.0, 1.0)
        scores[start:end] = torch.where(denom > 1e-12, local_scores, fallback)

    positive = scores > 0.0
    diagnostics: dict[str, object] = {
        "historical_prior_teacher_k": int(k),
        "historical_prior_teacher_leave_one_out": True,
        "historical_prior_support_ratio": float(support_ratio),
        "historical_prior_support_pre_min_count": int(pre_min_support_count),
        "historical_prior_min_target": float(min_target),
        "historical_prior_stored_support_count": int(support_count),
        "historical_prior_stored_support_fraction": float(
            support_count / max(1, int(points.shape[0]))
        ),
        "historical_prior_clock_weight": float(
            getattr(model_config, "historical_prior_clock_weight", 0.0)
        ),
        "historical_prior_density_weight": float(
            getattr(model_config, "historical_prior_density_weight", 1.0)
        ),
        "historical_prior_teacher_positive_score_count": int(positive.sum().item()),
        "historical_prior_teacher_positive_score_fraction": float(
            int(positive.sum().item()) / max(1, int(scores.numel()))
        ),
        "historical_prior_teacher_score_mass": float(scores[positive].sum().item())
        if bool(positive.any().item())
        else 0.0,
    }
    if bool(positive.any().item()):
        diagnostics.update(
            {
                "historical_prior_teacher_score_p50": float(
                    _safe_quantile(scores[positive], 0.50).item()
                ),
                "historical_prior_teacher_score_p95": float(
                    _safe_quantile(scores[positive], 0.95).item()
                ),
            }
        )
    return scores, diagnostics


def range_historical_prior_retained_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Distill retained-frequency labels through a query-free historical KNN teacher.

    This is a training-only teacher. Eval compression still uses the configured
    neural blind model and receives no eval query boxes before masks are frozen.
    """
    retained_labels, retained_mask, base_diagnostics = range_retained_frequency_training_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        boundaries=boundaries,
        model_config=model_config,
        type_idx=type_idx,
    )
    retained_target = retained_labels[:, type_idx].float().clamp(0.0, 1.0)
    teacher_scores, teacher_diagnostics = _historical_prior_teacher_scores(
        points=points,
        targets=retained_target,
        boundaries=boundaries,
        model_config=model_config,
    )
    ratios = _target_budget_ratios(model_config)
    retained_frequency, used = _retained_frequency_from_scores(
        source_scores=teacher_scores,
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
    transformed_mask = retained_mask.clone()
    transformed_mask[:, type_idx] = True
    positive = transformed[:, type_idx] > 0.0
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "mode": "historical_prior_retained_frequency",
        "source": "range_retained_frequency_leave_one_out_historical_prior_teacher",
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
        "teacher_retained_frequency_budget_count": int(used),
        "base_retained_frequency_positive_label_count": int(
            _numeric_diagnostic(base_diagnostics, "positive_label_count")
        ),
        "base_retained_frequency_positive_label_fraction": _numeric_diagnostic(
            base_diagnostics,
            "positive_label_fraction",
        ),
        "base_retained_frequency_positive_label_mass": _numeric_diagnostic(
            base_diagnostics,
            "positive_label_mass",
        ),
        "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
        "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
    }
    diagnostics.update(teacher_diagnostics)
    diagnostics.update(temporal_blend_diagnostics)
    return transformed, transformed_mask, diagnostics
