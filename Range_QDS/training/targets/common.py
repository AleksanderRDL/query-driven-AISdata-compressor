"""Shared helpers for scalar range-training target builders."""

from __future__ import annotations

import math

import torch

from queries.query_types import QUERY_TYPE_ID_RANGE
from simplification.simplify_trajectories import (
    evenly_spaced_indices,
    simplify_with_temporal_score_hybrid,
)
from training.targets.modes import RANGE_TARGET_BALANCE_MODES
from training.training_losses import _safe_quantile


def _scaled_training_target_for_type(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    type_idx: int,
) -> torch.Tensor:
    """Rescale one pure-workload F1 label stream while preserving rank order."""
    target = labels[:, type_idx].clone()
    positive = labelled_mask[:, type_idx] & (labels[:, type_idx] > 0)
    if not bool(positive.any().item()):
        return target.zero_()
    scale = _safe_quantile(labels[positive, type_idx].detach(), 0.95).clamp(min=1e-6)
    return torch.clamp(target / scale, 0.0, 1.0)


def _apply_temporal_residual_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
    temporal_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop supervision for points the temporal base already keeps."""
    residual_labels = labels.clone()
    residual_mask = labelled_mask.clone()
    base_mask = torch.zeros((labels.shape[0],), dtype=torch.bool, device=labels.device)
    base_fraction = min(1.0, max(0.0, float(temporal_fraction)))

    for start, end in boundaries:
        point_count = int(end - start)
        if point_count <= 0:
            continue
        k_total = min(point_count, max(2, math.ceil(float(compression_ratio) * point_count)))
        k_base = (
            0 if base_fraction <= 0.0 else min(k_total, max(2, math.ceil(k_total * base_fraction)))
        )
        base_idx = evenly_spaced_indices(point_count, k_base, labels.device)
        base_mask[start + base_idx] = True

    residual_labels[base_mask] = 0.0
    residual_mask[base_mask] = False
    return residual_labels, residual_mask


def _target_budget_ratios(model_config: object) -> tuple[float, ...]:
    """Return configured budgets used to convert label values into retained frequency."""
    raw = getattr(model_config, "budget_loss_ratios", None) or []
    if not raw:
        raw = getattr(model_config, "range_audit_compression_ratios", None) or []
    if not raw:
        raw = [float(getattr(model_config, "compression_ratio", 0.05))]
    ratios = sorted({float(value) for value in raw if 0.0 < float(value) <= 1.0})
    return tuple(ratios) if ratios else (float(getattr(model_config, "compression_ratio", 0.05)),)


def _target_budget_weights(model_config: object, ratios: tuple[float, ...]) -> tuple[float, ...]:
    """Return normalized retained-frequency target weights for budget ratios."""
    if not ratios:
        return ()
    power = max(0.0, float(getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0))
    if power <= 0.0:
        weight = 1.0 / float(len(ratios))
        return tuple(weight for _ratio in ratios)
    raw = [float(max(float(ratio), 1e-9)) ** (-power) for ratio in ratios]
    total = sum(raw)
    if total <= 1e-12:
        weight = 1.0 / float(len(ratios))
        return tuple(weight for _ratio in ratios)
    return tuple(value / total for value in raw)


def aggregate_range_label_sets(
    label_sets: list[tuple[torch.Tensor, torch.Tensor]],
    type_idx: int = QUERY_TYPE_ID_RANGE,
    source: str = "range_training_label_replicates",
    aggregation: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Aggregate range labels over independent training workloads.

    The aggregation is training-only supervision. It does not expose validation
    or eval queries to the blind compressor.
    """
    if not label_sets:
        raise ValueError("label_sets must contain at least one label/mask pair.")

    first_labels, first_mask = label_sets[0]
    if first_labels.ndim != 2 or first_mask.shape != first_labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= first_labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(first_labels.shape)}.")
    aggregation_mode = aggregation.strip().lower()
    if aggregation_mode not in {"mean", "max"}:
        raise ValueError("aggregation must be 'mean' or 'max'.")

    label_values = torch.zeros_like(first_labels[:, type_idx], dtype=torch.float32)
    label_count = torch.zeros_like(label_values)
    aggregated_mask = first_mask.clone()
    for labels, labelled_mask in label_sets:
        if labels.shape != first_labels.shape or labelled_mask.shape != first_mask.shape:
            raise ValueError("all label sets must have identical shapes.")
        active = labelled_mask[:, type_idx].to(dtype=torch.bool)
        active_values = labels[active, type_idx].float()
        if aggregation_mode == "mean":
            label_values[active] += active_values
        else:
            label_values[active] = torch.maximum(label_values[active], active_values)
        label_count[active] += 1.0
        aggregated_mask |= labelled_mask

    aggregated = first_labels.clone()
    if aggregation_mode == "mean":
        target = label_values / label_count.clamp(min=1.0)
    else:
        target = label_values
    aggregated[:, type_idx] = target.to(dtype=aggregated.dtype)
    aggregated_mask[:, type_idx] = label_count > 0

    positive = aggregated_mask[:, type_idx] & (aggregated[:, type_idx] > 0.0)
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "source": source,
        "aggregation": aggregation_mode,
        "replicate_count": len(label_sets),
        "labelled_point_count": int((label_count > 0).sum().item()),
        "positive_label_count": positive_count,
        "positive_label_fraction": float(positive_count / max(1, int(aggregated.shape[0]))),
        "positive_label_mass": float(aggregated[positive, type_idx].sum().item())
        if positive_count > 0
        else 0.0,
    }
    return aggregated, aggregated_mask, diagnostics


def balance_range_training_target_by_trajectory(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    mode: str = "none",
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Optionally rebalance range target mass across train trajectories.

    This is a training-only transform. It is useful for diagnosing whether a
    blind prior is dominated by a few dense historical routes even though final
    retention budgets are allocated per trajectory.
    """
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")
    balance_mode = str(mode).lower()
    if balance_mode not in RANGE_TARGET_BALANCE_MODES:
        raise ValueError(f"range target balance mode must be one of {RANGE_TARGET_BALANCE_MODES}.")

    target = labels[:, type_idx].float()
    positive = labelled_mask[:, type_idx].to(dtype=torch.bool) & (target > 0.0)
    before_mass = float(target[positive].sum().item()) if bool(positive.any().item()) else 0.0
    before_count = int(positive.sum().item())
    if balance_mode == "none":
        return (
            labels,
            labelled_mask,
            {
                "enabled": False,
                "mode": "none",
                "positive_label_count": before_count,
                "positive_label_mass": before_mass,
            },
        )

    balanced = labels.clone()
    balanced_target = balanced[:, type_idx].float()
    trajectory_masses: list[float] = []
    balanced_trajectory_count = 0
    for start, end in boundaries:
        start_i = int(start)
        end_i = int(end)
        if end_i <= start_i:
            continue
        local_positive = positive[start_i:end_i]
        if not bool(local_positive.any().item()):
            continue
        local_values = balanced_target[start_i:end_i]
        local_mass = float(local_values[local_positive].sum().item())
        if local_mass <= 1e-12:
            continue
        local_values[local_positive] = local_values[local_positive] / local_mass
        balanced_target[start_i:end_i] = local_values
        trajectory_masses.append(local_mass)
        balanced_trajectory_count += 1

    balanced[:, type_idx] = balanced_target.to(dtype=balanced.dtype).clamp(0.0, 1.0)
    after_positive = labelled_mask[:, type_idx].to(dtype=torch.bool) & (balanced[:, type_idx] > 0.0)
    after_count = int(after_positive.sum().item())
    after_mass = float(balanced[after_positive, type_idx].sum().item()) if after_count > 0 else 0.0
    mass_tensor = (
        torch.tensor(trajectory_masses, dtype=torch.float32, device=labels.device)
        if trajectory_masses
        else torch.empty((0,), dtype=torch.float32, device=labels.device)
    )
    diagnostics: dict[str, object] = {
        "enabled": True,
        "mode": balance_mode,
        "positive_label_count": after_count,
        "positive_label_mass": after_mass,
        "positive_label_count_before_balance": before_count,
        "positive_label_mass_before_balance": before_mass,
        "balanced_trajectory_count": int(balanced_trajectory_count),
    }
    if int(mass_tensor.numel()) > 0:
        diagnostics.update(
            {
                "trajectory_positive_mass_p50_before_balance": float(
                    torch.quantile(mass_tensor, 0.50).item()
                ),
                "trajectory_positive_mass_p90_before_balance": float(
                    torch.quantile(mass_tensor, 0.90).item()
                ),
                "trajectory_positive_mass_max_before_balance": float(mass_tensor.max().item()),
            }
        )
    return balanced, labelled_mask.clone(), diagnostics


def _retained_frequency_from_scores(
    source_scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
    ratios: tuple[float, ...],
) -> tuple[torch.Tensor, int]:
    """Return retained-set frequency from one nonnegative score stream."""
    source_positive = source_scores > 0.0
    retained_frequency = torch.zeros_like(source_scores, dtype=torch.float32)
    used = 0
    used_weight = 0.0
    for ratio, budget_weight in zip(
        ratios, _target_budget_weights(model_config, ratios), strict=False
    ):
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
    return retained_frequency, used


def _temporal_retained_frequency(
    scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    ratios: tuple[float, ...],
    weights: tuple[float, ...] | None = None,
) -> tuple[torch.Tensor, int]:
    """Return retained frequency for pure evenly spaced temporal sampling."""
    retained_frequency = torch.zeros_like(scores, dtype=torch.float32)
    budget_weights = weights or tuple(1.0 / float(len(ratios)) for _ratio in ratios)
    used = 0
    used_weight = 0.0
    for ratio, budget_weight in zip(ratios, budget_weights, strict=False):
        retained = torch.zeros_like(scores, dtype=torch.bool)
        for start, end in boundaries:
            point_count = int(end - start)
            if point_count <= 0:
                continue
            keep_count = min(point_count, max(2, math.ceil(float(ratio) * point_count)))
            local_indices = evenly_spaced_indices(point_count, keep_count, scores.device)
            retained[start + local_indices] = True
        retained_frequency += float(budget_weight) * retained.to(dtype=retained_frequency.dtype)
        used += 1
        used_weight += float(budget_weight)
    if used_weight > 1e-12:
        retained_frequency = retained_frequency / float(used_weight)
    return retained_frequency, used


def _apply_temporal_target_blend(
    retained_frequency: torch.Tensor,
    boundaries: list[tuple[int, int]],
    model_config: object,
    ratios: tuple[float, ...],
) -> tuple[torch.Tensor, dict[str, object]]:
    """Blend query-blind temporal anchor frequency into a retained-frequency target."""
    blend = max(
        0.0, min(1.0, float(getattr(model_config, "range_temporal_target_blend", 0.0) or 0.0))
    )
    diagnostics: dict[str, object] = {"temporal_target_blend": float(blend)}
    if blend <= 0.0:
        return retained_frequency, diagnostics

    temporal_frequency, used = _temporal_retained_frequency(
        scores=retained_frequency,
        boundaries=boundaries,
        ratios=ratios,
        weights=_target_budget_weights(model_config, ratios),
    )
    target = ((1.0 - blend) * retained_frequency + blend * temporal_frequency).clamp(0.0, 1.0)
    positive = temporal_frequency > 0.0
    diagnostics.update(
        {
            "temporal_target_budget_count": int(used),
            "temporal_target_positive_label_count": int(positive.sum().item()),
            "temporal_target_positive_label_mass": (
                float(temporal_frequency[positive].sum().item())
                if bool(positive.any().item())
                else 0.0
            ),
        }
    )
    return target, diagnostics


def _temporal_base_mask_for_ratio(
    *,
    n_points: int,
    boundaries: list[tuple[int, int]],
    ratio: float,
    temporal_fraction: float,
    device: torch.device,
) -> torch.Tensor:
    """Return the query-blind temporal base retained at one total budget."""
    base_mask = torch.zeros((n_points,), dtype=torch.bool, device=device)
    base_fraction = min(1.0, max(0.0, float(temporal_fraction)))
    if base_fraction <= 0.0:
        return base_mask

    total_ratio = min(1.0, max(0.0, float(ratio)))
    if total_ratio <= 0.0:
        return base_mask
    for start, end in boundaries:
        point_count = int(end - start)
        if point_count <= 0:
            continue
        keep_count = min(point_count, max(2, math.ceil(total_ratio * point_count)))
        base_count = min(keep_count, max(2, math.ceil(keep_count * base_fraction)))
        base_indices = evenly_spaced_indices(point_count, base_count, device)
        base_mask[start + base_indices] = True
    return base_mask


def _scale01(values: torch.Tensor) -> torch.Tensor:
    """Return per-vector min-max scores in [0, 1]."""
    if int(values.numel()) == 0:
        return values.float()
    values_f = values.float()
    span = values_f.max() - values_f.min()
    if float(span.item()) <= 1e-12:
        return torch.zeros_like(values_f)
    return (values_f - values_f.min()) / span.clamp(min=1e-12)


def _numeric_diagnostic(diagnostics: dict[str, object], key: str, default: float = 0.0) -> float:
    """Read a numeric diagnostics field defensively."""
    value = diagnostics.get(key, default)
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)
