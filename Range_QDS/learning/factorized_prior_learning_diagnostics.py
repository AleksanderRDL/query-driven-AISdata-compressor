"""Prior-feature learning diagnostics for factorized heads."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch

from learning.factorized_prior_stage_diagnostics import (
    _feature_target_alignment_rows,
    _prediction_sensitivity,
    _prior_reconstruction_from_non_prior_features,
    _prior_stage_sensitivity_diagnostics,
)
from learning.factorized_prior_transfer_diagnostics import (
    _prior_to_head_transfer_sensitivity_diagnostics,
)
from learning.query_prior_fields import QUERY_PRIOR_FIELD_NAMES
from learning.targets.query_local_utility import (
    QUERY_LOCAL_UTILITY_HEAD_NAMES,
)


def _head_probability_sensitivity(
    *,
    primary_head_logits: torch.Tensor | None,
    ablation_head_logits: torch.Tensor | None,
) -> dict[str, Any]:
    if primary_head_logits is None or ablation_head_logits is None:
        return {"available": False, "reason": "missing_head_logits"}
    primary = primary_head_logits.detach().cpu().float()
    ablation = ablation_head_logits.detach().cpu().float()
    if primary.shape != ablation.shape or primary.ndim != 2:
        return {
            "available": False,
            "reason": "shape_mismatch",
            "primary_shape": list(primary.shape),
            "ablation_shape": list(ablation.shape),
        }
    primary_prob = torch.sigmoid(primary)
    ablation_prob = torch.sigmoid(ablation)
    delta = primary_prob - ablation_prob
    per_head: dict[str, Any] = {}
    for head_idx, head_name in enumerate(QUERY_LOCAL_UTILITY_HEAD_NAMES):
        if head_idx >= int(primary_prob.shape[1]):
            continue
        left = primary_prob[:, head_idx]
        right = ablation_prob[:, head_idx]
        local_delta = left - right
        per_head[str(head_name)] = {
            "mean_abs_probability_delta": float(local_delta.abs().mean().item()),
            "max_abs_probability_delta": float(local_delta.abs().max().item()),
            "primary_probability_std": float(left.std(unbiased=False).item())
            if int(left.numel()) > 1
            else 0.0,
            "ablation_probability_std": float(right.std(unbiased=False).item())
            if int(right.numel()) > 1
            else 0.0,
            "primary_probability_mean": float(left.mean().item()),
            "ablation_probability_mean": float(right.mean().item()),
        }
    return {
        "available": True,
        "point_count": int(primary_prob.shape[0]),
        "head_count": int(primary_prob.shape[1]),
        "mean_abs_head_probability_delta": float(delta.abs().mean().item()),
        "max_abs_head_probability_delta": float(delta.abs().max().item()),
        "per_head": per_head,
    }


def _prior_path_strength_diagnostics(
    *,
    model: torch.nn.Module | None,
    norm_points: torch.Tensor,
) -> dict[str, Any]:
    if model is None:
        return {"available": False, "reason": "missing_model"}
    point_encoder = getattr(model, "point_encoder", None)
    prior_encoder = getattr(model, "prior_feature_encoder", None)
    prior_features_fn = getattr(model, "_prior_features", None)
    if not callable(point_encoder) or not callable(prior_encoder) or not callable(prior_features_fn):
        return {"available": False, "reason": "missing_prior_path_modules"}
    point_encoder_fn = cast(Callable[[torch.Tensor], torch.Tensor], point_encoder)
    prior_encoder_fn = cast(Callable[[torch.Tensor], torch.Tensor], prior_encoder)
    prior_features_callable = cast(Callable[[torch.Tensor], torch.Tensor], prior_features_fn)
    original_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            points = norm_points.detach().cpu().float().unsqueeze(0)
            point_encoded = point_encoder_fn(points)
            prior_features = prior_features_callable(points)
            prior_encoded = prior_encoder_fn(prior_features)
            scale_tensor = getattr(model, "prior_feature_scale", None)
            scale = (
                float(scale_tensor.detach().cpu().item())
                if isinstance(scale_tensor, torch.Tensor)
                else 1.0
            )
            scaled_prior = prior_encoded * scale
    finally:
        model.train(original_training)
    point_std = float(point_encoded.std(unbiased=False).item())
    prior_std = float(prior_encoded.std(unbiased=False).item())
    scaled_std = float(scaled_prior.std(unbiased=False).item())
    point_norm = float(point_encoded.norm().item())
    scaled_norm = float(scaled_prior.norm().item())
    return {
        "available": True,
        "prior_feature_scale": float(scale),
        "point_encoder_output_std": point_std,
        "prior_encoder_output_std": prior_std,
        "scaled_prior_encoder_output_std": scaled_std,
        "scaled_prior_to_point_std_ratio": float(scaled_std / max(point_std, 1e-12)),
        "scaled_prior_to_point_l2_ratio": float(scaled_norm / max(point_norm, 1e-12)),
    }


def _prior_feature_learning_diagnostics(
    *,
    model: torch.nn.Module | None,
    norm_points: torch.Tensor | None,
    primary_predictions: torch.Tensor | None,
    zero_prior_predictions: torch.Tensor | None,
    primary_head_logits: torch.Tensor | None,
    zero_prior_head_logits: torch.Tensor | None,
    factorized_targets: torch.Tensor | None,
    factorized_mask: torch.Tensor | None,
    scalar_target: torch.Tensor | None,
    scalar_mask: torch.Tensor | None,
    raw_points: torch.Tensor | None = None,
    boundaries: list[tuple[int, int]] | None = None,
    typed_queries: list[dict[str, Any]] | None = None,
    window_length: int = 512,
    window_stride: int = 256,
    batch_size: int = 16,
    segment_budget_head_weight: float = 0.10,
    segment_level_loss_weight: float = 0.25,
    behavior_rank_loss_weight: float = 0.25,
    sparse_head_rank_loss_weight: float = 0.0,
    sparse_head_bce_target_mode: str = "raw",
    seed: int = 0,
) -> dict[str, Any]:
    """Localize query-prior blindness without changing model behavior."""
    del seed
    if norm_points is None or factorized_targets is None or factorized_mask is None:
        return {"prior_feature_learning_diagnostics_available": False}
    prior_dim = len(QUERY_PRIOR_FIELD_NAMES)
    if norm_points.ndim != 2 or int(norm_points.shape[1]) < prior_dim:
        return {
            "prior_feature_learning_diagnostics_available": False,
            "reason": "point_feature_shape_missing_prior_channels",
        }
    if factorized_targets.shape != factorized_mask.shape:
        return {
            "prior_feature_learning_diagnostics_available": False,
            "reason": "target_mask_shape_mismatch",
        }
    if int(factorized_targets.shape[0]) != int(norm_points.shape[0]):
        return {
            "prior_feature_learning_diagnostics_available": False,
            "reason": "point_target_row_mismatch",
        }

    points_cpu = norm_points.detach().cpu().float()
    prior_features = points_cpu[:, -prior_dim:]
    non_prior_features = points_cpu[:, :-prior_dim]
    targets = factorized_targets.detach().cpu().float().clamp(0.0, 1.0)
    masks = factorized_mask.detach().cpu().bool()
    prior_feature_names = [str(name) for name in QUERY_PRIOR_FIELD_NAMES]
    non_prior_feature_names = [f"non_prior_{idx}" for idx in range(int(non_prior_features.shape[1]))]
    prior_stats: dict[str, Any] = {}
    for idx, name in enumerate(prior_feature_names):
        values = prior_features[:, idx]
        prior_stats[name] = {
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()) if int(values.numel()) > 1 else 0.0,
            "nonzero_fraction": float((values.abs() > 1e-12).float().mean().item()),
        }

    head_rows: dict[str, Any] = {}
    prior_beats_non_prior = 0
    material_prior_signal_heads = 0
    for head_idx, head_name in enumerate(QUERY_LOCAL_UTILITY_HEAD_NAMES):
        if head_idx >= int(targets.shape[1]):
            continue
        valid = masks[:, head_idx]
        if int(valid.sum().item()) < 2:
            head_rows[str(head_name)] = {"available": False, "reason": "insufficient_valid_points"}
            continue
        target = targets[:, head_idx]
        prior_alignment = _feature_target_alignment_rows(
            features=prior_features,
            feature_names=prior_feature_names,
            target=target,
            valid=valid,
            include_per_feature=True,
        )
        non_prior_alignment = _feature_target_alignment_rows(
            features=non_prior_features,
            feature_names=non_prior_feature_names,
            target=target,
            valid=valid,
            include_per_feature=False,
        )
        prior_best = prior_alignment.get("best_spearman") or {}
        non_prior_best = non_prior_alignment.get("best_spearman") or {}
        prior_value = prior_best.get("value")
        non_prior_value = non_prior_best.get("value")
        prior_material = isinstance(prior_value, int | float) and float(prior_value) > 0.05
        prior_wins = (
            isinstance(prior_value, int | float)
            and isinstance(non_prior_value, int | float)
            and float(prior_value) > float(non_prior_value)
        )
        material_prior_signal_heads += int(bool(prior_material))
        prior_beats_non_prior += int(bool(prior_wins))
        head_rows[str(head_name)] = {
            "available": True,
            "valid_point_count": int(valid.sum().item()),
            "target_std": float(target[valid].std(unbiased=False).item())
            if int(valid.sum().item()) > 1
            else 0.0,
            "best_prior_channel": prior_best,
            "best_non_prior_feature": non_prior_best,
            "prior_best_spearman_beats_non_prior": bool(prior_wins),
            "prior_channel_alignment": prior_alignment.get("per_feature"),
            "best_prior_topk_target_mass_recall_at_5_percent": prior_alignment.get(
                "best_topk_target_mass_recall_at_5_percent"
            ),
            "best_non_prior_topk_target_mass_recall_at_5_percent": non_prior_alignment.get(
                "best_topk_target_mass_recall_at_5_percent"
            ),
        }

    final_target_alignment: dict[str, Any] = {"available": False, "reason": "missing_scalar_target"}
    if scalar_target is not None and scalar_mask is not None:
        scalar = scalar_target.detach().cpu().float().flatten().clamp(0.0, 1.0)
        valid = scalar_mask.detach().cpu().bool().flatten()
        if int(scalar.numel()) == int(norm_points.shape[0]) and int(valid.sum().item()) >= 2:
            prior_alignment = _feature_target_alignment_rows(
                features=prior_features,
                feature_names=prior_feature_names,
                target=scalar,
                valid=valid,
                include_per_feature=True,
            )
            non_prior_alignment = _feature_target_alignment_rows(
                features=non_prior_features,
                feature_names=non_prior_feature_names,
                target=scalar,
                valid=valid,
                include_per_feature=False,
            )
            final_target_alignment = {
                "available": True,
                "best_prior_channel": prior_alignment.get("best_spearman"),
                "best_non_prior_feature": non_prior_alignment.get("best_spearman"),
                "prior_channel_alignment": prior_alignment.get("per_feature"),
                "best_prior_topk_target_mass_recall_at_5_percent": prior_alignment.get(
                    "best_topk_target_mass_recall_at_5_percent"
                ),
                "best_non_prior_topk_target_mass_recall_at_5_percent": non_prior_alignment.get(
                    "best_topk_target_mass_recall_at_5_percent"
                ),
            }

    head_sensitivity = _head_probability_sensitivity(
        primary_head_logits=primary_head_logits,
        ablation_head_logits=zero_prior_head_logits,
    )
    final_probability_sensitivity = _prediction_sensitivity(
        primary=primary_predictions,
        ablation=zero_prior_predictions,
        sigmoid=True,
    )
    path_strength = _prior_path_strength_diagnostics(model=model, norm_points=points_cpu)
    stage_sensitivity = _prior_stage_sensitivity_diagnostics(
        model=model,
        norm_points=points_cpu,
        boundaries=boundaries,
        window_length=window_length,
        window_stride=window_stride,
        batch_size=batch_size,
        prior_dim=prior_dim,
    )
    head_transfer = _prior_to_head_transfer_sensitivity_diagnostics(
        model=model,
        norm_points=points_cpu,
        raw_points=raw_points,
        factorized_targets=targets,
        factorized_mask=masks,
        boundaries=boundaries,
        typed_queries=typed_queries,
        window_length=window_length,
        window_stride=window_stride,
        batch_size=batch_size,
        prior_dim=prior_dim,
        segment_budget_head_weight=segment_budget_head_weight,
        segment_level_loss_weight=segment_level_loss_weight,
        behavior_rank_loss_weight=behavior_rank_loss_weight,
        sparse_head_rank_loss_weight=sparse_head_rank_loss_weight,
        sparse_head_bce_target_mode=sparse_head_bce_target_mode,
    )
    reconstruction = _prior_reconstruction_from_non_prior_features(
        non_prior_features=non_prior_features,
        prior_features=prior_features,
    )

    mean_head_delta = head_sensitivity.get("mean_abs_head_probability_delta")
    final_delta = final_probability_sensitivity.get("mean_abs_delta")
    path_ratio = path_strength.get("scaled_prior_to_point_std_ratio")
    max_reconstruction_r2 = reconstruction.get("max_r2")
    if (
        material_prior_signal_heads > 0
        and isinstance(mean_head_delta, int | float)
        and float(mean_head_delta) < 1e-4
    ):
        classification = "prior_target_signal_available_but_trained_heads_invariant"
    elif isinstance(path_ratio, int | float) and float(path_ratio) < 0.05:
        classification = "prior_path_contribution_too_small"
    elif isinstance(max_reconstruction_r2, int | float) and float(max_reconstruction_r2) > 0.90:
        classification = "prior_channels_reconstructable_from_non_prior_features"
    elif isinstance(final_delta, int | float) and float(final_delta) < 1e-4:
        classification = "prior_signal_suppressed_before_final_score"
    else:
        classification = "diagnostic_only"

    return {
        "prior_feature_learning_diagnostics_available": True,
        "diagnostic_only": True,
        "prior_feature_names": prior_feature_names,
        "prior_feature_stats": prior_stats,
        "head_target_alignment": head_rows,
        "final_target_alignment": final_target_alignment,
        "prior_signal_head_count": int(material_prior_signal_heads),
        "prior_best_spearman_beats_non_prior_head_count": int(prior_beats_non_prior),
        "prior_reconstruction_from_non_prior_features": reconstruction,
        "prior_path_strength": path_strength,
        "prior_stage_sensitivity": stage_sensitivity,
        "prior_to_head_transfer_sensitivity": head_transfer,
        "zero_prior_sensitivity": {
            "head_probabilities": head_sensitivity,
            "final_probability": final_probability_sensitivity,
            "final_logit": _prediction_sensitivity(
                primary=primary_predictions,
                ablation=zero_prior_predictions,
                sigmoid=False,
            ),
        },
        "classification": classification,
    }
