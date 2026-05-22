"""Prior-stage sensitivity diagnostics for factorized heads."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import torch

from learning.query_prior_fields import QUERY_PRIOR_FIELD_NAMES
from learning.targets.query_local_utility import (
    QUERY_LOCAL_UTILITY_HEAD_NAMES,
    _rank_correlation,
)
from learning.trajectory_batching import batch_windows, build_trajectory_windows


def _topk_target_mass_recall(
    *,
    scores: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    ratio: float = 0.05,
) -> float | None:
    valid = valid.bool()
    if int(valid.sum().item()) < 2:
        return None
    valid_scores = scores[valid].float()
    valid_target = target[valid].float().clamp(0.0, 1.0)
    if int(valid_scores.numel()) <= 0:
        return None
    k = max(1, math.ceil(float(ratio) * int(valid_scores.numel())))
    selected = torch.topk(valid_scores, k=k, largest=True).indices
    ideal = torch.topk(valid_target, k=k, largest=True).indices
    selected_mass = float(valid_target[selected].sum().item())
    ideal_mass = float(valid_target[ideal].sum().item())
    return float(selected_mass / max(ideal_mass, 1e-12))


def _feature_target_alignment_rows(
    *,
    features: torch.Tensor,
    feature_names: list[str],
    target: torch.Tensor,
    valid: torch.Tensor,
    include_per_feature: bool,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    best_spearman: dict[str, Any] | None = None
    best_topk: dict[str, Any] | None = None
    for idx in range(int(features.shape[1])):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        scores = features[:, idx].float()
        row = {
            "feature_index": int(idx),
            "feature_name": str(name),
            "spearman": _rank_correlation(scores, target, valid),
            "topk_target_mass_recall_at_5_percent": _topk_target_mass_recall(
                scores=scores,
                target=target,
                valid=valid,
            ),
            "feature_std": float(scores[valid].std(unbiased=False).item())
            if int(valid.sum().item()) > 1
            else 0.0,
        }
        spearman = row["spearman"]
        if spearman is not None and (
            best_spearman is None or float(spearman) > float(best_spearman["value"])
        ):
            best_spearman = {
                "feature_index": int(idx),
                "feature_name": str(name),
                "value": float(spearman),
            }
        topk = row["topk_target_mass_recall_at_5_percent"]
        if topk is not None and (best_topk is None or float(topk) > float(best_topk["value"])):
            best_topk = {
                "feature_index": int(idx),
                "feature_name": str(name),
                "value": float(topk),
            }
        if include_per_feature:
            rows[str(name)] = row
    return {
        "per_feature": rows if include_per_feature else None,
        "best_spearman": best_spearman,
        "best_topk_target_mass_recall_at_5_percent": best_topk,
    }


def _prior_reconstruction_from_non_prior_features(
    *,
    non_prior_features: torch.Tensor,
    prior_features: torch.Tensor,
    ridge: float = 1e-3,
) -> dict[str, Any]:
    if int(non_prior_features.numel()) <= 0 or int(prior_features.numel()) <= 0:
        return {"available": False, "reason": "missing_features"}
    if int(non_prior_features.shape[0]) != int(prior_features.shape[0]):
        return {"available": False, "reason": "row_count_mismatch"}
    x = non_prior_features.detach().cpu().float()
    y = prior_features.detach().cpu().float()
    ones = torch.ones((int(x.shape[0]), 1), dtype=x.dtype)
    design = torch.cat([x, ones], dim=1)
    gram = design.T @ design
    penalty = torch.eye(int(gram.shape[0]), dtype=gram.dtype) * max(0.0, float(ridge))
    penalty[-1, -1] = 0.0
    rows: dict[str, Any] = {}
    for idx, name in enumerate(QUERY_PRIOR_FIELD_NAMES):
        if idx >= int(y.shape[1]):
            continue
        target = y[:, idx]
        try:
            weights = torch.linalg.solve(gram + penalty, design.T @ target)
            prediction = design @ weights
        except RuntimeError:
            weights = torch.linalg.pinv(gram + penalty) @ (design.T @ target)
            prediction = design @ weights
        residual = target - prediction
        total = target - target.mean()
        sse = float((residual * residual).sum().item())
        sst = float((total * total).sum().item())
        rows[str(name)] = {
            "r2": None if sst <= 1e-12 else float(1.0 - sse / sst),
            "mae": float(residual.abs().mean().item()),
            "target_std": float(target.std(unbiased=False).item())
            if int(target.numel()) > 1
            else 0.0,
            "prediction_std": float(prediction.std(unbiased=False).item())
            if int(prediction.numel()) > 1
            else 0.0,
        }
    r2_values = [
        float(row["r2"])
        for row in rows.values()
        if isinstance(row.get("r2"), int | float) and math.isfinite(float(row["r2"]))
    ]
    return {
        "available": bool(rows),
        "method": "ridge_linear_reconstruction_from_non_prior_model_features",
        "ridge": float(ridge),
        "per_prior_channel": rows,
        "mean_r2": float(sum(r2_values) / len(r2_values)) if r2_values else None,
        "max_r2": float(max(r2_values)) if r2_values else None,
    }


def _prediction_sensitivity(
    *,
    primary: torch.Tensor | None,
    ablation: torch.Tensor | None,
    sigmoid: bool,
) -> dict[str, Any]:
    if primary is None or ablation is None:
        return {"available": False, "reason": "missing_predictions"}
    left = primary.detach().cpu().float().flatten()
    right = ablation.detach().cpu().float().flatten()
    if left.shape != right.shape or int(left.numel()) == 0:
        return {
            "available": False,
            "reason": "shape_mismatch",
            "primary_shape": list(left.shape),
            "ablation_shape": list(right.shape),
        }
    if sigmoid:
        left = torch.sigmoid(left)
        right = torch.sigmoid(right)
    delta = left - right
    return {
        "available": True,
        "count": int(left.numel()),
        "mean_abs_delta": float(delta.abs().mean().item()),
        "max_abs_delta": float(delta.abs().max().item()),
        "primary_std": float(left.std(unbiased=False).item()) if int(left.numel()) > 1 else 0.0,
        "ablation_std": float(right.std(unbiased=False).item()) if int(right.numel()) > 1 else 0.0,
    }


def _stage_tensor_sensitivity(
    *,
    primary_parts: list[torch.Tensor],
    ablation_parts: list[torch.Tensor],
) -> dict[str, Any]:
    if not primary_parts or not ablation_parts or len(primary_parts) != len(ablation_parts):
        return {"available": False, "reason": "missing_stage_tensors"}
    primary = torch.cat([part.detach().cpu().float().reshape(-1) for part in primary_parts])
    ablation = torch.cat([part.detach().cpu().float().reshape(-1) for part in ablation_parts])
    if primary.shape != ablation.shape or int(primary.numel()) <= 0:
        return {
            "available": False,
            "reason": "stage_shape_mismatch",
            "primary_count": int(primary.numel()),
            "ablation_count": int(ablation.numel()),
        }
    delta = primary - ablation
    primary_std = float(primary.std(unbiased=False).item()) if int(primary.numel()) > 1 else 0.0
    ablation_std = float(ablation.std(unbiased=False).item()) if int(ablation.numel()) > 1 else 0.0
    mean_abs = float(delta.abs().mean().item())
    l2_primary = float(primary.norm().item())
    l2_delta = float(delta.norm().item())
    return {
        "available": True,
        "value_count": int(primary.numel()),
        "mean_abs_delta": mean_abs,
        "max_abs_delta": float(delta.abs().max().item()),
        "delta_l2": l2_delta,
        "primary_l2": l2_primary,
        "primary_std": primary_std,
        "ablation_std": ablation_std,
        "mean_abs_delta_to_primary_std": float(mean_abs / max(primary_std, 1e-12)),
        "delta_l2_to_primary_l2": float(l2_delta / max(l2_primary, 1e-12)),
    }


def _prior_stage_sensitivity_diagnostics(
    *,
    model: torch.nn.Module | None,
    norm_points: torch.Tensor,
    boundaries: list[tuple[int, int]] | None,
    window_length: int,
    window_stride: int,
    batch_size: int,
    prior_dim: int,
) -> dict[str, Any]:
    """Trace where prior-channel deltas are attenuated inside the range model."""
    if model is None:
        return {"available": False, "reason": "missing_model"}
    if boundaries is None:
        return {"available": False, "reason": "missing_boundaries"}
    if norm_points.ndim != 2 or int(norm_points.shape[1]) < int(prior_dim):
        return {"available": False, "reason": "point_feature_shape_missing_prior_channels"}
    point_encoder = getattr(model, "point_encoder", None)
    prior_feature_encoder = getattr(model, "prior_feature_encoder", None)
    local_context_encoder = getattr(model, "local_context_encoder", None)
    segment_context = getattr(model, "segment_context", None)
    shared_prior_encoder = getattr(model, "prior_encoder", None)
    heads = getattr(model, "heads", None)
    prior_features_fn = getattr(model, "_prior_features", None)
    positional_encoding_fn = getattr(model, "_positional_encoding", None)
    if (
        not callable(point_encoder)
        or not callable(prior_feature_encoder)
        or not callable(segment_context)
        or not callable(shared_prior_encoder)
        or not callable(prior_features_fn)
        or not isinstance(heads, torch.nn.ModuleDict)
    ):
        return {"available": False, "reason": "missing_stage_modules"}
    point_encoder_fn = cast(Callable[[torch.Tensor], torch.Tensor], point_encoder)
    prior_feature_encoder_fn = cast(Callable[[torch.Tensor], torch.Tensor], prior_feature_encoder)
    segment_context_fn = cast(Callable[[torch.Tensor], torch.Tensor], segment_context)
    shared_prior_encoder_fn = cast(Callable[[torch.Tensor], torch.Tensor], shared_prior_encoder)
    prior_features_callable = cast(Callable[[torch.Tensor], torch.Tensor], prior_features_fn)
    positional_callable = (
        cast(Callable[[int, torch.device, torch.dtype], torch.Tensor], positional_encoding_fn)
        if callable(positional_encoding_fn)
        else None
    )

    device = next(model.parameters(), torch.empty(0)).device
    zero_points = norm_points.detach().clone()
    zero_points[:, -int(prior_dim) :] = 0.0
    windows = batch_windows(
        build_trajectory_windows(
            points=norm_points.detach().cpu().float(),
            boundaries=boundaries,
            window_length=int(window_length),
            stride=int(window_stride),
        ),
        max(1, int(batch_size)),
    )
    zero_windows = batch_windows(
        build_trajectory_windows(
            points=zero_points.detach().cpu().float(),
            boundaries=boundaries,
            window_length=int(window_length),
            stride=int(window_stride),
        ),
        max(1, int(batch_size)),
    )
    if len(windows) != len(zero_windows):
        return {"available": False, "reason": "window_count_mismatch"}

    stages: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {
        "point_encoder_output": ([], []),
        "explicit_prior_encoder_scaled": ([], []),
        "pre_context_sum": ([], []),
        "post_local_context": ([], []),
        "segment_context_output": ([], []),
        "pre_shared_encoder_sum": ([], []),
        "shared_embedding": ([], []),
        "head_logits": ([], []),
        "head_probabilities": ([], []),
    }
    original_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            for primary_window, zero_window in zip(windows, zero_windows, strict=True):
                primary_points = primary_window.points.to(device=device)
                ablation_points = zero_window.points.to(device=device)
                padding_mask = primary_window.padding_mask.to(device=device)
                valid = ~padding_mask
                if not bool(valid.any().item()):
                    continue

                def forward_stages(
                    points: torch.Tensor,
                    local_padding_mask: torch.Tensor,
                ) -> dict[str, torch.Tensor]:
                    point_encoded = point_encoder_fn(points)
                    prior_features = prior_features_callable(points)
                    prior_encoded = prior_feature_encoder_fn(prior_features)
                    scale_tensor = getattr(model, "prior_feature_scale", None)
                    scale = (
                        scale_tensor.to(device=device, dtype=prior_encoded.dtype)
                        if isinstance(scale_tensor, torch.Tensor)
                        else prior_encoded.new_tensor(1.0)
                    )
                    explicit_prior = scale * prior_encoded
                    pre_context = point_encoded + explicit_prior
                    if local_context_encoder is not None:
                        local_input = pre_context
                        if positional_callable is not None:
                            local_input = local_input + positional_callable(
                                local_input.shape[1],
                                local_input.device,
                                local_input.dtype,
                            ).unsqueeze(0)
                        post_context = local_context_encoder(
                            local_input, src_key_padding_mask=local_padding_mask
                        )
                    else:
                        post_context = pre_context
                    segment = segment_context_fn(post_context.transpose(1, 2)).transpose(1, 2)
                    pre_shared = post_context + segment
                    shared = shared_prior_encoder_fn(pre_shared)
                    logits = torch.cat(
                        [heads[str(name)](shared) for name in QUERY_LOCAL_UTILITY_HEAD_NAMES],
                        dim=-1,
                    )
                    return {
                        "point_encoder_output": point_encoded,
                        "explicit_prior_encoder_scaled": explicit_prior,
                        "pre_context_sum": pre_context,
                        "post_local_context": post_context,
                        "segment_context_output": segment,
                        "pre_shared_encoder_sum": pre_shared,
                        "shared_embedding": shared,
                        "head_logits": logits,
                        "head_probabilities": torch.sigmoid(logits),
                    }

                primary_stages = forward_stages(primary_points, padding_mask)
                ablation_stages = forward_stages(ablation_points, padding_mask)
                for stage_name, (primary_parts, ablation_parts) in stages.items():
                    primary_parts.append(primary_stages[stage_name][valid])
                    ablation_parts.append(ablation_stages[stage_name][valid])
    finally:
        model.train(original_training)

    stage_rows = {
        stage_name: _stage_tensor_sensitivity(
            primary_parts=primary_parts,
            ablation_parts=ablation_parts,
        )
        for stage_name, (primary_parts, ablation_parts) in stages.items()
    }
    pre_context_delta = _as_numeric(stage_rows, "pre_context_sum", "mean_abs_delta")
    shared_delta = _as_numeric(stage_rows, "shared_embedding", "mean_abs_delta")
    head_delta = _as_numeric(stage_rows, "head_probabilities", "mean_abs_delta")
    return {
        "available": True,
        "diagnostic_only": True,
        "stage_sensitivity": stage_rows,
        "shared_to_pre_context_mean_abs_delta_ratio": (
            None
            if pre_context_delta is None
            else float((shared_delta or 0.0) / max(pre_context_delta, 1e-12))
        ),
        "head_probability_to_pre_context_mean_abs_delta_ratio": (
            None
            if pre_context_delta is None
            else float((head_delta or 0.0) / max(pre_context_delta, 1e-12))
        ),
    }


def _as_numeric(root: dict[str, Any], stage_name: str, key: str) -> float | None:
    row = root.get(stage_name)
    if not isinstance(row, dict):
        return None
    value = row.get(key)
    return float(value) if isinstance(value, int | float) else None
