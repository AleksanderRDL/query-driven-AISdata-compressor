"""Shared helpers for prior-to-head transfer diagnostics."""

from __future__ import annotations

import math
from typing import Any, cast

import torch

from learning.losses import _safe_quantile
from learning.targets.query_local_utility import _rank_correlation


def _head_mlp_transfer(
    head: torch.nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]] | None:
    if not isinstance(head, torch.nn.Sequential):
        return None
    modules = list(head.children())
    linear_positions = [
        idx for idx, module in enumerate(modules) if isinstance(module, torch.nn.Linear)
    ]
    if len(linear_positions) < 2:
        return None
    first_idx = int(linear_positions[0])
    final_idx = int(linear_positions[-1])
    first_linear = cast(torch.nn.Linear, modules[first_idx])
    final_linear = cast(torch.nn.Linear, modules[final_idx])
    first = first_linear(x)
    hidden = first
    for module in modules[first_idx + 1 : final_idx]:
        hidden = module(hidden)
    logit = final_linear(hidden)
    weight_stats = {
        "first_linear_weight_l2": float(first_linear.weight.detach().cpu().float().norm().item()),
        "final_linear_weight_l2": float(final_linear.weight.detach().cpu().float().norm().item()),
        "final_linear_bias_mean": float(final_linear.bias.detach().cpu().float().mean().item())
        if final_linear.bias is not None
        else 0.0,
    }
    return first, hidden, logit, weight_stats


def _head_final_linear_weight(head: torch.nn.Module) -> torch.Tensor | None:
    if not isinstance(head, torch.nn.Sequential):
        return None
    linear_layers = [
        module for module in head.children() if isinstance(module, torch.nn.Linear)
    ]
    if not linear_layers:
        return None
    final_linear = cast(torch.nn.Linear, linear_layers[-1])
    if int(final_linear.weight.shape[0]) != 1:
        return None
    return final_linear.weight.detach().cpu().float().reshape(-1)


def _prior_direction_subset_alignment_summary(
    *,
    hidden_delta: torch.Tensor,
    logit_delta: torch.Tensor,
    projected_delta: torch.Tensor,
    hidden_norm: torch.Tensor,
    target: torch.Tensor,
    probability: torch.Tensor,
    subset_mask: torch.Tensor,
) -> dict[str, Any]:
    """Summarize target/prior-direction alignment for an already finite subset."""
    local_mask = subset_mask.detach().cpu().bool().reshape(-1)
    if int(local_mask.numel()) != int(logit_delta.numel()):
        return {
            "available": False,
            "reason": "slice_mask_shape_mismatch",
            "slice_count": int(local_mask.numel()),
            "row_count": int(logit_delta.numel()),
        }
    count = int(local_mask.sum().item())
    if count < 2:
        return {"available": False, "reason": "insufficient_slice_rows", "count": count}
    local_hidden_delta = hidden_delta[local_mask]
    local_logit_delta = logit_delta[local_mask]
    local_projected_delta = projected_delta[local_mask]
    local_hidden_norm = hidden_norm[local_mask]
    local_target = target[local_mask]
    local_probability = probability[local_mask]
    bce_descent_alignment = (local_target - local_probability) * local_logit_delta
    hidden_delta_l2 = float(local_hidden_delta.norm().item())
    projected_delta_l2 = float(local_projected_delta.norm().item())
    top_minus_bottom_logit_delta = None
    top_minus_bottom_projected_delta = None
    top_minus_bottom_hidden_norm = None
    if count >= 4:
        quartile_count = max(1, math.ceil(0.25 * count))
        top_idx = torch.topk(local_target, k=quartile_count, largest=True).indices
        bottom_idx = torch.topk(local_target, k=quartile_count, largest=False).indices
        top_minus_bottom_logit_delta = float(
            local_logit_delta[top_idx].mean().item()
            - local_logit_delta[bottom_idx].mean().item()
        )
        top_minus_bottom_projected_delta = float(
            local_projected_delta[top_idx].mean().item()
            - local_projected_delta[bottom_idx].mean().item()
        )
        top_minus_bottom_hidden_norm = float(
            local_hidden_norm[top_idx].mean().item()
            - local_hidden_norm[bottom_idx].mean().item()
        )
    return {
        "available": True,
        "diagnostic_only": True,
        "count": count,
        "target_mean": float(local_target.mean().item()),
        "target_std": float(local_target.std(unbiased=False).item()) if count > 1 else 0.0,
        "hidden_delta_l2": hidden_delta_l2,
        "projected_hidden_delta_l2": projected_delta_l2,
        "projected_hidden_delta_l2_to_hidden_delta_l2": (
            projected_delta_l2 / max(hidden_delta_l2, 1e-12)
        ),
        "logit_delta_mean": float(local_logit_delta.mean().item()),
        "logit_delta_abs_mean": float(local_logit_delta.abs().mean().item()),
        "positive_logit_delta_fraction": float(
            (local_logit_delta > 0.0).float().mean().item()
        ),
        "projected_hidden_delta_abs_mean": float(local_projected_delta.abs().mean().item()),
        "target_to_logit_delta_spearman": _rank_correlation(
            local_logit_delta,
            local_target,
            torch.ones_like(local_target, dtype=torch.bool),
        ),
        "target_to_projected_hidden_delta_spearman": _rank_correlation(
            local_projected_delta,
            local_target,
            torch.ones_like(local_target, dtype=torch.bool),
        ),
        "target_to_hidden_delta_norm_spearman": _rank_correlation(
            local_hidden_norm,
            local_target,
            torch.ones_like(local_target, dtype=torch.bool),
        ),
        "target_top_quartile_minus_bottom_quartile_logit_delta": (
            top_minus_bottom_logit_delta
        ),
        "target_top_quartile_minus_bottom_quartile_projected_delta": (
            top_minus_bottom_projected_delta
        ),
        "target_top_quartile_minus_bottom_quartile_hidden_delta_norm": (
            top_minus_bottom_hidden_norm
        ),
        "bce_descent_alignment_mean": float(bce_descent_alignment.mean().item()),
        "bce_descent_alignment_sum": float(bce_descent_alignment.sum().item()),
        "bce_descent_alignment_positive_fraction": float(
            (bce_descent_alignment > 0.0).float().mean().item()
        ),
    }


def _concat_float_parts(parts: list[torch.Tensor], *, flatten: bool = False) -> torch.Tensor:
    if not parts:
        return torch.zeros(0, dtype=torch.float32)
    tensors = [part.detach().cpu().float() for part in parts]
    out = torch.cat(tensors, dim=0)
    return out.reshape(-1) if flatten else out


def _prior_output_layer_alignment_diagnostics(
    *,
    primary_hidden_parts: list[torch.Tensor],
    ablation_hidden_parts: list[torch.Tensor],
    primary_logit_parts: list[torch.Tensor],
    ablation_logit_parts: list[torch.Tensor],
    primary_probability_parts: list[torch.Tensor],
    target_parts: list[torch.Tensor],
    mask_parts: list[torch.Tensor],
    final_weight: torch.Tensor | None,
    slice_mask_parts: dict[str, dict[str, list[torch.Tensor]]] | None = None,
) -> dict[str, Any]:
    """Diagnose whether the final scalar projection uses prior-sensitive hidden deltas."""
    if final_weight is None:
        return {"available": False, "reason": "missing_final_linear_weight"}
    primary_hidden = _concat_float_parts(primary_hidden_parts)
    ablation_hidden = _concat_float_parts(ablation_hidden_parts)
    primary_logit = _concat_float_parts(primary_logit_parts, flatten=True)
    ablation_logit = _concat_float_parts(ablation_logit_parts, flatten=True)
    primary_probability = _concat_float_parts(primary_probability_parts, flatten=True)
    target = _concat_float_parts(target_parts, flatten=True).clamp(0.0, 1.0)
    mask = (
        torch.cat([part.detach().cpu().bool().reshape(-1) for part in mask_parts], dim=0)
        if mask_parts
        else torch.zeros(0, dtype=torch.bool)
    )
    if (
        primary_hidden.shape != ablation_hidden.shape
        or primary_hidden.ndim != 2
        or int(primary_hidden.numel()) == 0
    ):
        return {"available": False, "reason": "hidden_shape_mismatch_or_empty"}
    row_count = int(primary_hidden.shape[0])
    if (
        int(primary_logit.numel()) != row_count
        or int(ablation_logit.numel()) != row_count
        or int(primary_probability.numel()) != row_count
        or int(target.numel()) != row_count
        or int(mask.numel()) != row_count
    ):
        return {
            "available": False,
            "reason": "aligned_target_or_logit_shape_mismatch",
            "hidden_row_count": row_count,
            "primary_logit_count": int(primary_logit.numel()),
            "target_count": int(target.numel()),
            "mask_count": int(mask.numel()),
        }
    weight = final_weight.detach().cpu().float().reshape(-1)
    if int(weight.numel()) != int(primary_hidden.shape[1]):
        return {
            "available": False,
            "reason": "final_weight_hidden_dim_mismatch",
            "hidden_dim": int(primary_hidden.shape[1]),
            "weight_dim": int(weight.numel()),
        }
    finite = (
        torch.isfinite(primary_hidden).all(dim=1)
        & torch.isfinite(ablation_hidden).all(dim=1)
        & torch.isfinite(primary_logit)
        & torch.isfinite(ablation_logit)
        & torch.isfinite(primary_probability)
        & torch.isfinite(target)
        & mask
    )
    if not bool(finite.any().item()):
        return {"available": False, "reason": "no_finite_masked_rows"}

    hidden_delta = primary_hidden[finite] - ablation_hidden[finite]
    logit_delta = primary_logit[finite] - ablation_logit[finite]
    probability = primary_probability[finite].clamp(1e-6, 1.0 - 1e-6)
    local_target = target[finite]
    projected_delta = hidden_delta @ weight
    hidden_norm = hidden_delta.norm(dim=1)
    weight_norm = weight.norm()
    nonzero_direction = hidden_norm > 1e-12
    if bool(nonzero_direction.any().item()) and float(weight_norm.item()) > 0.0:
        cosine = projected_delta[nonzero_direction] / (
            hidden_norm[nonzero_direction] * weight_norm.clamp_min(1e-12)
        )
    else:
        cosine = torch.zeros(0, dtype=torch.float32)
    hidden_delta_l2 = float(hidden_delta.norm().item())
    projected_delta_l2 = float(projected_delta.norm().item())
    logit_delta_l2 = float(logit_delta.norm().item())
    hidden_delta_norm = hidden_norm[nonzero_direction]
    bce_descent_alignment = (local_target - probability) * logit_delta
    result: dict[str, Any] = {
        "available": True,
        "diagnostic_only": True,
        "valid_aligned_point_count": int(finite.sum().item()),
        "nonzero_hidden_delta_count": int(nonzero_direction.sum().item()),
        "final_weight_l2": float(weight_norm.item()),
        "hidden_delta_l2": hidden_delta_l2,
        "projected_hidden_delta_l2": projected_delta_l2,
        "logit_delta_l2": logit_delta_l2,
        "projected_hidden_delta_l2_to_hidden_delta_l2": (
            projected_delta_l2 / max(hidden_delta_l2, 1e-12)
        ),
        "logit_delta_l2_to_projected_hidden_delta_l2": (
            logit_delta_l2 / max(projected_delta_l2, 1e-12)
        ),
        "final_weight_to_hidden_delta_signed_cosine_mean": float(cosine.mean().item())
        if int(cosine.numel())
        else None,
        "final_weight_to_hidden_delta_abs_cosine_mean": float(cosine.abs().mean().item())
        if int(cosine.numel())
        else None,
        "final_weight_to_hidden_delta_abs_cosine_p95": float(
            _safe_quantile(cosine.abs(), 0.95).item()
        )
        if int(cosine.numel())
        else None,
        "hidden_delta_norm_mean": float(hidden_delta_norm.mean().item())
        if int(hidden_delta_norm.numel())
        else 0.0,
        "logit_delta_mean": float(logit_delta.mean().item()),
        "logit_delta_abs_mean": float(logit_delta.abs().mean().item()),
        "projected_hidden_delta_abs_mean": float(projected_delta.abs().mean().item()),
        "target_to_logit_delta_spearman": _rank_correlation(
            logit_delta,
            local_target,
            torch.ones_like(local_target, dtype=torch.bool),
        ),
        "target_to_projected_hidden_delta_spearman": _rank_correlation(
            projected_delta,
            local_target,
            torch.ones_like(local_target, dtype=torch.bool),
        ),
        "target_to_hidden_delta_norm_spearman": _rank_correlation(
            hidden_norm,
            local_target,
            torch.ones_like(local_target, dtype=torch.bool),
        ),
        "bce_descent_alignment_mean": float(bce_descent_alignment.mean().item()),
        "bce_descent_alignment_sum": float(bce_descent_alignment.sum().item()),
        "bce_descent_alignment_positive_fraction": float(
            (bce_descent_alignment > 0.0).float().mean().item()
        ),
    }
    if slice_mask_parts:
        slice_alignment: dict[str, dict[str, Any]] = {}
        for group_name, group_parts in slice_mask_parts.items():
            group_rows: dict[str, Any] = {}
            for slice_name, parts in group_parts.items():
                if not parts:
                    continue
                slice_mask = torch.cat(
                    [part.detach().cpu().bool().reshape(-1) for part in parts],
                    dim=0,
                )
                if int(slice_mask.numel()) != row_count:
                    group_rows[str(slice_name)] = {
                        "available": False,
                        "reason": "slice_mask_shape_mismatch",
                        "slice_count": int(slice_mask.numel()),
                        "row_count": row_count,
                    }
                    continue
                group_rows[str(slice_name)] = _prior_direction_subset_alignment_summary(
                    hidden_delta=hidden_delta,
                    logit_delta=logit_delta,
                    projected_delta=projected_delta,
                    hidden_norm=hidden_norm,
                    target=local_target,
                    probability=probability,
                    subset_mask=slice_mask[finite],
                )
            if group_rows:
                slice_alignment[str(group_name)] = group_rows
        result["slice_alignment"] = {
            "available": bool(slice_alignment),
            "diagnostic_only": True,
            "groups": slice_alignment,
        }
    return result


def _loss_gradient_alignment_summary(
    *,
    descent_alignment_parts: list[torch.Tensor],
    gradient_parts: list[torch.Tensor],
    logit_delta_parts: list[torch.Tensor],
    loss_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not descent_alignment_parts or not gradient_parts or not logit_delta_parts:
        return {"available": False, "reason": "missing_loss_gradient_parts"}
    descent_alignment = _concat_float_parts(descent_alignment_parts, flatten=True)
    gradient = _concat_float_parts(gradient_parts, flatten=True)
    logit_delta = _concat_float_parts(logit_delta_parts, flatten=True)
    if (
        int(descent_alignment.numel()) == 0
        or int(gradient.numel()) != int(descent_alignment.numel())
        or int(logit_delta.numel()) != int(descent_alignment.numel())
    ):
        return {
            "available": False,
            "reason": "loss_gradient_shape_mismatch_or_empty",
            "descent_alignment_count": int(descent_alignment.numel()),
            "gradient_count": int(gradient.numel()),
            "logit_delta_count": int(logit_delta.numel()),
        }
    finite = (
        torch.isfinite(descent_alignment)
        & torch.isfinite(gradient)
        & torch.isfinite(logit_delta)
    )
    if not bool(finite.any().item()):
        return {"available": False, "reason": "no_finite_loss_gradient_rows"}
    local_alignment = descent_alignment[finite]
    local_gradient = gradient[finite]
    local_logit_delta = logit_delta[finite]
    return {
        "available": True,
        "diagnostic_only": True,
        "loss_scope": "configured_factorized_loss_window_segment_proxy",
        "loss_config": loss_config or {},
        "aligned_point_count": int(finite.sum().item()),
        "descent_alignment_mean": float(local_alignment.mean().item()),
        "descent_alignment_sum": float(local_alignment.sum().item()),
        "descent_alignment_positive_fraction": float(
            (local_alignment > 0.0).float().mean().item()
        ),
        "logit_gradient_abs_mean": float(local_gradient.abs().mean().item()),
        "logit_gradient_abs_p95": float(_safe_quantile(local_gradient.abs(), 0.95).item()),
        "logit_delta_abs_mean": float(local_logit_delta.abs().mean().item()),
    }


def _ratio_from_summaries(
    numerator: dict[str, Any],
    denominator: dict[str, Any],
    field: str,
) -> float | None:
    left = numerator.get(field)
    right = denominator.get(field)
    if not isinstance(left, int | float) or not isinstance(right, int | float) or right == 0.0:
        return None
    return float(left) / max(float(right), 1e-12)


def _classify_head_transfer(
    *,
    shared: dict[str, Any],
    first: dict[str, Any],
    hidden: dict[str, Any],
    logit: dict[str, Any],
    probability: dict[str, Any],
    sigmoid_derivative_mean: float | None,
) -> str:
    shared_delta = shared.get("mean_abs_delta")
    probability_delta = probability.get("mean_abs_delta")
    if not isinstance(shared_delta, int | float) or float(shared_delta) <= 1e-8:
        return "prior_sensitive_shared_direction_missing"
    first_ratio = _ratio_from_summaries(first, shared, "delta_l2")
    hidden_ratio = _ratio_from_summaries(hidden, first, "delta_l2")
    logit_ratio = _ratio_from_summaries(logit, hidden, "delta_l2")
    probability_ratio = _ratio_from_summaries(probability, logit, "mean_abs_delta")
    if first_ratio is not None and first_ratio < 0.20:
        return "first_layer_suppresses_prior_direction"
    if hidden_ratio is not None and hidden_ratio < 0.20:
        return "activation_suppresses_prior_direction"
    if logit_ratio is not None and logit_ratio < 0.20:
        return "output_layer_suppresses_prior_direction"
    if (
        probability_ratio is not None
        and probability_ratio < 0.05
        and sigmoid_derivative_mean is not None
        and sigmoid_derivative_mean < 0.05
    ):
        return "sigmoid_base_rate_saturation_suppresses_logit_delta"
    if isinstance(probability_delta, int | float) and float(probability_delta) < 1e-4:
        return "head_probability_invariant_after_mlp"
    return "prior_direction_reaches_head_output"


def _classify_prior_channel_output_alignment(row: dict[str, Any]) -> str:
    if not bool(row.get("available")):
        reason = row.get("reason")
        return f"unavailable_{reason}" if isinstance(reason, str) else "unavailable"
    spearman = row.get("target_to_logit_delta_spearman")
    if not isinstance(spearman, int | float) or not math.isfinite(float(spearman)):
        return "rank_alignment_unavailable"
    value = float(spearman)
    if value < -0.05:
        return "wrong_way"
    if value > 0.05:
        return "target_aligned"
    return "weak_or_flat"


def _summarize_prior_channel_direction_decomposition(
    channel_rows: dict[str, Any],
) -> dict[str, Any]:
    classification_counts: dict[str, int] = {}
    by_head: dict[str, Any] = {}
    for channel_name, channel_row in channel_rows.items():
        per_head = _as_dict_for_diagnostics(channel_row.get("per_head"))
        for head_name, head_row_raw in per_head.items():
            head_row = _as_dict_for_diagnostics(head_row_raw)
            classification = str(head_row.get("classification", "unavailable"))
            classification_counts[classification] = (
                int(classification_counts.get(classification, 0)) + 1
            )
            alignment = _as_dict_for_diagnostics(head_row.get("output_layer_alignment"))
            spearman = alignment.get("target_to_logit_delta_spearman")
            head_summary = by_head.setdefault(
                str(head_name),
                {
                    "channel_count": 0,
                    "target_aligned_channels": [],
                    "wrong_way_channels": [],
                    "weak_or_flat_channels": [],
                    "rank_alignment_unavailable_channels": [],
                    "min_target_to_logit_delta_spearman": None,
                    "max_target_to_logit_delta_spearman": None,
                    "strongest_aligned_channel": None,
                    "strongest_wrong_way_channel": None,
                },
            )
            head_summary["channel_count"] = int(head_summary["channel_count"]) + 1
            if classification == "target_aligned":
                head_summary["target_aligned_channels"].append(str(channel_name))
            elif classification == "wrong_way":
                head_summary["wrong_way_channels"].append(str(channel_name))
            elif classification == "weak_or_flat":
                head_summary["weak_or_flat_channels"].append(str(channel_name))
            else:
                head_summary["rank_alignment_unavailable_channels"].append(str(channel_name))
            if isinstance(spearman, int | float) and math.isfinite(float(spearman)):
                value = float(spearman)
                current_min = head_summary["min_target_to_logit_delta_spearman"]
                current_max = head_summary["max_target_to_logit_delta_spearman"]
                if current_min is None or value < float(current_min):
                    head_summary["min_target_to_logit_delta_spearman"] = value
                    head_summary["strongest_wrong_way_channel"] = {
                        "channel": str(channel_name),
                        "value": value,
                    }
                if current_max is None or value > float(current_max):
                    head_summary["max_target_to_logit_delta_spearman"] = value
                    head_summary["strongest_aligned_channel"] = {
                        "channel": str(channel_name),
                        "value": value,
                    }
    return {
        "available": bool(channel_rows),
        "diagnostic_only": True,
        "classification_thresholds": {
            "wrong_way": "target_to_logit_delta_spearman < -0.05",
            "target_aligned": "target_to_logit_delta_spearman > 0.05",
            "weak_or_flat": "absolute spearman <= 0.05",
        },
        "classification_counts": classification_counts,
        "by_head": by_head,
    }


def _as_dict_for_diagnostics(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
