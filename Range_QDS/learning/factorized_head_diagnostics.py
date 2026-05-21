"""Factorized QueryLocalUtility head diagnostics and initialization helpers."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn.functional as F

from learning.fit_diagnostics import _discriminative_sample, _kendall_tau
from learning.losses import _safe_quantile
from learning.optimization_epoch import _factorized_query_local_utility_loss
from learning.query_prior_fields import QUERY_PRIOR_FIELD_NAMES
from learning.targets.query_local_utility import (
    QUERY_LOCAL_UTILITY_FINAL_LABEL_FORMULA,
    QUERY_LOCAL_UTILITY_HEAD_NAMES,
    _rank_correlation,
    _topk_overlap_and_mass_recall,
    query_local_utility_point_score,
)
from learning.targets.query_local_utility_family import (
    DIAGNOSTIC_TRAINABILITY_FOCUS_FAMILIES,
    FAMILY_TRAINABILITY_GROUP_KEYS,
    _range_query_family_evidence,
)
from learning.trajectory_batching import batch_windows, build_trajectory_windows


def _initialize_factorized_head_output_biases_from_targets(
    model: torch.nn.Module,
    *,
    head_targets: torch.Tensor | None,
    head_mask: torch.Tensor | None,
    min_probability: float = 1e-4,
) -> dict[str, Any]:
    """Center factorized sigmoid heads on their empirical training base rates."""
    head_names = tuple(str(name) for name in getattr(model, "head_names", ()))
    heads = getattr(model, "heads", None)
    if head_targets is None or head_mask is None or not head_names or heads is None:
        return {"available": False, "reason": "missing_factorized_heads_or_targets"}
    if head_targets.shape != head_mask.shape or int(head_targets.shape[-1]) != len(head_names):
        return {"available": False, "reason": "shape_mismatch"}
    rows: dict[str, dict[str, float | int | bool | None]] = {}
    clamp = max(1e-8, min(0.49, float(min_probability)))
    with torch.no_grad():
        for head_idx, head_name in enumerate(head_names):
            try:
                head_module = heads[head_name]
            except KeyError, TypeError:
                rows[head_name] = {
                    "initialized": False,
                    "target_mean": None,
                    "bias": None,
                    "valid_count": 0,
                }
                continue
            linear_layers = [
                module for module in head_module.modules() if isinstance(module, torch.nn.Linear)
            ]
            if not linear_layers or linear_layers[-1].bias is None:
                rows[head_name] = {
                    "initialized": False,
                    "target_mean": None,
                    "bias": None,
                    "valid_count": 0,
                }
                continue
            valid = head_mask[..., head_idx].to(dtype=torch.bool)
            valid_count = int(valid.sum().item())
            if valid_count <= 0:
                rows[head_name] = {
                    "initialized": False,
                    "target_mean": None,
                    "bias": None,
                    "valid_count": 0,
                }
                continue
            target_mean = float(head_targets[..., head_idx][valid].float().mean().item())
            probability = min(1.0 - clamp, max(clamp, target_mean))
            bias_value = math.log(probability / (1.0 - probability))
            linear_layers[-1].bias.fill_(float(bias_value))
            rows[head_name] = {
                "initialized": True,
                "target_mean": float(target_mean),
                "clamped_probability": float(probability),
                "bias": float(bias_value),
                "valid_count": int(valid_count),
            }
    return {
        "available": True,
        "method": "empirical_target_mean_logit_output_bias",
        "min_probability": float(clamp),
        "heads": rows,
    }


def _segment_head_fit_diagnostics(
    *,
    head_logits: torch.Tensor | None,
    factorized_targets: torch.Tensor | None,
    factorized_mask: torch.Tensor | None,
    canonical_segment_ids: torch.Tensor | None = None,
    seed: int,
) -> dict[str, Any]:
    """Summarize training-set fit for the segment-budget auxiliary head."""
    if head_logits is None or factorized_targets is None or factorized_mask is None:
        return {"segment_head_diagnostics_available": False}
    try:
        segment_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("segment_budget_target")
    except ValueError:
        return {
            "segment_head_diagnostics_available": False,
            "reason": "segment_budget_head_missing",
        }
    if (
        int(head_logits.shape[0]) != int(factorized_targets.shape[0])
        or int(head_logits.shape[-1]) <= segment_idx
    ):
        return {"segment_head_diagnostics_available": False, "reason": "shape_mismatch"}
    valid = factorized_mask[:, segment_idx].detach().cpu().bool()
    targets = factorized_targets[:, segment_idx].detach().cpu().float().clamp(0.0, 1.0)
    scores = torch.sigmoid(head_logits[:, segment_idx].detach().cpu().float())
    if not bool(valid.any().item()):
        return {"segment_head_diagnostics_available": False, "reason": "no_valid_segment_targets"}
    generator = torch.Generator().manual_seed(int(seed) + 811)
    sampled_scores, sampled_targets = _discriminative_sample(
        scores[valid],
        targets[valid],
        n_each=200,
        generator=generator,
    )
    tau = _kendall_tau(sampled_scores, sampled_targets)
    valid_scores = scores[valid]
    valid_targets = targets[valid]
    k = max(1, math.ceil(0.05 * int(valid_scores.numel())))
    selected = torch.topk(valid_scores, k=k, largest=True).indices
    ideal = torch.topk(valid_targets, k=k, largest=True).indices
    selected_mass = float(valid_targets[selected].sum().item())
    ideal_mass = float(valid_targets[ideal].sum().item())
    diagnostics: dict[str, Any] = {
        "segment_head_diagnostics_available": True,
        "segment_head_point_tau": float(tau),
        "segment_head_point_topk_mass_recall_at_5_percent": float(
            selected_mass / max(ideal_mass, 1e-12)
        ),
        "segment_head_valid_point_count": int(valid_scores.numel()),
        "segment_head_target_mass": float(valid_targets.sum().item()),
    }
    if canonical_segment_ids is None:
        diagnostics["segment_head_canonical_segment_diagnostics_available"] = False
        diagnostics["segment_head_diagnostics_note"] = (
            "point_level_only_missing_canonical_segment_ids"
        )
        diagnostics["segment_head_tau"] = diagnostics["segment_head_point_tau"]
        diagnostics["segment_head_topk_mass_recall_at_5_percent"] = diagnostics[
            "segment_head_point_topk_mass_recall_at_5_percent"
        ]
        return diagnostics

    segment_ids = canonical_segment_ids.detach().cpu().long()
    if int(segment_ids.numel()) != int(valid.numel()):
        diagnostics["segment_head_canonical_segment_diagnostics_available"] = False
        diagnostics["segment_head_canonical_segment_reason"] = "segment_id_shape_mismatch"
        diagnostics["segment_head_tau"] = diagnostics["segment_head_point_tau"]
        diagnostics["segment_head_topk_mass_recall_at_5_percent"] = diagnostics[
            "segment_head_point_topk_mass_recall_at_5_percent"
        ]
        return diagnostics

    valid_segment_mask = valid & (segment_ids >= 0)
    if not bool(valid_segment_mask.any().item()):
        diagnostics["segment_head_canonical_segment_diagnostics_available"] = False
        diagnostics["segment_head_canonical_segment_reason"] = "no_valid_canonical_segments"
        diagnostics["segment_head_tau"] = diagnostics["segment_head_point_tau"]
        diagnostics["segment_head_topk_mass_recall_at_5_percent"] = diagnostics[
            "segment_head_point_topk_mass_recall_at_5_percent"
        ]
        return diagnostics

    pooled_scores: list[torch.Tensor] = []
    pooled_targets: list[torch.Tensor] = []
    for segment_id in torch.unique(segment_ids[valid_segment_mask], sorted=True).tolist():
        local = valid_segment_mask & (segment_ids == int(segment_id))
        if bool(local.any().item()):
            pooled_scores.append(scores[local].mean())
            pooled_targets.append(targets[local].mean())
    if pooled_scores:
        segment_scores = torch.stack(pooled_scores)
        segment_targets = torch.stack(pooled_targets)
        segment_sampled_scores, segment_sampled_targets = _discriminative_sample(
            segment_scores,
            segment_targets,
            n_each=200,
            generator=generator,
        )
        segment_k = max(1, math.ceil(0.05 * int(segment_scores.numel())))
        segment_selected = torch.topk(segment_scores, k=segment_k, largest=True).indices
        segment_ideal = torch.topk(segment_targets, k=segment_k, largest=True).indices
        segment_selected_mass = float(segment_targets[segment_selected].sum().item())
        segment_ideal_mass = float(segment_targets[segment_ideal].sum().item())
        segment_tau = float(_kendall_tau(segment_sampled_scores, segment_sampled_targets))
        segment_topk_recall = float(segment_selected_mass / max(segment_ideal_mass, 1e-12))
        diagnostics.update(
            {
                "segment_head_canonical_segment_diagnostics_available": True,
                "segment_head_canonical_segment_count": int(segment_scores.numel()),
                "segment_head_canonical_segment_tau": segment_tau,
                "segment_head_canonical_segment_topk_mass_recall_at_5_percent": (
                    segment_topk_recall
                ),
                "segment_head_tau": segment_tau,
                "segment_head_topk_mass_recall_at_5_percent": segment_topk_recall,
            }
        )
    return diagnostics


def _behavior_head_training_signal_diagnostics(
    *,
    head_logits: torch.Tensor | None,
    factorized_targets: torch.Tensor | None,
    factorized_mask: torch.Tensor | None,
    boundaries: list[tuple[int, int]] | None,
    behavior_rank_loss_weight: float,
    top_fraction: float = 0.05,
    min_target_gap: float = 0.05,
) -> dict[str, Any]:
    """Summarize behavior-head loss pressure versus a constant-bias baseline."""
    if head_logits is None or factorized_targets is None or factorized_mask is None:
        return {"behavior_head_training_signal_available": False}
    if head_logits.shape != factorized_targets.shape or factorized_mask.shape != head_logits.shape:
        return {
            "behavior_head_training_signal_available": False,
            "reason": "shape_mismatch",
        }
    try:
        behavior_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("conditional_behavior_utility")
    except ValueError:
        return {
            "behavior_head_training_signal_available": False,
            "reason": "behavior_head_missing",
        }
    if int(head_logits.shape[-1]) <= behavior_idx:
        return {
            "behavior_head_training_signal_available": False,
            "reason": "behavior_head_index_out_of_range",
        }

    logits = head_logits[:, behavior_idx].detach().cpu().float()
    targets = factorized_targets[:, behavior_idx].detach().cpu().float().clamp(0.0, 1.0)
    valid = factorized_mask[:, behavior_idx].detach().cpu().bool()
    if not bool(valid.any().item()):
        return {
            "behavior_head_training_signal_available": False,
            "reason": "no_valid_behavior_targets",
        }

    valid_logits = logits[valid]
    valid_targets = targets[valid]
    target_mean = valid_targets.mean().clamp(1e-5, 1.0 - 1e-5)
    bias_logit = torch.logit(target_mean)
    bias_logits = torch.full_like(valid_logits, float(bias_logit.item()))
    bce_loss = F.binary_cross_entropy_with_logits(valid_logits, valid_targets, reduction="mean")
    bias_bce_loss = F.binary_cross_entropy_with_logits(
        bias_logits,
        valid_targets,
        reduction="mean",
    )
    probabilities = torch.sigmoid(valid_logits)
    target_std = (
        float(valid_targets.std(unbiased=False).item()) if int(valid_targets.numel()) > 1 else 0.0
    )
    prediction_std = (
        float(probabilities.std(unbiased=False).item()) if int(probabilities.numel()) > 1 else 0.0
    )

    spans: list[float] = []
    pair_losses: list[torch.Tensor] = []
    bias_pair_losses: list[torch.Tensor] = []
    pair_gaps: list[torch.Tensor] = []
    correct_pairs = 0
    tied_pairs = 0
    pair_count = 0
    rows_considered = 0
    rows_with_pairs = 0
    row_slices = (
        [(0, int(logits.numel()))]
        if boundaries is None
        else [(int(start), int(end)) for start, end in boundaries]
    )
    fraction = min(1.0, max(0.0, float(top_fraction)))
    min_gap = max(0.0, float(min_target_gap))
    for start, end in row_slices:
        start_i = max(0, int(start))
        end_i = min(int(logits.numel()), int(end))
        if end_i <= start_i:
            continue
        local_valid = torch.where(valid[start_i:end_i])[0] + start_i
        valid_count = int(local_valid.numel())
        if valid_count < 2:
            continue
        rows_considered += 1
        local_targets = targets[local_valid]
        target_span = float((local_targets.max() - local_targets.min()).item())
        spans.append(target_span)
        if target_span <= min_gap:
            continue
        local_logits = logits[local_valid]
        top_count = max(1, math.ceil(fraction * valid_count))
        top_positions = torch.topk(local_targets, k=top_count, largest=True).indices
        top_targets = local_targets[top_positions]
        top_logits = local_logits[top_positions]
        target_gap = top_targets.unsqueeze(1) - local_targets.unsqueeze(0)
        pair_mask = target_gap > min_gap
        if not bool(pair_mask.any().item()):
            continue
        rows_with_pairs += 1
        logit_gap = top_logits.unsqueeze(1) - local_logits.unsqueeze(0)
        selected_logit_gaps = logit_gap[pair_mask]
        selected_target_gaps = target_gap[pair_mask]
        pair_losses.append(F.softplus(-selected_logit_gaps) * selected_target_gaps)
        bias_pair_losses.append(
            F.softplus(torch.zeros_like(selected_logit_gaps)) * selected_target_gaps
        )
        pair_gaps.append(selected_target_gaps)
        pair_count += int(selected_target_gaps.numel())
        correct_pairs += int((selected_logit_gaps > 0.0).sum().item())
        tied_pairs += int((selected_logit_gaps == 0.0).sum().item())

    if pair_losses:
        loss_values = torch.cat(pair_losses)
        bias_loss_values = torch.cat(bias_pair_losses)
        gap_values = torch.cat(pair_gaps)
        rank_loss = float(loss_values.mean().item())
        bias_rank_loss = float(bias_loss_values.mean().item())
        mean_target_gap = float(gap_values.mean().item())
        max_target_gap = float(gap_values.max().item())
    else:
        rank_loss = 0.0
        bias_rank_loss = 0.0
        mean_target_gap = 0.0
        max_target_gap = 0.0

    rank_improvement = float(bias_rank_loss - rank_loss)
    bce_improvement = float(bias_bce_loss.item() - bce_loss.item())
    pair_accuracy = float(correct_pairs / pair_count) if pair_count > 0 else None
    pair_tie_fraction = float(tied_pairs / pair_count) if pair_count > 0 else None
    std_ratio = float(prediction_std / max(target_std, 1e-12))
    weighted_rank_loss = max(0.0, float(behavior_rank_loss_weight)) * rank_loss
    if pair_count <= 0:
        category = "behavior_rank_pressure_missing"
    elif std_ratio < 0.10 and rank_improvement <= 1e-4 and bce_improvement <= 1e-4:
        category = "rank_pressure_available_but_head_near_bias"
    elif rank_improvement <= 1e-4:
        category = "rank_pressure_available_but_not_improved_over_bias"
    elif std_ratio < 0.10:
        category = "rank_pressure_improves_but_prediction_still_flat"
    else:
        category = "behavior_head_training_signal_partially_learned"
    return {
        "behavior_head_training_signal_available": True,
        "head": "conditional_behavior_utility",
        "diagnostic_only": True,
        "valid_point_count": int(valid_targets.numel()),
        "positive_target_count": int((valid_targets > 0.0).sum().item()),
        "positive_target_fraction": float((valid_targets > 0.0).float().mean().item()),
        "target_mean": float(target_mean.item()),
        "target_std": target_std,
        "prediction_mean": float(probabilities.mean().item()),
        "prediction_std": prediction_std,
        "prediction_std_to_target_std": std_ratio,
        "bias_baseline_logit": float(bias_logit.item()),
        "bias_baseline_probability": float(target_mean.item()),
        "bce_loss": float(bce_loss.item()),
        "bias_baseline_bce_loss": float(bias_bce_loss.item()),
        "bce_improvement_vs_bias": bce_improvement,
        "behavior_rank_loss_weight": float(behavior_rank_loss_weight),
        "rank_rows_considered": int(rows_considered),
        "rank_rows_with_pairs": int(rows_with_pairs),
        "rank_rows_with_pairs_fraction": float(rows_with_pairs / max(1, rows_considered)),
        "rank_pair_count": int(pair_count),
        "rank_pair_accuracy": pair_accuracy,
        "rank_pair_tie_fraction": pair_tie_fraction,
        "rank_pair_mean_target_gap": mean_target_gap,
        "rank_pair_max_target_gap": max_target_gap,
        "row_target_span_mean": float(sum(spans) / len(spans)) if spans else 0.0,
        "row_target_span_max": float(max(spans)) if spans else 0.0,
        "rank_loss": rank_loss,
        "bias_baseline_rank_loss": bias_rank_loss,
        "rank_loss_improvement_vs_bias": rank_improvement,
        "weighted_behavior_rank_loss": weighted_rank_loss,
        "weighted_rank_loss_to_behavior_bce_ratio": float(
            weighted_rank_loss / max(float(bce_loss.item()), 1e-12)
        ),
        "classification": category,
    }


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


def _prior_to_head_transfer_sensitivity_diagnostics(
    *,
    model: torch.nn.Module | None,
    norm_points: torch.Tensor,
    factorized_targets: torch.Tensor,
    factorized_mask: torch.Tensor,
    boundaries: list[tuple[int, int]] | None,
    window_length: int,
    window_stride: int,
    batch_size: int,
    prior_dim: int,
    raw_points: torch.Tensor | None = None,
    typed_queries: list[dict[str, Any]] | None = None,
    segment_budget_head_weight: float = 0.10,
    segment_level_loss_weight: float = 0.25,
    behavior_rank_loss_weight: float = 0.25,
    sparse_head_rank_loss_weight: float = 0.0,
    sparse_head_bce_target_mode: str = "raw",
) -> dict[str, Any]:
    """Diagnose how prior-sensitive shared directions pass through each head MLP."""
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
    target_cpu = factorized_targets.detach().cpu().float().clamp(0.0, 1.0)
    mask_cpu = factorized_mask.detach().cpu().bool()
    if target_cpu.shape != mask_cpu.shape:
        return {"available": False, "reason": "target_mask_shape_mismatch"}
    family_slice_masks: dict[str, dict[str, torch.Tensor]] = {}
    if raw_points is not None and typed_queries is not None:
        raw_points_cpu = raw_points.detach().cpu().float()
        if int(raw_points_cpu.shape[0]) == int(norm_points.shape[0]):
            range_queries = [
                query for query in typed_queries if str(query.get("type", "")).lower() == "range"
            ]
            if range_queries:
                family_evidence = _range_query_family_evidence(
                    points=raw_points_cpu,
                    boundaries=boundaries,
                    range_queries=range_queries,
                    group_keys=FAMILY_TRAINABILITY_GROUP_KEYS,
                )
                for group_key, family_rows in family_evidence.items():
                    group_masks: dict[str, torch.Tensor] = {}
                    for family_name, row in family_rows.items():
                        query_hit_probability = row.get("query_hit_probability")
                        if isinstance(query_hit_probability, torch.Tensor):
                            group_masks[str(family_name)] = (
                                query_hit_probability.detach().cpu().float() > 0.0
                            )
                    if group_masks:
                        family_slice_masks[str(group_key)] = group_masks

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
    device = next(model.parameters(), torch.empty(0)).device
    per_head_parts: dict[str, dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]]] = {
        str(name): {
            "shared_embedding": ([], []),
            "first_linear": ([], []),
            "hidden_activation": ([], []),
            "logit": ([], []),
            "probability": ([], []),
        }
        for name in QUERY_LOCAL_UTILITY_HEAD_NAMES
        if str(name) in heads
    }
    prior_channel_names = [
        str(QUERY_PRIOR_FIELD_NAMES[idx])
        if idx < len(QUERY_PRIOR_FIELD_NAMES)
        else f"prior_channel_{idx}"
        for idx in range(int(prior_dim))
    ]
    per_channel_head_parts: dict[
        str,
        dict[str, dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]]],
    ] = {
        channel_name: {
            head_name: {
                "hidden_activation": ([], []),
                "logit": ([], []),
                "probability": ([], []),
            }
            for head_name in per_head_parts
        }
        for channel_name in prior_channel_names
    }
    head_index_by_name = {
        str(name): int(idx) for idx, name in enumerate(QUERY_LOCAL_UTILITY_HEAD_NAMES)
    }
    per_head_target_parts: dict[str, list[torch.Tensor]] = {
        head_name: [] for head_name in per_head_parts
    }
    per_head_mask_parts: dict[str, list[torch.Tensor]] = {
        head_name: [] for head_name in per_head_parts
    }
    per_head_loss_gradient_parts: dict[str, dict[str, list[torch.Tensor]]] = {
        head_name: {
            "descent_alignment": [],
            "gradient": [],
            "logit_delta": [],
        }
        for head_name in per_head_parts
    }
    window_slice_labels = {
        0: "window_start",
        1: "window_middle",
        2: "window_end",
    }
    per_head_slice_mask_parts: dict[str, dict[str, dict[str, list[torch.Tensor]]]] = {
        head_name: {
            "window_slice": {label: [] for label in window_slice_labels.values()},
            **{
                group_name: {family_name: [] for family_name in family_masks}
                for group_name, family_masks in family_slice_masks.items()
            },
        }
        for head_name in per_head_parts
    }
    head_final_weights: dict[str, torch.Tensor] = {
        head_name: weight
        for head_name in per_head_parts
        if (weight := _head_final_linear_weight(heads[head_name])) is not None
    }
    head_weight_stats: dict[str, dict[str, float]] = {}
    loss_config: dict[str, Any] = {
        "segment_budget_head_weight": float(segment_budget_head_weight),
        "segment_level_loss_weight": float(segment_level_loss_weight),
        "behavior_rank_loss_weight": float(behavior_rank_loss_weight),
        "sparse_head_rank_loss_weight": float(sparse_head_rank_loss_weight),
        "sparse_head_bce_target_mode": str(sparse_head_bce_target_mode),
        "segment_id_scope": "window_order_fixed_chunks",
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
                valid_cpu = valid.detach().cpu()
                global_indices_cpu = primary_window.global_indices.detach().cpu().long()
                valid_global = global_indices_cpu[valid_cpu]
                valid_global = valid_global.detach().cpu().long().reshape(-1)
                global_indices_usable = bool(
                    int(valid_global.numel())
                    and bool((valid_global >= 0).all().item())
                    and bool((valid_global < int(target_cpu.shape[0])).all().item())
                )
                valid_counts = valid_cpu.sum(dim=1).clamp(min=1)
                valid_ranks = torch.cumsum(valid_cpu.to(dtype=torch.long), dim=1) - 1
                valid_fractions = (
                    (valid_ranks.to(dtype=torch.float32) + 0.5)
                    / valid_counts.unsqueeze(1).to(dtype=torch.float32)
                )
                window_bucket = torch.where(
                    valid_fractions < (1.0 / 3.0),
                    torch.zeros_like(valid_ranks),
                    torch.where(
                        valid_fractions < (2.0 / 3.0),
                        torch.ones_like(valid_ranks),
                        torch.full_like(valid_ranks, 2),
                    ),
                )
                valid_window_bucket = window_bucket[valid_cpu].detach().cpu().long().reshape(-1)

                def shared_embedding(
                    points: torch.Tensor,
                    local_padding_mask: torch.Tensor,
                ) -> torch.Tensor:
                    point_encoded = point_encoder_fn(points)
                    prior_features = prior_features_callable(points)
                    prior_encoded = prior_feature_encoder_fn(prior_features)
                    scale_tensor = getattr(model, "prior_feature_scale", None)
                    scale = (
                        scale_tensor.to(device=device, dtype=prior_encoded.dtype)
                        if isinstance(scale_tensor, torch.Tensor)
                        else prior_encoded.new_tensor(1.0)
                    )
                    h = point_encoded + scale * prior_encoded
                    if local_context_encoder is not None:
                        local_input = h
                        if positional_callable is not None:
                            local_input = local_input + positional_callable(
                                local_input.shape[1],
                                local_input.device,
                                local_input.dtype,
                            ).unsqueeze(0)
                        h = local_context_encoder(
                            local_input,
                            src_key_padding_mask=local_padding_mask,
                        )
                    segment = segment_context_fn(h.transpose(1, 2)).transpose(1, 2)
                    return shared_prior_encoder_fn(h + segment)

                primary_shared = shared_embedding(primary_points, padding_mask)
                ablation_shared = shared_embedding(ablation_points, padding_mask)
                primary_logits_by_head: dict[str, torch.Tensor] = {}
                ablation_logits_by_head: dict[str, torch.Tensor] = {}
                primary_transfer_by_head: dict[
                    str,
                    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                ] = {}
                for head_name, stage_parts in per_head_parts.items():
                    transfer_primary = _head_mlp_transfer(heads[head_name], primary_shared)
                    transfer_ablation = _head_mlp_transfer(heads[head_name], ablation_shared)
                    if transfer_primary is None or transfer_ablation is None:
                        continue
                    primary_first, primary_hidden, primary_logit, weights = transfer_primary
                    ablation_first, ablation_hidden, ablation_logit, _ = transfer_ablation
                    primary_logits_by_head[head_name] = primary_logit
                    ablation_logits_by_head[head_name] = ablation_logit
                    primary_transfer_by_head[head_name] = (
                        primary_hidden,
                        primary_logit,
                        torch.sigmoid(primary_logit),
                    )
                    head_weight_stats[head_name] = weights
                    pairs = {
                        "shared_embedding": (primary_shared, ablation_shared),
                        "first_linear": (primary_first, ablation_first),
                        "hidden_activation": (primary_hidden, ablation_hidden),
                        "logit": (primary_logit, ablation_logit),
                        "probability": (
                            torch.sigmoid(primary_logit),
                            torch.sigmoid(ablation_logit),
                        ),
                    }
                    for stage_name, (primary_tensor, ablation_tensor) in pairs.items():
                        primary_parts, ablation_parts = stage_parts[stage_name]
                        primary_parts.append(primary_tensor[valid])
                        ablation_parts.append(ablation_tensor[valid])
                    head_idx = head_index_by_name.get(head_name)
                    if (
                        global_indices_usable
                        and head_idx is not None
                        and head_idx < int(target_cpu.shape[1])
                        and head_idx < int(mask_cpu.shape[1])
                    ):
                        per_head_target_parts[head_name].append(
                            target_cpu[valid_global, head_idx]
                        )
                        per_head_mask_parts[head_name].append(mask_cpu[valid_global, head_idx])
                        slice_parts = per_head_slice_mask_parts[head_name]
                        for bucket_id, label in window_slice_labels.items():
                            slice_parts["window_slice"][label].append(
                                valid_window_bucket == int(bucket_id)
                            )
                        for group_name, family_masks in family_slice_masks.items():
                            for family_name, family_mask in family_masks.items():
                                slice_parts[group_name][family_name].append(
                                    family_mask[valid_global]
                                )
                ordered_head_names = [
                    str(name)
                    for name in QUERY_LOCAL_UTILITY_HEAD_NAMES
                    if str(name) in primary_logits_by_head and str(name) in ablation_logits_by_head
                ]
                if (
                    global_indices_usable
                    and len(ordered_head_names) == len(QUERY_LOCAL_UTILITY_HEAD_NAMES)
                ):
                    window_targets = torch.zeros(
                        (
                            int(valid_cpu.shape[0]),
                            int(valid_cpu.shape[1]),
                            len(QUERY_LOCAL_UTILITY_HEAD_NAMES),
                        ),
                        dtype=torch.float32,
                    )
                    window_mask = torch.zeros_like(window_targets, dtype=torch.bool)
                    global_ok = (
                        valid_cpu
                        & (global_indices_cpu >= 0)
                        & (global_indices_cpu < int(target_cpu.shape[0]))
                    )
                    if bool(global_ok.any().item()):
                        window_targets[global_ok] = target_cpu[global_indices_cpu[global_ok]]
                        window_mask[global_ok] = mask_cpu[global_indices_cpu[global_ok]]
                    primary_logits = (
                        torch.cat(
                            [primary_logits_by_head[name] for name in ordered_head_names],
                            dim=-1,
                        )
                        .detach()
                        .clone()
                        .requires_grad_(True)
                    )
                    ablation_logits = torch.cat(
                        [ablation_logits_by_head[name] for name in ordered_head_names],
                        dim=-1,
                    ).detach()
                    with torch.enable_grad():
                        loss = _factorized_query_local_utility_loss(
                            head_logits=primary_logits,
                            head_targets=window_targets.to(device=device),
                            head_mask=window_mask.to(device=device),
                            global_indices=global_indices_cpu.to(device=device),
                            segment_budget_head_weight=float(segment_budget_head_weight),
                            segment_level_loss_weight=float(segment_level_loss_weight),
                            behavior_rank_loss_weight=float(behavior_rank_loss_weight),
                            sparse_head_rank_loss_weight=float(sparse_head_rank_loss_weight),
                            sparse_head_bce_target_mode=str(sparse_head_bce_target_mode),
                        )
                        gradient = torch.autograd.grad(
                            loss,
                            primary_logits,
                            allow_unused=False,
                        )[0].detach()
                    logit_delta = primary_logits.detach() - ablation_logits
                    for head_idx, head_name in enumerate(ordered_head_names):
                        local_mask = window_mask[..., head_idx].to(device=device)
                        if not bool(local_mask.any().item()):
                            continue
                        local_gradient = gradient[..., head_idx][local_mask]
                        local_delta = logit_delta[..., head_idx][local_mask]
                        parts = per_head_loss_gradient_parts[head_name]
                        parts["gradient"].append(local_gradient)
                        parts["logit_delta"].append(local_delta)
                        parts["descent_alignment"].append(-local_gradient * local_delta)

                if primary_transfer_by_head:
                    prior_start_idx = int(primary_points.shape[-1]) - int(prior_dim)
                    for channel_idx, channel_name in enumerate(prior_channel_names):
                        channel_points = primary_points.clone()
                        channel_points[..., prior_start_idx + int(channel_idx)] = 0.0
                        channel_shared = shared_embedding(channel_points, padding_mask)
                        channel_head_parts = per_channel_head_parts[channel_name]
                        for head_name, primary_transfer in primary_transfer_by_head.items():
                            channel_transfer = _head_mlp_transfer(
                                heads[head_name],
                                channel_shared,
                            )
                            if channel_transfer is None:
                                continue
                            primary_hidden, primary_logit, primary_probability = primary_transfer
                            _, channel_hidden, channel_logit, _ = channel_transfer
                            pairs = {
                                "hidden_activation": (primary_hidden, channel_hidden),
                                "logit": (primary_logit, channel_logit),
                                "probability": (
                                    primary_probability,
                                    torch.sigmoid(channel_logit),
                                ),
                            }
                            for stage_name, (primary_tensor, channel_tensor) in pairs.items():
                                primary_parts, ablation_parts = channel_head_parts[head_name][
                                    stage_name
                                ]
                                primary_parts.append(primary_tensor[valid])
                                ablation_parts.append(channel_tensor[valid])
    finally:
        model.train(original_training)

    per_head: dict[str, Any] = {}
    suppression_counts: dict[str, int] = {}
    for head_idx, head_name_raw in enumerate(QUERY_LOCAL_UTILITY_HEAD_NAMES):
        head_name = str(head_name_raw)
        stage_parts = per_head_parts.get(head_name)
        if stage_parts is None:
            continue
        stage_summary = {
            stage_name: _stage_tensor_sensitivity(
                primary_parts=primary_parts,
                ablation_parts=ablation_parts,
            )
            for stage_name, (primary_parts, ablation_parts) in stage_parts.items()
        }
        if not bool(stage_summary.get("shared_embedding", {}).get("available")):
            per_head[head_name] = {"available": False, "reason": "missing_transfer_tensors"}
            continue
        valid = mask_cpu[:, head_idx] if head_idx < int(mask_cpu.shape[1]) else torch.zeros(0)
        target = target_cpu[:, head_idx] if head_idx < int(target_cpu.shape[1]) else torch.zeros(0)
        target_valid = target[valid] if int(valid.numel()) == int(target.numel()) else torch.zeros(0)
        probability_stage = stage_parts["probability"][0]
        probability = (
            torch.cat([part.detach().cpu().float().reshape(-1) for part in probability_stage])
            if probability_stage
            else torch.zeros(0)
        )
        sigmoid_derivative = probability * (1.0 - probability) if int(probability.numel()) else None
        sigmoid_derivative_mean = (
            float(sigmoid_derivative.mean().item()) if sigmoid_derivative is not None else None
        )
        classification = _classify_head_transfer(
            shared=stage_summary["shared_embedding"],
            first=stage_summary["first_linear"],
            hidden=stage_summary["hidden_activation"],
            logit=stage_summary["logit"],
            probability=stage_summary["probability"],
            sigmoid_derivative_mean=sigmoid_derivative_mean,
        )
        suppression_counts[classification] = int(suppression_counts.get(classification, 0)) + 1
        output_alignment = _prior_output_layer_alignment_diagnostics(
            primary_hidden_parts=stage_parts["hidden_activation"][0],
            ablation_hidden_parts=stage_parts["hidden_activation"][1],
            primary_logit_parts=stage_parts["logit"][0],
            ablation_logit_parts=stage_parts["logit"][1],
            primary_probability_parts=stage_parts["probability"][0],
            target_parts=per_head_target_parts.get(head_name, []),
            mask_parts=per_head_mask_parts.get(head_name, []),
            final_weight=head_final_weights.get(head_name),
            slice_mask_parts=per_head_slice_mask_parts.get(head_name),
        )
        loss_gradient_alignment = _loss_gradient_alignment_summary(
            descent_alignment_parts=per_head_loss_gradient_parts.get(head_name, {}).get(
                "descent_alignment", []
            ),
            gradient_parts=per_head_loss_gradient_parts.get(head_name, {}).get("gradient", []),
            logit_delta_parts=per_head_loss_gradient_parts.get(head_name, {}).get(
                "logit_delta", []
            ),
            loss_config=loss_config,
        )
        per_head[head_name] = {
            "available": True,
            "classification": classification,
            "stage_sensitivity": stage_summary,
            "output_layer_alignment": output_alignment,
            "configured_loss_gradient_alignment": loss_gradient_alignment,
            "first_linear_delta_l2_to_shared_delta_l2": _ratio_from_summaries(
                stage_summary["first_linear"], stage_summary["shared_embedding"], "delta_l2"
            ),
            "hidden_delta_l2_to_first_linear_delta_l2": _ratio_from_summaries(
                stage_summary["hidden_activation"], stage_summary["first_linear"], "delta_l2"
            ),
            "logit_delta_l2_to_hidden_delta_l2": _ratio_from_summaries(
                stage_summary["logit"], stage_summary["hidden_activation"], "delta_l2"
            ),
            "probability_mean_abs_delta_to_logit_mean_abs_delta": _ratio_from_summaries(
                stage_summary["probability"], stage_summary["logit"], "mean_abs_delta"
            ),
            "sigmoid_derivative_mean": sigmoid_derivative_mean,
            "sigmoid_derivative_min": float(sigmoid_derivative.min().item())
            if sigmoid_derivative is not None and int(sigmoid_derivative.numel()) > 0
            else None,
            "sigmoid_derivative_lt_0_02_fraction": float(
                (sigmoid_derivative < 0.02).float().mean().item()
            )
            if sigmoid_derivative is not None and int(sigmoid_derivative.numel()) > 0
            else None,
            "target_mean": float(target_valid.mean().item()) if int(target_valid.numel()) else None,
            "target_std": float(target_valid.std(unbiased=False).item())
            if int(target_valid.numel()) > 1
            else 0.0,
            "valid_target_count": int(target_valid.numel()),
            **head_weight_stats.get(head_name, {}),
        }
    channel_rows: dict[str, Any] = {}
    for channel_name, per_head_channel_parts in per_channel_head_parts.items():
        channel_per_head: dict[str, Any] = {}
        for head_name_raw in QUERY_LOCAL_UTILITY_HEAD_NAMES:
            head_name = str(head_name_raw)
            stage_parts = per_head_channel_parts.get(head_name)
            if stage_parts is None:
                continue
            output_alignment = _prior_output_layer_alignment_diagnostics(
                primary_hidden_parts=stage_parts["hidden_activation"][0],
                ablation_hidden_parts=stage_parts["hidden_activation"][1],
                primary_logit_parts=stage_parts["logit"][0],
                ablation_logit_parts=stage_parts["logit"][1],
                primary_probability_parts=stage_parts["probability"][0],
                target_parts=per_head_target_parts.get(head_name, []),
                mask_parts=per_head_mask_parts.get(head_name, []),
                final_weight=head_final_weights.get(head_name),
                slice_mask_parts=per_head_slice_mask_parts.get(head_name),
            )
            channel_per_head[head_name] = {
                "available": bool(output_alignment.get("available")),
                "classification": _classify_prior_channel_output_alignment(
                    output_alignment,
                ),
                "output_layer_alignment": output_alignment,
            }
        channel_rows[channel_name] = {
            "available": bool(channel_per_head),
            "diagnostic_only": True,
            "per_head": channel_per_head,
        }
    channel_decomposition = _summarize_prior_channel_direction_decomposition(channel_rows)
    channel_decomposition["channel_count"] = len(channel_rows)
    channel_decomposition["per_channel"] = channel_rows
    return {
        "available": bool(per_head),
        "diagnostic_only": True,
        "per_head": per_head,
        "classification_counts": suppression_counts,
        "prior_channel_direction_decomposition": channel_decomposition,
    }


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


def _factorized_head_fit_diagnostics(
    *,
    head_logits: torch.Tensor | None,
    factorized_targets: torch.Tensor | None,
    factorized_mask: torch.Tensor | None,
    points: torch.Tensor | None = None,
    boundaries: list[tuple[int, int]] | None = None,
    typed_queries: list[dict[str, Any]] | None = None,
    seed: int,
) -> dict[str, Any]:
    """Summarize training-set fit for every factorized QueryLocalUtility head."""
    if head_logits is None or factorized_targets is None or factorized_mask is None:
        return {"factorized_head_fit_diagnostics_available": False}
    if head_logits.shape != factorized_targets.shape or factorized_mask.shape != head_logits.shape:
        return {"factorized_head_fit_diagnostics_available": False, "reason": "shape_mismatch"}
    if int(head_logits.shape[-1]) != len(QUERY_LOCAL_UTILITY_HEAD_NAMES):
        return {"factorized_head_fit_diagnostics_available": False, "reason": "head_count_mismatch"}

    diagnostics: dict[str, Any] = {
        "factorized_head_fit_diagnostics_available": True,
        "factorized_head_fit": {},
    }
    head_rows: dict[str, dict[str, Any]] = {}
    generator = torch.Generator().manual_seed(int(seed) + 1201)
    for head_idx, head_name in enumerate(QUERY_LOCAL_UTILITY_HEAD_NAMES):
        valid = factorized_mask[:, head_idx].detach().cpu().bool()
        if not bool(valid.any().item()):
            head_rows[str(head_name)] = {"available": False, "reason": "no_valid_targets"}
            continue
        scores = torch.sigmoid(head_logits[:, head_idx].detach().cpu().float())[valid]
        targets = factorized_targets[:, head_idx].detach().cpu().float().clamp(0.0, 1.0)[valid]
        sampled_scores, sampled_targets = _discriminative_sample(
            scores,
            targets,
            n_each=200,
            generator=generator,
        )
        k = max(1, math.ceil(0.05 * int(scores.numel())))
        selected = torch.topk(scores, k=k, largest=True).indices
        ideal = torch.topk(targets, k=k, largest=True).indices
        selected_mass = float(targets[selected].sum().item())
        ideal_mass = float(targets[ideal].sum().item())
        tau = float(_kendall_tau(sampled_scores, sampled_targets))
        topk_recall = float(selected_mass / max(ideal_mass, 1e-12))
        head_rows[str(head_name)] = {
            "available": True,
            "valid_point_count": int(scores.numel()),
            "positive_target_count": int((targets > 0.0).sum().item()),
            "positive_target_fraction": float((targets > 0.0).float().mean().item()),
            "target_mean": float(targets.mean().item()),
            "target_std": float(targets.std(unbiased=False).item())
            if int(targets.numel()) > 1
            else 0.0,
            "target_mass": float(targets.sum().item()),
            "prediction_mean": float(scores.mean().item()),
            "prediction_std": float(scores.std(unbiased=False).item())
            if int(scores.numel()) > 1
            else 0.0,
            "kendall_tau": tau,
            "topk_mass_recall_at_5_percent": topk_recall,
        }
        diagnostics[f"{head_name}_head_tau"] = tau
        diagnostics[f"{head_name}_head_topk_mass_recall_at_5_percent"] = topk_recall
    diagnostics["factorized_head_fit"] = head_rows
    diagnostics["family_conditioned_head_trainability"] = (
        _family_conditioned_head_trainability_diagnostics(
            head_logits=head_logits,
            factorized_targets=factorized_targets,
            factorized_mask=factorized_mask,
            points=points,
            boundaries=boundaries,
            typed_queries=typed_queries,
            seed=seed,
        )
    )
    return diagnostics


def _family_conditioned_head_trainability_diagnostics(
    *,
    head_logits: torch.Tensor | None,
    factorized_targets: torch.Tensor | None,
    factorized_mask: torch.Tensor | None,
    points: torch.Tensor | None,
    boundaries: list[tuple[int, int]] | None,
    typed_queries: list[dict[str, Any]] | None,
    seed: int,
    ratio: float = 0.05,
) -> dict[str, Any]:
    """Return head-fit diagnostics split by workload family."""
    if (
        head_logits is None
        or factorized_targets is None
        or factorized_mask is None
        or points is None
        or boundaries is None
        or typed_queries is None
    ):
        return {"available": False, "reason": "missing_inputs"}
    if head_logits.shape != factorized_targets.shape or factorized_mask.shape != head_logits.shape:
        return {"available": False, "reason": "shape_mismatch"}
    if int(head_logits.shape[-1]) != len(QUERY_LOCAL_UTILITY_HEAD_NAMES):
        return {"available": False, "reason": "head_count_mismatch"}
    range_queries = [
        query for query in typed_queries if str(query.get("type", "")).lower() == "range"
    ]
    if not range_queries:
        return {"available": False, "reason": "no_range_queries"}

    logits = head_logits.detach().cpu().float()
    targets = factorized_targets.detach().cpu().float().clamp(0.0, 1.0)
    masks = factorized_mask.detach().cpu().bool()
    probabilities = torch.sigmoid(logits)
    points_cpu = points.detach().cpu().float()
    family_evidence = _range_query_family_evidence(
        points=points_cpu,
        boundaries=boundaries,
        range_queries=range_queries,
        group_keys=FAMILY_TRAINABILITY_GROUP_KEYS,
    )
    target_composed = query_local_utility_point_score(
        q_hit=targets[:, 0],
        behavior=targets[:, 1],
        boundary=targets[:, 2],
        replacement=targets[:, 3],
    )
    predicted_composed = query_local_utility_point_score(
        q_hit=probabilities[:, 0],
        behavior=probabilities[:, 1],
        boundary=probabilities[:, 2],
        replacement=probabilities[:, 3],
    )
    generator = torch.Generator().manual_seed(int(seed) + 4211)

    def fit_row(
        *,
        scores: torch.Tensor,
        target_values: torch.Tensor,
        reference: torch.Tensor,
        valid: torch.Tensor,
    ) -> dict[str, Any]:
        valid = valid.bool()
        if int(valid.sum().item()) < 2:
            return {"available": False, "reason": "insufficient_valid_points"}
        valid_scores = scores[valid].float()
        valid_targets = target_values[valid].float().clamp(0.0, 1.0)
        sampled_scores, sampled_targets = _discriminative_sample(
            valid_scores,
            valid_targets,
            n_each=200,
            generator=generator,
        )
        k = max(1, math.ceil(float(ratio) * int(valid_scores.numel())))
        selected = torch.topk(valid_scores, k=k, largest=True).indices
        ideal = torch.topk(valid_targets, k=k, largest=True).indices
        selected_target_mass = float(valid_targets[selected].sum().item())
        ideal_target_mass = float(valid_targets[ideal].sum().item())
        ship_topk = _topk_overlap_and_mass_recall(
            ranker=scores,
            reference=reference,
            valid=valid,
            ratio=ratio,
        )
        return {
            "available": True,
            "valid_point_count": int(valid_scores.numel()),
            "positive_target_count": int((valid_targets > 0.0).sum().item()),
            "target_mass": float(valid_targets.sum().item()),
            "target_mean": float(valid_targets.mean().item()),
            "target_std": float(valid_targets.std(unbiased=False).item())
            if int(valid_targets.numel()) > 1
            else 0.0,
            "prediction_mean": float(valid_scores.mean().item()),
            "prediction_std": float(valid_scores.std(unbiased=False).item())
            if int(valid_scores.numel()) > 1
            else 0.0,
            "kendall_tau_with_head_target": float(_kendall_tau(sampled_scores, sampled_targets)),
            "topk_head_target_mass_recall": float(
                selected_target_mass / max(ideal_target_mass, 1e-12)
            ),
            "spearman_with_family_ship_query_evidence": _rank_correlation(
                scores,
                reference,
                valid,
            ),
            "topk_family_ship_query_evidence_mass_recall": ship_topk["reference_mass_recall"],
        }

    out: dict[str, Any] = {
        "available": True,
        "diagnostic_only": True,
        "topk_ratio": float(ratio),
        "group_by": {},
        "focus_families": {
            group_key: sorted(values)
            for group_key, values in DIAGNOSTIC_TRAINABILITY_FOCUS_FAMILIES.items()
        },
    }
    for group_key, family_rows in family_evidence.items():
        group_out: dict[str, Any] = {}
        for family, evidence in family_rows.items():
            family_valid = evidence["query_hit_probability"].detach().cpu().bool()
            reference = evidence["ship_query_evidence"].detach().cpu().float()
            head_rows: dict[str, Any] = {}
            weak_heads = []
            for head_idx, head_name in enumerate(QUERY_LOCAL_UTILITY_HEAD_NAMES):
                valid = family_valid & masks[:, head_idx]
                row = fit_row(
                    scores=probabilities[:, head_idx],
                    target_values=targets[:, head_idx],
                    reference=reference,
                    valid=valid,
                )
                head_rows[str(head_name)] = row
                spearman = row.get("spearman_with_family_ship_query_evidence")
                if row.get("available") is True and (spearman is None or float(spearman) < 0.0):
                    weak_heads.append(str(head_name))
            composed_valid = family_valid & masks[:, 0] & masks[:, 1] & masks[:, 2] & masks[:, 3]
            composed_row = fit_row(
                scores=predicted_composed,
                target_values=target_composed,
                reference=reference,
                valid=composed_valid,
            )
            spearman = composed_row.get("spearman_with_family_ship_query_evidence")
            if composed_row.get("available") is True and (
                spearman is None or float(spearman) < 0.0
            ):
                weak_heads.append("factorized_composed_score")
            focus_family = family in DIAGNOSTIC_TRAINABILITY_FOCUS_FAMILIES.get(
                group_key, frozenset()
            )
            group_out[str(family)] = {
                "available": bool(family_valid.any().item()),
                "focus_family": focus_family,
                "query_count": int(evidence["query_count"]),
                "valid_hit_point_count": int(family_valid.sum().item()),
                "ship_query_evidence_positive_point_count": int(
                    (reference[family_valid] > 0.0).sum().item()
                )
                if bool(family_valid.any().item())
                else 0,
                "ship_query_evidence_mass": float(reference[family_valid].sum().item())
                if bool(family_valid.any().item())
                else 0.0,
                "head_fit": head_rows,
                "factorized_composed_score_fit": composed_row,
                "weak_ship_evidence_heads": weak_heads,
                "head_trainability_status": (
                    "weak_family_head_signal" if focus_family and weak_heads else "diagnostic_only"
                ),
            }
        out["group_by"][group_key] = group_out
    return out


def _factorized_final_score_composition_diagnostics(
    *,
    head_logits: torch.Tensor | None,
    factorized_targets: torch.Tensor | None,
    scalar_target: torch.Tensor | None,
    scalar_mask: torch.Tensor | None,
    seed: int,
) -> dict[str, Any]:
    """Summarize how the factorized heads compose into the scalar QueryLocalUtility score."""
    if head_logits is None or scalar_target is None or scalar_mask is None:
        return {"factorized_final_score_composition_available": False}
    logits = head_logits.detach().cpu().float()
    target = scalar_target.detach().cpu().float().flatten().clamp(0.0, 1.0)
    mask = scalar_mask.detach().cpu().bool().flatten()
    if logits.ndim != 2 or int(logits.shape[1]) != len(QUERY_LOCAL_UTILITY_HEAD_NAMES):
        return {
            "factorized_final_score_composition_available": False,
            "reason": "head_shape_mismatch",
        }
    if int(logits.shape[0]) != int(target.numel()) or int(mask.numel()) != int(target.numel()):
        return {
            "factorized_final_score_composition_available": False,
            "reason": "target_shape_mismatch",
        }
    if not bool(mask.any().item()):
        return {
            "factorized_final_score_composition_available": False,
            "reason": "no_labelled_points",
        }

    def composed_score(probabilities: torch.Tensor) -> torch.Tensor:
        q_hit = probabilities[:, 0].float().clamp(0.0, 1.0)
        behavior = probabilities[:, 1].float().clamp(0.0, 1.0)
        boundary = probabilities[:, 2].float().clamp(0.0, 1.0)
        replacement = probabilities[:, 3].float().clamp(0.0, 1.0)
        return query_local_utility_point_score(
            q_hit=q_hit,
            behavior=behavior,
            boundary=boundary,
            replacement=replacement,
        )

    def topk_mass_and_overlap(scores: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
        k = max(1, math.ceil(0.05 * int(scores.numel())))
        selected = torch.topk(scores, k=k, largest=True).indices
        ideal = torch.topk(reference, k=k, largest=True).indices
        selected_mass = float(reference[selected].sum().item())
        ideal_mass = float(reference[ideal].sum().item())
        selected_mask = torch.zeros_like(reference, dtype=torch.bool)
        ideal_mask = torch.zeros_like(reference, dtype=torch.bool)
        selected_mask[selected] = True
        ideal_mask[ideal] = True
        return float(selected_mass / max(ideal_mass, 1e-12)), float(
            (selected_mask & ideal_mask).sum().item() / k
        )

    probabilities = torch.sigmoid(logits)
    composed = composed_score(probabilities)[mask]
    target_valid = target[mask]
    generator = torch.Generator().manual_seed(int(seed) + 1701)
    sampled_scores, sampled_targets = _discriminative_sample(
        composed,
        target_valid,
        n_each=200,
        generator=generator,
    )
    topk_recall, topk_overlap = topk_mass_and_overlap(composed, target_valid)
    target_std = (
        float(target_valid.std(unbiased=False).item()) if int(target_valid.numel()) > 1 else 0.0
    )
    prediction_std = (
        float(composed.std(unbiased=False).item()) if int(composed.numel()) > 1 else 0.0
    )
    prediction_p05 = float(_safe_quantile(composed, 0.05).item())
    prediction_p95 = float(_safe_quantile(composed, 0.95).item())
    target_p05 = float(_safe_quantile(target_valid, 0.05).item())
    target_p95 = float(_safe_quantile(target_valid, 0.95).item())
    replacement_multiplier = (0.75 + 0.25 * probabilities[:, 3].float().clamp(0.0, 1.0))[mask]
    query_hit_branch = (0.50 * probabilities[:, 0].float().clamp(0.0, 1.0))[mask]
    behavior_branch = (0.45 * probabilities[:, 1].float().clamp(0.0, 1.0))[mask]
    diagnostics: dict[str, Any] = {
        "factorized_final_score_composition_available": True,
        "factorized_final_score_formula": QUERY_LOCAL_UTILITY_FINAL_LABEL_FORMULA,
        "factorized_final_score_prediction_mean": float(composed.mean().item()),
        "factorized_final_score_prediction_std": prediction_std,
        "factorized_final_score_prediction_p05": prediction_p05,
        "factorized_final_score_prediction_p95": prediction_p95,
        "factorized_final_score_prediction_p95_minus_p05": float(prediction_p95 - prediction_p05),
        "factorized_final_score_target_mean": float(target_valid.mean().item()),
        "factorized_final_score_target_std": target_std,
        "factorized_final_score_target_p05": target_p05,
        "factorized_final_score_target_p95": target_p95,
        "factorized_final_score_target_p95_minus_p05": float(target_p95 - target_p05),
        "factorized_final_score_prediction_std_to_target_std": (
            None if target_std <= 1e-12 else float(prediction_std / target_std)
        ),
        "factorized_final_score_tau": float(_kendall_tau(sampled_scores, sampled_targets)),
        "factorized_final_score_topk_mass_recall_at_5_percent": topk_recall,
        "factorized_final_score_topk_overlap_at_5_percent": topk_overlap,
        "factorized_replacement_multiplier_mean": float(replacement_multiplier.mean().item()),
        "factorized_replacement_multiplier_std": (
            float(replacement_multiplier.std(unbiased=False).item())
            if int(replacement_multiplier.numel()) > 1
            else 0.0
        ),
        "factorized_query_hit_branch_mean": float(query_hit_branch.mean().item()),
        "factorized_query_hit_branch_std": (
            float(query_hit_branch.std(unbiased=False).item())
            if int(query_hit_branch.numel()) > 1
            else 0.0
        ),
        "factorized_behavior_branch_mean": float(behavior_branch.mean().item()),
        "factorized_behavior_branch_std": (
            float(behavior_branch.std(unbiased=False).item())
            if int(behavior_branch.numel()) > 1
            else 0.0
        ),
    }

    if factorized_targets is not None and factorized_targets.shape == logits.shape:
        target_probabilities = factorized_targets.detach().cpu().float().clamp(0.0, 1.0)
        target_composed = composed_score(target_probabilities)[mask]
        sampled_target_composed, sampled_label = _discriminative_sample(
            target_composed,
            target_valid,
            n_each=200,
            generator=generator,
        )
        target_topk_recall, target_topk_overlap = topk_mass_and_overlap(
            target_composed, target_valid
        )
        diagnostics.update(
            {
                "factorized_target_formula_label_mae": float(
                    (target_composed - target_valid).abs().mean().item()
                ),
                "factorized_target_formula_label_tau": float(
                    _kendall_tau(sampled_target_composed, sampled_label)
                ),
                "factorized_target_formula_topk_mass_recall_at_5_percent": target_topk_recall,
                "factorized_target_formula_topk_overlap_at_5_percent": target_topk_overlap,
            }
        )
    return diagnostics
