"""Factorized QueryLocalUtility head diagnostics and initialization helpers."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from learning.factorized_prior_learning_diagnostics import (
    _prior_feature_learning_diagnostics as _prior_feature_learning_diagnostics,
)
from learning.factorized_prior_transfer_core import (
    _prior_output_layer_alignment_diagnostics as _prior_output_layer_alignment_diagnostics,
)
from learning.fit_diagnostics import discriminative_sample, kendall_tau
from learning.losses import safe_quantile
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
            except (KeyError, TypeError):
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
    sampled_scores, sampled_targets = discriminative_sample(
        scores[valid],
        targets[valid],
        n_each=200,
        generator=generator,
    )
    tau = kendall_tau(sampled_scores, sampled_targets)
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
        segment_sampled_scores, segment_sampled_targets = discriminative_sample(
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
        segment_tau = float(kendall_tau(segment_sampled_scores, segment_sampled_targets))
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
        sampled_scores, sampled_targets = discriminative_sample(
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
        tau = float(kendall_tau(sampled_scores, sampled_targets))
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
        sampled_scores, sampled_targets = discriminative_sample(
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
            "kendall_tau_with_head_target": float(kendall_tau(sampled_scores, sampled_targets)),
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
    sampled_scores, sampled_targets = discriminative_sample(
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
    prediction_p05 = float(safe_quantile(composed, 0.05).item())
    prediction_p95 = float(safe_quantile(composed, 0.95).item())
    target_p05 = float(safe_quantile(target_valid, 0.05).item())
    target_p95 = float(safe_quantile(target_valid, 0.95).item())
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
        "factorized_final_score_tau": float(kendall_tau(sampled_scores, sampled_targets)),
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
        sampled_target_composed, sampled_label = discriminative_sample(
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
                    kendall_tau(sampled_target_composed, sampled_label)
                ),
                "factorized_target_formula_topk_mass_recall_at_5_percent": target_topk_recall,
                "factorized_target_formula_topk_overlap_at_5_percent": target_topk_overlap,
            }
        )
    return diagnostics
