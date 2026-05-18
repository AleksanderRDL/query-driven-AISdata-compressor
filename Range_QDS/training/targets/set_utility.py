"""Train-query set-utility scalar target builders."""

from __future__ import annotations

import math

import torch

from scoring.method_scoring import score_range_usefulness
from scoring.query_cache import ScoringQueryCache
from selection.retained_mask_selectors import (
    deterministic_topk_with_jitter,
    evenly_spaced_indices,
)
from training.targets.common import (
    _target_budget_ratios,
    _target_budget_weights,
    _temporal_base_mask_for_ratio,
)
from workloads.query_types import QUERY_TYPE_ID_RANGE
from workloads.range_geometry import points_in_range_box, segment_box_bracket_indices


def _range_set_utility_candidates(
    *,
    points: torch.Tensor,
    labels: torch.Tensor,
    type_idx: int,
    boundaries: list[tuple[int, int]],
    query: dict[str, object],
    base_mask: torch.Tensor,
    limit: int,
) -> torch.Tensor:
    """Return bounded train-query candidates for marginal set-utility scoring."""
    params = query.get("params")
    if not isinstance(params, dict):
        return torch.empty((0,), dtype=torch.long, device=points.device)
    range_mask = points_in_range_box(points, params)
    in_box = torch.where(range_mask)[0].to(dtype=torch.long)
    crossing = segment_box_bracket_indices(points, boundaries, params).to(
        device=points.device, dtype=torch.long
    )
    if in_box.numel() == 0 and crossing.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=points.device)
    candidates = torch.unique(torch.cat([in_box, crossing]), sorted=True)
    candidates = candidates[~base_mask[candidates]]
    if candidates.numel() <= 0:
        return candidates

    candidate_limit = int(limit)
    if candidate_limit <= 0 or int(candidates.numel()) <= candidate_limit:
        return candidates

    label_scores = labels[candidates, type_idx].float().clamp(min=0.0)
    top_count = min(int(candidates.numel()), max(1, candidate_limit // 2))
    top_local = deterministic_topk_with_jitter(
        label_scores,
        keep_count=top_count,
        trajectory_id=int(candidates.numel()) + 104729,
    )
    spaced_count = max(0, candidate_limit - top_count)
    spaced_local = evenly_spaced_indices(int(candidates.numel()), spaced_count, points.device)
    limited = torch.unique(
        torch.cat([candidates[top_local], candidates[spaced_local]]), sorted=True
    )
    if int(limited.numel()) > candidate_limit:
        limited_scores = labels[limited, type_idx].float().clamp(min=0.0)
        keep_local = deterministic_topk_with_jitter(
            limited_scores,
            keep_count=candidate_limit,
            trajectory_id=int(limited.numel()) + 1299709,
        )
        limited = torch.sort(limited[keep_local]).values
    return limited


def _range_set_utility_scores(
    points: torch.Tensor,
    labels: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, object]],
    model_config: object,
    ratios: tuple[float, ...],
    type_idx: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return one-step marginal RangeUseful-gain targets from train queries.

    This target scores candidate residual points by the actual train-query
    RangeUseful gain from adding that single point to the query-blind temporal
    base. It is still workload-blind at inference because only the resulting
    aggregate labels are used to train point-only scoring.
    """
    n_points = int(points.shape[0])
    target = torch.zeros((n_points,), dtype=torch.float32, device=points.device)
    temporal_fraction = float(getattr(model_config, "mlqds_temporal_fraction", 0.0))
    multiplier = max(0.0, float(getattr(model_config, "range_set_utility_multiplier", 1.0) or 0.0))
    if multiplier <= 0.0:
        raise ValueError("range_set_utility_multiplier must be positive for set_utility_frequency.")
    mass_mode = str(getattr(model_config, "range_set_utility_mass_mode", "gain")).lower()
    if mass_mode not in {"gain", "point", "query"}:
        raise ValueError("range_set_utility_mass_mode must be 'gain', 'point', or 'query'.")
    candidate_limit = int(getattr(model_config, "range_set_utility_candidate_limit", 128) or 0)

    range_queries = [
        query
        for query in typed_queries
        if str(query.get("type", "")).lower() == "range" and isinstance(query.get("params"), dict)
    ]
    range_query_count = len(range_queries)
    if range_query_count <= 0:
        return target, {
            "set_utility_range_query_count": 0,
            "set_utility_used_budget_count": 0,
            "set_utility_multiplier": float(multiplier),
            "set_utility_mass_mode": mass_mode,
            "set_utility_candidate_limit": int(candidate_limit),
        }

    used_budget_count = 0
    budget_weights = _target_budget_weights(model_config, ratios)
    used_weight = 0.0
    total_candidate_count = 0
    total_scored_candidate_count = 0
    total_selected_count = 0
    total_positive_gain_count = 0
    total_gain_mass = 0.0
    per_budget: list[dict[str, object]] = []

    for ratio, budget_weight in zip(ratios, budget_weights, strict=False):
        ratio_value = min(1.0, max(0.0, float(ratio)))
        if ratio_value <= 0.0:
            continue
        used_budget_count += 1
        base_mask = _temporal_base_mask_for_ratio(
            n_points=n_points,
            boundaries=boundaries,
            ratio=ratio_value,
            temporal_fraction=temporal_fraction,
            device=points.device,
        )
        budget_scores = torch.zeros_like(target)
        budget_candidate_count = 0
        budget_scored_candidate_count = 0
        budget_selected_count = 0
        budget_positive_gain_count = 0
        budget_gain_mass = 0.0

        for query_index, query in enumerate(range_queries):
            params = query.get("params")
            if not isinstance(params, dict):
                continue
            range_mask = points_in_range_box(points, params)
            hit_count = int(range_mask.sum().item())
            if hit_count <= 0:
                continue
            desired_count = min(
                hit_count,
                max(1, math.ceil(multiplier * ratio_value * float(hit_count))),
            )
            if hit_count >= 2:
                desired_count = max(2, desired_count)
            base_hit_count = int((base_mask & range_mask).sum().item())
            residual_needed = max(0, desired_count - base_hit_count)
            if residual_needed <= 0:
                continue

            candidates = _range_set_utility_candidates(
                points=points,
                labels=labels,
                type_idx=type_idx,
                boundaries=boundaries,
                query=query,
                base_mask=base_mask,
                limit=candidate_limit,
            )
            budget_candidate_count += int(candidates.numel())
            if candidates.numel() <= 0:
                continue

            query_list = [query]
            query_cache = ScoringQueryCache.for_workload(points, boundaries, query_list)
            retained = base_mask.clone()
            base_score = float(
                score_range_usefulness(
                    points=points,
                    boundaries=boundaries,
                    retained_mask=retained,
                    typed_queries=query_list,
                    query_cache=query_cache,
                )["range_usefulness_score"]
            )
            gains = torch.zeros(
                (int(candidates.numel()),), dtype=torch.float32, device=points.device
            )
            for candidate_pos, candidate_idx_tensor in enumerate(candidates):
                candidate_idx = int(candidate_idx_tensor.item())
                retained[candidate_idx] = True
                score = float(
                    score_range_usefulness(
                        points=points,
                        boundaries=boundaries,
                        retained_mask=retained,
                        typed_queries=query_list,
                        query_cache=query_cache,
                    )["range_usefulness_score"]
                )
                retained[candidate_idx] = False
                gains[candidate_pos] = max(0.0, score - base_score)

            positive_gain = gains > 1e-12
            positive_gain_count = int(positive_gain.sum().item())
            budget_scored_candidate_count += int(candidates.numel())
            budget_positive_gain_count += positive_gain_count
            if positive_gain_count <= 0:
                continue

            keep_count = min(int(residual_needed), positive_gain_count)
            positive_local = torch.where(positive_gain)[0]
            selected_local_in_positive = deterministic_topk_with_jitter(
                gains[positive_local],
                keep_count=keep_count,
                trajectory_id=(query_index + 1) * 1009 + int(ratio_value * 10000),
            )
            selected_local = positive_local[selected_local_in_positive]
            selected_indices = candidates[selected_local]
            selected_gains = gains[selected_local]
            selected_gain_mass = float(selected_gains.sum().item())
            budget_selected_count += int(selected_indices.numel())
            budget_gain_mass += selected_gain_mass

            if mass_mode == "gain":
                budget_scores[selected_indices] += selected_gains
            elif mass_mode == "point":
                budget_scores[selected_indices] += 1.0
            else:
                query_weight = 1.0 / float(max(1, int(selected_indices.numel())))
                budget_scores[selected_indices] += float(query_weight)

        budget_scores = budget_scores / float(range_query_count)
        target += float(budget_weight) * budget_scores
        used_weight += float(budget_weight)
        total_candidate_count += budget_candidate_count
        total_scored_candidate_count += budget_scored_candidate_count
        total_selected_count += budget_selected_count
        total_positive_gain_count += budget_positive_gain_count
        total_gain_mass += budget_gain_mass
        positive = budget_scores > 0.0
        per_budget.append(
            {
                "budget_ratio": float(ratio_value),
                "budget_weight": float(budget_weight),
                "temporal_base_point_count": int(base_mask.sum().item()),
                "candidate_count": int(budget_candidate_count),
                "scored_candidate_count": int(budget_scored_candidate_count),
                "positive_gain_candidate_count": int(budget_positive_gain_count),
                "selected_count": int(budget_selected_count),
                "selected_gain_mass": float(budget_gain_mass),
                "positive_label_count": int(positive.sum().item()),
                "positive_label_mass": (
                    float(budget_scores[positive].sum().item())
                    if bool(positive.any().item())
                    else 0.0
                ),
            }
        )

    if used_weight > 1e-12:
        target = target / float(used_weight)
    positive = target > 0.0
    diagnostics = {
        "set_utility_range_query_count": int(range_query_count),
        "set_utility_used_budget_count": int(used_budget_count),
        "set_utility_budget_weights": list(budget_weights),
        "set_utility_budget_weight_power": float(
            getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0
        ),
        "set_utility_multiplier": float(multiplier),
        "set_utility_mass_mode": mass_mode,
        "set_utility_candidate_limit": int(candidate_limit),
        "set_utility_candidate_count": int(total_candidate_count),
        "set_utility_scored_candidate_count": int(total_scored_candidate_count),
        "set_utility_positive_gain_candidate_count": int(total_positive_gain_count),
        "set_utility_selected_count": int(total_selected_count),
        "set_utility_selected_gain_mass": float(total_gain_mass),
        "set_utility_source_positive_count": int(positive.sum().item()),
        "set_utility_source_positive_mass": (
            float(target[positive].sum().item()) if bool(positive.any().item()) else 0.0
        ),
        "set_utility_per_budget": per_budget,
    }
    return target, diagnostics


def range_set_utility_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, object]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build one-step marginal RangeUseful-gain labels for a blind student."""
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")
    if int(points.shape[0]) != int(labels.shape[0]):
        raise ValueError("points and labels must have the same point count.")

    ratios = _target_budget_ratios(model_config)
    target, utility_diagnostics = _range_set_utility_scores(
        points=points,
        labels=labels,
        boundaries=boundaries,
        typed_queries=typed_queries,
        model_config=model_config,
        ratios=ratios,
        type_idx=type_idx,
    )
    if not bool((target > 0.0).any().item()):
        raise ValueError("set_utility_frequency target found no positive source scores.")

    transformed = labels.clone()
    transformed[:, type_idx] = target.clamp(0.0, 1.0)
    transformed_mask = labelled_mask.clone()
    transformed_mask[:, type_idx] = True
    positive = transformed[:, type_idx] > 0.0
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "mode": "set_utility_frequency",
        "source": "range_train_query_marginal_usefulness_gain",
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
        "mlqds_temporal_fraction": float(getattr(model_config, "mlqds_temporal_fraction", 0.0)),
        "mlqds_hybrid_mode": str(getattr(model_config, "mlqds_hybrid_mode", "fill")),
    }
    diagnostics.update(utility_diagnostics)
    return transformed, transformed_mask, diagnostics
