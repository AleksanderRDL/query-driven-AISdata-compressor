"""Train-query local-swap scalar target builders."""

from __future__ import annotations

import math

import torch

from learning.targets.common import _target_budget_ratios, _target_budget_weights
from learning.targets.set_utility import _range_set_utility_candidates
from scoring.method_scoring import score_range_usefulness
from scoring.query_cache import ScoringQueryCache
from selection.retained_mask_selectors import (
    deterministic_topk_with_jitter,
    evenly_spaced_indices,
)
from workloads.query_types import QUERY_TYPE_ID_RANGE
from workloads.range_geometry import points_in_range_box


def _local_swap_base_plan(
    *,
    n_points: int,
    boundaries: list[tuple[int, int]],
    ratio: float,
    temporal_fraction: float,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, int]:
    """Return full temporal base plus removable points for local-swap targets."""
    base_mask = torch.zeros((int(n_points),), dtype=torch.bool, device=device)
    trajectory_ids = torch.full((int(n_points),), -1, dtype=torch.long, device=device)
    removable_by_trajectory: list[torch.Tensor] = []
    base_fraction = min(1.0, max(0.0, float(temporal_fraction)))
    total_capacity = 0

    for trajectory_id, (start, end) in enumerate(boundaries):
        point_count = int(end - start)
        trajectory_ids[start:end] = int(trajectory_id)
        if point_count <= 0:
            removable_by_trajectory.append(torch.empty((0,), dtype=torch.long, device=device))
            continue
        keep_count = min(point_count, max(2, math.ceil(float(ratio) * point_count)))
        base_indices = evenly_spaced_indices(point_count, keep_count, device)
        base_mask[start + base_indices] = True
        protected_count = min(keep_count, max(2, math.ceil(keep_count * base_fraction)))
        swap_count = min(keep_count - protected_count, point_count - keep_count)
        removable_local = base_indices[(base_indices != 0) & (base_indices != point_count - 1)]
        swap_count = min(max(0, int(swap_count)), int(removable_local.numel()))
        removable_global = start + removable_local
        if swap_count <= 0:
            removable_global = removable_global[:0]
        removable_by_trajectory.append(removable_global.to(dtype=torch.long))
        total_capacity += int(swap_count)

    return base_mask, removable_by_trajectory, trajectory_ids, int(total_capacity)


def _nearest_local_swap_removal(
    candidate_idx: int,
    *,
    trajectory_ids: torch.Tensor,
    removable_by_trajectory: list[torch.Tensor],
) -> int | None:
    """Return the nearest removable temporal-base point for one candidate."""
    if candidate_idx < 0 or candidate_idx >= int(trajectory_ids.numel()):
        return None
    trajectory_id = int(trajectory_ids[candidate_idx].item())
    if trajectory_id < 0 or trajectory_id >= len(removable_by_trajectory):
        return None
    removable = removable_by_trajectory[trajectory_id]
    if int(removable.numel()) <= 0:
        return None
    distances = torch.abs(removable.to(dtype=torch.long) - int(candidate_idx))
    best = torch.argmin(distances)
    return int(removable[best].item())


def _range_local_swap_utility_scores(
    points: torch.Tensor,
    labels: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, object]],
    model_config: object,
    ratios: tuple[float, ...],
    type_idx: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return train-query utility labels for the actual local-swap action."""
    hybrid_mode = str(getattr(model_config, "mlqds_hybrid_mode", "fill")).lower()
    if hybrid_mode not in {"local_swap", "local_delta_swap"}:
        raise ValueError(
            "local_swap_utility_frequency requires mlqds_hybrid_mode='local_swap' or 'local_delta_swap'."
        )

    n_points = int(points.shape[0])
    target = torch.zeros((n_points,), dtype=torch.float32, device=points.device)
    temporal_fraction = float(getattr(model_config, "mlqds_temporal_fraction", 0.0))
    multiplier = max(0.0, float(getattr(model_config, "range_set_utility_multiplier", 1.0) or 0.0))
    if multiplier <= 0.0:
        raise ValueError(
            "range_set_utility_multiplier must be positive for local_swap_utility_frequency."
        )
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
            "local_swap_utility_range_query_count": 0,
            "local_swap_utility_used_budget_count": 0,
            "local_swap_utility_multiplier": float(multiplier),
            "local_swap_utility_mass_mode": mass_mode,
            "local_swap_utility_candidate_limit": int(candidate_limit),
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
        base_mask, removable_by_trajectory, trajectory_ids, swap_capacity = _local_swap_base_plan(
            n_points=n_points,
            boundaries=boundaries,
            ratio=ratio_value,
            temporal_fraction=temporal_fraction,
            device=points.device,
        )
        if swap_capacity <= 0:
            per_budget.append(
                {
                    "budget_ratio": float(ratio_value),
                    "budget_weight": float(budget_weight),
                    "swap_capacity": 0,
                    "candidate_count": 0,
                    "scored_candidate_count": 0,
                    "positive_gain_candidate_count": 0,
                    "selected_count": 0,
                    "selected_gain_mass": 0.0,
                    "positive_label_count": 0,
                    "positive_label_mass": 0.0,
                }
            )
            continue
        used_budget_count += 1
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
            if int(candidates.numel()) <= 0:
                continue

            query_list = [query]
            query_cache = ScoringQueryCache.for_workload(points, boundaries, query_list)
            base_score = float(
                score_range_usefulness(
                    points=points,
                    boundaries=boundaries,
                    retained_mask=base_mask,
                    typed_queries=query_list,
                    query_cache=query_cache,
                )["range_usefulness_score"]
            )
            scored_indices: list[int] = []
            gains: list[float] = []
            retained = base_mask.clone()
            for candidate_idx_tensor in candidates:
                candidate_idx = int(candidate_idx_tensor.item())
                remove_idx = _nearest_local_swap_removal(
                    candidate_idx,
                    trajectory_ids=trajectory_ids,
                    removable_by_trajectory=removable_by_trajectory,
                )
                if remove_idx is None or remove_idx == candidate_idx:
                    continue
                retained[remove_idx] = False
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
                retained[candidate_idx] = bool(base_mask[candidate_idx].item())
                retained[remove_idx] = True
                gain = max(0.0, score - base_score)
                scored_indices.append(candidate_idx)
                gains.append(gain)

            budget_scored_candidate_count += len(scored_indices)
            if not scored_indices:
                continue
            gain_tensor = torch.tensor(gains, dtype=torch.float32, device=points.device)
            positive_gain = gain_tensor > 1e-12
            positive_gain_count = int(positive_gain.sum().item())
            budget_positive_gain_count += positive_gain_count
            if positive_gain_count <= 0:
                continue
            desired_count = min(
                positive_gain_count,
                max(1, math.ceil(multiplier * ratio_value * float(hit_count))),
                int(swap_capacity),
            )
            positive_local = torch.where(positive_gain)[0]
            selected_local_in_positive = deterministic_topk_with_jitter(
                gain_tensor[positive_local],
                keep_count=desired_count,
                trajectory_id=(query_index + 1) * 9176 + int(ratio_value * 10000),
            )
            selected_local = positive_local[selected_local_in_positive]
            selected_indices = torch.tensor(
                [scored_indices[int(local.item())] for local in selected_local],
                dtype=torch.long,
                device=points.device,
            )
            selected_gains = gain_tensor[selected_local]
            selected_gain_mass = float(selected_gains.sum().item())
            budget_selected_count += int(selected_indices.numel())
            budget_gain_mass += selected_gain_mass
            if mass_mode == "gain":
                budget_scores[selected_indices] += selected_gains
            elif mass_mode == "point":
                budget_scores[selected_indices] += 1.0
            else:
                budget_scores[selected_indices] += 1.0 / float(
                    max(1, int(selected_indices.numel()))
                )

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
                "swap_capacity": int(swap_capacity),
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
        "local_swap_utility_range_query_count": int(range_query_count),
        "local_swap_utility_used_budget_count": int(used_budget_count),
        "local_swap_utility_budget_weights": list(budget_weights),
        "local_swap_utility_budget_weight_power": float(
            getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0
        ),
        "local_swap_utility_multiplier": float(multiplier),
        "local_swap_utility_mass_mode": mass_mode,
        "local_swap_utility_candidate_limit": int(candidate_limit),
        "local_swap_utility_candidate_count": int(total_candidate_count),
        "local_swap_utility_scored_candidate_count": int(total_scored_candidate_count),
        "local_swap_utility_positive_gain_candidate_count": int(total_positive_gain_count),
        "local_swap_utility_selected_count": int(total_selected_count),
        "local_swap_utility_selected_gain_mass": float(total_gain_mass),
        "local_swap_utility_source_positive_count": int(positive.sum().item()),
        "local_swap_utility_source_positive_mass": (
            float(target[positive].sum().item()) if bool(positive.any().item()) else 0.0
        ),
        "local_swap_utility_per_budget": per_budget,
    }
    return target, diagnostics


def range_local_swap_utility_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, object]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build train-query labels from positive local-swap RangeUseful gains."""
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")
    if int(points.shape[0]) != int(labels.shape[0]):
        raise ValueError("points and labels must have the same point count.")

    ratios = _target_budget_ratios(model_config)
    target, utility_diagnostics = _range_local_swap_utility_scores(
        points=points,
        labels=labels,
        boundaries=boundaries,
        typed_queries=typed_queries,
        model_config=model_config,
        ratios=ratios,
        type_idx=type_idx,
    )
    if not bool((target > 0.0).any().item()):
        raise ValueError("local_swap_utility_frequency target found no positive source scores.")

    transformed = labels.clone()
    transformed[:, type_idx] = target.clamp(0.0, 1.0)
    transformed_mask = labelled_mask.clone()
    transformed_mask[:, type_idx] = True
    positive = transformed[:, type_idx] > 0.0
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "mode": "local_swap_utility_frequency",
        "source": "range_train_query_local_swap_usefulness_gain",
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


def _range_local_swap_gain_cost_scores(
    points: torch.Tensor,
    labels: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, object]],
    model_config: object,
    ratios: tuple[float, ...],
    type_idx: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return paired add-value/removal-cost labels for local-delta swaps.

    For a candidate paired with its nearest removable temporal anchor:

    - candidate value = score(base - anchor + candidate) - score(base - anchor)
    - anchor cost = score(base) - score(base - anchor)

    The local-delta selector accepts a replacement when candidate score exceeds
    the paired anchor score, so these labels directly encode the desired gate:
    candidate value > anchor cost iff the exact one-step replacement improves
    train-query RangeUseful.
    """
    hybrid_mode = str(getattr(model_config, "mlqds_hybrid_mode", "fill")).lower()
    if hybrid_mode != "local_delta_swap":
        raise ValueError(
            "local_swap_gain_cost_frequency requires mlqds_hybrid_mode='local_delta_swap'."
        )

    n_points = int(points.shape[0])
    target = torch.zeros((n_points,), dtype=torch.float32, device=points.device)
    temporal_fraction = float(getattr(model_config, "mlqds_temporal_fraction", 0.0))
    multiplier = max(0.0, float(getattr(model_config, "range_set_utility_multiplier", 1.0) or 0.0))
    if multiplier <= 0.0:
        raise ValueError(
            "range_set_utility_multiplier must be positive for local_swap_gain_cost_frequency."
        )
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
            "local_swap_gain_cost_range_query_count": 0,
            "local_swap_gain_cost_used_budget_count": 0,
            "local_swap_gain_cost_multiplier": float(multiplier),
            "local_swap_gain_cost_mass_mode": mass_mode,
            "local_swap_gain_cost_candidate_limit": int(candidate_limit),
        }

    used_budget_count = 0
    budget_weights = _target_budget_weights(model_config, ratios)
    used_weight = 0.0
    total_candidate_count = 0
    total_scored_candidate_count = 0
    total_positive_net_gain_count = 0
    total_selected_count = 0
    total_candidate_value_mass = 0.0
    total_removal_cost_mass = 0.0
    per_budget: list[dict[str, object]] = []

    for ratio, budget_weight in zip(ratios, budget_weights, strict=False):
        ratio_value = min(1.0, max(0.0, float(ratio)))
        if ratio_value <= 0.0:
            continue
        base_mask, removable_by_trajectory, trajectory_ids, swap_capacity = _local_swap_base_plan(
            n_points=n_points,
            boundaries=boundaries,
            ratio=ratio_value,
            temporal_fraction=temporal_fraction,
            device=points.device,
        )
        if swap_capacity <= 0:
            per_budget.append(
                {
                    "budget_ratio": float(ratio_value),
                    "budget_weight": float(budget_weight),
                    "swap_capacity": 0,
                    "candidate_count": 0,
                    "scored_candidate_count": 0,
                    "positive_net_gain_count": 0,
                    "selected_count": 0,
                    "selected_candidate_value_mass": 0.0,
                    "selected_removal_cost_mass": 0.0,
                    "positive_label_count": 0,
                    "positive_label_mass": 0.0,
                }
            )
            continue

        used_budget_count += 1
        budget_scores = torch.zeros_like(target)
        budget_candidate_count = 0
        budget_scored_candidate_count = 0
        budget_positive_net_gain_count = 0
        budget_selected_count = 0
        budget_candidate_value_mass = 0.0
        budget_removal_cost_mass = 0.0

        for query_index, query in enumerate(range_queries):
            params = query.get("params")
            if not isinstance(params, dict):
                continue
            range_mask = points_in_range_box(points, params)
            hit_count = int(range_mask.sum().item())
            if hit_count <= 0:
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
            if int(candidates.numel()) <= 0:
                continue

            query_list = [query]
            query_cache = ScoringQueryCache.for_workload(points, boundaries, query_list)
            base_score = float(
                score_range_usefulness(
                    points=points,
                    boundaries=boundaries,
                    retained_mask=base_mask,
                    typed_queries=query_list,
                    query_cache=query_cache,
                )["range_usefulness_score"]
            )
            removal_score_cache: dict[int, float] = {}
            removal_cost_cache: dict[int, float] = {}
            scored_records: list[tuple[int, int, float, float, float]] = []

            for candidate_idx_tensor in candidates:
                candidate_idx = int(candidate_idx_tensor.item())
                remove_idx = _nearest_local_swap_removal(
                    candidate_idx,
                    trajectory_ids=trajectory_ids,
                    removable_by_trajectory=removable_by_trajectory,
                )
                if remove_idx is None or remove_idx == candidate_idx:
                    continue
                if remove_idx not in removal_score_cache:
                    retained_without = base_mask.clone()
                    retained_without[remove_idx] = False
                    removal_score = float(
                        score_range_usefulness(
                            points=points,
                            boundaries=boundaries,
                            retained_mask=retained_without,
                            typed_queries=query_list,
                            query_cache=query_cache,
                        )["range_usefulness_score"]
                    )
                    removal_score_cache[remove_idx] = removal_score
                    removal_cost_cache[remove_idx] = max(0.0, base_score - removal_score)
                removal_score = removal_score_cache[remove_idx]
                retained_replacement = base_mask.clone()
                retained_replacement[remove_idx] = False
                retained_replacement[candidate_idx] = True
                replacement_score = float(
                    score_range_usefulness(
                        points=points,
                        boundaries=boundaries,
                        retained_mask=retained_replacement,
                        typed_queries=query_list,
                        query_cache=query_cache,
                    )["range_usefulness_score"]
                )
                candidate_value = max(0.0, replacement_score - removal_score)
                removal_cost = removal_cost_cache[remove_idx]
                net_gain = replacement_score - base_score
                scored_records.append(
                    (candidate_idx, remove_idx, net_gain, candidate_value, removal_cost)
                )

            budget_scored_candidate_count += len(scored_records)
            if not scored_records:
                continue
            net_gains = torch.tensor(
                [record[2] for record in scored_records],
                dtype=torch.float32,
                device=points.device,
            )
            positive_net_gain = net_gains > 1e-12
            positive_net_gain_count = int(positive_net_gain.sum().item())
            budget_positive_net_gain_count += positive_net_gain_count
            if positive_net_gain_count <= 0:
                continue
            desired_count = min(
                positive_net_gain_count,
                max(1, math.ceil(multiplier * ratio_value * float(hit_count))),
                int(swap_capacity),
            )
            positive_local = torch.where(positive_net_gain)[0]
            tie_positions = torch.arange(
                int(positive_local.numel()),
                dtype=torch.float32,
                device=points.device,
            )
            tie_jitter = 1e-6 * torch.sin(
                tie_positions * 12.9898 + float((query_index + 1) * 9176 + int(ratio_value * 10000))
            )
            ordered_positive = positive_local[
                torch.argsort(net_gains[positive_local] + tie_jitter, descending=True)
            ]

            selected_records: list[tuple[int, int, float, float, float]] = []
            used_removals: set[int] = set()
            for local_idx_tensor in ordered_positive:
                record = scored_records[int(local_idx_tensor.item())]
                remove_idx = int(record[1])
                if remove_idx in used_removals:
                    continue
                selected_records.append(record)
                used_removals.add(remove_idx)
                if len(selected_records) >= desired_count:
                    break
            if not selected_records:
                continue

            selected_count = len(selected_records)
            budget_selected_count += selected_count
            selected_candidate_value_mass = sum(float(record[3]) for record in selected_records)
            selected_removal_cost_by_idx: dict[int, float] = {}
            for (
                _candidate_idx,
                remove_idx,
                _net_gain,
                _candidate_value,
                removal_cost,
            ) in selected_records:
                selected_removal_cost_by_idx[int(remove_idx)] = max(
                    selected_removal_cost_by_idx.get(int(remove_idx), 0.0),
                    float(removal_cost),
                )
            selected_removal_cost_mass = sum(selected_removal_cost_by_idx.values())
            budget_candidate_value_mass += selected_candidate_value_mass
            budget_removal_cost_mass += selected_removal_cost_mass

            if mass_mode == "gain":
                for (
                    candidate_idx,
                    _remove_idx,
                    _net_gain,
                    candidate_value,
                    _removal_cost,
                ) in selected_records:
                    budget_scores[int(candidate_idx)] += float(candidate_value)
                for remove_idx, removal_cost in selected_removal_cost_by_idx.items():
                    budget_scores[int(remove_idx)] += float(removal_cost)
            elif mass_mode == "point":
                for (
                    candidate_idx,
                    _remove_idx,
                    _net_gain,
                    _candidate_value,
                    _removal_cost,
                ) in selected_records:
                    budget_scores[int(candidate_idx)] += 1.0
                for remove_idx, removal_cost in selected_removal_cost_by_idx.items():
                    if removal_cost > 0.0:
                        budget_scores[int(remove_idx)] += 1.0
            else:
                candidate_mass = 1.0 / float(max(1, selected_count))
                removal_mass = 1.0 / float(max(1, len(selected_removal_cost_by_idx)))
                for (
                    candidate_idx,
                    _remove_idx,
                    _net_gain,
                    _candidate_value,
                    _removal_cost,
                ) in selected_records:
                    budget_scores[int(candidate_idx)] += candidate_mass
                for remove_idx, removal_cost in selected_removal_cost_by_idx.items():
                    if removal_cost > 0.0:
                        budget_scores[int(remove_idx)] += removal_mass

        budget_scores = budget_scores / float(range_query_count)
        target += float(budget_weight) * budget_scores
        used_weight += float(budget_weight)
        total_candidate_count += budget_candidate_count
        total_scored_candidate_count += budget_scored_candidate_count
        total_positive_net_gain_count += budget_positive_net_gain_count
        total_selected_count += budget_selected_count
        total_candidate_value_mass += budget_candidate_value_mass
        total_removal_cost_mass += budget_removal_cost_mass
        positive = budget_scores > 0.0
        per_budget.append(
            {
                "budget_ratio": float(ratio_value),
                "budget_weight": float(budget_weight),
                "swap_capacity": int(swap_capacity),
                "candidate_count": int(budget_candidate_count),
                "scored_candidate_count": int(budget_scored_candidate_count),
                "positive_net_gain_count": int(budget_positive_net_gain_count),
                "selected_count": int(budget_selected_count),
                "selected_candidate_value_mass": float(budget_candidate_value_mass),
                "selected_removal_cost_mass": float(budget_removal_cost_mass),
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
        "local_swap_gain_cost_range_query_count": int(range_query_count),
        "local_swap_gain_cost_used_budget_count": int(used_budget_count),
        "local_swap_gain_cost_budget_weights": list(budget_weights),
        "local_swap_gain_cost_budget_weight_power": float(
            getattr(model_config, "range_target_budget_weight_power", 0.0) or 0.0
        ),
        "local_swap_gain_cost_multiplier": float(multiplier),
        "local_swap_gain_cost_mass_mode": mass_mode,
        "local_swap_gain_cost_candidate_limit": int(candidate_limit),
        "local_swap_gain_cost_candidate_count": int(total_candidate_count),
        "local_swap_gain_cost_scored_candidate_count": int(total_scored_candidate_count),
        "local_swap_gain_cost_positive_net_gain_count": int(total_positive_net_gain_count),
        "local_swap_gain_cost_selected_count": int(total_selected_count),
        "local_swap_gain_cost_selected_candidate_value_mass": float(total_candidate_value_mass),
        "local_swap_gain_cost_selected_removal_cost_mass": float(total_removal_cost_mass),
        "local_swap_gain_cost_source_positive_count": int(positive.sum().item()),
        "local_swap_gain_cost_source_positive_mass": (
            float(target[positive].sum().item()) if bool(positive.any().item()) else 0.0
        ),
        "local_swap_gain_cost_per_budget": per_budget,
    }
    return target, diagnostics


def range_local_swap_gain_cost_frequency_training_labels(
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, object]],
    model_config: object,
    type_idx: int = QUERY_TYPE_ID_RANGE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build train-query labels for local-delta candidate value and base cost."""
    if labels.ndim != 2 or labelled_mask.shape != labels.shape:
        raise ValueError("labels and labelled_mask must have matching shape [n_points, n_types].")
    if type_idx < 0 or type_idx >= labels.shape[1]:
        raise ValueError(f"type_idx={type_idx} is outside label shape {tuple(labels.shape)}.")
    if int(points.shape[0]) != int(labels.shape[0]):
        raise ValueError("points and labels must have the same point count.")

    ratios = _target_budget_ratios(model_config)
    target, gain_cost_diagnostics = _range_local_swap_gain_cost_scores(
        points=points,
        labels=labels,
        boundaries=boundaries,
        typed_queries=typed_queries,
        model_config=model_config,
        ratios=ratios,
        type_idx=type_idx,
    )
    if not bool((target > 0.0).any().item()):
        raise ValueError("local_swap_gain_cost_frequency target found no positive source scores.")

    transformed = labels.clone()
    transformed[:, type_idx] = target.clamp(0.0, 1.0)
    transformed_mask = labelled_mask.clone()
    transformed_mask[:, type_idx] = True
    positive = transformed[:, type_idx] > 0.0
    positive_count = int(positive.sum().item())
    diagnostics: dict[str, object] = {
        "mode": "local_swap_gain_cost_frequency",
        "source": "range_train_query_local_swap_candidate_value_and_removal_cost",
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
    diagnostics.update(gain_cost_diagnostics)
    return transformed, transformed_mask, diagnostics
