"""Length-feasibility diagnostics for query-driven experiment reporting."""

from __future__ import annotations

import math
from typing import Any

import torch

from scoring.metrics import compute_length_preservation


def _local_distance_matrix_km(local_points: torch.Tensor) -> torch.Tensor:
    """Return pairwise haversine distances for one trajectory."""
    points = local_points.detach().cpu().float()
    point_count = int(points.shape[0])
    if point_count <= 0:
        return torch.empty((0, 0), dtype=torch.float32)
    lat = torch.deg2rad(points[:, 1].float())
    lon = torch.deg2rad(points[:, 2].float())
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (
        torch.sin(dlat / 2.0) ** 2
        + torch.cos(lat[:, None]) * torch.cos(lat[None, :]) * torch.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt(torch.clamp(1.0 - a, min=1e-9)))
    return (6371.0 * c).to(dtype=torch.float32)


def _max_length_required_mask(
    local_points: torch.Tensor,
    required_mask: torch.Tensor,
    keep_count: int,
) -> torch.Tensor:
    """Return a max-length local mask with all required points retained."""
    point_count = int(local_points.shape[0])
    keep = max(0, min(int(keep_count), point_count))
    required = required_mask.detach().cpu().bool().clone()
    if point_count <= 0 or keep <= 0:
        return torch.zeros((point_count,), dtype=torch.bool)
    required[0] = True
    required[-1] = True
    required_count = int(required.sum().item())
    if required_count > keep:
        raise ValueError(f"required point count {required_count} exceeds keep_count {keep}.")
    if keep >= point_count:
        return torch.ones((point_count,), dtype=torch.bool)

    distances = _local_distance_matrix_km(local_points)
    neg_inf = -1.0e30
    dp = torch.full((keep + 1, point_count), neg_inf, dtype=torch.float32)
    previous = torch.full((keep + 1, point_count), -1, dtype=torch.long)
    required_prefix = torch.cat(
        [torch.zeros((1,), dtype=torch.long), torch.cumsum(required.to(dtype=torch.long), dim=0)]
    )
    dp[1, 0] = 0.0
    for selected_count in range(2, keep + 1):
        for right_idx in range(1, point_count):
            left_indices = torch.arange(0, right_idx, dtype=torch.long)
            skipped_required = required_prefix[right_idx] - required_prefix[left_indices + 1]
            previous_scores = dp[selected_count - 1, :right_idx]
            valid = (
                (skipped_required == 0)
                & torch.isfinite(previous_scores)
                & (previous_scores > neg_inf * 0.5)
            )
            if not bool(valid.any().item()):
                continue
            candidates = previous_scores + distances[:right_idx, right_idx]
            candidates[~valid] = neg_inf
            best_left = int(torch.argmax(candidates).item())
            dp[selected_count, right_idx] = candidates[best_left]
            previous[selected_count, right_idx] = best_left

    if (
        not torch.isfinite(dp[keep, point_count - 1])
        or float(dp[keep, point_count - 1].item()) <= neg_inf * 0.5
    ):
        raise ValueError("No feasible required-point max-length mask found.")
    retained = torch.zeros((point_count,), dtype=torch.bool)
    cursor = point_count - 1
    for selected_count in range(keep, 0, -1):
        retained[cursor] = True
        if selected_count == 1:
            break
        cursor = int(previous[selected_count, cursor].item())
        if cursor < 0:
            raise ValueError("Failed to reconstruct required-point max-length mask.")
    return retained


def score_protected_length_feasibility(
    *,
    scores: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
    learned_slot_fraction_min: float,
) -> dict[str, Any]:
    """Estimate length feasibility while forcing a minimum set of high-score learned points."""
    score_cpu = scores.detach().cpu().float()
    points_cpu = points.detach().cpu().float()
    score_count = int(score_cpu.numel())
    if score_count != int(points_cpu.shape[0]):
        return {"available": False, "reason": "score_point_count_mismatch"}
    ratio = max(0.0, min(1.0, float(compression_ratio)))
    if ratio <= 0.0 or not boundaries:
        return {"available": False, "reason": "empty_budget"}

    local_budgets: list[int] = []
    total_budget = 0
    candidate_rows: list[tuple[float, int, int, int]] = []
    positive_trajectory_count = 0
    for trajectory_id, (start, end) in enumerate(boundaries):
        start_i = int(start)
        end_i = int(end)
        count = max(0, end_i - start_i)
        local_budget = min(count, max(2, math.ceil(ratio * count))) if count > 0 else 0
        local_budgets.append(local_budget)
        total_budget += local_budget
        if local_budget > 0:
            positive_trajectory_count += 1
        if local_budget <= 2 or count <= 2:
            continue
        for local_idx in range(1, count - 1):
            point_idx = start_i + local_idx
            candidate_rows.append(
                (float(score_cpu[point_idx].item()), -point_idx, trajectory_id, local_idx)
            )

    protected_target = min(
        max(0, math.ceil(float(total_budget) * max(0.0, float(learned_slot_fraction_min)))),
        max(0, total_budget - 2 * positive_trajectory_count),
    )
    protected_by_trajectory: list[set[int]] = [set() for _ in boundaries]
    protected_counts = [0 for _ in boundaries]
    protected_total = 0
    candidate_rows.sort(reverse=True)
    for _score, _neg_idx, trajectory_id, local_idx in candidate_rows:
        if protected_total >= protected_target:
            break
        local_capacity = max(0, local_budgets[trajectory_id] - 2)
        if protected_counts[trajectory_id] >= local_capacity:
            continue
        protected_by_trajectory[trajectory_id].add(int(local_idx))
        protected_counts[trajectory_id] += 1
        protected_total += 1

    retained = torch.zeros((score_count,), dtype=torch.bool)
    for trajectory_id, (start, end) in enumerate(boundaries):
        start_i = int(start)
        end_i = int(end)
        count = max(0, end_i - start_i)
        local_budget = local_budgets[trajectory_id]
        if count <= 0 or local_budget <= 0:
            continue
        if local_budget >= count:
            retained[start_i:end_i] = True
            continue
        required = torch.zeros((count,), dtype=torch.bool)
        required[0] = True
        required[-1] = True
        for local_idx in protected_by_trajectory[trajectory_id]:
            required[int(local_idx)] = True
        try:
            local_retained = _max_length_required_mask(
                points_cpu[start_i:end_i],
                required,
                local_budget,
            )
        except ValueError as exc:
            return {"available": False, "reason": "required_mask_infeasible", "error": str(exc)}
        retained[start_i:end_i] = local_retained

    retained_count = int(retained.sum().item())
    length_preservation = compute_length_preservation(points_cpu, boundaries, retained)
    return {
        "available": True,
        "diagnostic_only": True,
        "description": "Max-length mask with endpoints plus top learned-score non-endpoint points protected.",
        "compression_ratio": float(compression_ratio),
        "total_budget_count": total_budget,
        "retained_count": retained_count,
        "protected_score_point_count": protected_total,
        "protected_score_point_fraction_of_budget": float(protected_total / max(1, total_budget)),
        "protected_score_point_fraction_min": float(learned_slot_fraction_min),
        "length_preservation": float(length_preservation),
        "length_gate_target": 0.80,
        "length_gate_would_pass": bool(length_preservation >= 0.80),
    }


def score_protected_length_frontier(
    *,
    scores: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float,
    learned_slot_fraction_min: float,
    protected_fractions: tuple[float, ...] = (0.0, 0.10, 0.15, 0.25),
) -> dict[str, Any]:
    """Return length upper-bound frontier as high-score point protection increases."""
    rows: list[dict[str, Any]] = []
    max_passing_fraction: float | None = None
    materiality_row: dict[str, Any] | None = None
    materiality_floor = float(learned_slot_fraction_min)
    for raw_fraction in protected_fractions:
        fraction = max(0.0, min(1.0, float(raw_fraction)))
        diagnostic = score_protected_length_feasibility(
            scores=scores,
            points=points,
            boundaries=boundaries,
            compression_ratio=compression_ratio,
            learned_slot_fraction_min=fraction,
        )
        if not bool(diagnostic.get("available", False)):
            return {
                "available": False,
                "reason": "frontier_row_unavailable",
                "failed_fraction": fraction,
                "row": diagnostic,
            }
        row = {
            "protected_score_point_fraction_min": fraction,
            "protected_score_point_count": int(diagnostic.get("protected_score_point_count", 0)),
            "protected_score_point_fraction_of_budget": float(
                diagnostic.get("protected_score_point_fraction_of_budget", 0.0)
            ),
            "length_preservation": float(diagnostic.get("length_preservation", 0.0)),
            "length_gate_would_pass": bool(diagnostic.get("length_gate_would_pass", False)),
        }
        rows.append(row)
        if row["length_gate_would_pass"]:
            max_passing_fraction = (
                fraction if max_passing_fraction is None else max(max_passing_fraction, fraction)
            )
        if abs(fraction - materiality_floor) <= 1e-12:
            materiality_row = row
    if materiality_row is None:
        materiality_diagnostic = score_protected_length_feasibility(
            scores=scores,
            points=points,
            boundaries=boundaries,
            compression_ratio=compression_ratio,
            learned_slot_fraction_min=materiality_floor,
        )
        if bool(materiality_diagnostic.get("available", False)):
            materiality_row = {
                "protected_score_point_fraction_min": materiality_floor,
                "protected_score_point_count": int(
                    materiality_diagnostic.get("protected_score_point_count", 0)
                ),
                "protected_score_point_fraction_of_budget": float(
                    materiality_diagnostic.get("protected_score_point_fraction_of_budget", 0.0)
                ),
                "length_preservation": float(
                    materiality_diagnostic.get("length_preservation", 0.0)
                ),
                "length_gate_would_pass": bool(
                    materiality_diagnostic.get("length_gate_would_pass", False)
                ),
            }
    return {
        "available": True,
        "diagnostic_only": True,
        "description": "Max-length upper-bound frontier while protecting increasing fractions of top learned-score points.",
        "compression_ratio": float(compression_ratio),
        "learned_slot_fraction_min": materiality_floor,
        "length_gate_target": 0.80,
        "rows": rows,
        "max_protected_fraction_passing_length_gate": max_passing_fraction,
        "materiality_floor_length_gate_would_pass": (
            None if materiality_row is None else bool(materiality_row["length_gate_would_pass"])
        ),
        "materiality_floor_length_preservation": (
            None if materiality_row is None else float(materiality_row["length_preservation"])
        ),
    }
