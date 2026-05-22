"""Typed F1-contribution label construction. See learning/README.md for details."""

from __future__ import annotations

from typing import Any

import torch

from workloads.query_types import NUM_QUERY_TYPES, QUERY_NAME_TO_ID
from workloads.range_geometry import points_in_range_box

RANGE_LABEL_MODES = ("point_f1",)


def _range_boundary_weights(
    points: torch.Tensor,
    box_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    boundary_prior_weight: float,
) -> torch.Tensor:
    """Optional boundary-crossing prior for range labels.

    ``boundary_prior_weight=0`` keeps pure point-F1 labels. Positive values
    boost in-box points whose previous or next trajectory neighbour is outside
    the query box, then mean-normalize over the in-box support so total query
    label mass remains the same.
    """
    weights = torch.zeros(points.shape[0], dtype=torch.float32, device=points.device)
    in_box_idx = torch.where(box_mask)[0]
    if in_box_idx.numel() == 0:
        return weights
    boost = max(0.0, float(boundary_prior_weight))
    if boost <= 0.0:
        weights[in_box_idx] = 1.0
        return weights

    boundary_full = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    for start, end in boundaries:
        if end <= start:
            continue
        traj_in = box_mask[start:end]
        if not bool(traj_in.any().item()):
            continue
        prev_out = torch.zeros_like(traj_in)
        prev_out[1:] = traj_in[1:] & ~traj_in[:-1]
        prev_out[0] = traj_in[0]
        next_out = torch.zeros_like(traj_in)
        next_out[:-1] = traj_in[:-1] & ~traj_in[1:]
        next_out[-1] = traj_in[-1]
        boundary_full[start:end] = prev_out | next_out

    boundary_in_box = boundary_full[in_box_idx].float()
    raw = 1.0 + boost * boundary_in_box
    mean_raw = float(raw.mean().item())
    weights[in_box_idx] = raw / mean_raw if mean_raw > 1e-12 else 1.0
    return weights


def _add_range_point_f1_labels(
    labels: torch.Tensor,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    box_support: torch.Tensor,
    type_idx: int,
    range_boundary_prior_weight: float,
    component_weight: float = 1.0,
) -> None:
    """Add point-F1 singleton gain for every in-box point."""
    hit_count = int(box_support.sum().item())
    if hit_count <= 0:
        return
    base_gain = float(2.0 / (hit_count + 1.0))
    boundary_weights = _range_boundary_weights(
        points,
        box_support,
        boundaries,
        range_boundary_prior_weight,
    )
    labels[box_support, type_idx] += (
        float(component_weight) * base_gain * boundary_weights[box_support]
    )


def _compute_typed_importance_labels(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    range_boundary_prior_weight: float = 0.0,
    range_label_mode: str = "point_f1",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-point per-type labels as expected query-F1 contribution."""
    range_label_mode = str(range_label_mode).lower()
    if range_label_mode not in RANGE_LABEL_MODES:
        raise ValueError(f"range_label_mode must be one of {RANGE_LABEL_MODES}.")

    n = points.shape[0]
    labels = torch.zeros((n, NUM_QUERY_TYPES), dtype=torch.float32, device=points.device)
    labelled_mask = torch.zeros((n, NUM_QUERY_TYPES), dtype=torch.bool, device=points.device)
    query_counts = torch.zeros((NUM_QUERY_TYPES,), dtype=torch.float32, device=points.device)

    for q in typed_queries:
        qtype = str(q["type"]).lower()
        if qtype != "range":
            raise ValueError(
                f"Only range queries are supported for labels; got query type: {qtype}"
            )
        t_idx = QUERY_NAME_TO_ID[qtype]
        params = q["params"]
        query_counts[t_idx] += 1.0

        box_support = points_in_range_box(points, params)
        _add_range_point_f1_labels(
            labels=labels,
            points=points,
            boundaries=boundaries,
            box_support=box_support,
            type_idx=t_idx,
            range_boundary_prior_weight=range_boundary_prior_weight,
        )

    for type_idx in range(NUM_QUERY_TYPES):
        count = float(query_counts[type_idx].item())
        if count > 0.0:
            labels[:, type_idx] = labels[:, type_idx] / count
            labels[:, type_idx] = torch.clamp(labels[:, type_idx], 0.0, 1.0)
            labelled_mask[:, type_idx] = True

    return labels, labelled_mask


def compute_typed_importance_labels(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    range_boundary_prior_weight: float = 0.0,
    range_label_mode: str = "point_f1",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-point per-type labels as expected query-F1 contribution."""
    labels, labelled_mask = _compute_typed_importance_labels(
        points=points,
        boundaries=boundaries,
        typed_queries=typed_queries,
        range_boundary_prior_weight=range_boundary_prior_weight,
        range_label_mode=range_label_mode,
    )
    return labels, labelled_mask
