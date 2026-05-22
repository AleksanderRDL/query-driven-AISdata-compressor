"""Small helpers for model training orchestration."""

from __future__ import annotations

import math

import torch

from learning.model_features import WORKLOAD_BLIND_RANGE_MODEL_TYPE
from learning.query_prior_fields import QUERY_PRIOR_FIELD_NAMES
from learning.scaler import FeatureScaler
from learning.targets.common import scaled_training_target_for_type
from learning.targets.query_local_utility import QUERY_LOCAL_UTILITY_TARGET_MODES
from workloads.typed_workload import TypedQueryWorkload


def _historical_prior_support_mask(
    targets: torch.Tensor,
    boundaries: list[tuple[int, int]],
    support_ratio: float,
) -> torch.Tensor:
    """Return a per-trajectory top-target support mask for historical priors."""
    ratio = min(1.0, max(0.0, float(support_ratio)))
    support_mask = torch.zeros((int(targets.shape[0]),), dtype=torch.bool, device=targets.device)
    if ratio >= 1.0:
        support_mask[:] = True
        return support_mask
    if ratio <= 0.0:
        return support_mask

    for start, end in boundaries:
        point_count = int(end - start)
        if point_count <= 0:
            continue
        keep_count = min(point_count, max(1, math.ceil(ratio * point_count)))
        if keep_count >= point_count:
            support_mask[start:end] = True
            continue
        local_targets = targets[start:end].float()
        local_indices = torch.topk(local_targets, k=keep_count, largest=True).indices
        support_mask[start + local_indices] = True
    return support_mask


def _fit_scaler_for_model(
    points: torch.Tensor, queries: torch.Tensor, model_type: str
) -> FeatureScaler:
    """Fit feature scaling, preserving semantic zero for query-prior channels."""
    scaler = FeatureScaler.fit(points, queries)
    if str(model_type).lower() == WORKLOAD_BLIND_RANGE_MODEL_TYPE:
        prior_dim = len(QUERY_PRIOR_FIELD_NAMES)
        if int(scaler.point_min.numel()) >= prior_dim:
            prior_slice = slice(-prior_dim, None)
            scaler.point_min[prior_slice] = torch.minimum(
                scaler.point_min[prior_slice],
                torch.zeros_like(scaler.point_min[prior_slice]),
            )
            scaler.point_max[prior_slice] = torch.maximum(
                scaler.point_max[prior_slice],
                torch.ones_like(scaler.point_max[prior_slice]),
            )
    return scaler


def _require_validation_inputs(
    validation_trajectories: list[torch.Tensor] | None,
    validation_boundaries: list[tuple[int, int]] | None,
    validation_workload: TypedQueryWorkload | None,
) -> tuple[list[torch.Tensor], list[tuple[int, int]], TypedQueryWorkload]:
    """Return validation inputs after enforcing the checkpoint-score contract."""
    if (
        validation_trajectories is None
        or validation_boundaries is None
        or validation_workload is None
    ):
        raise RuntimeError("Validation scoring requested without complete validation inputs.")
    return validation_trajectories, validation_boundaries, validation_workload


def _canonical_segment_ids_for_boundaries(
    *,
    point_count: int,
    boundaries: list[tuple[int, int]],
    segment_size: int,
) -> torch.Tensor:
    """Return stable selector-aligned segment ids for every flattened point."""
    ids = torch.full((int(point_count),), -1, dtype=torch.long)
    size = max(1, int(segment_size))
    segment_id = 0
    for start, end in boundaries:
        for seg_start in range(int(start), int(end), size):
            seg_end = min(int(end), seg_start + size)
            if seg_end <= seg_start:
                continue
            ids[seg_start:seg_end] = int(segment_id)
            segment_id += 1
    return ids


def _scalar_training_target_for_mode(
    *,
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    workload_type_id: int,
    range_training_target_mode: str,
) -> tuple[torch.Tensor, str]:
    """Return the scalar target used by the primary loss and its diagnostic basis."""
    mode = str(range_training_target_mode).lower()
    if mode in QUERY_LOCAL_UTILITY_TARGET_MODES:
        return labels[:, int(workload_type_id)].clone().float().clamp(0.0, 1.0), (
            "raw_query_local_utility_final_label_for_loss"
        )
    return scaled_training_target_for_type(labels, labelled_mask, int(workload_type_id)), (
        "scaled_training_target_for_loss"
    )


historical_prior_support_mask = _historical_prior_support_mask
require_validation_inputs = _require_validation_inputs
canonical_segment_ids_for_boundaries = _canonical_segment_ids_for_boundaries
