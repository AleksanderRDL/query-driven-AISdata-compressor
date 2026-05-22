"""Training target and query-prior setup for model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from config.run_config import ModelConfig
from learning.importance_labels import compute_typed_importance_labels
from learning.model_features import WORKLOAD_BLIND_RANGE_MODEL_TYPE, build_model_point_features
from learning.model_training_helpers import canonical_segment_ids_for_boundaries
from learning.query_prior_fields import build_train_query_prior_fields
from learning.targets.query_local_utility import (
    QUERY_LOCAL_UTILITY_HEAD_NAMES,
    QUERY_LOCAL_UTILITY_TARGET_MODES,
    build_query_local_utility_targets,
)
from workloads.generation.workload_profiles import RANGE_QUERY_MIX_PROFILE_ID
from workloads.query_types import NUM_QUERY_TYPES
from workloads.typed_workload import TypedQueryWorkload


@dataclass(frozen=True)
class TrainingTargetInputs:
    all_points: torch.Tensor
    train_point_source_ids: torch.Tensor | None
    prior_queries: list[dict[str, Any]]
    labels: torch.Tensor
    labelled_mask: torch.Tensor
    factorized_targets: torch.Tensor | None
    factorized_mask: torch.Tensor | None
    factorized_target_diagnostics: dict[str, Any]
    canonical_segment_ids: torch.Tensor | None
    range_training_target_mode: str
    query_prior_field: dict[str, Any] | None
    points: torch.Tensor
    point_dim: int


def build_training_target_inputs(
    *,
    train_trajectories: list[torch.Tensor],
    train_boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    model_config: ModelConfig,
    precomputed_labels: tuple[torch.Tensor, torch.Tensor] | None,
    train_trajectory_source_ids: list[int] | None,
    train_trajectory_mmsis: list[int] | None,
    query_prior_workloads: list[TypedQueryWorkload] | None,
    query_prior_workload_seeds: list[int] | None,
) -> TrainingTargetInputs:
    all_points = torch.cat(train_trajectories, dim=0)
    train_point_source_ids: torch.Tensor | None = None
    if train_trajectory_source_ids is not None:
        if len(train_trajectory_source_ids) != len(train_boundaries):
            raise ValueError(
                "train_trajectory_source_ids must match train_boundaries length: "
                f"got {len(train_trajectory_source_ids)} ids for {len(train_boundaries)} boundaries."
            )
        train_point_source_ids = torch.empty((int(all_points.shape[0]),), dtype=torch.long)
        for source_id, (start, end) in zip(
            train_trajectory_source_ids, train_boundaries, strict=True
        ):
            if int(source_id) < 0:
                raise ValueError("train_trajectory_source_ids must be non-negative.")
            train_point_source_ids[start:end] = int(source_id)
    prior_workloads = list(query_prior_workloads or [workload])
    prior_queries: list[dict[str, Any]] = []
    for prior_workload in prior_workloads:
        prior_queries.extend(prior_workload.typed_queries)

    factorized_targets: torch.Tensor | None = None
    factorized_mask: torch.Tensor | None = None
    factorized_target_diagnostics: dict[str, Any] = {}
    canonical_segment_ids: torch.Tensor | None = None
    range_training_target_mode = str(
        getattr(model_config, "range_training_target_mode", "")
    ).lower()
    if range_training_target_mode in QUERY_LOCAL_UTILITY_TARGET_MODES:
        factorized_bundle = build_query_local_utility_targets(
            points=all_points,
            boundaries=train_boundaries,
            typed_queries=prior_queries,
            target_mode=range_training_target_mode,
        )
        labels = factorized_bundle.labels
        labelled_mask = factorized_bundle.labelled_mask
        factorized_targets = factorized_bundle.head_targets
        factorized_mask = factorized_bundle.head_mask
        factorized_target_diagnostics = factorized_bundle.diagnostics
        factorized_segment_size = int(factorized_target_diagnostics.get("segment_size_points", 32))
        canonical_segment_ids = canonical_segment_ids_for_boundaries(
            point_count=int(all_points.shape[0]),
            boundaries=train_boundaries,
            segment_size=factorized_segment_size,
        )
        factorized_target_diagnostics["canonical_segment_ids_available"] = True
        factorized_target_diagnostics["canonical_segment_size_points"] = int(
            factorized_segment_size
        )
        factorized_target_diagnostics["canonical_segment_count"] = int(
            torch.unique(canonical_segment_ids[canonical_segment_ids >= 0]).numel()
        )
        factorized_target_diagnostics["segment_budget_target_training"] = (
            "point_repeated_plus_canonical_segment_level_listwise_loss"
        )
    elif precomputed_labels is None:
        labels, labelled_mask = compute_typed_importance_labels(
            points=all_points,
            boundaries=train_boundaries,
            typed_queries=workload.typed_queries,
            range_label_mode=str(getattr(model_config, "range_label_mode", "point_f1")),
            range_boundary_prior_weight=float(
                getattr(model_config, "range_boundary_prior_weight", 0.0)
            ),
        )
    else:
        labels, labelled_mask = precomputed_labels
        expected_shape = (all_points.shape[0], NUM_QUERY_TYPES)
        if labels.shape != expected_shape or labelled_mask.shape != expected_shape:
            raise ValueError(
                "precomputed_labels must match flattened training points and query type count: "
                f"expected {expected_shape}, got labels={tuple(labels.shape)} mask={tuple(labelled_mask.shape)}"
            )

    query_prior_field: dict[str, Any] | None = None
    if str(model_config.model_type).lower() == WORKLOAD_BLIND_RANGE_MODEL_TYPE:
        prior_seed = None
        if query_prior_workload_seeds:
            prior_seed = int(query_prior_workload_seeds[0])
        behavior_prior_values = None
        if factorized_targets is not None:
            try:
                behavior_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index(
                    "conditional_behavior_utility"
                )
                behavior_prior_values = factorized_targets[:, behavior_idx]
            except ValueError:
                behavior_prior_values = None
        query_prior_field = build_train_query_prior_fields(
            points=all_points,
            boundaries=train_boundaries,
            typed_queries=prior_queries,
            labels=labels,
            behavior_values=behavior_prior_values,
            workload_profile_id=str(
                (workload.generation_diagnostics or {})
                .get("query_generation", {})
                .get(
                    "workload_profile_id",
                    RANGE_QUERY_MIX_PROFILE_ID,
                )
            ),
            train_workload_seed=prior_seed,
            grid_bins=int(getattr(model_config, "query_prior_grid_bins", 64)),
            smoothing_passes=int(getattr(model_config, "query_prior_smoothing_passes", 2)),
            out_of_extent_sampling="nearest",
        )
    points = build_model_point_features(
        all_points,
        workload,
        model_config.model_type,
        boundaries=train_boundaries,
        trajectory_mmsis=train_trajectory_mmsis,
        query_prior_field=query_prior_field,
    )
    point_dim = int(points.shape[1])
    return TrainingTargetInputs(
        all_points=all_points,
        train_point_source_ids=train_point_source_ids,
        prior_queries=prior_queries,
        labels=labels,
        labelled_mask=labelled_mask,
        factorized_targets=factorized_targets,
        factorized_mask=factorized_mask,
        factorized_target_diagnostics=factorized_target_diagnostics,
        canonical_segment_ids=canonical_segment_ids,
        range_training_target_mode=range_training_target_mode,
        query_prior_field=query_prior_field,
        points=points,
        point_dim=point_dim,
    )
