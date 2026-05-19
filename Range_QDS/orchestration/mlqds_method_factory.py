"""Shared MLQDS method construction for orchestration stages."""

from __future__ import annotations

import torch

from config.run_config import RunConfig
from learning.outputs import TrainingOutputs
from scoring.methods import MLQDSMethod
from workloads.query_types import single_workload_type
from workloads.typed_workload import TypedQueryWorkload


def build_mlqds_method(
    *,
    name: str,
    trained: TrainingOutputs,
    workload: TypedQueryWorkload,
    workload_map: dict[str, float],
    config: RunConfig,
    trajectory_mmsis: list[int] | None = None,
    range_geometry_blend: float | None = None,
    range_geometry_scores: torch.Tensor | None = None,
    inference_device: str | torch.device | None = None,
    inference_batch_size: int | None = None,
    amp_mode: str | None = None,
) -> MLQDSMethod:
    """Build an MLQDS scoring method from the active run config."""
    return MLQDSMethod(
        name=name,
        trained=trained,
        workload=workload,
        workload_type=single_workload_type(workload_map),
        score_mode=config.model.mlqds_score_mode,
        score_temperature=config.model.mlqds_score_temperature,
        rank_confidence_weight=config.model.mlqds_rank_confidence_weight,
        temporal_fraction=config.model.mlqds_temporal_fraction,
        diversity_bonus=config.model.mlqds_diversity_bonus,
        hybrid_mode=config.model.mlqds_hybrid_mode,
        stratified_center_weight=config.model.mlqds_stratified_center_weight,
        min_learned_swaps=config.model.mlqds_min_learned_swaps,
        selector_type=config.model.selector_type,
        learned_segment_geometry_gain_weight=config.model.learned_segment_geometry_gain_weight,
        learned_segment_allocation_length_support_weight=(
            config.model.learned_segment_allocation_length_support_weight
        ),
        learned_segment_allocation_weight_floor=(
            config.model.learned_segment_allocation_weight_floor
        ),
        learned_segment_score_blend_weight=config.model.learned_segment_score_blend_weight,
        learned_segment_transfer_calibration_mode=(
            config.model.learned_segment_transfer_calibration_mode
        ),
        learned_segment_fairness_preallocation=config.model.learned_segment_fairness_preallocation,
        learned_segment_length_repair_fraction=config.model.learned_segment_length_repair_fraction,
        learned_segment_length_repair_score_protection_fraction=(
            config.model.learned_segment_length_repair_score_protection_fraction
        ),
        learned_segment_length_support_blend_weight=(
            config.model.learned_segment_length_support_blend_weight
        ),
        range_geometry_blend=(
            config.model.mlqds_range_geometry_blend
            if range_geometry_blend is None
            else float(range_geometry_blend)
        ),
        range_geometry_scores=range_geometry_scores,
        trajectory_mmsis=trajectory_mmsis,
        inference_device=inference_device,
        inference_batch_size=(
            config.model.inference_batch_size
            if inference_batch_size is None
            else int(inference_batch_size)
        ),
        amp_mode=config.model.amp_mode if amp_mode is None else amp_mode,
    )
