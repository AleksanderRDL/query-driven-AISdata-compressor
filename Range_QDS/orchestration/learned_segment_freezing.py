"""Config-aware helpers for freezing learned-segment diagnostic masks."""

from __future__ import annotations

import torch

from config.run_config import RunConfig
from orchestration.selector_diagnostics import learned_segment_frozen_method
from scoring.methods import FrozenMaskMethod


def learned_segment_frozen_method_from_config(
    *,
    config: RunConfig,
    name: str,
    scores: torch.Tensor,
    boundaries: list[tuple[int, int]],
    compression_ratio: float | None = None,
    segment_scores: torch.Tensor | None = None,
    segment_point_scores: torch.Tensor | None = None,
    path_length_support_scores: torch.Tensor | None = None,
    points: torch.Tensor | None = None,
    learned_segment_geometry_gain_weight: float | None = None,
    learned_segment_allocation_length_support_weight: float | None = None,
    learned_segment_allocation_weight_floor: float | None = None,
    learned_segment_score_blend_weight: float | None = None,
    learned_segment_transfer_calibration_mode: str | None = None,
    learned_segment_fairness_preallocation: bool | None = None,
    learned_segment_length_repair_fraction: float | None = None,
    learned_segment_length_repair_score_protection_fraction: float | None = None,
    learned_segment_length_support_blend_weight: float | None = None,
) -> FrozenMaskMethod:
    """Freeze a learned-segment mask using RunConfig defaults plus explicit overrides."""
    model_config = config.model
    return learned_segment_frozen_method(
        name=name,
        scores=scores,
        boundaries=boundaries,
        compression_ratio=(
            float(model_config.compression_ratio)
            if compression_ratio is None
            else float(compression_ratio)
        ),
        segment_scores=segment_scores,
        segment_point_scores=segment_point_scores,
        path_length_support_scores=path_length_support_scores,
        points=points,
        learned_segment_geometry_gain_weight=(
            float(model_config.learned_segment_geometry_gain_weight)
            if learned_segment_geometry_gain_weight is None
            else float(learned_segment_geometry_gain_weight)
        ),
        learned_segment_allocation_length_support_weight=(
            float(model_config.learned_segment_allocation_length_support_weight)
            if learned_segment_allocation_length_support_weight is None
            else float(learned_segment_allocation_length_support_weight)
        ),
        learned_segment_allocation_weight_floor=(
            float(model_config.learned_segment_allocation_weight_floor)
            if learned_segment_allocation_weight_floor is None
            else float(learned_segment_allocation_weight_floor)
        ),
        learned_segment_score_blend_weight=(
            float(model_config.learned_segment_score_blend_weight)
            if learned_segment_score_blend_weight is None
            else float(learned_segment_score_blend_weight)
        ),
        learned_segment_transfer_calibration_mode=(
            str(model_config.learned_segment_transfer_calibration_mode)
            if learned_segment_transfer_calibration_mode is None
            else str(learned_segment_transfer_calibration_mode)
        ),
        learned_segment_fairness_preallocation=(
            bool(model_config.learned_segment_fairness_preallocation)
            if learned_segment_fairness_preallocation is None
            else bool(learned_segment_fairness_preallocation)
        ),
        learned_segment_length_repair_fraction=(
            float(model_config.learned_segment_length_repair_fraction)
            if learned_segment_length_repair_fraction is None
            else float(learned_segment_length_repair_fraction)
        ),
        learned_segment_length_repair_score_protection_fraction=(
            float(model_config.learned_segment_length_repair_score_protection_fraction)
            if learned_segment_length_repair_score_protection_fraction is None
            else float(learned_segment_length_repair_score_protection_fraction)
        ),
        learned_segment_length_support_blend_weight=(
            float(model_config.learned_segment_length_support_blend_weight)
            if learned_segment_length_support_blend_weight is None
            else float(learned_segment_length_support_blend_weight)
        ),
    )
