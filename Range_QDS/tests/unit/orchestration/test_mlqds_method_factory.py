"""Tests for shared MLQDS method construction."""

from __future__ import annotations

from typing import Any, cast

import torch

from config.run_config import build_run_config
from learning.outputs import TrainingOutputs
from orchestration.mlqds_method_factory import build_mlqds_method
from workloads.query_types import pad_query_features
from workloads.typed_workload import TypedQueryWorkload


def _trained_stub() -> TrainingOutputs:
    return TrainingOutputs(
        model=torch.nn.Linear(1, 1),
        scaler=cast(Any, object()),
        labels=torch.zeros((4, 1), dtype=torch.float32),
        labelled_mask=torch.ones((4, 1), dtype=torch.bool),
        history=[],
        feature_context={},
    )


def _range_workload() -> TypedQueryWorkload:
    typed_queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": 0.0,
                "t_end": 3.0,
            },
        }
    ]
    query_features, type_ids = pad_query_features(typed_queries)
    return TypedQueryWorkload(
        query_features=query_features,
        typed_queries=typed_queries,
        type_ids=type_ids,
    )


def test_build_mlqds_method_carries_selector_scoring_and_runtime_config() -> None:
    config = build_run_config(
        mlqds_score_mode="rank",
        mlqds_score_temperature=0.73,
        mlqds_rank_confidence_weight=0.21,
        mlqds_temporal_fraction=0.34,
        mlqds_diversity_bonus=0.05,
        mlqds_hybrid_mode="fill",
        mlqds_stratified_center_weight=0.13,
        mlqds_min_learned_swaps=3,
        selector_type="learned_segment_budget_v1",
        learned_segment_geometry_gain_weight=0.19,
        learned_segment_allocation_length_support_weight=0.23,
        learned_segment_allocation_weight_floor=0.41,
        learned_segment_score_blend_weight=0.29,
        learned_segment_transfer_calibration_mode="segment_score_allocation_weight_zblend",
        learned_segment_fairness_preallocation=False,
        learned_segment_length_repair_fraction=0.17,
        learned_segment_length_repair_score_protection_fraction=0.11,
        learned_segment_length_support_blend_weight=0.07,
        mlqds_range_geometry_blend=0.37,
        inference_batch_size=9,
        amp_mode="bf16",
    )
    workload = _range_workload()
    range_geometry_scores = torch.arange(4, dtype=torch.float32)

    method = build_mlqds_method(
        name="MLQDS",
        trained=_trained_stub(),
        workload=workload,
        workload_map={"range": 1.0},
        config=config,
        trajectory_mmsis=[111, 222],
        range_geometry_scores=range_geometry_scores,
        inference_device="cpu",
        inference_batch_size=5,
        amp_mode="off",
    )

    assert method.name == "MLQDS"
    assert method.workload is workload
    assert method.workload_type == "range"
    assert method.score_mode == "rank"
    assert method.score_temperature == 0.73
    assert method.rank_confidence_weight == 0.21
    assert method.temporal_fraction == 0.34
    assert method.diversity_bonus == 0.05
    assert method.hybrid_mode == "fill"
    assert method.stratified_center_weight == 0.13
    assert method.min_learned_swaps == 3
    assert method.selector_type == "learned_segment_budget_v1"
    assert method.learned_segment_geometry_gain_weight == 0.19
    assert method.learned_segment_allocation_length_support_weight == 0.23
    assert method.learned_segment_allocation_weight_floor == 0.41
    assert method.learned_segment_score_blend_weight == 0.29
    assert method.learned_segment_transfer_calibration_mode == (
        "segment_score_allocation_weight_zblend"
    )
    assert method.learned_segment_fairness_preallocation is False
    assert method.learned_segment_length_repair_fraction == 0.17
    assert method.learned_segment_length_repair_score_protection_fraction == 0.11
    assert method.learned_segment_length_support_blend_weight == 0.07
    assert method.range_geometry_blend == 0.37
    assert method.range_geometry_scores is range_geometry_scores
    assert method.trajectory_mmsis == [111, 222]
    assert method.inference_device == "cpu"
    assert method.inference_batch_size == 5
    assert method.amp_mode == "off"

    diagnostic_method = build_mlqds_method(
        name="MLQDS_diagnostic",
        trained=_trained_stub(),
        workload=workload,
        workload_map={"range": 1.0},
        config=config,
        range_geometry_blend=0.0,
    )
    assert diagnostic_method.range_geometry_blend == 0.0
    assert diagnostic_method.inference_batch_size == 9
    assert diagnostic_method.amp_mode == "bf16"
