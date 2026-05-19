"""Focused tests for orchestration target preparation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch

from config.run_config import build_run_config, derive_seed_bundle
from learning.targets.query_useful_v1 import (
    QUERY_USEFUL_V1_QUERY_SHIP_LOCAL_HEADS_TARGET_MODE,
    QUERY_USEFUL_V1_SEGMENT_BUDGET_QUERY_SHIP_MAX_POOL_TARGET_MODE,
)
from orchestration.learning_target_stage import prepare_training_targets
from orchestration.range_runtime_cache import RangeRuntimeCache
from workloads.query_types import pad_query_features
from workloads.typed_workload import TypedQueryWorkload


@contextmanager
def _noop_phase(_name: str) -> Iterator[None]:
    yield


def _empty_workload() -> TypedQueryWorkload:
    return TypedQueryWorkload(
        query_features=torch.zeros((0, 12), dtype=torch.float32),
        typed_queries=[],
        type_ids=torch.zeros((0,), dtype=torch.long),
    )


def _range_workload() -> TypedQueryWorkload:
    typed_queries = [
        {
            "type": "range",
            "params": {
                "lat_min": 0.0,
                "lat_max": 1.0,
                "lon_min": 0.0,
                "lon_max": 1.0,
                "t_start": 0.0,
                "t_end": 2.0,
            },
        }
    ]
    query_features, type_ids = pad_query_features(typed_queries)
    return TypedQueryWorkload(
        query_features=query_features,
        typed_queries=typed_queries,
        type_ids=type_ids,
    )


def _runtime_caches() -> dict[str, RangeRuntimeCache]:
    return {
        "train": RangeRuntimeCache(),
        "eval": RangeRuntimeCache(),
        "selection": RangeRuntimeCache(),
    }


def _minimal_points() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.5, 0.5],
            [2.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )


def test_factorized_target_preparation_does_not_precompute_train_labels() -> None:
    config = build_run_config(
        model_type="workload_blind_range_v2",
        range_training_target_mode="query_useful_v1_factorized",
    )
    workload = _empty_workload()

    prepared = prepare_training_targets(
        config=config,
        seeds=derive_seed_bundle(7),
        train_traj=[_minimal_points()],
        train_points=_minimal_points(),
        train_boundaries=[(0, 3)],
        train_workload=workload,
        train_workload_map={"range": 1.0},
        train_label_workloads=[workload],
        train_label_workload_seeds=[101],
        train_source_ids=None,
        train_mmsis=None,
        selection_workload=None,
        selection_points=None,
        selection_boundaries=None,
        eval_workload_map={"range": 1.0},
        range_runtime_caches=_runtime_caches(),
        phase=_noop_phase,
    )

    assert prepared.train_labels is None
    assert prepared.range_training_target_mode == "query_useful_v1_factorized"
    assert prepared.range_training_target_transform["enabled"] is True
    assert prepared.range_training_target_transform["target_family"] == "QueryUsefulV1Factorized"
    assert prepared.range_training_target_transform["final_success_allowed"] is True
    assert prepared.selection_query_cache is None
    assert prepared.selection_geometry_scores is None


def test_experimental_query_ship_max_pool_target_preparation_is_guarded() -> None:
    config = build_run_config(
        model_type="workload_blind_range_v2",
        range_training_target_mode=QUERY_USEFUL_V1_SEGMENT_BUDGET_QUERY_SHIP_MAX_POOL_TARGET_MODE,
    )
    workload = _empty_workload()

    prepared = prepare_training_targets(
        config=config,
        seeds=derive_seed_bundle(7),
        train_traj=[_minimal_points()],
        train_points=_minimal_points(),
        train_boundaries=[(0, 3)],
        train_workload=workload,
        train_workload_map={"range": 1.0},
        train_label_workloads=[workload],
        train_label_workload_seeds=[101],
        train_source_ids=None,
        train_mmsis=None,
        selection_workload=None,
        selection_points=None,
        selection_boundaries=None,
        eval_workload_map={"range": 1.0},
        range_runtime_caches=_runtime_caches(),
        phase=_noop_phase,
    )

    assert prepared.train_labels is None
    assert (
        prepared.range_training_target_mode
        == QUERY_USEFUL_V1_SEGMENT_BUDGET_QUERY_SHIP_MAX_POOL_TARGET_MODE
    )
    transform = prepared.range_training_target_transform
    assert transform["enabled"] is True
    assert transform["target_family"] == "QueryUsefulV1Factorized"
    assert transform["final_success_allowed"] is False
    assert "Experimental QueryUsefulV1 target mode" in transform["diagnostic_reason"]


def test_experimental_query_ship_local_heads_target_preparation_is_guarded() -> None:
    config = build_run_config(
        model_type="workload_blind_range_v2",
        range_training_target_mode=QUERY_USEFUL_V1_QUERY_SHIP_LOCAL_HEADS_TARGET_MODE,
    )
    workload = _empty_workload()

    prepared = prepare_training_targets(
        config=config,
        seeds=derive_seed_bundle(7),
        train_traj=[_minimal_points()],
        train_points=_minimal_points(),
        train_boundaries=[(0, 3)],
        train_workload=workload,
        train_workload_map={"range": 1.0},
        train_label_workloads=[workload],
        train_label_workload_seeds=[101],
        train_source_ids=None,
        train_mmsis=None,
        selection_workload=None,
        selection_points=None,
        selection_boundaries=None,
        eval_workload_map={"range": 1.0},
        range_runtime_caches=_runtime_caches(),
        phase=_noop_phase,
    )

    assert prepared.train_labels is None
    assert prepared.range_training_target_mode == QUERY_USEFUL_V1_QUERY_SHIP_LOCAL_HEADS_TARGET_MODE
    transform = prepared.range_training_target_transform
    assert transform["enabled"] is True
    assert transform["target_family"] == "QueryUsefulV1Factorized"
    assert transform["final_success_allowed"] is False
    assert "Experimental QueryUsefulV1 target mode" in transform["diagnostic_reason"]


@pytest.mark.parametrize(
    "target_mode",
    [
        "query_useful_v1_factorized_segment_budget_query_hit_ship_blend",
        "query_useful_v1_factorized_segment_budget_final_score_ship_blend",
    ],
)
def test_rejected_experimental_query_useful_target_modes_are_not_prepared(
    target_mode: str,
) -> None:
    config = build_run_config(
        model_type="workload_blind_range_v2",
        range_training_target_mode=target_mode,
    )
    workload = _empty_workload()

    with pytest.raises(RuntimeError, match="QueryUsefulV1 factorized target mode"):
        prepare_training_targets(
            config=config,
            seeds=derive_seed_bundle(7),
            train_traj=[_minimal_points()],
            train_points=_minimal_points(),
            train_boundaries=[(0, 3)],
            train_workload=workload,
            train_workload_map={"range": 1.0},
            train_label_workloads=[workload],
            train_label_workload_seeds=[101],
            train_source_ids=None,
            train_mmsis=None,
            selection_workload=None,
            selection_points=None,
            selection_boundaries=None,
            eval_workload_map={"range": 1.0},
            range_runtime_caches=_runtime_caches(),
            phase=_noop_phase,
        )


def test_target_preparation_builds_selection_query_cache_for_range_validation() -> None:
    config = build_run_config(
        model_type="workload_blind_range_v2",
        range_training_target_mode="query_useful_v1_factorized",
    )
    train_workload = _empty_workload()
    selection_workload = _range_workload()
    runtime_caches = _runtime_caches()
    points = _minimal_points()

    prepared = prepare_training_targets(
        config=config,
        seeds=derive_seed_bundle(7),
        train_traj=[points],
        train_points=points,
        train_boundaries=[(0, 3)],
        train_workload=train_workload,
        train_workload_map={"range": 1.0},
        train_label_workloads=[train_workload],
        train_label_workload_seeds=[101],
        train_source_ids=None,
        train_mmsis=None,
        selection_workload=selection_workload,
        selection_points=points,
        selection_boundaries=[(0, 3)],
        eval_workload_map={"range": 1.0},
        range_runtime_caches=runtime_caches,
        phase=_noop_phase,
    )

    assert prepared.selection_query_cache is not None
    assert runtime_caches["selection"].query_cache is prepared.selection_query_cache
    prepared.selection_query_cache.validate(points, [(0, 3)], selection_workload.typed_queries)
    assert prepared.selection_geometry_scores is None


def test_target_preparation_rejects_invalid_replicate_aggregation() -> None:
    config = build_run_config(
        model_type="workload_blind_range_v2",
        range_training_target_mode="query_useful_v1_factorized",
        range_replicate_target_aggregation="unsupported",
    )
    workload = _empty_workload()

    with pytest.raises(ValueError, match="range_replicate_target_aggregation"):
        prepare_training_targets(
            config=config,
            seeds=derive_seed_bundle(7),
            train_traj=[_minimal_points()],
            train_points=_minimal_points(),
            train_boundaries=[(0, 3)],
            train_workload=workload,
            train_workload_map={"range": 1.0},
            train_label_workloads=[workload],
            train_label_workload_seeds=[101],
            train_source_ids=None,
            train_mmsis=None,
            selection_workload=None,
            selection_points=None,
            selection_boundaries=None,
            eval_workload_map={"range": 1.0},
            range_runtime_caches=_runtime_caches(),
            phase=_noop_phase,
        )


def test_target_preparation_rejects_replicates_for_query_aware_model() -> None:
    config = build_run_config(
        model_type="baseline",
        range_training_target_mode="query_useful_v1_factorized",
    )
    workload = _empty_workload()

    with pytest.raises(RuntimeError, match="workload-blind model types"):
        prepare_training_targets(
            config=config,
            seeds=derive_seed_bundle(7),
            train_traj=[_minimal_points()],
            train_points=_minimal_points(),
            train_boundaries=[(0, 3)],
            train_workload=workload,
            train_workload_map={"range": 1.0},
            train_label_workloads=[workload, workload],
            train_label_workload_seeds=[101, 202],
            train_source_ids=None,
            train_mmsis=None,
            selection_workload=None,
            selection_points=None,
            selection_boundaries=None,
            eval_workload_map={"range": 1.0},
            range_runtime_caches=_runtime_caches(),
            phase=_noop_phase,
        )


def test_target_preparation_rejects_balancing_without_precomputed_labels() -> None:
    config = build_run_config(
        model_type="workload_blind_range_v2",
        range_training_target_mode="query_useful_v1_factorized",
        range_target_balance_mode="trajectory_unit_mass",
    )
    workload = _empty_workload()

    with pytest.raises(RuntimeError, match="range_target_balance_mode"):
        prepare_training_targets(
            config=config,
            seeds=derive_seed_bundle(7),
            train_traj=[_minimal_points()],
            train_points=_minimal_points(),
            train_boundaries=[(0, 3)],
            train_workload=workload,
            train_workload_map={"range": 1.0},
            train_label_workloads=[workload],
            train_label_workload_seeds=[101],
            train_source_ids=None,
            train_mmsis=None,
            selection_workload=None,
            selection_points=None,
            selection_boundaries=None,
            eval_workload_map={"range": 1.0},
            range_runtime_caches=_runtime_caches(),
            phase=_noop_phase,
        )
