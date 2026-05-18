"""Tests for shared QDS model construction."""

from __future__ import annotations

from typing import Any, cast

import pytest

from config.run_config import build_run_config
from learning.model_factory import (
    build_qds_model,
    qds_model_constructor_kwargs,
    validate_supported_model_type,
)
from learning.model_features import SUPPORTED_MODEL_TYPES
from models.historical_prior_qds_model import (
    HistoricalPriorRangeQDSModel,
    HistoricalPriorStudentRangeQDSModel,
)
from models.trajectory_qds_model import TrajectoryQDSModel
from models.workload_blind_qds_model import SegmentContextRangeQDSModel, WorkloadBlindRangeQDSModel
from models.workload_blind_range_v2 import WorkloadBlindRangeV2Model


def test_build_qds_model_maps_every_supported_model_type_to_expected_class() -> None:
    expected_classes = {
        "baseline": TrajectoryQDSModel,
        "range_aware": TrajectoryQDSModel,
        "workload_blind_range": WorkloadBlindRangeQDSModel,
        "range_prior": WorkloadBlindRangeQDSModel,
        "range_prior_clock_density": WorkloadBlindRangeQDSModel,
        "segment_context_range": SegmentContextRangeQDSModel,
        "historical_prior": HistoricalPriorRangeQDSModel,
        "historical_prior_mmsi": HistoricalPriorRangeQDSModel,
        "historical_prior_student": HistoricalPriorStudentRangeQDSModel,
        "workload_blind_range_v2": WorkloadBlindRangeV2Model,
    }
    assert set(expected_classes) == set(SUPPORTED_MODEL_TYPES)

    for model_type, expected_cls in expected_classes.items():
        config = build_run_config(
            model_type=model_type,
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            query_chunk_size=64,
        )
        model = build_qds_model(
            model_type=model_type,
            model_config=config.model,
            point_dim=32,
            query_dim=12,
            prior_feature_count=3,
        )

        assert isinstance(model, expected_cls)
        model_attrs = cast(Any, model)
        assert int(model_attrs.point_dim) == 32
        assert int(model_attrs.query_dim) == 12
        assert int(model_attrs.embed_dim) == 16
        assert int(model_attrs.query_chunk_size) == 64


def test_historical_prior_constructor_kwargs_include_configured_prior_controls() -> None:
    config = build_run_config(
        model_type="historical_prior",
        historical_prior_k=7,
        historical_prior_clock_weight=0.25,
        historical_prior_mmsi_weight=2.5,
        historical_prior_density_weight=1.5,
        historical_prior_min_target=0.2,
        historical_prior_source_aggregation="mean",
    )

    kwargs = qds_model_constructor_kwargs(
        model_type="historical_prior",
        model_config=config.model,
        point_dim=19,
        query_dim=12,
        prior_feature_count=5,
    )

    assert kwargs["historical_prior_k"] == 7
    assert kwargs["historical_prior_clock_weight"] == 0.25
    assert kwargs["historical_prior_mmsi_weight"] == 2.5
    assert kwargs["historical_prior_density_weight"] == 1.5
    assert kwargs["historical_prior_min_target"] == 0.2
    assert kwargs["historical_prior_source_aggregation"] == "mean"
    assert kwargs["prior_feature_count"] == 5


def test_validate_supported_model_type_rejects_unknown_model_type() -> None:
    with pytest.raises(ValueError, match="Unsupported checkpoint model_type"):
        validate_supported_model_type("removed_experimental_model")
