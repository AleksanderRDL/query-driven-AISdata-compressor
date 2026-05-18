"""Shared model construction for training and checkpoint loading."""

from __future__ import annotations

from typing import Any, cast

import torch

from config.run_config import ModelConfig
from learning.model_features import (
    HISTORICAL_PRIOR_MODEL_TYPES,
    NONPARAMETRIC_HISTORICAL_PRIOR_MODEL_TYPES,
    SUPPORTED_MODEL_TYPES,
    WORKLOAD_BLIND_RANGE_V2_MODEL_TYPE,
    is_workload_blind_model_type,
)
from models.historical_prior_qds_model import (
    HistoricalPriorRangeQDSModel,
    HistoricalPriorStudentRangeQDSModel,
)
from models.trajectory_qds_model import TrajectoryQDSModel
from models.workload_blind_qds_model import SegmentContextRangeQDSModel, WorkloadBlindRangeQDSModel
from models.workload_blind_range_v2 import WorkloadBlindRangeV2Model

HistoricalPriorModel = HistoricalPriorRangeQDSModel | HistoricalPriorStudentRangeQDSModel


def validate_supported_model_type(model_type: str) -> str:
    """Return a normalized supported model type or raise the canonical error."""
    normalized = str(model_type).lower()
    if normalized not in SUPPORTED_MODEL_TYPES:
        choices = ", ".join(SUPPORTED_MODEL_TYPES)
        raise ValueError(f"Unsupported checkpoint model_type={normalized!r}; choices: {choices}.")
    return normalized


def qds_model_class(model_type: str) -> type[torch.nn.Module]:
    """Return the concrete model class for a supported model type."""
    normalized = validate_supported_model_type(model_type)
    if normalized in NONPARAMETRIC_HISTORICAL_PRIOR_MODEL_TYPES:
        return HistoricalPriorRangeQDSModel
    if normalized == "historical_prior_student":
        return HistoricalPriorStudentRangeQDSModel
    if normalized == "segment_context_range":
        return SegmentContextRangeQDSModel
    if normalized == WORKLOAD_BLIND_RANGE_V2_MODEL_TYPE:
        return WorkloadBlindRangeV2Model
    if is_workload_blind_model_type(normalized):
        return WorkloadBlindRangeQDSModel
    return TrajectoryQDSModel


def qds_model_constructor_kwargs(
    *,
    model_type: str,
    model_config: ModelConfig,
    point_dim: int,
    query_dim: int,
    prior_feature_count: int = 0,
) -> dict[str, Any]:
    """Return constructor kwargs shared by training and checkpoint loading."""
    normalized = validate_supported_model_type(model_type)
    kwargs: dict[str, Any] = {
        "point_dim": int(point_dim),
        "query_dim": int(query_dim),
        "embed_dim": int(model_config.embed_dim),
        "num_heads": int(model_config.num_heads),
        "num_layers": int(model_config.num_layers),
        "type_embed_dim": int(model_config.type_embed_dim),
        "query_chunk_size": int(model_config.query_chunk_size),
        "dropout": float(model_config.dropout),
    }
    if normalized in HISTORICAL_PRIOR_MODEL_TYPES:
        kwargs.update(
            {
                "historical_prior_k": int(getattr(model_config, "historical_prior_k", 32)),
                "historical_prior_clock_weight": float(
                    getattr(model_config, "historical_prior_clock_weight", 0.0)
                ),
                "historical_prior_mmsi_weight": float(
                    getattr(model_config, "historical_prior_mmsi_weight", 1.0)
                ),
                "historical_prior_density_weight": float(
                    getattr(model_config, "historical_prior_density_weight", 1.0)
                ),
                "historical_prior_min_target": float(
                    getattr(model_config, "historical_prior_min_target", 0.0)
                ),
                "historical_prior_source_aggregation": str(
                    getattr(model_config, "historical_prior_source_aggregation", "none")
                ),
                "prior_feature_count": max(0, int(prior_feature_count)),
            }
        )
    return kwargs


def build_qds_model(
    *,
    model_type: str,
    model_config: ModelConfig,
    point_dim: int,
    query_dim: int,
    prior_feature_count: int = 0,
) -> torch.nn.Module:
    """Build a QDS model with the shared constructor mapping."""
    model_cls = qds_model_class(model_type)
    kwargs = qds_model_constructor_kwargs(
        model_type=model_type,
        model_config=model_config,
        point_dim=point_dim,
        query_dim=query_dim,
        prior_feature_count=prior_feature_count,
    )
    return cast(torch.nn.Module, model_cls(**kwargs))


def require_historical_prior_model(
    model: torch.nn.Module,
    *,
    model_type: str,
) -> HistoricalPriorModel:
    """Return a historical-prior model instance or raise a construction invariant error."""
    if not isinstance(model, (HistoricalPriorRangeQDSModel, HistoricalPriorStudentRangeQDSModel)):
        raise TypeError(f"{model_type} did not build a historical-prior model.")
    return model
