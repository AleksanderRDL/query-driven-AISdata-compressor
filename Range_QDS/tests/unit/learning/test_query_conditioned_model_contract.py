"""Tests query-conditioned model input contracts. See models/README.md for details."""

from __future__ import annotations

import pytest
import torch

from models.trajectory_qds_model import TrajectoryQDSModel


def test_query_conditioned_model_requires_query_type_ids() -> None:
    """Assert query-conditioned attention rejects missing query type IDs."""
    model = TrajectoryQDSModel(point_dim=7, query_dim=12)
    model.train()
    points = torch.randn(1, 32, 7)
    queries = torch.randn(8, 12)
    with pytest.raises(RuntimeError):
        _ = model(points=points, queries=queries, query_type_ids=None)
