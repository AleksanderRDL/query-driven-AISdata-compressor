"""Typed query execution against flattened or trajectory AIS data. See workloads/README.md for details."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from data_preparation.trajectory_index import (
    default_boundaries,
    trajectory_ids_intersecting_indices,
)
from workloads.query_types import TypedQuery, validated_range_query_params
from workloads.range_geometry import points_in_range_box


def execute_range_query(
    points: torch.Tensor,
    params: Mapping[str, float],
    boundaries: list[tuple[int, int]] | None = None,
) -> set[int]:
    """Execute a range query returning matching trajectory IDs. See workloads/README.md for details."""
    mask = points_in_range_box(points, params)
    if not mask.any():
        return set()
    return trajectory_ids_intersecting_indices(
        torch.where(mask)[0], default_boundaries(points, boundaries)
    )


def execute_typed_query(
    points: torch.Tensor,
    query: TypedQuery,
    boundaries: list[tuple[int, int]] | None = None,
) -> set[int]:
    """Execute one typed query and return type-specific result object. See workloads/README.md for details."""
    return execute_range_query(points, validated_range_query_params(query), boundaries)
