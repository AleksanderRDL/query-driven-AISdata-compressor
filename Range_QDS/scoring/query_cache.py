"""Scoring query cache and retained-independent range audit support."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class RangeTrajectoryAuditSupport:
    """Retained-independent per-trajectory support for one range query."""

    trajectory_id: int
    start: int
    end: int
    in_offsets_cpu: torch.Tensor
    turn_weights_cpu: torch.Tensor
    distance_offsets_km_cpu: torch.Tensor
    full_time_span: float
    full_length_km: float


@dataclass(frozen=True)
class RangeQueryAuditSupport:
    """Retained-independent support reused across range audit methods and ratios."""

    range_mask: torch.Tensor
    trajectories: tuple[RangeTrajectoryAuditSupport, ...]


def _points_cache_token(points: torch.Tensor) -> tuple[int, int, tuple[int, ...], str, str]:
    """Return an identity token for caller-owned scoring caches."""
    data_ptr = int(points.data_ptr()) if points.numel() > 0 else 0
    return (
        id(points),
        data_ptr,
        tuple(int(dim) for dim in points.shape),
        str(points.device),
        str(points.dtype),
    )


def _queries_cache_token(typed_queries: list[dict]) -> tuple[int, int, tuple[int, ...]]:
    """Return an identity token for a typed-query workload list."""
    return (id(typed_queries), len(typed_queries), tuple(id(query) for query in typed_queries))


@dataclass
class ScoringQueryCache:
    """Caller-owned cache for full-data query results during repeated method scoring."""

    points_token: tuple[int, int, tuple[int, ...], str, str]
    boundaries_key: tuple[tuple[int, int], ...]
    queries_token: tuple[int, int, tuple[int, ...]]
    support_masks: dict[int, torch.Tensor] = field(default_factory=dict)
    range_audit_supports: dict[int, RangeQueryAuditSupport] = field(default_factory=dict)

    @classmethod
    def for_workload(
        cls,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        typed_queries: list[dict],
    ) -> ScoringQueryCache:
        """Build a cache scoped to exactly one points/boundaries/workload object."""
        return cls(
            points_token=_points_cache_token(points),
            boundaries_key=tuple((int(start), int(end)) for start, end in boundaries),
            queries_token=_queries_cache_token(typed_queries),
        )

    def validate(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        typed_queries: list[dict],
    ) -> None:
        """Fail fast if this cache is reused for a different scoring scope."""
        if (
            self.points_token != _points_cache_token(points)
            or self.boundaries_key != tuple((int(start), int(end)) for start, end in boundaries)
            or self.queries_token != _queries_cache_token(typed_queries)
        ):
            raise ValueError(
                "ScoringQueryCache was built for different points, boundaries, or typed queries."
            )

    def get_support_mask(
        self, query_index: int, builder: Callable[[], torch.Tensor]
    ) -> torch.Tensor:
        """Return a cached full-data support mask."""
        if query_index not in self.support_masks:
            self.support_masks[query_index] = builder()
        return self.support_masks[query_index]

    def get_range_audit_support(
        self,
        query_index: int,
        builder: Callable[[], RangeQueryAuditSupport],
    ) -> RangeQueryAuditSupport:
        """Return cached retained-independent range-audit support."""
        if query_index not in self.range_audit_supports:
            self.range_audit_supports[query_index] = builder()
        return self.range_audit_supports[query_index]
