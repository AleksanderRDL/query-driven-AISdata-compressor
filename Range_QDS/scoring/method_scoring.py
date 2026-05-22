"""Method scoring and fixed-width results table helpers. See scoring/README.md for details."""

from __future__ import annotations

import time
from typing import Any, cast

import torch

from scoring.methods import Method
from scoring.metrics import (
    MethodScore,
    compute_geometric_distortion,
    compute_length_preservation,
)
from scoring.query_cache import ScoringQueryCache
from scoring.query_local_utility import query_local_utility_from_range_audit
from scoring.range_audit_scoring import (
    RANGE_QUERY_COMPONENT_KEYS as RANGE_QUERY_COMPONENT_KEYS,
)
from scoring.range_audit_scoring import (
    _mean,
    _range_point_f1,
)
from scoring.range_audit_scoring import (
    score_range_boundary_preservation as score_range_boundary_preservation,
)
from scoring.range_audit_scoring import (
    score_range_usefulness as score_range_usefulness,
)
from workloads.query_types import normalize_pure_workload_map
from workloads.range_geometry import points_in_range_box


def score_retained_mask(
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    retained_mask: torch.Tensor,
    typed_queries: list[dict],
    workload_map: dict[str, float],
    query_cache: ScoringQueryCache | None = None,
) -> tuple[float, dict[str, float], float, dict[str, float]]:
    """Score a precomputed retained mask with the final query-F1 semantics.

    Returns (aggregate_answer_f1, per_type_answer_f1, aggregate_combined,
    per_type_combined). Range workloads use point-subset F1 over each query box;
    answer and combined scores are identical under the range-only contract.
    """
    if query_cache is not None:
        query_cache.validate(points, boundaries, typed_queries)

    workload_weights = normalize_pure_workload_map(workload_map)
    scores: list[float] = []
    for query_index, query in enumerate(typed_queries):
        qtype = str(query.get("type", "")).lower()
        if qtype != "range":
            raise ValueError(
                f"Only range queries are supported for range scoring; got query type: {qtype}"
            )
        if query_cache is not None:
            range_mask = query_cache.get_support_mask(
                query_index,
                lambda query=query: points_in_range_box(points, query["params"]),
            )
        else:
            range_mask = points_in_range_box(points, query["params"])
        scores.append(_range_point_f1(retained_mask, range_mask))

    range_score = _mean(scores)
    aggregate = float(workload_weights["range"] * range_score)
    per_type = {"range": range_score}
    return aggregate, per_type, aggregate, dict(per_type)


def _retained_point_gap_stats(
    retained_mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
) -> tuple[float, float, float]:
    """Return average and max original-index gaps between retained points."""
    total_gap = 0.0
    total_norm_gap = 0.0
    max_gap = 0.0
    gap_count = 0
    for start, end in boundaries:
        n = int(end - start)
        if n <= 1:
            continue
        offsets = torch.where(retained_mask[start:end])[0].float()
        if offsets.numel() < 2:
            continue
        gaps = offsets[1:] - offsets[:-1]
        denom = float(max(1, n - 1))
        total_gap += float(gaps.sum().item())
        total_norm_gap += float((gaps / denom).sum().item())
        max_gap = max(max_gap, float(gaps.max().item()))
        gap_count += int(gaps.numel())

    if gap_count <= 0:
        return 0.0, 0.0, 0.0
    return float(total_gap / gap_count), float(total_norm_gap / gap_count), float(max_gap)


def _endpoint_sanity(retained_mask: torch.Tensor, boundaries: list[tuple[int, int]]) -> float:
    """Return the fraction of eligible trajectories retaining both endpoints."""
    eligible = 0
    passing = 0
    for start, end in boundaries:
        n = int(end - start)
        if n <= 1:
            continue
        retained_count = int(retained_mask[start:end].sum().item())
        if retained_count < 2:
            continue
        eligible += 1
        if bool(retained_mask[start].item()) and bool(retained_mask[end - 1].item()):
            passing += 1
    if eligible <= 0:
        return 1.0
    return float(passing / eligible)


def score_method(
    method: Method,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict],
    workload_map: dict[str, float],
    compression_ratio: float,
    return_mask: bool = False,
    query_cache: ScoringQueryCache | None = None,
) -> MethodScore:
    """Evaluate one simplification method on typed queries at matched ratio. See scoring/README.md for details."""
    t0 = time.time()
    retained_mask = method.simplify(points, boundaries, compression_ratio)
    measured_latency_ms = (time.time() - t0) * 1000.0
    latency_ms = float(getattr(method, "latency_ms", measured_latency_ms) or measured_latency_ms)

    range_only = bool(typed_queries) and all(
        str(query.get("type", "")).lower() == "range" for query in typed_queries
    )
    range_audit: dict[str, Any] | None = None
    if range_only:
        range_audit = score_range_usefulness(
            points=points,
            boundaries=boundaries,
            retained_mask=retained_mask,
            typed_queries=typed_queries,
            query_cache=query_cache,
        )
        range_point = float(range_audit.get("range_point_f1", 0.0))
        aggregate = range_point
        aggregate_combined = range_point
        per_type = {"range": range_point}
        per_type_combined = dict(per_type)
    else:
        aggregate, per_type, aggregate_combined, per_type_combined = score_retained_mask(
            points=points,
            boundaries=boundaries,
            retained_mask=retained_mask,
            typed_queries=typed_queries,
            workload_map=workload_map,
            query_cache=query_cache,
        )
    comp = float(retained_mask.float().mean().item())
    avg_gap, avg_norm_gap, max_gap = _retained_point_gap_stats(retained_mask, boundaries)
    geometric = compute_geometric_distortion(points, boundaries, retained_mask)
    avg_length_preserved = compute_length_preservation(points, boundaries, retained_mask)
    combined = float(aggregate) * max(0.0, min(1.0, avg_length_preserved))
    if range_audit is None:
        range_audit = score_range_usefulness(
            points=points,
            boundaries=boundaries,
            retained_mask=retained_mask,
            typed_queries=typed_queries,
            query_cache=query_cache,
        )
    endpoint_sanity = _endpoint_sanity(retained_mask, boundaries)
    range_audit["endpoint_sanity"] = endpoint_sanity
    boundary_f1 = float(range_audit.get("range_entry_exit_f1", 0.0))
    query_local_utility = query_local_utility_from_range_audit(
        range_audit,
        length_preservation=avg_length_preserved,
        avg_sed_km=float(geometric.get("avg_sed_km", 0.0)),
        endpoint_sanity=endpoint_sanity,
    )
    range_audit.update(query_local_utility)
    query_local_utility_components_raw = query_local_utility.get(
        "query_local_utility_components", {}
    )
    query_local_utility_components = (
        {str(key): float(value) for key, value in query_local_utility_components_raw.items()}
        if isinstance(query_local_utility_components_raw, dict)
        else {}
    )
    query_local_utility_score = float(
        cast(Any, query_local_utility.get("query_local_utility_score", 0.0)) or 0.0
    )
    query_local_utility_schema = int(
        cast(Any, query_local_utility.get("query_local_utility_schema_version", 0)) or 0
    )

    return MethodScore(
        aggregate_f1=float(aggregate),
        per_type_f1=per_type,
        aggregate_combined_f1=float(aggregate_combined),
        per_type_combined_f1=per_type_combined,
        compression_ratio=comp,
        latency_ms=latency_ms,
        avg_retained_point_gap=avg_gap,
        avg_retained_point_gap_norm=avg_norm_gap,
        max_retained_point_gap=max_gap,
        geometric_distortion=geometric,
        avg_length_preserved=avg_length_preserved,
        combined_query_shape_score=combined,
        query_point_recall=float(range_audit.get("query_point_recall", 0.0)),
        range_point_f1=float(range_audit.get("range_point_f1", per_type.get("range", 0.0))),
        range_ship_f1=float(range_audit.get("range_ship_f1", 0.0)),
        range_ship_coverage=float(range_audit.get("range_ship_coverage", 0.0)),
        range_entry_exit_f1=boundary_f1,
        range_crossing_f1=float(range_audit.get("range_crossing_f1", 0.0)),
        range_temporal_coverage=float(range_audit.get("range_temporal_coverage", 0.0)),
        range_gap_coverage=float(range_audit.get("range_gap_coverage", 0.0)),
        range_gap_time_coverage=float(range_audit.get("range_gap_time_coverage", 0.0)),
        range_gap_distance_coverage=float(range_audit.get("range_gap_distance_coverage", 0.0)),
        range_gap_min_coverage=float(range_audit.get("range_gap_min_coverage", 0.0)),
        range_turn_coverage=float(range_audit.get("range_turn_coverage", 0.0)),
        range_shape_score=float(range_audit.get("range_shape_score", 0.0)),
        range_query_local_interpolation_fidelity=float(
            range_audit.get("range_query_local_interpolation_fidelity", 0.0)
        ),
        range_usefulness_score=float(range_audit.get("range_usefulness_score", 0.0)),
        range_usefulness_gap_time_score=float(
            range_audit.get("range_usefulness_gap_time_score", 0.0)
        ),
        range_usefulness_gap_distance_score=float(
            range_audit.get("range_usefulness_gap_distance_score", 0.0)
        ),
        range_usefulness_gap_min_score=float(
            range_audit.get("range_usefulness_gap_min_score", 0.0)
        ),
        range_usefulness_schema_version=int(
            range_audit.get("range_usefulness_schema_version", 0) or 0
        ),
        range_usefulness_gap_ablation_version=int(
            range_audit.get("range_usefulness_gap_ablation_version", 0) or 0
        ),
        query_local_utility_score=query_local_utility_score,
        query_local_utility_schema_version=query_local_utility_schema,
        query_local_utility_components=query_local_utility_components,
        range_audit=range_audit,
        retained_mask=retained_mask if return_mask else None,
    )
