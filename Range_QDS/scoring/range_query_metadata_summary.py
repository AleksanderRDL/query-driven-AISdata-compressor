"""Range-query metadata component summaries."""

from __future__ import annotations

from typing import Any

from scoring.query_local_utility import (
    QUERY_LOCAL_UTILITY_COMPONENT_WEIGHTS,
    query_local_utility_components_from_range_audit,
)
from scoring.range_usefulness import range_usefulness_score_from_components

RANGE_QUERY_METADATA_COMPONENT_SUMMARY_SCHEMA_VERSION = 2

RANGE_QUERY_COMPONENT_KEYS: tuple[str, ...] = (
    "query_point_recall",
    "range_point_f1",
    "range_ship_f1",
    "range_ship_coverage",
    "range_entry_exit_f1",
    "range_crossing_f1",
    "range_temporal_coverage",
    "range_gap_coverage",
    "range_gap_time_coverage",
    "range_gap_distance_coverage",
    "range_gap_min_coverage",
    "range_turn_coverage",
    "range_shape_score",
    "range_query_local_interpolation_fidelity",
)

QUERY_LOCAL_UTILITY_QUERY_LOCAL_EXCLUDED_COMPONENTS: frozenset[str] = frozenset(
    {
        "endpoint_or_skeleton_sanity",
        "global_shape_guardrail_score",
        "length_preservation_guardrail",
    }
)


def _mean(values: list[float], default: float = 0.0) -> float:
    """Return a float mean with an explicit empty-list default."""
    return float(sum(values) / len(values)) if values else float(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if result != result:
        return float(default)
    return float(result)


def _range_query_family_labels(query: dict[str, Any]) -> tuple[str, str]:
    metadata = query.get("_metadata")
    if not isinstance(metadata, dict):
        return "unspecified", "unspecified"

    def label(key: str) -> str:
        raw = metadata.get(key)
        if raw is None:
            return "unspecified"
        value = str(raw).strip()
        return value if value else "unspecified"

    return label("anchor_family"), label("footprint_family")


def _query_local_query_local_utility_summary(range_components: dict[str, float]) -> dict[str, Any]:
    query_components = query_local_utility_components_from_range_audit(
        range_components,
        length_preservation=1.0,
        avg_sed_km=0.0,
        endpoint_sanity=1.0,
    )
    included = {
        key: float(value)
        for key, value in query_components.items()
        if key not in QUERY_LOCAL_UTILITY_QUERY_LOCAL_EXCLUDED_COMPONENTS
    }
    weighted_score = 0.0
    weight_sum = 0.0
    for key, value in included.items():
        weight = float(QUERY_LOCAL_UTILITY_COMPONENT_WEIGHTS.get(key, 0.0))
        weighted_score += weight * float(value)
        weight_sum += weight
    normalized = weighted_score / weight_sum if weight_sum > 0.0 else 0.0
    return {
        "query_local_utility_query_local_components": included,
        "query_local_utility_query_local_weighted_score": float(weighted_score),
        "query_local_utility_query_local_weighted_score_normalized": float(normalized),
        "query_local_utility_query_local_weight_sum": float(weight_sum),
    }


def _range_query_component_summary_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    range_components = {
        key: _mean(
            [
                _safe_float(
                    row.get("range_components", {}).get(key)
                    if isinstance(row.get("range_components"), dict)
                    else None
                )
                for row in rows
            ]
        )
        for key in RANGE_QUERY_COMPONENT_KEYS
    }
    full_point_counts = [_safe_float(row.get("full_point_hit_count")) for row in rows]
    retained_point_counts = [_safe_float(row.get("retained_point_hit_count")) for row in rows]
    full_ship_counts = [_safe_float(row.get("full_trajectory_hit_count")) for row in rows]
    retained_ship_counts = [
        _safe_float(row.get("retained_trajectory_hit_count")) for row in rows
    ]
    ship_evidence_rows: list[dict[str, Any]] = []
    for row in rows:
        ship_evidence = row.get("ship_evidence_counts")
        if isinstance(ship_evidence, dict):
            ship_evidence_rows.append(ship_evidence)
    missed_ship_counts = [
        _safe_float(row.get("missed_trajectory_hit_count")) for row in ship_evidence_rows
    ]
    missed_ship_fractions = [
        _safe_float(row.get("missed_trajectory_hit_fraction")) for row in ship_evidence_rows
    ]
    single_full = [
        _safe_float(row.get("single_point_full_trajectory_hit_count"))
        for row in ship_evidence_rows
    ]
    single_retained = [
        _safe_float(row.get("single_point_retained_trajectory_hit_count"))
        for row in ship_evidence_rows
    ]
    multi_full = [
        _safe_float(row.get("multi_point_full_trajectory_hit_count"))
        for row in ship_evidence_rows
    ]
    multi_retained = [
        _safe_float(row.get("multi_point_retained_trajectory_hit_count"))
        for row in ship_evidence_rows
    ]
    full_ship_total = sum(full_ship_counts)
    retained_ship_total = sum(retained_ship_counts)
    single_full_total = sum(single_full)
    single_retained_total = sum(single_retained)
    multi_full_total = sum(multi_full)
    multi_retained_total = sum(multi_retained)
    return {
        "query_count": len(rows),
        "hit_counts": {
            "full_point_hit_count_total": int(sum(full_point_counts)),
            "retained_point_hit_count_total": int(sum(retained_point_counts)),
            "full_trajectory_hit_count_total": int(full_ship_total),
            "retained_trajectory_hit_count_total": int(retained_ship_total),
            "full_trajectory_hit_count_mean": _mean(full_ship_counts),
            "retained_trajectory_hit_count_mean": _mean(retained_ship_counts),
            "full_point_hit_count_mean": _mean(full_point_counts),
            "retained_point_hit_count_mean": _mean(retained_point_counts),
        },
        "ship_evidence_counts": {
            "full_trajectory_hit_count_total": int(full_ship_total),
            "retained_trajectory_hit_count_total": int(retained_ship_total),
            "missed_trajectory_hit_count_total": int(sum(missed_ship_counts)),
            "missed_trajectory_hit_count_mean": _mean(missed_ship_counts),
            "missed_trajectory_hit_fraction_mean": _mean(missed_ship_fractions),
            "ship_presence_recall": float(
                retained_ship_total / full_ship_total if full_ship_total > 0.0 else 1.0
            ),
            "single_point_full_trajectory_hit_count_total": int(single_full_total),
            "single_point_retained_trajectory_hit_count_total": int(single_retained_total),
            "single_point_ship_presence_recall": float(
                single_retained_total / single_full_total if single_full_total > 0.0 else 1.0
            ),
            "multi_point_full_trajectory_hit_count_total": int(multi_full_total),
            "multi_point_retained_trajectory_hit_count_total": int(multi_retained_total),
            "multi_point_ship_presence_recall": float(
                multi_retained_total / multi_full_total if multi_full_total > 0.0 else 1.0
            ),
            "full_query_hit_points_per_hit_trajectory_mean": _mean(
                [
                    _safe_float(row.get("full_query_hit_points_per_hit_trajectory_mean"))
                    for row in ship_evidence_rows
                ]
            ),
            "retained_query_hit_points_per_hit_trajectory_mean": _mean(
                [
                    _safe_float(row.get("retained_query_hit_points_per_hit_trajectory_mean"))
                    for row in ship_evidence_rows
                ]
            ),
        },
        "range_components": range_components,
        "range_usefulness_score": float(range_usefulness_score_from_components(range_components)),
        **_query_local_query_local_utility_summary(range_components),
    }


def _range_query_metadata_component_summary(
    query_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not query_rows:
        return {
            "available": False,
            "diagnostic_only": True,
            "schema_version": int(RANGE_QUERY_METADATA_COMPONENT_SUMMARY_SCHEMA_VERSION),
            "reason": "no_range_queries",
            "query_count": 0,
        }

    def grouped(fields: tuple[str, ...]) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in query_rows:
            key = "::".join(str(row.get(field, "unspecified")) for field in fields)
            groups.setdefault(key, []).append(row)
        return {
            key: _range_query_component_summary_for_rows(rows)
            for key, rows in sorted(groups.items())
        }

    query_local_components = [
        key
        for key in QUERY_LOCAL_UTILITY_COMPONENT_WEIGHTS
        if key not in QUERY_LOCAL_UTILITY_QUERY_LOCAL_EXCLUDED_COMPONENTS
    ]
    return {
        "available": True,
        "diagnostic_only": True,
        "schema_version": int(RANGE_QUERY_METADATA_COMPONENT_SUMMARY_SCHEMA_VERSION),
        "source": "range_query_metadata_and_range_audit_component_rows",
        "query_count": len(query_rows),
        "component_keys": list(RANGE_QUERY_COMPONENT_KEYS),
        "query_local_utility_query_local_component_keys": query_local_components,
        "excluded_query_local_utility_components": sorted(
            QUERY_LOCAL_UTILITY_QUERY_LOCAL_EXCLUDED_COMPONENTS
        ),
        "query_rows": query_rows,
        "group_by": {
            "anchor_family": grouped(("anchor_family",)),
            "footprint_family": grouped(("footprint_family",)),
            "anchor_footprint_family": grouped(("anchor_family", "footprint_family")),
        },
    }
