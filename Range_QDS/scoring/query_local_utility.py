"""QueryLocalUtility scoring.

This is the primary metric for query-driven, workload-blind Range-QDS
candidates.  It reuses retained-independent range audit support but changes the
aggregate: query-local point mass and behavior explanation dominate; global
geometry is only a light guardrail.  Schema 5 simplifies point mass and local
behavior into explicit query-local components and stops deriving point mass from
the legacy ``range_point_f1`` aggregate or filling missing behavior fields from
older fallback components.
"""

QUERY_LOCAL_UTILITY_SCHEMA_VERSION = 5

QUERY_LOCAL_UTILITY_WEIGHTS: dict[str, float] = {
    "query_point_mass": 0.50,
    "query_local_behavior": 0.45,
    "global_sanity": 0.05,
}

QUERY_LOCAL_UTILITY_COMPONENT_WEIGHTS: dict[str, float] = {
    "query_point_recall": 0.50,
    "query_local_interpolation_fidelity": 0.20,
    "query_local_turn_change_coverage": 0.15,
    "query_local_continuity": 0.10,
    "endpoint_or_skeleton_sanity": 0.02,
    "global_shape_guardrail_score": 0.02,
    "length_preservation_guardrail": 0.01,
}


def _component_value(components: dict[str, float], key: str, default: float = 0.0) -> float:
    """Return a finite component value clamped to [0, 1]."""
    try:
        value = float(components.get(key, default))
    except (TypeError, ValueError):
        value = float(default)
    if value != value:
        return float(default)
    return float(max(0.0, min(1.0, value)))


def _guardrail_from_sed(avg_sed_km: float | None) -> float:
    """Convert average SED into a soft global-shape guardrail score."""
    if avg_sed_km is None:
        return 1.0
    try:
        sed = max(0.0, float(avg_sed_km))
    except (TypeError, ValueError):
        return 1.0
    return float(1.0 / (1.0 + sed))


def query_local_utility_components_from_range_audit(
    range_audit: dict[str, float],
    *,
    length_preservation: float | None = None,
    avg_sed_km: float | None = None,
    endpoint_sanity: float = 1.0,
) -> dict[str, float]:
    """Return QueryLocalUtility component values from explicit query-local audit fields."""
    query_point = _component_value(range_audit, "query_point_recall")
    turn = _component_value(range_audit, "range_turn_coverage")
    interpolation = _component_value(range_audit, "range_query_local_interpolation_fidelity")
    continuity = _component_value(range_audit, "range_gap_min_coverage")
    length_score = (
        1.0 if length_preservation is None else max(0.0, min(1.0, float(length_preservation)))
    )
    global_shape = _guardrail_from_sed(avg_sed_km)

    return {
        "query_point_recall": query_point,
        "query_local_interpolation_fidelity": interpolation,
        "query_local_turn_change_coverage": turn,
        "query_local_continuity": continuity,
        "endpoint_or_skeleton_sanity": max(0.0, min(1.0, float(endpoint_sanity))),
        "global_shape_guardrail_score": global_shape,
        "length_preservation_guardrail": length_score,
    }


def query_local_utility_score_from_components(components: dict[str, float]) -> float:
    """Return the weighted QueryLocalUtility aggregate."""
    total = 0.0
    for component, weight in QUERY_LOCAL_UTILITY_COMPONENT_WEIGHTS.items():
        total += float(weight) * _component_value(components, component)
    return float(total)


def query_local_utility_from_range_audit(
    range_audit: dict[str, float],
    *,
    length_preservation: float | None = None,
    avg_sed_km: float | None = None,
    endpoint_sanity: float = 1.0,
) -> dict[str, object]:
    """Return QueryLocalUtility score, components, and schema metadata."""
    components = query_local_utility_components_from_range_audit(
        range_audit,
        length_preservation=length_preservation,
        avg_sed_km=avg_sed_km,
        endpoint_sanity=endpoint_sanity,
    )
    return {
        "query_local_utility_schema_version": int(QUERY_LOCAL_UTILITY_SCHEMA_VERSION),
        "query_local_utility_score": query_local_utility_score_from_components(components),
        "query_local_utility_metric_maturity": (
            "query_local_direct_point_mass_behavior_without_legacy_fallbacks"
        ),
        "query_local_utility_query_local_component_source": (
            "query_point_recall_range_query_local_interpolation_fidelity_"
            "range_turn_coverage_range_gap_min_coverage"
        ),
        "query_local_utility_components": components,
        "query_local_utility_component_weights": dict(QUERY_LOCAL_UTILITY_COMPONENT_WEIGHTS),
        "query_local_utility_group_weights": dict(QUERY_LOCAL_UTILITY_WEIGHTS),
    }
