"""Derived workload/scoring compatibility diagnostics for query-driven runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from orchestration.diagnostics.artifact_utils import (
    as_dict as _as_dict,
)
from orchestration.diagnostics.artifact_utils import (
    load_json_dict as _load_json,
)

PRIMARY_METHOD = "MLQDS"
BASELINE_METHOD = "DouglasPeucker"
GROUP_KEYS = ("anchor_family", "footprint_family", "anchor_footprint_family")
QUERY_POINT_MASS_COMPONENTS = frozenset(
    {
        "query_point_recall",
    }
)
QUERY_LOCAL_BEHAVIOR_COMPONENTS = frozenset(
    {
        "query_local_interpolation_fidelity",
        "query_local_turn_change_coverage",
        "query_local_continuity",
    }
)
GUARDRAIL_COMPONENTS = frozenset(
    {
        "endpoint_or_skeleton_sanity",
        "global_shape_guardrail_score",
        "length_preservation_guardrail",
    }
)
BLOCKER_FAMILIES = {
    "anchor_family": frozenset({"density"}),
    "footprint_family": frozenset({"medium_operational"}),
}
QUERY_LOCAL_BEHAVIOR_HEAVY_COMPONENT_WEIGHTS_V0 = {
    "query_point_recall": 0.45,
    "query_local_interpolation_fidelity": 0.25,
    "query_local_turn_change_coverage": 0.12,
    "query_local_continuity": 0.13,
    "endpoint_or_skeleton_sanity": 0.02,
    "global_shape_guardrail_score": 0.02,
    "length_preservation_guardrail": 0.01,
}
QUERY_POINT_MASS_HEAVY_COMPONENT_WEIGHTS_V0 = {
    "query_point_recall": 0.50,
    "query_local_interpolation_fidelity": 0.20,
    "query_local_turn_change_coverage": 0.15,
    "query_local_continuity": 0.10,
    "endpoint_or_skeleton_sanity": 0.02,
    "global_shape_guardrail_score": 0.02,
    "length_preservation_guardrail": 0.01,
}
REBALANCED_QUERY_MIX_V0 = {
    "anchor_family": {
        "density": 0.75,
        "sparse_background_control": 0.25,
    },
    "footprint_family": {
        "medium_operational": 0.70,
        "large_context": 0.30,
    },
}
BLOCKER_PRESERVING_QUERY_MIX_V0 = {
    "anchor_family": {
        "density": 0.80,
        "sparse_background_control": 0.20,
    },
    "footprint_family": {
        "medium_operational": 0.6923076923076923,
        "large_context": 0.3076923076923077,
    },
}


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return default


def _sorted_rows(
    rows: list[dict[str, Any]], key: str, *, reverse: bool = False
) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: _as_float(row.get(key)), reverse=reverse)


def _normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, _as_float(value)) for value in weights.values())
    if total <= 0.0:
        return {}
    return {str(key): max(0.0, _as_float(value)) / total for key, value in weights.items()}


def _weight_sum(weights: dict[str, float], components: frozenset[str]) -> float:
    normalized = _normalized_weights(weights)
    return sum(weight for component, weight in normalized.items() if component in components)


def _component_group_weight_summary(weights: dict[str, float]) -> dict[str, float]:
    normalized = _normalized_weights(weights)
    known = QUERY_POINT_MASS_COMPONENTS | QUERY_LOCAL_BEHAVIOR_COMPONENTS | GUARDRAIL_COMPONENTS
    return {
        "query_point_mass_weight": _weight_sum(weights, QUERY_POINT_MASS_COMPONENTS),
        "query_local_behavior_weight": _weight_sum(weights, QUERY_LOCAL_BEHAVIOR_COMPONENTS),
        "guardrail_weight": _weight_sum(weights, GUARDRAIL_COMPONENTS),
        "other_weight": sum(
            weight for component, weight in normalized.items() if component not in known
        ),
    }


def _group_summary(
    artifact: dict[str, Any],
    *,
    method: str,
    group_key: str,
    family: str,
) -> dict[str, Any]:
    return _as_dict(
        _as_dict(
            _as_dict(
                _as_dict(_as_dict(artifact.get("matched")).get(method)).get("range_audit")
            ).get("range_query_metadata_component_summary")
        )
        .get("group_by", {})
        .get(group_key, {})
        .get(family)
    )


def _query_local_component_deltas(
    artifact: dict[str, Any],
    *,
    group_key: str,
    family: str,
    primary_method: str,
    baseline_method: str,
) -> dict[str, float]:
    primary_components = _as_dict(
        _group_summary(
            artifact,
            method=primary_method,
            group_key=group_key,
            family=family,
        ).get("query_local_utility_query_local_components")
    )
    baseline_components = _as_dict(
        _group_summary(
            artifact,
            method=baseline_method,
            group_key=group_key,
            family=family,
        ).get("query_local_utility_query_local_components")
    )
    out: dict[str, float] = {}
    for component in sorted(set(primary_components) | set(baseline_components)):
        out[component] = _as_float(primary_components.get(component)) - _as_float(
            baseline_components.get(component)
        )
    return out


def _component_weights(artifact: dict[str, Any], *, method: str) -> dict[str, float]:
    weights = _as_dict(
        _as_dict(_as_dict(artifact.get("matched")).get(method))
        .get("range_audit", {})
        .get("query_local_utility_component_weights")
    )
    return {str(key): _as_float(value) for key, value in weights.items()}


def _method_components(artifact: dict[str, Any], *, method: str) -> dict[str, float]:
    components = _as_dict(
        _as_dict(_as_dict(artifact.get("matched")).get(method))
        .get("range_audit", {})
        .get("query_local_utility_components")
    )
    return {str(key): _as_float(value) for key, value in components.items()}


def _score_from_component_weights(
    components: dict[str, float],
    weights: dict[str, float],
) -> float:
    normalized = _normalized_weights(weights)
    return sum(
        _as_float(components.get(component)) * weight for component, weight in normalized.items()
    )


def _profile_by_family_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    total_queries = sum(_as_float(row.get("query_count")) for row in rows)
    if total_queries <= 0.0:
        return {}
    return {
        str(row.get("family")): _as_float(row.get("query_count")) / total_queries for row in rows
    }


def _critical_family_pressure_summary(
    rows: list[dict[str, Any]],
    *,
    profile_weights: dict[str, float] | None,
    critical_families: frozenset[str],
) -> dict[str, Any]:
    active_profile = _profile_by_family_from_rows(rows)
    candidate_profile = (
        active_profile if profile_weights is None else _normalized_weights(profile_weights)
    )
    family_rows = []
    ratios = []
    for family in sorted(critical_families):
        if family not in active_profile:
            continue
        active_weight = active_profile.get(family, 0.0)
        candidate_weight = candidate_profile.get(family, 0.0)
        ratio = candidate_weight / active_weight if active_weight > 0.0 else 0.0
        ratios.append(ratio)
        family_rows.append(
            {
                "family": family,
                "active_profile_weight": active_weight,
                "candidate_profile_weight": candidate_weight,
                "candidate_to_active_weight_ratio": ratio,
                "pressure_preserved": candidate_weight >= active_weight * 0.95,
            }
        )
    return {
        "available": bool(family_rows),
        "critical_families": family_rows,
        "min_candidate_to_active_weight_ratio": min(ratios) if ratios else None,
        "pressure_preserved": bool(family_rows)
        and all(bool(row["pressure_preserved"]) for row in family_rows),
    }


def _family_rows(
    artifact: dict[str, Any],
    *,
    group_key: str,
    primary_method: str,
    baseline_method: str,
) -> list[dict[str, Any]]:
    comparison_rows = _as_dict(
        _as_dict(
            _as_dict(artifact.get("workload_scoring_compatibility_diagnostics")).get(
                "comparisons_vs_baseline"
            )
        )
        .get(baseline_method, {})
        .get(group_key)
    )
    weights = _component_weights(artifact, method=primary_method)
    rows: list[dict[str, Any]] = []
    for family, raw_row in sorted(comparison_rows.items()):
        row = _as_dict(raw_row)
        query_count = _as_float(row.get("query_count"))
        query_local_deltas = _query_local_component_deltas(
            artifact,
            group_key=group_key,
            family=str(family),
            primary_method=primary_method,
            baseline_method=baseline_method,
        )
        weighted_query_local_deltas = {
            component: delta * weights.get(component, 0.0)
            for component, delta in query_local_deltas.items()
        }
        range_component_deltas = {
            str(component): _as_float(delta)
            for component, delta in _as_dict(row.get("range_component_deltas")).items()
        }
        ship_deltas = {
            str(component): _as_float(delta)
            for component, delta in _as_dict(row.get("ship_evidence_count_deltas")).items()
        }
        top_weighted_losses = _sorted_rows(
            [
                {
                    "component": component,
                    "delta": delta,
                    "weight": weights.get(component, 0.0),
                    "weighted_delta": weighted_query_local_deltas[component],
                }
                for component, delta in query_local_deltas.items()
                if weighted_query_local_deltas[component] < 0.0
            ],
            "weighted_delta",
        )[:5]
        rows.append(
            {
                "group_key": group_key,
                "family": str(family),
                "query_count": query_count,
                "query_local_score_delta": _as_float(row.get("query_local_score_delta")),
                "range_usefulness_delta": _as_float(row.get("range_usefulness_delta")),
                "range_component_deltas": range_component_deltas,
                "ship_evidence_count_deltas": ship_deltas,
                "query_local_component_deltas": query_local_deltas,
                "weighted_query_local_component_deltas": weighted_query_local_deltas,
                "top_weighted_query_local_losses": top_weighted_losses,
            }
        )
    return rows


def _weighted_component_rollup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, float] = {}
    weighted_totals: dict[str, float] = {}
    negative_family_counts: dict[str, int] = {}
    group_key = str(rows[0].get("group_key")) if rows else "unknown"
    query_count_total = sum(_as_float(row.get("query_count")) for row in rows)
    for row in rows:
        query_count = _as_float(row.get("query_count"))
        for component, delta in _as_dict(row.get("query_local_component_deltas")).items():
            delta_f = _as_float(delta)
            totals[component] = totals.get(component, 0.0) + delta_f * query_count
            if delta_f < 0.0:
                negative_family_counts[component] = negative_family_counts.get(component, 0) + 1
        for component, delta in _as_dict(row.get("weighted_query_local_component_deltas")).items():
            weighted_totals[component] = (
                weighted_totals.get(component, 0.0) + _as_float(delta) * query_count
            )
    if query_count_total <= 0.0:
        query_count_total = 1.0
    rollup = [
        {
            "group_key": group_key,
            "component": component,
            "mean_delta_weighted_by_query_count": totals.get(component, 0.0) / query_count_total,
            "mean_weighted_score_delta_by_query_count": weighted_totals.get(component, 0.0)
            / query_count_total,
            "negative_family_count": negative_family_counts.get(component, 0),
        }
        for component in sorted(totals)
    ]
    return _sorted_rows(rollup, "mean_weighted_score_delta_by_query_count")


def _group_delta_with_weights(
    rows: list[dict[str, Any]],
    *,
    component_weights: dict[str, float],
    profile_weights: dict[str, float] | None,
    critical_families: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    normalized_component_weights = _normalized_weights(component_weights)
    if profile_weights is None:
        total_queries = sum(_as_float(row.get("query_count")) for row in rows)
        profile_by_family = {
            str(row.get("family")): _as_float(row.get("query_count")) / total_queries
            for row in rows
            if total_queries > 0.0
        }
        profile_source = "artifact_query_count"
    else:
        profile_by_family = _normalized_weights(profile_weights)
        profile_source = "candidate_profile_weights"
    family_rows = []
    weighted_delta = 0.0
    for row in rows:
        family = str(row.get("family"))
        family_weight = profile_by_family.get(family, 0.0)
        component_delta = sum(
            _as_float(_as_dict(row.get("query_local_component_deltas")).get(component)) * weight
            for component, weight in normalized_component_weights.items()
        )
        contribution = family_weight * component_delta
        weighted_delta += contribution
        ship_deltas = _as_dict(row.get("ship_evidence_count_deltas"))
        family_rows.append(
            {
                "family": family,
                "family_weight": family_weight,
                "component_weighted_delta": component_delta,
                "profile_weighted_contribution": contribution,
                "range_usefulness_delta": row.get("range_usefulness_delta"),
                "ship_presence_recall_delta": ship_deltas.get("ship_presence_recall"),
                "missed_trajectory_hit_count_delta": ship_deltas.get(
                    "missed_trajectory_hit_count_total"
                ),
            }
        )
    return {
        "profile_source": profile_source,
        "weighted_query_local_delta": weighted_delta,
        "component_group_weights": _component_group_weight_summary(component_weights),
        "critical_family_pressure": _critical_family_pressure_summary(
            rows,
            profile_weights=profile_weights,
            critical_families=critical_families,
        ),
        "family_rows": _sorted_rows(family_rows, "profile_weighted_contribution"),
    }


def _blocker_preserving_outcome(group_rows: dict[str, Any]) -> dict[str, Any]:
    unresolved_rows = []
    weighted_deltas: dict[str, float] = {}
    pressure_preserved = True
    for group_key, group in group_rows.items():
        candidate = _as_dict(group).get("point_mass_heavy_component_blocker_preserving_profile")
        candidate = _as_dict(candidate)
        weighted_deltas[group_key] = _as_float(candidate.get("weighted_query_local_delta"))
        pressure = _as_dict(candidate.get("critical_family_pressure"))
        pressure_preserved = pressure_preserved and bool(pressure.get("pressure_preserved"))
        critical_families = BLOCKER_FAMILIES.get(group_key, frozenset())
        for row in candidate.get("family_rows", []):
            if not isinstance(row, dict) or row.get("family") not in critical_families:
                continue
            if _as_float(row.get("component_weighted_delta")) < 0.0:
                unresolved_rows.append(
                    {
                        "group_key": group_key,
                        "family": row.get("family"),
                        "component_weighted_delta": row.get("component_weighted_delta"),
                        "profile_weighted_contribution": row.get("profile_weighted_contribution"),
                        "ship_presence_recall_delta": row.get("ship_presence_recall_delta"),
                        "missed_trajectory_hit_count_delta": row.get(
                            "missed_trajectory_hit_count_delta"
                        ),
                    }
                )
    return {
        "candidate": "point_mass_heavy_component_blocker_preserving_profile",
        "critical_family_pressure_preserved": pressure_preserved,
        "weighted_query_local_deltas": weighted_deltas,
        "unresolved_blocker_family_count": len(unresolved_rows),
        "unresolved_blocker_families": _sorted_rows(
            unresolved_rows,
            "component_weighted_delta",
        ),
        "status": "still_blocked" if unresolved_rows else "candidate_needs_strict_probe",
        "interpretation": (
            "A blocker-preserving candidate still has negative critical-family "
            "query-local weighted components, so profile/scoring changes alone should not be "
            "accepted until the target/head signal for those families is fixed."
            if unresolved_rows
            else "The blocker-preserving candidate removes grouped blocker signs; "
            "strict retraining evidence would still be required."
        ),
    }


def _recalibration_candidates(
    artifact: dict[str, Any],
    *,
    groups: dict[str, Any],
    primary_method: str,
    baseline_method: str,
) -> dict[str, Any]:
    active_weights = _component_weights(artifact, method=primary_method)
    candidates = {
        "active_component_weights": active_weights,
        "query_local_behavior_heavy_component_weights_v0": (
            QUERY_LOCAL_BEHAVIOR_HEAVY_COMPONENT_WEIGHTS_V0
        ),
        "query_point_mass_heavy_component_weights_v0": QUERY_POINT_MASS_HEAVY_COMPONENT_WEIGHTS_V0,
    }
    primary_components = _method_components(artifact, method=primary_method)
    baseline_components = _method_components(artifact, method=baseline_method)
    scoring_rows = []
    for name, weights in candidates.items():
        primary_score = _score_from_component_weights(primary_components, weights)
        baseline_score = _score_from_component_weights(baseline_components, weights)
        scoring_rows.append(
            {
                "candidate": name,
                "diagnostic_only": True,
                "primary_score": primary_score,
                "baseline_score": baseline_score,
                "primary_minus_baseline": primary_score - baseline_score,
                "component_weights": _normalized_weights(weights),
                "component_group_weights": _component_group_weight_summary(weights),
            }
        )
    group_rows: dict[str, Any] = {}
    for group_key in ("anchor_family", "footprint_family"):
        rows = _as_dict(groups.get(group_key)).get("rows", [])
        if not isinstance(rows, list):
            rows = []
        group_rows[group_key] = {
            "active_component_active_profile": _group_delta_with_weights(
                rows,
                component_weights=active_weights,
                profile_weights=None,
                critical_families=BLOCKER_FAMILIES.get(group_key, frozenset()),
            ),
            "behavior_heavy_component_active_profile": _group_delta_with_weights(
                rows,
                component_weights=QUERY_LOCAL_BEHAVIOR_HEAVY_COMPONENT_WEIGHTS_V0,
                profile_weights=None,
                critical_families=BLOCKER_FAMILIES.get(group_key, frozenset()),
            ),
            "behavior_heavy_component_rebalanced_profile": _group_delta_with_weights(
                rows,
                component_weights=QUERY_LOCAL_BEHAVIOR_HEAVY_COMPONENT_WEIGHTS_V0,
                profile_weights=REBALANCED_QUERY_MIX_V0.get(group_key, {}),
                critical_families=BLOCKER_FAMILIES.get(group_key, frozenset()),
            ),
            "point_mass_heavy_component_active_profile": _group_delta_with_weights(
                rows,
                component_weights=QUERY_POINT_MASS_HEAVY_COMPONENT_WEIGHTS_V0,
                profile_weights=None,
                critical_families=BLOCKER_FAMILIES.get(group_key, frozenset()),
            ),
            "point_mass_heavy_component_blocker_preserving_profile": (
                _group_delta_with_weights(
                    rows,
                    component_weights=QUERY_POINT_MASS_HEAVY_COMPONENT_WEIGHTS_V0,
                    profile_weights=BLOCKER_PRESERVING_QUERY_MIX_V0.get(group_key, {}),
                    critical_families=BLOCKER_FAMILIES.get(group_key, frozenset()),
                )
            ),
        }
    active_delta = next(
        row["primary_minus_baseline"]
        for row in scoring_rows
        if row["candidate"] == "active_component_weights"
    )
    candidate_delta = next(
        row["primary_minus_baseline"]
        for row in scoring_rows
        if row["candidate"] == "query_local_behavior_heavy_component_weights_v0"
    )
    return {
        "diagnostic_only": True,
        "candidate_policy": (
            "Post-hoc scoring/profile recalibration probe only. It cannot prove "
            "learning causality or final success."
        ),
        "scoring_candidates": scoring_rows,
        "group_profile_candidates": group_rows,
        "blocker_preserving_outcome": _blocker_preserving_outcome(group_rows),
        "candidate_improves_score_delta": candidate_delta > active_delta,
        "masking_risk": (
            "high" if candidate_delta > active_delta else "not_improved_by_candidate_weights"
        ),
        "masking_risk_reason": (
            "Candidate improves post-hoc score delta by downweighting currently "
            "blocking query-local components; strict retraining evidence would "
            "still be required before any scoring/profile change."
        ),
    }


def _blocking_family_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocking = []
    for row in rows:
        ship_deltas = _as_dict(row.get("ship_evidence_count_deltas"))
        if _as_float(row.get("range_usefulness_delta")) < 0.0 and (
            _as_float(ship_deltas.get("missed_trajectory_hit_count_total")) > 0.0
            or _as_float(ship_deltas.get("ship_presence_recall")) < 0.0
        ):
            blocking.append(
                {
                    "group_key": row.get("group_key"),
                    "family": row.get("family"),
                    "query_count": row.get("query_count"),
                    "range_usefulness_delta": row.get("range_usefulness_delta"),
                    "query_local_score_delta": row.get("query_local_score_delta"),
                    "ship_presence_recall_delta": ship_deltas.get("ship_presence_recall"),
                    "missed_trajectory_hit_count_delta": ship_deltas.get(
                        "missed_trajectory_hit_count_total"
                    ),
                    "top_weighted_query_local_losses": row.get(
                        "top_weighted_query_local_losses", []
                    ),
                }
            )
    return _sorted_rows(blocking, "range_usefulness_delta")[:12]


def _artifact_summary(
    artifact: dict[str, Any],
    *,
    label: str,
    primary_method: str,
    baseline_method: str,
) -> dict[str, Any]:
    matched = _as_dict(artifact.get("matched"))
    primary = _as_dict(matched.get(primary_method))
    baseline = _as_dict(matched.get(baseline_method))
    scores = {
        "primary_query_local_utility": _as_float(primary.get("query_local_utility_score")),
        "baseline_query_local_utility": _as_float(baseline.get("query_local_utility_score")),
        "primary_minus_baseline_query_local_utility": _as_float(
            primary.get("query_local_utility_score")
        )
        - _as_float(baseline.get("query_local_utility_score")),
    }
    target_mode = _as_dict(
        _as_dict(artifact.get("training_target_diagnostics")).get("query_local_utility_factorized")
    ).get("target_mode")
    if target_mode is None:
        target_mode = _as_dict(_as_dict(artifact.get("config")).get("model")).get(
            "range_training_target_mode"
        )
    groups: dict[str, Any] = {}
    for group_key in GROUP_KEYS:
        rows = _family_rows(
            artifact,
            group_key=group_key,
            primary_method=primary_method,
            baseline_method=baseline_method,
        )
        groups[group_key] = {
            "rows": rows,
            "blocking_families": _blocking_family_rows(rows),
            "query_local_component_rollup": _weighted_component_rollup(rows),
        }
    return {
        "label": label,
        "scores": scores,
        "target_mode": target_mode,
        "groups": groups,
        "recalibration_diagnostics": _recalibration_candidates(
            artifact,
            groups=groups,
            primary_method=primary_method,
            baseline_method=baseline_method,
        ),
    }


def build_workload_component_compatibility_diagnostic(
    artifacts: list[tuple[str, dict[str, Any]]],
    *,
    primary_method: str = PRIMARY_METHOD,
    baseline_method: str = BASELINE_METHOD,
) -> dict[str, Any]:
    """Return a compact derived diagnosis from strict grouped artifact payloads."""
    summaries = [
        _artifact_summary(
            artifact,
            label=label,
            primary_method=primary_method,
            baseline_method=baseline_method,
        )
        for label, artifact in artifacts
    ]
    current = summaries[0] if summaries else {}
    anchor_rollup = _as_dict(_as_dict(current.get("groups")).get("anchor_family")).get(
        "query_local_component_rollup", []
    )
    footprint_rollup = _as_dict(_as_dict(current.get("groups")).get("footprint_family")).get(
        "query_local_component_rollup", []
    )
    negative_components = [
        row
        for row in list(anchor_rollup) + list(footprint_rollup)
        if _as_float(row.get("mean_weighted_score_delta_by_query_count")) < 0.0
        and int(_as_float(row.get("negative_family_count"))) >= 2
    ]
    negative_components = _sorted_rows(
        negative_components,
        "mean_weighted_score_delta_by_query_count",
    )[:12]
    blocking_families = []
    for group_key, group in _as_dict(current.get("groups")).items():
        if group_key == "anchor_footprint_family":
            continue
        group_blocking_families = _as_dict(group).get("blocking_families", [])
        if isinstance(group_blocking_families, list):
            blocking_families.extend(group_blocking_families)
    blocking_families = _sorted_rows(blocking_families, "range_usefulness_delta")[:12]
    recalibration = _as_dict(current.get("recalibration_diagnostics"))
    candidate_rows = recalibration.get("scoring_candidates", [])
    if not isinstance(candidate_rows, list):
        candidate_rows = []
    candidate_score_deltas = {
        str(row.get("candidate")): row.get("primary_minus_baseline")
        for row in candidate_rows
        if isinstance(row, dict)
    }
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "primary_method": primary_method,
        "baseline_method": baseline_method,
        "artifact_count": len(summaries),
        "artifacts": summaries,
        "summary": {
            "primary_artifact_label": current.get("label"),
            "primary_minus_baseline_query_local_utility": _as_dict(current.get("scores")).get(
                "primary_minus_baseline_query_local_utility"
            ),
            "blocking_families": blocking_families,
            "persistent_negative_query_local_components": negative_components,
            "recalibration_candidate_score_deltas": candidate_score_deltas,
            "recalibration_masking_risk": recalibration.get("masking_risk"),
            "blocker_preserving_candidate_status": _as_dict(
                recalibration.get("blocker_preserving_outcome")
            ).get("status"),
            "blocker_preserving_candidate_unresolved_families": _as_dict(
                recalibration.get("blocker_preserving_outcome")
            ).get("unresolved_blocker_families"),
            "interpretation": (
                "Grouped strict evidence points to workload/scoring compatibility, not "
                "another segment-budget proxy. Families with negative usefulness and "
                "missed ship evidence should drive the next workload/profile and "
                "QueryLocalUtility component recalibration checkpoint."
            ),
        },
    }


def _parse_labeled_artifact(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path)
    path = Path(value)
    return path.parent.name, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a derived workload/component compatibility diagnostic."
    )
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="Artifact path, optionally label=path. First artifact is the primary summary.",
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args(argv)

    artifacts = [
        (label, _load_json(path))
        for label, path in (_parse_labeled_artifact(value) for value in args.artifact)
    ]
    diagnostic = build_workload_component_compatibility_diagnostic(artifacts)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(diagnostic, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
