"""Metric and reporting-gate helpers for benchmark report rows."""

from __future__ import annotations

from typing import Any

from benchmarking.common import as_float
from training.model_features import is_workload_blind_model_type

MIN_MATCHED_LEARNED_SLOT_FRACTION_FOR_BLIND_CLAIM = 0.25
RANGE_COMPONENT_KEYS = (
    "range_point_f1",
    "range_ship_f1",
    "range_ship_coverage",
    "range_entry_exit_f1",
    "range_crossing_f1",
    "range_temporal_coverage",
    "range_gap_coverage",
    "range_turn_coverage",
    "range_shape_score",
    "range_query_local_interpolation_fidelity",
)
RANGE_USEFULNESS_GAP_VARIANT_KEYS = (
    ("gap_time", "range_usefulness_gap_time_score"),
    ("gap_distance", "range_usefulness_gap_distance_score"),
    ("gap_min", "range_usefulness_gap_min_score"),
)


def _metric_delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    """Return left - right for one numeric metric."""
    left_value = as_float(left.get(key))
    right_value = as_float(right.get(key))
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _metric_beats(left: dict[str, Any], right: dict[str, Any], key: str) -> bool | None:
    """Return whether left strictly beats right for a higher-is-better metric."""
    delta = _metric_delta(left, right, key)
    return None if delta is None else bool(delta > 0.0)


def _geometry_fields(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Flatten geometric-distortion metrics for one method."""
    geometry = metrics.get("geometric_distortion") or {}
    return {
        f"{prefix}_avg_sed_km": geometry.get("avg_sed_km"),
        f"{prefix}_max_sed_km": geometry.get("max_sed_km"),
        f"{prefix}_avg_ped_km": geometry.get("avg_ped_km"),
        f"{prefix}_max_ped_km": geometry.get("max_ped_km"),
        f"{prefix}_removed_points": geometry.get("removed_points"),
        f"{prefix}_avg_length_preserved": metrics.get("avg_length_preserved"),
        f"{prefix}_latency_ms": metrics.get("latency_ms"),
    }


def _worst_uniform_component_delta(component_deltas: dict[str, float | None]) -> dict[str, Any]:
    """Return the most negative MLQDS-vs-uniform range component delta."""
    numeric = [(key, value) for key, value in component_deltas.items() if value is not None]
    if not numeric:
        return {"worst_uniform_component_delta_metric": None, "worst_uniform_component_delta": None}
    key, value = min(numeric, key=lambda item: float(item[1]))
    if float(value) >= 0.0:
        return {
            "worst_uniform_component_delta_metric": "none",
            "worst_uniform_component_delta": 0.0,
        }
    return {
        "worst_uniform_component_delta_metric": key,
        "worst_uniform_component_delta": float(value),
    }


def _single_cell_range_status(
    *,
    returncode: int,
    model_type: Any,
    protocol_enabled: Any,
    primary_frozen: Any,
    audit_frozen: Any,
    audit_ratio_count: int,
    beats_uniform: bool | None,
    beats_dp: bool | None,
    selector_claim_status: str,
) -> str:
    """Classify one benchmark row against the single-cell blind RangeUseful gate."""
    if int(returncode) != 0:
        return "child_failed"
    if beats_uniform is None or beats_dp is None:
        return "missing_range_usefulness"
    if model_type == "range_aware":
        return "diagnostic_upper_bound"
    workload_blind = is_workload_blind_model_type(model_type)
    if not workload_blind:
        return "non_blind_model"
    protocol_ok = bool(protocol_enabled) and bool(primary_frozen)
    if audit_ratio_count > 0:
        protocol_ok = protocol_ok and bool(audit_frozen)
    if not protocol_ok:
        return "protocol_fail"
    if beats_uniform and beats_dp:
        if selector_claim_status in {"missing_selector_evidence", "selector_scaffold_dominated"}:
            return selector_claim_status
        return "beats_uniform_and_douglas_peucker"
    if beats_dp:
        return "fails_uniform"
    if beats_uniform:
        return "fails_douglas_peucker"
    return "fails_uniform_and_douglas_peucker"


def _selector_claim_evidence(
    selector_budget_row: dict[str, Any], model_type: Any
) -> dict[str, Any]:
    """Classify whether the matched selector budget leaves room for learned behavior.

    This is a reporting guard, not a model constraint. A workload-blind run that
    beats baselines with a tiny learned slot fraction is still useful as a
    diagnostic, but it is not evidence that the learned model caused the win.
    """
    if not is_workload_blind_model_type(model_type):
        return {
            "selector_claim_status": "not_workload_blind",
            "selector_claim_has_material_learned_budget": None,
            "selector_claim_min_learned_slot_fraction": None,
        }
    learned_fraction = as_float(selector_budget_row.get("learned_slot_fraction_of_budget"))
    if learned_fraction is None:
        return {
            "selector_claim_status": "missing_selector_evidence",
            "selector_claim_has_material_learned_budget": None,
            "selector_claim_min_learned_slot_fraction": MIN_MATCHED_LEARNED_SLOT_FRACTION_FOR_BLIND_CLAIM,
        }
    has_material_budget = learned_fraction >= MIN_MATCHED_LEARNED_SLOT_FRACTION_FOR_BLIND_CLAIM
    return {
        "selector_claim_status": (
            "model_has_material_budget" if has_material_budget else "selector_scaffold_dominated"
        ),
        "selector_claim_has_material_learned_budget": bool(has_material_budget),
        "selector_claim_min_learned_slot_fraction": MIN_MATCHED_LEARNED_SLOT_FRACTION_FOR_BLIND_CLAIM,
    }


def _effective_diversity_bonus(model_config: dict[str, Any]) -> float | None:
    """Return the diversity bonus actually consumed by the configured selector."""
    configured = model_config.get("mlqds_diversity_bonus")
    if configured is None:
        return None
    if str(model_config.get("mlqds_hybrid_mode", "fill")).lower() in {
        "stratified",
        "global_budget",
    }:
        return 0.0
    return float(configured)
