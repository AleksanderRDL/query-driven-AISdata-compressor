"""Final-candidate gate helpers for query-driven experiment artifacts."""

from __future__ import annotations

from typing import Any

import torch

from config.experiment_config import ExperimentConfig
from evaluation.metrics import MethodEvaluation
from training.query_prior_fields import QUERY_PRIOR_FIELD_NAMES, sample_query_prior_fields


def _points_outside_prior_extent_fraction(
    points: torch.Tensor, extent: dict[str, Any] | None
) -> float | None:
    """Return the fraction of points outside a train-prior spatial extent."""
    if not isinstance(extent, dict) or int(points.shape[0]) <= 0:
        return None
    lat = points[:, 1].detach().cpu().float()
    lon = points[:, 2].detach().cpu().float()
    outside = (
        (lat < float(extent.get("lat_min", -float("inf"))))
        | (lat > float(extent.get("lat_max", float("inf"))))
        | (lon < float(extent.get("lon_min", -float("inf"))))
        | (lon > float(extent.get("lon_max", float("inf"))))
    )
    return float(outside.float().mean().item())


def _spatial_extent_intersection_fraction(
    train_points: torch.Tensor, eval_points: torch.Tensor
) -> float | None:
    """Return train/eval lat-lon extent intersection as a fraction of eval extent area."""
    if int(train_points.shape[0]) <= 0 or int(eval_points.shape[0]) <= 0:
        return None
    train_lat_min = float(train_points[:, 1].min().item())
    train_lat_max = float(train_points[:, 1].max().item())
    train_lon_min = float(train_points[:, 2].min().item())
    train_lon_max = float(train_points[:, 2].max().item())
    eval_lat_min = float(eval_points[:, 1].min().item())
    eval_lat_max = float(eval_points[:, 1].max().item())
    eval_lon_min = float(eval_points[:, 2].min().item())
    eval_lon_max = float(eval_points[:, 2].max().item())
    eval_lat_span = eval_lat_max - eval_lat_min
    eval_lon_span = eval_lon_max - eval_lon_min
    if eval_lat_span <= 1e-12 or eval_lon_span <= 1e-12:
        inside = (
            eval_lat_min >= train_lat_min - 1e-12
            and eval_lat_max <= train_lat_max + 1e-12
            and eval_lon_min >= train_lon_min - 1e-12
            and eval_lon_max <= train_lon_max + 1e-12
        )
        return 1.0 if inside else 0.0
    lat_overlap = max(0.0, min(train_lat_max, eval_lat_max) - max(train_lat_min, eval_lat_min))
    lon_overlap = max(0.0, min(train_lon_max, eval_lon_max) - max(train_lon_min, eval_lon_min))
    eval_area = max(1e-12, eval_lat_span * eval_lon_span)
    return float(max(0.0, min(1.0, (lat_overlap * lon_overlap) / eval_area)))


def _support_overlap_gate(
    *,
    train_points: torch.Tensor,
    eval_points: torch.Tensor,
    query_prior_field: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return train/eval support-overlap evidence for final query-driven claims."""
    thresholds = {
        "eval_points_outside_train_prior_extent_fraction_max": 0.10,
        "sampled_prior_nonzero_fraction_min": 0.50,
        "primary_sampled_prior_nonzero_fraction_min": 0.30,
        "route_density_overlap_min": 0.25,
        "query_prior_support_overlap_min": 0.25,
    }
    if query_prior_field is None:
        return {
            "schema_version": 1,
            "gate_pass": False,
            "failed_checks": ["query_prior_field_missing"],
            **thresholds,
            "eval_points_outside_train_prior_extent_fraction": None,
            "sampled_prior_nonzero_fraction": 0.0,
            "primary_sampled_prior_nonzero_fraction": 0.0,
            "route_density_overlap": 0.0,
            "query_prior_support_overlap": 0.0,
            "train_eval_spatial_extent_intersection_fraction": _spatial_extent_intersection_fraction(
                train_points,
                eval_points,
            ),
        }
    sampled = sample_query_prior_fields(eval_points, query_prior_field).detach().cpu().float()
    if int(sampled.numel()) == 0:
        sampled_any = torch.zeros((int(eval_points.shape[0]),), dtype=torch.bool)
        primary = sampled_any
        route = sampled_any
        query_support = sampled_any
    else:
        sampled_any = (sampled.abs() > 1e-12).any(dim=1)
        feature_names: tuple[str, ...] = tuple(QUERY_PRIOR_FIELD_NAMES)

        def col(name: str) -> torch.Tensor:
            try:
                idx = feature_names.index(name)
            except ValueError:
                return torch.zeros((int(sampled.shape[0]),), dtype=torch.bool)
            if idx >= int(sampled.shape[1]):
                return torch.zeros((int(sampled.shape[0]),), dtype=torch.bool)
            return sampled[:, idx].abs() > 1e-12

        primary = col("spatial_query_hit_probability")
        spatiotemporal = col("spatiotemporal_query_hit_probability")
        route = col("route_density_prior")
        query_support = primary | spatiotemporal
    point_count = max(1, int(eval_points.shape[0]))
    outside = _points_outside_prior_extent_fraction(eval_points, query_prior_field.get("extent"))
    sampled_fraction = float(sampled_any.float().sum().item() / point_count)
    primary_fraction = float(primary.float().sum().item() / point_count)
    route_fraction = float(route.float().sum().item() / point_count)
    query_support_fraction = float(query_support.float().sum().item() / point_count)
    failed_checks: list[str] = []
    if outside is None:
        failed_checks.append("train_prior_extent_missing")
    elif outside > thresholds["eval_points_outside_train_prior_extent_fraction_max"] + 1e-12:
        failed_checks.append("eval_points_outside_train_prior_extent_too_high")
    if sampled_fraction + 1e-12 < thresholds["sampled_prior_nonzero_fraction_min"]:
        failed_checks.append("sampled_prior_nonzero_fraction_too_low")
    if primary_fraction + 1e-12 < thresholds["primary_sampled_prior_nonzero_fraction_min"]:
        failed_checks.append("primary_sampled_prior_nonzero_fraction_too_low")
    if route_fraction + 1e-12 < thresholds["route_density_overlap_min"]:
        failed_checks.append("route_density_overlap_too_low")
    if query_support_fraction + 1e-12 < thresholds["query_prior_support_overlap_min"]:
        failed_checks.append("query_prior_support_overlap_too_low")
    return {
        "schema_version": 1,
        "gate_pass": not failed_checks,
        "failed_checks": failed_checks,
        **thresholds,
        "eval_points_outside_train_prior_extent_fraction": outside,
        "sampled_prior_nonzero_fraction": sampled_fraction,
        "primary_sampled_prior_nonzero_fraction": primary_fraction,
        "route_density_overlap": route_fraction,
        "query_prior_support_overlap": query_support_fraction,
        "train_eval_spatial_extent_intersection_fraction": _spatial_extent_intersection_fraction(
            train_points,
            eval_points,
        ),
    }


def _normalize_fraction_for_gate(value: Any) -> float | None:
    """Normalize optional fraction/percent values for gate checks."""
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    if out > 1.0 and out <= 100.0:
        out /= 100.0
    return out


def _optional_float_for_gate(value: Any) -> float | None:
    """Coerce optional gate evidence to float without percent normalization."""
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _workload_stability_gate(
    *,
    config: ExperimentConfig,
    train_label_workloads: list[Any],
    eval_workload: Any,
    selection_workload: Any | None,
) -> dict[str, Any]:
    """Return final-candidate gate evidence for statistically stable workloads."""
    allowed_coverage_targets = (0.05, 0.10, 0.15, 0.30)
    min_train_replicates = 4
    min_queries_per_workload = 8
    gate_mode = str(getattr(config.query, "workload_stability_gate_mode", "final")).lower()
    required_profile_id = "range_workload_v1"
    coverage_tolerance = 1e-6
    failed_checks: list[str] = []

    configured_target = _normalize_fraction_for_gate(getattr(config.query, "target_coverage", None))
    configured_target_in_grid = configured_target is not None and any(
        abs(configured_target - target) <= 1e-9 for target in allowed_coverage_targets
    )
    if not configured_target_in_grid:
        failed_checks.append("coverage_target_not_in_final_grid")
    if len(train_label_workloads) < min_train_replicates:
        failed_checks.append("too_few_train_workload_replicates")

    overshoot = _normalize_fraction_for_gate(
        getattr(config.query, "range_max_coverage_overshoot", None)
    )
    max_allowed_overshoot = _coverage_overshoot_tolerance_for_target(configured_target)
    if (
        max_allowed_overshoot is None
        or overshoot is None
        or overshoot > max_allowed_overshoot + 1e-12
    ):
        failed_checks.append("coverage_overshoot_tolerance_too_loose")
    workload_rows: list[dict[str, Any]] = []
    workloads: list[tuple[str, Any]] = [
        *[(f"train_r{idx}", workload) for idx, workload in enumerate(train_label_workloads)],
        ("eval", eval_workload),
    ]
    if selection_workload is not None:
        workloads.append(("selection", selection_workload))

    for label, workload in workloads:
        generation = (getattr(workload, "generation_diagnostics", None) or {}).get(
            "query_generation", {}
        )
        if not isinstance(generation, dict):
            generation = {}
        profile_id = str(generation.get("workload_profile_id", ""))
        mode = str(generation.get("mode", ""))
        query_count = len(getattr(workload, "typed_queries", []) or [])
        query_count_mode = str(generation.get("query_count_mode", ""))
        target = _normalize_fraction_for_gate(generation.get("target_coverage"))
        final_coverage = _normalize_fraction_for_gate(getattr(workload, "coverage_fraction", None))
        coverage_mode = str(generation.get("coverage_calibration_mode", ""))
        coverage_guard_enabled = bool(generation.get("coverage_guard_enabled", False))
        stop_reason = str(generation.get("stop_reason", ""))
        is_calibrated_query_count_mode = query_count_mode == "calibrated_to_coverage"
        row_min_queries_per_workload = (
            1
            if gate_mode == "smoke" and is_calibrated_query_count_mode
            else min_queries_per_workload
        )
        acceptance = (getattr(workload, "generation_diagnostics", None) or {}).get(
            "range_acceptance", {}
        )
        if not isinstance(acceptance, dict):
            acceptance = {}
        row_failed: list[str] = []
        if profile_id != required_profile_id:
            row_failed.append("wrong_workload_profile")
        if mode != "target_coverage":
            row_failed.append("not_target_coverage_generation")
        if coverage_mode != "profile_sampled_query_count":
            row_failed.append("coverage_calibration_not_profile_sampled")
        if query_count < row_min_queries_per_workload:
            row_failed.append("too_few_queries")
        if gate_mode != "smoke":
            if (
                bool(acceptance.get("exhausted", False))
                or stop_reason == "range_acceptance_exhausted"
                or stop_reason == "range_coverage_guard_exhausted"
            ):
                row_failed.append("range_acceptance_or_coverage_guard_exhausted")
            attempts = int(acceptance.get("attempts", 0) or 0)
            rejected = int(acceptance.get("rejected", 0) or 0)
            accepted = int(acceptance.get("accepted", 0) or 0)
            rejection_rate = float(rejected / max(1, attempts))
            if attempts > 0 and rejection_rate > 0.85:
                row_failed.append("range_generation_rejection_rate_too_high")
            coverage_rejections = int(
                (acceptance.get("rejection_reasons", {}) or {}).get("coverage_overshoot", 0)
            )
            if accepted > 0 and coverage_rejections / max(1, accepted) > 2.0:
                row_failed.append("coverage_guard_rejection_pressure_too_high")
        if not coverage_guard_enabled:
            row_failed.append("coverage_guard_disabled")
        coverage_target_satisfied = False
        if (
            configured_target is not None
            and target is not None
            and abs(target - configured_target) > 1e-9
        ):
            row_failed.append("target_coverage_mismatch")
        if target is not None and final_coverage is not None:
            if final_coverage + coverage_tolerance < target:
                row_failed.append("coverage_below_target")
            if (
                overshoot is not None
                and final_coverage > min(1.0, target + overshoot) + coverage_tolerance
            ):
                row_failed.append("coverage_above_guard")
            coverage_target_satisfied = final_coverage + coverage_tolerance >= target and (
                overshoot is None
                or final_coverage <= min(1.0, target + overshoot) + coverage_tolerance
            )
        else:
            row_failed.append("missing_coverage_fields")
        if stop_reason != "target_coverage_reached" and not coverage_target_satisfied:
            row_failed.append("target_coverage_not_reached")
        failed_checks.extend(f"{label}:{check}" for check in row_failed)
        workload_rows.append(
            {
                "label": label,
                "profile_id": profile_id,
                "mode": mode,
                "coverage_calibration_mode": coverage_mode,
                "query_count_mode": query_count_mode,
                "query_count": int(query_count),
                "target_coverage": target,
                "final_coverage": final_coverage,
                "coverage_guard_enabled": coverage_guard_enabled,
                "stop_reason": stop_reason,
                "range_acceptance": acceptance,
                "coverage_target_satisfied": bool(coverage_target_satisfied),
                "failed_checks": row_failed,
            }
        )

    return {
        "schema_version": 1,
        "gate_pass": not failed_checks,
        "failed_checks": failed_checks,
        "configured_target_coverage": configured_target,
        "allowed_coverage_targets": list(allowed_coverage_targets),
        "configured_target_in_grid": bool(configured_target_in_grid),
        "gate_mode": gate_mode,
        "train_workload_replicate_count": len(train_label_workloads),
        "min_train_workload_replicates": int(min_train_replicates),
        "min_queries_per_workload": int(min_queries_per_workload),
        "range_max_coverage_overshoot": overshoot,
        "max_allowed_coverage_overshoot": max_allowed_overshoot,
        "required_profile_id": required_profile_id,
        "workloads": workload_rows,
    }


def _coverage_overshoot_tolerance_for_target(target: float | None) -> float | None:
    """Return guide-recommended absolute coverage overshoot tolerance."""
    if target is None:
        return None
    if target <= 0.05:
        return 0.005
    if target <= 0.10:
        return 0.0075
    if target <= 0.15:
        return 0.010
    return 0.020


def _global_sanity_gate(
    *,
    primary: MethodEvaluation,
    uniform: MethodEvaluation | None,
    compression_ratio: float,
) -> dict[str, Any]:
    """Return final-candidate geometry sanity gate evidence."""
    failed_checks: list[str] = []
    length_min = 0.80
    length_max = 1.20
    sed_ratio_threshold = (
        2.00
        if compression_ratio <= 0.01 + 1e-12
        else 1.75
        if compression_ratio <= 0.02 + 1e-12
        else 1.50
    )

    length_preserved = float(primary.avg_length_preserved)
    if length_preserved < length_min or length_preserved > length_max:
        failed_checks.append("length_preservation_outside_range")

    endpoint_sanity_raw = (
        primary.range_audit.get("endpoint_sanity")
        if isinstance(primary.range_audit, dict)
        else None
    )
    endpoint_sanity = _normalize_fraction_for_gate(endpoint_sanity_raw)
    if endpoint_sanity is None:
        failed_checks.append("endpoint_sanity_missing")
    elif endpoint_sanity < 1.0 - 1e-12:
        failed_checks.append("endpoints_not_retained_for_all_eligible_trajectories")

    primary_avg_sed = _optional_float_for_gate(primary.geometric_distortion.get("avg_sed_km"))
    uniform_avg_sed = (
        _optional_float_for_gate(uniform.geometric_distortion.get("avg_sed_km"))
        if uniform is not None
        else None
    )
    if primary_avg_sed is None or uniform_avg_sed is None:
        sed_ratio = None
        failed_checks.append("avg_sed_ratio_missing")
    elif uniform_avg_sed <= 1e-12:
        sed_ratio = 1.0 if primary_avg_sed <= 1e-12 else float("inf")
    else:
        sed_ratio = float(primary_avg_sed / uniform_avg_sed)
    if sed_ratio is not None and sed_ratio > sed_ratio_threshold + 1e-12:
        failed_checks.append("avg_sed_ratio_vs_uniform_too_high")

    return {
        "schema_version": 1,
        "gate_pass": not failed_checks,
        "failed_checks": failed_checks,
        "compression_ratio": float(compression_ratio),
        "endpoint_sanity": endpoint_sanity,
        "endpoint_sanity_required": 1.0,
        "avg_length_preserved": length_preserved,
        "length_preservation_min": length_min,
        "length_preservation_max": length_max,
        "avg_sed_km": primary_avg_sed,
        "uniform_avg_sed_km": uniform_avg_sed,
        "avg_sed_ratio_vs_uniform": sed_ratio,
        "avg_sed_ratio_vs_uniform_max": sed_ratio_threshold,
        "catastrophic_geometry_outlier_fraction": None,
        "catastrophic_geometry_outlier_fraction_max": 0.05,
        "catastrophic_geometry_outlier_status": "not_available_report_only",
    }


def _target_diffusion_gate(target_diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Return gate evidence for labels that are too diffuse for low-budget ranking."""
    max_support_fraction = 0.50
    min_top5_mass_fraction = 0.10
    final_support_threshold_key = "gt_0.01"
    default_head_support_threshold_key = "gt_0.01"
    head_support_threshold_keys = {
        "boundary_event_utility": "gt_0.05",
        "conditional_behavior_utility": "gt_0.01",
        "replacement_representative_value": "gt_0.05",
    }
    blocking_heads = frozenset(head_support_threshold_keys)
    low_budget_key = "0.05"
    failed_checks: list[str] = []

    factorized = target_diagnostics.get("query_useful_v1_factorized")
    if not isinstance(factorized, dict):
        return {
            "schema_version": 1,
            "gate_pass": False,
            "failed_checks": ["query_useful_v1_factorized_diagnostics_missing"],
            "final_support_threshold_key": final_support_threshold_key,
            "max_support_fraction": max_support_fraction,
            "min_top5_label_mass_fraction": min_top5_mass_fraction,
            "blocking_heads": sorted(blocking_heads),
            "head_rows": [],
        }

    final_supports = factorized.get("final_label_support_fraction_by_threshold")
    final_support = None
    if isinstance(final_supports, dict):
        final_support = _optional_float_for_gate(final_supports.get(final_support_threshold_key))
    if final_support is None:
        final_support = _optional_float_for_gate(factorized.get("final_label_positive_fraction"))
    if final_support is None:
        failed_checks.append("final_label_support_fraction_missing")
    elif final_support > max_support_fraction + 1e-12:
        failed_checks.append("final_label_support_fraction_above_max")

    support_by_head = factorized.get("support_fraction_by_threshold_by_head")
    if not isinstance(support_by_head, dict):
        support_by_head = {}
    positive_by_head = factorized.get("positive_fraction_by_head")
    if not isinstance(positive_by_head, dict):
        positive_by_head = {}
    topk_by_head = factorized.get("topk_label_mass_budget_grid")
    if not isinstance(topk_by_head, dict):
        topk_by_head = {}

    head_names = sorted(set(support_by_head) | set(positive_by_head) | set(topk_by_head))
    head_rows: list[dict[str, Any]] = []
    if not head_names:
        failed_checks.append("head_diffusion_diagnostics_missing")
    for head_name in head_names:
        blocking = head_name in blocking_heads
        support_threshold_key = head_support_threshold_keys.get(
            head_name, default_head_support_threshold_key
        )
        head_supports = support_by_head.get(head_name)
        support_fraction = None
        if isinstance(head_supports, dict):
            support_fraction = _optional_float_for_gate(head_supports.get(support_threshold_key))
        if support_fraction is None:
            support_fraction = _optional_float_for_gate(positive_by_head.get(head_name))

        topk_grid = topk_by_head.get(head_name)
        top5_mass = (
            _optional_float_for_gate(topk_grid.get(low_budget_key))
            if isinstance(topk_grid, dict)
            else None
        )
        head_failed: list[str] = []
        if support_fraction is None:
            head_failed.append("support_fraction_missing")
        elif support_fraction > max_support_fraction + 1e-12:
            head_failed.append("support_fraction_above_max")
        if top5_mass is None:
            head_failed.append("top5_label_mass_missing")
        elif top5_mass < min_top5_mass_fraction - 1e-12:
            head_failed.append("top5_label_mass_below_min")
        if blocking:
            failed_checks.extend(f"{head_name}:{check}" for check in head_failed)
        head_rows.append(
            {
                "head": str(head_name),
                "blocking": bool(blocking),
                "support_threshold_key": support_threshold_key,
                "support_fraction": support_fraction,
                "top5_label_mass_fraction": top5_mass,
                "failed_checks": head_failed,
            }
        )

    return {
        "schema_version": 1,
        "gate_pass": not failed_checks,
        "failed_checks": failed_checks,
        "final_support_threshold_key": final_support_threshold_key,
        "default_head_support_threshold_key": default_head_support_threshold_key,
        "head_support_threshold_keys": head_support_threshold_keys,
        "blocking_heads": sorted(blocking_heads),
        "max_support_fraction": max_support_fraction,
        "min_top5_label_mass_fraction": min_top5_mass_fraction,
        "final_label_support_fraction": final_support,
        "head_rows": head_rows,
    }
