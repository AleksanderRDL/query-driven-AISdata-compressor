"""Run-JSON audit extractors for benchmark report rows."""

from __future__ import annotations

from typing import Any

from benchmarking.common import LOW_COMPRESSION_THRESHOLD, as_float, audit_ratio_prefix
from benchmarking.reporting.metrics import RANGE_USEFULNESS_GAP_VARIANT_KEYS


def _csv_path_list(raw: Any) -> tuple[str, ...]:
    """Parse benchmark CSV path-list fields for report metadata."""
    if raw is None:
        return ()
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _data_source_row_fields(data_sources: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten train/validation/eval CSV source metadata into one report row."""
    data_sources = data_sources or {}
    csv_path = data_sources.get("csv_path")
    train_csv_path = data_sources.get("train_csv_path")
    validation_csv_path = data_sources.get("validation_csv_path")
    eval_csv_path = data_sources.get("eval_csv_path")
    selected_files = tuple(
        str(path) for path in data_sources.get("selected_cleaned_csv_files") or ()
    )
    return {
        "csv_path": csv_path,
        "train_csv_path": train_csv_path,
        "validation_csv_path": validation_csv_path,
        "eval_csv_path": eval_csv_path,
        "csv_file_count": len(_csv_path_list(csv_path)),
        "train_csv_file_count": len(_csv_path_list(train_csv_path)),
        "validation_csv_file_count": len(_csv_path_list(validation_csv_path)),
        "eval_csv_file_count": len(_csv_path_list(eval_csv_path)),
        "selected_cleaned_csv_file_count": len(selected_files),
        "selected_cleaned_csv_files": ";".join(selected_files),
    }


def _audit_summary(run_json: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize multi-compression RangeUseful audit wins and low-ratio failures."""
    audit = (run_json or {}).get("range_compression_audit") or {}
    if not isinstance(audit, dict):
        audit = {}
    ratios: list[float] = []
    uniform_deltas: list[float] = []
    dp_deltas: list[float] = []
    random_fill_deltas: list[float] = []
    query_uniform_deltas: list[float] = []
    query_dp_deltas: list[float] = []
    low_uniform_deltas: list[float] = []
    low_dp_deltas: list[float] = []
    low_random_fill_deltas: list[float] = []
    low_query_uniform_deltas: list[float] = []
    low_query_dp_deltas: list[float] = []
    variant_uniform_deltas: dict[str, list[float]] = {
        suffix: [] for suffix, _metric_key in RANGE_USEFULNESS_GAP_VARIANT_KEYS
    }
    variant_low_uniform_deltas: dict[str, list[float]] = {
        suffix: [] for suffix, _metric_key in RANGE_USEFULNESS_GAP_VARIANT_KEYS
    }
    missing_baseline_count = 0
    missing_temporal_random_fill_count = 0
    missing_query_local_utility_count = 0
    per_ratio_fields: dict[str, Any] = {}

    ratio_rows: list[tuple[float, dict[str, Any]]] = []
    for raw_ratio, methods in audit.items():
        if not isinstance(methods, dict):
            continue
        try:
            ratio = float(raw_ratio)
        except TypeError, ValueError:
            missing_baseline_count += 1
            continue
        ratio_rows.append((ratio, methods))

    for ratio, methods in sorted(ratio_rows, key=lambda item: item[0]):
        mlqds = methods.get("MLQDS") or {}
        uniform = methods.get("uniform") or {}
        dp = methods.get("DouglasPeucker") or {}
        random_fill = methods.get("TemporalRandomFill") or {}
        mlqds_score = as_float(mlqds.get("range_usefulness_score"))
        uniform_score = as_float(uniform.get("range_usefulness_score"))
        dp_score = as_float(dp.get("range_usefulness_score"))
        random_fill_score = as_float(random_fill.get("range_usefulness_score"))
        if mlqds_score is None or uniform_score is None or dp_score is None:
            missing_baseline_count += 1
            continue
        prefix = audit_ratio_prefix(ratio)
        ratios.append(ratio)
        uniform_delta = mlqds_score - uniform_score
        dp_delta = mlqds_score - dp_score
        uniform_deltas.append(uniform_delta)
        dp_deltas.append(dp_delta)
        random_fill_delta: float | None = None
        if random_fill_score is None:
            missing_temporal_random_fill_count += 1
        else:
            random_fill_delta = mlqds_score - random_fill_score
            random_fill_deltas.append(random_fill_delta)
        mlqds_query_score = as_float(mlqds.get("query_local_utility_score"))
        uniform_query_score = as_float(uniform.get("query_local_utility_score"))
        dp_query_score = as_float(dp.get("query_local_utility_score"))
        query_uniform_delta: float | None = None
        query_dp_delta: float | None = None
        query_fields: dict[str, Any] = {}
        if mlqds_query_score is None or uniform_query_score is None or dp_query_score is None:
            missing_query_local_utility_count += 1
        else:
            query_uniform_delta = float(mlqds_query_score - uniform_query_score)
            query_dp_delta = float(mlqds_query_score - dp_query_score)
            query_uniform_deltas.append(query_uniform_delta)
            query_dp_deltas.append(query_dp_delta)
            query_fields = {
                f"{prefix}_mlqds_query_local_utility": float(mlqds_query_score),
                f"{prefix}_uniform_query_local_utility": float(uniform_query_score),
                f"{prefix}_douglas_peucker_query_local_utility": float(dp_query_score),
                f"{prefix}_mlqds_vs_uniform_query_local_utility": query_uniform_delta,
                f"{prefix}_mlqds_vs_douglas_peucker_query_local_utility": query_dp_delta,
            }
        variant_fields: dict[str, Any] = {}
        for suffix, metric_key in RANGE_USEFULNESS_GAP_VARIANT_KEYS:
            mlqds_variant = as_float(mlqds.get(metric_key))
            uniform_variant = as_float(uniform.get(metric_key))
            if mlqds_variant is None or uniform_variant is None:
                continue
            variant_delta = mlqds_variant - uniform_variant
            variant_uniform_deltas[suffix].append(float(variant_delta))
            if ratio <= LOW_COMPRESSION_THRESHOLD:
                variant_low_uniform_deltas[suffix].append(float(variant_delta))
            variant_fields.update(
                {
                    f"{prefix}_mlqds_vs_uniform_range_usefulness_{suffix}": float(variant_delta),
                }
            )
        per_ratio_fields.update(
            {
                f"{prefix}_compression_ratio": float(ratio),
                f"{prefix}_mlqds_range_usefulness": float(mlqds_score),
                f"{prefix}_uniform_range_usefulness": float(uniform_score),
                f"{prefix}_douglas_peucker_range_usefulness": float(dp_score),
                f"{prefix}_temporal_random_fill_range_usefulness": random_fill_score,
                f"{prefix}_mlqds_vs_uniform_range_usefulness": float(uniform_delta),
                f"{prefix}_mlqds_vs_douglas_peucker_range_usefulness": float(dp_delta),
                f"{prefix}_mlqds_vs_temporal_random_fill_range_usefulness": random_fill_delta,
                **query_fields,
                **variant_fields,
            }
        )
        if ratio <= LOW_COMPRESSION_THRESHOLD:
            low_uniform_deltas.append(uniform_delta)
            low_dp_deltas.append(dp_delta)
            if random_fill_delta is not None:
                low_random_fill_deltas.append(random_fill_delta)
            if query_uniform_delta is not None:
                low_query_uniform_deltas.append(query_uniform_delta)
            if query_dp_delta is not None:
                low_query_dp_deltas.append(query_dp_delta)

    def _mean(values: list[float]) -> float | None:
        return float(sum(values) / len(values)) if values else None

    low_both = [
        1
        for uniform_delta, dp_delta in zip(low_uniform_deltas, low_dp_deltas, strict=True)
        if uniform_delta > 0.0 and dp_delta > 0.0
    ]
    all_both = [
        1
        for uniform_delta, dp_delta in zip(uniform_deltas, dp_deltas, strict=True)
        if uniform_delta > 0.0 and dp_delta > 0.0
    ]
    query_low_both = [
        1
        for uniform_delta, dp_delta in zip(
            low_query_uniform_deltas,
            low_query_dp_deltas,
            strict=True,
        )
        if uniform_delta > 0.0 and dp_delta > 0.0
    ]
    query_all_both = [
        1
        for uniform_delta, dp_delta in zip(query_uniform_deltas, query_dp_deltas, strict=True)
        if uniform_delta > 0.0 and dp_delta > 0.0
    ]
    summary: dict[str, Any] = {
        "audit_compression_ratio_count": len(ratios),
        "audit_low_compression_ratio_count": len(low_uniform_deltas),
        "audit_missing_baseline_count": int(missing_baseline_count),
        "audit_missing_temporal_random_fill_count": int(missing_temporal_random_fill_count),
        "audit_missing_query_local_utility_count": int(missing_query_local_utility_count),
        "audit_beats_uniform_range_usefulness_count": sum(
            1 for value in uniform_deltas if value > 0.0
        ),
        "audit_beats_douglas_peucker_range_usefulness_count": sum(
            1 for value in dp_deltas if value > 0.0
        ),
        "audit_beats_temporal_random_fill_range_usefulness_count": sum(
            1 for value in random_fill_deltas if value > 0.0
        ),
        "audit_beats_both_range_usefulness_count": len(all_both),
        "audit_low_beats_uniform_range_usefulness_count": sum(
            1 for value in low_uniform_deltas if value > 0.0
        ),
        "audit_low_beats_douglas_peucker_range_usefulness_count": sum(
            1 for value in low_dp_deltas if value > 0.0
        ),
        "audit_low_beats_temporal_random_fill_range_usefulness_count": sum(
            1 for value in low_random_fill_deltas if value > 0.0
        ),
        "audit_low_beats_both_range_usefulness_count": len(low_both),
        "audit_beats_uniform_query_local_utility_count": sum(
            1 for value in query_uniform_deltas if value > 0.0
        ),
        "audit_beats_douglas_peucker_query_local_utility_count": sum(
            1 for value in query_dp_deltas if value > 0.0
        ),
        "audit_beats_both_query_local_utility_count": len(query_all_both),
        "audit_low_beats_uniform_query_local_utility_count": sum(
            1 for value in low_query_uniform_deltas if value > 0.0
        ),
        "audit_low_beats_douglas_peucker_query_local_utility_count": sum(
            1 for value in low_query_dp_deltas if value > 0.0
        ),
        "audit_low_beats_both_query_local_utility_count": len(query_low_both),
        "audit_min_vs_uniform_range_usefulness": min(uniform_deltas) if uniform_deltas else None,
        "audit_mean_vs_uniform_range_usefulness": _mean(uniform_deltas),
        "audit_min_vs_uniform_query_local_utility": min(query_uniform_deltas)
        if query_uniform_deltas
        else None,
        "audit_mean_vs_uniform_query_local_utility": _mean(query_uniform_deltas),
        "audit_min_vs_temporal_random_fill_range_usefulness": (
            min(random_fill_deltas) if random_fill_deltas else None
        ),
        "audit_mean_vs_temporal_random_fill_range_usefulness": _mean(random_fill_deltas),
        "audit_min_low_vs_uniform_range_usefulness": min(low_uniform_deltas)
        if low_uniform_deltas
        else None,
        "audit_mean_low_vs_uniform_range_usefulness": _mean(low_uniform_deltas),
        "audit_min_low_vs_uniform_query_local_utility": (
            min(low_query_uniform_deltas) if low_query_uniform_deltas else None
        ),
        "audit_mean_low_vs_uniform_query_local_utility": _mean(low_query_uniform_deltas),
        "audit_min_low_vs_temporal_random_fill_range_usefulness": (
            min(low_random_fill_deltas) if low_random_fill_deltas else None
        ),
        "audit_mean_low_vs_temporal_random_fill_range_usefulness": _mean(low_random_fill_deltas),
    }
    for suffix, _metric_key in RANGE_USEFULNESS_GAP_VARIANT_KEYS:
        deltas = variant_uniform_deltas[suffix]
        low_deltas = variant_low_uniform_deltas[suffix]
        summary.update(
            {
                f"audit_beats_uniform_range_usefulness_{suffix}_count": sum(
                    1 for value in deltas if value > 0.0
                ),
                f"audit_low_beats_uniform_range_usefulness_{suffix}_count": sum(
                    1 for value in low_deltas if value > 0.0
                ),
                f"audit_min_vs_uniform_range_usefulness_{suffix}": min(deltas) if deltas else None,
                f"audit_mean_vs_uniform_range_usefulness_{suffix}": _mean(deltas),
                f"audit_min_low_vs_uniform_range_usefulness_{suffix}": (
                    min(low_deltas) if low_deltas else None
                ),
                f"audit_mean_low_vs_uniform_range_usefulness_{suffix}": _mean(low_deltas),
            }
        )
    summary.update(per_ratio_fields)
    return summary


def _target_budget_row(
    target_diagnostics: dict[str, Any], compression_ratio: Any
) -> dict[str, Any]:
    """Return the target-diagnostics budget row closest to the run compression ratio."""
    rows = target_diagnostics.get("budget_rows") or []
    if not isinstance(rows, list) or not rows:
        return {}
    try:
        target_ratio = float(compression_ratio)
    except TypeError, ValueError:
        last_row = rows[-1]
        return last_row if isinstance(last_row, dict) else {}

    best_row: dict[str, Any] = {}
    best_distance = float("inf")
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_ratio = row.get("total_budget_ratio")
        if raw_ratio is None:
            continue
        try:
            ratio = float(raw_ratio)
        except TypeError, ValueError:
            continue
        distance = abs(ratio - target_ratio)
        if distance < best_distance:
            best_distance = distance
            best_row = row
    return best_row


def _selector_budget_row(
    selector_diagnostics: dict[str, Any], compression_ratio: Any
) -> dict[str, Any]:
    """Return the selector-capacity budget row closest to the run compression ratio."""
    rows = selector_diagnostics.get("budget_rows") or []
    if not isinstance(rows, list) or not rows:
        return {}
    try:
        target_ratio = float(compression_ratio)
    except TypeError, ValueError:
        last_row = rows[-1]
        return last_row if isinstance(last_row, dict) else {}

    best_row: dict[str, Any] = {}
    best_distance = float("inf")
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_ratio = row.get("compression_ratio")
        if raw_ratio is None:
            continue
        try:
            ratio = float(raw_ratio)
        except TypeError, ValueError:
            continue
        distance = abs(ratio - target_ratio)
        if distance < best_distance:
            best_distance = distance
            best_row = row
    return best_row


def _selector_low_budget_summary(selector_diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Summarize learned-slot capacity over low compression ratios."""
    rows = []
    for row in selector_diagnostics.get("budget_rows") or []:
        if not isinstance(row, dict):
            continue
        ratio = as_float(row.get("compression_ratio"))
        if ratio is not None and ratio <= LOW_COMPRESSION_THRESHOLD:
            rows.append(row)
    if not rows:
        return {
            "eval_selector_low_budget_zero_learned_ratio_count": None,
            "eval_selector_low_budget_min_learned_slot_fraction": None,
        }
    learned_fractions = [float(row.get("learned_slot_fraction_of_budget") or 0.0) for row in rows]
    return {
        "eval_selector_low_budget_zero_learned_ratio_count": sum(
            1 for row in rows if int(row.get("learned_slot_count") or 0) <= 0
        ),
        "eval_selector_low_budget_min_learned_slot_fraction": min(learned_fractions),
    }


def _query_generation(run_json: dict[str, Any] | None, split: str) -> dict[str, Any]:
    """Return query-generation diagnostics for one split."""
    diagnostics = (run_json or {}).get("query_generation_diagnostics") or {}
    split_payload = diagnostics.get(split) or {}
    if not isinstance(split_payload, dict):
        return {}
    query_generation = split_payload.get("query_generation") or {}
    return query_generation if isinstance(query_generation, dict) else {}


def _range_acceptance(run_json: dict[str, Any] | None, split: str) -> dict[str, Any]:
    """Return range-acceptance diagnostics for one split."""
    diagnostics = (run_json or {}).get("query_generation_diagnostics") or {}
    split_payload = diagnostics.get(split) or {}
    if not isinstance(split_payload, dict):
        return {}
    acceptance = split_payload.get("range_acceptance") or {}
    return acceptance if isinstance(acceptance, dict) else {}


def _range_acceptance_health_fields(prefix: str, acceptance: dict[str, Any]) -> dict[str, Any]:
    """Return final-profile generator health fields for one split."""
    attempts = as_float(acceptance.get("attempts"))
    accepted = as_float(acceptance.get("accepted"))
    rejected = as_float(acceptance.get("rejected"))
    reasons = acceptance.get("rejection_reasons") or {}
    if not isinstance(reasons, dict):
        reasons = {}
    coverage_rejections = as_float(reasons.get("coverage_overshoot")) or 0.0
    rejection_rate = (
        float(rejected) / max(1.0, float(attempts))
        if rejected is not None and attempts is not None
        else None
    )
    coverage_guard_rejection_pressure = (
        float(coverage_rejections) / max(1.0, float(accepted)) if accepted is not None else None
    )
    return {
        f"{prefix}_range_acceptance_enabled": acceptance.get("enabled"),
        f"{prefix}_range_acceptance_exhausted": acceptance.get("exhausted"),
        f"{prefix}_range_acceptance_attempts": attempts,
        f"{prefix}_range_acceptance_accepted": accepted,
        f"{prefix}_range_acceptance_rejected": rejected,
        f"{prefix}_range_generation_rejection_rate": rejection_rate,
        f"{prefix}_coverage_guard_rejection_count": coverage_rejections,
        f"{prefix}_coverage_guard_rejection_pressure": coverage_guard_rejection_pressure,
        f"{prefix}_range_rejection_reasons": reasons,
        f"{prefix}_range_rejection_reasons_by_anchor_family": acceptance.get(
            "rejection_reasons_by_anchor_family"
        ),
        f"{prefix}_range_rejection_reasons_by_footprint_family": acceptance.get(
            "rejection_reasons_by_footprint_family"
        ),
    }


def _profile_query_plan_fields(prefix: str, generation: dict[str, Any]) -> dict[str, Any]:
    """Return planned workload-family quota fields for one split."""
    plan = generation.get("profile_query_plan") or {}
    if not isinstance(plan, dict):
        plan = {}
    return {
        f"{prefix}_profile_query_plan_enabled": plan.get("enabled"),
        f"{prefix}_profile_query_plan_requested_queries": plan.get("requested_queries"),
        f"{prefix}_profile_anchor_family_planned_counts": plan.get("anchor_family_planned_counts"),
        f"{prefix}_profile_footprint_family_planned_counts": plan.get(
            "footprint_family_planned_counts"
        ),
    }


def _workload_distribution_summary(run_json: dict[str, Any] | None, split: str) -> dict[str, Any]:
    """Return workload distribution summary for one split."""
    summaries = ((run_json or {}).get("workload_distribution_comparison") or {}).get(
        "summaries"
    ) or {}
    summary = summaries.get(split) or {}
    return summary if isinstance(summary, dict) else {}


def _query_floor_fields(prefix: str, generation: dict[str, Any]) -> dict[str, Any]:
    """Return coverage-target query-floor diagnostics for one split."""
    target_coverage = as_float(generation.get("target_coverage"))
    final_coverage = as_float(generation.get("final_coverage"))
    final_query_count = as_float(generation.get("final_query_count"))
    target_reached_query_count = as_float(generation.get("target_reached_query_count"))
    extra_after_target = as_float(generation.get("extra_queries_after_target_reached"))
    target_reached = (
        None
        if target_coverage is None or final_coverage is None
        else bool(final_coverage + 1e-12 >= target_coverage)
    )
    target_shortfall = (
        None
        if target_coverage is None or final_coverage is None
        else max(0.0, target_coverage - final_coverage)
    )
    target_overshoot = (
        None
        if target_coverage is None or final_coverage is None
        else max(0.0, final_coverage - target_coverage)
    )
    return {
        f"{prefix}_query_generation_mode": generation.get("mode"),
        f"{prefix}_query_generation_stop_reason": generation.get("stop_reason"),
        f"{prefix}_query_coverage_calibration_mode": generation.get("coverage_calibration_mode"),
        f"{prefix}_query_target_coverage": generation.get("target_coverage"),
        f"{prefix}_query_final_coverage": generation.get("final_coverage"),
        f"{prefix}_query_target_reached": target_reached,
        f"{prefix}_query_target_shortfall": target_shortfall,
        f"{prefix}_query_target_overshoot": target_overshoot,
        f"{prefix}_query_target_missed_by_max_queries": (
            bool(
                generation.get("stop_reason") == "max_queries_reached"
                and target_shortfall
                and target_shortfall > 0.0
            )
            if target_shortfall is not None
            else None
        ),
        f"{prefix}_query_minimum_queries": generation.get("minimum_queries"),
        f"{prefix}_query_max_queries": generation.get("max_queries"),
        f"{prefix}_query_final_count": generation.get("final_query_count"),
        f"{prefix}_query_target_reached_count": generation.get("target_reached_query_count"),
        f"{prefix}_query_coverage_at_target_reached": generation.get("coverage_at_target_reached"),
        f"{prefix}_query_extra_after_target_reached": generation.get(
            "extra_queries_after_target_reached"
        ),
        f"{prefix}_query_extra_after_target_fraction": (
            float(extra_after_target) / float(final_query_count)
            if extra_after_target is not None and final_query_count and final_query_count > 0.0
            else None
        ),
        f"{prefix}_query_floor_dominated": (
            bool(
                target_reached_query_count is not None
                and final_query_count is not None
                and final_query_count > target_reached_query_count
            )
        ),
        f"{prefix}_query_coverage_guard_enabled": generation.get("coverage_guard_enabled"),
        f"{prefix}_query_max_allowed_coverage": generation.get("max_allowed_coverage"),
    }


def _workload_generation_fields(run_json: dict[str, Any] | None, split: str) -> dict[str, Any]:
    """Flatten query-generation and workload-distribution diagnostics for one split."""
    generation = _query_generation(run_json, split)
    acceptance = _range_acceptance(run_json, split)
    summary = _workload_distribution_summary(run_json, split)
    fields = _query_floor_fields(split, generation)
    fields.update(_range_acceptance_health_fields(split, acceptance))
    fields.update(_profile_query_plan_fields(split, generation))
    fields.update(
        {
            f"{split}_workload_range_query_count": summary.get("range_query_count"),
            f"{split}_workload_coverage_fraction": summary.get("coverage_fraction"),
            f"{split}_workload_empty_query_rate": summary.get("empty_query_rate"),
            f"{split}_workload_too_broad_query_rate": summary.get("too_broad_query_rate"),
            f"{split}_workload_near_duplicate_query_rate": summary.get("near_duplicate_query_rate"),
            f"{split}_workload_point_hit_count_p50": summary.get("point_hit_count_p50"),
            f"{split}_workload_trajectory_hit_count_p50": summary.get("trajectory_hit_count_p50"),
            f"{split}_workload_oracle_gap_over_best_baseline": summary.get(
                "oracle_gap_over_best_baseline"
            ),
            f"{split}_workload_best_baseline": summary.get("best_baseline"),
        }
    )
    return fields
