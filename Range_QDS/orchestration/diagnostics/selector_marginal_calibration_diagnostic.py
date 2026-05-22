"""Derived selector-to-retained-marginal calibration diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from orchestration.diagnostics.artifact_utils import (
    as_bool as _as_bool,
)
from orchestration.diagnostics.artifact_utils import (
    as_dict as _as_dict,
)
from orchestration.diagnostics.artifact_utils import (
    as_float as _as_float,
)
from orchestration.diagnostics.artifact_utils import (
    as_list as _as_list,
)
from orchestration.diagnostics.artifact_utils import (
    delta as _delta,
)
from orchestration.diagnostics.artifact_utils import (
    load_json_dict as _load_json,
)

PRIMARY_METHOD = "MLQDS"
BASELINE_METHOD = "DouglasPeucker"
ALIGNMENT_PATH = (
    "selector_trace_diagnostics.eval_primary."
    "retained_decision_marginal_query_local_utility_alignment"
)
SCORE_FIELDS = ("raw_score", "selector_score", "segment_score")
TOP_RANK_FRACTION = 0.25
LOW_RANK_FRACTION = 0.75
MID_RANK_FRACTION = 0.50
TOP_EXAMPLE_LIMIT = 10


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _rank_fraction_from_rank(rank: Any, denominator: Any) -> float | None:
    rank_value = _as_float(rank)
    denominator_value = _as_float(denominator)
    if rank_value is None or denominator_value is None or denominator_value <= 0:
        return None
    return float(rank_value / denominator_value)


def _score_summary(artifact: dict[str, Any]) -> dict[str, float | None]:
    matched = _as_dict(artifact.get("matched"))
    primary = _as_dict(matched.get(PRIMARY_METHOD))
    uniform = _as_dict(matched.get("uniform"))
    baseline = _as_dict(matched.get(BASELINE_METHOD))
    primary_score = _as_float(primary.get("query_local_utility_score"))
    uniform_score = _as_float(uniform.get("query_local_utility_score"))
    baseline_score = _as_float(baseline.get("query_local_utility_score"))
    return {
        "primary_query_local_utility": primary_score,
        "uniform_query_local_utility": uniform_score,
        "baseline_query_local_utility": baseline_score,
        "primary_minus_uniform_query_local_utility": _delta(primary_score, uniform_score),
        "primary_minus_baseline_query_local_utility": _delta(primary_score, baseline_score),
    }


def _gate_summary(artifact: dict[str, Any]) -> dict[str, bool | None]:
    return {
        "target_diffusion": _as_bool(_as_dict(artifact.get("target_diffusion_gate")).get("gate_pass")),
        "predictability": _as_bool(_as_dict(artifact.get("predictability_audit")).get("gate_pass")),
        "learning_causality": _as_bool(
            _as_dict(artifact.get("learning_causality_summary")).get(
                "learning_causality_gate_pass"
            )
        ),
        "global_sanity": _as_bool(_as_dict(artifact.get("global_sanity_gate")).get("gate_pass")),
        "final_success_allowed": _as_bool(
            _as_dict(artifact.get("final_claim_summary")).get("final_success_allowed")
        ),
    }


def _alignment_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(
        _as_dict(_as_dict(artifact.get("selector_trace_diagnostics")).get("eval_primary")).get(
            "retained_decision_marginal_query_local_utility_alignment"
        )
    )


def _score_alignment_subset(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        field: {
            "spearman": _as_float(_as_dict(summary.get(field)).get("spearman")),
            "top_minus_bottom_marginal": _as_float(
                _as_dict(summary.get(field)).get("top_minus_bottom_marginal")
            ),
        }
        for field in SCORE_FIELDS
    }


def _rank_fraction(row: dict[str, Any], score_field: str) -> float | None:
    return _as_float(row.get(f"{score_field}_candidate_rank_fraction"))


def _rank(row: dict[str, Any], score_field: str) -> float | None:
    return _as_float(row.get(f"{score_field}_candidate_rank"))


def _marginal_rank_fraction(row: dict[str, Any]) -> float | None:
    return _as_float(row.get("marginal_query_local_utility_candidate_rank_fraction"))


def _marginal_rank(row: dict[str, Any]) -> float | None:
    return _as_float(row.get("marginal_query_local_utility_candidate_rank"))


def _rank_fraction_at_most(row: dict[str, Any], score_field: str, threshold: float) -> bool:
    value = _rank_fraction(row, score_field)
    return value is not None and value <= threshold


def _rank_fraction_at_least(row: dict[str, Any], score_field: str, threshold: float) -> bool:
    value = _rank_fraction(row, score_field)
    return value is not None and value >= threshold


def _marginal_rank_fraction_at_most(row: dict[str, Any], threshold: float) -> bool:
    value = _marginal_rank_fraction(row)
    return value is not None and value <= threshold


def _marginal_rank_fraction_at_least(row: dict[str, Any], threshold: float) -> bool:
    value = _marginal_rank_fraction(row)
    return value is not None and value >= threshold


def _stage_owner(row: dict[str, Any]) -> str:
    source = str(row.get("source") or "")
    stage = _as_dict(row.get("selector_stage_state"))
    if source == "length_repair" or stage.get("length_repair_retained") is True:
        return "length_repair"
    if source == "skeleton" or stage.get("skeleton_retained") is True:
        return "skeleton"
    if source == "learned" or stage.get("learned_retained") is True:
        return "learned"
    if source == "fallback" or stage.get("fallback_retained") is True:
        return "fallback"
    if source == "removed":
        return "removed"
    return source or "unknown"


def _row_ref(row: dict[str, Any]) -> dict[str, Any]:
    segment_context = _as_dict(row.get("selector_segment_context"))
    return {
        "point_index": row.get("point_index"),
        "trajectory_index": row.get("trajectory_index"),
        "source": row.get("source"),
        "decision": row.get("decision"),
        "stage_owner": _stage_owner(row),
        "marginal_query_local_utility": _as_float(row.get("marginal_query_local_utility")),
        "marginal_rank": _marginal_rank(row),
        "raw_score_rank": _rank(row, "raw_score"),
        "selector_score_rank": _rank(row, "selector_score"),
        "point_segment_score_rank": _rank(row, "segment_score"),
        "raw_score_rank_fraction": _rank_fraction(row, "raw_score"),
        "selector_score_rank_fraction": _rank_fraction(row, "selector_score"),
        "point_segment_score_rank_fraction": _rank_fraction(row, "segment_score"),
        "failure_buckets": [str(item) for item in _as_list(row.get("failure_buckets"))],
        "segment_index": segment_context.get("segment_index"),
        "selector_segment_score_rank": segment_context.get("segment_score_rank"),
        "selector_segment_length_support_rank": segment_context.get(
            "segment_length_support_rank"
        ),
        "segment_allocation_weight_rank": segment_context.get("segment_allocation_weight_rank"),
        "segment_allocation_count": segment_context.get("segment_allocation_count"),
        "selector_segment_learned_count": segment_context.get("learned_count"),
        "selector_segment_length_repair_count": segment_context.get("length_repair_count"),
    }


def _field_under_rank(row: dict[str, Any], score_field: str) -> float | None:
    score_rank = _rank(row, score_field)
    marginal_rank = _marginal_rank(row)
    if score_rank is None or marginal_rank is None:
        return None
    return float(score_rank - marginal_rank)


def _top_marginal_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    top_rows = [
        row
        for row in rows
        if _marginal_rank_fraction_at_most(row, TOP_RANK_FRACTION)
    ]
    out: dict[str, Any] = {
        "top_rank_fraction_max": TOP_RANK_FRACTION,
        "row_count": len(top_rows),
        "by_stage_owner": _count_by(top_rows, _stage_owner),
        "top_rows": [
            _row_ref(row)
            for row in sorted(top_rows, key=_marginal_sort_key)[:TOP_EXAMPLE_LIMIT]
        ],
        "score_under_rank": {},
    }
    for score_field in SCORE_FIELDS:
        deltas = [
            value
            for row in top_rows
            if (value := _field_under_rank(row, score_field)) is not None
        ]
        low_rank_count = sum(
            1
            for row in top_rows
            if _rank_fraction_at_least(row, score_field, MID_RANK_FRACTION)
        )
        very_low_rank_count = sum(
            1
            for row in top_rows
            if _rank_fraction_at_least(row, score_field, LOW_RANK_FRACTION)
        )
        out["score_under_rank"][score_field] = {
            "mean_score_rank_minus_marginal_rank": _mean(deltas),
            "low_rank_fraction_min": MID_RANK_FRACTION,
            "low_ranked_top_marginal_count": int(low_rank_count),
            "very_low_rank_fraction_min": LOW_RANK_FRACTION,
            "very_low_ranked_top_marginal_count": int(very_low_rank_count),
        }
    return out


def _high_score_low_marginal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _marginal_rank_fraction_at_least(row, LOW_RANK_FRACTION)
        and any(
            _rank_fraction_at_most(row, score_field, TOP_RANK_FRACTION)
            for score_field in SCORE_FIELDS
        )
    ]


def _overranked_low_marginal_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overranked = _high_score_low_marginal_rows(rows)
    return {
        "low_marginal_rank_fraction_min": LOW_RANK_FRACTION,
        "top_score_rank_fraction_max": TOP_RANK_FRACTION,
        "row_count": len(overranked),
        "by_stage_owner": _count_by(overranked, _stage_owner),
        "by_decision": _count_by(overranked, lambda row: str(row.get("decision") or "unknown")),
        "examples": [
            _row_ref(row)
            for row in sorted(overranked, key=_overrank_sort_key)[:TOP_EXAMPLE_LIMIT]
        ],
    }


def _bucket_summary(alignment: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    top_summary = _as_dict(alignment.get("top_marginal_miss_summary"))
    bucket_counts = {
        str(name): int(count)
        for name, count in _as_dict(top_summary.get("bucket_counts")).items()
        if isinstance(count, int | float)
    }
    if not bucket_counts:
        for row in rows:
            for bucket in _as_list(row.get("failure_buckets")):
                bucket_counts[str(bucket)] = bucket_counts.get(str(bucket), 0) + 1
    return {
        "bucket_counts": bucket_counts,
        "top_marginal_rows_in_selector_trace_only": _as_bool(
            top_summary.get("top_marginal_rows_in_selector_trace_only")
        ),
    }


def _component_alignment(alignment: dict[str, Any]) -> dict[str, Any]:
    overall = _as_dict(alignment.get("overall"))
    components = _as_dict(overall.get("score_component_alignment"))
    proxies = _as_dict(overall.get("query_free_teacher_proxy_alignment"))
    return {
        "best_score_components_by_spearman": _best_alignment_rows(components, limit=6),
        "best_query_free_proxies_by_spearman": _best_alignment_rows(proxies, limit=4),
    }


def _best_alignment_rows(rows: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, value in rows.items():
        row = _as_dict(value)
        if row.get("available") is not True:
            continue
        out.append(
            {
                "name": str(name),
                "spearman": _as_float(row.get("spearman")),
                "top_minus_bottom_marginal": _as_float(row.get("top_minus_bottom_marginal")),
            }
        )
    return sorted(
        out,
        key=lambda row: (
            -float(row["spearman"] if row["spearman"] is not None else -999.0),
            str(row["name"]),
        ),
    )[:limit]


def _teacher_summary(alignment: dict[str, Any]) -> dict[str, Any]:
    separated = _as_dict(alignment.get("separated_marginal_teacher_summary"))
    learned = _as_dict(alignment.get("learned_controllable_marginal_teacher_summary"))
    guard = _as_dict(alignment.get("query_free_teacher_proxy_guard_coupling_summary"))
    return {
        "separated_teacher_shape_viable": _as_bool(separated.get("teacher_target_shape_viable")),
        "separated_teacher_allowed_for_training": _as_bool(
            separated.get("candidate_for_train_side_teacher")
        ),
        "separated_teacher_rejection_reason": separated.get(
            "candidate_for_train_side_teacher_reason"
        ),
        "learned_controllable_retained_removal_count": learned.get(
            "learned_controllable_retained_removal_count"
        ),
        "guard_coupling_suspected": _as_bool(
            _as_dict(guard.get("endpoint_proxy_guard_coupling")).get(
                "guard_coupling_suspected"
            )
        ),
    }


def _segment_teacher_row_ref(row: dict[str, Any], *, segment_count: int | None) -> dict[str, Any]:
    return {
        "segment_index": row.get("segment_index"),
        "trajectory_index": row.get("trajectory_index"),
        "top_point_index": row.get("top_point_index"),
        "segment_target": _as_float(row.get("segment_target")),
        "raw_segment_positive_marginal_sum": _as_float(
            row.get("raw_segment_positive_marginal_sum")
        ),
        "selector_segment_score_rank": row.get("selector_segment_score_rank"),
        "selector_segment_score_rank_fraction": _rank_fraction_from_rank(
            row.get("selector_segment_score_rank"), segment_count
        ),
        "selector_segment_allocation_weight_rank": row.get(
            "selector_segment_allocation_weight_rank"
        ),
        "selector_segment_allocation_weight_rank_fraction": _rank_fraction_from_rank(
            row.get("selector_segment_allocation_weight_rank"), segment_count
        ),
        "selector_segment_length_support_rank": row.get(
            "selector_segment_length_support_rank"
        ),
        "selector_segment_allocation_count": row.get("selector_segment_allocation_count"),
        "selector_segment_learned_count": row.get("selector_segment_learned_count"),
    }


def _point_teacher_row_ref(row: dict[str, Any], *, segment_count: int | None) -> dict[str, Any]:
    return {
        "point_index": row.get("point_index"),
        "trajectory_index": row.get("trajectory_index"),
        "segment_index": row.get("segment_index"),
        "raw_point_marginal": _as_float(row.get("raw_point_marginal")),
        "point_target_global": _as_float(row.get("point_target_global")),
        "selector_score_rank_fraction": _as_float(
            row.get("selector_score_candidate_rank_fraction")
        ),
        "point_segment_score_rank_fraction": _as_float(
            row.get("segment_score_candidate_rank_fraction")
        ),
        "selector_segment_score_rank": row.get("selector_segment_score_rank"),
        "selector_segment_score_rank_fraction": _rank_fraction_from_rank(
            row.get("selector_segment_score_rank"), segment_count
        ),
        "selector_segment_allocation_count": row.get("selector_segment_allocation_count"),
    }


def _segment_teacher_diagnostics(
    alignment: dict[str, Any], trace: dict[str, Any]
) -> dict[str, Any]:
    separated = _as_dict(alignment.get("separated_marginal_teacher_summary"))
    segment_count = _as_float(trace.get("segments_considered_count"))
    segment_count_int = int(segment_count) if segment_count is not None else None
    segment_rows = [
        row for row in _as_list(separated.get("segment_target_rows")) if isinstance(row, dict)
    ]
    point_rows = [
        row for row in _as_list(separated.get("point_target_rows")) if isinstance(row, dict)
    ]
    top_segment_rows = sorted(
        segment_rows,
        key=lambda row: -float(_as_float(row.get("segment_target")) or 0.0),
    )[:TOP_EXAMPLE_LIMIT]
    top_point_rows = sorted(
        point_rows,
        key=lambda row: -float(_as_float(row.get("raw_point_marginal")) or 0.0),
    )[:TOP_EXAMPLE_LIMIT]
    segment_score_fractions = [
        value
        for row in top_segment_rows
        if (
            value := _rank_fraction_from_rank(
                row.get("selector_segment_score_rank"), segment_count_int
            )
        )
        is not None
    ]
    allocation_weight_fractions = [
        value
        for row in top_segment_rows
        if (
            value := _rank_fraction_from_rank(
                row.get("selector_segment_allocation_weight_rank"), segment_count_int
            )
        )
        is not None
    ]
    low_segment_score_count = sum(
        1 for value in segment_score_fractions if value >= MID_RANK_FRACTION
    )
    low_allocation_weight_count = sum(
        1 for value in allocation_weight_fractions if value >= MID_RANK_FRACTION
    )
    return {
        "available": _as_bool(separated.get("available")),
        "teacher_usage_split": separated.get("teacher_usage_split"),
        "teacher_usage_allowed_for_train_or_checkpoint": _as_bool(
            separated.get("teacher_usage_allowed_for_train_or_checkpoint")
        ),
        "candidate_for_train_side_teacher": _as_bool(
            separated.get("candidate_for_train_side_teacher")
        ),
        "candidate_for_train_side_teacher_reason": separated.get(
            "candidate_for_train_side_teacher_reason"
        ),
        "segment_count_denominator": segment_count_int,
        "segment_target_count": separated.get("segment_target_count"),
        "point_target_count": separated.get("point_target_count"),
        "top_segment_target_row_count": len(top_segment_rows),
        "top_segment_target_low_selector_segment_score_count": int(low_segment_score_count),
        "top_segment_target_low_allocation_weight_count": int(low_allocation_weight_count),
        "top_segment_target_mean_selector_segment_score_rank_fraction": _mean(
            segment_score_fractions
        ),
        "top_segment_target_mean_allocation_weight_rank_fraction": _mean(
            allocation_weight_fractions
        ),
        "top_segment_target_rows": [
            _segment_teacher_row_ref(row, segment_count=segment_count_int)
            for row in top_segment_rows
        ],
        "top_point_target_rows": [
            _point_teacher_row_ref(row, segment_count=segment_count_int)
            for row in top_point_rows
        ],
    }


def _allocation_diagnostics(trace: dict[str, Any]) -> dict[str, Any]:
    alignment = _as_dict(trace.get("segment_allocation_alignment_diagnostics"))
    point_selection = _as_dict(trace.get("allocation_point_selection_diagnostics"))
    counterfactual = _as_dict(trace.get("allocation_counterfactual_diagnostics"))
    top_groups = _as_dict(alignment.get("top_groups"))
    top_10 = _as_dict(top_groups.get("top_10_percent"))
    top_20 = _as_dict(top_groups.get("top_20_percent"))
    return {
        "segment_allocation_alignment": {
            "available": _as_bool(alignment.get("available")),
            "component_diagnosis": alignment.get("component_diagnosis"),
            "segment_count": alignment.get("segment_count"),
            "allocation_count_total": alignment.get("allocation_count_total"),
            "extra_allocation_count_total": alignment.get("extra_allocation_count_total"),
            "length_support_to_allocation_spearman": _as_float(
                alignment.get("length_support_to_allocation_spearman")
            ),
            "segment_score_to_allocation_spearman": _as_float(
                alignment.get("segment_score_to_allocation_spearman")
            ),
            "allocation_weight_to_allocation_spearman": _as_float(
                alignment.get("allocation_weight_to_allocation_spearman")
            ),
            "top_10_length_support_segment_score_overlap_fraction": _as_float(
                top_10.get("length_support_segment_score_overlap_fraction")
            ),
            "top_20_length_support_segment_score_overlap_fraction": _as_float(
                top_20.get("length_support_segment_score_overlap_fraction")
            ),
        },
        "allocation_point_selection": {
            "available": _as_bool(point_selection.get("available")),
            "component_diagnosis": point_selection.get("component_diagnosis"),
            "primary_length_preservation": _as_float(
                point_selection.get("primary_length_preservation")
            ),
            "same_allocation_length_only_point_selection_preservation": _as_float(
                point_selection.get("same_allocation_length_only_point_selection_preservation")
            ),
            "same_allocation_length_only_delta": _as_float(
                point_selection.get("same_allocation_length_only_delta")
            ),
            "same_allocation_length_only_gate_would_pass": _as_bool(
                point_selection.get("same_allocation_length_only_gate_would_pass")
            ),
        },
        "allocation_counterfactual": {
            "available": _as_bool(counterfactual.get("available")),
            "component_diagnosis": counterfactual.get("component_diagnosis"),
            "allocation_overlap_fraction": _as_float(
                counterfactual.get("allocation_overlap_fraction")
            ),
            "extra_allocation_overlap_fraction": _as_float(
                counterfactual.get("extra_allocation_overlap_fraction")
            ),
            "length_support_allocation_counterfactual_preservation": _as_float(
                counterfactual.get("length_support_allocation_counterfactual_preservation")
            ),
            "length_support_allocation_counterfactual_gate_would_pass": _as_bool(
                counterfactual.get("length_support_allocation_counterfactual_gate_would_pass")
            ),
        },
    }


def _failure_mode_summary(
    alignment: dict[str, Any], trace: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    overall = _as_dict(alignment.get("overall"))
    selector_spearman = _as_float(_as_dict(overall.get("selector_score")).get("spearman"))
    raw_spearman = _as_float(_as_dict(overall.get("raw_score")).get("spearman"))
    segment_spearman = _as_float(_as_dict(overall.get("segment_score")).get("spearman"))
    top = _top_marginal_diagnostics(rows)
    overranked = _overranked_low_marginal_diagnostics(rows)
    segment_teacher = _segment_teacher_diagnostics(alignment, trace)
    return {
        "negative_score_alignment": any(
            value is not None and value <= 0.0
            for value in (raw_spearman, selector_spearman, segment_spearman)
        ),
        "selector_spearman": selector_spearman,
        "high_score_low_exact_marginal_count": overranked["row_count"],
        "top_exact_marginal_low_selector_score_count": _as_dict(
            _as_dict(top.get("score_under_rank")).get("selector_score")
        ).get("low_ranked_top_marginal_count"),
        "top_exact_marginal_low_segment_score_count": _as_dict(
            _as_dict(top.get("score_under_rank")).get("segment_score")
        ).get("low_ranked_top_marginal_count"),
        "top_segment_target_low_selector_segment_score_count": segment_teacher.get(
            "top_segment_target_low_selector_segment_score_count"
        ),
        "top_segment_target_low_allocation_weight_count": segment_teacher.get(
            "top_segment_target_low_allocation_weight_count"
        ),
    }


def _alignment_summary(alignment: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in _as_list(alignment.get("rows")) if isinstance(row, dict)]
    return {
        "available": _as_bool(alignment.get("available")),
        "source_layout": ALIGNMENT_PATH,
        "candidate_count": alignment.get("candidate_count"),
        "overall": _score_alignment_subset(_as_dict(alignment.get("overall"))),
        "by_decision": {
            name: _score_alignment_subset(_as_dict(value))
            for name, value in _as_dict(alignment.get("by_decision")).items()
        },
        "by_source": {
            name: _score_alignment_subset(_as_dict(value))
            for name, value in _as_dict(alignment.get("by_source")).items()
        },
        "bucket_summary": _bucket_summary(alignment, rows),
        "top_marginal_diagnostics": _top_marginal_diagnostics(rows),
        "overranked_low_marginal_diagnostics": _overranked_low_marginal_diagnostics(rows),
        "component_alignment": _component_alignment(alignment),
        "teacher_summary": _teacher_summary(alignment),
        "segment_marginal_teacher_diagnostics": _segment_teacher_diagnostics(
            alignment, trace
        ),
        "allocation_diagnostics": _allocation_diagnostics(trace),
        "failure_mode_summary": _failure_mode_summary(alignment, trace, rows),
    }


def _count_by(rows: list[dict[str, Any]], key_fn: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(key_fn(row))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _marginal_sort_key(row: dict[str, Any]) -> tuple[float, int]:
    return (-float(row.get("marginal_query_local_utility") or 0.0), int(row.get("point_index") or 0))


def _overrank_sort_key(row: dict[str, Any]) -> tuple[float, int]:
    best_fraction = min(
        [
            value
            for score_field in SCORE_FIELDS
            if (value := _rank_fraction(row, score_field)) is not None
        ],
        default=1.0,
    )
    return (float(best_fraction), int(row.get("point_index") or 0))


def _artifact_summary(label: str, artifact: dict[str, Any]) -> dict[str, Any]:
    trace = _as_dict(_as_dict(artifact.get("selector_trace_diagnostics")).get("eval_primary"))
    alignment = _alignment_payload(artifact)
    return {
        "label": label,
        "scores": _score_summary(artifact),
        "gates": _gate_summary(artifact),
        "retained_marginal_alignment": _alignment_summary(alignment, trace),
    }


def _decision(summary: dict[str, Any]) -> str:
    alignment = _as_dict(summary.get("retained_marginal_alignment"))
    if alignment.get("available") is not True:
        return "add_retained_marginal_rows_before_calibration"
    teacher = _as_dict(alignment.get("teacher_summary"))
    top = _as_dict(alignment.get("top_marginal_diagnostics"))
    overranked = _as_dict(alignment.get("overranked_low_marginal_diagnostics"))
    segment_teacher = _as_dict(alignment.get("segment_marginal_teacher_diagnostics"))
    selector_spearman = _as_float(
        _as_dict(_as_dict(alignment.get("overall")).get("selector_score")).get("spearman")
    )
    top_score_under_rank = _as_dict(top.get("score_under_rank"))
    selector_top_low_count = _as_dict(top_score_under_rank.get("selector_score")).get(
        "low_ranked_top_marginal_count"
    )
    if selector_spearman is not None and selector_spearman <= 0.0:
        if int(segment_teacher.get("top_segment_target_low_selector_segment_score_count") or 0) > 0:
            return "diagnose_train_side_marginal_segment_calibration_not_promotion"
        if int(overranked.get("row_count") or 0) > 0:
            return "diagnose_overranked_low_marginal_scores_before_selector_change"
        if int(selector_top_low_count or 0) > 0:
            return "diagnose_under_ranked_high_marginal_rows_before_selector_change"
        return "diagnose_score_to_marginal_monotonicity_before_selector_change"
    if teacher.get("separated_teacher_shape_viable") is True and teacher.get(
        "separated_teacher_allowed_for_training"
    ) is not True:
        return "construct_train_side_marginal_calibration_evidence_before_promotion"
    return "diagnostic_only_no_promotion"


def build_selector_marginal_calibration_diagnostic(
    artifacts: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Build a derived diagnostic for selector-score to exact-marginal calibration."""
    summaries = [_artifact_summary(label, artifact) for label, artifact in artifacts]
    primary = summaries[-1] if summaries else {}
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "evidence_level": "derived_strict_artifact_diagnostic_no_new_probe",
        "primary_method": PRIMARY_METHOD,
        "baseline_method": BASELINE_METHOD,
        "artifact_count": len(summaries),
        "artifacts": summaries,
        "summary": {
            "primary_label": primary.get("label"),
            "retained_marginal_alignment_layout": ALIGNMENT_PATH,
            "decision": _decision(primary),
            "interpretation": (
                "Derived diagnosis only. Eval exact marginals can localize score/selector "
                "miscalibration, but they are not train-side teacher evidence."
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
        description="Build a selector-to-retained-marginal calibration diagnostic."
    )
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="Artifact path, optionally label=path. The last artifact is treated as primary.",
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args(argv)

    artifacts = [
        (label, _load_json(path))
        for label, path in (_parse_labeled_artifact(value) for value in args.artifact)
    ]
    diagnostic = build_selector_marginal_calibration_diagnostic(artifacts)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(diagnostic, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
