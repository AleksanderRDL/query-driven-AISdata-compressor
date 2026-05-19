"""Retained-decision marginal alignment summary helpers."""

from __future__ import annotations

import math
from typing import Any

from orchestration import selector_trace_payloads


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / float(len(left))
    right_mean = sum(right) / float(len(right))
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    if left_var <= 1e-12 or right_var <= 1e-12:
        return None
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    return float(covariance / math.sqrt(left_var * right_var))


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: float(values[idx]))
    ranks = [0.0 for _value in values]
    cursor = 0
    while cursor < len(order):
        end = cursor
        while end + 1 < len(order) and values[order[end + 1]] == values[order[cursor]]:
            end += 1
        average_rank = float(cursor + end + 2) / 2.0
        for rank_index in range(cursor, end + 1):
            ranks[order[rank_index]] = average_rank
        cursor = end + 1
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return _pearson(_average_ranks(left), _average_ranks(right))


def mean_or_none(values: list[float]) -> float | None:
    return float(sum(values) / float(len(values))) if values else None


def value_marginal_alignment_summary(
    values: list[float],
    marginal_values: list[float],
) -> dict[str, Any]:
    if len(values) != len(marginal_values) or len(values) < 2:
        return {
            "available": False,
            "reason": "fewer_than_two_values",
            "count": min(len(values), len(marginal_values)),
        }
    value_min = min(values)
    value_max = max(values)
    if value_max - value_min <= 1e-12:
        return {
            "available": False,
            "reason": "no_value_variation",
            "count": len(values),
            "value_min": value_min,
            "value_max": value_max,
        }
    order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)
    bucket_count = max(1, len(order) // 4)
    top = [marginal_values[idx] for idx in order[:bucket_count]]
    bottom = [marginal_values[idx] for idx in order[-bucket_count:]]
    top_mean = mean_or_none(top)
    bottom_mean = mean_or_none(bottom)
    return {
        "available": True,
        "count": len(values),
        "pearson": _pearson(values, marginal_values),
        "spearman": _spearman(values, marginal_values),
        "top_quartile_mean_marginal": top_mean,
        "bottom_quartile_mean_marginal": bottom_mean,
        "top_minus_bottom_marginal": (
            None if top_mean is None or bottom_mean is None else float(top_mean - bottom_mean)
        ),
        "value_min": value_min,
        "value_max": value_max,
    }


def score_alignment_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"candidate_count": len(rows)}
    if not rows:
        return summary
    marginals = [float(row["marginal_query_local_utility"]) for row in rows]
    summary.update(
        {
            "mean_marginal_query_local_utility": mean_or_none(marginals),
            "positive_marginal_fraction": float(
                sum(1 for value in marginals if value > 0.0) / float(len(marginals))
            ),
            "max_marginal_query_local_utility": max(marginals),
            "min_marginal_query_local_utility": min(marginals),
        }
    )
    for score_key in ("raw_score", "selector_score", "segment_score"):
        valid = [
            (float(row[score_key]), float(row["marginal_query_local_utility"]))
            for row in rows
            if row.get(score_key) is not None
        ]
        summary[score_key] = value_marginal_alignment_summary(
            [score for score, _marginal in valid],
            [marginal for _score, marginal in valid],
        )
    component_summary = nested_value_alignment_summary(rows, "score_components")
    if component_summary:
        summary["score_component_alignment"] = component_summary
    query_free_proxy_summary = nested_value_alignment_summary(rows, "query_free_teacher_proxies")
    if query_free_proxy_summary:
        summary["query_free_teacher_proxy_alignment"] = query_free_proxy_summary
    return summary


def nested_value_alignment_summary(
    rows: list[dict[str, Any]],
    row_field_name: str,
) -> dict[str, Any]:
    value_names = sorted(
        {str(name) for row in rows for name in (row.get(row_field_name) or {}).keys()}
    )
    if not value_names:
        return {}
    nested_summary: dict[str, Any] = {}
    for value_name in value_names:
        valid = [
            (
                float(row[row_field_name][value_name]),
                float(row["marginal_query_local_utility"]),
            )
            for row in rows
            if row.get(row_field_name) is not None
            and row[row_field_name].get(value_name) is not None
        ]
        nested_summary[value_name] = value_marginal_alignment_summary(
            [score for score, _marginal in valid],
            [marginal for _score, marginal in valid],
        )
    return nested_summary


def group_rows_by_field(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "")), []).append(row)
    return grouped


def guard_owned_retained_row(row: dict[str, Any]) -> bool:
    if str(row.get("decision")) != "retained_removal_loss":
        return False
    source = str(row.get("source", ""))
    stage_state = row.get("selector_stage_state") or {}
    return (
        source in {"skeleton", "length_repair", "fallback"}
        or bool(stage_state.get("skeleton_retained", False))
        or bool(stage_state.get("length_repair_retained", False))
        or bool(stage_state.get("fallback_retained", False))
    )


def _compact_proxy_subset_alignment_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"candidate_count": len(rows)}
    if not rows:
        return summary
    marginals = [float(row["marginal_query_local_utility"]) for row in rows]
    summary.update(
        {
            "mean_marginal_query_local_utility": mean_or_none(marginals),
            "positive_marginal_fraction": float(
                sum(1 for value in marginals if value > 0.0) / float(len(marginals))
            ),
            "max_marginal_query_local_utility": max(marginals),
            "min_marginal_query_local_utility": min(marginals),
        }
    )
    for score_key in ("raw_score", "selector_score", "segment_score"):
        valid = [
            (float(row[score_key]), float(row["marginal_query_local_utility"]))
            for row in rows
            if row.get(score_key) is not None
        ]
        summary[score_key] = value_marginal_alignment_summary(
            [score for score, _marginal in valid],
            [marginal for _score, marginal in valid],
        )
    proxy_summary = nested_value_alignment_summary(rows, "query_free_teacher_proxies")
    if proxy_summary:
        summary["query_free_teacher_proxy_alignment"] = proxy_summary
    return summary


def query_free_teacher_proxy_guard_coupling_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {
            "available": False,
            "diagnostic_only": True,
            "reason": "no_retained_marginal_rows",
        }

    retained_rows = [row for row in rows if row.get("decision") == "retained_removal_loss"]
    guard_rows = [row for row in retained_rows if guard_owned_retained_row(row)]
    learned_controllable_rows = [
        row
        for row in retained_rows
        if str(row.get("source")) == "learned" and not guard_owned_retained_row(row)
    ]
    non_guard_rows = [row for row in retained_rows if not guard_owned_retained_row(row)]
    removed_rows = [row for row in rows if row.get("decision") == "removed_addition_gain"]

    subsets = {
        "all_retained_removal": retained_rows,
        "learned_controllable_retained_removal": learned_controllable_rows,
        "non_guard_retained_removal": non_guard_rows,
        "guard_owned_retained_removal": guard_rows,
        "skeleton_retained_removal": [
            row for row in retained_rows if str(row.get("source")) == "skeleton"
        ],
        "length_repair_retained_removal": [
            row for row in retained_rows if str(row.get("source")) == "length_repair"
        ],
        "removed_addition_gain": removed_rows,
    }
    subset_summaries = {
        name: _compact_proxy_subset_alignment_summary(subset_rows)
        for name, subset_rows in subsets.items()
    }

    def endpoint_top_minus(subset_name: str) -> float | None:
        summary = subset_summaries.get(subset_name, {})
        proxy_summary = summary.get("query_free_teacher_proxy_alignment")
        if not isinstance(proxy_summary, dict):
            return None
        endpoint_summary = proxy_summary.get("query_free_endpoint_support")
        if not isinstance(endpoint_summary, dict):
            return None
        value = endpoint_summary.get("top_minus_bottom_marginal")
        return (
            float(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else None
        )

    all_top_minus = endpoint_top_minus("all_retained_removal")
    learned_top_minus = endpoint_top_minus("learned_controllable_retained_removal")
    guard_top_minus = endpoint_top_minus("guard_owned_retained_removal")
    guard_coupling_suspected = (
        all_top_minus is not None
        and all_top_minus > 0.0
        and (
            learned_top_minus is None
            or learned_top_minus <= 0.0
            or (guard_top_minus is not None and guard_top_minus > learned_top_minus)
        )
    )

    return {
        "available": True,
        "diagnostic_only": True,
        "proxy_family": "query_free_teacher_proxies",
        "primary_proxy": "query_free_endpoint_support",
        "retained_removal_count": len(retained_rows),
        "learned_controllable_retained_removal_count": len(learned_controllable_rows),
        "non_guard_retained_removal_count": len(non_guard_rows),
        "guard_owned_retained_removal_count": len(guard_rows),
        "subsets": subset_summaries,
        "endpoint_proxy_guard_coupling": {
            "available": all_top_minus is not None,
            "rule": (
                "Suspect guard coupling when endpoint proxy alignment is positive "
                "overall but absent, nonpositive, or weaker on learned-controllable "
                "retained-removal rows."
            ),
            "all_retained_top_minus_bottom_marginal": all_top_minus,
            "learned_controllable_top_minus_bottom_marginal": learned_top_minus,
            "guard_owned_top_minus_bottom_marginal": guard_top_minus,
            "guard_coupling_suspected": bool(guard_coupling_suspected),
        },
    }


def learned_controllable_marginal_teacher_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    retained_rows = [row for row in rows if row.get("decision") == "retained_removal_loss"]
    learned_rows = [
        row
        for row in retained_rows
        if str(row.get("source")) == "learned" and not guard_owned_retained_row(row)
    ]
    summary = score_alignment_summary(learned_rows)
    marginals = [float(row["marginal_query_local_utility"]) for row in learned_rows]
    value_variation = (max(marginals) - min(marginals)) if len(marginals) >= 2 else 0.0
    usable_candidate = len(marginals) >= 2 and value_variation > 1e-12
    summary.update(
        {
            "available": bool(learned_rows),
            "diagnostic_only": True,
            "teacher_signal": "exact_retained_removal_marginal_query_local_utility",
            "teacher_scope": "learned_controllable_retained_removal",
            "decision": "retained_removal_loss",
            "source": "learned",
            "guard_exclusion_policy": (
                "Excludes skeleton, fallback, length-repair, and any retained row whose "
                "selector stage state marks skeleton/fallback/length-repair ownership."
            ),
            "query_conditioned_teacher_requires_train_or_selection_workload": True,
            "eval_time_feature_allowed": False,
            "retained_removal_count": len(retained_rows),
            "learned_controllable_retained_removal_count": len(learned_rows),
            "teacher_value_variation": float(value_variation),
            "candidate_for_train_side_calibration": bool(usable_candidate),
        }
    )
    if not learned_rows:
        summary["reason"] = "no_learned_controllable_retained_removal_rows"
    elif not usable_candidate:
        summary["reason"] = "insufficient_teacher_value_variation"
    return summary


TRAIN_OR_CHECKPOINT_TEACHER_USAGE_SPLITS = frozenset({"train", "checkpoint_selection"})


def separated_teacher_candidate_rejection_reason(
    *,
    teacher_usage_split: str,
    teacher_target_shape_viable: bool,
) -> str:
    if not teacher_target_shape_viable:
        return "insufficient_teacher_target_shape"
    if str(teacher_usage_split).startswith("eval"):
        return "eval_split_query_conditioned_teacher_not_allowed_for_training"
    if str(teacher_usage_split) == "unknown":
        return "unknown_split_query_conditioned_teacher_not_allowed_for_training"
    return "split_not_allowed_for_train_or_checkpoint_teacher"


def separated_marginal_teacher_targets(
    rows: list[dict[str, Any]],
    *,
    teacher_usage_split: str = "unknown",
) -> dict[str, Any]:
    retained_rows = [row for row in rows if row.get("decision") == "retained_removal_loss"]
    learned_rows = [
        row
        for row in retained_rows
        if str(row.get("source")) == "learned" and not guard_owned_retained_row(row)
    ]
    contextual_rows = [
        row for row in learned_rows if isinstance(row.get("selector_segment_context"), dict)
    ]
    usage_split = str(teacher_usage_split)
    usage_allowed = usage_split in TRAIN_OR_CHECKPOINT_TEACHER_USAGE_SPLITS
    summary: dict[str, Any] = {
        "available": False,
        "diagnostic_only": True,
        "teacher_signal": "exact_retained_removal_marginal_query_local_utility",
        "teacher_scope": "learned_controllable_retained_removal",
        "teacher_shape": "separated_segment_and_within_segment_point_targets",
        "teacher_usage_split": usage_split,
        "teacher_usage_allowed_for_train_or_checkpoint": bool(usage_allowed),
        "teacher_target_shape_viable": False,
        "decision": "retained_removal_loss",
        "source": "learned",
        "eval_time_feature_allowed": False,
        "query_conditioned_teacher_requires_train_or_selection_workload": True,
        "guard_exclusion_policy": (
            "Excludes skeleton, fallback, length-repair, and any retained row whose "
            "selector stage state marks skeleton/fallback/length-repair ownership."
        ),
        "retained_removal_count": len(retained_rows),
        "learned_controllable_retained_removal_count": len(learned_rows),
        "rows_with_selector_segment_context": len(contextual_rows),
        "rows_missing_selector_segment_context": max(0, len(learned_rows) - len(contextual_rows)),
        "candidate_for_train_side_teacher": False,
        "candidate_for_train_side_teacher_reason": (
            separated_teacher_candidate_rejection_reason(
                teacher_usage_split=usage_split,
                teacher_target_shape_viable=False,
            )
        ),
    }
    if not learned_rows:
        summary["reason"] = "no_learned_controllable_retained_removal_rows"
        return summary
    if not contextual_rows:
        summary["reason"] = "missing_selector_segment_context"
        return summary

    grouped: dict[tuple[int | None, int | None, int | None, int | None], list[dict[str, Any]]] = {}
    for row in contextual_rows:
        context = row.get("selector_segment_context") or {}
        key = (
            selector_trace_payloads.optional_int(context.get("trajectory_index")),
            selector_trace_payloads.optional_int(context.get("segment_index")),
            selector_trace_payloads.optional_int(context.get("segment_start")),
            selector_trace_payloads.optional_int(context.get("segment_end")),
        )
        grouped.setdefault(key, []).append(row)

    max_segment_sum = 0.0
    max_point_positive = 0.0
    for group_rows in grouped.values():
        positives = [
            max(0.0, float(row.get("marginal_query_local_utility", 0.0))) for row in group_rows
        ]
        max_segment_sum = max(max_segment_sum, sum(positives))
        max_point_positive = max(max_point_positive, max(positives, default=0.0))

    segment_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    for (trajectory_idx, segment_idx, segment_start, segment_end), group_rows in grouped.items():
        context = group_rows[0].get("selector_segment_context") or {}
        positive_marginals = [
            max(0.0, float(row.get("marginal_query_local_utility", 0.0))) for row in group_rows
        ]
        raw_marginals = [float(row.get("marginal_query_local_utility", 0.0)) for row in group_rows]
        segment_positive_sum = sum(positive_marginals)
        segment_positive_max = max(positive_marginals, default=0.0)
        local_max = max(segment_positive_max, 1e-12)
        ordered_group = sorted(
            group_rows,
            key=lambda row: (
                -max(0.0, float(row.get("marginal_query_local_utility", 0.0))),
                int(row.get("point_index", -1)),
            ),
        )
        segment_rows.append(
            {
                "trajectory_index": trajectory_idx,
                "segment_index": segment_idx,
                "segment_start": segment_start,
                "segment_end": segment_end,
                "segment_length": selector_trace_payloads.optional_int(
                    context.get("segment_length")
                ),
                "row_count": len(group_rows),
                "positive_row_count": sum(1 for value in positive_marginals if value > 0.0),
                "raw_segment_positive_marginal_sum": float(segment_positive_sum),
                "raw_segment_max_point_marginal": float(segment_positive_max),
                "raw_segment_mean_point_marginal": mean_or_none(raw_marginals),
                "segment_target": (
                    float(segment_positive_sum / max_segment_sum)
                    if max_segment_sum > 1e-12
                    else 0.0
                ),
                "selector_segment_score_rank": selector_trace_payloads.optional_int(
                    context.get("segment_score_rank")
                ),
                "selector_segment_length_support_rank": selector_trace_payloads.optional_int(
                    context.get("segment_length_support_rank")
                ),
                "selector_segment_allocation_weight_rank": selector_trace_payloads.optional_int(
                    context.get("segment_allocation_weight_rank")
                ),
                "selector_segment_allocation_count": selector_trace_payloads.optional_int(
                    context.get("segment_allocation_count")
                ),
                "selector_segment_learned_count": selector_trace_payloads.optional_int(
                    context.get("learned_count")
                ),
                "top_point_index": selector_trace_payloads.optional_int(
                    ordered_group[0].get("point_index")
                )
                if ordered_group
                else None,
            }
        )
        for local_rank, row in enumerate(ordered_group, start=1):
            context = row.get("selector_segment_context") or {}
            positive_marginal = max(0.0, float(row.get("marginal_query_local_utility", 0.0)))
            point_rows.append(
                {
                    "point_index": selector_trace_payloads.optional_int(row.get("point_index")),
                    "trajectory_index": selector_trace_payloads.optional_int(
                        row.get("trajectory_index")
                    ),
                    "segment_index": segment_idx,
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "point_offset_in_segment": selector_trace_payloads.optional_int(
                        context.get("point_offset_in_segment")
                    ),
                    "raw_point_marginal": float(row.get("marginal_query_local_utility", 0.0)),
                    "point_target_within_segment": float(positive_marginal / local_max),
                    "point_target_global": (
                        float(positive_marginal / max_point_positive)
                        if max_point_positive > 1e-12
                        else 0.0
                    ),
                    "intra_segment_teacher_rank": int(local_rank),
                    "selector_score_candidate_rank_fraction": selector_trace_payloads.optional_float(
                        row.get("selector_score_candidate_rank_fraction")
                    ),
                    "segment_score_candidate_rank_fraction": selector_trace_payloads.optional_float(
                        row.get("segment_score_candidate_rank_fraction")
                    ),
                    "selector_segment_score_rank": selector_trace_payloads.optional_int(
                        context.get("segment_score_rank")
                    ),
                    "selector_segment_allocation_count": selector_trace_payloads.optional_int(
                        context.get("segment_allocation_count")
                    ),
                }
            )

    point_values = [float(row["raw_point_marginal"]) for row in point_rows]
    teacher_value_variation = max(point_values) - min(point_values) if point_values else 0.0
    positive_segment_target_count = sum(
        1 for row in segment_rows if float(row["raw_segment_positive_marginal_sum"]) > 0.0
    )
    positive_point_target_count = sum(
        1 for row in point_rows if float(row["raw_point_marginal"]) > 0.0
    )
    teacher_target_shape_viable = (
        len(point_rows) >= 2 and positive_point_target_count > 0 and teacher_value_variation > 1e-12
    )
    candidate_for_train_side_teacher = bool(teacher_target_shape_viable and usage_allowed)
    candidate_reason = (
        "candidate_available"
        if candidate_for_train_side_teacher
        else separated_teacher_candidate_rejection_reason(
            teacher_usage_split=usage_split,
            teacher_target_shape_viable=teacher_target_shape_viable,
        )
    )
    segment_rows = sorted(
        segment_rows,
        key=lambda row: (
            -float(row["raw_segment_positive_marginal_sum"]),
            row["trajectory_index"] if row["trajectory_index"] is not None else -1,
            row["segment_index"] if row["segment_index"] is not None else -1,
        ),
    )
    point_rows = sorted(
        point_rows,
        key=lambda row: (
            -float(row["raw_point_marginal"]),
            row["point_index"] if row["point_index"] is not None else -1,
        ),
    )
    summary.update(
        {
            "available": True,
            "reason": None,
            "teacher_value_variation": float(teacher_value_variation),
            "segment_target_normalization": (
                "segment_positive_marginal_sum_div_global_max_segment_sum"
            ),
            "point_target_normalization": (
                "positive_point_marginal_div_segment_max_and_global_max"
            ),
            "segment_target_count": len(segment_rows),
            "point_target_count": len(point_rows),
            "positive_segment_target_count": positive_segment_target_count,
            "positive_point_target_count": positive_point_target_count,
            "segment_target_rows": segment_rows,
            "point_target_rows": point_rows,
            "teacher_target_shape_viable": bool(teacher_target_shape_viable),
            "candidate_for_train_side_teacher": candidate_for_train_side_teacher,
            "candidate_for_train_side_teacher_reason": candidate_reason,
        }
    )
    return summary
