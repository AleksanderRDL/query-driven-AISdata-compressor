"""Derived selection-to-eval segment teacher transfer diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

PRIMARY_METHOD = "MLQDS"
BASELINE_METHOD = "DouglasPeucker"
SELECTOR_TRACE_PATH = "selector_trace_diagnostics"
SELECTION_TRACE_NAME = "selection_primary"
EVAL_TRACE_NAME = "eval_primary"
MARGINAL_ALIGNMENT_KEY = "retained_decision_marginal_query_local_utility_alignment"
SEGMENT_FEATURES = (
    "segment_score",
    "segment_allocation_weight",
    "segment_length_support_score",
    "segment_allocation_count",
    "learned_count",
    "length_repair_count",
)
TOP_FRACTIONS = (0.01, 0.05, 0.10)
LOW_TOP_OVERLAP_MAX = 0.10
WEAK_SPEARMAN_ABS_MAX = 0.05


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        value_float = float(value)
        return value_float if math.isfinite(value_float) else None
    return None


def as_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)))


def _rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for sorted_index in range(index, end):
            ranks[indexed[sorted_index][0]] = average_rank
        index = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_denom = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_denom = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_denom == 0.0 or right_denom == 0.0:
        return None
    return float(numerator / (left_denom * right_denom))


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return _pearson(_rankdata(left), _rankdata(right))


def score_summary(artifact: dict[str, Any]) -> dict[str, float | None]:
    matched = as_dict(artifact.get("matched"))
    primary = as_dict(matched.get(PRIMARY_METHOD))
    uniform = as_dict(matched.get("uniform"))
    baseline = as_dict(matched.get(BASELINE_METHOD))
    primary_score = as_float(primary.get("query_local_utility_score"))
    uniform_score = as_float(uniform.get("query_local_utility_score"))
    baseline_score = as_float(baseline.get("query_local_utility_score"))
    return {
        "primary_query_local_utility": primary_score,
        "uniform_query_local_utility": uniform_score,
        "baseline_query_local_utility": baseline_score,
        "primary_minus_uniform_query_local_utility": (
            None if primary_score is None or uniform_score is None else primary_score - uniform_score
        ),
        "primary_minus_baseline_query_local_utility": (
            None
            if primary_score is None or baseline_score is None
            else primary_score - baseline_score
        ),
    }


def gate_summary(artifact: dict[str, Any]) -> dict[str, bool | None]:
    workload_signature = as_dict(
        as_dict(artifact.get("workload_distribution_comparison")).get(
            "workload_signature_gate"
        )
    )
    return {
        "workload_stability": as_bool(
            as_dict(artifact.get("workload_stability_gate")).get("gate_pass")
        ),
        "support_overlap": as_bool(
            as_dict(artifact.get("support_overlap_gate")).get("gate_pass")
        ),
        "target_diffusion": as_bool(
            as_dict(artifact.get("target_diffusion_gate")).get("gate_pass")
        ),
        "workload_signature": as_bool(workload_signature.get("all_pass")),
        "predictability": as_bool(as_dict(artifact.get("predictability_audit")).get("gate_pass")),
        "learning_causality": as_bool(
            as_dict(artifact.get("learning_causality_summary")).get(
                "learning_causality_gate_pass"
            )
        ),
        "global_sanity": as_bool(as_dict(artifact.get("global_sanity_gate")).get("gate_pass")),
        "final_success_allowed": as_bool(
            as_dict(artifact.get("final_claim_summary")).get("final_success_allowed")
        ),
    }


def trace(artifact: dict[str, Any], name: str) -> dict[str, Any]:
    return as_dict(as_dict(artifact.get(SELECTOR_TRACE_PATH)).get(name))


def _alignment(trace: dict[str, Any]) -> dict[str, Any]:
    return as_dict(trace.get(MARGINAL_ALIGNMENT_KEY))


def _teacher_summary(trace: dict[str, Any]) -> dict[str, Any]:
    separated = as_dict(_alignment(trace).get("separated_marginal_teacher_summary"))
    return {
        "available": as_bool(separated.get("available")),
        "teacher_usage_split": separated.get("teacher_usage_split"),
        "teacher_target_shape_viable": as_bool(
            separated.get("teacher_target_shape_viable")
        ),
        "candidate_for_train_side_teacher": as_bool(
            separated.get("candidate_for_train_side_teacher")
        ),
        "candidate_for_train_side_teacher_reason": separated.get(
            "candidate_for_train_side_teacher_reason"
        ),
        "segment_target_count": separated.get("segment_target_count"),
        "point_target_count": separated.get("point_target_count"),
    }


def segment_rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _as_list(as_dict(trace.get("segment_source_attribution")).get("rows"))
    return [row for row in rows if isinstance(row, dict)]


def teacher_segment_targets(trace: dict[str, Any]) -> dict[int, float]:
    separated = as_dict(_alignment(trace).get("separated_marginal_teacher_summary"))
    targets: dict[int, float] = {}
    for row in _as_list(separated.get("segment_target_rows")):
        if not isinstance(row, dict):
            continue
        segment_index = row.get("segment_index")
        target = as_float(row.get("segment_target"))
        if isinstance(segment_index, int) and target is not None:
            targets[segment_index] = max(targets.get(segment_index, 0.0), target)
    return targets


def target_vector(rows: list[dict[str, Any]], targets: dict[int, float]) -> list[float]:
    out: list[float] = []
    for row in rows:
        segment_index = row.get("segment_index")
        out.append(targets.get(segment_index, 0.0) if isinstance(segment_index, int) else 0.0)
    return out


def _feature_vector(rows: list[dict[str, Any]], feature: str) -> list[float] | None:
    values: list[float] = []
    for row in rows:
        value = as_float(row.get(feature))
        if value is None:
            return None
        values.append(value)
    return values


def _top_indices_by_values(
    values_by_segment: dict[int, float],
    *,
    fraction: float,
) -> set[int]:
    if not values_by_segment:
        return set()
    count = max(1, math.ceil(len(values_by_segment) * fraction))
    return {
        segment
        for segment, _value in sorted(
            values_by_segment.items(), key=lambda item: (-item[1], item[0])
        )[:count]
    }


def _top_target_segments(targets: dict[int, float], *, fraction: float) -> set[int]:
    positive = {segment: value for segment, value in targets.items() if value > 0.0}
    return _top_indices_by_values(positive, fraction=fraction)


def _feature_values_by_segment(
    rows: list[dict[str, Any]], feature: str
) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in rows:
        segment_index = row.get("segment_index")
        value = as_float(row.get(feature))
        if isinstance(segment_index, int) and value is not None:
            out[segment_index] = value
    return out


def feature_topk_target_lift(
    *,
    rows: list[dict[str, Any]],
    targets: dict[int, float],
    feature: str,
    fraction: float,
) -> dict[str, float | int | None]:
    values_by_segment = _feature_values_by_segment(rows, feature)
    top_segments = _top_indices_by_values(values_by_segment, fraction=fraction)
    all_targets = [targets.get(segment, 0.0) for segment in values_by_segment]
    top_targets = [targets.get(segment, 0.0) for segment in top_segments]
    base_mean = mean(all_targets)
    top_mean = mean(top_targets)
    return {
        "fraction": fraction,
        "top_segment_count": len(top_segments),
        "top_target_mean": top_mean,
        "base_target_mean": base_mean,
        "top_target_lift": (
            None
            if base_mean is None or base_mean == 0.0 or top_mean is None
            else top_mean / base_mean
        ),
    }


def _split_feature_alignment(trace: dict[str, Any]) -> dict[str, Any]:
    rows = segment_rows(trace)
    targets = teacher_segment_targets(trace)
    target_values = target_vector(rows, targets)
    feature_rows: dict[str, Any] = {}
    for feature in SEGMENT_FEATURES:
        feature_values = _feature_vector(rows, feature)
        if feature_values is None:
            feature_rows[feature] = {"available": False, "reason": "missing_feature"}
            continue
        feature_rows[feature] = {
            "available": True,
            "spearman_with_segment_teacher_target": spearman(feature_values, target_values),
            "pearson_with_segment_teacher_target": _pearson(feature_values, target_values),
            "topk_target_lift": {
                str(fraction): feature_topk_target_lift(
                    rows=rows,
                    targets=targets,
                    feature=feature,
                    fraction=fraction,
                )
                for fraction in TOP_FRACTIONS
            },
        }
    positive_targets = [value for value in target_values if value > 0.0]
    return {
        "segment_candidate_count": len(rows),
        "positive_segment_target_count": len(positive_targets),
        "positive_segment_target_fraction": (
            None if not rows else len(positive_targets) / len(rows)
        ),
        "segment_teacher_target_mean": mean(target_values),
        "segment_teacher_target_std": _std(target_values),
        "feature_alignment": feature_rows,
    }


def _target_overlap(
    selection_targets: dict[int, float],
    eval_targets: dict[int, float],
) -> dict[str, Any]:
    selection_positive = {segment for segment, value in selection_targets.items() if value > 0.0}
    eval_positive = {segment for segment, value in eval_targets.items() if value > 0.0}
    overlap = selection_positive & eval_positive
    out: dict[str, Any] = {
        "segment_index_overlap_is_heuristic": True,
        "selection_positive_count": len(selection_positive),
        "eval_positive_count": len(eval_positive),
        "positive_overlap_count": len(overlap),
        "positive_overlap_fraction_of_selection": (
            None if not selection_positive else len(overlap) / len(selection_positive)
        ),
        "positive_overlap_fraction_of_eval": (
            None if not eval_positive else len(overlap) / len(eval_positive)
        ),
    }
    for fraction in TOP_FRACTIONS:
        selection_top = _top_target_segments(selection_targets, fraction=fraction)
        eval_top = _top_target_segments(eval_targets, fraction=fraction)
        top_overlap = selection_top & eval_top
        out[f"top_{fraction:g}_overlap_count"] = len(top_overlap)
        out[f"top_{fraction:g}_overlap_fraction_of_selection"] = (
            None if not selection_top else len(top_overlap) / len(selection_top)
        )
    common_segments = sorted(set(selection_targets) | set(eval_targets))
    selection_values = [selection_targets.get(segment, 0.0) for segment in common_segments]
    eval_values = [eval_targets.get(segment, 0.0) for segment in common_segments]
    out["selection_eval_teacher_target_spearman"] = spearman(selection_values, eval_values)
    out["selection_eval_teacher_target_pearson"] = _pearson(selection_values, eval_values)
    return out


def _feature_transfer_summary(
    selection_alignment: dict[str, Any],
    eval_alignment: dict[str, Any],
) -> dict[str, Any]:
    selection_features = as_dict(selection_alignment.get("feature_alignment"))
    eval_features = as_dict(eval_alignment.get("feature_alignment"))
    rows: list[dict[str, Any]] = []
    consistent_positive = 0
    contradictory = 0
    weak_both = 0
    for feature in SEGMENT_FEATURES:
        selection_feature = as_dict(selection_features.get(feature))
        eval_feature = as_dict(eval_features.get(feature))
        selection_spearman = as_float(
            selection_feature.get("spearman_with_segment_teacher_target")
        )
        eval_spearman = as_float(eval_feature.get("spearman_with_segment_teacher_target"))
        if selection_spearman is None or eval_spearman is None:
            status = "missing"
        elif (
            abs(selection_spearman) <= WEAK_SPEARMAN_ABS_MAX
            and abs(eval_spearman) <= WEAK_SPEARMAN_ABS_MAX
        ):
            status = "weak_both"
            weak_both += 1
        elif selection_spearman > 0.0 and eval_spearman > 0.0:
            status = "consistent_positive"
            consistent_positive += 1
        elif selection_spearman < 0.0 and eval_spearman < 0.0:
            status = "consistent_negative"
        else:
            status = "contradictory_sign"
            contradictory += 1
        rows.append(
            {
                "feature": feature,
                "selection_spearman": selection_spearman,
                "eval_spearman": eval_spearman,
                "status": status,
            }
        )
    return {
        "feature_rows": rows,
        "consistent_positive_feature_count": consistent_positive,
        "contradictory_feature_count": contradictory,
        "weak_both_feature_count": weak_both,
    }


def _artifact_summary(
    label: str,
    artifact: dict[str, Any],
    *,
    source_trace_name: str = SELECTION_TRACE_NAME,
    eval_trace_name: str = EVAL_TRACE_NAME,
) -> dict[str, Any]:
    selection_trace = trace(artifact, source_trace_name)
    eval_trace = trace(artifact, eval_trace_name)
    selection_targets = teacher_segment_targets(selection_trace)
    eval_targets = teacher_segment_targets(eval_trace)
    selection_alignment = _split_feature_alignment(selection_trace)
    eval_alignment = _split_feature_alignment(eval_trace)
    return {
        "label": label,
        "scores": score_summary(artifact),
        "gates": gate_summary(artifact),
        "selection_teacher": _teacher_summary(selection_trace),
        "eval_teacher": _teacher_summary(eval_trace),
        "target_overlap": _target_overlap(selection_targets, eval_targets),
        "selection_feature_alignment": selection_alignment,
        "eval_feature_alignment": eval_alignment,
        "feature_transfer_summary": _feature_transfer_summary(
            selection_alignment, eval_alignment
        ),
    }


def _decision(summary: dict[str, Any]) -> str:
    selection_teacher = as_dict(summary.get("selection_teacher"))
    if selection_teacher.get("candidate_for_train_side_teacher") is not True:
        return "add_split_eligible_selection_teacher_before_transfer_diagnosis"
    overlap = as_dict(summary.get("target_overlap"))
    feature_transfer = as_dict(summary.get("feature_transfer_summary"))
    top_overlap = as_float(overlap.get("top_0.1_overlap_fraction_of_selection"))
    target_spearman = as_float(overlap.get("selection_eval_teacher_target_spearman"))
    consistent_positive = int(feature_transfer.get("consistent_positive_feature_count") or 0)
    contradictory = int(feature_transfer.get("contradictory_feature_count") or 0)
    if (
        top_overlap is not None
        and top_overlap <= LOW_TOP_OVERLAP_MAX
        and (target_spearman is None or target_spearman <= WEAK_SPEARMAN_ABS_MAX)
    ):
        return "diagnose_transfer_features_before_guarded_calibration_probe"
    if consistent_positive == 0 or contradictory > consistent_positive:
        return "calibrate_teacher_features_before_training_semantics"
    return "guarded_selection_segment_calibration_probe_admissible"


def build_selection_eval_segment_teacher_transfer_diagnostic(
    artifacts: list[tuple[str, dict[str, Any]]],
    *,
    source_trace_name: str = SELECTION_TRACE_NAME,
    eval_trace_name: str = EVAL_TRACE_NAME,
) -> dict[str, Any]:
    """Build a derived diagnostic for selection-to-eval segment teacher transfer."""
    summaries = [
        _artifact_summary(
            label,
            artifact,
            source_trace_name=source_trace_name,
            eval_trace_name=eval_trace_name,
        )
        for label, artifact in artifacts
    ]
    for summary in summaries:
        summary["decision"] = _decision(summary)
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
            "selection_layout": (
                f"{SELECTOR_TRACE_PATH}.{source_trace_name}.{MARGINAL_ALIGNMENT_KEY}"
            ),
            "eval_layout": f"{SELECTOR_TRACE_PATH}.{eval_trace_name}.{MARGINAL_ALIGNMENT_KEY}",
            "decision": primary.get("decision"),
            "decision_scope": "primary_artifact_last_input",
            "interpretation": (
                "Derived diagnosis only. Treats non-teacher segments as zero target to "
                "test whether sparse selection-side segment marginals transfer to eval "
                "or to query-free selector features."
            ),
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON artifact: {path}")
    return payload


def _parse_labeled_artifact(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path)
    path = Path(value)
    return path.parent.name, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a selection-to-eval segment teacher transfer diagnostic."
    )
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="Artifact path, optionally label=path. The last artifact is primary.",
    )
    parser.add_argument(
        "--source_trace_name",
        default=SELECTION_TRACE_NAME,
        choices=["train_primary", "selection_primary"],
        help="Selector trace to treat as the train/checkpoint-side teacher source.",
    )
    parser.add_argument(
        "--eval_trace_name",
        default=EVAL_TRACE_NAME,
        choices=[EVAL_TRACE_NAME],
        help="Selector trace to treat as eval-side transfer target.",
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args(argv)

    artifacts = [
        (label, _load_json(path))
        for label, path in (_parse_labeled_artifact(value) for value in args.artifact)
    ]
    diagnostic = build_selection_eval_segment_teacher_transfer_diagnostic(
        artifacts,
        source_trace_name=str(args.source_trace_name),
        eval_trace_name=str(args.eval_trace_name),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(diagnostic, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
