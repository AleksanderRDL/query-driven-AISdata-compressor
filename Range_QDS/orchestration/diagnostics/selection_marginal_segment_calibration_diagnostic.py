"""Derived selection-side marginal segment calibration diagnostics."""

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
    load_json_dict as _load_json,
)

PRIMARY_METHOD = "MLQDS"
BASELINE_METHOD = "DouglasPeucker"
SELECTOR_TRACE_PATH = "selector_trace_diagnostics"
SELECTION_TRACE_NAME = "selection_primary"
EVAL_TRACE_NAME = "eval_primary"
MARGINAL_ALIGNMENT_KEY = "retained_decision_marginal_query_local_utility_alignment"
TOP_SEGMENT_LIMIT = 10
LOW_RANK_FRACTION = 0.50


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _rank_fraction(rank: Any, denominator: Any) -> float | None:
    rank_value = _as_float(rank)
    denominator_value = _as_float(denominator)
    if rank_value is None or denominator_value is None or denominator_value <= 0.0:
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
        "primary_minus_uniform_query_local_utility": (
            None
            if primary_score is None or uniform_score is None
            else primary_score - uniform_score
        ),
        "primary_minus_baseline_query_local_utility": (
            None
            if primary_score is None or baseline_score is None
            else primary_score - baseline_score
        ),
    }


def _gate_summary(artifact: dict[str, Any]) -> dict[str, bool | None]:
    workload_signature = _as_dict(
        _as_dict(artifact.get("workload_distribution_comparison")).get("workload_signature_gate")
    )
    return {
        "workload_stability": _as_bool(
            _as_dict(artifact.get("workload_stability_gate")).get("gate_pass")
        ),
        "support_overlap": _as_bool(
            _as_dict(artifact.get("support_overlap_gate")).get("gate_pass")
        ),
        "target_diffusion": _as_bool(
            _as_dict(artifact.get("target_diffusion_gate")).get("gate_pass")
        ),
        "workload_signature": _as_bool(workload_signature.get("all_pass")),
        "predictability": _as_bool(_as_dict(artifact.get("predictability_audit")).get("gate_pass")),
        "learning_causality": _as_bool(
            _as_dict(artifact.get("learning_causality_summary")).get("learning_causality_gate_pass")
        ),
        "global_sanity": _as_bool(_as_dict(artifact.get("global_sanity_gate")).get("gate_pass")),
        "final_success_allowed": _as_bool(
            _as_dict(artifact.get("final_claim_summary")).get("final_success_allowed")
        ),
    }


def _trace(artifact: dict[str, Any], trace_name: str) -> dict[str, Any]:
    return _as_dict(_as_dict(artifact.get(SELECTOR_TRACE_PATH)).get(trace_name))


def _alignment(trace: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(trace.get(MARGINAL_ALIGNMENT_KEY))


def _segment_rows(alignment: dict[str, Any]) -> list[dict[str, Any]]:
    separated = _as_dict(alignment.get("separated_marginal_teacher_summary"))
    return [row for row in _as_list(separated.get("segment_target_rows")) if isinstance(row, dict)]


def _point_rows(alignment: dict[str, Any]) -> list[dict[str, Any]]:
    separated = _as_dict(alignment.get("separated_marginal_teacher_summary"))
    return [row for row in _as_list(separated.get("point_target_rows")) if isinstance(row, dict)]


def _top_segment_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(_as_float(row.get("segment_target")) or 0.0),
            int(row.get("segment_index") or 0),
        ),
    )[:TOP_SEGMENT_LIMIT]


def _segment_row_ref(row: dict[str, Any], denominator: int | None) -> dict[str, Any]:
    return {
        "segment_index": row.get("segment_index"),
        "trajectory_index": row.get("trajectory_index"),
        "top_point_index": row.get("top_point_index"),
        "segment_target": _as_float(row.get("segment_target")),
        "raw_segment_positive_marginal_sum": _as_float(
            row.get("raw_segment_positive_marginal_sum")
        ),
        "selector_segment_score_rank": row.get("selector_segment_score_rank"),
        "selector_segment_score_rank_fraction": _rank_fraction(
            row.get("selector_segment_score_rank"), denominator
        ),
        "selector_segment_allocation_weight_rank": row.get(
            "selector_segment_allocation_weight_rank"
        ),
        "selector_segment_allocation_weight_rank_fraction": _rank_fraction(
            row.get("selector_segment_allocation_weight_rank"), denominator
        ),
        "selector_segment_length_support_rank": row.get("selector_segment_length_support_rank"),
        "selector_segment_allocation_count": row.get("selector_segment_allocation_count"),
        "selector_segment_learned_count": row.get("selector_segment_learned_count"),
    }


def _trace_teacher_summary(trace_name: str, trace: dict[str, Any]) -> dict[str, Any]:
    alignment = _alignment(trace)
    separated = _as_dict(alignment.get("separated_marginal_teacher_summary"))
    overall = _as_dict(alignment.get("overall"))
    segment_count = _as_float(trace.get("segments_considered_count"))
    segment_count_int = int(segment_count) if segment_count is not None else None
    segment_rows = _segment_rows(alignment)
    point_rows = _point_rows(alignment)
    top_rows = _top_segment_rows(segment_rows)
    segment_rank_fractions = [
        value
        for row in top_rows
        if (value := _rank_fraction(row.get("selector_segment_score_rank"), segment_count_int))
        is not None
    ]
    allocation_rank_fractions = [
        value
        for row in top_rows
        if (
            value := _rank_fraction(
                row.get("selector_segment_allocation_weight_rank"), segment_count_int
            )
        )
        is not None
    ]
    return {
        "trace_name": trace_name,
        "trace_layout": f"{SELECTOR_TRACE_PATH}.{trace_name}.{MARGINAL_ALIGNMENT_KEY}",
        "alignment_available": _as_bool(alignment.get("available")),
        "candidate_count": alignment.get("candidate_count"),
        "selector_score_spearman": _as_float(
            _as_dict(overall.get("selector_score")).get("spearman")
        ),
        "segment_score_spearman": _as_float(_as_dict(overall.get("segment_score")).get("spearman")),
        "raw_score_spearman": _as_float(_as_dict(overall.get("raw_score")).get("spearman")),
        "teacher_usage_split": separated.get("teacher_usage_split"),
        "candidate_for_train_side_teacher": _as_bool(
            separated.get("candidate_for_train_side_teacher")
        ),
        "candidate_for_train_side_teacher_reason": separated.get(
            "candidate_for_train_side_teacher_reason"
        ),
        "teacher_target_shape_viable": _as_bool(separated.get("teacher_target_shape_viable")),
        "segment_target_count": separated.get("segment_target_count"),
        "point_target_count": separated.get("point_target_count"),
        "top_segment_row_count": len(top_rows),
        "top_segment_low_selector_score_count": sum(
            1 for value in segment_rank_fractions if value >= LOW_RANK_FRACTION
        ),
        "top_segment_low_allocation_weight_count": sum(
            1 for value in allocation_rank_fractions if value >= LOW_RANK_FRACTION
        ),
        "top_segment_mean_selector_score_rank_fraction": _mean(segment_rank_fractions),
        "top_segment_mean_allocation_weight_rank_fraction": _mean(allocation_rank_fractions),
        "top_segment_rows": [_segment_row_ref(row, segment_count_int) for row in top_rows],
        "point_target_top_rows": [
            {
                "point_index": row.get("point_index"),
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
                "selector_segment_allocation_count": row.get("selector_segment_allocation_count"),
            }
            for row in sorted(
                point_rows,
                key=lambda row: -float(_as_float(row.get("raw_point_marginal")) or 0.0),
            )[:TOP_SEGMENT_LIMIT]
        ],
    }


def _segment_indices(rows: list[dict[str, Any]]) -> set[int]:
    out: set[int] = set()
    for row in rows:
        value = row.get("segment_index")
        if isinstance(value, int):
            out.add(value)
    return out


def _split_overlap(selection: dict[str, Any], eval_trace: dict[str, Any]) -> dict[str, Any]:
    selection_rows = _segment_rows(_alignment(selection))
    eval_rows = _segment_rows(_alignment(eval_trace))
    selection_segments = _segment_indices(selection_rows)
    eval_segments = _segment_indices(eval_rows)
    top_selection_segments = _segment_indices(_top_segment_rows(selection_rows))
    top_eval_segments = _segment_indices(_top_segment_rows(eval_rows))
    overlap = selection_segments & eval_segments
    top_overlap = top_selection_segments & top_eval_segments
    return {
        "selection_segment_target_count": len(selection_segments),
        "eval_segment_target_count": len(eval_segments),
        "segment_overlap_count": len(overlap),
        "segment_overlap_fraction_of_selection": (
            None if not selection_segments else len(overlap) / len(selection_segments)
        ),
        "top_segment_overlap_count": len(top_overlap),
        "top_segment_overlap_fraction_of_selection_top": (
            None if not top_selection_segments else len(top_overlap) / len(top_selection_segments)
        ),
        "overlap_segment_indices": sorted(overlap),
        "top_overlap_segment_indices": sorted(top_overlap),
    }


def _allocation_summary(trace: dict[str, Any]) -> dict[str, Any]:
    allocation = _as_dict(trace.get("segment_allocation_alignment_diagnostics"))
    point_selection = _as_dict(trace.get("allocation_point_selection_diagnostics"))
    return {
        "segment_allocation_available": _as_bool(allocation.get("available")),
        "segment_allocation_component_diagnosis": allocation.get("component_diagnosis"),
        "length_support_to_allocation_spearman": _as_float(
            allocation.get("length_support_to_allocation_spearman")
        ),
        "segment_score_to_allocation_spearman": _as_float(
            allocation.get("segment_score_to_allocation_spearman")
        ),
        "point_selection_available": _as_bool(point_selection.get("available")),
        "point_selection_component_diagnosis": point_selection.get("component_diagnosis"),
        "primary_length_preservation": _as_float(
            point_selection.get("primary_length_preservation")
        ),
        "same_allocation_length_only_point_selection_preservation": _as_float(
            point_selection.get("same_allocation_length_only_point_selection_preservation")
        ),
        "same_allocation_length_only_gate_would_pass": _as_bool(
            point_selection.get("same_allocation_length_only_gate_would_pass")
        ),
    }


def _artifact_summary(label: str, artifact: dict[str, Any]) -> dict[str, Any]:
    selection_trace = _trace(artifact, SELECTION_TRACE_NAME)
    eval_trace = _trace(artifact, EVAL_TRACE_NAME)
    return {
        "label": label,
        "scores": _score_summary(artifact),
        "gates": _gate_summary(artifact),
        "selection_teacher": _trace_teacher_summary(SELECTION_TRACE_NAME, selection_trace),
        "eval_teacher": _trace_teacher_summary(EVAL_TRACE_NAME, eval_trace),
        "selection_eval_segment_overlap": _split_overlap(selection_trace, eval_trace),
        "selection_allocation": _allocation_summary(selection_trace),
        "eval_allocation": _allocation_summary(eval_trace),
    }


def _decision(summary: dict[str, Any]) -> str:
    selection = _as_dict(summary.get("selection_teacher"))
    overlap = _as_dict(summary.get("selection_eval_segment_overlap"))
    if selection.get("alignment_available") is not True:
        return "add_selection_side_retained_marginal_alignment_before_calibration"
    if selection.get("candidate_for_train_side_teacher") is not True:
        return "make_selection_marginal_teacher_split_eligible_before_calibration"
    low_segment_count = int(selection.get("top_segment_low_selector_score_count") or 0)
    low_allocation_count = int(selection.get("top_segment_low_allocation_weight_count") or 0)
    top_overlap_fraction = _as_float(overlap.get("top_segment_overlap_fraction_of_selection_top"))
    if low_segment_count > 0 and low_allocation_count > 0:
        if top_overlap_fraction is not None and top_overlap_fraction < 0.25:
            return "diagnose_selection_marginal_segment_transfer_before_training_semantics"
        return "build_guarded_train_side_segment_marginal_calibration_probe"
    return "diagnostic_only_no_segment_calibration_branch"


def build_selection_marginal_segment_calibration_diagnostic(
    artifacts: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Build a derived diagnostic for train/selection-side marginal segment evidence."""
    summaries = [_artifact_summary(label, artifact) for label, artifact in artifacts]
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
                f"{SELECTOR_TRACE_PATH}.{SELECTION_TRACE_NAME}.{MARGINAL_ALIGNMENT_KEY}"
            ),
            "eval_layout": f"{SELECTOR_TRACE_PATH}.{EVAL_TRACE_NAME}.{MARGINAL_ALIGNMENT_KEY}",
            "decision": primary.get("decision"),
            "decision_scope": "primary_artifact_last_input",
            "interpretation": (
                "Derived diagnosis only. Selection exact marginals are split-eligible "
                "teacher evidence, but sparse split-specific segment targets still need "
                "transfer evidence before training semantics."
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
        description="Build a selection-side marginal segment calibration diagnostic."
    )
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="Artifact path, optionally label=path. The last artifact is primary.",
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args(argv)

    artifacts = [
        (label, _load_json(path))
        for label, path in (_parse_labeled_artifact(value) for value in args.artifact)
    ]
    diagnostic = build_selection_marginal_segment_calibration_diagnostic(artifacts)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(diagnostic, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
