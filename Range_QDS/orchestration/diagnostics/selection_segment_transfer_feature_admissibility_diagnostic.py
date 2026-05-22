"""Derived admissibility diagnostics for segment transfer-calibration features."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from orchestration.diagnostics.artifact_utils import load_json_dict as _load_json
from orchestration.diagnostics.selection_eval_segment_teacher_transfer_diagnostic import (
    EVAL_TRACE_NAME,
    SELECTION_TRACE_NAME,
    as_bool,
    as_dict,
    as_float,
    feature_topk_target_lift,
    gate_summary,
    mean,
    score_summary,
    segment_rows,
    spearman,
    target_vector,
    teacher_segment_targets,
    trace,
)

PRE_SELECTION_FEATURES = frozenset(
    {
        "segment_score",
        "segment_allocation_weight",
        "segment_length_support_score",
    }
)
POST_SELECTION_FEATURES = frozenset(
    {
        "segment_allocation_count",
        "learned_count",
        "length_repair_count",
        "retained_count",
        "retained_fraction",
    }
)
GUARD_FEATURES = frozenset({"segment_length_support_score"})
TOP_FRACTIONS = (0.01, 0.05, 0.10)
MIN_TRANSFER_SPEARMAN = 0.05
MIN_TOP5_LIFT = 1.0


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)))


def _zscore(values: list[float]) -> list[float]:
    mean = sum(values) / len(values) if values else 0.0
    std = _std(values) or 0.0
    if std == 0.0:
        return [0.0 for _value in values]
    return [(value - mean) / std for value in values]


def _feature_values(rows: list[dict[str, Any]], feature: str) -> list[float] | None:
    out: list[float] = []
    for row in rows:
        value = as_float(row.get(feature))
        if value is None:
            return None
        out.append(value)
    return out


def _values_by_segment(rows: list[dict[str, Any]], values: list[float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for row, value in zip(rows, values, strict=True):
        segment_index = row.get("segment_index")
        if isinstance(segment_index, int):
            out[segment_index] = value
    return out


def _lift_for_candidate(
    *,
    rows: list[dict[str, Any]],
    targets: dict[int, float],
    values: list[float],
    fraction: float,
) -> dict[str, float | int | None]:
    synthetic_feature = "__candidate_score__"
    rows_with_score: list[dict[str, Any]] = []
    for row, value in zip(rows, values, strict=True):
        copied = dict(row)
        copied[synthetic_feature] = value
        rows_with_score.append(copied)
    return feature_topk_target_lift(
        rows=rows_with_score,
        targets=targets,
        feature=synthetic_feature,
        fraction=fraction,
    )


def _single_feature(feature: str) -> Callable[[list[dict[str, Any]]], list[float] | None]:
    def _score(rows: list[dict[str, Any]]) -> list[float] | None:
        return _feature_values(rows, feature)

    return _score


def _zblend(
    weighted_features: tuple[tuple[str, float], ...],
) -> Callable[[list[dict[str, Any]]], list[float] | None]:
    def _score(rows: list[dict[str, Any]]) -> list[float] | None:
        columns: list[tuple[list[float], float]] = []
        for feature, weight in weighted_features:
            values = _feature_values(rows, feature)
            if values is None:
                return None
            columns.append((_zscore(values), weight))
        return [
            float(sum(values[index] * weight for values, weight in columns))
            for index in range(len(rows))
        ]

    return _score


TRANSFER_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "name": "segment_score",
        "description": "Active pre-selection learned segment score.",
        "features": ("segment_score",),
        "score_fn": _single_feature("segment_score"),
    },
    {
        "name": "segment_allocation_weight",
        "description": "Pre-allocation segment weight after score/length-support mixing.",
        "features": ("segment_allocation_weight",),
        "score_fn": _single_feature("segment_allocation_weight"),
    },
    {
        "name": "segment_score_allocation_weight_zblend",
        "description": "Diagnostic z-blend of active segment score and allocation weight.",
        "features": ("segment_score", "segment_allocation_weight"),
        "score_fn": _zblend(
            (
                ("segment_score", 0.5),
                ("segment_allocation_weight", 0.5),
            )
        ),
    },
    {
        "name": "segment_score_length_support_counter_blend",
        "description": "Diagnostic z-blend that subtracts length support; guard-risk only.",
        "features": ("segment_score", "segment_length_support_score"),
        "score_fn": _zblend(
            (
                ("segment_score", 1.0),
                ("segment_length_support_score", -0.25),
            )
        ),
        "guard_counter_signal": True,
    },
    {
        "name": "learned_count_post_selection_coupled",
        "description": "Post-selection learned retained count; diagnostic leak check.",
        "features": ("learned_count",),
        "score_fn": _single_feature("learned_count"),
    },
)


def _candidate_feature_classification(features: tuple[str, ...]) -> dict[str, Any]:
    post = sorted(feature for feature in features if feature in POST_SELECTION_FEATURES)
    guard = sorted(feature for feature in features if feature in GUARD_FEATURES)
    unknown = sorted(
        feature
        for feature in features
        if feature not in PRE_SELECTION_FEATURES and feature not in POST_SELECTION_FEATURES
    )
    return {
        "features": list(features),
        "post_selection_coupled_features": post,
        "guard_features": guard,
        "unknown_features": unknown,
        "uses_only_pre_selection_features": not post and not unknown,
        "uses_post_selection_coupling": bool(post),
    }


def _candidate_split_metrics(
    *,
    rows: list[dict[str, Any]],
    targets: dict[int, float],
    values: list[float],
) -> dict[str, Any]:
    target_values = target_vector(rows, targets)
    return {
        "spearman_with_segment_teacher_target": spearman(values, target_values),
        "score_mean": mean(values),
        "score_std": _std(values),
        "topk_target_lift": {
            str(fraction): _lift_for_candidate(
                rows=rows,
                targets=targets,
                values=values,
                fraction=fraction,
            )
            for fraction in TOP_FRACTIONS
        },
    }


def _candidate_summary(
    candidate: dict[str, Any],
    selection_rows: list[dict[str, Any]],
    selection_targets: dict[int, float],
    eval_rows: list[dict[str, Any]],
    eval_targets: dict[int, float],
) -> dict[str, Any]:
    raw_score_fn = candidate["score_fn"]
    if not callable(raw_score_fn):
        raise TypeError("candidate score_fn must be callable")
    score_fn = cast(Callable[[list[dict[str, Any]]], list[float] | None], raw_score_fn)
    selection_values = score_fn(selection_rows)
    eval_values = score_fn(eval_rows)
    features = tuple(str(feature) for feature in candidate.get("features", ()))
    classification = _candidate_feature_classification(features)
    guard_counter_signal = bool(candidate.get("guard_counter_signal", False))
    if selection_values is None or eval_values is None:
        return {
            "name": candidate["name"],
            "description": candidate["description"],
            "available": False,
            "classification": classification,
            "probe_admissible": False,
            "rejection_reason": "missing_candidate_feature",
        }
    selection_metrics = _candidate_split_metrics(
        rows=selection_rows,
        targets=selection_targets,
        values=selection_values,
    )
    eval_metrics = _candidate_split_metrics(
        rows=eval_rows,
        targets=eval_targets,
        values=eval_values,
    )
    selection_spearman = as_float(selection_metrics.get("spearman_with_segment_teacher_target"))
    eval_spearman = as_float(eval_metrics.get("spearman_with_segment_teacher_target"))
    selection_top5 = as_float(
        as_dict(as_dict(selection_metrics.get("topk_target_lift")).get("0.05")).get(
            "top_target_lift"
        )
    )
    eval_top5 = as_float(
        as_dict(as_dict(eval_metrics.get("topk_target_lift")).get("0.05")).get("top_target_lift")
    )
    transfer_consistent = (
        selection_spearman is not None
        and eval_spearman is not None
        and selection_spearman >= MIN_TRANSFER_SPEARMAN
        and eval_spearman >= MIN_TRANSFER_SPEARMAN
        and selection_top5 is not None
        and eval_top5 is not None
        and selection_top5 > MIN_TOP5_LIFT
        and eval_top5 > MIN_TOP5_LIFT
    )
    rejection_reason = None
    if not classification["uses_only_pre_selection_features"]:
        rejection_reason = "uses_post_selection_or_unknown_features"
    elif guard_counter_signal:
        rejection_reason = "uses_guard_counter_signal"
    elif not transfer_consistent:
        rejection_reason = "weak_selection_eval_transfer_alignment"
    return {
        "name": candidate["name"],
        "description": candidate["description"],
        "available": True,
        "classification": classification,
        "guard_counter_signal": guard_counter_signal,
        "selection_metrics": selection_metrics,
        "eval_metrics": eval_metrics,
        "transfer_consistent": transfer_consistent,
        "probe_admissible": rejection_reason is None,
        "rejection_reason": rejection_reason,
    }


def _feature_coupling_summary(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    post_positive = [
        row
        for row in candidate_rows
        if as_bool(row.get("available")) is True
        and as_bool(as_dict(row.get("classification")).get("uses_post_selection_coupling")) is True
        and as_float(
            as_dict(row.get("selection_metrics")).get("spearman_with_segment_teacher_target")
        )
        is not None
        and (
            as_float(
                as_dict(row.get("selection_metrics")).get("spearman_with_segment_teacher_target")
            )
            or 0.0
        )
        > 0.0
        and (
            as_float(as_dict(row.get("eval_metrics")).get("spearman_with_segment_teacher_target"))
            or 0.0
        )
        > 0.0
    ]
    admissible = [row for row in candidate_rows if row.get("probe_admissible") is True]
    return {
        "post_selection_positive_candidate_names": [str(row.get("name")) for row in post_positive],
        "admissible_candidate_names": [str(row.get("name")) for row in admissible],
        "post_selection_positive_candidate_count": len(post_positive),
        "admissible_candidate_count": len(admissible),
    }


def _artifact_summary(label: str, artifact: dict[str, Any]) -> dict[str, Any]:
    selection_trace = trace(artifact, SELECTION_TRACE_NAME)
    eval_trace = trace(artifact, EVAL_TRACE_NAME)
    selection_rows = segment_rows(selection_trace)
    eval_rows = segment_rows(eval_trace)
    selection_targets = teacher_segment_targets(selection_trace)
    eval_targets = teacher_segment_targets(eval_trace)
    candidate_rows = [
        _candidate_summary(
            candidate,
            selection_rows,
            selection_targets,
            eval_rows,
            eval_targets,
        )
        for candidate in TRANSFER_CANDIDATES
    ]
    selection_teacher = as_dict(
        as_dict(
            as_dict(
                selection_trace.get("retained_decision_marginal_query_local_utility_alignment")
            ).get("separated_marginal_teacher_summary")
        )
    )
    return {
        "label": label,
        "scores": score_summary(artifact),
        "gates": gate_summary(artifact),
        "selection_teacher": {
            "available": as_bool(selection_teacher.get("available")),
            "candidate_for_train_side_teacher": as_bool(
                selection_teacher.get("candidate_for_train_side_teacher")
            ),
            "segment_target_count": selection_teacher.get("segment_target_count"),
        },
        "selection_segment_candidate_count": len(selection_rows),
        "eval_segment_candidate_count": len(eval_rows),
        "candidate_rows": candidate_rows,
        "feature_coupling_summary": _feature_coupling_summary(candidate_rows),
    }


def _decision(summary: dict[str, Any]) -> str:
    coupling = as_dict(summary.get("feature_coupling_summary"))
    admissible_count = int(coupling.get("admissible_candidate_count") or 0)
    post_positive_count = int(coupling.get("post_selection_positive_candidate_count") or 0)
    if admissible_count > 0:
        return "guarded_pre_selection_transfer_calibration_probe_admissible"
    if post_positive_count > 0:
        return "do_not_use_post_selection_coupled_transfer_signal"
    return "diagnose_richer_pre_selection_transfer_features_before_probe"


def build_selection_segment_transfer_feature_admissibility_diagnostic(
    artifacts: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Build a derived diagnostic for transfer-calibration feature admissibility."""
    summaries = [_artifact_summary(label, artifact) for label, artifact in artifacts]
    primary = summaries[-1] if summaries else {}
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "evidence_level": "derived_strict_artifact_diagnostic_no_new_probe",
        "artifact_count": len(summaries),
        "artifacts": summaries,
        "summary": {
            "primary_label": primary.get("label"),
            "decision": _decision(primary),
            "pre_selection_features": sorted(PRE_SELECTION_FEATURES),
            "post_selection_coupled_features": sorted(POST_SELECTION_FEATURES),
            "interpretation": (
                "Derived diagnosis only. It separates valid pre-selection features from "
                "post-selection attribution so a guarded calibration probe cannot be "
                "justified by circular selector-outcome signals."
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
        description="Build a segment transfer-feature admissibility diagnostic."
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
    diagnostic = build_selection_segment_transfer_feature_admissibility_diagnostic(artifacts)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(diagnostic, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
