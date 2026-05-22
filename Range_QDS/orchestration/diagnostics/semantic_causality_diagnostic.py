"""Derived semantic-causality diagnostics for query-driven Range_QDS artifacts."""

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
    delta as _safe_delta,
)
from orchestration.diagnostics.artifact_utils import (
    load_json_dict as _load_json,
)
from orchestration.diagnostics.artifact_utils import (
    ratio as _safe_ratio,
)
from orchestration.diagnostics.semantic_causality_prior import prior_summary

PRIMARY_METHOD = "MLQDS"
BASELINE_METHOD = "DouglasPeucker"
BEHAVIOR_HEAD = "conditional_behavior_utility"
SEGMENT_HEAD = "segment_budget_target"
ROW_LIMIT = 24


def _path(root: dict[str, Any], *keys: str) -> Any:
    cursor: Any = root
    for key in keys:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor


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
        "primary_minus_uniform_query_local_utility": _safe_delta(primary_score, uniform_score),
        "primary_minus_baseline_query_local_utility": _safe_delta(primary_score, baseline_score),
    }


def _gate_summary(artifact: dict[str, Any]) -> dict[str, bool | None]:
    workload_signature = _as_dict(
        _path(artifact, "workload_distribution_comparison", "workload_signature_gate")
    )
    causality = _as_dict(artifact.get("learning_causality_summary"))
    return {
        "workload_stability": _as_bool(_path(artifact, "workload_stability_gate", "gate_pass")),
        "support_overlap": _as_bool(_path(artifact, "support_overlap_gate", "gate_pass")),
        "target_diffusion": _as_bool(_path(artifact, "target_diffusion_gate", "gate_pass")),
        "workload_signature": _as_bool(workload_signature.get("all_pass")),
        "predictability": _as_bool(_path(artifact, "predictability_audit", "gate_pass")),
        "prior_predictive_alignment": _as_bool(
            _path(
                artifact,
                "predictability_audit",
                "prior_predictive_alignment_gate",
                "gate_pass",
            )
        ),
        "learning_causality": _as_bool(causality.get("learning_causality_gate_pass")),
        "global_sanity": _as_bool(_path(artifact, "global_sanity_gate", "gate_pass")),
        "final_success_allowed": _as_bool(
            _path(artifact, "final_claim_summary", "final_success_allowed")
        ),
    }


def _target_diagnostics(artifact: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(
        _path(artifact, "training_target_diagnostics", "query_local_utility_factorized")
    )


def _fit_diagnostics(artifact: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(artifact.get("training_fit_diagnostics"))


def _eval_alignment(artifact: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(
        _path(
            artifact,
            "selector_trace_diagnostics",
            "eval_primary",
            "retained_decision_marginal_query_local_utility_alignment",
        )
    )


def _alignment_score(alignment: dict[str, Any], field: str) -> dict[str, float | int | None]:
    score = _as_dict(_as_dict(alignment.get("overall")).get(field))
    return {
        "count": _as_float(score.get("count")),
        "spearman": _as_float(score.get("spearman")),
        "pearson": _as_float(score.get("pearson")),
        "top_quartile_mean_marginal": _as_float(score.get("top_quartile_mean_marginal")),
        "bottom_quartile_mean_marginal": _as_float(score.get("bottom_quartile_mean_marginal")),
        "top_minus_bottom_marginal": _as_float(score.get("top_minus_bottom_marginal")),
        "value_min": _as_float(score.get("value_min")),
        "value_max": _as_float(score.get("value_max")),
    }


def _score_component_alignment(
    alignment: dict[str, Any], field: str
) -> dict[str, float | int | None]:
    component = _as_dict(
        _as_dict(_as_dict(alignment.get("overall")).get("score_component_alignment")).get(field)
    )
    return {
        "count": _as_float(component.get("count")),
        "spearman": _as_float(component.get("spearman")),
        "pearson": _as_float(component.get("pearson")),
        "top_minus_bottom_marginal": _as_float(component.get("top_minus_bottom_marginal")),
        "value_min": _as_float(component.get("value_min")),
        "value_max": _as_float(component.get("value_max")),
    }


def _behavior_classification(
    *,
    target_std: float | None,
    prediction_std: float | None,
    target_prediction_std_ratio: float | None,
    head_tau: float | None,
    retained_spearman: float | None,
    no_behavior_delta: float | None,
) -> dict[str, Any]:
    flat_prediction = (
        target_prediction_std_ratio is not None and target_prediction_std_ratio < 0.05
    ) or (
        target_std is not None
        and prediction_std is not None
        and target_std > 0.0
        and prediction_std < 0.01 * target_std
    )
    weak_rank = head_tau is None or abs(head_tau) < 0.05
    weak_retained = retained_spearman is None or retained_spearman <= 0.0
    weak_ablation = no_behavior_delta is None or no_behavior_delta < 0.005
    if flat_prediction and weak_rank:
        category = "target has signal but head does not learn it"
    elif not weak_rank and weak_retained:
        category = "head learns weak signal but final score suppresses it"
    elif not weak_retained and weak_ablation:
        category = "final score has signal but selector/segment allocation loses it"
    else:
        category = "artifact supports only partial behavior-failure classification"
    return {
        "protocol_category": category,
        "flat_prediction": flat_prediction,
        "weak_rank_alignment": weak_rank,
        "weak_or_wrong_way_retained_alignment": weak_retained,
        "below_material_ablation_delta": weak_ablation,
    }


def _behavior_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    target = _target_diagnostics(artifact)
    fit = _fit_diagnostics(artifact)
    fit_head = _as_dict(_as_dict(fit.get("factorized_head_fit")).get(BEHAVIOR_HEAD))
    target_alignment = _as_dict(target.get("conditional_behavior_target_alignment"))
    candidate_alignment = _as_dict(target.get("conditional_behavior_candidate_alignment"))
    target_trainability = _as_dict(target.get("family_conditioned_target_trainability"))
    alignment = _eval_alignment(artifact)
    behavior_component = _score_component_alignment(
        alignment, "head_probability_conditional_behavior_utility"
    )
    behavior_branch = _score_component_alignment(alignment, "factorized_behavior_branch")
    if not behavior_branch:
        behavior_branch = _score_component_alignment(alignment, "factorized_behavior_multiplier")
    causality = _as_dict(artifact.get("learning_causality_summary"))
    no_behavior_delta = _as_float(causality.get("no_behavior_head_ablation_delta"))
    target_std = _as_float(fit_head.get("target_std"))
    prediction_std = _as_float(fit_head.get("prediction_std"))
    ratio = _safe_ratio(prediction_std, target_std)
    classification = _behavior_classification(
        target_std=target_std,
        prediction_std=prediction_std,
        target_prediction_std_ratio=ratio,
        head_tau=_as_float(fit_head.get("kendall_tau")),
        retained_spearman=_as_float(behavior_component.get("spearman")),
        no_behavior_delta=no_behavior_delta,
    )
    return {
        "classification": classification,
        "target_location": {
            "training_mask": target.get("conditional_behavior_utility_training"),
            "variant": target.get("conditional_behavior_target_variant"),
            "positive_point_count": _path(target, "positive_point_count_by_head", BEHAVIOR_HEAD),
            "positive_fraction": _path(target, "positive_fraction_by_head", BEHAVIOR_HEAD),
            "positive_label_mass": _path(target, "positive_label_mass_by_head", BEHAVIOR_HEAD),
            "support_fraction_by_threshold": _path(
                target, "support_fraction_by_threshold_by_head", BEHAVIOR_HEAD
            ),
            "label_mass_by_segment_position": _path(
                target, "label_mass_by_segment_position", BEHAVIOR_HEAD
            ),
        },
        "target_alignment": {
            "spearman_with_final_score": _as_float(
                target_alignment.get("spearman_with_final_score")
            ),
            "topk_overlap_with_final_score": _as_float(
                target_alignment.get("topk_overlap_with_final_score")
            ),
            "spearman_with_query_hit_probability": _as_float(
                target_alignment.get("spearman_with_query_hit_probability")
            ),
            "spearman_with_replacement_representative_value": _as_float(
                target_alignment.get("spearman_with_replacement_representative_value")
            ),
            "spearman_with_segment_budget_target": _as_float(
                target_alignment.get("spearman_with_segment_budget_target")
            ),
            "spearman_with_path_length_support_target": _as_float(
                target_alignment.get("spearman_with_path_length_support_target")
            ),
        },
        "candidate_alignment_keys": sorted(candidate_alignment.keys()),
        "head_fit": {
            "valid_point_count": fit_head.get("valid_point_count"),
            "positive_target_count": fit_head.get("positive_target_count"),
            "positive_target_fraction": _as_float(fit_head.get("positive_target_fraction")),
            "target_mean": _as_float(fit_head.get("target_mean")),
            "target_std": target_std,
            "prediction_mean": _as_float(fit_head.get("prediction_mean")),
            "prediction_std": prediction_std,
            "prediction_std_to_target_std": ratio,
            "kendall_tau": _as_float(fit_head.get("kendall_tau")),
            "topk_mass_recall_at_5_percent": _as_float(
                fit_head.get("topk_mass_recall_at_5_percent")
            ),
        },
        "absorbed_or_redundant_with_other_heads": {
            "query_hit_probability_head_tau": _as_float(fit.get("query_hit_probability_head_tau")),
            "replacement_representative_value_head_tau": _as_float(
                fit.get("replacement_representative_value_head_tau")
            ),
            "segment_budget_target_head_tau": _as_float(fit.get("segment_budget_target_head_tau")),
            "factorized_final_score_tau": _as_float(fit.get("factorized_final_score_tau")),
            "behavior_target_vs_replacement_spearman": _as_float(
                target_alignment.get("spearman_with_replacement_representative_value")
            ),
        },
        "retained_marginal_alignment": {
            "behavior_head_probability": behavior_component,
            "behavior_branch": behavior_branch,
            "query_hit_head_probability": _score_component_alignment(
                alignment, "head_probability_query_hit_probability"
            ),
            "replacement_head_probability": _score_component_alignment(
                alignment, "head_probability_replacement_representative_value"
            ),
            "segment_budget_head_probability": _score_component_alignment(
                alignment, "head_probability_segment_budget_target"
            ),
        },
        "ablation_materiality": {
            "no_behavior_head_delta": no_behavior_delta,
            "threshold": _path(
                causality,
                "learning_causality_delta_gate",
                "thresholds",
                "without_behavior_utility_head_should_lose",
            ),
            "mask_diagnostics": _path(
                causality,
                "causality_ablation_mask_diagnostics",
                "MLQDS_without_behavior_utility_head",
            ),
        },
        "family_conditioned_target_trainability": target_trainability,
    }


def _ablation_scores(artifact: dict[str, Any]) -> dict[str, float | None]:
    causality = _as_dict(artifact.get("learning_causality_summary"))
    scores = _as_dict(causality.get("causality_ablation_scores"))
    keys = (
        "MLQDS_without_segment_budget_head",
        "MLQDS_without_segment_budget_allocation_only",
        "MLQDS_uniform_segment_allocation_only_diagnostic",
        "MLQDS_point_score_allocation_diagnostic",
        "MLQDS_segment_allocation_top25_band_diagnostic",
        "MLQDS_segment_allocation_top50_band_diagnostic",
        "MLQDS_segment_allocation_quartile_band_diagnostic",
        "MLQDS_without_segment_budget_point_blend_only",
        "MLQDS_without_segment_length_support_allocation",
        "MLQDS_path_length_support_allocation_only_diagnostic",
        "MLQDS_behavior_utility_allocation_only_diagnostic",
    )
    return {key: _as_float(scores.get(key)) for key in keys}


def _segment_classification(
    *,
    raw_spearman: float | None,
    selector_spearman: float | None,
    segment_spearman: float | None,
    point_score_delta: float | None,
    without_segment_budget_delta: float | None,
    segment_target_oracle_spearman: float | None,
) -> dict[str, Any]:
    raw_positive = raw_spearman is not None and raw_spearman > 0.0
    selector_positive = selector_spearman is not None and selector_spearman > 0.0
    segment_negative = segment_spearman is not None and segment_spearman < 0.0
    point_proxy_better = point_score_delta is not None and point_score_delta > 0.0
    segment_budget_material = (
        without_segment_budget_delta is not None and without_segment_budget_delta > 0.005
    )
    target_not_obviously_bad = (
        segment_target_oracle_spearman is not None and segment_target_oracle_spearman > 0.5
    )
    if raw_positive and selector_positive and segment_negative and point_proxy_better:
        category = "allocation scoring and point-selection scoring are mixed incorrectly"
    elif segment_negative and target_not_obviously_bad:
        category = "segment head fails to learn target"
    elif segment_negative:
        category = "segment target is misaligned with exact marginal utility"
    else:
        category = "artifact supports only partial segment-failure classification"
    return {
        "protocol_category": category,
        "raw_and_selector_positive": raw_positive and selector_positive,
        "segment_score_wrong_way": segment_negative,
        "pooled_point_score_allocation_better_than_primary": point_proxy_better,
        "segment_budget_head_material": segment_budget_material,
        "segment_target_oracle_alignment_positive": target_not_obviously_bad,
    }


def _segment_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    alignment = _eval_alignment(artifact)
    scores = _score_summary(artifact)
    primary_score = scores["primary_query_local_utility"]
    ablations = _ablation_scores(artifact)
    point_score = ablations["MLQDS_point_score_allocation_diagnostic"]
    no_segment_budget = ablations["MLQDS_without_segment_budget_head"]
    segment_target_oracle = _as_float(
        _path(
            artifact,
            "target_segment_oracle_alignment_audit",
            "source_alignment",
            "target_head_segment_budget_target_top20_mean",
            "spearman_vs_oracle_mass",
        )
    )
    raw = _alignment_score(alignment, "raw_score")
    selector = _alignment_score(alignment, "selector_score")
    segment = _alignment_score(alignment, "segment_score")
    return {
        "classification": _segment_classification(
            raw_spearman=_as_float(raw.get("spearman")),
            selector_spearman=_as_float(selector.get("spearman")),
            segment_spearman=_as_float(segment.get("spearman")),
            point_score_delta=_safe_delta(point_score, primary_score),
            without_segment_budget_delta=_safe_delta(primary_score, no_segment_budget),
            segment_target_oracle_spearman=segment_target_oracle,
        ),
        "retained_marginal_alignment": {
            "raw_score": raw,
            "selector_score": selector,
            "segment_score": segment,
            "segment_budget_head_probability": _score_component_alignment(
                alignment, "head_probability_segment_budget_target"
            ),
            "path_length_support_head_probability": _score_component_alignment(
                alignment, "head_probability_path_length_support_target"
            ),
        },
        "allocation_diagnostics": {
            "segment_score_to_allocation_spearman": _as_float(
                _path(
                    artifact,
                    "selector_trace_diagnostics",
                    "eval_primary",
                    "segment_allocation_alignment_diagnostics",
                    "segment_score_to_allocation_spearman",
                )
            ),
            "allocation_weight_to_allocation_spearman": _as_float(
                _path(
                    artifact,
                    "selector_trace_diagnostics",
                    "eval_primary",
                    "segment_allocation_alignment_diagnostics",
                    "allocation_weight_to_allocation_spearman",
                )
            ),
            "length_support_to_allocation_spearman": _as_float(
                _path(
                    artifact,
                    "selector_trace_diagnostics",
                    "eval_primary",
                    "segment_allocation_alignment_diagnostics",
                    "length_support_to_allocation_spearman",
                )
            ),
            "component_diagnosis": _path(
                artifact,
                "selector_trace_diagnostics",
                "eval_primary",
                "segment_allocation_alignment_diagnostics",
                "component_diagnosis",
            ),
        },
        "diagnostic_segment_rankers": {
            "primary_query_local_utility": primary_score,
            "neutral_segment_score_query_local_utility": no_segment_budget,
            "pooled_point_score_allocation_query_local_utility": point_score,
            "pooled_point_score_minus_primary": _safe_delta(point_score, primary_score),
            "without_segment_budget_minus_primary": _safe_delta(no_segment_budget, primary_score),
            "without_segment_length_support_query_local_utility": ablations[
                "MLQDS_without_segment_length_support_allocation"
            ],
            "path_length_support_allocation_query_local_utility": ablations[
                "MLQDS_path_length_support_allocation_only_diagnostic"
            ],
            "behavior_utility_allocation_query_local_utility": ablations[
                "MLQDS_behavior_utility_allocation_only_diagnostic"
            ],
            "segment_allocation_quartile_band_query_local_utility": ablations[
                "MLQDS_segment_allocation_quartile_band_diagnostic"
            ],
        },
        "segment_oracle_alignment": {
            "point_score_top20_mean_spearman_vs_oracle_mass": _as_float(
                _path(
                    artifact,
                    "target_segment_oracle_alignment_audit",
                    "source_alignment",
                    "point_score_top20_mean",
                    "spearman_vs_oracle_mass",
                )
            ),
            "target_head_query_hit_probability_top20_mean_spearman_vs_oracle_mass": (
                _as_float(
                    _path(
                        artifact,
                        "target_segment_oracle_alignment_audit",
                        "source_alignment",
                        "target_head_query_hit_probability_top20_mean",
                        "spearman_vs_oracle_mass",
                    )
                )
            ),
            "target_head_segment_budget_target_top20_mean_spearman_vs_oracle_mass": (
                segment_target_oracle
            ),
            "target_head_path_length_support_target_top20_mean_spearman_vs_oracle_mass": (
                _as_float(
                    _path(
                        artifact,
                        "target_segment_oracle_alignment_audit",
                        "source_alignment",
                        "target_head_path_length_support_target_top20_mean",
                        "spearman_vs_oracle_mass",
                    )
                )
            ),
        },
        "train_side_marginal_teacher": {
            "available": _as_bool(
                _path(
                    artifact,
                    "train_marginal_causality_diagnostics",
                    "selection_retained_decision_marginal_teacher",
                    "available",
                )
            ),
            "teacher_value_variation": _as_float(
                _path(
                    artifact,
                    "train_marginal_causality_diagnostics",
                    "selection_retained_decision_marginal_teacher",
                    "teacher_value_variation",
                )
            ),
            "separated_teacher_available": _as_bool(
                _path(
                    artifact,
                    "train_marginal_causality_diagnostics",
                    "selection_retained_decision_marginal_teacher",
                    "separated_marginal_teacher_summary",
                    "available",
                )
            ),
            "separated_teacher_candidate_for_train_side": _as_bool(
                _path(
                    artifact,
                    "train_marginal_causality_diagnostics",
                    "selection_retained_decision_marginal_teacher",
                    "separated_marginal_teacher_summary",
                    "candidate_for_train_side_teacher",
                )
            ),
        },
    }


def _row_ref(row: dict[str, Any]) -> dict[str, Any]:
    stage = _as_dict(row.get("selector_stage_state"))
    context = _as_dict(row.get("selector_segment_context"))
    head_probs = _as_dict(row.get("head_probabilities"))
    head_targets = _as_dict(row.get("head_targets"))
    component_delta = _as_dict(row.get("query_local_utility_component_delta"))
    query_family = _as_dict(row.get("query_family_hit_context"))
    run_ids = [str(item) for item in _as_list(row.get("query_hit_run_ids"))]
    behavior_component = {
        "query_local_interpolation_fidelity": _as_float(
            component_delta.get("query_local_interpolation_fidelity")
        ),
        "query_local_turn_change_coverage": _as_float(
            component_delta.get("query_local_turn_change_coverage")
        ),
        "query_local_continuity": _as_float(component_delta.get("query_local_continuity")),
    }
    return {
        "point_index": row.get("point_index"),
        "trajectory_id": row.get("trajectory_index"),
        "source_stage": row.get("source"),
        "retained_source": row.get("source"),
        "retained_decision_type": row.get("decision"),
        "exact_marginal_query_local_utility": _as_float(row.get("marginal_query_local_utility")),
        "query_point_recall_component": _as_float(component_delta.get("query_point_recall")),
        "query_local_behavior_component": behavior_component,
        "query_local_continuity_component": _as_float(
            component_delta.get("query_local_continuity")
        ),
        "query_hit_target": _as_float(head_targets.get("query_hit_probability")),
        "query_hit_head_probability": _as_float(head_probs.get("query_hit_probability")),
        "behavior_target": _as_float(head_targets.get(BEHAVIOR_HEAD)),
        "behavior_head_probability": _as_float(head_probs.get(BEHAVIOR_HEAD)),
        "replacement_target": _as_float(head_targets.get("replacement_representative_value")),
        "replacement_head_probability": _as_float(
            head_probs.get("replacement_representative_value")
        ),
        "segment_budget_target": _as_float(head_targets.get(SEGMENT_HEAD)),
        "segment_budget_head_probability": _as_float(head_probs.get(SEGMENT_HEAD)),
        "raw_score": _as_float(row.get("raw_score")),
        "selector_score": _as_float(row.get("selector_score")),
        "segment_score": _as_float(row.get("segment_score")),
        "retained_mask_membership": _as_bool(stage.get("final_retained")),
        "learned_retained": _as_bool(stage.get("learned_retained")),
        "skeleton_retained": _as_bool(stage.get("skeleton_retained")),
        "fallback_retained": _as_bool(stage.get("fallback_retained")),
        "length_repair_retained": _as_bool(stage.get("length_repair_retained")),
        "anchor_family": row.get("anchor_family") or query_family.get("anchor_family"),
        "footprint_family": row.get("footprint_family") or query_family.get("footprint_family"),
        "query_hit_run_id": run_ids[0] if run_ids else None,
        "query_hit_run_ids": run_ids,
        "segment_id": context.get("segment_index"),
        "segment_start": context.get("segment_start"),
        "segment_end": context.get("segment_end"),
        "segment_allocation_count": context.get("segment_allocation_count"),
        "segment_learned_count": context.get("learned_count"),
        "raw_score_rank_fraction": _as_float(row.get("raw_score_candidate_rank_fraction")),
        "selector_score_rank_fraction": _as_float(
            row.get("selector_score_candidate_rank_fraction")
        ),
        "segment_score_rank_fraction": _as_float(row.get("segment_score_candidate_rank_fraction")),
        "marginal_rank_fraction": _as_float(
            row.get("marginal_query_local_utility_candidate_rank_fraction")
        ),
        "failure_buckets": row.get("failure_buckets"),
    }


def _missing_row_fields(rows: list[dict[str, Any]], context_fields: dict[str, Any]) -> list[str]:
    head_targets_available = _as_dict(context_fields.get("head_targets"))
    component_delta_available = context_fields.get("query_local_utility_component_delta") is True
    family_context_available = context_fields.get("query_family_hit_context") is True
    hit_run_available = context_fields.get("query_hit_run_ids") is True
    if not rows:
        return [
            "row_level_query_local_utility_components",
            "row_level_query_hit_target",
            "row_level_behavior_target",
            "row_level_replacement_target",
            "row_level_segment_budget_target",
            "row_level_anchor_family",
            "row_level_footprint_family",
            "query_hit_run_id",
        ]
    first = rows[0]
    head_targets = _as_dict(first.get("head_targets"))
    component_delta = _as_dict(first.get("query_local_utility_component_delta"))
    family = _as_dict(first.get("query_family_hit_context"))
    missing: list[str] = []
    if not component_delta and not component_delta_available:
        missing.append("row_level_query_local_utility_components")
    if (
        "query_hit_probability" not in head_targets
        and head_targets_available.get("query_hit_probability") is not True
    ):
        missing.append("row_level_query_hit_target")
    if BEHAVIOR_HEAD not in head_targets and head_targets_available.get(BEHAVIOR_HEAD) is not True:
        missing.append("row_level_behavior_target")
    if (
        "replacement_representative_value" not in head_targets
        and head_targets_available.get("replacement_representative_value") is not True
    ):
        missing.append("row_level_replacement_target")
    if SEGMENT_HEAD not in head_targets and head_targets_available.get(SEGMENT_HEAD) is not True:
        missing.append("row_level_segment_budget_target")
    if (
        first.get("anchor_family") is None
        and family.get("anchor_family") is None
        and not family_context_available
    ):
        missing.append("row_level_anchor_family")
    if (
        first.get("footprint_family") is None
        and family.get("footprint_family") is None
        and not family_context_available
    ):
        missing.append("row_level_footprint_family")
    if not _as_list(first.get("query_hit_run_ids")) and not hit_run_available:
        missing.append("query_hit_run_id")
    return missing


def _rows_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    alignment = _eval_alignment(artifact)
    rows = [row for row in _as_list(alignment.get("rows")) if isinstance(row, dict)]
    rows_sorted = sorted(
        rows,
        key=lambda row: (
            _as_float(row.get("marginal_query_local_utility_candidate_rank_fraction"))
            if _as_float(row.get("marginal_query_local_utility_candidate_rank_fraction"))
            is not None
            else 1.0,
            int(row.get("point_index") or 0),
        ),
    )
    missing_fields = _missing_row_fields(
        rows_sorted,
        _as_dict(alignment.get("context_fields_available")),
    )
    return {
        "available": bool(rows),
        "source_path": (
            "selector_trace_diagnostics.eval_primary."
            "retained_decision_marginal_query_local_utility_alignment.rows"
        ),
        "row_count": len(rows),
        "emitted_row_count": min(len(rows_sorted), ROW_LIMIT),
        "missing_required_fields": missing_fields,
        "rows": [_row_ref(row) for row in rows_sorted[:ROW_LIMIT]],
        "by_source": alignment.get("by_source"),
        "by_decision": alignment.get("by_decision"),
        "top_marginal_miss_summary": alignment.get("top_marginal_miss_summary"),
    }


def _artifact_field_gaps(artifact: dict[str, Any]) -> dict[str, Any]:
    rows = _rows_payload(artifact)
    missing = _as_list(rows.get("missing_required_fields"))
    return {
        "row_level_semantic_fields_complete": not missing,
        "missing_required_row_fields": missing,
        "instrumentation_needed_for_full_minimum_rows": bool(missing),
        "instrumentation_scope": (
            "selector-trace rows need per-point target values, direct QueryLocalUtility "
            "component values, query family metadata, and query-hit-run/segment grouping."
        )
        if missing
        else None,
    }


def _artifact_summary(label: str, artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "scores": _score_summary(artifact),
        "gates": _gate_summary(artifact),
        "failed_learning_causality_checks": _path(
            artifact, "learning_causality_summary", "learning_causality_failed_checks"
        ),
        "behavior_head_semantics": _behavior_summary(artifact),
        "query_prior_materiality": prior_summary(artifact),
        "segment_score_calibration": _segment_summary(artifact),
        "representative_rows": _rows_payload(artifact),
        "artifact_field_gaps": _artifact_field_gaps(artifact),
    }


def _decision(summary: dict[str, Any]) -> dict[str, Any]:
    behavior = _as_dict(summary.get("behavior_head_semantics"))
    prior = _as_dict(summary.get("query_prior_materiality"))
    segment = _as_dict(summary.get("segment_score_calibration"))
    gaps = _as_dict(summary.get("artifact_field_gaps"))
    return {
        "behavior_failure": _path(behavior, "classification", "protocol_category"),
        "prior_failure": _path(prior, "classification", "protocol_category"),
        "segment_failure": _path(segment, "classification", "protocol_category"),
        "artifact_lacks_full_required_rows": _as_bool(
            gaps.get("instrumentation_needed_for_full_minimum_rows")
        ),
        "next_admissible_step": (
            "Add focused selector/target trace instrumentation for row-level semantic "
            "fields before a root fix or replay; do not run final grid."
        )
        if gaps.get("instrumentation_needed_for_full_minimum_rows") is True
        else (
            "Design a root fix for the classified blocker, restart at Level 1/2; "
            "do not run final grid."
        ),
    }


def build_semantic_causality_diagnostic(
    artifacts: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Build a derived diagnostic for semantic learning-causality failures."""
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
            "decision": _decision(primary),
            "interpretation": (
                "Derived diagnosis only. It localizes learning-causality failures "
                "from frozen-mask artifacts and does not change acceptance state."
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
        description="Build a semantic-causality diagnostic for query-driven artifacts."
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
    diagnostic = build_semantic_causality_diagnostic(artifacts)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(diagnostic, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
