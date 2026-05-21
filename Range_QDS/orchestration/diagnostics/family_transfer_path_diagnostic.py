"""Derived family/head transfer diagnostics for QueryLocalUtility candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PRIMARY_METHOD = "MLQDS"
BASELINE_METHOD = "DouglasPeucker"
# Historical diagnostic focus from pre-simplification artifacts; not active
# workload-profile requirements.
HISTORICAL_DIAGNOSTIC_FOCUS_FAMILIES = {
    "anchor_family": ("density",),
    "footprint_family": ("small_local", "medium_operational"),
}
FOCUS_HEADS = (
    "query_hit_probability",
    "conditional_behavior_utility",
    "segment_budget_target",
    "factorized_composed_score",
)
ACTIVE_METRIC_ALIGNMENT_FIELDS = {
    "query_hit_probability": "head_probability_query_hit_probability",
    "conditional_behavior_utility": "head_probability_conditional_behavior_utility",
    "segment_budget_target": "head_probability_segment_budget_target",
    "path_length_support_target": "head_probability_path_length_support_target",
    "replacement_representative_value": "head_probability_replacement_representative_value",
    "boundary_event_utility": "head_probability_boundary_event_utility",
    "factorized_behavior_branch": "factorized_behavior_branch",
    "factorized_replacement_multiplier": "factorized_replacement_multiplier",
    "factorized_composed_score": "factorized_composed_score",
}
CAUSALITY_COMPONENT_ABLATIONS = (
    "MLQDS_without_query_prior_features",
    "MLQDS_shuffled_prior_fields",
    "MLQDS_without_behavior_utility_head",
    "MLQDS_without_segment_budget_head",
)
BEHAVIOR_TARGET_REFERENCE_FIELDS = (
    "final_score",
    "query_hit_probability",
    "replacement_representative_value",
    "segment_budget_target",
    "path_length_support_target",
    "ship_query_evidence",
)
HEAD_TARGET_FIT_MIN_TAU = 0.20
RETAINED_MARGINAL_ALIGNMENT_PATH = (
    "selector_trace_diagnostics.eval_primary."
    "retained_decision_marginal_query_local_utility_alignment"
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return None


def _as_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) <= 1e-12:
        return None
    return float(numerator / denominator)


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
        "workload_stability": _as_bool(
            _as_dict(artifact.get("workload_stability_gate")).get("gate_pass")
        ),
        "support_overlap": _as_bool(_as_dict(artifact.get("support_overlap_gate")).get("gate_pass")),
        "target_diffusion": _as_bool(_as_dict(artifact.get("target_diffusion_gate")).get("gate_pass")),
        "workload_signature": _as_bool(
            _as_dict(
                _as_dict(artifact.get("workload_distribution_comparison")).get(
                    "workload_signature_gate"
                )
            ).get("all_pass")
        ),
        "prior_predictive_alignment": _as_bool(
            _as_dict(
                _as_dict(artifact.get("predictability_audit")).get(
                    "prior_predictive_alignment_gate"
                )
            ).get("gate_pass")
        ),
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


def _target_metadata(artifact: dict[str, Any]) -> dict[str, Any]:
    target = _as_dict(
        _as_dict(artifact.get("training_target_diagnostics")).get("query_local_utility_factorized")
    )
    return {
        "target_mode": target.get("target_mode"),
        "query_hit_target_variant": target.get("query_hit_target_variant"),
        "conditional_behavior_target_variant": target.get(
            "conditional_behavior_target_variant"
        ),
        "segment_budget_target_variant": target.get("segment_budget_target_variant"),
        "segment_budget_target_aggregation": target.get("segment_budget_target_aggregation"),
        "final_success_allowed": _as_bool(target.get("final_success_allowed")),
    }


def _predictability_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    audit = _as_dict(artifact.get("predictability_audit"))
    metrics = _as_dict(audit.get("metrics"))
    prior_gate = _as_dict(audit.get("prior_predictive_alignment_gate"))
    return {
        "gate_pass": _as_bool(audit.get("gate_pass")),
        "failed_checks": [
            str(name)
            for name, passed in _as_dict(audit.get("gate_checks")).items()
            if passed is False
        ],
        "spearman": _as_float(metrics.get("spearman")),
        "pr_auc_lift_over_base_rate": _as_float(metrics.get("pr_auc_lift_over_base_rate")),
        "lift_at_5_percent": _as_float(metrics.get("lift_at_5_percent")),
        "prior_predictive_alignment_gate_pass": _as_bool(prior_gate.get("gate_pass")),
        "prior_predictive_failed_checks": [
            str(item) for item in _as_list(prior_gate.get("failed_checks"))
        ],
        "family_conditioned_prior_predictability_available": bool(
            audit.get("family_conditioned_prior_predictability")
        ),
        "aggregate_best_prior_channel_by_head": _aggregate_prior_channel_summary(audit),
    }


def _aggregate_prior_channel_summary(audit: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_head = _as_dict(audit.get("best_prior_channel_by_head"))
    channel_by_head = _as_dict(audit.get("prior_channel_by_head_predictability"))
    for head in FOCUS_HEADS:
        if head == "factorized_composed_score":
            continue
        best = _as_dict(by_head.get(head))
        best_spearman = _as_dict(best.get("best_spearman"))
        best_lift = _as_dict(best.get("best_lift_at_5_percent"))
        best_spearman_channel = best_spearman.get("channel")
        best_lift_channel = best_lift.get("channel")
        out[head] = {
            "best_spearman_channel": best_spearman_channel,
            "best_spearman": _as_float(best_spearman.get("value")),
            "best_lift_at_5_percent_channel": best_lift_channel,
            "best_lift_at_5_percent": _as_float(best_lift.get("value")),
            "best_spearman_channel_metrics": _prior_channel_metrics(
                channel_by_head,
                head=head,
                channel=best_spearman_channel,
            ),
            "best_lift_channel_metrics": _prior_channel_metrics(
                channel_by_head,
                head=head,
                channel=best_lift_channel,
            ),
        }
    return out


def _prior_channel_metrics(
    channel_by_head: dict[str, Any],
    *,
    head: str,
    channel: Any,
) -> dict[str, Any]:
    row = _as_dict(_as_dict(channel_by_head.get(head)).get(str(channel)))
    return {
        "available": bool(row),
        "spearman": _as_float(row.get("spearman")),
        "positive_target_spearman": _as_float(row.get("positive_target_spearman")),
        "lift_at_5_percent": _as_float(row.get("lift_at_5_percent")),
        "pr_auc_lift_over_base_rate": _as_float(row.get("pr_auc_lift_over_base_rate")),
        "score_std": _as_float(row.get("score_std")),
        "target_mean": _as_float(row.get("target_mean")),
    }


def _target_family_ranker(
    artifact: dict[str, Any],
    *,
    group_key: str,
    family: str,
    head: str,
) -> dict[str, Any]:
    target = _as_dict(
        _as_dict(artifact.get("training_target_diagnostics")).get("query_local_utility_factorized")
    )
    row = _as_dict(
        _as_dict(
            _as_dict(target.get("family_conditioned_target_trainability")).get("group_by")
        )
        .get(group_key, {})
        .get(family)
    )
    rankers = _as_dict(row.get("ranker_alignment"))
    target_shapes = _as_dict(row.get("target_shapes"))
    ranker_name = "final_score" if head == "factorized_composed_score" else head
    ranker = _as_dict(rankers.get(ranker_name))
    shape = _as_dict(target_shapes.get(ranker_name))
    return {
        "available": bool(ranker),
        "spearman_with_ship_query_evidence": _as_float(
            ranker.get("spearman_with_ship_query_evidence")
        ),
        "topk_ship_query_evidence_mass_recall": _as_float(
            ranker.get("topk_ship_query_evidence_mass_recall")
        ),
        "ship_query_pair_coverage_at_topk": _as_float(
            ranker.get("ship_query_pair_coverage_at_topk")
        ),
        "target_mean": _as_float(shape.get("target_mean")),
        "target_std": _as_float(shape.get("target_std")),
    }


def _fitted_family_head(
    artifact: dict[str, Any],
    *,
    group_key: str,
    family: str,
    head: str,
) -> dict[str, Any]:
    fit = _as_dict(artifact.get("training_fit_diagnostics"))
    row = _as_dict(
        _as_dict(_as_dict(fit.get("family_conditioned_head_trainability")).get("group_by"))
        .get(group_key, {})
        .get(family)
    )
    if head == "factorized_composed_score":
        fitted = _as_dict(row.get("factorized_composed_score_fit"))
    else:
        fitted = _as_dict(_as_dict(row.get("head_fit")).get(head))
    target_std = _as_float(fitted.get("target_std"))
    prediction_std = _as_float(fitted.get("prediction_std"))
    return {
        "available": bool(fitted),
        "spearman_with_family_ship_query_evidence": _as_float(
            fitted.get("spearman_with_family_ship_query_evidence")
        ),
        "topk_family_ship_query_evidence_mass_recall": _as_float(
            fitted.get("topk_family_ship_query_evidence_mass_recall")
        ),
        "kendall_tau_with_head_target": _as_float(fitted.get("kendall_tau_with_head_target")),
        "topk_head_target_mass_recall": _as_float(fitted.get("topk_head_target_mass_recall")),
        "prediction_mean": _as_float(fitted.get("prediction_mean")),
        "prediction_std": prediction_std,
        "target_mean": _as_float(fitted.get("target_mean")),
        "target_std": target_std,
        "prediction_std_to_target_std": _ratio(prediction_std, target_std),
    }


def _family_row(artifact: dict[str, Any], *, group_key: str, family: str) -> dict[str, Any]:
    family_fit = _as_dict(
        _as_dict(
            _as_dict(artifact.get("training_fit_diagnostics")).get(
                "family_conditioned_head_trainability"
            )
        )
        .get("group_by", {})
        .get(group_key, {})
        .get(family)
    )
    heads: dict[str, Any] = {}
    statuses: list[str] = []
    for head in FOCUS_HEADS:
        target = _target_family_ranker(artifact, group_key=group_key, family=family, head=head)
        fitted = _fitted_family_head(artifact, group_key=group_key, family=family, head=head)
        status = _transfer_status(target, fitted)
        heads[head] = {
            "target": target,
            "fitted": fitted,
            "target_to_fit_ship_spearman_gap": _delta(
                fitted.get("spearman_with_family_ship_query_evidence"),
                target.get("spearman_with_ship_query_evidence"),
            ),
            "transfer_status": status,
        }
        if status != "ok_or_not_focus_blocker":
            statuses.append(status)
    return {
        "group_key": group_key,
        "family": family,
        "available": bool(family_fit),
        "query_count": family_fit.get("query_count"),
        "valid_hit_point_count": family_fit.get("valid_hit_point_count"),
        "weak_ship_evidence_heads": [
            str(item) for item in _as_list(family_fit.get("weak_ship_evidence_heads"))
        ],
        "heads": heads,
        "blocking_statuses": sorted(set(statuses)),
    }


def _transfer_status(target: dict[str, Any], fitted: dict[str, Any]) -> str:
    target_ship = _as_float(target.get("spearman_with_ship_query_evidence"))
    fitted_ship = _as_float(fitted.get("spearman_with_family_ship_query_evidence"))
    tau = _as_float(fitted.get("kendall_tau_with_head_target"))
    if target_ship is None or fitted_ship is None:
        return "unavailable"
    if target_ship >= 0.0 and fitted_ship < 0.0 and (tau or 0.0) >= HEAD_TARGET_FIT_MIN_TAU:
        return "fits_target_but_misorders_ship_evidence"
    if target_ship >= 0.0 and fitted_ship < 0.0:
        return "target_positive_fit_negative"
    if target_ship < 0.0:
        return "target_still_weak"
    return "ok_or_not_focus_blocker"


def _focus_family_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    focus_families = _focus_families(artifact)
    return [
        _family_row(artifact, group_key=group_key, family=family)
        for group_key, families in focus_families.items()
        for family in families
    ]


def _focus_families(artifact: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Return current artifact focus families, falling back for historical artifacts."""
    target = _as_dict(
        _as_dict(artifact.get("training_target_diagnostics")).get("query_local_utility_factorized")
    )
    focus = _as_dict(
        _as_dict(target.get("family_conditioned_target_trainability")).get("focus_families")
    )
    out: dict[str, tuple[str, ...]] = {}
    for group_key in ("anchor_family", "footprint_family"):
        families = [str(item) for item in _as_list(focus.get(group_key)) if str(item)]
        if families:
            out[group_key] = tuple(sorted(set(families)))
    if out:
        return out
    return dict(HISTORICAL_DIAGNOSTIC_FOCUS_FAMILIES)


def _retained_marginal_alignment(artifact: dict[str, Any]) -> dict[str, Any]:
    selector_trace = _as_dict(artifact.get("selector_trace_diagnostics"))
    alignment = _as_dict(
        _as_dict(selector_trace.get("eval_primary")).get(
            "retained_decision_marginal_query_local_utility_alignment"
        )
    )
    deprecated_learning_causality_alignment = _as_dict(
        _as_dict(
            _as_dict(artifact.get("learning_causality_summary")).get(
                "selection_causality_diagnostics"
            )
        ).get("retained_decision_marginal_query_local_utility_alignment")
    )
    overall = _as_dict(alignment.get("overall"))
    retained_removal = _as_dict(_as_dict(alignment.get("by_decision")).get("retained_removal_loss"))
    return {
        "available": _as_bool(alignment.get("available")),
        "candidate_count": alignment.get("candidate_count"),
        "source_layout": RETAINED_MARGINAL_ALIGNMENT_PATH,
        "deprecated_learning_causality_layout_present": bool(
            deprecated_learning_causality_alignment
        ),
        "overall": _score_alignment_subset(overall),
        "retained_removal_loss": _score_alignment_subset(retained_removal),
        "active_metric_score_component_alignment": _active_metric_score_component_alignment(
            alignment
        ),
        "selector_score_overall_spearman": _as_float(
            _as_dict(overall.get("selector_score")).get("spearman")
        ),
    }


def _score_alignment_subset(row: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "spearman": _as_float(_as_dict(row.get(name)).get("spearman")),
            "top_minus_bottom_marginal": _as_float(
                _as_dict(row.get(name)).get("top_minus_bottom_marginal")
            ),
        }
        for name in ("raw_score", "selector_score", "segment_score")
    }


def _active_metric_score_component_alignment(alignment: dict[str, Any]) -> dict[str, Any]:
    overall = _as_dict(_as_dict(alignment.get("overall")).get("score_component_alignment"))
    retained_removal_row = _as_dict(
        _as_dict(alignment.get("by_decision")).get("retained_removal_loss")
    )
    retained_removal = _as_dict(retained_removal_row.get("score_component_alignment"))
    return {
        "available": bool(overall or retained_removal),
        "source_layout": (
            f"{RETAINED_MARGINAL_ALIGNMENT_PATH}."
            "overall.score_component_alignment"
        ),
        "overall": _active_metric_alignment_subset(overall),
        "retained_removal_loss": _active_metric_alignment_subset(retained_removal),
        "legacy_ship_evidence_note": (
            "family rows still report ship-query evidence as a diagnostic proxy; "
            "active metric head alignment must be read here."
        ),
    }


def _active_metric_alignment_subset(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for public_name, artifact_name in ACTIVE_METRIC_ALIGNMENT_FIELDS.items():
        item = _as_dict(row.get(artifact_name))
        out[public_name] = {
            "available": bool(item),
            "artifact_field": artifact_name,
            "spearman": _as_float(item.get("spearman")),
            "top_minus_bottom_marginal": _as_float(item.get("top_minus_bottom_marginal")),
        }
    return out


def _causality_component_tradeoffs(artifact: dict[str, Any]) -> dict[str, Any]:
    rows = _as_dict(
        _as_dict(artifact.get("learning_causality_summary")).get(
            "causality_ablation_component_deltas"
        )
    )
    return {
        name: _causality_component_tradeoff_row(_as_dict(rows.get(name)))
        for name in CAUSALITY_COMPONENT_ABLATIONS
    }


def _causality_component_tradeoff_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": _as_bool(row.get("available")) if row else False,
        "delta_convention": "primary_minus_ablation",
        "query_local_utility_delta": _as_float(row.get("query_local_utility_delta")),
        "component_weighted_delta_sum": _as_float(
            row.get("component_weighted_delta_sum")
        ),
        "component_delta_residual": _as_float(row.get("component_delta_residual")),
        "top_positive_weighted_component_deltas": _component_delta_rows(
            row.get("top_positive_weighted_component_deltas")
        ),
        "top_negative_weighted_component_deltas": _component_delta_rows(
            row.get("top_negative_weighted_component_deltas")
        ),
    }


def _component_delta_rows(rows: Any) -> list[dict[str, Any]]:
    out = []
    for row in _as_list(rows):
        item = _as_dict(row)
        out.append(
            {
                "component": item.get("component"),
                "component_delta": _as_float(item.get("component_delta")),
                "weighted_delta": _as_float(item.get("weighted_delta")),
            }
        )
    return out


def _behavior_head_semantic_alignment(
    artifact: dict[str, Any],
    *,
    retained_marginal_alignment: dict[str, Any],
    component_tradeoffs: dict[str, Any],
) -> dict[str, Any]:
    target = _as_dict(
        _as_dict(artifact.get("training_target_diagnostics")).get("query_local_utility_factorized")
    )
    fit = _as_dict(artifact.get("training_fit_diagnostics"))
    head_fit = _as_dict(_as_dict(fit.get("factorized_head_fit")).get("conditional_behavior_utility"))
    active_alignment = _as_dict(
        retained_marginal_alignment.get("active_metric_score_component_alignment")
    )
    behavior_alignment = _active_metric_head_summary(
        active_alignment,
        head="conditional_behavior_utility",
    )
    no_behavior_tradeoff = _as_dict(
        component_tradeoffs.get("MLQDS_without_behavior_utility_head")
    )
    target_alignment = _behavior_target_reference_alignment(
        _as_dict(target.get("conditional_behavior_target_alignment"))
    )
    fit_std_ratio = _ratio(
        _as_float(head_fit.get("prediction_std")),
        _as_float(head_fit.get("target_std")),
    )
    row = {
        "available": bool(target_alignment or head_fit or behavior_alignment),
        "target_variant": target.get("conditional_behavior_target_variant"),
        "target_base_source": target.get("conditional_behavior_target_base_source"),
        "training_mask": target.get("conditional_behavior_utility_training"),
        "behavior_change_highpass_quantile": _as_float(
            target.get("behavior_change_highpass_quantile")
        ),
        "target_reference_alignment": target_alignment,
        "strongest_target_reference_by_spearman": _strongest_target_reference(
            target_alignment
        ),
        "head_fit": {
            "kendall_tau": _as_float(head_fit.get("kendall_tau")),
            "topk_mass_recall_at_5_percent": _as_float(
                head_fit.get("topk_mass_recall_at_5_percent")
            ),
            "prediction_std_to_target_std": fit_std_ratio,
            "prediction_std": _as_float(head_fit.get("prediction_std")),
            "target_std": _as_float(head_fit.get("target_std")),
        },
        "active_metric_head_alignment": behavior_alignment,
        "no_behavior_head_component_tradeoff": no_behavior_tradeoff,
    }
    row["semantic_statuses"] = _behavior_head_semantic_statuses(row)
    return row


def _behavior_target_reference_alignment(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for reference in BEHAVIOR_TARGET_REFERENCE_FIELDS:
        out[reference] = {
            "spearman": _as_float(row.get(f"spearman_with_{reference}")),
            "topk_overlap": _as_float(row.get(f"topk_overlap_with_{reference}")),
            "topk_mass_recall_ranked_by_behavior": _as_float(
                row.get(f"topk_{reference}_mass_recall_ranked_by_behavior")
            ),
        }
    return out


def _strongest_target_reference(target_alignment: dict[str, Any]) -> dict[str, Any]:
    best_name: str | None = None
    best_spearman: float | None = None
    for name, row in target_alignment.items():
        spearman = _as_float(_as_dict(row).get("spearman"))
        if spearman is None:
            continue
        if best_spearman is None or spearman > best_spearman:
            best_name = str(name)
            best_spearman = spearman
    return {
        "reference": best_name,
        "spearman": best_spearman,
    }


def _behavior_head_semantic_statuses(row: dict[str, Any]) -> list[str]:
    statuses: list[str] = []
    target_alignment = _as_dict(row.get("target_reference_alignment"))
    fit = _as_dict(row.get("head_fit"))
    active_alignment = _as_dict(row.get("active_metric_head_alignment"))
    no_behavior_tradeoff = _as_dict(row.get("no_behavior_head_component_tradeoff"))
    no_behavior_delta = _as_float(no_behavior_tradeoff.get("query_local_utility_delta"))
    retained_spearman = _as_float(active_alignment.get("retained_removal_spearman"))
    overall_spearman = _as_float(active_alignment.get("overall_spearman"))
    fit_tau = _as_float(fit.get("kendall_tau"))
    fit_std_ratio = _as_float(fit.get("prediction_std_to_target_std"))
    segment_spearman = _as_float(
        _as_dict(target_alignment.get("segment_budget_target")).get("spearman")
    )
    replacement_spearman = _as_float(
        _as_dict(target_alignment.get("replacement_representative_value")).get("spearman")
    )
    if no_behavior_delta is not None and no_behavior_delta < 0.0:
        statuses.append("primary_worse_than_no_behavior_head")
    if retained_spearman is not None and retained_spearman < 0.0:
        statuses.append("behavior_head_misorders_retained_marginals")
    if overall_spearman is not None and overall_spearman < 0.10:
        statuses.append("behavior_head_weak_overall_active_metric_alignment")
    if fit_tau is not None and fit_tau < 0.10:
        statuses.append("fitted_behavior_head_weak_target_fit")
    if fit_std_ratio is not None and fit_std_ratio < 0.05:
        statuses.append("fitted_behavior_head_low_contrast")
    if segment_spearman is not None and segment_spearman < 0.10:
        statuses.append("behavior_target_weak_segment_budget_alignment")
    if (
        replacement_spearman is not None
        and segment_spearman is not None
        and replacement_spearman - segment_spearman >= 0.25
    ):
        statuses.append("behavior_target_more_replacement_than_segment_aligned")
    return statuses or ["behavior_head_semantics_no_blocker_detected"]


def _workload_family_pressure(artifact: dict[str, Any]) -> dict[str, Any]:
    generation = _as_dict(artifact.get("query_generation_diagnostics"))
    train = _as_dict(generation.get("train"))
    profile = _as_dict(train.get("workload_profile"))
    signature = _as_dict(train.get("workload_signature"))
    focus_families = _focus_families(artifact)
    return {
        "anchor_family_weights": {
            family: _as_float(_as_dict(profile.get("anchor_family_weights")).get(family))
            for family in focus_families.get("anchor_family", ())
        },
        "footprint_family_weights": {
            family: _as_float(_as_dict(profile.get("footprint_family_weights")).get(family))
            for family in focus_families.get("footprint_family", ())
        },
        "anchor_family_counts": {
            family: _as_float(_as_dict(signature.get("anchor_family_counts")).get(family))
            for family in focus_families.get("anchor_family", ())
        },
        "footprint_family_counts": {
            family: _as_float(_as_dict(signature.get("footprint_family_counts")).get(family))
            for family in focus_families.get("footprint_family", ())
        },
    }


def _artifact_summary(label: str, artifact: dict[str, Any]) -> dict[str, Any]:
    family_rows = _focus_family_rows(artifact)
    retained_marginal_alignment = _retained_marginal_alignment(artifact)
    component_tradeoffs = _causality_component_tradeoffs(artifact)
    return {
        "label": label,
        "target": _target_metadata(artifact),
        "scores": _score_summary(artifact),
        "gates": _gate_summary(artifact),
        "predictability": _predictability_summary(artifact),
        "workload_family_pressure": _workload_family_pressure(artifact),
        "retained_marginal_alignment": retained_marginal_alignment,
        "causality_component_tradeoffs": component_tradeoffs,
        "behavior_head_semantic_alignment": _behavior_head_semantic_alignment(
            artifact,
            retained_marginal_alignment=retained_marginal_alignment,
            component_tradeoffs=component_tradeoffs,
        ),
        "focus_family_rows": family_rows,
        "blocked_family_head_rows": _blocked_family_head_rows(family_rows),
    }


def _blocked_family_head_rows(family_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocked: list[dict[str, Any]] = []
    for row in family_rows:
        for head, payload in _as_dict(row.get("heads")).items():
            status = str(_as_dict(payload).get("transfer_status"))
            if status in {
                "fits_target_but_misorders_ship_evidence",
                "target_positive_fit_negative",
                "target_still_weak",
            }:
                blocked.append(
                    {
                        "group_key": row.get("group_key"),
                        "family": row.get("family"),
                        "head": head,
                        "transfer_status": status,
                        "target_ship_spearman": _as_float(
                            _as_dict(_as_dict(payload).get("target")).get(
                                "spearman_with_ship_query_evidence"
                            )
                        ),
                        "fitted_ship_spearman": _as_float(
                            _as_dict(_as_dict(payload).get("fitted")).get(
                                "spearman_with_family_ship_query_evidence"
                            )
                        ),
                        "kendall_tau_with_head_target": _as_float(
                            _as_dict(_as_dict(payload).get("fitted")).get(
                                "kendall_tau_with_head_target"
                            )
                        ),
                        "prediction_std_to_target_std": _as_float(
                            _as_dict(_as_dict(payload).get("fitted")).get(
                                "prediction_std_to_target_std"
                            )
                        ),
                    }
                )
    return blocked


def _summary(artifact_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    primary = artifact_summaries[-1] if artifact_summaries else {}
    blocked = _as_list(primary.get("blocked_family_head_rows"))
    predictability = _as_dict(primary.get("predictability"))
    retained = _as_dict(primary.get("retained_marginal_alignment"))
    selector_spearman = _as_float(retained.get("selector_score_overall_spearman"))
    active_alignment = _as_dict(retained.get("active_metric_score_component_alignment"))
    behavior_alignment = _active_metric_head_summary(
        active_alignment,
        head="conditional_behavior_utility",
    )
    behavior_tradeoff = _as_dict(
        _as_dict(primary.get("causality_component_tradeoffs")).get(
            "MLQDS_without_behavior_utility_head"
        )
    )
    behavior_semantics = _as_dict(primary.get("behavior_head_semantic_alignment"))
    return {
        "primary_label": primary.get("label"),
        "primary_target_mode": _as_dict(primary.get("target")).get("target_mode"),
        "blocked_family_head_count": len(blocked),
        "blocked_family_head_rows": blocked,
        "family_conditioned_prior_predictability_available": predictability.get(
            "family_conditioned_prior_predictability_available"
        ),
        "retained_marginal_alignment_layout": retained.get("source_layout"),
        "retained_marginal_selector_score_spearman": selector_spearman,
        "active_metric_head_alignment_layout": active_alignment.get("source_layout"),
        "behavior_head_active_metric_alignment": behavior_alignment,
        "without_behavior_head_query_local_utility_delta": _as_float(
            behavior_tradeoff.get("query_local_utility_delta")
        ),
        "without_behavior_head_top_negative_weighted_component_deltas": _as_list(
            behavior_tradeoff.get("top_negative_weighted_component_deltas")
        ),
        "current_metric_behavior_head_status": _current_metric_behavior_head_status(
            behavior_alignment=behavior_alignment,
            behavior_ablation_delta=_as_float(
                behavior_tradeoff.get("query_local_utility_delta")
            ),
        ),
        "behavior_head_semantic_statuses": _as_list(
            behavior_semantics.get("semantic_statuses")
        ),
        "behavior_head_strongest_target_reference": behavior_semantics.get(
            "strongest_target_reference_by_spearman"
        ),
        "decision": _decision(primary, blocked, selector_spearman),
        "interpretation": (
            "This is a derived strict-artifact diagnosis. It can separate "
            "target/head transfer symptoms from missing diagnostic surfaces. "
            "Ship-evidence correlations are legacy diagnostic proxies; use "
            "active metric head alignment and component tradeoffs for current "
            "QueryLocalUtility behavior-head conclusions. This diagnostic does "
            "not prove a new candidate learns."
        ),
    }


def _active_metric_head_summary(
    active_alignment: dict[str, Any],
    *,
    head: str,
) -> dict[str, Any]:
    overall = _as_dict(_as_dict(active_alignment.get("overall")).get(head))
    retained = _as_dict(
        _as_dict(active_alignment.get("retained_removal_loss")).get(head)
    )
    return {
        "overall_spearman": _as_float(overall.get("spearman")),
        "overall_top_minus_bottom_marginal": _as_float(
            overall.get("top_minus_bottom_marginal")
        ),
        "retained_removal_spearman": _as_float(retained.get("spearman")),
        "retained_removal_top_minus_bottom_marginal": _as_float(
            retained.get("top_minus_bottom_marginal")
        ),
    }


def _current_metric_behavior_head_status(
    *,
    behavior_alignment: dict[str, Any],
    behavior_ablation_delta: float | None,
) -> str:
    retained_spearman = _as_float(behavior_alignment.get("retained_removal_spearman"))
    overall_spearman = _as_float(behavior_alignment.get("overall_spearman"))
    if retained_spearman is None and overall_spearman is None and behavior_ablation_delta is None:
        return "active_metric_behavior_head_diagnostics_unavailable"
    if behavior_ablation_delta is not None and behavior_ablation_delta < 0.0:
        return "behavior_head_hurts_active_metric_ablation"
    if retained_spearman is not None and retained_spearman < 0.0:
        return "behavior_head_misorders_retained_active_metric_marginals"
    if overall_spearman is not None and overall_spearman <= 0.0:
        return "behavior_head_weak_overall_active_metric_alignment"
    return "behavior_head_active_metric_alignment_nonnegative"


def _decision(
    primary: dict[str, Any],
    blocked_rows: list[Any],
    selector_spearman: float | None,
) -> str:
    gates = _as_dict(primary.get("gates"))
    predictability = _as_dict(primary.get("predictability"))
    if gates.get("target_diffusion") is False:
        return "reject_target_contract_before_transfer_work"
    if blocked_rows and predictability.get("family_conditioned_prior_predictability_available") is not True:
        return "add_family_conditioned_prior_predictability_before_model_or_scoring_change"
    if selector_spearman is not None and selector_spearman <= 0.0:
        return "diagnose_score_to_selector_marginal_calibration_before_promotion"
    if blocked_rows:
        return "continue_family_head_loss_transfer_diagnosis"
    if gates.get("learning_causality") is False:
        return "continue_failed_causality_child_diagnosis_not_promotion"
    return "diagnostic_only_no_promotion"


def build_family_transfer_path_diagnostic(
    artifacts: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Build a derived diagnostic for diffusion-preserving family/head transfer."""
    summaries = [_artifact_summary(label, artifact) for label, artifact in artifacts]
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "evidence_level": "derived_strict_artifact_diagnostic_no_new_probe",
        "primary_method": PRIMARY_METHOD,
        "baseline_method": BASELINE_METHOD,
        "artifact_count": len(summaries),
        "artifacts": summaries,
        "summary": _summary(summaries),
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
        description="Build a derived family/head transfer path diagnostic."
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
    diagnostic = build_family_transfer_path_diagnostic(artifacts)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(diagnostic, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
