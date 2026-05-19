"""Derived transfer diagnostics for guarded QueryLocalUtility segment targets."""

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
HEAD_NAMES = (
    "query_hit_probability",
    "conditional_behavior_utility",
    "boundary_event_utility",
    "replacement_representative_value",
    "segment_budget_target",
    "path_length_support_target",
)
CAUSALITY_CHILDREN = (
    "shuffled_scores_should_lose",
    "untrained_model_should_lose",
    "shuffled_prior_fields_should_lose",
    "without_query_prior_features_should_lose",
    "without_behavior_utility_head_should_lose",
    "without_segment_budget_head_should_lose",
    "prior_field_only_should_not_match_trained",
)
CAUSALITY_DELTA_FIELDS = {
    "shuffled_scores_should_lose": "shuffled_score_ablation_delta",
    "untrained_model_should_lose": "untrained_score_ablation_delta",
    "shuffled_prior_fields_should_lose": "shuffled_prior_field_ablation_delta",
    "without_query_prior_features_should_lose": "without_query_prior_features_delta",
    "without_behavior_utility_head_should_lose": "no_behavior_head_ablation_delta",
    "without_segment_budget_head_should_lose": "no_segment_budget_head_ablation_delta",
    "prior_field_only_should_not_match_trained": "prior_field_only_score_ablation_delta",
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_float(value: Any, default: float | None = 0.0) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return default


def _as_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _delta(candidate: float | None, reference: float | None) -> float | None:
    if candidate is None or reference is None:
        return None
    return float(candidate - reference)


def _matched_score_summary(
    artifact: dict[str, Any],
    *,
    primary_method: str,
    baseline_method: str,
) -> dict[str, float | None]:
    matched = _as_dict(artifact.get("matched"))
    primary = _as_dict(matched.get(primary_method))
    uniform = _as_dict(matched.get("uniform"))
    baseline = _as_dict(matched.get(baseline_method))
    primary_query = _as_float(primary.get("query_local_utility_score"), None)
    uniform_query = _as_float(uniform.get("query_local_utility_score"), None)
    baseline_query = _as_float(baseline.get("query_local_utility_score"), None)
    primary_range = _as_float(primary.get("range_usefulness_score"), None)
    baseline_range = _as_float(baseline.get("range_usefulness_score"), None)
    return {
        "primary_query_local_utility": primary_query,
        "uniform_query_local_utility": uniform_query,
        "baseline_query_local_utility": baseline_query,
        "primary_minus_uniform_query_local_utility": _delta(primary_query, uniform_query),
        "primary_minus_baseline_query_local_utility": _delta(primary_query, baseline_query),
        "primary_range_useful_legacy": primary_range,
        "baseline_range_useful_legacy": baseline_range,
        "primary_minus_baseline_range_useful_legacy": _delta(primary_range, baseline_range),
    }


def _gate_summary(artifact: dict[str, Any]) -> dict[str, bool | None]:
    return {
        "workload_stability": _as_bool(_as_dict(artifact.get("workload_stability_gate")).get("gate_pass")),
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


def _predictability_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    audit = _as_dict(artifact.get("predictability_audit"))
    metrics = _as_dict(audit.get("metrics"))
    return {
        "gate_pass": _as_bool(audit.get("gate_pass")),
        "failed_checks": [
            str(name)
            for name, passed in _as_dict(audit.get("gate_checks")).items()
            if passed is False
        ],
        "spearman": _as_float(metrics.get("spearman"), None),
        "pr_auc_lift_over_base_rate": _as_float(
            metrics.get("pr_auc_lift_over_base_rate"),
            None,
        ),
        "lift_at_5_percent": _as_float(metrics.get("lift_at_5_percent"), None),
        "positive_target_spearman": _as_float(metrics.get("positive_target_spearman"), None),
    }


def _causality_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    summary = _as_dict(artifact.get("learning_causality_summary"))
    thresholds = _as_dict(_as_dict(summary.get("learning_causality_delta_gate")).get("thresholds"))
    failed = {str(name) for name in _as_list(summary.get("learning_causality_failed_checks"))}
    children: dict[str, Any] = {}
    for child in CAUSALITY_CHILDREN:
        observed = _as_float(summary.get(CAUSALITY_DELTA_FIELDS[child]), None)
        threshold = _as_float(thresholds.get(child), None)
        children[child] = {
            "observed_delta": observed,
            "required_delta": threshold,
            "pass": None if threshold is None or observed is None else observed >= threshold,
            "failed": child in failed,
            "shortfall": (
                None
                if threshold is None or observed is None
                else max(0.0, threshold - observed)
            ),
        }
    return {
        "gate_pass": _as_bool(summary.get("learning_causality_gate_pass")),
        "failed_checks": sorted(failed),
        "children": children,
    }


def _target_family_row(
    artifact: dict[str, Any],
    *,
    group_key: str,
    family: str,
) -> dict[str, Any]:
    diagnostics = _as_dict(
        _as_dict(artifact.get("training_target_diagnostics")).get("query_local_utility_factorized")
    )
    row = _as_dict(
        _as_dict(
            _as_dict(diagnostics.get("family_conditioned_target_trainability")).get("group_by")
        )
        .get(group_key, {})
        .get(family)
    )
    rankers = _as_dict(row.get("ranker_alignment"))
    target_shapes = _as_dict(row.get("target_shapes"))
    out: dict[str, Any] = {
        "available": bool(row),
        "query_count": row.get("query_count"),
        "valid_hit_point_count": row.get("valid_hit_point_count"),
        "target_trainability_status": row.get("target_trainability_status"),
        "weak_ship_evidence_rankers": _as_list(row.get("weak_ship_evidence_rankers")),
        "rankers": {},
    }
    for name in (*HEAD_NAMES, "final_score"):
        ranker = _as_dict(rankers.get(name))
        shape = _as_dict(target_shapes.get(name))
        out["rankers"][name] = {
            "spearman_with_ship_query_evidence": _as_float(
                ranker.get("spearman_with_ship_query_evidence"),
                None,
            ),
            "topk_ship_query_evidence_mass_recall": _as_float(
                ranker.get("topk_ship_query_evidence_mass_recall"),
                None,
            ),
            "target_std": _as_float(shape.get("target_std"), None),
            "target_mean": _as_float(shape.get("target_mean"), None),
        }
    return out


def _fit_family_row(
    artifact: dict[str, Any],
    *,
    group_key: str,
    family: str,
) -> dict[str, Any]:
    fit = _as_dict(artifact.get("training_fit_diagnostics"))
    row = _as_dict(
        _as_dict(_as_dict(fit.get("family_conditioned_head_trainability")).get("group_by"))
        .get(group_key, {})
        .get(family)
    )
    head_fit = _as_dict(row.get("head_fit"))
    out: dict[str, Any] = {
        "available": bool(row),
        "query_count": row.get("query_count"),
        "valid_hit_point_count": row.get("valid_hit_point_count"),
        "head_trainability_status": row.get("head_trainability_status"),
        "weak_ship_evidence_heads": _as_list(row.get("weak_ship_evidence_heads")),
        "heads": {},
    }
    for name in HEAD_NAMES:
        head = _as_dict(head_fit.get(name))
        out["heads"][name] = {
            "spearman_with_family_ship_query_evidence": _as_float(
                head.get("spearman_with_family_ship_query_evidence"),
                None,
            ),
            "topk_family_ship_query_evidence_mass_recall": _as_float(
                head.get("topk_family_ship_query_evidence_mass_recall"),
                None,
            ),
            "kendall_tau_with_head_target": _as_float(
                head.get("kendall_tau_with_head_target"),
                None,
            ),
            "prediction_std": _as_float(head.get("prediction_std"), None),
            "target_std": _as_float(head.get("target_std"), None),
        }
    composed = _as_dict(row.get("factorized_composed_score_fit"))
    out["heads"]["factorized_composed_score"] = {
        "spearman_with_family_ship_query_evidence": _as_float(
            composed.get("spearman_with_family_ship_query_evidence"),
            None,
        ),
        "topk_family_ship_query_evidence_mass_recall": _as_float(
            composed.get("topk_family_ship_query_evidence_mass_recall"),
            None,
        ),
        "kendall_tau_with_head_target": _as_float(
            composed.get("kendall_tau_with_head_target"),
            None,
        ),
        "prediction_std": _as_float(composed.get("prediction_std"), None),
        "target_std": _as_float(composed.get("target_std"), None),
    }
    return out


def _family_local_candidate_row(
    artifact: dict[str, Any],
    *,
    group_key: str,
    family: str,
) -> dict[str, Any]:
    diagnostics = _as_dict(
        _as_dict(artifact.get("training_target_diagnostics")).get("query_local_utility_factorized")
    )
    row = _as_dict(
        _as_dict(
            _as_dict(diagnostics.get("family_local_target_candidate_alignment")).get("group_by")
        )
        .get(group_key, {})
        .get(family)
    )
    candidates = _as_dict(row.get("candidate_alignment"))
    max_candidate = _as_dict(candidates.get("family_query_hit_ship_segment_max_candidate"))
    fractional_candidate = _as_dict(
        candidates.get("family_ship_query_pair_fractional_segment_candidate")
    )
    return {
        "available": bool(row),
        "best_segment_candidate_two_stage_ship_query_pair_coverage": _as_float(
            row.get("best_segment_candidate_two_stage_ship_query_pair_coverage"),
            None,
        ),
        "best_segment_candidate_two_stage_ship_query_evidence_mass_recall": _as_float(
            row.get("best_segment_candidate_two_stage_ship_query_evidence_mass_recall"),
            None,
        ),
        "max_pool_candidate_spearman_with_ship_query_evidence": _as_float(
            max_candidate.get("spearman_with_ship_query_evidence"),
            None,
        ),
        "fractional_candidate_spearman_with_ship_query_evidence": _as_float(
            fractional_candidate.get("spearman_with_ship_query_evidence"),
            None,
        ),
    }


def _family_transfer_summary(
    artifact: dict[str, Any],
    *,
    group_key: str,
    family: str,
) -> dict[str, Any]:
    target = _target_family_row(artifact, group_key=group_key, family=family)
    fitted = _fit_family_row(artifact, group_key=group_key, family=family)
    candidate = _family_local_candidate_row(artifact, group_key=group_key, family=family)
    transfer: dict[str, Any] = {}
    target_rankers = _as_dict(target.get("rankers"))
    fitted_heads = _as_dict(fitted.get("heads"))
    for name in HEAD_NAMES:
        target_ship = _as_float(
            _as_dict(target_rankers.get(name)).get("spearman_with_ship_query_evidence"),
            None,
        )
        fitted_ship = _as_float(
            _as_dict(fitted_heads.get(name)).get(
                "spearman_with_family_ship_query_evidence"
            ),
            None,
        )
        transfer[name] = {
            "target_ship_spearman": target_ship,
            "fitted_ship_spearman": fitted_ship,
            "fitted_minus_target_ship_spearman": _delta(fitted_ship, target_ship),
            "transfer_status": _transfer_status(target_ship, fitted_ship),
        }
    return {
        "target": target,
        "fitted": fitted,
        "family_local_candidates": candidate,
        "target_to_fitted_transfer": transfer,
    }


def _transfer_status(target_ship: float | None, fitted_ship: float | None) -> str:
    if target_ship is None or fitted_ship is None:
        return "unavailable"
    if target_ship >= 0.0 and fitted_ship < 0.0:
        return "target_positive_fit_negative"
    if target_ship < 0.0:
        return "target_still_weak"
    if fitted_ship >= 0.0:
        return "target_and_fit_positive"
    return "diagnostic_only"


def _retained_marginal_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    alignment = _as_dict(
        _as_dict(_as_dict(artifact.get("selector_trace_diagnostics")).get("eval_primary")).get(
            "retained_decision_marginal_query_local_utility_alignment"
        )
    )
    overall = _as_dict(alignment.get("overall"))
    by_decision = _as_dict(alignment.get("by_decision"))
    retained_loss = _as_dict(by_decision.get("retained_removal_loss"))
    return {
        "available": _as_bool(alignment.get("available")),
        "candidate_count": alignment.get("candidate_count"),
        "overall": _score_alignment_subset(overall),
        "retained_removal_loss": _score_alignment_subset(retained_loss),
        "layout": (
            "selector_trace_diagnostics.eval_primary."
            "retained_decision_marginal_query_local_utility_alignment"
        ),
    }


def _score_alignment_subset(row: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "spearman": _as_float(_as_dict(row.get(name)).get("spearman"), None),
            "top_minus_bottom_marginal": _as_float(
                _as_dict(row.get(name)).get("top_minus_bottom_marginal"),
                None,
            ),
        }
        for name in ("raw_score", "selector_score", "segment_score")
    }


def _artifact_summary(
    artifact: dict[str, Any],
    *,
    label: str,
    primary_method: str,
    baseline_method: str,
) -> dict[str, Any]:
    target_diagnostics = _as_dict(
        _as_dict(artifact.get("training_target_diagnostics")).get("query_local_utility_factorized")
    )
    family_transfer = {
        group_key: {
            family: _family_transfer_summary(artifact, group_key=group_key, family=family)
            for family in families
        }
        for group_key, families in HISTORICAL_DIAGNOSTIC_FOCUS_FAMILIES.items()
    }
    return {
        "label": label,
        "target_mode": target_diagnostics.get("target_mode")
        or _as_dict(_as_dict(artifact.get("config")).get("model")).get(
            "range_training_target_mode"
        ),
        "segment_budget_target_variant": target_diagnostics.get(
            "segment_budget_target_variant"
        ),
        "segment_budget_target_aggregation": target_diagnostics.get(
            "segment_budget_target_aggregation"
        ),
        "scores": _matched_score_summary(
            artifact,
            primary_method=primary_method,
            baseline_method=baseline_method,
        ),
        "gates": _gate_summary(artifact),
        "predictability": _predictability_summary(artifact),
        "causality": _causality_summary(artifact),
        "family_transfer": family_transfer,
        "retained_marginal_alignment": _retained_marginal_summary(artifact),
    }


def _candidate_vs_reference_comparison(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    reference_causality = _as_dict(reference.get("causality"))
    candidate_causality = _as_dict(candidate.get("causality"))
    reference_children = _as_dict(reference_causality.get("children"))
    candidate_children = _as_dict(candidate_causality.get("children"))
    newly_passing = []
    still_failing = []
    worsened_shortfalls = []
    for child in CAUSALITY_CHILDREN:
        ref_child = _as_dict(reference_children.get(child))
        cand_child = _as_dict(candidate_children.get(child))
        if cand_child.get("pass") is True and ref_child.get("pass") is not True:
            newly_passing.append(child)
        if cand_child.get("pass") is False:
            still_failing.append(child)
        ref_shortfall = _as_float(ref_child.get("shortfall"), None)
        cand_shortfall = _as_float(cand_child.get("shortfall"), None)
        if ref_shortfall is not None and cand_shortfall is not None and cand_shortfall > ref_shortfall:
            worsened_shortfalls.append(child)
    focus_rows: list[dict[str, Any]] = []
    for group_key, families in HISTORICAL_DIAGNOSTIC_FOCUS_FAMILIES.items():
        for family in families:
            focus_rows.append(
                _family_comparison_row(
                    reference,
                    candidate,
                    group_key=group_key,
                    family=family,
                )
            )
    return {
        "reference_label": reference.get("label"),
        "candidate_label": candidate.get("label"),
        "score_delta": _score_delta_summary(
            _as_dict(reference.get("scores")),
            _as_dict(candidate.get("scores")),
        ),
        "gate_changes": _gate_changes(
            _as_dict(reference.get("gates")),
            _as_dict(candidate.get("gates")),
        ),
        "causality_changes": {
            "newly_passing_child_gates": newly_passing,
            "still_failing_child_gates": still_failing,
            "worsened_shortfall_child_gates": worsened_shortfalls,
        },
        "focus_family_transfer_rows": focus_rows,
        "families_with_positive_target_negative_fit": [
            row
            for row in focus_rows
            if row.get("segment_budget_transfer_status") == "target_positive_fit_negative"
        ],
        "decision": _comparison_decision(candidate, focus_rows, still_failing),
    }


def _score_delta_summary(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "primary_query_local_utility",
        "primary_minus_baseline_query_local_utility",
        "primary_range_useful_legacy",
        "primary_minus_baseline_range_useful_legacy",
    )
    return {key: _delta(_as_float(candidate.get(key), None), _as_float(reference.get(key), None)) for key in keys}


def _gate_changes(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {
            "reference": reference.get(key),
            "candidate": candidate.get(key),
            "changed": reference.get(key) != candidate.get(key),
        }
        for key in sorted(set(reference) | set(candidate))
    }


def _family_comparison_row(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    group_key: str,
    family: str,
) -> dict[str, Any]:
    ref_family = _as_dict(
        _as_dict(_as_dict(reference.get("family_transfer")).get(group_key)).get(family)
    )
    cand_family = _as_dict(
        _as_dict(_as_dict(candidate.get("family_transfer")).get(group_key)).get(family)
    )
    ref_transfer = _as_dict(_as_dict(ref_family.get("target_to_fitted_transfer")).get("segment_budget_target"))
    cand_transfer = _as_dict(_as_dict(cand_family.get("target_to_fitted_transfer")).get("segment_budget_target"))
    ref_composed = _as_dict(
        _as_dict(_as_dict(ref_family.get("fitted")).get("heads")).get(
            "factorized_composed_score"
        )
    )
    cand_composed = _as_dict(
        _as_dict(_as_dict(cand_family.get("fitted")).get("heads")).get(
            "factorized_composed_score"
        )
    )
    ref_target = _as_float(ref_transfer.get("target_ship_spearman"), None)
    cand_target = _as_float(cand_transfer.get("target_ship_spearman"), None)
    ref_fit = _as_float(ref_transfer.get("fitted_ship_spearman"), None)
    cand_fit = _as_float(cand_transfer.get("fitted_ship_spearman"), None)
    ref_composed_ship = _as_float(
        ref_composed.get("spearman_with_family_ship_query_evidence"),
        None,
    )
    cand_composed_ship = _as_float(
        cand_composed.get("spearman_with_family_ship_query_evidence"),
        None,
    )
    return {
        "group_key": group_key,
        "family": family,
        "reference_segment_target_ship_spearman": ref_target,
        "candidate_segment_target_ship_spearman": cand_target,
        "segment_target_ship_spearman_delta": _delta(cand_target, ref_target),
        "reference_segment_fit_ship_spearman": ref_fit,
        "candidate_segment_fit_ship_spearman": cand_fit,
        "segment_fit_ship_spearman_delta": _delta(cand_fit, ref_fit),
        "candidate_segment_fit_minus_target_ship_spearman": _delta(cand_fit, cand_target),
        "reference_composed_fit_ship_spearman": ref_composed_ship,
        "candidate_composed_fit_ship_spearman": cand_composed_ship,
        "composed_fit_ship_spearman_delta": _delta(cand_composed_ship, ref_composed_ship),
        "segment_budget_transfer_status": cand_transfer.get("transfer_status"),
        "candidate_weak_ship_evidence_heads": _as_list(
            _as_dict(cand_family.get("fitted")).get("weak_ship_evidence_heads")
        ),
    }


def _comparison_decision(
    candidate: dict[str, Any],
    focus_rows: list[dict[str, Any]],
    still_failing: list[str],
) -> str:
    gates = _as_dict(candidate.get("gates"))
    if gates.get("predictability") is True and gates.get("learning_causality") is True:
        return "candidate_needs_next_strict_stage_before_promotion"
    if any(
        row.get("segment_budget_transfer_status") == "target_positive_fit_negative"
        for row in focus_rows
    ):
        return "continue_with_family_transfer_diagnosis_not_promotion"
    if still_failing:
        return "continue_with_failed_causality_child_diagnosis_not_promotion"
    return "diagnostic_only_no_promotion"


def build_query_ship_max_pool_transfer_diagnostic(
    artifacts: list[tuple[str, dict[str, Any]]],
    *,
    primary_method: str = PRIMARY_METHOD,
    baseline_method: str = BASELINE_METHOD,
) -> dict[str, Any]:
    """Return a derived comparison for guarded segment aggregation target variants."""
    summaries = [
        _artifact_summary(
            artifact,
            label=label,
            primary_method=primary_method,
            baseline_method=baseline_method,
        )
        for label, artifact in artifacts
    ]
    reference = summaries[0] if summaries else {}
    candidate = summaries[1] if len(summaries) > 1 else {}
    comparison = (
        _candidate_vs_reference_comparison(reference, candidate)
        if reference and candidate
        else {}
    )
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "evidence_level": "derived_strict_artifact_diagnostic_no_new_probe",
        "primary_method": primary_method,
        "baseline_method": baseline_method,
        "artifact_count": len(summaries),
        "artifacts": summaries,
        "summary": {
            "reference_label": reference.get("label"),
            "candidate_label": candidate.get("label"),
            "candidate_target_mode": candidate.get("target_mode"),
            "candidate_final_success_allowed": _as_dict(candidate.get("gates")).get(
                "final_success_allowed"
            ),
            "comparison": comparison,
            "interpretation": (
                "The guarded query-ship max-pool target is useful diagnostic evidence "
                "only. It should be promoted only after unchanged strict predictability "
                "and learning-causality gates pass, with family transfer rows no longer "
                "showing positive target signal but negative fitted signal."
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
        description="Build a derived QueryLocalUtility segment-target transfer diagnostic."
    )
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help=(
            "Artifact path, optionally label=path. Pass reference first and "
            "candidate second for comparison."
        ),
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args(argv)

    artifacts = [
        (label, _load_json(path))
        for label, path in (_parse_labeled_artifact(value) for value in args.artifact)
    ]
    diagnostic = build_query_ship_max_pool_transfer_diagnostic(artifacts)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(diagnostic, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
