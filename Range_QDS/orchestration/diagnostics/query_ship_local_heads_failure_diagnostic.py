"""Derived failure diagnostic for the guarded query-ship local-head target."""

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
COMPOSED_HEAD_NAME = "factorized_composed_score"
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


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return None


def _as_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _delta(candidate: float | None, reference: float | None) -> float | None:
    if candidate is None or reference is None:
        return None
    return float(candidate - reference)


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
    workload_signature = _as_bool(
        _as_dict(
            _as_dict(artifact.get("workload_distribution_comparison")).get(
                "workload_signature_gate"
            )
        ).get("all_pass")
    )
    return {
        "workload_stability": _as_bool(
            _as_dict(artifact.get("workload_stability_gate")).get("gate_pass")
        ),
        "support_overlap": _as_bool(_as_dict(artifact.get("support_overlap_gate")).get("gate_pass")),
        "target_diffusion": _as_bool(_as_dict(artifact.get("target_diffusion_gate")).get("gate_pass")),
        "workload_signature": workload_signature,
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
        "final_label_variant": target.get("final_label_variant"),
        "segment_budget_target_variant": target.get("segment_budget_target_variant"),
        "segment_budget_target_aggregation": target.get("segment_budget_target_aggregation"),
        "final_success_allowed": _as_bool(target.get("final_success_allowed")),
    }


def _target_diffusion_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    gate = _as_dict(artifact.get("target_diffusion_gate"))
    rows = _as_list(gate.get("head_rows"))
    return {
        "gate_pass": _as_bool(gate.get("gate_pass")),
        "failed_checks": [str(item) for item in _as_list(gate.get("failed_checks"))],
        "final_label_support_fraction": _as_float(gate.get("final_label_support_fraction")),
        "max_support_fraction": _as_float(gate.get("max_support_fraction")),
        "blocking_head_rows": [
            {
                "head": row.get("head"),
                "support_fraction": _as_float(row.get("support_fraction")),
                "top5_label_mass_fraction": _as_float(row.get("top5_label_mass_fraction")),
                "failed_checks": [str(item) for item in _as_list(row.get("failed_checks"))],
            }
            for row in rows
            if isinstance(row, dict) and row.get("blocking") is True
        ],
    }


def _predictability_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    audit = _as_dict(artifact.get("predictability_audit"))
    metrics = _as_dict(audit.get("metrics"))
    prior_gate = _as_dict(audit.get("prior_predictive_alignment_gate"))
    return {
        "gate_pass": _as_bool(audit.get("gate_pass")),
        "spearman": _as_float(metrics.get("spearman")),
        "positive_target_spearman": _as_float(metrics.get("positive_target_spearman")),
        "pr_auc_lift_over_base_rate": _as_float(metrics.get("pr_auc_lift_over_base_rate")),
        "lift_at_5_percent": _as_float(metrics.get("lift_at_5_percent")),
        "prior_predictive_alignment_gate_pass": _as_bool(prior_gate.get("gate_pass")),
        "prior_predictive_failed_checks": [
            str(item) for item in _as_list(prior_gate.get("failed_checks"))
        ],
        "positive_spearman_head_count": prior_gate.get("positive_spearman_head_count"),
    }


def _causality_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    summary = _as_dict(artifact.get("learning_causality_summary"))
    thresholds = _as_dict(_as_dict(summary.get("learning_causality_delta_gate")).get("thresholds"))
    failed = {str(item) for item in _as_list(summary.get("learning_causality_failed_checks"))}
    children: dict[str, Any] = {}
    for child, field in CAUSALITY_DELTA_FIELDS.items():
        observed = _as_float(summary.get(field))
        threshold = _as_float(thresholds.get(child))
        children[child] = {
            "observed_delta": observed,
            "required_delta": threshold,
            "pass": None if observed is None or threshold is None else observed >= threshold,
            "failed": child in failed,
            "shortfall": (
                None
                if observed is None or threshold is None
                else max(0.0, threshold - observed)
            ),
        }
    return {
        "gate_pass": _as_bool(summary.get("learning_causality_gate_pass")),
        "failed_checks": sorted(failed),
        "children": children,
    }


def _target_family_rankers(
    artifact: dict[str, Any],
    *,
    group_key: str,
    family: str,
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
    return {
        "available": bool(row),
        "weak_ship_evidence_rankers": [str(item) for item in _as_list(row.get("weak_ship_evidence_rankers"))],
        "rankers": {
            name: _as_float(
                _as_dict(rankers.get(name)).get("spearman_with_ship_query_evidence")
            )
            for name in (*HEAD_NAMES, "final_score")
        },
    }


def _fitted_family_heads(
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
    heads = {
        name: _as_float(
            _as_dict(head_fit.get(name)).get(
                "spearman_with_family_ship_query_evidence"
            )
        )
        for name in HEAD_NAMES
    }
    heads[COMPOSED_HEAD_NAME] = _as_float(
        _as_dict(row.get("factorized_composed_score_fit")).get(
            "spearman_with_family_ship_query_evidence"
        )
    )
    return {
        "available": bool(row),
        "weak_ship_evidence_heads": [str(item) for item in _as_list(row.get("weak_ship_evidence_heads"))],
        "heads": heads,
    }


def _family_failure_row(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    group_key: str,
    family: str,
) -> dict[str, Any]:
    ref_targets = _target_family_rankers(reference, group_key=group_key, family=family)
    cand_targets = _target_family_rankers(candidate, group_key=group_key, family=family)
    ref_fit = _fitted_family_heads(reference, group_key=group_key, family=family)
    cand_fit = _fitted_family_heads(candidate, group_key=group_key, family=family)
    target_rankers = _as_dict(cand_targets.get("rankers"))
    fitted_heads = _as_dict(cand_fit.get("heads"))
    positive_target_negative_fit = [
        name
        for name in ("query_hit_probability", "conditional_behavior_utility", "segment_budget_target")
        if _is_positive(target_rankers.get(name)) and _is_negative(fitted_heads.get(name))
    ]
    return {
        "group_key": group_key,
        "family": family,
        "candidate_positive_target_negative_fit_heads": positive_target_negative_fit,
        "candidate_target_spearman": target_rankers,
        "candidate_fitted_spearman": fitted_heads,
        "reference_target_spearman": _as_dict(ref_targets.get("rankers")),
        "reference_fitted_spearman": _as_dict(ref_fit.get("heads")),
        "candidate_weak_ship_evidence_heads": _as_list(cand_fit.get("weak_ship_evidence_heads")),
        "target_to_fit_gap": {
            name: _delta(_as_float(fitted_heads.get(name)), _as_float(target_rankers.get(name)))
            for name in ("query_hit_probability", "conditional_behavior_utility", "segment_budget_target")
        },
    }


def _is_positive(value: Any) -> bool:
    numeric = _as_float(value)
    return numeric is not None and numeric >= 0.0


def _is_negative(value: Any) -> bool:
    numeric = _as_float(value)
    return numeric is not None and numeric < 0.0


def _artifact_summary(label: str, artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "target": _target_metadata(artifact),
        "scores": _score_summary(artifact),
        "gates": _gate_summary(artifact),
        "target_diffusion": _target_diffusion_summary(artifact),
        "predictability": _predictability_summary(artifact),
        "causality": _causality_summary(artifact),
    }


def _comparison(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_summary = _artifact_summary("reference", reference)
    candidate_summary = _artifact_summary("candidate", candidate)
    family_rows = [
        _family_failure_row(reference, candidate, group_key=group_key, family=family)
        for group_key, families in HISTORICAL_DIAGNOSTIC_FOCUS_FAMILIES.items()
        for family in families
    ]
    return {
        "score_delta": {
            "primary_query_local_utility": _delta(
                candidate_summary["scores"]["primary_query_local_utility"],
                reference_summary["scores"]["primary_query_local_utility"],
            ),
            "primary_minus_baseline_query_local_utility": _delta(
                candidate_summary["scores"]["primary_minus_baseline_query_local_utility"],
                reference_summary["scores"]["primary_minus_baseline_query_local_utility"],
            ),
        },
        "gate_regressions": [
            name
            for name, reference_value in reference_summary["gates"].items()
            if reference_value is True and candidate_summary["gates"].get(name) is False
        ],
        "target_diffusion_failure": candidate_summary["target_diffusion"],
        "prior_predictive_failure": candidate_summary["predictability"][
            "prior_predictive_failed_checks"
        ],
        "causality_failed_checks": candidate_summary["causality"]["failed_checks"],
        "family_failure_rows": family_rows,
        "decision": _decision(candidate_summary, family_rows),
    }


def _decision(candidate_summary: dict[str, Any], family_rows: list[dict[str, Any]]) -> str:
    if candidate_summary["gates"].get("target_diffusion") is False:
        return "reject_broad_local_heads_preserve_diffusion_before_next_transfer_probe"
    if any(row["candidate_positive_target_negative_fit_heads"] for row in family_rows):
        return "continue_model_loss_prior_transfer_diagnosis_not_promotion"
    if candidate_summary["gates"].get("learning_causality") is False:
        return "continue_causality_child_diagnosis_not_promotion"
    return "diagnostic_only_no_promotion"


def build_query_ship_local_heads_failure_diagnostic(
    artifacts: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Build a derived failure diagnosis for the local-head target contract."""
    summaries = [_artifact_summary(label, artifact) for label, artifact in artifacts]
    reference = artifacts[0][1] if artifacts else {}
    candidate = artifacts[1][1] if len(artifacts) > 1 else {}
    comparison = _comparison(reference, candidate) if reference and candidate else {}
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "evidence_level": "derived_strict_artifact_diagnostic_no_new_probe",
        "primary_method": PRIMARY_METHOD,
        "baseline_method": BASELINE_METHOD,
        "artifact_count": len(summaries),
        "artifacts": summaries,
        "summary": {
            "reference_label": artifacts[0][0] if artifacts else None,
            "candidate_label": artifacts[1][0] if len(artifacts) > 1 else None,
            "comparison": comparison,
            "interpretation": (
                "The local-head target made target-side family signs positive but "
                "failed target diffusion and did not transfer to fitted family-head "
                "ordering. Treat it as a rejected diagnostic, not a promotion path."
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
        description="Build a derived query-ship local-head target failure diagnostic."
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
    diagnostic = build_query_ship_local_heads_failure_diagnostic(artifacts)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(diagnostic, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
