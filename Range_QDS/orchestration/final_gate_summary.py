"""Final single-cell gate and summary assembly for run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from config.run_config import (
    DEFAULT_LEARNED_SEGMENT_ALLOCATION_LENGTH_SUPPORT_WEIGHT,
    DEFAULT_LEARNED_SEGMENT_ALLOCATION_WEIGHT_FLOOR,
    RunConfig,
)
from learning.model_features import WORKLOAD_BLIND_RANGE_V2_MODEL_TYPE
from learning.outputs import TrainingOutputs
from learning.targets.query_local_utility import QUERY_LOCAL_UTILITY_FACTORIZED_TARGET_MODE
from orchestration.causality import (
    LEARNING_CAUSALITY_MIN_MATERIAL_DELTA,
    build_learned_slot_summary,
    causality_ablation_tradeoff_summary,
    learning_causality_delta_gate_config,
    prior_sample_gate_failures,
    query_local_utility_component_delta_summary,
    query_local_utility_delta,
)
from orchestration.gates import (
    evaluate_global_sanity_gate,
    evaluate_support_overlap_gate,
    evaluate_target_diffusion_gate,
    evaluate_workload_stability_gate,
)
from scoring.metrics import MethodScore
from selection.selector_types import (
    LEARNED_SEGMENT_BUDGET_SELECTOR_TYPE,
    TEMPORAL_HYBRID_SELECTOR_TYPE,
)
from workloads.generation.workload_profiles import RANGE_QUERY_MIX_FINAL_PROFILE_IDS


@dataclass(frozen=True)
class FinalRunSummaries:
    """Final-claim and gate payloads used by run artifacts."""

    final_candidate: bool
    final_claim_summary: dict[str, Any]
    diagnostic_summary: dict[str, Any]
    legacy_range_useful_summary: dict[str, Any]
    learning_causality_summary: dict[str, Any]
    support_overlap_gate: dict[str, Any]
    global_sanity_gate: dict[str, Any]
    target_diffusion_gate: dict[str, Any]
    workload_stability_gate: dict[str, Any]


def build_final_run_summaries(
    *,
    config: RunConfig,
    trained: TrainingOutputs,
    train_points: torch.Tensor,
    test_points: torch.Tensor,
    train_label_workloads: list[Any],
    eval_workload: Any,
    selection_workload: Any | None,
    matched: dict[str, MethodScore],
    selector_budget_diagnostics: dict[str, Any],
    primary_selector_trace: dict[str, Any] | None,
    causality_ablation_scores: dict[str, MethodScore],
    causality_ablation_mask_diagnostics: dict[str, dict[str, Any]],
    causal_ablation_freeze_failures: dict[str, str],
    prior_sensitivity_diagnostics: dict[str, Any],
    prior_channel_ablation_diagnostics: dict[str, Any],
    head_ablation_sensitivity_diagnostics: dict[str, Any],
    selection_causality_diagnostics: dict[str, Any],
    segment_oracle_allocation_audit: dict[str, Any],
    target_segment_oracle_alignment_audit: dict[str, Any],
    segment_budget_head_ablation_mode: str | None,
    predictability_audit: dict[str, Any],
    workload_distribution_comparison: dict[str, Any],
) -> FinalRunSummaries:
    """Build final query-driven gate summaries from already computed scores."""
    uniform_score = matched.get("uniform")
    douglas_peucker_score = matched.get("DouglasPeucker")
    workload_signature_gate = workload_distribution_comparison.get("workload_signature_gate", {})
    predictability_gate_pass = bool(predictability_audit.get("gate_pass", False))
    prior_predictive_alignment_gate = predictability_audit.get(
        "prior_predictive_alignment_gate", {}
    )
    prior_predictive_alignment_gate_pass = bool(
        isinstance(prior_predictive_alignment_gate, dict)
        and prior_predictive_alignment_gate.get("gate_pass", False)
    )
    signature_gate_pass = bool(
        isinstance(workload_signature_gate, dict)
        and workload_signature_gate.get("all_available")
        and workload_signature_gate.get("all_pass")
    )
    workload_stability_gate = evaluate_workload_stability_gate(
        config=config,
        train_label_workloads=train_label_workloads,
        eval_workload=eval_workload,
        selection_workload=selection_workload,
    )
    workload_stability_gate_pass = bool(workload_stability_gate.get("gate_pass", False))
    support_overlap_gate = evaluate_support_overlap_gate(
        train_points=train_points,
        eval_points=test_points,
        query_prior_field=trained.feature_context.get("query_prior_field"),
    )
    support_overlap_gate_pass = bool(support_overlap_gate.get("gate_pass", False))
    target_diffusion_gate = evaluate_target_diffusion_gate(trained.target_diagnostics)
    target_diffusion_gate_pass = bool(target_diffusion_gate.get("gate_pass", False))
    final_candidate = (
        str(config.query.workload_profile_id or "").lower() in RANGE_QUERY_MIX_FINAL_PROFILE_IDS
        and str(config.model.model_type).lower() == WORKLOAD_BLIND_RANGE_V2_MODEL_TYPE
        and str(config.model.range_training_target_mode).lower()
        == QUERY_LOCAL_UTILITY_FACTORIZED_TARGET_MODE
        and str(getattr(config.model, "selector_type", "")).lower()
        == LEARNED_SEGMENT_BUDGET_SELECTOR_TYPE
    )
    legacy_range_useful_summary = {
        "metric": "RangeUsefulLegacy",
        "schema": "range_usefulness_schema_version",
        "diagnostic_only": True,
        "mlqds_score": matched["MLQDS"].range_usefulness_score,
        "uniform_score": (
            uniform_score.range_usefulness_score if uniform_score is not None else None
        ),
        "douglas_peucker_score": (
            douglas_peucker_score.range_usefulness_score
            if douglas_peucker_score is not None
            else None
        ),
    }
    learned_slot_summary = build_learned_slot_summary(
        selector_budget_diagnostics,
        float(config.model.compression_ratio),
        primary_selector_trace,
    )
    primary_score = matched["MLQDS"]
    shuffled_delta = query_local_utility_delta(
        primary_score, causality_ablation_scores, "MLQDS_shuffled_scores"
    )
    prior_only_delta = query_local_utility_delta(
        primary_score,
        causality_ablation_scores,
        "MLQDS_prior_field_only_score",
    )
    untrained_delta = query_local_utility_delta(
        primary_score, causality_ablation_scores, "MLQDS_untrained_model"
    )
    shuffled_prior_delta = query_local_utility_delta(
        primary_score,
        causality_ablation_scores,
        "MLQDS_shuffled_prior_fields",
    )
    no_query_prior_delta = query_local_utility_delta(
        primary_score,
        causality_ablation_scores,
        "MLQDS_without_query_prior_features",
    )
    no_behavior_head_delta = query_local_utility_delta(
        primary_score,
        causality_ablation_scores,
        "MLQDS_without_behavior_utility_head",
    )
    no_segment_budget_head_delta = query_local_utility_delta(
        primary_score,
        causality_ablation_scores,
        "MLQDS_without_segment_budget_head",
    )
    no_fairness_preallocation_delta = query_local_utility_delta(
        primary_score,
        causality_ablation_scores,
        "MLQDS_without_trajectory_fairness_preallocation",
    )
    no_geometry_tie_breaker_delta = query_local_utility_delta(
        primary_score,
        causality_ablation_scores,
        "MLQDS_without_geometry_tie_breaker",
    )
    allocation_length_support_weight = float(
        getattr(
            config.model,
            "learned_segment_allocation_length_support_weight",
            DEFAULT_LEARNED_SEGMENT_ALLOCATION_LENGTH_SUPPORT_WEIGHT,
        )
    )
    allocation_weight_floor = float(
        getattr(
            config.model,
            "learned_segment_allocation_weight_floor",
            DEFAULT_LEARNED_SEGMENT_ALLOCATION_WEIGHT_FLOOR,
        )
    )
    no_segment_length_support_allocation_delta = query_local_utility_delta(
        primary_score,
        causality_ablation_scores,
        "MLQDS_without_segment_length_support_allocation",
    )
    causality_ablation_component_deltas = query_local_utility_component_delta_summary(
        primary=primary_score,
        ablations=causality_ablation_scores,
    )
    causality_ablation_tradeoff_diagnostics = causality_ablation_tradeoff_summary(
        component_deltas=causality_ablation_component_deltas,
        mask_diagnostics=causality_ablation_mask_diagnostics,
    )
    for name, tradeoff_diagnostics in causality_ablation_tradeoff_diagnostics.items():
        if name in head_ablation_sensitivity_diagnostics:
            head_ablation_sensitivity_diagnostics[name]["query_local_utility_component_tradeoff"] = (
                tradeoff_diagnostics
            )
    for _prior_channel_name, channel_diagnostics in prior_channel_ablation_diagnostics.items():
        if not isinstance(channel_diagnostics, dict) or not bool(
            channel_diagnostics.get("available", False)
        ):
            continue
        method_name = str(channel_diagnostics.get("method_name", ""))
        channel_score = causality_ablation_scores.get(method_name)
        if channel_score is not None:
            channel_diagnostics["query_local_utility_score"] = float(
                channel_score.query_local_utility_score
            )
            channel_diagnostics["query_local_utility_delta"] = query_local_utility_delta(
                primary_score,
                causality_ablation_scores,
                method_name,
            )
        if method_name in causality_ablation_mask_diagnostics:
            channel_diagnostics["retained_mask"] = causality_ablation_mask_diagnostics[method_name]
        if method_name in causality_ablation_component_deltas:
            channel_diagnostics["query_local_utility_component_deltas"] = (
                causality_ablation_component_deltas[method_name]
            )
        if method_name in causality_ablation_tradeoff_diagnostics:
            channel_diagnostics["query_local_utility_component_tradeoff"] = (
                causality_ablation_tradeoff_diagnostics[method_name]
            )
    required_causality_ablation_names = (
        "MLQDS_shuffled_scores",
        "MLQDS_untrained_model",
        "MLQDS_shuffled_prior_fields",
        "MLQDS_without_query_prior_features",
        "MLQDS_without_behavior_utility_head",
        "MLQDS_without_segment_budget_head",
    )
    missing_causality_ablations = [
        name
        for name in required_causality_ablation_names
        if name not in causality_ablation_scores
    ]
    failed_causality_checks: list[str] = []
    delta_checks = {
        "shuffled_scores_should_lose": shuffled_delta,
        "untrained_model_should_lose": untrained_delta,
        "shuffled_prior_fields_should_lose": shuffled_prior_delta,
        "without_query_prior_features_should_lose": no_query_prior_delta,
        "without_behavior_utility_head_should_lose": no_behavior_head_delta,
        "without_segment_budget_head_should_lose": no_segment_budget_head_delta,
        "prior_field_only_should_not_match_trained": prior_only_delta,
    }
    delta_gate_config = learning_causality_delta_gate_config(
        primary=primary_score,
        uniform=uniform_score,
    )
    delta_thresholds = delta_gate_config.get("thresholds", {})
    for check_name, delta in delta_checks.items():
        threshold = float(delta_thresholds.get(check_name, LEARNING_CAUSALITY_MIN_MATERIAL_DELTA))
        if delta is not None and float(delta) + 1e-12 < threshold:
            failed_causality_checks.append(check_name)
    prior_sample_failures = prior_sample_gate_failures(prior_sensitivity_diagnostics)
    failed_causality_checks.extend(prior_sample_failures)
    learned_slot_fraction = float(
        learned_slot_summary.get("learned_controlled_retained_slot_fraction") or 0.0
    )
    learned_slot_fraction_min = 0.0
    if float(config.model.compression_ratio) >= 0.10:
        learned_slot_fraction_min = 0.35
    elif float(config.model.compression_ratio) >= 0.05:
        learned_slot_fraction_min = 0.25
    if learned_slot_fraction_min > 0.0 and learned_slot_fraction < learned_slot_fraction_min:
        failed_causality_checks.append("learned_controlled_slot_fraction_below_minimum")
    ablation_status = "not_run"
    if causality_ablation_scores or causal_ablation_freeze_failures:
        ablation_status = (
            "complete"
            if not missing_causality_ablations and not causal_ablation_freeze_failures
            else "partial"
        )
    learning_causality_gate_pass = (
        ablation_status == "complete"
        and not failed_causality_checks
        and not missing_causality_ablations
    )
    learning_causality_summary = {
        "selector_diagnostics_present": bool(selector_budget_diagnostics),
        "training_fit_diagnostics_present": bool(trained.fit_diagnostics),
        "selector_type": str(getattr(config.model, "selector_type", TEMPORAL_HYBRID_SELECTOR_TYPE)),
        "selector_final_candidate": str(
            getattr(config.model, "selector_type", TEMPORAL_HYBRID_SELECTOR_TYPE)
        )
        == LEARNED_SEGMENT_BUDGET_SELECTOR_TYPE,
        "query_prior_field_available": bool(trained.feature_context.get("query_prior_field")),
        **learned_slot_summary,
        "shuffled_score_ablation_delta": shuffled_delta,
        "untrained_score_ablation_delta": untrained_delta,
        "shuffled_prior_field_ablation_delta": shuffled_prior_delta,
        "no_query_prior_field_ablation_delta": no_query_prior_delta,
        "no_behavior_head_ablation_delta": no_behavior_head_delta,
        "no_segment_budget_head_ablation_delta": no_segment_budget_head_delta,
        "no_trajectory_fairness_preallocation_ablation_delta": no_fairness_preallocation_delta,
        "no_geometry_tie_breaker_ablation_delta": no_geometry_tie_breaker_delta,
        "no_segment_length_support_allocation_ablation_delta": (
            no_segment_length_support_allocation_delta
        ),
        "segment_budget_head_ablation_mode": segment_budget_head_ablation_mode,
        "learned_segment_selector_config": {
            "geometry_gain_weight": float(config.model.learned_segment_geometry_gain_weight),
            "allocation_length_support_weight": allocation_length_support_weight,
            "allocation_weight_floor": allocation_weight_floor,
            "segment_score_blend_weight": float(config.model.learned_segment_score_blend_weight),
            "segment_transfer_calibration_mode": str(
                getattr(config.model, "learned_segment_transfer_calibration_mode", "none")
            ),
            "fairness_preallocation_enabled": bool(
                config.model.learned_segment_fairness_preallocation
            ),
            "length_repair_fraction": float(config.model.learned_segment_length_repair_fraction),
            "length_repair_score_protection_fraction": float(
                config.model.learned_segment_length_repair_score_protection_fraction
            ),
            "length_support_blend_weight": float(
                config.model.learned_segment_length_support_blend_weight
            ),
        },
        "prior_field_only_score_ablation_delta": prior_only_delta,
        "without_query_prior_features_delta": no_query_prior_delta,
        "learning_causality_delta_gate": delta_gate_config,
        "prior_sensitivity_diagnostics": prior_sensitivity_diagnostics,
        "prior_channel_ablation_diagnostics": prior_channel_ablation_diagnostics,
        "head_ablation_sensitivity_diagnostics": head_ablation_sensitivity_diagnostics,
        "selection_causality_diagnostics": dict(selection_causality_diagnostics),
        "segment_oracle_allocation_audit": segment_oracle_allocation_audit,
        "target_segment_oracle_alignment_audit": target_segment_oracle_alignment_audit,
        "score_protected_length_feasibility": (
            primary_selector_trace.get("score_protected_length_feasibility")
            if isinstance(primary_selector_trace, dict)
            else None
        ),
        "score_protected_length_frontier": (
            primary_selector_trace.get("score_protected_length_frontier")
            if isinstance(primary_selector_trace, dict)
            else None
        ),
        "prior_sample_gate_pass": not prior_sample_failures,
        "prior_sample_gate_failures": prior_sample_failures,
        "causality_ablation_scores": {
            name: metrics.query_local_utility_score
            for name, metrics in causality_ablation_scores.items()
        },
        "causality_ablation_component_deltas": causality_ablation_component_deltas,
        "causality_ablation_mask_diagnostics": causality_ablation_mask_diagnostics,
        "causality_ablation_tradeoff_diagnostics": causality_ablation_tradeoff_diagnostics,
        "causality_ablation_freeze_failures": causal_ablation_freeze_failures,
        "causality_ablation_missing": missing_causality_ablations,
        "learning_causality_gate_pass": learning_causality_gate_pass,
        "learning_causality_failed_checks": failed_causality_checks,
        "learned_controlled_slot_fraction_min": learned_slot_fraction_min,
        "learning_causality_ablation_status": ablation_status,
        "predictability_gate_pass": predictability_gate_pass,
        "prior_predictive_alignment_gate_pass": prior_predictive_alignment_gate_pass,
        "workload_signature_gate_pass": signature_gate_pass,
        "support_overlap_gate_pass": support_overlap_gate_pass,
    }
    global_sanity_gate = evaluate_global_sanity_gate(
        primary=matched["MLQDS"],
        uniform=uniform_score,
        compression_ratio=float(config.model.compression_ratio),
    )
    global_sanity_gate_pass = bool(global_sanity_gate.get("gate_pass", False))
    global_sanity_required_for_initial_local_learning = False
    blocking_gates: list[str] = []
    if final_candidate:
        if not workload_stability_gate_pass:
            blocking_gates.append("workload_stability_gate")
        if not support_overlap_gate_pass:
            blocking_gates.append("support_overlap_gate")
        if not predictability_gate_pass:
            blocking_gates.append("predictability_gate")
        if not prior_predictive_alignment_gate_pass:
            blocking_gates.append("prior_predictive_alignment_gate")
        if not target_diffusion_gate_pass:
            blocking_gates.append("target_diffusion_gate")
        if not signature_gate_pass:
            blocking_gates.append("workload_signature_gate")
        if not learning_causality_gate_pass:
            blocking_gates.append("learning_causality_ablations")
        if (
            global_sanity_required_for_initial_local_learning
            and not global_sanity_gate_pass
        ):
            blocking_gates.append("global_sanity_gates")
        single_cell_blocking_gates = list(blocking_gates)
        blocking_gates.append("full_workload_profile_compression_grid")
        if single_cell_blocking_gates:
            final_claim_reason = (
                "Strict single-cell evidence is blocked by required gates before the final grid: "
                + ", ".join(single_cell_blocking_gates)
                + "."
            )
        else:
            final_claim_reason = (
                "Strict single-cell gates passed; final success still requires the benchmark-level "
                "full workload-profile/compression grid."
            )
        final_claim_summary = {
            "primary_metric": "QueryLocalUtility",
            "status": "candidate_blocked_by_required_gates"
            if blocking_gates
            else "candidate_ready_for_final_claim",
            "final_success_allowed": not blocking_gates,
            "blocking_gates": blocking_gates,
            "workload_stability_gate_pass": workload_stability_gate_pass,
            "support_overlap_gate_pass": support_overlap_gate_pass,
            "predictability_gate_pass": predictability_gate_pass,
            "prior_predictive_alignment_gate_pass": prior_predictive_alignment_gate_pass,
            "target_diffusion_gate_pass": target_diffusion_gate_pass,
            "workload_signature_gate_pass": signature_gate_pass,
            "learning_causality_gate_pass": learning_causality_gate_pass,
            "global_sanity_gate_pass": global_sanity_gate_pass,
            "global_sanity_gate_required_for_initial_local_learning": (
                global_sanity_required_for_initial_local_learning
            ),
            "global_sanity_gate_role": (
                "diagnostic_guardrail_during_initial_query_local_learning"
            ),
            "mlqds_score": matched["MLQDS"].query_local_utility_score,
            "uniform_score": uniform_score.query_local_utility_score
            if uniform_score is not None
            else None,
            "douglas_peucker_score": (
                douglas_peucker_score.query_local_utility_score
                if douglas_peucker_score is not None
                else None
            ),
            "reason": final_claim_reason,
        }
    else:
        final_claim_summary = {
            "primary_metric": None,
            "status": "not_final_query_driven_candidate",
            "final_success_allowed": False,
            "reason": "Requires range_query_mix, QueryLocalUtility factorized target, workload_blind_range_v2, and learned_segment_budget_v1.",
        }
    learning_causality_summary["final_success_allowed"] = bool(
        final_candidate and not blocking_gates
    )
    diagnostic_summary = {
        "legacy_range_useful_available": True,
        "query_local_utility_available": True,
        "range_component_diagnostics_available": True,
        "workload_blind_protocol_available": True,
        "predictability_audit_available": bool(predictability_audit.get("available", False)),
        "prior_predictive_alignment_gate_available": isinstance(
            prior_predictive_alignment_gate, dict
        ),
        "workload_stability_gate_available": bool(workload_stability_gate),
        "support_overlap_gate_available": bool(support_overlap_gate),
        "global_sanity_gate_available": bool(global_sanity_gate),
        "target_diffusion_gate_available": bool(target_diffusion_gate),
        "workload_signature_gate_available": bool(
            isinstance(workload_signature_gate, dict)
            and workload_signature_gate.get("all_available")
        ),
    }
    return FinalRunSummaries(
        final_candidate=final_candidate,
        final_claim_summary=final_claim_summary,
        diagnostic_summary=diagnostic_summary,
        legacy_range_useful_summary=legacy_range_useful_summary,
        learning_causality_summary=learning_causality_summary,
        support_overlap_gate=support_overlap_gate,
        global_sanity_gate=global_sanity_gate,
        target_diffusion_gate=target_diffusion_gate,
        workload_stability_gate=workload_stability_gate,
    )
