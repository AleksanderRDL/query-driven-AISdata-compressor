"""Query-driven causality and final-summary tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from learning.model_features import (
    WORKLOAD_BLIND_RANGE_MODEL_DISABLED_PRIOR_FIELDS,
    WORKLOAD_BLIND_RANGE_MODEL_PRIOR_TRANSFORM,
    WORKLOAD_BLIND_RANGE_POINT_DIM,
    build_workload_blind_range_point_features,
)
from learning.model_training import (
    _fit_scaler_for_model,
)
from learning.outputs import TrainingOutputs
from learning.query_prior_fields import (
    QUERY_PRIOR_FIELD_NAMES,
    build_train_query_prior_fields,
    sample_query_prior_fields,
    zero_query_prior_field_like,
)
from orchestration.causality import (
    PRIOR_ABLATION_DIAGNOSTIC_CHAIN,
    PRIOR_ABLATION_SCORE_OUTPUT_SEMANTICS,
    build_learned_slot_summary,
    causality_ablation_diagnostics_payload,
    causality_ablation_tradeoff_summary,
    head_ablation_sensitivity,
    head_output_sensitivity,
    learning_causality_delta_gate_config,
    model_prior_feature_sensitivity,
    prior_ablation_sensitivity_from_tensors,
    prior_ablation_sensitivity_payload,
    prior_feature_sample_sensitivity,
    prior_sample_gate_failures,
    query_local_utility_component_delta_summary,
    retained_mask_comparison,
    score_ablation_sensitivity,
    training_outputs_with_query_prior_field,
)
from orchestration.final_gate_summary import build_final_run_summaries
from orchestration.selection_causality_diagnostics import build_selection_causality_diagnostics
from scoring.metrics import MethodScore

HISTORICAL_SMALL_LOCAL_FAMILY = "small_local"


def _boundaries(trajectories: list[torch.Tensor]) -> list[tuple[int, int]]:
    cursor = 0
    out = []
    for trajectory in trajectories:
        end = cursor + int(trajectory.shape[0])
        out.append((cursor, end))
        cursor = end
    return out

def test_learning_causality_summary_reports_learned_slot_budget_without_ablation_claims() -> None:
    selector_diagnostics = {
        "eval": {
            "budget_rows": [
                {
                    "compression_ratio": 0.10,
                    "total_budget_count": 20,
                    "minimal_skeleton_slot_cap": 4,
                    "learned_slot_count": 16,
                    "learned_slot_fraction_of_budget": 0.80,
                    "no_fixed_85_percent_temporal_scaffold": True,
                }
            ]
        }
    }

    summary = build_learned_slot_summary(selector_diagnostics, 0.10)

    assert summary["learned_controlled_retained_slots"] == 16
    assert summary["learned_controlled_retained_slot_fraction"] == 0.80
    assert summary["learned_slot_accounting_status"] == "budget_level_accounting_only"


def test_learning_causality_summary_prefers_point_attribution_when_available() -> None:
    selector_diagnostics = {
        "eval": {
            "budget_rows": [
                {
                    "compression_ratio": 0.10,
                    "total_budget_count": 20,
                    "minimal_skeleton_slot_cap": 4,
                    "learned_slot_count": 16,
                    "learned_slot_fraction_of_budget": 0.80,
                    "no_fixed_85_percent_temporal_scaffold": True,
                }
            ]
        }
    }
    trace = {
        "point_attribution_available": True,
        "learned_controlled_retained_slots": 12,
        "learned_controlled_retained_slot_fraction": 0.60,
        "skeleton_retained_count": 4,
        "fallback_retained_count": 4,
        "unattributed_retained_count": 0,
        "trajectories_with_at_least_one_learned_decision": 3,
        "trajectories_with_zero_learned_decisions": 1,
        "segment_budget_entropy": 1.2,
        "segment_budget_entropy_normalized": 0.8,
        "segments_with_learned_budget": 5,
        "retained_mask_matches_frozen_primary": True,
    }

    summary = build_learned_slot_summary(selector_diagnostics, 0.10, trace)

    assert summary["learned_controlled_retained_slots"] == 12
    assert summary["planned_learned_controlled_retained_slots"] == 16
    assert summary["actual_learned_controlled_retained_slot_fraction"] == 0.60
    assert summary["trajectories_with_at_least_one_learned_decision"] == 3
    assert summary["selector_trace_retained_mask_matches_primary"] is True
    assert summary["learned_slot_accounting_status"] == "point_attribution_available"


def test_selection_causality_diagnostics_reports_unavailable_preconditions() -> None:
    missing_split = build_selection_causality_diagnostics(
        trained=cast(Any, object()),
        selection_points=None,
        selection_boundaries=None,
        selection_workload=None,
        eval_workload_map={"range": 1.0},
        selection_query_cache=None,
        config=cast(Any, SimpleNamespace(model=SimpleNamespace(selector_type="temporal_hybrid"))),
        seeds=SimpleNamespace(eval_query_seed=1),
    )

    wrong_selector = build_selection_causality_diagnostics(
        trained=cast(Any, object()),
        selection_points=torch.zeros((2, 8), dtype=torch.float32),
        selection_boundaries=[(0, 2)],
        selection_workload=SimpleNamespace(typed_queries=[]),
        eval_workload_map={"range": 1.0},
        selection_query_cache=None,
        config=cast(Any, SimpleNamespace(model=SimpleNamespace(selector_type="temporal_hybrid"))),
        seeds=SimpleNamespace(eval_query_seed=1),
    )

    assert missing_split == {"available": False, "reason": "missing_selection_split"}
    assert wrong_selector == {
        "available": False,
        "reason": "requires_learned_segment_budget",
    }


def _final_summary_config(
    *,
    final_candidate: bool = True,
    range_training_target_mode: str = "query_local_utility_factorized",
) -> SimpleNamespace:
    return SimpleNamespace(
        query=SimpleNamespace(
            workload_profile_id=(
                "range_query_mix_local" if final_candidate else "legacy_generator"
            ),
            target_coverage=0.10,
            range_max_coverage_overshoot=0.0075,
            workload_stability_gate_mode="final",
        ),
        model=SimpleNamespace(
            model_type="workload_blind_range",
            range_training_target_mode=range_training_target_mode,
            selector_type="learned_segment_budget",
            compression_ratio=0.10,
            learned_segment_geometry_gain_weight=0.0,
            learned_segment_score_blend_weight=0.05,
            learned_segment_fairness_preallocation=True,
            learned_segment_length_repair_fraction=0.6,
            learned_segment_length_repair_score_protection_fraction=0.0,
            learned_segment_length_support_blend_weight=0.0,
        ),
    )


def _final_summary_workload() -> SimpleNamespace:
    return SimpleNamespace(
        typed_queries=[{"type": "range", "params": {}} for _idx in range(8)],
        coverage_fraction=0.10,
        generation_diagnostics={
            "query_generation": {
                "workload_profile_id": "range_query_mix_local",
                "mode": "target_coverage",
                "coverage_calibration_mode": "profile_sampled_query_count",
                "query_count_mode": "calibrated_to_coverage",
                "target_coverage": 0.10,
                "coverage_guard_enabled": True,
                "stop_reason": "target_coverage_reached",
            },
            "range_acceptance": {
                "accepted": 8,
                "attempts": 8,
                "exhausted": False,
                "rejected": 0,
                "rejection_reasons": {},
            },
        },
    )


def _final_summary_metrics(score: float, *, range_score: float = 0.2) -> MethodScore:
    return MethodScore(
        aggregate_f1=0.0,
        per_type_f1={},
        query_local_utility_score=score,
        range_usefulness_score=range_score,
        avg_length_preserved=0.9,
        geometric_distortion={"avg_sed_km": 1.0},
        range_audit={"endpoint_sanity": 1.0},
    )


def test_final_run_summaries_block_final_grid_until_benchmark_evidence() -> None:
    workloads = [_final_summary_workload() for _idx in range(4)]
    matched = {
        "MLQDS": _final_summary_metrics(0.30),
        "uniform": _final_summary_metrics(0.25),
        "DouglasPeucker": _final_summary_metrics(0.20),
    }

    summaries = build_final_run_summaries(
        config=cast(Any, _final_summary_config(final_candidate=True)),
        trained=cast(
            Any,
            SimpleNamespace(
                fit_diagnostics={},
                target_diagnostics={},
                feature_context={},
            ),
        ),
        train_points=torch.zeros((3, 8), dtype=torch.float32),
        test_points=torch.zeros((3, 8), dtype=torch.float32),
        train_label_workloads=workloads,
        eval_workload=_final_summary_workload(),
        selection_workload=_final_summary_workload(),
        matched=matched,
        selector_budget_diagnostics={},
        primary_selector_trace=None,
        causality_ablation_scores={},
        causality_ablation_mask_diagnostics={},
        causal_ablation_freeze_failures={},
        prior_sensitivity_diagnostics={},
        prior_channel_ablation_diagnostics={},
        head_ablation_sensitivity_diagnostics={},
        selection_causality_diagnostics={"available": False, "reason": "not_run"},
        segment_oracle_allocation_audit={},
        target_segment_oracle_alignment_audit={},
        segment_budget_head_ablation_mode="neutral_constant_segment_scores",
        predictability_audit={
            "available": True,
            "gate_pass": True,
            "prior_predictive_alignment_gate": {"gate_pass": True},
        },
        workload_distribution_comparison={
            "workload_signature_gate": {"all_available": True, "all_pass": True}
        },
    )

    assert summaries.final_candidate is True
    assert summaries.final_claim_summary["primary_metric"] == "QueryLocalUtility"
    assert summaries.final_claim_summary["final_success_allowed"] is False
    assert (
        summaries.final_claim_summary["global_sanity_gate_required_for_initial_local_learning"]
        is False
    )
    assert (
        "full_workload_profile_compression_grid" in summaries.final_claim_summary["blocking_gates"]
    )
    assert summaries.learning_causality_summary["final_success_allowed"] is False
    assert summaries.diagnostic_summary["workload_stability_gate_available"] is True


def test_final_run_summaries_report_global_sanity_without_initial_blocking() -> None:
    workloads = [_final_summary_workload() for _idx in range(4)]
    primary = _final_summary_metrics(0.30)
    primary.range_audit["endpoint_sanity"] = 0.0
    matched = {
        "MLQDS": primary,
        "uniform": _final_summary_metrics(0.25),
        "DouglasPeucker": _final_summary_metrics(0.20),
    }

    summaries = build_final_run_summaries(
        config=cast(Any, _final_summary_config(final_candidate=True)),
        trained=cast(
            Any,
            SimpleNamespace(
                fit_diagnostics={},
                target_diagnostics={},
                feature_context={},
            ),
        ),
        train_points=torch.zeros((3, 8), dtype=torch.float32),
        test_points=torch.zeros((3, 8), dtype=torch.float32),
        train_label_workloads=workloads,
        eval_workload=_final_summary_workload(),
        selection_workload=_final_summary_workload(),
        matched=matched,
        selector_budget_diagnostics={},
        primary_selector_trace=None,
        causality_ablation_scores={},
        causality_ablation_mask_diagnostics={},
        causal_ablation_freeze_failures={},
        prior_sensitivity_diagnostics={},
        prior_channel_ablation_diagnostics={},
        head_ablation_sensitivity_diagnostics={},
        selection_causality_diagnostics={"available": False, "reason": "not_run"},
        segment_oracle_allocation_audit={},
        target_segment_oracle_alignment_audit={},
        segment_budget_head_ablation_mode="neutral_constant_segment_scores",
        predictability_audit={
            "available": True,
            "gate_pass": True,
            "prior_predictive_alignment_gate": {"gate_pass": True},
        },
        workload_distribution_comparison={
            "workload_signature_gate": {"all_available": True, "all_pass": True}
        },
    )

    assert summaries.global_sanity_gate["gate_pass"] is False
    assert summaries.final_claim_summary["global_sanity_gate_pass"] is False
    assert (
        summaries.final_claim_summary["global_sanity_gate_role"]
        == "diagnostic_guardrail_during_initial_query_local_learning"
    )
    assert "global_sanity_gates" not in summaries.final_claim_summary["blocking_gates"]


def test_learning_causality_summary_does_not_duplicate_retained_marginal_trace() -> None:
    workloads = [_final_summary_workload() for _idx in range(4)]
    trace = {
        "retained_decision_marginal_query_local_utility_alignment": {
            "available": True,
            "diagnostic_only": True,
            "exact_query_local_utility_marginals": True,
            "performance_mode": "exact_cached_query_support",
            "candidate_count": 2,
            "score_fields_available": {"selector_score": True},
            "score_component_fields_available": {"factorized_composed_score": True},
            "context_fields_available": {"trajectory_index": True},
            "query_free_teacher_proxy_guard_coupling_summary": {
                "available": True,
                "diagnostic_only": True,
            },
            "learned_controllable_marginal_teacher_summary": {
                "available": True,
                "candidate_count": 1,
                "eval_time_feature_allowed": False,
            },
            "separated_marginal_teacher_summary": {
                "available": True,
                "teacher_usage_split": "eval_primary",
                "teacher_usage_allowed_for_train_or_checkpoint": False,
                "teacher_target_shape_viable": True,
                "candidate_for_train_side_teacher": False,
                "candidate_for_train_side_teacher_reason": (
                    "eval_split_query_conditioned_teacher_not_allowed_for_training"
                ),
                "segment_target_count": 1,
                "point_target_count": 1,
                "segment_target_rows": [{"segment_index": 7}],
                "point_target_rows": [{"point_index": 1}],
            },
            "top_marginal_miss_summary": {
                "available": True,
                "top_marginal_rows": [{"point_index": 1}],
            },
            "overall": {"candidate_count": 2},
            "by_source": {"learned": {"candidate_count": 1}},
            "by_decision": {"retained_removal_loss": {"candidate_count": 1}},
            "rows": [{"point_index": 1}],
        }
    }

    summaries = build_final_run_summaries(
        config=cast(Any, _final_summary_config(final_candidate=True)),
        trained=cast(
            Any,
            SimpleNamespace(
                fit_diagnostics={},
                target_diagnostics={},
                feature_context={},
            ),
        ),
        train_points=torch.zeros((3, 8), dtype=torch.float32),
        test_points=torch.zeros((3, 8), dtype=torch.float32),
        train_label_workloads=workloads,
        eval_workload=_final_summary_workload(),
        selection_workload=_final_summary_workload(),
        matched={
            "MLQDS": _final_summary_metrics(0.30),
            "uniform": _final_summary_metrics(0.25),
            "DouglasPeucker": _final_summary_metrics(0.20),
        },
        selector_budget_diagnostics={},
        primary_selector_trace=trace,
        causality_ablation_scores={},
        causality_ablation_mask_diagnostics={},
        causal_ablation_freeze_failures={},
        prior_sensitivity_diagnostics={},
        prior_channel_ablation_diagnostics={},
        head_ablation_sensitivity_diagnostics={},
        selection_causality_diagnostics={"available": True},
        segment_oracle_allocation_audit={},
        target_segment_oracle_alignment_audit={},
        segment_budget_head_ablation_mode="neutral_constant_segment_scores",
        predictability_audit={
            "available": True,
            "gate_pass": True,
            "prior_predictive_alignment_gate": {"gate_pass": True},
        },
        workload_distribution_comparison={
            "workload_signature_gate": {"all_available": True, "all_pass": True}
        },
    )

    selection = summaries.learning_causality_summary["selection_causality_diagnostics"]
    retained = trace["retained_decision_marginal_query_local_utility_alignment"]
    assert selection == {"available": True}
    assert "retained_decision_marginal_alignment" not in selection
    assert retained["available"] is True
    assert retained["candidate_count"] == 2
    assert retained["score_component_fields_available"] == {
        "factorized_composed_score": True
    }
    assert retained["rows"] == [{"point_index": 1}]


def test_final_run_summaries_reject_non_final_candidate_profile() -> None:
    workloads = [_final_summary_workload() for _idx in range(4)]

    summaries = build_final_run_summaries(
        config=cast(Any, _final_summary_config(final_candidate=False)),
        trained=cast(
            Any,
            SimpleNamespace(
                fit_diagnostics={},
                target_diagnostics={},
                feature_context={},
            ),
        ),
        train_points=torch.zeros((3, 8), dtype=torch.float32),
        test_points=torch.zeros((3, 8), dtype=torch.float32),
        train_label_workloads=workloads,
        eval_workload=_final_summary_workload(),
        selection_workload=None,
        matched={
            "MLQDS": _final_summary_metrics(0.30),
            "uniform": _final_summary_metrics(0.25),
        },
        selector_budget_diagnostics={},
        primary_selector_trace=None,
        causality_ablation_scores={},
        causality_ablation_mask_diagnostics={},
        causal_ablation_freeze_failures={},
        prior_sensitivity_diagnostics={},
        prior_channel_ablation_diagnostics={},
        head_ablation_sensitivity_diagnostics={},
        selection_causality_diagnostics={"available": False, "reason": "not_run"},
        segment_oracle_allocation_audit={},
        target_segment_oracle_alignment_audit={},
        segment_budget_head_ablation_mode="neutral_constant_segment_scores",
        predictability_audit={},
        workload_distribution_comparison={},
    )

    assert summaries.final_candidate is False
    assert summaries.final_claim_summary == {
        "primary_metric": None,
        "status": "not_final_query_driven_candidate",
        "final_success_allowed": False,
        "reason": (
            "Requires range_query_mix, QueryLocalUtility factorized target, "
            "workload_blind_range, and learned_segment_budget."
        ),
    }


def test_learning_causality_delta_gate_requires_material_ablation_loss() -> None:
    primary = MethodScore(
        aggregate_f1=0.0,
        per_type_f1={},
        query_local_utility_score=0.30,
    )
    uniform = MethodScore(
        aggregate_f1=0.0,
        per_type_f1={},
        query_local_utility_score=0.25,
    )

    gate = learning_causality_delta_gate_config(primary=primary, uniform=uniform)
    thresholds = gate["thresholds"]

    assert gate["min_material_query_local_utility_delta"] == 0.005
    assert abs(gate["mlqds_uniform_query_local_utility_gap"] - 0.05) < 1e-12
    assert abs(thresholds["shuffled_scores_should_lose"] - 0.03) < 1e-12
    assert thresholds["without_segment_budget_head_should_lose"] == 0.005
    assert thresholds["prior_field_only_should_not_match_trained"] == 0.005


def test_query_local_utility_component_delta_summary_reports_weighted_tradeoffs() -> None:
    primary = MethodScore(
        aggregate_f1=0.0,
        per_type_f1={},
        query_local_utility_score=0.368,
        query_local_utility_components={
            "query_point_recall": 0.60,
            "query_local_turn_change_coverage": 0.40,
            "length_preservation_guardrail": 0.80,
        },
    )
    ablation = MethodScore(
        aggregate_f1=0.0,
        per_type_f1={},
        query_local_utility_score=0.332,
        query_local_utility_components={
            "query_point_recall": 0.50,
            "query_local_turn_change_coverage": 0.50,
            "length_preservation_guardrail": 0.70,
        },
    )

    summary = query_local_utility_component_delta_summary(
        primary=primary,
        ablations={"MLQDS_without_behavior_utility_head": ablation},
        top_k=2,
    )
    row = summary["MLQDS_without_behavior_utility_head"]

    assert row["available"] is True
    assert row["query_local_utility_delta"] == pytest.approx(0.036)
    assert row["component_deltas"]["query_point_recall"] == pytest.approx(0.10)
    assert row["weighted_component_deltas"]["query_point_recall"] == pytest.approx(
        0.05
    )
    assert row["weighted_component_deltas"]["query_local_turn_change_coverage"] == pytest.approx(
        -0.015
    )
    assert row["component_weighted_delta_sum"] == pytest.approx(0.036)
    assert row["component_delta_residual"] == pytest.approx(0.0)
    assert (
        row["top_positive_weighted_component_deltas"][0]["component"]
        == "query_point_recall"
    )
    assert (
        row["top_negative_weighted_component_deltas"][0]["component"]
        == "query_local_turn_change_coverage"
    )


def test_causality_ablation_tradeoff_summary_connects_mask_and_component_changes() -> None:
    primary = MethodScore(
        aggregate_f1=0.0,
        per_type_f1={},
        query_local_utility_score=0.368,
        query_local_utility_components={
            "query_point_recall": 0.60,
            "query_local_turn_change_coverage": 0.40,
            "length_preservation_guardrail": 0.80,
        },
    )
    ablation = MethodScore(
        aggregate_f1=0.0,
        per_type_f1={},
        query_local_utility_score=0.332,
        query_local_utility_components={
            "query_point_recall": 0.50,
            "query_local_turn_change_coverage": 0.50,
            "length_preservation_guardrail": 0.70,
        },
    )
    primary_mask = torch.tensor([True, True, False, False])
    ablation_mask = torch.tensor([False, True, True, False])

    component_deltas = query_local_utility_component_delta_summary(
        primary=primary,
        ablations={"MLQDS_without_behavior_utility_head": ablation},
        top_k=2,
    )
    mask_diagnostics = {
        "MLQDS_without_behavior_utility_head": retained_mask_comparison(
            primary_mask=primary_mask,
            ablation_mask=ablation_mask,
            expected_shape=primary_mask.shape,
        )
    }

    summary = causality_ablation_tradeoff_summary(
        component_deltas=component_deltas,
        mask_diagnostics=mask_diagnostics,
    )
    row = summary["MLQDS_without_behavior_utility_head"]

    assert row["available"] is True
    assert row["tradeoff_status"] == "mask_change_helped_primary_metric"
    assert row["retained_symmetric_difference_count"] == 2.0
    assert row["query_local_utility_delta_per_changed_retained_decision"] == pytest.approx(0.018)
    assert row["positive_weighted_component_delta_sum"] == pytest.approx(0.051)
    assert row["negative_weighted_component_delta_sum"] == pytest.approx(-0.015)
    assert (
        row["dominant_positive_weighted_component_delta"]["component"]
        == "query_point_recall"
    )
    assert row["dominant_negative_weighted_component_delta"]["component"] == (
        "query_local_turn_change_coverage"
    )


def test_causality_ablation_diagnostics_payload_reuses_component_and_mask_tradeoffs() -> None:
    primary = MethodScore(
        aggregate_f1=0.0,
        per_type_f1={},
        query_local_utility_score=0.368,
        query_local_utility_components={
            "query_point_recall": 0.60,
            "query_local_turn_change_coverage": 0.40,
        },
    )
    ablation = MethodScore(
        aggregate_f1=0.0,
        per_type_f1={},
        query_local_utility_score=0.332,
        query_local_utility_components={
            "query_point_recall": 0.50,
            "query_local_turn_change_coverage": 0.50,
        },
    )
    mask_diagnostics = {
        "MLQDS_without_behavior_utility_head": {
            "available": True,
            "retained_symmetric_difference_count": 4,
            "retained_mask_changed": True,
            "retained_mask_jaccard": 0.5,
            "retained_mask_hamming_fraction": 0.25,
        }
    }

    payload = causality_ablation_diagnostics_payload(
        primary=primary,
        ablations={"MLQDS_without_behavior_utility_head": ablation},
        mask_diagnostics=mask_diagnostics,
    )
    row = payload["tradeoff_diagnostics"]["MLQDS_without_behavior_utility_head"]

    assert payload["available"] is True
    assert payload["primary_query_local_utility_score"] == pytest.approx(0.368)
    assert payload["ablation_scores"]["MLQDS_without_behavior_utility_head"] == pytest.approx(0.332)
    assert payload["ablation_query_local_utility_deltas"][
        "MLQDS_without_behavior_utility_head"
    ] == pytest.approx(0.036)
    assert row["retained_symmetric_difference_count"] == 4.0
    assert row["dominant_negative_weighted_component_delta"]["component"] == (
        "query_local_turn_change_coverage"
    )


def test_score_ablation_sensitivity_reports_score_and_mask_changes() -> None:
    primary_scores = torch.tensor([0.9, 0.8, 0.1, 0.0], dtype=torch.float32)
    ablation_scores = torch.tensor([0.1, 0.8, 0.9, 0.0], dtype=torch.float32)
    primary_mask = torch.tensor([True, True, False, False])
    ablation_mask = torch.tensor([False, True, True, False])

    diagnostics = score_ablation_sensitivity(
        primary_scores=primary_scores,
        ablation_scores=ablation_scores,
        primary_mask=primary_mask,
        ablation_mask=ablation_mask,
    )

    assert diagnostics["available"] is True
    assert diagnostics["mean_abs_score_delta"] > 0.0
    assert diagnostics["retained_mask_changed"] is True
    assert diagnostics["retained_mask_jaccard"] == 1.0 / 3.0
    assert diagnostics["score_topk_jaccard_at_retained_count"] == 1.0 / 3.0


def test_prior_ablation_sensitivity_payload_exposes_score_output_chain() -> None:
    score_output = {"available": True, "mean_abs_score_delta": 0.25}

    payload = prior_ablation_sensitivity_payload(
        sampled_prior_features={"available": True, "mean_abs_feature_delta": 0.75},
        model_prior_features={"available": True, "mean_abs_feature_delta": 0.5},
        score_output=score_output,
        retained_mask={"available": True, "retained_mask_changed": True},
        raw_prediction={"available": True, "mean_abs_score_delta": 0.4},
        head_output={"available": True, "mean_abs_head_probability_delta": 0.01},
    )

    assert payload["available"] is True
    assert payload["diagnostic_chain"] == list(PRIOR_ABLATION_DIAGNOSTIC_CHAIN)
    assert "selector_score" not in payload
    assert payload["score_output"]["mean_abs_score_delta"] == pytest.approx(0.25)
    assert payload["score_output"]["semantics"] == PRIOR_ABLATION_SCORE_OUTPUT_SEMANTICS
    assert payload["retained_mask"]["retained_mask_changed"] is True


def test_prior_ablation_sensitivity_from_tensors_builds_consistent_chain() -> None:
    primary_scores = torch.tensor([0.9, 0.8, 0.1, 0.0], dtype=torch.float32)
    ablation_scores = torch.tensor([0.1, 0.8, 0.9, 0.0], dtype=torch.float32)
    primary_raw = torch.tensor([3.0, 2.0, 1.0, 0.0], dtype=torch.float32)
    ablation_raw = torch.tensor([2.0, 2.0, 2.0, 0.0], dtype=torch.float32)
    primary_heads = torch.tensor([[2.0, -1.0, 0.0, 0.5, 1.0, -0.5]], dtype=torch.float32)
    ablation_heads = primary_heads + 0.1
    primary_mask = torch.tensor([True, True, False, False])
    ablation_mask = torch.tensor([False, True, True, False])

    payload = prior_ablation_sensitivity_from_tensors(
        sampled_prior_features={"available": True},
        model_prior_features={"available": True},
        primary_scores=primary_scores,
        ablation_scores=ablation_scores,
        primary_raw_predictions=primary_raw,
        ablation_raw_predictions=ablation_raw,
        primary_head_logits=primary_heads,
        ablation_head_logits=ablation_heads,
        primary_mask=primary_mask,
        ablation_mask=ablation_mask,
    )

    assert "selector_score" not in payload
    assert payload["retained_mask"]["retained_mask_changed"] is True
    assert payload["retained_mask"]["retained_mask_jaccard"] == pytest.approx(1.0 / 3.0)
    assert payload["score_output"]["retained_mask_changed"] is True
    assert payload["score_output"]["retained_mask_jaccard"] == pytest.approx(1.0 / 3.0)
    assert payload["raw_prediction"]["mean_abs_score_delta"] > 0.0
    assert payload["head_output"]["head_probabilities_changed"] is True


def test_training_outputs_with_query_prior_field_keeps_metadata_aligned() -> None:
    base_prior = {
        "schema_version": 3,
        "field_names": ["spatial_query_hit_probability"],
        "contains_eval_queries": False,
    }
    ablation_prior = {
        **base_prior,
        "ablation": "zero_query_prior_features",
        "diagnostics": {"zeroed_prior_features_preserve_train_extent": True},
    }
    trained = TrainingOutputs(
        model=torch.nn.Linear(1, 1),
        scaler=cast(Any, object()),
        labels=torch.ones(1),
        labelled_mask=torch.ones(1, dtype=torch.bool),
        history=[{"loss": 1.0}],
        epochs_trained=3,
        feature_context={
            "query_prior_field": base_prior,
            "query_prior_field_metadata": {"stale": True},
            "other": "kept",
        },
    )

    updated = training_outputs_with_query_prior_field(trained, ablation_prior)

    assert updated is not trained
    assert updated.model is trained.model
    assert updated.history is trained.history
    assert updated.feature_context["other"] == "kept"
    assert updated.feature_context["query_prior_field"] is ablation_prior
    assert updated.feature_context["query_prior_field_metadata"]["ablation"] == (
        "zero_query_prior_features"
    )
    assert "stale" not in updated.feature_context["query_prior_field_metadata"]


def test_head_ablation_sensitivity_reports_selector_raw_and_segment_channels() -> None:
    primary_scores = torch.tensor([0.9, 0.8, 0.1, 0.0], dtype=torch.float32)
    ablation_scores = torch.tensor([0.1, 0.8, 0.9, 0.0], dtype=torch.float32)
    primary_raw_predictions = torch.tensor([3.0, 2.0, 1.0, 0.0], dtype=torch.float32)
    ablation_raw_predictions = torch.tensor([2.0, 2.0, 2.0, 0.0], dtype=torch.float32)
    primary_segment_scores = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    ablation_segment_scores = torch.zeros(4, dtype=torch.float32)
    primary_mask = torch.tensor([True, True, False, False])
    ablation_mask = torch.tensor([False, True, True, False])

    diagnostics = head_ablation_sensitivity(
        primary_scores=primary_scores,
        ablation_scores=ablation_scores,
        primary_raw_predictions=primary_raw_predictions,
        ablation_raw_predictions=ablation_raw_predictions,
        primary_segment_scores=primary_segment_scores,
        ablation_segment_scores=ablation_segment_scores,
        primary_mask=primary_mask,
        ablation_mask=ablation_mask,
    )

    assert diagnostics["selector_score"]["available"] is True
    assert diagnostics["selector_score"]["retained_mask_changed"] is True
    assert diagnostics["raw_prediction"]["mean_abs_score_delta"] > 0.0
    assert diagnostics["segment_score"]["mean_abs_score_delta"] > 0.0


def test_head_output_sensitivity_reports_per_head_logit_and_probability_deltas() -> None:
    primary_head_logits = torch.tensor(
        [
            [2.0, -1.0, 0.0, 0.5, 1.0, -0.5],
            [1.5, -0.5, 0.2, 0.3, 0.8, -0.2],
        ],
        dtype=torch.float32,
    )
    ablation_head_logits = primary_head_logits.clone()
    ablation_head_logits[:, 0] -= 0.4
    ablation_head_logits[:, 4] += 0.2

    diagnostics = head_output_sensitivity(
        primary_head_logits=primary_head_logits,
        ablation_head_logits=ablation_head_logits,
    )

    assert diagnostics["available"] is True
    assert diagnostics["head_logits_changed"] is True
    assert diagnostics["head_probabilities_changed"] is True
    assert diagnostics["mean_abs_head_logit_delta"] > 0.0
    assert diagnostics["mean_abs_head_probability_delta"] > 0.0
    assert diagnostics["per_head"]["query_hit_probability"][
        "mean_abs_logit_delta"
    ] == pytest.approx(0.4)
    assert diagnostics["per_head"]["segment_budget_target"][
        "mean_abs_logit_delta"
    ] == pytest.approx(0.2)
    assert diagnostics["per_head"]["conditional_behavior_utility"][
        "mean_abs_logit_delta"
    ] == pytest.approx(0.0)


def test_retained_mask_comparison_reports_ablation_overlap() -> None:
    primary_mask = torch.tensor([True, True, False, False])
    ablation_mask = torch.tensor([False, True, True, False])

    diagnostics = retained_mask_comparison(
        primary_mask=primary_mask,
        ablation_mask=ablation_mask,
        expected_shape=primary_mask.shape,
    )

    assert diagnostics["available"] is True
    assert diagnostics["primary_retained_count"] == 2
    assert diagnostics["ablation_retained_count"] == 2
    assert diagnostics["retained_intersection_count"] == 1
    assert diagnostics["retained_union_count"] == 3
    assert diagnostics["retained_symmetric_difference_count"] == 2
    assert diagnostics["retained_mask_changed"] is True
    assert diagnostics["retained_mask_jaccard"] == 1.0 / 3.0
    assert diagnostics["retained_mask_hamming_fraction"] == 0.5


def test_prior_feature_sample_sensitivity_reports_input_level_changes() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=torch.float32,
    )
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 3.0,
            "lat_min": -1.0,
            "lat_max": 3.0,
            "lon_min": -1.0,
            "lon_max": 3.0,
        },
    }
    prior = build_train_query_prior_fields(
        points=points,
        boundaries=[(0, 3)],
        typed_queries=[query],
        workload_profile_id="range_query_mix",
        grid_bins=4,
        time_bins=2,
        smoothing_passes=0,
    )

    diagnostics = prior_feature_sample_sensitivity(
        points=points,
        primary_prior_field=prior,
        ablation_prior_field=None,
    )

    assert diagnostics["available"] is True
    assert diagnostics["point_count"] == 3
    assert diagnostics["feature_count"] == 6
    assert diagnostics["sampled_inputs_changed"] is True
    assert diagnostics["mean_abs_feature_delta"] > 0.0
    assert diagnostics["ablation_nonzero_fraction"] == 0.0
    assert diagnostics["per_feature"]["spatial_query_hit_probability"]["mean_abs_delta"] > 0.0


def test_model_prior_feature_sensitivity_reports_post_builder_and_scaler_changes() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 5.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 1.0, 10.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 3.0,
            "lat_min": -1.0,
            "lat_max": 3.0,
            "lon_min": -1.0,
            "lon_max": 3.0,
        },
    }
    prior = build_train_query_prior_fields(
        points=points,
        boundaries=[(0, 3)],
        typed_queries=[query],
        workload_profile_id="range_query_mix",
        grid_bins=4,
        time_bins=2,
        smoothing_passes=0,
    )
    zeroed = zero_query_prior_field_like(prior)
    queries = torch.zeros((1, 12), dtype=torch.float32)
    model_points = build_workload_blind_range_point_features(points, prior)
    scaler = _fit_scaler_for_model(model_points, queries, "workload_blind_range")

    raw_sampled = sample_query_prior_fields(points, prior)
    route_density_idx = QUERY_PRIOR_FIELD_NAMES.index("route_density_prior")
    assert raw_sampled[:, route_density_idx].abs().mean().item() > 0.0

    diagnostics = model_prior_feature_sensitivity(
        points=points,
        point_dim=WORKLOAD_BLIND_RANGE_POINT_DIM,
        scaler=scaler,
        primary_prior_field=prior,
        ablation_prior_field=zeroed,
        boundaries=[(0, 3)],
    )

    assert diagnostics["available"] is True
    assert diagnostics["disabled_prior_fields"] == list(
        WORKLOAD_BLIND_RANGE_MODEL_DISABLED_PRIOR_FIELDS
    )
    assert (
        diagnostics["model_prior_feature_transform"]
        == WORKLOAD_BLIND_RANGE_MODEL_PRIOR_TRANSFORM
    )
    model_input = diagnostics["model_input_prior_features"]
    normalized = diagnostics["normalized_model_prior_features"]
    assert model_input["sampled_inputs_changed"] is True
    assert normalized["sampled_inputs_changed"] is True
    assert model_input["per_feature"]["spatial_query_hit_probability"]["mean_abs_delta"] > 0.0
    assert normalized["per_feature"]["spatial_query_hit_probability"]["mean_abs_delta"] > 0.0
    assert model_input["per_feature"]["route_density_prior"]["mean_abs_delta"] == 0.0
    assert normalized["per_feature"]["route_density_prior"]["mean_abs_delta"] == 0.0
    assert diagnostics["scaler_prior_feature_ranges"]["route_density_prior"] == 1.0


def test_prior_sample_gate_failures_explain_empty_or_out_of_extent_priors() -> None:
    diagnostics = {
        "shuffled_prior_fields": {
            "sampled_prior_features": {
                "available": True,
                "primary_nonzero_fraction": 0.0,
                "sampled_inputs_changed": False,
                "points_outside_prior_extent_fraction": 1.0,
            },
            "model_prior_features": {
                "model_input_prior_features": {
                    "available": True,
                    "sampled_inputs_changed": False,
                },
                "normalized_model_prior_features": {
                    "available": True,
                    "sampled_inputs_changed": False,
                },
            },
        }
    }

    failures = prior_sample_gate_failures(diagnostics)

    assert "sampled_query_prior_features_all_zero" in failures
    assert "shuffled_prior_fields_did_not_change_sampled_inputs" in failures
    assert "shuffled_prior_fields_did_not_change_model_inputs" in failures
    assert "shuffled_prior_fields_did_not_change_normalized_model_inputs" in failures
    assert "eval_points_mostly_outside_query_prior_extent" in failures
