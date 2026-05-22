"""Query-driven causality and final-summary tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from orchestration.causality import (
    build_learned_slot_summary,
    causality_ablation_diagnostics_payload,
    causality_ablation_tradeoff_summary,
    learning_causality_delta_gate_config,
    query_local_utility_component_delta_summary,
    retained_mask_comparison,
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


def _final_summary_metrics(score: float) -> MethodScore:
    return MethodScore(
        aggregate_f1=0.0,
        per_type_f1={},
        query_local_utility_score=score,
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
    assert retained["score_component_fields_available"] == {"factorized_composed_score": True}
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
    assert row["weighted_component_deltas"]["query_point_recall"] == pytest.approx(0.05)
    assert row["weighted_component_deltas"]["query_local_turn_change_coverage"] == pytest.approx(
        -0.015
    )
    assert row["component_weighted_delta_sum"] == pytest.approx(0.036)
    assert row["component_delta_residual"] == pytest.approx(0.0)
    assert row["top_positive_weighted_component_deltas"][0]["component"] == "query_point_recall"
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
    assert row["dominant_positive_weighted_component_delta"]["component"] == "query_point_recall"
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
