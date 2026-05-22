"""Focused tests for single-run artifact payload assembly."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from config.run_config import build_run_config
from learning.outputs import TrainingOutputs
from orchestration.final_gate_summary import FinalRunSummaries
from orchestration.run_payload import build_run_payload
from orchestration.scoring_stage import build_workload_scoring_compatibility_diagnostics
from scoring.metrics import MethodScore


def _workload(query_count: int, coverage: float) -> SimpleNamespace:
    return SimpleNamespace(
        typed_queries=[{"type": "range", "params": {}} for _ in range(query_count)],
        coverage_fraction=coverage,
        generation_diagnostics={"accepted": query_count},
    )


def _family_component_summary(score: float) -> dict[str, Any]:
    group = {
        "query_count": 1,
        "range_components": {"range_point_f1": score},
        "range_usefulness_score": score,
        "query_local_utility_query_local_weighted_score_normalized": score,
        "ship_evidence_counts": {
            "full_trajectory_hit_count_total": 10,
            "retained_trajectory_hit_count_total": int(score * 10),
            "missed_trajectory_hit_count_total": 10 - int(score * 10),
            "ship_presence_recall": score,
        },
    }
    return {
        "available": True,
        "group_by": {
            "anchor_family": {"density": group},
            "footprint_family": {},
            "anchor_footprint_family": {},
        },
    }


def test_workload_scoring_compatibility_diagnostics_compares_family_components() -> None:
    diagnostics = build_workload_scoring_compatibility_diagnostics(
        {
            "MLQDS": MethodScore(
                aggregate_f1=0.0,
                per_type_f1={},
                range_audit={
                    "range_query_metadata_component_summary": _family_component_summary(0.7)
                },
            ),
            "uniform": MethodScore(
                aggregate_f1=0.0,
                per_type_f1={},
                range_audit={
                    "range_query_metadata_component_summary": _family_component_summary(0.5)
                },
            ),
        }
    )

    assert diagnostics["available"] is True
    comparison = diagnostics["comparisons_vs_baseline"]["uniform"]["anchor_family"]["density"]
    assert comparison["query_local_score_delta"] == 0.19999999999999996
    assert comparison["range_component_deltas"]["range_point_f1"] == 0.19999999999999996
    assert comparison["top_primary_better_component_deltas"] == [
        {"component": "range_point_f1", "delta": 0.19999999999999996}
    ]
    assert comparison["top_baseline_better_component_deltas"] == []
    assert comparison["ship_evidence_count_deltas"]["ship_presence_recall"] == pytest.approx(0.2)
    assert comparison["ship_evidence_count_deltas"]["missed_trajectory_hit_count_total"] == -2


def test_build_run_payload_preserves_stable_artifact_fields() -> None:
    config = build_run_config()
    train_workload = _workload(2, 0.2)
    eval_workload = _workload(3, 0.3)
    trained = cast(
        TrainingOutputs,
        SimpleNamespace(
            history=[{"loss": 1.0}],
            target_diagnostics={"target": "ok"},
            fit_diagnostics={"fit": "ok"},
            feature_context={"query_prior_field_metadata": {"available": True}},
            best_epoch=2,
            best_loss=0.5,
            best_selection_score=0.25,
        ),
    )
    final_summaries = FinalRunSummaries(
        final_candidate=False,
        final_claim_summary={"status": "blocked"},
        diagnostic_summary={"available": True},
        legacy_range_useful_summary={"metric": "RangeUsefulLegacy"},
        learning_causality_summary={"selector_final_candidate": False},
        support_overlap_gate={"gate_pass": True},
        global_sanity_gate={"gate_pass": True},
        target_diffusion_gate={"gate_pass": True},
        workload_stability_gate={"gate_pass": True},
    )

    payload = build_run_payload(
        config=config,
        final_summaries=final_summaries,
        trained=trained,
        train_workload=train_workload,
        train_label_workloads=[train_workload],
        eval_workload=eval_workload,
        selection_workload=None,
        eval_workload_map={"range": 1.0},
        data_split_diagnostics={"split": "ok"},
        selector_budget_diagnostics={"eval": {}},
        primary_selector_trace=None,
        selection_selector_trace=None,
        train_selector_trace=None,
        train_marginal_causality_diagnostics=None,
        segment_oracle_allocation_audit={},
        target_segment_oracle_alignment_audit={},
        matched={"MLQDS": MethodScore(aggregate_f1=0.7, per_type_f1={"range": 0.7})},
        causality_ablation_scores={},
        learned_fill_diagnostics={},
        range_learned_fill_summary={},
        predictability_audit={"gate_pass": False},
        workload_scoring_compatibility_diagnostics={"available": True},
        range_compression_audit={},
        shift_pairs={},
        range_training_target_transform={},
        range_target_balance_diagnostics={},
        range_training_label_aggregation={},
        teacher_distillation_diagnostics={},
        selection_metric="score",
        workload_blind_eval=True,
        frozen_primary_masks={"MLQDS": cast(Any, object())},
        frozen_audit_methods_by_ratio={"0.0500": cast(Any, object())},
        data_audit=None,
        range_diagnostics_summary={"train": {}},
        workload_distribution_comparison={"deltas_vs_eval": {}},
        training_cuda_memory={"available": False},
        run_oracle_baseline=False,
    )

    assert payload["workload"] == "range"
    assert payload["train_query_count"] == 2
    assert payload["eval_query_count"] == 3
    assert payload["selection_query_count"] is None
    assert payload["matched"]["MLQDS"]["aggregate_f1"] == 0.7
    assert payload["workload_scoring_compatibility_diagnostics"] == {"available": True}
    assert payload["selector_trace_diagnostics"]["train_primary"] == {"available": False}
    assert payload["selector_trace_diagnostics"]["eval_primary"] == {"available": False}
    assert payload["selector_trace_diagnostics"]["selection_primary"] == {"available": False}
    assert payload["train_marginal_causality_diagnostics"] == {
        "available": False,
        "reason": "not_run",
    }
    assert payload["workload_blind_protocol"]["primary_masks_frozen_before_eval_query_scoring"]
    assert payload["workload_blind_protocol"]["frozen_audit_ratios"] == ["0.0500"]
    assert payload["training_history"] == [{"loss": 1.0}]
    assert payload["query_prior_field"] == {"available": True}
