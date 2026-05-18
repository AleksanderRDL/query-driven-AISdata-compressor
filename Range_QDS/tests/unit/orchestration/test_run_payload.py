"""Focused tests for single-run artifact payload assembly."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from config.run_config import build_run_config
from learning.outputs import TrainingOutputs
from orchestration.final_gate_summary import FinalRunSummaries
from orchestration.run_payload import build_run_payload
from scoring.metrics import MethodScore


def _workload(query_count: int, coverage: float) -> SimpleNamespace:
    return SimpleNamespace(
        typed_queries=[{"type": "range", "params": {}} for _ in range(query_count)],
        coverage_fraction=coverage,
        generation_diagnostics={"accepted": query_count},
    )


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
        segment_oracle_allocation_audit={},
        target_segment_oracle_alignment_audit={},
        matched={"MLQDS": MethodScore(aggregate_f1=0.7, per_type_f1={"range": 0.7})},
        causality_ablation_scores={},
        learned_fill_diagnostics={},
        range_learned_fill_summary={},
        predictability_audit={"gate_pass": False},
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
    assert payload["selector_trace_diagnostics"]["eval_primary"] == {"available": False}
    assert payload["workload_blind_protocol"]["primary_masks_frozen_before_eval_query_scoring"]
    assert payload["workload_blind_protocol"]["frozen_audit_ratios"] == ["0.0500"]
    assert payload["training_history"] == [{"loss": 1.0}]
    assert payload["query_prior_field"] == {"available": True}
