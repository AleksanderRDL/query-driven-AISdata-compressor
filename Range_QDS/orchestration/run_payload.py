"""Single-run JSON payload assembly."""

from __future__ import annotations

from typing import Any

from config.run_config import RunConfig
from learning.model_features import model_type_metadata
from learning.outputs import TrainingOutputs
from orchestration.final_gate_summary import FinalRunSummaries
from orchestration.range_diagnostics import method_score_payload
from runtime.torch_runtime import amp_runtime_snapshot, torch_runtime_snapshot
from scoring.metrics import MethodScore
from scoring.range_usefulness import range_usefulness_weight_summary
from workloads.query_types import single_workload_type


def build_run_payload(
    *,
    config: RunConfig,
    final_summaries: FinalRunSummaries,
    trained: TrainingOutputs,
    train_workload: Any,
    train_label_workloads: list[Any],
    eval_workload: Any,
    selection_workload: Any | None,
    eval_workload_map: dict[str, float],
    data_split_diagnostics: dict[str, Any],
    selector_budget_diagnostics: dict[str, Any],
    primary_selector_trace: dict[str, Any] | None,
    selection_selector_trace: dict[str, Any] | None,
    train_selector_trace: dict[str, Any] | None,
    train_marginal_causality_diagnostics: dict[str, Any] | None,
    segment_oracle_allocation_audit: dict[str, Any],
    target_segment_oracle_alignment_audit: dict[str, Any],
    matched: dict[str, MethodScore],
    causality_ablation_scores: dict[str, MethodScore],
    learned_fill_diagnostics: dict[str, MethodScore],
    range_learned_fill_summary: dict[str, Any],
    predictability_audit: dict[str, Any],
    workload_scoring_compatibility_diagnostics: dict[str, Any],
    range_compression_audit: dict[str, dict[str, Any]],
    shift_pairs: dict[str, dict[str, float]],
    range_training_target_transform: dict[str, Any],
    range_target_balance_diagnostics: dict[str, Any],
    range_training_label_aggregation: dict[str, Any],
    teacher_distillation_diagnostics: dict[str, Any],
    selection_metric: str,
    workload_blind_eval: bool,
    frozen_primary_masks: dict[str, Any],
    frozen_audit_methods_by_ratio: dict[str, Any],
    data_audit: dict[str, Any] | None,
    range_diagnostics_summary: dict[str, Any],
    workload_distribution_comparison: dict[str, Any],
    training_cuda_memory: dict[str, Any],
    run_oracle_baseline: bool,
) -> dict[str, Any]:
    """Build the stable ``example_run.json`` payload from completed stage outputs."""
    return {
        "config": config.to_dict(),
        "final_claim_summary": final_summaries.final_claim_summary,
        "diagnostic_summary": final_summaries.diagnostic_summary,
        "legacy_range_useful_summary": final_summaries.legacy_range_useful_summary,
        "learning_causality_summary": final_summaries.learning_causality_summary,
        "support_overlap_gate": final_summaries.support_overlap_gate,
        "global_sanity_gate": final_summaries.global_sanity_gate,
        "target_diffusion_gate": final_summaries.target_diffusion_gate,
        "workload": single_workload_type(eval_workload_map),
        "train_query_count": len(train_workload.typed_queries),
        "train_label_workload_count": len(train_label_workloads),
        "train_label_workload_query_counts": [
            len(workload.typed_queries) for workload in train_label_workloads
        ],
        "eval_query_count": len(eval_workload.typed_queries),
        "selection_query_count": len(selection_workload.typed_queries)
        if selection_workload is not None
        else None,
        "train_query_coverage": train_workload.coverage_fraction,
        "train_label_workload_coverages": [
            workload.coverage_fraction for workload in train_label_workloads
        ],
        "eval_query_coverage": eval_workload.coverage_fraction,
        "selection_query_coverage": selection_workload.coverage_fraction
        if selection_workload is not None
        else None,
        "query_generation_diagnostics": {
            "train": train_workload.generation_diagnostics,
            "train_label_workloads": [
                workload.generation_diagnostics for workload in train_label_workloads
            ],
            "eval": eval_workload.generation_diagnostics,
            "selection": selection_workload.generation_diagnostics
            if selection_workload is not None
            else None,
        },
        "data_split_diagnostics": data_split_diagnostics,
        "selector_budget_diagnostics": selector_budget_diagnostics,
        "selector_trace_diagnostics": {
            "train_primary": train_selector_trace
            if train_selector_trace is not None
            else {"available": False},
            "eval_primary": primary_selector_trace
            if primary_selector_trace is not None
            else {"available": False},
            "selection_primary": selection_selector_trace
            if selection_selector_trace is not None
            else {"available": False},
        },
        "train_marginal_causality_diagnostics": (
            train_marginal_causality_diagnostics
            if train_marginal_causality_diagnostics is not None
            else {"available": False, "reason": "not_run"}
        ),
        "segment_oracle_allocation_audit": segment_oracle_allocation_audit,
        "target_segment_oracle_alignment_audit": target_segment_oracle_alignment_audit,
        "matched": {name: method_score_payload(metrics) for name, metrics in matched.items()},
        "learning_causality_ablations": {
            name: method_score_payload(metrics)
            for name, metrics in causality_ablation_scores.items()
        },
        "learned_fill_diagnostics": {
            name: method_score_payload(metrics)
            for name, metrics in learned_fill_diagnostics.items()
        },
        "range_learned_fill_summary": range_learned_fill_summary,
        "predictability_audit": predictability_audit,
        "workload_scoring_compatibility_diagnostics": (
            workload_scoring_compatibility_diagnostics
        ),
        "workload_stability_gate": final_summaries.workload_stability_gate,
        "range_compression_audit": range_compression_audit,
        "shift": shift_pairs,
        "training_history": trained.history,
        "training_target_diagnostics": trained.target_diagnostics,
        "training_fit_diagnostics": trained.fit_diagnostics,
        "range_training_target_transform": range_training_target_transform,
        "model_metadata": model_type_metadata(config.model.model_type),
        "query_prior_field": trained.feature_context.get(
            "query_prior_field_metadata", {"available": False}
        ),
        "range_target_balance": range_target_balance_diagnostics,
        "range_training_label_aggregation": range_training_label_aggregation,
        "teacher_distillation": teacher_distillation_diagnostics,
        "best_epoch": trained.best_epoch,
        "best_loss": trained.best_loss,
        "best_selection_score": trained.best_selection_score,
        "checkpoint_selection_metric": selection_metric,
        "checkpoint_selection_metric_requested": config.model.checkpoint_selection_metric,
        "checkpoint_score_variant": config.model.checkpoint_score_variant,
        "final_metrics_mode": config.baselines.final_metrics_mode,
        "workload_blind_protocol": {
            "enabled": bool(workload_blind_eval),
            "model_type": config.model.model_type,
            "masks_frozen_before_eval_query_scoring": bool(workload_blind_eval),
            "eval_queries_seen_by_model": False,
            "eval_queries_seen_by_feature_builder": False,
            "eval_queries_seen_by_selector": False,
            "checkpoint_selected_on_eval_queries": False,
            "query_conditioned_range_aware_used_for_product_acceptance": False,
            "primary_masks_frozen_before_eval_query_scoring": bool(workload_blind_eval),
            "audit_masks_frozen_before_eval_query_scoring": bool(
                workload_blind_eval and bool(frozen_audit_methods_by_ratio)
            ),
            "frozen_audit_ratio_count": len(frozen_audit_methods_by_ratio),
            "frozen_method_names": sorted(frozen_primary_masks),
            "frozen_audit_ratios": sorted(frozen_audit_methods_by_ratio),
            "eval_geometry_blend_allowed": not bool(workload_blind_eval),
        },
        "range_usefulness_weight_summary": range_usefulness_weight_summary(),
        "checkpoint_smoothing_window": config.model.checkpoint_smoothing_window,
        "mlqds_score_mode": config.model.mlqds_score_mode,
        "mlqds_score_temperature": config.model.mlqds_score_temperature,
        "mlqds_rank_confidence_weight": config.model.mlqds_rank_confidence_weight,
        "mlqds_range_geometry_blend": config.model.mlqds_range_geometry_blend,
        "mlqds_hybrid_mode": config.model.mlqds_hybrid_mode,
        "mlqds_stratified_center_weight": config.model.mlqds_stratified_center_weight,
        "mlqds_min_learned_swaps": config.model.mlqds_min_learned_swaps,
        "oracle_diagnostic": {
            "kind": "additive_label_greedy",
            "enabled": run_oracle_baseline,
            "exact_optimum": False,
            "retained_mask_constructor": "per_trajectory_topk_with_endpoints",
            "purpose": "diagnostic label-greedy reference, not exact retained-set RangeUseful optimum",
        },
        "range_label_mode": config.model.range_label_mode,
        "range_boundary_prior_weight": config.model.range_boundary_prior_weight,
        "range_boundary_prior_enabled": config.model.range_boundary_prior_weight > 0.0,
        "data_audit": data_audit,
        "workload_diagnostics": range_diagnostics_summary,
        "workload_distribution_comparison": workload_distribution_comparison,
        "torch_runtime": {
            **torch_runtime_snapshot(),
            "amp": amp_runtime_snapshot(config.model.amp_mode),
        },
        "cuda_memory": {
            "training": training_cuda_memory,
        },
    }
