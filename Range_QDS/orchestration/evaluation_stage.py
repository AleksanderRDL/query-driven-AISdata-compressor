"""Evaluation-stage orchestration for single experiment runs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch

from config.experiment_config import ExperimentConfig, SeedBundle
from evaluation.baselines import FrozenMaskMethod, Method, OracleMethod
from evaluation.evaluate_methods import evaluate_method
from evaluation.metrics import MethodEvaluation
from evaluation.tables import (
    print_geometric_distortion_table,
    print_method_comparison_table,
    print_range_usefulness_table,
    print_shift_table,
)
from orchestration.causality import retained_mask_comparison
from orchestration.experiment_methods import (
    attach_range_geometry_scores,
    build_learned_fill_methods,
    evaluate_shift_pairs,
    prepare_eval_labels,
    prepare_eval_query_cache,
)
from orchestration.range_cache import RangeRuntimeCache, range_only_queries
from orchestration.range_diagnostics import evaluation_metrics_payload
from orchestration.segment_audits import (
    factorized_head_probability_sources_from_logits,
    segment_oracle_allocation_audit,
    target_segment_oracle_alignment_audit,
)
from queries.query_types import single_workload_type
from queries.workload import TypedQueryWorkload
from training.training_outputs import TrainingOutputs

PhaseLogger = Callable[[str], AbstractContextManager[None]]


@dataclass
class EvaluationStageOutputs:
    """Matched, diagnostic, audit, and shift evaluation outputs."""

    matched: dict[str, MethodEvaluation]
    matched_table: str
    geometric_table: str
    range_usefulness_table: str
    learned_fill_diagnostics: dict[str, MethodEvaluation]
    learned_fill_table: str
    causality_ablation_evaluations: dict[str, MethodEvaluation]
    causality_ablation_mask_diagnostics: dict[str, dict[str, Any]]
    range_compression_audit: dict[str, dict[str, Any]]
    range_compression_audit_table: str
    shift_pairs: dict[str, dict[str, float]]
    shift_table: str
    segment_oracle_allocation_audit: dict[str, Any]
    target_segment_oracle_alignment_audit: dict[str, Any]
    run_oracle_baseline: bool


def run_evaluation_stage(
    *,
    config: ExperimentConfig,
    seeds: SeedBundle,
    trained: TrainingOutputs,
    methods: list[Method],
    retention_methods: list[Method],
    workload_blind_eval: bool,
    audit_ratios: Sequence[float],
    frozen_primary_masks: dict[str, torch.Tensor],
    frozen_audit_methods_by_ratio: dict[str, list[Method]],
    frozen_primary_scores: dict[str, torch.Tensor],
    frozen_primary_head_logits: dict[str, torch.Tensor],
    frozen_primary_segment_scores: dict[str, torch.Tensor],
    frozen_primary_selector_segment_scores: dict[str, torch.Tensor],
    causality_ablation_methods: list[FrozenMaskMethod],
    train_workload: TypedQueryWorkload,
    train_workload_map: dict[str, float],
    eval_workload: TypedQueryWorkload,
    eval_workload_map: dict[str, float],
    test_points: torch.Tensor,
    test_boundaries: list[tuple[int, int]],
    test_mmsis: list[int] | None,
    range_runtime_caches: dict[str, RangeRuntimeCache],
    save_masks: bool,
    mlqds_range_geometry_blend: float,
    phase: PhaseLogger,
) -> EvaluationStageOutputs:
    """Run matched evaluation, diagnostics, audits, and workload-shift evaluation."""
    matched: dict[str, MethodEvaluation] = {}
    oracle_method: OracleMethod | None = None
    eval_labels: torch.Tensor | None = None
    segment_oracle_audit_payload: dict[str, Any] = {"available": False, "reason": "not_run"}
    target_segment_alignment_payload: dict[str, Any] = {
        "available": False,
        "reason": "not_run",
    }
    eval_is_range_only = len(range_only_queries(eval_workload.typed_queries)) == len(
        eval_workload.typed_queries
    )
    final_metrics_mode = str(getattr(config.baselines, "final_metrics_mode", "diagnostic")).lower()
    if final_metrics_mode not in {"diagnostic", "core"}:
        raise ValueError("final_metrics_mode must be either 'diagnostic' or 'core'.")
    run_final_diagnostics = final_metrics_mode == "diagnostic"
    run_oracle_baseline = bool(config.baselines.include_oracle and run_final_diagnostics)
    run_learned_fill_diagnostics = bool(eval_is_range_only and run_final_diagnostics)

    with phase("eval-query-cache-prep"):
        eval_query_cache = prepare_eval_query_cache(
            test_points=test_points,
            test_boundaries=test_boundaries,
            eval_workload=eval_workload,
            eval_is_range_only=eval_is_range_only,
            runtime_cache=range_runtime_caches["eval"],
        )
    if run_oracle_baseline or run_learned_fill_diagnostics or mlqds_range_geometry_blend > 0.0:
        with phase("eval-label-prep"):
            eval_labels = prepare_eval_labels(
                test_points=test_points,
                test_boundaries=test_boundaries,
                eval_workload=eval_workload,
                eval_workload_map=eval_workload_map,
                config=config,
                seeds=seeds,
                eval_is_range_only=eval_is_range_only,
                run_oracle_baseline=run_oracle_baseline,
                runtime_cache=range_runtime_caches["eval"],
            )
    if mlqds_range_geometry_blend > 0.0:
        if eval_labels is None:
            raise RuntimeError(
                "MLQDS range geometry blend requested but eval labels were not prepared."
            )
        attach_range_geometry_scores(
            methods=methods,
            eval_labels=eval_labels,
            eval_workload_map=eval_workload_map,
        )
    if (
        workload_blind_eval
        and str(getattr(config.model, "selector_type", "")).lower() == "learned_segment_budget_v1"
    ):
        segment_oracle_audit_payload = segment_oracle_allocation_audit(
            point_scores=frozen_primary_scores.get("MLQDS"),
            segment_budget_scores=frozen_primary_segment_scores.get("MLQDS"),
            selector_segment_scores=frozen_primary_selector_segment_scores.get("MLQDS"),
            eval_labels=eval_labels,
            boundaries=test_boundaries,
            workload_type=single_workload_type(eval_workload_map),
            head_scores_by_name=factorized_head_probability_sources_from_logits(
                frozen_primary_head_logits.get("MLQDS")
            ),
            retained_mask=frozen_primary_masks.get("MLQDS"),
        )
        try:
            target_segment_alignment_payload = target_segment_oracle_alignment_audit(
                points=test_points,
                boundaries=test_boundaries,
                typed_queries=eval_workload.typed_queries,
                eval_labels=eval_labels,
                workload_type=single_workload_type(eval_workload_map),
                retained_mask=frozen_primary_masks.get("MLQDS"),
            )
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            target_segment_alignment_payload = {
                "available": False,
                "reason": "target_alignment_failed",
                "diagnostic_only": True,
                "error": str(exc),
            }

    with phase("evaluate-matched"):
        for method in methods:
            with phase(f"  eval {method.name}"):
                matched[method.name] = evaluate_method(
                    method=method,
                    points=test_points,
                    boundaries=test_boundaries,
                    typed_queries=eval_workload.typed_queries,
                    workload_map=eval_workload_map,
                    compression_ratio=config.model.compression_ratio,
                    return_mask=method.name == "MLQDS" or save_masks,
                    query_cache=eval_query_cache,
                )

        if run_oracle_baseline:
            if eval_labels is None:
                raise RuntimeError("Oracle baseline requested but eval labels were not prepared.")
            oracle_method = OracleMethod(
                labels=eval_labels,
                workload_type=single_workload_type(eval_workload_map),
            )
            with phase(f"  eval {oracle_method.name}"):
                matched[oracle_method.name] = evaluate_method(
                    method=oracle_method,
                    points=test_points,
                    boundaries=test_boundaries,
                    typed_queries=eval_workload.typed_queries,
                    workload_map=eval_workload_map,
                    compression_ratio=config.model.compression_ratio,
                    query_cache=eval_query_cache,
                )

    causality_ablation_evaluations: dict[str, MethodEvaluation] = {}
    causality_ablation_mask_diagnostics: dict[str, dict[str, Any]] = {}
    if causality_ablation_methods:
        primary_ablation_mask = frozen_primary_masks.get("MLQDS")
        with phase("learning-causality-ablations"):
            for method in causality_ablation_methods:
                causality_ablation_mask_diagnostics[method.name] = retained_mask_comparison(
                    primary_mask=primary_ablation_mask,
                    ablation_mask=method.retained_mask,
                    expected_shape=(
                        primary_ablation_mask.shape
                        if isinstance(primary_ablation_mask, torch.Tensor)
                        else method.retained_mask.shape
                    ),
                )
                with phase(f"  ablation {method.name}"):
                    causality_ablation_evaluations[method.name] = evaluate_method(
                        method=method,
                        points=test_points,
                        boundaries=test_boundaries,
                        typed_queries=eval_workload.typed_queries,
                        workload_map=eval_workload_map,
                        compression_ratio=config.model.compression_ratio,
                        query_cache=eval_query_cache,
                    )

    learned_fill_diagnostics: dict[str, MethodEvaluation] = {"MLQDS": matched["MLQDS"]}
    learned_fill_table = ""
    diagnostic_methods: list[Method] = []
    if run_learned_fill_diagnostics:
        if eval_labels is None:
            raise RuntimeError(
                "Learned-fill diagnostics requested but eval labels were not prepared."
            )
        diagnostic_methods = build_learned_fill_methods(
            test_points=test_points,
            eval_labels=eval_labels,
            eval_workload_map=eval_workload_map,
            config=config,
            seeds=seeds,
        )
        with phase("learned-fill-diagnostics"):
            for method in diagnostic_methods:
                with phase(f"  fill {method.name}"):
                    learned_fill_diagnostics[method.name] = evaluate_method(
                        method=method,
                        points=test_points,
                        boundaries=test_boundaries,
                        typed_queries=eval_workload.typed_queries,
                        workload_map=eval_workload_map,
                        compression_ratio=config.model.compression_ratio,
                        query_cache=eval_query_cache,
                    )
        learned_fill_table = print_range_usefulness_table(learned_fill_diagnostics)

    matched_table = print_method_comparison_table(matched)
    geometric_table = print_geometric_distortion_table(matched)
    range_usefulness_table = print_range_usefulness_table(matched)
    range_compression_audit: dict[str, dict[str, Any]] = {}
    range_compression_audit_table = ""
    if audit_ratios:
        audit_methods = [
            *(retention_methods if workload_blind_eval else methods),
            *diagnostic_methods,
        ]
        if oracle_method is not None:
            audit_methods.append(oracle_method)
        audit_sections: list[str] = []
        with phase("range-compression-audit"):
            for ratio in audit_ratios:
                if abs(float(ratio) - float(config.model.compression_ratio)) <= 1e-9:
                    ratio_results = {
                        **matched,
                        **{
                            name: metrics
                            for name, metrics in learned_fill_diagnostics.items()
                            if name not in matched
                        },
                    }
                else:
                    ratio_results: dict[str, MethodEvaluation] = {}
                    ratio_key = f"{float(ratio):.4f}"
                    ratio_audit_methods = audit_methods
                    if workload_blind_eval and ratio_key in frozen_audit_methods_by_ratio:
                        ratio_audit_methods = [
                            *frozen_audit_methods_by_ratio[ratio_key],
                            *diagnostic_methods,
                        ]
                        if oracle_method is not None:
                            ratio_audit_methods.append(oracle_method)
                    for method in ratio_audit_methods:
                        with phase(f"  audit {method.name} ratio={ratio:.4f}"):
                            ratio_results[method.name] = evaluate_method(
                                method=method,
                                points=test_points,
                                boundaries=test_boundaries,
                                typed_queries=eval_workload.typed_queries,
                                workload_map=eval_workload_map,
                                compression_ratio=float(ratio),
                                query_cache=eval_query_cache,
                            )
                ratio_key = f"{float(ratio):.4f}"
                range_compression_audit[ratio_key] = {
                    name: evaluation_metrics_payload(metrics)
                    for name, metrics in ratio_results.items()
                }
                audit_sections.append(
                    f"compression_ratio={ratio_key}\n{print_range_usefulness_table(ratio_results)}"
                )
        range_compression_audit_table = "\n\n".join(audit_sections)

    with phase("evaluate-shift"):
        shift_pairs = evaluate_shift_pairs(
            matched_mlqds_score=float(matched["MLQDS"].aggregate_f1),
            trained=trained,
            train_workload=train_workload,
            train_workload_map=train_workload_map,
            eval_workload_map=eval_workload_map,
            config=config,
            test_points=test_points,
            test_boundaries=test_boundaries,
            test_mmsis=test_mmsis,
        )
    shift_table = print_shift_table(shift_pairs)

    return EvaluationStageOutputs(
        matched=matched,
        matched_table=matched_table,
        geometric_table=geometric_table,
        range_usefulness_table=range_usefulness_table,
        learned_fill_diagnostics=learned_fill_diagnostics,
        learned_fill_table=learned_fill_table,
        causality_ablation_evaluations=causality_ablation_evaluations,
        causality_ablation_mask_diagnostics=causality_ablation_mask_diagnostics,
        range_compression_audit=range_compression_audit,
        range_compression_audit_table=range_compression_audit_table,
        shift_pairs=shift_pairs,
        shift_table=shift_table,
        segment_oracle_allocation_audit=segment_oracle_audit_payload,
        target_segment_oracle_alignment_audit=target_segment_alignment_payload,
        run_oracle_baseline=run_oracle_baseline,
    )
