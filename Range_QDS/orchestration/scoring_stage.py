"""Scoring-stage orchestration for single runs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch

from config.run_config import RunConfig, SeedBundle
from learning.outputs import TrainingOutputs
from orchestration.causality import retained_mask_comparison
from orchestration.range_diagnostics import method_score_payload
from orchestration.range_runtime_cache import RangeRuntimeCache, range_only_queries
from orchestration.scoring_methods import (
    attach_range_geometry_scores,
    build_learned_fill_methods,
    prepare_eval_labels,
    prepare_eval_query_cache,
    score_shift_pairs,
)
from orchestration.segment_audits import (
    factorized_head_probability_sources_from_logits,
    segment_oracle_allocation_audit,
    target_segment_oracle_alignment_audit,
)
from scoring.method_scoring import RANGE_QUERY_COMPONENT_KEYS, score_method
from scoring.methods import FrozenMaskMethod, Method, OracleMethod
from scoring.metrics import MethodScore
from scoring.score_tables import (
    print_geometric_distortion_table,
    print_method_comparison_table,
    print_range_usefulness_table,
    print_shift_table,
)
from selection.selector_types import LEARNED_SEGMENT_BUDGET_SELECTOR_TYPE
from workloads.query_types import single_workload_type
from workloads.typed_workload import TypedQueryWorkload

PhaseLogger = Callable[[str], AbstractContextManager[None]]


def _range_query_metadata_summary(metrics: MethodScore) -> dict[str, Any] | None:
    payload = metrics.range_audit.get("range_query_metadata_component_summary")
    if isinstance(payload, dict) and bool(payload.get("available", False)):
        return payload
    return None


def _top_component_deltas(
    component_deltas: dict[str, float],
    *,
    reverse: bool,
) -> list[dict[str, float | str]]:
    signed_deltas = {
        component: delta
        for component, delta in component_deltas.items()
        if (delta > 0.0 if reverse else delta < 0.0)
    }
    return [
        {"component": component, "delta": float(delta)}
        for component, delta in sorted(
            signed_deltas.items(),
            key=lambda item: item[1],
            reverse=reverse,
        )[:5]
    ]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except TypeError, ValueError:
        return float(default)
    if result != result:
        return float(default)
    return float(result)


def _family_group_comparison(
    *,
    primary_group: dict[str, Any],
    baseline_group: dict[str, Any],
) -> dict[str, Any]:
    primary_components = primary_group.get("range_components")
    baseline_components = baseline_group.get("range_components")
    if not isinstance(primary_components, dict) or not isinstance(baseline_components, dict):
        return {"available": False, "reason": "missing_range_components"}

    component_deltas: dict[str, float] = {}
    for key in RANGE_QUERY_COMPONENT_KEYS:
        component_deltas[key] = float(primary_components.get(key, 0.0)) - float(
            baseline_components.get(key, 0.0)
        )
    primary_query_local = float(
        primary_group.get("query_local_utility_query_local_weighted_score_normalized", 0.0)
    )
    baseline_query_local = float(
        baseline_group.get("query_local_utility_query_local_weighted_score_normalized", 0.0)
    )
    primary_ship_evidence = primary_group.get("ship_evidence_counts")
    baseline_ship_evidence = baseline_group.get("ship_evidence_counts")
    ship_evidence_deltas: dict[str, float] = {}
    if isinstance(primary_ship_evidence, dict) and isinstance(baseline_ship_evidence, dict):
        for key in sorted(set(primary_ship_evidence) | set(baseline_ship_evidence)):
            ship_evidence_deltas[str(key)] = _safe_float(
                primary_ship_evidence.get(key)
            ) - _safe_float(baseline_ship_evidence.get(key))
    return {
        "available": True,
        "query_count": int(primary_group.get("query_count", 0) or 0),
        "baseline_query_count": int(baseline_group.get("query_count", 0) or 0),
        "primary_query_local_score_normalized": primary_query_local,
        "baseline_query_local_score_normalized": baseline_query_local,
        "query_local_score_delta": float(primary_query_local - baseline_query_local),
        "primary_range_usefulness_score": float(primary_group.get("range_usefulness_score", 0.0)),
        "baseline_range_usefulness_score": float(baseline_group.get("range_usefulness_score", 0.0)),
        "range_usefulness_delta": float(
            float(primary_group.get("range_usefulness_score", 0.0))
            - float(baseline_group.get("range_usefulness_score", 0.0))
        ),
        "range_component_deltas": component_deltas,
        "top_primary_better_component_deltas": _top_component_deltas(
            component_deltas,
            reverse=True,
        ),
        "top_baseline_better_component_deltas": _top_component_deltas(
            component_deltas,
            reverse=False,
        ),
        "primary_ship_evidence_counts": (
            primary_ship_evidence if isinstance(primary_ship_evidence, dict) else {}
        ),
        "baseline_ship_evidence_counts": (
            baseline_ship_evidence if isinstance(baseline_ship_evidence, dict) else {}
        ),
        "ship_evidence_count_deltas": ship_evidence_deltas,
    }


def build_workload_scoring_compatibility_diagnostics(
    matched: dict[str, MethodScore],
) -> dict[str, Any]:
    """Compare query-family score components for MLQDS and matched baselines."""
    primary = matched.get("MLQDS")
    if primary is None:
        return {
            "available": False,
            "diagnostic_only": True,
            "schema_version": 1,
            "reason": "missing_primary_method",
        }
    primary_summary = _range_query_metadata_summary(primary)
    if primary_summary is None:
        return {
            "available": False,
            "diagnostic_only": True,
            "schema_version": 1,
            "reason": "missing_primary_range_query_metadata_component_summary",
        }

    baseline_names: list[str] = []
    for name in ("uniform", "DouglasPeucker"):
        metrics = matched.get(name)
        if metrics is not None and _range_query_metadata_summary(metrics) is not None:
            baseline_names.append(name)
    if not baseline_names:
        return {
            "available": False,
            "diagnostic_only": True,
            "schema_version": 1,
            "reason": "missing_baseline_range_query_metadata_component_summary",
            "primary_method": "MLQDS",
        }

    comparisons: dict[str, Any] = {}
    primary_group_by = primary_summary.get("group_by", {})
    for baseline_name in baseline_names:
        baseline = matched[baseline_name]
        baseline_summary = _range_query_metadata_summary(baseline)
        if baseline_summary is None:
            continue
        baseline_group_by = baseline_summary.get("group_by", {})
        baseline_comparisons: dict[str, Any] = {}
        if isinstance(primary_group_by, dict) and isinstance(baseline_group_by, dict):
            for group_type, primary_groups in primary_group_by.items():
                baseline_groups = baseline_group_by.get(group_type)
                if not isinstance(primary_groups, dict) or not isinstance(baseline_groups, dict):
                    continue
                group_payload: dict[str, Any] = {}
                for group_key, primary_group in primary_groups.items():
                    baseline_group = baseline_groups.get(group_key)
                    if isinstance(primary_group, dict) and isinstance(baseline_group, dict):
                        group_payload[str(group_key)] = _family_group_comparison(
                            primary_group=primary_group,
                            baseline_group=baseline_group,
                        )
                baseline_comparisons[str(group_type)] = group_payload
        comparisons[baseline_name] = baseline_comparisons

    return {
        "available": True,
        "diagnostic_only": True,
        "schema_version": 1,
        "purpose": "workload_profile_scoring_component_compatibility",
        "primary_method": "MLQDS",
        "baseline_methods": baseline_names,
        "component_keys": list(RANGE_QUERY_COMPONENT_KEYS),
        "source": "matched.method.range_audit.range_query_metadata_component_summary",
        "query_rows_in_matched_range_audit_only": True,
        "comparisons_vs_baseline": comparisons,
    }


@dataclass
class ScoringStageOutputs:
    """Matched, diagnostic, audit, and shift scoring outputs."""

    matched: dict[str, MethodScore]
    matched_table: str
    geometric_table: str
    range_usefulness_table: str
    learned_fill_diagnostics: dict[str, MethodScore]
    learned_fill_table: str
    causality_ablation_scores: dict[str, MethodScore]
    causality_ablation_mask_diagnostics: dict[str, dict[str, Any]]
    range_compression_audit: dict[str, dict[str, Any]]
    range_compression_audit_table: str
    workload_scoring_compatibility_diagnostics: dict[str, Any]
    shift_pairs: dict[str, dict[str, float]]
    shift_table: str
    segment_oracle_allocation_audit: dict[str, Any]
    target_segment_oracle_alignment_audit: dict[str, Any]
    run_oracle_baseline: bool


def run_scoring_stage(
    *,
    config: RunConfig,
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
) -> ScoringStageOutputs:
    """Run matched scoring, diagnostics, audits, and workload-shift scoring."""
    matched: dict[str, MethodScore] = {}
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
        and str(getattr(config.model, "selector_type", "")).lower()
        == LEARNED_SEGMENT_BUDGET_SELECTOR_TYPE
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
                target_mode=str(getattr(config.model, "range_training_target_mode", "")),
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
                matched[method.name] = score_method(
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
                matched[oracle_method.name] = score_method(
                    method=oracle_method,
                    points=test_points,
                    boundaries=test_boundaries,
                    typed_queries=eval_workload.typed_queries,
                    workload_map=eval_workload_map,
                    compression_ratio=config.model.compression_ratio,
                    query_cache=eval_query_cache,
                )

    causality_ablation_scores: dict[str, MethodScore] = {}
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
                    causality_ablation_scores[method.name] = score_method(
                        method=method,
                        points=test_points,
                        boundaries=test_boundaries,
                        typed_queries=eval_workload.typed_queries,
                        workload_map=eval_workload_map,
                        compression_ratio=config.model.compression_ratio,
                        query_cache=eval_query_cache,
                    )

    learned_fill_diagnostics: dict[str, MethodScore] = {"MLQDS": matched["MLQDS"]}
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
                    learned_fill_diagnostics[method.name] = score_method(
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
    workload_scoring_compatibility_diagnostics = build_workload_scoring_compatibility_diagnostics(
        matched
    )
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
                    ratio_results: dict[str, MethodScore] = {}
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
                            ratio_results[method.name] = score_method(
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
                    name: method_score_payload(metrics) for name, metrics in ratio_results.items()
                }
                audit_sections.append(
                    f"compression_ratio={ratio_key}\n{print_range_usefulness_table(ratio_results)}"
                )
        range_compression_audit_table = "\n\n".join(audit_sections)

    with phase("evaluate-shift"):
        shift_pairs = score_shift_pairs(
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

    return ScoringStageOutputs(
        matched=matched,
        matched_table=matched_table,
        geometric_table=geometric_table,
        range_usefulness_table=range_usefulness_table,
        learned_fill_diagnostics=learned_fill_diagnostics,
        learned_fill_table=learned_fill_table,
        causality_ablation_scores=causality_ablation_scores,
        causality_ablation_mask_diagnostics=causality_ablation_mask_diagnostics,
        range_compression_audit=range_compression_audit,
        range_compression_audit_table=range_compression_audit_table,
        workload_scoring_compatibility_diagnostics=(workload_scoring_compatibility_diagnostics),
        shift_pairs=shift_pairs,
        shift_table=shift_table,
        segment_oracle_allocation_audit=segment_oracle_audit_payload,
        target_segment_oracle_alignment_audit=target_segment_alignment_payload,
        run_oracle_baseline=run_oracle_baseline,
    )
