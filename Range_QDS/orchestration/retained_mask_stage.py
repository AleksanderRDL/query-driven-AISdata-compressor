"""Retained-mask freezing for workload-blind runs."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, cast

import torch

from config.run_config import RunConfig, SeedBundle
from learning.outputs import TrainingOutputs
from orchestration.length_diagnostics import (
    score_protected_length_feasibility,
    score_protected_length_frontier,
)
from orchestration.retained_mask_ablation_stage import freeze_retained_mask_ablations
from orchestration.selector_diagnostics import (
    factorized_score_component_vectors_from_logits,
    query_free_retained_removal_teacher_proxy_vectors,
    query_prior_component_vectors_for_points,
    retained_decision_marginal_query_local_utility_diagnostics,
    selector_segment_score_source_label,
)
from scoring.methods import FrozenMaskMethod, Method
from selection.learned_segment_budget import (
    simplify_with_learned_segment_budget_with_trace,
)
from selection.selector_types import LEARNED_SEGMENT_BUDGET_SELECTOR_TYPE
from workloads.typed_workload import TypedQueryWorkload

PhaseLogger = Callable[[str], AbstractContextManager[None]]


@dataclass
class RetainedMaskFreezingOutputs:
    """Frozen masks and query-free diagnostics produced before eval query scoring."""

    methods: list[Method]
    frozen_primary_masks: dict[str, torch.Tensor]
    frozen_audit_methods_by_ratio: dict[str, list[Method]]
    frozen_primary_scores: dict[str, torch.Tensor]
    frozen_primary_raw_preds: dict[str, torch.Tensor]
    frozen_primary_head_logits: dict[str, torch.Tensor]
    frozen_primary_segment_scores: dict[str, torch.Tensor]
    frozen_primary_path_length_support_scores: dict[str, torch.Tensor]
    frozen_primary_selector_segment_scores: dict[str, torch.Tensor]
    primary_selector_trace: dict[str, Any] | None
    causality_ablation_methods: list[FrozenMaskMethod]
    causal_ablation_freeze_failures: dict[str, str]
    prior_sensitivity_diagnostics: dict[str, Any]
    prior_channel_ablation_diagnostics: dict[str, Any]
    head_ablation_sensitivity_diagnostics: dict[str, Any]
    segment_budget_head_ablation_mode: str | None
    freeze_timing_diagnostics: dict[str, Any]


def freeze_workload_blind_retained_masks(
    *,
    methods: list[Method],
    retention_methods: list[Method],
    workload_blind_eval: bool,
    audit_ratios: Sequence[float],
    config: RunConfig,
    trained: TrainingOutputs,
    eval_workload: TypedQueryWorkload,
    eval_workload_map: dict[str, float],
    test_mmsis: list[int] | None,
    test_points: torch.Tensor,
    test_boundaries: list[tuple[int, int]],
    seeds: SeedBundle,
    phase: PhaseLogger,
) -> RetainedMaskFreezingOutputs:
    """Freeze retained masks before eval query scoring for workload-blind runs."""
    frozen_primary_masks: dict[str, torch.Tensor] = {}
    frozen_audit_methods_by_ratio: dict[str, list[Method]] = {}
    frozen_primary_scores: dict[str, torch.Tensor] = {}
    frozen_primary_raw_preds: dict[str, torch.Tensor] = {}
    frozen_primary_head_logits: dict[str, torch.Tensor] = {}
    frozen_primary_segment_scores: dict[str, torch.Tensor] = {}
    frozen_primary_path_length_support_scores: dict[str, torch.Tensor] = {}
    frozen_primary_selector_segment_scores: dict[str, torch.Tensor] = {}
    primary_selector_trace: dict[str, Any] | None = None
    causality_ablation_methods: list[FrozenMaskMethod] = []
    causal_ablation_freeze_failures: dict[str, str] = {}
    prior_sensitivity_diagnostics: dict[str, Any] = {}
    prior_channel_ablation_diagnostics: dict[str, Any] = {}
    head_ablation_sensitivity_diagnostics: dict[str, Any] = {}
    segment_budget_head_ablation_mode: str | None = None
    freeze_timing_diagnostics: dict[str, Any] = {}
    if workload_blind_eval:
        freeze_started_at = time.perf_counter()
        freeze_timing_diagnostics = {
            "available": True,
            "diagnostic_only": True,
            "query_free": True,
            "stage": "freeze_workload_blind_retained_masks",
            "primary_method_simplify_seconds": {},
            "audit_method_simplify_seconds": {},
            "substage_seconds": {},
        }
        with phase("freeze-retained-masks"):
            for method in methods:
                with phase(f"  freeze {method.name}"):
                    freeze_t0 = time.perf_counter()
                    frozen_primary_masks[method.name] = (
                        method.simplify(
                            test_points,
                            test_boundaries,
                            config.model.compression_ratio,
                        )
                        .detach()
                        .cpu()
                    )
                    method_elapsed_seconds = float(time.perf_counter() - freeze_t0)
                    cast(Any, method).latency_ms = float(method_elapsed_seconds * 1000.0)
                    freeze_timing_diagnostics["primary_method_simplify_seconds"][method.name] = (
                        method_elapsed_seconds
                    )
                    score_snapshot_fn = getattr(method, "cached_score_snapshot", None)
                    score_snapshot = score_snapshot_fn() if callable(score_snapshot_fn) else None
                    score_cache = getattr(score_snapshot, "scores", None)
                    if isinstance(score_cache, torch.Tensor):
                        frozen_primary_scores[method.name] = score_cache.detach().cpu().float()
                    raw_pred_cache = getattr(score_snapshot, "raw_predictions", None)
                    if isinstance(raw_pred_cache, torch.Tensor):
                        frozen_primary_raw_preds[method.name] = (
                            raw_pred_cache.detach().cpu().float()
                        )
                    head_logit_cache = getattr(score_snapshot, "head_logits", None)
                    if isinstance(head_logit_cache, torch.Tensor):
                        frozen_primary_head_logits[method.name] = (
                            head_logit_cache.detach().cpu().float()
                        )
                    segment_score_cache = getattr(score_snapshot, "segment_scores", None)
                    if isinstance(segment_score_cache, torch.Tensor):
                        frozen_primary_segment_scores[method.name] = (
                            segment_score_cache.detach().cpu().float()
                        )
                    path_length_support_cache = getattr(
                        score_snapshot, "path_length_support_scores", None
                    )
                    if isinstance(path_length_support_cache, torch.Tensor):
                        frozen_primary_path_length_support_scores[method.name] = (
                            path_length_support_cache.detach().cpu().float()
                        )
                    selector_segment_score_cache = getattr(
                        score_snapshot, "selector_segment_scores", None
                    )
                    if isinstance(selector_segment_score_cache, torch.Tensor):
                        frozen_primary_selector_segment_scores[method.name] = (
                            selector_segment_score_cache.detach().cpu().float()
                        )
            primary_scores = frozen_primary_scores.get("MLQDS")
            primary_raw_preds = frozen_primary_raw_preds.get("MLQDS")
            if (
                primary_scores is not None
                and str(getattr(config.model, "selector_type", "")).lower()
                == LEARNED_SEGMENT_BUDGET_SELECTOR_TYPE
            ):
                primary_segment_scores = frozen_primary_segment_scores.get("MLQDS")
                primary_path_length_support_scores = frozen_primary_path_length_support_scores.get(
                    "MLQDS"
                )
                primary_selector_segment_scores = frozen_primary_selector_segment_scores.get(
                    "MLQDS"
                )
                trace_started_at = time.perf_counter()
                trace_mask, trace = simplify_with_learned_segment_budget_with_trace(
                    primary_scores,
                    test_boundaries,
                    float(config.model.compression_ratio),
                    segment_scores=primary_selector_segment_scores,
                    segment_point_scores=primary_segment_scores,
                    points=test_points.detach().cpu().float(),
                    geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                    segment_length_support_weight=float(
                        config.model.learned_segment_allocation_length_support_weight
                    ),
                    segment_allocation_weight_floor=float(
                        config.model.learned_segment_allocation_weight_floor
                    ),
                    segment_score_point_blend_weight=float(
                        config.model.learned_segment_score_blend_weight
                    ),
                    segment_transfer_calibration_mode=str(
                        config.model.learned_segment_transfer_calibration_mode
                    ),
                    fairness_preallocation_enabled=bool(
                        config.model.learned_segment_fairness_preallocation
                    ),
                    length_repair_fraction=float(
                        config.model.learned_segment_length_repair_fraction
                    ),
                    length_repair_score_protection_fraction=float(
                        config.model.learned_segment_length_repair_score_protection_fraction
                    ),
                    segment_score_source_label=selector_segment_score_source_label(
                        segment_scores=primary_selector_segment_scores,
                        path_length_support_scores=primary_path_length_support_scores,
                        length_support_blend_weight=float(
                            config.model.learned_segment_length_support_blend_weight
                        ),
                        base_segment_score_source="segment_budget_head_top20_mean",
                    ),
                )
                freeze_timing_diagnostics["substage_seconds"]["selector_trace_reconstruction"] = (
                    float(time.perf_counter() - trace_started_at)
                )
                frozen_mlqds_mask = frozen_primary_masks.get("MLQDS")
                if isinstance(frozen_mlqds_mask, torch.Tensor):
                    trace["retained_mask_matches_frozen_primary"] = bool(
                        torch.equal(trace_mask.detach().cpu(), frozen_mlqds_mask.detach().cpu())
                    )
                    trace["frozen_primary_retained_count"] = int(frozen_mlqds_mask.sum().item())
                trace["retained_mask_freeze_timing"] = freeze_timing_diagnostics
                diagnostic_started_at = time.perf_counter()
                try:
                    primary_head_logits = frozen_primary_head_logits.get("MLQDS")
                    sampled_prior_vectors, model_prior_vectors = (
                        query_prior_component_vectors_for_points(
                            test_points.detach().cpu().float(),
                            trained.feature_context.get("query_prior_field"),
                        )
                    )
                    teacher_proxy_vectors = query_free_retained_removal_teacher_proxy_vectors(
                        test_points.detach().cpu().float(),
                        test_boundaries,
                    )
                    trace["retained_decision_marginal_query_local_utility_alignment"] = (
                        retained_decision_marginal_query_local_utility_diagnostics(
                            points=test_points.detach().cpu().float(),
                            boundaries=test_boundaries,
                            typed_queries=eval_workload.typed_queries,
                            primary_retained_mask=trace_mask.detach().cpu().bool(),
                            raw_scores=primary_raw_preds,
                            selector_scores=primary_scores,
                            segment_scores=primary_segment_scores,
                            score_component_vectors=(
                                factorized_score_component_vectors_from_logits(primary_head_logits)
                            ),
                            query_free_teacher_proxy_vectors=teacher_proxy_vectors,
                            sampled_prior_vectors=sampled_prior_vectors,
                            model_prior_vectors=model_prior_vectors,
                            selector_trace=trace,
                            max_retained_per_source=32,
                            max_removed_candidates=64,
                            teacher_usage_split="eval_primary",
                        )
                    )
                except Exception as exc:  # pragma: no cover - diagnostic must not break freezing.
                    trace["retained_decision_marginal_query_local_utility_alignment"] = {
                        "available": False,
                        "diagnostic_only": True,
                        "reason": "diagnostic_failed",
                        "error": str(exc),
                    }
                finally:
                    freeze_timing_diagnostics["substage_seconds"]["retained_marginal_alignment"] = (
                        float(time.perf_counter() - diagnostic_started_at)
                    )
                compression_for_trace = float(config.model.compression_ratio)
                learned_fraction_min_for_trace = (
                    0.35
                    if compression_for_trace >= 0.10
                    else 0.25
                    if compression_for_trace >= 0.05
                    else 0.0
                )
                if learned_fraction_min_for_trace > 0.0:
                    length_diagnostic_started_at = time.perf_counter()
                    trace["score_protected_length_feasibility"] = (
                        score_protected_length_feasibility(
                            scores=primary_scores,
                            points=test_points,
                            boundaries=test_boundaries,
                            compression_ratio=compression_for_trace,
                            learned_slot_fraction_min=learned_fraction_min_for_trace,
                        )
                    )
                    trace["score_protected_length_frontier"] = score_protected_length_frontier(
                        scores=primary_scores,
                        points=test_points,
                        boundaries=test_boundaries,
                        compression_ratio=compression_for_trace,
                        learned_slot_fraction_min=learned_fraction_min_for_trace,
                    )
                    freeze_timing_diagnostics["substage_seconds"][
                        "score_protected_length_diagnostics"
                    ] = float(time.perf_counter() - length_diagnostic_started_at)
                primary_selector_trace = trace
                ablation_started_at = time.perf_counter()
                ablation_outputs = freeze_retained_mask_ablations(
                    config=config,
                    trained=trained,
                    eval_workload=eval_workload,
                    eval_workload_map=eval_workload_map,
                    test_mmsis=test_mmsis,
                    test_points=test_points,
                    test_boundaries=test_boundaries,
                    seeds=seeds,
                    primary_selector_trace=trace,
                    frozen_primary_masks=frozen_primary_masks,
                    primary_scores=primary_scores,
                    primary_raw_preds=primary_raw_preds,
                    primary_segment_scores=primary_segment_scores,
                    primary_path_length_support_scores=primary_path_length_support_scores,
                    primary_selector_segment_scores=primary_selector_segment_scores,
                    primary_head_logits=frozen_primary_head_logits.get("MLQDS"),
                )
                freeze_timing_diagnostics["substage_seconds"]["query_free_ablation_freeze"] = float(
                    time.perf_counter() - ablation_started_at
                )
                freeze_timing_diagnostics["ablation_freeze_timing"] = (
                    ablation_outputs.freeze_timing_diagnostics
                )
                primary_selector_trace = ablation_outputs.primary_selector_trace
                primary_selector_trace["retained_mask_freeze_timing"] = freeze_timing_diagnostics
                causality_ablation_methods = ablation_outputs.causality_ablation_methods
                causal_ablation_freeze_failures = ablation_outputs.causal_ablation_freeze_failures
                prior_sensitivity_diagnostics = ablation_outputs.prior_sensitivity_diagnostics
                prior_channel_ablation_diagnostics = (
                    ablation_outputs.prior_channel_ablation_diagnostics
                )
                head_ablation_sensitivity_diagnostics = (
                    ablation_outputs.head_ablation_sensitivity_diagnostics
                )
                segment_budget_head_ablation_mode = (
                    ablation_outputs.segment_budget_head_ablation_mode
                )
            freeze_timing_diagnostics["total_seconds"] = float(
                time.perf_counter() - freeze_started_at
            )
        methods = [
            FrozenMaskMethod(
                name=method.name,
                retained_mask=frozen_primary_masks[method.name],
                latency_ms=float(getattr(method, "latency_ms", 0.0)),
            )
            for method in methods
        ]
        print(
            "  workload_blind_protocol=enabled: primary retained masks frozen before eval query scoring",
            flush=True,
        )
        if audit_ratios:
            with phase("freeze-audit-retained-masks"):
                for ratio in audit_ratios:
                    if abs(float(ratio) - float(config.model.compression_ratio)) <= 1e-9:
                        continue
                    ratio_key = f"{float(ratio):.4f}"
                    frozen_ratio_methods: list[Method] = []
                    audit_timing_by_method: dict[str, float] = {}
                    for method in retention_methods:
                        with phase(f"  freeze audit {method.name} ratio={ratio:.4f}"):
                            freeze_t0 = time.perf_counter()
                            retained_mask = (
                                method.simplify(
                                    test_points,
                                    test_boundaries,
                                    float(ratio),
                                )
                                .detach()
                                .cpu()
                            )
                            audit_elapsed_seconds = float(time.perf_counter() - freeze_t0)
                            audit_timing_by_method[method.name] = audit_elapsed_seconds
                            frozen_ratio_methods.append(
                                FrozenMaskMethod(
                                    name=method.name,
                                    retained_mask=retained_mask,
                                    latency_ms=float(audit_elapsed_seconds * 1000.0),
                                )
                            )
                    frozen_audit_methods_by_ratio[ratio_key] = frozen_ratio_methods
                    freeze_timing_diagnostics["audit_method_simplify_seconds"][ratio_key] = (
                        audit_timing_by_method
                    )
            print(
                "  workload_blind_protocol=enabled: audit retained masks frozen before eval query scoring",
                flush=True,
            )
        freeze_timing_diagnostics["total_seconds"] = float(time.perf_counter() - freeze_started_at)
        if primary_selector_trace is not None:
            primary_selector_trace["retained_mask_freeze_timing"] = freeze_timing_diagnostics

    return RetainedMaskFreezingOutputs(
        methods=methods,
        frozen_primary_masks=frozen_primary_masks,
        frozen_audit_methods_by_ratio=frozen_audit_methods_by_ratio,
        frozen_primary_scores=frozen_primary_scores,
        frozen_primary_raw_preds=frozen_primary_raw_preds,
        frozen_primary_head_logits=frozen_primary_head_logits,
        frozen_primary_segment_scores=frozen_primary_segment_scores,
        frozen_primary_path_length_support_scores=frozen_primary_path_length_support_scores,
        frozen_primary_selector_segment_scores=frozen_primary_selector_segment_scores,
        primary_selector_trace=primary_selector_trace,
        causality_ablation_methods=causality_ablation_methods,
        causal_ablation_freeze_failures=causal_ablation_freeze_failures,
        prior_sensitivity_diagnostics=prior_sensitivity_diagnostics,
        prior_channel_ablation_diagnostics=prior_channel_ablation_diagnostics,
        head_ablation_sensitivity_diagnostics=head_ablation_sensitivity_diagnostics,
        segment_budget_head_ablation_mode=segment_budget_head_ablation_mode,
        freeze_timing_diagnostics=freeze_timing_diagnostics,
    )
