"""Retained-mask freezing for workload-blind experiment runs."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, cast

import torch

from config.experiment_config import ExperimentConfig, SeedBundle
from evaluation.baselines import FrozenMaskMethod, Method, MLQDSMethod
from orchestration.causality import (
    _head_ablation_sensitivity,
    _prior_feature_sample_sensitivity,
    _score_ablation_sensitivity,
)
from orchestration.length_diagnostics import (
    _score_protected_length_feasibility,
    _score_protected_length_frontier,
)
from orchestration.model_ablations import (
    _raw_predictions_without_factorized_head,
    _reset_module_parameters,
    _scores_without_factorized_head,
    _shuffled_query_prior_field,
)
from orchestration.selector_diagnostics import (
    _learned_segment_frozen_method,
    _neutral_segment_scores_for_ablation,
    _pre_repair_frozen_method_from_trace,
    _segment_score_quantile_bands_for_ablation,
    _segment_score_top_band_for_ablation,
    _selector_segment_score_source_label,
)
from queries.query_types import single_workload_type
from queries.workload import TypedQueryWorkload
from simplification.learned_segment_budget import (
    blend_segment_support_scores,
    simplify_with_learned_segment_budget_v1_with_trace,
)
from training.predictability_audit import query_prior_predictability_scores
from training.query_prior_fields import (
    QUERY_PRIOR_FIELD_NAMES,
    query_prior_field_metadata,
    zero_query_prior_field_channels,
    zero_query_prior_field_like,
)
from training.training_outputs import TrainingOutputs

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


def freeze_workload_blind_retained_masks(
    *,
    methods: list[Method],
    retention_methods: list[Method],
    workload_blind_eval: bool,
    audit_ratios: Sequence[float],
    config: ExperimentConfig,
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
    if workload_blind_eval:
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
                    cast(Any, method).latency_ms = float((time.perf_counter() - freeze_t0) * 1000.0)
                    score_cache = getattr(method, "_score_cache", None)
                    if isinstance(score_cache, torch.Tensor):
                        frozen_primary_scores[method.name] = score_cache.detach().cpu().float()
                    raw_pred_cache = getattr(method, "_raw_pred_cache", None)
                    if isinstance(raw_pred_cache, torch.Tensor):
                        frozen_primary_raw_preds[method.name] = (
                            raw_pred_cache.detach().cpu().float()
                        )
                    head_logit_cache = getattr(method, "_head_logit_cache", None)
                    if isinstance(head_logit_cache, torch.Tensor):
                        frozen_primary_head_logits[method.name] = (
                            head_logit_cache.detach().cpu().float()
                        )
                    segment_score_cache = getattr(method, "_segment_score_cache", None)
                    if isinstance(segment_score_cache, torch.Tensor):
                        frozen_primary_segment_scores[method.name] = (
                            segment_score_cache.detach().cpu().float()
                        )
                    path_length_support_cache = getattr(
                        method, "_path_length_support_score_cache", None
                    )
                    if isinstance(path_length_support_cache, torch.Tensor):
                        frozen_primary_path_length_support_scores[method.name] = (
                            path_length_support_cache.detach().cpu().float()
                        )
                    selector_segment_score_cache = getattr(
                        method, "_selector_segment_score_cache", None
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
                == "learned_segment_budget_v1"
            ):
                primary_segment_scores = frozen_primary_segment_scores.get("MLQDS")
                primary_path_length_support_scores = frozen_primary_path_length_support_scores.get(
                    "MLQDS"
                )
                primary_selector_segment_scores = frozen_primary_selector_segment_scores.get(
                    "MLQDS"
                )
                trace_mask, trace = simplify_with_learned_segment_budget_v1_with_trace(
                    primary_scores,
                    test_boundaries,
                    float(config.model.compression_ratio),
                    segment_scores=primary_selector_segment_scores,
                    segment_point_scores=primary_segment_scores,
                    points=test_points.detach().cpu().float(),
                    geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                    segment_score_point_blend_weight=float(
                        config.model.learned_segment_score_blend_weight
                    ),
                    fairness_preallocation_enabled=bool(
                        config.model.learned_segment_fairness_preallocation
                    ),
                    length_repair_fraction=float(
                        config.model.learned_segment_length_repair_fraction
                    ),
                    segment_score_source_label=_selector_segment_score_source_label(
                        segment_scores=primary_segment_scores,
                        path_length_support_scores=primary_path_length_support_scores,
                        length_support_blend_weight=float(
                            config.model.learned_segment_length_support_blend_weight
                        ),
                    ),
                )
                frozen_mlqds_mask = frozen_primary_masks.get("MLQDS")
                if isinstance(frozen_mlqds_mask, torch.Tensor):
                    trace["retained_mask_matches_frozen_primary"] = bool(
                        torch.equal(trace_mask.detach().cpu(), frozen_mlqds_mask.detach().cpu())
                    )
                    trace["frozen_primary_retained_count"] = int(frozen_mlqds_mask.sum().item())
                compression_for_trace = float(config.model.compression_ratio)
                learned_fraction_min_for_trace = (
                    0.35
                    if compression_for_trace >= 0.10
                    else 0.25
                    if compression_for_trace >= 0.05
                    else 0.0
                )
                if learned_fraction_min_for_trace > 0.0:
                    trace["score_protected_length_feasibility"] = (
                        _score_protected_length_feasibility(
                            scores=primary_scores,
                            points=test_points,
                            boundaries=test_boundaries,
                            compression_ratio=compression_for_trace,
                            learned_slot_fraction_min=learned_fraction_min_for_trace,
                        )
                    )
                    trace["score_protected_length_frontier"] = _score_protected_length_frontier(
                        scores=primary_scores,
                        points=test_points,
                        boundaries=test_boundaries,
                        compression_ratio=compression_for_trace,
                        learned_slot_fraction_min=learned_fraction_min_for_trace,
                    )
                primary_selector_trace = trace
                pre_repair_diagnostic_name = "MLQDS_pre_repair_allocation_diagnostic"
                try:
                    pre_repair_method = _pre_repair_frozen_method_from_trace(
                        name=pre_repair_diagnostic_name,
                        selector_trace=trace,
                        point_count=int(test_points.shape[0]),
                    )
                    causality_ablation_methods.append(pre_repair_method)
                    trace["pre_repair_frozen_method_diagnostic"] = {
                        "available": True,
                        "diagnostic_only": True,
                        "query_free": True,
                        "method_name": pre_repair_diagnostic_name,
                        "source": "selector_trace.pre_repair_retained_mask.indices",
                        "retained_count": int(pre_repair_method.retained_mask.sum().item()),
                    }
                except (
                    Exception
                ) as exc:  # pragma: no cover - optional diagnostic should not gate eval.
                    trace["pre_repair_frozen_method_diagnostic"] = {
                        "available": False,
                        "diagnostic_only": True,
                        "query_free": True,
                        "method_name": pre_repair_diagnostic_name,
                        "reason": "freeze_failed",
                        "error": str(exc),
                    }
                if float(config.model.learned_segment_geometry_gain_weight) > 0.0:
                    try:
                        causality_ablation_methods.append(
                            _learned_segment_frozen_method(
                                name="MLQDS_without_geometry_tie_breaker",
                                scores=primary_scores,
                                boundaries=test_boundaries,
                                compression_ratio=float(config.model.compression_ratio),
                                segment_scores=primary_selector_segment_scores,
                                segment_point_scores=primary_segment_scores,
                                points=test_points,
                                learned_segment_geometry_gain_weight=0.0,
                                learned_segment_score_blend_weight=float(
                                    config.model.learned_segment_score_blend_weight
                                ),
                                learned_segment_fairness_preallocation=bool(
                                    config.model.learned_segment_fairness_preallocation
                                ),
                                learned_segment_length_repair_fraction=float(
                                    config.model.learned_segment_length_repair_fraction
                                ),
                            )
                        )
                    except (
                        Exception
                    ) as exc:  # pragma: no cover - diagnostic should not break final eval.
                        causal_ablation_freeze_failures["MLQDS_without_geometry_tie_breaker"] = str(
                            exc
                        )
                generator = torch.Generator().manual_seed(int(seeds.eval_query_seed) + 91_337)
                shuffled_order = torch.randperm(int(primary_scores.numel()), generator=generator)
                shuffled_scores = primary_scores[shuffled_order]
                shuffled_segment_scores = (
                    primary_selector_segment_scores[shuffled_order]
                    if primary_selector_segment_scores is not None
                    else None
                )
                shuffled_segment_point_scores = (
                    primary_segment_scores[shuffled_order]
                    if primary_segment_scores is not None
                    else None
                )
                causality_ablation_methods.append(
                    _learned_segment_frozen_method(
                        name="MLQDS_shuffled_scores",
                        scores=shuffled_scores,
                        boundaries=test_boundaries,
                        compression_ratio=float(config.model.compression_ratio),
                        segment_scores=shuffled_segment_scores,
                        segment_point_scores=shuffled_segment_point_scores,
                        points=test_points,
                        learned_segment_geometry_gain_weight=float(
                            config.model.learned_segment_geometry_gain_weight
                        ),
                        learned_segment_score_blend_weight=float(
                            config.model.learned_segment_score_blend_weight
                        ),
                        learned_segment_fairness_preallocation=bool(
                            config.model.learned_segment_fairness_preallocation
                        ),
                        learned_segment_length_repair_fraction=float(
                            config.model.learned_segment_length_repair_fraction
                        ),
                    )
                )
                if primary_segment_scores is not None:
                    neutral_segment_scores = _neutral_segment_scores_for_ablation(
                        primary_segment_scores
                    )
                    no_segment_selector_scores = blend_segment_support_scores(
                        segment_scores=neutral_segment_scores,
                        path_length_support_scores=primary_path_length_support_scores,
                        path_length_support_weight=float(
                            config.model.learned_segment_length_support_blend_weight
                        ),
                    )
                    segment_budget_head_ablation_mode = "neutral_constant_segment_scores"
                    segment_budget_ablation_method = _learned_segment_frozen_method(
                        name="MLQDS_without_segment_budget_head",
                        scores=primary_scores,
                        boundaries=test_boundaries,
                        compression_ratio=float(config.model.compression_ratio),
                        segment_scores=no_segment_selector_scores,
                        segment_point_scores=neutral_segment_scores,
                        points=test_points,
                        learned_segment_geometry_gain_weight=float(
                            config.model.learned_segment_geometry_gain_weight
                        ),
                        learned_segment_score_blend_weight=float(
                            config.model.learned_segment_score_blend_weight
                        ),
                        learned_segment_fairness_preallocation=bool(
                            config.model.learned_segment_fairness_preallocation
                        ),
                        learned_segment_length_repair_fraction=float(
                            config.model.learned_segment_length_repair_fraction
                        ),
                    )
                    causality_ablation_methods.append(segment_budget_ablation_method)
                    segment_budget_sensitivity = _head_ablation_sensitivity(
                        primary_scores=primary_scores,
                        ablation_scores=primary_scores,
                        primary_raw_predictions=primary_raw_preds,
                        ablation_raw_predictions=primary_raw_preds,
                        primary_segment_scores=primary_selector_segment_scores,
                        ablation_segment_scores=no_segment_selector_scores,
                        primary_mask=frozen_primary_masks.get("MLQDS"),
                        ablation_mask=segment_budget_ablation_method.retained_mask,
                    )
                    segment_budget_sensitivity["disabled_head_name"] = "segment_budget_target"
                    segment_budget_sensitivity["ablation_mode"] = segment_budget_head_ablation_mode
                    head_ablation_sensitivity_diagnostics["MLQDS_without_segment_budget_head"] = (
                        segment_budget_sensitivity
                    )
                    if primary_selector_segment_scores is not None:
                        segment_allocation_ablation_method = _learned_segment_frozen_method(
                            name="MLQDS_without_segment_budget_allocation_only",
                            scores=primary_scores,
                            boundaries=test_boundaries,
                            compression_ratio=float(config.model.compression_ratio),
                            segment_scores=no_segment_selector_scores,
                            segment_point_scores=primary_segment_scores,
                            points=test_points,
                            learned_segment_geometry_gain_weight=float(
                                config.model.learned_segment_geometry_gain_weight
                            ),
                            learned_segment_score_blend_weight=float(
                                config.model.learned_segment_score_blend_weight
                            ),
                            learned_segment_fairness_preallocation=bool(
                                config.model.learned_segment_fairness_preallocation
                            ),
                            learned_segment_length_repair_fraction=float(
                                config.model.learned_segment_length_repair_fraction
                            ),
                        )
                        causality_ablation_methods.append(segment_allocation_ablation_method)
                        allocation_sensitivity = _head_ablation_sensitivity(
                            primary_scores=primary_scores,
                            ablation_scores=primary_scores,
                            primary_raw_predictions=primary_raw_preds,
                            ablation_raw_predictions=primary_raw_preds,
                            primary_segment_scores=primary_selector_segment_scores,
                            ablation_segment_scores=no_segment_selector_scores,
                            primary_mask=frozen_primary_masks.get("MLQDS"),
                            ablation_mask=segment_allocation_ablation_method.retained_mask,
                        )
                        allocation_sensitivity["disabled_head_name"] = "segment_budget_target"
                        allocation_sensitivity["ablation_mode"] = (
                            "neutral_constant_segment_scores_for_allocation_only"
                        )
                        allocation_sensitivity["diagnostic_only"] = True
                        head_ablation_sensitivity_diagnostics[
                            "MLQDS_without_segment_budget_allocation_only"
                        ] = allocation_sensitivity

                        point_score_allocation_method = _learned_segment_frozen_method(
                            name="MLQDS_point_score_allocation_diagnostic",
                            scores=primary_scores,
                            boundaries=test_boundaries,
                            compression_ratio=float(config.model.compression_ratio),
                            segment_scores=None,
                            segment_point_scores=primary_segment_scores,
                            points=test_points,
                            learned_segment_geometry_gain_weight=float(
                                config.model.learned_segment_geometry_gain_weight
                            ),
                            learned_segment_score_blend_weight=float(
                                config.model.learned_segment_score_blend_weight
                            ),
                            learned_segment_fairness_preallocation=bool(
                                config.model.learned_segment_fairness_preallocation
                            ),
                            learned_segment_length_repair_fraction=float(
                                config.model.learned_segment_length_repair_fraction
                            ),
                        )
                        causality_ablation_methods.append(point_score_allocation_method)
                        point_score_allocation_sensitivity = _head_ablation_sensitivity(
                            primary_scores=primary_scores,
                            ablation_scores=primary_scores,
                            primary_raw_predictions=primary_raw_preds,
                            ablation_raw_predictions=primary_raw_preds,
                            primary_segment_scores=primary_selector_segment_scores,
                            ablation_segment_scores=primary_scores,
                            primary_mask=frozen_primary_masks.get("MLQDS"),
                            ablation_mask=point_score_allocation_method.retained_mask,
                        )
                        point_score_allocation_sensitivity["disabled_head_name"] = (
                            "segment_budget_target"
                        )
                        point_score_allocation_sensitivity["ablation_mode"] = (
                            "point_score_top20_mean_for_allocation_only"
                        )
                        point_score_allocation_sensitivity["diagnostic_only"] = True
                        point_score_allocation_sensitivity["allocation_score_source"] = (
                            "point_score_top20_mean"
                        )
                        head_ablation_sensitivity_diagnostics[
                            "MLQDS_point_score_allocation_diagnostic"
                        ] = point_score_allocation_sensitivity

                        allocation_authority_variants = [
                            (
                                "MLQDS_segment_allocation_top25_band_diagnostic",
                                _segment_score_top_band_for_ablation(
                                    primary_selector_segment_scores,
                                    test_boundaries,
                                    top_fraction=0.25,
                                ),
                                "top25_binary_selector_segment_scores_for_allocation_only",
                            ),
                            (
                                "MLQDS_segment_allocation_top50_band_diagnostic",
                                _segment_score_top_band_for_ablation(
                                    primary_selector_segment_scores,
                                    test_boundaries,
                                    top_fraction=0.50,
                                ),
                                "top50_binary_selector_segment_scores_for_allocation_only",
                            ),
                            (
                                "MLQDS_segment_allocation_quartile_band_diagnostic",
                                _segment_score_quantile_bands_for_ablation(
                                    primary_selector_segment_scores,
                                    test_boundaries,
                                    band_count=4,
                                ),
                                "quartile_banded_selector_segment_scores_for_allocation_only",
                            ),
                        ]
                        for (
                            diagnostic_name,
                            authority_scores,
                            authority_mode,
                        ) in allocation_authority_variants:
                            authority_method = _learned_segment_frozen_method(
                                name=diagnostic_name,
                                scores=primary_scores,
                                boundaries=test_boundaries,
                                compression_ratio=float(config.model.compression_ratio),
                                segment_scores=authority_scores,
                                segment_point_scores=primary_segment_scores,
                                points=test_points,
                                learned_segment_geometry_gain_weight=float(
                                    config.model.learned_segment_geometry_gain_weight
                                ),
                                learned_segment_score_blend_weight=float(
                                    config.model.learned_segment_score_blend_weight
                                ),
                                learned_segment_fairness_preallocation=bool(
                                    config.model.learned_segment_fairness_preallocation
                                ),
                                learned_segment_length_repair_fraction=float(
                                    config.model.learned_segment_length_repair_fraction
                                ),
                            )
                            causality_ablation_methods.append(authority_method)
                            authority_sensitivity = _head_ablation_sensitivity(
                                primary_scores=primary_scores,
                                ablation_scores=primary_scores,
                                primary_raw_predictions=primary_raw_preds,
                                ablation_raw_predictions=primary_raw_preds,
                                primary_segment_scores=primary_selector_segment_scores,
                                ablation_segment_scores=authority_scores,
                                primary_mask=frozen_primary_masks.get("MLQDS"),
                                ablation_mask=authority_method.retained_mask,
                            )
                            authority_sensitivity["disabled_head_name"] = "segment_budget_target"
                            authority_sensitivity["ablation_mode"] = str(authority_mode)
                            authority_sensitivity["diagnostic_only"] = True
                            authority_sensitivity["allocation_authority_diagnostic"] = True
                            authority_sensitivity["allocation_score_source"] = (
                                "selector_segment_score_bands"
                            )
                            head_ablation_sensitivity_diagnostics[diagnostic_name] = (
                                authority_sensitivity
                            )

                        segment_point_blend_ablation_method = _learned_segment_frozen_method(
                            name="MLQDS_without_segment_budget_point_blend_only",
                            scores=primary_scores,
                            boundaries=test_boundaries,
                            compression_ratio=float(config.model.compression_ratio),
                            segment_scores=primary_selector_segment_scores,
                            segment_point_scores=primary_segment_scores,
                            points=test_points,
                            learned_segment_geometry_gain_weight=float(
                                config.model.learned_segment_geometry_gain_weight
                            ),
                            learned_segment_score_blend_weight=0.0,
                            learned_segment_fairness_preallocation=bool(
                                config.model.learned_segment_fairness_preallocation
                            ),
                            learned_segment_length_repair_fraction=float(
                                config.model.learned_segment_length_repair_fraction
                            ),
                        )
                        causality_ablation_methods.append(segment_point_blend_ablation_method)
                        point_blend_sensitivity = _head_ablation_sensitivity(
                            primary_scores=primary_scores,
                            ablation_scores=primary_scores,
                            primary_raw_predictions=primary_raw_preds,
                            ablation_raw_predictions=primary_raw_preds,
                            primary_segment_scores=primary_selector_segment_scores,
                            ablation_segment_scores=primary_selector_segment_scores,
                            primary_mask=frozen_primary_masks.get("MLQDS"),
                            ablation_mask=segment_point_blend_ablation_method.retained_mask,
                        )
                        point_blend_sensitivity["disabled_head_name"] = "segment_budget_target"
                        point_blend_sensitivity["ablation_mode"] = (
                            "disable_segment_score_point_blend_only"
                        )
                        point_blend_sensitivity["diagnostic_only"] = True
                        head_ablation_sensitivity_diagnostics[
                            "MLQDS_without_segment_budget_point_blend_only"
                        ] = point_blend_sensitivity
                    if bool(config.model.learned_segment_fairness_preallocation):
                        causality_ablation_methods.append(
                            _learned_segment_frozen_method(
                                name="MLQDS_without_trajectory_fairness_preallocation",
                                scores=primary_scores,
                                boundaries=test_boundaries,
                                compression_ratio=float(config.model.compression_ratio),
                                segment_scores=primary_selector_segment_scores,
                                segment_point_scores=primary_segment_scores,
                                points=test_points,
                                learned_segment_geometry_gain_weight=float(
                                    config.model.learned_segment_geometry_gain_weight
                                ),
                                learned_segment_score_blend_weight=float(
                                    config.model.learned_segment_score_blend_weight
                                ),
                                learned_segment_fairness_preallocation=False,
                                learned_segment_length_repair_fraction=float(
                                    config.model.learned_segment_length_repair_fraction
                                ),
                            )
                        )
                path_length_support_scores = frozen_primary_path_length_support_scores.get("MLQDS")
                if path_length_support_scores is not None:
                    try:
                        path_length_segment_method = _learned_segment_frozen_method(
                            name="MLQDS_path_length_support_segment_head_diagnostic",
                            scores=primary_scores,
                            boundaries=test_boundaries,
                            compression_ratio=float(config.model.compression_ratio),
                            segment_scores=path_length_support_scores,
                            points=test_points,
                            learned_segment_geometry_gain_weight=float(
                                config.model.learned_segment_geometry_gain_weight
                            ),
                            learned_segment_score_blend_weight=float(
                                config.model.learned_segment_score_blend_weight
                            ),
                            learned_segment_fairness_preallocation=bool(
                                config.model.learned_segment_fairness_preallocation
                            ),
                            learned_segment_length_repair_fraction=float(
                                config.model.learned_segment_length_repair_fraction
                            ),
                        )
                        causality_ablation_methods.append(path_length_segment_method)
                        head_ablation_sensitivity_diagnostics[
                            "MLQDS_path_length_support_segment_head_diagnostic"
                        ] = {
                            **_head_ablation_sensitivity(
                                primary_scores=primary_scores,
                                ablation_scores=primary_scores,
                                primary_raw_predictions=primary_raw_preds,
                                ablation_raw_predictions=primary_raw_preds,
                                primary_segment_scores=primary_selector_segment_scores,
                                ablation_segment_scores=path_length_support_scores,
                                primary_mask=frozen_primary_masks.get("MLQDS"),
                                ablation_mask=path_length_segment_method.retained_mask,
                            ),
                            "diagnostic_only": True,
                            "replacement_head_name": "path_length_support_target",
                            "ablation_mode": "path_length_support_as_segment_scores",
                        }
                        path_length_allocation_method = _learned_segment_frozen_method(
                            name="MLQDS_path_length_support_allocation_only_diagnostic",
                            scores=primary_scores,
                            boundaries=test_boundaries,
                            compression_ratio=float(config.model.compression_ratio),
                            segment_scores=path_length_support_scores,
                            segment_point_scores=primary_segment_scores,
                            points=test_points,
                            learned_segment_geometry_gain_weight=float(
                                config.model.learned_segment_geometry_gain_weight
                            ),
                            learned_segment_score_blend_weight=float(
                                config.model.learned_segment_score_blend_weight
                            ),
                            learned_segment_fairness_preallocation=bool(
                                config.model.learned_segment_fairness_preallocation
                            ),
                            learned_segment_length_repair_fraction=float(
                                config.model.learned_segment_length_repair_fraction
                            ),
                        )
                        causality_ablation_methods.append(path_length_allocation_method)
                        head_ablation_sensitivity_diagnostics[
                            "MLQDS_path_length_support_allocation_only_diagnostic"
                        ] = {
                            **_head_ablation_sensitivity(
                                primary_scores=primary_scores,
                                ablation_scores=primary_scores,
                                primary_raw_predictions=primary_raw_preds,
                                ablation_raw_predictions=primary_raw_preds,
                                primary_segment_scores=primary_selector_segment_scores,
                                ablation_segment_scores=path_length_support_scores,
                                primary_mask=frozen_primary_masks.get("MLQDS"),
                                ablation_mask=path_length_allocation_method.retained_mask,
                            ),
                            "diagnostic_only": True,
                            "replacement_head_name": "path_length_support_target",
                            "ablation_mode": "path_length_support_allocation_only",
                        }
                    except (
                        Exception
                    ) as exc:  # pragma: no cover - diagnostic should not break final eval.
                        head_ablation_sensitivity_diagnostics[
                            "MLQDS_path_length_support_segment_head_diagnostic"
                        ] = {
                            "available": False,
                            "diagnostic_only": True,
                            "reason": "freeze_failed",
                            "error": str(exc),
                        }
                primary_head_logits = frozen_primary_head_logits.get("MLQDS")
                if primary_head_logits is not None:
                    try:
                        behavior_raw_preds = _raw_predictions_without_factorized_head(
                            model=trained.model,
                            head_logits=primary_head_logits,
                            disabled_head_name="conditional_behavior_utility",
                        )
                        behavior_scores = _scores_without_factorized_head(
                            model=trained.model,
                            head_logits=primary_head_logits,
                            disabled_head_name="conditional_behavior_utility",
                            boundaries=test_boundaries,
                            workload_type=single_workload_type(eval_workload_map),
                            score_mode=config.model.mlqds_score_mode,
                            score_temperature=float(config.model.mlqds_score_temperature),
                            rank_confidence_weight=float(config.model.mlqds_rank_confidence_weight),
                        )
                        behavior_ablation_method = _learned_segment_frozen_method(
                            name="MLQDS_without_behavior_utility_head",
                            scores=behavior_scores,
                            boundaries=test_boundaries,
                            compression_ratio=float(config.model.compression_ratio),
                            segment_scores=primary_selector_segment_scores,
                            segment_point_scores=primary_segment_scores,
                            points=test_points,
                            learned_segment_geometry_gain_weight=float(
                                config.model.learned_segment_geometry_gain_weight
                            ),
                            learned_segment_score_blend_weight=float(
                                config.model.learned_segment_score_blend_weight
                            ),
                            learned_segment_fairness_preallocation=bool(
                                config.model.learned_segment_fairness_preallocation
                            ),
                            learned_segment_length_repair_fraction=float(
                                config.model.learned_segment_length_repair_fraction
                            ),
                        )
                        causality_ablation_methods.append(behavior_ablation_method)
                        behavior_sensitivity = _head_ablation_sensitivity(
                            primary_scores=primary_scores,
                            ablation_scores=behavior_scores,
                            primary_raw_predictions=primary_raw_preds,
                            ablation_raw_predictions=behavior_raw_preds,
                            primary_segment_scores=primary_selector_segment_scores,
                            ablation_segment_scores=primary_selector_segment_scores,
                            primary_mask=frozen_primary_masks.get("MLQDS"),
                            ablation_mask=behavior_ablation_method.retained_mask,
                        )
                        behavior_sensitivity["disabled_head_name"] = "conditional_behavior_utility"
                        behavior_sensitivity["ablation_mode"] = "neutral_multiplicative_head"
                        head_ablation_sensitivity_diagnostics[
                            "MLQDS_without_behavior_utility_head"
                        ] = behavior_sensitivity
                    except (
                        Exception
                    ) as exc:  # pragma: no cover - diagnostic should not break final eval.
                        causal_ablation_freeze_failures["MLQDS_without_behavior_utility_head"] = (
                            str(exc)
                        )
                try:
                    untrained_model = _reset_module_parameters(
                        trained.model,
                        seed=int(seeds.torch_seed) + 44_021,
                    )
                    untrained_outputs = TrainingOutputs(
                        model=untrained_model,
                        scaler=trained.scaler,
                        labels=trained.labels,
                        labelled_mask=trained.labelled_mask,
                        history=[],
                        feature_context=dict(trained.feature_context),
                    )
                    untrained_method = MLQDSMethod(
                        name="MLQDS_untrained_model",
                        trained=untrained_outputs,
                        workload=eval_workload,
                        workload_type=single_workload_type(eval_workload_map),
                        score_mode=config.model.mlqds_score_mode,
                        score_temperature=config.model.mlqds_score_temperature,
                        rank_confidence_weight=config.model.mlqds_rank_confidence_weight,
                        temporal_fraction=config.model.mlqds_temporal_fraction,
                        diversity_bonus=config.model.mlqds_diversity_bonus,
                        hybrid_mode=config.model.mlqds_hybrid_mode,
                        stratified_center_weight=config.model.mlqds_stratified_center_weight,
                        min_learned_swaps=config.model.mlqds_min_learned_swaps,
                        selector_type=config.model.selector_type,
                        trajectory_mmsis=test_mmsis,
                        inference_device=None,
                        amp_mode=config.model.amp_mode,
                        inference_batch_size=config.model.inference_batch_size,
                        learned_segment_geometry_gain_weight=config.model.learned_segment_geometry_gain_weight,
                        learned_segment_score_blend_weight=config.model.learned_segment_score_blend_weight,
                        learned_segment_fairness_preallocation=config.model.learned_segment_fairness_preallocation,
                        learned_segment_length_repair_fraction=config.model.learned_segment_length_repair_fraction,
                        learned_segment_length_support_blend_weight=(
                            config.model.learned_segment_length_support_blend_weight
                        ),
                    )
                    untrained_mask = untrained_method.simplify(
                        test_points,
                        test_boundaries,
                        float(config.model.compression_ratio),
                    )
                    causality_ablation_methods.append(
                        FrozenMaskMethod(
                            name="MLQDS_untrained_model",
                            retained_mask=untrained_mask.detach().cpu(),
                        )
                    )
                except (
                    Exception
                ) as exc:  # pragma: no cover - diagnostic should not break final eval.
                    causal_ablation_freeze_failures["MLQDS_untrained_model"] = str(exc)
                query_prior_field = trained.feature_context.get("query_prior_field")
                if isinstance(query_prior_field, dict):
                    prior_scores = (
                        query_prior_predictability_scores(test_points, query_prior_field)
                        .detach()
                        .cpu()
                    )
                    causality_ablation_methods.append(
                        _learned_segment_frozen_method(
                            name="MLQDS_prior_field_only_score",
                            scores=prior_scores,
                            boundaries=test_boundaries,
                            compression_ratio=float(config.model.compression_ratio),
                            points=test_points,
                            learned_segment_geometry_gain_weight=float(
                                config.model.learned_segment_geometry_gain_weight
                            ),
                            learned_segment_score_blend_weight=float(
                                config.model.learned_segment_score_blend_weight
                            ),
                            learned_segment_fairness_preallocation=bool(
                                config.model.learned_segment_fairness_preallocation
                            ),
                            learned_segment_length_repair_fraction=float(
                                config.model.learned_segment_length_repair_fraction
                            ),
                        )
                    )
                    try:
                        shuffled_prior_field = _shuffled_query_prior_field(
                            query_prior_field,
                            seed=int(seeds.eval_query_seed) + 71_003,
                        )
                        shuffled_prior_feature_sensitivity = _prior_feature_sample_sensitivity(
                            points=test_points,
                            primary_prior_field=query_prior_field,
                            ablation_prior_field=shuffled_prior_field,
                        )
                        shuffled_prior_trained = TrainingOutputs(
                            model=trained.model,
                            scaler=trained.scaler,
                            labels=trained.labels,
                            labelled_mask=trained.labelled_mask,
                            history=trained.history,
                            epochs_trained=trained.epochs_trained,
                            best_epoch=trained.best_epoch,
                            best_loss=trained.best_loss,
                            best_selection_score=trained.best_selection_score,
                            target_diagnostics=trained.target_diagnostics,
                            fit_diagnostics=trained.fit_diagnostics,
                            feature_context={
                                **trained.feature_context,
                                "query_prior_field": shuffled_prior_field,
                            },
                        )
                        shuffled_prior_method = MLQDSMethod(
                            name="MLQDS_shuffled_prior_fields",
                            trained=shuffled_prior_trained,
                            workload=eval_workload,
                            workload_type=single_workload_type(eval_workload_map),
                            score_mode=config.model.mlqds_score_mode,
                            score_temperature=config.model.mlqds_score_temperature,
                            rank_confidence_weight=config.model.mlqds_rank_confidence_weight,
                            temporal_fraction=config.model.mlqds_temporal_fraction,
                            diversity_bonus=config.model.mlqds_diversity_bonus,
                            hybrid_mode=config.model.mlqds_hybrid_mode,
                            stratified_center_weight=config.model.mlqds_stratified_center_weight,
                            min_learned_swaps=config.model.mlqds_min_learned_swaps,
                            selector_type=config.model.selector_type,
                            trajectory_mmsis=test_mmsis,
                            inference_device=None,
                            amp_mode=config.model.amp_mode,
                            inference_batch_size=config.model.inference_batch_size,
                            learned_segment_geometry_gain_weight=config.model.learned_segment_geometry_gain_weight,
                            learned_segment_score_blend_weight=config.model.learned_segment_score_blend_weight,
                            learned_segment_fairness_preallocation=config.model.learned_segment_fairness_preallocation,
                            learned_segment_length_repair_fraction=config.model.learned_segment_length_repair_fraction,
                            learned_segment_length_support_blend_weight=(
                                config.model.learned_segment_length_support_blend_weight
                            ),
                        )
                        shuffled_prior_mask = shuffled_prior_method.simplify(
                            test_points,
                            test_boundaries,
                            float(config.model.compression_ratio),
                        )
                        shuffled_prior_scores = getattr(shuffled_prior_method, "_score_cache", None)
                        shuffled_prior_raw_preds = getattr(
                            shuffled_prior_method, "_raw_pred_cache", None
                        )
                        score_sensitivity = _score_ablation_sensitivity(
                            primary_scores=primary_scores,
                            ablation_scores=shuffled_prior_scores
                            if isinstance(shuffled_prior_scores, torch.Tensor)
                            else None,
                            primary_mask=frozen_primary_masks.get("MLQDS"),
                            ablation_mask=shuffled_prior_mask,
                        )
                        raw_sensitivity = _score_ablation_sensitivity(
                            primary_scores=primary_raw_preds,
                            ablation_scores=(
                                shuffled_prior_raw_preds
                                if isinstance(shuffled_prior_raw_preds, torch.Tensor)
                                else None
                            ),
                            primary_mask=frozen_primary_masks.get("MLQDS"),
                            ablation_mask=shuffled_prior_mask,
                        )
                        prior_sensitivity_diagnostics["shuffled_prior_fields"] = {
                            "sampled_prior_features": shuffled_prior_feature_sensitivity,
                            "selector_score": score_sensitivity,
                            "raw_prediction": raw_sensitivity,
                        }
                        causality_ablation_methods.append(
                            FrozenMaskMethod(
                                name="MLQDS_shuffled_prior_fields",
                                retained_mask=shuffled_prior_mask.detach().cpu(),
                            )
                        )
                    except (
                        Exception
                    ) as exc:  # pragma: no cover - diagnostic should not break final eval.
                        causal_ablation_freeze_failures["MLQDS_shuffled_prior_fields"] = str(exc)
                    try:
                        zero_prior_field = zero_query_prior_field_like(query_prior_field)
                        zero_prior_feature_sensitivity = _prior_feature_sample_sensitivity(
                            points=test_points,
                            primary_prior_field=query_prior_field,
                            ablation_prior_field=zero_prior_field,
                        )
                        zero_prior_trained = TrainingOutputs(
                            model=trained.model,
                            scaler=trained.scaler,
                            labels=trained.labels,
                            labelled_mask=trained.labelled_mask,
                            history=trained.history,
                            epochs_trained=trained.epochs_trained,
                            best_epoch=trained.best_epoch,
                            best_loss=trained.best_loss,
                            best_selection_score=trained.best_selection_score,
                            target_diagnostics=trained.target_diagnostics,
                            fit_diagnostics=trained.fit_diagnostics,
                            feature_context={
                                **trained.feature_context,
                                "query_prior_field": zero_prior_field,
                                "query_prior_field_metadata": query_prior_field_metadata(
                                    zero_prior_field
                                ),
                            },
                        )
                        zero_prior_method = MLQDSMethod(
                            name="MLQDS_without_query_prior_features",
                            trained=zero_prior_trained,
                            workload=eval_workload,
                            workload_type=single_workload_type(eval_workload_map),
                            score_mode=config.model.mlqds_score_mode,
                            score_temperature=config.model.mlqds_score_temperature,
                            rank_confidence_weight=config.model.mlqds_rank_confidence_weight,
                            temporal_fraction=config.model.mlqds_temporal_fraction,
                            diversity_bonus=config.model.mlqds_diversity_bonus,
                            hybrid_mode=config.model.mlqds_hybrid_mode,
                            stratified_center_weight=config.model.mlqds_stratified_center_weight,
                            min_learned_swaps=config.model.mlqds_min_learned_swaps,
                            selector_type=config.model.selector_type,
                            trajectory_mmsis=test_mmsis,
                            inference_device=None,
                            amp_mode=config.model.amp_mode,
                            inference_batch_size=config.model.inference_batch_size,
                            learned_segment_geometry_gain_weight=config.model.learned_segment_geometry_gain_weight,
                            learned_segment_score_blend_weight=config.model.learned_segment_score_blend_weight,
                            learned_segment_fairness_preallocation=config.model.learned_segment_fairness_preallocation,
                            learned_segment_length_repair_fraction=config.model.learned_segment_length_repair_fraction,
                            learned_segment_length_support_blend_weight=(
                                config.model.learned_segment_length_support_blend_weight
                            ),
                        )
                        zero_prior_mask = zero_prior_method.simplify(
                            test_points,
                            test_boundaries,
                            float(config.model.compression_ratio),
                        )
                        zero_prior_scores = getattr(zero_prior_method, "_score_cache", None)
                        zero_prior_raw_preds = getattr(zero_prior_method, "_raw_pred_cache", None)
                        score_sensitivity = _score_ablation_sensitivity(
                            primary_scores=primary_scores,
                            ablation_scores=zero_prior_scores
                            if isinstance(zero_prior_scores, torch.Tensor)
                            else None,
                            primary_mask=frozen_primary_masks.get("MLQDS"),
                            ablation_mask=zero_prior_mask,
                        )
                        raw_sensitivity = _score_ablation_sensitivity(
                            primary_scores=primary_raw_preds,
                            ablation_scores=zero_prior_raw_preds
                            if isinstance(zero_prior_raw_preds, torch.Tensor)
                            else None,
                            primary_mask=frozen_primary_masks.get("MLQDS"),
                            ablation_mask=zero_prior_mask,
                        )
                        prior_sensitivity_diagnostics["without_query_prior_features"] = {
                            "sampled_prior_features": zero_prior_feature_sensitivity,
                            "selector_score": score_sensitivity,
                            "raw_prediction": raw_sensitivity,
                        }
                        causality_ablation_methods.append(
                            FrozenMaskMethod(
                                name="MLQDS_without_query_prior_features",
                                retained_mask=zero_prior_mask.detach().cpu(),
                            )
                        )
                    except (
                        Exception
                    ) as exc:  # pragma: no cover - diagnostic should not break final eval.
                        causal_ablation_freeze_failures["MLQDS_without_query_prior_features"] = str(
                            exc
                        )
                    for prior_channel_name in QUERY_PRIOR_FIELD_NAMES:
                        channel_method_name = f"MLQDS_without_prior_channel_{prior_channel_name}"
                        try:
                            channel_prior_field = zero_query_prior_field_channels(
                                query_prior_field,
                                [prior_channel_name],
                            )
                            channel_feature_sensitivity = _prior_feature_sample_sensitivity(
                                points=test_points,
                                primary_prior_field=query_prior_field,
                                ablation_prior_field=channel_prior_field,
                            )
                            channel_trained = TrainingOutputs(
                                model=trained.model,
                                scaler=trained.scaler,
                                labels=trained.labels,
                                labelled_mask=trained.labelled_mask,
                                history=trained.history,
                                epochs_trained=trained.epochs_trained,
                                best_epoch=trained.best_epoch,
                                best_loss=trained.best_loss,
                                best_selection_score=trained.best_selection_score,
                                target_diagnostics=trained.target_diagnostics,
                                fit_diagnostics=trained.fit_diagnostics,
                                feature_context={
                                    **trained.feature_context,
                                    "query_prior_field": channel_prior_field,
                                    "query_prior_field_metadata": query_prior_field_metadata(
                                        channel_prior_field
                                    ),
                                },
                            )
                            channel_method = MLQDSMethod(
                                name=channel_method_name,
                                trained=channel_trained,
                                workload=eval_workload,
                                workload_type=single_workload_type(eval_workload_map),
                                score_mode=config.model.mlqds_score_mode,
                                score_temperature=config.model.mlqds_score_temperature,
                                rank_confidence_weight=config.model.mlqds_rank_confidence_weight,
                                temporal_fraction=config.model.mlqds_temporal_fraction,
                                diversity_bonus=config.model.mlqds_diversity_bonus,
                                hybrid_mode=config.model.mlqds_hybrid_mode,
                                stratified_center_weight=config.model.mlqds_stratified_center_weight,
                                min_learned_swaps=config.model.mlqds_min_learned_swaps,
                                selector_type=config.model.selector_type,
                                trajectory_mmsis=test_mmsis,
                                inference_device=None,
                                amp_mode=config.model.amp_mode,
                                inference_batch_size=config.model.inference_batch_size,
                                learned_segment_geometry_gain_weight=config.model.learned_segment_geometry_gain_weight,
                                learned_segment_score_blend_weight=config.model.learned_segment_score_blend_weight,
                                learned_segment_fairness_preallocation=config.model.learned_segment_fairness_preallocation,
                                learned_segment_length_repair_fraction=config.model.learned_segment_length_repair_fraction,
                                learned_segment_length_support_blend_weight=(
                                    config.model.learned_segment_length_support_blend_weight
                                ),
                            )
                            channel_mask = channel_method.simplify(
                                test_points,
                                test_boundaries,
                                float(config.model.compression_ratio),
                            )
                            channel_scores = getattr(channel_method, "_score_cache", None)
                            channel_raw_preds = getattr(channel_method, "_raw_pred_cache", None)
                            prior_channel_ablation_diagnostics[prior_channel_name] = {
                                "available": True,
                                "method_name": channel_method_name,
                                "sampled_prior_features": channel_feature_sensitivity,
                                "selector_score": _score_ablation_sensitivity(
                                    primary_scores=primary_scores,
                                    ablation_scores=channel_scores
                                    if isinstance(channel_scores, torch.Tensor)
                                    else None,
                                    primary_mask=frozen_primary_masks.get("MLQDS"),
                                    ablation_mask=channel_mask,
                                ),
                                "raw_prediction": _score_ablation_sensitivity(
                                    primary_scores=primary_raw_preds,
                                    ablation_scores=(
                                        channel_raw_preds
                                        if isinstance(channel_raw_preds, torch.Tensor)
                                        else None
                                    ),
                                    primary_mask=frozen_primary_masks.get("MLQDS"),
                                    ablation_mask=channel_mask,
                                ),
                            }
                            causality_ablation_methods.append(
                                FrozenMaskMethod(
                                    name=channel_method_name,
                                    retained_mask=channel_mask.detach().cpu(),
                                )
                            )
                        except Exception as exc:  # pragma: no cover - optional diagnostic only.
                            prior_channel_ablation_diagnostics[prior_channel_name] = {
                                "available": False,
                                "method_name": channel_method_name,
                                "error": str(exc),
                            }
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
                            frozen_ratio_methods.append(
                                FrozenMaskMethod(
                                    name=method.name,
                                    retained_mask=retained_mask,
                                    latency_ms=float((time.perf_counter() - freeze_t0) * 1000.0),
                                )
                            )
                    frozen_audit_methods_by_ratio[ratio_key] = frozen_ratio_methods
            print(
                "  workload_blind_protocol=enabled: audit retained masks frozen before eval query scoring",
                flush=True,
            )

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
    )
