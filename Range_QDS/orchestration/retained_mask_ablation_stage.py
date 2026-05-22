"""Query-free ablation retained-mask freezing for workload-blind runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from config.run_config import RunConfig, SeedBundle
from learning.outputs import TrainingOutputs
from learning.predictability_audit import query_prior_predictability_scores
from learning.query_prior_fields import (
    QUERY_PRIOR_FIELD_NAMES,
    zero_query_prior_field_channels,
    zero_query_prior_field_like,
)
from learning.targets.query_local_utility import QUERY_LOCAL_UTILITY_HEAD_NAMES
from orchestration.causality import (
    head_ablation_sensitivity,
    model_prior_feature_sensitivity,
    prior_ablation_sensitivity_from_tensors,
    prior_feature_sample_sensitivity,
    training_outputs_with_query_prior_field,
)
from orchestration.learned_segment_freezing import learned_segment_frozen_method_from_config
from orchestration.mlqds_method_factory import build_mlqds_method
from orchestration.model_ablations import (
    raw_predictions_without_factorized_head,
    reset_module_parameters,
    scores_without_factorized_head,
    shuffled_query_prior_field,
)
from orchestration.selector_diagnostics import (
    neutral_segment_scores_for_ablation,
    pre_repair_frozen_method_from_trace,
    segment_score_quantile_bands_for_ablation,
    segment_score_top_band_for_ablation,
)
from scoring.methods import FrozenMaskMethod
from selection.learned_segment_budget import blend_segment_support_scores
from workloads.query_types import single_workload_type
from workloads.typed_workload import TypedQueryWorkload


@dataclass
class RetainedMaskAblationOutputs:
    """Frozen query-free ablation methods and diagnostics."""

    primary_selector_trace: dict[str, Any]
    causality_ablation_methods: list[FrozenMaskMethod]
    causal_ablation_freeze_failures: dict[str, str]
    prior_sensitivity_diagnostics: dict[str, Any]
    prior_channel_ablation_diagnostics: dict[str, Any]
    head_ablation_sensitivity_diagnostics: dict[str, Any]
    segment_budget_head_ablation_mode: str | None
    freeze_timing_diagnostics: dict[str, Any]


def _initial_freeze_timing_diagnostics() -> dict[str, Any]:
    return {
        "available": True,
        "diagnostic_only": True,
        "query_free": True,
        "stage": "freeze_retained_mask_ablations",
        "substage_seconds": {},
        "prior_channel_seconds": {},
    }


@dataclass(frozen=True)
class _RetainedMaskAblationContext:
    config: RunConfig
    trained: TrainingOutputs
    eval_workload: TypedQueryWorkload
    eval_workload_map: dict[str, float]
    test_mmsis: list[int] | None
    test_points: torch.Tensor
    test_boundaries: list[tuple[int, int]]
    seeds: SeedBundle
    frozen_primary_masks: dict[str, torch.Tensor]
    primary_scores: torch.Tensor
    primary_raw_preds: torch.Tensor | None
    primary_segment_scores: torch.Tensor | None
    primary_path_length_support_scores: torch.Tensor | None
    primary_selector_segment_scores: torch.Tensor | None
    primary_head_logits: torch.Tensor | None

    def selector_method(self, **kwargs: Any) -> FrozenMaskMethod:
        kwargs.setdefault("boundaries", self.test_boundaries)
        kwargs.setdefault("points", self.test_points)
        return learned_segment_frozen_method_from_config(config=self.config, **kwargs)


@dataclass
class _RetainedMaskAblationState:
    trace: dict[str, Any]
    stage_started_at: float = field(default_factory=time.perf_counter)
    causality_ablation_methods: list[FrozenMaskMethod] = field(default_factory=list)
    causal_ablation_freeze_failures: dict[str, str] = field(default_factory=dict)
    prior_sensitivity_diagnostics: dict[str, Any] = field(default_factory=dict)
    prior_channel_ablation_diagnostics: dict[str, Any] = field(default_factory=dict)
    head_ablation_sensitivity_diagnostics: dict[str, Any] = field(default_factory=dict)
    segment_budget_head_ablation_mode: str | None = None
    freeze_timing_diagnostics: dict[str, Any] = field(
        default_factory=_initial_freeze_timing_diagnostics
    )

    def record_substage(self, name: str, started_at: float) -> None:
        substage_seconds = self.freeze_timing_diagnostics["substage_seconds"]
        previous = float(substage_seconds.get(name, 0.0))
        substage_seconds[name] = previous + float(time.perf_counter() - started_at)

    def record_prior_channel(self, name: str, started_at: float) -> None:
        self.freeze_timing_diagnostics["prior_channel_seconds"][name] = float(
            time.perf_counter() - started_at
        )

    def outputs(self) -> RetainedMaskAblationOutputs:
        self.freeze_timing_diagnostics["method_count"] = len(self.causality_ablation_methods)
        self.freeze_timing_diagnostics["failure_count"] = len(
            self.causal_ablation_freeze_failures
        )
        self.freeze_timing_diagnostics["total_seconds"] = float(
            time.perf_counter() - self.stage_started_at
        )
        self.trace["retained_mask_ablation_freeze_timing"] = self.freeze_timing_diagnostics
        return RetainedMaskAblationOutputs(
            primary_selector_trace=self.trace,
            causality_ablation_methods=self.causality_ablation_methods,
            causal_ablation_freeze_failures=self.causal_ablation_freeze_failures,
            prior_sensitivity_diagnostics=self.prior_sensitivity_diagnostics,
            prior_channel_ablation_diagnostics=self.prior_channel_ablation_diagnostics,
            head_ablation_sensitivity_diagnostics=self.head_ablation_sensitivity_diagnostics,
            segment_budget_head_ablation_mode=self.segment_budget_head_ablation_mode,
            freeze_timing_diagnostics=self.freeze_timing_diagnostics,
        )


def _factorized_head_scores_from_logits(
    head_logits: torch.Tensor | None,
    head_name: str,
) -> torch.Tensor | None:
    """Return raw factorized-head logits as query-free point scores."""
    if head_logits is None:
        return None
    head_names = tuple(str(name) for name in QUERY_LOCAL_UTILITY_HEAD_NAMES)
    if head_name not in head_names:
        return None
    head_idx = head_names.index(head_name)
    if head_logits.ndim != 2 or int(head_logits.shape[1]) <= head_idx:
        return None
    return head_logits[:, head_idx].detach().cpu().float().contiguous()


def _freeze_segment_budget_ablations(
    ablation_context: _RetainedMaskAblationContext,
    state: _RetainedMaskAblationState,
) -> None:
    if ablation_context.primary_segment_scores is None:
        return

    substage_started_at = time.perf_counter()
    neutral_segment_scores = neutral_segment_scores_for_ablation(
        ablation_context.primary_segment_scores
    )
    no_segment_selector_scores = blend_segment_support_scores(
        segment_scores=neutral_segment_scores,
        path_length_support_scores=ablation_context.primary_path_length_support_scores,
        path_length_support_weight=float(
            ablation_context.config.model.learned_segment_length_support_blend_weight
        ),
    )
    state.segment_budget_head_ablation_mode = "neutral_constant_segment_scores"
    segment_budget_ablation_method = ablation_context.selector_method(
        name="MLQDS_without_segment_budget_head",
        scores=ablation_context.primary_scores,
        segment_scores=no_segment_selector_scores,
        segment_point_scores=neutral_segment_scores,
    )
    state.causality_ablation_methods.append(segment_budget_ablation_method)
    segment_budget_sensitivity = head_ablation_sensitivity(
        primary_scores=ablation_context.primary_scores,
        ablation_scores=ablation_context.primary_scores,
        primary_raw_predictions=ablation_context.primary_raw_preds,
        ablation_raw_predictions=ablation_context.primary_raw_preds,
        primary_segment_scores=ablation_context.primary_selector_segment_scores,
        ablation_segment_scores=no_segment_selector_scores,
        primary_mask=ablation_context.frozen_primary_masks.get("MLQDS"),
        ablation_mask=segment_budget_ablation_method.retained_mask,
    )
    segment_budget_sensitivity["disabled_head_name"] = "segment_budget_target"
    segment_budget_sensitivity["ablation_mode"] = state.segment_budget_head_ablation_mode
    state.head_ablation_sensitivity_diagnostics["MLQDS_without_segment_budget_head"] = (
        segment_budget_sensitivity
    )
    if ablation_context.primary_selector_segment_scores is not None:
        segment_allocation_ablation_method = ablation_context.selector_method(
            name="MLQDS_without_segment_budget_allocation_only",
            scores=ablation_context.primary_scores,
            segment_scores=no_segment_selector_scores,
            segment_point_scores=ablation_context.primary_segment_scores,
        )
        state.causality_ablation_methods.append(segment_allocation_ablation_method)
        allocation_sensitivity = head_ablation_sensitivity(
            primary_scores=ablation_context.primary_scores,
            ablation_scores=ablation_context.primary_scores,
            primary_raw_predictions=ablation_context.primary_raw_preds,
            ablation_raw_predictions=ablation_context.primary_raw_preds,
            primary_segment_scores=ablation_context.primary_selector_segment_scores,
            ablation_segment_scores=no_segment_selector_scores,
            primary_mask=ablation_context.frozen_primary_masks.get("MLQDS"),
            ablation_mask=segment_allocation_ablation_method.retained_mask,
        )
        allocation_sensitivity["disabled_head_name"] = "segment_budget_target"
        allocation_sensitivity["ablation_mode"] = (
            "neutral_constant_segment_scores_for_allocation_only"
        )
        allocation_sensitivity["diagnostic_only"] = True
        state.head_ablation_sensitivity_diagnostics[
            "MLQDS_without_segment_budget_allocation_only"
        ] = allocation_sensitivity

        uniform_segment_allocation_method = ablation_context.selector_method(
            name="MLQDS_uniform_segment_allocation_only_diagnostic",
            scores=ablation_context.primary_scores,
            segment_scores=neutral_segment_scores,
            segment_point_scores=ablation_context.primary_segment_scores,
            learned_segment_allocation_length_support_weight=0.0,
        )
        state.causality_ablation_methods.append(uniform_segment_allocation_method)
        uniform_allocation_sensitivity = head_ablation_sensitivity(
            primary_scores=ablation_context.primary_scores,
            ablation_scores=ablation_context.primary_scores,
            primary_raw_predictions=ablation_context.primary_raw_preds,
            ablation_raw_predictions=ablation_context.primary_raw_preds,
            primary_segment_scores=ablation_context.primary_selector_segment_scores,
            ablation_segment_scores=neutral_segment_scores,
            primary_mask=ablation_context.frozen_primary_masks.get("MLQDS"),
            ablation_mask=uniform_segment_allocation_method.retained_mask,
        )
        uniform_allocation_sensitivity["disabled_head_name"] = "segment_budget_target"
        uniform_allocation_sensitivity["ablation_mode"] = (
            "neutral_constant_segment_scores_without_length_support_for_allocation_only"
        )
        uniform_allocation_sensitivity["diagnostic_only"] = True
        uniform_allocation_sensitivity["allocation_score_source"] = (
            "uniform_segment_scores_no_length_support"
        )
        state.head_ablation_sensitivity_diagnostics[
            "MLQDS_uniform_segment_allocation_only_diagnostic"
        ] = uniform_allocation_sensitivity

        point_score_allocation_method = ablation_context.selector_method(
            name="MLQDS_point_score_allocation_diagnostic",
            scores=ablation_context.primary_scores,
            segment_scores=None,
            segment_point_scores=ablation_context.primary_segment_scores,
        )
        state.causality_ablation_methods.append(point_score_allocation_method)
        point_score_allocation_sensitivity = head_ablation_sensitivity(
            primary_scores=ablation_context.primary_scores,
            ablation_scores=ablation_context.primary_scores,
            primary_raw_predictions=ablation_context.primary_raw_preds,
            ablation_raw_predictions=ablation_context.primary_raw_preds,
            primary_segment_scores=ablation_context.primary_selector_segment_scores,
            ablation_segment_scores=ablation_context.primary_scores,
            primary_mask=ablation_context.frozen_primary_masks.get("MLQDS"),
            ablation_mask=point_score_allocation_method.retained_mask,
        )
        point_score_allocation_sensitivity["disabled_head_name"] = "segment_budget_target"
        point_score_allocation_sensitivity["ablation_mode"] = (
            "point_score_top20_mean_for_allocation_only"
        )
        point_score_allocation_sensitivity["diagnostic_only"] = True
        point_score_allocation_sensitivity["allocation_score_source"] = "point_score_top20_mean"
        state.head_ablation_sensitivity_diagnostics["MLQDS_point_score_allocation_diagnostic"] = (
            point_score_allocation_sensitivity
        )

        allocation_authority_variants = [
            (
                "MLQDS_segment_allocation_top25_band_diagnostic",
                segment_score_top_band_for_ablation(
                    ablation_context.primary_selector_segment_scores,
                    ablation_context.test_boundaries,
                    top_fraction=0.25,
                ),
                "top25_binary_selector_segment_scores_for_allocation_only",
            ),
            (
                "MLQDS_segment_allocation_top50_band_diagnostic",
                segment_score_top_band_for_ablation(
                    ablation_context.primary_selector_segment_scores,
                    ablation_context.test_boundaries,
                    top_fraction=0.50,
                ),
                "top50_binary_selector_segment_scores_for_allocation_only",
            ),
            (
                "MLQDS_segment_allocation_quartile_band_diagnostic",
                segment_score_quantile_bands_for_ablation(
                    ablation_context.primary_selector_segment_scores,
                    ablation_context.test_boundaries,
                    band_count=4,
                ),
                "quartile_banded_selector_segment_scores_for_allocation_only",
            ),
        ]
        for diagnostic_name, authority_scores, authority_mode in allocation_authority_variants:
            authority_method = ablation_context.selector_method(
                name=diagnostic_name,
                scores=ablation_context.primary_scores,
                segment_scores=authority_scores,
                segment_point_scores=ablation_context.primary_segment_scores,
            )
            state.causality_ablation_methods.append(authority_method)
            authority_sensitivity = head_ablation_sensitivity(
                primary_scores=ablation_context.primary_scores,
                ablation_scores=ablation_context.primary_scores,
                primary_raw_predictions=ablation_context.primary_raw_preds,
                ablation_raw_predictions=ablation_context.primary_raw_preds,
                primary_segment_scores=ablation_context.primary_selector_segment_scores,
                ablation_segment_scores=authority_scores,
                primary_mask=ablation_context.frozen_primary_masks.get("MLQDS"),
                ablation_mask=authority_method.retained_mask,
            )
            authority_sensitivity["disabled_head_name"] = "segment_budget_target"
            authority_sensitivity["ablation_mode"] = str(authority_mode)
            authority_sensitivity["diagnostic_only"] = True
            authority_sensitivity["allocation_authority_diagnostic"] = True
            authority_sensitivity["allocation_score_source"] = "selector_segment_score_bands"
            state.head_ablation_sensitivity_diagnostics[diagnostic_name] = authority_sensitivity

        segment_point_blend_ablation_method = ablation_context.selector_method(
            name="MLQDS_without_segment_budget_point_blend_only",
            scores=ablation_context.primary_scores,
            segment_scores=ablation_context.primary_selector_segment_scores,
            segment_point_scores=ablation_context.primary_segment_scores,
            learned_segment_score_blend_weight=0.0,
        )
        state.causality_ablation_methods.append(segment_point_blend_ablation_method)
        point_blend_sensitivity = head_ablation_sensitivity(
            primary_scores=ablation_context.primary_scores,
            ablation_scores=ablation_context.primary_scores,
            primary_raw_predictions=ablation_context.primary_raw_preds,
            ablation_raw_predictions=ablation_context.primary_raw_preds,
            primary_segment_scores=ablation_context.primary_selector_segment_scores,
            ablation_segment_scores=ablation_context.primary_selector_segment_scores,
            primary_mask=ablation_context.frozen_primary_masks.get("MLQDS"),
            ablation_mask=segment_point_blend_ablation_method.retained_mask,
        )
        point_blend_sensitivity["disabled_head_name"] = "segment_budget_target"
        point_blend_sensitivity["ablation_mode"] = "disable_segment_score_point_blend_only"
        point_blend_sensitivity["diagnostic_only"] = True
        state.head_ablation_sensitivity_diagnostics[
            "MLQDS_without_segment_budget_point_blend_only"
        ] = point_blend_sensitivity
    state.record_substage("segment_budget_head_and_allocation_ablations", substage_started_at)
    if bool(ablation_context.config.model.learned_segment_fairness_preallocation):
        substage_started_at = time.perf_counter()
        state.causality_ablation_methods.append(
            ablation_context.selector_method(
                name="MLQDS_without_trajectory_fairness_preallocation",
                scores=ablation_context.primary_scores,
                segment_scores=ablation_context.primary_selector_segment_scores,
                segment_point_scores=ablation_context.primary_segment_scores,
                learned_segment_fairness_preallocation=False,
            )
        )
        state.record_substage("without_trajectory_fairness_preallocation", substage_started_at)


def _freeze_prior_field_variant(
    ablation_context: _RetainedMaskAblationContext,
    state: _RetainedMaskAblationState,
    *,
    query_prior_field: dict[str, Any],
    ablation_prior_field: dict[str, Any],
    method_name: str,
) -> dict[str, Any]:
    feature_sensitivity = prior_feature_sample_sensitivity(
        points=ablation_context.test_points,
        primary_prior_field=query_prior_field,
        ablation_prior_field=ablation_prior_field,
    )
    model_sensitivity = model_prior_feature_sensitivity(
        points=ablation_context.test_points,
        point_dim=int(
            getattr(
                ablation_context.trained.model,
                "point_dim",
                ablation_context.test_points.shape[1],
            )
        ),
        scaler=ablation_context.trained.scaler,
        primary_prior_field=query_prior_field,
        ablation_prior_field=ablation_prior_field,
        boundaries=ablation_context.test_boundaries,
        trajectory_mmsis=ablation_context.test_mmsis,
    )
    ablation_trained = training_outputs_with_query_prior_field(
        ablation_context.trained,
        ablation_prior_field,
    )
    ablation_method = build_mlqds_method(
        name=method_name,
        trained=ablation_trained,
        workload=ablation_context.eval_workload,
        workload_map=ablation_context.eval_workload_map,
        config=ablation_context.config,
        range_geometry_blend=0.0,
        trajectory_mmsis=ablation_context.test_mmsis,
    )
    ablation_mask = ablation_method.simplify(
        ablation_context.test_points,
        ablation_context.test_boundaries,
        float(ablation_context.config.model.compression_ratio),
    )
    ablation_snapshot = ablation_method.cached_score_snapshot()
    ablation_scores = ablation_snapshot.scores
    ablation_raw_preds = ablation_snapshot.raw_predictions
    ablation_head_logits = ablation_snapshot.head_logits
    ablation_selector_segment_scores = ablation_snapshot.selector_segment_scores
    sensitivity = prior_ablation_sensitivity_from_tensors(
        sampled_prior_features=feature_sensitivity,
        model_prior_features=model_sensitivity,
        primary_scores=ablation_context.primary_scores,
        ablation_scores=ablation_scores if isinstance(ablation_scores, torch.Tensor) else None,
        primary_raw_predictions=ablation_context.primary_raw_preds,
        ablation_raw_predictions=ablation_raw_preds
        if isinstance(ablation_raw_preds, torch.Tensor)
        else None,
        primary_head_logits=ablation_context.primary_head_logits,
        ablation_head_logits=ablation_head_logits
        if isinstance(ablation_head_logits, torch.Tensor)
        else None,
        primary_mask=ablation_context.frozen_primary_masks.get("MLQDS"),
        ablation_mask=ablation_mask,
        selector_trace=state.trace,
        primary_segment_scores=ablation_context.primary_selector_segment_scores,
        ablation_segment_scores=ablation_selector_segment_scores
        if isinstance(ablation_selector_segment_scores, torch.Tensor)
        else None,
    )
    state.causality_ablation_methods.append(
        FrozenMaskMethod(
            name=method_name,
            retained_mask=ablation_mask.detach().cpu(),
        )
    )
    return sensitivity


def _freeze_prior_field_ablations(
    ablation_context: _RetainedMaskAblationContext,
    state: _RetainedMaskAblationState,
) -> None:
    query_prior_field = ablation_context.trained.feature_context.get("query_prior_field")
    if not isinstance(query_prior_field, dict):
        return

    substage_started_at = time.perf_counter()
    try:
        prior_scores = (
            query_prior_predictability_scores(ablation_context.test_points, query_prior_field).detach().cpu()
        )
        state.causality_ablation_methods.append(
            ablation_context.selector_method(
                name="MLQDS_prior_field_only_score",
                scores=prior_scores,
            )
        )
    except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
        state.causal_ablation_freeze_failures["MLQDS_prior_field_only_score"] = str(exc)
    finally:
        state.record_substage("prior_field_only_score", substage_started_at)

    substage_started_at = time.perf_counter()
    try:
        shuffled_prior_field = shuffled_query_prior_field(
            query_prior_field,
            seed=int(ablation_context.seeds.eval_query_seed) + 71_003,
        )
        state.prior_sensitivity_diagnostics["shuffled_prior_fields"] = (
            _freeze_prior_field_variant(
                ablation_context,
                state,
                query_prior_field=query_prior_field,
                ablation_prior_field=shuffled_prior_field,
                method_name="MLQDS_shuffled_prior_fields",
            )
        )
    except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
        state.causal_ablation_freeze_failures["MLQDS_shuffled_prior_fields"] = str(exc)
    finally:
        state.record_substage("shuffled_prior_fields", substage_started_at)

    substage_started_at = time.perf_counter()
    try:
        zero_prior_field = zero_query_prior_field_like(query_prior_field)
        state.prior_sensitivity_diagnostics["without_query_prior_features"] = (
            _freeze_prior_field_variant(
                ablation_context,
                state,
                query_prior_field=query_prior_field,
                ablation_prior_field=zero_prior_field,
                method_name="MLQDS_without_query_prior_features",
            )
        )
    except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
        state.causal_ablation_freeze_failures["MLQDS_without_query_prior_features"] = str(exc)
    finally:
        state.record_substage("without_query_prior_features", substage_started_at)

    prior_channel_started_at = time.perf_counter()
    for prior_channel_name in QUERY_PRIOR_FIELD_NAMES:
        channel_method_name = f"MLQDS_without_prior_channel_{prior_channel_name}"
        channel_started_at = time.perf_counter()
        try:
            channel_prior_field = zero_query_prior_field_channels(
                query_prior_field,
                [prior_channel_name],
            )
            channel_sensitivity = _freeze_prior_field_variant(
                ablation_context,
                state,
                query_prior_field=query_prior_field,
                ablation_prior_field=channel_prior_field,
                method_name=channel_method_name,
            )
            state.prior_channel_ablation_diagnostics[prior_channel_name] = {
                **channel_sensitivity,
                "method_name": channel_method_name,
            }
        except Exception as exc:  # pragma: no cover - optional diagnostic only.
            state.prior_channel_ablation_diagnostics[prior_channel_name] = {
                "available": False,
                "method_name": channel_method_name,
                "error": str(exc),
            }
        finally:
            state.record_prior_channel(prior_channel_name, channel_started_at)
    state.record_substage("per_prior_channel_ablations", prior_channel_started_at)


def freeze_retained_mask_ablations(
    *,
    config: RunConfig,
    trained: TrainingOutputs,
    eval_workload: TypedQueryWorkload,
    eval_workload_map: dict[str, float],
    test_mmsis: list[int] | None,
    test_points: torch.Tensor,
    test_boundaries: list[tuple[int, int]],
    seeds: SeedBundle,
    primary_selector_trace: dict[str, Any],
    frozen_primary_masks: dict[str, torch.Tensor],
    primary_scores: torch.Tensor,
    primary_raw_preds: torch.Tensor | None,
    primary_segment_scores: torch.Tensor | None,
    primary_path_length_support_scores: torch.Tensor | None,
    primary_selector_segment_scores: torch.Tensor | None,
    primary_head_logits: torch.Tensor | None,
) -> RetainedMaskAblationOutputs:
    """Freeze query-free ablation retained masks before eval query scoring."""
    ablation_context = _RetainedMaskAblationContext(
        config=config,
        trained=trained,
        eval_workload=eval_workload,
        eval_workload_map=eval_workload_map,
        test_mmsis=test_mmsis,
        test_points=test_points,
        test_boundaries=test_boundaries,
        seeds=seeds,
        frozen_primary_masks=frozen_primary_masks,
        primary_scores=primary_scores,
        primary_raw_preds=primary_raw_preds,
        primary_segment_scores=primary_segment_scores,
        primary_path_length_support_scores=primary_path_length_support_scores,
        primary_selector_segment_scores=primary_selector_segment_scores,
        primary_head_logits=primary_head_logits,
    )
    state = _RetainedMaskAblationState(trace=primary_selector_trace)
    allocation_length_support_weight = float(
        config.model.learned_segment_allocation_length_support_weight
    )

    pre_repair_diagnostic_name = "MLQDS_pre_repair_allocation_diagnostic"
    substage_started_at = time.perf_counter()
    try:
        pre_repair_method = pre_repair_frozen_method_from_trace(
            name=pre_repair_diagnostic_name,
            selector_trace=state.trace,
            point_count=int(test_points.shape[0]),
        )
        state.causality_ablation_methods.append(pre_repair_method)
        state.trace["pre_repair_frozen_method_diagnostic"] = {
            "available": True,
            "diagnostic_only": True,
            "query_free": True,
            "method_name": pre_repair_diagnostic_name,
            "source": "selector_trace.pre_repair_retained_mask.indices",
            "retained_count": int(pre_repair_method.retained_mask.sum().item()),
        }
    except Exception as exc:  # pragma: no cover - optional diagnostic should not gate eval.
        state.trace["pre_repair_frozen_method_diagnostic"] = {
            "available": False,
            "diagnostic_only": True,
            "query_free": True,
            "method_name": pre_repair_diagnostic_name,
            "reason": "freeze_failed",
            "error": str(exc),
        }
    finally:
        state.record_substage("pre_repair_from_trace", substage_started_at)
    if float(config.model.learned_segment_geometry_gain_weight) > 0.0:
        substage_started_at = time.perf_counter()
        try:
            state.causality_ablation_methods.append(
                ablation_context.selector_method(
                    name="MLQDS_without_geometry_tie_breaker",
                    scores=primary_scores,
                    segment_scores=primary_selector_segment_scores,
                    segment_point_scores=primary_segment_scores,
                    learned_segment_geometry_gain_weight=0.0,
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            state.causal_ablation_freeze_failures["MLQDS_without_geometry_tie_breaker"] = str(exc)
        finally:
            state.record_substage("without_geometry_tie_breaker", substage_started_at)
    if allocation_length_support_weight > 0.0:
        substage_started_at = time.perf_counter()
        try:
            state.causality_ablation_methods.append(
                ablation_context.selector_method(
                    name="MLQDS_without_segment_length_support_allocation",
                    scores=primary_scores,
                    segment_scores=primary_selector_segment_scores,
                    segment_point_scores=primary_segment_scores,
                    learned_segment_allocation_length_support_weight=0.0,
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            state.causal_ablation_freeze_failures[
                "MLQDS_without_segment_length_support_allocation"
            ] = str(exc)
        finally:
            state.record_substage(
                "without_segment_length_support_allocation", substage_started_at
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
        primary_segment_scores[shuffled_order] if primary_segment_scores is not None else None
    )
    substage_started_at = time.perf_counter()
    try:
        state.causality_ablation_methods.append(
            ablation_context.selector_method(
                name="MLQDS_shuffled_scores",
                scores=shuffled_scores,
                segment_scores=shuffled_segment_scores,
                segment_point_scores=shuffled_segment_point_scores,
            )
        )
    except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
        state.causal_ablation_freeze_failures["MLQDS_shuffled_scores"] = str(exc)
    finally:
        state.record_substage("shuffled_scores", substage_started_at)
    _freeze_segment_budget_ablations(ablation_context, state)
    path_length_support_scores = primary_path_length_support_scores
    if path_length_support_scores is not None:
        substage_started_at = time.perf_counter()
        try:
            path_length_segment_method = ablation_context.selector_method(
                name="MLQDS_path_length_support_segment_head_diagnostic",
                scores=primary_scores,
                segment_scores=path_length_support_scores,
            )
            state.causality_ablation_methods.append(path_length_segment_method)
            state.head_ablation_sensitivity_diagnostics[
                "MLQDS_path_length_support_segment_head_diagnostic"
            ] = {
                **head_ablation_sensitivity(
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
            path_length_allocation_method = ablation_context.selector_method(
                name="MLQDS_path_length_support_allocation_only_diagnostic",
                scores=primary_scores,
                segment_scores=path_length_support_scores,
                segment_point_scores=primary_segment_scores,
            )
            state.causality_ablation_methods.append(path_length_allocation_method)
            state.head_ablation_sensitivity_diagnostics[
                "MLQDS_path_length_support_allocation_only_diagnostic"
            ] = {
                **head_ablation_sensitivity(
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
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            state.head_ablation_sensitivity_diagnostics[
                "MLQDS_path_length_support_segment_head_diagnostic"
            ] = {
                "available": False,
                "diagnostic_only": True,
                "reason": "freeze_failed",
                "error": str(exc),
            }
        finally:
            state.record_substage("path_length_support_ablations", substage_started_at)
    behavior_segment_scores = _factorized_head_scores_from_logits(
        primary_head_logits,
        "conditional_behavior_utility",
    )
    if behavior_segment_scores is not None:
        substage_started_at = time.perf_counter()
        try:
            behavior_segment_method = ablation_context.selector_method(
                name="MLQDS_behavior_utility_segment_head_diagnostic",
                scores=primary_scores,
                segment_scores=behavior_segment_scores,
            )
            state.causality_ablation_methods.append(behavior_segment_method)
            state.head_ablation_sensitivity_diagnostics[
                "MLQDS_behavior_utility_segment_head_diagnostic"
            ] = {
                **head_ablation_sensitivity(
                    primary_scores=primary_scores,
                    ablation_scores=primary_scores,
                    primary_raw_predictions=primary_raw_preds,
                    ablation_raw_predictions=primary_raw_preds,
                    primary_segment_scores=primary_selector_segment_scores,
                    ablation_segment_scores=behavior_segment_scores,
                    primary_mask=frozen_primary_masks.get("MLQDS"),
                    ablation_mask=behavior_segment_method.retained_mask,
                ),
                "diagnostic_only": True,
                "replacement_head_name": "conditional_behavior_utility",
                "ablation_mode": "conditional_behavior_utility_as_segment_scores",
            }
            behavior_allocation_method = ablation_context.selector_method(
                name="MLQDS_behavior_utility_allocation_only_diagnostic",
                scores=primary_scores,
                segment_scores=behavior_segment_scores,
                segment_point_scores=primary_segment_scores,
            )
            state.causality_ablation_methods.append(behavior_allocation_method)
            state.head_ablation_sensitivity_diagnostics[
                "MLQDS_behavior_utility_allocation_only_diagnostic"
            ] = {
                **head_ablation_sensitivity(
                    primary_scores=primary_scores,
                    ablation_scores=primary_scores,
                    primary_raw_predictions=primary_raw_preds,
                    ablation_raw_predictions=primary_raw_preds,
                    primary_segment_scores=primary_selector_segment_scores,
                    ablation_segment_scores=behavior_segment_scores,
                    primary_mask=frozen_primary_masks.get("MLQDS"),
                    ablation_mask=behavior_allocation_method.retained_mask,
                ),
                "diagnostic_only": True,
                "replacement_head_name": "conditional_behavior_utility",
                "ablation_mode": "conditional_behavior_utility_allocation_only",
            }
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            state.head_ablation_sensitivity_diagnostics[
                "MLQDS_behavior_utility_segment_head_diagnostic"
            ] = {
                "available": False,
                "diagnostic_only": True,
                "reason": "freeze_failed",
                "error": str(exc),
            }
        finally:
            state.record_substage(
                "behavior_utility_segment_score_ablations", substage_started_at
            )
    if primary_head_logits is not None:
        substage_started_at = time.perf_counter()
        try:
            behavior_raw_preds = raw_predictions_without_factorized_head(
                model=trained.model,
                head_logits=primary_head_logits,
                disabled_head_name="conditional_behavior_utility",
            )
            behavior_scores = scores_without_factorized_head(
                model=trained.model,
                head_logits=primary_head_logits,
                disabled_head_name="conditional_behavior_utility",
                boundaries=test_boundaries,
                workload_type=single_workload_type(eval_workload_map),
                score_mode=config.model.mlqds_score_mode,
                score_temperature=float(config.model.mlqds_score_temperature),
                rank_confidence_weight=float(config.model.mlqds_rank_confidence_weight),
            )
            behavior_ablation_method = ablation_context.selector_method(
                name="MLQDS_without_behavior_utility_head",
                scores=behavior_scores,
                segment_scores=primary_selector_segment_scores,
                segment_point_scores=primary_segment_scores,
            )
            state.causality_ablation_methods.append(behavior_ablation_method)
            behavior_sensitivity = head_ablation_sensitivity(
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
            state.head_ablation_sensitivity_diagnostics["MLQDS_without_behavior_utility_head"] = (
                behavior_sensitivity
            )
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            state.causal_ablation_freeze_failures["MLQDS_without_behavior_utility_head"] = (
                str(exc)
            )
        finally:
            state.record_substage("without_behavior_utility_head", substage_started_at)
    substage_started_at = time.perf_counter()
    try:
        untrained_model = reset_module_parameters(
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
        untrained_method = build_mlqds_method(
            name="MLQDS_untrained_model",
            trained=untrained_outputs,
            workload=eval_workload,
            workload_map=eval_workload_map,
            config=config,
            range_geometry_blend=0.0,
            trajectory_mmsis=test_mmsis,
        )
        untrained_mask = untrained_method.simplify(
            test_points,
            test_boundaries,
            float(config.model.compression_ratio),
        )
        state.causality_ablation_methods.append(
            FrozenMaskMethod(
                name="MLQDS_untrained_model",
                retained_mask=untrained_mask.detach().cpu(),
            )
        )
    except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
        state.causal_ablation_freeze_failures["MLQDS_untrained_model"] = str(exc)
    finally:
        state.record_substage("untrained_model", substage_started_at)
    _freeze_prior_field_ablations(ablation_context, state)
    return state.outputs()
