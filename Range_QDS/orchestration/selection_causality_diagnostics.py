"""Checkpoint-selection causality diagnostics for query-driven runs."""

from __future__ import annotations

from typing import Any

import torch

from config.run_config import RunConfig
from learning.outputs import TrainingOutputs
from learning.predictability_audit import query_prior_predictability_scores
from learning.query_prior_fields import query_prior_field_metadata, zero_query_prior_field_like
from orchestration.causality import (
    causality_ablation_diagnostics_payload,
    head_ablation_sensitivity,
    head_output_sensitivity,
    model_prior_feature_sensitivity,
    prior_feature_sample_sensitivity,
    retained_mask_comparison,
    score_ablation_sensitivity,
)
from orchestration.mlqds_method_factory import build_mlqds_method
from orchestration.model_ablations import (
    raw_predictions_without_factorized_head,
    scores_without_factorized_head,
    shuffled_query_prior_field,
)
from orchestration.selector_diagnostics import (
    learned_segment_frozen_method,
    neutral_segment_scores_for_ablation,
)
from scoring.method_scoring import score_method
from scoring.methods import FrozenMaskMethod, MLQDSMethod
from scoring.metrics import MethodScore
from scoring.query_cache import ScoringQueryCache
from selection.learned_segment_budget import blend_segment_support_scores
from workloads.query_types import single_workload_type


def build_selection_causality_diagnostics(
    *,
    trained: TrainingOutputs,
    selection_points: torch.Tensor | None,
    selection_boundaries: list[tuple[int, int]] | None,
    selection_workload: Any | None,
    eval_workload_map: dict[str, float],
    selection_query_cache: ScoringQueryCache | None,
    config: RunConfig,
    seeds: Any,
) -> dict[str, Any]:
    """Return checkpoint-validation ablation diagnostics without changing selection."""
    if selection_points is None or selection_boundaries is None or selection_workload is None:
        return {"available": False, "reason": "missing_selection_split"}
    if str(getattr(config.model, "selector_type", "")).lower() != "learned_segment_budget_v1":
        return {"available": False, "reason": "requires_learned_segment_budget_v1"}

    workload_type = single_workload_type(eval_workload_map)

    def _mlqds_method(
        *,
        name: str,
        trained_outputs: TrainingOutputs,
        workload: Any,
    ) -> MLQDSMethod:
        return build_mlqds_method(
            name=name,
            trained=trained_outputs,
            workload=workload,
            workload_map=eval_workload_map,
            config=config,
            range_geometry_blend=0.0,
        )

    primary_method = _mlqds_method(
        name="MLQDS_selection_primary",
        trained_outputs=trained,
        workload=selection_workload,
    )
    try:
        primary_mask = (
            primary_method.simplify(
                selection_points,
                selection_boundaries,
                float(config.model.compression_ratio),
            )
            .detach()
            .cpu()
        )
    except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
        return {"available": False, "reason": "primary_mask_freeze_failed", "error": str(exc)}

    primary_score = score_method(
        method=FrozenMaskMethod(name="MLQDS", retained_mask=primary_mask),
        points=selection_points,
        boundaries=selection_boundaries,
        typed_queries=selection_workload.typed_queries,
        workload_map=eval_workload_map,
        compression_ratio=config.model.compression_ratio,
        query_cache=selection_query_cache,
    )
    primary_scores = getattr(primary_method, "_score_cache", None)
    primary_raw_preds = getattr(primary_method, "_raw_pred_cache", None)
    primary_head_logits = getattr(primary_method, "_head_logit_cache", None)
    primary_segment_scores = getattr(primary_method, "_segment_score_cache", None)
    primary_path_length_support_scores = getattr(
        primary_method, "_path_length_support_score_cache", None
    )
    primary_selector_segment_scores = getattr(primary_method, "_selector_segment_score_cache", None)
    if isinstance(primary_scores, torch.Tensor):
        primary_scores = primary_scores.detach().cpu().float()
    if isinstance(primary_raw_preds, torch.Tensor):
        primary_raw_preds = primary_raw_preds.detach().cpu().float()
    if isinstance(primary_head_logits, torch.Tensor):
        primary_head_logits = primary_head_logits.detach().cpu().float()
    if isinstance(primary_segment_scores, torch.Tensor):
        primary_segment_scores = primary_segment_scores.detach().cpu().float()
    if isinstance(primary_path_length_support_scores, torch.Tensor):
        primary_path_length_support_scores = (
            primary_path_length_support_scores.detach().cpu().float()
        )
    if isinstance(primary_selector_segment_scores, torch.Tensor):
        primary_selector_segment_scores = primary_selector_segment_scores.detach().cpu().float()

    ablation_methods: list[FrozenMaskMethod] = []
    freeze_failures: dict[str, str] = {}
    prior_sensitivity: dict[str, Any] = {}
    head_sensitivity: dict[str, Any] = {}

    geometry_gain_weight = float(config.model.learned_segment_geometry_gain_weight)
    allocation_length_support_weight = float(
        config.model.learned_segment_allocation_length_support_weight
    )
    allocation_weight_floor = float(config.model.learned_segment_allocation_weight_floor)
    repair_score_protection_fraction = float(
        config.model.learned_segment_length_repair_score_protection_fraction
    )
    if isinstance(primary_scores, torch.Tensor) and geometry_gain_weight > 0.0:
        try:
            selection_segment_scores = (
                primary_selector_segment_scores
                if isinstance(primary_selector_segment_scores, torch.Tensor)
                else None
            )
            ablation_methods.append(
                learned_segment_frozen_method(
                    name="MLQDS_without_geometry_tie_breaker",
                    scores=primary_scores,
                    boundaries=selection_boundaries,
                    compression_ratio=float(config.model.compression_ratio),
                    segment_scores=selection_segment_scores,
                    segment_point_scores=primary_segment_scores,
                    points=selection_points,
                    learned_segment_geometry_gain_weight=0.0,
                    learned_segment_allocation_length_support_weight=(
                        allocation_length_support_weight
                    ),
                    learned_segment_allocation_weight_floor=allocation_weight_floor,
                    learned_segment_score_blend_weight=float(
                        config.model.learned_segment_score_blend_weight
                    ),
                    learned_segment_fairness_preallocation=bool(
                        config.model.learned_segment_fairness_preallocation
                    ),
                    learned_segment_length_repair_fraction=float(
                        config.model.learned_segment_length_repair_fraction
                    ),
                    learned_segment_length_repair_score_protection_fraction=(
                        repair_score_protection_fraction
                    ),
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            freeze_failures["MLQDS_without_geometry_tie_breaker"] = str(exc)

    if isinstance(primary_scores, torch.Tensor) and allocation_length_support_weight > 0.0:
        try:
            selection_segment_scores = (
                primary_selector_segment_scores
                if isinstance(primary_selector_segment_scores, torch.Tensor)
                else None
            )
            ablation_methods.append(
                learned_segment_frozen_method(
                    name="MLQDS_without_segment_length_support_allocation",
                    scores=primary_scores,
                    boundaries=selection_boundaries,
                    compression_ratio=float(config.model.compression_ratio),
                    segment_scores=selection_segment_scores,
                    segment_point_scores=primary_segment_scores,
                    points=selection_points,
                    learned_segment_geometry_gain_weight=float(
                        config.model.learned_segment_geometry_gain_weight
                    ),
                    learned_segment_allocation_length_support_weight=0.0,
                    learned_segment_allocation_weight_floor=allocation_weight_floor,
                    learned_segment_score_blend_weight=float(
                        config.model.learned_segment_score_blend_weight
                    ),
                    learned_segment_fairness_preallocation=bool(
                        config.model.learned_segment_fairness_preallocation
                    ),
                    learned_segment_length_repair_fraction=float(
                        config.model.learned_segment_length_repair_fraction
                    ),
                    learned_segment_length_repair_score_protection_fraction=(
                        repair_score_protection_fraction
                    ),
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            freeze_failures["MLQDS_without_segment_length_support_allocation"] = str(exc)

    if isinstance(primary_scores, torch.Tensor) and isinstance(
        primary_segment_scores, torch.Tensor
    ):
        try:
            neutral_segment_scores = neutral_segment_scores_for_ablation(primary_segment_scores)
            no_segment_selector_scores = blend_segment_support_scores(
                segment_scores=neutral_segment_scores,
                path_length_support_scores=(
                    primary_path_length_support_scores
                    if isinstance(primary_path_length_support_scores, torch.Tensor)
                    else None
                ),
                path_length_support_weight=float(
                    config.model.learned_segment_length_support_blend_weight
                ),
            )
            no_segment = learned_segment_frozen_method(
                name="MLQDS_without_segment_budget_head",
                scores=primary_scores,
                boundaries=selection_boundaries,
                compression_ratio=float(config.model.compression_ratio),
                segment_scores=no_segment_selector_scores,
                segment_point_scores=neutral_segment_scores,
                points=selection_points,
                learned_segment_geometry_gain_weight=float(
                    config.model.learned_segment_geometry_gain_weight
                ),
                learned_segment_allocation_length_support_weight=allocation_length_support_weight,
                learned_segment_allocation_weight_floor=allocation_weight_floor,
                learned_segment_score_blend_weight=float(
                    config.model.learned_segment_score_blend_weight
                ),
                learned_segment_fairness_preallocation=bool(
                    config.model.learned_segment_fairness_preallocation
                ),
                learned_segment_length_repair_fraction=float(
                    config.model.learned_segment_length_repair_fraction
                ),
                learned_segment_length_repair_score_protection_fraction=(
                    repair_score_protection_fraction
                ),
            )
            ablation_methods.append(no_segment)
            head_sensitivity["MLQDS_without_segment_budget_head"] = {
                **head_ablation_sensitivity(
                    primary_scores=primary_scores,
                    ablation_scores=primary_scores,
                    primary_raw_predictions=primary_raw_preds
                    if isinstance(primary_raw_preds, torch.Tensor)
                    else None,
                    ablation_raw_predictions=primary_raw_preds
                    if isinstance(primary_raw_preds, torch.Tensor)
                    else None,
                    primary_segment_scores=(
                        primary_selector_segment_scores
                        if isinstance(primary_selector_segment_scores, torch.Tensor)
                        else primary_segment_scores
                    ),
                    ablation_segment_scores=no_segment_selector_scores,
                    primary_mask=primary_mask,
                    ablation_mask=no_segment.retained_mask,
                ),
                "disabled_head_name": "segment_budget_target",
                "ablation_mode": "neutral_constant_segment_scores",
            }
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            freeze_failures["MLQDS_without_segment_budget_head"] = str(exc)

    if isinstance(primary_scores, torch.Tensor) and isinstance(
        primary_path_length_support_scores, torch.Tensor
    ):
        try:
            path_length_segment_method = learned_segment_frozen_method(
                name="MLQDS_path_length_support_segment_head_diagnostic",
                scores=primary_scores,
                boundaries=selection_boundaries,
                compression_ratio=float(config.model.compression_ratio),
                segment_scores=primary_path_length_support_scores,
                points=selection_points,
                learned_segment_geometry_gain_weight=float(
                    config.model.learned_segment_geometry_gain_weight
                ),
                learned_segment_allocation_length_support_weight=allocation_length_support_weight,
                learned_segment_allocation_weight_floor=allocation_weight_floor,
                learned_segment_score_blend_weight=float(
                    config.model.learned_segment_score_blend_weight
                ),
                learned_segment_fairness_preallocation=bool(
                    config.model.learned_segment_fairness_preallocation
                ),
                learned_segment_length_repair_fraction=float(
                    config.model.learned_segment_length_repair_fraction
                ),
                learned_segment_length_repair_score_protection_fraction=(
                    repair_score_protection_fraction
                ),
            )
            ablation_methods.append(path_length_segment_method)
            head_sensitivity["MLQDS_path_length_support_segment_head_diagnostic"] = {
                **head_ablation_sensitivity(
                    primary_scores=primary_scores,
                    ablation_scores=primary_scores,
                    primary_raw_predictions=primary_raw_preds
                    if isinstance(primary_raw_preds, torch.Tensor)
                    else None,
                    ablation_raw_predictions=primary_raw_preds
                    if isinstance(primary_raw_preds, torch.Tensor)
                    else None,
                    primary_segment_scores=(
                        primary_selector_segment_scores
                        if isinstance(primary_selector_segment_scores, torch.Tensor)
                        else primary_segment_scores
                        if isinstance(primary_segment_scores, torch.Tensor)
                        else None
                    ),
                    ablation_segment_scores=primary_path_length_support_scores,
                    primary_mask=primary_mask,
                    ablation_mask=path_length_segment_method.retained_mask,
                ),
                "diagnostic_only": True,
                "replacement_head_name": "path_length_support_target",
                "ablation_mode": "path_length_support_as_segment_scores",
            }
            path_length_allocation_method = learned_segment_frozen_method(
                name="MLQDS_path_length_support_allocation_only_diagnostic",
                scores=primary_scores,
                boundaries=selection_boundaries,
                compression_ratio=float(config.model.compression_ratio),
                segment_scores=primary_path_length_support_scores,
                segment_point_scores=primary_segment_scores,
                points=selection_points,
                learned_segment_geometry_gain_weight=float(
                    config.model.learned_segment_geometry_gain_weight
                ),
                learned_segment_allocation_length_support_weight=allocation_length_support_weight,
                learned_segment_allocation_weight_floor=allocation_weight_floor,
                learned_segment_score_blend_weight=float(
                    config.model.learned_segment_score_blend_weight
                ),
                learned_segment_fairness_preallocation=bool(
                    config.model.learned_segment_fairness_preallocation
                ),
                learned_segment_length_repair_fraction=float(
                    config.model.learned_segment_length_repair_fraction
                ),
                learned_segment_length_repair_score_protection_fraction=(
                    repair_score_protection_fraction
                ),
            )
            ablation_methods.append(path_length_allocation_method)
            head_sensitivity["MLQDS_path_length_support_allocation_only_diagnostic"] = {
                **head_ablation_sensitivity(
                    primary_scores=primary_scores,
                    ablation_scores=primary_scores,
                    primary_raw_predictions=primary_raw_preds
                    if isinstance(primary_raw_preds, torch.Tensor)
                    else None,
                    ablation_raw_predictions=primary_raw_preds
                    if isinstance(primary_raw_preds, torch.Tensor)
                    else None,
                    primary_segment_scores=(
                        primary_selector_segment_scores
                        if isinstance(primary_selector_segment_scores, torch.Tensor)
                        else primary_segment_scores
                        if isinstance(primary_segment_scores, torch.Tensor)
                        else None
                    ),
                    ablation_segment_scores=primary_path_length_support_scores,
                    primary_mask=primary_mask,
                    ablation_mask=path_length_allocation_method.retained_mask,
                ),
                "diagnostic_only": True,
                "replacement_head_name": "path_length_support_target",
                "ablation_mode": "path_length_support_allocation_only",
            }
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            freeze_failures["MLQDS_path_length_support_segment_head_diagnostic"] = str(exc)

    if (
        isinstance(primary_scores, torch.Tensor)
        and isinstance(primary_head_logits, torch.Tensor)
        and isinstance(primary_selector_segment_scores, torch.Tensor)
    ):
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
                boundaries=selection_boundaries,
                workload_type=workload_type,
                score_mode=config.model.mlqds_score_mode,
                score_temperature=float(config.model.mlqds_score_temperature),
                rank_confidence_weight=float(config.model.mlqds_rank_confidence_weight),
            )
            no_behavior = learned_segment_frozen_method(
                name="MLQDS_without_behavior_utility_head",
                scores=behavior_scores,
                boundaries=selection_boundaries,
                compression_ratio=float(config.model.compression_ratio),
                segment_scores=primary_selector_segment_scores,
                segment_point_scores=primary_segment_scores,
                points=selection_points,
                learned_segment_geometry_gain_weight=float(
                    config.model.learned_segment_geometry_gain_weight
                ),
                learned_segment_allocation_length_support_weight=allocation_length_support_weight,
                learned_segment_allocation_weight_floor=allocation_weight_floor,
                learned_segment_score_blend_weight=float(
                    config.model.learned_segment_score_blend_weight
                ),
                learned_segment_fairness_preallocation=bool(
                    config.model.learned_segment_fairness_preallocation
                ),
                learned_segment_length_repair_fraction=float(
                    config.model.learned_segment_length_repair_fraction
                ),
                learned_segment_length_repair_score_protection_fraction=(
                    repair_score_protection_fraction
                ),
            )
            ablation_methods.append(no_behavior)
            head_sensitivity["MLQDS_without_behavior_utility_head"] = {
                **head_ablation_sensitivity(
                    primary_scores=primary_scores,
                    ablation_scores=behavior_scores,
                    primary_raw_predictions=primary_raw_preds
                    if isinstance(primary_raw_preds, torch.Tensor)
                    else None,
                    ablation_raw_predictions=behavior_raw_preds,
                    primary_segment_scores=primary_selector_segment_scores,
                    ablation_segment_scores=primary_selector_segment_scores,
                    primary_mask=primary_mask,
                    ablation_mask=no_behavior.retained_mask,
                ),
                "disabled_head_name": "conditional_behavior_utility",
                "ablation_mode": "neutral_multiplicative_head",
            }
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            freeze_failures["MLQDS_without_behavior_utility_head"] = str(exc)

    query_prior_field = trained.feature_context.get("query_prior_field")
    if isinstance(query_prior_field, dict) and isinstance(primary_scores, torch.Tensor):
        try:
            prior_scores = (
                query_prior_predictability_scores(selection_points, query_prior_field)
                .detach()
                .cpu()
            )
            ablation_methods.append(
                learned_segment_frozen_method(
                    name="MLQDS_prior_field_only_score",
                    scores=prior_scores,
                    boundaries=selection_boundaries,
                    compression_ratio=float(config.model.compression_ratio),
                    points=selection_points,
                    learned_segment_geometry_gain_weight=float(
                        config.model.learned_segment_geometry_gain_weight
                    ),
                    learned_segment_allocation_length_support_weight=(
                        allocation_length_support_weight
                    ),
                    learned_segment_allocation_weight_floor=allocation_weight_floor,
                    learned_segment_score_blend_weight=float(
                        config.model.learned_segment_score_blend_weight
                    ),
                    learned_segment_fairness_preallocation=bool(
                        config.model.learned_segment_fairness_preallocation
                    ),
                    learned_segment_length_repair_fraction=float(
                        config.model.learned_segment_length_repair_fraction
                    ),
                    learned_segment_length_repair_score_protection_fraction=(
                        repair_score_protection_fraction
                    ),
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            freeze_failures["MLQDS_prior_field_only_score"] = str(exc)
        prior_ablation_fields = {
            "MLQDS_shuffled_prior_fields": shuffled_query_prior_field(
                query_prior_field,
                seed=int(seeds.eval_query_seed) + 72_003,
            ),
            "MLQDS_without_query_prior_features": zero_query_prior_field_like(query_prior_field),
        }
        for ablation_name, ablation_field in prior_ablation_fields.items():
            try:
                prior_sensitivity_key = (
                    "shuffled_prior_fields"
                    if ablation_name == "MLQDS_shuffled_prior_fields"
                    else "without_query_prior_features"
                )
                prior_feature_sensitivity = prior_feature_sample_sensitivity(
                    points=selection_points,
                    primary_prior_field=query_prior_field,
                    ablation_prior_field=ablation_field,
                )
                model_prior_sensitivity = model_prior_feature_sensitivity(
                    points=selection_points,
                    point_dim=int(getattr(trained.model, "point_dim", selection_points.shape[1])),
                    scaler=trained.scaler,
                    primary_prior_field=query_prior_field,
                    ablation_prior_field=ablation_field,
                    boundaries=selection_boundaries,
                )
                ablation_trained = TrainingOutputs(
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
                        "query_prior_field": ablation_field,
                        "query_prior_field_metadata": query_prior_field_metadata(ablation_field),
                    },
                )
                ablation_method = _mlqds_method(
                    name=ablation_name,
                    trained_outputs=ablation_trained,
                    workload=selection_workload,
                )
                ablation_mask = ablation_method.simplify(
                    selection_points,
                    selection_boundaries,
                    float(config.model.compression_ratio),
                )
                ablation_point_scores = getattr(ablation_method, "_score_cache", None)
                ablation_raw_preds = getattr(ablation_method, "_raw_pred_cache", None)
                ablation_head_logits = getattr(ablation_method, "_head_logit_cache", None)
                prior_sensitivity[prior_sensitivity_key] = {
                    "sampled_prior_features": prior_feature_sensitivity,
                    "model_prior_features": model_prior_sensitivity,
                    "selector_score": score_ablation_sensitivity(
                        primary_scores=primary_scores,
                        ablation_scores=ablation_point_scores
                        if isinstance(ablation_point_scores, torch.Tensor)
                        else None,
                        primary_mask=primary_mask,
                        ablation_mask=ablation_mask,
                    ),
                    "raw_prediction": score_ablation_sensitivity(
                        primary_scores=primary_raw_preds
                        if isinstance(primary_raw_preds, torch.Tensor)
                        else None,
                        ablation_scores=ablation_raw_preds
                        if isinstance(ablation_raw_preds, torch.Tensor)
                        else None,
                        primary_mask=primary_mask,
                        ablation_mask=ablation_mask,
                    ),
                    "head_output": head_output_sensitivity(
                        primary_head_logits=primary_head_logits
                        if isinstance(primary_head_logits, torch.Tensor)
                        else None,
                        ablation_head_logits=ablation_head_logits
                        if isinstance(ablation_head_logits, torch.Tensor)
                        else None,
                    ),
                }
                ablation_methods.append(
                    FrozenMaskMethod(
                        name=ablation_name,
                        retained_mask=ablation_mask.detach().cpu(),
                    )
                )
            except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
                freeze_failures[ablation_name] = str(exc)

    if not ablation_methods:
        return {
            "available": False,
            "reason": "no_validation_ablations_frozen",
            "ablation_freeze_failures": freeze_failures,
        }

    ablation_scores: dict[str, MethodScore] = {}
    mask_diagnostics: dict[str, dict[str, Any]] = {}
    for method in ablation_methods:
        mask_diagnostics[method.name] = retained_mask_comparison(
            primary_mask=primary_mask,
            ablation_mask=method.retained_mask,
            expected_shape=primary_mask.shape,
        )
        ablation_scores[method.name] = score_method(
            method=method,
            points=selection_points,
            boundaries=selection_boundaries,
            typed_queries=selection_workload.typed_queries,
            workload_map=eval_workload_map,
            compression_ratio=config.model.compression_ratio,
            query_cache=selection_query_cache,
        )

    payload = causality_ablation_diagnostics_payload(
        primary=primary_score,
        ablations=ablation_scores,
        mask_diagnostics=mask_diagnostics,
    )
    for name, tradeoff_diagnostics in payload["tradeoff_diagnostics"].items():
        if name in head_sensitivity:
            head_sensitivity[name]["query_useful_component_tradeoff"] = tradeoff_diagnostics
    payload.update(
        {
            "split": "checkpoint_selection",
            "diagnostic_only": True,
            "query_count": len(selection_workload.typed_queries),
            "ablation_freeze_failures": freeze_failures,
            "prior_sensitivity_diagnostics": prior_sensitivity,
            "head_ablation_sensitivity_diagnostics": head_sensitivity,
        }
    )
    return payload
