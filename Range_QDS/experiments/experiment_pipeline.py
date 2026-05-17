"""Experiment orchestration helpers for training and evaluation runs. See experiments/README.md for details."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from evaluation.baselines import (
    FrozenMaskMethod,
    Method,
    MLQDSMethod,
    OracleMethod,
)
from evaluation.evaluate_methods import evaluate_method
from evaluation.metrics import MethodEvaluation
from evaluation.query_cache import EvaluationQueryCache
from evaluation.range_usefulness import range_usefulness_weight_summary
from evaluation.tables import (
    print_geometric_distortion_table,
    print_method_comparison_table,
    print_range_usefulness_table,
    print_shift_table,
)
from experiments.experiment_config import ExperimentConfig, derive_seed_bundle
from experiments.causality import (
    LEARNING_CAUSALITY_MIN_MATERIAL_DELTA,
    _causality_ablation_diagnostics_payload,
    _causality_ablation_tradeoff_summary,
    _head_ablation_sensitivity,
    _learned_slot_summary,
    _learning_causality_delta_gate_config,
    _prior_feature_sample_sensitivity,
    _prior_sample_gate_failures,
    _query_useful_delta,
    _query_useful_component_delta_summary,
    _retained_mask_comparison,
    _score_ablation_sensitivity,
)
from experiments.experiment_data import build_experiment_datasets, prepare_experiment_split
from experiments.gates import (
    _global_sanity_gate,
    _support_overlap_gate,
    _target_diffusion_gate,
    _workload_stability_gate,
)
from experiments.geojson_writers import report_trajectory_length_loss, write_queries_geojson, write_simplified_csv
from experiments.length_diagnostics import (
    _score_protected_length_feasibility,
    _score_protected_length_frontier,
)
from experiments.model_ablations import (
    _raw_predictions_without_factorized_head,
    _reset_module_parameters,
    _scores_without_factorized_head,
    _shuffled_query_prior_field,
)
from experiments.experiment_methods import (
    attach_range_geometry_scores,
    build_learned_fill_methods,
    build_primary_methods,
    evaluate_shift_pairs,
    prepare_eval_labels,
    prepare_eval_query_cache,
)
from experiments.experiment_outputs import ExperimentOutputs, write_experiment_results
from experiments.range_cache import (
    RangeRuntimeCache,
    prepare_range_label_cache,
    range_only_queries,
)
from experiments.range_diagnostics import (
    _evaluation_metrics_payload,
    _print_range_diagnostics_summary,
    _print_range_distribution_comparison,
    _range_audit_ratios,
    _range_learned_fill_summary,
    _range_workload_diagnostics,
    _range_workload_distribution_comparison,
)
from experiments.segment_audits import (
    _factorized_head_probability_sources_from_logits,
    _segment_oracle_allocation_audit,
    _target_segment_oracle_alignment_audit,
)
from experiments.selector_diagnostics import (
    _learned_segment_frozen_method,
    _neutral_segment_scores_for_ablation,
    _pre_repair_frozen_method_from_trace,
    _segment_score_quantile_bands_for_ablation,
    _segment_score_top_band_for_ablation,
    _selector_segment_score_source_label,
)
from experiments.experiment_workloads import (
    generate_experiment_workloads,
    resolve_workload_maps,
    workload_name,
)
from queries.query_types import QUERY_TYPE_ID_RANGE, single_workload_type
from simplification.mlqds_scoring import workload_type_head
from simplification.learned_segment_budget import (
    blend_segment_support_scores,
    learned_segment_budget_diagnostics,
    simplify_with_learned_segment_budget_v1_with_trace,
)
from simplification.simplify_trajectories import temporal_hybrid_selector_budget_diagnostics
from training.train_model import train_model
from training.checkpoints import ModelArtifacts, save_checkpoint
from training.model_features import is_workload_blind_model_type, model_type_metadata
from training.predictability_audit import query_prior_predictability_audit, query_prior_predictability_scores
from training.query_prior_fields import (
    QUERY_PRIOR_FIELD_NAMES,
    query_prior_field_metadata,
    zero_query_prior_field_channels,
    zero_query_prior_field_like,
)
from training.training_outputs import TrainingOutputs
from training.teacher_distillation import (
    build_range_teacher_config,
    distill_range_teacher_labels,
    range_teacher_distillation_enabled,
)
from training.training_targets import (
    aggregate_range_component_label_sets,
    aggregate_range_component_retained_frequency_training_labels,
    aggregate_range_continuity_retained_frequency_training_labels,
    aggregate_range_global_budget_retained_frequency_training_labels,
    aggregate_range_label_sets,
    aggregate_range_marginal_coverage_training_labels,
    aggregate_range_retained_frequency_training_labels,
    aggregate_range_structural_retained_frequency_training_labels,
    balance_range_training_target_by_trajectory,
    range_component_retained_frequency_training_labels,
    range_continuity_retained_frequency_training_labels,
    range_global_budget_retained_frequency_training_labels,
    range_historical_prior_retained_frequency_training_labels,
    range_local_swap_gain_cost_frequency_training_labels,
    range_local_swap_utility_frequency_training_labels,
    range_query_residual_frequency_training_labels,
    range_set_utility_frequency_training_labels,
    range_query_spine_frequency_training_labels,
    range_marginal_coverage_training_labels,
    range_retained_frequency_training_labels,
    range_structural_retained_frequency_training_labels,
)
from experiments.torch_runtime import (
    amp_runtime_snapshot,
    cuda_memory_snapshot,
    reset_cuda_peak_memory_stats,
    torch_runtime_snapshot,
)


@contextmanager
def _phase(name: str):
    """Log a named phase with wall-clock timing."""
    print(f"[{name}] starting...", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[{name}] done in {dt:.2f}s", flush=True)


def _selection_causality_diagnostics(
    *,
    trained: TrainingOutputs,
    selection_points: torch.Tensor | None,
    selection_boundaries: list[tuple[int, int]] | None,
    selection_workload: Any | None,
    eval_workload_map: dict[str, float],
    selection_query_cache: EvaluationQueryCache | None,
    config: ExperimentConfig,
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
        return MLQDSMethod(
            name=name,
            trained=trained_outputs,
            workload=workload,
            workload_type=workload_type,
            score_mode=config.model.mlqds_score_mode,
            score_temperature=config.model.mlqds_score_temperature,
            rank_confidence_weight=config.model.mlqds_rank_confidence_weight,
            temporal_fraction=config.model.mlqds_temporal_fraction,
            diversity_bonus=config.model.mlqds_diversity_bonus,
            hybrid_mode=config.model.mlqds_hybrid_mode,
            stratified_center_weight=config.model.mlqds_stratified_center_weight,
            min_learned_swaps=config.model.mlqds_min_learned_swaps,
            selector_type=config.model.selector_type,
            trajectory_mmsis=None,
            inference_device=None,
            amp_mode=config.model.amp_mode,
            inference_batch_size=config.model.inference_batch_size,
            learned_segment_geometry_gain_weight=config.model.learned_segment_geometry_gain_weight,
            learned_segment_score_blend_weight=config.model.learned_segment_score_blend_weight,
            learned_segment_fairness_preallocation=config.model.learned_segment_fairness_preallocation,
            learned_segment_length_repair_fraction=config.model.learned_segment_length_repair_fraction,
            learned_segment_length_support_blend_weight=config.model.learned_segment_length_support_blend_weight,
        )

    primary_method = _mlqds_method(
        name="MLQDS_selection_primary",
        trained_outputs=trained,
        workload=selection_workload,
    )
    try:
        primary_mask = primary_method.simplify(
            selection_points,
            selection_boundaries,
            float(config.model.compression_ratio),
        ).detach().cpu()
    except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
        return {"available": False, "reason": "primary_mask_freeze_failed", "error": str(exc)}

    primary_eval = evaluate_method(
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
    primary_path_length_support_scores = getattr(primary_method, "_path_length_support_score_cache", None)
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
        primary_path_length_support_scores = primary_path_length_support_scores.detach().cpu().float()
    if isinstance(primary_selector_segment_scores, torch.Tensor):
        primary_selector_segment_scores = primary_selector_segment_scores.detach().cpu().float()

    ablation_methods: list[FrozenMaskMethod] = []
    freeze_failures: dict[str, str] = {}
    prior_sensitivity: dict[str, Any] = {}
    head_sensitivity: dict[str, Any] = {}

    geometry_gain_weight = float(config.model.learned_segment_geometry_gain_weight)
    if isinstance(primary_scores, torch.Tensor) and geometry_gain_weight > 0.0:
        try:
            selection_segment_scores = (
                primary_selector_segment_scores if isinstance(primary_selector_segment_scores, torch.Tensor) else None
            )
            ablation_methods.append(
                _learned_segment_frozen_method(
                    name="MLQDS_without_geometry_tie_breaker",
                    scores=primary_scores,
                    boundaries=selection_boundaries,
                    compression_ratio=float(config.model.compression_ratio),
                    segment_scores=selection_segment_scores,
                    segment_point_scores=primary_segment_scores,
                    points=selection_points,
                    learned_segment_geometry_gain_weight=0.0,
                    learned_segment_score_blend_weight=float(config.model.learned_segment_score_blend_weight),
                    learned_segment_fairness_preallocation=bool(config.model.learned_segment_fairness_preallocation),
                    learned_segment_length_repair_fraction=float(config.model.learned_segment_length_repair_fraction),
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            freeze_failures["MLQDS_without_geometry_tie_breaker"] = str(exc)

    if isinstance(primary_scores, torch.Tensor) and isinstance(primary_segment_scores, torch.Tensor):
        try:
            neutral_segment_scores = _neutral_segment_scores_for_ablation(primary_segment_scores)
            no_segment_selector_scores = blend_segment_support_scores(
                segment_scores=neutral_segment_scores,
                path_length_support_scores=(
                    primary_path_length_support_scores
                    if isinstance(primary_path_length_support_scores, torch.Tensor)
                    else None
                ),
                path_length_support_weight=float(config.model.learned_segment_length_support_blend_weight),
            )
            no_segment = _learned_segment_frozen_method(
                name="MLQDS_without_segment_budget_head",
                scores=primary_scores,
                boundaries=selection_boundaries,
                compression_ratio=float(config.model.compression_ratio),
                segment_scores=no_segment_selector_scores,
                segment_point_scores=neutral_segment_scores,
                points=selection_points,
                learned_segment_geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                learned_segment_score_blend_weight=float(config.model.learned_segment_score_blend_weight),
                learned_segment_fairness_preallocation=bool(config.model.learned_segment_fairness_preallocation),
                learned_segment_length_repair_fraction=float(config.model.learned_segment_length_repair_fraction),
            )
            ablation_methods.append(no_segment)
            head_sensitivity["MLQDS_without_segment_budget_head"] = {
                **_head_ablation_sensitivity(
                    primary_scores=primary_scores,
                    ablation_scores=primary_scores,
                    primary_raw_predictions=primary_raw_preds if isinstance(primary_raw_preds, torch.Tensor) else None,
                    ablation_raw_predictions=primary_raw_preds if isinstance(primary_raw_preds, torch.Tensor) else None,
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

    if isinstance(primary_scores, torch.Tensor) and isinstance(primary_path_length_support_scores, torch.Tensor):
        try:
            path_length_segment_method = _learned_segment_frozen_method(
                name="MLQDS_path_length_support_segment_head_diagnostic",
                scores=primary_scores,
                boundaries=selection_boundaries,
                compression_ratio=float(config.model.compression_ratio),
                segment_scores=primary_path_length_support_scores,
                points=selection_points,
                learned_segment_geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                learned_segment_score_blend_weight=float(config.model.learned_segment_score_blend_weight),
                learned_segment_fairness_preallocation=bool(config.model.learned_segment_fairness_preallocation),
                learned_segment_length_repair_fraction=float(config.model.learned_segment_length_repair_fraction),
            )
            ablation_methods.append(path_length_segment_method)
            head_sensitivity["MLQDS_path_length_support_segment_head_diagnostic"] = {
                **_head_ablation_sensitivity(
                    primary_scores=primary_scores,
                    ablation_scores=primary_scores,
                    primary_raw_predictions=primary_raw_preds if isinstance(primary_raw_preds, torch.Tensor) else None,
                    ablation_raw_predictions=primary_raw_preds if isinstance(primary_raw_preds, torch.Tensor) else None,
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
            path_length_allocation_method = _learned_segment_frozen_method(
                name="MLQDS_path_length_support_allocation_only_diagnostic",
                scores=primary_scores,
                boundaries=selection_boundaries,
                compression_ratio=float(config.model.compression_ratio),
                segment_scores=primary_path_length_support_scores,
                segment_point_scores=primary_segment_scores,
                points=selection_points,
                learned_segment_geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                learned_segment_score_blend_weight=float(config.model.learned_segment_score_blend_weight),
                learned_segment_fairness_preallocation=bool(config.model.learned_segment_fairness_preallocation),
                learned_segment_length_repair_fraction=float(config.model.learned_segment_length_repair_fraction),
            )
            ablation_methods.append(path_length_allocation_method)
            head_sensitivity["MLQDS_path_length_support_allocation_only_diagnostic"] = {
                **_head_ablation_sensitivity(
                    primary_scores=primary_scores,
                    ablation_scores=primary_scores,
                    primary_raw_predictions=primary_raw_preds if isinstance(primary_raw_preds, torch.Tensor) else None,
                    ablation_raw_predictions=primary_raw_preds if isinstance(primary_raw_preds, torch.Tensor) else None,
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
            behavior_raw_preds = _raw_predictions_without_factorized_head(
                model=trained.model,
                head_logits=primary_head_logits,
                disabled_head_name="conditional_behavior_utility",
            )
            behavior_scores = _scores_without_factorized_head(
                model=trained.model,
                head_logits=primary_head_logits,
                disabled_head_name="conditional_behavior_utility",
                boundaries=selection_boundaries,
                workload_type=workload_type,
                score_mode=config.model.mlqds_score_mode,
                score_temperature=float(config.model.mlqds_score_temperature),
                rank_confidence_weight=float(config.model.mlqds_rank_confidence_weight),
            )
            no_behavior = _learned_segment_frozen_method(
                name="MLQDS_without_behavior_utility_head",
                scores=behavior_scores,
                boundaries=selection_boundaries,
                compression_ratio=float(config.model.compression_ratio),
                segment_scores=primary_selector_segment_scores,
                segment_point_scores=primary_segment_scores,
                points=selection_points,
                learned_segment_geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                learned_segment_score_blend_weight=float(config.model.learned_segment_score_blend_weight),
                learned_segment_fairness_preallocation=bool(config.model.learned_segment_fairness_preallocation),
                learned_segment_length_repair_fraction=float(config.model.learned_segment_length_repair_fraction),
            )
            ablation_methods.append(no_behavior)
            head_sensitivity["MLQDS_without_behavior_utility_head"] = {
                **_head_ablation_sensitivity(
                    primary_scores=primary_scores,
                    ablation_scores=behavior_scores,
                    primary_raw_predictions=primary_raw_preds if isinstance(primary_raw_preds, torch.Tensor) else None,
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
            prior_scores = query_prior_predictability_scores(selection_points, query_prior_field).detach().cpu()
            ablation_methods.append(
                _learned_segment_frozen_method(
                    name="MLQDS_prior_field_only_score",
                    scores=prior_scores,
                    boundaries=selection_boundaries,
                    compression_ratio=float(config.model.compression_ratio),
                    points=selection_points,
                    learned_segment_geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                    learned_segment_score_blend_weight=float(config.model.learned_segment_score_blend_weight),
                    learned_segment_fairness_preallocation=bool(config.model.learned_segment_fairness_preallocation),
                    learned_segment_length_repair_fraction=float(config.model.learned_segment_length_repair_fraction),
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            freeze_failures["MLQDS_prior_field_only_score"] = str(exc)
        prior_ablation_fields = {
            "MLQDS_shuffled_prior_fields": _shuffled_query_prior_field(
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
                prior_feature_sensitivity = _prior_feature_sample_sensitivity(
                    points=selection_points,
                    primary_prior_field=query_prior_field,
                    ablation_prior_field=ablation_field,
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
                ablation_scores = getattr(ablation_method, "_score_cache", None)
                ablation_raw_preds = getattr(ablation_method, "_raw_pred_cache", None)
                prior_sensitivity[prior_sensitivity_key] = {
                    "sampled_prior_features": prior_feature_sensitivity,
                    "selector_score": _score_ablation_sensitivity(
                        primary_scores=primary_scores,
                        ablation_scores=ablation_scores if isinstance(ablation_scores, torch.Tensor) else None,
                        primary_mask=primary_mask,
                        ablation_mask=ablation_mask,
                    ),
                    "raw_prediction": _score_ablation_sensitivity(
                        primary_scores=primary_raw_preds if isinstance(primary_raw_preds, torch.Tensor) else None,
                        ablation_scores=ablation_raw_preds if isinstance(ablation_raw_preds, torch.Tensor) else None,
                        primary_mask=primary_mask,
                        ablation_mask=ablation_mask,
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

    ablation_evaluations: dict[str, MethodEvaluation] = {}
    mask_diagnostics: dict[str, dict[str, Any]] = {}
    for method in ablation_methods:
        mask_diagnostics[method.name] = _retained_mask_comparison(
            primary_mask=primary_mask,
            ablation_mask=method.retained_mask,
            expected_shape=primary_mask.shape,
        )
        ablation_evaluations[method.name] = evaluate_method(
            method=method,
            points=selection_points,
            boundaries=selection_boundaries,
            typed_queries=selection_workload.typed_queries,
            workload_map=eval_workload_map,
            compression_ratio=config.model.compression_ratio,
            query_cache=selection_query_cache,
        )

    payload = _causality_ablation_diagnostics_payload(
        primary=primary_eval,
        ablations=ablation_evaluations,
        mask_diagnostics=mask_diagnostics,
    )
    for name, tradeoff_diagnostics in payload["tradeoff_diagnostics"].items():
        if name in head_sensitivity:
            head_sensitivity[name]["query_useful_component_tradeoff"] = tradeoff_diagnostics
    payload.update(
        {
            "split": "checkpoint_selection",
            "diagnostic_only": True,
            "query_count": int(len(selection_workload.typed_queries)),
            "ablation_freeze_failures": freeze_failures,
            "prior_sensitivity_diagnostics": prior_sensitivity,
            "head_ablation_sensitivity_diagnostics": head_sensitivity,
        }
    )
    return payload


def run_experiment_pipeline(
    config: ExperimentConfig,
    trajectories: list[torch.Tensor],
    results_dir: str,
    save_model: str | None = None,
    save_queries_dir: str | None = None,
    save_simplified_dir: str | None = None,
    trajectory_mmsis: list[int] | None = None,
    validation_trajectories: list[torch.Tensor] | None = None,
    eval_trajectories: list[torch.Tensor] | None = None,
    eval_trajectory_mmsis: list[int] | None = None,
    trajectory_source_ids: list[int] | None = None,
    data_audit: dict[str, Any] | None = None,
) -> ExperimentOutputs:
    """Run training, matched evaluation, and shifted evaluation tables. See experiments/README.md for details."""
    pipeline_t0 = time.perf_counter()
    train_workload_map, eval_workload_map = resolve_workload_maps(config.query.workload)
    if eval_trajectories is None:
        print(
            f"[pipeline] {len(trajectories)} trajectories, workload={workload_name(eval_workload_map)}",
            flush=True,
        )
    else:
        validation_part = (
            f", validation={len(validation_trajectories)} trajectories"
            if validation_trajectories is not None
            else ""
        )
        print(
            f"[pipeline] train={len(trajectories)} trajectories{validation_part}, "
            f"eval={len(eval_trajectories)} trajectories, "
            f"workload={workload_name(eval_workload_map)}",
            flush=True,
        )

    seeds = derive_seed_bundle(config.data.seed)
    selection_metric = str(getattr(config.model, "checkpoint_selection_metric", "score")).lower()
    validation_score_every = int(getattr(config.model, "validation_score_every", 0) or 0)
    needs_validation_score = selection_metric in {"score", "uniform_gap"} or validation_score_every > 0
    with _phase("split"):
        data_split = prepare_experiment_split(
            config=config,
            seeds=seeds,
            trajectories=trajectories,
            needs_validation_score=needs_validation_score,
            trajectory_mmsis=trajectory_mmsis,
            validation_trajectories=validation_trajectories,
            eval_trajectories=eval_trajectories,
            eval_trajectory_mmsis=eval_trajectory_mmsis,
            trajectory_source_ids=trajectory_source_ids,
        )
        train_traj = data_split.train_traj
        test_traj = data_split.test_traj
        selection_traj = data_split.selection_traj
        train_mmsis = data_split.train_mmsis
        test_mmsis = data_split.test_mmsis
        train_source_ids = data_split.train_source_ids

    with _phase("build-datasets"):
        datasets = build_experiment_datasets(data_split)
        train_points = datasets.train_points
        test_points = datasets.test_points
        selection_points = datasets.selection_points
        train_boundaries = datasets.train_boundaries
        test_boundaries = datasets.test_boundaries
        selection_boundaries = datasets.selection_boundaries

    with _phase("generate-workloads"):
        workloads = generate_experiment_workloads(
            config=config,
            seeds=seeds,
            train_traj=train_traj,
            test_traj=test_traj,
            selection_traj=selection_traj,
            train_points=train_points,
            test_points=test_points,
            selection_points=selection_points,
            train_boundaries=train_boundaries,
            test_boundaries=test_boundaries,
            selection_boundaries=selection_boundaries,
            train_workload_map=train_workload_map,
            eval_workload_map=eval_workload_map,
        )
        train_workload = workloads.train_workload
        train_label_workloads = workloads.train_label_workloads
        train_label_workload_seeds = workloads.train_label_workload_seeds
        eval_workload = workloads.eval_workload
        selection_workload = workloads.selection_workload

    range_diagnostics_summary: dict[str, Any] = {}
    range_diagnostics_rows: list[dict[str, Any]] = []
    range_runtime_caches = {
        "train": RangeRuntimeCache(),
        "eval": RangeRuntimeCache(),
        "selection": RangeRuntimeCache(),
    }
    workload_distribution_comparison: dict[str, Any] = {"deltas_vs_eval": {}}

    if save_queries_dir:
        with _phase("write-queries-geojson"):
            write_queries_geojson(save_queries_dir, eval_workload.typed_queries)

    reset_cuda_peak_memory_stats()
    train_labels: tuple[torch.Tensor, torch.Tensor] | None = None
    range_training_target_mode = str(getattr(config.model, "range_training_target_mode", "point_value")).lower()
    range_replicate_target_aggregation = str(
        getattr(config.model, "range_replicate_target_aggregation", "label_mean")
    ).lower()
    if range_replicate_target_aggregation not in {"label_mean", "label_max", "frequency_mean"}:
        raise ValueError(
            "range_replicate_target_aggregation must be 'label_mean', 'label_max', or 'frequency_mean'."
        )
    if len(train_label_workloads) > 1 and not is_workload_blind_model_type(config.model.model_type):
        raise RuntimeError("range_train_workload_replicates > 1 is only valid for workload-blind model types.")
    range_training_target_transform: dict[str, Any] = {
        "mode": range_training_target_mode,
        "enabled": False,
    }
    range_target_balance_diagnostics: dict[str, Any] = {
        "enabled": False,
        "mode": str(getattr(config.model, "range_target_balance_mode", "none")).lower(),
    }
    range_training_label_aggregation: dict[str, Any] = {
        "enabled": False,
        "replicate_count": int(len(train_label_workloads)),
        "seeds": [int(seed) for seed in train_label_workload_seeds],
    }
    teacher_distillation_diagnostics: dict[str, Any] = {
        "enabled": False,
        "mode": str(getattr(config.model, "range_teacher_distillation_mode", "none")),
    }
    if range_training_target_mode == "query_useful_v1_factorized":
        range_training_target_transform.update(
            {
                "enabled": True,
                "target_family": "QueryUsefulV1Factorized",
                "final_success_allowed": True,
            }
        )
    selection_query_cache: EvaluationQueryCache | None = None
    selection_geometry_scores: torch.Tensor | None = None
    mlqds_range_geometry_blend = max(0.0, min(1.0, float(getattr(config.model, "mlqds_range_geometry_blend", 0.0))))
    with _phase("range-training-prep"):
        train_label_sets: list[tuple[torch.Tensor, torch.Tensor]] = []
        train_component_label_sets: list[dict[str, torch.Tensor] | None] = []
        if range_training_target_mode != "query_useful_v1_factorized" or range_teacher_distillation_enabled(config.model):
            for replicate_index, label_workload in enumerate(train_label_workloads):
                label_cache_name = "train" if replicate_index == 0 else f"train_r{replicate_index}"
                runtime_cache = range_runtime_caches["train"] if replicate_index == 0 else RangeRuntimeCache()
                label_result = prepare_range_label_cache(
                    cache_label=label_cache_name,
                    points=train_points,
                    boundaries=train_boundaries,
                    workload=label_workload,
                    workload_map=train_workload_map,
                    config=config,
                    seed=train_label_workload_seeds[replicate_index],
                    runtime_cache=runtime_cache,
                    range_boundary_prior_weight=float(getattr(config.model, "range_boundary_prior_weight", 0.0)),
                )
                if label_result is not None:
                    train_label_sets.append(label_result)
                    train_component_label_sets.append(runtime_cache.component_labels)
        if train_label_sets:
            train_labels = train_label_sets[0]
            if (
                len(train_label_sets) > 1
                and range_training_target_mode == "point_value"
                and not range_teacher_distillation_enabled(config.model)
            ):
                if range_replicate_target_aggregation == "frequency_mean":
                    raise ValueError("range_replicate_target_aggregation='frequency_mean' requires a frequency target.")
                labels, labelled_mask, aggregation_diagnostics = aggregate_range_label_sets(
                    train_label_sets,
                    aggregation="max" if range_replicate_target_aggregation == "label_max" else "mean",
                )
                train_labels = (labels, labelled_mask)
                range_training_label_aggregation.update(aggregation_diagnostics)
                range_training_label_aggregation["enabled"] = True
                range_training_label_aggregation["replicate_target_aggregation"] = (
                    range_replicate_target_aggregation
                )
        if (
            selection_workload is not None
            and selection_points is not None
            and selection_boundaries is not None
            and len(range_only_queries(selection_workload.typed_queries)) == len(selection_workload.typed_queries)
        ):
            selection_query_cache = EvaluationQueryCache.for_workload(
                selection_points,
                selection_boundaries,
                selection_workload.typed_queries,
            )
            range_runtime_caches["selection"].query_cache = selection_query_cache
            if mlqds_range_geometry_blend > 0.0:
                selection_labels = prepare_range_label_cache(
                    cache_label="selection",
                    points=selection_points,
                    boundaries=selection_boundaries,
                    workload=selection_workload,
                    workload_map=eval_workload_map,
                    config=config,
                    seed=seeds.eval_query_seed + 17,
                    runtime_cache=range_runtime_caches["selection"],
                    range_boundary_prior_weight=float(getattr(config.model, "range_boundary_prior_weight", 0.0)),
                )
                if selection_labels is not None:
                    labels, _labelled_mask = selection_labels
                    _, selection_type_id = workload_type_head(single_workload_type(eval_workload_map))
                    selection_geometry_scores = labels[:, selection_type_id].float()
    if (
        train_labels is not None
        and len(range_only_queries(train_workload.typed_queries)) == len(train_workload.typed_queries)
    ):
        print("  prepared train range labels for precomputed training target", flush=True)
    if selection_query_cache is not None:
        print("  prepared checkpoint-validation range query cache", flush=True)
    if range_teacher_distillation_enabled(config.model):
        if not is_workload_blind_model_type(config.model.model_type):
            raise RuntimeError("range teacher distillation is only valid for workload-blind model types.")
        if train_labels is None:
            raise RuntimeError("range teacher distillation requires precomputed range training labels.")
        for label_workload in train_label_workloads:
            if len(range_only_queries(label_workload.typed_queries)) != len(label_workload.typed_queries):
                raise RuntimeError("range teacher distillation requires pure range training workloads.")
        teacher_config = build_range_teacher_config(config.model)
        print(
            f"  range teacher distillation enabled: mode={config.model.range_teacher_distillation_mode} "
            f"teacher_epochs={teacher_config.epochs} "
            f"replicates={len(train_label_workloads)}",
            flush=True,
        )
        distilled_label_sets: list[tuple[torch.Tensor, torch.Tensor]] = []
        per_teacher: list[dict[str, Any]] = []
        for replicate_index, label_workload in enumerate(train_label_workloads):
            with _phase(f"train-range-teacher-r{replicate_index} ({teacher_config.epochs} epochs)"):
                teacher_trained = train_model(
                    train_trajectories=train_traj,
                    train_boundaries=train_boundaries,
                    workload=label_workload,
                    model_config=teacher_config,
                    seed=seeds.torch_seed + 31 + replicate_index,
                    train_workload_map=train_workload_map,
                    precomputed_labels=train_label_sets[replicate_index],
                    train_trajectory_source_ids=train_source_ids,
                    train_trajectory_mmsis=train_mmsis,
                )
            with _phase(f"distill-range-teacher-r{replicate_index}-labels"):
                distilled_labels, replicate_diagnostics = distill_range_teacher_labels(
                    teacher=teacher_trained,
                    teacher_model_type=teacher_config.model_type,
                    points=train_points,
                    boundaries=train_boundaries,
                    workload=label_workload,
                    model_config=config.model,
                )
            replicate_diagnostics["replicate_index"] = int(replicate_index)
            replicate_diagnostics["seed"] = int(train_label_workload_seeds[replicate_index])
            per_teacher.append(replicate_diagnostics)
            distilled_label_sets.append(distilled_labels)
        if len(distilled_label_sets) == 1:
            train_labels = distilled_label_sets[0]
            teacher_distillation_diagnostics = dict(per_teacher[0])
        else:
            teacher_aggregation_mode = "max" if range_replicate_target_aggregation == "label_max" else "mean"
            labels, labelled_mask, aggregation_diagnostics = aggregate_range_label_sets(
                distilled_label_sets,
                source="range_teacher_distillation_replicates",
                aggregation=teacher_aggregation_mode,
            )
            train_labels = (labels, labelled_mask)
            positive = labelled_mask[:, QUERY_TYPE_ID_RANGE] & (labels[:, QUERY_TYPE_ID_RANGE] > 0.0)
            teacher_distillation_diagnostics = {
                "enabled": True,
                "mode": str(getattr(config.model, "range_teacher_distillation_mode", "none")),
                "teacher_model_type": str(teacher_config.model_type),
                "teacher_epochs": int(teacher_config.epochs),
                "replicate_count": int(len(distilled_label_sets)),
                "replicate_target_aggregation": range_replicate_target_aggregation,
                "aggregation": aggregation_diagnostics,
                "per_replicate": per_teacher,
                "labelled_point_count": int(labelled_mask[:, QUERY_TYPE_ID_RANGE].sum().item()),
                "positive_label_count": int(positive.sum().item()),
                "positive_label_fraction": float(positive.sum().item() / max(1, int(labels.shape[0]))),
                "positive_label_mass": (
                    float(labels[positive, QUERY_TYPE_ID_RANGE].sum().item()) if bool(positive.any().item()) else 0.0
                ),
                "budget_loss_ratios": list(getattr(config.model, "budget_loss_ratios", [])),
                "mlqds_temporal_fraction": float(getattr(config.model, "mlqds_temporal_fraction", 0.0)),
                "mlqds_hybrid_mode": str(getattr(config.model, "mlqds_hybrid_mode", "fill")),
            }
            range_training_label_aggregation.update(aggregation_diagnostics)
            range_training_label_aggregation["enabled"] = True
            range_training_label_aggregation["target_mode"] = "teacher_distillation"
            range_training_label_aggregation["replicate_target_aggregation"] = range_replicate_target_aggregation
            print(
                f"  distilled range labels: replicate_count={len(distilled_label_sets)} "
                f"positives={teacher_distillation_diagnostics['positive_label_count']} "
                f"fraction={teacher_distillation_diagnostics['positive_label_fraction']:.4f} "
                f"mass={teacher_distillation_diagnostics['positive_label_mass']:.4f}",
                flush=True,
            )
    elif range_training_target_mode in {
        "query_spine_frequency",
        "query_residual_frequency",
        "set_utility_frequency",
        "local_swap_utility_frequency",
        "local_swap_gain_cost_frequency",
    }:
        if train_labels is None:
            raise RuntimeError(f"{range_training_target_mode} target mode requires precomputed range training labels.")
        if len(train_label_sets) > 1:
            raise RuntimeError(f"{range_training_target_mode} does not yet support multiple train workload replicates.")
        target_phase = range_training_target_mode.replace("_", "-")
        with _phase(f"range-{target_phase}-target"):
            labels, labelled_mask = train_labels
            target_fn = (
                range_local_swap_gain_cost_frequency_training_labels
                if range_training_target_mode == "local_swap_gain_cost_frequency"
                else (
                    range_local_swap_utility_frequency_training_labels
                    if range_training_target_mode == "local_swap_utility_frequency"
                    else (
                        range_set_utility_frequency_training_labels
                        if range_training_target_mode == "set_utility_frequency"
                        else (
                            range_query_residual_frequency_training_labels
                            if range_training_target_mode == "query_residual_frequency"
                            else range_query_spine_frequency_training_labels
                        )
                    )
                )
            )
            labels, labelled_mask, range_training_target_transform = target_fn(
                labels=labels,
                labelled_mask=labelled_mask,
                points=train_points,
                boundaries=train_boundaries,
                typed_queries=train_workload.typed_queries,
                model_config=config.model,
            )
            range_training_target_transform["enabled"] = True
            range_training_target_transform["replicate_count"] = len(train_label_sets)
            train_labels = (labels, labelled_mask)
            print(
                f"  {target_phase} target: "
                f"positives={range_training_target_transform['positive_label_count']} "
                f"fraction={range_training_target_transform['positive_label_fraction']:.4f} "
                f"mass={range_training_target_transform['positive_label_mass']:.4f}",
                flush=True,
            )
    elif range_training_target_mode in {
        "retained_frequency",
        "global_budget_retained_frequency",
        "marginal_coverage_frequency",
        "historical_prior_retained_frequency",
        "structural_retained_frequency",
    }:
        if train_labels is None:
            raise RuntimeError(
                f"{range_training_target_mode} target mode requires precomputed range training labels."
            )
        target_fn = (
            range_marginal_coverage_training_labels
            if range_training_target_mode == "marginal_coverage_frequency"
            else range_global_budget_retained_frequency_training_labels
            if range_training_target_mode == "global_budget_retained_frequency"
            else range_structural_retained_frequency_training_labels
            if range_training_target_mode == "structural_retained_frequency"
            else range_historical_prior_retained_frequency_training_labels
            if range_training_target_mode == "historical_prior_retained_frequency"
            else range_retained_frequency_training_labels
        )
        aggregate_target_fn = (
            aggregate_range_marginal_coverage_training_labels
            if range_training_target_mode == "marginal_coverage_frequency"
            else aggregate_range_global_budget_retained_frequency_training_labels
            if range_training_target_mode == "global_budget_retained_frequency"
            else aggregate_range_structural_retained_frequency_training_labels
            if range_training_target_mode == "structural_retained_frequency"
            else aggregate_range_retained_frequency_training_labels
        )
        target_phase = range_training_target_mode.replace("_", "-")
        with _phase(f"range-{target_phase}-target"):
            if len(train_label_sets) > 1:
                if range_replicate_target_aggregation == "frequency_mean":
                    if range_training_target_mode == "historical_prior_retained_frequency":
                        raise RuntimeError(
                            "historical_prior_retained_frequency does not support "
                            "range_replicate_target_aggregation='frequency_mean'; use label_mean or label_max."
                        )
                    aggregate_target_kwargs = {
                        "label_sets": train_label_sets,
                        "boundaries": train_boundaries,
                        "model_config": config.model,
                    }
                    if range_training_target_mode == "structural_retained_frequency":
                        aggregate_target_kwargs["points"] = train_points
                    labels, labelled_mask, range_training_target_transform = aggregate_target_fn(
                        **aggregate_target_kwargs
                    )
                    range_training_label_aggregation["enabled"] = True
                    range_training_label_aggregation["target_mode"] = range_training_target_mode
                    range_training_label_aggregation["replicate_target_aggregation"] = "frequency_mean"
                else:
                    labels, labelled_mask, aggregation_diagnostics = aggregate_range_label_sets(
                        label_sets=train_label_sets,
                        source=(
                            f"range_label_{'max' if range_replicate_target_aggregation == 'label_max' else 'mean'}"
                            f"_before_{range_training_target_mode}"
                        ),
                        aggregation="max" if range_replicate_target_aggregation == "label_max" else "mean",
                    )
                    range_training_label_aggregation.update(aggregation_diagnostics)
                    range_training_label_aggregation["enabled"] = True
                    range_training_label_aggregation["target_mode"] = range_training_target_mode
                    range_training_label_aggregation["replicate_target_aggregation"] = (
                        range_replicate_target_aggregation
                    )
                    target_kwargs = {
                        "labels": labels,
                        "labelled_mask": labelled_mask,
                        "boundaries": train_boundaries,
                        "model_config": config.model,
                    }
                    if range_training_target_mode in {
                        "historical_prior_retained_frequency",
                        "structural_retained_frequency",
                    }:
                        target_kwargs["points"] = train_points
                    labels, labelled_mask, range_training_target_transform = target_fn(**target_kwargs)
                    range_training_target_transform["label_aggregation"] = aggregation_diagnostics
                range_training_target_transform["replicate_target_aggregation"] = (
                    range_replicate_target_aggregation
                )
            else:
                labels, labelled_mask = train_labels
                target_kwargs = {
                    "labels": labels,
                    "labelled_mask": labelled_mask,
                    "boundaries": train_boundaries,
                    "model_config": config.model,
                }
                if range_training_target_mode in {
                    "historical_prior_retained_frequency",
                    "structural_retained_frequency",
                }:
                    target_kwargs["points"] = train_points
                labels, labelled_mask, range_training_target_transform = target_fn(**target_kwargs)
            range_training_target_transform["enabled"] = True
            range_training_target_transform["replicate_count"] = len(train_label_sets)
            train_labels = (labels, labelled_mask)
            print(
                f"  {target_phase} target: positives={range_training_target_transform['positive_label_count']} "
                f"fraction={range_training_target_transform['positive_label_fraction']:.4f} "
                f"mass={range_training_target_transform['positive_label_mass']:.4f}",
                flush=True,
            )
    elif range_training_target_mode not in {"point_value", "query_useful_v1_factorized"}:
        if range_training_target_mode in {"component_retained_frequency", "continuity_retained_frequency"}:
            if train_labels is None:
                raise RuntimeError(
                    f"{range_training_target_mode} target mode requires precomputed range training labels."
                )
            if not train_component_label_sets or any(component_labels is None for component_labels in train_component_label_sets):
                raise RuntimeError(
                    f"{range_training_target_mode} requires range component labels; use range_label_mode=usefulness."
                )
            target_fn = (
                range_continuity_retained_frequency_training_labels
                if range_training_target_mode == "continuity_retained_frequency"
                else range_component_retained_frequency_training_labels
            )
            aggregate_target_fn = (
                aggregate_range_continuity_retained_frequency_training_labels
                if range_training_target_mode == "continuity_retained_frequency"
                else aggregate_range_component_retained_frequency_training_labels
            )
            target_phase = range_training_target_mode.replace("_", "-")
            with _phase(f"range-{target_phase}-target"):
                if len(train_label_sets) > 1:
                    if range_replicate_target_aggregation == "frequency_mean":
                        labels, labelled_mask, range_training_target_transform = (
                            aggregate_target_fn(
                                label_sets=train_label_sets,
                                component_label_sets=train_component_label_sets,
                                boundaries=train_boundaries,
                                model_config=config.model,
                            )
                        )
                        range_training_label_aggregation["replicate_target_aggregation"] = "frequency_mean"
                    else:
                        aggregation_mode = "max" if range_replicate_target_aggregation == "label_max" else "mean"
                        labels, labelled_mask, component_labels, aggregation_diagnostics = (
                            aggregate_range_component_label_sets(
                                label_sets=train_label_sets,
                                component_label_sets=train_component_label_sets,
                                aggregation=aggregation_mode,
                            )
                        )
                        range_training_label_aggregation.update(aggregation_diagnostics)
                        range_training_label_aggregation["replicate_target_aggregation"] = (
                            range_replicate_target_aggregation
                        )
                        labels, labelled_mask, range_training_target_transform = (
                            target_fn(
                                labels=labels,
                                labelled_mask=labelled_mask,
                                component_labels=component_labels,
                                boundaries=train_boundaries,
                                model_config=config.model,
                            )
                        )
                        range_training_target_transform["label_aggregation"] = aggregation_diagnostics
                    range_training_label_aggregation["enabled"] = True
                    range_training_label_aggregation["target_mode"] = range_training_target_mode
                else:
                    labels, labelled_mask = train_labels
                    component_labels = train_component_label_sets[0]
                    if component_labels is None:
                        raise RuntimeError("component_retained_frequency requires component labels.")
                    labels, labelled_mask, range_training_target_transform = (
                        target_fn(
                            labels=labels,
                            labelled_mask=labelled_mask,
                            component_labels=component_labels,
                            boundaries=train_boundaries,
                            model_config=config.model,
                        )
                    )
                range_training_target_transform["enabled"] = True
                range_training_target_transform["replicate_count"] = len(train_label_sets)
                range_training_target_transform["replicate_target_aggregation"] = range_replicate_target_aggregation
                train_labels = (labels, labelled_mask)
                print(
                    f"  {target_phase} target: "
                    f"positives={range_training_target_transform['positive_label_count']} "
                    f"fraction={range_training_target_transform['positive_label_fraction']:.4f} "
                    f"mass={range_training_target_transform['positive_label_mass']:.4f}",
                    flush=True,
                )
        else:
            raise RuntimeError(
                "range_training_target_mode must be 'point_value', 'retained_frequency', "
                "'global_budget_retained_frequency', 'historical_prior_retained_frequency', "
                "'marginal_coverage_frequency', 'query_spine_frequency', "
                "'query_residual_frequency', 'set_utility_frequency', 'local_swap_utility_frequency', "
                "'local_swap_gain_cost_frequency', 'structural_retained_frequency', "
                "'component_retained_frequency', or "
                "'continuity_retained_frequency', or 'query_useful_v1_factorized'."
            )
    range_target_balance_mode = str(getattr(config.model, "range_target_balance_mode", "none")).lower()
    if range_target_balance_mode != "none":
        if train_labels is None:
            raise RuntimeError("range_target_balance_mode requires precomputed range training labels.")
        with _phase("range-target-balance"):
            labels, labelled_mask = train_labels
            labels, labelled_mask, range_target_balance_diagnostics = balance_range_training_target_by_trajectory(
                labels=labels,
                labelled_mask=labelled_mask,
                boundaries=train_boundaries,
                mode=range_target_balance_mode,
            )
            train_labels = (labels, labelled_mask)
            print(
                f"  target balance={range_target_balance_diagnostics['mode']} "
                f"positives={range_target_balance_diagnostics['positive_label_count']} "
                f"mass={range_target_balance_diagnostics['positive_label_mass']:.4f} "
                f"trajectories={range_target_balance_diagnostics['balanced_trajectory_count']}",
                flush=True,
            )
    if range_training_target_mode != "query_useful_v1_factorized":
        range_training_target_transform.setdefault("target_family", "legacy_range_useful_scalar")
        range_training_target_transform.setdefault("final_success_allowed", False)
        range_training_target_transform.setdefault(
            "legacy_reason",
            "Old RangeUseful/scalar-target diagnostic path. "
            "Not valid for query-driven rework acceptance.",
        )
    with _phase(f"train-model ({config.model.epochs} epochs)"):
        trained = train_model(
            train_trajectories=train_traj,
            train_boundaries=train_boundaries,
            workload=train_workload,
            model_config=config.model,
            seed=seeds.torch_seed,
            train_workload_map=train_workload_map,
            validation_trajectories=selection_traj,
            validation_boundaries=selection_boundaries,
            validation_workload=selection_workload,
            validation_workload_map=eval_workload_map if selection_workload is not None else None,
            precomputed_labels=train_labels,
            validation_points=selection_points,
            precomputed_validation_query_cache=selection_query_cache,
            precomputed_validation_geometry_scores=selection_geometry_scores,
            train_trajectory_source_ids=train_source_ids,
            train_trajectory_mmsis=train_mmsis,
            query_prior_workloads=train_label_workloads,
            query_prior_workload_seeds=train_label_workload_seeds,
        )
    training_cuda_memory = cuda_memory_snapshot()
    if training_cuda_memory.get("available"):
        print(
            f"  train_cuda_peak_allocated={training_cuda_memory['max_allocated_mb']:.1f} MiB  "
            f"peak_reserved={training_cuda_memory['max_reserved_mb']:.1f} MiB",
            flush=True,
        )

    if save_model:
        with _phase("save-model"):
            artifacts = ModelArtifacts(
                model=trained.model,
                scaler=trained.scaler,
                config=config,
                epochs_trained=trained.epochs_trained,
                workload_type=single_workload_type(eval_workload_map),
                query_prior_field=trained.feature_context.get("query_prior_field"),
            )
            save_checkpoint(save_model, artifacts)
            print(
                f"  saved checkpoint to {save_model}  "
                f"(epochs_trained={trained.epochs_trained}, "
                f"best_epoch={trained.best_epoch}, best_loss={trained.best_loss:.8f}, "
                f"workload={workload_name(eval_workload_map)})",
                flush=True,
            )
    methods = build_primary_methods(
        trained=trained,
        eval_workload=eval_workload,
        eval_workload_map=eval_workload_map,
        config=config,
        trajectory_mmsis=test_mmsis,
    )
    retention_methods = list(methods)
    workload_blind_eval = is_workload_blind_model_type(config.model.model_type)
    audit_ratios = _range_audit_ratios(config)
    selector_budget_ratios = tuple(
        sorted({float(config.model.compression_ratio), *(float(ratio) for ratio in audit_ratios)})
    )
    if str(getattr(config.model, "selector_type", "temporal_hybrid")).lower() == "learned_segment_budget_v1":
        selector_budget_diagnostics = {
            "train": learned_segment_budget_diagnostics(train_boundaries, selector_budget_ratios),
            "eval": learned_segment_budget_diagnostics(test_boundaries, selector_budget_ratios),
        }
    else:
        selector_budget_diagnostics = {
            "train": temporal_hybrid_selector_budget_diagnostics(
                train_boundaries,
                selector_budget_ratios,
                temporal_fraction=float(config.model.mlqds_temporal_fraction),
                hybrid_mode=str(config.model.mlqds_hybrid_mode),
                min_learned_swaps=int(config.model.mlqds_min_learned_swaps),
            ),
            "eval": temporal_hybrid_selector_budget_diagnostics(
                test_boundaries,
                selector_budget_ratios,
                temporal_fraction=float(config.model.mlqds_temporal_fraction),
                hybrid_mode=str(config.model.mlqds_hybrid_mode),
                min_learned_swaps=int(config.model.mlqds_min_learned_swaps),
            ),
        }
    selection_causality_diagnostics: dict[str, Any] = {"available": False, "reason": "not_run"}
    if workload_blind_eval:
        with _phase("selection-causality-diagnostics"):
            selection_causality_diagnostics = _selection_causality_diagnostics(
                trained=trained,
                selection_points=selection_points,
                selection_boundaries=selection_boundaries,
                selection_workload=selection_workload,
                eval_workload_map=eval_workload_map,
                selection_query_cache=selection_query_cache,
                config=config,
                seeds=seeds,
            )
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
        with _phase("freeze-retained-masks"):
            for method in methods:
                with _phase(f"  freeze {method.name}"):
                    freeze_t0 = time.perf_counter()
                    frozen_primary_masks[method.name] = method.simplify(
                        test_points,
                        test_boundaries,
                        config.model.compression_ratio,
                    ).detach().cpu()
                    setattr(method, "latency_ms", float((time.perf_counter() - freeze_t0) * 1000.0))
                    score_cache = getattr(method, "_score_cache", None)
                    if isinstance(score_cache, torch.Tensor):
                        frozen_primary_scores[method.name] = score_cache.detach().cpu().float()
                    raw_pred_cache = getattr(method, "_raw_pred_cache", None)
                    if isinstance(raw_pred_cache, torch.Tensor):
                        frozen_primary_raw_preds[method.name] = raw_pred_cache.detach().cpu().float()
                    head_logit_cache = getattr(method, "_head_logit_cache", None)
                    if isinstance(head_logit_cache, torch.Tensor):
                        frozen_primary_head_logits[method.name] = head_logit_cache.detach().cpu().float()
                    segment_score_cache = getattr(method, "_segment_score_cache", None)
                    if isinstance(segment_score_cache, torch.Tensor):
                        frozen_primary_segment_scores[method.name] = segment_score_cache.detach().cpu().float()
                    path_length_support_cache = getattr(method, "_path_length_support_score_cache", None)
                    if isinstance(path_length_support_cache, torch.Tensor):
                        frozen_primary_path_length_support_scores[method.name] = (
                            path_length_support_cache.detach().cpu().float()
                        )
                    selector_segment_score_cache = getattr(method, "_selector_segment_score_cache", None)
                    if isinstance(selector_segment_score_cache, torch.Tensor):
                        frozen_primary_selector_segment_scores[method.name] = (
                            selector_segment_score_cache.detach().cpu().float()
                        )
            primary_scores = frozen_primary_scores.get("MLQDS")
            primary_raw_preds = frozen_primary_raw_preds.get("MLQDS")
            if primary_scores is not None and str(getattr(config.model, "selector_type", "")).lower() == "learned_segment_budget_v1":
                primary_segment_scores = frozen_primary_segment_scores.get("MLQDS")
                primary_path_length_support_scores = frozen_primary_path_length_support_scores.get("MLQDS")
                primary_selector_segment_scores = frozen_primary_selector_segment_scores.get("MLQDS")
                trace_mask, trace = simplify_with_learned_segment_budget_v1_with_trace(
                    primary_scores,
                    test_boundaries,
                    float(config.model.compression_ratio),
                    segment_scores=primary_selector_segment_scores,
                    segment_point_scores=primary_segment_scores,
                    points=test_points.detach().cpu().float(),
                    geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                    segment_score_point_blend_weight=float(config.model.learned_segment_score_blend_weight),
                    fairness_preallocation_enabled=bool(config.model.learned_segment_fairness_preallocation),
                    length_repair_fraction=float(config.model.learned_segment_length_repair_fraction),
                    segment_score_source_label=_selector_segment_score_source_label(
                        segment_scores=primary_segment_scores,
                        path_length_support_scores=primary_path_length_support_scores,
                        length_support_blend_weight=float(config.model.learned_segment_length_support_blend_weight),
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
                    0.35 if compression_for_trace >= 0.10 else 0.25 if compression_for_trace >= 0.05 else 0.0
                )
                if learned_fraction_min_for_trace > 0.0:
                    trace["score_protected_length_feasibility"] = _score_protected_length_feasibility(
                        scores=primary_scores,
                        points=test_points,
                        boundaries=test_boundaries,
                        compression_ratio=compression_for_trace,
                        learned_slot_fraction_min=learned_fraction_min_for_trace,
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
                except Exception as exc:  # pragma: no cover - optional diagnostic should not gate eval.
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
                    except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
                        causal_ablation_freeze_failures["MLQDS_without_geometry_tie_breaker"] = str(exc)
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
                        learned_segment_geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                        learned_segment_score_blend_weight=float(config.model.learned_segment_score_blend_weight),
                        learned_segment_fairness_preallocation=bool(config.model.learned_segment_fairness_preallocation),
                        learned_segment_length_repair_fraction=float(config.model.learned_segment_length_repair_fraction),
                    )
                )
                if primary_segment_scores is not None:
                    neutral_segment_scores = _neutral_segment_scores_for_ablation(primary_segment_scores)
                    no_segment_selector_scores = blend_segment_support_scores(
                        segment_scores=neutral_segment_scores,
                        path_length_support_scores=primary_path_length_support_scores,
                        path_length_support_weight=float(config.model.learned_segment_length_support_blend_weight),
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
                        learned_segment_geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                        learned_segment_score_blend_weight=float(config.model.learned_segment_score_blend_weight),
                        learned_segment_fairness_preallocation=bool(config.model.learned_segment_fairness_preallocation),
                        learned_segment_length_repair_fraction=float(config.model.learned_segment_length_repair_fraction),
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
                            learned_segment_geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                            learned_segment_score_blend_weight=float(config.model.learned_segment_score_blend_weight),
                            learned_segment_fairness_preallocation=bool(config.model.learned_segment_fairness_preallocation),
                            learned_segment_length_repair_fraction=float(config.model.learned_segment_length_repair_fraction),
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
                        allocation_sensitivity["ablation_mode"] = "neutral_constant_segment_scores_for_allocation_only"
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
                            learned_segment_geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                            learned_segment_score_blend_weight=float(config.model.learned_segment_score_blend_weight),
                            learned_segment_fairness_preallocation=bool(config.model.learned_segment_fairness_preallocation),
                            learned_segment_length_repair_fraction=float(config.model.learned_segment_length_repair_fraction),
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
                        point_score_allocation_sensitivity["disabled_head_name"] = "segment_budget_target"
                        point_score_allocation_sensitivity["ablation_mode"] = "point_score_top20_mean_for_allocation_only"
                        point_score_allocation_sensitivity["diagnostic_only"] = True
                        point_score_allocation_sensitivity["allocation_score_source"] = "point_score_top20_mean"
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
                        for diagnostic_name, authority_scores, authority_mode in allocation_authority_variants:
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
                            authority_sensitivity["allocation_score_source"] = "selector_segment_score_bands"
                            head_ablation_sensitivity_diagnostics[diagnostic_name] = authority_sensitivity

                        segment_point_blend_ablation_method = _learned_segment_frozen_method(
                            name="MLQDS_without_segment_budget_point_blend_only",
                            scores=primary_scores,
                            boundaries=test_boundaries,
                            compression_ratio=float(config.model.compression_ratio),
                            segment_scores=primary_selector_segment_scores,
                            segment_point_scores=primary_segment_scores,
                            points=test_points,
                            learned_segment_geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                            learned_segment_score_blend_weight=0.0,
                            learned_segment_fairness_preallocation=bool(config.model.learned_segment_fairness_preallocation),
                            learned_segment_length_repair_fraction=float(config.model.learned_segment_length_repair_fraction),
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
                        point_blend_sensitivity["ablation_mode"] = "disable_segment_score_point_blend_only"
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
                                learned_segment_geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                                learned_segment_score_blend_weight=float(config.model.learned_segment_score_blend_weight),
                                learned_segment_fairness_preallocation=False,
                                learned_segment_length_repair_fraction=float(config.model.learned_segment_length_repair_fraction),
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
                            learned_segment_geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                            learned_segment_score_blend_weight=float(config.model.learned_segment_score_blend_weight),
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
                            learned_segment_geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                            learned_segment_score_blend_weight=float(config.model.learned_segment_score_blend_weight),
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
                    except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
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
                            learned_segment_geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                            learned_segment_score_blend_weight=float(config.model.learned_segment_score_blend_weight),
                            learned_segment_fairness_preallocation=bool(config.model.learned_segment_fairness_preallocation),
                            learned_segment_length_repair_fraction=float(config.model.learned_segment_length_repair_fraction),
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
                        head_ablation_sensitivity_diagnostics["MLQDS_without_behavior_utility_head"] = (
                            behavior_sensitivity
                        )
                    except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
                        causal_ablation_freeze_failures["MLQDS_without_behavior_utility_head"] = str(exc)
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
                except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
                    causal_ablation_freeze_failures["MLQDS_untrained_model"] = str(exc)
                query_prior_field = trained.feature_context.get("query_prior_field")
                if isinstance(query_prior_field, dict):
                    prior_scores = query_prior_predictability_scores(test_points, query_prior_field).detach().cpu()
                    causality_ablation_methods.append(
                        _learned_segment_frozen_method(
                            name="MLQDS_prior_field_only_score",
                            scores=prior_scores,
                            boundaries=test_boundaries,
                            compression_ratio=float(config.model.compression_ratio),
                            points=test_points,
                            learned_segment_geometry_gain_weight=float(config.model.learned_segment_geometry_gain_weight),
                            learned_segment_score_blend_weight=float(config.model.learned_segment_score_blend_weight),
                            learned_segment_fairness_preallocation=bool(config.model.learned_segment_fairness_preallocation),
                            learned_segment_length_repair_fraction=float(config.model.learned_segment_length_repair_fraction),
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
                        shuffled_prior_raw_preds = getattr(shuffled_prior_method, "_raw_pred_cache", None)
                        score_sensitivity = _score_ablation_sensitivity(
                            primary_scores=primary_scores,
                            ablation_scores=shuffled_prior_scores if isinstance(shuffled_prior_scores, torch.Tensor) else None,
                            primary_mask=frozen_primary_masks.get("MLQDS"),
                            ablation_mask=shuffled_prior_mask,
                        )
                        raw_sensitivity = _score_ablation_sensitivity(
                            primary_scores=primary_raw_preds,
                            ablation_scores=(
                                shuffled_prior_raw_preds if isinstance(shuffled_prior_raw_preds, torch.Tensor) else None
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
                    except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
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
                                "query_prior_field_metadata": query_prior_field_metadata(zero_prior_field),
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
                            ablation_scores=zero_prior_scores if isinstance(zero_prior_scores, torch.Tensor) else None,
                            primary_mask=frozen_primary_masks.get("MLQDS"),
                            ablation_mask=zero_prior_mask,
                        )
                        raw_sensitivity = _score_ablation_sensitivity(
                            primary_scores=primary_raw_preds,
                            ablation_scores=zero_prior_raw_preds if isinstance(zero_prior_raw_preds, torch.Tensor) else None,
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
                    except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
                        causal_ablation_freeze_failures["MLQDS_without_query_prior_features"] = str(exc)
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
                                    "query_prior_field_metadata": query_prior_field_metadata(channel_prior_field),
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
                                    ablation_scores=channel_scores if isinstance(channel_scores, torch.Tensor) else None,
                                    primary_mask=frozen_primary_masks.get("MLQDS"),
                                    ablation_mask=channel_mask,
                                ),
                                "raw_prediction": _score_ablation_sensitivity(
                                    primary_scores=primary_raw_preds,
                                    ablation_scores=(
                                        channel_raw_preds if isinstance(channel_raw_preds, torch.Tensor) else None
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
            with _phase("freeze-audit-retained-masks"):
                for ratio in audit_ratios:
                    if abs(float(ratio) - float(config.model.compression_ratio)) <= 1e-9:
                        continue
                    ratio_key = f"{float(ratio):.4f}"
                    frozen_ratio_methods: list[Method] = []
                    for method in retention_methods:
                        with _phase(f"  freeze audit {method.name} ratio={ratio:.4f}"):
                            freeze_t0 = time.perf_counter()
                            retained_mask = method.simplify(
                                test_points,
                                test_boundaries,
                                float(ratio),
                            ).detach().cpu()
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

    matched: dict[str, MethodEvaluation] = {}
    oracle_method: OracleMethod | None = None
    eval_labels: torch.Tensor | None = None
    segment_oracle_allocation_audit: dict[str, Any] = {"available": False, "reason": "not_run"}
    target_segment_oracle_alignment_audit: dict[str, Any] = {"available": False, "reason": "not_run"}
    save_masks = bool(save_simplified_dir)
    eval_is_range_only = len(range_only_queries(eval_workload.typed_queries)) == len(eval_workload.typed_queries)
    final_metrics_mode = str(getattr(config.baselines, "final_metrics_mode", "diagnostic")).lower()
    if final_metrics_mode not in {"diagnostic", "core"}:
        raise ValueError("final_metrics_mode must be either 'diagnostic' or 'core'.")
    run_final_diagnostics = final_metrics_mode == "diagnostic"
    run_oracle_baseline = bool(config.baselines.include_oracle and run_final_diagnostics)
    run_learned_fill_diagnostics = bool(eval_is_range_only and run_final_diagnostics)
    with _phase("eval-query-cache-prep"):
        eval_query_cache = prepare_eval_query_cache(
            test_points=test_points,
            test_boundaries=test_boundaries,
            eval_workload=eval_workload,
            eval_is_range_only=eval_is_range_only,
            runtime_cache=range_runtime_caches["eval"],
        )
    if run_oracle_baseline or run_learned_fill_diagnostics or mlqds_range_geometry_blend > 0.0:
        with _phase("eval-label-prep"):
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
            raise RuntimeError("MLQDS range geometry blend requested but eval labels were not prepared.")
        attach_range_geometry_scores(
            methods=methods,
            eval_labels=eval_labels,
            eval_workload_map=eval_workload_map,
        )
    if workload_blind_eval and str(getattr(config.model, "selector_type", "")).lower() == "learned_segment_budget_v1":
        segment_oracle_allocation_audit = _segment_oracle_allocation_audit(
            point_scores=frozen_primary_scores.get("MLQDS"),
            segment_budget_scores=frozen_primary_segment_scores.get("MLQDS"),
            selector_segment_scores=frozen_primary_selector_segment_scores.get("MLQDS"),
            eval_labels=eval_labels,
            boundaries=test_boundaries,
            workload_type=single_workload_type(eval_workload_map),
            head_scores_by_name=_factorized_head_probability_sources_from_logits(
                frozen_primary_head_logits.get("MLQDS")
            ),
            retained_mask=frozen_primary_masks.get("MLQDS"),
        )
        try:
            target_segment_oracle_alignment_audit = _target_segment_oracle_alignment_audit(
                points=test_points,
                boundaries=test_boundaries,
                typed_queries=eval_workload.typed_queries,
                eval_labels=eval_labels,
                workload_type=single_workload_type(eval_workload_map),
                retained_mask=frozen_primary_masks.get("MLQDS"),
            )
        except Exception as exc:  # pragma: no cover - diagnostic should not break final eval.
            target_segment_oracle_alignment_audit = {
                "available": False,
                "reason": "target_alignment_failed",
                "diagnostic_only": True,
                "error": str(exc),
            }
    with _phase("evaluate-matched"):
        for method in methods:
            with _phase(f"  eval {method.name}"):
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
            oracle_method = OracleMethod(labels=eval_labels, workload_type=single_workload_type(eval_workload_map))
            with _phase(f"  eval {oracle_method.name}"):
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
        with _phase("learning-causality-ablations"):
            for method in causality_ablation_methods:
                causality_ablation_mask_diagnostics[method.name] = _retained_mask_comparison(
                    primary_mask=primary_ablation_mask,
                    ablation_mask=method.retained_mask,
                    expected_shape=(
                        primary_ablation_mask.shape
                        if isinstance(primary_ablation_mask, torch.Tensor)
                        else method.retained_mask.shape
                    ),
                )
                with _phase(f"  ablation {method.name}"):
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
            raise RuntimeError("Learned-fill diagnostics requested but eval labels were not prepared.")
        diagnostic_methods = build_learned_fill_methods(
            test_points=test_points,
            eval_labels=eval_labels,
            eval_workload_map=eval_workload_map,
            config=config,
            seeds=seeds,
        )
        with _phase("learned-fill-diagnostics"):
            for method in diagnostic_methods:
                with _phase(f"  fill {method.name}"):
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
        audit_methods = [*(retention_methods if workload_blind_eval else methods), *diagnostic_methods]
        if oracle_method is not None:
            audit_methods.append(oracle_method)
        audit_sections: list[str] = []
        with _phase("range-compression-audit"):
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
                        ratio_audit_methods = [*frozen_audit_methods_by_ratio[ratio_key], *diagnostic_methods]
                        if oracle_method is not None:
                            ratio_audit_methods.append(oracle_method)
                    for method in ratio_audit_methods:
                        with _phase(f"  audit {method.name} ratio={ratio:.4f}"):
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
                    name: _evaluation_metrics_payload(metrics) for name, metrics in ratio_results.items()
                }
                audit_sections.append(f"compression_ratio={ratio_key}\n{print_range_usefulness_table(ratio_results)}")
        range_compression_audit_table = "\n\n".join(audit_sections)

    with _phase("evaluate-shift"):
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

    with _phase("range-diagnostics"):
        train_summary, train_rows = _range_workload_diagnostics(
            "train",
            train_points,
            train_boundaries,
            train_workload,
            train_workload_map,
            config,
            seeds.train_query_seed,
            range_runtime_caches["train"],
        )
        eval_summary, eval_rows = _range_workload_diagnostics(
            "eval",
            test_points,
            test_boundaries,
            eval_workload,
            eval_workload_map,
            config,
            seeds.eval_query_seed,
            range_runtime_caches["eval"],
        )
        range_diagnostics_summary["train"] = train_summary
        range_diagnostics_summary["eval"] = eval_summary
        range_diagnostics_rows.extend(train_rows)
        range_diagnostics_rows.extend(eval_rows)
        for replicate_index, replicate_workload in enumerate(train_label_workloads[1:], start=1):
            replicate_label = f"train_r{replicate_index}"
            replicate_summary, replicate_rows = _range_workload_diagnostics(
                replicate_label,
                train_points,
                train_boundaries,
                replicate_workload,
                train_workload_map,
                config,
                train_label_workload_seeds[replicate_index],
                RangeRuntimeCache(),
            )
            range_diagnostics_summary[replicate_label] = replicate_summary
            range_diagnostics_rows.extend(replicate_rows)
        if selection_workload is not None and selection_points is not None and selection_boundaries is not None:
            selection_summary, selection_rows = _range_workload_diagnostics(
                "selection",
                selection_points,
                selection_boundaries,
                selection_workload,
                eval_workload_map,
                config,
                seeds.eval_query_seed + 17,
                range_runtime_caches["selection"],
            )
            range_diagnostics_summary["selection"] = selection_summary
            range_diagnostics_rows.extend(selection_rows)
        _print_range_diagnostics_summary(range_diagnostics_summary)
        workload_distribution_comparison = _range_workload_distribution_comparison(range_diagnostics_summary)
        _print_range_distribution_comparison(workload_distribution_comparison)

    range_learned_fill_summary = _range_learned_fill_summary(
        learned_fill_diagnostics=learned_fill_diagnostics,
        training_target_diagnostics=trained.target_diagnostics,
        range_diagnostics_summary=range_diagnostics_summary,
        compression_ratio=float(config.model.compression_ratio),
    )
    predictability_audit = query_prior_predictability_audit(
        points=test_points,
        boundaries=test_boundaries,
        eval_typed_queries=eval_workload.typed_queries,
        query_prior_field=trained.feature_context.get("query_prior_field"),
    )
    uniform_eval = matched.get("uniform")
    douglas_peucker_eval = matched.get("DouglasPeucker")
    workload_signature_gate = workload_distribution_comparison.get("workload_signature_gate", {})
    predictability_gate_pass = bool(predictability_audit.get("gate_pass", False))
    prior_predictive_alignment_gate = predictability_audit.get("prior_predictive_alignment_gate", {})
    prior_predictive_alignment_gate_pass = bool(
        isinstance(prior_predictive_alignment_gate, dict) and prior_predictive_alignment_gate.get("gate_pass", False)
    )
    signature_gate_pass = bool(
        isinstance(workload_signature_gate, dict)
        and workload_signature_gate.get("all_available")
        and workload_signature_gate.get("all_pass")
    )
    workload_stability_gate = _workload_stability_gate(
        config=config,
        train_label_workloads=train_label_workloads,
        eval_workload=eval_workload,
        selection_workload=selection_workload,
    )
    workload_stability_gate_pass = bool(workload_stability_gate.get("gate_pass", False))
    support_overlap_gate = _support_overlap_gate(
        train_points=train_points,
        eval_points=test_points,
        query_prior_field=trained.feature_context.get("query_prior_field"),
    )
    support_overlap_gate_pass = bool(support_overlap_gate.get("gate_pass", False))
    target_diffusion_gate = _target_diffusion_gate(trained.target_diagnostics)
    target_diffusion_gate_pass = bool(target_diffusion_gate.get("gate_pass", False))
    final_candidate = (
        str(config.query.workload_profile_id or "").lower() == "range_workload_v1"
        and str(config.model.model_type).lower() == "workload_blind_range_v2"
        and str(config.model.range_training_target_mode).lower() == "query_useful_v1_factorized"
        and str(getattr(config.model, "selector_type", "")).lower() == "learned_segment_budget_v1"
    )
    legacy_range_useful_summary = {
        "metric": "RangeUsefulLegacy",
        "schema": "range_usefulness_schema_version",
        "diagnostic_only": True,
        "mlqds_score": matched["MLQDS"].range_usefulness_score,
        "uniform_score": uniform_eval.range_usefulness_score if uniform_eval is not None else None,
        "douglas_peucker_score": (
            douglas_peucker_eval.range_usefulness_score
            if douglas_peucker_eval is not None
            else None
        ),
    }
    learned_slot_summary = _learned_slot_summary(
        selector_budget_diagnostics,
        float(config.model.compression_ratio),
        primary_selector_trace,
    )
    primary_eval = matched["MLQDS"]
    shuffled_delta = _query_useful_delta(primary_eval, causality_ablation_evaluations, "MLQDS_shuffled_scores")
    prior_only_delta = _query_useful_delta(
        primary_eval,
        causality_ablation_evaluations,
        "MLQDS_prior_field_only_score",
    )
    untrained_delta = _query_useful_delta(primary_eval, causality_ablation_evaluations, "MLQDS_untrained_model")
    shuffled_prior_delta = _query_useful_delta(
        primary_eval,
        causality_ablation_evaluations,
        "MLQDS_shuffled_prior_fields",
    )
    no_query_prior_delta = _query_useful_delta(
        primary_eval,
        causality_ablation_evaluations,
        "MLQDS_without_query_prior_features",
    )
    no_behavior_head_delta = _query_useful_delta(
        primary_eval,
        causality_ablation_evaluations,
        "MLQDS_without_behavior_utility_head",
    )
    no_segment_budget_head_delta = _query_useful_delta(
        primary_eval,
        causality_ablation_evaluations,
        "MLQDS_without_segment_budget_head",
    )
    no_fairness_preallocation_delta = _query_useful_delta(
        primary_eval,
        causality_ablation_evaluations,
        "MLQDS_without_trajectory_fairness_preallocation",
    )
    no_geometry_tie_breaker_delta = _query_useful_delta(
        primary_eval,
        causality_ablation_evaluations,
        "MLQDS_without_geometry_tie_breaker",
    )
    causality_ablation_component_deltas = _query_useful_component_delta_summary(
        primary=primary_eval,
        ablations=causality_ablation_evaluations,
    )
    causality_ablation_tradeoff_diagnostics = _causality_ablation_tradeoff_summary(
        component_deltas=causality_ablation_component_deltas,
        mask_diagnostics=causality_ablation_mask_diagnostics,
    )
    for name, tradeoff_diagnostics in causality_ablation_tradeoff_diagnostics.items():
        if name in head_ablation_sensitivity_diagnostics:
            head_ablation_sensitivity_diagnostics[name]["query_useful_component_tradeoff"] = tradeoff_diagnostics
    for prior_channel_name, channel_diagnostics in prior_channel_ablation_diagnostics.items():
        if not isinstance(channel_diagnostics, dict) or not bool(channel_diagnostics.get("available", False)):
            continue
        method_name = str(channel_diagnostics.get("method_name", ""))
        channel_eval = causality_ablation_evaluations.get(method_name)
        if channel_eval is not None:
            channel_diagnostics["query_useful_v1_score"] = float(channel_eval.query_useful_v1_score)
            channel_diagnostics["query_useful_v1_delta"] = _query_useful_delta(
                primary_eval,
                causality_ablation_evaluations,
                method_name,
            )
        if method_name in causality_ablation_mask_diagnostics:
            channel_diagnostics["retained_mask"] = causality_ablation_mask_diagnostics[method_name]
        if method_name in causality_ablation_component_deltas:
            channel_diagnostics["query_useful_component_deltas"] = causality_ablation_component_deltas[method_name]
        if method_name in causality_ablation_tradeoff_diagnostics:
            channel_diagnostics["query_useful_component_tradeoff"] = causality_ablation_tradeoff_diagnostics[method_name]
    required_causality_ablation_names = (
        "MLQDS_shuffled_scores",
        "MLQDS_untrained_model",
        "MLQDS_shuffled_prior_fields",
        "MLQDS_without_query_prior_features",
        "MLQDS_without_behavior_utility_head",
        "MLQDS_without_segment_budget_head",
    )
    missing_causality_ablations = [
        name for name in required_causality_ablation_names if name not in causality_ablation_evaluations
    ]
    failed_causality_checks: list[str] = []
    delta_checks = {
        "shuffled_scores_should_lose": shuffled_delta,
        "untrained_model_should_lose": untrained_delta,
        "shuffled_prior_fields_should_lose": shuffled_prior_delta,
        "without_query_prior_features_should_lose": no_query_prior_delta,
        "without_behavior_utility_head_should_lose": no_behavior_head_delta,
        "without_segment_budget_head_should_lose": no_segment_budget_head_delta,
        "prior_field_only_should_not_match_trained": prior_only_delta,
    }
    delta_gate_config = _learning_causality_delta_gate_config(
        primary=primary_eval,
        uniform=uniform_eval,
    )
    delta_thresholds = delta_gate_config.get("thresholds", {})
    for check_name, delta in delta_checks.items():
        threshold = float(delta_thresholds.get(check_name, LEARNING_CAUSALITY_MIN_MATERIAL_DELTA))
        if delta is not None and float(delta) + 1e-12 < threshold:
            failed_causality_checks.append(check_name)
    prior_sample_failures = _prior_sample_gate_failures(prior_sensitivity_diagnostics)
    failed_causality_checks.extend(prior_sample_failures)
    learned_slot_fraction = float(learned_slot_summary.get("learned_controlled_retained_slot_fraction") or 0.0)
    learned_slot_fraction_min = 0.0
    if float(config.model.compression_ratio) >= 0.10:
        learned_slot_fraction_min = 0.35
    elif float(config.model.compression_ratio) >= 0.05:
        learned_slot_fraction_min = 0.25
    if learned_slot_fraction_min > 0.0 and learned_slot_fraction < learned_slot_fraction_min:
        failed_causality_checks.append("learned_controlled_slot_fraction_below_minimum")
    ablation_status = "not_run"
    if causality_ablation_evaluations or causal_ablation_freeze_failures:
        ablation_status = "complete" if not missing_causality_ablations and not causal_ablation_freeze_failures else "partial"
    learning_causality_gate_pass = (
        ablation_status == "complete" and not failed_causality_checks and not missing_causality_ablations
    )
    learning_causality_summary = {
        "selector_diagnostics_present": bool(selector_budget_diagnostics),
        "training_fit_diagnostics_present": bool(trained.fit_diagnostics),
        "selector_type": str(getattr(config.model, "selector_type", "temporal_hybrid")),
        "selector_final_candidate": str(getattr(config.model, "selector_type", "temporal_hybrid"))
        == "learned_segment_budget_v1",
        "query_prior_field_available": bool(trained.feature_context.get("query_prior_field")),
        **learned_slot_summary,
        "shuffled_score_ablation_delta": shuffled_delta,
        "untrained_score_ablation_delta": untrained_delta,
        "shuffled_prior_field_ablation_delta": shuffled_prior_delta,
        "no_query_prior_field_ablation_delta": no_query_prior_delta,
        "no_behavior_head_ablation_delta": no_behavior_head_delta,
        "no_segment_budget_head_ablation_delta": no_segment_budget_head_delta,
        "no_trajectory_fairness_preallocation_ablation_delta": no_fairness_preallocation_delta,
        "no_geometry_tie_breaker_ablation_delta": no_geometry_tie_breaker_delta,
        "segment_budget_head_ablation_mode": segment_budget_head_ablation_mode,
        "learned_segment_selector_config": {
            "geometry_gain_weight": float(config.model.learned_segment_geometry_gain_weight),
            "segment_score_blend_weight": float(config.model.learned_segment_score_blend_weight),
            "fairness_preallocation_enabled": bool(config.model.learned_segment_fairness_preallocation),
            "length_repair_fraction": float(config.model.learned_segment_length_repair_fraction),
            "length_support_blend_weight": float(config.model.learned_segment_length_support_blend_weight),
        },
        "prior_field_only_score_ablation_delta": prior_only_delta,
        "without_query_prior_features_delta": no_query_prior_delta,
        "learning_causality_delta_gate": delta_gate_config,
        "prior_sensitivity_diagnostics": prior_sensitivity_diagnostics,
        "prior_channel_ablation_diagnostics": prior_channel_ablation_diagnostics,
        "head_ablation_sensitivity_diagnostics": head_ablation_sensitivity_diagnostics,
        "selection_causality_diagnostics": selection_causality_diagnostics,
        "segment_oracle_allocation_audit": segment_oracle_allocation_audit,
        "target_segment_oracle_alignment_audit": target_segment_oracle_alignment_audit,
        "score_protected_length_feasibility": (
            primary_selector_trace.get("score_protected_length_feasibility")
            if isinstance(primary_selector_trace, dict)
            else None
        ),
        "score_protected_length_frontier": (
            primary_selector_trace.get("score_protected_length_frontier")
            if isinstance(primary_selector_trace, dict)
            else None
        ),
        "prior_sample_gate_pass": not prior_sample_failures,
        "prior_sample_gate_failures": prior_sample_failures,
        "causality_ablation_scores": {
            name: metrics.query_useful_v1_score for name, metrics in causality_ablation_evaluations.items()
        },
        "causality_ablation_component_deltas": causality_ablation_component_deltas,
        "causality_ablation_mask_diagnostics": causality_ablation_mask_diagnostics,
        "causality_ablation_tradeoff_diagnostics": causality_ablation_tradeoff_diagnostics,
        "causality_ablation_freeze_failures": causal_ablation_freeze_failures,
        "causality_ablation_missing": missing_causality_ablations,
        "learning_causality_gate_pass": learning_causality_gate_pass,
        "learning_causality_failed_checks": failed_causality_checks,
        "learned_controlled_slot_fraction_min": learned_slot_fraction_min,
        "learning_causality_ablation_status": ablation_status,
        "predictability_gate_pass": predictability_gate_pass,
        "prior_predictive_alignment_gate_pass": prior_predictive_alignment_gate_pass,
        "workload_signature_gate_pass": signature_gate_pass,
        "support_overlap_gate_pass": support_overlap_gate_pass,
    }
    global_sanity_gate = _global_sanity_gate(
        primary=matched["MLQDS"],
        uniform=uniform_eval,
        compression_ratio=float(config.model.compression_ratio),
    )
    global_sanity_gate_pass = bool(global_sanity_gate.get("gate_pass", False))
    blocking_gates: list[str] = []
    if final_candidate:
        if not workload_stability_gate_pass:
            blocking_gates.append("workload_stability_gate")
        if not support_overlap_gate_pass:
            blocking_gates.append("support_overlap_gate")
        if not predictability_gate_pass:
            blocking_gates.append("predictability_gate")
        if not prior_predictive_alignment_gate_pass:
            blocking_gates.append("prior_predictive_alignment_gate")
        if not target_diffusion_gate_pass:
            blocking_gates.append("target_diffusion_gate")
        if not signature_gate_pass:
            blocking_gates.append("workload_signature_gate")
        if not learning_causality_gate_pass:
            blocking_gates.append("learning_causality_ablations")
        if not global_sanity_gate_pass:
            blocking_gates.append("global_sanity_gates")
        single_cell_blocking_gates = list(blocking_gates)
        blocking_gates.append("full_coverage_compression_grid")
        if single_cell_blocking_gates:
            final_claim_reason = (
                "Strict single-cell evidence is blocked by required gates before the final grid: "
                + ", ".join(single_cell_blocking_gates)
                + "."
            )
        else:
            final_claim_reason = (
                "Strict single-cell gates passed; final success still requires the benchmark-level "
                "full coverage/compression grid."
            )
        final_claim_summary = {
            "primary_metric": "QueryUsefulV1",
            "status": "candidate_blocked_by_required_gates" if blocking_gates else "candidate_ready_for_final_claim",
            "final_success_allowed": not blocking_gates,
            "blocking_gates": blocking_gates,
            "workload_stability_gate_pass": workload_stability_gate_pass,
            "support_overlap_gate_pass": support_overlap_gate_pass,
            "predictability_gate_pass": predictability_gate_pass,
            "prior_predictive_alignment_gate_pass": prior_predictive_alignment_gate_pass,
            "target_diffusion_gate_pass": target_diffusion_gate_pass,
            "workload_signature_gate_pass": signature_gate_pass,
            "learning_causality_gate_pass": learning_causality_gate_pass,
            "global_sanity_gate_pass": global_sanity_gate_pass,
            "mlqds_score": matched["MLQDS"].query_useful_v1_score,
            "uniform_score": uniform_eval.query_useful_v1_score if uniform_eval is not None else None,
            "douglas_peucker_score": (
                douglas_peucker_eval.query_useful_v1_score
                if douglas_peucker_eval is not None
                else None
            ),
            "reason": final_claim_reason,
        }
    else:
        final_claim_summary = {
            "primary_metric": None,
            "status": "not_final_query_driven_candidate",
            "final_success_allowed": False,
            "reason": "Requires range_workload_v1, QueryUsefulV1 factorized target, workload_blind_range_v2, and learned_segment_budget_v1.",
        }
    learning_causality_summary["final_success_allowed"] = bool(final_candidate and not blocking_gates)

    dump = {
        "config": config.to_dict(),
        "final_claim_summary": final_claim_summary,
        "diagnostic_summary": {
            "legacy_range_useful_available": True,
            "query_useful_v1_available": True,
            "range_component_diagnostics_available": True,
            "workload_blind_protocol_available": True,
            "predictability_audit_available": bool(predictability_audit.get("available", False)),
            "prior_predictive_alignment_gate_available": isinstance(prior_predictive_alignment_gate, dict),
            "workload_stability_gate_available": bool(workload_stability_gate),
            "support_overlap_gate_available": bool(support_overlap_gate),
            "global_sanity_gate_available": bool(global_sanity_gate),
            "target_diffusion_gate_available": bool(target_diffusion_gate),
            "workload_signature_gate_available": bool(
                isinstance(workload_signature_gate, dict) and workload_signature_gate.get("all_available")
            ),
        },
        "legacy_range_useful_summary": legacy_range_useful_summary,
        "learning_causality_summary": learning_causality_summary,
        "support_overlap_gate": support_overlap_gate,
        "global_sanity_gate": global_sanity_gate,
        "target_diffusion_gate": target_diffusion_gate,
        "workload": single_workload_type(eval_workload_map),
        "train_query_count": len(train_workload.typed_queries),
        "train_label_workload_count": len(train_label_workloads),
        "train_label_workload_query_counts": [len(workload.typed_queries) for workload in train_label_workloads],
        "eval_query_count": len(eval_workload.typed_queries),
        "selection_query_count": len(selection_workload.typed_queries) if selection_workload is not None else None,
        "train_query_coverage": train_workload.coverage_fraction,
        "train_label_workload_coverages": [workload.coverage_fraction for workload in train_label_workloads],
        "eval_query_coverage": eval_workload.coverage_fraction,
        "selection_query_coverage": selection_workload.coverage_fraction if selection_workload is not None else None,
        "query_generation_diagnostics": {
            "train": train_workload.generation_diagnostics,
            "train_label_workloads": [workload.generation_diagnostics for workload in train_label_workloads],
            "eval": eval_workload.generation_diagnostics,
            "selection": selection_workload.generation_diagnostics if selection_workload is not None else None,
        },
        "data_split_diagnostics": data_split.split_diagnostics,
        "selector_budget_diagnostics": selector_budget_diagnostics,
        "selector_trace_diagnostics": {
            "eval_primary": primary_selector_trace if primary_selector_trace is not None else {"available": False}
        },
        "segment_oracle_allocation_audit": segment_oracle_allocation_audit,
        "target_segment_oracle_alignment_audit": target_segment_oracle_alignment_audit,
        "matched": {name: _evaluation_metrics_payload(m) for name, m in matched.items()},
        "learning_causality_ablations": {
            name: _evaluation_metrics_payload(metrics)
            for name, metrics in causality_ablation_evaluations.items()
        },
        "learned_fill_diagnostics": {
            name: _evaluation_metrics_payload(metrics) for name, metrics in learned_fill_diagnostics.items()
        },
        "range_learned_fill_summary": range_learned_fill_summary,
        "predictability_audit": predictability_audit,
        "workload_stability_gate": workload_stability_gate,
        "range_compression_audit": range_compression_audit,
        "shift": shift_pairs,
        "training_history": trained.history,
        "training_target_diagnostics": trained.target_diagnostics,
        "training_fit_diagnostics": trained.fit_diagnostics,
        "range_training_target_transform": range_training_target_transform,
        "model_metadata": model_type_metadata(config.model.model_type),
        "query_prior_field": trained.feature_context.get("query_prior_field_metadata", {"available": False}),
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
            "frozen_audit_ratio_count": int(len(frozen_audit_methods_by_ratio)),
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

    with _phase("write-results"):
        out_dir = write_experiment_results(
            results_dir=results_dir,
            matched_table=matched_table,
            shift_table=shift_table,
            geometric_table=geometric_table,
            range_usefulness_table=range_usefulness_table,
            learned_fill_table=learned_fill_table,
            learned_fill_diagnostics=learned_fill_diagnostics,
            range_learned_fill_summary=range_learned_fill_summary,
            range_compression_audit=range_compression_audit,
            range_compression_audit_table=range_compression_audit_table,
            range_diagnostics_summary=range_diagnostics_summary,
            workload_distribution_comparison=workload_distribution_comparison,
            range_diagnostics_rows=range_diagnostics_rows,
            dump=dump,
        )
        print(f"  wrote results to {out_dir}", flush=True)

    if save_simplified_dir:
        with _phase("write-simplified-csv"):
            out_dir = Path(save_simplified_dir)
            eval_mask = matched["MLQDS"].retained_mask
            if eval_mask is None:
                eval_mlqds = MLQDSMethod(
                    name="MLQDS",
                    trained=trained,
                    workload=eval_workload,
                    workload_type=single_workload_type(eval_workload_map),
                    score_mode=config.model.mlqds_score_mode,
                    score_temperature=config.model.mlqds_score_temperature,
                    rank_confidence_weight=config.model.mlqds_rank_confidence_weight,
                    temporal_fraction=config.model.mlqds_temporal_fraction,
                    diversity_bonus=config.model.mlqds_diversity_bonus,
                    hybrid_mode=config.model.mlqds_hybrid_mode,
                    selector_type=config.model.selector_type,
                    learned_segment_geometry_gain_weight=config.model.learned_segment_geometry_gain_weight,
                    learned_segment_score_blend_weight=config.model.learned_segment_score_blend_weight,
                    learned_segment_fairness_preallocation=config.model.learned_segment_fairness_preallocation,
                    learned_segment_length_repair_fraction=config.model.learned_segment_length_repair_fraction,
                    learned_segment_length_support_blend_weight=(
                        config.model.learned_segment_length_support_blend_weight
                    ),
                    stratified_center_weight=config.model.mlqds_stratified_center_weight,
                    min_learned_swaps=config.model.mlqds_min_learned_swaps,
                    trajectory_mmsis=test_mmsis,
                    inference_batch_size=config.model.inference_batch_size,
                    amp_mode=config.model.amp_mode,
                )
                eval_mask = eval_mlqds.simplify(test_points, test_boundaries, config.model.compression_ratio)
            write_simplified_csv(
                str(out_dir / "ML_simplified_eval.csv"),
                test_points,
                test_boundaries,
                eval_mask,
                trajectory_mmsis=test_mmsis,
            )
            for ref_name, csv_name in (("uniform", "uniform_simplified_eval.csv"),
                                       ("DouglasPeucker", "DP_simplified_eval.csv")):
                ref_eval = matched.get(ref_name)
                ref_mask = ref_eval.retained_mask if ref_eval is not None else None
                if ref_mask is not None:
                    write_simplified_csv(
                        str(out_dir / csv_name),
                        test_points,
                        test_boundaries,
                        ref_mask,
                        trajectory_mmsis=test_mmsis,
                    )

        with _phase("trajectory-length-loss"):
            report_trajectory_length_loss(
                test_points,
                test_boundaries,
                eval_mask,
                top_k=25,
                trajectory_mmsis=test_mmsis,
            )

    print(f"[pipeline] total runtime {time.perf_counter() - pipeline_t0:.2f}s", flush=True)
    return ExperimentOutputs(
        matched_table=matched_table,
        shift_table=shift_table,
        metrics_dump=dump,
        geometric_table=geometric_table,
        range_usefulness_table=range_usefulness_table,
        range_compression_audit_table=range_compression_audit_table,
    )
