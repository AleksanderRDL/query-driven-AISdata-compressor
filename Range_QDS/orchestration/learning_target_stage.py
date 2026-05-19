"""Training-target preparation for single-run orchestration."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch

from config.run_config import RunConfig, SeedBundle
from learning.model_features import is_workload_blind_model_type
from learning.model_training import train_model
from learning.targets.aggregation import (
    aggregate_range_component_label_sets,
    aggregate_range_component_retained_frequency_training_labels,
    aggregate_range_continuity_retained_frequency_training_labels,
    aggregate_range_global_budget_retained_frequency_training_labels,
    aggregate_range_marginal_coverage_training_labels,
    aggregate_range_retained_frequency_training_labels,
    aggregate_range_structural_retained_frequency_training_labels,
    range_component_retained_frequency_training_labels,
    range_continuity_retained_frequency_training_labels,
)
from learning.targets.common import (
    aggregate_range_label_sets,
    balance_range_training_target_by_trajectory,
)
from learning.targets.local_swap import (
    range_local_swap_gain_cost_frequency_training_labels,
    range_local_swap_utility_frequency_training_labels,
)
from learning.targets.marginal_coverage import range_marginal_coverage_training_labels
from learning.targets.query_residual import range_query_residual_frequency_training_labels
from learning.targets.query_spine import range_query_spine_frequency_training_labels
from learning.targets.query_useful_v1 import (
    QUERY_USEFUL_V1_FACTORIZED_TARGET_MODE,
    QUERY_USEFUL_V1_TARGET_MODES,
)
from learning.targets.retained_frequency import (
    range_global_budget_retained_frequency_training_labels,
    range_historical_prior_retained_frequency_training_labels,
    range_retained_frequency_training_labels,
)
from learning.targets.set_utility import range_set_utility_frequency_training_labels
from learning.targets.structural import range_structural_retained_frequency_training_labels
from learning.teacher_distillation import (
    build_range_teacher_config,
    distill_range_teacher_labels,
    range_teacher_distillation_enabled,
)
from orchestration.range_runtime_cache import (
    RangeRuntimeCache,
    prepare_range_label_cache,
    range_only_queries,
)
from scoring.query_cache import ScoringQueryCache
from selection.model_score_conversion import workload_type_head
from workloads.query_types import QUERY_TYPE_ID_RANGE, single_workload_type
from workloads.typed_workload import TypedQueryWorkload

PhaseLogger = Callable[[str], AbstractContextManager[None]]
RangeLabels = tuple[torch.Tensor, torch.Tensor]


@dataclass
class TargetPreparationOutputs:
    """Prepared training labels and validation caches consumed by model learning."""

    train_labels: RangeLabels | None
    range_training_target_mode: str
    range_training_target_transform: dict[str, Any]
    range_target_balance_diagnostics: dict[str, Any]
    range_training_label_aggregation: dict[str, Any]
    teacher_distillation_diagnostics: dict[str, Any]
    selection_query_cache: ScoringQueryCache | None
    selection_geometry_scores: torch.Tensor | None


def prepare_training_targets(
    *,
    config: RunConfig,
    seeds: SeedBundle,
    train_traj: list[torch.Tensor],
    train_points: torch.Tensor,
    train_boundaries: list[tuple[int, int]],
    train_workload: TypedQueryWorkload,
    train_workload_map: dict[str, float],
    train_label_workloads: list[TypedQueryWorkload],
    train_label_workload_seeds: list[int],
    train_source_ids: list[int] | None,
    train_mmsis: list[int] | None,
    selection_workload: TypedQueryWorkload | None,
    selection_points: torch.Tensor | None,
    selection_boundaries: list[tuple[int, int]] | None,
    eval_workload_map: dict[str, float],
    range_runtime_caches: dict[str, RangeRuntimeCache],
    phase: PhaseLogger,
) -> TargetPreparationOutputs:
    """Prepare training labels, target transforms, and validation query caches."""
    train_labels: RangeLabels | None = None
    range_training_target_mode = str(
        getattr(config.model, "range_training_target_mode", "point_value")
    ).lower()
    range_replicate_target_aggregation = str(
        getattr(config.model, "range_replicate_target_aggregation", "label_mean")
    ).lower()
    if range_replicate_target_aggregation not in {"label_mean", "label_max", "frequency_mean"}:
        raise ValueError(
            "range_replicate_target_aggregation must be 'label_mean', 'label_max', or 'frequency_mean'."
        )
    if len(train_label_workloads) > 1 and not is_workload_blind_model_type(config.model.model_type):
        raise RuntimeError(
            "range_train_workload_replicates > 1 is only valid for workload-blind model types."
        )
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
        "replicate_count": len(train_label_workloads),
        "seeds": [int(seed) for seed in train_label_workload_seeds],
    }
    teacher_distillation_diagnostics: dict[str, Any] = {
        "enabled": False,
        "mode": str(getattr(config.model, "range_teacher_distillation_mode", "none")),
    }
    if range_training_target_mode in QUERY_USEFUL_V1_TARGET_MODES:
        range_training_target_transform.update(
            {
                "enabled": True,
                "target_family": "QueryUsefulV1Factorized",
                "final_success_allowed": (
                    range_training_target_mode == QUERY_USEFUL_V1_FACTORIZED_TARGET_MODE
                ),
            }
        )
        if range_training_target_mode != QUERY_USEFUL_V1_FACTORIZED_TARGET_MODE:
            range_training_target_transform["diagnostic_reason"] = (
                "Experimental QueryUsefulV1 target mode. "
                "Not valid for final-candidate acceptance until promoted."
            )
    selection_query_cache: ScoringQueryCache | None = None
    selection_geometry_scores: torch.Tensor | None = None
    mlqds_range_geometry_blend = max(
        0.0, min(1.0, float(getattr(config.model, "mlqds_range_geometry_blend", 0.0)))
    )
    with phase("range-training-prep"):
        train_label_sets: list[RangeLabels] = []
        train_component_label_sets: list[dict[str, torch.Tensor] | None] = []
        if (
            range_training_target_mode not in QUERY_USEFUL_V1_TARGET_MODES
            or range_teacher_distillation_enabled(config.model)
        ):
            for replicate_index, label_workload in enumerate(train_label_workloads):
                label_cache_name = "train" if replicate_index == 0 else f"train_r{replicate_index}"
                runtime_cache = (
                    range_runtime_caches["train"] if replicate_index == 0 else RangeRuntimeCache()
                )
                label_result = prepare_range_label_cache(
                    cache_label=label_cache_name,
                    points=train_points,
                    boundaries=train_boundaries,
                    workload=label_workload,
                    workload_map=train_workload_map,
                    config=config,
                    seed=train_label_workload_seeds[replicate_index],
                    runtime_cache=runtime_cache,
                    range_boundary_prior_weight=float(
                        getattr(config.model, "range_boundary_prior_weight", 0.0)
                    ),
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
                    raise ValueError(
                        "range_replicate_target_aggregation='frequency_mean' requires a frequency target."
                    )
                labels, labelled_mask, aggregation_diagnostics = aggregate_range_label_sets(
                    train_label_sets,
                    aggregation="max"
                    if range_replicate_target_aggregation == "label_max"
                    else "mean",
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
            and len(range_only_queries(selection_workload.typed_queries))
            == len(selection_workload.typed_queries)
        ):
            selection_query_cache = ScoringQueryCache.for_workload(
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
                    range_boundary_prior_weight=float(
                        getattr(config.model, "range_boundary_prior_weight", 0.0)
                    ),
                )
                if selection_labels is not None:
                    labels, _labelled_mask = selection_labels
                    _, selection_type_id = workload_type_head(
                        single_workload_type(eval_workload_map)
                    )
                    selection_geometry_scores = labels[:, selection_type_id].float()
    if train_labels is not None and len(range_only_queries(train_workload.typed_queries)) == len(
        train_workload.typed_queries
    ):
        print("  prepared train range labels for precomputed training target", flush=True)
    if selection_query_cache is not None:
        print("  prepared checkpoint-validation range query cache", flush=True)
    if range_teacher_distillation_enabled(config.model):
        if not is_workload_blind_model_type(config.model.model_type):
            raise RuntimeError(
                "range teacher distillation is only valid for workload-blind model types."
            )
        if train_labels is None:
            raise RuntimeError(
                "range teacher distillation requires precomputed range training labels."
            )
        for label_workload in train_label_workloads:
            if len(range_only_queries(label_workload.typed_queries)) != len(
                label_workload.typed_queries
            ):
                raise RuntimeError(
                    "range teacher distillation requires pure range training workloads."
                )
        teacher_config = build_range_teacher_config(config.model)
        print(
            f"  range teacher distillation enabled: mode={config.model.range_teacher_distillation_mode} "
            f"teacher_epochs={teacher_config.epochs} "
            f"replicates={len(train_label_workloads)}",
            flush=True,
        )
        distilled_label_sets: list[RangeLabels] = []
        per_teacher: list[dict[str, Any]] = []
        for replicate_index, label_workload in enumerate(train_label_workloads):
            with phase(f"train-range-teacher-r{replicate_index} ({teacher_config.epochs} epochs)"):
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
            with phase(f"distill-range-teacher-r{replicate_index}-labels"):
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
            teacher_aggregation_mode = (
                "max" if range_replicate_target_aggregation == "label_max" else "mean"
            )
            labels, labelled_mask, aggregation_diagnostics = aggregate_range_label_sets(
                distilled_label_sets,
                source="range_teacher_distillation_replicates",
                aggregation=teacher_aggregation_mode,
            )
            train_labels = (labels, labelled_mask)
            positive = labelled_mask[:, QUERY_TYPE_ID_RANGE] & (
                labels[:, QUERY_TYPE_ID_RANGE] > 0.0
            )
            teacher_distillation_diagnostics = {
                "enabled": True,
                "mode": str(getattr(config.model, "range_teacher_distillation_mode", "none")),
                "teacher_model_type": str(teacher_config.model_type),
                "teacher_epochs": int(teacher_config.epochs),
                "replicate_count": len(distilled_label_sets),
                "replicate_target_aggregation": range_replicate_target_aggregation,
                "aggregation": aggregation_diagnostics,
                "per_replicate": per_teacher,
                "labelled_point_count": int(labelled_mask[:, QUERY_TYPE_ID_RANGE].sum().item()),
                "positive_label_count": int(positive.sum().item()),
                "positive_label_fraction": float(
                    positive.sum().item() / max(1, int(labels.shape[0]))
                ),
                "positive_label_mass": (
                    float(labels[positive, QUERY_TYPE_ID_RANGE].sum().item())
                    if bool(positive.any().item())
                    else 0.0
                ),
                "budget_loss_ratios": list(getattr(config.model, "budget_loss_ratios", [])),
                "mlqds_temporal_fraction": float(
                    getattr(config.model, "mlqds_temporal_fraction", 0.0)
                ),
                "mlqds_hybrid_mode": str(getattr(config.model, "mlqds_hybrid_mode", "fill")),
            }
            range_training_label_aggregation.update(aggregation_diagnostics)
            range_training_label_aggregation["enabled"] = True
            range_training_label_aggregation["target_mode"] = "teacher_distillation"
            range_training_label_aggregation["replicate_target_aggregation"] = (
                range_replicate_target_aggregation
            )
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
            raise RuntimeError(
                f"{range_training_target_mode} target mode requires precomputed range training labels."
            )
        if len(train_label_sets) > 1:
            raise RuntimeError(
                f"{range_training_target_mode} does not yet support multiple train workload replicates."
            )
        target_phase = range_training_target_mode.replace("_", "-")
        with phase(f"range-{target_phase}-target"):
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
        with phase(f"range-{target_phase}-target"):
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
                    range_training_label_aggregation["replicate_target_aggregation"] = (
                        "frequency_mean"
                    )
                else:
                    labels, labelled_mask, aggregation_diagnostics = aggregate_range_label_sets(
                        label_sets=train_label_sets,
                        source=(
                            f"range_label_{'max' if range_replicate_target_aggregation == 'label_max' else 'mean'}"
                            f"_before_{range_training_target_mode}"
                        ),
                        aggregation="max"
                        if range_replicate_target_aggregation == "label_max"
                        else "mean",
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
                    labels, labelled_mask, range_training_target_transform = target_fn(
                        **target_kwargs
                    )
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
    elif range_training_target_mode not in {"point_value", *QUERY_USEFUL_V1_TARGET_MODES}:
        if range_training_target_mode in {
            "component_retained_frequency",
            "continuity_retained_frequency",
        }:
            if train_labels is None:
                raise RuntimeError(
                    f"{range_training_target_mode} target mode requires precomputed range training labels."
                )
            if not train_component_label_sets or any(
                component_labels is None for component_labels in train_component_label_sets
            ):
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
            with phase(f"range-{target_phase}-target"):
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
                        range_training_label_aggregation["replicate_target_aggregation"] = (
                            "frequency_mean"
                        )
                    else:
                        aggregation_mode = (
                            "max" if range_replicate_target_aggregation == "label_max" else "mean"
                        )
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
                        labels, labelled_mask, range_training_target_transform = target_fn(
                            labels=labels,
                            labelled_mask=labelled_mask,
                            component_labels=component_labels,
                            boundaries=train_boundaries,
                            model_config=config.model,
                        )
                        range_training_target_transform["label_aggregation"] = (
                            aggregation_diagnostics
                        )
                    range_training_label_aggregation["enabled"] = True
                    range_training_label_aggregation["target_mode"] = range_training_target_mode
                else:
                    labels, labelled_mask = train_labels
                    component_labels = train_component_label_sets[0]
                    if component_labels is None:
                        raise RuntimeError(
                            "component_retained_frequency requires component labels."
                        )
                    labels, labelled_mask, range_training_target_transform = target_fn(
                        labels=labels,
                        labelled_mask=labelled_mask,
                        component_labels=component_labels,
                        boundaries=train_boundaries,
                        model_config=config.model,
                    )
                range_training_target_transform["enabled"] = True
                range_training_target_transform["replicate_count"] = len(train_label_sets)
                range_training_target_transform["replicate_target_aggregation"] = (
                    range_replicate_target_aggregation
                )
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
                "'continuity_retained_frequency', or a QueryUsefulV1 factorized target mode."
            )
    range_target_balance_mode = str(
        getattr(config.model, "range_target_balance_mode", "none")
    ).lower()
    if range_target_balance_mode != "none":
        if train_labels is None:
            raise RuntimeError(
                "range_target_balance_mode requires precomputed range training labels."
            )
        with phase("range-target-balance"):
            labels, labelled_mask = train_labels
            labels, labelled_mask, range_target_balance_diagnostics = (
                balance_range_training_target_by_trajectory(
                    labels=labels,
                    labelled_mask=labelled_mask,
                    boundaries=train_boundaries,
                    mode=range_target_balance_mode,
                )
            )
            train_labels = (labels, labelled_mask)
            print(
                f"  target balance={range_target_balance_diagnostics['mode']} "
                f"positives={range_target_balance_diagnostics['positive_label_count']} "
                f"mass={range_target_balance_diagnostics['positive_label_mass']:.4f} "
                f"trajectories={range_target_balance_diagnostics['balanced_trajectory_count']}",
                flush=True,
            )
    if range_training_target_mode not in QUERY_USEFUL_V1_TARGET_MODES:
        range_training_target_transform.setdefault("target_family", "legacy_range_useful_scalar")
        range_training_target_transform.setdefault("final_success_allowed", False)
        range_training_target_transform.setdefault(
            "legacy_reason",
            "Old RangeUseful/scalar-target diagnostic path. "
            "Not valid for query-driven rework acceptance.",
        )
    return TargetPreparationOutputs(
        train_labels=train_labels,
        range_training_target_mode=range_training_target_mode,
        range_training_target_transform=range_training_target_transform,
        range_target_balance_diagnostics=range_target_balance_diagnostics,
        range_training_label_aggregation=range_training_label_aggregation,
        teacher_distillation_diagnostics=teacher_distillation_diagnostics,
        selection_query_cache=selection_query_cache,
        selection_geometry_scores=selection_geometry_scores,
    )
