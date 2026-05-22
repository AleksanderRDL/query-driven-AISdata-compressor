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
from learning.targets.aggregation import aggregate_range_component_label_sets
from learning.targets.common import (
    aggregate_range_label_sets,
    balance_range_training_target_by_trajectory,
)
from learning.targets.query_local_utility import (
    QUERY_LOCAL_UTILITY_FACTORIZED_TARGET_MODE,
    QUERY_LOCAL_UTILITY_TARGET_MODES,
)
from learning.targets.registry import RangeTargetModeSpec, range_scalar_target_mode_spec
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


def _label_aggregation_mode(replicate_target_aggregation: str) -> str:
    return "max" if replicate_target_aggregation == "label_max" else "mean"


def _target_summary_line(target_phase: str, diagnostics: dict[str, Any]) -> str:
    return (
        f"  {target_phase} target: "
        f"positives={diagnostics['positive_label_count']} "
        f"fraction={diagnostics['positive_label_fraction']:.4f} "
        f"mass={diagnostics['positive_label_mass']:.4f}"
    )


def _component_labels_required(
    mode: str,
    component_label_sets: list[dict[str, torch.Tensor] | None],
) -> None:
    if not component_label_sets or any(
        component_labels is None for component_labels in component_label_sets
    ):
        raise RuntimeError(
            f"{mode} requires range component labels; use range_label_mode=usefulness."
        )


def _run_range_target_transform(
    *,
    spec: RangeTargetModeSpec,
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    train_points: torch.Tensor,
    train_boundaries: list[tuple[int, int]],
    train_workload: TypedQueryWorkload,
    model_config: object,
    component_labels: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    target_kwargs: dict[str, Any] = {
        "labels": labels,
        "labelled_mask": labelled_mask,
        "boundaries": train_boundaries,
        "model_config": model_config,
    }
    if spec.requires_points:
        target_kwargs["points"] = train_points
    if spec.requires_typed_queries:
        target_kwargs["typed_queries"] = train_workload.typed_queries
    if spec.requires_component_labels:
        if component_labels is None:
            raise RuntimeError(f"{spec.mode} requires component labels.")
        target_kwargs["component_labels"] = component_labels
    return spec.target_fn(**target_kwargs)


def _run_frequency_mean_range_target(
    *,
    spec: RangeTargetModeSpec,
    train_label_sets: list[RangeLabels],
    train_component_label_sets: list[dict[str, torch.Tensor] | None],
    train_points: torch.Tensor,
    train_boundaries: list[tuple[int, int]],
    model_config: object,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not spec.supports_frequency_mean or spec.aggregate_target_fn is None:
        if spec.mode == "historical_prior_retained_frequency":
            raise RuntimeError(
                "historical_prior_retained_frequency does not support "
                "range_replicate_target_aggregation='frequency_mean'; use label_mean or label_max."
            )
        raise RuntimeError(
            f"{spec.mode} does not support range_replicate_target_aggregation='frequency_mean'."
        )
    aggregate_target_kwargs: dict[str, Any] = {
        "label_sets": train_label_sets,
        "boundaries": train_boundaries,
        "model_config": model_config,
    }
    if spec.requires_points:
        aggregate_target_kwargs["points"] = train_points
    if spec.requires_component_labels:
        aggregate_target_kwargs["component_label_sets"] = train_component_label_sets
    return spec.aggregate_target_fn(**aggregate_target_kwargs)


def _run_label_aggregated_range_target(
    *,
    spec: RangeTargetModeSpec,
    train_label_sets: list[RangeLabels],
    train_component_label_sets: list[dict[str, torch.Tensor] | None],
    train_points: torch.Tensor,
    train_boundaries: list[tuple[int, int]],
    train_workload: TypedQueryWorkload,
    model_config: object,
    replicate_target_aggregation: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
    aggregation_mode = _label_aggregation_mode(replicate_target_aggregation)
    if spec.requires_component_labels:
        labels, labelled_mask, component_labels, aggregation_diagnostics = (
            aggregate_range_component_label_sets(
                label_sets=train_label_sets,
                component_label_sets=train_component_label_sets,
                aggregation=aggregation_mode,
            )
        )
    else:
        labels, labelled_mask, aggregation_diagnostics = aggregate_range_label_sets(
            label_sets=train_label_sets,
            source=f"range_label_{aggregation_mode}_before_{spec.mode}",
            aggregation=aggregation_mode,
        )
        component_labels = None

    labels, labelled_mask, target_transform = _run_range_target_transform(
        spec=spec,
        labels=labels,
        labelled_mask=labelled_mask,
        train_points=train_points,
        train_boundaries=train_boundaries,
        train_workload=train_workload,
        model_config=model_config,
        component_labels=component_labels,
    )
    target_transform["label_aggregation"] = aggregation_diagnostics
    return labels, labelled_mask, target_transform, aggregation_diagnostics


def _prepare_scalar_range_target(
    *,
    spec: RangeTargetModeSpec,
    train_labels: RangeLabels | None,
    train_label_sets: list[RangeLabels],
    train_component_label_sets: list[dict[str, torch.Tensor] | None],
    train_points: torch.Tensor,
    train_boundaries: list[tuple[int, int]],
    train_workload: TypedQueryWorkload,
    model_config: object,
    replicate_target_aggregation: str,
    range_training_label_aggregation: dict[str, Any],
    phase: PhaseLogger,
) -> tuple[RangeLabels, dict[str, Any]]:
    if train_labels is None:
        raise RuntimeError(f"{spec.mode} target mode requires precomputed range training labels.")
    if len(train_label_sets) > 1 and not spec.supports_multiple_replicates:
        raise RuntimeError(f"{spec.mode} does not yet support multiple train workload replicates.")
    if spec.requires_component_labels:
        _component_labels_required(spec.mode, train_component_label_sets)

    target_phase = spec.mode.replace("_", "-")
    with phase(f"range-{target_phase}-target"):
        if len(train_label_sets) > 1:
            if replicate_target_aggregation == "frequency_mean":
                labels, labelled_mask, target_transform = _run_frequency_mean_range_target(
                    spec=spec,
                    train_label_sets=train_label_sets,
                    train_component_label_sets=train_component_label_sets,
                    train_points=train_points,
                    train_boundaries=train_boundaries,
                    model_config=model_config,
                )
                range_training_label_aggregation["enabled"] = True
                range_training_label_aggregation["target_mode"] = spec.mode
                range_training_label_aggregation["replicate_target_aggregation"] = "frequency_mean"
            else:
                labels, labelled_mask, target_transform, aggregation_diagnostics = (
                    _run_label_aggregated_range_target(
                        spec=spec,
                        train_label_sets=train_label_sets,
                        train_component_label_sets=train_component_label_sets,
                        train_points=train_points,
                        train_boundaries=train_boundaries,
                        train_workload=train_workload,
                        model_config=model_config,
                        replicate_target_aggregation=replicate_target_aggregation,
                    )
                )
                range_training_label_aggregation.update(aggregation_diagnostics)
                range_training_label_aggregation["enabled"] = True
                range_training_label_aggregation["target_mode"] = spec.mode
                range_training_label_aggregation["replicate_target_aggregation"] = (
                    replicate_target_aggregation
                )
            target_transform["replicate_target_aggregation"] = replicate_target_aggregation
        else:
            labels, labelled_mask = train_labels
            component_labels = (
                train_component_label_sets[0] if spec.requires_component_labels else None
            )
            labels, labelled_mask, target_transform = _run_range_target_transform(
                spec=spec,
                labels=labels,
                labelled_mask=labelled_mask,
                train_points=train_points,
                train_boundaries=train_boundaries,
                train_workload=train_workload,
                model_config=model_config,
                component_labels=component_labels,
            )
        target_transform["enabled"] = True
        target_transform["replicate_count"] = len(train_label_sets)
        if spec.requires_component_labels:
            target_transform["replicate_target_aggregation"] = replicate_target_aggregation
        print(_target_summary_line(target_phase, target_transform), flush=True)
        return (labels, labelled_mask), target_transform


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
    if range_training_target_mode in QUERY_LOCAL_UTILITY_TARGET_MODES:
        range_training_target_transform.update(
            {
                "enabled": True,
                "target_family": "QueryLocalUtilityFactorized",
                "final_success_allowed": (
                    range_training_target_mode == QUERY_LOCAL_UTILITY_FACTORIZED_TARGET_MODE
                ),
            }
        )
        if range_training_target_mode != QUERY_LOCAL_UTILITY_FACTORIZED_TARGET_MODE:
            range_training_target_transform["diagnostic_reason"] = (
                "Experimental QueryLocalUtility target mode. "
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
            range_training_target_mode not in QUERY_LOCAL_UTILITY_TARGET_MODES
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
    elif range_training_target_mode not in {"point_value", *QUERY_LOCAL_UTILITY_TARGET_MODES}:
        spec = range_scalar_target_mode_spec(range_training_target_mode)
        train_labels, range_training_target_transform = _prepare_scalar_range_target(
            spec=spec,
            train_labels=train_labels,
            train_label_sets=train_label_sets,
            train_component_label_sets=train_component_label_sets,
            train_points=train_points,
            train_boundaries=train_boundaries,
            train_workload=train_workload,
            model_config=config.model,
            replicate_target_aggregation=range_replicate_target_aggregation,
            range_training_label_aggregation=range_training_label_aggregation,
            phase=phase,
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
    if range_training_target_mode not in QUERY_LOCAL_UTILITY_TARGET_MODES:
        range_training_target_transform.setdefault("target_family", "legacy_range_useful_scalar")
        range_training_target_transform.setdefault("final_success_allowed", False)
        range_training_target_transform.setdefault(
            "legacy_reason",
            "Old RangeUseful/scalar-target diagnostic path. "
            "Not valid for QueryLocalUtility final acceptance.",
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
