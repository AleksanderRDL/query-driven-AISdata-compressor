"""Single-run learning/scoring pipeline helpers. See orchestration/README.md for details."""

from __future__ import annotations

import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from config.run_config import RunConfig, SeedBundle, derive_seed_bundle
from learning.checkpoints import ModelArtifacts, save_checkpoint
from learning.model_features import is_workload_blind_model_type
from learning.model_training import train_model
from learning.predictability_audit import (
    query_prior_predictability_audit,
)
from orchestration.data_splits import build_run_datasets, prepare_run_split
from orchestration.final_gate_summary import build_final_run_summaries
from orchestration.learning_target_stage import RangeLabels, prepare_training_targets
from orchestration.range_diagnostics import (
    build_range_learned_fill_summary,
    print_range_diagnostics_summary,
    print_range_distribution_comparison,
    range_audit_ratios,
    range_workload_diagnostics,
    range_workload_distribution_comparison,
)
from orchestration.range_runtime_cache import RangeRuntimeCache
from orchestration.retained_mask_stage import freeze_workload_blind_retained_masks
from orchestration.run_artifacts import RunOutputs, write_run_results
from orchestration.run_exports import export_eval_queries_geojson, export_simplified_eval_csvs
from orchestration.run_payload import RunPayloadInputs, build_run_payload
from orchestration.scoring_methods import (
    build_primary_methods,
)
from orchestration.scoring_stage import run_scoring_stage
from orchestration.selection_causality_diagnostics import build_selection_causality_diagnostics
from orchestration.workload_stage import (
    generate_run_workloads,
    resolve_workload_maps,
    workload_name,
)
from runtime.torch_runtime import (
    cuda_memory_snapshot,
    reset_cuda_peak_memory_stats,
)
from selection.learned_segment_budget import (
    learned_segment_budget_diagnostics,
)
from selection.retained_mask_selectors import temporal_hybrid_selector_budget_diagnostics
from selection.selector_types import (
    LEARNED_SEGMENT_BUDGET_SELECTOR_TYPE,
    TEMPORAL_HYBRID_SELECTOR_TYPE,
)
from workloads.query_types import single_workload_type


@dataclass(frozen=True)
class _SelectionCausalityStageOutputs:
    selection_payload: dict[str, Any]
    selection_selector_trace: dict[str, Any] | None
    train_marginal_payload: dict[str, Any]
    train_selector_trace: dict[str, Any] | None


@dataclass(frozen=True)
class _RangeDiagnosticsStageOutputs:
    summary: dict[str, Any]
    rows: list[dict[str, Any]]
    distribution_comparison: dict[str, Any]


@contextmanager
def _phase(name: str):
    """Log a named phase with wall-clock timing."""
    print(f"[{name}] starting...", flush=True)
    phase_started_at = time.perf_counter()
    try:
        yield
    finally:
        elapsed_seconds = time.perf_counter() - phase_started_at
        print(f"[{name}] done in {elapsed_seconds:.2f}s", flush=True)


def _print_pipeline_header(
    *,
    trajectories: list[torch.Tensor],
    validation_trajectories: list[torch.Tensor] | None,
    eval_trajectories: list[torch.Tensor] | None,
    eval_workload_map: dict[str, float],
) -> None:
    if eval_trajectories is None:
        print(
            f"[pipeline] {len(trajectories)} trajectories, workload={workload_name(eval_workload_map)}",
            flush=True,
        )
        return
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


def _selection_metric_and_validation_need(config: RunConfig) -> tuple[str, bool]:
    selection_metric = str(getattr(config.model, "checkpoint_selection_metric", "score")).lower()
    validation_score_every = int(getattr(config.model, "validation_score_every", 0) or 0)
    return (
        selection_metric,
        selection_metric in {"score", "uniform_gap"} or validation_score_every > 0,
    )


def _selector_budget_diagnostics(
    *,
    config: RunConfig,
    train_boundaries: list[tuple[int, int]],
    test_boundaries: list[tuple[int, int]],
    audit_ratios: Sequence[float],
) -> dict[str, Any]:
    selector_budget_ratios = tuple(
        sorted({float(config.model.compression_ratio), *(float(ratio) for ratio in audit_ratios)})
    )
    if (
        str(getattr(config.model, "selector_type", TEMPORAL_HYBRID_SELECTOR_TYPE)).lower()
        == LEARNED_SEGMENT_BUDGET_SELECTOR_TYPE
    ):
        return {
            "train": learned_segment_budget_diagnostics(train_boundaries, selector_budget_ratios),
            "eval": learned_segment_budget_diagnostics(test_boundaries, selector_budget_ratios),
        }
    return {
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


def _new_range_runtime_caches() -> dict[str, RangeRuntimeCache]:
    return {
        "train": RangeRuntimeCache(),
        "eval": RangeRuntimeCache(),
        "selection": RangeRuntimeCache(),
    }


def _mlqds_range_geometry_blend(config: RunConfig) -> float:
    return max(0.0, min(1.0, float(getattr(config.model, "mlqds_range_geometry_blend", 0.0))))


def _run_selection_causality_stages(
    *,
    workload_blind_eval: bool,
    config: RunConfig,
    trained: Any,
    train_points: torch.Tensor,
    train_boundaries: list[tuple[int, int]],
    train_workload: Any,
    train_workload_map: dict[str, float],
    selection_points: torch.Tensor | None,
    selection_boundaries: list[tuple[int, int]] | None,
    selection_workload: Any | None,
    eval_workload_map: dict[str, float],
    selection_query_cache: Any,
    range_runtime_caches: dict[str, RangeRuntimeCache],
    seeds: SeedBundle,
) -> _SelectionCausalityStageOutputs:
    selection_payload: dict[str, Any] = {"available": False, "reason": "not_run"}
    selection_selector_trace: dict[str, Any] | None = None
    train_marginal_payload: dict[str, Any] = {"available": False, "reason": "disabled"}
    train_selector_trace: dict[str, Any] | None = None
    if not workload_blind_eval:
        return _SelectionCausalityStageOutputs(
            selection_payload=selection_payload,
            selection_selector_trace=selection_selector_trace,
            train_marginal_payload=train_marginal_payload,
            train_selector_trace=train_selector_trace,
        )

    if bool(getattr(config.model, "query_local_utility_train_marginal_diagnostics", False)):
        with _phase("train-marginal-causality-diagnostics"):
            train_marginal_payload = build_selection_causality_diagnostics(
                trained=trained,
                selection_points=train_points,
                selection_boundaries=train_boundaries,
                selection_workload=train_workload,
                eval_workload_map=train_workload_map,
                selection_query_cache=range_runtime_caches["train"].query_cache,
                config=config,
                seeds=seeds,
                diagnostic_split="train",
                selector_trace_layout_name="train_primary",
            )
            raw_train_trace = train_marginal_payload.pop(
                "selection_selector_trace_diagnostics",
                None,
            )
            if isinstance(raw_train_trace, dict):
                train_selector_trace = raw_train_trace
    with _phase("selection-causality-diagnostics"):
        selection_payload = build_selection_causality_diagnostics(
            trained=trained,
            selection_points=selection_points,
            selection_boundaries=selection_boundaries,
            selection_workload=selection_workload,
            eval_workload_map=eval_workload_map,
            selection_query_cache=selection_query_cache,
            config=config,
            seeds=seeds,
        )
        raw_selection_trace = selection_payload.pop(
            "selection_selector_trace_diagnostics",
            None,
        )
        if isinstance(raw_selection_trace, dict):
            selection_selector_trace = raw_selection_trace
    return _SelectionCausalityStageOutputs(
        selection_payload=selection_payload,
        selection_selector_trace=selection_selector_trace,
        train_marginal_payload=train_marginal_payload,
        train_selector_trace=train_selector_trace,
    )


def _run_range_diagnostics_stage(
    *,
    config: RunConfig,
    seeds: SeedBundle,
    train_points: torch.Tensor,
    test_points: torch.Tensor,
    selection_points: torch.Tensor | None,
    train_boundaries: list[tuple[int, int]],
    test_boundaries: list[tuple[int, int]],
    selection_boundaries: list[tuple[int, int]] | None,
    train_workload: Any,
    train_label_workloads: list[Any],
    train_label_workload_seeds: list[int],
    eval_workload: Any,
    selection_workload: Any | None,
    train_workload_map: dict[str, float],
    eval_workload_map: dict[str, float],
    range_runtime_caches: dict[str, RangeRuntimeCache],
) -> _RangeDiagnosticsStageOutputs:
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    with _phase("range-diagnostics"):
        train_summary, train_rows = range_workload_diagnostics(
            "train",
            train_points,
            train_boundaries,
            train_workload,
            train_workload_map,
            config,
            seeds.train_query_seed,
            range_runtime_caches["train"],
        )
        eval_summary, eval_rows = range_workload_diagnostics(
            "eval",
            test_points,
            test_boundaries,
            eval_workload,
            eval_workload_map,
            config,
            seeds.eval_query_seed,
            range_runtime_caches["eval"],
        )
        summary["train"] = train_summary
        summary["eval"] = eval_summary
        rows.extend(train_rows)
        rows.extend(eval_rows)
        for replicate_index, replicate_workload in enumerate(train_label_workloads[1:], start=1):
            replicate_label = f"train_r{replicate_index}"
            replicate_summary, replicate_rows = range_workload_diagnostics(
                replicate_label,
                train_points,
                train_boundaries,
                replicate_workload,
                train_workload_map,
                config,
                train_label_workload_seeds[replicate_index],
                RangeRuntimeCache(),
            )
            summary[replicate_label] = replicate_summary
            rows.extend(replicate_rows)
        if (
            selection_workload is not None
            and selection_points is not None
            and selection_boundaries is not None
        ):
            selection_summary, selection_rows = range_workload_diagnostics(
                "selection",
                selection_points,
                selection_boundaries,
                selection_workload,
                eval_workload_map,
                config,
                seeds.eval_query_seed + 17,
                range_runtime_caches["selection"],
            )
            summary["selection"] = selection_summary
            rows.extend(selection_rows)
        print_range_diagnostics_summary(summary)
        distribution_comparison = range_workload_distribution_comparison(summary)
        print_range_distribution_comparison(distribution_comparison)
    return _RangeDiagnosticsStageOutputs(
        summary=summary,
        rows=rows,
        distribution_comparison=distribution_comparison,
    )


def _train_model_stage(
    *,
    config: RunConfig,
    seeds: SeedBundle,
    train_traj: list[torch.Tensor],
    selection_traj: list[torch.Tensor] | None,
    train_boundaries: list[tuple[int, int]],
    selection_boundaries: list[tuple[int, int]] | None,
    train_workload: Any,
    selection_workload: Any | None,
    train_workload_map: dict[str, float],
    eval_workload_map: dict[str, float],
    train_labels: RangeLabels | None,
    selection_points: torch.Tensor | None,
    selection_query_cache: Any,
    selection_geometry_scores: Any,
    train_source_ids: list[int] | None,
    train_mmsis: list[int] | None,
    train_label_workloads: list[Any],
    train_label_workload_seeds: list[int],
) -> Any:
    with _phase(f"train-model ({config.model.epochs} epochs)"):
        return train_model(
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


def _save_model_checkpoint(
    *,
    save_model: str | None,
    trained: Any,
    config: RunConfig,
    eval_workload_map: dict[str, float],
) -> None:
    if not save_model:
        return
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


def _print_training_cuda_memory(training_cuda_memory: dict[str, Any]) -> None:
    if training_cuda_memory.get("available"):
        print(
            f"  train_cuda_peak_allocated={training_cuda_memory['max_allocated_mb']:.1f} MiB  "
            f"peak_reserved={training_cuda_memory['max_reserved_mb']:.1f} MiB",
            flush=True,
        )


def _write_pipeline_results(
    *,
    results_dir: str,
    scoring_stage: Any,
    range_learned_fill_summary: dict[str, Any],
    range_diagnostics: _RangeDiagnosticsStageOutputs,
    run_payload: dict[str, Any],
) -> None:
    with _phase("write-results"):
        out_dir = write_run_results(
            results_dir=results_dir,
            matched_table=scoring_stage.matched_table,
            shift_table=scoring_stage.shift_table,
            geometric_table=scoring_stage.geometric_table,
            range_audit_table=scoring_stage.range_audit_table,
            learned_fill_table=scoring_stage.learned_fill_table,
            learned_fill_diagnostics=scoring_stage.learned_fill_diagnostics,
            range_learned_fill_summary=range_learned_fill_summary,
            range_compression_audit=scoring_stage.range_compression_audit,
            range_compression_audit_table=scoring_stage.range_compression_audit_table,
            range_diagnostics_summary=range_diagnostics.summary,
            workload_distribution_comparison=range_diagnostics.distribution_comparison,
            range_diagnostics_rows=range_diagnostics.rows,
            run_payload=run_payload,
        )
        print(f"  wrote results to {out_dir}", flush=True)


def _export_simplified_outputs(
    *,
    save_simplified_dir: str | None,
    scoring_stage: Any,
    config: RunConfig,
    trained: Any,
    eval_workload: Any,
    eval_workload_map: dict[str, float],
    test_points: torch.Tensor,
    test_boundaries: list[tuple[int, int]],
    test_mmsis: list[int] | None,
) -> None:
    export_simplified_eval_csvs(
        save_simplified_dir=save_simplified_dir,
        matched=scoring_stage.matched,
        config=config,
        trained=trained,
        eval_workload=eval_workload,
        eval_workload_map=eval_workload_map,
        test_points=test_points,
        test_boundaries=test_boundaries,
        test_mmsis=test_mmsis,
        phase=_phase,
    )


def run_learning_scoring_pipeline(
    config: RunConfig,
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
) -> RunOutputs:
    """Run training, matched scoring, and shifted scoring tables. See orchestration/README.md for details."""
    pipeline_t0 = time.perf_counter()
    train_workload_map, eval_workload_map = resolve_workload_maps(config.query.workload)
    _print_pipeline_header(
        trajectories=trajectories,
        validation_trajectories=validation_trajectories,
        eval_trajectories=eval_trajectories,
        eval_workload_map=eval_workload_map,
    )
    seeds = derive_seed_bundle(config.data.seed)
    selection_metric, needs_validation_score = _selection_metric_and_validation_need(config)
    with _phase("split"):
        data_split = prepare_run_split(
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
        datasets = build_run_datasets(data_split)
        train_points = datasets.train_points
        test_points = datasets.test_points
        selection_points = datasets.selection_points
        train_boundaries = datasets.train_boundaries
        test_boundaries = datasets.test_boundaries
        selection_boundaries = datasets.selection_boundaries
    with _phase("generate-workloads"):
        workloads = generate_run_workloads(
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

    range_runtime_caches = _new_range_runtime_caches()

    export_eval_queries_geojson(
        save_queries_dir=save_queries_dir,
        eval_workload=eval_workload,
        phase=_phase,
    )

    reset_cuda_peak_memory_stats()
    mlqds_range_geometry_blend = _mlqds_range_geometry_blend(config)
    target_preparation = prepare_training_targets(
        config=config,
        seeds=seeds,
        train_traj=train_traj,
        train_points=train_points,
        train_boundaries=train_boundaries,
        train_workload=train_workload,
        train_workload_map=train_workload_map,
        train_label_workloads=train_label_workloads,
        train_label_workload_seeds=train_label_workload_seeds,
        train_source_ids=train_source_ids,
        train_mmsis=train_mmsis,
        selection_workload=selection_workload,
        selection_points=selection_points,
        selection_boundaries=selection_boundaries,
        eval_workload_map=eval_workload_map,
        range_runtime_caches=range_runtime_caches,
        phase=_phase,
    )
    train_labels = target_preparation.train_labels
    range_training_target_transform = target_preparation.range_training_target_transform
    range_target_balance_diagnostics = target_preparation.range_target_balance_diagnostics
    range_training_label_aggregation = target_preparation.range_training_label_aggregation
    teacher_distillation_diagnostics = target_preparation.teacher_distillation_diagnostics
    selection_query_cache = target_preparation.selection_query_cache
    selection_geometry_scores = target_preparation.selection_geometry_scores
    trained = _train_model_stage(
        config=config,
        seeds=seeds,
        train_traj=train_traj,
        selection_traj=selection_traj,
        train_boundaries=train_boundaries,
        selection_boundaries=selection_boundaries,
        train_workload=train_workload,
        selection_workload=selection_workload,
        train_workload_map=train_workload_map,
        eval_workload_map=eval_workload_map,
        train_labels=train_labels,
        selection_points=selection_points,
        selection_query_cache=selection_query_cache,
        selection_geometry_scores=selection_geometry_scores,
        train_source_ids=train_source_ids,
        train_mmsis=train_mmsis,
        train_label_workloads=train_label_workloads,
        train_label_workload_seeds=train_label_workload_seeds,
    )
    training_cuda_memory = cuda_memory_snapshot()
    _print_training_cuda_memory(training_cuda_memory)

    _save_model_checkpoint(
        save_model=save_model,
        trained=trained,
        config=config,
        eval_workload_map=eval_workload_map,
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
    audit_ratios = range_audit_ratios(config)
    selector_budget_diagnostics = _selector_budget_diagnostics(
        config=config,
        train_boundaries=train_boundaries,
        test_boundaries=test_boundaries,
        audit_ratios=audit_ratios,
    )
    selection_causality = _run_selection_causality_stages(
        workload_blind_eval=workload_blind_eval,
        config=config,
        trained=trained,
        train_points=train_points,
        train_boundaries=train_boundaries,
        train_workload=train_workload,
        train_workload_map=train_workload_map,
        selection_points=selection_points,
        selection_boundaries=selection_boundaries,
        selection_workload=selection_workload,
        eval_workload_map=eval_workload_map,
        selection_query_cache=selection_query_cache,
        range_runtime_caches=range_runtime_caches,
        seeds=seeds,
    )
    retained_mask_freezing = freeze_workload_blind_retained_masks(
        methods=methods,
        retention_methods=retention_methods,
        workload_blind_eval=workload_blind_eval,
        audit_ratios=audit_ratios,
        config=config,
        trained=trained,
        eval_workload=eval_workload,
        eval_workload_map=eval_workload_map,
        test_mmsis=test_mmsis,
        test_points=test_points,
        test_boundaries=test_boundaries,
        seeds=seeds,
        phase=_phase,
    )
    scoring_stage = run_scoring_stage(
        config=config,
        seeds=seeds,
        trained=trained,
        methods=retained_mask_freezing.methods,
        retention_methods=retention_methods,
        workload_blind_eval=workload_blind_eval,
        audit_ratios=audit_ratios,
        frozen_primary_masks=retained_mask_freezing.frozen_primary_masks,
        frozen_audit_methods_by_ratio=retained_mask_freezing.frozen_audit_methods_by_ratio,
        frozen_primary_scores=retained_mask_freezing.frozen_primary_scores,
        frozen_primary_head_logits=retained_mask_freezing.frozen_primary_head_logits,
        frozen_primary_segment_scores=retained_mask_freezing.frozen_primary_segment_scores,
        frozen_primary_selector_segment_scores=(
            retained_mask_freezing.frozen_primary_selector_segment_scores
        ),
        causality_ablation_methods=retained_mask_freezing.causality_ablation_methods,
        train_workload=train_workload,
        train_workload_map=train_workload_map,
        eval_workload=eval_workload,
        eval_workload_map=eval_workload_map,
        test_points=test_points,
        test_boundaries=test_boundaries,
        test_mmsis=test_mmsis,
        range_runtime_caches=range_runtime_caches,
        save_masks=bool(save_simplified_dir),
        mlqds_range_geometry_blend=mlqds_range_geometry_blend,
        phase=_phase,
    )
    range_diagnostics = _run_range_diagnostics_stage(
        config=config,
        seeds=seeds,
        train_points=train_points,
        test_points=test_points,
        selection_points=selection_points,
        train_boundaries=train_boundaries,
        test_boundaries=test_boundaries,
        selection_boundaries=selection_boundaries,
        train_workload=train_workload,
        train_label_workloads=train_label_workloads,
        train_label_workload_seeds=train_label_workload_seeds,
        eval_workload=eval_workload,
        selection_workload=selection_workload,
        train_workload_map=train_workload_map,
        eval_workload_map=eval_workload_map,
        range_runtime_caches=range_runtime_caches,
    )
    range_diagnostics_summary = range_diagnostics.summary
    workload_distribution_comparison = range_diagnostics.distribution_comparison

    range_learned_fill_summary = build_range_learned_fill_summary(
        learned_fill_diagnostics=scoring_stage.learned_fill_diagnostics,
        training_target_diagnostics=trained.target_diagnostics,
        range_diagnostics_summary=range_diagnostics_summary,
        compression_ratio=float(config.model.compression_ratio),
    )
    predictability_audit = query_prior_predictability_audit(
        points=test_points,
        boundaries=test_boundaries,
        eval_typed_queries=eval_workload.typed_queries,
        query_prior_field=trained.feature_context.get("query_prior_field"),
        target_mode=str(getattr(config.model, "range_training_target_mode", "")),
    )
    final_summaries = build_final_run_summaries(
        config=config,
        trained=trained,
        train_points=train_points,
        test_points=test_points,
        train_label_workloads=train_label_workloads,
        eval_workload=eval_workload,
        selection_workload=selection_workload,
        matched=scoring_stage.matched,
        selector_budget_diagnostics=selector_budget_diagnostics,
        primary_selector_trace=retained_mask_freezing.primary_selector_trace,
        causality_ablation_scores=scoring_stage.causality_ablation_scores,
        causality_ablation_mask_diagnostics=scoring_stage.causality_ablation_mask_diagnostics,
        causal_ablation_freeze_failures=retained_mask_freezing.causal_ablation_freeze_failures,
        prior_sensitivity_diagnostics=retained_mask_freezing.prior_sensitivity_diagnostics,
        prior_channel_ablation_diagnostics=(
            retained_mask_freezing.prior_channel_ablation_diagnostics
        ),
        head_ablation_sensitivity_diagnostics=(
            retained_mask_freezing.head_ablation_sensitivity_diagnostics
        ),
        selection_causality_diagnostics=selection_causality.selection_payload,
        segment_oracle_allocation_audit=scoring_stage.segment_oracle_allocation_audit,
        target_segment_oracle_alignment_audit=scoring_stage.target_segment_oracle_alignment_audit,
        segment_budget_head_ablation_mode=retained_mask_freezing.segment_budget_head_ablation_mode,
        predictability_audit=predictability_audit,
        workload_distribution_comparison=workload_distribution_comparison,
    )
    run_payload = build_run_payload(
        RunPayloadInputs(
            config=config,
            final_summaries=final_summaries,
            trained=trained,
            train_workload=train_workload,
            train_label_workloads=train_label_workloads,
            eval_workload=eval_workload,
            selection_workload=selection_workload,
            eval_workload_map=eval_workload_map,
            data_split_diagnostics=data_split.split_diagnostics,
            selector_budget_diagnostics=selector_budget_diagnostics,
            primary_selector_trace=retained_mask_freezing.primary_selector_trace,
            selection_selector_trace=selection_causality.selection_selector_trace,
            train_selector_trace=selection_causality.train_selector_trace,
            train_marginal_causality_diagnostics=selection_causality.train_marginal_payload,
            segment_oracle_allocation_audit=scoring_stage.segment_oracle_allocation_audit,
            target_segment_oracle_alignment_audit=scoring_stage.target_segment_oracle_alignment_audit,
            matched=scoring_stage.matched,
            causality_ablation_scores=scoring_stage.causality_ablation_scores,
            learned_fill_diagnostics=scoring_stage.learned_fill_diagnostics,
            range_learned_fill_summary=range_learned_fill_summary,
            predictability_audit=predictability_audit,
            workload_scoring_compatibility_diagnostics=(
                scoring_stage.workload_scoring_compatibility_diagnostics
            ),
            range_compression_audit=scoring_stage.range_compression_audit,
            shift_pairs=scoring_stage.shift_pairs,
            range_training_target_transform=range_training_target_transform,
            range_target_balance_diagnostics=range_target_balance_diagnostics,
            range_training_label_aggregation=range_training_label_aggregation,
            teacher_distillation_diagnostics=teacher_distillation_diagnostics,
            selection_metric=selection_metric,
            workload_blind_eval=workload_blind_eval,
            frozen_primary_masks=retained_mask_freezing.frozen_primary_masks,
            frozen_audit_methods_by_ratio=retained_mask_freezing.frozen_audit_methods_by_ratio,
            data_audit=data_audit,
            range_diagnostics_summary=range_diagnostics_summary,
            workload_distribution_comparison=workload_distribution_comparison,
            training_cuda_memory=training_cuda_memory,
            run_oracle_baseline=scoring_stage.run_oracle_baseline,
        )
    )

    _write_pipeline_results(
        results_dir=results_dir,
        scoring_stage=scoring_stage,
        range_learned_fill_summary=range_learned_fill_summary,
        range_diagnostics=range_diagnostics,
        run_payload=run_payload,
    )

    _export_simplified_outputs(
        save_simplified_dir=save_simplified_dir,
        scoring_stage=scoring_stage,
        config=config,
        trained=trained,
        eval_workload=eval_workload,
        eval_workload_map=eval_workload_map,
        test_points=test_points,
        test_boundaries=test_boundaries,
        test_mmsis=test_mmsis,
    )

    print(f"[pipeline] total runtime {time.perf_counter() - pipeline_t0:.2f}s", flush=True)
    return RunOutputs(
        matched_table=scoring_stage.matched_table,
        shift_table=scoring_stage.shift_table,
        metrics_dump=run_payload,
        geometric_table=scoring_stage.geometric_table,
        range_audit_table=scoring_stage.range_audit_table,
        range_compression_audit_table=scoring_stage.range_compression_audit_table,
    )
