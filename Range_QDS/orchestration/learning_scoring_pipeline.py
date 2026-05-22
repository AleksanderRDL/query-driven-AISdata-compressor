"""Single-run learning/scoring pipeline helpers. See orchestration/README.md for details."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

import torch

from config.run_config import RunConfig, derive_seed_bundle
from learning.checkpoints import ModelArtifacts, save_checkpoint
from learning.model_features import is_workload_blind_model_type
from learning.model_training import train_model
from learning.predictability_audit import (
    query_prior_predictability_audit,
)
from orchestration.data_splits import build_run_datasets, prepare_run_split
from orchestration.final_gate_summary import build_final_run_summaries
from orchestration.learning_target_stage import prepare_training_targets
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
from orchestration.run_payload import build_run_payload
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
    needs_validation_score = (
        selection_metric in {"score", "uniform_gap"} or validation_score_every > 0
    )
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

    range_diagnostics_summary: dict[str, Any] = {}
    range_diagnostics_rows: list[dict[str, Any]] = []
    range_runtime_caches = {
        "train": RangeRuntimeCache(),
        "eval": RangeRuntimeCache(),
        "selection": RangeRuntimeCache(),
    }
    workload_distribution_comparison: dict[str, Any] = {"deltas_vs_eval": {}}

    export_eval_queries_geojson(
        save_queries_dir=save_queries_dir,
        eval_workload=eval_workload,
        phase=_phase,
    )

    reset_cuda_peak_memory_stats()
    mlqds_range_geometry_blend = max(
        0.0, min(1.0, float(getattr(config.model, "mlqds_range_geometry_blend", 0.0)))
    )
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
    audit_ratios = range_audit_ratios(config)
    selector_budget_ratios = tuple(
        sorted({float(config.model.compression_ratio), *(float(ratio) for ratio in audit_ratios)})
    )
    if (
        str(getattr(config.model, "selector_type", TEMPORAL_HYBRID_SELECTOR_TYPE)).lower()
        == LEARNED_SEGMENT_BUDGET_SELECTOR_TYPE
    ):
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
    selection_causality_payload: dict[str, Any] = {"available": False, "reason": "not_run"}
    selection_selector_trace: dict[str, Any] | None = None
    train_marginal_causality_payload: dict[str, Any] = {
        "available": False,
        "reason": "disabled",
    }
    train_selector_trace: dict[str, Any] | None = None
    if workload_blind_eval:
        if bool(getattr(config.model, "query_local_utility_train_marginal_diagnostics", False)):
            with _phase("train-marginal-causality-diagnostics"):
                train_marginal_causality_payload = build_selection_causality_diagnostics(
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
                raw_train_trace = train_marginal_causality_payload.pop(
                    "selection_selector_trace_diagnostics",
                    None,
                )
                if isinstance(raw_train_trace, dict):
                    train_selector_trace = raw_train_trace
        with _phase("selection-causality-diagnostics"):
            selection_causality_payload = build_selection_causality_diagnostics(
                trained=trained,
                selection_points=selection_points,
                selection_boundaries=selection_boundaries,
                selection_workload=selection_workload,
                eval_workload_map=eval_workload_map,
                selection_query_cache=selection_query_cache,
                config=config,
                seeds=seeds,
            )
            raw_selection_trace = selection_causality_payload.pop(
                "selection_selector_trace_diagnostics",
                None,
            )
            if isinstance(raw_selection_trace, dict):
                selection_selector_trace = raw_selection_trace
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
    methods = retained_mask_freezing.methods
    frozen_primary_masks = retained_mask_freezing.frozen_primary_masks
    frozen_audit_methods_by_ratio = retained_mask_freezing.frozen_audit_methods_by_ratio
    frozen_primary_scores = retained_mask_freezing.frozen_primary_scores
    frozen_primary_head_logits = retained_mask_freezing.frozen_primary_head_logits
    frozen_primary_segment_scores = retained_mask_freezing.frozen_primary_segment_scores
    frozen_primary_selector_segment_scores = (
        retained_mask_freezing.frozen_primary_selector_segment_scores
    )
    primary_selector_trace = retained_mask_freezing.primary_selector_trace
    causality_ablation_methods = retained_mask_freezing.causality_ablation_methods
    causal_ablation_freeze_failures = retained_mask_freezing.causal_ablation_freeze_failures
    prior_sensitivity_diagnostics = retained_mask_freezing.prior_sensitivity_diagnostics
    prior_channel_ablation_diagnostics = retained_mask_freezing.prior_channel_ablation_diagnostics
    head_ablation_sensitivity_diagnostics = (
        retained_mask_freezing.head_ablation_sensitivity_diagnostics
    )
    segment_budget_head_ablation_mode = retained_mask_freezing.segment_budget_head_ablation_mode
    scoring_stage = run_scoring_stage(
        config=config,
        seeds=seeds,
        trained=trained,
        methods=methods,
        retention_methods=retention_methods,
        workload_blind_eval=workload_blind_eval,
        audit_ratios=audit_ratios,
        frozen_primary_masks=frozen_primary_masks,
        frozen_audit_methods_by_ratio=frozen_audit_methods_by_ratio,
        frozen_primary_scores=frozen_primary_scores,
        frozen_primary_head_logits=frozen_primary_head_logits,
        frozen_primary_segment_scores=frozen_primary_segment_scores,
        frozen_primary_selector_segment_scores=frozen_primary_selector_segment_scores,
        causality_ablation_methods=causality_ablation_methods,
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
    matched = scoring_stage.matched
    matched_table = scoring_stage.matched_table
    geometric_table = scoring_stage.geometric_table
    range_usefulness_table = scoring_stage.range_usefulness_table
    learned_fill_diagnostics = scoring_stage.learned_fill_diagnostics
    learned_fill_table = scoring_stage.learned_fill_table
    causality_ablation_scores = scoring_stage.causality_ablation_scores
    causality_ablation_mask_diagnostics = scoring_stage.causality_ablation_mask_diagnostics
    range_compression_audit = scoring_stage.range_compression_audit
    range_compression_audit_table = scoring_stage.range_compression_audit_table
    workload_scoring_compatibility_diagnostics = (
        scoring_stage.workload_scoring_compatibility_diagnostics
    )
    shift_pairs = scoring_stage.shift_pairs
    shift_table = scoring_stage.shift_table
    segment_oracle_allocation_audit = scoring_stage.segment_oracle_allocation_audit
    target_segment_oracle_alignment_audit = scoring_stage.target_segment_oracle_alignment_audit
    run_oracle_baseline = scoring_stage.run_oracle_baseline

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
        range_diagnostics_summary["train"] = train_summary
        range_diagnostics_summary["eval"] = eval_summary
        range_diagnostics_rows.extend(train_rows)
        range_diagnostics_rows.extend(eval_rows)
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
            range_diagnostics_summary[replicate_label] = replicate_summary
            range_diagnostics_rows.extend(replicate_rows)
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
            range_diagnostics_summary["selection"] = selection_summary
            range_diagnostics_rows.extend(selection_rows)
        print_range_diagnostics_summary(range_diagnostics_summary)
        workload_distribution_comparison = range_workload_distribution_comparison(
            range_diagnostics_summary
        )
        print_range_distribution_comparison(workload_distribution_comparison)

    range_learned_fill_summary = build_range_learned_fill_summary(
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
        matched=matched,
        selector_budget_diagnostics=selector_budget_diagnostics,
        primary_selector_trace=primary_selector_trace,
        causality_ablation_scores=causality_ablation_scores,
        causality_ablation_mask_diagnostics=causality_ablation_mask_diagnostics,
        causal_ablation_freeze_failures=causal_ablation_freeze_failures,
        prior_sensitivity_diagnostics=prior_sensitivity_diagnostics,
        prior_channel_ablation_diagnostics=prior_channel_ablation_diagnostics,
        head_ablation_sensitivity_diagnostics=head_ablation_sensitivity_diagnostics,
        selection_causality_diagnostics=selection_causality_payload,
        segment_oracle_allocation_audit=segment_oracle_allocation_audit,
        target_segment_oracle_alignment_audit=target_segment_oracle_alignment_audit,
        segment_budget_head_ablation_mode=segment_budget_head_ablation_mode,
        predictability_audit=predictability_audit,
        workload_distribution_comparison=workload_distribution_comparison,
    )
    dump = build_run_payload(
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
        primary_selector_trace=primary_selector_trace,
        selection_selector_trace=selection_selector_trace,
        train_selector_trace=train_selector_trace,
        train_marginal_causality_diagnostics=train_marginal_causality_payload,
        segment_oracle_allocation_audit=segment_oracle_allocation_audit,
        target_segment_oracle_alignment_audit=target_segment_oracle_alignment_audit,
        matched=matched,
        causality_ablation_scores=causality_ablation_scores,
        learned_fill_diagnostics=learned_fill_diagnostics,
        range_learned_fill_summary=range_learned_fill_summary,
        predictability_audit=predictability_audit,
        workload_scoring_compatibility_diagnostics=(workload_scoring_compatibility_diagnostics),
        range_compression_audit=range_compression_audit,
        shift_pairs=shift_pairs,
        range_training_target_transform=range_training_target_transform,
        range_target_balance_diagnostics=range_target_balance_diagnostics,
        range_training_label_aggregation=range_training_label_aggregation,
        teacher_distillation_diagnostics=teacher_distillation_diagnostics,
        selection_metric=selection_metric,
        workload_blind_eval=workload_blind_eval,
        frozen_primary_masks=frozen_primary_masks,
        frozen_audit_methods_by_ratio=frozen_audit_methods_by_ratio,
        data_audit=data_audit,
        range_diagnostics_summary=range_diagnostics_summary,
        workload_distribution_comparison=workload_distribution_comparison,
        training_cuda_memory=training_cuda_memory,
        run_oracle_baseline=run_oracle_baseline,
    )

    with _phase("write-results"):
        out_dir = write_run_results(
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

    export_simplified_eval_csvs(
        save_simplified_dir=save_simplified_dir,
        matched=matched,
        config=config,
        trained=trained,
        eval_workload=eval_workload,
        eval_workload_map=eval_workload_map,
        test_points=test_points,
        test_boundaries=test_boundaries,
        test_mmsis=test_mmsis,
        phase=_phase,
    )

    print(f"[pipeline] total runtime {time.perf_counter() - pipeline_t0:.2f}s", flush=True)
    return RunOutputs(
        matched_table=matched_table,
        shift_table=shift_table,
        metrics_dump=dump,
        geometric_table=geometric_table,
        range_usefulness_table=range_usefulness_table,
        range_compression_audit_table=range_compression_audit_table,
    )
