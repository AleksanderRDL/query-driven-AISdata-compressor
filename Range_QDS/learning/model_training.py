"""Ranking-based model training on trajectory windows. See learning/README.md for details."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch.amp.grad_scaler import GradScaler

from config.run_config import ModelConfig
from learning.factorized_head_diagnostics import (
    _initialize_factorized_head_output_biases_from_targets,
)
from learning.fit_diagnostics import (
    _training_target_diagnostics,
)
from learning.losses import (
    _budget_loss_ratios,
    _effective_budget_loss_ratios,
    _effective_temporal_residual_label_mode,
    _temporal_base_masks_for_budget_ratios,
)
from learning.model_factory import build_qds_model
from learning.model_features import WORKLOAD_BLIND_RANGE_MODEL_TYPE
from learning.model_setup import (
    _pure_query_type_id,
    _query_frequency_workload_map,
    _single_active_type_id,
    _workload_map_tensor,
)
from learning.model_training_fit_diagnostics import build_final_training_fit_diagnostics
from learning.model_training_helpers import (
    _fit_scaler_for_model,
)
from learning.model_training_helpers import (
    _scalar_training_target_for_mode as _scalar_training_target_for_mode,
)
from learning.model_training_historical_prior import configure_historical_prior_training
from learning.model_training_loop import TrainingEpochLoopPlan, run_training_epochs
from learning.model_training_targets import build_training_target_inputs
from learning.model_training_validation import build_validation_scoring_plan
from learning.outputs import TrainingOutputs
from learning.query_prior_fields import (
    QUERY_PRIOR_FIELD_NAMES,
    query_prior_field_metadata,
)
from learning.supervised_windows import filter_supervised_windows
from learning.targets.common import (
    _apply_temporal_residual_labels,
)
from learning.targets.query_local_utility import (
    QUERY_LOCAL_UTILITY_FACTORIZED_TARGET_MODE,
)
from learning.trajectory_batching import batch_windows, build_trajectory_windows
from runtime.torch_runtime import normalize_amp_mode
from workloads.query_types import ID_TO_QUERY_NAME
from workloads.typed_workload import TypedQueryWorkload


def train_model(
    train_trajectories: list[torch.Tensor],
    train_boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    model_config: ModelConfig,
    seed: int,
    train_workload_map: dict[str, float] | None = None,
    validation_trajectories: list[torch.Tensor] | None = None,
    validation_boundaries: list[tuple[int, int]] | None = None,
    validation_workload: TypedQueryWorkload | None = None,
    validation_workload_map: dict[str, float] | None = None,
    precomputed_labels: tuple[torch.Tensor, torch.Tensor] | None = None,
    validation_points: torch.Tensor | None = None,
    precomputed_validation_query_cache: Any | None = None,
    precomputed_validation_geometry_scores: torch.Tensor | None = None,
    train_trajectory_source_ids: list[int] | None = None,
    train_trajectory_mmsis: list[int] | None = None,
    query_prior_workloads: list[TypedQueryWorkload] | None = None,
    query_prior_workload_seeds: list[int] | None = None,
) -> TrainingOutputs:
    """Train one pure-workload model with trajectory-window ranking losses."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    target_inputs = build_training_target_inputs(
        train_trajectories=train_trajectories,
        train_boundaries=train_boundaries,
        workload=workload,
        model_config=model_config,
        precomputed_labels=precomputed_labels,
        train_trajectory_source_ids=train_trajectory_source_ids,
        train_trajectory_mmsis=train_trajectory_mmsis,
        query_prior_workloads=query_prior_workloads,
        query_prior_workload_seeds=query_prior_workload_seeds,
    )
    all_points = target_inputs.all_points
    train_point_source_ids = target_inputs.train_point_source_ids
    prior_queries = target_inputs.prior_queries
    labels = target_inputs.labels
    labelled_mask = target_inputs.labelled_mask
    factorized_targets = target_inputs.factorized_targets
    factorized_mask = target_inputs.factorized_mask
    factorized_target_diagnostics = target_inputs.factorized_target_diagnostics
    canonical_segment_ids = target_inputs.canonical_segment_ids
    range_training_target_mode = target_inputs.range_training_target_mode
    query_prior_field = target_inputs.query_prior_field
    points = target_inputs.points
    point_dim = target_inputs.point_dim
    run_tag = "main"
    requested_temporal_residual_label_mode = str(
        getattr(model_config, "temporal_residual_label_mode", "none")
    ).lower()
    if requested_temporal_residual_label_mode not in {"none", "temporal"}:
        raise ValueError("temporal_residual_label_mode must be 'none' or 'temporal'.")
    temporal_residual_label_mode = _effective_temporal_residual_label_mode(
        model_config,
        requested_temporal_residual_label_mode,
    )
    if requested_temporal_residual_label_mode != temporal_residual_label_mode:
        print(
            f"  [{run_tag}] temporal_residual_label_mode={requested_temporal_residual_label_mode} "
            f"ignored for mlqds_hybrid_mode={getattr(model_config, 'mlqds_hybrid_mode', 'fill')}; "
            f"using {temporal_residual_label_mode}",
            flush=True,
        )
    loss_objective = str(getattr(model_config, "loss_objective", "budget_topk")).lower()
    if loss_objective not in {
        "ranking_bce",
        "budget_topk",
        "stratified_budget_topk",
        "pointwise_bce",
    }:
        raise ValueError(
            "loss_objective must be 'ranking_bce', 'budget_topk', "
            "'stratified_budget_topk', or 'pointwise_bce'."
        )
    if (
        loss_objective == "stratified_budget_topk"
        and str(getattr(model_config, "mlqds_hybrid_mode", "fill")).lower() != "stratified"
    ):
        raise ValueError(
            "loss_objective='stratified_budget_topk' requires mlqds_hybrid_mode='stratified'."
        )
    configured_budget_ratios = _budget_loss_ratios(model_config)
    budget_ratios = configured_budget_ratios
    temporal_residual_budget_masks: tuple[tuple[float, float, torch.Tensor], ...] = ()
    temporal_residual_union_mask: torch.Tensor | None = None
    workload_type_id = _pure_query_type_id(workload.type_ids)
    if temporal_residual_label_mode == "temporal" and loss_objective == "budget_topk":
        budget_ratios = _effective_budget_loss_ratios(model_config, temporal_residual_label_mode)
        temporal_residual_budget_masks = _temporal_base_masks_for_budget_ratios(
            n_points=int(labels.shape[0]),
            boundaries=train_boundaries,
            budget_ratios=configured_budget_ratios,
            temporal_fraction=float(getattr(model_config, "mlqds_temporal_fraction", 0.50)),
            device=labels.device,
        )
        if temporal_residual_budget_masks:
            temporal_residual_union_mask = torch.zeros(
                (labels.shape[0],), dtype=torch.bool, device=labels.device
            )
            for _total_ratio, _effective_ratio, base_mask in temporal_residual_budget_masks:
                temporal_residual_union_mask |= base_mask
    elif temporal_residual_label_mode == "temporal":
        labels, labelled_mask = _apply_temporal_residual_labels(
            labels=labels,
            labelled_mask=labelled_mask,
            boundaries=train_boundaries,
            compression_ratio=model_config.compression_ratio,
            temporal_fraction=float(getattr(model_config, "mlqds_temporal_fraction", 0.50)),
        )
    training_target, training_target_basis = _scalar_training_target_for_mode(
        labels=labels,
        labelled_mask=labelled_mask,
        workload_type_id=workload_type_id,
        range_training_target_mode=range_training_target_mode,
    )
    training_labelled_mask = labelled_mask[:, workload_type_id]
    base_type_weights_cpu = _workload_map_tensor(
        train_workload_map or _query_frequency_workload_map(workload),
        device=torch.device("cpu"),
    )
    active_type_id = _single_active_type_id(base_type_weights_cpu)
    if active_type_id != workload_type_id:
        raise ValueError(
            "Training workload map and workload query type must refer to the same pure workload."
        )
    active_type_ids = [active_type_id]
    amp_mode = normalize_amp_mode(getattr(model_config, "amp_mode", "off"))
    budget_loss_temperature = float(getattr(model_config, "budget_loss_temperature", 0.10))
    target_diagnostics = _training_target_diagnostics(
        labels=labels,
        labelled_mask=labelled_mask,
        workload_type_id=workload_type_id,
        configured_budget_ratios=configured_budget_ratios,
        effective_budget_ratios=budget_ratios,
        temporal_residual_budget_masks=temporal_residual_budget_masks,
        temporal_residual_label_mode=temporal_residual_label_mode,
        loss_objective=loss_objective,
        temporal_fraction=float(getattr(model_config, "mlqds_temporal_fraction", 0.50)),
        range_training_target_mode=range_training_target_mode,
    )
    target_diagnostics["supervised_scalar_target_basis"] = training_target_basis
    if factorized_target_diagnostics:
        target_diagnostics[QUERY_LOCAL_UTILITY_FACTORIZED_TARGET_MODE] = (
            factorized_target_diagnostics
        )
        target_diagnostics["query_local_utility_loss_weights"] = {
            "aux_loss_weight": float(
                getattr(model_config, "query_local_utility_aux_loss_weight", 0.50)
            ),
            "segment_budget_head_weight": float(
                getattr(model_config, "query_local_utility_segment_budget_head_weight", 0.10)
            ),
            "segment_level_loss_weight": float(
                getattr(model_config, "query_local_utility_segment_level_loss_weight", 0.25)
            ),
            "behavior_rank_loss_weight": float(
                getattr(model_config, "query_local_utility_behavior_rank_loss_weight", 0.0)
            ),
            "sparse_head_rank_loss_weight": float(
                getattr(model_config, "query_local_utility_sparse_head_rank_loss_weight", 0.0)
            ),
            "sparse_head_bce_target_mode": str(
                getattr(model_config, "query_local_utility_sparse_head_bce_target_mode", "raw")
            ).lower(),
        }
    if query_prior_field is not None:
        target_diagnostics["query_prior_field"] = query_prior_field_metadata(query_prior_field)
    if budget_ratios != configured_budget_ratios:
        print(
            f"  [{run_tag}] effective_budget_loss_ratios={list(budget_ratios)} "
            f"from configured={list(configured_budget_ratios)} "
            f"temporal_residual_label_mode={temporal_residual_label_mode} "
            f"mlqds_temporal_fraction={float(getattr(model_config, 'mlqds_temporal_fraction', 0.0)):.3f}",
            flush=True,
        )
    for row in target_diagnostics.get("budget_rows", []):
        print(
            f"  [{run_tag}] residual_budget total={row['total_budget_ratio']:.4f} "
            f"effective_fill={row['effective_fill_budget_ratio']:.4f} "
            f"base_points={row['temporal_base_point_count']} "
            f"candidates={row['candidate_point_count']} "
            f"residual_pos={row['residual_positive_label_count']}",
            flush=True,
        )

    scaler = _fit_scaler_for_model(points, workload.query_features, model_config.model_type)
    if str(model_config.model_type).lower() == WORKLOAD_BLIND_RANGE_MODEL_TYPE:
        target_diagnostics["workload_blind_range_prior_feature_scaling"] = {
            "semantic_zero_preserved": True,
            "prior_feature_names": list(QUERY_PRIOR_FIELD_NAMES),
            "point_min": scaler.point_min[-len(QUERY_PRIOR_FIELD_NAMES) :].detach().cpu().tolist(),
            "point_max": scaler.point_max[-len(QUERY_PRIOR_FIELD_NAMES) :].detach().cpu().tolist(),
        }
    norm_points, norm_queries = scaler.transform(points, workload.query_features)

    model_type = str(model_config.model_type).lower()
    model = build_qds_model(
        model_type=model_type,
        model_config=model_config,
        point_dim=point_dim,
        query_dim=int(norm_queries.shape[1]),
    )
    head_bias_initialization = _initialize_factorized_head_output_biases_from_targets(
        model,
        head_targets=factorized_targets,
        head_mask=factorized_mask,
    )
    if bool(head_bias_initialization.get("available", False)):
        target_diagnostics["factorized_head_bias_initialization"] = head_bias_initialization
    if query_prior_field is not None:
        cast(Any, model).query_prior_field = query_prior_field
    historical_prior_outputs = configure_historical_prior_training(
        model=model,
        model_type=model_type,
        norm_points=norm_points,
        train_boundaries=train_boundaries,
        training_target=training_target,
        train_point_source_ids=train_point_source_ids,
        model_config=model_config,
        target_diagnostics=target_diagnostics,
        training_labelled_mask=training_labelled_mask,
        workload_type_id=workload_type_id,
        scaler=scaler,
        labels=labels,
        labelled_mask=labelled_mask,
        seed=seed,
        run_tag=run_tag,
    )
    if historical_prior_outputs is not None:
        return historical_prior_outputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    norm_points_dev = norm_points.to(device)
    norm_queries_dev = norm_queries.to(device)
    type_ids_dev = workload.type_ids.to(device)
    training_target_dev = training_target.to(device)
    labelled_mask_dev = training_labelled_mask.to(device)
    factorized_targets_dev = (
        factorized_targets.to(device) if factorized_targets is not None else None
    )
    factorized_mask_dev = factorized_mask.to(device) if factorized_mask is not None else None
    canonical_segment_ids_dev = (
        canonical_segment_ids.to(device) if canonical_segment_ids is not None else None
    )
    if temporal_residual_budget_masks:
        temporal_residual_budget_masks = tuple(
            (total_ratio, effective_ratio, base_mask.to(device=device, non_blocking=True))
            for total_ratio, effective_ratio, base_mask in temporal_residual_budget_masks
        )
    if temporal_residual_union_mask is not None:
        temporal_residual_union_mask = temporal_residual_union_mask.to(
            device=device, non_blocking=True
        )

    opt = torch.optim.Adam(model.parameters(), lr=model_config.lr)
    grad_scaler = GradScaler("cuda", enabled=(amp_mode == "fp16" and device.type == "cuda"))
    windows_cpu = build_trajectory_windows(
        points=norm_points,
        boundaries=train_boundaries,
        window_length=model_config.window_length,
        stride=model_config.window_stride,
    )
    raw_window_count = len(windows_cpu)
    windows_cpu, prefiltered_zero_windows = filter_supervised_windows(
        windows=windows_cpu,
        training_target=training_target,
        labelled_mask=training_labelled_mask,
        active_type_id=active_type_id,
        require_positive=loss_objective != "pointwise_bce",
    )
    if int(prefiltered_zero_windows.sum().item()) > 0:
        filtered_parts = []
        for type_idx in active_type_ids:
            type_name = ID_TO_QUERY_NAME.get(type_idx, f"t{type_idx}")
            filtered_parts.append(f"{type_name}={int(prefiltered_zero_windows[type_idx].item())}")
        print(
            f"  [{run_tag}] filtered {raw_window_count - len(windows_cpu)}/{raw_window_count} "
            f"zero-positive training windows before forward ({', '.join(filtered_parts)})",
            flush=True,
        )
    train_batch_size = max(1, int(getattr(model_config, "train_batch_size", 1)))
    windows = batch_windows(windows_cpu, train_batch_size)
    trained_window_count = len(windows_cpu)
    diag_windows = windows_cpu
    diag_every = max(1, int(getattr(model_config, "diagnostic_every", 1)))
    diag_fraction = float(getattr(model_config, "diagnostic_window_fraction", 1.0))
    diag_fraction = min(1.0, max(0.05, diag_fraction))
    validation_plan = build_validation_scoring_plan(
        selection_metric=str(getattr(model_config, "checkpoint_selection_metric", "score")),
        validation_score_every=int(getattr(model_config, "validation_score_every", 0) or 0),
        diag_every=diag_every,
        validation_trajectories=validation_trajectories,
        validation_boundaries=validation_boundaries,
        validation_workload=validation_workload,
        validation_workload_map=validation_workload_map,
        validation_points=validation_points,
        precomputed_validation_query_cache=precomputed_validation_query_cache,
        model_config=model_config,
        run_tag=run_tag,
    )
    epoch_loop_result = run_training_epochs(
        TrainingEpochLoopPlan(
            model=model,
            windows=windows,
            opt=opt,
            grad_scaler=grad_scaler,
            model_config=model_config,
            device=device,
            amp_mode=amp_mode,
            norm_points_dev=norm_points_dev,
            norm_queries_dev=norm_queries_dev,
            type_ids_dev=type_ids_dev,
            training_target_dev=training_target_dev,
            labelled_mask_dev=labelled_mask_dev,
            prefiltered_zero_windows=prefiltered_zero_windows,
            active_type_id=active_type_id,
            active_type_ids=active_type_ids,
            loss_objective=loss_objective,
            budget_ratios=budget_ratios,
            budget_loss_temperature=budget_loss_temperature,
            temporal_residual_budget_masks=temporal_residual_budget_masks,
            temporal_residual_union_mask=temporal_residual_union_mask,
            factorized_targets_dev=factorized_targets_dev,
            factorized_mask_dev=factorized_mask_dev,
            canonical_segment_ids_dev=canonical_segment_ids_dev,
            raw_window_count=raw_window_count,
            trained_window_count=trained_window_count,
            diag_windows=diag_windows,
            diag_every=diag_every,
            diag_fraction=diag_fraction,
            train_batch_size=train_batch_size,
            scaler=scaler,
            validation_plan=validation_plan,
            validation_workload_map=validation_workload_map,
            precomputed_validation_geometry_scores=precomputed_validation_geometry_scores,
            seed=seed,
            run_tag=run_tag,
        )
    )
    history = epoch_loop_result.history
    epochs_trained = epoch_loop_result.epochs_trained
    best_epoch = epoch_loop_result.best_epoch
    best_selection = epoch_loop_result.best_selection
    best_loss = epoch_loop_result.best_loss
    best_selection_score = epoch_loop_result.best_selection_score
    best_state_dict = epoch_loop_result.best_state_dict

    model = model.to("cpu")
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(
            f"  [{run_tag}] restored best diagnostic epoch {best_epoch}/{epochs_trained} "
            f"(selection={best_selection:+.3f}, loss={best_loss:.8f}, "
            f"val_selection_score={best_selection_score:.6f})",
            flush=True,
        )
    fit_diagnostics = build_final_training_fit_diagnostics(
        model=model,
        norm_points=norm_points,
        train_boundaries=train_boundaries,
        norm_queries=norm_queries,
        query_type_ids=workload.type_ids,
        model_config=model_config,
        device=device,
        amp_mode=amp_mode,
        train_batch_size=train_batch_size,
        model_type=model_type,
        training_target=training_target,
        training_labelled_mask=training_labelled_mask,
        workload_type_id=workload_type_id,
        training_target_basis=training_target_basis,
        factorized_targets=factorized_targets,
        factorized_mask=factorized_mask,
        all_points=all_points,
        prior_queries=prior_queries,
        canonical_segment_ids=canonical_segment_ids,
        seed=seed,
        run_tag=run_tag,
    )
    return TrainingOutputs(
        model=model,
        scaler=scaler,
        labels=labels,
        labelled_mask=labelled_mask,
        history=history,
        epochs_trained=epochs_trained,
        best_epoch=best_epoch,
        best_loss=best_loss,
        best_selection_score=best_selection_score,
        target_diagnostics=target_diagnostics,
        fit_diagnostics=fit_diagnostics,
        feature_context={
            "query_prior_field": query_prior_field,
            "query_prior_field_metadata": query_prior_field_metadata(query_prior_field),
        },
    )
