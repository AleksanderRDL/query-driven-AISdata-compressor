"""Epoch-loop orchestration for model training."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch.amp.grad_scaler import GradScaler

from config.run_config import ModelConfig
from learning.checkpoint_selection import (
    CheckpointCandidate,
    record_validation_stats,
    selection_from_stats,
    selection_score,
)
from learning.checkpoint_validation import validation_checkpoint_scores
from learning.fit_diagnostics import discriminative_sample, kendall_tau
from learning.losses import safe_quantile
from learning.model_setup import model_state_on_cpu
from learning.model_training_helpers import require_validation_inputs
from learning.model_training_validation import ValidationScoringPlan
from learning.optimization_epoch import train_one_epoch
from learning.supervised_windows import trajectory_batch_to_device
from learning.trajectory_batching import TrajectoryBatch, batch_windows
from runtime.torch_runtime import torch_autocast_context
from workloads.query_types import ID_TO_QUERY_NAME, NUM_QUERY_TYPES


@dataclass(frozen=True)
class TrainingEpochLoopPlan:
    model: torch.nn.Module
    windows: list[TrajectoryBatch]
    opt: torch.optim.Optimizer
    grad_scaler: GradScaler
    model_config: ModelConfig
    device: torch.device
    amp_mode: str
    norm_points_dev: torch.Tensor
    norm_queries_dev: torch.Tensor
    type_ids_dev: torch.Tensor
    training_target_dev: torch.Tensor
    labelled_mask_dev: torch.Tensor
    prefiltered_zero_windows: torch.Tensor
    active_type_id: int
    active_type_ids: list[int]
    loss_objective: str
    budget_ratios: tuple[float, ...]
    budget_loss_temperature: float
    temporal_residual_budget_masks: tuple[tuple[float, float, torch.Tensor], ...]
    temporal_residual_union_mask: torch.Tensor | None
    factorized_targets_dev: torch.Tensor | None
    factorized_mask_dev: torch.Tensor | None
    canonical_segment_ids_dev: torch.Tensor | None
    raw_window_count: int
    trained_window_count: int
    diag_windows: list[TrajectoryBatch]
    diag_every: int
    diag_fraction: float
    train_batch_size: int
    scaler: Any
    validation_plan: ValidationScoringPlan
    validation_workload_map: dict[str, float] | None
    precomputed_validation_geometry_scores: torch.Tensor | None
    seed: int
    run_tag: str


@dataclass(frozen=True)
class TrainingEpochLoopResult:
    history: list[dict[str, float]]
    epochs_trained: int
    best_epoch: int
    best_selection: float
    best_loss: float
    best_selection_score: float
    best_state_dict: dict[str, torch.Tensor] | None


@dataclass(frozen=True)
class EpochCheckpointSelection:
    selection: float | None
    smoothed_selection: float | None
    is_new_best_model: bool
    validation_round_had_selection: bool
    validation_round_improved: bool
    best_selection: float
    best_loss: float
    best_selection_score: float
    best_epoch: int
    best_state_dict: dict[str, torch.Tensor] | None


def _diagnostic_full_scores(
    *,
    model: torch.nn.Module,
    diag_windows: list[TrajectoryBatch],
    diag_fraction: float,
    diagnostic_sample_generator: torch.Generator,
    norm_points_dev: torch.Tensor,
    norm_queries_dev: torch.Tensor,
    type_ids_dev: torch.Tensor,
    model_config: ModelConfig,
    train_batch_size: int,
    device: torch.device,
    amp_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if diag_fraction < 1.0 and len(diag_windows) > 8:
        diagnostic_window_count = max(8, int(len(diag_windows) * diag_fraction))
        sample_indices = torch.randperm(
            len(diag_windows),
            generator=diagnostic_sample_generator,
        )[:diagnostic_window_count].tolist()
        diagnostic_windows = [diag_windows[i] for i in sample_indices]
    else:
        diagnostic_windows = diag_windows

    model.eval()
    with torch.no_grad():
        diagnostic_score_sum = norm_points_dev.new_zeros((norm_points_dev.shape[0],))
        diagnostic_score_count = norm_points_dev.new_zeros((norm_points_dev.shape[0],))
        diagnostic_batch_size = max(
            1, int(getattr(model_config, "inference_batch_size", train_batch_size))
        )
        for diagnostic_batch_cpu in batch_windows(diagnostic_windows, diagnostic_batch_size):
            diagnostic_batch = trajectory_batch_to_device(diagnostic_batch_cpu, device)
            with torch_autocast_context(device, amp_mode):
                window_scores = model(
                    points=diagnostic_batch.points,
                    queries=norm_queries_dev,
                    query_type_ids=type_ids_dev,
                    padding_mask=diagnostic_batch.padding_mask,
                )
            window_scores = window_scores.float()
            for batch_idx in range(window_scores.shape[0]):
                point_indices = diagnostic_batch.global_indices[batch_idx]
                valid_points = point_indices >= 0
                diagnostic_score_sum[point_indices[valid_points]] = (
                    diagnostic_score_sum[point_indices[valid_points]]
                    + window_scores[batch_idx, valid_points]
                )
                diagnostic_score_count[point_indices[valid_points]] = (
                    diagnostic_score_count[point_indices[valid_points]] + 1.0
                )
        covered_mask = diagnostic_score_count > 0
        diagnostic_score_count = diagnostic_score_count.clamp(min=1.0)
        full_scores = diagnostic_score_sum / diagnostic_score_count

    return full_scores, covered_mask


def _window_training_stats(
    *,
    positive_windows: torch.Tensor,
    skipped_zero_windows: torch.Tensor,
    ranking_pair_counts: torch.Tensor,
) -> dict[str, float]:
    stats: dict[str, float] = {}
    for type_idx in range(NUM_QUERY_TYPES):
        stats[f"positive_windows_t{type_idx}"] = float(positive_windows[type_idx].item())
        stats[f"skipped_zero_windows_t{type_idx}"] = float(skipped_zero_windows[type_idx].item())
        stats[f"ranking_pairs_t{type_idx}"] = float(ranking_pair_counts[type_idx].item())
    return stats


def _diagnostic_stats(
    *,
    epoch: int,
    epoch_loss: torch.Tensor,
    window_count: int,
    full_scores: torch.Tensor,
    covered_mask: torch.Tensor,
    positive_windows: torch.Tensor,
    skipped_zero_windows: torch.Tensor,
    ranking_pair_counts: torch.Tensor,
    active_type_id: int,
    labelled_mask_dev: torch.Tensor,
    training_target_dev: torch.Tensor,
    diagnostic_sample_generator: torch.Generator,
    seed: int,
) -> dict[str, float]:
    stats: dict[str, float] = {
        "epoch": float(epoch),
        "loss": float(epoch_loss.item() / max(1, window_count)),
        "pred_std": (
            float(full_scores[covered_mask].std().item())
            if bool(covered_mask.any().item())
            else 0.0
        ),
    }
    stats.update(
        _window_training_stats(
            positive_windows=positive_windows,
            skipped_zero_windows=skipped_zero_windows,
            ranking_pair_counts=ranking_pair_counts,
        )
    )
    for type_idx in range(NUM_QUERY_TYPES):
        stats[f"pred_p50_t{type_idx}"] = 0.0
        stats[f"pred_p90_t{type_idx}"] = 0.0
        stats[f"pred_p99_t{type_idx}"] = 0.0
        stats[f"positive_fraction_t{type_idx}"] = 0.0
        stats[f"label_p95_t{type_idx}"] = 0.0
        stats[f"kendall_tau_t{type_idx}"] = 0.0

    type_scores = full_scores
    stats[f"pred_p50_t{active_type_id}"] = float(safe_quantile(type_scores, 0.50).item())
    stats[f"pred_p90_t{active_type_id}"] = float(safe_quantile(type_scores, 0.90).item())
    stats[f"pred_p99_t{active_type_id}"] = float(safe_quantile(type_scores, 0.99).item())
    labelled_type = labelled_mask_dev
    positive_type = labelled_type & (training_target_dev > 0)
    labelled_count = max(1, int(labelled_type.sum().item()))
    stats[f"positive_fraction_t{active_type_id}"] = float(
        positive_type.sum().item() / labelled_count
    )
    if bool(positive_type.any().item()):
        stats[f"label_p95_t{active_type_id}"] = float(
            safe_quantile(training_target_dev[positive_type], 0.95).item()
        )
    eval_mask = labelled_mask_dev & covered_mask
    if bool(eval_mask.any().item()):
        diagnostic_sample_generator.manual_seed(int(seed) + 777)
        pred_sample, target_sample = discriminative_sample(
            type_scores[eval_mask].detach().cpu(),
            training_target_dev[eval_mask].detach().cpu(),
            n_each=100,
            generator=diagnostic_sample_generator,
        )
        stats[f"kendall_tau_t{active_type_id}"] = kendall_tau(pred_sample, target_sample)

    if stats["pred_std"] < 1e-3:
        stats["collapse_warning"] = 1.0
    return stats


def _loss_only_stats(
    *,
    epoch: int,
    epoch_loss: torch.Tensor,
    window_count: int,
    positive_windows: torch.Tensor,
    skipped_zero_windows: torch.Tensor,
    ranking_pair_counts: torch.Tensor,
) -> dict[str, float]:
    stats = {
        "epoch": float(epoch),
        "loss": float(epoch_loss.item() / max(1, window_count)),
    }
    stats.update(
        _window_training_stats(
            positive_windows=positive_windows,
            skipped_zero_windows=skipped_zero_windows,
            ranking_pair_counts=ranking_pair_counts,
        )
    )
    return stats


def _score_validation_checkpoint(
    *,
    model: torch.nn.Module,
    scaler: Any,
    validation_plan: ValidationScoringPlan,
    validation_workload_map: dict[str, float] | None,
    model_config: ModelConfig,
    device: torch.device,
    precomputed_validation_geometry_scores: torch.Tensor | None,
) -> tuple[float, dict[str, float], dict[str, float]]:
    validation_trajectories, validation_boundaries, validation_workload = (
        require_validation_inputs(
            validation_plan.validation_trajectories,
            validation_plan.validation_boundaries,
            validation_plan.validation_workload,
        )
    )
    return validation_checkpoint_scores(
        model=model,
        scaler=scaler,
        trajectories=validation_trajectories,
        boundaries=validation_boundaries,
        workload=validation_workload,
        workload_map=validation_workload_map or {},
        model_config=model_config,
        device=device,
        validation_points=validation_plan.validation_points_for_score,
        query_cache=validation_plan.validation_query_cache,
        range_geometry_scores=precomputed_validation_geometry_scores,
    )


def _record_direct_validation_score(
    *,
    stats: dict[str, float],
    model: torch.nn.Module,
    scaler: Any,
    validation_plan: ValidationScoringPlan,
    validation_workload_map: dict[str, float] | None,
    model_config: ModelConfig,
    device: torch.device,
    precomputed_validation_geometry_scores: torch.Tensor | None,
) -> float:
    score_t0 = time.perf_counter()
    validation_score, per_type_score, validation_metrics = _score_validation_checkpoint(
        model=model,
        scaler=scaler,
        validation_plan=validation_plan,
        validation_workload_map=validation_workload_map,
        model_config=model_config,
        device=device,
        precomputed_validation_geometry_scores=precomputed_validation_geometry_scores,
    )
    record_validation_stats(
        stats,
        validation_score=validation_score,
        per_type_score=per_type_score,
        validation_metrics=validation_metrics,
        validation_uniform_result=validation_plan.validation_uniform_result,
        validation_workload_map=validation_workload_map,
    )
    return time.perf_counter() - score_t0


def _collect_checkpoint_candidates(
    *,
    stats: dict[str, float],
    checkpoint_candidates: list[CheckpointCandidate],
    model: torch.nn.Module,
    scaler: Any,
    validation_plan: ValidationScoringPlan,
    validation_workload_map: dict[str, float] | None,
    model_config: ModelConfig,
    device: torch.device,
    precomputed_validation_geometry_scores: torch.Tensor | None,
    candidate_avg_tau: float,
    epoch: int,
    full_score_due: bool,
    use_checkpoint_candidate_pool: bool,
    checkpoint_candidate_pool_size: int,
) -> tuple[list[CheckpointCandidate], list[CheckpointCandidate], float]:
    evaluated_checkpoint_candidates: list[CheckpointCandidate] = []
    elapsed = 0.0

    stats["checkpoint_score_candidate"] = 1.0
    stats["checkpoint_candidate_cheap_score"] = selection_score(
        candidate_avg_tau,
        stats["pred_std"],
        stats["loss"],
    )
    stats["checkpoint_full_score_due"] = 1.0 if full_score_due else 0.0
    if not use_checkpoint_candidate_pool:
        return checkpoint_candidates, evaluated_checkpoint_candidates, elapsed

    checkpoint_candidates.append(
        CheckpointCandidate(
            epoch_number=epoch + 1,
            epoch_index=epoch,
            cheap_score=float(stats["checkpoint_candidate_cheap_score"]),
            loss=float(stats["loss"]),
            state_dict=model_state_on_cpu(model),
            stats=stats,
            avg_tau=candidate_avg_tau,
        )
    )
    checkpoint_candidates.sort(key=lambda candidate: candidate.cheap_score, reverse=True)
    checkpoint_candidates = checkpoint_candidates[:checkpoint_candidate_pool_size]
    if not (full_score_due and checkpoint_candidates):
        return checkpoint_candidates, evaluated_checkpoint_candidates, elapsed

    score_t0 = time.perf_counter()
    current_state_dict = model_state_on_cpu(model)
    for candidate in sorted(checkpoint_candidates, key=lambda item: item.epoch_number):
        candidate_t0 = time.perf_counter()
        model.load_state_dict(candidate.state_dict)
        validation_score, per_type_score, validation_metrics = _score_validation_checkpoint(
            model=model,
            scaler=scaler,
            validation_plan=validation_plan,
            validation_workload_map=validation_workload_map,
            model_config=model_config,
            device=device,
            precomputed_validation_geometry_scores=precomputed_validation_geometry_scores,
        )
        record_validation_stats(
            candidate.stats,
            validation_score=validation_score,
            per_type_score=per_type_score,
            validation_metrics=validation_metrics,
            validation_uniform_result=validation_plan.validation_uniform_result,
            validation_workload_map=validation_workload_map,
        )
        candidate.stats["checkpoint_candidate_evaluated"] = 1.0
        candidate.stats["checkpoint_full_score_round_epoch"] = float(epoch + 1)
        candidate.stats["checkpoint_validation_seconds"] = float(time.perf_counter() - candidate_t0)
        evaluated_checkpoint_candidates.append(candidate)
    model.load_state_dict(current_state_dict)
    elapsed += time.perf_counter() - score_t0
    return [], evaluated_checkpoint_candidates, elapsed


def _select_epoch_checkpoint(
    *,
    stats: dict[str, float],
    avg_tau: float,
    evaluated_checkpoint_candidates: list[CheckpointCandidate],
    selection_history: list[float],
    smoothing_window: int,
    selection_metric: str,
    validation_uniform_result: tuple[float, dict[str, float]] | None,
    validation_workload_map: dict[str, float] | None,
    model_config: ModelConfig,
    model: torch.nn.Module,
    best_selection: float,
    best_loss: float,
    best_selection_score: float,
    best_epoch: int,
    best_state_dict: dict[str, torch.Tensor] | None,
    epoch: int,
    epoch_label_width: int,
    effective_epochs: int,
    run_tag: str,
) -> EpochCheckpointSelection:
    selection: float | None = None
    smoothed_selection: float | None = None
    is_new_best_model = False
    validation_round_had_selection = False
    validation_round_improved = False

    if evaluated_checkpoint_candidates:
        for candidate in sorted(
            evaluated_checkpoint_candidates, key=lambda item: item.epoch_number
        ):
            candidate_selection = selection_from_stats(
                stats=candidate.stats,
                avg_tau=candidate.avg_tau,
                selection_metric=selection_metric,
                validation_uniform_result=validation_uniform_result,
                validation_workload_map=validation_workload_map,
                model_config=model_config,
            )
            if candidate_selection is None:
                continue
            validation_round_had_selection = True
            candidate.stats["selection_score"] = candidate_selection
            selection_history.append(float(candidate_selection))
            window = selection_history[-smoothing_window:]
            candidate_smoothed = float(sum(window) / len(window))
            candidate.stats["selection_score_smoothed"] = candidate_smoothed
            candidate_is_new_best = candidate_smoothed > best_selection + 1e-4 or (
                abs(candidate_smoothed - best_selection) <= 1e-4
                and candidate.loss < best_loss - 1e-8
            )
            if candidate_is_new_best:
                validation_round_improved = True
                best_selection = candidate_smoothed
                best_loss = candidate.loss
                best_selection_score = float(
                    candidate.stats.get("val_selection_score", best_selection_score)
                )
                best_epoch = candidate.epoch_number
                best_state_dict = candidate.state_dict
                candidate.stats["checkpoint_promoted"] = 1.0
            else:
                candidate.stats["checkpoint_promoted"] = 0.0
            if candidate.stats is not stats:
                status = "promoted" if candidate_is_new_best else "checked"
                print(
                    f"  [{run_tag}] checkpoint candidate epoch "
                    f"{candidate.epoch_number:0{epoch_label_width}d}/{effective_epochs}  "
                    f"cheap={candidate.cheap_score:+.3f}  "
                    f"select={candidate_selection:+.3f}  "
                    f"smoothed={candidate_smoothed:+.3f}  {status}",
                    flush=True,
                )
        if "selection_score" in stats:
            selection = float(stats["selection_score"])
            smoothed_selection = float(stats["selection_score_smoothed"])
            is_new_best_model = bool(stats.get("checkpoint_promoted", 0.0))
    else:
        selection = selection_from_stats(
            stats=stats,
            avg_tau=avg_tau,
            selection_metric=selection_metric,
            validation_uniform_result=validation_uniform_result,
            validation_workload_map=validation_workload_map,
            model_config=model_config,
        )
        if selection is not None:
            validation_round_had_selection = True
            stats["selection_score"] = selection
            selection_history.append(float(selection))
            window = selection_history[-smoothing_window:]
            smoothed_selection = float(sum(window) / len(window))
            stats["selection_score_smoothed"] = smoothed_selection
            is_new_best_model = smoothed_selection > best_selection + 1e-4 or (
                abs(smoothed_selection - best_selection) <= 1e-4
                and stats["loss"] < best_loss - 1e-8
            )
            validation_round_improved = is_new_best_model
            if is_new_best_model:
                best_selection = smoothed_selection
                best_loss = stats["loss"]
                best_selection_score = float(stats.get("val_selection_score", best_selection_score))
                best_epoch = epoch + 1
                best_state_dict = model_state_on_cpu(model)

    return EpochCheckpointSelection(
        selection=selection,
        smoothed_selection=smoothed_selection,
        is_new_best_model=is_new_best_model,
        validation_round_had_selection=validation_round_had_selection,
        validation_round_improved=validation_round_improved,
        best_selection=best_selection,
        best_loss=best_loss,
        best_selection_score=best_selection_score,
        best_epoch=best_epoch,
        best_state_dict=best_state_dict,
    )


def run_training_epochs(plan: TrainingEpochLoopPlan) -> TrainingEpochLoopResult:
    model = plan.model
    windows = plan.windows
    opt = plan.opt
    grad_scaler = plan.grad_scaler
    model_config = plan.model_config
    device = plan.device
    amp_mode = plan.amp_mode
    norm_points_dev = plan.norm_points_dev
    norm_queries_dev = plan.norm_queries_dev
    type_ids_dev = plan.type_ids_dev
    training_target_dev = plan.training_target_dev
    labelled_mask_dev = plan.labelled_mask_dev
    prefiltered_zero_windows = plan.prefiltered_zero_windows
    active_type_id = plan.active_type_id
    active_type_ids = plan.active_type_ids
    loss_objective = plan.loss_objective
    budget_ratios = plan.budget_ratios
    budget_loss_temperature = plan.budget_loss_temperature
    temporal_residual_budget_masks = plan.temporal_residual_budget_masks
    temporal_residual_union_mask = plan.temporal_residual_union_mask
    factorized_targets_dev = plan.factorized_targets_dev
    factorized_mask_dev = plan.factorized_mask_dev
    canonical_segment_ids_dev = plan.canonical_segment_ids_dev
    raw_window_count = plan.raw_window_count
    trained_window_count = plan.trained_window_count
    diag_windows = plan.diag_windows
    diag_every = plan.diag_every
    diag_fraction = plan.diag_fraction
    train_batch_size = plan.train_batch_size
    scaler = plan.scaler
    validation_plan = plan.validation_plan
    validation_score_every = validation_plan.validation_score_every
    has_validation_score = validation_plan.has_validation_score
    validation_uniform_result = validation_plan.validation_uniform_result
    selection_metric = validation_plan.selection_metric
    validation_workload_map = plan.validation_workload_map
    precomputed_validation_geometry_scores = plan.precomputed_validation_geometry_scores
    seed = plan.seed
    run_tag = plan.run_tag

    training_sample_generator = torch.Generator().manual_seed(int(seed) + 99)
    # Separate fixed-seed generator for diagnostics so the tau subsample
    # stays consistent across epochs and doesn't oscillate with training state.
    diagnostic_sample_generator = torch.Generator().manual_seed(int(seed) + 777)
    history: list[dict[str, float]] = []

    effective_epochs = max(1, int(model_config.epochs))
    patience = int(getattr(model_config, "early_stopping_patience", 0) or 0)
    smoothing_window = max(1, int(getattr(model_config, "checkpoint_smoothing_window", 1) or 1))
    checkpoint_full_score_every = max(
        1, int(getattr(model_config, "checkpoint_full_score_every", 1) or 1)
    )
    checkpoint_candidate_pool_size = max(
        1, int(getattr(model_config, "checkpoint_candidate_pool_size", 1) or 1)
    )
    checkpoint_candidates: list[CheckpointCandidate] = []
    selection_history: list[float] = []
    best_selection = float("-inf")
    best_loss = float("inf")
    best_selection_score = 0.0
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    epoch_label_width = len(str(effective_epochs))
    epochs_trained = 0
    for epoch in range(effective_epochs):
        epoch_t0 = time.perf_counter()
        epoch_result = train_one_epoch(
            model=model,
            windows=windows,
            opt=opt,
            grad_scaler=grad_scaler,
            model_config=model_config,
            device=device,
            amp_mode=amp_mode,
            norm_queries_dev=norm_queries_dev,
            type_ids_dev=type_ids_dev,
            training_target_dev=training_target_dev,
            labelled_mask_dev=labelled_mask_dev,
            prefiltered_zero_windows=prefiltered_zero_windows,
            active_type_id=active_type_id,
            loss_objective=loss_objective,
            budget_ratios=budget_ratios,
            budget_loss_temperature=budget_loss_temperature,
            temporal_residual_budget_masks=temporal_residual_budget_masks,
            temporal_residual_union_mask=temporal_residual_union_mask,
            training_sample_generator=training_sample_generator,
            factorized_targets_dev=factorized_targets_dev,
            factorized_mask_dev=factorized_mask_dev,
            canonical_segment_ids_dev=canonical_segment_ids_dev,
        )
        epoch_timing = {
            "forward_s": float(epoch_result.timing["forward_s"]),
            "loss_s": float(epoch_result.timing["loss_s"]),
            "backward_s": float(epoch_result.timing["backward_s"]),
            "diagnostic_s": 0.0,
            "validation_score_s": 0.0,
        }
        evaluated_checkpoint_candidates: list[CheckpointCandidate] = []
        epoch_loss = epoch_result.loss
        positive_windows = epoch_result.positive_windows
        skipped_zero_windows = epoch_result.skipped_zero_windows
        ranking_pair_counts = epoch_result.ranking_pair_counts

        # Diagnostic pass only on selected epochs (every `diag_every` epochs and
        # the final epoch).  Subsample windows by `diag_fraction` to further cut
        # cost: pred_std and tau are statistical aggregates and noise from a
        # ~20% sample is tiny compared to the training noise we're measuring.
        is_last_epoch = (epoch + 1) == effective_epochs
        is_diag_epoch = ((epoch + 1) % diag_every == 0) or is_last_epoch or epoch == 0
        if is_diag_epoch:
            diagnostic_t0 = time.perf_counter()
            full_scores, covered_mask = _diagnostic_full_scores(
                model=model,
                diag_windows=diag_windows,
                diag_fraction=diag_fraction,
                diagnostic_sample_generator=diagnostic_sample_generator,
                norm_points_dev=norm_points_dev,
                norm_queries_dev=norm_queries_dev,
                type_ids_dev=type_ids_dev,
                model_config=model_config,
                train_batch_size=train_batch_size,
                device=device,
                amp_mode=amp_mode,
            )
            stats = _diagnostic_stats(
                epoch=epoch,
                epoch_loss=epoch_loss,
                window_count=len(windows),
                full_scores=full_scores,
                covered_mask=covered_mask,
                positive_windows=positive_windows,
                skipped_zero_windows=skipped_zero_windows,
                ranking_pair_counts=ranking_pair_counts,
                active_type_id=active_type_id,
                labelled_mask_dev=labelled_mask_dev,
                training_target_dev=training_target_dev,
                diagnostic_sample_generator=diagnostic_sample_generator,
                seed=seed,
            )
            epoch_timing["diagnostic_s"] += time.perf_counter() - diagnostic_t0

            candidate_tau_vals = [stats[f"kendall_tau_t{t}"] for t in active_type_ids]
            candidate_avg_tau = sum(candidate_tau_vals) / max(1, len(candidate_tau_vals))
            validation_score_due = (
                validation_score_every <= 0
                or (epoch + 1) % validation_score_every == 0
                or is_last_epoch
                or epoch == 0
            )
            full_score_due = validation_score_due and (
                checkpoint_full_score_every <= 1
                or (epoch + 1) % checkpoint_full_score_every == 0
                or is_last_epoch
            )
            use_checkpoint_candidate_pool = (
                has_validation_score
                and validation_score_due
                and selection_metric in {"score", "uniform_gap"}
                and checkpoint_full_score_every > 1
            )
            should_run_validation_score = (
                has_validation_score
                and full_score_due
                and (selection_metric in {"score", "uniform_gap"} or validation_score_every > 0)
                and not use_checkpoint_candidate_pool
            )
            if should_run_validation_score:
                epoch_timing["validation_score_s"] += _record_direct_validation_score(
                    stats=stats,
                    model=model,
                    scaler=scaler,
                    validation_plan=validation_plan,
                    validation_workload_map=validation_workload_map,
                    model_config=model_config,
                    device=device,
                    precomputed_validation_geometry_scores=precomputed_validation_geometry_scores,
                )
            if (
                has_validation_score
                and validation_score_due
                and selection_metric in {"score", "uniform_gap"}
            ):
                (
                    checkpoint_candidates,
                    evaluated_checkpoint_candidates,
                    checkpoint_score_seconds,
                ) = _collect_checkpoint_candidates(
                    stats=stats,
                    checkpoint_candidates=checkpoint_candidates,
                    model=model,
                    scaler=scaler,
                    validation_plan=validation_plan,
                    validation_workload_map=validation_workload_map,
                    model_config=model_config,
                    device=device,
                    precomputed_validation_geometry_scores=precomputed_validation_geometry_scores,
                    candidate_avg_tau=candidate_avg_tau,
                    epoch=epoch,
                    full_score_due=full_score_due,
                    use_checkpoint_candidate_pool=use_checkpoint_candidate_pool,
                    checkpoint_candidate_pool_size=checkpoint_candidate_pool_size,
                )
                epoch_timing["validation_score_s"] += checkpoint_score_seconds
        else:
            # Skip diagnostics this epoch; log only loss.  Patience counters
            # are only updated on diagnostic epochs below.
            stats = _loss_only_stats(
                epoch=epoch,
                epoch_loss=epoch_loss,
                window_count=len(windows),
                positive_windows=positive_windows,
                skipped_zero_windows=skipped_zero_windows,
                ranking_pair_counts=ranking_pair_counts,
            )

        epoch_dt = time.perf_counter() - epoch_t0
        stats["epoch_seconds"] = float(epoch_dt)
        stats["epoch_forward_seconds"] = float(epoch_timing["forward_s"])
        stats["epoch_loss_seconds"] = float(epoch_timing["loss_s"])
        stats["epoch_backward_seconds"] = float(epoch_timing["backward_s"])
        stats["epoch_diagnostic_seconds"] = float(epoch_timing["diagnostic_s"])
        stats["epoch_validation_score_seconds"] = float(epoch_timing["validation_score_s"])
        stats["epoch_f1_seconds"] = stats["epoch_validation_score_seconds"]
        stats["raw_training_window_count"] = float(raw_window_count)
        stats["trained_training_window_count"] = float(trained_window_count)
        stats["filtered_zero_window_count"] = float(raw_window_count - trained_window_count)
        for type_idx in range(NUM_QUERY_TYPES):
            stats[f"filtered_zero_windows_t{type_idx}"] = float(
                prefiltered_zero_windows[type_idx].item()
            )
        history.append(stats)

        epochs_trained = epoch + 1

        if is_diag_epoch:
            tau_vals = [stats[f"kendall_tau_t{t}"] for t in active_type_ids]
            avg_tau = sum(tau_vals) / max(1, len(tau_vals))
            collapse = "  COLLAPSE" if stats.get("collapse_warning") else ""
            checkpoint_selection = _select_epoch_checkpoint(
                stats=stats,
                avg_tau=avg_tau,
                evaluated_checkpoint_candidates=evaluated_checkpoint_candidates,
                selection_history=selection_history,
                smoothing_window=smoothing_window,
                selection_metric=selection_metric,
                validation_uniform_result=validation_uniform_result,
                validation_workload_map=validation_workload_map,
                model_config=model_config,
                model=model,
                best_selection=best_selection,
                best_loss=best_loss,
                best_selection_score=best_selection_score,
                best_epoch=best_epoch,
                best_state_dict=best_state_dict,
                epoch=epoch,
                epoch_label_width=epoch_label_width,
                effective_epochs=effective_epochs,
                run_tag=run_tag,
            )
            selection = checkpoint_selection.selection
            smoothed_selection = checkpoint_selection.smoothed_selection
            is_new_best_model = checkpoint_selection.is_new_best_model
            validation_round_had_selection = checkpoint_selection.validation_round_had_selection
            validation_round_improved = checkpoint_selection.validation_round_improved
            best_selection = checkpoint_selection.best_selection
            best_loss = checkpoint_selection.best_loss
            best_selection_score = checkpoint_selection.best_selection_score
            best_epoch = checkpoint_selection.best_epoch
            best_state_dict = checkpoint_selection.best_state_dict
            markers = []
            if epoch > 0 and is_new_best_model:
                markers.append("*** NEW BEST MODEL ***")
            best_marker = ("  " + "  ".join(markers)) if markers else ""
            smoothed_label = (
                f"  smoothed_w{smoothing_window}={smoothed_selection:+.3f}"
                if smoothing_window > 1 and smoothed_selection is not None
                else ""
            )
            selection_text = f"{selection:+.3f}" if selection is not None else "skipped"
            print(
                f"  [{run_tag}] epoch {epoch + 1:0{epoch_label_width}d}/{effective_epochs}  "
                f"loss={stats['loss']:.8f}  avg_tau={avg_tau:+.3f}  "
                f"pred_std={stats['pred_std']:.6g}  select={selection_text}{smoothed_label}  "
                f"({epoch_dt:.2f}s){collapse}{best_marker}",
                flush=True,
            )
            if "val_selection_score" in stats:
                print(
                    f"    [{run_tag}] val_selection_score={stats['val_selection_score']:.6f}  "
                    f"range_point_f1={stats.get('val_range_point_f1', 0.0):.6f}  "
                    f"query_local_utility={stats.get('val_query_local_utility', 0.0):.6f}  "
                    f"answer_f1={stats.get('val_answer_f1', 0.0):.6f}  "
                    f"combined_f1={stats.get('val_combined_f1', 0.0):.6f}",
                    flush=True,
                )
            if "val_uniform_score" in stats:
                print(
                    f"    [{run_tag}] val_vs_uniform aggregate={stats['val_selection_uniform_gap']:+.6f}  "
                    f"type_deficit={stats['val_selection_type_deficit']:.6f}  "
                    f"range={stats.get('val_selection_score_gap_range', 0.0):+.6f}",
                    flush=True,
                )
            diag_parts = []
            for type_idx in active_type_ids:
                type_name = ID_TO_QUERY_NAME.get(type_idx, f"t{type_idx}")
                diag_parts.append(
                    f"{type_name}:pos={stats[f'positive_fraction_t{type_idx}']:.4f},"
                    f"p95={stats[f'label_p95_t{type_idx}']:.3f},"
                    f"pairs={int(stats[f'ranking_pairs_t{type_idx}'])},"
                    f"skip={int(stats[f'skipped_zero_windows_t{type_idx}'])},"
                    f"filtered={int(stats[f'filtered_zero_windows_t{type_idx}'])}"
                )
            if diag_parts:
                print(f"    [{run_tag}] label_diag  " + "  ".join(diag_parts), flush=True)
            print(
                f"    [{run_tag}] epoch_timing  "
                f"forward={stats['epoch_forward_seconds']:.2f}s  "
                f"loss={stats['epoch_loss_seconds']:.2f}s  "
                f"backward={stats['epoch_backward_seconds']:.2f}s  "
                f"diagnostic={stats['epoch_diagnostic_seconds']:.2f}s  "
                f"validation_score={stats['epoch_validation_score_seconds']:.2f}s  "
                f"filtered_zero_windows={int(stats['filtered_zero_window_count'])}",
                flush=True,
            )

            if patience > 0 and validation_round_had_selection:
                if is_new_best_model or validation_round_improved:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(
                            f"  [{run_tag}] early stopping at epoch {epoch + 1:0{epoch_label_width}d}: "
                            f"selection score did not improve over {patience} diag epochs "
                            f"(best_selection={best_selection:+.3f}, best_loss={best_loss:.8f})",
                            flush=True,
                        )
                        break
        else:
            print(
                f"  [{run_tag}] epoch {epoch + 1:0{epoch_label_width}d}/{effective_epochs}  "
                f"loss={stats['loss']:.8f}  (no-diag)  ({epoch_dt:.2f}s)",
                flush=True,
            )
            print(
                f"    [{run_tag}] epoch_timing  "
                f"forward={stats['epoch_forward_seconds']:.2f}s  "
                f"loss={stats['epoch_loss_seconds']:.2f}s  "
                f"backward={stats['epoch_backward_seconds']:.2f}s  "
                f"filtered_zero_windows={int(stats['filtered_zero_window_count'])}",
                flush=True,
            )
    return TrainingEpochLoopResult(
        history=history,
        epochs_trained=epochs_trained,
        best_epoch=best_epoch,
        best_selection=best_selection,
        best_loss=best_loss,
        best_selection_score=best_selection_score,
        best_state_dict=best_state_dict,
    )
