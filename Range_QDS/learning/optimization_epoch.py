"""One-epoch optimization helpers for trajectory-window learning."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as F

from config.run_config import (
    DEFAULT_QUERY_LOCAL_UTILITY_BEHAVIOR_RANK_LOSS_WEIGHT,
    ModelConfig,
)
from learning.losses import (
    _balanced_pointwise_loss_rows,
    _budget_stratified_recall_loss_rows,
    _budget_temporal_cdf_loss_rows,
    _budget_topk_recall_loss_rows,
    _budget_topk_temporal_residual_loss_rows,
    _pointwise_bce_loss_rows,
    _ranking_loss_for_type,
)
from learning.supervised_windows import trajectory_batch_to_device
from learning.targets.query_local_utility import QUERY_LOCAL_UTILITY_HEAD_NAMES
from learning.trajectory_batching import TrajectoryBatch
from runtime.torch_runtime import torch_autocast_context
from workloads.query_types import NUM_QUERY_TYPES


class _GradScalerLike(Protocol):
    """Minimal GradScaler surface used by the epoch optimizer."""

    def is_enabled(self) -> bool: ...

    def scale(self, outputs: torch.Tensor) -> Any: ...

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None: ...

    def step(self, optimizer: torch.optim.Optimizer) -> Any: ...

    def update(self, new_scale: float | torch.Tensor | None = None) -> None: ...


@dataclass
class TrainingEpochResult:
    """Aggregated optimization result for one training epoch."""

    loss: torch.Tensor
    positive_windows: torch.Tensor
    skipped_zero_windows: torch.Tensor
    ranking_pair_counts: torch.Tensor
    timing: dict[str, float]


@dataclass(frozen=True)
class SegmentBudgetHeadLossParts:
    """Decomposed segment-budget head losses for training and diagnostics."""

    total: torch.Tensor
    pooled_bce: torch.Tensor
    pairwise_rank: torch.Tensor
    pooled_bce_count: int
    pairwise_rank_count: int


@dataclass(frozen=True)
class FactorizedQueryLocalUtilityLossParts:
    """Decomposed factorized QueryLocalUtility auxiliary loss."""

    total: torch.Tensor
    point_bce: torch.Tensor
    segment_point_bce_contribution: torch.Tensor
    segment_level: SegmentBudgetHeadLossParts
    behavior_rank: torch.Tensor
    sparse_head_rank: torch.Tensor


def _factorized_query_local_utility_loss(
    *,
    head_logits: torch.Tensor,
    head_targets: torch.Tensor,
    head_mask: torch.Tensor,
    global_indices: torch.Tensor | None = None,
    segment_ids: torch.Tensor | None = None,
    segment_size: int = 32,
    segment_budget_head_weight: float = 0.10,
    segment_level_loss_weight: float = 0.25,
    behavior_rank_loss_weight: float = DEFAULT_QUERY_LOCAL_UTILITY_BEHAVIOR_RANK_LOSS_WEIGHT,
    sparse_head_rank_loss_weight: float = 0.0,
    sparse_head_bce_target_mode: str = "raw",
) -> torch.Tensor:
    """Return auxiliary multi-head QueryLocalUtility loss for the range model."""
    return _factorized_query_local_utility_loss_parts(
        head_logits=head_logits,
        head_targets=head_targets,
        head_mask=head_mask,
        global_indices=global_indices,
        segment_ids=segment_ids,
        segment_size=segment_size,
        segment_budget_head_weight=segment_budget_head_weight,
        segment_level_loss_weight=segment_level_loss_weight,
        behavior_rank_loss_weight=behavior_rank_loss_weight,
        sparse_head_rank_loss_weight=sparse_head_rank_loss_weight,
        sparse_head_bce_target_mode=sparse_head_bce_target_mode,
    ).total


def _factorized_query_local_utility_loss_parts(
    *,
    head_logits: torch.Tensor,
    head_targets: torch.Tensor,
    head_mask: torch.Tensor,
    global_indices: torch.Tensor | None = None,
    segment_ids: torch.Tensor | None = None,
    segment_size: int = 32,
    segment_budget_head_weight: float = 0.10,
    segment_level_loss_weight: float = 0.25,
    behavior_rank_loss_weight: float = DEFAULT_QUERY_LOCAL_UTILITY_BEHAVIOR_RANK_LOSS_WEIGHT,
    sparse_head_rank_loss_weight: float = 0.0,
    sparse_head_bce_target_mode: str = "raw",
) -> FactorizedQueryLocalUtilityLossParts:
    """Return factorized loss parts without changing the production objective."""
    if head_logits.shape != head_targets.shape or head_mask.shape != head_logits.shape:
        raise ValueError("factorized head logits, targets, and mask must have matching shape.")
    valid = head_mask.to(dtype=torch.bool)
    if not bool(valid.any().item()):
        zero = head_logits.new_tensor(0.0)
        return FactorizedQueryLocalUtilityLossParts(
            total=zero,
            point_bce=zero,
            segment_point_bce_contribution=zero,
            segment_level=SegmentBudgetHeadLossParts(
                total=zero,
                pooled_bce=zero,
                pairwise_rank=zero,
                pooled_bce_count=0,
                pairwise_rank_count=0,
            ),
            behavior_rank=zero,
            sparse_head_rank=zero,
        )
    # q_hit, boundary, and segment budget are probability-like.  Behavior and
    # replacement are smooth utilities but BCE keeps useful gradients near 0/1.
    bce_targets = _calibrated_sparse_head_bce_targets(
        head_targets=head_targets,
        head_mask=head_mask,
        mode=sparse_head_bce_target_mode,
    )
    per_element = F.binary_cross_entropy_with_logits(
        head_logits,
        bce_targets,
        reduction="none",
    )
    segment_weight = max(0.0, float(segment_budget_head_weight))
    default_weights = {
        "query_hit_probability": 0.30,
        "conditional_behavior_utility": 0.25,
        "boundary_event_utility": 0.15,
        "replacement_representative_value": 0.20,
        "segment_budget_target": segment_weight,
        "path_length_support_target": 0.05,
    }
    if int(head_logits.shape[-1]) == len(QUERY_LOCAL_UTILITY_HEAD_NAMES):
        weights = [default_weights.get(str(name), 0.05) for name in QUERY_LOCAL_UTILITY_HEAD_NAMES]
    else:
        weights = [1.0 for _idx in range(int(head_logits.shape[-1]))]
    head_weights = head_logits.new_tensor(weights).view(1, 1, -1)
    valid_float = valid.to(dtype=per_element.dtype)
    weighted = per_element * head_weights * valid_float
    denom = (head_weights * valid_float).sum().clamp(min=1.0)
    point_loss = weighted.sum() / denom
    segment_point_bce_contribution = head_logits.new_tensor(0.0)
    if int(head_logits.shape[-1]) > 4:
        segment_point_bce_contribution = weighted[..., 4].sum() / denom
    segment_loss = _segment_budget_head_segment_level_loss_parts(
        head_logits=head_logits,
        head_targets=head_targets,
        head_mask=head_mask,
        global_indices=global_indices,
        segment_ids=segment_ids,
        segment_size=segment_size,
    )
    behavior_rank_loss = _behavior_head_rank_loss(
        head_logits=head_logits,
        head_targets=head_targets,
        head_mask=head_mask,
    )
    sparse_rank_weight = max(0.0, float(sparse_head_rank_loss_weight))
    sparse_head_rank_loss = head_logits.new_tensor(0.0)
    if sparse_rank_weight > 0.0:
        sparse_head_rank_loss = _sparse_head_rank_loss(
            head_logits=head_logits,
            head_targets=head_targets,
            head_mask=head_mask,
        )
    total = (
        point_loss
        + max(0.0, float(segment_level_loss_weight)) * segment_loss.total
        + max(0.0, float(behavior_rank_loss_weight)) * behavior_rank_loss
        + sparse_rank_weight * sparse_head_rank_loss
    )
    return FactorizedQueryLocalUtilityLossParts(
        total=total,
        point_bce=point_loss,
        segment_point_bce_contribution=segment_point_bce_contribution,
        segment_level=segment_loss,
        behavior_rank=behavior_rank_loss,
        sparse_head_rank=sparse_head_rank_loss,
    )


factorized_query_local_utility_loss = _factorized_query_local_utility_loss


def _sparse_head_rank_loss(
    *,
    head_logits: torch.Tensor,
    head_targets: torch.Tensor,
    head_mask: torch.Tensor,
    head_names: tuple[str, ...] = ("query_hit_probability", "boundary_event_utility"),
    top_fraction: float = 0.05,
) -> torch.Tensor:
    """Return a rank loss for tiny-magnitude factorized heads.

    This normalizes target gaps by the row span, so sparse soft labels still
    provide ordering pressure when BCE is dominated by base-rate calibration.
    """
    if head_logits.shape != head_targets.shape or head_mask.shape != head_logits.shape:
        raise ValueError("factorized head logits, targets, and mask must have matching shape.")
    fraction = min(1.0, max(0.0, float(top_fraction)))
    if fraction <= 0.0:
        return head_logits.new_tensor(0.0)
    losses: list[torch.Tensor] = []
    schema = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES)
    for head_name in head_names:
        try:
            head_idx = schema.index(head_name)
        except ValueError:  # pragma: no cover - constant schema guard.
            continue
        if int(head_logits.shape[-1]) <= head_idx:
            continue
        logits = head_logits[..., head_idx].float()
        targets = head_targets[..., head_idx].float().clamp(0.0, 1.0)
        mask = head_mask[..., head_idx].to(dtype=torch.bool)
        for row in range(int(logits.shape[0])):
            valid_positions = torch.where(mask[row])[0]
            valid_count = int(valid_positions.numel())
            if valid_count < 2:
                continue
            local_targets = targets[row, valid_positions]
            target_span = local_targets.max() - local_targets.min()
            if float(target_span.item()) <= 0.0:
                continue
            local_logits = logits[row, valid_positions]
            top_count = max(1, math.ceil(fraction * valid_count))
            top_positions = torch.topk(local_targets, k=top_count, largest=True).indices
            top_targets = local_targets[top_positions]
            top_logits = local_logits[top_positions]
            target_gap = top_targets.unsqueeze(1) - local_targets.unsqueeze(0)
            pair_mask = target_gap > 0.0
            if not bool(pair_mask.any().item()):
                continue
            logit_gap = top_logits.unsqueeze(1) - local_logits.unsqueeze(0)
            normalized_gap = (target_gap[pair_mask] / target_span.clamp(min=1e-8)).clamp(0.0, 1.0)
            losses.append((F.softplus(-logit_gap[pair_mask]) * normalized_gap).mean())
    if not losses:
        return head_logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def _calibrated_sparse_head_bce_targets(
    *,
    head_targets: torch.Tensor,
    head_mask: torch.Tensor,
    mode: str = "raw",
    head_names: tuple[str, ...] = ("query_hit_probability", "boundary_event_utility"),
) -> torch.Tensor:
    """Return BCE targets, optionally rescaling sparse heads within each window."""
    out = head_targets.float().clamp(0.0, 1.0)
    normalized_mode = str(mode).lower()
    if normalized_mode in {"", "raw", "none"}:
        return out
    if normalized_mode != "window_max_normalized":
        raise ValueError("sparse_head_bce_target_mode must be 'raw' or 'window_max_normalized'.")
    out = out.clone()
    schema = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES)
    for head_name in head_names:
        try:
            head_idx = schema.index(head_name)
        except ValueError:  # pragma: no cover - constant schema guard.
            continue
        if int(out.shape[-1]) <= head_idx:
            continue
        targets = out[..., head_idx]
        mask = head_mask[..., head_idx].to(dtype=torch.bool)
        for row in range(int(targets.shape[0])):
            valid = mask[row]
            if not bool(valid.any().item()):
                continue
            local = targets[row, valid]
            max_value = local.max()
            if float(max_value.item()) <= 1e-12:
                continue
            targets[row, valid] = (local / max_value.clamp(min=1e-12)).clamp(0.0, 1.0)
    return out


def _behavior_head_rank_loss(
    *,
    head_logits: torch.Tensor,
    head_targets: torch.Tensor,
    head_mask: torch.Tensor,
    top_fraction: float = 0.05,
    min_target_gap: float = 0.05,
) -> torch.Tensor:
    """Return a listwise loss for ranking conditional behavior utility points."""
    if head_logits.shape != head_targets.shape or head_mask.shape != head_logits.shape:
        raise ValueError("factorized head logits, targets, and mask must have matching shape.")
    try:
        behavior_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("conditional_behavior_utility")
    except ValueError:  # pragma: no cover - constant schema guard.
        return head_logits.new_tensor(0.0)
    if int(head_logits.shape[-1]) <= behavior_idx:
        return head_logits.new_tensor(0.0)

    behavior_logits = head_logits[..., behavior_idx].float()
    behavior_targets = head_targets[..., behavior_idx].float().clamp(0.0, 1.0)
    behavior_mask = head_mask[..., behavior_idx].to(dtype=torch.bool)
    losses: list[torch.Tensor] = []
    fraction = min(1.0, max(0.0, float(top_fraction)))
    min_gap = max(0.0, float(min_target_gap))
    for row in range(int(behavior_logits.shape[0])):
        valid_positions = torch.where(behavior_mask[row])[0]
        valid_count = int(valid_positions.numel())
        if valid_count < 2:
            continue
        local_targets = behavior_targets[row, valid_positions]
        if float((local_targets.max() - local_targets.min()).item()) <= min_gap:
            continue
        local_logits = behavior_logits[row, valid_positions]
        top_count = max(1, math.ceil(fraction * valid_count))
        top_positions = torch.topk(local_targets, k=top_count, largest=True).indices
        top_targets = local_targets[top_positions]
        top_logits = local_logits[top_positions]
        target_gap = top_targets.unsqueeze(1) - local_targets.unsqueeze(0)
        pair_mask = target_gap > min_gap
        if not bool(pair_mask.any().item()):
            continue
        logit_gap = top_logits.unsqueeze(1) - local_logits.unsqueeze(0)
        weighted_loss = F.softplus(-logit_gap[pair_mask]) * target_gap[pair_mask]
        losses.append(weighted_loss.mean())
    if not losses:
        return head_logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def _segment_budget_head_segment_level_loss(
    *,
    head_logits: torch.Tensor,
    head_targets: torch.Tensor,
    head_mask: torch.Tensor,
    global_indices: torch.Tensor | None = None,
    segment_ids: torch.Tensor | None = None,
    segment_size: int = 32,
) -> torch.Tensor:
    """Return segment-level/listwise loss for the segment-budget head."""
    return _segment_budget_head_segment_level_loss_parts(
        head_logits=head_logits,
        head_targets=head_targets,
        head_mask=head_mask,
        global_indices=global_indices,
        segment_ids=segment_ids,
        segment_size=segment_size,
    ).total


def _segment_budget_head_segment_level_loss_parts(
    *,
    head_logits: torch.Tensor,
    head_targets: torch.Tensor,
    head_mask: torch.Tensor,
    global_indices: torch.Tensor | None = None,
    segment_ids: torch.Tensor | None = None,
    segment_size: int = 32,
) -> SegmentBudgetHeadLossParts:
    """Return exact segment-level loss parts used by the auxiliary objective."""
    if head_logits.shape != head_targets.shape or head_mask.shape != head_logits.shape:
        raise ValueError("factorized head logits, targets, and mask must have matching shape.")
    zero = head_logits.new_tensor(0.0)
    if int(head_logits.shape[-1]) <= 4:
        return SegmentBudgetHeadLossParts(
            total=zero,
            pooled_bce=zero,
            pairwise_rank=zero,
            pooled_bce_count=0,
            pairwise_rank_count=0,
        )
    size = max(1, int(segment_size))
    seg_logits = head_logits[..., 4].float()
    seg_targets = head_targets[..., 4].float().clamp(0.0, 1.0)
    seg_mask = head_mask[..., 4].to(dtype=torch.bool)
    if global_indices is not None:
        seg_mask = seg_mask & (global_indices >= 0)
    if segment_ids is not None and segment_ids.shape[:2] != seg_mask.shape:
        raise ValueError("segment_ids must match factorized head batch dimensions.")
    losses: list[torch.Tensor] = []
    bce_losses: list[torch.Tensor] = []
    pairwise_losses: list[torch.Tensor] = []
    for row in range(int(seg_logits.shape[0])):
        valid_positions = torch.where(seg_mask[row])[0]
        if int(valid_positions.numel()) <= 0:
            continue
        row_segment_logits: list[torch.Tensor] = []
        row_segment_targets: list[torch.Tensor] = []
        if segment_ids is not None:
            row_segment_ids = segment_ids[row].to(device=seg_logits.device, dtype=torch.long)
            valid_positions = valid_positions[row_segment_ids[valid_positions] >= 0]
            if int(valid_positions.numel()) <= 0:
                continue
            unique_segment_ids = torch.unique(row_segment_ids[valid_positions], sorted=True)
            for segment_id in unique_segment_ids.tolist():
                positions = valid_positions[row_segment_ids[valid_positions] == int(segment_id)]
                if int(positions.numel()) <= 0:
                    continue
                row_segment_logits.append(seg_logits[row, positions].mean())
                row_segment_targets.append(seg_targets[row, positions].mean())
        else:
            for start in range(0, int(valid_positions.numel()), size):
                positions = valid_positions[start : start + size]
                if int(positions.numel()) <= 0:
                    continue
                row_segment_logits.append(seg_logits[row, positions].mean())
                row_segment_targets.append(seg_targets[row, positions].mean())
        if not row_segment_logits:
            continue
        pooled_logits = torch.stack(row_segment_logits)
        pooled_targets = torch.stack(row_segment_targets).clamp(0.0, 1.0)
        bce_loss = F.binary_cross_entropy_with_logits(
            pooled_logits, pooled_targets, reduction="mean"
        )
        bce_losses.append(bce_loss)
        losses.append(bce_loss)
        if int(pooled_logits.numel()) >= 2:
            target_diff = pooled_targets.unsqueeze(1) - pooled_targets.unsqueeze(0)
            logit_diff = pooled_logits.unsqueeze(1) - pooled_logits.unsqueeze(0)
            pair_mask = target_diff.abs() > 0.05
            if bool(pair_mask.any().item()):
                direction = torch.sign(target_diff[pair_mask])
                pair_loss = F.softplus(-direction * logit_diff[pair_mask]).mean()
                pairwise_losses.append(pair_loss)
                losses.append(pair_loss)
    if not losses:
        return SegmentBudgetHeadLossParts(
            total=zero,
            pooled_bce=zero,
            pairwise_rank=zero,
            pooled_bce_count=0,
            pairwise_rank_count=0,
        )
    return SegmentBudgetHeadLossParts(
        total=torch.stack(losses).mean(),
        pooled_bce=torch.stack(bce_losses).mean() if bce_losses else zero,
        pairwise_rank=torch.stack(pairwise_losses).mean() if pairwise_losses else zero,
        pooled_bce_count=len(bce_losses),
        pairwise_rank_count=len(pairwise_losses),
    )


def _train_one_epoch(
    *,
    model: torch.nn.Module,
    windows: list[TrajectoryBatch],
    opt: torch.optim.Optimizer,
    grad_scaler: _GradScalerLike,
    model_config: ModelConfig,
    device: torch.device,
    amp_mode: str,
    norm_queries_dev: torch.Tensor,
    type_ids_dev: torch.Tensor,
    training_target_dev: torch.Tensor,
    labelled_mask_dev: torch.Tensor,
    prefiltered_zero_windows: torch.Tensor,
    active_type_id: int,
    loss_objective: str,
    budget_ratios: tuple[float, ...],
    budget_loss_temperature: float,
    temporal_residual_budget_masks: tuple[tuple[float, float, torch.Tensor], ...],
    temporal_residual_union_mask: torch.Tensor | None,
    training_sample_generator: torch.Generator,
    factorized_targets_dev: torch.Tensor | None = None,
    factorized_mask_dev: torch.Tensor | None = None,
    canonical_segment_ids_dev: torch.Tensor | None = None,
) -> TrainingEpochResult:
    """Run forward/loss/backward optimization over all training windows."""
    timing = {
        "forward_s": 0.0,
        "loss_s": 0.0,
        "backward_s": 0.0,
    }
    model.train()
    epoch_loss = torch.tensor(0.0, device=device)
    positive_windows = torch.zeros((NUM_QUERY_TYPES,), dtype=torch.long)
    skipped_zero_windows = prefiltered_zero_windows.clone()
    ranking_pair_counts = torch.zeros((NUM_QUERY_TYPES,), dtype=torch.long)

    for window_batch_cpu in windows:
        window_batch = trajectory_batch_to_device(window_batch_cpu, device)
        forward_t0 = time.perf_counter()
        with torch_autocast_context(device, amp_mode):
            forward_with_heads = getattr(model, "forward_with_heads", None)
            forward_with_heads_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None
            if factorized_targets_dev is not None and callable(forward_with_heads):
                forward_with_heads_fn = cast(
                    Callable[..., tuple[torch.Tensor, torch.Tensor]], forward_with_heads
                )
                pred_batch, head_logits_batch = forward_with_heads_fn(
                    window_batch.points,
                    padding_mask=window_batch.padding_mask,
                )
            else:
                pred_batch = model(
                    points=window_batch.points,
                    queries=norm_queries_dev,
                    query_type_ids=type_ids_dev,
                    padding_mask=window_batch.padding_mask,
                )
                head_logits_batch = None
        timing["forward_s"] += time.perf_counter() - forward_t0
        loss_t0 = time.perf_counter()
        pred_batch = pred_batch.float()
        loss: torch.Tensor | None = None
        batch_size = pred_batch.shape[0]
        batch_global_idx = window_batch.global_indices.to(device=device)
        valid_batch = batch_global_idx >= 0
        safe_global_idx = batch_global_idx.clamp(min=0)
        batch_labels = training_target_dev[safe_global_idx]
        batch_label_mask = labelled_mask_dev[safe_global_idx] & valid_batch
        aux_loss = pred_batch.new_tensor(0.0)
        aux_loss_weight = max(
            0.0, float(getattr(model_config, "query_local_utility_aux_loss_weight", 0.50))
        )
        batch_segment_ids = None
        if (
            head_logits_batch is not None
            and factorized_targets_dev is not None
            and factorized_mask_dev is not None
        ):
            batch_head_targets = factorized_targets_dev[safe_global_idx].float()
            batch_head_mask = factorized_mask_dev[safe_global_idx] & valid_batch.unsqueeze(-1)
            if canonical_segment_ids_dev is not None:
                batch_segment_ids = canonical_segment_ids_dev[safe_global_idx].to(
                    device=device, dtype=torch.long
                )
                batch_segment_ids = torch.where(
                    valid_batch,
                    batch_segment_ids,
                    torch.full_like(batch_segment_ids, -1),
                )
            aux_loss = _factorized_query_local_utility_loss(
                head_logits=head_logits_batch.float(),
                head_targets=batch_head_targets,
                head_mask=batch_head_mask,
                global_indices=batch_global_idx,
                segment_ids=batch_segment_ids,
                segment_budget_head_weight=float(
                    getattr(model_config, "query_local_utility_segment_budget_head_weight", 0.10)
                ),
                segment_level_loss_weight=float(
                    getattr(model_config, "query_local_utility_segment_level_loss_weight", 0.25)
                ),
                behavior_rank_loss_weight=float(
                    getattr(
                        model_config,
                        "query_local_utility_behavior_rank_loss_weight",
                        DEFAULT_QUERY_LOCAL_UTILITY_BEHAVIOR_RANK_LOSS_WEIGHT,
                    )
                ),
                sparse_head_rank_loss_weight=float(
                    getattr(model_config, "query_local_utility_sparse_head_rank_loss_weight", 0.0)
                ),
                sparse_head_bce_target_mode=str(
                    getattr(model_config, "query_local_utility_sparse_head_bce_target_mode", "raw")
                ),
            )
        positive_row_mask = (batch_label_mask & (batch_labels > 0)).any(dim=1)
        positive_windows[active_type_id] += int(positive_row_mask.sum().item())
        skipped_zero_windows[active_type_id] += int((~positive_row_mask).sum().item())
        pointwise_mask_batch = batch_label_mask
        if temporal_residual_union_mask is not None:
            base_for_batch = temporal_residual_union_mask[safe_global_idx] & valid_batch
            pointwise_mask_batch = batch_label_mask & (~base_for_batch)
        pointwise_loss_rows, _pointwise_active_rows = _balanced_pointwise_loss_rows(
            pred=pred_batch,
            target=batch_labels,
            valid_mask=pointwise_mask_batch,
            generator=training_sample_generator,
        )

        if loss_objective in {
            "budget_topk",
            "stratified_budget_topk",
        }:
            if temporal_residual_budget_masks:
                rank_loss_rows, _rank_active_rows = _budget_topk_temporal_residual_loss_rows(
                    pred=pred_batch,
                    target=batch_labels,
                    valid_mask=batch_label_mask,
                    global_idx=safe_global_idx,
                    temporal_base_masks=temporal_residual_budget_masks,
                    temperature=budget_loss_temperature,
                )
            elif loss_objective == "stratified_budget_topk":
                rank_loss_rows, _rank_active_rows = _budget_stratified_recall_loss_rows(
                    pred=pred_batch,
                    target=batch_labels,
                    valid_mask=batch_label_mask,
                    budget_ratios=budget_ratios,
                    temperature=budget_loss_temperature,
                    center_weight=float(
                        getattr(model_config, "mlqds_stratified_center_weight", 0.0)
                    ),
                )
            else:
                rank_loss_rows, _rank_active_rows = _budget_topk_recall_loss_rows(
                    pred=pred_batch,
                    target=batch_labels,
                    valid_mask=batch_label_mask,
                    budget_ratios=budget_ratios,
                    temperature=budget_loss_temperature,
                )

            if bool(positive_row_mask.any().item()):
                row_losses = (
                    rank_loss_rows + model_config.pointwise_loss_weight * pointwise_loss_rows
                )
                temporal_distribution_weight = float(
                    getattr(model_config, "temporal_distribution_loss_weight", 0.0) or 0.0
                )
                if temporal_distribution_weight > 0.0:
                    temporal_distribution_rows, _distribution_active_rows = (
                        _budget_temporal_cdf_loss_rows(
                            pred=pred_batch,
                            valid_mask=batch_label_mask,
                            budget_ratios=budget_ratios,
                            temperature=budget_loss_temperature,
                        )
                    )
                    row_losses = (
                        row_losses + temporal_distribution_weight * temporal_distribution_rows
                    )
                loss = (
                    row_losses[positive_row_mask].sum() / float(batch_size)
                    + aux_loss_weight * aux_loss
                    + model_config.l2_score_weight * (pred_batch**2).mean()
                )
        elif loss_objective == "pointwise_bce":
            pointwise_direct_rows, pointwise_direct_active_rows = _pointwise_bce_loss_rows(
                pred=pred_batch,
                target=batch_labels,
                valid_mask=batch_label_mask,
            )
            if bool(pointwise_direct_active_rows.any().item()):
                loss = (
                    pointwise_direct_rows[pointwise_direct_active_rows].sum() / float(batch_size)
                    + aux_loss_weight * aux_loss
                    + model_config.l2_score_weight * (pred_batch**2).mean()
                )
        else:
            loss_terms: list[torch.Tensor] = []
            for row_index in torch.where(positive_row_mask.detach().cpu())[0].tolist():
                row = int(row_index)
                window_global_idx = batch_global_idx[row]
                valid_window = window_global_idx >= 0
                valid_global_idx = window_global_idx[valid_window]
                valid_pred = pred_batch[row][valid_window]
                rank_loss, pair_count = _ranking_loss_for_type(
                    pred=valid_pred,
                    target=training_target_dev[valid_global_idx],
                    valid_mask=labelled_mask_dev[valid_global_idx],
                    pairs_per_type=model_config.ranking_pairs_per_type,
                    top_quantile=model_config.ranking_top_quantile,
                    margin=model_config.rank_margin,
                    generator=training_sample_generator,
                )
                ranking_pair_counts[active_type_id] += int(pair_count)
                loss_terms.append(
                    rank_loss + model_config.pointwise_loss_weight * pointwise_loss_rows[row]
                )
            if loss_terms:
                loss = (
                    torch.stack(loss_terms).sum() / float(batch_size)
                    + aux_loss_weight * aux_loss
                    + model_config.l2_score_weight * (pred_batch**2).mean()
                )
        timing["loss_s"] += time.perf_counter() - loss_t0

        if loss is not None:
            backward_t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss with amp_mode={amp_mode}: {float(loss.item())}"
                )
            clip_norm = float(getattr(model_config, "gradient_clip_norm", 0.0) or 0.0)
            if grad_scaler.is_enabled():
                grad_scaler.scale(loss).backward()
                if clip_norm > 0.0:
                    grad_scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                grad_scaler.step(opt)
                grad_scaler.update()
            else:
                loss.backward()
                if clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                opt.step()
            epoch_loss = epoch_loss + loss.detach()
            timing["backward_s"] += time.perf_counter() - backward_t0

    return TrainingEpochResult(
        loss=epoch_loss,
        positive_windows=positive_windows,
        skipped_zero_windows=skipped_zero_windows,
        ranking_pair_counts=ranking_pair_counts,
        timing=timing,
    )


train_one_epoch = _train_one_epoch
