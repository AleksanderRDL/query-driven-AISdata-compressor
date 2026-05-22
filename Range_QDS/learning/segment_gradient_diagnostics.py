"""Segment-head loss and gradient-path diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from config.run_config import ModelConfig
from learning.losses import (
    _balanced_pointwise_loss_rows,
    _budget_stratified_recall_loss_rows,
    _budget_temporal_cdf_loss_rows,
    _budget_topk_recall_loss_rows,
)
from learning.optimization_epoch import _factorized_query_local_utility_loss_parts
from learning.targets.query_local_utility import QUERY_LOCAL_UTILITY_HEAD_NAMES
from learning.trajectory_batching import TrajectoryBatch, batch_windows, build_trajectory_windows


@dataclass
class _ComponentAccumulator:
    loss_sum: float = 0.0
    loss_count: int = 0
    active_batch_count: int = 0
    gradient_l1_sum: float = 0.0
    gradient_l2_sq_sum: float = 0.0
    gradient_max_abs: float = 0.0
    gradient_count: int = 0

    def add(self, loss: torch.Tensor, gradient: torch.Tensor | None, mask: torch.Tensor) -> None:
        self.loss_sum += float(loss.detach().cpu().item())
        self.loss_count += 1
        if gradient is None:
            return
        selected = gradient.detach()[mask.detach()]
        selected = selected[torch.isfinite(selected)]
        if int(selected.numel()) <= 0:
            return
        abs_values = selected.abs().float()
        self.active_batch_count += 1
        self.gradient_l1_sum += float(abs_values.sum().cpu().item())
        self.gradient_l2_sq_sum += float((selected.float() ** 2).sum().cpu().item())
        self.gradient_max_abs = max(self.gradient_max_abs, float(abs_values.max().cpu().item()))
        self.gradient_count += int(selected.numel())

    def as_dict(self) -> dict[str, Any]:
        mean_loss = self.loss_sum / max(1, self.loss_count)
        gradient_l2 = math.sqrt(max(0.0, self.gradient_l2_sq_sum))
        return {
            "loss_mean": float(mean_loss),
            "loss_observation_count": int(self.loss_count),
            "gradient_available": self.gradient_count > 0,
            "gradient_observation_count": int(self.gradient_count),
            "gradient_active_batch_count": int(self.active_batch_count),
            "gradient_l1": float(self.gradient_l1_sum),
            "gradient_l2": float(gradient_l2),
            "gradient_mean_abs": float(self.gradient_l1_sum / max(1, self.gradient_count)),
            "gradient_max_abs": float(self.gradient_max_abs),
        }


@dataclass(frozen=True)
class _PreparedDiagnosticInputs:
    point_scores: torch.Tensor
    head_logits: torch.Tensor
    factorized_targets: torch.Tensor
    factorized_mask: torch.Tensor
    scalar_target: torch.Tensor
    scalar_mask: torch.Tensor
    boundaries: list[tuple[int, int]]
    canonical_segment_ids: torch.Tensor | None
    point_count: int
    objective: str


def _gradient_for(loss: torch.Tensor, leaf: torch.Tensor) -> torch.Tensor | None:
    if not bool(getattr(loss, "requires_grad", False)):
        return None
    return torch.autograd.grad(
        loss,
        leaf,
        retain_graph=True,
        allow_unused=True,
    )[0]


def _add_component(
    components: dict[str, _ComponentAccumulator],
    name: str,
    loss: torch.Tensor,
    leaf: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    components.setdefault(name, _ComponentAccumulator()).add(
        loss=loss,
        gradient=_gradient_for(loss, leaf),
        mask=mask,
    )


def _gather_window_values(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    safe_indices = indices.clamp(min=0)
    gathered = values[safe_indices]
    return torch.where(indices.unsqueeze(-1) >= 0, gathered, torch.zeros_like(gathered))


def _gather_window_vector(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    safe_indices = indices.clamp(min=0)
    gathered = values[safe_indices]
    return torch.where(indices >= 0, gathered, torch.zeros_like(gathered))


def _segment_target_stats(
    *,
    factorized_targets: torch.Tensor,
    factorized_mask: torch.Tensor,
    canonical_segment_ids: torch.Tensor | None,
) -> dict[str, Any]:
    try:
        segment_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("segment_budget_target")
    except ValueError:  # pragma: no cover - schema constant guard.
        return {"available": False, "reason": "segment_budget_head_missing"}
    if int(factorized_targets.shape[-1]) <= segment_idx:
        return {"available": False, "reason": "segment_budget_head_index_out_of_range"}
    valid = factorized_mask[:, segment_idx].detach().cpu().bool()
    targets = factorized_targets[:, segment_idx].detach().cpu().float().clamp(0.0, 1.0)
    valid_targets = targets[valid]
    if int(valid_targets.numel()) <= 0:
        return {"available": False, "reason": "no_valid_segment_targets"}
    row: dict[str, Any] = {
        "available": True,
        "valid_point_count": int(valid_targets.numel()),
        "positive_point_fraction": float((valid_targets > 0.0).float().mean().item()),
        "point_target_mean": float(valid_targets.mean().item()),
        "point_target_std": float(valid_targets.std(unbiased=False).item())
        if int(valid_targets.numel()) > 1
        else 0.0,
    }
    if canonical_segment_ids is None or int(canonical_segment_ids.numel()) != int(valid.numel()):
        row["canonical_segment_available"] = False
        return row
    segment_ids = canonical_segment_ids.detach().cpu().long()
    pooled_targets: list[torch.Tensor] = []
    for segment_id in torch.unique(segment_ids[valid & (segment_ids >= 0)], sorted=True).tolist():
        segment_mask = valid & (segment_ids == int(segment_id))
        if bool(segment_mask.any().item()):
            pooled_targets.append(targets[segment_mask].mean())
    if not pooled_targets:
        row["canonical_segment_available"] = False
        row["canonical_segment_reason"] = "no_valid_canonical_segments"
        return row
    pooled = torch.stack(pooled_targets).float()
    row.update(
        {
            "canonical_segment_available": True,
            "pooled_segment_count": int(pooled.numel()),
            "pooled_target_mean": float(pooled.mean().item()),
            "pooled_target_std": float(pooled.std(unbiased=False).item())
            if int(pooled.numel()) > 1
            else 0.0,
            "pooled_positive_fraction": float((pooled > 0.0).float().mean().item()),
            "pooled_gt_0_01_fraction": float((pooled > 0.01).float().mean().item()),
            "pooled_p95_minus_p05": float(
                (torch.quantile(pooled, 0.95) - torch.quantile(pooled, 0.05)).item()
            )
            if int(pooled.numel()) > 1
            else 0.0,
        }
    )
    return row


def _primary_budget_loss_components(
    *,
    pred_batch: torch.Tensor,
    batch_labels: torch.Tensor,
    batch_label_mask: torch.Tensor,
    positive_row_mask: torch.Tensor,
    model_config: ModelConfig,
    budget_ratios: tuple[float, ...],
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    if not bool(positive_row_mask.any().item()):
        return {}
    pointwise_loss_rows, _pointwise_active_rows = _balanced_pointwise_loss_rows(
        pred=pred_batch,
        target=batch_labels,
        valid_mask=batch_label_mask,
        generator=generator,
    )
    objective = str(getattr(model_config, "loss_objective", "budget_topk")).lower()
    if objective == "stratified_budget_topk":
        rank_loss_rows, _rank_active_rows = _budget_stratified_recall_loss_rows(
            pred=pred_batch,
            target=batch_labels,
            valid_mask=batch_label_mask,
            budget_ratios=budget_ratios,
            temperature=float(getattr(model_config, "budget_loss_temperature", 0.25)),
            center_weight=float(getattr(model_config, "mlqds_stratified_center_weight", 0.0)),
        )
    elif objective == "budget_topk":
        rank_loss_rows, _rank_active_rows = _budget_topk_recall_loss_rows(
            pred=pred_batch,
            target=batch_labels,
            valid_mask=batch_label_mask,
            budget_ratios=budget_ratios,
            temperature=float(getattr(model_config, "budget_loss_temperature", 0.25)),
        )
    else:
        return {}

    denominator = float(pred_batch.shape[0])
    rank_loss = rank_loss_rows[positive_row_mask].sum() / denominator
    pointwise_loss = (
        max(0.0, float(getattr(model_config, "pointwise_loss_weight", 0.0)))
        * pointwise_loss_rows[positive_row_mask].sum()
        / denominator
    )
    components = {
        "primary_budget_rank_loss": rank_loss,
        "primary_balanced_point_bce": pointwise_loss,
    }
    temporal_weight = max(0.0, float(getattr(model_config, "temporal_distribution_loss_weight", 0.0)))
    if temporal_weight > 0.0:
        temporal_rows, _temporal_active_rows = _budget_temporal_cdf_loss_rows(
            pred=pred_batch,
            valid_mask=batch_label_mask,
            budget_ratios=budget_ratios,
            temperature=float(getattr(model_config, "budget_loss_temperature", 0.25)),
        )
        components["primary_temporal_distribution_loss"] = (
            temporal_weight * temporal_rows[positive_row_mask].sum() / denominator
        )
    components["primary_budget_total"] = sum(components.values(), pred_batch.new_tensor(0.0))
    l2_weight = max(0.0, float(getattr(model_config, "l2_score_weight", 0.0)))
    if l2_weight > 0.0:
        components["primary_score_l2"] = l2_weight * (pred_batch**2).mean()
    return components


def _classify_gradient_path(components: dict[str, dict[str, Any]], pairwise_count: int) -> str:
    pairwise = components.get("aux_segment_pairwise_rank_actual_share", {})
    primary = components.get("primary_budget_total", {})
    point_bce = components.get("aux_factorized_point_bce", {})
    pair_l2 = float(pairwise.get("gradient_l2") or 0.0)
    primary_l2 = float(primary.get("gradient_l2") or 0.0)
    point_l2 = float(point_bce.get("gradient_l2") or 0.0)
    if pairwise_count <= 0:
        return "segment_pairwise_rank_path_inactive"
    if pair_l2 <= 0.0:
        return "segment_pairwise_rank_gradient_blocked_or_zero"
    if primary_l2 > 0.0 and pair_l2 / primary_l2 < 0.01:
        return "segment_pairwise_rank_gradient_dominated_by_primary_budget_loss"
    if point_l2 > 0.0 and pair_l2 / point_l2 < 0.05:
        return "segment_pairwise_rank_gradient_dominated_by_factorized_point_bce"
    return "segment_pairwise_rank_gradient_material"


def _prepare_diagnostic_inputs(
    *,
    point_scores: torch.Tensor | None,
    head_logits: torch.Tensor | None,
    factorized_targets: torch.Tensor | None,
    factorized_mask: torch.Tensor | None,
    scalar_target: torch.Tensor | None,
    scalar_mask: torch.Tensor | None,
    boundaries: list[tuple[int, int]] | None,
    canonical_segment_ids: torch.Tensor | None,
    model_config: ModelConfig,
) -> tuple[_PreparedDiagnosticInputs | None, dict[str, Any] | None]:
    if (
        point_scores is None
        or head_logits is None
        or factorized_targets is None
        or factorized_mask is None
        or scalar_target is None
        or scalar_mask is None
        or boundaries is None
    ):
        return None, {"available": False, "reason": "missing_inputs"}
    if head_logits.shape != factorized_targets.shape or factorized_mask.shape != head_logits.shape:
        return None, {"available": False, "reason": "factorized_shape_mismatch"}

    point_scores_cpu = point_scores.detach().cpu().float().flatten()
    scalar_target_cpu = scalar_target.detach().cpu().float().flatten()
    scalar_mask_cpu = scalar_mask.detach().cpu().bool().flatten()
    head_logits_cpu = head_logits.detach().cpu().float()
    factorized_targets_cpu = factorized_targets.detach().cpu().float()
    factorized_mask_cpu = factorized_mask.detach().cpu().bool()
    point_count = int(point_scores_cpu.numel())
    if (
        int(head_logits_cpu.shape[0]) != point_count
        or int(scalar_target_cpu.numel()) != point_count
        or int(scalar_mask_cpu.numel()) != point_count
    ):
        return None, {"available": False, "reason": "point_count_mismatch"}
    if int(head_logits_cpu.shape[-1]) != len(QUERY_LOCAL_UTILITY_HEAD_NAMES):
        return None, {"available": False, "reason": "head_count_mismatch"}
    objective = str(getattr(model_config, "loss_objective", "budget_topk")).lower()
    if objective not in {"budget_topk", "stratified_budget_topk"}:
        return None, {
            "available": False,
            "reason": "unsupported_primary_loss_objective",
            "loss_objective": objective,
        }

    canonical_segment_ids_cpu = (
        canonical_segment_ids.detach().cpu().long()
        if canonical_segment_ids is not None
        else None
    )
    return (
        _PreparedDiagnosticInputs(
            point_scores=point_scores_cpu,
            head_logits=head_logits_cpu,
            factorized_targets=factorized_targets_cpu,
            factorized_mask=factorized_mask_cpu,
            scalar_target=scalar_target_cpu,
            scalar_mask=scalar_mask_cpu,
            boundaries=boundaries,
            canonical_segment_ids=canonical_segment_ids_cpu,
            point_count=point_count,
            objective=objective,
        ),
        None,
    )


def _window_batches(
    *,
    point_count: int,
    boundaries: list[tuple[int, int]],
    model_config: ModelConfig,
    train_batch_size: int,
) -> list[TrajectoryBatch]:
    dummy_points = torch.zeros((point_count, 1), dtype=torch.float32)
    return batch_windows(
        build_trajectory_windows(
            points=dummy_points,
            boundaries=boundaries,
            window_length=int(getattr(model_config, "window_length", 512)),
            stride=int(getattr(model_config, "window_stride", 256)),
        ),
        max(1, int(train_batch_size)),
    )


def _configured_budget_ratios(model_config: ModelConfig) -> tuple[float, ...]:
    ratios = tuple(
        sorted(
            {
                float(value)
                for value in (getattr(model_config, "budget_loss_ratios", None) or [])
                if 0.0 < float(value) <= 1.0
            }
        )
    )
    if ratios:
        return ratios
    return (float(getattr(model_config, "compression_ratio", 0.05)),)


def _add_aux_segment_gradient_components(
    *,
    components: dict[str, _ComponentAccumulator],
    inputs: _PreparedDiagnosticInputs,
    indices: torch.Tensor,
    valid_batch: torch.Tensor,
    model_config: ModelConfig,
) -> tuple[int, int]:
    segment_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("segment_budget_target")
    batch_head_logits = _gather_window_values(inputs.head_logits, indices).detach().clone()
    batch_head_logits.requires_grad_(True)
    batch_head_targets = _gather_window_values(inputs.factorized_targets, indices)
    batch_head_mask = _gather_window_values(
        inputs.factorized_mask.to(dtype=torch.float32), indices
    ).bool() & valid_batch.unsqueeze(-1)
    batch_segment_ids = None
    if inputs.canonical_segment_ids is not None:
        batch_segment_ids = _gather_window_vector(inputs.canonical_segment_ids, indices).long()
        batch_segment_ids = torch.where(
            valid_batch,
            batch_segment_ids,
            torch.full_like(batch_segment_ids, -1),
        )
    parts = _factorized_query_local_utility_loss_parts(
        head_logits=batch_head_logits,
        head_targets=batch_head_targets,
        head_mask=batch_head_mask,
        global_indices=indices,
        segment_ids=batch_segment_ids,
        segment_budget_head_weight=float(
            getattr(model_config, "query_local_utility_segment_budget_head_weight", 0.10)
        ),
        segment_level_loss_weight=float(
            getattr(model_config, "query_local_utility_segment_level_loss_weight", 0.25)
        ),
        behavior_rank_loss_weight=float(
            getattr(model_config, "query_local_utility_behavior_rank_loss_weight", 0.25)
        ),
        sparse_head_rank_loss_weight=float(
            getattr(model_config, "query_local_utility_sparse_head_rank_loss_weight", 0.0)
        ),
        sparse_head_bce_target_mode=str(
            getattr(model_config, "query_local_utility_sparse_head_bce_target_mode", "raw")
        ),
    )
    segment_head_mask = torch.zeros_like(batch_head_mask, dtype=torch.bool)
    segment_head_mask[..., segment_idx] = batch_head_mask[..., segment_idx]
    aux_weight = max(
        0.0, float(getattr(model_config, "query_local_utility_aux_loss_weight", 0.50))
    )
    segment_level_weight = max(
        0.0, float(getattr(model_config, "query_local_utility_segment_level_loss_weight", 0.25))
    )
    segment_share_denominator = max(
        1,
        int(parts.segment_level.pooled_bce_count) + int(parts.segment_level.pairwise_rank_count),
    )
    pooled_bce_share = (
        parts.segment_level.pooled_bce
        * float(parts.segment_level.pooled_bce_count)
        / float(segment_share_denominator)
    )
    pairwise_share = (
        parts.segment_level.pairwise_rank
        * float(parts.segment_level.pairwise_rank_count)
        / float(segment_share_denominator)
    )
    component_losses = {
        "aux_factorized_point_bce": aux_weight * parts.point_bce,
        "aux_segment_point_bce_contribution": (
            aux_weight * parts.segment_point_bce_contribution
        ),
        "aux_pooled_segment_bce_actual_share": (
            aux_weight * segment_level_weight * pooled_bce_share
        ),
        "aux_segment_pairwise_rank_actual_share": (
            aux_weight * segment_level_weight * pairwise_share
        ),
        "aux_segment_level_total": aux_weight * segment_level_weight * parts.segment_level.total,
        "aux_total": aux_weight * parts.total,
    }
    for component_name, component_loss in component_losses.items():
        _add_component(
            components,
            component_name,
            component_loss,
            batch_head_logits,
            segment_head_mask,
        )
    return int(parts.segment_level.pooled_bce_count), int(parts.segment_level.pairwise_rank_count)


def _add_primary_budget_gradient_components(
    *,
    components: dict[str, _ComponentAccumulator],
    inputs: _PreparedDiagnosticInputs,
    indices: torch.Tensor,
    valid_batch: torch.Tensor,
    model_config: ModelConfig,
    budget_ratios: tuple[float, ...],
    gradient_generator: torch.Generator,
) -> int:
    safe_indices = indices.clamp(min=0)
    batch_scores = _gather_window_vector(inputs.point_scores, indices).detach().clone()
    batch_scores.requires_grad_(True)
    batch_labels = inputs.scalar_target[safe_indices]
    batch_label_mask = inputs.scalar_mask[safe_indices] & valid_batch
    positive_row_mask = (batch_label_mask & (batch_labels > 0.0)).any(dim=1)
    for component_name, component_loss in _primary_budget_loss_components(
        pred_batch=batch_scores,
        batch_labels=batch_labels,
        batch_label_mask=batch_label_mask,
        positive_row_mask=positive_row_mask,
        model_config=model_config,
        budget_ratios=budget_ratios,
        generator=gradient_generator,
    ).items():
        _add_component(
            components,
            component_name,
            component_loss,
            batch_scores,
            valid_batch,
        )
    return int(positive_row_mask.sum().item())


def _gradient_ratios(component_rows: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    pairwise = component_rows.get("aux_segment_pairwise_rank_actual_share", {})
    primary = component_rows.get("primary_budget_total", {})
    point_bce = component_rows.get("aux_factorized_point_bce", {})
    pair_l2 = float(pairwise.get("gradient_l2") or 0.0)
    primary_l2 = float(primary.get("gradient_l2") or 0.0)
    point_l2 = float(point_bce.get("gradient_l2") or 0.0)
    return {
        "pairwise_rank_to_primary_budget_total_l2": (
            None if primary_l2 <= 0.0 else float(pair_l2 / primary_l2)
        ),
        "pairwise_rank_to_factorized_point_bce_l2": (
            None if point_l2 <= 0.0 else float(pair_l2 / point_l2)
        ),
    }


def segment_rank_loss_gradient_path_diagnostics(
    *,
    point_scores: torch.Tensor | None,
    head_logits: torch.Tensor | None,
    factorized_targets: torch.Tensor | None,
    factorized_mask: torch.Tensor | None,
    scalar_target: torch.Tensor | None,
    scalar_mask: torch.Tensor | None,
    boundaries: list[tuple[int, int]] | None,
    model_config: ModelConfig,
    canonical_segment_ids: torch.Tensor | None = None,
    train_batch_size: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure segment-rank loss scale and output-gradient contribution."""
    inputs, error = _prepare_diagnostic_inputs(
        point_scores=point_scores,
        head_logits=head_logits,
        factorized_targets=factorized_targets,
        factorized_mask=factorized_mask,
        scalar_target=scalar_target,
        scalar_mask=scalar_mask,
        boundaries=boundaries,
        canonical_segment_ids=canonical_segment_ids,
        model_config=model_config,
    )
    if error is not None:
        return error
    if inputs is None:  # pragma: no cover - defensive type guard.
        return {"available": False, "reason": "missing_inputs"}

    windows = _window_batches(
        point_count=inputs.point_count,
        boundaries=inputs.boundaries,
        model_config=model_config,
        train_batch_size=train_batch_size,
    )
    budget_ratios = _configured_budget_ratios(model_config)

    components: dict[str, _ComponentAccumulator] = {}
    segment_bce_count = 0
    segment_pairwise_count = 0
    positive_window_count = 0
    gradient_generator = torch.Generator().manual_seed(int(seed) + 9137)

    for window_batch in windows:
        indices = window_batch.global_indices.detach().cpu().long()
        valid_batch = indices >= 0
        bce_count, pairwise_count = _add_aux_segment_gradient_components(
            components=components,
            inputs=inputs,
            indices=indices,
            valid_batch=valid_batch,
            model_config=model_config,
        )
        segment_bce_count += bce_count
        segment_pairwise_count += pairwise_count
        positive_window_count += _add_primary_budget_gradient_components(
            components=components,
            inputs=inputs,
            indices=indices,
            valid_batch=valid_batch,
            model_config=model_config,
            budget_ratios=budget_ratios,
            gradient_generator=gradient_generator,
        )

    component_rows = {name: accumulator.as_dict() for name, accumulator in components.items()}
    return {
        "available": True,
        "diagnostic_only": True,
        "evidence_level": "instrumentation_only_training_window_loss_gradient_probe",
        "classification": _classify_gradient_path(component_rows, segment_pairwise_count),
        "loss_objective": inputs.objective,
        "window_count": int(sum(int(batch.global_indices.shape[0]) for batch in windows)),
        "batch_count": len(windows),
        "positive_window_count": int(positive_window_count),
        "budget_loss_ratios": list(budget_ratios),
        "aux_loss_weight": float(
            getattr(model_config, "query_local_utility_aux_loss_weight", 0.50)
        ),
        "segment_level_loss_weight": float(
            getattr(model_config, "query_local_utility_segment_level_loss_weight", 0.25)
        ),
        "segment_budget_head_weight": float(
            getattr(model_config, "query_local_utility_segment_budget_head_weight", 0.10)
        ),
        "segment_loss_observation_counts": {
            "pooled_bce": int(segment_bce_count),
            "pairwise_rank": int(segment_pairwise_count),
        },
        "segment_target_stats": _segment_target_stats(
            factorized_targets=inputs.factorized_targets,
            factorized_mask=inputs.factorized_mask,
            canonical_segment_ids=inputs.canonical_segment_ids,
        ),
        "gradient_ratios": _gradient_ratios(component_rows),
        "components": component_rows,
    }
