"""Prior-to-head transfer diagnostics for factorized heads."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch

from learning.factorized_prior_stage_diagnostics import _stage_tensor_sensitivity
from learning.factorized_prior_transfer_core import (
    _classify_head_transfer,
    _classify_prior_channel_output_alignment,
    _head_final_linear_weight,
    _head_mlp_transfer,
    _loss_gradient_alignment_summary,
    _prior_output_layer_alignment_diagnostics,
    _ratio_from_summaries,
    _summarize_prior_channel_direction_decomposition,
)
from learning.optimization_epoch import _factorized_query_local_utility_loss
from learning.query_prior_fields import QUERY_PRIOR_FIELD_NAMES
from learning.targets.query_local_utility import QUERY_LOCAL_UTILITY_HEAD_NAMES
from learning.targets.query_local_utility_family import (
    FAMILY_TRAINABILITY_GROUP_KEYS,
    _range_query_family_evidence,
)
from learning.trajectory_batching import batch_windows, build_trajectory_windows


def _prior_to_head_transfer_sensitivity_diagnostics(
    *,
    model: torch.nn.Module | None,
    norm_points: torch.Tensor,
    factorized_targets: torch.Tensor,
    factorized_mask: torch.Tensor,
    boundaries: list[tuple[int, int]] | None,
    window_length: int,
    window_stride: int,
    batch_size: int,
    prior_dim: int,
    raw_points: torch.Tensor | None = None,
    typed_queries: list[dict[str, Any]] | None = None,
    segment_budget_head_weight: float = 0.10,
    segment_level_loss_weight: float = 0.25,
    behavior_rank_loss_weight: float = 0.25,
    sparse_head_rank_loss_weight: float = 0.0,
    sparse_head_bce_target_mode: str = "raw",
) -> dict[str, Any]:
    """Diagnose how prior-sensitive shared directions pass through each head MLP."""
    if model is None:
        return {"available": False, "reason": "missing_model"}
    if boundaries is None:
        return {"available": False, "reason": "missing_boundaries"}
    if norm_points.ndim != 2 or int(norm_points.shape[1]) < int(prior_dim):
        return {"available": False, "reason": "point_feature_shape_missing_prior_channels"}
    point_encoder = getattr(model, "point_encoder", None)
    prior_feature_encoder = getattr(model, "prior_feature_encoder", None)
    local_context_encoder = getattr(model, "local_context_encoder", None)
    segment_context = getattr(model, "segment_context", None)
    shared_prior_encoder = getattr(model, "prior_encoder", None)
    heads = getattr(model, "heads", None)
    prior_features_fn = getattr(model, "_prior_features", None)
    positional_encoding_fn = getattr(model, "_positional_encoding", None)
    if (
        not callable(point_encoder)
        or not callable(prior_feature_encoder)
        or not callable(segment_context)
        or not callable(shared_prior_encoder)
        or not callable(prior_features_fn)
        or not isinstance(heads, torch.nn.ModuleDict)
    ):
        return {"available": False, "reason": "missing_stage_modules"}
    point_encoder_fn = cast(Callable[[torch.Tensor], torch.Tensor], point_encoder)
    prior_feature_encoder_fn = cast(Callable[[torch.Tensor], torch.Tensor], prior_feature_encoder)
    segment_context_fn = cast(Callable[[torch.Tensor], torch.Tensor], segment_context)
    shared_prior_encoder_fn = cast(Callable[[torch.Tensor], torch.Tensor], shared_prior_encoder)
    prior_features_callable = cast(Callable[[torch.Tensor], torch.Tensor], prior_features_fn)
    positional_callable = (
        cast(Callable[[int, torch.device, torch.dtype], torch.Tensor], positional_encoding_fn)
        if callable(positional_encoding_fn)
        else None
    )
    target_cpu = factorized_targets.detach().cpu().float().clamp(0.0, 1.0)
    mask_cpu = factorized_mask.detach().cpu().bool()
    if target_cpu.shape != mask_cpu.shape:
        return {"available": False, "reason": "target_mask_shape_mismatch"}
    family_slice_masks: dict[str, dict[str, torch.Tensor]] = {}
    if raw_points is not None and typed_queries is not None:
        raw_points_cpu = raw_points.detach().cpu().float()
        if int(raw_points_cpu.shape[0]) == int(norm_points.shape[0]):
            range_queries = [
                query for query in typed_queries if str(query.get("type", "")).lower() == "range"
            ]
            if range_queries:
                family_evidence = _range_query_family_evidence(
                    points=raw_points_cpu,
                    boundaries=boundaries,
                    range_queries=range_queries,
                    group_keys=FAMILY_TRAINABILITY_GROUP_KEYS,
                )
                for group_key, family_rows in family_evidence.items():
                    group_masks: dict[str, torch.Tensor] = {}
                    for family_name, row in family_rows.items():
                        query_hit_probability = row.get("query_hit_probability")
                        if isinstance(query_hit_probability, torch.Tensor):
                            group_masks[str(family_name)] = (
                                query_hit_probability.detach().cpu().float() > 0.0
                            )
                    if group_masks:
                        family_slice_masks[str(group_key)] = group_masks

    zero_points = norm_points.detach().clone()
    zero_points[:, -int(prior_dim) :] = 0.0
    windows = batch_windows(
        build_trajectory_windows(
            points=norm_points.detach().cpu().float(),
            boundaries=boundaries,
            window_length=int(window_length),
            stride=int(window_stride),
        ),
        max(1, int(batch_size)),
    )
    zero_windows = batch_windows(
        build_trajectory_windows(
            points=zero_points.detach().cpu().float(),
            boundaries=boundaries,
            window_length=int(window_length),
            stride=int(window_stride),
        ),
        max(1, int(batch_size)),
    )
    if len(windows) != len(zero_windows):
        return {"available": False, "reason": "window_count_mismatch"}
    device = next(model.parameters(), torch.empty(0)).device
    per_head_parts: dict[str, dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]]] = {
        str(name): {
            "shared_embedding": ([], []),
            "first_linear": ([], []),
            "hidden_activation": ([], []),
            "logit": ([], []),
            "probability": ([], []),
        }
        for name in QUERY_LOCAL_UTILITY_HEAD_NAMES
        if str(name) in heads
    }
    prior_channel_names = [
        str(QUERY_PRIOR_FIELD_NAMES[idx])
        if idx < len(QUERY_PRIOR_FIELD_NAMES)
        else f"prior_channel_{idx}"
        for idx in range(int(prior_dim))
    ]
    per_channel_head_parts: dict[
        str,
        dict[str, dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]]],
    ] = {
        channel_name: {
            head_name: {
                "hidden_activation": ([], []),
                "logit": ([], []),
                "probability": ([], []),
            }
            for head_name in per_head_parts
        }
        for channel_name in prior_channel_names
    }
    head_index_by_name = {
        str(name): int(idx) for idx, name in enumerate(QUERY_LOCAL_UTILITY_HEAD_NAMES)
    }
    per_head_target_parts: dict[str, list[torch.Tensor]] = {
        head_name: [] for head_name in per_head_parts
    }
    per_head_mask_parts: dict[str, list[torch.Tensor]] = {
        head_name: [] for head_name in per_head_parts
    }
    per_head_loss_gradient_parts: dict[str, dict[str, list[torch.Tensor]]] = {
        head_name: {
            "descent_alignment": [],
            "gradient": [],
            "logit_delta": [],
        }
        for head_name in per_head_parts
    }
    window_slice_labels = {
        0: "window_start",
        1: "window_middle",
        2: "window_end",
    }
    per_head_slice_mask_parts: dict[str, dict[str, dict[str, list[torch.Tensor]]]] = {
        head_name: {
            "window_slice": {label: [] for label in window_slice_labels.values()},
            **{
                group_name: {family_name: [] for family_name in family_masks}
                for group_name, family_masks in family_slice_masks.items()
            },
        }
        for head_name in per_head_parts
    }
    head_final_weights: dict[str, torch.Tensor] = {
        head_name: weight
        for head_name in per_head_parts
        if (weight := _head_final_linear_weight(heads[head_name])) is not None
    }
    head_weight_stats: dict[str, dict[str, float]] = {}
    loss_config: dict[str, Any] = {
        "segment_budget_head_weight": float(segment_budget_head_weight),
        "segment_level_loss_weight": float(segment_level_loss_weight),
        "behavior_rank_loss_weight": float(behavior_rank_loss_weight),
        "sparse_head_rank_loss_weight": float(sparse_head_rank_loss_weight),
        "sparse_head_bce_target_mode": str(sparse_head_bce_target_mode),
        "segment_id_scope": "window_order_fixed_chunks",
    }
    original_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            for primary_window, zero_window in zip(windows, zero_windows, strict=True):
                primary_points = primary_window.points.to(device=device)
                ablation_points = zero_window.points.to(device=device)
                padding_mask = primary_window.padding_mask.to(device=device)
                valid = ~padding_mask
                if not bool(valid.any().item()):
                    continue
                valid_cpu = valid.detach().cpu()
                global_indices_cpu = primary_window.global_indices.detach().cpu().long()
                valid_global = global_indices_cpu[valid_cpu]
                valid_global = valid_global.detach().cpu().long().reshape(-1)
                global_indices_usable = bool(
                    int(valid_global.numel())
                    and bool((valid_global >= 0).all().item())
                    and bool((valid_global < int(target_cpu.shape[0])).all().item())
                )
                valid_counts = valid_cpu.sum(dim=1).clamp(min=1)
                valid_ranks = torch.cumsum(valid_cpu.to(dtype=torch.long), dim=1) - 1
                valid_fractions = (
                    valid_ranks.to(dtype=torch.float32) + 0.5
                ) / valid_counts.unsqueeze(1).to(dtype=torch.float32)
                window_bucket = torch.where(
                    valid_fractions < (1.0 / 3.0),
                    torch.zeros_like(valid_ranks),
                    torch.where(
                        valid_fractions < (2.0 / 3.0),
                        torch.ones_like(valid_ranks),
                        torch.full_like(valid_ranks, 2),
                    ),
                )
                valid_window_bucket = window_bucket[valid_cpu].detach().cpu().long().reshape(-1)

                def shared_embedding(
                    points: torch.Tensor,
                    local_padding_mask: torch.Tensor,
                ) -> torch.Tensor:
                    point_encoded = point_encoder_fn(points)
                    prior_features = prior_features_callable(points)
                    prior_encoded = prior_feature_encoder_fn(prior_features)
                    scale_tensor = getattr(model, "prior_feature_scale", None)
                    scale = (
                        scale_tensor.to(device=device, dtype=prior_encoded.dtype)
                        if isinstance(scale_tensor, torch.Tensor)
                        else prior_encoded.new_tensor(1.0)
                    )
                    h = point_encoded + scale * prior_encoded
                    if local_context_encoder is not None:
                        local_input = h
                        if positional_callable is not None:
                            local_input = local_input + positional_callable(
                                local_input.shape[1],
                                local_input.device,
                                local_input.dtype,
                            ).unsqueeze(0)
                        h = local_context_encoder(
                            local_input,
                            src_key_padding_mask=local_padding_mask,
                        )
                    segment = segment_context_fn(h.transpose(1, 2)).transpose(1, 2)
                    return shared_prior_encoder_fn(h + segment)

                primary_shared = shared_embedding(primary_points, padding_mask)
                ablation_shared = shared_embedding(ablation_points, padding_mask)
                primary_logits_by_head: dict[str, torch.Tensor] = {}
                ablation_logits_by_head: dict[str, torch.Tensor] = {}
                primary_transfer_by_head: dict[
                    str,
                    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                ] = {}
                for head_name, stage_parts in per_head_parts.items():
                    transfer_primary = _head_mlp_transfer(heads[head_name], primary_shared)
                    transfer_ablation = _head_mlp_transfer(heads[head_name], ablation_shared)
                    if transfer_primary is None or transfer_ablation is None:
                        continue
                    primary_first, primary_hidden, primary_logit, weights = transfer_primary
                    ablation_first, ablation_hidden, ablation_logit, _ = transfer_ablation
                    primary_logits_by_head[head_name] = primary_logit
                    ablation_logits_by_head[head_name] = ablation_logit
                    primary_transfer_by_head[head_name] = (
                        primary_hidden,
                        primary_logit,
                        torch.sigmoid(primary_logit),
                    )
                    head_weight_stats[head_name] = weights
                    pairs = {
                        "shared_embedding": (primary_shared, ablation_shared),
                        "first_linear": (primary_first, ablation_first),
                        "hidden_activation": (primary_hidden, ablation_hidden),
                        "logit": (primary_logit, ablation_logit),
                        "probability": (
                            torch.sigmoid(primary_logit),
                            torch.sigmoid(ablation_logit),
                        ),
                    }
                    for stage_name, (primary_tensor, ablation_tensor) in pairs.items():
                        primary_parts, ablation_parts = stage_parts[stage_name]
                        primary_parts.append(primary_tensor[valid])
                        ablation_parts.append(ablation_tensor[valid])
                    head_idx = head_index_by_name.get(head_name)
                    if (
                        global_indices_usable
                        and head_idx is not None
                        and head_idx < int(target_cpu.shape[1])
                        and head_idx < int(mask_cpu.shape[1])
                    ):
                        per_head_target_parts[head_name].append(target_cpu[valid_global, head_idx])
                        per_head_mask_parts[head_name].append(mask_cpu[valid_global, head_idx])
                        slice_parts = per_head_slice_mask_parts[head_name]
                        for bucket_id, label in window_slice_labels.items():
                            slice_parts["window_slice"][label].append(
                                valid_window_bucket == int(bucket_id)
                            )
                        for group_name, family_masks in family_slice_masks.items():
                            for family_name, family_mask in family_masks.items():
                                slice_parts[group_name][family_name].append(
                                    family_mask[valid_global]
                                )
                ordered_head_names = [
                    str(name)
                    for name in QUERY_LOCAL_UTILITY_HEAD_NAMES
                    if str(name) in primary_logits_by_head and str(name) in ablation_logits_by_head
                ]
                if global_indices_usable and len(ordered_head_names) == len(
                    QUERY_LOCAL_UTILITY_HEAD_NAMES
                ):
                    window_targets = torch.zeros(
                        (
                            int(valid_cpu.shape[0]),
                            int(valid_cpu.shape[1]),
                            len(QUERY_LOCAL_UTILITY_HEAD_NAMES),
                        ),
                        dtype=torch.float32,
                    )
                    window_mask = torch.zeros_like(window_targets, dtype=torch.bool)
                    global_ok = (
                        valid_cpu
                        & (global_indices_cpu >= 0)
                        & (global_indices_cpu < int(target_cpu.shape[0]))
                    )
                    if bool(global_ok.any().item()):
                        window_targets[global_ok] = target_cpu[global_indices_cpu[global_ok]]
                        window_mask[global_ok] = mask_cpu[global_indices_cpu[global_ok]]
                    primary_logits = (
                        torch.cat(
                            [primary_logits_by_head[name] for name in ordered_head_names],
                            dim=-1,
                        )
                        .detach()
                        .clone()
                        .requires_grad_(True)
                    )
                    ablation_logits = torch.cat(
                        [ablation_logits_by_head[name] for name in ordered_head_names],
                        dim=-1,
                    ).detach()
                    with torch.enable_grad():
                        loss = _factorized_query_local_utility_loss(
                            head_logits=primary_logits,
                            head_targets=window_targets.to(device=device),
                            head_mask=window_mask.to(device=device),
                            global_indices=global_indices_cpu.to(device=device),
                            segment_budget_head_weight=float(segment_budget_head_weight),
                            segment_level_loss_weight=float(segment_level_loss_weight),
                            behavior_rank_loss_weight=float(behavior_rank_loss_weight),
                            sparse_head_rank_loss_weight=float(sparse_head_rank_loss_weight),
                            sparse_head_bce_target_mode=str(sparse_head_bce_target_mode),
                        )
                        gradient = torch.autograd.grad(
                            loss,
                            primary_logits,
                            allow_unused=False,
                        )[0].detach()
                    logit_delta = primary_logits.detach() - ablation_logits
                    for head_idx, head_name in enumerate(ordered_head_names):
                        local_mask = window_mask[..., head_idx].to(device=device)
                        if not bool(local_mask.any().item()):
                            continue
                        local_gradient = gradient[..., head_idx][local_mask]
                        local_delta = logit_delta[..., head_idx][local_mask]
                        parts = per_head_loss_gradient_parts[head_name]
                        parts["gradient"].append(local_gradient)
                        parts["logit_delta"].append(local_delta)
                        parts["descent_alignment"].append(-local_gradient * local_delta)

                if primary_transfer_by_head:
                    prior_start_idx = int(primary_points.shape[-1]) - int(prior_dim)
                    for channel_idx, channel_name in enumerate(prior_channel_names):
                        channel_points = primary_points.clone()
                        channel_points[..., prior_start_idx + int(channel_idx)] = 0.0
                        channel_shared = shared_embedding(channel_points, padding_mask)
                        channel_head_parts = per_channel_head_parts[channel_name]
                        for head_name, primary_transfer in primary_transfer_by_head.items():
                            channel_transfer = _head_mlp_transfer(
                                heads[head_name],
                                channel_shared,
                            )
                            if channel_transfer is None:
                                continue
                            primary_hidden, primary_logit, primary_probability = primary_transfer
                            _, channel_hidden, channel_logit, _ = channel_transfer
                            pairs = {
                                "hidden_activation": (primary_hidden, channel_hidden),
                                "logit": (primary_logit, channel_logit),
                                "probability": (
                                    primary_probability,
                                    torch.sigmoid(channel_logit),
                                ),
                            }
                            for stage_name, (primary_tensor, channel_tensor) in pairs.items():
                                primary_parts, ablation_parts = channel_head_parts[head_name][
                                    stage_name
                                ]
                                primary_parts.append(primary_tensor[valid])
                                ablation_parts.append(channel_tensor[valid])
    finally:
        model.train(original_training)

    per_head: dict[str, Any] = {}
    suppression_counts: dict[str, int] = {}
    for head_idx, head_name_raw in enumerate(QUERY_LOCAL_UTILITY_HEAD_NAMES):
        head_name = str(head_name_raw)
        stage_parts = per_head_parts.get(head_name)
        if stage_parts is None:
            continue
        stage_summary = {
            stage_name: _stage_tensor_sensitivity(
                primary_parts=primary_parts,
                ablation_parts=ablation_parts,
            )
            for stage_name, (primary_parts, ablation_parts) in stage_parts.items()
        }
        if not bool(stage_summary.get("shared_embedding", {}).get("available")):
            per_head[head_name] = {"available": False, "reason": "missing_transfer_tensors"}
            continue
        valid = mask_cpu[:, head_idx] if head_idx < int(mask_cpu.shape[1]) else torch.zeros(0)
        target = target_cpu[:, head_idx] if head_idx < int(target_cpu.shape[1]) else torch.zeros(0)
        target_valid = (
            target[valid] if int(valid.numel()) == int(target.numel()) else torch.zeros(0)
        )
        probability_stage = stage_parts["probability"][0]
        probability = (
            torch.cat([part.detach().cpu().float().reshape(-1) for part in probability_stage])
            if probability_stage
            else torch.zeros(0)
        )
        sigmoid_derivative = probability * (1.0 - probability) if int(probability.numel()) else None
        sigmoid_derivative_mean = (
            float(sigmoid_derivative.mean().item()) if sigmoid_derivative is not None else None
        )
        classification = _classify_head_transfer(
            shared=stage_summary["shared_embedding"],
            first=stage_summary["first_linear"],
            hidden=stage_summary["hidden_activation"],
            logit=stage_summary["logit"],
            probability=stage_summary["probability"],
            sigmoid_derivative_mean=sigmoid_derivative_mean,
        )
        suppression_counts[classification] = int(suppression_counts.get(classification, 0)) + 1
        output_alignment = _prior_output_layer_alignment_diagnostics(
            primary_hidden_parts=stage_parts["hidden_activation"][0],
            ablation_hidden_parts=stage_parts["hidden_activation"][1],
            primary_logit_parts=stage_parts["logit"][0],
            ablation_logit_parts=stage_parts["logit"][1],
            primary_probability_parts=stage_parts["probability"][0],
            target_parts=per_head_target_parts.get(head_name, []),
            mask_parts=per_head_mask_parts.get(head_name, []),
            final_weight=head_final_weights.get(head_name),
            slice_mask_parts=per_head_slice_mask_parts.get(head_name),
        )
        loss_gradient_alignment = _loss_gradient_alignment_summary(
            descent_alignment_parts=per_head_loss_gradient_parts.get(head_name, {}).get(
                "descent_alignment", []
            ),
            gradient_parts=per_head_loss_gradient_parts.get(head_name, {}).get("gradient", []),
            logit_delta_parts=per_head_loss_gradient_parts.get(head_name, {}).get(
                "logit_delta", []
            ),
            loss_config=loss_config,
        )
        per_head[head_name] = {
            "available": True,
            "classification": classification,
            "stage_sensitivity": stage_summary,
            "output_layer_alignment": output_alignment,
            "configured_loss_gradient_alignment": loss_gradient_alignment,
            "first_linear_delta_l2_to_shared_delta_l2": _ratio_from_summaries(
                stage_summary["first_linear"], stage_summary["shared_embedding"], "delta_l2"
            ),
            "hidden_delta_l2_to_first_linear_delta_l2": _ratio_from_summaries(
                stage_summary["hidden_activation"], stage_summary["first_linear"], "delta_l2"
            ),
            "logit_delta_l2_to_hidden_delta_l2": _ratio_from_summaries(
                stage_summary["logit"], stage_summary["hidden_activation"], "delta_l2"
            ),
            "probability_mean_abs_delta_to_logit_mean_abs_delta": _ratio_from_summaries(
                stage_summary["probability"], stage_summary["logit"], "mean_abs_delta"
            ),
            "sigmoid_derivative_mean": sigmoid_derivative_mean,
            "sigmoid_derivative_min": float(sigmoid_derivative.min().item())
            if sigmoid_derivative is not None and int(sigmoid_derivative.numel()) > 0
            else None,
            "sigmoid_derivative_lt_0_02_fraction": float(
                (sigmoid_derivative < 0.02).float().mean().item()
            )
            if sigmoid_derivative is not None and int(sigmoid_derivative.numel()) > 0
            else None,
            "target_mean": float(target_valid.mean().item()) if int(target_valid.numel()) else None,
            "target_std": float(target_valid.std(unbiased=False).item())
            if int(target_valid.numel()) > 1
            else 0.0,
            "valid_target_count": int(target_valid.numel()),
            **head_weight_stats.get(head_name, {}),
        }
    channel_rows: dict[str, Any] = {}
    for channel_name, per_head_channel_parts in per_channel_head_parts.items():
        channel_per_head: dict[str, Any] = {}
        for head_name_raw in QUERY_LOCAL_UTILITY_HEAD_NAMES:
            head_name = str(head_name_raw)
            stage_parts = per_head_channel_parts.get(head_name)
            if stage_parts is None:
                continue
            output_alignment = _prior_output_layer_alignment_diagnostics(
                primary_hidden_parts=stage_parts["hidden_activation"][0],
                ablation_hidden_parts=stage_parts["hidden_activation"][1],
                primary_logit_parts=stage_parts["logit"][0],
                ablation_logit_parts=stage_parts["logit"][1],
                primary_probability_parts=stage_parts["probability"][0],
                target_parts=per_head_target_parts.get(head_name, []),
                mask_parts=per_head_mask_parts.get(head_name, []),
                final_weight=head_final_weights.get(head_name),
                slice_mask_parts=per_head_slice_mask_parts.get(head_name),
            )
            channel_per_head[head_name] = {
                "available": bool(output_alignment.get("available")),
                "classification": _classify_prior_channel_output_alignment(
                    output_alignment,
                ),
                "output_layer_alignment": output_alignment,
            }
        channel_rows[channel_name] = {
            "available": bool(channel_per_head),
            "diagnostic_only": True,
            "per_head": channel_per_head,
        }
    channel_decomposition = _summarize_prior_channel_direction_decomposition(channel_rows)
    channel_decomposition["channel_count"] = len(channel_rows)
    channel_decomposition["per_channel"] = channel_rows
    return {
        "available": bool(per_head),
        "diagnostic_only": True,
        "per_head": per_head,
        "classification_counts": suppression_counts,
        "prior_channel_direction_decomposition": channel_decomposition,
    }
