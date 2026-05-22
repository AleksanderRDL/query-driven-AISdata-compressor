"""Final fit diagnostics for trained models."""

from __future__ import annotations

import time
from typing import Any

import torch

from config.run_config import ModelConfig
from learning.factorized_head_diagnostics import (
    _behavior_head_training_signal_diagnostics,
    _factorized_final_score_composition_diagnostics,
    _factorized_head_fit_diagnostics,
    _prior_feature_learning_diagnostics,
    _segment_head_fit_diagnostics,
)
from learning.fit_diagnostics import train_target_fit_diagnostics
from learning.inference import windowed_predict_with_heads
from learning.model_features import WORKLOAD_BLIND_RANGE_MODEL_TYPE
from learning.query_prior_fields import QUERY_PRIOR_FIELD_NAMES
from workloads.query_types import ID_TO_QUERY_NAME


def build_final_training_fit_diagnostics(
    *,
    model: torch.nn.Module,
    norm_points: torch.Tensor,
    train_boundaries: list[tuple[int, int]],
    norm_queries: torch.Tensor,
    query_type_ids: torch.Tensor,
    model_config: ModelConfig,
    device: torch.device,
    amp_mode: Any,
    train_batch_size: int,
    model_type: str,
    training_target: torch.Tensor,
    training_labelled_mask: torch.Tensor,
    workload_type_id: int,
    training_target_basis: str,
    factorized_targets: torch.Tensor | None,
    factorized_mask: torch.Tensor | None,
    all_points: torch.Tensor,
    prior_queries: list[dict[str, Any]],
    canonical_segment_ids: torch.Tensor | None,
    seed: int,
    run_tag: str,
) -> dict[str, Any]:
    fit_t0 = time.perf_counter()
    fit_diagnostics: dict[str, Any] = {}
    try:
        train_predictions, train_head_logits = windowed_predict_with_heads(
            model=model,
            norm_points=norm_points,
            boundaries=train_boundaries,
            queries=norm_queries,
            query_type_ids=query_type_ids,
            window_length=model_config.window_length,
            window_stride=model_config.window_stride,
            batch_size=max(1, int(getattr(model_config, "inference_batch_size", train_batch_size))),
            device=device,
            amp_mode=amp_mode,
        )
        zero_prior_predictions: torch.Tensor | None = None
        zero_prior_head_logits: torch.Tensor | None = None
        if model_type == WORKLOAD_BLIND_RANGE_MODEL_TYPE and int(norm_points.shape[1]) >= len(
            QUERY_PRIOR_FIELD_NAMES
        ):
            zero_prior_norm_points = norm_points.clone()
            zero_prior_norm_points[:, -len(QUERY_PRIOR_FIELD_NAMES) :] = 0.0
            zero_prior_predictions, zero_prior_head_logits = windowed_predict_with_heads(
                model=model,
                norm_points=zero_prior_norm_points,
                boundaries=train_boundaries,
                queries=norm_queries,
                query_type_ids=query_type_ids,
                window_length=model_config.window_length,
                window_stride=model_config.window_stride,
                batch_size=max(
                    1, int(getattr(model_config, "inference_batch_size", train_batch_size))
                ),
                device=device,
                amp_mode=amp_mode,
            )
        fit_diagnostics = train_target_fit_diagnostics(
            predictions=train_predictions,
            target=training_target,
            labelled_mask=training_labelled_mask,
            boundaries=train_boundaries,
            model_config=model_config,
            workload_type=ID_TO_QUERY_NAME.get(workload_type_id, str(workload_type_id)),
            seed=seed,
        )
        fit_diagnostics["target_basis"] = training_target_basis
        fit_diagnostics.update(
            _factorized_head_fit_diagnostics(
                head_logits=train_head_logits,
                factorized_targets=factorized_targets,
                factorized_mask=factorized_mask,
                points=all_points,
                boundaries=train_boundaries,
                typed_queries=prior_queries,
                seed=seed,
            )
        )
        fit_diagnostics.update(
            _factorized_final_score_composition_diagnostics(
                head_logits=train_head_logits,
                factorized_targets=factorized_targets,
                scalar_target=training_target,
                scalar_mask=training_labelled_mask,
                seed=seed,
            )
        )
        fit_diagnostics.update(
            _segment_head_fit_diagnostics(
                head_logits=train_head_logits,
                factorized_targets=factorized_targets,
                factorized_mask=factorized_mask,
                canonical_segment_ids=canonical_segment_ids,
                seed=seed,
            )
        )
        fit_diagnostics["behavior_head_training_signal"] = (
            _behavior_head_training_signal_diagnostics(
                head_logits=train_head_logits,
                factorized_targets=factorized_targets,
                factorized_mask=factorized_mask,
                boundaries=train_boundaries,
                behavior_rank_loss_weight=float(
                    getattr(model_config, "query_local_utility_behavior_rank_loss_weight", 0.0)
                ),
            )
        )
        fit_diagnostics["prior_feature_learning_signal"] = _prior_feature_learning_diagnostics(
            model=model,
            norm_points=norm_points,
            primary_predictions=train_predictions,
            zero_prior_predictions=zero_prior_predictions,
            primary_head_logits=train_head_logits,
            zero_prior_head_logits=zero_prior_head_logits,
            factorized_targets=factorized_targets,
            factorized_mask=factorized_mask,
            scalar_target=training_target,
            scalar_mask=training_labelled_mask,
            raw_points=all_points,
            boundaries=train_boundaries,
            typed_queries=prior_queries,
            window_length=model_config.window_length,
            window_stride=model_config.window_stride,
            batch_size=max(1, int(getattr(model_config, "inference_batch_size", train_batch_size))),
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
            seed=seed,
        )
        fit_diagnostics["seconds"] = float(time.perf_counter() - fit_t0)
        matched_delta = fit_diagnostics.get("matched_mlqds_vs_uniform_target_recall")
        low_delta = fit_diagnostics.get("low_budget_mean_mlqds_vs_uniform_target_recall")
        matched_text = f"{float(matched_delta):+.4f}" if matched_delta is not None else "n/a"
        low_text = f"{float(low_delta):+.4f}" if low_delta is not None else "n/a"
        print(
            f"  [{run_tag}] train_target_fit "
            f"tau={fit_diagnostics.get('score_target_kendall_tau', 0.0):+.3f} "
            f"matched_delta={matched_text} low_delta={low_text} "
            f"({fit_diagnostics['seconds']:.2f}s)",
            flush=True,
        )
    except Exception as exc:  # pragma: no cover - diagnostic must not mask training result.
        fit_diagnostics = {
            "enabled": False,
            "error": str(exc),
            "seconds": float(time.perf_counter() - fit_t0),
        }
        print(f"  [{run_tag}] train_target_fit failed: {exc}", flush=True)
    return fit_diagnostics
