"""Historical-prior setup paths for model training."""

from __future__ import annotations

import time
from typing import Any

import torch

from config.run_config import ModelConfig
from learning.fit_diagnostics import train_target_fit_diagnostics
from learning.losses import _safe_quantile
from learning.model_factory import require_historical_prior_model
from learning.model_features import (
    HISTORICAL_PRIOR_MODEL_TYPES,
    NONPARAMETRIC_HISTORICAL_PRIOR_MODEL_TYPES,
)
from learning.model_training_helpers import _historical_prior_support_mask
from learning.outputs import TrainingOutputs
from workloads.query_types import ID_TO_QUERY_NAME


def configure_historical_prior_training(
    *,
    model: torch.nn.Module,
    model_type: str,
    norm_points: torch.Tensor,
    train_boundaries: list[tuple[int, int]],
    training_target: torch.Tensor,
    train_point_source_ids: torch.Tensor | None,
    model_config: ModelConfig,
    target_diagnostics: dict[str, Any],
    training_labelled_mask: torch.Tensor,
    workload_type_id: int,
    scaler: Any,
    labels: torch.Tensor,
    labelled_mask: torch.Tensor,
    seed: int,
    run_tag: str,
) -> TrainingOutputs | None:
    if model_type in HISTORICAL_PRIOR_MODEL_TYPES:
        historical_model = require_historical_prior_model(model, model_type=model_type)
        prior_points = norm_points
        prior_targets = training_target
        support_ratio = min(
            1.0, max(0.0, float(getattr(model_config, "historical_prior_support_ratio", 1.0)))
        )
        support_mask = _historical_prior_support_mask(
            targets=training_target,
            boundaries=train_boundaries,
            support_ratio=support_ratio,
        )
        if not bool(support_mask.any().item()):
            raise ValueError("historical_prior_support_ratio removed every training point.")
        prior_points = prior_points[support_mask]
        prior_targets = prior_targets[support_mask]
        prior_source_ids = (
            train_point_source_ids[support_mask] if train_point_source_ids is not None else None
        )
        target_diagnostics["historical_prior_support_ratio"] = float(support_ratio)
        target_diagnostics["historical_prior_support_pre_min_count"] = int(prior_targets.shape[0])
        target_diagnostics["historical_prior_support_pre_min_fraction"] = float(
            int(prior_targets.shape[0]) / max(1, int(training_target.shape[0]))
        )
        target_diagnostics["historical_prior_min_target"] = float(
            getattr(model_config, "historical_prior_min_target", 0.0)
        )
        historical_model.set_prior(prior_points, prior_targets, source_ids=prior_source_ids)
        target_diagnostics["historical_prior_stored_support_count"] = int(
            historical_model.historical_targets.shape[0]
        )
        target_diagnostics["historical_prior_stored_support_fraction"] = float(
            int(historical_model.historical_targets.shape[0])
            / max(1, int(training_target.shape[0]))
        )
        stored_sources = torch.unique(historical_model.historical_source_ids).numel()
        target_diagnostics["historical_prior_source_aggregation"] = str(
            getattr(model_config, "historical_prior_source_aggregation", "none")
        )
        target_diagnostics["historical_prior_source_count"] = int(stored_sources)
    if model_type in NONPARAMETRIC_HISTORICAL_PRIOR_MODEL_TYPES:
        history = [
            {
                "epoch": 0.0,
                "loss": 0.0,
                "pred_std": float(training_target.std(unbiased=False).item()),
                f"positive_fraction_t{workload_type_id}": float(
                    target_diagnostics.get("positive_label_fraction", 0.0)
                ),
                f"label_p95_t{workload_type_id}": float(
                    _safe_quantile(training_target[training_labelled_mask], 0.95).item()
                )
                if bool(training_labelled_mask.any().item())
                else 0.0,
                f"kendall_tau_t{workload_type_id}": 1.0,
                "raw_training_window_count": 0.0,
                "trained_training_window_count": 0.0,
                "filtered_zero_window_count": 0.0,
            }
        ]
        fit_t0 = time.perf_counter()
        try:
            with torch.no_grad():
                train_predictions = model(
                    norm_points.unsqueeze(0),
                    queries=None,
                    query_type_ids=None,
                ).squeeze(0)
            fit_diagnostics = train_target_fit_diagnostics(
                predictions=train_predictions,
                target=training_target,
                labelled_mask=training_labelled_mask,
                boundaries=train_boundaries,
                model_config=model_config,
                workload_type=ID_TO_QUERY_NAME.get(workload_type_id, str(workload_type_id)),
                seed=seed,
            )
            fit_diagnostics["seconds"] = float(time.perf_counter() - fit_t0)
            fit_diagnostics["model_fits_stored_train_support"] = True
            matched_delta = fit_diagnostics.get("matched_mlqds_vs_uniform_target_recall")
            low_delta = fit_diagnostics.get("low_budget_mean_mlqds_vs_uniform_target_recall")
            matched_text = f"{float(matched_delta):+.4f}" if matched_delta is not None else "n/a"
            low_text = f"{float(low_delta):+.4f}" if low_delta is not None else "n/a"
            print(
                f"  [{run_tag}] historical_prior_train_target_fit "
                f"tau={fit_diagnostics.get('score_target_kendall_tau', 0.0):+.3f} "
                f"matched_delta={matched_text} low_delta={low_text} "
                f"({fit_diagnostics['seconds']:.2f}s)",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - diagnostic must not mask fitted prior.
            fit_diagnostics = {
                "enabled": False,
                "error": str(exc),
                "seconds": float(time.perf_counter() - fit_t0),
                "model_fits_stored_train_support": True,
            }
            print(f"  [{run_tag}] historical_prior_train_target_fit failed: {exc}", flush=True)
        return TrainingOutputs(
            model=model.eval(),
            scaler=scaler,
            labels=labels,
            labelled_mask=labelled_mask,
            history=history,
            epochs_trained=0,
            best_epoch=0,
            best_loss=0.0,
            best_selection_score=0.0,
            target_diagnostics=target_diagnostics,
            fit_diagnostics=fit_diagnostics,
        )
    return None
