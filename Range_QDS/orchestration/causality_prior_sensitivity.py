"""Prior and head ablation sensitivity diagnostics."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from learning.model_features import (
    WORKLOAD_BLIND_RANGE_MODEL_DISABLED_PRIOR_FIELDS,
    WORKLOAD_BLIND_RANGE_MODEL_PRIOR_TRANSFORM,
    build_query_free_point_features_for_dim,
)
from learning.outputs import TrainingOutputs
from learning.query_prior_fields import (
    QUERY_PRIOR_FIELD_NAMES,
    query_prior_field_metadata,
    sample_query_prior_fields,
)
from learning.targets.query_local_utility import QUERY_LOCAL_UTILITY_HEAD_NAMES
from orchestration.causality_marginal_paths import (
    marginal_row_delta_path_diagnostics,
    score_rank_margin_boundary_diagnostics,
)
from orchestration.causality_score_sensitivity import (
    retained_mask_comparison,
    score_ablation_sensitivity,
)

PRIOR_ABLATION_SCORE_OUTPUT_SEMANTICS = "final_selector_score_after_mlqds_score_conversion"
PRIOR_ABLATION_DIAGNOSTIC_CHAIN = (
    "sampled_prior_features",
    "model_prior_features",
    "head_output",
    "raw_prediction",
    "score_output",
    "marginal_row_delta_path",
    "retained_mask",
)


def head_ablation_sensitivity(
    *,
    primary_scores: torch.Tensor | None,
    ablation_scores: torch.Tensor | None,
    primary_mask: torch.Tensor | None,
    ablation_mask: torch.Tensor | None,
    primary_raw_predictions: torch.Tensor | None = None,
    ablation_raw_predictions: torch.Tensor | None = None,
    primary_segment_scores: torch.Tensor | None = None,
    ablation_segment_scores: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Return score, raw-prediction, and segment-score sensitivity for one ablation."""
    diagnostics: dict[str, Any] = {
        "selector_score": score_ablation_sensitivity(
            primary_scores=primary_scores,
            ablation_scores=ablation_scores,
            primary_mask=primary_mask,
            ablation_mask=ablation_mask,
        )
    }
    if primary_raw_predictions is not None or ablation_raw_predictions is not None:
        diagnostics["raw_prediction"] = score_ablation_sensitivity(
            primary_scores=primary_raw_predictions,
            ablation_scores=ablation_raw_predictions,
            primary_mask=primary_mask,
            ablation_mask=ablation_mask,
        )
    if primary_segment_scores is not None or ablation_segment_scores is not None:
        diagnostics["segment_score"] = score_ablation_sensitivity(
            primary_scores=primary_segment_scores,
            ablation_scores=ablation_segment_scores,
            primary_mask=primary_mask,
            ablation_mask=ablation_mask,
        )
    return diagnostics


def _head_logit_matrix(head_logits: torch.Tensor) -> torch.Tensor:
    """Return head logits as [point, head] for sensitivity diagnostics."""
    logits = head_logits.detach().cpu().float()
    if logits.ndim == 3 and int(logits.shape[0]) == 1:
        logits = logits.squeeze(0)
    return logits


def head_output_sensitivity(
    *,
    primary_head_logits: torch.Tensor | None,
    ablation_head_logits: torch.Tensor | None,
) -> dict[str, Any]:
    """Return per-head logit and probability sensitivity for one model ablation."""
    if primary_head_logits is None or ablation_head_logits is None:
        return {"available": False, "reason": "missing_head_logits"}
    primary = _head_logit_matrix(primary_head_logits)
    ablation = _head_logit_matrix(ablation_head_logits)
    if primary.ndim != 2 or ablation.ndim != 2 or primary.shape != ablation.shape:
        return {
            "available": False,
            "reason": "head_logit_shape_mismatch",
            "primary_shape": list(primary.shape),
            "ablation_shape": list(ablation.shape),
        }
    head_names: list[str] = [str(name) for name in QUERY_LOCAL_UTILITY_HEAD_NAMES]
    logit = _feature_matrix_sensitivity(
        primary=primary,
        ablation=ablation,
        feature_names=head_names,
        point_count=int(primary.shape[0]),
    )
    probability = _feature_matrix_sensitivity(
        primary=torch.sigmoid(primary),
        ablation=torch.sigmoid(ablation),
        feature_names=head_names,
        point_count=int(primary.shape[0]),
    )
    per_head: dict[str, dict[str, float | int | bool | None]] = {}
    logit_per_feature = logit.get("per_feature") if isinstance(logit, dict) else {}
    probability_per_feature = (
        probability.get("per_feature") if isinstance(probability, dict) else {}
    )
    for head_name in head_names:
        logit_row = (
            logit_per_feature.get(head_name, {}) if isinstance(logit_per_feature, dict) else {}
        )
        probability_row = (
            probability_per_feature.get(head_name, {})
            if isinstance(probability_per_feature, dict)
            else {}
        )
        per_head[head_name] = {
            "finite_count": logit_row.get("finite_count"),
            "mean_abs_logit_delta": logit_row.get("mean_abs_delta"),
            "max_abs_logit_delta": logit_row.get("max_abs_delta"),
            "mean_abs_probability_delta": probability_row.get("mean_abs_delta"),
            "max_abs_probability_delta": probability_row.get("max_abs_delta"),
            "primary_probability_mean": probability_row.get("primary_mean"),
            "ablation_probability_mean": probability_row.get("ablation_mean"),
        }
    return {
        "available": bool(logit.get("available") and probability.get("available")),
        "point_count": int(primary.shape[0]),
        "head_count": int(primary.shape[1]),
        "head_names": head_names,
        "head_logits_changed": bool(logit.get("sampled_inputs_changed", False)),
        "head_probabilities_changed": bool(probability.get("sampled_inputs_changed", False)),
        "mean_abs_head_logit_delta": logit.get("mean_abs_feature_delta"),
        "max_abs_head_logit_delta": logit.get("max_abs_feature_delta"),
        "mean_abs_head_probability_delta": probability.get("mean_abs_feature_delta"),
        "max_abs_head_probability_delta": probability.get("max_abs_feature_delta"),
        "logit": logit,
        "probability": probability,
        "per_head": per_head,
    }


def prior_ablation_sensitivity_payload(
    *,
    sampled_prior_features: dict[str, Any],
    model_prior_features: dict[str, Any],
    score_output: dict[str, Any],
    retained_mask: dict[str, Any],
    raw_prediction: dict[str, Any],
    head_output: dict[str, Any],
    score_rank_margin_boundary: dict[str, Any] | None = None,
    marginal_row_delta_path: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one canonical prior-ablation sensitivity chain for run artifacts."""
    named_score_output = dict(score_output)
    named_score_output["semantics"] = PRIOR_ABLATION_SCORE_OUTPUT_SEMANTICS
    return {
        "available": True,
        "diagnostic_chain": list(PRIOR_ABLATION_DIAGNOSTIC_CHAIN),
        "sampled_prior_features": sampled_prior_features,
        "model_prior_features": model_prior_features,
        "score_output": named_score_output,
        "retained_mask": retained_mask,
        "raw_prediction": raw_prediction,
        "head_output": head_output,
        "score_rank_margin_boundary": (
            score_rank_margin_boundary
            if isinstance(score_rank_margin_boundary, dict)
            else {"available": False, "reason": "not_computed"}
        ),
        "marginal_row_delta_path": (
            marginal_row_delta_path
            if isinstance(marginal_row_delta_path, dict)
            else {"available": False, "reason": "not_computed"}
        ),
    }


def prior_ablation_sensitivity_from_tensors(
    *,
    sampled_prior_features: dict[str, Any],
    model_prior_features: dict[str, Any],
    primary_scores: torch.Tensor | None,
    ablation_scores: torch.Tensor | None,
    primary_raw_predictions: torch.Tensor | None,
    ablation_raw_predictions: torch.Tensor | None,
    primary_head_logits: torch.Tensor | None,
    ablation_head_logits: torch.Tensor | None,
    primary_mask: torch.Tensor | None,
    ablation_mask: torch.Tensor | None,
    selector_trace: dict[str, Any] | None = None,
    primary_segment_scores: torch.Tensor | None = None,
    ablation_segment_scores: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Return the full prior-ablation sensitivity chain from cached tensors."""
    return prior_ablation_sensitivity_payload(
        sampled_prior_features=sampled_prior_features,
        model_prior_features=model_prior_features,
        score_output=score_ablation_sensitivity(
            primary_scores=primary_scores,
            ablation_scores=ablation_scores,
            primary_mask=primary_mask,
            ablation_mask=ablation_mask,
        ),
        retained_mask=retained_mask_comparison(
            primary_mask=primary_mask,
            ablation_mask=ablation_mask,
            expected_shape=primary_scores.shape if primary_scores is not None else None,
        ),
        raw_prediction=score_ablation_sensitivity(
            primary_scores=primary_raw_predictions,
            ablation_scores=ablation_raw_predictions,
            primary_mask=primary_mask,
            ablation_mask=ablation_mask,
        ),
        head_output=head_output_sensitivity(
            primary_head_logits=primary_head_logits,
            ablation_head_logits=ablation_head_logits,
        ),
        score_rank_margin_boundary=score_rank_margin_boundary_diagnostics(
            primary_scores=primary_scores,
            ablation_scores=ablation_scores,
            primary_mask=primary_mask,
            ablation_mask=ablation_mask,
            selector_trace=selector_trace,
        ),
        marginal_row_delta_path=marginal_row_delta_path_diagnostics(
            selector_trace=selector_trace,
            primary_scores=primary_scores,
            ablation_scores=ablation_scores,
            primary_raw_predictions=primary_raw_predictions,
            ablation_raw_predictions=ablation_raw_predictions,
            primary_segment_scores=primary_segment_scores,
            ablation_segment_scores=ablation_segment_scores,
            primary_head_logits=primary_head_logits,
            ablation_head_logits=ablation_head_logits,
            primary_mask=primary_mask,
            ablation_mask=ablation_mask,
        ),
    )


def training_outputs_with_query_prior_field(
    trained: TrainingOutputs,
    query_prior_field: dict[str, Any],
) -> TrainingOutputs:
    """Return training outputs with a swapped query-prior field and matching metadata."""
    return replace(
        trained,
        feature_context={
            **trained.feature_context,
            "query_prior_field": query_prior_field,
            "query_prior_field_metadata": query_prior_field_metadata(query_prior_field),
        },
    )


def prior_feature_sample_sensitivity(
    *,
    points: torch.Tensor,
    primary_prior_field: dict[str, Any] | None,
    ablation_prior_field: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return sampled query-prior feature sensitivity at eval-compression points."""
    if primary_prior_field is None:
        return {"available": False, "reason": "missing_primary_prior_field"}
    primary = sample_query_prior_fields(points, primary_prior_field).detach().cpu().float()
    ablation = sample_query_prior_fields(points, ablation_prior_field).detach().cpu().float()
    return _feature_matrix_sensitivity(
        primary=primary,
        ablation=ablation,
        feature_names=QUERY_PRIOR_FIELD_NAMES,
        point_count=int(points.shape[0]),
        primary_prior_field=primary_prior_field,
        points=points,
    )


def _feature_matrix_sensitivity(
    *,
    primary: torch.Tensor,
    ablation: torch.Tensor,
    feature_names: tuple[str, ...] | list[str],
    point_count: int,
    primary_prior_field: dict[str, Any] | None = None,
    points: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Return feature-delta diagnostics for aligned primary/ablation matrices."""
    if int(primary.numel()) == 0 or primary.shape != ablation.shape:
        return {
            "available": False,
            "reason": "feature_shape_mismatch",
            "primary_shape": list(primary.shape),
            "ablation_shape": list(ablation.shape),
        }
    finite = torch.isfinite(primary) & torch.isfinite(ablation)
    if not bool(finite.any().item()):
        return {"available": False, "reason": "no_finite_sampled_features"}
    delta = primary - ablation
    finite_delta = delta[finite]
    per_feature: dict[str, dict[str, float | int]] = {}
    named_features = list(feature_names)
    for idx in range(int(primary.shape[1])):
        name = named_features[idx] if idx < len(named_features) else f"feature_{idx}"
        primary_col = primary[:, idx]
        ablation_col = ablation[:, idx]
        col_finite = torch.isfinite(primary_col) & torch.isfinite(ablation_col)
        if not bool(col_finite.any().item()):
            per_feature[name] = {"finite_count": 0}
            continue
        primary_f = primary_col[col_finite]
        ablation_f = ablation_col[col_finite]
        col_delta = primary_f - ablation_f
        per_feature[name] = {
            "finite_count": int(col_finite.sum().item()),
            "mean_abs_delta": float(col_delta.abs().mean().item()),
            "max_abs_delta": float(col_delta.abs().max().item()),
            "primary_mean": float(primary_f.mean().item()),
            "ablation_mean": float(ablation_f.mean().item()),
            "primary_std": float(primary_f.std(unbiased=False).item())
            if int(primary_f.numel()) > 1
            else 0.0,
            "ablation_std": float(ablation_f.std(unbiased=False).item())
            if int(ablation_f.numel()) > 1
            else 0.0,
            "primary_nonzero_fraction": float((primary_f.abs() > 1e-12).float().mean().item()),
            "ablation_nonzero_fraction": float((ablation_f.abs() > 1e-12).float().mean().item()),
        }
    primary_flat = primary[torch.isfinite(primary)]
    ablation_flat = ablation[torch.isfinite(ablation)]
    outside_extent_fraction: float | None = None
    extent = primary_prior_field.get("extent") if isinstance(primary_prior_field, dict) else None
    if isinstance(extent, dict) and points is not None and int(points.shape[0]) > 0:
        lat = points[:, 1].detach().cpu().float()
        lon = points[:, 2].detach().cpu().float()
        outside = (
            (lat < float(extent.get("lat_min", -float("inf"))))
            | (lat > float(extent.get("lat_max", float("inf"))))
            | (lon < float(extent.get("lon_min", -float("inf"))))
            | (lon > float(extent.get("lon_max", float("inf"))))
        )
        outside_extent_fraction = float(outside.float().mean().item())
    return {
        "available": True,
        "point_count": int(point_count),
        "feature_count": int(primary.shape[1]),
        "finite_value_count": int(finite.sum().item()),
        "sampled_inputs_changed": bool(float(finite_delta.abs().max().item()) > 1e-9),
        "mean_abs_feature_delta": float(finite_delta.abs().mean().item()),
        "max_abs_feature_delta": float(finite_delta.abs().max().item()),
        "mean_signed_feature_delta": float(finite_delta.mean().item()),
        "primary_feature_mean": float(primary_flat.mean().item())
        if int(primary_flat.numel()) > 0
        else 0.0,
        "ablation_feature_mean": float(ablation_flat.mean().item())
        if int(ablation_flat.numel()) > 0
        else 0.0,
        "primary_feature_std": (
            float(primary_flat.std(unbiased=False).item()) if int(primary_flat.numel()) > 1 else 0.0
        ),
        "ablation_feature_std": (
            float(ablation_flat.std(unbiased=False).item())
            if int(ablation_flat.numel()) > 1
            else 0.0
        ),
        "primary_nonzero_fraction": float((primary.abs() > 1e-12).float().mean().item()),
        "ablation_nonzero_fraction": float((ablation.abs() > 1e-12).float().mean().item()),
        "points_outside_prior_extent_fraction": outside_extent_fraction,
        "per_feature": per_feature,
    }


def model_prior_feature_sensitivity(
    *,
    points: torch.Tensor,
    point_dim: int,
    scaler: Any,
    primary_prior_field: dict[str, Any] | None,
    ablation_prior_field: dict[str, Any] | None,
    boundaries: list[tuple[int, int]] | None = None,
    trajectory_mmsis: list[int] | None = None,
) -> dict[str, Any]:
    """Return prior-feature sensitivity at the actual model-input and scaler levels."""
    if primary_prior_field is None:
        return {"available": False, "reason": "missing_primary_prior_field"}
    prior_dim = len(QUERY_PRIOR_FIELD_NAMES)
    point_dim_int = int(point_dim)
    if point_dim_int < prior_dim:
        return {
            "available": False,
            "reason": "point_dim_smaller_than_prior_dim",
            "point_dim": point_dim_int,
            "prior_feature_count": prior_dim,
        }
    try:
        primary_model_points = build_query_free_point_features_for_dim(
            points,
            point_dim_int,
            boundaries=boundaries,
            trajectory_mmsis=trajectory_mmsis,
            query_prior_field=primary_prior_field,
        )
        ablation_model_points = build_query_free_point_features_for_dim(
            points,
            point_dim_int,
            boundaries=boundaries,
            trajectory_mmsis=trajectory_mmsis,
            query_prior_field=ablation_prior_field,
        )
    except Exception as exc:
        return {
            "available": False,
            "reason": "model_point_feature_build_failed",
            "error": str(exc),
            "point_dim": point_dim_int,
        }
    if primary_model_points.shape != ablation_model_points.shape:
        return {
            "available": False,
            "reason": "model_point_feature_shape_mismatch",
            "primary_shape": list(primary_model_points.shape),
            "ablation_shape": list(ablation_model_points.shape),
        }
    try:
        primary_normalized = scaler.transform_points(primary_model_points)
        ablation_normalized = scaler.transform_points(ablation_model_points)
    except Exception as exc:
        return {
            "available": False,
            "reason": "model_point_feature_scaling_failed",
            "error": str(exc),
            "point_dim": point_dim_int,
        }
    prior_slice = slice(-prior_dim, None)
    model_prior_features = _feature_matrix_sensitivity(
        primary=primary_model_points[:, prior_slice].detach().cpu().float(),
        ablation=ablation_model_points[:, prior_slice].detach().cpu().float(),
        feature_names=QUERY_PRIOR_FIELD_NAMES,
        point_count=int(points.shape[0]),
    )
    normalized_prior_features = _feature_matrix_sensitivity(
        primary=primary_normalized[:, prior_slice].detach().cpu().float(),
        ablation=ablation_normalized[:, prior_slice].detach().cpu().float(),
        feature_names=QUERY_PRIOR_FIELD_NAMES,
        point_count=int(points.shape[0]),
    )
    scaler_min = getattr(scaler, "point_min", None)
    scaler_max = getattr(scaler, "point_max", None)
    scaler_prior_ranges: dict[str, float] = {}
    if isinstance(scaler_min, torch.Tensor) and isinstance(scaler_max, torch.Tensor):
        min_prior = scaler_min.detach().cpu().float()[prior_slice]
        max_prior = scaler_max.detach().cpu().float()[prior_slice]
        ranges = torch.clamp(max_prior - min_prior, min=0.0)
        for idx, name in enumerate(QUERY_PRIOR_FIELD_NAMES):
            if idx < int(ranges.numel()):
                scaler_prior_ranges[name] = float(ranges[idx].item())
    return {
        "available": bool(
            model_prior_features.get("available") and normalized_prior_features.get("available")
        ),
        "point_dim": point_dim_int,
        "prior_feature_count": prior_dim,
        "disabled_prior_fields": list(WORKLOAD_BLIND_RANGE_MODEL_DISABLED_PRIOR_FIELDS),
        "model_prior_feature_transform": WORKLOAD_BLIND_RANGE_MODEL_PRIOR_TRANSFORM,
        "model_input_prior_features": model_prior_features,
        "normalized_model_prior_features": normalized_prior_features,
        "scaler_prior_feature_ranges": scaler_prior_ranges,
    }


def prior_sample_gate_failures(prior_sensitivity_diagnostics: dict[str, Any]) -> list[str]:
    """Return failures showing prior-feature ablations did not exercise useful inputs."""
    shuffled = prior_sensitivity_diagnostics.get("shuffled_prior_fields")
    if not isinstance(shuffled, dict):
        return []
    sampled = shuffled.get("sampled_prior_features")
    if not isinstance(sampled, dict) or not sampled.get("available"):
        return []
    failures: list[str] = []
    primary_nonzero = float(sampled.get("primary_nonzero_fraction") or 0.0)
    if primary_nonzero <= 1e-6:
        failures.append("sampled_query_prior_features_all_zero")
    if not bool(sampled.get("sampled_inputs_changed", False)):
        failures.append("shuffled_prior_fields_did_not_change_sampled_inputs")
    model_prior = shuffled.get("model_prior_features")
    if isinstance(model_prior, dict):
        model_input = model_prior.get("model_input_prior_features")
        if isinstance(model_input, dict) and model_input.get("available"):
            if not bool(model_input.get("sampled_inputs_changed", False)):
                failures.append("shuffled_prior_fields_did_not_change_model_inputs")
        normalized = model_prior.get("normalized_model_prior_features")
        if isinstance(normalized, dict) and normalized.get("available"):
            if not bool(normalized.get("sampled_inputs_changed", False)):
                failures.append("shuffled_prior_fields_did_not_change_normalized_model_inputs")
    outside_fraction = sampled.get("points_outside_prior_extent_fraction")
    if outside_fraction is not None and float(outside_fraction) > 0.50:
        failures.append("eval_points_mostly_outside_query_prior_extent")
    return failures
