"""Query-driven causality and final-summary tests."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from learning.model_features import (
    WORKLOAD_BLIND_RANGE_MODEL_DISABLED_PRIOR_FIELDS,
    WORKLOAD_BLIND_RANGE_MODEL_PRIOR_TRANSFORM,
    WORKLOAD_BLIND_RANGE_POINT_DIM,
    build_workload_blind_range_point_features,
)
from learning.model_training import (
    _fit_scaler_for_model,
)
from learning.outputs import TrainingOutputs
from learning.query_prior_fields import (
    QUERY_PRIOR_FIELD_NAMES,
    build_train_query_prior_fields,
    sample_query_prior_fields,
    zero_query_prior_field_like,
)
from orchestration.causality import (
    PRIOR_ABLATION_DIAGNOSTIC_CHAIN,
    PRIOR_ABLATION_SCORE_OUTPUT_SEMANTICS,
    head_ablation_sensitivity,
    head_output_sensitivity,
    model_prior_feature_sensitivity,
    prior_ablation_sensitivity_from_tensors,
    prior_ablation_sensitivity_payload,
    prior_feature_sample_sensitivity,
    prior_sample_gate_failures,
    retained_mask_comparison,
    score_ablation_sensitivity,
    score_rank_margin_boundary_diagnostics,
    training_outputs_with_query_prior_field,
)

HISTORICAL_SMALL_LOCAL_FAMILY = "small_local"


def _boundaries(trajectories: list[torch.Tensor]) -> list[tuple[int, int]]:
    cursor = 0
    out = []
    for trajectory in trajectories:
        end = cursor + int(trajectory.shape[0])
        out.append((cursor, end))
        cursor = end
    return out


def test_score_ablation_sensitivity_reports_score_and_mask_changes() -> None:
    primary_scores = torch.tensor([0.9, 0.8, 0.1, 0.0], dtype=torch.float32)
    ablation_scores = torch.tensor([0.1, 0.8, 0.9, 0.0], dtype=torch.float32)
    primary_mask = torch.tensor([True, True, False, False])
    ablation_mask = torch.tensor([False, True, True, False])

    diagnostics = score_ablation_sensitivity(
        primary_scores=primary_scores,
        ablation_scores=ablation_scores,
        primary_mask=primary_mask,
        ablation_mask=ablation_mask,
    )

    assert diagnostics["available"] is True
    assert diagnostics["mean_abs_score_delta"] > 0.0
    assert diagnostics["retained_mask_changed"] is True
    assert diagnostics["retained_mask_jaccard"] == 1.0 / 3.0
    assert diagnostics["score_topk_jaccard_at_retained_count"] == 1.0 / 3.0


def test_score_rank_margin_boundary_diagnostic_links_prior_delta_to_marginal_rows() -> None:
    primary_scores = torch.tensor([1.0, 0.9, 0.2, 0.1], dtype=torch.float32)
    ablation_scores = torch.tensor([0.99, 0.89, 0.15, 0.2], dtype=torch.float32)
    primary_mask = torch.tensor([True, True, False, False])
    selector_trace = {
        "retained_decision_marginal_query_local_utility_alignment": {
            "rows": [
                {
                    "point_index": 2,
                    "decision": "removed_addition_gain",
                    "source": "removed",
                    "marginal_query_local_utility": 0.50,
                    "marginal_query_local_utility_candidate_rank_fraction": 0.25,
                    "selector_score_candidate_rank_fraction": 0.90,
                    "failure_buckets": ["high_marginal_under_ranked_by_scores"],
                },
                {
                    "point_index": 0,
                    "decision": "retained_removal_loss",
                    "source": "learned",
                    "marginal_query_local_utility": 0.30,
                    "marginal_query_local_utility_candidate_rank_fraction": 0.50,
                },
                {
                    "point_index": 3,
                    "decision": "removed_addition_gain",
                    "source": "removed",
                    "marginal_query_local_utility": 0.05,
                    "marginal_query_local_utility_candidate_rank_fraction": 1.0,
                },
            ]
        }
    }

    diagnostic = score_rank_margin_boundary_diagnostics(
        primary_scores=primary_scores,
        ablation_scores=ablation_scores,
        primary_mask=primary_mask,
        ablation_mask=primary_mask.clone(),
        selector_trace=selector_trace,
    )

    assert diagnostic["available"] is True
    assert diagnostic["classification"] == "prior_score_deltas_below_topk_rank_margin"
    topk = diagnostic["topk_score_boundary"]
    assert topk["topk_boundary_margin"] == pytest.approx(0.7)
    assert topk["score_delta_crosses_topk_boundary"] is False
    marginal = diagnostic["marginal_row_score_delta_alignment"]
    assert marginal["missed_high_marginal_row_count"] == 1
    assert marginal["missed_high_marginal_mean_score_delta"] == pytest.approx(0.05)
    assert marginal["under_ranked_high_marginal_positive_score_delta_fraction"] == pytest.approx(
        1.0
    )


def test_prior_ablation_sensitivity_payload_exposes_score_output_chain() -> None:
    score_output = {"available": True, "mean_abs_score_delta": 0.25}

    payload = prior_ablation_sensitivity_payload(
        sampled_prior_features={"available": True, "mean_abs_feature_delta": 0.75},
        model_prior_features={"available": True, "mean_abs_feature_delta": 0.5},
        score_output=score_output,
        retained_mask={"available": True, "retained_mask_changed": True},
        raw_prediction={"available": True, "mean_abs_score_delta": 0.4},
        head_output={"available": True, "mean_abs_head_probability_delta": 0.01},
    )

    assert payload["available"] is True
    assert payload["diagnostic_chain"] == list(PRIOR_ABLATION_DIAGNOSTIC_CHAIN)
    assert "selector_score" not in payload
    assert payload["score_output"]["mean_abs_score_delta"] == pytest.approx(0.25)
    assert payload["score_output"]["semantics"] == PRIOR_ABLATION_SCORE_OUTPUT_SEMANTICS
    assert payload["retained_mask"]["retained_mask_changed"] is True
    assert payload["score_rank_margin_boundary"]["available"] is False
    assert payload["marginal_row_delta_path"]["available"] is False


def test_prior_ablation_sensitivity_from_tensors_builds_consistent_chain() -> None:
    primary_scores = torch.tensor([0.9, 0.8, 0.1, 0.0], dtype=torch.float32)
    ablation_scores = torch.tensor([0.1, 0.8, 0.9, 0.0], dtype=torch.float32)
    primary_raw = torch.tensor([3.0, 2.0, 1.0, 0.0], dtype=torch.float32)
    ablation_raw = torch.tensor([2.0, 2.0, 2.0, 0.0], dtype=torch.float32)
    primary_segment = torch.tensor([0.7, 0.6, 0.1, 0.0], dtype=torch.float32)
    ablation_segment = torch.tensor([0.2, 0.6, 0.2, 0.0], dtype=torch.float32)
    primary_heads = torch.tensor([[2.0, -1.0, 0.0, 0.5, 1.0, -0.5]], dtype=torch.float32)
    ablation_heads = primary_heads + 0.1
    primary_mask = torch.tensor([True, True, False, False])
    ablation_mask = torch.tensor([False, True, True, False])

    payload = prior_ablation_sensitivity_from_tensors(
        sampled_prior_features={"available": True},
        model_prior_features={"available": True},
        primary_scores=primary_scores,
        ablation_scores=ablation_scores,
        primary_raw_predictions=primary_raw,
        ablation_raw_predictions=ablation_raw,
        primary_head_logits=primary_heads,
        ablation_head_logits=ablation_heads,
        primary_mask=primary_mask,
        ablation_mask=ablation_mask,
        primary_segment_scores=primary_segment,
        ablation_segment_scores=ablation_segment,
        selector_trace={
            "retained_decision_marginal_query_local_utility_alignment": {
                "rows": [
                    {
                        "point_index": 0,
                        "decision": "retained_removal_loss",
                        "source": "learned",
                        "marginal_query_local_utility": 0.4,
                    }
                ]
            }
        },
    )

    assert "selector_score" not in payload
    assert payload["retained_mask"]["retained_mask_changed"] is True
    assert payload["retained_mask"]["retained_mask_jaccard"] == pytest.approx(1.0 / 3.0)
    assert payload["score_output"]["retained_mask_changed"] is True
    assert payload["score_output"]["retained_mask_jaccard"] == pytest.approx(1.0 / 3.0)
    assert payload["raw_prediction"]["mean_abs_score_delta"] > 0.0
    assert payload["head_output"]["head_probabilities_changed"] is True
    assert payload["score_rank_margin_boundary"]["available"] is True
    row_path = payload["marginal_row_delta_path"]
    assert row_path["available"] is True
    assert row_path["stage_available"]["segment_score"] is True
    assert row_path["top_marginal_rows"][0]["raw_prediction"]["delta"] == pytest.approx(1.0)
    assert row_path["top_marginal_rows"][0]["segment_score"]["delta"] == pytest.approx(0.5)
    assert row_path["top_marginal_rows"][0]["head_deltas"]["query_hit_probability"][
        "logit_delta"
    ] == pytest.approx(-0.1)
    composition = row_path["top_marginal_rows"][0]["factorized_composition"]
    assert composition["available"] is True
    assert composition["composed_score"]["delta"] < 0.0
    assert composition["contribution_deltas"]["query_hit_branch_shapley"] < 0.0
    assert row_path["groups"]["top_marginal"]["factorized_composition_available"] is True
    assert (
        row_path["groups"]["top_marginal"]["factorized_most_negative_mean_contribution"]["name"]
        in composition["contribution_deltas"]
    )


def test_training_outputs_with_query_prior_field_keeps_metadata_aligned() -> None:
    base_prior = {
        "schema_version": 3,
        "field_names": ["spatial_query_hit_probability"],
        "contains_eval_queries": False,
    }
    ablation_prior = {
        **base_prior,
        "ablation": "zero_query_prior_features",
        "diagnostics": {"zeroed_prior_features_preserve_train_extent": True},
    }
    trained = TrainingOutputs(
        model=torch.nn.Linear(1, 1),
        scaler=cast(Any, object()),
        labels=torch.ones(1),
        labelled_mask=torch.ones(1, dtype=torch.bool),
        history=[{"loss": 1.0}],
        epochs_trained=3,
        feature_context={
            "query_prior_field": base_prior,
            "query_prior_field_metadata": {"stale": True},
            "other": "kept",
        },
    )

    updated = training_outputs_with_query_prior_field(trained, ablation_prior)

    assert updated is not trained
    assert updated.model is trained.model
    assert updated.history is trained.history
    assert updated.feature_context["other"] == "kept"
    assert updated.feature_context["query_prior_field"] is ablation_prior
    assert updated.feature_context["query_prior_field_metadata"]["ablation"] == (
        "zero_query_prior_features"
    )
    assert "stale" not in updated.feature_context["query_prior_field_metadata"]


def test_head_ablation_sensitivity_reports_selector_raw_and_segment_channels() -> None:
    primary_scores = torch.tensor([0.9, 0.8, 0.1, 0.0], dtype=torch.float32)
    ablation_scores = torch.tensor([0.1, 0.8, 0.9, 0.0], dtype=torch.float32)
    primary_raw_predictions = torch.tensor([3.0, 2.0, 1.0, 0.0], dtype=torch.float32)
    ablation_raw_predictions = torch.tensor([2.0, 2.0, 2.0, 0.0], dtype=torch.float32)
    primary_segment_scores = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    ablation_segment_scores = torch.zeros(4, dtype=torch.float32)
    primary_mask = torch.tensor([True, True, False, False])
    ablation_mask = torch.tensor([False, True, True, False])

    diagnostics = head_ablation_sensitivity(
        primary_scores=primary_scores,
        ablation_scores=ablation_scores,
        primary_raw_predictions=primary_raw_predictions,
        ablation_raw_predictions=ablation_raw_predictions,
        primary_segment_scores=primary_segment_scores,
        ablation_segment_scores=ablation_segment_scores,
        primary_mask=primary_mask,
        ablation_mask=ablation_mask,
    )

    assert diagnostics["selector_score"]["available"] is True
    assert diagnostics["selector_score"]["retained_mask_changed"] is True
    assert diagnostics["raw_prediction"]["mean_abs_score_delta"] > 0.0
    assert diagnostics["segment_score"]["mean_abs_score_delta"] > 0.0


def test_head_output_sensitivity_reports_per_head_logit_and_probability_deltas() -> None:
    primary_head_logits = torch.tensor(
        [
            [2.0, -1.0, 0.0, 0.5, 1.0, -0.5],
            [1.5, -0.5, 0.2, 0.3, 0.8, -0.2],
        ],
        dtype=torch.float32,
    )
    ablation_head_logits = primary_head_logits.clone()
    ablation_head_logits[:, 0] -= 0.4
    ablation_head_logits[:, 4] += 0.2

    diagnostics = head_output_sensitivity(
        primary_head_logits=primary_head_logits,
        ablation_head_logits=ablation_head_logits,
    )

    assert diagnostics["available"] is True
    assert diagnostics["head_logits_changed"] is True
    assert diagnostics["head_probabilities_changed"] is True
    assert diagnostics["mean_abs_head_logit_delta"] > 0.0
    assert diagnostics["mean_abs_head_probability_delta"] > 0.0
    assert diagnostics["per_head"]["query_hit_probability"][
        "mean_abs_logit_delta"
    ] == pytest.approx(0.4)
    assert diagnostics["per_head"]["segment_budget_target"][
        "mean_abs_logit_delta"
    ] == pytest.approx(0.2)
    assert diagnostics["per_head"]["conditional_behavior_utility"][
        "mean_abs_logit_delta"
    ] == pytest.approx(0.0)


def test_retained_mask_comparison_reports_ablation_overlap() -> None:
    primary_mask = torch.tensor([True, True, False, False])
    ablation_mask = torch.tensor([False, True, True, False])

    diagnostics = retained_mask_comparison(
        primary_mask=primary_mask,
        ablation_mask=ablation_mask,
        expected_shape=primary_mask.shape,
    )

    assert diagnostics["available"] is True
    assert diagnostics["primary_retained_count"] == 2
    assert diagnostics["ablation_retained_count"] == 2
    assert diagnostics["retained_intersection_count"] == 1
    assert diagnostics["retained_union_count"] == 3
    assert diagnostics["retained_symmetric_difference_count"] == 2
    assert diagnostics["retained_mask_changed"] is True
    assert diagnostics["retained_mask_jaccard"] == 1.0 / 3.0
    assert diagnostics["retained_mask_hamming_fraction"] == 0.5


def test_prior_feature_sample_sensitivity_reports_input_level_changes() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=torch.float32,
    )
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 3.0,
            "lat_min": -1.0,
            "lat_max": 3.0,
            "lon_min": -1.0,
            "lon_max": 3.0,
        },
    }
    prior = build_train_query_prior_fields(
        points=points,
        boundaries=[(0, 3)],
        typed_queries=[query],
        workload_profile_id="range_query_mix",
        grid_bins=4,
        time_bins=2,
        smoothing_passes=0,
    )

    diagnostics = prior_feature_sample_sensitivity(
        points=points,
        primary_prior_field=prior,
        ablation_prior_field=None,
    )

    assert diagnostics["available"] is True
    assert diagnostics["point_count"] == 3
    assert diagnostics["feature_count"] == 6
    assert diagnostics["sampled_inputs_changed"] is True
    assert diagnostics["mean_abs_feature_delta"] > 0.0
    assert diagnostics["ablation_nonzero_fraction"] == 0.0
    assert diagnostics["per_feature"]["spatial_query_hit_probability"]["mean_abs_delta"] > 0.0


def test_model_prior_feature_sensitivity_reports_post_builder_and_scaler_changes() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 5.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 1.0, 10.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 3.0,
            "lat_min": -1.0,
            "lat_max": 3.0,
            "lon_min": -1.0,
            "lon_max": 3.0,
        },
    }
    prior = build_train_query_prior_fields(
        points=points,
        boundaries=[(0, 3)],
        typed_queries=[query],
        workload_profile_id="range_query_mix",
        grid_bins=4,
        time_bins=2,
        smoothing_passes=0,
    )
    zeroed = zero_query_prior_field_like(prior)
    queries = torch.zeros((1, 12), dtype=torch.float32)
    model_points = build_workload_blind_range_point_features(points, prior)
    scaler = _fit_scaler_for_model(model_points, queries, "workload_blind_range")

    raw_sampled = sample_query_prior_fields(points, prior)
    route_density_idx = QUERY_PRIOR_FIELD_NAMES.index("route_density_prior")
    assert raw_sampled[:, route_density_idx].abs().mean().item() > 0.0

    diagnostics = model_prior_feature_sensitivity(
        points=points,
        point_dim=WORKLOAD_BLIND_RANGE_POINT_DIM,
        scaler=scaler,
        primary_prior_field=prior,
        ablation_prior_field=zeroed,
        boundaries=[(0, 3)],
    )

    assert diagnostics["available"] is True
    assert diagnostics["disabled_prior_fields"] == list(
        WORKLOAD_BLIND_RANGE_MODEL_DISABLED_PRIOR_FIELDS
    )
    assert (
        diagnostics["model_prior_feature_transform"] == WORKLOAD_BLIND_RANGE_MODEL_PRIOR_TRANSFORM
    )
    model_input = diagnostics["model_input_prior_features"]
    normalized = diagnostics["normalized_model_prior_features"]
    assert model_input["sampled_inputs_changed"] is True
    assert normalized["sampled_inputs_changed"] is True
    assert model_input["per_feature"]["spatial_query_hit_probability"]["mean_abs_delta"] > 0.0
    assert normalized["per_feature"]["spatial_query_hit_probability"]["mean_abs_delta"] > 0.0
    assert model_input["per_feature"]["route_density_prior"]["mean_abs_delta"] == 0.0
    assert normalized["per_feature"]["route_density_prior"]["mean_abs_delta"] == 0.0
    assert diagnostics["scaler_prior_feature_ranges"]["route_density_prior"] == 1.0


def test_prior_sample_gate_failures_explain_empty_or_out_of_extent_priors() -> None:
    diagnostics = {
        "shuffled_prior_fields": {
            "sampled_prior_features": {
                "available": True,
                "primary_nonzero_fraction": 0.0,
                "sampled_inputs_changed": False,
                "points_outside_prior_extent_fraction": 1.0,
            },
            "model_prior_features": {
                "model_input_prior_features": {
                    "available": True,
                    "sampled_inputs_changed": False,
                },
                "normalized_model_prior_features": {
                    "available": True,
                    "sampled_inputs_changed": False,
                },
            },
        }
    }

    failures = prior_sample_gate_failures(diagnostics)

    assert "sampled_query_prior_features_all_zero" in failures
    assert "shuffled_prior_fields_did_not_change_sampled_inputs" in failures
    assert "shuffled_prior_fields_did_not_change_model_inputs" in failures
    assert "shuffled_prior_fields_did_not_change_normalized_model_inputs" in failures
    assert "eval_points_mostly_outside_query_prior_extent" in failures
