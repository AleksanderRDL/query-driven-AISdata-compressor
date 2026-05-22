"""QueryLocalUtility loss, fit, and training diagnostic tests."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from learning.factorized_head_diagnostics import (
    _behavior_head_training_signal_diagnostics,
    _factorized_final_score_composition_diagnostics,
    _factorized_head_fit_diagnostics,
    _initialize_factorized_head_output_biases_from_targets,
    _prior_feature_learning_diagnostics,
    _prior_output_layer_alignment_diagnostics,
)
from learning.fit_diagnostics import _training_target_diagnostics
from learning.model_features import (
    WORKLOAD_BLIND_RANGE_POINT_DIM,
)
from learning.model_training import (
    _scalar_training_target_for_mode,
)
from learning.optimization_epoch import (
    _behavior_head_rank_loss,
    _calibrated_sparse_head_bce_targets,
    _factorized_query_local_utility_loss,
    _segment_budget_head_segment_level_loss,
    _sparse_head_rank_loss,
)
from learning.query_prior_fields import QUERY_PRIOR_FIELD_NAMES
from learning.targets.query_local_utility import (
    QUERY_LOCAL_UTILITY_FINAL_LABEL_FORMULA,
    QUERY_LOCAL_UTILITY_HEAD_NAMES,
    query_local_utility_point_score,
)
from models.workload_blind_range import WorkloadBlindRangeModel

HISTORICAL_SMALL_LOCAL_FAMILY = "small_local"


def _boundaries(trajectories: list[torch.Tensor]) -> list[tuple[int, int]]:
    cursor = 0
    out = []
    for trajectory in trajectories:
        end = cursor + int(trajectory.shape[0])
        out.append((cursor, end))
        cursor = end
    return out


def test_segment_budget_head_has_segment_level_loss() -> None:
    head_targets = torch.zeros((1, 8, len(QUERY_LOCAL_UTILITY_HEAD_NAMES)), dtype=torch.float32)
    head_mask = torch.ones_like(head_targets, dtype=torch.bool)
    segment_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("segment_budget_target")
    head_targets[:, :4, segment_idx] = 1.0
    aligned = torch.zeros_like(head_targets)
    reversed_logits = torch.zeros_like(head_targets)
    aligned[:, :4, segment_idx] = 4.0
    aligned[:, 4:, segment_idx] = -4.0
    reversed_logits[:, :4, segment_idx] = -4.0
    reversed_logits[:, 4:, segment_idx] = 4.0

    aligned_loss = _segment_budget_head_segment_level_loss(
        head_logits=aligned,
        head_targets=head_targets,
        head_mask=head_mask,
        segment_size=4,
    )
    reversed_loss = _segment_budget_head_segment_level_loss(
        head_logits=reversed_logits,
        head_targets=head_targets,
        head_mask=head_mask,
        segment_size=4,
    )

    assert float(aligned_loss.item()) < float(reversed_loss.item())


def test_factorized_query_local_utility_loss_exposes_segment_budget_weights() -> None:
    head_targets = torch.zeros((1, 8, len(QUERY_LOCAL_UTILITY_HEAD_NAMES)), dtype=torch.float32)
    head_mask = torch.zeros_like(head_targets, dtype=torch.bool)
    segment_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("segment_budget_target")
    head_targets[:, :4, segment_idx] = 1.0
    head_mask[:, :, segment_idx] = True
    reversed_logits = torch.zeros_like(head_targets)
    reversed_logits[:, :4, segment_idx] = -4.0
    reversed_logits[:, 4:, segment_idx] = 4.0

    implicit_default = _factorized_query_local_utility_loss(
        head_logits=reversed_logits,
        head_targets=head_targets,
        head_mask=head_mask,
        segment_size=4,
    )
    explicit_default = _factorized_query_local_utility_loss(
        head_logits=reversed_logits,
        head_targets=head_targets,
        head_mask=head_mask,
        segment_size=4,
        segment_budget_head_weight=0.10,
        segment_level_loss_weight=0.25,
    )
    stronger_segment_pressure = _factorized_query_local_utility_loss(
        head_logits=reversed_logits,
        head_targets=head_targets,
        head_mask=head_mask,
        segment_size=4,
        segment_budget_head_weight=0.40,
        segment_level_loss_weight=1.0,
    )
    point_only = _factorized_query_local_utility_loss(
        head_logits=reversed_logits,
        head_targets=head_targets,
        head_mask=head_mask,
        segment_size=4,
        segment_budget_head_weight=0.10,
        segment_level_loss_weight=0.0,
    )

    assert torch.allclose(implicit_default, explicit_default)
    assert float(stronger_segment_pressure.item()) > float(implicit_default.item())
    assert float(implicit_default.item()) > float(point_only.item())


def test_behavior_head_rank_loss_penalizes_reversed_behavior_order() -> None:
    head_targets = torch.zeros((1, 8, len(QUERY_LOCAL_UTILITY_HEAD_NAMES)), dtype=torch.float32)
    head_mask = torch.zeros_like(head_targets, dtype=torch.bool)
    behavior_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("conditional_behavior_utility")
    head_targets[0, :, behavior_idx] = torch.tensor(
        [1.0, 0.9, 0.8, 0.7, 0.1, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
    head_mask[0, :, behavior_idx] = True
    aligned_logits = torch.zeros_like(head_targets)
    reversed_logits = torch.zeros_like(head_targets)
    aligned_logits[0, :, behavior_idx] = torch.linspace(4.0, -4.0, 8)
    reversed_logits[0, :, behavior_idx] = torch.linspace(-4.0, 4.0, 8)

    aligned = _behavior_head_rank_loss(
        head_logits=aligned_logits,
        head_targets=head_targets,
        head_mask=head_mask,
    )
    reversed_loss = _behavior_head_rank_loss(
        head_logits=reversed_logits,
        head_targets=head_targets,
        head_mask=head_mask,
    )
    without_behavior_rank = _factorized_query_local_utility_loss(
        head_logits=reversed_logits,
        head_targets=head_targets,
        head_mask=head_mask,
        behavior_rank_loss_weight=0.0,
    )
    implicit_default = _factorized_query_local_utility_loss(
        head_logits=reversed_logits,
        head_targets=head_targets,
        head_mask=head_mask,
    )
    with_behavior_rank = _factorized_query_local_utility_loss(
        head_logits=reversed_logits,
        head_targets=head_targets,
        head_mask=head_mask,
        behavior_rank_loss_weight=1.0,
    )

    assert float(aligned.item()) < float(reversed_loss.item())
    assert float(implicit_default.item()) > float(without_behavior_rank.item())
    assert float(with_behavior_rank.item()) > float(without_behavior_rank.item())


def test_behavior_head_training_signal_diagnostic_compares_rank_to_bias_baseline() -> None:
    head_targets = torch.zeros((8, len(QUERY_LOCAL_UTILITY_HEAD_NAMES)), dtype=torch.float32)
    head_mask = torch.zeros_like(head_targets, dtype=torch.bool)
    behavior_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("conditional_behavior_utility")
    head_targets[:, behavior_idx] = torch.tensor(
        [1.0, 0.9, 0.8, 0.7, 0.1, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
    head_mask[:, behavior_idx] = True
    target_mean = head_targets[:, behavior_idx].mean().clamp(1e-5, 1.0 - 1e-5)
    bias_logits = torch.zeros_like(head_targets)
    bias_logits[:, behavior_idx] = torch.logit(target_mean)
    aligned_logits = bias_logits.clone()
    aligned_logits[:, behavior_idx] = torch.linspace(4.0, -4.0, 8)

    bias_diagnostic = _behavior_head_training_signal_diagnostics(
        head_logits=bias_logits,
        factorized_targets=head_targets,
        factorized_mask=head_mask,
        boundaries=[(0, 8)],
        behavior_rank_loss_weight=0.25,
    )
    aligned_diagnostic = _behavior_head_training_signal_diagnostics(
        head_logits=aligned_logits,
        factorized_targets=head_targets,
        factorized_mask=head_mask,
        boundaries=[(0, 8)],
        behavior_rank_loss_weight=0.25,
    )

    assert bias_diagnostic["behavior_head_training_signal_available"] is True
    assert bias_diagnostic["rank_pair_count"] > 0
    assert bias_diagnostic["rank_pair_tie_fraction"] == pytest.approx(1.0)
    assert bias_diagnostic["classification"] == "rank_pressure_available_but_head_near_bias"
    assert aligned_diagnostic["rank_pair_accuracy"] == pytest.approx(1.0)
    assert aligned_diagnostic["rank_loss_improvement_vs_bias"] > 0.0
    assert aligned_diagnostic["classification"] == "behavior_head_training_signal_partially_learned"


def test_prior_feature_learning_diagnostic_localizes_invariant_heads() -> None:
    head_count = len(QUERY_LOCAL_UTILITY_HEAD_NAMES)
    prior = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.1, 0.0, 0.0, 0.2, 0.0],
            [0.4, 0.2, 0.1, 0.0, 0.4, 0.0],
            [0.6, 0.3, 0.1, 0.1, 0.6, 0.0],
            [0.8, 0.5, 0.2, 0.1, 0.8, 0.0],
            [1.0, 0.7, 0.3, 0.2, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    non_prior = torch.stack([prior[:, 0], torch.linspace(0.0, 1.0, prior.shape[0])], dim=1)
    norm_points = torch.cat([non_prior, prior], dim=1)
    head_targets = torch.zeros((prior.shape[0], head_count), dtype=torch.float32)
    head_mask = torch.ones_like(head_targets, dtype=torch.bool)
    query_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("query_hit_probability")
    behavior_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("conditional_behavior_utility")
    head_targets[:, query_idx] = prior[:, 0]
    head_targets[:, behavior_idx] = prior[:, 4]
    scalar_target = head_targets[:, query_idx].clone()
    invariant_logits = torch.zeros_like(head_targets)

    diagnostic = _prior_feature_learning_diagnostics(
        model=None,
        norm_points=norm_points,
        primary_predictions=torch.zeros((prior.shape[0],), dtype=torch.float32),
        zero_prior_predictions=torch.zeros((prior.shape[0],), dtype=torch.float32),
        primary_head_logits=invariant_logits,
        zero_prior_head_logits=invariant_logits,
        factorized_targets=head_targets,
        factorized_mask=head_mask,
        scalar_target=scalar_target,
        scalar_mask=torch.ones((prior.shape[0],), dtype=torch.bool),
        seed=11,
    )

    assert diagnostic["prior_feature_learning_diagnostics_available"] is True
    assert diagnostic["prior_signal_head_count"] >= 2
    query_row = diagnostic["head_target_alignment"]["query_hit_probability"]
    assert query_row["best_prior_channel"]["feature_name"] == "spatial_query_hit_probability"
    assert query_row["best_prior_channel"]["value"] > 0.99
    sensitivity = diagnostic["zero_prior_sensitivity"]["head_probabilities"]
    assert sensitivity["mean_abs_head_probability_delta"] == pytest.approx(0.0)
    reconstruction = diagnostic["prior_reconstruction_from_non_prior_features"]
    assert reconstruction["per_prior_channel"]["spatial_query_hit_probability"]["r2"] > 0.99
    assert diagnostic["classification"] == (
        "prior_target_signal_available_but_trained_heads_invariant"
    )


def test_prior_feature_learning_diagnostic_reports_model_stage_sensitivity() -> None:
    head_count = len(QUERY_LOCAL_UTILITY_HEAD_NAMES)
    prior = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.1, 0.0, 0.0, 0.2, 0.0],
            [0.4, 0.2, 0.1, 0.0, 0.4, 0.0],
            [0.6, 0.3, 0.1, 0.1, 0.6, 0.0],
            [0.8, 0.5, 0.2, 0.1, 0.8, 0.0],
            [1.0, 0.7, 0.3, 0.2, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    norm_points = torch.cat(
        [torch.linspace(0.0, 1.0, prior.shape[0]).unsqueeze(1), prior],
        dim=1,
    )
    model = WorkloadBlindRangeModel(
        point_dim=int(norm_points.shape[1]),
        query_dim=0,
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    head_targets = torch.zeros((prior.shape[0], head_count), dtype=torch.float32)
    head_mask = torch.ones_like(head_targets, dtype=torch.bool)

    diagnostic = _prior_feature_learning_diagnostics(
        model=model,
        norm_points=norm_points,
        primary_predictions=torch.zeros((prior.shape[0],), dtype=torch.float32),
        zero_prior_predictions=torch.zeros((prior.shape[0],), dtype=torch.float32),
        primary_head_logits=torch.zeros_like(head_targets),
        zero_prior_head_logits=torch.zeros_like(head_targets),
        factorized_targets=head_targets,
        factorized_mask=head_mask,
        scalar_target=torch.zeros((prior.shape[0],), dtype=torch.float32),
        scalar_mask=torch.ones((prior.shape[0],), dtype=torch.bool),
        boundaries=[(0, int(prior.shape[0]))],
        window_length=int(prior.shape[0]),
        window_stride=int(prior.shape[0]),
        batch_size=1,
        seed=13,
    )

    stage = diagnostic["prior_stage_sensitivity"]
    assert stage["available"] is True
    assert stage["stage_sensitivity"]["pre_context_sum"]["mean_abs_delta"] > 0.0
    assert stage["stage_sensitivity"]["shared_embedding"]["mean_abs_delta"] > 0.0
    assert stage["stage_sensitivity"]["head_probabilities"]["mean_abs_delta"] > 0.0
    transfer = diagnostic["prior_to_head_transfer_sensitivity"]
    assert transfer["available"] is True
    assert "query_hit_probability" in transfer["per_head"]
    query_transfer = transfer["per_head"]["query_hit_probability"]
    assert query_transfer["available"] is True
    assert query_transfer["stage_sensitivity"]["shared_embedding"]["mean_abs_delta"] > 0.0
    assert query_transfer["stage_sensitivity"]["first_linear"]["mean_abs_delta"] > 0.0
    assert query_transfer["stage_sensitivity"]["logit"]["mean_abs_delta"] > 0.0
    assert query_transfer["stage_sensitivity"]["probability"]["mean_abs_delta"] > 0.0
    assert query_transfer["first_linear_delta_l2_to_shared_delta_l2"] is not None
    assert query_transfer["probability_mean_abs_delta_to_logit_mean_abs_delta"] is not None
    assert query_transfer["sigmoid_derivative_mean"] is not None
    alignment = query_transfer["output_layer_alignment"]
    assert alignment["available"] is True
    assert alignment["valid_aligned_point_count"] == prior.shape[0]
    assert alignment["final_weight_to_hidden_delta_abs_cosine_mean"] is not None
    assert alignment["bce_descent_alignment_positive_fraction"] is not None
    slice_alignment = alignment["slice_alignment"]
    assert slice_alignment["available"] is True
    assert "window_slice" in slice_alignment["groups"]
    loss_alignment = query_transfer["configured_loss_gradient_alignment"]
    assert loss_alignment["available"] is True
    assert loss_alignment["aligned_point_count"] == prior.shape[0]
    assert loss_alignment["descent_alignment_positive_fraction"] is not None
    channel_decomposition = transfer["prior_channel_direction_decomposition"]
    assert channel_decomposition["available"] is True
    assert channel_decomposition["channel_count"] == len(QUERY_PRIOR_FIELD_NAMES)
    assert "query_hit_probability" in channel_decomposition["by_head"]
    query_hit_by_head = channel_decomposition["by_head"]["query_hit_probability"]
    assert query_hit_by_head["channel_count"] == len(QUERY_PRIOR_FIELD_NAMES)
    channel_name = "spatial_query_hit_probability"
    assert channel_name in channel_decomposition["per_channel"]
    channel_query_transfer = channel_decomposition["per_channel"][channel_name]["per_head"][
        "query_hit_probability"
    ]
    assert channel_query_transfer["available"] is True
    assert channel_query_transfer["classification"] in {
        "target_aligned",
        "wrong_way",
        "weak_or_flat",
        "rank_alignment_unavailable",
    }
    channel_alignment = channel_query_transfer["output_layer_alignment"]
    assert channel_alignment["available"] is True
    assert channel_alignment["valid_aligned_point_count"] == prior.shape[0]
    assert channel_alignment["slice_alignment"]["available"] is True


def test_prior_output_layer_alignment_diagnostic_reports_projection_and_bce_direction() -> None:
    primary_hidden = torch.tensor(
        [[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    ablation_hidden = torch.zeros_like(primary_hidden)
    final_weight = torch.tensor([1.0, 0.0], dtype=torch.float32)
    primary_logit = primary_hidden @ final_weight
    target = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)

    diagnostic = _prior_output_layer_alignment_diagnostics(
        primary_hidden_parts=[primary_hidden],
        ablation_hidden_parts=[ablation_hidden],
        primary_logit_parts=[primary_logit.unsqueeze(1)],
        ablation_logit_parts=[torch.zeros_like(primary_logit).unsqueeze(1)],
        primary_probability_parts=[torch.sigmoid(primary_logit).unsqueeze(1)],
        target_parts=[target],
        mask_parts=[torch.ones_like(target, dtype=torch.bool)],
        final_weight=final_weight,
        slice_mask_parts={
            "window_slice": {
                "window_start": [torch.tensor([True, True, True, True])],
                "window_end": [torch.tensor([False, False, True, True])],
            }
        },
    )

    assert diagnostic["available"] is True
    assert diagnostic["valid_aligned_point_count"] == 4
    assert diagnostic["final_weight_to_hidden_delta_abs_cosine_mean"] == pytest.approx(0.75)
    assert diagnostic["projected_hidden_delta_l2_to_hidden_delta_l2"] == pytest.approx(
        (6.0 / 7.0) ** 0.5
    )
    assert diagnostic["target_to_logit_delta_spearman"] is not None
    assert diagnostic["target_to_logit_delta_spearman"] > 0.0
    assert diagnostic["bce_descent_alignment_mean"] > 0.0
    assert diagnostic["bce_descent_alignment_positive_fraction"] == pytest.approx(0.75)
    slice_alignment = diagnostic["slice_alignment"]
    assert slice_alignment["available"] is True
    start = slice_alignment["groups"]["window_slice"]["window_start"]
    assert start["available"] is True
    assert start["target_to_logit_delta_spearman"] is not None
    assert start["target_top_quartile_minus_bottom_quartile_logit_delta"] is not None


def test_sparse_head_rank_loss_penalizes_reversed_tiny_query_and_boundary_targets() -> None:
    head_targets = torch.zeros((1, 8, len(QUERY_LOCAL_UTILITY_HEAD_NAMES)), dtype=torch.float32)
    head_mask = torch.zeros_like(head_targets, dtype=torch.bool)
    query_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("query_hit_probability")
    boundary_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("boundary_event_utility")
    tiny_order = torch.tensor(
        [0.0010, 0.0008, 0.0006, 0.0004, 0.0001, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
    head_targets[0, :, query_idx] = tiny_order
    head_targets[0, :, boundary_idx] = tiny_order * 0.1
    head_mask[0, :, query_idx] = True
    head_mask[0, :, boundary_idx] = True
    aligned_logits = torch.zeros_like(head_targets)
    reversed_logits = torch.zeros_like(head_targets)
    aligned_logits[0, :, query_idx] = torch.linspace(4.0, -4.0, 8)
    aligned_logits[0, :, boundary_idx] = torch.linspace(4.0, -4.0, 8)
    reversed_logits[0, :, query_idx] = torch.linspace(-4.0, 4.0, 8)
    reversed_logits[0, :, boundary_idx] = torch.linspace(-4.0, 4.0, 8)

    aligned = _sparse_head_rank_loss(
        head_logits=aligned_logits,
        head_targets=head_targets,
        head_mask=head_mask,
    )
    reversed_loss = _sparse_head_rank_loss(
        head_logits=reversed_logits,
        head_targets=head_targets,
        head_mask=head_mask,
    )
    without_sparse_rank = _factorized_query_local_utility_loss(
        head_logits=reversed_logits,
        head_targets=head_targets,
        head_mask=head_mask,
        sparse_head_rank_loss_weight=0.0,
    )
    with_sparse_rank = _factorized_query_local_utility_loss(
        head_logits=reversed_logits,
        head_targets=head_targets,
        head_mask=head_mask,
        sparse_head_rank_loss_weight=1.0,
    )

    assert float(aligned.item()) < float(reversed_loss.item())
    assert float(with_sparse_rank.item()) > float(without_sparse_rank.item())


def test_sparse_head_bce_target_calibration_rescales_tiny_query_and_boundary_heads() -> None:
    head_targets = torch.zeros((1, 4, len(QUERY_LOCAL_UTILITY_HEAD_NAMES)), dtype=torch.float32)
    head_mask = torch.zeros_like(head_targets, dtype=torch.bool)
    query_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("query_hit_probability")
    boundary_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("boundary_event_utility")
    behavior_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("conditional_behavior_utility")
    head_targets[0, :, query_idx] = torch.tensor([0.0010, 0.0005, 0.0, 0.0])
    head_targets[0, :, boundary_idx] = torch.tensor([0.000010, 0.000005, 0.0, 0.0])
    head_targets[0, :, behavior_idx] = torch.tensor([0.20, 0.40, 0.60, 0.80])
    head_mask[0, :, query_idx] = True
    head_mask[0, :, boundary_idx] = True
    head_mask[0, :, behavior_idx] = True

    raw = _calibrated_sparse_head_bce_targets(
        head_targets=head_targets,
        head_mask=head_mask,
        mode="raw",
    )
    calibrated = _calibrated_sparse_head_bce_targets(
        head_targets=head_targets,
        head_mask=head_mask,
        mode="window_max_normalized",
    )

    assert torch.allclose(raw, head_targets)
    assert calibrated[0, :, query_idx].tolist() == pytest.approx([1.0, 0.5, 0.0, 0.0])
    assert calibrated[0, :, boundary_idx].tolist() == pytest.approx([1.0, 0.5, 0.0, 0.0])
    assert torch.allclose(calibrated[0, :, behavior_idx], head_targets[0, :, behavior_idx])


def test_sparse_head_bce_target_calibration_makes_aligned_tiny_heads_cheaper() -> None:
    head_targets = torch.zeros((1, 8, len(QUERY_LOCAL_UTILITY_HEAD_NAMES)), dtype=torch.float32)
    head_mask = torch.zeros_like(head_targets, dtype=torch.bool)
    query_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("query_hit_probability")
    boundary_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("boundary_event_utility")
    tiny_order = torch.tensor(
        [0.0010, 0.0008, 0.0006, 0.0004, 0.0001, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
    head_targets[0, :, query_idx] = tiny_order
    head_targets[0, :, boundary_idx] = tiny_order * 0.1
    head_mask[0, :, query_idx] = True
    head_mask[0, :, boundary_idx] = True
    aligned_logits = torch.zeros_like(head_targets)
    reversed_logits = torch.zeros_like(head_targets)
    aligned_logits[0, :, query_idx] = torch.linspace(4.0, -4.0, 8)
    aligned_logits[0, :, boundary_idx] = torch.linspace(4.0, -4.0, 8)
    reversed_logits[0, :, query_idx] = torch.linspace(-4.0, 4.0, 8)
    reversed_logits[0, :, boundary_idx] = torch.linspace(-4.0, 4.0, 8)

    raw_aligned = _factorized_query_local_utility_loss(
        head_logits=aligned_logits,
        head_targets=head_targets,
        head_mask=head_mask,
    )
    raw_reversed = _factorized_query_local_utility_loss(
        head_logits=reversed_logits,
        head_targets=head_targets,
        head_mask=head_mask,
    )
    calibrated_aligned = _factorized_query_local_utility_loss(
        head_logits=aligned_logits,
        head_targets=head_targets,
        head_mask=head_mask,
        sparse_head_bce_target_mode="window_max_normalized",
    )
    calibrated_reversed = _factorized_query_local_utility_loss(
        head_logits=reversed_logits,
        head_targets=head_targets,
        head_mask=head_mask,
        sparse_head_bce_target_mode="window_max_normalized",
    )

    raw_gap = float((raw_reversed - raw_aligned).abs().item())
    calibrated_gap = float((calibrated_reversed - calibrated_aligned).item())
    assert raw_gap < 0.01
    assert float(calibrated_aligned.item()) < float(calibrated_reversed.item())
    assert calibrated_gap > raw_gap * 100.0


def test_factorized_head_fit_diagnostics_reports_each_head() -> None:
    head_targets = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.1, 0.0, 1.0],
            [0.2, 0.1, 0.0, 0.2, 0.2, 0.8],
            [0.4, 0.3, 0.2, 0.4, 0.4, 0.6],
            [0.6, 0.6, 0.4, 0.6, 0.6, 0.4],
            [0.8, 0.8, 0.6, 0.8, 0.8, 0.2],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    head_logits = torch.logit(head_targets.clamp(1e-4, 1.0 - 1e-4))
    head_mask = torch.ones_like(head_targets, dtype=torch.bool)
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.1, 0.0, 1.0, 0.0],
            [2.0, 0.2, 0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0, 1.0, 0.0],
            [4.0, 2.0, 0.0, 1.0, 0.0],
            [5.0, 3.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    typed_queries = [
        {
            "type": "range",
            "params": {
                "t_start": -1.0,
                "t_end": 6.0,
                "lat_min": -0.1,
                "lat_max": 0.25,
                "lon_min": -1.0,
                "lon_max": 1.0,
            },
            "_metadata": {
                "anchor_family": "density",
                "footprint_family": "medium_operational",
            },
        }
    ]

    diagnostics = _factorized_head_fit_diagnostics(
        head_logits=head_logits,
        factorized_targets=head_targets,
        factorized_mask=head_mask,
        points=points,
        boundaries=[(0, 3), (3, 6)],
        typed_queries=typed_queries,
        seed=19,
    )
    behavior = diagnostics["factorized_head_fit"]["conditional_behavior_utility"]

    assert diagnostics["factorized_head_fit_diagnostics_available"] is True
    assert set(diagnostics["factorized_head_fit"]) == set(QUERY_LOCAL_UTILITY_HEAD_NAMES)
    assert behavior["available"] is True
    assert behavior["valid_point_count"] == 6
    assert behavior["positive_target_count"] == 5
    assert behavior["kendall_tau"] > 0.99
    assert behavior["topk_mass_recall_at_5_percent"] == 1.0
    assert diagnostics["conditional_behavior_utility_head_tau"] == behavior["kendall_tau"]
    family_fit = diagnostics["family_conditioned_head_trainability"]
    assert family_fit["available"] is True
    assert family_fit["diagnostic_only"] is True
    density = family_fit["group_by"]["anchor_family"]["density"]
    assert density["focus_family"] is True
    assert density["query_count"] == 1
    assert density["valid_hit_point_count"] == 4
    assert density["head_fit"]["segment_budget_target"]["available"] is True
    assert density["factorized_composed_score_fit"]["available"] is True


def test_factorized_final_score_composition_diagnostics_match_scalar_target() -> None:
    point_count = 8
    head_targets = torch.zeros(
        (point_count, len(QUERY_LOCAL_UTILITY_HEAD_NAMES)), dtype=torch.float32
    )
    head_targets[:, 0] = torch.linspace(0.20, 0.90, steps=point_count)
    head_targets[:, 1] = torch.linspace(0.10, 0.80, steps=point_count)
    head_targets[:, 2] = torch.linspace(0.00, 0.35, steps=point_count)
    head_targets[:, 3] = torch.linspace(0.10, 0.90, steps=point_count)
    head_targets[:, 4] = torch.linspace(0.00, 1.00, steps=point_count)
    head_targets[:, 5] = torch.linspace(1.00, 0.00, steps=point_count)
    scalar_target = query_local_utility_point_score(
        q_hit=head_targets[:, 0],
        behavior=head_targets[:, 1],
        boundary=head_targets[:, 2],
        replacement=head_targets[:, 3],
    )
    head_logits = torch.logit(head_targets.clamp(1e-4, 1.0 - 1e-4))

    diagnostics = _factorized_final_score_composition_diagnostics(
        head_logits=head_logits,
        factorized_targets=head_targets,
        scalar_target=scalar_target,
        scalar_mask=torch.ones((point_count,), dtype=torch.bool),
        seed=23,
    )

    assert diagnostics["factorized_final_score_composition_available"] is True
    assert diagnostics["factorized_final_score_formula"] == QUERY_LOCAL_UTILITY_FINAL_LABEL_FORMULA
    assert diagnostics["factorized_final_score_tau"] > 0.99
    assert diagnostics["factorized_final_score_topk_mass_recall_at_5_percent"] == pytest.approx(1.0)
    assert diagnostics["factorized_final_score_prediction_std_to_target_std"] == pytest.approx(
        1.0, abs=1e-4
    )
    assert diagnostics["factorized_target_formula_label_mae"] < 1e-5
    assert diagnostics["factorized_target_formula_topk_mass_recall_at_5_percent"] == pytest.approx(
        1.0
    )
    assert diagnostics["factorized_replacement_multiplier_mean"] > 0.75


def test_factorized_scalar_training_target_keeps_raw_query_local_utility_scale() -> None:
    labels = torch.tensor([[0.0], [0.01], [0.02], [0.10]], dtype=torch.float32)
    labelled_mask = torch.ones_like(labels, dtype=torch.bool)

    factorized_target, factorized_basis = _scalar_training_target_for_mode(
        labels=labels,
        labelled_mask=labelled_mask,
        workload_type_id=0,
        range_training_target_mode="query_local_utility_factorized",
    )
    legacy_target, legacy_basis = _scalar_training_target_for_mode(
        labels=labels,
        labelled_mask=labelled_mask,
        workload_type_id=0,
        range_training_target_mode="point_value",
    )

    assert factorized_basis == "raw_query_local_utility_final_label_for_loss"
    assert legacy_basis == "scaled_training_target_for_loss"
    assert torch.allclose(factorized_target, labels[:, 0])
    assert float(legacy_target[-1].item()) == pytest.approx(1.0)
    assert not torch.allclose(legacy_target, factorized_target)


def test_factorized_head_bias_initialization_uses_training_base_rates() -> None:
    model = WorkloadBlindRangeModel(
        point_dim=WORKLOAD_BLIND_RANGE_POINT_DIM,
        query_dim=0,
        embed_dim=16,
        num_heads=2,
        num_layers=0,
        dropout=0.0,
    )
    head_targets = torch.tensor(
        [
            [0.0010, 0.10, 0.0000, 0.20, 0.05, 0.30],
            [0.0020, 0.20, 0.0001, 0.10, 0.15, 0.10],
            [0.0000, 0.00, 0.0000, 0.00, 0.10, 0.40],
            [0.0010, 0.30, 0.0000, 0.30, 0.20, 0.20],
        ],
        dtype=torch.float32,
    )
    head_mask = torch.ones_like(head_targets, dtype=torch.bool)
    behavior_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("conditional_behavior_utility")
    head_mask[2, behavior_idx] = False

    diagnostics = _initialize_factorized_head_output_biases_from_targets(
        model,
        head_targets=head_targets,
        head_mask=head_mask,
    )

    assert diagnostics["available"] is True
    for head_idx, head_name in enumerate(QUERY_LOCAL_UTILITY_HEAD_NAMES):
        valid = head_mask[:, head_idx]
        target_mean = float(head_targets[:, head_idx][valid].mean().item())
        probability = max(1e-4, min(1.0 - 1e-4, target_mean))
        expected_bias = float(torch.logit(torch.tensor(probability)).item())
        head_module = cast(torch.nn.Sequential, model.heads[head_name])
        final_linear = cast(torch.nn.Linear, head_module[-1])

        assert diagnostics["heads"][head_name]["target_mean"] == pytest.approx(target_mean)
        assert diagnostics["heads"][head_name]["bias"] == pytest.approx(expected_bias)
        assert final_linear.bias is not None
        assert float(final_linear.bias.item()) == pytest.approx(expected_bias)


def test_factorized_training_diagnostics_do_not_claim_legacy_scalar_target() -> None:
    labels = torch.tensor([[1.0], [0.0], [0.5]], dtype=torch.float32)
    labelled_mask = torch.ones_like(labels, dtype=torch.bool)

    diagnostics = _training_target_diagnostics(
        labels=labels,
        labelled_mask=labelled_mask,
        workload_type_id=0,
        configured_budget_ratios=(0.1,),
        effective_budget_ratios=(0.1,),
        temporal_residual_budget_masks=(),
        temporal_residual_label_mode="none",
        loss_objective="budget_topk",
        temporal_fraction=0.0,
        range_training_target_mode="query_local_utility_factorized",
    )

    assert diagnostics["target_family"] == "QueryLocalUtilityFactorized"
    assert diagnostics["final_success_allowed"] is True
    assert "legacy_reason" not in diagnostics
