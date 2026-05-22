"""Query-driven protocol gate and model-feature tests."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from data_preparation.ais_loader import generate_synthetic_ais_data
from learning.model_features import (
    WORKLOAD_BLIND_RANGE_MODEL_DISABLED_PRIOR_FIELDS,
    WORKLOAD_BLIND_RANGE_POINT_DIM,
    build_workload_blind_range_point_features,
)
from learning.model_training import (
    _fit_scaler_for_model,
)
from learning.query_prior_fields import (
    QUERY_PRIOR_FIELD_NAMES,
    build_train_query_prior_fields,
    query_prior_field_metadata,
    sample_query_prior_fields,
    zero_query_prior_field_channels,
    zero_query_prior_field_like,
)
from learning.targets.query_local_utility import (
    QUERY_LOCAL_UTILITY_HEAD_NAMES,
    query_local_utility_point_score,
)
from models.workload_blind_range import WorkloadBlindRangeModel
from orchestration.gates import (
    evaluate_support_overlap_gate,
    evaluate_target_diffusion_gate,
)
from orchestration.model_ablations import reset_module_parameters
from orchestration.range_diagnostics import range_workload_distribution_comparison
from selection.learned_segment_budget import (
    learned_segment_budget_diagnostics,
    simplify_with_learned_segment_budget,
    simplify_with_learned_segment_budget_with_trace,
)
from workloads.query_types import QUERY_TYPE_ID_RANGE

HISTORICAL_SMALL_LOCAL_FAMILY = "small_local"


def _boundaries(trajectories: list[torch.Tensor]) -> list[tuple[int, int]]:
    cursor = 0
    out = []
    for trajectory in trajectories:
        end = cursor + int(trajectory.shape[0])
        out.append((cursor, end))
        cursor = end
    return out


def test_target_diffusion_gate_blocks_broad_low_budget_labels() -> None:
    diagnostics = {
        "query_local_utility_factorized": {
            "final_label_support_fraction_by_threshold": {"gt_0.01": 0.80},
            "support_fraction_by_threshold_by_head": {
                "query_hit_probability": {"gt_0.01": 0.70},
                "conditional_behavior_utility": {"gt_0.01": 0.70},
                "replacement_representative_value": {"gt_0.05": 0.20},
            },
            "topk_label_mass_budget_grid": {
                "query_hit_probability": {"0.05": 0.08},
                "conditional_behavior_utility": {"0.05": 0.08},
                "replacement_representative_value": {"0.05": 0.35},
            },
        }
    }

    gate = evaluate_target_diffusion_gate(diagnostics)

    assert gate["gate_pass"] is False
    assert "final_label_support_fraction_above_max" in gate["failed_checks"]
    assert "conditional_behavior_utility:support_fraction_above_max" in gate["failed_checks"]
    assert "conditional_behavior_utility:top5_label_mass_below_min" in gate["failed_checks"]
    assert "query_hit_probability:support_fraction_above_max" not in gate["failed_checks"]


def test_target_diffusion_gate_accepts_concentrated_factorized_labels() -> None:
    diagnostics = {
        "query_local_utility_factorized": {
            "final_label_support_fraction_by_threshold": {"gt_0.01": 0.30},
            "support_fraction_by_threshold_by_head": {
                "query_hit_probability": {"gt_0.01": 0.35},
                "conditional_behavior_utility": {"gt_0.01": 0.20},
                "replacement_representative_value": {"gt_0.05": 0.20},
            },
            "topk_label_mass_budget_grid": {
                "query_hit_probability": {"0.05": 0.25},
                "conditional_behavior_utility": {"0.05": 0.35},
                "replacement_representative_value": {"0.05": 0.35},
            },
        }
    }

    gate = evaluate_target_diffusion_gate(diagnostics)

    assert gate["gate_pass"] is True
    assert gate["failed_checks"] == []


def test_prior_behavior_field_uses_behavior_values_not_hit_probability() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    behavior_values = torch.tensor([0.9, 0.2, 0.7], dtype=torch.float32)
    labels = torch.zeros((3, QUERY_TYPE_ID_RANGE + 1), dtype=torch.float32)
    labels[:, QUERY_TYPE_ID_RANGE] = 1.0
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 3.0,
            "lat_min": -1.0,
            "lat_max": 3.0,
            "lon_min": -1.0,
            "lon_max": 1.0,
        },
    }

    prior = build_train_query_prior_fields(
        points=points,
        boundaries=[(0, 3)],
        typed_queries=[query],
        labels=labels,
        behavior_values=behavior_values,
        workload_profile_id="range_query_mix",
        grid_bins=4,
        smoothing_passes=0,
    )
    features = build_workload_blind_range_point_features(points, prior)

    spatial_query_hit_probability = features[:, -6]
    behavior_utility_prior = features[:, -2]
    assert torch.allclose(
        spatial_query_hit_probability, torch.ones_like(spatial_query_hit_probability)
    )
    assert torch.allclose(behavior_utility_prior, behavior_values)
    assert not torch.allclose(behavior_utility_prior, spatial_query_hit_probability)


def test_workload_blind_range_scaler_preserves_semantic_zero_for_prior_ablation() -> None:
    points = torch.zeros((3, WORKLOAD_BLIND_RANGE_POINT_DIM), dtype=torch.float32)
    points[:, -6:] = torch.tensor(
        [
            [0.20, 0.10, 0.01, 0.02, 0.30, 0.50],
            [0.25, 0.20, 0.02, 0.03, 0.40, 0.60],
            [0.30, 0.30, 0.03, 0.04, 0.50, 0.70],
        ],
        dtype=torch.float32,
    )
    queries = torch.zeros((1, 12), dtype=torch.float32)

    scaler = _fit_scaler_for_model(points, queries, "workload_blind_range")
    zero_prior_points = points.clone()
    zero_prior_points[:, -6:] = 0.0
    transformed = scaler.transform_points(zero_prior_points)

    assert torch.allclose(scaler.point_min[-6:], torch.zeros(6))
    assert torch.allclose(scaler.point_max[-6:], torch.ones(6))
    assert torch.allclose(transformed[:, -6:], torch.zeros((3, 6)))


def test_query_prior_field_rasterizes_query_boxes_not_only_hit_points() -> None:
    train_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    eval_points = torch.tensor([[0.5, 2.0, 2.0], [0.5, 4.0, 4.0]], dtype=torch.float32)
    query = {
        "type": "range",
        "params": {
            "t_start": 0.0,
            "t_end": 1.0,
            "lat_min": 1.5,
            "lat_max": 2.5,
            "lon_min": 1.5,
            "lon_max": 2.5,
        },
    }

    prior = build_train_query_prior_fields(
        points=train_points,
        boundaries=[(0, 1)],
        typed_queries=[query],
        workload_profile_id="range_query_mix",
        grid_bins=8,
        time_bins=4,
        smoothing_passes=0,
    )
    sampled = sample_query_prior_fields(eval_points, prior)

    assert prior["spatial_query_field_source"] == "train_query_box_density"
    assert prior["out_of_extent_sampling"] == "zero"
    assert prior["diagnostics"]["raw_nonzero_point_hit_cells"] == 0
    assert prior["diagnostics"]["raw_nonzero_spatial_query_cells"] > 0
    assert float(sampled[0, 0].item()) > 0.0
    assert float(sampled[0, 1].item()) > 0.0
    assert torch.allclose(sampled[1], torch.zeros_like(sampled[1]))


def test_sample_query_prior_fields_nearest_mode_clamps_out_of_extent_points() -> None:
    train_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    eval_points = torch.tensor(
        [[0.5, 5.0, 5.0], [0.25, -2.0, -3.0]],
        dtype=torch.float32,
    )
    query = {
        "type": "range",
        "params": {
            "t_start": 0.0,
            "t_end": 1.0,
            "lat_min": 0.0,
            "lat_max": 1.0,
            "lon_min": 0.0,
            "lon_max": 1.0,
        },
    }
    prior = build_train_query_prior_fields(
        points=train_points,
        boundaries=[(0, 2)],
        typed_queries=[query],
        workload_profile_id="range_query_mix",
        grid_bins=8,
        time_bins=2,
        smoothing_passes=0,
        out_of_extent_sampling="nearest",
    )
    sampled_nearest = sample_query_prior_fields(eval_points, prior)
    sampled_zero = sample_query_prior_fields(
        eval_points,
        dict(prior, out_of_extent_sampling="zero"),
    )

    assert prior["out_of_extent_sampling"] == "nearest"
    assert bool((sampled_nearest.abs().sum(dim=1) > 0.0).all().item())
    assert torch.allclose(sampled_zero, torch.zeros_like(sampled_zero))


def test_zero_prior_field_like_preserves_metadata_and_shape() -> None:
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 2.0,
            "lat_min": -1.0,
            "lat_max": 2.0,
            "lon_min": -1.0,
            "lon_max": 2.0,
        },
    }
    prior = build_train_query_prior_fields(
        points=points,
        boundaries=[(0, 2)],
        typed_queries=[query],
        workload_profile_id="range_query_mix",
        grid_bins=4,
        time_bins=2,
        smoothing_passes=0,
    )

    zeroed = zero_query_prior_field_like(prior)

    assert zeroed["extent"] == prior["extent"]
    assert zeroed["grid_bins"] == prior["grid_bins"]
    assert zeroed["time_bins"] == prior["time_bins"]
    assert zeroed["ablation"] == "zero_query_prior_features"
    assert query_prior_field_metadata(zeroed)["contains_eval_queries"] is False
    for name in zeroed["field_names"]:
        assert zeroed[name].shape == prior[name].shape
        assert torch.count_nonzero(zeroed[name]).item() == 0


def test_zero_query_prior_field_channels_only_zeros_requested_channels() -> None:
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 2.0,
            "lat_min": -1.0,
            "lat_max": 2.0,
            "lon_min": -1.0,
            "lon_max": 2.0,
        },
    }
    prior = build_train_query_prior_fields(
        points=points,
        boundaries=[(0, 2)],
        typed_queries=[query],
        workload_profile_id="range_query_mix",
        grid_bins=4,
        time_bins=2,
        smoothing_passes=0,
    )

    zeroed = zero_query_prior_field_channels(prior, ["route_density_prior"])

    assert zeroed["extent"] == prior["extent"]
    assert zeroed["ablation"] == "zero_query_prior_channels"
    assert zeroed["zeroed_prior_channels"] == ["route_density_prior"]
    assert query_prior_field_metadata(zeroed)["contains_eval_queries"] is False
    assert torch.count_nonzero(zeroed["route_density_prior"]).item() == 0
    for name in zeroed["field_names"]:
        if name == "route_density_prior":
            continue
        assert torch.equal(zeroed[name], prior[name])


def test_no_query_prior_ablation_preserves_train_extent() -> None:
    train_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 10.0, 10.0]], dtype=torch.float32)
    eval_points = train_points.clone()
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 2.0,
            "lat_min": -1.0,
            "lat_max": 11.0,
            "lon_min": -1.0,
            "lon_max": 11.0,
        },
    }
    prior = build_train_query_prior_fields(
        points=train_points,
        boundaries=[(0, 2)],
        typed_queries=[query],
        workload_profile_id="range_query_mix",
        grid_bins=4,
        time_bins=2,
        smoothing_passes=0,
    )
    zeroed = zero_query_prior_field_like(prior)

    with_prior = build_workload_blind_range_point_features(eval_points, prior)
    no_prior = build_workload_blind_range_point_features(eval_points, zeroed)
    without_field = build_workload_blind_range_point_features(eval_points, None)

    assert with_prior.shape == no_prior.shape == without_field.shape
    assert torch.allclose(with_prior[:, :-6], no_prior[:, :-6])
    assert not torch.allclose(no_prior[:, :-6], without_field[:, :-6])
    assert torch.count_nonzero(no_prior[:, -6:]).item() == 0


def test_workload_blind_range_excludes_route_density_from_model_features() -> None:
    train_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 10.0, 10.0]], dtype=torch.float32)
    eval_points = train_points.clone()
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 2.0,
            "lat_min": -1.0,
            "lat_max": 11.0,
            "lon_min": -1.0,
            "lon_max": 11.0,
        },
    }
    prior = build_train_query_prior_fields(
        points=train_points,
        boundaries=[(0, 2)],
        typed_queries=[query],
        workload_profile_id="range_query_mix",
        grid_bins=4,
        time_bins=2,
        smoothing_passes=0,
    )
    route_idx = list(QUERY_PRIOR_FIELD_NAMES).index("route_density_prior")

    sampled = sample_query_prior_fields(eval_points, prior)
    features = build_workload_blind_range_point_features(eval_points, prior)

    assert WORKLOAD_BLIND_RANGE_MODEL_DISABLED_PRIOR_FIELDS == ("route_density_prior",)
    assert torch.count_nonzero(sampled[:, route_idx]).item() > 0
    prior_start = -len(QUERY_PRIOR_FIELD_NAMES)
    assert torch.count_nonzero(features[:, prior_start + route_idx]).item() == 0
    for idx, name in enumerate(QUERY_PRIOR_FIELD_NAMES):
        if name == "route_density_prior":
            continue
        assert torch.equal(features[:, prior_start + idx], sampled[:, idx])


def test_support_overlap_gate_passes_same_support_eval_points() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.5, 0.5],
            [2.0, 1.0, 1.0],
            [3.0, 0.25, 0.75],
        ],
        dtype=torch.float32,
    )
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 4.0,
            "lat_min": -0.5,
            "lat_max": 1.5,
            "lon_min": -0.5,
            "lon_max": 1.5,
        },
    }
    prior = build_train_query_prior_fields(
        points=points,
        boundaries=[(0, 4)],
        typed_queries=[query],
        workload_profile_id="range_query_mix",
        grid_bins=4,
        time_bins=2,
        smoothing_passes=0,
    )

    gate = evaluate_support_overlap_gate(
        train_points=points, eval_points=points, query_prior_field=prior
    )

    assert gate["gate_pass"] is True
    assert gate["failed_checks"] == []
    assert gate["eval_points_outside_train_prior_extent_fraction"] == 0.0
    assert gate["sampled_prior_nonzero_fraction"] >= 0.50


def test_support_overlap_gate_blocks_out_of_extent_eval_points() -> None:
    train_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    eval_points = torch.tensor([[0.0, 10.0, 10.0], [1.0, 11.0, 11.0]], dtype=torch.float32)
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 2.0,
            "lat_min": -0.5,
            "lat_max": 1.5,
            "lon_min": -0.5,
            "lon_max": 1.5,
        },
    }
    prior = build_train_query_prior_fields(
        points=train_points,
        boundaries=[(0, 2)],
        typed_queries=[query],
        workload_profile_id="range_query_mix",
        grid_bins=4,
        time_bins=2,
        smoothing_passes=0,
    )

    gate = evaluate_support_overlap_gate(
        train_points=train_points, eval_points=eval_points, query_prior_field=prior
    )

    assert gate["gate_pass"] is False
    assert "eval_points_outside_train_prior_extent_too_high" in gate["failed_checks"]
    assert "sampled_prior_nonzero_fraction_too_low" in gate["failed_checks"]


def test_workload_signature_gate_rejects_profile_mismatch_and_tiny_query_counts() -> None:
    summaries = {
        "train": {
            "range": {"range_query_count": 4},
            "range_signal": {},
            "generation": {
                "workload_signature": {
                    "profile_id": "legacy_generator",
                    "query_count": 4,
                    "anchor_family_counts": {"density": 4},
                    "footprint_family_counts": {"medium_operational": 4},
                    "point_hit_counts_per_query": [1, 2, 3, 4],
                    "ship_hit_counts_per_query": [1, 1, 2, 2],
                    "near_duplicate_rate": 0.0,
                    "broad_query_rate": 0.0,
                }
            },
        },
        "eval": {
            "range": {"range_query_count": 4},
            "range_signal": {},
            "generation": {
                "workload_signature": {
                    "profile_id": "range_query_mix",
                    "query_count": 4,
                    "anchor_family_counts": {"density": 4},
                    "footprint_family_counts": {"medium_operational": 4},
                    "point_hit_counts_per_query": [1, 2, 3, 4],
                    "ship_hit_counts_per_query": [1, 1, 2, 2],
                    "near_duplicate_rate": 0.0,
                    "broad_query_rate": 0.0,
                }
            },
        },
    }

    gate = range_workload_distribution_comparison(summaries)["workload_signature_gate"]["pairs"][
        "train"
    ]

    assert gate["gate_pass"] is False
    assert "profile_id_mismatch" in gate["failed_checks"]
    assert "train_signature_query_count_below_min" in gate["failed_checks"]
    assert "eval_signature_query_count_below_min" in gate["failed_checks"]


def test_workload_signature_gate_rejects_query_count_mismatch() -> None:
    def signature(query_count: int) -> dict[str, Any]:
        return {
            "profile_id": "range_query_mix",
            "query_count": query_count,
            "anchor_family_counts": {"density": query_count},
            "footprint_family_counts": {"medium_operational": query_count},
            "point_hit_counts_per_query": [3 for _ in range(query_count)],
            "ship_hit_counts_per_query": [1 for _ in range(query_count)],
            "near_duplicate_rate": 0.0,
            "broad_query_rate": 0.0,
        }

    summaries = {
        "train": {
            "range": {"range_query_count": 8},
            "range_signal": {},
            "generation": {"workload_signature": signature(8)},
        },
        "eval": {
            "range": {"range_query_count": 12},
            "range_signal": {},
            "generation": {"workload_signature": signature(12)},
        },
    }

    gate = range_workload_distribution_comparison(summaries)["workload_signature_gate"]["pairs"][
        "train"
    ]

    assert gate["gate_pass"] is False
    assert gate["metrics"]["query_count_delta"] == 4
    assert gate["metrics"]["query_count_relative_delta"] == pytest.approx(4 / 12)
    assert gate["thresholds"]["query_count_relative_delta_max"] == 0.15
    assert "query_count_mismatch" in gate["failed_checks"]


def test_workload_signature_gate_treats_calibrated_query_count_as_diagnostic() -> None:
    def signature(query_count: int) -> dict[str, Any]:
        return {
            "profile_id": "range_query_mix_local",
            "target_coverage": 0.10,
            "coverage_actual": 0.10,
            "query_count_mode": "calibrated_to_coverage",
            "coverage_calibration_mode": "profile_sampled_query_count",
            "query_count": query_count,
            "anchor_family_counts": {"density": query_count},
            "footprint_family_counts": {"medium_operational": query_count},
            "point_hit_counts_per_query": [3 for _ in range(query_count)],
            "ship_hit_counts_per_query": [1 for _ in range(query_count)],
            "near_duplicate_rate": 0.0,
            "broad_query_rate": 0.0,
        }

    summaries = {
        "train": {
            "range": {"range_query_count": 24},
            "range_signal": {},
            "generation": {"workload_signature": signature(24)},
        },
        "eval": {
            "range": {"range_query_count": 48},
            "range_signal": {},
            "generation": {"workload_signature": signature(48)},
        },
    }

    gate = range_workload_distribution_comparison(summaries)["workload_signature_gate"]["pairs"][
        "train"
    ]

    assert gate["gate_pass"] is True
    assert gate["metrics"]["query_count_relative_delta"] == pytest.approx(0.50)
    assert gate["metrics"]["query_count_relative_delta_enforced"] is False
    assert gate["metrics"]["query_count_check_mode"] == (
        "diagnostic_min_only_for_coverage_calibrated"
    )
    assert "query_count_mismatch" not in gate["failed_checks"]


def test_workload_signature_gate_allows_small_calibrated_query_count_drift() -> None:
    def signature(query_count: int) -> dict[str, Any]:
        return {
            "profile_id": "range_query_mix",
            "query_count": query_count,
            "anchor_family_counts": {"density": query_count},
            "footprint_family_counts": {"medium_operational": query_count},
            "point_hit_counts_per_query": [3 for _ in range(query_count)],
            "ship_hit_counts_per_query": [1 for _ in range(query_count)],
            "near_duplicate_rate": 0.0,
            "broad_query_rate": 0.0,
        }

    summaries = {
        "train": {
            "range": {"range_query_count": 8},
            "range_signal": {},
            "generation": {"workload_signature": signature(8)},
        },
        "eval": {
            "range": {"range_query_count": 9},
            "range_signal": {},
            "generation": {"workload_signature": signature(9)},
        },
    }

    gate = range_workload_distribution_comparison(summaries)["workload_signature_gate"]["pairs"][
        "train"
    ]

    assert gate["gate_pass"] is True
    assert gate["metrics"]["query_count_delta"] == 1
    assert gate["metrics"]["query_count_relative_delta"] == pytest.approx(1 / 9)


def test_workload_signature_gate_reports_normalized_hit_distribution_diagnostics() -> None:
    def signature(
        total_points: int,
        total_trajectories: int,
        *,
        point_counts: list[int],
        ship_counts: list[int],
    ) -> dict[str, Any]:
        return {
            "profile_id": "range_query_mix",
            "target_coverage": 0.30,
            "coverage_actual": 0.30,
            "query_count_mode": "calibrated_to_coverage",
            "coverage_calibration_mode": "profile_sampled_query_count",
            "query_count": 8,
            "total_points": total_points,
            "total_trajectories": total_trajectories,
            "anchor_family_counts": {"density": 8},
            "footprint_family_counts": {"medium_operational": 8},
            "point_hit_counts_per_query": point_counts,
            "point_hit_fractions_per_query": [0.10, 0.20, 0.10, 0.20, 0.10, 0.20, 0.10, 0.20],
            "ship_hit_counts_per_query": ship_counts,
            "ship_hit_fractions_per_query": [0.10, 0.20, 0.10, 0.20, 0.10, 0.20, 0.10, 0.20],
            "near_duplicate_rate": 0.0,
            "broad_query_rate": 0.0,
        }

    summaries = {
        "train": {
            "range": {"range_query_count": 8},
            "range_signal": {},
            "generation": {
                "workload_signature": signature(
                    100,
                    10,
                    point_counts=[10, 20, 10, 20, 10, 20, 10, 20],
                    ship_counts=[1, 2, 1, 2, 1, 2, 1, 2],
                )
            },
        },
        "eval": {
            "range": {"range_query_count": 8},
            "range_signal": {},
            "generation": {
                "workload_signature": signature(
                    50,
                    5,
                    point_counts=[5, 10, 5, 10, 5, 10, 5, 10],
                    ship_counts=[1, 1, 1, 1, 1, 1, 1, 1],
                )
            },
        },
    }

    gate = range_workload_distribution_comparison(summaries)["workload_signature_gate"]["pairs"][
        "train"
    ]
    metrics = gate["metrics"]

    assert metrics["point_hit_fraction_distribution_ks"] == 0.0
    assert metrics["ship_hit_fraction_distribution_ks"] == 0.0
    assert metrics["point_hit_distribution_gate_metric"] == "point_hit_fraction_distribution_ks"
    assert metrics["ship_hit_distribution_enforced"] is False
    assert gate["gate_pass"] is True
    assert metrics["train_total_points"] == 100
    assert metrics["eval_total_points"] == 50
    assert metrics["train_total_trajectories"] == 10
    assert metrics["eval_total_trajectories"] == 5


def test_workload_blind_range_features_and_selector_are_query_free() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=40, seed=83)
    points = torch.cat(trajectories, dim=0)
    boundaries = _boundaries(trajectories)
    features = build_workload_blind_range_point_features(points)
    model = WorkloadBlindRangeModel(
        point_dim=WORKLOAD_BLIND_RANGE_POINT_DIM,
        query_dim=12,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
    )

    pred, head_logits = model.forward_with_heads(features.unsqueeze(0), padding_mask=None)
    no_behavior_pred = model.final_logit_from_head_logits(
        head_logits,
        disabled_head_names=("conditional_behavior_utility",),
    )
    segment_scores = head_logits.squeeze(0)[:, 4]
    retained = simplify_with_learned_segment_budget(
        pred.squeeze(0),
        boundaries,
        compression_ratio=0.10,
        segment_scores=segment_scores,
    )
    retained_with_trace, trace = simplify_with_learned_segment_budget_with_trace(
        pred.squeeze(0),
        boundaries,
        compression_ratio=0.10,
        segment_scores=segment_scores,
    )
    diagnostics = learned_segment_budget_diagnostics(boundaries, (0.05, 0.10))

    assert pred.shape == (1, points.shape[0])
    assert head_logits.shape == (1, points.shape[0], len(QUERY_LOCAL_UTILITY_HEAD_NAMES))
    assert no_behavior_pred.shape == pred.shape
    assert torch.isfinite(pred).all()
    assert retained.dtype == torch.bool
    assert torch.equal(retained, retained_with_trace)
    assert int(retained.sum().item()) > 0
    assert trace["point_attribution_available"] is True
    assert trace["skeleton_retained_count"] + trace["learned_controlled_retained_slots"] + trace[
        "fallback_retained_count"
    ] == int(retained.sum().item())
    assert trace["trajectories_with_at_least_one_learned_decision"] >= 0
    assert 0.0 <= trace["segment_budget_entropy_normalized"] <= 1.0
    assert trace["segment_score_source"] == "segment_budget_head_top20_mean"
    assert diagnostics["selector_type"] == "learned_segment_budget"
    assert diagnostics["budget_rows"][0]["no_fixed_85_percent_temporal_scaffold"] is True


def test_workload_blind_range_has_dedicated_prior_feature_encoder() -> None:
    torch.manual_seed(17)
    model = WorkloadBlindRangeModel(
        point_dim=WORKLOAD_BLIND_RANGE_POINT_DIM,
        query_dim=12,
        embed_dim=32,
        num_heads=2,
        num_layers=0,
        dropout=0.0,
    )
    base = torch.zeros((1, 4, WORKLOAD_BLIND_RANGE_POINT_DIM), dtype=torch.float32)
    with_prior = base.clone()
    with_prior[..., -6:] = torch.tensor([1.0, 0.5, 0.25, 0.0, 0.75, 1.0], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        base_score, base_heads = model.forward_with_heads(base)
        prior_score, prior_heads = model.forward_with_heads(with_prior)

    assert model.prior_feature_dim == 6
    assert isinstance(model.prior_feature_encoder[0], torch.nn.Linear)
    prior_layer = cast(torch.nn.Linear, model.prior_feature_encoder[0])
    assert tuple(prior_layer.weight.shape) == (32, 6)
    prior_output = cast(torch.nn.Linear, model.prior_feature_encoder[-1])
    assert float(prior_output.weight.detach().std(unbiased=False).item()) > 0.05
    assert abs(float(model.prior_feature_scale.detach().item()) - 0.25) < 1e-6
    assert not torch.allclose(base_score, prior_score)
    assert not torch.allclose(base_heads, prior_heads)
    assert float((base_heads - prior_heads).abs().mean().item()) > 1e-4


def test_workload_blind_range_untrained_reset_restores_standalone_parameters() -> None:
    model = WorkloadBlindRangeModel(
        point_dim=WORKLOAD_BLIND_RANGE_POINT_DIM,
        query_dim=12,
        embed_dim=32,
        num_heads=2,
        num_layers=0,
        dropout=0.0,
    )
    model.prior_feature_scale.data.fill_(3.0)

    reset_model = cast(WorkloadBlindRangeModel, reset_module_parameters(model, seed=101))

    assert torch.allclose(reset_model.prior_feature_scale.detach(), torch.tensor(0.25))
    assert torch.allclose(model.prior_feature_scale.detach(), torch.tensor(3.0))


def test_factorized_head_ablation_uses_neutral_multiplicative_heads() -> None:
    model = WorkloadBlindRangeModel(
        point_dim=WORKLOAD_BLIND_RANGE_POINT_DIM,
        query_dim=12,
        embed_dim=32,
        num_heads=2,
        num_layers=0,
        dropout=0.0,
    )
    head_logits = torch.zeros((1, 1, len(QUERY_LOCAL_UTILITY_HEAD_NAMES)), dtype=torch.float32)
    head_logits[..., 1] = -10.0

    disabled = model.final_logit_from_head_logits(
        head_logits,
        disabled_head_names=("conditional_behavior_utility",),
    )
    for parameter in model.calibration_head.parameters():
        parameter.data.fill_(100.0)
    disabled_with_large_calibration = model.final_logit_from_head_logits(
        head_logits,
        disabled_head_names=("conditional_behavior_utility",),
    )
    path_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("path_length_support_target")
    segment_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("segment_budget_target")
    low_path = head_logits.clone()
    high_path = head_logits.clone()
    low_path[..., path_idx] = -10.0
    high_path[..., path_idx] = 10.0
    low_segment = head_logits.clone()
    high_segment = head_logits.clone()
    low_segment[..., segment_idx] = -10.0
    high_segment[..., segment_idx] = 10.0
    expected_score = query_local_utility_point_score(
        q_hit=torch.tensor(0.5),
        behavior=torch.tensor(0.0),
        boundary=torch.tensor(0.5),
        replacement=torch.tensor(0.5),
    )

    assert torch.allclose(disabled.squeeze(), torch.logit(expected_score), atol=1e-6)
    assert torch.allclose(disabled, disabled_with_large_calibration)
    assert torch.allclose(
        model.final_logit_from_head_logits(low_path), model.final_logit_from_head_logits(high_path)
    )
    assert torch.allclose(
        model.final_logit_from_head_logits(low_segment),
        model.final_logit_from_head_logits(high_segment),
    )
    assert all(not parameter.requires_grad for parameter in model.calibration_head.parameters())


def test_workload_blind_range_final_score_composition_matches_query_local_utility_target_formula() -> (
    None
):
    model = WorkloadBlindRangeModel(
        point_dim=WORKLOAD_BLIND_RANGE_POINT_DIM,
        query_dim=12,
        embed_dim=32,
        num_heads=2,
        num_layers=0,
        dropout=0.0,
    )
    head_probabilities = torch.tensor(
        [[[0.8, 0.4, 0.2, 0.1, 0.9, 0.1], [0.8, 0.4, 0.2, 0.9, 0.1, 0.9]]],
        dtype=torch.float32,
    )
    head_logits = torch.logit(head_probabilities.clamp(1e-4, 1.0 - 1e-4))

    final_probabilities = torch.sigmoid(model.final_logit_from_head_logits(head_logits))

    expected = query_local_utility_point_score(
        q_hit=head_probabilities[..., 0],
        behavior=head_probabilities[..., 1],
        boundary=head_probabilities[..., 2],
        replacement=head_probabilities[..., 3],
    )
    assert torch.allclose(final_probabilities, expected, atol=1e-5)
    assert final_probabilities[0, 0] < 0.5
    assert final_probabilities[0, 1] > final_probabilities[0, 0]
