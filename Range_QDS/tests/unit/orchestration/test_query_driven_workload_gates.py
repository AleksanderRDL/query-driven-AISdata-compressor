"""Query-driven workload, predictability, and final gate tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from data_preparation.ais_loader import generate_synthetic_ais_data
from learning.predictability_audit import (
    _prior_channel_scores,
    _rankdata,
    query_prior_predictability_audit,
    query_prior_predictability_scores,
)
from learning.query_prior_fields import (
    build_train_query_prior_fields,
)
from learning.targets.query_local_utility import (
    build_query_local_utility_targets,
)
from orchestration.gates import (
    evaluate_global_sanity_gate,
    evaluate_workload_stability_gate,
)
from orchestration.range_diagnostics import range_workload_distribution_comparison
from scoring.geometry_thresholds import (
    FINAL_LENGTH_PRESERVATION_MIN,
    max_sed_ratio_for_compression,
)
from scoring.metrics import MethodScore
from workloads.generation.anchors import _anchor_weights_for_family
from workloads.generation.generator import generate_typed_query_workload
from workloads.generation.workload_profiles import range_workload_profile

HISTORICAL_SMALL_LOCAL_FAMILY = "small_local"


def _boundaries(trajectories: list[torch.Tensor]) -> list[tuple[int, int]]:
    cursor = 0
    out = []
    for trajectory in trajectories:
        end = cursor + int(trajectory.shape[0])
        out.append((cursor, end))
        cursor = end
    return out

def test_workload_signature_gate_reports_pass_for_matching_profiles() -> None:
    signature = {
        "profile_id": "range_query_mix",
        "query_count": 8,
        "anchor_family_counts": {"density": 6, "sparse_background_control": 2},
        "footprint_family_counts": {"medium_operational": 8},
        "point_hits_per_query": {"p10": 3.0, "p50": 5.0, "p90": 8.0},
        "ship_hits_per_query": {"p10": 1.0, "p50": 2.0, "p90": 3.0},
        "near_duplicate_rate": 0.0,
        "broad_query_rate": 0.0,
    }
    summaries = {
        "train": {"range": {}, "range_signal": {}, "generation": {"workload_signature": signature}},
        "eval": {
            "range": {},
            "range_signal": {},
            "generation": {"workload_signature": dict(signature)},
        },
    }

    comparison = range_workload_distribution_comparison(summaries)
    gate = comparison["workload_signature_gate"]

    assert gate["all_available"] is True
    assert gate["all_pass"] is True
    assert gate["pairs"]["train"]["gate_pass"] is True


def test_predictability_audit_is_diagnostic_only_and_reports_gate_fields() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=4, n_points_per_ship=32, seed=84)
    points = torch.cat(trajectories, dim=0)
    boundaries = _boundaries(trajectories)
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=5,
        workload_map={"range": 1.0},
        seed=13,
        workload_profile_id="range_query_mix",
        range_max_point_hit_fraction=1.0,
        range_duplicate_iou_threshold=1.0,
    )
    targets = build_query_local_utility_targets(
        points=points,
        boundaries=boundaries,
        typed_queries=workload.typed_queries,
    )
    prior = build_train_query_prior_fields(
        points=points,
        boundaries=boundaries,
        typed_queries=workload.typed_queries,
        labels=targets.labels,
        workload_profile_id="range_query_mix",
        train_workload_seed=13,
    )

    audit = query_prior_predictability_audit(
        points=points,
        boundaries=boundaries,
        eval_typed_queries=workload.typed_queries,
        query_prior_field=prior,
    )

    assert audit["available"] is True
    assert audit["used_for_training"] is False
    assert audit["used_for_checkpoint_selection"] is False
    assert audit["used_for_retained_mask_decision"] is False
    assert "spearman" in audit["metrics"]
    assert audit["metrics"]["rank_correlation_method"] == "average_tie_ranks"
    assert "score_decile_calibration" in audit["metrics"]
    assert "lift_at_5_percent" in audit["metrics"]
    assert "lift_at_5_percent" in audit["gate_checks"]
    channel_by_head = audit["prior_channel_by_head_predictability"]
    assert "query_hit_probability" in channel_by_head
    assert "spatiotemporal_query_hit_probability" in channel_by_head["query_hit_probability"]
    assert (
        "lift_at_5_percent"
        in channel_by_head["query_hit_probability"]["spatiotemporal_query_hit_probability"]
    )
    assert (
        channel_by_head["query_hit_probability"]["spatiotemporal_query_hit_probability"][
            "rank_correlation_method"
        ]
        == "average_tie_ranks"
    )
    assert (
        "score_decile_calibration"
        in channel_by_head["query_hit_probability"]["spatiotemporal_query_hit_probability"]
    )
    best_by_head = audit["best_prior_channel_by_head"]
    assert "query_hit_probability" in best_by_head
    assert "channel" in best_by_head["query_hit_probability"]["best_lift_at_5_percent"]
    family_prior = audit["family_conditioned_prior_predictability"]
    assert family_prior["available"] is True
    assert family_prior["diagnostic_only"] is True
    assert family_prior["used_for_gate"] is False
    assert HISTORICAL_SMALL_LOCAL_FAMILY not in family_prior["group_by"]["footprint_family"]
    assert "medium_operational" in family_prior["group_by"]["footprint_family"]
    medium_operational = family_prior["group_by"]["footprint_family"]["medium_operational"]
    assert medium_operational["focus_family"] is True
    assert medium_operational["valid_hit_point_count"] > 0
    assert "segment_budget_target" in medium_operational["heads"]
    segment_budget = medium_operational["heads"]["segment_budget_target"]
    assert segment_budget["available"] is True
    assert "mapped_prior_channel" in segment_budget
    assert "best_spearman" in segment_budget
    assert "best_lift_at_5_percent" in segment_budget


def test_predictability_rankdata_uses_average_tie_ranks() -> None:
    values = torch.tensor([2.0, 2.0, 5.0, 5.0, 9.0])

    ranks = _rankdata(values)

    assert torch.allclose(ranks, torch.tensor([0.5, 0.5, 2.5, 2.5, 4.0]))


def test_segment_budget_prior_does_not_blend_raw_route_density() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    prior = {
        "grid_bins": 1,
        "time_bins": 1,
        "extent": {
            "t_min": 0.0,
            "t_max": 1.0,
            "lat_min": 0.0,
            "lat_max": 1.0,
            "lon_min": 0.0,
            "lon_max": 1.0,
        },
        "out_of_extent_sampling": "nearest",
        "spatial_query_hit_probability": torch.tensor([0.20]),
        "spatiotemporal_query_hit_probability": torch.tensor([[0.10]]),
        "endpoint_likelihood": torch.tensor([0.30]),
        "crossing_likelihood": torch.tensor([0.40]),
        "behavior_utility_prior": torch.tensor([0.80]),
        "route_density_prior": torch.tensor([1.00]),
    }

    scores = _prior_channel_scores(points, prior)

    assert torch.allclose(
        scores["segment_budget_prior"], scores["replacement_representative_prior"]
    )
    assert not torch.allclose(scores["segment_budget_prior"], scores["route_density_prior"])


def test_query_prior_predictability_score_gates_behavior_utility_by_query_mass() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    prior = {
        "grid_bins": 1,
        "time_bins": 1,
        "extent": {
            "t_min": 0.0,
            "t_max": 1.0,
            "lat_min": 0.0,
            "lat_max": 1.0,
            "lon_min": 0.0,
            "lon_max": 1.0,
        },
        "out_of_extent_sampling": "nearest",
        "spatial_query_hit_probability": torch.tensor([0.01]),
        "spatiotemporal_query_hit_probability": torch.tensor([[0.01]]),
        "endpoint_likelihood": torch.tensor([0.0]),
        "crossing_likelihood": torch.tensor([0.0]),
        "behavior_utility_prior": torch.tensor([0.90]),
        "route_density_prior": torch.tensor([0.0]),
    }

    score = query_prior_predictability_scores(points, prior)

    assert torch.allclose(score.cpu(), torch.full((2,), 0.014))


def test_range_query_mix_profile_uses_only_simple_footprints() -> None:
    profile = range_workload_profile("range_query_mix")
    assert profile.final_success_allowed is True
    assert profile.target_coverage == pytest.approx(0.30)
    assert profile.max_coverage_overshoot == pytest.approx(0.020)
    assert range_workload_profile("range_query_mix_local").target_coverage == pytest.approx(0.10)
    assert set(profile.footprint_family_weights) == {
        "medium_operational",
        "large_context",
    }
    assert all(
        footprint["elongation_allowed"] is False
        for footprint in profile.footprint_families.values()
    )


def test_range_query_mix_rejects_removed_anchor_families() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.1, 0.1, 4.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.2, 0.2, 5.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 1.0, 1.0, 0.2, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    density, _density_prob = _anchor_weights_for_family(points, "density")

    assert density is not None
    for removed_family in (
        "boundary_entry_exit",
        "crossing_turn_change",
        "port_or_approach_zone",
    ):
        with pytest.raises(ValueError, match="Unknown range workload anchor family"):
            _anchor_weights_for_family(points, removed_family)


def test_final_profile_does_not_chase_uncovered_points_unless_declared() -> None:
    trajectories = generate_synthetic_ais_data(
        n_ships=4, n_points_per_ship=32, seed=91, route_families=1
    )
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=22,
        max_queries=8,
        workload_profile_id="range_query_mix",
        range_max_point_hit_fraction=1.0,
        range_duplicate_iou_threshold=1.0,
    )
    generation = (workload.generation_diagnostics or {})["query_generation"]
    uncovered_anchor_chasing_workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=22,
        target_coverage=0.30,
        max_queries=8,
        workload_profile_id="range_query_mix",
        coverage_calibration_mode="uncovered_anchor_chasing",
        range_max_point_hit_fraction=1.0,
        range_duplicate_iou_threshold=1.0,
    )
    chasing_generation = (uncovered_anchor_chasing_workload.generation_diagnostics or {})[
        "query_generation"
    ]

    assert generation["coverage_calibration_mode"] == "profile_sampled_query_count"
    assert generation["target_coverage"] == pytest.approx(0.30)
    assert chasing_generation["coverage_calibration_mode"] == "uncovered_anchor_chasing"


def test_workload_stability_gate_rejects_legacy_fixed_count_workloads() -> None:
    config = SimpleNamespace(
        query=SimpleNamespace(
            target_coverage=None,
            range_max_coverage_overshoot=None,
            workload_profile_id="legacy_generator",
        )
    )
    workload = SimpleNamespace(
        typed_queries=[{} for _ in range(4)],
        coverage_fraction=0.25,
        generation_diagnostics={
            "query_generation": {
                "mode": "fixed_count",
                "workload_profile_id": "legacy_generator",
                "coverage_calibration_mode": "legacy_fixed_or_target_coverage",
                "coverage_guard_enabled": False,
                "stop_reason": "fixed_count_completed",
            }
        },
    )

    gate = evaluate_workload_stability_gate(
        config=cast(Any, config),
        train_label_workloads=[workload],
        eval_workload=workload,
        selection_workload=None,
    )

    assert gate["gate_pass"] is False
    assert "workload_profile_not_in_final_grid" in gate["failed_checks"]
    assert "too_few_train_workload_replicates" in gate["failed_checks"]
    assert "train_r0:not_target_coverage_generation" in gate["failed_checks"]
    assert "eval:too_few_queries" in gate["failed_checks"]


def test_workload_stability_gate_accepts_coverage_calibrated_replicates() -> None:
    config = SimpleNamespace(
        query=SimpleNamespace(
            target_coverage=0.30,
            range_max_coverage_overshoot=0.020,
            workload_profile_id="range_query_mix",
        )
    )

    def workload() -> SimpleNamespace:
        return SimpleNamespace(
            typed_queries=[{} for _ in range(8)],
            coverage_fraction=0.305,
            generation_diagnostics={
                "query_generation": {
                    "mode": "target_coverage",
                    "workload_profile_id": "range_query_mix",
                    "coverage_calibration_mode": "profile_sampled_query_count",
                    "query_count_mode": "calibrated_to_coverage",
                    "target_coverage": 0.30,
                    "coverage_guard_enabled": True,
                    "stop_reason": "target_coverage_reached",
                }
            },
        )

    gate = evaluate_workload_stability_gate(
        config=cast(Any, config),
        train_label_workloads=[workload(), workload(), workload(), workload()],
        eval_workload=workload(),
        selection_workload=None,
    )

    assert gate["gate_pass"] is True
    assert gate["failed_checks"] == []
    assert gate["train_workload_replicate_count"] == 4


def test_workload_stability_gate_rejects_exhausted_generation_after_coverage_satisfied() -> None:
    config = SimpleNamespace(
        query=SimpleNamespace(
            target_coverage=0.10,
            range_max_coverage_overshoot=0.0075,
            workload_profile_id="range_query_mix_local",
            workload_stability_gate_mode="final",
        )
    )
    workload = SimpleNamespace(
        typed_queries=[{} for _ in range(12)],
        coverage_fraction=0.105,
        generation_diagnostics={
            "query_generation": {
                "mode": "target_coverage",
                "workload_profile_id": "range_query_mix_local",
                "coverage_calibration_mode": "profile_sampled_query_count",
                "query_count_mode": "calibrated_to_coverage",
                "target_coverage": 0.10,
                "coverage_guard_enabled": True,
                "stop_reason": "range_acceptance_exhausted",
            },
            "range_acceptance": {
                "enabled": True,
                "attempts": 6000,
                "accepted": 12,
                "rejected": 5988,
                "exhausted": True,
            },
        },
    )

    gate = evaluate_workload_stability_gate(
        config=cast(Any, config),
        train_label_workloads=[workload, workload, workload, workload],
        eval_workload=workload,
        selection_workload=None,
    )

    assert gate["gate_pass"] is False
    assert "train_r0:range_acceptance_or_coverage_guard_exhausted" in gate["failed_checks"]
    assert "eval:range_acceptance_or_coverage_guard_exhausted" in gate["failed_checks"]
    assert gate["workloads"][0]["coverage_target_satisfied"] is True


def test_workload_stability_gate_rejects_calibrated_low_query_count_in_final_mode() -> None:
    config = SimpleNamespace(
        query=SimpleNamespace(
            target_coverage=0.05,
            range_max_coverage_overshoot=0.005,
            workload_profile_id="range_query_mix_focused",
            workload_stability_gate_mode="final",
        )
    )
    workload = SimpleNamespace(
        typed_queries=[{} for _ in range(7)],
        coverage_fraction=0.054,
        generation_diagnostics={
            "query_generation": {
                "mode": "target_coverage",
                "workload_profile_id": "range_query_mix_focused",
                "coverage_calibration_mode": "profile_sampled_query_count",
                "query_count_mode": "calibrated_to_coverage",
                "target_coverage": 0.05,
                "coverage_guard_enabled": True,
                "stop_reason": "target_coverage_reached",
            }
        },
    )

    gate = evaluate_workload_stability_gate(
        config=cast(Any, config),
        train_label_workloads=[workload, workload, workload, workload],
        eval_workload=workload,
        selection_workload=None,
    )

    assert gate["gate_pass"] is False
    assert "train_r0:too_few_queries" in gate["failed_checks"]
    assert "eval:too_few_queries" in gate["failed_checks"]


def test_workload_stability_gate_smoke_mode_allows_calibrated_low_query_count() -> None:
    config = SimpleNamespace(
        query=SimpleNamespace(
            target_coverage=0.05,
            range_max_coverage_overshoot=0.005,
            workload_profile_id="range_query_mix_focused",
            workload_stability_gate_mode="smoke",
        )
    )
    workload = SimpleNamespace(
        typed_queries=[{} for _ in range(7)],
        coverage_fraction=0.054,
        generation_diagnostics={
            "query_generation": {
                "mode": "target_coverage",
                "workload_profile_id": "range_query_mix_focused",
                "coverage_calibration_mode": "profile_sampled_query_count",
                "query_count_mode": "calibrated_to_coverage",
                "target_coverage": 0.05,
                "coverage_guard_enabled": True,
                "stop_reason": "target_coverage_reached",
            }
        },
    )

    gate = evaluate_workload_stability_gate(
        config=cast(Any, config),
        train_label_workloads=[workload, workload, workload, workload],
        eval_workload=workload,
        selection_workload=None,
    )

    assert gate["gate_pass"] is True
    assert gate["failed_checks"] == []
    assert gate["gate_mode"] == "smoke"


def test_global_sanity_gate_enforces_endpoint_length_and_sed_ratio() -> None:
    primary = MethodScore(
        aggregate_f1=0.0,
        per_type_f1={},
        avg_length_preserved=0.90,
        geometric_distortion={"avg_sed_km": 0.90},
        range_audit={"endpoint_sanity": 1.0},
    )
    uniform = MethodScore(
        aggregate_f1=0.0,
        per_type_f1={},
        avg_length_preserved=0.95,
        geometric_distortion={"avg_sed_km": 0.60},
        range_audit={"endpoint_sanity": 1.0},
    )

    gate = evaluate_global_sanity_gate(primary=primary, uniform=uniform, compression_ratio=0.05)

    assert gate["gate_pass"] is True
    assert gate["length_preservation_min"] == pytest.approx(FINAL_LENGTH_PRESERVATION_MIN)
    assert gate["avg_sed_ratio_vs_uniform"] == 1.5
    assert gate["avg_sed_ratio_vs_uniform_max"] == pytest.approx(
        max_sed_ratio_for_compression(0.05)
    )
    assert gate["catastrophic_geometry_outlier_status"] == "not_available_report_only"

    primary.avg_length_preserved = 0.70
    primary.range_audit["endpoint_sanity"] = 0.5
    primary.geometric_distortion["avg_sed_km"] = 1.20
    gate = evaluate_global_sanity_gate(primary=primary, uniform=uniform, compression_ratio=0.05)

    assert gate["gate_pass"] is False
    assert "length_preservation_outside_range" in gate["failed_checks"]
    assert "endpoints_not_retained_for_all_eligible_trajectories" in gate["failed_checks"]
    assert "avg_sed_ratio_vs_uniform_too_high" in gate["failed_checks"]
