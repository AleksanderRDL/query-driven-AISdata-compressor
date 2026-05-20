"""Query-driven workload profile and generation tests."""

from __future__ import annotations

import torch

from data_preparation.ais_loader import generate_synthetic_ais_data
from workloads.generation.coverage import _record_rejection_for_query
from workloads.generation.generator import generate_typed_query_workload
from workloads.generation.profile_query_plan import (
    POINT_HIT_TARGET_BAND_FRACTION,
    _profile_query_plan,
    _profile_query_settings,
    _weighted_choice_with_deterministic_key,
)
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

def test_profile_query_plan_preserves_weighted_family_quotas() -> None:
    profile = range_workload_profile("range_query_mix")

    plan = _profile_query_plan(profile, requested_queries=20, workload_seed=123)

    assert plan["enabled"] is True
    assert len(plan["anchor_family_sequence"]) == 20
    assert len(plan["footprint_family_sequence"]) == 20
    assert plan["anchor_family_planned_counts"] == {
        "density": 16,
        "sparse_background_control": 4,
    }
    assert plan["footprint_family_planned_counts"] == {
        "medium_operational": 14,
        "large_context": 6,
    }


def test_profile_query_plan_prefixes_preserve_family_mix_when_workloads_expand() -> None:
    profile = range_workload_profile("range_query_mix")

    plan = _profile_query_plan(profile, requested_queries=256, workload_seed=123)
    anchor_prefix = plan["anchor_family_sequence"][:48]
    footprint_prefix = plan["footprint_family_sequence"][:48]

    assert {family: anchor_prefix.count(family) for family in set(anchor_prefix)} == {
        "density": 38,
        "sparse_background_control": 10,
    }
    assert {family: footprint_prefix.count(family) for family in set(footprint_prefix)} == {
        "medium_operational": 33,
        "large_context": 15,
    }


def test_range_query_mix_footprints_match_implementation_guide_defaults() -> None:
    profile = range_workload_profile("range_query_mix")

    assert profile.footprint_families == {
        "medium_operational": {
            "spatial_radius_km": 2.2,
            "time_half_window_hours": 5.0,
            "elongation_allowed": False,
            "min_point_hit_fraction": 0.006,
            "max_point_hit_fraction": 0.030,
        },
        "large_context": {
            "spatial_radius_km": 4.0,
            "time_half_window_hours": 8.0,
            "elongation_allowed": False,
            "min_point_hit_fraction": 0.010,
            "max_point_hit_fraction": 0.045,
        },
    }


def test_range_query_mix_target_coverage_keeps_requested_query_count() -> None:
    trajectories = generate_synthetic_ais_data(
        n_ships=5, n_points_per_ship=48, seed=87, route_families=1
    )
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=8,
        workload_map={"range": 1.0},
        seed=14,
        target_coverage=0.05,
        max_queries=64,
        workload_profile_id="range_query_mix",
        coverage_calibration_mode="profile_sampled_query_count",
        range_max_point_hit_fraction=1.0,
        range_duplicate_iou_threshold=1.0,
        range_max_coverage_overshoot=1.0,
    )

    generation = (workload.generation_diagnostics or {})["query_generation"]
    assert generation["query_count_mode"] == "calibrated_to_coverage"
    assert generation["coverage_calibration_mode"] == "profile_sampled_query_count"
    assert generation["final_query_count"] >= 8


def test_range_query_mix_records_profile_signature() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=5, n_points_per_ship=48, seed=81)
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=6,
        workload_map={"range": 1.0},
        seed=9,
        workload_profile_id="range_query_mix",
        range_max_point_hit_fraction=1.0,
        range_duplicate_iou_threshold=1.0,
    )

    diagnostics = workload.generation_diagnostics or {}
    signature = diagnostics["workload_signature"]
    profile = diagnostics["workload_profile"]
    generation = diagnostics["query_generation"]

    assert profile["profile_id"] == "range_query_mix"
    assert generation["range_time_domain_mode"] == "anchor_day"
    assert signature["profile_id"] == "range_query_mix"
    assert signature["workload_profile_version"] == profile["version"]
    assert signature["target_coverage"] == profile["target_coverage"]
    assert signature["query_count_mode"] == profile["query_count_mode"]
    assert signature["coverage_calibration_mode"] == profile["coverage_calibration_mode"]
    assert sum(signature["anchor_family_counts"].values()) == len(workload.typed_queries)
    assert sum(signature["footprint_family_counts"].values()) == len(workload.typed_queries)
    assert len(signature["anchor_family_per_query"]) == len(workload.typed_queries)
    assert len(signature["footprint_family_per_query"]) == len(workload.typed_queries)
    assert len(signature["anchor_footprint_pair_per_query"]) == len(workload.typed_queries)
    assert set(signature["anchor_family_per_query"]) <= set(signature["anchor_family_counts"])
    assert set(signature["footprint_family_per_query"]) <= set(signature["footprint_family_counts"])
    assert set(signature["anchor_footprint_pair_per_query"]) <= set(
        signature["anchor_footprint_pair_counts"]
    )
    assert signature["query_count"] == len(workload.typed_queries)
    assert signature["total_points"] == sum(int(trajectory.shape[0]) for trajectory in trajectories)
    assert signature["total_trajectories"] == len(trajectories)
    assert len(signature["point_hit_counts_per_query"]) == len(workload.typed_queries)
    assert len(signature["point_hit_fractions_per_query"]) == len(workload.typed_queries)
    assert len(signature["ship_hit_counts_per_query"]) == len(workload.typed_queries)
    assert len(signature["ship_hit_fractions_per_query"]) == len(workload.typed_queries)


def test_range_acceptance_rejections_record_anchor_footprint_pair() -> None:
    state: dict[str, object] = {}
    query = {
        "_metadata": {
            "anchor_family": "density",
            "footprint_family": "large_context",
        }
    }

    _record_rejection_for_query(state, "coverage_overshoot", query)

    assert state["rejection_reasons"] == {"coverage_overshoot": 1}
    assert state["rejection_reasons_by_anchor_family"] == {
        "density": {"coverage_overshoot": 1}
    }
    assert state["rejection_reasons_by_footprint_family"] == {
        "large_context": {"coverage_overshoot": 1}
    }
    assert state["rejection_reasons_by_anchor_footprint_pair"] == {
        "density|large_context": {"coverage_overshoot": 1}
    }


def test_deterministic_profile_sampling_does_not_advance_generator() -> None:
    profile = range_workload_profile("range_query_mix")
    gen = torch.Generator().manual_seed(12345)
    before = gen.get_state()

    chosen_anchor = _weighted_choice_with_deterministic_key(
        profile.anchor_family_weights,
        gen,
        fallback="density",
        deterministic_value=0.33,
    )
    chosen_footprint = _weighted_choice_with_deterministic_key(
        profile.footprint_family_weights,
        gen,
        fallback="medium_operational",
        deterministic_value=0.77,
    )

    settings = _profile_query_settings(
        profile, torch.Generator().manual_seed(1), query_index=7, workload_seed=19
    )
    settings_deterministic = _profile_query_settings(
        profile, torch.Generator().manual_seed(1), query_index=7, workload_seed=19
    )
    assert settings == settings_deterministic

    after = gen.get_state()
    assert chosen_anchor in profile.anchor_family_weights
    assert chosen_footprint in profile.footprint_family_weights
    assert torch.equal(before, after)

    baseline = torch.Generator().manual_seed(12345)
    expected_seq = torch.randint(0, 999, (3,), generator=baseline).tolist()
    observed_seq = torch.randint(0, 999, (3,), generator=gen).tolist()
    assert observed_seq == expected_seq


def test_profile_point_hit_targets_are_prefix_stable_within_footprint_band() -> None:
    profile = range_workload_profile("range_query_mix")
    plan = _profile_query_plan(profile, requested_queries=20, workload_seed=123)

    settings = [
        _profile_query_settings(
            profile,
            torch.Generator().manual_seed(1),
            query_index=query_index,
            workload_seed=19,
            query_plan=plan,
        )
        for query_index in range(20)
    ]
    same_plan_settings = [
        _profile_query_settings(
            profile,
            torch.Generator().manual_seed(1),
            query_index=query_index,
            workload_seed=99,
            query_plan=plan,
        )
        for query_index in range(20)
    ]

    assert [
        setting["target_point_hit_fraction"] for setting in settings
    ] == [setting["target_point_hit_fraction"] for setting in same_plan_settings]
    for setting in settings:
        footprint = profile.footprint_families[str(setting["footprint_family"])]
        min_fraction = float(footprint["min_point_hit_fraction"])
        max_fraction = float(footprint["max_point_hit_fraction"])
        target = setting["target_point_hit_fraction"]
        assert isinstance(target, float)
        assert min_fraction <= target <= min_fraction + (
            max_fraction - min_fraction
        ) * POINT_HIT_TARGET_BAND_FRACTION


def test_synthetic_route_families_create_same_support_trajectories() -> None:
    trajectories = generate_synthetic_ais_data(
        n_ships=5, n_points_per_ship=32, seed=86, route_families=1
    )
    points = torch.cat(trajectories, dim=0)

    assert float(points[:, 1].max().item() - points[:, 1].min().item()) < 0.50
    assert float(points[:, 2].max().item() - points[:, 2].min().item()) < 0.50
