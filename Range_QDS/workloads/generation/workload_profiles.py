"""Versioned range workload profiles.

The query-driven implementation treats the range-query distribution as a
versioned product object, not as loose benchmark knobs.  ``range_query_mix`` is
the default in-distribution profile for final candidates.  Runs without an
explicit profile remain legacy diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RangeWorkloadProfile:
    profile_id: str
    version: int
    anchor_family_weights: dict[str, float]
    footprint_family_weights: dict[str, float]
    footprint_families: dict[str, dict[str, Any]]
    target_coverage: float | None
    max_coverage_overshoot: float | None
    time_domain_mode: str = "anchor_day"
    query_count_mode: str = "calibrated_to_coverage"
    coverage_calibration_mode: str = "profile_sampled_query_count"
    min_points_per_query: int = 3
    min_trajectories_per_query: int = 1
    max_near_duplicate_hitset_jaccard: float = 0.65
    max_empty_query_fraction: float = 0.0
    max_near_duplicate_fraction: float = 0.05
    max_broad_query_fraction: float = 0.05
    max_attempt_multiplier: int = 20
    final_success_allowed: bool = False


RANGE_QUERY_MIX_FOCUSED_PROFILE_ID = "range_query_mix_focused"
RANGE_QUERY_MIX_LOCAL_PROFILE_ID = "range_query_mix_local"
RANGE_QUERY_MIX_OPERATIONAL_PROFILE_ID = "range_query_mix_operational"
RANGE_QUERY_MIX_PROFILE_ID = "range_query_mix"
RANGE_QUERY_MIX_FINAL_PROFILE_IDS = (
    RANGE_QUERY_MIX_FOCUSED_PROFILE_ID,
    RANGE_QUERY_MIX_LOCAL_PROFILE_ID,
    RANGE_QUERY_MIX_OPERATIONAL_PROFILE_ID,
    RANGE_QUERY_MIX_PROFILE_ID,
)


LEGACY_GENERATOR_PROFILE = RangeWorkloadProfile(
    profile_id="legacy_generator",
    version=0,
    anchor_family_weights={},
    footprint_family_weights={},
    footprint_families={},
    target_coverage=None,
    max_coverage_overshoot=None,
    time_domain_mode="dataset",
    query_count_mode="legacy_fixed_or_target_coverage",
    coverage_calibration_mode="uncovered_anchor_chasing",
    min_points_per_query=0,
    min_trajectories_per_query=0,
    max_near_duplicate_hitset_jaccard=0.85,
    max_empty_query_fraction=1.0,
    max_near_duplicate_fraction=1.0,
    max_broad_query_fraction=1.0,
    max_attempt_multiplier=50,
    final_success_allowed=False,
)


def _range_query_mix_profile(
    *,
    profile_id: str,
    target_coverage: float,
    max_coverage_overshoot: float,
) -> RangeWorkloadProfile:
    """Build one named query-local range mix profile variant."""
    return RangeWorkloadProfile(
        profile_id=profile_id,
        version=1,
        anchor_family_weights={
            "density": 0.80,
            "sparse_background_control": 0.20,
        },
        footprint_family_weights={
            "medium_operational": 0.6923076923076923,
            "large_context": 0.3076923076923077,
        },
        footprint_families={
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
        },
        target_coverage=target_coverage,
        max_coverage_overshoot=max_coverage_overshoot,
        time_domain_mode="anchor_day",
        query_count_mode="calibrated_to_coverage",
        coverage_calibration_mode="profile_sampled_query_count",
        min_points_per_query=3,
        min_trajectories_per_query=1,
        max_near_duplicate_hitset_jaccard=0.65,
        max_empty_query_fraction=0.0,
        max_near_duplicate_fraction=0.05,
        max_broad_query_fraction=0.05,
        max_attempt_multiplier=20,
        final_success_allowed=True,
    )


RANGE_QUERY_MIX_FOCUSED_PROFILE = _range_query_mix_profile(
    profile_id=RANGE_QUERY_MIX_FOCUSED_PROFILE_ID,
    target_coverage=0.05,
    max_coverage_overshoot=0.005,
)
RANGE_QUERY_MIX_LOCAL_PROFILE = _range_query_mix_profile(
    profile_id=RANGE_QUERY_MIX_LOCAL_PROFILE_ID,
    target_coverage=0.10,
    max_coverage_overshoot=0.0075,
)
RANGE_QUERY_MIX_OPERATIONAL_PROFILE = _range_query_mix_profile(
    profile_id=RANGE_QUERY_MIX_OPERATIONAL_PROFILE_ID,
    target_coverage=0.15,
    max_coverage_overshoot=0.010,
)
RANGE_QUERY_MIX_PROFILE = _range_query_mix_profile(
    profile_id=RANGE_QUERY_MIX_PROFILE_ID,
    target_coverage=0.30,
    max_coverage_overshoot=0.020,
)

PROFILE_BY_ID: dict[str, RangeWorkloadProfile] = {
    LEGACY_GENERATOR_PROFILE.profile_id: LEGACY_GENERATOR_PROFILE,
    RANGE_QUERY_MIX_FOCUSED_PROFILE.profile_id: RANGE_QUERY_MIX_FOCUSED_PROFILE,
    RANGE_QUERY_MIX_LOCAL_PROFILE.profile_id: RANGE_QUERY_MIX_LOCAL_PROFILE,
    RANGE_QUERY_MIX_OPERATIONAL_PROFILE.profile_id: RANGE_QUERY_MIX_OPERATIONAL_PROFILE,
    RANGE_QUERY_MIX_PROFILE.profile_id: RANGE_QUERY_MIX_PROFILE,
}
WORKLOAD_PROFILE_CHOICES = tuple(PROFILE_BY_ID)


def normalize_workload_profile_id(profile_id: str | None) -> str:
    """Return a canonical workload profile id."""
    normalized = str(profile_id or LEGACY_GENERATOR_PROFILE.profile_id).strip().lower()
    if not normalized:
        return LEGACY_GENERATOR_PROFILE.profile_id
    return normalized


def range_workload_profile(profile_id: str | None) -> RangeWorkloadProfile:
    """Return a known range workload profile."""
    normalized = normalize_workload_profile_id(profile_id)
    try:
        return PROFILE_BY_ID[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILE_BY_ID))
        raise ValueError(
            f"Unknown workload_profile_id={profile_id!r}; choices: {choices}."
        ) from exc


def max_point_hit_fraction_for_coverage(target_coverage: float | None) -> float:
    """Return the profile broad-query point-hit ceiling for a coverage target."""
    if target_coverage is None:
        return 0.05
    target = float(target_coverage)
    if target > 1.0 and target <= 100.0:
        target = target / 100.0
    if target <= 0.05:
        return 0.020
    if target <= 0.10:
        return 0.025
    if target <= 0.15:
        return 0.030
    return 0.050


def workload_profile_metadata(profile: RangeWorkloadProfile) -> dict[str, Any]:
    """Return JSON-safe profile metadata for artifacts."""
    return {
        "profile_id": profile.profile_id,
        "version": int(profile.version),
        "anchor_family_weights": dict(profile.anchor_family_weights),
        "footprint_family_weights": dict(profile.footprint_family_weights),
        "footprint_families": {
            name: dict(values) for name, values in profile.footprint_families.items()
        },
        "target_coverage": profile.target_coverage,
        "max_coverage_overshoot": profile.max_coverage_overshoot,
        "time_domain_mode": profile.time_domain_mode,
        "query_count_mode": profile.query_count_mode,
        "coverage_calibration_mode": profile.coverage_calibration_mode,
        "query_acceptance": {
            "min_points_per_query": int(profile.min_points_per_query),
            "min_trajectories_per_query": int(profile.min_trajectories_per_query),
            "max_near_duplicate_hitset_jaccard": float(profile.max_near_duplicate_hitset_jaccard),
            "max_empty_query_fraction": float(profile.max_empty_query_fraction),
            "max_near_duplicate_fraction": float(profile.max_near_duplicate_fraction),
            "max_broad_query_fraction": float(profile.max_broad_query_fraction),
            "max_attempt_multiplier": int(profile.max_attempt_multiplier),
        },
        "final_success_allowed": bool(profile.final_success_allowed),
    }
