"""Profile planning helpers for generated range query workloads."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch

from workloads.generation.anchors import _weighted_sample_one
from workloads.generation.workload_profiles import LEGACY_GENERATOR_PROFILE, RangeWorkloadProfile

POINT_HIT_TARGET_BAND_FRACTION = 0.25
LOW_DISCREPANCY_STEP = 0.6180339887498949


def _deterministic_unit_from_key_parts(*parts: object) -> float:
    """Return a deterministic unit-uniform-like value from arbitrary key material."""
    key_material = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(key_material.encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return raw / float(1 << 64)


def _weighted_choice_with_deterministic_key(
    mapping: dict[str, float],
    generator: torch.Generator,
    fallback: str,
    deterministic_value: float | None = None,
) -> str:
    """Sample one key from a non-negative weight mapping using deterministic ordering."""
    if not mapping:
        return fallback
    keys = [str(key) for key in mapping]
    weights = torch.tensor([max(0.0, float(mapping[key])) for key in keys], dtype=torch.float32)
    total = float(weights.sum().item())
    if total <= 0.0:
        idx = _weighted_sample_one(torch.ones((max(1, len(keys)),), dtype=torch.float32), generator)
        return keys[min(int(idx), len(keys) - 1)]
    if deterministic_value is None:
        idx = _weighted_sample_one(weights, generator)
        return keys[int(idx)]
    u = float(deterministic_value) % 1.0
    cdf = torch.cumsum(weights, dim=0)
    target = u * total
    idx = int(torch.searchsorted(cdf, torch.tensor(target, dtype=cdf.dtype)).item())
    return keys[min(max(idx, 0), len(keys) - 1)]


def _largest_remainder_counts(mapping: dict[str, float], count: int) -> dict[str, int]:
    """Return deterministic integer family quotas whose sum is ``count``."""
    keys = [str(key) for key in mapping]
    total_count = max(0, int(count))
    if total_count <= 0 or not keys:
        return dict.fromkeys(keys, 0)
    weights = [max(0.0, float(mapping[key])) for key in keys]
    total_weight = sum(weights)
    if total_weight <= 0.0:
        base = total_count // len(keys)
        remainder = total_count - base * len(keys)
        return {key: base + (1 if idx < remainder else 0) for idx, key in enumerate(keys)}
    exact = [total_count * weight / total_weight for weight in weights]
    floors = [math.floor(value) for value in exact]
    remainder = total_count - sum(floors)
    order = sorted(
        range(len(keys)),
        key=lambda idx: (exact[idx] - floors[idx], weights[idx], -idx),
        reverse=True,
    )
    counts = dict(zip(keys, floors, strict=True))
    for idx in order[:remainder]:
        counts[keys[idx]] += 1
    return counts


def _quota_sequence(
    mapping: dict[str, float], count: int, *, seed: int, namespace: str
) -> list[str]:
    """Return a deterministic prefix-balanced sequence matching weighted quotas exactly."""
    quotas = _largest_remainder_counts(mapping, count)
    total_count = sum(int(value) for value in quotas.values())
    if total_count <= 0:
        return []
    used = {str(family): 0 for family in quotas}
    sequence: list[str] = []
    for slot_index in range(total_count):
        candidates = [
            str(family) for family, quota in quotas.items() if used[str(family)] < int(quota)
        ]
        if not candidates:
            break

        def candidate_key(family: str, slot_index: int = slot_index) -> tuple[float, int, float]:
            quota = int(quotas[family])
            desired = float(slot_index + 1) * float(quota) / float(total_count)
            deficit = desired - float(used[family])
            tie_breaker = _deterministic_unit_from_key_parts(namespace, seed, slot_index, family)
            return deficit, quota, tie_breaker

        chosen = max(candidates, key=candidate_key)
        used[chosen] += 1
        sequence.append(chosen)
    return sequence


def _profile_query_plan(
    profile: RangeWorkloadProfile,
    *,
    requested_queries: int,
    workload_seed: int,
) -> dict[str, Any]:
    """Return deterministic final-profile family assignments for planned query slots."""
    if profile.profile_id == LEGACY_GENERATOR_PROFILE.profile_id:
        return {
            "enabled": False,
            "requested_queries": int(max(0, requested_queries)),
            "anchor_family_sequence": [],
            "footprint_family_sequence": [],
            "anchor_family_planned_counts": {},
            "footprint_family_planned_counts": {},
        }
    count = max(1, int(requested_queries))
    anchor_sequence = _quota_sequence(
        profile.anchor_family_weights,
        count,
        seed=int(workload_seed),
        namespace=f"{profile.profile_id}:anchor_family",
    )
    footprint_sequence = _quota_sequence(
        profile.footprint_family_weights,
        count,
        seed=int(workload_seed),
        namespace=f"{profile.profile_id}:footprint_family",
    )
    return {
        "enabled": True,
        "requested_queries": int(count),
        "anchor_family_sequence": anchor_sequence,
        "footprint_family_sequence": footprint_sequence,
        "anchor_family_planned_counts": _largest_remainder_counts(
            profile.anchor_family_weights, count
        ),
        "footprint_family_planned_counts": _largest_remainder_counts(
            profile.footprint_family_weights, count
        ),
    }


def _profile_query_settings(
    profile: RangeWorkloadProfile,
    generator: torch.Generator,
    query_index: int | None = None,
    workload_seed: int | None = None,
    query_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sample query-level profile settings for one range_query_mix query."""
    if profile.profile_id == LEGACY_GENERATOR_PROFILE.profile_id:
        return {}
    query_key = str(int(workload_seed) if workload_seed is not None else "unseeded")
    if query_index is None or query_index < 0:
        anchor_value: float | None = None
        footprint_value: float | None = None
    else:
        anchor_value = _deterministic_unit_from_key_parts(
            profile.profile_id, "anchor", query_key, query_index
        )
        footprint_value = _deterministic_unit_from_key_parts(
            profile.profile_id,
            "footprint",
            query_key,
            query_index,
        )
    anchor_sequence = (
        query_plan.get("anchor_family_sequence") if isinstance(query_plan, dict) else None
    )
    footprint_sequence = (
        query_plan.get("footprint_family_sequence") if isinstance(query_plan, dict) else None
    )
    if (
        isinstance(query_index, int)
        and query_index >= 0
        and isinstance(anchor_sequence, list)
        and query_index < len(anchor_sequence)
    ):
        anchor_family = str(anchor_sequence[query_index])
    else:
        anchor_family = _weighted_choice_with_deterministic_key(
            profile.anchor_family_weights,
            generator,
            fallback="density",
            deterministic_value=anchor_value,
        )
    if (
        isinstance(query_index, int)
        and query_index >= 0
        and isinstance(footprint_sequence, list)
        and query_index < len(footprint_sequence)
    ):
        footprint_family = str(footprint_sequence[query_index])
    else:
        footprint_family = _weighted_choice_with_deterministic_key(
            profile.footprint_family_weights,
            generator,
            fallback="medium_operational",
            deterministic_value=footprint_value,
        )
    footprint = dict(profile.footprint_families.get(footprint_family) or {})
    footprint_occurrence_index = int(query_index or 0)
    if (
        isinstance(query_index, int)
        and query_index >= 0
        and isinstance(footprint_sequence, list)
        and query_index < len(footprint_sequence)
    ):
        footprint_occurrence_index = (
            sum(
                1
                for family in footprint_sequence[: query_index + 1]
                if str(family) == footprint_family
            )
            - 1
        )
    min_point_hit_fraction = footprint.get("min_point_hit_fraction")
    max_point_hit_fraction = footprint.get("max_point_hit_fraction")
    target_point_hit_fraction: float | None = None
    if isinstance(min_point_hit_fraction, (int, float)) and isinstance(
        max_point_hit_fraction, (int, float)
    ):
        min_fraction = float(min_point_hit_fraction)
        max_fraction = float(max_point_hit_fraction)
        if max_fraction >= min_fraction:
            target_unit = ((max(0, footprint_occurrence_index) * LOW_DISCREPANCY_STEP) + 0.5) % 1.0
            target_point_hit_fraction = (
                min_fraction
                + (max_fraction - min_fraction) * POINT_HIT_TARGET_BAND_FRACTION * target_unit
            )
    return {
        "anchor_family": anchor_family,
        "footprint_family": footprint_family,
        "range_spatial_km": float(footprint.get("spatial_radius_km", 2.2)),
        "range_time_hours": float(footprint.get("time_half_window_hours", 5.0)),
        "elongation_allowed": bool(footprint.get("elongation_allowed", False)),
        "min_point_hit_fraction": min_point_hit_fraction,
        "max_point_hit_fraction": max_point_hit_fraction,
        "target_point_hit_fraction": target_point_hit_fraction,
    }
