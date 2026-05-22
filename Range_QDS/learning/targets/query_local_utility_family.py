"""Family-conditioned QueryLocalUtility evidence helpers."""

from __future__ import annotations

from typing import Any

import torch

from workloads.query_types import validated_range_query_params
from workloads.range_geometry import points_in_range_box

FAMILY_TRAINABILITY_GROUP_KEYS = ("anchor_family", "footprint_family")
# Diagnostic focus for current blocker localization; this is not a workload
# profile definition.
DIAGNOSTIC_TRAINABILITY_FOCUS_FAMILIES = {
    "anchor_family": frozenset({"density"}),
    "footprint_family": frozenset({"medium_operational"}),
}


def _query_family_label(query: dict[str, Any], group_key: str) -> str:
    metadata = query.get("_metadata")
    if not isinstance(metadata, dict):
        return "unspecified"
    raw_value = metadata.get(group_key)
    if raw_value is None:
        return "unspecified"
    value = str(raw_value).strip()
    return value if value else "unspecified"


def _range_query_family_evidence(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    range_queries: list[dict[str, Any]],
    group_keys: tuple[str, ...] = FAMILY_TRAINABILITY_GROUP_KEYS,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return per-family query-hit and one-credit-per-ship evidence tensors."""
    device = points.device
    n_points = int(points.shape[0])
    out: dict[str, dict[str, dict[str, Any]]] = {group_key: {} for group_key in group_keys}
    for group_key in group_keys:
        families = sorted({_query_family_label(query, group_key) for query in range_queries})
        for family in families:
            family_queries = [
                query for query in range_queries if _query_family_label(query, group_key) == family
            ]
            query_hit_count = torch.zeros((n_points,), dtype=torch.float32, device=device)
            ship_query_evidence_mass = torch.zeros_like(query_hit_count)
            query_hit_masks: list[torch.Tensor] = []
            for query in family_queries:
                params = validated_range_query_params(query)
                mask = points_in_range_box(points, params).to(
                    device=device,
                    dtype=torch.bool,
                )
                query_hit_masks.append(mask)
                if not bool(mask.any().item()):
                    continue
                query_hit_count[mask] += 1.0
                for start, end in boundaries:
                    local_mask = mask[int(start) : int(end)]
                    if not bool(local_mask.any().item()):
                        continue
                    local_hit_count = int(local_mask.sum().item())
                    if local_hit_count <= 0:
                        continue
                    local_indices = torch.where(local_mask)[0] + int(start)
                    ship_query_evidence_mass[local_indices] += float(1.0 / local_hit_count)
            query_count = float(max(1, len(family_queries)))
            out[group_key][family] = {
                "query_count": len(family_queries),
                "query_hit_probability": (query_hit_count / query_count).clamp(0.0, 1.0),
                "ship_query_evidence": (ship_query_evidence_mass / query_count).clamp(0.0, 1.0),
                "query_hit_masks": query_hit_masks,
            }
    return out
