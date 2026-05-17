"""Anchor sampling helpers for generated range query workloads."""

from __future__ import annotations

import torch

DENSITY_ANCHOR_PROBABILITY = 0.70
DENSITY_GRID_BINS = 64
RANGE_ANCHOR_MODES = ("mixed_density", "dense", "uniform", "sparse")
RANGE_WORKLOAD_V1_ANCHOR_FAMILIES = (
    "density_route",
    "boundary_entry_exit",
    "crossing_turn_change",
    "port_or_approach_zone",
    "sparse_background_control",
)


def _density_anchor_weights(points: torch.Tensor, bins: int = DENSITY_GRID_BINS) -> torch.Tensor:
    """Return per-point spatial density weights from a lat/lon grid density map."""
    if points.shape[0] == 0:
        return torch.empty((0,), dtype=torch.float32, device=points.device)

    bin_count = max(1, int(bins))
    lat = points[:, 1]
    lon = points[:, 2]
    lat_min = lat.min()
    lon_min = lon.min()
    lat_span = torch.clamp(lat.max() - lat_min, min=1e-6)
    lon_span = torch.clamp(lon.max() - lon_min, min=1e-6)

    lat_bins = torch.clamp(((lat - lat_min) / lat_span * (bin_count - 1)).long(), 0, bin_count - 1)
    lon_bins = torch.clamp(((lon - lon_min) / lon_span * (bin_count - 1)).long(), 0, bin_count - 1)
    bin_ids = lat_bins * bin_count + lon_bins
    cell_counts = torch.bincount(bin_ids.cpu(), minlength=bin_count * bin_count).to(
        device=points.device,
        dtype=torch.float32,
    )
    weights = cell_counts[bin_ids]
    total = weights.sum()
    if float(total.item()) <= 0.0:
        return torch.ones((points.shape[0],), dtype=torch.float32, device=points.device) / max(
            1, points.shape[0]
        )
    return weights / total


def _sparse_anchor_weights(points: torch.Tensor, bins: int = DENSITY_GRID_BINS) -> torch.Tensor:
    """Return per-point weights that sample occupied spatial cells more evenly."""
    if points.shape[0] == 0:
        return torch.empty((0,), dtype=torch.float32, device=points.device)

    bin_count = max(1, int(bins))
    lat = points[:, 1]
    lon = points[:, 2]
    lat_min = lat.min()
    lon_min = lon.min()
    lat_span = torch.clamp(lat.max() - lat_min, min=1e-6)
    lon_span = torch.clamp(lon.max() - lon_min, min=1e-6)

    lat_bins = torch.clamp(((lat - lat_min) / lat_span * (bin_count - 1)).long(), 0, bin_count - 1)
    lon_bins = torch.clamp(((lon - lon_min) / lon_span * (bin_count - 1)).long(), 0, bin_count - 1)
    bin_ids = lat_bins * bin_count + lon_bins
    cell_counts = torch.bincount(bin_ids.cpu(), minlength=bin_count * bin_count).to(
        device=points.device,
        dtype=torch.float32,
    )
    weights = 1.0 / torch.clamp(cell_counts[bin_ids], min=1.0)
    total = weights.sum()
    if float(total.item()) <= 0.0:
        return torch.ones((points.shape[0],), dtype=torch.float32, device=points.device) / max(
            1, points.shape[0]
        )
    return weights / total


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    """Return a probability vector, falling back to uniform when needed."""
    if int(weights.numel()) == 0:
        return weights.to(dtype=torch.float32)
    clean = torch.nan_to_num(weights.float().clamp(min=0.0), nan=0.0, posinf=0.0, neginf=0.0)
    total = float(clean.sum().item())
    if total <= 1e-12:
        return torch.ones_like(clean, dtype=torch.float32) / float(clean.numel())
    return clean / total


def _endpoint_anchor_weights(points: torch.Tensor) -> torch.Tensor:
    """Return endpoint-biased anchor weights from query-free trajectory flags."""
    if points.shape[0] == 0:
        return torch.empty((0,), dtype=torch.float32, device=points.device)
    weights = torch.ones((points.shape[0],), dtype=torch.float32, device=points.device)
    if points.shape[1] > 6:
        endpoint = (points[:, 5].float() > 0.5) | (points[:, 6].float() > 0.5)
        weights = weights + 8.0 * endpoint.float()
    if points.shape[1] > 7:
        weights = weights + 3.0 * points[:, 7].float().clamp(min=0.0)
    return _normalize_weights(weights)


def _turn_change_anchor_weights(points: torch.Tensor) -> torch.Tensor:
    """Return turn/change-biased anchor weights from query-free point features."""
    if points.shape[0] == 0:
        return torch.empty((0,), dtype=torch.float32, device=points.device)
    weights = torch.ones((points.shape[0],), dtype=torch.float32, device=points.device)
    if points.shape[1] > 7:
        weights = weights + 8.0 * points[:, 7].float().clamp(min=0.0)
    if points.shape[1] > 4:
        prev_idx = torch.clamp(torch.arange(points.shape[0], device=points.device) - 1, min=0)
        heading_delta = torch.abs(points[:, 4].float() - points[prev_idx, 4].float())
        heading_delta = torch.minimum(heading_delta, 360.0 - heading_delta).clamp(min=0.0) / 180.0
        weights = weights + 4.0 * heading_delta
    if points.shape[1] > 3:
        prev_idx = torch.clamp(torch.arange(points.shape[0], device=points.device) - 1, min=0)
        speed_delta = torch.abs(points[:, 3].float() - points[prev_idx, 3].float())
        if float(speed_delta.max().item()) > 1e-6:
            speed_delta = speed_delta / speed_delta.max().clamp(min=1e-6)
        weights = weights + 2.0 * speed_delta
    return _normalize_weights(weights)


def _port_or_approach_anchor_weights(points: torch.Tensor) -> torch.Tensor:
    """Return hotspot/approach weights distinct from generic route density."""
    if points.shape[0] == 0:
        return torch.empty((0,), dtype=torch.float32, device=points.device)
    density = _density_anchor_weights(points)
    endpoint = _endpoint_anchor_weights(points)
    weights = 0.35 * density + 0.35 * endpoint
    if points.shape[1] > 3:
        speed = points[:, 3].float().clamp(min=0.0)
        if int(speed.numel()) > 0:
            low_speed_cutoff = torch.quantile(speed, 0.30)
            slow = (speed <= low_speed_cutoff).float()
            weights = weights + 0.30 * _normalize_weights(slow)
    return _normalize_weights(weights)


def _normalize_range_anchor_mode(mode: str) -> str:
    """Normalize range-query anchor sampling mode names."""
    normalized = str(mode).strip().lower()
    if normalized not in RANGE_ANCHOR_MODES:
        raise ValueError(f"range_anchor_mode must be one of {RANGE_ANCHOR_MODES}; got {mode!r}.")
    return normalized


def _anchor_weights_for_mode(points: torch.Tensor, mode: str) -> tuple[torch.Tensor | None, float]:
    """Return optional point weights and use probability for the configured anchor mode."""
    normalized = _normalize_range_anchor_mode(mode)
    if normalized == "uniform":
        return None, 0.0
    if normalized == "sparse":
        return _sparse_anchor_weights(points), 1.0
    weights = _density_anchor_weights(points)
    if normalized == "dense":
        return weights, 1.0
    return weights, DENSITY_ANCHOR_PROBABILITY


def _anchor_weights_for_family(
    points: torch.Tensor,
    family: str,
) -> tuple[torch.Tensor | None, float]:
    """Return anchor weights for a range_workload_v1 anchor family."""
    normalized = str(family).strip().lower()
    if normalized == "sparse_background_control":
        return _sparse_anchor_weights(points), 1.0
    if normalized == "boundary_entry_exit":
        return _endpoint_anchor_weights(points), 1.0
    if normalized == "crossing_turn_change":
        return _turn_change_anchor_weights(points), 1.0
    if normalized == "port_or_approach_zone":
        return _port_or_approach_anchor_weights(points), 1.0
    if normalized == "density_route":
        return _density_anchor_weights(points), 1.0
    raise ValueError(f"Unknown range workload anchor family: {family!r}.")


def _weighted_sample_one(
    weights: torch.Tensor,
    generator: torch.Generator,
) -> int:
    """Sample one index from a non-negative weight vector.

    torch.multinomial caps at 2^24 categories, which the AIS combined-day CSVs
    exceed (23M+ points). Falls back to inverse-CDF sampling via cumsum +
    searchsorted, which has no size limit.
    """
    weight_count = int(weights.numel())
    if weight_count == 0:
        return 0
    total = float(weights.sum().item())
    if total <= 0.0:
        return int(torch.randint(0, weight_count, (1,), generator=generator).item())
    if weight_count <= (1 << 24):
        return int(torch.multinomial(weights, 1, generator=generator).item())
    cdf = torch.cumsum(weights, dim=0)
    sample_threshold = float(torch.rand(1, generator=generator).item()) * total
    return int(torch.searchsorted(cdf, torch.tensor(sample_threshold, dtype=cdf.dtype)).item())


def _sample_anchor_point(
    points: torch.Tensor,
    generator: torch.Generator,
    candidate_mask: torch.Tensor | None = None,
    anchor_weights: torch.Tensor | None = None,
    anchor_weight_probability: float = 1.0,
) -> torch.Tensor:
    """Sample one point row from the cloud. See queries/README.md for details."""
    if candidate_mask is not None and bool(candidate_mask.any().item()):
        candidate_indices = torch.where(candidate_mask)[0]
    else:
        candidate_indices = None

    use_weighted = (
        anchor_weights is not None
        and anchor_weights.numel() == points.shape[0]
        and float(torch.rand(1, generator=generator).item()) < float(anchor_weight_probability)
    )
    if use_weighted:
        if anchor_weights is None:
            raise RuntimeError("Weighted anchor sampling requested without anchor weights.")
        if candidate_indices is not None:
            candidate_weights = anchor_weights[candidate_indices].float()
            if float(candidate_weights.sum().item()) > 0.0:
                sampled_candidate_offset = _weighted_sample_one(candidate_weights, generator)
                return points[int(candidate_indices[sampled_candidate_offset].item())]
        else:
            weights = anchor_weights.float()
            if float(weights.sum().item()) > 0.0:
                sampled_point_idx = _weighted_sample_one(weights, generator)
                return points[sampled_point_idx]

    if candidate_indices is not None:
        candidate_offset = int(
            torch.randint(0, candidate_indices.shape[0], (1,), generator=generator).item()
        )
        point_idx = int(candidate_indices[candidate_offset].item())
    else:
        point_idx = int(torch.randint(0, points.shape[0], (1,), generator=generator).item())
    return points[point_idx]
