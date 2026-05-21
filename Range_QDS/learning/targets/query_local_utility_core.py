"""Core QueryLocalUtility target helpers."""

from __future__ import annotations

import torch


def query_local_utility_point_score(
    *,
    q_hit: torch.Tensor,
    behavior: torch.Tensor,
    boundary: torch.Tensor,
    replacement: torch.Tensor,
) -> torch.Tensor:
    """Return the scalar QueryLocalUtility point score used by labels and model logits."""
    q_hit = q_hit.float().clamp(0.0, 1.0)
    behavior = behavior.float().clamp(0.0, 1.0)
    boundary = boundary.float().clamp(0.0, 1.0)
    replacement = replacement.float().clamp(0.0, 1.0)
    query_local = 0.50 * q_hit + 0.45 * behavior
    replacement_multiplier = 0.75 + 0.25 * replacement
    boundary_bonus = 0.05 * boundary
    return (query_local * replacement_multiplier + boundary_bonus).clamp(0.0, 1.0)

def _normalize_0_1(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Normalize non-negative values to [0, 1] on the selected support."""
    out = values.float().clamp(min=0.0)
    support = mask if mask is not None else torch.ones_like(out, dtype=torch.bool)
    if not bool(support.any().item()):
        return torch.zeros_like(out)
    local = out[support]
    max_value = float(local.max().item()) if int(local.numel()) > 0 else 0.0
    if max_value <= 1e-12:
        return torch.zeros_like(out)
    return (out / max_value).clamp(0.0, 1.0)
