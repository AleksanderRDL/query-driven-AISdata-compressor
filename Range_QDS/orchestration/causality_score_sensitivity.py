"""Score and retained-mask ablation sensitivity diagnostics."""

from __future__ import annotations

from typing import Any

import torch


def score_ablation_sensitivity(
    *,
    primary_scores: torch.Tensor | None,
    ablation_scores: torch.Tensor | None,
    primary_mask: torch.Tensor | None,
    ablation_mask: torch.Tensor | None,
) -> dict[str, Any]:
    """Return score- and mask-level sensitivity for a frozen ablation."""
    if primary_scores is None or ablation_scores is None:
        return {"available": False, "reason": "missing_scores"}
    primary = primary_scores.detach().cpu().float().flatten()
    ablation = ablation_scores.detach().cpu().float().flatten()
    if int(primary.numel()) == 0 or primary.shape != ablation.shape:
        return {
            "available": False,
            "reason": "score_shape_mismatch",
            "primary_score_count": int(primary.numel()),
            "ablation_score_count": int(ablation.numel()),
        }
    finite = torch.isfinite(primary) & torch.isfinite(ablation)
    if not bool(finite.any().item()):
        return {"available": False, "reason": "no_finite_scores"}
    primary_f = primary[finite]
    ablation_f = ablation[finite]
    delta = primary_f - ablation_f
    primary_std = float(primary_f.std(unbiased=False).item()) if int(primary_f.numel()) > 1 else 0.0
    ablation_std = (
        float(ablation_f.std(unbiased=False).item()) if int(ablation_f.numel()) > 1 else 0.0
    )

    topk_jaccard: float | None = None
    mask_diagnostics = retained_mask_comparison(
        primary_mask=primary_mask,
        ablation_mask=ablation_mask,
        expected_shape=primary.shape,
    )
    if primary_mask is not None and ablation_mask is not None:
        primary_bool = primary_mask.detach().cpu().bool().flatten()
        ablation_bool = ablation_mask.detach().cpu().bool().flatten()
        if primary_bool.shape == ablation_bool.shape == primary.shape:
            retained_count = int(primary_bool.sum().item())
            if retained_count > 0:
                k = min(retained_count, int(primary.numel()))
                primary_top = torch.zeros_like(primary_bool)
                ablation_top = torch.zeros_like(ablation_bool)
                primary_top[torch.topk(primary, k=k, largest=True).indices] = True
                ablation_top[torch.topk(ablation, k=k, largest=True).indices] = True
                top_intersection = int((primary_top & ablation_top).sum().item())
                top_union = int((primary_top | ablation_top).sum().item())
                topk_jaccard = float(top_intersection / max(1, top_union))

    return {
        "available": True,
        "score_count": int(primary.numel()),
        "finite_score_count": int(finite.sum().item()),
        "mean_abs_score_delta": float(delta.abs().mean().item()),
        "max_abs_score_delta": float(delta.abs().max().item()),
        "mean_signed_score_delta": float(delta.mean().item()),
        "primary_score_std": primary_std,
        "ablation_score_std": ablation_std,
        "retained_count": mask_diagnostics.get("primary_retained_count"),
        "retained_mask_changed": mask_diagnostics.get("retained_mask_changed"),
        "retained_mask_jaccard": mask_diagnostics.get("retained_mask_jaccard"),
        "retained_mask_hamming_fraction": mask_diagnostics.get("retained_mask_hamming_fraction"),
        "score_topk_jaccard_at_retained_count": topk_jaccard,
    }


def retained_mask_comparison(
    *,
    primary_mask: torch.Tensor | None,
    ablation_mask: torch.Tensor | None,
    expected_shape: torch.Size | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Return retained-mask overlap diagnostics for a frozen ablation."""
    if primary_mask is None or ablation_mask is None:
        return {"available": False, "reason": "missing_masks"}
    primary_bool = primary_mask.detach().cpu().bool().flatten()
    ablation_bool = ablation_mask.detach().cpu().bool().flatten()
    if expected_shape is not None:
        expected_numel = 1
        for dim in tuple(expected_shape):
            expected_numel *= int(dim)
        if (
            int(primary_bool.numel()) != expected_numel
            or int(ablation_bool.numel()) != expected_numel
        ):
            return {
                "available": False,
                "reason": "mask_shape_mismatch",
                "primary_mask_count": int(primary_bool.numel()),
                "ablation_mask_count": int(ablation_bool.numel()),
                "expected_mask_count": expected_numel,
            }
    if primary_bool.shape != ablation_bool.shape:
        return {
            "available": False,
            "reason": "mask_shape_mismatch",
            "primary_mask_count": int(primary_bool.numel()),
            "ablation_mask_count": int(ablation_bool.numel()),
        }
    intersection = int((primary_bool & ablation_bool).sum().item())
    union = int((primary_bool | ablation_bool).sum().item())
    primary_count = int(primary_bool.sum().item())
    ablation_count = int(ablation_bool.sum().item())
    symmetric_difference = int((primary_bool != ablation_bool).sum().item())
    return {
        "available": True,
        "primary_retained_count": primary_count,
        "ablation_retained_count": ablation_count,
        "retained_intersection_count": intersection,
        "retained_union_count": union,
        "retained_symmetric_difference_count": symmetric_difference,
        "retained_mask_changed": bool(symmetric_difference > 0),
        "retained_mask_jaccard": float(intersection / max(1, union)),
        "retained_mask_hamming_fraction": float(
            symmetric_difference / max(1, int(primary_bool.numel()))
        ),
    }
