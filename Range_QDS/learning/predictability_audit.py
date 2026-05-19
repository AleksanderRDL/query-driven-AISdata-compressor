"""Offline predictability audit for train-derived query priors."""

from __future__ import annotations

import itertools
from typing import Any

import torch

from learning.query_prior_fields import QUERY_PRIOR_FIELD_NAMES, sample_query_prior_fields
from learning.targets.query_local_utility import (
    QUERY_LOCAL_UTILITY_HEAD_NAMES,
    QUERY_LOCAL_UTILITY_TARGET_MODES,
    build_query_local_utility_targets,
)
from learning.targets.query_local_utility_family import (
    DIAGNOSTIC_TRAINABILITY_FOCUS_FAMILIES,
    FAMILY_TRAINABILITY_GROUP_KEYS,
    _range_query_family_evidence,
)

PREDICTABILITY_AUDIT_SCHEMA_VERSION = 3
PREDICTABILITY_GATE_THRESHOLDS = {
    "lift_at_1_percent": 1.10,
    "lift_at_2_percent": 1.15,
    "lift_at_5_percent": 1.20,
    "spearman_min": 0.15,
    "pr_auc_lift_over_base_rate": 1.25,
}
PRIOR_ALIGNMENT_GATE_THRESHOLDS = {
    "query_hit_spearman_min": 0.10,
    "query_hit_lift_at_5_percent_min": 1.10,
    "segment_budget_lift_at_5_percent_min": 1.05,
    "min_positive_spearman_head_count": 2,
}
HEAD_PRIOR_SCORE_MAP = {
    "query_hit_probability": "query_mass_prior",
    "conditional_behavior_utility": "behavior_utility_prior",
    "boundary_event_utility": "boundary_event_prior",
    "replacement_representative_value": "replacement_representative_prior",
    "segment_budget_target": "segment_budget_prior",
    "path_length_support_target": "behavior_utility_prior",
}


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    """Return deterministic average ranks, matching standard Spearman tie handling."""
    flat = values.detach().cpu().float().flatten()
    if int(flat.numel()) == 0:
        return flat
    order = torch.argsort(flat, stable=True)
    sorted_values = flat[order]
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    ends = torch.cumsum(counts, dim=0).to(dtype=torch.float32)
    starts = ends - counts.to(dtype=torch.float32)
    average_ranks = (starts + ends - 1.0) * 0.5
    sorted_ranks = torch.repeat_interleave(average_ranks, counts)
    ranks = torch.empty_like(flat)
    ranks[order] = sorted_ranks.to(dtype=torch.float32)
    return ranks


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return Pearson correlation for finite 1-D tensors."""
    left = left.detach().cpu().float().flatten()
    right = right.detach().cpu().float().flatten()
    finite = torch.isfinite(left) & torch.isfinite(right)
    if int(finite.sum().item()) < 2:
        return 0.0
    left = left[finite]
    right = right[finite]
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denom = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(right_centered)
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float((left_centered * right_centered).sum().item() / float(denom.item()))


def _spearman(score: torch.Tensor, target: torch.Tensor) -> float:
    """Return rank correlation between score and target."""
    return _pearson(_rankdata(score), _rankdata(target))


def _kendall_tau_sampled(
    score: torch.Tensor, target: torch.Tensor, max_pairs: int = 50_000
) -> float:
    """Return deterministic sampled Kendall tau for audit-scale diagnostics."""
    score = score.detach().cpu().float().flatten()
    target = target.detach().cpu().float().flatten()
    n = int(score.numel())
    if n < 2:
        return 0.0
    pair_count = n * (n - 1) // 2
    if pair_count <= max_pairs:
        left, right = torch.triu_indices(n, n, offset=1)
    else:
        generator = torch.Generator().manual_seed(1_706_011)
        left = torch.randint(0, n, (max_pairs,), generator=generator)
        right = torch.randint(0, n, (max_pairs,), generator=generator)
        keep = left != right
        left = left[keep]
        right = right[keep]
    score_delta = score[left] - score[right]
    target_delta = target[left] - target[right]
    valid = (score_delta.abs() > 1e-8) & (target_delta.abs() > 1e-8)
    if not bool(valid.any().item()):
        return 0.0
    concordant = (score_delta[valid] * target_delta[valid]) > 0.0
    return float((2.0 * concordant.float().mean() - 1.0).item())


def _auc(score: torch.Tensor, positive: torch.Tensor) -> float | None:
    """Return ROC AUC via rank-sum, or None when undefined."""
    score = score.detach().cpu().float().flatten()
    positive = positive.detach().cpu().bool().flatten()
    pos_count = int(positive.sum().item())
    neg_count = int((~positive).sum().item())
    if pos_count <= 0 or neg_count <= 0:
        return None
    ranks = _rankdata(score) + 1.0
    rank_sum_pos = float(ranks[positive].sum().item())
    auc = (rank_sum_pos - pos_count * (pos_count + 1) / 2.0) / max(
        1.0, float(pos_count * neg_count)
    )
    return float(max(0.0, min(1.0, auc)))


def _pr_auc(score: torch.Tensor, positive: torch.Tensor) -> float | None:
    """Return average precision as a PR-AUC proxy."""
    score = score.detach().cpu().float().flatten()
    positive = positive.detach().cpu().bool().flatten()
    pos_count = int(positive.sum().item())
    if pos_count <= 0:
        return None
    order = torch.argsort(score, descending=True, stable=True)
    sorted_positive = positive[order].float()
    cumulative_positive = torch.cumsum(sorted_positive, dim=0)
    ranks = torch.arange(1, int(score.numel()) + 1, dtype=torch.float32)
    precision_at_hits = cumulative_positive / ranks
    return float((precision_at_hits * sorted_positive).sum().item() / max(1, pos_count))


def _topk_indices(score: torch.Tensor, ratio: float) -> torch.Tensor:
    """Return global top-k indices for a budget ratio."""
    n = int(score.numel())
    if n <= 0:
        return torch.empty((0,), dtype=torch.long)
    keep = min(n, max(1, int(torch.ceil(torch.tensor(float(ratio) * n)).item())))
    return torch.topk(score.detach().cpu().float(), k=keep, largest=True).indices


def _ndcg_at(score: torch.Tensor, target: torch.Tensor, ratio: float) -> float:
    """Return NDCG at global budget ratio."""
    score_cpu = score.detach().cpu().float().flatten()
    target_cpu = target.detach().cpu().float().flatten().clamp(min=0.0)
    idx = _topk_indices(score_cpu, ratio)
    ideal_idx = _topk_indices(target_cpu, ratio)
    if int(idx.numel()) == 0 or int(ideal_idx.numel()) == 0:
        return 0.0
    gains = target_cpu[idx]
    ideal_gains = target_cpu[ideal_idx]
    discounts = 1.0 / torch.log2(torch.arange(2, int(idx.numel()) + 2, dtype=torch.float32))
    dcg = float((gains * discounts).sum().item())
    idcg = float((ideal_gains * discounts).sum().item())
    if idcg <= 1e-12:
        return 0.0
    return float(dcg / idcg)


def _lift_at(score: torch.Tensor, target: torch.Tensor, ratio: float) -> float:
    """Return top-budget mean-target lift over base target mean."""
    score_cpu = score.detach().cpu().float().flatten()
    target_cpu = target.detach().cpu().float().flatten().clamp(min=0.0)
    if int(score_cpu.numel()) == 0:
        return 0.0
    base = float(target_cpu.mean().item())
    if base <= 1e-12:
        return 0.0
    idx = _topk_indices(score_cpu, ratio)
    if int(idx.numel()) == 0:
        return 0.0
    return float(target_cpu[idx].mean().item() / base)


def _score_decile_calibration(
    score: torch.Tensor, target: torch.Tensor, bucket_count: int = 10
) -> dict[str, Any]:
    """Return equal-count score-bucket calibration diagnostics ordered low-to-high."""
    score_cpu = score.detach().cpu().float().flatten()
    target_cpu = target.detach().cpu().float().flatten().clamp(min=0.0)
    valid = torch.isfinite(score_cpu) & torch.isfinite(target_cpu)
    score_cpu = score_cpu[valid]
    target_cpu = target_cpu[valid]
    if int(score_cpu.numel()) == 0:
        return {
            "available": False,
            "bucket_count": 0,
            "score_mean_low_to_high": [],
            "target_mean_low_to_high": [],
            "positive_rate_low_to_high": [],
            "target_mean_adjacent_monotonicity_violations": 0,
        }
    order = torch.argsort(score_cpu, stable=True)
    buckets = torch.chunk(order, max(1, int(bucket_count)))
    score_means: list[float] = []
    target_means: list[float] = []
    positive_rates: list[float] = []
    for bucket in buckets:
        if int(bucket.numel()) == 0:
            continue
        bucket_target = target_cpu[bucket]
        score_means.append(float(score_cpu[bucket].mean().item()))
        target_means.append(float(bucket_target.mean().item()))
        positive_rates.append(float((bucket_target > 0.0).float().mean().item()))
    violations = sum(1 for left, right in itertools.pairwise(target_means) if right + 1e-12 < left)
    return {
        "available": True,
        "bucket_count": len(target_means),
        "score_mean_low_to_high": score_means,
        "target_mean_low_to_high": target_means,
        "positive_rate_low_to_high": positive_rates,
        "target_mean_adjacent_monotonicity_violations": int(violations),
    }


def _score_target_metrics(
    *,
    score: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Return predictability metrics for one score-target pair."""
    score_cpu = score.detach().cpu().float().flatten()
    target_cpu = target.detach().cpu().float().flatten().clamp(min=0.0)
    if valid_mask is None:
        valid = torch.ones_like(target_cpu, dtype=torch.bool)
    else:
        valid = valid_mask.detach().cpu().bool().flatten()
    valid = valid & torch.isfinite(score_cpu) & torch.isfinite(target_cpu)
    if not bool(valid.any().item()):
        return {
            "available": False,
            "reason": "no_valid_pairs",
            "valid_count": 0,
        }
    score_valid = score_cpu[valid]
    target_valid = target_cpu[valid]
    positive = target_valid > 0.0
    base_rate = float(positive.float().mean().item()) if int(target_valid.numel()) > 0 else 0.0
    pr_auc = _pr_auc(score_valid, positive)
    auc = _auc(score_valid, positive)
    pr_auc_lift = float(pr_auc / max(base_rate, 1e-12)) if pr_auc is not None else None
    budget_ratios = (0.01, 0.02, 0.05, 0.10)
    lifts = {
        f"lift_at_{int(ratio * 100)}_percent": _lift_at(score_valid, target_valid, ratio)
        for ratio in budget_ratios
    }
    ndcg = {
        f"ndcg_at_{int(ratio * 100)}_percent": _ndcg_at(score_valid, target_valid, ratio)
        for ratio in budget_ratios
    }
    positive_target_spearman = (
        _spearman(score_valid[positive], target_valid[positive])
        if int(positive.sum().item()) >= 2
        else None
    )
    return {
        "available": True,
        "valid_count": int(target_valid.numel()),
        "positive_count": int(positive.sum().item()),
        "base_positive_rate": base_rate,
        "target_mean": float(target_valid.mean().item()),
        "target_mass": float(target_valid.sum().item()),
        "score_mean": float(score_valid.mean().item()),
        "score_std": float(score_valid.std(unbiased=False).item())
        if int(score_valid.numel()) > 1
        else 0.0,
        "score_unique_count": int(torch.unique(score_valid).numel()),
        "target_unique_count": int(torch.unique(target_valid).numel()),
        "score_zero_fraction": float((score_valid == 0.0).float().mean().item()),
        "target_zero_fraction": float((target_valid == 0.0).float().mean().item()),
        "rank_correlation_method": "average_tie_ranks",
        "spearman": _spearman(score_valid, target_valid),
        "positive_target_spearman": positive_target_spearman,
        "kendall_tau": _kendall_tau_sampled(score_valid, target_valid),
        "auc": auc,
        "pr_auc": pr_auc,
        "pr_auc_lift_over_base_rate": pr_auc_lift,
        "score_decile_calibration": _score_decile_calibration(score_valid, target_valid),
        **lifts,
        **ndcg,
    }


def _prior_channel_scores(
    points: torch.Tensor, query_prior_field: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """Return sampled prior channels and derived prior scores."""
    sampled = sample_query_prior_fields(points, query_prior_field).detach().cpu().float()
    point_count = int(points.shape[0])
    channels: dict[str, torch.Tensor] = {}
    for idx, name in enumerate(QUERY_PRIOR_FIELD_NAMES):
        if idx < int(sampled.shape[1]):
            channels[name] = sampled[:, idx].clamp(0.0, 1.0)
        else:
            channels[name] = torch.zeros((point_count,), dtype=torch.float32)
    query_mass = torch.clamp(
        0.70 * channels["spatial_query_hit_probability"]
        + 0.30 * channels["spatiotemporal_query_hit_probability"],
        0.0,
        1.0,
    )
    boundary_event = torch.maximum(
        channels["endpoint_likelihood"],
        channels["crossing_likelihood"],
    )
    behavior = channels["behavior_utility_prior"]
    replacement = torch.clamp(
        query_mass * (0.50 + behavior) + 0.25 * boundary_event.square(), 0.0, 1.0
    )
    # Segment-budget labels are segment aggregations of query-local utility.
    # Raw route density is kept as a separate channel; blending it here can make
    # the alignment gate test density rather than the segment-budget target.
    segment_budget = replacement
    channels.update(
        {
            "query_mass_prior": query_mass,
            "boundary_event_prior": boundary_event,
            "replacement_representative_prior": replacement,
            "segment_budget_prior": segment_budget,
            "combined_prior_score": _prior_predictability_score(points, query_prior_field)
            .detach()
            .cpu()
            .float(),
        }
    )
    return channels


def _per_head_predictability(
    *,
    points: torch.Tensor,
    query_prior_field: dict[str, Any],
    head_targets: torch.Tensor,
    head_mask: torch.Tensor,
) -> dict[str, Any]:
    """Return per-head prior predictability diagnostics for factorized targets."""
    channels = _prior_channel_scores(points, query_prior_field)
    per_head: dict[str, Any] = {}
    positive_spearman_count = 0
    for head_idx, head_name in enumerate(QUERY_LOCAL_UTILITY_HEAD_NAMES):
        if head_idx >= int(head_targets.shape[1]):
            continue
        score_name = HEAD_PRIOR_SCORE_MAP.get(head_name, "combined_prior_score")
        metrics = _score_target_metrics(
            score=channels[score_name],
            target=head_targets[:, head_idx],
            valid_mask=head_mask[:, head_idx],
        )
        metrics["score_source"] = score_name
        per_head[head_name] = metrics
        if metrics.get("available") and float(metrics.get("spearman", 0.0)) > 0.0:
            positive_spearman_count += 1

    query_hit = per_head.get("query_hit_probability", {})
    segment_budget = per_head.get("segment_budget_target", {})
    failed_checks: list[str] = []
    if (
        float(query_hit.get("spearman", 0.0))
        < PRIOR_ALIGNMENT_GATE_THRESHOLDS["query_hit_spearman_min"]
    ):
        failed_checks.append("query_hit_spearman_below_min")
    if (
        float(query_hit.get("lift_at_5_percent", 0.0))
        < PRIOR_ALIGNMENT_GATE_THRESHOLDS["query_hit_lift_at_5_percent_min"]
    ):
        failed_checks.append("query_hit_lift_at_5_percent_below_min")
    if (
        float(segment_budget.get("lift_at_5_percent", 0.0))
        < PRIOR_ALIGNMENT_GATE_THRESHOLDS["segment_budget_lift_at_5_percent_min"]
    ):
        failed_checks.append("segment_budget_lift_at_5_percent_below_min")
    if positive_spearman_count < int(
        PRIOR_ALIGNMENT_GATE_THRESHOLDS["min_positive_spearman_head_count"]
    ):
        failed_checks.append("too_few_positive_spearman_heads")

    channel_vs_final: dict[str, Any] = {}
    final_target = head_targets[
        :, list(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("segment_budget_target")
    ]
    final_mask = head_mask[:, list(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("segment_budget_target")]
    for channel_name, score in channels.items():
        channel_vs_final[channel_name] = _score_target_metrics(
            score=score,
            target=final_target,
            valid_mask=final_mask,
        )
    channel_vs_head: dict[str, Any] = {}
    best_channel_by_head: dict[str, Any] = {}
    for head_idx, head_name in enumerate(QUERY_LOCAL_UTILITY_HEAD_NAMES):
        if head_idx >= int(head_targets.shape[1]):
            continue
        head_rows: dict[str, Any] = {}
        for channel_name, score in channels.items():
            head_rows[channel_name] = _score_target_metrics(
                score=score,
                target=head_targets[:, head_idx],
                valid_mask=head_mask[:, head_idx],
            )
        channel_vs_head[head_name] = head_rows
        available_rows = {
            channel_name: metrics
            for channel_name, metrics in head_rows.items()
            if isinstance(metrics, dict) and metrics.get("available")
        }
        if available_rows:
            best_lift_name, best_lift_metrics = max(
                available_rows.items(),
                key=lambda item: float(item[1].get("lift_at_5_percent", 0.0) or 0.0),
            )
            best_spearman_name, best_spearman_metrics = max(
                available_rows.items(),
                key=lambda item: float(item[1].get("spearman", 0.0) or 0.0),
            )
            best_channel_by_head[head_name] = {
                "best_lift_at_5_percent": {
                    "channel": best_lift_name,
                    "value": float(best_lift_metrics.get("lift_at_5_percent", 0.0) or 0.0),
                },
                "best_spearman": {
                    "channel": best_spearman_name,
                    "value": float(best_spearman_metrics.get("spearman", 0.0) or 0.0),
                },
            }
    return {
        "schema_version": 1,
        "per_head": per_head,
        "channel_vs_segment_budget_target": channel_vs_final,
        "channel_vs_head_target": channel_vs_head,
        "best_channel_by_head": best_channel_by_head,
        "prior_predictive_alignment_gate": {
            "gate_pass": not failed_checks,
            "failed_checks": failed_checks,
            "thresholds": dict(PRIOR_ALIGNMENT_GATE_THRESHOLDS),
            "positive_spearman_head_count": int(positive_spearman_count),
        },
    }


def _metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": bool(metrics.get("available", False)),
        "valid_count": metrics.get("valid_count"),
        "positive_count": metrics.get("positive_count"),
        "spearman": metrics.get("spearman"),
        "positive_target_spearman": metrics.get("positive_target_spearman"),
        "lift_at_5_percent": metrics.get("lift_at_5_percent"),
        "pr_auc_lift_over_base_rate": metrics.get("pr_auc_lift_over_base_rate"),
        "score_std": metrics.get("score_std"),
        "target_mean": metrics.get("target_mean"),
    }


def _family_conditioned_prior_predictability(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    range_queries: list[dict[str, Any]],
    query_prior_field: dict[str, Any],
    head_targets: torch.Tensor,
    head_mask: torch.Tensor,
) -> dict[str, Any]:
    """Return focus-family prior predictability diagnostics without changing gates."""
    family_evidence = _range_query_family_evidence(
        points=points,
        boundaries=boundaries,
        range_queries=range_queries,
        group_keys=FAMILY_TRAINABILITY_GROUP_KEYS,
    )
    channels = _prior_channel_scores(points, query_prior_field)
    out: dict[str, Any] = {
        "schema_version": 1,
        "available": bool(range_queries) and bool(family_evidence),
        "diagnostic_only": True,
        "used_for_gate": False,
        "used_for_training": False,
        "used_for_checkpoint_selection": False,
        "used_for_retained_mask_decision": False,
        "focus_families": {
            group_key: sorted(values)
            for group_key, values in DIAGNOSTIC_TRAINABILITY_FOCUS_FAMILIES.items()
        },
        "group_by": {},
        "interpretation": (
            "Family-conditioned prior predictability localizes whether "
            "train-derived prior channels expose held-out family target signal. "
            "It is diagnostic only and does not change gates."
        ),
    }
    if not range_queries:
        out["available"] = False
        out["reason"] = "no_range_queries"
        return out

    for group_key, family_rows in family_evidence.items():
        group_out: dict[str, Any] = {}
        for family, evidence in family_rows.items():
            focus_family = family in DIAGNOSTIC_TRAINABILITY_FOCUS_FAMILIES.get(
                group_key, frozenset()
            )
            if not focus_family:
                continue
            family_valid = evidence["query_hit_probability"].detach().cpu().float() > 0.0
            family_out: dict[str, Any] = {
                "available": bool(family_valid.any().item()),
                "focus_family": True,
                "query_count": int(evidence["query_count"]),
                "valid_hit_point_count": int(family_valid.sum().item()),
                "heads": {},
                "weak_family_prior_heads": [],
            }
            if not family_out["available"]:
                family_out["reason"] = "no_family_hit_points"
                group_out[str(family)] = family_out
                continue
            weak_heads: list[str] = []
            for head_idx, head_name in enumerate(QUERY_LOCAL_UTILITY_HEAD_NAMES):
                if head_idx >= int(head_targets.shape[1]):
                    continue
                valid_mask = head_mask[:, head_idx].detach().cpu().bool() & family_valid
                target = head_targets[:, head_idx].detach().cpu().float()
                channel_rows = {
                    channel_name: _score_target_metrics(
                        score=score,
                        target=target,
                        valid_mask=valid_mask,
                    )
                    for channel_name, score in channels.items()
                }
                available_rows = {
                    channel_name: metrics
                    for channel_name, metrics in channel_rows.items()
                    if isinstance(metrics, dict) and metrics.get("available")
                }
                if not available_rows:
                    family_out["heads"][head_name] = {
                        "available": False,
                        "reason": "no_valid_family_head_pairs",
                    }
                    weak_heads.append(str(head_name))
                    continue
                mapped_channel = HEAD_PRIOR_SCORE_MAP.get(head_name, "combined_prior_score")
                mapped_metrics = available_rows.get(mapped_channel, {})
                best_spearman_name, best_spearman_metrics = max(
                    available_rows.items(),
                    key=lambda item: float(item[1].get("spearman", 0.0) or 0.0),
                )
                best_lift_name, best_lift_metrics = max(
                    available_rows.items(),
                    key=lambda item: float(item[1].get("lift_at_5_percent", 0.0) or 0.0),
                )
                best_spearman = float(best_spearman_metrics.get("spearman", 0.0) or 0.0)
                best_lift = float(best_lift_metrics.get("lift_at_5_percent", 0.0) or 0.0)
                weak_prior = best_spearman <= 0.0 or best_lift < 1.05
                if weak_prior:
                    weak_heads.append(str(head_name))
                family_out["heads"][head_name] = {
                    "available": True,
                    "mapped_prior_channel": mapped_channel,
                    "mapped_prior_metrics": _metric_subset(mapped_metrics),
                    "best_spearman": {
                        "channel": best_spearman_name,
                        "value": best_spearman,
                        "metrics": _metric_subset(best_spearman_metrics),
                    },
                    "best_lift_at_5_percent": {
                        "channel": best_lift_name,
                        "value": best_lift,
                        "metrics": _metric_subset(best_lift_metrics),
                    },
                    "family_prior_status": (
                        "weak_family_prior_alignment" if weak_prior else "diagnostic_only"
                    ),
                }
            family_out["weak_family_prior_heads"] = weak_heads
            family_out["family_prior_status"] = (
                "weak_focus_family_prior_alignment" if weak_heads else "diagnostic_only"
            )
            group_out[str(family)] = family_out
        out["group_by"][group_key] = group_out
    return out


def _prior_predictability_score(
    points: torch.Tensor, query_prior_field: dict[str, Any]
) -> torch.Tensor:
    """Build a simple train-prior score from sampled prior-field channels."""
    sampled = sample_query_prior_fields(points, query_prior_field).float()
    if sampled.shape[1] < 6:
        return torch.zeros((int(points.shape[0]),), dtype=torch.float32, device=points.device)
    spatial_query_hit_probability = sampled[:, 0].clamp(0.0, 1.0)
    spatiotemporal_query_hit_probability = sampled[:, 1].clamp(0.0, 1.0)
    endpoint_likelihood = sampled[:, 2].clamp(0.0, 1.0)
    crossing_likelihood = sampled[:, 3].clamp(0.0, 1.0)
    behavior_utility_prior = sampled[:, 4].clamp(0.0, 1.0)

    # The aggregate target is query-local usefulness. Behavior utility is useful
    # only where future query mass is plausible; do not let query-free behavior
    # bypass the query-hit prior and dominate final-target predictability.
    query_mass = torch.clamp(
        0.70 * spatial_query_hit_probability + 0.30 * spatiotemporal_query_hit_probability,
        0.0,
        1.0,
    )
    boundary_event = torch.maximum(endpoint_likelihood, crossing_likelihood)
    score = torch.clamp(
        query_mass * (0.50 + behavior_utility_prior) + 0.25 * boundary_event.square(),
        0.0,
        1.0,
    )
    return score


def query_prior_predictability_scores(
    points: torch.Tensor, query_prior_field: dict[str, Any]
) -> torch.Tensor:
    """Return the query-prior-only score used by predictability and causality diagnostics."""
    return _prior_predictability_score(points, query_prior_field)


def query_prior_predictability_audit(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    eval_typed_queries: list[dict[str, Any]],
    query_prior_field: dict[str, Any] | None,
    target_mode: str = "query_local_utility_factorized",
) -> dict[str, Any]:
    """Measure whether train-derived query-prior fields predict held-out eval usefulness."""
    if query_prior_field is None:
        return {
            "schema_version": PREDICTABILITY_AUDIT_SCHEMA_VERSION,
            "available": False,
            "gate_pass": False,
            "reason": "missing_query_prior_field",
        }
    eval_targets = build_query_local_utility_targets(
        points=points,
        boundaries=boundaries,
        typed_queries=eval_typed_queries,
        target_mode=(
            str(target_mode).lower()
            if str(target_mode).lower() in QUERY_LOCAL_UTILITY_TARGET_MODES
            else "query_local_utility_factorized"
        ),
    )
    target = eval_targets.labels[:, 0].float().detach().cpu()
    score = _prior_predictability_score(points, query_prior_field).detach().cpu()
    positive = target > 0.0
    base_rate = float(positive.float().mean().item()) if int(target.numel()) > 0 else 0.0
    pr_auc = _pr_auc(score, positive)
    auc = _auc(score, positive)
    budget_ratios = (0.01, 0.02, 0.05, 0.10)
    lifts = {
        f"lift_at_{int(ratio * 100)}_percent": _lift_at(score, target, ratio)
        for ratio in budget_ratios
    }
    ndcg = {
        f"ndcg_at_{int(ratio * 100)}_percent": _ndcg_at(score, target, ratio)
        for ratio in budget_ratios
    }
    pr_auc_lift = float(pr_auc / max(base_rate, 1e-12)) if pr_auc is not None else None
    spearman = _spearman(score, target)
    metrics: dict[str, Any] = {
        "rank_correlation_method": "average_tie_ranks",
        "score_unique_count": int(torch.unique(score).numel()),
        "target_unique_count": int(torch.unique(target).numel()),
        "score_zero_fraction": float((score == 0.0).float().mean().item()),
        "target_zero_fraction": float((target == 0.0).float().mean().item()),
        "spearman": spearman,
        "positive_target_spearman": _spearman(score[positive], target[positive])
        if int(positive.sum().item()) >= 2
        else None,
        "kendall_tau": _kendall_tau_sampled(score, target),
        "auc": auc,
        "pr_auc": pr_auc,
        "base_positive_rate": base_rate,
        "pr_auc_lift_over_base_rate": pr_auc_lift,
        "score_decile_calibration": _score_decile_calibration(score, target),
        **lifts,
        **ndcg,
    }
    checks = {
        "lift_at_1_percent": lifts["lift_at_1_percent"]
        >= PREDICTABILITY_GATE_THRESHOLDS["lift_at_1_percent"],
        "lift_at_2_percent": lifts["lift_at_2_percent"]
        >= PREDICTABILITY_GATE_THRESHOLDS["lift_at_2_percent"],
        "lift_at_5_percent": lifts["lift_at_5_percent"]
        >= PREDICTABILITY_GATE_THRESHOLDS["lift_at_5_percent"],
        "spearman_min": spearman >= PREDICTABILITY_GATE_THRESHOLDS["spearman_min"],
        "pr_auc_lift_over_base_rate": (
            pr_auc_lift is not None
            and pr_auc_lift >= PREDICTABILITY_GATE_THRESHOLDS["pr_auc_lift_over_base_rate"]
        ),
    }
    per_head_predictability = _per_head_predictability(
        points=points,
        query_prior_field=query_prior_field,
        head_targets=eval_targets.head_targets.detach().cpu().float(),
        head_mask=eval_targets.head_mask.detach().cpu().bool(),
    )
    range_queries = [
        query
        for query in eval_typed_queries
        if str(query.get("type", "")).lower() == "range" and isinstance(query.get("params"), dict)
    ]
    family_prior_predictability = _family_conditioned_prior_predictability(
        points=points,
        boundaries=boundaries,
        range_queries=range_queries,
        query_prior_field=query_prior_field,
        head_targets=eval_targets.head_targets.detach().cpu().float(),
        head_mask=eval_targets.head_mask.detach().cpu().bool(),
    )
    return {
        "schema_version": PREDICTABILITY_AUDIT_SCHEMA_VERSION,
        "available": True,
        "scoring_stage": "after_masks_frozen_diagnostic_only",
        "score_source": "train_query_prior_fields",
        "target_source": "heldout_eval_query_local_utility_targets",
        "target_mode": str(target_mode).lower(),
        "score_formula": "query_mass_gated_behavior_boundary",
        "used_for_training": False,
        "used_for_checkpoint_selection": False,
        "used_for_retained_mask_decision": False,
        "thresholds": dict(PREDICTABILITY_GATE_THRESHOLDS),
        "metrics": metrics,
        "per_head_predictability": per_head_predictability["per_head"],
        "prior_channel_predictability": per_head_predictability["channel_vs_segment_budget_target"],
        "prior_channel_by_head_predictability": per_head_predictability["channel_vs_head_target"],
        "best_prior_channel_by_head": per_head_predictability["best_channel_by_head"],
        "family_conditioned_prior_predictability": family_prior_predictability,
        "prior_predictive_alignment_gate": per_head_predictability[
            "prior_predictive_alignment_gate"
        ],
        "gate_checks": checks,
        "gate_pass": all(checks.values()),
        "eval_point_count": int(points.shape[0]),
        "eval_positive_target_count": int(positive.sum().item()),
        "eval_query_count": len(
            [q for q in eval_typed_queries if str(q.get("type", "")).lower() == "range"]
        ),
    }
