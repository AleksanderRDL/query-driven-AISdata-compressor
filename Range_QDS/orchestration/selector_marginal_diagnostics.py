"""Retained-decision marginal QueryLocalUtility diagnostics."""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from learning.targets.query_local_utility import (
    QUERY_LOCAL_UTILITY_HEAD_NAMES,
    build_query_local_utility_targets,
)
from orchestration import selector_marginal_alignment, selector_trace_payloads
from scoring.method_scoring import endpoint_sanity, score_range_audit
from scoring.metrics import compute_geometric_distortion, compute_length_preservation
from scoring.query_cache import ScoringQueryCache
from scoring.query_local_utility import query_local_utility_from_range_audit
from workloads.query_types import QUERY_TYPE_ID_RANGE, validated_range_query_params
from workloads.range_geometry import points_in_range_box


def _score_vector_or_none(values: torch.Tensor | None, point_count: int) -> torch.Tensor | None:
    if values is None:
        return None
    vector = values.detach().cpu().float()
    if vector.ndim == 2 and int(vector.shape[1]) == 1:
        vector = vector[:, 0]
    elif vector.ndim != 1:
        return None
    if int(vector.numel()) != int(point_count):
        return None
    return vector


def _score_component_vectors(
    score_component_vectors: dict[str, torch.Tensor] | None,
    point_count: int,
) -> dict[str, torch.Tensor]:
    if not score_component_vectors:
        return {}
    out: dict[str, torch.Tensor] = {}
    for name, values in sorted(score_component_vectors.items()):
        vector = _score_vector_or_none(values, point_count)
        if vector is not None:
            out[str(name)] = vector
    return out


def _trajectory_index_for_point(index: int, boundaries: list[tuple[int, int]]) -> int | None:
    point_index = int(index)
    for trajectory_idx, (start, end) in enumerate(boundaries):
        if int(start) <= point_index < int(end):
            return int(trajectory_idx)
    return None


def _prefixed_row_values(
    *,
    vectors: dict[str, torch.Tensor],
    prefix: str,
    index: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, vector in vectors.items():
        if not name.startswith(prefix):
            continue
        out[name.removeprefix(prefix)] = float(vector[int(index)].item())
    return out


def _row_score_component_values(
    *,
    vectors: dict[str, torch.Tensor],
    index: int,
) -> dict[str, float]:
    return {
        name: float(vector[int(index)].item())
        for name, vector in vectors.items()
        if not name.startswith("head_probability_") and not name.startswith("head_logit_")
    }


def _rank_rows_desc(
    rows: list[dict[str, Any]],
    *,
    output_name: str,
    value_getter: Any,
) -> None:
    valid: list[tuple[int, float]] = []
    for row_idx, row in enumerate(rows):
        value = value_getter(row)
        if value is None:
            continue
        valid.append((row_idx, float(value)))
    if not valid:
        return
    ordered = sorted(valid, key=lambda item: (-item[1], int(rows[item[0]]["point_index"])))
    count = len(ordered)
    for rank_idx, (row_idx, _value) in enumerate(ordered, start=1):
        rows[row_idx][f"{output_name}_candidate_rank"] = int(rank_idx)
        rows[row_idx][f"{output_name}_candidate_rank_fraction"] = float(rank_idx / count)


def _top_marginal_miss_bucket_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_counts: dict[str, int] = {}
    for row in rows:
        for bucket in row.get("failure_buckets", []):
            bucket_counts[str(bucket)] = bucket_counts.get(str(bucket), 0) + 1
    top_rows = sorted(
        rows,
        key=lambda row: (-float(row["marginal_query_local_utility"]), int(row["point_index"])),
    )[: min(16, len(rows))]
    return {
        "available": bool(rows),
        "diagnostic_only": True,
        "bucket_policy": {
            "top_rank_fraction_max": 0.25,
            "low_rank_fraction_min": 0.75,
            "prior_present_threshold": 0.05,
            "head_positive_threshold": 0.20,
            "heuristic_only": True,
        },
        "bucket_counts": bucket_counts,
        "top_marginal_row_count": len(top_rows),
        "top_marginal_rows": top_rows,
    }


def _attach_top_marginal_miss_diagnostics(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    _rank_rows_desc(
        rows,
        output_name="marginal_query_local_utility",
        value_getter=lambda row: row.get("marginal_query_local_utility"),
    )
    for score_name in ("raw_score", "selector_score", "segment_score"):
        _rank_rows_desc(
            rows,
            output_name=score_name,
            value_getter=lambda row, key=score_name: row.get(key),
        )

    component_names = sorted(
        {str(name) for row in rows for name in (row.get("score_components") or {})}
    )
    component_rank_by_row: list[dict[str, int]] = [{} for _row in rows]
    for component_name in component_names:
        valid = [
            (row_idx, float(row["score_components"][component_name]))
            for row_idx, row in enumerate(rows)
            if row.get("score_components") is not None
            and row["score_components"].get(component_name) is not None
        ]
        ordered = sorted(valid, key=lambda item: (-item[1], int(rows[item[0]]["point_index"])))
        for rank_idx, (row_idx, _value) in enumerate(ordered, start=1):
            component_rank_by_row[row_idx][component_name] = int(rank_idx)

    for row_idx, row in enumerate(rows):
        marginal_rank = row.get("marginal_query_local_utility_candidate_rank")
        if marginal_rank is not None and component_rank_by_row[row_idx]:
            row["query_local_utility_component_candidate_ranks"] = component_rank_by_row[row_idx]
            row["query_local_utility_component_minus_marginal_rank"] = {
                name: int(component_rank) - int(marginal_rank)
                for name, component_rank in component_rank_by_row[row_idx].items()
            }

    query_free_proxy_names = sorted(
        {str(name) for row in rows for name in (row.get("query_free_teacher_proxies") or {})}
    )
    query_free_proxy_rank_by_row: list[dict[str, int]] = [{} for _row in rows]
    for proxy_name in query_free_proxy_names:
        valid = [
            (row_idx, float(row["query_free_teacher_proxies"][proxy_name]))
            for row_idx, row in enumerate(rows)
            if row.get("query_free_teacher_proxies") is not None
            and row["query_free_teacher_proxies"].get(proxy_name) is not None
        ]
        ordered = sorted(valid, key=lambda item: (-item[1], int(rows[item[0]]["point_index"])))
        for rank_idx, (row_idx, _value) in enumerate(ordered, start=1):
            query_free_proxy_rank_by_row[row_idx][proxy_name] = int(rank_idx)

    for row_idx, row in enumerate(rows):
        marginal_rank = row.get("marginal_query_local_utility_candidate_rank")
        if marginal_rank is not None and query_free_proxy_rank_by_row[row_idx]:
            row["query_free_teacher_proxy_candidate_ranks"] = query_free_proxy_rank_by_row[row_idx]
            row["query_free_teacher_proxy_minus_marginal_rank"] = {
                name: int(proxy_rank) - int(marginal_rank)
                for name, proxy_rank in query_free_proxy_rank_by_row[row_idx].items()
            }

    for row_idx, row in enumerate(rows):
        buckets: list[str] = []
        marginal_fraction = row.get("marginal_query_local_utility_candidate_rank_fraction")
        raw_fraction = row.get("raw_score_candidate_rank_fraction")
        selector_fraction = row.get("selector_score_candidate_rank_fraction")
        segment_fraction = row.get("segment_score_candidate_rank_fraction")
        top_marginal = marginal_fraction is not None and float(marginal_fraction) <= 0.25
        low_any_score = any(
            value is not None and float(value) >= 0.75
            for value in (raw_fraction, selector_fraction, segment_fraction)
        )
        top_any_score = any(
            value is not None and float(value) <= 0.25
            for value in (raw_fraction, selector_fraction, segment_fraction)
        )
        low_marginal = marginal_fraction is not None and float(marginal_fraction) >= 0.75
        sampled_prior_values = list((row.get("sampled_prior_channels") or {}).values())
        model_prior_values = list((row.get("model_prior_channels") or {}).values())
        head_probability_values = list((row.get("head_probabilities") or {}).values())
        max_sampled_prior = max([float(value) for value in sampled_prior_values], default=0.0)
        max_model_prior = max([float(value) for value in model_prior_values], default=0.0)
        max_head_probability = max(
            [float(value) for value in head_probability_values],
            default=0.0,
        )
        if top_marginal and low_any_score:
            buckets.append("high_marginal_under_ranked_by_scores")
        if top_any_score and low_marginal:
            buckets.append("high_score_low_exact_marginal")
        if sampled_prior_values and max_sampled_prior <= 1e-6:
            buckets.append("prior_missing_or_out_of_support")
        if max_sampled_prior >= 0.05 and max_head_probability < 0.20:
            buckets.append("prior_present_but_head_flat")
        if (
            max_head_probability >= 0.20
            and raw_fraction is not None
            and float(raw_fraction) >= 0.75
        ):
            buckets.append("head_positive_but_final_score_suppresses_it")
        if (
            raw_fraction is not None
            and float(raw_fraction) <= 0.25
            and any(
                value is not None and float(value) >= 0.75
                for value in (selector_fraction, segment_fraction)
            )
        ):
            buckets.append("raw_score_good_but_segment_allocation_loses_it")
        stage_state = row.get("selector_stage_state") or {}
        if (
            row.get("source") in {"skeleton", "length_repair", "fallback"}
            or bool(stage_state.get("length_repair_retained", False))
            or (
                bool(stage_state.get("final_retained", False))
                and not bool(stage_state.get("pre_repair_retained", True))
            )
        ):
            buckets.append("length_repair_or_skeleton_overrides_learned_decision")
        if top_marginal and max_model_prior >= 0.05 and max_head_probability < 0.20:
            buckets.append("model_prior_present_but_head_flat")
        top_query_free_proxy = any(
            int(proxy_rank) <= max(1, math.ceil(0.25 * len(rows)))
            for proxy_rank in query_free_proxy_rank_by_row[row_idx].values()
        )
        if top_marginal and top_query_free_proxy and low_any_score:
            buckets.append("query_free_teacher_proxy_high_but_active_score_under_ranked")
        row["failure_buckets"] = sorted(set(buckets))


def _bounded_candidate_indices(
    *,
    mask: torch.Tensor,
    ranking_scores: torch.Tensor | None,
    limit: int,
) -> list[int]:
    indices = torch.where(mask.detach().cpu().bool())[0]
    if int(indices.numel()) <= 0:
        return []
    max_count = max(0, int(limit))
    if max_count <= 0:
        return []
    if int(indices.numel()) <= max_count:
        return [int(idx) for idx in indices.tolist()]
    if ranking_scores is None:
        positions = torch.linspace(0, int(indices.numel()) - 1, steps=max_count).round().long()
        return [int(idx) for idx in indices[positions].unique(sorted=True).tolist()]

    scores = ranking_scores[indices].float()
    high_count = max(1, max_count // 2)
    low_count = max(0, max_count - high_count)
    high_local = torch.topk(scores, k=min(high_count, int(scores.numel())), largest=True).indices
    if low_count > 0:
        low_local = torch.topk(scores, k=min(low_count, int(scores.numel())), largest=False).indices
        selected = torch.cat([indices[high_local], indices[low_local]])
    else:
        selected = indices[high_local]
    return [int(idx) for idx in selected.unique(sorted=True).tolist()]


def _query_local_utility_payload_for_mask(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    retained_mask: torch.Tensor,
    query_cache: ScoringQueryCache | None,
) -> tuple[float, dict[str, float]]:
    range_audit = score_range_audit(
        points=points,
        boundaries=boundaries,
        retained_mask=retained_mask,
        typed_queries=typed_queries,
        query_cache=query_cache,
    )
    geometric = compute_geometric_distortion(points, boundaries, retained_mask)
    length = compute_length_preservation(points, boundaries, retained_mask)
    endpoint_sanity_score = endpoint_sanity(retained_mask.detach().cpu().bool(), boundaries)
    query_local_utility = query_local_utility_from_range_audit(
        range_audit,
        length_preservation=length,
        avg_sed_km=float(geometric.get("avg_sed_km", 0.0)),
        endpoint_sanity=endpoint_sanity_score,
    )
    score = query_local_utility.get("query_local_utility_score", 0.0)
    components_payload = query_local_utility.get("query_local_utility_components", {})
    components = {
        str(name): float(value)
        for name, value in (
            components_payload.items() if isinstance(components_payload, dict) else []
        )
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return (
        float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0,
        components,
    )


def _query_local_utility_score_for_mask(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    retained_mask: torch.Tensor,
    query_cache: ScoringQueryCache | None,
) -> float:
    score, _components = _query_local_utility_payload_for_mask(
        points=points,
        boundaries=boundaries,
        typed_queries=typed_queries,
        retained_mask=retained_mask,
        query_cache=query_cache,
    )
    return float(score)


def _component_delta(
    left: dict[str, float],
    right: dict[str, float],
) -> dict[str, float]:
    keys = sorted(set(left) | set(right))
    return {key: float(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys}


def _target_component_vectors(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor | None, str | None]:
    try:
        bundle = build_query_local_utility_targets(
            points=points.detach().cpu().float(),
            boundaries=boundaries,
            typed_queries=typed_queries,
        )
    except Exception as exc:  # pragma: no cover - diagnostic must not break scoring.
        return {}, {}, None, str(exc)
    head_targets = bundle.head_targets.detach().cpu().float()
    head_mask = bundle.head_mask.detach().cpu().bool()
    target_vectors = {
        str(name): head_targets[:, head_idx].contiguous()
        for head_idx, name in enumerate(QUERY_LOCAL_UTILITY_HEAD_NAMES)
        if head_idx < int(head_targets.shape[1])
    }
    target_masks = {
        str(name): head_mask[:, head_idx].contiguous()
        for head_idx, name in enumerate(QUERY_LOCAL_UTILITY_HEAD_NAMES)
        if head_idx < int(head_mask.shape[1])
    }
    scalar_target = (
        bundle.labels[:, QUERY_TYPE_ID_RANGE].detach().cpu().float().contiguous()
        if int(bundle.labels.shape[1]) > QUERY_TYPE_ID_RANGE
        else None
    )
    return target_vectors, target_masks, scalar_target, None


def _query_family_records(
    points: torch.Tensor,
    typed_queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    points_cpu = points.detach().cpu().float()
    records: list[dict[str, Any]] = []
    for query_index, query in enumerate(typed_queries):
        if str(query.get("type", "")).lower() != "range":
            continue
        try:
            params = validated_range_query_params(query)
            mask = points_in_range_box(points_cpu, params).detach().cpu().bool().flatten()
        except ValueError:
            continue
        metadata = query.get("_metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        anchor_family = str(metadata_dict.get("anchor_family", "unspecified"))
        footprint_family = str(metadata_dict.get("footprint_family", "unspecified"))
        records.append(
            {
                "query_index": int(query_index),
                "mask": mask,
                "anchor_family": anchor_family,
                "footprint_family": footprint_family,
                "anchor_footprint_family": f"{anchor_family}|{footprint_family}",
            }
        )
    return records


def _trajectory_context_for_point(
    index: int, boundaries: list[tuple[int, int]]
) -> tuple[int | None, int, int]:
    for trajectory_index, (start, end) in enumerate(boundaries):
        if int(start) <= int(index) < int(end):
            return int(trajectory_index), int(start), int(end)
    return None, 0, 0


def _query_hit_run_id(
    *,
    index: int,
    mask: torch.Tensor,
    boundaries: list[tuple[int, int]],
    query_index: int,
) -> str | None:
    if int(index) < 0 or int(index) >= int(mask.numel()) or not bool(mask[int(index)].item()):
        return None
    trajectory_index, start, end = _trajectory_context_for_point(index, boundaries)
    if trajectory_index is None:
        return None
    left = int(index)
    while left > int(start) and bool(mask[left - 1].item()):
        left -= 1
    right = int(index) + 1
    while right < int(end) and bool(mask[right].item()):
        right += 1
    return f"q{int(query_index)}:traj{trajectory_index}:points{left}-{right}"


def _dominant_label(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0]


def _query_hit_context_for_point(
    *,
    index: int,
    records: list[dict[str, Any]],
    boundaries: list[tuple[int, int]],
) -> dict[str, Any]:
    anchor_counts: dict[str, int] = {}
    footprint_counts: dict[str, int] = {}
    pair_counts: dict[str, int] = {}
    query_indices: list[int] = []
    run_ids: list[str] = []
    for record in records:
        mask = record.get("mask")
        if not isinstance(mask, torch.Tensor):
            continue
        if int(index) < 0 or int(index) >= int(mask.numel()) or not bool(mask[int(index)].item()):
            continue
        query_index = int(record.get("query_index", -1))
        query_indices.append(query_index)
        anchor = str(record.get("anchor_family", "unspecified"))
        footprint = str(record.get("footprint_family", "unspecified"))
        pair = str(record.get("anchor_footprint_family", f"{anchor}|{footprint}"))
        anchor_counts[anchor] = int(anchor_counts.get(anchor, 0)) + 1
        footprint_counts[footprint] = int(footprint_counts.get(footprint, 0)) + 1
        pair_counts[pair] = int(pair_counts.get(pair, 0)) + 1
        run_id = _query_hit_run_id(
            index=index,
            mask=mask,
            boundaries=boundaries,
            query_index=query_index,
        )
        if run_id is not None:
            run_ids.append(run_id)
    return {
        "query_hit_count": len(query_indices),
        "range_query_count": len(records),
        "query_hit_fraction": (
            float(len(query_indices) / len(records)) if len(records) > 0 else None
        ),
        "anchor_family": _dominant_label(anchor_counts),
        "footprint_family": _dominant_label(footprint_counts),
        "anchor_footprint_family": _dominant_label(pair_counts),
        "anchor_family_counts": anchor_counts,
        "footprint_family_counts": footprint_counts,
        "anchor_footprint_family_counts": pair_counts,
        "query_indices": query_indices,
        "query_hit_run_ids": run_ids,
    }


def _row_target_values(
    vectors: dict[str, torch.Tensor],
    *,
    index: int,
) -> dict[str, float]:
    return {
        name: float(vector[int(index)].item())
        for name, vector in vectors.items()
        if int(index) < int(vector.numel())
    }


def _row_target_masks(
    vectors: dict[str, torch.Tensor],
    *,
    index: int,
) -> dict[str, bool]:
    return {
        name: bool(vector[int(index)].item())
        for name, vector in vectors.items()
        if int(index) < int(vector.numel())
    }


def retained_decision_marginal_query_local_utility_diagnostics(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    primary_retained_mask: torch.Tensor,
    raw_scores: torch.Tensor | None = None,
    selector_scores: torch.Tensor | None = None,
    segment_scores: torch.Tensor | None = None,
    score_component_vectors: dict[str, torch.Tensor] | None = None,
    query_free_teacher_proxy_vectors: dict[str, torch.Tensor] | None = None,
    sampled_prior_vectors: dict[str, torch.Tensor] | None = None,
    model_prior_vectors: dict[str, torch.Tensor] | None = None,
    source_masks: dict[str, torch.Tensor] | None = None,
    selector_trace: dict[str, Any] | None = None,
    query_cache: ScoringQueryCache | None = None,
    max_retained_per_source: int = 64,
    max_removed_candidates: int = 128,
    teacher_usage_split: str = "unknown",
) -> dict[str, Any]:
    """Score bounded point-level QueryLocalUtility marginals after masks are frozen.

    Retained candidates use leave-one-out loss. Removed candidates use add-one
    gain. The diagnostic is bounded and diagnostic-only; it is not a training
    target and must not affect retained-mask construction.
    """
    point_count = int(primary_retained_mask.numel())
    primary_mask = primary_retained_mask.detach().cpu().bool().flatten()
    if int(points.shape[0]) != point_count:
        raise ValueError(
            "primary_retained_mask must match points: "
            f"mask={point_count} points={int(points.shape[0])}"
        )
    if int(primary_mask.sum().item()) <= 0:
        return {
            "available": False,
            "diagnostic_only": True,
            "reason": "empty_primary_retained_mask",
        }

    source_mask_map: dict[str, torch.Tensor] = {}
    if source_masks is not None:
        source_mask_map.update(
            {
                str(name): mask.detach().cpu().bool().flatten()
                for name, mask in source_masks.items()
                if int(mask.numel()) == point_count
            }
        )
    if selector_trace is not None:
        source_mask_map.update(
            selector_trace_payloads.source_masks_from_selector_trace(
                selector_trace, point_count=point_count
            )
        )
    attributed = torch.zeros_like(primary_mask)
    for mask in source_mask_map.values():
        attributed |= mask & primary_mask
    unattributed = primary_mask & ~attributed
    if bool(unattributed.any().item()):
        source_mask_map["unattributed"] = unattributed
    if not source_mask_map:
        source_mask_map["retained"] = primary_mask

    raw_vector = _score_vector_or_none(raw_scores, point_count)
    selector_vector = _score_vector_or_none(selector_scores, point_count)
    segment_vector = _score_vector_or_none(segment_scores, point_count)
    component_vectors = _score_component_vectors(score_component_vectors, point_count)
    query_free_proxy_vectors = _score_component_vectors(
        query_free_teacher_proxy_vectors,
        point_count,
    )
    sampled_prior_component_vectors = _score_component_vectors(sampled_prior_vectors, point_count)
    model_prior_component_vectors = _score_component_vectors(model_prior_vectors, point_count)
    target_component_vectors, target_component_masks, scalar_target_vector, target_error = (
        _target_component_vectors(
            points=points,
            boundaries=boundaries,
            typed_queries=typed_queries,
        )
    )
    query_family_records = _query_family_records(points, typed_queries)
    trace_mask_state = selector_trace_payloads.trace_mask_state_from_selector_trace(
        selector_trace=selector_trace, point_count=point_count
    )
    selector_segment_context_rows = (
        selector_trace_payloads.selector_segment_context_rows_from_trace(
            selector_trace=selector_trace,
            point_count=point_count,
        )
    )
    ranking_vector = selector_vector if selector_vector is not None else raw_vector
    if ranking_vector is None:
        ranking_vector = segment_vector

    started_at = time.perf_counter()
    effective_query_cache = query_cache
    query_cache_created = False
    if effective_query_cache is None:
        effective_query_cache = ScoringQueryCache.for_workload(points, boundaries, typed_queries)
        query_cache_created = True

    primary_score, primary_components = _query_local_utility_payload_for_mask(
        points=points,
        boundaries=boundaries,
        typed_queries=typed_queries,
        retained_mask=primary_mask,
        query_cache=effective_query_cache,
    )

    rows: list[dict[str, Any]] = []

    def _row(
        *,
        index: int,
        source: str,
        decision: str,
        marginal: float,
        component_delta: dict[str, float],
        candidate_components: dict[str, float],
    ) -> dict[str, Any]:
        query_hit_context = _query_hit_context_for_point(
            index=int(index),
            records=query_family_records,
            boundaries=boundaries,
        )
        scalar_target = (
            None
            if scalar_target_vector is None or int(index) >= int(scalar_target_vector.numel())
            else float(scalar_target_vector[int(index)].item())
        )
        return {
            "point_index": int(index),
            "source": source,
            "decision": decision,
            "marginal_query_local_utility": float(marginal),
            "raw_score": None if raw_vector is None else float(raw_vector[int(index)].item()),
            "selector_score": (
                None if selector_vector is None else float(selector_vector[int(index)].item())
            ),
            "segment_score": (
                None if segment_vector is None else float(segment_vector[int(index)].item())
            ),
            "trajectory_index": _trajectory_index_for_point(int(index), boundaries),
            "selector_stage_state": {
                name: bool(mask[int(index)].item()) for name, mask in trace_mask_state.items()
            },
            "selector_segment_context": selector_trace_payloads.selector_segment_context_for_point(
                index=int(index),
                segment_rows=selector_segment_context_rows,
            ),
            "query_local_utility_target": scalar_target,
            "head_targets": _row_target_values(target_component_vectors, index=int(index)),
            "head_target_masks": _row_target_masks(target_component_masks, index=int(index)),
            "head_probabilities": _prefixed_row_values(
                vectors=component_vectors,
                prefix="head_probability_",
                index=int(index),
            ),
            "head_logits": _prefixed_row_values(
                vectors=component_vectors,
                prefix="head_logit_",
                index=int(index),
            ),
            "sampled_prior_channels": {
                name: float(vector[int(index)].item())
                for name, vector in sampled_prior_component_vectors.items()
            },
            "model_prior_channels": {
                name: float(vector[int(index)].item())
                for name, vector in model_prior_component_vectors.items()
            },
            "query_local_utility_score_components": _row_score_component_values(
                vectors=component_vectors,
                index=int(index),
            ),
            "primary_query_local_utility_components": dict(primary_components),
            "candidate_query_local_utility_components": dict(candidate_components),
            "query_local_utility_component_delta": dict(component_delta),
            "query_free_teacher_proxies": {
                name: float(vector[int(index)].item())
                for name, vector in query_free_proxy_vectors.items()
            },
            "query_family_hit_context": query_hit_context,
            "anchor_family": query_hit_context.get("anchor_family"),
            "footprint_family": query_hit_context.get("footprint_family"),
            "anchor_footprint_family": query_hit_context.get("anchor_footprint_family"),
            "query_hit_run_ids": list(query_hit_context.get("query_hit_run_ids") or []),
            "score_components": {
                name: float(vector[int(index)].item()) for name, vector in component_vectors.items()
            },
        }

    for source, source_mask in sorted(source_mask_map.items()):
        candidate_mask = source_mask & primary_mask
        for index in _bounded_candidate_indices(
            mask=candidate_mask,
            ranking_scores=ranking_vector,
            limit=max_retained_per_source,
        ):
            candidate = primary_mask.clone()
            candidate[int(index)] = False
            score, candidate_components = _query_local_utility_payload_for_mask(
                points=points,
                boundaries=boundaries,
                typed_queries=typed_queries,
                retained_mask=candidate,
                query_cache=effective_query_cache,
            )
            rows.append(
                _row(
                    index=index,
                    source=source,
                    decision="retained_removal_loss",
                    marginal=float(primary_score - score),
                    component_delta=_component_delta(primary_components, candidate_components),
                    candidate_components=candidate_components,
                )
            )

    removed_mask = ~primary_mask
    for index in _bounded_candidate_indices(
        mask=removed_mask,
        ranking_scores=ranking_vector,
        limit=max_removed_candidates,
    ):
        candidate = primary_mask.clone()
        candidate[int(index)] = True
        score, candidate_components = _query_local_utility_payload_for_mask(
            points=points,
            boundaries=boundaries,
            typed_queries=typed_queries,
            retained_mask=candidate,
            query_cache=effective_query_cache,
        )
        rows.append(
            _row(
                index=index,
                source="removed",
                decision="removed_addition_gain",
                marginal=float(score - primary_score),
                component_delta=_component_delta(candidate_components, primary_components),
                candidate_components=candidate_components,
            )
        )

    _attach_top_marginal_miss_diagnostics(rows)
    top_marginal_miss_diagnostics = _top_marginal_miss_bucket_summary(rows)
    top_marginal_miss_summary = dict(top_marginal_miss_diagnostics)
    top_marginal_miss_summary.pop("top_marginal_rows", None)
    top_marginal_miss_summary["top_marginal_rows_in_selector_trace_only"] = True

    by_source = {
        name: selector_marginal_alignment.score_alignment_summary(group_rows)
        for name, group_rows in sorted(
            selector_marginal_alignment.group_rows_by_field(rows, "source").items()
        )
    }
    by_decision = {
        name: selector_marginal_alignment.score_alignment_summary(group_rows)
        for name, group_rows in sorted(
            selector_marginal_alignment.group_rows_by_field(rows, "decision").items()
        )
    }
    guard_coupling_summary = (
        selector_marginal_alignment.query_free_teacher_proxy_guard_coupling_summary(rows)
    )
    learned_teacher_summary = (
        selector_marginal_alignment.learned_controllable_marginal_teacher_summary(rows)
    )
    separated_teacher_summary = selector_marginal_alignment.separated_marginal_teacher_targets(
        rows,
        teacher_usage_split=teacher_usage_split,
    )
    elapsed_seconds = float(time.perf_counter() - started_at)
    return {
        "available": True,
        "diagnostic_only": True,
        "exact_query_local_utility_marginals": True,
        "performance_mode": "exact_cached_query_support",
        "elapsed_seconds": elapsed_seconds,
        "query_cache_provided": query_cache is not None,
        "query_cache_created": query_cache_created,
        "query_cache_support_mask_count": int(
            len(effective_query_cache.support_masks) if effective_query_cache is not None else 0
        ),
        "query_cache_range_audit_support_count": int(
            len(effective_query_cache.range_audit_supports)
            if effective_query_cache is not None
            else 0
        ),
        "masks_frozen_before_query_scoring_required": True,
        "description": (
            "Bounded retained-decision marginal QueryLocalUtility diagnostic. "
            "Retained rows are leave-one-out loss; removed rows are add-one gain."
        ),
        "primary_query_local_utility": float(primary_score),
        "retained_count": int(primary_mask.sum().item()),
        "point_count": point_count,
        "max_retained_per_source": int(max_retained_per_source),
        "max_removed_candidates": int(max_removed_candidates),
        "score_fields_available": {
            "raw_score": raw_vector is not None,
            "selector_score": selector_vector is not None,
            "segment_score": segment_vector is not None,
        },
        "score_component_fields_available": dict.fromkeys(sorted(component_vectors.keys()), True),
        "context_fields_available": {
            "sampled_prior_channels": dict.fromkeys(
                sorted(sampled_prior_component_vectors.keys()), True
            ),
            "model_prior_channels": dict.fromkeys(
                sorted(model_prior_component_vectors.keys()), True
            ),
            "query_free_teacher_proxies": dict.fromkeys(
                sorted(query_free_proxy_vectors.keys()), True
            ),
            "selector_stage_state": dict.fromkeys(sorted(trace_mask_state.keys()), True),
            "selector_segment_context": bool(selector_segment_context_rows),
            "trajectory_index": True,
            "query_local_utility_target": scalar_target_vector is not None,
            "head_targets": dict.fromkeys(sorted(target_component_vectors.keys()), True),
            "head_target_masks": dict.fromkeys(sorted(target_component_masks.keys()), True),
            "target_diagnostic_error": target_error,
            "query_local_utility_component_delta": True,
            "primary_query_local_utility_components": True,
            "candidate_query_local_utility_components": True,
            "query_family_hit_context": bool(query_family_records),
            "query_hit_run_ids": bool(query_family_records),
        },
        "candidate_count": len(rows),
        "overall": selector_marginal_alignment.score_alignment_summary(rows),
        "by_source": by_source,
        "by_decision": by_decision,
        "query_free_teacher_proxy_guard_coupling_summary": guard_coupling_summary,
        "learned_controllable_marginal_teacher_summary": learned_teacher_summary,
        "separated_marginal_teacher_summary": separated_teacher_summary,
        "top_marginal_miss_summary": top_marginal_miss_summary,
        "top_marginal_miss_diagnostics": top_marginal_miss_diagnostics,
        "rows": sorted(
            rows,
            key=lambda row: (
                str(row["decision"]),
                str(row["source"]),
                -float(row["marginal_query_local_utility"]),
                int(row["point_index"]),
            ),
        ),
    }
