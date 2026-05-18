"""Segment-level audit helpers for query-driven experiment diagnostics."""

from __future__ import annotations

import math
from typing import Any

import torch

from learning.targets.query_useful_v1 import (
    QUERY_USEFUL_V1_HEAD_NAMES,
    build_query_useful_v1_targets,
)
from selection.model_score_conversion import workload_type_head


def _average_ranks(values: list[float]) -> list[float]:
    """Return average ranks for deterministic Spearman diagnostics."""
    if not values:
        return []
    ordered = sorted(enumerate(float(value) for value in values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = 0.5 * float(cursor + end - 1)
        for idx, _value in ordered[cursor:end]:
            ranks[idx] = average_rank
        cursor = end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Return Pearson correlation for diagnostic lists."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(float(x) for x in xs) / float(len(xs))
    y_mean = sum(float(y) for y in ys) / float(len(ys))
    cov = 0.0
    x_var = 0.0
    y_var = 0.0
    for x_raw, y_raw in zip(xs, ys, strict=True):
        x = float(x_raw) - x_mean
        y = float(y_raw) - y_mean
        cov += x * y
        x_var += x * x
        y_var += y * y
    denom = math.sqrt(x_var * y_var)
    if denom <= 1e-12:
        return None
    return float(cov / denom)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Return tie-aware Spearman correlation for diagnostic lists."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def segment_top_mean(values: torch.Tensor, start: int, end: int) -> float:
    """Return top-20% mean for one segment, matching selector segment aggregation."""
    local = values[int(start) : int(end)].detach().cpu().float()
    if int(local.numel()) <= 0:
        return 0.0
    top_count = min(int(local.numel()), max(1, math.ceil(0.20 * int(local.numel()))))
    return float(torch.topk(local, k=top_count).values.mean().item())


def factorized_head_probability_sources_from_logits(
    head_logits: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Return diagnostic-only per-head probability score sources from frozen logits."""
    if head_logits is None:
        return {}
    logits = head_logits.detach().cpu().float()
    if logits.ndim != 2:
        return {}
    point_count = int(logits.shape[0])
    head_count = min(int(logits.shape[1]), len(QUERY_USEFUL_V1_HEAD_NAMES))
    if point_count <= 0 or head_count <= 0:
        return {}
    probabilities = torch.sigmoid(logits[:, :head_count])
    return {
        f"head_{head_name!s}_sigmoid_top20_mean": probabilities[:, head_idx].contiguous()
        for head_idx, head_name in enumerate(QUERY_USEFUL_V1_HEAD_NAMES[:head_count])
    }


def _segment_oracle_alignment_for_scores(
    *,
    score_by_segment: list[float],
    oracle_mass_by_segment: list[float],
    top_fractions: tuple[float, ...] = (0.10, 0.25, 0.50),
) -> dict[str, Any]:
    """Return ranking alignment between one segment score and eval-only oracle mass."""
    if len(score_by_segment) != len(oracle_mass_by_segment) or not score_by_segment:
        return {"available": False, "reason": "segment_score_or_oracle_missing"}
    total_oracle_mass = sum(max(0.0, float(value)) for value in oracle_mass_by_segment)
    order = sorted(
        range(len(score_by_segment)),
        key=lambda idx: (float(score_by_segment[idx]), -idx),
        reverse=True,
    )
    rows: list[dict[str, float | int]] = []
    for fraction in top_fractions:
        top_count = max(1, min(len(order), math.ceil(float(fraction) * len(order))))
        selected = order[:top_count]
        selected_oracle_mass = sum(max(0.0, float(oracle_mass_by_segment[idx])) for idx in selected)
        rows.append(
            {
                "top_fraction": float(fraction),
                "segment_count": int(top_count),
                "oracle_mass_recall": (
                    0.0
                    if total_oracle_mass <= 1e-12
                    else float(selected_oracle_mass / total_oracle_mass)
                ),
                "mean_oracle_mass": float(selected_oracle_mass / float(top_count)),
            }
        )
    return {
        "available": True,
        "spearman_vs_oracle_mass": _spearman(score_by_segment, oracle_mass_by_segment),
        "pearson_vs_oracle_mass": _pearson(score_by_segment, oracle_mass_by_segment),
        "top_fraction_rows": rows,
    }


def _descending_rank_and_order(values: list[float]) -> tuple[list[int], list[int]]:
    """Return 1-based descending ranks and index order for compact diagnostics."""
    order = sorted(range(len(values)), key=lambda idx: (float(values[idx]), -idx), reverse=True)
    ranks = [0 for _ in values]
    for rank, idx in enumerate(order, start=1):
        ranks[int(idx)] = int(rank)
    return ranks, order


def _segment_transfer_rows(
    *,
    segment_rows: list[dict[str, Any]],
    source_segment_scores: dict[str, list[float]],
    oracle_mass_by_segment: list[float],
    retained_count_by_segment: list[int] | None,
    top_n: int = 16,
) -> dict[str, Any]:
    """Return paired segment rows for localizing score-to-allocation transfer failures."""
    segment_count = len(segment_rows)
    if segment_count <= 0 or len(oracle_mass_by_segment) != segment_count:
        return {"available": False, "reason": "segment_rows_or_oracle_missing"}
    source_names = [
        name for name, values in source_segment_scores.items() if len(values) == segment_count
    ]
    if not source_names:
        return {"available": False, "reason": "source_scores_missing"}

    row_limit = max(1, min(int(top_n), int(segment_count)))
    oracle_ranks, oracle_order = _descending_rank_and_order(oracle_mass_by_segment)
    source_ranks: dict[str, list[int]] = {}
    selected_indices: set[int] = set(oracle_order[:row_limit])
    selected_reasons: dict[int, set[str]] = {
        int(idx): {"oracle_mass_top"} for idx in oracle_order[:row_limit]
    }
    for name in source_names:
        ranks, order = _descending_rank_and_order(source_segment_scores[name])
        source_ranks[name] = ranks
        for idx in order[:row_limit]:
            selected_indices.add(int(idx))
            selected_reasons.setdefault(int(idx), set()).add(f"{name}_top")

    retained_ranks: list[int] | None = None
    retained_order: list[int] | None = None
    if retained_count_by_segment is not None and len(retained_count_by_segment) == segment_count:
        retained_values = [float(value) for value in retained_count_by_segment]
        retained_ranks, retained_order = _descending_rank_and_order(retained_values)
        for idx in retained_order[:row_limit]:
            if retained_values[int(idx)] <= 0.0:
                continue
            selected_indices.add(int(idx))
            selected_reasons.setdefault(int(idx), set()).add("retained_count_top")

    def sort_key(idx: int) -> tuple[int, int, int]:
        best_source_rank = min(source_ranks[name][idx] for name in source_names)
        return (int(oracle_ranks[idx]), int(best_source_rank), int(idx))

    rows: list[dict[str, Any]] = []
    for idx in sorted(selected_indices, key=sort_key):
        base = segment_rows[int(idx)]
        row: dict[str, Any] = {
            "segment_index": int(idx),
            "trajectory_id": int(base["trajectory_id"]),
            "start": int(base["start"]),
            "end": int(base["end"]),
            "length": int(base["length"]),
            "oracle_mass": float(oracle_mass_by_segment[int(idx)]),
            "oracle_mass_rank": int(oracle_ranks[int(idx)]),
            "oracle_top20_mean": float(base.get("oracle_top20_mean", 0.0)),
            "selection_reasons": sorted(selected_reasons.get(int(idx), set())),
        }
        for name in source_names:
            row[f"{name}_score"] = float(source_segment_scores[name][int(idx)])
            row[f"{name}_rank"] = int(source_ranks[name][int(idx)])
        if (
            retained_count_by_segment is not None
            and len(retained_count_by_segment) == segment_count
        ):
            retained_count = int(retained_count_by_segment[int(idx)])
            row["frozen_primary_retained_count"] = retained_count
            row["frozen_primary_retained_fraction"] = float(
                retained_count / max(1, int(base["length"]))
            )
            if retained_ranks is not None:
                row["frozen_primary_retained_count_rank"] = int(retained_ranks[int(idx)])
        rows.append(row)

    result: dict[str, Any] = {
        "available": True,
        "row_selection": "union_of_top_oracle_top_source_and_top_retained_segments",
        "row_limit_per_source": int(row_limit),
        "row_count": len(rows),
        "rows": rows,
    }
    if retained_count_by_segment is not None and len(retained_count_by_segment) == segment_count:
        retained_float = [float(value) for value in retained_count_by_segment]
        total_oracle_mass = sum(max(0.0, float(value)) for value in oracle_mass_by_segment)
        retained_oracle_mass = sum(
            max(0.0, float(oracle_mass_by_segment[idx]))
            for idx, count in enumerate(retained_count_by_segment)
            if int(count) > 0
        )
        result["retained_segment_summary"] = {
            "available": True,
            "frozen_primary_retained_count_total": int(sum(retained_count_by_segment)),
            "segments_with_any_frozen_primary_retained_point": int(
                sum(1 for count in retained_count_by_segment if int(count) > 0)
            ),
            "retained_count_spearman_vs_oracle_mass": _spearman(
                retained_float, oracle_mass_by_segment
            ),
            "retained_count_pearson_vs_oracle_mass": _pearson(
                retained_float, oracle_mass_by_segment
            ),
            "oracle_mass_recall_in_segments_with_any_retained_point": (
                0.0
                if total_oracle_mass <= 1e-12
                else float(retained_oracle_mass / total_oracle_mass)
            ),
        }
    else:
        result["retained_segment_summary"] = {"available": False, "reason": "retained_mask_missing"}
    return result


def _all_segment_transfer_rows(
    *,
    segment_rows: list[dict[str, Any]],
    source_segment_scores: dict[str, list[float]],
    oracle_mass_by_segment: list[float],
    retained_count_by_segment: list[int] | None,
) -> dict[str, Any]:
    """Return eval-labeled rows for every segment after masks have been frozen."""
    segment_count = len(segment_rows)
    if segment_count <= 0 or len(oracle_mass_by_segment) != segment_count:
        return {"available": False, "reason": "segment_rows_or_oracle_missing"}
    source_names = [
        name for name, values in source_segment_scores.items() if len(values) == segment_count
    ]
    if not source_names:
        return {"available": False, "reason": "source_scores_missing"}

    oracle_ranks, _oracle_order = _descending_rank_and_order(oracle_mass_by_segment)
    source_ranks: dict[str, list[int]] = {
        name: _descending_rank_and_order(source_segment_scores[name])[0] for name in source_names
    }
    retained_ranks: list[int] | None = None
    if retained_count_by_segment is not None and len(retained_count_by_segment) == segment_count:
        retained_ranks = _descending_rank_and_order(
            [float(value) for value in retained_count_by_segment]
        )[0]

    rows: list[dict[str, Any]] = []
    for idx, base in enumerate(segment_rows):
        row: dict[str, Any] = {
            "segment_index": int(idx),
            "trajectory_id": int(base["trajectory_id"]),
            "start": int(base["start"]),
            "end": int(base["end"]),
            "length": int(base["length"]),
            "oracle_mass": float(oracle_mass_by_segment[int(idx)]),
            "oracle_mass_rank": int(oracle_ranks[int(idx)]),
            "oracle_positive": bool(float(oracle_mass_by_segment[int(idx)]) > 0.0),
            "oracle_top20_mean": float(base.get("oracle_top20_mean", 0.0)),
            "canonical_order_rank": int(idx + 1),
            "neutral_allocation_score": 0.0,
            "neutral_allocation_order_rank": int(idx + 1),
        }
        for name in source_names:
            row[f"{name}_score"] = float(source_segment_scores[name][int(idx)])
            row[f"{name}_rank"] = int(source_ranks[name][int(idx)])
        if (
            retained_count_by_segment is not None
            and len(retained_count_by_segment) == segment_count
        ):
            retained_count = int(retained_count_by_segment[int(idx)])
            row["frozen_primary_retained_count"] = retained_count
            row["frozen_primary_retained_fraction"] = float(
                retained_count / max(1, int(base["length"]))
            )
            if retained_ranks is not None:
                row["frozen_primary_retained_count_rank"] = int(retained_ranks[int(idx)])
        rows.append(row)

    return {
        "available": True,
        "diagnostic_only": True,
        "uses_eval_labels_after_mask_freeze": True,
        "row_scope": "all_segments",
        "row_count": len(rows),
        "rows": rows,
    }


def segment_oracle_allocation_audit(
    *,
    point_scores: torch.Tensor | None,
    segment_budget_scores: torch.Tensor | None,
    selector_segment_scores: torch.Tensor | None,
    eval_labels: torch.Tensor | None,
    boundaries: list[tuple[int, int]],
    workload_type: str,
    head_scores_by_name: dict[str, torch.Tensor] | None = None,
    retained_mask: torch.Tensor | None = None,
    segment_size: int = 32,
    paired_row_limit: int = 16,
) -> dict[str, Any]:
    """Compare allocation score rankings with eval-only segment oracle mass.

    This diagnostic must run only after workload-blind masks have been frozen.
    It uses eval labels for audit only and is not an acceptance shortcut.
    """
    if eval_labels is None:
        return {"available": False, "reason": "eval_labels_not_available"}
    if point_scores is None:
        return {"available": False, "reason": "point_scores_not_available"}
    point_count = int(point_scores.numel())
    labels = eval_labels.detach().cpu().float()
    if labels.ndim != 2 or labels.shape[0] != point_count:
        return {"available": False, "reason": "label_score_shape_mismatch"}
    _workload_name, type_id = workload_type_head(workload_type)
    if int(labels.shape[1]) <= int(type_id):
        return {"available": False, "reason": "workload_type_label_missing"}
    retained_values: torch.Tensor | None = None
    if retained_mask is not None:
        retained_candidate = retained_mask.detach().cpu().bool()
        if int(retained_candidate.numel()) == point_count:
            retained_values = retained_candidate

    score_sources: dict[str, torch.Tensor] = {
        "point_score_top20_mean": point_scores.detach().cpu().float()
    }
    if segment_budget_scores is not None and int(segment_budget_scores.numel()) == point_count:
        score_sources["segment_budget_head_top20_mean"] = (
            segment_budget_scores.detach().cpu().float()
        )
    if selector_segment_scores is not None and int(selector_segment_scores.numel()) == point_count:
        score_sources["selector_allocation_score_top20_mean"] = (
            selector_segment_scores.detach().cpu().float()
        )
    if isinstance(head_scores_by_name, dict):
        for raw_name, raw_values in head_scores_by_name.items():
            if not isinstance(raw_name, str) or raw_name in score_sources:
                continue
            if not isinstance(raw_values, torch.Tensor):
                continue
            values = raw_values.detach().cpu().float()
            if values.ndim == 2 and int(values.shape[1]) == 1:
                values = values[:, 0]
            if values.ndim != 1 or int(values.numel()) != point_count:
                continue
            score_sources[raw_name] = values

    oracle_values = labels[:, int(type_id)].float()
    segment_rows: list[dict[str, Any]] = []
    source_segment_scores: dict[str, list[float]] = {name: [] for name in score_sources}
    oracle_mass_by_segment: list[float] = []
    oracle_top_mean_by_segment: list[float] = []
    retained_count_by_segment: list[int] | None = [] if retained_values is not None else None
    size = max(1, int(segment_size))
    for trajectory_id, (start, end) in enumerate(boundaries):
        for seg_start in range(int(start), int(end), size):
            seg_end = min(int(end), int(seg_start) + size)
            if seg_end <= seg_start:
                continue
            oracle_segment = oracle_values[seg_start:seg_end]
            oracle_mass = float(torch.clamp(oracle_segment, min=0.0).sum().item())
            oracle_top_mean = segment_top_mean(oracle_values, seg_start, seg_end)
            oracle_mass_by_segment.append(oracle_mass)
            oracle_top_mean_by_segment.append(oracle_top_mean)
            if retained_count_by_segment is not None and retained_values is not None:
                retained_count_by_segment.append(
                    int(retained_values[seg_start:seg_end].sum().item())
                )
            for name, values in score_sources.items():
                source_segment_scores[name].append(segment_top_mean(values, seg_start, seg_end))
            segment_rows.append(
                {
                    "trajectory_id": int(trajectory_id),
                    "start": int(seg_start),
                    "end": int(seg_end),
                    "length": int(seg_end - seg_start),
                    "oracle_mass": oracle_mass,
                    "oracle_top20_mean": oracle_top_mean,
                }
            )

    source_alignment = {
        name: _segment_oracle_alignment_for_scores(
            score_by_segment=scores,
            oracle_mass_by_segment=oracle_mass_by_segment,
        )
        for name, scores in source_segment_scores.items()
    }
    best_source = None
    best_recall = -float("inf")
    for name, alignment in source_alignment.items():
        rows = alignment.get("top_fraction_rows") if isinstance(alignment, dict) else None
        if not isinstance(rows, list):
            continue
        top25 = next(
            (row for row in rows if abs(float(row.get("top_fraction", 0.0)) - 0.25) <= 1e-9), None
        )
        if not isinstance(top25, dict):
            continue
        recall = float(top25.get("oracle_mass_recall", 0.0))
        if recall > best_recall:
            best_recall = recall
            best_source = name

    return {
        "available": True,
        "diagnostic_only": True,
        "uses_eval_labels_after_mask_freeze": True,
        "description": "Segment score ranking alignment against eval-only oracle label mass.",
        "workload_type": str(workload_type),
        "segment_size": int(size),
        "segment_count": len(segment_rows),
        "oracle_mass_total": float(sum(oracle_mass_by_segment)),
        "oracle_positive_segment_count": int(sum(value > 0.0 for value in oracle_mass_by_segment)),
        "score_source_names": list(score_sources.keys()),
        "source_alignment": source_alignment,
        "paired_segment_transfer_rows": _segment_transfer_rows(
            segment_rows=segment_rows,
            source_segment_scores=source_segment_scores,
            oracle_mass_by_segment=oracle_mass_by_segment,
            retained_count_by_segment=retained_count_by_segment,
            top_n=int(paired_row_limit),
        ),
        "all_segment_transfer_rows": _all_segment_transfer_rows(
            segment_rows=segment_rows,
            source_segment_scores=source_segment_scores,
            oracle_mass_by_segment=oracle_mass_by_segment,
            retained_count_by_segment=retained_count_by_segment,
        ),
        "best_source_by_top25_oracle_mass_recall": best_source,
    }


def target_segment_oracle_alignment_audit(
    *,
    points: torch.Tensor,
    boundaries: list[tuple[int, int]],
    typed_queries: list[dict[str, Any]],
    eval_labels: torch.Tensor | None,
    workload_type: str,
    retained_mask: torch.Tensor | None = None,
    segment_size: int = 32,
    paired_row_limit: int = 16,
) -> dict[str, Any]:
    """Compare eval QueryUsefulV1 target heads with eval-only oracle segment mass after mask freeze."""
    if eval_labels is None:
        return {"available": False, "reason": "eval_labels_not_available"}
    if not typed_queries:
        return {"available": False, "reason": "typed_queries_not_available"}
    _workload_name, type_id = workload_type_head(workload_type)
    targets = build_query_useful_v1_targets(
        points=points,
        boundaries=boundaries,
        typed_queries=typed_queries,
        segment_size=segment_size,
    )
    final_target = targets.labels[:, int(type_id)].detach().cpu().float()
    head_targets = targets.head_targets.detach().cpu().float()
    head_score_sources = {
        f"target_head_{head_name!s}_top20_mean": head_targets[:, head_idx]
        for head_idx, head_name in enumerate(QUERY_USEFUL_V1_HEAD_NAMES)
        if int(head_targets.shape[1]) > head_idx
    }
    audit = segment_oracle_allocation_audit(
        point_scores=final_target,
        segment_budget_scores=None,
        selector_segment_scores=None,
        eval_labels=eval_labels,
        boundaries=boundaries,
        workload_type=workload_type,
        head_scores_by_name=head_score_sources,
        retained_mask=retained_mask,
        segment_size=segment_size,
        paired_row_limit=paired_row_limit,
    )
    if not bool(audit.get("available", False)):
        audit["target_alignment_attempted"] = True
        return audit

    source_semantics = {
        "point_score_top20_mean": "eval_query_useful_v1_final_target_top20_mean",
    }
    source_semantics.update(
        {
            f"target_head_{head_name!s}_top20_mean": (
                f"eval_query_useful_v1_factorized_target_head:{head_name!s}"
            )
            for head_name in QUERY_USEFUL_V1_HEAD_NAMES
        }
    )
    target_diagnostics = targets.diagnostics
    audit.update(
        {
            "description": (
                "Eval QueryUsefulV1 target-head segment alignment against eval-only oracle label mass."
            ),
            "target_alignment_attempted": True,
            "target_family": target_diagnostics.get("target_family"),
            "target_range_query_count": target_diagnostics.get("range_query_count"),
            "source_semantics": source_semantics,
            "target_diagnostics_summary": {
                "final_label_positive_fraction": target_diagnostics.get(
                    "final_label_positive_fraction"
                ),
                "final_label_mass": target_diagnostics.get("final_label_mass"),
                "segment_budget_target_base_source": target_diagnostics.get(
                    "segment_budget_target_base_source"
                ),
                "final_label_formula": target_diagnostics.get("final_label_formula"),
            },
        }
    )
    return audit
