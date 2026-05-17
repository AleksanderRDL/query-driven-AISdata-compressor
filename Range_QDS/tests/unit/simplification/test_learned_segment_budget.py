from __future__ import annotations

import inspect

import torch

from evaluation.metrics import compute_length_preservation
from simplification.learned_segment_budget import (
    LEARNED_SEGMENT_BUDGET_SCHEMA_VERSION,
    LEARNED_SEGMENT_BUDGET_TRACE_SCHEMA_VERSION,
    learned_segment_budget_diagnostics,
    simplify_with_learned_segment_budget_v1,
    simplify_with_learned_segment_budget_v1_with_trace,
)


def test_learned_segment_budget_public_api_and_trace_accounting() -> None:
    scores = torch.linspace(0.0, 1.0, steps=24, dtype=torch.float32)
    boundaries = [(0, 12), (12, 24)]

    retained, trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        boundaries,
        compression_ratio=0.30,
        segment_size=4,
    )
    without_trace = simplify_with_learned_segment_budget_v1(
        scores,
        boundaries,
        compression_ratio=0.30,
        segment_size=4,
    )
    diagnostics = learned_segment_budget_diagnostics(boundaries, (0.10, 0.30))
    source_count = (
        int(trace["skeleton_retained_count"])
        + int(trace["learned_controlled_retained_slots"])
        + int(trace["fallback_retained_count"])
        + int(trace["length_repair_retained_count"])
    )

    assert torch.equal(retained, without_trace)
    assert trace["schema_version"] == LEARNED_SEGMENT_BUDGET_TRACE_SCHEMA_VERSION
    assert trace["selector_type"] == "learned_segment_budget_v1"
    assert source_count == int(retained.sum().item())
    assert diagnostics["schema_version"] == LEARNED_SEGMENT_BUDGET_SCHEMA_VERSION
    assert diagnostics["budget_rows"][0]["no_fixed_85_percent_temporal_scaffold"] is True


def test_learned_segment_budget_public_api_uses_trajectory_budget_keyword() -> None:
    signature = inspect.signature(simplify_with_learned_segment_budget_v1)

    assert "max_budget_share_per_trajectory" in signature.parameters
    assert "max_budget_share_per_ship" not in signature.parameters


def test_learned_segment_budget_length_repair_preserves_budget_and_reports_attribution() -> None:
    steps = torch.arange(0, 32, dtype=torch.float32)
    points = torch.stack(
        [
            steps,
            torch.where((steps.long() % 2) == 0, torch.zeros_like(steps), torch.ones_like(steps)),
            steps * 0.10,
        ],
        dim=1,
    )
    scores = torch.zeros((32,), dtype=torch.float32)
    scores[14:19] = torch.tensor([5.0, 6.0, 7.0, 6.0, 5.0])
    boundaries = [(0, 32)]

    retained, trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        boundaries,
        compression_ratio=0.20,
        points=points,
        geometry_gain_weight=0.0,
        length_repair_fraction=0.0,
    )
    repaired, repaired_trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        boundaries,
        compression_ratio=0.20,
        points=points,
        geometry_gain_weight=0.0,
        length_repair_fraction=1.0,
    )

    assert int(repaired.sum().item()) == int(retained.sum().item())
    assert repaired_trace["length_repair_swap_count"] > 0
    assert repaired_trace["length_repair_retained_count"] > 0
    assert (
        repaired_trace["learned_controlled_retained_slots"]
        < trace["learned_controlled_retained_slots"]
    )
    assert compute_length_preservation(points, boundaries, repaired) > compute_length_preservation(
        points,
        boundaries,
        retained,
    )


def test_learned_segment_budget_reports_query_free_segment_source_attribution() -> None:
    scores = torch.linspace(0.0, 1.0, steps=24, dtype=torch.float32)
    segment_scores = torch.zeros((24,), dtype=torch.float32)
    segment_scores[8:12] = 5.0
    segment_scores[20:24] = 4.0

    retained, trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        [(0, 12), (12, 24)],
        compression_ratio=0.30,
        segment_size=4,
        segment_scores=segment_scores,
    )

    attribution = trace["segment_source_attribution"]
    summary = attribution["summary"]
    first_row = attribution["rows"][0]
    assert attribution["available"] is True
    assert attribution["diagnostic_only"] is True
    assert attribution["query_free"] is True
    assert summary["retained_count_total"] == int(retained.sum().item())
    assert summary["skeleton_count_total"] == trace["skeleton_retained_count"]
    assert summary["learned_count_total"] == trace["learned_controlled_retained_slots"]
    assert summary["segment_allocation_count_total"] == trace["segment_budget_allocation_count"]
    assert {
        "segment_index",
        "allocation_order_index",
        "trajectory_id",
        "segment_score",
        "segment_score_rank",
        "segment_allocation_count",
        "retained_count",
        "skeleton_count",
        "learned_count",
        "fallback_count",
        "length_repair_count",
        "unattributed_count",
    }.issubset(first_row)
