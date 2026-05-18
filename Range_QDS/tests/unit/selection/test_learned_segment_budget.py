from __future__ import annotations

# pyright: reportPrivateImportUsage=false
import inspect
import math

import pytest
import torch

from scoring.metrics import compute_length_preservation
from selection.learned_segment_budget import (
    LEARNED_SEGMENT_BUDGET_SCHEMA_VERSION,
    LEARNED_SEGMENT_BUDGET_TRACE_SCHEMA_VERSION,
    learned_segment_budget_diagnostics,
    simplify_with_learned_segment_budget_v1,
    simplify_with_learned_segment_budget_v1_with_trace,
)
from selection.learned_segment_budget.allocation import _allocate_segment_budgets
from selection.learned_segment_budget.diagnostics import (
    _segment_allocation_alignment_diagnostics,
)
from selection.learned_segment_budget.length_repair import _apply_length_repair_swaps


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
    assert repaired_trace["length_repair_score_protection_fraction"] == 0.0
    assert repaired_trace["length_repair_score_protected_count"] == 0
    assert (
        repaired_trace["learned_controlled_retained_slots"]
        < trace["learned_controlled_retained_slots"]
    )
    assert compute_length_preservation(points, boundaries, repaired) > compute_length_preservation(
        points,
        boundaries,
        retained,
    )


def test_learned_segment_budget_length_repair_can_protect_top_scores() -> None:
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

    retained, _trace = simplify_with_learned_segment_budget_v1_with_trace(
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
        length_repair_score_protection_fraction=1.0,
    )

    assert torch.equal(repaired, retained)
    assert repaired_trace["length_repair_swap_count"] == 0
    assert repaired_trace["length_repair_score_protection_fraction"] == 1.0
    assert repaired_trace["length_repair_score_protected_count"] > 0


def test_learned_segment_budget_length_repair_spends_budget_on_global_net_gain() -> None:
    points_per_trajectory = 60
    steps = torch.arange(0, points_per_trajectory, dtype=torch.float32)
    zigzag_lat = torch.where(
        (steps.long() % 2) == 0,
        torch.zeros_like(steps),
        torch.full_like(steps, 2.0),
    )
    zigzag_points = torch.stack([steps, zigzag_lat, steps * 0.05], dim=1)
    flat_points = torch.stack([steps, torch.zeros_like(steps), torch.zeros_like(steps)], dim=1)
    points = torch.cat([zigzag_points, flat_points], dim=0)
    scores = torch.linspace(0.0, 1.0, steps=points_per_trajectory * 2, dtype=torch.float32)
    retained = torch.zeros((points_per_trajectory * 2,), dtype=torch.bool)
    learned = torch.zeros_like(retained)
    fallback = torch.zeros_like(retained)
    length_repair = torch.zeros_like(retained)
    boundaries = [(0, points_per_trajectory), (points_per_trajectory, points_per_trajectory * 2)]

    endpoint_indices = [
        0,
        points_per_trajectory - 1,
        points_per_trajectory,
        points_per_trajectory * 2 - 1,
    ]
    retained[endpoint_indices] = True
    local_learned_indices = list(range(2, points_per_trajectory - 2, 2))
    retained[local_learned_indices] = True
    learned[local_learned_indices] = True
    flat_learned_indices = [points_per_trajectory + idx for idx in local_learned_indices]
    retained[flat_learned_indices] = True
    learned[flat_learned_indices] = True
    repair_fraction = 0.25
    old_per_trajectory_cap = math.ceil(repair_fraction * len(local_learned_indices))

    swap_count = _apply_length_repair_swaps(
        scores=scores,
        points=points,
        boundaries=boundaries,
        retained=retained,
        learned_mask=learned,
        fallback_mask=fallback,
        length_repair_mask=length_repair,
        repair_fraction=repair_fraction,
    )

    assert swap_count > old_per_trajectory_cap
    assert int(length_repair[:points_per_trajectory].sum().item()) == swap_count
    assert int(length_repair[points_per_trajectory:].sum().item()) == 0
    assert int(retained.sum().item()) == 2 * (len(local_learned_indices) + 2)


def test_learned_segment_budget_allocation_can_use_query_free_length_support() -> None:
    retained = torch.zeros((64,), dtype=torch.bool)
    retained[[0, 63]] = True
    segment_rows = [
        {
            "trajectory_id": 0,
            "start": 0,
            "end": 32,
            "length": 32,
            "score": 1.0,
            "length_support_score": 0.0,
        },
        {
            "trajectory_id": 0,
            "start": 32,
            "end": 64,
            "length": 32,
            "score": 0.0,
            "length_support_score": 10.0,
        },
    ]

    score_only = _allocate_segment_budgets(
        segment_rows=segment_rows,
        retained=retained,
        remaining=4,
        budget=6,
        boundaries=[(0, 64)],
        max_budget_share_per_trajectory=1.0,
        fairness_preallocation_enabled=False,
        segment_length_support_weight=0.0,
    )
    length_supported = _allocate_segment_budgets(
        segment_rows=segment_rows,
        retained=retained,
        remaining=4,
        budget=6,
        boundaries=[(0, 64)],
        max_budget_share_per_trajectory=1.0,
        fairness_preallocation_enabled=False,
        segment_length_support_weight=1.0,
    )

    assert score_only.get(0, 0) > score_only.get(1, 0)
    assert length_supported.get(1, 0) > length_supported.get(0, 0)


def test_learned_segment_budget_allocation_uses_length_support_when_scores_are_flat() -> None:
    retained = torch.zeros((64,), dtype=torch.bool)
    retained[[0, 63]] = True
    segment_rows = [
        {
            "trajectory_id": 0,
            "start": 0,
            "end": 32,
            "length": 32,
            "score": 0.0,
            "length_support_score": 0.0,
        },
        {
            "trajectory_id": 0,
            "start": 32,
            "end": 64,
            "length": 32,
            "score": 0.0,
            "length_support_score": 10.0,
        },
    ]

    length_supported = _allocate_segment_budgets(
        segment_rows=segment_rows,
        retained=retained,
        remaining=4,
        budget=6,
        boundaries=[(0, 64)],
        max_budget_share_per_trajectory=1.0,
        fairness_preallocation_enabled=False,
        segment_length_support_weight=1.0,
    )

    assert length_supported.get(1, 0) > length_supported.get(0, 0)
    assert segment_rows[1]["allocation_weight"] > segment_rows[0]["allocation_weight"]


def test_learned_segment_budget_fairness_preallocation_uses_allocation_weight() -> None:
    retained = torch.zeros((64,), dtype=torch.bool)
    retained[[0, 63]] = True
    segment_rows = [
        {
            "trajectory_id": 0,
            "start": 0,
            "end": 32,
            "length": 32,
            "score": 0.0,
            "length_support_score": 0.0,
        },
        {
            "trajectory_id": 0,
            "start": 32,
            "end": 64,
            "length": 32,
            "score": 0.0,
            "length_support_score": 10.0,
        },
    ]

    length_supported = _allocate_segment_budgets(
        segment_rows=segment_rows,
        retained=retained,
        remaining=1,
        budget=3,
        boundaries=[(0, 64)],
        max_budget_share_per_trajectory=1.0,
        fairness_preallocation_enabled=True,
        segment_length_support_weight=1.0,
    )

    assert length_supported == {1: 1}


def test_learned_segment_budget_allocation_floor_controls_score_contrast() -> None:
    retained = torch.zeros((64,), dtype=torch.bool)
    retained[[0, 63]] = True
    segment_rows = [
        {
            "trajectory_id": 0,
            "start": 0,
            "end": 32,
            "length": 32,
            "score": 10.0,
            "length_support_score": 0.0,
        },
        {
            "trajectory_id": 0,
            "start": 32,
            "end": 64,
            "length": 32,
            "score": 0.0,
            "length_support_score": 0.0,
        },
    ]

    default_floor = _allocate_segment_budgets(
        segment_rows=[dict(row) for row in segment_rows],
        retained=retained,
        remaining=4,
        budget=6,
        boundaries=[(0, 64)],
        max_budget_share_per_trajectory=1.0,
        fairness_preallocation_enabled=False,
        segment_allocation_weight_floor=0.50,
    )
    zero_floor = _allocate_segment_budgets(
        segment_rows=[dict(row) for row in segment_rows],
        retained=retained,
        remaining=4,
        budget=6,
        boundaries=[(0, 64)],
        max_budget_share_per_trajectory=1.0,
        fairness_preallocation_enabled=False,
        segment_allocation_weight_floor=0.0,
    )

    assert default_floor == {0: 3, 1: 1}
    assert zero_floor == {0: 4}


def test_learned_segment_budget_separates_allocation_length_support_from_geometry_gain() -> None:
    steps = torch.arange(0, 16, dtype=torch.float32)
    straight = torch.stack([steps, torch.zeros_like(steps), steps * 0.05], dim=1)
    zigzag = torch.stack(
        [
            steps + 16.0,
            torch.where((steps.long() % 2) == 0, torch.zeros_like(steps), torch.ones_like(steps)),
            steps * 0.05,
        ],
        dim=1,
    )
    points = torch.cat([straight, zigzag], dim=0)
    scores = torch.zeros((32,), dtype=torch.float32)
    segment_scores = torch.zeros_like(scores)
    segment_scores[:16] = 10.0

    _retained_score, score_trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        [(0, 32)],
        compression_ratio=0.20,
        segment_size=16,
        segment_scores=segment_scores,
        points=points,
        geometry_gain_weight=0.0,
        segment_length_support_weight=0.0,
    )
    _retained_support, support_trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        [(0, 32)],
        compression_ratio=0.20,
        segment_size=16,
        segment_scores=segment_scores,
        points=points,
        geometry_gain_weight=0.0,
        segment_length_support_weight=1.0,
    )

    score_rows = {
        int(row["start"]): row for row in score_trace["segment_source_attribution"]["rows"]
    }
    support_rows = {
        int(row["start"]): row for row in support_trace["segment_source_attribution"]["rows"]
    }
    assert support_trace["geometry_tie_breaker_weight"] == 0.0
    assert support_trace["segment_length_support_weight"] == 1.0
    assert score_rows[0]["segment_allocation_count"] > score_rows[16]["segment_allocation_count"]
    assert (
        support_rows[16]["segment_allocation_count"] > support_rows[0]["segment_allocation_count"]
    )


def test_learned_segment_budget_reports_length_support_allocation_counterfactual() -> None:
    steps = torch.arange(0, 16, dtype=torch.float32)
    straight = torch.stack([steps, torch.zeros_like(steps), steps * 0.05], dim=1)
    zigzag = torch.stack(
        [
            steps + 16.0,
            torch.where((steps.long() % 2) == 0, torch.zeros_like(steps), torch.ones_like(steps)),
            steps * 0.05,
        ],
        dim=1,
    )
    points = torch.cat([straight, zigzag], dim=0)
    scores = torch.zeros((32,), dtype=torch.float32)
    segment_scores = torch.zeros_like(scores)
    segment_scores[:16] = 10.0

    _retained, trace = simplify_with_learned_segment_budget_v1_with_trace(
        scores,
        [(0, 32)],
        compression_ratio=0.25,
        segment_size=16,
        segment_scores=segment_scores,
        points=points,
        geometry_gain_weight=0.0,
        segment_length_support_weight=0.0,
        length_repair_fraction=0.0,
    )

    allocation_diagnostic = trace["allocation_point_selection_diagnostics"]
    counterfactual = trace["allocation_counterfactual_diagnostics"]
    assert counterfactual["available"] is True
    assert counterfactual["diagnostic_only"] is True
    assert counterfactual["query_free"] is True
    assert (
        counterfactual["counterfactual_allocation_count_total"]
        == (trace["segment_budget_allocation_count"])
    )
    assert counterfactual["extra_allocation_overlap_fraction"] == pytest.approx(0.0)
    assert (
        counterfactual["length_support_allocation_counterfactual_preservation"]
        > (allocation_diagnostic["same_allocation_length_only_point_selection_preservation"])
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
    assert trace["segment_allocation_weight_floor"] == pytest.approx(0.50)
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
        "segment_length_support_score",
        "segment_length_support_rank",
        "segment_allocation_weight",
        "segment_allocation_weight_rank",
        "retained_count",
        "skeleton_count",
        "learned_count",
        "fallback_count",
        "length_repair_count",
        "unattributed_count",
    }.issubset(first_row)
    alignment = trace["segment_allocation_alignment_diagnostics"]
    assert alignment["available"] is True
    assert alignment["diagnostic_only"] is True
    assert alignment["query_free"] is True
    assert alignment["segment_count"] == attribution["segment_count"]
    assert alignment["allocation_count_total"] == summary["segment_allocation_count_total"]
    assert "top_10_percent" in alignment["top_groups"]


def test_segment_allocation_alignment_diagnostic_flags_score_dominated_extras() -> None:
    segment_rows = [
        {
            "trajectory_id": 0,
            "start": 0,
            "end": 8,
            "score": 10.0,
            "length_support_score": 0.0,
            "allocation_weight": 1.0,
        },
        {
            "trajectory_id": 0,
            "start": 8,
            "end": 16,
            "score": 8.0,
            "length_support_score": 0.0,
            "allocation_weight": 0.8,
        },
        {
            "trajectory_id": 0,
            "start": 16,
            "end": 24,
            "score": 0.0,
            "length_support_score": 10.0,
            "allocation_weight": 0.1,
        },
        {
            "trajectory_id": 0,
            "start": 24,
            "end": 32,
            "score": 0.0,
            "length_support_score": 9.0,
            "allocation_weight": 0.1,
        },
        {
            "trajectory_id": 0,
            "start": 32,
            "end": 40,
            "score": 0.0,
            "length_support_score": 8.0,
            "allocation_weight": 0.1,
        },
    ]

    diagnostic = _segment_allocation_alignment_diagnostics(
        segment_rows=segment_rows,
        segment_allocations={0: 3, 1: 2, 2: 1, 3: 1},
    )

    assert diagnostic["available"] is True
    assert diagnostic["component_diagnosis"] == (
        "extra_slots_score_dominated_not_length_support_aligned"
    )
    assert diagnostic["extra_allocation_count_total"] == 3
    assert diagnostic["length_support_to_allocation_pearson"] < 0.0
    assert diagnostic["segment_score_to_allocation_pearson"] > 0.8
    assert (
        diagnostic["trajectory_top_length_support_extra_capture"][
            "top3_length_support_extra_count_histogram"
        ]["0"]
        == 1
    )
