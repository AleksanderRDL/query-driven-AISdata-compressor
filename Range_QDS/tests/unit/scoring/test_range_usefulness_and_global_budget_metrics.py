"""Tests F1-based query metrics. See scoring/README.md for details."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from scoring.method_scoring import (
    _retained_point_gap_stats,
    score_range_usefulness,
)
from scoring.methods import ScoreGlobalBudgetMethod
from scoring.metrics import MethodScore, compute_length_preservation
from scoring.query_cache import ScoringQueryCache
from scoring.query_local_utility import (
    QUERY_LOCAL_UTILITY_COMPONENT_WEIGHTS,
    QUERY_LOCAL_UTILITY_SCHEMA_VERSION,
    QUERY_LOCAL_UTILITY_WEIGHTS,
    query_local_utility_from_range_audit,
)
from scoring.range_usefulness import range_usefulness_weight_summary
from scoring.score_tables import print_method_comparison_table, print_range_usefulness_table
from selection.model_score_conversion import simplify_mlqds_predictions
from selection.retained_mask_selectors import (
    simplify_with_global_score_budget,
    simplify_with_temporal_score_hybrid,
)


class KeepAllMethod:
    name = "KeepAll"

    def simplify(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        compression_ratio: float,
    ) -> torch.Tensor:
        return torch.ones((points.shape[0],), dtype=torch.bool, device=points.device)


class DropAllMethod:
    name = "DropAll"

    def simplify(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        compression_ratio: float,
    ) -> torch.Tensor:
        return torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)


class FixedMaskMethod:
    def __init__(self, retained_mask: torch.Tensor) -> None:
        self.retained_mask = retained_mask
        self.name = "FixedMask"

    def simplify(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        compression_ratio: float,
    ) -> torch.Tensor:
        return self.retained_mask.clone()


def test_range_usefulness_shape_score_penalizes_curved_endpoint_shortcut() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 0.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 2.0,
                "lon_min": -1.0,
                "lon_max": 3.0,
                "t_start": -1.0,
                "t_end": 3.0,
            },
        }
    ]
    retained = torch.tensor([True, False, True])

    audit = score_range_usefulness(points, [(0, 3)], retained, queries)

    assert audit["range_temporal_coverage"] == pytest.approx(1.0)
    assert 0.0 < audit["range_shape_score"] < 0.65


def test_range_usefulness_turn_coverage_penalizes_missing_change_point() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 0.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 2.0,
                "lon_min": -1.0,
                "lon_max": 3.0,
                "t_start": -1.0,
                "t_end": 3.0,
            },
        }
    ]
    endpoint_retained = torch.tensor([True, False, True])
    all_retained = torch.tensor([True, True, True])

    endpoint_audit = score_range_usefulness(points, [(0, 3)], endpoint_retained, queries)
    full_audit = score_range_usefulness(points, [(0, 3)], all_retained, queries)

    assert endpoint_audit["range_turn_coverage"] == pytest.approx(0.0)
    assert full_audit["range_turn_coverage"] == pytest.approx(1.0)
    assert full_audit["range_usefulness_score"] > endpoint_audit["range_usefulness_score"]


def test_range_usefulness_cache_reuses_retained_independent_support() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.1, 1.0],
            [2.0, 0.0, 0.2, 1.0],
            [3.0, 0.0, 0.3, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 4)]
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 4.0,
            },
        }
    ]
    query_cache = ScoringQueryCache.for_workload(points, boundaries, queries)

    score_range_usefulness(
        points, boundaries, torch.tensor([True, False, False, True]), queries, query_cache
    )
    support = query_cache.range_audit_supports[0]
    score_range_usefulness(
        points, boundaries, torch.tensor([False, True, True, False]), queries, query_cache
    )

    assert len(query_cache.range_audit_supports) == 1
    assert query_cache.range_audit_supports[0] is support
    assert support.boundary_indices_cpu.tolist() == [0, 3]


def test_retained_point_gap_stats_measure_original_spacing() -> None:
    retained = torch.tensor([True, False, False, True, False, True, True, False, True])

    avg_gap, avg_norm_gap, max_gap = _retained_point_gap_stats(
        retained, boundaries=[(0, 6), (6, 9)]
    )

    assert avg_gap == pytest.approx((3.0 + 2.0 + 2.0) / 3.0)
    assert avg_norm_gap == pytest.approx(((3.0 / 5.0) + (2.0 / 5.0) + (2.0 / 2.0)) / 3.0)
    assert max_gap == pytest.approx(3.0)


def test_global_score_budget_preserves_skeleton_and_spends_remaining_budget_globally() -> None:
    scores = torch.tensor([0.0, 0.1, 0.2, 0.3, 10.0, 9.0, 8.0, 7.0], dtype=torch.float32)
    boundaries = [(0, 4), (4, 8)]

    retained = simplify_with_global_score_budget(scores, boundaries, compression_ratio=0.75)

    assert int(retained.sum().item()) == 6
    assert retained[[0, 3, 4, 7]].all()
    assert retained.tolist() == [True, False, False, True, True, True, True, True]


def test_temporal_score_hybrid_global_budget_delegates_to_global_allocator() -> None:
    scores = torch.tensor([0.0, 0.1, 0.2, 0.3, 10.0, 9.0, 8.0, 7.0], dtype=torch.float32)
    boundaries = [(0, 4), (4, 8)]

    retained = simplify_with_temporal_score_hybrid(
        scores=scores,
        boundaries=boundaries,
        compression_ratio=0.75,
        temporal_fraction=0.99,
        diversity_bonus=10.0,
        hybrid_mode="global_budget",
    )

    assert retained.tolist() == [True, False, False, True, True, True, True, True]


def test_temporal_score_hybrid_global_fill_preserves_base_and_spends_residual_globally() -> None:
    scores = torch.tensor([5.0, 4.0, 0.0, 0.0, 10.0, 9.0, 8.0, 7.0], dtype=torch.float32)
    boundaries = [(0, 4), (4, 8)]

    retained = simplify_with_temporal_score_hybrid(
        scores=scores,
        boundaries=boundaries,
        compression_ratio=0.75,
        temporal_fraction=0.34,
        diversity_bonus=0.0,
        hybrid_mode="global_fill",
    )

    assert int(retained.sum().item()) == 6
    assert retained[[0, 3, 4, 7]].all()
    assert retained.tolist() == [True, False, False, True, True, True, True, True]


def test_score_global_budget_method_validates_score_count() -> None:
    points = torch.zeros((4, 4), dtype=torch.float32)
    method = ScoreGlobalBudgetMethod(name="Global", scores=torch.ones((3,), dtype=torch.float32))

    with pytest.raises(ValueError, match="scores must match flattened points"):
        method.simplify(points, [(0, 4)], compression_ratio=0.5)


def test_method_comparison_table_labels_f1() -> None:
    table = print_method_comparison_table(
        {
            "A": MethodScore(
                aggregate_f1=1.0,
                per_type_f1={"range": 1.0},
                compression_ratio=1.0,
                latency_ms=0.0,
            )
        }
    )

    assert "RangePointF1" in table
    assert "RangeUseful" in table
    assert "AvgPtGap" in table
    assert "EntryExitF1" in table
    assert "AnswerF1" not in table
    assert "AggregateErr" not in table


def test_range_usefulness_table_reports_audit_components() -> None:
    table = print_range_usefulness_table(
        {
            "A": MethodScore(
                aggregate_f1=0.5,
                per_type_f1={"range": 0.5},
                range_point_f1=0.5,
                range_ship_f1=0.7,
                range_ship_coverage=0.65,
                range_entry_exit_f1=0.25,
                range_crossing_f1=0.2,
                range_temporal_coverage=0.8,
                range_gap_coverage=0.4,
                range_gap_time_coverage=0.35,
                range_gap_distance_coverage=0.45,
                range_turn_coverage=0.3,
                range_shape_score=0.6,
                range_usefulness_score=0.5035,
            )
        }
    )

    assert "RangePointF1" in table
    assert "ShipF1" in table
    assert "ShipCov" in table
    assert "CrossingF1" in table
    assert "GapCov" in table
    assert "GapTime" in table
    assert "GapDist" in table
    assert "TurnCov" in table
    assert "RangeUseful" in table


def test_range_usefulness_weight_summary_groups_current_schema() -> None:
    summary = range_usefulness_weight_summary()

    assert summary["schema_version"] == 7
    assert summary["total_weight"] == pytest.approx(1.0)
    group_weights = summary["group_weights"]
    assert isinstance(group_weights, dict)
    assert group_weights["point_statistical_coverage"] == pytest.approx(0.22)
    assert group_weights["ship_representation"] == pytest.approx(0.26)
    assert group_weights["boundary_context"] == pytest.approx(0.20)
    assert group_weights["temporal_continuity"] == pytest.approx(0.19)
    assert group_weights["route_fidelity"] == pytest.approx(0.13)


def test_query_local_utility_uses_direct_query_local_components() -> None:
    removed_components = {
        "ship_balanced_query_point_recall",
        "ship_f1",
        "ship_coverage",
        "multi_point_ship_evidence",
        "entry_exit_f1",
        "crossing_f1",
        "query_boundary_evidence",
    }
    base_audit = {
        "query_point_recall": 0.55,
        "range_point_f1": 0.55,
        "range_ship_coverage": 0.05,
        "range_ship_f1": 0.10,
        "range_temporal_coverage": 0.35,
        "range_gap_time_coverage": 0.40,
        "range_gap_distance_coverage": 0.30,
        "range_gap_min_coverage": 0.30,
        "range_turn_coverage": 0.65,
        "range_shape_score": 0.70,
        "range_query_local_interpolation_fidelity": 0.75,
        "range_entry_exit_f1": 0.05,
        "range_crossing_f1": 0.95,
    }
    changed_removed_only = {
        **base_audit,
        "range_point_f1": 0.0,
        "range_ship_coverage": 1.0,
        "range_ship_f1": 1.0,
        "range_entry_exit_f1": 1.0,
        "range_crossing_f1": 0.0,
    }

    base = query_local_utility_from_range_audit(
        base_audit,
        length_preservation=0.80,
        avg_sed_km=0.25,
    )
    changed = query_local_utility_from_range_audit(
        changed_removed_only,
        length_preservation=0.80,
        avg_sed_km=0.25,
    )

    assert QUERY_LOCAL_UTILITY_SCHEMA_VERSION == 5
    assert base["query_local_utility_schema_version"] == 5
    assert sum(QUERY_LOCAL_UTILITY_WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(QUERY_LOCAL_UTILITY_COMPONENT_WEIGHTS.values()) == pytest.approx(1.0)
    assert QUERY_LOCAL_UTILITY_WEIGHTS["query_point_mass"] == pytest.approx(0.50)
    assert QUERY_LOCAL_UTILITY_WEIGHTS["query_local_behavior"] == pytest.approx(0.45)
    assert QUERY_LOCAL_UTILITY_WEIGHTS["global_sanity"] == pytest.approx(0.05)
    assert "ship_presence_and_coverage" not in QUERY_LOCAL_UTILITY_WEIGHTS
    assert "boundary_and_event_evidence" not in QUERY_LOCAL_UTILITY_WEIGHTS
    assert removed_components.isdisjoint(QUERY_LOCAL_UTILITY_COMPONENT_WEIGHTS)
    components = cast(dict[str, float], base["query_local_utility_components"])
    component_weights = cast(dict[str, float], base["query_local_utility_component_weights"])
    assert removed_components.isdisjoint(components)
    assert removed_components.isdisjoint(component_weights)
    assert base["query_local_utility_score"] == pytest.approx(changed["query_local_utility_score"])

    legacy_range_point_only_audit = {
        "range_point_f1": 1.0,
        "range_shape_score": 1.0,
        "range_gap_time_coverage": 1.0,
        "range_gap_distance_coverage": 1.0,
    }
    legacy_range_point_only = query_local_utility_from_range_audit(legacy_range_point_only_audit)
    legacy_range_point_components = cast(
        dict[str, float], legacy_range_point_only["query_local_utility_components"]
    )
    assert legacy_range_point_components["query_point_recall"] == pytest.approx(0.0)
    assert legacy_range_point_components["query_local_interpolation_fidelity"] == pytest.approx(0.0)
    assert legacy_range_point_components["query_local_turn_change_coverage"] == pytest.approx(0.0)
    assert legacy_range_point_components["query_local_continuity"] == pytest.approx(0.0)


def test_method_comparison_table_shows_close_f1_values() -> None:
    table = print_method_comparison_table(
        {
            "MLQDS": MethodScore(
                aggregate_f1=0.1823688112,
                per_type_f1={"range": 0.1823688112},
                compression_ratio=0.1008,
                latency_ms=0.0,
            ),
            "Random": MethodScore(
                aggregate_f1=0.1824232682,
                per_type_f1={"range": 0.1824232682},
                compression_ratio=0.1008,
                latency_ms=0.0,
            ),
        }
    )

    assert "0.182369" in table
    assert "0.182423" in table


def test_method_comparison_table_reports_canonical_baseline_diffs() -> None:
    table = print_method_comparison_table(
        {
            "MLQDS": MethodScore(
                aggregate_f1=0.6,
                per_type_f1={"range": 0.6},
                compression_ratio=0.2,
                latency_ms=0.0,
            ),
            "uniform": MethodScore(
                aggregate_f1=0.5,
                per_type_f1={"range": 0.5},
                compression_ratio=0.2,
                latency_ms=0.0,
            ),
            "DouglasPeucker": MethodScore(
                aggregate_f1=0.55,
                per_type_f1={"range": 0.55},
                compression_ratio=0.2,
                latency_ms=0.0,
            ),
        }
    )

    assert "Diff vs MLQDS" in table
    assert "vs uniform" in table
    assert "vs DouglasPeucker" in table
    assert "vs Random" not in table


def test_length_preservation_reports_preserved_path_length() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    retained = torch.tensor([True, False, True])

    assert compute_length_preservation(points, [(0, 3)], retained) == pytest.approx(1.0)


def test_mlqds_range_geometry_blend_can_override_model_score_order() -> None:
    predictions = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], dtype=torch.float32)
    geometry_scores = torch.tensor([0.0, 0.0, 10.0, 9.0, 0.0], dtype=torch.float32)
    boundaries = [(0, 5)]

    retained = simplify_mlqds_predictions(
        predictions,
        boundaries,
        workload_type="range",
        compression_ratio=0.4,
        temporal_fraction=0.0,
        diversity_bonus=0.0,
        range_geometry_scores=geometry_scores,
        range_geometry_blend=1.0,
    )

    assert retained.tolist() == [False, False, True, True, False]
