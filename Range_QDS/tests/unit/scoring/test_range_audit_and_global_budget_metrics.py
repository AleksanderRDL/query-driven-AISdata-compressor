"""Tests for QueryLocalUtility range-audit support and global-budget selectors."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from scoring.method_scoring import _retained_point_gap_stats, score_range_audit
from scoring.methods import ScoreGlobalBudgetMethod
from scoring.metrics import MethodScore, compute_length_preservation
from scoring.query_cache import ScoringQueryCache
from scoring.query_local_utility import (
    QUERY_LOCAL_UTILITY_COMPONENT_WEIGHTS,
    QUERY_LOCAL_UTILITY_SCHEMA_VERSION,
    QUERY_LOCAL_UTILITY_WEIGHTS,
    query_local_utility_from_range_audit,
)
from scoring.score_tables import print_method_comparison_table, print_range_audit_table
from selection.model_score_conversion import simplify_mlqds_predictions
from selection.retained_mask_selectors import (
    simplify_with_global_score_budget,
    simplify_with_temporal_score_hybrid,
)


def test_range_audit_reports_query_local_components_only() -> None:
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

    endpoint_audit = score_range_audit(
        points, [(0, 3)], torch.tensor([True, False, True]), queries
    )
    full_audit = score_range_audit(points, [(0, 3)], torch.tensor([True, True, True]), queries)
    endpoint_utility = query_local_utility_from_range_audit(endpoint_audit)
    full_utility = query_local_utility_from_range_audit(full_audit)

    assert set(endpoint_audit) == {
        "range_query_count",
        "query_point_recall",
        "range_point_f1",
        "range_gap_min_coverage",
        "range_turn_coverage",
        "range_query_local_interpolation_fidelity",
        "range_query_metadata_component_summary",
    }
    assert endpoint_audit["range_turn_coverage"] == pytest.approx(0.0)
    assert full_audit["range_turn_coverage"] == pytest.approx(1.0)
    assert float(cast(Any, full_utility["query_local_utility_score"])) > float(
        cast(Any, endpoint_utility["query_local_utility_score"])
    )


def test_range_audit_cache_reuses_retained_independent_support() -> None:
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

    score_range_audit(
        points, boundaries, torch.tensor([True, False, False, True]), queries, query_cache
    )
    support = query_cache.range_audit_supports[0]
    score_range_audit(
        points, boundaries, torch.tensor([False, True, True, False]), queries, query_cache
    )

    assert len(query_cache.range_audit_supports) == 1
    assert query_cache.range_audit_supports[0] is support
    assert support.range_mask.tolist() == [True, True, True, True]
    assert len(support.trajectories) == 1
    assert set(support.__dataclass_fields__) == {"range_mask", "trajectories"}


def test_query_local_utility_uses_current_component_contract() -> None:
    audit = {
        "query_point_recall": 0.55,
        "range_gap_min_coverage": 0.30,
        "range_turn_coverage": 0.65,
        "range_query_local_interpolation_fidelity": 0.75,
    }

    result = query_local_utility_from_range_audit(
        audit,
        length_preservation=0.80,
        avg_sed_km=0.25,
    )

    assert QUERY_LOCAL_UTILITY_SCHEMA_VERSION == 5
    assert result["query_local_utility_schema_version"] == 5
    assert sum(QUERY_LOCAL_UTILITY_WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(QUERY_LOCAL_UTILITY_COMPONENT_WEIGHTS.values()) == pytest.approx(1.0)
    assert QUERY_LOCAL_UTILITY_WEIGHTS["query_point_mass"] == pytest.approx(0.50)
    assert QUERY_LOCAL_UTILITY_WEIGHTS["query_local_behavior"] == pytest.approx(0.45)
    assert QUERY_LOCAL_UTILITY_WEIGHTS["global_sanity"] == pytest.approx(0.05)
    components = cast(dict[str, float], result["query_local_utility_components"])
    assert set(components) == set(QUERY_LOCAL_UTILITY_COMPONENT_WEIGHTS)
    assert components["query_point_recall"] == pytest.approx(0.55)
    assert components["query_local_continuity"] == pytest.approx(0.30)
    assert components["query_local_turn_change_coverage"] == pytest.approx(0.65)
    assert components["query_local_interpolation_fidelity"] == pytest.approx(0.75)


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


def test_method_comparison_table_labels_range_contract() -> None:
    table = print_method_comparison_table(
        {
            "A": MethodScore(
                aggregate_f1=1.0,
                per_type_f1={"range": 1.0},
                query_point_recall=0.8,
                query_local_utility_score=0.7,
                compression_ratio=1.0,
                latency_ms=0.0,
            )
        }
    )

    assert "RangePointF1" in table
    assert "QueryLocalUtility" in table
    assert "QueryRecall" in table
    assert "EntryExitF1" not in table
    assert "AnswerF1" not in table
    assert "0.700000" in table


def test_range_audit_table_reports_current_components() -> None:
    table = print_range_audit_table(
        {
            "A": MethodScore(
                aggregate_f1=0.5,
                per_type_f1={"range": 0.5},
                query_point_recall=0.45,
                range_point_f1=0.5,
                range_gap_min_coverage=0.4,
                range_turn_coverage=0.3,
                range_query_local_interpolation_fidelity=0.6,
                query_local_utility_score=0.5035,
            )
        }
    )

    assert "RangePointF1" in table
    assert "QueryRecall" in table
    assert "QueryLocalUtility" in table
    assert "GapMin" in table
    assert "TurnCov" in table
    assert "InterpFid" in table
    assert "ShipF1" not in table
    assert "CrossingF1" not in table


def test_method_comparison_table_reports_canonical_baseline_diffs() -> None:
    table = print_method_comparison_table(
        {
            "MLQDS": MethodScore(
                aggregate_f1=0.6,
                per_type_f1={"range": 0.6},
                query_local_utility_score=0.7,
                compression_ratio=0.2,
                latency_ms=0.0,
            ),
            "uniform": MethodScore(
                aggregate_f1=0.5,
                per_type_f1={"range": 0.5},
                query_local_utility_score=0.6,
                compression_ratio=0.2,
                latency_ms=0.0,
            ),
            "DouglasPeucker": MethodScore(
                aggregate_f1=0.55,
                per_type_f1={"range": 0.55},
                query_local_utility_score=0.65,
                compression_ratio=0.2,
                latency_ms=0.0,
            ),
        }
    )

    assert "Diff vs MLQDS" in table
    assert "vs uniform" in table
    assert "vs DouglasPeucker" in table
    assert "RangePointF1 / QueryLocalUtility" in table


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
