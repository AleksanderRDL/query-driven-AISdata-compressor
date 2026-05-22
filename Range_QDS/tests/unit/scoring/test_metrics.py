"""Tests F1-based query metrics. See scoring/README.md for details."""

from __future__ import annotations

import pytest
import torch

from scoring.method_scoring import (
    score_method,
    score_range_boundary_preservation,
    score_range_usefulness,
    score_retained_mask,
)
from scoring.methods import OracleMethod, UniformTemporalMethod
from scoring.metrics import f1_score
from scoring.query_cache import ScoringQueryCache
from selection.model_score_conversion import pure_workload_scores
from selection.retained_mask_selectors import (
    simplify_with_scores,
    simplify_with_temporal_score_hybrid,
    temporal_hybrid_selector_budget_diagnostics,
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


def test_f1_score_identical_sets() -> None:
    assert f1_score({1, 2, 3}, {1, 2, 3}) == pytest.approx(1.0)


def test_f1_score_disjoint_sets() -> None:
    assert f1_score({1, 2}, {3, 4}) == pytest.approx(0.0)


def test_f1_score_both_empty() -> None:
    assert f1_score(set(), set()) == pytest.approx(1.0)


def test_f1_score_one_empty() -> None:
    assert f1_score({1}, set()) == pytest.approx(0.0)
    assert f1_score(set(), {1}) == pytest.approx(0.0)


def test_f1_score_partial_overlap() -> None:
    # precision = 2/3, recall = 2/4, F1 = 4/7.
    assert f1_score({1, 2, 3, 4}, {2, 3, 5}) == pytest.approx(4.0 / 7.0)


def test_score_method_scores_noop_above_degenerate_baseline() -> None:
    trajectories = [
        torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, 0.2, 0.2, 1.0]], dtype=torch.float32),
        torch.tensor([[0.0, 5.0, 5.0, 1.0], [1.0, 5.2, 5.2, 1.0]], dtype=torch.float32),
    ]
    points = torch.cat(trajectories, dim=0)
    boundaries = [(0, 2), (2, 4)]
    typed_queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 2.0,
            },
        }
    ]

    keep_all = score_method(
        method=KeepAllMethod(),
        points=points,
        boundaries=boundaries,
        typed_queries=typed_queries,
        workload_map={"range": 1.0},
        compression_ratio=1.0,
    )
    drop_all = score_method(
        method=DropAllMethod(),
        points=points,
        boundaries=boundaries,
        typed_queries=typed_queries,
        workload_map={"range": 1.0},
        compression_ratio=0.0,
    )

    assert keep_all.aggregate_f1 == pytest.approx(1.0)
    assert drop_all.aggregate_f1 == pytest.approx(0.0)
    assert keep_all.aggregate_f1 > drop_all.aggregate_f1


def test_score_retained_mask_matches_score_method() -> None:
    trajectories = [
        torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, 0.2, 0.2, 1.0]], dtype=torch.float32),
        torch.tensor([[0.0, 5.0, 5.0, 1.0], [1.0, 5.2, 5.2, 1.0]], dtype=torch.float32),
    ]
    points = torch.cat(trajectories, dim=0)
    boundaries = [(0, 2), (2, 4)]
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 2.0,
            },
        }
    ]
    retained = torch.tensor([True, False, True, True])

    aggregate, per_type, _, _ = score_retained_mask(
        points=points,
        boundaries=boundaries,
        retained_mask=retained,
        typed_queries=queries,
        workload_map={"range": 1.0},
    )
    scored = score_method(
        method=FixedMaskMethod(retained),
        points=points,
        boundaries=boundaries,
        typed_queries=queries,
        workload_map={"range": 1.0},
        compression_ratio=0.75,
    )

    assert aggregate == pytest.approx(scored.aggregate_f1)
    assert per_type == pytest.approx(scored.per_type_f1)


def test_score_retained_mask_cache_reuses_full_query_results(monkeypatch) -> None:
    import scoring.method_scoring as scoring_methods

    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.2, 1.0],
            [0.0, 5.0, 5.0, 1.0],
            [1.0, 5.0, 5.2, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 2), (2, 4)]
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 2.0,
            },
        }
    ]
    original_points_in_range_box = scoring_methods.points_in_range_box
    calls = {"range_mask": 0}

    def counting_points_in_range_box(
        query_points: torch.Tensor,
        params: dict[str, float],
    ) -> torch.Tensor:
        calls["range_mask"] += 1
        return original_points_in_range_box(query_points, params)

    monkeypatch.setattr(scoring_methods, "points_in_range_box", counting_points_in_range_box)
    query_cache = ScoringQueryCache.for_workload(points, boundaries, queries)

    for retained in (
        torch.tensor([True, False, True, True]),
        torch.tensor([True, True, True, False]),
    ):
        score_retained_mask(
            points=points,
            boundaries=boundaries,
            retained_mask=retained,
            typed_queries=queries,
            workload_map={"range": 1.0},
            query_cache=query_cache,
        )

    assert calls == {"range_mask": 1}


def test_score_retained_mask_cache_rejects_different_workload() -> None:
    points = torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    boundaries = [(0, 2)]
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 0.5,
                "lon_min": -1.0,
                "lon_max": 0.5,
                "t_start": -1.0,
                "t_end": 0.5,
            },
        }
    ]
    other_queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 2.0,
                "lon_min": -1.0,
                "lon_max": 2.0,
                "t_start": -1.0,
                "t_end": 2.0,
            },
        }
    ]
    query_cache = ScoringQueryCache.for_workload(points, boundaries, queries)

    with pytest.raises(ValueError, match="ScoringQueryCache"):
        score_retained_mask(
            points=points,
            boundaries=boundaries,
            retained_mask=torch.tensor([True, False]),
            typed_queries=other_queries,
            workload_map={"range": 1.0},
            query_cache=query_cache,
        )


def test_uniform_temporal_is_evenly_spaced() -> None:
    points = torch.stack(
        [torch.tensor([float(i), 0.0, float(i), 1.0], dtype=torch.float32) for i in range(10)]
    )
    boundaries = [(0, 10)]

    retained = UniformTemporalMethod().simplify(points, boundaries, compression_ratio=0.3)

    assert torch.where(retained)[0].tolist() == [0, 4, 9]


def test_temporal_score_hybrid_keeps_base_and_score_fill() -> None:
    scores = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    retained = simplify_with_temporal_score_hybrid(
        scores=scores,
        boundaries=[(0, 10)],
        compression_ratio=0.3,
        temporal_fraction=0.5,
        diversity_bonus=0.0,
    )

    assert torch.where(retained)[0].tolist() == [0, 5, 9]


def test_temporal_score_hybrid_zero_temporal_fraction_is_pure_score() -> None:
    scores = torch.tensor([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 4.0, 5.0, 6.0])
    retained = simplify_with_temporal_score_hybrid(
        scores=scores,
        boundaries=[(0, 10)],
        compression_ratio=0.3,
        temporal_fraction=0.0,
        diversity_bonus=0.0,
    )

    assert torch.where(retained)[0].tolist() == [4, 5, 6]


def test_temporal_score_hybrid_diversity_bonus_spreads_learned_fill() -> None:
    scores = torch.zeros((10,), dtype=torch.float32)
    retained = simplify_with_temporal_score_hybrid(
        scores=scores,
        boundaries=[(0, 10)],
        compression_ratio=0.5,
        temporal_fraction=0.4,
        diversity_bonus=10.0,
    )

    retained_idx = torch.where(retained)[0].tolist()
    assert retained_idx[0] == 0
    assert retained_idx[-1] == 9
    assert len(retained_idx) == 5
    assert any(3 <= idx <= 6 for idx in retained_idx)


def test_temporal_score_hybrid_swap_starts_from_full_uniform_and_replaces_low_score_base() -> None:
    scores = torch.tensor([0.0, 3.0, -1.0, 0.0, 1.0, 10.0, 0.0, 1.0, 0.0, 0.0])
    retained = simplify_with_temporal_score_hybrid(
        scores=scores,
        boundaries=[(0, 10)],
        compression_ratio=0.5,
        temporal_fraction=0.8,
        diversity_bonus=0.0,
        hybrid_mode="swap",
    )

    assert torch.where(retained)[0].tolist() == [0, 4, 5, 7, 9]


def test_temporal_score_hybrid_local_swap_removes_nearest_base_point() -> None:
    scores = torch.tensor([0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    retained = simplify_with_temporal_score_hybrid(
        scores=scores,
        boundaries=[(0, 10)],
        compression_ratio=0.4,
        temporal_fraction=0.75,
        diversity_bonus=0.0,
        hybrid_mode="local_swap",
    )

    assert torch.where(retained)[0].tolist() == [0, 3, 5, 9]


def test_temporal_score_hybrid_min_learned_swaps_overrides_rounding_tie() -> None:
    scores = torch.tensor([0.0, 0.0, 0.0, 0.0, -1.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    no_swap = simplify_with_temporal_score_hybrid(
        scores=scores,
        boundaries=[(0, 10)],
        compression_ratio=0.3,
        temporal_fraction=0.95,
        diversity_bonus=0.0,
        hybrid_mode="local_swap",
    )
    forced_swap = simplify_with_temporal_score_hybrid(
        scores=scores,
        boundaries=[(0, 10)],
        compression_ratio=0.3,
        temporal_fraction=0.95,
        diversity_bonus=0.0,
        hybrid_mode="local_swap",
        min_learned_swaps=1,
    )

    assert torch.where(no_swap)[0].tolist() == [0, 4, 9]
    assert torch.where(forced_swap)[0].tolist() == [0, 5, 9]


def test_temporal_hybrid_selector_budget_diagnostics_exposes_zero_learned_slots() -> None:
    diagnostics = temporal_hybrid_selector_budget_diagnostics(
        boundaries=[(0, 192), (192, 384)],
        compression_ratios=[0.01, 0.02, 0.05],
        temporal_fraction=0.85,
        hybrid_mode="local_swap",
    )

    rows = {row["compression_ratio"]: row for row in diagnostics["budget_rows"]}
    assert rows[0.01]["learned_slot_count"] == 0
    assert rows[0.01]["endpoint_only_trajectory_fraction"] == 1.0
    assert rows[0.02]["learned_slot_count"] == 0
    assert rows[0.02]["zero_learned_slot_trajectory_fraction"] == 1.0
    assert rows[0.05]["learned_slot_count"] == 2
    assert rows[0.05]["learned_slot_fraction_of_budget"] == pytest.approx(0.1)


def test_temporal_score_hybrid_local_delta_swap_requires_positive_replacement() -> None:
    scores = torch.tensor([0.0, 1.0, 0.0, 5.0, 4.0, 4.5, 6.0, 5.0, 0.0, 0.0])
    retained = simplify_with_temporal_score_hybrid(
        scores=scores,
        boundaries=[(0, 10)],
        compression_ratio=0.4,
        temporal_fraction=0.75,
        diversity_bonus=0.0,
        hybrid_mode="local_delta_swap",
    )

    assert torch.where(retained)[0].tolist() == [0, 3, 6, 9]


def test_temporal_score_hybrid_local_delta_swap_uses_score_delta_not_raw_score() -> None:
    scores = torch.tensor([0.0, 9.0, 0.0, 8.5, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    retained = simplify_with_temporal_score_hybrid(
        scores=scores,
        boundaries=[(0, 10)],
        compression_ratio=0.4,
        temporal_fraction=0.75,
        diversity_bonus=0.0,
        hybrid_mode="local_delta_swap",
    )

    assert torch.where(retained)[0].tolist() == [0, 3, 7, 9]


def test_temporal_score_hybrid_stratified_uses_learned_scores_inside_bins() -> None:
    scores = torch.tensor([0.0, 0.0, 9.0, 0.0, 0.0, 8.0, 0.0, 7.0, 0.0, 0.0])
    retained = simplify_with_temporal_score_hybrid(
        scores=scores,
        boundaries=[(0, 10)],
        compression_ratio=0.5,
        temporal_fraction=0.0,
        diversity_bonus=0.0,
        hybrid_mode="stratified",
    )

    assert torch.where(retained)[0].tolist() == [0, 2, 5, 7, 9]


def test_temporal_score_hybrid_stratified_center_weight_regularizes_within_bin() -> None:
    scores = torch.zeros((10,), dtype=torch.float32)
    scores[1] = 1.0
    scores[4] = 0.8

    plain = simplify_with_temporal_score_hybrid(
        scores=scores,
        boundaries=[(0, 10)],
        compression_ratio=0.3,
        temporal_fraction=0.0,
        diversity_bonus=0.0,
        hybrid_mode="stratified",
        stratified_center_weight=0.0,
    )
    regularized = simplify_with_temporal_score_hybrid(
        scores=scores,
        boundaries=[(0, 10)],
        compression_ratio=0.3,
        temporal_fraction=0.0,
        diversity_bonus=0.0,
        hybrid_mode="stratified",
        stratified_center_weight=0.5,
    )

    assert torch.where(plain)[0].tolist() == [0, 1, 9]
    assert torch.where(regularized)[0].tolist() == [0, 4, 9]


def test_pure_workload_scores_rank_mode_is_canonical_per_trajectory() -> None:
    predictions = torch.tensor([0.1, 0.9, 0.5, 0.2], dtype=torch.float32)

    scores = pure_workload_scores(predictions, [(0, 4)], "range", score_mode="rank")

    assert scores.tolist() == pytest.approx([0.0, 1.0, 2.0 / 3.0, 1.0 / 3.0])


def test_pure_workload_scores_support_raw_and_sigmoid_modes() -> None:
    predictions = torch.tensor([0.1, 0.9, 0.5, 0.2], dtype=torch.float32)

    raw = pure_workload_scores(predictions, [(0, 4)], "range", score_mode="raw")
    sigmoid = pure_workload_scores(predictions, [(0, 4)], "range", score_mode="sigmoid")

    assert raw.tolist() == pytest.approx([0.1, 0.9, 0.5, 0.2])
    assert sigmoid.tolist() == pytest.approx(torch.sigmoid(predictions).tolist())


def test_pure_workload_scores_support_tie_aware_rank() -> None:
    predictions = torch.tensor([0.1, 0.9, 0.9, 0.2], dtype=torch.float32)

    scores = pure_workload_scores(predictions, [(0, 4)], "range", score_mode="rank_tie")

    assert scores.tolist() == pytest.approx([0.0, 5.0 / 6.0, 5.0 / 6.0, 1.0 / 3.0])


def test_pure_workload_scores_support_calibrated_modes() -> None:
    predictions = torch.tensor([0.1, 0.9, 0.5, 0.2], dtype=torch.float32)

    zscore = pure_workload_scores(predictions, [(0, 4)], "range", score_mode="zscore_sigmoid")
    blend = pure_workload_scores(
        predictions,
        [(0, 4)],
        "range",
        score_mode="rank_confidence",
        rank_confidence_weight=0.50,
    )
    temp_sigmoid = pure_workload_scores(
        predictions,
        [(0, 4)],
        "range",
        score_mode="temperature_sigmoid",
        score_temperature=2.0,
    )

    assert torch.all((zscore >= 0.0) & (zscore <= 1.0))
    assert torch.all((blend >= 0.0) & (blend <= 1.0))
    assert temp_sigmoid.tolist() == pytest.approx(torch.sigmoid(predictions / 2.0).tolist())


def test_pure_workload_scores_zscore_mode_handles_flat_logits() -> None:
    predictions = torch.ones((4,), dtype=torch.float32)

    scores = pure_workload_scores(predictions, [(0, 4)], "range", score_mode="zscore_sigmoid")

    assert scores.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_pure_workload_scores_reject_unknown_mode() -> None:
    predictions = torch.zeros((4,), dtype=torch.float32)

    with pytest.raises(ValueError, match="score_mode"):
        pure_workload_scores(predictions, [(0, 4)], "range", score_mode="not-a-mode")


def test_oracle_method_uses_explicit_workload_head() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.1, 1.0],
            [2.0, 0.0, 0.2, 1.0],
            [3.0, 0.0, 0.3, 1.0],
            [4.0, 0.0, 0.4, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.zeros((5, 4), dtype=torch.float32)
    labels[1, 0] = 1.0
    labels[2, 0] = -1.0
    labels[1, 1] = -1.0
    labels[2, 1] = 1.0

    retained = OracleMethod(labels=labels, workload_type="range").simplify(
        points,
        boundaries=[(0, 5)],
        compression_ratio=0.4,
    )

    assert bool(retained[1].item()) is True
    assert bool(retained[2].item()) is False
    assert OracleMethod(labels=labels, workload_type="range").oracle_kind == "additive_label_greedy"


def test_score_simplifier_skips_empty_boundaries() -> None:
    scores = torch.tensor([0.1, 0.9, 0.2], dtype=torch.float32)

    retained = simplify_with_scores(scores, [(0, 0), (0, 3)], compression_ratio=0.4)

    assert retained.tolist() == [True, True, True]


def test_range_boundary_preservation_is_separate_from_range_f1() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.1, 1.0],
            [2.0, 0.0, 0.2, 1.0],
            [3.0, 0.0, 0.3, 1.0],
            [4.0, 9.0, 9.0, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 5)]
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 3.5,
            },
        }
    ]
    retained = torch.tensor([True, False, False, True, True])

    aggregate, per_type, _, _ = score_retained_mask(
        points, boundaries, retained, queries, {"range": 1.0}
    )
    boundary_f1 = score_range_boundary_preservation(points, boundaries, retained, queries)

    assert aggregate == pytest.approx(2.0 / 3.0)
    assert per_type["range"] == pytest.approx(2.0 / 3.0)
    assert boundary_f1 == pytest.approx(1.0)


def test_range_usefulness_audit_separates_point_hits_from_local_shape() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.1, 1.0],
            [2.0, 0.0, 0.2, 1.0],
            [3.0, 0.0, 0.3, 1.0],
            [4.0, 0.0, 0.4, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 5)]
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 5.0,
            },
        }
    ]
    endpoint_retained = torch.tensor([True, False, False, False, True])
    middle_retained = torch.tensor([False, True, True, False, False])

    endpoint_audit = score_range_usefulness(points, boundaries, endpoint_retained, queries)
    middle_audit = score_range_usefulness(points, boundaries, middle_retained, queries)

    assert endpoint_audit["range_point_f1"] == pytest.approx(middle_audit["range_point_f1"])
    assert endpoint_audit["range_entry_exit_f1"] > middle_audit["range_entry_exit_f1"]
    assert endpoint_audit["range_temporal_coverage"] > middle_audit["range_temporal_coverage"]
    assert endpoint_audit["range_usefulness_score"] > middle_audit["range_usefulness_score"]


def test_range_usefulness_ship_f1_requires_each_hit_ship_present() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.1, 1.0],
            [0.0, 0.2, 0.0, 1.0],
            [1.0, 0.2, 0.1, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 2), (2, 4)]
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 2.0,
            },
        }
    ]
    retained = torch.tensor([True, False, False, False])

    audit = score_range_usefulness(points, boundaries, retained, queries)

    assert audit["query_point_recall"] == pytest.approx(0.25)
    assert audit["range_point_f1"] == pytest.approx(0.4)
    assert audit["range_ship_f1"] == pytest.approx(2.0 / 3.0)


def test_range_usefulness_ship_coverage_penalizes_sparse_hit_ship_representation() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.1, 1.0],
            [2.0, 0.0, 0.2, 1.0],
            [3.0, 0.0, 0.3, 1.0],
            [0.0, 0.2, 0.0, 1.0],
            [1.0, 0.2, 0.1, 1.0],
            [2.0, 0.2, 0.2, 1.0],
            [3.0, 0.2, 0.3, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 4), (4, 8)]
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
    retained = torch.tensor([True, True, True, True, True, False, False, False])

    audit = score_range_usefulness(points, boundaries, retained, queries)

    assert audit["range_ship_f1"] == pytest.approx(1.0)
    assert audit["range_ship_coverage"] == pytest.approx((1.0 + 0.4) / 2.0)
    assert audit["range_ship_coverage"] < audit["range_ship_f1"]


def test_range_usefulness_reports_query_family_component_summary() -> None:
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
    params = {
        "lat_min": -1.0,
        "lat_max": 1.0,
        "lon_min": -1.0,
        "lon_max": 1.0,
        "t_start": -1.0,
        "t_end": 4.0,
    }
    queries = [
        {
            "type": "range",
            "params": params,
            "_metadata": {
                "anchor_family": "density",
                "footprint_family": "medium_operational",
            },
        },
        {
            "type": "range",
            "params": params,
            "_metadata": {
                "anchor_family": "sparse_background_control",
                "footprint_family": "large_context",
            },
        },
    ]
    retained = torch.tensor([True, True, True, True])

    audit = score_range_usefulness(points, boundaries, retained, queries)

    summary = audit["range_query_metadata_component_summary"]
    assert summary["available"] is True
    assert summary["diagnostic_only"] is True
    assert summary["query_count"] == 2
    assert len(summary["query_rows"]) == 2
    assert "length_preservation_guardrail" in summary["excluded_query_local_utility_components"]
    anchor_groups = summary["group_by"]["anchor_family"]
    assert set(anchor_groups) == {"density", "sparse_background_control"}
    density = anchor_groups["density"]
    assert density["query_count"] == 1
    assert density["range_components"]["query_point_recall"] == pytest.approx(1.0)
    assert density["range_components"]["range_point_f1"] == pytest.approx(1.0)
    assert density["query_local_utility_query_local_weighted_score_normalized"] == pytest.approx(
        1.0
    )


def test_range_usefulness_reports_ship_evidence_counts_by_query_family() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.1, 1.0],
            [0.0, 0.2, 0.0, 1.0],
            [1.0, 0.2, 0.1, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 2), (2, 4)]
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 2.0,
            },
            "_metadata": {
                "anchor_family": "density",
                "footprint_family": "medium_operational",
            },
        }
    ]
    retained = torch.tensor([True, False, False, False])

    audit = score_range_usefulness(points, boundaries, retained, queries)

    row = audit["range_query_metadata_component_summary"]["query_rows"][0]
    ship_counts = row["ship_evidence_counts"]
    assert ship_counts["full_trajectory_hit_count"] == 2
    assert ship_counts["retained_trajectory_hit_count"] == 1
    assert ship_counts["missed_trajectory_hit_count"] == 1
    assert ship_counts["missed_trajectory_hit_fraction"] == pytest.approx(0.5)
    assert ship_counts["multi_point_full_trajectory_hit_count"] == 2
    assert ship_counts["multi_point_retained_trajectory_hit_count"] == 1
    assert ship_counts["multi_point_ship_presence_recall"] == pytest.approx(0.5)

    density = audit["range_query_metadata_component_summary"]["group_by"]["anchor_family"][
        "density"
    ]
    group_counts = density["ship_evidence_counts"]
    assert group_counts["full_trajectory_hit_count_total"] == 2
    assert group_counts["retained_trajectory_hit_count_total"] == 1
    assert group_counts["missed_trajectory_hit_count_total"] == 1
    assert group_counts["ship_presence_recall"] == pytest.approx(0.5)
    assert group_counts["multi_point_ship_presence_recall"] == pytest.approx(0.5)


def test_range_usefulness_gap_coverage_penalizes_interior_gap() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.1, 1.0],
            [2.0, 0.0, 0.2, 1.0],
            [3.0, 0.0, 0.3, 1.0],
            [4.0, 0.0, 0.4, 1.0],
        ],
        dtype=torch.float32,
    )
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 5.0,
            },
        }
    ]
    endpoint_retained = torch.tensor([True, False, False, False, True])
    all_retained = torch.tensor([True, True, True, True, True])

    endpoint_audit = score_range_usefulness(points, [(0, 5)], endpoint_retained, queries)
    full_audit = score_range_usefulness(points, [(0, 5)], all_retained, queries)

    assert endpoint_audit["range_temporal_coverage"] == pytest.approx(1.0)
    assert endpoint_audit["range_gap_coverage"] == pytest.approx(0.0)
    assert endpoint_audit["range_gap_time_coverage"] == pytest.approx(0.0)
    assert endpoint_audit["range_gap_distance_coverage"] == pytest.approx(0.0)
    assert endpoint_audit["range_shape_score"] == pytest.approx(1.0)
    assert full_audit["range_gap_coverage"] == pytest.approx(1.0)
    assert full_audit["range_gap_time_coverage"] == pytest.approx(1.0)
    assert full_audit["range_gap_distance_coverage"] == pytest.approx(1.0)
    assert full_audit["range_usefulness_score"] > endpoint_audit["range_usefulness_score"]


def test_range_usefulness_gap_time_detects_irregular_sampling_gap() -> None:
    points = torch.tensor(
        [
            [0.0, 0.000, 0.0, 1.0],
            [1.0, 0.001, 0.0, 1.0],
            [2.0, 0.002, 0.0, 1.0],
            [100.0, 0.003, 0.0, 1.0],
            [101.0, 0.004, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 102.0,
            },
        }
    ]
    retained = torch.tensor([True, False, True, False, True])

    audit = score_range_usefulness(points, [(0, 5)], retained, queries)

    assert audit["range_gap_coverage"] == pytest.approx(2.0 / 3.0)
    assert audit["range_gap_time_coverage"] < 0.03
    assert audit["range_gap_distance_coverage"] == pytest.approx(0.5, abs=0.02)
    assert audit["range_gap_min_coverage"] == pytest.approx(audit["range_gap_time_coverage"])
    assert audit["range_usefulness_gap_ablation_version"] == 1
    assert audit["range_usefulness_gap_time_score"] == pytest.approx(
        audit["range_usefulness_score"]
        - 0.09 * (audit["range_gap_coverage"] - audit["range_gap_time_coverage"])
    )
    assert audit["range_usefulness_gap_distance_score"] == pytest.approx(
        audit["range_usefulness_score"]
        - 0.09 * (audit["range_gap_coverage"] - audit["range_gap_distance_coverage"])
    )
    assert audit["range_usefulness_gap_min_score"] < audit["range_usefulness_score"]


def test_range_usefulness_crossing_f1_requires_transition_brackets() -> None:
    points = torch.tensor(
        [
            [0.0, -2.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [2.0, 0.2, 0.0, 1.0],
            [3.0, 2.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
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
    inside_only = torch.tensor([False, True, True, False])
    with_brackets = torch.tensor([True, True, True, True])

    inside_audit = score_range_usefulness(points, [(0, 4)], inside_only, queries)
    bracket_audit = score_range_usefulness(points, [(0, 4)], with_brackets, queries)

    assert inside_audit["range_point_f1"] == pytest.approx(1.0)
    assert inside_audit["range_entry_exit_f1"] == pytest.approx(1.0)
    assert inside_audit["range_crossing_f1"] == pytest.approx(2.0 / 3.0)
    assert bracket_audit["range_crossing_f1"] == pytest.approx(1.0)
    assert bracket_audit["range_usefulness_score"] > inside_audit["range_usefulness_score"]


def test_range_usefulness_crossing_f1_detects_between_sample_box_crossing() -> None:
    points = torch.tensor(
        [
            [0.0, -2.0, 0.0, 1.0],
            [1.0, 2.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 2.0,
            },
        }
    ]
    drop_crossing = torch.tensor([False, False])
    keep_crossing = torch.tensor([True, True])

    dropped_audit = score_range_usefulness(points, [(0, 2)], drop_crossing, queries)
    kept_audit = score_range_usefulness(points, [(0, 2)], keep_crossing, queries)

    assert dropped_audit["range_point_f1"] == pytest.approx(1.0)
    assert dropped_audit["range_crossing_f1"] == pytest.approx(0.0)
    assert kept_audit["range_crossing_f1"] == pytest.approx(1.0)
    assert kept_audit["range_usefulness_score"] > dropped_audit["range_usefulness_score"]
