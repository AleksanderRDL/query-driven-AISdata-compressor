"""Query-driven learned segment-budget selector tests."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from learning.targets.query_local_utility import (
    QUERY_LOCAL_UTILITY_HEAD_NAMES,
)
from orchestration.length_diagnostics import (
    _max_length_required_mask,
    score_protected_length_feasibility,
    score_protected_length_frontier,
)
from orchestration.segment_audits import (
    factorized_head_probability_sources_from_logits,
    segment_oracle_allocation_audit,
    target_segment_oracle_alignment_audit,
)
from orchestration.selector_diagnostics import (
    learned_segment_frozen_method,
    neutral_segment_scores_for_ablation,
    pre_repair_frozen_method_from_trace,
    retained_decision_marginal_query_local_utility_diagnostics,
    segment_score_quantile_bands_for_ablation,
    segment_score_top_band_for_ablation,
)
from orchestration.selector_marginal_alignment import separated_marginal_teacher_targets
from orchestration.selector_teacher_vectors import (
    hybrid_marginal_teacher_selector_score_vectors,
    separated_marginal_teacher_selector_score_vectors,
)
from orchestration.selector_trace_payloads import source_masks_from_selector_trace
from scoring.geometry_thresholds import (
    FINAL_LENGTH_PRESERVATION_MIN,
)
from scoring.metrics import compute_length_preservation
from scoring.query_cache import ScoringQueryCache
from selection.learned_segment_budget import (
    blend_segment_support_scores,
    simplify_with_learned_segment_budget,
    simplify_with_learned_segment_budget_with_trace,
)
from selection.model_score_conversion import simplify_mlqds_predictions
from workloads.query_types import QUERY_TYPE_ID_RANGE

HISTORICAL_SMALL_LOCAL_FAMILY = "small_local"


def _boundaries(trajectories: list[torch.Tensor]) -> list[tuple[int, int]]:
    cursor = 0
    out = []
    for trajectory in trajectories:
        end = cursor + int(trajectory.shape[0])
        out.append((cursor, end))
        cursor = end
    return out


def test_learned_segment_budget_trace_exposes_fallback_dominance_regression() -> None:
    scores = torch.linspace(0.0, 1.0, steps=32)
    boundaries = [(0, 32)]

    retained, trace = simplify_with_learned_segment_budget_with_trace(
        scores,
        boundaries,
        compression_ratio=0.20,
    )

    assert int(retained.sum().item()) == 7
    assert trace["minimal_skeleton_slot_cap"] == 1
    assert trace["skeleton_retained_count"] == 2
    assert trace["skeleton_cap_exceeded_for_endpoint_sanity"] is True
    assert bool(retained[0].item()) is True
    assert bool(retained[-1].item()) is True
    assert trace["learned_controlled_retained_slots"] == 5
    assert trace["fallback_retained_count"] == 0


def test_learned_segment_budget_uses_geometry_gain_within_learned_budget() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.25],
            [2.0, 1.0, 0.50],
            [3.0, 0.0, 0.75],
            [4.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.ones((5,), dtype=torch.float32)
    boundaries = [(0, 5)]

    retained = simplify_with_learned_segment_budget(
        scores,
        boundaries,
        compression_ratio=0.60,
        points=points,
    )

    endpoint_only = torch.tensor([True, False, False, False, True])
    assert retained.tolist() == [True, False, True, False, True]
    assert compute_length_preservation(points, boundaries, retained) > compute_length_preservation(
        points,
        boundaries,
        endpoint_only,
    )


def test_no_geometry_tie_breaker_ablation_freezes_same_scores_without_geometry_gain() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.25],
            [2.0, 1.0, 0.50],
            [3.0, 0.0, 0.75],
            [4.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.ones((5,), dtype=torch.float32)
    boundaries = [(0, 5)]

    geometry_method = learned_segment_frozen_method(
        name="MLQDS",
        scores=scores,
        boundaries=boundaries,
        compression_ratio=0.60,
        points=points,
        learned_segment_geometry_gain_weight=1.0,
    )
    no_geometry_method = learned_segment_frozen_method(
        name="MLQDS_without_geometry_tie_breaker",
        scores=scores,
        boundaries=boundaries,
        compression_ratio=0.60,
        points=points,
        learned_segment_geometry_gain_weight=0.0,
    )

    assert geometry_method.retained_mask.tolist() == [True, False, True, False, True]
    assert no_geometry_method.retained_mask.tolist() == [True, True, False, False, True]
    assert not torch.equal(geometry_method.retained_mask, no_geometry_method.retained_mask)


def test_point_score_allocation_diagnostic_uses_point_score_segments() -> None:
    scores = torch.zeros((64,), dtype=torch.float32)
    scores[8:16] = 10.0
    scores[40:48] = 1.0
    bad_segment_scores = torch.zeros_like(scores)
    bad_segment_scores[32:64] = 10.0
    boundaries = [(0, 64)]

    point_allocation_method = learned_segment_frozen_method(
        name="MLQDS_point_score_allocation_diagnostic",
        scores=scores,
        boundaries=boundaries,
        compression_ratio=0.125,
        segment_scores=None,
        learned_segment_geometry_gain_weight=0.0,
        learned_segment_length_repair_fraction=0.0,
    )
    bad_segment_method = learned_segment_frozen_method(
        name="MLQDS_bad_segment_allocation",
        scores=scores,
        boundaries=boundaries,
        compression_ratio=0.125,
        segment_scores=bad_segment_scores,
        learned_segment_geometry_gain_weight=0.0,
        learned_segment_length_repair_fraction=0.0,
    )

    assert int(point_allocation_method.retained_mask[:32].sum().item()) > int(
        bad_segment_method.retained_mask[:32].sum().item()
    )
    assert int(point_allocation_method.retained_mask[32:].sum().item()) < int(
        bad_segment_method.retained_mask[32:].sum().item()
    )


def test_segment_allocation_authority_bands_coarsen_segment_scores() -> None:
    segment_scores = torch.zeros((16,), dtype=torch.float32)
    segment_scores[0:4] = 0.1
    segment_scores[4:8] = 0.2
    segment_scores[8:12] = 0.9
    segment_scores[12:16] = 0.8
    boundaries = [(0, 16)]

    top_half = segment_score_top_band_for_ablation(
        segment_scores,
        boundaries,
        segment_size=4,
        top_fraction=0.50,
    )
    quartiles = segment_score_quantile_bands_for_ablation(
        segment_scores,
        boundaries,
        segment_size=4,
        band_count=4,
    )

    assert top_half.tolist() == [0.0] * 8 + [1.0] * 8
    assert quartiles[0:4].unique().tolist() == [0.0]
    assert quartiles[4:8].unique().tolist() == [1.0]
    assert quartiles[8:12].unique().tolist() == [3.0]
    assert quartiles[12:16].unique().tolist() == [2.0]


def test_max_length_required_mask_keeps_required_points_and_improves_path_length() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.2],
            [2.0, 1.0, 1.0],
            [3.0, 2.0, 0.2],
            [4.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    required = torch.tensor([True, True, False, False, True])

    retained = _max_length_required_mask(points, required, keep_count=4)

    assert retained.tolist() == [True, True, True, False, True]
    assert bool(retained[1].item()) is True
    assert compute_length_preservation(points, [(0, 5)], retained) > compute_length_preservation(
        points,
        [(0, 5)],
        required,
    )


def test_score_protected_length_feasibility_reports_protected_score_upper_bound() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.2],
            [2.0, 1.0, 1.0],
            [3.0, 2.0, 0.2],
            [4.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.0, 10.0, 1.0, 0.0, 0.0], dtype=torch.float32)

    diagnostic = score_protected_length_feasibility(
        scores=scores,
        points=points,
        boundaries=[(0, 5)],
        compression_ratio=0.80,
        learned_slot_fraction_min=0.25,
    )

    assert diagnostic["available"] is True
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["retained_count"] == 4
    assert diagnostic["protected_score_point_count"] == 1
    assert diagnostic["protected_score_point_fraction_of_budget"] == pytest.approx(0.25)
    assert diagnostic["length_gate_target"] == pytest.approx(FINAL_LENGTH_PRESERVATION_MIN)
    assert diagnostic["length_preservation"] > compute_length_preservation(
        points,
        [(0, 5)],
        torch.tensor([True, True, False, False, True]),
    )


def test_score_protected_length_frontier_reports_materiality_floor() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.2],
            [2.0, 1.0, 1.0],
            [3.0, 2.0, 0.2],
            [4.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.0, 10.0, 1.0, 0.0, 0.0], dtype=torch.float32)

    frontier = score_protected_length_frontier(
        scores=scores,
        points=points,
        boundaries=[(0, 5)],
        compression_ratio=0.80,
        learned_slot_fraction_min=0.25,
        protected_fractions=(0.0, 0.25, 0.50),
    )

    assert frontier["available"] is True
    assert frontier["diagnostic_only"] is True
    assert frontier["learned_slot_fraction_min"] == pytest.approx(0.25)
    assert frontier["length_gate_target"] == pytest.approx(FINAL_LENGTH_PRESERVATION_MIN)
    assert len(frontier["rows"]) == 3
    assert frontier["materiality_floor_length_preservation"] == pytest.approx(
        frontier["rows"][1]["length_preservation"]
    )
    assert (
        frontier["materiality_floor_length_gate_would_pass"]
        == frontier["rows"][1]["length_gate_would_pass"]
    )
    assert frontier["rows"][0]["protected_score_point_count"] == 0
    assert frontier["rows"][1]["protected_score_point_count"] == 1


def test_learned_segment_budget_trace_reports_geometry_diagnostics_without_changing_mask() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.25],
            [2.0, 1.0, 0.50],
            [3.0, 0.0, 0.75],
            [4.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.ones((5,), dtype=torch.float32)
    boundaries = [(0, 5)]

    retained, trace = simplify_with_learned_segment_budget_with_trace(
        scores,
        boundaries,
        compression_ratio=0.60,
        points=points,
    )
    without_trace = simplify_with_learned_segment_budget(
        scores,
        boundaries,
        compression_ratio=0.60,
        points=points,
    )
    endpoint_only = torch.tensor([True, False, False, False, True])
    geometry = trace["geometry_diagnostics"]

    assert torch.equal(retained, without_trace)
    assert geometry["available"] is True
    assert geometry["trajectory_count"] == 1
    assert geometry["retained_length_preservation"] == pytest.approx(
        compute_length_preservation(points, boundaries, retained)
    )
    assert geometry["skeleton_length_preservation"] == pytest.approx(
        compute_length_preservation(points, boundaries, endpoint_only)
    )
    assert geometry["learned_length_gain_over_skeleton"] > 0.0
    assert geometry["trajectory_length_preservation_gate_target"] == pytest.approx(
        FINAL_LENGTH_PRESERVATION_MIN
    )
    assert geometry["trajectory_length_preservation_below_gate_count"] in {0, 1}
    assert geometry["worst_trajectories"][0]["trajectory_id"] == 0


def test_learned_segment_budget_trace_separates_allocation_from_point_selection() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.2],
            [2.0, 0.0, 0.4],
            [3.0, 1.5, 0.6],
            [4.0, -1.5, 0.8],
            [5.0, 0.0, 1.0],
            [6.0, 0.0, 1.2],
            [7.0, 0.0, 1.4],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.0, 10.0, 9.0, 0.1, 0.1, 8.0, 7.0, 0.0], dtype=torch.float32)
    boundaries = [(0, 8)]

    retained, trace = simplify_with_learned_segment_budget_with_trace(
        scores,
        boundaries,
        compression_ratio=0.625,
        segment_size=8,
        points=points,
        geometry_gain_weight=0.0,
    )
    without_trace = simplify_with_learned_segment_budget(
        scores,
        boundaries,
        compression_ratio=0.625,
        segment_size=8,
        points=points,
        geometry_gain_weight=0.0,
    )
    diagnostic = trace["allocation_point_selection_diagnostics"]

    assert torch.equal(retained, without_trace)
    assert diagnostic["available"] is True
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["primary_retained_stage"] == "pre_length_repair"
    assert diagnostic["primary_length_preservation"] == pytest.approx(
        compute_length_preservation(points, boundaries, retained)
    )
    assert (
        diagnostic["same_allocation_length_only_point_selection_preservation"]
        > (diagnostic["primary_length_preservation"])
    )
    assert diagnostic["counterfactual_retained_count"] == diagnostic["total_budget_count"]


def test_learned_segment_budget_length_repair_swaps_learned_slots_for_path_gain() -> None:
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

    retained, trace = simplify_with_learned_segment_budget_with_trace(
        scores,
        boundaries,
        compression_ratio=0.20,
        points=points,
        geometry_gain_weight=0.0,
        length_repair_fraction=0.0,
    )
    repaired, repaired_trace = simplify_with_learned_segment_budget_with_trace(
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


def test_segment_support_score_blend_uses_length_head_at_full_weight() -> None:
    segment_scores = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    path_scores = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)

    half = blend_segment_support_scores(
        segment_scores=segment_scores,
        path_length_support_scores=path_scores,
        path_length_support_weight=0.5,
    )
    full = blend_segment_support_scores(
        segment_scores=segment_scores,
        path_length_support_scores=path_scores,
        path_length_support_weight=1.0,
    )

    assert torch.allclose(cast(torch.Tensor, half), torch.tensor([1.5, 2.5, 3.5]))
    assert torch.allclose(cast(torch.Tensor, full), path_scores)


def test_learned_segment_budget_trace_accepts_explicit_segment_score_source_label() -> None:
    scores = torch.linspace(0.0, 1.0, steps=16)
    segment_scores = torch.linspace(1.0, 0.0, steps=16)

    _retained, trace = simplify_with_learned_segment_budget_with_trace(
        scores,
        [(0, 16)],
        compression_ratio=0.25,
        segment_scores=segment_scores,
        segment_score_source_label="path_length_support_head_top20_mean",
    )

    assert trace["segment_score_source"] == "path_length_support_head_top20_mean"


def test_learned_segment_budget_geometry_gain_uses_trajectory_retained_anchors() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 5.0, 2.0],
            [3.0, 0.0, 3.0],
            [4.0, 0.0, 4.0],
            [5.0, 0.0, 5.0],
            [6.0, 0.0, 6.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.ones((7,), dtype=torch.float32)
    segment_scores = torch.zeros((7,), dtype=torch.float32)
    segment_scores[2:4] = 10.0
    boundaries = [(0, 7)]

    retained, trace = simplify_with_learned_segment_budget_with_trace(
        scores,
        boundaries,
        compression_ratio=0.40,
        segment_size=2,
        segment_scores=segment_scores,
        points=points,
    )

    assert retained.tolist() == [True, False, True, False, False, False, True]
    assert trace["learned_controlled_retained_slots"] == 1
    assert trace["fallback_retained_count"] == 0


def test_no_segment_budget_head_ablation_uses_neutral_segment_scores() -> None:
    scores = torch.linspace(0.0, 1.0, steps=32)
    scores[27] = 10.0
    boundaries = [(0, 32)]
    learned_segment_scores = torch.zeros_like(scores)
    learned_segment_scores[24:32] = 5.0

    neutral_segment_scores = neutral_segment_scores_for_ablation(learned_segment_scores)
    learned_retained, learned_trace = simplify_with_learned_segment_budget_with_trace(
        scores,
        boundaries,
        compression_ratio=0.15,
        segment_size=8,
        segment_scores=learned_segment_scores,
    )
    ablated_retained, ablated_trace = simplify_with_learned_segment_budget_with_trace(
        scores,
        boundaries,
        compression_ratio=0.15,
        segment_size=8,
        segment_scores=neutral_segment_scores,
    )

    assert torch.count_nonzero(neutral_segment_scores).item() == 0
    assert learned_trace["segment_score_source"] == "segment_budget_head_top20_mean"
    assert ablated_trace["segment_score_source"] == "segment_budget_head_top20_mean"
    assert bool(learned_retained[27].item()) is True
    assert bool(ablated_retained[27].item()) is False
    assert not torch.equal(learned_retained, ablated_retained)


def test_learned_segment_budget_can_split_allocation_and_point_segment_scores() -> None:
    scores = torch.zeros((8,), dtype=torch.float32)
    allocation_scores = torch.zeros_like(scores)
    allocation_scores[4:8] = 10.0
    point_segment_scores = torch.zeros_like(scores)
    point_segment_scores[5] = 10.0

    retained, trace = simplify_with_learned_segment_budget_with_trace(
        scores,
        [(0, 8)],
        compression_ratio=0.375,
        segment_size=4,
        segment_scores=allocation_scores,
        segment_point_scores=point_segment_scores,
        geometry_gain_weight=0.0,
        segment_score_point_blend_weight=1.0,
        min_temporal_spacing_fraction_within_segment=0.0,
    )

    assert retained.tolist() == [True, False, False, False, False, True, False, True]
    assert trace["segment_budget_allocation_count"] == 1
    assert trace["learned_controlled_retained_slots"] == 1
    assert trace["fallback_retained_count"] == 0


def test_learned_segment_budget_transfer_calibration_is_guarded_non_default() -> None:
    steps = torch.arange(0, 16, dtype=torch.float32)
    points = torch.stack(
        [
            steps,
            torch.where((steps.long() % 2) == 0, torch.zeros_like(steps), torch.ones_like(steps)),
            steps * 0.10,
        ],
        dim=1,
    )
    scores = torch.zeros((16,), dtype=torch.float32)
    segment_scores = torch.tensor(
        [0.0, 0.1, 0.2, 0.1, 0.8, 0.9, 1.0, 0.8, 0.2, 0.1, 0.0, 0.1, 0.5, 0.4, 0.3, 0.2],
        dtype=torch.float32,
    )

    _default_retained, default_trace = simplify_with_learned_segment_budget_with_trace(
        scores,
        [(0, 16)],
        compression_ratio=0.50,
        segment_size=4,
        segment_scores=segment_scores,
        points=points,
        segment_length_support_weight=0.50,
        geometry_gain_weight=0.0,
    )
    _calibrated_retained, calibrated_trace = simplify_with_learned_segment_budget_with_trace(
        scores,
        [(0, 16)],
        compression_ratio=0.50,
        segment_size=4,
        segment_scores=segment_scores,
        points=points,
        segment_length_support_weight=0.50,
        segment_transfer_calibration_mode="segment_score_allocation_weight_zblend",
        geometry_gain_weight=0.0,
    )

    assert default_trace["segment_transfer_calibration"]["mode"] == "none"
    assert default_trace["segment_transfer_calibration"]["applied"] is False
    assert default_trace["segment_length_support_weight"] == pytest.approx(0.50)
    calibration = calibrated_trace["segment_transfer_calibration"]
    assert calibration["mode"] == "segment_score_allocation_weight_zblend"
    assert calibration["applied"] is True
    assert calibration["uses_post_selection_attribution"] is False
    assert calibration["uses_length_support_counter_signal"] is False
    assert calibration["base_segment_length_support_weight"] == pytest.approx(0.50)
    assert calibration["effective_segment_length_support_weight"] == pytest.approx(0.0)
    assert calibrated_trace["segment_length_support_weight"] == pytest.approx(0.0)
    assert (
        calibrated_trace["segment_score_source"]
        == "segment_budget_head_top20_mean+segment_score_allocation_weight_zblend"
    )
    rows = calibrated_trace["segment_source_attribution"]["rows"]
    assert rows
    assert {row["segment_transfer_calibration_mode"] for row in rows} == {
        "segment_score_allocation_weight_zblend"
    }
    assert any(
        row["segment_pre_transfer_calibration_score"] != row["segment_score"] for row in rows
    )


def test_mlqds_scoring_passes_segment_point_scores_to_learned_selector() -> None:
    predictions = torch.zeros((64,), dtype=torch.float32)
    allocation_scores = torch.zeros_like(predictions)
    allocation_scores[32:64] = 10.0
    point_segment_scores = torch.zeros_like(predictions)
    point_segment_scores[40] = 10.0

    retained = simplify_mlqds_predictions(
        predictions,
        [(0, 64)],
        workload_type="range",
        compression_ratio=0.0625,
        temporal_fraction=0.0,
        diversity_bonus=0.0,
        selector_type="learned_segment_budget",
        score_mode="raw",
        segment_scores=allocation_scores,
        segment_point_scores=point_segment_scores,
        learned_segment_geometry_gain_weight=0.0,
        learned_segment_score_blend_weight=1.0,
        learned_segment_length_repair_fraction=0.0,
    )

    assert bool(retained[40].item()) is True
    assert int(retained[:32].sum().item()) == 1
    assert int(retained[32:].sum().item()) == 3


def test_segment_oracle_allocation_audit_reports_ranking_alignment_after_freeze() -> None:
    point_scores = torch.tensor([0.9, 0.8, 0.1, 0.0, 0.1, 0.0, 0.7, 0.6], dtype=torch.float32)
    segment_scores = torch.tensor([0.9, 0.8, 0.1, 0.0, 0.95, 0.9, 0.2, 0.1], dtype=torch.float32)
    selector_scores = torch.tensor([0.9, 0.8, 0.1, 0.0, 0.95, 0.9, 0.2, 0.1], dtype=torch.float32)
    head_logits = torch.zeros((8, len(QUERY_LOCAL_UTILITY_HEAD_NAMES)), dtype=torch.float32)
    query_hit_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("query_hit_probability")
    clamped_point_scores = point_scores.clamp(1e-4, 1.0 - 1e-4)
    head_logits[:, query_hit_idx] = torch.logit(clamped_point_scores)
    labels = torch.zeros((8, 4), dtype=torch.float32)
    labels[0:2, QUERY_TYPE_ID_RANGE] = 1.0
    labels[6:8, QUERY_TYPE_ID_RANGE] = 0.5
    retained_mask = torch.tensor([True, False, False, False, False, False, True, False])
    head_sources = factorized_head_probability_sources_from_logits(head_logits)

    audit = segment_oracle_allocation_audit(
        point_scores=point_scores,
        segment_budget_scores=segment_scores,
        selector_segment_scores=selector_scores,
        eval_labels=labels,
        boundaries=[(0, 8)],
        workload_type="range",
        head_scores_by_name=head_sources,
        retained_mask=retained_mask,
        segment_size=2,
        paired_row_limit=2,
    )

    assert audit["available"] is True
    assert audit["diagnostic_only"] is True
    assert audit["uses_eval_labels_after_mask_freeze"] is True
    alignment = audit["source_alignment"]
    assert alignment["segment_budget_head_top20_mean"]["spearman_vs_oracle_mass"] < 1.0
    assert alignment["point_score_top20_mean"]["spearman_vs_oracle_mass"] == pytest.approx(1.0)
    assert alignment["head_query_hit_probability_sigmoid_top20_mean"][
        "spearman_vs_oracle_mass"
    ] == pytest.approx(1.0)
    assert audit["best_source_by_top25_oracle_mass_recall"] == "point_score_top20_mean"
    assert "head_segment_budget_target_sigmoid_top20_mean" in audit["score_source_names"]
    transfer_rows = audit["paired_segment_transfer_rows"]
    assert transfer_rows["available"] is True
    assert transfer_rows["row_limit_per_source"] == 2
    assert transfer_rows["retained_segment_summary"]["available"] is True
    assert transfer_rows["retained_segment_summary"]["frozen_primary_retained_count_total"] == 2
    assert (
        transfer_rows["retained_segment_summary"]["segments_with_any_frozen_primary_retained_point"]
        == 2
    )
    first_row = transfer_rows["rows"][0]
    assert {
        "segment_index",
        "trajectory_id",
        "oracle_mass",
        "oracle_mass_rank",
        "point_score_top20_mean_score",
        "point_score_top20_mean_rank",
        "segment_budget_head_top20_mean_score",
        "segment_budget_head_top20_mean_rank",
        "head_query_hit_probability_sigmoid_top20_mean_score",
        "head_query_hit_probability_sigmoid_top20_mean_rank",
        "frozen_primary_retained_count",
        "frozen_primary_retained_count_rank",
    }.issubset(first_row)
    assert first_row["oracle_mass_rank"] == 1
    all_rows = audit["all_segment_transfer_rows"]
    assert all_rows["available"] is True
    assert all_rows["diagnostic_only"] is True
    assert all_rows["uses_eval_labels_after_mask_freeze"] is True
    assert all_rows["row_scope"] == "all_segments"
    assert all_rows["row_count"] == 4
    assert len(all_rows["rows"]) == 4
    assert all_rows["rows"][0]["segment_index"] == 0
    assert all_rows["rows"][0]["oracle_mass_rank"] == 1
    assert all_rows["rows"][0]["canonical_order_rank"] == 1
    assert all_rows["rows"][0]["neutral_allocation_order_rank"] == 1
    assert all_rows["rows"][3]["segment_index"] == 3
    assert all_rows["rows"][3]["oracle_mass_rank"] == 2


def test_target_segment_oracle_alignment_audit_reports_eval_target_sources_after_freeze() -> None:
    points = torch.zeros((8, 8), dtype=torch.float32)
    points[:, 0] = torch.arange(8, dtype=torch.float32)
    points[:, 1] = torch.linspace(0.0, 0.7, steps=8)
    points[:, 2] = torch.linspace(0.0, 0.7, steps=8)
    points[:, 7] = 1.0
    query = {
        "type": "range",
        "params": {
            "t_start": -0.5,
            "t_end": 3.5,
            "lat_min": -0.1,
            "lat_max": 0.35,
            "lon_min": -0.1,
            "lon_max": 0.35,
        },
    }
    labels = torch.zeros((8, 4), dtype=torch.float32)
    labels[0:4, QUERY_TYPE_ID_RANGE] = 1.0
    retained_mask = torch.tensor([True, False, True, False, False, False, False, True])

    audit = target_segment_oracle_alignment_audit(
        points=points,
        boundaries=[(0, 8)],
        typed_queries=[query],
        eval_labels=labels,
        workload_type="range",
        retained_mask=retained_mask,
        segment_size=2,
        paired_row_limit=2,
    )

    assert audit["available"] is True
    assert audit["diagnostic_only"] is True
    assert audit["uses_eval_labels_after_mask_freeze"] is True
    assert audit["target_alignment_attempted"] is True
    assert (
        audit["source_semantics"]["point_score_top20_mean"]
        == "eval_query_local_utility_final_target_top20_mean"
    )
    assert (
        audit["source_semantics"]["target_head_segment_budget_target_top20_mean"]
        == "eval_query_local_utility_factorized_target_head:segment_budget_target"
    )
    assert "target_head_query_hit_probability_top20_mean" in audit["score_source_names"]
    assert "target_head_segment_budget_target_top20_mean" in audit["source_alignment"]
    rows = audit["all_segment_transfer_rows"]["rows"]
    assert len(rows) == 4
    assert "target_head_query_hit_probability_top20_mean_rank" in rows[0]
    assert (
        audit["target_diagnostics_summary"]["segment_budget_target_base_source"]
        == "query_local_utility_final_score"
    )


def test_learned_segment_allocation_guarantees_one_slot_per_trajectory_when_possible() -> None:
    scores = torch.ones((24,), dtype=torch.float32)
    # Favor trajectory 0 strongly in segment scores and keep trajectory 1 low.
    segment_scores = torch.zeros((24,), dtype=torch.float32)
    segment_scores[0:12] = 10.0
    segment_scores[12:] = 0.1

    boundaries = [(0, 12), (12, 24)]
    retained = simplify_with_learned_segment_budget(
        scores,
        boundaries,
        compression_ratio=0.30,
        segment_size=4,
        segment_scores=segment_scores,
    )
    _, trace = simplify_with_learned_segment_budget_with_trace(
        scores,
        boundaries,
        compression_ratio=0.30,
        segment_size=4,
        segment_scores=segment_scores,
    )

    learned_counts = trace["trajectory_learned_decision_counts"]
    assert len(learned_counts) == 2
    assert int(learned_counts[0]) >= 1
    assert int(learned_counts[1]) >= 1
    assert bool(retained[0].item()) is True
    assert bool(retained[11].item()) is True
    assert bool(retained[12].item()) is True
    assert bool(retained[23].item()) is True
    assert trace["trajectories_with_at_least_one_learned_decision"] >= 2


def test_learned_segment_trace_reports_query_free_segment_source_attribution() -> None:
    scores = torch.linspace(0.0, 1.0, steps=24, dtype=torch.float32)
    segment_scores = torch.zeros((24,), dtype=torch.float32)
    segment_scores[8:12] = 5.0
    segment_scores[20:24] = 4.0

    retained, trace = simplify_with_learned_segment_budget_with_trace(
        scores,
        [(0, 12), (12, 24)],
        compression_ratio=0.30,
        segment_size=4,
        segment_scores=segment_scores,
    )

    attribution = trace["segment_source_attribution"]
    pre_repair = trace["pre_repair_segment_source_attribution"]
    assert attribution["available"] is True
    assert attribution["diagnostic_only"] is True
    assert attribution["query_free"] is True
    assert pre_repair["available"] is True
    assert pre_repair["diagnostic_only"] is True
    assert pre_repair["query_free"] is True
    assert attribution["segment_count"] == trace["segments_considered_count"]
    summary = attribution["summary"]
    assert summary["retained_count_total"] == int(retained.sum().item())
    assert summary["skeleton_count_total"] == trace["skeleton_retained_count"]
    assert summary["learned_count_total"] == trace["learned_controlled_retained_slots"]
    assert summary["fallback_count_total"] == trace["fallback_retained_count"]
    assert summary["length_repair_count_total"] == trace["length_repair_retained_count"]
    assert summary["segment_allocation_count_total"] == trace["segment_budget_allocation_count"]
    retained_payload = trace["retained_mask"]
    assert retained_payload["available"] is True
    assert retained_payload["diagnostic_only"] is True
    assert retained_payload["query_free"] is True
    assert retained_payload["retained_count"] == int(retained.sum().item())
    assert retained_payload["indices"] == torch.where(retained)[0].tolist()
    assert trace["skeleton_retained_mask"]["retained_count"] == trace["skeleton_retained_count"]
    assert (
        trace["learned_retained_mask"]["retained_count"]
        == trace["learned_controlled_retained_slots"]
    )
    assert trace["fallback_retained_mask"]["retained_count"] == trace["fallback_retained_count"]
    assert (
        trace["length_repair_retained_mask"]["retained_count"]
        == trace["length_repair_retained_count"]
    )
    first_row = attribution["rows"][0]
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


def test_learned_segment_trace_reports_pre_repair_source_attribution() -> None:
    scores = torch.zeros((32,), dtype=torch.float32)
    scores[8:24] = torch.linspace(1.0, 2.0, steps=16, dtype=torch.float32)
    points = torch.zeros((32, 5), dtype=torch.float32)
    points[:, 0] = torch.arange(32, dtype=torch.float32)
    points[:, 1] = torch.linspace(0.0, 0.1, steps=32)
    points[:, 2] = torch.sin(torch.linspace(0.0, 12.56, steps=32)) * 0.05

    _, trace = simplify_with_learned_segment_budget_with_trace(
        scores,
        [(0, 32)],
        compression_ratio=0.25,
        segment_size=4,
        points=points,
        geometry_gain_weight=0.0,
        length_repair_fraction=1.0,
    )

    pre_summary = trace["pre_repair_segment_source_attribution"]["summary"]
    final_summary = trace["segment_source_attribution"]["summary"]
    pre_mask_payload = trace["pre_repair_retained_mask"]
    assert trace["length_repair_swap_count"] > 0
    assert pre_summary["retained_count_total"] == final_summary["retained_count_total"]
    assert pre_summary["length_repair_count_total"] == 0
    assert final_summary["length_repair_count_total"] == trace["length_repair_retained_count"]
    assert pre_summary["learned_count_total"] == trace["segment_budget_allocation_count"]
    assert final_summary["learned_count_total"] < pre_summary["learned_count_total"]
    assert pre_mask_payload["available"] is True
    assert pre_mask_payload["diagnostic_only"] is True
    assert pre_mask_payload["query_free"] is True
    assert pre_mask_payload["retained_count"] == pre_summary["retained_count_total"]
    assert pre_mask_payload["indices"] == sorted(set(pre_mask_payload["indices"]))
    assert trace["retained_mask"]["retained_count"] == final_summary["retained_count_total"]
    assert (
        trace["length_repair_retained_mask"]["retained_count"]
        == final_summary["length_repair_count_total"]
    )

    pre_repair_method = pre_repair_frozen_method_from_trace(
        name="MLQDS_pre_repair_allocation_diagnostic",
        selector_trace=trace,
        point_count=int(scores.numel()),
    )
    assert pre_repair_method.retained_mask.dtype == torch.bool
    assert int(pre_repair_method.retained_mask.sum().item()) == pre_summary["retained_count_total"]
    assert torch.equal(
        torch.where(pre_repair_method.retained_mask)[0],
        torch.tensor(pre_mask_payload["indices"], dtype=torch.long),
    )


def test_source_masks_from_selector_trace_reads_schema7_source_payloads() -> None:
    trace = {
        "skeleton_retained_mask": {
            "available": True,
            "retained_count": 2,
            "indices": [0, 4],
        },
        "learned_retained_mask": {
            "available": True,
            "retained_count": 1,
            "indices": [2],
        },
        "fallback_retained_mask": {
            "available": True,
            "retained_count": 0,
            "indices": [],
        },
        "length_repair_retained_mask": {
            "available": True,
            "retained_count": 1,
            "indices": [3],
        },
    }

    masks = source_masks_from_selector_trace(trace, point_count=5)

    assert torch.equal(masks["skeleton"], torch.tensor([True, False, False, False, True]))
    assert torch.equal(masks["learned"], torch.tensor([False, False, True, False, False]))
    assert torch.equal(masks["fallback"], torch.zeros((5,), dtype=torch.bool))
    assert torch.equal(masks["length_repair"], torch.tensor([False, False, False, True, False]))


def test_retained_decision_marginal_query_local_utility_diagnostic_scores_true_marginals() -> None:
    points = torch.zeros((5, 5), dtype=torch.float32)
    points[:, 0] = torch.arange(5, dtype=torch.float32)
    points[:, 1] = torch.linspace(0.0, 4.0, steps=5)
    points[:, 2] = torch.linspace(0.0, 4.0, steps=5)
    points[:, 3] = torch.tensor([0.0, 0.1, 1.0, 0.9, 0.0])
    points[:, 4] = torch.tensor([0.0, 5.0, 90.0, 95.0, 100.0])
    retained = torch.tensor([True, False, True, False, True])
    selector_scores = torch.tensor([0.0, 0.1, 0.95, 0.90, 0.0])
    raw_scores = torch.tensor([0.0, 0.2, 1.20, 1.00, 0.0])
    segment_scores = torch.tensor([0.0, 0.0, 2.0, 1.8, 0.0])
    query = {
        "type": "range",
        "params": {
            "t_start": 1.5,
            "t_end": 3.5,
            "lat_min": 1.5,
            "lat_max": 3.5,
            "lon_min": 1.5,
            "lon_max": 3.5,
        },
    }
    source_masks = {
        "skeleton": torch.tensor([True, False, False, False, True]),
        "learned": torch.tensor([False, False, True, False, False]),
    }

    diagnostics = retained_decision_marginal_query_local_utility_diagnostics(
        points=points,
        boundaries=[(0, 5)],
        typed_queries=[query],
        primary_retained_mask=retained,
        raw_scores=raw_scores,
        selector_scores=selector_scores,
        segment_scores=segment_scores,
        score_component_vectors={
            "factorized_composed_score": torch.tensor([0.0, 0.2, 0.85, 0.75, 0.0]),
            "head_logit_query_hit_probability": torch.tensor([-8.0, -2.0, 2.0, 1.5, -8.0]),
            "head_probability_query_hit_probability": torch.tensor([0.0, 0.1, 0.9, 0.8, 0.0]),
        },
        sampled_prior_vectors={
            "spatial_query_hit_probability": torch.tensor([0.0, 0.1, 0.7, 0.8, 0.0]),
            "route_density_prior": torch.tensor([0.0, 0.0, 0.2, 0.1, 0.0]),
        },
        query_free_teacher_proxy_vectors={
            "query_free_constant_proxy": torch.zeros((5,), dtype=torch.float32),
            "query_free_path_length_support_target": torch.tensor([0.0, 0.1, 0.8, 0.9, 0.0]),
            "query_free_endpoint_support": torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0]),
        },
        model_prior_vectors={
            "spatial_query_hit_probability": torch.tensor([0.0, 0.1, 0.7, 0.8, 0.0]),
            "route_density_prior": torch.zeros((5,), dtype=torch.float32),
        },
        source_masks=source_masks,
        selector_trace={
            "pre_repair_retained_mask": {"available": True, "indices": [0, 2]},
            "length_repair_retained_mask": {"available": True, "indices": [4]},
            "retained_mask": {"available": True, "indices": [0, 2, 4]},
            "segment_source_attribution": {
                "available": True,
                "rows": [
                    {
                        "segment_index": 7,
                        "allocation_order_index": 3,
                        "trajectory_id": 0,
                        "start": 1,
                        "end": 4,
                        "length": 3,
                        "segment_score": 2.5,
                        "segment_score_rank": 1,
                        "segment_score_source": "segment_budget_head_top20_mean",
                        "segment_length_support_score": 0.75,
                        "segment_length_support_rank": 2,
                        "segment_allocation_weight": 1.25,
                        "segment_allocation_weight_rank": 1,
                        "segment_allocation_count": 2,
                        "retained_count": 1,
                        "retained_fraction": 1.0 / 3.0,
                        "skeleton_count": 0,
                        "learned_count": 1,
                        "fallback_count": 0,
                        "length_repair_count": 0,
                        "unattributed_count": 0,
                    }
                ],
            },
        },
        max_retained_per_source=8,
        max_removed_candidates=4,
    )

    rows = diagnostics["rows"]
    learned_removal = next(
        row
        for row in rows
        if row["source"] == "learned" and row["decision"] == "retained_removal_loss"
    )
    removed_addition = next(
        row
        for row in rows
        if row["point_index"] == 3 and row["decision"] == "removed_addition_gain"
    )
    assert diagnostics["available"] is True
    assert diagnostics["diagnostic_only"] is True
    assert diagnostics["exact_query_local_utility_marginals"] is True
    assert diagnostics["performance_mode"] == "exact_cached_query_support"
    assert diagnostics["query_cache_created"] is True
    assert diagnostics["query_cache_provided"] is False
    assert diagnostics["query_cache_range_audit_support_count"] == 1
    assert diagnostics["elapsed_seconds"] >= 0.0
    assert diagnostics["score_fields_available"] == {
        "raw_score": True,
        "selector_score": True,
        "segment_score": True,
    }
    assert diagnostics["score_component_fields_available"] == {
        "factorized_composed_score": True,
        "head_logit_query_hit_probability": True,
        "head_probability_query_hit_probability": True,
    }
    assert diagnostics["context_fields_available"]["sampled_prior_channels"] == {
        "route_density_prior": True,
        "spatial_query_hit_probability": True,
    }
    assert diagnostics["context_fields_available"]["model_prior_channels"] == {
        "route_density_prior": True,
        "spatial_query_hit_probability": True,
    }
    assert diagnostics["context_fields_available"]["query_free_teacher_proxies"] == {
        "query_free_constant_proxy": True,
        "query_free_endpoint_support": True,
        "query_free_path_length_support_target": True,
    }
    assert diagnostics["context_fields_available"]["selector_stage_state"] == {
        "final_retained": True,
        "length_repair_retained": True,
        "pre_repair_retained": True,
    }
    assert diagnostics["context_fields_available"]["selector_segment_context"] is True
    assert diagnostics["top_marginal_miss_summary"]["available"] is True
    assert (
        diagnostics["top_marginal_miss_summary"]["top_marginal_rows_in_selector_trace_only"] is True
    )
    assert diagnostics["top_marginal_miss_diagnostics"]["top_marginal_rows"]
    assert learned_removal["score_components"][
        "head_probability_query_hit_probability"
    ] == pytest.approx(0.9)
    assert learned_removal["head_probabilities"]["query_hit_probability"] == pytest.approx(0.9)
    assert learned_removal["head_logits"]["query_hit_probability"] == pytest.approx(2.0)
    assert learned_removal["sampled_prior_channels"]["spatial_query_hit_probability"] == (
        pytest.approx(0.7)
    )
    assert learned_removal["model_prior_channels"]["spatial_query_hit_probability"] == (
        pytest.approx(0.7)
    )
    assert learned_removal["query_free_teacher_proxies"][
        "query_free_path_length_support_target"
    ] == pytest.approx(0.8)
    assert learned_removal["query_local_utility_score_components"]["factorized_composed_score"] == (
        pytest.approx(0.85)
    )
    assert learned_removal["selector_stage_state"]["pre_repair_retained"] is True
    assert learned_removal["selector_stage_state"]["final_retained"] is True
    assert learned_removal["selector_segment_context"] == {
        "source": "segment_source_attribution",
        "segment_index": 7,
        "allocation_order_index": 3,
        "trajectory_index": 0,
        "segment_start": 1,
        "segment_end": 4,
        "segment_length": 3,
        "point_offset_in_segment": 1,
        "point_fraction_in_segment": pytest.approx(0.5),
        "segment_score": pytest.approx(2.5),
        "segment_score_rank": 1,
        "segment_score_source": "segment_budget_head_top20_mean",
        "segment_length_support_score": pytest.approx(0.75),
        "segment_length_support_rank": 2,
        "segment_allocation_weight": pytest.approx(1.25),
        "segment_allocation_weight_rank": 1,
        "segment_allocation_count": 2,
        "retained_count": 1,
        "retained_fraction": pytest.approx(1.0 / 3.0),
        "skeleton_count": 0,
        "learned_count": 1,
        "fallback_count": 0,
        "length_repair_count": 0,
        "unattributed_count": 0,
    }
    assert learned_removal["trajectory_index"] == 0
    assert "marginal_query_local_utility_candidate_rank" in learned_removal
    assert "query_local_utility_component_minus_marginal_rank" in learned_removal
    assert "query_free_teacher_proxy_minus_marginal_rank" in learned_removal
    assert "failure_buckets" in learned_removal
    guard_summary = diagnostics["query_free_teacher_proxy_guard_coupling_summary"]
    assert guard_summary["available"] is True
    assert guard_summary["primary_proxy"] == "query_free_endpoint_support"
    assert guard_summary["retained_removal_count"] == 4
    assert guard_summary["learned_controllable_retained_removal_count"] == 1
    assert guard_summary["guard_owned_retained_removal_count"] == 3
    assert (
        guard_summary["subsets"]["learned_controllable_retained_removal"][
            "query_free_teacher_proxy_alignment"
        ]["query_free_path_length_support_target"]["available"]
        is False
    )
    teacher_summary = diagnostics["learned_controllable_marginal_teacher_summary"]
    assert teacher_summary["available"] is True
    assert (
        teacher_summary["teacher_signal"] == "exact_retained_removal_marginal_query_local_utility"
    )
    assert teacher_summary["teacher_scope"] == "learned_controllable_retained_removal"
    assert teacher_summary["learned_controllable_retained_removal_count"] == 1
    assert teacher_summary["candidate_for_train_side_calibration"] is False
    assert teacher_summary["eval_time_feature_allowed"] is False
    separated_teacher = diagnostics["separated_marginal_teacher_summary"]
    assert separated_teacher["available"] is True
    assert separated_teacher["teacher_shape"] == (
        "separated_segment_and_within_segment_point_targets"
    )
    assert separated_teacher["eval_time_feature_allowed"] is False
    assert separated_teacher["learned_controllable_retained_removal_count"] == 1
    assert separated_teacher["rows_with_selector_segment_context"] == 1
    assert separated_teacher["segment_target_count"] == 1
    assert separated_teacher["point_target_count"] == 1
    assert separated_teacher["teacher_usage_split"] == "unknown"
    assert separated_teacher["teacher_usage_allowed_for_train_or_checkpoint"] is False
    assert separated_teacher["teacher_target_shape_viable"] is False
    assert separated_teacher["candidate_for_train_side_teacher"] is False
    assert (
        separated_teacher["candidate_for_train_side_teacher_reason"]
        == "insufficient_teacher_target_shape"
    )
    assert separated_teacher["segment_target_rows"][0]["segment_index"] == 7
    assert separated_teacher["segment_target_rows"][0]["segment_target"] == pytest.approx(1.0)
    assert separated_teacher["point_target_rows"][0]["point_index"] == 2
    assert separated_teacher["point_target_rows"][0]["point_target_within_segment"] == (
        pytest.approx(1.0)
    )
    assert learned_removal["marginal_query_local_utility"] > 0.0
    assert removed_addition["marginal_query_local_utility"] > 0.0
    assert diagnostics["by_source"]["learned"]["mean_marginal_query_local_utility"] > 0.0
    assert (
        diagnostics["by_decision"]["removed_addition_gain"]["selector_score"]["available"] is True
    )
    assert (
        diagnostics["overall"]["score_component_alignment"][
            "head_probability_query_hit_probability"
        ]["available"]
        is True
    )
    assert (
        diagnostics["overall"]["query_free_teacher_proxy_alignment"][
            "query_free_path_length_support_target"
        ]["available"]
        is True
    )
    constant_proxy = diagnostics["overall"]["query_free_teacher_proxy_alignment"][
        "query_free_constant_proxy"
    ]
    assert constant_proxy["available"] is False
    assert constant_proxy["reason"] == "no_value_variation"
    assert "top_minus_bottom_marginal" not in constant_proxy


def test_separated_marginal_teacher_targets_separate_shape_from_split_eligibility() -> None:
    def learned_row(
        *,
        point_index: int,
        marginal: float,
        segment_index: int,
    ) -> dict[str, Any]:
        return {
            "point_index": point_index,
            "trajectory_index": 0,
            "decision": "retained_removal_loss",
            "source": "learned",
            "marginal_query_local_utility": marginal,
            "selector_segment_context": {
                "trajectory_index": 0,
                "segment_index": segment_index,
                "segment_start": segment_index * 8,
                "segment_end": segment_index * 8 + 8,
                "segment_length": 8,
                "point_offset_in_segment": point_index % 8,
                "segment_score_rank": segment_index + 1,
                "segment_allocation_count": 2,
            },
        }

    rows = [
        learned_row(point_index=2, marginal=0.40, segment_index=0),
        learned_row(point_index=3, marginal=0.10, segment_index=0),
        learned_row(point_index=11, marginal=0.25, segment_index=1),
    ]

    eval_summary = separated_marginal_teacher_targets(
        rows,
        teacher_usage_split="eval_primary",
    )
    assert eval_summary["available"] is True
    assert eval_summary["teacher_target_shape_viable"] is True
    assert eval_summary["teacher_usage_split"] == "eval_primary"
    assert eval_summary["teacher_usage_allowed_for_train_or_checkpoint"] is False
    assert eval_summary["candidate_for_train_side_teacher"] is False
    assert (
        eval_summary["candidate_for_train_side_teacher_reason"]
        == "eval_split_query_conditioned_teacher_not_allowed_for_training"
    )

    checkpoint_summary = separated_marginal_teacher_targets(
        rows,
        teacher_usage_split="checkpoint_selection",
    )
    assert checkpoint_summary["available"] is True
    assert checkpoint_summary["teacher_target_shape_viable"] is True
    assert checkpoint_summary["teacher_usage_split"] == "checkpoint_selection"
    assert checkpoint_summary["teacher_usage_allowed_for_train_or_checkpoint"] is True
    assert checkpoint_summary["candidate_for_train_side_teacher"] is True
    assert checkpoint_summary["candidate_for_train_side_teacher_reason"] == "candidate_available"

    eval_segment_scores, eval_point_scores, eval_vector_diag = (
        separated_marginal_teacher_selector_score_vectors(
            eval_summary,
            point_count=16,
        )
    )
    assert eval_segment_scores is None
    assert eval_point_scores is None
    assert (
        eval_vector_diag["reason"]
        == "eval_split_query_conditioned_teacher_not_allowed_for_training"
    )

    compact_checkpoint_summary = dict(checkpoint_summary)
    compact_checkpoint_summary.pop("segment_target_rows")
    compact_checkpoint_summary.pop("point_target_rows")
    compact_segment_scores, compact_point_scores, compact_vector_diag = (
        separated_marginal_teacher_selector_score_vectors(
            compact_checkpoint_summary,
            point_count=16,
        )
    )
    assert compact_segment_scores is None
    assert compact_point_scores is None
    assert compact_vector_diag["reason"] == "missing_target_rows_full_selector_trace_required"

    segment_scores, point_scores, vector_diag = separated_marginal_teacher_selector_score_vectors(
        checkpoint_summary,
        point_count=16,
    )
    assert vector_diag["available"] is True
    assert vector_diag["teacher_usage_split"] == "checkpoint_selection"
    assert vector_diag["positive_segment_score_point_count"] == 16
    assert vector_diag["positive_point_score_count"] == 3
    assert segment_scores is not None
    assert point_scores is not None
    assert segment_scores[:8].tolist() == pytest.approx([1.0] * 8)
    assert segment_scores[8:16].tolist() == pytest.approx([0.5] * 8)
    assert point_scores[2] == pytest.approx(1.0)
    assert point_scores[3] == pytest.approx(0.25)
    assert point_scores[11] == pytest.approx(1.0)

    primary_point_scores = torch.linspace(0.0, 15.0, steps=16)
    primary_segment_scores = torch.linspace(15.0, 0.0, steps=16)
    hybrid_segment_scores, hybrid_point_scores, hybrid_diag = (
        hybrid_marginal_teacher_selector_score_vectors(
            primary_point_scores=primary_point_scores,
            primary_segment_scores=primary_segment_scores,
            primary_segment_score_source_label="primary_selector_segment_scores",
            teacher_point_scores=point_scores,
            teacher_segment_scores=segment_scores,
            teacher_weight=0.25,
        )
    )
    assert hybrid_diag["available"] is True
    assert hybrid_diag["teacher_weight"] == pytest.approx(0.25)
    assert hybrid_diag["primary_segment_score_source"] == "primary_selector_segment_scores"
    assert hybrid_diag["teacher_positive_point_score_count"] == 3
    assert hybrid_diag["teacher_positive_segment_score_point_count"] == 16
    assert hybrid_diag["hybrid_positive_point_score_count"] == 15
    assert hybrid_segment_scores is not None
    assert hybrid_point_scores is not None
    assert hybrid_point_scores[2] == pytest.approx(0.75 * (2.0 / 15.0) + 0.25)
    assert hybrid_point_scores[3] == pytest.approx(0.75 * (3.0 / 15.0) + 0.0625)
    assert hybrid_point_scores[11] == pytest.approx(0.75 * (11.0 / 15.0) + 0.25)
    assert hybrid_segment_scores[0] == pytest.approx(0.75 + 0.25)
    assert hybrid_segment_scores[8] == pytest.approx(0.75 * (7.0 / 15.0) + 0.125)

    mismatch_segment_scores, mismatch_point_scores, mismatch_diag = (
        hybrid_marginal_teacher_selector_score_vectors(
            primary_point_scores=primary_point_scores[:-1],
            primary_segment_scores=primary_segment_scores[:-1],
            teacher_point_scores=point_scores,
            teacher_segment_scores=segment_scores,
            teacher_weight=0.25,
        )
    )
    assert mismatch_segment_scores is None
    assert mismatch_point_scores is None
    assert mismatch_diag["available"] is False
    assert mismatch_diag["reason"] == "score_shape_mismatch"


def test_retained_decision_marginal_query_local_utility_diagnostic_uses_provided_cache() -> None:
    points = torch.zeros((5, 5), dtype=torch.float32)
    points[:, 0] = torch.arange(5, dtype=torch.float32)
    points[:, 1] = torch.linspace(0.0, 4.0, steps=5)
    points[:, 2] = torch.linspace(0.0, 4.0, steps=5)
    retained = torch.tensor([True, False, True, False, True])
    query = {
        "type": "range",
        "params": {
            "t_start": 1.5,
            "t_end": 3.5,
            "lat_min": 1.5,
            "lat_max": 3.5,
            "lon_min": 1.5,
            "lon_max": 3.5,
        },
    }
    query_cache = ScoringQueryCache.for_workload(points, [(0, 5)], [query])

    diagnostics = retained_decision_marginal_query_local_utility_diagnostics(
        points=points,
        boundaries=[(0, 5)],
        typed_queries=[query],
        primary_retained_mask=retained,
        selector_scores=torch.linspace(0.0, 1.0, steps=5),
        query_cache=query_cache,
        max_retained_per_source=2,
        max_removed_candidates=2,
    )

    assert diagnostics["available"] is True
    assert diagnostics["query_cache_provided"] is True
    assert diagnostics["query_cache_created"] is False
    assert diagnostics["query_cache_range_audit_support_count"] == 1
    assert len(query_cache.range_audit_supports) == 1
    assert diagnostics["candidate_count"] == 4
    assert diagnostics["context_fields_available"]["selector_segment_context"] is False
    assert all(row["selector_segment_context"] is None for row in diagnostics["rows"])


def test_segment_source_attribution_uses_canonical_segment_index_after_score_sort() -> None:
    scores = torch.linspace(0.0, 1.0, steps=16, dtype=torch.float32)
    segment_scores = torch.zeros((16,), dtype=torch.float32)
    segment_scores[8:12] = 10.0
    segment_scores[0:4] = 1.0
    segment_scores[4:8] = 2.0
    segment_scores[12:16] = 3.0

    _, trace = simplify_with_learned_segment_budget_with_trace(
        scores,
        [(0, 16)],
        compression_ratio=0.50,
        segment_size=4,
        segment_scores=segment_scores,
    )

    rows = trace["segment_source_attribution"]["rows"]
    by_bounds = {(row["start"], row["end"]): row for row in rows}
    assert by_bounds[(0, 4)]["segment_index"] == 0
    assert by_bounds[(4, 8)]["segment_index"] == 1
    assert by_bounds[(8, 12)]["segment_index"] == 2
    assert by_bounds[(12, 16)]["segment_index"] == 3
    assert by_bounds[(8, 12)]["allocation_order_index"] == 0
    assert by_bounds[(8, 12)]["segment_score_rank"] == 1
