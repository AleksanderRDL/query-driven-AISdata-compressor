"""Query-driven learned segment-budget selector tests."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from orchestration.selector_diagnostics import (
    retained_decision_marginal_query_local_utility_diagnostics,
)
from orchestration.selector_marginal_alignment import separated_marginal_teacher_targets
from orchestration.selector_teacher_vectors import (
    hybrid_marginal_teacher_selector_score_vectors,
    separated_marginal_teacher_selector_score_vectors,
)
from scoring.query_cache import ScoringQueryCache
from selection.learned_segment_budget import (
    simplify_with_learned_segment_budget_with_trace,
)

HISTORICAL_SMALL_LOCAL_FAMILY = "small_local"


def _boundaries(trajectories: list[torch.Tensor]) -> list[tuple[int, int]]:
    cursor = 0
    out = []
    for trajectory in trajectories:
        end = cursor + int(trajectory.shape[0])
        out.append((cursor, end))
        cursor = end
    return out


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
        "_metadata": {
            "anchor_family": "density",
            "footprint_family": "medium_operational",
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
    assert diagnostics["context_fields_available"]["query_local_utility_target"] is True
    assert diagnostics["context_fields_available"]["head_targets"] == {
        "boundary_event_utility": True,
        "conditional_behavior_utility": True,
        "path_length_support_target": True,
        "query_hit_probability": True,
        "replacement_representative_value": True,
        "segment_budget_target": True,
    }
    assert diagnostics["context_fields_available"]["target_diagnostic_error"] is None
    assert diagnostics["context_fields_available"]["query_family_hit_context"] is True
    assert diagnostics["context_fields_available"]["query_hit_run_ids"] is True
    assert diagnostics["context_fields_available"]["query_local_utility_component_delta"] is True
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
    assert learned_removal["query_local_utility_target"] > 0.0
    assert learned_removal["head_targets"]["query_hit_probability"] == pytest.approx(1.0)
    assert "conditional_behavior_utility" in learned_removal["head_targets"]
    assert learned_removal["head_targets"]["conditional_behavior_utility"] >= 0.0
    assert learned_removal["head_target_masks"]["conditional_behavior_utility"] is True
    assert "query_point_recall" in learned_removal["query_local_utility_component_delta"]
    assert learned_removal["primary_query_local_utility_components"]["query_point_recall"] >= 0.0
    assert learned_removal["candidate_query_local_utility_components"]["query_point_recall"] >= 0.0
    assert learned_removal["anchor_family"] == "density"
    assert learned_removal["footprint_family"] == "medium_operational"
    assert learned_removal["query_family_hit_context"]["anchor_family_counts"] == {"density": 1}
    assert learned_removal["query_hit_run_ids"] == ["q0:traj0:points2-4"]
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
