"""QueryLocalUtility target, prior, and candidate-target tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from config.run_config import (
    DEFAULT_VALIDATION_ENDPOINT_PENALTY_WEIGHT,
    DEFAULT_VALIDATION_GLOBAL_SANITY_PENALTY_WEIGHT,
    DEFAULT_VALIDATION_LENGTH_PRESERVATION_MIN,
    DEFAULT_VALIDATION_SED_PENALTY_WEIGHT,
)
from data_preparation.ais_loader import generate_synthetic_ais_data
from learning.checkpoint_validation import (
    _validation_factorized_target_fit_metrics,
    _validation_query_local_utility_selection_score,
)
from learning.query_prior_fields import (
    build_train_query_prior_fields,
)
from learning.targets.query_local_utility import (
    QUERY_LOCAL_UTILITY_FINAL_LABEL_FORMULA,
    QUERY_LOCAL_UTILITY_HEAD_NAMES,
    build_query_local_utility_targets,
    query_local_utility_point_score,
)
from scoring.geometry_thresholds import (
    max_sed_ratio_for_compression,
)
from scoring.method_scoring import score_range_audit
from scoring.query_local_utility import query_local_utility_from_range_audit
from workloads.generation.generator import generate_typed_query_workload
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


def test_query_local_utility_prioritizes_query_local_components() -> None:
    weak = {
        "query_point_recall": 0.1,
        "range_turn_coverage": 0.1,
        "range_gap_min_coverage": 0.1,
        "range_query_local_interpolation_fidelity": 0.1,
    }
    strong = dict(weak)
    strong.update(
        {
            "query_point_recall": 0.7,
            "range_query_local_interpolation_fidelity": 0.7,
            "range_turn_coverage": 0.8,
            "range_gap_min_coverage": 0.6,
        }
    )

    strong_score = float(
        cast(Any, query_local_utility_from_range_audit(strong)["query_local_utility_score"])
    )
    weak_score = float(
        cast(Any, query_local_utility_from_range_audit(weak)["query_local_utility_score"])
    )
    assert strong_score > weak_score


def test_query_local_utility_has_true_query_local_interpolation_component() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 2.0],
            [3.0, 0.0, 3.0],
            [4.0, 0.0, 4.0],
        ],
        dtype=torch.float32,
    )
    retained = torch.tensor([True, False, False, False, True])
    query = {
        "type": "range",
        "params": {
            "t_start": 0.5,
            "t_end": 3.5,
            "lat_min": -1.0,
            "lat_max": 1.0,
            "lon_min": 0.5,
            "lon_max": 3.5,
        },
    }

    audit = score_range_audit(
        points=points,
        boundaries=[(0, 5)],
        retained_mask=retained,
        typed_queries=[query],
    )
    useful = query_local_utility_from_range_audit(audit)
    components = cast(dict[str, float], useful["query_local_utility_components"])

    assert audit["range_query_local_interpolation_fidelity"] == 0.0
    assert components["query_local_interpolation_fidelity"] == 0.0
    assert (
        useful["query_local_utility_metric_maturity"]
        == "query_local_direct_point_mass_behavior_without_legacy_fallbacks"
    )


def test_validation_query_local_utility_penalizes_bad_global_sanity() -> None:
    cfg = SimpleNamespace(
        validation_global_sanity_penalty_enabled=True,
        validation_global_sanity_penalty_weight=DEFAULT_VALIDATION_GLOBAL_SANITY_PENALTY_WEIGHT,
        validation_sed_penalty_weight=DEFAULT_VALIDATION_SED_PENALTY_WEIGHT,
        validation_endpoint_penalty_weight=DEFAULT_VALIDATION_ENDPOINT_PENALTY_WEIGHT,
        validation_length_preservation_min=DEFAULT_VALIDATION_LENGTH_PRESERVATION_MIN,
    )
    good = {
        "avg_length_preserved": 0.90,
        "avg_sed_ratio_vs_uniform": 1.00,
        "avg_sed_ratio_vs_uniform_max": max_sed_ratio_for_compression(0.05),
        "endpoint_sanity": 1.00,
    }
    bad = {
        "avg_length_preserved": 0.40,
        "avg_sed_ratio_vs_uniform": 2.50,
        "avg_sed_ratio_vs_uniform_max": max_sed_ratio_for_compression(0.05),
        "endpoint_sanity": 0.00,
    }

    assert _validation_query_local_utility_selection_score(0.50, bad, cast(Any, cfg)) < (
        _validation_query_local_utility_selection_score(0.50, good, cast(Any, cfg)) - 0.10
    )


def test_query_local_utility_point_score_uses_additive_qhit_behavior_branches() -> None:
    q_hit = torch.tensor([0.20], dtype=torch.float32)
    behavior = torch.tensor([0.40], dtype=torch.float32)
    replacement = torch.tensor([0.60], dtype=torch.float32)
    boundary = torch.tensor([0.30], dtype=torch.float32)

    score = query_local_utility_point_score(
        q_hit=q_hit,
        behavior=behavior,
        replacement=replacement,
        boundary=boundary,
    )

    expected = (0.50 * q_hit + 0.45 * behavior) * (0.75 + 0.25 * replacement) + (0.05 * boundary)
    assert torch.allclose(score, expected)


def test_query_local_utility_behavior_branch_not_multiplied_by_tiny_qhit() -> None:
    q_hit = torch.tensor([0.02], dtype=torch.float32)
    replacement = torch.tensor([1.0], dtype=torch.float32)
    boundary = torch.tensor([0.0], dtype=torch.float32)

    no_behavior = query_local_utility_point_score(
        q_hit=q_hit,
        behavior=torch.tensor([0.0], dtype=torch.float32),
        replacement=replacement,
        boundary=boundary,
    )
    with_behavior = query_local_utility_point_score(
        q_hit=q_hit,
        behavior=torch.tensor([0.60], dtype=torch.float32),
        replacement=replacement,
        boundary=boundary,
    )
    qhit_increment = (
        query_local_utility_point_score(
            q_hit=q_hit + 0.01,
            behavior=torch.tensor([0.60], dtype=torch.float32),
            replacement=replacement,
            boundary=boundary,
        )
        - with_behavior
    )

    assert no_behavior.item() == pytest.approx(0.01)
    assert (with_behavior - no_behavior).item() == pytest.approx(0.27)
    assert qhit_increment.item() == pytest.approx(0.005)


def test_validation_factorized_target_fit_metrics_are_diagnostic_only() -> None:
    points = torch.zeros((8, 8), dtype=torch.float32)
    points[:, 0] = torch.arange(8, dtype=torch.float32)
    points[:, 1] = torch.linspace(0.0, 0.7, steps=8)
    points[:, 2] = torch.linspace(0.0, 0.7, steps=8)
    points[:, 7] = torch.tensor([0.0, 0.2, 1.0, 0.3, 0.0, 0.0, 0.0, 0.0])
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
    workload = SimpleNamespace(typed_queries=[query])
    targets = build_query_local_utility_targets(
        points=points,
        boundaries=[(0, 8)],
        typed_queries=[query],
        segment_size=4,
    )
    head_logits = torch.logit(targets.head_targets.clamp(1e-4, 1.0 - 1e-4))

    metrics = _validation_factorized_target_fit_metrics(
        head_logits=head_logits,
        points=points,
        boundaries=[(0, 8)],
        workload=cast(Any, workload),
        segment_size=4,
    )

    assert metrics["factorized_target_fit_available"] == 1.0
    assert metrics["factorized_target_fit_used_for_checkpoint_selection"] == 0.0
    assert metrics["head_segment_budget_target_target_fit_available"] == 1.0
    assert metrics["head_segment_budget_target_top5_mass_recall"] > 0.99
    assert metrics["segment_budget_canonical_segment_fit_available"] == 1.0
    assert metrics["segment_budget_canonical_segment_top5_mass_recall"] > 0.99


def test_factorized_targets_and_prior_fields_are_train_query_derived() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=4, n_points_per_ship=32, seed=82)
    points = torch.cat(trajectories, dim=0)
    boundaries = _boundaries(trajectories)
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=5,
        workload_map={"range": 1.0},
        seed=12,
        workload_profile_id="range_query_mix",
        range_max_point_hit_fraction=1.0,
        range_duplicate_iou_threshold=1.0,
    )

    targets = build_query_local_utility_targets(
        points=points,
        boundaries=boundaries,
        typed_queries=workload.typed_queries,
    )
    prior = build_train_query_prior_fields(
        points=points,
        boundaries=boundaries,
        typed_queries=workload.typed_queries,
        labels=targets.labels,
        workload_profile_id="range_query_mix",
        train_workload_seed=12,
    )

    assert targets.head_targets.shape == (points.shape[0], len(QUERY_LOCAL_UTILITY_HEAD_NAMES))
    assert targets.labels.shape[0] == points.shape[0]
    assert targets.diagnostics["target_family"] == "QueryLocalUtilityFactorized"
    assert "support_fraction_by_threshold_by_head" in targets.diagnostics
    assert "final_label_support_fraction_by_threshold" in targets.diagnostics
    assert prior["built_from_split"] == "train_only"
    assert prior["contains_eval_queries"] is False
    assert prior["contains_validation_queries"] is False


def test_factorized_replacement_target_is_query_local_and_final_label_keeps_query_mass() -> None:
    points = torch.zeros((10, 8), dtype=torch.float32)
    points[:, 0] = torch.arange(10, dtype=torch.float32)
    points[:, 1] = torch.linspace(0.0, 1.0, steps=10)
    points[:, 2] = torch.linspace(0.0, 1.0, steps=10)
    points[:, 5] = 0.0
    points[:, 6] = 0.0
    points[0, 5] = 1.0
    points[-1, 6] = 1.0
    points[:, 7] = 1.0
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 10.0,
            "lat_min": -1.0,
            "lat_max": 2.0,
            "lon_min": -1.0,
            "lon_max": 2.0,
        },
    }

    targets = build_query_local_utility_targets(
        points=points,
        boundaries=[(0, 10)],
        typed_queries=[query],
    )

    replacement = targets.head_targets[
        :, tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("replacement_representative_value")
    ]
    final_score = targets.labels[:, QUERY_TYPE_ID_RANGE]
    assert int((replacement > 0.0).sum().item()) == 4
    assert int((final_score > 0.0).sum().item()) == 10
    assert (
        targets.diagnostics["replacement_representative_value_normalization"]
        == "conditional_on_query_hit"
    )
    assert targets.diagnostics["final_label_formula"] == QUERY_LOCAL_UTILITY_FINAL_LABEL_FORMULA


def test_query_hit_head_target_uses_ship_query_evidence_multiplier() -> None:
    points = torch.zeros((8, 8), dtype=torch.float32)
    points[:, 0] = torch.arange(8, dtype=torch.float32)
    points[:6, 1] = 0.0
    points[6:, 1] = 10.0
    points[:, 2] = 0.0
    points[:, 7] = 1.0
    long_ship_query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 6.0,
            "lat_min": -1.0,
            "lat_max": 1.0,
            "lon_min": -1.0,
            "lon_max": 1.0,
        },
        "_metadata": {
            "anchor_family": "density",
            "footprint_family": "medium_operational",
        },
    }
    short_ship_query = {
        "type": "range",
        "params": {
            "t_start": 5.5,
            "t_end": 8.5,
            "lat_min": 9.0,
            "lat_max": 11.0,
            "lon_min": -1.0,
            "lon_max": 1.0,
        },
        "_metadata": {
            "anchor_family": "sparse_background_control",
            "footprint_family": "medium_operational",
        },
    }

    targets = build_query_local_utility_targets(
        points=points,
        boundaries=[(0, 6), (6, 8)],
        typed_queries=[long_ship_query, short_ship_query],
        segment_size=2,
    )
    query_hit_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("query_hit_probability")
    q_hit_gate = targets.head_targets[:, query_hit_idx]

    assert targets.diagnostics["query_hit_target_variant"] == (
        "raw_query_hit_ship_evidence_multiplier"
    )
    assert targets.diagnostics["query_hit_target_semantics"] == (
        "raw_query_hit_scale_preserving_ship_evidence_ranker"
    )
    assert targets.diagnostics["query_hit_target_family_conditioned"] is True
    assert q_hit_gate[6] > q_hit_gate[0]
    assert q_hit_gate[6] == pytest.approx(0.5)
    assert float(q_hit_gate.max().item()) <= 0.5 + 1e-6
    assert q_hit_gate[0] < 0.5


def test_conditional_behavior_target_is_masked_to_query_hits() -> None:
    points = torch.zeros((6, 8), dtype=torch.float32)
    points[:, 0] = torch.arange(6, dtype=torch.float32)
    points[:, 1] = torch.arange(6, dtype=torch.float32)
    points[:, 2] = 0.0
    points[0, 5] = 1.0
    points[-1, 6] = 1.0
    points[2, 7] = 1.0
    query = {
        "type": "range",
        "params": {
            "t_start": 1.0,
            "t_end": 3.0,
            "lat_min": 1.0,
            "lat_max": 3.0,
            "lon_min": -1.0,
            "lon_max": 1.0,
        },
        "_metadata": {
            "anchor_family": "density",
            "footprint_family": "medium_operational",
        },
    }

    targets = build_query_local_utility_targets(
        points=points,
        boundaries=[(0, 6)],
        typed_queries=[query],
    )
    query_hit_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("query_hit_probability")
    behavior_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("conditional_behavior_utility")

    hit_mask = targets.head_targets[:, query_hit_idx] > 0.0

    assert torch.equal(targets.head_mask[:, behavior_idx], hit_mask)
    assert targets.head_mask[:, query_hit_idx].all()
    assert (
        targets.diagnostics["conditional_behavior_utility_training"] == "masked_to_query_hit_points"
    )


def test_conditional_behavior_target_includes_segment_query_support() -> None:
    points = torch.zeros((8, 8), dtype=torch.float32)
    points[:, 0] = torch.arange(8, dtype=torch.float32)
    points[:, 1] = torch.arange(8, dtype=torch.float32)
    points[:, 2] = 0.0
    points[2, 7] = 1.0
    query = {
        "type": "range",
        "params": {
            "t_start": 0.0,
            "t_end": 3.0,
            "lat_min": 0.0,
            "lat_max": 3.0,
            "lon_min": -1.0,
            "lon_max": 1.0,
        },
        "_metadata": {
            "anchor_family": "density",
            "footprint_family": "medium_operational",
        },
    }

    targets = build_query_local_utility_targets(
        points=points,
        boundaries=[(0, 8)],
        typed_queries=[query],
        segment_size=4,
    )
    query_hit_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("query_hit_probability")
    behavior_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("conditional_behavior_utility")
    q_hit = targets.head_targets[:, query_hit_idx]
    behavior = targets.head_targets[:, behavior_idx]
    hit_mask = q_hit > 0.0

    assert targets.diagnostics["conditional_behavior_target_variant"] == (
        "query_segment_local_behavior_utility"
    )
    assert (
        "segment_raw_query_hit_evidence_multiplier_support"
        in targets.diagnostics["conditional_behavior_target_base_source"]
    )
    assert int(hit_mask.sum().item()) == 4
    assert behavior[2] > 0.0
    assert int((behavior > 0.0).sum().item()) < int(hit_mask.sum().item())
    assert torch.all(behavior[~hit_mask] == 0.0)


def test_path_length_support_target_is_query_free_segment_geometry() -> None:
    points = torch.zeros((8, 8), dtype=torch.float32)
    points[:, 0] = torch.arange(8, dtype=torch.float32)
    points[:, 1] = torch.arange(8, dtype=torch.float32)
    points[:, 2] = torch.tensor([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    first_query = {
        "type": "range",
        "params": {
            "t_start": 0.0,
            "t_end": 7.0,
            "lat_min": 0.0,
            "lat_max": 3.5,
            "lon_min": -1.0,
            "lon_max": 4.0,
        },
    }
    second_query = {
        "type": "range",
        "params": {
            "t_start": 3.0,
            "t_end": 7.0,
            "lat_min": 3.0,
            "lat_max": 7.5,
            "lon_min": -1.0,
            "lon_max": 1.0,
        },
        "_metadata": {
            "anchor_family": "density",
            "footprint_family": "medium_operational",
        },
    }

    first_targets = build_query_local_utility_targets(
        points=points,
        boundaries=[(0, 8)],
        typed_queries=[first_query],
        segment_size=2,
    )
    second_targets = build_query_local_utility_targets(
        points=points,
        boundaries=[(0, 8)],
        typed_queries=[second_query],
        segment_size=2,
    )
    path_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("path_length_support_target")
    first_path = first_targets.head_targets[:, path_idx]
    second_path = second_targets.head_targets[:, path_idx]

    assert torch.allclose(first_path, second_path)
    assert float(first_path.sum().item()) > 0.0
    assert int((first_path > 0.0).sum().item()) < int(first_path.numel())
    assert first_targets.head_mask[:, path_idx].all()
    assert first_targets.diagnostics["path_length_support_target_query_free"] is True
    assert first_targets.diagnostics["path_length_support_target_highpass_quantile"] == 0.50
    assert (
        first_targets.diagnostics["path_length_support_target_base_source"]
        == "per_point_path_length_removal_loss_segment_highpass_mass"
    )


def test_conditional_behavior_target_alignment_diagnostics_report_final_mass_recall() -> None:
    points = torch.zeros((10, 8), dtype=torch.float32)
    points[:, 0] = torch.arange(10, dtype=torch.float32)
    points[:, 1] = torch.arange(10, dtype=torch.float32)
    points[:, 2] = 0.0
    points[:, 7] = torch.linspace(0.0, 1.0, steps=10)
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 10.0,
            "lat_min": -1.0,
            "lat_max": 10.0,
            "lon_min": -1.0,
            "lon_max": 1.0,
        },
        "_metadata": {
            "anchor_family": "density",
            "footprint_family": "medium_operational",
        },
    }

    targets = build_query_local_utility_targets(
        points=points,
        boundaries=[(0, 5), (5, 10)],
        typed_queries=[query],
    )
    alignment = targets.diagnostics["conditional_behavior_target_alignment"]

    assert alignment["valid_point_count"] == 10
    assert alignment["topk_ratio"] == 0.05
    assert alignment["spearman_with_final_score"] is not None
    assert alignment["spearman_with_final_score"] > 0.0
    assert alignment["topk_final_score_mass_recall_ranked_by_behavior"] > 0.0
    assert "spearman_with_ship_query_evidence" in alignment
    assert "topk_ship_query_evidence_mass_recall_ranked_by_behavior" in alignment
    assert "spearman_with_replacement_representative_value" in alignment
    assert "topk_overlap_with_segment_budget_target" in alignment
    candidates = targets.diagnostics["conditional_behavior_candidate_alignment"]
    current = candidates["current_local_behavior"]
    replacement_gated = candidates["replacement_gated_local_behavior"]

    assert set(candidates) == {
        "current_local_behavior",
        "replacement_gated_local_behavior",
        "segment_gated_local_behavior",
        "replacement_support_only_local_behavior",
        "replacement_segment_gated_local_behavior",
    }
    assert current["valid_point_count"] == 10
    assert replacement_gated["valid_point_count"] == 10
    assert "support_fraction_by_threshold" in replacement_gated
    assert "topk_final_score_mass_recall_ranked_by_behavior" in replacement_gated
    assert current["ship_query_pair_count"] == 2
    assert current["ship_query_topk_selected_point_count"] == 1
    assert current["ship_query_pair_coverage_at_topk"] == 0.5
    assert "topk_ship_query_evidence_mass_recall_ranked_by_behavior" in replacement_gated

    partial = targets.diagnostics["conditional_behavior_replacement_partial_alignment"]
    assert partial["available"] is True
    assert partial["diagnostic_only"] is True
    assert partial["control"] == "replacement_representative_value"
    assert partial["valid_point_count"] == 10
    assert partial["behavior_replacement_spearman"] is not None
    partial_final = partial["references"]["final_score"]
    assert partial_final["behavior_spearman"] is not None
    assert partial_final["replacement_spearman"] is not None
    assert partial_final["behavior_partial_spearman_controlling_replacement"] is not None

    ship_alignment = targets.diagnostics["ship_query_evidence_target_alignment"]
    assert ship_alignment["available"] is True
    assert ship_alignment["diagnostic_only"] is True
    assert ship_alignment["reference"] == "ship_query_evidence"
    assert ship_alignment["valid_point_count"] == 10
    assert ship_alignment["reference_positive_point_count"] == 10
    assert ship_alignment["reference_mass"] == pytest.approx(2.0)
    assert set(ship_alignment["rankers"]) == {
        "final_score",
        "query_hit_probability",
        "conditional_behavior_utility",
        "boundary_event_utility",
        "replacement_representative_value",
        "segment_budget_target",
        "path_length_support_target",
    }
    final_alignment = ship_alignment["rankers"]["final_score"]
    assert "spearman_with_ship_query_evidence" in final_alignment
    assert final_alignment["topk_ship_query_evidence_mass_recall"] > 0.0
    assert final_alignment["ship_query_pair_count"] == 2
    segment_candidates = targets.diagnostics["segment_budget_ship_presence_candidate_alignment"]
    assert segment_candidates["available"] is True
    assert segment_candidates["diagnostic_only"] is True
    assert segment_candidates["active_training_target_unchanged"] is True
    assert set(segment_candidates["candidates"]) == {
        "active_segment_budget_target",
        "ship_presence_segment_budget_candidate",
        "final_score_ship_presence_blend_segment_budget_candidate",
        "query_hit_ship_presence_blend_segment_budget_candidate",
    }
    family_trainability = targets.diagnostics["family_conditioned_target_trainability"]
    assert family_trainability["available"] is True
    assert family_trainability["diagnostic_only"] is True
    density_row = family_trainability["group_by"]["anchor_family"]["density"]
    medium_operational_row = family_trainability["group_by"]["footprint_family"][
        "medium_operational"
    ]
    assert density_row["focus_family"] is True
    assert density_row["query_count"] == 1
    assert density_row["valid_hit_point_count"] == 10
    assert "segment_budget_target" in density_row["ranker_alignment"]
    assert medium_operational_row["focus_family"] is True
    family_candidates = targets.diagnostics["family_local_target_candidate_alignment"]
    assert family_candidates["available"] is True
    assert family_candidates["diagnostic_only"] is True
    assert family_candidates["active_training_target_unchanged"] is False
    assert family_candidates["candidate_usage"] == (
        "active_query_hit_target_uses_raw_query_hit_ship_evidence_multiplier; "
        "remaining_family_candidates_diagnostic_only"
    )
    candidate_row = family_candidates["group_by"]["footprint_family"]["medium_operational"]
    assert candidate_row["focus_family"] is True
    assert set(candidate_row["candidate_alignment"]) == {
        "family_query_hit_ship_blend_candidate",
        "family_ship_gated_behavior_candidate",
        "family_boundary_replacement_ship_score_candidate",
        "family_local_composed_score_candidate",
        "family_local_segment_budget_candidate",
        "family_query_hit_ship_segment_top20_mean_candidate",
        "family_query_hit_ship_segment_max_candidate",
        "family_composed_segment_top20_mean_candidate",
        "family_ship_query_pair_fractional_segment_candidate",
    }
    assert "active_final_score" in candidate_row["active_baseline_alignment"]
    segment_top20 = candidate_row["candidate_alignment"][
        "family_query_hit_ship_segment_top20_mean_candidate"
    ]
    assert segment_top20["two_stage_point_ranker"] == "family_query_hit_ship_blend_candidate"
    assert "two_stage_ship_query_pair_coverage" in segment_top20


def test_ship_presence_segment_budget_candidate_improves_ship_evidence_alignment() -> None:
    points = torch.zeros((8, 8), dtype=torch.float32)
    points[:, 0] = torch.arange(8, dtype=torch.float32)
    points[:, 1] = torch.arange(8, dtype=torch.float32) * 0.01
    points[:, 2] = 0.0
    points[:6, 7] = 1.0
    points[6:, 7] = 0.0
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 8.0,
            "lat_min": -1.0,
            "lat_max": 1.0,
            "lon_min": -1.0,
            "lon_max": 1.0,
        },
    }

    targets = build_query_local_utility_targets(
        points=points,
        boundaries=[(0, 6), (6, 8)],
        typed_queries=[query],
        segment_size=2,
    )

    candidates = targets.diagnostics["segment_budget_ship_presence_candidate_alignment"][
        "candidates"
    ]
    active = candidates["active_segment_budget_target"]
    ship_presence = candidates["ship_presence_segment_budget_candidate"]
    query_hit_blend = candidates["query_hit_ship_presence_blend_segment_budget_candidate"]

    assert active["target_std"] > 0.0
    assert (
        ship_presence["spearman_with_ship_query_evidence"]
        > active["spearman_with_ship_query_evidence"]
    )
    assert ship_presence["topk_ship_query_evidence_mass_recall"] == pytest.approx(1.0)
    assert ship_presence["ship_query_pair_coverage_at_topk"] == pytest.approx(0.5)
    assert (
        query_hit_blend["spearman_with_ship_query_evidence"]
        > active["spearman_with_ship_query_evidence"]
    )


def test_family_local_target_candidate_reports_active_evidence_gate_semantics() -> None:
    points = torch.zeros((8, 8), dtype=torch.float32)
    points[:, 0] = torch.arange(8, dtype=torch.float32)
    points[:, 1] = torch.arange(8, dtype=torch.float32) * 0.01
    points[:, 2] = 0.0
    points[:6, 7] = 1.0
    points[6:, 7] = 0.0
    query = {
        "type": "range",
        "params": {
            "t_start": -1.0,
            "t_end": 8.0,
            "lat_min": -1.0,
            "lat_max": 1.0,
            "lon_min": -1.0,
            "lon_max": 1.0,
        },
        "_metadata": {
            "anchor_family": "density",
            "footprint_family": "medium_operational",
        },
    }

    targets = build_query_local_utility_targets(
        points=points,
        boundaries=[(0, 6), (6, 8)],
        typed_queries=[query],
        segment_size=2,
    )

    family_candidates = targets.diagnostics["family_local_target_candidate_alignment"]
    medium_operational = family_candidates["group_by"]["footprint_family"]["medium_operational"]
    active_final = medium_operational["active_baseline_alignment"]["active_final_score"]
    query_hit_candidate = medium_operational["candidate_alignment"][
        "family_query_hit_ship_blend_candidate"
    ]
    composed_candidate = medium_operational["candidate_alignment"][
        "family_local_composed_score_candidate"
    ]
    segment_top20 = medium_operational["candidate_alignment"][
        "family_query_hit_ship_segment_top20_mean_candidate"
    ]
    segment_sum = medium_operational["candidate_alignment"]["family_local_segment_budget_candidate"]

    assert family_candidates["candidate_usage"] == (
        "active_query_hit_target_uses_raw_query_hit_ship_evidence_multiplier; "
        "remaining_family_candidates_diagnostic_only"
    )
    assert family_candidates["active_training_target_unchanged"] is False
    assert query_hit_candidate["spearman_with_ship_query_evidence"] is not None
    assert composed_candidate["spearman_with_ship_query_evidence"] is not None
    assert active_final["spearman_with_ship_query_evidence"] is not None
    assert "topk_active_final_score_mass_recall" in composed_candidate
    assert (
        segment_top20["two_stage_ship_query_pair_coverage"]
        >= segment_sum["two_stage_ship_query_pair_coverage"]
    )
    assert (
        segment_top20["two_stage_selected_point_count"]
        == segment_sum["two_stage_selected_point_count"]
    )
