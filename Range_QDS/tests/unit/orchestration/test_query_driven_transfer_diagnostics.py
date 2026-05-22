"""Derived query-driven diagnostic artifact tests."""

from __future__ import annotations

import pytest
import torch

from orchestration.diagnostics.family_transfer_path_diagnostic import (
    build_family_transfer_path_diagnostic,
)
from orchestration.diagnostics.selection_eval_segment_teacher_transfer_diagnostic import (
    build_selection_eval_segment_teacher_transfer_diagnostic,
)
from orchestration.diagnostics.selection_marginal_segment_calibration_diagnostic import (
    build_selection_marginal_segment_calibration_diagnostic,
)
from orchestration.diagnostics.selection_segment_transfer_feature_admissibility_diagnostic import (
    build_selection_segment_transfer_feature_admissibility_diagnostic,
)
from orchestration.diagnostics.selector_marginal_calibration_diagnostic import (
    build_selector_marginal_calibration_diagnostic,
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


def test_family_transfer_path_diagnostic_uses_artifact_focus_families_when_present() -> None:
    artifact = {
        "matched": {
            "MLQDS": {"query_local_utility_score": 0.14},
            "uniform": {"query_local_utility_score": 0.12},
            "DouglasPeucker": {"query_local_utility_score": 0.11},
        },
        "final_claim_summary": {"final_success_allowed": False},
        "workload_stability_gate": {"gate_pass": True},
        "support_overlap_gate": {"gate_pass": True},
        "target_diffusion_gate": {"gate_pass": True},
        "global_sanity_gate": {"gate_pass": True},
        "workload_distribution_comparison": {"workload_signature_gate": {"all_pass": True}},
        "query_generation_diagnostics": {
            "train": {
                "workload_profile": {
                    "anchor_family_weights": {"density": 0.8},
                    "footprint_family_weights": {
                        "medium_operational": 0.7,
                        "large_context": 0.3,
                    },
                },
                "workload_signature": {
                    "anchor_family_counts": {"density": 32},
                    "footprint_family_counts": {
                        "medium_operational": 28,
                        "large_context": 12,
                    },
                },
            }
        },
        "predictability_audit": {
            "gate_pass": True,
            "prior_predictive_alignment_gate": {"gate_pass": True, "failed_checks": []},
            "metrics": {},
        },
        "learning_causality_summary": {"learning_causality_gate_pass": False},
        "training_target_diagnostics": {
            "query_local_utility_factorized": {
                "target_mode": "query_local_utility_factorized",
                "family_conditioned_target_trainability": {
                    "focus_families": {
                        "anchor_family": ["density"],
                        "footprint_family": ["medium_operational"],
                    },
                    "group_by": {
                        "anchor_family": {"density": {}},
                        "footprint_family": {"medium_operational": {}},
                    },
                },
            }
        },
        "training_fit_diagnostics": {
            "family_conditioned_head_trainability": {
                "group_by": {
                    "anchor_family": {"density": {}},
                    "footprint_family": {"medium_operational": {}},
                }
            }
        },
    }

    diagnostic = build_family_transfer_path_diagnostic([("current", artifact)])

    focus_pairs = {
        (row["group_key"], row["family"]) for row in diagnostic["artifacts"][0]["focus_family_rows"]
    }
    assert focus_pairs == {
        ("anchor_family", "density"),
        ("footprint_family", "medium_operational"),
    }
    pressure = diagnostic["artifacts"][0]["workload_family_pressure"]
    assert set(pressure["footprint_family_weights"]) == {"medium_operational"}
    assert HISTORICAL_SMALL_LOCAL_FAMILY not in pressure["footprint_family_weights"]


def test_selector_marginal_calibration_diagnostic_separates_score_and_segment_failures() -> None:
    artifact = {
        "matched": {
            "MLQDS": {"query_local_utility_score": 0.1673},
            "uniform": {"query_local_utility_score": 0.142},
            "DouglasPeucker": {"query_local_utility_score": 0.1671},
        },
        "final_claim_summary": {"final_success_allowed": False},
        "target_diffusion_gate": {"gate_pass": True},
        "global_sanity_gate": {"gate_pass": True},
        "predictability_audit": {"gate_pass": False},
        "learning_causality_summary": {"learning_causality_gate_pass": False},
        "selector_trace_diagnostics": {
            "eval_primary": {
                "segments_considered_count": 100,
                "segment_allocation_alignment_diagnostics": {
                    "available": True,
                    "segment_count": 100,
                    "allocation_count_total": 120,
                    "extra_allocation_count_total": 20,
                    "length_support_to_allocation_spearman": -0.01,
                    "segment_score_to_allocation_spearman": 0.83,
                    "allocation_weight_to_allocation_spearman": 0.82,
                    "top_groups": {
                        "top_10_percent": {
                            "length_support_segment_score_overlap_fraction": 0.10,
                        },
                        "top_20_percent": {
                            "length_support_segment_score_overlap_fraction": 0.20,
                        },
                    },
                    "component_diagnosis": (
                        "extra_slots_score_dominated_not_length_support_aligned"
                    ),
                },
                "allocation_point_selection_diagnostics": {
                    "available": True,
                    "primary_length_preservation": 0.66,
                    "same_allocation_length_only_point_selection_preservation": 0.76,
                    "same_allocation_length_only_delta": 0.10,
                    "same_allocation_length_only_gate_would_pass": True,
                    "component_diagnosis": (
                        "point_selection_can_clear_length_with_current_allocation"
                    ),
                },
                "allocation_counterfactual_diagnostics": {
                    "available": True,
                    "allocation_overlap_fraction": 0.83,
                    "extra_allocation_overlap_fraction": 0.37,
                    "length_support_allocation_counterfactual_preservation": 0.77,
                    "length_support_allocation_counterfactual_gate_would_pass": True,
                    "component_diagnosis": (
                        "length_support_allocation_counterfactual_can_clear_length"
                    ),
                },
                "retained_decision_marginal_query_local_utility_alignment": {
                    "available": True,
                    "candidate_count": 4,
                    "overall": {
                        "raw_score": {
                            "spearman": -0.06,
                            "top_minus_bottom_marginal": 0.00001,
                        },
                        "selector_score": {
                            "spearman": -0.04,
                            "top_minus_bottom_marginal": 0.00002,
                        },
                        "segment_score": {
                            "spearman": -0.02,
                            "top_minus_bottom_marginal": 0.00002,
                        },
                        "score_component_alignment": {
                            "head_probability_path_length_support_target": {
                                "available": True,
                                "spearman": 0.28,
                                "top_minus_bottom_marginal": 0.00007,
                            }
                        },
                        "query_free_teacher_proxy_alignment": {
                            "query_free_endpoint_support": {
                                "available": True,
                                "spearman": 0.58,
                                "top_minus_bottom_marginal": 0.00016,
                            }
                        },
                    },
                    "by_decision": {
                        "retained_removal_loss": {
                            "selector_score": {
                                "spearman": -0.03,
                                "top_minus_bottom_marginal": -0.00006,
                            }
                        }
                    },
                    "top_marginal_miss_summary": {
                        "bucket_counts": {
                            "high_score_low_exact_marginal": 1,
                            "high_marginal_under_ranked_by_scores": 1,
                        }
                    },
                    "separated_marginal_teacher_summary": {
                        "available": True,
                        "teacher_usage_split": "eval_primary",
                        "teacher_usage_allowed_for_train_or_checkpoint": False,
                        "teacher_target_shape_viable": True,
                        "candidate_for_train_side_teacher": False,
                        "candidate_for_train_side_teacher_reason": (
                            "eval_split_query_conditioned_teacher_not_allowed_for_training"
                        ),
                        "segment_target_count": 1,
                        "point_target_count": 1,
                        "segment_target_rows": [
                            {
                                "segment_index": 12,
                                "trajectory_index": 3,
                                "top_point_index": 384,
                                "segment_target": 1.0,
                                "raw_segment_positive_marginal_sum": 0.003,
                                "selector_segment_score_rank": 80,
                                "selector_segment_allocation_weight_rank": 90,
                                "selector_segment_length_support_rank": 65,
                                "selector_segment_allocation_count": 1,
                                "selector_segment_learned_count": 1,
                            }
                        ],
                        "point_target_rows": [
                            {
                                "point_index": 384,
                                "trajectory_index": 3,
                                "segment_index": 12,
                                "raw_point_marginal": 0.003,
                                "point_target_global": 1.0,
                                "selector_score_candidate_rank_fraction": 0.80,
                                "segment_score_candidate_rank_fraction": 0.90,
                                "selector_segment_score_rank": 80,
                                "selector_segment_allocation_count": 1,
                            }
                        ],
                    },
                    "learned_controllable_marginal_teacher_summary": {
                        "learned_controllable_retained_removal_count": 1,
                    },
                    "rows": [
                        {
                            "point_index": 384,
                            "trajectory_index": 3,
                            "source": "learned",
                            "decision": "retained_removal_loss",
                            "marginal_query_local_utility": 0.003,
                            "marginal_query_local_utility_candidate_rank": 1,
                            "marginal_query_local_utility_candidate_rank_fraction": 0.05,
                            "raw_score_candidate_rank": 75,
                            "raw_score_candidate_rank_fraction": 0.75,
                            "selector_score_candidate_rank": 80,
                            "selector_score_candidate_rank_fraction": 0.80,
                            "segment_score_candidate_rank": 90,
                            "segment_score_candidate_rank_fraction": 0.90,
                            "failure_buckets": [
                                "high_marginal_under_ranked_by_scores",
                            ],
                            "selector_segment_context": {
                                "segment_index": 12,
                                "segment_score_rank": 80,
                                "segment_length_support_rank": 65,
                                "segment_allocation_weight_rank": 90,
                                "segment_allocation_count": 1,
                                "learned_count": 1,
                                "length_repair_count": 0,
                            },
                            "selector_stage_state": {"learned_retained": True},
                        },
                        {
                            "point_index": 32,
                            "trajectory_index": 0,
                            "source": "removed",
                            "decision": "removed_addition_gain",
                            "marginal_query_local_utility": 0.00001,
                            "marginal_query_local_utility_candidate_rank": 4,
                            "marginal_query_local_utility_candidate_rank_fraction": 0.90,
                            "raw_score_candidate_rank": 1,
                            "raw_score_candidate_rank_fraction": 0.02,
                            "selector_score_candidate_rank": 2,
                            "selector_score_candidate_rank_fraction": 0.04,
                            "segment_score_candidate_rank": 1,
                            "segment_score_candidate_rank_fraction": 0.02,
                            "failure_buckets": ["high_score_low_exact_marginal"],
                            "selector_segment_context": {
                                "segment_index": 1,
                                "segment_score_rank": 1,
                                "segment_allocation_weight_rank": 2,
                                "segment_allocation_count": 2,
                                "learned_count": 2,
                                "length_repair_count": 0,
                            },
                        },
                    ],
                },
            }
        },
    }

    diagnostic = build_selector_marginal_calibration_diagnostic([("checkpoint93", artifact)])

    summary = diagnostic["summary"]
    assert summary["decision"] == ("diagnose_train_side_marginal_segment_calibration_not_promotion")
    assert summary["retained_marginal_alignment_layout"] == (
        "selector_trace_diagnostics.eval_primary."
        "retained_decision_marginal_query_local_utility_alignment"
    )
    alignment = diagnostic["artifacts"][0]["retained_marginal_alignment"]
    assert alignment["failure_mode_summary"]["selector_spearman"] == pytest.approx(-0.04)
    assert alignment["failure_mode_summary"]["high_score_low_exact_marginal_count"] == 1
    top_row = alignment["top_marginal_diagnostics"]["top_rows"][0]
    assert top_row["point_segment_score_rank"] == 90
    assert top_row["selector_segment_score_rank"] == 80
    segment_teacher = alignment["segment_marginal_teacher_diagnostics"]
    assert segment_teacher["candidate_for_train_side_teacher"] is False
    assert segment_teacher["top_segment_target_low_selector_segment_score_count"] == 1
    assert segment_teacher["top_segment_target_low_allocation_weight_count"] == 1
    allocation = alignment["allocation_diagnostics"]
    assert allocation["segment_allocation_alignment"]["component_diagnosis"] == (
        "extra_slots_score_dominated_not_length_support_aligned"
    )
    assert (
        allocation["allocation_point_selection"]["same_allocation_length_only_gate_would_pass"]
        is True
    )


def test_selection_marginal_segment_calibration_diagnostic_flags_split_transfer_gap() -> None:
    def _trace(
        *,
        split: str,
        candidate: bool,
        first_segment: int,
        second_segment: int,
    ) -> dict[str, object]:
        return {
            "segments_considered_count": 100,
            "segment_allocation_alignment_diagnostics": {
                "available": True,
                "length_support_to_allocation_spearman": -0.01,
                "segment_score_to_allocation_spearman": 0.84,
                "component_diagnosis": ("extra_slots_score_dominated_not_length_support_aligned"),
            },
            "allocation_point_selection_diagnostics": {
                "available": True,
                "primary_length_preservation": 0.67,
                "same_allocation_length_only_point_selection_preservation": 0.77,
                "same_allocation_length_only_gate_would_pass": True,
                "component_diagnosis": ("point_selection_can_clear_length_with_current_allocation"),
            },
            "retained_decision_marginal_query_local_utility_alignment": {
                "available": True,
                "candidate_count": 160,
                "overall": {
                    "raw_score": {"spearman": -0.16},
                    "selector_score": {"spearman": -0.15},
                    "segment_score": {"spearman": -0.10},
                },
                "separated_marginal_teacher_summary": {
                    "available": True,
                    "teacher_usage_split": split,
                    "teacher_target_shape_viable": True,
                    "candidate_for_train_side_teacher": candidate,
                    "candidate_for_train_side_teacher_reason": (
                        "candidate_available"
                        if candidate
                        else "eval_split_query_conditioned_teacher_not_allowed_for_training"
                    ),
                    "segment_target_count": 2,
                    "point_target_count": 2,
                    "segment_target_rows": [
                        {
                            "segment_index": first_segment,
                            "trajectory_index": 1,
                            "top_point_index": first_segment * 32,
                            "segment_target": 1.0,
                            "raw_segment_positive_marginal_sum": 0.004,
                            "selector_segment_score_rank": 80,
                            "selector_segment_allocation_weight_rank": 90,
                            "selector_segment_length_support_rank": 70,
                            "selector_segment_allocation_count": 1,
                            "selector_segment_learned_count": 1,
                        },
                        {
                            "segment_index": second_segment,
                            "trajectory_index": 2,
                            "top_point_index": second_segment * 32,
                            "segment_target": 0.5,
                            "raw_segment_positive_marginal_sum": 0.002,
                            "selector_segment_score_rank": 75,
                            "selector_segment_allocation_weight_rank": 85,
                            "selector_segment_length_support_rank": 60,
                            "selector_segment_allocation_count": 1,
                            "selector_segment_learned_count": 1,
                        },
                    ],
                    "point_target_rows": [
                        {
                            "point_index": first_segment * 32,
                            "segment_index": first_segment,
                            "raw_point_marginal": 0.004,
                            "point_target_global": 1.0,
                            "selector_score_candidate_rank_fraction": 0.75,
                            "segment_score_candidate_rank_fraction": 0.80,
                            "selector_segment_score_rank": 80,
                            "selector_segment_allocation_count": 1,
                        }
                    ],
                },
            },
        }

    artifact = {
        "matched": {
            "MLQDS": {"query_local_utility_score": 0.1673},
            "uniform": {"query_local_utility_score": 0.142},
            "DouglasPeucker": {"query_local_utility_score": 0.1671},
        },
        "workload_stability_gate": {"gate_pass": True},
        "support_overlap_gate": {"gate_pass": True},
        "target_diffusion_gate": {"gate_pass": True},
        "workload_distribution_comparison": {"workload_signature_gate": {"all_pass": True}},
        "predictability_audit": {"gate_pass": False},
        "learning_causality_summary": {"learning_causality_gate_pass": False},
        "global_sanity_gate": {"gate_pass": True},
        "final_claim_summary": {"final_success_allowed": False},
        "selector_trace_diagnostics": {
            "selection_primary": _trace(
                split="checkpoint_selection",
                candidate=True,
                first_segment=10,
                second_segment=20,
            ),
            "eval_primary": _trace(
                split="eval_primary",
                candidate=False,
                first_segment=30,
                second_segment=40,
            ),
        },
    }

    diagnostic = build_selection_marginal_segment_calibration_diagnostic(
        [("checkpoint93", artifact)]
    )

    summary = diagnostic["summary"]
    assert summary["decision"] == (
        "diagnose_selection_marginal_segment_transfer_before_training_semantics"
    )
    assert summary["decision_scope"] == "primary_artifact_last_input"
    assert summary["selection_layout"] == (
        "selector_trace_diagnostics.selection_primary."
        "retained_decision_marginal_query_local_utility_alignment"
    )
    result = diagnostic["artifacts"][0]
    assert result["decision"] == (
        "diagnose_selection_marginal_segment_transfer_before_training_semantics"
    )
    selection = result["selection_teacher"]
    assert selection["candidate_for_train_side_teacher"] is True
    assert selection["top_segment_low_selector_score_count"] == 2
    assert selection["top_segment_low_allocation_weight_count"] == 2
    overlap = result["selection_eval_segment_overlap"]
    assert overlap["segment_overlap_count"] == 0
    assert overlap["top_segment_overlap_fraction_of_selection_top"] == 0.0
    assert result["selection_allocation"]["same_allocation_length_only_gate_would_pass"] is True


def test_selection_eval_segment_teacher_transfer_diagnostic_blocks_direct_probe_on_weak_transfer() -> (
    None
):
    def _trace(
        *,
        split: str,
        candidate: bool,
        target_segments: tuple[int, int],
    ) -> dict[str, object]:
        return {
            "segment_source_attribution": {
                "available": True,
                "rows": [
                    {
                        "segment_index": 1,
                        "segment_score": 0.90,
                        "segment_allocation_weight": 0.90,
                        "segment_length_support_score": 0.10,
                        "segment_allocation_count": 2,
                        "learned_count": 1,
                        "length_repair_count": 0,
                    },
                    {
                        "segment_index": 2,
                        "segment_score": 0.80,
                        "segment_allocation_weight": 0.80,
                        "segment_length_support_score": 0.20,
                        "segment_allocation_count": 2,
                        "learned_count": 1,
                        "length_repair_count": 0,
                    },
                    {
                        "segment_index": 3,
                        "segment_score": 0.10,
                        "segment_allocation_weight": 0.10,
                        "segment_length_support_score": 0.90,
                        "segment_allocation_count": 1,
                        "learned_count": 0,
                        "length_repair_count": 1,
                    },
                    {
                        "segment_index": 4,
                        "segment_score": 0.20,
                        "segment_allocation_weight": 0.20,
                        "segment_length_support_score": 0.80,
                        "segment_allocation_count": 1,
                        "learned_count": 0,
                        "length_repair_count": 1,
                    },
                ],
            },
            "retained_decision_marginal_query_local_utility_alignment": {
                "available": True,
                "separated_marginal_teacher_summary": {
                    "available": True,
                    "teacher_usage_split": split,
                    "teacher_target_shape_viable": True,
                    "candidate_for_train_side_teacher": candidate,
                    "candidate_for_train_side_teacher_reason": (
                        "candidate_available"
                        if candidate
                        else "eval_split_query_conditioned_teacher_not_allowed_for_training"
                    ),
                    "segment_target_count": 2,
                    "point_target_count": 2,
                    "segment_target_rows": [
                        {
                            "segment_index": target_segments[0],
                            "segment_target": 1.0,
                        },
                        {
                            "segment_index": target_segments[1],
                            "segment_target": 0.5,
                        },
                    ],
                },
            },
        }

    artifact = {
        "matched": {
            "MLQDS": {"query_local_utility_score": 0.1673},
            "uniform": {"query_local_utility_score": 0.142},
            "DouglasPeucker": {"query_local_utility_score": 0.1671},
        },
        "workload_stability_gate": {"gate_pass": True},
        "support_overlap_gate": {"gate_pass": True},
        "target_diffusion_gate": {"gate_pass": True},
        "workload_distribution_comparison": {"workload_signature_gate": {"all_pass": True}},
        "predictability_audit": {"gate_pass": False},
        "learning_causality_summary": {"learning_causality_gate_pass": False},
        "global_sanity_gate": {"gate_pass": True},
        "final_claim_summary": {"final_success_allowed": False},
        "selector_trace_diagnostics": {
            "train_primary": _trace(
                split="train",
                candidate=True,
                target_segments=(1, 2),
            ),
            "selection_primary": _trace(
                split="checkpoint_selection",
                candidate=True,
                target_segments=(1, 2),
            ),
            "eval_primary": _trace(
                split="eval_primary",
                candidate=False,
                target_segments=(3, 4),
            ),
        },
    }

    diagnostic = build_selection_eval_segment_teacher_transfer_diagnostic(
        [("checkpoint93", artifact)]
    )

    summary = diagnostic["summary"]
    assert summary["decision"] == ("diagnose_transfer_features_before_guarded_calibration_probe")
    assert summary["decision_scope"] == "primary_artifact_last_input"
    result = diagnostic["artifacts"][0]
    assert result["decision"] == ("diagnose_transfer_features_before_guarded_calibration_probe")
    overlap = result["target_overlap"]
    assert overlap["positive_overlap_count"] == 0
    assert overlap["top_0.1_overlap_fraction_of_selection"] == 0.0
    assert overlap["selection_eval_teacher_target_spearman"] < 0.0
    selection_features = result["selection_feature_alignment"]["feature_alignment"]
    eval_features = result["eval_feature_alignment"]["feature_alignment"]
    assert selection_features["segment_score"]["spearman_with_segment_teacher_target"] > 0.0
    assert eval_features["segment_score"]["spearman_with_segment_teacher_target"] < 0.0
    transfer = result["feature_transfer_summary"]
    assert transfer["contradictory_feature_count"] > 0

    train_source_diagnostic = build_selection_eval_segment_teacher_transfer_diagnostic(
        [("checkpoint93", artifact)],
        source_trace_name="train_primary",
    )
    assert (
        train_source_diagnostic["summary"]["selection_layout"]
        == "selector_trace_diagnostics.train_primary.retained_decision_marginal_query_local_utility_alignment"
    )
    assert (
        train_source_diagnostic["artifacts"][0]["selection_teacher"]["teacher_usage_split"]
        == "train"
    )


def test_selection_segment_transfer_feature_admissibility_rejects_post_selection_signal() -> None:
    def _trace() -> dict[str, object]:
        return {
            "segment_source_attribution": {
                "available": True,
                "rows": [
                    {
                        "segment_index": 1,
                        "segment_score": 0.10,
                        "segment_allocation_weight": 0.10,
                        "segment_length_support_score": 0.30,
                        "segment_allocation_count": 1,
                        "learned_count": 0,
                        "length_repair_count": 1,
                        "retained_count": 2,
                        "retained_fraction": 0.05,
                    },
                    {
                        "segment_index": 2,
                        "segment_score": 0.20,
                        "segment_allocation_weight": 0.20,
                        "segment_length_support_score": 0.20,
                        "segment_allocation_count": 1,
                        "learned_count": 0,
                        "length_repair_count": 1,
                        "retained_count": 2,
                        "retained_fraction": 0.05,
                    },
                    {
                        "segment_index": 3,
                        "segment_score": 0.30,
                        "segment_allocation_weight": 0.30,
                        "segment_length_support_score": 0.10,
                        "segment_allocation_count": 2,
                        "learned_count": 1,
                        "length_repair_count": 0,
                        "retained_count": 3,
                        "retained_fraction": 0.10,
                    },
                    {
                        "segment_index": 4,
                        "segment_score": 0.40,
                        "segment_allocation_weight": 0.40,
                        "segment_length_support_score": 0.05,
                        "segment_allocation_count": 2,
                        "learned_count": 1,
                        "length_repair_count": 0,
                        "retained_count": 3,
                        "retained_fraction": 0.10,
                    },
                ],
            },
            "retained_decision_marginal_query_local_utility_alignment": {
                "separated_marginal_teacher_summary": {
                    "available": True,
                    "candidate_for_train_side_teacher": True,
                    "segment_target_count": 2,
                    "segment_target_rows": [
                        {"segment_index": 3, "segment_target": 1.0},
                        {"segment_index": 4, "segment_target": 0.8},
                    ],
                }
            },
        }

    artifact = {
        "matched": {
            "MLQDS": {"query_local_utility_score": 0.1673},
            "uniform": {"query_local_utility_score": 0.142},
            "DouglasPeucker": {"query_local_utility_score": 0.1671},
        },
        "workload_stability_gate": {"gate_pass": True},
        "support_overlap_gate": {"gate_pass": True},
        "target_diffusion_gate": {"gate_pass": True},
        "workload_distribution_comparison": {"workload_signature_gate": {"all_pass": True}},
        "predictability_audit": {"gate_pass": False},
        "learning_causality_summary": {"learning_causality_gate_pass": False},
        "global_sanity_gate": {"gate_pass": True},
        "final_claim_summary": {"final_success_allowed": False},
        "selector_trace_diagnostics": {
            "selection_primary": _trace(),
            "eval_primary": _trace(),
        },
    }

    diagnostic = build_selection_segment_transfer_feature_admissibility_diagnostic(
        [("checkpoint93", artifact)]
    )

    result = diagnostic["artifacts"][0]
    candidates = {row["name"]: row for row in result["candidate_rows"]}
    learned_count = candidates["learned_count_post_selection_coupled"]
    assert learned_count["classification"]["uses_post_selection_coupling"] is True
    assert learned_count["probe_admissible"] is False
    assert learned_count["rejection_reason"] == "uses_post_selection_or_unknown_features"
    assert result["feature_coupling_summary"]["post_selection_positive_candidate_names"] == [
        "learned_count_post_selection_coupled"
    ]
    assert diagnostic["summary"]["decision"] == (
        "guarded_pre_selection_transfer_calibration_probe_admissible"
    )
    assert "segment_score" in result["feature_coupling_summary"]["admissible_candidate_names"]
