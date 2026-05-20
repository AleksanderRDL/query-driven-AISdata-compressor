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
from orchestration.diagnostics.workload_component_compatibility import (
    build_workload_component_compatibility_diagnostic,
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

def test_workload_component_compatibility_diagnostic_finds_blocking_families() -> None:
    artifact = {
        "matched": {
            "MLQDS": {
                "query_local_utility_score": 0.16,
                    "range_audit": {
                        "query_local_utility_component_weights": {
                            "query_point_recall": 0.50,
                            "query_local_interpolation_fidelity": 0.10,
                        },
                        "query_local_utility_components": {
                        "query_point_recall": 0.20,
                        "query_local_interpolation_fidelity": 0.70,
                    },
                    "range_query_metadata_component_summary": {
                        "group_by": {
                            "anchor_family": {
                                "density": {
                                    "query_local_utility_query_local_components": {
                                        "query_point_recall": 0.20,
                                        "query_local_interpolation_fidelity": 0.70,
                                    }
                                },
                                "sparse_background_control": {
                                    "query_local_utility_query_local_components": {
                                        "query_point_recall": 0.30,
                                        "query_local_interpolation_fidelity": 0.60,
                                    }
                                },
                            }
                        }
                    },
                },
            },
            "DouglasPeucker": {
                "query_local_utility_score": 0.18,
                "range_audit": {
                    "query_local_utility_components": {
                        "query_point_recall": 0.50,
                        "query_local_interpolation_fidelity": 0.60,
                    },
                    "range_query_metadata_component_summary": {
                        "group_by": {
                            "anchor_family": {
                                "density": {
                                    "query_local_utility_query_local_components": {
                                        "query_point_recall": 0.50,
                                        "query_local_interpolation_fidelity": 0.60,
                                    }
                                },
                                "sparse_background_control": {
                                    "query_local_utility_query_local_components": {
                                        "query_point_recall": 0.40,
                                        "query_local_interpolation_fidelity": 0.50,
                                    }
                                },
                            }
                        }
                    }
                },
            },
        },
        "workload_scoring_compatibility_diagnostics": {
            "comparisons_vs_baseline": {
                "DouglasPeucker": {
                    "anchor_family": {
                        "density": {
                            "query_count": 10,
                            "query_local_score_delta": -0.02,
                            "range_usefulness_delta": -0.03,
                            "range_component_deltas": {"range_ship_f1": -0.20},
                            "ship_evidence_count_deltas": {
                                "missed_trajectory_hit_count_total": 3,
                                "ship_presence_recall": -0.10,
                            },
                        },
                        "sparse_background_control": {
                            "query_count": 5,
                            "query_local_score_delta": -0.01,
                            "range_usefulness_delta": -0.01,
                            "range_component_deltas": {"range_ship_f1": -0.10},
                            "ship_evidence_count_deltas": {
                                "missed_trajectory_hit_count_total": 1,
                                "ship_presence_recall": -0.05,
                            },
                        },
                    }
                }
            }
        },
    }

    diagnostic = build_workload_component_compatibility_diagnostic([("strict", artifact)])

    assert diagnostic["summary"]["primary_minus_baseline_query_local_utility"] == pytest.approx(
        -0.02
    )
    blocking = diagnostic["summary"]["blocking_families"]
    assert blocking[0]["family"] == "density"
    assert blocking[0]["missed_trajectory_hit_count_delta"] == 3
    components = diagnostic["summary"]["persistent_negative_query_local_components"]
    assert components[0]["component"] == "query_point_recall"
    assert components[0]["negative_family_count"] == 2
    candidate_deltas = diagnostic["summary"]["recalibration_candidate_score_deltas"]
    assert candidate_deltas["active_component_weights"] < 0.0
    assert (
        candidate_deltas["query_local_behavior_heavy_component_weights_v0"]
        > candidate_deltas["active_component_weights"]
    )
    assert (
        candidate_deltas["query_point_mass_heavy_component_weights_v0"]
        > candidate_deltas["active_component_weights"]
    )
    assert (
        diagnostic["summary"]["blocker_preserving_candidate_status"]
        == "still_blocked"
    )
    unresolved = diagnostic["summary"]["blocker_preserving_candidate_unresolved_families"]
    assert unresolved[0]["family"] == "density"
    recalibration = diagnostic["artifacts"][0]["recalibration_diagnostics"]
    assert recalibration["diagnostic_only"] is True
    assert recalibration["masking_risk"] == "high"
    blocker_outcome = recalibration["blocker_preserving_outcome"]
    assert blocker_outcome["critical_family_pressure_preserved"] is False
    assert blocker_outcome["unresolved_blocker_family_count"] == 1


def test_family_transfer_path_diagnostic_flags_missing_family_prior_surface() -> None:
    artifact = {
        "matched": {
            "MLQDS": {"query_local_utility_score": 0.1673},
            "uniform": {"query_local_utility_score": 0.142},
            "DouglasPeucker": {"query_local_utility_score": 0.1671},
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
                    "anchor_family_weights": {"density": 0.80},
                    "footprint_family_weights": {HISTORICAL_SMALL_LOCAL_FAMILY: 0.25},
                },
                "workload_signature": {
                    "anchor_family_counts": {"density": 7},
                    "footprint_family_counts": {HISTORICAL_SMALL_LOCAL_FAMILY: 12},
                },
            }
        },
        "predictability_audit": {
            "gate_pass": False,
            "gate_checks": {"spearman_min": False, "pr_auc_lift_over_base_rate": False},
            "prior_predictive_alignment_gate": {
                "gate_pass": True,
                "failed_checks": [],
            },
            "metrics": {
                "spearman": 0.11,
                "pr_auc_lift_over_base_rate": 1.23,
                "lift_at_5_percent": 1.20,
            },
            "best_prior_channel_by_head": {
                "segment_budget_target": {
                    "best_spearman": {
                        "channel": "endpoint_likelihood",
                        "value": 0.149,
                    },
                    "best_lift_at_5_percent": {
                        "channel": "spatiotemporal_query_hit_probability",
                        "value": 1.41,
                    },
                }
            },
            "prior_channel_by_head_predictability": {
                "segment_budget_target": {
                    "endpoint_likelihood": {
                        "available": True,
                        "spearman": 0.149,
                        "positive_target_spearman": -0.10,
                        "lift_at_5_percent": 1.15,
                        "pr_auc_lift_over_base_rate": 1.22,
                        "score_std": 0.01,
                        "target_mean": 0.09,
                    },
                    "spatiotemporal_query_hit_probability": {
                        "available": True,
                        "spearman": 0.126,
                        "positive_target_spearman": -0.05,
                        "lift_at_5_percent": 1.41,
                        "pr_auc_lift_over_base_rate": 1.20,
                        "score_std": 0.02,
                        "target_mean": 0.09,
                    },
                }
            },
        },
        "learning_causality_summary": {
            "learning_causality_gate_pass": False,
            "selection_causality_diagnostics": {"available": True},
            "causality_ablation_component_deltas": {
                "MLQDS_without_behavior_utility_head": {
                    "available": True,
                    "query_local_utility_delta": -0.0004,
                    "component_weighted_delta_sum": -0.0004,
                    "component_delta_residual": 0.0,
                    "top_positive_weighted_component_deltas": [
                        {
                            "component": "query_point_recall",
                            "component_delta": 0.001,
                            "weighted_delta": 0.0005,
                        }
                    ],
                    "top_negative_weighted_component_deltas": [
                        {
                            "component": "query_local_turn_change_coverage",
                            "component_delta": -0.004,
                            "weighted_delta": -0.0006,
                        }
                    ],
                }
            },
        },
        "training_target_diagnostics": {
            "query_local_utility_factorized": {
                "target_mode": "query_local_utility_factorized",
                "segment_budget_target_variant": "active_final_score",
                "segment_budget_target_aggregation": "top20_mean",
                "conditional_behavior_target_variant": "query_segment_local_behavior_utility",
                "conditional_behavior_target_base_source": (
                    "normalized_query_hit_conditioned_trajectory_change_times_"
                    "0.45_plus_0.35_segment_behavior_support_plus_0.20_segment_query_hit_support"
                ),
                "conditional_behavior_utility_training": "masked_to_query_hit_points",
                "behavior_change_highpass_quantile": 0.70,
                "conditional_behavior_target_alignment": {
                    "spearman_with_final_score": 0.32,
                    "spearman_with_query_hit_probability": 0.11,
                    "spearman_with_replacement_representative_value": 0.55,
                    "spearman_with_segment_budget_target": 0.03,
                    "spearman_with_path_length_support_target": 0.20,
                    "spearman_with_ship_query_evidence": -0.04,
                    "topk_replacement_representative_value_mass_recall_ranked_by_behavior": 0.80,
                    "topk_segment_budget_target_mass_recall_ranked_by_behavior": 0.40,
                },
                "final_success_allowed": True,
                "family_conditioned_target_trainability": {
                    "group_by": {
                        "footprint_family": {
                            HISTORICAL_SMALL_LOCAL_FAMILY: {
                                "ranker_alignment": {
                                    "segment_budget_target": {
                                        "spearman_with_ship_query_evidence": 0.12,
                                        "topk_ship_query_evidence_mass_recall": 0.34,
                                    }
                                },
                                "target_shapes": {
                                    "segment_budget_target": {
                                        "target_mean": 0.39,
                                        "target_std": 0.19,
                                    }
                                },
                            }
                        }
                    }
                },
            }
        },
        "training_fit_diagnostics": {
            "factorized_head_fit": {
                "conditional_behavior_utility": {
                    "kendall_tau": 0.04,
                    "topk_mass_recall_at_5_percent": 0.12,
                    "prediction_std": 0.002,
                    "target_std": 0.20,
                }
            },
            "family_conditioned_head_trainability": {
                "group_by": {
                    "footprint_family": {
                        HISTORICAL_SMALL_LOCAL_FAMILY: {
                            "query_count": 4,
                            "valid_hit_point_count": 64,
                            "weak_ship_evidence_heads": ["segment_budget_target"],
                            "head_fit": {
                                "segment_budget_target": {
                                    "spearman_with_family_ship_query_evidence": -0.08,
                                    "topk_family_ship_query_evidence_mass_recall": 0.28,
                                    "kendall_tau_with_head_target": 0.34,
                                    "topk_head_target_mass_recall": 0.61,
                                    "prediction_std": 0.05,
                                    "target_std": 0.19,
                                }
                            },
                        }
                    }
                }
            }
        },
        "selector_trace_diagnostics": {
            "eval_primary": {
                "retained_decision_marginal_query_local_utility_alignment": {
                    "available": True,
                    "candidate_count": 160,
                    "overall": {
                        "selector_score": {
                            "spearman": -0.04,
                            "top_minus_bottom_marginal": 0.00002,
                        },
                        "raw_score": {
                            "spearman": -0.06,
                            "top_minus_bottom_marginal": 0.00001,
                        },
                        "score_component_alignment": {
                            "head_probability_conditional_behavior_utility": {
                                "spearman": 0.25,
                                "top_minus_bottom_marginal": 0.0005,
                            },
                            "head_probability_query_hit_probability": {
                                "spearman": 0.35,
                                "top_minus_bottom_marginal": 0.0007,
                            },
                        },
                    },
                    "by_decision": {
                        "retained_removal_loss": {
                            "selector_score": {
                                "spearman": -0.03,
                                "top_minus_bottom_marginal": -0.00006,
                            },
                            "score_component_alignment": {
                                "head_probability_conditional_behavior_utility": {
                                    "spearman": -0.31,
                                    "top_minus_bottom_marginal": -0.0004,
                                }
                            },
                        }
                    },
                }
            }
        },
    }

    diagnostic = build_family_transfer_path_diagnostic([("checkpoint86", artifact)])

    summary = diagnostic["summary"]
    assert summary["decision"] == (
        "add_family_conditioned_prior_predictability_before_model_or_scoring_change"
    )
    assert summary["retained_marginal_alignment_layout"] == (
        "selector_trace_diagnostics.eval_primary."
        "retained_decision_marginal_query_local_utility_alignment"
    )
    blocked = summary["blocked_family_head_rows"]
    assert blocked[0]["family"] == HISTORICAL_SMALL_LOCAL_FAMILY
    assert blocked[0]["head"] == "segment_budget_target"
    assert blocked[0]["transfer_status"] == "fits_target_but_misorders_ship_evidence"
    retained = diagnostic["artifacts"][0]["retained_marginal_alignment"]
    assert retained["deprecated_learning_causality_layout_present"] is False
    active_alignment = retained["active_metric_score_component_alignment"]
    assert active_alignment["source_layout"] == (
        "selector_trace_diagnostics.eval_primary."
        "retained_decision_marginal_query_local_utility_alignment."
        "overall.score_component_alignment"
    )
    assert active_alignment["overall"]["conditional_behavior_utility"][
        "artifact_field"
    ] == "head_probability_conditional_behavior_utility"
    assert active_alignment["retained_removal_loss"]["conditional_behavior_utility"][
        "spearman"
    ] == pytest.approx(-0.31)
    assert summary["behavior_head_active_metric_alignment"][
        "retained_removal_spearman"
    ] == pytest.approx(-0.31)
    assert summary["current_metric_behavior_head_status"] == (
        "behavior_head_hurts_active_metric_ablation"
    )
    assert summary["without_behavior_head_query_local_utility_delta"] == pytest.approx(
        -0.0004
    )
    assert summary["without_behavior_head_top_negative_weighted_component_deltas"][0][
        "component"
    ] == "query_local_turn_change_coverage"
    semantics = diagnostic["artifacts"][0]["behavior_head_semantic_alignment"]
    assert semantics["target_reference_alignment"]["replacement_representative_value"][
        "spearman"
    ] == pytest.approx(0.55)
    assert semantics["strongest_target_reference_by_spearman"][
        "reference"
    ] == "replacement_representative_value"
    assert semantics["strongest_target_reference_by_spearman"][
        "spearman"
    ] == pytest.approx(0.55)
    assert semantics["no_behavior_head_component_tradeoff"][
        "delta_convention"
    ] == "primary_minus_ablation"
    assert "fitted_behavior_head_low_contrast" in semantics["semantic_statuses"]
    assert "behavior_target_weak_segment_budget_alignment" in summary[
        "behavior_head_semantic_statuses"
    ]
    prior = diagnostic["artifacts"][0]["predictability"][
        "aggregate_best_prior_channel_by_head"
    ]
    assert prior["segment_budget_target"]["best_spearman"] == pytest.approx(0.149)


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
        (row["group_key"], row["family"])
        for row in diagnostic["artifacts"][0]["focus_family_rows"]
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

    diagnostic = build_selector_marginal_calibration_diagnostic(
        [("checkpoint93", artifact)]
    )

    summary = diagnostic["summary"]
    assert summary["decision"] == (
        "diagnose_train_side_marginal_segment_calibration_not_promotion"
    )
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
    assert allocation["allocation_point_selection"][
        "same_allocation_length_only_gate_would_pass"
    ] is True


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
                "component_diagnosis": (
                    "extra_slots_score_dominated_not_length_support_aligned"
                ),
            },
            "allocation_point_selection_diagnostics": {
                "available": True,
                "primary_length_preservation": 0.67,
                "same_allocation_length_only_point_selection_preservation": 0.77,
                "same_allocation_length_only_gate_would_pass": True,
                "component_diagnosis": (
                    "point_selection_can_clear_length_with_current_allocation"
                ),
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
        "workload_distribution_comparison": {
            "workload_signature_gate": {"all_pass": True}
        },
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


def test_selection_eval_segment_teacher_transfer_diagnostic_blocks_direct_probe_on_weak_transfer() -> None:
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
        "workload_distribution_comparison": {
            "workload_signature_gate": {"all_pass": True}
        },
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
    assert summary["decision"] == (
        "diagnose_transfer_features_before_guarded_calibration_probe"
    )
    assert summary["decision_scope"] == "primary_artifact_last_input"
    result = diagnostic["artifacts"][0]
    assert result["decision"] == (
        "diagnose_transfer_features_before_guarded_calibration_probe"
    )
    overlap = result["target_overlap"]
    assert overlap["positive_overlap_count"] == 0
    assert overlap["top_0.1_overlap_fraction_of_selection"] == 0.0
    assert overlap["selection_eval_teacher_target_spearman"] < 0.0
    selection_features = result["selection_feature_alignment"]["feature_alignment"]
    eval_features = result["eval_feature_alignment"]["feature_alignment"]
    assert selection_features["segment_score"][
        "spearman_with_segment_teacher_target"
    ] > 0.0
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
    assert train_source_diagnostic["artifacts"][0]["selection_teacher"][
        "teacher_usage_split"
    ] == "train"


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
        "workload_distribution_comparison": {
            "workload_signature_gate": {"all_pass": True}
        },
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
    assert result["feature_coupling_summary"][
        "post_selection_positive_candidate_names"
    ] == ["learned_count_post_selection_coupled"]
    assert diagnostic["summary"]["decision"] == (
        "guarded_pre_selection_transfer_calibration_probe_admissible"
    )
    assert "segment_score" in result["feature_coupling_summary"][
        "admissible_candidate_names"
    ]
