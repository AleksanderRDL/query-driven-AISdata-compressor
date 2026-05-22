"""Derived query-driven diagnostic artifact tests."""

from __future__ import annotations

import pytest
import torch

from orchestration.diagnostics.family_transfer_path_diagnostic import (
    build_family_transfer_path_diagnostic,
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
                    },
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
                            "query_local_utility_delta": -0.03,
                            "range_component_deltas": {"query_point_recall": -0.20},
                        },
                        "sparse_background_control": {
                            "query_count": 5,
                            "query_local_score_delta": -0.01,
                            "query_local_utility_delta": -0.01,
                            "range_component_deltas": {"query_point_recall": -0.10},
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
    assert blocking[0]["query_local_score_delta"] == pytest.approx(-0.02)
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
    assert diagnostic["summary"]["blocker_preserving_candidate_status"] == "still_blocked"
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
            },
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
    assert (
        active_alignment["overall"]["conditional_behavior_utility"]["artifact_field"]
        == "head_probability_conditional_behavior_utility"
    )
    assert active_alignment["retained_removal_loss"]["conditional_behavior_utility"][
        "spearman"
    ] == pytest.approx(-0.31)
    assert summary["behavior_head_active_metric_alignment"][
        "retained_removal_spearman"
    ] == pytest.approx(-0.31)
    assert summary["current_metric_behavior_head_status"] == (
        "behavior_head_hurts_active_metric_ablation"
    )
    assert summary["without_behavior_head_query_local_utility_delta"] == pytest.approx(-0.0004)
    assert (
        summary["without_behavior_head_top_negative_weighted_component_deltas"][0]["component"]
        == "query_local_turn_change_coverage"
    )
    semantics = diagnostic["artifacts"][0]["behavior_head_semantic_alignment"]
    assert semantics["target_reference_alignment"]["replacement_representative_value"][
        "spearman"
    ] == pytest.approx(0.55)
    assert (
        semantics["strongest_target_reference_by_spearman"]["reference"]
        == "replacement_representative_value"
    )
    assert semantics["strongest_target_reference_by_spearman"]["spearman"] == pytest.approx(0.55)
    assert (
        semantics["no_behavior_head_component_tradeoff"]["delta_convention"]
        == "primary_minus_ablation"
    )
    assert "fitted_behavior_head_low_contrast" in semantics["semantic_statuses"]
    assert (
        "behavior_target_weak_segment_budget_alignment"
        in summary["behavior_head_semantic_statuses"]
    )
    prior = diagnostic["artifacts"][0]["predictability"]["aggregate_best_prior_channel_by_head"]
    assert prior["segment_budget_target"]["best_spearman"] == pytest.approx(0.149)
