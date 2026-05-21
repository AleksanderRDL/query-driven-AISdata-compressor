"""Semantic causality diagnostic tests."""

from __future__ import annotations

import pytest

from orchestration.diagnostics.semantic_causality_diagnostic import (
    build_semantic_causality_diagnostic,
)


def _score_summary(score: float, spearman: float) -> dict[str, float | bool]:
    return {
        "available": True,
        "count": 10,
        "spearman": spearman,
        "pearson": spearman,
        "top_quartile_mean_marginal": score,
        "bottom_quartile_mean_marginal": score / 2.0,
        "top_minus_bottom_marginal": score / 2.0,
        "value_min": 0.0,
        "value_max": 1.0,
    }


def _prior_chain(*, mask_changed: bool) -> dict[str, object]:
    return {
        "available": True,
        "diagnostic_chain": [
            "sampled_prior_features",
            "model_prior_features",
            "head_output",
            "raw_prediction",
            "score_output",
            "marginal_row_delta_path",
            "retained_mask",
        ],
        "sampled_prior_features": {
            "available": True,
            "sampled_inputs_changed": True,
            "mean_abs_feature_delta": 0.02,
            "max_abs_feature_delta": 0.1,
        },
        "model_prior_features": {
            "available": True,
            "disabled_prior_fields": ["route_density_prior"],
            "model_prior_feature_transform": "identity_probability",
            "normalized_model_prior_features": {
                "available": True,
                "sampled_inputs_changed": True,
                "mean_abs_feature_delta": 0.02,
                "max_abs_feature_delta": 0.1,
            },
            "model_input_prior_features": {
                "available": True,
                "sampled_inputs_changed": True,
                "mean_abs_feature_delta": 0.02,
                "max_abs_feature_delta": 0.1,
            },
        },
        "head_output": {
            "available": True,
            "head_logits_changed": True,
            "head_probabilities_changed": True,
            "mean_abs_head_logit_delta": 0.0001,
            "mean_abs_head_probability_delta": 0.00001,
            "probability": {"per_feature": {}},
            "logit": {"per_feature": {}},
        },
        "raw_prediction": {
            "available": True,
            "mean_abs_score_delta": 0.0002,
            "retained_mask_changed": mask_changed,
            "score_topk_jaccard_at_retained_count": 1.0,
        },
        "score_output": {
            "available": True,
            "mean_abs_score_delta": 0.0007,
            "retained_mask_changed": mask_changed,
            "score_topk_jaccard_at_retained_count": 1.0,
        },
        "retained_mask": {
            "available": True,
            "retained_mask_changed": mask_changed,
            "retained_mask_jaccard": 1.0,
        },
        "score_rank_margin_boundary": {
            "available": True,
            "classification": "prior_score_deltas_below_topk_rank_margin",
            "topk_score_boundary": {
                "topk_boundary_margin": 0.20,
                "max_abs_score_delta_to_topk_boundary_margin": 0.25,
                "score_delta_crosses_topk_boundary": False,
            },
            "marginal_row_score_delta_alignment": {
                "score_delta_to_marginal_spearman": 0.10,
                "top_marginal_mean_score_delta": 0.01,
                "missed_high_marginal_mean_score_delta": 0.02,
                "under_ranked_high_marginal_mean_score_delta": 0.03,
                "classification": "prior_delta_not_obviously_wrong_way_for_marginal_rows",
            },
        },
        "marginal_row_delta_path": {
            "available": True,
            "classification": "raw_score_suppresses_positive_head_probability_delta",
            "row_count": 4,
            "stage_available": {
                "head_output": True,
                "raw_prediction": True,
                "score_output": True,
                "segment_score": True,
                "retained_mask": True,
            },
            "groups": {
                "top_marginal": {
                    "score_output_mean_delta": 0.0,
                    "raw_prediction_mean_delta": 0.0,
                    "segment_score_mean_delta": 0.0,
                    "factorized_composition_available": True,
                    "factorized_composed_score_mean_delta": -0.01,
                    "factorized_composed_logit_mean_delta": -0.02,
                    "factorized_raw_prediction_delta_residual_mean": 0.001,
                    "factorized_contribution_mean_delta": {
                        "query_hit_branch_shapley": -0.03,
                        "behavior_branch_shapley": -0.01,
                        "replacement_modulation_shapley": 0.02,
                        "boundary_bonus": 0.004,
                        "clamp": 0.0,
                    },
                    "factorized_contribution_positive_delta_fraction": {
                        "query_hit_branch_shapley": 0.0,
                        "behavior_branch_shapley": 0.0,
                        "replacement_modulation_shapley": 1.0,
                        "boundary_bonus": 1.0,
                        "clamp": 0.0,
                    },
                    "factorized_most_negative_mean_contribution": {
                        "name": "query_hit_branch_shapley",
                        "delta": -0.03,
                    },
                    "factorized_most_positive_mean_contribution": {
                        "name": "replacement_modulation_shapley",
                        "delta": 0.02,
                    },
                    "max_head_probability_mean_delta": 0.02,
                    "max_head_probability_mean_delta_head": "query_hit_probability",
                    "max_head_logit_mean_delta": 0.03,
                    "max_head_logit_mean_delta_head": "query_hit_probability",
                },
                "missed_high_marginal": {
                    "score_output_mean_delta": 0.0,
                    "raw_prediction_mean_delta": 0.0,
                },
                "under_ranked_high_marginal": {
                    "score_output_mean_delta": 0.0,
                    "raw_prediction_mean_delta": 0.0,
                },
            },
        },
    }


def test_semantic_causality_diagnostic_classifies_current_blockers() -> None:
    artifact = {
        "matched": {
            "MLQDS": {"query_local_utility_score": 0.143},
            "uniform": {"query_local_utility_score": 0.125},
            "DouglasPeucker": {"query_local_utility_score": 0.115},
        },
        "training_target_diagnostics": {
            "query_local_utility_factorized": {
                "conditional_behavior_utility_training": "masked_to_query_hit_points",
                "conditional_behavior_target_variant": "query_segment_local_behavior_utility",
                "positive_point_count_by_head": {"conditional_behavior_utility": 2271},
                "positive_fraction_by_head": {"conditional_behavior_utility": 0.305},
                "positive_label_mass_by_head": {"conditional_behavior_utility": 628.1},
                "support_fraction_by_threshold_by_head": {
                    "conditional_behavior_utility": {"gt_0.10": 0.234}
                },
                "label_mass_by_segment_position": {
                    "conditional_behavior_utility": {"middle": 0.641}
                },
                "conditional_behavior_target_alignment": {
                    "spearman_with_final_score": 0.31,
                    "spearman_with_replacement_representative_value": 0.55,
                    "spearman_with_segment_budget_target": 0.01,
                },
                "conditional_behavior_candidate_alignment": {"current": {}},
                "family_conditioned_target_trainability": {"available": True},
            }
        },
        "training_fit_diagnostics": {
            "query_hit_probability_head_tau": 0.37,
            "replacement_representative_value_head_tau": 0.02,
            "segment_budget_target_head_tau": 0.35,
            "factorized_final_score_tau": 0.36,
            "factorized_head_fit": {
                "conditional_behavior_utility": {
                    "valid_point_count": 7448,
                    "positive_target_count": 2271,
                    "positive_target_fraction": 0.305,
                    "target_mean": 0.084,
                    "target_std": 0.166,
                    "prediction_mean": 0.082,
                    "prediction_std": 0.0026,
                    "kendall_tau": 0.025,
                    "topk_mass_recall_at_5_percent": 0.129,
                }
            },
            "prior_feature_learning_signal": {
                "prior_feature_learning_diagnostics_available": True,
                "classification": "prior_target_signal_available_but_trained_heads_invariant",
                "prior_to_head_transfer_sensitivity": {
                    "available": True,
                    "classification_counts": {
                        "output_layer_suppresses_prior_direction": 1
                    },
                    "per_head": {
                        "query_hit_probability": {
                            "classification": "output_layer_suppresses_prior_direction",
                            "output_layer_alignment": {
                                "available": True,
                                "final_weight_to_hidden_delta_abs_cosine_mean": 0.04,
                                "projected_hidden_delta_l2_to_hidden_delta_l2": 0.01,
                                "target_to_logit_delta_spearman": -0.12,
                                "bce_descent_alignment_mean": -0.0003,
                                "bce_descent_alignment_positive_fraction": 0.31,
                            },
                            "configured_loss_gradient_alignment": {
                                "available": True,
                                "loss_scope": "configured_factorized_loss_window_segment_proxy",
                                "descent_alignment_mean": -0.00002,
                                "descent_alignment_positive_fraction": 0.44,
                                "logit_gradient_abs_mean": 0.0007,
                            },
                        }
                    },
                    "prior_channel_direction_decomposition": {
                        "available": True,
                        "diagnostic_only": True,
                        "channel_count": 1,
                        "classification_counts": {"wrong_way": 1},
                        "by_head": {
                            "query_hit_probability": {
                                "channel_count": 1,
                                "target_aligned_channels": [],
                                "wrong_way_channels": ["spatial_query_hit_probability"],
                                "weak_or_flat_channels": [],
                                "rank_alignment_unavailable_channels": [],
                                "min_target_to_logit_delta_spearman": -0.17,
                                "max_target_to_logit_delta_spearman": -0.17,
                                "strongest_aligned_channel": {
                                    "channel": "spatial_query_hit_probability",
                                    "value": -0.17,
                                },
                                "strongest_wrong_way_channel": {
                                    "channel": "spatial_query_hit_probability",
                                    "value": -0.17,
                                },
                            }
                        },
                        "per_channel": {
                            "spatial_query_hit_probability": {
                                "available": True,
                                "per_head": {
                                    "query_hit_probability": {
                                        "available": True,
                                        "classification": "wrong_way",
                                        "output_layer_alignment": {
                                            "available": True,
                                            "target_to_logit_delta_spearman": -0.17,
                                            "target_to_projected_hidden_delta_spearman": -0.15,
                                            "projected_hidden_delta_l2_to_hidden_delta_l2": 0.03,
                                            "bce_descent_alignment_mean": -0.0004,
                                        },
                                    }
                                },
                            }
                        },
                    },
                },
            },
        },
        "learning_causality_summary": {
            "learning_causality_failed_checks": [
                "shuffled_prior_fields_should_lose",
                "without_query_prior_features_should_lose",
                "without_behavior_utility_head_should_lose",
            ],
            "learning_causality_gate_pass": False,
            "no_behavior_head_ablation_delta": 0.0015,
            "shuffled_prior_field_ablation_delta": 0.0,
            "without_query_prior_features_delta": 0.0,
            "prior_sample_gate_pass": True,
            "prior_sample_gate_failures": [],
            "learning_causality_delta_gate": {
                "thresholds": {
                    "without_behavior_utility_head_should_lose": 0.005,
                    "shuffled_prior_fields_should_lose": 0.005,
                    "without_query_prior_features_should_lose": 0.005,
                }
            },
            "prior_sensitivity_diagnostics": {
                "shuffled_prior_fields": _prior_chain(mask_changed=False),
                "without_query_prior_features": _prior_chain(mask_changed=False),
            },
            "prior_channel_ablation_diagnostics": {},
            "causality_ablation_scores": {
                "MLQDS_without_segment_budget_head": 0.133,
                "MLQDS_without_segment_budget_allocation_only": 0.133,
                "MLQDS_uniform_segment_allocation_only_diagnostic": 0.129,
                "MLQDS_point_score_allocation_diagnostic": 0.145,
                "MLQDS_segment_allocation_top25_band_diagnostic": 0.138,
                "MLQDS_segment_allocation_top50_band_diagnostic": 0.144,
                "MLQDS_segment_allocation_quartile_band_diagnostic": 0.144,
                "MLQDS_without_segment_budget_point_blend_only": 0.141,
                "MLQDS_without_segment_length_support_allocation": 0.141,
                "MLQDS_path_length_support_allocation_only_diagnostic": 0.133,
                "MLQDS_behavior_utility_allocation_only_diagnostic": 0.133,
            },
        },
        "selector_trace_diagnostics": {
            "eval_primary": {
                "segment_allocation_alignment_diagnostics": {
                    "segment_score_to_allocation_spearman": 0.84,
                    "allocation_weight_to_allocation_spearman": 0.84,
                    "length_support_to_allocation_spearman": 0.03,
                    "component_diagnosis": "extra_slots_score_dominated_not_length_support_aligned",
                },
                "retained_decision_marginal_query_local_utility_alignment": {
                    "available": True,
                    "overall": {
                        "raw_score": _score_summary(0.001, 0.28),
                        "selector_score": _score_summary(0.001, 0.29),
                        "segment_score": _score_summary(-0.001, -0.08),
                        "score_component_alignment": {
                            "head_probability_conditional_behavior_utility": _score_summary(
                                -0.001, -0.05
                            ),
                            "factorized_behavior_branch": _score_summary(-0.001, -0.05),
                            "head_probability_query_hit_probability": _score_summary(
                                0.001, 0.28
                            ),
                            "head_probability_replacement_representative_value": _score_summary(
                                0.0, -0.04
                            ),
                            "head_probability_segment_budget_target": _score_summary(
                                -0.001, -0.08
                            ),
                            "head_probability_path_length_support_target": _score_summary(
                                0.001, 0.30
                            ),
                        },
                    },
                    "rows": [
                        {
                            "point_index": 5,
                            "trajectory_index": 1,
                            "source": "learned",
                            "decision": "retained_removal_loss",
                            "marginal_query_local_utility": 0.1,
                            "raw_score": 0.2,
                            "selector_score": 0.3,
                            "segment_score": -1.0,
                            "selector_stage_state": {"final_retained": True},
                            "selector_segment_context": {"segment_index": 7},
                            "head_probabilities": {
                                "query_hit_probability": 0.01,
                                "conditional_behavior_utility": 0.08,
                                "replacement_representative_value": 0.23,
                                "segment_budget_target": 0.16,
                            },
                        }
                    ],
                },
            }
        },
        "target_segment_oracle_alignment_audit": {
            "source_alignment": {
                "point_score_top20_mean": {"spearman_vs_oracle_mass": 0.93},
                "target_head_query_hit_probability_top20_mean": {
                    "spearman_vs_oracle_mass": 0.95
                },
                "target_head_segment_budget_target_top20_mean": {
                    "spearman_vs_oracle_mass": 0.93
                },
                "target_head_path_length_support_target_top20_mean": {
                    "spearman_vs_oracle_mass": 0.04
                },
            }
        },
        "train_marginal_causality_diagnostics": {
            "selection_retained_decision_marginal_teacher": {
                "available": True,
                "teacher_value_variation": 0.001,
                "separated_marginal_teacher_summary": {
                    "available": True,
                    "candidate_for_train_side_teacher": True,
                },
            }
        },
    }

    diagnostic = build_semantic_causality_diagnostic([("current", artifact)])
    decision = diagnostic["summary"]["decision"]

    assert decision["behavior_failure"] == "target has signal but head does not learn it"
    assert decision["prior_failure"] == "model ignores prior inputs"
    assert (
        decision["segment_failure"]
        == "allocation scoring and point-selection scoring are mixed incorrectly"
    )
    assert decision["artifact_lacks_full_required_rows"] is True
    rank_margin = diagnostic["artifacts"][0]["query_prior_materiality"]["chains"][
        "shuffled_prior_fields"
    ]["score_rank_margin_boundary"]
    assert rank_margin["classification"] == "prior_score_deltas_below_topk_rank_margin"
    assert rank_margin["topk_boundary_margin"] == pytest.approx(0.20)
    row_path = diagnostic["artifacts"][0]["query_prior_materiality"]["chains"][
        "shuffled_prior_fields"
    ]["marginal_row_delta_path"]
    assert row_path["classification"] == "raw_score_suppresses_positive_head_probability_delta"
    assert row_path["top_marginal_max_head_probability_mean_delta"] == pytest.approx(0.02)
    assert row_path["top_marginal_max_head_probability_mean_delta_head"] == (
        "query_hit_probability"
    )
    composition = row_path["top_marginal_factorized_composition"]
    assert composition["available"] is True
    assert composition["composed_score_mean_delta"] == pytest.approx(-0.01)
    assert composition["most_negative_mean_contribution_name"] == (
        "query_hit_branch_shapley"
    )
    assert composition["most_negative_mean_contribution_delta"] == pytest.approx(-0.03)
    assert composition["most_positive_mean_contribution_name"] == (
        "replacement_modulation_shapley"
    )
    assert rank_margin["missed_high_marginal_mean_score_delta"] == pytest.approx(0.02)
    transfer = diagnostic["artifacts"][0]["query_prior_materiality"][
        "training_prior_learning_signal"
    ]["prior_to_head_transfer_sensitivity"]
    query_alignment = transfer["per_head"]["query_hit_probability"]["output_layer_alignment"]
    assert query_alignment["final_weight_to_hidden_delta_abs_cosine_mean"] == pytest.approx(0.04)
    assert query_alignment["bce_descent_alignment_mean"] == pytest.approx(-0.0003)
    loss_alignment = transfer["per_head"]["query_hit_probability"][
        "configured_loss_gradient_alignment"
    ]
    assert loss_alignment["descent_alignment_mean"] == pytest.approx(-0.00002)
    channel_decomposition = transfer["prior_channel_direction_decomposition"]
    assert channel_decomposition["classification_counts"] == {"wrong_way": 1}
    channel_query = channel_decomposition["per_channel"]["spatial_query_hit_probability"][
        "per_head"
    ]["query_hit_probability"]
    assert channel_query["classification"] == "wrong_way"
    assert channel_query["output_layer_alignment"][
        "target_to_logit_delta_spearman"
    ] == pytest.approx(-0.17)
