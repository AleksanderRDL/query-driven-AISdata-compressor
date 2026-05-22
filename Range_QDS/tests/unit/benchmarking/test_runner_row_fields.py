"""Tests for benchmark row field extraction."""

from __future__ import annotations

import pytest
from row_field_fixtures import benchmark_row_run_json

from benchmarking.reporting.row_fields import _row_from_run


def test_benchmark_row_records_effective_child_torch_runtime(tmp_path) -> None:
    run_json = benchmark_row_run_json()

    row = _row_from_run(
        workload="range",
        run_label="custom_runtime",
        command=[
            "uv",
            "run",
            "--group",
            "dev",
            "--",
            "python",
            "-m",
            "orchestration.train_and_score",
        ],
        returncode=0,
        elapsed_seconds=10.0,
        run_dir=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        run_json_path=tmp_path / "example_run.json",
        timings={
            "phase_timings": [
                {"name": "train-model", "seconds": 6.0},
                {"name": "evaluate-matched", "seconds": 2.0},
            ],
            "epoch_timings": [],
            "inference_step_timings": [],
        },
        run_json=run_json,
    )

    assert row["float32_matmul_precision"] == "high"
    assert row["allow_tf32"] is True
    assert row["amp_mode"] == "bf16"
    assert row["child_float32_matmul_precision"] == "high"
    assert row["child_tf32_matmul_allowed"] is True
    assert row["child_amp_enabled"] is True
    assert row["child_amp_dtype"] == "bfloat16"
    assert row["model_type"] == "workload_blind_range"
    assert row["historical_prior_k"] == 7
    assert row["historical_prior_mmsi_weight"] == 2.5
    assert row["historical_prior_source_aggregation"] == "mean"
    assert row["historical_prior_source_count"] == 4
    assert row["historical_prior_stored_support_count"] == 1234
    assert row["compression_ratio"] == 0.05
    assert row["n_queries"] == 24
    assert row["max_queries"] == 512
    assert row["query_target_coverage"] == 0.30
    assert row["range_spatial_km"] == 2.2
    assert row["range_time_hours"] == 5.0
    assert row["mlqds_temporal_fraction"] == 0.0
    assert row["mlqds_diversity_bonus"] == 0.0
    assert row["mlqds_effective_diversity_bonus"] == 0.0
    assert row["mlqds_hybrid_mode"] == "global_budget"
    assert row["mlqds_stratified_center_weight"] == 0.45
    assert row["mlqds_min_learned_swaps"] == 1
    assert row["mlqds_score_mode"] == "rank"
    assert row["mlqds_score_temperature"] == 1.0
    assert row["mlqds_rank_confidence_weight"] == 0.15
    assert row["mlqds_range_geometry_blend"] == 0.0
    assert row["temporal_residual_label_mode"] == "none"
    assert row["range_label_mode"] == "point_f1"
    assert row["range_replicate_target_aggregation"] == "label_mean"
    assert row["range_temporal_target_blend"] == 0.15
    assert row["range_target_budget_weight_power"] == 0.50
    assert row["range_marginal_target_radius_scale"] == 0.65
    assert row["range_query_spine_fraction"] == 0.30
    assert row["range_query_spine_mass_mode"] == "query"
    assert row["range_query_residual_multiplier"] == 1.25
    assert row["range_query_residual_mass_mode"] == "point"
    assert row["range_set_utility_multiplier"] == 1.75
    assert row["range_set_utility_candidate_limit"] == 64
    assert row["range_set_utility_mass_mode"] == "query"
    assert row["loss_objective"] == "budget_topk"
    assert row["budget_loss_ratios"] == [0.01, 0.02, 0.05, 0.10]
    assert row["budget_loss_temperature"] == 0.10
    assert row["temporal_distribution_loss_weight"] == 0.05
    assert row["range_train_workload_replicates"] == 3
    assert row["range_time_domain_mode"] == "anchor_day"
    assert row["range_anchor_mode"] == "sparse"
    assert row["range_train_anchor_modes"] == ["mixed_density", "sparse"]
    assert row["range_train_footprints"] == ["1.1:2.5", "2.2:5.0"]
    assert row["range_max_coverage_overshoot"] == 0.02
    assert row["train_query_final_count"] == 28
    assert row["train_query_final_coverage"] == 0.21
    assert row["train_query_target_reached"] is True
    assert row["train_query_target_shortfall"] == 0.0
    assert row["train_query_target_overshoot"] == pytest.approx(0.01)
    assert row["train_query_target_missed_by_max_queries"] is False
    assert row["train_query_extra_after_target_reached"] == 4
    assert row["eval_query_final_count"] == 160
    assert row["eval_query_final_coverage"] == 0.215
    assert row["eval_query_target_reached"] is True
    assert row["eval_query_target_shortfall"] == 0.0
    assert row["eval_query_target_overshoot"] == pytest.approx(0.015)
    assert row["eval_query_target_missed_by_max_queries"] is False
    assert row["eval_query_target_reached_count"] == 31
    assert row["eval_query_extra_after_target_reached"] == 129
    assert row["eval_query_extra_after_target_fraction"] == pytest.approx(129 / 160)
    assert row["eval_query_floor_dominated"] is True
    assert row["eval_query_generation_stop_reason"] == "target_coverage_reached"
    assert row["eval_workload_near_duplicate_query_rate"] == 0.40
    assert row["eval_workload_best_baseline"] == "DouglasPeucker"
    assert row["selection_query_final_count"] == 21
    assert row["selection_query_target_reached"] is True
    assert row["selection_query_target_shortfall"] == 0.0
    assert row["selection_query_target_overshoot"] == pytest.approx(0.005)
    assert row["selection_query_target_missed_by_max_queries"] is False
    assert row["selection_query_extra_after_target_reached"] == 0
    assert row["selection_query_floor_dominated"] is False
    assert row["checkpoint_full_score_every"] == 3
    assert row["checkpoint_candidate_pool_size"] == 2
    assert row["best_selection_score"] == 0.42
    assert row["final_loss"] == 0.8
    assert row["final_kendall_tau_t0"] == 0.2
    assert row["final_pred_std"] == 0.2
    assert row["epoch_forward_mean_seconds"] == pytest.approx(0.3)
    assert row["epoch_loss_mean_seconds"] == pytest.approx(0.5)
    assert row["epoch_validation_score_mean_seconds"] == pytest.approx(0.35)
    assert row["runtime_bottleneck_phase"] == "train-model"
    assert row["runtime_bottleneck_seconds"] == 6.0
    assert row["runtime_bottleneck_fraction"] == pytest.approx(0.6)
    assert row["evaluate_matched_seconds"] == 2.0
    assert row["mlqds_latency_ms"] == 8.0
    assert row["mlqds_inference_only_latency_ms"] == 8.0
    assert row["mlqds_inference_only_latency_seconds"] == pytest.approx(0.008)
    assert row["single_cell_range_status"] == "beats_uniform_and_douglas_peucker"
    assert row["final_claim_status"] == "candidate_blocked_by_required_gates"
    assert row["final_success_allowed"] is False
    assert row["final_claim_blocking_gates"] == ["predictability_gate"]
    assert row["workload_stability_gate_pass"] is False
    assert row["workload_stability_failed_checks"] == ["train_r0:not_target_coverage_generation"]
    assert row["workload_stability_train_replicates"] == 1
    assert row["workload_stability_configured_target_coverage"] == 0.10
    assert row["support_overlap_gate_pass"] is True
    assert row["support_overlap_failed_checks"] == []
    assert row["support_eval_points_outside_train_prior_extent_fraction"] == 0.02
    assert row["support_sampled_prior_nonzero_fraction"] == 0.80
    assert row["support_primary_sampled_prior_nonzero_fraction"] == 0.60
    assert row["support_route_density_overlap"] == 0.70
    assert row["support_query_prior_support_overlap"] == 0.65
    assert row["support_train_eval_spatial_extent_intersection_fraction"] == 0.90
    assert row["global_sanity_gate_pass"] is True
    assert row["global_sanity_failed_checks"] == []
    assert row["global_sanity_endpoint_sanity"] == 1.0
    assert row["global_sanity_avg_sed_ratio_vs_uniform"] == pytest.approx(1.09)
    assert row["global_sanity_avg_length_preserved"] == pytest.approx(0.88)
    assert row["predictability_gate_pass"] is False
    assert row["predictability_spearman"] == 0.12
    assert row["predictability_lift_at_5_percent"] == 1.18
    assert row["predictability_pr_auc_lift_over_base_rate"] == 1.20
    assert row["prior_predictive_alignment_gate_pass"] is False
    assert row["prior_predictive_alignment_failed_checks"] == ["query_hit_spearman_below_min"]
    assert row["prior_positive_spearman_head_count"] == 1
    assert row["predictability_query_hit_spearman"] == 0.03
    assert row["predictability_segment_budget_lift_at_5_percent"] == 1.02
    assert row["prior_channel_query_mass_spearman"] == 0.04
    assert row["prior_channel_combined_score_lift_at_5_percent"] == 1.03
    assert row["workload_signature_gate_available"] is True
    assert row["workload_signature_gate_pass"] is False
    assert row["workload_signature_pair_count"] == 1
    assert row["workload_signature_failed_pairs"] == ["train"]
    assert row["train_eval_anchor_family_l1_distance"] == 0.16
    assert row["train_eval_point_hit_distribution_ks"] == 0.22
    assert row["train_eval_point_hit_fraction_distribution_ks"] == 0.12
    assert row["train_eval_ship_hit_fraction_distribution_ks"] == 0.08
    assert row["train_eval_query_count_delta"] == 8
    assert row["train_eval_query_count_relative_delta"] == 0.125
    assert row["train_signature_total_points"] == 1000
    assert row["eval_signature_total_points"] == 500
    assert row["train_signature_total_trajectories"] == 20
    assert row["eval_signature_total_trajectories"] == 10
    assert row["train_eval_point_hit_distribution_ks_proxy"] == 0.22
    assert row["train_eval_point_hit_distribution_used_quantile_proxy"] is False
    assert row["learning_causality_ablation_status"] == "partial"
    assert row["learning_causality_gate_pass"] is False
    assert row["learning_causality_failed_checks"] == ["shuffled_scores_should_lose"]
    assert row["causality_ablation_missing"] == ["MLQDS_without_segment_budget_head"]
    assert row["learned_controlled_retained_slot_fraction"] == 0.72
    assert row["planned_learned_controlled_retained_slot_fraction"] == 0.80
    assert row["actual_learned_controlled_retained_slot_fraction"] == 0.72
    assert row["trajectories_with_at_least_one_learned_decision"] == 5
    assert row["trajectories_with_zero_learned_decisions"] == 1
    assert row["segment_budget_entropy"] == 1.4
    assert row["segment_budget_entropy_normalized"] == 0.7
    assert row["selector_trace_retained_mask_matches_primary"] is True
    assert row["shuffled_score_ablation_delta"] == 0.04
    assert row["untrained_score_ablation_delta"] == 0.06
    assert row["shuffled_prior_field_ablation_delta"] == 0.05
    assert row["no_query_prior_field_ablation_delta"] == 0.03
    assert row["no_behavior_head_ablation_delta"] == 0.07
    assert row["no_segment_budget_head_ablation_delta"] == 0.08
    assert row["no_trajectory_fairness_preallocation_ablation_delta"] == 0.015
    assert row["shuffled_prior_retained_mask_jaccard"] == 0.82
    assert row["shuffled_prior_retained_symmetric_difference_count"] == 12
    assert row["no_query_prior_retained_mask_jaccard"] == 0.74
    assert row["no_query_prior_retained_symmetric_difference_count"] == 18
    assert row["no_behavior_retained_mask_jaccard"] == 0.97
    assert row["no_behavior_retained_symmetric_difference_count"] == 2
    assert row["no_segment_budget_retained_mask_jaccard"] == 0.51
    assert row["no_segment_budget_retained_symmetric_difference_count"] == 44
    assert row["no_geometry_tie_breaker_ablation_delta"] == -0.01
    assert row["no_geometry_retained_mask_jaccard"] == 0.62
    assert row["no_geometry_retained_symmetric_difference_count"] == 30
    assert row["no_segment_length_support_allocation_ablation_delta"] == 0.004
    assert row["no_segment_length_support_allocation_retained_mask_jaccard"] == 0.91
    assert row["no_segment_length_support_allocation_retained_symmetric_difference_count"] == 8
    assert row["learned_segment_geometry_gain_weight"] == 0.12
    assert row["learned_segment_allocation_length_support_weight"] == 0.4
    assert row["learned_segment_allocation_weight_floor"] == 0.25
    assert row["learned_segment_score_blend_weight"] == 0.05
    assert row["learned_segment_transfer_calibration_mode"] == (
        "segment_score_allocation_weight_zblend"
    )
    assert row["learned_segment_fairness_preallocation_enabled"] is True
    assert row["learned_segment_length_repair_fraction"] == 0.25
    assert row["learned_segment_length_repair_score_protection_fraction"] == 0.15
    assert row["learned_segment_length_support_blend_weight"] == 1.0
    assert row["learning_causality_min_material_delta"] == 0.005
    assert row["learning_causality_shuffled_fraction_of_uniform_gap_min"] == 0.60
    assert row["learning_causality_mlqds_uniform_gap"] == 0.05
    assert row["learning_causality_delta_thresholds"]["shuffled_scores_should_lose"] == 0.03
    assert row["segment_budget_head_ablation_mode"] == "neutral_constant_segment_scores"
    assert row["prior_sample_gate_pass"] is False
    assert row["prior_sample_gate_failures"] == ["sampled_query_prior_features_all_zero"]
    assert row["shuffled_prior_sampled_inputs_changed"] is False
    assert row["shuffled_prior_sampled_primary_nonzero_fraction"] == 0.0
    assert row["shuffled_prior_sampled_ablation_nonzero_fraction"] == 0.01
    assert row["shuffled_prior_sampled_mean_abs_feature_delta"] == 0.002
    assert row["shuffled_prior_sampled_max_abs_feature_delta"] == 0.10
    assert row["shuffled_prior_sampled_outside_extent_fraction"] == 0.75
    assert row["shuffled_prior_model_inputs_changed"] is True
    assert row["shuffled_prior_model_input_mean_abs_feature_delta"] == 0.003
    assert row["shuffled_prior_normalized_model_inputs_changed"] is True
    assert row["shuffled_prior_normalized_model_mean_abs_feature_delta"] == 0.004
    assert row["shuffled_prior_head_logits_changed"] is True
    assert row["shuffled_prior_head_logit_mean_abs_delta"] == 0.006
    assert row["shuffled_prior_head_probability_mean_abs_delta"] == 0.0015
    assert row["shuffled_prior_score_output_mean_abs_delta"] == 0.012
    assert row["shuffled_prior_score_output_max_abs_delta"] == 0.08
    assert row["shuffled_prior_score_output_topk_jaccard_at_retained_count"] == 0.72
    assert row["no_prior_sampled_primary_nonzero_fraction"] == 0.0
    assert row["no_prior_sampled_mean_abs_feature_delta"] == 0.0
    assert row["no_prior_sampled_outside_extent_fraction"] == 0.75
    assert row["no_prior_model_inputs_changed"] is False
    assert row["no_prior_model_input_mean_abs_feature_delta"] == 0.0
    assert row["no_prior_normalized_model_inputs_changed"] is False
    assert row["no_prior_normalized_model_mean_abs_feature_delta"] == 0.0
    assert row["no_prior_head_logits_changed"] is False
    assert row["no_prior_head_logit_mean_abs_delta"] == 0.0
    assert row["no_prior_head_probability_mean_abs_delta"] == 0.0
    assert row["no_prior_score_output_mean_abs_delta"] == 0.0
    assert row["no_prior_score_output_max_abs_delta"] == 0.0
    assert row["no_prior_score_output_topk_jaccard_at_retained_count"] == 1.0
    assert row["workload_blind_candidate"] is True
    assert row["selector_claim_status"] == "model_has_material_budget"
    assert row["selector_claim_has_material_learned_budget"] is True
    assert row["workload_blind_protocol_enabled"] is True
    assert row["primary_masks_frozen_before_eval_query_scoring"] is True
    assert row["audit_masks_frozen_before_eval_query_scoring"] is True
    assert row["eval_geometry_blend_allowed"] is False
    assert row["beats_uniform_query_local_utility"] is True
    assert row["beats_douglas_peucker_query_local_utility"] is True
    assert row["beats_temporal_random_fill_query_local_utility"] is True
    assert row["audit_compression_ratio_count"] == 3
    assert row["audit_low_compression_ratio_count"] == 2
    assert row["audit_beats_uniform_query_local_utility_count"] == 2
    assert row["audit_beats_douglas_peucker_query_local_utility_count"] == 3
    assert row["audit_beats_temporal_random_fill_query_local_utility_count"] == 1
    assert row["audit_beats_both_query_local_utility_count"] == 2
    assert row["audit_low_beats_uniform_query_local_utility_count"] == 1
    assert row["audit_low_beats_temporal_random_fill_query_local_utility_count"] == 1
    assert row["audit_low_beats_both_query_local_utility_count"] == 1
    assert row["audit_beats_uniform_query_local_utility_count"] == 2
    assert row["audit_beats_douglas_peucker_query_local_utility_count"] == 3
    assert row["audit_low_beats_uniform_query_local_utility_count"] == 1
    assert row["audit_min_low_vs_uniform_query_local_utility"] == pytest.approx(-0.02)
    assert row["audit_mean_low_vs_uniform_query_local_utility"] == pytest.approx(0.025)
    assert row["audit_min_low_vs_temporal_random_fill_query_local_utility"] == pytest.approx(0.0)
    assert row["audit_mean_vs_temporal_random_fill_query_local_utility"] == pytest.approx(
        0.05 / 3.0
    )
    assert row["audit_mean_low_vs_temporal_random_fill_query_local_utility"] == pytest.approx(
        0.025
    )
    assert row["audit_ratio_0p0100_compression_ratio"] == pytest.approx(0.01)
    assert row["audit_ratio_0p0100_mlqds_query_local_utility"] == pytest.approx(0.11)
    assert row["audit_ratio_0p0100_uniform_query_local_utility"] == pytest.approx(0.13)
    assert row["audit_ratio_0p0100_mlqds_vs_uniform_query_local_utility"] == pytest.approx(-0.02)
    assert row["audit_ratio_0p0100_douglas_peucker_query_local_utility"] == pytest.approx(0.10)
    assert row["audit_ratio_0p0100_temporal_random_fill_query_local_utility"] == pytest.approx(0.11)
    assert row["audit_ratio_0p0100_mlqds_vs_douglas_peucker_query_local_utility"] == pytest.approx(
        0.01
    )
    assert row[
        "audit_ratio_0p0100_mlqds_vs_temporal_random_fill_query_local_utility"
    ] == pytest.approx(0.0)
    assert row["audit_ratio_0p0500_mlqds_vs_uniform_query_local_utility"] == pytest.approx(0.07)
    assert row[
        "audit_ratio_0p1000_mlqds_vs_temporal_random_fill_query_local_utility"
    ] == pytest.approx(0.0)
    assert row["range_boundary_prior_weight"] == 0.0
    assert row["range_boundary_prior_enabled"] is False
    assert row["teacher_distillation_enabled"] is True
    assert row["teacher_distillation_mode"] == "retained_frequency"
    assert row["teacher_model_type"] == "range_aware"
    assert row["teacher_replicate_count"] == 4
    assert row["teacher_positive_label_fraction"] == 0.25
    assert row["teacher_positive_label_mass"] == 16.0
    assert row["train_positive_label_mass"] == 12.5
    assert row["train_label_mass_basis"] == "pre_clamp_component_contributions"
    assert row["train_label_mass_range_point_f1"] == 0.22
    assert row["train_label_mass_range_turn_coverage"] == 0.08
    assert row["train_target_positive_label_mass"] == 11.0
    assert row["train_target_budget_ratio"] == 0.05
    assert row["train_target_effective_fill_budget_ratio"] == 0.041
    assert row["train_target_temporal_base_label_mass_fraction"] == 0.35
    assert row["train_target_residual_label_mass_fraction"] == 0.65
    assert row["train_target_residual_positive_label_fraction"] == 0.20
    assert row["train_fit_score_target_kendall_tau"] == 0.31
    assert row["train_fit_matched_mlqds_target_recall"] == 0.74
    assert row["train_fit_matched_uniform_target_recall"] == 0.62
    assert row["train_fit_matched_mlqds_vs_uniform_target_recall"] == 0.12
    assert row["train_fit_low_budget_mean_mlqds_vs_uniform_target_recall"] == -0.04
    assert row["range_target_transform_mode"] == "local_swap_utility_frequency"
    assert row["range_target_transform_positive_label_count"] == 17
    assert row["range_target_transform_positive_label_mass"] == 3.5
    assert row["local_swap_utility_scored_candidate_count"] == 40
    assert row["local_swap_utility_positive_gain_candidate_count"] == 11
    assert row["local_swap_utility_selected_count"] == 7
    assert row["local_swap_utility_selected_gain_mass"] == 1.25
    assert row["local_swap_utility_source_positive_mass"] == 3.5
    assert row["local_swap_gain_cost_scored_candidate_count"] == 44
    assert row["local_swap_gain_cost_positive_net_gain_count"] == 12
    assert row["local_swap_gain_cost_selected_count"] == 8
    assert row["local_swap_gain_cost_selected_candidate_value_mass"] == 1.50
    assert row["local_swap_gain_cost_selected_removal_cost_mass"] == 0.40
    assert row["local_swap_gain_cost_source_positive_mass"] == 3.75
    assert row["mlqds_primary_metric"] == "query_local_utility"
    assert row["mlqds_primary_score"] == 0.46
    assert row["mlqds_aggregate_f1"] == 0.40
    assert row["mlqds_query_point_recall"] == 0.31
    assert row["mlqds_range_point_f1"] == 0.40
    assert row["mlqds_query_local_utility_score"] == 0.46
    assert row["uniform_range_point_f1"] == 0.35
    assert row["uniform_query_point_recall"] == 0.26
    assert row["uniform_query_local_utility_score"] == 0.39
    assert row["douglas_peucker_range_point_f1"] == 0.36
    assert row["douglas_peucker_query_point_recall"] == 0.29
    assert row["douglas_peucker_query_local_utility_score"] == 0.41
    assert row["mlqds_avg_sed_km"] == 0.60
    assert row["uniform_avg_sed_km"] == 0.55
    assert row["douglas_peucker_avg_sed_km"] == 0.45
    assert row["mlqds_avg_length_preserved"] == 0.88
    assert row["uniform_avg_length_preserved"] == 0.91
    assert row["douglas_peucker_avg_length_preserved"] == 0.93
    assert row["mlqds_vs_uniform_avg_sed_km"] == pytest.approx(0.05)
    assert row["mlqds_vs_uniform_avg_ped_km"] == pytest.approx(0.02)
    assert row["mlqds_vs_uniform_avg_length_preserved"] == pytest.approx(-0.03)
    assert row["mlqds_range_gap_min_coverage"] == 0.36
    assert row["mlqds_range_turn_coverage"] == 0.52
    assert row["mlqds_range_query_local_interpolation_fidelity"] == 0.62
    assert row["mlqds_vs_uniform_range_gap_min_coverage"] == pytest.approx(-0.12)
    assert row["mlqds_vs_uniform_range_turn_coverage"] == pytest.approx(0.01)
    assert row["mlqds_vs_uniform_range_query_local_interpolation_fidelity"] == pytest.approx(0.04)
    assert row["worst_uniform_component_delta_metric"] == "mlqds_vs_uniform_range_gap_min_coverage"
    assert row["worst_uniform_component_delta"] == pytest.approx(-0.12)
    assert row["temporal_random_fill_range_point_f1"] == 0.38
    assert row["temporal_random_fill_query_local_utility_score"] == 0.41
    assert row["temporal_oracle_fill_range_point_f1"] == 0.55
    assert row["temporal_oracle_fill_query_local_utility_score"] == 0.70
    assert row["mlqds_vs_temporal_random_fill_query_local_utility"] == pytest.approx(0.05)
    assert row["temporal_oracle_fill_gap_query_local_utility"] == pytest.approx(0.24)
    assert row["collapse_warning_any"] is True
    assert row["collapse_warning_count"] == 1
    assert row["best_epoch_collapse_warning"] is False
    assert row["min_pred_std"] == 0.0
    assert row["best_epoch_pred_std"] == 0.2
    assert row["oracle_kind"] == "additive_label_greedy"
    assert row["oracle_exact_optimum"] is False
    assert row["mlqds_vs_uniform_range_point_f1"] == pytest.approx(0.05)
    assert row["mlqds_vs_uniform_query_point_recall"] == pytest.approx(0.05)
    assert row["mlqds_vs_douglas_peucker_range_point_f1"] == pytest.approx(0.04)
    assert row["mlqds_vs_douglas_peucker_query_point_recall"] == pytest.approx(0.02)
    assert row["mlqds_vs_uniform_query_local_utility"] == pytest.approx(0.07)
    assert row["mlqds_vs_douglas_peucker_query_local_utility"] == pytest.approx(0.05)
