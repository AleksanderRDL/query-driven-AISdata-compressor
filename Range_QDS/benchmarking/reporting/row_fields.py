"""Benchmark child-run row construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarking.common import as_float
from benchmarking.reporting.audit_extractors import (
    _audit_summary,
    _data_source_row_fields,
    _selector_budget_row,
    _selector_low_budget_summary,
    _target_budget_row,
    _workload_generation_fields,
)
from benchmarking.reporting.metrics import (
    RANGE_COMPONENT_KEYS,
    _effective_diversity_bonus,
    _geometry_fields,
    _metric_beats,
    _metric_delta,
    _selector_claim_evidence,
    _single_cell_range_status,
    _worst_uniform_component_delta,
)
from benchmarking.reporting.training_target_row_fields import training_target_row_fields
from benchmarking.row_runtime import (
    collapse_warning_summary,
    dominant_runtime_phase_fields,
    last_history_value,
    mean_epoch_seconds,
    mean_history_value,
    phase_seconds,
    phase_seconds_with_prefix,
)
from learning.model_features import is_workload_blind_model_type, model_type_metadata


def _milliseconds_to_seconds(value: Any) -> float | None:
    milliseconds = as_float(value)
    return None if milliseconds is None else milliseconds / 1000.0


def _row_from_run(
    *,
    workload: str,
    run_label: str,
    command: list[str],
    returncode: int,
    elapsed_seconds: float,
    run_dir: Path,
    stdout_path: Path,
    run_json_path: Path,
    timings: dict[str, Any],
    run_json: dict[str, Any] | None,
    data_sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one compact comparison row."""
    mlqds = (run_json or {}).get("matched", {}).get("MLQDS", {})
    uniform = (run_json or {}).get("matched", {}).get("uniform", {})
    dp = (run_json or {}).get("matched", {}).get("DouglasPeucker", {})
    learned_fill = (run_json or {}).get("learned_fill_diagnostics", {})
    temporal_random_fill = learned_fill.get("TemporalRandomFill", {})
    temporal_oracle_fill = learned_fill.get("TemporalOracleFill", {})
    cuda_memory = (run_json or {}).get("cuda_memory", {}).get("training", {})
    child_torch_runtime = (run_json or {}).get("torch_runtime") or {}
    child_amp = child_torch_runtime.get("amp") or {}
    data_config = (run_json or {}).get("config", {}).get("data", {})
    model_config = (run_json or {}).get("config", {}).get("model", {})
    query_config = (run_json or {}).get("config", {}).get("query", {})
    baseline_config = (run_json or {}).get("config", {}).get("baselines", {})
    oracle_diagnostic = (run_json or {}).get("oracle_diagnostic") or {}
    workload_blind_protocol = (run_json or {}).get("workload_blind_protocol") or {}
    teacher_distillation = (run_json or {}).get("teacher_distillation") or {}
    collapse_summary = collapse_warning_summary(run_json)
    train_label_diagnostics = (
        (run_json or {})
        .get("workload_diagnostics", {})
        .get("train", {})
        .get("range_signal", {})
        .get("labels", {})
    )
    label_mass_fraction = train_label_diagnostics.get("component_positive_label_mass_fraction", {})
    target_diagnostics = (run_json or {}).get("training_target_diagnostics") or {}
    target_transform = (run_json or {}).get("range_training_target_transform") or {}
    fit_diagnostics = (run_json or {}).get("training_fit_diagnostics") or {}
    final_claim_summary = (run_json or {}).get("final_claim_summary") or {}
    legacy_range_useful_summary = (run_json or {}).get("legacy_range_useful_summary") or {}
    predictability_audit = (run_json or {}).get("predictability_audit") or {}
    predictability_metrics = predictability_audit.get("metrics") or {}
    prior_predictive_alignment_gate = (
        predictability_audit.get("prior_predictive_alignment_gate") or {}
    )
    per_head_predictability = predictability_audit.get("per_head_predictability") or {}
    query_hit_predictability = per_head_predictability.get("query_hit_probability") or {}
    behavior_predictability = per_head_predictability.get("conditional_behavior_utility") or {}
    replacement_predictability = (
        per_head_predictability.get("replacement_representative_value") or {}
    )
    segment_budget_predictability = per_head_predictability.get("segment_budget_target") or {}
    prior_channel_predictability = predictability_audit.get("prior_channel_predictability") or {}
    learning_causality = (run_json or {}).get("learning_causality_summary") or {}
    learning_delta_gate = learning_causality.get("learning_causality_delta_gate") or {}
    learned_segment_selector_config = (
        learning_causality.get("learned_segment_selector_config") or {}
    )
    causality_mask_diagnostics = learning_causality.get("causality_ablation_mask_diagnostics") or {}
    shuffled_prior_mask = causality_mask_diagnostics.get("MLQDS_shuffled_prior_fields") or {}
    no_query_prior_mask = causality_mask_diagnostics.get("MLQDS_without_query_prior_features") or {}
    no_behavior_mask = causality_mask_diagnostics.get("MLQDS_without_behavior_utility_head") or {}
    no_segment_budget_mask = (
        causality_mask_diagnostics.get("MLQDS_without_segment_budget_head") or {}
    )
    no_geometry_mask = causality_mask_diagnostics.get("MLQDS_without_geometry_tie_breaker") or {}
    no_length_support_allocation_mask = (
        causality_mask_diagnostics.get("MLQDS_without_segment_length_support_allocation") or {}
    )
    prior_sensitivity = learning_causality.get("prior_sensitivity_diagnostics") or {}
    shuffled_prior_sample = (prior_sensitivity.get("shuffled_prior_fields") or {}).get(
        "sampled_prior_features"
    ) or {}
    shuffled_prior_model = (prior_sensitivity.get("shuffled_prior_fields") or {}).get(
        "model_prior_features"
    ) or {}
    shuffled_prior_head_output = (prior_sensitivity.get("shuffled_prior_fields") or {}).get(
        "head_output"
    ) or {}
    shuffled_prior_score_output = (prior_sensitivity.get("shuffled_prior_fields") or {}).get(
        "score_output"
    ) or {}
    shuffled_prior_model_input = shuffled_prior_model.get("model_input_prior_features") or {}
    shuffled_prior_normalized = shuffled_prior_model.get("normalized_model_prior_features") or {}
    no_prior_sample = (prior_sensitivity.get("without_query_prior_features") or {}).get(
        "sampled_prior_features"
    ) or {}
    no_prior_model = (prior_sensitivity.get("without_query_prior_features") or {}).get(
        "model_prior_features"
    ) or {}
    no_prior_head_output = (prior_sensitivity.get("without_query_prior_features") or {}).get(
        "head_output"
    ) or {}
    no_prior_score_output = (prior_sensitivity.get("without_query_prior_features") or {}).get(
        "score_output"
    ) or {}
    no_prior_model_input = no_prior_model.get("model_input_prior_features") or {}
    no_prior_normalized = no_prior_model.get("normalized_model_prior_features") or {}
    workload_stability_gate = (run_json or {}).get("workload_stability_gate") or {}
    support_overlap_gate = (run_json or {}).get("support_overlap_gate") or {}
    global_sanity_gate = (run_json or {}).get("global_sanity_gate") or {}
    target_diffusion_gate = (run_json or {}).get("target_diffusion_gate") or {}
    workload_signature_gate = ((run_json or {}).get("workload_distribution_comparison") or {}).get(
        "workload_signature_gate"
    ) or {}
    signature_pairs = workload_signature_gate.get("pairs") or {}
    signature_train_pair = (workload_signature_gate.get("pairs") or {}).get("train") or {}
    signature_train_metrics = signature_train_pair.get("metrics") or {}
    point_hit_signature_distance = signature_train_metrics.get(
        "point_hit_distribution_ks",
        signature_train_metrics.get("point_hit_distribution_ks_proxy"),
    )
    ship_hit_signature_distance = signature_train_metrics.get(
        "ship_hit_distribution_ks",
        signature_train_metrics.get("ship_hit_distribution_ks_proxy"),
    )
    point_hit_fraction_signature_distance = signature_train_metrics.get(
        "point_hit_fraction_distribution_ks"
    )
    ship_hit_fraction_signature_distance = signature_train_metrics.get(
        "ship_hit_fraction_distribution_ks"
    )
    eval_selector_diagnostics = ((run_json or {}).get("selector_budget_diagnostics") or {}).get(
        "eval"
    ) or {}
    target_budget_row = _target_budget_row(
        target_diagnostics, model_config.get("compression_ratio")
    )
    selector_budget_row = _selector_budget_row(
        eval_selector_diagnostics, model_config.get("compression_ratio")
    )
    selector_low_budget_summary = _selector_low_budget_summary(eval_selector_diagnostics)
    selector_claim_evidence = _selector_claim_evidence(
        selector_budget_row,
        model_config.get("model_type"),
    )
    mlqds_aggregate_f1 = mlqds.get("aggregate_f1")
    mlqds_query_point_recall = mlqds.get("query_point_recall")
    mlqds_range_point_f1 = mlqds.get("range_point_f1", mlqds_aggregate_f1)
    mlqds_range_usefulness = mlqds.get("range_usefulness_score")
    mlqds_inference_only_latency_ms = mlqds.get("latency_ms")
    mlqds_query_local_utility = mlqds.get("query_local_utility_score")
    mlqds_gap_time_usefulness = mlqds.get("range_usefulness_gap_time_score")
    mlqds_gap_distance_usefulness = mlqds.get("range_usefulness_gap_distance_score")
    mlqds_gap_min_usefulness = mlqds.get("range_usefulness_gap_min_score")
    if (
        final_claim_summary.get("primary_metric") == "QueryLocalUtility"
        and mlqds_query_local_utility is not None
    ):
        mlqds_primary_score = mlqds_query_local_utility
        mlqds_primary_metric = "query_local_utility"
    else:
        mlqds_primary_score = (
            mlqds_range_usefulness if mlqds_range_usefulness is not None else mlqds_range_point_f1
        )
        mlqds_primary_metric = (
            "range_usefulness" if mlqds_range_usefulness is not None else "range_point_f1"
        )
    random_fill_range_usefulness = temporal_random_fill.get("range_usefulness_score")
    oracle_fill_range_usefulness = temporal_oracle_fill.get("range_usefulness_score")
    uniform_aggregate_f1 = uniform.get("aggregate_f1")
    uniform_query_point_recall = uniform.get("query_point_recall")
    uniform_range_point_f1 = uniform.get("range_point_f1", uniform_aggregate_f1)
    uniform_range_usefulness = uniform.get("range_usefulness_score")
    uniform_query_local_utility = uniform.get("query_local_utility_score")
    uniform_gap_time_usefulness = uniform.get("range_usefulness_gap_time_score")
    uniform_gap_distance_usefulness = uniform.get("range_usefulness_gap_distance_score")
    uniform_gap_min_usefulness = uniform.get("range_usefulness_gap_min_score")
    dp_aggregate_f1 = dp.get("aggregate_f1")
    dp_query_point_recall = dp.get("query_point_recall")
    dp_range_point_f1 = dp.get("range_point_f1", dp_aggregate_f1)
    dp_range_usefulness = dp.get("range_usefulness_score")
    dp_query_local_utility = dp.get("query_local_utility_score")
    dp_gap_time_usefulness = dp.get("range_usefulness_gap_time_score")
    dp_gap_distance_usefulness = dp.get("range_usefulness_gap_distance_score")
    dp_gap_min_usefulness = dp.get("range_usefulness_gap_min_score")
    component_deltas = {
        f"mlqds_vs_uniform_{key}": _metric_delta(mlqds, uniform, key)
        for key in RANGE_COMPONENT_KEYS
    }
    worst_component_delta = _worst_uniform_component_delta(component_deltas)
    audit = _audit_summary(run_json)
    runtime_bottleneck = dominant_runtime_phase_fields(timings, elapsed_seconds)
    beats_uniform_range_usefulness = _metric_beats(mlqds, uniform, "range_usefulness_score")
    beats_dp_range_usefulness = _metric_beats(mlqds, dp, "range_usefulness_score")
    beats_temporal_random_fill_range_usefulness = _metric_beats(
        mlqds,
        temporal_random_fill,
        "range_usefulness_score",
    )
    single_cell_range_status = _single_cell_range_status(
        returncode=returncode,
        model_type=model_config.get("model_type"),
        protocol_enabled=workload_blind_protocol.get("enabled"),
        primary_frozen=workload_blind_protocol.get(
            "primary_masks_frozen_before_eval_query_scoring"
        ),
        audit_frozen=workload_blind_protocol.get("audit_masks_frozen_before_eval_query_scoring"),
        audit_ratio_count=int(audit["audit_compression_ratio_count"]),
        beats_uniform=beats_uniform_range_usefulness,
        beats_dp=beats_dp_range_usefulness,
        selector_claim_status=str(selector_claim_evidence["selector_claim_status"]),
    )
    return {
        "workload": workload,
        "run_label": run_label,
        **_data_source_row_fields(data_sources),
        "returncode": int(returncode),
        "elapsed_seconds": float(elapsed_seconds),
        "train_seconds": phase_seconds_with_prefix(timings, "train-model"),
        "evaluate_matched_seconds": phase_seconds(timings, "evaluate-matched"),
        "epoch_mean_seconds": mean_epoch_seconds(timings),
        "peak_allocated_mb": cuda_memory.get("max_allocated_mb"),
        "best_epoch": (run_json or {}).get("best_epoch"),
        "best_loss": (run_json or {}).get("best_loss"),
        "best_selection_score": (run_json or {}).get("best_selection_score"),
        "final_loss": last_history_value(run_json, "loss"),
        "final_kendall_tau_t0": last_history_value(run_json, "kendall_tau_t0"),
        "final_pred_std": last_history_value(run_json, "pred_std"),
        "epoch_forward_mean_seconds": mean_history_value(run_json, "epoch_forward_seconds"),
        "epoch_loss_mean_seconds": mean_history_value(run_json, "epoch_loss_seconds"),
        "epoch_backward_mean_seconds": mean_history_value(run_json, "epoch_backward_seconds"),
        "epoch_diagnostic_mean_seconds": mean_history_value(run_json, "epoch_diagnostic_seconds"),
        "epoch_validation_score_mean_seconds": mean_history_value(
            run_json, "epoch_validation_score_seconds"
        ),
        "single_cell_range_status": single_cell_range_status,
        "final_claim_status": final_claim_summary.get(
            "status", "not_available_until_query_local_utility"
        ),
        "final_success_allowed": bool(final_claim_summary.get("final_success_allowed", False)),
        "final_claim_blocking_gates": final_claim_summary.get("blocking_gates"),
        "workload_stability_gate_pass": workload_stability_gate.get("gate_pass"),
        "workload_stability_failed_checks": workload_stability_gate.get("failed_checks"),
        "workload_stability_train_replicates": workload_stability_gate.get(
            "train_workload_replicate_count"
        ),
        "workload_stability_configured_target_coverage": workload_stability_gate.get(
            "configured_target_coverage"
        ),
        "workload_stability_configured_workload_profile_id": workload_stability_gate.get(
            "configured_workload_profile_id"
        ),
        "workload_stability_configured_workload_profile_in_grid": workload_stability_gate.get(
            "configured_workload_profile_in_grid"
        ),
        "workload_stability_gate_mode": workload_stability_gate.get(
            "gate_mode", query_config.get("workload_stability_gate_mode")
        ),
        "support_overlap_gate_pass": support_overlap_gate.get("gate_pass"),
        "support_overlap_failed_checks": support_overlap_gate.get("failed_checks"),
        "support_eval_points_outside_train_prior_extent_fraction": support_overlap_gate.get(
            "eval_points_outside_train_prior_extent_fraction"
        ),
        "support_sampled_prior_nonzero_fraction": support_overlap_gate.get(
            "sampled_prior_nonzero_fraction"
        ),
        "support_primary_sampled_prior_nonzero_fraction": support_overlap_gate.get(
            "primary_sampled_prior_nonzero_fraction"
        ),
        "support_route_density_overlap": support_overlap_gate.get("route_density_overlap"),
        "support_query_prior_support_overlap": support_overlap_gate.get(
            "query_prior_support_overlap"
        ),
        "support_train_eval_spatial_extent_intersection_fraction": support_overlap_gate.get(
            "train_eval_spatial_extent_intersection_fraction"
        ),
        "global_sanity_gate_pass": global_sanity_gate.get("gate_pass"),
        "global_sanity_failed_checks": global_sanity_gate.get("failed_checks"),
        "global_sanity_endpoint_sanity": global_sanity_gate.get("endpoint_sanity"),
        "global_sanity_avg_sed_ratio_vs_uniform": global_sanity_gate.get(
            "avg_sed_ratio_vs_uniform"
        ),
        "global_sanity_avg_sed_ratio_vs_uniform_max": global_sanity_gate.get(
            "avg_sed_ratio_vs_uniform_max"
        ),
        "global_sanity_avg_length_preserved": global_sanity_gate.get("avg_length_preserved"),
        "target_diffusion_gate_pass": target_diffusion_gate.get("gate_pass"),
        "target_diffusion_failed_checks": target_diffusion_gate.get("failed_checks"),
        "target_diffusion_final_label_support_fraction": target_diffusion_gate.get(
            "final_label_support_fraction"
        ),
        "predictability_gate_pass": predictability_audit.get("gate_pass"),
        "predictability_spearman": predictability_metrics.get("spearman"),
        "predictability_kendall_tau": predictability_metrics.get("kendall_tau"),
        "predictability_lift_at_1_percent": predictability_metrics.get("lift_at_1_percent"),
        "predictability_lift_at_2_percent": predictability_metrics.get("lift_at_2_percent"),
        "predictability_lift_at_5_percent": predictability_metrics.get("lift_at_5_percent"),
        "predictability_pr_auc_lift_over_base_rate": predictability_metrics.get(
            "pr_auc_lift_over_base_rate"
        ),
        "prior_predictive_alignment_gate_pass": prior_predictive_alignment_gate.get("gate_pass"),
        "prior_predictive_alignment_failed_checks": prior_predictive_alignment_gate.get(
            "failed_checks"
        ),
        "prior_predictive_alignment_thresholds": prior_predictive_alignment_gate.get("thresholds"),
        "prior_positive_spearman_head_count": prior_predictive_alignment_gate.get(
            "positive_spearman_head_count"
        ),
        "predictability_query_hit_spearman": query_hit_predictability.get("spearman"),
        "predictability_query_hit_lift_at_5_percent": query_hit_predictability.get(
            "lift_at_5_percent"
        ),
        "predictability_query_hit_pr_auc_lift_over_base_rate": query_hit_predictability.get(
            "pr_auc_lift_over_base_rate"
        ),
        "predictability_behavior_spearman": behavior_predictability.get("spearman"),
        "predictability_behavior_lift_at_5_percent": behavior_predictability.get(
            "lift_at_5_percent"
        ),
        "predictability_replacement_spearman": replacement_predictability.get("spearman"),
        "predictability_replacement_lift_at_5_percent": replacement_predictability.get(
            "lift_at_5_percent"
        ),
        "predictability_segment_budget_spearman": segment_budget_predictability.get("spearman"),
        "predictability_segment_budget_lift_at_5_percent": segment_budget_predictability.get(
            "lift_at_5_percent"
        ),
        "prior_channel_query_mass_spearman": (
            (prior_channel_predictability.get("query_mass_prior") or {}).get("spearman")
            if isinstance(prior_channel_predictability, dict)
            else None
        ),
        "prior_channel_combined_score_lift_at_5_percent": (
            (prior_channel_predictability.get("combined_prior_score") or {}).get(
                "lift_at_5_percent"
            )
            if isinstance(prior_channel_predictability, dict)
            else None
        ),
        "workload_signature_gate_pass": workload_signature_gate.get("all_pass"),
        "workload_signature_gate_available": workload_signature_gate.get("all_available"),
        "workload_signature_pair_count": len(signature_pairs)
        if isinstance(signature_pairs, dict)
        else None,
        "workload_signature_failed_pairs": (
            [
                label
                for label, pair in signature_pairs.items()
                if isinstance(pair, dict) and not bool(pair.get("gate_pass", False))
            ]
            if isinstance(signature_pairs, dict)
            else None
        ),
        "train_eval_anchor_family_l1_distance": signature_train_metrics.get(
            "anchor_family_l1_distance"
        ),
        "train_eval_footprint_family_l1_distance": signature_train_metrics.get(
            "footprint_family_l1_distance"
        ),
        "train_eval_point_hit_distribution_ks": point_hit_signature_distance,
        "train_eval_ship_hit_distribution_ks": ship_hit_signature_distance,
        "train_eval_point_hit_fraction_distribution_ks": point_hit_fraction_signature_distance,
        "train_eval_ship_hit_fraction_distribution_ks": ship_hit_fraction_signature_distance,
        "train_eval_query_count_delta": signature_train_metrics.get("query_count_delta"),
        "train_eval_query_count_relative_delta": signature_train_metrics.get(
            "query_count_relative_delta"
        ),
        "train_eval_point_hit_distribution_used_quantile_proxy": signature_train_metrics.get(
            "point_hit_distribution_used_quantile_proxy"
        ),
        "train_eval_ship_hit_distribution_used_quantile_proxy": signature_train_metrics.get(
            "ship_hit_distribution_used_quantile_proxy"
        ),
        "train_signature_total_points": signature_train_metrics.get("train_total_points"),
        "eval_signature_total_points": signature_train_metrics.get("eval_total_points"),
        "train_signature_total_trajectories": signature_train_metrics.get(
            "train_total_trajectories"
        ),
        "eval_signature_total_trajectories": signature_train_metrics.get("eval_total_trajectories"),
        "train_eval_point_hit_distribution_ks_proxy": point_hit_signature_distance,
        "train_eval_ship_hit_distribution_ks_proxy": ship_hit_signature_distance,
        "learning_causality_ablation_status": learning_causality.get(
            "learning_causality_ablation_status"
        ),
        "learning_causality_gate_pass": learning_causality.get("learning_causality_gate_pass"),
        "learning_causality_failed_checks": learning_causality.get(
            "learning_causality_failed_checks"
        ),
        "causality_ablation_missing": learning_causality.get("causality_ablation_missing"),
        "learned_controlled_retained_slot_fraction": learning_causality.get(
            "learned_controlled_retained_slot_fraction"
        ),
        "planned_learned_controlled_retained_slot_fraction": learning_causality.get(
            "planned_learned_controlled_retained_slot_fraction"
        ),
        "actual_learned_controlled_retained_slot_fraction": learning_causality.get(
            "actual_learned_controlled_retained_slot_fraction"
        ),
        "trajectories_with_at_least_one_learned_decision": learning_causality.get(
            "trajectories_with_at_least_one_learned_decision"
        ),
        "trajectories_with_zero_learned_decisions": learning_causality.get(
            "trajectories_with_zero_learned_decisions"
        ),
        "segment_budget_entropy": learning_causality.get("segment_budget_entropy"),
        "segment_budget_entropy_normalized": learning_causality.get(
            "segment_budget_entropy_normalized"
        ),
        "selector_trace_retained_mask_matches_primary": learning_causality.get(
            "selector_trace_retained_mask_matches_primary"
        ),
        "shuffled_score_ablation_delta": learning_causality.get("shuffled_score_ablation_delta"),
        "untrained_score_ablation_delta": learning_causality.get("untrained_score_ablation_delta"),
        "shuffled_prior_field_ablation_delta": learning_causality.get(
            "shuffled_prior_field_ablation_delta"
        ),
        "prior_field_only_score_ablation_delta": learning_causality.get(
            "prior_field_only_score_ablation_delta"
        ),
        "no_query_prior_field_ablation_delta": learning_causality.get(
            "no_query_prior_field_ablation_delta"
        ),
        "no_behavior_head_ablation_delta": learning_causality.get(
            "no_behavior_head_ablation_delta"
        ),
        "no_segment_budget_head_ablation_delta": learning_causality.get(
            "no_segment_budget_head_ablation_delta"
        ),
        "no_trajectory_fairness_preallocation_ablation_delta": learning_causality.get(
            "no_trajectory_fairness_preallocation_ablation_delta"
        ),
        "shuffled_prior_retained_mask_jaccard": shuffled_prior_mask.get("retained_mask_jaccard"),
        "shuffled_prior_retained_symmetric_difference_count": shuffled_prior_mask.get(
            "retained_symmetric_difference_count"
        ),
        "no_query_prior_retained_mask_jaccard": no_query_prior_mask.get("retained_mask_jaccard"),
        "no_query_prior_retained_symmetric_difference_count": no_query_prior_mask.get(
            "retained_symmetric_difference_count"
        ),
        "no_behavior_retained_mask_jaccard": no_behavior_mask.get("retained_mask_jaccard"),
        "no_behavior_retained_symmetric_difference_count": no_behavior_mask.get(
            "retained_symmetric_difference_count"
        ),
        "no_segment_budget_retained_mask_jaccard": no_segment_budget_mask.get(
            "retained_mask_jaccard"
        ),
        "no_segment_budget_retained_symmetric_difference_count": no_segment_budget_mask.get(
            "retained_symmetric_difference_count"
        ),
        "no_geometry_tie_breaker_ablation_delta": learning_causality.get(
            "no_geometry_tie_breaker_ablation_delta"
        ),
        "no_segment_length_support_allocation_ablation_delta": learning_causality.get(
            "no_segment_length_support_allocation_ablation_delta"
        ),
        "no_geometry_retained_mask_jaccard": no_geometry_mask.get("retained_mask_jaccard"),
        "no_geometry_retained_symmetric_difference_count": no_geometry_mask.get(
            "retained_symmetric_difference_count"
        ),
        "no_segment_length_support_allocation_retained_mask_jaccard": (
            no_length_support_allocation_mask.get("retained_mask_jaccard")
        ),
        "no_segment_length_support_allocation_retained_symmetric_difference_count": (
            no_length_support_allocation_mask.get("retained_symmetric_difference_count")
        ),
        "learning_causality_min_material_delta": learning_delta_gate.get(
            "min_material_query_local_utility_delta"
        ),
        "learning_causality_shuffled_fraction_of_uniform_gap_min": learning_delta_gate.get(
            "shuffled_score_delta_fraction_of_uniform_gap_min"
        ),
        "learning_causality_mlqds_uniform_gap": learning_delta_gate.get(
            "mlqds_uniform_query_local_utility_gap"
        ),
        "learning_causality_delta_thresholds": learning_delta_gate.get("thresholds"),
        "segment_budget_head_ablation_mode": learning_causality.get(
            "segment_budget_head_ablation_mode"
        ),
        "learned_segment_geometry_gain_weight": learned_segment_selector_config.get(
            "geometry_gain_weight", model_config.get("learned_segment_geometry_gain_weight")
        ),
        "learned_segment_allocation_length_support_weight": learned_segment_selector_config.get(
            "allocation_length_support_weight",
            model_config.get("learned_segment_allocation_length_support_weight"),
        ),
        "learned_segment_allocation_weight_floor": learned_segment_selector_config.get(
            "allocation_weight_floor",
            model_config.get("learned_segment_allocation_weight_floor"),
        ),
        "learned_segment_score_blend_weight": learned_segment_selector_config.get(
            "segment_score_blend_weight", model_config.get("learned_segment_score_blend_weight")
        ),
        "learned_segment_transfer_calibration_mode": learned_segment_selector_config.get(
            "segment_transfer_calibration_mode",
            model_config.get("learned_segment_transfer_calibration_mode"),
        ),
        "learned_segment_fairness_preallocation_enabled": learned_segment_selector_config.get(
            "fairness_preallocation_enabled",
            model_config.get("learned_segment_fairness_preallocation"),
        ),
        "learned_segment_length_repair_fraction": learned_segment_selector_config.get(
            "length_repair_fraction", model_config.get("learned_segment_length_repair_fraction")
        ),
        "learned_segment_length_repair_score_protection_fraction": (
            learned_segment_selector_config.get(
                "length_repair_score_protection_fraction",
                model_config.get("learned_segment_length_repair_score_protection_fraction"),
            )
        ),
        "learned_segment_length_support_blend_weight": learned_segment_selector_config.get(
            "length_support_blend_weight",
            model_config.get("learned_segment_length_support_blend_weight"),
        ),
        "prior_sample_gate_pass": learning_causality.get("prior_sample_gate_pass"),
        "prior_sample_gate_failures": learning_causality.get("prior_sample_gate_failures"),
        "shuffled_prior_sampled_inputs_changed": shuffled_prior_sample.get(
            "sampled_inputs_changed"
        ),
        "shuffled_prior_sampled_primary_nonzero_fraction": shuffled_prior_sample.get(
            "primary_nonzero_fraction"
        ),
        "shuffled_prior_sampled_ablation_nonzero_fraction": shuffled_prior_sample.get(
            "ablation_nonzero_fraction"
        ),
        "shuffled_prior_sampled_mean_abs_feature_delta": shuffled_prior_sample.get(
            "mean_abs_feature_delta"
        ),
        "shuffled_prior_sampled_max_abs_feature_delta": shuffled_prior_sample.get(
            "max_abs_feature_delta"
        ),
        "shuffled_prior_sampled_outside_extent_fraction": shuffled_prior_sample.get(
            "points_outside_prior_extent_fraction"
        ),
        "shuffled_prior_model_inputs_changed": shuffled_prior_model_input.get(
            "sampled_inputs_changed"
        ),
        "shuffled_prior_model_input_mean_abs_feature_delta": shuffled_prior_model_input.get(
            "mean_abs_feature_delta"
        ),
        "shuffled_prior_normalized_model_inputs_changed": shuffled_prior_normalized.get(
            "sampled_inputs_changed"
        ),
        "shuffled_prior_normalized_model_mean_abs_feature_delta": (
            shuffled_prior_normalized.get("mean_abs_feature_delta")
        ),
        "shuffled_prior_head_logits_changed": shuffled_prior_head_output.get("head_logits_changed"),
        "shuffled_prior_head_logit_mean_abs_delta": shuffled_prior_head_output.get(
            "mean_abs_head_logit_delta"
        ),
        "shuffled_prior_head_probability_mean_abs_delta": shuffled_prior_head_output.get(
            "mean_abs_head_probability_delta"
        ),
        "shuffled_prior_score_output_mean_abs_delta": shuffled_prior_score_output.get(
            "mean_abs_score_delta"
        ),
        "shuffled_prior_score_output_max_abs_delta": shuffled_prior_score_output.get(
            "max_abs_score_delta"
        ),
        "shuffled_prior_score_output_topk_jaccard_at_retained_count": (
            shuffled_prior_score_output.get("score_topk_jaccard_at_retained_count")
        ),
        "no_prior_sampled_primary_nonzero_fraction": no_prior_sample.get(
            "primary_nonzero_fraction"
        ),
        "no_prior_sampled_mean_abs_feature_delta": no_prior_sample.get("mean_abs_feature_delta"),
        "no_prior_sampled_outside_extent_fraction": no_prior_sample.get(
            "points_outside_prior_extent_fraction"
        ),
        "no_prior_model_inputs_changed": no_prior_model_input.get("sampled_inputs_changed"),
        "no_prior_model_input_mean_abs_feature_delta": no_prior_model_input.get(
            "mean_abs_feature_delta"
        ),
        "no_prior_normalized_model_inputs_changed": no_prior_normalized.get(
            "sampled_inputs_changed"
        ),
        "no_prior_normalized_model_mean_abs_feature_delta": no_prior_normalized.get(
            "mean_abs_feature_delta"
        ),
        "no_prior_head_logits_changed": no_prior_head_output.get("head_logits_changed"),
        "no_prior_head_logit_mean_abs_delta": no_prior_head_output.get("mean_abs_head_logit_delta"),
        "no_prior_head_probability_mean_abs_delta": no_prior_head_output.get(
            "mean_abs_head_probability_delta"
        ),
        "no_prior_score_output_mean_abs_delta": no_prior_score_output.get(
            "mean_abs_score_delta"
        ),
        "no_prior_score_output_max_abs_delta": no_prior_score_output.get(
            "max_abs_score_delta"
        ),
        "no_prior_score_output_topk_jaccard_at_retained_count": (
            no_prior_score_output.get("score_topk_jaccard_at_retained_count")
        ),
        "legacy_range_useful_diagnostic_only": bool(
            legacy_range_useful_summary.get("diagnostic_only", True)
        ),
        **selector_claim_evidence,
        "workload_blind_candidate": is_workload_blind_model_type(model_config.get("model_type")),
        "workload_blind_protocol_enabled": workload_blind_protocol.get("enabled"),
        "primary_masks_frozen_before_eval_query_scoring": workload_blind_protocol.get(
            "primary_masks_frozen_before_eval_query_scoring"
        ),
        "audit_masks_frozen_before_eval_query_scoring": workload_blind_protocol.get(
            "audit_masks_frozen_before_eval_query_scoring"
        ),
        "eval_geometry_blend_allowed": workload_blind_protocol.get("eval_geometry_blend_allowed"),
        "beats_uniform_range_usefulness": beats_uniform_range_usefulness,
        "beats_douglas_peucker_range_usefulness": beats_dp_range_usefulness,
        "beats_temporal_random_fill_range_usefulness": beats_temporal_random_fill_range_usefulness,
        **audit,
        **runtime_bottleneck,
        "mlqds_primary_metric": mlqds_primary_metric,
        "mlqds_primary_score": mlqds_primary_score,
        "mlqds_aggregate_f1": mlqds_aggregate_f1,
        "mlqds_query_point_recall": mlqds_query_point_recall,
        "mlqds_range_point_f1": mlqds_range_point_f1,
        "mlqds_range_usefulness": mlqds_range_usefulness,
        "mlqds_range_usefulness_score": mlqds_range_usefulness,
        "mlqds_query_local_utility_score": mlqds_query_local_utility,
        "mlqds_range_usefulness_gap_time_score": mlqds_gap_time_usefulness,
        "mlqds_range_usefulness_gap_distance_score": mlqds_gap_distance_usefulness,
        "mlqds_range_usefulness_gap_min_score": mlqds_gap_min_usefulness,
        "mlqds_type_f1": (mlqds.get("per_type_f1") or {}).get(workload),
        "mlqds_range_ship_f1": mlqds.get("range_ship_f1"),
        "mlqds_range_ship_coverage": mlqds.get("range_ship_coverage"),
        "mlqds_range_entry_exit_f1": mlqds.get("range_entry_exit_f1"),
        "mlqds_range_crossing_f1": mlqds.get("range_crossing_f1"),
        "mlqds_range_temporal_coverage": mlqds.get("range_temporal_coverage"),
        "mlqds_range_gap_coverage": mlqds.get("range_gap_coverage"),
        "mlqds_range_gap_time_coverage": mlqds.get("range_gap_time_coverage"),
        "mlqds_range_gap_distance_coverage": mlqds.get("range_gap_distance_coverage"),
        "mlqds_range_gap_min_coverage": mlqds.get("range_gap_min_coverage"),
        "mlqds_range_turn_coverage": mlqds.get("range_turn_coverage"),
        "mlqds_range_shape_score": mlqds.get("range_shape_score"),
        **_geometry_fields("mlqds", mlqds),
        "range_usefulness_schema_version": mlqds.get("range_usefulness_schema_version"),
        "range_usefulness_gap_ablation_version": mlqds.get("range_usefulness_gap_ablation_version"),
        "final_metrics_mode": (run_json or {}).get(
            "final_metrics_mode", baseline_config.get("final_metrics_mode")
        ),
        "uniform_aggregate_f1": uniform_aggregate_f1,
        "uniform_query_point_recall": uniform_query_point_recall,
        "uniform_range_point_f1": uniform_range_point_f1,
        "uniform_range_usefulness": uniform_range_usefulness,
        "uniform_range_usefulness_score": uniform_range_usefulness,
        "uniform_query_local_utility_score": uniform_query_local_utility,
        "uniform_range_usefulness_gap_time_score": uniform_gap_time_usefulness,
        "uniform_range_usefulness_gap_distance_score": uniform_gap_distance_usefulness,
        "uniform_range_usefulness_gap_min_score": uniform_gap_min_usefulness,
        "uniform_range_ship_f1": uniform.get("range_ship_f1"),
        "uniform_range_ship_coverage": uniform.get("range_ship_coverage"),
        "uniform_range_entry_exit_f1": uniform.get("range_entry_exit_f1"),
        "uniform_range_crossing_f1": uniform.get("range_crossing_f1"),
        "uniform_range_temporal_coverage": uniform.get("range_temporal_coverage"),
        "uniform_range_gap_coverage": uniform.get("range_gap_coverage"),
        "uniform_range_turn_coverage": uniform.get("range_turn_coverage"),
        "uniform_range_shape_score": uniform.get("range_shape_score"),
        **_geometry_fields("uniform", uniform),
        "douglas_peucker_aggregate_f1": dp_aggregate_f1,
        "douglas_peucker_query_point_recall": dp_query_point_recall,
        "douglas_peucker_range_point_f1": dp_range_point_f1,
        "douglas_peucker_range_usefulness": dp_range_usefulness,
        "douglas_peucker_range_usefulness_score": dp_range_usefulness,
        "douglas_peucker_query_local_utility_score": dp_query_local_utility,
        "douglas_peucker_range_usefulness_gap_time_score": dp_gap_time_usefulness,
        "douglas_peucker_range_usefulness_gap_distance_score": dp_gap_distance_usefulness,
        "douglas_peucker_range_usefulness_gap_min_score": dp_gap_min_usefulness,
        "douglas_peucker_range_ship_f1": dp.get("range_ship_f1"),
        "douglas_peucker_range_ship_coverage": dp.get("range_ship_coverage"),
        "douglas_peucker_range_entry_exit_f1": dp.get("range_entry_exit_f1"),
        "douglas_peucker_range_crossing_f1": dp.get("range_crossing_f1"),
        "douglas_peucker_range_temporal_coverage": dp.get("range_temporal_coverage"),
        "douglas_peucker_range_gap_coverage": dp.get("range_gap_coverage"),
        "douglas_peucker_range_turn_coverage": dp.get("range_turn_coverage"),
        "douglas_peucker_range_shape_score": dp.get("range_shape_score"),
        **_geometry_fields("douglas_peucker", dp),
        "mlqds_vs_uniform_range_point_f1": (
            float(mlqds_range_point_f1) - float(uniform_range_point_f1)
            if mlqds_range_point_f1 is not None and uniform_range_point_f1 is not None
            else None
        ),
        "mlqds_vs_douglas_peucker_query_point_recall": (
            float(mlqds_query_point_recall) - float(dp_query_point_recall)
            if mlqds_query_point_recall is not None and dp_query_point_recall is not None
            else None
        ),
        "mlqds_vs_douglas_peucker_range_point_f1": (
            float(mlqds_range_point_f1) - float(dp_range_point_f1)
            if mlqds_range_point_f1 is not None and dp_range_point_f1 is not None
            else None
        ),
        "mlqds_vs_uniform_range_usefulness": (
            float(mlqds_range_usefulness) - float(uniform_range_usefulness)
            if mlqds_range_usefulness is not None and uniform_range_usefulness is not None
            else None
        ),
        "mlqds_vs_uniform_query_local_utility": (
            float(mlqds_query_local_utility) - float(uniform_query_local_utility)
            if mlqds_query_local_utility is not None and uniform_query_local_utility is not None
            else None
        ),
        "mlqds_vs_douglas_peucker_range_usefulness": (
            float(mlqds_range_usefulness) - float(dp_range_usefulness)
            if mlqds_range_usefulness is not None and dp_range_usefulness is not None
            else None
        ),
        "mlqds_vs_douglas_peucker_query_local_utility": (
            float(mlqds_query_local_utility) - float(dp_query_local_utility)
            if mlqds_query_local_utility is not None and dp_query_local_utility is not None
            else None
        ),
        "mlqds_vs_uniform_range_usefulness_gap_time": (
            float(mlqds_gap_time_usefulness) - float(uniform_gap_time_usefulness)
            if mlqds_gap_time_usefulness is not None and uniform_gap_time_usefulness is not None
            else None
        ),
        "mlqds_vs_uniform_range_usefulness_gap_distance": (
            float(mlqds_gap_distance_usefulness) - float(uniform_gap_distance_usefulness)
            if mlqds_gap_distance_usefulness is not None
            and uniform_gap_distance_usefulness is not None
            else None
        ),
        "mlqds_vs_uniform_range_usefulness_gap_min": (
            float(mlqds_gap_min_usefulness) - float(uniform_gap_min_usefulness)
            if mlqds_gap_min_usefulness is not None and uniform_gap_min_usefulness is not None
            else None
        ),
        "mlqds_vs_douglas_peucker_range_usefulness_gap_time": (
            float(mlqds_gap_time_usefulness) - float(dp_gap_time_usefulness)
            if mlqds_gap_time_usefulness is not None and dp_gap_time_usefulness is not None
            else None
        ),
        "mlqds_vs_douglas_peucker_range_usefulness_gap_distance": (
            float(mlqds_gap_distance_usefulness) - float(dp_gap_distance_usefulness)
            if mlqds_gap_distance_usefulness is not None and dp_gap_distance_usefulness is not None
            else None
        ),
        "mlqds_vs_douglas_peucker_range_usefulness_gap_min": (
            float(mlqds_gap_min_usefulness) - float(dp_gap_min_usefulness)
            if mlqds_gap_min_usefulness is not None and dp_gap_min_usefulness is not None
            else None
        ),
        **component_deltas,
        **worst_component_delta,
        "mlqds_vs_uniform_avg_sed_km": _metric_delta(
            {"value": (mlqds.get("geometric_distortion") or {}).get("avg_sed_km")},
            {"value": (uniform.get("geometric_distortion") or {}).get("avg_sed_km")},
            "value",
        ),
        "mlqds_vs_uniform_avg_ped_km": _metric_delta(
            {"value": (mlqds.get("geometric_distortion") or {}).get("avg_ped_km")},
            {"value": (uniform.get("geometric_distortion") or {}).get("avg_ped_km")},
            "value",
        ),
        "mlqds_vs_uniform_avg_length_preserved": _metric_delta(
            mlqds,
            uniform,
            "avg_length_preserved",
        ),
        "mlqds_latency_ms": mlqds_inference_only_latency_ms,
        "mlqds_inference_only_latency_ms": mlqds_inference_only_latency_ms,
        "mlqds_inference_only_latency_seconds": _milliseconds_to_seconds(
            mlqds_inference_only_latency_ms
        ),
        "avg_length_preserved": mlqds.get("avg_length_preserved"),
        "combined_query_shape_score": mlqds.get("combined_query_shape_score"),
        "temporal_random_fill_range_point_f1": temporal_random_fill.get("range_point_f1"),
        "temporal_random_fill_range_usefulness_score": random_fill_range_usefulness,
        "temporal_oracle_fill_range_point_f1": temporal_oracle_fill.get("range_point_f1"),
        "temporal_oracle_fill_range_usefulness_score": oracle_fill_range_usefulness,
        "mlqds_vs_temporal_random_fill_range_usefulness": (
            float(mlqds_range_usefulness) - float(random_fill_range_usefulness)
            if mlqds_range_usefulness is not None and random_fill_range_usefulness is not None
            else None
        ),
        "temporal_oracle_fill_gap_range_usefulness": (
            float(oracle_fill_range_usefulness) - float(mlqds_range_usefulness)
            if mlqds_range_usefulness is not None and oracle_fill_range_usefulness is not None
            else None
        ),
        "collapse_warning": collapse_summary["collapse_warning_any"],
        "collapse_warning_any": collapse_summary["collapse_warning_any"],
        "collapse_warning_count": collapse_summary["collapse_warning_count"],
        "best_epoch_collapse_warning": collapse_summary["best_epoch_collapse_warning"],
        "min_pred_std": collapse_summary["min_pred_std"],
        "best_epoch_pred_std": collapse_summary["best_epoch_pred_std"],
        "model_type": model_config.get("model_type"),
        **{
            f"model_metadata_{key}": value
            for key, value in model_type_metadata(str(model_config.get("model_type", ""))).items()
        },
        "historical_prior_k": model_config.get("historical_prior_k"),
        "historical_prior_clock_weight": model_config.get("historical_prior_clock_weight"),
        "historical_prior_mmsi_weight": model_config.get("historical_prior_mmsi_weight"),
        "historical_prior_density_weight": model_config.get("historical_prior_density_weight"),
        "historical_prior_min_target": model_config.get("historical_prior_min_target"),
        "historical_prior_support_ratio": model_config.get("historical_prior_support_ratio"),
        "historical_prior_source_aggregation": model_config.get(
            "historical_prior_source_aggregation"
        ),
        "historical_prior_source_count": target_diagnostics.get("historical_prior_source_count"),
        "historical_prior_stored_support_count": target_diagnostics.get(
            "historical_prior_stored_support_count"
        ),
        "checkpoint_score_variant": model_config.get("checkpoint_score_variant"),
        "compression_ratio": model_config.get("compression_ratio"),
        "n_queries": query_config.get("n_queries"),
        "max_queries": query_config.get("max_queries"),
        "query_target_coverage": query_config.get("target_coverage"),
        "range_spatial_km": query_config.get("range_spatial_km"),
        "range_time_hours": query_config.get("range_time_hours"),
        "loss_objective": model_config.get("loss_objective"),
        "budget_loss_ratios": model_config.get("budget_loss_ratios"),
        "budget_loss_temperature": model_config.get("budget_loss_temperature"),
        "temporal_distribution_loss_weight": model_config.get("temporal_distribution_loss_weight"),
        "range_train_workload_replicates": query_config.get("range_train_workload_replicates"),
        "validation_split_mode": data_config.get("validation_split_mode"),
        "val_fraction": data_config.get("val_fraction"),
        "eval_selector_matched_learned_slot_fraction": selector_budget_row.get(
            "learned_slot_fraction_of_budget"
        ),
        "eval_selector_matched_zero_learned_trajectory_fraction": selector_budget_row.get(
            "zero_learned_slot_trajectory_fraction"
        ),
        "eval_selector_matched_endpoint_only_trajectory_fraction": selector_budget_row.get(
            "endpoint_only_trajectory_fraction"
        ),
        **selector_low_budget_summary,
        "range_time_domain_mode": query_config.get("range_time_domain_mode"),
        "range_anchor_mode": query_config.get("range_anchor_mode"),
        "range_train_anchor_modes": query_config.get("range_train_anchor_modes"),
        "range_train_footprints": query_config.get("range_train_footprints"),
        "range_max_coverage_overshoot": query_config.get("range_max_coverage_overshoot"),
        "workload_profile_id": query_config.get("workload_profile_id"),
        "coverage_calibration_mode": query_config.get("coverage_calibration_mode"),
        "workload_stability_gate_mode_config": query_config.get("workload_stability_gate_mode"),
        **_workload_generation_fields(run_json, "train"),
        **_workload_generation_fields(run_json, "eval"),
        **_workload_generation_fields(run_json, "selection"),
        "checkpoint_full_score_every": model_config.get("checkpoint_full_score_every"),
        "checkpoint_candidate_pool_size": model_config.get("checkpoint_candidate_pool_size"),
        "mlqds_temporal_fraction": model_config.get("mlqds_temporal_fraction"),
        "mlqds_diversity_bonus": model_config.get("mlqds_diversity_bonus"),
        "mlqds_effective_diversity_bonus": _effective_diversity_bonus(model_config),
        "mlqds_hybrid_mode": model_config.get("mlqds_hybrid_mode"),
        "mlqds_stratified_center_weight": model_config.get("mlqds_stratified_center_weight"),
        "mlqds_min_learned_swaps": model_config.get("mlqds_min_learned_swaps"),
        "mlqds_score_mode": model_config.get("mlqds_score_mode"),
        "mlqds_score_temperature": model_config.get("mlqds_score_temperature"),
        "mlqds_rank_confidence_weight": model_config.get("mlqds_rank_confidence_weight"),
        "mlqds_range_geometry_blend": model_config.get("mlqds_range_geometry_blend"),
        "temporal_residual_label_mode": model_config.get("temporal_residual_label_mode"),
        "range_label_mode": model_config.get("range_label_mode"),
        "range_training_target_mode": model_config.get("range_training_target_mode"),
        "range_target_balance_mode": model_config.get("range_target_balance_mode"),
        "range_replicate_target_aggregation": model_config.get(
            "range_replicate_target_aggregation"
        ),
        "range_component_target_blend": model_config.get("range_component_target_blend"),
        "range_temporal_target_blend": model_config.get("range_temporal_target_blend"),
        "range_structural_target_blend": model_config.get("range_structural_target_blend"),
        "range_structural_target_source_mode": model_config.get(
            "range_structural_target_source_mode"
        ),
        "range_target_budget_weight_power": model_config.get("range_target_budget_weight_power"),
        "range_marginal_target_radius_scale": model_config.get(
            "range_marginal_target_radius_scale"
        ),
        "range_query_spine_fraction": model_config.get("range_query_spine_fraction"),
        "range_query_spine_mass_mode": model_config.get("range_query_spine_mass_mode"),
        "range_query_residual_multiplier": model_config.get("range_query_residual_multiplier"),
        "range_query_residual_mass_mode": model_config.get("range_query_residual_mass_mode"),
        "range_set_utility_multiplier": model_config.get("range_set_utility_multiplier"),
        "range_set_utility_candidate_limit": model_config.get("range_set_utility_candidate_limit"),
        "range_set_utility_mass_mode": model_config.get("range_set_utility_mass_mode"),
        **training_target_row_fields(
            model_config=model_config,
            teacher_distillation=teacher_distillation,
            train_label_diagnostics=train_label_diagnostics,
            label_mass_fraction=label_mass_fraction,
            target_diagnostics=target_diagnostics,
            target_transform=target_transform,
            fit_diagnostics=fit_diagnostics,
            target_budget_row=target_budget_row,
            oracle_diagnostic=oracle_diagnostic,
        ),
        "float32_matmul_precision": model_config.get("float32_matmul_precision"),
        "allow_tf32": model_config.get("allow_tf32"),
        "amp_mode": model_config.get("amp_mode"),
        "extra_args": "",
        "child_float32_matmul_precision": child_torch_runtime.get("float32_matmul_precision"),
        "child_tf32_matmul_allowed": child_torch_runtime.get("tf32_matmul_allowed"),
        "child_tf32_cudnn_allowed": child_torch_runtime.get("tf32_cudnn_allowed"),
        "child_amp_enabled": child_amp.get("enabled"),
        "child_amp_dtype": child_amp.get("dtype"),
        "child_torch_runtime": child_torch_runtime or None,
        "train_batch_size": model_config.get("train_batch_size"),
        "inference_batch_size": model_config.get("inference_batch_size"),
        "run_dir": str(run_dir),
        "example_run_path": str(run_json_path) if run_json_path.exists() else None,
        "stdout_path": str(stdout_path),
        "command": command,
    }
