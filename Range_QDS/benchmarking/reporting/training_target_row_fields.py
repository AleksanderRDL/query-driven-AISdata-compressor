"""Training-target benchmark row field construction."""

from __future__ import annotations

from typing import Any


def training_target_row_fields(
    *,
    model_config: dict[str, Any],
    teacher_distillation: dict[str, Any],
    train_label_diagnostics: dict[str, Any],
    label_mass_fraction: dict[str, Any],
    target_diagnostics: dict[str, Any],
    target_transform: dict[str, Any],
    fit_diagnostics: dict[str, Any],
    target_budget_row: dict[str, Any],
    oracle_diagnostic: dict[str, Any],
) -> dict[str, Any]:
    """Return row fields for training target and teacher diagnostics."""
    return {
        "local_swap_utility_scored_candidate_count": target_transform.get(
            "local_swap_utility_scored_candidate_count"
        ),
        "local_swap_utility_positive_gain_candidate_count": target_transform.get(
            "local_swap_utility_positive_gain_candidate_count"
        ),
        "local_swap_utility_selected_count": target_transform.get(
            "local_swap_utility_selected_count"
        ),
        "local_swap_utility_selected_gain_mass": target_transform.get(
            "local_swap_utility_selected_gain_mass"
        ),
        "local_swap_utility_source_positive_mass": target_transform.get(
            "local_swap_utility_source_positive_mass"
        ),
        "local_swap_gain_cost_scored_candidate_count": target_transform.get(
            "local_swap_gain_cost_scored_candidate_count"
        ),
        "local_swap_gain_cost_positive_net_gain_count": target_transform.get(
            "local_swap_gain_cost_positive_net_gain_count"
        ),
        "local_swap_gain_cost_selected_count": target_transform.get(
            "local_swap_gain_cost_selected_count"
        ),
        "local_swap_gain_cost_selected_candidate_value_mass": target_transform.get(
            "local_swap_gain_cost_selected_candidate_value_mass"
        ),
        "local_swap_gain_cost_selected_removal_cost_mass": target_transform.get(
            "local_swap_gain_cost_selected_removal_cost_mass"
        ),
        "local_swap_gain_cost_source_positive_mass": target_transform.get(
            "local_swap_gain_cost_source_positive_mass"
        ),
        "range_boundary_prior_weight": model_config.get("range_boundary_prior_weight"),
        "range_boundary_prior_enabled": bool(
            float(model_config.get("range_boundary_prior_weight") or 0.0) > 0.0
        ),
        "range_teacher_distillation_mode": model_config.get("range_teacher_distillation_mode"),
        "range_teacher_epochs": model_config.get("range_teacher_epochs"),
        "teacher_distillation_enabled": teacher_distillation.get("enabled"),
        "teacher_distillation_mode": teacher_distillation.get("mode"),
        "teacher_model_type": teacher_distillation.get("teacher_model_type"),
        "teacher_replicate_count": teacher_distillation.get("replicate_count"),
        "teacher_positive_label_count": teacher_distillation.get("positive_label_count"),
        "teacher_positive_label_fraction": teacher_distillation.get("positive_label_fraction"),
        "teacher_positive_label_mass": teacher_distillation.get("positive_label_mass"),
        "train_positive_label_mass": train_label_diagnostics.get("positive_label_mass"),
        "train_label_mass_basis": train_label_diagnostics.get("component_label_mass_basis"),
        "train_label_mass_range_point_f1": label_mass_fraction.get("range_point_f1"),
        "train_label_mass_range_ship_f1": label_mass_fraction.get("range_ship_f1"),
        "train_label_mass_range_ship_coverage": label_mass_fraction.get("range_ship_coverage"),
        "train_label_mass_range_entry_exit_f1": label_mass_fraction.get("range_entry_exit_f1"),
        "train_label_mass_range_crossing_f1": label_mass_fraction.get("range_crossing_f1"),
        "train_label_mass_range_temporal_coverage": label_mass_fraction.get(
            "range_temporal_coverage"
        ),
        "train_label_mass_range_gap_coverage": label_mass_fraction.get("range_gap_coverage"),
        "train_label_mass_range_turn_coverage": label_mass_fraction.get("range_turn_coverage"),
        "train_label_mass_range_shape_score": label_mass_fraction.get("range_shape_score"),
        "train_target_positive_label_mass": target_diagnostics.get("positive_label_mass"),
        "range_target_transform_mode": target_transform.get("mode"),
        "range_target_transform_target_family": target_transform.get("target_family"),
        "range_target_transform_final_success_allowed": target_transform.get(
            "final_success_allowed"
        ),
        "range_target_transform_positive_label_count": target_transform.get("positive_label_count"),
        "range_target_transform_positive_label_fraction": target_transform.get(
            "positive_label_fraction"
        ),
        "range_target_transform_positive_label_mass": target_transform.get("positive_label_mass"),
        "range_target_transform_base_positive_label_mass": target_transform.get(
            "base_retained_frequency_positive_label_mass"
        ),
        "range_structural_score_positive_mass": target_transform.get(
            "structural_score_positive_mass"
        ),
        "range_structural_score_p95": target_transform.get("structural_score_p95"),
        "historical_prior_teacher_score_p95": target_transform.get(
            "historical_prior_teacher_score_p95"
        ),
        "historical_prior_teacher_score_mass": target_transform.get(
            "historical_prior_teacher_score_mass"
        ),
        "historical_prior_teacher_positive_score_fraction": target_transform.get(
            "historical_prior_teacher_positive_score_fraction"
        ),
        "historical_prior_teacher_support_count": target_transform.get(
            "historical_prior_stored_support_count"
        ),
        "train_fit_score_target_kendall_tau": fit_diagnostics.get("score_target_kendall_tau"),
        "train_fit_model_fits_stored_train_support": fit_diagnostics.get(
            "model_fits_stored_train_support"
        ),
        "train_fit_matched_mlqds_target_recall": fit_diagnostics.get("matched_mlqds_target_recall"),
        "train_fit_matched_uniform_target_recall": fit_diagnostics.get(
            "matched_uniform_target_recall"
        ),
        "train_fit_matched_mlqds_vs_uniform_target_recall": fit_diagnostics.get(
            "matched_mlqds_vs_uniform_target_recall"
        ),
        "train_fit_low_budget_mean_mlqds_vs_uniform_target_recall": fit_diagnostics.get(
            "low_budget_mean_mlqds_vs_uniform_target_recall"
        ),
        "train_target_budget_ratio": target_budget_row.get("total_budget_ratio"),
        "train_target_effective_fill_budget_ratio": target_budget_row.get(
            "effective_fill_budget_ratio"
        ),
        "train_target_temporal_base_label_mass_fraction": target_budget_row.get(
            "temporal_base_label_mass_fraction"
        ),
        "train_target_residual_label_mass_fraction": target_budget_row.get(
            "residual_label_mass_fraction"
        ),
        "train_target_residual_positive_label_fraction": target_budget_row.get(
            "residual_positive_label_fraction"
        ),
        "oracle_kind": oracle_diagnostic.get("kind"),
        "oracle_exact_optimum": oracle_diagnostic.get("exact_optimum"),
    }

