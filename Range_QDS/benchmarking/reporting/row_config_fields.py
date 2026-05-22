"""Config and training-target fields for benchmark reporting rows."""

from __future__ import annotations

from benchmarking.reporting.audit_extractors import (
    _selector_low_budget_summary,
    _workload_generation_fields,
)
from benchmarking.reporting.metrics import _effective_diversity_bonus
from benchmarking.reporting.row_context import RowContext, RowFields, _mapping
from benchmarking.reporting.training_target_row_fields import training_target_row_fields
from learning.model_features import model_type_metadata


def _model_identity_fields(ctx: RowContext) -> RowFields:
    model_config = ctx.model_config
    target_diagnostics = ctx.target_diagnostics
    return {
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
    }


def _core_training_config_fields(ctx: RowContext) -> RowFields:
    model_config = ctx.model_config
    query_config = ctx.query_config
    data_config = ctx.data_config
    return {
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
    }


def _selector_query_config_fields(ctx: RowContext) -> RowFields:
    query_config = ctx.query_config
    selector_budget_row = ctx.selector_budget_row
    return {
        "eval_selector_matched_learned_slot_fraction": selector_budget_row.get(
            "learned_slot_fraction_of_budget"
        ),
        "eval_selector_matched_zero_learned_trajectory_fraction": selector_budget_row.get(
            "zero_learned_slot_trajectory_fraction"
        ),
        "eval_selector_matched_endpoint_only_trajectory_fraction": selector_budget_row.get(
            "endpoint_only_trajectory_fraction"
        ),
        **_selector_low_budget_summary(ctx.eval_selector_diagnostics),
        "range_time_domain_mode": query_config.get("range_time_domain_mode"),
        "range_anchor_mode": query_config.get("range_anchor_mode"),
        "range_train_anchor_modes": query_config.get("range_train_anchor_modes"),
        "range_train_footprints": query_config.get("range_train_footprints"),
        "range_max_coverage_overshoot": query_config.get("range_max_coverage_overshoot"),
        "workload_profile_id": query_config.get("workload_profile_id"),
        "coverage_calibration_mode": query_config.get("coverage_calibration_mode"),
        "workload_stability_gate_mode_config": query_config.get("workload_stability_gate_mode"),
        **_workload_generation_fields(ctx.run_json, "train"),
        **_workload_generation_fields(ctx.run_json, "eval"),
        **_workload_generation_fields(ctx.run_json, "selection"),
    }


def _mlqds_selector_config_fields(ctx: RowContext) -> RowFields:
    model_config = ctx.model_config
    return {
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
    }


def _range_target_config_fields(ctx: RowContext) -> RowFields:
    model_config = ctx.model_config
    return {
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
    }


def _training_target_fields(ctx: RowContext) -> RowFields:
    train_label_diagnostics = ctx.train_label_diagnostics
    label_mass_fraction = _mapping(
        train_label_diagnostics.get("component_positive_label_mass_fraction")
    )
    return training_target_row_fields(
        model_config=ctx.model_config,
        teacher_distillation=_mapping(ctx.run.get("teacher_distillation")),
        train_label_diagnostics=train_label_diagnostics,
        label_mass_fraction=label_mass_fraction,
        target_diagnostics=ctx.target_diagnostics,
        target_transform=_mapping(ctx.run.get("range_training_target_transform")),
        fit_diagnostics=_mapping(ctx.run.get("training_fit_diagnostics")),
        target_budget_row=ctx.target_budget_row,
        oracle_diagnostic=_mapping(ctx.run.get("oracle_diagnostic")),
    )


def _config_training_fields(ctx: RowContext) -> RowFields:
    fields: RowFields = {}
    fields.update(_model_identity_fields(ctx))
    fields.update(_core_training_config_fields(ctx))
    fields.update(_selector_query_config_fields(ctx))
    fields.update(_mlqds_selector_config_fields(ctx))
    fields.update(_range_target_config_fields(ctx))
    fields.update(_training_target_fields(ctx))
    return fields
