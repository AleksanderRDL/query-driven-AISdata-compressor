"""Learning-causality fields for benchmark reporting rows."""

from __future__ import annotations

from benchmarking.reporting.row_context import RowContext, RowFields, _mapping


def _learning_summary_fields(learning_causality: RowFields) -> RowFields:
    return {
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
    }


def _ablation_mask_fields(learning_causality: RowFields) -> RowFields:
    causality_mask_diagnostics = _mapping(
        learning_causality.get("causality_ablation_mask_diagnostics")
    )
    shuffled_prior_mask = _mapping(causality_mask_diagnostics.get("MLQDS_shuffled_prior_fields"))
    no_query_prior_mask = _mapping(
        causality_mask_diagnostics.get("MLQDS_without_query_prior_features")
    )
    no_behavior_mask = _mapping(
        causality_mask_diagnostics.get("MLQDS_without_behavior_utility_head")
    )
    no_segment_budget_mask = _mapping(
        causality_mask_diagnostics.get("MLQDS_without_segment_budget_head")
    )
    no_geometry_mask = _mapping(
        causality_mask_diagnostics.get("MLQDS_without_geometry_tie_breaker")
    )
    no_length_support_allocation_mask = _mapping(
        causality_mask_diagnostics.get("MLQDS_without_segment_length_support_allocation")
    )
    return {
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
    }


def _delta_gate_and_selector_config_fields(
    ctx: RowContext, learning_causality: RowFields
) -> RowFields:
    model_config = ctx.model_config
    learning_delta_gate = _mapping(learning_causality.get("learning_causality_delta_gate"))
    selector_config = _mapping(learning_causality.get("learned_segment_selector_config"))
    return {
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
        "learned_segment_geometry_gain_weight": selector_config.get(
            "geometry_gain_weight", model_config.get("learned_segment_geometry_gain_weight")
        ),
        "learned_segment_allocation_length_support_weight": selector_config.get(
            "allocation_length_support_weight",
            model_config.get("learned_segment_allocation_length_support_weight"),
        ),
        "learned_segment_allocation_weight_floor": selector_config.get(
            "allocation_weight_floor",
            model_config.get("learned_segment_allocation_weight_floor"),
        ),
        "learned_segment_score_blend_weight": selector_config.get(
            "segment_score_blend_weight", model_config.get("learned_segment_score_blend_weight")
        ),
        "learned_segment_transfer_calibration_mode": selector_config.get(
            "segment_transfer_calibration_mode",
            model_config.get("learned_segment_transfer_calibration_mode"),
        ),
        "learned_segment_fairness_preallocation_enabled": selector_config.get(
            "fairness_preallocation_enabled",
            model_config.get("learned_segment_fairness_preallocation"),
        ),
        "learned_segment_length_repair_fraction": selector_config.get(
            "length_repair_fraction", model_config.get("learned_segment_length_repair_fraction")
        ),
        "learned_segment_length_repair_score_protection_fraction": selector_config.get(
            "length_repair_score_protection_fraction",
            model_config.get("learned_segment_length_repair_score_protection_fraction"),
        ),
        "learned_segment_length_support_blend_weight": selector_config.get(
            "length_support_blend_weight",
            model_config.get("learned_segment_length_support_blend_weight"),
        ),
    }


def _prior_ablation_fields(learning_causality: RowFields) -> RowFields:
    prior_sensitivity = _mapping(learning_causality.get("prior_sensitivity_diagnostics"))
    shuffled_prior = _mapping(prior_sensitivity.get("shuffled_prior_fields"))
    shuffled_sample = _mapping(shuffled_prior.get("sampled_prior_features"))
    shuffled_model = _mapping(shuffled_prior.get("model_prior_features"))
    shuffled_head = _mapping(shuffled_prior.get("head_output"))
    shuffled_score = _mapping(shuffled_prior.get("score_output"))
    shuffled_model_input = _mapping(shuffled_model.get("model_input_prior_features"))
    shuffled_normalized = _mapping(shuffled_model.get("normalized_model_prior_features"))
    no_prior = _mapping(prior_sensitivity.get("without_query_prior_features"))
    no_prior_sample = _mapping(no_prior.get("sampled_prior_features"))
    no_prior_model = _mapping(no_prior.get("model_prior_features"))
    no_prior_head = _mapping(no_prior.get("head_output"))
    no_prior_score = _mapping(no_prior.get("score_output"))
    no_prior_model_input = _mapping(no_prior_model.get("model_input_prior_features"))
    no_prior_normalized = _mapping(no_prior_model.get("normalized_model_prior_features"))
    return {
        "prior_sample_gate_pass": learning_causality.get("prior_sample_gate_pass"),
        "prior_sample_gate_failures": learning_causality.get("prior_sample_gate_failures"),
        "shuffled_prior_sampled_inputs_changed": shuffled_sample.get("sampled_inputs_changed"),
        "shuffled_prior_sampled_primary_nonzero_fraction": shuffled_sample.get(
            "primary_nonzero_fraction"
        ),
        "shuffled_prior_sampled_ablation_nonzero_fraction": shuffled_sample.get(
            "ablation_nonzero_fraction"
        ),
        "shuffled_prior_sampled_mean_abs_feature_delta": shuffled_sample.get(
            "mean_abs_feature_delta"
        ),
        "shuffled_prior_sampled_max_abs_feature_delta": shuffled_sample.get(
            "max_abs_feature_delta"
        ),
        "shuffled_prior_sampled_outside_extent_fraction": shuffled_sample.get(
            "points_outside_prior_extent_fraction"
        ),
        "shuffled_prior_model_inputs_changed": shuffled_model_input.get("sampled_inputs_changed"),
        "shuffled_prior_model_input_mean_abs_feature_delta": shuffled_model_input.get(
            "mean_abs_feature_delta"
        ),
        "shuffled_prior_normalized_model_inputs_changed": shuffled_normalized.get(
            "sampled_inputs_changed"
        ),
        "shuffled_prior_normalized_model_mean_abs_feature_delta": shuffled_normalized.get(
            "mean_abs_feature_delta"
        ),
        "shuffled_prior_head_logits_changed": shuffled_head.get("head_logits_changed"),
        "shuffled_prior_head_logit_mean_abs_delta": shuffled_head.get("mean_abs_head_logit_delta"),
        "shuffled_prior_head_probability_mean_abs_delta": shuffled_head.get(
            "mean_abs_head_probability_delta"
        ),
        "shuffled_prior_score_output_mean_abs_delta": shuffled_score.get("mean_abs_score_delta"),
        "shuffled_prior_score_output_max_abs_delta": shuffled_score.get("max_abs_score_delta"),
        "shuffled_prior_score_output_topk_jaccard_at_retained_count": shuffled_score.get(
            "score_topk_jaccard_at_retained_count"
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
        "no_prior_head_logits_changed": no_prior_head.get("head_logits_changed"),
        "no_prior_head_logit_mean_abs_delta": no_prior_head.get("mean_abs_head_logit_delta"),
        "no_prior_head_probability_mean_abs_delta": no_prior_head.get(
            "mean_abs_head_probability_delta"
        ),
        "no_prior_score_output_mean_abs_delta": no_prior_score.get("mean_abs_score_delta"),
        "no_prior_score_output_max_abs_delta": no_prior_score.get("max_abs_score_delta"),
        "no_prior_score_output_topk_jaccard_at_retained_count": no_prior_score.get(
            "score_topk_jaccard_at_retained_count"
        ),
    }


def _learning_causality_fields(ctx: RowContext) -> RowFields:
    learning_causality = _mapping(ctx.run.get("learning_causality_summary"))
    fields: RowFields = {}
    fields.update(_learning_summary_fields(learning_causality))
    fields.update(_ablation_mask_fields(learning_causality))
    fields.update(_delta_gate_and_selector_config_fields(ctx, learning_causality))
    fields.update(_prior_ablation_fields(learning_causality))
    return fields
