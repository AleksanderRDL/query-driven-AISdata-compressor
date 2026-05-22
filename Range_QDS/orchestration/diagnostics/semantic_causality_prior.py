"""Prior-flow semantic causality summary helpers."""

from __future__ import annotations

from typing import Any

from orchestration.diagnostics.artifact_utils import (
    as_bool as _as_bool,
)
from orchestration.diagnostics.artifact_utils import (
    as_dict as _as_dict,
)
from orchestration.diagnostics.artifact_utils import (
    as_float as _as_float,
)

PRIOR_DIAGNOSTIC_NAMES = (
    "shuffled_prior_fields",
    "without_query_prior_features",
)


def _path(root: dict[str, Any], *keys: str) -> Any:
    cursor: Any = root
    for key in keys:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor


def _stage_summary(chain: dict[str, Any], key: str) -> dict[str, Any]:
    stage = _as_dict(chain.get(key))
    if key == "head_output":
        probability = _as_dict(stage.get("probability"))
        logit = _as_dict(stage.get("logit"))
        return {
            "available": _as_bool(stage.get("available")),
            "head_logits_changed": _as_bool(stage.get("head_logits_changed")),
            "head_probabilities_changed": _as_bool(stage.get("head_probabilities_changed")),
            "mean_abs_head_logit_delta": _as_float(stage.get("mean_abs_head_logit_delta")),
            "max_abs_head_logit_delta": _as_float(stage.get("max_abs_head_logit_delta")),
            "mean_abs_head_probability_delta": _as_float(
                stage.get("mean_abs_head_probability_delta")
            ),
            "max_abs_head_probability_delta": _as_float(
                stage.get("max_abs_head_probability_delta")
            ),
            "probability_per_head": {
                name: {
                    "mean_abs_delta": _as_float(_as_dict(values).get("mean_abs_delta")),
                    "max_abs_delta": _as_float(_as_dict(values).get("max_abs_delta")),
                    "primary_std": _as_float(_as_dict(values).get("primary_std")),
                }
                for name, values in _as_dict(probability.get("per_feature")).items()
            },
            "logit_per_head": {
                name: {
                    "mean_abs_delta": _as_float(_as_dict(values).get("mean_abs_delta")),
                    "max_abs_delta": _as_float(_as_dict(values).get("max_abs_delta")),
                    "primary_std": _as_float(_as_dict(values).get("primary_std")),
                }
                for name, values in _as_dict(logit.get("per_feature")).items()
            },
        }
    return {
        "available": _as_bool(stage.get("available")),
        "sampled_inputs_changed": _as_bool(stage.get("sampled_inputs_changed")),
        "mean_abs_feature_delta": _as_float(stage.get("mean_abs_feature_delta")),
        "max_abs_feature_delta": _as_float(stage.get("max_abs_feature_delta")),
        "mean_abs_score_delta": _as_float(stage.get("mean_abs_score_delta")),
        "max_abs_score_delta": _as_float(stage.get("max_abs_score_delta")),
        "primary_score_std": _as_float(stage.get("primary_score_std")),
        "ablation_score_std": _as_float(stage.get("ablation_score_std")),
        "retained_mask_changed": _as_bool(stage.get("retained_mask_changed")),
        "retained_mask_jaccard": _as_float(stage.get("retained_mask_jaccard")),
        "retained_mask_hamming_fraction": _as_float(
            stage.get("retained_mask_hamming_fraction")
        ),
        "score_topk_jaccard_at_retained_count": _as_float(
            stage.get("score_topk_jaccard_at_retained_count")
        ),
    }


def _row_path_composition_summary(group: dict[str, Any]) -> dict[str, Any]:
    negative = _as_dict(group.get("factorized_most_negative_mean_contribution"))
    positive = _as_dict(group.get("factorized_most_positive_mean_contribution"))
    return {
        "available": _as_bool(group.get("factorized_composition_available")),
        "composed_score_mean_delta": _as_float(
            group.get("factorized_composed_score_mean_delta")
        ),
        "composed_logit_mean_delta": _as_float(
            group.get("factorized_composed_logit_mean_delta")
        ),
        "raw_prediction_delta_residual_mean": _as_float(
            group.get("factorized_raw_prediction_delta_residual_mean")
        ),
        "contribution_mean_delta": group.get("factorized_contribution_mean_delta"),
        "contribution_positive_delta_fraction": group.get(
            "factorized_contribution_positive_delta_fraction"
        ),
        "most_negative_mean_contribution_name": negative.get("name"),
        "most_negative_mean_contribution_delta": _as_float(negative.get("delta")),
        "most_positive_mean_contribution_name": positive.get("name"),
        "most_positive_mean_contribution_delta": _as_float(positive.get("delta")),
    }


def _prior_chain_summary(chain: dict[str, Any]) -> dict[str, Any]:
    model_prior = _as_dict(chain.get("model_prior_features"))
    rank_margin = _as_dict(chain.get("score_rank_margin_boundary"))
    topk_boundary = _as_dict(rank_margin.get("topk_score_boundary"))
    marginal_alignment = _as_dict(rank_margin.get("marginal_row_score_delta_alignment"))
    row_path = _as_dict(chain.get("marginal_row_delta_path"))
    row_path_groups = _as_dict(row_path.get("groups"))
    row_path_top = _as_dict(row_path_groups.get("top_marginal"))
    row_path_missed = _as_dict(row_path_groups.get("missed_high_marginal"))
    row_path_under_ranked = _as_dict(row_path_groups.get("under_ranked_high_marginal"))
    return {
        "available": _as_bool(chain.get("available")),
        "diagnostic_chain": chain.get("diagnostic_chain"),
        "sampled_prior_features": _stage_summary(chain, "sampled_prior_features"),
        "model_prior_features": {
            "available": _as_bool(model_prior.get("available")),
            "disabled_prior_fields": model_prior.get("disabled_prior_fields"),
            "model_prior_feature_transform": model_prior.get("model_prior_feature_transform"),
            "model_input_prior_features": _stage_summary(
                model_prior, "model_input_prior_features"
            ),
            "normalized_model_prior_features": _stage_summary(
                model_prior, "normalized_model_prior_features"
            ),
        },
        "head_output": _stage_summary(chain, "head_output"),
        "raw_prediction": _stage_summary(chain, "raw_prediction"),
        "score_output": _stage_summary(chain, "score_output"),
        "retained_mask": _stage_summary(chain, "retained_mask"),
        "score_rank_margin_boundary": {
            "available": _as_bool(rank_margin.get("available")),
            "classification": rank_margin.get("classification"),
            "topk_boundary_margin": _as_float(topk_boundary.get("topk_boundary_margin")),
            "max_abs_score_delta_to_topk_boundary_margin": _as_float(
                topk_boundary.get("max_abs_score_delta_to_topk_boundary_margin")
            ),
            "score_delta_crosses_topk_boundary": _as_bool(
                topk_boundary.get("score_delta_crosses_topk_boundary")
            ),
            "score_delta_to_marginal_spearman": _as_float(
                marginal_alignment.get("score_delta_to_marginal_spearman")
            ),
            "top_marginal_mean_score_delta": _as_float(
                marginal_alignment.get("top_marginal_mean_score_delta")
            ),
            "missed_high_marginal_mean_score_delta": _as_float(
                marginal_alignment.get("missed_high_marginal_mean_score_delta")
            ),
            "under_ranked_high_marginal_mean_score_delta": _as_float(
                marginal_alignment.get("under_ranked_high_marginal_mean_score_delta")
            ),
            "marginal_alignment_classification": marginal_alignment.get("classification"),
        },
        "marginal_row_delta_path": {
            "available": _as_bool(row_path.get("available")),
            "classification": row_path.get("classification"),
            "row_count": _as_float(row_path.get("row_count")),
            "stage_available": row_path.get("stage_available"),
            "top_marginal_score_output_mean_delta": _as_float(
                row_path_top.get("score_output_mean_delta")
            ),
            "top_marginal_raw_prediction_mean_delta": _as_float(
                row_path_top.get("raw_prediction_mean_delta")
            ),
            "top_marginal_segment_score_mean_delta": _as_float(
                row_path_top.get("segment_score_mean_delta")
            ),
            "top_marginal_max_head_probability_mean_delta": _as_float(
                row_path_top.get("max_head_probability_mean_delta")
            ),
            "top_marginal_max_head_probability_mean_delta_head": row_path_top.get(
                "max_head_probability_mean_delta_head"
            ),
            "top_marginal_max_head_logit_mean_delta": _as_float(
                row_path_top.get("max_head_logit_mean_delta")
            ),
            "top_marginal_max_head_logit_mean_delta_head": row_path_top.get(
                "max_head_logit_mean_delta_head"
            ),
            "missed_high_marginal_score_output_mean_delta": _as_float(
                row_path_missed.get("score_output_mean_delta")
            ),
            "missed_high_marginal_raw_prediction_mean_delta": _as_float(
                row_path_missed.get("raw_prediction_mean_delta")
            ),
            "under_ranked_high_marginal_score_output_mean_delta": _as_float(
                row_path_under_ranked.get("score_output_mean_delta")
            ),
            "under_ranked_high_marginal_raw_prediction_mean_delta": _as_float(
                row_path_under_ranked.get("raw_prediction_mean_delta")
            ),
            "top_marginal_factorized_composition": _row_path_composition_summary(
                row_path_top
            ),
            "missed_high_marginal_factorized_composition": (
                _row_path_composition_summary(row_path_missed)
            ),
            "under_ranked_high_marginal_factorized_composition": (
                _row_path_composition_summary(row_path_under_ranked)
            ),
        },
    }


def _classify_prior_flow(chains: dict[str, Any]) -> dict[str, Any]:
    shuffled = _as_dict(chains.get("shuffled_prior_fields"))
    sampled_changed = _path(shuffled, "sampled_prior_features", "sampled_inputs_changed")
    model_changed = _path(
        shuffled,
        "model_prior_features",
        "normalized_model_prior_features",
        "sampled_inputs_changed",
    )
    head_delta = _as_float(
        _path(shuffled, "head_output", "mean_abs_head_probability_delta")
    )
    score_delta = _as_float(_path(shuffled, "score_output", "mean_abs_score_delta"))
    mask_changed = _path(shuffled, "retained_mask", "retained_mask_changed")
    topk_jaccard = _as_float(
        _path(shuffled, "score_output", "score_topk_jaccard_at_retained_count")
    )
    if sampled_changed is not True:
        category = "prior sampling/support failure"
    elif model_changed is not True:
        category = "prior feature normalization/scaling failure"
    elif (head_delta is None or head_delta < 0.0001) and mask_changed is False:
        category = "model ignores prior inputs"
    elif score_delta is not None and score_delta > 0.0 and mask_changed is False:
        category = "priors change scores but selector ignores them"
    elif topk_jaccard == 1.0 and mask_changed is False:
        category = "priors change scores but selector ignores them"
    else:
        category = "artifact supports only partial prior-flow classification"
    return {
        "protocol_category": category,
        "sampled_priors_changed": sampled_changed,
        "model_priors_changed": model_changed,
        "mean_abs_head_probability_delta": head_delta,
        "mean_abs_score_delta": score_delta,
        "retained_mask_changed": mask_changed,
        "score_topk_jaccard_at_retained_count": topk_jaccard,
    }


def _output_layer_alignment_summary(alignment_raw: Any) -> dict[str, Any]:
    alignment = _as_dict(alignment_raw)
    return {
        "available": _as_bool(alignment.get("available")),
        "final_weight_to_hidden_delta_abs_cosine_mean": _as_float(
            alignment.get("final_weight_to_hidden_delta_abs_cosine_mean")
        ),
        "projected_hidden_delta_l2_to_hidden_delta_l2": _as_float(
            alignment.get("projected_hidden_delta_l2_to_hidden_delta_l2")
        ),
        "target_to_logit_delta_spearman": _as_float(
            alignment.get("target_to_logit_delta_spearman")
        ),
        "target_to_projected_hidden_delta_spearman": _as_float(
            alignment.get("target_to_projected_hidden_delta_spearman")
        ),
        "bce_descent_alignment_mean": _as_float(
            alignment.get("bce_descent_alignment_mean")
        ),
        "bce_descent_alignment_positive_fraction": _as_float(
            alignment.get("bce_descent_alignment_positive_fraction")
        ),
        "slice_alignment": alignment.get("slice_alignment"),
    }


def _prior_channel_direction_decomposition_summary(transfer: dict[str, Any]) -> dict[str, Any]:
    decomposition = _as_dict(transfer.get("prior_channel_direction_decomposition"))
    if not decomposition:
        return {"available": False}
    per_channel: dict[str, Any] = {}
    for channel_name, channel_raw in _as_dict(decomposition.get("per_channel")).items():
        channel = _as_dict(channel_raw)
        per_head: dict[str, Any] = {}
        for head_name, head_raw in _as_dict(channel.get("per_head")).items():
            head = _as_dict(head_raw)
            per_head[str(head_name)] = {
                "available": _as_bool(head.get("available")),
                "classification": head.get("classification"),
                "output_layer_alignment": _output_layer_alignment_summary(
                    head.get("output_layer_alignment")
                ),
            }
        per_channel[str(channel_name)] = {
            "available": _as_bool(channel.get("available")),
            "per_head": per_head,
        }
    return {
        "available": _as_bool(decomposition.get("available")),
        "diagnostic_only": _as_bool(decomposition.get("diagnostic_only")),
        "channel_count": _as_float(decomposition.get("channel_count")),
        "classification_thresholds": decomposition.get("classification_thresholds"),
        "classification_counts": decomposition.get("classification_counts"),
        "by_head": decomposition.get("by_head"),
        "per_channel": per_channel,
    }


def _prior_learning_signal_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    signal = _as_dict(
        _path(artifact, "training_fit_diagnostics", "prior_feature_learning_signal")
    )
    if not signal:
        return {"available": False}
    path_strength = _as_dict(signal.get("prior_path_strength"))
    reconstruction = _as_dict(signal.get("prior_reconstruction_from_non_prior_features"))
    stage = _as_dict(signal.get("prior_stage_sensitivity"))
    transfer = _as_dict(signal.get("prior_to_head_transfer_sensitivity"))
    zero_sensitivity = _as_dict(signal.get("zero_prior_sensitivity"))
    head_sensitivity = _as_dict(zero_sensitivity.get("head_probabilities"))
    final_probability = _as_dict(zero_sensitivity.get("final_probability"))
    final_logit = _as_dict(zero_sensitivity.get("final_logit"))
    return {
        "available": _as_bool(signal.get("prior_feature_learning_diagnostics_available")),
        "classification": signal.get("classification"),
        "prior_signal_head_count": _as_float(signal.get("prior_signal_head_count")),
        "prior_best_spearman_beats_non_prior_head_count": _as_float(
            signal.get("prior_best_spearman_beats_non_prior_head_count")
        ),
        "prior_path_strength": {
            "available": _as_bool(path_strength.get("available")),
            "prior_feature_scale": _as_float(path_strength.get("prior_feature_scale")),
            "scaled_prior_to_point_std_ratio": _as_float(
                path_strength.get("scaled_prior_to_point_std_ratio")
            ),
            "scaled_prior_to_point_l2_ratio": _as_float(
                path_strength.get("scaled_prior_to_point_l2_ratio")
            ),
        },
        "prior_reconstruction_from_non_prior_features": {
            "available": _as_bool(reconstruction.get("available")),
            "mean_r2": _as_float(reconstruction.get("mean_r2")),
            "max_r2": _as_float(reconstruction.get("max_r2")),
        },
        "prior_stage_sensitivity": {
            "available": _as_bool(stage.get("available")),
            "shared_to_pre_context_mean_abs_delta_ratio": _as_float(
                stage.get("shared_to_pre_context_mean_abs_delta_ratio")
            ),
            "head_probability_to_pre_context_mean_abs_delta_ratio": _as_float(
                stage.get("head_probability_to_pre_context_mean_abs_delta_ratio")
            ),
            "pre_context_mean_abs_delta": _as_float(
                _path(stage, "stage_sensitivity", "pre_context_sum", "mean_abs_delta")
            ),
            "post_local_context_mean_abs_delta": _as_float(
                _path(stage, "stage_sensitivity", "post_local_context", "mean_abs_delta")
            ),
            "pre_shared_mean_abs_delta": _as_float(
                _path(stage, "stage_sensitivity", "pre_shared_encoder_sum", "mean_abs_delta")
            ),
            "shared_embedding_mean_abs_delta": _as_float(
                _path(stage, "stage_sensitivity", "shared_embedding", "mean_abs_delta")
            ),
            "head_probability_mean_abs_delta": _as_float(
                _path(stage, "stage_sensitivity", "head_probabilities", "mean_abs_delta")
            ),
        },
        "prior_to_head_transfer_sensitivity": {
            "available": _as_bool(transfer.get("available")),
            "classification_counts": transfer.get("classification_counts"),
            "per_head": {
                str(name): {
                    "classification": _as_dict(row).get("classification"),
                    "first_linear_delta_l2_to_shared_delta_l2": _as_float(
                        _as_dict(row).get("first_linear_delta_l2_to_shared_delta_l2")
                    ),
                    "hidden_delta_l2_to_first_linear_delta_l2": _as_float(
                        _as_dict(row).get("hidden_delta_l2_to_first_linear_delta_l2")
                    ),
                    "logit_delta_l2_to_hidden_delta_l2": _as_float(
                        _as_dict(row).get("logit_delta_l2_to_hidden_delta_l2")
                    ),
                    "probability_mean_abs_delta_to_logit_mean_abs_delta": _as_float(
                        _as_dict(row).get("probability_mean_abs_delta_to_logit_mean_abs_delta")
                    ),
                    "sigmoid_derivative_mean": _as_float(
                        _as_dict(row).get("sigmoid_derivative_mean")
                    ),
                    "target_std": _as_float(_as_dict(row).get("target_std")),
                    "output_layer_alignment": {
                        **_output_layer_alignment_summary(
                            _as_dict(row).get("output_layer_alignment")
                        )
                    },
                    "configured_loss_gradient_alignment": {
                        "available": _as_bool(
                            _path(row, "configured_loss_gradient_alignment", "available")
                        ),
                        "loss_scope": _path(
                            row,
                            "configured_loss_gradient_alignment",
                            "loss_scope",
                        ),
                        "descent_alignment_mean": _as_float(
                            _path(
                                row,
                                "configured_loss_gradient_alignment",
                                "descent_alignment_mean",
                            )
                        ),
                        "descent_alignment_positive_fraction": _as_float(
                            _path(
                                row,
                                "configured_loss_gradient_alignment",
                                "descent_alignment_positive_fraction",
                            )
                        ),
                        "logit_gradient_abs_mean": _as_float(
                            _path(
                                row,
                                "configured_loss_gradient_alignment",
                                "logit_gradient_abs_mean",
                            )
                        ),
                    },
                    "probability_mean_abs_delta": _as_float(
                        _path(row, "stage_sensitivity", "probability", "mean_abs_delta")
                    )
                    if isinstance(row, dict)
                    else None,
                    "logit_mean_abs_delta": _as_float(
                        _path(row, "stage_sensitivity", "logit", "mean_abs_delta")
                    )
                    if isinstance(row, dict)
                    else None,
                }
                for name, row in _as_dict(transfer.get("per_head")).items()
            },
            "prior_channel_direction_decomposition": (
                _prior_channel_direction_decomposition_summary(transfer)
            ),
        },
        "zero_prior_sensitivity": {
            "mean_abs_head_probability_delta": _as_float(
                head_sensitivity.get("mean_abs_head_probability_delta")
            ),
            "max_abs_head_probability_delta": _as_float(
                head_sensitivity.get("max_abs_head_probability_delta")
            ),
            "mean_abs_final_probability_delta": _as_float(
                final_probability.get("mean_abs_delta")
            ),
            "mean_abs_final_logit_delta": _as_float(final_logit.get("mean_abs_delta")),
        },
    }


def prior_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    causality = _as_dict(artifact.get("learning_causality_summary"))
    prior = _as_dict(causality.get("prior_sensitivity_diagnostics"))
    chains = {
        name: _prior_chain_summary(_as_dict(prior.get(name)))
        for name in PRIOR_DIAGNOSTIC_NAMES
    }
    channel_diag = _as_dict(causality.get("prior_channel_ablation_diagnostics"))
    channel_summary = {
        name: {
            "sampled_mean_abs_delta": _as_float(
                _path(diag, "sampled_prior_features", "mean_abs_feature_delta")
            ),
            "model_mean_abs_delta": _as_float(
                _path(
                    diag,
                    "model_prior_features",
                    "normalized_model_prior_features",
                    "mean_abs_feature_delta",
                )
            ),
            "head_probability_mean_abs_delta": _as_float(
                _path(diag, "head_output", "mean_abs_head_probability_delta")
            ),
            "raw_prediction_mean_abs_delta": _as_float(
                _path(diag, "raw_prediction", "mean_abs_score_delta")
            ),
            "score_output_mean_abs_delta": _as_float(
                _path(diag, "score_output", "mean_abs_score_delta")
            ),
            "retained_mask_changed": _as_bool(
                _path(diag, "retained_mask", "retained_mask_changed")
            ),
            "score_topk_jaccard_at_retained_count": _as_float(
                _path(diag, "score_output", "score_topk_jaccard_at_retained_count")
            ),
        }
        for name, diag in channel_diag.items()
        if isinstance(diag, dict)
    }
    return {
        "classification": _classify_prior_flow(prior),
        "ablation_deltas": {
            "shuffled_prior_field_ablation_delta": _as_float(
                causality.get("shuffled_prior_field_ablation_delta")
            ),
            "without_query_prior_features_delta": _as_float(
                causality.get("without_query_prior_features_delta")
            ),
            "threshold_shuffled_prior_fields_should_lose": _path(
                causality,
                "learning_causality_delta_gate",
                "thresholds",
                "shuffled_prior_fields_should_lose",
            ),
            "threshold_without_query_prior_features_should_lose": _path(
                causality,
                "learning_causality_delta_gate",
                "thresholds",
                "without_query_prior_features_should_lose",
            ),
        },
        "chains": chains,
        "prior_channel_ablation_summary": channel_summary,
        "training_prior_learning_signal": _prior_learning_signal_summary(artifact),
        "prior_sample_gate_pass": _as_bool(causality.get("prior_sample_gate_pass")),
        "prior_sample_gate_failures": causality.get("prior_sample_gate_failures"),
    }

