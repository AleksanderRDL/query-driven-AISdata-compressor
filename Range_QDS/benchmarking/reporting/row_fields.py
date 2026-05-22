"""Benchmark child-run row construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarking.reporting.audit_extractors import _data_source_row_fields
from benchmarking.reporting.metrics import (
    _metric_beats,
    _single_cell_range_status,
)
from benchmarking.reporting.row_config_fields import _config_training_fields
from benchmarking.reporting.row_context import RowContext, RowFields, _mapping
from benchmarking.reporting.row_learning_causality import _learning_causality_fields
from benchmarking.reporting.row_metric_fields import _method_metric_fields
from benchmarking.row_runtime import (
    dominant_runtime_phase_fields,
    last_history_value,
    mean_epoch_seconds,
    mean_history_value,
    phase_seconds,
    phase_seconds_with_prefix,
)
from learning.model_features import is_workload_blind_model_type


def _single_cell_status(ctx: RowContext) -> str:
    workload_blind_protocol = ctx.workload_blind_protocol
    return _single_cell_range_status(
        returncode=ctx.returncode,
        model_type=ctx.model_config.get("model_type"),
        protocol_enabled=workload_blind_protocol.get("enabled"),
        primary_frozen=workload_blind_protocol.get(
            "primary_masks_frozen_before_eval_query_scoring"
        ),
        audit_frozen=workload_blind_protocol.get("audit_masks_frozen_before_eval_query_scoring"),
        audit_ratio_count=int(ctx.audit["audit_compression_ratio_count"]),
        beats_uniform=_metric_beats(ctx.mlqds, ctx.uniform, "range_usefulness_score"),
        beats_dp=_metric_beats(ctx.mlqds, ctx.douglas_peucker, "range_usefulness_score"),
        selector_claim_status=str(ctx.selector_claim_evidence["selector_claim_status"]),
    )


def _identity_runtime_fields(ctx: RowContext) -> RowFields:
    final_claim_summary = _mapping(ctx.run.get("final_claim_summary"))
    return {
        "workload": ctx.workload,
        "run_label": ctx.run_label,
        **_data_source_row_fields(ctx.data_sources),
        "returncode": int(ctx.returncode),
        "elapsed_seconds": float(ctx.elapsed_seconds),
        "train_seconds": phase_seconds_with_prefix(ctx.timings, "train-model"),
        "evaluate_matched_seconds": phase_seconds(ctx.timings, "evaluate-matched"),
        "epoch_mean_seconds": mean_epoch_seconds(ctx.timings),
        "peak_allocated_mb": ctx.cuda_memory.get("max_allocated_mb"),
        "best_epoch": ctx.run.get("best_epoch"),
        "best_loss": ctx.run.get("best_loss"),
        "best_selection_score": ctx.run.get("best_selection_score"),
        "final_loss": last_history_value(ctx.run_json, "loss"),
        "final_kendall_tau_t0": last_history_value(ctx.run_json, "kendall_tau_t0"),
        "final_pred_std": last_history_value(ctx.run_json, "pred_std"),
        "epoch_forward_mean_seconds": mean_history_value(ctx.run_json, "epoch_forward_seconds"),
        "epoch_loss_mean_seconds": mean_history_value(ctx.run_json, "epoch_loss_seconds"),
        "epoch_backward_mean_seconds": mean_history_value(ctx.run_json, "epoch_backward_seconds"),
        "epoch_diagnostic_mean_seconds": mean_history_value(
            ctx.run_json, "epoch_diagnostic_seconds"
        ),
        "epoch_validation_score_mean_seconds": mean_history_value(
            ctx.run_json, "epoch_validation_score_seconds"
        ),
        "single_cell_range_status": _single_cell_status(ctx),
        "final_claim_status": final_claim_summary.get(
            "status", "not_available_until_query_local_utility"
        ),
        "final_success_allowed": bool(final_claim_summary.get("final_success_allowed", False)),
        "final_claim_blocking_gates": final_claim_summary.get("blocking_gates"),
    }


def _gate_fields(ctx: RowContext) -> RowFields:
    workload_stability_gate = _mapping(ctx.run.get("workload_stability_gate"))
    support_overlap_gate = _mapping(ctx.run.get("support_overlap_gate"))
    global_sanity_gate = _mapping(ctx.run.get("global_sanity_gate"))
    target_diffusion_gate = _mapping(ctx.run.get("target_diffusion_gate"))
    query_config = ctx.query_config
    return {
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
    }


def _predictability_fields(ctx: RowContext) -> RowFields:
    predictability_audit = _mapping(ctx.run.get("predictability_audit"))
    predictability_metrics = _mapping(predictability_audit.get("metrics"))
    prior_predictive_alignment_gate = _mapping(
        predictability_audit.get("prior_predictive_alignment_gate")
    )
    per_head_predictability = _mapping(predictability_audit.get("per_head_predictability"))
    query_hit_predictability = _mapping(per_head_predictability.get("query_hit_probability"))
    behavior_predictability = _mapping(per_head_predictability.get("conditional_behavior_utility"))
    replacement_predictability = _mapping(
        per_head_predictability.get("replacement_representative_value")
    )
    segment_budget_predictability = _mapping(per_head_predictability.get("segment_budget_target"))
    prior_channel_predictability = _mapping(
        predictability_audit.get("prior_channel_predictability")
    )
    return {
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
        "prior_channel_query_mass_spearman": _mapping(
            prior_channel_predictability.get("query_mass_prior")
        ).get("spearman"),
        "prior_channel_combined_score_lift_at_5_percent": _mapping(
            prior_channel_predictability.get("combined_prior_score")
        ).get("lift_at_5_percent"),
    }


def _workload_signature_fields(ctx: RowContext) -> RowFields:
    workload_signature_gate = _mapping(
        _mapping(ctx.run.get("workload_distribution_comparison")).get("workload_signature_gate")
    )
    signature_pairs_raw = workload_signature_gate.get("pairs")
    signature_pairs = _mapping(signature_pairs_raw)
    signature_train_metrics = _mapping(_mapping(signature_pairs.get("train")).get("metrics"))
    point_hit_signature_distance = signature_train_metrics.get(
        "point_hit_distribution_ks",
        signature_train_metrics.get("point_hit_distribution_ks_proxy"),
    )
    ship_hit_signature_distance = signature_train_metrics.get(
        "ship_hit_distribution_ks",
        signature_train_metrics.get("ship_hit_distribution_ks_proxy"),
    )
    return {
        "workload_signature_gate_pass": workload_signature_gate.get("all_pass"),
        "workload_signature_gate_available": workload_signature_gate.get("all_available"),
        "workload_signature_pair_count": (
            len(signature_pairs_raw) if isinstance(signature_pairs_raw, dict) else None
        ),
        "workload_signature_failed_pairs": (
            [
                label
                for label, pair in signature_pairs.items()
                if isinstance(pair, dict) and not bool(pair.get("gate_pass", False))
            ]
            if isinstance(signature_pairs_raw, dict)
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
        "train_eval_point_hit_fraction_distribution_ks": signature_train_metrics.get(
            "point_hit_fraction_distribution_ks"
        ),
        "train_eval_ship_hit_fraction_distribution_ks": signature_train_metrics.get(
            "ship_hit_fraction_distribution_ks"
        ),
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
    }


def _protocol_audit_fields(ctx: RowContext) -> RowFields:
    workload_blind_protocol = ctx.workload_blind_protocol
    return {
        **ctx.selector_claim_evidence,
        "workload_blind_candidate": is_workload_blind_model_type(
            str(ctx.model_config.get("model_type", ""))
        ),
        "workload_blind_protocol_enabled": workload_blind_protocol.get("enabled"),
        "primary_masks_frozen_before_eval_query_scoring": workload_blind_protocol.get(
            "primary_masks_frozen_before_eval_query_scoring"
        ),
        "audit_masks_frozen_before_eval_query_scoring": workload_blind_protocol.get(
            "audit_masks_frozen_before_eval_query_scoring"
        ),
        "eval_geometry_blend_allowed": workload_blind_protocol.get("eval_geometry_blend_allowed"),
        "beats_uniform_range_usefulness": _metric_beats(
            ctx.mlqds, ctx.uniform, "range_usefulness_score"
        ),
        "beats_douglas_peucker_range_usefulness": _metric_beats(
            ctx.mlqds, ctx.douglas_peucker, "range_usefulness_score"
        ),
        "beats_temporal_random_fill_range_usefulness": _metric_beats(
            ctx.mlqds,
            ctx.temporal_random_fill,
            "range_usefulness_score",
        ),
        **ctx.audit,
        **dominant_runtime_phase_fields(ctx.timings, ctx.elapsed_seconds),
    }


def _torch_and_path_fields(ctx: RowContext) -> RowFields:
    model_config = ctx.model_config
    child_torch_runtime = ctx.child_torch_runtime
    child_amp = ctx.child_amp
    return {
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
        "run_dir": str(ctx.run_dir),
        "example_run_path": str(ctx.run_json_path) if ctx.run_json_path.exists() else None,
        "stdout_path": str(ctx.stdout_path),
        "command": ctx.command,
    }


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
    ctx = RowContext(
        workload=workload,
        run_label=run_label,
        command=command,
        returncode=returncode,
        elapsed_seconds=elapsed_seconds,
        run_dir=run_dir,
        stdout_path=stdout_path,
        run_json_path=run_json_path,
        timings=timings,
        run_json=run_json,
        data_sources=data_sources,
    )
    row: RowFields = {}
    for fields in (
        _identity_runtime_fields(ctx),
        _gate_fields(ctx),
        _predictability_fields(ctx),
        _workload_signature_fields(ctx),
        _learning_causality_fields(ctx),
        _protocol_audit_fields(ctx),
        _method_metric_fields(ctx),
        _config_training_fields(ctx),
        _torch_and_path_fields(ctx),
    ):
        row.update(fields)
    return row
