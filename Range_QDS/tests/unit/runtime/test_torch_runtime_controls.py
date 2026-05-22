"""Tests for torch runtime precision controls."""

from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from config.run_config import (
    DEFAULT_BUDGET_LOSS_RATIOS,
    DEFAULT_BUDGET_LOSS_TEMPERATURE,
    DEFAULT_LEARNED_SEGMENT_ALLOCATION_LENGTH_SUPPORT_WEIGHT,
    DEFAULT_LEARNED_SEGMENT_ALLOCATION_WEIGHT_FLOOR,
    DEFAULT_LEARNED_SEGMENT_GEOMETRY_GAIN_WEIGHT,
    DEFAULT_LEARNED_SEGMENT_SCORE_BLEND_WEIGHT,
    DEFAULT_QUERY_LOCAL_UTILITY_BEHAVIOR_RANK_LOSS_WEIGHT,
    DEFAULT_VALIDATION_ENDPOINT_PENALTY_WEIGHT,
    DEFAULT_VALIDATION_GLOBAL_SANITY_PENALTY_WEIGHT,
    DEFAULT_VALIDATION_LENGTH_PRESERVATION_MIN,
    DEFAULT_VALIDATION_SED_PENALTY_WEIGHT,
    RunConfig,
    build_run_config,
    format_run_config_log_line,
)
from orchestration.learning_scoring_cli import build_parser
from orchestration.train_and_score import _split_max_segments
from runtime.torch_runtime import (
    apply_torch_runtime_settings,
)


def test_apply_torch_runtime_settings_sets_precision_and_tf32() -> None:
    old_precision = torch.get_float32_matmul_precision()
    old_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    try:
        snapshot = apply_torch_runtime_settings(float32_matmul_precision="high", allow_tf32=True)

        assert snapshot["float32_matmul_precision"] == "high"
        assert snapshot["tf32_matmul_allowed"] is True
        assert torch.get_float32_matmul_precision() == "high"
        assert bool(torch.backends.cuda.matmul.allow_tf32) is True
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def test_run_config_roundtrips_precision_controls() -> None:
    cfg = build_run_config(
        train_csv_path="train.csv",
        validation_csv_path="validation.csv",
        eval_csv_path="eval.csv",
        train_max_segments=48,
        validation_max_segments=32,
        eval_max_segments=24,
        float32_matmul_precision="high",
        allow_tf32=True,
        embed_dim=96,
        num_heads=8,
        num_layers=2,
        dropout=0.05,
        train_batch_size=64,
        inference_batch_size=32,
        query_chunk_size=512,
        amp_mode="bf16",
        model_type="historical_prior",
        historical_prior_k=7,
        historical_prior_clock_weight=0.25,
        historical_prior_mmsi_weight=2.5,
        historical_prior_density_weight=3.5,
        historical_prior_min_target=0.25,
        historical_prior_support_ratio=0.4,
        range_boundary_prior_weight=1.0,
        range_label_mode="point_f1",
        range_target_balance_mode="trajectory_unit_mass",
        range_replicate_target_aggregation="frequency_mean",
        range_component_target_blend=0.35,
        range_temporal_target_blend=0.20,
        range_target_budget_weight_power=0.75,
        range_marginal_target_radius_scale=0.75,
        range_query_spine_fraction=0.25,
        range_query_spine_mass_mode="query",
        range_query_residual_multiplier=1.50,
        range_query_residual_mass_mode="point",
        range_set_utility_multiplier=1.75,
        range_set_utility_candidate_limit=64,
        range_set_utility_mass_mode="point",
        loss_objective="ranking_bce",
        budget_loss_ratios=[0.02, 0.05],
        budget_loss_temperature=0.2,
        query_local_utility_aux_loss_weight=0.75,
        query_local_utility_segment_budget_head_weight=0.35,
        query_local_utility_segment_level_loss_weight=0.90,
        query_local_utility_behavior_rank_loss_weight=0.40,
        query_local_utility_sparse_head_rank_loss_weight=0.25,
        query_local_utility_sparse_head_bce_target_mode="window_max_normalized",
        temporal_distribution_loss_weight=0.07,
        ranking_pairs_per_type=192,
        ranking_top_quantile=0.9,
        range_spatial_km=2.2,
        range_time_hours=6.0,
        range_max_coverage_overshoot=0.02,
        range_time_domain_mode="anchor_day",
        range_anchor_mode="sparse",
        range_train_anchor_modes=["mixed_density", "sparse"],
        range_train_footprints=["1.1:2.5", "2.2:5"],
        range_train_workload_replicates=3,
        mlqds_hybrid_mode="swap",
        mlqds_stratified_center_weight=0.45,
        mlqds_min_learned_swaps=1,
        mlqds_score_mode="rank_confidence",
        mlqds_score_temperature=0.5,
        mlqds_rank_confidence_weight=0.3,
        mlqds_range_geometry_blend=0.4,
        range_audit_compression_ratios=[0.01, 0.05],
        checkpoint_full_score_every=3,
        checkpoint_candidate_pool_size=2,
        range_diagnostics_mode="cached",
        validation_split_mode="source_stratified",
        train_fraction=0.34,
        val_fraction=0.33,
        final_metrics_mode="core",
    )
    restored = RunConfig.from_dict(cfg.to_dict())

    assert restored.model.float32_matmul_precision == "high"
    assert restored.model.embed_dim == 96
    assert restored.model.num_heads == 8
    assert restored.model.num_layers == 2
    assert restored.model.dropout == 0.05
    assert restored.data.train_csv_path == "train.csv"
    assert restored.data.validation_csv_path == "validation.csv"
    assert restored.data.eval_csv_path == "eval.csv"
    assert restored.data.train_max_segments == 48
    assert restored.data.validation_max_segments == 32
    assert restored.data.eval_max_segments == 24
    assert restored.data.range_diagnostics_mode == "cached"
    assert restored.data.validation_split_mode == "source_stratified"
    assert restored.data.train_fraction == 0.34
    assert restored.data.val_fraction == 0.33
    assert restored.baselines.final_metrics_mode == "core"
    assert restored.model.allow_tf32 is True
    assert restored.model.train_batch_size == 64
    assert restored.model.inference_batch_size == 32
    assert restored.model.query_chunk_size == 512
    assert restored.model.amp_mode == "bf16"
    assert restored.model.model_type == "historical_prior"
    assert restored.model.historical_prior_k == 7
    assert restored.model.historical_prior_clock_weight == 0.25
    assert restored.model.historical_prior_mmsi_weight == 2.5
    assert restored.model.historical_prior_density_weight == 3.5
    assert restored.model.historical_prior_min_target == 0.25
    assert restored.model.historical_prior_support_ratio == 0.4
    assert restored.model.range_boundary_prior_weight == 1.0
    assert restored.model.range_label_mode == "point_f1"
    assert restored.model.range_target_balance_mode == "trajectory_unit_mass"
    assert restored.model.range_replicate_target_aggregation == "frequency_mean"
    assert restored.model.range_component_target_blend == 0.35
    assert restored.model.range_temporal_target_blend == 0.20
    assert restored.model.range_target_budget_weight_power == 0.75
    assert restored.model.range_marginal_target_radius_scale == 0.75
    assert restored.model.range_query_spine_fraction == 0.25
    assert restored.model.range_query_spine_mass_mode == "query"
    assert restored.model.range_query_residual_multiplier == 1.50
    assert restored.model.range_query_residual_mass_mode == "point"
    assert restored.model.range_set_utility_multiplier == 1.75
    assert restored.model.range_set_utility_candidate_limit == 64
    assert restored.model.range_set_utility_mass_mode == "point"
    assert restored.model.loss_objective == "ranking_bce"
    assert restored.model.budget_loss_ratios == [0.02, 0.05]
    assert restored.model.budget_loss_temperature == 0.2
    assert restored.model.query_local_utility_aux_loss_weight == 0.75
    assert restored.model.query_local_utility_segment_budget_head_weight == 0.35
    assert restored.model.query_local_utility_segment_level_loss_weight == 0.90
    assert restored.model.query_local_utility_behavior_rank_loss_weight == 0.40
    assert restored.model.query_local_utility_sparse_head_rank_loss_weight == 0.25
    assert restored.model.query_local_utility_sparse_head_bce_target_mode == "window_max_normalized"
    assert restored.model.temporal_distribution_loss_weight == 0.07
    assert restored.model.ranking_pairs_per_type == 192
    assert restored.model.ranking_top_quantile == 0.9
    assert restored.query.range_spatial_km == 2.2
    assert restored.query.range_time_hours == 6.0
    assert restored.query.range_max_coverage_overshoot == 0.02
    assert restored.query.range_time_domain_mode == "anchor_day"
    assert restored.query.range_anchor_mode == "sparse"
    assert restored.query.range_train_anchor_modes == ["mixed_density", "sparse"]
    assert restored.query.range_train_footprints == ["1.1:2.5", "2.2:5"]
    assert restored.query.range_train_workload_replicates == 3
    assert restored.model.mlqds_hybrid_mode == "swap"
    assert restored.model.mlqds_stratified_center_weight == 0.45
    assert restored.model.mlqds_min_learned_swaps == 1
    assert restored.model.mlqds_score_mode == "rank_confidence"
    assert restored.model.mlqds_score_temperature == 0.5
    assert restored.model.mlqds_rank_confidence_weight == 0.3
    assert restored.model.mlqds_range_geometry_blend == 0.4
    assert restored.model.range_audit_compression_ratios == [0.01, 0.05]
    assert restored.model.checkpoint_selection_metric == "score"
    assert restored.model.checkpoint_full_score_every == 3
    assert restored.model.checkpoint_candidate_pool_size == 2


def test_cli_exposes_training_and_scoring_tuning_controls() -> None:
    args = build_parser().parse_args(
        [
            "--ranking_pairs_per_type",
            "64",
            "--ranking_top_quantile",
            "0.70",
            "--mlqds_score_mode",
            "rank_confidence",
            "--embed_dim",
            "96",
            "--num_heads",
            "8",
            "--num_layers",
            "2",
            "--dropout",
            "0.05",
            "--mlqds_hybrid_mode",
            "swap",
            "--mlqds_stratified_center_weight",
            "0.45",
            "--mlqds_min_learned_swaps",
            "1",
            "--mlqds_score_temperature",
            "0.50",
            "--mlqds_rank_confidence_weight",
            "0.30",
            "--mlqds_range_geometry_blend",
            "0.40",
            "--validation_split_mode",
            "source_stratified",
            "--train_fraction",
            "0.34",
            "--val_fraction",
            "0.33",
            "--range_audit_compression_ratios",
            "0.01,0.02,0.10",
            "--range_label_mode",
            "point_f1",
            "--range_target_balance_mode",
            "trajectory_unit_mass",
            "--range_train_workload_replicates",
            "4",
            "--range_time_domain_mode",
            "anchor_day",
            "--range_anchor_mode",
            "sparse",
            "--range_train_anchor_modes",
            "mixed_density,sparse",
            "--range_train_footprints",
            "1.1:2.5,2.2x5.0",
            "--range_max_coverage_overshoot",
            "0.02",
            "--range_replicate_target_aggregation",
            "frequency_mean",
            "--range_component_target_blend",
            "0.40",
            "--range_temporal_target_blend",
            "0.15",
            "--range_structural_target_blend",
            "0.35",
            "--range_structural_target_source_mode",
            "boost",
            "--range_target_budget_weight_power",
            "0.75",
            "--range_marginal_target_radius_scale",
            "0.65",
            "--range_query_spine_fraction",
            "0.20",
            "--range_query_spine_mass_mode",
            "query",
            "--range_query_residual_multiplier",
            "1.25",
            "--range_query_residual_mass_mode",
            "point",
            "--range_set_utility_multiplier",
            "1.75",
            "--range_set_utility_candidate_limit",
            "64",
            "--range_set_utility_mass_mode",
            "query",
            "--query_prior_grid_bins",
            "128",
            "--query_prior_smoothing_passes",
            "0",
            "--loss_objective",
            "budget_topk",
            "--budget_loss_ratios",
            "0.01,0.05",
            "--budget_loss_temperature",
            "0.20",
            "--query_local_utility_aux_loss_weight",
            "0.75",
            "--query_local_utility_segment_budget_head_weight",
            "0.35",
            "--query_local_utility_segment_level_loss_weight",
            "0.90",
            "--query_local_utility_behavior_rank_loss_weight",
            "0.40",
            "--query_local_utility_sparse_head_rank_loss_weight",
            "0.25",
            "--query_local_utility_sparse_head_bce_target_mode",
            "window_max_normalized",
            "--temporal_distribution_loss_weight",
            "0.20",
            "--checkpoint_full_score_every",
            "3",
            "--checkpoint_candidate_pool_size",
            "2",
            "--range_diagnostics_mode",
            "cached",
            "--final_metrics_mode",
            "core",
            "--model_type",
            "historical_prior",
            "--historical_prior_k",
            "7",
            "--historical_prior_clock_weight",
            "0.25",
            "--historical_prior_mmsi_weight",
            "2.5",
            "--historical_prior_density_weight",
            "3.5",
            "--historical_prior_min_target",
            "0.25",
            "--historical_prior_support_ratio",
            "0.40",
            "--historical_prior_source_aggregation",
            "mean",
            "--train_max_segments",
            "48",
            "--validation_max_segments",
            "32",
            "--eval_max_segments",
            "24",
            "--validation_csv_path",
            "validation.csv",
        ]
    )

    cfg = build_run_config(
        ranking_pairs_per_type=args.ranking_pairs_per_type,
        ranking_top_quantile=args.ranking_top_quantile,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mlqds_hybrid_mode=args.mlqds_hybrid_mode,
        mlqds_stratified_center_weight=args.mlqds_stratified_center_weight,
        mlqds_min_learned_swaps=args.mlqds_min_learned_swaps,
        mlqds_score_mode=args.mlqds_score_mode,
        mlqds_score_temperature=args.mlqds_score_temperature,
        mlqds_rank_confidence_weight=args.mlqds_rank_confidence_weight,
        mlqds_range_geometry_blend=args.mlqds_range_geometry_blend,
        range_audit_compression_ratios=args.range_audit_compression_ratios,
        range_label_mode=args.range_label_mode,
        range_target_balance_mode=args.range_target_balance_mode,
        range_train_workload_replicates=args.range_train_workload_replicates,
        range_time_domain_mode=args.range_time_domain_mode,
        range_anchor_mode=args.range_anchor_mode,
        range_train_anchor_modes=args.range_train_anchor_modes,
        range_train_footprints=args.range_train_footprints,
        range_max_coverage_overshoot=args.range_max_coverage_overshoot,
        range_replicate_target_aggregation=args.range_replicate_target_aggregation,
        range_component_target_blend=args.range_component_target_blend,
        range_temporal_target_blend=args.range_temporal_target_blend,
        range_structural_target_blend=args.range_structural_target_blend,
        range_structural_target_source_mode=args.range_structural_target_source_mode,
        range_target_budget_weight_power=args.range_target_budget_weight_power,
        range_marginal_target_radius_scale=args.range_marginal_target_radius_scale,
        range_query_spine_fraction=args.range_query_spine_fraction,
        range_query_spine_mass_mode=args.range_query_spine_mass_mode,
        range_query_residual_multiplier=args.range_query_residual_multiplier,
        range_query_residual_mass_mode=args.range_query_residual_mass_mode,
        range_set_utility_multiplier=args.range_set_utility_multiplier,
        range_set_utility_candidate_limit=args.range_set_utility_candidate_limit,
        range_set_utility_mass_mode=args.range_set_utility_mass_mode,
        query_prior_grid_bins=args.query_prior_grid_bins,
        query_prior_smoothing_passes=args.query_prior_smoothing_passes,
        loss_objective=args.loss_objective,
        budget_loss_ratios=args.budget_loss_ratios,
        budget_loss_temperature=args.budget_loss_temperature,
        query_local_utility_aux_loss_weight=args.query_local_utility_aux_loss_weight,
        query_local_utility_segment_budget_head_weight=args.query_local_utility_segment_budget_head_weight,
        query_local_utility_segment_level_loss_weight=args.query_local_utility_segment_level_loss_weight,
        query_local_utility_behavior_rank_loss_weight=args.query_local_utility_behavior_rank_loss_weight,
        query_local_utility_sparse_head_rank_loss_weight=args.query_local_utility_sparse_head_rank_loss_weight,
        query_local_utility_sparse_head_bce_target_mode=args.query_local_utility_sparse_head_bce_target_mode,
        temporal_distribution_loss_weight=args.temporal_distribution_loss_weight,
        checkpoint_full_score_every=args.checkpoint_full_score_every,
        checkpoint_candidate_pool_size=args.checkpoint_candidate_pool_size,
        range_diagnostics_mode=args.range_diagnostics_mode,
        final_metrics_mode=args.final_metrics_mode,
        model_type=args.model_type,
        historical_prior_k=args.historical_prior_k,
        historical_prior_clock_weight=args.historical_prior_clock_weight,
        historical_prior_mmsi_weight=args.historical_prior_mmsi_weight,
        historical_prior_density_weight=args.historical_prior_density_weight,
        historical_prior_min_target=args.historical_prior_min_target,
        historical_prior_support_ratio=args.historical_prior_support_ratio,
        historical_prior_source_aggregation=args.historical_prior_source_aggregation,
        train_max_segments=args.train_max_segments,
        validation_max_segments=args.validation_max_segments,
        eval_max_segments=args.eval_max_segments,
        validation_csv_path=args.validation_csv_path,
    )

    assert args.ranking_pairs_per_type == 64
    assert args.ranking_top_quantile == 0.70
    assert args.embed_dim == 96
    assert args.num_heads == 8
    assert args.num_layers == 2
    assert args.dropout == 0.05
    assert args.mlqds_hybrid_mode == "swap"
    assert args.mlqds_stratified_center_weight == 0.45
    assert args.mlqds_min_learned_swaps == 1
    assert args.mlqds_score_mode == "rank_confidence"
    assert args.mlqds_score_temperature == 0.50
    assert args.mlqds_rank_confidence_weight == 0.30
    assert args.mlqds_range_geometry_blend == 0.40
    assert args.validation_split_mode == "source_stratified"
    assert args.train_fraction == 0.34
    assert args.val_fraction == 0.33
    assert args.range_audit_compression_ratios == [0.01, 0.02, 0.10]
    assert args.range_label_mode == "point_f1"
    assert args.range_target_balance_mode == "trajectory_unit_mass"
    assert args.range_train_workload_replicates == 4
    assert args.range_time_domain_mode == "anchor_day"
    assert args.range_anchor_mode == "sparse"
    assert args.range_train_anchor_modes == ["mixed_density", "sparse"]
    assert args.range_train_footprints == ["1.1:2.5", "2.2:5"]
    assert args.range_max_coverage_overshoot == 0.02
    assert args.range_replicate_target_aggregation == "frequency_mean"
    assert args.range_component_target_blend == 0.40
    assert args.range_temporal_target_blend == 0.15
    assert args.range_structural_target_blend == 0.35
    assert args.range_structural_target_source_mode == "boost"
    assert args.range_target_budget_weight_power == 0.75
    assert args.range_marginal_target_radius_scale == 0.65
    assert args.range_query_spine_fraction == 0.20
    assert args.range_query_spine_mass_mode == "query"
    assert args.range_query_residual_multiplier == 1.25
    assert args.range_query_residual_mass_mode == "point"
    assert args.range_set_utility_multiplier == 1.75
    assert args.range_set_utility_candidate_limit == 64
    assert args.range_set_utility_mass_mode == "query"
    assert args.query_prior_grid_bins == 128
    assert args.query_prior_smoothing_passes == 0
    assert args.loss_objective == "budget_topk"
    assert args.budget_loss_ratios == [0.01, 0.05]
    assert args.budget_loss_temperature == 0.20
    assert args.query_local_utility_aux_loss_weight == 0.75
    assert args.query_local_utility_segment_budget_head_weight == 0.35
    assert args.query_local_utility_segment_level_loss_weight == 0.90
    assert args.query_local_utility_behavior_rank_loss_weight == 0.40
    assert args.query_local_utility_sparse_head_rank_loss_weight == 0.25
    assert args.query_local_utility_sparse_head_bce_target_mode == "window_max_normalized"
    assert args.temporal_distribution_loss_weight == 0.20
    assert args.checkpoint_full_score_every == 3
    assert args.checkpoint_candidate_pool_size == 2
    assert args.range_diagnostics_mode == "cached"
    assert args.final_metrics_mode == "core"
    assert args.model_type == "historical_prior"
    assert args.historical_prior_k == 7
    assert args.historical_prior_clock_weight == 0.25
    assert args.historical_prior_mmsi_weight == 2.5
    assert args.historical_prior_density_weight == 3.5
    assert args.historical_prior_min_target == 0.25
    assert args.historical_prior_support_ratio == 0.40
    assert args.historical_prior_source_aggregation == "mean"
    assert args.train_max_segments == 48
    assert args.validation_max_segments == 32
    assert args.eval_max_segments == 24
    assert args.validation_csv_path == "validation.csv"
    assert cfg.model.ranking_pairs_per_type == 64
    assert cfg.model.ranking_top_quantile == 0.70
    assert cfg.model.embed_dim == 96
    assert cfg.model.num_heads == 8
    assert cfg.model.num_layers == 2
    assert cfg.model.dropout == 0.05
    assert cfg.model.mlqds_hybrid_mode == "swap"
    assert cfg.model.mlqds_stratified_center_weight == 0.45
    assert cfg.model.mlqds_min_learned_swaps == 1
    assert cfg.model.mlqds_score_mode == "rank_confidence"
    assert cfg.model.mlqds_score_temperature == 0.50
    assert cfg.model.mlqds_rank_confidence_weight == 0.30
    assert cfg.model.mlqds_range_geometry_blend == 0.40
    assert cfg.model.range_audit_compression_ratios == [0.01, 0.02, 0.10]
    assert cfg.model.range_label_mode == "point_f1"
    assert cfg.model.range_target_balance_mode == "trajectory_unit_mass"
    assert cfg.query.range_train_workload_replicates == 4
    assert cfg.query.range_time_domain_mode == "anchor_day"
    assert cfg.query.range_anchor_mode == "sparse"
    assert cfg.query.range_train_anchor_modes == ["mixed_density", "sparse"]
    assert cfg.query.range_train_footprints == ["1.1:2.5", "2.2:5"]
    assert cfg.query.range_max_coverage_overshoot == 0.02
    assert cfg.model.range_replicate_target_aggregation == "frequency_mean"
    assert cfg.model.range_component_target_blend == 0.40
    assert cfg.model.range_temporal_target_blend == 0.15
    assert cfg.model.range_structural_target_blend == 0.35
    assert cfg.model.range_structural_target_source_mode == "boost"
    assert cfg.model.range_target_budget_weight_power == 0.75
    assert cfg.model.range_marginal_target_radius_scale == 0.65
    assert cfg.model.range_query_spine_fraction == 0.20
    assert cfg.model.range_query_spine_mass_mode == "query"
    assert cfg.model.range_query_residual_multiplier == 1.25
    assert cfg.model.range_query_residual_mass_mode == "point"
    assert cfg.model.range_set_utility_multiplier == 1.75
    assert cfg.model.range_set_utility_candidate_limit == 64
    assert cfg.model.range_set_utility_mass_mode == "query"
    assert cfg.model.query_prior_grid_bins == 128
    assert cfg.model.query_prior_smoothing_passes == 0
    assert cfg.model.loss_objective == "budget_topk"
    assert cfg.model.budget_loss_ratios == [0.01, 0.05]
    assert cfg.model.budget_loss_temperature == 0.20
    assert cfg.model.query_local_utility_aux_loss_weight == 0.75
    assert cfg.model.query_local_utility_segment_budget_head_weight == 0.35
    assert cfg.model.query_local_utility_segment_level_loss_weight == 0.90
    assert cfg.model.query_local_utility_behavior_rank_loss_weight == 0.40
    assert cfg.model.query_local_utility_sparse_head_rank_loss_weight == 0.25
    assert cfg.model.query_local_utility_sparse_head_bce_target_mode == "window_max_normalized"
    assert cfg.model.temporal_distribution_loss_weight == 0.20
    assert cfg.model.checkpoint_full_score_every == 3
    assert cfg.model.checkpoint_candidate_pool_size == 2
    assert cfg.data.range_diagnostics_mode == "cached"
    assert cfg.baselines.final_metrics_mode == "core"
    assert cfg.model.model_type == "historical_prior"
    assert cfg.model.historical_prior_k == 7
    assert cfg.model.historical_prior_clock_weight == 0.25
    assert cfg.model.historical_prior_mmsi_weight == 2.5
    assert cfg.model.historical_prior_density_weight == 3.5
    assert cfg.model.historical_prior_min_target == 0.25
    assert cfg.model.historical_prior_support_ratio == 0.40
    assert cfg.model.historical_prior_source_aggregation == "mean"
    assert cfg.data.train_max_segments == 48
    assert cfg.data.validation_max_segments == 32
    assert cfg.data.eval_max_segments == 24
    assert cfg.data.validation_csv_path == "validation.csv"


def test_run_config_loads_missing_runtime_and_mlqds_defaults() -> None:
    payload = build_run_config().to_dict()
    payload["model"].pop("float32_matmul_precision")
    payload["model"].pop("allow_tf32")
    payload["model"].pop("inference_batch_size")
    payload["model"].pop("amp_mode")
    payload["model"].pop("historical_prior_clock_weight")
    payload["model"].pop("historical_prior_mmsi_weight")
    payload["model"].pop("historical_prior_density_weight")
    payload["model"].pop("historical_prior_min_target")
    payload["model"].pop("historical_prior_support_ratio")
    payload["model"].pop("historical_prior_source_aggregation")
    payload["model"].pop("range_boundary_prior_weight")
    payload["model"].pop("range_label_mode")
    payload["model"].pop("range_target_balance_mode")
    payload["model"].pop("range_replicate_target_aggregation")
    payload["model"].pop("range_component_target_blend")
    payload["model"].pop("range_temporal_target_blend")
    payload["model"].pop("range_structural_target_blend")
    payload["model"].pop("range_structural_target_source_mode")
    payload["model"].pop("range_target_budget_weight_power")
    payload["model"].pop("range_marginal_target_radius_scale")
    payload["model"].pop("range_query_spine_fraction")
    payload["model"].pop("range_query_spine_mass_mode")
    payload["model"].pop("range_query_residual_multiplier")
    payload["model"].pop("range_query_residual_mass_mode")
    payload["model"].pop("range_set_utility_multiplier")
    payload["model"].pop("range_set_utility_candidate_limit")
    payload["model"].pop("range_set_utility_mass_mode")
    payload["model"].pop("loss_objective")
    payload["model"].pop("budget_loss_ratios")
    payload["model"].pop("budget_loss_temperature")
    payload["model"].pop("query_local_utility_behavior_rank_loss_weight")
    payload["model"].pop("query_local_utility_sparse_head_rank_loss_weight")
    payload["model"].pop("query_local_utility_sparse_head_bce_target_mode")
    payload["model"].pop("temporal_distribution_loss_weight")
    payload["model"].pop("range_audit_compression_ratios")
    payload["model"].pop("mlqds_score_mode")
    payload["model"].pop("mlqds_score_temperature")
    payload["model"].pop("mlqds_rank_confidence_weight")
    payload["model"].pop("mlqds_stratified_center_weight")
    payload["model"].pop("mlqds_min_learned_swaps")
    payload["model"].pop("checkpoint_full_score_every")
    payload["model"].pop("checkpoint_candidate_pool_size")
    payload["model"].pop("query_prior_grid_bins")
    payload["model"].pop("query_prior_smoothing_passes")
    payload["query"].pop("range_train_workload_replicates")
    payload["query"].pop("range_time_domain_mode")
    payload["query"].pop("range_anchor_mode")
    payload["query"].pop("range_train_anchor_modes")
    payload["query"].pop("range_train_footprints")
    payload["query"].pop("range_max_coverage_overshoot")
    payload["data"].pop("train_max_segments")
    payload["data"].pop("validation_max_segments")
    payload["data"].pop("eval_max_segments")
    payload["data"].pop("range_diagnostics_mode")
    payload["baselines"].pop("final_metrics_mode")

    restored = RunConfig.from_dict(payload)

    assert restored.model.float32_matmul_precision == "highest"
    assert restored.model.allow_tf32 is False
    assert restored.model.inference_batch_size == 16
    assert restored.model.amp_mode == "off"
    assert restored.model.historical_prior_clock_weight == 0.0
    assert restored.model.historical_prior_mmsi_weight == 1.0
    assert restored.model.historical_prior_density_weight == 1.0
    assert restored.model.historical_prior_min_target == 0.0
    assert restored.model.historical_prior_support_ratio == 1.0
    assert restored.model.historical_prior_source_aggregation == "none"
    assert restored.model.range_boundary_prior_weight == 0.0
    assert restored.model.range_label_mode == "usefulness"
    assert restored.model.range_target_balance_mode == "none"
    assert restored.model.range_replicate_target_aggregation == "label_mean"
    assert restored.model.range_component_target_blend == 1.0
    assert restored.model.range_temporal_target_blend == 0.0
    assert restored.model.range_structural_target_blend == 0.25
    assert restored.model.range_structural_target_source_mode == "blend"
    assert restored.model.range_target_budget_weight_power == 0.0
    assert restored.model.range_marginal_target_radius_scale == 0.50
    assert restored.model.range_query_spine_fraction == 0.10
    assert restored.model.range_query_spine_mass_mode == "hit_group"
    assert restored.model.range_query_residual_multiplier == 1.0
    assert restored.model.range_query_residual_mass_mode == "query"
    assert restored.model.range_set_utility_multiplier == 1.0
    assert restored.model.range_set_utility_candidate_limit == 128
    assert restored.model.range_set_utility_mass_mode == "gain"
    assert restored.model.loss_objective == "budget_topk"
    assert restored.model.budget_loss_ratios == DEFAULT_BUDGET_LOSS_RATIOS
    assert restored.model.budget_loss_temperature == DEFAULT_BUDGET_LOSS_TEMPERATURE
    assert restored.model.temporal_distribution_loss_weight == 0.0
    assert restored.model.range_audit_compression_ratios == []
    assert restored.model.mlqds_score_mode == "rank"
    assert restored.model.mlqds_score_temperature == 1.0
    assert restored.model.mlqds_rank_confidence_weight == 0.15
    assert restored.model.mlqds_stratified_center_weight == 0.0
    assert restored.model.mlqds_min_learned_swaps == 0
    assert restored.model.mlqds_range_geometry_blend == 0.0
    assert restored.model.checkpoint_selection_metric == "score"
    assert restored.model.checkpoint_score_variant == "range_usefulness"
    assert restored.model.checkpoint_full_score_every == 1
    assert restored.model.checkpoint_candidate_pool_size == 1
    assert restored.model.query_prior_grid_bins == 64
    assert restored.model.query_prior_smoothing_passes == 2
    assert restored.query.range_train_workload_replicates == 1
    assert restored.query.range_time_domain_mode == "dataset"
    assert restored.query.range_anchor_mode == "mixed_density"
    assert restored.query.range_train_anchor_modes == []
    assert restored.query.range_train_footprints == []
    assert restored.query.range_max_coverage_overshoot is None
    assert restored.data.train_max_segments is None
    assert restored.data.validation_max_segments is None
    assert restored.data.eval_max_segments is None
    assert restored.data.range_diagnostics_mode == "full"
    assert restored.baselines.final_metrics_mode == "diagnostic"


def test_run_config_log_line_uses_effective_config_and_runtime_settings() -> None:
    cfg = build_run_config(
        model_type="workload_blind_range",
        query_coverage=0.25,
        float32_matmul_precision="high",
        allow_tf32=True,
        range_training_target_mode="query_local_utility_factorized",
    )

    line = format_run_config_log_line(
        cfg,
        {
            "float32_matmul_precision": "highest",
            "tf32_matmul_allowed": False,
        },
    )

    assert line.startswith("[config] ")
    assert "model=workload_blind_range" in line
    assert "query_coverage=0.25" in line
    assert "range_training_target_mode=query_local_utility_factorized" in line
    assert "float32_matmul_precision=highest" in line
    assert "allow_tf32=False" in line


def test_split_max_segments_falls_back_to_global_cap() -> None:
    args = Namespace(
        max_segments=120,
        train_max_segments=240,
        validation_max_segments=None,
        eval_max_segments=80,
    )

    assert _split_max_segments(args, "train") == 240
    assert _split_max_segments(args, "validation") == 120
    assert _split_max_segments(args, "eval") == 80


def test_validation_score_config_uses_current_names() -> None:
    payload = build_run_config(
        validation_score_every=2,
        checkpoint_full_score_every=4,
        checkpoint_score_variant="answer",
        learned_segment_length_repair_fraction=0.25,
        learned_segment_length_repair_score_protection_fraction=0.15,
        learned_segment_allocation_length_support_weight=0.5,
        learned_segment_allocation_weight_floor=0.35,
        learned_segment_length_support_blend_weight=0.75,
        query_prior_grid_bins=128,
        query_prior_smoothing_passes=0,
        temporal_residual_label_mode="none",
    ).to_dict()
    restored = RunConfig.from_dict(payload)
    args = build_parser().parse_args(
        [
            "--validation_score_every",
            "3",
            "--checkpoint_full_score_every",
            "5",
            "--checkpoint_score_variant",
            "combined",
            "--learned_segment_length_repair_fraction",
            "0.5",
            "--learned_segment_length_repair_score_protection_fraction",
            "0.1",
            "--learned_segment_allocation_length_support_weight",
            "0.25",
            "--learned_segment_allocation_weight_floor",
            "0.2",
            "--learned_segment_length_support_blend_weight",
            "1.0",
            "--query_prior_grid_bins",
            "96",
            "--query_prior_smoothing_passes",
            "1",
            "--temporal_residual_label_mode",
            "none",
        ]
    )

    assert restored.model.validation_score_every == 2
    assert restored.model.checkpoint_full_score_every == 4
    assert restored.model.checkpoint_score_variant == "answer"
    assert restored.model.learned_segment_length_repair_fraction == pytest.approx(0.25)
    assert restored.model.learned_segment_length_repair_score_protection_fraction == pytest.approx(
        0.15
    )
    assert restored.model.learned_segment_allocation_length_support_weight == pytest.approx(0.5)
    assert restored.model.learned_segment_allocation_weight_floor == pytest.approx(0.35)
    assert restored.model.learned_segment_length_support_blend_weight == pytest.approx(0.75)
    assert restored.model.query_prior_grid_bins == 128
    assert restored.model.query_prior_smoothing_passes == 0
    assert restored.model.query_local_utility_behavior_rank_loss_weight == pytest.approx(
        DEFAULT_QUERY_LOCAL_UTILITY_BEHAVIOR_RANK_LOSS_WEIGHT
    )
    assert restored.model.query_local_utility_sparse_head_rank_loss_weight == 0.0
    assert restored.model.query_local_utility_sparse_head_bce_target_mode == "raw"
    assert restored.model.temporal_residual_label_mode == "none"
    assert not hasattr(restored.model, "f1_diagnostic_every")
    assert not hasattr(restored.model, "residual_label_mode")
    assert args.validation_score_every == 3
    assert args.checkpoint_full_score_every == 5
    assert args.checkpoint_score_variant == "combined"
    assert args.learned_segment_length_repair_fraction == pytest.approx(0.5)
    assert args.learned_segment_length_repair_score_protection_fraction == pytest.approx(0.1)
    assert args.learned_segment_allocation_length_support_weight == pytest.approx(0.25)
    assert args.learned_segment_allocation_weight_floor == pytest.approx(0.2)
    assert args.learned_segment_length_support_blend_weight == pytest.approx(1.0)
    assert args.query_prior_grid_bins == 96
    assert args.query_prior_smoothing_passes == 1
    assert args.temporal_residual_label_mode == "none"


def test_direct_config_and_cli_default_to_non_residual_training() -> None:
    cfg = build_run_config()
    args = build_parser().parse_args([])

    assert cfg.model.temporal_residual_label_mode == "none"
    assert cfg.model.query_local_utility_behavior_rank_loss_weight == pytest.approx(
        DEFAULT_QUERY_LOCAL_UTILITY_BEHAVIOR_RANK_LOSS_WEIGHT
    )
    assert cfg.model.query_local_utility_sparse_head_rank_loss_weight == 0.0
    assert cfg.model.query_local_utility_sparse_head_bce_target_mode == "raw"
    assert cfg.model.learned_segment_geometry_gain_weight == pytest.approx(
        DEFAULT_LEARNED_SEGMENT_GEOMETRY_GAIN_WEIGHT
    )
    assert cfg.model.learned_segment_allocation_length_support_weight == pytest.approx(
        DEFAULT_LEARNED_SEGMENT_ALLOCATION_LENGTH_SUPPORT_WEIGHT
    )
    assert cfg.model.learned_segment_allocation_weight_floor == pytest.approx(
        DEFAULT_LEARNED_SEGMENT_ALLOCATION_WEIGHT_FLOOR
    )
    assert cfg.model.learned_segment_score_blend_weight == pytest.approx(
        DEFAULT_LEARNED_SEGMENT_SCORE_BLEND_WEIGHT
    )
    assert cfg.model.learned_segment_length_repair_score_protection_fraction == 0.0
    assert cfg.model.validation_global_sanity_penalty_weight == pytest.approx(
        DEFAULT_VALIDATION_GLOBAL_SANITY_PENALTY_WEIGHT
    )
    assert cfg.model.validation_sed_penalty_weight == pytest.approx(
        DEFAULT_VALIDATION_SED_PENALTY_WEIGHT
    )
    assert cfg.model.validation_endpoint_penalty_weight == pytest.approx(
        DEFAULT_VALIDATION_ENDPOINT_PENALTY_WEIGHT
    )
    assert cfg.model.validation_length_preservation_min == pytest.approx(
        DEFAULT_VALIDATION_LENGTH_PRESERVATION_MIN
    )
    assert args.temporal_residual_label_mode == "none"
    assert args.query_local_utility_behavior_rank_loss_weight == pytest.approx(
        DEFAULT_QUERY_LOCAL_UTILITY_BEHAVIOR_RANK_LOSS_WEIGHT
    )
    assert args.query_local_utility_sparse_head_rank_loss_weight == 0.0
    assert args.query_local_utility_sparse_head_bce_target_mode == "raw"
    assert args.learned_segment_geometry_gain_weight == pytest.approx(
        DEFAULT_LEARNED_SEGMENT_GEOMETRY_GAIN_WEIGHT
    )
    assert args.learned_segment_allocation_length_support_weight == pytest.approx(
        DEFAULT_LEARNED_SEGMENT_ALLOCATION_LENGTH_SUPPORT_WEIGHT
    )
    assert args.learned_segment_allocation_weight_floor == pytest.approx(
        DEFAULT_LEARNED_SEGMENT_ALLOCATION_WEIGHT_FLOOR
    )
    assert args.learned_segment_score_blend_weight == pytest.approx(
        DEFAULT_LEARNED_SEGMENT_SCORE_BLEND_WEIGHT
    )
    assert args.learned_segment_length_repair_score_protection_fraction == 0.0
    assert args.validation_global_sanity_penalty_weight == pytest.approx(
        DEFAULT_VALIDATION_GLOBAL_SANITY_PENALTY_WEIGHT
    )
    assert args.validation_sed_penalty_weight == pytest.approx(
        DEFAULT_VALIDATION_SED_PENALTY_WEIGHT
    )
    assert args.validation_endpoint_penalty_weight == pytest.approx(
        DEFAULT_VALIDATION_ENDPOINT_PENALTY_WEIGHT
    )
    assert args.validation_length_preservation_min == pytest.approx(
        DEFAULT_VALIDATION_LENGTH_PRESERVATION_MIN
    )
