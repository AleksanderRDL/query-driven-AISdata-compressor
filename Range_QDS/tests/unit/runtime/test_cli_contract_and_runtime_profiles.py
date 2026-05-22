"""Tests for torch runtime precision controls."""

from __future__ import annotations

from inspect import signature

import pytest
import torch

from benchmarking.profiles import DEFAULT_PROFILE
from benchmarking.runtime_benchmark import (
    _batch_size_sweep_summary,
    _extra_args_include_training_data_source,
    _parse_train_batch_sizes,
    _profile_train_args,
    _runtime_child_args,
)
from config.run_config import (
    RUN_CONFIG_NAMESPACE_ALIASES,
    RunConfig,
    build_run_config,
    build_run_config_from_namespace,
)
from learning.checkpoints import _checkpoint_config_payload
from learning.model_features import SUPPORTED_MODEL_TYPES
from orchestration.learning_scoring_cli import build_parser
from runtime.torch_runtime import (
    amp_runtime_snapshot,
    normalize_amp_mode,
    torch_autocast_context,
)


def test_namespace_run_config_builder_uses_parser_contract_and_aliases() -> None:
    args = build_parser().parse_args(
        [
            "--epochs",
            "2",
            "--range_train_anchor_modes",
            "dense,sparse",
            "--no-validation_global_sanity_penalty",
        ]
    )

    cfg = build_run_config_from_namespace(args)

    assert cfg.model.epochs == 2
    assert cfg.query.range_train_anchor_modes == ["dense", "sparse"]
    assert cfg.model.validation_global_sanity_penalty_enabled is False


def test_cli_parser_covers_run_config_builder_contract() -> None:
    args = build_parser().parse_args([])
    missing = [
        name
        for name in signature(build_run_config).parameters
        if not hasattr(args, RUN_CONFIG_NAMESPACE_ALIASES.get(name, name))
    ]

    assert missing == []


def test_cli_model_type_choices_use_supported_model_registry() -> None:
    parser = build_parser()
    model_action = next(action for action in parser._actions if action.dest == "model_type")

    assert tuple(model_action.choices or ()) == SUPPORTED_MODEL_TYPES


def test_parser_accepts_stratified_budget_loss_diagnostic() -> None:
    args = build_parser().parse_args(
        [
            "--loss_objective",
            "stratified_budget_topk",
            "--mlqds_hybrid_mode",
            "stratified",
        ]
    )

    assert args.loss_objective == "stratified_budget_topk"
    assert args.mlqds_hybrid_mode == "stratified"


def test_parser_accepts_local_swap_gain_cost_target() -> None:
    args = build_parser().parse_args(
        [
            "--range_training_target_mode",
            "local_swap_gain_cost_frequency",
            "--mlqds_hybrid_mode",
            "local_delta_swap",
        ]
    )

    assert args.range_training_target_mode == "local_swap_gain_cost_frequency"
    assert args.mlqds_hybrid_mode == "local_delta_swap"


def test_parser_accepts_global_budget_target_and_selector() -> None:
    args = build_parser().parse_args(
        [
            "--range_training_target_mode",
            "global_budget_retained_frequency",
            "--mlqds_hybrid_mode",
            "global_budget",
        ]
    )

    assert args.range_training_target_mode == "global_budget_retained_frequency"
    assert args.mlqds_hybrid_mode == "global_budget"


def test_parser_accepts_global_fill_selector() -> None:
    args = build_parser().parse_args(
        [
            "--mlqds_hybrid_mode",
            "global_fill",
        ]
    )

    assert args.mlqds_hybrid_mode == "global_fill"


def test_parser_accepts_structural_target_blend() -> None:
    args = build_parser().parse_args(
        [
            "--range_training_target_mode",
            "structural_retained_frequency",
            "--range_structural_target_blend",
            "0.40",
            "--range_structural_target_source_mode",
            "boost",
        ]
    )

    assert args.range_training_target_mode == "structural_retained_frequency"
    assert args.range_structural_target_blend == 0.40
    assert args.range_structural_target_source_mode == "boost"


def test_run_config_rejects_unknown_model_keys() -> None:
    payload = build_run_config().to_dict()
    payload["model"]["f1_diagnostic_every"] = 1

    with pytest.raises(TypeError, match="f1_diagnostic_every"):
        RunConfig.from_dict(payload)


def test_checkpoint_config_payload_filters_stale_section_keys() -> None:
    payload = build_run_config().to_dict()
    payload["legacy_top_level"] = True
    payload["data"]["stale_data_key"] = 1
    payload["query"]["stale_query_key"] = 2
    payload["model"]["f1_diagnostic_every"] = 3
    payload["model"]["residual_label_mode"] = "temporal"
    payload["baselines"]["stale_baseline_key"] = 4

    restored = RunConfig.from_dict(_checkpoint_config_payload(payload))

    assert not hasattr(restored.data, "stale_data_key")
    assert not hasattr(restored.query, "stale_query_key")
    assert not hasattr(restored.model, "f1_diagnostic_every")
    assert not hasattr(restored.model, "residual_label_mode")
    assert not hasattr(restored.baselines, "stale_baseline_key")


def test_checkpoint_config_payload_supplies_missing_sections() -> None:
    restored = RunConfig.from_dict(
        _checkpoint_config_payload({"model": {"model_type": "baseline"}})
    )

    assert restored.data.min_points_per_segment == 4
    assert restored.query.workload == "range"
    assert restored.model.model_type == "baseline"
    assert restored.baselines.final_metrics_mode == "diagnostic"


def test_amp_helpers_default_to_cuda_only_autocast() -> None:
    assert normalize_amp_mode(None) == "off"
    assert normalize_amp_mode(" BF16 ") == "bf16"

    cpu_snapshot = amp_runtime_snapshot("bf16", device="cpu")

    assert cpu_snapshot == {
        "mode": "bf16",
        "enabled": False,
        "device_type": "cpu",
        "dtype": "bfloat16",
    }
    with torch_autocast_context("cpu", "bf16"):
        value = torch.ones((2,), dtype=torch.float32) + 1.0
    assert value.dtype == torch.float32


def test_runtime_child_args_forward_amp_mode() -> None:
    assert _runtime_child_args("high", True, "bf16") == [
        "--float32_matmul_precision",
        "high",
        "--allow_tf32",
        "--amp_mode",
        "bf16",
    ]


def test_parse_train_batch_sizes() -> None:
    assert _parse_train_batch_sizes("16, 32,64") == [16, 32, 64]
    assert _parse_train_batch_sizes(None) is None


def test_runtime_profile_uses_workload_aware_diagnostic_shape(tmp_path) -> None:
    args = _profile_train_args(
        DEFAULT_PROFILE, seed=42, results_dir=tmp_path / "run", checkpoint=tmp_path / "m.pt"
    )

    assert "--n_queries" in args
    assert args[args.index("--n_queries") + 1] == "80"
    assert args[args.index("--max_queries") + 1] == "2048"
    assert args[args.index("--compression_ratio") + 1] == "0.05"
    assert args[args.index("--query_chunk_size") + 1] == "2048"
    assert args[args.index("--train_batch_size") + 1] == "64"
    assert args[args.index("--inference_batch_size") + 1] == "64"
    assert args[args.index("--model_type") + 1] == "range_aware"
    assert args[args.index("--query_coverage") + 1] == "0.20"
    assert args[args.index("--range_spatial_km") + 1] == "2.2"
    assert args[args.index("--range_time_hours") + 1] == "5.0"
    assert args[args.index("--range_footprint_jitter") + 1] == "0.0"
    assert args[args.index("--range_max_coverage_overshoot") + 1] == "0.02"
    assert args[args.index("--range_time_domain_mode") + 1] == "anchor_day"
    assert args[args.index("--range_anchor_mode") + 1] == "mixed_density"
    assert args[args.index("--range_diagnostics_mode") + 1] == "cached"
    assert args[args.index("--final_metrics_mode") + 1] == "diagnostic"
    assert args[args.index("--early_stopping_patience") + 1] == "5"
    assert args[args.index("--validation_score_every") + 1] == "1"
    assert args[args.index("--checkpoint_smoothing_window") + 1] == "1"
    assert args[args.index("--checkpoint_full_score_every") + 1] == "4"
    assert args[args.index("--checkpoint_candidate_pool_size") + 1] == "2"
    assert args[args.index("--loss_objective") + 1] == "budget_topk"
    assert args[args.index("--budget_loss_ratios") + 1] == "0.05,0.10"
    assert args[args.index("--range_audit_compression_ratios") + 1] == (
        "0.01,0.02,0.05,0.10,0.15,0.20,0.30"
    )
    assert args[args.index("--budget_loss_temperature") + 1] == "0.25"
    assert args[args.index("--temporal_distribution_loss_weight") + 1] == "0.000"
    assert args[args.index("--mlqds_temporal_fraction") + 1] == "0.25"
    assert args[args.index("--mlqds_score_mode") + 1] == "rank"
    assert args[args.index("--mlqds_score_temperature") + 1] == "1.00"
    assert args[args.index("--mlqds_rank_confidence_weight") + 1] == "0.15"
    assert args[args.index("--mlqds_range_geometry_blend") + 1] == "0.00"
    assert args[args.index("--mlqds_diversity_bonus") + 1] == "0.00"
    assert args[args.index("--mlqds_hybrid_mode") + 1] == "fill"
    assert args[args.index("--mlqds_stratified_center_weight") + 1] == "0.00"
    assert args[args.index("--temporal_residual_label_mode") + 1] == "none"
    assert args[args.index("--range_label_mode") + 1] == "usefulness"
    assert args[args.index("--range_temporal_target_blend") + 1] == "0.000"
    assert args[args.index("--range_target_budget_weight_power") + 1] == "0.00"
    assert args[args.index("--range_marginal_target_radius_scale") + 1] == "0.50"
    assert args[args.index("--range_query_spine_fraction") + 1] == "0.10"
    assert args[args.index("--range_query_spine_mass_mode") + 1] == "hit_group"
    assert args[args.index("--range_query_residual_multiplier") + 1] == "1.00"
    assert args[args.index("--range_query_residual_mass_mode") + 1] == "query"
    assert args[args.index("--range_set_utility_multiplier") + 1] == "1.00"
    assert args[args.index("--range_set_utility_candidate_limit") + 1] == "128"
    assert args[args.index("--range_set_utility_mass_mode") + 1] == "gain"
    assert args[args.index("--range_boundary_prior_weight") + 1] == "0.0"
    assert "--n_ships" not in args
    assert "--n_points" not in args


def test_runtime_profile_requires_real_training_data_source() -> None:
    assert _extra_args_include_training_data_source("--csv_path ../AISDATA/cleaned/day.csv")
    assert _extra_args_include_training_data_source(
        "--train_csv_path=train.csv --validation_csv_path validation.csv --eval_csv_path eval.csv"
    )
    assert not _extra_args_include_training_data_source("--max_segments 10")
    assert not _extra_args_include_training_data_source("--validation_csv_path validation.csv")


def test_batch_size_sweep_summary_extracts_timing_memory_and_score() -> None:
    rows = _batch_size_sweep_summary(
        [
            {
                "name": "train_bs32",
                "train_batch_size": 32,
                "returncode": 0,
                "elapsed_seconds": 12.5,
                "timings": {"epoch_timings": [{"seconds": 2.0}, {"seconds": 3.0}]},
                "metrics": {
                    "best_selection_score": 0.4,
                    "batch_size": {"train_batch_size": 32},
                    "cuda_memory": {
                        "training": {
                            "max_allocated_mb": 123.0,
                            "max_reserved_mb": 256.0,
                        }
                    },
                    "methods": {"MLQDS": {"aggregate_f1": 0.5}},
                },
            }
        ]
    )

    assert rows == [
        {
            "train_batch_size": 32,
            "returncode": 0,
            "elapsed_seconds": 12.5,
            "epoch_time_mean_seconds": 2.5,
            "epoch_time_min_seconds": 2.0,
            "epoch_time_max_seconds": 3.0,
            "peak_allocated_mb": 123.0,
            "peak_reserved_mb": 256.0,
            "best_selection_score": 0.4,
            "mlqds_aggregate_f1": 0.5,
            "mlqds_range_usefulness_score": None,
            "mlqds_range_ship_coverage": None,
            "mlqds_range_crossing_f1": None,
            "mlqds_range_gap_coverage": None,
            "mlqds_range_gap_time_coverage": None,
            "mlqds_range_gap_distance_coverage": None,
            "mlqds_range_turn_coverage": None,
        }
    ]
