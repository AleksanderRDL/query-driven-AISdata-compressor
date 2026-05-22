"""Tests for range benchmark run helpers."""

from __future__ import annotations

import argparse

import pytest

from benchmarking.final_grid import query_driven_final_grid_summary
from benchmarking.profiles import (
    BLIND_EXPECTED_QUERY_LOCAL_UTILITY_PROFILE,
    BLIND_RETAINED_FREQUENCY_PROFILE,
    BLIND_TEACHER_DISTILL_PROFILE,
    DEFAULT_PROFILE,
    RANGE_BLIND_COVERAGE_MIN_QUERY_FLOOR,
    benchmark_profile_args,
    benchmark_profile_settings,
)
from benchmarking.reporting.audit_extractors import _query_floor_fields
from benchmarking.reporting.metrics import MIN_MATCHED_LEARNED_SLOT_FRACTION_FOR_BLIND_CLAIM
from benchmarking.reporting.row_fields import _row_from_run
from benchmarking.runner import (
    DEFAULT_WORKLOADS,
    PURE_WORKLOADS,
    BenchmarkDataSources,
    _parse_name_list,
    _parse_workload_profile_ids,
    _profile_args,
    _run_config,
    _runner_environment_metadata,
    _workload_profile_label_suffix,
)
from config.run_config import build_run_config
from orchestration.workload_stage import resolve_workload_maps, validation_query_count


def _profile_core_args() -> list[str]:
    """Expected workload-aware diagnostic child args without data-source/cap flags."""
    return [
        "--n_queries",
        "80",
        "--query_coverage",
        "0.20",
        "--range_spatial_fraction",
        "0.0165",
        "--range_time_fraction",
        "0.033",
        "--range_spatial_km",
        "2.2",
        "--range_time_hours",
        "5.0",
        "--range_footprint_jitter",
        "0.0",
        "--range_max_coverage_overshoot",
        "0.02",
        "--range_time_domain_mode",
        "anchor_day",
        "--range_anchor_mode",
        "mixed_density",
        "--range_diagnostics_mode",
        "cached",
        "--final_metrics_mode",
        "diagnostic",
        "--float32_matmul_precision",
        "high",
        "--allow_tf32",
        "--amp_mode",
        "bf16",
        "--query_chunk_size",
        "2048",
        "--train_batch_size",
        "64",
        "--inference_batch_size",
        "64",
        "--model_type",
        "range_aware",
        "--max_queries",
        "2048",
        "--compression_ratio",
        "0.05",
        "--epochs",
        "8",
        "--early_stopping_patience",
        "5",
        "--checkpoint_smoothing_window",
        "1",
        "--checkpoint_full_score_every",
        "4",
        "--checkpoint_candidate_pool_size",
        "2",
        "--loss_objective",
        "budget_topk",
        "--budget_loss_ratios",
        "0.05,0.10",
        "--range_audit_compression_ratios",
        "0.01,0.02,0.05,0.10,0.15,0.20,0.30",
        "--budget_loss_temperature",
        "0.25",
        "--temporal_distribution_loss_weight",
        "0.000",
        "--mlqds_temporal_fraction",
        "0.25",
        "--mlqds_score_mode",
        "rank",
        "--mlqds_score_temperature",
        "1.00",
        "--mlqds_rank_confidence_weight",
        "0.15",
        "--mlqds_range_geometry_blend",
        "0.00",
        "--mlqds_diversity_bonus",
        "0.00",
        "--mlqds_hybrid_mode",
        "fill",
        "--selector_type",
        "temporal_hybrid",
        "--mlqds_stratified_center_weight",
        "0.00",
        "--temporal_residual_label_mode",
        "none",
        "--range_label_mode",
        "point_f1",
        "--range_training_target_mode",
        "point_value",
        "--range_temporal_target_blend",
        "0.000",
        "--range_target_budget_weight_power",
        "0.00",
        "--range_marginal_target_radius_scale",
        "0.50",
        "--range_query_spine_fraction",
        "0.10",
        "--range_query_spine_mass_mode",
        "hit_group",
        "--range_query_residual_multiplier",
        "1.00",
        "--range_query_residual_mass_mode",
        "query",
        "--range_set_utility_multiplier",
        "1.00",
        "--range_set_utility_candidate_limit",
        "128",
        "--range_set_utility_mass_mode",
        "gain",
        "--range_boundary_prior_weight",
        "0.0",
        "--range_teacher_distillation_mode",
        "none",
        "--range_teacher_epochs",
        "4",
        "--checkpoint_selection_metric",
        "uniform_gap",
        "--checkpoint_score_variant",
        "query_local_utility",
    ]


def test_benchmark_name_list_rejects_mixed_workloads() -> None:
    assert _parse_name_list("range", allowed=PURE_WORKLOADS, arg_name="--workloads") == ["range"]
    assert _parse_name_list(None, allowed=DEFAULT_WORKLOADS, arg_name="--workloads") == ["range"]

    with pytest.raises(ValueError, match="unknown"):
        _parse_name_list("range,legacy", allowed=PURE_WORKLOADS, arg_name="--workloads")


def test_benchmark_workload_profile_parser_accepts_known_profiles() -> None:
    assert _parse_workload_profile_ids("range_query_mix_focused,range_query_mix") == [
        "range_query_mix_focused",
        "range_query_mix",
    ]
    assert _workload_profile_label_suffix("range_query_mix_focused") == ("range_query_mix_focused")

    with pytest.raises(ValueError, match="Unknown workload_profile_id"):
        _parse_workload_profile_ids("not_a_profile")
    with pytest.raises(ValueError, match="duplicate"):
        _parse_workload_profile_ids("range_query_mix,range_query_mix")


def test_run_workload_resolution_is_pure_only() -> None:
    assert resolve_workload_maps("range") == ({"range": 1.0}, {"range": 1.0})

    with pytest.raises(ValueError, match="no longer supported"):
        resolve_workload_maps("legacy")


def test_profile_args_own_runtime_and_checkpoint_defaults() -> None:
    args = _profile_core_args()

    assert args[args.index("--float32_matmul_precision") + 1] == "high"
    assert "--allow_tf32" in args
    assert args[args.index("--amp_mode") + 1] == "bf16"


def test_benchmark_environment_metadata_is_scoped_to_parent_process() -> None:
    environment = _runner_environment_metadata()

    assert environment["scope"] == "runner_parent_process"
    assert "rows[*].child_torch_runtime" in environment["note"]


def test_run_config_records_profile_checkpoint_selection_metric(tmp_path) -> None:
    args = argparse.Namespace(
        profile=DEFAULT_PROFILE,
        seed=42,
        cache_dir=None,
        refresh_cache=False,
        no_cache_warmup=False,
        min_points_per_segment=4,
        max_points_per_segment=None,
        max_time_gap_seconds=3600.0,
        max_segments=None,
        max_trajectories=None,
        validation_score_every=1,
        extra_args=None,
        workload_profile_ids="range_query_mix_focused,range_query_mix",
        continue_on_failure=False,
    )
    data_sources = BenchmarkDataSources(
        train_csv_path="train.csv",
        validation_csv_path="validation.csv",
        eval_csv_path="eval.csv",
        selected_cleaned_csv_files=("train.csv", "validation.csv", "eval.csv"),
    )

    payload = _run_config(
        args=args,
        run_id="run",
        workloads=["range"],
        run_label="label",
        data_sources=data_sources,
        results_dir=tmp_path,
        extra_args=[],
    )

    assert payload["checkpoint_selection_metric"] == "uniform_gap"
    assert payload["workload_profile_ids"] == [
        "range_query_mix_focused",
        "range_query_mix",
    ]
    assert payload["profile_settings"]["profile_role"] == "workload_aware_diagnostic"
    assert payload["profile_settings"]["final_product_claim"] is False
    assert payload["profile_settings"]["workload_blind"] is False
    assert payload["profile_settings"]["mlqds_effective_diversity_bonus"] == 0.0
    assert payload["profile_settings"]["range_workload_profile_sweep_ids"] == [
        "range_query_mix_focused",
        "range_query_mix_local",
        "range_query_mix_operational",
        "range_query_mix",
    ]
    assert payload["profile_settings"]["range_compression_sweep_ratios"] == [
        0.01,
        0.02,
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
    ]


def test_blind_profiles_use_small_query_floor_for_coverage_control() -> None:
    """Blind profiles should not force dense duplicate workloads after target coverage is met."""
    for profile in (
        BLIND_EXPECTED_QUERY_LOCAL_UTILITY_PROFILE,
        BLIND_RETAINED_FREQUENCY_PROFILE,
        BLIND_TEACHER_DISTILL_PROFILE,
    ):
        args = benchmark_profile_args(profile)
        settings = benchmark_profile_settings(profile)

        assert args[args.index("--n_queries") + 1] == str(RANGE_BLIND_COVERAGE_MIN_QUERY_FLOOR)
        assert settings["n_queries"] == RANGE_BLIND_COVERAGE_MIN_QUERY_FLOOR
        assert settings["workload_blind"] is True
        assert settings["profile_diagnostic_only"] is True
        assert settings["final_success_allowed"] is False
        assert settings["final_product_candidate"] is False
        assert settings["final_product_claim"] is False
        assert "QueryLocalUtility" in str(settings["final_product_claim_gate"])


def _final_grid_row(workload_profile_id: str, *, mlqds_delta: float = 0.05) -> dict[str, object]:
    row: dict[str, object] = {
        "workload": "range",
        "run_label": workload_profile_id,
        "returncode": 0,
        "workload_profile_id": workload_profile_id,
        "compression_ratio": 0.05,
        "mlqds_query_local_utility_score": 0.55,
        "uniform_query_local_utility_score": 0.50,
        "douglas_peucker_query_local_utility_score": 0.49,
        "workload_stability_gate_pass": True,
        "predictability_gate_pass": True,
        "prior_predictive_alignment_gate_pass": True,
        "target_diffusion_gate_pass": True,
        "workload_signature_gate_pass": True,
        "learning_causality_gate_pass": True,
        "prior_sample_gate_pass": True,
        "global_sanity_gate_pass": True,
        "support_overlap_gate_pass": True,
        "mlqds_primary_metric": "query_local_utility",
    }
    for ratio in (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        prefix = f"audit_ratio_{ratio:.4f}".replace(".", "p")
        row[f"{prefix}_mlqds_query_local_utility"] = 0.50 + mlqds_delta
        row[f"{prefix}_uniform_query_local_utility"] = 0.50
        row[f"{prefix}_douglas_peucker_query_local_utility"] = 0.49
    return row


def test_query_driven_final_grid_summary_accepts_complete_passing_grid() -> None:
    profile_ids = (
        "range_query_mix_focused",
        "range_query_mix_local",
        "range_query_mix_operational",
        "range_query_mix",
    )
    rows = [_final_grid_row(profile_id) for profile_id in profile_ids]
    run_config = {
        "profile_settings": {
            "final_product_candidate": True,
            "range_workload_profile_sweep_ids": list(profile_ids),
            "range_compression_sweep_ratios": [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
        }
    }

    summary = query_driven_final_grid_summary(rows, run_config)

    assert summary["status"] == "final_grid_pass"
    assert summary["final_success_allowed"] is True
    assert summary["grid_complete"] is True
    assert summary["observed_cell_count"] == 28
    assert summary["beats_uniform_queryuseful_cells"] == 28
    assert summary["beats_douglas_peucker_queryuseful_cells"] == 28
    assert summary["low_budget_beats_uniform_queryuseful_cells"] == 12
    assert summary["matched_5_percent_compression_cells_uniform"] == 4
    assert summary["failed_checks"] == []


def test_query_driven_final_grid_summary_blocks_missing_or_failed_evidence() -> None:
    rows = [
        _final_grid_row("range_query_mix_focused"),
        _final_grid_row("range_query_mix_local", mlqds_delta=-0.02),
    ]
    rows[0]["predictability_gate_pass"] = False

    summary = query_driven_final_grid_summary(
        rows,
        {"profile_settings": {"final_product_candidate": True}},
    )

    assert summary["status"] == "final_grid_blocked"
    assert summary["final_success_allowed"] is False
    assert "workload_profile_grid_incomplete" in summary["failed_checks"]
    assert "compression_grid_incomplete" in summary["failed_checks"]
    assert "too_few_uniform_queryuseful_wins" in summary["failed_checks"]
    assert "required_single_run_gates_failed" in summary["failed_checks"]
    assert summary["child_gate_failures"][0]["failed_gates"] == ["predictability_gate_pass"]


def test_query_driven_final_grid_summary_blocks_prior_alignment_failure() -> None:
    rows = [
        _final_grid_row(profile_id)
        for profile_id in (
            "range_query_mix_focused",
            "range_query_mix_local",
            "range_query_mix_operational",
            "range_query_mix",
        )
    ]
    rows[0]["prior_predictive_alignment_gate_pass"] = False

    summary = query_driven_final_grid_summary(
        rows,
        {"profile_settings": {"final_product_candidate": True}},
    )

    assert summary["status"] == "final_grid_blocked"
    assert summary["final_success_allowed"] is False
    assert "required_single_run_gates_failed" in summary["failed_checks"]
    assert "prior_predictive_alignment_gate_pass" in summary["required_single_run_gate_names"]
    assert summary["child_gate_failures"][0]["failed_gates"] == [
        "prior_predictive_alignment_gate_pass"
    ]


def test_query_floor_fields_flags_coverage_target_miss() -> None:
    fields = _query_floor_fields(
        "eval",
        {
            "mode": "target_coverage",
            "target_coverage": 0.30,
            "final_coverage": 0.251,
            "minimum_queries": 8,
            "max_queries": 128,
            "final_query_count": 128,
            "target_reached_query_count": None,
            "coverage_at_target_reached": None,
            "extra_queries_after_target_reached": None,
            "stop_reason": "max_queries_reached",
        },
    )

    assert fields["eval_query_target_reached"] is False
    assert fields["eval_query_target_shortfall"] == pytest.approx(0.049)
    assert fields["eval_query_target_overshoot"] == 0.0
    assert fields["eval_query_target_missed_by_max_queries"] is True


def test_benchmark_row_reports_zero_effective_diversity_for_stratified(tmp_path) -> None:
    """Configured diversity should not imply an effect in stratified mode."""
    run_json = {
        "config": {
            "query": {},
            "model": {
                "model_type": "range_prior",
                "mlqds_diversity_bonus": 0.05,
                "mlqds_hybrid_mode": "stratified",
                "compression_ratio": 0.05,
            },
        },
        "workload_blind_protocol": {
            "enabled": True,
            "primary_masks_frozen_before_eval_query_scoring": True,
            "audit_masks_frozen_before_eval_query_scoring": True,
        },
        "matched": {
            "MLQDS": {"query_local_utility_score": 0.2},
            "uniform": {"query_local_utility_score": 0.1},
            "DouglasPeucker": {"query_local_utility_score": 0.1},
        },
    }

    row = _row_from_run(
        workload="range",
        run_label="run",
        command=[],
        returncode=0,
        elapsed_seconds=1.0,
        run_dir=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        run_json_path=tmp_path / "example_run.json",
        timings={"phase_timings": [], "epoch_timings": [], "inference_step_timings": []},
        run_json=run_json,
    )

    assert row["mlqds_diversity_bonus"] == 0.05
    assert row["mlqds_effective_diversity_bonus"] == 0.0


def test_benchmark_row_marks_blind_protocol_fail_even_with_good_score(tmp_path) -> None:
    run_json = {
        "config": {"model": {"model_type": "range_prior"}},
        "workload_blind_protocol": {
            "enabled": False,
            "primary_masks_frozen_before_eval_query_scoring": False,
            "audit_masks_frozen_before_eval_query_scoring": False,
        },
        "matched": {
            "MLQDS": {"query_local_utility_score": 0.50, "range_point_f1": 0.40},
            "uniform": {"query_local_utility_score": 0.40, "range_point_f1": 0.30},
            "DouglasPeucker": {"query_local_utility_score": 0.30, "range_point_f1": 0.20},
        },
    }

    row = _row_from_run(
        workload="range",
        run_label="leaky_blind",
        command=[],
        returncode=0,
        elapsed_seconds=1.0,
        run_dir=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        run_json_path=tmp_path / "example_run.json",
        timings={"phase_timings": [], "epoch_timings": [], "inference_step_timings": []},
        run_json=run_json,
    )

    assert row["workload_blind_candidate"] is True
    assert row["beats_uniform_query_local_utility"] is True
    assert row["beats_douglas_peucker_query_local_utility"] is True
    assert row["single_cell_range_status"] == "protocol_fail"


def test_benchmark_row_blocks_blind_success_when_selector_scaffold_dominated(tmp_path) -> None:
    run_json = {
        "config": {"model": {"model_type": "range_prior", "compression_ratio": 0.05}},
        "workload_blind_protocol": {
            "enabled": True,
            "primary_masks_frozen_before_eval_query_scoring": True,
            "audit_masks_frozen_before_eval_query_scoring": True,
        },
        "matched": {
            "MLQDS": {"query_local_utility_score": 0.50},
            "uniform": {"query_local_utility_score": 0.40},
            "DouglasPeucker": {"query_local_utility_score": 0.30},
        },
        "selector_budget_diagnostics": {
            "eval": {
                "budget_rows": [
                    {
                        "compression_ratio": 0.05,
                        "learned_slot_fraction_of_budget": 0.10,
                    }
                ]
            }
        },
    }

    row = _row_from_run(
        workload="range",
        run_label="scaffold_dominated",
        command=[],
        returncode=0,
        elapsed_seconds=1.0,
        run_dir=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        run_json_path=tmp_path / "example_run.json",
        timings={"phase_timings": [], "epoch_timings": [], "inference_step_timings": []},
        run_json=run_json,
    )

    assert row["beats_uniform_query_local_utility"] is True
    assert row["beats_douglas_peucker_query_local_utility"] is True
    assert row["selector_claim_status"] == "selector_scaffold_dominated"
    assert row["selector_claim_has_material_learned_budget"] is False
    assert row["selector_claim_min_learned_slot_fraction"] == (
        MIN_MATCHED_LEARNED_SLOT_FRACTION_FOR_BLIND_CLAIM
    )
    assert row["single_cell_range_status"] == "selector_scaffold_dominated"


def test_benchmark_row_allows_blind_success_with_material_learned_budget(tmp_path) -> None:
    run_json = {
        "config": {"model": {"model_type": "range_prior", "compression_ratio": 0.05}},
        "workload_blind_protocol": {
            "enabled": True,
            "primary_masks_frozen_before_eval_query_scoring": True,
            "audit_masks_frozen_before_eval_query_scoring": True,
        },
        "matched": {
            "MLQDS": {"query_local_utility_score": 0.50},
            "uniform": {"query_local_utility_score": 0.40},
            "DouglasPeucker": {"query_local_utility_score": 0.30},
        },
        "selector_budget_diagnostics": {
            "eval": {
                "budget_rows": [
                    {
                        "compression_ratio": 0.05,
                        "learned_slot_fraction_of_budget": 0.35,
                    }
                ]
            }
        },
    }

    row = _row_from_run(
        workload="range",
        run_label="material_learned_budget",
        command=[],
        returncode=0,
        elapsed_seconds=1.0,
        run_dir=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        run_json_path=tmp_path / "example_run.json",
        timings={"phase_timings": [], "epoch_timings": [], "inference_step_timings": []},
        run_json=run_json,
    )

    assert row["selector_claim_status"] == "model_has_material_budget"
    assert row["selector_claim_has_material_learned_budget"] is True
    assert row["single_cell_range_status"] == "beats_uniform_and_douglas_peucker"


def test_benchmark_row_uses_query_local_utility_for_final_candidate_status(tmp_path) -> None:
    run_json = {
        "config": {"model": {"model_type": "range_prior", "compression_ratio": 0.05}},
        "final_claim_summary": {"primary_metric": "QueryLocalUtility"},
        "workload_blind_protocol": {
            "enabled": True,
            "primary_masks_frozen_before_eval_query_scoring": True,
            "audit_masks_frozen_before_eval_query_scoring": True,
        },
        "matched": {
            "MLQDS": {"query_local_utility_score": 0.50},
            "uniform": {"query_local_utility_score": 0.40},
            "DouglasPeucker": {"query_local_utility_score": 0.30},
        },
        "selector_budget_diagnostics": {
            "eval": {
                "budget_rows": [
                    {
                        "compression_ratio": 0.05,
                        "learned_slot_fraction_of_budget": 0.35,
                    }
                ]
            }
        },
    }

    row = _row_from_run(
        workload="range",
        run_label="query_local_candidate",
        command=[],
        returncode=0,
        elapsed_seconds=1.0,
        run_dir=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        run_json_path=tmp_path / "example_run.json",
        timings={"phase_timings": [], "epoch_timings": [], "inference_step_timings": []},
        run_json=run_json,
    )

    assert row["beats_uniform_query_local_utility"] is True
    assert row["beats_douglas_peucker_query_local_utility"] is True
    assert row["single_cell_range_status"] == "beats_uniform_and_douglas_peucker"


def test_benchmark_row_records_data_source_metadata(tmp_path) -> None:
    run_json = {
        "config": {"model": {"model_type": "historical_prior", "compression_ratio": 0.05}},
        "workload_blind_protocol": {
            "enabled": True,
            "primary_masks_frozen_before_eval_query_scoring": True,
            "audit_masks_frozen_before_eval_query_scoring": True,
        },
        "matched": {
            "MLQDS": {"query_local_utility_score": 0.50},
            "uniform": {"query_local_utility_score": 0.40},
            "DouglasPeucker": {"query_local_utility_score": 0.30},
        },
        "selector_budget_diagnostics": {
            "eval": {
                "budget_rows": [
                    {
                        "compression_ratio": 0.01,
                        "learned_slot_count": 0,
                        "learned_slot_fraction_of_budget": 0.0,
                    },
                    {
                        "compression_ratio": 0.05,
                        "learned_slot_count": 12,
                        "learned_slot_fraction_of_budget": 0.20,
                        "zero_learned_slot_trajectory_fraction": 0.10,
                        "endpoint_only_trajectory_fraction": 0.0,
                    },
                ]
            }
        },
    }

    row = _row_from_run(
        workload="range",
        run_label="multi_day",
        command=[],
        returncode=0,
        elapsed_seconds=1.0,
        run_dir=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        run_json_path=tmp_path / "example_run.json",
        timings={"phase_timings": [], "epoch_timings": [], "inference_step_timings": []},
        run_json=run_json,
        data_sources={
            "csv_path": None,
            "train_csv_path": "train_a.csv,train_b.csv",
            "validation_csv_path": "validation_a.csv,validation_b.csv",
            "eval_csv_path": "eval_a.csv,eval_b.csv",
            "selected_cleaned_csv_files": [
                "train_a.csv",
                "train_b.csv",
                "validation_a.csv",
                "validation_b.csv",
                "eval_a.csv",
                "eval_b.csv",
            ],
        },
    )

    assert row["train_csv_path"] == "train_a.csv,train_b.csv"
    assert row["validation_csv_path"] == "validation_a.csv,validation_b.csv"
    assert row["eval_csv_path"] == "eval_a.csv,eval_b.csv"
    assert row["train_csv_file_count"] == 2
    assert row["validation_csv_file_count"] == 2
    assert row["eval_csv_file_count"] == 2
    assert row["selected_cleaned_csv_file_count"] == 6
    assert row["selected_cleaned_csv_files"] == (
        "train_a.csv;train_b.csv;validation_a.csv;validation_b.csv;eval_a.csv;eval_b.csv"
    )
    assert row["eval_selector_matched_learned_slot_fraction"] == 0.20
    assert row["eval_selector_matched_zero_learned_trajectory_fraction"] == 0.10
    assert row["eval_selector_low_budget_zero_learned_ratio_count"] == 1
    assert row["eval_selector_low_budget_min_learned_slot_fraction"] == 0.0


def test_profile_args_use_csv_when_provided() -> None:
    args = argparse.Namespace(
        csv_path="../AISDATA/cleaned/day.csv",
        train_csv_path=None,
        validation_csv_path=None,
        eval_csv_path=None,
        cache_dir="Range_QDS/artifacts/cache/benchmark",
        refresh_cache=True,
        min_points_per_segment=4,
        max_points_per_segment=128,
        max_time_gap_seconds=3600.0,
        max_segments=16,
        max_trajectories=8,
    )
    data_sources = BenchmarkDataSources(csv_path="../AISDATA/cleaned/day.csv")

    assert _profile_args(DEFAULT_PROFILE, args, data_sources) == [
        "--csv_path",
        "../AISDATA/cleaned/day.csv",
        *_profile_core_args(),
        "--min_points_per_segment",
        "4",
        "--max_time_gap_seconds",
        "3600.0",
        "--max_points_per_segment",
        "128",
        "--max_segments",
        "16",
        "--max_trajectories",
        "8",
        "--cache_dir",
        "Range_QDS/artifacts/cache/benchmark",
        "--refresh_cache",
    ]


def test_profile_args_use_three_day_train_validation_eval_sources() -> None:
    args = argparse.Namespace(
        csv_path=None,
        train_csv_path="../AISDATA/cleaned/day1.csv",
        validation_csv_path="../AISDATA/cleaned/day2.csv",
        eval_csv_path="../AISDATA/cleaned/day3.csv",
        cache_dir="Range_QDS/artifacts/cache/benchmark",
        refresh_cache=True,
        min_points_per_segment=4,
        max_points_per_segment=None,
        max_time_gap_seconds=3600.0,
        max_segments=None,
        max_trajectories=None,
    )
    data_sources = BenchmarkDataSources(
        train_csv_path="../AISDATA/cleaned/day1.csv",
        validation_csv_path="../AISDATA/cleaned/day2.csv",
        eval_csv_path="../AISDATA/cleaned/day3.csv",
    )

    assert _profile_args(DEFAULT_PROFILE, args, data_sources, include_refresh_cache=False) == [
        "--train_csv_path",
        "../AISDATA/cleaned/day1.csv",
        "--validation_csv_path",
        "../AISDATA/cleaned/day2.csv",
        "--eval_csv_path",
        "../AISDATA/cleaned/day3.csv",
        *_profile_core_args(),
        "--min_points_per_segment",
        "4",
        "--max_time_gap_seconds",
        "3600.0",
        "--cache_dir",
        "Range_QDS/artifacts/cache/benchmark",
    ]


def test_profile_args_support_workload_blind_profiles() -> None:
    args = argparse.Namespace(
        csv_path=None,
        train_csv_path="../AISDATA/cleaned/day1.csv",
        validation_csv_path="../AISDATA/cleaned/day2.csv",
        eval_csv_path="../AISDATA/cleaned/day3.csv",
        cache_dir="Range_QDS/artifacts/cache/benchmark",
        refresh_cache=False,
        min_points_per_segment=4,
        max_points_per_segment=256,
        max_time_gap_seconds=3600.0,
        max_segments=120,
        max_trajectories=None,
    )
    data_sources = BenchmarkDataSources(
        train_csv_path="../AISDATA/cleaned/day1.csv",
        validation_csv_path="../AISDATA/cleaned/day2.csv",
        eval_csv_path="../AISDATA/cleaned/day3.csv",
    )

    profile_args = _profile_args(
        BLIND_RETAINED_FREQUENCY_PROFILE, args, data_sources, include_refresh_cache=False
    )

    assert profile_args[:6] == [
        "--train_csv_path",
        "../AISDATA/cleaned/day1.csv",
        "--validation_csv_path",
        "../AISDATA/cleaned/day2.csv",
        "--eval_csv_path",
        "../AISDATA/cleaned/day3.csv",
    ]
    assert profile_args[profile_args.index("--model_type") + 1] == "scalar_workload_blind_range"
    assert (
        profile_args[profile_args.index("--range_training_target_mode") + 1] == "retained_frequency"
    )
    assert profile_args[profile_args.index("--range_max_coverage_overshoot") + 1] == "0.02"
    assert profile_args[profile_args.index("--range_audit_compression_ratios") + 1] == (
        "0.01,0.02,0.05,0.10,0.15,0.20,0.30"
    )
    assert "--max_points_per_segment" in profile_args
    assert profile_args[profile_args.index("--max_segments") + 1] == "120"


def test_workload_aware_diagnostic_profile_uses_requested_training_shape() -> None:
    args = argparse.Namespace(
        csv_path=None,
        train_csv_path="../AISDATA/cleaned/day1.csv",
        validation_csv_path="../AISDATA/cleaned/day2.csv",
        eval_csv_path="../AISDATA/cleaned/day3.csv",
        cache_dir="Range_QDS/artifacts/cache/benchmark",
        refresh_cache=False,
        min_points_per_segment=4,
        max_points_per_segment=None,
        max_time_gap_seconds=3600.0,
        max_segments=None,
        max_trajectories=None,
    )
    data_sources = BenchmarkDataSources(
        train_csv_path="../AISDATA/cleaned/day1.csv",
        validation_csv_path="../AISDATA/cleaned/day2.csv",
        eval_csv_path="../AISDATA/cleaned/day3.csv",
    )

    profile_args = _profile_args(DEFAULT_PROFILE, args, data_sources, include_refresh_cache=False)

    assert profile_args[6 : 6 + len(_profile_core_args())] == _profile_core_args()
    assert "--max_points_per_segment" not in profile_args
    assert "--max_segments" not in profile_args
    assert "--max_trajectories" not in profile_args


def test_selection_query_count_matches_eval_query_count() -> None:
    cfg = build_run_config(n_queries=80, query_coverage=0.20, max_queries=None)

    assert validation_query_count(cfg) == 80


def test_csv_config_suppresses_inactive_synthetic_metadata() -> None:
    cfg = build_run_config(
        n_ships=24,
        n_points=200,
        train_csv_path="../AISDATA/cleaned/day1.csv",
        validation_csv_path="../AISDATA/cleaned/day2.csv",
        eval_csv_path="../AISDATA/cleaned/day3.csv",
    )

    assert cfg.data.n_ships is None
    assert cfg.data.n_points_per_ship is None
