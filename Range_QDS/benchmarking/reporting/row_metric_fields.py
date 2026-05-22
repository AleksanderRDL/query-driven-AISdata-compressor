"""Matched-method metric fields for benchmark reporting rows."""

from __future__ import annotations

from typing import Any

from benchmarking.common import as_float
from benchmarking.reporting.metrics import (
    RANGE_COMPONENT_KEYS,
    _geometry_fields,
    _metric_delta,
    _worst_uniform_component_delta,
)
from benchmarking.reporting.row_context import RowContext, RowFields, _mapping
from benchmarking.row_runtime import collapse_warning_summary


def _milliseconds_to_seconds(value: Any) -> float | None:
    milliseconds = as_float(value)
    return None if milliseconds is None else milliseconds / 1000.0


def _metric_difference(left: Any, right: Any) -> float | None:
    return float(left) - float(right) if left is not None and right is not None else None


def _matched_metric_values(ctx: RowContext) -> RowFields:
    mlqds = ctx.mlqds
    uniform = ctx.uniform
    dp = ctx.douglas_peucker
    mlqds_aggregate_f1 = mlqds.get("aggregate_f1")
    mlqds_range_usefulness = mlqds.get("range_usefulness_score")
    mlqds_range_point_f1 = mlqds.get("range_point_f1", mlqds_aggregate_f1)
    mlqds_query_local_utility = mlqds.get("query_local_utility_score")
    final_claim_summary = _mapping(ctx.run.get("final_claim_summary"))
    if (
        final_claim_summary.get("primary_metric") == "QueryLocalUtility"
        and mlqds_query_local_utility is not None
    ):
        mlqds_primary_metric = "query_local_utility"
        mlqds_primary_score = mlqds_query_local_utility
    else:
        mlqds_primary_metric = (
            "range_usefulness" if mlqds_range_usefulness is not None else "range_point_f1"
        )
        mlqds_primary_score = (
            mlqds_range_usefulness if mlqds_range_usefulness is not None else mlqds_range_point_f1
        )
    uniform_aggregate_f1 = uniform.get("aggregate_f1")
    dp_aggregate_f1 = dp.get("aggregate_f1")
    return {
        "mlqds_primary_metric": mlqds_primary_metric,
        "mlqds_primary_score": mlqds_primary_score,
        "mlqds_aggregate_f1": mlqds_aggregate_f1,
        "mlqds_query_point_recall": mlqds.get("query_point_recall"),
        "mlqds_range_point_f1": mlqds_range_point_f1,
        "mlqds_range_usefulness": mlqds_range_usefulness,
        "mlqds_query_local_utility": mlqds_query_local_utility,
        "mlqds_gap_time_usefulness": mlqds.get("range_usefulness_gap_time_score"),
        "mlqds_gap_distance_usefulness": mlqds.get("range_usefulness_gap_distance_score"),
        "mlqds_gap_min_usefulness": mlqds.get("range_usefulness_gap_min_score"),
        "mlqds_inference_only_latency_ms": mlqds.get("latency_ms"),
        "uniform_aggregate_f1": uniform_aggregate_f1,
        "uniform_query_point_recall": uniform.get("query_point_recall"),
        "uniform_range_point_f1": uniform.get("range_point_f1", uniform_aggregate_f1),
        "uniform_range_usefulness": uniform.get("range_usefulness_score"),
        "uniform_query_local_utility": uniform.get("query_local_utility_score"),
        "uniform_gap_time_usefulness": uniform.get("range_usefulness_gap_time_score"),
        "uniform_gap_distance_usefulness": uniform.get("range_usefulness_gap_distance_score"),
        "uniform_gap_min_usefulness": uniform.get("range_usefulness_gap_min_score"),
        "dp_aggregate_f1": dp_aggregate_f1,
        "dp_query_point_recall": dp.get("query_point_recall"),
        "dp_range_point_f1": dp.get("range_point_f1", dp_aggregate_f1),
        "dp_range_usefulness": dp.get("range_usefulness_score"),
        "dp_query_local_utility": dp.get("query_local_utility_score"),
        "dp_gap_time_usefulness": dp.get("range_usefulness_gap_time_score"),
        "dp_gap_distance_usefulness": dp.get("range_usefulness_gap_distance_score"),
        "dp_gap_min_usefulness": dp.get("range_usefulness_gap_min_score"),
        "random_fill_range_usefulness": ctx.temporal_random_fill.get("range_usefulness_score"),
        "oracle_fill_range_usefulness": ctx.temporal_oracle_fill.get("range_usefulness_score"),
    }


def _mlqds_metric_fields(ctx: RowContext, values: RowFields) -> RowFields:
    mlqds = ctx.mlqds
    return {
        "mlqds_primary_metric": values["mlqds_primary_metric"],
        "mlqds_primary_score": values["mlqds_primary_score"],
        "mlqds_aggregate_f1": values["mlqds_aggregate_f1"],
        "mlqds_query_point_recall": values["mlqds_query_point_recall"],
        "mlqds_range_point_f1": values["mlqds_range_point_f1"],
        "mlqds_range_usefulness": values["mlqds_range_usefulness"],
        "mlqds_range_usefulness_score": values["mlqds_range_usefulness"],
        "mlqds_query_local_utility_score": values["mlqds_query_local_utility"],
        "mlqds_range_usefulness_gap_time_score": values["mlqds_gap_time_usefulness"],
        "mlqds_range_usefulness_gap_distance_score": values["mlqds_gap_distance_usefulness"],
        "mlqds_range_usefulness_gap_min_score": values["mlqds_gap_min_usefulness"],
        "mlqds_type_f1": _mapping(mlqds.get("per_type_f1")).get(ctx.workload),
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
        "final_metrics_mode": ctx.run.get(
            "final_metrics_mode", ctx.baseline_config.get("final_metrics_mode")
        ),
    }


def _mlqds_latency_fields(ctx: RowContext, values: RowFields) -> RowFields:
    mlqds = ctx.mlqds
    latency_ms = values["mlqds_inference_only_latency_ms"]
    return {
        "mlqds_latency_ms": latency_ms,
        "mlqds_inference_only_latency_ms": latency_ms,
        "mlqds_inference_only_latency_seconds": _milliseconds_to_seconds(latency_ms),
        "avg_length_preserved": mlqds.get("avg_length_preserved"),
        "combined_query_shape_score": mlqds.get("combined_query_shape_score"),
    }


def _baseline_metric_fields(ctx: RowContext, values: RowFields) -> RowFields:
    uniform = ctx.uniform
    dp = ctx.douglas_peucker
    return {
        "uniform_aggregate_f1": values["uniform_aggregate_f1"],
        "uniform_query_point_recall": values["uniform_query_point_recall"],
        "uniform_range_point_f1": values["uniform_range_point_f1"],
        "uniform_range_usefulness": values["uniform_range_usefulness"],
        "uniform_range_usefulness_score": values["uniform_range_usefulness"],
        "uniform_query_local_utility_score": values["uniform_query_local_utility"],
        "uniform_range_usefulness_gap_time_score": values["uniform_gap_time_usefulness"],
        "uniform_range_usefulness_gap_distance_score": values["uniform_gap_distance_usefulness"],
        "uniform_range_usefulness_gap_min_score": values["uniform_gap_min_usefulness"],
        "uniform_range_ship_f1": uniform.get("range_ship_f1"),
        "uniform_range_ship_coverage": uniform.get("range_ship_coverage"),
        "uniform_range_entry_exit_f1": uniform.get("range_entry_exit_f1"),
        "uniform_range_crossing_f1": uniform.get("range_crossing_f1"),
        "uniform_range_temporal_coverage": uniform.get("range_temporal_coverage"),
        "uniform_range_gap_coverage": uniform.get("range_gap_coverage"),
        "uniform_range_turn_coverage": uniform.get("range_turn_coverage"),
        "uniform_range_shape_score": uniform.get("range_shape_score"),
        **_geometry_fields("uniform", uniform),
        "douglas_peucker_aggregate_f1": values["dp_aggregate_f1"],
        "douglas_peucker_query_point_recall": values["dp_query_point_recall"],
        "douglas_peucker_range_point_f1": values["dp_range_point_f1"],
        "douglas_peucker_range_usefulness": values["dp_range_usefulness"],
        "douglas_peucker_range_usefulness_score": values["dp_range_usefulness"],
        "douglas_peucker_query_local_utility_score": values["dp_query_local_utility"],
        "douglas_peucker_range_usefulness_gap_time_score": values["dp_gap_time_usefulness"],
        "douglas_peucker_range_usefulness_gap_distance_score": values["dp_gap_distance_usefulness"],
        "douglas_peucker_range_usefulness_gap_min_score": values["dp_gap_min_usefulness"],
        "douglas_peucker_range_ship_f1": dp.get("range_ship_f1"),
        "douglas_peucker_range_ship_coverage": dp.get("range_ship_coverage"),
        "douglas_peucker_range_entry_exit_f1": dp.get("range_entry_exit_f1"),
        "douglas_peucker_range_crossing_f1": dp.get("range_crossing_f1"),
        "douglas_peucker_range_temporal_coverage": dp.get("range_temporal_coverage"),
        "douglas_peucker_range_gap_coverage": dp.get("range_gap_coverage"),
        "douglas_peucker_range_turn_coverage": dp.get("range_turn_coverage"),
        "douglas_peucker_range_shape_score": dp.get("range_shape_score"),
        **_geometry_fields("douglas_peucker", dp),
    }


def _comparison_metric_fields(ctx: RowContext, values: RowFields) -> RowFields:
    component_deltas = {
        f"mlqds_vs_uniform_{key}": _metric_delta(ctx.mlqds, ctx.uniform, key)
        for key in RANGE_COMPONENT_KEYS
    }
    return {
        "mlqds_vs_uniform_range_point_f1": _metric_difference(
            values["mlqds_range_point_f1"], values["uniform_range_point_f1"]
        ),
        "mlqds_vs_douglas_peucker_query_point_recall": _metric_difference(
            values["mlqds_query_point_recall"], values["dp_query_point_recall"]
        ),
        "mlqds_vs_douglas_peucker_range_point_f1": _metric_difference(
            values["mlqds_range_point_f1"], values["dp_range_point_f1"]
        ),
        "mlqds_vs_uniform_range_usefulness": _metric_difference(
            values["mlqds_range_usefulness"], values["uniform_range_usefulness"]
        ),
        "mlqds_vs_uniform_query_local_utility": _metric_difference(
            values["mlqds_query_local_utility"], values["uniform_query_local_utility"]
        ),
        "mlqds_vs_douglas_peucker_range_usefulness": _metric_difference(
            values["mlqds_range_usefulness"], values["dp_range_usefulness"]
        ),
        "mlqds_vs_douglas_peucker_query_local_utility": _metric_difference(
            values["mlqds_query_local_utility"], values["dp_query_local_utility"]
        ),
        "mlqds_vs_uniform_range_usefulness_gap_time": _metric_difference(
            values["mlqds_gap_time_usefulness"], values["uniform_gap_time_usefulness"]
        ),
        "mlqds_vs_uniform_range_usefulness_gap_distance": _metric_difference(
            values["mlqds_gap_distance_usefulness"], values["uniform_gap_distance_usefulness"]
        ),
        "mlqds_vs_uniform_range_usefulness_gap_min": _metric_difference(
            values["mlqds_gap_min_usefulness"], values["uniform_gap_min_usefulness"]
        ),
        "mlqds_vs_douglas_peucker_range_usefulness_gap_time": _metric_difference(
            values["mlqds_gap_time_usefulness"], values["dp_gap_time_usefulness"]
        ),
        "mlqds_vs_douglas_peucker_range_usefulness_gap_distance": _metric_difference(
            values["mlqds_gap_distance_usefulness"], values["dp_gap_distance_usefulness"]
        ),
        "mlqds_vs_douglas_peucker_range_usefulness_gap_min": _metric_difference(
            values["mlqds_gap_min_usefulness"], values["dp_gap_min_usefulness"]
        ),
        **component_deltas,
        **_worst_uniform_component_delta(component_deltas),
        "mlqds_vs_uniform_avg_sed_km": _metric_delta(
            {"value": _mapping(ctx.mlqds.get("geometric_distortion")).get("avg_sed_km")},
            {"value": _mapping(ctx.uniform.get("geometric_distortion")).get("avg_sed_km")},
            "value",
        ),
        "mlqds_vs_uniform_avg_ped_km": _metric_delta(
            {"value": _mapping(ctx.mlqds.get("geometric_distortion")).get("avg_ped_km")},
            {"value": _mapping(ctx.uniform.get("geometric_distortion")).get("avg_ped_km")},
            "value",
        ),
        "mlqds_vs_uniform_avg_length_preserved": _metric_delta(
            ctx.mlqds,
            ctx.uniform,
            "avg_length_preserved",
        ),
    }


def _fill_and_collapse_fields(ctx: RowContext, values: RowFields) -> RowFields:
    collapse_summary = collapse_warning_summary(ctx.run_json)
    return {
        "temporal_random_fill_range_point_f1": ctx.temporal_random_fill.get("range_point_f1"),
        "temporal_random_fill_range_usefulness_score": values["random_fill_range_usefulness"],
        "temporal_oracle_fill_range_point_f1": ctx.temporal_oracle_fill.get("range_point_f1"),
        "temporal_oracle_fill_range_usefulness_score": values["oracle_fill_range_usefulness"],
        "mlqds_vs_temporal_random_fill_range_usefulness": _metric_difference(
            values["mlqds_range_usefulness"], values["random_fill_range_usefulness"]
        ),
        "temporal_oracle_fill_gap_range_usefulness": _metric_difference(
            values["oracle_fill_range_usefulness"], values["mlqds_range_usefulness"]
        ),
        "collapse_warning": collapse_summary["collapse_warning_any"],
        "collapse_warning_any": collapse_summary["collapse_warning_any"],
        "collapse_warning_count": collapse_summary["collapse_warning_count"],
        "best_epoch_collapse_warning": collapse_summary["best_epoch_collapse_warning"],
        "min_pred_std": collapse_summary["min_pred_std"],
        "best_epoch_pred_std": collapse_summary["best_epoch_pred_std"],
    }


def _method_metric_fields(ctx: RowContext) -> RowFields:
    values = _matched_metric_values(ctx)
    fields: RowFields = {}
    fields.update(_mlqds_metric_fields(ctx, values))
    fields.update(_baseline_metric_fields(ctx, values))
    fields.update(_comparison_metric_fields(ctx, values))
    fields.update(_mlqds_latency_fields(ctx, values))
    fields.update(_fill_and_collapse_fields(ctx, values))
    return fields
