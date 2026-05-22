"""Fixed-width scoring table renderers."""

from __future__ import annotations

from scoring.metrics import MethodScore


def _range_focused_results(results: dict[str, MethodScore]) -> bool:
    """Return true when a result table represents a pure range workload."""
    saw_range = False
    for metrics in results.values():
        if int(metrics.range_audit.get("range_query_count", 0) or 0) > 0 or (
            set(metrics.per_type_f1) <= {"range"} and "range" in metrics.per_type_f1
        ):
            saw_range = True
        for qtype, value in metrics.per_type_f1.items():
            if qtype != "range" and abs(float(value)) > 1e-12:
                return False
    return saw_range


def _range_point_metric(metrics: MethodScore) -> float:
    """Return the explicit range point metric, falling back for synthetic/test rows."""
    if int(metrics.range_audit.get("range_query_count", 0) or 0) > 0:
        return float(metrics.range_point_f1)
    return float(metrics.per_type_f1.get("range", metrics.aggregate_f1))


def _range_secondary_metric(metrics: MethodScore) -> float:
    """Return the active range table's secondary metric."""
    return float(metrics.query_local_utility_score)


def print_method_comparison_table(results: dict[str, MethodScore]) -> str:
    """Render fixed-width method comparison table with workload-specific labels."""
    range_focused = _range_focused_results(results)
    col1 = 24
    col2 = 14
    col3 = 18 if range_focused else 13
    col4, col5, col6, col7 = 12, 12, 14, 13
    primary_label = "RangePointF1" if range_focused else "AnswerF1"
    secondary_label = "QueryLocalUtility" if range_focused else "CombinedF1"
    support_label = "QueryRecall" if range_focused else "BoundaryF1"
    lines = []
    header = (
        f"{'Method':<{col1}}"
        f"{primary_label:>{col2}}"
        f"{secondary_label:>{col3}}"
        f"{'Compression':>{col4}}"
        f"{'AvgPtGap':>{col5}}"
        f"{'Latency(ms)':>{col6}}"
        f"{support_label:>{col7}}"
        f"{'Type':>{col7}}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    type_rows: tuple[str, ...] = () if range_focused else ("range",)
    for name, metrics in results.items():
        primary = _range_point_metric(metrics) if range_focused else float(metrics.aggregate_f1)
        secondary = (
            _range_secondary_metric(metrics)
            if range_focused
            else float(metrics.aggregate_combined_f1)
        )
        query_recall = float(metrics.query_point_recall) if range_focused else 0.0
        lines.append(
            f"{name:<{col1}}"
            f"{primary:>{col2}.6f}"
            f"{secondary:>{col3}.6f}"
            f"{metrics.compression_ratio:>{col4}.4f}"
            f"{metrics.avg_retained_point_gap:>{col5}.2f}"
            f"{metrics.latency_ms:>{col6}.2f}"
            f"{query_recall:>{col7}.6f}"
            f"{'all':>{col7}}"
        )
        lines.extend(
            (
                f"{'  - ' + t_name:<{col1}}"
                f"{metrics.per_type_f1.get(t_name, 0.0):>{col2}.6f}"
                f"{metrics.per_type_combined_f1.get(t_name, 0.0):>{col3}.6f}"
                f"{'':>{col4}}"
                f"{'':>{col5}}"
                f"{'':>{col6}}"
                f"{'':>{col7}}"
                f"{t_name:>{col7}}"
            )
            for t_name in type_rows
        )

    def _rel_pct(diff: float, baseline: float) -> str:
        if abs(baseline) < 1e-9:
            return "  n/a"
        return f"{100.0 * diff / baseline:+.1f}%"

    mlqds = results.get("MLQDS")
    diff_references = [
        ("uniform", results.get("uniform")),
        ("DouglasPeucker", results.get("DouglasPeucker")),
    ]
    if mlqds is not None and any(ref is not None for _, ref in diff_references):
        lines.append("-" * len(header))
        metric_pair = (
            f"RangePointF1 / {secondary_label}" if range_focused else "AnswerF1 / CombinedF1"
        )
        lines.append(f"{f'Diff vs MLQDS ({metric_pair}; abs and % vs baseline)':<{col1}}")
        for ref_name, ref in diff_references:
            if ref is None:
                continue
            if range_focused:
                agg_ans = _range_point_metric(mlqds) - _range_point_metric(ref)
                mlqds_secondary = _range_secondary_metric(mlqds)
                ref_secondary = _range_secondary_metric(ref)
                agg_comb = mlqds_secondary - ref_secondary
                agg_ans_pct = _rel_pct(agg_ans, _range_point_metric(ref))
                agg_comb_pct = _rel_pct(agg_comb, ref_secondary)
            else:
                agg_ans = mlqds.aggregate_f1 - ref.aggregate_f1
                agg_comb = mlqds.aggregate_combined_f1 - ref.aggregate_combined_f1
                agg_ans_pct = _rel_pct(agg_ans, ref.aggregate_f1)
                agg_comb_pct = _rel_pct(agg_comb, ref.aggregate_combined_f1)
            label = f"  vs {ref_name}"
            lines.append(
                f"{label:<{col1}}"
                f"{agg_ans:>+{col2}.6f}"
                f"{agg_comb:>+{col3}.6f}"
                f"{'':>{col4}}"
                f"{'':>{col5}}"
                f"{'':>{col6}}"
                f"{'':>{col7}}"
                f"{'all':>{col7}}"
            )
            lines.append(
                f"{'      (% vs baseline)':<{col1}}"
                f"{agg_ans_pct:>{col2}}"
                f"{agg_comb_pct:>{col3}}"
                f"{'':>{col4}}"
                f"{'':>{col5}}"
                f"{'':>{col6}}"
                f"{'':>{col7}}"
                f"{'all':>{col7}}"
            )
            for t_name in type_rows:
                ref_ans = ref.per_type_f1.get(t_name, 0.0)
                ref_comb = ref.per_type_combined_f1.get(t_name, 0.0)
                t_ans = mlqds.per_type_f1.get(t_name, 0.0) - ref_ans
                t_comb = mlqds.per_type_combined_f1.get(t_name, 0.0) - ref_comb
                t_ans_pct = _rel_pct(t_ans, ref_ans)
                t_comb_pct = _rel_pct(t_comb, ref_comb)
                lines.append(
                    f"{'    - ' + t_name:<{col1}}"
                    f"{t_ans:>+{col2}.6f}"
                    f"{t_comb:>+{col3}.6f}"
                    f"{'':>{col4}}"
                    f"{'':>{col5}}"
                    f"{'':>{col6}}"
                    f"{'':>{col7}}"
                    f"{t_name:>{col7}}"
                )
                lines.append(
                    f"{'      (% vs baseline)':<{col1}}"
                    f"{t_ans_pct:>{col2}}"
                    f"{t_comb_pct:>{col3}}"
                    f"{'':>{col4}}"
                    f"{'':>{col5}}"
                    f"{'':>{col6}}"
                    f"{'':>{col7}}"
                    f"{t_name:>{col7}}"
                )
    return "\n".join(lines)


def print_range_audit_table(results: dict[str, MethodScore]) -> str:
    """Render QueryLocalUtility range-audit components."""
    col1, col2, col3, col4, col5, col6, col7 = 24, 14, 13, 18, 10, 10, 12
    header = (
        f"{'Method':<{col1}}"
        f"{'RangePointF1':>{col2}}"
        f"{'QueryRecall':>{col3}}"
        f"{'QueryLocalUtility':>{col4}}"
        f"{'GapMin':>{col5}}"
        f"{'TurnCov':>{col6}}"
        f"{'InterpFid':>{col7}}"
    )
    lines = [header, "-" * len(header)]
    for name, metrics in results.items():
        lines.append(
            f"{name:<{col1}}"
            f"{_range_point_metric(metrics):>{col2}.6f}"
            f"{metrics.query_point_recall:>{col3}.6f}"
            f"{metrics.query_local_utility_score:>{col4}.6f}"
            f"{metrics.range_gap_min_coverage:>{col5}.6f}"
            f"{metrics.range_turn_coverage:>{col6}.6f}"
            f"{metrics.range_query_local_interpolation_fidelity:>{col7}.6f}"
        )
    return "\n".join(lines)


def print_geometric_distortion_table(results: dict[str, MethodScore]) -> str:
    """Render geometric-distortion + shape-aware utility comparison."""
    col1, col2, col3, col4, col5, col6, col7 = 24, 11, 11, 11, 11, 13, 13
    header = (
        f"{'Method':<{col1}}"
        f"{'AvgSED_km':>{col2}}"
        f"{'MaxSED_km':>{col3}}"
        f"{'AvgPED_km':>{col4}}"
        f"{'MaxPED_km':>{col5}}"
        f"{'LengthPres':>{col6}}"
        f"{'F1xLen':>{col7}}"
    )
    lines = [header, "-" * len(header)]
    for name, metrics in results.items():
        geometric = metrics.geometric_distortion or {}
        lines.append(
            f"{name:<{col1}}"
            f"{geometric.get('avg_sed_km', 0.0):>{col2}.4f}"
            f"{geometric.get('max_sed_km', 0.0):>{col3}.2f}"
            f"{geometric.get('avg_ped_km', 0.0):>{col4}.4f}"
            f"{geometric.get('max_ped_km', 0.0):>{col5}.2f}"
            f"{metrics.avg_length_preserved:>{col6}.4f}"
            f"{metrics.combined_query_shape_score:>{col7}.6f}"
        )
    return "\n".join(lines)


def print_shift_table(shift_grid: dict[str, dict[str, float]]) -> str:
    """Render train-workload to eval-workload aggregate F1 matrix table."""
    eval_cols = sorted({k for row in shift_grid.values() for k in row})
    col_w = 22
    header_label = "Train\\Eval"
    line = f"{header_label:<{col_w}}" + "".join(f"{c:>{col_w}}" for c in eval_cols)
    out = [line, "-" * len(line)]
    for train_name in sorted(shift_grid.keys()):
        row = f"{train_name:<{col_w}}"
        for eval_name in eval_cols:
            val = shift_grid[train_name].get(eval_name, float("nan"))
            row += f"{val:>{col_w}.4f}"
        out.append(row)
    return "\n".join(out)
