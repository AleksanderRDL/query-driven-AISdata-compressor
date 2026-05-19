"""Final-grid acceptance evidence for query-driven benchmark reports."""

from __future__ import annotations

from typing import Any

from benchmarking.common import (
    LOW_COMPRESSION_THRESHOLD,
    as_float,
    audit_ratio_prefix,
)
from workloads.generation.workload_profiles import RANGE_QUERY_MIX_FINAL_PROFILE_IDS

QUERY_DRIVEN_FINAL_WORKLOAD_PROFILE_IDS = RANGE_QUERY_MIX_FINAL_PROFILE_IDS
QUERY_DRIVEN_FINAL_COMPRESSION_RATIOS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
QUERY_DRIVEN_MIN_UNIFORM_WINS = 19
QUERY_DRIVEN_MIN_DP_WINS = 24
QUERY_DRIVEN_MIN_LOW_BUDGET_UNIFORM_WINS = 7
QUERY_DRIVEN_MIN_MATCHED_5_PERCENT_COMPRESSION_UNIFORM_WINS = 3


def _ratio_close(left: float | None, right: float, tol: float = 1e-9) -> bool:
    """Return whether two optional ratios should be treated as the same grid value."""
    return left is not None and abs(float(left) - float(right)) <= tol


def _normalized_grid_float(value: Any) -> float | None:
    """Coerce a grid fraction or percent to a normalized fraction."""
    number = as_float(value)
    if number is None:
        return None
    if number > 1.0 and number <= 100.0:
        number /= 100.0
    return float(number)


def query_driven_final_grid_summary(
    rows: list[dict[str, Any]],
    run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return benchmark-level QueryLocalUtility final-grid acceptance evidence."""
    run_config = run_config or {}
    profile_settings = run_config.get("profile_settings") or {}
    required_workload_profile_ids = tuple(
        str(value)
        for value in (
            profile_settings.get("range_workload_profile_sweep_ids")
            or QUERY_DRIVEN_FINAL_WORKLOAD_PROFILE_IDS
        )
    )
    required_ratios = tuple(
        float(value)
        for value in (
            profile_settings.get("range_compression_sweep_ratios")
            or QUERY_DRIVEN_FINAL_COMPRESSION_RATIOS
        )
    )
    final_candidate = bool(profile_settings.get("final_product_candidate")) or any(
        row.get("mlqds_primary_metric") == "query_local_utility" for row in rows
    )
    profile_rows: dict[str, dict[str, Any]] = {}
    duplicate_workload_profile_ids: list[str] = []
    for row in rows:
        workload_profile_id = row.get("workload_profile_id")
        if not isinstance(workload_profile_id, str):
            continue
        if workload_profile_id not in required_workload_profile_ids:
            continue
        if workload_profile_id in profile_rows:
            duplicate_workload_profile_ids.append(workload_profile_id)
            continue
        profile_rows[workload_profile_id] = row

    missing_workload_profile_ids = [
        profile_id
        for profile_id in required_workload_profile_ids
        if profile_id not in profile_rows
    ]
    cells: list[dict[str, Any]] = []
    missing_cells: list[dict[str, Any]] = []
    for workload_profile_id in required_workload_profile_ids:
        row = profile_rows.get(workload_profile_id)
        for ratio in required_ratios:
            if row is None:
                missing_cells.append(
                    {
                        "workload_profile_id": workload_profile_id,
                        "compression_ratio": float(ratio),
                        "reason": "missing_workload_profile_row",
                    }
                )
                continue
            prefix = audit_ratio_prefix(ratio)
            mlqds = as_float(row.get(f"{prefix}_mlqds_query_local_utility"))
            uniform = as_float(row.get(f"{prefix}_uniform_query_local_utility"))
            dp = as_float(row.get(f"{prefix}_douglas_peucker_query_local_utility"))
            if mlqds is None or uniform is None or dp is None:
                if _ratio_close(
                    _normalized_grid_float(row.get("compression_ratio")), ratio, tol=1e-6
                ):
                    mlqds = as_float(row.get("mlqds_query_local_utility_score"))
                    uniform = as_float(row.get("uniform_query_local_utility_score"))
                    dp = as_float(row.get("douglas_peucker_query_local_utility_score"))
            if mlqds is None or uniform is None or dp is None:
                missing_cells.append(
                    {
                        "workload_profile_id": workload_profile_id,
                        "compression_ratio": float(ratio),
                        "reason": "missing_query_local_utility_scores",
                    }
                )
                continue
            cells.append(
                {
                    "workload_profile_id": workload_profile_id,
                    "compression_ratio": float(ratio),
                    "mlqds_query_local_utility": float(mlqds),
                    "uniform_query_local_utility": float(uniform),
                    "douglas_peucker_query_local_utility": float(dp),
                    "mlqds_vs_uniform_query_local_utility": float(mlqds - uniform),
                    "mlqds_vs_douglas_peucker_query_local_utility": float(mlqds - dp),
                    "beats_uniform": bool(mlqds > uniform),
                    "beats_douglas_peucker": bool(mlqds > dp),
                    "low_budget": bool(ratio <= LOW_COMPRESSION_THRESHOLD),
                }
            )

    uniform_wins = sum(1 for cell in cells if cell["beats_uniform"])
    dp_wins = sum(1 for cell in cells if cell["beats_douglas_peucker"])
    low_uniform_wins = sum(1 for cell in cells if cell["low_budget"] and cell["beats_uniform"])
    matched_5_percent_compression_uniform_wins = sum(
        1
        for cell in cells
        if _ratio_close(as_float(cell.get("compression_ratio")), 0.05, tol=1e-9)
        and cell["beats_uniform"]
    )
    required_cell_count = int(len(required_workload_profile_ids) * len(required_ratios))
    grid_complete = (
        len(cells) == required_cell_count
        and not missing_cells
        and not missing_workload_profile_ids
    )
    required_single_run_gate_names = (
        "workload_stability_gate_pass",
        "support_overlap_gate_pass",
        "predictability_gate_pass",
        "prior_predictive_alignment_gate_pass",
        "target_diffusion_gate_pass",
        "workload_signature_gate_pass",
        "learning_causality_gate_pass",
        "prior_sample_gate_pass",
        "global_sanity_gate_pass",
    )
    child_gate_failures: list[dict[str, Any]] = []
    for workload_profile_id, row in sorted(profile_rows.items()):
        failed = [name for name in required_single_run_gate_names if row.get(name) is not True]
        if int(row.get("returncode", 1) or 0) != 0:
            failed.append("child_returncode_nonzero")
        if failed:
            child_gate_failures.append(
                {
                    "workload_profile_id": workload_profile_id,
                    "run_label": row.get("run_label"),
                    "failed_gates": failed,
                }
            )

    numeric_success_pass = (
        grid_complete
        and uniform_wins >= QUERY_DRIVEN_MIN_UNIFORM_WINS
        and dp_wins >= QUERY_DRIVEN_MIN_DP_WINS
        and low_uniform_wins >= QUERY_DRIVEN_MIN_LOW_BUDGET_UNIFORM_WINS
        and matched_5_percent_compression_uniform_wins
        >= QUERY_DRIVEN_MIN_MATCHED_5_PERCENT_COMPRESSION_UNIFORM_WINS
    )
    failed_checks: list[str] = []
    if not final_candidate:
        failed_checks.append("not_final_product_candidate_profile")
    if missing_workload_profile_ids:
        failed_checks.append("workload_profile_grid_incomplete")
    if missing_cells:
        failed_checks.append("compression_grid_incomplete")
    if duplicate_workload_profile_ids:
        failed_checks.append("duplicate_workload_profile_rows")
    if uniform_wins < QUERY_DRIVEN_MIN_UNIFORM_WINS:
        failed_checks.append("too_few_uniform_queryuseful_wins")
    if dp_wins < QUERY_DRIVEN_MIN_DP_WINS:
        failed_checks.append("too_few_douglas_peucker_queryuseful_wins")
    if low_uniform_wins < QUERY_DRIVEN_MIN_LOW_BUDGET_UNIFORM_WINS:
        failed_checks.append("too_few_low_budget_uniform_queryuseful_wins")
    if (
        matched_5_percent_compression_uniform_wins
        < QUERY_DRIVEN_MIN_MATCHED_5_PERCENT_COMPRESSION_UNIFORM_WINS
    ):
        failed_checks.append("too_few_matched_5_percent_compression_uniform_queryuseful_wins")
    if child_gate_failures:
        failed_checks.append("required_single_run_gates_failed")

    final_success_allowed = bool(
        final_candidate and numeric_success_pass and not child_gate_failures and not failed_checks
    )
    return {
        "schema_version": 1,
        "primary_metric": "QueryLocalUtility",
        "status": "final_grid_pass" if final_success_allowed else "final_grid_blocked",
        "final_success_allowed": final_success_allowed,
        "failed_checks": failed_checks,
        "final_product_candidate_profile": bool(final_candidate),
        "required_workload_profile_ids": list(required_workload_profile_ids),
        "required_compression_ratios": list(required_ratios),
        "required_cell_count": required_cell_count,
        "observed_cell_count": len(cells),
        "grid_complete": bool(grid_complete),
        "missing_workload_profile_ids": missing_workload_profile_ids,
        "duplicate_workload_profile_ids": duplicate_workload_profile_ids,
        "missing_cells": missing_cells,
        "beats_uniform_queryuseful_cells": int(uniform_wins),
        "beats_uniform_queryuseful_cells_min": QUERY_DRIVEN_MIN_UNIFORM_WINS,
        "beats_douglas_peucker_queryuseful_cells": int(dp_wins),
        "beats_douglas_peucker_queryuseful_cells_min": QUERY_DRIVEN_MIN_DP_WINS,
        "low_budget_beats_uniform_queryuseful_cells": int(low_uniform_wins),
        "low_budget_beats_uniform_queryuseful_cells_min": (
            QUERY_DRIVEN_MIN_LOW_BUDGET_UNIFORM_WINS
        ),
        "matched_5_percent_compression_cells_uniform": int(
            matched_5_percent_compression_uniform_wins
        ),
        "matched_5_percent_compression_cells_uniform_min": (
            QUERY_DRIVEN_MIN_MATCHED_5_PERCENT_COMPRESSION_UNIFORM_WINS
        ),
        "numeric_success_bars_pass": bool(numeric_success_pass),
        "required_single_run_gate_names": list(required_single_run_gate_names),
        "child_gate_failures": child_gate_failures,
        "cells": cells,
    }
