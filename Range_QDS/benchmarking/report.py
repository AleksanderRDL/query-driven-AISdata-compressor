"""Benchmark report artifact construction and output writing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarking.artifacts import (
    artifact_index,
    format_artifact_readme,
    write_csv,
    write_json,
)
from benchmarking.final_grid import query_driven_final_grid_summary
from benchmarking.table import _format_report_table


def _write_text(path: Path, text: str) -> None:
    """Write text, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_benchmark_report_artifact(
    *,
    timestamp_utc: str,
    command: list[str],
    run_id: str,
    results_dir: Path,
    run_family_root: Path,
    profile: str,
    seed: int,
    workloads: list[str],
    run_label: str,
    run_config: dict[str, Any],
    data_sources: dict[str, Any],
    cache_warmup: list[dict[str, Any]],
    environment: dict[str, Any],
    git: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the machine-readable benchmark report artifact."""
    artifact = {
        "schema_version": 5,
        "timestamp_utc": timestamp_utc,
        "command": command,
        "run_id": run_id,
        "artifact_root": str(results_dir),
        "family_root": str(run_family_root),
        "profile": profile,
        "seed": int(seed),
        "workloads": workloads,
        "run_label": run_label,
        "run_config": run_config,
        "data_sources": data_sources,
        "cache_warmup": cache_warmup,
        "environment": environment,
        "git": git,
        "rows": rows,
    }
    artifact["query_driven_final_grid_summary"] = query_driven_final_grid_summary(rows, run_config)
    return artifact


def write_benchmark_report_files(
    results_dir: Path, artifact: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Write benchmark report JSON, CSV, Markdown, index, and README files."""
    write_json(results_dir / "benchmark_report.json", artifact)
    write_csv(results_dir / "benchmark_report.csv", rows)
    _write_text(results_dir / "benchmark_report.md", _format_report_table(rows))
    index = artifact_index(results_dir, artifact, rows)
    write_json(results_dir / "artifact_index.json", index)
    _write_text(results_dir / "README.md", format_artifact_readme(artifact, rows))
    return index
