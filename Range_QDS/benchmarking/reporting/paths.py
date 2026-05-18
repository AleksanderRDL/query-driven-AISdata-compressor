"""Benchmark child-run path helpers."""

from __future__ import annotations

from pathlib import Path


def _child_run_dir(results_dir: Path, workload: str, run_label: str, workload_count: int) -> Path:
    """Return the child run output directory for a benchmark row."""
    if workload_count == 1:
        return results_dir / run_label
    return results_dir / workload / run_label
