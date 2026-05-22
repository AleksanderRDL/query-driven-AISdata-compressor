"""Tests for benchmark runner artifacts, data sources, and capture helpers."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

import benchmarking.runner as runner
from benchmarking.artifacts import index_entry, write_family_indexes
from benchmarking.child_process import BenchmarkChildResult
from benchmarking.profiles import DEFAULT_PROFILE
from benchmarking.reporting.paths import _child_run_dir
from benchmarking.runner import BenchmarkDataSources, _resolve_data_sources, _run_capture_streaming
from benchmarking.table import _format_report_table


def test_child_run_dir_uses_readable_layout(tmp_path) -> None:
    run_label = "custom_run"

    assert _child_run_dir(tmp_path, "range", run_label, 1) == tmp_path / "custom_run"
    assert _child_run_dir(tmp_path, "range", run_label, 2) == tmp_path / "range" / "custom_run"


def test_family_index_upserts_current_status_and_appends_events(tmp_path) -> None:
    args = argparse.Namespace(
        profile=DEFAULT_PROFILE,
        seed=42,
        max_points_per_segment=3000,
        max_segments=None,
        max_trajectories=None,
    )
    run_label = "custom_run"
    sources = BenchmarkDataSources(
        train_csv_path="day1.csv", validation_csv_path="day2.csv", eval_csv_path="day3.csv"
    )
    git = {"commit": "abc123", "dirty": False}
    running_status = {
        "status": "running",
        "started_at_utc": "2026-05-10T00:00:00+00:00",
        "finished_at_utc": None,
        "exit_status": None,
        "failures": None,
    }
    completed_status = {
        **running_status,
        "status": "completed",
        "finished_at_utc": "2026-05-10T00:01:00+00:00",
        "exit_status": 0,
        "failures": 0,
    }

    write_family_indexes(
        tmp_path,
        index_entry(
            run_id="run-a",
            status_payload=running_status,
            args=args,
            workloads=["range"],
            run_label=run_label,
            data_sources=sources,
            results_dir=tmp_path / "runs" / "run-a",
            rows=[],
            git=git,
        ),
    )
    write_family_indexes(
        tmp_path,
        index_entry(
            run_id="run-a",
            status_payload=completed_status,
            args=args,
            workloads=["range"],
            run_label=run_label,
            data_sources=sources,
            results_dir=tmp_path / "runs" / "run-a",
            rows=[
                {
                    "run_label": "custom_run",
                    "mlqds_primary_metric": "range_usefulness",
                    "mlqds_primary_score": 0.42,
                    "mlqds_aggregate_f1": 0.4,
                    "mlqds_range_point_f1": 0.4,
                    "mlqds_range_usefulness": 0.42,
                }
            ],
            git=git,
        ),
    )

    with open(tmp_path / "runs_index.csv", encoding="utf-8", newline="") as f:
        index_rows = list(csv.DictReader(f))
    events_text = (tmp_path / "runs_index_events.jsonl").read_text(encoding="utf-8")
    assert len(index_rows) == 1
    assert index_rows[0]["run_id"] == "run-a"
    assert index_rows[0]["status"] == "completed"
    assert index_rows[0]["run_label"] == "custom_run"
    assert index_rows[0]["best_mlqds_primary_metric"] == "range_usefulness"
    assert index_rows[0]["best_mlqds_primary_score"] == "0.42"
    assert index_rows[0]["best_mlqds_aggregate_f1"] == "0.4"
    assert index_rows[0]["best_mlqds_range_point_f1"] == "0.4"
    assert index_rows[0]["best_mlqds_range_usefulness"] == "0.42"
    assert index_rows[0]["best_mlqds_run_label"] == "custom_run"
    assert events_text.count('"run_id": "run-a"') == 2


def test_benchmark_report_records_concrete_family_root(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    family = tmp_path / "range_family"
    results_dir = family / "runs" / "artifact-test"
    train_csv = tmp_path / "train.csv"
    validation_csv = tmp_path / "validation.csv"
    eval_csv = tmp_path / "eval.csv"
    for path in (train_csv, validation_csv, eval_csv):
        path.write_text("mmsi,timestamp,lat,lon\n", encoding="utf-8")

    def fake_run_capture_streaming(
        command: list[str],
        cwd: Path,
        stdout_path: Path,
        *,
        max_stdout_chars: int = 1_000_000,
    ) -> BenchmarkChildResult:
        run_dir = stdout_path.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("[train-model] done in 1.00s\n", encoding="utf-8")
        (run_dir / "example_run.json").write_text(
            json.dumps(
                {
                    "config": {"model": {"model_type": "range_aware", "compression_ratio": 0.05}},
                    "matched": {"MLQDS": {"range_point_f1": 0.4, "range_usefulness_score": 0.5}},
                }
            ),
            encoding="utf-8",
        )
        return BenchmarkChildResult(
            returncode=0,
            stdout="",
            stdout_truncated=False,
            timings={"phase_timings": [], "epoch_timings": [], "inference_step_timings": []},
            elapsed_seconds=1.0,
        )

    monkeypatch.setattr(runner, "_run_capture_streaming", fake_run_capture_streaming)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--results_dir",
            str(results_dir),
            "--run_id",
            "artifact-test",
            "--run_label",
            "unit",
            "--workload_profile_ids",
            "range_query_mix_focused,range_query_mix_local",
            "--workloads",
            "range",
            "--train_csv_path",
            str(train_csv),
            "--validation_csv_path",
            str(validation_csv),
            "--eval_csv_path",
            str(eval_csv),
            "--no_cache_warmup",
        ],
    )

    runner.main()

    artifact = json.loads((results_dir / "benchmark_report.json").read_text(encoding="utf-8"))
    assert artifact["family_root"] == str(family)
    assert "<function" not in artifact["family_root"]
    assert artifact["run_config"]["workload_profile_ids"] == [
        "range_query_mix_focused",
        "range_query_mix_local",
    ]
    assert [row["run_label"] for row in artifact["rows"]] == [
        "unit_range_query_mix_focused",
        "unit_range_query_mix_local",
    ]
    assert artifact["rows"][0]["command"][-2:] == [
        "--workload_profile_id",
        "range_query_mix_focused",
    ]
    assert artifact["rows"][1]["command"][-2:] == [
        "--workload_profile_id",
        "range_query_mix_local",
    ]
    assert artifact["rows"][0]["train_csv_path"] == str(train_csv)
    assert artifact["rows"][0]["validation_csv_path"] == str(validation_csv)
    assert artifact["rows"][0]["eval_csv_path"] == str(eval_csv)
    assert artifact["rows"][0]["selected_cleaned_csv_file_count"] == 3


def test_resolve_data_sources_selects_three_cleaned_days(tmp_path) -> None:
    (tmp_path / "aisdk-2026-02-02_cleaned.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "aisdk-2026-02-03_cleaned.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "aisdk-2026-02-04_cleaned.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("ignore\n", encoding="utf-8")
    args = argparse.Namespace(
        csv_path=str(tmp_path), train_csv_path=None, validation_csv_path=None, eval_csv_path=None
    )

    sources = _resolve_data_sources(args)

    assert sources.csv_path is None
    assert sources.train_csv_path == str(tmp_path / "aisdk-2026-02-02_cleaned.csv")
    assert sources.validation_csv_path == str(tmp_path / "aisdk-2026-02-03_cleaned.csv")
    assert sources.eval_csv_path == str(tmp_path / "aisdk-2026-02-04_cleaned.csv")
    assert sources.csv_sources == (
        sources.train_csv_path,
        sources.validation_csv_path,
        sources.eval_csv_path,
    )


def test_resolve_data_sources_requires_paired_train_eval() -> None:
    args = argparse.Namespace(
        csv_path=None, train_csv_path="train.csv", validation_csv_path=None, eval_csv_path=None
    )

    with pytest.raises(ValueError, match="supplied together"):
        _resolve_data_sources(args)


def test_resolve_data_sources_rejects_duplicate_explicit_splits(tmp_path) -> None:
    day = tmp_path / "day.csv"
    day.write_text("x\n", encoding="utf-8")
    args = argparse.Namespace(
        csv_path=None,
        train_csv_path=str(day),
        validation_csv_path=None,
        eval_csv_path=str(day),
    )

    with pytest.raises(ValueError, match="must be distinct"):
        _resolve_data_sources(args)


def test_resolve_data_sources_accepts_multiple_train_csvs(tmp_path) -> None:
    train_a = tmp_path / "train_a.csv"
    train_b = tmp_path / "train_b.csv"
    validation = tmp_path / "validation.csv"
    eval_day = tmp_path / "eval.csv"
    for path in (train_a, train_b, validation, eval_day):
        path.write_text("x\n", encoding="utf-8")
    args = argparse.Namespace(
        csv_path=None,
        train_csv_path=f"{train_a},{train_b}",
        validation_csv_path=str(validation),
        eval_csv_path=str(eval_day),
    )

    sources = _resolve_data_sources(args)

    assert sources.train_csv_path == f"{train_a},{train_b}"
    assert sources.selected_cleaned_csv_files == (
        str(train_a),
        str(train_b),
        str(validation),
        str(eval_day),
    )
    assert sources.csv_sources == sources.selected_cleaned_csv_files


def test_resolve_data_sources_accepts_multi_validation_and_eval_csvs(tmp_path) -> None:
    train_a = tmp_path / "train_a.csv"
    train_b = tmp_path / "train_b.csv"
    validation_a = tmp_path / "validation_a.csv"
    validation_b = tmp_path / "validation_b.csv"
    eval_a = tmp_path / "eval_a.csv"
    eval_b = tmp_path / "eval_b.csv"
    for path in (train_a, train_b, validation_a, validation_b, eval_a, eval_b):
        path.write_text("x\n", encoding="utf-8")
    args = argparse.Namespace(
        csv_path=None,
        train_csv_path=f"{train_a},{train_b}",
        validation_csv_path=f"{validation_a},{validation_b}",
        eval_csv_path=f"{eval_a},{eval_b}",
    )

    sources = _resolve_data_sources(args)

    assert sources.train_csv_path == f"{train_a},{train_b}"
    assert sources.validation_csv_path == f"{validation_a},{validation_b}"
    assert sources.eval_csv_path == f"{eval_a},{eval_b}"
    assert sources.selected_cleaned_csv_files == (
        str(train_a),
        str(train_b),
        str(validation_a),
        str(validation_b),
        str(eval_a),
        str(eval_b),
    )
    assert sources.csv_sources == sources.selected_cleaned_csv_files


def test_resolve_data_sources_rejects_duplicate_multi_train_csv(tmp_path) -> None:
    train = tmp_path / "train.csv"
    eval_day = tmp_path / "eval.csv"
    train.write_text("x\n", encoding="utf-8")
    eval_day.write_text("x\n", encoding="utf-8")
    args = argparse.Namespace(
        csv_path=None,
        train_csv_path=f"{train},{train}",
        validation_csv_path=None,
        eval_csv_path=str(eval_day),
    )

    with pytest.raises(ValueError, match="must be distinct"):
        _resolve_data_sources(args)


def test_resolve_data_sources_rejects_duplicate_multi_eval_csv(tmp_path) -> None:
    train = tmp_path / "train.csv"
    eval_day = tmp_path / "eval.csv"
    train.write_text("x\n", encoding="utf-8")
    eval_day.write_text("x\n", encoding="utf-8")
    args = argparse.Namespace(
        csv_path=None,
        train_csv_path=str(train),
        validation_csv_path=None,
        eval_csv_path=f"{eval_day},{eval_day}",
    )

    with pytest.raises(ValueError, match="must be distinct"):
        _resolve_data_sources(args)


def test_benchmark_markdown_table_is_compact() -> None:
    table = _format_report_table(
        [
            {
                "workload": "range",
                "run_label": "custom",
                "returncode": 0,
                "elapsed_seconds": 12.34567,
                "epoch_mean_seconds": 1.25,
                "peak_allocated_mb": 123.0,
                "best_selection_score": 0.5,
                "single_cell_range_status": "fails_uniform",
                "audit_low_beats_uniform_range_usefulness_count": 0,
                "worst_uniform_component_delta_metric": "mlqds_vs_uniform_range_gap_coverage",
                "runtime_bottleneck_phase": "train-model",
                "eval_query_extra_after_target_reached": 100,
                "eval_query_floor_dominated": True,
                "mlqds_primary_metric": "range_usefulness",
                "mlqds_primary_score": 0.41,
                "mlqds_aggregate_f1": 0.4,
                "mlqds_range_point_f1": 0.4,
                "mlqds_range_usefulness": 0.41,
                "uniform_range_point_f1": 0.3,
                "uniform_range_usefulness": 0.32,
                "douglas_peucker_range_point_f1": 0.2,
                "douglas_peucker_range_usefulness": 0.22,
                "mlqds_vs_uniform_range_point_f1": 0.1,
                "mlqds_vs_uniform_range_usefulness": 0.09,
                "mlqds_vs_douglas_peucker_range_point_f1": 0.2,
                "mlqds_vs_douglas_peucker_range_usefulness": 0.19,
                "mlqds_latency_ms": 10.0,
                "mlqds_inference_only_latency_ms": 10.0,
                "collapse_warning": False,
            }
        ]
    )

    assert "| workload | run_label |" in table
    assert "train_label_mass_range_point_f1" in table
    assert "single_cell_range_status" in table
    assert "audit_low_beats_uniform_range_usefulness_count" in table
    assert "runtime_bottleneck_phase" in table
    assert "eval_query_extra_after_target_reached" in table
    assert "eval_query_floor_dominated" in table
    assert "mlqds_avg_sed_km" in table
    assert "mlqds_primary_score" in table
    assert "mlqds_range_usefulness" in table
    assert "mlqds_inference_only_latency_ms" in table
    assert "| range | custom | 0 | 12.3457 |" in table


def test_run_capture_streaming_writes_log_and_console(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    stdout_path = tmp_path / "child" / "stdout.log"

    result = _run_capture_streaming(
        [sys.executable, "-c", "print('alpha', flush=True); print('beta', flush=True)"],
        cwd=tmp_path,
        stdout_path=stdout_path,
    )

    assert result.returncode == 0
    assert result.stdout == "alpha\nbeta\n"
    assert stdout_path.read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert "alpha\nbeta\n" in capsys.readouterr().out


def test_run_capture_streaming_retains_bounded_tail_but_keeps_timings(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    stdout_path = tmp_path / "child" / "stdout.log"
    command = (
        "print('[train-model] done in 1.23s', flush=True)\n"
        "for i in range(20):\n"
        "    print(f'filler-{i:03d}-' + 'x' * 40, flush=True)\n"
    )

    result = _run_capture_streaming(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        stdout_path=stdout_path,
        max_stdout_chars=64,
    )

    full_log = stdout_path.read_text(encoding="utf-8")
    assert result.returncode == 0
    assert result.stdout_truncated is True
    assert len(result.stdout) <= 64
    assert full_log.startswith("[train-model] done in 1.23s\n")
    assert len(full_log) > len(result.stdout)
    assert result.timings["phase_timings"] == [{"name": "train-model", "seconds": 1.23}]
    assert "[train-model] done in 1.23s\n" in capsys.readouterr().out


def test_mark_benchmark_failed_updates_stale_running_status_and_family_index(tmp_path) -> None:
    family = tmp_path / "range_family"
    results_dir = family / "runs" / "stale-run"
    results_dir.mkdir(parents=True)
    status_file = results_dir / "run_status.json"
    status_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "stale-run",
                "status": "running",
                "started_at_utc": "2026-05-10T00:00:00+00:00",
                "finished_at_utc": None,
                "exit_status": None,
                "failures": None,
                "message": "benchmark run started",
                "results_dir": str(results_dir),
            }
        ),
        encoding="utf-8",
    )
    with open(family / "runs_index.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "status",
                "finished_at_utc",
                "exit_status",
                "failures",
                "results_dir",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "stale-run",
                "status": "running",
                "finished_at_utc": "",
                "exit_status": "",
                "failures": "",
                "results_dir": str(results_dir),
            }
        )

    script = Path(__file__).resolve().parents[3] / "scripts" / "mark_benchmark_failed.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--status-file",
            str(status_file),
            "--exit-status",
            "-9",
            "--message",
            "killed",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(status_file.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["exit_status"] == -9
    assert payload["failures"] == 1
    assert payload["message"] == "killed"

    with open(family / "runs_index.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["status"] == "failed"
    assert rows[0]["exit_status"] == "-9"
    assert rows[0]["failures"] == "1"
    assert '"run_id": "stale-run"' in (family / "runs_index_events.jsonl").read_text(
        encoding="utf-8"
    )
