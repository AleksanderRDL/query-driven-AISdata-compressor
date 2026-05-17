# QDS Artifacts

Local generated outputs live under `Range_QDS/artifacts/`. Keep caches,
checkpoints, benchmark logs, and smoke output out of git; only this README
should be tracked.

## Layout

```text
Range_QDS/artifacts/
  benchmarks/
    range_workload_aware_diagnostic/
      latest_run.txt
      latest_queue.txt
      runs_index.csv
      runs_index_events.jsonl
      runs/<run_id>/
      queues/<queue_id>/
  cache/
  results/
```

- `benchmarks/`: comparable benchmark families and queue reports.
- `cache/`: segmented trajectory caches, workload caches, and diagnostics.
- `results/`: smoke and manual single-run output.

Start with the generated run-local `README.md`, `artifact_index.json`,
`run_status.json`, `benchmark_report.md`, and `benchmark_report.csv`.

## Commands

```bash
make benchmark-preflight
make benchmark-queue-preflight
make list-runs
make clean-smoke-artifacts
make clean-smoke-artifacts CONFIRM=1
```

Use descriptive run IDs for comparable runs:

```bash
ATTACH=0 BENCHMARK_RUN_ID=range_workload_aware_diagnostic_seed42_a make range-benchmark-tmux
```

## Cleanup

Safe cleanup targets:

- `Range_QDS/artifacts/results/smoke_*`
- `Range_QDS/artifacts/results/post_training_runtime_smoke`
- `Range_QDS/artifacts/benchmarks/*smoke*`
- `Range_QDS/artifacts/benchmarks/*layout_smoke*`
- smoke-only caches
- stale workload-aware diagnostic caches after their report numbers are captured

Keep benchmark-family runs until their report rows have been reviewed or moved
to an explicit archive.
