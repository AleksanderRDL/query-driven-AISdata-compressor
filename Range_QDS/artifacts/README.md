# QDS Artifacts

Local generated outputs live under `Range_QDS/artifacts/`. Keep caches,
checkpoints, benchmark logs, and smoke output out of git; only this README
should be tracked.

The repository should not retain every historical run. Keep only artifacts that
are part of the current evidence boundary, a documented active diagnostic, or a
still-useful negative lesson. Once the progress log captures the relevant
numbers and decision, prune stale raw outputs instead of letting old runs look
like current evidence.

## Layout

```text
Range_QDS/artifacts/
  benchmarks/
    query_driven_workload_blind_v2/
      latest_run.txt
      latest_queue.txt
      runs_index.csv
      runs_index_events.jsonl
      runs/<run_id>/
      queues/<queue_id>/
  cache/
    query_driven_workload_blind_v2/
  manual/
  results/
```

- `benchmarks/`: comparable benchmark families and queue reports.
- `cache/`: segmented trajectory caches, workload caches, and diagnostics.
- `manual/`: local generated historical/manual reports. Treat markdown here as
  run output, not maintained source documentation.
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
ATTACH=0 BENCHMARK_RUN_ID=query_driven_v2_seed42_a make range-benchmark-tmux
```

## Cleanup

Safe cleanup targets:

- result directories matching smoke-only run names
- obsolete one-off post-training runtime smoke directories
- benchmark directories matching smoke or layout-smoke names
- smoke-only caches
- stale diagnostic caches after their report numbers are captured

Keep benchmark-family runs until their report rows have been reviewed or moved
to an explicit archive. Keep `cache/` and `manual/` as disposable generated
state unless a current guide or progress entry explicitly depends on a specific
file there.
