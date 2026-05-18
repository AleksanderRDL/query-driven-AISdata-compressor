# Benchmarking Module

Owns benchmark profiles, benchmark child-run orchestration, queue execution,
family indexes, report rows, tables, runtime benchmark wrappers, and final-grid
acceptance summaries. Single training/scoring run wiring belongs in
`../orchestration/`.

From `Range_QDS/`, use the Make targets for routine benchmark work:

```bash
make benchmark-preflight
ATTACH=0 make range-benchmark-tmux
ATTACH=0 BENCHMARK_SEEDS=42,43,44 make range-benchmark-queue-tmux
make list-runs
```

Direct module entry points:

```bash
uv run --group dev -- python -m benchmarking.runner --help
uv run --group dev -- python -m benchmarking.runtime_benchmark --help
```

Default Make targets use the active `range_workload_v1_workload_blind_v2`
profile and write under `Range_QDS/artifacts/benchmarks/query_driven_workload_blind_v2`
and `Range_QDS/artifacts/cache/query_driven_workload_blind_v2`. Set
`BENCHMARK_PROFILE`, `BENCHMARK_FAMILY`, and `BENCHMARK_CACHE` only for a
diagnostic profile or a separate artifact family.

## Key Files

| File | Purpose |
| --- | --- |
| `profiles.py` | Durable benchmark profile defaults. |
| `runner.py` | Benchmark child-run and workload-profile-grid orchestration. |
| `runtime_benchmark.py` | Runtime benchmark wrapper for train/inference timing. |
| `inputs.py` | Data-source, workload, and environment resolution. |
| `child_process.py` | Child process execution and timing parsing. |
| `artifacts.py` | Family indexes, status files, README, CSV, and JSON writers. |
| `report.py` | Benchmark report artifact construction and file output. |
| `reporting/` | Child-run row fields, metric helpers, audit extractors, and report paths. |
| `row_runtime.py` | Runtime, phase, epoch, and collapse-warning row helpers. |
| `table.py` | Markdown table formatting for benchmark summaries. |
| `final_grid.py` | Final-grid QueryUsefulV1 acceptance evidence. |
| `common.py` | Shared benchmark numeric/report helpers. |

## Profiles

Current query-driven candidate profile:

- `range_workload_v1_workload_blind_v2`

This is the active QueryUsefulV1 workload-blind path. It is not accepted unless
the guide-required workload stability, support, predictability, causality,
global-sanity, and final-grid gates pass.

Diagnostic profiles still exist in `profiles.py` for regression,
teacher, and scalar-target checks. They are not final acceptance evidence.

Keep durable defaults in `profiles.py`. Put one-off variations in
queue rows or `BENCHMARK_CHILD_EXTRA_ARGS`, not in profile defaults.

For workload-profile grid checks, use:

```bash
uv run --group dev -- python -m benchmarking.runner \
  --profile range_workload_v1_workload_blind_v2 \
  --workloads range \
  --workload_profile_ids range_workload_v1_focused,range_workload_v1_local,range_workload_v1_operational,range_workload_v1
```

Do not also pass conflicting `--workload_profile_id`, `--query_coverage`,
`--range_max_coverage_overshoot`, or `--coverage_calibration_mode` through child
extra args for the same benchmark.

## Artifacts

Start with:

- run-local `README.md`
- `run_status.json`
- `benchmark_report.md`
- `benchmark_report.csv`
- `benchmark_report.json`
- child `example_run.json`

Benchmark report rows expose `mlqds_inference_only_latency_ms` and
`mlqds_inference_only_latency_seconds`. These are copied from child
`matched.MLQDS.latency_ms`, which times retained-mask application only. Do not
interpret them as `evaluate_matched_seconds`; that phase includes matched-method
scoring work.

Artifact layout and cleanup rules live in
[`../artifacts/README.md`](../artifacts/README.md). JSON inspection helpers
live in [`../docs/dev-tooling-guide.md`](../docs/dev-tooling-guide.md).
