# Benchmarking Module

Owns benchmark profiles, benchmark child-run orchestration, queue execution,
family indexes, report rows, tables, runtime benchmark wrappers, and final-grid
acceptance summaries. Single train/eval run wiring belongs in
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
uv run --group dev -- python -m benchmarking.benchmark_runner --help
uv run --group dev -- python -m benchmarking.benchmark_runtime --help
```

Default Make targets use the active `range_workload_v1_workload_blind_v2`
profile and write under `Range_QDS/artifacts/benchmarks/query_driven_workload_blind_v2`
and `Range_QDS/artifacts/cache/query_driven_workload_blind_v2`. Set
`BENCHMARK_PROFILE`, `BENCHMARK_FAMILY`, and `BENCHMARK_CACHE` only for a
diagnostic profile or a separate artifact family.

## Key Files

| File | Purpose |
| --- | --- |
| `benchmark_profiles.py` | Durable benchmark profile defaults. |
| `benchmark_runner.py` | Benchmark child-run and coverage-grid orchestration. |
| `benchmark_runtime.py` | Runtime benchmark wrapper for train/inference timing. |
| `benchmark_inputs.py` | Data-source, workload, and environment resolution. |
| `benchmark_process.py` | Child process execution and timing parsing. |
| `benchmark_artifacts.py` | Family indexes, status files, README, CSV, and JSON writers. |
| `benchmark_report.py` | Benchmark report artifact construction and file output. |
| `reporting/` | Child-run row fields, metric helpers, audit extractors, and report paths. |
| `benchmark_row_runtime.py` | Runtime, phase, epoch, and collapse-warning row helpers. |
| `benchmark_table.py` | Markdown table formatting for benchmark summaries. |
| `benchmark_final_grid.py` | Final-grid QueryUsefulV1 acceptance evidence. |
| `benchmark_common.py` | Shared benchmark numeric/report helpers. |

## Profiles

Current query-driven candidate profile:

- `range_workload_v1_workload_blind_v2`

This is the active QueryUsefulV1 workload-blind path. It is not accepted unless
the guide-required workload stability, support, predictability, causality,
global-sanity, and final-grid gates pass.

Diagnostic profiles still exist in `benchmark_profiles.py` for regression,
teacher, and scalar-target checks. They are not final acceptance evidence.

Keep durable defaults in `benchmark_profiles.py`. Put one-off variations in
queue rows or `BENCHMARK_CHILD_EXTRA_ARGS`, not in profile defaults.

For coverage-grid checks, use:

```bash
uv run --group dev -- python -m benchmarking.benchmark_runner \
  --profile range_workload_v1_workload_blind_v2 \
  --workloads range \
  --coverage_targets 0.05,0.10,0.15,0.30
```

Do not also pass a conflicting `--query_coverage` through child extra args for
the same benchmark.

## Artifacts

Start with:

- run-local `README.md`
- `run_status.json`
- `benchmark_report.md`
- `benchmark_report.csv`
- `benchmark_report.json`
- child `example_run.json`

Artifact layout and cleanup rules live in
[`../artifacts/README.md`](../artifacts/README.md). JSON inspection helpers
live in [`../docs/dev-tooling-guide.md`](../docs/dev-tooling-guide.md).
