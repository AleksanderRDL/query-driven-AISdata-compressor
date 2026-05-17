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

The benchmark Makefile defaults use the active
`range_workload_v1_workload_blind_v2` profile and the
`query_driven_workload_blind_v2` artifact/cache families. Set
`BENCHMARK_PROFILE`, `BENCHMARK_FAMILY`, and `BENCHMARK_CACHE` explicitly only
for diagnostic profiles or separate artifact families.

## Key Files

| File | Purpose |
| --- | --- |
| `benchmark_profiles.py` | Durable benchmark profile defaults. |
| `benchmark_runner.py` | Benchmark child-run and coverage-grid orchestration. |
| `benchmark_runtime.py` | Runtime benchmark wrapper for train/inference timing. |
| `benchmark_inputs.py` | Data-source, workload, and environment resolution. |
| `benchmark_process.py` | Child process execution and timing parsing. |
| `benchmark_artifacts.py` | Family indexes, status files, README, CSV, and JSON writers. |
| `benchmark_report.py` | Benchmark child-run row shaping and audit-field flattening. |
| `benchmark_row_runtime.py` | Runtime, phase, epoch, and collapse-warning row helpers. |
| `benchmark_table.py` | Markdown table formatting for benchmark summaries. |
| `benchmark_final_grid.py` | Final-grid QueryUsefulV1 acceptance evidence. |
| `benchmark_common.py` | Shared benchmark numeric/report helpers. |

## Profiles

Current query-driven candidate profile:

- `range_workload_v1_workload_blind_v2`

This is the active QueryUsefulV1 workload-blind path. It is still blocked
unless the guide-required workload stability, support, predictability,
causality, global-sanity, and final-grid gates pass.

Legacy diagnostic profiles remain useful for regression and teacher checks:

- `range_workload_aware_diagnostic`
- `range_workload_blind_expected_usefulness`
- `range_workload_blind_retained_frequency`
- `range_workload_blind_teacher_distill`

Legacy profiles report diagnostic RangeUseful/scalar behavior and are not final
acceptance evidence.

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
[`../artifacts/README.md`](../artifacts/README.md). Use jq/Rich helpers from
[`../docs/dev-tooling-guide.md`](../docs/dev-tooling-guide.md) for one-run
inspection.
