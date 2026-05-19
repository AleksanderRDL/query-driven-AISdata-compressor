# AIS-QDS

AIS-QDS trains and scores query-driven AIS trajectory compressors. The active
target is workload-blind range compression: choose retained points before future
eval queries are known, then score the frozen retained set.

Current source of truth:

- redesign protocol and gates: [`docs/query-driven-rework-guide.md`](docs/query-driven-rework-guide.md)
- checkpoint log: [`docs/query-driven-rework-progress.md`](docs/query-driven-rework-progress.md)
- tooling commands: [`docs/dev-tooling-guide.md`](docs/dev-tooling-guide.md)

## Setup

Run commands from `Range_QDS/`. Make delegates Python commands through root
`uv --group dev`.

```bash
cd Range_QDS
make sync
make lock-check
make check-env
make test
```

## Common Commands

```bash
make typecheck
make lint
make lint-full
make lint-yaml
make smoke
make smoke-csv CLEANED_CSV=../AISDATA/cleaned/<file-or-directory>
make benchmark-preflight
ATTACH=0 make range-benchmark-tmux
ATTACH=0 BENCHMARK_SEEDS=42,43,44 make range-benchmark-queue-tmux
make list-runs
make clean-smoke-artifacts CONFIRM=1
```

Direct CLI example:

```bash
cd ..
uv run --group dev -- python -m orchestration.train_and_score \
  --csv_path AISDATA/cleaned/<cleaned-ais-file.csv> \
  --cache_dir Range_QDS/artifacts/cache/manual_csv \
  --workload range \
  --workload_profile_id range_query_mix \
  --coverage_calibration_mode profile_sampled_query_count \
  --model_type workload_blind_range_v2 \
  --range_training_target_mode query_local_utility_factorized \
  --selector_type learned_segment_budget_v1 \
  --checkpoint_score_variant query_local_utility \
  --checkpoint_selection_metric uniform_gap \
  --n_queries 128 \
  --epochs 6 \
  --compression_ratio 0.10 \
  --mlqds_temporal_fraction 0.0 \
  --final_metrics_mode diagnostic \
  --results_dir Range_QDS/artifacts/results/manual_range
```

## Where To Look

| Need | File |
| --- | --- |
| Redesign objective and acceptance criteria | [`docs/query-driven-rework-guide.md`](docs/query-driven-rework-guide.md) |
| Code layout | [`CODE_LAYOUT.md`](CODE_LAYOUT.md) |
| Single-run orchestration | [`orchestration/README.md`](orchestration/README.md) |
| Benchmark profiles, queues, reports, artifact names | [`benchmarking/README.md`](benchmarking/README.md) |
| Artifact layout and cleanup | [`artifacts/README.md`](artifacts/README.md) |
| Training labels, loss, checkpoint selection | [`learning/README.md`](learning/README.md) |
| Workload generation and query execution | [`workloads/README.md`](workloads/README.md) |
| Retained-mask selection and selector diagnostics | [`selection/README.md`](selection/README.md) |
| Scoring methods and metrics | [`scoring/README.md`](scoring/README.md) |
| Data preparation, loading, and segmented cache | [`data_preparation/README.md`](data_preparation/README.md) |
| Model architecture | [`models/README.md`](models/README.md) |
| Developer tooling | [`docs/dev-tooling-guide.md`](docs/dev-tooling-guide.md) |

## Requirements

Dependency source of truth lives in root [`../pyproject.toml`](../pyproject.toml).
Use `uv sync --group dev` from the repo root.

## Output Policy

Run, cache, and benchmark output should stay under `Range_QDS/artifacts/`
unless a run explicitly needs another local path. Source data belongs under
`../AISDATA/`; model outputs do not.
