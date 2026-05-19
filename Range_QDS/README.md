# AIS-QDS

AIS-QDS trains and scores query-driven AIS trajectory compressors. The active
target is workload-blind range compression: choose retained points before future
eval queries are known, then score the frozen retained set.

Current source of truth:

- redesign protocol and gates: [`docs/query-driven-rework-guide.md`](docs/query-driven-rework-guide.md)
- checkpoint log: [`docs/query-driven-rework-progress.md`](docs/query-driven-rework-progress.md)
- tooling commands: [`docs/dev-tooling-guide.md`](docs/dev-tooling-guide.md)

## Current Defaults

New query-driven checkpoints should use this stack unless a diagnostic
explicitly overrides it:

| Surface | Default |
| --- | --- |
| Primary metric | `QueryLocalUtility` schema `5` |
| Workload profile | `range_query_mix` |
| Target mode | `query_local_utility_factorized` |
| Model | `workload_blind_range_v2` |
| Selector | `learned_segment_budget_v1` |
| Checkpoint score variant | `query_local_utility` |

`QueryLocalUtility` schema `5` weights direct query-point mass at `0.50`,
query-local behavior at `0.45`, and global sanity guardrails at `0.05`.
The active `range_query_mix` profile uses anchor weights `density=0.80` and
`sparse_background_control=0.20`; footprint weights are
`medium_operational=0.6923076923076923` and
`large_context=0.3076923076923077`.

Historical names and families such as `QueryUsefulV1`, `query_useful_v1`,
`range_workload_v1`, `density_route`, `small_local`,
`boundary_entry_exit`, `crossing_turn_change`, `port_or_approach_zone`, and
`route_corridor_like` are not current defaults.

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
