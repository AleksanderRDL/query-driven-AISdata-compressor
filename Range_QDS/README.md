# AIS-QDS

AIS-QDS trains and scores query-driven AIS trajectory compressors. The active
target is workload-blind range compression: choose retained points before future
eval queries are known, then score the frozen retained set.

Current source of truth:

- implementation/research protocol and gates: [`docs/query-driven-implementation-research-guide.md`](docs/query-driven-implementation-research-guide.md)
- checkpoint log: [`docs/query-driven-implementation-progress.md`](docs/query-driven-implementation-progress.md)
- tooling commands: [`docs/dev-tooling-guide.md`](docs/dev-tooling-guide.md)

## Current Defaults

New query-driven checkpoints should use this stack unless a diagnostic
explicitly overrides it:

| Surface | Default |
| --- | --- |
| Primary metric | `QueryLocalUtility` |
| Workload profile | `range_query_mix` |
| Target mode | `query_local_utility_factorized` |
| Model | `workload_blind_range` |
| Selector | `learned_segment_budget` |
| Checkpoint score variant | `query_local_utility` |

`QueryLocalUtility` weights direct query-point mass at `0.50`, query-local
behavior at `0.45`, and global sanity guardrails at `0.05`.
Global sanity is reported and should improve, but it is not an initial hard
blocker while the project is still proving local query behavior and learning
causality.
The active `range_query_mix` profile uses anchor weights `density=0.80` and
`sparse_background_control=0.20`; footprint weights are
`medium_operational=0.6923076923076923` and
`large_context=0.3076923076923077`. The active footprint acceptance bands are
point-hit fraction `[0.006,0.030]` for `medium_operational` and
`[0.010,0.045]` for `large_context`. Proposals target a deterministic
low-band point-hit fraction before unchanged acceptance gates run.

Older metric/profile names and removed workload families are historical only.
Do not use them for new checkpoints unless a checkpoint explicitly reintroduces
one with evidence.

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
  --model_type workload_blind_range \
  --range_training_target_mode query_local_utility_factorized \
  --selector_type learned_segment_budget \
  --checkpoint_score_variant query_local_utility \
  --checkpoint_selection_metric uniform_gap \
  --n_queries 128 \
  --epochs 6 \
  --compression_ratio 0.10 \
  --mlqds_temporal_fraction 0.0 \
  --final_metrics_mode diagnostic \
  --results_dir Range_QDS/artifacts/results/<manual-run-id>
```

## Where To Look

| Need | File |
| --- | --- |
| Implementation objective and acceptance criteria | [`docs/query-driven-implementation-research-guide.md`](docs/query-driven-implementation-research-guide.md) |
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
Use the root `uv`-managed virtual environment with CPython `3.14.5`, pinned by
[`../.python-version`](../.python-version). Run `uv sync --python 3.14.5 --group dev`
from the repo root.

## Output Policy

Run, cache, and benchmark output should stay under `Range_QDS/artifacts/`
unless a run explicitly needs another local path. Source data belongs under
`../AISDATA/`; model outputs do not.
