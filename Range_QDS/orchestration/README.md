# Orchestration Module

Owns single-run CLI parsing, train/eval pipeline wiring, workload assembly,
runtime metadata emission, and artifact writing. Shared config dataclasses live
in `../config/`; shared torch runtime controls live in `../runtime/`.
Benchmark campaign machinery lives in `../benchmarking/`.

Run CLI entry points from the repository root:

```bash
uv run --group dev -- python -m orchestration.run_ais_experiment --help
uv run --group dev -- python -m orchestration.run_inference --help
```

From `Range_QDS/`, use `make smoke` and `make smoke-csv` for tiny
implementation checks.

## Key Files

| File | Purpose |
| --- | --- |
| `experiment_cli.py` | CLI flags over shared config dataclasses. |
| `experiment_pipeline.py` | End-to-end single-run train/eval orchestration. |
| `experiment_data.py` | Train, validation, selection, and eval data splits. |
| `experiment_workloads.py` | Workload generation and workload-map resolution. |
| `experiment_methods.py` | Evaluation method construction. |
| `experiment_outputs.py` | Run artifact payloads and writers. |
| `target_preparation.py` | Training-label preparation, target transforms, teacher distillation, and validation query caches. |
| `retained_masks.py` | Workload-blind primary/audit retained-mask freezing and selector-trace capture. |
| `retained_mask_ablations.py` | Query-free retained-mask ablation construction and freeze diagnostics. |
| `final_summary.py` | Final single-cell gate, final-claim, and causality summary assembly. |
| `range_cache.py` / `workload_cache.py` | Run-local range label and workload caches. |
| `range_diagnostics.py` | Range workload, learned-fill, and gate diagnostics. |
| `gates.py` | Single-run final-candidate gate helpers. |
| `causality.py` / `model_ablations.py` / `selection_causality.py` | Learning-causality and ablation diagnostics. |
| `segment_audits.py` / `length_diagnostics.py` / `selector_diagnostics.py` | Selector and geometry diagnostic helpers. |
| `run_ais_experiment.py` | Main training/evaluation entry point. |
| `run_inference.py` | Evaluate a saved checkpoint without retraining. |

## Data Modes

- `--csv_path`: one cleaned CSV file or a directory split internally.
- `--train_csv_path`, `--validation_csv_path`, `--eval_csv_path`: explicit
  split sources. Comma-separated lists support multi-day splits.
- no CSV path: deterministic synthetic data.

CSV loading segments MMSI tracks by `--max_time_gap_seconds` and can cache
post-segmentation tensors with `--cache_dir`. `--max_segments`,
`--max_points_per_segment`, and split-specific caps are runtime controls; do
not treat tiny capped probes as scientific evidence.

## Workload-Blind Rule

Final or diagnostic workload-blind claims require retained masks to be chosen
before held-out eval queries are scored. Treat a run as invalid if
`workload_blind_protocol.primary_masks_frozen_before_eval_query_scoring` or
`workload_blind_protocol.audit_masks_frozen_before_eval_query_scoring` is
false.

QueryUsefulV1 is the active primary metric for the rework. RangeUseful outputs
must remain under `legacy_range_useful_summary` or diagnostic fields, not
`final_claim_summary`.
