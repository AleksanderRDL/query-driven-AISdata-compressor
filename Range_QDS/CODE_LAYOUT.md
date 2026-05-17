# QDS Code Layout

This is the top-down map for the active `Range_QDS` codebase. It is not an
exhaustive file index; each package README carries local details.

## Main Flow

```text
data -> queries -> training -> simplification -> evaluation -> experiments/reports
```

Main entry point:

```bash
uv run --group dev -- python -m experiments.run_ais_experiment
```

Operational tooling reference: `docs/dev-tooling-guide.md`.

## Package Responsibilities

| Path | Owns | Should not own |
| --- | --- | --- |
| `config/` | Shared experiment/config dataclasses, flat config builder, and deterministic seed derivation. | CLI parsing, orchestration, runtime mutation, or training logic. |
| `data/` | AIS loading, segmentation, source/day combination, trajectory caches, flattened boundaries. | Query workload policy, model behavior, benchmark gates. |
| `queries/` | Typed query data, range geometry, workload profiles, query generation, query execution, workload diagnostics. | Training labels, model scoring, retained-mask selection. |
| `models/` | PyTorch model definitions and model-local utilities. | Training loops, checkpoint policy, experiment metrics. |
| `training/` | Feature builders, target builders, priors, losses, batching, checkpoint persistence, inference helpers. | Experiment orchestration, benchmark reporting, final-claim gates. |
| `simplification/` | Query-free score-to-mask selectors and selector diagnostics. | Query generation, model training, benchmark reporting. |
| `evaluation/` | Metrics, baseline methods, query caches, range/query-useful scoring, printable evaluation tables. | Training target construction or experiment command assembly. |
| `runtime/` | Process-local runtime controls shared by entrypoints and training, such as torch precision and AMP helpers. | Experiment config, benchmark policy, model training, or artifact writing. |
| `experiments/` | Config, CLI parsing, workload assembly, pipeline orchestration, artifact writing, benchmark profiles, reports. | Low-level model/selector/query/runtime primitives that need to be reused without orchestration. |
| `scripts/` | Small operational tools over existing artifacts or profiles. | Scientific logic not already owned by packages above. |
| `tests/` | Guardrails, regression tests, property tests, and focused unit/integration coverage. | Production helpers used only to make tests pass. |

The earlier `training -> experiments` config/runtime dependency has been
removed. Shared config lives in `config/`; shared torch runtime controls live
in `runtime/`.

## Current Pressure Points

These are the files that most weaken top-down reasoning today. Line counts are
approximate and should be treated as refactor signals, not automatic defects.

| File | Current issue | Recommended split |
| --- | --- | --- |
| `experiments/experiment_pipeline.py` (~3.0k lines after gate, causality-helper, segment-audit, length-diagnostic, selector-diagnostic, and model-ablation extraction) | Still mixes orchestration, phase timing, selection-causality ablation freezing, artifact assembly, and run output. | Do not split further unless `_selection_causality_diagnostics` is first narrowed enough to move without carrying evaluation orchestration into another module. Leave `run_experiment_pipeline` as the orchestrator. |
| `training/training_targets.py` (~3k lines after target-mode registry extraction) | Legacy RangeUseful/scalar target builders, set-utility diagnostics, local-swap targets, and aggregation paths still live together. Public mode registries now live in `training/target_modes.py`; active QueryUsefulV1 labels live in `training/query_useful_targets.py`. | Split legacy/scalar diagnostics, set-utility targets, local-swap targets, and aggregation/balancing only with focused target-family tests. |
| `experiments/benchmark_report.py` (~1.6k lines after table-format, final-grid, and runtime-row extraction) | Child-run row flattening and audit extraction are still interleaved. Shared numeric helpers live in `experiments/benchmark_common.py`; final-grid acceptance lives in `experiments/benchmark_final_grid.py`; runtime/history row helpers live in `experiments/benchmark_row_runtime.py`; table formatting lives in `experiments/benchmark_table.py`. | Extract narrow row-field builder clusters only after preserving report field regression coverage. Do not move `_row_from_run` as one broad block. |
| `simplification/learned_segment_budget.py` (~1.5k lines) | Budget allocation, length repair, trace payloads, and diagnostics are tightly packed. | Split allocation, length repair, and diagnostics/trace once selector behavior is stable. |
| `queries/query_generator.py` (~1.3k lines) | Anchor weighting, profile planning, acceptance filtering, signature generation, and workload assembly share one module. | Split range anchor sampling/profile planning from acceptance/signature diagnostics. |

## Refactor Rules

- Preserve the public experiment commands and artifact field names unless the
  checkpoint explicitly says it is changing them.
- Move behavior only with focused tests around the moved boundary. Do not do
  broad file splits during scientific probe checkpoints.
- Prefer extraction of pure helpers first: row-field builders, signature
  builders, allocation diagnostics.
- Avoid permanent compatibility shims. If a temporary facade is needed during a
  split, mark its removal checkpoint in the progress log.
- Keep final-claim gates close to their tests. The gate code should eventually
  be importable without importing the full experiment pipeline.
