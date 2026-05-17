# QDS Code Layout

This is the top-down map for the active `Range_QDS` codebase. It is not an
exhaustive file index; each package README carries local details.

## Main Flow

```text
data -> queries -> training -> simplification -> evaluation -> benchmarking
```

Single-run entry point:

```bash
uv run --group dev -- python -m orchestration.run_ais_experiment
```

Benchmarking entry point:

```bash
uv run --group dev -- python -m benchmarking.benchmark_runner
```

Operational tooling reference: `docs/dev-tooling-guide.md`.

## Package Responsibilities

| Path | Owns | Should not own |
| --- | --- | --- |
| `config/` | Shared config dataclasses, flat config builder, and deterministic seed derivation. | CLI parsing, orchestration, runtime mutation, or training logic. |
| `runtime/` | Process-local runtime controls shared by entrypoints and training, such as torch precision and AMP helpers. | Config, benchmark policy, model training, or artifact writing. |
| `data/` | AIS loading, segmentation, source/day combination, trajectory caches, flattened boundaries. | Query workload policy, model behavior, benchmark gates. |
| `queries/` | Typed query data, range geometry, query execution, workload diagnostics, and query-generation subpackages. | Training labels, model scoring, retained-mask selection. |
| `training/` | Feature builders, target builders, priors, losses, batching, checkpoint persistence, inference helpers. | Orchestration, benchmarking, reporting, final-claim gates. |
| `simplification/` | Query-free score-to-mask selectors and selector diagnostics. | Query generation, model training, benchmark reporting. |
| `evaluation/` | Metrics, baseline methods, query caches, range/query-useful scoring, printable evaluation tables. | Training target construction or command assembly. |
| `orchestration/` | Single-run CLI parsing, data/workload assembly, pipeline wiring, artifact writing, and run-level diagnostics/gates. | Benchmark campaign policy, final-grid summaries, low-level model/query/selector primitives. |
| `benchmarking/` | Benchmark profiles, benchmark runners, queues, reports, runtime benchmarks, family indexes, and final-grid summaries. | Single-run train/eval internals or low-level model/query/selector primitives. |
| `scripts/` | Small operational tools over existing artifacts or profiles. | Scientific logic not already owned by packages above. |
| `tests/` | Unit, integration, property, regression, and guardrail tests. | Production helpers used only to make tests pass. |

## Subpackage Layout

| Path | Owns |
| --- | --- |
| `queries/generation/` | Query workload generation, workload profiles, anchor policy, coverage guards, and signatures. |
| `training/targets/` | Target mode registries, active QueryUsefulV1 labels, and legacy/scalar RangeUseful target builders. |
| `simplification/learned_segment_budget/` | Learned segment-budget selector public API and implementation. |
| `tests/unit/<component>/` | Component-scoped tests for data, queries, training, simplification, evaluation, orchestration, benchmarking, and runtime. |
| `tests/integration/` | Cross-stage behavior tests. |
| `tests/guardrails/` | Protocol and cleanup guardrails. |
| `tests/property/` | Hypothesis/property tests. |
| `tests/regression/` | Stable report/schema regression tests. |

## Current Pressure Points

These are the files that most weaken top-down reasoning today. Line counts are
approximate and should be treated as refactor signals, not automatic defects.

| File | Current issue | Recommended split |
| --- | --- | --- |
| `orchestration/experiment_pipeline.py` | Still mixes orchestration, phase timing, selection-causality ablation freezing, artifact assembly, and run output. | Keep `run_experiment_pipeline` as the orchestrator; narrow `_selection_causality_diagnostics` before moving more logic. |
| `training/targets/legacy.py` | Legacy RangeUseful/scalar target builders, set-utility diagnostics, local-swap targets, and aggregation paths still live together. | Split legacy/scalar diagnostics, set-utility targets, local-swap targets, and aggregation/balancing with focused target-family tests. |
| `benchmarking/benchmark_report.py` | Child-run row flattening and audit extraction are still interleaved. | Extract narrow row-field builder clusters while preserving report field regression coverage. |
| `simplification/learned_segment_budget/core.py` | Budget allocation, length repair, trace payloads, and diagnostics are still tightly packed. | Split allocation, length repair, diagnostics, and trace payload construction inside the package once selector behavior is stable. |
| `queries/generation/workload.py` | Anchor weighting, profile planning, acceptance filtering, signature generation, and workload assembly share one module. | Split anchor sampling/profile planning from acceptance/signature diagnostics. |

## Refactor Rules

- Preserve public commands and artifact field names unless the checkpoint
  explicitly says it is changing them.
- Move behavior only with focused tests around the moved boundary. Do not do
  broad file splits during scientific probe checkpoints.
- Prefer extraction of pure helpers first: row-field builders, signature
  builders, allocation diagnostics.
- Avoid permanent compatibility shims. If a temporary facade is needed during a
  split, mark its removal checkpoint in the progress log.
- Keep final-claim gates close to their tests. Benchmarking final-grid code
  should remain importable without importing the single-run pipeline.
