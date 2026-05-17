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
| `benchmarking/reporting/` | Benchmark row construction, metric helpers, audit extraction, and report path helpers. |
| `training/targets/` | Target mode registries, active QueryUsefulV1 labels, shared scalar helpers, retained-frequency targets, structural/marginal targets, query-spine/residual targets, set-utility/local-swap targets, and aggregation. |
| `simplification/learned_segment_budget/` | Learned segment-budget selector orchestration, allocation, length repair, diagnostics, and trace construction. |
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
| `orchestration/experiment_pipeline.py` | Still coordinates model training, evaluation, artifact assembly, and run output. It is no longer the owner of target prep, retained-mask freezing, selection causality, or final summary assembly. | Keep `run_experiment_pipeline` as the orchestrator; next extract matched-evaluation/audit evaluation prep only if focused tests preserve metric payloads and artifact field names. |
| `orchestration/target_preparation.py` | Owns several target families plus teacher-distillation runtime in one large module. This is cleaner than keeping it in the pipeline, but it is not small. | Split by target-family dispatch only after behavior is stable; avoid moving target builders back into orchestration. |
| `orchestration/retained_masks.py` | Owns primary/audit freeze ordering, score-cache capture, and selector-trace capture. | Keep it focused on protocol ordering; move only if primary/audit freeze mechanics grow beyond the current boundary. |
| `orchestration/retained_mask_ablations.py` | Owns query-free causality ablation mask construction and freeze diagnostics. It still repeats MLQDS method construction arguments. | Add a small local MLQDS diagnostic-method factory before adding more ablation variants. |

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
