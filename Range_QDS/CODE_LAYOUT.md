# QDS Code Layout

This is the top-down map for the active `Range_QDS` codebase. It is not an
exhaustive file index; each package README carries local details.

## Main Flow

```text
data preparation -> workloads -> learning -> selection -> scoring -> benchmarking
```

Single-run entry point:

```bash
uv run --group dev -- python -m orchestration.train_and_score
```

Benchmarking entry point:

```bash
uv run --group dev -- python -m benchmarking.runner
```

Operational tooling reference: `docs/dev-tooling-guide.md`.

## Package Responsibilities

| Path | Owns | Should not own |
| --- | --- | --- |
| `config/` | Shared config dataclasses, flat config builder, and deterministic seed derivation. | CLI parsing, orchestration, runtime mutation, or learning logic. |
| `runtime/` | Process-local runtime controls shared by entrypoints and learning, such as torch precision and AMP helpers. | Config, benchmark policy, model learning, or artifact writing. |
| `data_preparation/` | AIS loading, segmentation, source/day combination, trajectory caches, flattened boundaries. | Query workload policy, model behavior, benchmark gates. |
| `workloads/` | Typed query data, range geometry, query execution, workload diagnostics, and query-generation subpackages. | Learning labels, model scoring, retained-mask selection. |
| `learning/` | Feature builders, target builders, priors, losses, batching, checkpoint persistence, inference helpers. | Orchestration, benchmarking, reporting, final-claim gates. |
| `selection/` | Query-free score-to-mask selectors and selector diagnostics. | Query generation, model learning, benchmark reporting. |
| `scoring/` | Method wrappers, metrics, query caches, active QueryLocalUtility scoring, and printable scoring tables. | Learning target construction or command assembly. |
| `orchestration/` | Single-run CLI parsing, data-preparation/workload assembly, pipeline wiring, artifact writing, and run-level diagnostics/gates. | Benchmark campaign policy, final-grid summaries, low-level model/query/selector primitives. |
| `benchmarking/` | Benchmark profiles, benchmark runners, queues, reports, runtime benchmarks, family indexes, and final-grid summaries. | Single-run train/eval internals or low-level model/query/selector primitives. |
| `scripts/` | Small operational tools over existing artifacts or profiles. | Scientific logic not already owned by packages above. |
| `docs/` | Rework protocol, progress log, and developer tooling guidance. | Generated run reports or duplicated package API docs. |
| `artifacts/` | Local generated caches, run outputs, benchmark families, and generated report markdown. | Maintained source documentation or importable code. |
| `tests/` | Unit, integration, property, regression, and guardrail tests. | Production helpers used only to make tests pass. |

## Subpackage Layout

| Path | Owns |
| --- | --- |
| `workloads/generation/` | Query workload generation, workload profiles, anchor policy, coverage guards, and signatures. |
| `benchmarking/reporting/` | Benchmark row construction, metric helpers, audit extraction, and report path helpers. |
| `learning/targets/` | Target mode registries, active QueryLocalUtility labels, shared scalar helpers, retained-frequency targets, structural/marginal targets, query-spine/residual targets, set-utility/local-swap targets, and aggregation. |
| `selection/learned_segment_budget/` | Learned segment-budget selector orchestration, allocation, length repair, diagnostics, and trace construction. |
| `tests/unit/<component>/` | Component-scoped tests for data preparation, workloads, learning, selection, scoring, orchestration, benchmarking, and runtime. |
| `tests/integration/` | Cross-stage behavior tests. |
| `tests/guardrails/` | Protocol and cleanup guardrails. |
| `tests/property/` | Hypothesis/property tests. |
| `tests/regression/` | Stable report/schema regression tests. |

## Current Pressure Points

These are the files that most weaken top-down reasoning today. Line counts are
approximate and should be treated as refactor signals, not automatic defects.

| File | Current issue | Recommended split |
| --- | --- | --- |
| `orchestration/learning_scoring_pipeline.py` | Coordinates end-to-end stage order after target prep, retained-mask freezing, scoring mechanics, final summaries, payload assembly, and optional exports have direct owners. | Keep this file as the stage orchestrator. Extract only if a stage block grows a new independent responsibility. |
| `orchestration/scoring_stage.py` | Owns matched scoring, ablation scoring, learned-fill diagnostics, compression audit scoring, and shift scoring for a single run. | Keep it limited to scoring-stage mechanics; do not move final summary gates or artifact writing here. |
| `orchestration/learning_target_stage.py` | Owns several target families plus teacher-distillation runtime in one large module. This is cleaner than keeping it in the pipeline, but it is not small. | Split by target-family dispatch only after behavior is stable; avoid moving target builders back into orchestration. |
| `orchestration/retained_mask_stage.py` | Owns primary/audit freeze ordering, score-cache capture, and selector-trace capture. | Keep it focused on protocol ordering; move only if primary/audit freeze mechanics grow beyond the current boundary. |
| `orchestration/retained_mask_ablation_stage.py` | Owns query-free causality ablation mask construction and freeze diagnostics. MLQDS construction now delegates to `orchestration/mlqds_method_factory.py`. | Split ablation families only if the stage grows more variants or starts mixing artifact/report assembly into freeze logic. |
| `learning/model_training.py` | Owns the fitting loop, validation cadence, and checkpoint-selection orchestration after factorized-head diagnostics, target diagnostics, model construction, and checkpoint scoring helpers moved to focused learning modules. | Split only if the optimizer loop or validation/checkpoint handoff grows a new independent responsibility; preserve selected-score behavior with focused regression. |

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
