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
| `docs/` | Implementation/research protocol, progress log, and developer tooling guidance. | Generated run reports or duplicated package API docs. |
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

These are the files and directories that most weaken top-down reasoning today.
Line counts are approximate and should be treated as refactor signals, not
automatic defects.

| Path | Current issue | Recommended split |
| --- | --- | --- |
| `tests/unit/orchestration/test_query_driven_implementation.py` | Roughly 6.3k lines. It tests workload profiles, target construction, priors, gates, selector diagnostics, learned-segment behavior, final summaries, and causality in one file. This makes failures hard to localize and encourages new unrelated tests to land in the same place. | Split by owning module and behavior: `test_query_local_utility_targets.py`, `test_query_priors.py`, `test_workload_gates.py`, `test_selector_diagnostics.py`, `test_learned_segment_budget_integration.py`, `test_final_gate_summary.py`, and historical diagnostic reader tests. Move shared fixture builders into small local helpers. |
| `orchestration/selector_diagnostics.py` | Roughly 1.9k lines. It mixes selector trace decoding, score vector construction, marginal alignment summaries, teacher-proxy construction, and retained-decision diagnostics. | Extract pure trace parsing and mask construction first, then retained-marginal alignment summaries, then teacher-proxy vector builders. Keep run-stage wiring in orchestration; do not move query-scoring-dependent diagnostics into `selection/`. |
| `learning/targets/query_local_utility.py` | Roughly 1.7k lines. It owns active target construction plus segment target variants, family evidence, candidate diagnostics, trainability diagnostics, and experimental target helpers. | Keep `build_query_local_utility_targets` as the public entry point. Extract segment target math, family evidence, and diagnostic/candidate helpers into sibling modules such as `query_local_utility_segments.py` and `query_local_utility_diagnostics.py` before considering a package conversion. |
| `scoring/method_scoring.py` | Roughly 1.25k lines. It mixes legacy RangeUseful audit construction, active QueryLocalUtility input components, per-query detail rows, trajectory evidence counts, and method scoring. | Extract range-audit support construction and query-row summarization from method execution. Keep artifact field names stable and cover row payloads with regression tests before moving code. |
| `orchestration/*_diagnostic.py` | Many derived diagnostic scripts live flat beside pipeline stages. The package now contains stage code, CLI entrypoints, gates, payload assembly, and one-off diagnostic analyzers at the same level. | Create `orchestration/diagnostics/` for derived artifact analyzers. Move scripts in small batches and update direct imports instead of leaving permanent compatibility facades. |
| `learning/model_training.py` | Roughly 1.3k lines. It owns fitting, validation cadence, checkpoint-selection orchestration, and selected-output construction. It is large but currently coherent. | Defer this split until the target/selector diagnostics stabilize. If it grows again, split validation/checkpoint-selection driver code from optimizer-loop mechanics. |
| `benchmarking/reporting/row_fields.py` | Roughly 1k lines. Row extraction, schema defaults, diagnostic field flattening, and table compatibility sit together. | Split stable schema/key declarations from extraction helpers. Regression snapshots must be updated deliberately because report field churn is easy to miss. |
| `Range_QDS/artifacts/` | Local generated output is intentionally ignored, but it currently dominates the tree size locally. This can hide source layout problems and make codebase scans noisy. | Keep artifacts out of source imports. Periodically clean smoke/manual output after the relevant metrics are captured. Do not store maintained docs under `artifacts/manual/`. |

## Recommended Refactor Order

1. Split the 6.3k-line query-driven orchestration test file. This has the best
   risk/reward ratio because it improves reviewability without changing
   production behavior.
2. Move derived artifact analyzers into `orchestration/diagnostics/`. This
   clarifies the top-level orchestration package without touching scientific
   semantics.
3. Extract pure helper modules from `learning/targets/query_local_utility.py`.
   Start with segment target math and family evidence; leave the public builder
   in place.
4. Extract retained-decision marginal alignment and trace-mask helpers from
   `orchestration/selector_diagnostics.py`.
5. Split `scoring/method_scoring.py` only after row/regression tests cover the
   exact payload fields that downstream reports depend on.
6. Leave `learning/model_training.py` until the target and selector surfaces are
   calmer. Prematurely splitting the training loop would create churn without
   improving the current blocker diagnosis.

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
