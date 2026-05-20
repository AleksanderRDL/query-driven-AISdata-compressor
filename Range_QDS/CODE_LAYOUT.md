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
| Query-driven implementation tests | The old 6.3k-line orchestration test file has been split by owner, but several new files are still large enough to need discipline: derived diagnostics, learned-segment selection, target construction, causality/final summary, and workload gates. | Keep adding new tests to the owning split file. If any split file grows beyond roughly 1.5k lines, extract fixture builders or split by the production module under test. |
| `orchestration/selector_diagnostics.py` | Roughly 0.9k lines after extracting selector trace payload parsing, retained-marginal alignment summaries, and teacher score-vector builders. It still owns score component vector construction and retained-decision mask scoring. | Consider score component vector helpers next. Keep exact retained-decision mask scoring in orchestration until there is a narrower behavior regression boundary; do not move query-scoring-dependent diagnostics into `selection/`. |
| `learning/targets/query_local_utility.py` | Roughly 1.4k lines after extracting segment/path math and family evidence. It still owns active target construction, candidate diagnostics, trainability diagnostics, and experimental target helper wiring. | Keep `build_query_local_utility_targets` as the public entry point. Continue by extracting diagnostic/candidate helpers into sibling modules such as `query_local_utility_diagnostics.py` before considering a package conversion. |
| `scoring/method_scoring.py` | Roughly 1.25k lines. It mixes legacy RangeUseful audit construction, active QueryLocalUtility input components, per-query detail rows, trajectory evidence counts, and method scoring. | Extract range-audit support construction and query-row summarization from method execution. Keep artifact field names stable and cover row payloads with regression tests before moving code. |
| `orchestration/diagnostics/` | Derived artifact analyzers now live outside the flat stage namespace. Keep them from becoming a second orchestration package by limiting them to completed-artifact readers and small report builders. | Keep pipeline stages, gates, and payload assembly in top-level `orchestration/`. Add new one-off artifact analyzers under `diagnostics/` instead of the flat package. |
| `learning/model_training.py` | Roughly 1.3k lines. It owns fitting, validation cadence, checkpoint-selection orchestration, and selected-output construction. It is large but currently coherent. | Defer this split until the target/selector diagnostics stabilize. If it grows again, split validation/checkpoint-selection driver code from optimizer-loop mechanics. |
| `benchmarking/reporting/row_fields.py` | Roughly 1k lines. Row extraction, schema defaults, diagnostic field flattening, and table compatibility sit together. | Split stable schema/key declarations from extraction helpers. Regression snapshots must be updated deliberately because report field churn is easy to miss. |
| `Range_QDS/artifacts/` | Local generated output is intentionally ignored and can quickly dominate source scans if not pruned. | Keep artifacts out of source imports. Periodically clean smoke/manual output after the relevant metrics are captured. Do not store maintained docs under `artifacts/manual/`. |

## Recommended Refactor Order

1. Keep the query-driven test split intact. Do not recreate a catch-all
   orchestration test file for cross-module behavior.
2. Continue extracting pure helper modules from
   `learning/targets/query_local_utility.py`. Segment target math and family
   evidence already live in sibling modules; next split candidate/diagnostic
   helpers while leaving the public builder in place.
3. Continue extracting helpers from `orchestration/selector_diagnostics.py`.
   Trace-mask and segment-context payload parsing already lives in
   `orchestration/selector_trace_payloads.py`; retained-decision marginal
   alignment summaries already live in
   `orchestration/selector_marginal_alignment.py`; teacher score-vector
   builders already live in `orchestration/selector_teacher_vectors.py`. Next
   consider score component vector helpers or stop if the remaining file is
   coherent enough.
4. Split `scoring/method_scoring.py` only after row/regression tests cover the
   exact payload fields that downstream reports depend on.
5. Leave `learning/model_training.py` until the target and selector surfaces are
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
