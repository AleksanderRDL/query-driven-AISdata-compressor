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
| Query-driven implementation tests | The old oversized orchestration and benchmark test files have been split by owner. The largest remaining test files are under the 1k-line target but several are close enough to regress quickly. | Keep adding new tests to the owning split file. If any split file grows beyond roughly 1k lines, extract fixture builders or split by the production module under test immediately. |
| `config/run_config.py` | Roughly 790 lines. Builder argument explosion has been removed, but the flat override boundary is still broad and easy to drift from the CLI/parser fields. | Keep override ownership in `RunConfigOverrides`, `RUN_CONFIG_DEFAULT_OVERRIDES`, and `build_run_config`. Add fields through the dataclass-driven path; do not reintroduce giant signatures. |
| `workloads/generation/generator.py` | Roughly 850 lines. Coverage loops, fixed-count top-up, diagnostics, and signature payloads have owners, but the public generator still coordinates many subdomains. | Keep profile planning, coverage, signatures, and validation in their owner modules. Extract only when a new generation subdomain starts accumulating policy. |
| `orchestration/retained_mask_ablation_stage.py` | Roughly 900 lines. Context/state helpers now keep the public freeze function bounded, but the file still owns ablation orchestration and payload assembly. | Split any new ablation family into focused helpers. Keep query-free mask freezing in orchestration; do not push eval-query-sensitive logic into `selection/`. |
| `orchestration/selection_causality_diagnostics.py` | Roughly 925 lines. Teacher-selector, prior-field, and segment-head ablations are helper-owned, but the module still combines scoring, gates, and report payload assembly. | Separate scoring/report shaping if new diagnostics are added. Keep final-claim gate policy near its tests. |
| `orchestration/selector_diagnostics.py` | Roughly 330 lines after extracting selector trace payload parsing, retained-marginal alignment summaries, and teacher score-vector builders. | Leave it alone unless score component vector construction starts growing again. Do not move query-scoring-dependent diagnostics into `selection/`. |
| `learning/targets/query_local_utility.py` | Roughly 600 lines after extracting segment/path math and family evidence. It still owns active target construction, candidate diagnostics, trainability diagnostics, and experimental target helper wiring. | Keep `build_query_local_utility_targets` as the public entry point. Extract more only when a concrete target subdomain starts growing again. |
| `scoring/method_scoring.py` | Roughly 250 lines after range-audit and query-row extraction. It should stay focused on method execution and stable score payload assembly. | Do not reintroduce report flattening or audit construction here. Keep row payload changes covered by regression tests before moving fields. |
| `orchestration/diagnostics/` | Derived artifact analyzers now live outside the flat stage namespace. Keep them from becoming a second orchestration package by limiting them to completed-artifact readers and small report builders. | Keep pipeline stages, gates, and payload assembly in top-level `orchestration/`. Add new one-off artifact analyzers under `diagnostics/` instead of the flat package. |
| `learning/model_training.py` / `learning/model_training_loop.py` | Training is split into setup/output assembly and epoch-loop orchestration. `train_model` and the epoch loop remain large explicit orchestration functions under known guardrail budgets. | Keep future training changes in the owning helper module: validation setup in `model_training_validation.py`, epoch/checkpoint behavior in `model_training_loop.py`, and target/model setup in `model_training.py`. |
| `benchmarking/reporting/row_fields.py` | Roughly 370 lines after domain row builders moved to sibling modules. The remaining file should only coordinate row assembly and small cross-domain fields. | Put new diagnostic flattening in the owning `row_*_fields.py` module. Regression snapshots must be updated deliberately because report field churn is easy to miss. |
| `Range_QDS/artifacts/` | Local generated output is intentionally ignored and can quickly dominate source scans if not pruned. | Keep artifacts out of source imports. Periodically clean smoke/manual output after the relevant metrics are captured. Do not store maintained docs under `artifacts/manual/`. |

## Recommended Refactor Order

1. Keep the query-driven test split intact. Do not recreate a catch-all
   orchestration test file for cross-module behavior.
2. Keep `config/run_config.py` on the dataclass-driven builder path. New run
   fields should update override tests and parser/config boundaries together.
3. Keep workload-generation policy out of `generation/generator.py` when a
   direct owner already exists under `workloads/generation/`.
4. Split the remaining ablation and causality orchestration pressure points:
   `orchestration/retained_mask_ablation_stage.py` and
   `orchestration/selection_causality_diagnostics.py`.
5. Continue extracting pure helper modules from
   `learning/targets/query_local_utility.py`. Segment target math and family
   evidence already live in sibling modules; next split candidate/diagnostic
   helpers while leaving the public builder in place.
6. Leave `orchestration/selector_diagnostics.py` mostly alone unless it starts
   growing again; the earlier trace, marginal-alignment, and teacher-vector
   extractions were enough for now.
7. Split `scoring/method_scoring.py` only after row/regression tests cover the
   exact payload fields that downstream reports depend on.

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
