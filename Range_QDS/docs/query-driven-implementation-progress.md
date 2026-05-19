# Query-Driven Checkpoint Progress

This is the short checkpoint log required by
`docs/query-driven-implementation-research-guide.md`. The guide is the source
of truth. Raw metrics and stdout belong in `artifacts/results/`.

## Current State - 2026-05-20

Status: active, not complete.

Current default stack:

- primary metric: `QueryLocalUtility` schema `5`
- score groups: `query_point_mass=0.50`, `query_local_behavior=0.45`,
  `global_sanity=0.05`
- score components: `query_point_recall=0.50`,
  `query_local_interpolation_fidelity=0.20`,
  `query_local_turn_change_coverage=0.15`,
  `query_local_continuity=0.10`,
  `endpoint_or_skeleton_sanity=0.02`,
  `global_shape_guardrail_score=0.02`,
  `length_preservation_guardrail=0.01`
- workload profile: `range_query_mix`
- active anchors: `density=0.80`, `sparse_background_control=0.20`
- active footprints: `medium_operational=0.6923076923076923`,
  `large_context=0.3076923076923077`
- target/model/selector: `query_local_utility_factorized`,
  `workload_blind_range_v2`, `learned_segment_budget_v1`

Evidence boundary:

- No strict workload-health or learning-coherence rerun has been performed
  under schema `5` and the two-footprint `range_query_mix` profile.
- Final grid has not been run.
- Final success remains `false`.

Current pre-simplification strict-cell reference:

- artifact:
  `artifacts/results/query_driven_v2_checkpoint85_segment_aggregation_current_best_strict_local/example_run.json`
- MLQDS QueryLocalUtility: `0.1662115143`
- uniform QueryLocalUtility: `0.1421296610`
- Douglas-Peucker QueryLocalUtility: `0.1671038781`
- passed: workload stability, support overlap, target diffusion, workload
  signature, prior-predictive alignment, global sanity
- failed: predictability, learning causality

Active blockers:

- Predictability and learning causality are still unproven under the current
  defaults.
- Exact retained-decision marginal alignment remains a central selector/target
  diagnostic. Current artifacts should read it from
  `selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_local_utility_alignment`,
  not from `learning_causality_summary.selection_causality_diagnostics`.
- Historical `small_local`, `crossing_turn_change`, and old
  `QueryUsefulV1` diagnostics are pre-simplification evidence. They are not
  active workload requirements under the current profile.

Next checkpoint:

- Run the guide-required smaller strict evidence levels under schema `5` and
  the two-footprint `range_query_mix` profile before any final-grid run.

## Checkpoint Group 1 - Protocol, Workload, And Baseline Contract

Status: completed / historical foundation.

Goal:

- Establish workload-blind range compression as the product contract.
- Make workload generation, query-prior construction, retained-mask freezing,
  scoring, and final gates auditable.

Changes:

- Added workload-blind protocol flags and final-gate checks.
- Stabilized workload signatures and support-overlap diagnostics.
- Added train-only query-prior fields and retained-mask freeze ordering.
- Added retained-decision marginal diagnostics and selector trace payloads.

Tests:

- Historical implementation checkpoints ran focused `py_compile`, `ruff`,
  `pyright`, and pytest suites at the time of each code change.
- Detailed per-checkpoint command logs were intentionally removed from this
  condensed progress file; raw evidence remains in artifacts and git history.

Experiment artifact:

- representative:
  `artifacts/results/query_driven_v2_checkpoint61_selection_marginal_teacher_current_best_strict_local/example_run.json`
- representative:
  `artifacts/results/query_driven_v2_checkpoint65_selector_decision_surface_diagnosis/selector_decision_surface_diagnosis.json`

Key results:

- Workload/profile gates can pass at the workload-healthy strict cell.
- Direct selection-side exact-marginal teacher and simple hybrid teacher
  selectors were guarded but did not beat the primary learned selector at
  strict scale.
- Selector score and segment allocation can move masks without ranking exact
  retained-decision marginal value correctly.

Decision:

- Keep the workload-blind protocol and selector-trace diagnostics.
- Do not promote direct exact-marginal teacher or hybrid teacher selectors from
  the old strict artifacts.

## Checkpoint Group 2 - Prior, Head, And Selector Materiality Diagnostics

Status: completed / rejected variants.

Goal:

- Diagnose why learned workload-derived signal was not material enough in the
  retained masks.

Changes:

- Tested prior scaling, dense-head rank pressure, behavior-rank pressure,
  score/selector formula variants, query-free teacher guards, and
  retained-marginal teacher consumers.
- Added diagnostics that separate raw score, selector score, segment score,
  segment rank, and exact marginal value.

Tests:

- Historical code checkpoints ran focused static checks and unit/regression
  tests when each diagnostic surface was added.

Experiment artifact:

- representative:
  `artifacts/results/query_driven_v2_checkpoint72_checkpoint_teacher_hybrid_current_best_strict_local/example_run.json`
- representative:
  `artifacts/results/query_driven_v2_checkpoint99_segment_transfer_calibration_zblend_current_best_strict_local/example_run.json`

Key results:

- Better factorized-head fit was not enough. Several variants improved local
  target/head diagnostics while reducing retained-mask quality or causality.
- Exact retained-marginal information was available but did not transfer into
  a safe production selector through direct or hybrid consumers.
- Checkpoint99 slightly improved over some rejected paths but remained below
  the strongest pre-simplification strict reference and still failed causality
  children.

Decision:

- Do not revive old scalar proxy losses, broad local-head targets, or selector
  blend tuning without a sharper marginal-alignment hypothesis.

## Checkpoint Group 3 - Workload And Scoring Compatibility Diagnosis

Status: completed / diagnostic.

Goal:

- Determine whether the workload profile and scoring components produced a
  trainable query-local signal.

Changes:

- Added workload/component compatibility diagnostics.
- Added post-hoc recalibration probes and blocker-preserving recalibration
  checks.
- Added family-conditioned target/head trainability diagnostics.

Tests:

- Historical diagnostic checkpoints ran focused validation when instrumentation
  changed.

Experiment artifact:

- `artifacts/results/query_driven_v2_checkpoint80_workload_component_compatibility_diagnosis/workload_component_compatibility_diagnosis.json`
- `artifacts/results/query_driven_v2_checkpoint82_blocker_preserving_recalibration_diagnosis/workload_component_blocker_preserving_recalibration_diagnosis.json`
- `artifacts/results/query_driven_v2_checkpoint83_family_trainability_current_best_strict_local/example_run.json`

Key results:

- Pre-simplification blockers concentrated in `small_local`, `density`,
  `crossing_turn_change`, and `medium_operational`.
- Persistent negative components were dominated by ship/point-mass evidence and
  related recall terms.
- A query-local-sensible post-hoc recalibration could flip the
  MLQDS-minus-Douglas-Peucker delta positive, but the gain came with high
  masking risk because it downweighted or moved away from the same blockers.
- Blocker-preserving recalibration stayed blocked.

Decision:

- Do not adopt post-hoc recalibration as acceptance evidence.
- Fix the metric/profile contract directly, then rerun strict evidence.

## Checkpoint Group 4 - Family-Local And Segment Target Diagnostics

Status: completed / rejected or diagnostic variants.

Goal:

- Test whether family-local point/segment targets could make the signal
  trainable without weakening gates.

Changes:

- Added family-local query-hit/ship, ship-gated behavior, composed score, and
  segment-budget candidate diagnostics.
- Added segment aggregation diagnostics and two-stage allocation/point-choice
  views.
- Tested guarded query-ship max-pool and query-ship local-head target variants.
- Added family-prior predictability, selector marginal calibration, and
  segment-transfer diagnostics.

Tests:

- Historical checkpoints ran focused static checks and unit tests for added
  diagnostic payloads.

Experiment artifact:

- `artifacts/results/query_driven_v2_checkpoint85_segment_aggregation_current_best_strict_local/example_run.json`
- `artifacts/results/query_driven_v2_checkpoint86_query_ship_max_pool_target_current_best_strict_local/example_run.json`
- `artifacts/results/query_driven_v2_checkpoint93_family_prior_predictability_max_pool_current_best_strict_local/example_run.json`

Key results:

- Family-local point candidates recovered ship-query evidence strongly in the
  old profile: `small_local` Spearman `0.9740`, `density` Spearman `0.9191`.
- Segment aggregation diagnostics showed useful two-stage signals:
  `small_local` best pair coverage `0.4000`, `density` best pair coverage
  `0.6075`.
- The guarded query-ship max-pool target slightly beat Douglas-Peucker on the
  old metric reference (`0.1673482145` vs `0.1671038781`) but still failed
  predictability and learning causality.
- Broader query-ship local heads worsened target diffusion and retained-mask
  quality.

Decision:

- Do not promote family-local or guarded segment target variants.
- Preserve the diagnostics, but treat their evidence as pre-simplification.

## Checkpoint Group 5 - Current Metric And Workload Defaults

Status: completed / Level 0 implementation only.

Goal:

- Simplify the active metric and workload profile at the root instead of
  layering more target or selector compensation on weak old signals.

Changes:

- Removed explicit ship-presence/coverage and boundary/event evidence groups
  from the primary aggregate.
- Rebalanced groups to point mass `0.50`, query-local behavior `0.45`, and
  global sanity `0.05`.
- Renamed the active metric to `QueryLocalUtility` and active keys to
  `query_local_utility`; old `query_useful_v1` production aliases were not
  retained.
- Renamed the active workload profile to `range_query_mix`, renamed
  `density_route` to `density`, and removed old anchor/footprint families from
  the active profile.
- Made schema `5` use direct `query_point_recall`, direct query-local
  interpolation fidelity, turn coverage, and min-gap continuity. It stopped
  sourcing point mass from `range_point_f1` and stopped using fallback behavior
  components.
- Removed `small_local` from active `range_query_mix` footprints and
  renormalized to `medium_operational` plus `large_context`.

Tests:

- Current focused code validation for this default stack included
  `py_compile`, `ruff`, `pyright`, workload/property/guardrail tests,
  orchestration/scoring tests, benchmarking/report regression tests, and
  learning/orchestration payload tests.

Experiment artifact:

- path: none
- command: none

Key results:

- Current code defaults now match the stack in `Current State`.
- This is implementation evidence only. No strict retraining,
  workload-health, or learning-coherence rerun has been performed under
  schema `5` and the two-footprint profile.

Decision:

- Use schema `5` and the two-footprint `range_query_mix` profile for all new
  checkpoints.
- Do not compare schema `5` scores against old schema scores as if they are the
  same metric.

## Checkpoint Group 6 - Documentation And Guide Cleanup

Status: completed / docs only.

Goal:

- Make maintained docs point at the current implementation contract without
  carrying stale legacy/changelog noise in the source-of-truth guide.

Changes:

- Documented current scoring and workload defaults across maintained docs.
- Retitled the guide as an implementation/research guide, removed duplicated
  checkpoint chronology from it, and moved history back to this progress log.
- Condensed this progress log into logical checkpoint groups.
- Updated stale links from removed legacy filenames to the current
  `query-driven-implementation-*` docs.
- Replaced the obsolete `Next-Iterations.md` chronology with a concise current
  next-action plan.
- Rewrote `keep-in-mind.md` as concise project guidance instead of raw notes.

Tests:

- `git diff --check`
- Stale-guide and stale-default `rg` scans over maintained docs.
- Broken old-doc-link `rg` scan for removed legacy filenames.

Experiment artifact:

- path: none
- command: none

Key results:

- Documentation only. No strict retraining, workload-health, or
  learning-coherence rerun was performed.
- The progress log now summarizes old work by logical phase instead of
  maintaining dozens of checkpoint entries.
- Maintained docs no longer link to removed legacy filenames.

Decision:

- Keep this progress log compact. Add new entries only when they change the
  current state, evidence boundary, accepted defaults, or next action.

## Checkpoint Group 7 - Stale-Code And Naming Cleanup

Status: completed / code hygiene only.

Goal:

- Remove or update clearly stale implementation wording and misleading names
  after the scoring/profile simplification, without deleting historical
  diagnostics that are still useful for comparing old blockers.

Changes:

- Replaced remaining production/test prose that used old redesign wording with
  implementation/current-acceptance wording.
- Renamed ambiguous diagnostic focus constants so historical blocker families
  are not confused with active workload-profile requirements.
- Removed `small_local` from active target-trainability focus families; it now
  remains only in explicitly historical transfer diagnostics and tests.
- Renamed vague workload/scoring recalibration probe names from "sensible" and
  "point-mass-preserving" to behavior-heavy and point-mass-heavy names.
- Renamed the deprecated retained-marginal alignment layout flag so it points
  callers at the current
  `selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_local_utility_alignment`
  path.

Tests:

- `git diff --check`
- `python3 -m py_compile` on touched production and test files.
- `uv run --group dev -- ruff check` on touched production and test files.
- `uv run --group dev -- pyright` on touched production files.
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_implementation.py Range_QDS/tests/unit/scoring/test_metrics.py Range_QDS/tests/guardrails/test_implementation_guardrails.py -q`
  (`197 passed`)

Experiment artifact:

- path: none
- command: none

Key results:

- Code hygiene only. No strict retraining, workload-health, or
  learning-coherence rerun was performed.
- Active diagnostic focus now matches the current workload defaults:
  `density` and `medium_operational`.
- Historical `small_local` transfer diagnostics remain isolated and explicitly
  marked as historical.

Decision:

- Keep compatibility/legacy paths only when they still serve diagnostics,
  artifact comparability, or checkpoint loading. Do not present them as active
  defaults.

## Checkpoint Group 8 - Test Stale-Logic Cleanup

Status: completed / tests only.

Goal:

- Remove or clarify stale test logic after the production cleanup, without
  deleting negative guardrails for legacy/reporting behavior that is still
  intentionally supported.

Changes:

- Made historical `small_local` transfer-diagnostic fixtures explicit via
  `HISTORICAL_SMALL_LOCAL_FAMILY` instead of leaving old family names as
  neutral-looking active fixtures.
- Renamed the historical `small_local` transfer-gap test and local variables so
  the test reads as historical artifact coverage.
- Renamed misleading QueryLocalUtility no-fallback test variables to
  `legacy_range_point_only_*`, matching the schema-5 assertion.
- Renamed the uncovered-anchor-chasing workload fixture so it no longer reads as
  the active/default generator path.
- Made the legacy fixed-count workload-stability fixture internally consistent:
  `legacy_generator` plus `legacy_fixed_or_target_coverage`.
- Replaced a generic `legacy` benchmark run label in a guardrail fixture with a
  `range_useful_diagnostic` label.

Tests:

- `git diff --check`
- `python3 -m py_compile Range_QDS/tests/unit/orchestration/test_query_driven_implementation.py Range_QDS/tests/unit/scoring/test_metrics.py Range_QDS/tests/guardrails/test_implementation_guardrails.py`
- `uv run --group dev -- ruff check Range_QDS/tests/unit/orchestration/test_query_driven_implementation.py Range_QDS/tests/unit/scoring/test_metrics.py Range_QDS/tests/guardrails/test_implementation_guardrails.py`
- `uv run --group dev -- pyright Range_QDS/tests/unit/orchestration/test_query_driven_implementation.py Range_QDS/tests/unit/scoring/test_metrics.py Range_QDS/tests/guardrails/test_implementation_guardrails.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_implementation.py Range_QDS/tests/unit/scoring/test_metrics.py Range_QDS/tests/guardrails/test_implementation_guardrails.py -q`
  (`197 passed`)

Experiment artifact:

- path: none
- command: none

Key results:

- Tests only. No strict retraining, workload-health, or learning-coherence rerun
  was performed.
- Remaining legacy/test references are negative guardrails, report-compatibility
  checks, or explicitly historical diagnostic fixtures.

Decision:

- Keep legacy test data only when it protects a current invariant: removed APIs
  stay removed, final claims stay separated from legacy metrics, or historical
  diagnostic readers still parse old artifacts correctly.

## Checkpoint Group 9 - Code Organization Review

Status: completed / architecture review only.

Goal:

- Identify structural pressure points that make the project harder to understand
  top-down, without starting broad file moves during an active research track.

Changes:

- Updated `CODE_LAYOUT.md` with current organization pressure points and a
  conservative refactor order.
- Recorded the largest source/test hotspots and separated immediate
  maintainability wins from higher-risk production refactors.

Findings:

- Highest-return cleanup is splitting
  `tests/unit/orchestration/test_query_driven_implementation.py`, which is
  roughly 6.3k lines and crosses too many ownership boundaries.
- The flat `orchestration/` package now mixes pipeline stages, CLI entrypoints,
  gates, payload assembly, and derived artifact analyzers. The derived analyzers
  should move under `orchestration/diagnostics/`.
- `orchestration/selector_diagnostics.py`,
  `learning/targets/query_local_utility.py`, and `scoring/method_scoring.py`
  are the main production modularization candidates, but should be split by
  pure helper boundaries with focused regression tests.
- `Range_QDS/artifacts/` is correctly ignored, but local generated output
  dominates tree size and should be kept out of source/docs reasoning.

Tests:

- `git diff --check`
- Structure scans using `find`, `du`, `wc -l`, and targeted `rg` import/function
  listings.

Experiment artifact:

- path: none
- command: none

Key results:

- No production code or test behavior changed.
- Refactor order is now documented so future cleanup can be staged without
  confusing research evidence with code movement.

Decision:

- Do not start production module moves until the target/selector blocker
  diagnostics are stable enough to protect with small focused tests. Split the
  oversized orchestration test file first.

## Validation

Current docs validation:

- `git diff --check`
- Stale-guide `rg` scan for legacy naming, old metric/profile names, old
  schemas, and stale strict-reference prose. Remaining guide hits are
  explicit historical exclusions or the `range_point_f1` no-fallback caveat.
- Stale-default `rg` scan over maintained docs. Remaining hits are explicit
  historical-name or legacy-diagnostic references.
- Broken old-doc-link `rg` scan for removed legacy filenames.

Current focused implementation validation before docs-only condensation:

- `python3 -m py_compile` on active workload/profile, orchestration
  compatibility, and focused orchestration test files.
- `uv run --group dev -- ruff check` on active workload/profile,
  compatibility, orchestration, and property-test surfaces.
- `uv run --group dev -- pyright` on the same focused surfaces.
- `uv run --group dev -- pytest` workload/profile/property/guardrail suites:
  `42 passed`.
- `uv run --group dev -- pytest` orchestration/scoring suites: `178 passed`.
- `uv run --group dev -- pytest` benchmarking/report regression suites:
  `40 passed`.
- `uv run --group dev -- pytest` learning/orchestration payload suites:
  `56 passed`.
