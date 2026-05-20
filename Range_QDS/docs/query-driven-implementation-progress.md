# Query-Driven Checkpoint Progress

This is the short checkpoint log required by
`docs/query-driven-implementation-research-guide.md`. The guide is the source
of truth. Raw metrics and stdout belong in `artifacts/results/`.

## Current State - 2026-05-20

Status: active, not accepted.

Current default stack:

- primary metric: `QueryLocalUtility`
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
- footprint point-hit fraction bands: `medium_operational=[0.006,0.030]`,
  `large_context=[0.010,0.045]`
- target/model/selector: `query_local_utility_factorized`,
  `workload_blind_range`, `learned_segment_budget`
- active behavior target:
  `conditional_behavior_target_variant=query_segment_local_behavior_utility`,
  `replacement_representative_keep_fraction=0.35`,
  `segment_budget_target_aggregation=top20_mean`

Evidence boundary:

- Final grid has not been run.
- Final success remains `false`.
- Global sanity is a reported guardrail during the current local-query-learning
  phase, not the initial hard blocker.
- Small smokes are implementation checks only. Training coherence requires the
  guide's smaller strict evidence levels before any full grid.
- Current strict evidence starts from the two-footprint `range_query_mix`
  replay under current `QueryLocalUtility`; older strict-cell evidence is
  diagnostic only and must not be compared as current-metric acceptance
  evidence.

Latest current-default strict replay:

- artifact:
  `artifacts/results/query_driven_train_marginal_scale64_query40_level3_range_query_mix_seed2527/example_run.json`
- scale: Level 3 source-stratified strict training replay, seed `2527`, 64
  ships, 256 points, 40 requested queries, 5 epochs, train-side marginal
  diagnostics enabled
- MLQDS QueryLocalUtility: `0.1394788551`
- uniform QueryLocalUtility: `0.1247681518`
- Douglas-Peucker QueryLocalUtility: `0.1153266238`
- passed: workload stability, support overlap, target diffusion, workload
  signature, predictability, prior-predictive alignment, global sanity
- failed: learning causality, final grid
- failed causality children: `shuffled_prior_fields_should_lose`,
  `without_query_prior_features_should_lose`,
  `without_behavior_utility_head_should_lose`
- material positive controls: shuffled score loses by `0.0218749319`,
  untrained loses by `0.0242913109`, and no-segment-budget loses by
  `0.0108408237`
- non-causal query-field controls: shuffled-prior and no-query-prior deltas are
  both `-0.0000086480` and change only 4 retained decisions
- behavior-head control: no-behavior delta is `-0.0003343468`
- segment allocation remains high-entropy and score-dominated:
  normalized entropy `0.9869`, segment-score/allocation Spearman `0.8664`,
  length-support/allocation Spearman `0.1639`,
  segment-score/length-support Spearman `0.0472`, and top-20%
  score/length-support overlap `0.2105`
- train/eval segment-teacher transfer decision:
  `diagnose_transfer_features_before_guarded_calibration_probe`; selection/eval
  target Spearman is `-0.6151`, top-k overlap is zero through top `10%`, and
  one selector feature is contradictory-sign
- current active-metric behavior-head evidence is bad: overall exact-marginal
  Spearman `0.0577`, retained-removal Spearman `-0.3817`, and no-behavior
  ablation delta `-0.0003343468`, dominated by the primary model's deficit
  versus no-behavior on `query_local_turn_change_coverage`

Latest rejected diagnostics:

- Head-contrast loss diagnostic:
  `artifacts/results/query_driven_head_contrast_sparse025_behavior015_level3_scale64_query40_seed2527/example_run.json`
  improved MLQDS only slightly to `0.1402280700`, still failed the same
  learning-causality children, made query-prior ablations more wrong-way, and
  did not fix feature flow.
- Square-root prior-transform diagnostic:
  `artifacts/results/query_driven_prior_sqrt_level3_scale64_query40_seed2527/example_run.json`
  increased visible prior-feature differences but still failed directionality;
  shuffled-prior, no-query-prior, and no-behavior deltas remained wrong-way.
- Segment-aware behavior target wiring:
  additive behavior composition failed target diffusion in a Level 1 smoke;
  multiplicative composition passed the tiny wiring smoke and is the active
  candidate for the next guide-compliant smaller strict level. The smoke is not
  evidence of learning.

Next admissible work:

- Start from the guide's smaller strict evidence levels under current defaults
  or an explicitly justified metric/profile variant.
- Diagnose by gate before changing code: workload health, support overlap,
  target diffusion, prior predictability, learning causality, then selector
  allocation.
- Do not run the final grid until the required smaller evidence levels pass.

## Checkpoint Group 1 - Protocol, Baselines, And Current Defaults

Status: completed / foundation.

Covered prior checkpoints: 1-6.

Scope:

- Established the workload-blind protocol, leakage rules, acceptance evidence
  levels, and final-claim gates.
- Simplified the primary score into `QueryLocalUtility` with point mass,
  query-local behavior, and light global sanity.
- Simplified the active workload profile to `range_query_mix` with `density`
  and `sparse_background_control` anchors plus `medium_operational` and
  `large_context` footprints.
- Removed old explicit ship-presence, boundary/event, `small_local`,
  `density_route`, and route-corridor-style active defaults.

Decision:

- Current defaults are the starting point for new evidence.
- Older `QueryUsefulV1`, `range_workload_v1`, old family names, and old scalar
  range-audit components are historical or diagnostic only.
- Global sanity should improve, but it is not an initial hard gate while local
  query behavior and learning causality are being proven.

## Checkpoint Group 2 - Codebase Layout And Maintained-Docs Foundation

Status: completed / cleanup and structure.

Covered prior checkpoints: 7.

Scope:

- Reorganized and documented the top-level package responsibilities in
  `CODE_LAYOUT.md`.
- Split large query-driven tests by owner and removed stale catch-all test
  structure.
- Moved derived artifact analyzers under `orchestration/diagnostics/`.
- Updated package READMEs so the project can be understood top-down.

Decision:

- Keep production ownership boundaries narrow:
  `workloads` owns generation/execution, `learning` owns targets and training,
  `selection` owns query-free mask construction, `scoring` owns metrics, and
  `benchmarking` owns final-grid/report policy.
- Future refactors should extract pure helpers first and avoid compatibility
  shims unless they are explicitly temporary.

## Checkpoint Group 3 - Workload/Profile Calibration

Status: completed / current-default workload health.

Covered prior checkpoints: 8-13.

Scope:

- Rebased evidence around the current metric/profile pair instead of old
  range-audit assumptions.
- Diagnosed workload-signature failures as profile/query-count/split
  compatibility problems.
- Added deterministic profile family planning, stricter profile acceptance
  bands, and point-hit targeted proposal calibration.
- Fixed synthetic route-family splits so generator-only Level 3 probes pass
  workload stability and workload signature at the `range_query_mix` 48-query
  floor on seeds `2524` and `2525`.

Decision:

- `range_query_mix` is viable enough for strict diagnostics after calibration.
- Generator-only success is not training coherence and not final success.
- Profile/metric weights remain adjustable research choices, but changes need
  gate-by-gate evidence that the workload/scoring pair is producing incoherent
  or untrainable signal.

## Checkpoint Group 4 - Strict Replay And Causality Localization

Status: completed / blocker localization.

Covered prior checkpoints: 14-18.

Scope:

- Ran the generator-fixed strict replay and localized the primary blocker to
  learning causality rather than workload health alone.
- Tested behavior-rank and allocation-floor diagnostics without changing active
  defaults.
- Added behavior-head-as-segment allocation diagnostics.

Key result:

- MLQDS could beat uniform and Douglas-Peucker on current QueryLocalUtility in
  strict replay, but learning causality still failed.
- Query-prior and behavior-head ablations were weak, wrong-way, or immaterial.
- Removing selector allocation floor increased score in one diagnostic but did
  not create reliable learned control and did not pass causality.

Decision:

- Do not promote score gains that fail child causality gates.
- Do not compensate for weak learning with selector tricks or temporal
  scaffolding.

## Checkpoint Group 5 - Selector Allocation And Length-Support Diagnostics

Status: completed / selector diagnosis.

Covered prior checkpoints: 19-23.

Scope:

- Isolated segment allocation behavior, length-support alignment, and learned
  score dominance.
- Tested guarded segment transfer calibration, route-density prior exposure,
  and length-support allocation-weight probes.

Key result:

- Segment allocation remained high-entropy and score-dominated.
- Length-support and segment-score signals were poorly aligned.
- Non-default calibration or length-support probes did not justify a selector
  default change under unchanged gates.

Decision:

- Selector allocation is a real blocker, but the stronger root issue is still
  trainable query-local signal and transfer. Fixing allocation alone is not
  enough.

## Checkpoint Group 6 - Train-Marginal Transfer And Scale Diagnostics

Status: completed / transfer diagnosis.

Covered prior checkpoints: 24-28.

Scope:

- Added train-side marginal diagnostics and strict train-marginal replay.
- Localized transfer failure across scale, query count, and split composition.
- Confirmed the current 64/256/40 replay passes workload, support, target
  diffusion, predictability, prior-predictive, and global-sanity gates while
  still failing learning causality.

Key result:

- Train-side exact marginal teacher is non-leaky and shape-viable, but transfer
  is unacceptable: selection/eval target Spearman `-0.6151` and top-k teacher
  overlap is zero through top `10%`.

Decision:

- Current blocker is semantic directionality and train/eval transfer, not just
  missing instrumentation.
- Use retained-decision marginal alignment from
  `selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_local_utility_alignment`,
  not from `learning_causality_summary.selection_causality_diagnostics`.

## Checkpoint Group 7 - Head/Prior/Behavior Target Diagnostics

Status: completed / rejected diagnostic variants plus active target candidate.

Covered prior checkpoints: 29-34.

Scope:

- Tested sparse-head rank/BCE contrast, model-facing prior transforms, current
  focus family/head transfer, active-metric alignment cleanup, behavior-head
  semantic alignment, and segment-aware behavior target wiring.

Key results:

- Head-contrast and square-root prior-transform variants were rejected. They
  improved some surface metrics but did not fix causality or retained-mask
  directionality.
- Family/head transfer diagnostics now read active focus families from the
  artifact and surface active-metric exact-marginal alignment.
- Behavior-head semantic alignment showed the target was most aligned with
  replacement representative value, weakly aligned with segment budget, and
  still harmful under the no-behavior ablation.
- Multiplicative segment-aware behavior target passed Level 1 wiring; additive
  composition failed target diffusion.

Decision:

- Keep the multiplicative segment-aware behavior target as the active candidate
  for the next smaller strict evidence level.
- Do not claim learning from Level 1 wiring.

## Checkpoint Group 8 - Repository Hygiene, Canonicalization, And Stale-Code Cleanup

Status: completed / cleanup.

Covered prior checkpoints: 35-38.

Scope:

- Pruned stale artifacts and caches.
- Canonicalized current names:
  `workload_blind_range`, `learned_segment_budget`,
  `range_query_mix_workload_blind`, `query_driven_workload_blind`.
- Renamed the old scalar scorer to `scalar_workload_blind_range`.
- Updated maintained docs to stop presenting chronological names or old family
  names as active defaults.
- Removed obsolete code paths tied to old `small_local` / ship-evidence proxy
  derived diagnostics.
- Removed chronological workload-profile `version` /
  `workload_profile_version` payload fields from current workload profile
  artifacts.

Key results:

- Removed `274` stale result directories and disposable cache/manual output.
- Retained `26` result directories tied to current evidence or useful negative
  lessons.
- Current code no longer uses old chronological active model, selector, class,
  or metric-schema names.
- Removed two obsolete query-ship derived analyzers tied to the old
  `small_local` / ship-evidence proxy path.
- Renamed `range_v2_prior_feature_scaling` to
  `workload_blind_range_prior_feature_scaling`.

Decision:

- Keep `schema_version` fields only as artifact/report compatibility metadata.
- Do not keep old derived diagnostic modules just because progress history
  references their results. The historical result can live in the progress log;
  active code should not preserve misleading diagnostic surfaces.
- Use semantic variation names for diagnostic paths, not chronological suffixes.

## Validation Summary

Latest focused validation:

- `python3 -m py_compile` on touched source and test files after stale-code
  cleanup.
- `uv run --group dev -- ruff check` on touched source and test files after
  stale-code cleanup.
- `uv run --group dev -- pytest` on focused workload-profile, diagnostics,
  guardrail, protocol-gate, and model-feature suites after stale-code cleanup:
  `80 passed`.
- `git diff --check` after stale-code cleanup.
- Targeted non-doc stale-code scans after stale-code cleanup.

Recent broader validation:

- Focused canonical naming pytest set covering model factory/features,
  QueryLocalUtility training, learned segment-budget selector,
  protocol/causality gates, retained masks, learning target stage, benchmark
  profile/report regressions, and scoring metrics: `259 passed`.
- `python3 -m py_compile`, `ruff check`, and `pyright` on renamed
  model/selector/benchmark implementation surfaces.
- Maintained-doc stale-name/stale-phrasing scans found no active-doc hits
  outside progress history and next-iteration guardrails.
- Maintained artifact-reference scan found no missing retained result
  directories.
- Earlier focused suites:
  - workload/profile/property/guardrail: `42 passed`
  - orchestration/scoring: `178 passed`
  - benchmarking/report regression: `40 passed`
  - learning/orchestration payload: `56 passed`

Validation caveat:

- These validations prove implementation integrity and cleanup consistency.
  They are not scientific evidence of training coherence or final success.
