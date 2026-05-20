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
  guide's strict evidence levels before any full grid.
- Current strict evidence starts from the two-footprint `range_query_mix`
  replays under current `QueryLocalUtility`; older range-audit evidence is
  diagnostic only and must not be compared as current acceptance evidence.

Latest current-default strict replay:

- artifact:
  `artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/example_run.json`
- scale: Level 3 source-stratified strict training replay, seed `2527`, 64
  ships, 256 points, 40 requested queries, 5 epochs, train-side marginal
  diagnostics enabled
- MLQDS QueryLocalUtility: `0.1431090566`
- uniform QueryLocalUtility: `0.1247681518`
- Douglas-Peucker QueryLocalUtility: `0.1153266238`
- passed: workload stability, support overlap, target diffusion, workload
  signature, predictability, prior-predictive alignment, global sanity
- failed: learning causality, final grid
- failed causality children: `shuffled_prior_fields_should_lose`,
  `without_query_prior_features_should_lose`,
  `without_behavior_utility_head_should_lose`
- material positive controls: shuffled score loses by `0.0241102168`,
  untrained loses by `0.0177732905`, no-segment-budget loses by
  `0.0099825888`, and prior-field-only loses by `0.0319842784`
- non-causal query-field controls: shuffled-prior and no-query-prior deltas are
  both `0.0`
- behavior-head control: no-behavior delta is only `0.0014985765`, below the
  `0.005` materiality gate
- selector alignment is mixed: exact retained-marginal raw-score Spearman
  `0.2779` and selector-score Spearman `0.2881`, but segment-score Spearman
  `-0.0812` and behavior-component Spearman `-0.0486`
- behavior-head fit is still weak: conditional-behavior prediction std
  `0.002631` versus target std `0.166493`, with Kendall tau `0.0251`

Latest smaller strict diagnostics:

- `artifacts/results/query_driven_behavior_segment_target_mult_level2_seed2532/example_run.json`
  was a Level 2 32/192/24-query probe. It passed support, target diffusion,
  predictability, and prior alignment, but failed workload stability, workload
  signature, learning causality, and global sanity. It is a blocker-localizing
  diagnostic, not learning evidence.
- `artifacts/results/query_driven_behavior_segment_target_mult_scale48_query32_seed2533/example_run.json`
  increased scale to 48/192/32 queries. Workload stability recovered, but
  workload signature, predictability, prior alignment, learning causality, and
  global sanity still failed. It does not justify model or selector tuning
  without more gate-level diagnosis.

Current blocker:

- The active multiplicative segment-aware behavior target improves the strict
  Level 3 score versus the previous current-default replay, and it beats
  uniform and Douglas-Peucker in that cell.
- It still fails learned causality. Query-prior fields are currently immaterial
  to retained masks at the gate, and the behavior-utility head is too weak to
  count as causal.
- The next useful work is semantic causality diagnosis: query-prior feature
  flow, behavior target/loss coupling, segment-score calibration, and
  workload/scoring compatibility if the prior/behavior signals remain
  incoherent under healthy gates.

Next admissible work:

- Diagnose by gate before changing code: workload health, support overlap,
  target diffusion, prior predictability, learning causality, then selector
  allocation.
- Do not run the final grid until the required smaller evidence levels pass.
- Do not loosen gates, add a large temporal scaffold, or promote a diagnostic
  variant that only improves surface score while failing causality.

## Checkpoint Phase 1 - Defaults, Protocol, And Workload Profile

Status: completed / foundation.

Condenses prior checkpoints 1-13.

Scope:

- Established the workload-blind protocol, leakage rules, evidence levels, and
  final-claim gates.
- Simplified the primary score into `QueryLocalUtility` with direct point mass,
  direct query-local behavior, and light global sanity.
- Simplified the active workload profile to `range_query_mix` with `density`
  and `sparse_background_control` anchors plus `medium_operational` and
  `large_context` footprints.
- Removed old active defaults tied to explicit ship-presence, boundary/event,
  `small_local`, `density_route`, and route-corridor-style components.
- Calibrated family planning, acceptance bands, point-hit proposal targeting,
  and source-stratified synthetic route-family splits so generator-only Level 3
  probes can pass workload stability and signature at the current profile's
  practical query floor.

Decision:

- Current metric/profile defaults are the starting point for new evidence.
- Generator-only success is not training coherence and not final success.
- Anchor/profile dimensions and score weights are adjustable research choices,
  but changes need gate-by-gate evidence that the current workload/scoring pair
  is incoherent or untrainable.

## Checkpoint Phase 2 - Strict Replay And Transfer Diagnosis

Status: completed / blocker localization.

Condenses prior checkpoints 14-28.

Scope:

- Ran strict current-default replays after generator/profile fixes.
- Localized the main blocker to learning causality rather than workload health
  alone.
- Tested behavior-rank, allocation-floor, behavior-head segment allocation,
  route-density prior exposure, length-support weighting, guarded transfer
  calibration, and train-side marginal diagnostics.

Key result:

- MLQDS can beat uniform and Douglas-Peucker on `QueryLocalUtility` in a
  healthy strict synthetic cell, but child causality gates still fail.
- Selector allocation is high-entropy and score-dominated, while length-support
  and segment-score signals are poorly aligned.
- Train-side exact retained-marginal teachers are non-leaky and shape-viable,
  but older selection/eval transfer evidence was contradictory: target
  Spearman `-0.6151` and top-k overlap zero through top `10%`.

Decision:

- Do not promote score gains that fail child causality gates.
- Do not compensate for weak learning with selector tricks or temporal
  scaffolding.
- Read active retained-decision marginal alignment from
  `selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_local_utility_alignment`,
  not from `learning_causality_summary.selection_causality_diagnostics`.

## Checkpoint Phase 3 - Target, Prior, And Head Diagnostics

Status: active / latest research blocker.

Condenses prior checkpoints 29-34 plus the active-target strict probes.

Scope:

- Tested sparse-head rank/BCE contrast, model-facing prior transforms,
  family/head transfer diagnostics, active-metric retained-marginal alignment,
  behavior-head semantic alignment, and segment-aware behavior target wiring.
- Rejected the head-contrast and square-root prior-transform probes as active
  defaults. They changed some surface metrics but did not fix query-prior or
  behavior-head causality.
- Promoted the multiplicative `query_segment_local_behavior_utility` target
  from Level 1 wiring to stricter diagnostics.
- Ran Level 2, intermediate 48/192/32-query, and Level 3 64/256/40-query
  probes under unchanged current defaults.

Key result:

- Level 2 exposed workload/signature pressure at 32/192 scale.
- The 48/192/32-query diagnostic fixed workload stability but still failed
  signature, predictability, prior alignment, and causality.
- The Level 3 active-target replay passed every required pre-causality gate and
  global sanity, and beat uniform/DP, but learning causality still failed on
  query-prior and behavior-head controls.

Decision:

- The active behavior target is better than the earlier current-default replay,
  but not accepted.
- Next work should diagnose why query-prior features and behavior-head outputs
  do not materially control retained masks under a healthy Level 3 cell.
- Do not claim training coherence from Level 1 smoke, Level 2 blocker
  localization, or a Level 3 score gap with failed causality children.

## Checkpoint Phase 4 - Repository Hygiene And Canonicalization

Status: completed / cleanup.

Condenses prior checkpoints 35-38.

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

Key result:

- Removed `274` stale result directories and disposable cache/manual output.
- Retained `26` result directories at cleanup time. The current retained result
  set is `29` directories after the three active-target strict diagnostics.
- Current code no longer uses old chronological active model, selector, class,
  or metric-schema names.

Decision:

- Keep `schema_version` fields only as artifact/report compatibility metadata.
- Do not keep old derived diagnostic modules just because progress history
  references their results.
- Use semantic variation names for diagnostic paths, not chronological suffixes.

## Validation Summary

Latest focused validation:

- `uv run --group dev -- python -m orchestration.train_and_score` for the
  Level 2 multiplicative behavior-target diagnostic:
  `query_driven_behavior_segment_target_mult_level2_seed2532`.
- `uv run --group dev -- python -m orchestration.train_and_score` for the
  48/192/32-query multiplicative behavior-target diagnostic:
  `query_driven_behavior_segment_target_mult_scale48_query32_seed2533`.
- `uv run --group dev -- python -m orchestration.train_and_score` for the
  Level 3 64/256/40-query multiplicative behavior-target replay:
  `query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527`.
- `git diff --check` after progress-log condensation.

Recent implementation validation:

- `python3 -m py_compile` on touched source and test files after stale-code
  cleanup.
- `uv run --group dev -- ruff check` on touched source and test files after
  stale-code cleanup.
- `uv run --group dev -- pytest` on focused workload-profile, diagnostics,
  guardrail, protocol-gate, and model-feature suites after stale-code cleanup:
  `80 passed`.
- Focused canonical naming pytest set covering model factory/features,
  `QueryLocalUtility` training, learned segment-budget selector,
  protocol/causality gates, retained masks, learning target stage, benchmark
  profile/report regressions, and scoring metrics: `259 passed`.
- Earlier focused suites: workload/profile/property/guardrail `42 passed`,
  orchestration/scoring `178 passed`, benchmarking/report regression
  `40 passed`, and learning/orchestration payload `56 passed`.

Validation caveat:

- These validations prove implementation integrity and diagnostic consistency.
  They are not final scientific evidence of training coherence or success.
