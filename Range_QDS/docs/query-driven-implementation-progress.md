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
- Direct target rewrites for query-local turn-change and continuity have now
  failed smaller evidence levels and were reverted. Do not repeat that as a
  superficial behavior-label edit.
- The next useful work is semantic causality diagnosis: query-prior feature
  flow, behavior target/loss coupling, segment-score calibration, and
  workload/scoring compatibility if the prior/behavior signals remain
  incoherent under healthy gates.

Most recent rejected diagnostic:

- Checkpoint Phase 30 tested direct active-metric behavior-label variants:
  turn plus continuity anchors, turn-only behavior/replacement scoring, and
  turn-only behavior with restored replacement scoring. The Level 1 smokes only
  proved wiring. All Level 2 diagnostics failed required child gates. The
  best aggregate Level 2 score, the restored-replacement variant, beat uniform
  and Douglas-Peucker by a small amount but still failed predictability, target
  diffusion, workload signature, learning causality, and global sanity. The
  source was restored to `query_segment_local_behavior_utility`.

- Checkpoint Phases 17-20 tested a guarded component-local heads target and a
  coverage-shrink generator patch. The target branch passed small target
  diffusion checks after replacement capping, but its Level 3 replay failed
  workload health/signature, predictability, causality, global sanity, and
  baseline comparisons. The coverage-shrink path also failed required gates and
  was removed because it changed query geometry while keeping old footprint
  metadata. The component-local target mode and aggregate replacement
  sparsifier were both removed from source after the replacement-cap replay
  failed the Level 2 promotion gates.

- Checkpoint Phase 11 tested a temporary
  `query_local_utility_factorized_segment_gated_behavior` target that gated the
  behavior target by normalized provisional segment-budget signal. Artifacts:
  `artifacts/results/query_driven_segment_gated_behavior_level1_smoke_seed2605/example_run.json`,
  paired active-target smoke
  `artifacts/results/query_driven_active_target_level1_pair_seed2605/example_run.json`,
  and Level 2 diagnostic
  `artifacts/results/query_driven_segment_gated_behavior_level2_seed2606/example_run.json`.
  Level 2 passed workload stability and support overlap, but failed workload
  signature, target diffusion, predictability, learning causality, global
  sanity, and the uniform comparison. The temporary target mode was removed
  after rejection.

- Checkpoint Phase 9 tested existing
  `query_local_utility_sparse_head_bce_target_mode=window_max_normalized` at
  minimum strict Level 2 scale to probe sparse-head base-rate saturation.
  Artifact:
  `artifacts/results/query_driven_sparse_bce_window_norm_level2_seed2604/example_run.json`.
  It beat uniform but failed target diffusion, learning causality, global
  sanity, and Douglas-Peucker comparison; no-query-prior and shuffled-prior
  deltas stayed `0.0`, and no-behavior-head delta stayed only `0.0008535674`.

- Checkpoint Phase 8 tested a lower constant behavior floor in the factorized
  final target/model composition. The production code was reverted after the
  Level 2 diagnostic failed target diffusion, lost badly to uniform and
  Douglas-Peucker, and turned behavior/segment-budget ablations wrong-way.
  Artifacts:
  `artifacts/results/query_driven_low_floor_behavior_level1_smoke_seed2602/example_run.json`
  and
  `artifacts/results/query_driven_low_floor_behavior_level2_seed2603/example_run.json`.

- Checkpoint Phase 7 static diagnosis checked the behavior-head training
  contract against the current Level 3 artifact and source. The behavior target
  remains nonzero only on query-hit support, with
  `conditional_behavior_utility_training=masked_to_query_hit_points`.
  Widening the behavior mask to supervise zeros on every non-hit point was
  rejected before implementation: it would make the behavior head relearn
  query-hit support instead of query-local behavior value, weakening the
  guide's required head separation. The current blocker should stay classified
  as target/head semantic alignment, not a missing broad negative-supervision
  hack.

- `artifacts/results/query_driven_behavior_rank015_level3_scale64_query40_seed2527/example_run.json`
  tested behavior-head rank pressure alone at
  `query_local_utility_behavior_rank_loss_weight=0.15`, with the same healthy
  64/256/40 strict cell, seed `2527`, source-stratified split,
  `range_query_mix`, 4 train workload replicates, 5 epochs, and train-side
  marginal diagnostics.
- The replay kept workload stability, support overlap, target diffusion,
  workload signature, predictability, prior-predictive alignment, and global
  sanity green, but learning causality still failed.
- MLQDS QueryLocalUtility was `0.1426722765`, below the current reference
  `0.1431090566`.
- Failed causality children remained `shuffled_prior_fields_should_lose`,
  `without_query_prior_features_should_lose`, and
  `without_behavior_utility_head_should_lose`.
- No-query-prior and shuffled-prior deltas stayed exactly `0.0`. The
  no-behavior-head delta decreased to `0.0010929459`, still below the
  `0.005` materiality gate. Behavior-head prediction std stayed effectively
  flat at `0.002632` versus target std `0.166493`.
- Decision: rejected. Do not promote this behavior-rank auxiliary as a default
  or continue by simply increasing its weight. The failed path is semantic:
  behavior target/head outputs are still not aligned with retained marginal
  QueryLocalUtility, and query-prior features still do not affect retained
  masks.

- `artifacts/results/query_driven_prior_postcontext_level3_scale64_query40_seed2527/example_run.json`
  tested a post-context residual injection of the existing train-derived prior
  embedding into the `workload_blind_range` head representation.
- The replay kept workload stability, support overlap, target diffusion,
  workload signature, predictability, prior-predictive alignment, and global
  sanity green, but learning causality still failed.
- MLQDS QueryLocalUtility dropped to `0.1421332296` from the current reference
  `0.1431090566`.
- Failed causality children remained `shuffled_prior_fields_should_lose`,
  `without_query_prior_features_should_lose`, and
  `without_behavior_utility_head_should_lose`.
- Query-prior ablations became slightly anti-causal:
  shuffled-prior and no-query-prior deltas were both `-0.0005514228`.
  The behavior-head ablation also became wrong-way at `-0.0000960810`.
- Decision: rejected. The source default was restored. Do not continue by
  adding another generic prior residual or scalar prior amplification; the
  next fix needs to address target/head/selector semantics, especially the
  weak behavior head and segment-score marginal alignment.

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

## Checkpoint Phase 5 - Post-Context Prior Residual Diagnostic

Status: completed / rejected.

Scope:

- Diagnosed the current Level 3 learning-causality blocker under the healthy
  64/256/40 strict cell, using seed `2527`, source-stratified split,
  `range_query_mix`, 4 train workload replicates, 5 epochs, and train-side
  marginal diagnostics.
- Tested whether reusing the existing train-derived prior embedding after
  local/segment context would make query-prior fields materially affect the
  factorized heads and retained masks.
- Ran a Level 1 wiring smoke first:
  `artifacts/results/query_driven_prior_postcontext_level1_smoke_seed2601/example_run.json`.

Key result:

- Level 1 smoke completed and emitted gates/artifacts. It is implementation
  evidence only, not learning evidence.
- Strict Level 3 replay artifact:
  `artifacts/results/query_driven_prior_postcontext_level3_scale64_query40_seed2527/example_run.json`.
- Pre-causality gates stayed green and MLQDS still beat uniform and
  Douglas-Peucker, but the primary score regressed versus the current
  reference: `0.1421332296` vs `0.1431090566`.
- Learning causality still failed on the same children. Prior ablations moved
  from immaterial to weakly wrong-way, and the behavior-head ablation moved
  from below-threshold positive to wrong-way.
- Segment-score retained-marginal Spearman improved only from `-0.0812` to
  `-0.0414`; behavior-component alignment stayed negative at `-0.0263`, and
  behavior prediction std stayed nearly flat at `0.002639` versus target std
  `0.166493`.

Decision:

- Rejected and reverted. The diagnostic proved that a generic post-context
  prior residual is not the root fix.
- The current evidence boundary remains the active-target Level 3 replay:
  `query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527`.
- Next work should focus on behavior-head target/loss coupling and segment
  score semantics before more prior path amplification.

## Checkpoint Phase 6 - Behavior-Rank-Only Diagnostic

Status: completed / rejected.

Scope:

- Tested whether the weak `conditional_behavior_utility` head was mainly a
  missing training-pressure problem by enabling only
  `query_local_utility_behavior_rank_loss_weight=0.15`.
- Kept the same strict comparison cell as the current evidence boundary:
  seed `2527`, source-stratified split, `range_query_mix`, 64 ships, 256
  points, 40 requested queries, 4 train workload replicates, 5 epochs,
  `learned_segment_budget`, and train-side marginal diagnostics.

Key result:

- Artifact:
  `artifacts/results/query_driven_behavior_rank015_level3_scale64_query40_seed2527/example_run.json`.
- Pre-causality gates stayed green and MLQDS still beat uniform and
  Douglas-Peucker, but primary score regressed to `0.1426722765`.
- Learning causality still failed on query-prior and behavior-head dependence.
  Shuffled-prior and no-query-prior deltas stayed `0.0`; no-behavior-head
  delta was only `0.0010929459`.
- Behavior-head training fit improved only trivially: Kendall tau moved from
  `0.0251` to `0.0297`, top-5% mass recall from `0.1294` to `0.1380`, and
  prediction std stayed flat at `0.002632`.
- Exact retained-marginal behavior-component Spearman stayed negative at
  `-0.0466`; segment-score retained-marginal Spearman stayed negative at
  `-0.0594`.

Decision:

- Rejected. The existing behavior-rank auxiliary does not fix behavior-head
  causality or prior dependence in the current healthy strict cell.
- Next work should not be another generic behavior-rank weight sweep. Diagnose
  behavior target semantics and segment-score allocation semantics directly.

## Checkpoint Phase 7 - Behavior-Mask Contract Diagnosis

Status: completed / rejected before implementation.

Scope:

- Inspected the current target code and Level 3 artifact to decide whether the
  weak behavior head was caused by under-supervision from
  `conditional_behavior_utility_training=masked_to_query_hit_points`.
- Considered widening behavior-head supervision to every point by assigning
  zeros outside query-hit support.

Decision:

- Rejected before code changes. All-point zero supervision would make
  `conditional_behavior_utility` relearn query-hit support and blur the head
  separation required by the guide.
- The blocker remains target/head semantic alignment and weak retained-mask
  materiality, not a missing broad negative-supervision mask.

## Checkpoint Phase 8 - Low-Floor Behavior Formula Diagnostic

Status: completed / rejected and reverted.

Scope:

- Tested whether the behavior head was non-causal because the factorized final
  formula gave behavior too much constant floor:
  `q_hit * (0.5 + behavior) * (0.75 + 0.25 * replacement) + boundary`.
- Temporarily lowered the behavior floor while preserving the old maximum
  behavior multiplier, then restarted evidence at Level 1 before running a
  minimum strict Level 2 diagnostic.

Key result:

- Level 1 smoke artifact:
  `artifacts/results/query_driven_low_floor_behavior_level1_smoke_seed2602/example_run.json`.
  It completed and emitted the new formula metadata, but MLQDS lost to uniform;
  this was wiring evidence only.
- Level 2 artifact:
  `artifacts/results/query_driven_low_floor_behavior_level2_seed2603/example_run.json`.
- Level 2 workload stability and support overlap passed, but target diffusion,
  learning causality, and global sanity failed. MLQDS QueryLocalUtility was
  `0.0623695808`, versus uniform `0.0975959472` and Douglas-Peucker
  `0.0988619696`.
- The child gates moved in the wrong direction: no-behavior-head delta was
  `-0.0134858653`, no-segment-budget-head delta was `-0.0176702380`, and
  no-query-prior/shuffled-prior deltas remained `0.0`.
- Behavior-head fit remained low contrast: prediction std `0.0043339` versus
  target std `0.1738829`, with Kendall tau `0.0452`.

Decision:

- Rejected and reverted. Lowering the floor is not a root fix; it damages
  target diffusion and primary retained-mask quality before solving behavior
  causality.
- Current evidence boundary remains
  `query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527`.
- Next work should diagnose why behavior/prior signal is learned at too little
  contrast, not change the target formula to force behavior dependence.

## Checkpoint Phase 9 - Sparse-Head BCE Normalization Diagnostic

Status: completed / rejected as default.

Scope:

- Tested whether sparse head base-rate saturation was suppressing useful
  query-prior gradients and retained-mask causality.
- Used the existing diagnostic mode
  `query_local_utility_sparse_head_bce_target_mode=window_max_normalized`
  without changing production code.
- Ran a minimum strict Level 2 single-cell: seed `2604`, 24 ships, 128 points,
  3 synthetic route families, source-stratified split, `range_query_mix`, 4
  train workload replicates, 16 requested queries, 3 epochs, 5% compression.

Key result:

- Artifact:
  `artifacts/results/query_driven_sparse_bce_window_norm_level2_seed2604/example_run.json`.
- The run beat uniform on QueryLocalUtility (`0.0959336767` vs
  `0.0742409207`) but lost to Douglas-Peucker (`0.1178497144`).
- Workload stability and support overlap passed, but target diffusion,
  learning causality, and global sanity failed.
- Failed causality children included shuffled scores, shuffled prior fields,
  no-query-prior features, no-behavior-head, no-segment-budget-head, and
  prior-field-only matching the trained model.
- No-query-prior and shuffled-prior deltas remained `0.0`; no-behavior-head
  delta was only `0.0008535674`; no-segment-budget-head delta was
  `0.0041510034`, below the `0.005` materiality gate.
- Query-hit fit improved relative to the base-rate-saturation symptom
  (`query_hit_probability` top-5% mass recall `0.4624`), but behavior stayed
  low contrast: prediction std `0.0037949` versus target std `0.2005860`, with
  Kendall tau `0.0054`.

Decision:

- Rejected as a default and not promoted to Level 3. Window-normalized sparse
  BCE improves some sparse-head fit diagnostics, but it does not fix the
  target/behavior/prior causality blocker and fails child gates at Level 2.
- Next work should inspect why the learned heads have useful rank in the
  training target but still collapse to low-contrast outputs in retained-mask
  use, likely in loss weighting, head calibration, or selector score
  conversion.

## Checkpoint Phase 10 - Selector Routing Diagnosis

Status: completed / diagnostic.

Scope:

- Parsed the current Level 3 evidence boundary artifact:
  `artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/example_run.json`.
- Tested whether the weak behavior-head causality was mainly a selector-routing
  issue by comparing retained-mask changes, segment allocation changes, and
  component deltas across behavior, segment-budget, point-score allocation,
  and selector diagnostic ablations.

Key result:

- `without_behavior_utility_head` changed final point scores and retained
  masks, but left the selected segment set unchanged. Its QueryLocalUtility
  delta was only `0.0014985765`, below the `0.005` materiality gate.
- `without_segment_budget_head` caused a material delta of `0.0099825888` and
  changed substantially more retained decisions. The allocation-only segment
  ablation had nearly the same effect.
- `without_segment_budget_point_blend_only` stayed small at `0.0016363669`,
  while a point-score allocation diagnostic beat the primary mask by
  `0.0020213369`.
- Query-prior ablations remained exactly immaterial at this boundary:
  shuffled-prior and no-query-prior deltas were both `0.0`.

Decision:

- More selector weight sweeps are not the root fix. The segment-budget route
  has material control, but the behavior head has too little leverage in
  allocation and does not align enough with retained QueryLocalUtility.
- Next target work should be judged by whether it fixes behavior/prior
  causality under unchanged gates, not by whether it merely changes point
  scores inside already selected segments.

## Checkpoint Phase 11 - Segment-Gated Behavior Target Diagnostic

Status: completed / rejected and removed.

Scope:

- Temporarily added an experimental target mode,
  `query_local_utility_factorized_segment_gated_behavior`, that multiplied the
  active query-segment-local behavior target by
  `0.25 + 0.75 * normalized(provisional_final_score_segment_budget)`.
- Kept the active segment-budget target source as `active_final_score` with
  `top20_mean` aggregation and marked the variant `final_success_allowed=false`.
- Ran focused target/stage/guardrail tests before any experiment, then a Level
  1 smoke, a paired active-target Level 1 smoke at the same seed/scale, and a
  minimum strict Level 2 diagnostic.

Key result:

- Focused validation before the probe: py-compile under `uv`, ruff on touched
  files, and target/stage/guardrail pytest all passed.
- Level 1 variant artifact:
  `artifacts/results/query_driven_segment_gated_behavior_level1_smoke_seed2605/example_run.json`.
  It ran end-to-end and emitted the expected protocol and target metadata. Its
  target-diffusion warning was inherited from tiny-smoke query-hit support:
  the paired active-target smoke had the same final label support fraction
  `0.515625`.
- Level 2 artifact:
  `artifacts/results/query_driven_segment_gated_behavior_level2_seed2606/example_run.json`.
  MLQDS QueryLocalUtility was `0.0963151040`, below uniform
  `0.1203994768`, but above Douglas-Peucker `0.0705488790`.
- Level 2 passed workload stability and support overlap, and prior-predictive
  alignment passed, but workload signature failed on
  `point_hit_fraction_distribution_ks`, target diffusion failed because
  `replacement_representative_value` support was `0.5167`, predictability
  failed, learning causality failed, and global sanity failed.
- Causality remained wrong for the intended fix: no-behavior-head delta was
  `-0.0057099285`, no-query-prior and shuffled-prior deltas were both
  `-0.0038377814`, while no-segment-budget-head delta was material at
  `0.0188382640`.
- Behavior fit stayed weak for the purpose of retained-mask causality:
  behavior-head Kendall tau was `0.0579`, top-5% mass recall was `0.14695`,
  and final-score prediction std was only `4.3%` of target std.

Decision:

- Rejected. Segment-gating the behavior target did not fix behavior-head or
  query-prior causality and lost to uniform at Level 2.
- Removed the temporary target mode and its tests instead of leaving a stale
  experimental production path.
- Current evidence boundary remains the active-target Level 3 replay:
  `query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527`.

## Checkpoint Phase 12 - Head And Prior Contrast Diagnosis

Status: completed / blocker localization.

Scope:

- Re-read the active model, loss, feature, prior-field, score-conversion, and
  learned segment-budget paths after the segment-gated target rejection.
- Parsed the current Level 3 evidence boundary and nearby rejected diagnostics
  to localize whether the persistent no-prior/no-behavior failures originate
  in target support, output contrast, prior-feature visibility, or selector
  conversion.

Key result:

- The active Level 3 target is not missing scalar target composition: target
  formula diagnostics report exact composition (`factorized_target_formula_label_tau=1.0`).
- The learned outputs are severely low-contrast. In the active Level 3 replay,
  final-score prediction std is `0.0002196449` versus target std `0.0070392722`
  (`3.1%`). Behavior-head prediction std is `0.0026310128` versus target std
  `0.1664928645`, with Kendall tau only `0.0251`.
- Head bias initialization centers each sigmoid head on the empirical target
  mean, then the default auxiliary loss is mostly BCE against soft labels.
  That explains why heads stay calibrated near base rates unless a head has
  enough rank/listwise pressure and transferable features. Segment-budget has
  some material control; behavior does not.
- Query-prior ablations do exercise sampled prior fields, but the model-facing
  signal is tiny. In the active Level 3 replay, sampled prior mean is
  `0.0910`, but after disabling `route_density_prior` the model-facing prior
  mean is only `0.0116`; shuffling or zeroing priors changes head probabilities
  by roughly `1e-5` and leaves retained masks unchanged (`Jaccard=1.0`).
- Re-enabling `route_density_prior` and applying a square-root prior transform
  were already rejected in stricter historical diagnostics. They increased
  prior visibility, but did not fix the same causality children and in some
  cases moved deltas wrong-way.

Decision:

- The root blocker is not a missing behavior mask, a selector allocation weight,
  or a stale target branch. It is poor train-to-eval semantic transfer through
  low-contrast factorized outputs, plus a query-prior pathway that is visible
  in diagnostics but functionally ignored by the trained heads.
- Do not continue by re-enabling route density, adding another generic prior
  residual, raising behavior-rank weight blindly, or doing selector sweeps.
- The next admissible implementation checkpoint should be a small, isolated
  prior/head-contrast mechanism with Level 1 then Level 2 evidence, and it must
  be removed if it does not move no-prior and no-behavior causality under
  unchanged gates.

## Checkpoint Phase 13 - Prior-Head Contrast Diagnostic

Status: completed / rejected and removed.

Scope:

- Temporarily added a guarded train-time counterfactual prior-head contrast
  auxiliary. When explicitly enabled, training ran an extra zero-prior forward
  pass and pressured the `query_hit_probability`,
  `conditional_behavior_utility`, and `segment_budget_target` heads to carry
  information that disappeared when train-derived query-prior features were
  removed.
- The target, selector, metric, prior fields, and default config stayed
  unchanged. The diagnostic knob defaulted to disabled and was removed after
  the Level 2 decision.
- Ran Level 0 static/unit validation, a Level 1 smoke, a Level 2 diagnostic,
  and a paired same-seed default-control Level 2 diagnostic.

Key result:

- Level 1 artifact:
  `artifacts/results/query_driven_prior_head_contrast_level1_smoke_seed2607/example_run.json`.
  It proved only wiring: artifact metadata recorded
  `prior_head_contrast_loss_weight=0.15`, protocol flags were intact, and
  causality diagnostics ran. It remained tiny, gate-blocked evidence and makes
  no learning claim.
- Level 2 contrast artifact:
  `artifacts/results/query_driven_prior_head_contrast_level2_seed2608/example_run.json`.
  MLQDS QueryLocalUtility was `0.1036186648`, above uniform `0.0796331695`
  and Douglas-Peucker `0.0914024989`, but target diffusion, workload
  signature, learning causality, and global sanity all failed.
- Same-seed default-control artifact:
  `artifacts/results/query_driven_prior_head_contrast_level2_default_control_seed2608/example_run.json`.
  It shared the target-diffusion and workload-signature failures. The contrast
  run improved MLQDS by `0.0048993284` over control and moved shuffled-prior
  and no-query-prior deltas from `0.0` to `0.0031837274`, but this still missed
  the `0.005` causality gate.
- The intended behavior-head child did not improve. No-behavior-head delta
  moved wrong-way from control `0.0019755929` to contrast `-0.0003626264`.
  No-segment-budget remained wrong-way at `-0.0024169183`, shuffled scores
  still beat primary by `0.0009844345`, and prior-field-only remained too
  close to trained (`0.0047245682`, below the `0.005` separation gate).

Decision:

- Rejected. The mechanism made prior ablations nonzero, but not material, and
  it worsened the behavior-head blocker. It is not an acceptable path toward
  learned workload-blind causality.
- Removed the experimental loss, config/CLI plumbing, tests, and training
  diagnostics instead of leaving another stale disabled production branch.
- Current evidence boundary remains the active-target Level 3 replay:
  `query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527`.
- Next work should avoid another counterfactual-prior forcing loss unless it
  directly explains behavior-head directionality. The stronger signal is still
  target/loss/head semantic transfer, especially why behavior and
  segment-budget decisions can change masks but do not improve active
  QueryLocalUtility causality.

## Checkpoint Phase 14 - Query-Ship Local Heads Target Diagnostic

Status: completed / rejected.

Scope:

- Diagnosed the current Level 3 evidence boundary with the family/head transfer
  path diagnostic:
  `artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/family_transfer_path_diagnostic.json`.
- Tested the existing guarded
  `query_local_utility_factorized_query_ship_local_heads` target mode as a
  target-semantics probe only. This mode is explicitly marked experimental in
  artifacts with `final_success_allowed=false`; it is not final-candidate
  evidence.
- Ran a Level 1 smoke and then a Level 2 minimum strict diagnostic. No source
  code was changed.

Key result:

- Active Level 3 family/head transfer diagnosis says the current behavior head
  misorders active QueryLocalUtility retained-decision marginals. Overall
  active-metric behavior-head Spearman is `-0.0486`, retained-removal Spearman
  is `-0.2002`, and the behavior target is most aligned with
  `replacement_representative_value` (`0.5517` Spearman), not the active
  behavior metric.
- Level 1 query-ship-local-heads smoke:
  `artifacts/results/query_driven_query_ship_local_heads_level1_smoke_seed2609/example_run.json`.
  It verified only wiring: workload-blind protocol flags were intact, the
  target diagnostics reported `query_ship_local_presence_utility`,
  `query_ship_local_behavior_utility`, and `query_ship_local_heads_max_pool`,
  and `final_success_allowed=false` was present.
- Level 2 query-ship-local-heads artifact:
  `artifacts/results/query_driven_query_ship_local_heads_level2_seed2610/example_run.json`.
  MLQDS QueryLocalUtility was `0.0804790934`, below uniform
  `0.0966365965` and Douglas-Peucker `0.0883197473`.
- Level 2 failed target diffusion
  (`final_label_support_fraction_above_max`,
  `conditional_behavior_utility:support_fraction_above_max`), predictability,
  prior-predictive alignment, workload signature on the checkpoint-selection
  split, learning causality, and baseline comparisons.
- Causality remained directionally wrong: shuffled scores beat primary by
  `0.0043721967`, prior-field-only beat primary by `0.0074137916`, prior
  shuffling and removing query-prior features changed nothing, removing the
  behavior head improved the metric by `0.0014619883`, and only removing the
  segment-budget head produced a material positive delta (`0.0053794837`).
- The Level 2 family/head transfer diagnostic:
  `artifacts/results/query_driven_query_ship_local_heads_level2_seed2610/family_transfer_path_diagnostic.json`
  returned `reject_target_contract_before_transfer_work`. Behavior-head
  retained-marginal alignment stayed negative (overall Spearman `-0.0320`,
  retained-removal Spearman `-0.4326`) and the target became even more
  replacement-aligned (`0.8785` Spearman).

Decision:

- Rejected. The query-ship-local-heads target contract does not fix the active
  behavior-head blocker. It makes the target too diffuse, loses to both
  baselines at Level 2, fails prior alignment, and makes the behavior head
  harmful under active QueryLocalUtility.
- Do not promote this target mode or run Level 3 for it.
- Next work should build or diagnose a behavior target from active
  QueryLocalUtility retained-decision marginals or direct query-local
  interpolation/turn/continuity components, not from ship-query evidence or
  replacement-heavy proxies.

## Checkpoint Phase 15 - Marginal Teacher Transfer Diagnosis

Status: completed / diagnostic blocker.

Scope:

- Ran the existing active-metric selector marginal diagnostics on the current
  Level 3 evidence boundary:
  `query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527`.
- Goal was to test whether exact QueryLocalUtility retained-decision marginals
  are already usable as the next train-side behavior/segment target, or whether
  they are too selector-coupled and split-fragile.

Artifacts:

- `artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/selection_marginal_segment_calibration_diagnostic.json`
- `artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/selection_eval_segment_teacher_transfer_diagnostic.json`
- `artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/selection_segment_transfer_feature_admissibility_diagnostic.json`
- `artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/selector_marginal_calibration_diagnostic.json`

Key result:

- The selection-side separated marginal teacher is present and shape-viable:
  29 positive segment targets and 32 point targets on the checkpoint-selection
  trace. The eval trace has 32 positive segment targets and 32 point targets.
- The top selection/eval teacher targets do not transfer. Positive segment
  overlap is moderate overall (`20/29` selection positives overlap eval
  positives), but top-target overlap is `0.0` at the top `1%`, `5%`, and
  `10%` fractions. Selection-vs-eval teacher Spearman is `-0.1222`.
- Active pre-selection signals do not predict the teacher reliably. On the
  selection trace, segment-score Spearman with the segment teacher is
  `-0.0761`; on eval it is `-0.0099`. Segment allocation weight is also weak
  and sign-unstable (`-0.0676` selection, `0.0051` eval).
- The feature-admissibility diagnostic reports zero admissible transfer
  candidates. The only consistently positive signals are post-selection
  `segment_allocation_count` / `learned_count`, which are not admissible
  train/eval-time features. The guard-counter blend has attractive eval lift
  but is rejected because it explicitly subtracts the length-support guard.
- Selector marginal calibration still shows useful overall point-score
  alignment, but segment-score allocation is wrong for the exact marginal
  teacher: eval overall selector-score Spearman is `0.2881`, while segment-score
  Spearman is `-0.0812`; 21 top exact-marginal rows are low-ranked by
  segment score.
- The train-side marginal-teacher selector diagnostic in the active Level 3
  artifact is also negative. The pure train marginal teacher selector scored
  `0.1102741375`, below the primary train score by `0.0143729406`, despite a
  large retained-mask change (`Jaccard=0.1338`). Hybrid teacher blends at
  weights `0.10` and `0.25` were immaterial or worse (`+0.0003105753` and
  `-0.0014553550` teacher-minus-primary).

Decision:

- Do not turn the current exact retained-removal marginal teacher directly into
  a new production target. In its current form it is too selector-coupled,
  sparse, and split-fragile.
- The next implementation candidate needs a broader query-local component
  target that is computed from train workloads before selection, especially
  from direct interpolation, turn-change, and continuity evidence. A retained
  marginal teacher may be useful only as a diagnostic or calibration audit, not
  as the primary target semantics.

## Checkpoint Phase 16 - Component Compatibility Check

Status: completed / no metric-loosening action.

Scope:

- Ran the existing workload/component compatibility diagnostic on the active
  Level 3 evidence boundary:
  `artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/workload_component_compatibility_diagnostic.json`.

Key result:

- The active Level 3 candidate already beats Douglas-Peucker on the primary
  QueryLocalUtility score by `0.0277824328`.
- A post-hoc behavior-heavy component weighting would increase that diagnostic
  delta slightly to `0.0300194168`, but the diagnostic labels the masking risk
  `high`.
- No persistent negative query-local components or unresolved blocker families
  were reported under the blocker-preserving profile candidate.

Decision:

- Do not change QueryLocalUtility weights to hide learning-causality failures.
  The blocker remains learned causality and target/head transfer, not a metric
  threshold problem.
- Next implementation should stay on target semantics: build a train-workload
  target from direct query-local interpolation, turn-change, and continuity
  support, then restart at Level 1.

## Checkpoint Phase 17 - Component-Local Heads Target Restart

Hypothesis:

- A viable next target must use train range workloads and direct
  `QueryLocalUtility` component mechanics, not ship-query evidence,
  replacement-value behavior credit, or retained-marginal teachers.

Implementation:

- Added guarded experimental target mode
  `query_local_utility_factorized_query_component_local_heads`.
- Added direct component-local behavior support from query-hit-conditioned
  turn-change plus sparse query-run interpolation/continuity anchors.
- Added a guarded direct scalar formula:
  `query_hit_times_direct_query_local_component_behavior_plus_boundary_event`.
- Kept `final_success_allowed=false`; this mode is not eligible for final
  acceptance until promoted by strict evidence.

Evidence:

- Level 0 validation used the `uv` environment:
  `uv run --group dev -- python` is Python `3.14.5`.
- Focused compile, ruff, and target/orchestration pytest passed after the new
  target wiring (`25 passed`, later `26 passed` after replacement-cap tests).
- Initial Level 1 component-local smoke failed target diffusion because
  query-run anchors made behavior support too broad. The overlapping-anchor
  path was capped with top-45% valid support.
- The next Level 1 smoke
  `query_driven_query_component_local_heads_sparse_level1_smoke_seed2612`
  moved the failure to final-label support (`0.5234375 > 0.5`) because the
  active scalar formula still gave every query-hit point baseline mass.
- After switching the experimental scalar to direct component behavior plus
  boundary, Level 1
  `query_driven_query_component_local_heads_direct_score_level1_smoke_seed2613`
  passed target diffusion:
  final-label support `0.23046875`, behavior support `0.4573643411`,
  guarded metadata present, and `final_success_allowed=false`.

Decision:

- Direct component-local scalar fixed the broad query-hit baseline problem.
- No learning claim is made from Level 1.

## Checkpoint Phase 18 - Replacement Aggregate Sparsifier

Hypothesis:

- The replacement representative head can still become too diffuse after
  overlapping query-level supports are aggregated, even when each query uses
  the intended keep fraction.

Evidence:

- Level 2 direct-score run
  `query_driven_query_component_local_heads_direct_score_level2_seed2614`
  passed workload stability and support overlap, but failed target diffusion:
  `replacement_representative_value` support was `0.51875` against the `0.50`
  max.

Implementation:

- Enforced the existing replacement keep fraction after query aggregation with
  `replacement_representative_aggregate_sparsifier:
  top35_query_hit_support`.
- Added a focused overlapping-query unit test for the aggregate cap.

Restarted evidence:

- Level 1
  `query_driven_query_component_local_heads_replacement_capped_level1_smoke_seed2615`
  passed target diffusion:
  final-label support `0.19921875`, replacement support `0.16796875`.
- Level 2
  `query_driven_query_component_local_heads_replacement_capped_level2_seed2616`
  passed workload stability, support overlap, and target diffusion.
  MLQDS beat uniform and DouglasPeucker on QueryLocalUtility:
  `0.0962736705` vs `0.0666538885` and `0.0777492372`.

Remaining Level 2 blockers:

- Workload signature failed on selection vs eval:
  point-hit-fraction KS `0.2424242424 > 0.2`.
- Predictability and prior predictive alignment failed.
- Learning causality failed; shuffled/prior/no-prior/no-behavior/no-segment
  controls were not sufficiently worse.
- Global sanity failed length preservation.

Decision:

- Do not tune model or selector from this Level 2 result. Per guide, increase
  scale or fix workload generation/signature first.

## Checkpoint Phase 19 - Level 3 Scale Diagnosis for Capped Target

Hypothesis:

- The Level 2 selection-signature failure might be scale-sensitive; the guide's
  supported `64/256` synthetic Level 3 cell with a 48-query floor should
  distinguish workload/signature failure from target failure.

Run:

- `query_driven_query_component_local_heads_replacement_capped_level3_scale64_query48_seed2617`
  with 64 ships, 256 points/ship, 4 synthetic families, 48 accepted-query
  floor, 4 train workload replicates, 5 epochs, source-stratified split, and
  `range_query_mix`.

Gate results:

- Passed: support overlap, target diffusion, prior predictive alignment.
- Failed: workload stability, workload signature, predictability, learning
  causality, global sanity, and QueryLocalUtility baseline comparisons.
- Workload stability failed on `train_r2` rejection pressure.
- Workload signature failed on `train_r3` point-hit-fraction KS:
  `0.2291666667 > 0.2`.
- Predictability gate still failed top-k lift checks at 1% and 2%.
- Learning causality failed because shuffled scores, shuffled priors,
  no-query-prior features, no-behavior head, no-segment-budget head, and
  prior-only controls were not materially worse.
- MLQDS lost QueryLocalUtility to uniform and DouglasPeucker:
  `0.1123046587` vs `0.1269927281` and `0.1198890654`.

Component diagnosis:

- MLQDS beat uniform/DP slightly on query point recall, but lost enough on
  interpolation, continuity, global shape, and length to lose the weighted
  QueryLocalUtility score.
- Prior-only and shuffled-score controls beat the trained primary, so this is
  not acceptable learning evidence.

Decision:

- Stop target promotion. The capped component-local target cleared target
  diffusion, but Level 3 blocks on workload health/signature, predictability,
  causality, global sanity, and baselines.
- Next admissible work is targeted diagnosis of workload rejection/signature and
  why prior-only/shuffled controls beat the trained model. Do not run the final
  grid.

## Checkpoint Phase 20 - Rejected Coverage-Shrink Diagnostic

Hypothesis:

- The `range_query_mix` Level 3 workload-stability failure might be caused by
  otherwise valid profile queries hitting the coverage guard after the accepted
  query floor was already near the coverage target.

Diagnostic implementation tested and removed:

- Tested a profile-only coverage-guard shrink path that tried smaller versions
  of an otherwise accepted range query before recording a coverage-overshoot
  rejection.
- Removed the path after diagnosis. It changed real query geometry while
  retaining the original footprint-family metadata, which can make workload
  signature evidence misleading. The probe also did not clear required gates.

Evidence:

- Coarse shrink artifact:
  `query_driven_component_local_coverage_shrink_level3_scale64_query48_seed2617`.
  Predictability passed, but workload stability still failed on `train_r2`.
  Most shrink attempts failed `too_low_point_hit_fraction`, so the issue was
  not merely a missing coverage-refinement path.
- Dense shrink artifact:
  `query_driven_component_local_coverage_shrink_dense_level3_scale64_query48_seed2617`.
  It passed support overlap, target diffusion, predictability, and
  prior-predictive alignment, but failed workload stability, workload
  signature, learning causality, global sanity, and QueryLocalUtility baseline
  comparisons.
- Dense gate details: `train_r2` failed rejection pressure and coverage-guard
  pressure; `train_r3` failed point-hit-fraction signature KS
  `0.2291666667 > 0.2`; MLQDS QueryLocalUtility was `0.1140967925` versus
  uniform `0.1294115718` and Douglas-Peucker `0.1198775480`.
- Dense shrink success was too small to matter: `train_r2` had only 2 accepted
  shrink attempts out of 5453 shrink checks, with `too_low_point_hit_fraction`
  dominating the rejection reasons.

Decision:

- Do not keep the shrink path. It is not a root fix, and it risks hiding
  footprint-family drift behind unchanged metadata.
- Current blocker remains workload generation/profile feasibility plus
  learned-causality failure. Do not run the final grid.

## Checkpoint Phase 21 - Failed Experimental Target Cleanup

Hypothesis:

- The failed component-local heads branch should not remain as callable
  production/test code after the Level 3 evidence rejected it.

Implementation:

- Removed `query_local_utility_factorized_query_component_local_heads`, its
  direct component-local final-score helper, behavior-anchor helper, and the
  guarded stage/unit tests for that target mode.

Evidence:

- Source/test search found no remaining `query_component_local` or
  `component_local_heads` references outside the progress log.
- `uv run --group dev -- python -m py_compile` on touched target files passed.
- `uv run --group dev -- ruff check` on touched target files passed.
- Focused target/orchestration pytest passed: `24 passed`.

Decision:

- The current source no longer exposes the failed component-local branch.
- The active evidence boundary remains the healthy default Level 3 replay
  `query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527`.
- Next admissible work remains causality diagnosis under the healthy default
  cell, especially behavior-head semantic alignment and query-prior materiality.

## Checkpoint Phase 22 - Replacement Aggregate Cap Rejection

Hypothesis:

- The aggregate replacement cap might be a root-scoped sparse-target fix worth
  keeping in the active target after the component-local branch was removed.

Evidence:

- Level 1 active-target smoke artifact:
  `query_driven_replacement_aggregate_cap_active_level1_smoke_seed2620`.
  It ran end to end, but failed workload stability on the tiny
  checkpoint-selection split (`0` accepted selection queries) and failed target
  diffusion on final-label support (`0.55`). This is implementation evidence
  only, not learning evidence.
- Level 2 active-target diagnostic artifact:
  `query_driven_replacement_aggregate_cap_active_level2_seed2621`.
  It passed workload stability, support overlap, target diffusion,
  predictability, and prior-predictive alignment.
- The same Level 2 run failed workload signature, learning causality, global
  sanity, and baseline comparisons. Point-hit-fraction KS failed for train
  (`0.2402402402 > 0.20`) and selection (`0.3570412518 > 0.20`). MLQDS
  QueryLocalUtility was `0.0807711803`, below uniform `0.0994709462` and
  Douglas-Peucker `0.0923894335`.
- Causality was worse than the active healthy boundary: shuffled scores,
  untrained, no-query-prior, shuffled-prior, no-behavior-head,
  no-segment-budget-head, and prior-only separation all failed; no-behavior and
  no-segment-budget deltas were wrong-way.

Decision:

- Do not keep the aggregate replacement cap as an active target change. It is
  not promoted by the Level 2 replay and would move the source away from the
  known healthy default Level 3 evidence boundary.
- Removed the cap and its test. The source target semantics are back to the
  current default stack.
- Do not run the final grid.

## Checkpoint Phase 23 - Healthy-Cell Prior And Selector Causality Diagnosis

Hypothesis:

- Under the healthy default Level 3 cell, query-prior ablations fail because
  train-derived prior fields either do not reach the model with material
  contrast, or they reach the model but do not move selector rankings. The
  behavior-head failure should be diagnosed against exact retained-marginal
  utility before changing target or selector code.

Artifact:

- `artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/example_run.json`

Gate context:

- This is the current healthy evidence-boundary replay: workload stability,
  support overlap, target diffusion, workload signature, predictability,
  prior-predictive alignment, global sanity, and baseline comparisons pass.
  Learning causality fails only on shuffled-prior, no-query-prior, and
  no-behavior-head children.

Findings:

- Prior fields are predictive before model use. Aggregate prior predictability
  has Spearman `0.2617`, lift@5% `1.5534`, and PR-AUC lift `1.3574`.
  Individual channels are stronger: spatial query-hit Spearman `0.4667`,
  endpoint likelihood Spearman `0.4358`, and crossing likelihood Spearman
  `0.4388`.
- The model-facing prior path is too weak to affect masks. Zeroing all query
  priors changes sampled prior fields by mean absolute `0.0910`, but after the
  active model transform and disabled route-density channel the useful
  model-input prior delta is only `0.0116`. Head probability deltas are around
  `1e-5`; final selector-score mean absolute delta is `0.0007636` against
  selector-score std `0.2774`; top-k Jaccard at the retained count is `1.0`.
- The selector has enough learned-controlled slots, so this is not a slot-count
  excuse: 132 learned-controlled retained slots, fraction `0.8462`.
- Segment allocation is score-driven but the score is wrong for the active
  marginal objective. Eval segment score span is only `0.0356`, segment-budget
  entropy normalized is `0.9869`, and all 96 segments receive learned budget.
  Segment score to allocation Spearman is `0.8386`, but segment score to exact
  retained-marginal QueryLocalUtility Spearman is `-0.0812`.
- Behavior head alignment is also wrong: behavior multiplier to exact
  retained-marginal utility Spearman is `-0.0486`, and no-behavior-head changes
  26 retained decisions but improves QueryLocalUtility by only `0.0015`, below
  the `0.005` materiality gate.
- The strongest query-free retained-marginal proxy in the trace is endpoint
  support, not the learned behavior or segment-budget heads: endpoint proxy
  Spearman is `0.5497`, while query-free path-length support is slightly
  negative (`-0.0187`).

Decision:

- Do not tune architecture, selector floors, or prior scalar gain from this
  alone. The current failure is semantic/ranking alignment: useful train-derived
  priors exist, but model-facing prior contrast is too small to alter retained
  ranks, and the segment/behavior heads misorder the exact active marginal
  utility.
- Next admissible implementation should target active-metric segment/behavior
  alignment under unchanged gates, with a Level 1 smoke then Level 2 minimum
  strict replay. A candidate must move no-query-prior and no-behavior causality
  materially; score-only or fit-only improvements are insufficient.

## Checkpoint Phase 24 - Query-Run Evidence Behavior Target Probe

Hypothesis:

- A sparse in-query run-evidence behavior target might align the behavior head
  with active QueryLocalUtility interpolation/continuity support better than the
  current behavior-change-only target, without changing the accepted default
  target or adding selector scaffolding.

Implementation:

- Added a guarded experimental target mode locally, with
  `final_success_allowed=false`, that changed only the behavior/derived final
  score path. The accepted `query_local_utility_factorized` path was not
  changed.
- First Level 1 smoke (`query_driven_query_run_evidence_behavior_level1_smoke_seed2622`)
  exposed a target-diffusion defect in the new behavior target:
  `conditional_behavior_utility` support `0.86875` was above the `0.5` max and
  top-5 label mass `0.0788` was below `0.1`.
- Revised the experimental mode with an aggregate top-35% sparsifier for the
  query-run evidence. The second Level 1 smoke
  (`query_driven_query_run_evidence_behavior_level1_smoke_seed2623`) ran end to
  end, had support overlap pass, and had target diffusion pass. It remains
  implementation evidence only; the final workload gate failed on the expected
  tiny-smoke replicate count and global/causality gates were not meaningful.

Level 2 diagnostic:

- Run:
  `query_driven_query_run_evidence_behavior_level2_seed2624`
  (`24` ships, `128` points/ship, `3` route families, `24` requested queries,
  `4` train workload replicates, `3` epochs).
- Workload stability passed and support overlap passed.
- Target diffusion failed on unchanged `replacement_representative_value`
  support: `0.54375 > 0.5`. The new behavior head itself passed diffusion:
  support `0.2653`, top-5 label mass `0.2365`.
- Predictability failed. Prior-predictive alignment failed on query-hit
  Spearman, query-hit lift@5%, and segment-budget lift@5%.
- Global sanity failed on length preservation: `0.5919 < 0.75`.
- Learning causality failed all required ablation checks. Learned-controlled
  slots were present (`30`, fraction `0.7143`), but deltas were wrong:
  shuffled score `-0.0172`, untrained `-0.0143`, no query prior `-0.000117`,
  shuffled prior `-0.000117`, no behavior `-0.000620`, no segment `-0.0129`.
- MLQDS also lost baselines on QueryLocalUtility: MLQDS `0.06478`, uniform
  `0.08370`, DouglasPeucker `0.10690`.

Decision:

- Reject this target mode. It improved the specific behavior-target diffusion
  issue after sparsification, but it did not survive Level 2: target diffusion,
  predictability, global sanity, causality, and baselines all block it.
- Removed the experimental target mode and focused tests from source/test code.
  No `query_run_evidence` symbols remain outside this progress log and run
  artifacts.
- Do not run Level 3 or the final grid for this candidate.

## Checkpoint Phase 25 - Replacement Boundary And Current-Cell Causality Diagnosis

Hypothesis:

- The failed replacement-diffusion child gate in the rejected query-run
  evidence Level 2 probe might be a scale/seed boundary issue rather than a
  candidate-specific target defect. Before changing source again, diagnose the
  gate under an active-default control and re-anchor causality conclusions to
  the current healthy Level 3 evidence boundary.

Evidence:

- Runtime check: `uv run --group dev -- python --version` reports Python
  `3.14.5`. The checkpoint stream should not cite bare local Python runtimes.
- Replacement target diffusion is scale-sensitive. Healthy 64/256/40 strict
  active-default artifacts pass with replacement support around `0.39` to
  `0.41`, while several minimum 24/128 Level 2 artifacts exceed the `0.50`
  support cap.
- Active-default same-seed Level 2 control:
  `query_driven_active_default_level2_control_seed2624`.
  This reproduced the rejected candidate's unchanged replacement failure:
  `replacement_representative_value` support `0.54375 > 0.50`, with top-5 mass
  `0.1220`. It also failed predictability, prior alignment, global sanity, and
  all required learning-causality children; MLQDS QueryLocalUtility was
  `0.062195`, below uniform `0.083703` and Douglas-Peucker `0.106902`.
- Therefore the replacement failure at this probe scale is not evidence for
  changing the active replacement target. The correct boundary remains the
  healthy 64/256/40 strict current-default cell.

Current Level 3 boundary diagnosis:

- Artifact:
  `artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/example_run.json`.
- Pre-causality gates are still green and MLQDS beats uniform and
  Douglas-Peucker: `0.1431090566` vs `0.1247681518` and `0.1153266238`.
- Learning causality still fails only on query-prior and behavior-head
  materiality: shuffled-prior delta `0.0`, no-query-prior delta `0.0`,
  no-behavior delta `0.0014985765 < 0.005`.
- Prior fields reach sampling but not decisions. Shuffling priors changes
  sampled prior values by mean absolute `0.08946`, but model-facing useful
  prior input changes only `0.01142`; head probability delta is
  `0.00000967`, final selector-score delta is `0.000735` against selector-score
  std `0.277365`, and the retained mask is unchanged.
- Head outputs remain low-contrast. Factorized final score prediction std is
  only `0.0312` of target std. The behavior head has target std `0.166493` but
  prediction std `0.002631` and Kendall tau `0.0251`.
- Eval retained-marginal alignment is mixed and does not justify a train-side
  marginal target. Overall eval selector-score Spearman is `0.2881` and
  query-hit probability Spearman is `0.2806`, but segment-score Spearman is
  `-0.0812` and behavior-probability Spearman is `-0.0486`. On the
  learned-controllable subset, behavior is effectively non-causal
  (Spearman `-0.0103`), query-hit is nearly flat (`0.0224`), segment-score is
  weak (`0.0803`), while replacement (`0.4465`) and path support (`0.2383`)
  are the stronger learned-subset signals.
- Train-side separated retained-marginal teacher is diagnostic-only and not a
  root fix. Teacher-only selection scored `0.110274` on the train diagnostic,
  `0.014373` below the primary. A 10% blend gave only a tiny train diagnostic
  gain (`0.000311`), while a 25% blend lost (`0.001455`).

Decision:

- No source change in this checkpoint.
- Do not tune another generic prior residual, scalar prior amplification,
  sparse-BCE/rank setting, selector floor, or replacement sparsifier from the
  failed Level 2 boundary. Those paths have already failed the child gates or
  are not supported by the current boundary diagnostics.
- Next admissible implementation needs to address active-metric behavior and
  segment semantics at the target/loss/head boundary, then restart at Level 1
  and promote only if Level 2 gates pass. A score-only, fit-only, or tiny-probe
  improvement is not learning evidence.
- Do not run the final grid.

## Checkpoint Phase 26 - Target-Semantics Cleanup

Hypothesis:

- The active blocker is misaligned behavior/segment supervision and low
  model-facing prior contrast, not a runtime issue or a replacement-target
  sparsity defect. Before adding any new candidate, remove stale target modes
  that the guide/progress log already rejected so they cannot be mistaken for
  admissible final-candidate paths.

Evidence:

- Runtime check remains explicit: `uv run --group dev -- python --version`
  reports Python `3.14.5`.
- Current Level 3 target-side diagnostics show the active behavior target is
  still closer to replacement than to active query-local evidence:
  Spearman with replacement representative value `0.5517`, final score
  `0.3056`, path-length support `0.2719`, query-hit probability `0.1366`,
  segment-budget target `0.0116`, and ship-query evidence `-0.0304`.
- Behavior candidate diagnostics do not justify a quick gated-target swap:
  replacement/segment-gated variants remain weak or negative against
  ship-query evidence.
- Segment ship-presence candidates are diagnostic proxies only. The active
  segment budget has Spearman `0.529` with ship-query evidence and a
  ship-presence candidate reaches `0.625`, but the previous query-ship-local
  training target failed Level 2 on target diffusion, predictability,
  prior alignment, workload signature, causality, and baselines. That path is
  rejected, not promoted.

Implementation:

- Removed the rejected
  `query_local_utility_factorized_segment_budget_query_ship_max_pool` and
  `query_local_utility_factorized_query_ship_local_heads` target modes from
  `learning/targets/query_local_utility.py`.
- Removed their helper signals, constants, segment-budget branches, composed
  head branch, and tests that treated them as guarded-but-callable
  experimental modes.
- Kept only `query_local_utility_factorized` in
  `QUERY_LOCAL_UTILITY_TARGET_MODES`. Tests now assert the rejected strings are
  rejected by orchestration mode validation, not silently accepted with
  `final_success_allowed=false`.
- Updated the diagnostic fixture to describe the active target path rather
  than the rejected query-ship max-pool path.

Decision:

- This is cleanup and protocol hygiene, not a learning-success claim.
- No final grid and no new training probe were run.
- Next admissible implementation still needs a root fix at the active
  target/loss/head boundary, followed by Level 1 then Level 2 evidence under
  unchanged gates.

## Checkpoint Phase 27 - Behavior Loss/Head Diagnosis

Hypothesis:

- The behavior head's low prediction contrast might be caused by missing
  training pressure rather than target semantics.

Targeted evidence:

- Current Level 3 loss config uses behavior-rank pressure `0.0`, sparse-head
  rank pressure `0.0`, and raw sparse-head BCE targets. The behavior head is
  initialized to the target mean (`0.08434`) and remains nearly flat:
  prediction std `0.002631` versus target std `0.166493`, Kendall tau
  `0.0251`, top-5% target-mass recall `0.1294`.
- The existing strict behavior-rank diagnostic already tested the simple
  missing-pressure hypothesis:
  `query_driven_behavior_rank015_level3_scale64_query40_seed2527`.
  It regressed primary QueryLocalUtility from `0.1431090566` to
  `0.1426722765`, left query-prior deltas at `0.0`, reduced no-behavior
  materiality from `0.0014986` to `0.0010929`, and kept behavior retained-
  marginal Spearman negative (`-0.0466`).
- Behavior-rank pressure improved fit only trivially: tau `0.0251 -> 0.0297`,
  top-5 mass recall `0.1294 -> 0.1380`, and prediction std stayed effectively
  unchanged (`0.002631 -> 0.002632`).
- The guide explicitly keeps behavior-rank disabled by default because better
  head fit can still worsen retained-mask causality.

Decision:

- Do not change loss weights or run another generic behavior-rank/sparse-head
  sweep. That hypothesis has already failed the strict healthy cell.
- The remaining blocker is not generic head pressure. It is target semantics
  and model-facing prior/segment alignment: the current behavior target is
  replacement-aligned, weak against active retained marginals, and not made
  causal by stronger behavior-rank loss.
- No source change and no new probe in this checkpoint.

## Checkpoint Phase 28 - Prior Transform Rejection

Hypothesis:

- The model-facing prior path might be weak because raw train-derived prior
  probabilities are too small after route-density removal. A monotone
  square-root transform would preserve semantic zero while giving useful
  query-hit, endpoint/crossing, and behavior-prior channels more numeric
  contrast.

Targeted evidence:

- The current strict artifact shows the attenuation chain clearly:
  shuffled-prior sampled features change by mean absolute `0.08946`, but
  model-input prior features change only `0.01142`; head probabilities move by
  only `0.00000967`, final selector scores by `0.000735`, and the retained mask
  is unchanged.
- Route density should stay disabled. In the same artifact, route-density
  prior Spearman with the segment target is only `0.0233`; restoring it would
  amplify the wrong channel.
- A current artifact already tested the square-root model-prior transform:
  `query_driven_prior_sqrt_level3_scale64_query40_seed2527`.
  It increased model-input prior delta to `0.08666`, head probability delta to
  `0.00009596`, score delta to `0.00423`, and did change the retained mask.
- Despite that, the strict run failed the same causality children:
  shuffled-prior, no-query-prior, and no-behavior. The query-prior deltas were
  wrong-way (`-0.0003197` shuffled prior, `-0.0003081` no-query-prior), the
  no-behavior delta was also wrong-way (`-0.0005148`), and primary
  QueryLocalUtility regressed to `0.1396786660` versus the current
  `0.1431090566`.

Decision:

- Reject the square-root prior transform. It proves numeric prior contrast can
  be made visible, but visible contrast is not useful causal learning under
  the active target/selector semantics.
- Reverted the temporary source/test edit before keeping it. No new probe was
  run because the existing Level 3 strict artifact is stronger evidence than a
  repeated Level 1 smoke.
- Next work should not be generic prior amplification. It needs to explain why
  stronger prior contrast misranks retained decisions, likely by comparing
  model/prior component rankings against exact retained-marginal
  `QueryLocalUtility` rather than against target fit alone.

## Checkpoint Phase 29 - Prior Contrast Tradeoff Diagnosis

Hypothesis:

- The square-root prior transform failed because stronger prior contrast moved
  retained decisions toward the wrong active `QueryLocalUtility` components,
  not because it failed to affect the model.

Targeted evidence:

- Current identity-prior strict replay:
  `query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527`.
  Exact retained-marginal alignment is mixed: raw-score Spearman `0.2779`,
  selector-score Spearman `0.2881`, query-hit probability Spearman `0.2806`,
  but segment-score Spearman `-0.0812` and behavior-probability Spearman
  `-0.0486`.
- Square-root prior replay:
  `query_driven_prior_sqrt_level3_scale64_query40_seed2527`.
  Raw-score alignment improved to `0.3077`, selector-score to `0.3138`,
  segment-score to `0.1757`, and behavior-probability to `0.0475`.
  That confirms prior contrast can improve some retained-marginal ranking
  diagnostics.
- It still lowered primary QueryLocalUtility (`0.1396786660` vs current
  `0.1431090566`) and failed causality. The no-query-prior ablation changed
  12 retained decisions but hurt the primary by `0.0003081`; shuffled-prior
  changed 10 retained decisions and hurt by `0.0003197`.
- Component tradeoff explains the failure. No-query-prior primary-minus-
  ablation gained interpolation (`+0.0002882` weighted) and a tiny global-shape
  amount, but lost more on turn-change (`-0.0003311` weighted), continuity
  (`-0.0002702` weighted), and length (`-0.0000079` weighted). Shuffled-prior
  had the same pattern.

Decision:

- Do not add a prior-channel transform or scalar prior amplification. It makes
  masks move, but the moved decisions are not component-correct.
- The next source fix should target direct behavior/segment semantics for
  query-local turn-change and continuity, while preserving interpolation. A
  useful candidate must improve exact retained-marginal component tradeoffs,
  not only target fit, head dispersion, or segment-score Spearman.
- No source change and no new probe in this checkpoint.

## Checkpoint Phase 30 - Direct Turn/Continuity Target Rejection

Hypothesis:

- The active behavior target may be too replacement-aligned and may understate
  the active `QueryLocalUtility` turn-change and continuity components. A
  direct behavior target built from active-metric turn/continuity semantics
  might improve causality more than generic prior amplification or loss
  pressure.

Implementation attempts:

- Temporarily tested a turn-plus-continuity behavior target.
- Narrowed that to active-metric turn-change only after continuity anchors made
  labels too broad.
- Restored the old replacement scoring while keeping the turn-only behavior
  label after replacement support remained too diffuse.
- Reverted all three target variants after the smaller evidence levels failed.
  The active source is again
  `conditional_behavior_target_variant=query_segment_local_behavior_utility`.

Evidence:

- Level 1 smokes
  `query_driven_turn_continuity_behavior_level1_smoke_seed2626`,
  `query_driven_turn_change_behavior_level1_smoke_seed2628`, and
  `query_driven_turn_change_replacement_restored_level1_smoke_seed2630`
  completed and emitted artifacts. They are wiring evidence only; they are not
  learning evidence.
- Turn plus continuity Level 2
  `query_driven_turn_continuity_behavior_level2_seed2627` failed target
  diffusion, workload signature, learning causality, and global sanity. MLQDS
  lost to uniform and Douglas-Peucker on QueryLocalUtility (`0.0709957117`
  versus `0.0802865926` and `0.1002105214`). Behavior support was too broad:
  `conditional_behavior_utility` support at `gt_0.01` was `0.8333333333`.
- Turn-only Level 2
  `query_driven_turn_change_behavior_level2_seed2629` reduced behavior support
  to `0.2837345004`, but still failed predictability, prior alignment, target
  diffusion, workload signature, learning causality, and global sanity. MLQDS
  was `0.0783734133`, barely above uniform `0.0762835828` and below
  Douglas-Peucker `0.0921610819`. Query-prior deltas stayed `0.0`, and
  no-behavior was wrong-way at `-0.0000943255`.
- Turn-only behavior with restored replacement Level 2
  `query_driven_turn_change_replacement_restored_level2_seed2631` had the best
  aggregate score in this rejected group: MLQDS `0.0934793306`, uniform
  `0.0919107575`, Douglas-Peucker `0.0722754241`. That is still not
  promotable. It failed predictability, target diffusion, workload signature,
  learning causality, and global sanity. Replacement support was too broad
  (`0.5276041667` at `gt_0.05`), behavior alignment was negative against final
  score (`-0.0391017924`), query hit (`-0.0953541326`), replacement
  (`-0.0387882204`), and segment budget (`-0.2307628259`), and training fit
  was wrong-way for the final factorized target (`tau=-0.0850678733`).
- The restored-replacement causality children show the root failure clearly:
  shuffled scores lost by only `0.0030855237`, untrained lost by
  `0.0049875458`, prior-only beat trained by `0.0249382960`, shuffled-prior
  and no-query-prior deltas stayed `0.0`, no-behavior delta was only
  `0.0008223167`, and no-segment remained the only material head ablation at
  `0.0091297256`.

Decision:

- Rejected. Direct continuity anchors are too diffuse under overlapping range
  workloads, and active-metric turn-only behavior labels do not transfer into
  causal retained-mask decisions.
- Do not promote or leave these variants as guarded production paths. They were
  reverted instead of kept as compatibility modes.
- The next admissible work should diagnose why the prior-only and segment
  routes can dominate or beat trained heads under these target variants. Focus
  on target/head/selector semantics and retained-marginal calibration, not on
  more large temporal scaffolding, scalar prior amplification, or another
  behavior-label broadening pass.
- No final grid was run, and no final success claim is allowed.

## Checkpoint Phase 31 - Current Causality Path Diagnosis

Hypothesis:

- After rejecting direct behavior-label rewrites, the current blocker may be a
  score-to-selector calibration problem: query-prior inputs might be present
  but too weak to affect retained masks, while segment-budget allocation
  controls slots and behavior still misorders the active retained-marginal
  metric.

Targeted evidence:

- Current Level 3 artifact:
  `query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527`.
- Learned-control volume is not the blocker. The selector reports
  `88 / 104` learned-controlled retained slots (`0.8461538462`), with no
  large temporal scaffold.
- Query-prior information reaches sampled/model input fields but not retained
  masks. Shuffled prior sampled features change by mean absolute `0.0894552`;
  model-input prior features change by `0.0114177`; head probabilities move by
  only `0.0000096689`; final selector scores move by `0.0007349812`; the
  retained mask is unchanged. Shuffled-prior and no-query-prior deltas remain
  `0.0`.
- Segment budget is the only material learned-head route in the active replay:
  no-segment-budget delta is `0.0099825888`, while no-behavior delta is only
  `0.0014985765`.
- Segment scores strongly drive allocation, but that does not prove they rank
  retained marginal utility correctly. Selection trace reports
  `segment_score_to_allocation_spearman=0.8386278694`, high normalized segment
  entropy (`0.9856628455`), and only a narrow segment-score span
  (`0.0354789495`).
- The family-transfer diagnostic localizes behavior as a semantic/head problem:
  behavior active-metric Spearman is `-0.0486353219` overall and
  `-0.2002050581` on retained-removal marginals; the strongest behavior-target
  reference is still replacement (`spearman=0.5517435806`). It flags
  `behavior_head_misorders_retained_marginals`,
  `fitted_behavior_head_low_contrast`, and
  `behavior_target_more_replacement_than_segment_aligned`.
- Existing train-side exact retained-marginal instrumentation is diagnostic
  only. `Next-Iterations.md` records that the latest 64/256/40 train/eval
  transfer diagnostic still rejects guarded calibration
  (`diagnose_transfer_features_before_guarded_calibration_probe`, target
  Spearman `-0.6151`, top-k teacher overlap zero through top `10%`).

Decision:

- Do not build a train-marginal teacher or checkpoint-selection teacher into
  training semantics yet. The guide permits these as diagnostics only until
  strict replay proves transfer and causality.
- The next admissible implementation should be narrower: diagnose or adjust the
  behavior target/head transfer path by family and active retained-marginal
  alignment, while preserving target diffusion and pre-causality gates.
- Do not run another final-grid or large probe. Restart any source candidate at
  Level 1/2 under unchanged gates.

## Checkpoint Phase 32 - Behavior-Head Transfer Root-Cause Pass

Hypothesis:

- The next aligned move is not another behavior-label rewrite. The blocker is
  that the current behavior target/head path is replacement-aligned,
  low-contrast, and anti-aligned with exact retained-removal marginal
  `QueryLocalUtility`.

Expected files:

- Progress log only. No target or selector source change unless the existing
  diagnostics lacked the active-metric component split or exposed a concrete
  root fix.

Targeted evidence:

- Runtime protocol check used the repo environment:
  `uv run --group dev -- python --version` reports Python `3.14.5`.
- The existing family-transfer diagnostic already reports the active retained
  marginal component view from
  `selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_local_utility_alignment`.
  Adding another diagnostic would duplicate the current evidence.
- Current active behavior head alignment is negative: overall Spearman
  `-0.0486353219`, retained-removal Spearman `-0.2002050581`, and retained-
  removal top-minus-bottom marginal `-0.0002625089`.
- The same retained-removal view has replacement positive
  (`spearman=0.2321257690`, top-minus-bottom `0.0005743019`) while raw/final
  composed score is negative (`spearman=-0.0642868128`). This is not a missing
  input-feature story.
- Target-side behavior alignment is strongest with replacement
  (`spearman=0.5517435806`) and nearly absent for segment budget
  (`0.0116441826`); ship-query evidence is weakly negative
  (`-0.0304147801`).
- Head fit is still effectively flat: behavior target std `0.1664928645`,
  prediction std `0.0026310128`, Kendall tau `0.0250981047`.
- Learning causality remains blocked on query-prior and behavior dependence:
  shuffled-prior delta `0.0`, no-query-prior delta `0.0`, no-behavior delta
  `0.0014985765`, and no-segment delta `0.0099825888`.
- Existing behavior-candidate diagnostics do not offer a hidden easy
  replacement-gating fix. All current local behavior candidates remain
  replacement-aligned (`spearman` about `0.548-0.599`) and ship-evidence
  negative or near zero.

Decision:

- No source edit in this checkpoint. The active diagnostics are sufficient to
  reject another generic behavior-rank, prior-scale, or replacement-gated
  behavior pass.
- The next source candidate should either de-couple conditional behavior from
  replacement support or change behavior-loss semantics so the head ranks
  query-local interpolation/turn/continuity utility instead of regressing to a
  low-contrast mean. That candidate must start at Level 1 wiring and Level 2
  blocker localization under unchanged gates; no final grid is allowed.

## Checkpoint Phase 33 - Behavior/Replacement Decoupling Diagnostic

Hypothesis:

- Before adding another target mode or behavior-loss path, the target builder
  should expose whether the active behavior signal contains any useful
  non-replacement component. The current strict artifact shows replacement is
  positive on retained-removal marginals while behavior and composed score are
  negative, so a replacement-residualized behavior diagnostic is the narrowest
  next source change.

Files changed:

- `learning/targets/query_local_utility.py`
- `tests/unit/learning/test_query_local_utility_targets.py`
- this progress log

Implementation:

- Added a diagnostic-only `_positive_residualized_signal` helper that removes a
  linear nuisance signal on valid support, keeps only positive residuals, and
  renormalizes them.
- Added two behavior-candidate diagnostics:
  `replacement_residualized_local_behavior` and
  `segment_gated_replacement_residualized_local_behavior`.
- Active labels, target modes, model inputs, selector behavior, gates, and
  training semantics are unchanged.

Decision:

- This is Level 0 implementation evidence only. It does not prove learning or
  improve the current strict artifact.
- Existing strict artifacts cannot be used to infer the new candidate values,
  because they do not contain the raw train points/workloads needed to rebuild
  target diagnostics. The next admissible evidence is a small wiring run or a
  Level 2 blocker-localizing replay under unchanged gates, not a final grid.
- Retired by Phase 35 after the Level 1 wiring smoke showed this pseudo-target
  did not decouple from replacement rank.

## Checkpoint Phase 34 - Residualized Behavior Diagnostic Level 1 Wiring

Hypothesis:

- A fresh tiny smoke can verify that the replacement-residualized behavior
  diagnostics are emitted in end-to-end artifacts under active workload/profile
  settings. It cannot prove learning, workload health, or target quality.

Files/artifacts changed:

- New Level 1 artifact:
  `artifacts/results/query_driven_behavior_residualized_diag_level1_smoke_seed2632/example_run.json`
- This progress log.

Run boundary:

- Used `uv run --group dev -- python -m orchestration.train_and_score`.
- Scale: `n_ships=12`, `n_points=64`, `synthetic_route_families=2`,
  `n_queries=8`, `max_queries=96`, `range_train_workload_replicates=1`,
  `epochs=1`, compression `0.05`.
- Kept active profile/gate settings for this smoke:
  `workload_profile_id=range_query_mix`,
  `coverage_calibration_mode=profile_sampled_query_count`,
  `workload_stability_gate_mode=final`,
  `validation_split_mode=source_stratified`, and
  `range_max_coverage_overshoot=0.02`.
- A first loose smoke was discarded because it omitted
  `range_max_coverage_overshoot=0.02`; it must not be used as evidence.

Result:

- The corrected artifact emitted the new candidate keys:
  `replacement_residualized_local_behavior` and
  `segment_gated_replacement_residualized_local_behavior`.
- Active target semantics were unchanged:
  `target_mode=query_local_utility_factorized` and
  `conditional_behavior_target_variant=query_segment_local_behavior_utility`.
- Query counts were train `18`, eval `13`, selection `14`; this is Level 1
  wiring scale only.
- Gates at this tiny scale were not healthy: workload stability, workload
  signature, predictability, learning causality, and global sanity failed.
  Support overlap and target diffusion passed. No final claim is allowed.
- Candidate warning: positive residualization did not remove replacement
  ranking on this smoke. The current behavior candidate had replacement
  Spearman `0.1974`; `replacement_residualized_local_behavior` was higher at
  `0.2922`, and
  `segment_gated_replacement_residualized_local_behavior` was `0.2925`.
  Ship-evidence Spearman remained negative for all behavior candidates.

Decision:

- Treat this as successful artifact-wiring evidence and a caution against
  blindly promoting positive residualization as target semantics.
- The next admissible step is either a better diagnostic residualization
  formulation or a Level 2 blocker-localizing replay only if the candidate is
  revised to have a clearer target/head hypothesis. Do not run a final grid.
- Retired by Phase 35. Current source uses partial-alignment diagnostics
  instead of residualized pseudo-target candidates.

## Checkpoint Phase 35 - Behavior/Replacement Partial-Alignment Diagnostic

Hypothesis:

- The Level 1 residualized pseudo-target smoke showed that positive
  residualization increased replacement rank correlation. The right root
  diagnostic is therefore not another candidate label, but partial rank
  alignment: behavior versus each reference while controlling for replacement
  rank.

Files/artifacts changed:

- `learning/targets/query_local_utility.py`
- `tests/unit/learning/test_query_local_utility_targets.py`
- `artifacts/results/query_driven_behavior_partial_alignment_level1_smoke_seed2632/example_run.json`
- `Next-Iterations.md`
- this progress log

Implementation:

- Removed the `replacement_residualized_local_behavior` and
  `segment_gated_replacement_residualized_local_behavior` pseudo-target
  candidates from current source.
- Added diagnostic-only
  `conditional_behavior_replacement_partial_alignment`, with raw behavior
  Spearman, raw replacement Spearman, and behavior partial Spearman
  controlling replacement for final score, query-hit, ship evidence, segment
  budget, and path-length support.
- Active labels, target modes, model inputs, selector behavior, gates, and
  training semantics remain unchanged.

Level 1 wiring evidence:

- Artifact:
  `artifacts/results/query_driven_behavior_partial_alignment_level1_smoke_seed2632/example_run.json`.
- Same tiny wiring scale as the retired residualized smoke: `n_ships=12`,
  `n_points=64`, `synthetic_route_families=2`, seed `2632`, `n_queries=8`,
  `max_queries=96`, one train workload replicate, one epoch, source-stratified
  split, active `range_query_mix`, active `range_max_coverage_overshoot=0.02`,
  and final-mode workload gates.
- The artifact emitted `conditional_behavior_replacement_partial_alignment`.
  Candidate keys no longer include the retired residualized pseudo-targets.
- Tiny-run gates stayed unsuitable for evidence: workload stability,
  workload signature, predictability, learning causality, and global sanity
  failed. Support overlap and target diffusion passed. No learning or final
  claim is allowed.
- Diagnostic readout at this scale: behavior/replacement Spearman was
  `0.1974`; behavior partial Spearman controlling replacement was positive for
  final score (`0.5749`) and segment budget (`0.2177`), but negative for ship
  evidence (`-0.2586`).

Decision:

- Treat this as the current diagnostic surface for behavior/replacement
  separation. Do not promote residualized pseudo-targets.
- Next implementation should use this partial-alignment surface on a Level 2
  blocker-localizing replay before deciding whether behavior target semantics
  or loss pressure need another change. Do not run a final grid.

## Checkpoint Phase 36 - Level 2 Partial-Alignment Blocker Localization

Hypothesis:

- The Level 1 partial-alignment smoke only proved wiring. At Level 2, the new
  diagnostic can guide behavior/replacement target decisions only if the
  required pre-causality gates are healthy enough to make the readout
  meaningful.

Files/artifacts changed:

- `artifacts/results/query_driven_behavior_partial_alignment_level2_seed2633/example_run.json`
- `artifacts/results/query_driven_behavior_partial_alignment_level2_seed2633/family_transfer_path_diagnostic.json`
- `Next-Iterations.md`
- this progress log

Run configuration:

- Used the repo `uv` environment: `uv run --group dev -- python --version`
  reports Python `3.14.5`.
- Strict Level 2 diagnostic replay:
  `n_ships=32`, `n_points=128`, `synthetic_route_families=2`, seed `2633`,
  `n_queries=24`, `max_queries=192`, `range_train_workload_replicates=4`,
  source-stratified split, `range_query_mix`, `range_max_coverage_overshoot=0.02`,
  three epochs, and `final_metrics_mode=diagnostic`.

Evidence:

- MLQDS lost to uniform and slightly to Douglas-Peucker on QueryLocalUtility:
  `0.0811594668` versus uniform `0.1133670216` and Douglas-Peucker
  `0.0824648546`.
- Gates passed only workload stability and support overlap. Target diffusion
  failed because `replacement_representative_value` support fraction was
  `0.5607` above the `0.5` max. Workload signature failed the normalized
  point-hit-fraction KS check on every train/eval pair: train `0.2500`,
  train_r1 `0.2069`, train_r2 `0.2571`, train_r3 `0.2941`, and selection
  `0.3104`.
- Predictability failed lift at `1%`, lift at `2%`, Spearman, and PR-AUC lift
  checks: Spearman was `0.1026`, positive-target Spearman was `-0.1001`,
  lift@1% was `0.7900`, lift@2% was `1.0918`, lift@5% was `1.6543`, and
  PR-AUC lift over base rate was `1.2408`.
- Prior-predictive alignment failed
  `query_hit_spearman_below_min` and
  `segment_budget_lift_at_5_percent_below_min`. Best segment-budget lift@5%
  was only `1.0326`.
- Global sanity failed length preservation: average preserved length was
  `0.5927`, below the `0.75` minimum.
- Learning causality failed every material child check. Deltas were:
  shuffled `-0.0016`, untrained `-0.0066`, prior-only `-0.0133`,
  shuffled-prior `0.0`, no-query-prior `0.0`, no-behavior-head `0.0`, and
  no-segment-budget `-0.0102`.
- The partial-alignment diagnostic was emitted. Behavior/replacement Spearman
  was `0.2327`. Behavior partial Spearman controlling replacement was
  final score `0.1854`, query-hit `0.1207`, ship evidence `-0.0488`,
  segment budget `-0.0165`, and path support `0.1740`. Replacement aligned
  more strongly than behavior to final score and ship evidence.
- Training fit still showed a weak behavior head: conditional-behavior target
  std `0.1596`, prediction std `0.00239`, and Kendall tau `-0.1102`.
  Segment-budget fit was the only strong head readout, with Kendall tau
  `0.4127`.
- The family-transfer diagnostic decided
  `reject_target_contract_before_transfer_work`. Behavior active-metric
  alignment was weak but nonnegative: overall Spearman `0.0808`,
  retained-removal Spearman `0.0378`, and no-behavior-head QueryLocalUtility
  delta `0.0`.

Decision:

- Stop before changing target semantics, loss weights, selector coupling, or
  prior channels. This Level 2 artifact validates the diagnostic surface but
  is not admissible evidence for promoting a behavior-head fix.
- The current strict Level 3 64/256/40 replay remains the blocker-localizing
  reference. Use the Level 2 partial-alignment run only as a failed
  pre-causality diagnostic and as evidence that behavior/replacement separation
  needs gate-stable scale before it can guide implementation.

## Validation Summary

Latest focused validation:

- After the Level 2 partial-alignment blocker-localization replay: the run
  completed in the `uv` Python `3.14.5` environment; the family-transfer
  diagnostic completed; no production source changed.
- After the behavior/replacement partial-alignment diagnostic:
  `uv run --group dev -- python -m py_compile` on touched target/test files
  passed; `uv run --group dev -- ruff check` on touched files passed; focused
  QueryLocalUtility target pytest passed (`12 passed`); focused target,
  learning-target-stage, and guardrail pytest passed (`40 passed`);
  `git diff --check` passed. The Level 1 smoke emitted the new
  partial-alignment diagnostics. This is wiring evidence only.
- After the residualized behavior Level 1 wiring probe: the corrected smoke
  ran end to end and emitted the new diagnostics. This is not learning
  evidence. `git diff --check` passed after the log update.
- After the behavior/replacement decoupling diagnostic edit:
  `uv run --group dev -- python -m py_compile` on the touched target/test files
  passed; `uv run --group dev -- ruff check` on the touched files passed;
  focused QueryLocalUtility target pytest passed (`12 passed`); focused target,
  learning-target-stage, and guardrail pytest passed (`40 passed`);
  `git diff --check` passed. No probe and no final grid were run.
- After the behavior-head transfer root-cause pass: no source files changed and
  no probe or final grid was run. The checkpoint used existing strict Level 3
  artifacts and diagnostics plus the `uv` Python `3.14.5` runtime check.
- After the current causality path diagnosis: no source files changed; evidence
  came from the current Level 3 artifact, existing focused diagnostic JSONs,
  and `Next-Iterations.md`. No new probe and no final grid were run.
- After rejecting the direct turn/continuity target variants: the active source
  is restored to `query_segment_local_behavior_utility`; no failed turn,
  continuity, or replacement-source target strings remain in active
  source/docs/tests outside this progress-log evidence entry. `uv run --group
  dev -- python --version` reports Python `3.14.5`; py-compile, ruff, and
  focused target/orchestration/guardrail/diagnostic pytest passed (`47
  passed`); `git diff --check` passed. No final grid was run.
- After the prior contrast tradeoff diagnosis: no source/test files changed;
  evidence came from existing current and square-root strict artifacts. No
  final grid or new probe was run.
- After rejecting the square-root prior transform: the temporary source/test
  edit was reverted; `git diff` shows no remaining changes in
  `learning/model_features.py` or the protocol-gate test. Focused prior/model
  feature pytest passed (`66 passed`) and ruff passed on the inspected files.
- After the behavior loss/head diagnosis: no additional source/test files were
  changed; evidence came from existing strict artifacts and the guide. No final
  grid or new probe was run.
- After removing the rejected query-ship experimental target modes:
  production source no longer registers or branches on those modes; remaining
  string mentions are limited to rejected-mode tests, this log, and historical
  artifacts. `uv run --group dev -- python -m py_compile` on touched source and
  tests passed; `uv run --group dev -- ruff check` on the same files passed;
  focused target/orchestration/guardrail/diagnostic pytest passed
  (`47 passed`); `git diff --check` passed.
- After the replacement-boundary and current-cell causality diagnosis: no
  source files changed; only this progress log is modified.
- After rejecting and removing the query-run evidence behavior target path:
  source/test search found no remaining `query_run_evidence`,
  `QUERY_RUN_EVIDENCE`, `query_run_replacement`, or
  `query_run_evidence_behavior` symbols outside this progress log and
  artifacts; source/test diffs were removed, leaving only this log modified.
- For the rejected query-run evidence behavior target before removal:
  `uv run --group dev -- python -m py_compile` on touched target/test files
  passed; `uv run --group dev -- ruff check` on the same files passed; focused
  target/orchestration/guardrail pytest passed (`44 passed`); Level 1 smoke
  `query_driven_query_run_evidence_behavior_level1_smoke_seed2623` and Level 2
  diagnostic `query_driven_query_run_evidence_behavior_level2_seed2624` ran.
- `uv run --group dev -- python --version`: Python `3.14.5`.
- After the healthy-cell prior/selector causality diagnosis: no source files
  changed; `git diff --check` passed.
- After rejecting the replacement aggregate cap: source/test search found no
  remaining `query_component_local`, `component_local_heads`,
  `replacement_representative_aggregate_sparsifier`, or
  `top35_query_hit_support` references outside this progress log; source/test
  code diffs were removed, leaving only this log modified; `git diff --check`
  passed.
- After removing the failed component-local target branch:
  `uv run --group dev -- python -m py_compile` on touched target files passed;
  `uv run --group dev -- ruff check` on the same files passed; focused
  target/orchestration pytest passed (`24 passed`).
- After rejecting and removing the coverage-shrink diagnostic path:
  `uv run --group dev -- python -m py_compile` on the touched target and
  target-stage test files passed; `uv run --group dev -- ruff check` on the
  same files passed; focused target/orchestration pytest passed (`26 passed`);
  `git diff --check` passed.
- `uv run --group dev -- python -m py_compile` on
  `Range_QDS/learning/targets/query_local_utility.py`,
  `Range_QDS/tests/unit/learning/test_query_local_utility_targets.py`, and
  `Range_QDS/tests/unit/orchestration/test_learning_target_stage.py`.
- `uv run --group dev -- ruff check` on the same touched source/test files.
- `uv run --group dev -- pytest` on focused target and orchestration tests:
  `26 passed`.
- `uv run --group dev -- python -m orchestration.train_and_score` for
  `query_driven_query_component_local_heads_replacement_capped_level1_smoke_seed2615`.
- `uv run --group dev -- python -m orchestration.train_and_score` for
  `query_driven_query_component_local_heads_replacement_capped_level2_seed2616`.
- `uv run --group dev -- python -m orchestration.train_and_score` for
  `query_driven_query_component_local_heads_replacement_capped_level3_scale64_query48_seed2617`.

- `uv run --group dev -- python -m orchestration.diagnostics.workload_component_compatibility`
  for the active Level 3 replay:
  `query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/workload_component_compatibility_diagnostic.json`.
- `uv run --group dev -- python -m orchestration.diagnostics.selection_marginal_segment_calibration_diagnostic`
  for the active Level 3 replay:
  `query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/selection_marginal_segment_calibration_diagnostic.json`.
- `uv run --group dev -- python -m orchestration.diagnostics.selection_eval_segment_teacher_transfer_diagnostic`
  for the active Level 3 replay:
  `query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/selection_eval_segment_teacher_transfer_diagnostic.json`.
- `uv run --group dev -- python -m orchestration.diagnostics.selection_segment_transfer_feature_admissibility_diagnostic`
  for the active Level 3 replay:
  `query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/selection_segment_transfer_feature_admissibility_diagnostic.json`.
- `uv run --group dev -- python -m orchestration.diagnostics.selector_marginal_calibration_diagnostic`
  for the active Level 3 replay:
  `query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/selector_marginal_calibration_diagnostic.json`.
- `uv run --group dev -- python -m orchestration.diagnostics.family_transfer_path_diagnostic`
  for the active Level 3 replay:
  `query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/family_transfer_path_diagnostic.json`.
- `uv run --group dev -- python -m orchestration.train_and_score` for the
  rejected experimental query-ship-local-heads Level 2 diagnostic:
  `query_driven_query_ship_local_heads_level2_seed2610`.
- `uv run --group dev -- python -m orchestration.diagnostics.family_transfer_path_diagnostic`
  for the rejected query-ship-local-heads Level 2 diagnostic:
  `query_driven_query_ship_local_heads_level2_seed2610/family_transfer_path_diagnostic.json`.
- `uv run --group dev -- python -m orchestration.train_and_score` for the
  rejected experimental query-ship-local-heads Level 1 smoke:
  `query_driven_query_ship_local_heads_level1_smoke_seed2609`.
- `uv run --group dev -- python -m orchestration.train_and_score` for the
  rejected prior-head contrast Level 2 diagnostic:
  `query_driven_prior_head_contrast_level2_seed2608`.
- `uv run --group dev -- python -m orchestration.train_and_score` for the
  paired prior-head contrast default-control Level 2 diagnostic:
  `query_driven_prior_head_contrast_level2_default_control_seed2608`.
- `uv run --group dev -- python -m orchestration.train_and_score` for the
  rejected prior-head contrast Level 1 smoke:
  `query_driven_prior_head_contrast_level1_smoke_seed2607`.
- `uv run --group dev -- python -m orchestration.train_and_score` for the
  rejected segment-gated behavior Level 2 diagnostic:
  `query_driven_segment_gated_behavior_level2_seed2606`.
- `uv run --group dev -- python -m orchestration.train_and_score` for the
  rejected segment-gated behavior Level 1 smoke:
  `query_driven_segment_gated_behavior_level1_smoke_seed2605`.
- `uv run --group dev -- python -m orchestration.train_and_score` for the
  paired active-target Level 1 smoke:
  `query_driven_active_target_level1_pair_seed2605`.
- `uv run --group dev -- python -m orchestration.train_and_score` for the
  sparse-head BCE normalization Level 2 diagnostic:
  `query_driven_sparse_bce_window_norm_level2_seed2604`.
- `uv run --group dev -- python -m orchestration.train_and_score` for the
  reverted low-floor behavior Level 1 smoke:
  `query_driven_low_floor_behavior_level1_smoke_seed2602`.
- `uv run --group dev -- python -m orchestration.train_and_score` for the
  reverted low-floor behavior Level 2 diagnostic:
  `query_driven_low_floor_behavior_level2_seed2603`.
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

- After removing the rejected prior-head contrast diagnostic path:
  source/test search found no remaining `prior_head_contrast` symbols outside
  the progress log; `uv run --group dev -- python -m py_compile` on touched
  source/test files passed; `uv run --group dev -- ruff check` on the same
  files passed; focused pytest covering QueryLocalUtility training, runtime
  config/CLI, learning-target-stage, guardrail separation, and prior-feature
  protocol tests passed (`44 passed`); `git diff --check` passed.
- After removing the rejected segment-gated behavior target path:
  `uv run --group dev -- python -m py_compile` on touched source/test files,
  `uv run --group dev -- pytest` on target, learning-target-stage, and
  guardrail tests (`24 passed`), `uv run --group dev -- ruff check` on touched
  source/test files, and `git diff --check`.
- `uv run --group dev -- python -m py_compile` on
  `Range_QDS/learning/targets/query_local_utility.py`,
  `Range_QDS/models/workload_blind_range.py`, and
  `Range_QDS/learning/factorized_head_diagnostics.py`.
- `uv run --group dev -- pytest` on focused `QueryLocalUtility` target,
  training, and formula-composition tests: `26 passed`.
- `uv run --group dev -- python -m py_compile
  Range_QDS/models/workload_blind_range.py`.
- Focused model-prior tests after rejecting and reverting the post-context
  residual diagnostic: `2 passed`.
- `uv run --group dev -- python -m orchestration.train_and_score` for the
  behavior-rank-only strict diagnostic:
  `query_driven_behavior_rank015_level3_scale64_query40_seed2527`.
- Historical note: a stale-code cleanup entry previously cited bare
  `python3 -m py_compile`. Treat that as invalid protocol evidence for the
  current checkpoint stream; validation must use the repo `uv` environment
  (`uv run --group dev -- python`, currently Python `3.14.5`).
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
