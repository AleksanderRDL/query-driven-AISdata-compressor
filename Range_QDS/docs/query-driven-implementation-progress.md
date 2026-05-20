# Query-Driven Checkpoint Progress

This is the short checkpoint log required by
`docs/query-driven-implementation-research-guide.md`. The guide is the source
of truth. Current raw metrics and stdout belong in `artifacts/results/`.

## Current State - 2026-05-20

Status: active, not complete.

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
- target detail:
  `conditional_behavior_target_variant=query_segment_local_behavior_utility`,
  `replacement_representative_keep_fraction=0.35`,
  `segment_budget_target_aggregation=top20_mean`

Evidence boundary:

- A current-default strict train-marginal replay has been run under current QueryLocalUtility
  and the two-footprint `range_query_mix` profile at the healthy
  64-ship/256-point/40-requested-query diagnostic shape. It is evidence of
  old behavior-head blocker location, not evidence of final success. It
  predates the active multiplicative `query_segment_local_behavior_utility`
  target change in Checkpoint Group 34.
- The 64/256/40 replay passes workload stability, support overlap, target
  diffusion, workload signature, predictability, prior-predictive alignment,
  and global sanity.
- The replay still fails learning causality. The remaining blocker is semantic
  trainability: retained masks barely depend on query-prior fields or the
  behavior-utility head in the required direction.
- Final grid has not been run.
- Final success remains `false`.
- Generator-only source-stratified Level 3 probes now pass workload stability
  and workload signature at the `range_query_mix` 48-query floor on seeds
  `2524` and `2525`. This is workload-generation evidence only, not training
  coherence or final success evidence.

Latest current-default strict train-marginal replay:

- artifact:
  `artifacts/results/query_driven_train_marginal_scale64_query40_level3_range_query_mix_seed2527/example_run.json`
- train/eval transfer diagnostic:
  `artifacts/results/query_driven_train_marginal_scale64_query40_level3_range_query_mix_seed2527/train_eval_segment_teacher_transfer_diagnostic.json`
- current-focus family/head transfer diagnostic:
  `artifacts/results/query_driven_train_marginal_scale64_query40_level3_range_query_mix_seed2527/family_transfer_path_diagnostic.json`
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
- material positive causality controls: shuffled score loses by `0.0218749319`,
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
- non-leak check: train-side marginal selector uses `train` queries and does
  not use eval queries
- training note: checkpoint selection restored epoch 1; later epochs degraded
  validation selection score
- current-focus family/head transfer note: the derived diagnostic now uses the
  artifact's active focus families, not the historical `small_local` list. It
  blocks on `conditional_behavior_utility` for `density` and
  `medium_operational` under the legacy ship-evidence proxy. The same
  diagnostic now also reports active-metric retained-marginal alignment from
  `selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_local_utility_alignment.overall.score_component_alignment`.
  Current active-metric behavior-head evidence is bad: overall exact-marginal
  Spearman `0.0577`, retained-removal Spearman `-0.3817`, and no-behavior
  ablation delta `-0.0003343468`, dominated by the primary model's deficit
  versus no-behavior on `query_local_turn_change_coverage`.

Latest rejected head-contrast diagnostic:

- artifact:
  `artifacts/results/query_driven_head_contrast_sparse025_behavior015_level3_scale64_query40_seed2527/example_run.json`
- train/eval transfer diagnostic:
  `artifacts/results/query_driven_head_contrast_sparse025_behavior015_level3_scale64_query40_seed2527/train_eval_segment_teacher_transfer_diagnostic.json`
- scale: same strict 64/256/40 seed `2527` cell as the current-default replay;
  diagnostic-only with `query_local_utility_sparse_head_rank_loss_weight=0.25`,
  `query_local_utility_sparse_head_bce_target_mode=window_max_normalized`, and
  `query_local_utility_behavior_rank_loss_weight=0.15`
- result: rejected; no active default changed
- MLQDS QueryLocalUtility improved only slightly to `0.1402280700`, but the
  learning-causality gate still failed
- failed causality children stayed the same:
  `shuffled_prior_fields_should_lose`,
  `without_query_prior_features_should_lose`,
  `without_behavior_utility_head_should_lose`
- query-prior ablations became more wrong-way:
  shuffled-prior and no-query-prior deltas both `-0.0006932385`
- no-behavior remained wrong-way at `-0.0000416291`
- head contrast did not fix feature flow: no-query-prior still changed head
  probabilities by only `0.0000149776` on average and changed only 6 retained
  decisions
- train/eval transfer still rejected guarded segment-marginal calibration:
  target Spearman `-0.6785`, top-k overlap zero through top `10%`
- training note: validation selection again restored epoch 1; later epochs
  increased prediction variance but degraded selection score

Latest rejected prior-transform diagnostic:

- artifact:
  `artifacts/results/query_driven_prior_sqrt_level3_scale64_query40_seed2527/example_run.json`
- train/eval transfer diagnostic:
  `artifacts/results/query_driven_prior_sqrt_level3_scale64_query40_seed2527/train_eval_segment_teacher_transfer_diagnostic.json`
- scale: same strict 64/256/40 seed `2527` cell as the current-default replay;
  diagnostic-only with model-facing query-prior probabilities transformed by
  square root before the range model
- result: rejected; production default restored to identity prior probabilities
- MLQDS QueryLocalUtility improved only slightly to `0.1396786660`, but
  learning causality still failed
- failed causality children stayed the same:
  `shuffled_prior_fields_should_lose`,
  `without_query_prior_features_should_lose`,
  `without_behavior_utility_head_should_lose`
- no-query-prior model-feature delta increased to `0.0889404789`, score delta
  increased to `0.0043487814`, and retained symmetric difference increased to
  12 decisions, but head probability delta was still only `0.0000988260`
- shuffled-prior, no-query-prior, and no-behavior deltas stayed wrong-way at
  `-0.0003197163`, `-0.0003080556`, and `-0.0005148169`
- train/eval transfer still rejected guarded segment-marginal calibration:
  target Spearman `-0.6084`, top-k overlap zero through top `10%`
- conclusion: scalar prior transforms can make the input visibly different,
  but they do not fix directionality or behavior-head causality

Earlier current-default strict diagnostic:

- artifact:
  `artifacts/results/query_driven_segment_length_conflict_diag_level3_range_query_mix_seed2524/example_run.json`
- scale: Level 3 source-stratified strict training replay; blocker-localizing
  only
- MLQDS QueryLocalUtility: `0.1423908599`
- uniform QueryLocalUtility: `0.1283395087`
- Douglas-Peucker QueryLocalUtility: `0.1179874073`
- passed: workload stability, support overlap, target diffusion, workload
  signature, prior-predictive alignment
- failed: predictability, learning causality
- reported guardrail failure: global sanity, from
  `avg_sed_ratio_vs_uniform_too_high`
- added diagnostics: behavior-head-as-segment allocation probes and a uniform
  no-length-support segment allocation probe; these do not change default
  selection semantics
- latest segment-allocation diagnosis:
  `score_dominated_length_support_conflict`; segment-score/allocation Spearman
  is `0.8522`, length-support/allocation Spearman is `0.2985`,
  segment-score/length-support Spearman is only `0.1162`, and top-20%
  segment-score/length-support overlap is only `0.2105`

Latest rejected non-default loss diagnostic:

- artifact:
  `artifacts/results/query_driven_behavior_rank015_segment_diagnostic_level3_range_query_mix_seed2524/example_run.json`
- scale: Level 3 source-stratified strict training replay with unchanged gates
  and `query_local_utility_behavior_rank_loss_weight=0.15`
- result: diagnostic-only rejection; no active default changed
- MLQDS QueryLocalUtility increased to `0.1437732921`, but predictability and
  learning causality still failed
- behavior-head retained-marginal Spearman improved to `0.1589`, but the
  no-behavior ablation was still wrong-way at `-0.00013`
- behavior-head-as-segment scored `0.1394331953`, worse than primary; behavior
  allocation-only scored `0.1434139457`, still below primary
- no-segment-budget still beat primary badly: `0.1513084685` vs `0.1437732921`
- conclusion: behavior-rank pressure helps the symptom but does not provide a
  viable segment-allocation authority

Latest rejected non-default selector diagnostic:

- artifact:
  `artifacts/results/query_driven_allocation_floor0_level3_range_query_mix_seed2524/example_run.json`
- scale: Level 3 source-stratified strict training replay with unchanged gates
  and `learned_segment_allocation_weight_floor=0.0`
- result: diagnostic-only rejection; no active default changed
- MLQDS QueryLocalUtility increased to `0.1516471003`, but predictability and
  learning causality still failed and the same causality children remained
  wrong-way or immaterial
- no-segment-budget beat primary by `0.01068`; no-segment allocation-only beat
  by `0.00912`; untrained beat by `0.00880`
- segment-score retained-marginal Spearman turned negative at `-0.0771`; the
  active `conditional_behavior_utility` retained-marginal Spearman was
  `-0.3657`
- segment allocation entropy stayed very high at `0.9832`, so removing the
  allocation floor did not create reliable learned control
- global sanity still failed as a reported guardrail with AvgSED ratio
  `1.5771` vs uniform

Latest rejected selector transfer-calibration diagnostic:

- artifact:
  `artifacts/results/query_driven_segment_transfer_zblend_level3_range_query_mix_seed2524/example_run.json`
- scale: Level 3 source-stratified strict training replay with unchanged gates
  and
  `learned_segment_transfer_calibration_mode=segment_score_allocation_weight_zblend`
- result: diagnostic-only rejection; no active default changed
- MLQDS QueryLocalUtility dropped to `0.1406240561` from the current default
  `0.1423908599`
- workload stability, support overlap, target diffusion, workload signature,
  and prior-predictive alignment stayed green, but predictability and learning
  causality still failed
- no-segment-budget scored `0.1511892102`, no-segment allocation-only scored
  `0.1506543116`, and untrained scored `0.1487909782`, all still above the
  calibrated primary
- segment-score/allocation Spearman rose to `0.8894`, while
  length-support/allocation Spearman stayed weak at `0.2495`; the same
  `score_dominated_length_support_conflict` diagnosis remained
- conclusion: the derived admissibility diagnostics made the zblend probe legal
  to test, not good enough to promote

Latest rejected model-facing prior-channel diagnostic:

- artifact:
  `artifacts/results/query_driven_route_density_prior_enabled_level3_range_query_mix_seed2524/example_run.json`
- scale: Level 3 source-stratified strict training replay with unchanged gates
  and `route_density_prior` exposed to `workload_blind_range`
- result: diagnostic-only rejection; production default restored with
  `route_density_prior` still disabled in the model-facing prior vector
- MLQDS QueryLocalUtility dropped to `0.1421144423` from the current default
  `0.1423908599`
- workload stability, support overlap, target diffusion, workload signature,
  and prior-predictive alignment stayed green, but predictability and learning
  causality still failed
- removing only `route_density_prior` from the enabled replay improved
  QueryLocalUtility by `0.0002200603` and changed 16 retained decisions, so the
  channel is visible but directionally wrong in this strict cell
- no-query-prior remained far below materiality at `-0.0002253149`, and
  no-segment/untrained controls still beat primary

Latest rejected selector-allocation semantic diagnostic:

- artifact:
  `artifacts/results/query_driven_length_support_weight035_level3_range_query_mix_seed2524/example_run.json`
- scale: Level 3 source-stratified strict training replay with unchanged gates
  and `learned_segment_allocation_length_support_weight=0.35`
- result: diagnostic-only rejection; no active default changed
- MLQDS QueryLocalUtility improved to `0.1449775496`, and global sanity passed
  with AvgSED ratio vs uniform `1.4930`
- untrained now lost by `0.0055573732`, but predictability and learning
  causality still failed
- no-segment-budget still beat primary by `0.0063010396`; no-query-prior and
  no-behavior remained wrong-way/immaterial
- removing segment length-support allocation hurt primary by `0.0066176491`,
  confirming that query-free length support is helping allocation
- conclusion: length-support allocation semantics matter, but this is not a
  learned-query causality fix

Earlier derived selector-to-eval segment-marginal transfer diagnostic:

- artifact:
  `artifacts/results/query_driven_length_support_weight035_level3_range_query_mix_seed2524/selection_eval_segment_teacher_transfer_diagnostic.json`
- compared artifacts: current default
  `query_driven_segment_length_conflict_diag_level3_range_query_mix_seed2524`
  versus rejected length-support `0.35`
  `query_driven_length_support_weight035_level3_range_query_mix_seed2524`
- result: diagnostic-only; no active default changed
- seed `2524` current default: per-artifact decision
  `guarded_selection_segment_calibration_probe_admissible`; selection-side
  segment-marginal teacher shape was viable; selection/eval target Spearman
  was `0.1167`; four selector features had consistent positive selection/eval
  sign
- length-support `0.35`: per-artifact decision
  `diagnose_transfer_features_before_guarded_calibration_probe`;
  selection/eval target Spearman drops to `0.0291`; top-k teacher overlap
  remains zero through top `10%`; `segment_score` is contradictory-sign,
  `segment_allocation_weight` is weak on both splits, and
  `segment_length_support_score` is consistently negative
- conclusion: do not promote length-support weighting or train against its
  selector surface. The latest 64/256/40 train/eval exact-marginal diagnostic
  is stricter and rejects guarded segment-marginal calibration for now.

Latest train-side marginal diagnostics:

- Level 1 smoke artifact:
  `artifacts/results/query_driven_train_marginal_diag_level1_smoke_seed2526/example_run.json`
- Level 1 train/eval transfer diagnostic:
  `artifacts/results/query_driven_train_marginal_diag_level1_smoke_seed2526/train_eval_segment_teacher_transfer_diagnostic.json`
- Level 2 artifact:
  `artifacts/results/query_driven_train_marginal_diag_level2_range_query_mix_seed2527/example_run.json`
- Level 2 train/eval transfer diagnostic:
  `artifacts/results/query_driven_train_marginal_diag_level2_range_query_mix_seed2527/train_eval_segment_teacher_transfer_diagnostic.json`
- Level 3 artifact:
  `artifacts/results/query_driven_train_marginal_diag_level3_range_query_mix_seed2527/example_run.json`
- Level 3 train/eval transfer diagnostic:
  `artifacts/results/query_driven_train_marginal_diag_level3_range_query_mix_seed2527/train_eval_segment_teacher_transfer_diagnostic.json`
- Level 3 64/256/40 artifact:
  `artifacts/results/query_driven_train_marginal_scale64_query40_level3_range_query_mix_seed2527/example_run.json`
- Level 3 64/256/40 train/eval transfer diagnostic:
  `artifacts/results/query_driven_train_marginal_scale64_query40_level3_range_query_mix_seed2527/train_eval_segment_teacher_transfer_diagnostic.json`
- result: train-side exact marginal instrumentation works and remains
  non-leaky, but it is not admissible as training semantics yet
- Level 2 restored non-smoke scale and produced an admissible-looking transfer
  diagnostic, but workload signature, target diffusion, predictability,
  learning causality, global sanity, and Douglas-Peucker comparison blocked it
- Original 48/192 Level 3 target diffusion and global sanity passed, and MLQDS beat uniform,
  but workload stability, workload signature, prior-predictive alignment,
  predictability, learning causality, and Douglas-Peucker comparison blocked it
- Original 48/192 Level 3 train/eval segment-teacher transfer decision:
  `diagnose_transfer_features_before_guarded_calibration_probe`; target
  Spearman was `-0.4043`, top-k overlap was zero through top `10%`, and five
  selector features were contradictory-sign
- The 64/256/40 Level 3 replay fixed the workload-health blocker and passed
  predictability, prior-predictive alignment, and global sanity. It still fails
  learning causality because query-prior and behavior-head ablations are
  immaterial or wrong-way.
- The 64/256/40 train/eval segment-teacher transfer diagnostic still rejects a
  guarded calibration probe: target Spearman is `-0.6151`, top-k overlap is
  zero through top `10%`, and the decision remains
  `diagnose_transfer_features_before_guarded_calibration_probe`.

Latest workload/profile localization:

- 48 ships / 192 points / 40 requested queries:
  `artifacts/results/query_driven_profile_query_count40_diag_level3_range_query_mix_seed2527/example_run.json`
- 48 ships / 192 points / 32 requested queries:
  `artifacts/results/query_driven_profile_query_count32_diag_level3_range_query_mix_seed2527/example_run.json`
- 64 ships / 256 points / 48 requested queries:
  `artifacts/results/query_driven_profile_scale64_diag_level3_range_query_mix_seed2527/example_run.json`
- 64 ships / 256 points / 40 requested queries:
  `artifacts/results/query_driven_profile_scale64_query40_diag_level3_range_query_mix_seed2527/example_run.json`
- result: the `2527` workload failure is not a reason to change model
  semantics; it localizes to the interaction between minimum Level 3 scale,
  requested query count, and coverage-guard pressure
- at 48/192, reducing requested queries from 48 to 40 cleared workload
  stability but left checkpoint-selection point-hit-fraction KS failed at
  `0.225`; reducing to 32 still failed one train replicate at `0.2132`
- at 64/256 and 48 requested queries, workload signature passed but workload
  stability failed from extreme train coverage-overshoot pressure
- at 64/256 and 40 requested queries, workload stability and workload signature
  both passed; support overlap, target diffusion, prior-predictive alignment,
  and global sanity also passed in the diagnostic artifact
- follow-up evidence: the proper Level 3 train-marginal replay at 64/256 with
  40 requested queries completed. Workload/profile gates stayed green; the next
  blocker is semantic causality, not workload health.

Historical pre-simplification strict-cell reference:

- artifact retention: raw pre-simplification artifacts were pruned during
  Checkpoint Group 35; summary retained here for context only
- MLQDS QueryLocalUtility: `0.1662115143`
- uniform QueryLocalUtility: `0.1421296610`
- Douglas-Peucker QueryLocalUtility: `0.1671038781`
- passed: workload stability, support overlap, target diffusion, workload
  signature, prior-predictive alignment, global sanity
- failed: predictability, learning causality

Active blockers:

- Workload/profile health is no longer the current blocker at the 64/256/40
  diagnostic shape. Keep it guarded, but do not tune model semantics against
  the earlier failed 48/192/48 seed `2527` cell.
- Predictability passes in the 64/256/40 replay. Aggregate Spearman is
  `0.2600`, PR-AUC lift is `1.3589`, lift@1% is `2.6426`, lift@2% is `2.1323`,
  and lift@5% is `1.5325`.
- Learning causality is the main blocker. The primary mask does not materially
  depend on query-prior fields or the behavior head: shuffled/zeroed prior
  fields change only 4 retained decisions, no-query-prior and shuffled-prior
  deltas are both `-0.0000086480`, and removing the behavior head is slightly
  better by `0.0003343468`.
- Some causality controls are now healthy. Shuffled scores lose by
  `0.0218749319`, the untrained model loses by `0.0242913109`, prior-field-only
  loses by `0.0283540770`, and no-segment-budget loses by `0.0108408237`.
  Do not treat the remaining failure as a generic "model not learning" claim;
  it is specifically query-prior/behavior-head insensitivity.
- Sparse-head normalization plus behavior-rank pressure is not enough. The
  strict head-contrast probe left the same causality children failed and made
  query-prior ablations more wrong-way. Do not promote those loss settings as
  defaults from current evidence.
- Square-root model-facing prior transformation is not enough. It made the
  prior input and retained masks move more than the identity path, but the same
  query-prior and behavior-head causality children remained wrong-way. Do not
  continue by stacking scalar prior transforms without a directionality
  hypothesis.
- A lower allocation floor is not a root fix by itself. The floor-0 strict
  diagnostic improved aggregate QueryLocalUtility but made segment-budget
  removal and the untrained control even more favorable. Do not promote a lower
  floor until segment/head transfer is directionally correct under causality
  ablations.
- Replacing segment-budget allocation authority with
  `conditional_behavior_utility` is not a root fix either. Under the default
  strict replay, behavior-head-as-segment scored `0.13933` and behavior
  allocation-only scored `0.14187`, both below primary `0.14239`. Under
  behavior-rank loss, behavior allocation-only rose to `0.14341`, but still
  trailed primary `0.14377` and did not fix no-behavior/no-segment causality.
- Uniform/fair segment allocation without query-free length support was not the
  hidden winner in the older seed `2524` branch. It scored only `0.12412`. The
  high no-segment result in that branch was specifically a geometric
  length-support fallback beating the learned segment score, not neutral
  allocation beating learning. In the latest 64/256/40 replay, no-segment-budget
  loses as required; the remaining issue is query-prior and behavior-head
  insensitivity.
- Segment-budget allocation now has an explicit conflict diagnostic. In the
  latest strict replay, learned segment score strongly controls allocation but
  is weakly aligned with query-free length support:
  `segment_score_to_allocation_spearman=0.8522`,
  `length_support_to_allocation_spearman=0.2985`,
  `segment_score_to_length_support_spearman=0.1162`, and top-20% overlap
  `0.2105`.
- A guarded selector transfer calibration using
  `segment_score_allocation_weight_zblend` is not a root fix. It was admissible
  as a non-circular probe, but strict replay reduced primary QueryLocalUtility
  and worsened the no-segment and untrained causality failures. Do not promote
  selector transfer blending without fixing target/head transfer or allocation
  semantics.
- Exposing `route_density_prior` to the model-facing prior vector is not a root
  fix. It made the channel visible to retained-mask selection, but the direction
  was slightly anti-causal and the strict replay still failed predictability and
  learning causality. Keep `route_density_prior` disabled by default until a
  later workload/scoring redesign changes that evidence.
- Increasing segment allocation length-support weight to `0.35` is a useful
  localization result, not a default. It improved QueryLocalUtility, passed the
  global-sanity guardrail, and made the untrained ablation lose, but it still
  failed prior/head/segment causality. Do not promote it until train-side
  segment marginal calibration proves learned segment authority can contribute
  beyond query-free length support.
- Derived family/head diagnostics localize the weakest transfer to
  `conditional_behavior_utility`: target-side family evidence is weak for both
  `density` and `medium_operational`, and fitted predictions have very low
  variance relative to target variance.
- The 64-query floor still fails workload stability under the current
  `target_coverage=0.30` and `range_max_coverage_overshoot=0.020` envelope,
  mostly from coverage-overshoot pressure near the ceiling. Do not use that as
  the next default training probe without redesigning query-count/coverage
  policy.
- Single-dataset `source_stratified` splitting is now available for synthetic
  route-family probes. It fixes tested selection/eval route-family signature
  drift, but it does not by itself prove workload health. The current accepted
  generator evidence also depends on point-hit targeted proposals and the
  48-query floor.
- Exact retained-decision marginal alignment remains a central selector/target
  diagnostic. Current artifacts should read it from
  `selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_local_utility_alignment`,
  not from `learning_causality_summary.selection_causality_diagnostics`.
- Historical `small_local`, `crossing_turn_change`, and old
  `QueryUsefulV1` diagnostics are pre-simplification evidence. They are not
  active workload requirements under the current profile.
- Current workload-signature hard checks for coverage-calibrated
  `profile_sampled_query_count` workloads use normalized point-hit-fraction
  KS. Raw point/ship hit counts remain diagnostics because the active
  `QueryLocalUtility` score no longer contains the old explicit ship-coverage
  aggregate.
- Global sanity is computed and reported as a guardrail, but it is not an
  initial hard blocker while local query behavior, prior predictability, and
  learning causality remain unresolved.
- Anchor/footprint weights, footprint spatial/temporal dimensions, and scoring
  weights are adjustable research design variables. Change them only when
  gate-by-gate diagnosis shows the current workload/scoring pair is not
  producing a coherent trainable query-local signal.

Next checkpoint:

- Diagnose the semantic causality failure from the 64/256/40 strict replay
  before changing architecture. The immediate target is to explain why
  query-prior fields and the behavior head barely affect retained masks while
  score shuffling, untrained control, prior-field-only, and segment-budget
  removal now lose as expected. Start with model-facing query-prior feature
  flow, behavior-head loss/target transfer, selector marginal alignment, and
  why train/eval segment-marginal teacher transfer is still contradictory.
  Keep global sanity reported as a guardrail, not the first hard blocker.

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
  condensed progress file. Current retained raw artifacts are limited to the
  active evidence boundary; older pruned evidence remains summarized here and
  recoverable from git history if needed.

Artifact retention:

- Raw pre-canonical artifacts were pruned during Checkpoint Group 35. Keep the summary
  below as the current reference for this historical foundation.

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

Artifact retention:

- Raw pre-canonical artifacts were pruned during Checkpoint Group 35. Keep the summary
  below as the current reference for these rejected variants.

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

Artifact retention:

- Raw pre-canonical artifacts were pruned during Checkpoint Group 35. Keep the summary
  below as the current reference for this historical diagnosis.

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

Artifact retention:

- Raw pre-canonical artifacts were pruned during Checkpoint Group 35. Keep the summary
  below as the current reference for these pre-simplification diagnostics.

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
- Made QueryLocalUtility use direct `query_point_recall`, direct query-local
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
  QueryLocalUtility and the two-footprint profile.

Decision:

- Use QueryLocalUtility and the two-footprint `range_query_mix` profile for all new
  checkpoints.
- Do not compare current QueryLocalUtility scores against old metric scores as if they are the
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
- Folded the prior keep-in-mind guidance into the maintained guide and
  next-iteration docs instead of keeping a separate stale note file.

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

## Checkpoint Group 7 - Cleanup, Layout, And Refactor Foundation

Status: completed / hygiene and organization.

Goal:

- Clean up stale post-reset naming and improve top-down codebase structure
  without changing the scientific behavior of the current defaults.

Changes:

- Replaced stale redesign/legacy wording with current implementation and
  acceptance wording.
- Made remaining old names explicit historical diagnostics, negative
  guardrails, artifact compatibility fields, or removed-API checks.
- Removed the old catch-all query-driven orchestration test file and split its
  tests by owner across workload, learning, orchestration, selection, and
  diagnostic test files.
- Added `orchestration/diagnostics/` for completed-artifact analyzers.
- Extracted QueryLocalUtility segment and family helpers into
  `learning/targets/query_local_utility_segments.py` and
  `learning/targets/query_local_utility_family.py`.
- Extracted selector trace, marginal-alignment, and teacher-vector helpers into
  `orchestration/selector_trace_payloads.py`,
  `orchestration/selector_marginal_alignment.py`, and
  `orchestration/selector_teacher_vectors.py`.
- Updated `CODE_LAYOUT.md` after the refactors so it describes the remaining
  pressure points rather than completed work.

Tests:

- Focused `py_compile`, `ruff`, `pyright`, `pytest --collect-only`, and pytest
  runs on the touched production, diagnostic, and split test surfaces.
- Representative split query-driven suite: `123 passed`.
- Representative hygiene and guardrail suite: `197 passed`.
- `git diff --check`.

Experiment artifact:

- path: none
- command: none

Key results:

- No strict retraining, workload-health, scoring, or workload behavior changed.
- Query-driven tests are now owner-scoped instead of concentrated in one
  monolith.
- Completed-artifact analyzers are separated from run-stage orchestration.
- `query_local_utility.py` and `selector_diagnostics.py` no longer own helper
  surfaces that belong to target-family/segment or selector-trace modules.

Decision:

- Keep legacy paths only when they protect diagnostics, artifact comparability,
  checkpoint loading, removed-API guardrails, or final-claim separation.
- Add future one-off artifact analyzers under `orchestration/diagnostics/`.
- Do not recreate a catch-all query-driven test file.

## Checkpoint Group 8 - Current-Default Evidence And Target/Prior Probes

Status: completed / blocker-localizing evidence only.

Goal:

- Start the evidence boundary from QueryLocalUtility and the simplified
  `range_query_mix` workload instead of relying on old pre-simplification
  strict-cell results.

Hypotheses:

- Fresh strict evidence would localize the current blockers under the new metric
  and workload pair.
- `replacement_representative_value` support could be narrowed without
  weakening gates.
- Raising prior-feature scale might reveal whether query-prior fields were
  underpowered.

Changes:

- Ran a fresh Level 3 strict diagnostic under QueryLocalUtility, `range_query_mix`,
  `query_local_utility_factorized`, `workload_blind_range`,
  `learned_segment_budget`, and `mlqds_temporal_fraction=0.0`.
- Changed the active replacement representative keep fraction from `0.50` to
  `0.35` and recorded it in target diagnostics.
- Rejected a trial segment-budget highpass because it broke prior
  predictability.
- Temporarily raised `workload_blind_range` prior-feature residual
  initialization from `0.25` to `1.0`, then reverted it after strict evidence
  showed wrong-direction causality.

Tests:

- Focused default-wiring, scoring, workload, guardrail, target, and protocol
  tests.
- Focused `py_compile`, `ruff`, and `pyright` on the changed target and model
  surfaces.
- Representative target test suite: `13 passed`.

Artifact retention:

- Early early QueryLocalUtility raw artifacts were pruned during Checkpoint Group 35. Keep the
  summary below as the current reference for these rejected startup variants.

Key results:

- Fresh baseline status was `candidate_blocked_by_required_gates`; final
  success remained `false`.
- Fresh baseline QueryLocalUtility: MLQDS `0.1183492471`, uniform
  `0.1002604990`, Douglas-Peucker `0.1062839736`.
- Fresh baseline passed workload stability, support overlap, and
  prior-predictive alignment. It failed predictability, target diffusion,
  workload signature, learning causality, and global sanity.
- Replacement sparsening cleared target diffusion on the baseline seed:
  replacement support dropped from `0.5142` to `0.4335`.
- Replacement-sparse baseline still failed predictability, workload signature,
  learning causality, and global sanity.
- Prior-scale replay made prior fields more material but anti-causal at Level
  3. Shuffled or zeroed prior fields and prior-field-only scoring did not lose
  as required, and retained-decision marginal alignment stayed negative.

Decision:

- Keep replacement sparsening.
- Keep the prior-feature residual default at `0.25`.
- Do not use the old strict-cell scores as current-metric acceptance evidence.
- Do not move to model-capacity, selector-blend, or temporal-scaffolding tuning
  while workload signature and causality gates remain blocked.

## Checkpoint Group 9 - Current-Metric Workload Signature Calibration

Status: completed / partial generator-profile fix, still blocked.

Goal:

- Calibrate workload-signature diagnostics and generation around the current
  QueryLocalUtility metric/profile pair, not around old range-audit assumptions.

Hypothesis:

- The hard workload-signature failure was partly stale gate semantics. Raw
  point/ship hit counts are not comparable across unequal train/eval split
  sizes, and ship-hit parity is no longer a primary scoring component. The
  current profile should hard-check normalized point-hit fraction and retain raw
  point/ship counts as diagnostics.

Changes:

- Added `anchor_family_per_query` and `footprint_family_per_query` to generated
  range workload signatures.
- Added per-footprint accepted point-hit fraction bands:
  `medium_operational=[0.006,0.030]`,
  `large_context=[0.010,0.045]`.
- Threaded profile-owned point-hit fraction acceptance through range workload
  generation and rejection diagnostics.
- Changed workload-signature hard checks for
  `calibrated_to_coverage/profile_sampled_query_count` workloads to enforce
  normalized point-hit-fraction KS. Raw point counts and ship-hit count/fraction
  KS remain reported diagnostics.

Tests:

- `python3 -m py_compile` on changed workload generation, signature,
  diagnostics, and focused test files.
- `uv run --group dev -- pytest
  Range_QDS/tests/unit/workloads/test_query_driven_profiles.py
  Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py -q`
  (`29 passed`)
- `uv run --group dev -- ruff check` on the same changed code/test files.
- `uv run --group dev -- pyright` on the same changed code/test files.

Experiment artifacts:

- family-diagnostic probe:
  `artifacts/results/query_driven_signature_family_diag_level2_range_query_mix_seed2524/example_run.json`
- current-metric point-fraction-band diagnostic:
  `artifacts/results/query_driven_profile_point_fraction_bands_level2_range_query_mix_seed2524/example_run.json`

Key results:

- The family-diagnostic probe showed accepted family quotas were not the main
  problem. Drift was inside accepted family semantics.
- The current-metric point-fraction-band diagnostic passed support overlap,
  target diffusion, workload signature, and global sanity.
- Workload stability still failed because the selection workload had
  coverage-overshoot rejection pressure `2.219` above the `2.0` threshold.
- MLQDS QueryLocalUtility was `0.1206136524`, uniform was `0.0940594431`, and
  Douglas-Peucker was `0.1117748069`. This is Level 2 diagnostic evidence only.
- Predictability and prior-predictive alignment still failed. Aggregate
  score-vs-target Spearman was `-0.1543`; prior alignment failed
  `query_hit_spearman_below_min`.
- Learning causality still failed. Shuffled prior fields and no-prior-feature
  ablations left the mask unchanged; removing the segment-budget head hurt by
  `-0.0104`; the untrained model was better by `+0.0341`.

Decision:

- Keep the current-metric signature correction and point-fraction profile
  bands.
- Treat the latest Level 2 run as blocker localization, not success.
- Next checkpoint must fix workload-stability rejection pressure under
  unchanged gates, then replay Level 2/3 before returning to model or selector
  tuning.

## Checkpoint Group 10 - Workload Stability Pair Diagnosis

Status: completed / diagnostic instrumentation, no accepted profile change.

Goal:

- Diagnose the remaining Level 2 workload-stability failure before changing
  model, selector, or scoring behavior.

Hypothesis:

- Selection workload instability is localized to specific anchor/footprint
  combinations near the coverage guard, not to broad family-count skew.

Changes:

- Added `rejection_reasons_by_anchor_footprint_pair` to range acceptance
  diagnostics.
- Added `anchor_footprint_pair_counts` and
  `anchor_footprint_pair_per_query` to workload signatures.
- Updated final-claim handling so global sanity remains computed and reported
  but does not block the initial local-query-learning phase.
- Rejected a plan-horizon behavior change and large-footprint shrink probes
  because they traded workload stability for signature drift or poor expansion
  behavior. Those rejected variants are not kept in production code.

Tests:

- `python3 -m py_compile` on touched workload generation/signature and focused
  test files.
- `uv run --group dev -- pytest
  Range_QDS/tests/unit/workloads/test_query_driven_profiles.py
  Range_QDS/tests/unit/workloads/test_workload_generation.py -q`
  (`29 passed`)

Experiment artifact:

- path: none
- command shape: generator-only Level 2 reproduction of
  `query_driven_profile_point_fraction_bands_level2_range_query_mix_seed2524`
  using the current worktree.

Key results:

- Workload signature still passed for all train/eval/selection pairs under the
  current full-horizon profile plan.
- Workload stability still failed only
  `selection:coverage_guard_rejection_pressure_too_high`.
- Selection coverage-overshoot pressure was `2.219`; pair attribution:
  `sparse_background_control|large_context=42`,
  `density|large_context=24`, and `density|medium_operational=5`.
- The planned-prefix probe reduced selection pressure to `0.625` but failed
  train signature (`point_hit_fraction_ks=0.3125`) and was poor for larger
  expansion cases, so it was rejected.
- Large-context shrink probes reduced overshoot pressure but introduced
  normalized point-fraction signature failures, so they were rejected as
  one-sided fixes.

Decision:

- Keep the pair-level diagnostics.
- Do not change model or selector code yet.
- Next checkpoint should co-calibrate footprint dimensions, footprint weights,
  and point-hit fraction bands together. The acceptance target is unchanged:
  workload stability and normalized point-fraction signatures must both pass at
  Level 2 before a Level 3 replay.

## Checkpoint Group 11 - Workload Profile New-Beginning Calibration

Status: completed / diagnostic rejection, no accepted default change.

Goal:

- Re-check workload health from the current QueryLocalUtility defaults instead of
  carrying forward a one-seed `large_context` explanation.

Hypothesis:

- A sensible co-calibration of anchor weights, footprint weights, footprint
  dimensions, and point-hit bands would clear workload stability while keeping
  normalized point-fraction signatures under unchanged gates.

Changes:

- No production code or default profile changes were accepted.
- Updated `Next-Iterations.md` to point the next checkpoint at
  profile/query-count/split compatibility rather than more one-off weight
  nudges.

Tests:

- Generator-only diagnostics at guide Level 2 and Level 3 scales. No training,
  scoring, or final-grid run was performed.

Experiment artifact:

- path: none
- command shape: in-process generator-only probes using current worktree
  `range_query_mix`, `profile_sampled_query_count`, final workload-stability
  gates, 4 train workload replicates, and unchanged signature thresholds.

Key results:

- A `large_context=3.6km/7.25h` plus max point-hit fraction `0.050` variant
  passed the known Level 2 seed (`2524`) with workload stability and signature
  gates, but failed adjacent Level 2 seeds and Level 3 generator probes. It is
  rejected as a default.
- Existing low-coverage variants (`range_query_mix_focused`,
  `range_query_mix_local`, `range_query_mix_operational`) are incompatible with
  `n_queries=32` at this probe shape. They require 32 accepted queries while
  enforcing tight coverage ceilings, causing heavy coverage-guard rejection.
- Level 3 current-default rejection pressure is dominated by
  `too_low_point_hit_fraction` on larger train splits, especially
  `sparse_background_control|medium_operational` and
  `density|medium_operational`.
- Removing or lowering min point-hit fraction floors clears stability in some
  probes but breaks normalized point-fraction signature gates. The floors are
  doing real distribution-shaping work and cannot just be removed.
- Increasing medium footprint dimensions reduces rejection pressure but still
  leaves selection point-fraction signature failures or coverage pressure.
- A synthetic route-family-stratified split removes selection/eval signature
  drift on tested Level 3 seeds, but does not by itself fix train rejection
  pressure.

Decision:

- Do not change score weights, model code, selector code, or workload defaults
  from these probes.
- Treat the next checkpoint as a root generator/profile design checkpoint:
  make point-hit fraction floors, requested query count, target coverage, and
  synthetic split composition mutually compatible before returning to training
  coherence.
- Global sanity remains a tracked guardrail, not an initial hard blocker while
  local query behavior and causality are unresolved.

## Checkpoint Group 12 - Synthetic Route-Family Split Fix

Status: completed / partial generator-health support, still blocked.

Goal:

- Remove synthetic split composition as a confounder before further workload
  profile changes.

Hypothesis:

- Random single-dataset splits were creating avoidable selection/eval workload
  signature drift when `synthetic_route_families > 0`.

Changes:

- `prepare_run_split` now supports `validation_split_mode=source_stratified`
  for single-dataset train/selection/eval splits.
- Synthetic `train_and_score` runs now expose route-family ids as trajectory
  source ids when `synthetic_route_families > 0`.
- Added focused split tests for balanced single-dataset source-stratified
  splits and missing-source-id failures.
- Rejected a profile-anchor feasibility prefilter experiment because it traded
  rejection pressure for signature and coverage failures. The experiment was
  removed from production code.

Tests:

- `python3 -m py_compile` on touched split, train entrypoint, generator, and
  split test files.
- `uv run --group dev -- pytest
  Range_QDS/tests/unit/orchestration/test_data_splits.py -q` (`6 passed`).

Experiment artifact:

- path: none
- command shape: generator-only Level 3 probes with
  `validation_split_mode=source_stratified`, `range_query_mix`,
  `profile_sampled_query_count`, and unchanged workload-stability/signature
  gates.

Key results:

- Source-stratified synthetic splits balanced route families at Level 3:
  train `11` per family, selection `2` per family, eval `3` per family for
  64 ships and 4 route families.
- Workload signature passed on tested source-stratified Level 3 seeds `2524`
  and `2525`.
- Workload stability still failed because train workloads had high
  `range_generation_rejection_rate_too_high`; max rejection rate remained
  about `0.94-0.96`.
- Therefore the remaining blocker is profile acceptance/proposal compatibility,
  especially point-hit fraction floors on train splits. It is not solved by
  split balancing.

Decision:

- Use source-stratified synthetic route-family splits for future workload
  signature diagnostics.
- Do not claim workload health. The next checkpoint must reduce train rejection
  pressure under unchanged gates or redesign the profile/query-count policy
  with evidence.

## Checkpoint Group 13 - Point-Hit Targeted Proposal Calibration

Status: completed / accepted generator fix, training still blocked.

Goal:

- Make the active point-hit fraction bands produce stable accepted query
  distributions instead of relying on rejection after poorly aimed proposals.

Hypothesis:

- Min-only point-hit filtering creates floor-clustered train workloads and
  unstable normalized point-fraction signatures across split sizes. A bounded
  proposal calibration aimed at a deterministic low-band point-hit target should
  preserve the profile bands while reducing rejection pressure.

Changes:

- Range query proposals now receive the profile's per-query min/max point-hit
  fractions and a target point-hit fraction before acceptance filtering.
- `range_query_mix` profile settings now assign a prefix-stable low-discrepancy
  point-hit target inside the lower `25%` of each footprint family's band.
- Query construction performs bounded spatial/temporal scale calibration toward
  min/max/target point-hit counts, then still uses the unchanged acceptance and
  coverage gates.
- Updated stale low-coverage workload tests to use the active
  `range_query_mix` coverage/overshoot envelope.

Tests:

- `python3 -m py_compile` on touched workload generator/profile/test files.
- `uv run --group dev -- pytest
  Range_QDS/tests/unit/workloads/test_query_driven_profiles.py
  Range_QDS/tests/unit/workloads/test_workload_generation.py -q`
  (`31 passed`)
- `uv run --group dev -- ruff check` on touched workload generator/profile/test
  files.
- `uv run --group dev -- pyright` on touched workload generator/profile/test
  files.
- `git diff --check`

Experiment artifact:

- path: none
- command shape: generator-only Level 3 source-stratified probes with
  `n_ships=64`, `n_points=256`, `synthetic_route_families=4`,
  `range_query_mix`, `profile_sampled_query_count`, 4 train workload
  replicates, `max_queries=384`, and unchanged final workload stability and
  signature gates.

Key results:

- At `n_queries=48`, seeds `2524` and `2525` passed workload stability and
  workload signature.
- On the tested `n_queries=48` probes, maximum observed rejection rate stayed
  below the `0.85` gate threshold and maximum coverage-overshoot pressure
  stayed below the `2.0` gate threshold.
- `n_queries=32` and `40` also passed on tested seeds, but they are weaker
  evidence than the 48-query Level 3 probe.
- `n_queries=64` still failed workload stability on tested seeds. Signatures
  mostly passed, but coverage-overshoot pressure and rejection rate became too
  high near the `0.30/0.020` coverage envelope.

Decision:

- Keep the point-hit targeted proposal calibration.
- Use `n_queries=48` as the next strict synthetic training replay floor for the
  current `range_query_mix` profile.
- Do not claim learning coherence or final success from these generator-only
  probes. The next checkpoint must replay strict training and then diagnose
  predictability, prior alignment, and learning causality.

## Checkpoint Group 14 - Generator-Fixed Strict Replay And Causality Blocker

Status: completed / blocker-localizing evidence only.

Goal:

- Replay strict training after the accepted generator fix and diagnose the
  first remaining blocker by gate/component.

Hypothesis:

- If the generator fix removed the workload-health blocker, a 48-query Level 3
  source-stratified strict replay should keep workload gates green and expose
  the next blocker in predictability, prior alignment, or learning causality.

Changes:

- No production code changes.
- Added derived strict-artifact diagnostics for family/head transfer and local
  head failure localization.
- Updated this progress boundary and next-iteration plan.

Tests:

- `uv run --group dev -- python -m orchestration.train_and_score ...` with
  `validation_split_mode=source_stratified`, `n_queries=48`,
  `max_queries=384`, 4 train workload replicates, `range_query_mix`,
  `query_local_utility_factorized`, `workload_blind_range`,
  `learned_segment_budget`, and unchanged final gates.
- `uv run --group dev -- python -m orchestration.diagnostics.family_transfer_path_diagnostic ...`
- `uv run --group dev -- python -m orchestration.diagnostics.query_ship_local_heads_failure_diagnostic ...`

Experiment artifacts:

- strict replay:
  `artifacts/results/query_driven_generator_fixed_level3_range_query_mix_seed2524/example_run.json`
- family/head diagnostic:
  `artifacts/results/query_driven_generator_fixed_level3_range_query_mix_seed2524/family_transfer_path_diagnostic.json`
- local-head diagnostic:
  `artifacts/results/query_driven_generator_fixed_level3_range_query_mix_seed2524/query_ship_local_heads_failure_diagnostic.json`

Key results:

- Workload stability, support overlap, target diffusion, workload signature,
  and prior-predictive alignment passed.
- MLQDS beat uniform and Douglas-Peucker on QueryLocalUtility:
  `0.1422742877` vs `0.1283395087` and `0.1179874073`.
- Predictability failed only top-1% lift: lift@1% `1.0925` vs threshold
  `1.10`. Aggregate Spearman `0.2567`, lift@2% `1.4105`, lift@5% `1.4768`,
  and PR-AUC lift `1.4703` passed.
- Learning causality failed. Shuffled scores lose as required, but untrained,
  shuffled-prior, no-prior, no-behavior-head, and no-segment-budget-head
  controls do not lose.
- Query-prior ablations barely move the selected mask: shuffled/no-prior
  retained-decision Jaccard is `0.9747`.
- `conditional_behavior_utility` is the weakest family/head transfer path.
  Target-side family evidence is weak for `density` and
  `medium_operational`, and fitted predictions are nearly flat relative to the
  target variance.
- Global sanity failed by AvgSED ratio vs uniform, but remains a reported
  guardrail rather than the first hard blocker for this phase.

Decision:

- Do not run the final grid.
- Do not loosen gates or increase temporal scaffolding.
- Do not keep tuning the generator at the current 48-query floor unless
  workload gates regress.
- Next checkpoint should diagnose and fix the semantic trainability path:
  target/loss/head/selector coupling must make query-prior fields and the
  behavior/budget heads materially affect retained masks in the right
  direction. Score/workload weights and footprint dimensions remain adjustable
  only if diagnosis shows the current workload/scoring pair is incoherent.

## Checkpoint Group 15 - Segment-Budget Target/Selector Alignment

Status: completed / blocker-localizing evidence only.

Goal:

- Remove the known mismatch where the active segment-budget target used segment
  summed point mass while `learned_segment_budget` allocated segments from a
  top-20% segment summary.

Hypothesis:

- If segment-budget causality was blocked by target/selector aggregation
  mismatch, training the active segment-budget head on the same top-20% segment
  summary consumed by the selector should make the no-segment-budget ablation
  lose without regressing workload gates.

Changes:

- Changed the active `query_local_utility_factorized` segment-budget target
  aggregation from `sum` to `top20_mean`.
- Updated selector trace labels from stale `*_mean` names to `*_top20_mean`
  where the selector actually pools top-20% segment scores.
- Updated maintained docs to state the new active target detail.

Tests:

- `uv run --group dev -- pytest Range_QDS/tests/unit/learning/test_query_local_utility_targets.py -q`
  (`13 passed`)
- `uv run --group dev -- pytest Range_QDS/tests/unit/selection/test_query_driven_learned_segment_budget.py Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py -q`
  (`50 passed`)
- `uv run --group dev -- ruff check` on touched target, selector,
  orchestration, and focused test files.
- `uv run --group dev -- pyright` on the same touched files.
- Level 3 source-stratified strict replay with `range_query_mix`, 48 queries,
  4 train workload replicates, seed `2524`, and unchanged gates.
- `uv run --group dev -- python -m orchestration.diagnostics.family_transfer_path_diagnostic ...`
  on the new strict artifact.

Experiment artifacts:

- strict replay:
  `artifacts/results/query_driven_segment_budget_top20_level3_range_query_mix_seed2524/example_run.json`
- family/head diagnostic:
  `artifacts/results/query_driven_segment_budget_top20_level3_range_query_mix_seed2524/family_transfer_path_diagnostic.json`

Key results:

- Workload stability, support overlap, target diffusion, workload signature,
  and prior-predictive alignment stayed green.
- MLQDS QueryLocalUtility changed only marginally: `0.1423908599` vs uniform
  `0.1283395087` and Douglas-Peucker `0.1179874073`.
- Predictability still fails only lift@1%: `1.0925` vs threshold `1.10`.
- Learning causality still fails the same core checks. The no-segment-budget
  ablation still beats primary by `0.00889`; no-behavior beats by `0.00166`;
  untrained beats by `0.00640`; shuffled/no-prior fields are essentially flat
  at `-0.00005`.
- Prior-field ablations still change only 4 retained decisions. Removing the
  segment-budget head changes 60 decisions, but in the wrong direction.
- Segment-head target fit improved, but not enough to matter: segment-budget
  head tau `0.5877`, canonical segment tau `0.4617`, and top-5 mass recall
  `0.5104`.
- Retained-decision marginal alignment remains weak: selector-score Spearman
  `0.1483`, raw-score Spearman `0.1675`, segment-score Spearman `0.1320`.
  Query-free endpoint support still aligns more strongly at `0.5467`.
- Family/head diagnostics still block on `conditional_behavior_utility` for
  `density` and `medium_operational`; target-side ship-evidence alignment is
  weak and fitted predictions remain low-variance relative to target variance.
- Global sanity remains a reported guardrail failure: AvgSED ratio vs uniform
  `1.6066` exceeds the `1.5` threshold.

Decision:

- Keep the top-20% segment-budget target and trace labels because they are the
  correct semantics for the active selector.
- Do not claim learning success. This fix improved consistency, not causality.
- The next checkpoint should not continue changing segment aggregation in
  isolation. Diagnose the semantic learning path around
  `conditional_behavior_utility`, query-prior feature transfer, and the overly
  diffuse segment allocation signal. Score weights, workload weights, and
  footprint dimensions remain adjustable if that diagnosis shows the current
  workload/scoring pair is not producing a trainable local-query signal.

## Checkpoint Group 16 - Rejected Composite Behavior-Target Probe

Status: completed / rejected diagnostic, reverted from active defaults.

Goal:

- Test whether the failed `conditional_behavior_utility` path was caused by
  the active behavior target being too query-free and anti-aligned with
  ship-query evidence.

Hypothesis:

- Replacing query-hit-conditioned trajectory-change behavior with a compact
  query-local behavior utility could improve predictability and make the
  no-behavior-head ablation lose.

Changes:

- Temporarily changed active `conditional_behavior_utility` to combine
  normalized ship-query evidence, behavior change, replacement value, and
  boundary/event utility.
- Ran focused tests and a Level 3 source-stratified strict replay.
- Reverted the active target change after the strict replay showed wrong-way
  retained-mask behavior. The current default remains
  `conditional_behavior_target_variant=active_local_behavior_change`.

Tests:

- `uv run --group dev -- pytest Range_QDS/tests/unit/learning/test_query_local_utility_targets.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py -q`
  (`24 passed`) after reverting the rejected target.
- `uv run --group dev -- ruff check ...` on touched target/test files.
- `uv run --group dev -- pyright ...` on touched target/test files.
- Level 3 source-stratified strict replay with unchanged gates.

Experiment artifact:

- rejected strict replay:
  `artifacts/results/query_driven_query_local_behavior_level3_range_query_mix_seed2524/example_run.json`

Key results:

- Workload stability, support overlap, workload signature, predictability, and
  prior-predictive alignment passed.
- Target diffusion failed after the target change.
- MLQDS QueryLocalUtility dropped to `0.1303204850`, only barely above uniform
  `0.1283395087` and well below the accepted current-default diagnostic
  `0.1423908599`.
- Learning causality worsened. Shuffled scores beat primary by `0.00121`,
  untrained beat primary by `0.01757`, no-behavior beat by `0.00905`, and
  no-segment-budget beat by `0.02174`.
- Query-prior ablations still barely moved the retained mask: shuffled/no-prior
  changed only 2 retained decisions.
- The behavior head itself remained poorly aligned with retained marginal
  utility: behavior-head retained-marginal Spearman was `-0.1183`.
- Global sanity remained a reported guardrail failure with AvgSED ratio
  `1.6164` vs uniform.

Decision:

- Reject the composite behavior target as an active default. It made aggregate
  prior predictability pass but did not produce causal retained-mask learning.
- Do not continue by stuffing more query-presence mass into the behavior head.
  The next checkpoint should isolate why the model/selector turns usable
  query-hit and segment priors into low-contrast or wrong-direction retained
  masks. Focus on prior feature attenuation, segment allocation entropy, and
  selector score contrast before changing scoring/profile weights.

## Checkpoint Group 17 - Rejected Allocation-Floor Contrast Probe

Status: completed / rejected diagnostic, no default change.

Goal:

- Test whether the active selector's high segment-allocation floor was
  flattening useful segment-budget signal enough to hide causality.

Hypothesis:

- If learned segment scores are useful but overly flattened, setting
  `learned_segment_allocation_weight_floor=0.0` should make the segment-budget
  head more decisive and should make no-segment-budget and untrained ablations
  lose under the same strict workload gates.

Changes:

- No production code changes.
- Ran a Level 3 source-stratified strict replay with unchanged gates and only
  `learned_segment_allocation_weight_floor=0.0`.
- Updated the evidence boundary and next-iteration guidance.

Tests:

- `uv run --group dev -- python -m orchestration.train_and_score ...` with
  `validation_split_mode=source_stratified`, `n_queries=48`,
  `max_queries=384`, 4 train workload replicates, `range_query_mix`,
  `query_local_utility_factorized`, `workload_blind_range`,
  `learned_segment_budget`, `learned_segment_allocation_weight_floor=0.0`,
  and unchanged final gates.
- `git diff --check`

Experiment artifact:

- rejected strict replay:
  `artifacts/results/query_driven_allocation_floor0_level3_range_query_mix_seed2524/example_run.json`

Key results:

- Workload stability, support overlap, target diffusion, workload signature,
  and prior-predictive alignment stayed green.
- MLQDS QueryLocalUtility improved to `0.1516471003` vs uniform
  `0.1283395087`, but the candidate remained blocked by predictability and
  learning causality.
- Predictability still failed only lift@1%: `1.0925` vs threshold `1.10`.
- Learning causality still failed untrained, shuffled-prior,
  no-query-prior, no-behavior-head, and no-segment-budget checks.
- The no-segment-budget ablation beat primary by `0.01068`, worse than the
  current-default `0.00889` wrong-way delta. The no-segment allocation-only
  diagnostic beat primary by `0.00912`.
- The untrained control beat primary by `0.00880`. Shuffled/no-prior field
  deltas were only `+0.00066`, far below the `0.005` materiality threshold.
- Segment-score retained-decision marginal alignment turned negative:
  Spearman `-0.0771`. The behavior head was worse: retained-marginal Spearman
  `-0.3657`.
- Segment allocation entropy remained very high at `0.9832`, and the active
  segment score span narrowed to `0.1475`; lower floor did not create
  reliable learned control.
- QueryLocalUtility gains came from point mass and query-local interpolation,
  but range usefulness still lost to uniform (`0.2067` vs `0.2220`) and global
  sanity stayed a reported guardrail failure.

Decision:

- Reject zero allocation floor as an active default. It exposes the wrong
  learned segment signal more strongly instead of fixing causality.
- Do not continue with selector floor tuning until segment/head transfer is
  directionally correct. The next checkpoint should focus on why learned
  `segment_budget_target` and `conditional_behavior_utility` outputs are
  weakly or negatively aligned with retained marginal QueryLocalUtility.
- Score weights, workload weights, and footprint dimensions remain legitimate
  research knobs if diagnosis shows the current workload/scoring pair is
  incoherent, but this run points first to head semantics/loss transfer rather
  than profile-only tuning.

## Checkpoint Group 18 - Behavior-Head Segment Authority Diagnostic

Status: completed / diagnostic added, no default selection change.

Goal:

- Test whether `conditional_behavior_utility` is a better segment-allocation
  authority than the active `segment_budget_target` path.

Hypothesis:

- If the behavior head contains useful local-query ordering that the
  segment-budget consumer dilutes, replacing segment scores with raw behavior
  head logits should improve QueryLocalUtility and reduce the no-behavior or
  no-segment causality failures.

Changes:

- Added diagnostic-only retained-mask ablations:
  `MLQDS_behavior_utility_segment_head_diagnostic` and
  `MLQDS_behavior_utility_allocation_only_diagnostic`.
- Added a focused unit test that verifies those ablations are frozen when
  factorized head logits are available.
- Did not change default scoring, workload generation, training targets, or
  primary selector semantics.

Tests:

- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_retained_mask_stage.py -q`
  (`7 passed`)
- `uv run --group dev -- ruff check Range_QDS/orchestration/retained_mask_ablation_stage.py Range_QDS/tests/unit/orchestration/test_retained_mask_stage.py`
- `uv run --group dev -- pyright Range_QDS/orchestration/retained_mask_ablation_stage.py Range_QDS/tests/unit/orchestration/test_retained_mask_stage.py`
- Two Level 3 source-stratified strict replays with unchanged gates: one
  current-default replay, one behavior-rank-loss contrast replay.

Experiment artifacts:

- current-default strict diagnostic:
  `artifacts/results/query_driven_behavior_segment_diagnostic_level3_range_query_mix_seed2524/example_run.json`
- behavior-rank-loss strict diagnostic:
  `artifacts/results/query_driven_behavior_rank015_segment_diagnostic_level3_range_query_mix_seed2524/example_run.json`

Key results:

- Current-default replay reproduced primary QueryLocalUtility `0.1423908599`
  vs uniform `0.1283395087`; workload stability, support overlap, target
  diffusion, workload signature, and prior-predictive alignment stayed green.
- Current-default behavior-head-as-segment scored `0.1393342307`; behavior
  allocation-only scored `0.1418743810`. Both trail primary.
- Behavior-rank loss improved primary QueryLocalUtility to `0.1437732921` and
  behavior retained-marginal Spearman to `0.1589`, but predictability and
  learning causality still failed.
- Under behavior-rank loss, behavior-head-as-segment scored `0.1394331953`;
  behavior allocation-only scored `0.1434139457`. Both still trail primary.
- The no-segment-budget ablation remains the stronger warning: it scored
  `0.1512785892` under default and `0.1513084685` under behavior-rank loss,
  beating primary in both strict replays.
- Global sanity remained a reported guardrail failure, not an initial local
  learning blocker: AvgSED ratio vs uniform was `1.6066` under default and
  `1.5638` under behavior-rank loss.

Decision:

- Reject behavior-head segment authority as an active default.
- Behavior-rank pressure is also not enough to promote; it improves the
  behavior-head symptom but does not fix causality.
- The next checkpoint should treat the active segment-budget allocation path
  as harmful until proven otherwise. Diagnose why neutral/no-segment allocation
  beats learned segment-budget allocation before changing gates or promoting a
  replacement segment authority.

## Checkpoint Group 19 - Segment Allocation Length-Support Isolation

Status: completed / diagnostic added, no default selection change.

Goal:

- Separate “learned segment-budget allocation is harmful” from “uniform segment
  allocation is secretly better.”

Hypothesis:

- The no-segment-budget win may be coming from the ablation's query-free
  geometric length-support fallback, not from removing learned allocation
  authority alone.

Changes:

- Added `MLQDS_uniform_segment_allocation_only_diagnostic`, which uses neutral
  segment allocation scores, disables segment length-support allocation, and
  keeps the primary segment point-score blend. This isolates uniform/fair
  segment allocation from length-support fallback.
- Added focused unit coverage for the new diagnostic method.
- Did not change default scoring, workload generation, training targets, or
  primary selector semantics.

Tests:

- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_retained_mask_stage.py -q`
  (`7 passed`)
- `uv run --group dev -- ruff check Range_QDS/orchestration/retained_mask_ablation_stage.py Range_QDS/tests/unit/orchestration/test_retained_mask_stage.py`
- `uv run --group dev -- pyright Range_QDS/orchestration/retained_mask_ablation_stage.py Range_QDS/tests/unit/orchestration/test_retained_mask_stage.py`
- Level 3 source-stratified strict replay with unchanged gates.

Experiment artifact:

- `artifacts/results/query_driven_uniform_segment_allocation_diag_level3_range_query_mix_seed2524/example_run.json`

Key results:

- Primary reproduced QueryLocalUtility `0.1423908599` vs uniform
  `0.1283395087`; workload stability, support overlap, target diffusion,
  workload signature, and prior-predictive alignment stayed green.
- Predictability still failed only lift@1%: `1.0925` vs `1.10`.
- Learning causality still failed the same children: untrained,
  shuffled-prior, no-query-prior, no-behavior, and no-segment-budget.
- Uniform/fair segment allocation without length support scored only
  `0.1241232224`, far below primary. This rejects the idea that neutral
  allocation alone explains the no-segment win.
- Active learned segment allocation without length support scored
  `0.1383599005`, also below primary.
- Length-support fallback without segment-budget allocation remains much better:
  no-segment-budget scored `0.1512785892`, and no-segment allocation-only
  scored `0.1507434816`.
- Active allocation strongly follows segment score rather than length support:
  segment-score/allocation Spearman `0.8522`, length-support/allocation
  Spearman `0.2985`, with only `0.2105` overlap between top-20% segment-score
  and top-20% length-support groups.
- Global sanity remained a reported guardrail failure: AvgSED ratio vs uniform
  `1.6066`.

Decision:

- Keep the new diagnostic. It clarifies future ablation interpretation.
- Do not promote uniform allocation. It is bad.
- The next fix should target the conflict between learned segment-budget
  authority and query-free geometric length-support allocation. The current
  segment head is strong enough to dominate allocation, but it is not aligned
  with the allocation pattern that improves QueryLocalUtility.

## Checkpoint Group 20 - Segment Score And Length-Support Conflict Diagnostic

Status: completed / diagnostic clarified, no default selection change.

Goal:

- Make the segment-allocation diagnostic distinguish weak length-support
  influence from a real learned-score versus length-support conflict.

Hypothesis:

- The current diagnostic can understate the blocker by reporting that length
  support materially influences allocation even when the learned segment score
  dominates allocation and has low top-k overlap with query-free geometric
  length support.

Changes:

- Added direct segment-score/length-support and
  allocation-weight/length-support Pearson and Spearman fields to
  `segment_allocation_alignment_diagnostics`.
- Added the `score_dominated_length_support_conflict` diagnosis for cases where
  segment score dominates allocation while top segment-score and length-support
  groups have low overlap.
- Kept default scoring, workload generation, training targets, and primary
  selector semantics unchanged.

Tests:

- `uv run --group dev -- pytest Range_QDS/tests/unit/selection/test_learned_segment_budget.py -q`
  (`14 passed`)
- `uv run --group dev -- ruff check Range_QDS/selection/learned_segment_budget/diagnostics.py Range_QDS/tests/unit/selection/test_learned_segment_budget.py`
- `uv run --group dev -- pyright Range_QDS/selection/learned_segment_budget/diagnostics.py Range_QDS/tests/unit/selection/test_learned_segment_budget.py`
- `git diff --check`
- Level 3 source-stratified strict replay with unchanged gates.

Experiment artifact:

- `artifacts/results/query_driven_segment_length_conflict_diag_level3_range_query_mix_seed2524/example_run.json`

Key results:

- Primary scores reproduced the previous current-default strict replay:
  MLQDS QueryLocalUtility `0.1423908599`, uniform `0.1283395087`,
  Douglas-Peucker `0.1179874073`.
- Workload stability, support overlap, target diffusion, workload signature,
  and prior-predictive alignment stayed green.
- Predictability still failed only lift@1%: `1.0925` vs `1.10`.
- Learning causality still failed the same child gates: untrained,
  shuffled-prior, no-query-prior, no-behavior, and no-segment-budget.
- The segment allocation diagnostic now reports
  `score_dominated_length_support_conflict`:
  segment-score/allocation Spearman `0.8522`,
  length-support/allocation Spearman `0.2985`,
  segment-score/length-support Spearman `0.1162`,
  allocation-weight/length-support Spearman `0.2502`, and top-20%
  segment-score/length-support overlap `0.2105`.
- Global sanity remained a reported guardrail failure:
  AvgSED ratio vs uniform `1.6066`.

Decision:

- Keep the clarified diagnostic.
- Do not interpret the length-support fallback win as a uniform allocation win.
- The next implementation change should target segment target semantics or
  selector coupling so learned segment authority aligns with the local-query
  regions that geometric length support is currently preserving. Do not tune
  scoring/workload weights or footprint dimensions blindly; they remain valid
  research variables only if the next gate-by-gate diagnosis shows the
  workload/scoring pair is incoherent.

## Checkpoint Group 21 - Guarded Segment Transfer Calibration Probe

Status: completed / selector calibration rejected, no default change.

Goal:

- Test whether a non-circular segment transfer calibration can repair retained
  marginal alignment without changing workload, scoring, target, or final gates.

Hypothesis:

- The current blocker is not workload health or uniform allocation. It may be a
  mismatch between the learned segment-budget score consumed by the selector
  and the query-free length-support allocation pattern that currently improves
  QueryLocalUtility.

Changes:

- No production default changed.
- Ran a strict replay with only
  `learned_segment_transfer_calibration_mode=segment_score_allocation_weight_zblend`.
- Ran the existing derived selector transfer diagnostics on the resulting
  artifact.

Tests:

- Level 3 source-stratified strict replay with unchanged workload gates and
  unchanged score/profile defaults.
- `selection_eval_segment_teacher_transfer_diagnostic`
- `selection_segment_transfer_feature_admissibility_diagnostic`
- `selection_marginal_segment_calibration_diagnostic`
- `selector_marginal_calibration_diagnostic`

Experiment artifact:

- `artifacts/results/query_driven_segment_transfer_zblend_level3_range_query_mix_seed2524/example_run.json`

Key results:

- Primary MLQDS QueryLocalUtility dropped to `0.1406240561` from the current
  default `0.1423908599`; uniform stayed `0.1283395087`, and Douglas-Peucker
  stayed `0.1179874073`.
- Workload stability, support overlap, target diffusion, workload signature,
  and prior-predictive alignment stayed green.
- Predictability still failed lift@1%, and learning causality still failed
  untrained, shuffled-prior, no-query-prior, no-behavior, and no-segment.
- No-segment-budget scored `0.1511892102`, no-segment allocation-only scored
  `0.1506543116`, and untrained scored `0.1487909782`, all above primary.
- The zblend trace was query-blind and non-circular:
  `uses_post_selection_attribution=false`,
  `uses_length_support_counter_signal=false`, and effective segment
  length-support weight `0.0`.
- The conflict diagnosis remained `score_dominated_length_support_conflict`:
  segment-score/allocation Spearman `0.8894`,
  length-support/allocation Spearman `0.2495`,
  segment-score/length-support Spearman `0.2176`, and top-20% score/support
  overlap `0.2632`.
- Behavior and path-length allocation-only diagnostics beat the zblend primary
  (`0.1472759011` and `0.1445999283` respectively), but they still trailed the
  no-segment length-support fallback and did not pass causality.
- Global sanity remained a reported guardrail failure, not the initial hard
  blocker for this phase.

Decision:

- Reject `segment_score_allocation_weight_zblend` as a default.
- Treat the derived admissibility diagnostics as permission to run the probe,
  not as promotion evidence.
- Next work should target train-to-eval head transfer, prior-feature usage, and
  segment allocation semantics. Do not keep tuning selector calibration in
  isolation.

## Checkpoint Group 22 - Route-Density Prior Exposure Probe

Status: completed / model-facing prior-channel change rejected, no default
change.

Goal:

- Test whether the disabled `route_density_prior` channel is blocking
  model-facing prior usage under the current `density=0.80` workload profile.

Hypothesis:

- Because the active workload mostly samples dense-route anchors, suppressing
  `route_density_prior` might hide a useful train-derived support signal from
  `workload_blind_range` and weaken retained-mask causality.

Changes:

- Temporarily exposed `route_density_prior` in the local model-facing
  `workload_blind_range` prior vector for the strict replay.
- Reverted the source change after the replay because the evidence rejected it.
- No production default changed.

Tests:

- Level 3 source-stratified strict replay with unchanged workload gates,
  unchanged score/profile defaults, and only the route-density model-input mask
  changed.
- `git diff -- Range_QDS/learning/model_features.py` after revert showed no
  remaining source diff for the temporary change.

Experiment artifact:

- `artifacts/results/query_driven_route_density_prior_enabled_level3_range_query_mix_seed2524/example_run.json`

Key results:

- Primary MLQDS QueryLocalUtility dropped to `0.1421144423` from the current
  default `0.1423908599`; uniform stayed `0.1283395087`, and Douglas-Peucker
  stayed `0.1179874073`.
- Workload stability, support overlap, target diffusion, workload signature,
  and prior-predictive alignment stayed green.
- Predictability still failed, and learning causality still failed untrained,
  shuffled-prior, no-query-prior, no-behavior, and no-segment.
- The model-facing route-density channel was genuinely active:
  shuffled-prior mean absolute model-input route-density delta was `0.4901`,
  and removing only route density changed 16 retained decisions.
- Direction was wrong: removing only `route_density_prior` from the enabled
  replay improved QueryLocalUtility by `0.0002200603`.
- No-query-prior remained immaterial at `-0.0002253149`, still far below the
  `0.005` materiality threshold.
- Segment allocation stayed `score_dominated_length_support_conflict`:
  segment-score/allocation Spearman `0.8508`, length-support/allocation
  Spearman `0.2985`, segment-score/length-support Spearman `0.1110`.
- Global sanity remained a reported guardrail failure, not the first hard
  blocker for this phase.

Decision:

- Keep `route_density_prior` disabled in the model-facing prior vector.
- Do not continue by enabling broad density prior channels. The channel is
  visible when enabled, but the strict evidence is slightly anti-causal.
- Next work should move to target/head transfer and segment allocation
  semantics rather than more prior-channel exposure.

## Checkpoint Group 23 - Length-Support Allocation Weight Probe

Status: completed / selector-allocation semantic probe rejected as default.

Goal:

- Test whether the learned segment-budget failure is partly an allocation
  authority problem: active segment score may be over-dominating the query-free
  length-support signal that improves QueryLocalUtility in no-segment fallback.

Hypothesis:

- If the segment-score/length-support conflict is real, increasing
  `learned_segment_allocation_length_support_weight` should improve primary
  QueryLocalUtility and global sanity. If learning causality still fails, the
  result should be treated as allocation localization, not as a default.

Changes:

- No production code changed.
- Ran a strict replay with only
  `learned_segment_allocation_length_support_weight=0.35`.
- Ran derived family-transfer, selector-marginal, and selection-marginal
  segment calibration diagnostics on the resulting artifact.

Tests:

- Level 3 source-stratified strict replay with unchanged workload gates and
  unchanged score/profile/target defaults.
- `uv run --group dev -- python -m orchestration.diagnostics.family_transfer_path_diagnostic ...`
- `uv run --group dev -- python -m orchestration.diagnostics.selector_marginal_calibration_diagnostic ...`
- `uv run --group dev -- python -m orchestration.diagnostics.selection_marginal_segment_calibration_diagnostic ...`

Experiment artifacts:

- strict replay:
  `artifacts/results/query_driven_length_support_weight035_level3_range_query_mix_seed2524/example_run.json`
- derived diagnostics:
  `artifacts/results/query_driven_length_support_weight035_level3_range_query_mix_seed2524/family_transfer_path_diagnostic.json`
  `artifacts/results/query_driven_length_support_weight035_level3_range_query_mix_seed2524/selector_marginal_calibration_diagnostic.json`
  `artifacts/results/query_driven_length_support_weight035_level3_range_query_mix_seed2524/selection_marginal_segment_calibration_diagnostic.json`

Key results:

- Primary MLQDS QueryLocalUtility improved to `0.1449775496` from the current
  default `0.1423908599`; uniform stayed `0.1283395087`, and Douglas-Peucker
  stayed `0.1179874073`.
- Workload stability, support overlap, target diffusion, workload signature,
  and prior-predictive alignment stayed green.
- Global sanity passed: AvgSED ratio vs uniform `1.4930`.
- Predictability still failed, and learning causality still failed
  shuffled-prior, no-query-prior, no-behavior, and no-segment.
- Untrained now lost by `0.0055573732`, clearing that child gate.
- No-segment-budget still beat primary: `0.1512785892` vs `0.1449775496`.
- Removing segment length-support allocation hurt primary by `0.0066176491`,
  so length support is a real positive allocation signal in this strict cell.
- Segment allocation still reported `score_dominated_length_support_conflict`:
  segment-score/allocation Spearman `0.7951`, length-support/allocation
  Spearman `0.3028`, and segment-score/length-support Spearman `0.1162`.
- Retained marginal selector-score Spearman improved to `0.1867`, but the
  derived selector diagnostic still requires train-side marginal calibration
  before any promotion.
- Family transfer remained blocked only on `conditional_behavior_utility` for
  `density` and `medium_operational`; the target-side behavior signal is still
  weak/negative against ship-query evidence.

Decision:

- Reject `learned_segment_allocation_length_support_weight=0.35` as a default.
- Treat it as evidence that allocation semantics and query-free length support
  are materially important.
- Next work should build guarded train-side segment marginal calibration
  evidence and target the learned segment authority gap, while keeping the
  `conditional_behavior_utility` target weakness in view.

## Checkpoint Group 24 - Train-Side Segment Marginal Transfer Diagnosis

Status: completed / derived diagnostic only, no default change.

Goal:

- Decide whether the next implementation checkpoint can safely build a guarded
  train-side segment-marginal calibration target, and whether the rejected
  length-support `0.35` selector surface should be used as that basis.

Hypothesis:

- Selection-side exact retained-decision marginals may provide a legal
  train-side teacher, but only if the teacher shape and selector feature
  alignment transfer from checkpoint-selection to eval without contradictory
  signs.

Changes:

- No scoring, workload, training, or selector production semantics changed.
- Added per-artifact `decision` fields to
  `selection_eval_segment_teacher_transfer_diagnostic` and made the top-level
  decision scope explicit as `primary_artifact_last_input`.
- Ran `selection_eval_segment_teacher_transfer_diagnostic` over the current
  default artifact and the rejected length-support `0.35` artifact.
- Updated `Next-Iterations.md` to treat score/workload/profile weights and
  footprint dimensions as valid research variables while keeping global sanity
  diagnostic-first during the local-query-learning phase.

Tests:

- `uv run --group dev -- python -m orchestration.diagnostics.selection_eval_segment_teacher_transfer_diagnostic ...`
- `python3 -m py_compile orchestration/diagnostics/selection_eval_segment_teacher_transfer_diagnostic.py tests/unit/orchestration/test_query_driven_diagnostics.py`
- `uv run --group dev -- pytest tests/unit/orchestration/test_query_driven_diagnostics.py -q`
  (`8 passed`)
- `uv run --group dev -- ruff check orchestration/diagnostics/selection_eval_segment_teacher_transfer_diagnostic.py tests/unit/orchestration/test_query_driven_diagnostics.py`
- `uv run --group dev -- pyright orchestration/diagnostics/selection_eval_segment_teacher_transfer_diagnostic.py tests/unit/orchestration/test_query_driven_diagnostics.py`
- `git diff --check`

Experiment artifact:

- `artifacts/results/query_driven_length_support_weight035_level3_range_query_mix_seed2524/selection_eval_segment_teacher_transfer_diagnostic.json`

Key results:

- The current default has a viable selection-side teacher shape and remains
  weakly admissible: per-artifact decision
  `guarded_selection_segment_calibration_probe_admissible`, selection/eval
  target Spearman `0.1167`, selection-positive overlap fraction `0.7059`, and
  four selector features with consistent positive selection/eval sign.
- The length-support `0.35` variant is not a safe promotion surface despite its
  better QLU/global sanity: per-artifact decision
  `diagnose_transfer_features_before_guarded_calibration_probe`, selection/eval
  target Spearman falls to `0.0291`, top `1%`, `5%`, and `10%` teacher overlap
  is zero, `segment_score` becomes contradictory-sign, and
  `segment_length_support_score` is consistently negative.
- The diagnostic decision for the length-support artifact is
  `diagnose_transfer_features_before_guarded_calibration_probe`.
- This confirms the user's global-sanity instruction in practice: the
  length-support replay passed global sanity, but it did not prove local-query
  learning coherence and should not advance the evidence boundary.

Decision:

- Do not build the next segment-marginal training probe on the length-support
  `0.35` selector surface.
- If a train-side segment-marginal calibration target is implemented, start
  from the current default selector surface and require non-contradictory
  selection/eval transfer-feature diagnostics before larger strict evidence.
- Keep score weights, anchor/footprint weights, and footprint spatial/temporal
  dimensions available for future checkpoints if gate-by-gate diagnosis shows
  the workload/scoring pair is incoherent.

## Checkpoint Group 25 - Train-Side Marginal Diagnostic Instrumentation

Status: completed / Level 1 implementation evidence only.

Goal:

- Add a non-leaky train-side exact retained-decision marginal diagnostic path
  so future checkpoints can compare train, checkpoint-selection, and eval
  segment-marginal teacher surfaces before changing training semantics.

Hypothesis:

- A train-side marginal teacher can be emitted after the trained model freezes a
  train-split mask using only training workloads. This should provide legal
  calibration evidence without passing eval queries into model, feature builder,
  selector, or checkpoint selection.

Changes:

- Added opt-in `--query_local_utility_train_marginal_diagnostics`.
- Generalized selection causality diagnostics so the same exact-marginal
  machinery can label its source split as `train` and write
  `selector_trace_diagnostics.train_primary`.
- Added `train_marginal_causality_diagnostics` to run payloads.
- Generalized the segment teacher transfer diagnostic to compare
  `train_primary` to `eval_primary`, while keeping the existing
  `selection_primary` default.
- Added per-artifact decisions and explicit `primary_artifact_last_input`
  decision scope to the selection-marginal segment calibration diagnostic, so
  multi-artifact derived outputs cannot be mistaken for every artifact sharing
  the top-level decision.
- Kept scoring, workload generation, training targets, model architecture, and
  selector semantics unchanged.

Tests:

- `python3 -m py_compile` on changed config/orchestration/diagnostic/test files
- `uv run --group dev -- pytest tests/unit/orchestration/test_query_driven_diagnostics.py tests/unit/orchestration/test_run_payload.py tests/unit/orchestration/test_query_driven_causality_and_summary.py -q`
  (`31 passed`)
- `uv run --group dev -- ruff check` on changed focused files
- `uv run --group dev -- pyright` on changed focused files
- `git diff --check`

Experiment artifacts:

- Level 1 smoke:
  `artifacts/results/query_driven_train_marginal_diag_level1_smoke_seed2526/example_run.json`
- train/eval transfer diagnostic:
  `artifacts/results/query_driven_train_marginal_diag_level1_smoke_seed2526/train_eval_segment_teacher_transfer_diagnostic.json`

Key results:

- The smoke completed end to end with the train-marginal flag enabled.
- The artifact includes `selector_trace_diagnostics.train_primary` with
  `teacher_usage_split=train`, candidate-eligible train-side teacher shape, and
  `train_marginal_causality_diagnostics.split=train`.
- The train-side teacher selector records `uses_train_queries=true` and
  `uses_eval_queries=false`.
- The train/eval transfer diagnostic can read `train_primary` and emitted a
  per-artifact decision. On this tiny smoke it was
  `guarded_selection_segment_calibration_probe_admissible`, but that is only
  wiring evidence because the run is undersized and gate-blocked.
- Final success stayed false. The smoke blocked on workload stability,
  predictability, prior-predictive alignment, workload signature, learning
  causality, and the full grid.

Decision:

- Keep the opt-in train-side marginal diagnostic path.
- Do not change training semantics from this smoke.
- Next checkpoint should run the train-side marginal diagnostic at Level 2 or
  Level 3 scale under current defaults. Only if train/eval transfer remains
  non-contradictory should a guarded calibration target or second-stage
  training probe be implemented.

## Checkpoint Group 26 - Train-Side Marginal Strict Diagnostics

Status: completed / Level 2 and Level 3 diagnostic evidence; no default
change.

Goal:

- Test whether the opt-in train-side marginal diagnostic remains useful at
  non-smoke scale and whether it justifies a guarded segment-marginal training
  target.

Hypothesis:

- If current-default workloads are healthy, `selector_trace_diagnostics.train_primary`
  should provide non-leaky train-side segment-marginal teacher evidence whose
  transfer features are not contradictory against eval.

Changes:

- No production scoring, workload, training target, model, or selector default
  changed.
- Ran Level 2 and Level 3 current-default single-cell diagnostics with
  `--query_local_utility_train_marginal_diagnostics`.
- Built train/eval transfer diagnostics from `train_primary` to `eval_primary`
  for both artifacts.

Tests:

- Level 2 strict diagnostic:
  `uv run --group dev -- python -m orchestration.train_and_score ...`
- Level 3 strict diagnostic:
  `uv run --group dev -- python -m orchestration.train_and_score ...`
- Train/eval transfer diagnostics:
  `uv run --group dev -- python -m orchestration.diagnostics.selection_eval_segment_teacher_transfer_diagnostic ...`

Experiment artifacts:

- Level 2:
  `artifacts/results/query_driven_train_marginal_diag_level2_range_query_mix_seed2527/example_run.json`
- Level 2 transfer:
  `artifacts/results/query_driven_train_marginal_diag_level2_range_query_mix_seed2527/train_eval_segment_teacher_transfer_diagnostic.json`
- Level 3:
  `artifacts/results/query_driven_train_marginal_diag_level3_range_query_mix_seed2527/example_run.json`
- Level 3 transfer:
  `artifacts/results/query_driven_train_marginal_diag_level3_range_query_mix_seed2527/train_eval_segment_teacher_transfer_diagnostic.json`

Key results:

- Level 2: MLQDS QueryLocalUtility `0.0726558613`, uniform `0.0652153472`,
  Douglas-Peucker `0.0786932508`; MLQDS beat uniform by `0.0074405141` but
  lost to Douglas-Peucker by `0.0060373895`.
- Level 2 passed workload stability and support overlap. It failed workload
  signature, target diffusion, predictability, learning causality, and global
  sanity.
- Level 2 train/eval transfer emitted decision
  `guarded_selection_segment_calibration_probe_admissible`, but this was not
  promotable because the parent artifact was gate-blocked and teacher target
  Spearman was already negative at `-0.1413`.
- Level 3: MLQDS QueryLocalUtility `0.1181764932`, uniform `0.1109312569`,
  Douglas-Peucker `0.1242589426`; MLQDS beat uniform by `0.0072452364` but
  lost to Douglas-Peucker by `0.0060824493`.
- Level 3 passed support overlap, target diffusion, and global sanity. It
  failed workload stability, workload signature, prior-predictive alignment,
  predictability, and learning causality.
- Level 3 workload stability failed from coverage-overshoot rejection pressure:
  `train_r3` had 99 coverage-overshoot rejections for 48 accepted queries.
- Level 3 workload signature failed on point-hit-fraction KS for `train_r2`
  (`0.2083`) and `selection` (`0.2917`) with anchor and footprint family counts
  balanced, so the drift is in realized point-hit distribution, not planned
  family quota.
- Level 3 causality still failed because prior ablations changed no retained
  masks: shuffled-prior and no-query-prior retained-mask Jaccard were both
  `1.0`, and their QueryLocalUtility deltas were both `0.0`.
- Level 3 segment allocation remained high-entropy and score-dominated:
  normalized entropy `0.9862`, segment-score/allocation Spearman `0.8166`,
  length-support/allocation Spearman `0.0284`, and
  segment-score/length-support Spearman `0.0314`.
- Level 3 train/eval transfer decision was
  `diagnose_transfer_features_before_guarded_calibration_probe`; target
  Spearman was `-0.4043`, top-k teacher overlap was zero through top `10%`,
  and five selector features were contradictory-sign.
- The train-side path stayed non-leaky: Level 3
  `train_marginal_causality_diagnostics.split=train`,
  `uses_train_queries=true`, `uses_eval_queries=false`, and
  `selector_trace_diagnostics.train_primary` had a candidate train-side
  teacher with 32 point targets and 32 segment targets.

Decision:

- Do not build or promote a segment-marginal calibration target from this
  evidence.
- Do not tune model/head/selector semantics against the Level 3 cell while
  workload stability and workload signature are failed.
- Next checkpoint should localize the `range_query_mix` coverage/query-count
  pressure and realized point-hit-fraction drift. Candidate levers are query
  count, target coverage, anchor/footprint weights, and footprint
  spatial/temporal dimensions, but any change needs smaller evidence levels
  before it becomes a default.

## Checkpoint Group 27 - Workload/Profile Scale And Query-Count Localization

Status: completed / workload-profile diagnostic evidence; no default change.

Goal:

- Localize whether the seed `2527` failure was caused by the `range_query_mix`
  profile itself, the 48-query requested floor, or the minimum Level 3
  48-ship/192-point scale.

Hypothesis:

- If the failure is mainly coverage/query-count pressure and small split
  instability, then lowering requested queries and increasing to the previously
  accepted 64-ship/256-point diagnostic scale should clear workload stability
  and workload signature without changing profile definitions.

Changes:

- No production code or default profile changed.
- Ran four diagnostic single-cell contrasts with one training epoch only.
  These are workload/profile localization artifacts, not training-coherence
  evidence.

Tests:

- 48 ships / 192 points / 40 requested queries:
  `uv run --group dev -- python -m orchestration.train_and_score ...`
- 48 ships / 192 points / 32 requested queries:
  `uv run --group dev -- python -m orchestration.train_and_score ...`
- 64 ships / 256 points / 48 requested queries:
  `uv run --group dev -- python -m orchestration.train_and_score ...`
- 64 ships / 256 points / 40 requested queries:
  `uv run --group dev -- python -m orchestration.train_and_score ...`

Experiment artifacts:

- `artifacts/results/query_driven_profile_query_count40_diag_level3_range_query_mix_seed2527/example_run.json`
- `artifacts/results/query_driven_profile_query_count32_diag_level3_range_query_mix_seed2527/example_run.json`
- `artifacts/results/query_driven_profile_scale64_diag_level3_range_query_mix_seed2527/example_run.json`
- `artifacts/results/query_driven_profile_scale64_query40_diag_level3_range_query_mix_seed2527/example_run.json`

Key results:

- 48/192 with 40 requested queries passed workload stability but still failed
  workload signature on checkpoint-selection point-hit-fraction KS `0.225`.
- 48/192 with 32 requested queries passed workload stability but still failed
  workload signature on `train_r2` point-hit-fraction KS `0.2132`.
- 64/256 with 48 requested queries passed workload signature but failed
  workload stability: `train_r0` and `train_r1` hit high rejection rate and
  coverage-overshoot pressure.
- 64/256 with 40 requested queries passed workload stability, workload
  signature, support overlap, target diffusion, prior-predictive alignment, and
  global sanity. It generated 40-44 queries per workload with no child gate
  failures in workload stability or workload signature.
- The successful workload/profile diagnostic still used only one epoch. Its
  model scores and predictability are not accepted training evidence even
  though the artifact's final summary only blocked on learning causality and
  the final grid.

Decision:

- Do not change profile family weights or footprint dimensions from this
  checkpoint.
- Treat 64 ships / 256 points / 40 requested queries as the next strict
  single-cell diagnostic shape for seed `2527`.
- Next checkpoint should run a proper Level 3 train-marginal replay at this
  shape with 5-10 epochs and unchanged gates before model/selector changes or
  segment-marginal training semantics.

## Checkpoint Group 28 - Healthy-Workload Train-Marginal Replay

Status: completed / strict single-cell diagnostic evidence; no default change.

Goal:

- Test whether the healthy 64-ship/256-point/40-requested-query workload shape
  resolves the earlier seed `2527` train-marginal blocker enough to diagnose
  semantic learning causality cleanly.

Hypothesis:

- If the earlier contradictory train/eval marginal result was mainly caused by
  unhealthy workload generation, then a proper 5-epoch Level 3 replay at
  64/256/40 should keep workload gates green and localize any remaining
  failure to model/target/selector causality.

Expected files changed:

- Documentation/progress only unless the replay exposed a wiring bug.

Stop condition:

- Produce the strict replay artifact and train/eval transfer diagnostic, then
  either clear workload gates for semantic diagnosis or fail by a localized
  workload/profile gate.

Changes:

- No production code, default profile, scoring weights, or selector settings
  changed.
- Ran a proper Level 3 train-marginal replay with 5 epochs and train-side
  marginal diagnostics enabled.
- Ran the train/eval segment-teacher transfer diagnostic using
  `selector_trace_diagnostics.train_primary`.

Tests:

- `uv run --group dev -- python -m orchestration.train_and_score ...`
- `uv run --group dev -- python -m orchestration.diagnostics.selection_eval_segment_teacher_transfer_diagnostic ...`

Experiment artifacts:

- `artifacts/results/query_driven_train_marginal_scale64_query40_level3_range_query_mix_seed2527/example_run.json`
- `artifacts/results/query_driven_train_marginal_scale64_query40_level3_range_query_mix_seed2527/train_eval_segment_teacher_transfer_diagnostic.json`

Key results:

- MLQDS QueryLocalUtility: `0.1394788551`; uniform:
  `0.1247681518`; Douglas-Peucker: `0.1153266238`.
- Passed workload stability, support overlap, target diffusion, workload
  signature, predictability, prior-predictive alignment, and global sanity.
- Failed learning causality and the final grid requirement.
- Failed causality children: `shuffled_prior_fields_should_lose`,
  `without_query_prior_features_should_lose`, and
  `without_behavior_utility_head_should_lose`.
- Healthy causality controls: shuffled scores lose by `0.0218749319`,
  untrained loses by `0.0242913109`, prior-field-only loses by `0.0283540770`,
  and no-segment-budget loses by `0.0108408237`.
- Query-prior ablations are immaterial or wrong-way: shuffled-prior and
  no-query-prior deltas are both `-0.0000086480`, with only 4 retained
  decisions changed.
- No-behavior-head is also wrong-way at `-0.0003343468`.
- Segment allocation remains high-entropy and score-dominated:
  entropy `0.9869`, segment-score/allocation Spearman `0.8664`,
  length-support/allocation Spearman `0.1639`,
  segment-score/length-support Spearman `0.0472`, top-20%
  score/length-support overlap `0.2105`.
- Train/eval transfer diagnostic still rejects guarded segment-marginal
  calibration: decision
  `diagnose_transfer_features_before_guarded_calibration_probe`,
  selection/eval target Spearman `-0.6151`, top-k overlap zero through top
  `10%`.
- Training selection restored epoch 1; later epochs degraded validation
  selection score.

Decision:

- Treat 64/256/40 as the next useful strict single-cell diagnostic shape for
  seed `2527`; do not claim final success.
- Do not build a segment-marginal calibration training target from this
  evidence. The train-side teacher is non-leaky and shape-viable, but
  train/eval transfer is contradictory.
- The next checkpoint should diagnose query-prior feature flow,
  behavior-head target/loss transfer, and selector marginal alignment before
  any model/selector architecture change.
- Global sanity passed here and should keep being reported, but it is not the
  initial hard blocker while query-prior/behavior-head causality is failed.

## Checkpoint Group 29 - Head-Contrast Loss Diagnostic

Status: completed / rejected non-default diagnostic; no default change.

Goal:

- Test whether the latest causality failure is mainly a head-contrast/loss
  pressure problem, given that the query-prior fields have predictive signal but
  model head probabilities barely move when query priors are removed.

Hypothesis:

- If raw sparse-head BCE and missing behavior ranking pressure are flattening
  the query-hit and behavior heads, then enabling sparse-head rank loss,
  window-max-normalized sparse-head BCE targets, and behavior-rank loss should
  increase useful prior/head dependence enough for the query-prior and
  behavior-head causality children to move in the right direction.

Expected files changed:

- Artifacts and docs only. No production code should change unless the run
  exposes a root implementation bug.

Stop condition:

- Run the strict 64/256/40 single-cell probe and train/eval transfer diagnostic,
  then reject the loss settings unless they materially improve the failed
  causality children under unchanged gates.

Changes:

- No production code, default profile, scoring weights, or selector settings
  changed.
- Ran a non-default strict diagnostic with
  `query_local_utility_sparse_head_rank_loss_weight=0.25`,
  `query_local_utility_sparse_head_bce_target_mode=window_max_normalized`, and
  `query_local_utility_behavior_rank_loss_weight=0.15`.
- Ran the train/eval segment-teacher transfer diagnostic using
  `selector_trace_diagnostics.train_primary`.

Tests:

- `uv run --group dev -- python -m orchestration.train_and_score ...`
- `uv run --group dev -- python -m orchestration.diagnostics.selection_eval_segment_teacher_transfer_diagnostic ...`

Experiment artifacts:

- `artifacts/results/query_driven_head_contrast_sparse025_behavior015_level3_scale64_query40_seed2527/example_run.json`
- `artifacts/results/query_driven_head_contrast_sparse025_behavior015_level3_scale64_query40_seed2527/train_eval_segment_teacher_transfer_diagnostic.json`

Key results:

- MLQDS QueryLocalUtility: `0.1402280700`; uniform:
  `0.1247681518`; Douglas-Peucker: `0.1153266238`.
- Passed workload stability, support overlap, target diffusion, workload
  signature, predictability, prior-predictive alignment, and global sanity.
- Failed learning causality and final grid.
- Failed causality children were unchanged:
  `shuffled_prior_fields_should_lose`,
  `without_query_prior_features_should_lose`, and
  `without_behavior_utility_head_should_lose`.
- Shuffled-prior and no-query-prior were both wrong-way at `-0.0006932385`,
  worse than the current-default `-0.0000086480`.
- No-behavior-head stayed wrong-way at `-0.0000416291`.
- No-query-prior changed head probabilities by only `0.0000149776` on average
  and changed only 6 retained decisions.
- Segment allocation stayed `score_dominated_length_support_conflict`:
  segment-score/allocation Spearman `0.8713`, length-support/allocation
  Spearman `0.1482`, segment-score/length-support Spearman `0.0294`, top-20%
  score/length-support overlap `0.2105`.
- Train/eval transfer diagnostic still rejected guarded segment-marginal
  calibration: decision
  `diagnose_transfer_features_before_guarded_calibration_probe`,
  selection/eval target Spearman `-0.6785`, top-k overlap zero through top
  `10%`.
- Checkpoint selection restored epoch 1; later epochs increased prediction
  variance but degraded validation selection.

Decision:

- Reject sparse-head normalization plus behavior-rank pressure as a default
  fix.
- The failure is not solved by adding scalar head-loss pressure. The next
  checkpoint should diagnose or redesign the model-facing query-prior path and
  behavior/head target coupling at the architecture/target level, while keeping
  the 64/256/40 strict cell as the comparison shape.
- Do not build a segment-marginal calibration training target from this
  evidence.

## Checkpoint Group 30 - Prior-Transform Contrast Diagnostic

Status: completed / rejected non-default diagnostic; production default
restored.

Goal:

- Test whether the query-prior causality failure is mainly caused by
  small-probability prior channels being numerically flattened before the model
  heads and selector.

Hypothesis:

- If useful query-prior channels reach the model but are too small under the
  identity probability representation, then a square-root model-facing prior
  transform should increase head movement and make query-prior ablations lose
  under the same strict 64/256/40 evidence cell.

Expected files changed:

- `learning/model_features.py`, `models/workload_blind_range.py`, focused
  tests, and docs/progress during the diagnostic. Production code should be
  restored unless the causality children pass under unchanged gates.

Stop condition:

- Reject the transform unless the strict replay makes query-prior and
  behavior-head ablations directionally correct, keeps workload/profile gates
  green, and avoids creating a new transfer blocker.

Changes:

- Temporarily changed the range-model prior transform from identity
  probabilities to square-root probabilities, keeping `route_density_prior`
  disabled.
- Added focused test coverage for the temporary transform.
- Ran a Level 1 wiring smoke, then a strict 64/256/40 Level 3 replay and
  train/eval segment-teacher transfer diagnostic.
- Reverted the transform from production defaults after the strict probe failed
  the stop condition.

Tests:

- `python3 -m py_compile Range_QDS/learning/model_features.py Range_QDS/models/workload_blind_range.py Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py Range_QDS/tests/unit/orchestration/test_query_driven_causality_and_summary.py`
- `uv run --group dev -- python -m orchestration.train_and_score ...`
- `uv run --group dev -- python -m orchestration.diagnostics.selection_eval_segment_teacher_transfer_diagnostic ...`

Experiment artifacts:

- `artifacts/results/query_driven_prior_sqrt_level1_smoke_seed2530/example_run.json`
- `artifacts/results/query_driven_prior_sqrt_level3_scale64_query40_seed2527/example_run.json`
- `artifacts/results/query_driven_prior_sqrt_level3_scale64_query40_seed2527/train_eval_segment_teacher_transfer_diagnostic.json`

Key results:

- Level 1 confirmed wiring only: model prior feature delta became material, but
  retained-mask change was not scientific evidence.
- Level 3 MLQDS QueryLocalUtility: `0.1396786660`; uniform:
  `0.1247681518`; Douglas-Peucker: `0.1153266238`.
- Workload stability, support overlap, target diffusion, workload signature,
  predictability, prior-predictive alignment, and global sanity passed.
- Learning causality still failed. Failed children stayed
  `shuffled_prior_fields_should_lose`,
  `without_query_prior_features_should_lose`, and
  `without_behavior_utility_head_should_lose`.
- No-query-prior model-feature delta increased to `0.0889404789`, score delta
  to `0.0043487814`, and retained symmetric difference to 12 decisions.
- Head probability movement remained tiny at `0.0000988260`; query-hit head
  probability delta was only `0.0000158745`.
- Shuffled-prior, no-query-prior, and no-behavior deltas were wrong-way:
  `-0.0003197163`, `-0.0003080556`, and `-0.0005148169`.
- Train/eval transfer still rejected guarded segment-marginal calibration:
  decision `diagnose_transfer_features_before_guarded_calibration_probe`,
  selection/eval target Spearman `-0.6084`, top-k overlap zero through top
  `10%`.

Decision:

- Reject square-root prior transformation as an active default.
- Keep the production range model checkpoint schema at `6` and model-facing prior transform
  at identity probability, with `route_density_prior` still disabled.
- The next checkpoint should treat prior-scaling as insufficient evidence.
  Focus on semantic directionality: workload/scoring compatibility,
  target/head transfer, and selector allocation authority under causality
  ablations.
- Global sanity is green in this probe, but that does not matter enough to
  promote a non-causal local-query path.

## Checkpoint Group 31 - Current-Focus Family Transfer Diagnostic

Status: completed / derived strict-artifact diagnostic; diagnostic code cleaned.

Goal:

- Re-check the family/head transfer blocker on the current 64/256/40 strict
  artifact and remove stale historical focus-family noise from the derived
  diagnostic.

Hypothesis:

- If the remaining behavior-head causality failure is a real target/head
  transfer problem, then the current-focus diagnostic should block on active
  `range_query_mix` families (`density`, `medium_operational`) rather than on
  removed historical families.

Expected files changed:

- `orchestration/diagnostics/family_transfer_path_diagnostic.py`,
  `tests/unit/orchestration/test_query_driven_diagnostics.py`, this progress
  log, and Next-Iterations. No production model, scoring, workload, or selector
  default should change.

Stop condition:

- Produce a derived diagnostic artifact for the current strict replay, confirm
  active focus families are used, and stop before target/model changes unless
  the diagnostic identifies a root implementation bug.

Changes:

- Updated the family transfer diagnostic to read active focus families from
  `training_target_diagnostics.query_local_utility_factorized.family_conditioned_target_trainability.focus_families`
  when present, with the historical focus list only as an old-artifact
  fallback.
- Added a regression test proving current artifacts do not emit historical
  `small_local` focus rows when active focus families are present.
- Regenerated the derived diagnostic for the current 64/256/40 strict replay.

Tests:

- `uv run --group dev -- python -m orchestration.diagnostics.family_transfer_path_diagnostic ...`
- `python3 -m py_compile Range_QDS/orchestration/diagnostics/family_transfer_path_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_diagnostics.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_diagnostics.py -k "family_transfer_path"` (`2 passed`)
- `git diff --check`

Experiment artifact:

- `artifacts/results/query_driven_train_marginal_scale64_query40_level3_range_query_mix_seed2527/family_transfer_path_diagnostic.json`

Key results:

- Decision: `continue_family_head_loss_transfer_diagnosis`.
- Focus rows now contain only active current families: `density` and
  `medium_operational`. The stale `small_local` unavailable row is gone for
  current artifacts.
- Both active focus rows block on `conditional_behavior_utility` with
  `target_still_weak`.
- `density`: target ship-evidence Spearman `-0.0615`, fitted ship-evidence
  Spearman `0.0183`, fitted Kendall tau with head target `-0.0474`, prediction
  std to target std ratio `0.0154`.
- `medium_operational`: target ship-evidence Spearman `-0.1081`, fitted
  ship-evidence Spearman `-0.0283`, fitted Kendall tau with head target
  `0.0279`, prediction std to target std ratio `0.0149`.
- Retained marginal alignment is read from the correct layout:
  `selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_local_utility_alignment`.
  Selector-score overall Spearman is `0.3020`.

Decision:

- Do not change scoring weights, workload weights, or selector architecture
  from this checkpoint.
- The next code checkpoint should target behavior-head semantics or target/head
  transfer directly. More scalar prior scaling and generic loss pressure have
  already failed.
- Global sanity is not the blocker here; the active blocker is local
  behavior-head transfer and query-prior/head causality.

## Checkpoint Group 32 - Active-Metric Family Transfer Diagnostic Cleanup

Status: completed / diagnostic layout fixed; no model or scoring defaults
changed.

Goal:

- Prevent the family/head transfer diagnostic from overclaiming from
  ship-evidence proxy rows after active scoring removed explicit
  ship-presence/coverage and boundary/event components.

Hypothesis:

- If the latest behavior-head blocker is being diagnosed through stale proxy
  references, then the derived diagnostic should expose active
  `QueryLocalUtility` retained-marginal head alignment and behavior-head
  component tradeoffs before any target/model change.

Expected files changed:

- `orchestration/diagnostics/family_transfer_path_diagnostic.py`,
  `tests/unit/orchestration/test_query_driven_diagnostics.py`,
  `docs/query-driven-implementation-research-guide.md`,
  `docs/Next-Iterations.md`, and this progress log.

Stop condition:

- Derived diagnostic artifact reports the correct retained-marginal source
  layout, active head-to-marginal alignment, and behavior-head component
  tradeoffs. Stop before training changes.

Changes:

- Added active metric score-component alignment extraction from
  `selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_local_utility_alignment.overall.score_component_alignment`
  and retained-removal alignment under the same retained-marginal artifact
  layout.
- Added behavior/prior/segment causality component tradeoff summaries for the
  main failed ablation paths.
- Updated the summary interpretation so ship-evidence rows are explicitly
  treated as legacy diagnostic proxies, not the primary current-metric
  conclusion.
- Added regression coverage for the new active-metric alignment and component
  tradeoff fields.
- Regenerated the current strict replay's family transfer diagnostic.

Tests:

- `python3 -m py_compile Range_QDS/orchestration/diagnostics/family_transfer_path_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_diagnostics.py`
- `uv run --group dev -- ruff check Range_QDS/orchestration/diagnostics/family_transfer_path_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_diagnostics.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_diagnostics.py -k "family_transfer_path"`
  (`2 passed`)
- `uv run --group dev -- python -m orchestration.diagnostics.family_transfer_path_diagnostic ...`

Experiment artifact:

- `artifacts/results/query_driven_train_marginal_scale64_query40_level3_range_query_mix_seed2527/family_transfer_path_diagnostic.json`

Key results:

- Decision remains `continue_family_head_loss_transfer_diagnosis`.
- Correct active alignment layout is now reported as
  `selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_local_utility_alignment.overall.score_component_alignment`.
- Behavior-head active exact-marginal alignment is weak overall and bad on
  retained removals: overall Spearman `0.0577`, retained-removal Spearman
  `-0.3817`.
- `MLQDS_without_behavior_utility_head` remains wrong-way:
  QueryLocalUtility delta `-0.0003343468`.
- The no-behavior component tradeoff is `primary_minus_ablation`; it is
  dominated by the primary model's `query_local_turn_change_coverage` deficit
  versus no-behavior, weighted delta `-0.0007365`, partly offset by query-point
  recall weighted delta `+0.0002841`.

Decision:

- Do not change defaults from this checkpoint.
- Next implementation work should diagnose behavior-head semantics against
  active `QueryLocalUtility` components, not ship-evidence proxy alignment
  alone.
- Global sanity remains a reported guardrail, not the initial hard blocker.

## Checkpoint Group 33 - Behavior-Head Semantic Alignment Diagnostic

Status: completed / derived strict-artifact diagnostic; no model or scoring
defaults changed.

Goal:

- Determine whether the behavior-head blocker is a current-metric semantic
  problem, not just a stale ship-evidence proxy problem.

Hypothesis:

- If the `conditional_behavior_utility` path is misaligned, then the current
  strict artifact should show weak fitted-head contrast, weak active
  retained-marginal alignment, and behavior-target references that do not point
  cleanly at the selector's active segment-budget path.

Expected files changed:

- `orchestration/diagnostics/family_transfer_path_diagnostic.py`,
  `tests/unit/orchestration/test_query_driven_diagnostics.py`,
  `docs/query-driven-implementation-research-guide.md`,
  `docs/Next-Iterations.md`, and this progress log.

Stop condition:

- Derived diagnostic exposes behavior target source/alignment, fitted-head
  contrast, active retained-marginal alignment, and no-behavior component
  tradeoff with explicit delta convention. Stop before target/model changes.

Changes:

- Added `behavior_head_semantic_alignment` to the family transfer diagnostic.
- The new row reports behavior target variant/source, training mask,
  target-reference alignment, fitted head tau/std contrast, active metric
  retained-marginal alignment, and no-behavior component tradeoff.
- Added semantic status labels for low contrast, weak target fit, retained
  marginal misordering, weak segment-budget alignment, and primary-vs-ablation
  deficits.
- Added `delta_convention=primary_minus_ablation` to causality component
  tradeoff rows to prevent misreading negative component deltas.
- Added regression coverage for the semantic row and delta convention.
- Regenerated the current strict replay's family transfer diagnostic.

Tests:

- `python3 -m py_compile Range_QDS/orchestration/diagnostics/family_transfer_path_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_diagnostics.py`
- `uv run --group dev -- ruff check Range_QDS/orchestration/diagnostics/family_transfer_path_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_diagnostics.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_diagnostics.py -k "family_transfer_path"`
  (`2 passed`)
- `uv run --group dev -- python -m orchestration.diagnostics.family_transfer_path_diagnostic ...`

Experiment artifact:

- `artifacts/results/query_driven_train_marginal_scale64_query40_level3_range_query_mix_seed2527/family_transfer_path_diagnostic.json`

Key results:

- Decision remains `continue_family_head_loss_transfer_diagnosis`.
- Behavior target variant/source:
  `active_local_behavior_change` from
  `query_hit_conditioned_trajectory_change`, masked to query-hit points.
- Strongest behavior-target reference is
  `replacement_representative_value`, Spearman `0.5531`.
- Segment-budget alignment is weak: `segment_budget_target` Spearman `0.0230`.
- Query-hit alignment is also weak: `query_hit_probability` Spearman `0.1247`.
- Fitted behavior head is low contrast: prediction std to target std ratio
  `0.0151`, Kendall tau `0.0606`.
- Active retained-marginal behavior-head alignment remains weak overall
  (`0.0577`) and negative on retained removals (`-0.3817`).
- No-behavior ablation remains wrong-way under the explicit
  `primary_minus_ablation` convention: QueryLocalUtility delta
  `-0.0003343468`, dominated by the primary model's
  `query_local_turn_change_coverage` deficit versus no-behavior.

Decision:

- Do not tune selector floors, behavior-as-segment substitutions, scalar loss
  pressure, or prior scaling from this evidence.
- Next code checkpoint should change behavior target/head semantics only if it
  directly addresses low head contrast and weak segment-budget/query-hit
  coupling, then validate with the guide's smaller evidence ladder before any
  strict Level 3 claim.
- Global sanity remains a reported guardrail, not the first blocker.

## Checkpoint Group 34 - Segment-Aware Behavior Target Wiring

Status: completed / active target semantics changed; Level 1 wiring only, no
learning claim.

Goal:

- Replace the behavior target that mostly learned sparse local change and
  replacement support with a query-hit-masked behavior target that has explicit
  segment/query coupling.

Hypothesis:

- If the behavior head is low-contrast and weakly coupled to segment budget,
  then the active behavior target should keep local behavior-change support but
  reweight it by segment behavior support and segment query-hit support. Adding
  broad positive segment support directly should fail target diffusion and must
  be rejected.

Expected files changed:

- `learning/targets/query_local_utility.py`,
  `tests/unit/learning/test_query_local_utility_targets.py`,
  `docs/query-driven-implementation-research-guide.md`,
  `docs/Next-Iterations.md`, `learning/README.md`, and this progress log.

Stop condition:

- Level 0 target tests pass, a Level 1 smoke confirms the active target path
  runs and target diffusion does not fail from behavior support broadening, and
  no strict learning claim is made.

Changes:

- Added `query_segment_local_behavior_utility` as the active
  `conditional_behavior_utility` target variant for
  `query_local_utility_factorized`.
- The final active formula is multiplicative:
  `normalized_query_hit_conditioned_trajectory_change * (0.45 + 0.35 *
  segment_behavior_support + 0.20 * segment_query_hit_support)`.
- Kept behavior supervision masked to query-hit points.
- Added a focused target test proving segment/query support reweights behavior
  without making every query-hit point positive.
- Rejected an intermediate additive formula during Level 1 wiring because it
  made behavior support broad and failed target diffusion.

Tests:

- `python3 -m py_compile Range_QDS/learning/targets/query_local_utility.py Range_QDS/tests/unit/learning/test_query_local_utility_targets.py`
- `uv run --group dev -- ruff check Range_QDS/learning/targets/query_local_utility.py Range_QDS/tests/unit/learning/test_query_local_utility_targets.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/learning/test_query_local_utility_targets.py -k "conditional_behavior or factorized"`
  (`6 passed`)
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_learning_target_stage.py Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py -k "query_local_utility or target_diffusion or factorized"`
  (`7 passed`)
- `uv run --group dev -- pytest Range_QDS/tests/unit/learning/test_query_local_utility_training.py -k "behavior or factorized"`
  (`7 passed`)
- Level 1 smoke:
  `uv run --group dev -- python -m orchestration.train_and_score ...`

Experiment artifacts:

- rejected additive wiring smoke:
  `artifacts/results/query_driven_behavior_segment_target_level1_smoke_seed2531/example_run.json`
- active multiplicative wiring smoke:
  `artifacts/results/query_driven_behavior_segment_target_mult_level1_smoke_seed2531/example_run.json`

Key results:

- Additive variant failed target diffusion:
  `conditional_behavior_utility:support_fraction_above_max` and
  `conditional_behavior_utility:top5_label_mass_below_min`; behavior support was
  positive on all train points. It is rejected.
- Multiplicative variant passed target diffusion in the Level 1 smoke.
- Behavior support stayed concentrated: support fraction `gt_0.01 = 0.3000`,
  top-5 label mass `0.2505`.
- Behavior target alignment improved in the tiny smoke compared with the old
  strict-artifact failure signature: final-score Spearman `0.6012`,
  query-hit Spearman `0.4876`, and segment-budget Spearman `0.3836`.
- Tiny-smoke model fit is still not evidence of learning: behavior-head Kendall
  tau was `-0.0545`, prediction std to target std stayed low, and the run is too
  small with an unhealthy selection workload.

Decision:

- Keep the multiplicative segment-aware behavior target as the active candidate
  for the next smaller strict evidence level.
- Do not claim learning from the Level 1 smoke.
- Next checkpoint should run a guide-compliant Level 2 strict diagnostic before
  any Level 3 or final-grid claim. If Level 2 fails target diffusion, workload
  health, predictability, or causality, diagnose that gate before changing code.
- Global sanity remains reported, not the initial hard blocker.

## Checkpoint Group 35 - Artifact Retention Cleanup

Status: completed / repository hygiene.

Hypothesis:

- The accumulated `artifacts/` tree contains old pre-canonical sweeps, early QueryLocalUtility
  startup variants, caches, and manual reports that are no longer part of the
  current evidence boundary. Retaining them makes stale runs look more relevant
  than they are.

Expected files changed:

- Generated artifact directories, `artifacts/README.md`, `README.md`,
  `docs/query-driven-implementation-research-guide.md`, and this progress log.

Stop condition:

- Retained result artifacts are limited to the current strict replay and
  rejected-current-path diagnostics, active target smokes, current workload
  profile diagnostics, and older seed-2524 diagnostics that still explain the
  active segment/behavior blockers. Stale documented paths are either removed
  or converted to summary-only historical notes.

Changes:

- Removed `274` stale result directories from `artifacts/results/`.
- Cleared disposable generated contents under `artifacts/cache/` and
  `artifacts/manual/`.
- Retained `26` result directories tied to current evidence or still-relevant
  diagnostic lessons.
- Replaced stale pre-simplification and early QueryLocalUtility artifact path
  references with summary-only retention notes.
- Changed example output paths in docs to placeholders so examples do not look
  like retained evidence.
- Tightened `artifacts/README.md` around artifact retention policy.

Validation:

- Rebuilt the result-directory inventory after cleanup.
- Scanned maintained docs for `artifacts/results/` references and checked them
  against retained result directories.
- Confirmed the only remaining apparent missing result references are
  placeholder examples, not claimed retained evidence.

Decision:

- Treat historical raw run directories as disposable once the progress log
  captures the numbers and decision.
- Do not retain old pre-canonical sweep artifacts or early rejected QueryLocalUtility startup
  variants unless a future checkpoint explicitly depends on re-inspecting their
  raw payloads.

## Checkpoint Group 36 - Canonical Naming Cleanup

Status: completed / naming cleanup.

Hypothesis:

- Chronological names such as `workload_blind_range_v2`,
  `learned_segment_budget_v1`, `range_query_mix_workload_blind_v2`, and
  active-doc `schema 5` wording are no longer useful distinctions. The current
  implementations should own canonical names, while older still-available
  diagnostic paths should use semantic names.

Expected files changed:

- Model/feature registries, model module names, selector public API,
  benchmark profile defaults/scripts, focused tests, maintained docs, retained
  artifact directory names, and this progress log.

Stop condition:

- The current model is addressed as `workload_blind_range`, the old scalar
  scorer is addressed as `scalar_workload_blind_range`, the current selector is
  addressed as `learned_segment_budget`, maintained docs do not present
  chronological names as active defaults, retained artifact references resolve,
  and remaining version/schema fields are limited to artifact compatibility
  metadata or historical notes.

Changes:

- Renamed active model type from `workload_blind_range_v2` to
  `workload_blind_range`.
- Renamed the old scalar scorer model type from `workload_blind_range` to
  `scalar_workload_blind_range` so the active trainable model owns the
  canonical name.
- Renamed the old scalar scorer class from `WorkloadBlindRangeQDSModel` to
  `ScalarWorkloadBlindRangeQDSModel` to avoid confusion with the active
  `WorkloadBlindRangeModel`.
- Renamed `models/workload_blind_range_v2.py` and
  `WorkloadBlindRangeV2Model` to `models/workload_blind_range.py` and
  `WorkloadBlindRangeModel`.
- Renamed active model feature helpers and constants to remove the `V2` suffix.
- Renamed selector type and public helpers from `learned_segment_budget_v1` to
  `learned_segment_budget`.
- Renamed the active benchmark profile and default benchmark artifact/cache
  family from `range_query_mix_workload_blind_v2` /
  `query_driven_workload_blind_v2` to `range_query_mix_workload_blind` /
  `query_driven_workload_blind`.
- Removed active-doc phrasing that used `QueryLocalUtility schema 5` as the
  current metric name. `schema_version` fields remain documented as payload
  compatibility metadata.
- Renamed retained local result directories that used the old `schema5` run-id
  prefix and updated maintained references.

Validation:

- `python3 -m py_compile` on changed Python files.
- `uv run --group dev -- ruff check` on touched implementation and focused
  test files.
- `uv run --group dev -- pyright` on touched implementation surfaces.
- Focused pytest set covering model factory/features, QueryLocalUtility
  training, learned segment-budget selector, protocol/causality gates, retained
  masks, learning target stage, benchmark profile/report regressions, and
  scoring metrics: `259 passed`.
- Stale-name scan found no remaining old names outside this checkpoint's
  rename summary.
- Maintained artifact-reference scan found `26` referenced result directories,
  `26` retained result directories, and `0` missing references.

Decision:

- Keep canonical names for the current path: `QueryLocalUtility`,
  `workload_blind_range`, `learned_segment_budget`, and
  `range_query_mix_workload_blind`.
- Keep `schema_version` fields only as artifact/report compatibility metadata.
- Use semantic variation names for still-available diagnostic paths, such as
  `scalar_workload_blind_range`; do not introduce chronological suffixes for
  ordinary implementation changes.

## Checkpoint Group 37 - Maintained Documentation Cleanup

Status: completed / docs cleanup.

Hypothesis:

- After the canonical naming cleanup, maintained docs still contain a small
  amount of wording that can make old versioned names, old profile families, or
  pre-cleanup artifact pressure look like current guidance.

Expected files changed:

- Maintained Markdown only: root/module READMEs, the implementation/research
  guide, code-layout notes, artifact policy, and this progress log.

Stop condition:

- Maintained docs describe `QueryLocalUtility`, `workload_blind_range`,
  `learned_segment_budget`, `range_query_mix`, and artifact policy as the
  current defaults. Old names remain only in explicit next-iteration guardrails
  or progress-history entries.

Changes:

- Replaced root and guide old-name lists with a shorter rule: older
  metric/profile names and removed workload families are historical only unless
  a later checkpoint deliberately reintroduces one with evidence.
- Removed remaining active-doc `versioned` wording around the workload profile
  contract.
- Updated `selection/README.md` to describe selector schema constants as
  artifact metadata, not active versioned product naming.
- Updated `workloads/README.md` so workload profiles are described as current
  product definitions, and removed the explicit stale family-name list from the
  module README.
- Replaced the stale benchmark run-id example
  `query_driven_v2_seed42_a` with `query_driven_seed42_a`.
- Updated `CODE_LAYOUT.md` to describe artifact growth as an ongoing pruning
  risk instead of claiming the current tree is dominated by artifacts.
- Clarified that `QueryLocalUtility` changes should record artifact/report
  schema metadata only when payload semantics actually change, and should not
  name current implementations by schema number.

Validation:

- `git diff --check`
- Targeted stale-name and old-link scan over active maintained Markdown,
  excluding progress history and next-iteration guardrails.
- Targeted stale-phrasing scan over active maintained Markdown, excluding
  progress history.
- Maintained artifact-reference scan found no missing retained result
  directories.

Decision:

- Keep historical names in `Next-Iterations.md` only where they prevent
  reintroducing known-bad families or explain the current diagnostic boundary.
- Keep historical names in the progress log as history, not active guidance.

## Checkpoint Group 38 - Stale Code Cleanup

Status: completed / code cleanup.

Hypothesis:

- After canonical naming and docs cleanup, stale code is concentrated in
  residual chronological diagnostic names, old profile-version payload fields,
  generated bytecode caches, and obsolete derived diagnostics that still center
  `small_local` and ship-evidence proxy semantics.

Expected files changed:

- Focused workload-generation metadata code, CLI help text, active training
  diagnostics, focused tests, removal of obsolete diagnostic modules, and this
  progress log.

Stop condition:

- Non-doc code no longer references the removed query-ship diagnostic modules,
  old active model/selector names, or old `range_v2` naming; current workload
  profile artifacts no longer emit a chronological profile-version field; and
  focused tests pass.

Changes:

- Removed obsolete derived artifact analyzers
  `query_ship_local_heads_failure_diagnostic.py` and
  `query_ship_max_pool_transfer_diagnostic.py`. Their logic hardcoded the old
  `small_local` / ship-evidence proxy path and is no longer a current
  diagnostic surface.
- Removed the tests that preserved those obsolete diagnostic modules.
- Removed chronological workload-profile `version` metadata from
  `RangeWorkloadProfile`, `workload_profile_metadata`, query-generation
  diagnostics, and workload signatures.
- Updated workload-profile tests to assert that current artifacts do not emit
  `version` / `workload_profile_version`.
- Renamed active model diagnostic payload
  `range_v2_prior_feature_scaling` to
  `workload_blind_range_prior_feature_scaling`.
- Renamed remaining stale `range_v2` / `query_driven_v2` test names and local
  checkpoint filenames to current `workload_blind_range` /
  `query_driven_workload_blind` wording.
- Updated stale workload-profile CLI help text from versioned/default-legacy
  wording to named-profile/current-candidate wording.
- Removed generated `__pycache__` directories from `Range_QDS`.

Validation:

- `python3 -m py_compile` on touched source and test files.
- `uv run --group dev -- ruff check` on touched source and test files.
- `uv run --group dev -- pytest` on focused workload-profile, diagnostics,
  guardrail, protocol-gate, and model-feature suites: `80 passed`.
- `git diff --check`
- Targeted non-doc scans found no references to the removed diagnostic modules,
  old active model/selector symbols, or old `range_v2`/`query_driven_v2`
  naming. Remaining profile-version strings are test assertions that the field
  is absent plus unrelated scoring schema metadata.

Decision:

- Do not keep old derived diagnostic modules just because progress history
  references the commands. The progress log preserves the historical result;
  active code should not preserve a misleading diagnostic surface.
- Keep scoring schema metadata because it is artifact/report compatibility, not
  active product naming.

## Validation

Latest focused validation:

- `python3 -m py_compile` on touched source and test files after Checkpoint
  Group 38.
- `uv run --group dev -- ruff check` on touched source and test files after
  Checkpoint Group 38.
- `uv run --group dev -- pytest` on focused workload-profile, diagnostics,
  guardrail, protocol-gate, and model-feature suites after Checkpoint Group 38
  (`80 passed`).
- `git diff --check` after Checkpoint Group 38.
- Targeted non-doc stale-code scans after Checkpoint Group 38.
- `git diff --check` after Checkpoint Group 37.
- Active maintained-doc stale-name scan after Checkpoint Group 37 found no hits
  outside progress history and next-iteration guardrails.
- Active maintained-doc stale-phrasing scan after Checkpoint Group 37 found no
  hits for the removed active-doc wording.
- Maintained artifact-reference scan after Checkpoint Group 37 found no missing
  retained result directories.
- `python3 -m py_compile` on changed Python files after Checkpoint Group 36.
- `uv run --group dev -- ruff check` on renamed model/selector/benchmark
  implementation and focused test files after Checkpoint Group 36.
- `uv run --group dev -- pyright` on renamed model/selector/benchmark
  implementation surfaces after Checkpoint Group 36.
- `uv run --group dev -- pytest` focused model/selector/orchestration/
  benchmark/scoring suites after Checkpoint Group 36 (`259 passed`).
- Artifact cleanup inventory and retained-reference scan after Checkpoint Group
  35.
- `python3 -m py_compile Range_QDS/learning/targets/query_local_utility.py Range_QDS/tests/unit/learning/test_query_local_utility_targets.py`
- `uv run --group dev -- ruff check Range_QDS/learning/targets/query_local_utility.py Range_QDS/tests/unit/learning/test_query_local_utility_targets.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/learning/test_query_local_utility_targets.py -k "conditional_behavior or factorized"`
  (`6 passed`)
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_learning_target_stage.py Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py -k "query_local_utility or target_diffusion or factorized"`
  (`7 passed`)
- `uv run --group dev -- pytest Range_QDS/tests/unit/learning/test_query_local_utility_training.py -k "behavior or factorized"`
  (`7 passed`)
- Level 1 wiring smoke completed for
  `query_driven_behavior_segment_target_mult_level1_smoke_seed2531`; target
  diffusion passed. This is implementation evidence only.

Previous focused validation:

- `git diff --check`
- `python3 -m py_compile Range_QDS/orchestration/diagnostics/family_transfer_path_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_diagnostics.py`
- `uv run --group dev -- ruff check Range_QDS/orchestration/diagnostics/family_transfer_path_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_diagnostics.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_diagnostics.py -k "family_transfer_path"`
  (`2 passed`)
- Derived family-transfer diagnostic regenerated for the current strict
  64/256/40 artifact.

Earlier focused validation:

- `git diff --check`
- `python3 -m py_compile Range_QDS/learning/model_features.py Range_QDS/models/workload_blind_range.py Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py Range_QDS/tests/unit/orchestration/test_query_driven_causality_and_summary.py`
  (`43 passed`)

Earlier broad docs validation before the organization-only refactors:

- `git diff --check`
- Stale-guide `rg` scan for legacy naming, old metric/profile names, old
  schemas, and stale strict-reference prose. Remaining guide hits are
  explicit historical exclusions or the `range_point_f1` no-fallback caveat.
- Stale-default `rg` scan over maintained docs. Remaining hits are explicit
  historical-name or legacy-diagnostic references.
- Broken old-doc-link `rg` scan for removed legacy filenames.

Earlier focused implementation validation before docs-only condensation:

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
