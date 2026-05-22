# Range_QDS Query-Driven Implementation Progress

This is the condensed context log for the query-driven implementation work.
It keeps evidence boundaries, rejected paths, and next admissible work. Raw
command output and full metric dumps belong in artifacts, not here.

Protocol, gates, and active defaults live in
[`query-driven-implementation-research-guide.md`](query-driven-implementation-research-guide.md).
The immediate handoff lives in [`Next-Iterations.md`](Next-Iterations.md).

## Current Evidence Boundary

Current status: **active, not accepted**.

Current strict reference artifacts:

```text
artifacts/results/additive_qhit_behavior_score_composition_level2_seed2539/example_run.json
artifacts/results/additive_qhit_behavior_score_composition_level2_seed2539/semantic_diagnostic.json
```

Current blocker-localizing artifact:

```text
artifacts/results/additive_level2_child_gate_root_localization/diagnostic.json
```

Known gate state from [`Next-Iterations.md`](Next-Iterations.md):

- passed: workload stability, support overlap, target diffusion, predictability,
  prior-predictive alignment
- failed: workload signature at strict Level 2, learning causality, global
  sanity, final grid

Active conclusion:

- Additive q-hit/behavior composition fixed the immediate target-diffusion and
  behavior-materiality blockers, but it is not promotion evidence.
- Strict Level 2 barely beats uniform and loses badly to Douglas-Peucker:
  MLQDS `0.0995482993`, uniform `0.0992909061`,
  Douglas-Peucker `0.1182249577`.
- Remaining blockers are prior materiality, segment-budget-head materiality,
  the known Level 2 workload-signature KS issue, global sanity/length, and the
  unrun final grid.
- Next admissible step is
  `segment_budget_target_selectivity_candidate_level1_wiring`. Do not run
  Level 2/3 or the final grid until a narrow segment-budget target candidate
  survives Level 1 without degrading QueryLocalUtility, length, target
  diffusion, or segment-head materiality.

Standing rejected paths:

- Do not repeat scalar prior boosts, generic prior residuals, route-density
  exposure, prior adapters, prior-only auxiliary losses, prior-conditioned
  head-input projection, final-head alignment losses, channel-factorized prior
  encoding, selector floors, raw coverage overrides, length scaffolds, pooled
  point-score primary allocation, or top-k segment-rank loss variants without a
  new diagnostic that invalidates the prior rejection.
- Do not loosen gates to make the run pass. Gate failures are part of the
  evidence.

## Maintenance Rule

Append only context-logical checkpoints. A checkpoint should explain what the
artifact proves, what it rejects, and what the next admissible step is. Do not
copy full command lines or every scalar metric unless the number changes the
decision.

## Checkpoint 1 - Baseline Trace and First Strict Failures

Covered phases: `1-6`.

Status: completed diagnostic and implementation boundary.

Key artifacts:

- `artifacts/results/semantic_causality_diagnosis_current_reference/diagnostic.json`
- `artifacts/results/query_driven_behavior_rank_default_level2_seed2532/example_run.json`
- `artifacts/results/query_driven_behavior_rank_default_level2_seed2532/semantic_diagnostic.json`

Retained facts:

- The initial semantic diagnostic found target signal, but the learned heads
  and selector path lost it. Behavior targets had signal; behavior outputs were
  flat. Priors moved in inputs but did not move outputs or retained masks.
  Segment allocation mixed point and segment scores incorrectly.
- Trace rows were extended to include head targets/masks, QLU components,
  family context, query-hit run ids, and prior context. That instrumentation
  remains useful.
- Behavior rank loss default was changed to `0.25`. The strict Level 2 replay
  scored MLQDS `0.1026929755`, uniform `0.1003283552`,
  Douglas-Peucker `0.1121031831`.
- That run passed support overlap, target diffusion, predictability, and
  prior-predictive alignment, but failed workload stability, workload
  signature, learning causality, global sanity, and final success.

Decision:

- Behavior-rank default stayed as an implementation improvement, not as
  acceptance evidence.
- No Level 3 or final grid was justified from this boundary.

## Checkpoint 2 - Rejected Early Behavior and Prior Experiments

Covered phases: `7-13`.

Status: rejected paths cleaned up.

Key artifacts:

- `artifacts/results/query_driven_current_stack_level2_seed2537/example_run.json`
- `artifacts/results/query_driven_current_stack_level2_seed2537/semantic_diagnostic.json`
- `artifacts/results/query_driven_prior_adapter_level2_seed2538/example_run.json`
- `artifacts/results/query_driven_prior_adapter_level2_seed2538/semantic_diagnostic.json`

Retained facts:

- Normalized behavior-rank gaps failed strict Level 2. MLQDS `0.1109622965`
  beat uniform `0.1014750196` but lost to Douglas-Peucker `0.1126127687`;
  predictability, prior materiality, behavior, segment, learning causality, and
  final success still failed.
- The semantic prior-head adapter failed strict Level 2 worse: MLQDS
  `0.0855909368`, uniform `0.1012227315`, Douglas-Peucker `0.0997721666`.
  Prior ablations moved wrong-way or were immaterial.
- Both experiments were reverted. Production returned to schema version `6`
  with behavior-rank default and diagnostic trace instrumentation retained.

Decision:

- Do not repeat normalized behavior-gap weighting or the semantic prior-head
  adapter without a new diagnostic that explains why the strict failures no
  longer apply.

## Checkpoint 3 - Behavior and Prior Materiality Localization

Covered phases: `14-17, 20`.

Status: accepted diagnostic boundary, not promotion evidence.

Key artifacts:

- `artifacts/results/query_driven_behavior_signal_level2_seed2539/example_run.json`
- `artifacts/results/query_prior_materiality_root_diagnosis_seed2539/example_run.json`
- `artifacts/results/prior_feature_integration_stage_diagnosis_seed2539/example_run.json`
- `artifacts/results/prior_to_head_transfer_sensitivity_seed2539/example_run.json`

Retained facts:

- Strict Level 2 behavior-signal replay scored MLQDS `0.1090154720`, uniform
  `0.0992909061`, Douglas-Peucker `0.1182249577`. It passed workload
  stability, support overlap, predictability, prior-predictive alignment, and
  target diffusion, but failed workload signature, learning causality, global
  sanity, and final success.
- Behavior became material by ablation in that artifact:
  no-behavior-head delta `0.0100696`. It was still heavily compressed:
  prediction std / target std `0.02776`.
- Prior child gates were exactly failed: shuffled-prior delta `0.0` and
  no-query-prior delta `0.0`.
- Prior channels were not empty and not simply redundant. They carried target
  signal, but trained heads were invariant. Zero-prior head-probability delta
  was about `4.94e-05`, and final-probability delta was about `7.78e-07`.
- Stage diagnostics showed prior signal survived to shared embeddings, then
  died at the head/output interface. Head-probability/pre-context ratio was
  about `0.024`.
- Prior-to-head transfer classified all `6/6` heads as
  `output_layer_suppresses_prior_direction`.

Decision:

- The blocker was not prior sampling, prior scale, or feature absence. It was
  trained-head/output invariance to prior-sensitive directions.
- Scalar boosts, generic residuals, direct prior adapters, and selector
  threshold fixes were not justified.

## Checkpoint 4 - Rejected Prior Root Fixes and Channel Boundary Diagnostics

Covered phases: `18-19, 21-29`.

Status: rejected root fixes, accepted diagnostics.

Key artifacts:

- `artifacts/results/prior_only_aux_learning_pressure_seed2539/example_run.json`
- `artifacts/results/prior_conditioned_head_input_level1_smoke/example_run.json`
- `artifacts/results/head_output_layer_prior_direction_level1_smoke/example_run.json`
- `artifacts/results/head_output_decision_surface_root_fix_level1_smoke/example_run.json`
- `artifacts/results/prior_channel_direction_decomposition_level2_seed2539/example_run.json`
- `artifacts/results/prior_score_rank_margin_boundary_level2_seed2539/example_run.json`

Retained facts:

- Prior-only auxiliary pressure failed. It lowered strict Level 2 MLQDS to
  `0.1077803118`, left prior deltas at `0.0`, and regressed behavior
  materiality below the threshold.
- Prior-conditioned head-input projection failed at Level 1 and was reverted.
  Head-input deltas moved, but head probabilities and masks did not.
- Prior-direction contrastive loss and final-head decision-surface alignment
  both failed Level 1. They moved diagnostic projections at best, but required
  prior ablations stayed exactly `0.0`.
- Channel decomposition showed prior channels were conflicted, not absent:
  over `36` channel/head pairs, `14` were target-aligned, `10` wrong-way,
  `6` weak/flat, and `6` rank-unavailable.
- Channel ablation did not move retained masks. Max channel score-output delta
  was `0.000245419`, and score top-k Jaccard stayed `1.0`.
- Rank-margin replay showed prior-induced score deltas did not help
  high-marginal missed or under-ranked rows. Required prior ablations left
  retained masks unchanged, with classification
  `prior_delta_non_positive_for_top_marginal_rows`.

Decision:

- The next fix could not be a selector-boundary workaround or another aggregate
  prior boost.
- Required work had to localize where useful high-marginal row movement was
  lost.

## Checkpoint 5 - Row-Delta and Factorized Composition Localization

Covered phases: `30-35`.

Status: accepted diagnostic boundary.

Key artifacts:

- `artifacts/results/prior_marginal_row_delta_path_level2_seed2539/example_run.json`
- `artifacts/results/factorized_prior_delta_composition_level2_seed2539/example_run.json`
- `artifacts/results/factorized_prior_delta_composition_level2_seed2539/semantic_diagnostic.json`

Retained facts:

- Strict row-path diagnosis found positive head-probability movement in places,
  mainly replacement and boundary, but the composed raw score moved
  high-marginal rows wrong-way.
- For required prior ablations, top/missed/under-ranked score-output deltas
  were all `0.0`, retained masks stayed fixed, and segment-score deltas were
  negative.
- Factorized composition proved the dominant negative term was
  `query_hit_product_shapley` on top, missed-high, and under-ranked
  high-marginal rows. Composition residuals were tiny, so this was not a
  diagnostic math artifact.
- Direct q-hit row diagnosis showed q-hit logit/probability deltas were
  negative on all high-marginal groups for both `without_query_prior_features`
  and `shuffled_prior_fields`.
- Joined selector rows showed these rows did have query-hit support. The
  failure was not unsupported q-hit rows. They were behavior-dominated:
  local behavior/interpolation contribution was about `2x-3x` point recall in
  Level 2 derived groups.

Decision:

- Classify the row-level failure as supported-but-behavior-dominated q-hit
  gating conflict.
- Do not rewrite the whole formula or target from this alone; design had to
  address q-hit suppressing behavior-dominated local utility.

## Checkpoint 6 - Q-Hit Target Revisions and Scale Gating

Covered phases: `36-41`.

Status: one broad target rejected, one narrowed target retained as current path,
Level 3 re-entry blocked by learning causality.

Key artifacts:

- `artifacts/results/query_evidence_gate_level2_seed2539/example_run.json`
- `artifacts/results/raw_query_hit_evidence_multiplier_level2_seed2539/example_run.json`
- `artifacts/results/raw_query_hit_evidence_multiplier_level3_signature_reentry_seed2527/example_run.json`

Retained facts:

- Broad `query_evidence_gate_hit_ship_blend` target failed strict target
  diffusion. Final label support at `gt_0.01` was `0.7058531746`, above the
  `0.5` cap. It was rejected.
- Narrow `raw_query_hit_ship_evidence_multiplier` retained raw q-hit scale and
  passed strict Level 2 target diffusion:
  final support `0.0887896825`, q-hit support `0.2244543651`.
- Narrow-target Level 2 scored MLQDS `0.1087337015`, uniform `0.0992909061`,
  Douglas-Peucker `0.1182249577`, but failed workload signature first by gate
  order. The failed check was point-hit-fraction KS, with train KS
  `0.2090620032 > 0.2`.
- Workload-signature diagnosis classified that Level 2 failure as
  target-independent small-split KS instability. The Level 3 reference scale
  had already shown healthy signature behavior.
- Level 3 re-entry for the narrowed target passed workload stability, support
  overlap, target diffusion, workload signature, predictability, and
  prior-predictive alignment. It still failed learning causality and global
  sanity. MLQDS `0.1248364151` barely beat uniform `0.1247681518`.
- Level 3 semantic classifications: prior ignored, behavior target has signal
  but head does not learn it, and segment allocation mixes point and segment
  scores incorrectly.

Decision:

- The workload-signature issue was not a reason to loosen gates.
- The active blocker moved back to learning causality and score-level collapse
  at Level 3 scale.

## Checkpoint 7 - Level 3 Collapse and Score-Composition Diagnosis

Covered phases: `42-45`.

Status: diagnostic and design boundary.

Key artifacts:

- `artifacts/results/narrow_target_score_composition_diagnosis/diagnostic.json`
- `artifacts/results/reference_config_current_target_level2_control_seed2539/example_run.json`
- `artifacts/results/qhit_behavior_composition_root_fix_design/design.json`

Retained facts:

- Current Level 3 differed from the historical healthy Level 3 reference in
  model and scoring defaults: current used larger model defaults, behavior-rank
  weight `0.25`, and `mlqds_score_mode=rank`; historical used smaller model
  defaults, behavior-rank weight `0.0`, and `rank_confidence`.
- Historical-default control with the current target made Level 2 worse, not
  better: MLQDS `0.0850006110` versus uniform `0.0992909061`. The stale-default
  hypothesis was false.
- Narrow-target composition diagnosis classified the primary failure as
  `multiplicative_qhit_gate_suppresses_behavior_local_movement_value`.
- Current Level 3 target support was much sparser than historical:
  q-hit support `0.2504439` versus `0.4135298`; final-label support
  `0.0775036` versus `0.1491477`.
- High-marginal rows still had q-hit support, but local
  behavior/interpolation dominated point recall. Multiplicative q-hit movement
  suppressed these rows before selector allocation could help.

Design decision:

- Keep current head targets.
- Change only scalar score composition from:

```text
score = q_hit * (0.5 + behavior) * (0.75 + 0.25 * replacement)
        + 0.25 * boundary
```

to:

```text
score = (0.50 * q_hit + 0.45 * behavior)
        * (0.75 + 0.25 * replacement)
        + 0.05 * boundary
```

## Checkpoint 8 - Current Additive Composition Boundary

Covered phases: `46-48`.

Status: current strict Level 2 diagnostic boundary, not accepted.

Key artifacts:

- `artifacts/results/additive_qhit_behavior_score_composition_level1_smoke/example_run.json`
- `artifacts/results/additive_qhit_behavior_score_composition_level2_seed2539/example_run.json`
- `artifacts/results/additive_qhit_behavior_score_composition_level2_seed2539/semantic_diagnostic.json`
- `artifacts/results/additive_level2_child_gate_root_localization/diagnostic.json`

Retained facts:

- Active formula:
  `additive_raw_query_hit_and_behavior_with_conditional_replacement_modulation_plus_boundary`.
- Level 1 wiring passed target diffusion and showed expected local movement,
  but it underperformed uniform and was not evidence.
- Strict Level 2 target diffusion passed:
  final support `gt_0.01 = 0.2351190476`, max allowed `0.5`.
- Strict Level 2 scores: MLQDS `0.0995482993`, uniform `0.0992909061`,
  Douglas-Peucker `0.1182249577`.
- Workload signature failed with the known Level 2 point-hit-fraction KS issue.
  Global sanity also failed because length preservation `0.703978` was below
  the `0.75` floor.
- Learning-causality deltas:
  shuffled-score `0.00638639`, untrained `0.0110130`, no-behavior-head
  `0.00622185`, shuffled-prior `0.0`, no-query-prior `0.0`,
  no-segment-budget-head `-0.000488398`, prior-field-only `0.00177562`.
- Behavior became material by ablation but remained weak:
  behavior tau `0.040724`, prediction std / target std about `0.103`.
- Prior localization was unchanged: priors are predictive and reach the model,
  but retained-mask Jaccard stayed `1.0`; high-marginal score-output delta was
  `0.0`.
- Segment localization became the main actionable child path: the target is
  oracle-aligned, but the learned segment head is compressed and non-causal.
  Pooled final point-score allocation was `+0.015237081` above primary in the
  strict diagnostic, but this still had to be tested in the real selector path.

Decision:

- Additive composition remains the current target/score path.
- It is not promotion evidence.
- Next work had to localize or fix segment-budget materiality without hiding
  prior failure or weakening guardrails.

## Checkpoint 9 - Segment Allocation Source Attempt Rejected

Covered phases: `49-51`.

Status: production selector semantic change rejected and reverted. Trace
fidelity instrumentation kept.

Key artifacts:

- `artifacts/results/pooled_point_score_segment_allocation_level1_smoke/example_run.json`
- `artifacts/results/pooled_point_score_segment_allocation_level1_smoke/rejection_diagnostic.json`
- `artifacts/results/pooled_point_score_allocation_failure_diagnosis/diagnostic.json`
- `artifacts/results/segment_allocation_mask_delta_diagnostic/diagnostic.json`

Retained facts:

- Promoting pooled final point-score segment scores to the primary
  `learned_segment_budget` allocation source failed the Level 1 stop condition.
- Same-seed additive Level 1 reference scored QLU `0.1064832750`; pooled
  primary scored `0.0856186098`. Length also dropped slightly.
- Failure classification:
  `counterfactual_to_production_score_to_mask_mismatch`.
- Mask-delta diagnosis made the failure concrete:
  retained count stayed `15`, common retained count was `12`, Jaccard
  `0.6666666667`. The pooled path removed learned points `[82,178,274]` and
  added `[61,157,253]`.
- Removed points carried raw marginal QLU sum `0.0210543920` and query-hit
  count `2`; added points carried raw marginal QLU sum `0.0001858789` and
  query-hit count `0`. The delta matched the observed QLU loss.
- Path-length-support allocation matching the additive reference was rejected
  as query-free guardrail compensation from failed evidence.

Decision:

- Do not promote pooled point-score allocation, path-length allocation, selector
  floors, or raw coverage/guardrail compensation from this line of evidence.
- The segment issue is in learned segment-head compression/ranking, not in a
  simple replacement of the primary allocation source.

## Checkpoint 10 - Segment-Head Compression and Rejected Top-K Rank Loss

Covered phases: `52-53`.

Status: diagnostic accepted, production loss patch rejected and reverted.

Key artifacts:

- `artifacts/results/segment_budget_head_compression_root_diagnostic/diagnostic.json`
- `artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring/example_run.json`
- `artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring/rejection_diagnostic.json`

Retained facts:

- Segment-budget target is broad but not useless. It has oracle-mass Spearman
  `0.8431893688` and top-25% oracle mass recall `0.4900304343`.
- Learned segment head is compressed:
  target std `0.2152189165`, prediction std `0.0109573500`, prediction/target
  std ratio `0.0509125787`, Kendall tau `0.2148176732`.
- Selector follows the bad segment score:
  segment_score_to_allocation_spearman `0.8771587805`.
- At the learned retained boundary, useful scores are positive-aligned while
  segment score is wrong-way: raw-score Spearman `0.7192082111`, query-hit
  branch `0.7441348974`, behavior branch `0.7111436950`, segment-score
  `-0.5381231672`.
- A normalized top-k segment-rank loss patch produced no material Level 1
  change. QLU delta was `0.0`, length delta was `0.0`, prediction/target std
  barely moved, Kendall tau worsened, and learning-causality readout was
  unchanged.

Decision:

- Reject top-k segment-rank loss. Keeping it would leave a failed experiment in
  production.
- Next admissible step:
  `segment_rank_loss_gradient_path_diagnostic`. Measure actual segment-rank
  loss magnitude and gradient contribution against point BCE, pooled segment
  BCE, existing pairwise segment loss, auxiliary-loss scaling, and the primary
  budget loss before adding any new loss term or scalar.

## Checkpoint 11 - Non-Admissible Best-Config Benchmark

Covered phase: `54`.

Status: useful operational datapoint, not acceptance evidence.

Key artifacts:

- `artifacts/results/non_admissible_best_config_single_benchmark/example_run.json`
- `artifacts/results/non_admissible_best_config_single_benchmark/semantic_diagnostic.json`

Retained facts:

- Larger single replay used seed `2527`, `64` ships, `256` points, `40`
  requested queries, `5` epochs, and compression ratio `0.05`, with an audit
  over ratios `0.01-0.30`.
- Primary QLU at ratio `0.05`: MLQDS `0.1309654535`,
  uniform `0.1247681518`, Douglas-Peucker `0.1153266238`.
- Global sanity passed in this non-admissible run:
  length preservation `0.8336178577`, average SED ratio `1.4986013224` under
  the `1.5` cap.
- Pre-causality gates and target diffusion passed. Learning causality still
  failed, and final grid remained unrun.
- Failed child checks:
  shuffled-prior fields, no-query-prior features, and no-behavior-head.
  No-behavior-head delta was `0.0023741583`, below the `0.005` threshold.
  Prior deltas stayed weak: shuffled-prior `0.0`,
  without-query-prior-features `0.0003339241`.
- Segment head was material but poorly aligned:
  no-segment-budget-head delta `0.0153767832`, segment-score retained-marginal
  Spearman `-0.0971248003`.

Decision:

- Do not claim final success from this run.
- It only shows the current configuration can produce a promising surface score
  at one larger point. It does not clear the required learning-causality work.
- Next admissible step at that time was
  `segment_rank_loss_gradient_path_diagnostic`; instrumentation has since been
  added in Checkpoint 12.

## Checkpoint 12 - Segment-Rank Gradient Instrumentation

Covered phase: `55`.

Status: instrumentation wiring accepted; actual strict diagnostic still unrun.

Key artifacts:

- none. The probe was a direct tiny training-fit invocation, not a benchmark
  artifact.

Retained facts:

- Existing local artifacts were unavailable and did not expose the required
  loss/gradient fields.
- The factorized auxiliary loss now exposes canonical decomposed parts while
  preserving the scalar production objective.
- Training-fit diagnostics now emit `segment_rank_loss_gradient_path`, including
  aux-scaled segment point BCE, pooled segment BCE, pairwise segment-rank share,
  segment-level total, aux total, primary budget-rank loss, primary balanced
  point BCE, score L2, gradient norms, and pairwise-to-primary gradient ratios.
- A tiny wiring probe produced `available=true`, `pairwise_rank_count=1`, and a
  finite pairwise-to-primary gradient ratio. This proves only instrumentation
  wiring, not segment-head root cause.

Decision:

- Do not change segment loss weights, target semantics, selector semantics, or
  run a Level 1/2 replay from this wiring probe.
- Next admissible step is
  `segment_rank_loss_gradient_path_diagnostic_actual_measurement`: run the
  smallest diagnostic-only current-config measurement that writes the
  gradient-path artifact, then decide whether the segment-rank path is tiny,
  blocked, dominated, or pointed at the wrong target.

## Checkpoint 13 - Segment-Rank Gradient Actual Measurement

Covered phase: `56`.

Status: diagnostic completed; not acceptance evidence.

Key artifact:

- `artifacts/results/segment_rank_loss_gradient_path_diagnostic/diagnostic.json`

Retained facts:

- The run used the current strict handoff config: seed `2539`, `32` ships,
  `192` points per ship, `24` requested queries, `4` epochs, `4` train workload
  replicates, `range_query_mix`, QueryLocalUtility factorized target, and
  train-side marginal diagnostics enabled.
- The source-stratified synthetic replay required source ids. The handoff did
  not state `synthetic_route_families`, so the diagnostic recorded the explicit
  assumption `synthetic_route_families=4`.
- `segment_rank_loss_gradient_path.available=true`.
- Classification:
  `segment_pairwise_rank_gradient_material`.
- Pairwise segment-rank observations: `20`; pooled segment-BCE observations:
  `20`.
- Pairwise segment-rank output-gradient L2 ratios:
  `0.0485257410` versus primary budget total and `4.0774655132` versus
  factorized point BCE.
- The active segment-budget target is too broad: segment target positive
  fraction `0.9916666667`, `gt_0.01` fraction `0.9416666667`.
- The active segment-budget target is poorly aligned with ship-query evidence:
  aggregate Spearman `0.0825715404`, aggregate top-k overlap `0.0`; in the
  `medium_operational` focus family Spearman is `-0.0921973738` with top-k
  overlap `0.0`.

Decision:

- The segment-rank path is not inactive or gradient-blocked. It is active, but
  its output-gradient is much smaller than the primary budget objective.
- The stronger root cause is target selectivity. The current segment target is
  nearly everywhere positive and points away from the ship-query evidence that
  the segment head needs to expose.
- Do not increase the segment-rank scalar, tune selector allocation, or add a
  length scaffold from this evidence.
- Next admissible step is
  `segment_budget_target_selectivity_candidate_level1_wiring`: pick one narrow
  target candidate already supported by diagnostics, wire only that target
  semantic change, and reject at Level 1 if QueryLocalUtility, length, target
  diffusion, or segment-head causality degrades.
