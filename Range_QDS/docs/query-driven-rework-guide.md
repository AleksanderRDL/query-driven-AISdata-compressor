# Range_QDS Query-Driven Rework Guide

This is the active operating guide for the Range_QDS redesign. It is written for a new engineer or implementation agent continuing the work from the current repository state.

The project is **not** trying to build generic trajectory simplification. The goal is:

> Train from a stable future range-query workload distribution, then compress validation/eval AIS trajectories **before future eval queries are known**, while preserving the points and trajectory evidence most likely to matter for those future queries.

The final result must come from learned workload-blind model behavior. A win caused mostly by query-conditioned inference, temporal scaffolding, checkpoint leakage, historical KNN lookup, or selector tricks is not acceptable.

---

## 1. End-state objective

The desired final system is a query-driven, workload-blind AIS compressor.

At deployment/eval time, the system receives only trajectories and train-derived artifacts. It must produce retained masks before future range queries are known. Later, those future queries should still be answered well from the compressed data.

The final system should satisfy four things at once:

1. **Workload-blind compression**
   - No eval query boxes, query tensors, query/point containment labels, query boundary distances, or eval-query-derived features before retained masks are frozen.

2. **Query-driven learned behavior**
   - The model is trained from generated/historical training workloads.
   - The model learns stable workload priors and query-local behavior value.
   - The learned model materially affects retained masks.

3. **Future-query usefulness**
   - Compressed trajectories preserve likely in-query point mass.
   - Within likely query ranges, retained points should explain ship behavior: presence, entry/exit, crossings, temporal span, turns, local shape, speed/heading changes, and enough local evidence to reconstruct movement.

4. **Sensible global trajectories**
   - Global geometry is not the primary goal, but retained trajectories must not become nonsensical.
   - Endpoint sanity, rough length preservation, and bounded geometry distortion are guardrails.

The target product result is not “best possible geometric simplification.” It is “best useful data retained for likely future range-query workloads.”

---

## 2. Current evidence and active blocker

The current active strict-cell reference is:

```text
artifacts/results/query_driven_v2_checkpoint85_segment_aggregation_current_best_strict_local/example_run.json
```

The latest rejected strict diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint49_segment_context_formula_current_best_strict_local/example_run.json
```

The latest teacher-proxy strict diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint53_teacher_proxy_current_best_strict_local/example_run.json
```

The latest query-free teacher guard-coupling smoke is:

```text
artifacts/results/query_driven_v2_checkpoint54_query_free_teacher_guard_coupling_smoke/example_run.json
```

The latest query-free teacher guard-coupling strict diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint55_query_free_teacher_guard_coupling_current_best_strict_local/example_run.json
```

The latest learned-controllable retained-removal derived diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint56_learned_controllable_retained_removal_diagnosis/learned_controllable_retained_removal_diagnosis.json
```

The latest train/selection-side marginal-teacher smoke is:

```text
artifacts/results/query_driven_v2_checkpoint57_selection_marginal_teacher_smoke/example_run.json
```

The latest train/selection-side marginal-teacher minimum strict diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint58_selection_marginal_teacher_min_strict/example_run.json
```

The latest train/selection-side marginal-teacher standard strict diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint59_selection_marginal_teacher_standard_strict/example_run.json
```

The latest workload/profile health generation diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint60_workload_profile_health_generation_diagnostic/workload_profile_health_generation_diagnostic.json
```

The latest workload-healthy train/selection-side marginal-teacher strict diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint61_selection_marginal_teacher_current_best_strict_local/example_run.json
```

The latest selection-to-eval marginal calibration derived diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint62_selection_to_eval_marginal_calibration_diagnosis/selection_to_eval_marginal_calibration_diagnosis.json
```

The latest selector-shaped loss rejected diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint64_learned_segment_budget_loss_current_best_strict_local/example_run.json
```

The latest selector decision-surface derived diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint65_selector_decision_surface_diagnosis/selector_decision_surface_diagnosis.json
```

The latest checkpoint-side marginal teacher consumer failure diagnosis is:

```text
artifacts/results/query_driven_v2_checkpoint70_checkpoint_teacher_consumer_failure_diagnosis/checkpoint_teacher_consumer_failure_diagnosis.json
```

The latest checkpoint-side marginal teacher hybrid smoke is:

```text
artifacts/results/query_driven_v2_checkpoint71_checkpoint_teacher_hybrid_consumer_smoke/example_run.json
```

The latest checkpoint-side marginal teacher hybrid strict diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint72_checkpoint_teacher_hybrid_current_best_strict_local/example_run.json
```

The latest hybrid strict failure and profile/scoring diagnosis is:

```text
artifacts/results/query_driven_v2_checkpoint73_hybrid_strict_failure_profile_scoring_diagnosis/hybrid_strict_failure_profile_scoring_diagnosis.json
```

The latest workload/scoring compatibility minimum strict diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint74_workload_scoring_compatibility_min_strict/example_run.json
```

The latest workload/scoring compatibility current-best strict diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint75_workload_scoring_compatibility_current_best_strict_local/example_run.json
```

The latest ship-evidence target/scoring strict diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint76_ship_evidence_current_best_strict_local/example_run.json
```

The latest ship-presence segment-budget candidate strict diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint77_ship_presence_segment_budget_candidate_current_best_strict_local/example_run.json
```

The latest rejected blended segment-budget target strict diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint78_query_hit_ship_blend_target_current_best_strict_local/example_run.json
```

The latest rejected final-score/ship segment-budget target strict diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint79_final_score_ship_blend_target_current_best_strict_local/example_run.json
```

The latest derived workload/component compatibility diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint80_workload_component_compatibility_diagnosis/workload_component_compatibility_diagnosis.json
```

The latest derived recalibration candidate diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint81_recalibration_candidate_diagnosis/workload_component_recalibration_candidate_diagnosis.json
```

The latest derived blocker-preserving recalibration diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint82_blocker_preserving_recalibration_diagnosis/workload_component_blocker_preserving_recalibration_diagnosis.json
```

The latest family-conditioned target/head strict diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint83_family_trainability_current_best_strict_local/example_run.json
```

The latest family-local candidate strict diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint84_family_local_candidate_current_best_strict_local/example_run.json
```

The latest segment aggregation strict diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint85_segment_aggregation_current_best_strict_local/example_run.json
```

The latest guarded segment aggregation target strict diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint86_query_ship_max_pool_target_current_best_strict_local/example_run.json
```

The latest derived segment-target transfer diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint87_query_ship_max_pool_transfer_diagnosis/query_ship_max_pool_transfer_diagnosis.json
```

The latest rejected query-ship local-head target strict diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint90_query_ship_local_heads_current_best_strict_local/example_run.json
```

The latest derived query-ship local-head failure diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint91_query_ship_local_heads_failure_diagnosis/query_ship_local_heads_failure_diagnosis.json
```

The latest diffusion-preserving family/head transfer-path diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint92_family_transfer_path_diagnosis/family_transfer_path_diagnosis.json
```

The latest family-conditioned prior predictability strict diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint93_family_prior_predictability_max_pool_current_best_strict_local/example_run.json
```

The latest family-prior transfer-path derived diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint94_family_prior_transfer_path_diagnosis/family_prior_transfer_path_diagnosis.json
```

The latest selector-to-retained-marginal calibration derived diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint95_selector_marginal_calibration_diagnosis/selector_marginal_calibration_diagnosis.json
```

The latest selection-side marginal segment calibration derived diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint96_selection_marginal_segment_calibration_diagnosis/selection_marginal_segment_calibration_diagnosis.json
```

The latest selection-to-eval segment teacher transfer derived diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint97_selection_eval_segment_teacher_transfer_diagnosis/selection_eval_segment_teacher_transfer_diagnosis.json
```

The latest segment transfer-feature admissibility derived diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint98_selection_segment_transfer_feature_admissibility_diagnosis/selection_segment_transfer_feature_admissibility_diagnosis.json
```

The latest guarded segment transfer-calibration strict diagnostic is:

```text
artifacts/results/query_driven_v2_checkpoint99_segment_transfer_calibration_zblend_current_best_strict_local/example_run.json
```

This is still diagnostic evidence, not final acceptance evidence. The final grid has not been run, and final success is not allowed.

Current strict-cell result:

```text
QueryUsefulV1:
  MLQDS:           0.1662115143
  uniform:         0.1421296610
  DouglasPeucker:  0.1671038781

RangeUsefulLegacy:
  MLQDS:           0.1524363397
  uniform:         0.1303214771
  DouglasPeucker:  0.1526760352

Length preservation:
  MLQDS:           0.7915916346
  active minimum:  0.7500000000
```

Latest guarded target strict-cell result:

```text
QueryUsefulV1:
  MLQDS:           0.1673482145
  uniform:         0.1421296610
  DouglasPeucker:  0.1671038781

RangeUsefulLegacy:
  MLQDS:           0.1532247905
  uniform:         0.1303214771
  DouglasPeucker:  0.1526760352
```

Gate status:

```text
Passed:
  workload_stability_gate
  support_overlap_gate
  prior_predictive_alignment_gate
  target_diffusion_gate
  workload_signature_gate
  global_sanity_gates

Blocked:
  predictability_gate
  learning_causality_ablations
  full_workload_profile_compression_grid
```

The previous workload-generation/signature blocker is resolved for this strict
cell by the mode-aware signature invariant. Do not spend the next checkpoint
increasing workload scale, widening caps, or running the full matrix unless a
focused probe shows those gates regressed.

The active blockers are **predictability** and **learning causality**. MLQDS
beats uniform but still narrowly loses to Douglas-Peucker on QueryUsefulV1, so
there is no acceptable product win even before the causality failures.

Predictability gate:

```text
failed:
  spearman_min:
    observed:  0.1109086186
    required:  0.1500000000
  pr_auc_lift_over_base_rate:
    observed:  1.2304850435
    required:  1.2500000000
passed:
  lift_at_1_percent: 1.1339085990
  lift_at_2_percent: 1.4429388677
  lift_at_5_percent: 1.2035399978

query_hit_probability:
  spearman: 0.1010042808
  lift@5:   1.3721010168

segment_budget_target:
  spearman: 0.1545043692
  lift@5:   1.1383350577
```

Failed causality child gates:

```text
shuffled_scores_should_lose:
  required delta: 0.0144491119
  observed delta: 0.0089580664
  shortfall:      0.0054910455

shuffled_prior_fields_should_lose:
  required delta: 0.0050000000
  observed delta: -0.0001133659
  shortfall:      0.0051133659

without_query_prior_features_should_lose:
  required delta: 0.0050000000
  observed delta: 0.0000575989
  shortfall:      0.0049424011

without_behavior_utility_head_should_lose:
  required delta: 0.0050000000
  observed delta: 0.0033472765
  shortfall:      0.0016527235

without_segment_budget_head_should_lose:
  required delta: 0.0050000000
  observed delta: 0.0036430341
  shortfall:      0.0013569659
```

Passing causality child gates:

```text
untrained_model_should_lose:
  margin: 0.0033867379

prior_field_only_should_not_match_trained:
  margin: 0.0017087725
```

Selector control is not the current blocker:

```text
learned-controlled retained-slot fraction: 0.3383413462
required minimum:                         0.2500000000
```

Current interpretation:

- Workload signature is no longer the blocker under the mode-aware invariant.
- Aggregate prior predictability is close but still below the hard gate.
- Query-hit prior lift is useful; segment-budget transfer is weaker. Diagnose
  prior/target alignment before model tuning.
- Score ordering still has weak retained-set marginal value. The retained
  marginal payload shows overall selector-score Spearman `-0.0077522559` and
  raw-score Spearman `-0.0248828079` against exact QueryUsefulV1 marginals.
- The segment-context scalar-score candidate was rejected by checkpoint49. It
  improved fit diagnostics but worsened MLQDS QueryUsefulV1 versus checkpoint42
  and failed predictability plus all learning-causality child gates. The active
  scalar formula is reverted to the accepted point-score contract.
- Prior and head ablations move final masks too little, and removing behavior or
  segment-budget heads does not hurt enough. Do not compensate by weakening
  length repair, adding large temporal scaffolding, or loosening causality gates.
- Checkpoints58-59 prove the row-free selection-side exact marginal teacher
  instrumentation runs beyond smoke and finds learned-controllable candidates,
  but both artifacts fail workload stability, workload signature,
  predictability, prior-predictive alignment, and learning causality. Do not
  tune model, loss, or selector from those failed-gate artifacts.
- Checkpoint59 is especially clear: MLQDS QueryUsefulV1 is `0.1430895194`
  versus uniform `0.1445766821`, and active scores are anti-aligned with
  learned-controllable exact marginal value at standard strict scale. Selection
  raw/selector/segment Spearman are `-0.2474340176`, `-0.2606304985`, and
  `-0.3911290323`; eval raw/selector/segment Spearman are `-0.3885630499`,
  `-0.3958944282`, and `-0.2943548387`.
- Checkpoint60 isolates that failure as scale/profile-health-sensitive.
  The checkpoint59 96-ship split passed workload stability only `1/3` seeds and
  workload signature `0/3`; a 192-ship balanced split passed stability `3/3`
  but signature only `2/3`; the current-best 384-ship balanced split passed
  workload stability and signature `5/5`. Use the 384 balanced scale for the
  next strict teacher diagnostic unless a cheaper profile fix is explicitly
  being tested as workload evidence only.
- Checkpoint61 reran the selection-side exact marginal teacher at that
  workload-healthy 384 balanced scale. Workload stability, workload signature,
  support overlap, target diffusion, prior-predictive alignment, and global
  sanity pass. Predictability and learning causality still fail. The
  selection-side learned-controllable teacher has 32 candidates and is viable
  as a train/checkpoint calibration signal, but active raw/selector scores are
  anti-aligned with exact selection marginal value: Spearman `-0.1616568915`
  and `-0.2562316716`.
- Checkpoint62 derives selection-to-eval marginal calibration from checkpoint61
  learned-controllable rows. It does not justify a production calibration path:
  the best fitted transfer candidate has eval Spearman `0.2749266862`, but
  selection leave-one-out Spearman is `-0.1173020528` and eval
  top-minus-bottom marginal is `-0.0003171658`. Current row features are not a
  robust train-side calibration signal.
- Checkpoint63-64 tested an explicit learned-segment-budget-shaped loss
  objective, then removed the objective from production paths after strict
  rejection. Checkpoint64 passed workload/profile gates but worsened MLQDS
  QueryUsefulV1 to `0.1630146227`, still lost to Douglas-Peucker
  `0.1671038781`, and made learning causality worse: shuffled-score delta
  dropped to `0.0017380253`, untrained delta became `-0.0021686561`, and
  no-segment-budget-head delta was only `0.0021650137`. Do not re-add this
  loss without new root evidence.
- Checkpoint65 joins checkpoint61 learned-controllable retained-removal exact
  marginal rows to selector segment attribution. All 32 selection rows join.
  The top exact-marginal quartile is under-ranked by both point selector score
  and point segment score in `6/8` rows, and `6/8` also live in the lower half
  of selector segment ranks. The bottom exact-marginal quartile has better mean
  selector segment rank (`332.375`) than the top quartile (`603.0`). This
  points to a separated segment-level plus within-segment point-level marginal
  teacher, not another scalar proxy over current labels.
- Checkpoint 5.122 removes a diagnostic usability trap found by checkpoint65:
  future retained-marginal rows now carry `selector_segment_context` directly
  when the learned-segment trace has segment attribution. This does not change
  model or selector behavior. It prevents future checkpoints from repeating a
  fragile external join between point rows and segment rows.
- Checkpoint 5.123 adds a diagnostic-only separated marginal teacher
  construction over bounded exact retained-removal rows. It produces
  segment-level targets from positive exact marginal mass by selector segment
  and within-segment point targets from point exact marginals, while excluding
  skeleton, fallback, and length-repair-owned rows. It is not yet wired into
  training, and it must not be treated as production calibration.
- Checkpoint 5.124 proves the separated marginal teacher payload is emitted
  end to end in a real Level 1 artifact. Eval trace emits 4
  learned-controllable rows, 2 segment targets, and 4 point targets; selection
  trace emits 2 learned-controllable rows, 1 segment target, and 2 point
  targets. This is schema/runtime evidence only: the smoke fails workload,
  predictability, prior-alignment, target diffusion, workload signature,
  causality, and global sanity gates.
- Checkpoint 5.125 separates target-shape viability from train/checkpoint usage
  eligibility. Eval-side exact marginal payloads now keep
  `candidate_for_train_side_teacher=false` even when target shape is viable;
  checkpoint-selection payloads can mark the candidate true when their target
  shape is viable. This is still Level 1 implementation evidence only.
- Checkpoint 5.126 adds a guarded diagnostic consumer for full
  checkpoint-selection separated teacher rows. It rejects eval payloads and
  compact row-free summaries before building selector score vectors. In the
  Level 1 smoke, the checkpoint teacher selector score is `0.1193385015`,
  `+0.0024014111` over the selection primary, but this remains smoke-scale
  implementation evidence only.
- Checkpoint 5.127 runs the guarded consumer at the workload-healthy
  384-ship strict cell. Workload/profile gates pass and the consumer remains
  leakage-guarded, but the direct teacher-selector scores `0.1558174990` on
  checkpoint selection versus the primary `0.1601869377`. Do not promote the
  direct consumer to training semantics; diagnose its sparse support and
  ship/boundary/shape component loss first.
- Checkpoint 5.128 diagnoses that strict failure from checkpoint69. The direct
  retained-removal-only consumer has only 32 positive point teacher scores for
  a 1638-point budget and 32 positive segments out of 1008 considered segments.
  It churns the mask heavily and loses mostly through ship/point recall. The
  next teacher must be less sparse or hybridized before any loss wiring.
- Checkpoint 5.129 adds guarded diagnostic-only hybrid checkpoint-teacher
  selectors that blend dense primary selector scores with exact marginal
  teacher vectors. Checkpoint71 proves schema/runtime emission only; its small
  positive smoke deltas are not evidence for training semantics.
- Checkpoint 5.130 tests that hybrid at the workload-healthy strict cell.
  Workload/profile gates pass, but the direct teacher and both hybrid blends
  lose to the checkpoint-selection primary.
- Checkpoint 5.131 diagnoses the strict failure and incorporates
  `docs/keep-in-mind.md`: workload profiles, anchor-family weights, and
  QueryUsefulV1 components are not fixed constants. The next work should
  diagnose workload/scoring compatibility for a coherent query-local trainable
  signal before adding training semantics.

Relevant diagnostics before and around the current-best artifact:

```text
checkpoint23 prior sqrt transform standard strict:
  MLQDS QueryUsefulV1: 0.1523652257
  failed: target diffusion, learning causality
  decision: reject sqrt_probability prior transform

checkpoint24 head dispersion diagnosis:
  current-best factorized final-score prediction_std_to_target_std: 0.0914818734
  low-dispersion heads below 0.10 ratio:
    conditional_behavior_utility
    replacement_representative_value
    segment_budget_target
    path_length_support_target

checkpoint27 dense-head rank standard strict:
  MLQDS QueryUsefulV1: 0.1478283847
  factorized final-score prediction_std_to_target_std: 0.1081234205
  failed: target diffusion, workload signature, learning causality
  shuffled-score delta: 0.0002758070
  decision: reject dense-head rank pressure and remove its plumbing

checkpoint28 score/selector alignment derived diagnosis:
  shuffled-score delta per changed retained decision: 0.0000047513
  without-segment-budget delta per changed decision: 0.0000160170
  without-behavior delta per changed decision: 0.0000255560
  decision: score movement has weak retained-set marginal value

checkpoint29 retained-decision marginal instrumentation:
  old strict artifacts cannot rank exact final retained decisions by marginal
    QueryUsefulV1 because final/source mask indices were missing
  learned segment-budget trace schema: 7
  new query-free trace masks:
    retained_mask
    skeleton_retained_mask
    learned_retained_mask
    fallback_retained_mask
    length_repair_retained_mask
  decision: use a small replay or targeted diagnostic to compute marginal
    alignment by raw score, selector score, segment score, source, and repair
    stage

checkpoint30 retained-marginal helper unit diagnostic:
  added bounded diagnostic helper:
    orchestration.selector_diagnostics.retained_decision_marginal_query_useful_diagnostics
  retained candidates: leave-one-out QueryUsefulV1 loss
  removed candidates: add-one QueryUsefulV1 gain
  alignment fields:
    raw_score
    selector_score
    segment_score
    source
    decision
  decision: helper is implementation evidence only; next evidence needs a small
    replay or diagnostic payload hook on a real current-best-style run

checkpoint31 retained-marginal payload hook:
  payload key:
    selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_useful_alignment
  rule:
    retained-decision marginal alignment lives under the selector trace path.
    Do not use learning_causality_summary.selection_causality_diagnostics as a
    retained-marginal source.
  emitted after the primary MLQDS mask is frozen
  candidate limits:
    retained per source: 32
    removed candidates: 64
  decision: unit-validated hook only; next evidence must be the smallest
    guide-allowed replay that emits this payload on a real learned-selector run

checkpoint32 retained-marginal payload Level 1 smoke:
  MLQDS QueryUsefulV1: 0.1003881274
  uniform QueryUsefulV1: 0.1005303922
  Douglas-Peucker QueryUsefulV1: 0.1042713959
  payload:
    available: true
    diagnostic_only: true
    candidate_count: 72
    score_fields_available: raw_score, selector_score, segment_score
  workload query counts:
    train: 8
    eval: 5
    selection: 0
  failed gates:
    workload stability
    predictability
    prior-predictive alignment
    workload signature
    learning causality
    global sanity
  decision: schema/path evidence only. Zero selection queries means this smoke
    did not cleanly exercise the selector-workload question.

checkpoint33 retained-marginal payload Level 1 smoke with selection queries:
  MLQDS QueryUsefulV1: 0.2912429205
  uniform QueryUsefulV1: 0.2889764732
  Douglas-Peucker QueryUsefulV1: 0.2902431939
  workload query counts:
    train: 8
    train_r1: 8
    eval: 8
    selection: 8
  payload:
    available: true
    diagnostic_only: true
    candidate_count: 74
    score_fields_available: raw_score, selector_score, segment_score
  final retained sources:
    skeleton: 4
    learned: 2
    length_repair: 4
    fallback: 0
  learned-controlled retained-slot fraction: 0.20
  marginal summary:
    learned retained mean removal loss: 0.0004861619
    length-repair retained mean removal loss: 0.0007530456
    skeleton retained mean removal loss: 0.0684498070
    removed candidate mean add-one gain: 0.0067197904
  decision: Level 1 implementation evidence only. The payload works with a
    nonempty selection workload, but quality and causality claims remain
    forbidden at this scale.

checkpoint34 retained-marginal payload Level 2 minimum strict:
  MLQDS QueryUsefulV1: 0.1380248104
  uniform QueryUsefulV1: 0.1096775731
  Douglas-Peucker QueryUsefulV1: 0.1386078304
  gates passed:
    workload stability
    support overlap
    global sanity
  gates failed:
    target diffusion
    workload signature
    predictability
    prior-predictive alignment
    learning causality
  workload generation:
    healthy; no row exhausted; all rows reached target coverage
    signature failed from query-count mismatch and point/ship hit KS distances
  learning causality:
    shuffled-score delta: 0.0074525188 versus required 0.0170083424
    prior ablations changed sampled/model priors but changed 0 retained decisions
  retained-marginal payload:
    available: true
    diagnostic_only: true
    candidate_count: 99
    removed candidate positive add-one gain fraction: 0.8125
    raw/selector scores were negatively aligned with removed-candidate gain
  decision: do not tune model or selector from this artifact. Increase to a
    standard strict single-cell because Level 2 failed workload signature at
    small split/query scale.

checkpoint35 retained-marginal payload standard strict v1:
  MLQDS QueryUsefulV1: 0.1247339820
  uniform QueryUsefulV1: 0.1404554573
  Douglas-Peucker QueryUsefulV1: 0.1345268094
  gates passed:
    workload stability
    support overlap
    global sanity
  gates failed:
    target diffusion
    workload signature
    predictability
    prior-predictive alignment
    learning causality
  workload generation:
    healthy; no row exhausted; all rows reached target coverage
  workload signature:
    failed mainly from synthetic split imbalance. Train query counts were
    89-100, eval query count was 32, and selection query count was 40.
  split caveat:
    this run used default 0.70/0.15/0.15 synthetic fractions. The current-best
    strict artifact used balanced 0.34/0.33/0.33-style splits and passed
    workload signature.
  retained-marginal payload:
    available: true
    diagnostic_only: true
    candidate_count: 137
  decision: invalid for model tuning. Rerun one corrected standard strict
    single-cell with balanced synthetic splits and the local 10% profile before
    changing model, selector, or target behavior.

checkpoint36 retained-marginal payload standard strict balanced local:
  MLQDS QueryUsefulV1: 0.1549194326
  uniform QueryUsefulV1: 0.1152263547
  Douglas-Peucker QueryUsefulV1: 0.1749545436
  gates passed:
    workload stability
    support overlap
    target diffusion
    global sanity
  gates failed:
    workload signature
    predictability
    prior-predictive alignment
    learning causality
  workload generation:
    healthy; no row exhausted; all rows reached target coverage
  workload signature:
    still failed at 96 ships. Train query counts were 32-48, eval query count
    was 32, and selection query count was 33. Some rows failed query-count
    mismatch and point/ship-hit KS checks.
  learning causality:
    shuffled-score delta: -0.0235930255 versus required 0.0238158467
    prior ablations changed sampled/model priors but changed 0 retained
    decisions
  retained-marginal payload:
    available: true
    diagnostic_only: true
    candidate_count: 160
  decision: not interpretable as acceptance evidence because signature still
    fails. The next evidence step needs a larger balanced current-best-scale
    strict cell, or a performance-aware retained-marginal diagnostic before
    running that larger cell.

checkpoint37 retained-marginal cached query support:
  evidence level: implementation only
  change:
    retained-decision marginal QueryUsefulV1 diagnostics now reuse
    ScoringQueryCache for retained-independent range-query support.
  semantics:
    exact QueryUsefulV1 marginals are preserved. The diagnostic still scores
    frozen masks after retained-mask construction and remains diagnostic-only.
  payload metadata:
    exact_query_useful_v1_marginals
    performance_mode
    elapsed_seconds
    query_cache_provided
    query_cache_created
    query_cache_support_mask_count
    query_cache_range_audit_support_count
    query_cache_range_segment_geometry_available
  tests:
    py_compile passed
    ruff passed
    pyright passed
    focused pytest passed: 111 tests
  decision: this removes a diagnostic scaling issue, not a learning blocker.
    Next run should be a larger balanced current-best-scale strict single-cell
    with the cached retained-marginal payload.

checkpoint38 cached retained-marginal current-best-scale strict local:
  MLQDS QueryUsefulV1: 0.1662115143
  uniform QueryUsefulV1: 0.1421296610
  Douglas-Peucker QueryUsefulV1: 0.1671038781
  gates passed:
    workload stability
    support overlap
    target diffusion
    prior-predictive alignment
    global sanity
  gates failed:
    workload signature
    predictability
    learning causality
  workload generation:
    healthy; no row exhausted; all rows reached local 10% target coverage
    train query counts: 118, 148, 153, 139
    eval query count: 144
    selection query count: 126
  workload signature:
    failed only train-vs-eval query_count_mismatch for train_r0:
    relative delta 0.1805555556 versus max 0.15. Anchor, footprint,
    point-hit KS, ship-hit KS, duplicate, and broad-query checks passed.
  predictability:
    failed spearman_min and pr_auc_lift_over_base_rate:
    Spearman 0.1109086186 versus min 0.15
    PR-AUC lift 1.2304850435 versus min 1.25
    lift@5 passed narrowly: 1.2035399978 versus min 1.2
  learning causality:
    failed shuffled scores, shuffled priors, no query priors, no behavior head,
    and no segment-budget head.
    shuffled-score delta 0.0089580664 versus required 0.0144491119
    shuffled-prior delta -0.0001133659 versus required 0.005
    no-query-prior delta 0.0000575989 versus required 0.005
    no-behavior-head delta 0.0033472765 versus required 0.005
    no-segment-budget-head delta 0.0036430341 versus required 0.005
  retained-marginal payload:
    available: true
    diagnostic_only: true
    exact_query_useful_v1_marginals: true
    performance_mode: exact_cached_query_support
    elapsed_seconds: 17.8225840520
    candidate_count: 160
    overall raw Spearman: -0.0248828079
    overall selector Spearman: -0.0077522559
    retained-removal selector top-minus-bottom marginal: -0.0000446724
  runtime:
    total pipeline runtime: 606.68s
    freeze-retained-masks runtime: 351.32s
    retained-marginal payload runtime: 17.82s
  decision: failed before model conclusions are admissible because workload
    signature failed. Do not tune model/selector from this artifact. Next work
    should diagnose workload-profile/query-count stability at current-best scale
    and instrument the remaining retained-mask freeze cost.

checkpoint39 retained-mask freeze timing instrumentation:
  evidence level: implementation only
  change:
    retained-mask freezing now emits query-free timing diagnostics:
      retained_mask_freeze_timing
      retained_mask_ablation_freeze_timing
  timing coverage:
    primary method simplify seconds
    audit method simplify seconds
    selector trace reconstruction
    retained-marginal alignment
    score-protected length diagnostics
    query-free ablation freeze total
    ablation substage seconds
    prior-channel ablation seconds
    method count
    failure count
    total seconds
  tests:
    py_compile passed
    ruff passed
    pyright passed
    focused pytest passed: 111 tests
  decision: this does not change masks, scoring, or gates. It makes the next
    strict rerun auditable enough to locate the checkpoint38 freeze bottleneck.

checkpoint40 workload query-count stability generation-only:
  evidence level: targeted generation diagnostic
  scale:
    384 ships, 256 points, 4 route families, balanced 0.34/0.33 split,
    48 minimum queries, 256 max queries, 4 train workload replicates,
    range_workload_v1_local, profile_sampled_query_count
  seeds:
    2324, 2325, 2326, 2327, 2328
  signature results:
    pass: 2/5
    fail: 3/5
    failure mode: query_count_mismatch only
  query-count range:
    minimum observed row count: 101
    maximum observed row count: 197
  generation health:
    every workload reached target coverage
    every workload stopped with target_coverage_reached
  interpretation:
    the local 10% profile is seed/split-sensitive under the strict
    query-count signature check. This is not generator exhaustion and not model
    evidence.
  decision: next work must stabilize profile/query-count behavior or revise the
    signature invariant in the guide; do not tune model/selector from these
    artifacts.

checkpoint41 query-count signature invariant:
  evidence level: targeted generation diagnostic plus gate implementation
  accepted-query floor probe:
    n_queries 160:
      signature pass 4/5
      workload stability pass 2/5
      failure modes:
        query_count_mismatch
        range_generation_rejection_rate_too_high
        coverage_guard_rejection_pressure_too_high
        range_acceptance_or_coverage_guard_exhausted
    n_queries 192:
      signature pass 5/5
      workload stability pass 0/5
      failure modes:
        range_generation_rejection_rate_too_high
        coverage_guard_rejection_pressure_too_high
        range_acceptance_or_coverage_guard_exhausted
    decision:
      raising the accepted-query floor is not a valid root fix because it makes
      the coverage guard/rejection-pressure gates fail.
  mode-aware signature gate:
    fixed-count or legacy signatures:
      enforce relative query-count parity
    calibrated_to_coverage + profile_sampled_query_count signatures:
      require matching profile id, query_count_mode, coverage_calibration_mode,
      and target_coverage; enforce minimum query count and distribution checks;
      record query-count relative delta as diagnostic instead of a parity
      blocker
    validation:
      checkpoint40 scale rerun after the gate change passed workload signature
      and workload stability in 5/5 nearby seeds. Query counts still ranged
      from 101 to 197, but all pairs used
      diagnostic_min_only_for_coverage_calibrated and did not enforce relative
      query-count parity.
    decision:
      this is a guide-level invariant change, not a model success claim. Do
      not use it to loosen learning-causality, predictability, support, or
      global-sanity gates.

checkpoint42 mode-aware current-best strict local:
  MLQDS QueryUsefulV1: 0.1662115143
  uniform QueryUsefulV1: 0.1421296610
  Douglas-Peucker QueryUsefulV1: 0.1671038781
  gates passed:
    workload stability
    support overlap
    prior-predictive alignment
    target diffusion
    workload signature
    global sanity
  gates failed:
    predictability
    learning causality
  final claim:
    final_success_allowed: false
    reason: candidate_blocked_by_required_gates
  predictability:
    Spearman 0.1109086186 versus min 0.15
    PR-AUC lift 1.2304850435 versus min 1.25
    lift@5 passed at 1.2035399978
  learning causality:
    failed shuffled scores, shuffled priors, no query priors, no behavior head,
    and no segment-budget head
    shuffled-score delta 0.0089580664 versus required 0.0144491119
    no-query-prior delta 0.0000575989 versus required 0.005
  retained-marginal payload:
    available: true
    exact_query_useful_v1_marginals: true
    candidate_count: 160
    overall raw Spearman: -0.0248828079
    overall selector Spearman: -0.0077522559
    retained-removal selector top-minus-bottom marginal: -0.0000446724
  timing:
    total runtime: 625.69s
    freeze-retained-masks: 363.45s
    retained-marginal alignment: 17.79s
    score-protected length diagnostics: 63.28s
    query-free ablation freeze: 260.07s
  decision:
    workload gate is now clean enough. Next work should diagnose why
    train-derived priors and learned heads do not translate into marginally
    valuable retained decisions; do not run the final grid or tune from
    generation-only evidence.
```

These diagnostics matter. Prior rescaling and generic head-fit/ranking pressure
are not the next rational levers. The dense-head rank probe improved fit
diagnostics while degrading retained-mask usefulness and learning causality.
Better factorized head fit alone is not evidence of learned workload-blind
success. The next evidence must tie scores to retained-decision marginal value,
not only to factorized-label fit or mask movement.

Latest prior-path sensitivity from the derived diagnosis:

```text
shuffled_prior_fields:
  sampled_prior_mean_abs_delta:       0.1004762650
  model_input_prior_mean_abs_delta:   0.0101600057
  head_probability_mean_abs_delta:    0.0000115752
  score_mean_abs_delta:               0.0002874043
  retained_symmetric_difference:      16
  retained_mask_jaccard:              0.9904306220

without_query_prior_features:
  sampled_prior_mean_abs_delta:       0.1055359766
  model_input_prior_mean_abs_delta:   0.0106786611
  head_probability_mean_abs_delta:    0.0000116865
  score_mean_abs_delta:               0.0002997211
  retained_symmetric_difference:      24
  retained_mask_jaccard:              0.9856801909
```

Future prior-ablation artifacts should expose one canonical diagnostic chain:

```text
sampled_prior_features
model_prior_features
head_output
raw_prediction
score_output
retained_mask
```

Do not reintroduce `selector_score` as a compatibility alias for this prior-ablation payload. `score_output` is the canonical score-stage name.

checkpoint43 derived prior/head/selector marginal diagnosis:
  artifact:
    artifacts/results/query_driven_v2_checkpoint43_prior_head_selector_marginal_diagnosis/prior_head_selector_marginal_diagnosis.json
  evidence level:
    derived_strict_artifact_diagnostic_no_new_probe
  strict source:
    checkpoint42 mode-aware current-best strict local
  decision:
    no new success claim
    final_success_allowed remains false
    final grid remains blocked
  blocker classification:
    workload signature: resolved
    predictability: still blocking but close; aggregate misses Spearman/PR-AUC
      while several individual prior channels have useful lift
    prior-to-head transfer: blocking; prior fields reach sampled/model inputs
      but barely move head probabilities, scores, or retained masks
    head fit: mixed; query-hit and segment-budget heads carry signal, while
      behavior, boundary-event, and path-length heads are weak or flat
    selector marginal alignment: blocking; raw, selector, and segment scores
      rank exact retained-decision marginal QueryUsefulV1 weakly or negatively
    learning causality: blocking; score/segment ablations move masks but do not
      clear required material-delta gates, and prior ablations barely move masks

checkpoint44 exact marginal under-rank diagnosis and layout fix:
  artifact:
    artifacts/results/query_driven_v2_checkpoint44_exact_marginal_under_rank_diagnosis/exact_marginal_under_rank_diagnosis.json
  evidence level:
    derived_strict_artifact_diagnostic_no_new_probe
  strict source:
    checkpoint42 mode-aware current-best strict local
  key row-level finding:
    checkpoint42 bounded rows prove score/marginal under-ranking. The second
    highest exact marginal candidate was a length-repair retained point with
    marginal rank 2/160 but selector-score rank 147/160. A learned retained
    point had marginal rank 3/160 but selector-score rank 112/160.
  current artifact limitation:
    checkpoint42 rows contain raw_score, selector_score, and segment_score only.
    They prove under-ranking but cannot explain head-level score composition.
  code fix:
    future retained-marginal rows include diagnostic-only score_components for
    factorized head probabilities and composed score terms.
    retained-marginal alignment and full bounded rows remain under
    selector_trace_diagnostics.eval_primary.
    learning_causality_summary.selection_causality_diagnostics is not a
    canonical retained-marginal location.
  decision:
    no new success claim
    final_success_allowed remains false
    final grid remains blocked

checkpoint45 retained-marginal component payload smoke:
  artifact:
    artifacts/results/query_driven_v2_checkpoint45_retained_marginal_component_payload_smoke/example_run.json
  evidence level:
    implementation_payload_smoke_only
  key finding:
    retained-marginal alignment is available under
    selector_trace_diagnostics.eval_primary.
    row score_components include factorized head probabilities and composed
    score terms.
  limitation:
    tiny smoke, zero accepted selection-workload queries, failed gates; no
    learning or success claim.

checkpoint46 score-formula composition static diagnosis:
  artifact:
    artifacts/results/query_driven_v2_checkpoint46_score_formula_composition_static_diagnosis/score_formula_composition_static_diagnosis.json
  evidence level:
    static_code_and_strict_artifact_diagnostic_no_new_probe
  key finding:
    final raw point score uses query_hit, behavior, replacement, and boundary,
    but omits segment_budget_target and path_length_support_target.
    In checkpoint42, segment_budget_target is the strongest head by tau/top-k
    fit, yet it reaches the selector mainly through segment allocation and only
    a 0.05 point-score blend. Path-length support is configured out and also
    fits poorly, so turning its blend on would be a masking fix.
  decision:
    no new success claim
    final_success_allowed remains false
    final grid remains blocked

checkpoint47 segment-context score formula implementation smoke:
  artifact:
    artifacts/results/query_driven_v2_checkpoint47_segment_context_score_formula_smoke/example_run.json
  evidence level:
    Level 1 implementation/runtime smoke only
  code change:
    QueryUsefulV1 scalar labels and workload_blind_range_v2 final logits now
    share the same formula. The local point score is blended with
    segment_budget_target context at weight 0.15. path_length_support_target
    stays out of the scalar score because the current strict evidence shows it
    is weak and disabled in allocation.
  smoke result:
    MLQDS QueryUsefulV1: 0.1140836937
    uniform QueryUsefulV1: 0.1135960307
    Douglas-Peucker QueryUsefulV1: 0.2382488243
    failed gates: workload stability, predictability, prior-predictive
      alignment, target diffusion, workload signature, learning causality,
      global sanity
    selection workload: 0 accepted queries
  decision:
    schema/runtime evidence only
    no learning or success claim
    final_success_allowed remains false
    final grid remains blocked

checkpoint48 segment-context formula minimum strict diagnostic:
  artifact:
    artifacts/results/query_driven_v2_checkpoint48_segment_context_score_formula_min_strict/example_run.json
  evidence level:
    Level 2 minimum strict diagnostic
  scale:
    32 ships, 128 points, 3 route families, 24 accepted queries per workload,
    4 train workload replicates, 3 epochs, 5% compression
  key result:
    MLQDS QueryUsefulV1: 0.0806495175
    uniform QueryUsefulV1: 0.0527182052
    Douglas-Peucker QueryUsefulV1: 0.1970008932
  gate result:
    passed: support overlap
    failed: workload stability, workload signature, target diffusion,
      predictability, prior-predictive alignment, learning causality, global
      sanity
  blocker classification:
    workload stability failed from high rejection rate and coverage-guard
    pressure across train/eval/selection workloads
    workload signature failed point/ship-hit KS checks despite equal query
    counts and matched family distributions
    target diffusion failed because final label support fraction was 0.6445
    versus max 0.5
    global sanity failed length preservation, 0.5979 versus min 0.75
  decision:
    do not tune model or selector from checkpoint48
    the Level 2 scale is not clean enough to judge the formula fix
    final_success_allowed remains false
    final grid remains blocked

checkpoint49 segment-context formula current-best strict diagnostic:
  artifact:
    artifacts/results/query_driven_v2_checkpoint49_segment_context_formula_current_best_strict_local/example_run.json
  evidence level:
    Level 3 current-best strict local diagnostic
  scale:
    384 ships, 256 points, 4 route families, 48 accepted queries per workload,
    4 train workload replicates, 3 epochs, 5% compression
  key result:
    MLQDS QueryUsefulV1: 0.1555318121
    uniform QueryUsefulV1: 0.1421296610
    Douglas-Peucker QueryUsefulV1: 0.1671038781
  gate result:
    passed: workload stability, support overlap, target diffusion, workload
      signature, prior-predictive alignment, global sanity
    failed: predictability, learning causality
  blocker classification:
    predictability Spearman passed at 0.1546, but PR-AUC lift, lift@1, and
    lift@5 failed
    learning causality failed all child gates
    final-score/head fit improved, but product score worsened versus
    checkpoint42 and still lost to Douglas-Peucker
    retained-marginal overall alignment improved, but retained-removal
    alignment stayed weak/negative and segment context hurt the composed score
    there
  decision:
    reject the segment-context scalar-score formula
    revert active scalar labels and v2 final logits to the accepted point-score
      formula
    keep retained-marginal layout and component diagnostics
    final_success_allowed remains false
    final grid remains blocked

checkpoint50 top-marginal-miss payload smoke:
  artifact:
    artifacts/results/query_driven_v2_checkpoint50_top_marginal_miss_payload_smoke/example_run.json
  evidence level:
    Level 1 schema/runtime smoke only
  guidance source:
    docs/Next-Iterations.md now guides the next diagnostic sequence
  code change:
    retained-marginal rows now include trajectory index, selector stage state,
    head probabilities/logits, sampled prior channels, model-facing prior
    channels, QueryUsefulV1 score components, candidate ranks, component-vs-
    marginal rank deltas, and heuristic failure buckets
  smoke result:
    MLQDS QueryUsefulV1: 0.1165340475
    uniform QueryUsefulV1: 0.1134645206
    Douglas-Peucker QueryUsefulV1: 0.1135542137
    retained-marginal candidate rows: 72
    failed gates: workload stability, predictability, prior-predictive
      alignment, workload signature, learning causality, global sanity
  decision:
    schema evidence only
    no learning or success claim
    final_success_allowed remains false
    final grid remains blocked

checkpoint51 top-marginal-miss current-best strict diagnostic:
  artifact:
    artifacts/results/query_driven_v2_checkpoint51_top_marginal_miss_current_best_strict_local/example_run.json
  evidence level:
    Level 3 current-best strict local diagnostic
  key result:
    MLQDS QueryUsefulV1: 0.1662115143
    uniform QueryUsefulV1: 0.1421296610
    Douglas-Peucker QueryUsefulV1: 0.1671038781
  gate result:
    passed: workload stability, support overlap, target diffusion, workload
      signature, prior-predictive alignment, global sanity
    failed: predictability, learning causality
  blocker classification:
    top-marginal-miss rows show prior/model-prior channels are present, so this
    is not a missing-prior-support failure
    retained-removal rows are under-ranked by active raw, selector, segment,
    query-hit, behavior, replacement, and segment-budget components
    path-length-support head probability aligns better with retained-removal
    marginal, but path-head training fit is still weak/negative
    skeleton and length-repair state explains many high-marginal removal rows
  decision:
    do not turn path support on as a scalar blend
    next code checkpoint should build a train-only retained-removal/path-support
      marginal teacher or selector calibration diagnostic
    final_success_allowed remains false
    final grid remains blocked

checkpoint52 teacher-proxy payload smoke:
  artifact:
    artifacts/results/query_driven_v2_checkpoint52_teacher_proxy_payload_smoke/example_run.json
  evidence level:
    Level 1 schema/runtime smoke only
  key result:
    teacher-proxy fields for endpoint support, path-length support, and
      endpoint-or-path support are wired into retained-marginal diagnostics
    smoke gates fail broadly and are not learning evidence
  decision:
    schema evidence only
    no learning or success claim
    final_success_allowed remains false
    final grid remains blocked

checkpoint53 teacher-proxy current-best strict diagnostic:
  artifact:
    artifacts/results/query_driven_v2_checkpoint53_teacher_proxy_current_best_strict_local/example_run.json
  evidence level:
    Level 3 current-best strict local diagnostic
  key result:
    MLQDS QueryUsefulV1: 0.1662115143
    uniform QueryUsefulV1: 0.1421296610
    Douglas-Peucker QueryUsefulV1: 0.1671038781
  gate result:
    passed: workload stability, support overlap, target diffusion, workload
      signature, prior-predictive alignment, global sanity
    failed: predictability, learning causality
  blocker classification:
    endpoint support is the strongest query-free retained-removal proxy in this
      strict cell: retained-removal Spearman 0.5557900341, top-minus-bottom
      marginal 0.0001728310
    path-length support is anti-aligned here: retained-removal Spearman
      -0.1745653832, top-minus-bottom marginal -0.0002565908
    endpoint-or-path support is weaker than endpoint-only and has negative
      retained-removal top-minus-bottom marginal
    top-marginal-miss bucket counts still point at skeleton/length-repair and
      score suppression, not missing prior fields
  decision:
    do not turn path support on as a scalar blend
    do not claim success from endpoint-proxy alignment while causality fails
    next code checkpoint should build or diagnose a train-side retained-removal
      teacher/calibration path centered on endpoint support, while separating
      real marginal value from endpoint/global-sanity guard satisfaction
    final_success_allowed remains false
    final grid remains blocked

checkpoint54 query-free teacher guard-coupling payload:
  artifact:
    artifacts/results/query_driven_v2_checkpoint54_query_free_teacher_guard_coupling_smoke/example_run.json
  evidence level:
    Level 1 schema/runtime smoke only
  key result:
    misleading train_only_teacher_* diagnostic field names were replaced with
      query_free_teacher_* names
    query_free_teacher_proxy_guard_coupling_summary now separates aggregate
      retained-removal rows from learned-controllable, non-guard, guard-owned,
      skeleton, length-repair, and removed-addition rows
    the compact learning-causality retained-marginal summary carries the
      guard-coupling summary but still excludes full rows
  smoke finding:
    guard_coupling_suspected: true
    all retained endpoint top-minus-bottom: 0.0604293668
    learned-controllable endpoint top-minus-bottom: -0.0000582940
    guard-owned endpoint top-minus-bottom: 0.0117840235
  decision:
    schema and misuse-prevention evidence only
    do not build an endpoint teacher from aggregate endpoint alignment alone
    rerun the current-best strict cell with guard-coupling diagnostics before
      accepting endpoint support as a teacher target or selector calibration
      feature
    final_success_allowed remains false
    final grid remains blocked

checkpoint55 query-free teacher guard-coupling current-best strict:
  artifact:
    artifacts/results/query_driven_v2_checkpoint55_query_free_teacher_guard_coupling_current_best_strict_local/example_run.json
  evidence level:
    Level 3 current-best strict local diagnostic
  key result:
    MLQDS QueryUsefulV1: 0.1662115143
    uniform QueryUsefulV1: 0.1421296610
    Douglas-Peucker QueryUsefulV1: 0.1671038781
  gate result:
    passed: workload stability, support overlap, target diffusion, workload
      signature, prior-predictive alignment, global sanity
    failed: predictability, learning causality
  guard-coupling result:
    retained-marginal candidates: 160
    query_free_teacher_proxies present; old train_only_teacher_proxies absent
    selector trace keeps full rows; learning-causality summary is compact
    all retained endpoint top-minus-bottom: 0.0001728310
    learned-controllable endpoint top-minus-bottom: -0.0002858786
    guard-owned endpoint top-minus-bottom: 0.0001468642
    guard_coupling_suspected: true
  decision:
    aggregate endpoint support is not a valid teacher target
    do not build an endpoint teacher from checkpoint53 aggregate alignment
    next checkpoint should target learned-controllable retained-removal
      alignment directly, excluding skeleton/length-repair guard-owned effects
    final_success_allowed remains false
    final grid remains blocked

checkpoint56 learned-controllable retained-removal diagnosis:
  artifact:
    artifacts/results/query_driven_v2_checkpoint56_learned_controllable_retained_removal_diagnosis/learned_controllable_retained_removal_diagnosis.json
  evidence level:
    derived_strict_artifact_diagnostic_no_new_probe
  code fix:
    constant-valued diagnostic fields now report unavailable with
      reason: no_value_variation, preventing fake top-minus-bottom marginal
      summaries from stable sorting
  key result:
    learned-controllable retained-removal candidates: 32
    exact marginal positive fraction: 1.0
    max exact marginal: 0.0016853170
    raw score top-minus-bottom: -0.0001884099
    selector score top-minus-bottom: 0.0000460299
    segment score top-minus-bottom: -0.0001754074
    endpoint support has no value variation on learned-controllable rows
    path support Spearman: -0.0163707962
    the top learned-marginal row is ranked last within learned rows by raw and
      selector scores, 30/32 by segment score, and has no query-free proxy
      support
  decision:
    endpoint, path, endpoint-or-path, raw score, selector score, and segment
      score are not valid learned-controllable teacher targets
    next checkpoint should instrument a train-side exact or bounded marginal
      teacher for learned-controllable candidates, excluding skeleton and
      length-repair guard-owned effects
    final_success_allowed remains false
    final grid remains blocked

checkpoint57 selection-side marginal teacher smoke:
  artifact:
    artifacts/results/query_driven_v2_checkpoint57_selection_marginal_teacher_smoke/example_run.json
  evidence level:
    Level 1 schema/runtime smoke only
  code change:
    selector_trace_diagnostics now has selection_primary beside eval_primary.
    Full selection-workload exact retained-decision marginal rows live under
      selector_trace_diagnostics.selection_primary.retained_decision_marginal_query_useful_alignment.rows.
    learning_causality_summary.selection_causality_diagnostics exposes compact
      row-free selection_retained_decision_marginal_teacher.
    retained marginal diagnostics now include a
      learned_controllable_marginal_teacher_summary that excludes skeleton,
      fallback, and length-repair guard-owned retained-removal rows.
  smoke result:
    MLQDS QueryUsefulV1: 0.1274302100
    uniform QueryUsefulV1: 0.1262302903
    Douglas-Peucker QueryUsefulV1: 0.2389240113
    selection marginal rows: 72
    learned-controllable retained-removal teacher rows: 1
    selection trace matches frozen primary selection mask: true
    failed gates: workload stability, predictability, prior-predictive
      alignment, workload signature, learning causality, global sanity
  decision:
    schema/runtime evidence only
    no teacher usefulness, learning, or success claim
    next evidence should use a strict diagnostic scale before changing loss or
      selector calibration
    final_success_allowed remains false
    final grid remains blocked

checkpoint58 selection-side marginal teacher minimum strict:
  artifact:
    artifacts/results/query_driven_v2_checkpoint58_selection_marginal_teacher_min_strict/example_run.json
  evidence level:
    Level 2 minimum strict diagnostic
  key result:
    MLQDS QueryUsefulV1: 0.1392315675
    uniform QueryUsefulV1: 0.1055556512
    Douglas-Peucker QueryUsefulV1: 0.1120862571
    gates passed: support overlap, target diffusion, global sanity
    gates failed: workload stability, workload signature, predictability,
      prior-predictive alignment, learning causality
    selection marginal rows: 137
    learned-controllable selection teacher candidates: 25
    selection teacher candidate_for_train_side_calibration: true
    selection raw/selector Spearman: 0.3561538462 / 0.3615384615
    compact teacher summaries are row-free; full rows remain under
      selector_trace_diagnostics.selection_primary and eval_primary
  decision:
    strict-scale teacher support exists, but this is not clean model evidence
      because workload/signature gates failed
    do not tune loss or selector from checkpoint58 alone
    final_success_allowed remains false
    final grid remains blocked

checkpoint59 selection-side marginal teacher standard strict:
  artifact:
    artifacts/results/query_driven_v2_checkpoint59_selection_marginal_teacher_standard_strict/example_run.json
  evidence level:
    Level 3 standard strict diagnostic
  key result:
    MLQDS QueryUsefulV1: 0.1430895194
    uniform QueryUsefulV1: 0.1445766821
    Douglas-Peucker QueryUsefulV1: 0.1385893349
    gates passed: support overlap, target diffusion, global sanity
    gates failed: workload stability, workload signature, predictability,
      prior-predictive alignment, learning causality
    workload stability failed coverage-guard rejection pressure on train_r2
      and selection
    workload signature failed train_r1/train_r2 point-hit and ship-hit KS
    selection marginal rows: 160
    learned-controllable selection teacher candidates: 32
    selection teacher candidate_for_train_side_calibration: true
    selection raw/selector/segment Spearman: -0.2474340176 / -0.2606304985 /
      -0.3911290323
    eval raw/selector/segment Spearman: -0.3885630499 / -0.3958944282 /
      -0.2943548387
  decision:
    selection teacher candidate support survives, but active scores do not rank
      exact learned-controllable retained-removal marginal value at this scale
    checkpoint59 is blocked before model conclusions are admissible because
      workload stability and signature fail
    next work must restore strict workload/profile health before loss,
      calibration, or selector changes
    final_success_allowed remains false
    final grid remains blocked

checkpoint60 workload/profile health generation diagnostic:
  artifact:
    artifacts/results/query_driven_v2_checkpoint60_workload_profile_health_generation_diagnostic/workload_profile_health_generation_diagnostic.json
  evidence level:
    targeted_generation_only_diagnostic
  hypothesis:
    checkpoint59 workload failures are scale/profile-generation issues that
      must be cleared before model/loss/selector tuning
  scenarios:
    checkpoint59_scale_unbalanced_96:
      seeds: 2728, 2729, 2730
      workload stability pass rate: 1/3
      workload signature pass rate: 0/3
      failures: coverage-guard rejection pressure, high rejection rate, and
        point/ship-hit KS drift
    medium_balanced_192:
      seeds: 2324, 2325, 2326
      workload stability pass rate: 3/3
      workload signature pass rate: 2/3
      failures: residual point/ship-hit KS drift
    current_best_balanced_384:
      seeds: 2324, 2325, 2326, 2327, 2328
      workload stability pass rate: 5/5
      workload signature pass rate: 5/5
  decision:
    checkpoint59 and 192-ship diagnostics are not clean enough for model tuning
    current-best 384-ship balanced scale restores workload/profile health in
      the current worktree
    next strict teacher diagnostic should use the 384 balanced scale with
      selection marginal diagnostics retained
    final_success_allowed remains false
    final grid remains blocked

checkpoint61 selection-side marginal teacher current-best strict:
  artifact:
    artifacts/results/query_driven_v2_checkpoint61_selection_marginal_teacher_current_best_strict_local/example_run.json
  evidence level:
    Level 3 current-best strict local diagnostic
  scale:
    384 ships, 256 points, 4 route families, balanced 0.34/0.33 split,
    48 minimum queries, 256 max queries, 4 train workload replicates,
    range_workload_v1_local, profile_sampled_query_count
  key result:
    MLQDS QueryUsefulV1: 0.1662115143
    uniform QueryUsefulV1: 0.1421296610
    Douglas-Peucker QueryUsefulV1: 0.1671038781
    MLQDS RangeUsefulLegacy: 0.1524363397
    uniform RangeUsefulLegacy: 0.1303214771
    Douglas-Peucker RangeUsefulLegacy: 0.1526760352
    gates passed: workload stability, support overlap, target diffusion,
      workload signature, prior-predictive alignment, global sanity
    gates failed: predictability, learning causality
  predictability:
    Spearman: 0.1109086186 versus min 0.15
    PR-AUC lift: 1.2304850435 versus min 1.25
    lift@1, lift@2, lift@5 pass
    prior-predictive alignment passes
  learning causality:
    failed shuffled scores, shuffled priors, no query priors, no behavior head,
      and no segment-budget head
    shuffled-score delta: 0.0089580664 versus required 0.0144491119
    shuffled-prior delta: -0.0001133659 versus required 0.005
    no-query-prior delta: 0.0000575989 versus required 0.005
    no-behavior-head delta: 0.0033472765 versus required 0.005
    no-segment-budget-head delta: 0.0036430341 versus required 0.005
  selection marginal teacher:
    full selection rows: 160
    learned-controllable selection teacher candidates: 32
    candidate_for_train_side_calibration: true
    selection raw/selector/segment Spearman: -0.1616568915 / -0.2562316716 /
      -0.1755865103
    eval raw/selector/segment Spearman: 0.2085777126 / 0.0909090909 /
      0.1880498534, but raw and segment top-minus-bottom marginals remain
      negative
    compact teacher summaries are row-free; full rows remain under
      selector_trace_diagnostics.selection_primary and eval_primary
  runtime:
    total runtime: 643.12s
    selection retained-marginal payload: 17.04s
    eval retained-marginal payload: 18.46s
    freeze-retained-masks: 353.56s
  decision:
    workload/profile health is clean enough to move past workload diagnosis
    final success remains blocked by predictability and learning causality
    active scores do not rank learned-controllable exact marginal value on the
      selection workload
    next code checkpoint can target train/checkpoint-side retained-removal
      marginal calibration or loss/selector alignment, but must prove
      improvement under unchanged strict gates
    final_success_allowed remains false
    final grid remains blocked

checkpoint62 selection-to-eval marginal calibration diagnosis:
  artifact:
    artifacts/results/query_driven_v2_checkpoint62_selection_to_eval_marginal_calibration_diagnosis/selection_to_eval_marginal_calibration_diagnosis.json
  evidence level:
    derived diagnostic only from checkpoint61
  hypothesis:
    a workload-blind calibration trained on selection learned-controllable
      retained-removal exact marginal rows should transfer to eval
      learned-controllable rows better than current active scores
  filters:
    source: learned
    decision: retained_removal_loss
    excludes skeleton, fallback, and length-repair guard-owned rows
  key result:
    selection candidate count: 32
    eval candidate count: 32
    best active eval Spearman: 0.2085777126
    required transfer Spearman with margin: 0.2585777126
    best transfer candidate: score-component ridge, eval Spearman 0.2749266862
    best transfer selection leave-one-out Spearman: -0.1173020528
    best transfer eval top-minus-bottom marginal: -0.0003171658
    candidate_for_production_calibration: false
  decision:
    do not add production calibration from current retained-marginal row
      features
    post-hoc row-feature calibration is not the next root fix
    next work should target loss/selector alignment or a stronger train-side
      marginal target, still excluding guard-owned effects
    final_success_allowed remains false
    final grid remains blocked

checkpoint63 learned-segment-budget loss smoke:
  artifact:
    artifacts/results/query_driven_v2_checkpoint63_learned_segment_budget_loss_smoke/example_run.json
  evidence level:
    Level 1 schema/runtime smoke only
  hypothesis:
    a selector-shaped loss that separates learned-segment allocation from point
      choice should provide a cleaner path for changing learned-controlled
      retained decisions than post-hoc row-feature calibration
  code change:
    temporarily added an explicit learned_segment_budget_topk loss objective for
      the checkpoint
  smoke result:
    artifact emitted and retained-marginal layouts stayed valid
    MLQDS QueryUsefulV1: 0.0300736024
    uniform QueryUsefulV1: 0.0989417253
    Douglas-Peucker QueryUsefulV1: 0.1068160240
    failed gates: workload stability, predictability, prior-predictive
      alignment, workload signature, learning causality, global sanity
  decision:
    schema/runtime evidence only
    no learning claim
    strict workload-healthy evidence required before accepting or rejecting the
      loss direction

checkpoint64 learned-segment-budget loss current-best strict:
  artifact:
    artifacts/results/query_driven_v2_checkpoint64_learned_segment_budget_loss_current_best_strict_local/example_run.json
  evidence level:
    Level 3 current-best strict local diagnostic
  key result:
    MLQDS QueryUsefulV1: 0.1630146227
    uniform QueryUsefulV1: 0.1421296610
    Douglas-Peucker QueryUsefulV1: 0.1671038781
    MLQDS RangeUsefulLegacy: 0.1484470460
    uniform RangeUsefulLegacy: 0.1303214771
    Douglas-Peucker RangeUsefulLegacy: 0.1526760352
    gates passed: workload stability, support overlap, target diffusion,
      workload signature, prior-predictive alignment, global sanity
    gates failed: predictability, learning causality
  predictability:
    unchanged versus checkpoint61 because this audit is target/prior-side:
      Spearman 0.1109086186 and PR-AUC lift 1.2304850435 still fail
  learning causality:
    worse than checkpoint61
    failed shuffled scores, untrained model, shuffled priors, no query priors,
      no behavior head, no segment-budget head, and prior-only checks
    shuffled-score delta: 0.0017380253
    untrained delta: -0.0021686561
    shuffled-prior delta: -0.0000202293
    no-query-prior delta: -0.0000059541
    no-behavior-head delta: 0.0036746171
    no-segment-budget-head delta: 0.0021650137
  retained marginal teacher:
    selection learned-controllable raw/selector/segment Spearman:
      -0.1495601173 / -0.1946480938 / -0.1664222874
    eval learned-controllable raw/selector/segment Spearman:
      -0.2514662757 / -0.1070381232 / -0.3145161290
  decision:
    reject the learned_segment_budget_topk loss direction
    remove the temporary objective from production code and CLI
    improved segment-head target fit is not enough when exact retained-marginal
      alignment and causality get worse
    next work should target exact marginal teacher construction or selector
      decision-surface diagnostics, not another proxy loss over current scalar
      labels
    final_success_allowed remains false
    final grid remains blocked

Current next checkpoint direction:

```text
Primary hypothesis:
  The segment-context scalar-score formula is not the root fix. Checkpoint53
  and checkpoint55 show the retained-set blocker is not missing prior support;
  it is weak learned-controllable retained-removal marginal alignment.
  Checkpoint56 shows existing endpoint/path proxies are not valid
  learned-controllable teacher targets. Checkpoints57-59 wire and scale the
  selection-side exact marginal teacher path. Candidate support exists, but
  checkpoint58-59 fail workload/signature gates, and checkpoint59 shows active
  scores anti-align with exact learned-controllable marginal value. Checkpoint60
  shows the 384-ship balanced current-best scale restores workload/profile
  health. Checkpoint61 verifies the strict workload-healthy cell and shows
  selection-side learned-controllable marginal value is still under-ranked.
  Checkpoint62 rejects post-hoc calibration from current row features because
  selection-validated alignment does not transfer with positive retained-set
  marginal ordering.
  Checkpoint64 rejects a selector-shaped proxy loss over current scalar labels:
  it improves segment-head fit but worsens retained-mask causality and exact
  retained-marginal alignment.
  Checkpoint65 localizes the selection-side failure: top exact-marginal learned
  rows are usually under-ranked by both point scores and their selector
  segments.
  Checkpoint 5.122 wires selector segment context into future marginal rows,
  so the next teacher construction checkpoint can consume row-local segment
  ranks/allocation counts directly.
  Checkpoint 5.123 builds the first bounded exact marginal teacher target
  payload, split into segment-level and within-segment point-level targets.
  Checkpoint 5.124 confirms that payload is emitted end to end at Level 1
  smoke scale, but it is not learning evidence.
  Checkpoint 5.125 makes separated marginal teacher usage split-aware, so
  eval exact marginals cannot be mislabeled as train/checkpoint teacher
  candidates.
  Checkpoint 5.126 adds a guarded checkpoint-side consumer that converts full
  separated teacher rows into selector diagnostic score vectors while rejecting
  eval and compact summaries.
  Checkpoint 5.127 rejects immediate training promotion of that direct consumer
  at strict scale: it is guarded, but it loses to the checkpoint-selection
  primary.
  Checkpoint 5.128 localizes the loss to sparse retained-removal-only support
  and ship/point recall loss.
  Checkpoint 5.129 adds a guarded diagnostic-only hybrid consumer that preserves
  dense primary support while injecting exact marginal signal. Checkpoint71 is
  Level 1 smoke evidence only.
  Checkpoint 5.130 rejects that hybrid at strict scale: primary selection QUV1
  is 0.1601869377, direct teacher is 0.1558174990, hybrid w10 is 0.1561941598,
  and hybrid w25 is 0.1563610880.
  Checkpoint 5.131 says the next checkpoint should stop treating profile
  weights and QueryUsefulV1 component weights as constants and instead diagnose
  whether they produce a simple, query-local, trainable signal together.
  Use docs/Next-Iterations.md to guide the next checkpoint sequence.
  Keep docs/keep-in-mind.md in view: the profile/scoring design only needs to
  be sensible for research, not a perfect real-use-case replica, but tiny smokes
  are not adequate evidence for training coherence.

Expected focus:
  target/loss-to-selector alignment
  train/checkpoint-side retained-removal marginal calibration
  exact marginal teacher use that remains workload-blind at eval time
  exact retained-decision marginal alignment by component
  why retained-removal rows are under-ranked despite reasonable overall fit
  excluding skeleton/length-repair guard-owned effects from any teacher signal
  loss/selector alignment that changes learned-controlled retained decisions
    under unchanged causality gates
  exact marginal teacher construction rather than another proxy over current
    scalar labels
  separated segment-level and within-segment point-level retained-removal
    marginal targets
  workload-profile/scoring-component compatibility for a simpler query-local
    trainable signal
  validation beyond tiny smokes before any training-coherence claim

Preferred scope:
  checkpoint75 proves the checkpoint 5.132 workload/scoring compatibility
  payload at the workload-healthy current-best strict cell. It still fails
  predictability and learning causality. The clean strict blocker is
  ship-level evidence: MLQDS beats Douglas-Peucker on query-local
  interpolation, shape, speed/heading, entry/exit, and length, but loses enough
  ship-F1 and related ship/point recall to lose QueryUsefulV1 narrowly. Inspect
  workload_scoring_compatibility_diagnostics first, then per-method
  matched.<method>.range_audit.range_query_metadata_component_summary if grouped
  summaries are insufficient. Do not run the final grid. Do not loosen
  predictability or learning-causality gates. Do not compensate with temporal
  scaffold, raw coverage overrides, or weaker length guardrails.
  After checkpoint 5.135, future artifacts should also inspect
  ship_query_evidence_target_alignment in target diagnostics and
  ship_evidence_counts in the range query metadata summaries. Those fields are
  diagnostic only; they are meant to show whether workload profile families,
  QueryUsefulV1 component weights, and the target/head contract produce a
  trainable ship-level signal together before any semantics change.
  Checkpoint76 ran those fields at the workload-healthy current-best strict
  cell. Query-hit labels carry ship-evidence signal, but behavior is
  weak/negative and the current segment-budget target is anti-aligned with
  ship-query evidence. MLQDS also misses more query-hit ships than both uniform
  and Douglas-Peucker. The next admissible code checkpoint should construct and
  inspect a simple ship-presence-aware segment-budget/target candidate before
  changing training semantics or profile weights.
  Checkpoint 5.137 adds the candidate payload under
  segment_budget_ship_presence_candidate_alignment while leaving active labels
  unchanged. The next meaningful evidence should run it at the
  workload-healthy strict shape and compare target alignment/tradeoffs before
  any loss or head-target change.
  Checkpoint77 shows the pure ship-presence segment budget is too blunt: it
  improves ship-evidence alignment but drops final-score and query-hit top-k
  mass. The blended segment-budget candidates are the only plausible next
  training diagnostic. Test them as guarded variants under unchanged gates
  before any default target change.
  Checkpoint78 rejects the guarded query-hit/ship-presence segment-budget target
  variant. It improves target-side segment-budget ship-evidence Spearman from
  the active `-0.0722770682` to `0.0775842420`, but MLQDS QueryUsefulV1 drops
  to `0.1588862822`, predictability still fails, and learning causality fails
  all material ablation checks. Do not tune that blend further. The next
  checkpoint should diagnose workload-profile and QueryUsefulV1 component
  compatibility.
  Checkpoint79 rejects the guarded final-score/ship-presence segment-budget
  target variant as well. It improves target-side segment-budget ship-evidence
  Spearman to `0.1583725136`, but MLQDS QueryUsefulV1 is only `0.1592468202`
  and learning causality regresses. Both ship-blend target modes should stay
  out of active training options. The next checkpoint should diagnose which
  workload families and QueryUsefulV1 scoring components create non-trainable
  retained-set signal, not add another segment-budget proxy target.
  Checkpoint80 does that from grouped strict artifacts. The active strict
  blocker is concentrated in `small_local`, `density_route`,
  `crossing_turn_change`, and `medium_operational`. The largest persistent
  weighted component losses are `ship_f1`, `ship_balanced_query_point_recall`,
  `ship_coverage`, `query_balanced_point_recall`, and `query_point_mass_ratio`.
  Rejected ship-blend target artifacts widen the same density and small-local
  deficits. The next checkpoint should inspect a profile/scoring recalibration
  candidate at diagnostic level first, not wire another loss or target mode.
  Checkpoint81 runs that diagnostic recalibration probe. A query-local-sensible
  component-weight candidate flips the post-hoc active strict score delta from
  `-0.0008923639` to `0.0029786298`, but the diagnostic marks masking risk as
  high. The improvement comes from downweighting the same ship/point-mass
  blockers and profile-weighting away from density-route/small-local weakness.
  Do not adopt those weights directly. The next checkpoint should preserve or
  improve density-route and small-local ship/point evidence instead of hiding
  it.
  Checkpoint82 adds that blocker-preserving diagnostic. A smoothed
  ship/point-preserving component candidate keeps total ship/point evidence
  weight at `0.55`, and the blocker-preserving profile keeps density-route,
  crossing, small-local, and medium-operational weights within about 1% of the
  active strict profile. The post-hoc score delta is positive, but all critical
  families remain unresolved by ship evidence. Treat this as evidence that
  scoring/profile reweighting alone is not enough; the next checkpoint should
  add family-conditioned target/head trainability instrumentation for
  density-route and small-local before any new loss or scoring default.
  Checkpoint 5.147 adds that instrumentation. QueryUsefulV1 target diagnostics
  now expose `family_conditioned_target_trainability`, and factorized head-fit
  diagnostics expose `family_conditioned_head_trainability`. This is Level 0
  instrumentation only. The next evidence should run the workload-healthy
  strict shape and inspect those fields before proposing any target/head/scoring
  default change.
  Checkpoint83 provides that strict evidence. Scores and gates reproduce the
  current-best strict cell, but the family-conditioned rows isolate the blocker:
  `small_local` is weak on both target construction and fitted heads, while
  `density_route` is mostly target-side weak in behavior and segment-budget.
  The next checkpoint should build a diagnostic-only family-local target/head
  candidate for those families before any loss, selector, scoring, or workload
  profile default changes.
  Checkpoint 5.149 adds the Level 0 candidate surface:
  `family_local_target_candidate_alignment`. It compares family-local query-hit/
  ship, ship-gated behavior, boundary/replacement/ship, composed-score, and
  segment-budget candidates while leaving active training semantics unchanged.
  Treat it as implementation evidence only until a workload-healthy strict
  diagnostic reads the payload for `small_local` and `density_route`.
  Checkpoint84 provides that strict diagnostic. The point-level family query-hit/
  ship candidate strongly ranks ship-query evidence for `small_local` and
  `density_route`, but the segment-budget candidate is still anti-aligned and
  covers only about 5% of ship-query pairs at top-k. Do not promote the current
  candidate. The next checkpoint should diagnose segment aggregation/allocation
  from family-local point signal, not add another proxy loss.
  Checkpoint 5.151 adds diagnostic-only segment aggregation variants and a
  two-stage allocation/point-choice view. Checkpoint85 runs them at strict
  scale. Max-pooled and fractional ship-query segment candidates preserve much
  more family-local ship evidence under two-stage diagnostics, but this is still
  diagnostic evidence. A future guarded training variant must pass unchanged
  strict retained-mask and causality gates before any default change.
  Checkpoint 5.153 adds the guarded non-default
  `query_useful_v1_factorized_segment_budget_query_ship_max_pool` target mode.
  Checkpoint86 runs it at the current-best strict cell. It slightly beats
  Douglas-Peucker on QueryUsefulV1 and makes the no-segment-budget-head
  causality child pass, but predictability and learning causality still fail.
  Do not promote it. The next checkpoint should diagnose checkpoint85 versus
  checkpoint86 target/head/causality transfer, especially why `small_local`
  fitted segment/composed head alignment remains negative after the target-side
  segment signal turns positive.
  Checkpoint 5.155 adds that derived comparison. It narrows the transfer
  blocker: `density_route` has positive target and fitted segment signs, but
  `small_local` and `crossing_turn_change` show positive target-side segment
  signs with negative fitted segment/composed signs. The next checkpoint should
  target family/head transfer for those families under unchanged gates, not add
  another segment aggregation variant or selector blend.
  Checkpoints 5.156-5.157 test a broader guarded
  `query_useful_v1_factorized_query_ship_local_heads` target contract that makes
  the query-hit and behavior heads ship-evidence aware. Checkpoint90 rejects it:
  MLQDS QueryUsefulV1 drops to `0.1632708811`, target diffusion fails because
  the behavior head becomes too broad, and all fitted `small_local` and
  `crossing_turn_change` heads remain negative against family ship-query
  evidence despite positive target-side signs. Do not promote this mode or
  broaden behavior targets further as a transfer workaround.
  Checkpoint 5.158 adds the derived failure diagnosis for checkpoint90. It
  confirms the broad local-head target regresses target diffusion and
  prior-predictive alignment, and the family transfer gap is still large:
  `small_local` q-hit/behavior/segment target-to-fit gaps are `-0.2686`,
  `-0.3864`, and `-0.2000`; `crossing_turn_change` gaps are `-0.5243`,
  `-0.3484`, and `-0.1666`. The next checkpoint should preserve diffusion and
  diagnose model/loss/prior transfer, or recalibrate workload/scoring to make
  the family signal trainable without masking those families away.
  Checkpoint 5.159 runs that diffusion-preserving diagnosis from checkpoint86.
  It finds 11 focused family/head transfer blockers. `crossing_turn_change`
  query-hit, segment-budget, and composed heads fit their labels but still
  misorder ship-query evidence; `small_local` has the same segment-budget
  failure while its query-hit, behavior, and composed targets remain weak. The
  selector retained-decision marginal alignment is also negative at the correct
  layout
  `selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_useful_alignment`
  (`selector_score` Spearman `-0.0421`). The artifact lacks
  family-conditioned prior predictability, so the next code checkpoint should
  add that diagnostic surface before choosing a model/loss or workload/scoring
  calibration change.
  Checkpoint 5.160 adds that family-conditioned prior predictability surface and
  reruns the guarded max-pool strict cell. Scores and gates reproduce
  checkpoint86: MLQDS QueryUsefulV1 `0.1673482145`, uniform `0.1421296610`,
  Douglas-Peucker `0.1671038781`; predictability and learning causality still
  fail. The new family-prior rows show this is not a simple prior-absence
  problem: `crossing_turn_change` query-hit and segment-budget priors are useful
  (`0.3161` and `0.2583` best Spearman), and `small_local` behavior and
  segment-budget priors are useful (`0.1864` and `0.1357` best Spearman), though
  `small_local` query-hit top-k lift and behavior-prior rank for crossing remain
  weak. The derived checkpoint94 decision is
  `diagnose_score_to_selector_marginal_calibration_before_promotion`; retained
  marginal selector-score Spearman is still negative (`-0.0408`) at the correct
  selector-trace layout.
  Checkpoint 5.161 adds that selector-to-retained-marginal calibration
  diagnosis from checkpoint93. It separates two failure modes: 28 high-scored
  low-exact-marginal rows, mostly removed/addition candidates, and 19 top exact
  marginal rows that active selector and segment scores rank in the lower half
  of candidates. The separated eval-only marginal teacher has viable shape, but
  it is not train-side evidence; 4 of its top 10 segment targets are low-ranked
  by selector segment score and allocation weight. The next admissible branch is
  train/selection-side marginal segment calibration evidence, not promotion of
  checkpoint86/93 or another selector blend.
  Checkpoint 5.162 adds that selection-side marginal segment calibration
  diagnosis from checkpoint93. Selection-side exact marginals are available and
  `candidate_for_train_side_teacher=true`, but current scores anti-rank them
  more sharply than eval (`selector_score` Spearman `-0.1610`). Six of the top
  10 selection segment targets are low-ranked by selector segment score and by
  allocation weight. Selection/eval separated teacher segments overlap only
  `4/32`, with zero overlap among the top 10 segment targets. The next
  admissible branch is transfer/calibration diagnosis for the selection-side
  segment teacher, not direct training semantics.
  Checkpoint 5.163 adds the selection-to-eval segment teacher transfer
  diagnosis from checkpoint93. It treats non-teacher segments as zero target
  over all segment candidates. Selection/eval positive teacher segment overlap
  is `4/32`, top 1%, 5%, and 10% target overlap is zero, and sparse target
  Spearman is strongly negative (`-0.7663`) over the positive-target union.
  Simple selector features do have weak consistent positive alignment on both
  splits (`segment_score` about `0.08`, `learned_count` about `0.20`), so the
  next admissible branch is a guarded transfer-calibration probe or deeper
  feature-transfer diagnosis, not raw selection-teacher training.
  Checkpoint 5.164 separates pre-selection transfer features from
  post-selection attribution. It rejects `learned_count` as post-selection
  coupled despite its positive Spearman. Two pre-selection candidates are
  admissible for a guarded probe: active `segment_score` and a simple
  `segment_score`/`segment_allocation_weight` z-blend. The guard-counter
  length-support subtraction candidate is rejected. This only authorizes a
  guarded non-default probe judged by unchanged strict retained-mask quality
  and learning causality; it does not authorize final success or direct teacher
  supervision.
  Checkpoint 5.165 runs that guarded z-blend probe at the current-best strict
  cell as
  `artifacts/results/query_driven_v2_checkpoint99_segment_transfer_calibration_zblend_current_best_strict_local/example_run.json`.
  The calibration is trace-valid and non-default: no post-selection
  attribution, no length-support counter-signal, frozen-primary trace match,
  trace schema `8`, and final effective length-support allocation weight
  `0.0`. It still fails predictability and learning causality. MLQDS
  QueryUsefulV1 is `0.1672369132`, slightly above Douglas-Peucker
  `0.1671038781` but `0.0001113013` below checkpoint93. Shuffled scores,
  shuffled priors, no query-prior features, and no behavior-head causality
  children still fail. Reject the simple allocation-weight z-blend as a
  promotion path; continue with workload/scoring/target compatibility rather
  than more selector z-blend tuning.

Avoid:
  building an endpoint teacher from aggregate endpoint alignment
  treating constant-valued diagnostic fields as ranking signals
  adding production calibration from checkpoint62 row-feature fits
  re-adding the checkpoint64 learned_segment_budget_topk proxy loss
  treating checkpoint65 as a production calibration; it is a derived diagnostic
  treating the checkpoint 5.123 target payload as accepted training semantics
    before smoke and strict evidence
  turning on path support directly without fixing path-head target fit
  reviving the segment-context scalar-score blend without new root evidence
  re-adding sqrt_probability prior transform
  re-adding dense-head rank pressure
  loosening learning-causality gates
  compensating with large temporal scaffold or weaker length guardrails
  promoting the query-hit/ship-presence segment-budget target
  promoting the final-score/ship-presence segment-budget target
```

Do not run the full 4x7 grid until learning causality passes on required smaller evidence. Do not claim final success from this strict cell.

---

## 3. Design contract

The redesign is a contract between five components:

```text
1. versioned future-query workload profile
2. query-local metric
3. factorized train labels and train-only priors
4. trainable workload-blind model
5. selector with material learned control and sanity guardrails
```

All five must be aligned. If one is broken, a downstream model sweep will waste time.

### Correct order of work

1. Stabilize workload generation and signatures.
2. Verify train-derived priors predict held-out query usefulness.
3. Improve target/model only after the predictability audit says useful signal exists.
4. Improve selector only when learned scores are useful but retained masks are poor.
5. Run real-AIS probes only after synthetic/debug probes have clean workload health.
6. Run the full grid only after a strict support-valid single-cell probe passes.

---

## 4. Protocol rules

### Allowed

- Training labels from training workloads.
- Query-aware teachers on training workloads.
- Historical/train-derived priors built before eval compression.
- Validation workloads for checkpoint selection, if validation retained masks are blind and validation scoring does not use final eval queries.
- Eval workloads for scoring only after eval masks are frozen.

### Forbidden for final claims

- Eval queries passed into the model, feature builder, selector, retained-set decision, checkpoint selector, or query-prior builder before compression.
- Eval point/query containment features.
- Eval query boundary-distance features.
- Query cross-attention at eval compression time.
- Checkpoint selection using final eval-query performance.
- Treating query-conditioned `range_aware` as workload-blind success.
- Treating historical-prior KNN lookup as final learned-model success.
- Using eval geometry labels or geometry-label blending for a final workload-blind mask.
- Using a large temporal scaffold that makes learned scores mostly irrelevant.

### Required artifact flags

Every serious run should record:

```yaml
workload_blind_protocol:
  enabled: true
  masks_frozen_before_eval_query_scoring: true
  eval_queries_seen_by_model: false
  eval_queries_seen_by_feature_builder: false
  eval_queries_seen_by_selector: false
  checkpoint_selected_on_eval_queries: false
  query_conditioned_range_aware_used_for_product_acceptance: false
```

Any run that violates these rules is diagnostic only.

---

## 5. Workload profile requirements

The query workload is the product prior. It must be stable enough for the model to learn and broad enough to represent expected future use.

### Active profile

```text
workload_profile_id = range_workload_v1
```

### Recommended anchor-family weights

Use these unless real product query logs justify a change:

```yaml
anchor_family_weights:
  density_route: 0.40
  boundary_entry_exit: 0.20
  crossing_turn_change: 0.15
  port_or_approach_zone: 0.15
  sparse_background_control: 0.10
```

Rationale:

- `density_route` captures recurring traffic corridors.
- `boundary_entry_exit` supports range-query entry/exit evidence.
- `crossing_turn_change` supports behavior explanation.
- `port_or_approach_zone` captures stable AIS-relevant hotspots.
- `sparse_background_control` prevents overfitting to only dense areas, but should not dominate.

Do not increase sparse/background weight merely to make the benchmark broader. That makes uniform temporal sampling close to minimax and undermines the query-driven premise.

### Recommended footprint-family weights

Start with:

```yaml
footprint_family_weights:
  small_local: 0.25
  medium_operational: 0.45
  large_context: 0.20
  route_corridor_like: 0.10
```

Recommended nominal shapes:

```yaml
small_local:
  spatial_radius_km: 1.1
  time_half_window_hours: 2.5

medium_operational:
  spatial_radius_km: 2.2
  time_half_window_hours: 5.0

large_context:
  spatial_radius_km: 4.0
  time_half_window_hours: 8.0

route_corridor_like:
  spatial_radius_km: 2.2
  time_half_window_hours: 5.0
  elongation_allowed: true
```

Current blocker indicates the profile/acceptance settings may still be too hard to sample cleanly on small synthetic splits. Before changing the model, make accepted train/eval workload signatures stable.

### Coverage calibration

Final candidate workloads should use:

```text
coverage_calibration_mode = profile_sampled_query_count
workload_stability_gate_mode = final
```

Do not use `uncovered_anchor_chasing` for final claims unless it is explicitly declared part of the product workload. It can distort the query distribution by chasing uncovered points.

### Query count

Final-profile workloads must not pass with too few queries. A query workload can technically reach coverage with a small number of broad boxes, but that is not enough evidence that the workload profile is stable or learnable.

Hard gate:

```text
minimum accepted queries per workload for final-mode gates: 8
```

Recommended practical sizes:

```text
tiny smoke only:                         4-8 accepted queries
minimum strict diagnostic:               16 accepted queries
standard strict synthetic diagnostic:    32-64 accepted queries
standard real-AIS single-cell probe:     64-128 accepted queries
multi-seed / multi-day confirmation:     64-128 accepted queries per workload
final grid evidence:                     64-256 accepted queries per workload where runtime/data scale permits
```

Use the hard gate only as a lower bound. Do not interpret an 8-query workload as strong scientific evidence. At low query counts, anchor-family proportions, footprint-family proportions, point-hit distributions, ship-hit distributions, and top-k predictability metrics are too noisy.

If strict overshoot plus acceptance filters cannot generate at least 8 healthy accepted queries, the workload profile or footprint settings are wrong for that dataset scale. If it can generate 8 but not at least 16-32 without exhaustion or severe rejection pressure, treat the run as a small diagnostic only.

### Generator health requirements

A final-candidate workload should fail if:

```text
range_acceptance.exhausted == true
stop_reason in {range_acceptance_exhausted, range_coverage_guard_exhausted}
query_count < 8
rejection_rate > 0.85
coverage_guard_rejection_pressure > 2.0
coverage target below/above guard
profile id mismatch
coverage calibration mode not profile_sampled_query_count
```

Recommended diagnostic fields:

```text
accepted
rejected
attempts
rejection_rate
coverage_guard_rejection_pressure
rejection_reasons
rejection_reasons_by_anchor_family
rejection_reasons_by_footprint_family
planned_anchor_family_counts
accepted_anchor_family_counts
planned_footprint_family_counts
accepted_footprint_family_counts
```

### Workload signature gate

Train/eval signatures should compare:

```text
anchor-family distribution
footprint-family distribution
point hits per query
ship hits per query
near-duplicate rate
broad-query rate
query count semantics
profile id
```

Recommended thresholds:

```yaml
anchor_family_l1_distance_max: 0.12
footprint_family_l1_distance_max: 0.12
point_hit_distribution_ks_max: 0.20
ship_hit_distribution_ks_max: 0.20
near_duplicate_rate_max: 0.05
broad_query_rate_max: 0.05
query_count_relative_delta_max: 0.15  # fixed-count/legacy signatures only
min_signature_query_count: 8
```

For fixed-count or legacy signatures, query-count relative delta is a hard
signature check. For `calibrated_to_coverage` signatures using
`profile_sampled_query_count`, accepted query count is a target-coverage stopping
statistic. In that mode, the gate must require matching profile id,
`query_count_mode`, `coverage_calibration_mode`, and `target_coverage`; it must
still enforce `min_signature_query_count` and all distribution checks; and it
must record query-count relative delta as a diagnostic rather than a parity
blocker. Generation health remains a separate hard gate and must pass.

If planned family quotas match but accepted signatures fail, acceptance filters are skewing the generated workload. Fix the generator/profile first.

---

## 6. Metric requirements

### Primary metric

```text
QueryUsefulV1
```

It should emphasize:

```text
query-local point mass
query-local behavior explanation
ship presence and coverage inside query windows
entry/exit and crossing evidence
turn/shape/local interpolation
small global sanity guardrail
```

### Legacy metric

```text
RangeUsefulLegacy
```

Keep it for comparability and diagnostics. Do not use it as the final product metric.

### Current metric caveat

`QueryUsefulV1` now includes a true query-local interpolation component, but it is still partly a bridge over old range-audit components. It is acceptable as the active primary metric for current work, but future improvements should continue making it more query-local and less dependent on global proxies.

### Recommended future metric improvements

Add or strengthen:

```text
query-local interpolation fidelity
query-local speed/heading reconstruction
query-local retained evidence factor
query-local point-mass distribution preservation
per-ship query-local behavior explanation
```

Important rule:

> A trajectory should not get high query-local behavior score solely from outside-query anchors. At least one retained in-query point, or a clearly justified retained bracket/evidence rule, should be required for nonzero local evidence credit.

---

## 7. Target, prior, model, and selector requirements

### Factorized target

Active target:

```text
query_useful_v1_factorized
```

Required heads:

```text
query_hit_probability
conditional_behavior_utility
boundary_event_utility
replacement_representative_value
segment_budget_target
path_length_support_target
```

Recommended interpretation:

```text
final point score ≈ P(future query hits point)
                  × behavior usefulness if queried
                  × replacement/non-redundancy value
                  + boundary/event bonus
```

Keep the heads separate. Do not collapse everything prematurely into one retained-frequency scalar.

### Query-prior fields

Query-prior fields must be built from training workloads only.

Fields should include:

```text
spatial_query_hit_probability
spatiotemporal_query_hit_probability
boundary_entry_exit_likelihood
crossing_likelihood
behavior_utility_prior
route_density_prior
```

They must record:

```yaml
built_from_split: train_only
contains_eval_queries: false
contains_validation_queries: false
profile_id: range_workload_v1
train_workload_seed: ...
extent: ...
out_of_extent_sampling: ...
```

Current recommendation:

```text
out_of_extent_sampling = nearest
```

Use `nearest` for debug/real probes where train/eval route support overlaps but eval points may slightly exceed train bounds. Use `zero` only when explicitly testing support failure.

### Prior predictability

Before tuning the neural model, the train-derived prior must predict held-out eval usefulness.

Required diagnostics:

```text
aggregate Spearman / Kendall
PR-AUC lift over base rate
lift@1%, 2%, 5%, 10%
per-head predictability
prior-channel predictability
top-k eval target mass
sampled prior nonzero fraction
out-of-extent fraction
```

Per-head diagnostics are mandatory because aggregate failure can hide useful sub-signals.

Focus first on:

```text
query_hit_probability
segment_budget_target
```

If `query_hit_probability` fails, the future query-hit prior is not transferring. Fix workload generation, prior fields, or query-profile semantics before model tuning.

If `query_hit_probability` works but `segment_budget_target` fails, fix target/selector alignment.

If `segment_budget_target` works but query-hit fails, the selector may preserve structurally useful points in wrong query regions.

### Model

Active model:

```text
workload_blind_range_v2
```

The model must:

```text
score points query-free at eval compression time
use train-derived prior features only
produce factorized heads
support ablations by disabling heads
generalize across held-out days/seeds under the same workload profile
```

Recommended defaults for debug probes:

```yaml
embed_dim: 32
num_heads: 2
num_layers: 1
epochs: 3-5
loss_objective: budget_topk
mlqds_score_mode: rank_confidence
query_useful_segment_budget_head_weight: 0.10
query_useful_segment_level_loss_weight: 0.25
query_useful_behavior_rank_loss_weight: 0.0
query_useful_sparse_head_rank_loss_weight: 0.0
query_useful_sparse_head_bce_target_mode: raw
```

For real AIS probes, increase capacity only after workload health and predictability gates pass.

The behavior-rank auxiliary is training-only pressure on the
`conditional_behavior_utility` head. The Checkpoint 5.25-5.42 diagnostic group
rejected weight `0.15` as a default because it worsened retained-mask causality
despite slightly better head fit. Keep it disabled unless a future checkpoint
has a specific hypothesis, and do not treat better head fit alone as evidence
of learned workload-blind success.

The sparse-head rank auxiliary is training-only pressure on the numerically
sparse `query_hit_probability` and `boundary_event_utility` heads. It exists to
test the Checkpoint 5.25-5.42 head-saturation diagnosis. Default `0.0`
preserves the current candidate; any nonzero run is diagnostic until a strict
replay proves head dispersion improves retained-mask causality and global
sanity.

The sparse-head BCE target mode is a stronger diagnostic for the same blocker.
Default `raw` preserves current labels. `window_max_normalized` may be used only
to test whether per-window relative query-hit/boundary supervision fixes
base-rate saturation; it must not be treated as accepted target semantics until
a strict replay proves retained-mask causality and global sanity.

### Selector

Active selector:

```text
learned_segment_budget_v1
```

The selector should:

```text
retain a minimal rough skeleton
allocate most budget by learned segment value
select points within segments by learned value + non-redundancy
report learned-controlled retained slots
avoid fixed high temporal scaffold
```

Recommended default:

```yaml
learned_segment_geometry_gain_weight: 0.12
learned_segment_allocation_length_support_weight: 0.12
learned_segment_allocation_weight_floor: 0.50
learned_segment_score_blend_weight: 0.05
learned_segment_fairness_preallocation: true
learned_segment_length_repair_score_protection_fraction: 0.0
```

These are query-free selector guardrails. Geometry-gain tie-breaking and
segment allocation length support are separate controls and must be reported
separately. The allocation weight floor is a score-contrast diagnostic control;
lower values can make learned segment scores more decisive, but any lower-floor
result is diagnostic until a strict replay proves causality and global sanity.
Length-repair score protection is a query-free diagnostic control for testing
whether repair is erasing the highest learned-score decisions; default `0.0`
preserves current behavior. They must be ablated before any final causality
claim.

Causality ablation must include:

```text
without_trajectory_fairness_preallocation
without_segment_budget_head
without_behavior_utility_head
without_query_prior_features
shuffled_prior_fields
shuffled_scores
untrained_model
prior_field_only_score
```

If fairness preallocation or geometry tie-breaker drives the win more than learned heads, the result is not clean learned success.

---

## 8. Learning-causality requirements

A final candidate must prove that learning caused the retained-mask improvement.

Required evidence:

```text
trained model beats shuffled scores
trained model beats untrained model
trained model beats prior-field-only score
trained model beats shuffled prior fields
trained model beats no-query-prior-features
trained model beats no-behavior-head
trained model beats no-segment-budget-head
selector learned-controlled slot fraction is material
retained masks change when relevant learned signals are ablated
```

Default material delta:

```text
min QueryUsefulV1 delta: 0.005
```

For shuffled scores, if MLQDS beats uniform, require:

```text
shuffled score loss >= max(0.005, 0.60 × MLQDS-vs-uniform gap)
```

Recommended learned-slot thresholds:

```text
compression >= 10%: learned-controlled slot fraction >= 0.35
compression >= 5%:  learned-controlled slot fraction >= 0.25
```

Low-compression budgets may have limited learned slots, but the model still must matter where slots exist.

If an untrained model or shuffled score beats the trained model, stop model tuning and diagnose target/predictability/selector.

---

## 9. Required acceptance gates

A strict single-cell probe must pass all of these before running the full grid:

```text
workload_stability_gate_pass = true
support_overlap_gate_pass = true
predictability_gate_pass = true
prior_predictive_alignment_gate_pass = true
target_diffusion_gate_pass = true
workload_signature_gate_pass = true
learning_causality_gate_pass = true
global_sanity_gate_pass = true
MLQDS QueryUsefulV1 > uniform
MLQDS QueryUsefulV1 > DouglasPeucker
```

### Support-overlap gate schema

Required fields:

```json
{
  "gate_pass": false,
  "eval_points_outside_train_prior_extent_fraction": 0.0,
  "sampled_prior_nonzero_fraction": 0.0,
  "primary_sampled_prior_nonzero_fraction": 0.0,
  "route_density_overlap": 0.0,
  "query_prior_support_overlap": 0.0,
  "train_eval_spatial_extent_intersection_fraction": 0.0,
  "failed_checks": []
}
```

Recommended thresholds:

```text
eval_points_outside_train_prior_extent_fraction <= 0.10
sampled_prior_nonzero_fraction >= 0.50
primary_sampled_prior_nonzero_fraction >= 0.30
route_density_overlap >= 0.25
query_prior_support_overlap >= 0.25
```

Support overlap is necessary but not sufficient. It proves that priors are sampled nontrivially, not that they predict usefulness.

### Global sanity gate

Current hard checks:

```text
avg_length_preserved between 0.75 and 1.20
endpoint_sanity = 1.0 for eligible trajectories
avg_sed_ratio_vs_uniform <= threshold
```

SED ratio threshold:

```text
compression <= 1%:  2.00
compression <= 2%:  1.75
otherwise:          1.50
```

If this gate repeatedly fails, do not hide the problem with a large temporal scaffold. Add query-free learned/sanity-aware selector constraints or revise the metric/threshold only with justification.

---


## 10. Probe scale policy and evidence levels

Small probes are useful for code correctness and fast failure localization. They are dangerous for scientific conclusions.

A tiny run can pass or fail for reasons that disappear at realistic scale:

```text
one query family dominates by chance
one rejected query shifts family L1 distances
KS distances are meaningless with too few queries
lift@1% selects only a handful of points
1% and 2% compression mostly test endpoints and rounding
causality ablations have too few learned-controlled slots
global sanity swings because of one trajectory
```

Use the levels below. Do not make conclusions that exceed the evidence level.

### Level 0 — static verification and unit tests

Purpose:

```text
verify code integration
verify no-leakage invariants
verify report fields and gates exist
verify masks freeze before eval scoring
```

Recommended scope:

```text
no model-quality experiment required
unit tests and focused smoke tests only
```

Allowed conclusions:

```text
code path works or is broken
metadata/gates are present or missing
protocol invariants are enforced or violated
```

Forbidden conclusions:

```text
model learns
workload profile is stable
prior predictability is useful
selector is effective
final candidate is promising
```

### Level 1 — tiny smoke experiment

Purpose:

```text
confirm the end-to-end command runs
confirm artifacts are emitted
confirm no obvious tensor/shape/config bug
```

Recommended scale:

```text
ships:               4-12
points/ship:          32-96
synthetic families:   1-2
accepted queries:     4-8
train replicates:     1-2
epochs:               1-2
compression:          one ratio, usually 5% or 20%
workload profile:     one profile, usually range_workload_v1 for implementation smoke
```

Allowed conclusions:

```text
implementation path runs
artifact schema is valid
gates are emitted
obvious bugs exist or do not exist
```

Forbidden conclusions:

```text
learning is real
predictability is good
generator is healthy
workload signatures are stable
uniform/DP wins are meaningful
```

### Level 2 — minimum strict diagnostic single-cell

Purpose:

```text
localize the current blocker under final-mode gates
separate generator/profile failure from prior/target/model/selector failure
```

Minimum recommended scale:

```text
ships:               24-32
points/ship:          128-192
synthetic families:   2-3
accepted queries:     16-32
train replicates:     4
epochs:               3-5
compression:          5%
workload profile:     one final profile, usually range_workload_v1
acceptance attempts:  20,000+
```

Allowed conclusions:

```text
which gate is currently blocking
whether generator health is obviously bad
whether support overlap exists
whether prior channels have nonzero sample support
whether target diffusion is obviously bad
```

Still forbidden:

```text
final model-quality claim
full predictability claim
low-budget robustness claim
workload-profile grid claim
```

If this level fails because of generator exhaustion, too few accepted queries, signature drift, or tiny learned-slot counts, do not tune the model. Increase scale or fix the workload profile first.

### Level 3 — standard strict diagnostic single-cell

Purpose:

```text
make a serious one-cell claim about signal, learning causality, and selector behavior
```

Recommended scale:

```text
ships:               48-96
points/ship:          192-384
synthetic families:   3-6
accepted queries:     32-64
train replicates:     4-8
epochs:               5-10
compression:          5%
workload profile:     one final profile, usually range_workload_v1_local or range_workload_v1
acceptance attempts:  30,000-60,000
```

Required evidence:

```text
workload_stability_gate_pass = true
workload_signature_gate_pass = true
support_overlap_gate_pass = true
target_diffusion_gate_pass = true
predictability_gate_pass = true
prior_predictive_alignment_gate_pass = true
learning_causality_gate_pass = true
global_sanity_gate_pass = true
MLQDS QueryUsefulV1 > uniform
MLQDS QueryUsefulV1 > DouglasPeucker
```

Allowed conclusion:

```text
candidate is worth testing on real AIS
candidate is not worth testing on real AIS
```

### Level 4 — real AIS strict single-cell

Purpose:

```text
show the same strict single-cell behavior transfers to real held-out AIS data
```

Recommended scale depends on data availability, but do not use tiny caps for evidence:

```text
train days/sources:       at least 2-4 historical days when available
validation day/source:    separate from train and eval
eval day/source:          separate held-out day/source
trajectories:             at least 48-128 after segmentation
points/trajectory cap:    192-512 where runtime permits
accepted queries:         64-128 per workload
train replicates:         4-8
epochs:                   5-10
compression:              5%
workload profile:         one final profile, usually range_workload_v1_local or range_workload_v1
```

Required evidence:

```text
same gates as Level 3
route/support overlap is real but not identical leakage
per-head predictability is stable
causality ablations remain meaningful
global sanity remains within thresholds
```

Allowed conclusion:

```text
candidate is worth multi-seed/multi-day confirmation
candidate needs target/prior/model/selector work
```

### Level 5 — multi-seed / multi-day confirmation

Purpose:

```text
verify the single-cell result is not a seed/day accident
```

Recommended scale:

```text
seeds:                    3-5
real train/eval splits:    2-4 when data is available
workload profiles:         at least range_workload_v1_local and range_workload_v1
compression ratios:        at least 2%, 5%, and 10%
accepted queries:          64-128 per workload
train replicates:          4-8
```

Required evidence:

```text
mean and worst-case QueryUsefulV1 vs uniform/DP
gate pass rate
per-head predictability stability
learning-causality stability
global sanity stability
runtime/latency stability
```

Allowed conclusion:

```text
ready for full 4x7 grid
not ready for full 4x7 grid
```

### Level 6 — final 4x7 grid

Purpose:

```text
final acceptance
```

Required grid:

```text
workload profiles:    range_workload_v1_focused, range_workload_v1_local,
                      range_workload_v1_operational, range_workload_v1
compression ratios:   1%, 2%, 5%, 10%, 15%, 20%, 30%
cells:                28
```

Required evidence:

```text
no missing cells
all child gates pass
numeric success bars pass
all component/runtime/latency fields reported
real AIS held-out-day evidence included
```

Benchmark rows must include inference-only MLQDS latency as
`mlqds_inference_only_latency_ms` and `mlqds_inference_only_latency_seconds`.
This is retained-mask application time only. It must not include query scoring,
range diagnostics, report construction, or matched-evaluation phase time.

### Scale-specific interpretation rules

Use these rules when deciding what to do next:

```text
If Level 1 passes, run Level 2. Do not claim learning.
If Level 2 fails on generator/signature, fix workload generation or increase scale.
If Level 2 passes but predictability fails, fix prior fields / workload profile / targets.
If Level 3 fails causality, diagnose target/loss/model/selector.
If Level 3 passes, run Level 4 on real AIS.
If Level 4 passes, run Level 5.
If Level 5 passes, run Level 6.
```

Never run the full grid to “see what happens” when the standard strict single-cell evidence is already failing. The grid is expensive and mostly useful after the one-cell gates show the candidate is coherent.

### Exploratory real-scale diagnostics

It is acceptable to occasionally run a real-scale diagnostic slice for the most promising current candidate before every strict gate passes, but only to answer a specific scaling or instrumentation question. This is not the full final grid and it is not acceptance evidence.

The default pre-gate form is a representative slice. A full 4x7 snapshot before required gates pass is exceptional: it needs a concrete scaling question, unchanged strict gates, production-like caps, and an explicit label that it is observational diagnostics only.

An occasional benchmark snapshot can be useful to see how the current best candidate/config behaves at realistic scale, especially when tiny or single-cell probes may be hiding runtime, workload-count, or scale-sensitive quality failures. It must be treated as a checkpoint diagnostic, not as proof of progress.

Treat this as a scarce calibration tool: at most one snapshot per materially different current-best candidate unless the previous snapshot exposed an instrumentation or runtime defect that needs a recheck. The result may inform prioritization, capacity planning, and whether the current direction is worth more focused diagnosis. It must not be fed into threshold changes, checkpoint selection, selector tuning, or final comparison tables without a separate strict single-cell diagnosis.

Do not promote a snapshot result into the current-best evidence boundary. The evidence boundary moves only when a strict single-cell probe passes or when a focused diagnostic gives a narrower blocker with unchanged gates.

Use this only when the question is concrete:

```text
does the candidate collapse at realistic trajectory/query scale?
do artifacts, gate fields, runtime fields, and child reports survive production-like caps?
does a tiny-probe pattern disappear when query/workload counts become meaningful?
is runtime or memory already infeasible for the current direction?
```

Recommended scope:

```text
one representative cell, or a small slice such as 1x7 or 2x2
strict gates unchanged
fixed candidate/config except for the diagnostic variable being tested
production-like caps/workload shape where runtime permits
clearly marked exploratory output directory and report labels
```

Before running a pre-gate benchmark snapshot, record the exact question it is
meant to answer and why the next smaller evidence level cannot answer it. After
the run, log the failed child gates first; do not summarize the snapshot as a
candidate-quality result.

Allowed conclusions:

```text
the candidate has or lacks obvious scaling/instrumentation viability
the benchmark harness needs repair before larger runs
the next checkpoint should continue, narrow, or pivot
```

Forbidden conclusions:

```text
final success
final model quality
final grid robustness
gate success by visual inspection or encouraging partial numbers
```

Do not run these slices or snapshots repeatedly to search for a lucky result, compensate for failed causality/support/workload gates, or justify loosening thresholds. If the result is interesting but child gates still fail, diagnose the failed gates before changing code.

### Precision and runtime diagnostics

It is worth testing precision/runtime configurations occasionally, but only after
the candidate is coherent enough that runtime and numerical stability matter.
Treat TF32, AMP FP16, and AMP BF16 comparisons as engineering diagnostics, not
as model-quality experiments.

Use precision sweeps to answer concrete questions:

```text
does TF32/BF16/FP16 materially reduce runtime or memory?
does a precision mode flip any strict gate or child gate?
are retained masks and QueryUsefulV1 stable under the same seed/config/data?
does the artifact report enough torch-runtime metadata to reproduce the result?
```

Required protocol:

```text
change only precision/runtime knobs
use the same candidate, seeds, data split, query scale, and caps
record float32_matmul_precision, allow_tf32, amp_mode, and child torch_runtime
compare against an FP32/highest-precision control artifact
reject any mode that flips a gate or causes material metric drift
```

Do not use precision changes to tune selector behavior, compensate for weak
learning, or claim success from a candidate that fails the standard evidence
ladder. If precision affects quality, treat that as a numerical-stability bug or
hardware-specific diagnostic until reproduced under the normal gates.

---

## 11. Full final grid requirements

Run the full final grid only after a strict single-cell probe passes.

The exploratory pre-gate snapshot exception in Section 10 is not a final-grid
run, even if it uses the same 4x7 shape. Keep that output under exploratory
labels, report failed child gates first, and exclude it from acceptance claims,
current-best evidence boundaries, and final comparison tables.

Workload profiles:

```text
range_workload_v1_focused
range_workload_v1_local
range_workload_v1_operational
range_workload_v1
```

Compression ratios:

```text
0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30
```

Required cells:

```text
4 workload profiles × 7 compression ratios = 28 cells
```

### Numeric success bars

The benchmark-level final grid should pass:

```text
MLQDS beats uniform on QueryUsefulV1 in at least 19 / 28 cells
MLQDS beats Douglas-Peucker on QueryUsefulV1 in at least 24 / 28 cells
MLQDS beats uniform in at least 7 / 12 low-budget cells
MLQDS beats uniform in at least 3 / 4 matched 5% compression cells
```

Low-budget cells are:

```text
compression ratios 0.01, 0.02, 0.05 across all 4 workload profiles
```

These thresholds are a practical minimum for claiming “most grid cells” without requiring impossible perfection. If the project later demands a stricter standard, raise these thresholds, do not lower them to fit weak results.

### Required report fields

Every final-grid run should report:

```text
QueryUsefulV1
RangeUsefulLegacy
RangePointF1
ShipF1
ShipCov
EntryExitF1
CrossingF1
TemporalCov
GapCov
GapCovTime
GapCovDistance
TurnCov
ShapeScore
query-local interpolation fidelity
global SED/PED distortion
length preservation
runtime
latency
workload stability gate
workload signature gate
support overlap gate
predictability gate
prior-predictive alignment gate
target diffusion gate
learning causality gate
global sanity gate
selector learned-slot attribution
per-head predictability
prior-channel predictability
generation rejection diagnostics
```

Do not report only aggregate wins. Component regressions are often where false success hides.

---

## 12. Ambiguity-resolution recommendations

When a value or strategy is ambiguous, use these defaults unless evidence says otherwise.

### Workload profile

```text
Use range_workload_v1.
Use profile_sampled_query_count.
Use final gate mode for any acceptance evidence.
Use 4-8 train workload replicates.
Use workload-profile defaults unless a diagnostic explicitly overrides them:
  range_workload_v1_focused     -> target_coverage 0.05, overshoot 0.005
  range_workload_v1_local       -> target_coverage 0.10, overshoot 0.0075
  range_workload_v1_operational -> target_coverage 0.15, overshoot 0.010
  range_workload_v1             -> target_coverage 0.30, overshoot 0.020
```

Recommended query scale by evidence level:

```text
tiny smoke:                       4-8 accepted queries
minimum strict diagnostic:         16-32 accepted queries
standard strict diagnostic:        32-64 accepted queries
real AIS single-cell evidence:     64-128 accepted queries
multi-seed confirmation:           64-128 accepted queries
final grid:                        64-256 accepted queries where data_preparation/runtime permits
```

A run with only 8 accepted queries can verify gates exist. It should not be used to claim workload stability or model learning.

### Data

For synthetic debug:

```text
Use shared-route support: synthetic_route_families >= 2 for minimum strict diagnostics.
Use synthetic_route_families = 3-6 for standard strict diagnostics.
Use enough scale that the profile can generate healthy accepted queries.
Minimum strict diagnostic: n_ships >= 24-32 and n_points >= 128-192.
Standard strict diagnostic: n_ships >= 48-96 and n_points >= 192-384.
```

For real evidence:

```text
Use real AIS held-out days.
Use separate train, validation, and eval CSVs.
Prefer at least 2-4 train days/sources when available.
Use at least 48-128 trajectories after segmentation for evidence runs.
Use point caps of 192-512 where runtime permits.
Require route/support overlap but not identical query sets.
```

Synthetic shared-route success is not final evidence. It is a debugging step.

### Model and training

Start small until gates pass:

```text
embed_dim = 32
num_heads = 2
num_layers = 1
epochs = 3-5
train_batch_size = 8
inference_batch_size = 8
loss_objective = budget_topk
checkpoint_selection_metric = uniform_gap
checkpoint_score_variant = query_useful_v1
```

Increase model capacity only after:

```text
workload health passes
signature stability passes
prior predictability has useful signal
trained model is not worse than controls
```

### Selector

Keep:

```text
mlqds_temporal_fraction = 0.0
selector_type = learned_segment_budget_v1
```

Do not reintroduce high temporal scaffolding to improve scores. Use query-free sanity constraints and ablations instead.

### Metric

Use `QueryUsefulV1` as primary. Use `RangeUsefulLegacy` only as diagnostic.

If QueryUsefulV1 rewards behavior that conflicts with the actual product goal, improve the metric explicitly and bump/report the schema version. Do not silently change interpretation.

---

## 13. Risk register

### Risk: drawing scientific conclusions from undersized probes

Small probes can prove code correctness, but they cannot prove workload stability, predictability, learning causality, or final success. This project is especially sensitive because query-family distributions, top-k predictability, learned-slot attribution, and global geometry all become unstable at low query/trajectory counts.

Mitigation:

```text
Use Level 0-1 only for implementation verification.
Use Level 2 only for blocker localization.
Use Level 3+ before claiming useful signal.
Use Level 4+ before trusting real-data behavior.
Use Level 6 only after earlier levels pass.
```

If a result changes dramatically when moving from tiny smoke to standard diagnostic scale, trust the larger run and treat the smaller one as biased.

### Risk 1 — Workload profile is not learnable

Symptom:

```text
workload signatures pass
support overlap passes
but predictability/prior alignment fails
```

Meaning:

The workload family may not have stable cross-day spatial/temporal signal, or the train-derived prior fields are too weak.

Fix direction:

```text
inspect per-head predictability
improve query-prior raster/field construction
use more historical train days
improve route/hotspot representation
narrow or clarify workload profile
```

### Risk 2 — Accepted workloads are rejection-biased

Symptom:

```text
planned family quotas match
accepted anchor/footprint distributions differ
range_acceptance exhausted
too_broad/coverage_overshoot dominates
workload signature fails
```

Fix direction:

```text
adjust footprint sizes
reduce overly broad families
improve acceptance filters
calibrate query count and footprint scale separately
report rejection skew by family
```

### Risk 3 — Model fits target but target is wrong

Symptom:

```text
train target fit good
predictability maybe acceptable
eval QueryUsefulV1 poor
ablations show learned heads not useful
```

Fix direction:

```text
compare per-head target fit against eval usefulness
improve replacement/segment target
add true counterfactual marginal gain samples
avoid diffuse labels
```

### Risk 4 — Selector dominates learning

Symptom:

```text
fairness/geometry ablation drives gains
trained-vs-shuffled delta small
learned-slot fraction low
untrained model competitive
```

Fix direction:

```text
increase learned decision authority carefully
improve model score reliability first
report and ablate every query-free selector heuristic
```

### Risk 5 — Global sanity blocks useful query behavior

Symptom:

```text
MLQDS improves QueryUsefulV1 but fails length/SED
```

Fix direction:

```text
add query-free sanity head or selector constraint
preserve endpoints and rough path-length without high temporal scaffold
penalize bad validation sanity more strongly
```

### Risk 6 — Uniform is close to optimal

Symptom:

```text
query workload broad/random
query-hit predictability low
uniform beats learned model consistently
```

Fix direction:

```text
revisit product workload assumptions
do not force ML success under arbitrary broad workloads
narrow workload profile only if it matches real use
```

---

## 14. Failure diagnosis tree

Use this order after every strict probe.

### Step 1 — Workload generation

If any fail:

```text
workload_stability_gate
workload_signature_gate
generator health fields
```

then do not tune model. Fix generator/profile/acceptance.

Primary questions:

```text
Are at least 8 queries accepted per workload?
Did generation exhaust?
Are rejection rates too high?
Are accepted anchor/footprint family distributions close to planned?
Are point/ship-hit distributions close across train/eval?
```

### Step 2 — Support and predictability

If support overlap fails:

```text
fix train/eval route support or prior-field extent/sampling
```

If support passes but predictability fails:

```text
inspect per-head predictability
query_hit_probability must improve first
segment_budget signal alone is not enough
```

### Step 3 — Causality

If trained model loses to untrained/shuffled/prior-only controls:

```text
stop architecture sweeps
inspect target alignment and loss
verify scores materially change retained masks
```

### Step 4 — Global sanity

If QueryUsefulV1 improves but global sanity fails:

```text
add sanity-aware selector/model constraints
do not add large temporal scaffold
```

### Step 5 — Baseline comparison

If gates pass but MLQDS loses uniform/DP:

```text
diagnose component deltas
check whether metric rewards intended behavior
check whether selector is allocating budget to wrong segments
```

---

## 15. Forward roadmap: start checkpoints from here

Use concise checkpoints. Numbering restarts here.

### Checkpoint 1 — Workload generator health and signature stability

Goal:

```text
Produce a strict support-valid synthetic/debug single-cell workload where generation is healthy and train/eval signatures pass at a scale large enough to evaluate workload stability.
```

Use the standard strict diagnostic scale unless runtime makes it impossible. A smaller 24-32 ship / 16-query run is allowed only as a quick preliminary failure-localization step.

Recommended command shape. This example uses the local workload-profile variant
as a representative strict diagnostic. Do not express this as a raw coverage
override; coverage and overshoot are profile-owned settings.

```bash
uv run --group dev -- python -m orchestration.train_and_score \
  --results_dir Range_QDS/artifacts/results/query_driven_v2_checkpoint01_generator_health_probe_standard_profile_local_r05 \
  --n_ships 64 \
  --n_points 256 \
  --synthetic_route_families 4 \
  --seed 2324 \
  --n_queries 48 \
  --max_queries 256 \
  --range_train_workload_replicates 4 \
  --workload_profile_id range_workload_v1_local \
  --coverage_calibration_mode profile_sampled_query_count \
  --workload_stability_gate_mode final \
  --model_type workload_blind_range_v2 \
  --range_training_target_mode query_useful_v1_factorized \
  --selector_type learned_segment_budget_v1 \
  --checkpoint_score_variant query_useful_v1 \
  --checkpoint_selection_metric uniform_gap \
  --validation_score_every 1 \
  --checkpoint_full_score_every 1 \
  --checkpoint_candidate_pool_size 1 \
  --epochs 3 \
  --embed_dim 32 \
  --num_heads 2 \
  --num_layers 1 \
  --train_batch_size 8 \
  --inference_batch_size 8 \
  --compression_ratio 0.05 \
  --mlqds_temporal_fraction 0.0 \
  --mlqds_hybrid_mode fill \
  --mlqds_score_mode rank_confidence \
  --range_acceptance_max_attempts 40000 \
  --final_metrics_mode diagnostic
```

Pass condition for this checkpoint:

```text
workload_stability_gate_pass = true
workload_signature_gate_pass = true
support_overlap_gate_pass = true
target_diffusion_gate_pass = true
generation not exhausted
accepted query count >= 32 preferred, >= 16 minimum for a diagnostic
rejection rate acceptable
coverage guard rejection pressure acceptable
```

If it fails, change workload/profile/generator settings or increase dataset scale. Do not tune the model.

Possible fixes:

```text
increase dataset scale
increase accepted query target only when generation is healthy
reduce large_context weight
reduce footprint radii
reduce footprint jitter
make route_corridor_like less broad
increase max attempts only if rejection rate is not structurally high
separate debug profile from final profile if needed
```

### Checkpoint 2 — Prior predictability and target alignment

Goal:

```text
Make train-derived priors predict held-out query usefulness under a healthy workload.
```

Pass condition:

```text
predictability_gate_pass = true
prior_predictive_alignment_gate_pass = true
query_hit_probability has positive Spearman and useful lift@5
segment_budget_target has useful lift@5
```

If query-hit fails:

```text
fix prior fields/workload profile
do not tune model
```

If query-hit passes but behavior/replacement/segment fails:

```text
fix factorized targets and segment labels
```

### Checkpoint 3 — Learned model causality

Goal:

```text
Make the trained model beat ablations under one strict healthy single-cell probe.
```

Pass condition:

```text
learning_causality_gate_pass = true
MLQDS > uniform on QueryUsefulV1
MLQDS > DouglasPeucker on QueryUsefulV1
trained model beats untrained/shuffled/prior-only/no-head ablations
```

If not, diagnose target/loss/model capacity only after Checkpoints 1-2 pass.

### Checkpoint 4 — Global sanity correction

Goal:

```text
Pass length/endpoint/SED global sanity without a high temporal scaffold.
```

Possible fixes:

```text
add length/skeleton auxiliary head
add selector constraint for path-length support
increase geometry-gain tie-breaker carefully
add validation sanity hard constraint
```

Do not use `mlqds_temporal_fraction=0.85` or similar scaffolded masking.

### Checkpoint 5 — Real AIS strict single-cell

Goal:

```text
Repeat strict single-cell evidence on real held-out AIS days.
```

Use separate:

```text
train_csv_path
validation_csv_path
eval_csv_path
```

Pass condition:

```text
all single-cell gates pass
MLQDS beats uniform and DP on QueryUsefulV1
learning causality passes
global sanity passes
```

### Checkpoint 6 — Full final grid

Goal:

```text
Run the 4 workload-profile × 7 compression grid only after a strict real-AIS
single-cell passes.
```

Pass condition:

```text
all child gates pass
numeric success bars pass
component/reporting checklist complete
```

---

## 16. What not to do

Do not compensate for failures with:

```text
query-conditioned inference
eval-query feature leakage
checkpoint selection on final eval queries
range_aware as final result
historical_prior KNN as final learned success
large temporal scaffold
geometry-label blending disguised as learning
loose coverage overshoot
tiny query counts in final gates
artificially easy workload profiles not matching product intent
```

Do not continue sweeping:

```text
KNN neighbor count
source-day agreement aggregation
local-swap temporal fraction
min learned swaps around current KNN score
retained-frequency budget weighting
pointwise MLP imitation of KNN
old RangeUseful scalar target blends
```

Those were useful historically but are no longer the main path.

---

## 17. Progress-log format

Use concise checkpoints. Each checkpoint should record:

```md
## Checkpoint N — <short name>

Status: completed / partial / failed

Goal:
- ...

Changes:
- ...

Tests:
- ...

Experiment artifact:
- path: ...
- command: ...

Key results:
- MLQDS QueryUsefulV1: ...
- uniform QueryUsefulV1: ...
- Douglas-Peucker QueryUsefulV1: ...
- gates passed: ...
- gates failed: ...

Decision:
- continue / pivot / stop and diagnose
```

Keep the progress log short. Detailed stdout and raw metrics belong in artifacts.

---

## 18. Completion definition

The redesign is complete only when:

```text
1. the full workload-profile/compression grid is present
2. QueryUsefulV1 final-grid numeric success bars pass
3. all child gates pass
4. workload-blind protocol flags prove no eval query leakage
5. learning causality proves trained model behavior matters
6. RangeUsefulLegacy and all component metrics are reported
7. real AIS held-out-day evidence passes
8. retained trajectories are globally sane
9. failures and limitations are documented honestly
```

Anything less is either a useful diagnostic, a partial result, or a negative result.
