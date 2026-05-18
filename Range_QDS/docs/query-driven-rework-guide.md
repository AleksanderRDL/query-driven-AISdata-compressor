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

The latest relevant strict single-cell evidence is:

```text
artifacts/results/query_driven_v2_checkpoint42_mode_aware_current_best_strict_local/example_run.json
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
- Prior and head ablations move final masks too little, and removing behavior or
  segment-budget heads does not hurt enough. Do not compensate by weakening
  length repair, adding large temporal scaffolding, or loosening causality gates.

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

Current next checkpoint direction:

```text
Primary hypothesis:
  Checkpoint43 classifies the remaining failure as score-composition and
  selector-marginal alignment, with prior-to-head transfer as a contributing
  blocker. The next useful move is to explain why high-marginal retained,
  skeleton, length-repair, and removed candidates are under-ranked by the raw,
  selector, and segment scores.

Expected focus:
  exact marginal rows by source and decision
  final selector score composition for high-marginal under-ranked points
  head output contribution to raw score versus segment allocation
  whether behavior and segment heads are acting as broad allocation priors
  instead of point-level marginal value signals

Preferred scope:
  focused artifact diagnostics first. If code changes are needed, prefer
  instrumentation or root target/prior/selector alignment fixes. Do not run the
  final grid. Do not loosen predictability or learning-causality gates. Do not
  compensate with temporal scaffold, raw coverage overrides, or weaker length
  guardrails.

Avoid:
  re-adding sqrt_probability prior transform
  re-adding dense-head rank pressure
  loosening learning-causality gates
  compensating with large temporal scaffold or weaker length guardrails
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
