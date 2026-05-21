# Range_QDS Query-Driven Implementation and Research Guide

This is the operating guide for continuing the `Range_QDS` query-driven AIS
trajectory compression work. It is written for a new engineer or implementation
agent starting from the current repository state.

Use this document for protocol, defaults, gates, and admissible next work. Use
`docs/query-driven-implementation-progress.md` for the short evidence boundary
and checkpoint log. Use `Range_QDS/artifacts/results/` for raw run outputs.

---

## 0. Current state in one page

Project status: **active, not accepted**.

The implementation now targets a query-driven, workload-blind AIS compressor. It
is no longer treated as a “rework” of the legacy code path. Historical metric,
profile, and model names are diagnostic only unless explicitly reintroduced by a
new checkpoint with evidence.

### Active default stack

| Surface | Default |
| --- | --- |
| Primary metric | `QueryLocalUtility` |
| Score groups | `query_point_mass=0.50`, `query_local_behavior=0.45`, `global_sanity=0.05` |
| Workload profile | `range_query_mix` |
| Training target | `query_local_utility_factorized` |
| Model | `workload_blind_range` |
| Selector | `learned_segment_budget` |
| Checkpoint score | `query_local_utility` |
| Checkpoint selection | `uniform_gap` |

### Active `QueryLocalUtility` components

| Component | Weight | Source |
| --- | ---: | --- |
| `query_point_recall` | `0.50` | direct query-local point recall |
| `query_local_interpolation_fidelity` | `0.20` | direct query-local interpolation fidelity |
| `query_local_turn_change_coverage` | `0.15` | direct query-local turn/change coverage |
| `query_local_continuity` | `0.10` | direct query-local min gap coverage |
| `endpoint_or_skeleton_sanity` | `0.02` | light global/skeleton sanity |
| `global_shape_guardrail_score` | `0.02` | light SED-derived guardrail |
| `length_preservation_guardrail` | `0.01` | light length guardrail |

`QueryLocalUtility` must not source point mass from legacy `range_point_f1`, and
must not fill missing behavior from older ship, boundary, shape, temporal, gap,
or replacement fallback components. If a required direct component is missing,
it is zero. This is intentional.

### Active `range_query_mix` profile

| Family | Weight / parameters |
| --- | --- |
| Anchor `density` | `0.80` |
| Anchor `sparse_background_control` | `0.20` |
| Footprint `medium_operational` | weight `0.6923076923076923`, `2.2 km`, `5.0 h`, no elongation, point-hit fraction band `[0.006, 0.030]` |
| Footprint `large_context` | weight `0.3076923076923077`, `4.0 km`, `8.0 h`, no elongation, point-hit fraction band `[0.010, 0.045]` |

Profile variants for sweeps remain `range_query_mix_focused`,
`range_query_mix_local`, `range_query_mix_operational`, and `range_query_mix`.
The current evidence boundary uses the two-footprint `range_query_mix` path.

### Current evidence boundary

Latest current-code strict reference artifact:

```text
artifacts/results/additive_qhit_behavior_score_composition_level2_seed2539/example_run.json
artifacts/results/additive_qhit_behavior_score_composition_level2_seed2539/semantic_diagnostic.json
```

Latest blocker-localization artifact:

```text
artifacts/results/additive_level2_child_gate_root_localization/diagnostic.json
```

Latest rejected wiring artifact:

```text
artifacts/results/pooled_point_score_segment_allocation_level1_smoke/example_run.json
artifacts/results/pooled_point_score_segment_allocation_level1_smoke/semantic_diagnostic.json
artifacts/results/pooled_point_score_segment_allocation_level1_smoke/rejection_diagnostic.json
```

Latest failure-diagnosis artifact:

```text
artifacts/results/pooled_point_score_allocation_failure_diagnosis/diagnostic.json
```

Latest mask-delta diagnostic artifact:

```text
artifacts/results/segment_allocation_mask_delta_diagnostic/diagnostic.json
```

Latest segment-head compression diagnostic artifact:

```text
artifacts/results/segment_budget_head_compression_root_diagnostic/diagnostic.json
```

Latest rejected rank-loss wiring artifact:

```text
artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring/example_run.json
artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring/semantic_diagnostic.json
artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring/rejection_diagnostic.json
```

Configuration summary:

```text
level: strict Level 2 source-stratified synthetic replay
seed: 2539
ships: 32
points per ship: 192
requested queries: 24
epochs: 4
train workload replicates: 4
train-side marginal diagnostics: enabled
query_hit_target_variant: raw_query_hit_ship_evidence_multiplier
score formula: additive_raw_query_hit_and_behavior_with_conditional_replacement_modulation_plus_boundary
```

Key scores:

```text
MLQDS QueryLocalUtility:           0.0995482993
uniform QueryLocalUtility:         0.0992909061
Douglas-Peucker QueryLocalUtility: 0.1182249577
MLQDS - uniform:                   +0.0002573932
MLQDS - Douglas-Peucker:           -0.0186766584
```

Current gate state:

```text
passed:
  workload stability
  support overlap
  target diffusion  # final support gt_0.01 = 0.2351190476
  predictability
  prior-predictive alignment

failed:
  workload signature  # known point-hit-fraction KS recurrence at Level 2
  learning causality
  global sanity       # length preservation 0.703978 < 0.75
  final grid          # not admissible
```

Learning-causality child state:

```text
passed at threshold but not sufficient for final success:
  shuffled_scores_should_lose:       0.00638639
  untrained_model_should_lose:       0.0110130
  without_behavior_head_should_lose: 0.00622185

failed:
  shuffled_prior_fields_should_lose:        0.0
  without_query_prior_features_should_lose: 0.0
  without_segment_budget_head_should_lose: -0.000488398
```

Blocker localization from the derived diagnostic:

```text
prior:   priors are predictive and reach the model, but retained masks do not move
behavior: behavior ablation is material, but conditional_behavior_utility remains weak
segment: segment-budget head is compressed/non-causal; pooled point-score allocation looked better counterfactually
length:  length floor fails; pure length allocation hurts QueryLocalUtility
```

Rejected Phase 49 wiring checkpoint:

```text
same-seed additive Level 1 MLQDS QueryLocalUtility:    0.1064832750
same-seed pooled point-score MLQDS QueryLocalUtility: 0.0856186098
delta:                                                 -0.0208646652
same-seed additive Level 1 length preservation:        0.5402177987
same-seed pooled point-score length preservation:      0.5340749041
delta:                                                 -0.0061428946
selector trace source:                                point_score_top20_mean
target diffusion:                                     passed, final support gt_0.01 = 0.234375
```

Phase 50 failure diagnosis:

```text
classification: counterfactual_to_production_score_to_mask_mismatch
direct QLU-loss source: point-level/local components, not length-score term
additive retained-segment counts: [3,0,2,3,0,2,3,0,2]
pooled retained-segment counts:   [3,1,1,3,1,1,3,1,1]
additive allocation diagnosis:    length_support_materially_influences_allocation
pooled allocation diagnosis:      score_dominated_length_support_conflict
```

Phase 51 mask-delta diagnosis:

```text
classification: learned_slot_spreading_swapped_query_hit_points_for_zero_hit_coverage
retained-mask Jaccard: 0.6666666667
removed learned points: [82,178,274]
added learned points:   [61,157,253]
removed raw marginal QLU sum: 0.0210543920
added raw marginal QLU sum:   0.0001858789
net estimate:                 -0.0208685131
observed QLU delta:            -0.0208646652
removed query-hit count: 2
added query-hit count:   0
```

Phase 52 segment-head compression root diagnostic:

```text
classification: broad_soft_target_plus_underpowered_rank_pressure_causes_compressed_wrong_way_segment_head
segment target positive fraction: 0.9444444444
segment target support gt_0.01: 0.9126984127
segment target top-5%-mass recall: 0.1433802217
target segment oracle alignment:
  spearman vs oracle mass: 0.8431893688
  top-25% oracle mass recall: 0.4900304343
learned segment fit:
  target std: 0.2152189165
  prediction std: 0.0109573500
  prediction std / target std: 0.0509125787
  Kendall tau: 0.2148176732
selector allocation:
  segment_score_to_allocation_spearman: 0.8771587805
learned retained boundary:
  raw_score_spearman: 0.7192082111
  query_hit_branch_spearman: 0.7441348974
  behavior_branch_spearman: 0.7111436950
  segment_score_spearman: -0.5381231672
```

Phase 53 segment top-k rank-loss Level 1 wiring:

```text
status: rejected; production loss patch reverted
target diffusion: passed, final support gt_0.01 = 0.234375
same-seed reference MLQDS QueryLocalUtility: 0.1064832750
candidate MLQDS QueryLocalUtility:           0.1064832750
candidate - reference:                       0.0
same-seed reference length:                  0.5402177987
candidate length:                            0.5402177987
candidate - reference length:                0.0
segment fit:
  reference prediction std / target std: 0.0259016158
  candidate prediction std / target std: 0.0265600364
  reference Kendall tau: 0.1532986111
  candidate Kendall tau: 0.1497829861
learned retained boundary:
  reference segment_score_spearman: 0.4014447884
  candidate segment_score_spearman: 0.3808049536
```

Next admissible checkpoint:

```text
segment_rank_loss_gradient_path_diagnostic
```

This checkpoint is diagnostic only. It must quantify the actual segment-rank
loss magnitude and gradient contribution against point BCE, pooled segment BCE,
existing pairwise segment loss, auxiliary-loss scaling, and the primary budget
loss before adding another loss term or scalar. It must not change selector,
score, target, model, prior, metric, workload, or production loss semantics.
It cannot claim final success, and it cannot proceed to Level 2, Level 3, or
final grid.

## 1. End-state objective

The desired final system is a query-driven, workload-blind AIS compressor.

At deployment/eval time, the system receives trajectories and train-derived
artifacts only. It must produce retained masks before future range queries are
known. Future queries are scored only after those masks are frozen.

The final system must satisfy four requirements:

1. **Workload-blind compression**
   - No eval query boxes, query tensors, query/point containment labels,
     eval-query boundary distances, or eval-query-derived features before mask
     freeze.

2. **Query-driven learned behavior**
   - The model learns from generated or historical training workloads.
   - It learns stable workload priors and query-local behavior value.
   - Learned scores, priors, and heads materially affect retained masks.

3. **Future-query usefulness**
   - Compressed trajectories preserve likely in-query point mass.
   - Within likely query ranges, retained points preserve enough local evidence
     for interpolation, turns/behavior changes, continuity, and movement
     reconstruction.

4. **Sensible global trajectories**
   - Global geometry is not the primary objective, but retained trajectories
     must remain plausible.
   - Endpoint sanity, rough length preservation, and bounded shape distortion
     remain guardrails.

A result caused mainly by query-conditioned inference, checkpoint leakage,
historical KNN lookup, a large temporal scaffold, or selector tricks is not a
valid final result.

---

## 2. Design contract

The implementation contract connects five components:

```text
1. future-query workload profile
2. query-local metric
3. factorized train labels and train-only priors
4. trainable workload-blind model
5. selector with material learned control and sanity guardrails
```

All five must be aligned. If one is incoherent, model or selector sweeps waste
time.

Correct order of work:

1. Stabilize workload generation and signatures.
2. Verify train-derived priors predict held-out query usefulness.
3. Improve target/model only after useful signal exists.
4. Improve selector only when learned scores are useful but retained masks are
   poor.
5. Run real-AIS probes only after synthetic/debug probes have clean workload
   health.
6. Run the final grid only after strict support-valid single-cell evidence clears
   required gates.

---

## 3. Protocol rules

### Allowed

- Training labels from training workloads.
- Query-aware teachers on training workloads.
- Historical or train-derived priors built before eval compression.
- Validation workloads for checkpoint selection if validation masks remain blind
  and final eval queries are not used for selection.
- Eval workloads for scoring only after masks are frozen.

### Forbidden for final claims

- Eval queries passed into the model, feature builder, selector, retained-set
  decision, checkpoint selector, or query-prior builder before compression.
- Eval point/query containment features.
- Eval query boundary-distance features.
- Query cross-attention at eval compression time.
- Checkpoint selection using final eval-query performance.
- Treating query-conditioned `range_aware` as workload-blind success.
- Treating historical-prior KNN lookup as final learned-model success.
- Using eval geometry labels or geometry-label blending for a final
  workload-blind mask.
- Using a large temporal scaffold that makes learned scores mostly irrelevant.

### Required serious-run artifact flags

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

## 4. Evidence levels and promotion rules

Use the smallest evidence level that can answer the checkpoint question.

| Level | Purpose | May change acceptance state? |
| --- | --- | --- |
| Static/code inspection | Naming, schema, protocol, or guardrail checks | No |
| Unit/guardrail tests | Validate implementation contracts | No |
| Level 1 smoke | Wiring, artifact fields, CLI compatibility | No |
| Level 2 minimum strict | Early gate localization | No final claim; may justify a targeted next probe |
| Level 3 strict single-cell | Main current evidence level | Can define current blocker boundary |
| Final grid | Multi-profile/compression/seed evidence | Required for final claim |

Promotion rules:

- Do not promote a variant from Level 1 smoke.
- Do not promote a Level 2 run that fails workload health, signature, target
  diffusion, support overlap, or predictability unless the checkpoint was only
  intended to localize that gate failure.
- Do not promote a Level 3 run that improves surface score while failing
  learning causality.
- If metric/profile/target semantics change, restart at smaller strict levels
  before treating a Level 3 comparison as current evidence.
- The final grid is blocked until strict smaller evidence passes required gates.

---

## 5. Gate order

Diagnose in this order. Do not skip ahead to model or selector changes when an
earlier gate is unhealthy.

1. **Workload stability**
   - Query generation must be healthy, non-exhausted, and within coverage and
     rejection-pressure limits.

2. **Support overlap**
   - Eval points must lie inside useful train-prior support often enough for
     train-derived priors to be meaningful.

3. **Target diffusion**
   - Labels must have enough support and not collapse into all-zero/all-one
     targets.

4. **Workload signature**
   - Train/eval workload signatures must match profile semantics, anchor and
     footprint distributions, hit fractions, duplicate rates, and broad-query
     rates.

5. **Predictability and prior-predictive alignment**
   - Train-derived priors must predict held-out query-local utility well enough
     to justify learned use.

6. **Learning causality**
   - Shuffled scores, missing priors, missing heads, untrained models, and
     non-learned controls must behave as expected.

7. **Selector and retained-marginal alignment**
   - Scores and segment allocation must rank exact retained-decision marginal
     utility, not just move masks.

8. **Global sanity**
   - Endpoint, length, and shape guardrails must be computed and improved where
     possible. In the current local-query-learning phase, global sanity is not
     the first hard blocker ahead of learning causality when it already passes.

---

## 6. Active metric requirements

`QueryLocalUtility` is the active primary metric.

Rules:

- Direct query-local fields only.
- No fallback from legacy `range_point_f1` into point mass.
- No fallback from old shape, temporal, average-gap, ship, boundary, crossing,
  or replacement fields into query-local behavior.
- Keep `RangeUsefulLegacy` and `RangePointF1` as diagnostic/compatibility
  outputs only.
- Final claims must use `QueryLocalUtility` protocol summaries, not legacy
  RangeUseful summaries.

Rationale:

The old aggregate mixed too many proxies. The current metric intentionally asks a
narrower question: are the retained points directly useful for local range-query
point mass and local movement evidence?

---

## 7. Active workload-profile requirements

Active profile:

```text
workload_profile_id = range_query_mix
```

Required default properties:

```yaml
anchor_family_weights:
  density: 0.80
  sparse_background_control: 0.20

footprint_family_weights:
  medium_operational: 0.6923076923076923
  large_context: 0.3076923076923077

medium_operational:
  spatial_radius_km: 2.2
  time_half_window_hours: 5.0
  elongation_allowed: false
  point_hit_fraction_band: [0.006, 0.030]

large_context:
  spatial_radius_km: 4.0
  time_half_window_hours: 8.0
  elongation_allowed: false
  point_hit_fraction_band: [0.010, 0.045]
```

Coverage/profile rules:

```text
coverage_calibration_mode = profile_sampled_query_count
workload_stability_gate_mode = final
default final profile target coverage = 0.30
default final profile max coverage overshoot = 0.020
```

Final-profile query generation should fail if:

```text
range_acceptance.exhausted == true
stop_reason in {range_acceptance_exhausted, range_coverage_guard_exhausted}
accepted query count < 8
rejection_rate > 0.85
coverage_guard_rejection_pressure > 2.0
coverage target outside guard
profile id mismatch
coverage calibration mode not profile_sampled_query_count
```

Practical current scale notes:

- Current Level 3 evidence uses 40 requested queries and passes workload health.
- Generator-only Level 3 probes support `n_queries=48` at the current profile.
- A 64-query floor has shown coverage-overshoot pressure under the current
  `target_coverage=0.30` and `max_coverage_overshoot=0.020` envelope.
- Do not change model code to compensate for unhealthy query generation.

---

## 8. Active target/model/selector requirements

### Target

Active target mode:

```text
query_local_utility_factorized
```

Active heads:

```text
query_hit_probability
conditional_behavior_utility
boundary_event_utility
replacement_representative_value
segment_budget_target
path_length_support_target
```

Active target details:

```text
query_hit_target_variant = raw_query_hit_ship_evidence_multiplier
query_hit_target_base_source = raw_query_hit_probability_times_0.65_plus_0.35_positive_mean_normalized_ship_query_evidence
conditional_behavior_target_variant = query_segment_local_behavior_utility
replacement_representative_keep_fraction = 0.35
segment_budget_target_aggregation = top20_mean
```

Behavior supervision remains masked to query-hit points. Do not widen it to
all-point zero supervision; that makes the behavior head relearn query-hit
support instead of query-local behavior value.

The query-hit head now uses a raw-q-hit-scale-preserving ship-evidence
multiplier. Ship/family query evidence may re-rank points inside raw q-hit
support, but it must not normalize sparse hit support into a broad gate.

The rejected Phase 36 broad gate, `query_evidence_gate_hit_ship_blend`, failed
strict Level 2 target diffusion: `final_label_support_fraction_above_max`, with
final support `0.705853` above the `0.5` gate. The narrowed Phase 38 target
passed Level 1 wiring target diffusion with final support `0.217014`, then
passed strict Level 2 target diffusion with final support `0.088790`. This is
historical context, not the current boundary. The current boundary uses the
additive q-hit / behavior score composition, which passed strict Level 2 target
diffusion with final support `0.235119` but still failed learning causality and
global sanity. The failed child gates have now been localized: priors remain
mask-immaterial, the behavior head remains weak despite material ablation, and
the segment-budget head is a compressed non-causal allocation signal.

### Model

Active final-candidate model:

```text
workload_blind_range
```

Rules:

- The model must ignore eval query tensors at compression time.
- It may consume query-free context features and train-derived prior fields.
- Query-conditioned `baseline` and `range_aware` models are diagnostic/teacher
  paths only.
- Historical-prior models are diagnostic unless they beat and explain their
  non-learned controls.

### Selector

Active selector:

```text
learned_segment_budget
```

Rules:

- Retained masks must be frozen before eval query scoring.
- The selector must preserve attribution for skeleton, learned, fallback, and
  length-repair decisions.
- Learned control must remain material; do not hide weak learning behind a large
  temporal scaffold or broad geometry override.

---

## 9. Current blocker and what comes next

The current code path passed strict Level 2 target diffusion under the additive
q-hit / behavior composition, but is still blocked by learning causality and
global sanity. `Next-Iterations.md` remains the source of truth for the current
evidence boundary.

The semantic blocker is:

```text
query-prior fields are immaterial to retained masks
behavior ablation is material, but the conditional-behavior head is still weak
segment-budget head is compressed and not causal
pooled point-score allocation looked better counterfactually but failed as primary path
global length preservation is below the acceptance floor
```

Next admissible work:

1. **Segment-rank loss gradient-path diagnostic**
   - Phase 53 rejected the top-k rank-loss patch because it did not materially
     change the mask, score, segment-head compression, or retained-boundary
     alignment.
   - Measure actual segment-rank loss magnitude and gradient contribution before
     adding another loss term, scalar, or target change.
   - Do not change metric/profile/target/model/prior/selector semantics.
   - Do not rerun pooled point-score promotion, path-length allocation, selector
     allocation-source patches, selector floors, raw coverage overrides, or
     larger segment-rank scalars.

2. **Query-prior materiality remains unresolved**
   - The additive strict Level 2 localization shows priors are predictive and
     reach head logits, but retained-mask Jaccard stays `1.0`, mean absolute
     head-probability delta is about `2.9e-05`, and high-marginal score-output
     delta is `0.0`.
   - Do not add another scalar prior boost, generic prior residual,
     route-density exposure, prior adapter, or prior-only loss from this
     evidence.

Recommended next checkpoint shape:

```text
checkpoint: segment_rank_loss_gradient_path_diagnostic
scope: diagnostic only; quantify segment-rank loss magnitude and gradient
contribution against point BCE, pooled segment BCE, existing pairwise segment
loss, auxiliary-loss scaling, and the primary budget loss
reference artifacts:
artifacts/results/additive_qhit_behavior_score_composition_level2_seed2539/example_run.json
artifacts/results/additive_qhit_behavior_score_composition_level2_seed2539/semantic_diagnostic.json
artifacts/results/additive_level2_child_gate_root_localization/diagnostic.json
artifacts/results/pooled_point_score_segment_allocation_level1_smoke/example_run.json
artifacts/results/pooled_point_score_segment_allocation_level1_smoke/semantic_diagnostic.json
artifacts/results/pooled_point_score_segment_allocation_level1_smoke/rejection_diagnostic.json
artifacts/results/pooled_point_score_allocation_failure_diagnosis/diagnostic.json
artifacts/results/segment_allocation_mask_delta_diagnostic/diagnostic.json
artifacts/results/segment_budget_head_compression_root_diagnostic/diagnostic.json
artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring/example_run.json
artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring/semantic_diagnostic.json
artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring/rejection_diagnostic.json
derive first, instrument only if fields are missing
no Level 2
no final grid
no metric/profile changes
no selector-floor, raw-coverage, length-scaffold, path-length allocation,
behavior-rank, larger segment-rank scalar, prior-scale, generic residual, or
historical-default sweep
```

---

## 10. Explicitly rejected paths

Do not repeat these without a new hypothesis that explains why prior evidence no
longer applies:

- generic post-context prior residuals;
- scalar prior amplification or route-density-prior exposure as a standalone fix;
- behavior-rank-only weight sweeps;
- all-point zero behavior supervision;
- low-floor behavior formula changes;
- sparse-head BCE normalization as a default;
- direct turn/continuity behavior-label rewrites;
- component-local head target rewrites that fail promotion gates;
- coverage-shrink generator patches that change query geometry while preserving
  stale footprint metadata;
- selector allocation-floor tuning as a substitute for head/target semantics;
- length-support weighting as proof of learned query-local behavior;
- final-grid runs while learning causality fails.

---

## 11. Documentation and artifact hygiene

When adding a checkpoint:

1. Keep `query-driven-implementation-progress.md` short.
2. Record only:
   - hypothesis,
   - artifact path,
   - scale/seed,
   - gate state,
   - key numbers,
   - blocker interpretation,
   - decision.
3. Keep raw stdout and full metrics in `artifacts/results/`.
4. Do not rename defaults casually. Naming changes must be semantic and guarded.
5. Do not keep old chronological names as active product names.
6. Do not compare old `RangeUsefulLegacy` / `QueryUsefulV1` scores as acceptance
   evidence for current `QueryLocalUtility` defaults.
