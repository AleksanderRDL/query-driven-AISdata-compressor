# Range_QDS Query-Driven Implementation and Research Guide

This is the operating guide for work on the query-driven AIS dataset/trajectory compression. It is written for a new engineer or agent starting dev-work on the repository.

This guide is intentionally limited to stable intent, protocol, gate order, reference metric/workload definitions, and documentation rules.

---

## 0. End-state objective

The desired final system is a query-driven, workload-blind AIS compressor.

At deployment/eval time, the system receives trajectories and train-derived
artifacts only. It must produce retained masks before future range queries are
known. Future queries are scored only after those masks are frozen.

The final system must satisfy four requirements:

1. **Workload-blind compression**
   - No evaluation (thosed used in the very final benchmarking stage): 
        1. query boxes,  
        2. query tensors, evaluation 
        3. query/point containment labels,
        4. query boundary distances, or eval-query-derived features before mask freeze.

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

---

## 1. Design contract

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

## 2. Protocol rules

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

## 3. Evidence levels and promotion rules

Use the evidence level that can answer the checkpoint question.

| Level | Purpose | May change acceptance state? |
| --- | --- | --- |
| Static/code inspection | Naming, schema, protocol, or guardrail checks | No |
| Unit/guardrail tests | Validate implementation contracts | No |
| Level 1 smoke | Wiring, artifact fields, CLI compatibility | No |
| Level 2 minimum strict | Early gate localization | No final claim; may justify a targeted next probe |
| Level 3 strict single-cell | Main strict single-cell evidence level | Can define blocker boundary |
| Final grid | Multi-profile/compression/seed evidence | Required for final claim |

---

## 4. Gate order

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
     possible. Do not let a global-sanity failure obscure an earlier
     learning-causality failure unless the checkpoint explicitly targets
     guardrails.

---

## 5. Reference scoring profile

`QueryLocalUtility` is the primary metric for this protocol.

```yaml
Point mass:
  query_point_recall: 0.50

Query-local behavior:
  query_local_interpolation_fidelity: 0.20
  query_local_turn_change_coverage: 0.15
  query_local_continuity: 0.10

Global sanity:
  endpoint_or_skeleton_sanity: 0.02
  global_shape_guardrail_score: 0.02
  length_preservation_guardrail: 0.01
```

## 6. Reference workload profile

`range_query_mix` is the primary workload profile for this protocol.

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

Coverage/profile:

```text
coverage_calibration_mode = profile_sampled_query_count
workload_stability_gate_mode = final
final profile target coverage = 0.30
final profile max coverage overshoot = 0.020
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

---

## 7. Documentation and artifact hygiene

1. Keep this guide focused on stable objective, protocol, gates, and profile definitions.
2. Keep `Next-Iterations.md` focused only on information needed going into the immediate next couple of tasks/iterations.
3. Keep the historical log `query-driven-implementation-progress.md` short and record only
    - hypothesis
    - gate state
    - key numbers
    - blocker interpretation
