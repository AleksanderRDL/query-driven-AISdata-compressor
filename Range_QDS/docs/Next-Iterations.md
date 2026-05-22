# Next Iteration(s)

Use this file as the handoff for the next individual task or tasks to be worked on in the project. It is focused only on information needed going into the immediate next couple of tasks/iterations.

---

## 0. Continue from here

Current status: **active, not accepted**.

Current formula:

```text
additive_raw_query_hit_and_behavior_with_conditional_replacement_modulation_plus_boundary
```

Current strict replay configuration:

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

Strict Level 2 current boundary:

```text
MLQDS QueryLocalUtility:           0.0995482993
uniform QueryLocalUtility:         0.0992909061
Douglas-Peucker QueryLocalUtility: 0.1182249577
MLQDS - uniform:                   +0.0002573932
MLQDS - Douglas-Peucker:           -0.0186766584
target diffusion:                  passed, final support gt_0.01 = 0.2351190476
learning causality:                failed
global sanity:                     failed, length preservation = 0.703978 < 0.75
workload signature:                failed with known point-hit-fraction KS recurrence
```

Phase 48 localization:

```text
prior:
  predictive prior signal reaches the model, but it does not move retained masks
  sampled/model priors change; retained-mask Jaccard stays 1.0
  mean abs head-probability delta is about 2.9e-05
  high-marginal score-output delta is 0.0

behavior:
  behavior is now material by ablation, but the head is still weak
  conditional_behavior_utility tau is 0.040724
  prediction std / target std is about 0.103

segment:
  segment-budget target is oracle-aligned, but the learned segment head is
  compressed and non-causal
  pooled point-score allocation scored 0.114785381 in the strict Level 2
  diagnostic, +0.015237081 above primary

length:
  every trajectory remains below the 0.75 length floor
  length-only allocation can clear the floor counterfactually, but pure
  path-length allocation scores only 0.0858993835 on strict Level 2
```

Phase 49 rejected wiring result:

```text
rejected checkpoint: pooled_point_score_segment_allocation_level1_wiring
seed: 2557, same as additive Level 1 reference
wiring: passed; selector trace source = point_score_top20_mean
target diffusion: passed; final support gt_0.01 = 0.234375

same-seed Level 1 comparison:
  additive reference MLQDS QueryLocalUtility: 0.1064832750
  pooled point-score MLQDS QueryLocalUtility: 0.0856186098
  delta: -0.0208646652

  additive reference length preservation: 0.5402177987
  pooled point-score length preservation: 0.5340749041
  delta: -0.0061428946

new run baselines:
  uniform QueryLocalUtility:         0.1166761729
  Douglas-Peucker QueryLocalUtility: 0.0953898417
```

Interpretation:

- Promoting pooled final point-score segment allocation to the primary path
  failed the Level 1 stop condition. Production selector semantics were
  reverted.
- The strict Level 2 diagnostic advantage did not survive same-seed Level 1
  wiring promotion. The next step is diagnosis, not another selector tweak.
- The failed wiring run shows `path_length_support_allocation_query_local_utility
  = 0.1064832750`, exactly matching the additive Level 1 reference, but using
  that as a fix would be query-free length/guardrail compensation from failed
  evidence. It is not admissible as a production patch.
- Query-prior materiality remains unresolved. Segment-head materiality remains
  unresolved. Final success is not close.

Phase 50 failure diagnosis:

```text
classification: counterfactual_to_production_score_to_mask_mismatch
direct QLU-loss source: point-level/local components, not length-score term
weighted QLU deltas:
  query_point_recall:                 -0.0115942029
  query_local_interpolation_fidelity: -0.0042138399
  query_local_turn_change_coverage:   -0.0049234360
  length_preservation_guardrail:      -0.0000614289
selector retained-segment spread:
  additive reference: 6 segments retained, counts [3,0,2,3,0,2,3,0,2]
  pooled primary:     9 segments retained, counts [3,1,1,3,1,1,3,1,1]
allocation diagnosis:
  additive reference: length_support_materially_influences_allocation
  pooled primary:     score_dominated_length_support_conflict
```

The strict Level 2 counterfactual advantage did not validate the full
production retained-mask path. Pooled allocation broadens learned-slot
placement and degrades point-level query mass/local fidelity. Segment-level
oracle coverage improves, but that is the wrong proxy here.

Phase 51 mask-delta diagnosis:

```text
classification: learned_slot_spreading_swapped_query_hit_points_for_zero_hit_coverage
retained-mask Jaccard: 0.6666666667
common retained points: 12
removed learned points from additive path: [82,178,274]
added learned points in pooled path:       [61,157,253]
removed-point raw marginal QLU sum: 0.0210543920
added-point raw marginal QLU sum:   0.0001858789
net added-minus-removed estimate:  -0.0208685131
observed pooled-minus-additive QLU: -0.0208646652
absolute residual:                  0.0000038479
removed query-hit count: 2
added query-hit count:   0
```

The rejected pooled path swapped two query-hit learned points plus one small
positive learned point for three zero-query-hit learned points. That explains
the Level 1 loss. The active segment blocker is no longer an allocation-source
question; it is why the learned segment-budget head is compressed/non-causal
despite an oracle-aligned segment target.

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
  segment_score top-minus-bottom marginal: -0.0009740413
```

The segment target has real segment signal, but the learned segment head mostly
learns mean calibration. The selector strongly follows that head, so the segment
head is causally harmful or neutral at the retained-mask boundary. This does not
justify another selector-source change. It justifies a narrow segment-head rank
learning fix using the existing active target.

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
learning-causality:
  no-segment-budget-head ablation delta unchanged: 0.0251730292
  without-query-prior-features delta unchanged: 0.0
```

The top-k rank-loss patch was too weak or too poorly coupled to matter. It
preserved the Level 1 score, but did not materially change the mask, head
compression, segment rank fit, or learning-causality readout. Do not rerun this
path with a larger scalar until the gradient path is measured.

Next admissible step:

```text
segment_rank_loss_gradient_path_diagnostic
```

Scope:

```text
Diagnostic only. Quantify the actual segment-rank loss magnitude and gradient
contribution against point BCE, pooled segment BCE, existing pairwise segment
loss, auxiliary-loss scaling, and the primary budget loss before adding another
loss term or scalar weight.
```

Allowed evidence:

- derive from existing code/artifacts where possible;
- add focused instrumentation only if loss/gradient fields are missing;
- one tiny implementation probe only if required to measure gradients;
- no Level 1 replay unless the diagnostic identifies a materially new root fix.

Do not change metric/profile/target/model/prior/selector semantics. Do not add a
larger segment-rank scalar, selector floor, raw coverage override, length
scaffold, generic behavior-rank weight, prior boost, prior residual,
route-density exposure, or final-grid run. Do not rerun pooled point-score
promotion, path-length allocation, or any selector allocation-source patch.

---

## 1. Current blocker

For the current code path, the immediate blockers are **learning causality** and
**global sanity**. The additive score composition cleared strict Level 2 target
diffusion and made the behavior ablation material, but this is still not an
accepted query-driven compressor.

Strict Level 2 result:

```text
MLQDS QueryLocalUtility:           0.0995482993
uniform QueryLocalUtility:         0.0992909061
Douglas-Peucker QueryLocalUtility: 0.1182249577
MLQDS - uniform:                   +0.0002573932
MLQDS - Douglas-Peucker:           -0.0186766584
```

Gate state:

```text
passed:
  workload stability
  support overlap
  target diffusion  # final support gt_0.01 = 0.2351190476
  predictability
  prior-predictive alignment

failed:
  workload signature  # known Level 2 point-hit-fraction KS recurrence
  learning causality
  global sanity       # length preservation 0.703978 < 0.75
  final grid          # not admissible
```

Learning-causality child state:

```text
passed at threshold but not enough for final success:
  shuffled_scores_should_lose:       0.00638639
  untrained_model_should_lose:       0.0110130
  without_behavior_head_should_lose: 0.00622185

failed:
  shuffled_prior_fields_should_lose:        0.0
  without_query_prior_features_should_lose: 0.0
  without_segment_budget_head_should_lose: -0.000488398
```

Localized failures:

```text
prior:
  Priors are predictive and reach the model, but they do not move retained
  masks. Retained-mask Jaccard stays 1.0, mean abs head-probability delta is
  about 2.9e-05, and high-marginal score-output delta is 0.0.

behavior:
  Behavior is now material by ablation, but conditional_behavior_utility is
  still weak: tau = 0.040724 and prediction std / target std is about 0.103.

segment:
  The segment-budget target is oracle-aligned, but the learned segment head is
  compressed and non-causal. Pooled final point-score allocation scores
  0.114785381, +0.015237081 above primary. Neutral segment score also beats
  primary by +0.000488398.

length:
  Every trajectory remains below the 0.75 length floor. Length-only allocation
  can clear the length floor counterfactually, but pure path-length allocation
  scores only 0.0858993835 on QueryLocalUtility.
```

The narrow production candidate from Phase 48 was tested and rejected at
Level 1. Pooled final point-score segment allocation did prove its trace wiring,
but it degraded same-seed QueryLocalUtility and length versus the additive
Level 1 reference. Do not re-promote it without a materially new hypothesis.

Phase 50 explains the contradiction: the strict Level 2 diagnostic measured an
offline allocation alternative, while the Level 1 primary-path run changed the
actual retained mask. Phase 51 made the added/removed point deltas explicit.
Phase 52 then localized the remaining segment blocker to learned head
compression and wrong-way retained-boundary ranking, not selector allocation.

---

## 2. Immediate next checkpoint

Recommended checkpoint name:

```text
segment_rank_loss_gradient_path_diagnostic
```

Hypothesis:

> The rejected top-k rank-loss patch did not move the model because its gradient
> contribution is likely too small, badly normalized, or dominated by the point
> BCE / primary budget objective. Measure the actual loss and gradient path
> before adding another loss term or scalar.

Evidence level / probe scale:

```text
derived diagnostic or instrumentation-only tiny probe
```

Exact stop condition:

- Stop once the diagnostic explains whether the segment-rank loss is numerically
  too small, gradient-blocked, dominated by point/primary losses, or pointed at
  the wrong pooled targets.
- Stop earlier if existing artifacts lack the required loss/gradient fields; in
  that case specify the smallest instrumentation-only probe.
- Do not change selector, target, model, prior, score, or production loss
  semantics in this checkpoint.
- Do not run Level 2, Level 3, or final grid.

Expected artifact path if a run is planned:

```text
artifacts/results/segment_rank_loss_gradient_path_diagnostic/diagnostic.json
```

Forbidden in this checkpoint:

```text
no metric/profile/target/model/prior semantic changes
no selector allocation floor tweak
no raw coverage override
no length scaffold
no guardrail weakening
no generic behavior-rank weight
no larger segment-rank scalar before gradient-path diagnosis
no prior boost, prior residual, route-density exposure, prior adapter, or prior-only loss
no production selector semantic change
no rerun of pooled point-score promotion
no segment-allocation replacement patch
no path-length allocation patch
no Level 2
no Level 3
no final grid
```

---

## 3. Validation for the next checkpoint

For the next checkpoint:

```bash
jq empty artifacts/results/additive_level2_child_gate_root_localization/diagnostic.json
jq empty artifacts/results/pooled_point_score_segment_allocation_level1_smoke/rejection_diagnostic.json
jq empty artifacts/results/pooled_point_score_allocation_failure_diagnosis/diagnostic.json
jq empty artifacts/results/segment_allocation_mask_delta_diagnostic/diagnostic.json
jq empty artifacts/results/segment_budget_head_compression_root_diagnostic/diagnostic.json
jq empty artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring/rejection_diagnostic.json
git diff --check
```

If new instrumentation is genuinely required, validate only the touched
instrumentation path first:

```bash
uv run --group dev -- python -m py_compile <changed files>
uv run --group dev -- ruff check <changed files>
uv run --group dev -- pyright <changed files>
uv run --group dev -- pytest <focused unit tests> -q
```

For later replay only after a new root fix is justified:

1. Start at static checks and focused unit tests.
2. Run Level 1 only for the specific new root fix.
3. Run Level 2 only if Level 1 does not degrade QueryLocalUtility, length, or
   target diffusion.
4. Run Level 3 only if Level 2 clears the relevant child gates.
5. Do not run final grid until every required child gate passes.
6. Compare against the current reference cell only when metric/profile/defaults
   are unchanged or the evidence boundary has been explicitly reset.

---

## 4. Decision matrix for interpreting the next diagnostic

| Observation | Interpretation | Next action |
| --- | --- | --- |
| Behavior target has weak marginal signal | Target semantics are wrong | Redesign behavior target, restart at Level 1/2 |
| Behavior target has signal, head flat | Learning/loss/architecture issue | Add focused head-learning fix, not selector tuning |
| Behavior head has signal, final score suppresses it | Score composition issue | Adjust factorized composition with Level 1/2 restart |
| Behavior score has signal, selector loses it | Selector/segment allocation issue | Diagnose allocation split before changing target |
| Priors do not change sampled/model inputs | Prior support/sampling issue | Fix prior construction/sampling, then rerun strict small evidence |
| Priors change inputs but not heads | Model ignores priors | Focus prior encoder/feature integration, not prior scale alone |
| Priors change heads but not scores | Factorized composition suppresses priors | Diagnose head-to-score composition |
| Priors change scores but not masks | Selector insensitivity | Diagnose score-to-mask boundary and allocation |
| Segment head ranks marginal utility negatively | Segment head learning or target calibration issue | Do not tune allocation weights; fix the head learning path or target semantics |
| Pooled point score ranks segments better in a counterfactual but fails as primary path | Diagnostic/production mismatch or length-selection interaction | Explain the mismatch before changing selector semantics |
