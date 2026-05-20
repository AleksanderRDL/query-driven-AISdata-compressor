# Next Iterations

Use this file as the immediate handoff for the next person or agent. Use
`query-driven-implementation-research-guide.md` for the full protocol, gates,
evidence levels, and default stack. Use
`query-driven-implementation-progress.md` for the short checkpoint log.

---

## 0. Continue from here

Current status: **active, not accepted**.

Current reference artifact:

```text
artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/example_run.json
```

This is the current blocker-localizing reference. It is a source-stratified
Level 3 strict synthetic replay with 64 ships, 256 points, 40 requested queries,
5 epochs, seed `2527`, `range_query_mix`, `QueryLocalUtility`,
`query_local_utility_factorized`, `workload_blind_range`, and
`learned_segment_budget`.

Key result:

```text
MLQDS QueryLocalUtility:           0.1431090566
uniform QueryLocalUtility:         0.1247681518
Douglas-Peucker QueryLocalUtility: 0.1153266238
```

Gate state:

```text
passed:
  workload stability
  support overlap
  target diffusion
  workload signature
  predictability
  prior-predictive alignment
  global sanity

failed:
  learning causality
  final grid
```

Do not make final success claims from this run. Do not run the final grid yet.

---

## 1. Current defaults to preserve unless the checkpoint explicitly changes them

```yaml
primary_metric: QueryLocalUtility
score_group_weights:
  query_point_mass: 0.50
  query_local_behavior: 0.45
  global_sanity: 0.05
workload_profile_id: range_query_mix
anchor_family_weights:
  density: 0.80
  sparse_background_control: 0.20
footprint_family_weights:
  medium_operational: 0.6923076923076923
  large_context: 0.3076923076923077
footprint_point_hit_fraction_bands:
  medium_operational: [0.006, 0.030]
  large_context: [0.010, 0.045]
range_training_target_mode: query_local_utility_factorized
conditional_behavior_target_variant: query_segment_local_behavior_utility
replacement_representative_keep_fraction: 0.35
segment_budget_target_aggregation: top20_mean
model_type: workload_blind_range
selector_type: learned_segment_budget
checkpoint_score_variant: query_local_utility
checkpoint_selection_metric: uniform_gap
```

These defaults are not permanent doctrine, but they are the current coherent
reference. Do not change metric weights, workload profile, target semantics,
model architecture, and selector settings in the same checkpoint.

---

## 2. Current blocker

The current blocker is **learning causality**, specifically query-prior and
behavior-head dependence.

Current reference child-gate status:

```text
positive controls that work:
  shuffled scores lose by:       0.0241102168
  untrained loses by:            0.0177732905
  no segment-budget loses by:    0.0099825888
  prior-field-only loses by:     0.0319842784

failed controls:
  shuffled_prior_fields_should_lose:        0.0
  without_query_prior_features_should_lose: 0.0
  without_behavior_utility_head_should_lose: 0.0014985765 < 0.005
```

Current alignment readout:

```text
raw-score retained-marginal Spearman:       0.2779
selector-score retained-marginal Spearman:  0.2881
segment-score retained-marginal Spearman:  -0.0812
behavior-component retained-marginal Spearman: -0.0486
behavior prediction std: 0.002631
target std:             0.166493
behavior Kendall tau:   0.0251
```

Interpretation:

- The model score is useful enough to beat baselines in the reference cell.
- The segment-budget head is material enough that removing it loses.
- Query-prior features do not materially affect the retained mask.
- The behavior head is almost flat and does not count as causal learned
  behavior.
- Segment-score allocation remains misaligned with exact retained marginal
  utility.

---

## 3. Immediate next checkpoint

Recommended checkpoint name:

```text
semantic_causality_diagnosis_current_reference
```

Goal:

> Explain why query-prior and behavior-head signals fail causality in the
> current healthy Level 3 reference cell, before changing model, target, selector,
> or workload defaults.

Preferred scope:

```text
derived artifact diagnostic first
no training replay unless the diagnostic proves stored artifacts lack required fields
no final grid
no metric/profile changes
no generic loss-weight or prior-scale sweep
```

### Required diagnostic questions

#### A. Behavior-head semantic diagnosis

Answer these from the current reference artifact if possible:

1. Where is `conditional_behavior_utility` target mass located?
2. Is behavior target mass concentrated in points with positive retained
   marginal QueryLocalUtility?
3. Does the trained behavior head rank those points at all?
4. Is behavior signal being absorbed by `query_hit_probability`,
   `replacement_representative_value`, or `segment_budget_target`?
5. Is the final score composition suppressing behavior even where the behavior
   head has nonzero output?
6. Does the selector ignore behavior-aligned points because segment allocation
   sends budget elsewhere?

Minimum rows to emit:

```text
point_index
trajectory_id
source stage / retained source
retained decision type
exact marginal QueryLocalUtility
query_point_recall / behavior / continuity local components when available
query_hit target and head probability
behavior target and head probability
replacement target and head probability
segment-budget target and head probability
final raw score
selector score
segment score
retained mask membership
anchor family
footprint family
query-hit-run id or segment id where available
```

Group summaries to emit:

```text
by retained source
by decision type
by anchor family
by footprint family
by query-hit-run length bucket
by behavior-target quantile
by behavior-head-probability quantile
```

Stop condition:

- Classify behavior failure as one of:
  - target has no useful retained-marginal signal;
  - target has signal but head does not learn it;
  - head learns weak signal but final score suppresses it;
  - final score has signal but selector/segment allocation loses it;
  - behavior is redundant with another head and should not be required as a
    separate causal gate;
  - artifact lacks required fields and a focused instrumentation checkpoint is
    needed.

#### B. Query-prior materiality diagnosis

Answer these from the current reference artifact if possible:

1. Do sampled train-derived prior fields differ between primary and ablated
   prior settings?
2. Do normalized model prior fields differ?
3. Do head logits or probabilities differ?
4. Do raw scores, selector scores, segment scores, or retained masks differ?
5. If priors move model internals but not masks, which stage kills the signal?
6. If priors do not move internals, are they zero, out-of-support, disabled, or
   redundant with query-free point/context features?

Required chain:

```text
sampled_prior_features
model_prior_features
head_output
raw_prediction
score_output
segment_score
selector_score
retained_mask
```

Stop condition:

- Classify prior failure as one of:
  - prior sampling/support failure;
  - prior feature normalization/scaling failure;
  - model ignores prior inputs;
  - priors change heads but final score suppresses them;
  - priors change scores but selector ignores them;
  - priors are redundant with context features;
  - priors are anti-causal for this workload/profile;
  - artifact lacks required fields and a focused instrumentation checkpoint is
    needed.

#### C. Segment-score calibration diagnosis

Answer these from the current reference artifact if possible:

1. Why do raw and selector scores have positive retained-marginal Spearman while
   segment score is negative?
2. Is segment allocation ranking the wrong segments, or selecting weak points
   inside otherwise useful segments?
3. Would pooled point score by segment rank exact retained-marginal segment value
   better than the current segment head?
4. Is path-length/length-support influence improving score while weakening
   query-local segment semantics?

Compare diagnostic segment rankers:

```text
current segment-budget head
neutral segment score
pooled raw point score by segment
pooled selector score by segment
path-length support head
train-side retained-marginal segment teacher, diagnostic only if available
```

Stop condition:

- Classify segment failure as one of:
  - segment target is misaligned with exact marginal utility;
  - segment head fails to learn target;
  - allocation scoring and point-selection scoring are mixed incorrectly;
  - length-support/path sanity is overriding query-local segment allocation;
  - segment score should be demoted until behavior/prior semantics are fixed;
  - artifact lacks required fields and a focused instrumentation checkpoint is
    needed.

---

## 4. Likely files for a diagnostic-only checkpoint

Only touch code if existing artifacts cannot answer the diagnostic questions.
Likely files:

```text
orchestration/diagnostics/*.py
orchestration/selector_diagnostics.py
orchestration/causality.py
orchestration/run_payload.py
learning/targets/query_local_utility.py   # diagnostics only, not target semantics
selection/learned_segment_budget/trace.py # only if trace fields are missing
```

Avoid touching these unless the diagnostic proves a root fix:

```text
scoring/query_local_utility.py
workloads/generation/workload_profiles.py
models/workload_blind_range.py
learning/optimization_epoch.py
selection/learned_segment_budget/core.py
```

---

## 5. Validation for the next checkpoint

For a derived diagnostic-only checkpoint:

```bash
python -m py_compile <changed files>
ruff check <changed files>
pyright <changed files>
pytest <focused unit tests> -q
git diff --check
```

For an instrumentation checkpoint that changes emitted payloads:

```bash
pytest tests/unit/orchestration/test_query_driven_diagnostics.py -q
pytest tests/unit/orchestration/test_query_driven_causality_and_summary.py -q
pytest tests/unit/learning/test_query_local_utility_targets.py -q
pytest tests/guardrails/test_implementation_guardrails.py -q
```

For a scientific replay after a root fix:

1. Start with Level 1 smoke only for wiring.
2. Run Level 2 minimum strict if wiring is clean.
3. Run Level 3 only if Level 2 does not expose an earlier gate failure.
4. Compare against the current reference cell only when metric/profile/defaults
   are unchanged or the evidence boundary has been explicitly reset.

---

## 6. Do not do next

Do not:

- run the final grid;
- loosen gates;
- change `QueryLocalUtility` weights and target semantics in the same checkpoint;
- reintroduce legacy `range_point_f1` fallback into the active metric;
- add generic prior residuals or scalar prior amplification without a prior-flow
  localization result;
- repeat behavior-rank-only sweeps;
- widen behavior supervision to all-point zero negatives;
- replace the segment-budget head with the behavior head;
- tune selector allocation floor or length-support weight as a substitute for
  causal head/target semantics;
- promote a variant that only improves aggregate score while failing child
  causality gates;
- compare old `QueryUsefulV1` or `RangeUsefulLegacy` artifacts as current
  acceptance evidence.

---

## 7. Decision matrix for interpreting the next diagnostic

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
| Segment head ranks marginal utility negatively | Segment target/allocation semantics issue | Do not tune allocation weights; fix segment target/score semantics |
| Pooled point score ranks segments better | Allocation should use point-score-derived segment proxy or better segment target | Test as diagnostic first |

---

## 8. Minimal agent handoff prompt

Use this when starting a new agent:

```text
Continue the Range_QDS query-driven workload-blind compressor implementation.
Read docs/query-driven-implementation-research-guide.md and docs/query-driven-implementation-progress.md first.
Current reference artifact is artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/example_run.json.
Do not run the final grid. Do not change metric/profile defaults. The task is to diagnose failed learning causality in the current healthy Level 3 cell: query-prior ablations have zero effect, no-behavior-head loss is below gate, behavior head is nearly flat, and segment-score marginal alignment is negative. Prefer a derived artifact diagnostic before code changes. Produce a short progress entry with hypothesis, artifact path, key numbers, blocker classification, and decision.
```

---

## 9. Progress-log update format

When a checkpoint finishes, append only this shape to
`query-driven-implementation-progress.md`:

```markdown
## Checkpoint Phase N - <short name>

Status: completed / rejected / accepted-as-new-boundary / implementation-only.

Hypothesis:
- <one or two bullets>

Artifact:
- `<path>`

Scale:
- <static/unit/Level 1/Level 2/Level 3/final-grid>
- <seed and key dimensions if run>

Key results:
- <scores and gate state if run>
- <diagnostic numbers if derived>

Decision:
- <what this proves>
- <what not to do next>
- <next admissible step>
```

Keep raw command output and detailed metrics in artifacts, not in the progress
log.

