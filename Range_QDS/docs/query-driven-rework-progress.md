# Query-Driven Rework Progress

This is the concise checkpoint log required by `docs/query-driven-rework-guide.md`.
The guide is the source of truth. Detailed stdout, full command output, and raw
metrics belong in `artifacts/results/`, not here.

## Current State - 2026-05-18

Status: active, not complete.

Latest strict single-cell artifact:
- `artifacts/results/query_driven_v2_checkpoint42_mode_aware_current_best_strict_local/example_run.json`

Latest derived diagnostic artifact:
- `artifacts/results/query_driven_v2_checkpoint43_prior_head_selector_marginal_diagnosis/prior_head_selector_marginal_diagnosis.json`

Current strict result:
- MLQDS QueryUsefulV1: `0.1662115143`
- uniform QueryUsefulV1: `0.1421296610`
- Douglas-Peucker QueryUsefulV1: `0.1671038781`
- MLQDS RangeUsefulLegacy: `0.1524363397`
- MLQDS length preservation: `0.7915916346`

Current gate state:
- Passed: workload stability, support overlap, prior-predictive alignment,
  target diffusion, workload signature, global sanity.
- Failed: predictability, learning causality.
- Not run: full workload-profile/compression final grid.
- Final success allowed: `false`.

Current blockers:
- Predictability misses aggregate Spearman and PR-AUC lift: Spearman
  `0.1109086186 < 0.15`; PR-AUC lift `1.2304850435 < 1.25`.
- Learning causality fails shuffled scores, shuffled priors, no query priors,
  no behavior head, and no segment-budget head.
- MLQDS beats uniform but still narrowly loses to Douglas-Peucker on
  QueryUsefulV1, so there is no product win claim.

Current interpretation:
- The query-count workload-signature blocker is resolved by the mode-aware
  signature invariant.
- Train-derived priors show useful local lift, but the learned heads/selector do
  not turn that signal into enough marginal retained-mask value.
- Checkpoint43 classifies the current blocker as score-composition and selector
  marginal alignment, with prior-to-head transfer as a contributing failure.
- Exact retained-decision marginal diagnostics are more informative than generic
  head-fit metrics right now. Checkpoint42 retained-marginal alignment: selector
  Spearman `-0.0077522559`, raw-score Spearman `-0.0248828079`.
- Freeze time is still high: checkpoint42 freeze-retained-masks `363.45s`, mostly
  query-free ablation freeze `260.07s`.

Do not do next:
- Do not run the final grid.
- Do not loosen predictability, learning-causality, support, workload, or global
  sanity gates.
- Do not compensate with large temporal scaffold, raw coverage overrides, weak
  length guardrails, or query-conditioned inference.
- Do not tune model/selector from generation-only artifacts.

Next rational work:
- Focused artifact diagnostics on exact marginal rows by source/decision and
  final selector score composition for high-marginal under-ranked points.
- If code changes are needed, prefer root fixes to priors, targets, score
  propagation, or selector marginal alignment, not compatibility shims.

## Condensed Checkpoint Index

### Checkpoints 1-3 - Initial Workload, Priors, Model

Status: completed.

Decision:
- Workload generation, train-derived priors, and factorized workload-blind model
  behavior must be evaluated as one contract.
- Full final-grid work is gated behind strict single-cell evidence.

### Checkpoints 4.61-4.82 - Factorized Targets, Priors, Length Repair

Status: completed.

Key decisions:
- Adopted factorized query-useful training direction and learned segment-budget
  selector as the active path.
- Global net-gain length repair improved the candidate boundary.
- Route-density prior was removed from v2 model inputs but retained for support
  diagnostics.
- Direct semantic prior-to-head residual and prior scale experiments did not
  solve causality and were rejected/reverted.

Representative artifacts:
- `artifacts/results/query_driven_v2_checkpoint13_per_head_prior_materiality_strict_replay_c10_r05/example_run.json`
- `artifacts/results/query_driven_v2_checkpoint18_current_best_gate_reclassification_len075/gate_reclassification_summary.json`

### Checkpoints 4.83-5.15 - No-Length-Repair Diagnostics And Pipeline Cleanup

Status: completed.

Key decisions:
- No-length-repair scored higher but failed length preservation, so it remains a
  diagnostic only.
- Orchestration and learning code were split into smaller modules.
- Save-gate and naming cleanup made subsequent probes easier to audit.

Decision:
- Preserve the strict workload-blind protocol. Do not accept length-broken
  variants as candidates.

### Checkpoints 5.16-5.24 - Real-Scale Policy And Selector Length Diagnostics

Status: completed.

Key decisions:
- Real-scale slices are diagnostic only until synthetic/debug strict gates pass.
- Selector length and allocation changes did not remove the causality blocker.
- The active length repair path remains query-free and guarded by global sanity.

### Checkpoints 5.25-5.42 - Prior Materiality And Head Calibration

Status: completed.

Key decisions:
- Prior materiality is weak after the model/selector path: sampled priors move,
  model inputs move less, head probabilities barely move, and retained masks move
  too little.
- Behavior-rank and sparse-head rank/BCE calibration remain diagnostic defaults,
  not accepted target semantics.

Representative artifact:
- `artifacts/results/query_driven_v2_checkpoint19_learning_causality_failure_diagnosis_current_best/learning_causality_failure_diagnosis.json`

### Checkpoints 5.43-5.61 - Repair Frontier, Boundaries, Naming

Status: completed.

Key decisions:
- Exact-pair repair and length-allocation frontier work clarified the
  length/utility tradeoff.
- Code boundaries, final matrix axes, naming, and run-config defaults were
  cleaned up.
- No final-grid run was justified.

### Checkpoint 5.62 - Progress Log Compaction

Status: completed.

Decision:
- Earlier verbose progress was compacted once. This file was still allowed to
  grow again and needed another cleanup at checkpoint 5.99.

### Checkpoints 5.63-5.67 - Length Policy And Shared Constants

Status: completed.

Key decisions:
- Final and validation length preservation policy moved to `0.75`.
- Current-best strict artifact reclassified as global-sanity passing under the
  current policy.
- Duplicate and geometry constants were centralized.

Representative artifact:
- `artifacts/results/query_driven_v2_checkpoint18_current_best_gate_reclassification_len075/gate_reclassification_summary.json`

### Checkpoints 5.68-5.71 - Causality Diagnosis And Prior Schema Cleanup

Status: completed.

Key decisions:
- Learning causality remained the core blocker.
- Prior ablation payload naming was cleaned up; `score_output` is canonical.
- `selector_score` should not be reintroduced as a compatibility alias for the
  prior-ablation payload.

### Checkpoints 5.72-5.76 - Prior Transform Diagnostics

Status: completed / rejected.

Key results:
- `sqrt_probability` prior transform did not improve the candidate; it worsened
  required gates.

Decision:
- Reverted the transform. Do not re-add it without a new targeted hypothesis.

Representative artifacts:
- `artifacts/results/query_driven_v2_checkpoint23_prior_sqrt_transform_standard_strict/`

### Checkpoints 5.77-5.83 - Dense Head Rank Diagnostics

Status: completed / rejected.

Key results:
- Dense-head rank pressure improved some fit diagnostics but worsened retained
  usefulness and causality.

Decision:
- Reverted dense-head rank plumbing. Better head fit alone is not enough.

Representative artifacts:
- `artifacts/results/query_driven_v2_checkpoint27_dense_head_rank_standard_strict/`

### Checkpoint 5.84 - Score/Selector Alignment Derived Diagnosis

Status: completed.

Key result:
- Score movement had weak retained-set marginal value.

Decision:
- Future evidence must tie scores to retained-decision marginal QueryUsefulV1,
  not only factorized-label fit or mask movement.

### Checkpoints 5.85-5.94 - Retained-Marginal Instrumentation And Strict Local Probe

Status: completed.

Key changes:
- Added bounded exact retained-decision marginal QueryUsefulV1 diagnostics.
- Added cached query support so the diagnostic can run at current-best scale.

Key checkpoint38 result:
- MLQDS QueryUsefulV1: `0.1662115143`
- uniform QueryUsefulV1: `0.1421296610`
- Douglas-Peucker QueryUsefulV1: `0.1671038781`
- Gates failed: workload signature, predictability, learning causality.
- Workload signature failed only from a query-count mismatch under the old gate.

Representative artifact:
- `artifacts/results/query_driven_v2_checkpoint38_retained_marginal_payload_current_best_scale_cached/example_run.json`

Decision:
- Do not tune model/selector from checkpoint38 because workload signature failed.
- Diagnose workload query-count semantics first.

### Checkpoint 5.95 - Retained-Mask Freeze Timing Instrumentation

Status: completed.

Changes:
- Added query-free retained-mask freeze timing payloads:
  `retained_mask_freeze_timing` and `retained_mask_ablation_freeze_timing`.

Validation:
- py_compile, ruff, pyright, focused retained-mask tests, broader orchestration
  slice, and `git diff --check` passed.

Decision:
- Use timing payloads in the next strict rerun instead of adding ad hoc timing
  prints.

### Checkpoint 5.96 - Workload Query-Count Stability Generation-Only

Status: failed diagnostic.

Artifact:
- `artifacts/results/query_driven_v2_checkpoint40_workload_query_count_stability_generation_only/workload_query_count_stability_generation_only.json`

Key results:
- Scale: 384 ships, 256 points, 4 route families, balanced split, local 10%
  profile, 48 minimum queries, 256 max queries, 4 train replicates.
- Signature passed `2/5`, failed `3/5`.
- Failure mode: `query_count_mismatch` only.
- Query-count range: `101` to `197`.
- All workloads reached target coverage and stopped with `target_coverage_reached`.

Decision:
- Query count is a coverage-calibrated stopping statistic. Either stabilize the
  profile/generator or revise the signature invariant explicitly.

### Checkpoint 5.97 - Mode-Aware Query-Count Signature Gate

Status: completed.

Artifacts:
- `artifacts/results/query_driven_v2_checkpoint41_query_count_floor_generation_only/query_count_floor_generation_only.json`
- `artifacts/results/query_driven_v2_checkpoint41_mode_aware_signature_generation_only/mode_aware_signature_generation_only.json`

Changes:
- Workload signatures now include profile generation metadata.
- Fixed-count/legacy signatures still enforce query-count relative parity.
- `calibrated_to_coverage` + `profile_sampled_query_count` signatures require
  matching profile semantics and target coverage, but treat relative query-count
  delta as diagnostic-only.

Key results:
- Raising accepted-query floor to `160` or `192` was rejected because it created
  workload-stability failures from rejection pressure and coverage-guard issues.
- Mode-aware checkpoint40-scale generation passed signature and workload
  stability in `5/5` seeds while query counts still ranged `101` to `197`.

Validation:
- py_compile, ruff, pyright, focused orchestration tests, broader retained-mask
  slice, JSON validation, and `git diff --check` passed.

Decision:
- This is a guide-level invariant correction, not a success claim.
- Rerun one strict current-best-scale local cell.

### Checkpoint 5.98 - Mode-Aware Current-Best Strict Local

Status: completed / blocked by gates.

Artifact:
- `artifacts/results/query_driven_v2_checkpoint42_mode_aware_current_best_strict_local/example_run.json`

Key results:
- MLQDS QueryUsefulV1: `0.1662115143`
- uniform QueryUsefulV1: `0.1421296610`
- Douglas-Peucker QueryUsefulV1: `0.1671038781`
- MLQDS RangeUsefulLegacy: `0.1524363397`
- MLQDS length preservation: `0.7915916346`
- `final_success_allowed: false`

Gates:
- Passed: workload stability, support overlap, prior-predictive alignment,
  target diffusion, workload signature, global sanity.
- Failed: predictability, learning causality.

Key diagnostics:
- Workload signature passed under the mode-aware invariant.
- Predictability failed Spearman `0.1109086186 < 0.15` and PR-AUC lift
  `1.2304850435 < 1.25`; lift@5 passed at `1.2035399978`.
- Learning causality failed shuffled scores, shuffled priors, no query priors, no
  behavior head, and no segment-budget head.
- Retained-marginal exact diagnostics: `160` candidates, selector Spearman
  `-0.0077522559`, raw Spearman `-0.0248828079`.
- Timing: total runtime `625.69s`; freeze-retained-masks `363.45s`; query-free
  ablation freeze `260.07s`.

Decision:
- Workload gate is clean enough. Next work is focused prior/head/selector
  marginal alignment diagnosis.
- Do not run the final grid or loosen gates.

### Checkpoint 5.99 - Progress Log Condensation

Status: completed.

Goal:
- Make the progress log usable again without losing current decisions,
  artifacts, or gate state.

Changes:
- Replaced the verbose 4,027-line progress log with this concise current-state
  summary and checkpoint index.
- Kept artifact paths and decisions needed to continue work.
- Removed repeated command detail and raw metrics that already live in artifacts.

Decision:
- Keep future entries short. Add only the hypothesis, artifact path, gate result,
  key numbers, extra discoveries, and decision.

### Checkpoint 5.100 - Prior/Head/Selector Marginal Diagnosis

Status: completed / blocked by gates.

Hypothesis:
- Checkpoint42 already contains enough strict evidence to classify the remaining
  failure without another training run.

Artifact:
- `artifacts/results/query_driven_v2_checkpoint43_prior_head_selector_marginal_diagnosis/prior_head_selector_marginal_diagnosis.json`

Key results:
- Evidence level: `derived_strict_artifact_diagnostic_no_new_probe`.
- No new probe, training run, or grid run.
- Predictability still blocks: aggregate Spearman and PR-AUC lift miss gates.
- Prior-to-head transfer blocks: prior inputs change materially, but head
  probabilities, selector scores, and retained masks barely move.
- Selector marginal alignment blocks: exact retained-decision marginal alignment
  is weak or negative across raw, selector, and segment scores.

Decision:
- Do not run the final grid.
- Next checkpoint should inspect exact marginal rows by source/decision and the
  final score composition for high-marginal under-ranked candidates.
