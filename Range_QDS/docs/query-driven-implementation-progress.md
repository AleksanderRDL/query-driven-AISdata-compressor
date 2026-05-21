# Range_QDS Query-Driven Implementation Progress

This is the short checkpoint log for the query-driven implementation work.
Keep it brief. The protocol, gates, and active defaults live in
[`query-driven-implementation-research-guide.md`](query-driven-implementation-research-guide.md).
The immediate next-step handoff lives in [`Next-Iterations.md`](Next-Iterations.md).

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

Do not claim final success from the current reference artifact. Do not run the
final grid until the smaller required evidence levels pass.

## Entry Format

Append completed checkpoints using this shape:

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

Keep raw command output and detailed metrics in artifacts, not in this log.

## Checkpoint Phase 1 - semantic causality diagnosis

Status: completed.

Hypothesis:
- The current Level 3 reference artifact can localize the failed learning
  causality gates without a replay or production semantic change.
- If row-level fields are incomplete, the next step is focused instrumentation,
  not another tuning sweep.

Changed files:
- `orchestration/diagnostics/semantic_causality_diagnostic.py`
- `tests/unit/orchestration/test_semantic_causality_diagnostic.py`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile orchestration/diagnostics/semantic_causality_diagnostic.py tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check orchestration/diagnostics/semantic_causality_diagnostic.py tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pyright orchestration/diagnostics/semantic_causality_diagnostic.py tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pytest tests/unit/orchestration/test_semantic_causality_diagnostic.py -q`
- `git diff --check`

Artifact:
- `artifacts/results/semantic_causality_diagnosis_current_reference/diagnostic.json`

Scale:
- derived strict artifact diagnostic, no new probe, no replay
- source artifact:
  `artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/example_run.json`

Key results:
- Reference scores unchanged: MLQDS `0.1431090566`, uniform `0.1247681518`,
  Douglas-Peucker `0.1153266238`.
- Gates unchanged: workload stability, support overlap, target diffusion,
  workload signature, predictability, prior-predictive alignment, and global
  sanity pass; learning causality and final success fail.
- Behavior failure: target has signal but the head does not learn it. Behavior
  target std is `0.166493`, prediction std is `0.002631`, Kendall tau is
  `0.0251`, and no-behavior-head loss is `0.001499 < 0.005`.
- Prior failure: sampled/model priors change, but model response is effectively
  immaterial. Shuffled priors move normalized model prior fields by
  `0.011418` mean absolute delta, but head probability movement is
  `0.00000967`, retained-mask Jaccard is `1.0`, and ablation delta is `0.0`.
- Segment failure: allocation scoring and point-selection scoring are mixed
  incorrectly. Raw/selector retained-marginal Spearman are `0.2779`/`0.2881`,
  segment-score Spearman is `-0.0812`, point-score allocation scores
  `0.1451303935` versus primary `0.1431090566`, and removing the segment
  budget head loses `0.009983`.
- Artifact gap: retained decision rows exist, but they lack row-level target
  values, direct QueryLocalUtility component values, anchor/footprint family,
  and query-hit-run ids.

Decision:
- This is diagnostic evidence only; it does not change acceptance state.
- Do not run the final grid, loosen gates, add generic prior scaling, add a
  behavior-rank sweep, or tune selector floors.
- Next admissible step: focused instrumentation for selector/target trace rows
  so the missing semantic fields can be emitted before any root fix or replay.

## Checkpoint Phase 2 - semantic trace row instrumentation

Status: implementation-only.

Hypothesis:
- Missing row-level semantic fields can be added to retained-marginal selector
  trace diagnostics without changing mask freeze, model, target, selector, or
  scoring semantics.

Changed files:
- `orchestration/selector_diagnostics.py`
- `orchestration/diagnostics/semantic_causality_diagnostic.py`
- `tests/unit/selection/test_query_driven_learned_segment_budget.py`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile orchestration/selector_diagnostics.py orchestration/diagnostics/semantic_causality_diagnostic.py tests/unit/selection/test_query_driven_learned_segment_budget.py tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check orchestration/selector_diagnostics.py orchestration/diagnostics/semantic_causality_diagnostic.py tests/unit/selection/test_query_driven_learned_segment_budget.py tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pyright orchestration/selector_diagnostics.py orchestration/diagnostics/semantic_causality_diagnostic.py tests/unit/selection/test_query_driven_learned_segment_budget.py tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pytest tests/unit/selection/test_query_driven_learned_segment_budget.py::test_retained_decision_marginal_query_local_utility_diagnostic_scores_true_marginals tests/unit/orchestration/test_semantic_causality_diagnostic.py -q`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pytest tests/unit/orchestration/test_query_driven_diagnostics.py -q`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pytest tests/unit/orchestration/test_query_driven_causality_and_summary.py -q`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pytest tests/unit/learning/test_query_local_utility_targets.py -q`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pytest tests/guardrails/test_implementation_guardrails.py tests/unit/orchestration/test_retained_mask_stage.py -q`
- `git diff --check`

Artifact:
- none; instrumentation-only, no replay

Scale:
- static/unit/guardrail validation only

Key results:
- Retained-marginal rows now emit diagnostic-only `head_targets`,
  `head_target_masks`, `query_local_utility_target`, direct
  `query_local_utility_component_delta`, primary/candidate component payloads,
  query family hit context, and query-hit-run ids.
- The derived semantic-causality diagnostic now treats those fields as present
  when future artifacts include them, while preserving old-artifact gap
  detection.
- Validation passed: focused tests `2 passed`, query-driven diagnostics
  `7 passed`, causality/summary `21 passed`, target tests `12 passed`, and
  guardrail plus retained-mask-stage tests `26 passed`.

Decision:
- This only improves diagnostic observability. It does not prove learning
  causality and does not update the evidence boundary.
- Next admissible step: run a Level 1 wiring smoke only if a regenerated
  artifact is needed to verify the new row schema, then use the instrumented
  rows to design a root fix for the behavior-head/prior/segment blocker.

## Checkpoint Phase 3 - semantic trace schema smoke

Status: implementation-only.

Hypothesis:
- A tiny current-default run can emit the newly instrumented retained-marginal
  row schema end to end without changing selection or scoring semantics.
- This is wiring evidence only. It is not learning evidence.

Changed files:
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --n_ships 8 --n_points 64 --n_queries 8 --max_queries 8 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode smoke --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --range_train_workload_replicates 1 --query_local_utility_train_marginal_diagnostics --final_metrics_mode diagnostic --results_dir artifacts/results/semantic_trace_schema_level1_smoke`
- `jq '.selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_local_utility_alignment | {available,candidate_count,context_fields_available,first_row:.rows[0]}' artifacts/results/semantic_trace_schema_level1_smoke/example_run.json`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact level1_schema_smoke=artifacts/results/semantic_trace_schema_level1_smoke/example_run.json --output artifacts/results/semantic_trace_schema_level1_smoke/semantic_diagnostic.json`

Artifact:
- `artifacts/results/semantic_trace_schema_level1_smoke/example_run.json`
- `artifacts/results/semantic_trace_schema_level1_smoke/semantic_diagnostic.json`

Scale:
- Level 1 smoke, synthetic, `workload_stability_gate_mode=smoke`
- `n_ships=8`, `n_points=64`, `n_queries=8`, `epochs=1`

Key results:
- Scores are non-evidence wiring numbers: MLQDS `0.0597363805`, uniform
  `0.0693247600`, Douglas-Peucker `0.0283764236`.
- Gates failed as expected for this tiny smoke: workload, support, signature,
  predictability, prior alignment, causality, and final success are false.
- Selection workload had `0` queries, so only eval trace schema was inspected.
- Eval retained-marginal rows are available with `candidate_count=72`.
- Schema proof: context fields are present for `query_local_utility_target`,
  all six `head_targets`, all six `head_target_masks`,
  `query_local_utility_component_delta`, primary/candidate component payloads,
  query family hit context, and query-hit-run ids.
- The derived semantic diagnostic reports no missing required row fields and
  extracts target, component, family, and query-hit-run values from the new row
  schema.

Decision:
- This validates diagnostic row-schema wiring only. It does not change the
  evidence boundary and must not be promoted into learning evidence.
- Do not run a final grid, loosen gates, or tune selector floors from this
  smoke result.
- Next admissible step: static/loss-path diagnosis for the flat behavior head
  and weak prior/segment causal path before any root fix or replay.

## Checkpoint Phase 4 - behavior head loss-path diagnosis

Status: completed.

Hypothesis:
- The behavior head is flat because the current objective gives it mostly
  pointwise mean-matching pressure, while the useful separation/ranking loss is
  present but disabled by default.
- Query-prior and segment failures should be diagnosed as learning-path issues,
  not patched with selector-floor or temporal-scaffold tuning.

Changed files:
- `docs/query-driven-implementation-progress.md`

Validation commands:
- static/code inspection only
- `jq '.. | objects | select(has("query_local_utility_loss_weights")) | .query_local_utility_loss_weights' artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/example_run.json`
- `jq '.. | objects | select(has("factorized_head_fit")) | .factorized_head_fit.conditional_behavior_utility' artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/example_run.json`
- `jq '.selector_trace_diagnostics.eval_primary | {segment_score_min,segment_score_max,segment_score_mean,segment_score_std,segment_score_span,segment_score_source,segment_length_support_weight,segment_score_point_blend_weight,segment_allocation_weight_floor}' artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/example_run.json`

Artifact:
- none; derived from current reference artifact

Scale:
- static/code inspection plus current Level 3 reference diagnostics
- source artifact:
  `artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/example_run.json`

Key results:
- Behavior target is not the weak link: target std is `0.166493`, target-to-final
  Spearman is `0.305617`, and top-5% behavior-ranked points recover `0.782388`
  of final target mass.
- Trained behavior head is flat: prediction std is `0.002631`, Kendall tau is
  `0.025098`, and top-5% mass recall is `0.129381`.
- Active loss weights confirm the asymmetry: `behavior_rank_loss_weight=0.0`,
  `sparse_head_rank_loss_weight=0.0`, while segment budget has
  `segment_level_loss_weight=0.25`.
- Code path confirms the same: `_behavior_head_rank_loss` exists but is only
  used when the configured weight is positive; the segment-budget head always
  receives the extra segment-level/listwise loss under current defaults.
- Prior fields are sampled and normalized, but the model-facing path is only an
  additive shared encoder input. Current ablations show priors move inputs but
  not heads, scores, or masks, so this is model prior-use failure rather than
  prior support failure.
- Segment allocation is over-trusting a weak segment head: eval segment logits
  have span `0.035616` and std `0.014342`, but allocation min-max normalizes
  that tiny spread. Segment-score-to-allocation Spearman is `0.838571`, while
  segment-score retained-marginal Spearman in the reference diagnostic is
  `-0.0812`.

Decision:
- Root classification: behavior target has useful signal but the default
  learning objective does not make the behavior head rank it.
- Root classification: prior materiality failure is model-use failure after
  sampled/model prior inputs, not prior construction failure.
- Root classification: segment failure is a weak/flat segment-head signal being
  amplified by allocation, not proof that selector-floor tuning is admissible.
- Do not run a final grid, selector allocation-floor probe, behavior-rank-only
  sweep, or prior-scale-only sweep.
- Next admissible step: a focused head-learning fix that adds semantic
  separation pressure for the behavior head, followed by Level 1 wiring and then
  Level 2 strict gate localization before any Level 3 replay.

## Checkpoint Phase 5 - behavior rank default wiring

Status: implementation-only.

Hypothesis:
- A fixed nonzero behavior-head rank loss default can address the diagnosed
  flat behavior head without changing metric, workload profile, target
  semantics, selector semantics, or temporal scaffolding.
- Explicit `0.0` must still disable the behavior-rank term for ablation and
  debugging.

Changed files:
- `config/run_config.py`
- `learning/optimization_epoch.py`
- `orchestration/learning_scoring_cli.py`
- `tests/unit/learning/test_query_local_utility_training.py`
- `tests/unit/runtime/test_torch_runtime_controls.py`
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile config/run_config.py orchestration/learning_scoring_cli.py learning/optimization_epoch.py tests/unit/learning/test_query_local_utility_training.py tests/unit/runtime/test_torch_runtime_controls.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check config/run_config.py orchestration/learning_scoring_cli.py learning/optimization_epoch.py orchestration/selector_diagnostics.py orchestration/diagnostics/semantic_causality_diagnostic.py tests/unit/learning/test_query_local_utility_training.py tests/unit/runtime/test_torch_runtime_controls.py tests/unit/selection/test_query_driven_learned_segment_budget.py tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pyright config/run_config.py orchestration/learning_scoring_cli.py learning/optimization_epoch.py orchestration/selector_diagnostics.py orchestration/diagnostics/semantic_causality_diagnostic.py tests/unit/learning/test_query_local_utility_training.py tests/unit/runtime/test_torch_runtime_controls.py tests/unit/selection/test_query_driven_learned_segment_budget.py tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pytest tests/unit/orchestration/test_semantic_causality_diagnostic.py tests/unit/selection/test_query_driven_learned_segment_budget.py::test_retained_decision_marginal_query_local_utility_diagnostic_scores_true_marginals tests/unit/learning/test_query_local_utility_training.py tests/unit/runtime/test_torch_runtime_controls.py tests/unit/learning/test_model_learning_does_not_collapse.py::test_validation_selection_passes_segment_head_to_learned_selector tests/unit/learning/test_model_learning_does_not_collapse.py::test_validation_checkpoint_scores_report_factorized_causality_deltas -q`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --n_ships 8 --n_points 64 --n_queries 8 --max_queries 8 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode smoke --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --range_train_workload_replicates 1 --query_local_utility_train_marginal_diagnostics --final_metrics_mode diagnostic --results_dir artifacts/results/behavior_rank_default_level1_smoke`
- `git diff --check`

Artifact:
- `artifacts/results/behavior_rank_default_level1_smoke/example_run.json`

Scale:
- static/unit plus Level 1 smoke, synthetic, `workload_stability_gate_mode=smoke`
- `n_ships=8`, `n_points=64`, `n_queries=8`, `epochs=1`

Key results:
- Default behavior-rank loss is now `0.25` in config, CLI parser, training
  fallback, and emitted `query_local_utility_loss_weights`.
- Explicit zero remains supported and unit-tested through the loss function.
- Static validation passed: `py_compile`, `ruff`, `pyright`, and `git diff
  --check`.
- Unit validation passed: touched unit subset `37 passed`.
- Level 1 smoke scores are non-evidence wiring numbers: MLQDS `0.0271057949`,
  uniform `0.0693247600`, Douglas-Peucker `0.0283764236`.
- Level 1 smoke failed required gates as expected at this scale: workload,
  support, predictability, prior alignment, workload signature, learning
  causality, and final success are false.
- Smoke behavior head fit is not promotion evidence: prediction std
  `0.002824`, Kendall tau `0.300885`, top-5% mass recall `0.458790` over only
  `33` valid points.

Decision:
- This is a focused implementation change plus wiring proof only.
- It does not update the evidence boundary, does not prove learning causality,
  and must not be compared as a successful variant.
- Do not run the final grid or a Level 3 replay from this smoke.
- Next admissible step: Level 2 minimum strict gate localization with the new
  default. If behavior improves but prior ablations remain immaterial, the next
  fix must target prior feature integration, not behavior loss or selector
  allocation.

## Checkpoint Phase 6 - behavior rank default Level 2 strict

Status: completed.

Hypothesis:
- The behavior-rank default should improve behavior-head separation or
  no-behavior-head sensitivity at strict small scale before any Level 3 replay.
- If strict pre-causality gates fail, stop at that gate and do not tune around
  the failure.

Changed files:
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2532 --n_ships 32 --n_points 192 --synthetic_route_families 3 --n_queries 24 --max_queries 192 --epochs 4 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 30000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/query_driven_behavior_rank_default_level2_seed2532`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact behavior_rank_default_level2=artifacts/results/query_driven_behavior_rank_default_level2_seed2532/example_run.json --output artifacts/results/query_driven_behavior_rank_default_level2_seed2532/semantic_diagnostic.json`

Artifact:
- `artifacts/results/query_driven_behavior_rank_default_level2_seed2532/example_run.json`
- `artifacts/results/query_driven_behavior_rank_default_level2_seed2532/semantic_diagnostic.json`

Scale:
- Level 2 minimum strict, synthetic, source-stratified, seed `2532`
- `n_ships=32`, `n_points=192`, `n_queries=24`, `max_queries=192`,
  `epochs=4`, `range_train_workload_replicates=4`

Key results:
- Scores: MLQDS `0.1026929755`, uniform `0.1003283552`,
  Douglas-Peucker `0.1121031831`.
- Gate state: support overlap, predictability, prior-predictive alignment, and
  target diffusion pass; workload stability, workload signature, learning
  causality, global sanity, and final success fail.
- The run is blocked before promotion by workload stability and workload
  signature, so it cannot define a new evidence boundary.
- Behavior head moved in the expected direction but not enough. Versus the old
  same-seed Level 2 reference, behavior prediction std rose from `0.003902` to
  `0.006967`, top-5% mass recall rose from `0.127256` to `0.253796`, and
  no-behavior-head delta rose from `-0.002388` to `0.002179`.
- Behavior causality still fails: no-behavior-head delta is `0.002179 < 0.005`
  and behavior Kendall tau remains `0.005177`.
- Prior causality still fails: shuffled-prior delta is `-0.0000419` and
  no-query-prior delta is `-0.0000419`.
- Segment causality worsened: no-segment-budget-head delta is `-0.014099`,
  meaning the segment-budget head is harmful in this Level 2 cell.
- Semantic diagnostic reports complete row fields and classifies behavior as
  `target has signal but head does not learn it`; prior and segment are only
  partial classifications at this scale.

Decision:
- The behavior-rank default is only a partial behavior-head improvement. It is
  not sufficient and must not be tuned further as a rank-weight sweep.
- Do not run Level 3 or the final grid from this result.
- Next admissible step: diagnose/fix prior feature integration and segment-head
  allocation semantics. Do not use selector allocation-floor changes,
  length-support weighting, or stronger behavior-rank weight as substitutes for
  causal prior/head learning.

## Checkpoint Phase 7 - semantic prior-head adapter wiring

Status: implementation-only.

Hypothesis:
- Query priors are sampled and present, but the current model can ignore them
  because they only enter through a shared encoder path. A sparse semantic
  prior-to-head adapter should make train-derived priors observable at the
  factorized heads without changing selector rules, target semantics, coverage,
  or gates.

Changed files:
- `models/workload_blind_range.py`
- `learning/checkpoints.py`
- `tests/unit/learning/test_model_features.py`
- `tests/unit/orchestration/test_query_driven_protocol_gates.py`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/models/workload_blind_range.py Range_QDS/learning/checkpoints.py Range_QDS/tests/unit/learning/test_model_features.py Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/models/workload_blind_range.py Range_QDS/learning/checkpoints.py Range_QDS/tests/unit/learning/test_model_features.py Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pyright models/workload_blind_range.py learning/checkpoints.py tests/unit/learning/test_model_features.py tests/unit/orchestration/test_query_driven_protocol_gates.py)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest Range_QDS/tests/unit/learning/test_model_features.py::test_workload_blind_range_checkpoint_accepts_missing_prior_feature_modules Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py::test_workload_blind_range_has_dedicated_prior_feature_encoder Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py::test_workload_blind_range_prior_head_adapter_is_semantic_and_direct Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py::test_workload_blind_range_untrained_reset_restores_standalone_parameters`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2535 --n_ships 12 --n_points 96 --synthetic_route_families 3 --n_queries 8 --max_queries 64 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 12000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 2 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/prior_head_adapter_level1_smoke)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/prior_head_adapter_level1_smoke/example_run.json --output artifacts/results/prior_head_adapter_level1_smoke/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/prior_head_adapter_level1_smoke/example_run.json`
- `artifacts/results/prior_head_adapter_level1_smoke/semantic_diagnostic.json`

Scale:
- static/unit plus Level 1 wiring smoke, synthetic, source-stratified
- `n_ships=12`, `n_points=96`, `n_queries=8`, `max_queries=64`,
  `epochs=1`, `range_train_workload_replicates=2`

Key results:
- Model schema bumped from `6` to `7`.
- Added a trainable sparse prior-head adapter initialized from the documented
  semantic prior channels. Route-density remains disabled for the active model
  input and has zero direct head-adapter effect.
- Older `workload_blind_range` checkpoints missing `prior_head_adapter.*` still
  load under the existing compatibility path.
- Focused unit/static checks passed: `py_compile`, `ruff`, package-root
  `pyright`, and `4` focused pytest tests.
- Smoke scores are non-evidence wiring numbers: MLQDS `0.0834811181`, uniform
  `0.0499761776`, Douglas-Peucker `0.0923826838`.
- Smoke gates are not promotable: support overlap and target diffusion pass;
  workload stability, predictability, prior-predictive alignment, workload
  signature, learning causality, global sanity, and final success fail.
- Prior materiality moved in the intended implementation direction:
  shuffled-prior and no-query-prior deltas are both `0.0328155699`, retained
  masks change with Jaccard `0.578947`, model-prior mean absolute delta is
  `0.0269458`, and head probability delta is `0.0017777`.
- Learning causality still fails. Failed checks: shuffled scores,
  no-behavior-head, no-segment-budget-head, and prior-field-only separation.
- The prior-only failure is a warning: prior-field-only delta is only
  `0.0021987`, so the smoke does not prove learned use beyond priors.
- Behavior remains the hard blocker: no-behavior-head delta is `-0.0131589`;
  semantic diagnostic still classifies it as `target has signal but head does
  not learn it`. Behavior retained-marginal Spearman is `-0.206159`, with head
  probability range about `0.0762` to `0.0869`.
- Segment is not wrong-way in this smoke, but it is still not material:
  no-segment-budget-head delta is `-0.0003186`; segment-score retained-marginal
  Spearman is `0.0500`.

Decision:
- The prior integration root fix passes wiring checks and fixes the previous
  zero-materiality prior-path symptom at Level 1 scale only.
- This does not update the evidence boundary, because it is a smoke and still
  fails required causality and health gates.
- Phase 12 refuted this path at strict Level 2, and Phase 13 reverts it.
- Stop treating the semantic prior-head adapter as an accepted production fix.

## Checkpoint Phase 8 - behavior rank normalized-gap wiring

Status: implementation-only.

Hypothesis:
- The behavior target has rank signal, but the behavior rank loss was using
  absolute target gaps. That underweights rows whose behavior target scale is
  small even when their ordering is meaningful. Normalize behavior target gaps
  within each valid window while keeping the configured rank-loss weight fixed.

Changed files:
- `learning/optimization_epoch.py`
- `tests/unit/learning/test_query_local_utility_training.py`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/learning/optimization_epoch.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/learning/optimization_epoch.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pyright learning/optimization_epoch.py tests/unit/learning/test_query_local_utility_training.py)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_behavior_head_rank_loss_penalizes_reversed_behavior_order Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_behavior_head_rank_loss_normalizes_target_gap_scale`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2536 --n_ships 12 --n_points 96 --synthetic_route_families 3 --n_queries 8 --max_queries 64 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 12000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 2 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/behavior_rank_gap_normalized_level1_smoke)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/behavior_rank_gap_normalized_level1_smoke/example_run.json --output artifacts/results/behavior_rank_gap_normalized_level1_smoke/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/behavior_rank_gap_normalized_level1_smoke/example_run.json`
- `artifacts/results/behavior_rank_gap_normalized_level1_smoke/semantic_diagnostic.json`

Scale:
- static/unit plus Level 1 wiring smoke, synthetic, source-stratified
- `n_ships=12`, `n_points=96`, `n_queries=8`, `max_queries=64`,
  `epochs=1`, `range_train_workload_replicates=2`

Key results:
- The behavior rank loss now uses target-span-normalized pair gaps and treats
  `min_target_gap` as a normalized row-span threshold.
- Unit test verifies that a tiny-scale behavior target produces the same rank
  loss as the full-scale target and has nonzero behavior-logit gradients.
- Focused unit/static checks passed: `py_compile`, `ruff`, package-root
  `pyright`, and `2` focused pytest tests.
- Smoke scores are non-evidence wiring numbers and are weak: MLQDS
  `0.0560586393`, uniform `0.0698253996`, Douglas-Peucker `0.1021889989`.
- Smoke gates are not promotable: support overlap and target diffusion pass;
  workload stability, predictability, prior-predictive alignment, workload
  signature, learning causality, global sanity, and final success fail.
- Behavior fit moved in the expected local direction at smoke scale:
  behavior prediction std `0.004368`, Kendall tau `0.070610`, top-5% mass
  recall `0.265862`.
- Behavior retained-marginal alignment is no longer wrong-way in this smoke:
  behavior-head Spearman `0.066042` and top-minus-bottom marginal
  `0.002109`.
- No-behavior-head causality is still absent: no-behavior delta is
  `-0.0000575`.
- Prior and score ablations moved the wrong way in this smoke:
  shuffled-score delta `-0.061978`, shuffled-prior/no-prior deltas
  `-0.030335`, prior-field-only delta `-0.038738`. This is not admissible
  promotion evidence.
- Semantic diagnostic classifies the current failure as `final score has signal
  but selector/segment allocation loses it`. Segment is not wrong-way, but the
  point-score allocation diagnostic is better than primary.

Decision:
- This was kept only long enough for the next strict Level 2 boundary check.
  Phase 10 contradicted it at strict scale, and Phase 11 reverts it.
- Do not run Level 2, Level 3, or final grid from this result. The smoke fails
  too many child gates and underperforms baselines.
- Next admissible checkpoint: diagnose score-to-selector/segment allocation
  transfer under the current stack. This must be diagnostic-first; do not tune
  allocation floors or length-support weights.

## Checkpoint Phase 9 - segment allocation transfer diagnosis

Status: completed / diagnostic-only.

Hypothesis:
- After behavior ordering improves, remaining loss may be score-to-selector
  transfer: the segment allocation path may overuse weak segment-budget scores
  when point-score-derived segment pooling is better.
- Before changing selector semantics, compare existing diagnostic allocation
  variants across the strict reference and current small artifacts.

Changed files:
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `jq '.artifacts[0].segment_score_calibration.diagnostic_segment_rankers, .artifacts[0].segment_score_calibration.retained_marginal_alignment' artifacts/results/semantic_causality_diagnosis_current_reference/diagnostic.json`
- `jq '.artifacts[0].segment_score_calibration.diagnostic_segment_rankers' artifacts/results/query_driven_behavior_rank_default_level2_seed2532/semantic_diagnostic.json`
- `jq '.artifacts[0].segment_score_calibration.diagnostic_segment_rankers' artifacts/results/prior_head_adapter_level1_smoke/semantic_diagnostic.json`
- `jq '.artifacts[0].segment_score_calibration.diagnostic_segment_rankers' artifacts/results/behavior_rank_gap_normalized_level1_smoke/semantic_diagnostic.json`

Artifact:
- Reused `artifacts/results/semantic_causality_diagnosis_current_reference/diagnostic.json`
- Reused
  `artifacts/results/query_driven_behavior_rank_default_level2_seed2532/semantic_diagnostic.json`
- Reused `artifacts/results/prior_head_adapter_level1_smoke/semantic_diagnostic.json`
- Reused
  `artifacts/results/behavior_rank_gap_normalized_level1_smoke/semantic_diagnostic.json`

Scale:
- derived artifact diagnostic only
- no new training run

Key results:
- Current strict Level 3 reference:
  - primary `0.1431090566`
  - pooled-point allocation diagnostic `0.1451303935`
  - pooled minus primary `+0.0020213`
  - segment-score retained-marginal Spearman `-0.081151`
  - raw-score Spearman `0.277877`
  - selector-score Spearman `0.288138`
  - without-segment-budget minus primary `-0.0099826`
- Behavior-rank-default Level 2:
  - primary `0.1026929755`
  - pooled-point allocation diagnostic `0.1026929755`
  - pooled minus primary `0.0`
  - neutral/no-segment-budget diagnostic `0.1167917178`
  - without-segment-budget minus primary `+0.0140987`
- Prior-head-adapter Level 1 smoke:
  - primary `0.0834811181`
  - pooled-point allocation diagnostic `0.0756512320`
  - pooled minus primary `-0.0078299`
  - neutral segment diagnostic `0.0837997679`
  - without-segment-budget minus primary `+0.0003186`
- Behavior-gap-normalized Level 1 smoke:
  - primary `0.0560586393`
  - pooled-point allocation diagnostic `0.0859600735`
  - pooled minus primary `+0.0299014`
  - neutral segment diagnostic `0.0885803179`
  - without-segment-budget minus primary `+0.0325217`
- Diagnosis: point-score-derived segment pooling is not consistently better
  across artifacts. Neutral/no-segment allocation often beats primary in small
  artifacts, but the strict Level 3 reference still benefits from segment
  budgeting despite wrong-way segment retained-marginal alignment.

Decision:
- Do not replace segment allocation with pooled point-score segments yet.
- Do not tune allocation floor, length-support weight, or blend weight from
  this mixed diagnostic.
- The next admissible checkpoint is a strict Level 2 run of the current
  production stack to establish the new evidence boundary after the prior and
  behavior-loss changes. If that Level 2 fails workload health/signature, stop
  there; if it passes health but fails segment causality, then design a
  segment-target/allocation root fix from that strict artifact.

## Checkpoint Phase 10 - normalized-gap stack Level 2 strict boundary

Status: completed / rejected.

Hypothesis:
- The transient current stack with semantic prior-head adapter plus normalized
  behavior-rank gaps might improve prior materiality and behavior ordering at
  strict Level 2 scale before any Level 3 replay.
- If workload health/signature or child causality fails, stop and do not tune.

Changed files:
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2537 --n_ships 32 --n_points 192 --synthetic_route_families 3 --n_queries 24 --max_queries 192 --epochs 4 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 30000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/query_driven_current_stack_level2_seed2537)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/query_driven_current_stack_level2_seed2537/example_run.json --output artifacts/results/query_driven_current_stack_level2_seed2537/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/query_driven_current_stack_level2_seed2537/example_run.json`
- `artifacts/results/query_driven_current_stack_level2_seed2537/semantic_diagnostic.json`

Scale:
- Level 2 minimum strict, synthetic, source-stratified, seed `2537`
- `n_ships=32`, `n_points=192`, `n_queries=24`, `max_queries=192`,
  `epochs=4`, `range_train_workload_replicates=4`

Key results:
- Scores: MLQDS `0.1109622965`, uniform `0.1014750196`,
  Douglas-Peucker `0.1126127687`.
- The run beats uniform by `+0.0094873` but does not beat Douglas-Peucker.
- Workload stability, support overlap, and target diffusion pass.
- Blocking gates: predictability, prior-predictive alignment, workload
  signature, learning causality, global sanity, and final success.
- Learning causality fails:
  - shuffled-score delta `0.0102736`
  - shuffled-prior delta `0.0033548 < 0.005`
  - no-query-prior delta `0.0033795 < 0.005`
  - no-behavior-head delta `-0.0000801`
  - no-segment-budget-head delta `-0.0198025`
  - prior-field-only delta `0.0167703`
- Behavior is still flat and wrong-way:
  - prediction std `0.002153`
  - Kendall tau `-0.025622`
  - top-5% mass recall `0.103087`
  - retained-marginal Spearman `-0.213705`
- Segment is a hard failure in this strict artifact:
  - semantic diagnostic classifies `segment head fails to learn target`
  - segment-score retained-marginal Spearman `-0.073701`
  - pooled-point allocation diagnostic `0.1179660`, primary `0.1109623`
  - neutral/no-segment diagnostic `0.1307648`
  - no-segment-budget minus primary `+0.0198025`
- Prior materiality improved relative to exact-zero failures but is below gate.

Decision:
- Reject the normalized-gap behavior-rank change. It did not improve strict
  behavior learning and coincided with a flat/wrong-way behavior head.
- Do not run Level 3 or final grid.
- Keep the semantic prior-head adapter as an implementation candidate, but it
  still needs strict Level 2 evidence after reverting normalized-gap behavior
  loss.

## Checkpoint Phase 11 - revert normalized-gap behavior rank loss

Status: completed / cleanup.

Hypothesis:
- The normalized-gap behavior-rank change failed strict Level 2 evidence and
  should not remain in the production training path as an unproven experiment.

Changed files:
- `learning/optimization_epoch.py`
- `tests/unit/learning/test_query_local_utility_training.py`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/models/workload_blind_range.py Range_QDS/learning/checkpoints.py Range_QDS/learning/optimization_epoch.py Range_QDS/tests/unit/learning/test_model_features.py Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/models/workload_blind_range.py Range_QDS/learning/checkpoints.py Range_QDS/learning/optimization_epoch.py Range_QDS/tests/unit/learning/test_model_features.py Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pyright models/workload_blind_range.py learning/checkpoints.py learning/optimization_epoch.py tests/unit/learning/test_model_features.py tests/unit/orchestration/test_query_driven_protocol_gates.py tests/unit/learning/test_query_local_utility_training.py)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest Range_QDS/tests/unit/learning/test_model_features.py::test_workload_blind_range_checkpoint_accepts_missing_prior_feature_modules Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py::test_workload_blind_range_has_dedicated_prior_feature_encoder Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py::test_workload_blind_range_prior_head_adapter_is_semantic_and_direct Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py::test_workload_blind_range_untrained_reset_restores_standalone_parameters Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_behavior_head_rank_loss_penalizes_reversed_behavior_order`
- `git diff --check`

Artifact:
- none

Scale:
- code cleanup only

Key results:
- Restored `_behavior_head_rank_loss` to raw target-gap weighting and absolute
  `min_target_gap` semantics.
- Removed the normalized-gap-specific unit test.
- The strict Level 2 artifact from Phase 10 remains diagnostic evidence for the
  rejected transient stack, not for the cleaned production stack.
- Cleanup validation passed: `py_compile`, `ruff`, package-root `pyright`,
  `5` focused pytest tests, and `git diff --check`.

Decision:
- Normalized-gap behavior rank is rejected for now.
- Next admissible checkpoint is a strict Level 2 run for the cleaned current
  production stack: semantic prior-head adapter plus the existing behavior-rank
  default, with no selector changes and no final grid.

## Checkpoint Phase 12 - prior-adapter stack Level 2 strict boundary

Status: completed / rejected.

Hypothesis:
- After reverting normalized-gap behavior loss, the cleaned stack with only the
  semantic prior-head adapter should be tested at strict Level 2 before any
  further claim.
- If it fails baselines or child causality, reject the adapter rather than
  tuning scale or selector parameters.

Changed files:
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2538 --n_ships 32 --n_points 192 --synthetic_route_families 3 --n_queries 24 --max_queries 192 --epochs 4 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 30000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/query_driven_prior_adapter_level2_seed2538)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/query_driven_prior_adapter_level2_seed2538/example_run.json --output artifacts/results/query_driven_prior_adapter_level2_seed2538/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/query_driven_prior_adapter_level2_seed2538/example_run.json`
- `artifacts/results/query_driven_prior_adapter_level2_seed2538/semantic_diagnostic.json`

Scale:
- Level 2 minimum strict, synthetic, source-stratified, seed `2538`
- `n_ships=32`, `n_points=192`, `n_queries=24`, `max_queries=192`,
  `epochs=4`, `range_train_workload_replicates=4`

Key results:
- Scores: MLQDS `0.0855909368`, uniform `0.1012227315`,
  Douglas-Peucker `0.0997721666`.
- The run loses to both uniform and Douglas-Peucker.
- Workload stability, support overlap, prior-predictive alignment, and target
  diffusion pass.
- Blocking gates: predictability, workload signature, learning causality,
  global sanity, and final success.
- Learning causality fails:
  - shuffled-score delta `-0.0138287`
  - shuffled-prior delta `-0.0107175`
  - no-query-prior delta `-0.0085687`
  - no-behavior-head delta `0.0120694`
  - no-segment-budget-head delta `-0.0016661`
  - prior-field-only delta `0.0012139`
- Behavior head is still flat but locally more aligned than prior attempts:
  prediction std `0.003534`, Kendall tau `0.069139`, top-5% mass recall
  `0.259626`, retained-marginal Spearman `0.086958`.
- Segment allocation still looks like a transfer problem:
  pooled-point allocation diagnostic `0.0968839`, primary `0.0855909`,
  neutral/no-segment diagnostic `0.0872570`.
- Prior materiality is not fixed. Prior ablations are wrong-way and
  prior-field-only nearly matches trained.

Decision:
- Reject the semantic prior-head adapter. It is not just unproven; at strict
  Level 2 it is worse than baselines and fails prior causality.
- Do not tune adapter scale, run another Level 2 seed, run Level 3, or run the
  final grid from this path.
- Revert the adapter from production code and preserve only the diagnostic
  artifacts/log entries.

## Checkpoint Phase 13 - revert semantic prior-head adapter

Status: completed / cleanup.

Hypothesis:
- The semantic prior-head adapter failed strict Level 2 evidence and should not
  remain in production code as an experiment.

Changed files:
- `models/workload_blind_range.py`
- `learning/checkpoints.py`
- `tests/unit/learning/test_model_features.py`
- `tests/unit/orchestration/test_query_driven_protocol_gates.py`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/models/workload_blind_range.py Range_QDS/learning/checkpoints.py Range_QDS/learning/optimization_epoch.py Range_QDS/tests/unit/learning/test_model_features.py Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/models/workload_blind_range.py Range_QDS/learning/checkpoints.py Range_QDS/learning/optimization_epoch.py Range_QDS/tests/unit/learning/test_model_features.py Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pyright models/workload_blind_range.py learning/checkpoints.py learning/optimization_epoch.py tests/unit/learning/test_model_features.py tests/unit/orchestration/test_query_driven_protocol_gates.py tests/unit/learning/test_query_local_utility_training.py)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest Range_QDS/tests/unit/learning/test_model_features.py::test_workload_blind_range_checkpoint_accepts_missing_prior_feature_encoder Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py::test_workload_blind_range_has_dedicated_prior_feature_encoder Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py::test_workload_blind_range_untrained_reset_restores_standalone_parameters Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_behavior_head_rank_loss_penalizes_reversed_behavior_order`
- `git diff --check`

Artifact:
- none

Scale:
- code cleanup only

Key results:
- Restored `WORKLOAD_BLIND_RANGE_SCHEMA_VERSION` to `6`.
- Removed the semantic prior-head adapter module and direct head-logit prior
  addition.
- Removed `prior_head_adapter.*` checkpoint compatibility allowance.
- Restored checkpoint compatibility and reset tests to the prior-feature
  encoder path only.
- The Phase 12 strict Level 2 artifact remains diagnostic evidence for the
  rejected adapter stack, not for production.
- Cleanup validation passed: `py_compile`, `ruff`, package-root `pyright`,
  `4` focused pytest tests, and `git diff --check`.

Decision:
- The production stack is back to the previously accepted implementation state:
  behavior-rank default remains, semantic trace instrumentation remains, and
  the failed normalized-gap/adaptor experiments are removed.
- Next admissible step is not more tuning. The blocker is unresolved:
  behavior head learning and segment transfer still need a new root hypothesis.

## Checkpoint Phase 14 - behavior-head training-signal diagnostic

Status: completed / diagnostic-only.

Hypothesis:
- The behavior head may stay flat because the active loss leaves it close to
  the empirical target-mean output bias. We need direct evidence comparing
  trained behavior logits against a constant-bias baseline before changing
  behavior target, model architecture, or selector semantics.

Changed files:
- `learning/factorized_head_diagnostics.py`
- `learning/model_training.py`
- `tests/unit/learning/test_query_local_utility_training.py`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/learning/model_training.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/learning/model_training.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pyright learning/factorized_head_diagnostics.py learning/model_training.py tests/unit/learning/test_query_local_utility_training.py)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_behavior_head_training_signal_diagnostic_compares_rank_to_bias_baseline Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_behavior_head_rank_loss_penalizes_reversed_behavior_order`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2540 --n_ships 12 --n_points 96 --synthetic_route_families 3 --n_queries 8 --max_queries 64 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 12000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 2 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/behavior_signal_diagnostic_level1_smoke)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/behavior_signal_diagnostic_level1_smoke/example_run.json --output artifacts/results/behavior_signal_diagnostic_level1_smoke/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/behavior_signal_diagnostic_level1_smoke/example_run.json`
- `artifacts/results/behavior_signal_diagnostic_level1_smoke/semantic_diagnostic.json`

Scale:
- static/unit plus Level 1 wiring smoke, synthetic, source-stratified, seed
  `2540`
- `n_ships=12`, `n_points=96`, `n_queries=8`, `max_queries=64`,
  `epochs=1`, `range_train_workload_replicates=2`

Key results:
- Added `training_fit_diagnostics.behavior_head_training_signal`, diagnostic
  only.
- The diagnostic reports behavior BCE and rank loss against a constant
  empirical target-mean bias baseline, plus rank-pair counts, rank-pair
  accuracy, target-gap scale, and prediction-to-target std ratio.
- Focused validation passed: `py_compile`, `ruff`, package-root `pyright`, and
  `2` focused pytest tests.
- Smoke is not promotion evidence. Scores: MLQDS `0.0657499369`, uniform
  `0.0361752717`, Douglas-Peucker `0.1162665291`.
- Smoke gates are not promotable: workload stability, predictability,
  prior-predictive alignment, workload signature, learning causality, global
  sanity, and final success fail.
- Behavior diagnostic is informative:
  - valid behavior points `293`
  - positive behavior targets `97`
  - target std `0.239934`
  - prediction std `0.003462`
  - prediction std / target std `0.01443`
  - rank rows with pairs `6/6`
  - rank pair count `890`
  - rank pair accuracy `0.7461`
  - rank loss improvement over bias `0.0078255`
  - BCE improvement over bias `0.0011164`
  - weighted behavior-rank loss / behavior BCE ratio `0.2712`
  - classification `rank_pressure_improves_but_prediction_still_flat`
- Semantic diagnostic classifies the smoke behavior failure as `head learns
  weak signal but final score suppresses it`; query-prior failure remains
  `model ignores prior inputs`.

Decision:
- This checkpoint adds useful instrumentation and does not change model,
  target, selector, or metric semantics.
- Do not promote the smoke or run Level 2/3 from it.
- Next admissible step: use this diagnostic in a strict Level 2 replay of the
  current production stack, or design a root fix that increases behavior-head
  decision leverage without target rewrites, selector-floor tuning, or generic
  loss-weight sweeps.

## Checkpoint Phase 15 - behavior-signal diagnostic Level 2 strict

Status: completed / accepted-as-diagnostic-boundary.

Hypothesis:
- At strict Level 2 scale, the behavior head may have enough ordering signal to
  pass the no-behavior ablation, while still exposing low probability variance.
- If prior ablations remain immaterial, the next blocker is query-prior
  materiality, not another behavior-rank tweak.

Changed files:
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2539 --n_ships 32 --n_points 192 --synthetic_route_families 3 --n_queries 24 --max_queries 192 --epochs 4 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 30000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/query_driven_behavior_signal_level2_seed2539)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/query_driven_behavior_signal_level2_seed2539/example_run.json --output artifacts/results/query_driven_behavior_signal_level2_seed2539/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/query_driven_behavior_signal_level2_seed2539/example_run.json`
- `artifacts/results/query_driven_behavior_signal_level2_seed2539/semantic_diagnostic.json`

Scale:
- Level 2 minimum strict, synthetic, source-stratified, seed `2539`
- `n_ships=32`, `n_points=192`, `n_queries=24`, `max_queries=192`,
  `epochs=4`, `range_train_workload_replicates=4`

Key results:
- Scores: MLQDS `0.1090154720`, uniform `0.0992909061`,
  Douglas-Peucker `0.1182249577`.
- The run beats uniform but loses to Douglas-Peucker.
- Workload stability, support overlap, predictability, prior-predictive
  alignment, and target diffusion pass.
- Blocking gates: workload signature, learning causality, global sanity, and
  final success.
- Learning causality fails only on query priors:
  - shuffled-score delta `0.0258082`
  - shuffled-prior delta `0.0`
  - no-query-prior delta `0.0`
  - no-behavior-head delta `0.0100696`
  - no-segment-budget-head delta `0.0232512`
  - prior-field-only delta `0.0112428`
- Behavior training-signal diagnostic:
  - valid behavior points `2846`
  - target std `0.172177`
  - prediction std `0.004780`
  - prediction std / target std `0.02776`
  - rank rows with pairs `21/21`
  - rank pair count `20668`
  - rank pair accuracy `0.634459`
  - rank loss improvement over bias `0.0118484`
  - BCE improvement over bias `0.0003244`
  - weighted behavior-rank loss / behavior BCE ratio `0.3002`
  - classification `rank_pressure_improves_but_prediction_still_flat`
- Behavior retained-marginal Spearman is weak positive at `0.043445`.
- Semantic diagnostic classifies query-prior failure as `model ignores prior
  inputs`. Shuffled/no-prior changes move model inputs but not retained masks.
- Segment remains suspicious but not the front-door child-gate failure in this
  run: no-segment-budget delta passes, yet segment-score alignment is wrong-way
  and pooled-point allocation is slightly better than primary.

Decision:
- This is not promotion evidence: it loses to Douglas-Peucker and fails
  workload signature, global sanity, and learning causality.
- The behavior-rank default now has strict Level 2 diagnostic support for
  material no-behavior ablation, but the head remains compressed. Do not tune
  behavior-rank weight further from this result.
- The next admissible checkpoint should target query-prior materiality with a
  new root hypothesis. Do not repeat the rejected semantic prior-head adapter,
  scalar prior boosts, route-density exposure, or generic post-context residual.

## Checkpoint Phase 16 - query-prior materiality root diagnosis

Status: completed / accepted-as-diagnostic-boundary.

Hypothesis:
- Query-prior fields reach model inputs and contain target signal, but the
  trained factorized heads are functionally invariant to those prior channels.
- If non-prior features cannot reconstruct the prior channels well, the blocker
  is prior-feature integration or learning pressure, not prior sampling,
  normalization, redundancy, score composition, or selector thresholding.

Changed files:
- `learning/factorized_head_diagnostics.py`
- `learning/model_training.py`
- `orchestration/diagnostics/semantic_causality_diagnostic.py`
- `tests/unit/learning/test_query_local_utility_training.py`
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/learning/model_training.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/learning/model_training.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pyright learning/factorized_head_diagnostics.py learning/model_training.py tests/unit/learning/test_query_local_utility_training.py)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_prior_feature_learning_diagnostic_localizes_invariant_heads -q`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pyright orchestration/diagnostics/semantic_causality_diagnostic.py)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_behavior_head_training_signal_diagnostic_compares_rank_to_bias_baseline Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_prior_feature_learning_diagnostic_localizes_invariant_heads Range_QDS/tests/unit/selection/test_query_driven_learned_segment_budget.py::test_retained_decision_marginal_query_local_utility_diagnostic_scores_true_marginals Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py -q`
- `git diff --check`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2541 --n_ships 12 --n_points 96 --synthetic_route_families 3 --n_queries 8 --max_queries 64 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 12000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 2 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/query_prior_materiality_root_diagnosis_level1_smoke)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2539 --n_ships 32 --n_points 192 --synthetic_route_families 3 --n_queries 24 --max_queries 192 --epochs 4 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 30000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/query_prior_materiality_root_diagnosis_seed2539)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/query_prior_materiality_root_diagnosis_seed2539/example_run.json --output artifacts/results/query_prior_materiality_root_diagnosis_seed2539/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/query_prior_materiality_root_diagnosis_level1_smoke/example_run.json`
- `artifacts/results/query_prior_materiality_root_diagnosis_seed2539/example_run.json`
- `artifacts/results/query_prior_materiality_root_diagnosis_seed2539/semantic_diagnostic.json`

Scale:
- Level 1 wiring smoke, seed `2541`
- Level 2 minimum strict, synthetic, source-stratified, seed `2539`
- Level 2 dimensions: `n_ships=32`, `n_points=192`, `n_queries=24`,
  `max_queries=192`, `epochs=4`, `range_train_workload_replicates=4`

Key results:
- Scores match the previous strict Level 2 boundary: MLQDS `0.1090154720`,
  uniform `0.0992909061`, Douglas-Peucker `0.1182249577`.
- Gates remain non-promotable: workload stability, support overlap, target
  diffusion, predictability, and prior-predictive alignment pass; workload
  signature, learning causality, global sanity, and final success fail.
- Learning-causality prior child gates still fail: shuffled-prior delta `0.0`,
  no-query-prior delta `0.0`.
- New prior learning-signal diagnostic:
  - classification `prior_target_signal_available_but_trained_heads_invariant`
  - prior-signal head count `5`
  - prior best-Spearman beats non-prior count `4`
  - zero-prior mean abs head-probability delta `0.0000494446`
  - zero-prior max abs head-probability delta `0.000553429`
  - zero-prior mean abs final-probability delta `0.000000777954`
  - zero-prior mean abs final-logit delta `0.000178006`
  - prior feature scale `0.248315`
  - scaled-prior / point-encoder std ratio `0.426903`
  - scaled-prior / point-encoder L2 ratio `0.424374`
  - mean/max prior reconstruction R2 from non-prior features `0.382226` /
    `0.499199`
- Best prior channels beat best non-prior single features for query-hit,
  boundary, replacement, and segment-budget heads; they do not beat non-prior
  features for behavior or path-length heads.
- Semantic diagnostic still classifies the prior-flow failure as `model ignores
  prior inputs`, now with the training-side prior learning-signal summary.

Decision:
- This checkpoint rules out the easy explanations: prior sampling moves,
  normalized model priors move, prior channels contain target signal, prior
  channels are not simply redundant with non-prior features, and the prior path
  is not collapsed by scalar scale.
- The blocker is trained-head invariance to prior channels. Do not use scalar
  prior amplification, generic prior residuals, route-density exposure, prior
  adapter replay, selector thresholds, or final-grid runs.
- Next admissible step: a focused prior-feature integration or learning-pressure
  root fix, restarted at Level 1 wiring and Level 2 strict evidence.

## Checkpoint Phase 17 - prior-feature integration stage diagnosis

Status: completed / accepted-as-diagnostic-boundary.

Hypothesis:
- The existing prior path may carry nontrivial early signal, but local/segment
  context or the final head MLPs may attenuate it before factorized head
  probabilities.
- If shared embeddings still move while head probabilities barely move, the
  next root fix should target head/loss credit assignment rather than selector
  scoring or scalar prior amplification.

Changed files:
- `learning/factorized_head_diagnostics.py`
- `learning/model_training.py`
- `orchestration/diagnostics/semantic_causality_diagnostic.py`
- `tests/unit/learning/test_query_local_utility_training.py`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/learning/model_training.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/learning/model_training.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pyright learning/factorized_head_diagnostics.py learning/model_training.py orchestration/diagnostics/semantic_causality_diagnostic.py tests/unit/learning/test_query_local_utility_training.py)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_prior_feature_learning_diagnostic_reports_model_stage_sensitivity Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_prior_feature_learning_diagnostic_localizes_invariant_heads -q`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2542 --n_ships 12 --n_points 96 --synthetic_route_families 3 --n_queries 8 --max_queries 64 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 12000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 2 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/prior_feature_integration_stage_diagnosis_level1_smoke)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2539 --n_ships 32 --n_points 192 --synthetic_route_families 3 --n_queries 24 --max_queries 192 --epochs 4 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 30000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/prior_feature_integration_stage_diagnosis_seed2539)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/prior_feature_integration_stage_diagnosis_seed2539/example_run.json --output artifacts/results/prior_feature_integration_stage_diagnosis_seed2539/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/prior_feature_integration_stage_diagnosis_level1_smoke/example_run.json`
- `artifacts/results/prior_feature_integration_stage_diagnosis_seed2539/example_run.json`
- `artifacts/results/prior_feature_integration_stage_diagnosis_seed2539/semantic_diagnostic.json`

Scale:
- Level 1 wiring smoke, seed `2542`
- Level 2 minimum strict, synthetic, source-stratified, seed `2539`
- Level 2 dimensions: `n_ships=32`, `n_points=192`, `n_queries=24`,
  `max_queries=192`, `epochs=4`, `range_train_workload_replicates=4`

Key results:
- Scores are unchanged from the current strict Level 2 boundary: MLQDS
  `0.1090154720`, uniform `0.0992909061`, Douglas-Peucker `0.1182249577`.
- Learning causality remains blocked by prior child gates: shuffled-prior delta
  `0.0`, no-query-prior delta `0.0`.
- Stage sensitivity on the strict Level 2 replay:
  - pre-context mean abs prior delta `0.00202673`
  - post-local-context mean abs prior delta `0.00375001`
  - pre-shared-encoder mean abs prior delta `0.00421415`
  - shared-embedding mean abs prior delta `0.00147635`
  - head-logit mean abs prior delta `0.000358624`
  - head-probability mean abs prior delta `0.0000494458`
  - shared/pre-context delta ratio `0.72844`
  - head-probability/pre-context delta ratio `0.024397`

Decision:
- The prior signal is not fully washed out before the shared embedding. The
  severe attenuation happens at the head/loss interface: shared embeddings move
  materially, but head probabilities barely move.
- Do not implement scalar prior boosts, selector changes, final-grid runs, or a
  direct prior-head logit adapter.
- Next admissible step: a focused learning-pressure fix that forces the
  existing factorized heads to use train-derived prior channels, with Level 1
  wiring and Level 2 strict evidence before any Level 3 run.

## Checkpoint Phase 18 - prior-only auxiliary learning pressure

Status: rejected / cleaned up.

Hypothesis:
- Adding a training-only auxiliary factorized-head loss on prior-only
  `workload_blind_range` inputs might force the existing heads to map
  train-derived prior channels to query-local utility targets.
- This should make shuffled-prior and no-query-prior child ablations material
  without changing metric, selector, inference-time features, or adding a direct
  prior-to-logit adapter.

Changed files:
- `config/run_config.py`
- `learning/optimization_epoch.py`
- `learning/model_training.py`
- `orchestration/learning_scoring_cli.py`
- `orchestration/train_and_score.py`
- `tests/unit/learning/test_query_local_utility_training.py`
- `tests/unit/runtime/test_torch_runtime_controls.py`
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/config/run_config.py Range_QDS/learning/optimization_epoch.py Range_QDS/learning/model_training.py Range_QDS/orchestration/learning_scoring_cli.py Range_QDS/orchestration/train_and_score.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/config/run_config.py Range_QDS/learning/optimization_epoch.py Range_QDS/learning/model_training.py Range_QDS/orchestration/learning_scoring_cli.py Range_QDS/orchestration/train_and_score.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pyright config/run_config.py learning/optimization_epoch.py learning/model_training.py orchestration/learning_scoring_cli.py orchestration/train_and_score.py tests/unit/learning/test_query_local_utility_training.py tests/unit/runtime/test_torch_runtime_controls.py)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_prior_feature_learning_diagnostic_reports_model_stage_sensitivity Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py::test_direct_config_and_cli_default_to_non_residual_training -q`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2543 --n_ships 12 --n_points 96 --synthetic_route_families 3 --n_queries 8 --max_queries 64 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 12000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 2 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/prior_only_aux_learning_pressure_level1_smoke)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2539 --n_ships 32 --n_points 192 --synthetic_route_families 3 --n_queries 24 --max_queries 192 --epochs 4 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 30000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/prior_only_aux_learning_pressure_seed2539)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/prior_only_aux_learning_pressure_seed2539/example_run.json --output artifacts/results/prior_only_aux_learning_pressure_seed2539/semantic_diagnostic.json)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/config/run_config.py Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/learning/model_training.py Range_QDS/learning/optimization_epoch.py Range_QDS/orchestration/learning_scoring_cli.py Range_QDS/orchestration/selector_diagnostics.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py Range_QDS/tests/unit/selection/test_query_driven_learned_segment_budget.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/config/run_config.py Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/learning/model_training.py Range_QDS/learning/optimization_epoch.py Range_QDS/orchestration/learning_scoring_cli.py Range_QDS/orchestration/selector_diagnostics.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py Range_QDS/tests/unit/selection/test_query_driven_learned_segment_budget.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pyright config/run_config.py learning/factorized_head_diagnostics.py learning/model_training.py learning/optimization_epoch.py orchestration/learning_scoring_cli.py orchestration/selector_diagnostics.py orchestration/diagnostics/semantic_causality_diagnostic.py tests/unit/learning/test_query_local_utility_training.py tests/unit/runtime/test_torch_runtime_controls.py tests/unit/selection/test_query_driven_learned_segment_budget.py tests/unit/orchestration/test_semantic_causality_diagnostic.py)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_behavior_head_training_signal_diagnostic_compares_rank_to_bias_baseline Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_prior_feature_learning_diagnostic_localizes_invariant_heads Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_prior_feature_learning_diagnostic_reports_model_stage_sensitivity Range_QDS/tests/unit/selection/test_query_driven_learned_segment_budget.py::test_retained_decision_marginal_query_local_utility_diagnostic_scores_true_marginals Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py::test_direct_config_and_cli_default_to_non_residual_training -q`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest Range_QDS/tests/unit/orchestration/test_query_driven_diagnostics.py Range_QDS/tests/unit/orchestration/test_query_driven_causality_and_summary.py Range_QDS/tests/unit/learning/test_query_local_utility_targets.py Range_QDS/tests/guardrails/test_implementation_guardrails.py -q`
- `git diff --check`

Artifact:
- `artifacts/results/prior_only_aux_learning_pressure_level1_smoke/example_run.json`
- `artifacts/results/prior_only_aux_learning_pressure_seed2539/example_run.json`
- `artifacts/results/prior_only_aux_learning_pressure_seed2539/semantic_diagnostic.json`

Scale:
- Level 1 wiring smoke, seed `2543`
- Level 2 minimum strict, synthetic, source-stratified, seed `2539`
- Level 2 dimensions: `n_ships=32`, `n_points=192`, `n_queries=24`,
  `max_queries=192`, `epochs=4`, `range_train_workload_replicates=4`

Key results:
- Level 2 scores: MLQDS `0.1077803118`, uniform `0.0992909061`,
  Douglas-Peucker `0.1182249577`.
- The run still loses to Douglas-Peucker and scores below the previous strict
  Level 2 boundary MLQDS score `0.1090154720`.
- Prior child gates remain exactly failed:
  - shuffled-prior delta `0.0`
  - no-query-prior delta `0.0`
- Other causality changed in the wrong direction:
  - shuffled-score delta `0.0201403`
  - no-segment-budget delta `0.0200070`
  - no-behavior-head delta `0.0025521`, below the `0.005` materiality floor
- Prior stage diagnostic remains invariant:
  - zero-prior mean abs head-probability delta `0.0000432262`
  - zero-prior mean abs final-probability delta `0.000000896114`
  - head-probability/pre-context delta ratio `0.0213471`
- Behavior training-signal diagnostic still says
  `rank_pressure_improves_but_prediction_still_flat`; BCE is worse than the
  constant-bias baseline by `-0.0032841`.
- Cleanup/final validation passed after removing the prior-only auxiliary
  production path: `py_compile`, `ruff`, package-root `pyright`,
  `git diff --check`, `6` focused pytest tests, and `59` broader
  instrumentation/guardrail tests.

Decision:
- Reject this path. It does not make query-prior ablations material, lowers the
  primary score, and regresses the behavior child ablation below the materiality
  gate.
- The production stack was cleaned back to no prior-only auxiliary default or
  CLI/config surface.
- Do not repeat prior-only auxiliary learning pressure unless a new hypothesis
  explains why the current result would no longer apply.
- Next admissible step: a prior-conditioned head-input root fix, starting with
  Level 1 wiring and then Level 2 strict evidence before any Level 3 run.

## Checkpoint Phase 19 - prior-conditioned head-input projection

Status: rejected / cleaned up.

Hypothesis:
- Feeding a projected `[shared_embedding, explicit_prior_embedding]` tensor into
  the existing factorized heads might make train-derived priors material without
  direct prior-to-logit addition, selector changes, metric changes, or
  prior-only auxiliary replay.
- If Level 1 does not move head probabilities or retained masks, stop before
  Level 2 and diagnose the head-transfer attenuation.

Changed files:
- `models/workload_blind_range.py` (transient, reverted)
- `learning/checkpoints.py` (transient, reverted)
- `learning/factorized_head_diagnostics.py` (transient head-input stage,
  reverted)
- `orchestration/diagnostics/semantic_causality_diagnostic.py` (transient
  head-input summary, reverted)
- `tests/unit/learning/test_model_features.py` (transient, reverted)
- `tests/unit/orchestration/test_query_driven_protocol_gates.py` (transient,
  reverted)
- `tests/unit/learning/test_query_local_utility_training.py` (transient,
  reverted)
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/models/workload_blind_range.py Range_QDS/learning/checkpoints.py Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/learning/test_model_features.py Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/models/workload_blind_range.py Range_QDS/learning/checkpoints.py Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/learning/test_model_features.py Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pyright models/workload_blind_range.py learning/checkpoints.py learning/factorized_head_diagnostics.py orchestration/diagnostics/semantic_causality_diagnostic.py tests/unit/learning/test_model_features.py tests/unit/orchestration/test_query_driven_protocol_gates.py tests/unit/learning/test_query_local_utility_training.py)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py::test_workload_blind_range_has_dedicated_prior_feature_encoder Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py::test_workload_blind_range_untrained_reset_restores_standalone_parameters Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py::test_workload_blind_range_features_and_selector_are_query_free Range_QDS/tests/unit/learning/test_model_features.py::test_workload_blind_range_checkpoint_accepts_missing_prior_feature_and_head_input_modules Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_prior_feature_learning_diagnostic_reports_model_stage_sensitivity -q`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2544 --n_ships 12 --n_points 96 --synthetic_route_families 3 --n_queries 8 --max_queries 64 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 12000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 2 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/prior_conditioned_head_input_level1_smoke)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/prior_conditioned_head_input_level1_smoke/example_run.json --output artifacts/results/prior_conditioned_head_input_level1_smoke/semantic_diagnostic.json)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/config/run_config.py Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/learning/model_training.py Range_QDS/learning/optimization_epoch.py Range_QDS/orchestration/learning_scoring_cli.py Range_QDS/orchestration/selector_diagnostics.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py Range_QDS/tests/unit/selection/test_query_driven_learned_segment_budget.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/config/run_config.py Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/learning/model_training.py Range_QDS/learning/optimization_epoch.py Range_QDS/orchestration/learning_scoring_cli.py Range_QDS/orchestration/selector_diagnostics.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py Range_QDS/tests/unit/selection/test_query_driven_learned_segment_budget.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pyright config/run_config.py learning/factorized_head_diagnostics.py learning/model_training.py learning/optimization_epoch.py orchestration/learning_scoring_cli.py orchestration/selector_diagnostics.py orchestration/diagnostics/semantic_causality_diagnostic.py tests/unit/learning/test_query_local_utility_training.py tests/unit/runtime/test_torch_runtime_controls.py tests/unit/selection/test_query_driven_learned_segment_budget.py tests/unit/orchestration/test_semantic_causality_diagnostic.py)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_behavior_head_training_signal_diagnostic_compares_rank_to_bias_baseline Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_prior_feature_learning_diagnostic_localizes_invariant_heads Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_prior_feature_learning_diagnostic_reports_model_stage_sensitivity Range_QDS/tests/unit/selection/test_query_driven_learned_segment_budget.py::test_retained_decision_marginal_query_local_utility_diagnostic_scores_true_marginals Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py::test_direct_config_and_cli_default_to_non_residual_training -q`
- `git diff --check`

Artifact:
- `artifacts/results/prior_conditioned_head_input_level1_smoke/example_run.json`
- `artifacts/results/prior_conditioned_head_input_level1_smoke/semantic_diagnostic.json`

Scale:
- Level 1 wiring smoke, seed `2544`
- `n_ships=12`, `n_points=96`, `n_queries=8`, `max_queries=64`,
  `epochs=1`, `range_train_workload_replicates=2`

Key results:
- Level 1 scores: MLQDS `0.0340754791`, uniform `0.0523374582`,
  Douglas-Peucker `0.0742743774`.
- The smoke loses to both baselines and fails workload stability, workload
  signature, predictability, prior-predictive alignment, learning causality,
  global sanity, and final success.
- Prior child gates remain exactly failed:
  - shuffled-prior delta `0.0`
  - no-query-prior delta `0.0`
- Transient stage diagnostic localized the failed mechanism:
  - explicit-prior mean abs delta `0.002047`
  - shared-embedding mean abs delta `0.001072`
  - projected head-input mean abs delta `0.002159`
  - head-logit mean abs delta `0.000270`
  - head-probability mean abs delta `0.0000283`
  - head-probability/pre-context delta ratio `0.01353`
- Behavior regressed at smoke scale:
  - classification `rank_pressure_available_but_head_near_bias`
  - rank pair accuracy `0.4931`
  - rank loss improvement over bias `-0.000991`
  - BCE improvement over bias `-0.0000238`
- Cleanup validation passed after removing the production head-input projection:
  `py_compile`, `ruff`, package-root `pyright`, `git diff --check`, and `6`
  focused pytest tests.

Decision:
- Reject this path. Moving a prior-conditioned head-input tensor did not make
  factorized head probabilities or masks prior-sensitive, and the smoke
  underperformed both baselines.
- The production stack was cleaned back to no prior-conditioned head-input
  projection, no schema bump, and no checkpoint compatibility shim for that
  rejected path.
- Do not run Level 2 or Level 3 from this artifact. Do not repeat
  prior-conditioned head-input projection unless a new diagnostic explains why
  the head MLP attenuation would no longer apply.
- Next admissible step: prior-to-head transfer sensitivity diagnosis. Quantify
  head-layer suppression, bias/base-rate saturation, and per-head local
  sensitivity before another production architecture change.

## Checkpoint Phase 20 - prior-to-head transfer sensitivity diagnosis

Status: accepted-as-new-diagnostic-boundary.

Hypothesis:
- Query-prior channels are reaching the shared embedding, but the factorized
  head MLPs suppress the prior-sensitive direction before it becomes a material
  logit/probability/mask change.
- A diagnostic-only per-head transfer trace can localize the attenuation without
  changing metric, target, selector, model semantics, or production scoring.

Changed files:
- `learning/factorized_head_diagnostics.py`
- `learning/model_training.py`
- `orchestration/diagnostics/semantic_causality_diagnostic.py`
- `tests/unit/learning/test_query_local_utility_training.py`
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pyright learning/factorized_head_diagnostics.py orchestration/diagnostics/semantic_causality_diagnostic.py tests/unit/learning/test_query_local_utility_training.py)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_prior_feature_learning_diagnostic_reports_model_stage_sensitivity -q`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2545 --n_ships 12 --n_points 96 --synthetic_route_families 3 --n_queries 8 --max_queries 64 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 12000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 2 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/prior_to_head_transfer_sensitivity_level1_smoke)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/prior_to_head_transfer_sensitivity_level1_smoke/example_run.json --output artifacts/results/prior_to_head_transfer_sensitivity_level1_smoke/semantic_diagnostic.json)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2539 --n_ships 32 --n_points 192 --synthetic_route_families 3 --n_queries 24 --max_queries 192 --epochs 4 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 30000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/prior_to_head_transfer_sensitivity_seed2539)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/prior_to_head_transfer_sensitivity_seed2539/example_run.json --output artifacts/results/prior_to_head_transfer_sensitivity_seed2539/semantic_diagnostic.json)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/config/run_config.py Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/learning/model_training.py Range_QDS/learning/optimization_epoch.py Range_QDS/orchestration/learning_scoring_cli.py Range_QDS/orchestration/selector_diagnostics.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py Range_QDS/tests/unit/selection/test_query_driven_learned_segment_budget.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/config/run_config.py Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/learning/model_training.py Range_QDS/learning/optimization_epoch.py Range_QDS/orchestration/learning_scoring_cli.py Range_QDS/orchestration/selector_diagnostics.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py Range_QDS/tests/unit/selection/test_query_driven_learned_segment_budget.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pyright config/run_config.py learning/factorized_head_diagnostics.py learning/model_training.py learning/optimization_epoch.py orchestration/learning_scoring_cli.py orchestration/selector_diagnostics.py orchestration/diagnostics/semantic_causality_diagnostic.py tests/unit/learning/test_query_local_utility_training.py tests/unit/runtime/test_torch_runtime_controls.py tests/unit/selection/test_query_driven_learned_segment_budget.py tests/unit/orchestration/test_semantic_causality_diagnostic.py)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_behavior_head_training_signal_diagnostic_compares_rank_to_bias_baseline Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_prior_feature_learning_diagnostic_localizes_invariant_heads Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_prior_feature_learning_diagnostic_reports_model_stage_sensitivity Range_QDS/tests/unit/selection/test_query_driven_learned_segment_budget.py::test_retained_decision_marginal_query_local_utility_diagnostic_scores_true_marginals Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py::test_direct_config_and_cli_default_to_non_residual_training -q`
- `git diff --check`

Artifact:
- `artifacts/results/prior_to_head_transfer_sensitivity_level1_smoke/example_run.json`
- `artifacts/results/prior_to_head_transfer_sensitivity_level1_smoke/semantic_diagnostic.json`
- `artifacts/results/prior_to_head_transfer_sensitivity_seed2539/example_run.json`
- `artifacts/results/prior_to_head_transfer_sensitivity_seed2539/semantic_diagnostic.json`

Scale:
- Level 1 wiring smoke, seed `2545`
- Level 2 minimum strict, synthetic, source-stratified, seed `2539`
- Level 2 dimensions: `n_ships=32`, `n_points=192`, `n_queries=24`,
  `max_queries=192`, `epochs=4`, `range_train_workload_replicates=4`

Key results:
- Level 1 was wiring-only and non-promotable: MLQDS `0.0401020116`, uniform
  `0.0419019015`, Douglas-Peucker `0.0519044086`.
- Level 2 scores match the previous strict boundary: MLQDS `0.1090154720`,
  uniform `0.0992909061`, Douglas-Peucker `0.1182249577`.
- Level 2 gates pass workload stability, support overlap, target diffusion,
  predictability, and prior-predictive alignment; workload signature, learning
  causality, global sanity, and final success fail.
- Learning causality is still blocked by prior child gates:
  shuffled-prior delta `0.0`, no-query-prior delta `0.0`.
- Prior-to-head transfer localizes the attenuation:
  `output_layer_suppresses_prior_direction` for all `6/6` heads.
- Representative Level 2 ratios:
  - `query_hit_probability`: first/shared `0.400244`, hidden/first `0.611673`,
    logit/hidden `0.0538385`, probability/logit `0.0089897`
  - `conditional_behavior_utility`: first/shared `0.307635`, hidden/first
    `0.683590`, logit/hidden `0.0452444`, probability/logit `0.0701370`
  - `segment_budget_target`: first/shared `0.407783`, hidden/first `0.487005`,
    logit/hidden `0.0494308`, probability/logit `0.171204`
- Query-hit and boundary heads also show low-base-rate sigmoid saturation:
  query-hit sigmoid derivative mean `0.0090604`; boundary derivative mean
  `0.0001951`.

Decision:
- Accept this as the current diagnostic boundary, not promotion evidence.
- The next root fix must target head output-layer / decision-surface learning
  and calibration. The prior path reaches shared and hidden layers; output
  projections erase the prior-sensitive direction before logits become useful.
- Do not run Level 3 or the final grid from this boundary.
- Do not repeat generic behavior-rank tuning, scalar prior boosts, post-context
  prior residuals, selector allocation-floor changes, route-density exposure,
  prior-only auxiliary replay, direct prior-head logit adapters, or
  prior-conditioned head-input projection.
- Next admissible step: `head_output_layer_prior_direction_root_fix`, starting
  with static/unit validation and Level 1 wiring before any Level 2 strict
  replay.

## Checkpoint Phase 21 - head-output prior-direction contrastive loss

Status: rejected / cleaned up.

Hypothesis:
- The prior-sensitive shared and hidden activations are present, but the scalar
  head output layers lack supervised pressure to align their decision surface
  with that direction.
- A training-only primary-minus-zero-prior head-logit contrastive rank loss
  might make existing head output layers preserve target-relevant prior-induced
  hidden deltas, without adding an inference-time prior-to-logit adapter,
  selector compensation, or prior-only replay.

Changed files:
- `config/run_config.py` (transient, reverted)
- `learning/optimization_epoch.py` (transient, reverted)
- `learning/model_training.py` (transient, reverted)
- `orchestration/learning_scoring_cli.py` (transient, reverted)
- `orchestration/train_and_score.py` (transient, reverted)
- `tests/unit/learning/test_query_local_utility_training.py` (transient,
  reverted)
- `tests/unit/runtime/test_torch_runtime_controls.py` (transient, reverted)
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/config/run_config.py Range_QDS/learning/optimization_epoch.py Range_QDS/learning/model_training.py Range_QDS/orchestration/learning_scoring_cli.py Range_QDS/orchestration/train_and_score.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/config/run_config.py Range_QDS/learning/optimization_epoch.py Range_QDS/learning/model_training.py Range_QDS/orchestration/learning_scoring_cli.py Range_QDS/orchestration/train_and_score.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pyright config/run_config.py learning/optimization_epoch.py learning/model_training.py orchestration/learning_scoring_cli.py orchestration/train_and_score.py tests/unit/learning/test_query_local_utility_training.py tests/unit/runtime/test_torch_runtime_controls.py)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_prior_direction_head_rank_loss_penalizes_wrong_prior_delta_order Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_prior_feature_learning_diagnostic_reports_model_stage_sensitivity Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py::test_run_config_roundtrips_precision_controls Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py::test_direct_config_and_cli_default_to_non_residual_training Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py::test_run_config_loads_missing_runtime_and_mlqds_defaults -q`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2546 --n_ships 12 --n_points 96 --synthetic_route_families 3 --n_queries 8 --max_queries 64 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 12000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 2 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/head_output_layer_prior_direction_level1_smoke)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/head_output_layer_prior_direction_level1_smoke/example_run.json --output artifacts/results/head_output_layer_prior_direction_level1_smoke/semantic_diagnostic.json)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/config/run_config.py Range_QDS/learning/optimization_epoch.py Range_QDS/learning/model_training.py Range_QDS/orchestration/learning_scoring_cli.py Range_QDS/orchestration/train_and_score.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/config/run_config.py Range_QDS/learning/optimization_epoch.py Range_QDS/learning/model_training.py Range_QDS/orchestration/learning_scoring_cli.py Range_QDS/orchestration/train_and_score.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pyright config/run_config.py learning/optimization_epoch.py learning/model_training.py orchestration/learning_scoring_cli.py orchestration/train_and_score.py tests/unit/learning/test_query_local_utility_training.py tests/unit/runtime/test_torch_runtime_controls.py)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_prior_feature_learning_diagnostic_reports_model_stage_sensitivity Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_behavior_head_training_signal_diagnostic_compares_rank_to_bias_baseline Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py::test_run_config_roundtrips_precision_controls Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py::test_direct_config_and_cli_default_to_non_residual_training Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py::test_run_config_loads_missing_runtime_and_mlqds_defaults -q`
- `git diff --check`

Artifact:
- `artifacts/results/head_output_layer_prior_direction_level1_smoke/example_run.json`
- `artifacts/results/head_output_layer_prior_direction_level1_smoke/semantic_diagnostic.json`

Scale:
- Level 1 wiring smoke, seed `2546`
- `n_ships=12`, `n_points=96`, `n_queries=8`, `max_queries=64`,
  `epochs=1`, `range_train_workload_replicates=2`

Key results:
- The transient loss was active in the artifact:
  `prior_direction_loss_weight=0.2`.
- Level 1 scores were non-promotable and below both baselines: MLQDS
  `0.0790333334`, uniform `0.0866883767`, Douglas-Peucker `0.1174806473`.
- Gates failed as expected for a rejected smoke: workload stability,
  predictability, workload signature, learning causality, global sanity, and
  final success were false.
- Prior child gates remained exactly failed:
  - shuffled-prior delta `0.0`
  - no-query-prior delta `0.0`
  - without-query-prior-features delta `0.0`
- Prior sensitivity stayed tiny:
  - mean abs head-probability delta `0.0000479254`
  - mean abs final-probability delta `0.0000012029`
  - head-probability/pre-context delta ratio `0.0203382`
- Prior-to-head transfer still classified all heads as
  `output_layer_suppresses_prior_direction`.
- Other causality moved the wrong way or remained weak:
  - shuffled-score delta `-0.0271463`
  - no-behavior-head delta `0.0022398`
  - no-segment-budget-head delta `-0.0128944`
  - prior-field-only delta `-0.0307003`

Decision:
- Reject this path. The active contrastive loss did not make prior ablations
  material even at wiring scale, and the smoke underperformed both baselines.
- The production stack was cleaned back to no prior-direction contrastive loss,
  no CLI/config surface, and no training-loop twin-forward path.
- Do not run Level 2 or Level 3 from this artifact.
- Do not repeat primary-minus-zero-prior contrastive-loss replay unless a new
  diagnostic explains why this Level 1 failure would no longer apply.
- Next admissible step: `head_output_layer_gradient_alignment_diagnosis`.
  Diagnose final-layer gradient/weight alignment and sigmoid/base-rate
  saturation before another production model or loss change.

## Checkpoint Phase 22 - head-output gradient alignment diagnosis

Status: completed.

Hypothesis:
- Existing artifacts do not contain enough tensor or gradient state to explain
  why the output layer suppresses prior-sensitive hidden deltas.
- The final scalar head projections may be nearly orthogonal to the
  prior-sensitive hidden direction, and the configured factorized loss may give
  weak or conflicting descent signal along that direction.

Changed files:
- `learning/factorized_head_diagnostics.py`
- `orchestration/diagnostics/semantic_causality_diagnostic.py`
- `tests/unit/learning/test_query_local_utility_training.py`
- `tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/learning/model_training.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/learning/model_training.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pyright learning/factorized_head_diagnostics.py learning/model_training.py orchestration/diagnostics/semantic_causality_diagnostic.py tests/unit/learning/test_query_local_utility_training.py tests/unit/orchestration/test_semantic_causality_diagnostic.py)`
- `.venv/bin/pytest Range_QDS/tests/unit/learning/test_query_local_utility_training.py -q`
- `.venv/bin/pytest Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py -q`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2547 --n_ships 12 --n_points 96 --synthetic_route_families 3 --n_queries 8 --max_queries 64 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 12000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 2 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/head_output_layer_gradient_alignment_level1_smoke)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/head_output_layer_gradient_alignment_level1_smoke/example_run.json --output artifacts/results/head_output_layer_gradient_alignment_level1_smoke/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/head_output_layer_gradient_alignment_level1_smoke/example_run.json`
- `artifacts/results/head_output_layer_gradient_alignment_level1_smoke/semantic_diagnostic.json`

Scale:
- static/unit plus Level 1 wiring smoke, seed `2547`
- `n_ships=12`, `n_points=96`, `n_queries=8`, `max_queries=64`,
  `epochs=1`, `range_train_workload_replicates=2`

Key results:
- Level 1 scores are wiring-only and non-promotable: MLQDS `0.0936764156`,
  uniform `0.0606504771`, Douglas-Peucker `0.0819907562`.
- Gates failed as expected at smoke scale: workload stability, workload
  signature, predictability, prior-predictive alignment, learning causality,
  and global sanity were false.
- Prior child gates remained exactly failed:
  - shuffled-prior delta `0.0`
  - no-query-prior delta `0.0`
- Prior-to-head transfer still classified all six heads as
  `output_layer_suppresses_prior_direction`.
- Final output-layer projection is small:
  - final-weight/hidden-delta abs cosine mean range `0.0726` to `0.1542`
  - projected-hidden-delta L2 / hidden-delta L2 range `0.0390` to `0.0819`
  - target/logit-delta Spearman range `-0.1143` to `0.0665`
- Configured factorized-loss descent alignment is effectively zero or mixed
  along the prior-induced logit deltas. Segment loss uses the diagnostic
  window-order segment proxy because canonical segment IDs are not available in
  this fit diagnostic:
  - query-hit descent mean `-3.56e-11`, positive fraction `0.500`
  - behavior descent mean `1.43e-08`, positive fraction `0.154`
  - segment-budget descent mean `-6.00e-09`, positive fraction `0.542`

Decision:
- The diagnostic supports the output-layer decision-surface blocker. Hidden
  prior deltas survive into the head MLP, but the scalar final projections
  throw most of that direction away.
- The configured loss does not provide a clean descent signal along the current
  prior-induced logit deltas. More prior scale, selector tuning, or the rejected
  contrastive replay is not justified.
- Do not promote this Level 1 smoke. Do not run Level 2 until a root fix moves
  prior child deltas in the expected direction at Level 1.
- Next admissible step: design a root fix for head-output decision-surface
  credit assignment, then restart at static/unit plus Level 1 wiring.

## Checkpoint Phase 23 - head-output decision-surface alignment loss

Status: rejected / cleaned up.

Hypothesis:
- The prior-sensitive hidden direction reaches each head, but the final scalar
  projection is oriented away from target-relevant prior movement.
- A training-only final-head decision-surface alignment loss might rotate final
  head weights toward target-correlated prior hidden deltas without inference
  adapters, selector compensation, scalar prior boosts, or replaying rejected
  contrastive losses.

Changed files:
- `learning/optimization_epoch.py` (transient alignment helper removed)
- `learning/model_training.py` (transient diagnostic loss field removed)
- `orchestration/train_and_score.py` (transient CLI/config pass-through removed)
- `tests/unit/learning/test_query_local_utility_training.py` (transient helper
  test removed)
- `tests/unit/runtime/test_torch_runtime_controls.py` (transient config/CLI
  assertions removed)
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `.venv/bin/pytest Range_QDS/tests/unit/learning/test_query_local_utility_training.py -q`
- `.venv/bin/pytest Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py -q`
- `.venv/bin/python -m py_compile Range_QDS/learning/optimization_epoch.py Range_QDS/learning/model_training.py Range_QDS/orchestration/train_and_score.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py`
- `.venv/bin/ruff check Range_QDS/learning/optimization_epoch.py Range_QDS/learning/model_training.py Range_QDS/orchestration/train_and_score.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py`
- `(cd Range_QDS && ../.venv/bin/pyright learning/optimization_epoch.py learning/model_training.py orchestration/train_and_score.py tests/unit/learning/test_query_local_utility_training.py tests/unit/runtime/test_torch_runtime_controls.py)`
- `git diff --check`

Artifact:
- `artifacts/results/head_output_decision_surface_root_fix_level1_smoke/example_run.json`
- `artifacts/results/head_output_decision_surface_root_fix_level1_smoke/semantic_diagnostic.json`

Scale:
- Level 1 wiring smoke, seed `2548`
- `n_ships=12`, `n_points=96`, `n_queries=8`, `max_queries=64`,
  `epochs=1`, `range_train_workload_replicates=2`

Key results:
- The transient loss was active in the artifact:
  `query_local_utility_prior_direction_alignment_loss_weight=0.10`.
- Level 1 scores were non-promotable: MLQDS `0.0756977785`, uniform
  `0.0352864152`, Douglas-Peucker `0.0802253341`.
- Required smoke gates still failed: workload stability, predictability,
  prior-predictive alignment, workload signature, learning causality, global
  sanity, and final success were false.
- Prior child gates remained exactly failed:
  - shuffled-prior delta `0.0`
  - no-query-prior delta `0.0`
  - without-query-prior-features delta `0.0`
- The positive controls were mixed: shuffled-score delta `0.0205070765`,
  no-segment-budget delta `0.0199685780`, prior-field-only delta
  `0.0155178991`, but no-behavior-head delta regressed to `-0.0012989662`.
- Prior-to-head transfer still classified all six heads as
  `output_layer_suppresses_prior_direction`.
- Projection ratios moved for some heads but did not become causal:
  query-hit projected-hidden-delta ratio `0.147315` with target/logit Spearman
  `-0.073091`; segment-budget projected-hidden-delta ratio `0.125323` with
  target/logit Spearman `0.141161`.
- Segment-score retained-marginal alignment remained weak: segment-score
  Spearman `0.164679`, but top-minus-bottom marginal was slightly wrong-way at
  `-0.0000664`; raw and selector scores had positive Spearman and positive
  top-minus-bottom marginal.

Decision:
- Reject this path. Direct final-head decision-surface alignment can increase a
  projection readout without making query-prior ablations material at the mask
  level.
- The production stack was cleaned back to no
  `query_local_utility_prior_direction_alignment_loss_weight` surface and no
  final-head alignment helper.
- Do not run Level 2 or Level 3 from this artifact.
- Do not tune the alignment weight or repeat final-head cosine alignment unless
  a new diagnostic explains why this Level 1 failure no longer applies.
- Next admissible step: `prior_direction_target_covariance_slice_diagnosis`.
  Diagnose target/prior covariance and selector materiality by head,
  query/footprint family, window slice, and retained-boundary neighborhood
  before another production root fix.

## Checkpoint Phase 24 - prior-direction target-covariance slice instrumentation

Status: implementation-only / diagnostic wiring.

Hypothesis:
- The rejected final-head alignment loss may fail because the aggregate
  prior-sensitive direction is target-conflicting or selector-immaterial by
  head, query family, footprint family, or window slice.
- Existing artifacts lack per-row prior-induced logit/hidden deltas for those
  slices, so a focused diagnostic-only payload extension is required before
  another production root fix.

Changed files:
- `learning/factorized_head_diagnostics.py`
- `learning/model_training.py`
- `orchestration/diagnostics/semantic_causality_diagnostic.py`
- `tests/unit/learning/test_query_local_utility_training.py`
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `.venv/bin/python -m py_compile Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/learning/model_training.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `.venv/bin/ruff check Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/learning/model_training.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `(cd Range_QDS && ../.venv/bin/pyright learning/factorized_head_diagnostics.py learning/model_training.py orchestration/diagnostics/semantic_causality_diagnostic.py tests/unit/learning/test_query_local_utility_training.py tests/unit/orchestration/test_semantic_causality_diagnostic.py)`
- `.venv/bin/pytest Range_QDS/tests/unit/learning/test_query_local_utility_training.py -q`
- `.venv/bin/pytest Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py -q`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2549 --n_ships 12 --n_points 96 --synthetic_route_families 3 --n_queries 8 --max_queries 64 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 12000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 2 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/prior_direction_target_covariance_slice_diagnosis_level1_smoke)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/prior_direction_target_covariance_slice_diagnosis_level1_smoke/example_run.json --output artifacts/results/prior_direction_target_covariance_slice_diagnosis_level1_smoke/semantic_diagnostic.json)`
- `git diff --check`

Artifact:
- `artifacts/results/prior_direction_target_covariance_slice_diagnosis_level1_smoke/example_run.json`
- `artifacts/results/prior_direction_target_covariance_slice_diagnosis_level1_smoke/semantic_diagnostic.json`

Scale:
- static/unit plus Level 1 wiring smoke, seed `2549`
- `n_ships=12`, `n_points=96`, `n_queries=8`, `max_queries=64`,
  `epochs=1`, `range_train_workload_replicates=2`

Key results:
- The new diagnostic fields are present in both raw and semantic artifacts:
  `output_layer_alignment.slice_alignment.groups` contains `anchor_family`,
  `footprint_family`, and `window_slice`.
- Level 1 scores are wiring-only and non-promotable: MLQDS `0.0479948573`,
  uniform `0.0409533149`, Douglas-Peucker `0.0609564009`.
- Smoke gates fail as expected: workload stability, predictability, workload
  signature, learning causality, global sanity, and final success are false.
- Prior child gates remain exactly failed:
  - shuffled-prior delta `0.0`
  - no-query-prior delta `0.0`
- Prior-to-head transfer still classifies all six heads as
  `output_layer_suppresses_prior_direction`.
- Smoke-scale global target/logit-delta Spearman is sign-conflicted:
  query-hit `0.182657`, behavior `0.138766`, boundary `-0.453338`,
  replacement `0.232913`, segment-budget `-0.158855`, path-length `-0.239709`.
- Window slices expose cancellation:
  query-hit start `0.317451`, middle `0.200608`, end `-0.170491`;
  segment-budget start `0.031358`, middle `-0.211255`, end `-0.421283`.
- Retained-row context still shows selector/materiality problems:
  `high_marginal_under_ranked_by_scores=11`,
  `raw_score_good_but_segment_allocation_loses_it=7`, and
  `head_positive_but_final_score_suppresses_it=20`.

Decision:
- This validates missing diagnostic observability only. It does not update the
  strict evidence boundary and must not be used for a root-fix claim.
- The Level 1 readout supports the sign-conflict hypothesis, but smoke-scale
  numbers are not scientific evidence.
- Do not run a production root fix, tune final-head alignment weight, or repeat
  rejected prior-scale/adapter/selector paths from this artifact.
- Next admissible step: `prior_direction_target_covariance_slice_level2_replay`.
  Run one strict Level 2 diagnostic replay of the unchanged current stack with
  the new fields before designing another production change.

## Checkpoint Phase 25 - prior-direction target-covariance Level 2 replay

Status: accepted-as-diagnostic-boundary.

Hypothesis:
- The Level 1 covariance-slice readout is not scientific evidence; the unchanged
  current production stack needs a strict Level 2 replay with the new
  diagnostic-only slice fields.
- If prior child gates stay at zero and slice covariance is weak or wrong-way,
  the next step is channel-level decomposition, not another root fix.

Changed files:
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2539 --n_ships 32 --n_points 192 --synthetic_route_families 3 --n_queries 24 --max_queries 192 --epochs 4 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 30000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/prior_direction_target_covariance_slice_level2_seed2539)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/prior_direction_target_covariance_slice_level2_seed2539/example_run.json --output artifacts/results/prior_direction_target_covariance_slice_level2_seed2539/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/prior_direction_target_covariance_slice_level2_seed2539/example_run.json`
- `artifacts/results/prior_direction_target_covariance_slice_level2_seed2539/semantic_diagnostic.json`

Scale:
- Level 2 minimum strict, synthetic, source-stratified, seed `2539`
- `n_ships=32`, `n_points=192`, `n_queries=24`, `max_queries=192`,
  `epochs=4`, `range_train_workload_replicates=4`

Key results:
- Scores match the existing strict Level 2 boundary: MLQDS `0.1090154720`,
  uniform `0.0992909061`, Douglas-Peucker `0.1182249577`.
- The run beats uniform but loses to Douglas-Peucker.
- Gates pass workload stability, support overlap, target diffusion,
  predictability, and prior-predictive alignment.
- Blocking gates remain workload signature, learning causality, global sanity,
  and final success.
- Learning causality remains blocked only on prior child gates:
  - shuffled-score delta `0.0258081729`
  - shuffled-prior delta `0.0`
  - no-query-prior delta `0.0`
  - no-behavior-head delta `0.0100695759`
  - no-segment-budget delta `0.0232512395`
  - prior-field-only delta `0.0112427915`
- Prior-to-head transfer still classifies all six heads as
  `output_layer_suppresses_prior_direction`.
- Output-layer projection remains small:
  query-hit `0.053839`, behavior `0.047355`, boundary `0.128005`,
  replacement `0.098635`, segment-budget `0.049431`, path-length `0.190880`.
- Strict target/logit-delta covariance is weak or wrong-way:
  query-hit Spearman `0.039722`, behavior `0.176944`, boundary `0.221929`,
  replacement `0.015682`, segment-budget `-0.231059`, path-length `0.018200`.
- Slice conflicts localize the issue:
  query-hit is wrong-way for sparse-background-control `-0.193479`,
  large-context `-0.057120`, and window-start `-0.183080`; segment-budget is
  wrong-way in every anchor/footprint slice and window-start/window-end.
- Retained-marginal alignment remains weak: raw-score Spearman `0.038824`,
  selector-score Spearman `0.033110`, segment-score Spearman `-0.069131`.

Decision:
- Accept this as the current strict diagnostic boundary for prior-direction
  target-covariance localization, not as promotion evidence.
- The all-prior direction is not a clean target-aligned signal. It is weak for
  query-hit/replacement/path and wrong-way for segment-budget. This explains
  why direct final-head alignment can move projection ratios without moving
  retained masks.
- Do not run Level 3 or the final grid. Do not implement another aggregate
  final-head alignment loss, prior-scale boost, prior adapter, or selector
  compensation.
- Next admissible step: `prior_channel_direction_decomposition_diagnosis`.
  Decompose prior-induced head-output movement by prior channel and slice before
  any production root fix.

## Checkpoint Phase 26 - prior-channel direction decomposition

Status: accepted-as-diagnostic-boundary.

Hypothesis:
- The weak/wrong-way all-prior direction is caused by channel-specific conflicts
  rather than missing prior input support.
- If specific prior channels push heads in opposite target directions while
  score deltas remain mask-immaterial, the next step is a channel-aware root
  fix, not another aggregate prior boost or selector change.

Changed files:
- `learning/factorized_head_diagnostics.py`
- `orchestration/diagnostics/semantic_causality_diagnostic.py`
- `tests/unit/learning/test_query_local_utility_training.py`
- `tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `.venv/bin/python -m py_compile Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/learning/model_training.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `.venv/bin/ruff check Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/learning/model_training.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `(cd Range_QDS && ../.venv/bin/pyright learning/factorized_head_diagnostics.py learning/model_training.py orchestration/diagnostics/semantic_causality_diagnostic.py tests/unit/learning/test_query_local_utility_training.py tests/unit/orchestration/test_semantic_causality_diagnostic.py)`
- `.venv/bin/pytest Range_QDS/tests/unit/learning/test_query_local_utility_training.py -q`
- `.venv/bin/pytest Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py -q`
- `git diff --check`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2550 --n_ships 12 --n_points 96 --synthetic_route_families 3 --n_queries 8 --max_queries 64 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 12000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 2 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/prior_channel_direction_decomposition_level1_smoke)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/prior_channel_direction_decomposition_level1_smoke/example_run.json --output artifacts/results/prior_channel_direction_decomposition_level1_smoke/semantic_diagnostic.json)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2539 --n_ships 32 --n_points 192 --synthetic_route_families 3 --n_queries 24 --max_queries 192 --epochs 4 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 30000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/prior_channel_direction_decomposition_level2_seed2539)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/prior_channel_direction_decomposition_level2_seed2539/example_run.json --output artifacts/results/prior_channel_direction_decomposition_level2_seed2539/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/prior_channel_direction_decomposition_level1_smoke/example_run.json`
- `artifacts/results/prior_channel_direction_decomposition_level1_smoke/semantic_diagnostic.json`
- `artifacts/results/prior_channel_direction_decomposition_level2_seed2539/example_run.json`
- `artifacts/results/prior_channel_direction_decomposition_level2_seed2539/semantic_diagnostic.json`

Scale:
- static/unit plus Level 1 wiring smoke, seed `2550`
- Level 2 minimum strict, synthetic, source-stratified, seed `2539`
- Level 2 dimensions: `n_ships=32`, `n_points=192`, `n_queries=24`,
  `max_queries=192`, `epochs=4`, `range_train_workload_replicates=4`

Key results:
- Level 1 wiring fields are present in raw and semantic artifacts:
  `prior_channel_direction_decomposition.available=true`, `channel_count=6`,
  and semantic pass-through includes per-channel/per-head output alignment.
- Strict scores match the existing Level 2 boundary: MLQDS `0.1090154720`,
  uniform `0.0992909061`, Douglas-Peucker `0.1182249577`.
- The strict run still beats uniform and loses to Douglas-Peucker.
- Gates pass workload stability, support overlap, target diffusion,
  predictability, and prior-predictive alignment.
- Blocking gates remain workload signature, learning causality, global sanity,
  and final success.
- Prior child gates remain exactly failed:
  - shuffled-prior delta `0.0`
  - no-query-prior delta `0.0`
- Channel ablation materiality remains too small for masks:
  - max channel score-output delta `0.000245419`
  - every channel has score top-k Jaccard `1.0`
  - retained masks are unchanged for all channels except spatiotemporal
    query-hit prior, which still has top-k Jaccard `1.0`
- Channel/head classifications over 36 pairs:
  - target-aligned `14`
  - wrong-way `10`
  - weak/flat `6`
  - rank-unavailable `6`
- Query-hit channel directions are conflicted:
  - aligned: behavior-utility prior `0.307249`, endpoint likelihood `0.257721`
  - wrong-way: crossing likelihood `-0.274167`, spatiotemporal query-hit
    `-0.228612`, spatial query-hit `-0.074547`
- Behavior-head channel directions are mixed but not front-door blocked:
  crossing likelihood `0.169176`, spatiotemporal query-hit `0.145892`,
  behavior-utility prior `0.069941`, endpoint likelihood `-0.132445`, spatial
  query-hit `0.025763`.
- Segment-budget channel directions are sharply conflicted:
  - aligned: endpoint likelihood `0.534222`, behavior-utility prior `0.056547`
  - wrong-way: spatiotemporal query-hit `-0.417687`, crossing likelihood
    `-0.260276`, spatial query-hit `-0.117056`
- Route-density prior changes at sampling (`0.0920908`) but is disabled by the
  model transform: normalized model-prior delta `0.0`, head delta `0.0`.

Decision:
- Accept this as the current strict diagnostic boundary for channel-level
  prior-direction localization, not as promotion evidence.
- The active blocker is now classified as channel-conflicted prior integration:
  prior support exists, but head output layers mix channels with incompatible
  target directions and the resulting score movement is too small to alter
  retained masks.
- Do not run Level 3 or the final grid.
- Do not repeat scalar prior boosts, generic prior residuals, route-density
  exposure, selector floors/tricks, prior-only auxiliary replay,
  primary-minus-zero-prior contrastive replay, prior-conditioned head-input
  projection, final-head cosine/decision-surface alignment, or the rejected
  prior-head adapter.
- Next admissible step: `channel_aware_prior_integration_root_fix`. Design a
  narrow production fix for channel-aware prior integration / output-direction
  credit assignment, then restart at Level 1 wiring and Level 2 strict evidence.

## Checkpoint Phase 27 - channel-aware prior-path isolation

Status: rejected.

Hypothesis:
- Prior channels are being contaminated by the generic point encoder and then
  collapsed through one dense prior projection.
- Excluding prior columns from the generic point encoder and using a
  channel-factorized prior encoder should preserve per-channel evidence without
  a direct logit adapter, scalar boost, selector change, or target change.

Changed files:
- No production changes retained. The transient model/checkpoint/test changes
  were cleaned up after the Level 1 stop condition fired.
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/models/workload_blind_range.py Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/learning/checkpoints.py Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py Range_QDS/tests/unit/learning/test_model_features.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check Range_QDS/models/workload_blind_range.py Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/learning/checkpoints.py Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py Range_QDS/tests/unit/learning/test_model_features.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pyright -p Range_QDS --pythonpath /home/aleks_dev/dev_projects/P8/.venv/bin/python`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pytest Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py::test_workload_blind_range_has_dedicated_prior_feature_encoder Range_QDS/tests/unit/learning/test_model_features.py::test_workload_blind_range_checkpoint_accepts_missing_prior_feature_encoder Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_prior_feature_learning_diagnostic_reports_model_stage_sensitivity -q`
- `git diff --check`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2551 --n_ships 12 --n_points 96 --synthetic_route_families 3 --n_queries 8 --max_queries 64 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 12000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 2 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/channel_aware_prior_integration_level1_smoke)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/channel_aware_prior_integration_level1_smoke/example_run.json --output artifacts/results/channel_aware_prior_integration_level1_smoke/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/channel_aware_prior_integration_level1_smoke/example_run.json`
- `artifacts/results/channel_aware_prior_integration_level1_smoke/semantic_diagnostic.json`

Scale:
- static/unit plus Level 1 wiring smoke, seed `2551`
- `n_ships=12`, `n_points=96`, `n_queries=8`, `max_queries=64`,
  `epochs=1`, `range_train_workload_replicates=2`

Key results:
- Smoke scores: MLQDS `0.0846832700`, uniform `0.0595670876`,
  Douglas-Peucker `0.0632635823`.
- The smoke score beat both baselines, but that is not acceptance evidence.
- Learning causality still failed:
  - shuffled-score delta `0.0343491079`
  - shuffled-prior delta `0.0`
  - no-query-prior delta `0.0`
  - no-behavior-head delta `0.0035090672`
  - no-segment-budget delta `0.0079736130`
  - prior-field-only delta `-0.0068792957`
- Prior-flow after the transient change:
  - point-encoder prior delta `0.0`
  - explicit prior encoder mean abs delta `0.0018704315`
  - shared embedding mean abs delta `0.0013239448`
  - head probability mean abs delta `0.0000620892`
  - score-output mean abs delta `0.0006578953`
  - score-output max abs delta `0.0105263591`
  - score top-k Jaccard `1.0`
  - retained mask changed `false`
- Channel/head classifications worsened at smoke scale: target-aligned `10`,
  wrong-way `12`, weak/flat `8`, rank-unavailable `6`.

Decision:
- Reject this root fix. It removed generic point-encoder prior contamination,
  but the required prior child gates stayed exactly zero and retained masks did
  not move.
- Do not promote or retain the transient production change. Do not repeat
  channel-factorized prior encoding plus point-encoder prior masking without a
  materially new diagnosis.
- The strict evidence boundary remains
  `artifacts/results/prior_channel_direction_decomposition_level2_seed2539/example_run.json`.
- Next admissible step: `prior_score_rank_margin_boundary_diagnosis`. Diagnose
  whether prior-induced score deltas are below selector rank margins, point the
  wrong way for missed high-marginal points, or are being lost in segment
  allocation before attempting another production root fix.

## Checkpoint Phase 28 - prior score-rank margin boundary wiring

Status: implementation-only.

Hypothesis:
- Prior-induced score movement may exist but fail because it does not cross the
  score/selector boundary, or because it does not help missed high-marginal
  points.
- The current artifacts did not expose per-point prior score deltas on
  retained-marginal rows, so diagnostic-only instrumentation was required before
  any production root fix.

Changed files:
- `orchestration/causality.py`
- `orchestration/retained_mask_ablation_stage.py`
- `orchestration/diagnostics/semantic_causality_diagnostic.py`
- `tests/unit/orchestration/test_query_driven_causality_and_summary.py`
- `tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/orchestration/causality.py Range_QDS/orchestration/retained_mask_ablation_stage.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_causality_and_summary.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/orchestration/causality.py Range_QDS/orchestration/retained_mask_ablation_stage.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_causality_and_summary.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pytest Range_QDS/tests/unit/orchestration/test_query_driven_causality_and_summary.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py -q`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pyright -p Range_QDS --pythonpath /home/aleks_dev/dev_projects/P8/.venv/bin/python`
- `git diff --check`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2552 --n_ships 12 --n_points 96 --synthetic_route_families 3 --n_queries 8 --max_queries 64 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 12000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 2 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/prior_score_rank_margin_boundary_level1_smoke)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/prior_score_rank_margin_boundary_level1_smoke/example_run.json --output artifacts/results/prior_score_rank_margin_boundary_level1_smoke/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/prior_score_rank_margin_boundary_level1_smoke/example_run.json`
- `artifacts/results/prior_score_rank_margin_boundary_level1_smoke/semantic_diagnostic.json`

Scale:
- static/unit plus Level 1 wiring smoke, seed `2552`
- `n_ships=12`, `n_points=96`, `n_queries=8`, `max_queries=64`,
  `epochs=1`, `range_train_workload_replicates=2`

Key results:
- Smoke scores: MLQDS `0.0840625269`, uniform `0.0541400848`,
  Douglas-Peucker `0.0760089962`.
- Learning causality still fails the prior child gates:
  - shuffled-score delta `0.0213494218`
  - shuffled-prior delta `0.0`
  - no-query-prior delta `0.0`
  - no-behavior-head delta `-0.0044226151`
  - no-segment-budget delta `0.0170002879`
  - prior-field-only delta `0.0218526280`
- New score-rank boundary fields are present in raw and semantic artifacts.
- For `without_query_prior_features`, retained mask changed `false`,
  score-output mean abs delta `0.0008040933`, top-k score boundary margin
  `0.0105262995`, max abs score delta / top-k margin `2.0000028312`,
  score-delta/marginal Spearman `0.0965514615`, and score deltas did not cross
  the top-k boundary.
- For `without_query_prior_features`, top-marginal, missed-high marginal, and
  under-ranked high-marginal mean score deltas are all `0.0`; classification is
  `prior_delta_non_positive_for_top_marginal_rows`.
- `shuffled_prior_fields` shows the same retained-mask invariance and
  non-positive high-marginal classification, with score-output mean abs delta
  `0.0008771929`.

Decision:
- Accept this only as diagnostic wiring. It does not update the strict evidence
  boundary and is not learning evidence.
- The Level 1 smoke does not support another selector trick: prior deltas can be
  nonzero, but they do not help the high-marginal missed/under-ranked rows and
  do not change retained masks.
- Do not run Level 3 or the final grid. Do not implement a production root fix
  from this Level 1 artifact.
- Next admissible step: `prior_score_rank_margin_boundary_level2_replay`. Run
  one strict Level 2 replay of the unchanged production stack with the new
  rank-margin fields before deciding whether the next root fix belongs in
  prior/head semantics, score composition, or segment allocation.

## Checkpoint Phase 29 - prior score-rank margin Level 2 replay

Status: accepted-as-diagnostic-boundary.

Hypothesis:
- The Level 1 rank-margin readout is only wiring evidence; the unchanged
  current production stack needs a strict Level 2 replay with the new
  diagnostic-only rank-margin fields.
- If required prior ablations still do not move retained masks or high-marginal
  rows, the next step is row-level prior-delta path localization, not selector
  tuning or another production root fix.

Changed files:
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2539 --n_ships 32 --n_points 192 --synthetic_route_families 3 --n_queries 24 --max_queries 192 --epochs 4 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 30000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/prior_score_rank_margin_boundary_level2_seed2539)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/prior_score_rank_margin_boundary_level2_seed2539/example_run.json --output artifacts/results/prior_score_rank_margin_boundary_level2_seed2539/semantic_diagnostic.json)`
- `git diff --check`

Artifact:
- `artifacts/results/prior_score_rank_margin_boundary_level2_seed2539/example_run.json`
- `artifacts/results/prior_score_rank_margin_boundary_level2_seed2539/semantic_diagnostic.json`

Scale:
- Level 2 minimum strict, synthetic, source-stratified, seed `2539`
- `n_ships=32`, `n_points=192`, `n_queries=24`, `max_queries=192`,
  `epochs=4`, `range_train_workload_replicates=4`

Key results:
- Scores match the existing strict Level 2 boundary: MLQDS `0.1090154720`,
  uniform `0.0992909061`, Douglas-Peucker `0.1182249577`, Oracle
  `0.3095599786`.
- The run beats uniform but loses to Douglas-Peucker.
- Gates pass workload stability, support overlap, target diffusion,
  predictability, and prior-predictive alignment.
- Blocking gates remain workload signature, learning causality, global sanity,
  and final success.
- Learning-causality child state:
  - shuffled-score delta `0.0258081729`
  - untrained delta `0.0127838141`
  - shuffled-prior delta `0.0`
  - no-query-prior delta `0.0`
  - no-behavior-head delta `0.0100695759`
  - no-segment-budget delta `0.0232512395`
  - prior-field-only delta `0.0112427915`
- `without_query_prior_features`:
  - sampled/model priors change; head-probability mean abs delta
    `0.0000489547`; score-output mean abs delta `0.0002931391`
  - retained mask changed `false`; score top-k Jaccard `1.0`
  - top-k boundary margin `0.0052356124`; max abs score delta / margin
    `1.9999943078`; score deltas do not cross the top-k boundary
  - score-delta/marginal Spearman `0.0898510676`
  - top-marginal, missed-high marginal, and under-ranked high-marginal mean
    score deltas are all `0.0`
  - classification `prior_delta_non_positive_for_top_marginal_rows`
- `shuffled_prior_fields`:
  - retained mask changed `false`; score top-k Jaccard `1.0`
  - score-output mean abs delta `0.0003067735`; head-probability mean abs delta
    `0.0000483237`
  - score-delta/marginal Spearman `0.1042289998`
  - top-marginal, missed-high marginal, and under-ranked high-marginal mean
    score deltas are all `0.0`
  - classification `prior_delta_non_positive_for_top_marginal_rows`
- Channel note: only the spatiotemporal query-hit prior channel changed a
  retained mask, but its QueryLocalUtility delta was negative
  (`-0.0005847486`) and still did not help high-marginal rows.
- Segment-score calibration remains a separate blocker: raw-score Spearman
  `0.0388235897`, selector-score Spearman `0.0331097480`, segment-score
  Spearman `-0.0691307616`; semantic classification remains
  `allocation scoring and point-selection scoring are mixed incorrectly`.

Decision:
- Accept this as the current strict diagnostic boundary for score-to-mask
  prior-materiality localization. It is not promotion evidence.
- The rank-margin evidence rejects a selector-boundary workaround as the next
  move. Required prior ablations do not move retained masks, and prior-induced
  score deltas are exactly non-positive on the high-marginal rows that matter.
- The artifact still cannot distinguish whether high-marginal rows lose prior
  movement at head-logit/head-probability, raw-score, score-output, or
  segment-score composition.
- Do not run Level 3 or the final grid. Do not implement another prior-scale,
  residual, final-head alignment, channel-factorized encoder, or selector
  allocation fix from this artifact.
- Next admissible step: `prior_marginal_row_delta_path_diagnosis`. Add or derive
  diagnostic-only retained-marginal row deltas across head-logit,
  head-probability, raw-score, score-output, segment-score, and retained-mask
  stages.

## Checkpoint Phase 30 - prior marginal row delta path wiring

Status: implementation-only.

Hypothesis:
- High-marginal retained-decision rows either receive little prior-induced
  movement before selector, or that movement is erased between head/raw outputs
  and final selector scores.
- The strict rank-margin artifact did not expose row-level head/raw/segment
  deltas, so diagnostic-only instrumentation was required before any root fix.

Changed files:
- `orchestration/causality.py`
- `orchestration/retained_mask_ablation_stage.py`
- `orchestration/diagnostics/semantic_causality_diagnostic.py`
- `tests/unit/orchestration/test_query_driven_causality_and_summary.py`
- `tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/orchestration/causality.py Range_QDS/orchestration/retained_mask_ablation_stage.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_causality_and_summary.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/orchestration/causality.py Range_QDS/orchestration/retained_mask_ablation_stage.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_causality_and_summary.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pytest Range_QDS/tests/unit/orchestration/test_query_driven_causality_and_summary.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py -q`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pyright -p Range_QDS --pythonpath /home/aleks_dev/dev_projects/P8/.venv/bin/python`
- `git diff --check`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2553 --n_ships 12 --n_points 96 --synthetic_route_families 3 --n_queries 8 --max_queries 64 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 12000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 2 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/prior_marginal_row_delta_path_level1_smoke)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/prior_marginal_row_delta_path_level1_smoke/example_run.json --output artifacts/results/prior_marginal_row_delta_path_level1_smoke/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/prior_marginal_row_delta_path_level1_smoke/example_run.json`
- `artifacts/results/prior_marginal_row_delta_path_level1_smoke/semantic_diagnostic.json`

Scale:
- static/unit plus Level 1 wiring smoke, seed `2553`
- `n_ships=12`, `n_points=96`, `n_queries=8`, `max_queries=64`,
  `epochs=1`, `range_train_workload_replicates=2`

Key results:
- Smoke scores: MLQDS `0.0724735873`, uniform `0.0650750590`,
  Douglas-Peucker `0.0653689653`.
- Learning causality still fails the required prior child gates:
  - shuffled-score delta `0.0256370016`
  - shuffled-prior delta `0.0`
  - no-query-prior delta `0.0`
  - no-behavior-head delta `-0.0000301816`
  - no-segment-budget delta `0.0002001039`
  - prior-field-only delta `0.0048668701`
- New row-delta path fields are present in raw and semantic artifacts for
  `without_query_prior_features` and `shuffled_prior_fields`.
- For `without_query_prior_features`, all row-path stages are available;
  row count is `79`; classification is
  `score_composition_suppresses_positive_raw_delta`.
- For `without_query_prior_features`, top-marginal raw-score mean delta is
  `0.0000995874`, missed-high raw-score mean delta is `0.0001410246`, and
  under-ranked-high raw-score mean delta is `0.0002173696`, but top-marginal
  and missed-high score-output mean deltas are both `0.0`.
- For `without_query_prior_features`, top-marginal segment-score mean delta is
  `-0.0002221987`.
- `shuffled_prior_fields` shows the same classification: top-marginal
  raw-score mean delta `0.0001029253`, missed-high raw-score mean delta
  `0.0001446307`, score-output mean deltas `0.0`, and top-marginal
  segment-score mean delta `-0.0002177536`.

Decision:
- Accept this only as diagnostic wiring. It does not update the strict evidence
  boundary and is not learning evidence.
- The smoke readout points to score composition/rank conversion erasing small
  positive high-marginal raw-score movement before score-output, with segment
  scoring also moving the wrong way. This is not strict proof.
- Do not run Level 3 or the final grid. Do not implement a production root fix
  from this Level 1 artifact.
- Next admissible step: `prior_marginal_row_delta_path_level2_replay`. Run one
  strict Level 2 replay of the unchanged production stack with these row-path
  fields preserved before deciding on a root fix.

## Checkpoint Phase 31 - prior marginal row delta path Level 2 replay

Status: accepted-as-diagnostic-boundary.

Hypothesis:
- The Level 1 row-path readout is only wiring evidence; the unchanged current
  production stack needs a strict Level 2 replay with the row-delta path fields.
- If required prior ablations still do not move masks, use the strict row-path
  stage localization rather than tuning selector boundaries or running Level 3.

Changed files:
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2539 --n_ships 32 --n_points 192 --synthetic_route_families 3 --n_queries 24 --max_queries 192 --epochs 4 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 30000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/prior_marginal_row_delta_path_level2_seed2539)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/prior_marginal_row_delta_path_level2_seed2539/example_run.json --output artifacts/results/prior_marginal_row_delta_path_level2_seed2539/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/prior_marginal_row_delta_path_level2_seed2539/example_run.json`
- `artifacts/results/prior_marginal_row_delta_path_level2_seed2539/semantic_diagnostic.json`

Scale:
- Level 2 minimum strict, synthetic, source-stratified, seed `2539`
- `n_ships=32`, `n_points=192`, `n_queries=24`, `max_queries=192`,
  `epochs=4`, `range_train_workload_replicates=4`

Key results:
- Scores match the existing strict Level 2 boundary: MLQDS `0.1090154720`,
  uniform `0.0992909061`, Douglas-Peucker `0.1182249577`, Oracle
  `0.3095599786`.
- The run beats uniform but loses to Douglas-Peucker.
- Gates pass workload stability, support overlap, target diffusion,
  predictability, and prior-predictive alignment.
- Blocking gates remain workload signature, learning causality, global sanity,
  and final success.
- Learning-causality child state:
  - shuffled-score delta `0.0258081729`
  - untrained delta `0.0127838141`
  - shuffled-prior delta `0.0`
  - no-query-prior delta `0.0`
  - no-behavior-head delta `0.0100695759`
  - no-segment-budget delta `0.0232512395`
  - prior-field-only delta `0.0112427915`
- New strict row-path fields are present for `without_query_prior_features` and
  `shuffled_prior_fields`; both have all stages available and row count `112`.
- `without_query_prior_features` row-path classification:
  `raw_score_suppresses_positive_head_probability_delta`.
  - top-marginal raw-score mean delta `-0.0001583270`
  - missed-high raw-score mean delta `-0.0002422598`
  - under-ranked-high raw-score mean delta `-0.0003421042`
  - top/missed/under-ranked score-output mean deltas all `0.0`
  - top-marginal segment-score mean delta `-0.0001898493`
  - top-marginal max positive head-probability delta `0.0001161907` from
    `replacement_representative_value`
- `shuffled_prior_fields` shows the same classification.
  - top-marginal raw-score mean delta `-0.0001629421`
  - missed-high raw-score mean delta `-0.0002477169`
  - under-ranked-high raw-score mean delta `-0.0003356934`
  - top/missed/under-ranked score-output mean deltas all `0.0`
  - top-marginal segment-score mean delta `-0.0001886359`
  - top-marginal max positive head-probability delta `0.0001134766` from
    `replacement_representative_value`
- The strict readout contradicts the Level 1 smoke direction. High-marginal raw
  deltas are negative at strict scale, not positive.
- Segment-score calibration remains wrong-way: raw-score Spearman `0.0388236`,
  selector-score Spearman `0.0331097`, segment-score Spearman `-0.0691308`;
  semantic classification remains `allocation scoring and point-selection
  scoring are mixed incorrectly`.

Decision:
- Accept this as the current strict diagnostic boundary for retained-marginal
  prior row-path localization. It is not promotion evidence.
- The blocker is now narrower than selector rank margin: required prior
  ablations move some heads in positive directions, mainly replacement and
  boundary, but the composed raw score moves high-marginal rows the wrong way;
  score-output then remains unchanged and masks stay fixed.
- Do not run Level 3 or the final grid. Do not implement a selector workaround,
  scalar prior boost, generic residual, route-density exposure, final-head
  alignment replay, or channel-factorized replay from this artifact.
- Next admissible step: `factorized_prior_delta_composition_diagnosis`. Derive
  or instrument diagnostic-only per-row contribution deltas for the factorized
  raw score formula before any production root fix.

## Checkpoint Phase 32 - factorized prior-delta composition wiring

Status: implementation-only.

Hypothesis:
- Strict high-marginal raw-score deltas may go negative because the
  multiplicative query-hit / behavior / replacement term loses more than the
  positive replacement or boundary movement adds.
- The strict row-path artifact did not expose exact factorized contribution
  terms for all retained-marginal groups, so diagnostic-only instrumentation
  was required before any formula, target, selector, or model change.

Changed files:
- `orchestration/causality.py`
- `orchestration/diagnostics/semantic_causality_diagnostic.py`
- `tests/unit/orchestration/test_query_driven_causality_and_summary.py`
- `tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/orchestration/causality.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_causality_and_summary.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/orchestration/causality.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_causality_and_summary.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pytest Range_QDS/tests/unit/orchestration/test_query_driven_causality_and_summary.py::test_prior_ablation_sensitivity_from_tensors_builds_consistent_chain Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py::test_semantic_causality_diagnostic_classifies_current_blockers -q`
- `git diff --check`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2554 --n_ships 12 --n_points 96 --synthetic_route_families 3 --n_queries 8 --max_queries 64 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 12000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 2 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/factorized_prior_delta_composition_level1_smoke)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/factorized_prior_delta_composition_level1_smoke/example_run.json --output artifacts/results/factorized_prior_delta_composition_level1_smoke/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/factorized_prior_delta_composition_level1_smoke/example_run.json`
- `artifacts/results/factorized_prior_delta_composition_level1_smoke/semantic_diagnostic.json`

Scale:
- static/unit plus Level 1 wiring smoke, seed `2554`
- `n_ships=12`, `n_points=96`, `n_queries=8`, `max_queries=64`,
  `epochs=1`, `range_train_workload_replicates=2`

Key results:
- Smoke scores: MLQDS `0.0913145491`, uniform `0.0818613758`,
  Douglas-Peucker `0.0736158811`.
- Learning causality still fails required child gates:
  - shuffled-prior delta `0.0`
  - no-query-prior delta `0.0`
  - no-segment-budget delta `0.0028147276`
- New factorized composition fields are present in raw and semantic artifacts
  for `without_query_prior_features` and `shuffled_prior_fields`.
- `without_query_prior_features` smoke composition readout:
  - top-marginal composed-logit mean delta `-0.0003796321`
  - top-marginal composed-score mean delta `-0.0000037099`
  - top-marginal raw residual mean `0.0000000220`
  - top-marginal dominant negative contribution
    `query_hit_product_shapley=-0.0000035423`
  - missed-high dominant negative contribution
    `query_hit_product_shapley=-0.0000040692`
  - under-ranked-high dominant negative contribution
    `query_hit_product_shapley=-0.0000020308`
- `shuffled_prior_fields` shows the same wiring pattern:
  - top-marginal dominant negative contribution
    `query_hit_product_shapley=-0.0000036284`
  - missed-high dominant negative contribution
    `query_hit_product_shapley=-0.0000041154`
  - under-ranked-high dominant negative contribution
    `query_hit_product_shapley=-0.0000020374`

Decision:
- Accept this only as diagnostic wiring. It does not update the strict evidence
  boundary and is not learning evidence.
- The composition terms are now available and internally consistent: composed
  logit deltas match raw-prediction deltas up to tiny residuals.
- The smoke points at query-hit product contribution as the dominant negative
  term, but this is Level 1 only.
- Do not run Level 3 or the final grid. Do not change the factorized formula,
  head weights, selector, or prior integration from this smoke.
- Next admissible step: `factorized_prior_delta_composition_level2_replay`.
  Run one strict Level 2 replay of the unchanged production stack with these
  composition fields preserved.

## Checkpoint Phase 33 - factorized composition Level 2 replay

Status: accepted-as-diagnostic-boundary.

Hypothesis:
- The Level 1 factorized composition readout is only wiring evidence; the
  unchanged current production stack needs a strict Level 2 replay with the
  composition fields preserved.
- If `query_hit_product_shapley` remains the dominant negative term, the next
  diagnosis belongs in query-hit product direction, not selector boundary
  tuning or a factorized formula change.

Changed files:
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2539 --n_ships 32 --n_points 192 --synthetic_route_families 3 --n_queries 24 --max_queries 192 --epochs 4 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 30000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/factorized_prior_delta_composition_level2_seed2539)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/factorized_prior_delta_composition_level2_seed2539/example_run.json --output artifacts/results/factorized_prior_delta_composition_level2_seed2539/semantic_diagnostic.json)`
- `jq empty Range_QDS/artifacts/results/factorized_prior_delta_composition_level2_seed2539/example_run.json Range_QDS/artifacts/results/factorized_prior_delta_composition_level2_seed2539/semantic_diagnostic.json`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/orchestration/causality.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_causality_and_summary.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/orchestration/causality.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_causality_and_summary.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pytest Range_QDS/tests/unit/orchestration/test_query_driven_causality_and_summary.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py -q`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pyright -p Range_QDS --pythonpath /home/aleks_dev/dev_projects/P8/.venv/bin/python`
- `git diff --check`

Artifact:
- `artifacts/results/factorized_prior_delta_composition_level2_seed2539/example_run.json`
- `artifacts/results/factorized_prior_delta_composition_level2_seed2539/semantic_diagnostic.json`

Scale:
- Level 2 minimum strict, synthetic, source-stratified, seed `2539`
- `n_ships=32`, `n_points=192`, `n_queries=24`, `max_queries=192`,
  `epochs=4`, `range_train_workload_replicates=4`

Key results:
- Scores match the existing strict Level 2 boundary: MLQDS `0.1090154720`,
  uniform `0.0992909061`, Douglas-Peucker `0.1182249577`.
- The run beats uniform but loses to Douglas-Peucker.
- Gates pass workload stability, support overlap, target diffusion,
  predictability, and prior-predictive alignment.
- Blocking gates remain workload signature, learning causality, global sanity,
  and final success.
- Learning-causality child state:
  - shuffled-score delta `0.0258081729`
  - untrained delta `0.0127838141`
  - shuffled-prior delta `0.0`
  - no-query-prior delta `0.0`
  - no-behavior-head delta `0.0100695759`
  - no-segment-budget delta `0.0232512395`
  - prior-field-only delta `0.0112427915`
- `without_query_prior_features` keeps the strict row-path classification
  `raw_score_suppresses_positive_head_probability_delta`.
  - top raw-score mean delta `-0.0001583270`
  - top score-output mean delta `0.0`
  - top segment-score mean delta `-0.0001897471`
  - top query-hit probability mean delta `-0.0000018995`
  - top dominant negative contribution
    `query_hit_product_shapley=-0.0000009023`
  - missed-high dominant negative contribution
    `query_hit_product_shapley=-0.0000012009`
  - under-ranked-high dominant negative contribution
    `query_hit_product_shapley=-0.0000012261`
  - top dominant positive contribution
    `replacement_product_shapley=0.0000002075`
- `shuffled_prior_fields` shows the same failed term.
  - top raw-score mean delta `-0.0001630272`
  - top score-output mean delta `0.0`
  - top segment-score mean delta `-0.0001886530`
  - top query-hit probability mean delta `-0.0000019641`
  - top dominant negative contribution
    `query_hit_product_shapley=-0.0000009337`
  - missed-high dominant negative contribution
    `query_hit_product_shapley=-0.0000012420`
  - under-ranked-high dominant negative contribution
    `query_hit_product_shapley=-0.0000012065`
  - top dominant positive contribution
    `replacement_product_shapley=0.0000002025`
- Composition residuals are tiny: top raw residuals are about `1e-8` to
  `4e-8`, so the composed-logit diagnostic matches the raw-prediction path.

Decision:
- Accept this as the current strict diagnostic boundary for factorized
  prior-delta composition. It is not promotion evidence.
- The strict run confirms the Level 1 direction: query-hit product movement is
  the dominant negative term on the high-marginal rows that matter.
- Do not run Level 3 or the final grid. Do not change the factorized formula,
  query-hit target, prior path, selector, or guardrails from this checkpoint.
- Next admissible step: `query_hit_product_direction_diagnosis`. Derive why the
  query-hit product term is wrong-way on high-marginal rows before any
  production root fix.

## Checkpoint Phase 34 - query-hit product direction diagnosis

Status: completed / diagnostic-only.

Hypothesis:
- The strict composition failure is not a product-math artifact. It is caused
  by the query-hit head moving wrong-way on high-marginal rows.
- If row-level query-hit target/support fields are missing from the row-delta
  path, the next checkpoint should join or instrument those fields before any
  formula, target, model, or selector change.

Changed files:
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `jq '.learning_causality_summary.prior_sensitivity_diagnostics | with_entries(select(.key == "without_query_prior_features" or .key == "shuffled_prior_fields")) | map_values(.marginal_row_delta_path.groups | {top:{qhit_logit:.top_marginal.head_logit_mean_delta_by_head.query_hit_probability,qhit_prob:.top_marginal.head_probability_mean_delta_by_head.query_hit_probability,replacement_prob:.top_marginal.head_probability_mean_delta_by_head.replacement_representative_value,raw:.top_marginal.raw_prediction_mean_delta,contribution:.top_marginal.factorized_contribution_mean_delta},missed:{qhit_logit:.missed_high_marginal.head_logit_mean_delta_by_head.query_hit_probability,qhit_prob:.missed_high_marginal.head_probability_mean_delta_by_head.query_hit_probability,contribution:.missed_high_marginal.factorized_contribution_mean_delta},under:{qhit_logit:.under_ranked_high_marginal.head_logit_mean_delta_by_head.query_hit_probability,qhit_prob:.under_ranked_high_marginal.head_probability_mean_delta_by_head.query_hit_probability,contribution:.under_ranked_high_marginal.factorized_contribution_mean_delta}})' artifacts/results/factorized_prior_delta_composition_level2_seed2539/example_run.json`
- `jq '.artifacts[0].query_prior_materiality.training_prior_learning_signal.prior_to_head_transfer_sensitivity.prior_channel_direction_decomposition.per_channel | map_values(.per_head.query_hit_probability | {classification,spearman:.output_layer_alignment.target_to_logit_delta_spearman})' artifacts/results/factorized_prior_delta_composition_level2_seed2539/semantic_diagnostic.json`
- `jq empty artifacts/results/factorized_prior_delta_composition_level2_seed2539/example_run.json artifacts/results/factorized_prior_delta_composition_level2_seed2539/semantic_diagnostic.json`
- `git diff --check`

Artifact:
- Reused
  `artifacts/results/factorized_prior_delta_composition_level2_seed2539/example_run.json`
- Reused
  `artifacts/results/factorized_prior_delta_composition_level2_seed2539/semantic_diagnostic.json`

Scale:
- Derived strict Level 2 artifact diagnostic only.
- No new training run.

Key results:
- For `without_query_prior_features`, query-hit movement is negative on all
  high-marginal row groups:
  - top q-hit logit/probability deltas
    `-0.0001967124` / `-0.0000018995`
  - missed-high q-hit logit/probability deltas
    `-0.0002708435` / `-0.0000025330`
  - under-ranked q-hit logit/probability deltas
    `-0.0003658401` / `-0.0000026272`
- For `shuffled_prior_fields`, query-hit movement is also negative on all
  high-marginal row groups:
  - top q-hit logit/probability deltas
    `-0.0002005952` / `-0.0000019641`
  - missed-high q-hit logit/probability deltas
    `-0.0002756384` / `-0.0000026172`
  - under-ranked q-hit logit/probability deltas
    `-0.0003600650` / `-0.0000025853`
- Positive replacement movement exists but is too small after factorized
  composition. Top-row replacement probability deltas are about
  `0.000113` to `0.000116`, but replacement Shapley contribution is only about
  `0.00000020`, while query-hit Shapley contribution is about
  `-0.00000090` to `-0.00000093`.
- Strict channel evidence for the query-hit head is conflicted:
  - target-aligned: behavior-utility prior `0.3072801`,
    endpoint likelihood `0.2581299`
  - wrong-way: crossing likelihood `-0.2742706`, spatiotemporal query-hit
    prior `-0.2285822`, spatial query-hit prior `-0.0745400`
- Strict slice evidence also shows query-hit weakness/wrong-way movement:
  sparse-background-control `-0.1935252`, large-context `-0.0570811`,
  window-start `-0.1830583`.
- The row-delta path rows expose q-hit deltas and product contributions, but
  they do not directly carry `head_targets`, query-hit target masks, query-hit
  run ids, or sampled/model prior channel context. Those fields exist in the
  selector trace rows, so the next step should attempt a source/point join
  before adding more instrumentation.

Decision:
- Classify the immediate failure as `query-hit head moves wrong-way on
  high-marginal rows`; factorized composition then amplifies that small q-hit
  loss over the smaller positive replacement/boundary gains.
- This does not justify a formula change or target rewrite yet. The missing
  question is whether the high-marginal rows actually have query-hit target
  support, or whether q-hit dominance is suppressing non-hit rows that are
  valuable for QueryLocalUtility behavior/interpolation.
- Do not run Level 3 or the final grid. Do not implement a production root fix
  from this derived diagnostic alone.
- Next admissible step: `query_hit_row_target_support_diagnosis`. Join or
  instrument row-delta path groups with query-hit target/support and
  prior-channel context.

## Checkpoint Phase 35 - query-hit row target/support diagnosis

Status: completed / diagnostic-only.

Hypothesis:
- The wrong-way query-hit product movement may be correct if high-marginal rows
  lack query-hit target support, or wrong if supported rows are valuable mostly
  through local behavior/interpolation components.
- Existing artifacts may answer this by joining row-delta path rows to selector
  trace semantic rows by `source` and `point_index`.

Changed files:
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `jq '.learning_causality_summary.prior_sensitivity_diagnostics.without_query_prior_features.marginal_row_delta_path.top_marginal_rows[0:3]' artifacts/results/factorized_prior_delta_composition_level2_seed2539/example_run.json`
- `jq '.selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_local_utility_alignment.rows[0:3] | map({source, point_index, trajectory_index, head_targets, head_target_masks, query_hit_run_ids, query_family_hit_context, sampled_prior_channels, model_prior_channels, query_local_utility_component_delta, marginal_query_local_utility})' artifacts/results/factorized_prior_delta_composition_level2_seed2539/example_run.json`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python - <<'PY' ... derive group target/support and weighted component stats ... PY`

Artifact:
- Reused
  `artifacts/results/factorized_prior_delta_composition_level2_seed2539/example_run.json`
- Reused
  `artifacts/results/factorized_prior_delta_composition_level2_seed2539/semantic_diagnostic.json`

Scale:
- Derived strict Level 2 artifact diagnostic only.
- No new training run.

Key results:
- The stored top row-delta path rows join reliably to selector trace rows:
  `16/16` matched for both `without_query_prior_features` and
  `shuffled_prior_fields`.
- For stored top rows, query-hit target support is present, not absent:
  - mean q-hit target `0.0551471`
  - min q-hit target `0.0294118`
  - q-hit target positive fraction `1.0`
  - query-point-recall marginal component positive fraction `1.0`
- Reconstructing the row-delta path group definitions from selector rows shows
  the same support pattern:
  - top group: `28` rows, mean q-hit target `0.0472689`, min `0.0294118`,
    positive fraction `1.0`
  - missed-high group: `18` rows, mean q-hit target `0.0490196`, min
    `0.0294118`, positive fraction `1.0`
  - under-ranked group: `9` rows, mean/min q-hit target `0.0294118`,
    positive fraction `1.0`
- The rows are behavior-dominated relative to point recall:
  - top group weighted local behavior / weighted point recall ratio `2.0504`
  - missed-high ratio `2.0724`
  - under-ranked ratio `3.1009`
- The model prior channels are present on these rows. Direct q-hit priors are
  positive for all top/missed/under-ranked rows, while route-density remains
  disabled in model inputs as expected.

Decision:
- Classify the row-level failure as supported-but-behavior-dominated q-hit
  gating conflict, not missing query-hit support.
- High-marginal rows do have query-hit support, but their marginal value is
  mainly local behavior/interpolation/continuity. The query-hit head and direct
  q-hit prior channels move wrong-way, and the multiplicative q-hit product
  suppresses these supported local-utility rows.
- Do not run Level 3 or the final grid. Do not use a selector workaround,
  scalar prior boost, generic residual, route-density exposure, or direct head
  adapter.
- Next admissible step: `factorized_qhit_behavior_gate_root_fix_design`.
  Design one narrow root fix for supported behavior-dominated rows where q-hit
  movement suppresses local utility. Restart at static/unit and Level 1 if
  formula or target semantics change.

## Checkpoint Phase 36 - factorized q-hit behavior gate target root fix

Status: completed / implementation-only Level 1 wiring.

Hypothesis:
- The current q-hit head is trained on a tiny raw query-hit probability that is
  saturated and semantically weak for retained marginal utility.
- A narrow root fix is to make the q-hit head a train-workload
  query-evidence gate: positive mean over global and family-conditioned
  support-normalized `0.65 * query_hit_probability + 0.35 *
  ship_query_evidence`. This uses train workload labels only and does not
  change selectors, guardrails, priors, metric weights, or eval-time inputs.

Changed files:
- `learning/targets/query_local_utility.py`
- `tests/unit/learning/test_query_local_utility_targets.py`
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`
- `docs/query-driven-implementation-research-guide.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/learning/targets/query_local_utility.py Range_QDS/tests/unit/learning/test_query_local_utility_targets.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m ruff check Range_QDS/learning/targets/query_local_utility.py Range_QDS/tests/unit/learning/test_query_local_utility_targets.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m pytest Range_QDS/tests/unit/learning/test_query_local_utility_targets.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py Range_QDS/tests/unit/orchestration/test_query_driven_protocol_gates.py -q`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2555 --n_ships 12 --n_points 96 --synthetic_route_families 3 --n_queries 8 --max_queries 64 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 12000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 2 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/factorized_qhit_behavior_gate_level1_smoke)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/factorized_qhit_behavior_gate_level1_smoke/example_run.json --output artifacts/results/factorized_qhit_behavior_gate_level1_smoke/semantic_diagnostic.json)`
- `jq empty Range_QDS/artifacts/results/factorized_qhit_behavior_gate_level1_smoke/example_run.json Range_QDS/artifacts/results/factorized_qhit_behavior_gate_level1_smoke/semantic_diagnostic.json`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pyright learning/targets/query_local_utility.py tests/unit/learning/test_query_local_utility_targets.py)`
- `git diff --check`

Artifact:
- `artifacts/results/factorized_qhit_behavior_gate_level1_smoke/example_run.json`
- `artifacts/results/factorized_qhit_behavior_gate_level1_smoke/semantic_diagnostic.json`

Scale:
- Level 1 wiring smoke only, seed `2555`
- `n_ships=12`, `n_points=96`, `n_queries=8`, `max_queries=64`,
  `epochs=1`, `range_train_workload_replicates=2`

Key results:
- The target artifact confirms the new active semantics:
  `query_hit_target_variant=query_evidence_gate_hit_ship_blend`.
- New formula label:
  `query_evidence_gate_times_behavior_with_conditional_replacement_modulation_plus_boundary`.
- Query-hit target support is no longer a tiny raw probability scale:
  q-hit target std `0.168460`, q-hit head prediction std `0.004513`,
  q-hit tau `0.203795`, top-5% mass recall `0.353467`.
- Behavior remains flat at smoke scale:
  prediction std / target std `0.014378`, classification
  `rank_pressure_improves_but_prediction_still_flat`.
- Prior sensitivity is still not material:
  zero-prior mean abs head-probability delta `0.0001433`,
  final-probability delta `0.00003426`, classification
  `prior_signal_suppressed_before_final_score`.
- Level 1 QueryLocalUtility scores are non-promotable:
  MLQDS `0.1013722`, uniform `0.1153614`, Douglas-Peucker `0.1146120`.
- Target diffusion failed at smoke scale on
  `final_label_support_fraction_above_max`: final support `0.532986` vs max
  `0.5`.
- Learning causality still fails at smoke scale:
  `shuffled_prior_fields_should_lose`,
  `without_query_prior_features_should_lose`, and
  `without_behavior_utility_head_should_lose`.
- Required prior ablations are wrong-way at smoke scale:
  shuffled-prior delta `-0.0022679`, no-query-prior delta `-0.0022679`.
- Segment-budget head removal still hurts primary by `0.0105170`, so the
  selector/head stack is not simply disconnected.

Decision:
- Keep this as a target-semantics implementation checkpoint only. It is not a
  strict evidence boundary and must not be compared as a successful variant.
- The change is materially different from rejected scalar boosts/residuals and
  selector tricks because it changes the supervised q-hit semantics to the
  train-workload evidence signal that the strict diagnostics identified.
- Do not run Level 3 or the final grid.
- Next admissible step: Level 2 minimum strict gate localization for the new
  query-evidence gate target. If Level 2 fails target diffusion, predictability,
  or prior-predictive alignment, stop there and diagnose the target semantics
  before any further model or selector work.

## Checkpoint Phase 37 - query evidence gate Level 2 strict replay

Status: completed / rejected-as-current-boundary.

Hypothesis:
- The train-workload query-evidence q-hit target may fix the raw q-hit scale
  problem, but it may also make labels too broad.
- Run one strict Level 2 replay and stop at the first failed required gate.

Changed files:
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`
- `docs/query-driven-implementation-research-guide.md`

Validation commands:
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2539 --n_ships 32 --n_points 192 --synthetic_route_families 3 --n_queries 24 --max_queries 192 --epochs 4 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 30000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/query_evidence_gate_level2_seed2539)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/query_evidence_gate_level2_seed2539/example_run.json --output artifacts/results/query_evidence_gate_level2_seed2539/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/query_evidence_gate_level2_seed2539/example_run.json`
- `artifacts/results/query_evidence_gate_level2_seed2539/semantic_diagnostic.json`

Scale:
- Strict Level 2 source-stratified synthetic replay, seed `2539`
- `n_ships=32`, `n_points=192`, `n_queries=24`, `max_queries=192`,
  `epochs=4`, `range_train_workload_replicates=4`

Key results:
- Level 2 QueryLocalUtility:
  - MLQDS `0.1125981743`
  - uniform `0.0992909061`
  - Douglas-Peucker `0.1182249577`
  - Oracle `0.3095599786`
- Gate state:
  - passed: workload stability, support overlap, predictability,
    prior-predictive alignment
  - failed: target diffusion, workload signature, learning causality, global
    sanity, final grid
- First failed gate by protocol order is target diffusion:
  - `final_label_support_fraction_above_max`
  - final label support at `gt_0.01`: `0.7058531746`
  - max allowed support: `0.5`
- The new evidence gate lifted essentially all positive q-hit support above the
  diffusion threshold:
  - old raw q-hit target support at `gt_0.01`: `0.4340277778`
  - new q-hit target support at `gt_0.01`: `0.7058531746`
  - new q-hit target support at `gt_0.05`: `0.7058531746`
- Final target support also broadened sharply:
  - old final `gt_0.01`: `0.1604662698`
  - new final `gt_0.01`: `0.7058531746`
  - new final `gt_0.05`: `0.5188492063`
- q-hit learning improved superficially but on an invalid target:
  q-hit tau `0.3586653`, top-5% mass recall `0.5457720`,
  prediction std `0.0264098`, target std `0.1152468`.
- Behavior remains too flat:
  prediction std / target std `0.0430485`,
  classification `rank_pressure_improves_but_prediction_still_flat`.
- Prior materiality remains blocked:
  zero-prior mean abs head-probability delta `0.00005037`,
  final-probability delta `0.00000921`,
  classification `prior_target_signal_available_but_trained_heads_invariant`.
- Learning-causality children still fail:
  `shuffled_prior_fields_should_lose`,
  `without_query_prior_features_should_lose`,
  `without_behavior_utility_head_should_lose`.
- Required prior ablations remain immaterial:
  shuffled-prior delta `0.0`, no-query-prior delta `0.0`.

Decision:
- Reject the broad `query_evidence_gate_hit_ship_blend` target as the current
  production target semantics. It fails the earliest relevant strict gate:
  target diffusion.
- Do not run Level 3 or the final grid. Do not move to model or selector work
  while target diffusion is failed.
- The root cause is not workload health or support overlap. The query-evidence
  gate normalized every positive hit-support point high enough to make the
  final label broad; it collapsed the old useful sparse scale separation.
- Next admissible step: target-diffusion root fix design. Narrow, sparsify, or
  revert the q-hit evidence gate using train-workload evidence only, then
  restart at Level 1 and Level 2. Do not loosen the diffusion gate.

## Checkpoint Phase 38 - raw q-hit evidence multiplier target narrowing

Status: completed / implementation-only Level 1 wiring.

Hypothesis:
- The strict Level 2 target diffusion failure was caused by normalizing both
  q-hit and ship evidence on every positive support slice. That lifted one-hit
  points above the final-label support threshold.
- A narrow root fix is to preserve raw q-hit scale and use ship/family evidence
  only as a bounded multiplier inside raw q-hit support:
  `raw_q_hit * (0.65 + 0.35 * positive_mean_normalized_ship_evidence)`.

Changed files:
- `learning/targets/query_local_utility.py`
- `tests/unit/learning/test_query_local_utility_targets.py`
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`
- `docs/query-driven-implementation-research-guide.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/learning/targets/query_local_utility.py Range_QDS/tests/unit/learning/test_query_local_utility_targets.py`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/ruff check learning/targets/query_local_utility.py tests/unit/learning/test_query_local_utility_targets.py)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pytest tests/unit/learning/test_query_local_utility_targets.py -q)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pytest tests/unit/learning/test_query_local_utility_targets.py tests/unit/learning/test_query_local_utility_training.py tests/unit/orchestration/test_query_driven_protocol_gates.py -q)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pyright learning/targets/query_local_utility.py tests/unit/learning/test_query_local_utility_targets.py)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2556 --n_ships 12 --n_points 96 --synthetic_route_families 3 --n_queries 8 --max_queries 64 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 12000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 2 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/query_evidence_gate_narrow_level1_smoke)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/query_evidence_gate_narrow_level1_smoke/example_run.json --output artifacts/results/query_evidence_gate_narrow_level1_smoke/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/query_evidence_gate_narrow_level1_smoke/example_run.json`
- `artifacts/results/query_evidence_gate_narrow_level1_smoke/semantic_diagnostic.json`

Scale:
- Level 1 wiring smoke only, seed `2556`
- `n_ships=12`, `n_points=96`, `n_queries=8`, `max_queries=64`,
  `epochs=1`, `range_train_workload_replicates=2`

Key results:
- Active target semantics:
  `query_hit_target_variant=raw_query_hit_ship_evidence_multiplier`.
- Final formula:
  `raw_query_hit_evidence_multiplier_times_behavior_with_conditional_replacement_modulation_plus_boundary`.
- Target diffusion passed Level 1:
  - final support `gt_0.01`: `0.2170138889`
  - max allowed support: `0.5`
  - q-hit head support `gt_0.01`: `0.5399305556`
    (`support_fraction_above_max` on the q-hit head is nonblocking but must be
    watched at strict Level 2)
- Query-hit target remains ship-evidence aligned:
  Spearman with ship-query evidence `0.9518762`, top-5% ship-evidence mass
  recall `0.7250898`.
- Level 1 QueryLocalUtility scores are non-promotable:
  MLQDS `0.0903625784`, uniform `0.0684911082`, Douglas-Peucker
  `0.0653404385`.
- Behavior remains flat:
  prediction std / target std `0.0149105`, classification
  `rank_pressure_improves_but_prediction_still_flat`.
- Learning causality still fails at Level 1:
  `untrained_model_should_lose`, `without_behavior_utility_head_should_lose`,
  and `without_segment_budget_head_should_lose`.
- Required prior ablations are material at this tiny scale but not interpretable
  as scientific evidence:
  shuffled-prior delta `0.0182208868`, no-query-prior delta `0.0182208868`.

Decision:
- Keep the narrowed target as the current implementation path. It fixes the
  broad normalized gate's Level 1 target-shape problem without loosening gates
  or touching selector/model logic.
- Do not claim learning success. This is only a wiring check.
- Do not run Level 3 or the final grid.
- Next admissible step: one strict Level 2 replay for
  `raw_query_hit_ship_evidence_multiplier`, followed by the semantic diagnostic.
  If target diffusion fails, stop and diagnose target semantics. If target
  diffusion passes and learning causality fails, classify the failed child gate
  before any model or selector change.

## Checkpoint Phase 39 - raw q-hit evidence multiplier Level 2 strict replay

Status: completed / blocked-by-earlier-gate.

Hypothesis:
- The narrowed q-hit target preserves raw q-hit scale enough to pass strict
  Level 2 target diffusion.
- If it passes target diffusion, the next blocker should be diagnosed by gate
  order before any model, selector, Level 3, or final-grid work.

Changed files:
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`
- `docs/query-driven-implementation-research-guide.md`

Validation commands:
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2539 --n_ships 32 --n_points 192 --synthetic_route_families 3 --n_queries 24 --max_queries 192 --epochs 4 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 30000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/raw_query_hit_evidence_multiplier_level2_seed2539)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/raw_query_hit_evidence_multiplier_level2_seed2539/example_run.json --output artifacts/results/raw_query_hit_evidence_multiplier_level2_seed2539/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/raw_query_hit_evidence_multiplier_level2_seed2539/example_run.json`
- `artifacts/results/raw_query_hit_evidence_multiplier_level2_seed2539/semantic_diagnostic.json`

Scale:
- Level 2 minimum strict, synthetic, source-stratified, seed `2539`
- `n_ships=32`, `n_points=192`, `n_queries=24`, `max_queries=192`,
  `epochs=4`, `range_train_workload_replicates=4`

Key results:
- Scores:
  - MLQDS `0.1087337015`
  - uniform `0.0992909061`
  - Douglas-Peucker `0.1182249577`
  - Oracle `0.3095599786`
- Gate state:
  - passed: workload stability, support overlap, target diffusion,
    predictability, prior-predictive alignment
  - failed: workload signature, learning causality, global sanity, final grid
- Target diffusion passed:
  - final support `gt_0.01`: `0.0887896825`
  - max allowed support: `0.5`
  - q-hit head support `gt_0.01`: `0.2244543651`
- Workload signature failed first by protocol order:
  - failed check: `point_hit_fraction_distribution_ks`
  - train KS `0.2090620032 > 0.2`
  - train_r3 KS `0.2090620032 > 0.2`
  - selection KS `0.2576064909 > 0.2`
  - anchor and footprint family L1 distances passed.
- Learning causality remains failed behind workload signature:
  - shuffled-score delta `0.0244078737`
  - untrained delta `0.0125196269`
  - shuffled-prior delta `0.0`
  - no-query-prior delta `0.0`
  - no-behavior-head delta `-0.0068544427`
  - no-segment-budget delta `0.0256773910`
  - prior-field-only delta `0.0109610210`
- Behavior remains flat and non-causal:
  prediction std / target std `0.0382578`, classification
  `rank_pressure_improves_but_prediction_still_flat`; no-behavior ablation is
  wrong-way.
- Prior inputs move but masks do not:
  shuffled-prior sampled mean abs delta `0.1076817`, model-prior delta
  `0.0168450`, head-probability delta `0.0000454`, retained-mask Jaccard `1.0`.
- Segment path is still suspect:
  semantic diagnostic classifies `segment head fails to learn target`; segment
  retained-marginal Spearman is `-0.0390371`, while pooled point-score segment
  allocation is `+0.0025878` above primary. Removing the segment-budget head
  still hurts by `0.0256774`, so this is not a simple selector deletion case.

Decision:
- The narrowed q-hit target fixed the broad target-diffusion failure at strict
  Level 2. Keep it as the current target path.
- This is not promotion evidence. The earliest failed gate is workload
  signature, so model/selector work is not admissible yet.
- Do not run Level 3 or the final grid.
- Next admissible step: `workload_signature_gate_level2_diagnosis`. Diagnose
  whether the point-hit-fraction KS failure is generator/profile mismatch,
  split-scale instability, gate/schema issue, or target-independent workload
  variance. Do not loosen the gate.

## Checkpoint Phase 40 - workload-signature Level 2 diagnosis

Status: completed / diagnostic-only.

Hypothesis:
- The strict Level 2 workload-signature failure is target-independent
  split-scale / KS-statistic instability, especially from the small selection
  split, not a q-hit target issue or unhealthy query generator.

Changed files:
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`
- `docs/query-driven-implementation-research-guide.md`

Validation commands:
- `sed -n '360,660p' Range_QDS/orchestration/range_diagnostics.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python - <<'PY' ... compare workload_signature_gate pairs across strict Level 2 and Level 3 artifacts ... PY`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python - <<'PY' ... summarize point_hit_fractions_per_query by split and family ... PY`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python - <<'PY' ... family-conditioned KS drilldown for train/train_r3/selection vs eval ... PY`

Artifact:
- `artifacts/results/raw_query_hit_evidence_multiplier_level2_seed2539/example_run.json`
- `artifacts/results/query_evidence_gate_level2_seed2539/example_run.json`
- `artifacts/results/factorized_prior_delta_composition_level2_seed2539/example_run.json`
- `artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/example_run.json`

Scale:
- Static / derived artifact diagnosis only.
- No new training run.

Key results:
- The narrowed target, rejected broad target, and prior composition Level 2
  artifacts have identical workload-signature failures:
  train KS `0.2090620032`, train_r3 KS `0.2090620032`, selection KS
  `0.2576064909`, all failing only `point_hit_fraction_distribution_ks`.
- Workload stability, coverage generation, broad-query rate, duplicate rate,
  anchor-family L1, and footprint-family L1 pass. This is not a profile-family
  mismatch or generator-health failure.
- The gate implementation intentionally enforces normalized point-hit-fraction
  KS for coverage-calibrated `profile_sampled_query_count` signatures. Raw
  point-hit counts and ship-hit distributions are diagnostic in that mode.
- Selection is the clearest weak split: `3` trajectories / `576` points versus
  eval's `8` trajectories / `1536` points, mean point-hit fraction
  `0.0130507663` versus eval `0.0110868566`, and large-context KS `0.455556`.
- Train/train_r3 failures are mild threshold misses and concentrated in
  medium-operational density slices, not broad profile collapse.
- The existing Level 3 reference at `64` ships, `256` points, `40` requested
  queries passes workload signature, including selection KS exactly `0.2`.

Decision:
- Classify the strict Level 2 failure as target-independent split-scale /
  KS-statistic instability. Do not loosen the gate.
- Do not change target, profile, model, selector, raw coverage, temporal
  scaffold, or guardrails from this diagnosis.
- Next admissible step: `current_target_level3_signature_reentry`. Run one
  strict Level 3 single-cell replay at the existing healthy reference scale for
  the current target. Stop at the first failed gate; if earlier gates pass and
  learning causality fails, run the semantic diagnostic and classify the failed
  child gates before changing code.

## Checkpoint Phase 41 - current target Level 3 signature re-entry

Status: completed / blocked-by-learning-causality.

Hypothesis:
- The strict Level 2 workload-signature failure is a small-split KS artifact,
  so the current narrowed target should clear workload signature at Level 3
  reference scale.
- If pre-causality gates pass, learning causality should become the active
  blocker again.

Changed files:
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`
- `docs/query-driven-implementation-research-guide.md`

Validation commands:
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2527 --n_ships 64 --n_points 256 --synthetic_route_families 4 --n_queries 40 --max_queries 384 --epochs 5 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 40000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/raw_query_hit_evidence_multiplier_level3_signature_reentry_seed2527)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/raw_query_hit_evidence_multiplier_level3_signature_reentry_seed2527/example_run.json --output artifacts/results/raw_query_hit_evidence_multiplier_level3_signature_reentry_seed2527/semantic_diagnostic.json)`
- `jq '.final_claim_summary' artifacts/results/raw_query_hit_evidence_multiplier_level3_signature_reentry_seed2527/example_run.json`
- `jq '.artifacts[0] | {scores,gates,failed_learning_causality_checks}' artifacts/results/raw_query_hit_evidence_multiplier_level3_signature_reentry_seed2527/semantic_diagnostic.json`

Artifact:
- `artifacts/results/raw_query_hit_evidence_multiplier_level3_signature_reentry_seed2527/example_run.json`
- `artifacts/results/raw_query_hit_evidence_multiplier_level3_signature_reentry_seed2527/semantic_diagnostic.json`

Scale:
- Strict Level 3 single-cell replay, source-stratified synthetic, seed `2527`
- `n_ships=64`, `n_points=256`, `synthetic_route_families=4`,
  `n_queries=40`, `max_queries=384`, `epochs=5`,
  `range_train_workload_replicates=4`

Key results:
- Scores:
  - MLQDS `0.1248364151`
  - uniform `0.1247681518`
  - Douglas-Peucker `0.1153266238`
  - Oracle `0.2891527670`
  - MLQDS minus uniform `0.0000682634`
- Gate state:
  - passed: workload stability, support overlap, target diffusion, workload
    signature, predictability, prior-predictive alignment
  - failed: learning causality, global sanity diagnostic guardrail, final grid
- Workload signature passed at Level 3:
  - train KS `0.1079268293`
  - train_r1 KS `0.0964285714`
  - train_r2 KS `0.1542682927`
  - train_r3 KS `0.1659090909`
  - selection KS `0.2`
- Learning causality failed:
  - shuffled-score delta `0.0007002658 < 0.005`
  - untrained delta `0.0049848809 < 0.005`
  - shuffled-prior delta `0.0`
  - no-query-prior delta `0.0`
  - no-behavior-head delta `0.0038341870 < 0.005`
  - no-segment-budget delta `0.0085639804`
  - prior-field-only delta `0.0137116370`
- Semantic diagnostic classifications:
  - prior: `model ignores prior inputs`; sampled/model priors change but
    retained masks do not, mean head-probability delta `0.0000170`, mean score
    delta `0.0003447`.
  - behavior: `target has signal but head does not learn it`; behavior tau
    `-0.002805`, prediction std `0.0033458`, target std `0.1637143`.
  - segment: `allocation scoring and point-selection scoring are mixed
    incorrectly`; raw/selector retained-marginal Spearman are positive but weak
    (`0.0939718` / `0.0992330`), segment-score Spearman is wrong-way
    (`-0.2024585`), and pooled point-score allocation is `+0.002148` above
    primary.

Decision:
- The workload-signature blocker is resolved at Level 3 scale for the current
  target. This is not final success.
- The current strict boundary is learning-causality failure with near-zero
  uniform gap. Treat this as score-level learning collapse plus the known prior,
  behavior, and segment child failures.
- Do not run the final grid or another Level 3 variant. Do not use selector
  floor tweaks, temporal scaffolding, raw coverage overrides, generic
  behavior-rank tuning, or prior-scale/residual patches.
- Next admissible step:
  `learning_causality_level3_reentry_root_diagnosis`. Use existing artifacts to
  explain the near-zero uniform gap and child-gate failures before any model,
  target, loss, or selector code change.

## Checkpoint Phase 42 - Level 3 learning-causality root diagnosis

Status: completed / diagnostic-only.

Hypothesis:
- The near-zero Level 3 uniform gap is at least partly a current-default/config
  boundary issue, not only the narrowed q-hit target.
- The current re-entry run differs from the historical healthy Level 3
  reference in model/training/score defaults, and those differences must be
  isolated before any root fix.

Changed files:
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`
- `docs/query-driven-implementation-research-guide.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python - <<'PY' ... compare config, scores, gates, learning deltas, and head fit across current Level 3, historical Level 3, and current Level 2 artifacts ... PY`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python - <<'PY' ... compare training-history validation gap, per-head validation fit, and checkpoint selection across the same artifacts ... PY`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python - <<'PY' ... compare target-diffusion support by head across the same artifacts ... PY`

Artifact:
- `artifacts/results/raw_query_hit_evidence_multiplier_level3_signature_reentry_seed2527/example_run.json`
- `artifacts/results/raw_query_hit_evidence_multiplier_level3_signature_reentry_seed2527/semantic_diagnostic.json`
- `artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/example_run.json`
- `artifacts/results/raw_query_hit_evidence_multiplier_level2_seed2539/example_run.json`

Scale:
- Static / derived artifact diagnosis only.
- No new training run.

Key results:
- Current Level 3 and historical Level 3 use the same synthetic scale and seed:
  `64` ships, `256` points, `4` route families, `40` requested queries, seed
  `2527`.
- Current Level 3 defaults differ from the historical healthy Level 3
  reference:
  - current: `embed_dim=64`, `num_heads=4`, `num_layers=3`,
    `train_batch_size=16`, `behavior_rank_loss_weight=0.25`,
    `mlqds_score_mode=rank`
  - historical: `embed_dim=32`, `num_heads=2`, `num_layers=1`,
    `train_batch_size=8`, `behavior_rank_loss_weight=0.0`,
    `mlqds_score_mode=rank_confidence`
- The current Level 3 replay underperforms uniform on checkpoint validation at
  every epoch. Best validation query-local utility is `0.1060108373` versus
  validation uniform `0.1171570720`, with best selection uniform-gap
  `-0.0190929031`.
- The historical Level 3 reference has a positive validation gap at epoch 0:
  `0.1218223145` versus validation uniform `0.1171570720`, gap
  `+0.0046652425`.
- Current Level 2 with the same current defaults did not show the same
  validation collapse: best validation query-local utility is `0.1182967041`
  versus uniform `0.1061243832`, gap `+0.0077921147`, and final MLQDS minus
  uniform is `+0.0094427954`.
- Current Level 3 train-side fit is not the immediate problem:
  train target Kendall tau `0.4580767`, matched train target recall advantage
  `+0.1188167`, and low-budget recall advantage `+0.0973639`.
  The failure is validation/final generalization and score-to-mask materiality.
- The narrowed q-hit target changes target support at Level 3:
  final support drops from historical `0.1491477` to current `0.0775036`, and
  q-hit support drops from historical `0.4135298` to current `0.2504439`.
  This is a plausible contributor, but not proven as the sole root cause.
- Selector diagnostics show segment allocation is also a contributor but not
  sufficient alone: pooled point-score allocation is `+0.002148` above primary,
  and removing segment-length support reaches `0.1289066601`, still far below
  the historical primary `0.1431090566`.

Decision:
- Classify the active blocker as Level 3 score-level learning collapse across a
  config/default and target-composition boundary. The artifacts do not justify
  a model, selector, target, or loss change yet.
- The current defaults are suspect: the code now defaults to behavior-rank
  pressure and rank score mode even though behavior-rank-only tuning is not an
  admissible fix and the current Level 3 behavior head is still flat.
- Next admissible step:
  `reference_config_current_target_level2_control`. Run one strict Level 2
  diagnostic control with the current narrowed target but historical Level 3
  model/score defaults (`embed_dim=32`, `num_heads=2`, `num_layers=1`,
  `train_batch_size=8`, `inference_batch_size=8`,
  `query_local_utility_behavior_rank_loss_weight=0.0`,
  `mlqds_score_mode=rank_confidence`). Treat it as diagnostic if the known
  Level 2 workload-signature KS failure recurs. Do not run another Level 3
  until the smaller control explains whether the collapse is a stale-default
  problem or a target-composition problem.

## Checkpoint Phase 43 - reference-config current-target Level 2 control

Status: completed / rejected-as-root-fix-direction.

Hypothesis:
- If the current Level 3 collapse is mainly stale/suboptimal current defaults,
  then restoring historical Level 3 model/score defaults with the current
  narrowed target should improve strict Level 2 validation gap and score-control
  materiality relative to the current Level 2 boundary.

Changed files:
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`
- `docs/query-driven-implementation-research-guide.md`

Validation commands:
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2539 --n_ships 32 --n_points 192 --synthetic_route_families 3 --n_queries 24 --max_queries 192 --epochs 4 --embed_dim 32 --num_heads 2 --num_layers 1 --train_batch_size 8 --inference_batch_size 8 --query_local_utility_behavior_rank_loss_weight 0.0 --mlqds_score_mode rank_confidence --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 30000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/reference_config_current_target_level2_control_seed2539)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/reference_config_current_target_level2_control_seed2539/example_run.json --output artifacts/results/reference_config_current_target_level2_control_seed2539/semantic_diagnostic.json)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python - <<'PY' ... compare validation history for current Level 2 vs reference-config control ... PY`

Artifact:
- `artifacts/results/reference_config_current_target_level2_control_seed2539/example_run.json`
- `artifacts/results/reference_config_current_target_level2_control_seed2539/semantic_diagnostic.json`
- Comparison source:
  `artifacts/results/raw_query_hit_evidence_multiplier_level2_seed2539/example_run.json`

Scale:
- Strict Level 2 diagnostic control, source-stratified synthetic, seed `2539`
- `n_ships=32`, `n_points=192`, `synthetic_route_families=3`,
  `n_queries=24`, `max_queries=192`, `epochs=4`,
  `range_train_workload_replicates=4`
- Historical model/score defaults: `embed_dim=32`, `num_heads=2`,
  `num_layers=1`, `train_batch_size=8`, `inference_batch_size=8`,
  `query_local_utility_behavior_rank_loss_weight=0.0`,
  `mlqds_score_mode=rank_confidence`

Key results:
- Scores:
  - MLQDS `0.0850006110`
  - uniform `0.0992909061`
  - Douglas-Peucker `0.1182249577`
  - Oracle `0.3095599786`
  - MLQDS minus uniform `-0.0142902951`
- Gate state:
  - passed: workload stability, support overlap, target diffusion,
    predictability, prior-predictive alignment
  - failed: workload signature, learning causality, global sanity, final grid
  - the workload-signature failure is the known Level 2 KS issue and makes this
    diagnostic-only.
- The control underperforms current Level 2, not just current Level 3:
  - current Level 2 best validation uniform gap: `+0.0077921147`
  - reference-config control best validation uniform gap: `-0.0272109965`
  - current Level 2 final MLQDS minus uniform: `+0.0094427954`
  - reference-config control final MLQDS minus uniform: `-0.0142902951`
- Learning-causality remains failed and changes in the wrong direction:
  - shuffled-score delta `-0.0186187583`
  - untrained delta `-0.0353650391`
  - shuffled-prior delta `0.0`
  - no-query-prior delta `0.0`
  - no-behavior-head delta `0.0074820869`
  - no-segment-budget delta `-0.0052546605`
  - prior-field-only delta `-0.0127720695`
- Head fit is not a rescue:
  - q-hit tau `0.2366642`, prediction std `0.0004901`
  - behavior tau `-0.0061637`, prediction std `0.0073315`, target std
    `0.1688193`
  - segment tau `-0.1746138`, wrong-way
- Semantic diagnostic still classifies prior as `model ignores prior inputs`,
  behavior as `target has signal but head does not learn it`, and segment as
  `allocation scoring and point-selection scoring are mixed incorrectly`.

Decision:
- Reject a wholesale revert to the historical Level 3 model/score defaults as
  the next root-fix direction. It makes the current narrowed target worse at
  Level 2.
- Current defaults are not proven ideal, but the simple stale-default
  hypothesis is false.
- The active root is now target/score composition under the narrowed q-hit
  target, with segment allocation mixing as a secondary issue.
- Next admissible step: `narrow_target_score_composition_diagnosis`. Use the
  existing Level 2/Level 3 artifacts to diagnose whether the narrowed q-hit
  support and multiplicative factorized composition are suppressing
  query-local behavior/generalization. No new Level 3, no final grid, no
  behavior-rank sweep, no selector-floor patch.

## Checkpoint Phase 44 - narrow-target score-composition diagnosis

Status: completed / diagnostic-only.

Hypothesis:
- The narrowed raw-q-hit evidence target and multiplicative factorized score
  composition may suppress query-local behavior/generalization before selector
  allocation can help.
- Existing artifacts should be sufficient to classify whether this is q-hit
  support sparsity, q-hit validation fit, score composition, score-output
  conversion, segment allocation, or missing instrumentation.

Changed files:
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`
- `docs/query-driven-implementation-research-guide.md`
- Generated diagnostic artifact:
  `artifacts/results/narrow_target_score_composition_diagnosis/diagnostic.json`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python - <<'PY' ... derive target support, validation fit, factorized contribution, selector-row support, and segment alignment summaries ... PY`
- `jq empty Range_QDS/artifacts/results/narrow_target_score_composition_diagnosis/diagnostic.json`
- `git diff --check`

Artifact:
- `artifacts/results/narrow_target_score_composition_diagnosis/diagnostic.json`

Scale:
- Static / derived artifact diagnosis only.
- No new training run, no Level 3 variant, no final grid.
- Source artifacts:
  `artifacts/results/raw_query_hit_evidence_multiplier_level2_seed2539/example_run.json`,
  `artifacts/results/raw_query_hit_evidence_multiplier_level3_signature_reentry_seed2527/example_run.json`,
  `artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/example_run.json`,
  and
  `artifacts/results/reference_config_current_target_level2_control_seed2539/example_run.json`.

Key results:
- The current Level 3 narrowed target is much sparser than the historical
  Level 3 reference:
  - q-hit support at `gt_0.01`: current `0.2504439`, historical `0.4135298`
    (`0.6056x`)
  - final-label support at `gt_0.01`: current `0.0775036`, historical
    `0.1491477` (`0.5196x`)
- Current Level 3 q-hit validation fit is wrong-way at every epoch:
  q-hit tau range `[-0.1209831, -0.0788137]`, with best validation uniform gap
  `-0.0190929`. Historical Level 3 q-hit validation tau is also negative but
  much less severe, and its best validation uniform gap is positive
  (`+0.0046652`).
- Current Level 2 proves the high-marginal rows are not unsupported:
  top, missed-high, and under-ranked groups all have q-hit target positive
  fraction `1.0`. They are behavior-dominated: local behavior/interpolation
  contribution is `5.88x`, `6.17x`, and `8.41x` the point-recall contribution.
- Current Level 2 factorized composition on top-marginal rows:
  q-hit product Shapley is the dominant negative term
  (`-2.8488e-7`), behavior product is also negative (`-8.9955e-8`),
  replacement is positive but smaller (`+1.7822e-7`), raw-score delta is
  negative (`-8.7363e-6`), score-output delta is negative (`-0.0001870`),
  and segment-score delta is negative (`-0.0001842`).
- Current Level 3 shows the same failure on missed-high rows: q-hit support is
  positive for all missed-high rows, local behavior is `5.37x` point recall,
  q-hit product Shapley is the dominant negative term (`-1.1225e-7`), and
  segment-score delta is negative. Top rows also expose additional q-hit
  sparsity: q-hit target positive fraction is only `0.3667`.
- This is not primarily score-output/ranking conversion erasing useful positive
  raw-score movement. At strict scale, the composed raw deltas are already
  negative on the relevant rows.
- Segment allocation remains a secondary blocker: current Level 3 raw/selector
  retained-marginal Spearman are weak positive (`0.0939718` / `0.0992330`),
  while segment-score Spearman is wrong-way (`-0.2024585`) and pooled
  point-score allocation is `+0.002148` above primary.

Decision:
- Primary stop-condition classification:
  `multiplicative_qhit_gate_suppresses_behavior_local_movement_value`.
- Supporting classifications:
  `q_hit_support_too_sparse_for_generalization` and
  `q_hit_head_validation_fit_wrong_way`.
- Secondary classification:
  `segment_allocation_mixes_point_and_segment_scores_incorrectly`.
- Existing artifacts are sufficient; no instrumentation change is needed.
- Do not run Level 3 or the final grid. Do not implement selector floors,
  behavior-rank sweeps, prior-scale patches, route-density exposure, generic
  residuals, or a wholesale historical-default revert.
- Next admissible step:
  `qhit_behavior_composition_root_fix_design`. Design one narrow target/score
  composition root fix for behavior-dominated q-hit-supported rows, then
  restart at static/unit plus Level 1 wiring and strict Level 2 evidence if
  semantics change.

## Checkpoint Phase 45 - q-hit behavior-composition root-fix design

Status: completed / design-only.

Hypothesis:
- The current scalar composition makes q-hit the dominant gate for both point
  mass and local behavior. With sparse/wrong-way q-hit fit, tiny q-hit movement
  dominates behavior movement on rows whose marginal utility is mostly local
  behavior/interpolation.
- A narrow root fix should change score composition, not q-hit target support,
  selector allocation, prior scaling, or behavior loss weights.

Changed files:
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`
- `docs/query-driven-implementation-research-guide.md`
- Generated design artifact:
  `artifacts/results/qhit_behavior_composition_root_fix_design/design.json`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python - <<'PY' ... write design artifact ... PY`
- `jq empty Range_QDS/artifacts/results/qhit_behavior_composition_root_fix_design/design.json`
- `git diff --check`

Artifact:
- `artifacts/results/qhit_behavior_composition_root_fix_design/design.json`

Scale:
- Static/design checkpoint only.
- No production code change, no training run, no Level 3, no final grid.

Key design:
- Keep current head targets:
  `query_hit_target_variant=raw_query_hit_ship_evidence_multiplier` and
  `conditional_behavior_target_variant=query_segment_local_behavior_utility`.
- Change only the scalar factorized score/final-label composition:

```text
old:
score = q_hit * (0.5 + behavior) * (0.75 + 0.25 * replacement)
        + 0.25 * boundary

new:
score = (0.50 * q_hit + 0.45 * behavior)
        * (0.75 + 0.25 * replacement)
        + 0.05 * boundary
```

- This makes q-hit and behavior additive QueryLocalUtility branches before
  bounded replacement modulation. Behavior is still q-hit-masked by target
  construction; no all-point behavior supervision or q-hit/behavior floor is
  introduced.
- This is materially different from the rejected broad q-hit gate because it
  does not normalize q-hit support or lift every positive q-hit/evidence point.
  It leaves q-hit target semantics unchanged.
- This is not a prior-scale, residual, selector-floor, temporal scaffold,
  route-density, behavior-rank, or historical-default path.

Decision:
- Proceed to an implementation checkpoint only for
  `additive_qhit_behavior_score_composition`.
- Required Level 1 stop condition: static/unit checks pass, artifact emits the
  new formula, and target diffusion passes. If final support exceeds the `0.5`
  cap, stop and reject/redesign before Level 2.
- Required Level 2 stop condition: obey gate order. If target diffusion fails,
  stop. If the known Level 2 workload-signature KS issue recurs after earlier
  gates pass, treat as diagnostic only. If learning causality fails, run the
  semantic diagnostic and classify child gates before model/selector work.
- Do not run Level 3 or the final grid from this design alone.

## Checkpoint Phase 46 - additive q-hit behavior composition Level 1 wiring

Status: completed / implementation-only Level 1 wiring.

Hypothesis:
- Changing scalar composition to additive q-hit and behavior branches should
  make behavior locally material without broadening q-hit target support or
  changing selector/model/prior paths.
- Level 1 must only verify wiring and target shape; scores are not promotion
  evidence.

Changed files:
- `learning/targets/query_local_utility.py`
- `learning/factorized_head_diagnostics.py`
- `orchestration/causality.py`
- `orchestration/selector_diagnostics.py`
- `orchestration/diagnostics/semantic_causality_diagnostic.py`
- `orchestration/diagnostics/family_transfer_path_diagnostic.py`
- `tests/unit/learning/test_query_local_utility_targets.py`
- `tests/unit/learning/test_query_local_utility_training.py`
- `tests/unit/orchestration/test_query_driven_causality_and_summary.py`
- `tests/unit/orchestration/test_retained_mask_stage.py`
- `tests/unit/orchestration/test_semantic_causality_diagnostic.py`
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`
- `docs/query-driven-implementation-research-guide.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/learning/targets/query_local_utility.py Range_QDS/orchestration/selector_diagnostics.py Range_QDS/orchestration/causality.py Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/orchestration/diagnostics/family_transfer_path_diagnostic.py Range_QDS/tests/unit/learning/test_query_local_utility_targets.py Range_QDS/tests/unit/orchestration/test_retained_mask_stage.py Range_QDS/tests/unit/orchestration/test_query_driven_causality_and_summary.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/learning/targets/query_local_utility.py Range_QDS/orchestration/selector_diagnostics.py Range_QDS/orchestration/causality.py Range_QDS/learning/factorized_head_diagnostics.py Range_QDS/orchestration/diagnostics/semantic_causality_diagnostic.py Range_QDS/orchestration/diagnostics/family_transfer_path_diagnostic.py Range_QDS/tests/unit/learning/test_query_local_utility_targets.py Range_QDS/tests/unit/orchestration/test_retained_mask_stage.py Range_QDS/tests/unit/orchestration/test_query_driven_causality_and_summary.py Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/pyright learning/targets/query_local_utility.py orchestration/selector_diagnostics.py orchestration/causality.py learning/factorized_head_diagnostics.py orchestration/diagnostics/semantic_causality_diagnostic.py tests/unit/learning/test_query_local_utility_targets.py tests/unit/orchestration/test_retained_mask_stage.py tests/unit/orchestration/test_query_driven_causality_and_summary.py tests/unit/orchestration/test_semantic_causality_diagnostic.py)`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pytest Range_QDS/tests/unit/learning/test_query_local_utility_targets.py Range_QDS/tests/unit/orchestration/test_retained_mask_stage.py::test_factorized_score_component_vectors_from_logits_reports_score_terms Range_QDS/tests/unit/orchestration/test_query_driven_causality_and_summary.py::test_prior_ablation_sensitivity_from_tensors_builds_consistent_chain Range_QDS/tests/unit/orchestration/test_semantic_causality_diagnostic.py -q`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pytest Range_QDS/tests/unit/learning/test_query_local_utility_training.py::test_factorized_final_score_composition_diagnostics_match_scalar_target -q`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2557 --n_ships 12 --n_points 96 --synthetic_route_families 3 --n_queries 8 --max_queries 64 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 12000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 2 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/additive_qhit_behavior_score_composition_level1_smoke)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/additive_qhit_behavior_score_composition_level1_smoke/example_run.json --output artifacts/results/additive_qhit_behavior_score_composition_level1_smoke/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/additive_qhit_behavior_score_composition_level1_smoke/example_run.json`
- `artifacts/results/additive_qhit_behavior_score_composition_level1_smoke/semantic_diagnostic.json`

Scale:
- Static/unit plus Level 1 wiring smoke, seed `2557`.
- `n_ships=12`, `n_points=96`, `n_queries=8`, `max_queries=64`,
  `epochs=1`, `range_train_workload_replicates=2`.

Key results:
- Active formula:
  `additive_raw_query_hit_and_behavior_with_conditional_replacement_modulation_plus_boundary`.
- Level 1 target diffusion passed:
  - final support `gt_0.01`: `0.234375`
  - max allowed support: `0.5`
  - behavior support `gt_0.01`: `0.261745`
  - q-hit head support `gt_0.01`: `0.517361` (nonblocking head check; watch
    strict Level 2).
- Level 1 scores are non-promotable:
  MLQDS `0.1064832750`, uniform `0.1166761729`, Douglas-Peucker
  `0.0953898417`.
- Expected wiring movement happened at smoke scale:
  shuffled-score delta `0.0629099`, untrained delta `0.0250962`,
  no-behavior-head delta `0.0264494`, no-segment-budget delta `0.0251730`,
  prior-field-only delta `0.0246377`.
- Prior child gates still fail at smoke scale: shuffled-prior delta `0.0` and
  no-query-prior delta `0.0`.
- Semantic diagnostic still classifies prior as `model ignores prior inputs`;
  behavior is no longer below material ablation at smoke scale.

Decision:
- Accept this only as implementation wiring. It does not update final evidence
  and must not be promoted.
- The Level 1 stop condition passed, so the next admissible step is one strict
  Level 2 replay of the additive composition stack, followed by semantic
  diagnostic if earlier gates allow learning-causality interpretation.
- Do not run Level 3 or final grid from Level 1 evidence.

## Checkpoint Phase 47 - additive q-hit behavior composition strict Level 2

Status: completed / strict Level 2 diagnostic, not promotion evidence.

Hypothesis:
- The additive q-hit / behavior composition that passed Level 1 wiring should
  preserve strict Level 2 target diffusion and expose whether the behavior,
  prior, and segment child gates improve at admissible scale.

Changed files:
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`
- `docs/query-driven-implementation-research-guide.md`

Validation commands:
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2539 --n_ships 32 --n_points 192 --synthetic_route_families 3 --n_queries 24 --max_queries 192 --epochs 4 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 30000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/additive_qhit_behavior_score_composition_level2_seed2539)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/additive_qhit_behavior_score_composition_level2_seed2539/example_run.json --output artifacts/results/additive_qhit_behavior_score_composition_level2_seed2539/semantic_diagnostic.json)`

Artifact:
- `artifacts/results/additive_qhit_behavior_score_composition_level2_seed2539/example_run.json`
- `artifacts/results/additive_qhit_behavior_score_composition_level2_seed2539/semantic_diagnostic.json`

Scale:
- Strict Level 2 synthetic replay, seed `2539`.
- `n_ships=32`, `n_points=192`, `n_queries=24`, `max_queries=192`,
  `epochs=4`, `range_train_workload_replicates=4`.

Key results:
- Active formula:
  `additive_raw_query_hit_and_behavior_with_conditional_replacement_modulation_plus_boundary`.
- Target diffusion passed:
  - final support `gt_0.01`: `0.2351190476`
  - max allowed support: `0.5`
  - behavior support `gt_0.01`: `0.293043`
  - q-hit head support `gt_0.01`: `0.224454`
- QueryLocalUtility:
  - MLQDS: `0.0995482993`
  - uniform: `0.0992909061`
  - Douglas-Peucker: `0.1182249577`
  - MLQDS minus uniform: `+0.0002573932`
  - MLQDS minus Douglas-Peucker: `-0.0186766584`
- Gate state:
  - workload stability: passed
  - support overlap: passed
  - target diffusion: passed
  - workload signature: failed with the known point-hit-fraction KS recurrence
  - predictability: passed
  - prior predictive alignment: passed
  - learning causality: failed
  - global sanity: failed because length preservation is `0.703978` below the
    `0.75` floor
- Learning-causality deltas:
  - shuffled score: `0.00638639`
  - untrained model: `0.0110130`
  - no behavior head: `0.00622185`
  - shuffled prior fields: `0.0`
  - no query prior features: `0.0`
  - no segment budget head: `-0.000488398`
  - prior-field-only score: `0.00177562`
- Semantic diagnostic:
  - Prior remains `model ignores prior inputs`: sampled/model priors change,
    mean absolute head-probability delta is only `0.0000293`, score top-k
    Jaccard is `1.0`, and retained masks are unchanged.
  - Behavior failure is now partial rather than below material ablation:
    no-behavior-head delta is above threshold, but rank alignment is still weak.
  - Segment failure is now partial rather than wrong-way: segment-score
    retained-marginal Spearman is `0.174728`, but segment-budget-head
    materiality still fails, and pooled point-score allocation is `+0.0152371`
    above primary.

Decision:
- Reject promotion. This is not final success and must not advance to Level 3
  or final grid.
- Accept the additive score composition only as having cleared target diffusion
  and the immediate behavior-materiality blocker at strict Level 2.
- Remaining child-gate blockers are prior materiality and segment-budget-head
  materiality. The next step must localize these from the strict Level 2
  artifact before any production change.
- Next admissible step: `additive_level2_child_gate_root_localization`. Use the
  existing strict Level 2 artifact and semantic diagnostic unless a concrete
  instrumentation gap is found. Do not repeat scalar prior boosts, generic
  residuals, route-density exposure, selector floors, raw coverage overrides,
  or final-grid/Level-3 runs.

## Checkpoint Phase 48 - additive Level 2 child-gate root localization

Status: completed / derived diagnostic and root-fix design, not promotion
evidence.

Hypothesis:
- The remaining strict Level 2 failure is not target diffusion or behavior
  materiality. It is split between prior signal dying before retained-mask
  movement and segment allocation using a compressed, non-causal segment-budget
  head.

Changed files:
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`
- `docs/query-driven-implementation-research-guide.md`

Validation commands:
- `jq empty Range_QDS/artifacts/results/additive_level2_child_gate_root_localization/diagnostic.json`
- `git diff --check`
- `! rg -n 'checkpoint: additive_level2_child_gate_root_localization|strict Level 3 re-entry replay cleared|Current blocker-localizing reference artifact' Range_QDS/docs/Next-Iterations.md Range_QDS/docs/query-driven-implementation-research-guide.md`

Artifact:
- `artifacts/results/additive_level2_child_gate_root_localization/diagnostic.json`

Scale:
- Derived strict Level 2 artifact diagnosis only.
- No new training, no new probe, no production-code change.

Key results:
- Source artifacts:
  - `artifacts/results/additive_qhit_behavior_score_composition_level2_seed2539/example_run.json`
  - `artifacts/results/additive_qhit_behavior_score_composition_level2_seed2539/semantic_diagnostic.json`
- Strict Level 2 remained non-accepted:
  MLQDS `0.0995482993`, uniform `0.0992909061`,
  Douglas-Peucker `0.1182249577`.
- Target diffusion passed with final support `gt_0.01 = 0.2351190476`.
- Learning causality failed:
  shuffled-score delta `0.00638639`, untrained delta `0.0110130`,
  no-behavior-head delta `0.00622185`, shuffled-prior delta `0.0`,
  no-query-prior delta `0.0`, no-segment-budget-head delta `-0.000488398`.
- Prior localization:
  priors are predictive and reach the model, but retained-mask Jaccard stays
  `1.0`; mean absolute head-probability delta is about `2.9e-05`; high-marginal
  score-output delta is `0.0`.
- Behavior localization:
  behavior is material by ablation, but `conditional_behavior_utility` remains
  weak with tau `0.040724` and prediction std / target std about `0.103`.
- Segment localization:
  the segment-budget target is oracle-aligned, but the learned segment head is
  compressed and non-causal. Pooled final point-score allocation scores
  `0.114785381`, which is `+0.015237081` above primary.
- Length localization:
  MLQDS length preservation is `0.703978` below the `0.75` floor. Length-only
  allocation can clear the length floor counterfactually, but pure path-length
  allocation scores only `0.0858993835` on `QueryLocalUtility`.

Decision:
- Reject promotion. This is a diagnosis/design artifact, not scientific
  evidence of final success.
- Do not run Level 3 or final grid.
- Do not repeat prior-scale boosts, generic prior residuals, route-density
  exposure, prior adapters, prior-only losses, selector floors, raw coverage
  overrides, length scaffolds, or guardrail weakening.
- Next admissible step: `pooled_point_score_segment_allocation_level1_wiring`.
  Change only the QueryLocalUtility learned-segment-budget allocation source to
  pooled final point-score segment scores, then run static checks, focused unit
  tests, and one Level 1 wiring smoke before any strict Level 2 replay.

## Checkpoint Phase 49 - pooled point-score allocation Level 1 wiring

Status: rejected / production selector semantics reverted, trace-fidelity
instrumentation kept.

Hypothesis:
- For `QueryLocalUtility` with `learned_segment_budget`, primary segment
  allocation should use pooled final point-score segment scores rather than the
  compressed segment-budget head.
- This should remove a diagnosed non-causal segment child path without changing
  metric/profile/target/model/prior semantics.

Changed files:
- `selection/learned_segment_budget/core.py`
- `orchestration/selector_diagnostics.py`
- `orchestration/retained_mask_stage.py`
- `orchestration/selection_causality_diagnostics.py`
- `tests/unit/selection/test_query_driven_learned_segment_budget.py`
- `tests/unit/orchestration/test_retained_mask_stage.py`
- `tests/unit/learning/test_model_learning_does_not_collapse.py`
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`
- `docs/query-driven-implementation-research-guide.md`

Validation commands:
- `python -m py_compile selection/learned_segment_budget/core.py orchestration/selector_diagnostics.py orchestration/retained_mask_stage.py orchestration/selection_causality_diagnostics.py scoring/methods.py learning/checkpoint_validation.py orchestration/retained_mask_ablation_stage.py`
- `ruff check selection/learned_segment_budget/core.py orchestration/selector_diagnostics.py orchestration/retained_mask_stage.py orchestration/selection_causality_diagnostics.py scoring/methods.py learning/checkpoint_validation.py orchestration/retained_mask_ablation_stage.py`
- `pyright selection/learned_segment_budget/core.py orchestration/selector_diagnostics.py orchestration/retained_mask_stage.py orchestration/selection_causality_diagnostics.py scoring/methods.py learning/checkpoint_validation.py orchestration/retained_mask_ablation_stage.py`
- `pytest tests/unit/selection/test_query_driven_learned_segment_budget.py tests/unit/orchestration/test_retained_mask_stage.py tests/unit/learning/test_model_learning_does_not_collapse.py::test_validation_checkpoint_scores_report_factorized_causality_deltas -q`
- `pytest tests/unit/orchestration/test_query_driven_causality_and_summary.py tests/unit/orchestration/test_semantic_causality_diagnostic.py -q`
- `jq empty artifacts/results/pooled_point_score_segment_allocation_level1_smoke/rejection_diagnostic.json`

Artifact:
- `artifacts/results/pooled_point_score_segment_allocation_level1_smoke/example_run.json`
- `artifacts/results/pooled_point_score_segment_allocation_level1_smoke/semantic_diagnostic.json`
- `artifacts/results/pooled_point_score_segment_allocation_level1_smoke/rejection_diagnostic.json`

Scale:
- Static checks and focused unit tests.
- One same-seed Level 1 wiring smoke, seed `2557`, matching the additive
  Level 1 reference.
- No strict Level 2 replay, no Level 3, no final grid.

Key results:
- Selector wiring worked: trace source was `point_score_top20_mean`, and target
  diffusion still passed with final support `gt_0.01 = 0.234375`.
- Same-seed additive Level 1 reference:
  MLQDS QueryLocalUtility `0.1064832750`,
  length preservation `0.5402177987`.
- Pooled point-score Level 1 run:
  MLQDS QueryLocalUtility `0.0856186098`,
  uniform `0.1166761729`, Douglas-Peucker `0.0953898417`,
  length preservation `0.5340749041`.
- Delta versus additive reference:
  QueryLocalUtility `-0.0208646652`,
  length preservation `-0.0061428946`.
- Learning-causality readout in the rejected run:
  shuffled-score delta `0.0321358015`,
  untrained delta `0.0424364875`,
  no-behavior-head delta `0.0064257311`,
  shuffled-prior delta `0.0`,
  no-query-prior delta `0.0`,
  no-segment-budget-head delta `0.0`,
  prior-only delta `0.0037730760`.
- Semantic diagnostic still classifies priors as ignored by the model. The
  pooled point-score allocation diagnostic no longer beats primary because it
  is the primary path in this run.
- `path_length_support_allocation_query_local_utility` is `0.1064832750`,
  exactly matching the additive Level 1 reference, but using that as a fix
  would be query-free length/guardrail compensation from failed evidence.

Decision:
- Reject the pooled point-score primary allocation change. It passed trace
  wiring but failed the Level 1 stop condition by degrading both
  QueryLocalUtility and length.
- Production selector semantics were reverted. Only trace-fidelity
  instrumentation was kept.
- Do not rerun pooled point-score promotion, introduce length/path allocation
  scaffolds, selector floors, raw coverage overrides, or weaken guardrails.
- Next admissible step: `pooled_point_score_allocation_failure_diagnosis`.
  Diagnose why the strict Level 2 counterfactual diagnostic favored pooled
  point-score allocation while the same-seed Level 1 primary path failed. Use
  existing artifacts first; no production semantic changes and no Level 2/3 or
  final-grid run.

## Checkpoint Phase 50 - pooled point-score allocation failure diagnosis

Status: completed / diagnostic only, no production-code change.

Hypothesis:
- Pooled final point-score allocation looked favorable in the strict Level 2
  counterfactual diagnostic because that diagnostic did not validate the
  production score-to-mask and length-selection dynamics seen when the same
  signal became the Level 1 primary selector allocation source.

Changed files:
- `artifacts/results/pooled_point_score_allocation_failure_diagnosis/diagnostic.json`
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`
- `docs/query-driven-implementation-research-guide.md`

Validation commands:
- `jq empty artifacts/results/additive_level2_child_gate_root_localization/diagnostic.json`
- `jq empty artifacts/results/pooled_point_score_segment_allocation_level1_smoke/rejection_diagnostic.json`
- `jq empty artifacts/results/pooled_point_score_allocation_failure_diagnosis/diagnostic.json`
- `git diff --check`
- `rg -n 'checkpoint: pooled_point_score_allocation_failure_diagnosis|Next admissible checkpoint:.*pooled_point_score_allocation_failure_diagnosis' docs/Next-Iterations.md docs/query-driven-implementation-research-guide.md`

Artifact:
- `artifacts/results/pooled_point_score_allocation_failure_diagnosis/diagnostic.json`

Scale:
- Derived diagnostic from existing Level 1 and strict Level 2 artifacts.
- No training, no Level 2 replay, no Level 3, no final grid.

Key results:
- Classification:
  `counterfactual_to_production_score_to_mask_mismatch`.
- Same-seed additive reference:
  MLQDS QueryLocalUtility `0.1064832750`,
  retained-segment counts `[3,0,2,3,0,2,3,0,2]`,
  allocation diagnosis `length_support_materially_influences_allocation`.
- Same-seed pooled primary path:
  MLQDS QueryLocalUtility `0.0856186098`,
  retained-segment counts `[3,1,1,3,1,1,3,1,1]`,
  allocation diagnosis `score_dominated_length_support_conflict`.
- Weighted QueryLocalUtility loss is dominated by local components:
  query point recall `-0.0115942029`,
  local interpolation `-0.0042138399`,
  turn coverage `-0.0049234360`,
  length preservation guardrail only `-0.0000614289`.
- Segment-level oracle coverage improves under pooled allocation, but
  QueryLocalUtility drops. Segment coverage is the wrong success proxy here;
  the lost signal is point-level query mass and local trajectory fidelity.
- Existing retained-decision marginal diagnostics still report `segment_score`
  in the learned segment-head scale even when the selector trace source is
  `point_score_top20_mean`. This does not invalidate the rejection, but it is a
  real instrumentation ambiguity for future segment diagnostics.

Decision:
- The pooled point-score promotion remains rejected. The strict Level 2
  counterfactual advantage did not validate the full production retained-mask
  path.
- Do not use path-length-support allocation as a fix. It matched the additive
  Level 1 score in the rejected smoke, but that would be query-free
  length/guardrail compensation from failed evidence.
- Do not redefine no-segment-budget causality semantics to demote the segment
  head; the rejected run made that ablation zero only because primary
  allocation no longer used the head.
- Next admissible step: `segment_allocation_mask_delta_diagnostic`. Compare
  additive and pooled same-seed retained-mask deltas at point, segment,
  query-hit, and local-fidelity component levels, and clarify
  `allocation_score_source` versus `learned_segment_head_score` in marginal
  rows before any new production selector change.

## Checkpoint Phase 51 - segment allocation mask-delta diagnosis

Status: completed / diagnostic only, no production-code change.

Hypothesis:
- The pooled primary path lost QueryLocalUtility because it exchanged
  high-value retained learned points in productive segments for one-point
  learned coverage across more segments.

Changed files:
- `artifacts/results/segment_allocation_mask_delta_diagnostic/diagnostic.json`
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`
- `docs/query-driven-implementation-research-guide.md`

Validation commands:
- `jq empty artifacts/results/pooled_point_score_allocation_failure_diagnosis/diagnostic.json`
- `jq empty artifacts/results/segment_allocation_mask_delta_diagnostic/diagnostic.json`
- `git diff --check`
- `rg -n 'checkpoint: segment_allocation_mask_delta_diagnostic|Next admissible checkpoint:.*segment_allocation_mask_delta_diagnostic' docs/Next-Iterations.md docs/query-driven-implementation-research-guide.md`

Artifact:
- `artifacts/results/segment_allocation_mask_delta_diagnostic/diagnostic.json`

Scale:
- Derived diagnostic from existing same-seed Level 1 artifacts.
- No training, no production-code change, no Level 2/3/final grid.

Key results:
- Classification:
  `learned_slot_spreading_swapped_query_hit_points_for_zero_hit_coverage`.
- Retained mask:
  additive retained count `15`, pooled retained count `15`, common retained
  count `12`, Jaccard `0.6666666667`.
- Pooled removed additive learned points `[82,178,274]` and added
  `[61,157,253]`.
- Removed points:
  raw marginal QueryLocalUtility sum `0.0210543920`,
  query-hit count `2`.
- Added points:
  raw marginal QueryLocalUtility sum `0.0001858789`,
  query-hit count `0`.
- Net added-minus-removed estimate:
  `-0.0208685131`, matching observed pooled-minus-additive
  QueryLocalUtility `-0.0208646652` with residual `0.0000038479`.
- Segment redistribution:
  additive counts `[3,0,2,3,0,2,3,0,2]`;
  pooled counts `[3,1,1,3,1,1,3,1,1]`.

Decision:
- The pooled point-score primary path remains rejected. Its failure is not
  abstract: it swapped two query-hit learned points for zero-hit coverage.
- Do not tune selector allocation source, path-length allocation, floors, raw
  coverage, or guardrails from this evidence.
- Next admissible step:
  `segment_budget_head_compression_root_diagnostic`. Diagnose why the learned
  segment-budget head stays compressed/non-causal despite an oracle-aligned
  segment target, using target/prediction distributions, loss scale, and
  retained-marginal alignment before any selector-allocation patch.

## Checkpoint Phase 52 - segment-budget head compression root diagnostic

Status: completed / diagnostic only, no production-code change.

Hypothesis:
- The segment-budget child gate fails because the learned segment-budget head
  compresses a broad soft target into near-mean calibration, then drives
  allocation with a weak or wrong-way rank signal at the learned retained
  boundary.

Changed files:
- `artifacts/results/segment_budget_head_compression_root_diagnostic/diagnostic.json`
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`
- `docs/query-driven-implementation-research-guide.md`

Validation commands:
- `jq empty artifacts/results/segment_budget_head_compression_root_diagnostic/diagnostic.json`
- `git diff --check`
- `rg -n 'segment_budget_head_topk_rank_loss_level1_wiring|segment_budget_head_compression_root_diagnostic' docs/Next-Iterations.md docs/query-driven-implementation-research-guide.md`

Artifact:
- `artifacts/results/segment_budget_head_compression_root_diagnostic/diagnostic.json`

Scale:
- Derived diagnostic from existing strict Level 2, Phase 48, Phase 49, Phase 50,
  and Phase 51 artifacts.
- No new training, no production-code change, no Level 2/3/final grid.

Key results:
- Classification:
  `broad_soft_target_plus_underpowered_rank_pressure_causes_compressed_wrong_way_segment_head`.
- Segment target distribution is broad:
  positive fraction `0.9444444444`, gt_0.01 support `0.9126984127`,
  top-5%-mass recall `0.1433802217`.
- The target still contains segment signal:
  segment-budget target top20 mean has oracle-mass spearman `0.8431893688`
  and top-25% oracle mass recall `0.4900304343`.
- Learned segment fit is compressed:
  target std `0.2152189165`, prediction std `0.0109573500`,
  prediction std / target std `0.0509125787`, Kendall tau `0.2148176732`.
- The selector follows segment score:
  segment_score_to_allocation_spearman `0.8771587805`.
- At the learned retained boundary, the segment head is wrong-way while other
  scores are positive-aligned:
  raw_score spearman `0.7192082111`,
  query_hit_branch spearman `0.7441348974`,
  behavior_branch spearman `0.7111436950`,
  segment_score spearman `-0.5381231672`,
  segment-score top-minus-bottom marginal `-0.0009740413`.
- Exact train-side marginal teacher signal exists, but naive primary blends are
  not admissible from this evidence:
  separated teacher improves train-side QueryLocalUtility by `0.0061427389`,
  while W10/W25 blends are worse by about `0.00237`.

Decision:
- The root problem is not selector insensitivity and not a missing target signal.
  The segment head is learning a compressed surface from a broad soft target,
  and the selector is faithfully using that bad surface.
- Do not rerun pooled point-score promotion, path-length allocation, selector
  allocation-source changes, floors, raw coverage, length scaffolds, or weaker
  guardrails.
- Next admissible step:
  `segment_budget_head_topk_rank_loss_level1_wiring`. Implement a narrow
  segment-budget head rank-pressure fix using the existing active target, then
  validate with static/unit checks and one Level 1 wiring run only.

## Checkpoint Phase 53 - segment-budget top-k rank-loss Level 1 wiring

Status: rejected / production loss patch reverted.

Hypothesis:
- A normalized top-k rank term inside the existing segment-level loss would
  preserve ordering and variance in the segment-budget head without changing
  target or selector semantics.

Changed files:
- Temporary implementation edits to `learning/optimization_epoch.py` and
  `tests/unit/learning/test_query_local_utility_training.py`; reverted after
  the Level 1 stop condition failed.
- `artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring/example_run.json`
- `artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring/semantic_diagnostic.json`
- `artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring/rejection_diagnostic.json`
- `docs/Next-Iterations.md`
- `docs/query-driven-implementation-progress.md`
- `docs/query-driven-implementation-research-guide.md`

Validation commands:
- `/home/aleks_dev/dev_projects/P8/.venv/bin/python -m py_compile Range_QDS/learning/optimization_epoch.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/ruff check Range_QDS/learning/optimization_epoch.py Range_QDS/tests/unit/learning/test_query_local_utility_training.py`
- `/home/aleks_dev/dev_projects/P8/.venv/bin/pytest Range_QDS/tests/unit/learning/test_query_local_utility_training.py -q`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2557 --n_ships 12 --n_points 96 --synthetic_route_families 3 --n_queries 8 --max_queries 64 --epochs 1 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 12000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 2 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --results_dir artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring/example_run.json --output artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring/semantic_diagnostic.json)`
- `jq empty artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring/example_run.json artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring/semantic_diagnostic.json artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring/rejection_diagnostic.json`
- `git diff --check`

Artifact:
- `artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring/example_run.json`
- `artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring/semantic_diagnostic.json`
- `artifacts/results/segment_budget_head_topk_rank_loss_level1_wiring/rejection_diagnostic.json`

Scale:
- Static/unit validation plus one same-seed Level 1 wiring run, seed `2557`.
- No Level 2, no Level 3, no final grid.

Key results:
- Target diffusion passed:
  final support `gt_0.01 = 0.234375`.
- Same-seed additive Level 1 reference:
  MLQDS QueryLocalUtility `0.1064832750`,
  uniform `0.1166761729`,
  Douglas-Peucker `0.0953898417`,
  MLQDS length preservation `0.5402177987`.
- Top-k rank-loss candidate:
  MLQDS QueryLocalUtility `0.1064832750`,
  uniform `0.1166761729`,
  Douglas-Peucker `0.0953898417`,
  MLQDS length preservation `0.5402177987`.
- Candidate minus reference:
  QueryLocalUtility `0.0`, length `0.0`.
- Segment head fit did not materially improve:
  prediction std / target std moved from `0.0259016158` to `0.0265600364`,
  Kendall tau moved from `0.1532986111` to `0.1497829861`,
  top-5% mass recall stayed `0.4318017960`.
- Learned retained-boundary segment alignment weakened:
  segment_score_spearman moved from `0.4014447884` to `0.3808049536`;
  segment-score top-minus-bottom marginal moved from `0.0074108590` to
  `0.0072421090`.
- Learning-causality readout was unchanged:
  no-segment-budget-head ablation delta stayed `0.0251730292`, and
  without-query-prior-features delta stayed `0.0`.

Decision:
- Reject the top-k rank-loss patch. It preserved score and target diffusion, but
  did not materially change the retained mask, segment-head compression,
  segment-head rank fit, or learning-causality readout.
- The production loss change was reverted. Keeping it would leave a failed
  experiment in the production path.
- Do not retry this path with a larger scalar or another rank-loss variant until
  the gradient path is measured.
- Next admissible step:
  `segment_rank_loss_gradient_path_diagnostic`. Diagnose actual segment-rank
  loss magnitude and gradient contribution against point BCE, pooled segment
  BCE, existing pairwise segment loss, auxiliary-loss scaling, and the primary
  budget loss before adding another loss term or weight.

## Checkpoint Phase 54 - non-admissible best-config single benchmark

Status: completed / user-requested benchmark-style run, not admissible final evidence.

Hypothesis:
- A single larger replay of the best current QueryLocalUtility configuration can
  provide an operational datapoint, but cannot satisfy acceptance because the
  required learning-causality work is still blocked.

Changed files:
- `artifacts/results/non_admissible_best_config_single_benchmark/example_run.json`
- `artifacts/results/non_admissible_best_config_single_benchmark/semantic_diagnostic.json`
- `docs/query-driven-implementation-progress.md`

Validation commands:
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.train_and_score --seed 2527 --n_ships 64 --n_points 256 --synthetic_route_families 4 --n_queries 40 --max_queries 384 --epochs 5 --workload range --workload_profile_id range_query_mix --coverage_calibration_mode profile_sampled_query_count --workload_stability_gate_mode final --validation_split_mode source_stratified --range_acceptance_max_attempts 40000 --range_max_coverage_overshoot 0.02 --range_train_workload_replicates 4 --model_type workload_blind_range --range_training_target_mode query_local_utility_factorized --selector_type learned_segment_budget --checkpoint_score_variant query_local_utility --checkpoint_selection_metric uniform_gap --compression_ratio 0.05 --mlqds_temporal_fraction 0.0 --query_local_utility_train_marginal_diagnostics --range_audit_compression_ratios 0.01,0.02,0.05,0.10,0.15,0.20,0.30 --final_metrics_mode diagnostic --results_dir artifacts/results/non_admissible_best_config_single_benchmark)`
- `(cd Range_QDS && /home/aleks_dev/dev_projects/P8/.venv/bin/python -m orchestration.diagnostics.semantic_causality_diagnostic --artifact artifacts/results/non_admissible_best_config_single_benchmark/example_run.json --output artifacts/results/non_admissible_best_config_single_benchmark/semantic_diagnostic.json)`
- `jq empty artifacts/results/non_admissible_best_config_single_benchmark/example_run.json artifacts/results/non_admissible_best_config_single_benchmark/semantic_diagnostic.json`

Artifact:
- `artifacts/results/non_admissible_best_config_single_benchmark/example_run.json`
- `artifacts/results/non_admissible_best_config_single_benchmark/semantic_diagnostic.json`

Scale:
- Single larger synthetic replay, seed `2527`.
- `64` ships, `256` points, `40` requested range queries, `5` epochs,
  `range_query_mix`, train workload replicates `4`, compression ratio `0.05`.
- Included compression audit ratios
  `[0.01,0.02,0.05,0.10,0.15,0.20,0.30]`.
- This intentionally bypassed the current gate order at user request; it is not
  final-grid or acceptance evidence.

Key results:
- Primary matched QueryLocalUtility:
  MLQDS `0.1309654535`,
  uniform `0.1247681518`,
  Douglas-Peucker `0.1153266238`.
- Deltas:
  MLQDS - uniform `+0.0061973017`,
  MLQDS - Douglas-Peucker `+0.0156388297`.
- Global sanity passed:
  length preservation `0.8336178577`,
  avg SED ratio vs uniform `1.4986013224` under the `1.5` cap.
- Target diffusion passed:
  final support `gt_0.01 = 0.2172407670`.
- Workload stability, workload signature, support overlap, predictability, and
  prior-predictive alignment all passed.
- Final claim summary still blocked:
  learning causality failed and final grid remains unrun.
- Semantic diagnostic failed checks:
  `shuffled_prior_fields_should_lose`,
  `without_query_prior_features_should_lose`,
  `without_behavior_utility_head_should_lose`.
- Behavior head readout:
  no-behavior-head delta `0.0023741583`, below the `0.005` materiality
  threshold; classifier says the head learns weak signal but final score
  suppresses it.
- Prior materiality remains weak:
  shuffled-prior delta `0.0`, without-query-prior-features delta
  `0.0003339241`.
- Segment head is material but still poorly aligned:
  no-segment-budget-head ablation delta `0.0153767832`,
  segment_score retained-marginal spearman `-0.0971248003`.

Compression audit summary:
- At ratio `0.01`: MLQDS `0.0422729152`, uniform `0.0480020566`,
  Douglas-Peucker `0.0365121475`.
- At ratio `0.02`: MLQDS `0.0668207658`, uniform `0.0571586316`,
  Douglas-Peucker `0.0629060633`.
- At ratio `0.05`: MLQDS `0.1309654535`, uniform `0.1247681518`,
  Douglas-Peucker `0.1153266238`.
- At ratio `0.10`: MLQDS `0.2306514849`, uniform `0.2402856115`,
  Douglas-Peucker `0.2351846621`.
- At ratio `0.15`: MLQDS `0.3139446457`, uniform `0.3272923061`,
  Douglas-Peucker `0.3055560517`.
- At ratio `0.20`: MLQDS `0.3842742476`, uniform `0.3760423646`,
  Douglas-Peucker `0.3781862680`.
- At ratio `0.30`: MLQDS `0.4787655026`, uniform `0.4619465937`,
  Douglas-Peucker `0.4657829405`.

Decision:
- Useful datapoint, but not acceptance evidence.
- Surface score is promising at ratio `0.05`, and length clears the guardrail at
  this scale. The blocker remains learning causality, especially prior
  materiality and behavior-head materiality.
- Do not claim final success from this run.
- Next admissible step remains `segment_rank_loss_gradient_path_diagnostic`.
