# Query-Driven Checkpoint Progress

This is the short checkpoint log required by `docs/query-driven-rework-guide.md`.
The guide is the source of truth. Raw metrics and stdout belong in
`artifacts/results/`.

## Current State - 2026-05-19

Status: active, not complete.

Active implementation after checkpoint 5.171 uses `QueryLocalUtility` schema
`5` with the simplified `range_query_mix` profile. Checkpoint 5.172 updated
the maintained docs to treat those as the default stack going forward. No
strict workload-health or learning-coherence rerun has been performed under
schema `5` and that simplified profile.

Checkpoint 5.173 refactored the guide into the current implementation/research
guide and removed duplicated checkpoint chronology from that source-of-truth
document. The progress log remains the place for checkpoint history.

Current pre-simplification strict-cell reference with segment aggregation
diagnostics:
- `artifacts/results/query_driven_v2_checkpoint85_segment_aggregation_current_best_strict_local/example_run.json`
- MLQDS QueryLocalUtility: `0.1662115143`
- uniform QueryLocalUtility: `0.1421296610`
- Douglas-Peucker QueryLocalUtility: `0.1671038781`
- Passed: workload stability, support overlap, target diffusion, workload
  signature, prior-predictive alignment, global sanity.
- Failed: predictability, learning causality.
- Final grid: not run.
- Final success allowed: `false`.

Latest rejected strict diagnostic:
- `artifacts/results/query_driven_v2_checkpoint79_final_score_ship_blend_target_current_best_strict_local/example_run.json`
- MLQDS QueryLocalUtility dropped to `0.1592468202`, below both the active
  strict-cell reference and Douglas-Peucker `0.1671038781`.
- The final-score/ship-presence segment-budget variant improves target-side
  ship-evidence rank alignment more than the query-hit/ship variant, but it
  still makes retained-mask quality and causality worse. Reject it as a
  training target.

Latest pre-simplification derived workload/component compatibility diagnostic:
- `artifacts/results/query_driven_v2_checkpoint80_workload_component_compatibility_diagnosis/workload_component_compatibility_diagnosis.json`
- Blocking families in that strict reference are `small_local`,
  `density`, `crossing_turn_change`, and `medium_operational`.
- Persistent negative components are dominated by `ship_f1`,
  `ship_balanced_query_point_recall`, `ship_coverage`, and point-mass recall
  terms.

Latest derived recalibration diagnostic:
- `artifacts/results/query_driven_v2_checkpoint81_recalibration_candidate_diagnosis/workload_component_recalibration_candidate_diagnosis.json`
- A query-local-sensible component-weight candidate flips the post-hoc
  MLQDS-minus-Douglas-Peucker score delta from `-0.0008923639` to
  `0.0029786298`.
- This is high masking risk, not acceptance evidence: it improves by
  downweighting the same ship/point-mass blockers and by profile-weighting away
  from density/small-local weakness.

Latest blocker-preserving recalibration diagnostic:
- `artifacts/results/query_driven_v2_checkpoint82_blocker_preserving_recalibration_diagnosis/workload_component_blocker_preserving_recalibration_diagnosis.json`
- Ship/point-preserving component weights keep ship/point evidence weight at
  `0.55` and still produce a positive post-hoc score delta (`0.0015104602`).
- Status is `still_blocked`: density, crossing, small-local, and
  medium-operational keep unresolved ship-evidence signs under preserved
  critical-family profile pressure.

Latest family trainability strict diagnostic:
- `artifacts/results/query_driven_v2_checkpoint83_family_trainability_current_best_strict_local/example_run.json`
- `small_local` is the severe blocker. Target-side final score
  (`-0.1663`), query-hit (`-0.1105`), behavior (`-0.1067`), and
  segment-budget (`-0.3396`) all rank against family ship-query evidence; the
  trained heads and composed score also remain negative against that evidence.
- `density` is target-side weak in behavior (`-0.0867`) and especially
  segment-budget (`-0.2478`). Trained heads recover weak positive ship-evidence
  signs, but the composed retained-mask signal is still too weak to pass gates.

Latest family-local candidate instrumentation:
- `family_local_target_candidate_alignment` now emits diagnostic-only
  family-local candidates for query-hit/ship blend, ship-gated behavior,
  boundary/replacement/ship score, composed score, and segment budget.
- Active labels, losses, selectors, scoring weights, workload profiles, and
  gates are unchanged. This is Level 0 implementation evidence only.

Latest family-local candidate strict diagnostic:
- `artifacts/results/query_driven_v2_checkpoint84_family_local_candidate_current_best_strict_local/example_run.json`
- Scores and gates reproduce checkpoint83: predictability and learning
  causality still fail, final success remains `false`.
- Point-level family query-hit/ship candidates strongly recover ship-query
  evidence: `small_local` Spearman `0.9740`, `density` Spearman
  `0.9191`.
- The family-local segment-budget candidate is still anti-aligned:
  `small_local` Spearman `-0.5754`, `density` Spearman `-0.3675`, with
  about `0.05` ship-query pair coverage at top-k in both cases.

Latest segment aggregation strict diagnostic:
- `artifacts/results/query_driven_v2_checkpoint85_segment_aggregation_current_best_strict_local/example_run.json`
- Scores and gates reproduce checkpoint84: predictability and learning
  causality still fail, final success remains `false`.
- Two-stage allocation diagnostics change the diagnosis. For `small_local`,
  best segment-candidate two-stage pair coverage is `0.4000` and best mass
  recall is `0.8829`; for `density`, best two-stage pair coverage is
  `0.6075` and best mass recall is `0.7214`.
- Max-pooled and fractional ship-query segment aggregation are plausible
  diagnostic candidates. They are not active training semantics.

Latest guarded segment aggregation target implementation:
- Added non-default target mode
  `query_useful_v1_factorized_segment_budget_query_ship_max_pool`.
- It changes only the segment-budget head target: max-pooled segment
  aggregation over `0.65` normalized query-hit probability plus `0.35`
  normalized ship-query evidence.
- Active `query_useful_v1_factorized` semantics, scalar labels, selectors,
  scoring weights, workload profiles, and gates are unchanged.
- The mode is experimental and `final_success_allowed=false`.

Latest guarded segment aggregation target strict diagnostic:
- `artifacts/results/query_driven_v2_checkpoint86_query_ship_max_pool_target_current_best_strict_local/example_run.json`
- MLQDS QueryUsefulV1: `0.1673482145`
- uniform QueryUsefulV1: `0.1421296610`
- Douglas-Peucker QueryUsefulV1: `0.1671038781`
- Passed: workload stability, support overlap, target diffusion, workload
  signature, prior-predictive alignment, global sanity.
- Failed: predictability, learning causality.
- Final grid: not run.
- Final success allowed: `false`.
- Segment-budget ablation now passes (`0.0061773534` loss versus `0.005`
  minimum), but shuffled scores, prior ablations, and behavior-head ablation
  still fail.
- Target-side family segment-budget alignment improves for `small_local`
  (`0.1236`) and `density` (`0.2469`), but fitted `small_local`
  segment/composed head alignment is still negative (`-0.0836` / `-0.0827`).

Latest derived segment-target transfer diagnostic:
- `artifacts/results/query_driven_v2_checkpoint87_query_ship_max_pool_transfer_diagnosis/query_ship_max_pool_transfer_diagnosis.json`
- Evidence level: derived strict-artifact diagnostic, no new probe.
- The query-ship max-pool target improves MLQDS QueryUsefulV1 by
  `0.0011367002` versus checkpoint85 and makes
  `without_segment_budget_head_should_lose` newly pass.
- Remaining failed causality children: shuffled scores, shuffled priors, no
  query priors, no behavior head.
- `density` now has positive segment target and fitted segment signs
  (`0.2469` target, `0.1040` fitted), but the transfer gap is still large.
- `small_local` and `crossing_turn_change` are the sharper blockers: segment
  target signs flip positive, while fitted segment and composed score signs
  stay negative.

Latest rejected query-ship local-head target strict diagnostic:
- `artifacts/results/query_driven_v2_checkpoint90_query_ship_local_heads_current_best_strict_local/example_run.json`
- Evidence level: Level 3 current-best strict diagnostic, blocked by gates.
- MLQDS QueryUsefulV1: `0.1632708811`
- uniform QueryUsefulV1: `0.1421296610`
- Douglas-Peucker QueryUsefulV1: `0.1671038781`
- Passed: workload stability, support overlap, global sanity.
- Failed: target diffusion, prior-predictive alignment, predictability,
  learning causality.
- Final grid: not run.
- Final success allowed: `false`.
- Target-side family signs improve for `small_local` and
  `crossing_turn_change`, but fitted heads do not transfer: both families still
  have negative fitted query-hit, behavior, replacement, boundary, segment, and
  composed-score Spearman against family ship-query evidence.
- Reject the mode. It broadens behavior support too much and worsens retained
  quality versus checkpoint86 and Douglas-Peucker.

Latest derived query-ship local-head failure diagnostic:
- `artifacts/results/query_driven_v2_checkpoint91_query_ship_local_heads_failure_diagnosis/query_ship_local_heads_failure_diagnosis.json`
- Evidence level: derived strict-artifact diagnostic, no new probe.
- Decision:
  `reject_broad_local_heads_preserve_diffusion_before_next_transfer_probe`.
- Compared with checkpoint86, checkpoint90 regresses target diffusion and
  prior-predictive alignment, and MLQDS QueryUsefulV1 drops by `0.0040773333`.
- The target diffusion failure is localized to
  `conditional_behavior_utility:support_fraction_above_max` (`0.9396` support
  versus `0.5` maximum).
- The prior-predictive regression is `query_hit_spearman_below_min`.
- `small_local` positive target signs still become negative fitted signs:
  q-hit gap `-0.2686`, behavior gap `-0.3864`, segment gap `-0.2000`.
- `crossing_turn_change` has the same transfer failure: q-hit gap `-0.5243`,
  behavior gap `-0.3484`, segment gap `-0.1666`.

Latest diffusion-preserving family/head transfer-path diagnostic:
- `artifacts/results/query_driven_v2_checkpoint92_family_transfer_path_diagnosis/family_transfer_path_diagnosis.json`
- Evidence level: derived strict-artifact diagnostic, no new probe.
- Decision:
  `add_family_conditioned_prior_predictability_before_model_or_scoring_change`.
- checkpoint86 has 11 focused family/head blockers.
- `crossing_turn_change` query-hit, segment-budget, and composed heads fit
  their labels but still misorder ship-query evidence.
- `small_local` segment-budget has the same fit-target/misorder failure; its
  query-hit, behavior, and composed targets are still weak.
- Retained-decision marginal alignment is negative at the corrected layout:
  selector score Spearman `-0.0421` under
  `selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_useful_alignment`.
- The artifact does not expose family-conditioned prior predictability, so the
  next change should add that diagnostic surface before choosing model/loss or
  workload/scoring calibration.

Latest family-conditioned prior predictability strict diagnostic:
- `artifacts/results/query_driven_v2_checkpoint93_family_prior_predictability_max_pool_current_best_strict_local/example_run.json`
- Evidence level: Level 3 current-best strict diagnostic.
- MLQDS QueryUsefulV1: `0.1673482145`
- uniform QueryUsefulV1: `0.1421296610`
- Douglas-Peucker QueryUsefulV1: `0.1671038781`
- Passed: workload stability, support overlap, target diffusion, workload
  signature, prior-predictive alignment, global sanity.
- Failed: predictability, learning causality.
- Family-conditioned prior diagnostics are available and diagnostic-only.
- `crossing_turn_change` prior rank is useful for query-hit (`0.3161` best
  Spearman) and segment-budget (`0.2583`), but behavior prior rank is weak.
- `small_local` prior rank is useful for behavior (`0.1864`) and
  segment-budget (`0.1357`); query-hit rank is positive (`0.2352`) but top-k
  lift is weak (`0.9905`).

Latest family-prior transfer-path derived diagnostic:
- `artifacts/results/query_driven_v2_checkpoint94_family_prior_transfer_path_diagnosis/family_prior_transfer_path_diagnosis.json`
- Evidence level: derived strict-artifact diagnostic, no new probe.
- Decision:
  `diagnose_score_to_selector_marginal_calibration_before_promotion`.
- Family-conditioned prior predictability is now available.
- The same 11 focused family/head transfer blockers remain.
- Retained-decision marginal alignment is still negative at the corrected
  selector-trace path; selector score Spearman is `-0.0408`.

Latest selector-to-retained-marginal calibration diagnostic:
- `artifacts/results/query_driven_v2_checkpoint95_selector_marginal_calibration_diagnosis/selector_marginal_calibration_diagnosis.json`
- Evidence level: derived strict-artifact diagnostic, no new probe.
- Decision:
  `diagnose_train_side_marginal_segment_calibration_not_promotion`.
- Retained-decision marginal alignment is read only from
  `selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_useful_alignment`.
- The artifact separates high-scored low-exact-marginal rows from top exact
  marginal rows that active scores under-rank.
- checkpoint93 has 28 high-score/low-marginal rows, 19 top exact-marginal rows
  ranked in the lower half by selector score, and 19 top exact-marginal rows
  ranked in the lower half by segment score.
- The eval-only separated marginal teacher has viable shape but is not allowed
  as train-side evidence; 4 of its top 10 segment targets are low-ranked by
  selector segment score and allocation weight.

Latest selection-side marginal segment calibration diagnostic:
- `artifacts/results/query_driven_v2_checkpoint96_selection_marginal_segment_calibration_diagnosis/selection_marginal_segment_calibration_diagnosis.json`
- Evidence level: derived strict-artifact diagnostic, no new probe.
- Decision:
  `diagnose_selection_marginal_segment_transfer_before_training_semantics`.
- Selection-side exact marginals are present under
  `selector_trace_diagnostics.selection_primary.retained_decision_marginal_query_useful_alignment`.
- The selection separated marginal teacher is split-eligible:
  `candidate_for_train_side_teacher=true`.
- Current selector-score Spearman against selection exact marginals is
  `-0.1610307043`; segment-score Spearman is `-0.0989569905`.
- 6 of the top 10 selection segment teacher targets are low-ranked by selector
  segment score and allocation weight.
- Selection/eval separated teacher segment overlap is only `4/32`; top-10
  segment-target overlap is `0/10`.

Latest selection-to-eval segment teacher transfer diagnostic:
- `artifacts/results/query_driven_v2_checkpoint97_selection_eval_segment_teacher_transfer_diagnosis/selection_eval_segment_teacher_transfer_diagnosis.json`
- Evidence level: derived strict-artifact diagnostic, no new probe.
- Decision:
  `diagnose_transfer_features_before_guarded_calibration_probe`.
- Non-teacher segments are treated as zero target across all segment candidates.
- Selection/eval positive segment teacher overlap is `4/32`; top 1%, 5%, and
  10% target overlap is zero.
- Sparse selection/eval teacher target Spearman over the positive-target union
  is `-0.7662666997`.
- Simple selector features have only weak consistent positive alignment:
  `segment_score` Spearman is `0.0828684706` on selection and `0.0772136868`
  on eval; `learned_count` Spearman is `0.2038883619` on selection and
  `0.2068493470` on eval.

Latest segment transfer-feature admissibility diagnostic:
- `artifacts/results/query_driven_v2_checkpoint98_selection_segment_transfer_feature_admissibility_diagnosis/selection_segment_transfer_feature_admissibility_diagnosis.json`
- Evidence level: derived strict-artifact diagnostic, no new probe.
- Decision:
  `guarded_pre_selection_transfer_calibration_probe_admissible`.
- The diagnostic separates valid pre-selection features from post-selection
  attribution. `learned_count` remains diagnostic-only and is rejected as
  post-selection coupled.
- Probe-admissible candidates are `segment_score` and
  `segment_score_allocation_weight_zblend`.
- `segment_score`: selection Spearman `0.0828684706`, eval Spearman
  `0.0772136868`, selection top-5% target lift `5.5119235247`, eval top-5%
  target lift `6.1700998036`.
- `segment_score_allocation_weight_zblend`: selection Spearman `0.0555158249`,
  eval Spearman `0.0527781958`.
- The length-support counter-signal candidate is rejected as guard-risk.

Latest guarded segment transfer-calibration strict diagnostic:
- `artifacts/results/query_driven_v2_checkpoint99_segment_transfer_calibration_zblend_current_best_strict_local/example_run.json`
- Evidence level: Level 3 current-best strict diagnostic, blocked by gates.
- MLQDS QueryUsefulV1: `0.1672369132`
- uniform QueryUsefulV1: `0.1421296610`
- Douglas-Peucker QueryUsefulV1: `0.1671038781`
- Passed: workload stability, support overlap, target diffusion, workload
  signature, prior-predictive alignment, global sanity.
- Failed: predictability, learning causality.
- The calibration trace is valid and non-default: mode
  `segment_score_allocation_weight_zblend`, `applied=true`, retained mask
  matches the frozen primary, trace schema `8`, no post-selection attribution,
  no length-support counter-signal, and final effective length-support
  allocation weight `0.0`.
- Versus checkpoint93, MLQDS QueryUsefulV1 drops by `0.0001113013`; the
  MLQDS-minus-Douglas-Peucker margin narrows from `0.0002443363` to
  `0.0001330351`.
- The no-segment-budget-head child still passes (`0.0063452786`), but shuffled
  scores, shuffled priors, no query priors, and no behavior head still fail.

Current blocker:
- Predictability still misses aggregate Spearman and PR-AUC lift.
- Learning causality still fails material ablation deltas.
- Exact retained-decision marginal alignment is still bad. In checkpoint99,
  overall raw, selector, and segment score Spearman are `-0.0643384507`,
  `-0.0410055080`, and `-0.0211297316`.
- Retained-decision marginal alignment lives under
  `selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_useful_alignment`.
  Do not look for it under
  `learning_causality_summary.selection_causality_diagnostics`.

Current interpretation:
- The issue is not simply missing workload priors or insufficient smoke-scale
  validation. The workload profile, QueryUsefulV1 scoring components, target
  heads, and selector allocation are not yet producing a coherent retained-set
  signal.
- Ship-query evidence is a real scoring/target compatibility problem. Query-hit
  labels carry some ship-evidence signal, but current behavior and
  segment-budget targets still do not produce causally useful retained masks.
- Checkpoint83 shows this is family-specific. A segment-budget-only patch or
  another generic reweighting pass is unlikely to fix `small_local`; its
  target-side final/query-hit/behavior/segment signals and fitted heads are all
  misaligned with ship-query evidence.
- Checkpoint84 shows that family-local point scoring is not enough. The
  point-level candidates expose ship-query evidence, but the current segment
  budget construction turns that signal back into anti-aligned segment
  allocation.
- Checkpoint85 shows the previous point-top-k segment view was too coarse:
  two-stage allocation plus within-segment family point choice can recover much
  more ship evidence. The remaining question is whether a guarded segment
  aggregation target variant can train heads and improve unchanged strict gates.
- Pure ship-presence segment budgeting is too blunt. Query-hit/ship-presence and
  final-score/ship-presence segment-budget training variants are also rejected
  at strict scale.
- The query-ship max-pool segment-budget target is promising but not accepted.
  It slightly beats Douglas-Peucker on QueryUsefulV1 at this strict cell, and
  the segment-budget causality child gate passes, but global predictability and
  learning causality still fail. The remaining root issue is transfer/coherence:
  the target-side aggregation signal still does not become a reliable fitted
  `small_local` or `crossing_turn_change` head signal.
- The query-ship local-head target is rejected. It confirms a useful negative:
  making target-side q-hit/behavior signs positive is not enough, and broad
  behavior labels break target diffusion while the fitted family heads remain
  anti-aligned.
- Checkpoint91 confirms the next fix should not broaden local behavior labels.
  Checkpoint92 narrows that path: family-conditioned prior predictability is
  needed before changing model/loss or workload/scoring. Checkpoint93 adds it
  and shows the blocker is not simple prior absence. Checkpoint94 points next
  to score-to-selector retained-marginal calibration and loss/selector transfer.
  Checkpoint95 shows the next branch should be train/selection-side marginal
  segment calibration evidence. Checkpoint96 confirms that selection-side
  exact marginal segment evidence exists, but it is sparse and split-specific.
  Checkpoint97 shows direct selection-to-eval teacher transfer is not coherent,
  while a weak shared selector-feature signal exists. Checkpoint98 separates
  valid pre-selection features from circular post-selection attribution and
  makes a guarded pre-selection transfer-calibration probe admissible. It must
  be non-default and judged by unchanged strict gates. Checkpoint99 rejects the
  simple z-blend as a promotion path: it is correctly guarded, but it does not
  improve predictability, causality, or aggregate retained-mask quality versus
  checkpoint93. Keep workload/scoring compatibility in view.

Do not do next:
- Do not run the final grid.
- Do not loosen gates.
- Do not use tiny smokes as learning-coherence evidence.
- Do not compensate with large temporal scaffolding.
- Do not promote endpoint, path, pure ship-presence, query-hit/ship-presence, or
  final-score/ship-presence proxies into production training semantics.

Next rational work:
- Follow `docs/Next-Iterations.md` and keep `docs/keep-in-mind.md` in view.
- Diagnose score-to-selector retained-marginal calibration using the corrected
  selector-trace retained-decision marginal layout.
- If changing loss/model next, target the fit-target/misorder cases rather than
  adding broad priors or broad behavior labels.
- Do not add another selector blend or segment-budget proxy target.
- Do not broaden QueryUsefulV1 behavior targets again without a sharper
  diffusion-preserving target contract and transfer evidence.
- Do not treat checkpoint90's positive target-side family signs as progress
  unless fitted heads and unchanged gates move with them.

### Checkpoint 5.153 - Guarded Query-Ship Max-Pool Segment Target

Status: completed / Level 0 implementation only.

Goal:
- Isolate a checkpoint85-supported segment aggregation variant without changing
  the active default path.

Changes:
- Added experimental mode
  `query_useful_v1_factorized_segment_budget_query_ship_max_pool`.
- Active mode still uses summed segment mass over the final QueryUsefulV1 point
  score.
- The experimental mode uses max-pooled segment aggregation over a simple
  query-local query-hit/ship-evidence blend and remains blocked from final
  success claims.

Tests:
- `python3 -m py_compile ...`
- `uv run --group dev -- ruff check ...`
- `uv run --group dev -- pyright ...`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/orchestration/test_learning_target_stage.py Range_QDS/tests/guardrails/test_rework_guardrails.py -q`

Experiment artifact:
- path: none
- command: none

Key results:
- Affected test files pass: `141 passed`.
- No learning, causality, or retained-mask quality claim.

Decision:
- Continue to a current-best strict diagnostic before judging training
  coherence.

### Checkpoint 5.154 - Query-Ship Max-Pool Strict Diagnostic

Status: completed / Level 3 current-best strict diagnostic, blocked by gates.

Goal:
- Test whether the guarded max-pool segment-budget target improves retained-mask
  quality and learning causality under unchanged strict gates.

Changes:
- Used non-default target mode
  `query_useful_v1_factorized_segment_budget_query_ship_max_pool`.
- No scoring, selector, workload-profile, scalar-label, or gate change.

Tests:
- `jq empty` on checkpoint86 artifact.
- `git diff --check`.

Experiment artifact:
- path:
  `artifacts/results/query_driven_v2_checkpoint86_query_ship_max_pool_target_current_best_strict_local/example_run.json`
- command: current-best strict local cell with
  `--range_training_target_mode query_useful_v1_factorized_segment_budget_query_ship_max_pool`.

Key results:
- MLQDS QueryUsefulV1: `0.1673482145`
- uniform QueryUsefulV1: `0.1421296610`
- Douglas-Peucker QueryUsefulV1: `0.1671038781`
- gates passed: workload stability, support overlap, target diffusion,
  workload signature, prior-predictive alignment, global sanity.
- gates failed: predictability, learning causality.
- no-segment-budget-head ablation loss improved from checkpoint85 `0.0036430341`
  to `0.0061773534`, clearing that child gate.
- failed causality children remain: shuffled scores, shuffled priors, no query
  priors, no behavior head.

Decision:
- Do not promote the mode. It is a useful partial signal, not acceptance
  evidence. Next checkpoint should diagnose why the improved target-side
  segment signal does not transfer to a coherent `small_local` fitted head and
  prior/behavior causal signal.

### Checkpoint 5.155 - Query-Ship Max-Pool Transfer Diagnosis

Status: completed / derived strict-artifact diagnostic.

Goal:
- Compare checkpoint85 and checkpoint86 to separate target-side segment
  improvement from fitted-head transfer and causality improvement.

Changes:
- Added reusable derived diagnostic module
  `orchestration.query_ship_max_pool_transfer_diagnostic`.
- No model, selector, scoring, workload-profile, target, or gate semantics
  changed.

Tests:
- `python3 -m py_compile Range_QDS/orchestration/query_ship_max_pool_transfer_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- ruff check Range_QDS/orchestration/query_ship_max_pool_transfer_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pyright Range_QDS/orchestration/query_ship_max_pool_transfer_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- focused pytest for the transfer diagnostic.
- `jq empty` on the checkpoint87 artifact.

Experiment artifact:
- path:
  `artifacts/results/query_driven_v2_checkpoint87_query_ship_max_pool_transfer_diagnosis/query_ship_max_pool_transfer_diagnosis.json`
- command:
  `uv run --group dev -- python -m orchestration.query_ship_max_pool_transfer_diagnostic --artifact checkpoint85=... --artifact checkpoint86=... --output ...`

Key results:
- checkpoint86 improves MLQDS QueryUsefulV1 by `0.0011367002` over checkpoint85.
- Newly passing causality child: `without_segment_budget_head_should_lose`.
- Still failing causality children: shuffled scores, shuffled priors, no query
  priors, no behavior head.
- `density`: segment target and fitted segment signs are positive, but
  fitted-minus-target ship-evidence Spearman gap remains `-0.1429`.
- `small_local`: segment target improves from `-0.3396` to `0.1236`, but fitted
  segment remains `-0.0836` and composed score remains `-0.0827`.
- `crossing_turn_change`: segment target improves from `-0.3644` to `0.1142`,
  but fitted segment remains `-0.0598` and composed score remains `-0.0820`.

Decision:
- Do not promote query-ship max-pool. Continue with a targeted family/head
  transfer checkpoint for `small_local` and `crossing_turn_change`; do not rerun
  the strict cell or add selector blends.

### Checkpoint 5.156 - Guarded Query-Ship Local Heads Target

Status: completed / Level 0 implementation only.

Goal:
- Test a root target-contract change for the remaining `small_local` and
  `crossing_turn_change` transfer blocker by making the composed q-hit and
  behavior heads query-local ship-evidence aware.

Changes:
- Added experimental target mode
  `query_useful_v1_factorized_query_ship_local_heads`.
- The mode changes q-hit, behavior, scalar final label, and segment-budget
  target construction. Active `query_useful_v1_factorized` semantics remain
  unchanged.
- Kept selector, scoring weights, workload profile, gates, and active defaults
  unchanged.
- The mode is guarded and `final_success_allowed=false`.

Tests:
- `python3 -m py_compile ...`
- `uv run --group dev -- ruff check ...`
- `uv run --group dev -- pyright ...`
- Focused pytest for the new target mode and guardrails.

Experiment artifact:
- path: none
- command: none

Key results:
- Level 0 only. No learning or retained-mask quality claim.

Decision:
- Continue to strict current-best evidence before judging the target contract.

### Checkpoint 5.157 - Query-Ship Local Heads Strict Diagnostic

Status: completed / Level 3 current-best strict diagnostic, rejected.

Goal:
- Test whether the guarded query-ship local-head target improves family/head
  transfer, retained-mask quality, and causality under unchanged gates.

Changes:
- Used non-default target mode
  `query_useful_v1_factorized_query_ship_local_heads`.
- No selector, scoring weight, workload-profile, or gate change.

Tests:
- `jq empty` on checkpoint90 artifact.

Experiment artifact:
- path:
  `artifacts/results/query_driven_v2_checkpoint90_query_ship_local_heads_current_best_strict_local/example_run.json`
- command: current-best strict local cell with `--query_coverage 0.10`,
  `--max_queries 256`, and
  `--range_training_target_mode query_useful_v1_factorized_query_ship_local_heads`.

Key results:
- MLQDS QueryUsefulV1: `0.1632708811`
- uniform QueryUsefulV1: `0.1421296610`
- Douglas-Peucker QueryUsefulV1: `0.1671038781`
- gates passed: workload stability, support overlap, global sanity.
- gates failed: target diffusion, prior-predictive alignment, predictability,
  learning causality.
- target diffusion fails because `conditional_behavior_utility` support is too
  broad (`0.9396` above the `0.5` maximum).
- `small_local` target-side final/q-hit/behavior/segment signs are positive
  (`0.2274` / `0.1989` / `0.3016` / `0.1236`), but fitted composed/q-hit/
  behavior/segment signs are still negative (`-0.0710` / `-0.0698` /
  `-0.0848` / `-0.0764`).
- `crossing_turn_change` target-side final/q-hit/behavior/segment signs are
  positive (`0.3112` / `0.4492` / `0.2761` / `0.1142`), but fitted
  composed/q-hit/behavior/segment signs are still negative (`-0.0735` /
  `-0.0751` / `-0.0723` / `-0.0524`).
- Learning causality still fails. Only the behavior-head child clears its
  minimum (`0.0057660892`); shuffled scores, untrained model, prior fields,
  no-query-prior, segment-budget head, and prior-only checks fail.

Decision:
- Reject `query_useful_v1_factorized_query_ship_local_heads`. The next work
  should diagnose why positive target-side family signs still do not fit,
  without broadening behavior targets or masking the blocker families away.

### Checkpoint 5.158 - Query-Ship Local Heads Failure Diagnosis

Status: completed / derived strict-artifact diagnostic.

Goal:
- Diagnose checkpoint90 by failed gate and component before changing code again.

Changes:
- Added reusable derived diagnostic module
  `orchestration.query_ship_local_heads_failure_diagnostic`.
- No model, selector, scoring, workload-profile, target, or gate semantics
  changed.

Tests:
- `python3 -m py_compile Range_QDS/orchestration/query_ship_local_heads_failure_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- ruff check Range_QDS/orchestration/query_ship_local_heads_failure_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pyright Range_QDS/orchestration/query_ship_local_heads_failure_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- focused pytest for the failure diagnostic.
- `jq empty` on checkpoint91 artifact.

Experiment artifact:
- path:
  `artifacts/results/query_driven_v2_checkpoint91_query_ship_local_heads_failure_diagnosis/query_ship_local_heads_failure_diagnosis.json`
- command:
  `uv run --group dev -- python -m orchestration.query_ship_local_heads_failure_diagnostic --artifact checkpoint86=... --artifact checkpoint90=... --output ...`

Key results:
- checkpoint90 drops MLQDS QueryUsefulV1 by `0.0040773333` versus checkpoint86.
- Gate regressions versus checkpoint86: target diffusion and
  prior-predictive alignment.
- Target diffusion failure is
  `conditional_behavior_utility:support_fraction_above_max`; behavior support
  is `0.9396` against the `0.5` maximum.
- Prior-predictive failure is `query_hit_spearman_below_min`.
- `small_local` target-to-fit gaps remain large for q-hit (`-0.2686`),
  behavior (`-0.3864`), and segment (`-0.2000`).
- `crossing_turn_change` target-to-fit gaps remain large for q-hit (`-0.5243`),
  behavior (`-0.3484`), and segment (`-0.1666`).

Decision:
- Continue with diffusion-preserving model/loss/prior transfer diagnosis or
  workload/scoring calibration. Do not broaden behavior targets again and do not
  add another selector or segment proxy variant.

### Checkpoint 5.159 - Family Transfer Path Diagnosis

Status: completed / derived strict-artifact diagnostic.

Goal:
- Diagnose checkpoint86's remaining diffusion-preserving family/head transfer
  failure before changing model/loss, selector, workload, or scoring semantics.

Changes:
- Added reusable derived diagnostic module
  `orchestration.family_transfer_path_diagnostic`.
- No model, selector, scoring, workload-profile, target, or gate semantics
  changed.

Tests:
- `python3 -m py_compile Range_QDS/orchestration/family_transfer_path_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- ruff check Range_QDS/orchestration/family_transfer_path_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pyright Range_QDS/orchestration/family_transfer_path_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- focused pytest for the family transfer path diagnostic.
- `jq empty` on checkpoint92 artifact.

Experiment artifact:
- path:
  `artifacts/results/query_driven_v2_checkpoint92_family_transfer_path_diagnosis/family_transfer_path_diagnosis.json`
- command:
  `uv run --group dev -- python -m orchestration.family_transfer_path_diagnostic --artifact checkpoint86=... --output ...`

Key results:
- Evidence level: derived strict-artifact diagnostic, no new probe.
- checkpoint86 has 11 focused family/head blockers.
- `crossing_turn_change` query-hit, segment-budget, and composed heads fit
  their labels but still misorder ship-query evidence.
- `small_local` segment-budget has the same fit-target/misorder failure; its
  query-hit, behavior, and composed targets are still weak.
- Retained-decision marginal alignment is negative at the correct selector
  trace path; selector score Spearman is `-0.0421`.
- Family-conditioned prior predictability is not available in the artifact.

Decision:
- Add family-conditioned prior predictability before changing model/loss or
  workload/scoring semantics. Do not promote checkpoint86.

### Checkpoint 5.160 - Family Prior Predictability Strict Diagnostic

Status: completed / Level 0 instrumentation plus Level 3 strict diagnostic.

Goal:
- Add family-conditioned prior predictability diagnostics and rerun the guarded
  max-pool strict cell before choosing a model/loss or workload/scoring change.

Changes:
- Added diagnostic-only
  `predictability_audit.family_conditioned_prior_predictability`.
- The new rows are not used for gates, training, checkpoint selection, retained
  masks, targets, scoring, or selectors.

Tests:
- `python3 -m py_compile Range_QDS/learning/predictability_audit.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- ruff check Range_QDS/learning/predictability_audit.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pyright Range_QDS/learning/predictability_audit.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- focused pytest for predictability audit.
- `jq empty` on checkpoint93 and checkpoint94 artifacts.

Experiment artifact:
- path:
  `artifacts/results/query_driven_v2_checkpoint93_family_prior_predictability_max_pool_current_best_strict_local/example_run.json`
- command:
  current-best strict local cell with
  `--range_training_target_mode query_useful_v1_factorized_segment_budget_query_ship_max_pool`.
- derived path:
  `artifacts/results/query_driven_v2_checkpoint94_family_prior_transfer_path_diagnosis/family_prior_transfer_path_diagnosis.json`

Key results:
- MLQDS QueryUsefulV1: `0.1673482145`
- uniform QueryUsefulV1: `0.1421296610`
- Douglas-Peucker QueryUsefulV1: `0.1671038781`
- gates passed: workload stability, support overlap, target diffusion,
  workload signature, prior-predictive alignment, global sanity.
- gates failed: predictability, learning causality.
- `crossing_turn_change` has useful query-hit and segment-budget family-prior
  rank, but weak behavior-prior rank.
- `small_local` has useful behavior and segment-budget family-prior rank; its
  query-hit top-k lift is weak.
- Retained-decision marginal selector-score Spearman remains negative
  (`-0.0408`) at the corrected selector-trace path.

Decision:
- Do not promote max-pool. Diagnose score-to-selector retained-marginal
  calibration and fit-target/misorder transfer next.

### Checkpoint 5.161 - Selector Marginal Calibration Diagnosis

Status: completed / derived strict-artifact diagnostic.

Goal:
- Diagnose checkpoint93's retained-decision exact marginal alignment at the
  corrected selector-trace layout before changing model, loss, selector,
  workload, or scoring semantics.

Changes:
- Added reusable derived diagnostic module
  `orchestration.selector_marginal_calibration_diagnostic`.
- The diagnostic reports the canonical retained-marginal layout, top exact
  marginal under-ranking, high-score/low-marginal over-ranking, separated
  marginal teacher segment ranks, and allocation/point-selection context.
- No model, selector, scoring, workload-profile, target, or gate semantics
  changed.

Tests:
- `python3 -m py_compile Range_QDS/orchestration/selector_marginal_calibration_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- ruff check Range_QDS/orchestration/selector_marginal_calibration_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pyright Range_QDS/orchestration/selector_marginal_calibration_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- focused pytest for the selector marginal calibration diagnostic.
- `jq empty` on checkpoint95 artifact.

Experiment artifact:
- path:
  `artifacts/results/query_driven_v2_checkpoint95_selector_marginal_calibration_diagnosis/selector_marginal_calibration_diagnosis.json`
- command:
  `uv run --group dev -- python -m orchestration.selector_marginal_calibration_diagnostic --artifact checkpoint93=... --output ...`

Key results:
- Decision:
  `diagnose_train_side_marginal_segment_calibration_not_promotion`.
- Selector-score Spearman against exact retained-decision marginal
  QueryUsefulV1 remains negative: `-0.0407506153`.
- checkpoint93 has 28 high-score/low-exact-marginal rows.
- 19 top exact-marginal rows are ranked in the lower half by selector score,
  and 19 are ranked in the lower half by segment score.
- The eval-only separated marginal teacher has 32 segment targets and 32 point
  targets, but `candidate_for_train_side_teacher=false`.
- 4 of the top 10 segment teacher targets are low-ranked by selector segment
  score and allocation weight.
- Allocation diagnostics still show score-dominated extra slots:
  length-support-to-allocation Spearman `-0.0126515869`, while
  segment-score-to-allocation Spearman is `0.8385258914`.

Decision:
- Do not promote checkpoint93 or wire the eval exact marginal teacher. The next
  admissible checkpoint should construct or diagnose train/selection-side
  marginal segment calibration evidence under unchanged gates.

### Checkpoint 5.162 - Selection Marginal Segment Calibration Diagnosis

Status: completed / derived strict-artifact diagnostic.

Goal:
- Diagnose whether checkpoint93 already contains train/selection-side exact
  marginal segment teacher evidence, and whether it is safe to promote directly
  into training or selector semantics.

Changes:
- Added reusable derived diagnostic module
  `orchestration.selection_marginal_segment_calibration_diagnostic`.
- The diagnostic compares selection and eval retained-marginal layouts,
  split eligibility, segment teacher rank positions, selection/eval segment
  overlap, and allocation context.
- No model, selector, scoring, workload-profile, target, or gate semantics
  changed.

Tests:
- `python3 -m py_compile Range_QDS/orchestration/selection_marginal_segment_calibration_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- ruff check Range_QDS/orchestration/selection_marginal_segment_calibration_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pyright Range_QDS/orchestration/selection_marginal_segment_calibration_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- focused pytest for the selection marginal segment calibration diagnostic.
- `jq empty` on checkpoint96 artifact.

Experiment artifact:
- path:
  `artifacts/results/query_driven_v2_checkpoint96_selection_marginal_segment_calibration_diagnosis/selection_marginal_segment_calibration_diagnosis.json`
- command:
  `uv run --group dev -- python -m orchestration.selection_marginal_segment_calibration_diagnostic --artifact checkpoint93=... --output ...`

Key results:
- Decision:
  `diagnose_selection_marginal_segment_transfer_before_training_semantics`.
- Selection retained-marginal layout is available and split-eligible:
  `candidate_for_train_side_teacher=true`.
- Selection selector-score Spearman against exact retained-decision marginal
  QueryUsefulV1 is `-0.1610307043`.
- Selection segment-score Spearman is `-0.0989569905`.
- 6 of the top 10 selection segment teacher targets are low-ranked by selector
  segment score and allocation weight.
- Selection/eval segment teacher overlap is `4/32`; top-10 segment-target
  overlap is `0/10`.
- Selection allocation has the same score-dominated shape as eval:
  length-support-to-allocation Spearman `-0.0129085242`, segment-score-to-
  allocation Spearman `0.8385259067`.

Decision:
- Do not wire the selection exact marginal teacher directly into training or
  selectors. The next admissible checkpoint should diagnose selection-to-eval
  segment teacher transfer or build a guarded calibration probe whose success is
  judged by unchanged strict gates.

### Checkpoint 5.163 - Selection/Eval Segment Teacher Transfer Diagnosis

Status: completed / derived strict-artifact diagnostic.

Goal:
- Diagnose whether checkpoint93's split-eligible selection-side segment
  marginal teacher transfers to eval segment targets or to simple query-free
  selector features before building any training or selector calibration probe.

Changes:
- Added reusable derived diagnostic module
  `orchestration.selection_eval_segment_teacher_transfer_diagnostic`.
- The diagnostic treats non-teacher segments as zero target over all segment
  candidates, then compares selection/eval sparse target overlap and feature
  alignment.
- No model, selector, scoring, workload-profile, target, or gate semantics
  changed.

Tests:
- `python3 -m py_compile Range_QDS/orchestration/selection_eval_segment_teacher_transfer_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- ruff check Range_QDS/orchestration/selection_eval_segment_teacher_transfer_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pyright Range_QDS/orchestration/selection_eval_segment_teacher_transfer_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- focused pytest for the selection/eval segment teacher transfer diagnostic.
- `jq empty` on checkpoint97 artifact.

Experiment artifact:
- path:
  `artifacts/results/query_driven_v2_checkpoint97_selection_eval_segment_teacher_transfer_diagnosis/selection_eval_segment_teacher_transfer_diagnosis.json`
- command:
  `uv run --group dev -- python -m orchestration.selection_eval_segment_teacher_transfer_diagnostic --artifact checkpoint93=... --output ...`

Key results:
- Decision:
  `diagnose_transfer_features_before_guarded_calibration_probe`.
- Selection/eval positive teacher segment overlap is `4/32`.
- Top 1%, 5%, and 10% sparse target overlap is zero.
- Sparse selection/eval teacher target Spearman over the positive-target union
  is `-0.7662666997`.
- `segment_score` has weak but consistent positive alignment with segment
  teacher targets: selection `0.0828684706`, eval `0.0772136868`.
- `learned_count` is the strongest shared simple feature: selection
  `0.2038883619`, eval `0.2068493470`.
- Segment length support is consistently negative: selection `-0.0698714922`,
  eval `-0.0645857095`.

Decision:
- Do not train directly on raw selection segment teacher targets. The next
  admissible checkpoint should either diagnose richer transfer features or build
  a guarded transfer-calibration probe whose acceptance is unchanged strict
  retained-mask quality and learning causality, not teacher fit alone.

### Checkpoint 5.164 - Segment Transfer-Feature Admissibility Diagnosis

Status: completed / derived strict-artifact diagnostic.

Goal:
- Separate valid pre-selection transfer features from post-selection attribution
  before deciding whether a guarded segment calibration probe is admissible.

Changes:
- Added reusable derived diagnostic module
  `orchestration.selection_segment_transfer_feature_admissibility_diagnostic`.
- The diagnostic classifies candidate features as pre-selection,
  post-selection-coupled, or guard-counter-signal, then checks selection/eval
  sparse teacher alignment.
- No model, selector, scoring, workload-profile, target, or gate semantics
  changed.

Tests:
- `python3 -m py_compile Range_QDS/orchestration/selection_segment_transfer_feature_admissibility_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- ruff check Range_QDS/orchestration/selection_segment_transfer_feature_admissibility_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pyright Range_QDS/orchestration/selection_segment_transfer_feature_admissibility_diagnostic.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- focused pytest for the transfer-feature admissibility diagnostic.
- `jq empty` on checkpoint98 artifact.

Experiment artifact:
- path:
  `artifacts/results/query_driven_v2_checkpoint98_selection_segment_transfer_feature_admissibility_diagnosis/selection_segment_transfer_feature_admissibility_diagnosis.json`
- command:
  `uv run --group dev -- python -m orchestration.selection_segment_transfer_feature_admissibility_diagnostic --artifact checkpoint93=... --output ...`

Key results:
- Decision:
  `guarded_pre_selection_transfer_calibration_probe_admissible`.
- `learned_count_post_selection_coupled` is rejected despite positive
  selection/eval Spearman because it is post-selection attribution.
- Probe-admissible candidates:
  `segment_score`, `segment_score_allocation_weight_zblend`.
- `segment_score` has selection/eval Spearman `0.0828684706` / `0.0772136868`
  and top-5% target lift `5.5119235247` / `6.1700998036`.
- `segment_score_allocation_weight_zblend` has selection/eval Spearman
  `0.0555158249` / `0.0527781958`.
- `segment_score_length_support_counter_blend` is rejected because it uses a
  guard counter-signal.

Decision:
- A guarded non-default pre-selection segment transfer-calibration probe is now
  admissible. Do not use post-selection attribution, do not use length support
  as a counter-signal, and judge any probe by unchanged strict retained-mask
  quality plus learning causality.

### Checkpoint 5.165 - Guarded Segment Transfer-Calibration Strict Probe

Status: completed / Level 3 current-best strict diagnostic, rejected for
promotion.

Goal:
- Test the checkpoint98-admissible pre-selection
  `segment_score_allocation_weight_zblend` calibration under unchanged strict
  gates.

Changes:
- Added a guarded non-default learned-segment selector mode
  `segment_score_allocation_weight_zblend`.
- The mode z-blends segment score with the pre-selection allocation weight,
  rejects post-selection attribution, avoids length-support counter-signals,
  and sets effective final length-support allocation weight to `0.0` to avoid
  double-counting the query-free guard signal.
- Active defaults remain unchanged.

Tests:
- `python3 -m py_compile` on touched selector/config/orchestration/reporting
  files, including reporting row helpers.
- `uv run --group dev -- ruff check` on touched files.
- `uv run --group dev -- pyright` on touched files.
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/orchestration/test_mlqds_method_factory.py Range_QDS/tests/unit/benchmarking/test_runner.py -q` (`161 passed`)
- `jq empty` on checkpoint99 artifact.

Experiment artifact:
- path:
  `artifacts/results/query_driven_v2_checkpoint99_segment_transfer_calibration_zblend_current_best_strict_local/example_run.json`
- command:
  current-best strict local cell with
  `--range_training_target_mode query_useful_v1_factorized_segment_budget_query_ship_max_pool`
  and
  `--learned_segment_transfer_calibration_mode segment_score_allocation_weight_zblend`.

Key results:
- MLQDS QueryUsefulV1: `0.1672369132`
- uniform QueryUsefulV1: `0.1421296610`
- Douglas-Peucker QueryUsefulV1: `0.1671038781`
- gates passed: workload stability, support overlap, target diffusion,
  workload signature, prior-predictive alignment, global sanity.
- gates failed: predictability, learning causality.
- Versus checkpoint93, MLQDS QueryUsefulV1 is `-0.0001113013` lower and the
  MLQDS-minus-Douglas-Peucker margin narrows to `0.0001330351`.
- Causality still fails shuffled scores, shuffled prior fields, no query-prior
  features, and no behavior head.
- The segment-budget child still passes with delta `0.0063452786`.

Decision:
- Reject this z-blend as a promotion path. It is useful evidence that the
  allocation-weight calibration alone is not the missing trainable signal.
  Continue with workload/scoring/target compatibility diagnosis instead of
  more selector z-blend tuning.

### Checkpoint 5.166 - QueryUsefulV1 Schema 3 Scoring Simplification

Status: completed / Level 0 implementation only.

Goal:
- Remove the `ship_presence_and_coverage` and
  `boundary_and_event_evidence` groups from the primary QueryUsefulV1 score.

Changes:
- Bumped QueryUsefulV1 to schema `3`.
- Removed explicit ship-presence, ship-coverage, and boundary/event evidence
  components from the QueryUsefulV1 aggregate and payload, including the
  ship-coverage term hidden in `ship_balanced_query_point_recall`.
- Renormalized the remaining query-point-mass, query-local-behavior, and
  global-sanity weights.
- Trimmed the workload/component compatibility diagnostic candidates so they
  no longer propose removed QueryUsefulV1 components.
- RangeUsefulLegacy and raw range-audit ship/boundary fields remain available
  as diagnostics.

Tests:
- `python3 -m py_compile Range_QDS/scoring/query_useful_v1.py Range_QDS/orchestration/workload_component_compatibility.py Range_QDS/tests/unit/scoring/test_metrics.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- ruff check Range_QDS/scoring/query_useful_v1.py Range_QDS/orchestration/workload_component_compatibility.py Range_QDS/tests/unit/scoring/test_metrics.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pyright Range_QDS/scoring/query_useful_v1.py Range_QDS/orchestration/workload_component_compatibility.py Range_QDS/tests/unit/scoring/test_metrics.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/scoring/test_metrics.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py -q` (`178 passed`)
- `uv run --group dev -- pytest Range_QDS/tests/unit/benchmarking/test_runner.py Range_QDS/tests/unit/orchestration/test_run_payload.py Range_QDS/tests/unit/orchestration/test_range_diagnostics.py -q` (`55 passed`)
- `git diff --check`

Experiment artifact:
- path: none
- command: none

Key results:
- Implementation only. No strict retraining run has been performed under
  QueryUsefulV1 schema `3`.
- Prior schema `2` QueryUsefulV1 numbers are not directly comparable to schema
  `3` numbers.

Decision:
- Use schema `3` for the next checkpoint evidence. Re-run the required smaller
  strict evidence levels before making any learning-coherence or final-grid
  claims under the simplified metric.

### Checkpoint 5.167 - QueryLocalUtility Schema 4 Point-Mass Rebalance

Status: completed / Level 0 implementation only.

Goal:
- Rebalance the simplified QueryLocalUtility groups to point mass `0.50`,
  query-local behavior `0.45`, and global sanity `0.05`.

Changes:
- Bumped QueryLocalUtility to schema `4`.
- Kept the schema `3` component set: no explicit ship-presence, ship-coverage,
  or boundary/event evidence components in the primary aggregate.
- Rescaled component weights inside each remaining group while preserving the
  previous within-group proportions.

Tests:
- `python3 -m py_compile Range_QDS/scoring/query_local_utility.py Range_QDS/tests/unit/scoring/test_metrics.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- ruff check Range_QDS/scoring/query_local_utility.py Range_QDS/tests/unit/scoring/test_metrics.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pyright Range_QDS/scoring/query_local_utility.py Range_QDS/tests/unit/scoring/test_metrics.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/scoring/test_metrics.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py -q` (`178 passed`)

Experiment artifact:
- path: none
- command: none

Key results:
- Implementation only. No strict retraining run has been performed under
  QueryLocalUtility schema `4`.
- Schema `2` and schema `3` QueryLocalUtility numbers are not directly comparable
  to schema `4` numbers.

Decision:
- Use schema `4` for the next checkpoint evidence. Re-run the required smaller
  strict evidence levels before making learning-coherence or final-grid claims.

### Checkpoint 5.168 - QueryLocalUtility Naming Cleanup

Status: completed / Level 0 implementation only.

Goal:
- Rename the active primary metric from `QueryUsefulV1` to
  `QueryLocalUtility`.

Changes:
- Renamed the scoring module to `scoring/query_local_utility.py`.
- Renamed active score fields, schema fields, component fields, final-gate
  labels, report columns, target mode strings, CLI/config knobs, and factorized
  target/module symbols to `query_local_utility`.
- Updated scripts, regression snapshots, and current README references for the
  new name.
- Did not keep production aliases for the old `query_useful_v1` target modes or
  output fields.
- Promoted shared selection/eval segment-teacher diagnostic helpers from
  private to public names so orchestration guardrails stay clean after the
  rename.

Tests:
- `python3 -m py_compile` on scoring, learning, orchestration, benchmarking,
  config, model, and script Python files.
- `uv run --group dev -- ruff check` on touched production/test surfaces.
- `uv run --group dev -- pyright` on touched production/test surfaces.
- `uv run --group dev -- pytest Range_QDS/tests/unit/scoring/test_metrics.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/benchmarking/test_runner.py -q` (`215 passed`)
- `uv run --group dev -- pytest Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py Range_QDS/tests/unit/learning/test_model_learning_does_not_collapse.py Range_QDS/tests/unit/orchestration/test_learning_target_stage.py Range_QDS/tests/unit/orchestration/test_retained_mask_stage.py Range_QDS/tests/unit/orchestration/test_run_payload.py -q` (`78 passed`)
- `uv run --group dev -- pytest Range_QDS/tests/guardrails/test_rework_guardrails.py Range_QDS/tests/regression/test_benchmark_report_regression.py Range_QDS/tests/regression/test_gate_summary_regression.py -q` (`22 passed`)
- `git diff --check`

Experiment artifact:
- path: none
- command: none

Key results:
- Implementation only. No strict retraining run has been performed under the
  renamed metric surface.
- Historical artifacts using `query_useful_v1` keys are not expected to load
  through the renamed production readers unless a deliberate migration layer is
  added later.

Decision:
- Use `QueryLocalUtility` / `query_local_utility` names for new checkpoints.
  Next evidence remains a focused strict rerun under schema `4`.

### Checkpoint 5.169 - Workload Profile Simplification

Status: completed / Level 0 implementation only.

Goal:
- Simplify the active query workload profile and rename `range_workload_v1` to
  `range_query_mix`.

Changes:
- Renamed active workload profile IDs and benchmark profile from
  `range_workload_v1*` to `range_query_mix*`.
- Removed `boundary_entry_exit`, `crossing_turn_change`, and
  `port_or_approach_zone` from active anchor-family generation.
- Renamed `density_route` to `density`.
- Rebalanced active anchor weights to `density=0.80` and
  `sparse_background_control=0.20`.
- Removed `route_corridor_like` from active footprint families.
- Rebalanced active footprint weights to `small_local=0.2777777777777778`,
  `medium_operational=0.50`, and `large_context=0.2222222222222222`.
- Renamed the train-prior `boundary_entry_exit_likelihood` channel to
  `endpoint_likelihood`.
- Updated active diagnostics, guardrails, docs, and tests for the simplified
  family set. No production aliases were added for the removed profile IDs or
  families.

Tests:
- `python3 -m py_compile` on touched workload, learning, orchestration,
  benchmarking, and test files.
- `uv run --group dev -- ruff check` on touched production/test surfaces.
- `uv run --group dev -- pyright` on touched production/test surfaces.
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/scoring/test_metrics.py Range_QDS/tests/unit/workloads/test_workload_generation.py Range_QDS/tests/property/test_workload_profile_properties.py Range_QDS/tests/guardrails/test_rework_guardrails.py -q` (`220 passed`)
- `uv run --group dev -- pytest Range_QDS/tests/unit/benchmarking/test_runner.py Range_QDS/tests/regression/test_benchmark_report_regression.py Range_QDS/tests/regression/test_gate_summary_regression.py -q` (`40 passed`)
- `uv run --group dev -- pytest Range_QDS/tests/unit/learning/test_model_learning_does_not_collapse.py Range_QDS/tests/unit/orchestration/test_learning_target_stage.py Range_QDS/tests/unit/orchestration/test_retained_mask_stage.py Range_QDS/tests/unit/orchestration/test_run_payload.py -q` (`56 passed`)
- `git diff --check`

Experiment artifact:
- path: none
- command: none

Key results:
- Implementation only. No strict retraining or workload-health rerun has been
  performed for `range_query_mix`.
- Active production modules no longer reference `range_workload_v1`,
  `density_route`, `boundary_entry_exit`, `crossing_turn_change`,
  `port_or_approach_zone`, or `route_corridor_like`.

Decision:
- Use `range_query_mix` for new checkpoints. Next evidence must rerun the
  guide-required smaller strict workload/profile and learning-coherence probes
  before any final-grid or success claim.

### Checkpoint 5.170 - QueryLocalUtility Direct Local Components

Status: completed / Level 0 implementation only.

Goal:
- Simplify point-mass and query-local behavior components, and stop deriving
  active point mass from `range_point_f1` or behavior from fallback audit fields.

Changes:
- Bumped `QueryLocalUtility` to schema `5`.
- Replaced point-mass components with direct `query_point_recall` at weight
  `0.50`.
- Replaced query-local behavior components with direct interpolation fidelity
  (`0.20`), turn-change coverage (`0.15`), and continuity from
  `range_gap_min_coverage` (`0.10`).
- Kept global sanity at `0.05` through endpoint/skeleton, shape, and length
  guardrails.
- Added `query_point_recall` to range-audit rows, method payloads, benchmark
  report rows, and query-family summaries while keeping `range_point_f1` as a
  legacy diagnostic/reporting field only.
- Updated compatibility diagnostics, focused tests, and active docs for the
  schema `5` component names and weights.

Tests:
- `python3 -m py_compile` on touched scoring, orchestration, and focused test
  files.
- `uv run --group dev -- ruff check` on touched scoring, orchestration, and
  focused test files, including reporting row helpers.
- `uv run --group dev -- pyright` on touched scoring, orchestration, and
  focused test files, including reporting row helpers.
- `uv run --group dev -- pytest Range_QDS/tests/unit/scoring/test_metrics.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/benchmarking/test_runner.py Range_QDS/tests/regression/test_benchmark_report_regression.py Range_QDS/tests/regression/test_gate_summary_regression.py -q` (`218 passed`)
- `uv run --group dev -- pytest Range_QDS/tests/unit/learning/test_model_learning_does_not_collapse.py Range_QDS/tests/unit/orchestration/test_learning_target_stage.py Range_QDS/tests/unit/orchestration/test_retained_mask_stage.py Range_QDS/tests/unit/orchestration/test_run_payload.py -q` (`56 passed`)
- `uv run --group dev -- pytest Range_QDS/tests/guardrails/test_rework_guardrails.py Range_QDS/tests/property/test_workload_profile_properties.py Range_QDS/tests/unit/workloads/test_workload_generation.py -q` (`42 passed`)
- `git diff --check`
- `git diff --check`

Experiment artifact:
- path: none
- command: none

Key results:
- Implementation only. No strict retraining, workload-health, or
  learning-coherence rerun has been performed under schema `5`.
- Active `QueryLocalUtility` no longer changes when legacy `range_point_f1` or
  removed ship/boundary components change in isolation.

Decision:
- Use schema `5` for new checkpoints. Next evidence must rerun the
  guide-required smaller strict probes under schema `5` and `range_query_mix`
  before any final-grid or success claim.

### Checkpoint 5.171 - Remove Small-Local Footprint

Status: completed / Level 0 implementation only.

Goal:
- Remove `small_local` from the active `range_query_mix` footprint family set
  so schema `5` behavior scoring is not driven by tiny windows with weak local
  behavior evidence.

Changes:
- Removed `small_local` from active `range_query_mix*` footprint weights and
  footprint definitions.
- Renormalized remaining active footprint weights to
  `medium_operational=0.6923076923076923` and
  `large_context=0.3076923076923077`.
- Updated active workload/scoring compatibility diagnostics so active blocker
  footprint pressure no longer includes `small_local`.
- Updated focused workload/profile tests and active docs. Historical
  `small_local` diagnostic tests and checkpoint notes remain historical
  evidence, not active workload requirements.

Tests:
- `python3 -m py_compile Range_QDS/workloads/generation/workload_profiles.py Range_QDS/orchestration/workload_component_compatibility.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- ruff check Range_QDS/workloads/generation/workload_profiles.py Range_QDS/orchestration/workload_component_compatibility.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/property/test_workload_profile_properties.py`
- `uv run --group dev -- pyright Range_QDS/workloads/generation/workload_profiles.py Range_QDS/orchestration/workload_component_compatibility.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/property/test_workload_profile_properties.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/workloads/test_workload_generation.py Range_QDS/tests/property/test_workload_profile_properties.py Range_QDS/tests/guardrails/test_rework_guardrails.py -q` (`42 passed`)
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/scoring/test_metrics.py -q` (`178 passed`)
- `uv run --group dev -- pytest Range_QDS/tests/unit/benchmarking/test_runner.py Range_QDS/tests/regression/test_benchmark_report_regression.py Range_QDS/tests/regression/test_gate_summary_regression.py -q` (`40 passed`)
- `uv run --group dev -- pytest Range_QDS/tests/unit/learning/test_model_learning_does_not_collapse.py Range_QDS/tests/unit/orchestration/test_learning_target_stage.py Range_QDS/tests/unit/orchestration/test_retained_mask_stage.py Range_QDS/tests/unit/orchestration/test_run_payload.py -q` (`56 passed`)

Experiment artifact:
- path: none
- command: none

Key results:
- Implementation only. No strict retraining, workload-health, or
  learning-coherence rerun has been performed after removing `small_local`.

Decision:
- Use the two-footprint `range_query_mix` profile for new checkpoints. Next
  evidence must rerun the guide-required smaller strict probes before any
  final-grid or success claim.

### Checkpoint 5.172 - Current Default Documentation Alignment

Status: completed / docs only.

Goal:
- Update maintained project docs so new work defaults to `QueryLocalUtility`
  schema `5` and the two-footprint `range_query_mix` workload profile.

Changes:
- Added current-default sections to the guide plus root, scoring, workload,
  learning, orchestration, and benchmarking README docs.
- Documented schema `5` score weights: direct `query_point_recall=0.50`,
  query-local interpolation/turn/continuity totaling `0.45`, and global sanity
  totaling `0.05`.
- Documented active `range_query_mix` family weights:
  `density=0.80`, `sparse_background_control=0.20`,
  `medium_operational=0.6923076923076923`, and
  `large_context=0.3076923076923077`.
- Marked removed metric/profile names and removed workload families as
  historical or diagnostic references, not current defaults.
- Updated `Next-Iterations.md` and code-layout wording for the new default
  names.

Tests:
- `git diff --check`
- Stale-default `rg` scan over maintained docs. Remaining hits
  are explicit historical-name or legacy-diagnostic references.

Experiment artifact:
- path: none
- command: none

Key results:
- Documentation only. No strict retraining, workload-health, or
  learning-coherence rerun was performed.
- Current/default docs now point at schema `5` and the two-footprint
  `range_query_mix` profile instead of relying on historical checkpoint notes.

Decision:
- Use the documented schema `5` / two-footprint `range_query_mix` stack for
  future checkpoints. Do not claim learning coherence until the guide-required
  smaller strict evidence passes.

### Checkpoint 5.173 - Guide Refactor To Current Implementation Contract

Status: completed / docs only.

Goal:
- Make the source-of-truth guide read as the current implementation/research
  guide instead of a historical rework narrative.

Changes:
- Retitled the guide to `Range_QDS Query-Driven Implementation and Research
  Guide`.
- Replaced the long checkpoint-history section with a short current-state
  section that points checkpoint chronology to this progress log.
- Removed stale current-best/checkpoint prose from the guide body and changed
  the roadmap from reset-numbered checkpoints to ordered implementation
  phases.
- Kept current defaults explicit: `QueryLocalUtility` schema `5`,
  `range_query_mix`, `query_local_utility_factorized`,
  `workload_blind_range_v2`, and `learned_segment_budget_v1`.
- Updated neighboring README/layout wording that still called the current
  protocol a rework or redesign.

Tests:
- `git diff --check`
- Stale-guide `rg` scan for `rework`, `redesign`, historical metric/profile
  names, old schemas, and current-best/latest checkpoint prose.

Experiment artifact:
- path: none
- command: none

Key results:
- Documentation only. No strict retraining, workload-health, or
  learning-coherence rerun was performed.
- The guide dropped from about `3720` lines to about `1788` lines and no
  longer embeds checkpoint-by-checkpoint evidence history.
- Remaining guide hits for old metric/profile names are explicit historical
  exclusions or the `range_point_f1` no-fallback caveat.

Decision:
- Treat the guide as the implementation/research contract. Keep checkpoint
  history in this progress log and artifacts, not in the guide.

## Condensed Checkpoint Index

### Checkpoints 1-4.82 - Workloads, Priors, Factorized Baseline

Status: completed.

Decision:
- Workload generation, train-derived priors, factorized QueryLocalUtility, and
  learned segment-budget selection are one contract.
- Direct prior residuals and blunt prior scaling did not solve causality.

### Checkpoints 4.83-5.24 - Length Policy And Pipeline Cleanup

Status: completed.

Decision:
- No-length-repair variants are diagnostic only because they fail length/global
  sanity.
- Orchestration, naming, saved gates, and constants were cleaned up enough for
  auditable probes.

### Checkpoints 5.25-5.99 - Prior Materiality And Head Calibration

Status: completed / rejected variants.

Decision:
- Prior materiality weakens after model/selector propagation.
- Sqrt prior transforms and dense-head rank pressure were rejected.
- Better head fit alone is insufficient.

### Checkpoints 5.100-5.107 - Retained-Marginal Diagnostics

Status: completed / blocked by gates.

Decision:
- Future evidence must tie scores to exact retained-decision marginal
  QueryUsefulV1, not just factorized-label fit or mask movement.
- Retained-marginal alignment and rows stay in selector trace payloads. The
  learning-causality summary must not be treated as the canonical location.

### Checkpoints 5.108-5.121 - Teacher Proxies And Query-Free Guards

Status: completed / rejected.

Decision:
- Endpoint/path support is useful as a diagnostic proxy, but not a valid
  learned-controllable teacher.
- Path support and endpoint support must not be promoted without strict
  causality evidence.

### Checkpoints 5.122-5.131 - Selection-Side Exact Marginal Teacher

Status: completed / rejected direct and hybrid consumers.

Decision:
- The selection-side exact marginal teacher is guarded and emits usable rows.
- Direct and hybrid consumers lose at strict scale. The next root issue is
  workload/scoring/target compatibility, not more selector-blend tuning.

### Checkpoints 5.132-5.136 - Workload/Scoring Ship-Evidence Diagnostics

Status: completed / strict blocker isolated.

Decision:
- The workload-healthy strict cell passes workload/profile/global gates but
  still fails predictability and learning causality.
- MLQDS loses to Douglas-Peucker mainly on ship-level retained evidence.
- Query-hit labels carry ship-evidence signal; behavior and segment-budget
  targets do not.

### Checkpoint 5.137 - Ship-Presence Segment-Budget Candidate Payload

Status: completed / Level 0 instrumentation only.

Goal:
- Add diagnostic-only segment-budget candidates that compare active,
  ship-presence, final-score/ship-presence, and query-hit/ship-presence
  targets.

Changes:
- Added `segment_budget_ship_presence_candidate_alignment`.
- Active training labels stayed unchanged.

Tests:
- `python3 -m py_compile ...`
- `uv run --group dev -- ruff check ...`
- `uv run --group dev -- pyright ...`
- focused target-diagnostic pytest.

Experiment artifact:
- path: none
- command: none

Key results:
- Level 0 only. No learning claim.

Decision:
- Continue to strict-scale diagnostic before any semantics change.

### Checkpoint 5.138 - Ship-Presence Candidate Strict Diagnostic

Status: completed / Level 3 current-best strict diagnostic, blocked by gates.

Goal:
- Evaluate diagnostic-only ship-presence segment-budget candidates at the
  workload-healthy strict shape.

Changes:
- No active training-label change.

Tests:
- `jq empty` and payload checks on checkpoint77 artifact.

Experiment artifact:
- path:
  `artifacts/results/query_driven_v2_checkpoint77_ship_presence_segment_budget_candidate_current_best_strict_local/example_run.json`
- command:
  `uv run --group dev -- python -m orchestration.train_and_score ... --n_ships 384 --n_points 256 --n_queries 48 --range_train_workload_replicates 4 --workload_profile_id range_query_mix_local --final_metrics_mode diagnostic`

Key results:
- MLQDS QueryUsefulV1: `0.1662115143`
- uniform QueryUsefulV1: `0.1421296610`
- Douglas-Peucker QueryUsefulV1: `0.1671038781`
- gates passed: workload stability, support overlap, target diffusion,
  workload signature, prior-predictive alignment, global sanity.
- gates failed: predictability, learning causality.
- Pure ship-presence target improved ship-evidence alignment but harmed
  final-score and query-hit top-k mass.
- Final-score/ship-presence and query-hit/ship-presence blends were the only
  plausible candidates.

Decision:
- Continue with a guarded blended target variant. Do not make it default.

### Checkpoint 5.139 - Query-Hit/Ship Segment-Budget Target Variant

Status: completed / Level 0 implementation.

Goal:
- Add an explicit non-default target mode for a guarded 50/50
  query-hit/ship-presence segment-budget head target.

Changes:
- Added
  `query_useful_v1_factorized_segment_budget_query_hit_ship_blend`.
- Kept scalar final labels unchanged.
- Marked the variant `final_success_allowed=false`.
- Routed target mode through training, fit diagnostics, checkpoint validation,
  predictability audit, scoring, and segment target-oracle diagnostics.
- Added tests that the variant only changes the segment-budget head and is not
  final-candidate eligible.

Tests:
- `python3 -m py_compile ...`
- `uv run --group dev -- ruff check ...`
- `uv run --group dev -- pyright ...`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/orchestration/test_learning_target_stage.py Range_QDS/tests/unit/scoring/test_metrics.py Range_QDS/tests/unit/orchestration/test_run_payload.py Range_QDS/tests/guardrails/test_rework_guardrails.py -q`
- `git diff --check`

Experiment artifact:
- path: none
- command: none

Key results:
- `193 passed`.
- Level 0 only. No learning claim.

Decision:
- Continue to strict diagnostic under unchanged gates.

### Checkpoint 5.140 - Query-Hit/Ship Target Strict Diagnostic

Status: completed / Level 3 current-best strict diagnostic, rejected.

Goal:
- Test whether the guarded query-hit/ship segment-budget target improves
  ship-level retained evidence and learning causality without weakening gates.

Changes:
- No default training semantics changed.

Tests:
- `jq empty` and payload checks on checkpoint78 artifact.

Experiment artifact:
- path:
  `artifacts/results/query_driven_v2_checkpoint78_query_hit_ship_blend_target_current_best_strict_local/example_run.json`
- command:
  `uv run --group dev -- python -m orchestration.train_and_score ... --n_ships 384 --n_points 256 --n_queries 48 --range_train_workload_replicates 4 --workload_profile_id range_query_mix_local --range_training_target_mode query_useful_v1_factorized_segment_budget_query_hit_ship_blend --final_metrics_mode diagnostic`

Key results:
- MLQDS QueryUsefulV1: `0.1588862822`
- uniform QueryUsefulV1: `0.1421296610`
- Douglas-Peucker QueryUsefulV1: `0.1671038781`
- gates passed: workload stability, support overlap, target diffusion,
  workload signature, prior-predictive alignment, global sanity.
- gates failed: predictability, learning causality.
- Predictability still fails Spearman `0.1109086186 < 0.15` and PR-AUC lift
  `1.2304850435 < 1.25`.
- Learning causality fails all material ablation checks. Shuffled-score delta is
  `0.0010208660` against required `0.0100539727`; no segment-budget-head delta
  is `-0.0025751539` against required `0.005`.
- Target-side segment-budget ship-evidence Spearman improves from the active
  `-0.0722770682` to `0.0775842420`, but strict retained-mask quality worsens.
- Retained-decision marginal alignment remains negative for raw, selector, and
  segment scores.

Decision:
- Reject the query-hit/ship segment-budget target. The next checkpoint should
  diagnose scoring/profile/target compatibility, not tune this blend.

### Checkpoint 5.141 - Final-Score/Ship Target Variant Level 0

Status: completed / Level 0 implementation, then cleaned up after strict rejection.

Goal:
- Test the cleaner blended segment-budget hypothesis from checkpoint77:
  final-score/ship-presence should preserve more QueryUsefulV1 shape than the
  query-hit/ship variant while adding ship evidence.

Changes:
- Temporarily added
  `query_useful_v1_factorized_segment_budget_final_score_ship_blend` as a
  non-default target mode.
- Kept scalar final labels unchanged.
- Marked the variant non-final.

Tests:
- `python3 -m py_compile ...`
- `uv run --group dev -- ruff check ...`
- `uv run --group dev -- pyright ...`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/orchestration/test_learning_target_stage.py Range_QDS/tests/guardrails/test_rework_guardrails.py -q`

Experiment artifact:
- path: none
- command: none

Key results:
- Level 0 passed.
- Level 0 is implementation evidence only.

Decision:
- Proceeded to strict diagnostic before making any learning claim.

### Checkpoint 5.142 - Final-Score/Ship Target Strict Diagnostic

Status: completed / Level 3 current-best strict diagnostic, rejected.

Goal:
- Evaluate whether the guarded final-score/ship segment-budget target improves
  ship-level retained evidence and causality under unchanged gates.

Changes:
- No default training semantics changed.

Tests:
- `jq empty` and payload checks on checkpoint79 artifact.

Experiment artifact:
- path:
  `artifacts/results/query_driven_v2_checkpoint79_final_score_ship_blend_target_current_best_strict_local/example_run.json`
- command:
  `uv run --group dev -- python -m orchestration.train_and_score ... --n_ships 384 --n_points 256 --n_queries 48 --range_train_workload_replicates 4 --workload_profile_id range_query_mix_local --range_training_target_mode query_useful_v1_factorized_segment_budget_final_score_ship_blend --final_metrics_mode diagnostic`

Key results:
- MLQDS QueryUsefulV1: `0.1592468202`
- uniform QueryUsefulV1: `0.1421296610`
- Douglas-Peucker QueryUsefulV1: `0.1671038781`
- gates passed: workload stability, support overlap, target diffusion,
  workload signature, prior-predictive alignment, global sanity.
- gates failed: predictability, learning causality.
- Predictability still fails Spearman `0.1109086186 < 0.15` and PR-AUC lift
  `1.2304850435 < 1.25`.
- Learning causality fails all material ablation checks. Shuffled-score delta is
  `-0.0011753976` against required `0.0102702955`; no segment-budget-head delta
  is `-0.0037898325` against required `0.005`.
- Target-side segment-budget ship-evidence Spearman improves to `0.1583725136`,
  but strict retained-mask quality still worsens.
- MLQDS still misses `124` query-hit ships versus uniform `118` and
  Douglas-Peucker `115`.

Decision:
- Reject the final-score/ship segment-budget target. Both ship-blend target
  modes are now removed from active training options. Keep the diagnostic
  candidate payload only.

### Checkpoint 5.143 - Workload/Component Compatibility Diagnosis

Status: completed / derived strict-artifact diagnostic.

Goal:
- Localize which workload families and QueryUsefulV1 components make the
  retained-set signal non-trainable, without another heavy model run.

Changes:
- Added `orchestration.workload_component_compatibility`, a reusable derived
  diagnostic over grouped strict artifact payloads.
- The diagnostic compares MLQDS against Douglas-Peucker by anchor family,
  footprint family, and anchor/footprint family, using both
  `workload_scoring_compatibility_diagnostics` and per-method
  `range_query_metadata_component_summary` QueryUsefulV1 components.
- Added a unit test for blocking-family and component rollup behavior.

Tests:
- `python3 -m py_compile ...`
- `uv run --group dev -- ruff check ...`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_rework.py -q`

Experiment artifact:
- path:
  `artifacts/results/query_driven_v2_checkpoint80_workload_component_compatibility_diagnosis/workload_component_compatibility_diagnosis.json`
- command:
  `uv run --group dev -- python -m orchestration.workload_component_compatibility --artifact active_checkpoint77=... --artifact qhit_ship_checkpoint78=... --artifact final_ship_checkpoint79=... --output ...`

Key results:
- Active checkpoint77 MLQDS minus Douglas-Peucker QueryUsefulV1:
  `-0.0008923639`.
- Blocking families in the active strict reference:
  `small_local` (`range_usefulness_delta=-0.0214317072`,
  missed ships `+2`), `density` (`-0.0143572825`, missed ships `+7`),
  `crossing_turn_change` (`-0.0087157891`, missed ships `+4`), and
  `medium_operational` (`-0.0064233545`, missed ships `+10`).
- Largest persistent weighted component losses are `ship_f1`,
  `ship_balanced_query_point_recall`, `ship_coverage`,
  `query_balanced_point_recall`, and `query_point_mass_ratio`.
- Rejected ship-blend target artifacts widen the same density/small-local
  deficits rather than fixing them.

Decision:
- Continue with workload-profile and QueryUsefulV1 component recalibration
  diagnostics. Do not add another segment-budget proxy target.

### Checkpoint 5.144 - Recalibration Candidate Diagnosis

Status: completed / derived strict-artifact diagnostic.

Goal:
- Test whether a simple query-local component-weight and workload-profile
  candidate exposes a coherent trainable signal or merely hides known blockers.

Changes:
- Extended `orchestration.workload_component_compatibility` with
  diagnostic-only component-weight and profile candidates.
- Added summary fields for candidate score deltas and masking risk.
- Updated the unit test to cover recalibration diagnostics.

Tests:
- `python3 -m py_compile ...`
- `uv run --group dev -- ruff check ...`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_rework.py -q`

Experiment artifact:
- path:
  `artifacts/results/query_driven_v2_checkpoint81_recalibration_candidate_diagnosis/workload_component_recalibration_candidate_diagnosis.json`
- command:
  `uv run --group dev -- python -m orchestration.workload_component_compatibility --artifact active_checkpoint77=... --artifact qhit_ship_checkpoint78=... --artifact final_ship_checkpoint79=... --output ...`

Key results:
- Active weights MLQDS-minus-Douglas-Peucker score delta: `-0.0008923639`.
- Query-local-sensible component weights delta: `0.0029786298`.
- Candidate component plus rebalanced profile improves weighted query-local
  deltas for anchor families (`0.0050129390`) and footprint families
  (`0.0078903899`).
- Masking risk: `high`. Density-route and small-local deficits remain; the
  candidate mostly downweights or profile-weights away from weak ship/point
  evidence.

Decision:
- Do not adopt these weights. The next checkpoint must preserve or improve
  density-route and small-local ship/point evidence under unchanged gates.

### Checkpoint 5.145 - Retained-Marginal Layout Fix

Status: completed / implementation cleanup.

Goal:
- Remove the misleading retained-marginal alignment copy from learning-causality
  summaries so future diagnostics use the canonical selector-trace path.

Changes:
- `learning_causality_summary.selection_causality_diagnostics` no longer gets a
  `retained_decision_marginal_alignment` copy.
- The canonical path is
  `selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_useful_alignment`.
- Updated the unit test and guide text to prevent path misuse.

Tests:
- Covered by the focused validation below.

Experiment artifact:
- path: none
- command: none

Key results:
- Layout ambiguity removed for future run payloads. Existing old artifacts may
  still contain the transitional copy; do not use it as canonical evidence.

Decision:
- Continue with selector-trace-first retained-marginal diagnostics.

### Checkpoint 5.146 - Blocker-Preserving Recalibration Diagnosis

Status: completed / derived strict-artifact diagnostic.

Goal:
- Test whether a scoring/profile candidate can keep density-route/small-local
  pressure and ship/point evidence weight instead of hiding those blockers.

Changes:
- Added `ship_point_preserving_smooth_component_weights_v0`.
- Added `BLOCKER_PRESERVING_QUERY_MIX_V0` and critical-family pressure checks.
- Added `blocker_preserving_outcome` to the derived diagnostic.
- Extended the focused unit test.

Tests:
- `python3 -m py_compile ...`
- `uv run --group dev -- ruff check ...`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_rework.py -q`

Experiment artifact:
- path:
  `artifacts/results/query_driven_v2_checkpoint82_blocker_preserving_recalibration_diagnosis/workload_component_blocker_preserving_recalibration_diagnosis.json`
- command:
  `uv run --group dev -- python Range_QDS/orchestration/workload_component_compatibility.py --artifact active_checkpoint77=... --artifact qhit_ship_checkpoint78=... --artifact final_ship_checkpoint79=... --output ...`

Key results:
- Active weights MLQDS-minus-Douglas-Peucker score delta: `-0.0008923639`.
- Query-local-sensible candidate delta: `0.0029786298`, still masking risk.
- Ship/point-preserving smooth candidate delta: `0.0015104602` with
  ship/point evidence weight still `0.55`.
- Critical-family profile pressure is preserved:
  density-route ratio `0.9931034483`, small-local ratio `1.0`,
  medium-operational ratio `0.9969230769`.
- Status: `still_blocked`. Unresolved families are `small_local`,
  `density`, `medium_operational`, and `crossing_turn_change`.

Decision:
- Do not adopt the candidate. Scoring/profile reweighting alone is not enough.
  Next checkpoint should instrument family-conditioned target/head trainability.

### Checkpoint 5.147 - Family-Conditioned Target/Head Trainability Instrumentation

Status: completed / Level 0 instrumentation.

Goal:
- Add diagnostic surfaces that explain whether blocker families have usable
  target signal and whether trained heads fit that signal.

Changes:
- Added `family_conditioned_target_trainability` to QueryUsefulV1 target
  diagnostics.
- Added `family_conditioned_head_trainability` to factorized head-fit
  diagnostics.
- Wired training fit diagnostics to pass train points, boundaries, and prior
  workload queries into the family-conditioned head diagnostic.
- Added unit coverage for `density` and `small_local` family rows.

Tests:
- `python3 -m py_compile ...`
- `uv run --group dev -- ruff check ...`
- `uv run --group dev -- pyright ...`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_rework.py -q`

Experiment artifact:
- path: none
- command: none

Key results:
- Level 0 only. Future strict artifacts will expose target-side ranker
  alignment and trained-head fit by `anchor_family` and `footprint_family`.
- No labels, losses, selectors, scoring weights, or gates changed.

Decision:
- Continue to a workload-healthy strict diagnostic to inspect
  `density`/`small_local` target and head trainability before proposing
  another target or scoring change.

### Checkpoint 5.148 - Family Trainability Strict Diagnostic

Status: completed / strict diagnostic.

Goal:
- Run the workload-healthy current-best strict shape with the new
  family-conditioned target/head diagnostics and localize the active blocker.

Changes:
- Documentation only. No labels, losses, selectors, scoring weights, workload
  profiles, or gates changed.

Tests:
- `jq empty` on checkpoint83 artifact.
- `git diff --check`

Experiment artifact:
- path:
  `artifacts/results/query_driven_v2_checkpoint83_family_trainability_current_best_strict_local/example_run.json`
- command:
  `uv run --group dev -- python -m orchestration.train_and_score ...`
  using the current-best strict local shape with
  `--final_metrics_mode diagnostic`.

Key results:
- MLQDS QueryUsefulV1: `0.1662115143`
- uniform QueryUsefulV1: `0.1421296610`
- Douglas-Peucker QueryUsefulV1: `0.1671038781`
- gates passed: workload stability, support overlap, target diffusion,
  workload signature, prior-predictive alignment, global sanity.
- gates failed: predictability, learning causality.
- `small_local` is weak on both sides: target-side final/query-hit/behavior/
  segment-budget rankers are negative against family ship-query evidence, and
  fitted head/composed predictions remain negative.
- `density` is mainly target-side weak in behavior and segment-budget.
  Head predictions are weakly positive against ship-query evidence but not
  enough to make the retained-mask decision causal.

Decision:
- Continue with diagnostic-only family-local target/head construction for
  `small_local` and `density`. Do not promote a new scoring/profile
  default or run the final grid from this evidence.

### Checkpoint 5.149 - Family-Local Candidate Target Diagnostics

Status: completed / Level 0 instrumentation.

Goal:
- Add diagnostic-only family-local target candidates for the checkpoint83
  `small_local` and `density` blockers without changing active semantics.

Changes:
- Added `family_local_target_candidate_alignment` to QueryUsefulV1 target
  diagnostics.
- Candidate rankers include a family query-hit/ship blend, ship-gated behavior,
  boundary/replacement/ship score, composed score, and segment budget.
- Kept active labels, losses, selectors, scoring weights, workload profiles,
  and gates unchanged.
- Added focused unit coverage for schema and a synthetic small-local candidate
  signal probe.

Tests:
- `python3 -m py_compile Range_QDS/learning/targets/query_useful_v1.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- ruff check Range_QDS/learning/targets/query_useful_v1.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pyright Range_QDS/learning/targets/query_useful_v1.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_rework.py -q` (`113 passed`)

Experiment artifact:
- path: none
- command: none

Key results:
- Level 0 only. The new payload emits candidate rows by family and marks itself
  `diagnostic_only_family_local_not_training_semantics`.
- The synthetic small-local probe reports
  `diagnostic_candidate_improves_family_ship_signal`, but this is not evidence
  of training coherence.

Decision:
- Continue to a strict-scale diagnostic that reads
  `family_local_target_candidate_alignment` for `small_local` and
  `density`. Do not promote the candidates from unit tests or tiny probes.

### Checkpoint 5.150 - Family-Local Candidate Strict Diagnostic

Status: completed / strict diagnostic.

Goal:
- Run the workload-healthy current-best strict shape with
  `family_local_target_candidate_alignment` and inspect `small_local` and
  `density`.

Changes:
- Documentation only after the run. Active labels, losses, selectors, scoring
  weights, workload profiles, and gates stayed unchanged.

Tests:
- `jq empty` on checkpoint84 artifact.
- `git diff --check`

Experiment artifact:
- path:
  `artifacts/results/query_driven_v2_checkpoint84_family_local_candidate_current_best_strict_local/example_run.json`
- command:
  `uv run --group dev -- python -m orchestration.train_and_score ...`
  using the current-best strict local shape with
  `--final_metrics_mode diagnostic`.

Key results:
- MLQDS QueryUsefulV1: `0.1662115143`
- uniform QueryUsefulV1: `0.1421296610`
- Douglas-Peucker QueryUsefulV1: `0.1671038781`
- gates passed: workload stability, support overlap, target diffusion,
  workload signature, prior-predictive alignment, global sanity.
- gates failed: predictability, learning causality.
- Family query-hit/ship point candidates improve strict target signal:
  `small_local` Spearman `0.9740` versus best active baseline `-0.1105`;
  `density` Spearman `0.9191` versus best active baseline `0.0832`.
- The family-local segment-budget candidate fails: `small_local` Spearman
  `-0.5754`, `density` Spearman `-0.3675`; both cover only about `5%`
  of ship-query pairs at top-k.

Decision:
- Do not promote the family-local segment-budget candidate. Continue by
  diagnosing segment aggregation/allocation from family-local point signal,
  likely separating point choice from segment budget instead of summing point
  mass into segments.

### Checkpoint 5.151 - Segment Aggregation Diagnostic Instrumentation

Status: completed / Level 0 instrumentation.

Goal:
- Add diagnostic-only segment aggregation variants and separate segment
  allocation from within-segment point choice.

Changes:
- Added pooled segment candidates: query-hit/ship top20 mean, query-hit/ship
  max, composed top20 mean, and fractional ship-query pair segment credit.
- Added two-stage diagnostics that allocate by segment candidate and choose
  points inside selected segments with the family query-hit/ship point ranker.
- Active labels, losses, selectors, scoring weights, workload profiles, and
  gates stayed unchanged.

Tests:
- `python3 -m py_compile Range_QDS/learning/targets/query_useful_v1.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- ruff check Range_QDS/learning/targets/query_useful_v1.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pyright Range_QDS/learning/targets/query_useful_v1.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_rework.py -q` (`113 passed`)

Experiment artifact:
- path: none
- command: none

Key results:
- Level 0 only. The payload emits segment aggregation candidates and two-stage
  pair-coverage/mass-recall fields.

Decision:
- Continue to strict-scale evidence before any target-mode promotion.

### Checkpoint 5.152 - Segment Aggregation Strict Diagnostic

Status: completed / strict diagnostic.

Goal:
- Run the workload-healthy current-best strict shape with segment aggregation
  diagnostics and inspect `small_local` and `density`.

Changes:
- Documentation only after the run. Active labels, losses, selectors, scoring
  weights, workload profiles, and gates stayed unchanged.

Tests:
- `jq empty` on checkpoint85 artifact.
- `git diff --check`

Experiment artifact:
- path:
  `artifacts/results/query_driven_v2_checkpoint85_segment_aggregation_current_best_strict_local/example_run.json`
- command:
  `uv run --group dev -- python -m orchestration.train_and_score ...`
  using the current-best strict local shape with
  `--final_metrics_mode diagnostic`.

Key results:
- MLQDS QueryUsefulV1: `0.1662115143`
- uniform QueryUsefulV1: `0.1421296610`
- Douglas-Peucker QueryUsefulV1: `0.1671038781`
- gates passed: workload stability, support overlap, target diffusion,
  workload signature, prior-predictive alignment, global sanity.
- gates failed: predictability, learning causality.
- `small_local`: max-pooled segment candidate has point-level Spearman `0.6978`
  and two-stage mass recall `0.8829`; best two-stage pair coverage is `0.4000`.
- `density`: pair-fractional segment candidate has best two-stage pair
  coverage `0.6075`; max-pooled segment candidate has best two-stage mass
  recall `0.7214`.

Decision:
- Continue with a guarded, non-default segment aggregation target variant only
  if it uses checkpoint85 evidence. Do not promote the existing sum-based
  family-local segment target or claim success from diagnostic rows.

## Validation

Latest focused validation:
- `git diff --check`
- Stale-guide `rg` scan for rework/redesign wording, old metric/profile names,
  old schemas, and latest/current-best checkpoint prose. Remaining guide hits
  are explicit historical exclusions or the `range_point_f1` no-fallback caveat.
- Surrounding-doc `rg` scan for visible rework/redesign wording in maintained
  README/layout docs.
- Stale-default `rg` scan over maintained docs. Remaining hits
  are explicit historical-name or legacy-diagnostic references.
- `python3 -m py_compile Range_QDS/workloads/generation/workload_profiles.py Range_QDS/orchestration/workload_component_compatibility.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- ruff check Range_QDS/workloads/generation/workload_profiles.py Range_QDS/orchestration/workload_component_compatibility.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/property/test_workload_profile_properties.py`
- `uv run --group dev -- pyright Range_QDS/workloads/generation/workload_profiles.py Range_QDS/orchestration/workload_component_compatibility.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/property/test_workload_profile_properties.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/workloads/test_workload_generation.py Range_QDS/tests/property/test_workload_profile_properties.py Range_QDS/tests/guardrails/test_rework_guardrails.py -q` (`42 passed`)
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/scoring/test_metrics.py -q` (`178 passed`)
- `uv run --group dev -- pytest Range_QDS/tests/unit/benchmarking/test_runner.py Range_QDS/tests/regression/test_benchmark_report_regression.py Range_QDS/tests/regression/test_gate_summary_regression.py -q` (`40 passed`)
- `uv run --group dev -- pytest Range_QDS/tests/unit/learning/test_model_learning_does_not_collapse.py Range_QDS/tests/unit/orchestration/test_learning_target_stage.py Range_QDS/tests/unit/orchestration/test_retained_mask_stage.py Range_QDS/tests/unit/orchestration/test_run_payload.py -q` (`56 passed`)
