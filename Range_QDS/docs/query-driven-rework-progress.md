# Query-Driven Rework Progress

This is the short checkpoint log required by `docs/query-driven-rework-guide.md`.
Detailed stdout and raw metrics are kept in `Range_QDS/artifacts/results/`.

## High-Value Summary

The redesign has made real progress, but it is not complete. The project has moved from broad structural uncertainty to a narrower candidate-level blocker. The current best strict synthetic/debug cell beats both final baselines on `QueryUsefulV1`, while workload stability, support overlap, target diffusion, prior predictability, prior-predictive alignment, and workload signature gates pass. The remaining blockers are learning-causality materiality and global sanity, especially length preservation.

Current best single-cell evidence is promising but not final success:

```text
MLQDS QueryUsefulV1:           0.17183721530965693
uniform QueryUsefulV1:         0.14223795796380634
Douglas-Peucker QueryUsefulV1: 0.16362459837911367
length preservation:           0.7941408411227088
```

Interpretation:
- This is the best current candidate because it beats both uniform and Douglas-Peucker in one strict synthetic/debug cell while keeping the workload/prior gates healthy.
- It is not a final success claim because learning causality still fails and length preservation is below the active `0.80` gate.
- The full 4x7 grid should remain unrun until the strict single-cell gates pass.
- The next useful work is not more broad sweeping. It is targeted work on selector/length allocation and material learned causality from the current best candidate.

Major durable discoveries so far:
- Balanced synthetic split cardinalities were necessary to make workload-signature diagnostics meaningful. The old default `70/15/15` synthetic split created misleading raw hit-count and query-count drift.
- Prior predictability became healthy after target/predictability fixes. The current blocker is no longer generic prior support or target diffusion.
- Raw factorized scalar targets plus factorized head base-rate initialization materially improved model calibration and produced the first strict-cell MLQDS win over Douglas-Peucker in this sequence.
- `route_density_prior` is harmful under the current raw-factorized/head-initialized setup. It should stay available for diagnostics/support overlap, but be excluded from v2 model inputs. Do not generalize this finding to older target/model states.
- `learned_segment_length_repair_fraction=0.6` is material to the current best candidate. Removing repair improves `QueryUsefulV1` and some causality signs, but invalidates global geometry. Full repair or stronger geometry repair weakens learned control or loses to Douglas-Peucker.
- Training-fit improvements are not enough. Several changes improved fit diagnostics but worsened retained-mask quality.
- Behavior-head rank loss at weight `0.15` is rejected as a default. It slightly
  improved behavior-head train fit, but worsened retained-mask causality and
  reduced the strict-cell score.
- Lowering `learned_segment_allocation_weight_floor` from `0.50` to `0.10` is
  rejected as a standalone fix. It increased mask movement, but most ablations
  beat the primary selector, MLQDS regressed to `0.15366824272250135`, and
  length worsened to `0.7833962145166923`.

Current research question:

```text
Can the selector/model make train-derived prior, behavior, and score perturbations materially affect frozen retained masks while preserving at least 0.80 length and the current MLQDS win over uniform and Douglas-Peucker?
```

If a future checkpoint does not answer that question more clearly, it is probably low-value.

## Current State — 2026-05-18

Status: active, not complete

Best current code candidate:
- `workload_blind_range_v2`
- `route_density_prior` excluded from v2 model inputs
- hidden prior residual scale `0.25`
- no direct prior-to-head residual
- `learned_segment_score_blend_weight=0.05`
- `learned_segment_length_repair_fraction=0.6`
- length repair uses global net-gain allocation
- query-free segment length-support allocation uses
  `learned_segment_allocation_length_support_weight=0.12`
- within-segment geometry tie-breaking uses
  `learned_segment_geometry_gain_weight=0.12`
- as of Checkpoint 5.22, allocation length support is applied even when learned
  segment scores are flat, and fairness preallocation uses the same blended
  allocation weight as the main allocator
- behavior-rank auxiliary is available only as an explicit diagnostic control;
  default `query_useful_behavior_rank_loss_weight=0.0`

Best current strict artifact:
- path: `artifacts/results/query_driven_v2_checkpoint09_model_prior_materiality_strict_replay_c10_r05`

Best current strict result:
- MLQDS QueryUsefulV1: `0.17183721530965693`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- length preservation: `0.7941408411227088`
- gates passed: workload stability, support overlap, predictability, prior-predictive alignment, target diffusion, workload signature
- gates failed: learning causality, global sanity

Current blockers:
- Learning causality still fails. In the best strict artifact, shuffled-score delta is `0.008856345116771192` versus required `0.017759554407510352`; shuffled-prior and no-query-prior deltas are both `0.002743017030572781` versus required `0.005`.
- Several causality checks now pass: untrained model delta `0.0200725509132888`, behavior-head delta `0.00966018148201922`, segment-budget-head delta `0.014543435541987698`, prior-only delta `0.02288860641060822`, and learned-controlled slot fraction `0.33834134615384615` versus minimum `0.25`.
- Length preservation is close but still below the guide's active `0.80` gate: `0.7941408411227088`.
- The separated allocation-length-support ablation is not material:
  `MLQDS_without_segment_length_support_allocation` delta is only
  `0.00012394275871965843`, below the `0.005` material threshold.
- No-length-repair improves MLQDS QueryUsefulV1 to `0.1759846099523811`, but length collapses to `0.6790996203798462` and learning causality still fails. It is a diagnostic, not a candidate.
- Full 4x7 grid remains intentionally unrun because strict single-cell gates still fail.

Current decision:
- Do not run the full grid.
- Do not increase workload/caps yet; current standard strict cell already has healthy accepted query counts.
- Pre-gate benchmark snapshots are allowed only as scarce diagnostics for a
  named scale/runtime/instrumentation question. They cannot replace failed
  strict gates or become a tuning loop.
- Do not lower gates for a success claim while learning causality still fails.
- Do not lower the length gate to `0.75`; that would still leave learning causality failed.
- Keep `learned_segment_length_repair_fraction=0.6` in all summaries of the current candidate. It is material to the best-candidate trade-off.
- Keep behavior-rank disabled by default. The strict replay at weight `0.15`
  regressed the current candidate.
- Keep `learned_segment_allocation_weight_floor=0.50` as the default. The
  `0.10` strict replay made score authority harmful instead of causal.
- Next scientific checkpoint should target either prior-feature materiality or a
  selector mechanism that changes segment allocation, not only
  within-allocation point repair.

Current extra discoveries:
- The best candidate depends materially on `learned_segment_length_repair_fraction=0.6`; summaries must carry this knob because no-repair has stronger score causality but invalid global geometry.
- The score-protected length frontier in the best artifact only clears the `0.80` length gate while protecting about `10%` of budget for top learned-score points. At the guide's `25%` learned-slot materiality floor, the length upper bound is about `0.7911049677462703`, so the current selector/score distribution has a real learned-control-vs-length tension.
- Same-allocation length-only point selection would reach only `0.7597755220341236` length preservation, so the length blocker is not just point choice inside currently selected segments. Segment allocation is still part of the problem.
- Adding query-free segment length support at weight `0.12` gave only a tiny strict-cell gain over Checkpoint 5.18: MLQDS `+0.00012394275871965843`, length `+0.000013005202913918268`, and shuffled-score causality delta `+0.002352417155629921`.
- Checkpoint 5.20 separated query-free segment allocation length support from the
  within-segment geometry tie-breaker. Future artifacts should report
  `learned_segment_allocation_length_support_weight` and
  `MLQDS_without_segment_length_support_allocation` separately from the
  geometry tie-breaker.
- Checkpoint 5.21 replayed the best strict cell with the separated ablation:
  primary behavior was exactly unchanged from Checkpoint 5.19, and the new
  ablation showed allocation length support is too weak to explain a material
  learned win.
- Checkpoint 5.22 found a selector implementation flaw: allocation length
  support was ignored when learned segment scores were flat, and fairness
  preallocation picked by raw score rather than blended allocation weight. This
  matters most for neutral-head ablations and length-support diagnostics.
- Checkpoint 5.23 replayed the strict cell after that fix. Primary score,
  length, shuffled-prior/no-query-prior deltas, learned-slot fraction, and
  allocation length-support materiality were unchanged. Shuffled-score delta
  improved only `+0.00046848550650735454`, segment-budget-head delta improved
  `+0.0028946836748793836`, and prior-only delta improved
  `+0.0007186064333346565`.
- Checkpoint 5.26 showed raw and model-input prior channels are available, but
  active prior changes remain mostly suppressed before retained-mask decisions.
- Checkpoint 5.28 showed behavior-rank weight `0.15` is not the answer:
  behavior-head fit improved only slightly while QueryUsefulV1 and shuffled-score
  causality regressed.
- Checkpoint 5.31 showed a lower allocation floor made allocation visibly less
  uniform but not more useful: segments with learned budget dropped to `768`,
  shuffled-score retained-mask symdiff rose to `1906`, and same-allocation
  length-only preservation collapsed to `0.6968862694377511`.
- Prior-feature materiality remains weak under the lower-floor replay: model
  input prior fields changed by about `0.0128368`, but selector-score delta was
  only about `0.000534` and retained-mask Jaccard stayed `0.9785969084423306`.
- `max_budget_share_per_trajectory` is not a strict hard cap when the fair-share cap is larger; it is effectively `max(share_cap, fair_share_cap)`. Treat it as a soft trajectory-share limit when reasoning about selector allocation caps.

Why this candidate is current best:
- Earlier route-density exclusion failed under the Checkpoint 3.x target/model state, so route density should not be treated as generically bad across all historical runs.
- Checkpoint 4.72 later isolated `route_density_prior` as the dominant harmful prior channel under the newer raw-factorized/head-initialized setup: zeroing only route density improved QueryUsefulV1 to `0.16718745914649327`, while other prior channels were neutral or slightly helpful.
- Checkpoint 4.73 made the narrow code change: keep `route_density_prior` in prior fields for support diagnostics, but zero it for v2 model features.
- Checkpoint 4.74 restored the strict-cell MLQDS win over Douglas-Peucker while keeping the standard workload/prior gates healthy.
- Checkpoint 4.83 showed the current length-repair path suppresses some score/causality upside, but removing it destroys global geometry and still does not pass learning causality. Therefore `learned_segment_length_repair_fraction=0.6` remains part of the best current candidate.
- Checkpoint 5.18 improved the strict-cell MLQDS score over Checkpoint 4.74 with global net-gain repair, but it did not clear learning causality or length.
- Checkpoint 5.19 added query-free segment length-support allocation and produced the best strict-cell score so far, but the gain was too small to change the blocker diagnosis.
- Checkpoint 5.20 cleaned the ablation/config interface around that allocation
  length support, without generating new scientific evidence.
- Checkpoint 5.21 confirmed the cleaned interface preserves the Checkpoint 5.19
  primary result and gives a cleaner negative result for allocation length
  support.
- Checkpoint 5.22 fixed allocation-weight semantics but has not generated new
  strict scientific evidence yet.
- Checkpoint 5.23 supplied that strict replay. It remains blocked by learning
  causality and global sanity.
- Checkpoint 5.26 is the richest strict artifact for the current best behavior
  because it adds model-prior materiality diagnostics without changing the
  primary result.
- Checkpoint 5.28 is a negative behavior-rank replay, not a replacement for the
  current best candidate.
- Checkpoint 5.31 is a negative lower-allocation-floor replay, not a replacement
  for the current best candidate.
- The current problem is not workload health or generic prior harm. The remaining problem is making useful prior/behavior/score perturbations material enough in retained masks while preserving length.

Evidence boundary:
- A strict single-cell win is not a final success claim. Final acceptance still requires all strict single-cell gates plus the full 4x7 coverage/compression grid.
- Any future change must be judged against Checkpoint 5.19 for primary metrics
  and Checkpoint 5.21 for the cleaner separated-ablation diagnostics unless it
  intentionally redefines the candidate baseline. Keep Checkpoint 5.18 as the
  pre-segment-length-support comparison and Checkpoint 4.74 as the
  pre-global-repair historical comparison.
- Checkpoint 5.26 is the current-best strict evidence boundary after the
  allocation-weight and prior-materiality diagnostic fixes.
- Checkpoint 5.28 is rejected evidence for behavior-rank weight `0.15`; after
  Checkpoint 5.29, current-code defaults disable behavior-rank again.
- Checkpoint 5.31 is rejected evidence for
  `learned_segment_allocation_weight_floor=0.10`; keep the Checkpoint 5.26
  current-best evidence boundary.
- Checkpoint 4.83 is useful evidence about the repair-vs-causality trade-off, but it does not replace the current strict candidate because its length is invalid.
- Raw training-fit improvements are not enough. Checkpoint 4.79 showed better fit diagnostics can still worsen retained-mask quality and lose the Douglas-Peucker comparison.
- Length-only improvements are not enough. Checkpoints 4.65, 4.66, and 4.81 improved length slightly or nearly cleared it but weakened MLQDS, learned control, or causality.
- A no-repair score win is not enough. Checkpoint 4.83 beat both baselines on QueryUsefulV1 but failed global sanity badly and still failed learning causality.

Rejected-path memory:

| Path | Best observed effect | Rejection reason |
|---|---:|---|
| no length repair, `learned_segment_length_repair_fraction=0.0` | MLQDS `0.1759846099523811`; learned-controlled slot fraction `0.8461538461538461` | length collapsed to `0.6790996203798462`; learning causality still failed |
| full length repair | length `0.7980194800294772` | learned-controlled slot fraction collapsed to `0.203125`; MLQDS lost to Douglas-Peucker |
| geometry gain `0.25` | length `0.797193150044111` | MLQDS regressed and causality worsened |
| segment length-support allocation `0.12` | MLQDS `0.17183721530965693`; length `0.7941408411227088` | best score so far, but still fails learning causality and global sanity |
| behavior-rank loss weight `0.15` | behavior head tau improved from `-0.01658` to `-0.00302`; top-5 behavior mass recall improved from `0.20415` to `0.21870` | MLQDS regressed to `0.1662931067947708`; shuffled-score delta collapsed to `0.00005168542757363892`; prior/no-prior and no-behavior gates still failed |
| allocation weight floor `0.10` | shuffled-score symdiff `1906`, segment-budget symdiff `1380`, segments with learned budget `768` | MLQDS regressed to `0.15366824272250135`, lost to Douglas-Peucker, length worsened to `0.7833962145166923`, and causality failed by sign |
| full prior residual scale `1.0` after route removal | length `0.7939141083394758` | MLQDS `0.16109363670733973`, lost to Douglas-Peucker; shuffled-score causality failed by sign |
| semantic prior-to-head residual | improved training fit | retained-mask result worsened; MLQDS `0.16054051959902663`, lost to Douglas-Peucker; prior ablations became harmful |
| point-score blend `0.15` | length `0.7943720026689473` | MLQDS `0.1581758366351451`, lost to Douglas-Peucker; shuffled and untrained causality failed by sign |

Next-checkpoint guardrails:
- Prefer narrow changes that preserve the Checkpoint 5.19/5.21 DP win and
  healthy workload/prior gates.
- For length work, preserve learned-controlled slots; do not spend the budget with query-free repair that crowds out learned selection.
- For causality work, focus on making prior/behavior/score perturbations move retained masks materially, not merely improving per-head fit.
- Score-protected length filling is a plausible diagnostic direction, but it must respect the observed frontier: protecting `25%` learned-score budget currently appears incompatible with the `0.80` length gate.
- Do not re-test blunt prior-strength escalation unless there is a new mechanism that explains why it will avoid the Checkpoint 4.76 and 4.79 failures.
- Do not keep lowering allocation floor by itself. Checkpoint 5.31 showed that
  more score authority without length-compatible segment value amplifies bad
  decisions.
- Do not add temporal scaffold or change acceptance thresholds to manufacture a success claim.

Minimum pass condition for the next scientific candidate update:
- Keep the Checkpoint 5.19/5.21 baseline comparable unless there is an explicit
  reason to reset the baseline.
- Preserve the MLQDS win over uniform and Douglas-Peucker on `QueryUsefulV1`.
- Clear `global_sanity_gate`, especially length preservation `>=0.80`.
- Clear `learning_causality_gate` with material deltas, not only correct signs.
- Keep `workload_stability`, `workload_signature`, `support_overlap`, `target_diffusion`, `predictability`, and `prior_predictive_alignment` passing.
- Report whether the change affects learned-controlled slot fraction, segment-budget-head delta, shuffled-score delta, no-prior delta, no-behavior-head delta, and length.

## Checkpoint 1 — Workload Generator And Profile Health

Status: completed

Goal:
- Make range workload generation healthy enough for standard strict single-cell probes.

Changes:
- Aligned `range_workload_v1` footprints and query-plan behavior with the guide.
- Added workload-signature checks for query-count mismatch.
- Added prefix-balanced profile-plan behavior so expanded workloads preserve family mix.

Tests:
- Focused workload/profile tests in `tests/test_query_driven_rework.py`
- `tests/test_query_coverage_generation.py`
- ruff/static checks on changed workload modules

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint01_*`
- command: guide-aligned strict synthetic/debug single-cell probes

Key results:
- Early small probes exposed workload signature and query-count drift.
- Later strict cells generated healthy train/eval/selection workloads with accepted query counts above standard strict diagnostic minimum.

Decision:
- Continue from healthy strict synthetic cells; do not tune model from unhealthy workload evidence.

## Checkpoint 2 — Prior Predictability And Target Alignment

Status: completed

Goal:
- Make train-derived priors and QueryUsefulV1 targets measurable under healthy workloads.

Changes:
- Added prior predictability and target-diffusion diagnostics.
- Added support-overlap and prior-alignment gates.
- Added factorized QueryUsefulV1 target diagnostics.

Tests:
- Focused predictability, prior-field, target, and gate tests in `tests/test_query_driven_rework.py`

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint02_*`

Key results:
- Workload, support, predictability, prior-alignment, target-diffusion, and signature gates pass in the retained strict cell.
- Remaining failures moved to model/selector causality and global sanity.

Decision:
- Continue to model and selector checkpoints.

## Checkpoint 3 — Factorized Model And Selector Diagnostics

Status: completed

Goal:
- Make the learned workload-blind model interpretable and causally diagnosable.

Changes:
- Added `workload_blind_range_v2`.
- Added factorized QueryUsefulV1 heads.
- Added `learned_segment_budget_v1`.
- Added frozen-mask protocol and causality ablations.
- Added selector trace diagnostics, learned-slot accounting, head ablation sensitivity, and length feasibility audits.

Tests:
- Full project pytest currently passes.
- Focused model/selector tests in `tests/test_query_driven_rework.py`

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint03_*`

Key results:
- Early learned runs beat uniform inconsistently but failed causality and global sanity.
- Diagnostics showed retained-mask quality, segment allocation, and prior-path behavior needed targeted fixes.

Decision:
- Continue with targeted model/selector fixes only after gate-level diagnostics identify the component.

## Checkpoint 4.61 — Raw Factorized Scalar Target

Status: completed

Goal:
- Train factorized mode against the raw QueryUsefulV1 scalar target instead of legacy scaled labels.

Changes:
- Added raw scalar target handling for `query_useful_v1_factorized`.
- Kept legacy target scaling for legacy modes.

Tests:
- Focused target and factorized diagnostics tests.
- ruff, pyright, `git diff --check`.

Experiment artifact:
- path: not generated in this code checkpoint

Key results:
- Code checks passed.
- Strict replay still needed after the change.

Decision:
- Continue to strict replay.

## Checkpoint 4.62 — Raw Factorized Strict Replay

Status: completed; diagnostic failed

Goal:
- Test whether raw scalar targets fix factorized learning.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_raw_factorized_scalar_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_raw_factorized_scalar_diagnostic/raw_factorized_scalar_summary.json`

Key results:
- MLQDS QueryUsefulV1: `0.16057549994768916`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- length: `0.7933048661024167`
- gates failed: learning causality, global sanity
- factorized heads were badly calibrated against low base-rate targets.

Decision:
- Fix factorized head initialization.

## Checkpoint 4.63 — Factorized Head Base-Rate Initialization

Status: completed

Goal:
- Initialize factorized output-head biases from empirical training target base rates.

Changes:
- Added factorized head output-bias initialization from target means.
- Added diagnostics and focused regression test.

Tests:
- Focused factorized-head tests.
- ruff, pyright, `git diff --check`.

Experiment artifact:
- path: not generated in this code checkpoint

Key results:
- Focused checks passed.

Decision:
- Continue to strict replay.

## Checkpoint 4.64 — Head-Bias Initialization Strict Replay

Status: completed; improved but still blocked

Goal:
- Test whether base-rate initialization fixes factorized calibration and causality.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_head_bias_init_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_head_bias_init_diagnostic/head_bias_init_summary.json`

Key results:
- MLQDS QueryUsefulV1: `0.16512927110095915`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- length: `0.7931550386328327`
- first strict-cell MLQDS QueryUsefulV1 win over Douglas-Peucker in this sequence
- gates failed: learning causality, global sanity
- prior-feature removal slightly improved score, suggesting harmful prior integration.

Decision:
- Keep head-bias initialization.
- Diagnose global sanity and prior causality separately.

## Checkpoint 4.65 — Full Length Repair Diagnostic

Status: completed; diagnostic failed

Goal:
- Test whether full existing length repair clears the length gate.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_full_length_repair_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_full_length_repair_diagnostic/full_length_repair_summary.json`

Key results:
- MLQDS QueryUsefulV1: `0.16243558593475863`
- length: `0.7980194800294772`
- learned-controlled slot fraction dropped to `0.203125`
- gates failed: learning causality, global sanity

Decision:
- Reject full repair. It weakens learned control and loses to Douglas-Peucker.

## Checkpoint 4.66 — Higher Geometry-Gain Diagnostic

Status: completed; diagnostic failed

Goal:
- Test whether stronger geometry gain clears length without collapsing learned slots.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_geometry_gain025_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_geometry_gain025_diagnostic/geometry_gain025_summary.json`

Key results:
- MLQDS QueryUsefulV1: `0.16022420264941584`
- length: `0.797193150044111`
- learned-controlled slot fraction stayed healthy
- causality worsened

Decision:
- Reject geometry gain `0.25`.

## Checkpoint 4.67 — Query-Prior Branch Initialization

Status: completed

Goal:
- Fix the near-zero prior branch output initialization.

Changes:
- Changed prior output initialization from `std=1e-3` to Xavier.
- Added focused prior-branch tests.

Tests:
- Focused prior-branch tests.
- ruff, pyright, `git diff --check`.

Experiment artifact:
- path: not generated in this code checkpoint

Key results:
- Focused checks passed.

Decision:
- Continue to strict replay.

## Checkpoint 4.68 — Prior-Init Strict Replay

Status: completed; diagnostic failed

Goal:
- Test whether stronger prior initialization makes prior features causally useful.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_prior_init_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_prior_init_diagnostic/prior_init_summary.json`

Key results:
- MLQDS QueryUsefulV1: `0.14892550519596737`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- length: `0.7872785562747836`
- prior features became influential but harmful.

Decision:
- Reject full-strength prior path.

## Checkpoint 4.69 — Bounded Prior Residual Scale

Status: completed

Goal:
- Bound the prior residual scale without returning to near-zero suppression.

Changes:
- Set prior residual scale initialization/reset to `0.25`.
- Updated focused tests.

Tests:
- Focused prior-scale tests.
- ruff, pyright, `git diff --check`.

Experiment artifact:
- path: not generated in this code checkpoint

Key results:
- Focused checks passed.

Decision:
- Continue to strict replay.

## Checkpoint 4.70 — Bounded-Prior Strict Replay

Status: completed; diagnostic failed

Goal:
- Test whether bounded prior scale recovers useful prior sensitivity.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_bounded_prior_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_bounded_prior_diagnostic/bounded_prior_summary.json`

Key results:
- MLQDS QueryUsefulV1: `0.16008457877061275`
- length: `0.7936343750367146`
- prior ablations still improved score
- lost to Douglas-Peucker

Decision:
- Stop scale guessing. Diagnose prior channels.

## Checkpoint 4.71 — Per-Channel Prior Ablation Diagnostics

Status: completed

Goal:
- Add optional per-channel prior ablation diagnostics.

Changes:
- Added `zero_query_prior_field_channels`.
- Added optional per-channel prior ablation diagnostics under `learning_causality_summary.prior_channel_ablation_diagnostics`.
- Added focused tests.

Tests:
- Focused prior-channel tests.
- ruff, pyright, `git diff --check`.

Experiment artifact:
- path: not generated in this code checkpoint

Key results:
- Focused checks passed.

Decision:
- Continue to strict diagnostic replay.

## Checkpoint 4.72 — Prior-Channel Diagnostic Replay

Status: completed; diagnostic succeeded

Goal:
- Identify which prior channel causes harmful prior behavior.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_prior_channel_diag_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_prior_channel_diag_diagnostic/prior_channel_summary.json`

Key results:
- Base MLQDS QueryUsefulV1: `0.16008457877061275`
- zeroing `route_density_prior` alone improved QueryUsefulV1 to `0.16718745914649327`
- other prior channels were neutral or slightly helpful

Decision:
- Remove `route_density_prior` from v2 model inputs while keeping it available for support diagnostics.

## Checkpoint 4.73 — Exclude Route Density From V2 Model Input

Status: completed

Goal:
- Exclude the harmful route-density channel from v2 model features.

Changes:
- Added `WORKLOAD_BLIND_RANGE_V2_MODEL_DISABLED_PRIOR_FIELDS = ("route_density_prior",)`.
- Zeroed disabled prior channels in v2 feature construction.
- Bumped v2 schema to `6`.
- Added focused test proving route density remains in prior sampling but not v2 model features.

Tests:
- Focused route-density exclusion tests.
- ruff, pyright, `git diff --check`.

Experiment artifact:
- path: not generated in this code checkpoint

Key results:
- Focused checks passed.

Decision:
- Continue to strict replay.

## Checkpoint 4.74 — No-Route-Density Strict Replay

Status: completed; best current candidate

Goal:
- Test whether route-density exclusion fixes prior causality and restores the DP comparison.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_no_route_density_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_no_route_density_diagnostic/no_route_density_summary.json`

Key results:
- MLQDS QueryUsefulV1: `0.1669032451715525`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- length: `0.7938149625265364`
- prior/no-head deltas became positive but remained below material thresholds
- gates failed: learning causality, global sanity

Decision:
- Keep route-density exclusion.
- Continue from this candidate.

## Checkpoint 4.75 — Restore Prior Scale After Route Removal

Status: completed

Goal:
- Test a full prior residual scale after removing route density.

Changes:
- Temporarily set prior residual scale back to `1.0`.
- Updated focused tests.

Tests:
- Focused prior-scale tests.
- ruff, pyright, `git diff --check`.

Experiment artifact:
- path: not generated in this code checkpoint

Key results:
- Focused checks passed.

Decision:
- Continue to strict replay.

## Checkpoint 4.76 — No-Route-Density Scale-1 Replay

Status: completed; diagnostic failed

Goal:
- Test whether full prior scale works after route-density removal.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_no_route_density_scale1_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_no_route_density_scale1_diagnostic/no_route_density_scale1_summary.json`

Key results:
- MLQDS QueryUsefulV1: `0.16109363670733973`
- lost to Douglas-Peucker
- shuffled-score causality failed by sign
- length: `0.7939141083394758`

Decision:
- Reject scale `1.0`.

## Checkpoint 4.77 — Revert Failed Prior Scale

Status: completed

Goal:
- Restore the best current code candidate after failed scale test.

Changes:
- Reverted prior scale to `0.25`.
- Reverted schema to `6`.

Tests:
- Focused prior-scale and route-density tests.
- ruff, pyright, `git diff --check`.

Experiment artifact:
- path: not generated in this code checkpoint

Key results:
- Focused checks passed.

Decision:
- Continue from Checkpoint 4.74.

## Checkpoint 4.78 — Semantic Prior-To-Head Residual

Status: completed

Goal:
- Test a direct interpretable prior-to-head residual.

Changes:
- Temporarily added semantic direct prior-to-head residuals.
- Kept route density at zero influence.

Tests:
- Focused prior-branch tests.
- ruff, pyright, `git diff --check`.

Experiment artifact:
- path: not generated in this code checkpoint

Key results:
- Focused checks passed.

Decision:
- Continue to strict replay.

## Checkpoint 4.79 — Semantic Prior Residual Strict Replay

Status: completed; diagnostic failed

Goal:
- Test whether semantic prior residuals make causality material.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_semantic_prior_residual_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_semantic_prior_residual_diagnostic/semantic_prior_residual_summary.json`

Key results:
- MLQDS QueryUsefulV1: `0.16054051959902663`
- lost to Douglas-Peucker
- training fit improved, but retained-mask result worsened
- prior ablations became harmful again

Decision:
- Reject semantic prior residuals.

## Checkpoint 4.80 — Revert Semantic Residual

Status: completed

Goal:
- Remove the failed semantic residual path.

Changes:
- Removed direct prior-head residual code.
- Restored schema `6`.

Tests:
- Focused prior/route-density tests.
- ruff, pyright, `git diff --check`.

Experiment artifact:
- path: not generated in this code checkpoint

Key results:
- Focused checks passed.

Decision:
- Continue from Checkpoint 4.74.

## Checkpoint 4.81 — Higher Point-Score Blend Diagnostic

Status: completed; diagnostic failed

Goal:
- Test whether point scores are underweighted inside learned segments.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_score_blend015_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint04_score_blend015_diagnostic/score_blend015_summary.json`

Key results:
- MLQDS QueryUsefulV1: `0.1581758366351451`
- lost to Douglas-Peucker
- length: `0.7943720026689473`
- shuffled and untrained causality failed by sign

Decision:
- Reject `learned_segment_score_blend_weight=0.15`.
- Continue from Checkpoint 4.74.

## Checkpoint 4.82 — Commit-Prep Cleanup

Status: completed

Goal:
- Prepare the current codebase work for a checkpointed git save.

Changes:
- Condensed this progress log from a long raw checkpoint journal into a short guide-compliant ledger.
- Verified rejected scale-1, semantic-residual, and score-blend escalation paths are not active production code.
- Kept the best current candidate active: route-density excluded from v2 model inputs, prior scale `0.25`, no direct prior-head residual.

Tests:
- `../.venv/bin/ruff check evaluation/baselines.py experiments/benchmark_report.py experiments/experiment_cli.py experiments/experiment_config.py experiments/experiment_data.py experiments/experiment_methods.py experiments/experiment_pipeline.py experiments/range_diagnostics.py experiments/run_ais_experiment.py experiments/run_inference.py models/workload_blind_range_v2.py queries/query_generator.py queries/workload_profiles.py simplification/learned_segment_budget.py simplification/mlqds_scoring.py tests/test_benchmark_runner.py tests/test_experiment_data.py tests/test_query_coverage_generation.py tests/test_query_driven_rework.py tests/test_torch_runtime_controls.py tests/test_training_does_not_collapse.py training/checkpoints.py training/model_features.py training/predictability_audit.py training/query_prior_fields.py training/query_useful_targets.py training/train_model.py training/training_epoch.py training/training_validation.py`
- `../.venv/bin/python -m pyright evaluation/baselines.py experiments/benchmark_report.py experiments/experiment_cli.py experiments/experiment_config.py experiments/experiment_data.py experiments/experiment_methods.py experiments/experiment_pipeline.py experiments/range_diagnostics.py experiments/run_ais_experiment.py experiments/run_inference.py models/workload_blind_range_v2.py queries/query_generator.py queries/workload_profiles.py simplification/learned_segment_budget.py simplification/mlqds_scoring.py tests/test_benchmark_runner.py tests/test_experiment_data.py tests/test_query_coverage_generation.py tests/test_query_driven_rework.py tests/test_torch_runtime_controls.py tests/test_training_does_not_collapse.py training/checkpoints.py training/model_features.py training/predictability_audit.py training/query_prior_fields.py training/query_useful_targets.py training/train_model.py training/training_epoch.py training/training_validation.py`
- `git diff --check`
- `../.venv/bin/python -m pytest tests/test_query_driven_rework.py`
- `../.venv/bin/python -m pytest tests/test_training_does_not_collapse.py tests/test_experiment_data.py tests/test_query_coverage_generation.py`
- `../.venv/bin/python -m pytest tests/test_benchmark_runner.py tests/test_torch_runtime_controls.py`
- `../.venv/bin/python -m pytest`

Experiment artifact:
- path: not generated in this checkpoint
- command: no scientific probe was run; this was a cleanup and verification checkpoint.

Key results:
- ruff passed.
- pyright passed.
- `git diff --check` passed.
- Focused pytest batches passed.
- Full pytest passed: `408 passed, 1 warning`.

Decision:
- Codebase is ready for a checkpoint commit.
- Remaining rework blockers after the save are learning-causality materiality and length preservation.

## Checkpoint 4.83 — No-Length-Repair Causality Diagnostic

Status: completed; diagnostic failed

Goal:
- Test whether the current query-free length-repair swaps are suppressing material learned-score causality.

Changes:
- No code changes.
- Removed one aborted non-comparable artifact before rerunning with the same split geometry as Checkpoint 4.74.

Tests:
- Not run; this checkpoint was experiment-only.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint04_no_length_repair_causality_diag_c10_r05`
- command: strict synthetic single-cell matching Checkpoint 4.74 scale/seed/workload, with `learned_segment_length_repair_fraction=0.0`.

Key results:
- MLQDS QueryUsefulV1: `0.1759846099523811`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- length: `0.6790996203798462`
- learned-controlled slot fraction: `0.8461538461538461`
- gates passed: workload stability, support overlap, predictability, prior-predictive alignment, target diffusion, workload signature
- gates failed: learning causality, global sanity
- causality passed for shuffled scores, untrained model, prior-field-only, and segment-budget-head ablation.
- causality failed for shuffled prior fields, no query-prior features, and no behavior head.

Decision:
- Reject no-repair as a candidate: it destroys global sanity and still does not pass learning causality.
- Do not increase workload/caps for this blocker; the strict cell already has healthy workload scale.
- Next checkpoint should target either score-protected length filling as a query-free selector diagnostic, or a model/prior-path change that makes train-derived priors materially affect retained masks without reintroducing harmful route density.

## Checkpoint 4.84 — Discovery Log Hygiene

Status: completed

Goal:
- Make sure relevant extra discoveries are preserved in log and summary outputs.

Changes:
- Added a current extra-discoveries section near the top of this log.
- Promoted the material length-repair knob, score-protected length frontier conflict, and per-ship-cap naming issue into durable notes.

Tests:
- `git diff --check`

Experiment artifact:
- path: not generated
- command: no probe was run; this was documentation hygiene.

Key results:
- Relevant extra discoveries are now recorded in this log instead of only in chat summaries.

Decision:
- Continue future checkpoints from the Checkpoint 4.74 candidate and keep extra discoveries in both progress-log updates and final summaries.

## Checkpoint 4.85-4.90 — Tooling, Cleanup, And Layout Audit

Status: completed

Goal:
- Improve checkpoint-save hygiene before further redesign work by installing the
  new tooling workflow, cleaning stale docs/code/tests, and recording the next
  structural refactor order.

Changes:
- Reworked Makefiles and active commands around `uv --group dev`.
- Added jq artifact filters, Rich run summaries, Hypothesis property tests,
  pytest-regressions snapshots, yamllint, and property/regression test markers.
- Condensed active docs, restored tooling-guide principles, and removed stale
  claims that QueryUsefulV1 and `query_useful_v1_factorized` were future-only.
- Removed unused compatibility shims and aliases, including stale training,
  simplification, and query-useful target wrapper paths.
- Renamed stale diagnostic/profile fields to current names such as
  `profile_diagnostic_only`, `profile_note`, `selector_final_candidate`, and
  `unspecified`.
- Added guardrails for removed shims, active benchmark profile settings,
  implemented profile choices, scalar-vs-QueryUseful target separation, and
  renamed artifact fields.
- Expanded `CODE_LAYOUT.md` into a top-down architecture map with ownership
  boundaries, pressure points, and recommended extraction order.

Tests:
- `uv sync --group dev`
- `uv lock --check`
- `git diff --check`
- `uv run --group dev -- yamllint .`
- focused Ruff/Pyright checks on edited code and tests
- property/regression tests and focused guardrail, benchmark, model-feature,
  query-generation, and query-driven rework tests
- full Pyright and full pytest runs

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was tooling, docs, cleanup, and
  layout-audit work.

Key results:
- Full pytest progressed from `415 passed, 1 warning` to
  `421 passed, 1 warning` across this cleanup range.
- Full Pyright passed.
- yamllint passed.
- The initial full-Ruff target was still documented as blocked by existing lint
  debt; that debt was not hidden.

Extra discoveries:
- Benchmark/Makefile defaults still pointed at legacy diagnostic artifact
  families at this stage.
- `workload_blind_range_v2.calibration_head` needed an explicit checkpoint
  compatibility policy before removal could be considered.
- `training/` depended upward on `experiments/` through config/runtime helpers;
  neutral ownership was the correct fix.

Decision:
- Tooling, active docs, stale cleanup guardrails, and the layout audit were good
  enough to checkpoint.
- Continue with narrow structural refactors; no scientific success claim is
  made.

## Checkpoint 4.91-5.00 — Early Extractions And Save-Gate Disposition

Status: completed

Goal:
- Execute the first safe extraction sequence from `CODE_LAYOUT.md` and resolve
  cleanup discoveries that blocked a reliable save gate.

Changes:
- Extracted final-candidate gates, pure causality helpers, segment audits,
  length diagnostics, selector diagnostics, and model-ablation helpers from the
  old pipeline path.
- Extracted benchmark table formatting, final-grid acceptance, shared benchmark
  helpers, and runtime/history row helpers from the old benchmark report path.
- Made `make lint` a scoped correctness gate, added `make lint-full`, and fixed
  the broad Ruff debt so full lint passed.
- Moved shared config into `config/` and torch runtime controls into `runtime/`,
  removing the `training -> experiments` dependency.
- Extracted public target-mode registries so CLI choices and guardrails no
  longer imported the large target implementation.
- Changed benchmark Makefile, runner, runtime, preflight, list-runs, and tmux
  defaults away from legacy diagnostic artifact families.
- Documented the `calibration_head` checkpoint-compatibility policy.

Tests:
- `make -C Range_QDS lint`
- `make -C Range_QDS lint-full`
- `make -C Range_QDS typecheck`
- `make -C Range_QDS test`
- `uv run --group dev -- yamllint .`
- `git diff --check -- Range_QDS Makefile pyproject.toml`
- focused Ruff/Pyright and unit/regression tests for each extracted module group
- CLI/help checks and shell syntax checks for benchmark scripts

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was structural refactoring and
  save-gate cleanup.

Key results:
- Full Ruff, full Pyright, yamllint, whitespace checks, shell syntax checks,
  and full pytest passed by the end of the range.
- Full pytest remained at `421 passed, 1 warning`.
- The old pipeline dropped from about `4965` lines to about `2955` lines before
  the later package restructure.
- Benchmark defaults and active docs no longer pointed at legacy diagnostic
  artifact families.

Extra discoveries:
- Selection-causality and broad artifact assembly were not clean early
  extraction targets; forcing them then would have moved coupling rather than
  reducing it.
- `_row_from_run` was identified as the benchmark-report coupling point.
- Full Ruff cleanup exposed real defects and risks, including stale imports,
  closure binding risk, regex escaping issues, and script path-setup exceptions.

Decision:
- Save gates were made real: scoped lint, full lint, typecheck, YAML lint,
  whitespace checks, and full tests passed.
- No scientific success claim is made.

## Checkpoint 5.01-5.06 — High-Level Structure And Core Component Splits

Status: completed

Goal:
- Align the repository with the flow
  `data -> queries -> training -> simplification -> evaluation -> benchmarking`,
  consolidate artifacts under `Range_QDS`, and split large component modules
  into direct owners without compatibility facades.

Changes:
- Removed the old `experiments/` package and split ownership into
  `orchestration/` and `benchmarking/`.
- Reorganized tests into `unit`, `integration`, `guardrails`, `property`, and
  `regression` areas.
- Removed the stale `turn_aware` model path after the file deletion, including
  config choices, checkpoint-loading branches, feature-builder behavior, docs,
  and tests.
- Consolidated generated outputs under `Range_QDS/artifacts/`; root
  `artifacts/` was removed and direct defaults became cwd-independent.
- Split query generation into anchors, coverage/acceptance, profile planning,
  signatures, and workload assembly.
- Split benchmark reporting into report coordination, row fields, audit
  extractors, metrics/status helpers, and paths.
- Split learned segment-budget into allocation, length repair, diagnostics,
  trace, constants, and public core orchestration.
- Split scalar target-family construction out of the stale
  `training.targets.legacy` name into family-specific modules.

Tests:
- `make -C Range_QDS lint`
- `make -C Range_QDS lint-full`
- `make -C Range_QDS typecheck`
- `make -C Range_QDS test`
- `uv run --group dev -- yamllint .`
- `git diff --check -- Range_QDS pyproject.toml .gitignore`
- focused Ruff/Pyright checks and focused tests for each moved component
- CLI/help and import-smoke checks for new package owners

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was structural refactoring and
  artifact-root cleanup.

Key results:
- Full pytest progressed from `420 passed, 1 warning` to
  `424 passed, 1 warning`.
- Full Ruff and full Pyright passed.
- `Range_QDS/artifacts/` became the only active artifact root.
- `queries/generation/workload.py` dropped to `638` lines, benchmark report to
  `75` lines, and learned segment-budget core to `409` lines.

Extra discoveries:
- Deleting only `turn_aware_qds_model.py` would have left runtime-failing live
  references; the whole dispatch/config/docs/test path had to be removed.
- `Range_QDS/artifacts/README.md` had been ignored, so the artifact contract was
  not trackable until ignore rules were fixed.
- `training.targets.legacy` was active scalar target logic hidden behind a stale
  name, not dead code.

Decision:
- The repository structure now matches the intended component flow.
- No compatibility shims were left for moved packages, removed `turn_aware`, or
  deleted target-family facades.
- No scientific success claim is made.

## Checkpoint 5.07-5.12 — Orchestration Pipeline Extraction

Status: completed

Goal:
- Make `orchestration/experiment_pipeline.py` a readable coordinator by
  extracting coherent single-run stages while preserving protocol ordering and
  artifact fields.

Changes:
- Extracted selection-causality diagnostics, final summaries/gates, target
  preparation, retained-mask freezing, retained-mask ablation freezing, and the
  evaluation stage into direct orchestration owners.
- Added focused tests for each extracted boundary and updated active
  orchestration/layout docs.
- Kept final metrics-dump assembly and simplified CSV export in the pipeline
  because those are artifact-contract-sensitive.

Tests:
- `make -C Range_QDS lint`
- `make -C Range_QDS lint-full`
- `make -C Range_QDS typecheck`
- `make -C Range_QDS test`
- `uv run --group dev -- yamllint .`
- `git diff --check -- Range_QDS pyproject.toml .gitignore`
- focused Ruff/Pyright checks for each orchestration stage
- focused tests for selection causality, final summaries, target preparation,
  retained masks, retained-mask ablations, evaluation stage, and regressions
- import-smoke checks for extracted stage APIs

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was structural refactoring.

Key results:
- Full pytest progressed from `425 passed, 1 warning` to
  `438 passed, 1 warning`.
- Full Ruff and full Pyright passed after each extraction.
- `experiment_pipeline.py` dropped from about `3388` lines to `710` lines.
- `retained_masks.py` was narrowed from `1177` lines to `296` after ablation
  freezing moved into its own owner.

Extra discoveries:
- Retained-mask freezing was the correct protocol boundary because masks must
  be frozen before eval query scoring.
- `retained_mask_ablations.py` is still large and should get a local method
  factory before more ablation variants are added.
- The remaining pipeline pressure is metrics-dump/artifact assembly and
  simplified CSV export, which should move only with exact field-name tests.

Decision:
- The single-run pipeline is now a coordinator with extracted stage owners.
- No compatibility shims were introduced; the pipeline imports direct owners.
- No scientific success claim is made.

## Checkpoint 5.13-5.15 — Active Docs, Public APIs, Naming, And Test Guardrails

Status: completed

Goal:
- Clean active documentation, stale/misleading code names, and stale/misleading
  tests after the structural changes, then add guardrails for the cleanup
  outcomes.

Changes:
- Updated active docs for the current artifact family, artifact ownership, and
  current component ownership; condensed duplicated README prose while preserving
  tooling principles.
- Renamed learned segment-budget allocation terminology from ship-level wording
  to trajectory-level wording, including `max_budget_share_per_trajectory`.
- Promoted production-crossing orchestration helpers to public names and used
  verb-led builder/evaluator names where noun names would shadow payloads.
- Updated stale package docstrings that still described runtime/config modules
  as experiment-owned.
- Renamed the stale integration test file to
  `test_pipeline_metrics_reporting.py` and replaced fake `"legacy"` unsupported
  query fixtures with `"unsupported"`.
- Added guardrails for the selector public keyword and for preventing production
  orchestration modules from cross-importing private sibling helpers.

Tests:
- `make -C Range_QDS lint`
- `make -C Range_QDS lint-full`
- `make -C Range_QDS typecheck`
- `make -C Range_QDS test`
- `uv run --group dev -- yamllint .`
- `git diff --check`
- focused Ruff/Pyright and focused tests for docs-sensitive guardrails,
  orchestration public APIs, selector naming, and renamed integration tests
- `ruff format --check` on touched docs-adjacent code/test paths

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was documentation, code naming,
  and test cleanup.

Key results:
- Full pytest progressed from `438 passed, 1 warning` to
  `440 passed, 1 warning`.
- Full Ruff, full Pyright, yamllint, format checks, and whitespace checks passed.
- Search found no remaining repo references to `max_budget_share_per_ship`,
  `ship_allocations`, or `max_per_ship` except the new negative API guardrail.
- Search found no production cross-module private imports inside
  `Range_QDS/orchestration`.

Extra discoveries:
- Several `legacy_*` names remain intentionally active in diagnostics, artifact
  fields, CLI diagnostic profile IDs, and guardrail tests; removing them would
  require a schema/profile migration.
- `models/workload_blind_range_v2.py` still retains `calibration_head` for
  checkpoint compatibility.
- `test_query_driven_rework.py` is still an oversized omnibus test file and
  should eventually split by orchestration subcomponent.
- Some tests still import private helpers from lower-level packages, showing
  those packages do not yet expose clean public testing seams for every behavior.

Decision:
- Active docs, public API names, code naming, and test guardrails are
  checkpointed and verified.
- No scientific success claim is made. No probe or final-grid evidence was
  generated.

## Checkpoint 5.16 — Real-Scale Diagnostic Slice Policy

Status: completed

Goal:
- Clarify when a production-like benchmark slice is useful before the full
  final-grid prerequisites pass.

Changes:
- Added a guide section that allows occasional real-scale exploratory
  diagnostic slices for concrete scaling, instrumentation, runtime, or
  tiny-probe-collapse questions.
- Reaffirmed that these slices are not the full final grid, not acceptance
  evidence, and must keep strict gates unchanged.
- Added a follow-up clarification that pre-gate benchmark snapshots must not
  move the current-best evidence boundary.

Tests:
- `git diff --check -- Range_QDS/docs/query-driven-rework-guide.md Range_QDS/docs/query-driven-rework-progress.md`

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was a protocol documentation
  update.

Key results:
- The guide now distinguishes useful real-scale diagnostic slices from
  forbidden full-grid "see what happens" runs.

Decision:
- Continue using real-scale slices only for named diagnostic questions.
- Do not treat exploratory slice results as final success evidence.

## Checkpoint 5.17 — Global Net-Gain Length Repair

Status: completed

Goal:
- Improve the selector's query-free length repair so the existing bounded
  repair budget targets the highest path-length gains instead of being trapped
  by per-trajectory caps.

Changes:
- Reworked `learned_segment_budget_v1` length repair from independent
  per-trajectory swap caps to a global greedy net-gain budget.
- Kept the existing `learned_segment_length_repair_fraction` semantics as a
  bounded share of learned/fallback slots, but now spends that budget where it
  has the largest query-free path-length benefit.
- Added a focused selector test proving repair budget can move away from a
  zero-gain trajectory and into a high-gain trajectory.

Tests:
- `uv run --group dev -- ruff format Range_QDS/simplification/learned_segment_budget/length_repair.py Range_QDS/tests/unit/simplification/test_learned_segment_budget.py`
- `uv run --group dev -- ruff check Range_QDS`
- `uv run --group dev -- pyright Range_QDS`
- `uv run --group dev -- pytest Range_QDS/tests -q`
- focused selector/orchestration tests:
  `test_learned_segment_budget.py`,
  `test_learned_segment_selector_properties.py`,
  `test_query_driven_rework.py`,
  `test_retained_masks.py`, and `test_evaluation_stage.py`
- `git diff --check -- Range_QDS/simplification/learned_segment_budget/length_repair.py Range_QDS/tests/unit/simplification/test_learned_segment_budget.py`

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was a selector implementation
  checkpoint.

Key results:
- Full pytest passed: `441 passed, 1 warning`.
- Full Ruff passed.
- Full Pyright passed.
- The new unit case verifies that length repair can spend more than the old
  per-trajectory fractional cap on a high-gain trajectory while spending zero
  repair slots on a zero-gain trajectory.

Extra discoveries:
- Artifact audit confirmed the current blocker is a real two-sided selector
  trade-off, not a missing report field: no-repair has strong score causality
  but invalid geometry, current repair improves length but weakens causality,
  and the old length-floor experiment passed global sanity while failing the
  learned-slot materiality floor.
- Prior checkpoint artifacts already covered several rejected selector variants
  (`geometry_gain`, `length_support_blend`, `score_blend`, full repair, and
  no repair). The next scientific probe should test this allocator change
  directly instead of re-running those rejected knobs.

Decision:
- Continue with a strict single-cell probe of the global net-gain repair
  selector before making any final-quality claim.
- Do not run the full 4x7 grid; the current evidence is implementation-level
  plus historical artifact comparison only.

## Checkpoint 5.18 — Global Net-Gain Strict Single-Cell Probe

Status: failed

Goal:
- Test whether the global net-gain length-repair allocator clears the current
  strict single-cell blockers at the same candidate scale as the prior best
  artifact.

Changes:
- No additional code changes after Checkpoint 5.17.
- Generated one strict synthetic/debug single-cell artifact for the global
  net-gain allocator.

Tests:
- Artifact gate extraction with `jq`.
- `git diff --check`
- No full final grid was run.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint05_global_net_gain_repair06_strict_probe_c10_r05`
- command: `uv run --group dev -- python -m orchestration.run_ais_experiment ...`
  with `n_ships=384`, `n_points=256`, `n_queries=48`,
  `range_train_workload_replicates=4`, `compression_ratio=0.05`,
  `query_coverage=0.10`, `query_prior_grid_bins=128`,
  `query_prior_smoothing_passes=0`, and
  `learned_segment_length_repair_fraction=0.6`.

Key results:
- MLQDS QueryUsefulV1: `0.17171327255093727`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- length preservation: `0.7941278359197949`
- gates passed: workload stability, support overlap, predictability,
  prior-predictive alignment, target diffusion, workload signature
- gates failed: learning causality and global sanity
- final claim summary status: `candidate_blocked_by_required_gates`

Gate diagnosis:
- Global sanity failed only on length preservation: `0.7941278359197949`
  versus required `0.80`; SED ratio passed.
- Learning causality failed on shuffled scores, shuffled prior fields, and
  no-query-prior features.
- Shuffled-score delta was `0.006035442454633916` versus required
  `0.017685188752278556`.
- Shuffled-prior and no-query-prior deltas were both
  `0.002756827093483627` versus required `0.005`.
- Untrained, behavior-head, segment-budget-head, prior-only, and learned-slot
  fraction checks passed.

Extra discoveries:
- The allocator change improved MLQDS over Checkpoint 4.74
  (`0.17171327255093727` vs `0.1669032451715525`) and slightly improved length
  (`0.7941278359197949` vs `0.7938149625265364`), but the improvement is too
  small to clear global sanity.
- Same-allocation length-only point selection would reach only
  `0.7585142044823068`, so the length issue is not only point choice inside
  selected segments. Segment allocation needs to change.
- The score-protected length frontier still says the `0.80` length gate is only
  feasible while protecting about `10%` of budget for top learned-score points;
  the `25%` materiality floor has an upper-bound length of
  `0.7911049677462703`.
- Freeze-retained-masks took `202.51s`; global greedy repair is now a runtime
  risk if it stays in the candidate path.

Decision:
- Do not run the full grid.
- Do not call this a success despite beating both baselines.
- Next work should diagnose or change segment allocation / prior-feature
  materiality, not just make length repair greedier.

## Checkpoint 5.19 — Segment Length-Support Allocation

Status: failed

Goal:
- Test whether a small query-free segment length-support component in learned
  segment allocation can improve length preservation without erasing learned
  segment-budget materiality.

Changes:
- Added query-free per-segment path-length support to segment rows.
- Blended normalized learned segment score with normalized segment
  length-support score during segment-budget allocation.
- Reported `segment_length_support_weight`,
  `segment_length_support_score`, and `segment_length_support_rank` in
  selector traces/diagnostics.
- Added focused selector tests for global net-gain length repair,
  length-support allocation, and source-attribution fields.
- Removed a stale formatter-only blank line in `benchmarking/__init__.py` found
  by full format verification.
- Tightened the guide's exploratory benchmark-snapshot rule: snapshots are
  diagnostics only, should prefer representative slices, and cannot override
  failed child gates.

Tests:
- `uv run --group dev -- ruff format` on touched selector/orchestration/test
  files.
- `uv run --group dev -- ruff format --check Range_QDS`
- `uv run --group dev -- ruff check Range_QDS`
- `uv run --group dev -- pyright Range_QDS/simplification/learned_segment_budget Range_QDS/tests/unit/simplification/test_learned_segment_budget.py`
- `uv run --group dev -- pyright Range_QDS`
- `uv run --group dev -- pytest Range_QDS/tests/unit/simplification/test_learned_segment_budget.py Range_QDS/tests/property/test_learned_segment_selector_properties.py -q`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/orchestration/test_retained_masks.py Range_QDS/tests/unit/orchestration/test_evaluation_stage.py -q`
- `uv run --group dev -- pytest Range_QDS/tests -q`
- `uv run --group dev -- yamllint .`
- `git diff --check`
- Artifact gate extraction with `jq`.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint06_segment_length_support_allocation_repair06_strict_probe_c10_r05`
- command: `uv run --group dev -- python -m orchestration.run_ais_experiment ...`
  with `n_ships=384`, `n_points=256`, `synthetic_route_families=4`,
  `seed=2324`, `train_fraction=0.34`, `val_fraction=0.33`,
  `n_queries=48`, `max_queries=256`, `query_coverage=0.10`,
  `range_train_workload_replicates=4`, `compression_ratio=0.05`,
  `query_prior_grid_bins=128`, `query_prior_smoothing_passes=0`,
  `learned_segment_geometry_gain_weight=0.12`,
  `learned_segment_score_blend_weight=0.05`, and
  `learned_segment_length_repair_fraction=0.6`.

Key results:
- MLQDS QueryUsefulV1: `0.17183721530965693`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- length preservation: `0.7941408411227088`
- gates passed: workload stability, support overlap, predictability,
  prior-predictive alignment, target diffusion, workload signature
- gates failed: learning causality and global sanity
- final claim summary status: `candidate_blocked_by_required_gates`
- Full verification passed: Ruff, Pyright, yamllint, whitespace, and pytest
  (`442 passed, 1 warning`).

Gate diagnosis:
- Global sanity failed only on length preservation: `0.7941408411227088`
  versus required `0.80`; SED ratio passed at `0.9173337766436357`.
- Learning causality failed on shuffled scores, shuffled prior fields, and
  no-query-prior features.
- Shuffled-score delta improved to `0.008387859610263837`, but the threshold
  is `0.017759554407510352`.
- Shuffled-prior and no-query-prior deltas were both
  `0.002743017030572781` versus required `0.005`.
- Untrained, behavior-head, segment-budget-head, prior-only, and learned-slot
  fraction checks passed.

Extra discoveries:
- Relative to Checkpoint 5.18, segment length-support allocation changed little:
  MLQDS `+0.00012394275871965843`, length
  `+0.000013005202913918268`, shuffled-score delta
  `+0.002352417155629921`, and shuffled-prior/no-query-prior deltas slightly
  worsened by `-0.000013810062910846188`.
- Same-allocation length-only point selection would reach only
  `0.7597755220341236`, so this allocation still cannot clear the `0.80`
  length gate even if point choice inside selected segments is length-only.
- The score-protected length frontier still only clears `0.80` while protecting
  about `10%` of budget for top learned-score points. At the `25%` learned-slot
  materiality floor, length is still only `0.7911049677462703`.
- The implementation now uses `learned_segment_geometry_gain_weight` for both
  segment allocation length support and within-segment geometry tie-breaking.
  That is acceptable for this diagnostic checkpoint because both are query-free
  geometry pressure, but it is not a clean long-term ablation interface.

Decision:
- Do not run the full grid.
- Do not call this a success despite the best single-cell score so far.
- Continue from Checkpoint 5.19 only if the next checkpoint directly attacks
  learning-causality materiality or the allocation/length trade-off. Repeating
  small query-free geometry nudges is low-value.

## Checkpoint 5.20 — Separate Allocation Length-Support Control

Status: completed

Goal:
- Make the selector diagnostics cleaner by separating query-free segment
  allocation length-support pressure from the within-segment geometry
  tie-breaker.

Changes:
- Added `learned_segment_allocation_length_support_weight` to config, CLI,
  saved config plumbing, MLQDS methods, validation scoring, retained-mask
  freezing, inference, and benchmark reporting.
- Preserved the current candidate behavior by defaulting the new control to
  `0.12`.
- Kept `learned_segment_geometry_gain_weight=0.12` as the within-segment
  geometry tie-breaker only.
- Added `MLQDS_without_segment_length_support_allocation` as a query-free
  ablation method and exposed its delta/mask diagnostics in report rows.
- Updated the guide selector defaults so allocation length support and geometry
  tie-breaking are reported separately.
- Added focused tests for the separated selector control, config/CLI roundtrip,
  benchmark row fields, and regression field set.

Tests:
- `uv run --group dev -- ruff format` on touched config/orchestration/
  selector/reporting/test files.
- `uv run --group dev -- ruff check` on touched config/orchestration/
  selector/reporting/test files.
- `uv run --group dev -- pyright Range_QDS/config Range_QDS/orchestration Range_QDS/simplification Range_QDS/evaluation Range_QDS/training Range_QDS/benchmarking Range_QDS/tests/unit/simplification/test_learned_segment_budget.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py Range_QDS/tests/unit/benchmarking/test_benchmark_runner.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/simplification/test_learned_segment_budget.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py Range_QDS/tests/unit/benchmarking/test_benchmark_runner.py -q`
- `uv run --group dev -- ruff format --check Range_QDS`
- `uv run --group dev -- ruff check Range_QDS`
- `uv run --group dev -- pyright Range_QDS`
- `uv run --group dev -- pytest Range_QDS/tests -q`
- `uv run --group dev -- yamllint .`
- `git diff --check`

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was a selector diagnostics/config
  cleanup checkpoint.

Key results:
- Focused pytest passed: `66 passed`.
- Focused Ruff passed.
- Focused Pyright passed.
- Full pytest passed: `443 passed, 1 warning`.
- Full Ruff, full format check, full Pyright, yamllint, and whitespace checks
  passed.
- The selector can now report and ablate segment allocation length-support
  independently from the geometry tie-breaker.

Extra discoveries:
- Checkpoint 5.19 remains the current best strict scientific artifact; this
  checkpoint did not generate new gate evidence.
- Prior artifacts before Checkpoint 5.20 record the length-support allocation
  through `segment_length_support_weight` in selector traces, but not through
  the new config/report field. Treat old and new artifact schemas carefully
  when comparing reports.

Decision:
- Continue from Checkpoint 5.19/5.20 with no success claim.
- The next scientific checkpoint should use the separated knob if it tests
  allocation length support, and should report
  `MLQDS_without_segment_length_support_allocation`.

## Checkpoint 5.21 — Separated Allocation-Length-Support Strict Replay

Status: failed

Goal:
- Verify that the separated allocation-length-support control preserves the
  current best strict-cell behavior while adding a clean ablation for segment
  allocation length support.

Changes:
- No code changes after Checkpoint 5.20.
- Generated one strict synthetic/debug single-cell artifact with
  `learned_segment_allocation_length_support_weight=0.12` and
  `learned_segment_geometry_gain_weight=0.12`.
- Tightened the guide's pre-gate benchmark snapshot rule: occasional realistic
  benchmark snapshots are scarce diagnostics only, not acceptance evidence,
  threshold input, checkpoint selection input, or a tuning loop.

Tests:
- Artifact gate extraction with `jq`.
- `git diff --check -- Range_QDS/docs/query-driven-rework-guide.md Range_QDS/docs/query-driven-rework-progress.md`

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint07_separated_allocation_length_support_ablation_strict_probe_c10_r05`
- command: `uv run --group dev -- python -m orchestration.run_ais_experiment ...`
  with `n_ships=384`, `n_points=256`, `synthetic_route_families=4`,
  `seed=2324`, `train_fraction=0.34`, `val_fraction=0.33`,
  `n_queries=48`, `max_queries=256`, `query_coverage=0.10`,
  `range_train_workload_replicates=4`, `compression_ratio=0.05`,
  `query_prior_grid_bins=128`, `query_prior_smoothing_passes=0`,
  `learned_segment_allocation_length_support_weight=0.12`,
  `learned_segment_geometry_gain_weight=0.12`,
  `learned_segment_score_blend_weight=0.05`, and
  `learned_segment_length_repair_fraction=0.6`.

Key results:
- MLQDS QueryUsefulV1: `0.17183721530965693`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- length preservation: `0.7941408411227088`
- gates passed: workload stability, support overlap, predictability,
  prior-predictive alignment, target diffusion, workload signature
- gates failed: learning causality and global sanity
- final claim summary status: `candidate_blocked_by_required_gates`

Gate diagnosis:
- Global sanity failed only on length preservation: `0.7941408411227088`
  versus required `0.80`; SED ratio passed at `0.9173337766436357`.
- Learning causality failed on shuffled scores, shuffled prior fields, and
  no-query-prior features.
- Shuffled-score delta was `0.008387859610263837` versus required
  `0.017759554407510352`.
- Shuffled-prior and no-query-prior deltas were both
  `0.002743017030572781` versus required `0.005`.
- `MLQDS_without_segment_length_support_allocation` delta was only
  `0.00012394275871965843`, below the `0.005` material threshold.
- `MLQDS_without_geometry_tie_breaker` delta was `0.009818654458909754`.

Extra discoveries:
- Primary behavior was exactly unchanged from Checkpoint 5.19: same MLQDS,
  uniform, Douglas-Peucker, length, shuffled-score delta, shuffled-prior delta,
  no-query-prior delta, and learned-controlled slot fraction.
- Segment allocation length support is not a material source of the current
  win. It explains only the tiny Checkpoint 5.19 gain over Checkpoint 5.18.
- Same-allocation length-only point selection still cannot clear the length
  gate: it reaches only `0.7597755220341236`.
- The score-protected length frontier still only clears `0.80` while protecting
  about `10%` of budget for top learned-score points. At the `25%`
  learned-slot materiality floor, length is still only `0.7911049677462703`.
- Runtime remains a risk: freeze-retained-masks took about `212.81s`, and the
  full probe took about `380.40s`.

Decision:
- Do not run the full grid.
- Do not call this a success despite the current best strict-cell score.
- Do not keep nudging allocation length support as the next main path. The next
  scientific checkpoint needs a mechanism that makes prior/score causality
  material or changes the length/global trade-off more directly.

## Checkpoint 5.22 — Allocation-Weight Semantics Fix

Status: completed

Goal:
- Fix selector allocation semantics so the separated query-free segment
  length-support control is applied consistently in diagnostic and ablation
  cases.

Changes:
- Fixed `_segment_allocation_weights` so length support is still used when
  learned segment scores are flat. Previously flat segment scores returned
  uniform weights before length support was considered.
- Changed fairness preallocation to choose each trajectory's first learned
  segment by the blended allocation weight, with raw score and start index only
  as tie-breakers.
- Added `segment_allocation_weight` and `segment_allocation_weight_rank` to
  per-segment source-attribution diagnostics.
- Added focused unit coverage for flat-score length-support allocation,
  fairness preallocation using blended weights, and the new attribution fields.

Tests:
- `uv run --group dev -- ruff format Range_QDS/simplification/learned_segment_budget/allocation.py Range_QDS/simplification/learned_segment_budget/diagnostics.py Range_QDS/tests/unit/simplification/test_learned_segment_budget.py`
- `uv run --group dev -- ruff check Range_QDS/simplification/learned_segment_budget/allocation.py Range_QDS/simplification/learned_segment_budget/diagnostics.py Range_QDS/tests/unit/simplification/test_learned_segment_budget.py`
- `uv run --group dev -- pyright Range_QDS/simplification/learned_segment_budget Range_QDS/tests/unit/simplification/test_learned_segment_budget.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/simplification/test_learned_segment_budget.py -q`
- `uv run --group dev -- pytest Range_QDS/tests/property/test_learned_segment_selector_properties.py Range_QDS/tests/unit/orchestration/test_retained_masks.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py -q`

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was a selector implementation
  correctness checkpoint.

Key results:
- Focused Ruff passed.
- Focused Pyright passed: `0 errors, 0 warnings, 0 informations`.
- Focused selector pytest passed: `9 passed`.
- Broader selector/orchestration pytest passed: `100 passed`.

Extra discoveries:
- Checkpoint 5.21 remains the best strict historical artifact, but it predates
  this allocation-weight fix. It should not be treated as exact evidence for
  the current selector implementation.
- This bug matters most for neutral segment-score ablations and support-only or
  flat-score diagnostics. It may also alter fairness preallocation when
  allocation length support disagrees with raw segment score.

Decision:
- Continue with no success claim.
- The next scientific checkpoint should strict-replay the current best single
  cell with this allocation fix before further tuning or benchmark snapshots.

## Checkpoint 5.23 — Allocation-Weight Fix Strict Replay

Status: failed

Goal:
- Replay the current best strict single cell after the Checkpoint 5.22
  allocation-weight semantics fix and reset the current-code evidence boundary.

Changes:
- No code changes after Checkpoint 5.22.
- Generated one strict synthetic/debug single-cell artifact with the fixed
  allocation-weight semantics.

Tests:
- Artifact gate extraction with `jq`.
- No full final grid was run.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint08_allocation_weight_semantics_fix_strict_replay_c10_r05`
- command: `uv run --group dev -- python -m orchestration.run_ais_experiment ...`
  with `n_ships=384`, `n_points=256`, `synthetic_route_families=4`,
  `seed=2324`, `train_fraction=0.34`, `val_fraction=0.33`,
  `n_queries=48`, `max_queries=256`, `query_coverage=0.10`,
  `range_train_workload_replicates=4`, `compression_ratio=0.05`,
  `query_prior_grid_bins=128`, `query_prior_smoothing_passes=0`,
  `learned_segment_allocation_length_support_weight=0.12`,
  `learned_segment_geometry_gain_weight=0.12`,
  `learned_segment_score_blend_weight=0.05`, and
  `learned_segment_length_repair_fraction=0.6`.

Key results:
- MLQDS QueryUsefulV1: `0.17183721530965693`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- length preservation: `0.7941408411227088`
- gates passed: workload stability, support overlap, predictability,
  prior-predictive alignment, target diffusion, workload signature
- gates failed: learning causality and global sanity
- final claim summary status: `candidate_blocked_by_required_gates`

Gate diagnosis:
- Global sanity failed only on length preservation: `0.7941408411227088`
  versus required `0.80`; SED ratio passed at `0.9173337766436357`.
- Learning causality failed on shuffled scores, shuffled prior fields, and
  no-query-prior features.
- Shuffled-score delta was `0.008856345116771192` versus required
  `0.017759554407510352`.
- Shuffled-prior and no-query-prior deltas were both
  `0.002743017030572781` versus required `0.005`.
- `MLQDS_without_segment_length_support_allocation` delta stayed
  `0.00012394275871965843`, below the `0.005` material threshold.
- `MLQDS_without_geometry_tie_breaker` delta stayed `0.009818654458909754`.

Extra discoveries:
- Primary behavior was unchanged from Checkpoint 5.21: same MLQDS, uniform,
  Douglas-Peucker, length, shuffled-prior/no-query-prior deltas, allocation
  length-support delta, and learned-controlled slot fraction.
- The allocation-weight fix slightly improved some causality diagnostics:
  shuffled-score delta `+0.00046848550650735454`, segment-budget-head delta
  `+0.0028946836748793836`, and prior-only delta
  `+0.0007186064333346565`.
- These improvements are not enough to clear learning causality. The
  shuffled-score delta is still only about half of the required threshold, and
  query-prior materiality is unchanged.
- Same-allocation length-only point selection still cannot clear the length
  gate: it reaches only `0.7597755220341236`.
- The score-protected length frontier is unchanged: length clears `0.80` only
  while protecting about `10%` of budget for top learned-score points; at the
  `25%` learned-slot materiality floor, length is `0.7911049677462703`.
- Runtime remains a risk: freeze-retained-masks took `216.06s`, and the full
  probe took `391.47s`.

Decision:
- Do not run the full grid.
- Do not call this a success despite preserving the current best strict-cell
  score.
- The next scientific checkpoint should not keep tuning allocation length
  support. It should target query-prior materiality or a segment-allocation
  mechanism that changes the learned-control-vs-length trade-off.

## Checkpoint 5.24 — Pre-Gate Benchmark Snapshot Policy

Status: completed

Goal:
- Add the current-best benchmark snapshot note to the guide without weakening
  the evidence protocol.

Changes:
- Clarified that occasional real-scale benchmark snapshots may be useful for
  the current best candidate/config when small probes might hide runtime,
  workload-count, or scale-sensitive quality failures.
- Kept snapshots diagnostic-only: they may inform prioritization and capacity
  planning, but not final evidence, threshold changes, checkpoint selection,
  selector tuning, or final comparison tables without separate strict
  single-cell diagnosis.

Tests:
- Documentation-only change; no code tests run.

Experiment artifact:
- path: not generated
- command: no scientific probe was run.

Key results:
- MLQDS QueryUsefulV1: not applicable
- uniform QueryUsefulV1: not applicable
- Douglas-Peucker QueryUsefulV1: not applicable
- gates passed: not applicable
- gates failed: not applicable

Extra discoveries:
- The guide already had a pre-gate snapshot section. The missing part was
  explicit current-best candidate/config wording and a stronger warning against
  feeding snapshot results back into tuning or final reporting.

Decision:
- Continue with query-prior materiality diagnosis before further scientific
  probes.

## Checkpoint 5.25 — Model-Prior Materiality Diagnostics

Status: completed

Goal:
- Make prior-field causality diagnostics distinguish raw sampled prior changes
  from the actual v2 model-input and normalized prior-channel changes.

Changes:
- Added `model_prior_feature_sensitivity` diagnostics for query-prior
  ablations. It reports:
  - raw model-input prior-channel deltas after the feature builder has disabled
    v2-excluded fields such as `route_density_prior`
  - normalized model-input prior-channel deltas after the persisted scaler
  - scaler prior-channel ranges and disabled prior-field names
- Attached the new diagnostics to final retained-mask prior ablations,
  checkpoint-selection prior ablations, and per-prior-channel ablations.
- Tightened `prior_sample_gate_failures` so future reports can flag cases where
  shuffled raw prior fields change but model inputs or normalized model inputs
  do not.
- Added benchmark-row fields for model-input and normalized prior deltas so
  report tables do not hide the materiality path behind nested JSON.

Tests:
- `uv run --group dev -- ruff format Range_QDS/orchestration/causality.py Range_QDS/orchestration/retained_mask_ablations.py Range_QDS/orchestration/selection_causality.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- ruff check Range_QDS/orchestration/causality.py Range_QDS/orchestration/retained_mask_ablations.py Range_QDS/orchestration/selection_causality.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pyright Range_QDS/orchestration/causality.py Range_QDS/orchestration/retained_mask_ablations.py Range_QDS/orchestration/selection_causality.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_rework.py -q`
- `uv run --group dev -- ruff check Range_QDS/benchmarking/reporting/row_fields.py Range_QDS/tests/unit/benchmarking/test_benchmark_runner.py Range_QDS/tests/regression/test_benchmark_report_regression.py`
- `uv run --group dev -- pyright Range_QDS/benchmarking/reporting/row_fields.py Range_QDS/tests/unit/benchmarking/test_benchmark_runner.py Range_QDS/tests/regression/test_benchmark_report_regression.py`
- `uv run --group dev -- pytest Range_QDS/tests/regression/test_benchmark_report_regression.py::test_benchmark_row_field_set_regression Range_QDS/tests/unit/benchmarking/test_benchmark_runner.py -q`

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was a diagnostics/reporting
  checkpoint.

Key results:
- Focused orchestration pytest passed: `96 passed`.
- Focused benchmark/reporting pytest passed: `38 passed`.
- Focused Ruff and Pyright passed.
- MLQDS QueryUsefulV1: not applicable
- uniform QueryUsefulV1: not applicable
- Douglas-Peucker QueryUsefulV1: not applicable
- gates passed: not applicable
- gates failed: not applicable

Extra discoveries:
- The previous prior-sensitivity diagnostics could prove raw prior fields were
  sampled and changed, but not whether those changes survived the v2 feature
  builder and scaler. That made the Checkpoint 5.23 query-prior materiality
  failure underdiagnosed.
- `route_density_prior` can be nonzero in raw sampled priors while correctly
  contributing zero model-input delta because the current v2 candidate disables
  that channel. Future reports now expose that distinction.

Decision:
- Continue with no success claim.
- The next scientific checkpoint should strict-replay the current best single
  cell only if the new diagnostics are needed in an artifact; otherwise continue
  directly to a narrow prior/materiality or segment-allocation mechanism.

## Checkpoint 5.26 — Model-Prior Materiality Strict Replay

Status: failed

Goal:
- Replay the current best strict single cell after Checkpoint 5.25 so the new
  model-prior materiality diagnostics are present in a scientific artifact.

Changes:
- No code changes after Checkpoint 5.25.
- Generated one strict synthetic/debug single-cell artifact with the same
  current-candidate config as Checkpoint 5.23.

Tests:
- Artifact gate and diagnostic extraction with `jq`.
- No full final grid was run.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint09_model_prior_materiality_strict_replay_c10_r05`
- command: `uv run --group dev -- python -m orchestration.run_ais_experiment ...`
  with `n_ships=384`, `n_points=256`, `synthetic_route_families=4`,
  `seed=2324`, `train_fraction=0.34`, `val_fraction=0.33`,
  `n_queries=48`, `max_queries=256`, `query_coverage=0.10`,
  `range_train_workload_replicates=4`, `compression_ratio=0.05`,
  `query_prior_grid_bins=128`, `query_prior_smoothing_passes=0`,
  `learned_segment_allocation_length_support_weight=0.12`,
  `learned_segment_geometry_gain_weight=0.12`,
  `learned_segment_score_blend_weight=0.05`, and
  `learned_segment_length_repair_fraction=0.6`.

Key results:
- MLQDS QueryUsefulV1: `0.17183721530965693`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- length preservation: `0.7941408411227088`
- gates passed: workload stability, support overlap, predictability,
  prior-predictive alignment, target diffusion, workload signature
- gates failed: learning causality and global sanity
- final claim summary status: `candidate_blocked_by_required_gates`

Gate diagnosis:
- Learning causality failed on shuffled scores, shuffled prior fields, and
  no-query-prior features.
- Shuffled-score delta was `0.008856345116771192` versus required
  `0.017759554407510352`.
- Shuffled-prior and no-query-prior deltas were both
  `0.002743017030572781` versus required `0.005`.
- Global sanity failed only on length preservation: `0.7941408411227088`
  versus required `0.80`; SED ratio passed at `0.9173337766436357`.

Prior-materiality diagnosis:
- Raw no-prior ablation changed sampled prior fields strongly:
  mean absolute feature delta `0.13527387380599976`.
- After v2 feature building, the active model-input prior delta was much
  smaller but still real: no-prior model-input and normalized deltas were both
  `0.012836813926696777`.
- Shuffled-prior model-input and normalized deltas were both
  `0.012831311672925949`.
- Selector-score movement remained tiny despite real model-input movement:
  no-prior selector-score mean absolute delta `0.0005342491785995662`, raw
  prediction mean absolute delta `0.00037622981471940875`, retained-mask
  Jaccard `0.9785969084423306`, and top-k Jaccard
  `0.9916217833632556`.
- Channel ablations show the useful model-input prior materiality is mostly
  `behavior_utility_prior`: channel delta `0.004247008290286597`, model-input
  mean absolute delta `0.008451469242572784`, and retained-mask difference
  `32` points.
- Spatial query hit contributes weakly: channel delta
  `0.0009345698981791939` and retained-mask difference `4` points.
- Spatiotemporal query hit, boundary, crossing, and route-density channels are
  effectively non-material in the current trained model. `route_density_prior`
  is correctly zero at the model-input level because v2 disables it.

Extra discoveries:
- The prior materiality blocker is not caused by raw prior sampling failure,
  v2 feature-builder dropout of active channels, or scaler normalization. The
  active prior signals reach the model input.
- The model/selector suppresses active prior input changes before they become
  retained-mask decisions. This is now the sharper blocker than generic prior
  support.
- `behavior_utility_prior` is the only near-material prior channel in the
  current trained model. Future prior-materiality work should start there rather
  than treating all prior channels as equally weak.
- Runtime remains high but stable: selection-causality diagnostics took
  `56.68s`, freeze-retained-masks took `209.88s`, and the full probe took
  `378.64s`.

Decision:
- Do not run the full grid.
- Do not call this a success; primary behavior is unchanged from Checkpoint
  5.23 and required gates still fail.
- Next checkpoint should target model/selector mechanisms that turn
  `behavior_utility_prior` and score perturbations into material retained-mask
  changes without weakening the length gate.

## Checkpoint 5.27 — Behavior-Head Rank Loss

Status: completed

Goal:
- Add a narrow training-only pressure that makes the
  `conditional_behavior_utility` head preserve useful ordering among high-value
  behavior points, because Checkpoint 5.26 showed `behavior_utility_prior` is
  the only near-material prior channel but retained-mask movement is still too
  weak.

Changes:
- Added `query_useful_behavior_rank_loss_weight` to model config, CLI parsing,
  run config plumbing, command logging, config round-trip coverage, and training
  target diagnostics.
- Added `_behavior_head_rank_loss`, a listwise auxiliary loss over valid
  `conditional_behavior_utility` targets. It compares top behavior targets
  against lower targets with a minimum target gap and penalizes reversed logits.
- Included the behavior-rank term in `_factorized_query_useful_loss` with a
  default weight of `0.15`.
- Updated the guide's active QueryUsefulV1 head list to include
  `path_length_support_target` and documented the behavior-rank default as
  diagnostic until strict replay evidence exists.

Tests:
- `uv run --group dev -- ruff format Range_QDS/training/training_epoch.py Range_QDS/config/experiment_config.py Range_QDS/training/train_model.py Range_QDS/orchestration/experiment_cli.py Range_QDS/orchestration/run_ais_experiment.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py`
- `uv run --group dev -- ruff check Range_QDS/training/training_epoch.py Range_QDS/config/experiment_config.py Range_QDS/training/train_model.py Range_QDS/orchestration/experiment_cli.py Range_QDS/orchestration/run_ais_experiment.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py`
- `uv run --group dev -- pyright Range_QDS/training/training_epoch.py Range_QDS/config/experiment_config.py Range_QDS/training/train_model.py Range_QDS/orchestration/experiment_cli.py Range_QDS/orchestration/run_ais_experiment.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py -q`
- `uv run --group dev -- pytest Range_QDS/tests/unit/training/test_training_does_not_collapse.py -q`

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was a training-loss implementation
  checkpoint.

Key results:
- Focused Ruff passed.
- Focused Pyright passed: `0 errors, 0 warnings, 0 informations`.
- Focused orchestration/runtime pytest passed: `119 passed`.
- Adjacent training pytest passed: `39 passed, 1 warning`.
- MLQDS QueryUsefulV1: not applicable
- uniform QueryUsefulV1: not applicable
- Douglas-Peucker QueryUsefulV1: not applicable
- gates passed: not applicable
- gates failed: not applicable

Extra discoveries:
- The active guide was stale: it listed five QueryUsefulV1 heads while the
  current schema has six, including `path_length_support_target`.
- The implementation initially had an incomplete config path: the CLI/run path
  referenced `query_useful_behavior_rank_loss_weight`, but
  `build_experiment_config` did not accept or store it. That would have broken
  command-line use of the new control.
- This change does not prove better learning. It only creates an auditable
  mechanism for the next strict replay to test whether behavior-prior signal can
  become material in retained masks.

Decision:
- Do not run the full grid.
- Do not call this a success; no retained-mask artifact exists for this change.
- Next checkpoint should run a strict single-cell replay of the current best
  config with the behavior-rank loss enabled and diagnose learning causality,
  behavior-head fit, prior materiality, and global sanity before any further
  tuning.

## Checkpoint 5.28 — Behavior-Rank Strict Replay

Status: failed

Goal:
- Test whether `query_useful_behavior_rank_loss_weight=0.15` turns the
  near-material `behavior_utility_prior` channel into material retained-mask
  causality while preserving the current strict-cell win and length behavior.

Changes:
- Generated one strict synthetic/debug single-cell artifact with the same
  current-candidate config as Checkpoint 5.26, except behavior-rank loss was
  explicitly enabled at `0.15`.
- No full final grid was run.

Tests:
- Artifact gate and diagnostic extraction with `jq`.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint10_behavior_rank_loss_strict_replay_c10_r05`
- command: `uv run --group dev -- python -m orchestration.run_ais_experiment ...`
  with `n_ships=384`, `n_points=256`, `synthetic_route_families=4`,
  `seed=2324`, `train_fraction=0.34`, `val_fraction=0.33`,
  `n_queries=48`, `max_queries=256`, `query_coverage=0.10`,
  `range_train_workload_replicates=4`, `compression_ratio=0.05`,
  `query_prior_grid_bins=128`, `query_prior_smoothing_passes=0`,
  `query_useful_behavior_rank_loss_weight=0.15`,
  `learned_segment_allocation_length_support_weight=0.12`,
  `learned_segment_geometry_gain_weight=0.12`,
  `learned_segment_score_blend_weight=0.05`, and
  `learned_segment_length_repair_fraction=0.6`.

Key results:
- MLQDS QueryUsefulV1: `0.1662931067947708`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- length preservation: `0.7939681743351743`
- gates passed: workload stability, support overlap, predictability,
  prior-predictive alignment, target diffusion, workload signature
- gates failed: learning causality and global sanity
- final claim summary status: `candidate_blocked_by_required_gates`

Gate diagnosis:
- Learning causality failed on shuffled scores, shuffled prior fields,
  no-query-prior features, and no-behavior-head.
- Shuffled-score delta collapsed to `0.00005168542757363892` versus required
  `0.014433089298578677`.
- Shuffled-prior and no-query-prior deltas were both
  `0.0011307408759458626` versus required `0.005`.
- No-behavior-head delta was `0.004538679516121635`, just below the `0.005`
  material threshold.
- Global sanity failed only on length preservation: `0.7939681743351743`
  versus required `0.80`; SED ratio passed at `0.9203558450631464`.

Behavior and prior diagnosis:
- Behavior-head train fit improved only weakly versus Checkpoint 5.26: Kendall
  tau moved from `-0.01658122797936113` to `-0.0030223587756942243`, and
  top-5 mass recall moved from `0.2041450153552849` to
  `0.21870494090181075`.
- The behavior head stayed essentially unlearned: prediction std was only
  `0.019670329988002777` against target std `0.28323090076446533`.
- Active model-input prior deltas were still present:
  no-prior model-input delta `0.012836813926696777` and behavior-channel
  model-input delta `0.008451469242572784`.
- Selector-score movement stayed tiny: no-prior selector-score mean abs delta
  `0.0005560250720009208`; behavior-channel selector-score mean abs delta
  `0.0005180706502869725`.
- `behavior_utility_prior` was again the only meaningfully active prior
  channel, but its QueryUsefulV1 delta was only `0.0011291917375584049`.

Extra discoveries:
- Behavior-rank pressure did not solve prior materiality. It slightly improved
  a head-fit diagnostic while making the actual retained-mask causality much
  worse.
- The trained score became almost indistinguishable from shuffled scores under
  the required delta gate, so this is not a candidate even though MLQDS still
  barely beats Douglas-Peucker.
- The score-protected length frontier stayed unfavorable: protecting `25%` of
  budget for top learned-score points gives length only
  `0.7909692518906397`, below the `0.80` gate.
- Runtime remained high but stable: selection-causality diagnostics took about
  `57.97s`, freeze-retained-masks took about `215.07s`, and the full probe took
  `387.04s`.

Decision:
- Reject behavior-rank loss weight `0.15` as a default.
- Do not run the full grid.
- Do not call this a success; it regresses the current best strict candidate.
- Keep the behavior-rank mechanism only as an explicit diagnostic control unless
  a future checkpoint has a stronger mechanism-level hypothesis.

## Checkpoint 5.29 — Disable Rejected Behavior-Rank Default

Status: completed

Goal:
- Keep the codebase clean after the negative strict replay by preventing the
  rejected behavior-rank setting from becoming the default current candidate.

Changes:
- Changed `query_useful_behavior_rank_loss_weight` defaults from `0.15` to
  `0.0` in `ModelConfig`, `build_experiment_config`, CLI parsing, training
  fallback defaults, and training target diagnostics.
- Kept explicit CLI/config support for behavior-rank loss so Checkpoint 5.28
  remains reproducible and future diagnostics can opt in deliberately.
- Updated the guide to mark behavior-rank weight `0.15` as rejected by strict
  replay and to recommend default `0.0`.
- Added runtime/config assertions that direct config and CLI defaults keep
  behavior-rank disabled.

Tests:
- `uv run --group dev -- ruff format Range_QDS/training/training_epoch.py Range_QDS/config/experiment_config.py Range_QDS/training/train_model.py Range_QDS/orchestration/experiment_cli.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py`
- `uv run --group dev -- ruff check Range_QDS/training/training_epoch.py Range_QDS/config/experiment_config.py Range_QDS/training/train_model.py Range_QDS/orchestration/experiment_cli.py Range_QDS/orchestration/run_ais_experiment.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py`
- `uv run --group dev -- pyright Range_QDS/training/training_epoch.py Range_QDS/config/experiment_config.py Range_QDS/training/train_model.py Range_QDS/orchestration/experiment_cli.py Range_QDS/orchestration/run_ais_experiment.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py Range_QDS/tests/unit/training/test_training_does_not_collapse.py -q`
- `git diff --check`

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was cleanup after a failed
  scientific replay.

Key results:
- Focused Ruff passed.
- Focused Pyright passed: `0 errors, 0 warnings, 0 informations`.
- Focused orchestration/runtime/training pytest passed: `158 passed, 1 warning`.
- MLQDS QueryUsefulV1: not applicable
- uniform QueryUsefulV1: not applicable
- Douglas-Peucker QueryUsefulV1: not applicable
- gates passed: not applicable
- gates failed: not applicable

Extra discoveries:
- Leaving a failed training loss enabled by default would have silently moved
  the current code candidate away from the best strict artifact. That would make
  later comparisons misleading.

Decision:
- Treat Checkpoint 5.26 as the current-best strict evidence boundary.
- Treat Checkpoint 5.28 as a rejected-path artifact.
- Next scientific checkpoint should not tune behavior-rank weight. It should
  target a selector/model mechanism that materially changes prior or score
  perturbations in retained masks without sacrificing the current baseline win
  or length behavior.

## Checkpoint 5.30 — Configurable Segment Allocation Weight Floor

Status: completed

Goal:
- Expose the learned-segment allocation weight floor as an explicit selector
  control because the hard-coded `0.50` floor can make segment allocation nearly
  uniform and suppress learned-score materiality diagnostics.

Changes:
- Added `learned_segment_allocation_weight_floor` to model config, CLI parsing,
  run-command logging, MLQDS methods, validation scoring, inference replay,
  retained-mask trace recomputation, retained-mask ablations, selection
  causality, final summary, and benchmark-row reporting.
- Kept the default at `0.50`, preserving current selector behavior unless a
  future checkpoint explicitly opts into a lower floor.
- Added `segment_allocation_weight_floor` to learned-segment selector traces so
  artifacts show the exact allocation contrast setting used.
- Added focused selector coverage proving a lower floor concentrates allocation
  on the higher-scored segment, and config/CLI/reporting coverage for the new
  field.
- Updated the guide selector defaults and marked lower-floor runs as diagnostic
  until strict replay proves learning causality and global sanity.

Tests:
- `uv run --group dev -- ruff format` on touched selector/config/orchestration/
  reporting/test paths
- `uv run --group dev -- ruff check` on touched selector/config/orchestration/
  reporting/test paths
- `uv run --group dev -- pyright` on touched selector/config/orchestration/
  reporting/test paths
- `uv run --group dev -- pytest Range_QDS/tests/unit/simplification/test_learned_segment_budget.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py Range_QDS/tests/unit/benchmarking/test_benchmark_runner.py Range_QDS/tests/regression/test_benchmark_report_regression.py::test_benchmark_row_field_set_regression -q`
- `git diff --check` on touched paths

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was default-preserving selector
  plumbing and diagnostic visibility.

Key results:
- Focused Ruff passed.
- Focused Pyright passed: `0 errors, 0 warnings, 0 informations`.
- Focused pytest passed: `70 passed`.
- MLQDS QueryUsefulV1: not applicable
- uniform QueryUsefulV1: not applicable
- Douglas-Peucker QueryUsefulV1: not applicable
- gates passed: not applicable
- gates failed: not applicable

Extra discoveries:
- The previous hard-coded floor was invisible in run config, selector traces,
  and benchmark rows. That made future lower-floor diagnostics impossible to
  distinguish from default behavior by reading artifacts alone.
- A lower floor is only a diagnostic for allocation contrast. It must not be
  treated as learned success unless a strict replay clears learning causality
  and global sanity with unchanged gates.

Decision:
- Continue from the Checkpoint 5.26 strict evidence boundary.
- Next scientific checkpoint may test a lower allocation floor as a strict
  single-cell diagnostic, but only after stating the expected gate movement and
  keeping all current gates unchanged.

## Checkpoint 5.31 — Lower Allocation-Floor Strict Replay

Status: failed

Goal:
- Test whether reducing `learned_segment_allocation_weight_floor` from `0.50`
  to `0.10` makes learned segment scores materially affect retained masks while
  preserving workload health and global sanity.

Changes:
- Generated one corrected strict synthetic/debug single-cell artifact using the
  current-best strict cell settings except for allocation floor `0.10`.
- Tightened the guide's pre-gate benchmark-snapshot rule: representative slices
  are the default, and pre-gate full-grid snapshots are exceptional
  observational diagnostics only.
- Removed one invalid point-value replay artifact generated before explicitly
  carrying `--range_training_target_mode query_useful_v1_factorized`.

Tests:
- Artifact sanity with `jq`: confirmed QueryUsefulV1 primary metric,
  factorized target mode, and allocation floor `0.10`.
- Gate/component extraction with `jq`.
- No full final grid was run.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint11_allocation_floor010_query_useful_strict_replay_c10_r05`
- command: current-best strict replay command with
  `--learned_segment_allocation_weight_floor 0.10` and
  `--range_training_target_mode query_useful_v1_factorized`

Key results:
- MLQDS QueryUsefulV1: `0.15366824272250135`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- length preservation: `0.7833962145166923`
- gates passed: workload stability, support overlap, predictability,
  prior-predictive alignment, target diffusion, workload signature
- gates failed: learning causality, global sanity
- final status: `candidate_blocked_by_required_gates`

Gate diagnosis:
- Lowering the floor increased retained-mask movement, but the movement was not
  useful. Shuffled-score symdiff was `1906`, no-segment-budget-head symdiff was
  `1380`, and prior-only symdiff was `1696`.
- Learning causality failed by sign: shuffled-score delta
  `-0.008434220434683226` versus required `0.006858170855217005`; untrained
  delta `-0.0013016596544116188`; shuffled-prior and no-query-prior deltas both
  `-0.00025329797716061586`; no-behavior-head delta
  `-0.005274399668764362`; no-segment-budget-head delta
  `-0.0036255370451678814`; prior-only delta `-0.005605166103091003`.
- Global sanity failed on length. Endpoint sanity and SED ratio passed.
- Allocation became less uniform: segments with learned budget dropped to
  `768`, segment-budget entropy normalized was `0.9731191007713681`, and the
  learned-controlled slot fraction stayed `0.33834134615384615`.

Extra discoveries:
- The first replay command was invalid because it omitted
  `--range_training_target_mode query_useful_v1_factorized`; its artifact was
  removed. Replays must explicitly carry target mode.
- Lowering the allocation floor is not a solution by itself. It gives the score
  more authority, but the current score/segment-value path sends budget into
  worse segments.
- Same-allocation length-only point selection fell to
  `0.6968862694377511`, so the lower-floor allocations were length-hostile
  before within-segment point choice could recover them.
- Prior perturbations still barely matter at selector level: model-input prior
  delta is real at about `0.0128368`, but selector-score delta is about
  `0.000534` and retained-mask Jaccard stays `0.9785969084423306`.

Decision:
- Reject `learned_segment_allocation_weight_floor=0.10` as a standalone
  candidate.
- Keep the default allocation floor at `0.50`.
- Do not run the full grid.
- Next scientific work should not lower allocation floor further unless paired
  with a mechanism that makes segment value length-compatible and query-useful.
