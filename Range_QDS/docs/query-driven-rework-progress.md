# Query-Driven Rework Progress

This is the short checkpoint log required by `docs/query-driven-rework-guide.md`.
Detailed stdout and raw metrics are kept in `Range_QDS/artifacts/results/`.

## High-Value Summary

The redesign is active and not complete. The current best strict synthetic/debug cell beats both final baselines on `QueryUsefulV1`, and its stored global-sanity gate reclassifies as passing under the current `0.75` length policy. It is still not acceptance evidence because learning causality fails and the final matrix remains unrun.

Current best strict result:

```text
MLQDS QueryUsefulV1:           0.17183721530965693
uniform QueryUsefulV1:         0.14223795796380634
Douglas-Peucker QueryUsefulV1: 0.16362459837911367
length preservation:           0.7941408411227088
```

Current best strict artifact:
- `artifacts/results/query_driven_v2_checkpoint13_per_head_prior_materiality_strict_replay_c10_r05`

Current best gate reclassification artifact:
- `artifacts/results/query_driven_v2_checkpoint18_current_best_gate_reclassification_len075/gate_reclassification_summary.json`

Current best learning-causality diagnosis artifact:
- `artifacts/results/query_driven_v2_checkpoint19_learning_causality_failure_diagnosis_current_best/learning_causality_failure_diagnosis.json`

Main interpretation:
- The candidate is promising because it beats uniform and Douglas-Peucker in one strict cell while workload stability, support overlap, predictability, prior-predictive alignment, target diffusion, workload signature, and recomputed global sanity pass.
- It is not final success. Learning causality still fails, and the global-sanity update is a policy reclassification of a stored strict artifact, not a replay.
- The full workload-profile/compression final matrix remains intentionally unrun until strict single-cell gates pass.
- Small probes and implementation smokes are not scientific evidence of learning.

Current research question:

```text
Can the selector/model make train-derived prior, behavior, and score perturbations materially affect frozen retained masks while preserving length and the MLQDS win over uniform and Douglas-Peucker?
```

If a checkpoint does not answer that question more clearly, it is probably low-value.

## Current State - 2026-05-18

Active candidate defaults:
- method: `workload_blind_range_v2`
- `route_density_prior` excluded from v2 model inputs, but retained for support diagnostics
- hidden prior residual scale `0.25`
- no direct prior-to-head residual
- `learned_segment_score_blend_weight=0.05`
- `learned_segment_length_repair_fraction=0.6`
- length repair uses global net-gain allocation
- query-free segment length-support allocation uses `learned_segment_allocation_length_support_weight=0.12`
- within-segment geometry tie-breaking uses `learned_segment_geometry_gain_weight=0.12`
- behavior-rank, sparse-head rank, sparse-head BCE calibration, and score-protected repair controls remain default-off diagnostics

Current blockers:
- Learning causality fails. In the best strict artifact, shuffled-score delta is `0.008856345116771192` versus required `0.017759554407510352`; shuffled-prior and no-query-prior deltas are both `0.002743017030572781` versus required `0.005`.
- A focused diagnosis ranks the child-gate shortfalls as: shuffled scores
  `0.00890320929073916`, shuffled priors `0.002256982969427219`,
  no-query-prior `0.002256982969427219`.
- Recomputed global sanity under the current `0.75` length gate passes with no failed checks: length `0.7941408411227088`, endpoint sanity `1.0`, SED ratio `0.9173337766436357` versus max `1.5`.
- Per-head prior-output diagnostics show prior signal is mostly suppressed before retained-mask decisions: zeroing active model-input priors changes inputs by about `0.0128368`, but mean head probability changes only about `0.00001816`.
- No-length-repair improves score to `0.1759846099523811`, but length collapses to `0.6790996203798462`. That is diagnostic evidence, not a candidate.
- Under the old `0.80` length policy, segment allocation was part of the length blocker. Under the new `0.75` final gate, the same diagnostic is not a current blocker by itself; causality remains unresolved.

Current decision:
- Do not run the full final matrix.
- Do not treat real-scale diagnostic slices as success evidence.
- Do not increase workload/caps to compensate for failed causality unless the named diagnostic question requires scale.
- Do not convert the new `0.75` final and validation length thresholds into a success claim; learning causality still fails.
- Keep the current candidate boundary at the best strict artifact plus its `0.75` gate reclassification until a new strict candidate clears or narrows learning causality.

## Durable Discoveries Since The Current Candidate

- Global net-gain length repair improved the current strict score boundary. After the `0.75` policy reclassification, length is no longer the current strict-cell blocker; learning causality remains the blocker.
- Query-free segment length-support allocation at weight `0.12` had only tiny effect: MLQDS `+0.00012394275871965843`, length `+0.000013005202913918268`, shuffled-score causality delta `+0.002352417155629921` versus the prior strict cell.
- Allocation length-support was initially ignored when learned segment scores were flat; that implementation flaw was fixed, but the strict replay still did not change the blocker.
- Raw prior channels and model inputs are available. The problem is that useful movement is suppressed inside heads/selector/allocation before frozen retained masks.
- The current-best causality diagnosis narrows that statement: score shuffling
  moves masks substantially but does not lose enough quality, while prior
  ablations move only `36` retained decisions with retained-mask Jaccard about
  `0.9786`. The next blocker is score/prior materiality, not basic learned-slot
  availability.
- Behavior-rank loss `0.15`, allocation floor `0.10`, score-protected repair `0.10`, sparse-head rank `0.10`, and sparse-head BCE `window_max_normalized` are rejected default paths.
- Exact-pair length repair is still rejected as a default. It raised length to `0.7990875085863033`, but regressed MLQDS to `0.16997958695311988` and hurt learned-head causality.
- Under the old `0.80` length policy, the score-protected length frontier cleared length only near `10%` protected learned-score budget. Under the new `0.75` final length gate, this frontier is less useful as a blocker diagnosis; causality remains the blocker.
- Bounded exact-pair repair reduced diagnostic runtime from about `4502.94s` to `819.96s`, but MLQDS latency was still `15148ms`; it needs a runtime plan before future consideration.
- `max_budget_share_per_trajectory` is effectively softened by fair-share allocation when fair-share cap is larger; treat it as a soft trajectory-share limit.

## Rejected-Path Memory

| Path | Best observed effect | Rejection reason |
|---|---:|---|
| no length repair | MLQDS `0.1759846099523811`; learned-controlled slot fraction `0.8461538461538461` | length collapsed to `0.6790996203798462`; learning causality still failed |
| full length repair | length `0.7980194800294772` | learned-controlled slot fraction collapsed to `0.203125`; MLQDS lost to Douglas-Peucker |
| segment length-support allocation `0.12` | best strict score: MLQDS `0.17183721530965693` | still fails learning causality; historical artifact failed global sanity under the old `0.80` length gate |
| behavior-rank loss `0.15` | behavior-head fit improved slightly | MLQDS regressed to `0.1662931067947708`; shuffled-score delta collapsed to `0.00005168542757363892` |
| allocation floor `0.10` | allocation moved more visibly | MLQDS regressed to `0.15366824272250135`; length worsened to `0.7833962145166923`; causality failed by sign |
| score-protected repair `0.10` | learned-controlled slots rose to `0.3984375` | MLQDS regressed to `0.1621987738648618`; length worsened to `0.7885179226003864`; causality still failed |
| exact-pair length repair | length `0.7990875085863033` | regressed score and harmed behavior/segment-budget causality; old replay missed only the old `0.80` length gate |
| sparse-head rank `0.10` | MLQDS rose by only `0.00030555491606800877` | shuffled-score and prior/no-prior causality worsened |
| sparse-head BCE `window_max_normalized` | head dispersion increased | MLQDS regressed to `0.1548579044007669` and lost to Douglas-Peucker |
| prior residual scale `1.0` after route removal | length `0.7939141083394758` | MLQDS lost to Douglas-Peucker; shuffled-score causality failed by sign |
| semantic prior-to-head residual | training fit improved | retained-mask result worsened and prior ablations became harmful |
| point-score blend `0.15` | length `0.7943720026689473` | MLQDS lost to Douglas-Peucker; shuffled and untrained causality failed by sign |

## Next-Checkpoint Guardrails

- Preserve comparability with the current best strict artifact unless a checkpoint explicitly resets the candidate boundary.
- Diagnose by failed gate and component before changing code.
- For causality work, require material retained-mask movement, not just better train fit or larger head dispersion.
- For length work, preserve learned-controlled slots. Query-free repair that crowds out learned selection is not a solution.
- Do not keep rescaling sparse heads, lowering allocation floors, or protecting repair budget without a mechanism that explains how it fixes retained-mask causality and length together.
- Do not add temporal scaffolding or threshold changes to manufacture a success claim.
- Report learned-controlled slot fraction, shuffled-score delta, no-prior/no-query-prior delta, no-behavior-head delta, segment-budget-head delta, and length for any new strict candidate.

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
- gates failed: learning causality, global sanity under the old `0.80` length gate
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
- gates failed: learning causality, global sanity under the old `0.80` length gate
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
- `../.venv/bin/ruff check scoring/baselines.py experiments/benchmark_report.py experiments/experiment_cli.py experiments/experiment_config.py experiments/experiment_data.py experiments/experiment_methods.py experiments/experiment_pipeline.py experiments/range_diagnostics.py experiments/run_ais_experiment.py experiments/run_inference.py models/workload_blind_range_v2.py workloads/query_generator.py workloads/workload_profiles.py selection/learned_segment_budget.py selection/mlqds_scoring.py tests/test_benchmark_runner.py tests/test_experiment_data.py tests/test_query_coverage_generation.py tests/test_query_driven_rework.py tests/test_torch_runtime_controls.py tests/test_training_does_not_collapse.py training/checkpoints.py training/model_features.py training/predictability_audit.py training/query_prior_fields.py training/query_useful_targets.py training/train_model.py training/training_epoch.py training/training_validation.py`
- `../.venv/bin/python -m pyright scoring/baselines.py experiments/benchmark_report.py experiments/experiment_cli.py experiments/experiment_config.py experiments/experiment_data.py experiments/experiment_methods.py experiments/experiment_pipeline.py experiments/range_diagnostics.py experiments/run_ais_experiment.py experiments/run_inference.py models/workload_blind_range_v2.py workloads/query_generator.py workloads/workload_profiles.py selection/learned_segment_budget.py selection/mlqds_scoring.py tests/test_benchmark_runner.py tests/test_experiment_data.py tests/test_query_coverage_generation.py tests/test_query_driven_rework.py tests/test_torch_runtime_controls.py tests/test_training_does_not_collapse.py training/checkpoints.py training/model_features.py training/predictability_audit.py training/query_prior_fields.py training/query_useful_targets.py training/train_model.py training/training_epoch.py training/training_validation.py`
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
  `data preparation -> workloads -> training -> selection -> evaluation -> benchmarking`,
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
- `workloads/generation/workload.py` dropped to `638` lines, benchmark report to
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

## Checkpoint 5.17-5.24 - Selector Length And Allocation Diagnostics

Status: completed; diagnostic failed

Goal:
- Test whether selector-side length and allocation repairs can preserve the strict-cell MLQDS win while clearing length and learning-causality gates.

Changes:
- Reworked length repair from per-trajectory swap caps to global greedy net-gain allocation.
- Added and separated query-free segment allocation length-support from within-segment geometry tie-breaking.
- Fixed allocation-weight semantics so length support applies when learned segment scores are flat and fairness preallocation uses the blended allocation weight.
- Added a pre-gate benchmark-snapshot policy note.

Tests:
- Focused selector/orchestration tests for length repair, allocation, retained masks, and evaluation behavior.
- Full Ruff, Pyright, pytest, and whitespace checks passed across the implementation checkpoints.
- Strict single-cell replays were run only at the guide-approved diagnostic scale. No full final matrix was run.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint05_global_net_gain_repair06_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint06_segment_length_support_allocation_repair06_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint07_separated_allocation_length_support_ablation_strict_probe_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint08_allocation_weight_semantics_fix_strict_replay_c10_r05`
- command: strict synthetic/debug single-cell replays; see artifact metadata for exact CLI.

Key results:
- Best strict MLQDS QueryUsefulV1: `0.17183721530965693`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- length preservation: `0.7941408411227088`
- gates passed: workload stability, support overlap, predictability, prior-predictive alignment, target diffusion, workload signature
- gates failed: learning causality, global sanity under the old `0.80` length gate
- Shuffled-score delta improved only to `0.008856345116771192`, below the required `0.017759554407510352`.
- Shuffled-prior/no-query-prior deltas stayed around `0.002743017030572781`, below the required `0.005`.

Extra discoveries:
- Under the old `0.80` length policy, same-allocation length-only point selection reached only `0.7597755220341236`, so segment allocation was part of the length blocker. Under the new `0.75` final gate, this diagnostic is historical context rather than a current blocker.
- Under the old `0.80` length policy, the score-protected length frontier cleared length only near `10%` protected learned-score budget; at `25%` materiality length was about `0.7911049677462703`.
- Allocation length support was not material; `MLQDS_without_segment_length_support_allocation` delta was only `0.00012394275871965843`.
- Freeze-mask diagnostics were runtime-heavy, roughly `202-216s` in this range.

Decision:
- Do not run the full final matrix.
- Do not claim success from these strict-cell wins.
- Stop minor length-support tuning unless the next change directly addresses prior/score causality or the segment allocation/length trade-off.

## Checkpoint 5.25-5.42 - Prior Materiality And Head Calibration Diagnostics

Status: completed; diagnostic failed

Goal:
- Diagnose why train-derived prior, behavior, and score signals are too weak at retained-mask level despite healthy workload/prior gates.

Changes:
- Added model-prior materiality diagnostics and per-head prior-output diagnostics.
- Added default-off behavior-head rank loss, sparse-head rank loss, and sparse-head BCE calibration controls.
- Added default-off allocation-floor and score-protected repair diagnostics.
- Added precision-sweep policy guidance: precision variants are runtime/numerical diagnostics only, not a way to rescue failed learning gates.

Tests:
- Focused tests for diagnostics/config/default behavior plus strict single-cell replays for each candidate path.
- Full Ruff, Pyright, pytest, and whitespace checks passed across the implementation checkpoints.
- No full final matrix was run.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint09_model_prior_materiality_strict_replay_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint10_behavior_rank_loss_strict_replay_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint11_allocation_floor010_query_useful_strict_replay_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint12_score_protected_repair010_strict_replay_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint13_per_head_prior_materiality_strict_replay_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint14_sparse_head_rank010_strict_replay_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint15_sparse_head_bce_windowmax_strict_replay_c10_r05`
- command: strict synthetic/debug single-cell replays; see artifact metadata for exact CLI.

Key results:
- Current evidence boundary stayed at MLQDS `0.17183721530965693`, uniform `0.14223795796380634`, Douglas-Peucker `0.16362459837911367`, length `0.7941408411227088`.
- Behavior-rank replay regressed to MLQDS `0.1662931067947708`, length `0.7939681743351743`, and shuffled-score delta `0.00005168542757363892`.
- Allocation floor `0.10` regressed to MLQDS `0.15366824272250135`, length `0.7833962145166923`, and lost to Douglas-Peucker.
- Score-protected repair `0.10` regressed to MLQDS `0.1621987738648618`, length `0.7885179226003864`, and lost to Douglas-Peucker.
- Sparse-head rank `0.10` had a tiny score nudge to `0.17214277022572494`, but length stayed `0.7938028438559355` and causality worsened.
- Sparse-head BCE `window_max_normalized` regressed to MLQDS `0.1548579044007669`, length `0.7882238535165303`, and failed learning causality broadly.

Extra discoveries:
- Raw prior sampling and model-input prior channels exist; the model/selector suppresses prior movement before retained masks.
- Query-hit and boundary heads are saturated near zero. Positive target mass exists, but practical predicted mass above `0.01` is tiny.
- BCE/window-max calibration increased head movement and dispersion, but pointed it at retained-mask-harmful decisions.

Decision:
- Keep rejected auxiliary controls default-off.
- Keep the current best strict artifact as the evidence boundary.
- Do not run the full final matrix.

## Checkpoint 5.43-5.48 - Exact-Pair Repair And Length-Allocation Frontier

Status: completed; diagnostic failed

Goal:
- Test whether more length-greedy exact add/remove repair or allocation alignment can clear length without destroying learned causality.

Changes:
- Added exact-pair length repair diagnostics.
- Ran unbounded and bounded exact-pair strict replays.
- Added benchmark snapshot policy clarification for pre-gate 4x7-shaped outputs.
- Added segment allocation length-alignment diagnostics and a length-support allocation counterfactual.

Tests:
- Focused selector/diagnostic checks and strict single-cell replays.
- No full final matrix was run.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint16_exact_pair_length_repair_strict_replay_c10_r05`
- path: `artifacts/results/query_driven_v2_checkpoint17_bounded_exact_pair_length_repair_strict_replay_c10_r05`
- path: reused `artifacts/results/query_driven_v2_checkpoint13_per_head_prior_materiality_strict_replay_c10_r05` for trace diagnostics.

Key results:
- Exact-pair repair raised length to `0.7990875085863033`, above the new `0.75` final length gate but below the old `0.80` gate used by that historical replay.
- Exact-pair MLQDS regressed to `0.16997958695311988` and failed learning causality/global sanity.
- Segment-budget-head delta became harmful, about `-0.00503`.
- Current strict allocation correlated weakly with length support: Pearson `0.016301257873970753`, Spearman `0.017384034094447172`.
- Current strict allocation correlated strongly with learned score: Pearson `0.910737989020167`, Spearman `0.7620912342189505`.
- Top 10% length-support segments got `45/384` extra slots; top 10% score segments got `201/384`.
- `37/128` trajectories gave zero extra slots to their top three length-support segments.

Extra discoveries:
- The `0.12` length-support weight changed recorded weights more than allocation counts.
- Bounded exact-pair search reduced runtime materially but was still too slow for default use without a runtime plan.
- The old `0.80` length guardrail/code was unchanged in this historical range.

Decision:
- Reject exact-pair repair as a default.
- Future length work needs segment allocation/target alignment, not local repair tuning.
- Pre-gate benchmark snapshots remain scarce diagnostics, not evidence boundaries.

## Checkpoint 5.49-5.55 - Component Boundaries, Matrix Axis, Naming, And Latency

Status: completed

Goal:
- Align the codebase with the active pipeline vocabulary and remove stale component boundaries before further scientific work.

Changes:
- Renamed component ownership from `data` to `data_preparation`, `queries` to `workloads`, method evaluation to `scoring`, and `training` to `learning`.
- Split the old simplification boundary into the relevant `selection` and `scoring` owners.
- Replaced the final/probe matrix coverage axis with named workload-profile IDs; `range_workload_v1` now carries the default `30%` target coverage.
- Updated file names, docs, tests, and guardrails for the new component names.
- Added inference-only MLQDS benchmark latency fields from `matched.MLQDS.latency_ms`.

Tests:
- Focused Ruff/Pyright/tests for each rename and boundary change.
- Full Ruff, Pyright, pytest, help smokes, yamllint, and whitespace checks passed across the range.
- No scientific probe was run.

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was structure, naming, and benchmarking instrumentation work.

Key results:
- The active flow is now `data_preparation -> workloads -> learning -> selection -> scoring -> benchmarking`.
- Benchmark/probe matrix reasoning is now workload-profile/compression based, not raw coverage/compression based.
- Inference-only latency is available in benchmark outputs without diagnostics or training-time overhead.

Extra discoveries:
- Some stable artifact keys intentionally remain named `data`, `query`, or `legacy_*` for schema comparability and diagnostic vocabulary.
- The final matrix still has four workload profiles by seven compression ratios; what changed is the axis meaning, not the size.

Decision:
- Continue using the new component names and workload-profile matrix axis.
- Do not make scientific claims from structure-only changes.

## Checkpoint 5.56-5.59 - Orchestration And Learning Extraction Cleanup

Status: completed

Goal:
- Reduce large orchestration and learning modules by extracting direct owners while preserving artifact contracts and checkpoint behavior.

Changes:
- Added `orchestration/run_payload.py`, `orchestration/run_exports.py`, and `orchestration/mlqds_method_factory.py`.
- Extracted factorized-head diagnostics to `learning/factorized_head_diagnostics.py`.
- Added shared `learning/model_factory.py`.
- Reduced the responsibilities in `learning_scoring_pipeline.py` and `model_training.py`.
- Updated tests and docs around the new ownership boundaries.

Tests:
- Focused tests for run payload/export assembly, method construction, factorized-head diagnostics, and model construction.
- Full Ruff, Pyright, pytest, help smokes, and whitespace checks passed across the range; the range ended at `461 passed` before the next naming cleanup.
- No scientific probe was run.

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was extraction/refactor work.

Key results:
- Orchestration now owns run payload/export assembly and MLQDS method construction directly.
- Learning now owns model construction and factorized-head diagnostics directly.
- The pipeline remains the artifact-contract coordinator rather than a catch-all implementation file.

Extra discoveries:
- `score_checkpoint` no longer needs local constructor defaults because `mlqds_method_factory.py` owns method construction.
- Factorized-head diagnostic tests still cut across orchestration and learning; future test layout work should separate component-local tests from cross-component tests.
- The unsupported checkpoint model-type error remains checkpoint-biased wording, but it is accepted as clear enough for now.

Decision:
- Keep the extracted modules as active owners.
- Do not add compatibility wrappers for the old layout.
- Do not run scientific probes for extraction-only changes.

## Checkpoint 5.60-5.61 - Active Documentation And Run-Config Naming Cleanup

Status: completed

Goal:
- Remove stale active documentation and stale active-code naming after the structural changes.

Changes:
- Updated the root README, guide examples, package READMEs, `CODE_LAYOUT.md`, and active component docs to match the current structure.
- Replaced stale raw coverage-matrix references with workload-profile examples where active docs described current behavior.
- Renamed `config/experiment_config.py` to `config/run_config.py`.
- Renamed `ExperimentConfig` to `RunConfig` and `build_experiment_config` to `build_run_config`.
- Renamed single-run orchestration helpers from experiment-oriented names to run/pipeline-oriented names.
- Added a guardrail that prevents `config.experiment_config` from returning as a compatibility shim.
- Removed generated `__pycache__` directories from verification.

Tests:
- Documentation pass: `git diff --check`, yamllint, property/regression tests, and focused unit tests passed.
- Code naming pass: Ruff, Pyright, full pytest, help smokes, stale-name search, and `git diff --check` passed.
- Full pytest after the naming pass: `462 passed`, one existing PyTorch nested-tensor prototype warning.
- No scientific probe was run.

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was docs and active-code cleanup.

Key results:
- `config/run_config.py` is the active config owner.
- Active code imports `config.run_config`; no active `config.experiment_config`, `ExperimentConfig`, or `build_experiment_config` remains outside negative guardrails.
- Artifact JSON field names were intentionally left unchanged.

Extra discoveries:
- The active guide still uses the word "experiment" in evidence-level prose such as "tiny smoke experiment". That is not stale code naming.
- `legacy_*` artifact fields and `legacy_generator` remain active diagnostic/protocol vocabulary, not compatibility shims.
- `_checkpoint_config_payload` still filters stale saved-checkpoint keys. That is checkpoint loading hygiene, not an active API alias.

Decision:
- Treat `config/run_config.py` as the active config owner.
- Do not reintroduce `config.experiment_config`, `ExperimentConfig`, or `build_experiment_config`.
- Keep artifact JSON keys stable.

## Checkpoint 5.62 - Progress Log Compaction

Status: completed

Hypothesis:
- Checkpoints 5.17 onward had become too granular and were hiding the main evidence decisions among implementation and replay noise.

Expected files:
- `docs/query-driven-rework-progress.md`
- `docs/query-driven-rework-guide.md`

Stop condition:
- Detailed checkpoints 5.17 onward are replaced with fewer logical summaries that preserve decisions, failed probes, extra discoveries, artifact paths, and verification status without implying any scientific gate passed.

Goal:
- Make the progress log useful as a short guide-compliant ledger again.

Changes:
- Replaced 45 detailed checkpoint entries from 5.17 through 5.61 with six logical grouped summaries.
- Condensed the top current-state summary so it no longer depends on removed detailed checkpoint headings.
- Updated guide references that pointed at exact detailed checkpoint headings now covered by the 5.25-5.42 group.
- Preserved artifact paths, rejected-path outcomes, current blockers, extra discoveries, and no-full-grid decisions.

Tests:
- `rg -n "^## Checkpoint 5\." Range_QDS/docs/query-driven-rework-progress.md`
- `wc -l Range_QDS/docs/query-driven-rework-progress.md`
- `git diff --check -- Range_QDS/docs/query-driven-rework-progress.md Range_QDS/docs/query-driven-rework-guide.md`

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was progress-log cleanup.

Key results:
- MLQDS QueryUsefulV1: not applicable
- uniform QueryUsefulV1: not applicable
- Douglas-Peucker QueryUsefulV1: not applicable
- gates passed: not applicable
- gates failed: not applicable

Extra discoveries:
- Raw command transcripts and detailed stdout do not belong in the progress log. The compacted log keeps the artifact IDs needed to recover that detail from `Range_QDS/artifacts/results/`.

Decision:
- Continue using grouped checkpoint entries for long diagnostic/refactor stretches.
- Do not treat log compaction as scientific evidence.

## Checkpoint 5.63 - Final Length Gate Lowered To 0.75

Status: completed

Hypothesis:
- The final/global-sanity length-preservation gate can be lowered from `0.80`
  to `0.75` as a policy/code change without changing training validation
  penalties or turning historical artifacts into success evidence.

Expected files:
- `scoring/geometry_thresholds.py`
- `orchestration/gates.py`
- `orchestration/length_diagnostics.py`
- `selection/learned_segment_budget/diagnostics.py`
- `tests/unit/orchestration/test_query_driven_rework.py`
- `docs/query-driven-rework-guide.md`
- `docs/query-driven-rework-progress.md`

Stop condition:
- Final/global sanity gate code and selector/orchestration diagnostics report
  `0.75`; stale old final-gate references are gone; focused checks
  pass; no success claim is made from old artifacts.

Goal:
- Lower the final length-preservation gate to `0.75` while keeping the evidence
  protocol honest.

Changes:
- Added shared `FINAL_LENGTH_PRESERVATION_MIN = 0.75` and
  `FINAL_LENGTH_PRESERVATION_MAX = 1.20` in `scoring/geometry_thresholds.py`.
- Updated `evaluate_global_sanity_gate` to use the shared final length
  thresholds.
- Updated score-protected length diagnostics and selector allocation/point
  diagnostics to report the same `0.75` target.
- Renamed per-trajectory geometry diagnostic fields from `below_0_8` to
  `below_gate` and added `trajectory_length_preservation_gate_target`.
- Updated active guide thresholds and the progress-log current-state summary.
- At this checkpoint, `validation_length_preservation_min=0.80` was left
  unchanged because it was treated as a separate checkpoint-selection penalty.
  Checkpoint 5.64 supersedes this and aligns the validation default to `0.75`.

Tests:
- `uv run --group dev -- ruff check --fix Range_QDS/orchestration/gates.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- ruff check Range_QDS/scoring/geometry_thresholds.py Range_QDS/scoring/README.md Range_QDS/orchestration/gates.py Range_QDS/orchestration/length_diagnostics.py Range_QDS/selection/learned_segment_budget/diagnostics.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pyright Range_QDS/scoring/geometry_thresholds.py Range_QDS/orchestration/gates.py Range_QDS/orchestration/length_diagnostics.py Range_QDS/selection/learned_segment_budget/diagnostics.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_rework.py -q`: `101 passed`
- stale active final-gate scan for hard-coded `0.80`, excluding historical
  progress-log notes: clean
- `git diff --check -- Range_QDS/scoring/geometry_thresholds.py Range_QDS/scoring/README.md Range_QDS/orchestration/gates.py Range_QDS/orchestration/length_diagnostics.py Range_QDS/selection/learned_segment_budget/diagnostics.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/docs/query-driven-rework-guide.md Range_QDS/docs/query-driven-rework-progress.md`

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was a policy/code threshold update.

Key results:
- MLQDS QueryUsefulV1: not applicable
- uniform QueryUsefulV1: not applicable
- Douglas-Peucker QueryUsefulV1: not applicable
- gates passed: not applicable
- gates failed: not applicable

Extra discoveries:
- The current-best historical length value `0.7941408411227088` is above the
  new final length gate, but the stored artifact still contains gate summaries
  produced under the old `0.80` policy.
- Lowering the final length gate removes length as the obvious current blocker;
  learning causality remains unresolved and still blocks final acceptance.

Decision:
- Treat `0.75` as the active final/global-sanity length-preservation minimum.
- Do not treat old strict artifacts as final success without replaying or
  recomputing gate summaries under the new policy.

## Checkpoint 5.64 - Validation Length Default Aligned To 0.75

Status: completed

Hypothesis:
- `validation_length_preservation_min` should match the new `0.75`
  final/global-sanity length threshold so checkpoint selection is not still
  optimizing against the retired `0.80` policy.

Expected files:
- `config/run_config.py`
- `orchestration/learning_scoring_cli.py`
- `learning/checkpoint_validation.py`
- `learning/README.md`
- `tests/unit/orchestration/test_query_driven_rework.py`
- `docs/query-driven-rework-progress.md`

Stop condition:
- Config defaults, CLI defaults, validation fallback behavior, active docs, and
  validation tests use `0.75`; active references to the retired validation
  default are gone; focused checks pass.

Goal:
- Align validation/checkpoint-selection length pressure with the active final
  length gate.

Changes:
- Added `DEFAULT_VALIDATION_LENGTH_PRESERVATION_MIN` in `config/run_config.py`
  and set it to the shared final length threshold.
- Updated `ModelConfig`, `build_run_config`, and the
  `--validation_length_preservation_min` CLI default to use that constant.
- Updated checkpoint-validation fallback behavior for older config-like
  objects that do not carry the field.
- Updated the focused validation-penalty test to use the shared default.
- Updated `learning/README.md` and this progress log.

Tests:
- `uv run --group dev -- ruff check Range_QDS/config/run_config.py Range_QDS/orchestration/learning_scoring_cli.py Range_QDS/learning/checkpoint_validation.py Range_QDS/learning/README.md Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pyright Range_QDS/config/run_config.py Range_QDS/orchestration/learning_scoring_cli.py Range_QDS/learning/checkpoint_validation.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_rework.py -q`: `101 passed`
- `uv run --group dev -- python -m orchestration.train_and_score --help`
- config/CLI/fallback default smoke: constant `0.75`, CLI `0.75`, config
  `0.75`, fallback penalty matches a `0.75` length minimum
- stale active validation-default scan for the retired `0.80` value: clean
- `git diff --check -- Range_QDS/config/run_config.py Range_QDS/orchestration/learning_scoring_cli.py Range_QDS/learning/checkpoint_validation.py Range_QDS/learning/README.md Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/docs/query-driven-rework-progress.md`

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was a validation-default policy update.

Key results:
- MLQDS QueryUsefulV1: not applicable
- uniform QueryUsefulV1: not applicable
- Douglas-Peucker QueryUsefulV1: not applicable
- gates passed: not applicable
- gates failed: not applicable

Extra discoveries:
- `ranking_top_quantile=0.80`, test fixture values, and historical progress-log
  references still legitimately contain `0.80`; they are not validation length
  defaults.

Decision:
- Treat `validation_length_preservation_min=0.75` as the active default.
- Continue requiring learning-causality evidence before any success claim.

## Checkpoint 5.65 - Centralized Geometry And Validation Defaults

Status: completed

Hypothesis:
- The most valuable duplicate hard-coded policy left in active code is geometry
  gate logic and validation sanity defaults, where final gates, validation
  scoring, CLI defaults, and direct config defaults must not drift.

Expected files:
- `scoring/geometry_thresholds.py`
- `orchestration/gates.py`
- `learning/checkpoint_validation.py`
- `config/run_config.py`
- `orchestration/learning_scoring_cli.py`
- `tests/unit/orchestration/test_query_driven_rework.py`
- `tests/unit/runtime/test_torch_runtime_controls.py`
- `docs/query-driven-rework-progress.md`

Stop condition:
- SED-ratio thresholds are centralized; validation penalty defaults are shared
  by config, CLI, and fallback logic; focused scans find no active duplicate
  geometry-threshold or validation-default literals outside the central owners;
  focused checks pass.

Goal:
- Reduce policy drift risk from repeated literals without changing scientific
  evidence or running probes.

Changes:
- Added `max_sed_ratio_for_compression` and named SED-ratio threshold constants
  to `scoring/geometry_thresholds.py`.
- Updated final global-sanity gates and validation geometry metrics to use the
  shared SED-ratio helper.
- Added shared validation penalty defaults in `config/run_config.py` and used
  them in `ModelConfig`, `build_run_config`, CLI defaults, and
  checkpoint-validation fallback behavior.
- Added test assertions that direct config defaults and CLI defaults stay
  aligned.
- Updated focused tests to reference shared geometry/default helpers instead of
  repeating gate literals.

Tests:
- `uv run --group dev -- ruff check Range_QDS/scoring/geometry_thresholds.py Range_QDS/orchestration/gates.py Range_QDS/config/run_config.py Range_QDS/orchestration/learning_scoring_cli.py Range_QDS/learning/checkpoint_validation.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py`
- `uv run --group dev -- pyright Range_QDS/scoring/geometry_thresholds.py Range_QDS/orchestration/gates.py Range_QDS/config/run_config.py Range_QDS/orchestration/learning_scoring_cli.py Range_QDS/learning/checkpoint_validation.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py`
- `uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py -q`: `123 passed`
- config/CLI/default smoke: validation defaults `0.10`, `0.05`, `0.10`,
  `0.75`; SED thresholds `2.0`, `1.75`, `1.5`
- focused duplicate-literal scan for active SED-threshold and validation-default
  literals
- `git diff --check -- Range_QDS/scoring/geometry_thresholds.py Range_QDS/orchestration/gates.py Range_QDS/config/run_config.py Range_QDS/orchestration/learning_scoring_cli.py Range_QDS/learning/checkpoint_validation.py Range_QDS/tests/unit/orchestration/test_query_driven_rework.py Range_QDS/tests/unit/runtime/test_torch_runtime_controls.py`

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was cleanup/refactor work.

Key results:
- MLQDS QueryUsefulV1: not applicable
- uniform QueryUsefulV1: not applicable
- Douglas-Peucker QueryUsefulV1: not applicable
- gates passed: not applicable
- gates failed: not applicable

Extra discoveries:
- Validation penalty defaults had drifted: direct config defaults were
  `0.35/0.15/0.10`, while the active CLI and fallback behavior used
  `0.10/0.05/0.10`. This checkpoint centralized on the active CLI/fallback
  behavior to avoid silently strengthening future training.
- Remaining scan hits for `1.50`, `1.75`, `0.75`, and `0.80` are unrelated
  fixture values, target formula coefficients, or historical notes, not the
  centralized gate/default policy addressed here.

Decision:
- Keep geometry gate thresholds in `scoring/geometry_thresholds.py`.
- Keep validation sanity defaults in `config/run_config.py`.
- Do not treat this cleanup as scientific evidence.

## Checkpoint 5.66 - Current-Best Gate Reclassification Under 0.75

Status: completed

Hypothesis:
- After lowering the final and validation length thresholds to `0.75`, the
  immediate evidence gap is whether the current-best strict artifact can be
  reclassified under the new gate policy. This should narrow the blocker
  without rerunning training or claiming final success.

Expected files:
- `artifacts/results/query_driven_v2_checkpoint18_current_best_gate_reclassification_len075/gate_reclassification_summary.json`
- `docs/query-driven-rework-progress.md`

Stop condition:
- The current-best strict artifact schema is inspected; global sanity is
  recomputed through current code; the reclassified gate status and remaining
  blocker are recorded; no full matrix is run; no final success claim is made.

Goal:
- Clarify whether length/global sanity remains a strict single-cell blocker
  under the active `0.75` policy.

Changes:
- Added a derived diagnostic artifact that recomputes the current-best stored
  run's global sanity gate with current code and thresholds.
- Updated the progress-log current-state summary.
- Left the historical source artifact unchanged.

Tests:
- `uv run --group dev -- python - <<'PY' ...` generated
  `gate_reclassification_summary.json` from the stored strict artifact using
  `evaluate_global_sanity_gate`.
- inspected the generated summary for policy, scores, recomputed global sanity,
  recomputed final claim summary, and interpretation.
- no scientific replay or final matrix was run.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint18_current_best_gate_reclassification_len075/gate_reclassification_summary.json`
- command: derived diagnostic recomputation from
  `artifacts/results/query_driven_v2_checkpoint13_per_head_prior_materiality_strict_replay_c10_r05/example_run.json`

Key results:
- MLQDS QueryUsefulV1: `0.17183721530965693`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- recomputed global sanity gate: pass
- recomputed global sanity failed checks: `[]`
- length preservation: `0.7941408411227088`
- endpoint sanity: `1.0`
- SED ratio vs uniform: `0.9173337766436357` versus max `1.5`
- gates passed after reclassification: workload stability, support overlap,
  predictability, prior-predictive alignment, target diffusion, workload
  signature, global sanity
- gates failed after reclassification: learning causality

Extra discoveries:
- The current-best strict artifact's stored final claim blocked on both
  `learning_causality_ablations` and `global_sanity_gates`; under the current
  gate policy it blocks only on `learning_causality_ablations` before the full
  workload-profile/compression grid.
- This is a policy reclassification, not a replay. It is strong enough to
  clarify the next blocker, but not enough for final acceptance.

Decision:
- Treat learning causality as the current strict single-cell blocker.
- Do not run the full final matrix until learning causality clears on strict
  evidence.

## Checkpoint 5.67 - Centralized Duplicate Policy And Geometry Constants

Status: completed

Hypothesis:
- Remaining high-risk hard-coded duplication is concentrated in selector
  defaults, active v2 model/target/selector identifiers, and local
  equirectangular geometry constants. Centralizing these should reduce drift
  risk without changing the current candidate or scientific evidence.

Expected files:
- `config/run_config.py`
- `orchestration/learning_scoring_cli.py`
- `selection/selector_types.py`
- `selection/learned_segment_budget/constants.py`
- selector/scoring/orchestration callers that repeated selector defaults
- active model/target/workload modules that repeated product ids
- `workloads/range_geometry.py`
- geometry consumers in scoring, workload generation, target building, and
  learned-segment selection
- focused tests and this progress log

Stop condition:
- Shared selector defaults and active ids are used by config, CLI, scoring,
  diagnostics, and gate code; shared local-geometry constants/helper replace
  duplicated `111.32` distance logic; focused static checks and tests pass; no
  probe or benchmark evidence is claimed.

Goal:
- Remove centralization debt that could make later probe results depend on
  inconsistent defaults or subtly different geometry approximations.

Changes:
- Added `selection/selector_types.py` for canonical selector ids and choices.
- Added shared learned-segment default constants for geometry tie-breaker,
  allocation length-support, allocation weight floor, and score blend weight;
  wired them through config defaults, CLI defaults, validation fallback logic,
  scoring methods, selector diagnostics, and final-gate summaries.
- Added canonical `WORKLOAD_BLIND_RANGE_V2_MODEL_TYPE` and
  `QUERY_USEFUL_V1_FACTORIZED_TARGET_MODE` constants, then replaced production
  comparisons/defaults that previously repeated those strings.
- Moved the local km-per-degree scale and minimum cosine clamp into
  `workloads/range_geometry.py` and added
  `local_equirectangular_distance_km`; updated workload generation,
  QueryUseful target building, learned-segment allocation/repair, and scoring
  geometry to use the shared source.
- Added focused assertions that direct config defaults and CLI defaults stay
  aligned for learned-segment selector knobs.

Tests:
- `../.venv/bin/ruff check ...` on all changed Python files: passed
- `../.venv/bin/python -m pyright ...` on all changed Python files: `0 errors`
- `../.venv/bin/pytest tests/unit/runtime/test_torch_runtime_controls.py tests/unit/workloads/test_range_geometry.py tests/unit/selection/test_learned_segment_budget.py tests/unit/orchestration/test_retained_mask_stage.py tests/unit/orchestration/test_learning_target_stage.py tests/unit/learning/test_model_factory.py tests/unit/learning/test_model_features.py -q`: `77 passed`
- `../.venv/bin/pytest tests/unit/orchestration/test_query_driven_rework.py tests/unit/benchmarking/test_runner.py tests/guardrails/test_rework_guardrails.py -q`: `157 passed`
- `../.venv/bin/pytest tests/unit/scoring/test_metrics.py tests/unit/workloads/test_range_geometry.py -q`: `59 passed`

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was cleanup/refactor work.

Key results:
- MLQDS QueryUsefulV1: not applicable
- uniform QueryUsefulV1: not applicable
- Douglas-Peucker QueryUsefulV1: not applicable
- gates passed: not applicable
- gates failed: not applicable

Extra discoveries:
- `GEOMETRY_TIE_BREAKER_WEIGHT` had also been serving as the default
  segment-length-support allocation weight. The numeric value stays `0.12`,
  but it now has a separate named constant so future tuning cannot silently
  couple two different selector mechanisms.
- The duplicate-literal scan remains noisy for valid fixture values, schema
  keys, quantiles, metric field names, and target coefficients. Those were left
  local because centralizing them would reduce readability without reducing
  policy drift.

Decision:
- Keep active selector ids in `selection/selector_types.py`.
- Keep learned-segment selector defaults in
  `selection/learned_segment_budget/constants.py` and expose config/CLI
  defaults from `config/run_config.py`.
- Keep shared local geometry primitives in `workloads/range_geometry.py`.
- Continue with learning-causality work; this checkpoint does not change the
  current acceptance blocker.

## Checkpoint 5.68 - Learning-Causality Failure Diagnosis

Status: completed

Hypothesis:
- With global sanity reclassified as passing under the active `0.75` length
  policy, the next useful step is diagnosing the failed learning-causality child
  gates from the current-best strict artifact before changing model or selector
  code.

Expected files:
- `artifacts/results/query_driven_v2_checkpoint19_learning_causality_failure_diagnosis_current_best/learning_causality_failure_diagnosis.json`
- `docs/query-driven-rework-progress.md`

Stop condition:
- Failed causality child gates are ranked by threshold gap, mapped to likely
  component failures, and the next code/probe direction is recorded; no final
  matrix or scientific replay is run; no final success claim is made.

Goal:
- Convert the remaining strict blocker from a generic learning-causality failure
  into a component-level diagnosis.

Changes:
- Added a derived diagnostic artifact from the current-best strict run.
- Ranked child-gate margins and shortfalls for every required causality
  ablation.
- Recorded prior-path sensitivity, selected mask-movement diagnostics, selector
  state, and a recommended next checkpoint.
- Updated the progress-log summary and durable discoveries.

Tests:
- `python3 - <<'PY' ...` generated
  `learning_causality_failure_diagnosis.json` from the stored strict artifact.
- The source artifact and learning-causality summary were inspected before
  diagnosis.
- no scientific replay, probe, or final matrix was run.

Experiment artifact:
- path: `artifacts/results/query_driven_v2_checkpoint19_learning_causality_failure_diagnosis_current_best/learning_causality_failure_diagnosis.json`
- command: derived diagnostic extraction from
  `artifacts/results/query_driven_v2_checkpoint13_per_head_prior_materiality_strict_replay_c10_r05/example_run.json`

Key results:
- MLQDS QueryUsefulV1: `0.17183721530965693`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- gates passed in the stored strict cell after `0.75` reclassification:
  workload stability, support overlap, predictability,
  prior-predictive alignment, target diffusion, workload signature, global
  sanity
- gates failed: learning causality
- failed child gate shortfalls:
  - shuffled scores: `0.00890320929073916` shortfall; achieved
    `0.4986805926293919` of required delta
  - shuffled prior fields: `0.002256982969427219` shortfall; achieved
    `0.5486034061145562` of required delta
  - without query-prior features: `0.002256982969427219` shortfall; achieved
    `0.5486034061145562` of required delta
- passing child gate margins:
  - untrained model: `0.0150725509132888`
  - without behavior head: `0.00466018148201922`
  - without segment-budget head: `0.009543435541987698`
  - prior-field-only mismatch: `0.01788860641060822`
- learned-controlled slot fraction: `0.33834134615384615` versus required
  `0.25`

Extra discoveries:
- Shuffling scores causes large retained-mask movement
  (`1864` symmetric-difference decisions, Jaccard about `0.282`), but the
  quality loss is still only about half the required relative causality
  threshold. The learned score controls masks, but the ordering advantage is
  too weak.
- Prior ablations have the opposite shape: raw sampled priors change
  substantially and model-input priors change nontrivially, but mean head
  probability changes only about `0.00001816` and retained-mask movement is only
  `36` decisions with Jaccard about `0.9786`. The prior signal is available but
  mostly attenuated before score/mask decisions.
- Behavior and segment-budget heads pass the current materiality checks, so the
  next fix should preserve them while improving score/prior materiality.
- The pre-repair diagnostic remains higher scoring but length-broken, so simply
  weakening repair or adding temporal scaffold is the wrong direction.

Decision:
- Do not increase workload scale or run the final matrix for this blocker yet.
- Next checkpoint should inspect where prior perturbations collapse between
  model-input features, head outputs, score composition, and selector decisions.
- Expected next focus is `learning/model_features.py`,
  `models/workload_blind_range_v2.py`, `selection/model_score_conversion.py`,
  `orchestration/model_ablations.py`, and retained-mask ablation plumbing.

## Checkpoint 5.69 - Prior-Ablation Diagnostic Centralization

Status: completed

Hypothesis:
- The current learning-causality blocker needs cleaner instrumentation before
  model tuning. The prior-ablation diagnostic chain was duplicated across
  final-eval and checkpoint-selection paths, and the score-level stage was
  recorded under the less explicit `selector_score` name only.

Expected files:
- `orchestration/causality.py`
- `orchestration/retained_mask_ablation_stage.py`
- `orchestration/selection_causality_diagnostics.py`
- `tests/unit/orchestration/test_query_driven_rework.py`
- `docs/query-driven-rework-progress.md`

Stop condition:
- Prior-ablation diagnostics report sampled-prior, model-prior, head-output,
  raw-prediction, score-output, and retained-mask movement through one shared
  chain; focused static and unit checks pass; no probe or final matrix is run.

Goal:
- Remove duplicate diagnostic payload construction and make future prior
  materiality artifacts easier to interpret without changing acceptance gates.

Changes:
- Added shared prior-ablation diagnostic constants and payload builders in
  `orchestration/causality.py`.
- Renamed the prior-ablation score stage to `score_output`; no compatibility
  alias is emitted for that prior-ablation payload.
- Centralized query-prior `TrainingOutputs` cloning so swapped prior fields
  always carry matching `query_prior_field_metadata`.
- Replaced duplicated prior-sensitivity payload construction in final-eval
  retained-mask ablations and checkpoint-selection causality diagnostics.
- Added focused unit tests for the diagnostic chain, tensor-derived sensitivity
  payloads, and query-prior metadata alignment.

Tests:
- `python3 -m py_compile Range_QDS/orchestration/causality.py
  Range_QDS/orchestration/retained_mask_ablation_stage.py
  Range_QDS/orchestration/selection_causality_diagnostics.py
  Range_QDS/tests/unit/orchestration/test_query_driven_rework.py`
- `../.venv/bin/ruff check orchestration/causality.py
  orchestration/retained_mask_ablation_stage.py
  orchestration/selection_causality_diagnostics.py
  tests/unit/orchestration/test_query_driven_rework.py`
- `../.venv/bin/pyright orchestration/causality.py
  orchestration/retained_mask_ablation_stage.py
  orchestration/selection_causality_diagnostics.py
  tests/unit/orchestration/test_query_driven_rework.py`
- `../.venv/bin/pytest tests/unit/orchestration/test_query_driven_rework.py -q`

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was diagnostic cleanup and
  duplicate-centralization work.

Key results:
- MLQDS QueryUsefulV1: not applicable
- uniform QueryUsefulV1: not applicable
- Douglas-Peucker QueryUsefulV1: not applicable
- gates passed: not applicable
- gates failed: not applicable

Extra discoveries:
- Existing future-run instrumentation already had score sensitivity under
  `selector_score`; the root issue was schema naming, so the prior-ablation
  payload now exposes that stage only as `score_output`.
- The shuffled-prior final-eval ablation rebuilt `TrainingOutputs` without
  refreshing `query_prior_field_metadata`, while zero/channel ablations did.
  The centralized helper removes that inconsistency.

Decision:
- Keep this as instrumentation cleanup only. It does not prove learning
  causality and does not justify a final matrix.
- Next scientific step remains a focused prior/score materiality run using the
  updated diagnostic chain to locate where prior perturbations collapse.

## Checkpoint 5.70 - Prior Score-Output Schema Cleanup

Status: completed

Hypothesis:
- Keeping both `selector_score` and `score_output` in prior-ablation payloads is
  compatibility clutter. The root fix is to make `score_output` the only
  prior-ablation score-stage key and update active extraction paths.

Expected files:
- `orchestration/causality.py`
- `benchmarking/reporting/row_fields.py`
- `scripts/jq/causality.jq`
- `tests/unit/orchestration/test_query_driven_rework.py`
- `tests/unit/benchmarking/test_runner.py`
- `docs/query-driven-rework-progress.md`

Stop condition:
- Prior-ablation artifacts and reporting use `score_output` without emitting
  `selector_score`; focused static and unit checks pass; no probe or final
  matrix is run.

Goal:
- Fix the diagnostic naming issue at the source instead of layering a
  compatibility alias over it.

Changes:
- Removed the prior-ablation `selector_score` compatibility key.
- Renamed the helper argument to `score_output`.
- Added score-output extraction to benchmark row fields and the causality jq
  summary script.
- Updated focused tests to assert that prior-ablation payloads do not emit
  `selector_score`.

Tests:
- `python3 -m py_compile Range_QDS/orchestration/causality.py
  Range_QDS/orchestration/retained_mask_ablation_stage.py
  Range_QDS/orchestration/selection_causality_diagnostics.py
  Range_QDS/tests/unit/orchestration/test_query_driven_rework.py
  Range_QDS/benchmarking/reporting/row_fields.py
  Range_QDS/tests/unit/benchmarking/test_runner.py`
- `../.venv/bin/ruff check orchestration/causality.py
  orchestration/retained_mask_ablation_stage.py
  orchestration/selection_causality_diagnostics.py
  benchmarking/reporting/row_fields.py
  tests/unit/orchestration/test_query_driven_rework.py
  tests/unit/benchmarking/test_runner.py`
- `../.venv/bin/pyright orchestration/causality.py
  orchestration/retained_mask_ablation_stage.py
  orchestration/selection_causality_diagnostics.py
  benchmarking/reporting/row_fields.py
  tests/unit/orchestration/test_query_driven_rework.py
  tests/unit/benchmarking/test_runner.py`
- `../.venv/bin/pytest tests/unit/orchestration/test_query_driven_rework.py
  tests/unit/benchmarking/test_runner.py -q`
- `jq -n -f Range_QDS/scripts/jq/causality.jq`

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was schema cleanup.

Key results:
- MLQDS QueryUsefulV1: not applicable
- uniform QueryUsefulV1: not applicable
- Douglas-Peucker QueryUsefulV1: not applicable
- gates passed: not applicable
- gates failed: not applicable

Extra discoveries:
- Benchmark row extraction previously stopped at prior samples, model-prior
  features, and head output. It did not expose the score-output sensitivity
  needed to diagnose whether the prior signal collapses after the heads.

Decision:
- Treat stored artifacts that only contain `selector_score` as stale diagnostic
  artifacts. New prior-ablation diagnostics should use `score_output` only.

## Checkpoint 5.71 - Guide Section 2 Evidence Refresh

Status: completed

Hypothesis:
- Section 2 of the active rework guide was stale because it still described the
  older strict debug probe as the current blocker. The latest relevant evidence
  is the current-best strict replay plus the `0.75` length-policy
  reclassification and derived learning-causality diagnosis.

Expected files:
- `docs/query-driven-rework-guide.md`
- `docs/query-driven-rework-progress.md`

Stop condition:
- Section 2 reflects the latest relevant strict-cell evidence, names the active
  blocker correctly, keeps final-success claims forbidden, and targeted doc
  checks pass.

Goal:
- Prevent the guide from steering the next checkpoint toward superseded
  workload-health/global-sanity work.

Changes:
- Replaced the stale section-2 debug-probe narrative with current-best strict
  scores, active gate status, learning-causality child-gate failures, passing
  child gates, selector-control status, prior-path sensitivity, and next
  checkpoint direction.
- Recorded that global sanity passes only after the active `0.75`
  length-policy reclassification, and that the reclassification is diagnostic,
  not final acceptance evidence.
- Recorded `score_output` as the canonical prior-ablation score-stage key.

Tests:
- `git diff --check -- Range_QDS/docs/query-driven-rework-guide.md
  Range_QDS/docs/query-driven-rework-progress.md`
- `python3 - <<'PY' ...` verified that section 2 contains current-best
  markers and does not contain stale debug-probe markers.
- `rg -n "0\.0645|0\.1190|0\.1478|first active blocker is|Global sanity
  also failed|train accepted|dominant rejection reason"
  Range_QDS/docs/query-driven-rework-guide.md` returned no stale matches.

Experiment artifact:
- path: not generated
- command: no scientific probe was run; this was guide maintenance.

Key results:
- MLQDS QueryUsefulV1: `0.17183721530965693`
- uniform QueryUsefulV1: `0.14223795796380634`
- Douglas-Peucker QueryUsefulV1: `0.16362459837911367`
- gates passed after `0.75` reclassification: workload stability, support
  overlap, predictability, prior-predictive alignment, target diffusion,
  workload signature, global sanity
- gates failed: learning causality

Extra discoveries:
- Section 2 was materially misleading. It identified workload
  generation/signature stability and global sanity as the active first blocker,
  but the current-best strict cell has moved past those gates under the active
  policy.

Decision:
- Treat learning causality as the active blocker.
- Do not run the final matrix or increase workload scale before the narrower
  score/prior materiality diagnosis is resolved.
