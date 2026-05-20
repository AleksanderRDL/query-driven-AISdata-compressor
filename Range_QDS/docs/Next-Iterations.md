# Next Iterations

Use [`query-driven-implementation-research-guide.md`](query-driven-implementation-research-guide.md)
as the source of truth for gates, evidence levels, protocol rules, and probe
scale. Use [`query-driven-implementation-progress.md`](query-driven-implementation-progress.md)
for the condensed evidence boundary.

## Current Defaults

- primary metric: `QueryLocalUtility`
- workload profile: `range_query_mix`
- active anchors: `density=0.80`, `sparse_background_control=0.20`
- active footprints: `medium_operational=0.6923076923076923`,
  `large_context=0.3076923076923077`
- footprint point-hit fraction bands:
  `medium_operational=[0.006,0.030]`, `large_context=[0.010,0.045]`
- proposal calibration: deterministic low-band point-hit target inside the
  lower `25%` of each footprint band, followed by unchanged acceptance gates
- target/model/selector: `query_local_utility_factorized`,
  `workload_blind_range`, `learned_segment_budget`
- target detail: `conditional_behavior_target_variant=query_segment_local_behavior_utility`,
  `replacement_representative_keep_fraction=0.35`,
  `segment_budget_target_aggregation=top20_mean`

These are current defaults, not fixed doctrine. Scoring weights,
anchor-family weights, footprint-family weights, and footprint spatial/temporal
dimensions can change when a gate-by-gate diagnosis shows the workload/profile
and metric are not producing a coherent trainable local-query signal. Keep the
change evidence-gated; do not tune them to rescue one weak probe. Global sanity
must keep being computed and improved where possible, but during the current
phase it is a reported guardrail rather than the first hard blocker ahead of
local query behavior and learning causality.

The latest current-default strict blocker-localizing artifact is the
source-stratified Level 3 replay of the multiplicative
`query_segment_local_behavior_utility` target at the 64-ship/256-point/40-query
shape:
`artifacts/results/query_driven_behavior_segment_target_mult_level3_scale64_query40_seed2527/example_run.json`.

This replay remains the blocker-localizing reference. Later direct
turn/continuity behavior-target variants failed the guide's smaller evidence
levels and were rejected. MLQDS beats uniform and Douglas-Peucker on
QueryLocalUtility (`0.1431090566` versus `0.1247681518` and `0.1153266238`).
It passes workload stability, support overlap, target diffusion, workload
signature, predictability, prior-predictive alignment, and global sanity. It
still fails learning causality and the final grid requirement. Do not make
final success claims from it.

The smaller active-target diagnostics remain useful scale context:
`artifacts/results/query_driven_behavior_segment_target_mult_level2_seed2532/example_run.json`
failed workload stability and workload signature at 32/192/24-query scale.
`artifacts/results/query_driven_behavior_segment_target_mult_scale48_query32_seed2533/example_run.json`
fixed workload stability at 48/192/32-query scale but still failed workload
signature, predictability, prior alignment, and learning causality. Treat those
runs as blocker localization, not learning evidence.

A later Level 2 partial-alignment replay emitted the current
behavior/replacement diagnostic surface but is not promotion evidence:
`artifacts/results/query_driven_behavior_partial_alignment_level2_seed2633/example_run.json`
and
`artifacts/results/query_driven_behavior_partial_alignment_level2_seed2633/family_transfer_path_diagnostic.json`.
MLQDS lost to uniform on QueryLocalUtility (`0.0811594668` versus
`0.1133670216`) and failed target diffusion, workload signature,
predictability, prior-predictive alignment, learning causality, and global
sanity. Its partial-alignment readout was mixed: behavior partial Spearman
controlling replacement was positive for final score (`0.1854`), query-hit
(`0.1207`), and path support (`0.1740`), but negative for ship evidence
(`-0.0488`) and segment budget (`-0.0165`). Replacement aligned more strongly
than behavior to final score and ship evidence. Use this artifact only as a
diagnostic-surface validation and failed pre-causality localization, not as a
reason to promote behavior-head semantics or residualized pseudo-targets.

The remaining Level 3 failure is specific. Shuffled scores, untrained control,
prior-field-only, and no-segment-budget controls lose as expected. The failed
children are query-prior and behavior-head dependence:
`shuffled_prior_fields_should_lose`, `without_query_prior_features_should_lose`,
and `without_behavior_utility_head_should_lose`. Shuffled-prior and
no-query-prior deltas are both `0.0`. No-behavior-head loses by only
`0.0014985765`, below the `0.005` materiality gate.

Selector alignment is mixed. Exact retained-marginal raw-score Spearman is
`0.2779` and selector-score Spearman is `0.2881`, but segment-score Spearman is
`-0.0812` and behavior-component Spearman is `-0.0486`. The fitted behavior
head still has only about `1.6%` of target std and Kendall tau `0.0251`.
Treat the next behavior-head change as a target/transfer/coupling fix, not a
generic loss-weight, selector-blend, or scalar-prior-scale fix.

Older seed `2524` strict artifacts remain useful historical diagnostics for
segment-score/length-support conflict, uniform-no-length-support contrast, and
semantic-alignment checks. They are no longer the front-door evidence boundary
for the current-default path:
`artifacts/results/query_driven_segment_length_conflict_diag_level3_range_query_mix_seed2524/example_run.json`;
`artifacts/results/query_driven_uniform_segment_allocation_diag_level3_range_query_mix_seed2524/example_run.json`;
`artifacts/results/query_driven_segment_budget_top20_level3_range_query_mix_seed2524/example_run.json`.

Generator-only evidence remains useful background: source-stratified Level 3
probes with `range_query_mix`, `n_queries=48`, `max_queries=384`, and seeds
`2524`/`2525` pass workload stability and workload signature. The strict replay
above is the stronger current blocker-localizing artifact because it includes
training and causality checks.

Earlier generator-only profile calibration rejected one-seed footprint tweaks as
new defaults. A `large_context=3.6km/7.25h` and max point-hit fraction `0.050`
variant passed the known Level 2 seed, but failed adjacent Level 2 seeds and
Level 3 generator probes. Do not promote it without stronger evidence.

A guarded `workload_blind_range` prior-feature-scale replay was rejected as
a default. It made prior ablations material, but at Level 3 the effect was
anti-causal: shuffled/zeroed prior fields and prior-field-only score did not
lose as required, retained-marginal alignment stayed negative, and global
sanity still failed. Do not continue by simply increasing prior scale.

The previous generator-fixed strict replay remains useful as historical
pre-alignment comparison:
`artifacts/results/query_driven_generator_fixed_level3_range_query_mix_seed2524/example_run.json`.
The segment-budget top-20% fix made the target/selector contract honest, but it
did not fix causality in that older cell: no-segment-budget still beat primary,
query-prior ablations barely moved the retained mask, and the family/head
diagnostic blocked on `conditional_behavior_utility`.

A composite query-local behavior target was also rejected as an active default:
`artifacts/results/query_driven_query_local_behavior_level3_range_query_mix_seed2524/example_run.json`.
It made predictability pass, but target diffusion failed, QueryLocalUtility
dropped, shuffled/untrained/no-behavior/no-segment ablations all beat primary,
and prior ablations still barely moved the retained mask. Do not continue by
stuffing more query-presence mass into `conditional_behavior_utility`.

A selector allocation-floor contrast replay was rejected as an active default:
`artifacts/results/query_driven_allocation_floor0_level3_range_query_mix_seed2524/example_run.json`.
It raised MLQDS QueryLocalUtility to `0.1516471003`, but predictability and
learning causality still failed, no-segment-budget and untrained controls beat
primary by more, segment-score retained-marginal alignment turned negative, and
segment allocation entropy stayed high. Do not keep tuning selector floor until
head/segment transfer is directionally correct.

Behavior-head segment authority was rejected as an active default:
`artifacts/results/query_driven_behavior_segment_diagnostic_level3_range_query_mix_seed2524/example_run.json`
and
`artifacts/results/query_driven_behavior_rank015_segment_diagnostic_level3_range_query_mix_seed2524/example_run.json`.
Default behavior-head-as-segment scored `0.13933` and behavior allocation-only
`0.14187`, both below primary `0.14239`. With behavior-rank loss, primary rose
to `0.14377`, but behavior-head-as-segment stayed low at `0.13943` and
behavior allocation-only reached only `0.14341`. The no-segment-budget ablation
still beat primary at roughly `0.1513`. Do not replace the segment-budget head
with `conditional_behavior_utility`.

Uniform/fair segment allocation without length support was rejected as an
explanation for the older seed `2524` no-segment win:
`artifacts/results/query_driven_uniform_segment_allocation_diag_level3_range_query_mix_seed2524/example_run.json`.
It scored only `0.12412`, far below primary `0.14239`. Active learned segment
allocation without length support scored `0.13836`, also below primary. In that
older cell, the winning no-segment variants were specifically the
length-support fallback:
no-segment-budget `0.15128` and no-segment allocation-only `0.15074`. Read this
as a learned segment-budget versus query-free geometric length-support conflict,
not as evidence that neutral allocation is good.

The active segment allocator remains score-dominated but weakly
length-support-aligned. The latest 64/256/40 replay no longer has the
no-segment-budget ablation beating primary, but it still reports weak
segment-score/length-support overlap and fails query-prior/behavior-head
causality. This makes the next fix semantic: do not keep adding selector
contrast or swapping in another head until the target, head loss, and selector
allocation contract explain why learned segment scores should preserve
query-local regions directly.

A guarded segment transfer calibration was rejected as an active default:
`artifacts/results/query_driven_segment_transfer_zblend_level3_range_query_mix_seed2524/example_run.json`.
The derived admissibility diagnostics allowed the
`segment_score_allocation_weight_zblend` probe because it used pre-selection
signals and avoided post-selection attribution. The strict replay still reduced
MLQDS QueryLocalUtility to `0.1406240561`, left predictability and causality
failed, kept no-segment-budget (`0.1511892102`), no-segment allocation-only
(`0.1506543116`), and untrained (`0.1487909782`) above primary, and preserved
the same `score_dominated_length_support_conflict`. Do not continue by tuning
selector transfer calibration in isolation.

A model-facing `route_density_prior` exposure probe was rejected as an active
default:
`artifacts/results/query_driven_route_density_prior_enabled_level3_range_query_mix_seed2524/example_run.json`.
The channel was genuinely visible when enabled: shuffled-prior route-density
model-input delta was `0.4901`, and removing only route density changed 16
retained decisions. The direction was still wrong. Primary QueryLocalUtility
dropped to `0.1421144423`, removing route density from the enabled replay
improved QueryLocalUtility by `0.0002200603`, and predictability/causality still
failed. Keep `route_density_prior` disabled by default unless a later
workload/scoring redesign changes the evidence.

A selector allocation length-support probe was also rejected as an active
default:
`artifacts/results/query_driven_length_support_weight035_level3_range_query_mix_seed2524/example_run.json`.
Raising `learned_segment_allocation_length_support_weight` to `0.35` improved
MLQDS QueryLocalUtility to `0.1449775496`, made global sanity pass, and made
the untrained ablation lose. It still failed predictability and learning
causality: no-segment-budget remained better (`0.1512785892`), no-prior and
no-behavior were still wrong-way/immaterial, and the segment allocator still
reported `score_dominated_length_support_conflict`. Treat this as evidence that
query-free length support is a material allocation signal, not as a learned
query-local success.

An earlier selector-to-eval segment teacher transfer diagnostic compared the
seed `2524` current-default artifact and the rejected length-support `0.35`
artifact:
`artifacts/results/query_driven_length_support_weight035_level3_range_query_mix_seed2524/selection_eval_segment_teacher_transfer_diagnostic.json`.
In that older comparison, the seed `2524` current-default artifact was weakly
admissible for a guarded selection-side segment-marginal teacher: its
per-artifact decision was
`guarded_selection_segment_calibration_probe_admissible`, the train-side teacher
shape was viable, selection to eval target Spearman was `0.1167`, and four
selector features have consistent positive selection/eval sign. The
length-support `0.35` variant is not an admissible promotion path: its
per-artifact decision is
`diagnose_transfer_features_before_guarded_calibration_probe`, selection/eval
target Spearman drops to `0.0291`, top-k selection/eval teacher overlap remains
zero through top `10%`, `segment_score` becomes contradictory-sign,
`segment_allocation_weight` is weak on both splits, and
`segment_length_support_score` is consistently negative.
This means length-support weighting should not be promoted from that branch.
The latest 64/256/40 train/eval exact-marginal diagnostic is stricter and
currently rejects guarded segment-marginal calibration, so do not build that
training target until transfer-feature alignment improves.

Train-side exact marginal instrumentation is now available and has Level 1,
Level 2, and Level 3 diagnostic evidence:
`artifacts/results/query_driven_train_marginal_diag_level1_smoke_seed2526/example_run.json`
and
`artifacts/results/query_driven_train_marginal_diag_level1_smoke_seed2526/train_eval_segment_teacher_transfer_diagnostic.json`;
`artifacts/results/query_driven_train_marginal_diag_level2_range_query_mix_seed2527/example_run.json`
and
`artifacts/results/query_driven_train_marginal_diag_level2_range_query_mix_seed2527/train_eval_segment_teacher_transfer_diagnostic.json`;
`artifacts/results/query_driven_train_marginal_diag_level3_range_query_mix_seed2527/example_run.json`
and
`artifacts/results/query_driven_train_marginal_diag_level3_range_query_mix_seed2527/train_eval_segment_teacher_transfer_diagnostic.json`;
`artifacts/results/query_driven_train_marginal_scale64_query40_level3_range_query_mix_seed2527/example_run.json`
and
`artifacts/results/query_driven_train_marginal_scale64_query40_level3_range_query_mix_seed2527/train_eval_segment_teacher_transfer_diagnostic.json`.
The path is non-leaky: `selector_trace_diagnostics.train_primary` uses
`teacher_usage_split=train`, and the latest Level 3 train marginal selector
records `uses_train_queries=true` and `uses_eval_queries=false`. Do not turn it
into training semantics yet. Level 2 was blocked by workload signature, target
diffusion, predictability, learning causality, global sanity, and
Douglas-Peucker comparison. The original 48/192 Level 3 run passed target
diffusion and global sanity, but failed workload stability, workload signature,
prior-predictive alignment, predictability, learning causality, and
Douglas-Peucker comparison. The 64/256/40 Level 3 replay fixed workload health
and predictability, but learning causality still fails on query-prior and
behavior-head dependence. The latest train/eval transfer diagnostic still
rejects a guarded calibration probe:
`diagnose_transfer_features_before_guarded_calibration_probe`, target Spearman
`-0.6151`, and top-k teacher overlap zero through top `10%`.

A head-contrast loss probe has also been rejected as an active default:
`artifacts/results/query_driven_head_contrast_sparse025_behavior015_level3_scale64_query40_seed2527/example_run.json`
and
`artifacts/results/query_driven_head_contrast_sparse025_behavior015_level3_scale64_query40_seed2527/train_eval_segment_teacher_transfer_diagnostic.json`.
It used sparse-head rank loss `0.25`, window-max-normalized sparse BCE targets,
and behavior-rank loss `0.15`. MLQDS QueryLocalUtility rose only slightly to
`0.1402280700`, learning causality still failed, query-prior ablations became
more wrong-way at `-0.0006932385`, no-behavior stayed wrong-way, and train/eval
segment-teacher transfer remained rejected with target Spearman `-0.6785`.
Do not promote those loss settings without a new strict replay that fixes the
failed causality children.

A model-facing square-root prior transform probe has also been rejected as an
active default:
`artifacts/results/query_driven_prior_sqrt_level3_scale64_query40_seed2527/example_run.json`
and
`artifacts/results/query_driven_prior_sqrt_level3_scale64_query40_seed2527/train_eval_segment_teacher_transfer_diagnostic.json`.
It expanded model-facing prior contrast and changed more retained decisions
than the identity prior path, but it still failed the same causality children:
`shuffled_prior_fields_should_lose`,
`without_query_prior_features_should_lose`, and
`without_behavior_utility_head_should_lose`. MLQDS QueryLocalUtility rose only
slightly to `0.1396786660`; shuffled-prior, no-query-prior, and no-behavior
remained wrong-way at `-0.0003197163`, `-0.0003080556`, and `-0.0005148169`.
The train/eval transfer diagnostic still rejected guarded segment-marginal
calibration with selection/eval target Spearman `-0.6084` and top-k overlap
zero through top `10%`. The production default is restored to identity prior
probabilities. Do not continue by adding another scalar prior transform unless
the hypothesis also explains directionality.

## Next Checkpoint

Return to semantic causality diagnosis before architecture changes. Start from
this hypothesis:

> The current default has a healthy workload/profile cell at 64/256/40 and can
> produce a positive QueryLocalUtility gap over uniform and Douglas-Peucker.
> The remaining strict blocker is not broad workload instability and not global
> sanity. It is query-local semantic causality: retained masks are insensitive
> to query-prior fields and the behavior-utility head even though shuffled
> scores, untrained control, prior-field-only, and no-segment-budget controls
> now lose as expected.

Stop condition:

- If a focused diagnostic shows query-prior fields are not reaching the selector
  with material contrast, fix model-facing feature flow or scaling at the root
  and rerun the smaller evidence levels.
- If query-prior fields have contrast but do not align with retained marginal
  QueryLocalUtility, revisit workload/scoring compatibility, including scoring
  weights, anchor/footprint weights, or footprint spatial/temporal dimensions.
- If behavior-head target/loss transfer is weak, fix target/loss coupling before
  selector blending or architecture changes.
- If train/eval segment-marginal transfer remains contradictory, do not build a
  segment-marginal training target from it.
- If a replay regresses workload stability or workload signature, return to
  generator diagnostics. Do not tune model code against unhealthy workloads.
- If target/prior signal is coherent but ablations still match or beat the
  primary, fix target/loss/head/selector coupling. Do not keep changing segment
  aggregation, selector floor, scalar prior scale, or prior probability
  transforms in isolation, and do not solve this with a larger temporal
  scaffold.
- If length-support fallback continues to beat learned segment-budget
  allocation, diagnose allocation authority and segment target semantics before
  promoting another learned segment score. Do not misread this as a uniform
  allocation win.
- If a query-free length-support selector change improves QLU, keep it
  diagnostic until learned segment/behavior/prior causality also improves. Do
  not call query-free length support a learned query-local solution.
- If a train-side segment-marginal calibration probe is built, require
  selection/eval transfer-feature diagnostics to stay non-contradictory before
  running larger strict evidence.
- If train-side marginal diagnostics are run, treat Level 1 output as wiring
  only. Use Level 2 for blocker localization and Level 3 before making any
  serious trainability claim.
- If a target change makes predictability pass but reduces QueryLocalUtility or
  worsens ablation deltas, reject it as diagnostic-only rather than promoting
  it as a default.
- Keep global sanity reported and improve it where possible, but do not make it
  the first hard blocker while query-local learning is unresolved.

Required diagnosis during the next checkpoint:

- keep seed `2527`, source-stratified split, `range_query_mix`, `n_ships=64`,
  `n_points=256`, `n_queries=40`, `max_queries=384`, and
  `range_train_workload_replicates=4` for strict replay comparisons unless the
  diagnostic is explicitly implementation-only
- confirm workload gates remain green if a new replay is run
- selector retained-decision marginal alignment at
  `selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_local_utility_alignment`
- learning-causality child gates, especially untrained, shuffled-prior,
  no-query-prior, no-behavior-head, and no-segment-budget-head behavior
- retained-mask Jaccard/change counts under each ablation
- model-facing prior feature transforms and whether useful prior channels are
  disabled, flattened, or ignored before the score path reaches the selector
  (`route_density_prior` was inspected and remains rejected as a default
  model-facing channel)
- direct `query_point_recall` and query-local interpolation/turn/continuity
  component deltas
- global sanity guardrails as diagnostics, not initial hard blockers
- predictability top-1% lift by head and prior channel
- target-to-head transfer for `query_hit_probability`,
  `conditional_behavior_utility`, `replacement_representative_value`, and
  `segment_budget_target`; use this after workload signatures are stable, and
  make the next fix semantic/training-pressure oriented rather than another
  prior input scale change
- behavior/replacement partial-alignment diagnostics under
  `conditional_behavior_replacement_partial_alignment`; these are
  diagnostic-only and must not be treated as accepted target semantics. The
  retired positive-residual pseudo-target did not show replacement-rank
  decoupling at Level 1 and should not be promoted.
- family/head transfer for active families, especially whether the weak
  `conditional_behavior_utility` path is a target problem or a downstream
  low-contrast selector problem
- active-metric retained-marginal alignment under
  `selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_local_utility_alignment`;
  do not read this from
  `learning_causality_summary.selection_causality_diagnostics`, and do not use
  legacy ship-evidence proxy rows as the primary current-metric conclusion
- segment allocation entropy, target diffusion, and whether query-free
  geometric length-support fallback is still beating or masking learned
  segment-budget authority; treat the floor-0, behavior-head, and
  uniform-no-length diagnostics as evidence that selector contrast,
  behavior-head substitution, and uniform allocation were insufficient in the
  older branch
- guarded train-side segment marginal calibration evidence. The latest
  train/eval exact-marginal diagnostic rejects this path for now; build it only
  after transfer-feature alignment is non-contradictory.
- segment allocation conflict fields:
  `segment_score_to_length_support_spearman`,
  `allocation_weight_to_length_support_spearman`, top-k score/length-support
  overlap, and `component_diagnosis`; the current default is explicitly
  `score_dominated_length_support_conflict`
- selection/eval segment-marginal transfer fields: train-side teacher
  availability, selection/eval target Spearman, top-k teacher overlap, and
  feature sign consistency for `segment_score`, `segment_allocation_weight`,
  and `segment_length_support_score`
- train/eval segment-marginal transfer fields from
  `selector_trace_diagnostics.train_primary` when
  `--query_local_utility_train_marginal_diagnostics` is enabled

## Decision Rules

- Do not run the final grid before the smaller strict evidence passes.
- Do not loosen gates to make the run pass.
- Do not compensate for weak learning with large temporal scaffolding.
- Do not infer training coherence from tiny smokes.
- Do not reintroduce `small_local`, `crossing_turn_change`,
  `boundary_entry_exit`, `port_or_approach_zone`, `route_corridor_like`, or
  `density_route` unless a checkpoint explicitly justifies it with new
  evidence.
- Treat anchor/profile weights and `QueryLocalUtility` component weights as
  adjustable research choices, not immutable constants. Change them only after
  gate-by-gate diagnosis shows the workload/scoring pair is not producing a
  coherent trainable query-local signal.
- Treat footprint spatial/temporal dimensions as adjustable profile design
  choices when diagnostics show the current profile cannot produce stable,
  trainable query-local signal.
- Do not promote lower segment-allocation floors until learned segment/head
  outputs are directionally aligned with retained marginal QueryLocalUtility.
- Do not promote behavior-head segment allocation unless a new strict replay
  reverses the current evidence and passes the relevant causality checks.
- Report global sanity and keep improving it, but do not make it an initial hard
  blocker while local query behavior, prior predictability, and causality are
  still the main research blockers.

## Likely Follow-Up Branches

If workload health fails:

- Diagnose accepted vs rejected point-hit fractions before changing model code.
- Fix profile/generator/split compatibility before touching the selector.
- Keep source/route-family-stratified single-dataset splits for synthetic
  route-family probes.
- Do not promote `n_queries=64` for the current `0.30/0.020` coverage envelope
  unless a new checkpoint fixes the observed coverage-overshoot pressure.

If prior predictability fails:

- Diagnose train/eval support and prior channels before tuning heads.
- Check whether the workload profile is too broad, too sparse, or mismatched to
  the scoring components.

If predictability passes but learning causality fails:

- Inspect target-to-head transfer and selector marginal alignment.
- Check model-facing query-prior feature flow and behavior-head target/loss
  pressure before changing selector architecture.
- Prefer root target/selector fixes over scalar proxy losses or selector blends.

If QueryLocalUtility improves but global sanity fails:

- Treat it as the next optimization branch after local query learning is real.
- Add or tighten query-free sanity support.
- Do not restore high temporal scaffolding.
