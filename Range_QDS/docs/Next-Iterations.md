
Based on analyzing the results of the checkpoint42 mode-aware current-best strict local artifact, i have deduced the following:

Update after checkpoints 61-73:

- Checkpoint61 is now the workload-healthy strict-cell reference for the
  selection-side marginal teacher. Workload/profile gates pass, but
  predictability and learning causality still fail.
- Checkpoint62 rejects production calibration from current retained-marginal
  row features.
- Checkpoint64 rejects the learned-segment-budget-shaped proxy loss over
  current scalar labels.
- Checkpoint65 joins exact learned-controllable retained-removal marginals to
  selector segment attribution. Top exact-marginal selection rows are
  under-ranked by both point scores and selector segment rank.
- Checkpoint 5.122 adds row-local selector segment context to future
  retained-marginal diagnostics, so the next teacher-construction checkpoint
  should not rely on an external artifact join.
- Checkpoint 5.123 adds a diagnostic-only separated marginal teacher target
  payload over bounded exact retained-removal rows. It is not training
  semantics yet.
- Checkpoint 5.124 proves that payload is emitted end to end in a Level 1
  smoke artifact. The smoke is schema/runtime evidence only.
- Checkpoint 5.125 makes the separated exact marginal teacher payload
  split-aware. Eval summaries may expose target-shape viability, but they must
  not mark themselves as train-side teacher candidates. Checkpoint-selection
  summaries may mark candidacy only when target shape is viable.
- Checkpoint 5.126 adds a guarded diagnostic-only consumer that converts full
  checkpoint-selection separated teacher rows into selector score vectors. Eval
  and compact row-free summaries are rejected before consumption.
- Checkpoint 5.127 tests that consumer at the workload-healthy 384-ship strict
  cell. The consumer is leakage-guarded and valid, but the direct
  teacher-selector loses to the checkpoint-selection primary, so it is not
  accepted for training semantics.
- Checkpoint 5.128 diagnoses the strict direct-consumer failure. The direct
  retained-removal-only teacher is too sparse and narrow: 32 positive point
  scores for a 1638-point budget, 32 positive segments out of 1008, and primary
  wins mostly through ship/point recall.
- Checkpoint 5.129 adds guarded diagnostic-only hybrid checkpoint-teacher
  selectors that blend dense primary selector scores with exact marginal teacher
  vectors. Checkpoint71 proves artifact/runtime shape only; the w25 smoke
  improvement is too small and too under-scaled to justify training semantics.
- Checkpoint 5.130 tests the hybrid at the workload-healthy strict cell.
  Workload/profile gates pass, but the direct teacher and both hybrid blends
  lose to the checkpoint-selection primary.
- Checkpoint 5.131 diagnoses the strict failure. The next likely root issue is
  workload-profile/scoring-component compatibility, not more selector-blend
  tuning.

Going forward, use this document to choose the next checkpoint, but use
`query-driven-rework-guide.md` as the source of truth for gates, evidence
levels, and protocol. Also keep `docs/keep-in-mind.md` in view: workload
profiles, anchor-family weights, and QueryLocalUtility component weights are not
fixed constants. Do not run the final grid yet and do not add another scalar
proxy loss over current labels. The next admissible evidence should diagnose
whether the workload profile and scoring components produce a simple,
query-local, trainable signal together. Do not wire a training loss until that
signal has strict-scale evidence under unchanged gates.

Update after checkpoint 5.171: the active workload profile is now
`range_query_mix`, with only `density` and `sparse_background_control` anchor
families and only `medium_operational` and `large_context` footprints.
The active metric is `QueryLocalUtility` schema `5`: direct
`query_point_recall=0.50`, query-local interpolation/turn/continuity totaling
`0.45`, and global sanity totaling `0.05`. The active workload family weights
are `density=0.80`, `sparse_background_control=0.20`,
`medium_operational=0.6923076923076923`, and
`large_context=0.3076923076923077`.
Older references below to `small_local` or removed families describe
pre-simplified artifacts and must not be treated as active workload-family
requirements.

## Current Blockers

First, **predictability still fails**. Aggregate Spearman is `0.1109086186`, below the required `0.15`. PR-AUC lift is `1.2304850435`, below the required `1.25`. Lift@5 passes, and some individual channels have useful lift, but the aggregate gate does not pass.

Second, **learning causality fails**. The model’s learned components do not hurt enough when ablated:

|Causality child gate|Required loss|Observed loss|
|---|---|---|
|shuffled scores should lose|`0.0144491119`|`0.0089580664`|
|shuffled prior fields should lose|`0.0050000000`|`-0.0001133659`|
|without query-prior features should lose|`0.0050000000`|`0.0000575989`|
|without behavior head should lose|`0.0050000000`|`0.0033472765`|
|without segment-budget head should lose|`0.0050000000`|`0.0036430341`|

That means the system is not yet proving that learned workload-derived information is materially responsible for the retained mask quality.

Third, **selector marginal alignment is bad**. Scores and segment allocations move masks, but not toward exact retained-decision marginal value. That is worse than “model underfit.” It means the target/score/selector stack may be optimizing a proxy that is not the retained-set objective.

Fourth, **prior-to-head transfer is weak**. This is not a raw-prior availability problem. Priors exist, are sampled, and enter model features. The problem is that the head stack and score composition barely react.

Fifth, runtime is now a practical blocker. Checkpoint42 reports total runtime around `625.69s`, retained-mask freeze around `363.45s`, and query-free ablation freeze around `260.07s`. The exact retained-marginal payload itself is not the bottleneck; it was about `17.79s`.

## Remedies going forward

Do **not** run the final grid yet. That would only produce a larger, slower failure. The guide is right: final grid stays blocked until predictability and learning causality pass on smaller strict evidence.

Checkpoint 5.132 added the workload/scoring compatibility payload. The next
best checkpoint should use that payload in a **workload/scoring compatibility
diagnostic**, not another model tweak or selector-blend sweep. Use at least
Level 2 scale, and prefer the workload-healthy strict shape when runtime allows.
Tiny smokes are only for schema/runtime checks.

Compare workload families and QueryLocalUtility components together:

- anchor-family and footprint-family contributions;
- query-hit, ship, entry/exit, crossing, temporal span, turn, shape, speed/heading, and global guardrail component mass;
- target sparsity and monotonicity by score decile;
- per-head prior predictability and whether heads react to the profile;
- whether component weights produce trainable gradients aligned with retained-decision marginal value.

The goal is not perfect real-use-case realism. It is a sensible research
workload/scoring pair that yields a coherent trainable query-local signal.

Use these artifact paths first:

- top-level:
  `workload_scoring_compatibility_diagnostics`
- per method:
  `matched.<method>.range_audit.range_query_metadata_component_summary`

Grouped summaries should usually be enough. Full per-query rows live under the
per-method range audit and should be used only when the grouped family/component
view is insufficient.

Checkpoint74 proves those fields emit end to end at Level 2 scale, but it is
not clean model evidence. It fails workload stability, workload signature,
predictability, learning causality, and global sanity.

Checkpoint75 is the cleaner current blocker evidence because it uses the
workload-healthy current-best strict shape. It passes workload stability,
workload signature, support, target diffusion, prior-predictive alignment, and
global sanity, but still fails predictability and learning causality. It also
narrows the workload/scoring issue: MLQDS loses narrowly to Douglas-Peucker
because of ship-level evidence, especially `ship_f1`, while it is better on
query-local interpolation, shape, speed/heading, entry/exit, and length.

Use checkpoint75 over checkpoint74 for family signs. Against Douglas-Peucker,
MLQDS is negative on `density`, `crossing_turn_change`,
`medium_operational`, and `small_local`, but positive on
`boundary_entry_exit`, `port_or_approach_zone`, `sparse_background_control`,
`large_context`, and `route_corridor_like`.

Do not tune training semantics from checkpoint74. The next admissible move is a
focused workload/scoring/target compatibility checkpoint for ship-level
retained-set evidence under query-local constraints, especially
`small_local` and `medium_operational` footprints. Do not add another selector
blend or scalar proxy loss unless it improves ship-level retained-set marginal
value and unchanged causality gates.

Checkpoint 5.135 adds the missing diagnostic surface for that question. Future
artifacts should inspect `ship_query_evidence_target_alignment` in target
diagnostics and `ship_evidence_counts` under
`matched.<method>.range_audit.range_query_metadata_component_summary`. Use these
fields to decide whether the workload profile, QueryLocalUtility component weights,
or target/head contract is miscalibrated. Do not treat this instrumentation as
evidence of learning; it needs meaningful strict-scale runs, not more tiny smoke
validation.

Checkpoint76 ran those diagnostics at the workload-healthy current-best strict
cell. It reproduces checkpoint75 scores and failed gates, but narrows the next
root issue: query-hit labels carry ship-evidence signal, while the current
behavior and especially segment-budget target do not. MLQDS also misses more
query-hit ships than both uniform and Douglas-Peucker in aggregate. The next
admissible code checkpoint should be diagnostic-only: construct a simple
ship-presence-aware segment-budget/target candidate and compare its target
alignment before wiring training semantics. Do not spend the next checkpoint on
a plain workload-weight tweak unless it also fixes the target/scoring signal
compatibility.

Checkpoint 5.137 adds that diagnostic-only candidate payload. The next
meaningful evidence should run the workload-healthy strict shape again and
compare `segment_budget_ship_presence_candidate_alignment` against
checkpoint76. Do not promote the candidate to a loss or active head target from
unit tests alone. If the candidate improves ship-evidence alignment but harms
final-score/query-hit alignment badly, diagnose the tradeoff before changing
training semantics.

Checkpoint77 provides that strict diagnostic. The pure ship-presence segment
budget is too blunt: it improves ship-evidence alignment but drops final-score
and query-hit top-k mass. The blended candidates are more plausible. The next
code checkpoint should isolate a guarded blended segment-budget target variant
for training diagnostics, preferably final-score/ship-presence or
query-hit/ship-presence blend. Do not make it the default, and do not loosen
gates. The acceptance question is whether the blend improves ship-level
retained-set evidence and learning causality under unchanged gates, not whether
the target diagnostic alone looks cleaner.

Checkpoint78 rejects the guarded query-hit/ship-presence segment-budget target.
It improves the target-side segment-budget ship-evidence diagnostic, but MLQDS
QueryLocalUtility falls to `0.1588862822`, below both the current-best active target
and Douglas-Peucker, and learning causality gets worse. Do not keep tuning that
blend. The next admissible checkpoint should diagnose workload-profile and
QueryLocalUtility component compatibility together, or test a cleaner
final-score/ship-presence/query-local target design under unchanged gates. The
point is a trainable retained-set signal, not matching one target-alignment
diagnostic.

Checkpoint79 rejects the guarded final-score/ship-presence segment-budget target
too. It improves target-side segment-budget ship-evidence Spearman to
`0.1583725136`, but MLQDS QueryLocalUtility is only `0.1592468202`, still well below
the active target and Douglas-Peucker. Causality is worse, with negative
shuffled-score and no-segment-budget-head deltas. Both ship-blend target modes
should stay out of active training options. The next checkpoint should stop
testing segment-budget ship blends and instead diagnose which workload families
and QueryLocalUtility scoring components make the retained-set signal non-trainable.

Checkpoint80 provides that derived diagnosis from checkpoint77/78/79 grouped
payloads. The active strict blocker is not a generic ship-presence miss. It is
concentrated in `small_local`, `density`, `crossing_turn_change`, and
`medium_operational`, with schema `2` losses dominated by `ship_f1`,
`ship_balanced_query_point_recall`, `ship_coverage`, and point-mass recall
terms. The rejected ship-blend target artifacts worsen the same density and
small-local deficits. The next admissible checkpoint should propose and inspect
a workload-profile/QueryLocalUtility component recalibration candidate at diagnostic
level first. Do not wire a new loss or target mode until that candidate shows a
coherent strict-scale signal under the existing gates.

Checkpoint81 runs that diagnostic recalibration probe. A query-local-sensible
component-weight candidate flips the post-hoc active strict score delta from
`-0.0008923639` to `0.0029786298`, and rebalanced family weights improve the
derived weighted query-local deltas. This is not a fix. It is high masking risk:
the candidate wins by reducing pressure on the same ship/point-mass blockers
and by shifting profile mass away from density-route/small-local weakness. Do
not adopt those weights directly. The next admissible checkpoint should build a
profile/scoring candidate that keeps density-route and small-local pressure but
makes their ship/point evidence trainable, or diagnose why those families are
not currently providing usable target signal. Do not add another proxy loss
until this compatibility problem is understood at strict scale.

Checkpoint82 runs that blocker-preserving version. It keeps ship/point evidence
weight at the active total (`0.55`) and preserves critical-family profile
pressure for `density`, `crossing_turn_change`, `small_local`, and
`medium_operational`. The post-hoc score delta is still positive
(`0.0015104602`), but the status is `still_blocked`: all critical families keep
negative or missed ship-evidence signs. This means a plain scoring/profile
recalibration is not enough. The next checkpoint should add family-conditioned
target/head trainability diagnostics for density-route and small-local, then
use meaningful strict-scale evidence before changing any target, head, or
scoring default.

Checkpoint 5.147 adds that instrumentation. Future QueryLocalUtility target
diagnostics now expose `family_conditioned_target_trainability`, and training
fit diagnostics expose `family_conditioned_head_trainability`. These are schema
and runtime surfaces only until strict evidence exists. The next admissible
evidence is a workload-healthy strict diagnostic that reads those fields for
`density` and `small_local`. Do not infer training coherence from the
unit tests, and do not change target/head/scoring defaults until the strict
artifact shows which family/head combination is actually weak.

Checkpoint83 supplies that strict evidence. It reproduces the current-best
strict scores and failed gates, so the added instrumentation did not change the
semantics. `small_local` is the severe blocker: target-side final score,
query-hit, behavior, and segment-budget all rank against family ship-query
evidence, and the trained heads plus composed score remain negative against
that evidence. `density` is mainly target-side weak in behavior and
segment-budget; trained heads recover only weak positive ship-evidence signs.
The next admissible code checkpoint should be diagnostic-only family-local
target/head construction for `small_local` and `density`. Do not spend the
next iteration on another generic scalar proxy, selector blend, or scoring/
profile default unless the diagnostic shows a coherent strict-scale trainable
signal under unchanged gates.

Checkpoint 5.149 adds that Level 0 family-local candidate surface. Future
QueryLocalUtility target diagnostics now expose
`family_local_target_candidate_alignment`, including family query-hit/ship,
ship-gated behavior, boundary/replacement/ship, composed-score, and
segment-budget candidates. This is not training semantics and not learning
evidence. The next meaningful checkpoint should run the workload-healthy strict
shape and inspect this payload for `small_local` and `density`; only then
decide whether any family-local target/head candidate deserves a guarded
training variant.

Checkpoint84 runs that strict diagnostic. The point-level family query-hit/ship
candidate is strong for the blocker families (`small_local` Spearman `0.9740`,
`density` Spearman `0.9191` against ship-query evidence), but the derived
family-local segment-budget candidate is still anti-aligned (`-0.5754` and
`-0.3675`) and only covers about `5%` of ship-query pairs at top-k. Do not
promote the current candidate. The next checkpoint should diagnose segment
aggregation/allocation from family-local point signal, probably by separating
segment budget from within-segment point choice instead of using summed point
mass as the segment target.

Checkpoint 5.151 adds that diagnostic-only separation. Checkpoint85 runs it at
the workload-healthy strict cell. The old point-top-k view of segment candidates
was too pessimistic: two-stage allocation plus family-local point choice
recovers much more ship evidence. For `small_local`, the max-pooled segment
candidate reaches two-stage ship-evidence mass recall `0.8829`, and best pair
coverage is `0.4000`. For `density`, the fractional ship-query segment
candidate reaches best pair coverage `0.6075`, while the max-pooled candidate
gets best mass recall `0.7214`. This still does not pass gates because it is
diagnostic-only. The next admissible code checkpoint may isolate a guarded
non-default segment aggregation target variant, but it must be judged by
unchanged strict retained-mask quality and causality gates, not by these
diagnostic rows alone.

Checkpoint 5.153 adds the guarded non-default target mode
`query_local_utility_factorized_segment_budget_query_ship_max_pool`. Checkpoint86
runs it at the current-best strict cell. This is the first segment aggregation
variant with a useful partial signal: MLQDS QueryLocalUtility reaches
`0.1673482145`, slightly above Douglas-Peucker `0.1671038781`, and the
no-segment-budget-head causality child passes (`0.0061773534` loss). It still
fails predictability and learning causality overall, so do not promote it. The
next checkpoint should be a focused derived comparison of checkpoint85 and
checkpoint86 target/head/causality diagnostics, especially why `small_local`
target-side segment alignment turns positive while the fitted `small_local`
segment/composed head signal remains negative. Do not rerun the strict cell
unless a code/runtime defect requires it.

Checkpoint 5.155 adds that derived comparison as
`query_ship_max_pool_transfer_diagnosis.json`. It confirms the max-pool target
is not a promotion candidate yet. The no-segment-budget-head causality child now
passes, but shuffled scores, prior ablations, and behavior-head ablation still
fail. `density` has positive target and fitted segment signs, while
`small_local` and `crossing_turn_change` have positive target-side segment
signs but negative fitted segment/composed signs. The next checkpoint should
target family/head transfer for `small_local` and `crossing_turn_change`, not
another segment aggregation variant or selector blend. Keep workload/scoring
compatibility in view: if target/head transfer remains incoherent, the scoring
components or workload family profile may need calibration with the target
contract, but do not mask these blocker families away.

Checkpoint 5.157 rejects the broader guarded
`query_local_utility_factorized_query_ship_local_heads` target contract. It makes
target-side `small_local` and `crossing_turn_change` q-hit/behavior/final signs
positive, but the fitted heads stay negative for every composed head in both
families. It also fails target diffusion because the behavior target becomes too
broad, and MLQDS QueryLocalUtility drops to `0.1632708811`, below Douglas-Peucker
`0.1671038781`. Do not promote this mode, do not broaden behavior targets
again, and do not count this as learning coherence. The next checkpoint should
diagnose the model/loss/prior transfer failure under a diffusion-preserving
target contract, or inspect whether workload/profile family calibration can make
the signal trainable without reducing pressure on `small_local` and
`crossing_turn_change`.

Checkpoint 5.158 adds that derived failure diagnosis. It confirms checkpoint90
regresses target diffusion and prior-predictive alignment while lowering MLQDS
QueryLocalUtility by `0.0040773333` versus checkpoint86. The diffusion failure is
specific: `conditional_behavior_utility` support is `0.9396`, above the `0.5`
maximum. The transfer failure is also explicit: `small_local` q-hit/behavior/
segment target-to-fit gaps are `-0.2686`, `-0.3864`, and `-0.2000`;
`crossing_turn_change` gaps are `-0.5243`, `-0.3484`, and `-0.1666`. The next
checkpoint should preserve target diffusion and diagnose the model/loss/prior
transfer path, or recalibrate workload/scoring so the target contract remains
trainable without masking these families away.

Checkpoint 5.159 adds the diffusion-preserving transfer-path diagnosis for
checkpoint86 as
`artifacts/results/query_driven_v2_checkpoint92_family_transfer_path_diagnosis/family_transfer_path_diagnosis.json`.
It finds 11 focused family/head blockers. `crossing_turn_change` query-hit,
segment-budget, and composed heads fit their labels but still misorder
ship-query evidence; `small_local` has the same segment-budget problem while
its query-hit, behavior, and composed targets are still weak. Retained-decision
marginal alignment remains negative at the corrected selector-trace path. The
next checkpoint should add family-conditioned prior predictability diagnostics
before changing model/loss semantics or workload/scoring weights. Without that,
we cannot separate prior-resolution failure from loss/selector transfer failure.

Checkpoint 5.160 adds that diagnostic surface and reruns the guarded max-pool
strict cell as
`artifacts/results/query_driven_v2_checkpoint93_family_prior_predictability_max_pool_current_best_strict_local/example_run.json`.
Scores and gates reproduce checkpoint86: the candidate still slightly beats
Douglas-Peucker on QueryLocalUtility but still fails predictability and learning
causality. The new family-prior rows reject the lazy explanation that the
blocker families have no usable prior signal. `crossing_turn_change` query-hit
and segment-budget prior rank are useful; `small_local` behavior and
segment-budget prior rank are useful. Weak spots remain in `small_local`
query-hit top-k lift and crossing behavior-prior rank, but the next branch is
not “add priors blindly.” The derived checkpoint94 diagnosis says to inspect
score-to-selector marginal calibration before promotion because retained
decision marginal alignment remains negative at the corrected selector-trace
layout.

Checkpoint 5.161 adds that selector-to-retained-marginal calibration diagnosis
as
`artifacts/results/query_driven_v2_checkpoint95_selector_marginal_calibration_diagnosis/selector_marginal_calibration_diagnosis.json`.
It uses the corrected selector-trace layout and separates the failure modes:
28 high-scored low-exact-marginal rows, 19 top exact-marginal rows under-ranked
by selector score, and 19 under-ranked by segment score. The eval-only
separated marginal teacher has viable shape but is not train-side evidence; 4
of its top 10 segment targets are low-ranked by selector segment score and
allocation weight. The next checkpoint should build or diagnose train/selection
side marginal segment calibration evidence under unchanged gates. Do not promote
checkpoint86/93, wire eval exact marginals into training, or tune another
selector blend from this artifact.

Checkpoint 5.162 adds the selection-side marginal segment calibration diagnosis
as
`artifacts/results/query_driven_v2_checkpoint96_selection_marginal_segment_calibration_diagnosis/selection_marginal_segment_calibration_diagnosis.json`.
Selection-side exact marginal rows are present and split-eligible
(`candidate_for_train_side_teacher=true`), but active scores anti-rank them
strongly (`selector_score` Spearman `-0.1610`, `segment_score` Spearman
`-0.0990`). Six of the top 10 selection segment teacher targets are low-ranked
by selector segment score and allocation weight. Selection/eval segment teacher
overlap is only `4/32`, and top-10 overlap is zero. The next checkpoint should
diagnose selection-to-eval segment teacher transfer or build a guarded
calibration probe, with unchanged strict gates as the judge. Do not wire the
selection exact teacher directly into training or selectors from this derived
artifact.

Checkpoint 5.163 adds the selection-to-eval segment teacher transfer diagnosis
as
`artifacts/results/query_driven_v2_checkpoint97_selection_eval_segment_teacher_transfer_diagnosis/selection_eval_segment_teacher_transfer_diagnosis.json`.
It treats non-teacher segments as zero target across all segment candidates.
Selection/eval positive teacher segment overlap is only `4/32`; top 1%, 5%,
and 10% sparse target overlap is zero; sparse target Spearman over the
positive-target union is `-0.7663`. Simple selector features have weak shared
positive alignment (`segment_score` about `0.08`, `learned_count` about
`0.20` on both splits), while length support is consistently negative. The next
checkpoint should not train directly on raw selection teacher targets. Either
diagnose richer transfer features or build a guarded transfer-calibration probe
whose acceptance is unchanged strict retained-mask quality and learning
causality, not teacher fit.

Checkpoint 5.164 adds the segment transfer-feature admissibility diagnosis as
`artifacts/results/query_driven_v2_checkpoint98_selection_segment_transfer_feature_admissibility_diagnosis/selection_segment_transfer_feature_admissibility_diagnosis.json`.
It rejects the tempting `learned_count` signal as post-selection coupled. The
only probe-admissible pre-selection candidates are active `segment_score` and a
simple `segment_score`/`segment_allocation_weight` z-blend. The length-support
counter-signal candidate is rejected as guard-risk. The next checkpoint may
build a guarded non-default pre-selection segment transfer-calibration probe,
but it must be judged by unchanged strict retained-mask quality and learning
causality, not teacher fit. Do not use post-selection attribution.

Checkpoint 5.165 adds and runs that guarded non-default z-blend probe as
`artifacts/results/query_driven_v2_checkpoint99_segment_transfer_calibration_zblend_current_best_strict_local/example_run.json`.
The trace is clean: mode `segment_score_allocation_weight_zblend`,
`applied=true`, no post-selection attribution, no length-support
counter-signal, retained-mask reconstruction matches the frozen primary, and
final effective length-support allocation weight is `0.0`. The probe still
fails predictability and learning causality. MLQDS QueryLocalUtility is
`0.1672369132`, slightly above Douglas-Peucker but below checkpoint93 by
`0.0001113013`; shuffled scores, prior ablations, and behavior-head causality
still fail. Treat this as a rejection of the simple allocation-weight z-blend,
not as selector progress. The next checkpoint should return to coherent
workload/scoring/target signal construction, especially whether QueryLocalUtility
component weights and workload family pressure can produce trainable local
signals without masking active `density` or `medium_operational` pressure.
`small_local` and `crossing_turn_change` are now historical blockers unless a
future checkpoint deliberately reintroduces them.

Checkpoint 5.166 implements the first scoring simplification: the primary
metric is schema `3`, with `ship_presence_and_coverage`,
`boundary_and_event_evidence`, and the ship-coverage part of
`ship_balanced_query_point_recall` removed from the primary aggregate. Remaining
query-point-mass, query-local-behavior, and global-sanity weights are
renormalized. Treat this as implementation only. Do not compare schema `3`
scores against checkpoint99 schema `2` scores as if they were the same metric;
the next useful checkpoint is a focused strict rerun under schema `3`, followed
by gate-by-gate diagnosis if learning causality still fails.

Checkpoint 5.167 rebalances the simplified metric as schema `4`: point mass
`0.50`, query-local behavior `0.45`, and global sanity `0.05`, preserving the
schema `3` component set and within-group proportions. Do not compare schema
`4` scores against schema `2` or `3` scores as if they were the same metric.
The next useful checkpoint is now a focused strict rerun under schema `4`,
followed by gate-by-gate diagnosis if learning causality still fails.

Checkpoint 5.168 renames the active metric to `QueryLocalUtility` and the
production/reporting keys to `query_local_utility`. Old `query_useful_v1`
target modes and output fields are intentionally not kept as production aliases.
New checkpoint artifacts should use the new names; historical checkpoint
numbers remain historical evidence and are not a compatibility contract.

Checkpoint 5.169 simplifies the active workload profile to `range_query_mix`.
Active anchors are now `density=0.80` and `sparse_background_control=0.20`;
active footprints are `small_local=0.2777777777777778`,
`medium_operational=0.50`, and `large_context=0.2222222222222222`.
Removed profile IDs and families are historical only, not production aliases.

Checkpoint 5.170 simplifies `QueryLocalUtility` as schema `5`. Point mass is
now direct `query_point_recall` with weight `0.50`; query-local behavior is
direct interpolation fidelity (`0.20`), turn-change coverage (`0.15`), and
continuity from `range_gap_min_coverage` (`0.10`); global sanity remains
`0.05`. The active score must not source point mass from legacy
`range_point_f1` or fill missing behavior components from fallback audit fields.
The next useful evidence checkpoint is a focused strict rerun under schema `5`
and `range_query_mix`, followed by gate-by-gate diagnosis if predictability or
learning causality still fails. Do not compare schema `5` scores against schema
`2`, `3`, or `4` scores as if they were the same metric.

Checkpoint 5.171 removes `small_local` from the active footprint family set.
The active footprints are now `medium_operational=0.6923076923076923` and
`large_context=0.3076923076923077`. This is a plausible simplification after
schema `5`, because tiny windows are a bad fit for stable interpolation/turn/
continuity behavior scoring. It is not learning-coherence evidence; rerun the
required smaller strict probes before claiming the simplified profile improved
anything.

Checkpoint72 already compared:

- checkpoint-selection primary;
- direct exact marginal teacher selector;
- hybrid exact-teacher/primary blends;
- schema `2` component deltas for ship, point, entry/exit, crossing, temporal span, turn, shape, and length/global sanity.

The hybrid still loses at strict scale. Do not keep tuning hybrid weights unless
new workload/scoring evidence says the exact marginal signal is coherent but
only mis-scaled.

If another under-ranking artifact is needed after that strict probe, build it around the top marginal misses:

For each high-marginal retained/removable/addable point, dump:

`point_index`, trajectory id, source stage, decision type, exact marginal QueryLocalUtility, raw score, selector score, segment score, segment rank, pre/post length-repair state, each factorized head probability/logit, sampled prior channels, normalized model-prior channels, and QueryLocalUtility component deltas.

Then bucket the failures:

1. prior missing or out of support;
2. prior present but head flat;
3. head positive but final score suppresses it;
4. raw score good but segment allocation loses it;
5. length repair/skeleton overrides the learned decision;
6. high score but low exact marginal, meaning the target itself is misaligned.

That one artifact should tell you whether the next fix belongs in priors, heads, final score composition, segment allocation, or length repair.

If the failure is prior-to-head transfer, do not just “scale priors harder.” That has already been partially explored and is too blunt. Better candidates are: per-head/per-channel prior diagnostics, train-time prior-channel dropout, explicit prior-use auxiliary losses, or a small gated prior residual into the relevant heads. But accept any of those only if they improve retained-marginal alignment and causality deltas, not just factorized-head fit.

If the failure is selector marginal alignment, the fix is probably target/selector-level, not model-capacity-level. The selector needs a train-only approximation of retained-decision marginal value, or at least a calibration layer that maps factorized heads to marginal retained-set utility. The repo already has exact eval-side marginal diagnostics; the next research step is a train/selection-side marginal teacher that remains workload-blind at eval.

If the failure is segment allocation, split allocation and point choice more aggressively. The tests already cover plumbing for separate segment allocation scores and segment point scores. Use that separation: one head can decide where budget goes, another can decide which point inside the segment is marginally useful.

For runtime, reduce ablation cost before running more heavy probes. Cache frozen masks where semantics allow it, run only hypothesis-relevant ablations per checkpoint, and keep exact cached QueryLocalUtility marginal diagnostics. Do not optimize by weakening diagnostics; optimize by cutting irrelevant ablations.
