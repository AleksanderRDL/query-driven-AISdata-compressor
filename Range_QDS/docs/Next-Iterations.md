
Based on analyzing the results of the checkpoint42 mode-aware current-best strict local artifact, i have deduced the following:

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

The next best checkpoint should be a **score-composition and marginal-under-ranking diagnostic**, not another model tweak. Build one artifact around the top marginal misses:

For each high-marginal retained/removable/addable point, dump:

`point_index`, trajectory id, source stage, decision type, exact marginal QueryUsefulV1, raw score, selector score, segment score, segment rank, pre/post length-repair state, each factorized head probability/logit, sampled prior channels, normalized model-prior channels, and QueryUsefulV1 component deltas.

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

For runtime, reduce ablation cost before running more heavy probes. Cache frozen masks where semantics allow it, run only hypothesis-relevant ablations per checkpoint, and keep exact cached QueryUsefulV1 marginal diagnostics. Do not optimize by weakening diagnostics; optimize by cutting irrelevant ablations.