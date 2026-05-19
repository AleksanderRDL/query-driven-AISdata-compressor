# Next Iterations

Use [`query-driven-implementation-research-guide.md`](query-driven-implementation-research-guide.md)
as the source of truth for gates, evidence levels, protocol rules, and probe
scale. Use [`query-driven-implementation-progress.md`](query-driven-implementation-progress.md)
for the condensed evidence boundary.

## Current Defaults

- primary metric: `QueryLocalUtility` schema `5`
- workload profile: `range_query_mix`
- active anchors: `density=0.80`, `sparse_background_control=0.20`
- active footprints: `medium_operational=0.6923076923076923`,
  `large_context=0.3076923076923077`
- target/model/selector: `query_local_utility_factorized`,
  `workload_blind_range_v2`, `learned_segment_budget_v1`

No strict workload-health or learning-coherence rerun has passed under these
defaults. The final grid is still blocked.

## Next Checkpoint

Run a focused schema `5` / two-footprint `range_query_mix` strict diagnostic
before changing model semantics again.

Minimum useful shape:

- Use the guide's Level 2 or Level 3 scale, not a tiny smoke.
- Use `range_query_mix` or one named `range_query_mix_*` profile variant.
- Keep `coverage_calibration_mode=profile_sampled_query_count`.
- Keep `mlqds_temporal_fraction=0.0`.
- Keep `checkpoint_score_variant=query_local_utility`.
- Keep gates unchanged.

Required diagnosis if it fails:

- workload stability and accepted family distributions
- support overlap and query-prior out-of-extent behavior
- direct `query_point_recall`
- query-local interpolation, turn, and continuity components
- global sanity guardrails
- predictability by head and prior channel
- learning-causality child gates
- selector retained-decision marginal alignment at
  `selector_trace_diagnostics.eval_primary.retained_decision_marginal_query_local_utility_alignment`

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

## Likely Follow-Up Branches

If workload health fails:

- Diagnose accepted vs planned anchor and footprint distributions.
- Adjust profile/generator settings before touching model or selector code.

If prior predictability fails:

- Diagnose train/eval support and prior channels before tuning heads.
- Check whether the workload profile is too broad, too sparse, or mismatched to
  the scoring components.

If predictability passes but learning causality fails:

- Inspect target-to-head transfer and selector marginal alignment.
- Prefer root target/selector fixes over scalar proxy losses or selector blends.

If QueryLocalUtility improves but global sanity fails:

- Add or tighten query-free sanity support.
- Do not restore high temporal scaffolding.
