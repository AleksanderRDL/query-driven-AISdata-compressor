You are working on achieving a query-driven, workload-blind AIS trajectory compressor in `Range_QDS`.

Before doing anything, read these documents end to end:

1. `Range_QDS/docs/query-driven-implementation-research-guide.md`
2. `Range_QDS/docs/Next-Iterations.md`
3. `Range_QDS/docs/query-driven-implementation-progress.md` only as supporting context, not as the main task plan.

Treat `query-driven-implementation-research-guide.md` as the source of truth for the objective, protocol rules, acceptance criteria, evidence levels, gate order, probe scale policy, and final-success restrictions.

Treat `Next-Iterations.md` as the source of truth for the current evidence boundary, known blockers, rejected paths, and the next admissible checkpoint.

Current default stack:
- primary metric: `QueryLocalUtility`
- workload profile: `range_query_mix`
- target mode: `query_local_utility_factorized`
- model: `workload_blind_range`
- selector: `learned_segment_budget`
- checkpoint score variant: `query_local_utility`

Current blocker:
- The current strict reference beats uniform and Douglas-Peucker on `QueryLocalUtility`, but final success is still blocked.
- The active blocker is learning causality, especially:
  - query-prior ablations are immaterial;
  - the behavior-utility head is too weak/flat;
  - segment-score retained-marginal alignment is still wrong-way or weak.
- Do not run the final grid until the guide’s required smaller evidence levels pass.

Execution rules:
1. Work in focused checkpoints.
2. Start each checkpoint by stating:
   - hypothesis;
   - expected files to change;
   - evidence level / probe scale;
   - exact stop condition;
   - expected artifact path, if a run is planned.
3. Prefer targeted diagnostics over broad sweeps.
4. Diagnose by gate and component before changing production code.
5. Do not loosen gates just to make a run pass.
6. Do not compensate for weak learning with large temporal scaffolding, selector tricks, raw coverage overrides, or weaker guardrails.
7. Do not claim final success from smoke runs, tiny probes, loose overshoot runs, failed child gates, or generation-only evidence.
8. Small probes are implementation checks only. They are not scientific evidence of learning or final success.
9. Follow the probe scale policy in the guide. If changing metric/profile/target/model/selector semantics, restart from the guide’s smaller required evidence levels.
10. Keep the codebase clean. Remove, isolate, or clearly mark stale/misleading paths. Do not leave one-off experiment hacks in production paths.
11. Avoid compatibility shims unless they are explicitly required for persisted artifacts. Prefer root fixes.
12. Do not repeat rejected paths unless you introduce a materially new hypothesis and explain why the earlier rejection no longer applies.
13. After each checkpoint, update the short progress log with:
    - hypothesis;
    - changed files;
    - validation commands;
    - artifact path;
    - key scores/gates;
    - decision;
    - next admissible step.
14. If evidence contradicts the expected direction, stop and diagnose. Do not keep tuning.

Recommended first checkpoint:
- Start from the current reference artifact named in `Next-Iterations.md`.
- Run or implement only the semantic-causality diagnosis needed to explain:
  1. why query-prior fields do not affect retained masks;
  2. why `conditional_behavior_utility` remains too flat/weak;
  3. why segment-score alignment is negative while raw/selector score alignment is positive.
- Do not implement another generic behavior-rank weight, prior-scale boost, post-context prior residual, selector allocation-floor tweak, or final-grid run unless the guide and diagnostics justify it.