# Scoring Module

Scores frozen retained-point masks against pure workloads and compares MLQDS
with baseline and diagnostic methods.

## Files

| File | Purpose |
| --- | --- |
| `methods.py` | MLQDS, uniform temporal, Douglas-Peucker, oracle, and score-hybrid methods. |
| `metrics.py` | F1 helpers, range audits, and `MethodScore`. |
| `geometry_thresholds.py` | Shared final-candidate geometric gate thresholds used by scoring-adjacent diagnostics. |
| `query_cache.py` | Retained-independent query/audit cache. |
| `method_scoring.py` | Method execution and retained-mask scoring. |
| `score_tables.py` | Text tables for reports. |
| `query_local_utility.py` | Active primary query-driven metric for `range_query_mix`. |

## Methods

- `MLQDSMethod`: trained model plus persisted scaler and trajectory-local
  retained-mask selection.
- `UniformTemporalMethod`: evenly spaced points per trajectory.
- `DouglasPeuckerMethod`: geometry baseline that keeps highest-error points.
- `OracleMethod`: additive-label upper reference, not an exact optimum.
- `ScoreHybridMethod`: temporal-base residual-fill diagnostics.

## Range Metrics

Current range diagnostics:

- `RangeUsefulLegacy`: retained old aggregate range usefulness audit for
  diagnostics and artifact comparability. It is not the primary metric for the
  query-driven rework and cannot support final acceptance by itself.
- `RangePointF1`: retained in-box point-hit F1. Useful, but too narrow for
  final claims.
- `QueryLocalUtility`: active primary metric for the query-driven rework. It
  combines direct query-point recall, query-local interpolation/turn/continuity
  behavior, and light guardrails such as length preservation.

`QueryLocalUtility` schema `5` group weights:

| Group | Weight |
| --- | --- |
| `query_point_mass` | `0.50` |
| `query_local_behavior` | `0.45` |
| `global_sanity` | `0.05` |

Schema `5` component weights:

| Component | Weight |
| --- | --- |
| `query_point_recall` | `0.50` |
| `query_local_interpolation_fidelity` | `0.20` |
| `query_local_turn_change_coverage` | `0.15` |
| `query_local_continuity` | `0.10` |
| `endpoint_or_skeleton_sanity` | `0.02` |
| `global_shape_guardrail_score` | `0.02` |
| `length_preservation_guardrail` | `0.01` |

Schema `5` uses direct audit fields only: `query_point_recall`,
`range_query_local_interpolation_fidelity`, `range_turn_coverage`, and
`range_gap_min_coverage`. It must not source point mass from legacy
`range_point_f1`, and it must not fill missing behavior from older shape,
temporal, average-gap, ship, boundary, or replacement components.

Range audit components:

| Component | Meaning |
| --- | --- |
| `ShipF1` | Whether hit ships remain represented. |
| `ShipCov` | Per-ship point-subset coverage. |
| `EntryExitF1` | Sampled AIS entry/exit support. |
| `CrossingF1` | Point pairs bracketing range-boundary crossings. |
| `TemporalCov` | Retained time span inside the query. |
| `GapCov` | Count-normalized penalty for large missing runs. |
| `GapCovTime` | Time-span variant of the largest missing-run penalty. |
| `GapCovDistance` | Along-track-distance variant of the largest missing-run penalty. |
| `TurnCov` | Route-change support. |
| `ShapeScore` | Range-local route fidelity. |

`RangeUsefulLegacy` remains count-gap based for schema 7. New runs also emit
diagnostic aggregate variants that replace only the gap term:
`range_usefulness_gap_time_score`, `range_usefulness_gap_distance_score`, and
`range_usefulness_gap_min_score`.

Non-range workloads still report answer-set `AnswerF1` and `CombinedF1` for
diagnostic ablations. Current benchmark work is range-only.

## Reporting Rules

- Final benchmark audits should use exact retained-mask scoring.
- `final_claim_summary` is allowed only for the QueryLocalUtility protocol and must
  block if required single-run gates or final-grid gates fail. Old RangeUseful
  results belong under `legacy_range_useful_summary`.
- Checkpoint diagnostics may use cheaper sampling where explicitly configured.
- `ScoringQueryCache` should be reused across MLQDS, baselines, oracle
  diagnostics, and compression-ratio audits.
- Report component tables with aggregate scores. Aggregate-only improvements
  are not enough to understand failures.
