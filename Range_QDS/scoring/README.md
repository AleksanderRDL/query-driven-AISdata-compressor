# Scoring Module

Scores frozen retained-point masks against pure workloads and compares MLQDS
with baseline and diagnostic methods.

## Files

| File | Purpose |
| --- | --- |
| `methods.py` | MLQDS, uniform temporal, Douglas-Peucker, oracle, and score-hybrid methods. |
| `metrics.py` | Generic answer-set F1 helpers and `MethodScore`. |
| `geometry_thresholds.py` | Shared final-candidate geometric gate thresholds used by scoring-adjacent diagnostics. |
| `query_cache.py` | Retained-independent query/audit cache. |
| `method_scoring.py` | Method execution and retained-mask scoring. |
| `range_audit_scoring.py` | Range audit construction and retained-support scoring. |
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

- `RangePointF1`: retained in-box point-hit F1. Useful, but too narrow for
  final claims.
- `QueryLocalUtility`: active primary metric for the query-driven system. It
  combines direct query-point recall, query-local interpolation/turn/continuity
  behavior, and light guardrails such as length preservation.

Current `QueryLocalUtility` group weights:

| Group | Weight |
| --- | --- |
| `query_point_mass` | `0.50` |
| `query_local_behavior` | `0.45` |
| `global_sanity` | `0.05` |

Current `QueryLocalUtility` component weights:

| Component | Weight |
| --- | --- |
| `query_point_recall` | `0.50` |
| `query_local_interpolation_fidelity` | `0.20` |
| `query_local_turn_change_coverage` | `0.15` |
| `query_local_continuity` | `0.10` |
| `endpoint_or_skeleton_sanity` | `0.02` |
| `global_shape_guardrail_score` | `0.02` |
| `length_preservation_guardrail` | `0.01` |

`QueryLocalUtility` uses direct audit fields only: `query_point_recall`,
`range_query_local_interpolation_fidelity`, `range_turn_coverage`, and
`range_gap_min_coverage`. It must not source point mass from legacy
`range_point_f1`, and it must not fill missing behavior from older shape,
temporal, average-gap, ship, boundary, or replacement components.

Range audit components:

| Component | Meaning |
| --- | --- |
| `query_point_recall` | Fraction of original in-box points retained. |
| `range_point_f1` | In-box point-hit F1 used as the range answer-F1 score. |
| `range_gap_min_coverage` | Minimum of time-span and along-track missing-run continuity. |
| `range_turn_coverage` | Retained route-change support. |
| `range_query_local_interpolation_fidelity` | Reconstruction fidelity for in-query removed points from retained anchors. |

Non-range workloads are outside the active scoring contract. Generic answer-set
labels still exist for table formatting and historical artifacts, but current
training, scoring, and benchmark paths validate range queries only.

## Reporting Rules

- Final benchmark audits should use exact retained-mask scoring.
- `final_claim_summary` is allowed only for the QueryLocalUtility protocol and must
  block if required single-run gates or final-grid gates fail.
- Checkpoint diagnostics may use cheaper sampling where explicitly configured.
- `ScoringQueryCache` should be reused across MLQDS, baselines, oracle
  diagnostics, and compression-ratio audits.
- Report component tables with aggregate scores. Aggregate-only improvements
  are not enough to understand failures.
