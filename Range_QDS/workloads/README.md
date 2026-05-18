# Workloads Module

Defines workload containers, range query formats, workload generation, and query
execution.

Range is the active product/query surface. Non-range workloads are not part of
the supported training, scoring, or benchmark contract.

## Files

| File | Purpose |
| --- | --- |
| `typed_workload.py` | `TypedQueryWorkload` container. |
| `query_types.py` | Query IDs, pure workload validation, feature padding. |
| `generation/generator.py` | Public workload generator and query assembly. |
| `generation/anchors.py` | Anchor priors and weighted point sampling. |
| `generation/profile_query_plan.py` | Deterministic profile family quota planning and per-query settings. |
| `generation/coverage.py` | Point coverage masks, target normalization, and range acceptance filters. |
| `generation/workload_profiles.py` | Versioned product workload profiles, including `range_workload_v1`. |
| `generation/signatures.py` | Range workload signature payload construction. |
| `query_executor.py` | Range query execution. |
| `range_geometry.py` | Shared range-box and geographic distance helpers. |
| `workload_diagnostics.py` | Range workload quality and label diagnostics. |

## Query Types

| Type | ID | Meaning |
| --- | --- | --- |
| `range` | 0 | Spatiotemporal box; range scoring also measures retained point support. |

Workloads are range-only for active rework runs, e.g. `{"range": 1.0}`.
`pad_query_features` converts typed query dicts into `[M, 12]` features plus
`[M]` type IDs.

## Generation

`generation/generator.py` exposes `generate_typed_query_workload` and keeps
query assembly in one place. Anchor sampling, profile planning, coverage
acceptance, and signature payload construction live in their direct owner
modules under `workloads/generation/`.

Range generation controls:

- `range_spatial_fraction`, `range_time_fraction`: dataset-relative footprint.
- `range_spatial_km`, `range_time_hours`: absolute half-window footprint.
- `range_footprint_jitter`: random footprint scaling.
- `range_time_domain_mode`: `dataset` uses global time bounds; `anchor_day`
  clamps each query to the 24-hour source/calendar day containing its anchor.
- `range_anchor_mode`: `mixed_density` keeps the historical 70% density-biased
  / 30% uniform anchor prior; `dense`, `uniform`, and `sparse` are explicit
  generator settings for ablation and held-out workload tests.
- `range_train_footprints`: run-level train-only footprint families,
  expressed as `spatial_km:time_hours`, cycled across training workload
  replicates. Eval/checkpoint workloads still use the ordinary footprint flags.
- `target_coverage`: point-level query-signal coverage target.
- `max_queries`: optional cap when generation continues past `n_queries`.
- `range_max_coverage_overshoot`: optional absolute tolerance above
  `target_coverage`; candidate boxes that would push union point coverage over
  `target + tolerance` are rejected. Accepts fractions or percentages. Final
  benchmark rows should normally get this from the workload profile, not from
  one-off CLI coverage overrides.

Use `scripts/estimate_range_coverage.py` before changing query count,
footprint, or workload-profile target coverage.

`range_workload_v1` is the active product workload profile for the query-driven
rework and defaults to 30% target coverage. The final grid uses named
`range_workload_v1_*` profile variants rather than a raw coverage-target axis.
Legacy ad hoc generator settings remain useful for diagnostics, but they are
not final-success eligible.

## Execution

- `execute_range_query`: trajectory IDs with points inside the box.
- `execute_typed_query`: dispatch by the query `type` field.

The generator defines the future-query prior for workload-blind training. Do
not tune final claims only to one narrow generator setting.
