# Simplification Module

Turns per-point scores into retained-point masks.

## Files

| File | Purpose |
| --- | --- |
| `mlqds_scoring.py` | Canonical score conversion used by validation and final evaluation. |
| `simplify_trajectories.py` | Per-trajectory top-k retention with endpoint preservation. |
| `learned_segment_budget/` | Active learned-segment selector and attribution diagnostics. |

### `learned_segment_budget/`

| File | Purpose |
| --- | --- |
| `core.py` | Public selector orchestration and package API implementation. |
| `allocation.py` | Segment rows, score stats, and learned-slot budget allocation. |
| `length_repair.py` | Geometry-aware point selection and length-repair swaps. |
| `diagnostics.py` | Query-free selector geometry, attribution, and counterfactual diagnostics. |
| `trace.py` | JSON-serializable selector trace payload construction. |
| `constants.py` | Selector schema versions and default weights. |

## Rules

- Selection is trajectory-local; there is no global threshold across ships.
- Endpoints are preserved when a trajectory has retained points.
- Equal scores get deterministic pseudo-random jitter to avoid positional bias.
- Supported MLQDS score modes include `rank`, `rank_tie`, `raw`, `sigmoid`,
  `temperature_sigmoid`, `zscore_sigmoid`, and `rank_confidence`.
- `learned_segment_budget_v1` must report skeleton, learned, fallback, and
  length-repair attribution so causality diagnostics can distinguish genuine
  learned control from scaffolding.
