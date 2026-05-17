"""Learned segment-budget selector public API."""

from simplification.learned_segment_budget.core import (
    GEOMETRY_TIE_BREAKER_WEIGHT,
    LEARNED_SEGMENT_BUDGET_SCHEMA_VERSION,
    LEARNED_SEGMENT_BUDGET_TRACE_SCHEMA_VERSION,
    SEGMENT_ALLOCATION_WEIGHT_FLOOR,
    blend_segment_support_scores,
    learned_segment_budget_diagnostics,
    simplify_with_learned_segment_budget_v1,
    simplify_with_learned_segment_budget_v1_with_trace,
)

__all__ = [
    "GEOMETRY_TIE_BREAKER_WEIGHT",
    "LEARNED_SEGMENT_BUDGET_SCHEMA_VERSION",
    "LEARNED_SEGMENT_BUDGET_TRACE_SCHEMA_VERSION",
    "SEGMENT_ALLOCATION_WEIGHT_FLOOR",
    "blend_segment_support_scores",
    "learned_segment_budget_diagnostics",
    "simplify_with_learned_segment_budget_v1",
    "simplify_with_learned_segment_budget_v1_with_trace",
]
