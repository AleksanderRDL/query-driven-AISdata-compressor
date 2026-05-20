"""Selector type identifiers shared across config, scoring, and diagnostics."""

TEMPORAL_HYBRID_SELECTOR_TYPE = "temporal_hybrid"
LEARNED_SEGMENT_BUDGET_SELECTOR_TYPE = "learned_segment_budget"
SELECTOR_TYPE_CHOICES = (
    TEMPORAL_HYBRID_SELECTOR_TYPE,
    LEARNED_SEGMENT_BUDGET_SELECTOR_TYPE,
)

