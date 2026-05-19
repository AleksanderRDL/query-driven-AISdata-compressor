"""Range training target mode registries."""

from __future__ import annotations

from learning.targets.query_local_utility import QUERY_LOCAL_UTILITY_TARGET_MODES

SCALAR_RANGE_TARGET_MODES = frozenset(
    {
        "point_value",
        "retained_frequency",
        "global_budget_retained_frequency",
        "historical_prior_retained_frequency",
        "structural_retained_frequency",
        "component_retained_frequency",
        "continuity_retained_frequency",
        "marginal_coverage_frequency",
        "query_spine_frequency",
        "query_residual_frequency",
        "set_utility_frequency",
        "local_swap_utility_frequency",
        "local_swap_gain_cost_frequency",
    }
)
RANGE_TRAINING_TARGET_MODES = (
    "point_value",
    "retained_frequency",
    "global_budget_retained_frequency",
    "historical_prior_retained_frequency",
    "structural_retained_frequency",
    "component_retained_frequency",
    "continuity_retained_frequency",
    "marginal_coverage_frequency",
    "query_spine_frequency",
    "query_residual_frequency",
    "set_utility_frequency",
    "local_swap_utility_frequency",
    "local_swap_gain_cost_frequency",
    *tuple(sorted(QUERY_LOCAL_UTILITY_TARGET_MODES)),
)
RANGE_TARGET_BALANCE_MODES = ("none", "trajectory_unit_mass")
