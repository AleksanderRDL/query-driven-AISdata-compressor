"""Explicit dispatch registry for scalar range target transforms."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from learning.targets.aggregation import (
    aggregate_range_component_retained_frequency_training_labels,
    aggregate_range_continuity_retained_frequency_training_labels,
    aggregate_range_global_budget_retained_frequency_training_labels,
    aggregate_range_marginal_coverage_training_labels,
    aggregate_range_retained_frequency_training_labels,
    aggregate_range_structural_retained_frequency_training_labels,
    range_component_retained_frequency_training_labels,
    range_continuity_retained_frequency_training_labels,
)
from learning.targets.local_swap import (
    range_local_swap_gain_cost_frequency_training_labels,
    range_local_swap_utility_frequency_training_labels,
)
from learning.targets.marginal_coverage import range_marginal_coverage_training_labels
from learning.targets.modes import RANGE_TRAINING_TARGET_MODES
from learning.targets.query_residual import range_query_residual_frequency_training_labels
from learning.targets.query_spine import range_query_spine_frequency_training_labels
from learning.targets.retained_frequency import (
    range_global_budget_retained_frequency_training_labels,
    range_historical_prior_retained_frequency_training_labels,
    range_retained_frequency_training_labels,
)
from learning.targets.set_utility import range_set_utility_frequency_training_labels
from learning.targets.structural import range_structural_retained_frequency_training_labels

RangeTargetResult = tuple[torch.Tensor, torch.Tensor, dict[str, Any]]
RangeTargetCallable = Callable[..., RangeTargetResult]


@dataclass(frozen=True)
class RangeTargetModeSpec:
    """Execution requirements for one scalar range target mode."""

    mode: str
    target_fn: RangeTargetCallable
    aggregate_target_fn: RangeTargetCallable | None = None
    requires_points: bool = False
    requires_typed_queries: bool = False
    requires_component_labels: bool = False
    supports_multiple_replicates: bool = True

    @property
    def supports_frequency_mean(self) -> bool:
        return self.aggregate_target_fn is not None


RANGE_SCALAR_TARGET_MODE_SPECS: dict[str, RangeTargetModeSpec] = {
    "retained_frequency": RangeTargetModeSpec(
        mode="retained_frequency",
        target_fn=range_retained_frequency_training_labels,
        aggregate_target_fn=aggregate_range_retained_frequency_training_labels,
    ),
    "global_budget_retained_frequency": RangeTargetModeSpec(
        mode="global_budget_retained_frequency",
        target_fn=range_global_budget_retained_frequency_training_labels,
        aggregate_target_fn=aggregate_range_global_budget_retained_frequency_training_labels,
    ),
    "historical_prior_retained_frequency": RangeTargetModeSpec(
        mode="historical_prior_retained_frequency",
        target_fn=range_historical_prior_retained_frequency_training_labels,
        requires_points=True,
    ),
    "structural_retained_frequency": RangeTargetModeSpec(
        mode="structural_retained_frequency",
        target_fn=range_structural_retained_frequency_training_labels,
        aggregate_target_fn=aggregate_range_structural_retained_frequency_training_labels,
        requires_points=True,
    ),
    "component_retained_frequency": RangeTargetModeSpec(
        mode="component_retained_frequency",
        target_fn=range_component_retained_frequency_training_labels,
        aggregate_target_fn=aggregate_range_component_retained_frequency_training_labels,
        requires_component_labels=True,
    ),
    "continuity_retained_frequency": RangeTargetModeSpec(
        mode="continuity_retained_frequency",
        target_fn=range_continuity_retained_frequency_training_labels,
        aggregate_target_fn=aggregate_range_continuity_retained_frequency_training_labels,
        requires_component_labels=True,
    ),
    "marginal_coverage_frequency": RangeTargetModeSpec(
        mode="marginal_coverage_frequency",
        target_fn=range_marginal_coverage_training_labels,
        aggregate_target_fn=aggregate_range_marginal_coverage_training_labels,
    ),
    "query_spine_frequency": RangeTargetModeSpec(
        mode="query_spine_frequency",
        target_fn=range_query_spine_frequency_training_labels,
        requires_points=True,
        requires_typed_queries=True,
        supports_multiple_replicates=False,
    ),
    "query_residual_frequency": RangeTargetModeSpec(
        mode="query_residual_frequency",
        target_fn=range_query_residual_frequency_training_labels,
        requires_points=True,
        requires_typed_queries=True,
        supports_multiple_replicates=False,
    ),
    "set_utility_frequency": RangeTargetModeSpec(
        mode="set_utility_frequency",
        target_fn=range_set_utility_frequency_training_labels,
        requires_points=True,
        requires_typed_queries=True,
        supports_multiple_replicates=False,
    ),
    "local_swap_utility_frequency": RangeTargetModeSpec(
        mode="local_swap_utility_frequency",
        target_fn=range_local_swap_utility_frequency_training_labels,
        requires_points=True,
        requires_typed_queries=True,
        supports_multiple_replicates=False,
    ),
    "local_swap_gain_cost_frequency": RangeTargetModeSpec(
        mode="local_swap_gain_cost_frequency",
        target_fn=range_local_swap_gain_cost_frequency_training_labels,
        requires_points=True,
        requires_typed_queries=True,
        supports_multiple_replicates=False,
    ),
}


def range_scalar_target_mode_spec(mode: str) -> RangeTargetModeSpec:
    """Return the scalar transform spec for a configured range target mode."""
    try:
        return RANGE_SCALAR_TARGET_MODE_SPECS[mode]
    except KeyError as exc:
        valid_modes = "', '".join(RANGE_TRAINING_TARGET_MODES)
        raise RuntimeError(
            f"range_training_target_mode must be one of '{valid_modes}'."
        ) from exc
