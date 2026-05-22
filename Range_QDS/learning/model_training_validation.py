"""Validation-scoring setup for model training checkpoint selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from config.run_config import ModelConfig
from learning.checkpoint_validation import validation_uniform_score
from learning.model_training_helpers import require_validation_inputs
from workloads.typed_workload import TypedQueryWorkload


@dataclass(frozen=True)
class ValidationScoringPlan:
    selection_metric: str
    validation_score_every: int
    has_validation_score: bool
    validation_trajectories: list[torch.Tensor] | None
    validation_boundaries: list[tuple[int, int]] | None
    validation_workload: TypedQueryWorkload | None
    validation_points_for_score: torch.Tensor | None
    validation_query_cache: Any | None
    validation_uniform_result: tuple[float, dict[str, float]] | None


def _validation_query_cache(
    *,
    validation_trajectories: list[torch.Tensor],
    validation_boundaries: list[tuple[int, int]],
    validation_workload: TypedQueryWorkload,
    validation_points: torch.Tensor | None,
    precomputed_validation_query_cache: Any | None,
) -> tuple[torch.Tensor, Any]:
    validation_points_for_score = (
        validation_points
        if validation_points is not None
        else torch.cat(validation_trajectories, dim=0)
    )
    if precomputed_validation_query_cache is None:
        from scoring.query_cache import ScoringQueryCache

        return (
            validation_points_for_score,
            ScoringQueryCache.for_workload(
                validation_points_for_score,
                validation_boundaries,
                validation_workload.typed_queries,
            ),
        )
    precomputed_validation_query_cache.validate(
        validation_points_for_score,
        validation_boundaries,
        validation_workload.typed_queries,
    )
    return validation_points_for_score, precomputed_validation_query_cache


def _validation_uniform_baseline(
    *,
    validation_trajectories: list[torch.Tensor],
    validation_boundaries: list[tuple[int, int]],
    validation_workload: TypedQueryWorkload,
    validation_workload_map: dict[str, float] | None,
    validation_points_for_score: torch.Tensor | None,
    validation_query_cache: Any | None,
    model_config: ModelConfig,
    run_tag: str,
) -> tuple[float, dict[str, float]]:
    validation_uniform_result = validation_uniform_score(
        trajectories=validation_trajectories,
        boundaries=validation_boundaries,
        workload=validation_workload,
        workload_map=validation_workload_map or {},
        model_config=model_config,
        validation_points=validation_points_for_score,
        query_cache=validation_query_cache,
    )
    uniform_score, uniform_per_type = validation_uniform_result
    print(
        f"  [{run_tag}] validation uniform_score={uniform_score:.6f}  "
        f"range={uniform_per_type.get('range', 0.0):.6f}",
        flush=True,
    )
    return validation_uniform_result


def build_validation_scoring_plan(
    *,
    selection_metric: str,
    validation_score_every: int,
    diag_every: int,
    validation_trajectories: list[torch.Tensor] | None,
    validation_boundaries: list[tuple[int, int]] | None,
    validation_workload: TypedQueryWorkload | None,
    validation_workload_map: dict[str, float] | None,
    validation_points: torch.Tensor | None,
    precomputed_validation_query_cache: Any | None,
    model_config: ModelConfig,
    run_tag: str,
) -> ValidationScoringPlan:
    """Return validated checkpoint-scoring inputs and optional uniform baseline."""
    normalized_metric = str(selection_metric).lower()
    if normalized_metric not in {"loss", "score", "uniform_gap"}:
        raise ValueError("checkpoint_selection_metric must be 'loss', 'score', or 'uniform_gap'.")

    score_every = int(validation_score_every or 0)
    has_validation_score = (
        validation_trajectories is not None
        and validation_boundaries is not None
        and validation_workload is not None
        and validation_workload_map is not None
    )
    if normalized_metric in {"score", "uniform_gap"} and not has_validation_score:
        print(
            f"  [{run_tag}] WARNING: checkpoint_selection_metric={normalized_metric} "
            "requested without validation workload; "
            "falling back to loss selection.",
            flush=True,
        )
        normalized_metric = "loss"
    if normalized_metric in {"score", "uniform_gap"} and score_every <= 0:
        score_every = int(diag_every)

    validation_points_for_score: torch.Tensor | None = None
    validation_query_cache: Any | None = None
    if has_validation_score:
        validation_trajectories, validation_boundaries, validation_workload = (
            require_validation_inputs(
                validation_trajectories,
                validation_boundaries,
                validation_workload,
            )
        )
        validation_points_for_score, validation_query_cache = _validation_query_cache(
            validation_trajectories=validation_trajectories,
            validation_boundaries=validation_boundaries,
            validation_workload=validation_workload,
            validation_points=validation_points,
            precomputed_validation_query_cache=precomputed_validation_query_cache,
        )

    validation_uniform_result: tuple[float, dict[str, float]] | None = None
    if normalized_metric == "uniform_gap" and has_validation_score:
        validation_trajectories, validation_boundaries, validation_workload = (
            require_validation_inputs(
                validation_trajectories,
                validation_boundaries,
                validation_workload,
            )
        )
        validation_uniform_result = _validation_uniform_baseline(
            validation_trajectories=validation_trajectories,
            validation_boundaries=validation_boundaries,
            validation_workload=validation_workload,
            validation_workload_map=validation_workload_map,
            validation_points_for_score=validation_points_for_score,
            validation_query_cache=validation_query_cache,
            model_config=model_config,
            run_tag=run_tag,
        )

    return ValidationScoringPlan(
        selection_metric=normalized_metric,
        validation_score_every=score_every,
        has_validation_score=has_validation_score,
        validation_trajectories=validation_trajectories,
        validation_boundaries=validation_boundaries,
        validation_workload=validation_workload,
        validation_points_for_score=validation_points_for_score,
        validation_query_cache=validation_query_cache,
        validation_uniform_result=validation_uniform_result,
    )
