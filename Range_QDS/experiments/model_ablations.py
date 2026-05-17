"""Model and prior-field ablation helpers for experiment diagnostics."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, cast

import torch

from simplification.mlqds_scoring import mlqds_simplification_scores


def _reset_module_parameters(module: torch.nn.Module, seed: int) -> torch.nn.Module:
    """Return a deepcopy with reset trainable parameters for untrained-model ablations."""
    clone = copy.deepcopy(module)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        for child in clone.modules():
            reset_parameters = getattr(child, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()
    return clone


def _shuffled_query_prior_field(prior_field: dict[str, Any], seed: int) -> dict[str, Any]:
    """Return a copy of query-prior fields with spatial associations broken."""
    shuffled: dict[str, Any] = {}
    generator = torch.Generator().manual_seed(int(seed))
    for key, value in prior_field.items():
        if isinstance(value, torch.Tensor) and value.numel() > 1:
            flat = value.detach().cpu().flatten()
            order = torch.randperm(int(flat.numel()), generator=generator)
            shuffled[key] = flat[order].reshape(value.shape).to(dtype=value.dtype)
        else:
            shuffled[key] = copy.deepcopy(value)
    shuffled["ablation"] = "shuffled_prior_fields"
    shuffled["contains_eval_queries"] = False
    shuffled["contains_validation_queries"] = False
    return shuffled


def _raw_predictions_without_factorized_head(
    *,
    model: torch.nn.Module,
    head_logits: torch.Tensor,
    disabled_head_name: str,
) -> torch.Tensor:
    """Return final raw predictions with one factorized head neutralized."""
    compose = getattr(model, "final_logit_from_head_logits", None)
    if not callable(compose):
        raise RuntimeError(f"{type(model).__name__} does not expose final_logit_from_head_logits.")
    compose_fn = cast(Callable[..., torch.Tensor], compose)
    model_device = next(model.parameters()).device
    original_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            logits = head_logits.detach().to(model_device).unsqueeze(0)
            pred = compose_fn(
                logits,
                disabled_head_names=(str(disabled_head_name),),
            ).reshape(-1).detach().cpu()
    finally:
        model.train(original_training)
    return pred


def _scores_without_factorized_head(
    *,
    model: torch.nn.Module,
    head_logits: torch.Tensor,
    disabled_head_name: str,
    boundaries: list[tuple[int, int]],
    workload_type: str,
    score_mode: str,
    score_temperature: float,
    rank_confidence_weight: float,
) -> torch.Tensor:
    """Return simplification scores with one factorized head neutralized."""
    pred = _raw_predictions_without_factorized_head(
        model=model,
        head_logits=head_logits,
        disabled_head_name=disabled_head_name,
    )
    return mlqds_simplification_scores(
        pred,
        boundaries,
        workload_type,
        score_mode=score_mode,
        score_temperature=score_temperature,
        rank_confidence_weight=rank_confidence_weight,
    )
