"""Public sensitivity diagnostic exports for causality reporting."""

from __future__ import annotations

from orchestration.causality_marginal_paths import (
    marginal_row_delta_path_diagnostics,
    score_rank_margin_boundary_diagnostics,
)
from orchestration.causality_prior_sensitivity import (
    PRIOR_ABLATION_DIAGNOSTIC_CHAIN,
    PRIOR_ABLATION_SCORE_OUTPUT_SEMANTICS,
    head_ablation_sensitivity,
    head_output_sensitivity,
    model_prior_feature_sensitivity,
    prior_ablation_sensitivity_from_tensors,
    prior_ablation_sensitivity_payload,
    prior_feature_sample_sensitivity,
    prior_sample_gate_failures,
    training_outputs_with_query_prior_field,
)
from orchestration.causality_score_sensitivity import (
    retained_mask_comparison,
    score_ablation_sensitivity,
)

__all__ = [
    "PRIOR_ABLATION_DIAGNOSTIC_CHAIN",
    "PRIOR_ABLATION_SCORE_OUTPUT_SEMANTICS",
    "head_ablation_sensitivity",
    "head_output_sensitivity",
    "marginal_row_delta_path_diagnostics",
    "model_prior_feature_sensitivity",
    "prior_ablation_sensitivity_from_tensors",
    "prior_ablation_sensitivity_payload",
    "prior_feature_sample_sensitivity",
    "prior_sample_gate_failures",
    "retained_mask_comparison",
    "score_ablation_sensitivity",
    "score_rank_margin_boundary_diagnostics",
    "training_outputs_with_query_prior_field",
]
