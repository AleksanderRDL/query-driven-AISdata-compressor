"""Focused tests for workload-blind retained-mask freezing."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import pytest
import torch

from config.run_config import build_run_config, derive_seed_bundle
from learning.outputs import TrainingOutputs
from learning.targets.query_local_utility import QUERY_LOCAL_UTILITY_HEAD_NAMES
from orchestration.retained_mask_ablation_stage import freeze_retained_mask_ablations
from orchestration.retained_mask_stage import freeze_workload_blind_retained_masks
from orchestration.selector_diagnostics import (
    factorized_score_component_vectors_from_logits,
    query_free_retained_removal_teacher_proxy_vectors,
)
from scoring.methods import FrozenMaskMethod
from workloads.query_types import pad_query_features
from workloads.typed_workload import TypedQueryWorkload


@contextmanager
def _noop_phase(_name: str) -> Iterator[None]:
    yield


class _CachingMethod:
    def __init__(self, name: str, *, cache_offset: float = 0.0) -> None:
        self.name = name
        self.cache_offset = float(cache_offset)
        self.calls: list[float] = []
        self.latency_ms = 0.0

    def simplify(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        compression_ratio: float,
    ) -> torch.Tensor:
        del boundaries
        point_count = int(points.shape[0])
        budget = max(1, min(point_count, round(point_count * float(compression_ratio))))
        mask = torch.zeros((point_count,), dtype=torch.bool)
        mask[:budget] = True
        base = torch.arange(point_count, dtype=torch.float32) + self.cache_offset
        self.calls.append(float(compression_ratio))
        self._score_cache = base
        self._raw_pred_cache = base + 0.1
        self._head_logit_cache = torch.stack([base + 0.1 * idx for idx in range(6)], dim=1)
        self._segment_score_cache = base + 0.3
        self._path_length_support_score_cache = base + 0.4
        self._selector_segment_score_cache = base + 0.5
        self._selector_segment_score_source_cache = "point_score_top20_mean"
        return mask


def _workload() -> TypedQueryWorkload:
    typed_queries = [
        {
            "type": "range",
            "params": {
                "lat_min": 0.0,
                "lat_max": 1.0,
                "lon_min": 0.0,
                "lon_max": 1.0,
                "t_start": 0.0,
                "t_end": 3.0,
            },
        }
    ]
    query_features, type_ids = pad_query_features(typed_queries)
    return TypedQueryWorkload(
        query_features=query_features,
        typed_queries=typed_queries,
        type_ids=type_ids,
    )


def _points() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.1, 0.1],
            [2.0, 0.2, 0.2],
            [3.0, 0.3, 0.3],
        ],
        dtype=torch.float32,
    )


def _trained_stub() -> TrainingOutputs:
    return TrainingOutputs(
        model=torch.nn.Linear(1, 1),
        scaler=cast(Any, object()),
        labels=torch.zeros((4, 1), dtype=torch.float32),
        labelled_mask=torch.ones((4, 1), dtype=torch.bool),
        history=[],
        feature_context={},
    )


def test_retained_mask_freezing_is_noop_for_query_aware_runs() -> None:
    method = _CachingMethod("MLQDS")

    outputs = freeze_workload_blind_retained_masks(
        methods=[method],
        retention_methods=[method],
        workload_blind_eval=False,
        audit_ratios=[0.25],
        config=build_run_config(),
        trained=_trained_stub(),
        eval_workload=_workload(),
        eval_workload_map={"range": 1.0},
        test_mmsis=None,
        test_points=_points(),
        test_boundaries=[(0, 4)],
        seeds=derive_seed_bundle(7),
        phase=_noop_phase,
    )

    assert outputs.methods == [method]
    assert outputs.frozen_primary_masks == {}
    assert outputs.frozen_audit_methods_by_ratio == {}
    assert outputs.causality_ablation_methods == []
    assert method.calls == []


def test_retained_mask_freezing_captures_primary_caches_and_audit_masks() -> None:
    mlqds = _CachingMethod("MLQDS", cache_offset=10.0)
    uniform = _CachingMethod("uniform", cache_offset=20.0)
    points = _points()

    outputs = freeze_workload_blind_retained_masks(
        methods=[mlqds, uniform],
        retention_methods=[mlqds, uniform],
        workload_blind_eval=True,
        audit_ratios=[0.50, 0.25],
        config=build_run_config(
            model_type="workload_blind_range",
            selector_type="temporal_hybrid",
            compression_ratio=0.50,
        ),
        trained=_trained_stub(),
        eval_workload=_workload(),
        eval_workload_map={"range": 1.0},
        test_mmsis=None,
        test_points=points,
        test_boundaries=[(0, 4)],
        seeds=derive_seed_bundle(7),
        phase=_noop_phase,
    )

    assert all(isinstance(method, FrozenMaskMethod) for method in outputs.methods)
    assert [method.name for method in outputs.methods] == ["MLQDS", "uniform"]
    assert mlqds.calls == [0.50, 0.25]
    assert uniform.calls == [0.50, 0.25]
    assert outputs.frozen_primary_masks["MLQDS"].tolist() == [True, True, False, False]
    assert outputs.frozen_primary_scores["MLQDS"].tolist() == [10.0, 11.0, 12.0, 13.0]
    assert outputs.frozen_primary_raw_preds["MLQDS"].tolist() == pytest.approx(
        [10.1, 11.1, 12.1, 13.1]
    )
    assert outputs.frozen_primary_segment_scores["MLQDS"].tolist() == pytest.approx(
        [
            10.3,
            11.3,
            12.3,
            13.3,
        ]
    )
    assert outputs.frozen_primary_path_length_support_scores["MLQDS"].tolist() == pytest.approx(
        [
            10.4,
            11.4,
            12.4,
            13.4,
        ]
    )
    assert outputs.frozen_primary_selector_segment_scores["MLQDS"].tolist() == pytest.approx(
        [
            10.5,
            11.5,
            12.5,
            13.5,
        ]
    )
    assert "0.2500" in outputs.frozen_audit_methods_by_ratio
    audit_methods = outputs.frozen_audit_methods_by_ratio["0.2500"]
    assert [method.name for method in audit_methods] == ["MLQDS", "uniform"]
    assert isinstance(audit_methods[0], FrozenMaskMethod)
    assert audit_methods[0].retained_mask.tolist() == [True, False, False, False]
    timing = outputs.freeze_timing_diagnostics
    assert timing["available"] is True
    assert timing["diagnostic_only"] is True
    assert timing["query_free"] is True
    assert timing["primary_method_simplify_seconds"]["MLQDS"] >= 0.0
    assert timing["primary_method_simplify_seconds"]["uniform"] >= 0.0
    assert timing["audit_method_simplify_seconds"]["0.2500"]["MLQDS"] >= 0.0
    assert timing["audit_method_simplify_seconds"]["0.2500"]["uniform"] >= 0.0
    assert timing["total_seconds"] >= 0.0


def test_retained_mask_freezing_captures_learned_selector_trace() -> None:
    mlqds = _CachingMethod("MLQDS")

    outputs = freeze_workload_blind_retained_masks(
        methods=[mlqds],
        retention_methods=[mlqds],
        workload_blind_eval=True,
        audit_ratios=[],
        config=build_run_config(
            model_type="workload_blind_range",
            selector_type="learned_segment_budget",
            compression_ratio=0.75,
        ),
        trained=_trained_stub(),
        eval_workload=_workload(),
        eval_workload_map={"range": 1.0},
        test_mmsis=None,
        test_points=_points(),
        test_boundaries=[(0, 4)],
        seeds=derive_seed_bundle(7),
        phase=_noop_phase,
    )

    assert outputs.primary_selector_trace is not None
    assert isinstance(outputs.primary_selector_trace["retained_mask_matches_frozen_primary"], bool)
    assert outputs.primary_selector_trace["frozen_primary_retained_count"] == 3
    assert outputs.primary_selector_trace["segment_score_source"] == "point_score_top20_mean"
    timing = outputs.primary_selector_trace["retained_mask_freeze_timing"]
    assert timing["available"] is True
    assert timing["primary_method_simplify_seconds"]["MLQDS"] >= 0.0
    assert timing["substage_seconds"]["selector_trace_reconstruction"] >= 0.0
    assert timing["substage_seconds"]["retained_marginal_alignment"] >= 0.0
    assert timing["substage_seconds"]["score_protected_length_diagnostics"] >= 0.0
    assert timing["substage_seconds"]["query_free_ablation_freeze"] >= 0.0
    assert timing["ablation_freeze_timing"]["method_count"] == len(
        outputs.causality_ablation_methods
    )
    assert timing["total_seconds"] >= 0.0
    marginal = outputs.primary_selector_trace["retained_decision_marginal_query_local_utility_alignment"]
    assert marginal["available"] is True
    assert marginal["diagnostic_only"] is True
    assert marginal["exact_query_local_utility_marginals"] is True
    assert marginal["performance_mode"] == "exact_cached_query_support"
    assert marginal["elapsed_seconds"] >= 0.0
    assert marginal["query_cache_created"] is True
    assert marginal["query_cache_provided"] is False
    assert marginal["masks_frozen_before_query_scoring_required"] is True
    assert marginal["retained_count"] == 3
    assert marginal["score_fields_available"] == {
        "raw_score": True,
        "selector_score": True,
        "segment_score": True,
    }
    assert marginal["score_component_fields_available"]["factorized_composed_score"] is True
    assert (
        marginal["score_component_fields_available"]["head_probability_query_hit_probability"]
        is True
    )
    assert marginal["overall"]["score_component_alignment"]["factorized_composed_score"][
        "available"
    ] is True
    assert marginal["context_fields_available"]["query_free_teacher_proxies"] == {
        "query_free_endpoint_or_path_support": True,
        "query_free_endpoint_support": True,
        "query_free_path_length_support_target": True,
    }
    assert marginal["context_fields_available"]["selector_segment_context"] is True
    assert marginal["query_free_teacher_proxy_guard_coupling_summary"]["available"] is True
    assert marginal["rows"][0]["score_components"]["factorized_composed_score"] >= 0.0
    assert any(row["selector_segment_context"] is not None for row in marginal["rows"])
    assert marginal["candidate_count"] > 0
    assert outputs.causality_ablation_methods


def test_factorized_score_component_vectors_from_logits_reports_score_terms() -> None:
    logits = torch.tensor(
        [
            [0.0, 1.0, -1.0, 0.5, 0.2, -0.3],
            [2.0, -1.0, 0.0, -0.5, 0.7, 0.1],
        ],
        dtype=torch.float32,
    )

    components = factorized_score_component_vectors_from_logits(logits)

    assert sorted(components) == [
        "factorized_behavior_branch",
        "factorized_boundary_bonus",
        "factorized_composed_score",
        "factorized_pre_replacement_score",
        "factorized_query_hit_branch",
        "factorized_replacement_modulated_score",
        "factorized_replacement_multiplier",
        "head_logit_boundary_event_utility",
        "head_logit_conditional_behavior_utility",
        "head_logit_path_length_support_target",
        "head_logit_query_hit_probability",
        "head_logit_replacement_representative_value",
        "head_logit_segment_budget_target",
        "head_probability_boundary_event_utility",
        "head_probability_conditional_behavior_utility",
        "head_probability_path_length_support_target",
        "head_probability_query_hit_probability",
        "head_probability_replacement_representative_value",
        "head_probability_segment_budget_target",
    ]
    assert components["factorized_composed_score"].shape == (2,)
    assert torch.all(components["factorized_composed_score"] >= 0.0)


def test_query_free_retained_removal_teacher_proxy_vectors_report_path_and_endpoints() -> None:
    points = torch.zeros((5, 5), dtype=torch.float32)
    points[:, 0] = torch.arange(5, dtype=torch.float32)
    points[:, 1] = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    points[:, 2] = torch.tensor([0.0, 0.0, 2.0, 0.0, 0.0])

    proxies = query_free_retained_removal_teacher_proxy_vectors(
        points,
        [(0, 5)],
        segment_size=2,
    )

    assert sorted(proxies) == [
        "query_free_endpoint_or_path_support",
        "query_free_endpoint_support",
        "query_free_path_length_support_target",
    ]
    assert torch.equal(
        proxies["query_free_endpoint_support"],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0]),
    )
    assert torch.all(proxies["query_free_path_length_support_target"] >= 0.0)
    assert torch.all(proxies["query_free_endpoint_or_path_support"] >= proxies["query_free_endpoint_support"])


def test_retained_mask_ablations_freeze_pre_repair_and_shuffled_scores() -> None:
    trace: dict[str, Any] = {
        "pre_repair_retained_mask": {
            "available": True,
            "indices": [0, 2],
            "retained_count": 2,
        }
    }
    primary_scores = torch.tensor([0.1, 0.4, 0.3, 0.2], dtype=torch.float32)

    outputs = freeze_retained_mask_ablations(
        config=build_run_config(
            model_type="workload_blind_range",
            selector_type="learned_segment_budget",
            compression_ratio=0.50,
        ),
        trained=_trained_stub(),
        eval_workload=_workload(),
        eval_workload_map={"range": 1.0},
        test_mmsis=None,
        test_points=_points(),
        test_boundaries=[(0, 4)],
        seeds=derive_seed_bundle(7),
        primary_selector_trace=trace,
        frozen_primary_masks={"MLQDS": torch.tensor([True, True, False, False])},
        primary_scores=primary_scores,
        primary_raw_preds=primary_scores,
        primary_segment_scores=None,
        primary_path_length_support_scores=None,
        primary_selector_segment_scores=None,
        primary_head_logits=None,
    )

    method_names = {method.name for method in outputs.causality_ablation_methods}
    assert "MLQDS_pre_repair_allocation_diagnostic" in method_names
    assert "MLQDS_shuffled_scores" in method_names
    timing = outputs.freeze_timing_diagnostics
    assert timing["available"] is True
    assert timing["method_count"] == len(outputs.causality_ablation_methods)
    assert timing["failure_count"] == len(outputs.causal_ablation_freeze_failures)
    assert timing["substage_seconds"]["pre_repair_from_trace"] >= 0.0
    assert timing["substage_seconds"]["shuffled_scores"] >= 0.0
    assert timing["substage_seconds"]["untrained_model"] >= 0.0
    assert timing["total_seconds"] >= 0.0
    assert outputs.primary_selector_trace["retained_mask_ablation_freeze_timing"] is timing
    assert outputs.primary_selector_trace["pre_repair_frozen_method_diagnostic"] == {
        "available": True,
        "diagnostic_only": True,
        "query_free": True,
        "method_name": "MLQDS_pre_repair_allocation_diagnostic",
        "source": "selector_trace.pre_repair_retained_mask.indices",
        "retained_count": 2,
    }


def test_retained_mask_ablations_include_behavior_segment_score_diagnostics() -> None:
    primary_scores = torch.tensor([0.1, 0.4, 0.3, 0.2], dtype=torch.float32)
    head_logits = torch.zeros(
        (4, len(QUERY_LOCAL_UTILITY_HEAD_NAMES)),
        dtype=torch.float32,
    )
    behavior_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("conditional_behavior_utility")
    head_logits[:, behavior_idx] = torch.tensor([0.0, 1.0, 0.2, 0.8], dtype=torch.float32)
    segment_scores = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)

    outputs = freeze_retained_mask_ablations(
        config=build_run_config(
            model_type="workload_blind_range",
            selector_type="learned_segment_budget",
            compression_ratio=0.50,
        ),
        trained=_trained_stub(),
        eval_workload=_workload(),
        eval_workload_map={"range": 1.0},
        test_mmsis=None,
        test_points=_points(),
        test_boundaries=[(0, 4)],
        seeds=derive_seed_bundle(7),
        primary_selector_trace={},
        frozen_primary_masks={"MLQDS": torch.tensor([True, True, False, False])},
        primary_scores=primary_scores,
        primary_raw_preds=primary_scores,
        primary_segment_scores=segment_scores,
        primary_path_length_support_scores=None,
        primary_selector_segment_scores=segment_scores + 0.5,
        primary_head_logits=head_logits,
    )

    method_names = {method.name for method in outputs.causality_ablation_methods}
    assert "MLQDS_behavior_utility_segment_head_diagnostic" in method_names
    assert "MLQDS_behavior_utility_allocation_only_diagnostic" in method_names
    assert "MLQDS_uniform_segment_allocation_only_diagnostic" in method_names
    diagnostics = outputs.head_ablation_sensitivity_diagnostics
    assert diagnostics["MLQDS_uniform_segment_allocation_only_diagnostic"][
        "allocation_score_source"
    ] == "uniform_segment_scores_no_length_support"
    assert diagnostics["MLQDS_behavior_utility_segment_head_diagnostic"][
        "replacement_head_name"
    ] == "conditional_behavior_utility"
    assert diagnostics["MLQDS_behavior_utility_allocation_only_diagnostic"][
        "ablation_mode"
    ] == "conditional_behavior_utility_allocation_only"
