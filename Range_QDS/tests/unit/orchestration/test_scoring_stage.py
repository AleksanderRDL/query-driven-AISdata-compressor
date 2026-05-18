"""Focused tests for the single-run scoring stage."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any, cast

import pytest
import torch

from config.run_config import build_run_config, derive_seed_bundle
from learning.outputs import TrainingOutputs
from orchestration.range_runtime_cache import RangeRuntimeCache
from orchestration.scoring_stage import run_scoring_stage
from workloads.query_types import pad_query_features
from workloads.typed_workload import TypedQueryWorkload


class _StaticMaskMethod:
    def __init__(self, name: str, retained_mask: list[bool]) -> None:
        self.name = name
        self._retained_mask = torch.tensor(retained_mask, dtype=torch.bool)
        self.latency_ms = 0.0

    def simplify(
        self,
        points: torch.Tensor,
        boundaries: list[tuple[int, int]],
        compression_ratio: float,
    ) -> torch.Tensor:
        del boundaries, compression_ratio
        if int(points.shape[0]) != int(self._retained_mask.numel()):
            raise ValueError("test mask must match point count")
        return self._retained_mask.clone()


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


def _range_workload() -> TypedQueryWorkload:
    typed_queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
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


def _trained_stub() -> TrainingOutputs:
    return TrainingOutputs(
        model=torch.nn.Linear(1, 1),
        scaler=cast(Any, object()),
        labels=torch.zeros((4, 1), dtype=torch.float32),
        labelled_mask=torch.ones((4, 1), dtype=torch.bool),
        history=[],
        feature_context={},
    )


def test_scoring_stage_runs_core_matched_audit_and_shift() -> None:
    phase_names: list[str] = []

    @contextmanager
    def phase(name: str) -> Iterator[None]:
        phase_names.append(name)
        yield

    config = build_run_config(
        compression_ratio=0.50,
        final_metrics_mode="core",
    )
    workload = _range_workload()
    mlqds = _StaticMaskMethod("MLQDS", [True, True, False, True])
    uniform = _StaticMaskMethod("uniform", [True, False, True, True])

    outputs = run_scoring_stage(
        config=config,
        seeds=derive_seed_bundle(7),
        trained=_trained_stub(),
        methods=[mlqds, uniform],
        retention_methods=[mlqds, uniform],
        workload_blind_eval=False,
        audit_ratios=[0.50],
        frozen_primary_masks={},
        frozen_audit_methods_by_ratio={},
        frozen_primary_scores={},
        frozen_primary_head_logits={},
        frozen_primary_segment_scores={},
        frozen_primary_selector_segment_scores={},
        causality_ablation_methods=[],
        train_workload=workload,
        train_workload_map={"range": 1.0},
        eval_workload=workload,
        eval_workload_map={"range": 1.0},
        test_points=_points(),
        test_boundaries=[(0, 4)],
        test_mmsis=None,
        range_runtime_caches={"eval": RangeRuntimeCache()},
        save_masks=False,
        mlqds_range_geometry_blend=0.0,
        phase=phase,
    )

    assert set(outputs.matched) == {"MLQDS", "uniform"}
    assert outputs.matched["MLQDS"].retained_mask is not None
    assert outputs.learned_fill_diagnostics == {"MLQDS": outputs.matched["MLQDS"]}
    assert outputs.learned_fill_table == ""
    assert outputs.range_compression_audit.keys() == {"0.5000"}
    assert set(outputs.range_compression_audit["0.5000"]) == {"MLQDS", "uniform"}
    assert outputs.shift_pairs == {
        "range=1.0": {"range=1.0": pytest.approx(outputs.matched["MLQDS"].aggregate_f1)}
    }
    assert outputs.run_oracle_baseline is False
    assert outputs.segment_oracle_allocation_audit == {
        "available": False,
        "reason": "not_run",
    }
    assert "evaluate-matched" in phase_names
    assert "range-compression-audit" in phase_names
    assert "evaluate-shift" in phase_names


def test_scoring_stage_rejects_unknown_final_metrics_mode() -> None:
    config = build_run_config(final_metrics_mode="wide-open")
    workload = _range_workload()

    with pytest.raises(ValueError, match="final_metrics_mode"):
        run_scoring_stage(
            config=config,
            seeds=derive_seed_bundle(7),
            trained=_trained_stub(),
            methods=[_StaticMaskMethod("MLQDS", [True, True, False, True])],
            retention_methods=[],
            workload_blind_eval=False,
            audit_ratios=[],
            frozen_primary_masks={},
            frozen_audit_methods_by_ratio={},
            frozen_primary_scores={},
            frozen_primary_head_logits={},
            frozen_primary_segment_scores={},
            frozen_primary_selector_segment_scores={},
            causality_ablation_methods=[],
            train_workload=workload,
            train_workload_map={"range": 1.0},
            eval_workload=workload,
            eval_workload_map={"range": 1.0},
            test_points=_points(),
            test_boundaries=[(0, 4)],
            test_mmsis=None,
            range_runtime_caches={"eval": RangeRuntimeCache()},
            save_masks=False,
            mlqds_range_geometry_blend=0.0,
            phase=lambda _name: nullcontext(),
        )
