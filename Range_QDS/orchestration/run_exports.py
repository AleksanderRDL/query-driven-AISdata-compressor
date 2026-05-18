"""Optional single-run export helpers."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import torch

from config.run_config import RunConfig
from learning.outputs import TrainingOutputs
from orchestration.geojson_writers import (
    report_trajectory_length_loss,
    write_queries_geojson,
    write_simplified_csv,
)
from orchestration.mlqds_method_factory import build_mlqds_method
from scoring.metrics import MethodScore
from workloads.typed_workload import TypedQueryWorkload

PhaseLogger = Callable[[str], AbstractContextManager[None]]


def export_eval_queries_geojson(
    *,
    save_queries_dir: str | None,
    eval_workload: TypedQueryWorkload,
    phase: PhaseLogger,
) -> None:
    """Write optional eval query GeoJSON exports."""
    if not save_queries_dir:
        return
    with phase("write-queries-geojson"):
        write_queries_geojson(save_queries_dir, eval_workload.typed_queries)


def export_simplified_eval_csvs(
    *,
    save_simplified_dir: str | None,
    matched: dict[str, MethodScore],
    config: RunConfig,
    trained: TrainingOutputs,
    eval_workload: TypedQueryWorkload,
    eval_workload_map: dict[str, float],
    test_points: torch.Tensor,
    test_boundaries: list[tuple[int, int]],
    test_mmsis: list[int] | None,
    phase: PhaseLogger,
) -> None:
    """Write optional simplified eval CSVs and length-loss summary."""
    if not save_simplified_dir:
        return

    with phase("write-simplified-csv"):
        out_dir = Path(save_simplified_dir)
        eval_mask = matched["MLQDS"].retained_mask
        if eval_mask is None:
            eval_mlqds = build_mlqds_method(
                name="MLQDS",
                trained=trained,
                workload=eval_workload,
                workload_map=eval_workload_map,
                config=config,
                range_geometry_blend=0.0,
                trajectory_mmsis=test_mmsis,
            )
            eval_mask = eval_mlqds.simplify(
                test_points, test_boundaries, config.model.compression_ratio
            )
        write_simplified_csv(
            str(out_dir / "ML_simplified_eval.csv"),
            test_points,
            test_boundaries,
            eval_mask,
            trajectory_mmsis=test_mmsis,
        )
        for ref_name, csv_name in (
            ("uniform", "uniform_simplified_eval.csv"),
            ("DouglasPeucker", "DP_simplified_eval.csv"),
        ):
            ref_eval = matched.get(ref_name)
            ref_mask = ref_eval.retained_mask if ref_eval is not None else None
            if ref_mask is not None:
                write_simplified_csv(
                    str(out_dir / csv_name),
                    test_points,
                    test_boundaries,
                    ref_mask,
                    trajectory_mmsis=test_mmsis,
                )

    with phase("trajectory-length-loss"):
        report_trajectory_length_loss(
            test_points,
            test_boundaries,
            eval_mask,
            top_k=25,
            trajectory_mmsis=test_mmsis,
        )
