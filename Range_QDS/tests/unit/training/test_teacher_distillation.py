"""Tests for range teacher-student distillation."""

from __future__ import annotations

import torch

from config.experiment_config import ModelConfig
from models.trajectory_qds_model import TrajectoryQDSModel
from training.scaler import FeatureScaler
from training.teacher_distillation import build_range_teacher_config, distill_range_teacher_labels
from training.training_outputs import TrainingOutputs
from workloads.query_types import pad_query_features
from workloads.typed_workload import TypedQueryWorkload


def _toy_points() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 55.00, 12.00, 1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 55.01, 12.01, 1.0, 5.0, 0.0, 0.0, 0.1],
            [2.0, 55.02, 12.02, 1.0, 10.0, 0.0, 0.0, 0.2],
            [3.0, 55.03, 12.03, 1.0, 15.0, 0.0, 0.0, 0.3],
            [4.0, 55.04, 12.04, 1.0, 20.0, 0.0, 0.0, 0.2],
            [5.0, 55.05, 12.05, 1.0, 25.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )


def _toy_workload() -> TypedQueryWorkload:
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": 54.99,
                "lat_max": 55.06,
                "lon_min": 11.99,
                "lon_max": 12.06,
                "t_start": 0.0,
                "t_end": 6.0,
            },
        }
    ]
    query_features, type_ids = pad_query_features(queries)
    return TypedQueryWorkload(
        query_features=query_features, typed_queries=queries, type_ids=type_ids
    )


def test_build_range_teacher_config_uses_query_aware_loss_selected_teacher() -> None:
    student = ModelConfig(
        model_type="workload_blind_range",
        range_teacher_distillation_mode="retained_frequency",
        range_teacher_epochs=3,
        checkpoint_selection_metric="uniform_gap",
        mlqds_range_geometry_blend=0.4,
    )

    teacher = build_range_teacher_config(student)

    assert teacher.model_type == "range_aware"
    assert teacher.epochs == 3
    assert teacher.checkpoint_selection_metric == "loss"
    assert teacher.validation_score_every == 0
    assert teacher.mlqds_range_geometry_blend == 0.0


def test_distill_range_teacher_labels_retained_frequency_shape_and_bounds() -> None:
    torch.manual_seed(7)
    points = _toy_points()
    workload = _toy_workload()
    model = TrajectoryQDSModel(
        point_dim=7,
        query_dim=workload.query_features.shape[1],
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    scaler = FeatureScaler.fit(points[:, :7], workload.query_features)
    teacher = TrainingOutputs(
        model=model,
        scaler=scaler,
        labels=torch.zeros((points.shape[0], 1), dtype=torch.float32),
        labelled_mask=torch.zeros((points.shape[0], 1), dtype=torch.bool),
        history=[],
        epochs_trained=2,
    )
    config = ModelConfig(
        range_teacher_distillation_mode="retained_frequency",
        budget_loss_ratios=[0.33, 0.50],
        window_length=8,
        window_stride=4,
        inference_batch_size=2,
        mlqds_temporal_fraction=0.0,
    )

    (labels, labelled_mask), diagnostics = distill_range_teacher_labels(
        teacher=teacher,
        teacher_model_type="baseline",
        points=points,
        boundaries=[(0, points.shape[0])],
        workload=workload,
        model_config=config,
    )

    assert labels.shape == (points.shape[0], 1)
    assert labelled_mask.shape == labels.shape
    assert labelled_mask[:, 0].all()
    assert float(labels.min().item()) >= 0.0
    assert float(labels.max().item()) <= 1.0
    assert float(labels[:, 0].sum().item()) > 0.0
    assert diagnostics["enabled"] is True
    assert diagnostics["mode"] == "retained_frequency"
    assert diagnostics["teacher_epochs_trained"] == 2
