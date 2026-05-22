"""Tests that short training keeps non-collapsed typed predictions. See learning/README.md for details."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from config.run_config import build_run_config
from data_preparation.ais_loader import generate_synthetic_ais_data
from data_preparation.trajectory_dataset import TrajectoryDataset
from learning.checkpoint_validation import _validation_checkpoint_scores, _validation_query_score
from learning.importance_labels import compute_typed_importance_labels
from learning.model_training import train_model
from learning.outputs import TrainingOutputs
from learning.scaler import FeatureScaler
from learning.targets.query_local_utility import QUERY_LOCAL_UTILITY_HEAD_NAMES
from models.historical_prior_qds_model import HistoricalPriorRangeQDSModel
from models.trajectory_qds_model import TrajectoryQDSModel
from scoring.method_scoring import score_range_usefulness, score_retained_mask
from scoring.methods import MLQDSMethod
from workloads.generation.generator import generate_typed_query_workload
from workloads.query_types import NUM_QUERY_TYPES, QUERY_TYPE_ID_RANGE


def test_training_records_validation_selection_score() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=5, n_points_per_ship=24, seed=444)
    train_trajectories = trajectories[:4]
    validation_trajectories = trajectories[4:]
    train_ds = TrajectoryDataset(train_trajectories)
    validation_ds = TrajectoryDataset(validation_trajectories)
    train_boundaries = train_ds.get_trajectory_boundaries()
    validation_boundaries = validation_ds.get_trajectory_boundaries()

    cfg = build_run_config(
        epochs=8,
        n_queries=4,
        workload="range",
        checkpoint_selection_metric="score",
        validation_score_every=2,
        compression_ratio=0.5,
    )
    cfg.model.embed_dim = 16
    cfg.model.num_heads = 2
    cfg.model.num_layers = 1
    cfg.model.query_chunk_size = 8
    cfg.model.window_length = 16
    cfg.model.window_stride = 8
    cfg.model.ranking_pairs_per_type = 8
    cfg.model.train_batch_size = 4
    cfg.model.diagnostic_window_fraction = 1.0

    train_workload = generate_typed_query_workload(
        trajectories=train_trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=101,
    )
    validation_workload = generate_typed_query_workload(
        trajectories=validation_trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=202,
    )

    out = train_model(
        train_trajectories=train_trajectories,
        train_boundaries=train_boundaries,
        workload=train_workload,
        model_config=cfg.model,
        seed=303,
        validation_trajectories=validation_trajectories,
        validation_boundaries=validation_boundaries,
        validation_workload=validation_workload,
        validation_workload_map={"range": 1.0},
    )

    score_rows = [row for row in out.history if "val_selection_score" in row]
    assert score_rows
    assert [int(row["epoch"]) for row in score_rows] == [0, 1, 3, 5, 7]
    assert all(0.0 <= row["val_selection_score"] <= 1.0 for row in score_rows)
    assert all("val_range_point_f1" in row for row in score_rows)
    assert all("val_range_usefulness" in row for row in score_rows)
    assert all("val_query_f1" not in row for row in score_rows)
    assert all(
        "selection_score" not in row for row in out.history if "val_selection_score" not in row
    )
    assert out.best_selection_score == pytest.approx(
        max(row["val_selection_score"] for row in score_rows)
    )


def test_checkpoint_candidate_pool_defers_full_validation() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=5, n_points_per_ship=24, seed=445)
    train_trajectories = trajectories[:4]
    validation_trajectories = trajectories[4:]
    train_boundaries = TrajectoryDataset(train_trajectories).get_trajectory_boundaries()
    validation_boundaries = TrajectoryDataset(validation_trajectories).get_trajectory_boundaries()

    cfg = build_run_config(
        epochs=5,
        n_queries=4,
        workload="range",
        checkpoint_selection_metric="score",
        validation_score_every=1,
        checkpoint_full_score_every=3,
        checkpoint_candidate_pool_size=1,
        compression_ratio=0.5,
    )
    cfg.model.embed_dim = 16
    cfg.model.num_heads = 2
    cfg.model.num_layers = 1
    cfg.model.query_chunk_size = 8
    cfg.model.window_length = 16
    cfg.model.window_stride = 8
    cfg.model.ranking_pairs_per_type = 8
    cfg.model.train_batch_size = 4
    cfg.model.diagnostic_window_fraction = 1.0

    train_workload = generate_typed_query_workload(
        trajectories=train_trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=111,
    )
    validation_workload = generate_typed_query_workload(
        trajectories=validation_trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=222,
    )

    out = train_model(
        train_trajectories=train_trajectories,
        train_boundaries=train_boundaries,
        workload=train_workload,
        model_config=cfg.model,
        seed=333,
        validation_trajectories=validation_trajectories,
        validation_boundaries=validation_boundaries,
        validation_workload=validation_workload,
        validation_workload_map={"range": 1.0},
    )

    candidate_rows = [row for row in out.history if row.get("checkpoint_score_candidate") == 1.0]
    evaluated_rows = [
        row for row in out.history if row.get("checkpoint_candidate_evaluated") == 1.0
    ]

    assert len(candidate_rows) == 5
    assert 1 <= len(evaluated_rows) <= 3
    assert all("val_selection_score" in row for row in evaluated_rows)
    assert all("selection_score" in row for row in evaluated_rows)
    assert all(0.0 <= row["val_selection_score"] <= 1.0 for row in evaluated_rows)
    assert all(
        "selection_score" not in row
        for row in candidate_rows
        if row.get("checkpoint_candidate_evaluated") != 1.0
    )
    assert out.best_selection_score == pytest.approx(
        max(row["val_selection_score"] for row in evaluated_rows)
    )


@pytest.mark.parametrize(
    "score_mode",
    [
        "rank",
        "rank_tie",
        "raw",
        "sigmoid",
        "zscore_sigmoid",
        "rank_confidence",
        "temperature_sigmoid",
    ],
)
def test_validation_query_score_matches_final_mlqds_scoring(
    score_mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectories = generate_synthetic_ais_data(n_ships=2, n_points_per_ship=12, seed=515)
    ds = TrajectoryDataset(trajectories)
    points = ds.get_all_points()
    boundaries = ds.get_trajectory_boundaries()
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=6,
        workload_map={"range": 1.0},
        seed=516,
        range_spatial_fraction=0.40,
        range_time_fraction=0.40,
    )
    cfg = build_run_config(
        compression_ratio=0.40,
        workload="range",
        mlqds_temporal_fraction=0.25,
        mlqds_diversity_bonus=0.0,
        mlqds_hybrid_mode="swap",
        mlqds_score_mode=score_mode,
        mlqds_score_temperature=0.75,
        mlqds_rank_confidence_weight=0.35,
        checkpoint_score_variant="answer",
    )
    predictions = torch.linspace(-1.0, 1.0, steps=points.shape[0])

    model = TrajectoryQDSModel(
        point_dim=7,
        query_dim=int(workload.query_features.shape[1]),
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        query_chunk_size=8,
    )
    scaler = FeatureScaler.fit(points[:, :7], workload.query_features)
    trained = TrainingOutputs(
        model=model,
        scaler=scaler,
        labels=torch.zeros((points.shape[0], 4), dtype=torch.float32),
        labelled_mask=torch.ones((points.shape[0], 4), dtype=torch.bool),
        history=[],
    )

    monkeypatch.setattr(
        "scoring.methods.windowed_predict",
        lambda **_kwargs: predictions.clone(),
    )

    validation_score, validation_per_type = _validation_query_score(
        model=model,
        scaler=scaler,
        trajectories=trajectories,
        boundaries=boundaries,
        workload=workload,
        workload_map={"range": 1.0},
        model_config=cfg.model,
        device=torch.device("cpu"),
        validation_points=points,
        predict_logits_fn=lambda **_kwargs: predictions.clone(),
    )
    retained = MLQDSMethod(
        name="MLQDS",
        trained=trained,
        workload=workload,
        workload_type="range",
        score_mode=cfg.model.mlqds_score_mode,
        score_temperature=cfg.model.mlqds_score_temperature,
        rank_confidence_weight=cfg.model.mlqds_rank_confidence_weight,
        temporal_fraction=cfg.model.mlqds_temporal_fraction,
        diversity_bonus=cfg.model.mlqds_diversity_bonus,
        hybrid_mode=cfg.model.mlqds_hybrid_mode,
        inference_device="cpu",
        inference_batch_size=cfg.model.inference_batch_size,
    ).simplify(points, boundaries, cfg.model.compression_ratio)
    final_f1, final_per_type, _combined_f1, _combined_per_type = score_retained_mask(
        points=points,
        boundaries=boundaries,
        retained_mask=retained,
        typed_queries=workload.typed_queries,
        workload_map={"range": 1.0},
    )

    assert validation_score == pytest.approx(final_f1)
    assert validation_per_type["range"] == pytest.approx(final_per_type["range"])


def test_validation_range_usefulness_matches_final_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    trajectories = generate_synthetic_ais_data(n_ships=2, n_points_per_ship=12, seed=615)
    ds = TrajectoryDataset(trajectories)
    points = ds.get_all_points()
    boundaries = ds.get_trajectory_boundaries()
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=6,
        workload_map={"range": 1.0},
        seed=616,
        range_spatial_fraction=0.40,
        range_time_fraction=0.40,
    )
    cfg = build_run_config(
        compression_ratio=0.40,
        workload="range",
        mlqds_temporal_fraction=0.25,
        mlqds_diversity_bonus=0.0,
        mlqds_hybrid_mode="swap",
        mlqds_score_mode="rank",
        checkpoint_score_variant="range_usefulness",
    )
    predictions = torch.linspace(-1.0, 1.0, steps=points.shape[0])

    model = TrajectoryQDSModel(
        point_dim=7,
        query_dim=int(workload.query_features.shape[1]),
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        query_chunk_size=8,
    )
    scaler = FeatureScaler.fit(points[:, :7], workload.query_features)
    trained = TrainingOutputs(
        model=model,
        scaler=scaler,
        labels=torch.zeros((points.shape[0], 4), dtype=torch.float32),
        labelled_mask=torch.ones((points.shape[0], 4), dtype=torch.bool),
        history=[],
    )

    monkeypatch.setattr(
        "scoring.methods.windowed_predict",
        lambda **_kwargs: predictions.clone(),
    )

    validation_score, validation_per_type = _validation_query_score(
        model=model,
        scaler=scaler,
        trajectories=trajectories,
        boundaries=boundaries,
        workload=workload,
        workload_map={"range": 1.0},
        model_config=cfg.model,
        device=torch.device("cpu"),
        validation_points=points,
        predict_logits_fn=lambda **_kwargs: predictions.clone(),
    )
    retained = MLQDSMethod(
        name="MLQDS",
        trained=trained,
        workload=workload,
        workload_type="range",
        score_mode=cfg.model.mlqds_score_mode,
        temporal_fraction=cfg.model.mlqds_temporal_fraction,
        diversity_bonus=cfg.model.mlqds_diversity_bonus,
        hybrid_mode=cfg.model.mlqds_hybrid_mode,
        inference_device="cpu",
        inference_batch_size=cfg.model.inference_batch_size,
    ).simplify(points, boundaries, cfg.model.compression_ratio)
    audit = score_range_usefulness(
        points=points,
        boundaries=boundaries,
        retained_mask=retained,
        typed_queries=workload.typed_queries,
    )

    assert validation_score == pytest.approx(audit["range_usefulness_score"])
    assert validation_per_type["range"] == pytest.approx(audit["range_usefulness_score"])


def test_validation_selection_passes_segment_head_to_learned_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectories = generate_synthetic_ais_data(n_ships=1, n_points_per_ship=12, seed=617)
    ds = TrajectoryDataset(trajectories)
    points = ds.get_all_points()
    boundaries = ds.get_trajectory_boundaries()
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=2,
        workload_map={"range": 1.0},
        seed=618,
        range_spatial_fraction=0.60,
        range_time_fraction=0.60,
    )
    cfg = build_run_config(
        compression_ratio=0.40,
        workload="range",
        checkpoint_score_variant="answer",
    )
    cfg.model.selector_type = "learned_segment_budget"
    predictions = torch.zeros((points.shape[0],), dtype=torch.float32)
    head_logits = torch.zeros((points.shape[0], 5), dtype=torch.float32)
    head_logits[:, 4] = torch.linspace(-2.0, 2.0, steps=points.shape[0])
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "learning.checkpoint_validation._predict_workload_logits_with_heads",
        lambda **_kwargs: (predictions.clone(), head_logits.clone()),
    )

    def fake_simplify(*_args: object, **kwargs: object) -> torch.Tensor:
        if "segment_scores" not in captured:
            captured["segment_scores"] = kwargs.get("segment_scores")
            captured["points"] = kwargs.get("points")
        retained = torch.zeros((points.shape[0],), dtype=torch.bool)
        retained[:5] = True
        return retained

    monkeypatch.setattr("learning.checkpoint_validation.simplify_mlqds_predictions", fake_simplify)

    _validation_query_score(
        model=TrajectoryQDSModel(
            point_dim=7,
            query_dim=int(workload.query_features.shape[1]),
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            query_chunk_size=8,
        ),
        scaler=FeatureScaler.fit(points[:, :7], workload.query_features),
        trajectories=trajectories,
        boundaries=boundaries,
        workload=workload,
        workload_map={"range": 1.0},
        model_config=cfg.model,
        device=torch.device("cpu"),
        validation_points=points,
    )

    assert torch.allclose(cast(torch.Tensor, captured["segment_scores"]), head_logits[:, 4])
    assert captured["points"] is points


def test_validation_selection_can_blend_length_support_head_for_learned_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectories = generate_synthetic_ais_data(n_ships=1, n_points_per_ship=12, seed=619)
    ds = TrajectoryDataset(trajectories)
    points = ds.get_all_points()
    boundaries = ds.get_trajectory_boundaries()
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=2,
        workload_map={"range": 1.0},
        seed=620,
        range_spatial_fraction=0.60,
        range_time_fraction=0.60,
    )
    cfg = build_run_config(
        compression_ratio=0.40,
        workload="range",
        checkpoint_score_variant="answer",
        learned_segment_length_support_blend_weight=1.0,
    )
    cfg.model.selector_type = "learned_segment_budget"
    predictions = torch.zeros((points.shape[0],), dtype=torch.float32)
    head_logits = torch.zeros(
        (points.shape[0], len(QUERY_LOCAL_UTILITY_HEAD_NAMES)), dtype=torch.float32
    )
    segment_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("segment_budget_target")
    path_idx = tuple(QUERY_LOCAL_UTILITY_HEAD_NAMES).index("path_length_support_target")
    head_logits[:, segment_idx] = torch.linspace(-2.0, 2.0, steps=points.shape[0])
    head_logits[:, path_idx] = torch.linspace(2.0, -2.0, steps=points.shape[0])
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "learning.checkpoint_validation._predict_workload_logits_with_heads",
        lambda **_kwargs: (predictions.clone(), head_logits.clone()),
    )

    def fake_simplify(*_args: object, **kwargs: object) -> torch.Tensor:
        captured["segment_scores"] = kwargs.get("segment_scores")
        retained = torch.zeros((points.shape[0],), dtype=torch.bool)
        retained[:5] = True
        return retained

    monkeypatch.setattr("learning.checkpoint_validation.simplify_mlqds_predictions", fake_simplify)

    _validation_query_score(
        model=TrajectoryQDSModel(
            point_dim=7,
            query_dim=int(workload.query_features.shape[1]),
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            query_chunk_size=8,
        ),
        scaler=FeatureScaler.fit(points[:, :7], workload.query_features),
        trajectories=trajectories,
        boundaries=boundaries,
        workload=workload,
        workload_map={"range": 1.0},
        model_config=cfg.model,
        device=torch.device("cpu"),
        validation_points=points,
    )

    assert torch.allclose(cast(torch.Tensor, captured["segment_scores"]), head_logits[:, path_idx])


def test_validation_checkpoint_scores_report_factorized_causality_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FactorizedValidationModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def final_logit_from_head_logits(
            self,
            head_logits: torch.Tensor,
            *,
            disabled_head_names: tuple[str, ...] = (),
        ) -> torch.Tensor:
            if "conditional_behavior_utility" in disabled_head_names:
                return torch.full(head_logits.shape[:-1], -10.0, device=head_logits.device)
            return torch.full(head_logits.shape[:-1], 10.0, device=head_logits.device)

    trajectories = generate_synthetic_ais_data(n_ships=1, n_points_per_ship=12, seed=917)
    ds = TrajectoryDataset(trajectories)
    points = ds.get_all_points()
    boundaries = ds.get_trajectory_boundaries()
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=2,
        workload_map={"range": 1.0},
        seed=918,
        range_spatial_fraction=0.60,
        range_time_fraction=0.60,
    )
    cfg = build_run_config(
        compression_ratio=0.50,
        workload="range",
        checkpoint_score_variant="query_local_utility",
        validation_global_sanity_penalty_enabled=False,
    )
    cfg.model.selector_type = "learned_segment_budget"
    predictions = torch.ones((points.shape[0],), dtype=torch.float32)
    head_logits = torch.ones(
        (points.shape[0], len(QUERY_LOCAL_UTILITY_HEAD_NAMES)), dtype=torch.float32
    )

    monkeypatch.setattr(
        "learning.checkpoint_validation._predict_workload_logits_with_heads",
        lambda **_kwargs: (predictions.clone(), head_logits.clone()),
    )

    def fake_simplify(
        scores: torch.Tensor,
        *_args: object,
        segment_scores: torch.Tensor | None = None,
        segment_point_scores: torch.Tensor | None = None,
        **_kwargs: object,
    ) -> torch.Tensor:
        keep = 5
        if (
            segment_point_scores is not None
            and int(torch.count_nonzero(segment_point_scores).item()) == 0
        ) or (segment_scores is not None and int(torch.count_nonzero(segment_scores).item()) == 0):
            keep = 2
        elif float(scores.mean().item()) < 0.0:
            keep = 3
        retained = torch.zeros((points.shape[0],), dtype=torch.bool)
        retained[:keep] = True
        return retained

    def fake_range_usefulness(**kwargs: object) -> dict[str, float]:
        retained_mask = cast(torch.Tensor, kwargs["retained_mask"])
        score = float(retained_mask.float().mean().item())
        return {
            "range_usefulness_score": score,
            "range_point_f1": score,
            "range_ship_f1": score,
            "range_ship_coverage": score,
            "range_entry_exit_f1": score,
            "range_crossing_f1": score,
            "range_temporal_coverage": score,
            "range_gap_coverage": score,
            "range_turn_coverage": score,
            "range_shape_score": score,
            "range_query_local_interpolation_fidelity": score,
        }

    monkeypatch.setattr("learning.checkpoint_validation.simplify_mlqds_predictions", fake_simplify)
    monkeypatch.setattr(
        "learning.checkpoint_validation.score_range_usefulness", fake_range_usefulness
    )
    monkeypatch.setattr(
        "learning.checkpoint_validation.query_local_utility_from_range_audit",
        lambda audit, **_kwargs: {"query_local_utility_score": audit["range_usefulness_score"]},
    )

    validation_score, per_type_score, metrics = _validation_checkpoint_scores(
        model=FactorizedValidationModel(),
        scaler=FeatureScaler.fit(points[:, :7], workload.query_features),
        trajectories=trajectories,
        boundaries=boundaries,
        workload=workload,
        workload_map={"range": 1.0},
        model_config=cfg.model,
        device=torch.device("cpu"),
        validation_points=points,
    )

    point_count = float(points.shape[0])
    assert validation_score == pytest.approx(5.0 / point_count)
    assert per_type_score["range"] == pytest.approx(5.0 / point_count)
    assert metrics["checkpoint_causality_ablation_available"] == 1.0
    assert metrics["factorized_target_fit_available"] == 1.0
    assert metrics["factorized_target_fit_used_for_checkpoint_selection"] == 0.0
    assert metrics["head_segment_budget_target_target_fit_available"] == 1.0
    assert metrics["segment_budget_canonical_segment_fit_available"] == 1.0
    assert metrics["no_behavior_query_local_utility"] == pytest.approx(3.0 / point_count)
    assert metrics["no_behavior_query_local_utility_delta"] == pytest.approx(2.0 / point_count)
    assert metrics["no_segment_budget_query_local_utility"] == pytest.approx(2.0 / point_count)
    assert metrics["no_segment_budget_query_local_utility_delta"] == pytest.approx(
        3.0 / point_count
    )


def test_training_accepts_precomputed_importance_labels() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=12, seed=818)
    ds = TrajectoryDataset(trajectories)
    boundaries = ds.get_trajectory_boundaries()
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=819,
    )
    labels, labelled_mask = compute_typed_importance_labels(
        points=ds.get_all_points(),
        boundaries=boundaries,
        typed_queries=workload.typed_queries,
    )

    cfg = build_run_config(
        epochs=1,
        n_queries=4,
        workload="range",
        compression_ratio=0.5,
    )
    cfg.model.embed_dim = 16
    cfg.model.num_heads = 2
    cfg.model.num_layers = 1
    cfg.model.query_chunk_size = 8
    cfg.model.window_length = 8
    cfg.model.window_stride = 4
    cfg.model.ranking_pairs_per_type = 4
    cfg.model.train_batch_size = 2
    cfg.model.diagnostic_window_fraction = 1.0

    out = train_model(
        train_trajectories=trajectories,
        train_boundaries=boundaries,
        workload=workload,
        model_config=cfg.model,
        seed=821,
        precomputed_labels=(labels, labelled_mask),
    )

    assert out.history
    assert out.epochs_trained == 1


def test_historical_prior_training_returns_fitted_prior_and_diagnostics() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=12, seed=822)
    ds = TrajectoryDataset(trajectories)
    boundaries = ds.get_trajectory_boundaries()
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=823,
    )
    cfg = build_run_config(
        epochs=3,
        n_queries=4,
        workload="range",
        model_type="historical_prior",
        historical_prior_k=2,
        compression_ratio=0.5,
    )

    out = train_model(
        train_trajectories=trajectories,
        train_boundaries=boundaries,
        workload=workload,
        model_config=cfg.model,
        seed=824,
    )

    assert isinstance(out.model, HistoricalPriorRangeQDSModel)
    assert out.epochs_trained == 0
    assert out.best_epoch == 0
    assert out.history
    assert out.target_diagnostics["workload_type_id"] == 0
    assert "positive_fraction_t0" in out.history[0]
    assert out.fit_diagnostics["enabled"] is True
    assert out.fit_diagnostics["model_fits_stored_train_support"] is True
    assert out.fit_diagnostics["matched_mlqds_target_recall"] is not None
    assert out.model.historical_features.shape[0] == ds.get_all_points().shape[0]
    scores = out.model(
        out.model.historical_features[:4].unsqueeze(0),
        queries=None,
        query_type_ids=None,
    )
    assert torch.isfinite(scores).all()


def test_historical_prior_training_caps_support_per_trajectory() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=12, seed=825)
    ds = TrajectoryDataset(trajectories)
    boundaries = ds.get_trajectory_boundaries()
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=826,
    )
    points = ds.get_all_points()
    labels = torch.zeros((points.shape[0], NUM_QUERY_TYPES), dtype=torch.float32)
    labelled_mask = torch.zeros_like(labels, dtype=torch.bool)
    labelled_mask[:, QUERY_TYPE_ID_RANGE] = True
    for start, end in boundaries:
        labels[start:end, QUERY_TYPE_ID_RANGE] = torch.linspace(0.0, 1.0, steps=end - start)
    cfg = build_run_config(
        epochs=3,
        n_queries=4,
        workload="range",
        model_type="historical_prior",
        historical_prior_k=2,
        historical_prior_support_ratio=0.25,
        compression_ratio=0.5,
    )

    out = train_model(
        train_trajectories=trajectories,
        train_boundaries=boundaries,
        workload=workload,
        model_config=cfg.model,
        seed=827,
        precomputed_labels=(labels, labelled_mask),
    )

    assert isinstance(out.model, HistoricalPriorRangeQDSModel)
    assert out.model.historical_features.shape[0] == 9
    assert out.target_diagnostics["historical_prior_support_pre_min_count"] == 9
    assert out.target_diagnostics["historical_prior_stored_support_count"] == 9


def test_historical_prior_training_preserves_train_source_ids() -> None:
    trajectories = generate_synthetic_ais_data(n_ships=3, n_points_per_ship=12, seed=828)
    ds = TrajectoryDataset(trajectories)
    boundaries = ds.get_trajectory_boundaries()
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=4,
        workload_map={"range": 1.0},
        seed=829,
    )
    cfg = build_run_config(
        epochs=1,
        n_queries=4,
        workload="range",
        model_type="historical_prior",
        historical_prior_k=2,
        historical_prior_source_aggregation="mean",
        compression_ratio=0.5,
    )
    source_ids = [0, 0, 1]

    out = train_model(
        train_trajectories=trajectories,
        train_boundaries=boundaries,
        workload=workload,
        model_config=cfg.model,
        seed=830,
        train_trajectory_source_ids=source_ids,
    )

    assert isinstance(out.model, HistoricalPriorRangeQDSModel)
    expected = torch.cat(
        [
            torch.full((end - start,), source_id, dtype=torch.long)
            for source_id, (start, end) in zip(source_ids, boundaries, strict=True)
        ]
    )
    assert torch.equal(out.model.historical_source_ids, expected)
    assert out.target_diagnostics["historical_prior_source_aggregation"] == "mean"
    assert out.target_diagnostics["historical_prior_source_count"] == 2


def test_ranking_bce_objective_keeps_rank_signal(synthetic_dataset) -> None:
    """Assert the ranking_bce objective preserves its rank-signal invariant."""
    trajectories, _ = synthetic_dataset
    ds = TrajectoryDataset(trajectories)
    boundaries = ds.get_trajectory_boundaries()

    cfg = build_run_config(epochs=4, n_queries=80, workload="range", loss_objective="ranking_bce")
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=80,
        workload_map={"range": 1.0},
        seed=77,
    )
    out = train_model(
        train_trajectories=trajectories,
        train_boundaries=boundaries,
        workload=workload,
        model_config=cfg.model,
        seed=77,
    )

    diagnostics = [row for row in out.history if "pred_std" in row]
    last = diagnostics[-1]
    assert last["pred_std"] > 0.02

    best_range_tau = max(row["kendall_tau_t0"] for row in diagnostics)
    assert best_range_tau > 0.15


def test_pointwise_bce_objective_trains_on_range_labels(synthetic_dataset) -> None:
    """Assert the direct pointwise objective is accepted by the trainer."""
    trajectories, _ = synthetic_dataset
    ds = TrajectoryDataset(trajectories)
    boundaries = ds.get_trajectory_boundaries()

    cfg = build_run_config(epochs=1, n_queries=24, workload="range", loss_objective="pointwise_bce")
    cfg.model.embed_dim = 16
    cfg.model.num_heads = 2
    cfg.model.num_layers = 1
    cfg.model.query_chunk_size = 8
    cfg.model.window_length = 8
    cfg.model.window_stride = 4
    cfg.model.train_batch_size = 2
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=24,
        workload_map={"range": 1.0},
        seed=78,
    )
    out = train_model(
        train_trajectories=trajectories,
        train_boundaries=boundaries,
        workload=workload,
        model_config=cfg.model,
        seed=78,
    )

    assert out.history
    assert out.history[-1]["loss"] > 0.0
    assert out.fit_diagnostics["enabled"] is True
    assert out.fit_diagnostics["matched_mlqds_target_recall"] is not None


def test_range_coverage_training_keeps_score_spread(synthetic_dataset) -> None:
    """Assert coverage-targeted range training does not converge to constant scores."""
    trajectories, _ = synthetic_dataset
    ds = TrajectoryDataset(trajectories)
    boundaries = ds.get_trajectory_boundaries()

    cfg = build_run_config(
        epochs=4,
        n_queries=60,
        query_coverage=0.30,
        max_queries=160,
        workload="range",
        lr=1e-3,
    )
    workload = generate_typed_query_workload(
        trajectories=trajectories,
        n_queries=60,
        target_coverage=0.30,
        max_queries=160,
        workload_map={"range": 1.0},
        range_spatial_fraction=0.02,
        range_time_fraction=0.04,
        seed=91,
    )
    out = train_model(
        train_trajectories=trajectories,
        train_boundaries=boundaries,
        workload=workload,
        model_config=cfg.model,
        seed=91,
    )

    diagnostics = [row for row in out.history if "pred_std" in row]
    assert max(row["pred_std"] for row in diagnostics) > 0.01
    assert diagnostics[-1]["pred_std"] > 1e-3
