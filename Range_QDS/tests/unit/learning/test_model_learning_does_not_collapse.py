"""Tests that short training keeps non-collapsed typed predictions. See learning/README.md for details."""

from __future__ import annotations

import pytest
import torch

from config.run_config import build_run_config
from learning.checkpoint_selection import (
    selection_score as _selection_score,
)
from learning.checkpoint_selection import (
    uniform_gap_selection_score as _uniform_gap_selection_score,
)
from learning.checkpoint_selection import (
    validation_score_selection_score as _validation_score_selection_score,
)
from learning.fit_diagnostics import train_target_fit_diagnostics
from learning.losses import (
    _balanced_pointwise_loss,
    _balanced_pointwise_loss_rows,
    _budget_topk_recall_loss,
    _budget_topk_recall_loss_rows,
    _budget_topk_temporal_residual_loss,
    _budget_topk_temporal_residual_loss_rows,
    _effective_budget_loss_ratios,
    _ranking_loss_for_type,
)
from learning.model_setup import _single_active_type_id
from learning.supervised_windows import _filter_supervised_windows
from learning.targets.common import _apply_temporal_residual_labels
from learning.trajectory_batching import build_trajectory_windows


def test_selection_score_penalizes_collapsed_predictions() -> None:
    """Assert model selection does not prefer collapsed output solely because tau is nonnegative."""
    assert _selection_score(avg_tau=0.0, pred_std=0.0) < _selection_score(
        avg_tau=-0.05, pred_std=0.01
    )


def test_temporal_residual_budget_ratios_match_learned_fill_budget() -> None:
    cfg = build_run_config(
        compression_ratio=0.05,
        budget_loss_ratios=[0.01, 0.02, 0.05, 0.10],
        mlqds_temporal_fraction=0.75,
    )

    ratios = _effective_budget_loss_ratios(cfg.model, "temporal")

    assert ratios == pytest.approx(
        (
            0.0025188917,
            0.0050761421,
            0.0129870130,
            0.0270270270,
        )
    )
    assert _effective_budget_loss_ratios(cfg.model, "none") == pytest.approx(
        (0.01, 0.02, 0.05, 0.10)
    )

    stratified_cfg = build_run_config(
        compression_ratio=0.05,
        budget_loss_ratios=[0.01, 0.02, 0.05, 0.10],
        mlqds_hybrid_mode="stratified",
        mlqds_temporal_fraction=0.75,
    )
    assert _effective_budget_loss_ratios(stratified_cfg.model, "temporal") == pytest.approx(
        (0.01, 0.02, 0.05, 0.10)
    )
    global_cfg = build_run_config(
        compression_ratio=0.05,
        budget_loss_ratios=[0.01, 0.02, 0.05, 0.10],
        mlqds_hybrid_mode="global_budget",
        mlqds_temporal_fraction=0.75,
    )
    assert _effective_budget_loss_ratios(global_cfg.model, "temporal") == pytest.approx(
        (0.01, 0.02, 0.05, 0.10)
    )


def test_selection_score_uses_loss_before_tau_proxy() -> None:
    """Assert checkpoint selection does not restore a worse-loss epoch solely from noisy tau."""
    proxy_best = _selection_score(avg_tau=0.9, pred_std=0.1, loss=0.20)
    lower_loss = _selection_score(avg_tau=-0.1, pred_std=0.1, loss=0.10)

    assert lower_loss > proxy_best


def test_train_target_fit_diagnostics_reports_budget_target_recall() -> None:
    """Assert train-target fit diagnostics compare learned masks to uniform masks."""
    target = torch.tensor([0.0, 0.1, 1.0, 0.9, 0.2, 0.0], dtype=torch.float32)
    predictions = target.clone()
    labelled_mask = torch.ones_like(target, dtype=torch.bool)
    cfg = build_run_config(
        compression_ratio=0.5,
        budget_loss_ratios=[0.5],
        workload="range",
        mlqds_hybrid_mode="fill",
        mlqds_temporal_fraction=0.0,
    )

    diagnostics = train_target_fit_diagnostics(
        predictions=predictions,
        target=target,
        labelled_mask=labelled_mask,
        boundaries=[(0, int(target.numel()))],
        model_config=cfg.model,
        workload_type="range",
        seed=12,
    )

    assert diagnostics["enabled"] is True
    assert diagnostics["score_target_kendall_tau"] > 0.9
    assert diagnostics["matched_mlqds_target_recall"] == pytest.approx(1.0)
    assert diagnostics["matched_mlqds_vs_uniform_target_recall"] > 0.0
    assert (
        diagnostics["budget_rows"][0]["mlqds_target_mass"]
        > diagnostics["budget_rows"][0]["uniform_target_mass"]
    )


def test_validation_score_selection_score_penalizes_collapsed_predictions() -> None:
    assert _validation_score_selection_score(
        validation_score=0.8,
        pred_std=0.0,
    ) < _validation_score_selection_score(validation_score=0.2, pred_std=0.01)


def test_uniform_gap_selection_penalizes_active_type_deficit() -> None:
    workload_map = {"range": 1.0}
    uniform_per_type = {"range": 0.50}
    range_deficit = _uniform_gap_selection_score(
        validation_score=0.55,
        per_type_score={"range": 0.45},
        uniform_score=0.50,
        uniform_per_type_score=uniform_per_type,
        workload_map=workload_map,
        pred_std=0.1,
    )
    balanced = _uniform_gap_selection_score(
        validation_score=0.54,
        per_type_score={"range": 0.54},
        uniform_score=0.50,
        uniform_per_type_score=uniform_per_type,
        workload_map=workload_map,
        pred_std=0.1,
    )

    assert balanced > range_deficit


def test_balanced_pointwise_loss_pushes_constant_scores_apart() -> None:
    """Assert the anti-collapse BCE term has useful gradients from constant predictions."""
    pred = torch.zeros((8,), requires_grad=True)
    target = torch.tensor([1.0, 0.8, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
    valid_mask = torch.ones((8,), dtype=torch.bool)

    loss = _balanced_pointwise_loss(
        pred=pred,
        target=target,
        valid_mask=valid_mask,
        generator=torch.Generator().manual_seed(123),
        negatives_per_positive=2,
    )
    loss.backward()

    assert float(loss.item()) > 0.0
    assert pred.grad is not None
    assert float(pred.grad[:3].sum().item()) < 0.0
    assert float(pred.grad[3:].sum().item()) > 0.0


def test_balanced_pointwise_loss_rows_matches_scalar_when_all_zeros_selected() -> None:
    pred = torch.tensor(
        [
            [0.0, 0.1, -0.2, 0.3],
            [0.2, -0.1, 0.4, -0.3],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [1.0, 0.5, 0.0, 0.0],
            [0.0, 0.7, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.ones_like(target, dtype=torch.bool)
    generator = torch.Generator().manual_seed(123)

    row_loss, active_rows = _balanced_pointwise_loss_rows(
        pred=pred,
        target=target,
        valid_mask=valid_mask,
        generator=generator,
        negatives_per_positive=8,
    )
    scalar0 = _balanced_pointwise_loss(
        pred=pred[0],
        target=target[0],
        valid_mask=valid_mask[0],
        generator=torch.Generator().manual_seed(123),
        negatives_per_positive=8,
    )
    scalar1 = _balanced_pointwise_loss(
        pred=pred[1],
        target=target[1],
        valid_mask=valid_mask[1],
        generator=torch.Generator().manual_seed(123),
        negatives_per_positive=8,
    )

    assert active_rows.tolist() == [True, True]
    assert torch.allclose(row_loss, torch.stack([scalar0, scalar1]))


def test_balanced_pointwise_loss_rows_has_useful_gradients() -> None:
    pred = torch.zeros((2, 6), dtype=torch.float32, requires_grad=True)
    target = torch.tensor(
        [
            [1.0, 0.8, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.4, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.ones_like(target, dtype=torch.bool)

    row_loss, active_rows = _balanced_pointwise_loss_rows(
        pred=pred,
        target=target,
        valid_mask=valid_mask,
        generator=torch.Generator().manual_seed(123),
        negatives_per_positive=2,
    )
    row_loss[active_rows].mean().backward()

    assert pred.grad is not None
    assert float(pred.grad[target > 0].sum().item()) < 0.0
    assert float(pred.grad[target == 0].sum().item()) > 0.0


def test_ranking_pair_sampler_returns_finite_loss() -> None:
    pred = torch.linspace(0.1, 0.8, steps=8)
    target = torch.tensor([1.0, 0.9, 0.7, 0.4, 0.2, 0.1, 0.0, 0.0])
    valid_mask = torch.ones((8,), dtype=torch.bool)

    loss, pair_count = _ranking_loss_for_type(
        pred=pred,
        target=target,
        valid_mask=valid_mask,
        pairs_per_type=16,
        top_quantile=0.5,
        margin=0.05,
        generator=torch.Generator().manual_seed(123),
    )

    assert 0 < pair_count <= 16
    assert bool(torch.isfinite(loss).item())


def test_budget_topk_recall_loss_prefers_budget_aligned_scores() -> None:
    target = torch.tensor([1.0, 0.9, 0.1, 0.0, 0.0, 0.0], dtype=torch.float32)
    valid_mask = torch.ones((6,), dtype=torch.bool)
    good_pred = torch.tensor([3.0, 2.0, 0.5, -1.0, -2.0, -3.0], dtype=torch.float32)
    bad_pred = torch.flip(good_pred, dims=(0,))

    good_loss = _budget_topk_recall_loss(
        pred=good_pred,
        target=target,
        valid_mask=valid_mask,
        budget_ratios=(0.20, 0.40),
        temperature=0.10,
    )
    bad_loss = _budget_topk_recall_loss(
        pred=bad_pred,
        target=target,
        valid_mask=valid_mask,
        budget_ratios=(0.20, 0.40),
        temperature=0.10,
    )

    assert bool(torch.isfinite(good_loss).item())
    assert good_loss.item() < bad_loss.item()


def test_budget_topk_recall_loss_has_useful_gradients() -> None:
    pred = torch.zeros((6,), dtype=torch.float32, requires_grad=True)
    target = torch.tensor([1.0, 0.8, 0.2, 0.0, 0.0, 0.0], dtype=torch.float32)
    valid_mask = torch.ones((6,), dtype=torch.bool)

    loss = _budget_topk_recall_loss(
        pred=pred,
        target=target,
        valid_mask=valid_mask,
        budget_ratios=(0.20, 0.40),
        temperature=0.10,
    )
    loss.backward()

    assert pred.grad is not None
    assert float(pred.grad[:2].sum().item()) < 0.0
    assert float(pred.grad[3:].sum().item()) > 0.0


def test_budget_topk_recall_loss_rows_match_scalar_helper() -> None:
    pred = torch.tensor(
        [
            [3.0, 2.0, 0.5, -1.0, -2.0, -3.0],
            [-2.0, 1.5, 0.2, 0.1, -0.5, -9.0],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [1.0, 0.9, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.4, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, True, False],
        ]
    )

    row_loss, active_rows = _budget_topk_recall_loss_rows(
        pred=pred,
        target=target,
        valid_mask=valid_mask,
        budget_ratios=(0.20, 0.40),
        temperature=0.10,
    )

    assert active_rows.tolist() == [True, True]
    for row in range(pred.shape[0]):
        scalar = _budget_topk_recall_loss(
            pred=pred[row],
            target=target[row],
            valid_mask=valid_mask[row],
            budget_ratios=(0.20, 0.40),
            temperature=0.10,
        )
        assert row_loss[row].item() == pytest.approx(scalar.item(), abs=1e-6)


def test_budget_topk_temporal_residual_loss_rows_match_scalar_helper() -> None:
    pred = torch.tensor(
        [
            [3.0, 2.0, 0.5, -1.0, -2.0, -3.0],
            [-2.0, 1.5, 0.2, 0.1, -0.5, -9.0],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [1.0, 0.9, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.4, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, True, False],
        ]
    )
    global_idx = torch.tensor([[0, 1, 2, 3, 4, 5], [5, 6, 7, 8, 9, -1]])
    base_mask_a = torch.zeros((10,), dtype=torch.bool)
    base_mask_a[[0, 5, 9]] = True
    base_mask_b = torch.zeros((10,), dtype=torch.bool)
    base_mask_b[[0, 1, 5, 9]] = True
    temporal_base_masks = (
        (0.05, 0.02, base_mask_a),
        (0.10, 0.05, base_mask_b),
    )

    row_loss, active_rows = _budget_topk_temporal_residual_loss_rows(
        pred=pred,
        target=target,
        valid_mask=valid_mask,
        global_idx=global_idx.clamp(min=0),
        temporal_base_masks=temporal_base_masks,
        temperature=0.10,
    )

    assert active_rows.tolist() == [True, True]
    for row in range(pred.shape[0]):
        scalar = _budget_topk_temporal_residual_loss(
            pred=pred[row][valid_mask[row]],
            target=target[row][valid_mask[row]],
            valid_mask=torch.ones((int(valid_mask[row].sum().item()),), dtype=torch.bool),
            global_idx=global_idx[row][valid_mask[row]],
            temporal_base_masks=temporal_base_masks,
            temperature=0.10,
        )
        assert row_loss[row].item() == pytest.approx(scalar.item(), abs=1e-6)


def test_temporal_residual_labels_drop_base_points() -> None:
    labels = torch.ones((10, 4), dtype=torch.float32)
    labelled_mask = torch.ones((10, 4), dtype=torch.bool)

    residual_labels, residual_mask = _apply_temporal_residual_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        boundaries=[(0, 10)],
        compression_ratio=0.3,
        temporal_fraction=0.5,
    )

    assert torch.where(~residual_mask[:, 0])[0].tolist() == [0, 9]
    assert residual_labels[[0, 9]].sum().item() == pytest.approx(0.0)
    assert bool(residual_mask[5].all().item())


def test_temporal_residual_labels_keep_all_labels_when_base_disabled() -> None:
    labels = torch.ones((10, 4), dtype=torch.float32)
    labelled_mask = torch.ones((10, 4), dtype=torch.bool)

    residual_labels, residual_mask = _apply_temporal_residual_labels(
        labels=labels,
        labelled_mask=labelled_mask,
        boundaries=[(0, 10)],
        compression_ratio=0.3,
        temporal_fraction=0.0,
    )

    assert bool(residual_mask.all().item())
    assert float(residual_labels.sum().item()) == pytest.approx(float(labels.sum().item()))


def test_single_active_type_rejects_mixed_training_weights() -> None:
    assert _single_active_type_id(torch.tensor([1.0, 0.0, 0.0, 0.0])) == 0

    with pytest.raises(ValueError, match="Pure-workload"):
        _single_active_type_id(torch.tensor([0.5, 0.5, 0.0, 0.0]))


def test_filter_supervised_windows_removes_zero_positive_training_windows() -> None:
    points = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    windows = build_trajectory_windows(
        points, boundaries=[(0, 4), (4, 12)], window_length=4, stride=4
    )
    targets = torch.zeros((12, 4), dtype=torch.float32)
    labelled_mask = torch.ones((12, 4), dtype=torch.bool)
    targets[5, 0] = 1.0

    kept, filtered = _filter_supervised_windows(
        windows=windows,
        training_target=targets[:, 0],
        labelled_mask=labelled_mask[:, 0],
        active_type_id=0,
    )

    assert len(windows) == 3
    assert len(kept) == 1
    assert int(filtered[0].item()) == 2


def test_filter_supervised_windows_can_keep_zero_labelled_windows_for_pointwise_objective() -> None:
    points = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    windows = build_trajectory_windows(
        points, boundaries=[(0, 4), (4, 12)], window_length=4, stride=4
    )
    targets = torch.zeros((12,), dtype=torch.float32)
    labelled_mask = torch.ones((12,), dtype=torch.bool)

    kept, filtered = _filter_supervised_windows(
        windows=windows,
        training_target=targets,
        labelled_mask=labelled_mask,
        active_type_id=0,
        require_positive=False,
    )

    assert len(windows) == 3
    assert len(kept) == 3
    assert int(filtered[0].item()) == 0
