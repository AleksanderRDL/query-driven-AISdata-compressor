"""Tests F1-contribution training labels. See learning/README.md for details."""

from __future__ import annotations

import pytest
import torch

from learning.importance_labels import compute_typed_importance_labels
from workloads.query_types import QUERY_TYPE_ID_RANGE


def test_range_labels_match_singleton_point_f1_contribution() -> None:
    """Assert range labels equal the F1 gain of recovering one matching point."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 3.0],
            [1.0, 5.0, 5.0, 9.0],
            [0.0, 0.1, 0.1, 1.0],
            [1.0, 6.0, 6.0, 7.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 2), (2, 4)]
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 0.5,
            },
        }
    ]

    labels, labelled_mask = compute_typed_importance_labels(points, boundaries, queries)

    expected_gain = 2.0 / 3.0
    assert labels[0, QUERY_TYPE_ID_RANGE].item() == pytest.approx(expected_gain)
    assert labels[2, QUERY_TYPE_ID_RANGE].item() == pytest.approx(expected_gain)
    assert labels[1, QUERY_TYPE_ID_RANGE].item() == pytest.approx(0.0)
    assert labels[3, QUERY_TYPE_ID_RANGE].item() == pytest.approx(0.0)
    assert bool(labelled_mask[:, QUERY_TYPE_ID_RANGE].all().item())


def test_range_labels_reward_each_in_box_point() -> None:
    """Assert duplicate in-box points are scored as individual range-query hits."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.1, 0.1, 1.0],
            [2.0, 5.0, 5.0, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 3)]
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 1.5,
            },
        }
    ]

    labels, _ = compute_typed_importance_labels(points, boundaries, queries)

    expected_gain = 2.0 / 3.0
    assert labels[0, QUERY_TYPE_ID_RANGE].item() == pytest.approx(expected_gain)
    assert labels[1, QUERY_TYPE_ID_RANGE].item() == pytest.approx(expected_gain)
    assert labels[2, QUERY_TYPE_ID_RANGE].item() == pytest.approx(0.0)


def test_large_range_labels_do_not_use_quadratic_proximity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert range labels do not allocate an all-pairs distance matrix."""
    n_per_trajectory = 2500
    time = torch.arange(n_per_trajectory * 2, dtype=torch.float32)
    lat = torch.cat(
        [
            torch.linspace(0.0, 1.0, n_per_trajectory),
            torch.linspace(0.1, 1.1, n_per_trajectory),
        ]
    )
    lon = torch.cat(
        [
            torch.linspace(0.0, 1.0, n_per_trajectory),
            torch.linspace(0.1, 1.1, n_per_trajectory),
        ]
    )
    speed = torch.ones_like(time)
    heading = torch.zeros_like(time)
    is_start = torch.zeros_like(time)
    is_end = torch.zeros_like(time)
    turn = torch.zeros_like(time)
    points = torch.stack([time, lat, lon, speed, heading, is_start, is_end, turn], dim=1)
    boundaries = [(0, n_per_trajectory), (n_per_trajectory, n_per_trajectory * 2)]
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 2.0,
                "lon_min": -1.0,
                "lon_max": 2.0,
                "t_start": -1.0,
                "t_end": float(points[-1, 0].item()) + 1.0,
            },
        }
    ]

    def fail_cdist(*args, **kwargs):
        raise AssertionError("large range labels should not call dense torch.cdist")

    monkeypatch.setattr(torch, "cdist", fail_cdist)

    labels, labelled_mask = compute_typed_importance_labels(points, boundaries, queries)

    assert bool(labelled_mask[:, QUERY_TYPE_ID_RANGE].all().item())
    assert float(labels[:, QUERY_TYPE_ID_RANGE].sum().item()) > 0.0
    assert torch.isfinite(labels).all()


def test_range_boundary_prior_is_optional_and_mass_preserving() -> None:
    """Assert boundary weighting is opt-in and preserves total query label mass."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.1, 1.0],
            [2.0, 0.0, 0.2, 1.0],
            [3.0, 0.0, 0.3, 1.0],
            [4.0, 9.0, 9.0, 1.0],
        ],
        dtype=torch.float32,
    )
    boundaries = [(0, 5)]
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 3.5,
            },
        }
    ]

    pure, _ = compute_typed_importance_labels(points, boundaries, queries)
    boundary, _ = compute_typed_importance_labels(
        points,
        boundaries,
        queries,
        range_boundary_prior_weight=1.0,
    )

    pure_values = pure[:4, QUERY_TYPE_ID_RANGE]
    boundary_values = boundary[:4, QUERY_TYPE_ID_RANGE]
    assert pure_values.tolist() == pytest.approx([0.4, 0.4, 0.4, 0.4])
    assert boundary_values[0].item() > boundary_values[1].item()
    assert boundary_values[3].item() > boundary_values[2].item()
    assert float(boundary_values.sum().item()) == pytest.approx(float(pure_values.sum().item()))


def test_range_label_mode_rejects_unknown_mode() -> None:
    points = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    queries = [
        {
            "type": "range",
            "params": {
                "lat_min": -1.0,
                "lat_max": 1.0,
                "lon_min": -1.0,
                "lon_max": 1.0,
                "t_start": -1.0,
                "t_end": 1.0,
            },
        }
    ]

    with pytest.raises(ValueError, match="range_label_mode"):
        compute_typed_importance_labels(
            points,
            [(0, 1)],
            queries,
            range_label_mode="not-a-mode",
        )
