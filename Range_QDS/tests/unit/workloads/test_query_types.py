"""Tests canonical range-query schema validation."""

from __future__ import annotations

import math

import pytest

from workloads.query_types import pad_query_features, validated_range_query_params


def _range_query(**params: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "lat_min": 0,
        "lat_max": 1,
        "lon_min": 2,
        "lon_max": 3,
        "t_start": 4,
        "t_end": 5,
    }
    defaults.update(params)
    return {"type": "range", "params": defaults}


def test_validated_range_query_params_normalizes_numeric_values() -> None:
    params = validated_range_query_params(_range_query(lat_min="0.25", t_end=5))

    assert params == {
        "lat_min": 0.25,
        "lat_max": 1.0,
        "lon_min": 2.0,
        "lon_max": 3.0,
        "t_start": 4.0,
        "t_end": 5.0,
    }


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ({"type": "range", "params": {"lat_min": 0.0}}, "missing required keys"),
        (_range_query(t_start=math.nan), "must be finite"),
        (_range_query(lat_min=2.0, lat_max=1.0), "lower bounds"),
        ({"type": "polygon", "params": {}}, "Only range queries"),
        ({"type": "range", "params": None}, "params mapping"),
    ],
)
def test_validated_range_query_params_rejects_invalid_shapes(
    query: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validated_range_query_params(query)


def test_pad_query_features_uses_canonical_range_validation() -> None:
    features, type_ids = pad_query_features([_range_query(t_start="10", t_end="20")])

    assert features.shape == (1, 12)
    assert type_ids.tolist() == [0]
    assert float(features[0, 4].item()) == pytest.approx(10.0)
    assert float(features[0, 5].item()) == pytest.approx(20.0)

    with pytest.raises(ValueError, match="lower bounds"):
        pad_query_features([_range_query(t_start=20.0, t_end=10.0)])
