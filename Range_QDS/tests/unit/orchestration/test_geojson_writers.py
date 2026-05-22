"""Tests GeoJSON query export validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestration.geojson_writers import write_queries_geojson


def test_write_queries_geojson_rejects_unsupported_query_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Only range queries"):
        write_queries_geojson(str(tmp_path), [{"type": "polygon", "params": {}}])


def test_write_queries_geojson_writes_range_features(tmp_path: Path) -> None:
    write_queries_geojson(
        str(tmp_path),
        [
            {
                "type": "range",
                "params": {
                    "lat_min": 10.0,
                    "lat_max": 11.0,
                    "lon_min": 20.0,
                    "lon_max": 21.0,
                    "t_start": 3600.0,
                    "t_end": 7200.0,
                },
            }
        ],
    )

    geojson_payload = json.loads((tmp_path / "queries_range.geojson").read_text(encoding="utf-8"))

    assert geojson_payload["type"] == "FeatureCollection"
    assert geojson_payload["features"][0]["properties"]["query_type"] == "range"
    assert geojson_payload["features"][0]["properties"]["t_start_hm"] == "01:00"
