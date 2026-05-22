"""Typed query schemas and feature padding helpers. See workloads/README.md for details."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

QUERY_TYPE_ID_RANGE = 0
NUM_QUERY_TYPES = 1

QUERY_NAME_TO_ID = {
    "range": QUERY_TYPE_ID_RANGE,
}
ID_TO_QUERY_NAME = {v: k for k, v in QUERY_NAME_TO_ID.items()}
RANGE_QUERY_PARAM_KEYS = ("lat_min", "lat_max", "lon_min", "lon_max", "t_start", "t_end")
RangeQueryParams = dict[str, float]
TypedQuery = dict[str, Any]


def validated_range_query_params(query: Mapping[str, Any]) -> RangeQueryParams:
    """Return normalized range-query params after schema and bounds validation."""
    if str(query.get("type", "")).lower() != "range":
        raise ValueError(f"Only range queries are supported; got query type: {query.get('type')!r}")
    raw_params = query.get("params")
    if not isinstance(raw_params, Mapping):
        raise ValueError("Range query must contain a params mapping.")
    missing = [key for key in RANGE_QUERY_PARAM_KEYS if key not in raw_params]
    if missing:
        raise ValueError(f"Range query params missing required keys: {missing}")
    params: RangeQueryParams = {}
    for key in RANGE_QUERY_PARAM_KEYS:
        raw_value = raw_params[key]
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Range query param {key!r} must be numeric.") from exc
        if not math.isfinite(value):
            raise ValueError(f"Range query param {key!r} must be finite.")
        params[key] = value
    if (
        params["lat_min"] > params["lat_max"]
        or params["lon_min"] > params["lon_max"]
        or params["t_start"] > params["t_end"]
    ):
        raise ValueError("Range query lower bounds must not exceed upper bounds.")
    return params


def normalize_pure_workload_map(workload_map: dict[str, float]) -> dict[str, float]:
    """Normalize one pure workload map to ``{type: 1.0}``."""
    filtered = {k.lower(): float(v) for k, v in workload_map.items() if float(v) > 0.0}
    total = sum(filtered.values())
    if total <= 0.0:
        raise ValueError("Workload map must contain at least one positive weight.")
    for name in filtered:
        if name not in QUERY_NAME_TO_ID:
            raise ValueError(f"Only range workloads are supported; got query type: {name}")
    if len(filtered) != 1:
        raise ValueError(f"Expected exactly one active workload type; got {workload_map}.")
    return {k: v / total for k, v in filtered.items()}


def single_workload_type(workload_map: dict[str, float]) -> str:
    """Return the one active workload type, rejecting mixed workloads."""
    normalized = normalize_pure_workload_map(workload_map)
    return next(iter(normalized))


def pad_query_features(typed_queries: list[TypedQuery]) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert range query dicts to padded numeric features + type IDs. See workloads/README.md for details."""
    feature_dim = 12
    feats = torch.zeros((len(typed_queries), feature_dim), dtype=torch.float32)
    type_ids = torch.zeros((len(typed_queries),), dtype=torch.long)

    for i, query in enumerate(typed_queries):
        qtype = str(query["type"]).lower()
        if qtype == "range":
            params = validated_range_query_params(query)
            type_ids[i] = QUERY_TYPE_ID_RANGE
            feats[i, :6] = torch.tensor(
                [
                    params["lat_min"],
                    params["lat_max"],
                    params["lon_min"],
                    params["lon_max"],
                    params["t_start"],
                    params["t_end"],
                ],
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Only range queries are supported; got query type: {qtype}")
    return feats, type_ids
