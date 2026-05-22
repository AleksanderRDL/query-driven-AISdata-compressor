"""Shared context helpers for benchmark reporting rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from benchmarking.reporting.audit_extractors import (
    _audit_summary,
    _selector_budget_row,
    _target_budget_row,
)
from benchmarking.reporting.metrics import _selector_claim_evidence

RowFields = dict[str, Any]


def _mapping(value: Any) -> RowFields:
    return cast(RowFields, value) if isinstance(value, dict) else {}


@dataclass(frozen=True)
class RowContext:
    workload: str
    run_label: str
    command: list[str]
    returncode: int
    elapsed_seconds: float
    run_dir: Path
    stdout_path: Path
    run_json_path: Path
    timings: RowFields
    run_json: RowFields | None
    data_sources: RowFields | None = None

    @property
    def run(self) -> RowFields:
        return _mapping(self.run_json)

    @property
    def mlqds(self) -> RowFields:
        return _mapping(_mapping(self.run.get("matched")).get("MLQDS"))

    @property
    def uniform(self) -> RowFields:
        return _mapping(_mapping(self.run.get("matched")).get("uniform"))

    @property
    def douglas_peucker(self) -> RowFields:
        return _mapping(_mapping(self.run.get("matched")).get("DouglasPeucker"))

    @property
    def learned_fill(self) -> RowFields:
        return _mapping(self.run.get("learned_fill_diagnostics"))

    @property
    def temporal_random_fill(self) -> RowFields:
        return _mapping(self.learned_fill.get("TemporalRandomFill"))

    @property
    def temporal_oracle_fill(self) -> RowFields:
        return _mapping(self.learned_fill.get("TemporalOracleFill"))

    @property
    def config(self) -> RowFields:
        return _mapping(self.run.get("config"))

    @property
    def data_config(self) -> RowFields:
        return _mapping(self.config.get("data"))

    @property
    def model_config(self) -> RowFields:
        return _mapping(self.config.get("model"))

    @property
    def query_config(self) -> RowFields:
        return _mapping(self.config.get("query"))

    @property
    def baseline_config(self) -> RowFields:
        return _mapping(self.config.get("baselines"))

    @property
    def cuda_memory(self) -> RowFields:
        return _mapping(_mapping(self.run.get("cuda_memory")).get("training"))

    @property
    def child_torch_runtime(self) -> RowFields:
        return _mapping(self.run.get("torch_runtime"))

    @property
    def child_amp(self) -> RowFields:
        return _mapping(self.child_torch_runtime.get("amp"))

    @property
    def workload_blind_protocol(self) -> RowFields:
        return _mapping(self.run.get("workload_blind_protocol"))

    @property
    def train_label_diagnostics(self) -> RowFields:
        return _mapping(
            _mapping(
                _mapping(_mapping(self.run.get("workload_diagnostics")).get("train")).get(
                    "range_signal"
                )
            ).get("labels")
        )

    @property
    def target_diagnostics(self) -> RowFields:
        return _mapping(self.run.get("training_target_diagnostics"))

    @property
    def target_budget_row(self) -> RowFields:
        return _target_budget_row(
            self.target_diagnostics,
            self.model_config.get("compression_ratio"),
        )

    @property
    def eval_selector_diagnostics(self) -> RowFields:
        return _mapping(_mapping(self.run.get("selector_budget_diagnostics")).get("eval"))

    @property
    def selector_budget_row(self) -> RowFields:
        return _selector_budget_row(
            self.eval_selector_diagnostics,
            self.model_config.get("compression_ratio"),
        )

    @property
    def selector_claim_evidence(self) -> RowFields:
        return _selector_claim_evidence(
            self.selector_budget_row,
            self.model_config.get("model_type"),
        )

    @property
    def audit(self) -> RowFields:
        return _audit_summary(self.run_json)
