"""Guardrails for active Range_QDS implementation cleanup decisions."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

import learning.targets.query_local_utility as query_local_utility_targets
from benchmarking.profiles import (
    BLIND_EXPECTED_QUERY_LOCAL_UTILITY_PROFILE,
    BLIND_RETAINED_FREQUENCY_PROFILE,
    BLIND_TEACHER_DISTILL_PROFILE,
    DEFAULT_PROFILE,
    PROFILE_CHOICES,
    RANGE_QUERY_MIX_WORKLOAD_BLIND_PROFILE,
    benchmark_profile,
    benchmark_profile_args,
    benchmark_profile_settings,
)
from benchmarking.reporting.row_fields import _row_from_run
from learning.model_features import model_type_metadata
from learning.targets.modes import SCALAR_RANGE_TARGET_MODES
from learning.targets.query_local_utility import (
    QUERY_LOCAL_UTILITY_TARGET_MODES,
)
from learning.targets.registry import RANGE_SCALAR_TARGET_MODE_SPECS

REPO_ROOT = Path(__file__).resolve().parents[3]
MAX_PRODUCTION_FILE_LINES = 1000
MAX_PRODUCTION_FUNCTION_LINES = 350
KNOWN_LARGE_FUNCTION_BUDGETS = {
    (
        "Range_QDS/learning/factorized_prior_transfer_diagnostics.py",
        "_prior_to_head_transfer_sensitivity_diagnostics",
    ): 540,
    ("Range_QDS/learning/model_training.py", "train_model"): 399,
    ("Range_QDS/learning/model_training_loop.py", "run_training_epochs"): 363,
    ("Range_QDS/orchestration/final_gate_summary.py", "build_final_run_summaries"): 425,
    (
        "Range_QDS/orchestration/selector_marginal_diagnostics.py",
        "retained_decision_marginal_query_local_utility_diagnostics",
    ): 353,
    (
        "Range_QDS/selection/learned_segment_budget/core.py",
        "simplify_with_learned_segment_budget_with_trace",
    ): 355,
}


@pytest.mark.parametrize(
    "module_name",
    [
        "config.experiment_config",
        "learning.training_pipeline",
        "learning.targets.legacy",
        "selection.selector_diagnostics",
        "selection.legacy_temporal_hybrid",
        "simplification",
    ],
)
def test_removed_compatibility_shims_stay_removed(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is None


def test_removed_query_local_utility_target_build_alias_stays_removed() -> None:
    assert not hasattr(query_local_utility_targets, "build")


def test_production_files_stay_under_1000_lines() -> None:
    violations = []
    for path in sorted((REPO_ROOT / "Range_QDS").rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative.startswith("Range_QDS/tests/"):
            continue
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > MAX_PRODUCTION_FILE_LINES:
            violations.append(f"{relative}:{line_count}")

    assert violations == []


def test_large_production_functions_do_not_grow_without_explicit_budget() -> None:
    violations = []
    for path in sorted((REPO_ROOT / "Range_QDS").rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative.startswith("Range_QDS/tests/"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            span = int(node.end_lineno or node.lineno) - int(node.lineno) + 1
            budget = KNOWN_LARGE_FUNCTION_BUDGETS.get((relative, node.name))
            if budget is None and span > MAX_PRODUCTION_FUNCTION_LINES:
                violations.append(f"{relative}:{node.lineno}:{node.name}:{span}")
            elif budget is not None and span > budget:
                violations.append(f"{relative}:{node.lineno}:{node.name}:{span}>{budget}")

    assert violations == []


def test_orchestration_production_modules_do_not_cross_import_private_helpers() -> None:
    violations: list[str] = []
    orchestration_dir = REPO_ROOT / "Range_QDS" / "orchestration"

    for path in sorted(orchestration_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not node.module or not node.module.startswith("orchestration."):
                continue
            violations.extend(
                f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{alias.name}"
                for alias in node.names
                if alias.name.startswith("_") and not alias.name.startswith("__")
            )

    assert violations == []


def test_root_makefile_points_to_range_qds() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    range_dir_var = "RANGE_" + "QDS" + "_DIR"

    assert f"{range_dir_var} := $(REPO_ROOT)/Range_QDS" in makefile
    assert "QDS" + "_DIR := $(REPO_ROOT)/" + "QDS" not in makefile
    assert f"-C $({range_dir_var}) test" in makefile
    assert f"-C $({range_dir_var}) lint" in makefile
    assert f"-C $({range_dir_var}) typecheck" in makefile


def test_pyproject_uses_range_qds_paths() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'where = [".", "Range_QDS"]' in pyproject
    assert 'testpaths = ["Range_QDS/tests"]' in pyproject
    assert 'pythonpath = [".", "Range_QDS"]' in pyproject
    assert 'src = [".", "Range_QDS"]' in pyproject
    assert '"QDS"' not in pyproject


@pytest.mark.parametrize(
    "profile_name",
    [
        DEFAULT_PROFILE,
        BLIND_EXPECTED_QUERY_LOCAL_UTILITY_PROFILE,
        BLIND_RETAINED_FREQUENCY_PROFILE,
        BLIND_TEACHER_DISTILL_PROFILE,
    ],
)
def test_diagnostic_profiles_block_final_success(profile_name: str) -> None:
    profile = benchmark_profile(profile_name)
    settings = benchmark_profile_settings(profile_name)

    assert profile.final_success_allowed is False
    assert settings["profile_diagnostic_only"] is True
    assert settings["primary_metric_family"] == "QueryLocalUtility"
    assert settings["final_success_allowed"] is False
    assert settings["final_product_candidate"] is False
    assert "profile_legacy_diagnostic" not in settings
    assert "legacy_reason" not in settings


def test_advertised_benchmark_profiles_are_implemented() -> None:
    for profile_name in PROFILE_CHOICES:
        profile = benchmark_profile(profile_name)
        args = benchmark_profile_args(profile_name)
        settings = benchmark_profile_settings(profile_name)

        assert profile.name == profile_name
        assert args
        assert settings["profile_note"]


def test_query_driven_workload_blind_profile_is_final_candidate() -> None:
    profile = benchmark_profile(RANGE_QUERY_MIX_WORKLOAD_BLIND_PROFILE)
    settings = benchmark_profile_settings(RANGE_QUERY_MIX_WORKLOAD_BLIND_PROFILE)
    args = benchmark_profile_args(RANGE_QUERY_MIX_WORKLOAD_BLIND_PROFILE)

    assert profile.model_type == "workload_blind_range"
    assert profile.range_training_target_mode == "query_local_utility_factorized"
    assert profile.workload_profile_id == "range_query_mix"
    assert profile.selector_type == "learned_segment_budget"
    assert profile.range_train_workload_replicates == 4
    assert profile.query_coverage is None
    assert profile.range_max_coverage_overshoot is None
    assert settings["workload_profile_default_target_coverage"] == 0.30
    assert settings["workload_profile_default_max_coverage_overshoot"] == 0.020
    assert settings["range_workload_profile_sweep_ids"] == [
        "range_query_mix_focused",
        "range_query_mix_local",
        "range_query_mix_operational",
        "range_query_mix",
    ]
    assert settings["primary_metric_family"] == "QueryLocalUtility"
    assert settings["final_success_allowed"] is True
    assert settings["range_train_workload_replicates"] == 4
    assert settings["profile_diagnostic_only"] is False
    assert "profile_legacy_diagnostic" not in settings
    assert "legacy_reason" not in settings
    assert args[args.index("--range_train_workload_replicates") + 1] == "4"


def test_scalar_and_query_local_utility_target_modes_are_separated() -> None:
    assert "retained_frequency" in SCALAR_RANGE_TARGET_MODES
    assert "historical_prior_retained_frequency" in SCALAR_RANGE_TARGET_MODES
    assert "query_local_utility_factorized" not in SCALAR_RANGE_TARGET_MODES
    assert set(RANGE_SCALAR_TARGET_MODE_SPECS) == set(SCALAR_RANGE_TARGET_MODES) - {
        "point_value"
    }
    assert (
        frozenset(
            {
                "query_local_utility_factorized",
            }
        )
        == QUERY_LOCAL_UTILITY_TARGET_MODES
    )


def test_historical_prior_metadata_blocks_success() -> None:
    assert model_type_metadata("historical_prior") == {
        "model_family": "historical_prior_knn",
        "trainable_final_candidate": False,
        "final_success_allowed": False,
    }
    student_metadata = model_type_metadata("historical_prior_student")
    assert student_metadata["requires_ablation_against_standalone_knn"] is True
    assert student_metadata["final_success_allowed"] is False


def test_benchmark_row_uses_final_claim_status_for_query_local_utility(tmp_path: Path) -> None:
    row = _row_from_run(
        workload="range",
        run_label="range_audit_diagnostic",
        command=[
            "uv",
            "run",
            "--group",
            "dev",
            "--",
            "python",
            "-m",
            "orchestration.train_and_score",
        ],
        returncode=0,
        elapsed_seconds=1.0,
        run_dir=tmp_path,
        stdout_path=tmp_path / "stdout.txt",
        run_json_path=tmp_path / "example_run.json",
        timings={},
        run_json={
            "final_claim_summary": {
                "primary_metric": None,
                "status": "not_available_until_query_local_utility",
                "final_success_allowed": False,
            },
            "matched": {
                "MLQDS": {"query_local_utility_score": 0.7, "range_point_f1": 0.6},
                "uniform": {"query_local_utility_score": 0.5, "range_point_f1": 0.4},
                "DouglasPeucker": {"query_local_utility_score": 0.4, "range_point_f1": 0.3},
            },
            "config": {
                "model": {
                    "model_type": "historical_prior",
                    "compression_ratio": 0.05,
                    "range_training_target_mode": "retained_frequency",
                },
                "query": {},
                "data": {},
                "baselines": {},
            },
            "workload_blind_protocol": {
                "enabled": True,
                "primary_masks_frozen_before_eval_query_scoring": True,
                "audit_masks_frozen_before_eval_query_scoring": True,
            },
        },
    )

    assert row["final_claim_status"] == "not_available_until_query_local_utility"
    assert row["final_success_allowed"] is False
    assert row["model_metadata_model_family"] == "historical_prior_knn"
