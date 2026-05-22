"""Tests pipeline metric and diagnostic reporting contracts."""

from __future__ import annotations

from config.run_config import build_run_config
from orchestration.learning_scoring_pipeline import run_learning_scoring_pipeline


def test_pipeline_reports_active_query_local_metrics_and_diagnostics(
    synthetic_dataset, tmp_path
) -> None:
    """Assert the active query-driven candidate emits QueryLocalUtility contracts."""
    trajectories, _ = synthetic_dataset
    cfg = build_run_config(
        n_queries=24,
        epochs=1,
        model_type="workload_blind_range",
        range_training_target_mode="query_local_utility_factorized",
        selector_type="learned_segment_budget",
        checkpoint_score_variant="query_local_utility",
        checkpoint_selection_metric="uniform_gap",
        workload_profile_id="range_query_mix",
        range_train_workload_replicates=2,
        validation_score_every=1,
        compression_ratio=0.10,
    )

    out = run_learning_scoring_pipeline(
        config=cfg,
        trajectories=trajectories,
        results_dir=str(tmp_path),
        save_simplified_dir=str(tmp_path / "simplified"),
    )

    ml = out.metrics_dump["matched"]["MLQDS"]["aggregate_f1"]
    uni = out.metrics_dump["matched"]["uniform"]["aggregate_f1"]
    assert 0.0 <= ml <= 1.0
    assert 0.0 <= uni <= 1.0
    assert "Random" not in out.metrics_dump["matched"]
    assert "TemporalRandomFill" not in out.metrics_dump["matched"]
    assert "TemporalRandomFill" in out.metrics_dump["learned_fill_diagnostics"]
    assert "TemporalOracleFill" in out.metrics_dump["learned_fill_diagnostics"]
    assert "GlobalRandomBudget" in out.metrics_dump["learned_fill_diagnostics"]
    assert "GlobalOracleBudget" in out.metrics_dump["learned_fill_diagnostics"]
    assert "training_target_diagnostics" in out.metrics_dump
    assert "range_learned_fill_summary" in out.metrics_dump
    assert "workload_distribution_comparison" in out.metrics_dump
    assert "QueryLocalUtility" in out.matched_table
    assert "AggregateErr" not in out.matched_table
    assert "aggregate_error" not in out.metrics_dump["matched"]["MLQDS"]
    assert "per_type_f1" in out.metrics_dump["matched"]["MLQDS"]
    assert "query_local_utility_score" in out.metrics_dump["matched"]["MLQDS"]
    assert out.metrics_dump["checkpoint_selection_metric"] == "uniform_gap"
    assert out.metrics_dump["checkpoint_score_variant"] == "query_local_utility"
    assert out.metrics_dump["final_claim_summary"]["primary_metric"] == "QueryLocalUtility"
    assert out.metrics_dump["workload_blind_protocol"]["enabled"] is True
    assert (
        out.metrics_dump["workload_blind_protocol"][
            "primary_masks_frozen_before_eval_query_scoring"
        ]
        is True
    )
    assert out.metrics_dump["learning_causality_summary"]["selector_final_candidate"] is True
    assert "legacy_temporal_hybrid_selector" not in out.metrics_dump["learning_causality_summary"]
    assert out.metrics_dump["checkpoint_smoothing_window"] == 1
    learned_fill_summary = out.metrics_dump["range_learned_fill_summary"]
    assert learned_fill_summary["summary_version"] == 1
    assert learned_fill_summary["oracle_notes"]["exact_optimum"] is False
    assert "mlqds_vs_temporal_random_fill_query_local_utility" in learned_fill_summary
    assert "target_residual_label_mass_fraction" in learned_fill_summary
    assert "train_label_component_mass_fraction" in learned_fill_summary
    assert (
        out.metrics_dump["oracle_diagnostic"]["retained_mask_constructor"]
        == "per_trajectory_topk_with_endpoints"
    )

    # F1 is higher-is-better, so callers should rank with max(), not min().
    scores = [float(metrics["aggregate_f1"]) for metrics in out.metrics_dump["matched"].values()]
    assert max(scores) >= min(scores)
    assert (tmp_path / "simplified" / "ML_simplified_eval.csv").exists()
    assert (tmp_path / "range_learned_fill_summary.json").exists()
    assert not (tmp_path / "simplified" / "ML_simplified_train.csv").exists()
    assert not (tmp_path / "simplified" / "ML_simplified.csv").exists()


def test_core_final_metrics_mode_skips_diagnostic_baselines(synthetic_dataset, tmp_path) -> None:
    trajectories, _ = synthetic_dataset
    cfg = build_run_config(n_queries=16, epochs=1, final_metrics_mode="core")

    out = run_learning_scoring_pipeline(
        config=cfg,
        trajectories=trajectories,
        results_dir=str(tmp_path),
    )

    assert out.metrics_dump["final_metrics_mode"] == "core"
    assert "Oracle" not in out.metrics_dump["matched"]
    assert set(out.metrics_dump["learned_fill_diagnostics"]) == {"MLQDS"}
    assert out.metrics_dump["oracle_diagnostic"]["enabled"] is False
