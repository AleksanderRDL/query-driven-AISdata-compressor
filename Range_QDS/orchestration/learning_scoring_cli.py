"""CLI parsing helpers for the AIS-QDS run entrypoint. See orchestration/README.md for details."""

from __future__ import annotations

import argparse
from pathlib import Path

from config.run_config import (
    DEFAULT_BUDGET_LOSS_RATIOS,
    DEFAULT_BUDGET_LOSS_TEMPERATURE,
    DEFAULT_QUERY_LOCAL_UTILITY_BEHAVIOR_RANK_LOSS_WEIGHT,
    DEFAULT_VALIDATION_ENDPOINT_PENALTY_WEIGHT,
    DEFAULT_VALIDATION_GLOBAL_SANITY_PENALTY_WEIGHT,
    DEFAULT_VALIDATION_LENGTH_PRESERVATION_MIN,
    DEFAULT_VALIDATION_SED_PENALTY_WEIGHT,
    VALIDATION_SPLIT_MODES,
)
from learning.model_features import SUPPORTED_MODEL_TYPES
from orchestration.learning_scoring_cli_selector_args import (
    add_selector_and_range_target_arguments,
)
from runtime.torch_runtime import AMP_MODE_CHOICES, FLOAT32_MATMUL_PRECISION_CHOICES
from workloads.generation.anchors import RANGE_ANCHOR_MODES
from workloads.generation.generator import RANGE_TIME_DOMAIN_MODES
from workloads.generation.workload_profiles import WORKLOAD_PROFILE_CHOICES

QDS_ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
DEFAULT_RESULTS_DIR = QDS_ARTIFACTS_DIR / "results" / "latest"


def _compression_ratio_list(value: str) -> list[float]:
    """Parse comma-separated compression ratios for optional range audits."""
    ratios: list[float] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        ratio = float(item)
        if ratio <= 0.0 or ratio > 1.0:
            raise argparse.ArgumentTypeError("compression ratios must be in (0, 1].")
        ratios.append(ratio)
    if not ratios:
        raise argparse.ArgumentTypeError("provide at least one compression ratio.")
    return ratios


def _range_anchor_mode_list(value: str) -> list[str]:
    """Parse comma-separated train anchor-prior modes."""
    modes: list[str] = []
    for raw in value.split(","):
        mode = raw.strip().lower()
        if not mode:
            continue
        if mode not in RANGE_ANCHOR_MODES:
            raise argparse.ArgumentTypeError(
                f"range train anchor modes must be one of {RANGE_ANCHOR_MODES}."
            )
        modes.append(mode)
    if not modes:
        raise argparse.ArgumentTypeError("provide at least one range anchor mode.")
    return modes


def _range_train_footprint_list(value: str) -> list[str]:
    """Parse comma-separated train footprint families as spatial_km:time_hours."""
    footprints: list[str] = []
    for raw in value.split(","):
        item = raw.strip().lower().replace("x", ":")
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "range train footprints must use spatial_km:time_hours entries, "
                "for example 1.1:2.5,2.2:5.0."
            )
        try:
            spatial_km = float(parts[0])
            time_hours = float(parts[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "range train footprint values must be numeric."
            ) from exc
        if spatial_km <= 0.0 or time_hours <= 0.0:
            raise argparse.ArgumentTypeError("range train footprint values must be positive.")
        footprints.append(f"{spatial_km:g}:{time_hours:g}")
    if not footprints:
        raise argparse.ArgumentTypeError("provide at least one range train footprint.")
    return footprints


def _add_data_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument(
        "--train_csv_path",
        "--train_csv",
        dest="train_csv_path",
        type=str,
        default=None,
        help=(
            "Dedicated train CSV path. A comma-separated list trains on multiple historical "
            "CSV days while keeping validation/eval sources separate."
        ),
    )
    parser.add_argument(
        "--validation_csv_path",
        "--validation_csv",
        "--val_csv_path",
        "--val_csv",
        dest="validation_csv_path",
        type=str,
        default=None,
        help=(
            "Optional dedicated checkpoint-validation CSV path or comma-separated CSV list. "
            "Requires --train_csv_path and --eval_csv_path."
        ),
    )
    parser.add_argument(
        "--eval_csv_path",
        "--eval_csv",
        dest="eval_csv_path",
        type=str,
        default=None,
        help="Dedicated final-eval CSV path or comma-separated CSV list.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional directory for segmented AIS Parquet caches keyed by source file and load config.",
    )
    parser.add_argument(
        "--refresh_cache",
        action="store_true",
        help="Rebuild AIS cache entries even when a matching manifest exists.",
    )
    parser.add_argument(
        "--range_diagnostics_mode",
        type=str,
        default="full",
        choices=["full", "cached"],
        help="Use full range diagnostics or reuse persistent range-diagnostics caches when --cache_dir is set.",
    )
    parser.add_argument(
        "--validation_split_mode",
        type=str,
        default="random",
        choices=VALIDATION_SPLIT_MODES,
        help=(
            "Fallback checkpoint-validation split when separate train/eval CSVs are used without "
            "--validation_csv_path. 'random' samples from combined train trajectories; "
            "'source_stratified' holds out validation trajectories from each train CSV source."
        ),
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.70,
        help="Single-dataset train trajectory fraction. Ignored when --eval_csv_path is provided.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.15,
        help="Single-dataset checkpoint-validation trajectory fraction. Ignored when validation CSVs are provided.",
    )
    parser.add_argument(
        "--final_metrics_mode",
        type=str,
        default="diagnostic",
        choices=["diagnostic", "core"],
        help=(
            "Final scoring scope. 'diagnostic' keeps Oracle and learned-fill diagnostic baselines; "
            "'core' reports exact MLQDS/uniform/DouglasPeucker metrics only."
        ),
    )
    parser.add_argument("--n_ships", type=int, default=24)
    parser.add_argument("--n_points", type=int, default=200)
    parser.add_argument(
        "--synthetic_route_families",
        type=int,
        default=0,
        help=(
            "Optional synthetic-data route-family count. A positive value generates ships "
            "around shared corridors for same-support query-prior runs."
        ),
    )
    parser.add_argument(
        "--min_points_per_segment",
        type=int,
        default=4,
        help="Minimum points required to keep an AIS trajectory segment.",
    )
    parser.add_argument(
        "--max_points_per_segment",
        type=int,
        default=None,
        help="Optional AIS CSV downsampling cap per trajectory segment, useful for smoke runs.",
    )
    parser.add_argument(
        "--max_time_gap_seconds",
        type=float,
        default=3600.0,
        help="Split one vessel track into new trajectory segments when consecutive points exceed this time gap. Set <=0 to disable.",
    )
    parser.add_argument(
        "--max_segments",
        type=int,
        default=None,
        help="Optional cap applied during CSV segmentation, useful for smoke runs.",
    )
    parser.add_argument(
        "--train_max_segments",
        type=int,
        default=None,
        help="Optional train CSV segment cap. Defaults to --max_segments when unset.",
    )
    parser.add_argument(
        "--validation_max_segments",
        type=int,
        default=None,
        help="Optional validation CSV segment cap. Defaults to --max_segments when unset.",
    )
    parser.add_argument(
        "--eval_max_segments",
        type=int,
        default=None,
        help="Optional eval CSV segment cap. Defaults to --max_segments when unset.",
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=None,
        help="Optional cap on loaded AIS trajectories after CSV loading, useful for smoke runs.",
    )


def _add_query_workload_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n_queries", type=int, default=128)
    parser.add_argument(
        "--query_coverage",
        type=float,
        default=None,
        help=(
            "Bias generated queries toward this point-coverage target. Final calibrated profiles "
            "treat --n_queries as a minimum floor and may expand up to --max_queries. "
            "Accepts 0.30 or 30 for 30%%."
        ),
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Optional cap for coverage-targeted query generation when it may expand beyond --n_queries.",
    )
    parser.add_argument(
        "--range_spatial_fraction",
        type=float,
        default=0.08,
        help="Range query half-width as a fraction of dataset lat/lon span. Ignored when --range_spatial_km is set.",
    )
    parser.add_argument(
        "--range_time_fraction",
        type=float,
        default=0.15,
        help="Range query half-window as a fraction of dataset time span. Ignored when --range_time_hours is set.",
    )
    parser.add_argument(
        "--range_spatial_km",
        type=float,
        default=None,
        help="Nominal range query spatial half-width in kilometers. Keeps workload scale stable across datasets.",
    )
    parser.add_argument(
        "--range_time_hours",
        type=float,
        default=None,
        help="Nominal range query temporal half-window in hours. Keeps workload scale stable across datasets.",
    )
    parser.add_argument(
        "--range_footprint_jitter",
        type=float,
        default=0.5,
        help="Random +/- fraction applied to range query spatial and temporal half-windows. 0.0 makes footprints fixed.",
    )
    parser.add_argument(
        "--range_time_domain_mode",
        type=str,
        default="dataset",
        choices=RANGE_TIME_DOMAIN_MODES,
        help=(
            "Temporal clamp domain for generated range queries. 'dataset' uses global time bounds; "
            "'anchor_day' clamps each query to the 24-hour source/calendar day containing its anchor."
        ),
    )
    parser.add_argument(
        "--range_anchor_mode",
        type=str,
        default="mixed_density",
        choices=RANGE_ANCHOR_MODES,
        help=(
            "Anchor sampling prior for generated range queries. 'mixed_density' keeps the historical "
            "70 percent density-biased / 30 percent uniform mix; 'dense', 'uniform', and 'sparse' expose held-out "
            "generator settings."
        ),
    )
    parser.add_argument(
        "--range_train_anchor_modes",
        type=_range_anchor_mode_list,
        default=[],
        help=(
            "Optional comma-separated anchor priors cycled across train workload replicates. "
            "Leave unset to use --range_anchor_mode for learning. Eval and checkpoint selection "
            "continue using --range_anchor_mode."
        ),
    )
    parser.add_argument(
        "--range_train_footprints",
        type=_range_train_footprint_list,
        default=[],
        help=(
            "Optional comma-separated train-only range footprint families as spatial_km:time_hours, "
            "cycled across train workload replicates. Eval and checkpoint selection continue using "
            "--range_spatial_km/--range_time_hours."
        ),
    )
    parser.add_argument(
        "--range_min_point_hits",
        type=int,
        default=None,
        help="Optional range-query acceptance filter: reject boxes with fewer point hits.",
    )
    parser.add_argument(
        "--range_max_point_hit_fraction",
        type=float,
        default=None,
        help="Optional range-query acceptance filter: reject boxes hitting more than this point fraction.",
    )
    parser.add_argument(
        "--range_min_trajectory_hits",
        type=int,
        default=None,
        help="Optional range-query acceptance filter: reject boxes hitting fewer trajectories.",
    )
    parser.add_argument(
        "--range_max_trajectory_hit_fraction",
        type=float,
        default=None,
        help="Optional range-query acceptance filter: reject boxes hitting more than this trajectory fraction.",
    )
    parser.add_argument(
        "--range_max_box_volume_fraction",
        type=float,
        default=None,
        help="Optional range-query acceptance filter: reject boxes with larger normalized spatiotemporal volume.",
    )
    parser.add_argument(
        "--range_duplicate_iou_threshold",
        type=float,
        default=None,
        help="Optional range-query acceptance filter: reject boxes with IoU at or above this threshold versus accepted boxes.",
    )
    parser.add_argument(
        "--range_acceptance_max_attempts",
        type=int,
        default=None,
        help="Maximum candidate range boxes to try when acceptance filters are enabled.",
    )
    parser.add_argument(
        "--range_max_coverage_overshoot",
        type=float,
        default=None,
        help=(
            "Optional target-coverage guard: reject candidate range boxes that would push union "
            "point coverage above --query_coverage plus this absolute tolerance. Accepts fractions or percents."
        ),
    )
    parser.add_argument(
        "--range_train_workload_replicates",
        type=int,
        default=1,
        help=(
            "Number of independent train-workload seeds to aggregate into blind range supervision. "
            "Eval and checkpoint-selection workloads remain separate."
        ),
    )
    parser.add_argument(
        "--workload_profile_id",
        type=str,
        default=None,
        choices=WORKLOAD_PROFILE_CHOICES,
        help=(
            "Named range workload profile. range_query_mix is the current final "
            "candidate profile and owns a 30%% target coverage. Omitting this "
            "uses raw generator settings for diagnostics."
        ),
    )
    parser.add_argument(
        "--coverage_calibration_mode",
        type=str,
        default=None,
        choices=["profile_sampled_query_count", "uncovered_anchor_chasing"],
        help=(
            "Target-coverage calibration behavior. range_query_mix defaults to "
            "profile_sampled_query_count; uncovered_anchor_chasing is legacy/diagnostic only."
        ),
    )
    parser.add_argument(
        "--workload_stability_gate_mode",
        type=str,
        default="final",
        choices=["final", "smoke"],
        help=(
            "Gate strictness for workload-stability checks. 'final' requires enough queries "
            "and healthy generation. 'smoke' is only for tiny implementation smoke tests and "
            "is never final-success evidence."
        ),
    )


def _add_model_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--embed_dim", type=int, default=64, help="Transformer hidden dimension.")
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of transformer attention heads."
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of transformer encoder layers. For workload-blind models, 0 uses an MLP-only scorer.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability used by the model."
    )
    parser.add_argument(
        "--ranking_pairs_per_type",
        type=int,
        default=96,
        help="Number of positive/negative ranking pairs sampled per query type and training window.",
    )
    parser.add_argument(
        "--ranking_top_quantile",
        type=float,
        default=0.80,
        help="Label quantile used to define top-ranked positive candidates for ranking-pair sampling.",
    )
    parser.add_argument(
        "--pointwise_loss_weight",
        type=float,
        default=0.25,
        help="Weight for balanced pointwise BCE supervision alongside the active set/ranking loss.",
    )
    parser.add_argument(
        "--loss_objective",
        type=str,
        default="budget_topk",
        choices=[
            "ranking_bce",
            "budget_topk",
            "stratified_budget_topk",
            "pointwise_bce",
        ],
        help=(
            "Training objective. 'ranking_bce' is the pairwise ranking plus BCE ablation; "
            "'budget_topk' optimizes soft retained-budget target mass across budget ratios; "
            "'stratified_budget_topk' is a slower diagnostic that optimizes the "
            "stratified selector's per-stratum choices; "
            "'pointwise_bce' directly fits every valid soft label."
        ),
    )
    parser.add_argument(
        "--budget_loss_ratios",
        type=_compression_ratio_list,
        default=list(DEFAULT_BUDGET_LOSS_RATIOS),
        help="Comma-separated retained-point ratios used by budget-aware loss objectives.",
    )
    parser.add_argument(
        "--budget_loss_temperature",
        type=float,
        default=DEFAULT_BUDGET_LOSS_TEMPERATURE,
        help="Soft top-k temperature for --loss_objective budget_topk.",
    )
    parser.add_argument(
        "--query_local_utility_aux_loss_weight",
        type=float,
        default=0.50,
        help="Overall auxiliary loss weight for QueryLocalUtility factorized heads.",
    )
    parser.add_argument(
        "--query_local_utility_segment_budget_head_weight",
        type=float,
        default=0.10,
        help="BCE head weight for the QueryLocalUtility segment-budget factorized head.",
    )
    parser.add_argument(
        "--query_local_utility_segment_level_loss_weight",
        type=float,
        default=0.25,
        help="Listwise segment-level loss weight inside the QueryLocalUtility auxiliary loss.",
    )
    parser.add_argument(
        "--query_local_utility_behavior_rank_loss_weight",
        type=float,
        default=DEFAULT_QUERY_LOCAL_UTILITY_BEHAVIOR_RANK_LOSS_WEIGHT,
        help=(
            "Listwise behavior-head ranking loss weight inside the QueryLocalUtility auxiliary "
            "loss. Set to 0.0 only for explicit behavior-head ablations."
        ),
    )
    parser.add_argument(
        "--query_local_utility_sparse_head_rank_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Optional sparse-head ranking loss weight for query-hit and boundary QueryLocalUtility "
            "heads. Default 0.0 keeps the Checkpoint 5.37 head-calibration diagnostic disabled "
            "unless explicitly requested."
        ),
    )
    parser.add_argument(
        "--query_local_utility_sparse_head_bce_target_mode",
        type=str,
        default="raw",
        choices=["raw", "window_max_normalized"],
        help=(
            "Optional BCE target calibration for sparse QueryLocalUtility query-hit and boundary "
            "heads. 'raw' preserves current targets; 'window_max_normalized' trains those "
            "heads on per-window relative target scale as a diagnostic."
        ),
    )
    parser.add_argument(
        "--query_local_utility_train_marginal_diagnostics",
        action="store_true",
        help=(
            "Emit train-split exact retained-decision marginal diagnostics for guarded "
            "segment-marginal calibration probes. This is diagnostic-only and can be "
            "expensive on larger train splits."
        ),
    )
    parser.add_argument(
        "--temporal_distribution_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Optional weight for a budget-aware temporal CDF regularizer. "
            "0.0 disables it; small values discourage clustered soft top-k selections."
        ),
    )
    parser.add_argument(
        "--gradient_clip_norm",
        type=float,
        default=1.0,
        help="Max gradient norm. Set <=0 to disable clipping.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Number of trajectory windows per training optimizer step.",
    )
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=16,
        help="Number of trajectory windows per MLQDS inference or validation-score diagnostic batch.",
    )
    parser.add_argument(
        "--query_chunk_size",
        type=int,
        default=2048,
        help=(
            "Number of workload queries attended per cross-attention chunk. "
            "Set at least --n_queries to use one exact attention softmax for the full workload."
        ),
    )
    parser.add_argument("--compression_ratio", type=float, default=0.2)
    parser.add_argument(
        "--model_type",
        type=str,
        default="baseline",
        choices=SUPPORTED_MODEL_TYPES,
    )
    parser.add_argument(
        "--historical_prior_k",
        type=int,
        default=32,
        help="Nearest-neighbor count for model_type=historical_prior.",
    )
    parser.add_argument(
        "--historical_prior_clock_weight",
        type=float,
        default=0.0,
        help="Distance weight for historical-prior circular clock-time features. 0.0 preserves density-only behavior.",
    )
    parser.add_argument(
        "--historical_prior_mmsi_weight",
        type=float,
        default=1.0,
        help=(
            "Distance weight for model_type=historical_prior_mmsi deterministic vessel-id hash features. "
            "0.0 ignores identity; larger values prefer same-MMSI historical support."
        ),
    )
    parser.add_argument(
        "--historical_prior_density_weight",
        type=float,
        default=1.0,
        help="Distance weight for historical-prior spatial density/sparsity features.",
    )
    parser.add_argument(
        "--historical_prior_min_target",
        type=float,
        default=0.0,
        help=(
            "Minimum retained-frequency target stored by model_type=historical_prior. "
            "0.0 stores all train points; larger values keep only stronger useful-point support."
        ),
    )
    parser.add_argument(
        "--historical_prior_support_ratio",
        type=float,
        default=1.0,
        help=(
            "Per-train-trajectory top-target support cap for model_type=historical_prior. "
            "1.0 stores all points that pass min-target filtering; lower values reduce dense-route dominance."
        ),
    )
    parser.add_argument(
        "--historical_prior_source_aggregation",
        type=str,
        default="none",
        choices=["none", "mean", "min", "median"],
        help=(
            "How to combine historical-prior KNN scores across explicit train CSV sources. "
            "'none' preserves the original pooled prior; 'mean', 'min', and 'median' require signal "
            "to transfer across train days."
        ),
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="range",
        choices=["range"],
        help="Query workload type for this model run. Only range is supported.",
    )


def _add_checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help=(
            "Stop training if the active checkpoint selection score does not improve for this many "
            "eligible diagnostic epochs. 0 disables."
        ),
    )
    parser.add_argument(
        "--diagnostic_every",
        type=int,
        default=1,
        help="Run training diagnostics every N epochs. Use 1 so every epoch can be selected as best.",
    )
    parser.add_argument(
        "--diagnostic_window_fraction",
        type=float,
        default=0.2,
        help="Fraction of trajectory windows used for each diagnostic pass.",
    )
    parser.add_argument(
        "--checkpoint_selection_metric",
        type=str,
        default="score",
        choices=["loss", "score", "uniform_gap"],
        help=(
            "Select restored checkpoints by held-out validation score, training loss, or validation score "
            "with fair-uniform gap penalties."
        ),
    )
    parser.add_argument(
        "--validation_score_every",
        type=int,
        default=0,
        help="Run held-out validation scoring every N epochs. 0 disables unless checkpoint selection metric is score/uniform_gap.",
    )
    parser.add_argument(
        "--checkpoint_uniform_gap_weight",
        type=float,
        default=0.5,
        help="When checkpoint_selection_metric=uniform_gap, bonus/penalty weight for aggregate gap versus uniform.",
    )
    parser.add_argument(
        "--checkpoint_type_penalty_weight",
        type=float,
        default=1.0,
        help="When checkpoint_selection_metric=uniform_gap, penalty weight for per-type validation-score deficits versus uniform.",
    )
    parser.add_argument(
        "--checkpoint_smoothing_window",
        type=int,
        default=1,
        help="Pick checkpoints by rolling-mean selection score over the last K diagnostic epochs. Reduces selection bias from noisy single-epoch validation scores. 1 = original single-epoch behavior; 5 = average over 5 latest diagnostic epochs.",
    )
    parser.add_argument(
        "--checkpoint_full_score_every",
        type=int,
        default=1,
        help="Run exact validation scoring every N eligible validation-score epochs. 1 keeps exact validation every eligible epoch.",
    )
    parser.add_argument(
        "--checkpoint_candidate_pool_size",
        type=int,
        default=1,
        help="When checkpoint_full_score_every > 1, keep this many cheap-diagnostic candidate snapshots for the next exact validation round.",
    )
    parser.add_argument(
        "--checkpoint_score_variant",
        type=str,
        default="query_local_utility",
        choices=["answer", "combined", "query_local_utility"],
        help=(
            "Which validation score to use for checkpoint selection. "
            "'query_local_utility' = query-driven primary score for range_query_mix, "
            "'answer' = point/query F1, 'combined' = answer_f1 * point_subset_f1."
        ),
    )
    parser.add_argument(
        "--validation_global_sanity_penalty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply a light validation-only penalty to query_local_utility checkpoint selection when global sanity fails.",
    )
    parser.add_argument(
        "--validation_global_sanity_penalty_weight",
        type=float,
        default=DEFAULT_VALIDATION_GLOBAL_SANITY_PENALTY_WEIGHT,
        help="Validation checkpoint penalty weight for length preservation shortfall.",
    )
    parser.add_argument(
        "--validation_sed_penalty_weight",
        type=float,
        default=DEFAULT_VALIDATION_SED_PENALTY_WEIGHT,
        help="Validation checkpoint penalty weight for SED ratio above the final sanity threshold.",
    )
    parser.add_argument(
        "--validation_endpoint_penalty_weight",
        type=float,
        default=DEFAULT_VALIDATION_ENDPOINT_PENALTY_WEIGHT,
        help="Validation checkpoint penalty weight for endpoint retention failures.",
    )
    parser.add_argument(
        "--validation_length_preservation_min",
        type=float,
        default=DEFAULT_VALIDATION_LENGTH_PRESERVATION_MIN,
        help="Minimum validation length preservation used by query_local_utility checkpoint penalties.",
    )


def _add_runtime_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--range_audit_compression_ratios",
        type=_compression_ratio_list,
        default=None,
        help=(
            "Optional comma-separated retained-point ratios for a multi-budget range audit, "
            "for example 0.01,0.02,0.05,0.10. Disabled by default because it reruns method scoring."
        ),
    )
    parser.add_argument(
        "--float32_matmul_precision",
        type=str,
        default="highest",
        choices=FLOAT32_MATMUL_PRECISION_CHOICES,
        help="Torch float32 matmul precision. Use 'high' with --allow_tf32 for TF32 benchmarking.",
    )
    parser.add_argument(
        "--allow_tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow TF32 for CUDA float32 matmul. Defaults off for baseline comparability.",
    )
    parser.add_argument(
        "--amp_mode",
        choices=AMP_MODE_CHOICES,
        default="off",
        help="Optional CUDA autocast mode for model forward passes. Losses and diagnostics stay in FP32.",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default=None,
        help="Path to save trained model checkpoint (.pt). Disabled if not provided.",
    )
    parser.add_argument(
        "--save_queries_dir",
        type=str,
        default=None,
        help="Directory to save eval-workload queries as one GeoJSON per query type.",
    )
    parser.add_argument(
        "--save_simplified_dir",
        type=str,
        default=None,
        help="Directory to save MLQDS simplified trajectories as CSV.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build run CLI parser. See orchestration/README.md for details."""
    parser = argparse.ArgumentParser(description="Run AIS-QDS learning/scoring.")
    _add_data_source_arguments(parser)
    _add_query_workload_arguments(parser)
    _add_model_training_arguments(parser)
    _add_checkpoint_arguments(parser)
    add_selector_and_range_target_arguments(parser)
    _add_runtime_output_arguments(parser)
    return parser
