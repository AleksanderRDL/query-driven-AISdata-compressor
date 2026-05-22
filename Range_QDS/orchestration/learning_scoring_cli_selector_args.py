"""Selector and range-target CLI argument sections."""

from __future__ import annotations

import argparse

from config.run_config import (
    DEFAULT_LEARNED_SEGMENT_ALLOCATION_LENGTH_SUPPORT_WEIGHT,
    DEFAULT_LEARNED_SEGMENT_ALLOCATION_WEIGHT_FLOOR,
    DEFAULT_LEARNED_SEGMENT_GEOMETRY_GAIN_WEIGHT,
    DEFAULT_LEARNED_SEGMENT_SCORE_BLEND_WEIGHT,
    DEFAULT_LEARNED_SEGMENT_TRANSFER_CALIBRATION_MODE,
)
from learning.importance_labels import RANGE_LABEL_MODES
from learning.targets.modes import RANGE_TARGET_BALANCE_MODES, RANGE_TRAINING_TARGET_MODES
from learning.teacher_distillation import RANGE_TEACHER_DISTILLATION_MODES
from selection.learned_segment_budget import SEGMENT_TRANSFER_CALIBRATION_MODE_CHOICES
from selection.model_score_conversion import MLQDS_SCORE_MODES
from selection.selector_types import SELECTOR_TYPE_CHOICES, TEMPORAL_HYBRID_SELECTOR_TYPE


def _add_selector_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mlqds_temporal_fraction",
        type=float,
        default=0.0,
        help="Fraction of the retained budget reserved for evenly spaced temporal base points before MLQDS score fill. Default 0.0 = pure learned scoring; raise to add a uniform spine.",
    )
    parser.add_argument(
        "--mlqds_diversity_bonus",
        type=float,
        default=0.0,
        help=(
            "Spacing bonus for MLQDS fill/swap/local_swap/local_delta_swap candidates away from temporal base points. "
            "Ignored by mlqds_hybrid_mode=stratified or global_budget."
        ),
    )
    parser.add_argument(
        "--mlqds_hybrid_mode",
        type=str,
        default="fill",
        choices=[
            "fill",
            "swap",
            "local_swap",
            "local_delta_swap",
            "stratified",
            "global_fill",
            "global_budget",
        ],
        help=(
            "How temporal scaffolding and learned scores are combined. "
            "'fill' reserves part of the budget for a temporal spine, then fills the rest. "
            "'swap' starts from full uniform temporal sampling and replaces only the unprotected budget share. "
            "'local_swap' pairs each learned addition with the nearest unprotected temporal-base removal. "
            "'local_delta_swap' performs that local replacement only when the learned score improves over the paired base point. "
            "'stratified' selects the highest learned-score point inside each temporal/index stratum. "
            "'global_fill' keeps a temporal base per trajectory, then spends residual budget globally by learned score. "
            "'global_budget' keeps endpoint skeletons, then spends remaining budget globally by learned score."
        ),
    )
    parser.add_argument(
        "--selector_type",
        type=str,
        default=TEMPORAL_HYBRID_SELECTOR_TYPE,
        choices=SELECTOR_TYPE_CHOICES,
        help=(
            "Retained-mask selector. Use learned_segment_budget for query-driven final-candidate runs; "
            "temporal_hybrid is diagnostic-only selector behavior."
        ),
    )
    parser.add_argument(
        "--learned_segment_geometry_gain_weight",
        type=float,
        default=DEFAULT_LEARNED_SEGMENT_GEOMETRY_GAIN_WEIGHT,
        help=(
            "Geometry-gain tie-breaker weight for learned_segment_budget. "
            "This is query-free selector structure for within-segment point choice "
            "and is reported for causality audits."
        ),
    )
    parser.add_argument(
        "--learned_segment_allocation_length_support_weight",
        type=float,
        default=DEFAULT_LEARNED_SEGMENT_ALLOCATION_LENGTH_SUPPORT_WEIGHT,
        help=(
            "Query-free path-length support blend weight for learned_segment_budget "
            "segment allocation. This is separate from the within-segment "
            "geometry-gain tie-breaker."
        ),
    )
    parser.add_argument(
        "--learned_segment_allocation_weight_floor",
        type=float,
        default=DEFAULT_LEARNED_SEGMENT_ALLOCATION_WEIGHT_FLOOR,
        help=(
            "Base positive floor added to normalized learned_segment_budget "
            "segment allocation weights. Lower values increase score contrast; "
            "the default preserves current selector behavior."
        ),
    )
    parser.add_argument(
        "--learned_segment_score_blend_weight",
        type=float,
        default=DEFAULT_LEARNED_SEGMENT_SCORE_BLEND_WEIGHT,
        help=(
            "Within-segment blend weight for the segment-budget head in "
            "learned_segment_budget. Exposed so it cannot silently mask weak heads."
        ),
    )
    parser.add_argument(
        "--learned_segment_transfer_calibration_mode",
        type=str,
        default=DEFAULT_LEARNED_SEGMENT_TRANSFER_CALIBRATION_MODE,
        choices=SEGMENT_TRANSFER_CALIBRATION_MODE_CHOICES,
        help=(
            "Guarded non-default pre-selection segment transfer calibration for "
            "learned_segment_budget. 'none' preserves current behavior; "
            "segment_score_allocation_weight_zblend is diagnostic until unchanged "
            "strict gates justify it."
        ),
    )
    parser.add_argument(
        "--disable_learned_segment_fairness_preallocation",
        dest="learned_segment_fairness_preallocation",
        action="store_false",
        default=True,
        help="Disable the query-free one-learned-slot-per-active-trajectory selector preallocation.",
    )
    parser.add_argument(
        "--learned_segment_length_repair_fraction",
        type=float,
        default=0.0,
        help=(
            "Optional query-free learned-slot repair fraction for learned_segment_budget. "
            "0.0 leaves selected learned slots unchanged; values above 0 swap a bounded share "
            "toward path-length gain and are experimental until strict gates pass."
        ),
    )
    parser.add_argument(
        "--learned_segment_length_repair_score_protection_fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of the total retained budget protected from length-repair removal "
            "by top learned score. Query-free diagnostic control; 0.0 preserves current "
            "repair behavior."
        ),
    )
    parser.add_argument(
        "--learned_segment_length_support_blend_weight",
        type=float,
        default=0.0,
        help=(
            "Optional learned-segment allocation blend weight for the query-free "
            "path_length_support_target head. 0.0 keeps segment-budget allocation; "
            "1.0 uses the length-support head as the segment allocation signal."
        ),
    )
    parser.add_argument(
        "--query_prior_grid_bins",
        type=int,
        default=64,
        help=(
            "Lat/lon grid resolution for train-derived workload-blind query-prior fields. "
            "Higher values preserve sharper workload-local priors but can overfit small train splits."
        ),
    )
    parser.add_argument(
        "--query_prior_smoothing_passes",
        type=int,
        default=2,
        help=(
            "Number of binomial smoothing passes for train-derived query-prior fields. "
            "Set 0 for unsmoothed priors when strict predictability diagnostics justify it."
        ),
    )
    parser.add_argument(
        "--mlqds_stratified_center_weight",
        type=float,
        default=0.0,
        help=(
            "Optional center-distance penalty inside stratified learned-score bins. "
            "0.0 keeps pure learned top-score selection within each bin."
        ),
    )
    parser.add_argument(
        "--mlqds_min_learned_swaps",
        type=int,
        default=0,
        help=(
            "Diagnostic lower bound on learned replacements per trajectory for swap/local_swap/local_delta_swap modes. "
            "Default 0 preserves temporal-fraction rounding exactly."
        ),
    )
    parser.add_argument(
        "--mlqds_score_mode",
        type=str,
        default="rank",
        choices=MLQDS_SCORE_MODES,
        help="Convert pure workload logits to simplification scores using per-trajectory ranks, sigmoid logits, or raw logits.",
    )
    parser.add_argument(
        "--mlqds_score_temperature",
        type=float,
        default=1.0,
        help="Temperature for temperature_sigmoid, zscore_sigmoid, and rank_confidence score modes.",
    )
    parser.add_argument(
        "--mlqds_rank_confidence_weight",
        type=float,
        default=0.15,
        help="Blend weight for rank_confidence score mode. 0.0=pure rank, 1.0=pure zscore sigmoid.",
    )
    parser.add_argument(
        "--mlqds_range_geometry_blend",
        type=float,
        default=0.0,
        help=(
            "Blend model scores with cached range-geometry scores before MLQDS retention. "
            "0.0 uses model scores only; 1.0 uses range-geometry labels only."
        ),
    )
    parser.add_argument(
        "--temporal_residual_label_mode",
        type=str,
        default="none",
        choices=["none", "temporal"],
        help=(
            "Use labels directly, or explicitly train only on points not already kept by the temporal base. "
            "Default 'none' avoids accidental residual-only training in direct manual runs."
        ),
    )


def _add_range_target_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--range_label_mode",
        type=str,
        default="point_f1",
        choices=RANGE_LABEL_MODES,
        help=(
            "Range label construction mode. 'point_f1' assigns expected in-box point-F1 "
            "contribution. QueryLocalUtility training uses its dedicated factorized target mode."
        ),
    )
    parser.add_argument(
        "--range_training_target_mode",
        type=str,
        default="point_value",
        choices=RANGE_TRAINING_TARGET_MODES,
        help=(
            "Transform training range labels before fitting the model. 'point_value' uses the raw "
            "expected point-value labels; 'retained_frequency' trains on oracle retained-set membership "
            "frequency across configured budgets; 'global_budget_retained_frequency' trains on "
            "training-only global-budget oracle membership; 'historical_prior_retained_frequency' "
            "distills that target through a leave-one-out query-free historical KNN teacher; "
            "'marginal_coverage_frequency' trains on set-aware "
            "neighborhood coverage targets; 'query_spine_frequency' trains on query-derived temporal "
            "support anchors; 'query_residual_frequency' trains on train-query residual fill anchors; "
            "'set_utility_frequency' trains on one-step train-query QueryLocalUtility gain; "
            "'local_swap_utility_frequency' trains on one-step train-query local-swap QueryLocalUtility gain; "
            "'local_swap_gain_cost_frequency' trains local-delta candidate value against removal cost; "
            "'structural_retained_frequency' blends train workload target labels with query-free "
            "globality/uniqueness scores."
        ),
    )
    parser.add_argument(
        "--range_target_balance_mode",
        type=str,
        default="none",
        choices=RANGE_TARGET_BALANCE_MODES,
        help=(
            "Optional training-only range target mass rebalance. 'trajectory_unit_mass' rescales "
            "each train trajectory's positive range-target mass to one before fitting a blind prior."
        ),
    )
    parser.add_argument(
        "--range_replicate_target_aggregation",
        type=str,
        default="label_mean",
        choices=["label_mean", "label_max", "frequency_mean"],
        help=(
            "How multiple train range workloads become retained-frequency targets. "
            "'label_mean' averages raw point-value labels before target selection; "
            "'label_max' takes the max raw point-value label before target selection; "
            "'frequency_mean' averages per-workload retained-frequency targets."
        ),
    )
    parser.add_argument(
        "--range_temporal_target_blend",
        type=float,
        default=0.0,
        help=(
            "Blend a query-blind uniform temporal retained-frequency target into retained-frequency "
            "range supervision. This changes training labels only; inference temporal_fraction is separate."
        ),
    )
    parser.add_argument(
        "--range_structural_target_blend",
        type=float,
        default=0.25,
        help=(
            "Training-only blend weight for structural_retained_frequency. "
            "0.0 uses train workload target labels only; 1.0 uses query-free structural scores only."
        ),
    )
    parser.add_argument(
        "--range_structural_target_source_mode",
        type=str,
        default="blend",
        choices=["blend", "boost"],
        help=(
            "Source-score mode for structural_retained_frequency. 'blend' adds structural score support; "
            "'boost' only re-ranks train-useful points by structural prominence."
        ),
    )
    parser.add_argument(
        "--range_target_budget_weight_power",
        type=float,
        default=0.0,
        help=(
            "Training-only weighting for retained-frequency target budgets. "
            "0.0 averages configured budgets uniformly; positive values weight smaller compression ratios "
            "as ratio ** -power before normalization."
        ),
    )
    parser.add_argument(
        "--range_marginal_target_radius_scale",
        type=float,
        default=0.50,
        help=(
            "Neighborhood radius, as a fraction of target point spacing, for "
            "range_training_target_mode=marginal_coverage_frequency."
        ),
    )
    parser.add_argument(
        "--range_query_spine_fraction",
        type=float,
        default=0.10,
        help=(
            "Fraction of each in-query trajectory slice used as temporal support anchors for "
            "range_training_target_mode=query_spine_frequency."
        ),
    )
    parser.add_argument(
        "--range_query_spine_mass_mode",
        type=str,
        default="hit_group",
        choices=["hit_group", "query"],
        help=(
            "Mass normalization for query_spine_frequency labels. 'hit_group' preserves the old behavior: "
            "each train query/trajectory-hit group gets unit mass before averaging queries. 'query' gives each "
            "train query unit mass split across its hit trajectories."
        ),
    )
    parser.add_argument(
        "--range_query_residual_multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to budget_ratio * in-query point count when building "
            "range_training_target_mode=query_residual_frequency labels."
        ),
    )
    parser.add_argument(
        "--range_query_residual_mass_mode",
        type=str,
        default="query",
        choices=["query", "point"],
        help=(
            "Mass normalization for query_residual_frequency labels. 'query' gives each train query "
            "unit mass; 'point' gives each selected residual anchor unit mass before averaging queries."
        ),
    )
    parser.add_argument(
        "--range_set_utility_multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to budget_ratio * train-query hit count when building "
            "range_training_target_mode=set_utility_frequency, local_swap_utility_frequency, "
            "or local_swap_gain_cost_frequency labels."
        ),
    )
    parser.add_argument(
        "--range_set_utility_candidate_limit",
        type=int,
        default=128,
        help=(
            "Maximum candidates per train query and budget for one-step marginal QueryLocalUtility scoring. "
            "Used by set_utility_frequency, local_swap_utility_frequency, and "
            "local_swap_gain_cost_frequency. Use 0 to score every candidate."
        ),
    )
    parser.add_argument(
        "--range_set_utility_mass_mode",
        type=str,
        default="gain",
        choices=["gain", "point", "query"],
        help=(
            "Target mass for set_utility_frequency/local_swap_utility_frequency/"
            "local_swap_gain_cost_frequency labels: raw marginal gain/value, selected-point frequency, "
            "or query-equal selected-point frequency."
        ),
    )
    parser.add_argument(
        "--range_boundary_prior_weight",
        type=float,
        default=0.0,
        help=(
            "Optional range-label boundary prior. 0.0 keeps pure point-F1 labels; "
            "1.0 gives in-box boundary-crossing points 2x raw weight before normalization."
        ),
    )


def _add_teacher_distillation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--range_teacher_distillation_mode",
        type=str,
        default="none",
        choices=RANGE_TEACHER_DISTILLATION_MODES,
        help=(
            "Train a query-aware range teacher and convert its train-workload signal into query-blind "
            "student labels. 'rank_percentile' uses teacher rank labels; 'retained_frequency' uses "
            "teacher retained-set membership across budget ratios."
        ),
    )
    parser.add_argument(
        "--range_teacher_epochs",
        type=int,
        default=4,
        help="Epochs for the query-aware range teacher when teacher distillation is enabled.",
    )


def add_selector_and_range_target_arguments(parser: argparse.ArgumentParser) -> None:
    """Add selector controls and range-target training controls."""
    _add_selector_arguments(parser)
    _add_range_target_arguments(parser)
    _add_teacher_distillation_arguments(parser)
