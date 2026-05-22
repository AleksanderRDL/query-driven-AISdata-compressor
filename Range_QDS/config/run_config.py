"""Shared run configuration dataclasses. See config/README.md for details."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, TypedDict, Unpack

from scoring.geometry_thresholds import FINAL_LENGTH_PRESERVATION_MIN
from selection.learned_segment_budget.constants import (
    GEOMETRY_TIE_BREAKER_WEIGHT,
    SEGMENT_ALLOCATION_WEIGHT_FLOOR,
    SEGMENT_LENGTH_SUPPORT_ALLOCATION_WEIGHT,
    SEGMENT_SCORE_POINT_BLEND_WEIGHT,
    SEGMENT_TRANSFER_CALIBRATION_MODE_NONE,
)
from selection.selector_types import TEMPORAL_HYBRID_SELECTOR_TYPE

LCG_MULTIPLIER = 6364136223846793005
DEFAULT_BUDGET_LOSS_RATIOS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
DEFAULT_BUDGET_LOSS_TEMPERATURE = 0.25
DEFAULT_LEARNED_SEGMENT_GEOMETRY_GAIN_WEIGHT = GEOMETRY_TIE_BREAKER_WEIGHT
DEFAULT_LEARNED_SEGMENT_ALLOCATION_LENGTH_SUPPORT_WEIGHT = SEGMENT_LENGTH_SUPPORT_ALLOCATION_WEIGHT
DEFAULT_LEARNED_SEGMENT_ALLOCATION_WEIGHT_FLOOR = SEGMENT_ALLOCATION_WEIGHT_FLOOR
DEFAULT_LEARNED_SEGMENT_SCORE_BLEND_WEIGHT = SEGMENT_SCORE_POINT_BLEND_WEIGHT
DEFAULT_LEARNED_SEGMENT_TRANSFER_CALIBRATION_MODE = SEGMENT_TRANSFER_CALIBRATION_MODE_NONE
DEFAULT_VALIDATION_GLOBAL_SANITY_PENALTY_WEIGHT = 0.10
DEFAULT_VALIDATION_SED_PENALTY_WEIGHT = 0.05
DEFAULT_VALIDATION_ENDPOINT_PENALTY_WEIGHT = 0.10
DEFAULT_VALIDATION_LENGTH_PRESERVATION_MIN = FINAL_LENGTH_PRESERVATION_MIN
DEFAULT_QUERY_LOCAL_UTILITY_BEHAVIOR_RANK_LOSS_WEIGHT = 0.25
VALIDATION_SPLIT_MODES = ("random", "source_stratified")


@dataclass
class DataConfig:
    """Data loading and splitting configuration. See data_preparation/README.md for details."""

    n_ships: int | None = 24
    n_points_per_ship: int | None = 200
    synthetic_route_families: int = 0
    min_points_per_segment: int = 4
    max_points_per_segment: int | None = None
    max_time_gap_seconds: float | None = 3600.0
    max_segments: int | None = None
    train_max_segments: int | None = None
    validation_max_segments: int | None = None
    eval_max_segments: int | None = None
    max_trajectories: int | None = None
    csv_path: str | None = None
    train_csv_path: str | None = None
    validation_csv_path: str | None = None
    eval_csv_path: str | None = None
    cache_dir: str | None = None
    refresh_cache: bool = False
    range_diagnostics_mode: str = "full"
    seed: int = 42
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    validation_split_mode: str = "random"

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary. See orchestration/README.md for details."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataConfig:
        """Deserialize config from a dictionary. See orchestration/README.md for details."""
        return cls(**data)


@dataclass
class QueryConfig:
    """Query generation and pure workload configuration. See workloads/README.md for details."""

    n_queries: int = 128
    target_coverage: float | None = None
    max_queries: int | None = None
    range_spatial_fraction: float = 0.08
    range_time_fraction: float = 0.15
    range_spatial_km: float | None = None
    range_time_hours: float | None = None
    range_footprint_jitter: float = 0.5
    range_time_domain_mode: str = "dataset"
    range_anchor_mode: str = "mixed_density"
    range_train_anchor_modes: list[str] = field(default_factory=list)
    range_train_footprints: list[str] = field(default_factory=list)
    workload: str = "range"
    range_min_point_hits: int | None = None
    range_max_point_hit_fraction: float | None = None
    range_min_trajectory_hits: int | None = None
    range_max_trajectory_hit_fraction: float | None = None
    range_max_box_volume_fraction: float | None = None
    range_duplicate_iou_threshold: float | None = None
    range_acceptance_max_attempts: int | None = None
    range_max_coverage_overshoot: float | None = None
    range_train_workload_replicates: int = 1
    workload_profile_id: str | None = None
    coverage_calibration_mode: str | None = None
    workload_stability_gate_mode: str = "final"

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary. See orchestration/README.md for details."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueryConfig:
        """Deserialize config from a dictionary. See orchestration/README.md for details."""
        payload = dict(data)
        raw_train_anchor_modes = payload.get("range_train_anchor_modes", [])
        if isinstance(raw_train_anchor_modes, str):
            payload["range_train_anchor_modes"] = [
                item.strip() for item in raw_train_anchor_modes.split(",") if item.strip()
            ]
        elif raw_train_anchor_modes is None:
            payload["range_train_anchor_modes"] = []
        else:
            payload["range_train_anchor_modes"] = list(raw_train_anchor_modes)
        raw_train_footprints = payload.get("range_train_footprints", [])
        if isinstance(raw_train_footprints, str):
            payload["range_train_footprints"] = [
                item.strip() for item in raw_train_footprints.split(",") if item.strip()
            ]
        elif raw_train_footprints is None:
            payload["range_train_footprints"] = []
        else:
            payload["range_train_footprints"] = list(raw_train_footprints)
        return cls(**payload)


@dataclass
class ModelConfig:
    """Model architecture and training behavior config. See models/README.md for details."""

    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 3
    type_embed_dim: int = 16
    query_chunk_size: int = 2048
    dropout: float = 0.1
    window_length: int = 512
    window_stride: int = 256
    epochs: int = 6
    lr: float = 5e-4
    compression_ratio: float = 0.2
    model_type: str = "baseline"
    historical_prior_k: int = 32
    historical_prior_clock_weight: float = 0.0
    historical_prior_mmsi_weight: float = 1.0
    historical_prior_density_weight: float = 1.0
    historical_prior_min_target: float = 0.0
    historical_prior_support_ratio: float = 1.0
    historical_prior_source_aggregation: str = "none"
    rank_margin: float = 0.05
    ranking_pairs_per_type: int = 96
    ranking_top_quantile: float = 0.80
    pointwise_loss_weight: float = 0.25
    loss_objective: str = "budget_topk"
    budget_loss_ratios: list[float] = field(
        default_factory=lambda: list(DEFAULT_BUDGET_LOSS_RATIOS)
    )
    budget_loss_temperature: float = DEFAULT_BUDGET_LOSS_TEMPERATURE
    query_local_utility_aux_loss_weight: float = 0.50
    query_local_utility_segment_budget_head_weight: float = 0.10
    query_local_utility_segment_level_loss_weight: float = 0.25
    query_local_utility_behavior_rank_loss_weight: float = (
        DEFAULT_QUERY_LOCAL_UTILITY_BEHAVIOR_RANK_LOSS_WEIGHT
    )
    query_local_utility_sparse_head_rank_loss_weight: float = 0.0
    query_local_utility_sparse_head_bce_target_mode: str = "raw"
    query_local_utility_train_marginal_diagnostics: bool = False
    temporal_distribution_loss_weight: float = 0.0
    gradient_clip_norm: float = 1.0
    l2_score_weight: float = 1e-4
    early_stopping_patience: int = 0
    train_batch_size: int = 16
    inference_batch_size: int = 16
    diagnostic_every: int = 1
    diagnostic_window_fraction: float = 0.2
    checkpoint_selection_metric: str = "score"
    validation_score_every: int = 0
    checkpoint_uniform_gap_weight: float = 0.5
    checkpoint_type_penalty_weight: float = 1.0
    checkpoint_smoothing_window: int = 1
    checkpoint_full_score_every: int = 1
    checkpoint_candidate_pool_size: int = 1
    checkpoint_score_variant: str = "query_local_utility"
    validation_global_sanity_penalty_enabled: bool = True
    validation_global_sanity_penalty_weight: float = DEFAULT_VALIDATION_GLOBAL_SANITY_PENALTY_WEIGHT
    validation_sed_penalty_weight: float = DEFAULT_VALIDATION_SED_PENALTY_WEIGHT
    validation_endpoint_penalty_weight: float = DEFAULT_VALIDATION_ENDPOINT_PENALTY_WEIGHT
    validation_length_preservation_min: float = DEFAULT_VALIDATION_LENGTH_PRESERVATION_MIN
    mlqds_temporal_fraction: float = 0.0
    mlqds_diversity_bonus: float = 0.0
    mlqds_hybrid_mode: str = "fill"
    selector_type: str = TEMPORAL_HYBRID_SELECTOR_TYPE
    learned_segment_geometry_gain_weight: float = DEFAULT_LEARNED_SEGMENT_GEOMETRY_GAIN_WEIGHT
    learned_segment_allocation_length_support_weight: float = (
        DEFAULT_LEARNED_SEGMENT_ALLOCATION_LENGTH_SUPPORT_WEIGHT
    )
    learned_segment_allocation_weight_floor: float = DEFAULT_LEARNED_SEGMENT_ALLOCATION_WEIGHT_FLOOR
    learned_segment_score_blend_weight: float = DEFAULT_LEARNED_SEGMENT_SCORE_BLEND_WEIGHT
    learned_segment_transfer_calibration_mode: str = (
        DEFAULT_LEARNED_SEGMENT_TRANSFER_CALIBRATION_MODE
    )
    learned_segment_fairness_preallocation: bool = True
    learned_segment_length_repair_fraction: float = 0.0
    learned_segment_length_repair_score_protection_fraction: float = 0.0
    learned_segment_length_support_blend_weight: float = 0.0
    mlqds_stratified_center_weight: float = 0.0
    mlqds_min_learned_swaps: int = 0
    mlqds_score_mode: str = "rank"
    mlqds_score_temperature: float = 1.0
    mlqds_rank_confidence_weight: float = 0.15
    mlqds_range_geometry_blend: float = 0.0
    temporal_residual_label_mode: str = "none"
    range_label_mode: str = "point_f1"
    range_training_target_mode: str = "point_value"
    range_target_balance_mode: str = "none"
    range_replicate_target_aggregation: str = "label_mean"
    range_temporal_target_blend: float = 0.0
    range_structural_target_blend: float = 0.25
    range_structural_target_source_mode: str = "blend"
    range_target_budget_weight_power: float = 0.0
    range_marginal_target_radius_scale: float = 0.50
    range_query_spine_fraction: float = 0.10
    range_query_spine_mass_mode: str = "hit_group"
    range_query_residual_multiplier: float = 1.0
    range_query_residual_mass_mode: str = "query"
    range_set_utility_multiplier: float = 1.0
    range_set_utility_candidate_limit: int = 128
    range_set_utility_mass_mode: str = "gain"
    range_boundary_prior_weight: float = 0.0
    range_teacher_distillation_mode: str = "none"
    range_teacher_epochs: int = 4
    range_audit_compression_ratios: list[float] = field(default_factory=list)
    query_prior_grid_bins: int = 64
    query_prior_smoothing_passes: int = 2
    float32_matmul_precision: str = "highest"
    allow_tf32: bool = False
    amp_mode: str = "off"

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary. See orchestration/README.md for details."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Deserialize config from a dictionary. See orchestration/README.md for details."""
        return cls(**data)


@dataclass
class BaselineConfig:
    """Baseline methods configuration. See scoring/README.md for details."""

    include_oracle: bool = True
    final_metrics_mode: str = "diagnostic"

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary. See orchestration/README.md for details."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineConfig:
        """Deserialize config from a dictionary. See orchestration/README.md for details."""
        return cls(**data)


@dataclass
class RunConfig:
    """Top-level run config container. See orchestration/README.md for details."""

    data: DataConfig = field(default_factory=DataConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary. See orchestration/README.md for details."""
        return {
            "data": self.data.to_dict(),
            "query": self.query.to_dict(),
            "model": self.model.to_dict(),
            "baselines": self.baselines.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        """Deserialize config from a dictionary. See orchestration/README.md for details."""
        expected_keys = {"data", "query", "model", "baselines"}
        unknown_keys = set(data) - expected_keys
        if unknown_keys:
            raise TypeError(f"Unknown RunConfig keys: {sorted(unknown_keys)}")
        return cls(
            data=DataConfig.from_dict(data["data"]),
            query=QueryConfig.from_dict(data["query"]),
            model=ModelConfig.from_dict(data["model"]),
            baselines=BaselineConfig.from_dict(data["baselines"]),
        )


@dataclass
class SeedBundle:
    """Derived deterministic sub-seeds for run stages. See orchestration/README.md for details."""

    split_seed: int
    train_query_seed: int
    eval_query_seed: int
    torch_seed: int


class RunConfigOverrides(TypedDict, total=False):
    """Flat keyword overrides accepted by ``build_run_config``."""

    n_ships: int
    n_points: int
    synthetic_route_families: int
    min_points_per_segment: int
    max_points_per_segment: int | None
    max_time_gap_seconds: float | None
    max_segments: int | None
    train_max_segments: int | None
    validation_max_segments: int | None
    eval_max_segments: int | None
    max_trajectories: int | None
    train_fraction: float
    val_fraction: float
    n_queries: int
    query_coverage: float | None
    max_queries: int | None
    range_spatial_fraction: float
    range_time_fraction: float
    range_spatial_km: float | None
    range_time_hours: float | None
    range_footprint_jitter: float
    range_time_domain_mode: str
    range_anchor_mode: str
    range_train_anchor_modes: list[str] | None
    range_train_footprints: list[str] | None
    range_min_point_hits: int | None
    range_max_point_hit_fraction: float | None
    range_min_trajectory_hits: int | None
    range_max_trajectory_hit_fraction: float | None
    range_max_box_volume_fraction: float | None
    range_duplicate_iou_threshold: float | None
    range_acceptance_max_attempts: int | None
    range_max_coverage_overshoot: float | None
    range_train_workload_replicates: int
    workload_profile_id: str | None
    coverage_calibration_mode: str | None
    workload_stability_gate_mode: str
    epochs: int
    lr: float
    embed_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    ranking_pairs_per_type: int
    ranking_top_quantile: float
    pointwise_loss_weight: float
    loss_objective: str
    budget_loss_ratios: list[float] | None
    budget_loss_temperature: float
    query_local_utility_aux_loss_weight: float
    query_local_utility_segment_budget_head_weight: float
    query_local_utility_segment_level_loss_weight: float
    query_local_utility_behavior_rank_loss_weight: float
    query_local_utility_sparse_head_rank_loss_weight: float
    query_local_utility_sparse_head_bce_target_mode: str
    query_local_utility_train_marginal_diagnostics: bool
    temporal_distribution_loss_weight: float
    gradient_clip_norm: float
    compression_ratio: float
    csv_path: str | None
    train_csv_path: str | None
    validation_csv_path: str | None
    eval_csv_path: str | None
    cache_dir: str | None
    refresh_cache: bool
    range_diagnostics_mode: str
    validation_split_mode: str
    model_type: str
    historical_prior_k: int
    historical_prior_clock_weight: float
    historical_prior_mmsi_weight: float
    historical_prior_density_weight: float
    historical_prior_min_target: float
    historical_prior_support_ratio: float
    historical_prior_source_aggregation: str
    workload: str
    seed: int
    early_stopping_patience: int
    train_batch_size: int
    inference_batch_size: int
    query_chunk_size: int
    diagnostic_every: int
    diagnostic_window_fraction: float
    checkpoint_selection_metric: str
    validation_score_every: int | None
    checkpoint_uniform_gap_weight: float
    checkpoint_type_penalty_weight: float
    checkpoint_smoothing_window: int
    checkpoint_full_score_every: int | None
    checkpoint_candidate_pool_size: int
    checkpoint_score_variant: str | None
    validation_global_sanity_penalty_enabled: bool
    validation_global_sanity_penalty_weight: float
    validation_sed_penalty_weight: float
    validation_endpoint_penalty_weight: float
    validation_length_preservation_min: float
    mlqds_temporal_fraction: float
    mlqds_diversity_bonus: float
    mlqds_hybrid_mode: str
    selector_type: str
    learned_segment_geometry_gain_weight: float
    learned_segment_allocation_length_support_weight: float
    learned_segment_allocation_weight_floor: float
    learned_segment_score_blend_weight: float
    learned_segment_transfer_calibration_mode: str
    learned_segment_fairness_preallocation: bool
    learned_segment_length_repair_fraction: float
    learned_segment_length_repair_score_protection_fraction: float
    learned_segment_length_support_blend_weight: float
    mlqds_stratified_center_weight: float
    mlqds_min_learned_swaps: int
    mlqds_score_mode: str
    mlqds_score_temperature: float
    mlqds_rank_confidence_weight: float
    mlqds_range_geometry_blend: float
    temporal_residual_label_mode: str | None
    range_label_mode: str
    range_training_target_mode: str
    range_target_balance_mode: str
    range_replicate_target_aggregation: str
    range_temporal_target_blend: float
    range_structural_target_blend: float
    range_structural_target_source_mode: str
    range_target_budget_weight_power: float
    range_marginal_target_radius_scale: float
    range_query_spine_fraction: float
    range_query_spine_mass_mode: str
    range_query_residual_multiplier: float
    range_query_residual_mass_mode: str
    range_set_utility_multiplier: float
    range_set_utility_candidate_limit: int
    range_set_utility_mass_mode: str
    range_boundary_prior_weight: float
    range_teacher_distillation_mode: str
    range_teacher_epochs: int
    range_audit_compression_ratios: list[float] | None
    query_prior_grid_bins: int
    query_prior_smoothing_passes: int
    final_metrics_mode: str
    float32_matmul_precision: str
    allow_tf32: bool
    amp_mode: str


RUN_CONFIG_DEFAULT_OVERRIDES: dict[str, Any] = {
    "n_ships": 24,
    "n_points": 200,
    "synthetic_route_families": 0,
    "min_points_per_segment": 4,
    "max_points_per_segment": None,
    "max_time_gap_seconds": 3600.0,
    "max_segments": None,
    "train_max_segments": None,
    "validation_max_segments": None,
    "eval_max_segments": None,
    "max_trajectories": None,
    "train_fraction": 0.70,
    "val_fraction": 0.15,
    "n_queries": 128,
    "query_coverage": None,
    "max_queries": None,
    "range_spatial_fraction": 0.08,
    "range_time_fraction": 0.15,
    "range_spatial_km": None,
    "range_time_hours": None,
    "range_footprint_jitter": 0.5,
    "range_time_domain_mode": "dataset",
    "range_anchor_mode": "mixed_density",
    "range_train_anchor_modes": None,
    "range_train_footprints": None,
    "range_min_point_hits": None,
    "range_max_point_hit_fraction": None,
    "range_min_trajectory_hits": None,
    "range_max_trajectory_hit_fraction": None,
    "range_max_box_volume_fraction": None,
    "range_duplicate_iou_threshold": None,
    "range_acceptance_max_attempts": None,
    "range_max_coverage_overshoot": None,
    "range_train_workload_replicates": 1,
    "workload_profile_id": None,
    "coverage_calibration_mode": None,
    "workload_stability_gate_mode": "final",
    "epochs": 6,
    "lr": 5e-4,
    "embed_dim": 64,
    "num_heads": 4,
    "num_layers": 3,
    "dropout": 0.1,
    "ranking_pairs_per_type": 96,
    "ranking_top_quantile": 0.80,
    "pointwise_loss_weight": 0.25,
    "loss_objective": "budget_topk",
    "budget_loss_ratios": None,
    "budget_loss_temperature": DEFAULT_BUDGET_LOSS_TEMPERATURE,
    "query_local_utility_aux_loss_weight": 0.50,
    "query_local_utility_segment_budget_head_weight": 0.10,
    "query_local_utility_segment_level_loss_weight": 0.25,
    "query_local_utility_behavior_rank_loss_weight": (
        DEFAULT_QUERY_LOCAL_UTILITY_BEHAVIOR_RANK_LOSS_WEIGHT
    ),
    "query_local_utility_sparse_head_rank_loss_weight": 0.0,
    "query_local_utility_sparse_head_bce_target_mode": "raw",
    "query_local_utility_train_marginal_diagnostics": False,
    "temporal_distribution_loss_weight": 0.0,
    "gradient_clip_norm": 1.0,
    "compression_ratio": 0.2,
    "csv_path": None,
    "train_csv_path": None,
    "validation_csv_path": None,
    "eval_csv_path": None,
    "cache_dir": None,
    "refresh_cache": False,
    "range_diagnostics_mode": "full",
    "validation_split_mode": "random",
    "model_type": "baseline",
    "historical_prior_k": 32,
    "historical_prior_clock_weight": 0.0,
    "historical_prior_mmsi_weight": 1.0,
    "historical_prior_density_weight": 1.0,
    "historical_prior_min_target": 0.0,
    "historical_prior_support_ratio": 1.0,
    "historical_prior_source_aggregation": "none",
    "workload": "range",
    "seed": 42,
    "early_stopping_patience": 0,
    "train_batch_size": 16,
    "inference_batch_size": 16,
    "query_chunk_size": 2048,
    "diagnostic_every": 1,
    "diagnostic_window_fraction": 0.2,
    "checkpoint_selection_metric": "score",
    "validation_score_every": None,
    "checkpoint_uniform_gap_weight": 0.5,
    "checkpoint_type_penalty_weight": 1.0,
    "checkpoint_smoothing_window": 1,
    "checkpoint_full_score_every": None,
    "checkpoint_candidate_pool_size": 1,
    "checkpoint_score_variant": None,
    "validation_global_sanity_penalty_enabled": True,
    "validation_global_sanity_penalty_weight": DEFAULT_VALIDATION_GLOBAL_SANITY_PENALTY_WEIGHT,
    "validation_sed_penalty_weight": DEFAULT_VALIDATION_SED_PENALTY_WEIGHT,
    "validation_endpoint_penalty_weight": DEFAULT_VALIDATION_ENDPOINT_PENALTY_WEIGHT,
    "validation_length_preservation_min": DEFAULT_VALIDATION_LENGTH_PRESERVATION_MIN,
    "mlqds_temporal_fraction": 0.0,
    "mlqds_diversity_bonus": 0.0,
    "mlqds_hybrid_mode": "fill",
    "selector_type": TEMPORAL_HYBRID_SELECTOR_TYPE,
    "learned_segment_geometry_gain_weight": DEFAULT_LEARNED_SEGMENT_GEOMETRY_GAIN_WEIGHT,
    "learned_segment_allocation_length_support_weight": (
        DEFAULT_LEARNED_SEGMENT_ALLOCATION_LENGTH_SUPPORT_WEIGHT
    ),
    "learned_segment_allocation_weight_floor": DEFAULT_LEARNED_SEGMENT_ALLOCATION_WEIGHT_FLOOR,
    "learned_segment_score_blend_weight": DEFAULT_LEARNED_SEGMENT_SCORE_BLEND_WEIGHT,
    "learned_segment_transfer_calibration_mode": DEFAULT_LEARNED_SEGMENT_TRANSFER_CALIBRATION_MODE,
    "learned_segment_fairness_preallocation": True,
    "learned_segment_length_repair_fraction": 0.0,
    "learned_segment_length_repair_score_protection_fraction": 0.0,
    "learned_segment_length_support_blend_weight": 0.0,
    "mlqds_stratified_center_weight": 0.0,
    "mlqds_min_learned_swaps": 0,
    "mlqds_score_mode": "rank",
    "mlqds_score_temperature": 1.0,
    "mlqds_rank_confidence_weight": 0.15,
    "mlqds_range_geometry_blend": 0.0,
    "temporal_residual_label_mode": None,
    "range_label_mode": "point_f1",
    "range_training_target_mode": "point_value",
    "range_target_balance_mode": "none",
    "range_replicate_target_aggregation": "label_mean",
    "range_temporal_target_blend": 0.0,
    "range_structural_target_blend": 0.25,
    "range_structural_target_source_mode": "blend",
    "range_target_budget_weight_power": 0.0,
    "range_marginal_target_radius_scale": 0.50,
    "range_query_spine_fraction": 0.10,
    "range_query_spine_mass_mode": "hit_group",
    "range_query_residual_multiplier": 1.0,
    "range_query_residual_mass_mode": "query",
    "range_set_utility_multiplier": 1.0,
    "range_set_utility_candidate_limit": 128,
    "range_set_utility_mass_mode": "gain",
    "range_boundary_prior_weight": 0.0,
    "range_teacher_distillation_mode": "none",
    "range_teacher_epochs": 4,
    "range_audit_compression_ratios": None,
    "query_prior_grid_bins": 64,
    "query_prior_smoothing_passes": 2,
    "final_metrics_mode": "diagnostic",
    "float32_matmul_precision": "highest",
    "allow_tf32": False,
    "amp_mode": "off",
}

RUN_CONFIG_OVERRIDE_NAMES = frozenset(RUN_CONFIG_DEFAULT_OVERRIDES)


def _dataclass_kwargs(
    config_type: type[Any],
    values: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build dataclass kwargs from matching flat config names plus explicit overrides."""
    explicit = overrides or {}
    kwargs: dict[str, Any] = {}
    for config_field in fields(config_type):
        if config_field.name in explicit:
            kwargs[config_field.name] = explicit[config_field.name]
        elif config_field.name in values:
            kwargs[config_field.name] = values[config_field.name]
    return kwargs


def _flat_run_config_values(overrides: RunConfigOverrides) -> dict[str, Any]:
    unknown = set(overrides) - RUN_CONFIG_OVERRIDE_NAMES
    if unknown:
        raise TypeError(f"Unknown run config overrides: {sorted(unknown)}")
    return {**RUN_CONFIG_DEFAULT_OVERRIDES, **overrides}


def _apply_workload_profile_defaults(values: dict[str, Any]) -> None:
    workload_profile_id = values["workload_profile_id"]
    if workload_profile_id:
        from workloads.generation.workload_profiles import (
            LEGACY_GENERATOR_PROFILE,
            range_workload_profile,
        )

        workload_profile = range_workload_profile(workload_profile_id)
        if workload_profile.profile_id != LEGACY_GENERATOR_PROFILE.profile_id:
            if values["query_coverage"] is None:
                values["query_coverage"] = workload_profile.target_coverage
            if values["range_max_coverage_overshoot"] is None:
                values["range_max_coverage_overshoot"] = (
                    workload_profile.max_coverage_overshoot
                )
            if values["coverage_calibration_mode"] is None:
                values["coverage_calibration_mode"] = workload_profile.coverage_calibration_mode


def build_run_config(**overrides: Unpack[RunConfigOverrides]) -> RunConfig:
    """Build structured run config from typed flat keyword overrides."""
    values = _flat_run_config_values(overrides)
    _apply_workload_profile_defaults(values)
    uses_csv = bool(
        values["csv_path"]
        or values["train_csv_path"]
        or values["validation_csv_path"]
        or values["eval_csv_path"]
    )
    return RunConfig(
        data=DataConfig(
            **_dataclass_kwargs(
                DataConfig,
                values,
                {
                    "n_ships": None if uses_csv else values["n_ships"],
                    "n_points_per_ship": None if uses_csv else values["n_points"],
                    "synthetic_route_families": 0
                    if uses_csv
                    else int(values["synthetic_route_families"]),
                },
            )
        ),
        query=QueryConfig(
            **_dataclass_kwargs(
                QueryConfig,
                values,
                {
                    "target_coverage": values["query_coverage"],
                    "range_train_anchor_modes": list(values["range_train_anchor_modes"] or []),
                    "range_train_footprints": list(values["range_train_footprints"] or []),
                },
            )
        ),
        model=ModelConfig(
            **_dataclass_kwargs(
                ModelConfig,
                values,
                {
                    "budget_loss_ratios": list(
                        values["budget_loss_ratios"] or DEFAULT_BUDGET_LOSS_RATIOS
                    ),
                    "validation_score_every": 0
                    if values["validation_score_every"] is None
                    else values["validation_score_every"],
                    "checkpoint_full_score_every": 1
                    if values["checkpoint_full_score_every"] is None
                    else values["checkpoint_full_score_every"],
                    "checkpoint_score_variant": values["checkpoint_score_variant"]
                    or "query_local_utility",
                    "temporal_residual_label_mode": values["temporal_residual_label_mode"]
                    or "none",
                    "range_audit_compression_ratios": list(
                        values["range_audit_compression_ratios"] or []
                    ),
                },
            )
        ),
        baselines=BaselineConfig(final_metrics_mode=values["final_metrics_mode"]),
    )


RUN_CONFIG_NAMESPACE_ALIASES = {
    "validation_global_sanity_penalty_enabled": "validation_global_sanity_penalty",
}

RUN_CONFIG_LOG_FIELD_ALIASES = {
    ("model", "model_type"): "model",
    ("query", "target_coverage"): "query_coverage",
    ("data", "n_points_per_ship"): "n_points",
    ("model", "checkpoint_uniform_gap_weight"): "uniform_gap_weight",
    ("model", "checkpoint_type_penalty_weight"): "type_penalty_weight",
    ("model", "checkpoint_smoothing_window"): "smoothing_window",
    ("model", "checkpoint_full_score_every"): "full_score_every",
    ("model", "checkpoint_candidate_pool_size"): "candidate_pool",
}


def build_run_config_from_namespace(args: Any) -> RunConfig:
    """Build run config from a parsed CLI namespace using the canonical builder contract."""
    kwargs: dict[str, Any] = {}
    for name in RUN_CONFIG_OVERRIDE_NAMES:
        source_name = RUN_CONFIG_NAMESPACE_ALIASES.get(name, name)
        if hasattr(args, source_name):
            kwargs[name] = getattr(args, source_name)
    return build_run_config(**kwargs)


def iter_run_config_log_items(
    config: RunConfig,
    runtime_settings: dict[str, Any] | None = None,
) -> list[tuple[str, Any]]:
    """Return flattened effective run config fields for human-readable logs."""
    runtime_settings = runtime_settings or {}
    items: list[tuple[str, Any]] = []
    for section_name, section in (
        ("model", config.model),
        ("query", config.query),
        ("data", config.data),
        ("baselines", config.baselines),
    ):
        for config_field in fields(section):
            value = getattr(section, config_field.name)
            if section_name == "model" and config_field.name == "float32_matmul_precision":
                value = runtime_settings.get("float32_matmul_precision", value)
            elif section_name == "model" and config_field.name == "allow_tf32":
                value = runtime_settings.get("tf32_matmul_allowed", value)
            log_name = RUN_CONFIG_LOG_FIELD_ALIASES.get(
                (section_name, config_field.name),
                config_field.name,
            )
            items.append((log_name, value))
    return items


def format_run_config_log_line(
    config: RunConfig,
    runtime_settings: dict[str, Any] | None = None,
) -> str:
    """Format the effective run config as the single-line CLI config log."""
    fields_text = "  ".join(
        f"{name}={value}" for name, value in iter_run_config_log_items(config, runtime_settings)
    )
    return f"[config] {fields_text}"


def derive_seed_bundle(master_seed: int) -> SeedBundle:
    """Derive deterministic sub-seeds from a master seed. See orchestration/README.md for details."""
    return SeedBundle(
        split_seed=(master_seed * LCG_MULTIPLIER + 1) & 0xFFFF_FFFF,
        train_query_seed=(master_seed * LCG_MULTIPLIER + 3) & 0xFFFF_FFFF,
        eval_query_seed=(master_seed * LCG_MULTIPLIER + 5) & 0xFFFF_FFFF,
        torch_seed=(master_seed * LCG_MULTIPLIER + 7) & 0xFFFF_FFFF,
    )
