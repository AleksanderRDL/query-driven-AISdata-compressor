# Learning Module

Builds supervision, batches trajectory windows, trains learned scorers, selects
checkpoints, persists model artifacts, and exposes deterministic scorer
inference helpers. Retained-mask construction belongs in `../selection/`.

## Key Files

| File | Purpose |
| --- | --- |
| `targets/query_useful_v1.py` | Factorized QueryUsefulV1 label construction. |
| `targets/modes.py` | Public range target mode registries for CLI/config choices. |
| `targets/common.py` | Shared scalar target scaling, budgeting, balancing, and temporal-base helpers. |
| `targets/retained_frequency.py` | Retained-frequency, global-budget, and historical-prior scalar diagnostics. |
| `targets/structural.py` / `targets/marginal_coverage.py` | Query-free structural and neighborhood-marginal scalar diagnostics. |
| `targets/query_spine.py` / `targets/query_residual.py` | Train-query spine and residual-anchor scalar diagnostics. |
| `targets/set_utility.py` / `targets/local_swap.py` | Train-query set-utility and local-swap scalar diagnostics. |
| `targets/aggregation.py` | Multi-workload and component target aggregation. |
| `importance_labels.py` | Legacy typed F1/usefulness label construction for diagnostics. |
| `teacher_distillation.py` | Query-aware teacher-to-student label diagnostics. |
| `query_prior_fields.py` | Train-only query-prior field construction and sampling. |
| `model_features.py` | Query-free and query-conditioned point feature builders. |
| `scaler.py` | Persisted point/query feature scaling utilities. |
| `model_factory.py` | Shared model type validation and constructor mapping for training and checkpoint loading. |
| `losses.py` | Budget, ranking, pointwise, and auxiliary losses. |
| `checkpoint_validation.py` | Validation scoring and checkpoint selection support. |
| `checkpoint_selection.py` | Checkpoint candidate scoring and validation-stat bookkeeping. |
| `model_training.py` | Main fitting loop, validation cadence, and checkpoint-selection orchestration. |
| `factorized_target_diagnostics.py` | Factorized target support, entropy, top-k mass, and segment-position diagnostics. |
| `factorized_head_diagnostics.py` | Factorized QueryUsefulV1 head bias initialization and training-fit diagnostics. |
| `checkpoints.py` / `inference.py` | Save/load and deterministic prediction. |
| `trajectory_batching.py` / `supervised_windows.py` | Window construction without crossing trajectory boundaries. |
| `optimization_epoch.py` / `model_setup.py` | Epoch optimizer mechanics and model setup helpers. |
| `fit_diagnostics.py` / `outputs.py` | Scalar train-target fit diagnostics and learning result payloads. |

## Active Query-Driven Flow

The current candidate path is workload-blind at compression time:

1. Generate train-only `range_workload_v1` workloads.
2. Build factorized `QueryUsefulV1` labels.
3. Build train-derived query-prior fields.
4. Train `workload_blind_range_v2` from query-free point/context/prior features.
5. Select checkpoints without final eval-query scoring.
6. Freeze retained masks before held-out eval queries are scored.

The acceptance contract is in
[`../docs/query-driven-rework-guide.md`](../docs/query-driven-rework-guide.md).

## Final-Candidate Settings

| Setting | Active value |
| --- | --- |
| `workload_profile_id` | `range_workload_v1` |
| `model_type` | `workload_blind_range_v2` |
| `range_training_target_mode` | `query_useful_v1_factorized` |
| `selector_type` | `learned_segment_budget_v1` |
| `checkpoint_score_variant` | `query_useful_v1` |
| `checkpoint_selection_metric` | `uniform_gap` |

These settings are necessary but not sufficient. Final claims still require the
guide's single-cell and final-grid gates.

## Targets

`query_useful_v1_factorized` is the active target family. It keeps query-hit,
behavior, replacement, segment-budget, and prior-related signals separate
instead of collapsing them into one RangeUseful scalar.

Scalar `range_training_target_mode` values and teacher-distillation modes remain
for diagnostics and regression. They are registered in `targets/modes.py` or
`teacher_distillation.py` and are not final-success evidence. Use them only
when the checkpoint hypothesis explicitly needs that diagnostic path.

## Loss And Selection

- `loss_objective="budget_topk"` is the standard range loss.
- `ranking_bce` and `pointwise_bce` are diagnostics.
- `checkpoint_selection_metric="uniform_gap"` scores validation performance
  against fair uniform, with penalties for active-type deficits.
- `validation_length_preservation_min` controls checkpoint-selection sanity
  pressure; the strict global sanity gate is enforced later in run
  outputs.
- `training_fit_diagnostics` is train-data-only. It is useful for target-fit
  debugging, not final evidence.

## Model Inputs

`range_aware` and older query-conditioned models can see query features and are
diagnostic or teacher-only paths. Final blind candidates must score without
future eval query features.

`workload_blind_range_v2` is the active trainable QueryUsefulV1 candidate.
Other workload-blind and historical-prior feature families remain diagnostic or
compatibility paths; they must beat or explain their non-learned controls before
claiming learned value.

## Runtime And Persistence

Use `checkpoints.py` for save/load and `inference.py` for prediction.
Persisted checkpoints include model state, scaler stats, config, epochs
trained, workload type, and query-prior fields. `TrainingOutputs` carries
target diagnostics, epoch timing, and selected validation scores for run
artifacts. Runtime defaults belong in config and benchmark profiles, not this
README.
