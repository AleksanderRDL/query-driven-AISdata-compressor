# Models Module

Contains query-conditioned diagnostics, historical-prior diagnostics, and the
trainable workload-blind range scorers.

Important caveat: query-conditioned models are workload-aware because points
attend to query embeddings during prediction. They are useful diagnostics and
teacher candidates, but final range compression must use a query-free scoring
path such as `workload_blind_range_v2`.

## Files

| File | Purpose |
| --- | --- |
| `attention_utils.py` | Chunked point-to-query cross-attention. |
| `trajectory_qds_model.py` | Base transformer and normalization helper. |
| `workload_blind_qds_model.py` | Query-free range scorer used by workload-blind students. |
| `historical_prior_qds_model.py` | Query-free KNN historical-prior diagnostic model and prior-assisted student. |
| `workload_blind_range_v2.py` | Trainable factorized query-driven workload-blind model with prior-field and local/segment context encoders. |
| `../training/model_features.py` | Baseline, range-aware, and workload-blind point features. |

## Rules

- Query-conditioned models need `query_type_ids`; workload-blind models ignore
  query tensors and query type IDs.
- Query-conditioned attention is diagnostic only for final workload-blind
  claims.
- `workload_blind_range_v2` is the active trainable candidate, but acceptance
  still depends on the gates in `../docs/query-driven-rework-guide.md`.
- Historical-prior models are KNN diagnostics or prior-assisted students. They
  cannot claim learned value unless ablations beat or explain the standalone
  KNN prior.
- Current density/sparsity feature columns are current-split point-cloud
  context features. Do not call them train-derived query-prior fields.
- Feature dimensions are owned by `training/model_features.py`; update tests
  there when changing model input shape.

## Checkpoint Compatibility

`workload_blind_range_v2.calibration_head` is retained only as frozen
checkpoint-state compatibility. It is not part of final score composition, and
changing its weights must not affect factorized final scores. Removing it
requires an explicit checkpoint-loading migration or allowed-unexpected-key
policy for older saved states.
