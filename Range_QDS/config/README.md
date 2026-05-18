# Config Module

Shared configuration dataclasses used by orchestration, training, checkpoints,
and benchmarking.

## Key Files

| File | Purpose |
| --- | --- |
| `experiment_config.py` | Data, query, model, baseline, and top-level run config dataclasses plus the flat config builder and seed derivation. |

Keep CLI parsing in `orchestration/training_scoring_cli.py`. Keep runtime mutation in
`runtime/`. Config objects should remain serializable and policy-light.
