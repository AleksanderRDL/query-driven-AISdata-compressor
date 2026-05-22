# Config Module

Shared configuration dataclasses used by orchestration, learning, checkpoints,
and benchmarking.

## Key Files

| File | Purpose |
| --- | --- |
| `run_config.py` | Data, query, model, baseline, and top-level run config dataclasses plus dataclass-field-driven flat config assembly and seed derivation. |

Keep CLI parsing in `orchestration/learning_scoring_cli.py`. Keep runtime mutation in
`runtime/`. Config objects should remain serializable and policy-light.
