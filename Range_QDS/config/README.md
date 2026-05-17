# Config Module

Shared configuration dataclasses used by experiments, training, checkpoints,
and benchmark tooling.

## Key Files

| File | Purpose |
| --- | --- |
| `experiment_config.py` | Data, query, model, baseline, and top-level experiment config dataclasses plus the flat config builder and seed derivation. |

Keep CLI parsing in `experiments/experiment_cli.py`. Keep runtime mutation in
`runtime/`. Config objects should remain serializable and policy-light.
