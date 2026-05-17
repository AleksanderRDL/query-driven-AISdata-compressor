# Runtime Module

Shared process-local runtime controls used by orchestration entrypoints and
training loops.

## Key Files

| File | Purpose |
| --- | --- |
| `torch_runtime.py` | Torch precision, TF32, CUDA memory, and AMP helpers. |

Runtime helpers must stay policy-light. Benchmark policy belongs in
`benchmarking/`; orchestration behavior belongs in `orchestration/`; training
behavior belongs in `training/`.
