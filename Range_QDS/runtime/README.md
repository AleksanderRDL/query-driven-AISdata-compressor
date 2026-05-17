# Runtime Module

Shared process-local runtime controls used by experiment entrypoints and
training loops.

## Key Files

| File | Purpose |
| --- | --- |
| `torch_runtime.py` | Torch precision, TF32, CUDA memory, and AMP helpers. |

Runtime helpers must stay policy-light. Benchmark profiles and final-claim
gates belong in `experiments/`; training behavior belongs in `training/`.
