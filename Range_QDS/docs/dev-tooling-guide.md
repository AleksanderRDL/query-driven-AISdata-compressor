# Range_QDS Developer Tooling

This is the active operational reference for local developer tooling. The
scientific protocol and acceptance gates remain in
[`query-driven-implementation-research-guide.md`](query-driven-implementation-research-guide.md).

## Tooling Principles

Use tools to enforce invariants, not to decorate the project. The highest-value
checks are workflow and protocol invariants:

- eval queries must not affect compression
- retained masks must freeze before eval scoring
- final workloads must be healthy and signature-stable
- prior fields must be train-derived only
- zero-prior ablations must preserve extent and metadata
- selector diagnostics must prove learned control is material
- benchmark summaries must expose all required gates
- run and benchmark commands must be reproducible

Use `uv` as the single Python execution layer. Project work must use the
repo-local `uv` virtual environment with CPython `3.14.5`, pinned by the root
`.python-version`. Do not mix active command styles between `uv`, bare Python,
pip, and hard-coded virtualenv paths.

Do not turn learned metrics into brittle tests. Training runs are noisy; use
regression testing for stable schema/report shape, not stochastic model scores.

Keep tooling out of hot model paths. Tool imports should not slow model forward
passes, target construction, selector execution, metric scoring, or benchmark
inner loops.

Prefer small, readable checks. A useful tool integration makes failure clearer.
Avoid giant snapshots, excessive Rich formatting, or broad tests that obscure
which invariant failed.

## Command Layer

Run project Python commands through `uv` from the repository root:

```bash
uv python install 3.14.5
uv sync --python 3.14.5 --group dev
uv lock --check
uv run --group dev -- pytest Range_QDS/tests
uv run --group dev -- pyright Range_QDS/benchmarking Range_QDS/config Range_QDS/data_preparation Range_QDS/scoring Range_QDS/models Range_QDS/orchestration Range_QDS/workloads Range_QDS/runtime Range_QDS/selection Range_QDS/learning Range_QDS/scripts Range_QDS/tests
uv run --group dev -- python -m orchestration.train_and_score --help
```

From `Range_QDS/`, prefer Make targets:

```bash
make sync
make lock-check
make check-env
make test
make typecheck
make lint
make lint-full
make lint-yaml
```

`make lint` is intentionally scoped to high-signal correctness checks
(`F401,F821,F822,F823`) so it is a reliable checkpoint save gate. `make
lint-full` runs the full Ruff rule set across active QDS packages.

## Artifact Inspection

Use jq for quick JSON inspection:

```bash
make inspect-run RUN=Range_QDS/artifacts/results/<run>/example_run.json
make inspect-gates RUN=Range_QDS/artifacts/results/<run>/example_run.json
make inspect-scores RUN=Range_QDS/artifacts/results/<run>/example_run.json
make inspect-causality RUN=Range_QDS/artifacts/results/<run>/example_run.json
make inspect-generator RUN=Range_QDS/artifacts/results/<run>/example_run.json
make inspect-predictability RUN=Range_QDS/artifacts/results/<run>/example_run.json
```

Reusable filters live in `scripts/jq/`:

- `run_summary.jq`
- `gates.jq`
- `scores.jq`
- `causality.jq`
- `generator_health.jq`
- `predictability.jq`

For a readable terminal summary:

```bash
uv run --group dev -- python Range_QDS/scripts/summarize_run.py Range_QDS/artifacts/results/<run>/example_run.json
```

Rich output is display-only. JSON artifacts remain the source of truth.

## Test Tooling

### Hypothesis

Use Hypothesis for small invariant checks around edge cases that fixed examples
miss.

Good targets:

- `workloads/generation/generator.py`
- `workloads/generation/anchors.py`
- `workloads/generation/coverage.py`
- `workloads/generation/profile_query_plan.py`
- `workloads/generation/workload_profiles.py`
- `learning/query_prior_fields.py`
- `learning/model_features.py`
- `selection/learned_segment_budget/core.py`
- `selection/learned_segment_budget/allocation.py`
- `selection/learned_segment_budget/length_repair.py`
- `selection/learned_segment_budget/diagnostics.py`
- `selection/learned_segment_budget/trace.py`
- `scoring/query_local_utility.py`
- `scoring/method_scoring.py`
- `orchestration/range_diagnostics.py`

Good properties:

- profile query plans sum exactly to requested count
- prefix-balanced plans do not create avoidable family drift
- coverage guards keep generated workloads near requested targets
- `range_query_mix` does not silently fall back to uncovered-anchor chasing
- zeroed prior fields preserve extent, bins, metadata, and tensor shape
- out-of-extent zero mode returns zero sampled prior features
- selector masks never exceed retained budget
- selector trace attribution remains internally consistent
- endpoint and global-sanity safeguards remain explicit where configured
- final-grid summaries block if any required child gate is false

Bad uses:

- full training loops
- GPU-heavy tests
- real AIS files
- stochastic convergence assertions
- arbitrary large floating tensor generation into model training
- high `max_examples` defaults that turn local checks into broad sweeps

Default settings should stay conservative:

```python
settings(max_examples=50, deadline=None)
```

Use `deadline=None` because PyTorch tensor construction has variable timing.
Increase example counts only for a targeted debugging run or a scheduled deeper
check.

Current command:

```bash
uv run --group dev -- pytest Range_QDS/tests/property -q
```

Current coverage:

- workload-profile query plans preserve requested counts and active families
- zero-prior ablations preserve extent, metadata, and tensor shapes
- learned-segment selector budget accounting remains consistent

### pytest-regressions

Use pytest-regressions to protect stable report and schema shape.

Good uses:

- small normalized dictionaries
- benchmark row field sets
- final-grid summary shape
- final-claim summary shape
- query-generation diagnostic schemas
- workload-signature gate summary shape
- predictability and causality selected-field summaries

Bad uses:

- full trained `example_run.json` files
- full stdout logs
- runtime seconds
- timestamps
- absolute paths
- GPU memory values
- raw query arrays
- random seeds unless the seed is the behavior under test
- stochastic learned metrics from learning runs

Snapshots should catch accidental field removal, renaming, and contract drift.
They should not be updated just to make a test pass. Update them only after
reviewing an intentional schema change.

Current command:

```bash
uv run --group dev -- pytest Range_QDS/tests/regression -q
```

Snapshots protect small stable schemas only:

- final-grid summary shape
- benchmark row field set
- gate-summary normalization

Do not snapshot full trained `example_run.json` files, runtime logs, timestamps,
absolute paths, or stochastic model metrics.

YAML lint:

```bash
uv run --group dev -- yamllint .
make lint-yaml
```

Generated pytest-regressions YAML snapshots are excluded from yamllint because
their formatting is owned by the snapshot tool.

## Risks

- jq becoming hidden acceptance logic: jq is for inspection; acceptance belongs
  in Python code and tests.
- flaky Hypothesis tests: avoid stochastic convergence, GPU, real AIS data, and
  large default example counts.
- noisy regression snapshots: snapshot normalized summaries only and strip
  runtime/path/timestamp/noisy metric fields.
- Rich output replacing artifacts: JSON remains authoritative; Rich is display
  only.
- tooling distraction: add tools only when they clarify gates, reports,
  reproducibility, or failure diagnosis.

## Checkpoint Verification

For tooling or documentation checkpoints, run:

```bash
uv sync --python 3.14.5 --group dev
uv lock --check
git diff --check
uv run --group dev -- yamllint .
uv run --group dev -- pytest Range_QDS/tests/property Range_QDS/tests/regression -q
uv run --group dev -- pytest Range_QDS/tests/unit/orchestration/test_query_driven_*.py Range_QDS/tests/unit/learning/test_query_local_utility_*.py Range_QDS/tests/unit/selection/test_query_driven_learned_segment_budget.py Range_QDS/tests/unit/workloads/test_query_driven_profiles.py Range_QDS/tests/unit/benchmarking/test_runner.py -q
```

For code checkpoints touching model, selector, query generation, metrics, or
benchmark reporting, also run the focused tests named in
`query-driven-implementation-research-guide.md` and the full `Range_QDS/tests`
suite when the blast radius justifies it.

## Policy

- Keep tooling out of hot model, selector, metric, and training loops.
- Keep acceptance logic in Python code/tests, not jq or shell wrappers.
- Update regression snapshots only after reviewing an intentional schema change.
- Keep one-off run or benchmark hacks out of production paths.
- Record extra discoveries in `query-driven-implementation-progress.md`.
