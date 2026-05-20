# P8

AIS data tooling and query-driven trajectory simplification research.

## Workstreams

| Area | Path | Purpose |
| --- | --- | --- |
| Cleaning pipeline | [`ais_pipeline/`](ais_pipeline/) | Spark-based AIS CSV cleaning. |
| Database tools | [`db/`](db/) | Local PostGIS setup, CSV import, and range-query checks. |
| QDS research | [`Range_QDS/`](Range_QDS/) | ML trajectory simplification, benchmarks, and redesign work. |
| Data folders | [`AISDATA/`](AISDATA/) | Raw and cleaned AIS source data. |

## Python Environment

Use the repo-local virtual environment managed by `uv`. The project interpreter
is CPython `3.14.5`, pinned by [`.python-version`](.python-version). Do not use
system Python, bare `pip`, or manually managed virtualenvs for project work.

## Quick Start

Root cleaning pipeline:

```bash
uv python install 3.14.5
uv sync --python 3.14.5 --group dev
uv run --group dev -- python main.py
```

QDS work:

```bash
uv sync --python 3.14.5 --group dev
make check-env
make test
```

Database helpers:

```bash
make db-up
make db-smoke
make db-import CSV=AISDATA/cleaned/<file-or-directory>
make db-query QUERY_ARGS="--help"
```

## Documentation

- [`Range_QDS/README.md`](Range_QDS/README.md): QDS usage and where to look next.
- [`Range_QDS/docs/query-driven-rework-guide.md`](Range_QDS/docs/query-driven-rework-guide.md): canonical query-driven redesign source of truth.
- [`Range_QDS/docs/query-driven-rework-progress.md`](Range_QDS/docs/query-driven-rework-progress.md): checkpoint progress log.
- [`AISDATA/README.md`](AISDATA/README.md): data folder conventions.
- [`ais_pipeline/README.md`](ais_pipeline/README.md): cleaning pipeline layout.
- [`db/README.md`](db/README.md): database lifecycle and scripts.
