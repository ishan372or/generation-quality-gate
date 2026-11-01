# Generation Quality Gate

Generation Quality Gate is the Cornflakes Media technical-assessment project: a model-serving service that scores generated shots and returns `KEEP`, `REGENERATE`, or `REVIEW`. It combines learned identity, defect, and adherence scorers with calibration, a cost-based policy, model versioning, batching, caching, rollout primitives, persistence, and a dashboard.

This README intentionally distinguishes the original assessment requirements from the current repository. The authoritative assessment brief is the attached Cornflakes Media technical project brief; this repository is not presented as complete where the implementation is partial.

## Current implementation at a glance

Implemented and exercised:

- Synthetic data generation with 64-dimensional frame/reference embeddings, 12 defect features, and 8 adherence axes.
- Grouped train/validation/test splitting by `project_id`, preventing project leakage.
- XGBoost identity, defect, and adherence scorers.
- Isotonic calibration and PR-AUC, recall-at-precision, ECE, bootstrap confidence intervals, and cost evaluation.
- Bundle creation under `artifacts/bundle` with per-file SHA256 validation.
- Filesystem model registry with PostgreSQL/SQLAlchemy metadata registration and idempotent `v1` pinning.
- FastAPI scoring and feedback routes, model-server inference, dynamic micro-batching, content-addressed caching, and single-flight computation.
- Sticky canary assignment, promotion, and rollback primitives.
- Streamlit dashboard backed by the configured database.
- Docker Compose topology for two API services, one model server, Redis, and dashboard. PostgreSQL is external through `DATABASE_URL`.

Partially implemented or not yet implemented are listed in [EVAL_RESULTS.md](EVAL_RESULTS.md), rather than being implied by the architecture diagram.

## Run locally

The ML pipeline is reproducible from the repository root:

```bash
python scripts/run_pipeline.py
```

It writes the dataset and grouped splits under `artifacts/data` and `artifacts/splits`, models and calibrators under `artifacts/models`, evaluation output under `artifacts/evaluation.txt`, and the validated bundle under `artifacts/bundle`.

For local API/model-server smoke tests, use a local SQLite fallback by leaving `DATABASE_URL` unset. The deployed configuration should use external PostgreSQL.

## Run with Docker Compose

```bash
cp .env.example .env
# Set DATABASE_URL in .env to the external PostgreSQL connection string.
docker compose up --build
```

Services and URLs:

- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- Model server: internal Compose service `model-server:8001`
- API health: `/healthz`; readiness: `/readyz`; model list: `/v1/models`; metrics: `/metrics`

The model server validates `artifacts/bundle`, registers or validates `v1`, checks the feature schema and SHA256 digests, then loads the three models and calibrators. API containers use `MODEL_SERVER_URL` to call it. The Compose file does not run PostgreSQL; supply an external `DATABASE_URL` as required by the brief.

## Environment

Copy `.env.example` and configure:

```env
DATABASE_URL=postgresql+psycopg://user:password@host:5432/database
REDIS_URL=redis://redis:6379/0
MODEL_SERVER_URL=http://model-server:8001
MODEL_VERSION=v1
FEATURE_SCHEMA_VERSION=fs1
API_PORT=8000
DASHBOARD_PORT=8501
```

No database credentials are hardcoded in the application configuration.

## Scoring flow

`POST /v1/score` selects the production or rollout-selected model version, constructs a cache key from model version, feature schema, input features, and parameters, then uses Redis when available or the in-process fallback. A miss is sent to the model server, which performs dynamic batching and invokes the three scorers and calibrators. The API applies the configured cost policy and persists the result.

The current API also exposes `POST /v1/score/batch`, but it is a synchronous convenience endpoint rather than the assessment's required `202 + poll + Idempotency-Key` asynchronous job contract. See the limitations matrix in [EVAL_RESULTS.md](EVAL_RESULTS.md).

## Tests and verification

```bash
python -m compileall -q app model_server ml dashboard config scripts
python -m pytest -q
docker compose config
```

The repository currently contains no committed pytest test files, so `python -m pytest -q` reports that no tests ran. The manual verification covered the ML pipeline, scoring, feedback persistence, model-server inference, cache single-flight, dynamic batching, rollout assignment, registry idempotency, and SHA mismatch rejection.

## Assessment deliverables

- [ARCHITECTURE.md](ARCHITECTURE.md) explains the implemented system and its boundaries.
- [EVAL_RESULTS.md](EVAL_RESULTS.md) records the observed evaluation numbers and the requirement-by-requirement status.
- `artifacts/evaluation.txt` is the generated evaluation report.

The original brief also asks for baseline comparisons, calibration plots, slice evaluation, chaos/load scripts, and automated tests for T1, T2, T4, T5, and rollout. Those are explicitly marked as missing or partial instead of being claimed here.
