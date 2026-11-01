# Generation Quality Gate

This repository serves generated-shot quality scores as `KEEP`, `REGENERATE`, or `REVIEW` using the existing XGBoost scorers, isotonic calibrators, cost policy, model registry, cache, and dynamic batching.

## Run with Docker

```bash
cp .env.example .env
docker compose up --build
```

Available URLs:

- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

The compose stack starts Redis, two API containers, the model server, and Streamlit. PostgreSQL is external and is read from `DATABASE_URL`; no database credentials are hardcoded.

On startup, the model bundle at `artifacts/bundle` is validated, its SHA256 values are checked, and version `v1` is registered and pinned idempotently.

## ML pipeline

```bash
python scripts/run_pipeline.py
```

This generates the dataset, grouped splits, three scorers, calibration artifacts, evaluation output, and `artifacts/bundle`.

## Tests and checks

```bash
python -m compileall -q app model_server ml dashboard config scripts
python -m pytest -q
docker compose config
```

The API computes a content-addressed cache key from model version, feature schema, features, and parameters. Cache misses are sent to the model server, which batches requests before model inference. Scores and feedback are persisted through SQLAlchemy.
