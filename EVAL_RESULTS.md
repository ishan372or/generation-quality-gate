# Evaluation Results and Assessment Matrix

This report separates observed repository behavior from the requirements in the Cornflakes Media technical brief. The numbers below come from `python scripts/run_pipeline.py` on the generated test split.

## Reproduced ML evaluation

| Scorer | PR-AUC | PR-AUC 95% CI | Recall at 0.90 precision | Recall 95% CI | ECE |
|---|---:|---:|---:|---:|---:|
| Identity | 0.9973 | 0.9931 - 1.0000 | 1.0000 | 1.0000 - 1.0000 | 0.0159 |
| Defect | 0.9213 | 0.8862 - 0.9490 | 0.9161 | 0.8847 - 0.9435 | 0.0160 |
| Adherence | 0.9805 | 0.9725 - 0.9871 | 0.9352 | 0.8544 - 0.9714 | 0.0252 |

Decision policy result:

- Test samples: 750
- KEEP: 129
- REGENERATE: 529
- REVIEW: 92
- Expected cost: 281.33 per 1,000 shots
- Total test cost: 211.00

The report includes bootstrap confidence intervals and ECE. A baseline comparison, reliability plots, and slice table are not currently generated, so no baseline advantage is claimed.

## Requirement status

| Brief requirement | Status | Evidence or limitation |
|---|---|---|
| Reproducible generated data with config/seed | Implemented | `ml/datagen.py`, `config/data.yaml`, `scripts/run_pipeline.py`; Parquet output and split CSVs are generated. |
| Honest grouped split without leakage | Implemented | `GroupShuffleSplit` grouped by `project_id`; no committed automated T1 test. |
| Three learned scorers | Implemented | XGBoost identity, defect, and adherence models. |
| Baselines on same test split | Missing | No baseline scorer or baseline comparison report. |
| PR-AUC, recall at fixed precision, confidence intervals | Implemented | `ml/metrics.py` and `artifacts/evaluation.txt`. |
| Calibration and ECE | Partial | Isotonic calibration and ECE exist; reliability curves are not produced. |
| Cost-based KEEP/REGENERATE/REVIEW | Implemented | `ml/costs.py` and `config/train.yaml`. |
| Sync `/v1/score` | Implemented | API route returns model version, three scores, probability, and decision. |
| Async batch `202 + poll + Idempotency-Key` | Partial | `/v1/score/batch` exists but is synchronous and has no idempotency-key workflow. |
| Deadlines and expired queued work | Partial | Dynamic batcher supports deadlines; public score request does not expose/propagate a caller deadline. |
| Bounded overload with 429/503 and Retry-After | Missing | No bounded queue admission or Retry-After response. |
| Dynamic micro-batching | Implemented | Model-server batcher has configurable batch size/wait and routes results. No 200 req/s benchmark is committed. |
| Content-addressed cache and single-flight | Partial | Redis is used when reachable, with in-process fallback and deterministic keys; no Postgres cache tier or committed 100-request test. |
| Model registry with pins and digest checks | Partial | Bundle SHA/schema checks and SQLAlchemy metadata registration exist; registry also maintains filesystem copies. |
| Feature-schema mismatch refusal | Partial | Bundle schema is checked; caller-provided schema version is not yet part of the request contract. |
| Shadow/canary/rollback | Partial | Sticky canary, promote, rollback, and shadow state primitives exist. Shadow logging and automatic guardrails are not connected. |
| Drift monitoring and alerts | Partial | PSI utilities exist in `ml/drift.py`; live rolling-window monitoring and alerts are not wired. |
| Feedback strengths and late labels | Partial | Feedback is persisted with label/strength fields; feedback is not yet joined into an operational retraining queue. |
| Retraining and promotion gate | Partial | Retraining/gate modules exist; frozen golden/slice evaluation and automatic registration are not integrated end-to-end. |
| JSON logs and full metrics | Partial | Health/metrics endpoints exist; structured request/batch/queue/model histograms are not complete. |
| Dashboard | Partial | Dashboard reads database scores/feedback/rollout and can submit a score; drift, slice, canary controls, and model views are limited. |
| Docker Compose deployment | Partial | Compose defines two APIs, model server, Redis, and dashboard and validates with `docker compose config`; actual Docker startup was blocked because Docker Desktop's engine was unavailable. |
| External PostgreSQL | Implemented in configuration | `DATABASE_URL` is environment-driven and Compose does not create a local Postgres container; no live external database was available for verification. |
| Required automated tests T1/T2/T4/T5/rollout | Missing | No committed pytest tests are present; manual checks were run for split, single-flight, deadlines in the batcher, schema/bundle validation, and rollout. |
| Load/chaos tools and scenarios T2/T3/T8/T11 | Missing | No committed burst, kill, corrupt-model, or bad-canary scripts. |

## Commands run

```bash
python scripts/run_pipeline.py
python -m compileall -q app model_server ml dashboard config scripts
python -m pytest -q
docker compose config
```

Observed results:

- The ML pipeline completed successfully and reproduced the metrics above.
- Compilation passed.
- `python -m pytest -q` reported no tests ran because no test files are committed.
- Compose syntax/configuration validation passed.
- `docker compose build` could not run to completion because Docker Desktop's Linux engine was unavailable in the environment.
