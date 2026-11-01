# Generation Quality Gate Architecture

## Assessment target

The Cornflakes Media brief describes a quality gate called after every generated shot:

```text
worker -> score API -> cache/single-flight -> model-server micro-batcher
       -> identity | defect | adherence scorers
       -> calibration -> cost policy
       -> KEEP / REGENERATE / REVIEW
       -> feedback -> retraining -> shadow -> canary -> promote/rollback
```

The required deployment shape is at least two API replicas, one model-server container, PostgreSQL, Redis, and a small dashboard. PostgreSQL is external in this repository's Compose configuration and is supplied through `DATABASE_URL`.

## Implemented runtime

```text
api-1, api-2
    -> Redis cache when REDIS_URL is reachable; in-process fallback otherwise
    -> MODEL_SERVER_URL /infer
    -> model-server DynamicBatcher
    -> bundle loader and SHA256 validation
    -> XGBoost models + isotonic calibrators
    -> configured cost matrix
    -> SQLAlchemy scores/feedback/model metadata
    -> Streamlit dashboard
```

The API chooses a pinned production model or the existing sticky canary decision using `project_id`. The cache key contains model version, feature schema version, input content, and parameters. The model server converts the request feature payload through the shared `ml.features` code, batches rows, routes results back to callers, and returns calibrated scorer outputs plus the decision.

## ML design

The generator produces a 64-dimensional frame embedding and character-reference embedding, 12 defect features, and 8 adherence-axis features. Labels are generated alongside injected identity mismatch, artifacts, lighting shift, hard cases, lookalikes, and approximately 5% defect-label noise. The configured dataset currently uses a 40% defect rate, which is higher than the brief's example target of about 4%; this is an explicit implementation mismatch, not a hidden claim of rare-defect evaluation.

The split is grouped by `project_id`, so all characters and shots from a project stay in one split. Features are constructed by shared code for both training and serving. Identity uses cosine similarity plus absolute embedding difference and elementwise product; defect and adherence use their configured feature vectors. Models are XGBoost classifiers. Isotonic regression calibrators are fitted on validation outputs. The decision policy combines identity-good, defect-good, and adherence-good probabilities conservatively, then minimizes the configured expected cost of KEEP, REGENERATE, and REVIEW.

## Model lifecycle

`ml.bundle.create_bundle` writes the six model/calibrator files and `metadata.json` to `artifacts/bundle`. `validate_bundle` checks the feature schema and every listed file digest. `ModelRegistry.register_or_validate` copies or validates the bundle, stores metadata in the SQLAlchemy `models` table, and refuses a digest/schema mismatch. `v1` is pinned idempotently during startup. The current registry records filesystem bundle copies plus database metadata; the model files themselves remain local artifacts mounted into the image.

## Rollout state

The existing `RolloutManager` supports production, shadow, canary, promoted, and rollback states. Canary assignment is deterministic from the SHA256 hash of `project_id`, so a project remains sticky. Promotion and rollback are available without restarting the process.

The assessment additionally requires automatic guardrails based on disagreement, p95 latency, score-distribution shift, and safe automatic rollback. Those guardrails are not currently wired into an automated controller; drift and retraining modules exist, but the live API does not yet connect them to promotion decisions.

## Honest scope boundary

Implemented core behavior is documented above. The current repository does not yet claim the brief's asynchronous `202 + poll + Idempotency-Key` batch contract, bounded overload responses with `Retry-After`, full Postgres-backed Redis hot-cache persistence, shadow result logging, automatic canary guardrails, rolling PSI alerts, slice promotion gates, load/chaos scripts, baseline comparison, calibration plots, or the minimum committed automated tests. See [EVAL_RESULTS.md](EVAL_RESULTS.md) for the complete matrix.
