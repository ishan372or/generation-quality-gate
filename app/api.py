import os
import time
from collections import Counter

import httpx
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.batching import DynamicBatcher
from app.cache import ScoreCache
from app.db import Base, engine, get_db
from app.models import Feedback, ModelVersion, Score
from app.registry import ModelRegistry
from app.rollout import RolloutManager
from app.scoring import ScoringService
from ml.costs import load_config
from ml.features import build_feature_pipeline
from ml.schema import FEATURE_SCHEMA_VERSION

app = FastAPI(title="Generation Quality Gate", version="1.0.0")
Base.metadata.create_all(bind=engine)
registry = ModelRegistry()
rollout = RolloutManager(registry)
cache = ScoreCache()
batchers = {}
metrics = Counter()


class ScoreRequest(BaseModel):
    request_id: str
    project_id: str
    features: dict
    model_version: str = "v1"


class ScoreResponse(BaseModel):
    request_id: str
    project_id: str
    model_version: str
    identity_score: float
    defect_score: float
    adherence_score: float
    probability_bad: float
    decision: str


class BatchScoreRequest(BaseModel):
    requests: list[ScoreRequest]


class FeedbackRequest(BaseModel):
    request_id: str
    project_id: str
    feedback: str
    comment: str | None = None
    model_version: str = "v1"
    strength: str | None = None


def _ensure_model():
    bundle_path = os.getenv("MODEL_BUNDLE_PATH", "artifacts/bundle")
    if os.path.exists(bundle_path):
        registry.register_or_validate("v1", bundle_path)
    if rollout.current_version is None:
        if registry.pin_file.exists():
            rollout.set_production(registry.get_pinned()["model_version"])
        elif (registry.registry_dir / "v1").exists():
            registry.pin("v1")
            rollout.set_production("v1")


@app.on_event("startup")
def startup():
    _ensure_model()


@app.get("/health")
@app.get("/healthz")
def health():
    return {"status": "ok", "service": "api"}


@app.get("/readyz")
def ready(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        _ensure_model()
        if rollout.current_version is None:
            raise RuntimeError("No model version is loaded")
        registry.get(rollout.current_version)
        return {"status": "ready", "model_version": rollout.current_version}
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def _selected_version(project_id, requested):
    _ensure_model()
    if rollout.current_version is None:
        return requested
    return rollout.candidate_version if rollout.should_use_candidate(project_id) else rollout.current_version


def _local_batch(service, rows):
    costs = load_config()["decision"]["costs"]
    matrices, _ = build_feature_pipeline(pd.DataFrame(rows))
    return [service.score({name: matrix[i:i + 1] for name, matrix in matrices.items()}, costs)
            for i in range(len(rows))]


def _local_batcher(version):
    if version not in batchers:
        bundle = registry.get(version)
        service = ScoringService(bundle["models"], bundle["calibrators"])
        batchers[version] = DynamicBatcher(lambda rows: _local_batch(service, rows), max_batch_size=32, max_wait_ms=15)
        batchers[version].start()
    return batchers[version]


def _infer(version, features):
    model_server = os.getenv("MODEL_SERVER_URL", "").strip()
    if model_server:
        response = httpx.post(f"{model_server.rstrip('/')}/infer", json={"model_version": version, "features": features}, timeout=30)
        response.raise_for_status()
        return response.json()
    return _local_batcher(version).submit(features).wait(timeout=30)


@app.post("/v1/score", response_model=ScoreResponse)
def score(request: ScoreRequest, db: Session = Depends(get_db)):
    started = time.perf_counter()
    version = _selected_version(request.project_id, request.model_version)
    try:
        registry.get(version)
    except (KeyError, FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    key = cache.make_key(version, FEATURE_SCHEMA_VERSION, request.features)
    cached = cache.get(key)
    try:
        result = cached if cached is not None else cache.get_or_compute(key, lambda: _infer(version, request.features))
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail=f"Model server unavailable: {error}") from error
    except TimeoutError as error:
        raise HTTPException(status_code=504, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    cache_hit = cached is not None
    latency_ms = (time.perf_counter() - started) * 1000
    db.add(Score(request_id=request.request_id, project_id=request.project_id, model_version=version,
                 identity_score=result["identity_score"], defect_score=result["defect_score"],
                 adherence_score=result["adherence_score"], probability_bad=result["probability_bad"],
                 decision=result["decision"], cache_hit=cache_hit, latency_ms=latency_ms))
    db.commit()
    metrics["requests_total"] += 1
    metrics["cache_hits_total"] += int(cache_hit)
    metrics[f"decision_{result['decision']}"] += 1
    return {"request_id": request.request_id, "project_id": request.project_id, "model_version": version, **result}


@app.post("/v1/score/batch")
def score_batch(request: BatchScoreRequest, db: Session = Depends(get_db)):
    return [score(item, db) for item in request.requests]


@app.post("/v1/feedback")
def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    valid = {"accept", "retry", "review_good", "review_bad"}
    if request.feedback not in valid:
        raise HTTPException(status_code=400, detail=f"Invalid feedback. Expected one of: {sorted(valid)}")
    feedback = Feedback(request_id=request.request_id, project_id=request.project_id, feedback=request.feedback,
                        label=request.feedback, strength=request.strength, comment=request.comment)
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return {"status": "accepted", "feedback_id": feedback.id}


@app.get("/v1/models")
def list_models(db: Session = Depends(get_db)):
    return {"models": [{"name": item.name, "version": item.version, "status": item.status,
                         "feature_schema": item.feature_schema_version, "sha256": item.sha256,
                         "created_at": item.created_at.isoformat()} for item in db.query(ModelVersion).all()]}


@app.get("/metrics")
def metrics_endpoint():
    return dict(metrics)
