import os
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.db import Base, engine
from app.registry import ModelRegistry
from app.scoring import ScoringService
from ml.costs import load_config
from ml.features import build_feature_pipeline
from model_server.batcher import ModelServerBatcher

app = FastAPI(title="Generation Quality Gate Model Server", version="1.0.0")
Base.metadata.create_all(bind=engine)
registry = ModelRegistry()
batchers = {}
loaded = {}
stats = {"requests": 0, "batches": 0, "last_batch_size": 0}


class PredictRequest(BaseModel):
    model_version: str
    features: dict


def _ensure_bundle():
    bundle = os.getenv("MODEL_BUNDLE_PATH", "artifacts/bundle")
    if os.path.exists(bundle):
        registry.register_or_validate("v1", bundle)


def _get_batcher(version):
    _ensure_bundle()
    if version not in batchers:
        bundle = registry.get(version)
        service = ScoringService(bundle["models"], bundle["calibrators"])
        costs = load_config()["decision"]["costs"]

        def process(rows):
            matrices, _ = build_feature_pipeline(pd.DataFrame(rows))
            stats["batches"] += 1
            stats["last_batch_size"] = len(rows)
            return [service.score({name: matrix[i:i + 1] for name, matrix in matrices.items()}, costs)
                    for i in range(len(rows))]

        batchers[version] = ModelServerBatcher(process_batch=process, max_batch_size=32, max_wait_ms=15)
        loaded[version] = bundle
    return batchers[version]


@app.on_event("startup")
def startup():
    _ensure_bundle()


@app.get("/health")
@app.get("/healthz")
def health():
    return {"status": "ok", "service": "model-server", "loaded_versions": list(loaded)}


@app.post("/infer")
@app.post("/v1/predict")
def infer(request: PredictRequest):
    try:
        result = _get_batcher(request.model_version).submit(request.features, deadline=time.monotonic() + 30)
        stats["requests"] += 1
        return {"model_version": request.model_version, **result}
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except TimeoutError as error:
        raise HTTPException(status_code=504, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/metrics")
def metrics():
    return stats


@app.get("/admin/stats")
def admin_stats():
    return {**stats, "loaded_versions": list(loaded)}


@app.post("/admin/load")
def admin_load(version: str = "v1"):
    _get_batcher(version)
    return {"status": "loaded", "model_version": version}
