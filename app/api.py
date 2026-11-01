from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import Base, engine, get_db
from app.models import Feedback, Score
from app.batching import DynamicBatcher
from app.cache import ScoreCache
from app.registry import ModelRegistry
from app.rollout import RolloutManager
from app.scoring import ScoringService
from ml.schema import FEATURE_SCHEMA_VERSION
from ml.costs import load_config
from ml.features import build_feature_pipeline
import pandas as pd

app=FastAPI(title="Generation Quality Gate",version="1.0.0")
Base.metadata.create_all(bind=engine)

registry=ModelRegistry()
rollout=RolloutManager(registry)
cache=ScoreCache()
batchers={}


class ScoreRequest(BaseModel):
    request_id:str
    project_id:str
    features:dict
    model_version:str="v1"


class ScoreResponse(BaseModel):
    request_id:str
    project_id:str
    model_version:str
    identity_score:float
    defect_score:float
    adherence_score:float
    probability_bad:float
    decision:str


class FeedbackRequest(BaseModel):
    request_id:str
    project_id:str
    feedback:str
    comment:str|None=None


@app.get("/health")
def health():
    return {"status":"ok"}


def _process_batch(service,features_list):
    config=load_config()
    costs=config["decision"]["costs"]
    matrices, _ = build_feature_pipeline(pd.DataFrame(features_list))
    return [
        service.score(
            features={name: matrix[index:index + 1] for name, matrix in matrices.items()},
            costs=costs,
        )
        for index in range(len(features_list))
    ]


def _get_service(version):
    bundle=registry.get(version)

    if version not in batchers:
        service=ScoringService(
            models=bundle["models"],
            calibrators=bundle["calibrators"],
        )
        batcher=DynamicBatcher(
            process_batch=lambda features: _process_batch(service,features),
            max_batch_size=16,
            max_wait_ms=10,
        )
        batcher.start()
        batchers[version]=batcher

    return batchers[version]


@app.post("/v1/score",response_model=ScoreResponse)
def score(
    request:ScoreRequest,
    db:Session=Depends(get_db),
):
    try:
        if rollout.current_version is None:
            try:
                pinned=registry.get_pinned()
                version=pinned["model_version"]
                rollout.set_production(version)
            except (RuntimeError,FileNotFoundError):
                version=request.model_version
        else:
            version=(
                rollout.candidate_version
                if rollout.should_use_candidate(request.project_id)
                else rollout.current_version
            )

        registry.get(version)

    except (KeyError,RuntimeError,FileNotFoundError) as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    batcher=_get_service(version)

    cache_key = cache.make_key(
        model_version=version,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        features=request.features,
    )
    try:
        result=cache.get_or_compute(
            cache_key,
            lambda: batcher.submit(request.features).wait(timeout=30),
        )
    except TimeoutError as error:
        raise HTTPException(
            status_code=504,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    score_row=Score(
        request_id=request.request_id,
        project_id=request.project_id,
        model_version=version,
        identity_score=result["identity_score"],
        defect_score=result["defect_score"],
        adherence_score=result["adherence_score"],
        decision=result["decision"],
    )

    db.add(score_row)
    db.commit()

    return {
        "request_id":request.request_id,
        "project_id":request.project_id,
        "model_version":version,
        **result,
    }


@app.post("/v1/feedback")
def submit_feedback(
    request:FeedbackRequest,
    db:Session=Depends(get_db),
):
    valid_feedback={
        "accept",
        "retry",
        "review_good",
        "review_bad",
    }

    if request.feedback not in valid_feedback:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid feedback. Expected one of: {sorted(valid_feedback)}",
        )

    feedback=Feedback(
        request_id=request.request_id,
        project_id=request.project_id,
        feedback=request.feedback,
        comment=request.comment
    )

    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    return {
        "status":"accepted",
        "feedback_id":feedback.id,
    }
