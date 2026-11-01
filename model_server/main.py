import time

from fastapi import FastAPI,HTTPException
from pydantic import BaseModel

from app.scoring import ScoringService
from ml.costs import load_config

from model_server.batcher import ModelServerBatcher
from model_server.loader import ModelLoader


app=FastAPI(
    title="Generation Quality Gate Model Server",
    version="1.0.0",
)

loader=ModelLoader()
batchers={}

class PredictRequest(BaseModel):
    model_version:str
    features:dict

class PredictResponse(BaseModel):
    model_version:str
    identity_score:float
    defect_score:float
    adherence_score:float
    probability_bad:float
    decision:str

def _get_batcher(model_version):
    if model_version not in batchers:
        bundle=loader.load(model_version)
        service=ScoringService(
            models=bundle["models"],
            calibrators=bundle["calibrators"],
        )
        config=load_config()
        costs=config["decision"]["costs"]
        def process_batch(features_list):
            return [
                service.score(
                    features=features,
                    costs=costs,
                )
                for features in features_list
            ]
        batchers[model_version]=ModelServerBatcher(
            process_batch=process_batch,
            max_batch_size=16,
            max_wait_ms=10,
        )

    return batchers[model_version]


@app.get("/health")
def health():
    return {
        "status":"ok",
        "service":"model-server",
    }


@app.post("/v1/predict",response_model=PredictResponse)
def predict(request:PredictRequest):
    try:
        batcher=_get_batcher(request.model_version)

        result=batcher.submit(
            features=request.features,
            deadline=time.monotonic()+30,
        )

        return {
            "model_version":request.model_version,
            **result,
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

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