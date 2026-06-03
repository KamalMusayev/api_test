from typing import Dict, List, Union

import joblib
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse


API_KEY = "demo-secret-key"

artifact = joblib.load("model.pkl")
model = artifact["model"]
class_names = artifact["class_names"]
feature_names = artifact["feature_names"]
accuracy = artifact["accuracy"]

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Iris ML Prediction API",
    description="A local FastAPI API for serving an Iris classification model.",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": "Too many requests. Please try again later."
        }
    )


class SinglePredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description=(
            "Exactly four numeric Iris features: "
            "[sepal_length, sepal_width, petal_length, petal_width]"
        )
    )


class BatchItem(BaseModel):
    id: str = Field(..., min_length=1, max_length=64)
    features: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description=(
            "Exactly four numeric Iris features: "
            "[sepal_length, sepal_width, petal_length, petal_width]"
        )
    )


class BatchPredictRequest(BaseModel):
    items: List[BatchItem] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="Batch of 1 to 32 prediction items"
    )


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )


def predict_one(features: List[float]) -> Dict[str, Union[int, str, float]]:
    X = np.array([features])
    class_id = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    probability = float(probabilities[class_id])

    return {
        "class_id": class_id,
        "prediction": class_names[class_id],
        "probability": round(probability, 4)
    }


@app.get("/")
def root():
    return {
        "message": "Iris ML Prediction API is running",
        "model": "Logistic Regression",
        "accuracy": round(accuracy, 4),
        "features": feature_names,
        "docs": "/docs"
    }


@app.post("/predict")
@limiter.limit("10/minute")
def predict(
    request: Request,
    payload: Union[SinglePredictRequest, BatchPredictRequest],
    _: None = Depends(verify_api_key)
):
    if isinstance(payload, SinglePredictRequest):
        return predict_one(payload.features)

    results = {}

    for item in payload.items:
        results[item.id] = predict_one(item.features)

    return {
        "results": results
    }