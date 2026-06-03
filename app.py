from typing import Any, Optional

from fastapi import Body, FastAPI, Header, HTTPException

from model_utils import (
    InvalidAPIKeyError,
    InvalidFeaturesError,
    InvalidPayloadError,
    ModelFileMissingError,
    PredictionError,
    handle_predict_payload,
)


app = FastAPI(
    title="Iris ML Prediction API",
    description="A local FastAPI API for serving an Iris classification model.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Iris ML Prediction API is running",
        "docs": "/docs",
    }


@app.post("/predict")
def predict(
    payload: Any = Body(...),
    x_api_key: Optional[str] = Header(default=None),
):
    try:
        return handle_predict_payload(payload, api_key=x_api_key)
    except InvalidAPIKeyError as exc:
        raise HTTPException(status_code=401, detail=exc.to_dict()) from exc
    except (InvalidPayloadError, InvalidFeaturesError) as exc:
        raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
    except (ModelFileMissingError, PredictionError) as exc:
        raise HTTPException(status_code=500, detail=exc.to_dict()) from exc
