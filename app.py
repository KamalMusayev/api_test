import time
from collections import defaultdict, deque
from typing import Any, Optional

from fastapi import Body, FastAPI, Header, HTTPException, Request

from model_utils import (
    InvalidAPIKeyError,
    InvalidFeaturesError,
    InvalidPayloadError,
    ModelFileMissingError,
    PredictionError,
    handle_predict_payload,
)


RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60

request_history = defaultdict(deque)

app = FastAPI(
    title="Iris ML Prediction API",
    description="A local FastAPI API for serving an Iris classification model.",
    version="1.0.0",
)


def get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    if request.client:
        return request.client.host

    return "unknown"


def check_rate_limit(ip: str) -> None:
    now = time.time()
    timestamps = request_history[ip]

    while timestamps and now - timestamps[0] >= RATE_LIMIT_WINDOW_SECONDS:
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        retry_after = int(RATE_LIMIT_WINDOW_SECONDS - (now - timestamps[0])) + 1
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Maximum 10 requests per minute per IP.",
            headers={"Retry-After": str(retry_after)},
        )

    timestamps.append(now)


@app.get("/")
def root():
    return {
        "message": "Iris ML Prediction API is running",
        "docs": "/docs",
    }


@app.post("/predict")
def predict(
    request: Request,
    payload: Any = Body(...),
    x_api_key: Optional[str] = Header(default=None),
):
    client_ip = get_client_ip(request)
    check_rate_limit(client_ip)

    try:
        return handle_predict_payload(payload, api_key=x_api_key)
    except InvalidAPIKeyError as exc:
        raise HTTPException(status_code=401, detail=exc.to_dict()) from exc
    except (InvalidPayloadError, InvalidFeaturesError) as exc:
        raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
    except (ModelFileMissingError, PredictionError) as exc:
        raise HTTPException(status_code=500, detail=exc.to_dict()) from exc
