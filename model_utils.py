import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


API_KEY_ENV_NAME = "API_KEY"
DEFAULT_API_KEY = "demo-secret-key"
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
CLASS_NAMES = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}


class PredictionError(Exception):
    status_code = 500
    error = "prediction_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def to_dict(self) -> Dict[str, str]:
        return {
            "error": self.error,
            "message": self.message,
        }


class InvalidAPIKeyError(PredictionError):
    status_code = 401
    error = "invalid_api_key"


class InvalidPayloadError(PredictionError):
    status_code = 400
    error = "invalid_payload"


class InvalidFeaturesError(PredictionError):
    status_code = 400
    error = "invalid_features"


class ModelFileMissingError(PredictionError):
    status_code = 500
    error = "missing_model_file"


def get_expected_api_key() -> str:
    return os.getenv(API_KEY_ENV_NAME, DEFAULT_API_KEY)


def verify_api_key(api_key: Optional[str]) -> None:
    if api_key != get_expected_api_key():
        raise InvalidAPIKeyError("Invalid or missing API key")


@lru_cache(maxsize=1)
def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise ModelFileMissingError(f"Model file not found at {MODEL_PATH}")

    try:
        with MODEL_PATH.open("rb") as file:
            artifact = pickle.load(file)
    except ModelFileMissingError:
        raise
    except Exception as exc:
        raise PredictionError(f"Could not load model file: {exc}") from exc

    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"]

    return artifact


def validate_features(features: Any) -> List[float]:
    if not isinstance(features, list):
        raise InvalidFeaturesError("Features must be provided as a list")

    if len(features) != 4:
        raise InvalidFeaturesError("Features must contain exactly 4 numeric values")

    validated = []
    for value in features:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidFeaturesError("Each feature must be numeric")
        validated.append(float(value))

    return validated


def predict_one(features: Any) -> Dict[str, Any]:
    validated_features = validate_features(features)
    model = load_model()
    input_array = np.array([validated_features], dtype=float)

    try:
        prediction = int(model.predict(input_array)[0])
    except Exception as exc:
        raise PredictionError(f"Model prediction failed: {exc}") from exc

    result: Dict[str, Any] = {
        "prediction": prediction,
        "class_name": CLASS_NAMES.get(prediction, str(prediction)),
        "features": validated_features,
    }

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(input_array)[0]
            result["probabilities"] = {
                CLASS_NAMES.get(index, str(index)): round(float(probability), 6)
                for index, probability in enumerate(probabilities)
            }
        except Exception as exc:
            raise PredictionError(f"Model probability prediction failed: {exc}") from exc

    return result


def handle_predict_payload(payload: Any, api_key: Optional[str] = None) -> Dict[str, Any]:
    verify_api_key(api_key)

    if not isinstance(payload, dict):
        raise InvalidPayloadError("Payload must be a JSON object")

    has_single = "features" in payload
    has_batch = "items" in payload

    if has_single == has_batch:
        raise InvalidPayloadError("Payload must contain either 'features' or 'items'")

    if has_single:
        return predict_one(payload["features"])

    items = payload["items"]
    if not isinstance(items, list) or not items:
        raise InvalidPayloadError("Items must be a non-empty list")

    results = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise InvalidPayloadError("Each batch item must be a JSON object")
        if "features" not in item:
            raise InvalidPayloadError("Each batch item must contain features")

        item_id = item.get("id", str(index))
        results.append({
            "id": item_id,
            **predict_one(item["features"]),
        })

    return {"results": results}
