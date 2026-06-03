import os
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st

from model_utils import (
    DEFAULT_API_KEY,
    InvalidAPIKeyError,
    InvalidFeaturesError,
    InvalidPayloadError,
    ModelFileMissingError,
    PredictionError,
    handle_predict_payload,
)


def get_secret(name: str) -> Optional[str]:
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def get_config_value(name: str) -> Optional[str]:
    return get_secret(name) or os.getenv(name)


def get_api_key() -> str:
    return get_config_value("API_KEY") or DEFAULT_API_KEY


def get_api_url() -> Optional[str]:
    api_url = get_config_value("API_URL")
    return api_url.rstrip("/") if api_url else None


def call_predict(payload: Dict[str, Any], api_key: str) -> Tuple[int, Dict[str, Any]]:
    api_url = get_api_url()

    if api_url:
        url = api_url if api_url.endswith("/predict") else f"{api_url}/predict"
        response = requests.post(
            url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
            },
            timeout=10,
        )

        try:
            body = response.json()
        except ValueError:
            body = {"error": "invalid_response", "message": response.text}

        return response.status_code, body

    try:
        return 200, handle_predict_payload(payload, api_key=api_key)
    except InvalidAPIKeyError as exc:
        return 401, exc.to_dict()
    except (InvalidPayloadError, InvalidFeaturesError) as exc:
        return 400, exc.to_dict()
    except (ModelFileMissingError, PredictionError) as exc:
        return 500, exc.to_dict()


def show_response(status_code: int, body: Dict[str, Any], success_message: str) -> None:
    st.write("Status code:", status_code)

    if status_code == 200:
        st.success(success_message)
    else:
        st.error("Request failed")

    st.json(body)


st.set_page_config(
    page_title="Iris ML Prediction UI",
    page_icon=":material/local_florist:",
    layout="centered",
)

st.title("Iris ML Prediction")

api_url = get_api_url()
api_key = get_api_key()

if api_url:
    st.info(f"External API mode is active: {api_url}")
else:
    st.info("Direct model prediction mode is active.")

st.divider()

mode = st.radio(
    "Choose prediction mode:",
    ["Single prediction", "Batch prediction"],
    horizontal=True,
)

if mode == "Single prediction":
    st.subheader("Single Prediction")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Sepal length", value=5.1)
        petal_length = st.number_input("Petal length", value=1.4)
    with col2:
        sepal_width = st.number_input("Sepal width", value=3.5)
        petal_width = st.number_input("Petal width", value=0.2)

    payload = {
        "features": [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
        ]
    }

    st.write("Request payload:")
    st.json(payload)

    if st.button("Predict", type="primary"):
        try:
            status_code, body = call_predict(payload, api_key)
            show_response(status_code, body, "Prediction successful")
        except requests.exceptions.ConnectionError:
            st.error("External API is not reachable.")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

else:
    st.subheader("Batch Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Flower 1**")
        f1_sepal_length = st.number_input("Flower 1 sepal length", value=5.1)
        f1_sepal_width = st.number_input("Flower 1 sepal width", value=3.5)
        f1_petal_length = st.number_input("Flower 1 petal length", value=1.4)
        f1_petal_width = st.number_input("Flower 1 petal width", value=0.2)

    with col2:
        st.markdown("**Flower 2**")
        f2_sepal_length = st.number_input("Flower 2 sepal length", value=6.7)
        f2_sepal_width = st.number_input("Flower 2 sepal width", value=3.0)
        f2_petal_length = st.number_input("Flower 2 petal length", value=5.2)
        f2_petal_width = st.number_input("Flower 2 petal width", value=2.3)

    payload = {
        "items": [
            {
                "id": "flower_1",
                "features": [
                    f1_sepal_length,
                    f1_sepal_width,
                    f1_petal_length,
                    f1_petal_width,
                ],
            },
            {
                "id": "flower_2",
                "features": [
                    f2_sepal_length,
                    f2_sepal_width,
                    f2_petal_length,
                    f2_petal_width,
                ],
            },
        ]
    }

    st.write("Request payload:")
    st.json(payload)

    if st.button("Predict batch", type="primary"):
        try:
            status_code, body = call_predict(payload, api_key)
            show_response(status_code, body, "Batch prediction successful")
        except requests.exceptions.ConnectionError:
            st.error("External API is not reachable.")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

st.divider()

st.subheader("API Key Test")

wrong_key_payload = {
    "features": [5.1, 3.5, 1.4, 0.2],
}

if st.button("Test wrong API key"):
    try:
        status_code, body = call_predict(wrong_key_payload, "wrong-key")
        show_response(status_code, body, "Wrong API key was accepted")
    except requests.exceptions.ConnectionError:
        st.error("External API is not reachable.")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
