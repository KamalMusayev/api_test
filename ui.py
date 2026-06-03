import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/predict"
API_KEY = "demo-secret-key"


st.set_page_config(
    page_title="Iris ML Prediction UI",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris ML Prediction API Demo")

st.write(
    "This simple UI sends flower measurements to the FastAPI `/predict` endpoint "
    "and shows the model prediction."
)

st.divider()

mode = st.radio(
    "Choose prediction mode:",
    ["Single prediction", "Batch prediction"]
)

headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}


if mode == "Single prediction":
    st.subheader("Single Prediction")

    sepal_length = st.number_input("Sepal length", value=5.1)
    sepal_width = st.number_input("Sepal width", value=3.5)
    petal_length = st.number_input("Petal length", value=1.4)
    petal_width = st.number_input("Petal width", value=0.2)

    payload = {
        "features": [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    }

    st.write("Request payload:")
    st.json(payload)

    if st.button("Predict"):
        try:
            response = requests.post(
                API_URL,
                json=payload,
                headers=headers,
                timeout=10
            )

            st.write("Status code:", response.status_code)

            if response.status_code == 200:
                st.success("Prediction successful")
                st.json(response.json())
            else:
                st.error("Request failed")
                st.json(response.json())

        except requests.exceptions.ConnectionError:
            st.error("FastAPI server is not running. Start it with: py -m uvicorn app:app --reload")

        except Exception as e:
            st.error(f"Unexpected error: {e}")


else:
    st.subheader("Batch Prediction")

    st.write("This sends multiple flower records to the same `/predict` endpoint.")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Flower 1")
        f1_sepal_length = st.number_input("Flower 1 sepal length", value=5.1)
        f1_sepal_width = st.number_input("Flower 1 sepal width", value=3.5)
        f1_petal_length = st.number_input("Flower 1 petal length", value=1.4)
        f1_petal_width = st.number_input("Flower 1 petal width", value=0.2)

    with col2:
        st.write("Flower 2")
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
                    f1_petal_width
                ]
            },
            {
                "id": "flower_2",
                "features": [
                    f2_sepal_length,
                    f2_sepal_width,
                    f2_petal_length,
                    f2_petal_width
                ]
            }
        ]
    }

    st.write("Request payload:")
    st.json(payload)

    if st.button("Predict batch"):
        try:
            response = requests.post(
                API_URL,
                json=payload,
                headers=headers,
                timeout=10
            )

            st.write("Status code:", response.status_code)

            if response.status_code == 200:
                st.success("Batch prediction successful")
                st.json(response.json())
            else:
                st.error("Request failed")
                st.json(response.json())

        except requests.exceptions.ConnectionError:
            st.error("FastAPI server is not running. Start it with: py -m uvicorn app:app --reload")

        except Exception as e:
            st.error(f"Unexpected error: {e}")


st.divider()

st.subheader("API Key Test")

wrong_key_payload = {
    "features": [5.1, 3.5, 1.4, 0.2]
}

if st.button("Test wrong API key"):
    try:
        response = requests.post(
            API_URL,
            json=wrong_key_payload,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": "wrong-key"
            },
            timeout=10
        )

        st.write("Status code:", response.status_code)
        st.json(response.json())

    except requests.exceptions.ConnectionError:
        st.error("FastAPI server is not running.")

    except Exception as e:
        st.error(f"Unexpected error: {e}")