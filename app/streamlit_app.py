"""Streamlit ops dashboard for the churn prediction service.

Calls the FastAPI service over HTTP (so this demo *uses* the API the
same way a real retention tool would). The URL is configurable via
the ``CHURN_API_URL`` environment variable; default is the local
uvicorn at :8000.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import streamlit as st

from churn.data.load import load_raw

API_URL = os.environ.get("CHURN_API_URL", "http://localhost:8000")
DATA_PATH = Path("data/raw")

st.set_page_config(page_title="Churn Ops", page_icon="📉", layout="wide")
st.title("📉 Churn Ops Dashboard")
st.caption(f"backed by `{API_URL}` — calls `/predict`, `/explain`, `/model/info`")


@st.cache_data(show_spinner=False)
def load_test_customers() -> pd.DataFrame:
    df = load_raw(DATA_PATH)
    return df.head(200).reset_index(drop=True)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_model_info() -> dict[str, Any]:
    r = httpx.get(f"{API_URL}/model/info", timeout=5)
    r.raise_for_status()
    return r.json()


def call_predict(customer: dict[str, Any]) -> dict[str, Any]:
    r = httpx.post(f"{API_URL}/predict", json=customer, timeout=10)
    r.raise_for_status()
    return r.json()


def call_explain(customer: dict[str, Any], top_k: int = 8) -> dict[str, Any]:
    r = httpx.post(f"{API_URL}/explain", params={"top_k": top_k}, json=customer, timeout=15)
    r.raise_for_status()
    return r.json()


with st.sidebar:
    st.header("Mode")
    mode = st.radio(
        "Pick a workflow",
        ["Customer lookup", "What-if simulator", "About"],
        label_visibility="collapsed",
    )
    st.divider()
    try:
        info = fetch_model_info()
        st.success(f"Model {info['version']} loaded")
        st.metric("Decision threshold", f"{info['decision_threshold']:.3f}")
        st.metric("Test ROC-AUC", f"{info['training_metrics']['test_roc_auc']:.3f}")
    except Exception as exc:
        st.error(f"API not reachable at {API_URL}")
        st.caption(str(exc))


def render_prediction(pred: dict[str, Any]) -> None:
    cols = st.columns(3)
    cols[0].metric("Churn probability", f"{pred['churn_probability']:.1%}")
    cols[1].metric("Decision", pred["recommended_action"].title())
    cols[2].metric("Threshold", f"{pred['threshold_used']:.3f}")


def render_explanation(explanation: dict[str, Any]) -> None:
    df = pd.DataFrame(explanation["top_features"])
    df = df.rename(
        columns={
            "feature": "Feature",
            "shap_value": "SHAP value",
            "abs_value": "|SHAP|",
        }
    )
    df["Direction"] = df["SHAP value"].apply(lambda v: "↑ churn risk" if v > 0 else "↓ churn risk")
    st.dataframe(
        df[["Feature", "SHAP value", "Direction"]],
        hide_index=True,
        use_container_width=True,
    )


if mode == "Customer lookup":
    st.subheader("Pick a customer to score")
    customers = load_test_customers()
    selected_id = st.selectbox(
        "customerID",
        options=customers["customerID"].tolist(),
        index=0,
    )
    row = customers[customers["customerID"] == selected_id].iloc[0]
    customer_payload = {
        col: (None if pd.isna(row[col]) else row[col])
        for col in customers.columns
        if col != "Churn"
    }

    if st.button("Score this customer", type="primary"):
        with st.spinner("Calling /predict and /explain…"):
            pred = call_predict(customer_payload)
            explanation = call_explain(customer_payload, top_k=8)
        render_prediction(pred)
        st.divider()
        st.subheader("Why? (top SHAP contributions)")
        render_explanation(explanation)
        with st.expander("Raw customer row"):
            st.json(customer_payload)

elif mode == "What-if simulator":
    st.subheader("Adjust the customer and see what happens")
    col_left, col_right = st.columns([1, 1])
    with col_left:
        tenure = st.slider("tenure (months)", 0, 72, 12)
        monthly = st.slider("MonthlyCharges ($)", 15.0, 120.0, 65.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox(
            "PaymentMethod",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    with col_right:
        gender = st.selectbox("gender", ["Female", "Male"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        paperless = st.selectbox("PaperlessBilling", ["No", "Yes"])
        senior = st.selectbox("SeniorCitizen", [0, 1])

    service_default = "No internet service" if internet == "No" else "No"
    customer_payload = {
        "customerID": "what-if-001",
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet,
        "OnlineSecurity": service_default,
        "OnlineBackup": service_default,
        "DeviceProtection": service_default,
        "TechSupport": service_default,
        "StreamingTV": service_default,
        "StreamingMovies": service_default,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(monthly * max(tenure, 1)),
    }
    if st.button("Score what-if", type="primary"):
        with st.spinner("Calling /predict and /explain…"):
            pred = call_predict(customer_payload)
            explanation = call_explain(customer_payload, top_k=8)
        render_prediction(pred)
        st.divider()
        st.subheader("Drivers")
        render_explanation(explanation)

else:  # About
    st.subheader("About this model")
    try:
        info = fetch_model_info()
        st.json(info)
    except Exception as exc:
        st.error(f"Couldn't reach the model info endpoint: {exc}")
    st.markdown(
        "Repository: [github.com/MeetLunagariya/churn-prediction]"
        "(https://github.com/MeetLunagariya/churn-prediction)"
    )
