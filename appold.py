import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="💳",
    layout="wide"
)

ARTIFACT_DIR = Path(".")


# ------------------------------------------------------------
# Load model artifacts
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = ARTIFACT_DIR / "insurance_premium_random_forest.joblib"
    return joblib.load(model_path)


@st.cache_data
def load_features():
    features_path = ARTIFACT_DIR / "model_features.json"
    with open(features_path, "r") as f:
        return json.load(f)


@st.cache_data
def load_metadata():
    metadata_path = ARTIFACT_DIR / "model_metadata.json"
    with open(metadata_path, "r") as f:
        return json.load(f)


model = load_model()
feature_list = load_features()
metadata = load_metadata()


# ------------------------------------------------------------
# Feature preparation
# ------------------------------------------------------------
def prepare_features(user_input: dict, feature_order: list[str]) -> pd.DataFrame:
    df_input = pd.DataFrame([user_input])

    df_input["BMI"] = df_input["Weight"] / ((df_input["Height"] / 100) ** 2)
    df_input["TotalHealthRisks"] = df_input[
        [
            "Diabetes",
            "BloodPressureProblems",
            "AnyTransplants",
            "AnyChronicDiseases",
            "KnownAllergies",
            "HistoryOfCancerInFamily",
        ]
    ].sum(axis=1)

    df_input["Age_BMI_Interaction"] = df_input["Age"] * df_input["BMI"]
    df_input["Age_Surgery_Interaction"] = (
        df_input["Age"] * df_input["NumberOfMajorSurgeries"]
    )

    return df_input[feature_order]


def risk_label(total_risks: int, surgeries: int, transplant: int, chronic: int) -> str:
    if transplant == 1 or surgeries >= 2 or total_risks >= 4:
        return "High Risk Profile"
    if chronic == 1 or total_risks >= 2 or surgeries == 1:
        return "Moderate Risk Profile"
    return "Lower Risk Profile"


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("Insurance Premium Prediction App")
st.markdown(
    "Estimate insurance premium using health and demographic information."
)

with st.expander("Model Details", expanded=False):
    st.write(f"**Model:** {metadata.get('model_name', 'N/A')}")
    st.write(f"**Number of features:** {metadata.get('n_features', 'N/A')}")
    st.write(f"**Rows used for final training:** {metadata.get('n_rows_used_for_training', 'N/A')}")

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=66, value=35)
    height = st.slider("Height (cm)", min_value=145, max_value=188, value=170)
    weight = st.slider("Weight (kg)", min_value=51, max_value=132, value=70)
    surgeries = st.selectbox("Number of Major Surgeries", options=[0, 1, 2, 3], index=0)

with col2:
    diabetes = st.selectbox("Diabetes", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    bp = st.selectbox("Blood Pressure Problems", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    transplant = st.selectbox("Any Transplants", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    chronic = st.selectbox("Any Chronic Diseases", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    allergies = st.selectbox("Known Allergies", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    cancer_family = st.selectbox("History of Cancer in Family", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

predict_clicked = st.button("Predict Premium", type="primary")

if predict_clicked:
    raw_input = {
        "Age": age,
        "Diabetes": diabetes,
        "BloodPressureProblems": bp,
        "AnyTransplants": transplant,
        "AnyChronicDiseases": chronic,
        "Height": height,
        "Weight": weight,
        "KnownAllergies": allergies,
        "HistoryOfCancerInFamily": cancer_family,
        "NumberOfMajorSurgeries": surgeries,
    }

    prepared_df = prepare_features(raw_input, feature_list)
    prediction = model.predict(prepared_df)[0]

    bmi = prepared_df["BMI"].iloc[0]
    total_risks = int(prepared_df["TotalHealthRisks"].iloc[0])
    profile_label = risk_label(total_risks, surgeries, transplant, chronic)

    st.success(f"Estimated Premium: ₹ {prediction:,.0f}")

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("BMI", f"{bmi:.2f}")
    metric2.metric("Total Health Risks", total_risks)
    metric3.metric("Profile Segment", profile_label)

    st.subheader("Prediction Summary")
    summary_df = pd.DataFrame(
        {
            "Input Feature": [
                "Age",
                "Height",
                "Weight",
                "Diabetes",
                "BloodPressureProblems",
                "AnyTransplants",
                "AnyChronicDiseases",
                "KnownAllergies",
                "HistoryOfCancerInFamily",
                "NumberOfMajorSurgeries",
                "BMI",
                "TotalHealthRisks",
                "Age_BMI_Interaction",
                "Age_Surgery_Interaction",
            ],
            "Value": [
                age,
                height,
                weight,
                diabetes,
                bp,
                transplant,
                chronic,
                allergies,
                cancer_family,
                surgeries,
                round(bmi, 2),
                total_risks,
                round(prepared_df["Age_BMI_Interaction"].iloc[0], 2),
                round(prepared_df["Age_Surgery_Interaction"].iloc[0], 2),
            ],
        }
    )

    st.dataframe(summary_df, use_container_width=True)

st.markdown("---")
st.caption("Built as part of the Insurance Cost Prediction portfolio project.")