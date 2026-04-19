import json
from pathlib import Path

import joblib
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
    return joblib.load(ARTIFACT_DIR / "insurance_premium_random_forest.joblib")


@st.cache_data
def load_features():
    with open(ARTIFACT_DIR / "model_features.json", "r") as f:
        return json.load(f)


@st.cache_data
def load_metadata():
    with open(ARTIFACT_DIR / "model_metadata.json", "r") as f:
        return json.load(f)


model = load_model()
feature_list = load_features()
metadata = load_metadata()


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def prepare_features(user_input: dict, feature_order: list[str]) -> pd.DataFrame:
    input_df = pd.DataFrame([user_input])

    input_df["BMI"] = input_df["Weight"] / ((input_df["Height"] / 100) ** 2)
    input_df["TotalHealthRisks"] = input_df[
        [
            "Diabetes",
            "BloodPressureProblems",
            "AnyTransplants",
            "AnyChronicDiseases",
            "KnownAllergies",
            "HistoryOfCancerInFamily",
        ]
    ].sum(axis=1)

    input_df["Age_BMI_Interaction"] = input_df["Age"] * input_df["BMI"]
    input_df["Age_Surgery_Interaction"] = (
        input_df["Age"] * input_df["NumberOfMajorSurgeries"]
    )

    return input_df[feature_order]


def get_bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    return "Obese"


def risk_label(total_risks: int, surgeries: int, transplant: int, chronic: int) -> str:
    if transplant == 1 or surgeries >= 2 or total_risks >= 4:
        return "High Risk Profile"
    if chronic == 1 or total_risks >= 2 or surgeries == 1:
        return "Moderate Risk Profile"
    return "Lower Risk Profile"


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.header("About the App")
st.sidebar.write(
    "This app predicts insurance premium using health and demographic inputs."
)
st.sidebar.write(f"**Model:** {metadata.get('model_name', 'N/A')}")
st.sidebar.write(f"**Number of features:** {metadata.get('n_features', 'N/A')}")
st.sidebar.write(f"**Training rows used:** {metadata.get('n_rows_used_for_training', 'N/A')}")

st.sidebar.markdown("---")
st.sidebar.write("**Input guidance**")
st.sidebar.write("- Use actual health-condition values")
st.sidebar.write("- Height should be in cm")
st.sidebar.write("- Weight should be in kg")
st.sidebar.write("- Surgeries means major surgeries only")


# ------------------------------------------------------------
# Main UI
# ------------------------------------------------------------
st.title("Insurance Premium Prediction App")
st.markdown(
    "Estimate the likely insurance premium based on demographic and health-related information."
)

st.subheader("Customer Input Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=66, value=35)
    height = st.slider("Height (cm)", min_value=145, max_value=188, value=170)
    weight = st.slider("Weight (kg)", min_value=51, max_value=132, value=70)
    surgeries = st.selectbox("Number of Major Surgeries", [0, 1, 2, 3], index=0)

with col2:
    diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    bp = st.selectbox("Blood Pressure Problems", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    transplant = st.selectbox("Any Transplants", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    chronic = st.selectbox("Any Chronic Diseases", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    allergies = st.selectbox("Known Allergies", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    cancer_family = st.selectbox("History of Cancer in Family", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

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

    bmi = float(prepared_df["BMI"].iloc[0])
    bmi_category = get_bmi_category(bmi)
    total_risks = int(prepared_df["TotalHealthRisks"].iloc[0])
    profile_label = risk_label(total_risks, surgeries, transplant, chronic)

    st.success(f"Estimated Premium: ₹ {prediction:,.0f}")

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("BMI", f"{bmi:.2f}")
    metric2.metric("BMI Category", bmi_category)
    metric3.metric("Total Health Risks", total_risks)
    metric4.metric("Risk Profile", profile_label)

    st.subheader("Prediction Summary")
    summary_df = pd.DataFrame({
        "Feature": [
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
            "Age_Surgery_Interaction"
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
            round(float(prepared_df["Age_BMI_Interaction"].iloc[0]), 2),
            round(float(prepared_df["Age_Surgery_Interaction"].iloc[0]), 2),
        ]
    })

    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Interpretation")
    st.write(
        f"""
        - The predicted premium is influenced mainly by **age**, **major medical severity**,
          and combined risk factors.
        - This profile is currently classified as **{profile_label}**.
        - BMI category for this individual is **{bmi_category}**.
        """
    )

st.markdown("---")
st.caption("Built for the Insurance Cost Prediction portfolio project using Streamlit.")