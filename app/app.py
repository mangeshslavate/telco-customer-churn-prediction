import streamlit as st
import pandas as pd
import joblib
st.write("App Started Successfully")

# Load Pipeline
pipeline = joblib.load("../models/churn_pipeline.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to churn.")

st.header("Customer Information")

# -------------------------
# Input Fields
# -------------------------

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col2:

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

# Remaining Fields

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

# -------------------------
# Predict Button
# -------------------------

if st.button("Predict Churn"):

    avg_monthly_spend = total_charges / (tenure + 1)

    if tenure <= 12:
        tenure_group = "0-12"
    elif tenure <= 24:
        tenure_group = "13-24"
    elif tenure <= 48:
        tenure_group = "25-48"
    else:
        tenure_group = "49-72"

    if monthly_charges <= 35:
        charge_category = "Low"
    elif monthly_charges <= 70:
        charge_category = "Medium"
    else:
        charge_category = "High"

    input_df = pd.DataFrame({
        "gender":[gender],
        "SeniorCitizen":[senior_citizen],
        "Partner":[partner],
        "Dependents":[dependents],
        "tenure":[tenure],
        "PhoneService":[phone_service],
        "MultipleLines":[multiple_lines],
        "InternetService":[internet_service],
        "OnlineSecurity":[online_security],
        "OnlineBackup":[online_backup],
        "DeviceProtection":[device_protection],
        "TechSupport":[tech_support],
        "StreamingTV":[streaming_tv],
        "StreamingMovies":[streaming_movies],
        "Contract":[contract],
        "PaperlessBilling":[paperless_billing],
        "PaymentMethod":[payment_method],
        "MonthlyCharges":[monthly_charges],
        "TotalCharges":[total_charges],
        "AvgMonthlySpend":[avg_monthly_spend],
        "TenureGroup":[tenure_group],
        "ChargeCategory":[charge_category]
    })

    prediction = pipeline.predict(input_df)[0]

    probability = pipeline.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            f"⚠️ Customer is likely to CHURN\n\nProbability: {probability:.2%}"
        )

    else:

        st.success(
            f"✅ Customer is likely to STAY\n\nProbability of Churn: {probability:.2%}"
        )