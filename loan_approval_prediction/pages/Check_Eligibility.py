import streamlit as st
import pandas as pd
from model_utils import preprocess_data, train_logistic_regression
from sklearn.preprocessing import LabelEncoder
from suggestions import get_suggestions

st.title("Check Eligibility")
st.write("No data needed — just answer a few questions to see your loan approval chances.")
st.info("This tool is for educational purposes only and does not represent an actual bank decision. Real loan approval depends on many more factors.")

df = pd.read_csv("data/loan_approval_dataset.csv")
df.columns = df.columns.str.strip()

target_col = "loan_status"
id_cols = [col for col in df.columns if col.lower().endswith("_id") or col.lower() == "id"]

if "eligibility_model" not in st.session_state:
    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col] + id_cols)

    processed_X, encoders, scaler = preprocess_data(X_raw)

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y_raw)

    model = train_logistic_regression(processed_X, y)

    st.session_state["eligibility_model"] = model
    st.session_state["eligibility_encoders"] = encoders
    st.session_state["eligibility_scaler"] = scaler
    st.session_state["eligibility_y_encoder"] = y_encoder
    st.session_state["eligibility_feature_order"] = processed_X.columns.tolist()

st.success("Ready! Fill in your details below.")

st.subheader("Your Details")

no_of_dependents = st.number_input("How many dependents do you have?", min_value=0, max_value=10, value=0, step=1)

education = st.selectbox("What's your education level?", ["Graduate", "Not Graduate"])

self_employed = st.selectbox("Are you self-employed?", ["No", "Yes"])

income_annum = st.number_input("What's your annual income (in ₹)?", min_value=0, value=500000, step=10000)

loan_amount = st.number_input("How much loan are you requesting (in ₹)?", min_value=0, value=1000000, step=10000)

loan_term = st.number_input("Loan term (in years)?", min_value=1, max_value=30, value=10, step=1)

cibil_score = st.number_input("What's your CIBIL score?", min_value=300, max_value=900, value=650, step=1)

residential_assets_value = st.number_input("Value of residential assets (in ₹)?", min_value=0, value=0, step=10000)

commercial_assets_value = st.number_input("Value of commercial assets (in ₹)?", min_value=0, value=0, step=10000)

luxury_assets_value = st.number_input("Value of luxury assets (in ₹)?", min_value=0, value=0, step=10000)

bank_asset_value = st.number_input("Value of bank assets (in ₹)?", min_value=0, value=0, step=10000)

if st.button("Check My Eligibility"):
    input_data = {
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
    }

    input_df = pd.DataFrame([input_data])

    processed_input, _, _ = preprocess_data(
        input_df,
        encoders=st.session_state["eligibility_encoders"],
        scaler=st.session_state["eligibility_scaler"]
    )

    processed_input = processed_input[st.session_state["eligibility_feature_order"]]

    model = st.session_state["eligibility_model"]
    prediction = model.predict(processed_input)[0]
    predicted_label = str(st.session_state["eligibility_y_encoder"].inverse_transform([prediction])[0]).strip()

    if str(predicted_label).strip().lower() == "approved":
        st.success(f"Good news! Based on your details, you're likely to be {predicted_label}.")
    else:
        st.error(f"Based on your details, you're likely to be {predicted_label}.")
        tips = get_suggestions(input_data, df)
        if tips:
            st.subheader("Tips to improve your chances")
            for tip in tips:
                st.write("- " + tip)