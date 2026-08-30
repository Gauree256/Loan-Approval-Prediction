import streamlit as st
import pandas as pd
import plotly.express as px
from model_utils import preprocess_data, train_logistic_regression, evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title("Analyze My Data")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    st.session_state["df"] = pd.read_csv(uploaded_file)

if "df" in st.session_state:
    df = st.session_state["df"]
    st.write("Here's a preview of your data:")
    st.dataframe(df.head())

    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])

    st.write("Column names and types:")
    st.write(df.dtypes)

    st.write("Missing values per column:")
    st.write(df.isnull().sum())

    st.subheader("Loan Status Distribution")

    if "loan_status" in df.columns:
        status_counts = df["loan_status"].value_counts()
        st.bar_chart(status_counts)
    else:
        st.write("No 'loan_status' column found in this dataset.")

    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Blues")
        st.plotly_chart(fig)
    else:
        st.write("Not enough numeric columns for a correlation heatmap.")

    st.subheader("Train Model")

    target_col = "loan_status"
    id_cols = [col for col in df.columns if "id" in col.lower()]

    if target_col not in df.columns:
        st.warning("This dataset doesn't have a 'loan_status' column, so a model can't be trained.")
    elif st.button("Train Model"):
        y_raw = df[target_col]
        X_raw = df.drop(columns=[target_col] + id_cols)

        processed_X, encoders, scaler = preprocess_data(X_raw)

        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y_raw)
        X = processed_X

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = train_logistic_regression(X_train, y_train)
        results = evaluate_model(model, X_test, y_test)

        st.session_state["model"] = model
        st.session_state["results"] = results
        st.session_state["encoders"] = encoders
        st.session_state["scaler"] = scaler
        st.session_state["feature_cols_order"] = X.columns.tolist()
        st.session_state["y_encoder"] = y_encoder

        st.success("Model trained successfully!")

    if "results" in st.session_state:
        results = st.session_state["results"]

        st.subheader("Model Performance")

        st.metric("Accuracy", f"{results['accuracy']*100:.2f}%")

        st.write("Confusion Matrix")
        cm = results["confusion_matrix"]
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                            labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(fig_cm)

        st.write("Classification Report")
        report_df = pd.DataFrame(results["classification_report"]).transpose()
        st.dataframe(report_df)


    if "model" in st.session_state:
        st.subheader("Predict New Applicant")

        model = st.session_state["model"]
        feature_cols = [col for col in df.columns if col != target_col and col not in id_cols]

        input_data = {}
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                default_val = float(pd.to_numeric(df[col], errors="coerce").mean())
                input_data[col] = st.number_input(f"{col}", value=default_val, key=f"input_{col}")
            else:
                options = df[col].astype(str).str.strip().unique().tolist()
                input_data[col] = st.selectbox(f"{col}", options, key=f"input_{col}")

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])

            processed_input, _, _ = preprocess_data(
                input_df,
                encoders=st.session_state["encoders"],
                scaler=st.session_state["scaler"]
            )

            processed_input = processed_input[st.session_state["feature_cols_order"]]

            prediction = model.predict(processed_input)[0]
            predicted_label = st.session_state["y_encoder"].inverse_transform([prediction])[0]

            if str(predicted_label).strip().lower() == "approved":
                st.success(f"Prediction: {predicted_label} ✅")
            else:
                st.error(f"Prediction: {predicted_label} ❌")