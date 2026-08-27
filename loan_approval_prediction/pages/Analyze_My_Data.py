import streamlit as st
import pandas as pd
import plotly.express as px

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