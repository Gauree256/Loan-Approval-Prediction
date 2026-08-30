import streamlit as st

st.markdown(
    """
    <h1 style='text-align: center; color: #1E1B4B; font-size: 64px;'>LoanScope</h1>
    <p style='text-align: center; font-size: 18px; color: #4B5563;'>
    Analyze loan data — or check your own approval chances before you apply.
    </p>
    """,
    unsafe_allow_html=True
)

st.write("")
st.markdown(
    "<hr style='border: 2px solid #1E1B4B; margin-top: 20px; margin-bottom: 30px;'>",
    unsafe_allow_html=True
)

with st.container(border=True):
    st.subheader("📊 Analyze My Data")
    st.write("Upload a dataset, explore it, and train a model.")
    if st.button("Go to Analyze My Data"):
        st.switch_page("pages/Analyze_My_Data.py")

st.write("")

with st.container(border=True):
    st.subheader("✅ Check Eligibility")
    st.write("No data needed — just answer a few questions.")
    if st.button("Go to Check Eligibility"):
        st.switch_page("pages/Check_Eligibility.py")

st.divider()
st.caption("Built with Python, scikit-learn, and Streamlit — a machine learning project for loan approval prediction.")