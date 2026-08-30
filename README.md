# LoanScope

Analyze loan data — or check your own approval chances before you apply.

🔗 **Live App:** [Click here to try LoanScope](https://loan-approval-prediction-6tuyf67r633fsqe4uzvfhq.streamlit.app/)

![LoanScope Home Page](screenshots/homepage.png)
![Analyze My Data](screenshots/analyze1.png)
![Analyze My Data](screenshots/analyze2.png)
![Check Eligibility](screenshots/eligibility1.png)
![Check Eligibility](screenshots/eligibility2.png)

## What it does

LoanScope is a machine learning web app with two modes:

- **📊 Analyze My Data** — Upload your own loan dataset, explore it with interactive charts, train a logistic regression model, and predict outcomes for new applicants.
- **✅ Check Eligibility** — No dataset needed. Answer a few simple questions and instantly see your estimated loan approval chances, along with personalized tips to improve your odds.

## Features

- CSV upload with automatic data cleaning (missing values, encoding, scaling)
- Interactive visualizations (approval distribution, correlation heatmap)
- Logistic Regression model training and evaluation (accuracy, confusion matrix, classification report)
- Real-time prediction for new applicants
- Rule-based suggestion engine comparing user inputs to approved applicant profiles

## Tech Stack

- **Python**
- **Streamlit** — web app framework
- **scikit-learn** — machine learning (Logistic Regression)
- **pandas** — data processing
- **Plotly** — interactive charts

## Project Structure

```
loan_approval_prediction/
├── main.py                  # Home page
├── model_utils.py           # ML logic (preprocessing, training, evaluation)
├── suggestions.py           # Rule-based approval tips
├── data/                    # Default dataset
├── pages/
│   ├── Analyze_My_Data.py   # Mode A: upload & analyze
│   └── Check_Eligibility.py # Mode B: quick eligibility check
``` 

## Run Locally

​```
uv sync
uv run streamlit run main.py
​```

## Disclaimer
This project is for educational and portfolio purposes only. Predictions do not represent real bank decisions.
