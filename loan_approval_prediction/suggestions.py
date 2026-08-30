import pandas as pd


def get_suggestions(user_input, df, target_col="loan_status"):
    df = df.copy()
    df.columns = df.columns.str.strip()

    approved_df = df[df[target_col].astype(str).str.strip().str.lower() == "approved"]

    tips = []

    if "cibil_score" in approved_df.columns:
        avg_cibil = approved_df["cibil_score"].mean()
        if user_input["cibil_score"] < avg_cibil - 50:
            tips.append(
                f"Approved applicants typically have a CIBIL score around {avg_cibil:.0f}. "
                f"Yours is {user_input['cibil_score']}, which is notably lower."
            )

    if "income_annum" in approved_df.columns and "loan_amount" in approved_df.columns:
        approved_ratio = (approved_df["loan_amount"] / approved_df["income_annum"]).mean()
        user_ratio = user_input["loan_amount"] / user_input["income_annum"] if user_input["income_annum"] > 0 else 999

        if user_ratio > approved_ratio + 1:
            tips.append(
                "Your requested loan amount is high relative to your income, compared to typically approved applicants. "
                "Consider requesting a smaller loan or increasing your declared income."
            )

    if "loan_term" in approved_df.columns:
        avg_term = approved_df["loan_term"].mean()
        if user_input["loan_term"] > avg_term + 5:
            tips.append(
                f"Approved applicants often choose shorter loan terms (around {avg_term:.0f} years). "
                f"A shorter term may improve your chances."
            )

    return tips