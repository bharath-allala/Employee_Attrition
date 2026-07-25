"""
Shared feature-engineering logic for the Employee Attrition project.

This module is imported both by the training notebook and by the Streamlit
app. Keeping the logic here (instead of inline in the notebook) means the
pickled scikit-learn Pipeline can find `add_engineered_features` at
`src.feature_engineering.add_engineered_features` when it is unpickled
inside the Streamlit app.
"""

import pandas as pd

# Columns present in the raw IBM HR Attrition CSV that carry no predictive
# signal (constants) or are just an identifier.
COLUMNS_TO_DROP = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]

# The exact set of raw input columns the pipeline expects, in a sensible
# order for building a data-entry form. This is every original column
# except the target ("Attrition") and the columns dropped above.
RAW_FEATURE_COLUMNS = [
    "Age",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EnvironmentSatisfaction",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

CATEGORICAL_COLUMNS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

# Numeric columns AFTER engineered features have been added.
ENGINEERED_NUMERIC_COLUMNS = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "TenureRatio",
    "AvgSatisfaction",
    "IncomePerJobLevel",
    "YearsSincePromotionRatio",
    "PromotionStagnation",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a handful of engineered features to the raw HR dataframe.

    Expects the raw columns in RAW_FEATURE_COLUMNS to be present (constants
    such as EmployeeCount/Over18/StandardHours/EmployeeNumber should already
    have been dropped). Returns a NEW dataframe; does not mutate the input.
    """
    df = df.copy()

    # How much of the person's total working life has been spent at this
    # company (a high ratio can indicate lower external mobility/loyalty).
    df["TenureRatio"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)

    # A single blended satisfaction score across the four satisfaction-style
    # survey questions.
    satisfaction_cols = [
        "EnvironmentSatisfaction",
        "JobSatisfaction",
        "RelationshipSatisfaction",
        "WorkLifeBalance",
    ]
    df["AvgSatisfaction"] = df[satisfaction_cols].mean(axis=1)

    # Income normalized by job level, to compare pay relative to seniority.
    df["IncomePerJobLevel"] = df["MonthlyIncome"] / (df["JobLevel"] + 1)

    # How much of the person's tenure has passed since their last promotion.
    df["YearsSincePromotionRatio"] = df["YearsSinceLastPromotion"] / (
        df["YearsAtCompany"] + 1
    )

    # Flag employees who have been at the company a while with no recent
    # promotion - a classic attrition risk signal.
    df["PromotionStagnation"] = (
        (df["YearsSinceLastPromotion"] > 5) & (df["YearsAtCompany"] > 5)
    ).astype(int)

    return df


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop constant/ID columns from a freshly-loaded raw CSV dataframe."""
    cols_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    return df.drop(columns=cols_present)
