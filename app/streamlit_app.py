"""
Employee Attrition Prediction — Streamlit App
==============================================
Loads the trained pipeline (models/attrition_pipeline.pkl) and lets an HR user
enter a single employee's details to get a live attrition risk prediction.

Run locally with:
    streamlit run app/streamlit_app.py
"""

import json
import os
import sys

import joblib
import pandas as pd
import streamlit as st

# Make the shared `src` package importable regardless of the working directory
# the app is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_engineering import RAW_FEATURE_COLUMNS  # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "models", "attrition_pipeline.pkl")
METADATA_PATH = os.path.join(ROOT_DIR, "models", "model_metadata.json")

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="\U0001F4CA",
    layout="wide",
)


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    return {}


pipeline = load_pipeline()
metadata = load_metadata()

st.title("Employee Attrition Predictor")
st.write(
    "Enter an employee's details below to estimate their probability of leaving "
    "the company. This app is powered by a Logistic Regression pipeline trained "
    "on the IBM HR Analytics Employee Attrition dataset."
)

if metadata:
    with st.expander("About this model"):
        st.markdown(
            f"""
- **Model type:** {metadata.get('model_type', 'N/A')}
- **Cross-validated ROC-AUC:** {metadata.get('cv_roc_auc', 0):.3f}
- **Held-out test ROC-AUC:** {metadata.get('test_roc_auc', 0):.3f}
- Trained on the IBM HR Analytics Employee Attrition dataset (1,470 employees).
"""
        )

st.divider()
st.subheader("Employee Details")

with st.form("employee_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Personal**")
        age = st.slider("Age", 18, 60, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        distance_from_home = st.slider("Distance From Home (miles)", 1, 29, 5)
        education = st.selectbox(
            "Education Level",
            options=[1, 2, 3, 4, 5],
            index=2,
            format_func=lambda x: {
                1: "1 - Below College", 2: "2 - College", 3: "3 - Bachelor",
                4: "4 - Master", 5: "5 - Doctor",
            }[x],
        )
        education_field = st.selectbox(
            "Education Field",
            ["Life Sciences", "Medical", "Marketing", "Technical Degree",
             "Human Resources", "Other"],
        )

        st.markdown("**Work-Life**")
        overtime = st.selectbox("Works OverTime", ["Yes", "No"])
        business_travel = st.selectbox(
            "Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
        )
        work_life_balance = st.select_slider(
            "Work-Life Balance (1=Bad, 4=Best)", options=[1, 2, 3, 4], value=3
        )
        training_times_last_year = st.slider("Training Times Last Year", 0, 6, 2)

    with col2:
        st.markdown("**Job**")
        department = st.selectbox(
            "Department", ["Sales", "Research & Development", "Human Resources"]
        )
        job_role = st.selectbox(
            "Job Role",
            ["Sales Executive", "Research Scientist", "Laboratory Technician",
             "Manufacturing Director", "Healthcare Representative", "Manager",
             "Sales Representative", "Research Director", "Human Resources"],
        )
        job_level = st.select_slider("Job Level (1=Entry, 5=Senior)", options=[1, 2, 3, 4, 5], value=2)
        job_involvement = st.select_slider(
            "Job Involvement (1=Low, 4=High)", options=[1, 2, 3, 4], value=3
        )
        performance_rating = st.select_slider(
            "Performance Rating (1=Low, 4=Outstanding)", options=[1, 2, 3, 4], value=3
        )
        stock_option_level = st.select_slider("Stock Option Level", options=[0, 1, 2, 3], value=0)

        st.markdown("**Tenure**")
        total_working_years = st.slider("Total Working Years", 0, 40, 8)
        years_at_company = st.slider("Years At Company", 0, 40, 5)
        years_in_current_role = st.slider("Years In Current Role", 0, 18, 3)
        years_with_curr_manager = st.slider("Years With Current Manager", 0, 17, 3)
        years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
        num_companies_worked = st.slider("Num Companies Worked", 0, 9, 2)

    with col3:
        st.markdown("**Compensation**")
        monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=100)
        daily_rate = st.slider("Daily Rate", 100, 1500, 800)
        hourly_rate = st.slider("Hourly Rate", 30, 100, 65)
        monthly_rate = st.slider("Monthly Rate", 2000, 27000, 14000)
        percent_salary_hike = st.slider("Percent Salary Hike (last review)", 11, 25, 15)

        st.markdown("**Satisfaction**")
        environment_satisfaction = st.select_slider(
            "Environment Satisfaction (1=Low, 4=High)", options=[1, 2, 3, 4], value=3
        )
        job_satisfaction = st.select_slider(
            "Job Satisfaction (1=Low, 4=High)", options=[1, 2, 3, 4], value=3
        )
        relationship_satisfaction = st.select_slider(
            "Relationship Satisfaction (1=Low, 4=High)", options=[1, 2, 3, 4], value=3
        )

    submitted = st.form_submit_button("Predict Attrition Risk", use_container_width=True, type="primary")

if submitted:
    employee = {
        "Age": age,
        "BusinessTravel": business_travel,
        "DailyRate": daily_rate,
        "Department": department,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EducationField": education_field,
        "EnvironmentSatisfaction": environment_satisfaction,
        "Gender": gender,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": overtime,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
    }

    input_df = pd.DataFrame([employee])[RAW_FEATURE_COLUMNS]

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0, 1]

    st.divider()
    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns([1, 2])
    with result_col1:
        if prediction == 1:
            st.error("### \u26a0\ufe0f High Risk of Attrition")
        else:
            st.success("### \u2705 Low Risk of Attrition")
        st.metric("Predicted Attrition Probability", f"{probability * 100:.1f}%")

    with result_col2:
        st.progress(min(max(probability, 0.0), 1.0))
        if probability >= 0.5:
            st.write(
                "This employee shows a **elevated risk profile**. Consider reviewing "
                "workload/overtime, compensation relative to role, satisfaction, and "
                "career progression (promotion history) as potential retention levers."
            )
        else:
            st.write(
                "This employee currently shows a **lower risk profile** based on the "
                "factors entered."
            )

    with st.expander("Show raw input sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "value"}))

st.divider()
st.caption(
    "Model: Logistic Regression pipeline trained on the IBM HR Analytics Employee "
    "Attrition dataset. Predictions are probabilistic estimates for decision support, "
    "not a substitute for HR judgment."
)
