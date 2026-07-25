# 👨‍💼 Employee Attrition Prediction

## 📌 Project Overview

Employee attrition is a major challenge for organizations as it impacts productivity, increases recruitment costs, and affects overall business performance.

This project uses Machine Learning to predict whether an employee is likely to leave the company based on various demographic, job-related, and performance factors. The goal is to help HR teams identify employees at risk of attrition and take proactive retention measures.

An end-to-end machine learning project that predicts employee attrition using the
[IBM HR Analytics Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
dataset — from raw CSV to a deployable Streamlit web app.

## What's inside

```
attrition_project/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Raw dataset
├── notebooks/
│   └── 01_employee_attrition_pipeline.ipynb    # Full analysis: EDA -> cleaning ->
│                                                # feature engineering -> model
│                                                # comparison -> tuning -> evaluation
├── src/
│   └── feature_engineering.py                  # Shared feature-engineering logic,
│                                                # imported by BOTH the notebook and
│                                                # the Streamlit app (keeps train/serve
│                                                # feature logic identical)
├── models/
│   ├── attrition_pipeline.pkl                  # Full fitted sklearn Pipeline
│   │                                            # (feature engineering + encoding +
│   │                                            #  scaling + Logistic Regression)
│   └── model_metadata.json                     # Metrics + expected input schema
├── app/
│   └── streamlit_app.py                        # Interactive prediction web app
├── requirements.txt
└── README.md
```

## The model

- **Problem:** Binary classification — will an employee leave (`Attrition = Yes/No`)?
- **Key challenge:** The dataset is imbalanced (~16% attrition), so we optimize for
  **recall / ROC-AUC** on the leaving class rather than raw accuracy.
- **Final model:** Tuned **Logistic Regression** (`class_weight='balanced'`), chosen
  after comparing against Random Forest, Gradient Boosting, XGBoost, and SVC — it gave
  the best combination of ROC-AUC and recall on employees who actually leave, and is
  fully interpretable for HR stakeholders.
- **Feature engineering:** `TenureRatio`, `AvgSatisfaction`, `IncomePerJobLevel`,
  `YearsSincePromotionRatio`, and `PromotionStagnation` — see
  `src/feature_engineering.py` for details and rationale.
- Full details, EDA charts, and business insights are in the notebook.

Live Link : https://employeeattrition-4wsk64zztrrrxexvkc3fb2.streamlit.app/

## 🚀 Features

- Predict employee attrition using Machine Learning
- Data preprocessing and feature engineering
- Exploratory Data Analysis (EDA)
- Model training and evaluation
- Random Forest Classifier for prediction
- Model serialization using Pickle
- Interactive prediction interface (Streamlit)

- ## 📊 Dataset

The dataset contains employee-related information such as:

- Age
- Gender
- Department
- Job Role
- Education
- Monthly Income
- Business Travel
- Job Satisfaction
- Environment Satisfaction
- Work-Life Balance
- Years at Company
- Years Since Last Promotion
- Overtime
- Performance Rating
- Attrition (Target Variable)



