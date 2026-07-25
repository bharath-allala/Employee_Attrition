# Employee Attrition Prediction

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

## Running the notebook

```bash
cd attrition_project
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
pip install jupyter nbformat ipykernel
jupyter notebook notebooks/01_employee_attrition_pipeline.ipynb
```

Running the notebook end-to-end regenerates `models/attrition_pipeline.pkl` and
`models/model_metadata.json`.

## Running the Streamlit app locally

```bash
cd attrition_project
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Fill in the employee-details form and click **Predict Attrition Risk** to get a live
prediction and probability score.

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repository (see below).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and click **New app**.
3. Select your repository/branch, and set:
   - **Main file path:** `app/streamlit_app.py`
4. Streamlit Cloud will install `requirements.txt` automatically and deploy the app.

No secrets or API keys are required — the app only depends on the checked-in
`models/attrition_pipeline.pkl` artifact.

## Pushing to GitHub

```bash
cd attrition_project
git init
git add .
git commit -m "Employee attrition prediction: EDA, pipeline, and Streamlit app"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

`models/attrition_pipeline.pkl` is small (~10 KB) and is safe to commit directly, so the
Streamlit app works out of the box without a retraining step in CI.

## Retraining

If the underlying HR data changes, re-run the notebook top-to-bottom — it will
recompute EDA, retrain and re-tune the model, re-evaluate it, and overwrite
`models/attrition_pipeline.pkl` / `models/model_metadata.json` with the refreshed
artifacts. No changes to `app/streamlit_app.py` are needed as long as the raw input
schema (`src/feature_engineering.RAW_FEATURE_COLUMNS`) stays the same.
