# Customer Churn Prediction

End-to-end ML pipeline for telecom customer churn using Random Forest.

## Model

- **Algorithm:** Random Forest Classifier
- **Features:** 28 engineered features (gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, PaperlessBilling, MonthlyCharges, TotalCharges, avg_monthly_spend, tenure bins, one-hot encoded service categories)
- **Threshold:** F1-optimized on validation split
- **Class weights:** balanced
- **Metrics tracked:** ROC-AUC, accuracy, precision, recall, F1, confusion matrix

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train Model

```bash
python -m src.deploy_model --train
```

Generates:
- `models/rf_churn_model.joblib`
- `models/rf_churn_model_meta.json`
- `data/telco_churn_processed.csv`

## Run Dashboard

```bash
streamlit run app.py
```

URL: http://localhost:8501

## Run API

```bash
uvicorn src.api:app --reload
```

URL: http://localhost:8000/docs

## Other Commands

| Command | Purpose |
|---------|---------|
| `python -m src.crm_integration` | Score all customers & export CRM CSV |
| `python -m src.drift_monitor` | Run PSI drift analysis |
| `python -m src.shap_analysis` | Generate SHAP plots |
| `python -m src.ab_testing` | Power & revenue simulation |
# End-to-End Customer Churn Prediction

ML pipeline for predicting customer churn. Covers data ingestion, preprocessing, model training, and a small web app.
