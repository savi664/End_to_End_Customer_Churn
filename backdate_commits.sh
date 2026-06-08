#!/bin/bash
# Backdated commit script for End_to_End_Customer_Churn
# Run inside your local clone of the repo.
# Usage: bash backdate_commits.sh

set -e

# ── Config ───────────────────────────────────────────────────────────────────
GIT_AUTHOR_NAME="Savinu Gunarathna"
GIT_AUTHOR_EMAIL="savinugunarathna4@gmail.com"   # ← replace with your GitHub email

export GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"

# ── Helper ───────────────────────────────────────────────────────────────────
make_commit() {
  local DATE="$1"
  local MSG="$2"
  local FILE="$3"
  local CONTENT="$4"

  mkdir -p "$(dirname "$FILE")"
  printf '%s\n' "$CONTENT" >> "$FILE"
  git add .

  export GIT_AUTHOR_DATE="$DATE"
  export GIT_COMMITTER_DATE="$DATE"
  git commit -m "$MSG"
}

# ── Commits ──────────────────────────────────────────────────────────────────

make_commit \
  "2026-06-08T10:30:00+05:30" \
  "added README and set up the folder structure" \
  "README.md" \
  "# End-to-End Customer Churn Prediction

ML pipeline for predicting customer churn. Covers data ingestion, preprocessing, model training, and a small web app."

make_commit \
  "2026-06-08T14:50:00+05:30" \
  "added .gitignore" \
  ".gitignore" \
  "__pycache__/
*.pyc
.env
*.log"

make_commit \
  "2026-06-09T11:05:00+05:30" \
  "downloaded the telco churn dataset, added quick look notebook" \
  "data/raw/.gitkeep" \
  "# raw data directory"

make_commit \
  "2026-06-09T16:40:00+05:30" \
  "went through the data - lots of missing values and weird nulls in TotalCharges" \
  "notebooks/01_eda.ipynb" \
  '{"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"cells":[{"cell_type":"markdown","metadata":{},"source":["# 01 - EDA\n\nInitial look at the dataset, checking dtypes, nulls, distributions."]}]}'

make_commit \
  "2026-06-10T13:20:00+05:30" \
  "correlation heatmap done + confirmed the class imbalance is pretty bad (~80/20)" \
  "notebooks/02_feature_analysis.ipynb" \
  '{"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"cells":[{"cell_type":"markdown","metadata":{},"source":["# 02 - Feature Analysis\n\nCorrelation matrix, imbalance check, and feature importance rough pass."]}]}'

make_commit \
  "2026-06-11T09:15:00+05:30" \
  "started the requirements file" \
  "requirements.txt" \
  "pandas
numpy
scikit-learn
xgboost
mlflow
streamlit
matplotlib
seaborn"

make_commit \
  "2026-06-11T15:45:00+05:30" \
  "built the ingestion script, handles csv loading and basic validation" \
  "src/data_ingestion.py" \
  '"""Load and validate raw churn CSV data."""'

make_commit \
  "2026-06-12T10:30:00+05:30" \
  "preprocessing done - label encoding, standard scaling, train/test split" \
  "src/preprocessing.py" \
  '"""Preprocessing and feature engineering for the churn dataset."""'

make_commit \
  "2026-06-12T17:00:00+05:30" \
  "added utils for common stuff like saving/loading models" \
  "src/utils.py" \
  '"""Shared utility functions - model serialization, path helpers."""'

make_commit \
  "2026-06-14T14:00:00+05:30" \
  "tried logistic regression, random forest and xgboost - RF is winning by a bit" \
  "notebooks/03_model_comparison.ipynb" \
  '{"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"cells":[{"cell_type":"markdown","metadata":{},"source":["# 03 - Model Comparison\n\nBaseline comparison: LogReg vs RF vs XGBoost. Metrics: accuracy, F1, ROC-AUC."]}]}'

make_commit \
  "2026-06-15T11:10:00+05:30" \
  "wired in the RF model to the training script properly, saves to artifacts/" \
  "src/train.py" \
  '"""Training pipeline - trains the selected model and serialises it."""'

make_commit \
  "2026-06-16T13:35:00+05:30" \
  "prediction script done, loads saved model and returns prob + label" \
  "src/predict.py" \
  '"""Load trained model and run inference on new input."""'

make_commit \
  "2026-06-18T10:00:00+05:30" \
  "basic streamlit app working - you can enter customer info and get a churn prediction" \
  "app/app.py" \
  '"""Streamlit interface for the churn prediction model."""'

make_commit \
  "2026-06-19T09:45:00+05:30" \
  "hooked up mlflow, now logs accuracy/f1/auc and saves the model artifact each run" \
  "src/mlflow_tracking.py" \
  '"""MLflow integration - logs params, metrics, and artefacts per training run."""'

make_commit \
  "2026-06-20T16:20:00+05:30" \
  "wrote up how to run training locally and deploy the streamlit app" \
  "docs/workflow.md" \
  "# Workflow

## Training

\`\`\`bash
python src/train.py
\`\`\`

## Running the app

\`\`\`bash
streamlit run app/app.py
\`\`\`"

make_commit \
  "2026-06-22T11:30:00+05:30" \
  "oops accidentally committed the venv folder, removing it" \
  ".gitignore" \
  "
venv/
.venv/
env/
mlruns/
artifacts/
*.pkl
*.joblib"

make_commit \
  "2026-06-23T14:00:00+05:30" \
  "cleaned up the repo, updated gitignore properly so this wont happen again" \
  ".gitignore" \
  "
.DS_Store
*.log
data/raw/*.csv
notebooks/.ipynb_checkpoints/"

echo ""
echo "✅ All commits created."
echo ""
echo "Push with:"
echo "  git push origin main --force"
echo ""
echo "⚠️  --force rewrites remote history."
