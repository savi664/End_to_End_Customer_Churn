"""Small, shared preprocessing helpers for the Telco churn project.

The goal is to keep the feature logic in one place so the dashboard, API,
CRM export, and training code all score customers the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_PATH = DATA_DIR / "telco_customer_churn.csv"
PROCESSED_DATA_PATH = DATA_DIR / "telco_churn_processed.csv"
LEGACY_PROCESSED_DATA_PATH = DATA_DIR / "telco_customer_churn_processed.csv"

BINARY_COLUMNS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]

CATEGORICAL_COLUMNS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
    "tenure_bins",
]

NUMERIC_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "PaperlessBilling",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
]

DEFAULT_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "PaperlessBilling",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
    "MultipleLines_No phone service",
    "MultipleLines_Yes",
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_No internet service",
    "OnlineSecurity_Yes",
    "OnlineBackup_No internet service",
    "OnlineBackup_Yes",
    "DeviceProtection_No internet service",
    "DeviceProtection_Yes",
    "TechSupport_No internet service",
    "TechSupport_Yes",
    "StreamingTV_No internet service",
    "StreamingTV_Yes",
    "StreamingMovies_No internet service",
    "StreamingMovies_Yes",
    "Contract_One year",
    "Contract_Two year",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
    "tenure_bins_13-24 months",
    "tenure_bins_25-48 months",
    "tenure_bins_49-72 months",
]


def load_raw_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco customer churn CSV."""
    return pd.read_csv(path)


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw fields while preserving customer-facing columns."""
    cleaned = df.copy()

    cleaned["TotalCharges"] = pd.to_numeric(cleaned.get("TotalCharges"), errors="coerce")
    cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(0)
    cleaned["MonthlyCharges"] = pd.to_numeric(
        cleaned.get("MonthlyCharges"), errors="coerce"
    ).fillna(0)
    cleaned["tenure"] = pd.to_numeric(cleaned.get("tenure"), errors="coerce").fillna(0)
    cleaned["SeniorCitizen"] = pd.to_numeric(
        cleaned.get("SeniorCitizen"), errors="coerce"
    ).fillna(0)

    return cleaned


def _yes_no_to_int(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
        .fillna(0)
        .astype(int)
    )


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple features that are useful for churn prediction."""
    featured = clean_raw_data(df)

    featured["avg_monthly_spend"] = np.where(
        featured["tenure"] > 0,
        featured["TotalCharges"] / featured["tenure"],
        featured["MonthlyCharges"],
    )

    featured["tenure_bins"] = pd.cut(
        featured["tenure"],
        bins=[-0.1, 12, 24, 48, np.inf],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
    )

    return featured


def build_feature_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw customer records into the model-ready feature table."""
    work = add_business_features(raw_df)

    features = pd.DataFrame(index=work.index)
    features["gender"] = (
        work.get("gender", "Male")
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"female": 1, "male": 0})
        .fillna(0)
        .astype(int)
    )
    features["SeniorCitizen"] = pd.to_numeric(
        work.get("SeniorCitizen", 0), errors="coerce"
    ).fillna(0).astype(int)

    for column in BINARY_COLUMNS:
        features[column] = _yes_no_to_int(work.get(column, pd.Series("No", index=work.index)))

    for column in ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend"]:
        features[column] = pd.to_numeric(work[column], errors="coerce").fillna(0)

    category_frame = pd.DataFrame(index=work.index)
    for column in CATEGORICAL_COLUMNS:
        category_frame[column] = work.get(column, pd.Series("No", index=work.index))

    dummies = pd.get_dummies(category_frame, columns=CATEGORICAL_COLUMNS, drop_first=True)
    features = pd.concat([features, dummies], axis=1)

    for column in DEFAULT_FEATURE_COLUMNS:
        if column not in features.columns:
            features[column] = 0

    return features[DEFAULT_FEATURE_COLUMNS].astype(float)


def build_processed_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build the full processed dataset, including the churn label when present."""
    processed = build_feature_table(raw_df)

    if "Churn" in raw_df.columns:
        processed["Churn"] = _yes_no_to_int(raw_df["Churn"])

    return processed


def save_processed_dataset(
    raw_path: Path | str = RAW_DATA_PATH,
    output_path: Path | str = PROCESSED_DATA_PATH,
    write_legacy_copy: bool = True,
) -> Path:
    """Create the processed CSV used by the model and notebooks."""
    raw_df = load_raw_data(raw_path)
    processed = build_processed_dataset(raw_df)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output, index=False)

    if write_legacy_copy:
        LEGACY_PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(LEGACY_PROCESSED_DATA_PATH, index=False)

    return output


def find_processed_data_path() -> Path:
    """Return the preferred processed data path available on disk."""
    if PROCESSED_DATA_PATH.exists():
        return PROCESSED_DATA_PATH
    if LEGACY_PROCESSED_DATA_PATH.exists():
        return LEGACY_PROCESSED_DATA_PATH
    raise FileNotFoundError(
        "No processed dataset found. Run `python -m src.deploy_model --train` first."
    )


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model feature columns from a processed dataset."""
    return [column for column in df.columns if column not in {"Churn", "customerID"}]


def looks_like_raw_customers(df: pd.DataFrame) -> bool:
    """Check whether a frame appears to contain raw Telco fields."""
    raw_markers = {"Contract", "InternetService", "PaymentMethod", "TotalCharges"}
    return bool(raw_markers.intersection(df.columns))


def make_feature_matrix(
    df: pd.DataFrame,
    feature_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Create a model feature matrix from raw or already-processed customer rows."""
    if looks_like_raw_customers(df):
        features = build_feature_table(df)
    else:
        features = df.copy()
        features = features.drop(columns=["customerID", "customer_id", "Churn"], errors="ignore")

    if feature_columns is None:
        feature_columns = DEFAULT_FEATURE_COLUMNS

    feature_columns = list(feature_columns)
    for column in feature_columns:
        if column not in features.columns:
            features[column] = 0

    return features[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
