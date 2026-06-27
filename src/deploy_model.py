"""Train, package, load, and score the churn model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.preprocessing import (
    DEFAULT_FEATURE_COLUMNS,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    build_processed_dataset,
    get_feature_columns,
    load_raw_data,
    make_feature_matrix,
    save_processed_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "rf_churn_model.joblib"
META_PATH = MODEL_DIR / "rf_churn_model_meta.json"

DEFAULT_RANDOM_STATE = 42


def load_model(model_path: Path | str = MODEL_PATH) -> Any:
    """Load the trained model artifact."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python -m src.deploy_model --train`."
        )
    return joblib.load(path)


def load_meta(meta_path: Path | str = META_PATH) -> dict[str, Any]:
    """Load model metadata. Return a sensible fallback if metadata is missing."""
    path = Path(meta_path)
    if not path.exists():
        return {"features": DEFAULT_FEATURE_COLUMNS, "threshold": 0.5}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_meta(meta: dict[str, Any], meta_path: Path | str = META_PATH) -> Path:
    """Write model metadata as readable JSON."""
    path = Path(meta_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    return path


def load_model_bundle() -> tuple[Any, dict[str, Any]]:
    """Load the model and metadata together."""
    return load_model(), load_meta()


def assign_risk_tier(probability: float) -> str:
    """Translate a churn probability into a CRM-friendly risk tier."""
    if probability >= 0.70:
        return "high"
    if probability >= 0.40:
        return "medium"
    return "low"


def _best_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    best_index = int(np.nanargmax(f1_scores))
    return float(thresholds[best_index])


def _metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    labels = ["retained", "churned"]
    matrix = confusion_matrix(y_true, predictions).tolist()

    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision_churn": round(
            float(precision_score(y_true, predictions, zero_division=0)), 4
        ),
        "recall_churn": round(
            float(recall_score(y_true, predictions, zero_division=0)), 4
        ),
        "f1_churn": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "confusion_matrix": {
            "labels": labels,
            "values": matrix,
        },
    }


def build_model_card(
    features: list[str],
    threshold: float,
    metrics: dict[str, Any],
    row_count: int,
) -> dict[str, Any]:
    """Create readable metadata for the packaged model."""
    return {
        "model_name": "rf_churn_model",
        "model_type": "RandomForestClassifier",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "training_rows": row_count,
        "features": features,
        "n_features": len(features),
        "threshold": round(float(threshold), 4),
        "metrics": metrics,
        "notes": (
            "Random Forest trained on the Telco customer churn data. "
            "The decision threshold is selected on the validation split by F1 score."
        ),
    }


def train_random_forest(
    raw_path: Path | str = RAW_DATA_PATH,
    processed_path: Path | str = PROCESSED_DATA_PATH,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """Train and save the Random Forest churn model."""
    raw_df = load_raw_data(raw_path)
    processed_df = build_processed_dataset(raw_df)

    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    save_processed_dataset(raw_path=raw_path, output_path=processed_path)

    feature_columns = get_feature_columns(processed_df)
    X = processed_df[feature_columns]
    y = processed_df["Churn"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    valid_probabilities = model.predict_proba(X_valid)[:, 1]
    threshold = _best_f1_threshold(y_valid, valid_probabilities)
    metrics = _metrics(y_valid, valid_probabilities, threshold)
    model_card = build_model_card(
        features=feature_columns,
        threshold=threshold,
        metrics=metrics,
        row_count=len(processed_df),
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    save_meta(model_card, META_PATH)

    return model, model_card


def predict_churn(
    customers: pd.DataFrame,
    model: Any | None = None,
    meta: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Score raw or processed customer rows with the packaged model."""
    model = model or load_model()
    meta = meta or load_meta()
    features = meta.get("features", DEFAULT_FEATURE_COLUMNS)
    threshold = float(meta.get("threshold", 0.5))

    X = make_feature_matrix(customers, features)
    probabilities = model.predict_proba(X)[:, 1]

    scored = customers.copy()
    scored["churn_probability"] = probabilities
    scored["churn_pred"] = (probabilities >= threshold).astype(int)
    scored["risk_tier"] = scored["churn_probability"].map(assign_risk_tier)

    return scored


def get_model_info() -> dict[str, Any]:
    """Return a compact summary of the deployed model."""
    meta = load_meta()
    return {
        "model_path": str(MODEL_PATH),
        "meta_path": str(META_PATH),
        "model_exists": MODEL_PATH.exists(),
        "meta_exists": META_PATH.exists(),
        "metadata": meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or inspect the churn model.")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the Random Forest model and write models/rf_churn_model.*",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Rebuild the processed CSV without training the model.",
    )
    args = parser.parse_args()

    if args.check_data:
        output = save_processed_dataset()
        print(f"Processed data written to {output}")
        return

    if args.train:
        _, meta = train_random_forest()
        print(f"Model written to {MODEL_PATH}")
        print(f"Metadata written to {META_PATH}")
        print(json.dumps(meta["metrics"], indent=2))
        return

    print(json.dumps(get_model_info(), indent=2))


if __name__ == "__main__":
    main()
