"""FastAPI service for batch churn prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.deploy_model import META_PATH, MODEL_PATH, load_model_bundle, predict_churn

app = FastAPI(
    title="Churn Prediction API",
    description="Batch inference service for Telco customer churn prediction.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_meta: dict[str, Any] | None = None


class BatchRequest(BaseModel):
    """A batch of raw Telco records or already-processed feature rows."""

    customers: list[dict[str, Any]] = Field(
        ...,
        examples=[
            [
                {
                    "customerID": "7590-VHVEG",
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 1,
                    "PhoneService": "No",
                    "MultipleLines": "No phone service",
                    "InternetService": "DSL",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 29.85,
                    "TotalCharges": 29.85,
                }
            ]
        ],
    )
    customer_ids: list[str] | None = None


class PredictionResult(BaseModel):
    customer_id: str | None = None
    churn_probability: float
    churn_pred: int
    risk_tier: str


class BatchResponse(BaseModel):
    predictions: list[PredictionResult]
    total: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int


class HealthResponse(BaseModel):
    status: str
    artifact_available: bool
    loaded: bool
    metadata_available: bool
    artifact_type: str | None = None


def _ensure_model_loaded() -> tuple[Any, dict[str, Any]]:
    global _model, _meta

    if _model is None or _meta is None:
        _model, _meta = load_model_bundle()

    return _model, _meta


@app.on_event("startup")
def startup_event() -> None:
    """Load the model on startup when artifacts already exist."""
    try:
        _ensure_model_loaded()
    except FileNotFoundError:
        # The health endpoint will clearly report that the model is unavailable.
        pass


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    artifact_type = type(_model).__name__ if _model is not None else None
    return HealthResponse(
        status="ok",
        artifact_available=MODEL_PATH.exists(),
        loaded=_model is not None,
        metadata_available=META_PATH.exists(),
        artifact_type=artifact_type,
    )


@app.post("/predict", response_model=BatchResponse)
def predict(request: BatchRequest) -> BatchResponse:
    """Score a JSON batch of customers."""
    if not request.customers:
        raise HTTPException(status_code=400, detail="customers cannot be empty")

    try:
        model, meta = _ensure_model_loaded()
        input_df = pd.DataFrame(request.customers)
        scored = predict_churn(input_df, model=model, meta=meta)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not score batch: {exc}") from exc

    predictions: list[PredictionResult] = []
    for index, row in scored.iterrows():
        customer_id = None
        if request.customer_ids and index < len(request.customer_ids):
            customer_id = request.customer_ids[index]
        elif "customerID" in row:
            customer_id = str(row["customerID"])

        predictions.append(
            PredictionResult(
                customer_id=customer_id,
                churn_probability=round(float(row["churn_probability"]), 4),
                churn_pred=int(row["churn_pred"]),
                risk_tier=str(row["risk_tier"]),
            )
        )

    risk_counts = scored["risk_tier"].value_counts()
    return BatchResponse(
        predictions=predictions,
        total=len(predictions),
        high_risk_count=int(risk_counts.get("high", 0)),
        medium_risk_count=int(risk_counts.get("medium", 0)),
        low_risk_count=int(risk_counts.get("low", 0)),
    )


@app.post("/predict_csv")
def predict_csv(file_path: str) -> dict[str, Any]:
    """Score a local CSV file and write a sibling scored_output.csv file."""
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"CSV not found: {path}")

    try:
        model, meta = _ensure_model_loaded()
        df = pd.read_csv(path)
        scored = predict_churn(df, model=model, meta=meta)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not score CSV: {exc}") from exc

    output_path = path.parent / "scored_output.csv"
    scored.to_csv(output_path, index=False)
    risk_counts = scored["risk_tier"].value_counts()

    return {
        "file": str(output_path),
        "total": len(scored),
        "high_risk_count": int(risk_counts.get("high", 0)),
        "medium_risk_count": int(risk_counts.get("medium", 0)),
        "low_risk_count": int(risk_counts.get("low", 0)),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
