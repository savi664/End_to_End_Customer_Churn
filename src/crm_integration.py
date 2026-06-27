"""Score customers and export a CRM-ready retention worklist."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.deploy_model import load_model_bundle, predict_churn
from src.preprocessing import RAW_DATA_PATH, load_raw_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
CRM_EXPORT_PATH = REPORTS_DIR / "crm_customer_scores.csv"


ACTION_BY_RISK = {
    "high": "Call within 24 hours with a retention offer",
    "medium": "Send a proactive check-in and targeted discount",
    "low": "Keep in standard lifecycle engagement",
}

PRIORITY_BY_RISK = {"high": 1, "medium": 2, "low": 3}


def score_customers(
    customers: pd.DataFrame,
    model: Any | None = None,
    meta: dict | None = None,
) -> pd.DataFrame:
    """Add churn probabilities, binary predictions, and risk tiers."""
    if model is None or meta is None:
        model, meta = load_model_bundle()
    return predict_churn(customers, model=model, meta=meta)


def build_crm_export(scored_customers: pd.DataFrame) -> pd.DataFrame:
    """Keep the columns a CRM user needs for follow-up."""
    preferred_columns = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "Contract",
        "InternetService",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "churn_probability",
        "churn_pred",
        "risk_tier",
    ]
    export_columns = [
        column for column in preferred_columns if column in scored_customers.columns
    ]
    crm_df = scored_customers[export_columns].copy()

    crm_df["churn_probability"] = crm_df["churn_probability"].round(4)
    crm_df["retention_priority"] = crm_df["risk_tier"].map(PRIORITY_BY_RISK)
    crm_df["suggested_action"] = crm_df["risk_tier"].map(ACTION_BY_RISK)

    return crm_df.sort_values(
        ["retention_priority", "churn_probability"],
        ascending=[True, False],
    )


def export_crm_csv(
    crm_df: pd.DataFrame,
    output_path: Path | str = CRM_EXPORT_PATH,
) -> Path:
    """Write the CRM export to disk."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    crm_df.to_csv(output, index=False)
    return output


def get_risk_summary(scored_customers: pd.DataFrame) -> dict[str, int]:
    """Count customers in each risk tier."""
    if "risk_tier" not in scored_customers.columns:
        return {}
    return {
        str(tier): int(count)
        for tier, count in scored_customers["risk_tier"].value_counts().items()
    }


def get_action_summary(crm_df: pd.DataFrame) -> dict[str, int]:
    """Count customers by recommended retention action."""
    if "suggested_action" not in crm_df.columns:
        return {}
    return {
        str(action): int(count)
        for action, count in crm_df["suggested_action"].value_counts().items()
    }


def run_full_scoring(
    raw_path: Path | str = RAW_DATA_PATH,
    output_path: Path | str = CRM_EXPORT_PATH,
) -> dict[str, Any]:
    """Score the full raw dataset and export a CRM CSV."""
    raw_df = load_raw_data(raw_path)
    scored = score_customers(raw_df)
    crm_df = build_crm_export(scored)
    csv_path = export_crm_csv(crm_df, output_path)

    return {
        "total_customers": len(scored),
        "risk_summary": get_risk_summary(scored),
        "action_summary": get_action_summary(crm_df),
        "crm_csv": str(csv_path),
    }


def main() -> None:
    results = run_full_scoring()
    print(f"Scored {results['total_customers']} customers")
    print(f"Risk breakdown: {results['risk_summary']}")
    print(f"CRM export: {results['crm_csv']}")


if __name__ == "__main__":
    main()
