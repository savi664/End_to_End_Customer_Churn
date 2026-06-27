"""Population Stability Index (PSI) drift monitoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing import PROCESSED_DATA_PATH, build_processed_dataset, load_raw_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Compute PSI between two numeric distributions."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    min_value = min(reference.min(), current.min())
    max_value = max(reference.max(), current.max())

    if min_value == max_value:
        return 0.0

    breakpoints = np.linspace(min_value, max_value, bins + 1)
    ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(current, bins=breakpoints)[0].astype(float)

    ref_pct = np.maximum(ref_counts / max(ref_counts.sum(), eps), eps)
    cur_pct = np.maximum(cur_counts / max(cur_counts.sum(), eps), eps)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_level(psi: float, moderate_threshold: float, significant_threshold: float) -> str:
    """Convert a PSI value into a readable severity label."""
    if psi >= significant_threshold:
        return "significant"
    if psi >= moderate_threshold:
        return "moderate"
    return "none"


def compute_psi_per_feature(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_columns: list[str] | None = None,
    bins: int = 10,
    moderate_threshold: float = 0.10,
    significant_threshold: float = 0.20,
) -> pd.DataFrame:
    """Compute PSI and summary stats for each shared numeric feature."""
    if numeric_columns is None:
        numeric_columns = reference_df.select_dtypes(include=[np.number]).columns.tolist()

    rows: list[dict[str, Any]] = []
    for column in numeric_columns:
        if column == "Churn" or column not in current_df.columns:
            continue

        reference = pd.to_numeric(reference_df[column], errors="coerce").dropna().to_numpy()
        current = pd.to_numeric(current_df[column], errors="coerce").dropna().to_numpy()
        psi = compute_psi(reference, current, bins=bins)

        rows.append(
            {
                "feature": column,
                "psi": round(psi, 6),
                "drift_level": drift_level(
                    psi,
                    moderate_threshold=moderate_threshold,
                    significant_threshold=significant_threshold,
                ),
                "ref_mean": round(float(np.mean(reference)), 4) if len(reference) else 0,
                "cur_mean": round(float(np.mean(current)), 4) if len(current) else 0,
                "ref_std": round(float(np.std(reference)), 4) if len(reference) else 0,
                "cur_std": round(float(np.std(current)), 4) if len(current) else 0,
            }
        )

    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def _load_monitoring_frame(path: Path) -> pd.DataFrame:
    """Load either raw Telco data or already processed feature data."""
    df = pd.read_csv(path)
    if {"Contract", "InternetService", "PaymentMethod"}.intersection(df.columns):
        return build_processed_dataset(df)
    return df


def detect_drift(
    reference_path: Path | str | None = None,
    current_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    moderate_threshold: float = 0.10,
    significant_threshold: float = 0.20,
) -> dict[str, Any]:
    """Run a PSI drift check and write CSV/JSON reports."""
    ref_path = Path(reference_path) if reference_path else PROCESSED_DATA_PATH
    cur_path = Path(current_path) if current_path else PROCESSED_DATA_PATH
    out_dir = Path(output_dir) if output_dir else REPORTS_DIR

    if not ref_path.exists():
        ref_df = build_processed_dataset(load_raw_data())
    else:
        ref_df = _load_monitoring_frame(ref_path)

    if not cur_path.exists():
        cur_df = ref_df.copy()
    else:
        cur_df = _load_monitoring_frame(cur_path)

    numeric_columns = ref_df.select_dtypes(include=[np.number]).columns.tolist()
    psi_df = compute_psi_per_feature(
        ref_df,
        cur_df,
        numeric_columns=numeric_columns,
        moderate_threshold=moderate_threshold,
        significant_threshold=significant_threshold,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    psi_path = out_dir / "drift_psi_report.csv"
    alert_path = out_dir / "drift_alerts.json"
    psi_df.to_csv(psi_path, index=False)

    alerts = psi_df[psi_df["drift_level"] != "none"].to_dict("records")
    summary = {
        "total_features": int(len(psi_df)),
        "no_drift": int((psi_df["drift_level"] == "none").sum()),
        "moderate_drift": int((psi_df["drift_level"] == "moderate").sum()),
        "significant_drift": int((psi_df["drift_level"] == "significant").sum()),
        "mean_psi": round(float(psi_df["psi"].mean()), 6) if len(psi_df) else 0,
        "max_psi": round(float(psi_df["psi"].max()), 6) if len(psi_df) else 0,
        "worst_feature": str(psi_df.iloc[0]["feature"]) if len(psi_df) else None,
        "report_path": str(psi_path),
        "alert_path": str(alert_path),
        "alerts": alerts,
    }

    with alert_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary


def main() -> None:
    results = detect_drift()
    print("Drift analysis complete")
    print(f"Total features: {results['total_features']}")
    print(f"Moderate drift: {results['moderate_drift']}")
    print(f"Significant drift: {results['significant_drift']}")
    print(f"Report: {results['report_path']}")


if __name__ == "__main__":
    main()
