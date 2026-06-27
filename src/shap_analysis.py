"""SHAP explainability for the packaged churn model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.deploy_model import load_model, load_meta
from src.preprocessing import DEFAULT_FEATURE_COLUMNS, PROCESSED_DATA_PATH, make_feature_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def _load_shap_dependencies():
    """Import SHAP and matplotlib only when explanations are requested."""
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP analysis requires `shap` and `matplotlib`. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc

    return shap, plt


def load_model_and_data(
    data_path: Path | str = PROCESSED_DATA_PATH,
) -> tuple[Any, pd.DataFrame, pd.Series | None, list[str]]:
    """Load the packaged model and the feature matrix used for explanations."""
    model = load_model()
    meta = load_meta()
    feature_columns = meta.get("features", DEFAULT_FEATURE_COLUMNS)

    data = pd.read_csv(data_path)
    X = make_feature_matrix(data, feature_columns)
    y = data["Churn"] if "Churn" in data.columns else None

    return model, X, y, feature_columns


def compute_shap_values(model: Any, X: pd.DataFrame):
    """Return a SHAP Explanation object for churn class probabilities."""
    shap, _ = _load_shap_dependencies()
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    if getattr(explanation, "values", np.array([])).ndim == 3:
        values = explanation.values[:, :, 1]
        base_values = explanation.base_values[:, 1]
        explanation = shap.Explanation(
            values=values,
            base_values=base_values,
            data=X.to_numpy(),
            feature_names=list(X.columns),
        )

    return explanation


def plot_waterfall(explanation, index: int = 0) -> Path:
    """Save a SHAP waterfall plot for one customer."""
    shap, plt = _load_shap_dependencies()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    shap.plots.waterfall(explanation[index], max_display=15, show=False)
    output = REPORTS_DIR / f"shap_waterfall_{index}.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    return output


def plot_bar(explanation) -> Path:
    """Save a global SHAP feature importance bar chart."""
    shap, plt = _load_shap_dependencies()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    shap.plots.bar(explanation, max_display=20, show=False)
    output = REPORTS_DIR / "shap_bar.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    return output


def plot_beeswarm(explanation) -> Path:
    """Save a SHAP beeswarm chart."""
    shap, plt = _load_shap_dependencies()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    shap.plots.beeswarm(explanation, max_display=20, show=False)
    output = REPORTS_DIR / "shap_beeswarm.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    return output


def save_feature_importance(explanation) -> Path:
    """Write mean absolute SHAP values to CSV."""
    values = np.asarray(explanation.values)
    importance = (
        pd.DataFrame(
            {
                "feature": list(explanation.feature_names),
                "mean_abs_shap": np.abs(values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "shap_feature_importance.csv"
    importance.to_csv(output, index=False)
    return output


def generate_all_explanations(
    data_path: Path | str = PROCESSED_DATA_PATH,
    sample_size: int = 500,
    random_state: int = 42,
) -> dict[str, Any]:
    """Create all SHAP reports used by the dashboard."""
    model, X, _, _ = load_model_and_data(data_path)
    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=random_state)

    explanation = compute_shap_values(model, X)

    waterfall_path = plot_waterfall(explanation, index=0)
    bar_path = plot_bar(explanation)
    beeswarm_path = plot_beeswarm(explanation)
    importance_path = save_feature_importance(explanation)

    top_features = (
        pd.read_csv(importance_path).head(10).set_index("feature")["mean_abs_shap"].to_dict()
    )

    return {
        "waterfall": str(waterfall_path),
        "bar": str(bar_path),
        "beeswarm": str(beeswarm_path),
        "importance": str(importance_path),
        "top_features": top_features,
    }


def main() -> None:
    results = generate_all_explanations()
    print("SHAP analysis complete")
    for key, value in results.items():
        if key != "top_features":
            print(f"{key}: {value}")
    print("Top features:", results["top_features"])


if __name__ == "__main__":
    main()
