"""Streamlit dashboard for customer churn retention work."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.crm_integration import build_crm_export, export_crm_csv, score_customers
from src.deploy_model import MODEL_PATH
from src.preprocessing import RAW_DATA_PATH, clean_raw_data, load_raw_data

try:
    import plotly.express as px
except ImportError:
    px = None

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"

st.set_page_config(
    page_title="Churn Retention Dashboard",
    page_icon=":chart_with_downwards_trend:",
    layout="wide",
)


@st.cache_data
def cached_raw_data() -> pd.DataFrame:
    """Load and lightly clean the raw customer data."""
    return clean_raw_data(load_raw_data(RAW_DATA_PATH))


@st.cache_data(show_spinner="Scoring customers...")
def cached_scored_data() -> pd.DataFrame | None:
    """Score all customers when the model artifact is available."""
    if not MODEL_PATH.exists():
        return None
    return score_customers(cached_raw_data())


def churn_rate(df: pd.DataFrame) -> float:
    return float((df["Churn"] == "Yes").mean()) if "Churn" in df.columns else 0.0


def plotly_available() -> bool:
    """Show a clear message when dashboard chart dependencies are missing."""
    if px is not None:
        return True

    st.warning("Plotly is not installed, so interactive charts are unavailable.")
    st.code("pip install -r requirements.txt", language="bash")
    return False


def sidebar(df: pd.DataFrame) -> None:
    st.sidebar.header("Dataset")
    st.sidebar.metric("Customers", f"{len(df):,}")
    if "Churn" in df.columns:
        st.sidebar.metric("Churn Rate", f"{churn_rate(df):.1%}")
        st.sidebar.metric("Churned", f"{int((df['Churn'] == 'Yes').sum()):,}")
        st.sidebar.metric("Retained", f"{int((df['Churn'] == 'No').sum()):,}")


def overview_tab(df: pd.DataFrame) -> None:
    st.header("Customer Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Churn Rate", f"{churn_rate(df):.1%}")
    col3.metric("Avg Monthly Charge", f"${df['MonthlyCharges'].mean():,.2f}")
    col4.metric("Avg Tenure", f"{df['tenure'].mean():.1f} months")

    if not plotly_available():
        st.dataframe(df.head(50), use_container_width=True)
        return

    left, right = st.columns(2)
    with left:
        churn_fig = px.pie(
            df,
            names="Churn",
            hole=0.42,
            title="Churn Distribution",
            color="Churn",
            color_discrete_map={"Yes": "#c2410c", "No": "#15803d"},
        )
        st.plotly_chart(churn_fig, use_container_width=True)

    with right:
        contract_counts = df.groupby(["Contract", "Churn"]).size().reset_index(name="count")
        contract_fig = px.bar(
            contract_counts,
            x="Contract",
            y="count",
            color="Churn",
            barmode="group",
            title="Churn by Contract",
            color_discrete_map={"Yes": "#c2410c", "No": "#15803d"},
        )
        st.plotly_chart(contract_fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        tenure_fig = px.histogram(
            df,
            x="tenure",
            color="Churn",
            nbins=30,
            barmode="overlay",
            title="Tenure Distribution",
            color_discrete_map={"Yes": "#c2410c", "No": "#15803d"},
        )
        st.plotly_chart(tenure_fig, use_container_width=True)

    with right:
        charge_fig = px.histogram(
            df,
            x="MonthlyCharges",
            color="Churn",
            nbins=30,
            barmode="overlay",
            title="Monthly Charge Distribution",
            color_discrete_map={"Yes": "#c2410c", "No": "#15803d"},
        )
        st.plotly_chart(charge_fig, use_container_width=True)


def risk_scoring_tab(scored: pd.DataFrame | None) -> None:
    st.header("Customer Risk Scoring")

    if scored is None:
        st.warning("Model artifacts are missing.")
        st.code("python -m src.deploy_model --train", language="bash")
        return

    risk_counts = scored["risk_tier"].value_counts()
    high, medium, low = st.columns(3)
    high.metric("High Risk", f"{int(risk_counts.get('high', 0)):,}")
    medium.metric("Medium Risk", f"{int(risk_counts.get('medium', 0)):,}")
    low.metric("Low Risk", f"{int(risk_counts.get('low', 0)):,}")

    if not plotly_available():
        st.dataframe(scored.head(50), use_container_width=True)
        return

    left, right = st.columns(2)
    risk_colors = {"high": "#b91c1c", "medium": "#ca8a04", "low": "#15803d"}
    with left:
        risk_fig = px.pie(
            scored,
            names="risk_tier",
            title="Risk Tier Mix",
            color="risk_tier",
            color_discrete_map=risk_colors,
        )
        st.plotly_chart(risk_fig, use_container_width=True)

    with right:
        score_fig = px.histogram(
            scored,
            x="churn_probability",
            color="risk_tier",
            nbins=40,
            title="Predicted Churn Probability",
            color_discrete_map=risk_colors,
        )
        st.plotly_chart(score_fig, use_container_width=True)

    st.subheader("Highest Priority Customers")
    display_columns = [
        "customerID",
        "tenure",
        "Contract",
        "InternetService",
        "PaymentMethod",
        "MonthlyCharges",
        "churn_probability",
        "risk_tier",
    ]
    top_customers = scored.sort_values("churn_probability", ascending=False).head(25)
    st.dataframe(top_customers[display_columns], use_container_width=True)

    if st.button("Export CRM CSV"):
        crm_df = build_crm_export(scored)
        output_path = export_crm_csv(crm_df)
        st.success(f"Exported {len(crm_df):,} rows to {output_path}")
        st.dataframe(crm_df.head(25), use_container_width=True)


def predict_customer_tab() -> None:
    st.header("Predict Customer")

    if not MODEL_PATH.exists():
        st.warning("Model artifacts are missing.")
        st.code("python -m src.deploy_model --train", language="bash")
        return

    with st.form("single_customer_prediction_form"):
        left, middle, right = st.columns(3)

        with left:
            customer_id = st.text_input("Customer ID", value="new-customer")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.checkbox("Senior Citizen")
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure Months", min_value=0, max_value=120, value=12)

        with middle:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"],
            )
            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
            )
            online_security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"],
            )
            online_backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"],
            )
            device_protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"],
            )

        with right:
            tech_support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"],
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"],
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"],
            )
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
            )
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        charge_left, charge_right = st.columns(2)
        monthly_charges = charge_left.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=500.0,
            value=70.0,
            step=1.0,
        )
        total_charges = charge_right.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=50000.0,
            value=float(monthly_charges * max(tenure, 1)),
            step=10.0,
        )

        submitted = st.form_submit_button("Predict Churn")

    if not submitted:
        return

    customer = pd.DataFrame(
        [
            {
                "customerID": customer_id,
                "gender": gender,
                "SeniorCitizen": int(senior_citizen),
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        ]
    )

    prediction = score_customers(customer).iloc[0]
    churn_probability = float(prediction["churn_probability"])
    churn_pred = int(prediction["churn_pred"])
    risk_tier = str(prediction["risk_tier"]).title()

    result_left, result_middle, result_right = st.columns(3)
    result_left.metric("Churn Probability", f"{churn_probability:.1%}")
    result_middle.metric("Prediction", "Will Churn" if churn_pred else "Will Stay")
    result_right.metric("Risk Tier", risk_tier)

    st.dataframe(
        prediction[
            [
                "customerID",
                "churn_probability",
                "churn_pred",
                "risk_tier",
                "Contract",
                "InternetService",
                "MonthlyCharges",
                "TotalCharges",
            ]
        ]
        .to_frame()
        .T,
        use_container_width=True,
    )


def drift_tab() -> None:
    st.header("Data Drift Monitoring")

    report_path = REPORTS_DIR / "drift_psi_report.csv"
    if not report_path.exists():
        st.info("No drift report found yet.")
        st.code("python -m src.drift_monitor", language="bash")
        return

    drift = pd.read_csv(report_path)
    counts = drift["drift_level"].value_counts()
    none, moderate, significant = st.columns(3)
    none.metric("No Drift", f"{int(counts.get('none', 0)):,}")
    moderate.metric("Moderate Drift", f"{int(counts.get('moderate', 0)):,}")
    significant.metric("Significant Drift", f"{int(counts.get('significant', 0)):,}")

    if not plotly_available():
        st.dataframe(drift, use_container_width=True)
        return

    drift_fig = px.bar(
        drift.sort_values("psi"),
        x="psi",
        y="feature",
        color="drift_level",
        orientation="h",
        title="Population Stability Index by Feature",
        color_discrete_map={
            "none": "#15803d",
            "moderate": "#ca8a04",
            "significant": "#b91c1c",
        },
    )
    drift_fig.add_vline(x=0.10, line_dash="dash", line_color="#ca8a04")
    drift_fig.add_vline(x=0.20, line_dash="dash", line_color="#b91c1c")
    st.plotly_chart(drift_fig, use_container_width=True)
    st.dataframe(drift, use_container_width=True)


def ab_testing_tab() -> None:
    st.header("Retention A/B Testing")

    from src.ab_testing import (
        calculate_sample_size,
        estimate_revenue_impact,
        power_curve,
        simulate_ab_test,
    )

    effect_size = st.slider("Churn reduction", 0.01, 0.15, 0.05, 0.01)
    customers = st.slider("Customers in experiment", 500, 12000, 2000, 250)
    simulations = st.slider("Simulation runs", 100, 5000, 1000, 100)

    if st.button("Run Simulation"):
        result = simulate_ab_test(
            treatment_effect=effect_size,
            n_customers=customers,
            n_simulations=simulations,
        )
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Observed Power", f"{result['observed_power']:.1%}")
        col2.metric("Significant Runs", f"{result['significant_pct']:.1f}%")
        col3.metric("Mean Effect", f"{result['mean_effect_size']:.3f}")
        col4.metric("Median P-value", f"{result['median_p_value']:.4f}")

    revenue = estimate_revenue_impact(treatment_effect=effect_size)
    sample_size = calculate_sample_size(effect_size)
    col1, col2, col3 = st.columns(3)
    col1.metric("Saved Customers", f"{revenue['estimated_saved_customers']:,}")
    col2.metric("Annual Revenue Protected", f"${revenue['annual_revenue_saved']:,.0f}")
    col3.metric("Sample Size per Group", f"{sample_size:,}")

    if not plotly_available():
        return

    curve = power_curve()
    curve_fig = px.line(
        curve,
        x="sample_size",
        y="power",
        color="effect_size",
        title="Power Curve",
    )
    curve_fig.add_hline(y=0.80, line_dash="dash", line_color="#b91c1c")
    st.plotly_chart(curve_fig, use_container_width=True)


def shap_tab() -> None:
    st.header("SHAP Explainability")

    bar_path = REPORTS_DIR / "shap_bar.png"
    beeswarm_path = REPORTS_DIR / "shap_beeswarm.png"
    importance_path = REPORTS_DIR / "shap_feature_importance.csv"

    if not importance_path.exists():
        st.info("No SHAP report found yet.")
        st.code("python -m src.shap_analysis", language="bash")
        return

    importance = pd.read_csv(importance_path)
    if not plotly_available():
        st.dataframe(importance.head(20), use_container_width=True)
        return

    top_features = importance.head(15).sort_values("mean_abs_shap")
    importance_fig = px.bar(
        top_features,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title="Top Features by Mean Absolute SHAP Value",
    )
    st.plotly_chart(importance_fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        if bar_path.exists():
            st.image(str(bar_path), use_container_width=True)
    with right:
        if beeswarm_path.exists():
            st.image(str(beeswarm_path), use_container_width=True)


def main() -> None:
    st.title("Customer Churn Retention Dashboard")
    st.caption("Score customers, export CRM actions, monitor drift, and test retention ideas.")

    raw_df = cached_raw_data()
    scored_df = cached_scored_data()
    sidebar(raw_df)

    tabs = st.tabs(
        [
            "Overview",
            "Predict Customer",
            "Risk Scoring",
            "Drift Monitoring",
            "A/B Testing",
            "SHAP Explainability",
        ]
    )

    with tabs[0]:
        overview_tab(raw_df)
    with tabs[1]:
        predict_customer_tab()
    with tabs[2]:
        risk_scoring_tab(scored_df)
    with tabs[3]:
        drift_tab()
    with tabs[4]:
        ab_testing_tab()
    with tabs[5]:
        shap_tab()


if __name__ == "__main__":
    main()
