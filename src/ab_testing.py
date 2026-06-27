"""A/B test simulation and power analysis for retention campaigns."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

BASELINE_CHURN_RATE = 0.2654
AVERAGE_MONTHLY_REVENUE = 64.76


def calculate_power(
    effect_size: float,
    n_per_group: int,
    alpha: float = 0.05,
    baseline_churn_rate: float = BASELINE_CHURN_RATE,
) -> float:
    """Estimate power for a two-sided two-proportion z-test."""
    treatment_rate = max(baseline_churn_rate - effect_size, 0.0001)
    pooled_se = np.sqrt(
        baseline_churn_rate * (1 - baseline_churn_rate) / n_per_group
        + treatment_rate * (1 - treatment_rate) / n_per_group
    )
    z_effect = abs(baseline_churn_rate - treatment_rate) / pooled_se
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    power = 1 - stats.norm.cdf(z_alpha - z_effect) + stats.norm.cdf(-z_alpha - z_effect)
    return float(np.clip(power, 0, 1))


def calculate_sample_size(
    effect_size: float,
    power: float = 0.80,
    alpha: float = 0.05,
    baseline_churn_rate: float = BASELINE_CHURN_RATE,
) -> int:
    """Estimate the sample size needed per group."""
    treatment_rate = max(baseline_churn_rate - effect_size, 0.0001)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    numerator = (
        z_alpha * np.sqrt(2 * baseline_churn_rate * (1 - baseline_churn_rate))
        + z_beta
        * np.sqrt(
            baseline_churn_rate * (1 - baseline_churn_rate)
            + treatment_rate * (1 - treatment_rate)
        )
    ) ** 2
    denominator = (baseline_churn_rate - treatment_rate) ** 2
    return int(np.ceil(numerator / denominator))


def simulate_ab_test(
    control_churn_rate: float = BASELINE_CHURN_RATE,
    treatment_effect: float = 0.05,
    n_customers: int = 2_000,
    n_simulations: int = 1_000,
    alpha: float = 0.05,
    random_state: int | None = 42,
) -> dict[str, Any]:
    """Simulate many retention A/B tests and summarize the outcomes."""
    rng = np.random.default_rng(random_state)
    n_per_group = n_customers // 2
    treatment_churn_rate = max(control_churn_rate - treatment_effect, 0.0001)

    significant_runs = 0
    observed_effects: list[float] = []
    p_values: list[float] = []

    for _ in range(n_simulations):
        control = rng.binomial(1, control_churn_rate, n_per_group)
        treatment = rng.binomial(1, treatment_churn_rate, n_per_group)

        control_rate = float(control.mean())
        treatment_rate = float(treatment.mean())
        standard_error = np.sqrt(
            control_rate * (1 - control_rate) / n_per_group
            + treatment_rate * (1 - treatment_rate) / n_per_group
        )

        if standard_error == 0:
            continue

        z_score = (control_rate - treatment_rate) / standard_error
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        observed_effects.append(control_rate - treatment_rate)
        p_values.append(float(p_value))
        significant_runs += int(p_value < alpha)

    completed_runs = len(p_values)
    observed_power = significant_runs / completed_runs if completed_runs else 0

    return {
        "control_churn_rate": control_churn_rate,
        "treatment_churn_rate": treatment_churn_rate,
        "effect_size": treatment_effect,
        "n_customers": n_customers,
        "n_per_group": n_per_group,
        "n_simulations": completed_runs,
        "alpha": alpha,
        "observed_power": round(observed_power, 4),
        "mean_effect_size": round(float(np.mean(observed_effects)), 4),
        "median_p_value": round(float(np.median(p_values)), 4),
        "significant_pct": round(observed_power * 100, 2),
    }


def estimate_revenue_impact(
    total_customers: int = 7_043,
    churn_rate: float = BASELINE_CHURN_RATE,
    treatment_effect: float = 0.05,
    avg_monthly_revenue: float = AVERAGE_MONTHLY_REVENUE,
    avg_remaining_months: float = 12,
) -> dict[str, Any]:
    """Estimate retained customers and revenue protected by an intervention."""
    at_risk_customers = int(round(total_customers * churn_rate))
    saved_customers = int(round(total_customers * treatment_effect))
    monthly_revenue_saved = saved_customers * avg_monthly_revenue
    annual_revenue_saved = monthly_revenue_saved * avg_remaining_months

    return {
        "total_customers": total_customers,
        "at_risk_customers": at_risk_customers,
        "estimated_saved_customers": saved_customers,
        "monthly_revenue_saved": round(monthly_revenue_saved, 2),
        "annual_revenue_saved": round(annual_revenue_saved, 2),
        "avg_monthly_revenue_per_customer": avg_monthly_revenue,
    }


def power_curve(
    effect_sizes: list[float] | None = None,
    sample_sizes: list[int] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Build a small table for plotting power by effect and sample size."""
    effect_sizes = effect_sizes or [0.02, 0.03, 0.05, 0.07, 0.10]
    sample_sizes = sample_sizes or [500, 1_000, 2_000, 4_000, 8_000, 12_000]

    rows = []
    for effect_size in effect_sizes:
        for total_sample_size in sample_sizes:
            rows.append(
                {
                    "effect_size": effect_size,
                    "sample_size": total_sample_size,
                    "power": round(
                        calculate_power(
                            effect_size,
                            n_per_group=total_sample_size // 2,
                            alpha=alpha,
                        ),
                        4,
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    print("Sample size needed per group for 80% power")
    for effect in [0.03, 0.05, 0.07, 0.10]:
        print(f"{effect:.0%} churn reduction: {calculate_sample_size(effect):,}")

    simulation = simulate_ab_test()
    print(f"\nSimulated power: {simulation['observed_power']:.1%}")

    revenue = estimate_revenue_impact()
    print(f"Annual revenue protected: ${revenue['annual_revenue_saved']:,.0f}")


if __name__ == "__main__":
    main()
