"""
------------------------------------------------------
::||\\:::://\::[[==]]:://\::::||:\::://\::||::::::::::
::||::|:://__\:::||::://__\:::||\/::((::):||::::::::::
::||:/:://::::\::||:://::::\::||:/:::\\/::||::::v0.1a:
------------------------------------------------------

Expected CSV Input Format for Experiment Data
------------------------------------------------------

This CSV is designed to support pre-test planning, in-test monitoring,
and post-test modeling for A/B/n experiments, multivariate tests,
and segment-based analyses (e.g., channel, device, etc.).

Required Columns:
-----------------
- user_id : Unique identifier for each user (string or int)
- converted   : Binary indicator (1 = converted, 0 = did not convert)
- week: Integer or string indicating the time period (e.g., 1, 2, 3, or '2024-W01')

Optional Columns:
-----------------
- group   : Test group identifier (e.g., 'control', 'variant_a', 'variant_b')
   Required for A/B/n testing but not for pre-test MDE planning.
- spend   : Numeric value representing spend or revenue per user
- visits  : Number of sessions or pageviews per user
- sessions: Number of sessions (if distinct from visits)
- age : Customer age or age bucket
- past_purchases  : Number of purchases made prior to this test

Segment Columns (Optional for Filtering/Aggregation):
-----------------------------------------------------
- channel : Marketing or acquisition channel (e.g., 'Email', 'Search', 'Social')
- device  : Device type (e.g., 'Mobile', 'Desktop')
- region  : Geographic segment (e.g., 'US', 'EU')

Notes:
------
- The CSV may contain additional columns beyond those listed; only the specified
  columns are accessed or expected by the loader.
- All time-based calculations assume 'week' unless overridden via the `time_col` parameter.
- Missing values in feature or segment columns are automatically handled (e.g., dropped for modeling).
- There should be one row per user per time period. Aggregated metrics should not be pre-computed.

This format ensures compatibility with all analysis modules including:
- MDE estimation
- Frequentist significance testing
- Bayesian analysis
- Duration estimation
- Sample mismatch detection
- Regression and clustering
- Multi-dimensional segment analysis

"""
from itertools import combinations
from math import ceil, sqrt
from scipy.stats import norm, beta, chi2_contingency, chisquare
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def calculate_lift(group_summary, control_label):
    """
    Calculates conversion rate and lift for all variants compared to a single control.

    Parameters:
    -----------
    group_summary : pd.DataFrame
        Output from `load_test_data()["group_summary"]`. Must contain:
        - 'group': group label
        - 'users': number of users
        - 'conversions': number of conversions
    control_label : str
        Label of the control group to compare all other groups against

    Returns:
    --------
    pd.DataFrame:
        Columns:
            - group
            - control_conversion_rate
            - variant_conversion_rate
            - conversion_rate_lift (%)
            - users_variant
            - conversions_variant
            - users_control
            - conversions_control
    """

    if "group" not in group_summary.columns:
        raise ValueError("'group_summary' must contain a 'group' column.")

    control_row = group_summary[group_summary["group"] == control_label]
    if control_row.empty:
        raise ValueError(f"Control group '{control_label}' not found in group_summary.")

    control_users = control_row["users"].values[0]
    control_conversions = control_row["conversions"].values[0]
    control_cr = control_conversions / control_users

    # Filter all non-control groups
    variants = group_summary[group_summary["group"] != control_label].copy()
    if variants.empty:
        raise ValueError("No variant groups found to compare against control.")

    variants["variant_conversion_rate"] = variants["conversions"] / variants["users"]
    variants["control_conversion_rate"] = control_cr
    variants["conversion_rate_lift"] = ((variants["variant_conversion_rate"] - control_cr) / control_cr) * 100
    variants["users_control"] = control_users
    variants["conversions_control"] = control_conversions

    return variants[[
        "group",
        "control_conversion_rate",
        "variant_conversion_rate",
        "conversion_rate_lift",
        "users",
        "conversions",
        "users_control",
        "conversions_control"
    ]].rename(columns={
        "users": "users_variant",
        "conversions": "conversions_variant"
    })

def mde_time_table(weekly_summary, total_weeks=8, alpha=0.05, power=0.8):
    """
    Calculates Minimum Detectable Effect (MDE) values across a specified number of weeks
    using pre-aggregated weekly summary data.

    Parameters:
    -----------
    weekly_summary : pd.DataFrame
        Output from `load_test_data()["weekly_summary"]`.
        Must include:
            - 'week' (or other time_col as defined)
            - 'users'
            - 'conversions'
        If grouped by segment or group, ensure the dataframe is filtered before use.
    total_weeks : int
        Number of weeks to calculate MDE for (e.g., 8-week test plan)
    alpha : float
        Significance level (default 0.05)
    power : float
        Statistical power (default 0.8)

    Returns:
    --------
    dict:
        - weekly_mdes: list of MDE values per week (starting from week 1)
        - average_conversion_rate: overall rate across all weeks used
        - z_alpha: z-critical value for alpha
        - z_power: z-critical value for power
        - used_weeks: number of weeks used (up to total_weeks)
    """

    if not {"users", "conversions"}.issubset(weekly_summary.columns):
        raise ValueError("weekly_summary must include 'users' and 'conversions' columns.")

    # Ensure sorted by week
    weekly = weekly_summary.sort_values(by=weekly_summary.columns[0]).copy()

    # Limit to total_weeks
    weekly = weekly.head(total_weeks)

    cumulative_users = 0
    cumulative_conversions = 0
    mde_values = []

    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)

    for i in range(len(weekly)):
        cumulative_users += weekly.iloc[i]["users"]
        cumulative_conversions += weekly.iloc[i]["conversions"]

        if cumulative_users == 0:
            mde_values.append(None)
            continue

        cr = cumulative_conversions / cumulative_users
        std_dev = (cr * (1 - cr)) ** 0.5
        mde = (z_alpha + z_power) * std_dev / (cumulative_users ** 0.5)
        mde_values.append(mde)

    return {
        "weekly_mdes": mde_values,
        "average_conversion_rate": cumulative_conversions / cumulative_users if cumulative_users > 0 else 0,
        "z_alpha": z_alpha,
        "z_power": z_power,
        "used_weeks": len(weekly)
    }

def mde_fixed(
    weekly_summary,
    target_conversion_rate,
    alpha=0.05,
    power=0.8,
    max_weeks=52
):
    """
    Estimates the minimum number of weeks required to detect a lift to a target
    conversion rate using observed weekly baseline data.

    Parameters:
    -----------
    weekly_summary : pd.DataFrame
        Aggregated data from `load_test_data()["weekly_summary"]`.
        Must contain:
            - 'users': weekly user counts
            - 'conversions': weekly conversions
    target_conversion_rate : float
        The conversion rate you want to be able to detect (0 < x < 1)
    alpha : float
        Significance level (default: 0.05)
    power : float
        Statistical power (default: 0.8)
    max_weeks : int
        Max weeks to evaluate before failing (default: 52)

    Returns:
    --------
    dict:
        - required_weeks : int
        - baseline_cr : float
        - target_cr : float
        - absolute_mde : float
        - z_alpha : float
        - z_power : float
    """

    if not {"users", "conversions"}.issubset(weekly_summary.columns):
        raise ValueError("weekly_summary must include 'users' and 'conversions'.")

    total_users = weekly_summary["users"].sum()
    total_conversions = weekly_summary["conversions"].sum()

    if total_users == 0 or total_conversions == 0:
        raise ValueError("Weekly summary must contain at least some users and conversions.")

    baseline_cr = total_conversions / total_users

    if not (0 < target_conversion_rate < 1):
        raise ValueError("Target conversion rate must be between 0 and 1.")
    if not (0 < baseline_cr < 1):
        raise ValueError("Baseline conversion rate must be between 0 and 1.")
    if target_conversion_rate == baseline_cr:
        raise ValueError("Target conversion rate must differ from baseline.")

    absolute_mde = abs(target_conversion_rate - baseline_cr)
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)

    # Estimate average weekly traffic
    weeks_available = weekly_summary.shape[0]
    avg_weekly_users = total_users / weeks_available

    for week in range(1, max_weeks + 1):
        cumulative_users = avg_weekly_users * week
        pooled_std = sqrt(
            baseline_cr * (1 - baseline_cr) +
            target_conversion_rate * (1 - target_conversion_rate)
        )
        detectable_mde = (z_alpha + z_power) * pooled_std / sqrt(cumulative_users)

        if detectable_mde <= absolute_mde:
            return {
                "required_weeks": week,
                "baseline_cr": round(baseline_cr, 6),
                "target_cr": round(target_conversion_rate, 6),
                "absolute_mde": round(absolute_mde, 6),
                "z_alpha": round(z_alpha, 4),
                "z_power": round(z_power, 4)
            }

    raise ValueError(f"Target CR is too close to baseline. Required weeks exceed max_weeks ({max_weeks}).")

def z_test(group_summary, control_label, variant_label, alpha=0.05):
    """
    Calculates statistical significance (Z-test) and estimated power from structured group summary.

    Parameters:
    -----------
    group_summary : pd.DataFrame
        Output from `load_test_data()["group_summary"]`.
        Must contain:
            - 'group': group names
            - 'users': total users per group
            - 'conversions': total conversions per group
    control_label : str
        The group name representing the control
    variant_label : str
        The group name representing the variant
    alpha : float
        Significance level (default = 0.05)

    Returns:
    --------
    dict:
        - alpha: Significance level
        - control_cr: Control conversion rate
        - variant_cr: Variant conversion rate
        - lift: Absolute difference in rates
        - z_score: Z statistic
        - p_value: Two-tailed p-value
        - is_significant: Boolean flag
        - power: Estimated power
    """

    row_control = group_summary[group_summary["group"] == control_label]
    row_variant = group_summary[group_summary["group"] == variant_label]

    if row_control.empty or row_variant.empty:
        raise ValueError("Both control and variant labels must exist in group_summary.")

    control_users = row_control["users"].values[0]
    control_conversions = row_control["conversions"].values[0]
    variant_users = row_variant["users"].values[0]
    variant_conversions = row_variant["conversions"].values[0]

    control_rate = control_conversions / control_users
    variant_rate = variant_conversions / variant_users
    pooled_rate = (control_conversions + variant_conversions) / (control_users + variant_users)

    standard_error = (pooled_rate * (1 - pooled_rate) * (1 / control_users + 1 / variant_users)) ** 0.5
    z_score = (variant_rate - control_rate) / standard_error

    p_value = 2 * norm.cdf(-abs(z_score))
    p_value = min(p_value, 1)
    is_significant = p_value < alpha

    z_alpha = norm.ppf(1 - alpha / 2)
    power = 1 - norm.cdf(z_alpha - abs(z_score))

    return {
        "alpha": alpha,
        "control_cr": round(control_rate, 6),
        "variant_cr": round(variant_rate, 6),
        "lift": round(variant_rate - control_rate, 6),
        "z_score": round(z_score, 4),
        "p_value": round(p_value, 6),
        "is_significant": is_significant,
        "power": round(power, 4)
    }

def confidence_intervals(group_summary, control_label, variant_label, confidence_level=0.95):
    """
    Calculates confidence intervals for control and variant groups using conversion summary data.

    Parameters:
    -----------
    group_summary : pd.DataFrame
        Output from `load_test_data()["group_summary"]`.
        Must contain:
            - 'group': group name
            - 'users': number of users
            - 'conversions': number of conversions
    control_label : str
        Label for control group
    variant_label : str
        Label for variant group
    confidence_level : float
        Confidence level (default 0.95)

    Returns:
    --------
    dict:
        - ci_control: tuple (lower, upper) for control CR
        - ci_variant: tuple (lower, upper) for variant CR
        - control_cr: point estimate for control CR
        - variant_cr: point estimate for variant CR
        - z_critical: z-score for confidence level
    """

    row_control = group_summary[group_summary["group"] == control_label]
    row_variant = group_summary[group_summary["group"] == variant_label]

    if row_control.empty or row_variant.empty:
        raise ValueError("One or both group labels not found in group_summary.")

    control_users = row_control["users"].values[0]
    control_conversions = row_control["conversions"].values[0]
    variant_users = row_variant["users"].values[0]
    variant_conversions = row_variant["conversions"].values[0]

    control_cr = control_conversions / control_users
    variant_cr = variant_conversions / variant_users

    se_control = (control_cr * (1 - control_cr) / control_users) ** 0.5
    se_variant = (variant_cr * (1 - variant_cr) / variant_users) ** 0.5

    z = norm.ppf(1 - (1 - confidence_level) / 2)

    ci_control = (
        round(control_cr - z * se_control, 6),
        round(control_cr + z * se_control, 6)
    )
    ci_variant = (
        round(variant_cr - z * se_variant, 6),
        round(variant_cr + z * se_variant, 6)
    )

    return {
        "ci_control": ci_control,
        "ci_variant": ci_variant,
        "control_cr": round(control_cr, 6),
        "variant_cr": round(variant_cr, 6),
        "z_critical": round(z, 4)
    }

def days_to_significance(
    group_summary,
    control_label,
    variant_label,
    test_duration_days,
    alpha=0.05,
    max_days=91
):
    """
    Estimates the number of additional days needed for an A/B test to reach statistical significance.

    Parameters:
    -----------
    group_summary : pd.DataFrame
        Output from `load_test_data()["group_summary"]`.
        Must include:
            - 'group': group label
            - 'users': number of users in group
            - 'conversions': number of conversions in group
    control_label : str
        Group name for control
    variant_label : str
        Group name for variant
    test_duration_days : int or float
        Duration of the test so far (in days)
    alpha : float
        Significance level (default: 0.05)
    max_days : int
        Max number of additional days allowed before aborting (default: 365)

    Returns:
    --------
    dict:
        - additional_days_needed : int or float('inf')
        - current_p_value : float
        - control_cr : float
        - variant_cr : float
        - z_score : float
        - is_already_significant : bool
    """

    row_control = group_summary[group_summary["group"] == control_label]
    row_variant = group_summary[group_summary["group"] == variant_label]

    if row_control.empty or row_variant.empty:
        raise ValueError("Both control and variant group labels must exist in group_summary.")

    control_users = row_control["users"].values[0]
    control_conversions = row_control["conversions"].values[0]
    variant_users = row_variant["users"].values[0]
    variant_conversions = row_variant["conversions"].values[0]

    control_rate = control_conversions / control_users
    variant_rate = variant_conversions / variant_users

    # Daily traffic rate
    daily_control_users = control_users / test_duration_days
    daily_variant_users = variant_users / test_duration_days

    current_cu = control_users
    current_vu = variant_users
    current_cc = control_conversions
    current_vc = variant_conversions

    effect_size = abs(variant_rate - control_rate)
    if effect_size == 0:
        return {
            "additional_days_needed": float('inf'),
            "current_p_value": 1.0,
            "control_cr": control_rate,
            "variant_cr": variant_rate,
            "z_score": 0,
            "is_already_significant": False
        }

    additional_days = 0
    z_alpha = norm.ppf(1 - alpha / 2)

    while additional_days <= max_days:
        pooled_cr = (current_cc + current_vc) / (current_cu + current_vu)
        std_error = (pooled_cr * (1 - pooled_cr)) * (1 / current_cu + 1 / current_vu)
        z_score = effect_size / (std_error ** 0.5)

        p_value = 2 * norm.cdf(-abs(z_score))
        if p_value < alpha:
            return {
                "additional_days_needed": max(1, ceil(additional_days)),
                "current_p_value": round(p_value, 6),
                "control_cr": round(control_rate, 6),
                "variant_cr": round(variant_rate, 6),
                "z_score": round(z_score, 4),
                "is_already_significant": additional_days == 0
            }

        current_cu += daily_control_users
        current_vu += daily_variant_users
        additional_days += 1

    return {
        "additional_days_needed": float('inf'),
        "current_p_value": round(p_value, 6),
        "control_cr": round(control_rate, 6),
        "variant_cr": round(variant_rate, 6),
        "z_score": round(z_score, 4),
        "is_already_significant": False
    }

def bayesian_signiificance(
    group_summary,
    control_label,
    variant_label,
    prior_alpha=1.0,
    prior_beta=1.0,
    probability_threshold=0.95,
    sample_count=100_000
):
    """
    Calculates Bayesian significance metrics for an A/B test using posterior beta distributions.

    Parameters:
    -----------
    group_summary : pd.DataFrame
        Output of `load_test_data()["group_summary"]`.
        Must include:
            - 'group': group names
            - 'users': user count per group
            - 'conversions': conversion count per group
    control_label : str
        Group name representing the control
    variant_label : str
        Group name representing the variant
    prior_alpha : float
        Alpha parameter for prior Beta distribution
    prior_beta : float
        Beta parameter for prior Beta distribution
    probability_threshold : float
        Minimum probability to declare the result significant (default 0.95)
    sample_count : int
        Number of posterior samples to draw (default 100,000)

    Returns:
    --------
    dict
        - probability_to_beat_control
        - is_significant
        - expected_loss
        - posterior_control: (alpha, beta)
        - posterior_variant: (alpha, beta)
        - ci_control: (lower, upper)
        - ci_variant: (lower, upper)
    """

    row_control = group_summary[group_summary["group"] == control_label]
    row_variant = group_summary[group_summary["group"] == variant_label]

    if row_control.empty or row_variant.empty:
        raise ValueError("Both control and variant labels must exist in group_summary.")

    control_users = row_control["users"].values[0]
    control_conversions = row_control["conversions"].values[0]
    variant_users = row_variant["users"].values[0]
    variant_conversions = row_variant["conversions"].values[0]

    # Posterior parameters
    alpha_c = control_conversions + prior_alpha
    beta_c = (control_users - control_conversions) + prior_beta
    alpha_v = variant_conversions + prior_alpha
    beta_v = (variant_users - variant_conversions) + prior_beta

    # Sampling
    control_samples = np.random.beta(alpha_c, beta_c, sample_count)
    variant_samples = np.random.beta(alpha_v, beta_v, sample_count)

    prob_to_beat = np.mean(variant_samples > control_samples)
    expected_loss = np.mean(np.maximum(control_samples - variant_samples, 0))

    ci_control = (
        round(beta.ppf(0.025, alpha_c, beta_c), 6),
        round(beta.ppf(0.975, alpha_c, beta_c), 6)
    )
    ci_variant = (
        round(beta.ppf(0.025, alpha_v, beta_v), 6),
        round(beta.ppf(0.975, alpha_v, beta_v), 6)
    )

    return {
        "probability_to_beat_control": round(prob_to_beat, 6),
        "is_significant": prob_to_beat > probability_threshold,
        "expected_loss": round(expected_loss, 6),
        "posterior_control": (round(alpha_c, 2), round(beta_c, 2)),
        "posterior_variant": (round(alpha_v, 2), round(beta_v, 2)),
        "ci_control": ci_control,
        "ci_variant": ci_variant
    }

def bayesian_priors(historical_conversion_rate, historical_sample_size, confidence_weight=1.0):
    """
    Generates prior alpha/beta values for Bayesian A/B testing based on historical performance.

    Parameters:
    -----------
    historical_conversion_rate : float
        Historical average conversion rate (0 < x < 1)
    historical_sample_size : int
        Size of the historical sample
    confidence_weight : float
        Multiplier for weighting how strongly you trust historical data (e.g. 0.5 = weak, 2.0 = strong)

    Returns:
    --------
    dict:
        - prior_alpha : float
        - prior_beta : float
        - effective_sample_size : float
    """

    if not 0 <= historical_conversion_rate <= 1:
        raise ValueError("Conversion rate must be between 0 and 1.")
    if historical_sample_size <= 0:
        raise ValueError("Sample size must be > 0.")
    if confidence_weight <= 0:
        raise ValueError("Confidence weight must be > 0.")

    ess = historical_sample_size * confidence_weight
    alpha = ess * historical_conversion_rate
    beta_param = ess * (1 - historical_conversion_rate)

    return {
        "prior_alpha": round(alpha, 4),
        "prior_beta": round(beta_param, 4),
        "effective_sample_size": round(ess, 2)
    }

def chi_squared_pairwise(group_summary, alpha=0.05):
    """
    Performs pairwise chi-square tests for A/B/n or MVT using preprocessed group_summary.

    Parameters:
    -----------
    group_summary : pd.DataFrame
        Output from `load_test_data()["group_summary"]`, must include:
            - group (str)
            - conversions (int)
            - users (int)

    alpha : float
        Significance threshold for uncorrected and Bonferroni-corrected p-values

    Returns:
    --------
    pd.DataFrame
        Pairwise chi-square test results with interpretation and corrections.
        Columns:
        - group_1, group_2
        - conversion_rate_1, conversion_rate_2
        - lift_pct
        - chi2_stat, p_value, p_bonferroni
        - stat_sig, stat_sig_bonf
        - interpretation
    """

    # Basic pre-checks
    required_cols = {"group", "conversions", "users"}
    if not required_cols.issubset(group_summary.columns):
        raise ValueError(f"group_summary must include columns: {required_cols}")

    # Add 'failures' and conversion rates
    df = group_summary.copy()
    df["failures"] = df["users"] - df["conversions"]
    df["conversion_rate"] = df["conversions"] / df["users"]

    group_lookup = df.set_index("group").to_dict(orient="index")
    variation_names = list(df["group"])

    results = []
    for g1, g2 in combinations(variation_names, 2):
        d1 = group_lookup[g1]
        d2 = group_lookup[g2]

        table = [
            [d1["conversions"], d1["failures"]],
            [d2["conversions"], d2["failures"]]
        ]

        chi2, p_val, _, _ = chi2_contingency(table)
        cr1 = d1["conversion_rate"]
        cr2 = d2["conversion_rate"]
        lift = (cr1 - cr2) * 100

        results.append({
            "group_1": g1,
            "group_2": g2,
            "group_1_conversion_rate": round(cr1 * 100, 2),
            "group_2_conversion_rate": round(cr2 * 100, 2),
            "lift_pct": round(lift, 2),
            "chi2_stat": round(chi2, 4),
            "p_value": round(p_val, 6)
        })

    results_df = pd.DataFrame(results)

    # Bonferroni correction
    k = len(results_df)
    results_df["p_bonferroni"] = (results_df["p_value"] * k).clip(upper=1.0)
    results_df["stat_sig"] = results_df["p_value"] < alpha
    results_df["stat_sig_bonf"] = results_df["p_bonferroni"] < alpha

    def interpret(row):
        if row["stat_sig_bonf"]:
            return "Significant (corrected)"
        elif row["stat_sig"]:
            return "Significant"
        return "Not significant"

    results_df["interpretation"] = results_df.apply(interpret, axis=1)

    return results_df

def sample_rate_mismatch(group_summary, alpha=0.05):
    """
    Performs a generalized Sample Ratio Mismatch (SRM) test across all groups
    using chi-square goodness-of-fit, based on user counts from group_summary.

    Parameters:
    -----------
    group_summary : pd.DataFrame
        Output from `load_test_data()["group_summary"]`, must include:
            - group (str)
            - users (int)

    alpha : float
        Significance level for SRM detection (default 0.05)

    Returns:
    --------
    dict
        - is_mismatch: True if p-value < alpha
        - chi2_stat: Chi-square test statistic
        - p_value: p-value of test
        - observed: list of actual user counts per group
        - expected: list of expected user counts assuming equal allocation
        - group_names: ordered list of group names
    """

    if "group" not in group_summary.columns or "users" not in group_summary.columns:
        raise ValueError("group_summary must include 'group' and 'users' columns.")

    group_names = group_summary["group"].tolist()
    observed = group_summary["users"].tolist()

    total_users = sum(observed)
    expected = [total_users / len(observed)] * len(observed)

    chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
    is_mismatch = p_value < alpha

    return {
        "is_mismatch": is_mismatch,
        "chi2_stat": round(chi2_stat, 4),
        "p_value": round(p_value, 6),
        "observed": observed,
        "expected": expected,
        "group_names": group_names
    }

def linear_regression(df, target_column, feature_columns, output_path=None):
    """
    Runs OLS linear regression analysis using unified in-memory DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Preloaded customer dataset (e.g. from `load_test_data()`)

    target_column : str
        The dependent variable (e.g., 'spend', 'conversion_rate')

    feature_columns : list of str
        List of independent variable column names (e.g., ['age', 'sessions', 'past_purchases'])

    output_path : str or None
        If provided, saves a summary CSV to this path

    Returns:
    --------
    dict
        - coefficients: {feature_name: beta_value}
        - p_values: {feature_name: p_value}
        - r_squared: float, model R²
        - full_model: statsmodels OLS object (for diagnostics)
    """

    missing_cols = [col for col in [target_column] + feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataframe: {missing_cols}")

    X = df[feature_columns]
    X = sm.add_constant(X)
    y = df[target_column]

    model = sm.OLS(y, X).fit()

    results = {
        "coefficients": model.params.to_dict(),
        "p_values": model.pvalues.to_dict(),
        "r_squared": round(model.rsquared, 4),
        "full_model": model
    }

    if output_path:
        summary_df = pd.DataFrame({
            "coefficients": model.params,
            "p_values": model.pvalues
        })
        summary_df["r_squared"] = model.rsquared
        summary_df.to_csv(output_path, index=True)

    return results

def kmeans(df, feature_columns, n_clusters=3, output_path=None):
    """
    Runs K-Means clustering on in-memory customer data to segment based on provided features.

    Parameters:
    -----------
    df : pd.DataFrame
        Preloaded customer data (e.g., from `load_test_data()`)

    feature_columns : list of str
        Column names to use for clustering (e.g. ['sessions', 'spend', 'age'])

    n_clusters : int
        Number of clusters to compute (default = 3)

    output_path : str or None
        If provided, saves the DataFrame with 'cluster' column to this path

    Returns:
    --------
    dict
        - cluster_centers: 2D numpy array of cluster centroids (standardized scale)
        - labels: List of assigned cluster labels per row
        - clustered_df: Original DataFrame with added 'cluster' column
    """

    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataframe: {missing_cols}")

    features = df[feature_columns].copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(scaled_features)
    labels = kmeans.labels_

    clustered_df = df.copy()
    clustered_df["cluster"] = labels

    if output_path:
        clustered_df.to_csv(output_path, index=False)

    return {
        "cluster_centers": kmeans.cluster_centers_,
        "labels": labels.tolist(),
        "clustered_df": clustered_df
    }

def cohort_retention_table(df, user_id_col, date_col, cohort_freq='M'):
    """
    Calculates cohort-based retention table from user activity data.

    Expected DataFrame Format:
    --------------------------
    - user_id_col: Unique identifier for each user (e.g. 'user_id')
    - date_col: Timestamp or date of user activity (e.g. 'event_date')

    Parameters:
    -----------
    df : pd.DataFrame
        Event-level or session-level customer activity data
    user_id_col : str
        Name of the column containing unique user IDs
    date_col : str
        Name of the column containing datetime activity information
    cohort_freq : str
        Frequency for cohort grouping (default: 'M' for month). Use 'W' for weekly, etc.

    Returns:
    --------
    pd.DataFrame
        A retention matrix indexed by cohort period and offset (e.g., Month 0, Month 1, ...)
        showing % of retained users over time since cohort.
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['cohort'] = df.groupby(user_id_col)[date_col].transform('min').dt.to_period(cohort_freq)
    df['period'] = df[date_col].dt.to_period(cohort_freq)
    cohort_data = df.groupby(['cohort', 'period'])[user_id_col].nunique().reset_index()
    cohort_pivot = cohort_data.pivot(index='cohort', columns='period', values=user_id_col)

    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_sizes, axis=0).round(4)

    retention.columns = ['Month_' + str(i) for i in range(retention.shape[1])]
    retention.index.name = 'Cohort'
    
    return retention

def estimate_markov_attribution(journey_df, path_col, conversion_col):
    """
    Estimates channel attribution using a first-order Markov chain approximation.

    Expected DataFrame Format:
    --------------------------
    - path_col: List or string of channel names in order (e.g., ['Email', 'Organic', 'Direct'] or "Email>Organic>Direct")
    - conversion_col: 1 if converted, 0 otherwise (binary)

    Parameters:
    -----------
    journey_df : pd.DataFrame
        DataFrame with one row per user/journey and their channel paths
    path_col : str
        Column name containing ordered list or '>'-delimited string of channel steps
    conversion_col : str
        Binary column indicating conversion

    Returns:
    --------
    pd.DataFrame
        Attribution score per channel based on removal effect (simple Markov)
        Columns:
        - channel
        - baseline_conversion_rate
        - removal_conversion_rate
        - contribution_lift
    """

    df = journey_df.copy()

    # Normalize path input if string-based
    if df[path_col].apply(lambda x: isinstance(x, str)).all():
        df[path_col] = df[path_col].apply(lambda x: x.strip().split('>'))

    all_channels = set()
    for path in df[path_col]:
        all_channels.update(path)

    baseline_cr = df[conversion_col].mean()
    results = []

    for channel in all_channels:
        df_modified = df.copy()
        df_modified[path_col] = df_modified[path_col].apply(lambda path: [step for step in path if step != channel])
        df_modified = df_modified[df_modified[path_col].apply(len) > 0]
        modified_cr = df_modified[conversion_col].mean()
        lift = baseline_cr - modified_cr
        results.append({
            "channel": channel,
            "baseline_conversion_rate": round(baseline_cr, 4),
            "removal_conversion_rate": round(modified_cr, 4),
            "contribution_lift": round(lift, 4)
        })

    return pd.DataFrame(results).sort_values("contribution_lift", ascending=False)

def time_to_conversion(df, user_id_col, first_touch_col, conversion_time_col, conversion_flag_col):
    """
    Calculates cohort-based time-to-conversion statistics.

    Expected DataFrame Format:
    --------------------------
    - user_id_col: Unique user/session ID
    - first_touch_col: Datetime of first interaction
    - conversion_time_col: Datetime of conversion (NaT if not converted)
    - conversion_flag_col: 1 if converted, 0 otherwise

    Parameters:
    -----------
    df : pd.DataFrame
        Input data with user journeys and conversion timestamps
    user_id_col : str
        Name of column with user/session identifier
    first_touch_col : str
        Column with datetime of first exposure
    conversion_time_col : str
        Column with datetime of conversion event
    conversion_flag_col : str
        Binary column: 1 = converted, 0 = not converted

    Returns:
    --------
    pd.DataFrame
        Time-to-conversion summary per cohort:
        - cohort_week: Week of first touch
        - avg_days_to_convert
        - median_days_to_convert
        - conversion_rate
        - user_count
    """

    df = df.copy()
    df[first_touch_col] = pd.to_datetime(df[first_touch_col])
    df[conversion_time_col] = pd.to_datetime(df[conversion_time_col])

    # Calculate days to convert
    df["days_to_convert"] = (df[conversion_time_col] - df[first_touch_col]).dt.days
    df["cohort_week"] = df[first_touch_col].dt.to_period("W").astype(str)

    # Aggregate
    cohort_summary = df.groupby("cohort_week").agg(
        avg_days_to_convert=pd.NamedAgg(column="days_to_convert", aggfunc="mean"),
        median_days_to_convert=pd.NamedAgg(column="days_to_convert", aggfunc="median"),
        conversion_rate=pd.NamedAgg(column=conversion_flag_col, aggfunc="mean"),
        user_count=pd.NamedAgg(column=user_id_col, aggfunc="count")
    ).reset_index()

    return cohort_summary

def time_to_conversion_distribution(df, time_col, converted_col, bins=None):
    """
    Analyzes distribution of time-to-conversion for converted users.

    Expected DataFrame Format:
    --------------------------
    - time_col: Numeric column indicating time delay from exposure to conversion (e.g., in hours or days)
    - converted_col: Boolean or binary (0/1) column indicating whether user converted

    Parameters:
    -----------
    df : pd.DataFrame
        Input user-level data
    time_col : str
        Name of the column representing time until conversion
    converted_col : str
        Name of the column indicating conversion status (1 for converted, 0 for not)
    bins : list of int or None
        Optional list of bins for histogram grouping (e.g., [0,1,2,3,5,7,14,30]). If None, uses default pandas binning

    Returns:
    --------
    pd.DataFrame
        Contains:
        - time_bin: Time window (range) until conversion
        - count: Number of converted users in each bin
        - percent: Percentage of total conversions per bin
        - cumulative_percent: Cumulative percent of conversions up to that bin
    """

    # Filter to only converted users
    df_converted = df[df[converted_col] == 1]

    # Bin the time values
    if bins:
        df_converted["time_bin"] = pd.cut(df_converted[time_col], bins=bins, right=False)
    else:
        df_converted["time_bin"] = pd.qcut(df_converted[time_col], q=10, duplicates='drop')

    summary = df_converted.groupby("time_bin").size().reset_index(name="count")
    summary["percent"] = round(summary["count"] / summary["count"].sum() * 100, 2)
    summary["cumulative_percent"] = round(summary["percent"].cumsum(), 2)

    return summary

def identify_high_value_and_similar_customers(df, spend_col, frequency_col, n_neighbors=5, top_pct=0.1):
    """
    Identifies high-value customers based on spend and frequency, and finds similar customers
    based on nearest-neighbor proximity in feature space.

    Parameters:
    -----------
    df : pd.DataFrame
        Customer-level dataset containing at least spend and frequency columns
    spend_col : str
        Column name for monetary value (e.g. lifetime spend)
    frequency_col : str
        Column name for visit or transaction frequency
    n_neighbors : int, optional
        Number of similar customers to return for each high-value customer (default: 5)
    top_pct : float, optional
        Top percentile of customers to classify as high-value (default: 0.1 for top 10%)

    Returns:
    --------
    dict
        - high_value_customers: DataFrame of top X% high-value customers
        - similar_customers: DataFrame of customers similar to high-value customers based on spend/frequency
        - index_map: Dictionary mapping high-value customer index to similar customer indices
    """

    # Prepare data
    filtered_df = df[[spend_col, frequency_col]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(filtered_df)

    # Identify high-value customers by spend + frequency score
    df['score'] = df[spend_col] + df[frequency_col]
    threshold = df['score'].quantile(1 - top_pct)
    high_value = df[df['score'] >= threshold]
    high_value_indices = high_value.index.tolist()

    # Nearest neighbor search (excluding self-matches)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(scaled)
    distances, indices = nn.kneighbors(scaled)

    similar_rows = set()
    index_map = {}

    for idx in high_value_indices:
        # Exclude self-match from neighbors
        neighbors = [i for i in indices[idx] if i != idx][:n_neighbors]
        similar_rows.update(neighbors)
        index_map[idx] = neighbors

    similar_customers = df.loc[sorted(similar_rows)]

    return {
        "high_value_customers": high_value.drop(columns=["score"]),
        "similar_customers": similar_customers.drop(columns=["score"]),
        "index_map": index_map
    }

def customer_churn_retention(df, user_col, status_col, date_col, period='M'):
    """
    Calculates customer churn and retention rates across time periods (weekly, monthly, etc.)

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with user activity logs or subscription snapshots.
    user_col : str
        Column representing unique customer/user IDs.
    status_col : str
        Binary or categorical status indicator (e.g., 'active', 'churned', 'cancelled').
    date_col : str
        Column representing timestamp of the event/status record.
    period : str, optional
        Pandas time frequency for grouping (default: 'M' = monthly).
        Examples: 'W' = weekly, 'M' = monthly, 'Q' = quarterly

    Returns:
    --------
    pd.DataFrame
        Time-indexed churn and retention statistics including:
        - period_start: Beginning of time bucket
        - total_users: Users seen in that period
        - retained_users: Users who reappear in the next period
        - churned_users: Users who disappear by next period
        - retention_rate
        - churn_rate
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['period'] = df[date_col].dt.to_period(period).dt.to_timestamp()

    # Get active users per period
    active = df[df[status_col] == 'active'].groupby(['period'])[user_col].unique()

    results = []

    all_periods = sorted(active.index)

    for i in range(len(all_periods) - 1):
        current_period = all_periods[i]
        next_period = all_periods[i + 1]

        users_current = set(active.loc[current_period])
        users_next = set(active.loc[next_period])

        retained = users_current & users_next
        churned = users_current - users_next

        total = len(users_current)
        retained_count = len(retained)
        churned_count = len(churned)

        results.append({
            "period_start": current_period,
            "total_users": total,
            "retained_users": retained_count,
            "churned_users": churned_count,
            "retention_rate": round(retained_count / total, 4) if total > 0 else 0,
            "churn_rate": round(churned_count / total, 4) if total > 0 else 0,
        })

    return pd.DataFrame(results)

def predict_optimal_channels(df,
                             id_col,
                             feature_cols,
                             target_col,
                             test_size=0.2,
                             random_state=42):
    """
    Predicts optimal customer acquisition or communication channel using historical behavior.

    Expected DataFrame Format:
    --------------------------
    Must include:
    - A user ID column (id_col)
    - Feature columns (numerical or encoded categorical inputs)
    - A target column representing the historically successful channel per user

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing customer-level data.
    id_col : str
        Column uniquely identifying each customer/user.
    feature_cols : list of str
        Feature columns to use for training (e.g. recency, spend, exposures).
    target_col : str
        Column indicating the historically successful channel per user (e.g. 'best_channel').
    test_size : float
        Proportion of data to use for evaluation (default 0.2).
    random_state : int
        Random seed for reproducibility.

    Returns:
    --------
    dict
        - predicted_df: DataFrame with user IDs and predicted channels
        - feature_importance: Dict of feature importances from the model
        - model_report: Classification report (str)
    """

    if not all(col in df.columns for col in feature_cols + [id_col, target_col]):
        raise ValueError("Missing required columns in DataFrame.")

    df_clean = df.dropna(subset=feature_cols + [target_col])
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    ids = df_clean[id_col]

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=test_size, random_state=random_state, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    predicted_df = pd.DataFrame({
        id_col: id_test,
        "predicted_best_channel": y_pred
    })

    feature_importance = dict(zip(feature_cols, model.feature_importances_))
    report = classification_report(y_test, y_pred)

    return {
        "predicted_df": predicted_df,
        "feature_importance": feature_importance,
        "model_report": report
    }
