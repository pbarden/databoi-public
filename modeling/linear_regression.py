import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Callable, List, Dict
from scipy import stats as scistats

def linear_regression(
    persons_df: pd.DataFrame,
    sessions_df: Optional[pd.DataFrame],
    target_variable: str,
    feature_variables: List[str],
    field_map: Dict[tuple, bool],
    segment_filter_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    output_path: Optional[str] = None
) -> dict:
    """
    Runs OLS regression on person attributes, optionally filtered by segment or session date range.

    Parameters:
    -----------
    persons_df : pd.DataFrame
        Must contain ['id', 'source_type', 'attributes']

    sessions_df : pd.DataFrame or None
        Must contain ['person_id', 'session_start'] if not using segment_filter_fn

    target_variable : str
        Normalized field name to predict

    feature_variables : List[str]
        List of normalized field names to use as predictors

    field_map : dict
        Dict mapping (source_type, normalized_field) → True if field exists

    segment_filter_fn : Callable or None
        Optional function to filter persons_df

    start_date, end_date : datetime
        Optional date filtering if using sessions_df

    output_path : str
        Optional path to export CSV

    Returns:
    --------
    dict with regression results
    """

    # Step 1: Person selection
    if segment_filter_fn:
        filtered_persons = segment_filter_fn(persons_df)
    else:
        if sessions_df is None or start_date is None or end_date is None:
            raise ValueError("Must supply session data and date range if no segment filter is provided.")
        sessions_filtered = sessions_df[
            (sessions_df["session_start"] >= start_date) &
            (sessions_df["session_start"] <= end_date)
        ]
        person_ids = sessions_filtered["person_id"].unique()
        filtered_persons = persons_df[persons_df["id"].isin(person_ids)]

    if filtered_persons.empty:
        raise ValueError("No persons found for given segment or date range.")

    # Step 2: Clean and validate data
    records = []
    for _, person in filtered_persons.iterrows():
        source = person.get("source_type", "")
        attrs = person.get("attributes", {})
        row = {"person_id": person["id"]}

        # Target validation
        if not field_map.get((source, target_variable), False):
            row["target"] = None
        else:
            try:
                row["target"] = float(attrs.get(target_variable))
            except (TypeError, ValueError):
                row["target"] = None

        # Features validation
        for feat in feature_variables:
            if not field_map.get((source, feat), False):
                row[feat] = None
            else:
                try:
                    row[feat] = float(attrs.get(feat))
                except (TypeError, ValueError):
                    row[feat] = None

        records.append(row)

    df = pd.DataFrame(records)
    df = df.dropna(subset=["target"] + feature_variables)

    if df.empty:
        raise ValueError("No complete rows — missing required attribute fields.")

    # Step 3: Run OLS Regression
    X = df[feature_variables].copy()
    y = df["target"]
    X.insert(0, "const", 1)

    X_matrix = X.to_numpy()
    y_vector = y.to_numpy()

    XtX_inv = np.linalg.pinv(X_matrix.T @ X_matrix)
    betas = XtX_inv @ X_matrix.T @ y_vector
    y_pred = X_matrix @ betas

    n, k = X_matrix.shape
    dof = n - k
    residuals = y_vector - y_pred
    rss = np.sum(residuals**2)
    tss = np.sum((y_vector - np.mean(y_vector))**2)

    if tss == 0:
        raise ValueError("Target variable has no variance — cannot perform regression.")

    r_squared = 1 - rss / tss
    mse = rss / dof
    std_err = np.sqrt(np.diag(mse * XtX_inv))

    t_stats = betas / std_err
    p_vals = [2 * (1 - scistats.t.cdf(abs(t), df=dof)) for t in t_stats]

    names = ["const"] + feature_variables
    results = {
        "coefficients": dict(zip(names, np.round(betas, 6))),
        "p_values": dict(zip(names, np.round(p_vals, 6))),
        "r_squared": round(r_squared, 4),
        "full_model": {
            "betas": betas,
            "std_err": std_err,
            "residuals": residuals,
            "X": X_matrix,
            "y": y_vector,
            "y_pred": y_pred
        }
    }

    if output_path:
        summary_df = pd.DataFrame({
            "coefficients": np.round(betas, 6),
            "p_values": np.round(p_vals, 6)
        }, index=names)
        summary_df["r_squared"] = round(r_squared, 4)
        summary_df.to_csv(output_path)

    return results
