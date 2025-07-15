import pandas as pd
import numpy as np
from math import ceil
from datetime import datetime
from scipy.stats import norm

def days_to_significance(
    persons_df: pd.DataFrame,
    events_df: pd.DataFrame,
    control_segment_filter_fn,
    variant_segment_filter_fn,
    field_map: dict,
    conversion_field: str,
    start_date: datetime,
    end_date: datetime,
    test_duration_days: int,
    alpha: float = 0.05,
    max_days: int = 91
) -> dict:
    """
    Estimates how many additional days are needed for a test to reach statistical significance.

    Parameters:
    -----------
    persons_df : pd.DataFrame
        Contains person-level records. Requires 'id' and 'source_type'.

    events_df : pd.DataFrame
        Contains event-level records with:
        - 'person_id'
        - 'timestamp'
        - 'custom_dimensions': dict with conversion values.

    control_segment_filter_fn : callable
        Function to filter persons_df for the control group.

    variant_segment_filter_fn : callable
        Function to filter persons_df for the variant group.

    field_map : dict
        Maps (source_type, normalized_field) to raw tag. Example:
        {('my_source', 'converted'): 'converted_bool'}

    conversion_field : str
        The normalized field name, such as 'converted'.

    start_date, end_date : datetime
        Test window.

    test_duration_days : int
        How many days of data are already collected.

    alpha : float
        Significance level.

    max_days : int
        Max simulation horizon.

    Returns:
    --------
    dict with MDE result
    """

    control_persons = control_segment_filter_fn(persons_df)
    variant_persons = variant_segment_filter_fn(persons_df)

    if control_persons.empty or variant_persons.empty:
        raise ValueError("Empty control or variant group. Segment returned no persons.")

    control_ids = control_persons["id"].unique()
    variant_ids = variant_persons["id"].unique()

    # Infer source_type from control group
    source_type = control_persons["source_type"].iloc[0]
    raw_tag = field_map.get((source_type, conversion_field))

    if not raw_tag:
        raise ValueError(f"No raw tag found for normalized field '{conversion_field}' and source_type '{source_type}'.")

    def is_converted(row):
        dims = row.get("custom_dimensions", {})
        return isinstance(dims, dict) and dims.get(raw_tag) is not None

    event_filter = (
        (events_df["timestamp"] >= start_date) &
        (events_df["timestamp"] < end_date)
    )
    relevant_events = events_df[event_filter].copy()

    control_events = relevant_events[relevant_events["person_id"].isin(control_ids)]
    variant_events = relevant_events[relevant_events["person_id"].isin(variant_ids)]

    control_converted_ids = control_events[control_events.apply(is_converted, axis=1)]["person_id"].unique()
    variant_converted_ids = variant_events[variant_events.apply(is_converted, axis=1)]["person_id"].unique()

    n_control = len(control_ids)
    n_variant = len(variant_ids)
    conv_control = len(control_converted_ids)
    conv_variant = len(variant_converted_ids)

    if n_control == 0 or n_variant == 0 or test_duration_days <= 0:
        raise ValueError("Invalid group sizes or duration")

    cr_control = conv_control / n_control
    cr_variant = conv_variant / n_variant
    effect_size = abs(cr_variant - cr_control)

    if effect_size == 0:
        return {
            "additional_days_needed": float("inf"),
            "current_p_value": 1.0,
            "control_cr": round(cr_control, 6),
            "variant_cr": round(cr_variant, 6),
            "z_score": 0.0,
            "is_already_significant": False,
        }

    # Simulate accumulation
    daily_control_persons = n_control / test_duration_days
    daily_variant_persons = n_variant / test_duration_days

    cur_cu = n_control
    cur_vu = n_variant
    cur_cc = conv_control
    cur_vc = conv_variant

    additional_days = 0

    while additional_days <= max_days:
        pooled_cr = (cur_cc + cur_vc) / (cur_cu + cur_vu)
        std_error = (pooled_cr * (1 - pooled_cr)) * (1 / cur_cu + 1 / cur_vu)
        z_score = effect_size / (std_error ** 0.5)
        p_value = 2 * norm.cdf(-abs(z_score))

        if p_value < alpha:
            return {
                "additional_days_needed": max(1, ceil(additional_days)),
                "current_p_value": round(p_value, 6),
                "control_cr": round(cr_control, 6),
                "variant_cr": round(cr_variant, 6),
                "z_score": round(z_score, 4),
                "is_already_significant": additional_days == 0
            }

        cur_cu += daily_control_persons
        cur_vu += daily_variant_persons
        additional_days += 1

    return {
        "additional_days_needed": float('inf'),
        "current_p_value": round(p_value, 6),
        "control_cr": round(cr_control, 6),
        "variant_cr": round(cr_variant, 6),
        "z_score": round(z_score, 4),
        "is_already_significant": False
    }
