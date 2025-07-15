import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from math import sqrt

def mde_fixed(
    persons_df: pd.DataFrame,
    events_df: pd.DataFrame,
    segment_filter_fn,
    field_map: dict,
    conversion_event_name: str,
    start_date: datetime,
    end_date: datetime,
    target_conversion_rate: float,
    alpha: float = 0.05,
    power: float = 0.8,
    max_weeks: int = 12
) -> dict:
    """
    Calculates how many weeks of data are needed to detect a lift to `target_conversion_rate`
    using a real segment's observed conversion behavior on a specific event.

    Parameters:
    -----------
    persons_df : pd.DataFrame
        Person-level data. Must include ['id', 'source_type'] at minimum.

    events_df : pd.DataFrame
        Event-level data. Must include:
        - 'person_id'
        - 'timestamp'
        - 'custom_dimensions' (dict-like)

    segment_filter_fn : callable
        A function that filters persons_df to extract the correct segment.

    field_map : dict
        Dictionary mapping normalized fields to raw tags per source_type.
        Example: {('my_source', 'purchase'): 'purchase_usd'}

    conversion_event_name : str
        Normalized field name to look up in custom_dimensions.

    start_date, end_date : datetime
        Date window to restrict events.

    target_conversion_rate : float
        Desired CR used to compute minimum detectable effect.

    Returns:
    --------
    dict with required weeks and stats.
    """

    # Filter persons
    segment_persons = segment_filter_fn(persons_df)
    if segment_persons.empty:
        raise ValueError("Segment returned no matching persons.")

    source_type = segment_persons["source_type"].iloc[0]
    raw_tag = field_map.get((source_type, conversion_event_name))
    if not raw_tag:
        raise ValueError(f"No field mapping found for '{conversion_event_name}' on source '{source_type}'.")

    person_ids = segment_persons["id"].unique()
    events = events_df[
        (events_df["person_id"].isin(person_ids)) &
        (events_df["timestamp"] >= start_date) &
        (events_df["timestamp"] < end_date)
    ].copy()

    # Build week bins
    full_weeks = []
    cursor = pd.Timestamp(start_date).replace(hour=0, minute=0, second=0, microsecond=0)

    for _ in range(max_weeks):
        if cursor >= end_date:
            break
        full_weeks.append((cursor, min(cursor + timedelta(days=7), end_date)))
        cursor += timedelta(days=7)

    person_counts = []
    conv_counts = []

    for week_start, week_end in full_weeks:
        week_events = events[
            (events["timestamp"] >= week_start) & (events["timestamp"] < week_end)
        ]

        person_ids_this_week = week_events["person_id"].dropna().unique()

        def has_valid_conversion(row):
            dims = row.get("custom_dimensions", {})
            if isinstance(dims, dict):
                val = dims.get(raw_tag)
                return val is not None
            return False

        converted_events = week_events[week_events.apply(has_valid_conversion, axis=1)]
        converted_ids = converted_events["person_id"].dropna().unique()

        person_counts.append(len(person_ids_this_week))
        conv_counts.append(len(converted_ids))

    total_persons = sum(person_counts)
    total_conversions = sum(conv_counts)

    if total_persons == 0 or total_conversions == 0:
        raise ValueError("Insufficient data — no persons or conversions found.")

    baseline_cr = total_conversions / total_persons

    if not (0 < target_conversion_rate < 1):
        raise ValueError("Target conversion rate must be between 0 and 1.")
    if abs(target_conversion_rate - baseline_cr) < 1e-6:
        raise ValueError("Target CR too close to baseline — no MDE possible.")

    absolute_mde = abs(target_conversion_rate - baseline_cr)
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    avg_weekly_persons = total_persons / len(person_counts)

    for week in range(1, max_weeks + 1):
        cumulative_n = avg_weekly_persons * week
        pooled_sd = sqrt(
            baseline_cr * (1 - baseline_cr) +
            target_conversion_rate * (1 - target_conversion_rate)
        )
        detectable_mde = (z_alpha + z_power) * pooled_sd / sqrt(cumulative_n)

        if detectable_mde <= absolute_mde:
            return {
                "required_weeks": week,
                "baseline_cr": round(baseline_cr, 6),
                "target_cr": round(target_conversion_rate, 6),
                "absolute_mde": round(absolute_mde, 6),
                "z_alpha": round(z_alpha, 4),
                "z_power": round(z_power, 4)
            }

    raise ValueError(
        f"Target CR too close to baseline ({baseline_cr:.4f}). "
        f"Detectable MDE: {detectable_mde:.5f}. "
        f"Required weeks exceed max_weeks ({max_weeks})."
    )
