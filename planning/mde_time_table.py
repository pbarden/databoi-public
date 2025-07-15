import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import norm

def mde_time_table(
    persons_df: pd.DataFrame,
    events_df: pd.DataFrame,
    segment_filter_fn,
    conversion_event_name: str = None,
    conversion_value_field: str = None,
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
    total_weeks: int = 12,
    alpha: float = 0.05,
    power: float = 0.8,
    field_map: dict = None
) -> dict:
    """
    Calculates weekly Minimum Detectable Effects (MDEs) for binary or value-based conversions.

    Parameters:
    -----------
    persons_df : pd.DataFrame
        DataFrame with person-level data. Must include 'id', 'source_type', etc.

    events_df : pd.DataFrame
        DataFrame with event-level data. Must include:
        - 'person_id'
        - 'event_name'
        - 'timestamp'
        - 'custom_dimensions' (dict-like with conversion value field if applicable)

    segment_filter_fn : callable
        A function that filters `persons_df` and returns only matching segment members.

    conversion_event_name : str
        If provided, binary conversion counts matching event names.

    conversion_value_field : str
        If provided, value-based conversions come from this key in 'custom_dimensions'.

    start_date, end_date : pd.Timestamp
        Required date window for weekly analysis.

    total_weeks : int
        Number of weekly bins to compute.

    alpha, power : float
        Z-test thresholds.

    field_map : dict
        Optional. Maps normalized field names to raw tags. e.g., {'revenue': 'dim_1'}

    Returns:
    --------
    dict with weekly MDEs and stats.
    """

    assert start_date is not None and end_date is not None, "Start and end date are required."

    persons = segment_filter_fn(persons_df)
    if persons.empty:
        return {
            "weekly_mdes": [],
            "average_conversion_rate": 0,
            "z_alpha": None,
            "z_power": None,
            "used_weeks": 0,
            "message": "Segment returned no persons."
        }

    person_ids = persons["id"].unique()
    source_type = persons["source_type"].iloc[0] if "source_type" in persons.columns else None

    # Filter relevant events by time and person
    events = events_df[
        (events_df["person_id"].isin(person_ids)) &
        (events_df["timestamp"] >= start_date) &
        (events_df["timestamp"] < end_date)
    ].copy()

    use_value = bool(conversion_value_field)
    use_binary = bool(conversion_event_name) and not use_value

    # Resolve normalized field to raw field (if field_map provided)
    raw_tag = conversion_value_field
    if use_value and field_map:
        raw_tag = field_map.get(conversion_value_field, conversion_value_field)

    # Create weekly bins
    weekly_data = []
    base_day = pd.Timestamp(start_date).replace(hour=0, minute=0, second=0, microsecond=0)

    for i in range(total_weeks):
        w_start = base_day + timedelta(weeks=i)
        w_end = w_start + timedelta(days=7)
        if w_start > end_date:
            break

        week_events = events[(events["timestamp"] >= w_start) & (events["timestamp"] < w_end)]
        person_ids_in_week = week_events["person_id"].dropna().unique()
        person_count = len(person_ids_in_week)

        if use_binary:
            matching = week_events[week_events["event_name"] == conversion_event_name]
            conversion_count = matching["person_id"].dropna().nunique()
            weekly_data.append({
                "week_start": w_start,
                "persons": person_count,
                "conversions": conversion_count
            })

        elif use_value:
            values = []
            for _, row in week_events.iterrows():
                dims = row.get("custom_dimensions", {})
                if isinstance(dims, dict):
                    val = dims.get(raw_tag)
                    try:
                        val = float(val)
                        values.append(val)
                    except (ValueError, TypeError):
                        continue
            avg_value = sum(values) / person_count if person_count > 0 else 0
            weekly_data.append({
                "week_start": w_start,
                "persons": person_count,
                "conversions": avg_value
            })

    # MDE Calculation
    cumulative_persons = 0
    cumulative_metric = 0
    mde_values = []

    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)

    for i, week in enumerate(weekly_data):
        cumulative_persons += week["persons"]
        cumulative_metric += week["conversions"]

        if cumulative_persons == 0:
            mde_values.append(None)
            continue

        avg = cumulative_metric / cumulative_persons

        if use_binary:
            std_dev = np.sqrt(avg * (1 - avg))
        else:
            std_dev = np.std([w["conversions"] for w in weekly_data[:i+1]])

        mde = (z_alpha + z_power) * std_dev / np.sqrt(cumulative_persons)
        mde_values.append(round(mde, 6))

    return {
        "weekly_mdes": mde_values,
        "average_conversion_rate": round(cumulative_metric / cumulative_persons, 6) if cumulative_persons > 0 else 0,
        "z_alpha": round(z_alpha, 4),
        "z_power": round(z_power, 4),
        "used_weeks": len(weekly_data)
    }
