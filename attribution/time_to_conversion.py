import pandas as pd
from datetime import datetime
from typing import Optional

def time_to_conversion(
    sessions_df: pd.DataFrame,
    events_df: pd.DataFrame,
    field_map_df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    conversion_field: str,
    segment_filter_fn=None,
    session_filters: dict = None,
    min_days: int = 0,
    max_days: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate time-to-conversion from first touch to conversion using session and event data.

    Parameters
    ----------
    sessions_df : pd.DataFrame
        Must contain: session_id, person_id, session_start, source_type

    events_df : pd.DataFrame
        Must contain: session_id, timestamp, custom_dimensions, source_type

    field_map_df : pd.DataFrame
        Must contain: source_type, normalized_field, raw_tag

    start_date : datetime
    end_date : datetime

    conversion_field : str
        Normalized field to check in event custom_dimensions

    segment_filter_fn : Callable or None
        Optional function to filter sessions_df by person_id segment

    session_filters : dict
        Optional filters like {"some_field": "some_value"}

    min_days : int
        Minimum number of days to include for conversion delta

    max_days : int or None
        Maximum number of days to include

    Returns
    -------
    pd.DataFrame
    """
    session_filters = session_filters or {}

    # Step 1: Filter session records by date and optional fields
    df = sessions_df.copy()

    # Apply segment filter
    if segment_filter_fn:
        df = segment_filter_fn(df)
    if df.empty:
        raise ValueError("No sessions after segment filtering.")

    df = df[
        (df["session_start"] >= start_date) &
        (df["session_start"] <= end_date)
    ]

    # Apply extra session filters
    for k, v in session_filters.items():
        if k in df.columns:
            df = df[df[k] == v]

    if df.empty:
        raise ValueError("No sessions found for the given criteria.")

    # Map session_id to first touch and person_id
    session_map = df.set_index("session_id")[["person_id", "session_start", "source_type"]].to_dict("index")
    valid_session_ids = set(session_map.keys())

    # Step 2: Filter events to only those related to session_ids
    events = events_df[events_df["session_id"].isin(valid_session_ids)].copy()
    if events.empty:
        raise ValueError("No events available for selected sessions.")

    records = []
    # Create set of valid source_type/raw_tag combos from field_map_df
    raw_tag_lookup = field_map_df[
        field_map_df["normalized_field"] == conversion_field
    ].set_index("source_type")["raw_tag"].to_dict()

    for _, e in events.iterrows():
        source_type = e["source_type"]
        raw_tag = raw_tag_lookup.get(source_type)
        if not raw_tag:
            continue

        # Check if this event is a valid conversion event
        cd = e["custom_dimensions"]
        if not cd or raw_tag not in cd:
            continue

        session_id = e["session_id"]
        sdata = session_map.get(session_id)
        if not sdata:
            continue

        delta_days = (e["timestamp"] - sdata["session_start"]).days
        if delta_days < min_days or (max_days is not None and delta_days > max_days):
            continue

        records.append({
            "person_id": sdata["person_id"],
            "session_id": session_id,
            "first_touch": sdata["session_start"],
            "conversion_time": e["timestamp"],
            "converted": 1,
            "days_to_convert": delta_days
        })

    # Step 3: Add non-converted sessions
    converted_session_ids = set(r["session_id"] for r in records)
    for sid, sdata in session_map.items():
        if sid not in converted_session_ids:
            records.append({
                "person_id": sdata["person_id"],
                "session_id": sid,
                "first_touch": sdata["session_start"],
                "conversion_time": None,
                "converted": 0,
                "days_to_convert": None
            })

    full_df = pd.DataFrame(records)
    if full_df.empty:
        raise ValueError("No records found for time-to-conversion analysis.")

    full_df["cohort_week"] = full_df["first_touch"].dt.to_period("W").astype(str)

    summary = full_df.groupby("cohort_week").agg(
        avg_days_to_convert=pd.NamedAgg(column="days_to_convert", aggfunc="mean"),
        median_days_to_convert=pd.NamedAgg(column="days_to_convert", aggfunc="median"),
        conversion_rate=pd.NamedAgg(column="converted", aggfunc="mean"),
        person_count=pd.NamedAgg(column="person_id", aggfunc="count")
    ).reset_index()

    return summary.round(4)
