import pandas as pd
from collections import defaultdict
from datetime import datetime

def estimate_attribution(
    persons_df: pd.DataFrame,
    events_df: pd.DataFrame,
    field_map_df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    channel_field: str,
    conversion_event_name: str,
    segment_filter_fn=None,
    min_path_length: int = 1
) -> pd.DataFrame:
    """
    Estimates channel attribution via first-order Markov approximation using event data.

    Parameters
    ----------
    persons_df : pd.DataFrame
        DataFrame with at least 'id' and 'source_type' columns.

    events_df : pd.DataFrame
        DataFrame with event data including: 'person_id', 'timestamp', 'event_name', 'custom_dimensions', 'source_type'.

    field_map_df : pd.DataFrame
        DataFrame mapping normalized fields to raw tags. Must contain: 'source_type', 'normalized_field', 'raw_tag'.

    start_date : datetime
    end_date : datetime

    channel_field : str
        Normalized field key (e.g., 'channel_grouping') to resolve from custom_dimensions.

    conversion_event_name : str
        The event name that signals a conversion.

    segment_filter_fn : Callable or None
        Function to filter `persons_df` to match a segment. If None, all persons are used.

    min_path_length : int
        Minimum number of steps in a journey path for inclusion.

    Returns
    -------
    pd.DataFrame
    """
    if segment_filter_fn:
        persons_df = segment_filter_fn(persons_df)
    if persons_df.empty:
        return pd.DataFrame(columns=[
            "channel", "baseline_conversion_rate", "removal_conversion_rate", "contribution_lift"
        ])

    person_ids = set(persons_df["id"])

    # Step 2: Resolve raw channel tags for all source_types
    source_types = persons_df["source_type"].dropna().unique()
    raw_tags = set(
        field_map_df[
            (field_map_df["normalized_field"] == channel_field) &
            (field_map_df["source_type"].isin(source_types))
        ]["raw_tag"]
    )

    if not raw_tags:
        return pd.DataFrame(columns=[
            "channel", "baseline_conversion_rate", "removal_conversion_rate", "contribution_lift"
        ])

    # Step 3: Filter events by person and time range
    mask = (
        events_df["person_id"].isin(person_ids) &
        (events_df["timestamp"] >= start_date) &
        (events_df["timestamp"] <= end_date)
    )
    events = events_df.loc[mask].sort_values(by=["person_id", "timestamp"])

    if events.empty:
        return pd.DataFrame(columns=[
            "channel", "baseline_conversion_rate", "removal_conversion_rate", "contribution_lift"
        ])

    # Step 4: Build journeys
    journeys = defaultdict(list)
    conversions = set()

    for _, row in events.iterrows():
        # Add channel step from raw_tag
        for tag in raw_tags:
            if tag in row["custom_dimensions"]:
                journeys[row["person_id"]].append(str(row["custom_dimensions"][tag]).strip())
                break

        # Check for conversion
        if row["event_name"] == conversion_event_name:
            conversions.add(row["person_id"])

    # Step 5: Construct DataFrame with paths
    data = []
    for pid, path in journeys.items():
        if len(path) < min_path_length:
            continue
        data.append({
            "person_id": pid,
            "path": path,
            "converted": 1 if pid in conversions else 0
        })

    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=[
            "channel", "baseline_conversion_rate", "removal_conversion_rate", "contribution_lift"
        ])

    # Clean paths
    df["path"] = df["path"].apply(lambda x: [step for step in x if step])

    all_channels = set(step for path in df["path"] for step in path)
    baseline_cr = df["converted"].mean()

    results = []
    for channel in sorted(all_channels):
        df_mod = df.copy()
        df_mod["path"] = df_mod["path"].apply(lambda steps: [s for s in steps if s != channel])
        df_mod = df_mod[df_mod["path"].apply(len) > 0]

        mod_cr = df_mod["converted"].mean() if not df_mod.empty else 0.0
        lift = baseline_cr - mod_cr

        results.append({
            "channel": channel,
            "baseline_conversion_rate": round(baseline_cr, 4),
            "removal_conversion_rate": round(mod_cr, 4),
            "contribution_lift": round(lift, 4)
        })

    return pd.DataFrame(results).sort_values("contribution_lift", ascending=False)
