import pandas as pd
from scipy.stats import chisquare
from datetime import datetime
from typing import Callable, Dict, List

def sample_rate_mismatch(
    persons_df: pd.DataFrame,
    events_df: pd.DataFrame,
    group_field: str,
    group_values: List[str],
    start_date: datetime,
    end_date: datetime,
    field_map: Dict[tuple, str],
    segment_filter_fn: Callable[[pd.DataFrame], pd.DataFrame] = None,
    alpha: float = 0.05
) -> dict:
    """
    Detects SRM (Sample Ratio Mismatch) across multiple experiment groups by comparing actual
    person distributions vs expected proportions.

    Parameters:
    -----------
    persons_df : pd.DataFrame
        Person-level data. Must include 'id' and 'source_type'.

    events_df : pd.DataFrame
        Event-level data with at least:
        - 'person_id'
        - 'timestamp'
        - 'custom_dimensions' (dict)

    group_field : str
        Normalized field name (e.g., 'group')

    group_values : list[str]
        Expected experimental group values (e.g., ['A', 'B'])

    start_date, end_date : datetime
        Time range to restrict events

    field_map : dict
        Maps (source_type, normalized_field) → raw_tag

    segment_filter_fn : callable
        Optional function to filter `persons_df` to a segment

    alpha : float
        Significance threshold

    Returns:
    --------
    dict with SRM test results
    """

    if segment_filter_fn:
        persons_df = segment_filter_fn(persons_df)

    if persons_df.empty:
        raise ValueError("No persons found for the selected segment.")

    person_ids = persons_df["id"].unique()
    source_type_map = persons_df.set_index("id")["source_type"].to_dict()

    # Filter events to relevant persons and time range
    events = events_df[
        (events_df["person_id"].isin(person_ids)) &
        (events_df["timestamp"] >= start_date) &
        (events_df["timestamp"] < end_date)
    ].copy()

    group_counts = []

    for group in group_values:
        matched_ids = set()

        for person_id in person_ids:
            source = source_type_map.get(person_id)
            raw_tag = field_map.get((source, group_field))

            if not raw_tag:
                continue

            person_events = events[events["person_id"] == person_id]

            def match(row):
                dims = row.get("custom_dimensions", {})
                return isinstance(dims, dict) and dims.get(raw_tag) == group

            if not person_events.empty and person_events.apply(match, axis=1).any():
                matched_ids.add(person_id)

        group_counts.append({"group": group, "persons": len(matched_ids)})

    group_df = pd.DataFrame(group_counts)

    if group_df.empty or group_df["persons"].sum() == 0:
        raise ValueError("No person data found for any group in the selected range.")

    observed = group_df["persons"].tolist()
    expected = [sum(observed) / len(observed)] * len(observed)

    chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
    is_mismatch = p_value < alpha

    return {
        "is_mismatch": is_mismatch,
        "chi2_stat": round(chi2_stat, 4),
        "p_value": round(p_value, 6),
        "observed": observed,
        "expected": expected,
        "group_names": group_df["group"].tolist()
    }
