import pandas as pd
from datetime import datetime
from typing import List, Callable, Dict, Tuple
from itertools import combinations
from scipy.stats import chi2_contingency

def chi_squared_pairwise(
    persons_df: pd.DataFrame,
    events_df: pd.DataFrame,
    segment_filter_map: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]],
    field_map: Dict[Tuple[str, str], str],
    conversion_event_name: str,
    start_date: datetime,
    end_date: datetime,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Performs pairwise chi-squared tests across segments based on deduplicated person-level conversion.

    Parameters:
    -----------
    persons_df : pd.DataFrame
        Person-level data. Must include 'id' and 'source_type'.

    events_df : pd.DataFrame
        Event-level data. Must include:
        - 'person_id'
        - 'timestamp'
        - 'event_name'
        - 'custom_dimensions' (dict)

    segment_filter_map : dict
        Dictionary mapping segment names to functions that filter `persons_df` accordingly.

    field_map : dict
        Maps (source_type, normalized_field) to a raw_tag string.

    conversion_event_name : str
        Normalized field name to resolve to raw tag in custom_dimensions.

    start_date, end_date : datetime
        Analysis window.

    alpha : float
        Significance threshold (default 0.05)

    Returns:
    --------
    pd.DataFrame with pairwise chi-squared test results.
    """

    if len(segment_filter_map) < 2:
        raise ValueError("At least two segments are required.")

    group_summary = []

    for segment_name, filter_fn in segment_filter_map.items():
        filtered_persons = filter_fn(persons_df)
        person_ids = filtered_persons["id"].unique()
        person_count = len(person_ids)

        if person_count == 0:
            conversions = 0
        else:
            source_types = filtered_persons["source_type"].dropna().unique()
            raw_tag = None
            for src in source_types:
                raw_tag = field_map.get((src, conversion_event_name))
                if raw_tag:
                    break

            if not raw_tag:
                conversions = 0
            else:
                subset = events_df[
                    (events_df["person_id"].isin(person_ids)) &
                    (events_df["timestamp"] >= start_date) &
                    (events_df["timestamp"] < end_date) &
                    (events_df["event_name"] == conversion_event_name)
                ]

                def is_valid(row):
                    dims = row.get("custom_dimensions", {})
                    return isinstance(dims, dict) and dims.get(raw_tag) is not None

                converted_ids = subset[subset.apply(is_valid, axis=1)]["person_id"].dropna().unique()
                conversions = len(set(converted_ids))

        group_summary.append({
            "group": segment_name,
            "persons": person_count,
            "conversions": conversions
        })

    df = pd.DataFrame(group_summary)
    df["failures"] = df["persons"] - df["conversions"]
    df["conversion_rate"] = df["conversions"] / df["persons"]
    df = df[df["persons"] > 0].copy()

    if df.shape[0] < 2:
        raise ValueError("At least two valid segments with person data are required.")

    group_lookup = df.set_index("group").to_dict(orient="index")
    variations = df["group"].tolist()

    results = []
    for g1, g2 in combinations(variations, 2):
        d1, d2 = group_lookup[g1], group_lookup[g2]
        contingency = [
            [d1["conversions"], d1["failures"]],
            [d2["conversions"], d2["failures"]]
        ]
        chi2, p_val, _, _ = chi2_contingency(contingency)

        cr1, cr2 = d1["conversion_rate"], d2["conversion_rate"]
        lift = (cr2 - cr1) * 100

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
    total_tests = len(results_df)
    results_df["p_bonferroni"] = (results_df["p_value"] * total_tests).clip(upper=1.0)
    results_df["stat_sig"] = results_df["p_value"] < alpha
    results_df["stat_sig_bonf"] = results_df["p_bonferroni"] < alpha

    def interpret(row):
        if row["stat_sig_bonf"]:
            return "Significant (corrected)"
        elif row["stat_sig"]:
            return "Significant"
        else:
            return "Not significant"

    results_df["interpretation"] = results_df.apply(interpret, axis=1)
    return results_df
