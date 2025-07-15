import pandas as pd
from typing import Callable, Dict, Tuple
from datetime import datetime
from math import sqrt
from scipy.stats import norm

def confidence_intervals(
    persons_df: pd.DataFrame,
    events_df: pd.DataFrame,
    control_filter_fn: Callable[[pd.DataFrame], pd.DataFrame],
    variant_filter_fn: Callable[[pd.DataFrame], pd.DataFrame],
    field_map: Dict[Tuple[str, str], list],
    conversion_event_key: str,
    start_date: datetime,
    end_date: datetime,
    confidence_level: float = 0.95
) -> dict:
    """
    Computes confidence intervals for conversion rates between two segments.

    Parameters:
    -----------
    persons_df : pd.DataFrame
        Person-level data with at least 'id' and 'source_type'

    events_df : pd.DataFrame
        Event-level data with at least:
        - 'person_id'
        - 'timestamp'
        - 'custom_dimensions' (dict)

    control_filter_fn : callable
        Function to filter persons_df for control group

    variant_filter_fn : callable
        Function to filter persons_df for variant group

    field_map : dict
        Maps (source_type, normalized_field) -> list of raw tag strings

    conversion_event_key : str
        Normalized key used to lookup raw conversion field(s)

    start_date, end_date : datetime
        Time window for events

    confidence_level : float
        Desired confidence level (e.g., 0.95 for 95%)

    Returns:
    --------
    dict with confidence intervals, conversion rates, and metadata
    """

    control_df = control_filter_fn(persons_df)
    variant_df = variant_filter_fn(persons_df)

    if control_df.empty or variant_df.empty:
        raise ValueError("Both control and variant segments must contain persons.")

    def count_conversions(group_df: pd.DataFrame) -> int:
        converted_ids = set()
        for _, person in group_df.iterrows():
            raw_tags = field_map.get((person['source_type'], conversion_event_key), [])
            if not raw_tags:
                continue

            person_events = events_df[
                (events_df['person_id'] == person['id']) &
                (events_df['timestamp'] >= start_date) &
                (events_df['timestamp'] < end_date)
            ]

            def has_valid_conversion(row):
                dims = row.get("custom_dimensions", {})
                if not isinstance(dims, dict):
                    return False
                for tag in raw_tags:
                    if dims.get(tag) is not None:
                        return True
                return False

            if not person_events.empty and person_events.apply(has_valid_conversion, axis=1).any():
                converted_ids.add(person['id'])

        return len(converted_ids)

    control_converted = count_conversions(control_df)
    variant_converted = count_conversions(variant_df)

    control_n = len(control_df)
    variant_n = len(variant_df)

    control_cr = control_converted / control_n
    variant_cr = variant_converted / variant_n

    se_control = sqrt(control_cr * (1 - control_cr) / control_n)
    se_variant = sqrt(variant_cr * (1 - variant_cr) / variant_n)

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
        "z_critical": round(z, 4),
        "control_n": control_n,
        "variant_n": variant_n,
        "control_converted": control_converted,
        "variant_converted": variant_converted
    }
