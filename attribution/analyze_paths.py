import pandas as pd
from collections import defaultdict, Counter
from itertools import islice
from typing import Union, Callable

def analyze_paths(
    persons_df: pd.DataFrame,
    events_df: pd.DataFrame,
    segment_filter_fn: Callable[[pd.DataFrame], pd.DataFrame],
    field_map_df: pd.DataFrame,
    conversion_field_key: str,
    conversion_field_value: Union[str, list],
    window_size: int = 5
) -> dict:
    """
    Analyzes user event paths for a segment to identify patterns and lift
    associated with specific events in the lead-up to a conversion.

    Parameters:
    -----------
    persons_df : pd.DataFrame
        Person-level data (must include 'id', 'source_type').

    events_df : pd.DataFrame
        Event-level data (must include 'person_id', 'timestamp', 'event_name', 'custom_dimensions', 'source_type').

    segment_filter_fn : function
        Callable that filters the input persons_df to those in the desired segment.

    field_map_df : pd.DataFrame
        Lookup table of normalized field → raw tag for each source_type. Required columns:
        - 'source_type'
        - 'normalized_field'
        - 'raw_tag'

    conversion_field_key : str
        Normalized field name that denotes the conversion identifier (e.g. 'event_type').

    conversion_field_value : str | list
        Value or list of values that denote conversion (e.g., 'purchase').

    window_size : int
        Number of prior events to include before a conversion.

    Returns:
    --------
    dict
    """
    segment_users = segment_filter_fn(persons_df)
    user_ids = set(segment_users["id"])

    if not user_ids:
        return {
            'converted_users': 0,
            'non_converted_users': 0,
            'top_pre_conversion_paths': [],
            'top_non_converted_paths': [],
            'distinctive_events': {}
        }

    # Subset events
    events = events_df[events_df["person_id"].isin(user_ids)].copy()
    if events.empty:
        return {
            'converted_users': 0,
            'non_converted_users': 0,
            'top_pre_conversion_paths': [],
            'top_non_converted_paths': [],
            'distinctive_events': {}
        }

    # Sort by person, timestamp
    events = events.sort_values(by=["person_id", "timestamp"])

    # Build tag resolution map
    tag_map = {
        (row["source_type"], row["normalized_field"]): row["raw_tag"]
        for _, row in field_map_df.iterrows()
    }

    def resolve_conversion_tag(source_type):
        return tag_map.get((source_type, conversion_field_key), None)

    # Process events per person
    events_by_user = defaultdict(list)
    for _, row in events.iterrows():
        events_by_user[row["person_id"]].append(row)

    converted_paths = []
    nonconverted_paths = []
    event_counter_converted = Counter()
    event_counter_nonconverted = Counter()
    converted_users = set()

    def extract_conversion_value(event_row):
        raw_tag = resolve_conversion_tag(event_row["source_type"])
        return event_row["custom_dimensions"].get(raw_tag) if raw_tag else None

    def matches(value):
        if isinstance(conversion_field_value, list):
            return value in conversion_field_value
        return value == conversion_field_value

    for uid, user_events in events_by_user.items():
        conv_index = next(
            (i for i, e in enumerate(user_events) if matches(extract_conversion_value(e))),
            None
        )

        if conv_index is not None:
            converted_users.add(uid)
            pre = tuple(e["event_name"] for e in islice(user_events, max(0, conv_index - window_size), conv_index))
            if pre:
                converted_paths.append(pre)
                event_counter_converted.update(pre)
        else:
            path = tuple(e["event_name"] for e in user_events[-window_size:])
            if path:
                nonconverted_paths.append(path)
                event_counter_nonconverted.update(path)

    n_converted = len(converted_users)
    n_nonconverted = len(user_ids) - n_converted

    top_conv_paths = Counter(converted_paths).most_common(10)
    top_nonconv_paths = Counter(nonconverted_paths).most_common(10)

    # Lift calculation
    distinctive_events = {}
    all_event_names = set(event_counter_converted) | set(event_counter_nonconverted)
    for event_name in all_event_names:
        conv_rate = event_counter_converted[event_name] / n_converted if n_converted else 0
        nonconv_rate = event_counter_nonconverted[event_name] / n_nonconverted if n_nonconverted else 0

        if nonconv_rate == 0:
            lift = float('inf') if conv_rate > 0 else 0
        else:
            lift = round(conv_rate / nonconv_rate, 2)

        distinctive_events[event_name] = lift

    return {
        'converted_users': n_converted,
        'non_converted_users': n_nonconverted,
        'top_pre_conversion_paths': top_conv_paths,
        'top_non_converted_paths': top_nonconv_paths,
        'distinctive_events': dict(sorted(distinctive_events.items(), key=lambda x: -x[1])[:10])
    }
