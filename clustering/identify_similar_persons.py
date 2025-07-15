import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional

def identify_similar_persons(
    selected_person_id: int,
    monetary_metric: str,
    frequency_metric: str,
    persons_df: pd.DataFrame,
    events_df: pd.DataFrame,
    field_map_df: pd.DataFrame,
    n_neighbors: int = 5,
    top_pct: float = 0.1,
    segment_df: Optional[pd.DataFrame] = None
) -> dict:
    """
    Identifies similar persons based on monetary and frequency metrics.

    Parameters:
    -----------
    selected_person_id : int
        The person_id to compare against.

    monetary_metric : str
        Normalized field for monetary value.

    frequency_metric : str
        Normalized field for frequency.

    persons_df : pd.DataFrame
        Must contain: person_id, display_name, source_type, attributes (as dict)

    events_df : pd.DataFrame
        Must contain: person_id, event_name, value (float)

    field_map_df : pd.DataFrame
        Must contain: source_type, normalized_field

    n_neighbors : int
        Number of similar persons to return.

    top_pct : float
        Must be in (0, 1). Defines high-value cutoff.

    segment_df : Optional[pd.DataFrame]
        Subset of persons_df to restrict comparisons to (same structure as persons_df).

    Returns:
    --------
    dict with:
        - similar_persons: pd.DataFrame
        - index_map: dict
    """
    if not 0 < top_pct < 1:
        raise ValueError("top_pct must be between 0 and 1.")

    if segment_df is not None:
        persons_df = persons_df[persons_df["person_id"].isin(segment_df["person_id"])]

    if persons_df.empty:
        return {"similar_persons": pd.DataFrame(), "index_map": {}}

    if selected_person_id not in persons_df["person_id"].values:
        return {"error": "Selected person not found in dataset."}

    selected_person_row = persons_df[persons_df["person_id"] == selected_person_id].iloc[0]
    source_type = selected_person_row["source_type"]

    def check_field_available(field):
        return not field_map_df[
            (field_map_df["source_type"] == source_type) &
            (field_map_df["normalized_field"] == field)
        ].empty

    if not check_field_available(monetary_metric):
        raise ValueError(f"Field '{monetary_metric}' not mapped for source_type '{source_type}'")
    if not check_field_available(frequency_metric):
        raise ValueError(f"Field '{frequency_metric}' not mapped for source_type '{source_type}'")

    person_map = persons_df.set_index("person_id").to_dict(orient="index")

    # Event-level aggregates (fallback)
    purchase_events = events_df[events_df["event_name"] == "purchase"]
    monetary_agg = purchase_events.groupby("person_id")["value"].apply(list).to_dict()
    frequency_agg = purchase_events["person_id"].value_counts().to_dict()

    df_rows = []
    for pid, person in person_map.items():
        attrs = person.get("attributes", {}) or {}

        try:
            monetary_val = float(attrs.get(monetary_metric, sum(monetary_agg.get(pid, []))))
        except (TypeError, ValueError):
            monetary_val = sum(monetary_agg.get(pid, []))

        try:
            frequency_val = int(attrs.get(frequency_metric, frequency_agg.get(pid, 0)))
        except (TypeError, ValueError):
            frequency_val = frequency_agg.get(pid, 0)

        df_rows.append({
            "person_id": pid,
            "display_name": person.get("display_name", pid),
            "monetary": monetary_val,
            "frequency": frequency_val
        })

    df = pd.DataFrame(df_rows)
    if df.empty or df["monetary"].sum() == 0 or df["frequency"].sum() == 0:
        raise ValueError("Metrics are invalid or empty — no usable data.")

    # Normalize and compute distance
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[["monetary", "frequency"]])

    target_idx = df[df["person_id"] == selected_person_id].index[0]
    selected_vector = scaled[target_idx]

    distances = np.linalg.norm(scaled - selected_vector, axis=1)
    nearest_idx = np.argsort(distances)[1:n_neighbors+1]

    similar_df = df.iloc[nearest_idx].copy()
    return {
        "similar_persons": similar_df.reset_index(drop=True),
        "index_map": {selected_person_id: list(similar_df["person_id"])}
    }
