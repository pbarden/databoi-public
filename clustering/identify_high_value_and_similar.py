import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def identify_high_value_and_similar(
    persons_df: pd.DataFrame,
    events_df: pd.DataFrame,
    field_map_df: pd.DataFrame,
    monetary_metric: str,
    frequency_metric: str,
    n_neighbors: int = 5,
    top_pct: float = 0.1
):
    """
    Identifies high-value users based on monetary and frequency metrics,
    then finds similar users using nearest neighbors clustering.

    Parameters:
    -----------
    persons_df : pd.DataFrame
        Must include: person_id, display_name, source_type, attributes (as dict)

    events_df : pd.DataFrame
        Must include: person_id, event_name, value (optional)

    field_map_df : pd.DataFrame
        Must include: source_type, normalized_field

    monetary_metric : str
        Normalized field name (e.g., 'total_spend')

    frequency_metric : str
        Normalized field name (e.g., 'purchase_count')

    Returns:
    --------
    dict with keys:
        - high_value_persons : pd.DataFrame
        - similar_persons : pd.DataFrame
        - index_map : dict[int -> list[int]]
    """
    if not 0 < top_pct < 1:
        raise ValueError("top_pct must be between 0 and 1")

    if persons_df.empty:
        return {
            "high_value_persons": pd.DataFrame(),
            "similar_persons": pd.DataFrame(),
            "index_map": {}
        }

    # Build metric existence map
    def is_metric_available(source_type, field):
        return not field_map_df[
            (field_map_df["source_type"] == source_type) &
            (field_map_df["normalized_field"] == field)
        ].empty

    # Filter purchase events
    purchase_events = events_df[events_df["event_name"] == "purchase"].copy()
    if "value" not in purchase_events.columns:
        purchase_events["value"] = 0.0

    # Aggregate by person
    monetary_agg = purchase_events.groupby("person_id")["value"].sum().to_dict()
    frequency_agg = purchase_events.groupby("person_id").size().to_dict()

    rows = []
    person_map = persons_df.set_index("person_id").to_dict("index")

    for person_id, person_data in person_map.items():
        attrs = person_data.get("attributes", {}) or {}
        source_type = person_data.get("source_type", None)

        # Determine monetary value
        if is_metric_available(source_type, monetary_metric):
            val = attrs.get(monetary_metric)
            monetary = float(val) if val is not None else monetary_agg.get(person_id, 0.0)
        else:
            monetary = monetary_agg.get(person_id, 0.0)

        # Determine frequency value
        if is_metric_available(source_type, frequency_metric):
            val = attrs.get(frequency_metric)
            frequency = int(val) if val is not None else frequency_agg.get(person_id, 0)
        else:
            frequency = frequency_agg.get(person_id, 0)

        rows.append({
            "person_id": person_id,
            "monetary": monetary,
            "frequency": frequency
        })

    df = pd.DataFrame(rows)

    if df.empty or df[["monetary", "frequency"]].sum().sum() == 0:
        raise ValueError("Metric values are empty or invalid")

    # Normalize values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[["monetary", "frequency"]])

    # High-value user thresholding
    score = df["monetary"] + df["frequency"]
    threshold = score.quantile(1 - top_pct)
    high_value_indices = df[score >= threshold].index.tolist()

    # Nearest neighbors
    model = NearestNeighbors(n_neighbors=n_neighbors + 1)
    model.fit(scaled)
    _, indices = model.kneighbors(scaled)

    index_map = {}
    similar_indices = set()
    for idx in high_value_indices:
        neighbors = [i for i in indices[idx] if i != idx][:n_neighbors]
        index_map[idx] = neighbors
        similar_indices.update(neighbors)

    high_value_df = df.loc[high_value_indices].copy()
    similar_df = df.loc[list(similar_indices)].copy()

    # Append display_name
    name_lookup = persons_df.set_index("person_id")["display_name"].to_dict()
    high_value_df["display_name"] = high_value_df["person_id"].map(name_lookup)
    similar_df["display_name"] = similar_df["person_id"].map(name_lookup)

    return {
        "high_value_persons": high_value_df,
        "similar_persons": similar_df,
        "index_map": index_map
    }
