import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def most_similar_high_value_person(
    person_id: int,
    persons_df: pd.DataFrame,
    field_map_df: pd.DataFrame,
    monetary_metric: str,
    frequency_metric: str,
    n_neighbors: int = 5,
    top_pct: float = 0.1
) -> dict:
    """
    Finds the most similar high-value person to a given person based on selected metrics.

    Parameters:
    -----------
    person_id : int
        The person_id of the target user.

    persons_df : pd.DataFrame
        Must include: person_id, display_name, source_type, attributes (as dict)

    field_map_df : pd.DataFrame
        Must include: source_type, normalized_field

    monetary_metric : str
        Normalized field for spend (e.g., 'total_spend').

    frequency_metric : str
        Normalized field for frequency (e.g., 'purchase_count').

    Returns:
    --------
    dict with:
        - most_similar_person : dict
        - similarity_score : float
        - high_value_persons : list of dicts
        - index_map : dict
    """
    def is_metric_available(source_type, field):
        return not field_map_df[
            (field_map_df["source_type"] == source_type) &
            (field_map_df["normalized_field"] == field)
        ].empty

    data = []
    index_map = {}

    for _, row in persons_df.iterrows():
        source = row.get("source_type")
        attrs = row.get("attributes", {}) or {}

        if not (is_metric_available(source, monetary_metric) and is_metric_available(source, frequency_metric)):
            continue

        try:
            monetary_value = float(attrs.get(monetary_metric))
            frequency_value = float(attrs.get(frequency_metric))
        except (TypeError, ValueError):
            continue

        data.append({
            "person_id": row["person_id"],
            "display_name": row.get("display_name", row["person_id"]),
            "monetary": monetary_value,
            "frequency": frequency_value
        })
        index_map[row["person_id"]] = row.get("display_name", row["person_id"])

    df = pd.DataFrame(data)
    if df.empty:
        return {"error": "No valid persons with selected metrics"}

    df["score"] = df["monetary"] * df["frequency"]
    cutoff = df["score"].quantile(1 - top_pct)
    high_value = df[df["score"] >= cutoff].reset_index(drop=True)

    if person_id not in high_value["person_id"].values:
        return {"error": "Person not found in high-value set"}

    scaler = StandardScaler()
    high_value_scaled = scaler.fit_transform(high_value[["monetary", "frequency"]])
    target_vector = scaler.transform(high_value[high_value["person_id"] == person_id][["monetary", "frequency"]])

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(high_value_scaled)
    distances, indices = nn.kneighbors(target_vector)

    most_similar_index = indices[0][1]
    most_similar_row = high_value.iloc[most_similar_index]
    similarity_score = 1 / (distances[0][1] + 1e-6)

    return {
        "most_similar_person": most_similar_row[["person_id", "display_name", "monetary", "frequency"]].to_dict(),
        "similarity_score": similarity_score,
        "high_value_persons": high_value.to_dict(orient="records"),
        "index_map": index_map
    }
