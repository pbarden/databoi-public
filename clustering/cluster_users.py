import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Callable, List, Optional, Dict

def cluster_users(
    persons_df: pd.DataFrame,
    sessions_df: Optional[pd.DataFrame],
    events_df: Optional[pd.DataFrame],
    segment_filter_fn: Callable[[pd.DataFrame], pd.DataFrame],
    feature_fields: List[str],
    level: str = "person",
    n_clusters: int = 3
) -> dict:
    """
    Cluster users using KMeans on profile, session, or event-level data.

    Parameters:
    -----------
    persons_df : pd.DataFrame
        Person-level records. Should contain:
        - 'id'
        - 'attributes' (dict)
        - any flat feature columns if preprocessed

    sessions_df : pd.DataFrame
        Session-level records. Should contain:
        - 'person_id'
        - 'attributes' (dict) or structured fields

    events_df : pd.DataFrame
        Event-level records. Should contain:
        - 'person_id'
        - 'custom_dimensions' or equivalent

    segment_filter_fn : Callable
        Function that filters persons_df to the appropriate segment

    feature_fields : list[str]
        List of normalized field keys to use for clustering

    level : str
        One of "person", "session", or "event"

    n_clusters : int
        Number of clusters

    Returns:
    --------
    dict:
        {
            "cluster_centers": np.ndarray,
            "labels": list[int],
            "clustered_df": pd.DataFrame
        }
    """

    # Filter persons
    segment_persons = segment_filter_fn(persons_df)
    if segment_persons.empty:
        raise ValueError("Segment returned no persons. Cannot perform clustering.")

    person_ids = segment_persons["id"].unique()

    if level == "person":
        source_df = segment_persons.copy()
    elif level == "session":
        if sessions_df is None:
            raise ValueError("Session data required for 'session' level clustering.")
        source_df = sessions_df[sessions_df["person_id"].isin(person_ids)].copy()
    elif level == "event":
        if events_df is None:
            raise ValueError("Event data required for 'event' level clustering.")
        source_df = events_df[events_df["person_id"].isin(person_ids)].copy()
    else:
        raise ValueError("Invalid level. Choose from: 'person', 'session', or 'event'.")

    if source_df.empty:
        raise ValueError(f"No {level} records available for selected segment.")

    # Build feature matrix from normalized keys
    matrix = {}
    for field in feature_fields:
        values = []
        for _, row in source_df.iterrows():
            val = None

            if "attributes" in row and isinstance(row["attributes"], dict):
                val = row["attributes"].get(field)
            elif "custom_dimensions" in row and isinstance(row["custom_dimensions"], dict):
                val = row["custom_dimensions"].get(field)
            elif "device_info" in row and isinstance(row["device_info"], dict):
                val = row["device_info"].get(field)
            elif field in row:
                val = row[field]

            values.append(val)

        matrix[field] = pd.to_numeric(pd.Series(values), errors="coerce")

    df = pd.DataFrame(matrix).dropna()
    if df.empty:
        raise ValueError("No usable feature data extracted. Check your normalized field keys.")

    # Apply clustering
    scaled = StandardScaler().fit_transform(df)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(scaled)

    df["cluster"] = model.labels_

    return {
        "cluster_centers": model.cluster_centers_,
        "labels": model.labels_.tolist(),
        "clustered_df": df
    }
