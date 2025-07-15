import pandas as pd
from typing import Callable, Optional, Union, List, Dict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def causal_decision_tree(
    persons_df: pd.DataFrame,
    events_df: pd.DataFrame,
    segment_filter_fn: Callable[[pd.DataFrame], pd.DataFrame],
    target_field: str,
    target_value: Union[int, float, str] = 1,
    field_label_map: Optional[Dict[str, str]] = None,
    criterion: str = 'gini',
    max_depth: Optional[int] = None,
    min_samples_split: int = 2
) -> dict:
    """
    Runs a decision tree on profile + event data to infer patterns associated with a target field.

    Parameters:
    -----------
    persons_df : pd.DataFrame
        Person-level records. Must include:
        - 'id'
        - 'source_type'
        - 'attributes' (dict)

    events_df : pd.DataFrame
        Event-level records. Must include:
        - 'person_id'
        - 'event_name'
        - 'timestamp'
        - other behaviorally useful fields

    segment_filter_fn : function
        Function that filters persons_df and returns relevant rows

    target_field : str
        Field to be used as the classification target (must exist in person['attributes'])

    target_value : scalar
        Value indicating positive class

    field_label_map : dict (optional)
        Maps internal field names to readable labels

    criterion : str
        'gini' or 'entropy'

    max_depth : int or None
        Maximum tree depth

    min_samples_split : int
        Minimum split threshold

    Returns:
    --------
    dict with feature importances, report, and fitted tree object
    """

    segment_persons = segment_filter_fn(persons_df)
    if segment_persons.empty:
        return {"error": "No persons found in segment"}

    rows = []
    for _, person in segment_persons.iterrows():
        person_id = person["id"]
        source_type = person.get("source_type", "unknown")
        attrs = person.get("attributes", {}) or {}

        person_events = events_df[events_df["person_id"] == person_id]
        conversions = person_events[person_events.get("is_conversion", lambda x: False)(person_events)]

        flat = {}
        flat.update(attrs)

        flat["event_count"] = len(person_events)
        flat["conversion_count"] = len(conversions)
        flat["last_event"] = person_events["event_name"].iloc[-1] if not person_events.empty else None
        flat["converted"] = int(len(conversions) > 0)
        flat["churned"] = 1 if flat["conversion_count"] == 0 else 0
        flat["person_id"] = person_id

        rows.append(flat)

    df = pd.DataFrame(rows).set_index("person_id")

    if target_field not in df.columns:
        return {"error": f"Target field '{target_field}' not found in person attributes or event data."}

    df = df.dropna(subset=[target_field])
    if df.empty:
        return {"error": "No data rows available after filtering on target field."}

    # Encode all object/string fields except the target
    predictors = df.drop(columns=[target_field])
    for col in predictors.columns:
        if predictors[col].dtype == 'object':
            predictors[col] = pd.factorize(predictors[col])[0]

    X = predictors
    y = (df[target_field] == target_value).astype(int)

    if y.nunique() < 2:
        return {"error": "Target variable does not vary — decision tree cannot run"}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    tree_model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_test)

    # Use label mapping if provided
    label_map = field_label_map or {}
    importances = {
        label_map.get(col, col): round(score, 5)
        for col, score in zip(X.columns, tree_model.feature_importances_)
        if score > 0
    }

    return {
        "feature_importances": importances,
        "classification_report": classification_report(y_test, y_pred, output_dict=False),
        "tree": tree_model
    }
