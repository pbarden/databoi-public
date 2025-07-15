import pandas as pd
from typing import Callable, Optional, Union, Dict
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def gradient_boosted_tree(
    persons_df: pd.DataFrame,
    events_df: pd.DataFrame,
    segment_filter_fn: Callable[[pd.DataFrame], pd.DataFrame],
    target_field: str,
    target_value: Union[int, float, str] = 1,
    field_label_map: Optional[Dict[str, str]] = None,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    n_estimators: int = 100
) -> dict:
    """
    Applies a gradient-boosted tree to analyze causal signals for conversion/churn using profile + event data.

    Parameters:
    -----------
    persons_df : pd.DataFrame
        Person-level data with:
        - 'id'
        - 'source_type'
        - 'attributes' (dict)
        - 'segment' or any filterable tag

    events_df : pd.DataFrame
        Event-level data with:
        - 'person_id'
        - 'event_name'
        - 'timestamp'
        - optional 'is_conversion' flag

    segment_filter_fn : function
        A function that filters persons_df and returns the appropriate segment

    target_field : str
        Target field to model (must be inside attributes or derived metrics)

    target_value : int, float, or str
        Class value that represents the positive label

    field_label_map : dict
        Optional mapping from normalized field names to human-readable labels

    Returns:
    --------
    dict with:
        - feature importances
        - classification report
        - trained model
    """

    segment_persons = segment_filter_fn(persons_df)
    if segment_persons.empty:
        return {"error": "No persons found in segment"}

    rows = []
    for _, person in segment_persons.iterrows():
        pid = person["id"]
        source_type = person.get("source_type", "unknown")
        attrs = person.get("attributes", {}) or {}

        person_events = events_df[events_df["person_id"] == pid]
        is_conversion = person_events.get("is_conversion", lambda x: False)
        if callable(is_conversion):
            conversions = person_events[is_conversion(person_events)]
        else:
            conversions = person_events[person_events["is_conversion"] == True]

        flat = {}
        flat.update(attrs)
        flat["event_count"] = len(person_events)
        flat["conversion_count"] = len(conversions)
        flat["last_event"] = person_events["event_name"].iloc[-1] if not person_events.empty else None
        flat["converted"] = int(len(conversions) > 0)
        flat["churned"] = 1 if flat["conversion_count"] == 0 else 0
        flat["person_id"] = pid

        rows.append(flat)

    df = pd.DataFrame(rows).set_index("person_id")

    if target_field not in df.columns:
        return {"error": f"Target field '{target_field}' not found in dataset."}

    df = df.dropna(subset=[target_field])
    if df.empty:
        return {"error": "No data available after filtering on target field."}

    X = df.drop(columns=[target_field])
    y = (df[target_field] == target_value).astype(int)

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.factorize(X[col])[0]

    if y.nunique() < 2:
        return {"error": "Target variable does not vary — cannot model."}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    model = XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    label_map = field_label_map or {}
    importances = {
        label_map.get(field, field): round(score, 5)
        for field, score in zip(X.columns, model.feature_importances_)
        if score > 0
    }

    return {
        "feature_importances": importances,
        "classification_report": classification_report(y_test, y_pred, output_dict=False),
        "model": model
    }
