import pandas as pd
import math
from scipy.stats import norm

def run_z_test(df, segment_a_filter, segment_b_filter, metric_field='converted'):
    """
    Runs a Z-test on two customer segments for a binary conversion-type metric using pandas DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing person-level data with at least the following columns:
        - 'source_type': str
        - 'attributes': dict or JSON-like column containing metric_field
    segment_a_filter : function
        A callable that takes the full DataFrame and returns a filtered DataFrame for segment A
    segment_b_filter : function
        A callable that takes the full DataFrame and returns a filtered DataFrame for segment B
    metric_field : str
        The key inside 'attributes' dict to extract as the binary metric (e.g., 'converted')

    Returns
    -------
    dict:
        {
            'z_score': float,
            'p_value': float,
            'proportions': {'segment_a': float, 'segment_b': float},
            'counts': {'segment_a': int, 'segment_b': int}
        }
    """

    def extract_binary_metric(df_segment):
        values = []
        for _, row in df_segment.iterrows():
            if pd.isna(row['source_type']):
                continue

            attributes = row['attributes']
            if not isinstance(attributes, dict):
                continue

            val = attributes.get(metric_field)
            if val is None:
                continue

            try:
                binary_val = int(val)
                values.append(1 if binary_val else 0)
            except (ValueError, TypeError):
                continue

        return values

    df_a = segment_a_filter(df)
    df_b = segment_b_filter(df)

    if df_a.empty or df_b.empty:
        raise ValueError("Z-test requires non-empty segments A and B.")

    metrics_a = extract_binary_metric(df_a)
    metrics_b = extract_binary_metric(df_b)

    n1, x1 = len(metrics_a), sum(metrics_a)
    n2, x2 = len(metrics_b), sum(metrics_b)

    if n1 == 0 or n2 == 0:
        raise ValueError(
            f"Cannot run Z-test — segment missing valid persons with metric '{metric_field}'"
        )

    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    denominator = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

    if denominator == 0:
        raise ZeroDivisionError("Z-test denominator is zero — possibly identical proportions")

    z_score = (p1 - p2) / denominator
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    return {
        'z_score': z_score,
        'p_value': p_value,
        'proportions': {'segment_a': p1, 'segment_b': p2},
        'counts': {'segment_a': n1, 'segment_b': n2}
    }
