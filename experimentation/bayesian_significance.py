import numpy as np
from scipy.stats import beta

def bayesian_significance(
    persons_df,
    events_df,
    control_filter_fn,
    variant_filter_fn,
    field_map,
    conversion_event_name,
    start_date,
    end_date,
    prior_alpha=1.0,
    prior_beta=1.0,
    probability_threshold=0.95,
    sample_count=100_000
) -> dict:
    """
    Runs a Bayesian A/B significance test on filtered person segments.
    """

    control_persons = control_filter_fn(persons_df)
    variant_persons = variant_filter_fn(persons_df)

    if control_persons.empty or variant_persons.empty:
        raise ValueError("One or both segments returned no persons.")

    total_control = len(control_persons)
    total_variant = len(variant_persons)

    source_type = control_persons["source_type"].iloc[0]
    raw_tag = field_map.get((source_type, conversion_event_name))
    if not raw_tag:
        raise ValueError(f"Missing field map for '{conversion_event_name}' and source '{source_type}'.")

    def count_converted(persons_df):
        ids = persons_df["id"].unique()
        subset = events_df[
            (events_df["person_id"].isin(ids)) &
            (events_df["timestamp"] >= start_date) &
            (events_df["timestamp"] < end_date)
        ]
        def is_converted(row):
            cd = row.get("custom_dimensions", {})
            return cd.get(raw_tag) is not None if isinstance(cd, dict) else False
        return subset.apply(is_converted, axis=1).sum()

    conversions_control = count_converted(control_persons)
    conversions_variant = count_converted(variant_persons)

    alpha_c = conversions_control + prior_alpha
    beta_c = (total_control - conversions_control) + prior_beta
    alpha_v = conversions_variant + prior_alpha
    beta_v = (total_variant - conversions_variant) + prior_beta

    control_samples = np.random.beta(alpha_c, beta_c, sample_count)
    variant_samples = np.random.beta(alpha_v, beta_v, sample_count)

    prob_to_beat = np.mean(variant_samples > control_samples)
    expected_loss = np.mean(np.maximum(control_samples - variant_samples, 0))

    ci_control = (
        round(beta.ppf(0.025, alpha_c, beta_c), 6),
        round(beta.ppf(0.975, alpha_c, beta_c), 6)
    )
    ci_variant = (
        round(beta.ppf(0.025, alpha_v, beta_v), 6),
        round(beta.ppf(0.975, alpha_v, beta_v), 6)
    )

    return {
        "probability_to_beat_control": round(prob_to_beat, 6),
        "is_significant": prob_to_beat > probability_threshold,
        "expected_loss": round(expected_loss, 6),
        "posterior_control": (round(alpha_c, 2), round(beta_c, 2)),
        "posterior_variant": (round(alpha_v, 2), round(beta_v, 2)),
        "ci_control": ci_control,
        "ci_variant": ci_variant
    }
