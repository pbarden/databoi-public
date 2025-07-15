def generate_bayesian_priors(
    persons_df,
    events_df,
    segment_filter_fn,
    field_map,
    conversion_event_name,
    start_date,
    end_date,
    group_field=None,
    group_value=None,
    confidence_weight=1.0
) -> dict:
    """
    Computes alpha/beta priors from historical data using optional group filters.
    """

    persons = segment_filter_fn(persons_df)
    if persons.empty:
        return {
            "prior_alpha": 0.0,
            "prior_beta": 0.0,
            "effective_sample_size": 0.0,
            "observed_conversion_rate": 0.0,
            "person_count": 0,
            "conversion_count": 0
        }

    person_ids = persons["id"].unique()
    source_type = persons["source_type"].iloc[0]

    raw_conversion_tag = field_map.get((source_type, conversion_event_name))
    if not raw_conversion_tag:
        return {
            "prior_alpha": 0.0,
            "prior_beta": 0.0,
            "effective_sample_size": 0.0,
            "observed_conversion_rate": 0.0,
            "person_count": 0,
            "conversion_count": 0
        }

    events = events_df[
        (events_df["person_id"].isin(person_ids)) &
        (events_df["timestamp"] >= start_date) &
        (events_df["timestamp"] < end_date)
    ]

    if group_field and group_value:
        raw_group_tag = field_map.get((source_type, group_field))
        if raw_group_tag:
            def group_match(row):
                cd = row.get("custom_dimensions", {})
                return cd.get(raw_group_tag) == group_value if isinstance(cd, dict) else False
            events = events[events.apply(group_match, axis=1)]

    unique_persons = events["person_id"].dropna().unique()

    def is_conversion(row):
        cd = row.get("custom_dimensions", {})
        return cd.get(raw_conversion_tag) is not None if isinstance(cd, dict) else False

    conversion_ids = events[events.apply(is_conversion, axis=1)]["person_id"].dropna().unique()

    n = len(unique_persons)
    k = len(conversion_ids)

    if n == 0:
        return {
            "prior_alpha": 0.0,
            "prior_beta": 0.0,
            "effective_sample_size": 0.0,
            "observed_conversion_rate": 0.0,
            "person_count": 0,
            "conversion_count": 0
        }

    cr = k / n
    eff_n = n * confidence_weight
    alpha = round(eff_n * cr, 4)
    beta_val = round(eff_n * (1 - cr), 4)

    return {
        "prior_alpha": alpha,
        "prior_beta": beta_val,
        "effective_sample_size": round(eff_n, 2),
        "observed_conversion_rate": round(cr, 6),
        "person_count": n,
        "conversion_count": k
    }
