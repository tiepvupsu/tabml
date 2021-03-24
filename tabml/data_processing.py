from typing import Optional

import pandas as pd


def clip_one_column(
    df: pd.DataFrame,
    column_name: str,
    training_filter: Optional[str] = None,
    lower_percentile: Optional[float] = None,
    upper_percentile: Optional[float] = None,
) -> pd.Series:
    """Clips a column by upper and lower percentiles.

    NOTE: In many cases, we only want to find percentiles based on training data.

    Args:
        df:
            the input dataframe
        column_name:
            Name of column that clip operator applies to.
        training_filter:
            A boolean statement used to filter training data.
            In many cases, we only want to find percentiles based on training data.
            If None, this function is apply to all data.
        lower_percentile:
            a float number in [0, 1] reprsenting lower percentile.
        upper_percentile:
            a float number in [0, 1] reprsenting upper percentile.

    Raises:
        ValueError if:
            * lower_percentile > upper_percentile.
            * lower_percentile < 0
            * upper_percentile > 1
    """

    if lower_percentile is None:
        lower_percentile = 0

    if upper_percentile is None:
        upper_percentile = 1

    if lower_percentile < 0:
        raise ValueError(
            f"percentile should be in the interval [0, 1], got {lower_percentile}"
        )
    if upper_percentile > 1:
        raise ValueError(
            f"percentile should be in the interval [0, 1], got {upper_percentile}"
        )
    if lower_percentile > upper_percentile:
        raise ValueError(
            f"lower_percentile ({lower_percentile}) should not be greater than "
            f"upper_percentile ({upper_percentile})."
        )
    if training_filter is None:
        training_data = df[column_name]
    else:
        training_data = df.query(training_filter)[column_name]

    # find lower and upper thresholds based on training data
    lower = training_data.quantile(lower_percentile)
    upper = training_data.quantile(upper_percentile)
    # apply the clip to all data
    return df[column_name].clip(lower=lower, upper=upper)
