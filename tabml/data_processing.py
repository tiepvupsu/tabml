import hashlib
from typing import Any, List, Optional, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        lower_percentile: Optional[float] = None,
        upper_percentile: Optional[float] = None,
        interpolation="linear",
    ):
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
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.interpolation = interpolation
        self.lower = None
        self.upper = None

    def fit(self, X, y=None):
        self.lower = X.quantile(self.lower_percentile, interpolation=self.interpolation)
        self.upper = X.quantile(self.upper_percentile, interpolation=self.interpolation)
        return self

    def transform(self, X):
        return X.clip(lower=self.lower, upper=self.upper)


def find_boxplot_boundaries(
    col: pd.Series, whisker_coeff: float = 1.5
) -> Tuple[float, float]:
    """Findx minimum and maximum in boxplot.

    Args:
        col: a pandas serires of input.
        whisker_coeff: whisker coefficient in box plot
    """
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - whisker_coeff * IQR
    upper = Q3 + whisker_coeff * IQR
    return lower, upper


class BoxplotOutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, whisker_coeff: float = 1.5):
        self.whisker = whisker_coeff
        self.lower: float = float("-inf")
        self.upper: float = float("inf")

    def fit(self, X: pd.DataFrame, y=None):
        self.lower, self.upper = find_boxplot_boundaries(X, self.whisker)
        return self

    def transform(self, X: pd.DataFrame):
        return X.clip(self.lower, self.upper, axis="columns")


def fit_train_transform_all(
    whole_df: pd.DataFrame,
    input_columns: List[str],
    training_filters: List[str],
    transformer,
):
    """Fits transformer on training data and use it to transform all data.

    This is essentially helpful when we want to apply feature transformer only to
    training data to avoid data leakage.

    Args:
        whole_df:
            The whole dataframe, including all data and neccessary columns for filtering
            training data.
        input_columns:
            A list of column names used to fit the transformer
        training_filters:
            A list of filters used to extract training data
        transformer:
            A sklearn-like transformer which has .fit() and transform() methods.

    Returns:
        Transformed array of all data.
    """
    all_data = whole_df[input_columns]
    train_data = whole_df.query(" and ".join(training_filters))[input_columns]
    transformer.fit(X=train_data)
    return transformer.transform(all_data)


def cross_columns(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Crosses multiple columns to create a new column.

    Resulting column has dtype = string, values of input columns are joined by "_X_"

    Args:
        df: input dataframe.
        cols: columns to cross.
    """
    return df.apply(lambda x: "_X_".join(str(x[col]) for col in cols), axis=1)


def hash_modulo(val: Any, mod: int) -> int:
    # TODO: make it faster and with more ways of hashing
    md5 = hashlib.md5()
    md5.update(str(val).encode())
    return int(md5.hexdigest(), 16) % mod
