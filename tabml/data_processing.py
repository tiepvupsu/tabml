from typing import Optional

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


def transform(df, columns, training_filters, transformer):
    all_data = df[columns]
    train_data = df.query(" and ".join(training_filters))[columns]
    transformer.fit(X=train_data)
    return transformer.transform(all_data)
