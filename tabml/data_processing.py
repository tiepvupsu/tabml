import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
        # return array to be consistent with scikit transformer
        return np.array(X.clip(self.lower, self.upper, axis="columns"))


class CategoryEncoder:
    """A class to support encoding categorical columns.

    Attributes:
        vocab_map: a dictionary mapping each category to an integer
    """

    def __init__(self, vocab_map: Dict[Any, int]):
        self.vocab_map = vocab_map

    def get_encoded(self, series: pd.Series):
        """Maps each category to an integer based on vocabulary map.

        Unknown category will be mapped to the extra integer.
        """
        return (
            series.map(self.vocab_map)
            .fillna(max([*self.vocab_map.values()]) + 1)
            .astype(int)
        )

    @classmethod
    def from_list(cls, vocab: List[Any]):
        vocab_map = {word: i for i, word in enumerate(vocab)}
        return cls(vocab_map)

    @classmethod
    def from_lines_in_txt(cls, file_path: str):
        """Instantiates from a txt file, each line is a word."""
        with open(file_path, "r") as file:
            vocab = [line.strip() for line in file if line.strip()]
            return cls.from_list(vocab)

    @classmethod
    def from_mapping_in_csv(cls, file_path: str):
        """Instantiates from a csv file, each line is a mapping, no header."""
        df = pd.read_csv(file_path, names=["word", "code"])
        df["code"] = df["code"].astype(int)
        vocab_map = {word.strip(): code for word, code in zip(df["word"], df["code"])}
        return cls(vocab_map)


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
