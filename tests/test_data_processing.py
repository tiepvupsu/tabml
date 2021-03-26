import pandas as pd
import pytest
from qcore.asserts import AssertRaises, assert_eq

from tabml.data_processing import QuantileClipper


class TestQuantileClipper:
    def test_no_nan(self):
        # 5
        nums = 5 * [-1.0] + 90 * [1] + 5 * [99]
        df = pd.DataFrame(data={"col": nums})
        expected_data = 5 * [0.0] + 90 * [1] + 5 * [50]
        expected = pd.Series(data=expected_data)
        # got = QuantileClipper(lower_percentile=.05, upper_percentile=.95).fit(
        #     X=df["col"]).transform(df["col"])
        got = QuantileClipper(
            lower_percentile=0.05, upper_percentile=0.95, interpolation="midpoint"
        ).fit_transform(df["col"])
        pd.testing.assert_series_equal(expected, got, check_names=False)
