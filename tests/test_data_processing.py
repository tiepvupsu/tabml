import pandas as pd
from qcore.asserts import assert_eq

from tabml.data_processing import QuantileClipper, find_boxplot_boundaries


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


class TestFindBoxPlotBoundaries:
    def test_1(self):
        df = pd.DataFrame(data={"a": ["a", "b", "c", "d"], "val": [1, 2, 3, 4]})
        got = find_boxplot_boundaries(df["val"])
        # Q1 = 1.75, Q3 = 3.25, IQR = 1.5
        expected = -0.5, 5.5
        assert_eq(expected, got)
