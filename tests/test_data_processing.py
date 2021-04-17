import pandas as pd
from qcore.asserts import assert_eq

from tabml import data_processing


class TestQuantileClipper:
    def test_no_nan(self):
        # 5
        nums = 5 * [-1.0] + 90 * [1] + 5 * [99]
        df = pd.DataFrame(data={"col": nums})
        expected_data = 5 * [0.0] + 90 * [1] + 5 * [50]
        expected = pd.Series(data=expected_data)
        # got = QuantileClipper(lower_percentile=.05, upper_percentile=.95).fit(
        #     X=df["col"]).transform(df["col"])
        got = data_processing.QuantileClipper(
            lower_percentile=0.05, upper_percentile=0.95, interpolation="midpoint"
        ).fit_transform(df["col"])
        pd.testing.assert_series_equal(expected, got, check_names=False)


class TestFindBoxPlotBoundaries:
    def test_1(self):
        df = pd.DataFrame(data={"a": ["a", "b", "c", "d"], "val": [1, 2, 3, 4]})
        got = data_processing.find_boxplot_boundaries(df["val"])
        # Q1 = 1.75, Q3 = 3.25, IQR = 1.5
        expected = -0.5, 5.5
        assert_eq(expected, got)

    def test_2(self):
        df = pd.DataFrame(
            data={"val": [-10, 1, 2, 3, 4, 10], "val2": [-10, 1, 2, 3, 4, 10]}
        )
        got = data_processing.BoxplotOutlierClipper().fit_transform(df)
        expected = pd.DataFrame(
            data={"val": [-2.5, 1, 2, 3, 4, 7.5], "val2": [-2.5, 1, 2, 3, 4, 7.5]}
        )
        pd.testing.assert_frame_equal(expected, got)


class TestCrossColumns:
    def test_1(self):
        df = pd.DataFrame(data={"a": ["x", "y", "z"], "b": [1, 2, 3]})
        expected = pd.Series(data=["x_X_1", "y_X_2", "z_X_3"])
        got = data_processing.cross_columns(df, cols=["a", "b"])
        pd.testing.assert_series_equal(expected, got, check_names=False)
