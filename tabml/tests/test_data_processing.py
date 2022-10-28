import pandas as pd
import pytest

from tabml import data_processing
from tabml.utils.utils import write_str_to_file


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
        assert expected == got

    def test_2(self):
        df = pd.DataFrame(
            data={"val": [-10, 1, 2, 3, 4, 10], "val2": [-10, 1, 2, 3, 4, 10]}
        )
        got = pd.DataFrame(
            data=data_processing.BoxplotOutlierClipper().fit_transform(df),
            columns=["val", "val2"],
        )
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


class TestCategoryEncoder:
    @pytest.fixture(autouse=True)
    def setup_class(self, tmp_path):
        self.series = pd.Series(["a", "b", "c", "d"])

    def test_1(self):
        vocab_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        got = data_processing.CategoryEncoder(vocab_map).get_encoded(self.series)
        expected = pd.Series([0, 1, 2, 3])
        pd.testing.assert_series_equal(expected, got)

    def test_unknown(self):
        vocab_map = {"a": 0, "b": 1, "c": 5}
        got = data_processing.CategoryEncoder(vocab_map).get_encoded(self.series)
        expected = pd.Series([0, 1, 5, 6])
        pd.testing.assert_series_equal(expected, got)

    def test_from_list(self):
        vocab = ["a", "c", "b"]
        got = data_processing.CategoryEncoder.from_list(vocab).get_encoded(self.series)
        expected = pd.Series([0, 2, 1, 3])
        pd.testing.assert_series_equal(expected, got)

    def test_from_lines_in_txt(self, tmp_path):
        file_content = """a
            d
            c
        """
        txt_path = tmp_path / "foo.txt"
        write_str_to_file(file_content, txt_path)
        got = data_processing.CategoryEncoder.from_lines_in_txt(txt_path).get_encoded(
            self.series
        )
        expected = pd.Series([0, 3, 2, 1])
        pd.testing.assert_series_equal(expected, got)

    def test_from_mapping_in_csv(self, tmp_path):
        file_content = """
            a,0
            d, 2
            c,5
        """
        txt_path = tmp_path / "bar.txt"
        write_str_to_file(file_content, txt_path)
        got = data_processing.CategoryEncoder.from_mapping_in_csv(txt_path).get_encoded(
            self.series
        )
        expected = pd.Series([0, 6, 5, 2])
        pd.testing.assert_series_equal(expected, got)
