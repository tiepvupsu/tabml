import pandas as pd
import pytest

from tabml.feature_manager import BaseFeatureManager, BaseTransformingFeature
from tabml.utils.utils import write_str_to_file


class _DummyFeatureManager(BaseFeatureManager):
    """A dummy FeatureManager class to check compute_all_transforming_features."""

    def initialize_dataframe(self):
        dummy_data = {"a": [1, 2]}
        self.dataframe = pd.DataFrame(data=dummy_data, dtype="int32")
        self.dataframe["d1"] = pd.to_datetime(["2021-03-20", "2021-02-11"])

    def _get_base_transforming_class(self):
        return _DummyBaseTransformingFeature

    def load_raw_data(self):
        # required to initiate an instance of BaseFeatureManager
        raise NotImplementedError


class _DummyFeatureManager2(BaseFeatureManager):
    """A dummy FeatureManager class to check to update and clean logics."""

    def _compute_feature(self, feature_name: str):
        print(feature_name)

    def initialize_dataframe(self):
        dummy_data = {"a": [1]}
        self.dataframe = pd.DataFrame(data=dummy_data)

    def load_raw_data(self):
        # required to initiate an instance of BaseFeatureManager
        raise NotImplementedError


class _DummyBaseTransformingFeature(BaseTransformingFeature):
    pass


class FeatureB(_DummyBaseTransformingFeature):
    name = "b"

    def transform(self, df):
        return df["a"]


class FeatureC(_DummyBaseTransformingFeature):
    name = "c"

    def transform(self, df):
        return df["b"] * 2


class FeatureD(_DummyBaseTransformingFeature):
    name = "d"

    def transform(self, df):
        return df["a"] + 1


class FeatureE(_DummyBaseTransformingFeature):
    name = "e"

    def transform(self, df):
        return df["c"] - 1


class FeatureD2(_DummyBaseTransformingFeature):
    name = "d2"

    def transform(self, df):
        return pd.DatetimeIndex(df["d1"]) + pd.DateOffset(1)


class TestBaseFeatureManager:
    @pytest.fixture(autouse=True)
    def setup_class(cls, tmp_path):
        pb_str = """
            raw_data_dir: "dummy"
            dataset_name: "dummy"
            base_features {
              name: "a"
              dtype: INT32
            }
            base_features {
              name: "d1"
              dtype: DATETIME
            }
            transforming_features {
              index: 1
              name: "b"
              dependencies: "a"
              dtype: INT32
            }
            transforming_features {
              index: 2
              name: "c"
              dependencies: "b"
              dtype: INT32
            }
            transforming_features {
              index: 3
              name: "d"
              dependencies: "a"
              dtype: INT32
            }
            transforming_features {
              index: 4
              name: "e"
              dependencies: "c"
              dtype: INT32
            }
            transforming_features {
              index: 5
              name: "d2"
              dependencies: "d1"
              dtype: DATETIME
            }
        """
        pb_config_path = tmp_path / "tmp.pb"
        write_str_to_file(pb_str, pb_config_path)
        cls.fm = _DummyFeatureManager(pb_config_path)
        cls.fm2 = _DummyFeatureManager2(pb_config_path)
        cls.fm.initialize_dataframe()
        cls.fm2.initialize_dataframe()

    def test_compute_all_transforming_features(self, tmp_path):
        self.fm.compute_all_transforming_features()
        expected_dataframe = pd.DataFrame(
            data={
                "a": [1, 2],
                "d1": ["2021-03-20", "2021-02-11"],
                "b": [1, 2],
                "c": [2, 4],
                "d": [2, 3],
                "e": [1, 3],
                "d2": ["2021-03-21", "2021-02-12"],
            }
        ).astype(
            {
                "a": "int32",
                "b": "int32",
                "c": "int32",
                "d": "int32",
                "e": "int32",
                "d1": "datetime64[ns]",
                "d2": "datetime64[ns]",
            }
        )
        pd.testing.assert_frame_equal(expected_dataframe, self.fm.dataframe)

        # test save and load dataframe
        # make a temp dataset_path
        self.fm.dataset_path = tmp_path / "features" / "dataset.csv"
        self.fm.save_dataframe()
        # make sure dataframe is cleared
        self.fm.dataframe = None
        self.fm.load_dataframe()
        pd.testing.assert_frame_equal(expected_dataframe, self.fm.dataframe)

    def test_update_feature(self, capsys):
        self.fm2.update_feature("b")
        # check that "b", "c", "e" are computed.
        captured = capsys.readouterr()
        assert captured.out == "b\nc\ne\n"

    def test_extract_dataframe(self):
        self.fm.compute_all_transforming_features()
        features_to_select = ["a", "c", "d2"]
        filters = ["d2 > '2021-03-01'"]
        expected_dataframe = pd.DataFrame(
            data={"a": [1], "c": [2], "d2": ["2021-03-21"]}
        ).astype({"a": "int32", "c": "int32", "d2": "datetime64[ns]"})
        pd.testing.assert_frame_equal(
            expected_dataframe, self.fm.extract_dataframe(features_to_select, filters)
        )

        # test without filters
        expected_dataframe = pd.DataFrame(
            data={"a": [1, 2], "c": [2, 4], "d2": ["2021-03-21", "2021-02-12"]}
        ).astype({"a": "int32", "c": "int32", "d2": "datetime64[ns]"})
        pd.testing.assert_frame_equal(
            expected_dataframe, self.fm.extract_dataframe(features_to_select)
        )
