import pandas as pd
import pytest

from tabml.feature_manager import BaseFeatureManager, BaseTransformingFeature
from tabml.utils.utils import write_str_to_file


class _DummyFeatureManager(BaseFeatureManager):
    """A dummy FeatureManager class to check to update and clean logics."""

    def _compute_feature(self, feature_name: str):
        print(feature_name)

    def initialize_dataframe(self):
        dummy_data = {"a": [1]}
        self.dataframe = pd.DataFrame(data=dummy_data)

    def load_dataframe(self):
        dummy_data = {"a": [1], "b": [1], "c": [1], "d": [1], "e": [1]}
        self.dataframe = pd.DataFrame(data=dummy_data)

    def _get_base_transforming_class(self):
        return _DummyBaseTransformingFeature


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


class TestBaseFeatureEngineering:
    @pytest.fixture(autouse=True)
    def setup_class(cls, tmp_path):
        pb_str = """
            raw_data_dir: "dummy"
            dataset_name: "dummy"
            base_features {
              name: "a"
              dtype: STRING
            }
            transforming_features {
              index: 1
              name: "b"
              dependencies: "a"
            }
            transforming_features {
              index: 2
              name: "c"
              dependencies: "b"
            }
            transforming_features {
              index: 3
              name: "d"
              dependencies: "a"
            }
            transforming_features {
              index: 4
              name: "e"
              dependencies: "c"
            }
        """
        pb_config_path = tmp_path / "tmp.pb"
        write_str_to_file(pb_str, pb_config_path)
        cls.fm = _DummyFeatureManager(pb_config_path)
        cls.fm.initialize_dataframe()

    def test_compute_all_transforming_features(self):
        self.fm.compute_all_transforming_features()

    def test_update_feature(self, capsys):
        self.fm.update_feature("b")
        # check that "b", "c", "e" are computed.
        captured = capsys.readouterr()
        assert captured.out == "b\nc\ne\n"
