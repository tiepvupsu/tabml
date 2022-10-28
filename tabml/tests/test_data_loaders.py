from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabml.config_helpers import parse_pipeline_config
from tabml.data_loaders import BaseDataLoader
from tabml.feature_manager import BaseFeatureManager
from tabml.utils.utils import write_str_to_file


class TestBaseDataLoader:
    @pytest.fixture(autouse=True)
    def setup_class(self, tmp_path):
        # feature manager config
        dataset_dir = tmp_path / "dataset"
        fm_yaml_str = f"""
        raw_data_dir: "{dataset_dir}"
        dataset_name: "dummy"
        base_features:
          - name: "a"
            dtype: INT32
        transforming_features:
          - name: "b"
            index: 1
            dtype: INT32
          - name: "label"
            index: 2
            dtype: INT32
          - name: "is_train"
            index: 3
            dtype: BOOL
          - name: "is_validation"
            index: 4
            dtype: BOOL
        """
        fm_pb_path = tmp_path / "feature_config.yaml"
        write_str_to_file(fm_yaml_str, fm_pb_path)

        # create fake data
        df = pd.DataFrame(
            data={
                "a": [1, 2, 3, 4, 5],
                "b": [6, 7, 8, 9, 0],
                "c": [-1, -1, -1, -1, -1],
                "label": [0, 1, 1, 0, 1],
                "is_train": [True, False, True, True, False],
                "is_validation": [False, True, False, False, True],
            }
        )
        dataset_path = BaseFeatureManager(fm_pb_path).get_dataset_path()
        Path(dataset_path).parent.mkdir(parents=True)
        df.to_csv(dataset_path, index=False)

        # pipeline config
        pipeline_config = f"""
        config_name: "dummy"
        data_loader:
            cls_name: "tabml.data_loaders.BaseDataLoader"
            feature_config_path: "{fm_pb_path}"
            label_col: "label"
            features_to_model: ["a", "b"]
            train_filters: ["is_train"]
            validation_filters: ["is_validation"]
        model_wrapper:
            cls_name: "a"
        model_analysis:
            metrics: ["foo"]
            by_features: ["bar"]
            by_label: "bar"
            training_size: 50
        """

        pipeline_config_path = tmp_path / "pipeline_config.yaml"
        write_str_to_file(pipeline_config, pipeline_config_path)
        self.config = parse_pipeline_config(pipeline_config_path)

    def test_get_train_data_and_label(self):
        data_loader = BaseDataLoader(self.config.data_loader)
        got_data, got_label = data_loader.get_train_data_and_label()
        expected_data = pd.DataFrame(data={"a": [1, 3, 4], "b": [6, 8, 9]})
        expected_label = pd.Series(data={"label": [0, 1, 0]})
        # since index of expected_ and got_ are different, it's better to compare
        # their values.
        np.array_equal(expected_data.values, got_data.values)
        np.array_equal(expected_label.values, got_label.values)

    def test_get_validation_data_and_label(self):
        data_loader = BaseDataLoader(self.config.data_loader)
        got_data, got_label = data_loader.get_val_data_and_label()
        expected_data = pd.DataFrame(data={"a": [2, 5], "b": [7, 0]})
        expected_label = pd.Series(data={"label": [1, 1]})
        # since index of expected_ and got_ are different, it's better to compare
        # their values.
        np.array_equal(expected_data.values, got_data.values)
        np.array_equal(expected_label.values, got_label.values)
