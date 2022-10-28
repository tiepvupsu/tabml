from pathlib import Path

import pytest

from tabml import experiment_manager
from tabml.config_helpers import parse_pipeline_config


class TestExperimentManager:
    @pytest.fixture(autouse=True)
    def setup_class(self, tmp_path):
        yaml_str = """
            config_name: "foo"
            data_loader:
              cls_name: "bar"
              feature_config_path: "rab"
              features_to_model: ["c"]
              label_col: "d"
              train_filters: []
              validation_filters: []
            model_wrapper:
              cls_name: "bobar"
            model_analysis:
              metrics: ["a"]
              by_features: ["b"]
        """
        self.config = parse_pipeline_config(yaml_str=yaml_str)

    def test_not_create_new_run_dir(self, tmp_path):
        exp_root_dir = Path(tmp_path) / "exp"
        exp_manager = experiment_manager.ExperimentManager(
            self.config, exp_root_dir=exp_root_dir
        )
        exp_manager.create_new_run_dir()
        experiment_manager.ExperimentManager(
            self.config, should_create_new_run_dir=False, exp_root_dir=exp_root_dir
        )

    def test_not_creat_run_dir_not_exist(self, tmp_path):
        exp_root_dir = Path(tmp_path) / "exp2"
        Path(exp_root_dir).mkdir(parents=True)

        with pytest.raises(IOError) as excinfo:
            experiment_manager.ExperimentManager(
                self.config, should_create_new_run_dir=False, exp_root_dir=exp_root_dir
            )

        assert str(excinfo.value).startswith(
            "Could not find any run directory starting with"
        )
